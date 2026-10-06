"""Colour-independent minimap detection; no screen or keyboard access."""

from dataclasses import dataclass, replace

import numpy as np
from PIL import Image
from skimage import color, measure, morphology
from scipy.ndimage import maximum_filter, minimum_filter


@dataclass(frozen=True)
class PlayerMarker:
    x: float
    y: float
    rgb: tuple[int, int, int]
    identity: str = "身份不确定"
    label: str = "?"


@dataclass(frozen=True)
class Detection:
    players: tuple[PlayerMarker, ...]
    error: str = ""
    used_hint: bool = False
    error_kind: str = ""
    manual_self: bool = False


def _map_visibility(rgb):
    """Exclude the uniform strip below the game's clipped minimap viewport."""
    height, width = rgb.shape[:2]
    bottom = height
    pixels = rgb.astype(float)
    for y in range(max(3, height // 2), height-3):
        fill = np.median(pixels[y], axis=0)
        flat = np.mean(np.linalg.norm(pixels[y:y+3]-fill, axis=2) < 8)
        edges = np.linalg.norm(pixels[y-2:y+1]-pixels[y-3:y], axis=2)
        strongest = int(np.argmax(np.mean(edges, axis=1)))
        if flat >= .85 and np.mean(edges[strongest] > 20) >= .65:
            # Repeated resizing can blend two rows across the original edge.
            bottom = y-3+strongest
            break
    return np.indices((height, width))[0] < bottom, bottom


def _ring_sectors(ring, sectors, visible):
    # At a clipped boundary require evidence across the visible half-ring;
    # tiny sector fragments do not count as independent directions.
    return [i for i in range(12)
            if np.count_nonzero(ring & visible & (sectors == i)) >= 4]


def _halo_score(blue, x, y, visible=None):
    yy, xx = np.indices(blue.shape)
    radius = np.hypot(xx - x, yy - y)
    annulus = (radius >= 10) & (radius <= 15)
    if not np.any(annulus):
        return 0.0
    angles = (np.arctan2(yy - y, xx - x) + np.pi) * 12 / (2 * np.pi)
    sectors = np.minimum(angles.astype(int), 11)
    visible = np.ones(blue.shape, dtype=bool) if visible is None else visible
    available = _ring_sectors(annulus, sectors, visible)
    covered = sum(np.any(annulus & visible & blue & (sectors == i)) for i in available)
    coverage = np.mean(blue[annulus & visible]) if np.any(annulus & visible) else 0
    return float(coverage) if len(available) >= 6 and covered >= max(5, .75*len(available)) and coverage >= .18 else 0.0


def _faded_halo_score(lab, x, y, visible=None):
    """Transparent blue rings can look grey over yellow terrain.

    Compare the blue/yellow channel with the adjacent background in each
    direction, requiring a near-complete ring rather than one blue patch.
    """
    yy, xx = np.indices(lab.shape[:2])
    distance = np.hypot(xx-x, yy-y)
    sectors = np.minimum(((np.arctan2(yy-y, xx-x)+np.pi)*12/(2*np.pi)).astype(int), 11)
    visible = np.ones(lab.shape[:2], dtype=bool) if visible is None else visible
    for radius in range(9, 16):
        ring = (distance >= radius-1.5) & (distance <= radius+1.5)
        outside = (distance >= radius+3) & (distance <= radius+5)
        available = _ring_sectors(ring, sectors, visible)
        covered = 0
        for sector in available:
            inner_values = lab[:, :, 2][ring & visible & (sectors == sector)]
            outer_values = lab[:, :, 2][outside & visible & (sectors == sector)]
            if inner_values.size and outer_values.size:
                ring_blue = np.median(inner_values)
                if ring_blue < 22 and np.median(outer_values)-ring_blue >= 12:
                    covered += 1
        if len(available) >= 6 and covered >= max(5, .75*len(available)):
            return covered/len(available)
    return 0.0


def _clipped_dot_center(rgb, prop, bottom):
    """Fit the visible circular arc, never use a half-dot's shifted centroid."""
    y0, x0, y1, x1 = prop.bbox
    body_rgb = np.median(rgb[tuple(prop.coords.T)], axis=0)
    patch = rgb[max(0, y0-2):bottom, max(0, x0-2):x1+2]
    close = np.linalg.norm(patch.astype(float)-body_rgb, axis=2) < 40
    contours = measure.find_contours(np.pad(close, 1), .5)
    if not contours:
        return None
    arc = max(contours, key=len)-1
    arc[:, 0] += max(0, y0-2)
    arc[:, 1] += max(0, x0-2)
    arc = arc[arc[:, 0] < bottom-1]
    if len(arc) < 10 or np.ptp(arc[:, 1]) < 6:
        return None
    y, x = arc.T
    mx, my = np.mean(x), np.mean(y)
    fit = np.linalg.lstsq(np.column_stack((x-mx, y-my, np.ones(len(x)))),
                         -((x-mx)**2+(y-my)**2), rcond=None)[0]
    cx, cy = mx-fit[0]/2, my-fit[1]/2
    radius_squared = (fit[0]**2+fit[1]**2)/4-fit[2]
    if not 4**2 <= radius_squared <= 8.5**2:
        return None
    radius = np.sqrt(radius_squared)
    residual = np.sqrt(np.mean((np.hypot(x-cx, y-cy)-radius)**2))
    if residual > .55 or not x0 <= cx < x1 or not bottom-7 <= cy <= bottom+1:
        return None
    return float(cx), float(cy)


def recognize_players(image: np.ndarray, pixel_scale: float = 1.0, own_hint=None, *, manual_hint=None) -> Detection:
    """Classify afresh each frame; return original-image pixel coordinates."""
    if pixel_scale <= 0 or image.ndim != 3 or image.shape[2] < 3:
        return Detection((), "小地图图像无效", error_kind="context")
    height, width = image.shape[:2]
    normalized = np.asarray(Image.fromarray(image[:, :, :3].astype(np.uint8)).resize(
        (max(1, round(width / pixel_scale)), max(1, round(height / pixel_scale))),
        Image.Resampling.BILINEAR,
    ))
    hsv = color.rgb2hsv(normalized)
    hue, saturation, value = (hsv[:, :, i] for i in range(3))
    blue = (hue >= .58) & (hue <= .70) & (saturation > .8) & (value > .65)
    lab = color.rgb2lab(normalized)
    rgb_float = normalized.astype(float)
    visible, bottom = _map_visibility(normalized)
    candidates, masks = [], []
    # Overlapping hue bands find coloured dots of any hue; neutral bands
    # support white/grey dots too. Team colours are never predefined.
    for center in np.arange(0, 1, 1 / 30):
        difference = np.abs((hue - center + .5) % 1 - .5)
        masks.append((difference < .035) & (saturation > .35) & (value > .25))
    for center in np.arange(.2, 1.01, .1):
        masks.append((np.abs(value - center) < .06) & (saturation <= .35))
    # Same-hue terrain can merge with a dot. Seed additional masks from flat
    # colour cores, then use RGB distance rather than hue alone to separate it.
    spread = maximum_filter(normalized, size=(3, 3, 1)).astype(float)-minimum_filter(normalized, size=(3, 3, 1))
    flat = (np.max(spread, axis=2) <= 3) & (value > .25) & (saturation > .6)
    for prop in measure.regionprops(measure.label(flat)):
        if 8 <= prop.area <= 200:
            rgb = np.median(normalized[tuple(prop.coords.T)], axis=0)
            masks.append(np.linalg.norm(rgb_float-rgb, axis=2) < 20)
    yy, xx = np.indices(hue.shape)
    for mask in masks:
        mask = mask & visible
        mask = morphology.binary_opening(mask, morphology.disk(1))
        for prop in measure.regionprops(measure.label(mask)):
            y0, x0, y1, x1 = prop.bbox
            h, w = y1 - y0, x1 - x0
            clipped = y1 >= bottom-1
            if not (7 <= w <= 24 and (4 if clipped else 7) <= h <= 18
                    and (20 if clipped else 45) <= prop.area <= 240):
                continue
            if prop.extent < .45 or prop.solidity < .80 or (not clipped and prop.axis_minor_length / max(prop.axis_major_length, 1) < .5):
                continue
            cy, cx = prop.centroid
            if clipped:
                center = _clipped_dot_center(normalized, prop, bottom)
                if center is None:
                    continue
                cx, cy = center
            if min(cx, cy, normalized.shape[1]-cx) < 8 or (not clipped and normalized.shape[0]-cy < 8):
                continue
            radius = np.hypot(xx - cx, yy - cy)
            core = (radius <= 3) & visible
            ring = (radius >= 8) & (radius <= 10) & visible
            if np.count_nonzero(core) < 6 or not np.any(ring):
                continue
            # Player dots have a flat centre; illustrated terrain has shading.
            if np.max(np.std(rgb_float[core], axis=0)) > 4:
                continue
            body_lab = np.median(lab[core], axis=0)
            background = np.median(lab[ring], axis=0)
            if np.linalg.norm(body_lab - background) < 12:
                continue
            body_rgb = tuple(int(v) for v in np.median(normalized[core], axis=0))
            inner = (radius <= 4.5) & visible
            close = np.linalg.norm(rgb_float - body_rgb, axis=2) < 40
            if np.mean(close[inner]) < .85:
                continue
            outer = (radius >= 7.5) & (radius <= 9.5) & visible
            if np.mean(close[outer]) > .2:
                continue
            body = close & (radius <= 7) & visible
            body_y, body_x = np.where(body)
            if not clipped:
                cx, cy = float(np.mean(body_x)), float(np.mean(body_y))
            quality = float(np.count_nonzero(body))
            candidates.append((quality, cx, cy, body_rgb, body_lab))
    unique = []
    for candidate in sorted(candidates, key=lambda item: item[:3], reverse=True):
        _, cx, cy, _, _ = candidate
        if all(np.hypot(cx - other[1], cy - other[2]) > 6 for other in unique):
            unique.append(candidate)
    raw = tuple(PlayerMarker(cx * width / normalized.shape[1],
                             cy * height / normalized.shape[0], rgb)
                for _, cx, cy, rgb, _ in unique)
    if manual_hint is not None:
        selves = [i for i, player in enumerate(raw)
                  if np.hypot(player.x-manual_hint[0], player.y-manual_hint[1]) <= 4*pixel_scale]
        if len(selves) != 1:
            return Detection(_number_players(raw), "手动选择的自己已移动、消失或重叠，请在“我”列重新勾选",
                             error_kind="self_ambiguous")
    else:
        selves = [i for i, (_, cx, cy, _, _) in enumerate(unique)
                  if _halo_score(blue, cx, cy, visible) or _faded_halo_score(lab, cx, cy, visible)]
    used_hint = False
    if not selves and own_hint is not None:
        matches = [i for i, p in enumerate(raw)
                   if np.hypot(p.x-own_hint[0], p.y-own_hint[1]) <= 4*pixel_scale]
        if len(matches) == 1:
            selves = matches
            used_hint = True
    if len(selves) != 1:
        reason = "未找到自己的蓝色光圈" if not selves else "发现多个蓝色光圈，自己位置不确定"
        return Detection(_number_players(raw), reason,
                         error_kind="self_missing" if not selves else "self_ambiguous")
    own_lab = unique[selves[0]][4]
    classified = []
    for i, player in enumerate(raw):
        distance = float(color.deltaE_ciede2000(unique[i][4], own_lab))
        identity = "自己" if i == selves[0] else (
            "队友" if distance <= 12 else "敌人" if distance >= 25 else "身份不确定"
        )
        if identity == "自己" and used_hint:
            # Keep the last halo-confirmed coordinates; never extrapolate or
            # move the cache towards nearby dots during an animation gap.
            player = replace(player, x=own_hint[0], y=own_hint[1])
        classified.append(replace(player, identity=identity))
    return Detection(_number_players(tuple(classified)), used_hint=used_hint, manual_self=manual_hint is not None)


def _number_players(players):
    order = {"自己": 0, "队友": 1, "敌人": 2, "身份不确定": 3}
    counts = {key: 0 for key in order}
    numbered = []
    for player in sorted(players, key=lambda p: (order[p.identity], round(p.x / 4), p.y)):
        counts[player.identity] += 1
        label = "我" if player.identity == "自己" else (
            {"队友": "友", "敌人": "敌", "身份不确定": "?"}[player.identity]
            + str(counts[player.identity])
        )
        numbered.append(replace(player, label=label))
    return tuple(numbered)
