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


def _halo_score(blue, x, y):
    yy, xx = np.indices(blue.shape)
    radius = np.hypot(xx - x, yy - y)
    annulus = (radius >= 10) & (radius <= 15)
    if not np.any(annulus):
        return 0.0
    angles = (np.arctan2(yy - y, xx - x) + np.pi) * 12 / (2 * np.pi)
    sectors = np.minimum(angles.astype(int), 11)
    covered = sum(np.any(annulus & blue & (sectors == i)) for i in range(12))
    coverage = np.mean(blue[annulus])
    return float(coverage) if covered >= 9 and coverage >= .18 else 0.0


def _faded_halo_score(lab, x, y):
    """Transparent blue rings can look grey over yellow terrain.

    Compare the blue/yellow channel with the adjacent background in each
    direction, requiring a near-complete ring rather than one blue patch.
    """
    yy, xx = np.indices(lab.shape[:2])
    distance = np.hypot(xx-x, yy-y)
    sectors = np.minimum(((np.arctan2(yy-y, xx-x)+np.pi)*12/(2*np.pi)).astype(int), 11)
    for radius in range(9, 16):
        ring = (distance >= radius-1.5) & (distance <= radius+1.5)
        outside = (distance >= radius+3) & (distance <= radius+5)
        covered = 0
        for sector in range(12):
            inner_values = lab[:, :, 2][ring & (sectors == sector)]
            outer_values = lab[:, :, 2][outside & (sectors == sector)]
            if inner_values.size and outer_values.size:
                ring_blue = np.median(inner_values)
                if ring_blue < 22 and np.median(outer_values)-ring_blue >= 12:
                    covered += 1
        if covered >= 9:
            return covered/12
    return 0.0


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
        mask = morphology.binary_opening(mask, morphology.disk(1))
        for prop in measure.regionprops(measure.label(mask)):
            y0, x0, y1, x1 = prop.bbox
            h, w = y1 - y0, x1 - x0
            if not (7 <= w <= 24 and 7 <= h <= 18 and 45 <= prop.area <= 240):
                continue
            if prop.extent < .45 or prop.solidity < .80 or prop.axis_minor_length / max(prop.axis_major_length, 1) < .5:
                continue
            cy, cx = prop.centroid
            if min(cx, cy, width/pixel_scale-cx, height/pixel_scale-cy) < 8:
                continue
            radius = np.hypot(xx - cx, yy - cy)
            core = radius <= 3
            ring = (radius >= 8) & (radius <= 10)
            if not np.any(ring):
                continue
            # Player dots have a flat centre; illustrated terrain has shading.
            if np.max(np.std(rgb_float[core], axis=0)) > 4:
                continue
            body_lab = np.median(lab[core], axis=0)
            background = np.median(lab[ring], axis=0)
            if np.linalg.norm(body_lab - background) < 12:
                continue
            body_rgb = tuple(int(v) for v in np.median(normalized[core], axis=0))
            inner = radius <= 4.5
            close = np.linalg.norm(rgb_float - body_rgb, axis=2) < 40
            if np.mean(close[inner]) < .85:
                continue
            outer = (radius >= 7.5) & (radius <= 9.5)
            if np.mean(close[outer]) > .2:
                continue
            body = close & (radius <= 7)
            body_y, body_x = np.where(body)
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
                  if _halo_score(blue, cx, cy) or _faded_halo_score(lab, cx, cy)]
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
