"""Detect the own-turn announcement without OCR, player detection or firing."""

from pathlib import Path
from dataclasses import dataclass
import math

import numpy as np
from PIL import Image
from skimage import color
from skimage.feature import match_template


TURN_AREA = (300, 160, 1000, 430)
CONTROL_AREA = (550, 130, 450, 230)


@dataclass(frozen=True)
class InputState:
    degree: int
    x: float
    y: float


class InputChangeGate:
    """Recompute only after two stable observations of changed inputs."""

    def __init__(self):
        self.reset()

    def reset(self, baseline=None):
        self.baseline = baseline
        self.candidate = None
        self.last_requested = None

    @staticmethod
    def same(a, b, tolerance=1):
        return a.degree == b.degree and math.hypot(a.x-b.x, a.y-b.y) < tolerance

    def observe(self, state):
        if state is None or self.baseline is None:
            self.candidate = None
            return False
        if self.same(state, self.baseline, 2):
            self.candidate = self.last_requested = None
            return False
        if self.last_requested is not None and self.same(state, self.last_requested):
            return False
        if self.candidate is not None and self.same(state, self.candidate):
            self.last_requested = state
            self.candidate = None
            return True
        self.candidate = state
        return False


def _letter_mask(image):
    hsv = color.rgb2hsv(np.asarray(image))
    saturation, value = hsv[:, :, 1], hsv[:, :, 2]
    # White highlights match the letters rather than the gold terrain/glow.
    return ((value > .65) & (saturation < .2)).astype(float)


class OwnTurnDetector:
    def __init__(self, template=None, threshold=.65):
        path = Path(__file__).parent / "assets" / "own_turn_banner.png"
        template = Image.open(path).convert("RGB") if template is None else Image.fromarray(template)
        self.templates = []
        for scale in (.9, 1.0, 1.1):
            resized = template.resize((round(template.width*scale/2), round(template.height*scale/2)), Image.Resampling.BILINEAR)
            self.templates.append(_letter_mask(resized))
        self.threshold = threshold
        control = Image.open(path.with_name("own_turn_pass.png")).convert("RGB")
        self.control_templates = [self._control_mask(control.resize(
            (round(control.width*scale/2), round(control.height*scale/2)), Image.Resampling.BILINEAR))
            for scale in (.9, 1, 1.1)]

    @staticmethod
    def _control_mask(image):
        hsv = color.rgb2hsv(np.asarray(image))
        return ((hsv[:, :, 0] >= .065) & (hsv[:, :, 0] <= .2)
                & (hsv[:, :, 1] > .15) & (hsv[:, :, 2] > .55)).astype(float)

    def active(self, frame):
        if frame.ndim != 3 or frame.shape[2] < 3 or frame.shape[1] <= 0:
            return False
        ratio = frame.shape[1]/1500
        x, y, w, h = (round(v*ratio) for v in CONTROL_AREA)
        area = frame[y:y+h, x:x+w, :3]
        if area.shape[:2] != (h, w):
            return False
        mask = self._control_mask(Image.fromarray(area.astype(np.uint8)).resize(
            (225, 115), Image.Resampling.BILINEAR))
        return max(float(np.max(match_template(mask, t))) for t in self.control_templates) >= .65

    def score(self, frame):
        if frame.ndim != 3 or frame.shape[2] < 3 or frame.shape[1] <= 0:
            return 0.0
        ratio = frame.shape[1]/1500
        x,y,w,h = (round(v*ratio) for v in TURN_AREA)
        area = frame[y:y+h,x:x+w,:3]
        if area.shape[:2] != (h,w):
            return 0.0
        area = Image.fromarray(area.astype(np.uint8)).resize((500,215), Image.Resampling.BILINEAR)
        mask = _letter_mask(area)
        return max(float(np.max(match_template(mask, template))) for template in self.templates)

    def __call__(self, frame):
        return self.score(frame) >= self.threshold


class TurnLatch:
    """One event per announcement; brief detection gaps do not rearm it."""

    def __init__(self, release_after=1.5):
        self.release_after = release_after
        self.reset()

    def reset(self):
        self.seen = False
        self.missing_since = None

    def observe(self, visible, now):
        if visible:
            self.missing_since = None
            if not self.seen:
                self.seen = True
                return True
        else:
            if self.missing_since is None:
                self.missing_since = now
            if now-self.missing_since >= self.release_after:
                self.seen = False
        return False
