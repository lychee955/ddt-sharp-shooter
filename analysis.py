"""Snapshot analysis and a single bounded worker, independent of the UI."""

from dataclasses import dataclass, replace
from queue import Empty, Queue
import threading
import time

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

from force import solve_force
from vision import PlayerMarker, recognize_players
from turn import InputState, InputChangeGate


REFERENCE_WIDTH = 1500
# Some game layouts extend the map 60 reference pixels farther left.
# Include its whole viewport and both edges, not only the rightmost dots.
MINIMAP_REGION = (1170, 35, 330, 150)
WIND_REGION = (709, 22, 84, 54)
DEGREE_REGION = (43, 835, 64, 36)


@dataclass(frozen=True)
class TargetEstimate:
    player: PlayerMarker
    direction: str = "—"
    dx: float | None = None
    dy: float | None = None
    force: float | None = None
    status: str = ""


@dataclass(frozen=True)
class AnalysisResult:
    targets: tuple[TargetEstimate, ...] = ()
    minimap: np.ndarray | None = None
    wind: float | None = None
    wind_direction: str = ""
    degree: int | None = None
    timestamp: float = 0
    error: str = ""
    tracking_note: str = ""
    wind_image: np.ndarray | None = None
    degree_image: np.ndarray | None = None
    uses_cached_self: bool = False
    error_kind: str = ""
    stale_age: float | None = None
    phase: str = ""
    attempts: int = 1
    input_state: InputState | None = None
    frame: np.ndarray | None = None
    manual_self: bool = False


def crop_region(frame, reference):
    ratio = frame.shape[1] / REFERENCE_WIDTH
    x, y, width, height = (int(value * ratio) for value in reference)
    if x < 0 or y < 0 or width <= 0 or height <= 0 or x + width > frame.shape[1] or y + height > frame.shape[0]:
        raise ValueError("游戏区域尺寸或比例不正确，请拖动图标重新绑定完整游戏画面")
    return frame[y:y + height, x:x + width].copy()


def estimate_targets(players, degree, wind, wind_to_left, scale_pixels):
    own = next((p for p in players if p.identity == "自己"), None)
    if own is None or scale_pixels <= 0:
        raise ValueError("自己位置或小地图标尺无效")
    targets = []
    for player in players:
        if player.identity == "自己":
            targets.append(TargetEstimate(player, status="自己"))
            continue
        horizontal_pixels = abs(player.x - own.x)
        dx = horizontal_pixels / scale_pixels * 10
        dy = (own.y - player.y) / scale_pixels * 10
        left = player.x < own.x
        direction = "垂直" if horizontal_pixels < 1 else "左" if left else "右"
        if horizontal_pixels < 1:
            targets.append(TargetEstimate(player, direction, 0.0, dy,
                                          status="目标近乎垂直，请调整位置"))
            continue
        signed_wind = wind if left == wind_to_left else -wind
        try:
            solution = solve_force(degree, signed_wind, dx, dy)
            status = "当前角度力度不足" if solution.force > 100 else "可用"
            targets.append(TargetEstimate(player, direction, dx, dy, solution.force, status))
        except ValueError as exc:
            targets.append(TargetEstimate(player, direction, dx, dy, status=str(exc)))
    return tuple(targets)


def analyze_frame(frame, wind_reader, degree_reader, scale_reader, *, own_hint=None, detection=None):
    timestamp = time.time()
    minimap, players = None, ()
    wind, degree, wind_direction = None, None, ""
    tracking_note = ""
    wind_image, degree_image = None, None
    uses_cached_self, error_kind, manual_self = False, "context", False
    try:
        wind_image = crop_region(frame, WIND_REGION)
        degree_image = crop_region(frame, DEGREE_REGION)
        minimap = crop_region(frame, MINIMAP_REGION)
        detection = detection or recognize_players(minimap, frame.shape[1] / REFERENCE_WIDTH, own_hint)
        players = detection.players
        manual_self = detection.manual_self
        if detection.error:
            error_kind = detection.error_kind
            raise ValueError(detection.error)
        error_kind = "parameters"
        uses_cached_self = detection.used_hint
        if detection.used_hint:
            tracking_note = "蓝圈暂不可见：使用上次缓存位置，敌我颜色按本帧判断"
        if manual_self:
            tracking_note = "已手动选择自己，按显示的截图计算；t 刷新，取消勾选恢复蓝圈识别"
        scale_pixels = scale_reader(minimap.copy())
        if scale_pixels <= 0:
            raise ValueError("小地图标尺识别失败，请检查小地图是否被遮挡")
        wind, wind_to_left = wind_reader(wind_image.copy())
        if not np.isfinite(wind) or not 0 <= wind <= 99.9:
            wind = None
            raise ValueError("风力识别结果无效")
        wind_direction = "无风" if wind == 0 else "向左" if wind_to_left else "向右"
        text = degree_reader(degree_image.copy()).strip()
        if not text.isdigit() or not 0 <= int(text) <= 180:
            raise ValueError(f"角度识别失败（结果：{text!r}）；检查右侧截取预览，拖动图标重新绑定游戏画面")
        degree = int(text)
        targets = estimate_targets(players, degree, wind, wind_to_left, scale_pixels)
        own = next(p for p in players if p.identity == "自己")
        ratio = frame.shape[1]/REFERENCE_WIDTH
        return AnalysisResult(targets, minimap, wind, wind_direction, degree, timestamp,
                              tracking_note=tracking_note, wind_image=wind_image, degree_image=degree_image,
                              uses_cached_self=uses_cached_self, input_state=InputState(degree,own.x/ratio,own.y/ratio),
                              frame=frame, manual_self=manual_self)
    except Exception as exc:
        reason = f"{type(exc).__name__}: {exc}"
        targets = tuple(TargetEstimate(p, status="自己" if p.identity == "自己" else "本轮计算失败") for p in players)
        return AnalysisResult(targets, minimap, wind, wind_direction, degree, timestamp, reason,
                              tracking_note, wind_image, degree_image, uses_cached_self, error_kind,
                              frame=frame, manual_self=manual_self)


class SnapshotAnalyzer:
    """Reuse only the last halo-confirmed location, without prediction."""

    def __init__(self, wind_reader, degree_reader, scale_reader, *, clock=time.monotonic, max_age=2.0):
        self.readers = wind_reader, degree_reader, scale_reader
        self.clock, self.max_age = clock, max_age
        self._lock = threading.Lock()
        self._generation = 0
        self._confirmed = None
        self._missing_since = None
        self._manual = None

    def reset(self):
        with self._lock:
            self._generation += 1
            self._confirmed = None
            self._missing_since = None
            self._manual = None

    def analyze_selection(self, snapshot, own_index):
        """Recompute the displayed frame, so a clicked row never selects a different dot."""
        if snapshot.frame is None:
            raise ValueError("当前结果没有完整截图，请按 t 刷新后再选择自己")
        if own_index is None:
            self.reset()
        else:
            if not 0 <= own_index < len(snapshot.targets):
                raise ValueError("人物列表已更新，请重新选择")
            player = snapshot.targets[own_index].player
            ratio = snapshot.frame.shape[1]/REFERENCE_WIDTH
            minimap = crop_region(snapshot.frame, MINIMAP_REGION)
            with self._lock:
                self._generation += 1
                self._confirmed = self._missing_since = None
                self._manual = (player.x/ratio, player.y/ratio, minimap.copy())
        return replace(self(snapshot.frame), timestamp=snapshot.timestamp)

    def _observe_self(self, frame):
        now = self.clock()
        with self._lock:
            generation, previous, missing_since = self._generation, self._confirmed, self._missing_since
            manual = self._manual
        hint = None
        context_changed = False
        if previous is not None and (missing_since is None or now-missing_since <= self.max_age):
            own, previous_map = previous
            current_map = crop_region(frame, MINIMAP_REGION)
            if current_map.shape == previous_map.shape:
                # A moved viewport border or blinking dots affect few pixels;
                # a new map or changed crop must not inherit an old position.
                difference = np.max(np.abs(current_map.astype(float)-previous_map), axis=2)
                if np.percentile(difference, 90) < 15:
                    hint = own.x, own.y
                else:
                    context_changed = True
            else:
                context_changed = True
        minimap = crop_region(frame, MINIMAP_REGION)
        ratio = frame.shape[1]/REFERENCE_WIDTH
        manual_hint = None
        if manual is not None:
            mx, my, previous_map = manual
            # Resize the background for window scaling; never follow a row number
            # or guess a new position after map changes.
            current_map = np.asarray(Image.fromarray(minimap).resize(
                (previous_map.shape[1], previous_map.shape[0]), Image.Resampling.BILINEAR))
            current_background = gaussian_filter(current_map.astype(float), sigma=(1, 1, 0))
            previous_background = gaussian_filter(previous_map.astype(float), sigma=(1, 1, 0))
            difference = np.max(np.abs(current_background-previous_background), axis=2)
            if np.percentile(difference, 90) < 15:
                manual_hint = mx*ratio, my*ratio
            else:
                context_changed = True
                with self._lock:
                    if generation == self._generation:
                        self._manual = None
                hint = None
        detection = recognize_players(minimap,ratio,hint,manual_hint=manual_hint)
        own = next((p for p in detection.players if p.identity == "自己"),None)
        with self._lock:
            if generation == self._generation:
                if own is not None:
                    if not detection.used_hint and not detection.manual_self:
                        self._confirmed = (own, minimap.copy())
                    self._missing_since = None
                elif (context_changed or detection.error_kind in {"context", "self_ambiguous"}
                      or (missing_since is not None and now-missing_since > self.max_age)):
                    self._confirmed = None
                    self._missing_since = None
                elif previous is not None and missing_since is None:
                    self._missing_since = now
        return detection, context_changed

    def __call__(self, frame):
        detection, context_changed = self._observe_self(frame)
        result = analyze_frame(frame,*self.readers,detection=detection)
        return replace(result,error_kind="context") if context_changed and result.error else result

    def probe(self, frame):
        """Read angle and own position only; never read wind or solve force."""
        try:
            detection, _ = self._observe_self(frame)
            if detection.error:
                return None
            text = self.readers[1](crop_region(frame,DEGREE_REGION)).strip()
            if not text.isdigit() or not 0 <= int(text) <= 180:
                return None
            own = next(p for p in detection.players if p.identity == "自己")
            ratio = frame.shape[1]/REFERENCE_WIDTH
            return InputState(int(text),own.x/ratio,own.y/ratio)
        except (ValueError,TypeError):
            return None


class AnalysisWorker:
    """At most one capture/analysis in flight and one result waiting for the UI."""

    auto_request_on_configure = True

    def __init__(self, capture, analyze, interval=1.0):
        self.capture, self.analyze, self.interval = capture, analyze, interval
        self.results = Queue(maxsize=1)
        self._condition = threading.Condition()
        self._region = (0, 0, 0, 0)
        self._running = False
        self._requested = False
        self._selection_request = None
        self._closed = False
        self._generation = 0
        self._deadline = 0.0
        self._thread = threading.Thread(target=self._run, daemon=True)

    @property
    def running(self):
        with self._condition:
            return self._running

    def start(self):
        self._thread.start()

    def configure(self, region, running=False):
        with self._condition:
            self._generation += 1
            self._selection_request = None
            if callable(getattr(self.analyze, "reset", None)):
                self.analyze.reset()
            self._region = tuple(region)
            self._running = running and region[2] > 0 and region[3] > 0
            self._requested = self._running and self.auto_request_on_configure
            self._reset_cycle()
            self._deadline = 0
            self._condition.notify_all()

    def pause(self):
        with self._condition:
            self._generation += 1
            self._selection_request = None
            if callable(getattr(self.analyze, "reset", None)):
                self.analyze.reset()
            self._running = False
            self._requested = False
            self._reset_cycle()
            self._condition.notify_all()

    def _reset_cycle(self):
        pass

    def refresh(self):
        with self._condition:
            self._requested = True
            self._condition.notify_all()

    def select_self(self, snapshot, own_index):
        if snapshot.frame is None or not callable(getattr(self.analyze, "analyze_selection", None)):
            raise ValueError("请先按 t 获取人物截图，再勾选自己")
        with self._condition:
            if self._closed:
                return
            self._generation += 1
            self._selection_request = (snapshot, own_index)
            self._requested = True
            self._condition.notify_all()

    def close(self):
        with self._condition:
            self._closed = True
            self._generation += 1
            self._condition.notify_all()

    def take_result(self):
        try:
            generation, result = self.results.get_nowait()
        except Empty:
            return None
        with self._condition:
            return result if generation == self._generation else None

    def _run(self):
        while True:
            with self._condition:
                while not self._closed:
                    delay = max(0, self._deadline - time.monotonic())
                    if self._requested or (self._running and delay == 0):
                        break
                    self._condition.wait(delay if self._running else None)
                if self._closed:
                    return
                generation, region = self._generation, self._region
                selection, self._selection_request = self._selection_request, None
                self._requested = False
                started = time.monotonic()
            try:
                if region[2] <= 0 or region[3] <= 0:
                    raise ValueError("请先拖动瞄准图标绑定游戏窗口")
                result = (self.analyze.analyze_selection(*selection) if selection is not None
                          else self.analyze(self.capture(region)))
            except Exception as exc:
                result = AnalysisResult(timestamp=time.time(), error=f"{type(exc).__name__}: {exc}", error_kind="context")
            with self._condition:
                self._deadline = started + self.interval
                if generation != self._generation or self._closed:
                    if callable(getattr(self.analyze, "reset", None)):
                        self.analyze.reset()
                    continue
                try:
                    self.results.get_nowait()
                except Empty:
                    pass
                self.results.put_nowait((generation, result))


class TurnAnalysisWorker(AnalysisWorker):
    """Watch a cheap announcement; perform one bounded analysis per turn."""

    auto_request_on_configure = False

    def __init__(self, capture, analyze, turn_detector, *, interval=.5, retries=5, retry_delay=.35,
                 input_reader=None, active_detector=None):
        from turn import TurnLatch
        self.turn_detector = turn_detector
        self.latch = TurnLatch()
        self.retries, self.retry_delay = retries, retry_delay
        self._busy = False
        self.input_reader, self.active_detector = input_reader, active_detector
        self.change_gate = InputChangeGate()
        self._turn_open = False
        self._inactive_since = None
        super().__init__(capture, analyze, interval)

    def _reset_cycle(self):
        self.latch.reset()
        self._busy = False
        self.change_gate.reset()
        self._turn_open = False
        self._inactive_since = None

    def refresh(self):
        with self._condition:
            if not self._busy and not self._closed:
                self._requested = True
                self._condition.notify_all()

    def _publish(self, generation, result):
        with self._condition:
            if self._closed or generation != self._generation:
                return False
            try:
                self.results.get_nowait()
            except Empty:
                pass
            self.results.put_nowait((generation, result))
            return True

    def _batch(self, generation, region, first_frame):
        self._publish(generation, AnalysisResult(timestamp=time.time(), phase="computing"))
        frame = first_frame
        for attempt in range(1, self.retries+1):
            with self._condition:
                if self._closed or generation != self._generation:
                    return
            result = self.analyze(frame)
            if not result.error:
                with self._condition:
                    if self._closed or generation != self._generation:
                        return
                    self.change_gate.reset(result.input_state)
                self._publish(generation, replace(result, phase="complete", attempts=attempt))
                return
            if result.error_kind != "self_missing" or attempt == self.retries:
                self._publish(generation, replace(result, phase="failed", attempts=attempt))
                return
            self._publish(generation, replace(result, phase="computing", attempts=attempt,
                          error=f"蓝圈暂不可见，准备重试 {attempt+1}/{self.retries}"))
            with self._condition:
                interrupted = self._condition.wait_for(
                    lambda: self._closed or generation != self._generation, self.retry_delay)
                if interrupted:
                    return
            frame = self.capture(region)

    def _run(self):
        while True:
            with self._condition:
                while not self._closed:
                    delay = max(0, self._deadline-time.monotonic())
                    if self._requested or (self._running and delay == 0):
                        break
                    self._condition.wait(delay if self._running else None)
                if self._closed:
                    return
                generation, region = self._generation, self._region
                selection, self._selection_request = self._selection_request, None
                manual, monitoring = self._requested, self._running
                self._requested = False
            try:
                if selection is not None:
                    result = self.analyze.analyze_selection(*selection)
                    with self._condition:
                        if self._closed or generation != self._generation:
                            continue
                        self.change_gate.reset(result.input_state if not result.error else None)
                    self._publish(generation, replace(result, phase="failed" if result.error else "complete"))
                    continue
                if region[2] <= 0 or region[3] <= 0:
                    raise ValueError("请先拖动瞄准图标绑定游戏窗口")
                frame = self.capture(region)
                visible = self.turn_detector(frame) if monitoring else False
                active = self.active_detector(frame) if monitoring and self.active_detector is not None else visible
                with self._condition:
                    if self._closed or generation != self._generation:
                        continue
                    now = time.monotonic()
                    due = monitoring and self.latch.observe(visible,now)
                    if due or (monitoring and manual and active):
                        self._turn_open = True
                        self.change_gate.reset()
                    if monitoring and self._turn_open and not (visible or active):
                        if self._inactive_since is None:
                            self._inactive_since = now
                        elif now-self._inactive_since >= 1:
                            self._turn_open = False
                            self.change_gate.reset()
                    else:
                        self._inactive_since = None
                    due = due or manual or self._requested
                    self._requested = False
                    check_inputs = (not due and monitoring and self._turn_open and active
                                    and self.input_reader is not None and self.change_gate.baseline is not None)
                state = self.input_reader(frame) if check_inputs else None
                with self._condition:
                    if self._closed or generation != self._generation:
                        continue
                    due = due or (check_inputs and self.change_gate.observe(state))
                    due = due or self._requested
                    self._requested = False
                    self._busy = bool(due)
                if due:
                    self._batch(generation, region, frame)
            except Exception as exc:
                self._publish(generation, AnalysisResult(timestamp=time.time(), phase="failed",
                              error=f"{type(exc).__name__}: {exc}", error_kind="context"))
            finally:
                with self._condition:
                    self._busy = False
                    self._deadline = time.monotonic()+self.interval
                    if generation != self._generation and callable(getattr(self.analyze,"reset",None)):
                        self.analyze.reset()
