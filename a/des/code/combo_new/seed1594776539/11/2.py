import random

import numpy as np


BG = 0
SOLID = "solid"
CROSS = "cross"

DIRS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
OUTSIDE_OFFSETS = {
    "up": (-1, 1),
    "down": (3, 1),
    "left": (1, -1),
    "right": (1, 3),
}
RECEIVER_TOPLEFT_OFFSETS = {
    "up": (-2, -1),
    "down": (0, -1),
    "left": (-1, -2),
    "right": (-1, 0),
}
PLUS_POSITIONS = ((0, 1), (1, 0), (1, 2), (2, 1))
CORNER_POSITIONS = ((0, 0), (0, 2), (2, 0), (2, 2))
BORDER_POSITIONS = (
    (0, 0),
    (0, 1),
    (0, 2),
    (1, 0),
    (1, 2),
    (2, 0),
    (2, 1),
    (2, 2),
)

MAX_GENERATE_ATTEMPTS = 400
MAX_PAIR_PLACEMENT_ATTEMPTS = 120


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _receiver_color_at(grid: np.ndarray, top: int, left: int) -> int | None:
    h, w = grid.shape
    if top < 0 or left < 0 or top + 3 > h or left + 3 > w:
        return None
    window = grid[top : top + 3, left : left + 3]
    if int(window[1, 1]) != BG:
        return None
    color = int(window[0, 0])
    if color == BG:
        return None
    for dr, dc in BORDER_POSITIONS:
        if int(window[dr, dc]) != color:
            return None
    return color


def _transmitter_at(grid: np.ndarray, top: int, left: int) -> dict[str, object] | None:
    h, w = grid.shape
    if top < 0 or left < 0 or top + 3 > h or left + 3 > w:
        return None

    window = grid[top : top + 3, left : left + 3]
    signal = int(window[1, 1])
    if signal == BG:
        return None

    kind = None
    emitter = None

    solid_emitter = int(window[0, 0])
    if solid_emitter != BG and solid_emitter != signal:
        if all(
            int(window[dr, dc]) == solid_emitter
            for dr, dc in BORDER_POSITIONS
        ):
            kind = SOLID
            emitter = solid_emitter

    if kind is None:
        cross_emitter = int(window[0, 1])
        if cross_emitter != BG and cross_emitter != signal:
            if all(int(window[dr, dc]) == cross_emitter for dr, dc in PLUS_POSITIONS) and all(
                int(window[dr, dc]) == BG for dr, dc in CORNER_POSITIONS
            ):
                kind = CROSS
                emitter = cross_emitter

    if kind is None or emitter is None:
        return None

    line_dirs: list[str] = []
    for name, (dr, dc) in OUTSIDE_OFFSETS.items():
        rr = top + dr
        cc = left + dc
        if _in_bounds(grid.shape, rr, cc) and int(grid[rr, cc]) == signal:
            line_dirs.append(name)

    assert len(line_dirs) == 1, "each transmitter must connect to exactly one line"

    return {
        "top": top,
        "left": left,
        "kind": kind,
        "emitter": emitter,
        "signal": signal,
        "direction": line_dirs[0],
    }


def _detect_transmitters(grid: np.ndarray) -> list[dict[str, object]]:
    h, w = grid.shape
    transmitters = []
    for top in range(h - 2):
        for left in range(w - 2):
            transmitter = _transmitter_at(grid, top, left)
            if transmitter is not None:
                transmitters.append(transmitter)
    assert transmitters, "expected at least one transmitter"
    return transmitters


def _detect_receivers(grid: np.ndarray) -> dict[tuple[int, int], int]:
    h, w = grid.shape
    receivers: dict[tuple[int, int], int] = {}
    for top in range(h - 2):
        for left in range(w - 2):
            color = _receiver_color_at(grid, top, left)
            if color is not None:
                receivers[(top, left)] = color
    assert receivers, "expected at least one receiver"
    return receivers


def _trace_to_receiver(grid: np.ndarray, transmitter: dict[str, object]) -> tuple[int, int, int]:
    top = int(transmitter["top"])
    left = int(transmitter["left"])
    signal = int(transmitter["signal"])
    direction = str(transmitter["direction"])

    dr, dc = DIRS[direction]
    start_dr, start_dc = OUTSIDE_OFFSETS[direction]
    r = top + start_dr
    c = left + start_dc

    steps = 0
    while _in_bounds(grid.shape, r, c) and int(grid[r, c]) == signal:
        r += dr
        c += dc
        steps += 1

    assert steps >= 1, "signal line must contain at least one pixel"
    assert _in_bounds(grid.shape, r, c), "signal line must end at a receiver"

    top_offset, left_offset = RECEIVER_TOPLEFT_OFFSETS[direction]
    receiver_top = r + top_offset
    receiver_left = c + left_offset
    receiver_color = _receiver_color_at(grid, receiver_top, receiver_left)
    assert receiver_color is not None, "line must terminate at a hollow 3x3 receiver"
    assert receiver_color != signal, "receiver border color must differ from signal color"
    return receiver_top, receiver_left, receiver_color


def _paint_receiver(out: np.ndarray, top: int, left: int, kind: str, signal: int) -> None:
    if kind == SOLID:
        out[top + 1, left + 1] = signal
        return

    assert kind == CROSS
    for dr, dc in CORNER_POSITIONS:
        out[top + dr, left + dc] = signal


def main(grid: np.ndarray) -> np.ndarray:
    """
    Transform each receiver according to the shape of the transmitter connected to it by a straight signal line.

    Input structure:
    - The background is black.
    - Several 3×3 transmitters and 3×3 hollow-square receivers are scattered on the grid.
    - Every transmitter has a distinct signal-color center and connects to exactly one receiver by a one-pixel-wide horizontal or vertical straight line of that same signal color.
    - A transmitter is either a solid 3×3 block with only its center replaced by the signal color, or a cross whose corners are black and whose center is the signal color.

    Rule:
    - Trace each transmitter's signal-colored line until it reaches the matching hollow 3×3 receiver.
    - If the transmitter is solid, fill the receiver's center with the transmitter's signal color while keeping the receiver border unchanged.
    - If the transmitter is a cross, replace only the receiver's four corner pixels with the transmitter's signal color; keep the edge-midpoints and center unchanged.
    - Leave every transmitter and every signal line unchanged.

    Key insight:
    - The transmitter's internal shape, rather than its color, determines how the line's destination receiver is rewritten.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    transmitters = _detect_transmitters(g)
    receivers = _detect_receivers(g)
    assert len(transmitters) == len(receivers), "expected one receiver per transmitter"

    out = g.copy()
    matched_receivers: set[tuple[int, int]] = set()

    for transmitter in transmitters:
        receiver_top, receiver_left, _receiver_color = _trace_to_receiver(g, transmitter)
        receiver_pos = (receiver_top, receiver_left)
        assert receiver_pos not in matched_receivers, "each receiver must be matched exactly once"
        matched_receivers.add(receiver_pos)
        _paint_receiver(
            out,
            receiver_top,
            receiver_left,
            str(transmitter["kind"]),
            int(transmitter["signal"]),
        )

    assert matched_receivers == set(receivers), "every receiver must be connected to a transmitter"
    return out


def _transmitter_patch(kind: str, emitter: int, signal: int) -> np.ndarray:
    patch = np.zeros((3, 3), dtype=int)
    if kind == SOLID:
        patch[:, :] = emitter
    else:
        for dr, dc in PLUS_POSITIONS:
            patch[dr, dc] = emitter
    patch[1, 1] = signal
    return patch


def _receiver_patch(receiver: int) -> np.ndarray:
    patch = np.full((3, 3), receiver, dtype=int)
    patch[1, 1] = BG
    return patch


def _expand_bbox(top: int, left: int, bottom: int, right: int, shape: tuple[int, int]) -> tuple[int, int, int, int]:
    return (
        max(0, top - 1),
        max(0, left - 1),
        min(shape[0] - 1, bottom + 1),
        min(shape[1] - 1, right + 1),
    )


def _build_pair_layout(
    shape: tuple[int, int],
    axis: str,
    sign: int,
    line_len: int,
    top: int,
    left: int,
) -> dict[str, object]:
    if axis == "h":
        assert sign in (-1, 1)
        span = 6 + line_len
        assert left + span <= shape[1]
        if sign == 1:
            transmitter_top_left = (top, left)
            receiver_top_left = (top, left + 3 + line_len)
            line_cells = [(top + 1, left + 3 + k) for k in range(line_len)]
        else:
            receiver_top_left = (top, left)
            transmitter_top_left = (top, left + 3 + line_len)
            line_cells = [(top + 1, left + 3 + k) for k in range(line_len)]
        bbox_top, bbox_left = top, left
        bbox_bottom, bbox_right = top + 2, left + span - 1
    else:
        assert axis == "v"
        assert sign in (-1, 1)
        span = 6 + line_len
        assert top + span <= shape[0]
        if sign == 1:
            transmitter_top_left = (top, left)
            receiver_top_left = (top + 3 + line_len, left)
            line_cells = [(top + 3 + k, left + 1) for k in range(line_len)]
        else:
            receiver_top_left = (top, left)
            transmitter_top_left = (top + 3 + line_len, left)
            line_cells = [(top + 3 + k, left + 1) for k in range(line_len)]
        bbox_top, bbox_left = top, left
        bbox_bottom, bbox_right = top + span - 1, left + 2

    pad_top, pad_left, pad_bottom, pad_right = _expand_bbox(
        bbox_top, bbox_left, bbox_bottom, bbox_right, shape
    )

    return {
        "axis": axis,
        "sign": sign,
        "line_len": line_len,
        "transmitter_top_left": transmitter_top_left,
        "receiver_top_left": receiver_top_left,
        "line_cells": line_cells,
        "bbox": (bbox_top, bbox_left, bbox_bottom, bbox_right),
        "padded_bbox": (pad_top, pad_left, pad_bottom, pad_right),
    }


def _bbox_is_free(blocked: np.ndarray, bbox: tuple[int, int, int, int]) -> bool:
    top, left, bottom, right = bbox
    return not bool(blocked[top : bottom + 1, left : right + 1].any())


def _mark_bbox(blocked: np.ndarray, bbox: tuple[int, int, int, int]) -> None:
    top, left, bottom, right = bbox
    blocked[top : bottom + 1, left : right + 1] = True


def _sample_pair_kinds(pair_count: int) -> list[str]:
    if pair_count == 2:
        kinds = [SOLID, CROSS]
    else:
        if random.random() < 0.5:
            kinds = [SOLID, SOLID, CROSS]
        else:
            kinds = [SOLID, CROSS, CROSS]
    random.shuffle(kinds)
    return kinds


def _place_pair(blocked: np.ndarray) -> dict[str, object] | None:
    h, w = blocked.shape
    for _ in range(MAX_PAIR_PLACEMENT_ATTEMPTS):
        axis = random.choice(("h", "v"))
        sign = random.choice((-1, 1))

        max_line = (w - 8) if axis == "h" else (h - 8)
        if max_line < 2:
            continue
        line_len = random.randint(2, min(max_line, 10))

        if axis == "h":
            span = 6 + line_len
            if h < 5 or w < span + 2:
                continue
            top = random.randint(1, h - 4)
            left = random.randint(1, w - span - 1)
        else:
            span = 6 + line_len
            if w < 5 or h < span + 2:
                continue
            top = random.randint(1, h - span - 1)
            left = random.randint(1, w - 4)

        layout = _build_pair_layout(blocked.shape, axis, sign, line_len, top, left)
        if not _bbox_is_free(blocked, layout["padded_bbox"]):
            continue
        return layout
    return None


def _apply_expected_output(grid: np.ndarray, pairs: list[dict[str, object]]) -> np.ndarray:
    out = grid.copy()
    for pair in pairs:
        receiver_top, receiver_left = pair["receiver_top_left"]
        _paint_receiver(out, receiver_top, receiver_left, pair["kind"], pair["signal"])
    return out


def _gen_attempt() -> np.ndarray | None:
    pair_count = random.choice((2, 3))
    min_side = 14 if pair_count == 2 else 18
    h = random.randint(min_side, 30)
    w = random.randint(min_side, 30)

    grid = np.zeros((h, w), dtype=int)
    blocked = np.zeros((h, w), dtype=bool)
    colors = random.sample(list(range(1, 10)), 3 * pair_count)
    kinds = _sample_pair_kinds(pair_count)
    pairs: list[dict[str, object]] = []

    for idx in range(pair_count):
        layout = _place_pair(blocked)
        if layout is None:
            return None

        emitter, signal, receiver = colors[3 * idx : 3 * idx + 3]
        kind = kinds[idx]
        transmitter_top, transmitter_left = layout["transmitter_top_left"]
        receiver_top, receiver_left = layout["receiver_top_left"]

        grid[transmitter_top : transmitter_top + 3, transmitter_left : transmitter_left + 3] = _transmitter_patch(
            kind, emitter, signal
        )
        for r, c in layout["line_cells"]:
            grid[r, c] = signal
        grid[receiver_top : receiver_top + 3, receiver_left : receiver_left + 3] = _receiver_patch(receiver)

        _mark_bbox(blocked, layout["padded_bbox"])
        pairs.append(
            {
                "kind": kind,
                "signal": signal,
                "transmitter_top_left": layout["transmitter_top_left"],
                "receiver_top_left": layout["receiver_top_left"],
            }
        )

    expected = _apply_expected_output(grid, pairs)
    actual = main(grid)
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATE_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid transmitter-receiver input")
