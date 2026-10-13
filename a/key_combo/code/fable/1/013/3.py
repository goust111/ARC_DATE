from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np


_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import Color, find_components, get_bg, paste, randomly_spaced_indices


BG = Color.BLACK
FLOOR = Color.GREY
PALETTE = [Color.BLUE, Color.RED, Color.GREEN, Color.YELLOW, Color.PINK, Color.ORANGE, Color.TEAL, Color.MAROON]

GRID_HEIGHT_RANGE = (15, 23)
GRID_WIDTH_RANGE = (12, 21)
BODY_HEIGHT_RANGE = (2, 4)
BODY_WIDTH_RANGE = (2, 5)
BAR_COUNT_RANGE = (3, 6)
RETRY_LIMIT = 400

_GEN_CALL_COUNT = 0

Segment = tuple[int, int, int, int]  # row, c0, c1, color


def _is_solid_rect(comp: dict) -> bool:
    r0, c0, r1, c1 = map(int, comp["bbox"])
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    return int(comp["area"]) == h * w


def _row_runs(row: np.ndarray, color: int) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start: int | None = None
    target = int(color)
    for c, value in enumerate(row):
        if int(value) == target:
            if start is None:
                start = c
        elif start is not None:
            runs.append((start, c - 1))
            start = None
    if start is not None:
        runs.append((start, len(row) - 1))
    return runs


def _cut_same_color_runs(board: np.ndarray, row: int, c0: int, c1: int, color: int) -> list[Segment]:
    segments: list[Segment] = []
    for run_c0, run_c1 in _row_runs(board[int(row)], int(color)):
        if run_c1 < c0 or run_c0 > c1:
            continue
        overlap_c0 = max(run_c0, c0)
        overlap_c1 = min(run_c1, c1)
        board[int(row), overlap_c0 : overlap_c1 + 1] = BG
        if run_c0 < overlap_c0:
            segments.append((int(row), int(run_c0), int(overlap_c0 - 1), int(color)))
        if overlap_c1 < run_c1:
            segments.append((int(row), int(overlap_c1 + 1), int(run_c1), int(color)))
    return segments


def _extract_scene(grid: np.ndarray) -> tuple[dict, list[dict], np.ndarray]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and max(g.shape) <= 30
    assert np.all(g[:-1] >= 0) and np.all(g[-1] >= 0)

    bg = int(get_bg(g))
    floor_color = int(g[-1, 0])
    assert bg == BG, "background must be black"
    assert floor_color == FLOOR, "floor must be grey"
    assert np.all(g[-1] == floor_color), "bottom row must be a complete floor"

    work = g.copy()
    work[-1, :] = bg
    comps = find_components(work, bg=bg, connectivity=4, monochromatic=True, return_format="dict")
    assert comps, "expected a falling block and at least one bar"

    body_candidates = []
    bars = []
    for comp in comps:
        r0, c0, r1, c1 = map(int, comp["bbox"])
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        if h >= 2:
            assert _is_solid_rect(comp), "the falling object must be a solid rectangle"
            body_candidates.append(
                {"r0": r0, "c0": c0, "h": h, "w": w, "color": int(comp["color"]), "bbox": comp["bbox"]}
            )
        else:
            assert h == 1 and int(comp["area"]) == w, "bars must be one-cell-thick horizontal segments"
            bars.append({"row": r0, "c0": c0, "c1": c1, "color": int(comp["color"])})

    assert len(body_candidates) == 1, "expected exactly one multi-row falling rectangle"
    body = body_candidates[0]
    assert body["r0"] == min(int(comp["bbox"][0]) for comp in comps), "the falling block must be the top object"
    assert bars, "expected suspended bars"
    assert any(bar["color"] == body["color"] for bar in bars), "need at least one same-color bar"
    return body, bars, g.copy()


def _simulate_body(board: np.ndarray, body: dict) -> tuple[int, list[Segment]]:
    h = int(body["h"])
    w = int(body["w"])
    c0 = int(body["c0"])
    color = int(body["color"])
    top = int(body["r0"])
    H = board.shape[0]

    free_segments: list[Segment] = []
    while True:
        below_row = top + h
        assert below_row < H, "the complete floor must stop the block before it leaves the grid"

        below = board[below_row, c0 : c0 + w]
        occupied = below != BG
        if not np.any(occupied):
            top += 1
            continue

        blockers = below[occupied]
        if np.any(blockers != color):
            break

        free_segments.extend(_cut_same_color_runs(board, below_row, c0, c0 + w - 1, color))
        top += 1

    return int(top), free_segments


def _drop_segment(board: np.ndarray, segment: Segment) -> None:
    row, c0, c1, color = map(int, segment)
    H = board.shape[0]
    board[row, c0 : c1 + 1] = BG
    while row + 1 < H and np.all(board[row + 1, c0 : c1 + 1] == BG):
        row += 1
    board[row, c0 : c1 + 1] = color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Drop the top rectangle through same-colored bars, cut those bars into free segments, and settle the whole collapse.

    Input structure:
    - The background is black and the bottom row is one complete grey floor.
    - Above the floor are several non-touching one-cell-thick monochrome horizontal bars.
    - Exactly one solid monochrome rectangle sits at the top and acts as the falling block; at least one bar shares its color.

    Rule:
    - Let the top rectangle fall straight down one row at a time.
    - If the next row below it is empty, it keeps falling; if every occupied cell directly below is on a same-colored bar, it passes through and erases the overlapping bar cells, leaving the left and right leftovers as free segments; if any occupied cell below has a different color or belongs to the floor, the rectangle stops.
    - After the rectangle stops, each free segment falls straight down inertly, without cutting anything, and lower cut rows settle before higher cut rows.
    - Keep all colors unchanged and preserve the complete grey floor.

    Key insight:
    - Same color means “non-blocking” only for the original falling rectangle: each penetration removes a window from a bar, and the answer is the final resting state after those detached bar pieces fall in bottom-up order.
    """
    body, _bars, board = _extract_scene(np.asarray(grid, dtype=int))

    r0 = int(body["r0"])
    c0 = int(body["c0"])
    h = int(body["h"])
    w = int(body["w"])
    color = int(body["color"])

    board[r0 : r0 + h, c0 : c0 + w] = BG
    final_top, free_segments = _simulate_body(board, body)

    body_patch = np.full((h, w), color, dtype=int)
    paste(board, body_patch, final_top, c0)

    for segment in sorted(free_segments, key=lambda item: (-int(item[0]), int(item[1]), int(item[2]))):
        _drop_segment(board, segment)

    assert np.all(board[-1] == FLOOR), "floor must remain intact"
    return board


def _scenario_config(call_id: int) -> tuple[str, int, bool]:
    mode = int(call_id) % 4
    if mode == 0:
        return "floor", random.choice([1, 2]), False
    if mode == 1:
        return "barrier", 1, False
    if mode == 2:
        return "barrier", 2, True
    return random.choice(["floor", "barrier"]), random.choice([2, 3]), True


def _sample_rows(start_row: int, last_row: int, n_bars: int) -> list[int] | None:
    span = int(last_row - start_row + 1)
    if span <= 0 or n_bars <= 0:
        return None
    if n_bars > (span + 1) // 2:
        return None
    try:
        picked = randomly_spaced_indices(span, n_bars, border_size=0, padding=1)
    except ValueError:
        return None
    return sorted(int(start_row + idx) for idx in picked)


def _sample_cut_interval(
    rng: np.random.Generator,
    W: int,
    body_left: int,
    body_right: int,
    preferred: str,
) -> tuple[int, int] | None:
    modes = []
    if body_left >= 1 and body_right <= W - 2:
        modes.append("both")
    if body_left >= 1:
        modes.append("left")
    if body_right <= W - 2:
        modes.append("right")
    if not modes:
        return None

    if preferred in modes and rng.random() < 0.75:
        mode = preferred
    else:
        weights = np.array([4 if item == "both" else 2 for item in modes], dtype=float)
        mode = modes[int(rng.choice(len(modes), p=weights / weights.sum()))]

    if mode == "both":
        c0 = int(rng.integers(0, body_left))
        c1 = int(rng.integers(body_right + 1, W))
    elif mode == "left":
        c0 = int(rng.integers(0, body_left))
        c1 = int(rng.integers(body_left, body_right + 1))
    else:
        c0 = int(rng.integers(body_left, body_right + 1))
        c1 = int(rng.integers(body_right + 1, W))
    return (c0, c1) if c0 <= c1 else None


def _sample_barrier_interval(
    rng: np.random.Generator,
    W: int,
    body_left: int,
    body_right: int,
) -> tuple[int, int]:
    min_c0 = max(0, body_left - int(rng.integers(0, 4)))
    max_c0 = min(body_left, body_right)
    min_c1 = max(body_left, body_right)
    max_c1 = min(W - 1, body_right + int(rng.integers(0, 4)))
    c0 = int(rng.integers(min_c0, max_c0 + 1))
    c1 = int(rng.integers(min_c1, max_c1 + 1))
    if c0 > c1:
        c0, c1 = min(body_left, body_right), max(body_left, body_right)
    return c0, c1


def _sample_distractor_interval(
    rng: np.random.Generator,
    W: int,
    body_left: int,
    body_right: int,
) -> tuple[int, int] | None:
    choices: list[tuple[int, int]] = []
    left_space = body_left
    right_space = W - body_right - 1

    if left_space >= 2:
        length = int(rng.integers(2, min(7, left_space) + 1))
        c1 = int(rng.integers(length - 1, body_left))
        choices.append((c1 - length + 1, c1))
    if right_space >= 2:
        length = int(rng.integers(2, min(7, right_space) + 1))
        c0 = int(rng.integers(body_right + 1, W - length + 1))
        choices.append((c0, c0 + length - 1))

    if not choices:
        return None
    return choices[int(rng.integers(0, len(choices)))]


def _paint_scene(
    H: int,
    W: int,
    body_r0: int,
    body_c0: int,
    body_h: int,
    body_w: int,
    body_color: int,
    bars: list[tuple[int, int, int, int]],
) -> np.ndarray:
    grid = np.full((H, W), BG, dtype=int)
    grid[-1, :] = FLOOR
    for row, c0, c1, color in bars:
        grid[int(row), int(c0) : int(c1) + 1] = int(color)
    body_patch = np.full((body_h, body_w), int(body_color), dtype=int)
    paste(grid, body_patch, int(body_r0), int(body_c0))
    return grid


def _gen_attempt(call_id: int) -> np.ndarray | None:
    rng = np.random.default_rng()
    stop_mode, requested_cuts, aligned_cuts = _scenario_config(call_id)

    H = int(rng.integers(GRID_HEIGHT_RANGE[0], GRID_HEIGHT_RANGE[1] + 1))
    W = int(rng.integers(GRID_WIDTH_RANGE[0], GRID_WIDTH_RANGE[1] + 1))
    body_h = int(rng.integers(BODY_HEIGHT_RANGE[0], BODY_HEIGHT_RANGE[1] + 1))
    body_w = int(rng.integers(BODY_WIDTH_RANGE[0], BODY_WIDTH_RANGE[1] + 1))
    body_r0 = int(rng.integers(0, 2))
    if W - body_w - 3 <= 1:
        return None
    body_c0 = int(rng.integers(2, W - body_w - 1))
    body_right = body_c0 + body_w - 1

    start_row = body_r0 + body_h + 2
    last_row = H - 3
    span = last_row - start_row + 1
    if span < 3:
        return None

    body_color = int(rng.choice(PALETTE))
    other_colors = [color for color in PALETTE if color != body_color]
    barrier_count = 1 if stop_mode == "barrier" else 0
    max_bars = min(BAR_COUNT_RANGE[1], (span + 1) // 2)
    if max_bars < BAR_COUNT_RANGE[0]:
        return None

    min_bars = max(BAR_COUNT_RANGE[0], requested_cuts + barrier_count)
    if max_bars >= requested_cuts + barrier_count + 1:
        min_bars = max(min_bars, requested_cuts + barrier_count + 1)
    if min_bars > max_bars:
        return None

    n_bars = int(rng.integers(min_bars, max_bars + 1))
    rows = _sample_rows(start_row, last_row, n_bars)
    if rows is None:
        return None

    if stop_mode == "barrier":
        stop_pos = int(rng.integers(requested_cuts, n_bars))
        cut_positions = sorted(int(v) for v in rng.choice(stop_pos, size=requested_cuts, replace=False))
    else:
        stop_pos = None
        cut_positions = sorted(int(v) for v in rng.choice(n_bars, size=requested_cuts, replace=False))

    preferred_cut = random.choice(["left", "right", "both"]) if aligned_cuts else "both"
    bars: list[tuple[int, int, int, int]] = []
    for idx, row in enumerate(rows):
        if idx in cut_positions:
            interval = _sample_cut_interval(rng, W, body_c0, body_right, preferred_cut)
            if interval is None:
                return None
            c0, c1 = interval
            bars.append((row, c0, c1, body_color))
            continue

        if stop_pos is not None and idx == stop_pos:
            c0, c1 = _sample_barrier_interval(rng, W, body_c0, body_right)
            bars.append((row, c0, c1, int(rng.choice(other_colors))))
            continue

        interval = _sample_distractor_interval(rng, W, body_c0, body_right)
        if interval is None:
            return None
        c0, c1 = interval
        color_pool = PALETTE if rng.random() < 0.35 else other_colors
        bars.append((row, c0, c1, int(rng.choice(color_pool))))

    inp = _paint_scene(H, W, body_r0, body_c0, body_h, body_w, body_color, bars)
    try:
        out = main(inp)
    except Exception:
        return None

    if np.array_equal(inp, out):
        return None
    if not np.all(inp[-1] == FLOOR) or not np.all(out[-1] == FLOOR):
        return None
    return inp


def generate_input() -> np.ndarray:
    global _GEN_CALL_COUNT

    call_id = _GEN_CALL_COUNT
    _GEN_CALL_COUNT += 1
    for _ in range(RETRY_LIMIT):
        grid = _gen_attempt(call_id)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid collapse scene")
