from __future__ import annotations

import itertools
import random

import numpy as np


BLACK = 0
GREY = 5
MAROON = 9
OBJECT_COLORS = (1, 2, 3, 4, 6, 7, 8)

GRID_HEIGHT_RANGE = (13, 20)
GRID_WIDTH_RANGE = (13, 20)
TARGET_LINE_COUNT_RANGE = (2, 4)
TARGET_LINE_SPACING = 3
INTERVAL_COUNT_RANGE = (3, 5)
INTERVAL_WIDTH_RANGE = (1, 4)
INTERVAL_GAP = 1
EXTRA_RECT_COUNT_RANGE = (0, 2)
EXTRA_RECT_HEIGHT_RANGE = (1, 3)
EXTRA_RECT_WIDTH_RANGE = (1, 4)
MAX_GENERATION_ATTEMPTS = 400


def _arrow_rows_left(g: np.ndarray) -> list[int]:
    h, w = g.shape
    if h < 3 or w < 2:
        return []
    return [
        r
        for r in range(1, h - 1)
        if int(g[r, 0]) == GREY and int(g[r - 1, 1]) == GREY and int(g[r + 1, 1]) == GREY
    ]


def _arrow_rows_right(g: np.ndarray) -> list[int]:
    h, w = g.shape
    if h < 3 or w < 2:
        return []
    return [
        r
        for r in range(1, h - 1)
        if int(g[r, w - 1]) == GREY and int(g[r - 1, w - 2]) == GREY and int(g[r + 1, w - 2]) == GREY
    ]


def _arrow_cols_top(g: np.ndarray) -> list[int]:
    h, w = g.shape
    if h < 2 or w < 3:
        return []
    return [
        c
        for c in range(1, w - 1)
        if int(g[0, c]) == GREY and int(g[1, c - 1]) == GREY and int(g[1, c + 1]) == GREY
    ]


def _arrow_cols_bottom(g: np.ndarray) -> list[int]:
    h, w = g.shape
    if h < 2 or w < 3:
        return []
    return [
        c
        for c in range(1, w - 1)
        if int(g[h - 1, c]) == GREY and int(g[h - 2, c - 1]) == GREY and int(g[h - 2, c + 1]) == GREY
    ]


def _detect_arrow_edge(g: np.ndarray) -> tuple[str, list[int]]:
    candidates = {
        "left": _arrow_rows_left(g),
        "right": _arrow_rows_right(g),
        "top": _arrow_cols_top(g),
        "bottom": _arrow_cols_bottom(g),
    }
    nonempty = [(edge, lines) for edge, lines in candidates.items() if lines]
    assert len(nonempty) == 1, "expected arrows on exactly one edge"
    edge, lines = nonempty[0]
    return edge, sorted(set(map(int, lines)))


def _anchor_lines(g: np.ndarray, arrow_edge: str) -> set[int]:
    h, w = g.shape
    if arrow_edge == "left":
        return {int(r) for r in range(h) if int(g[r, w - 1]) == MAROON}
    if arrow_edge == "right":
        return {int(r) for r in range(h) if int(g[r, 0]) == MAROON}
    if arrow_edge == "top":
        return {int(c) for c in range(w) if int(g[h - 1, c]) == MAROON}
    if arrow_edge == "bottom":
        return {int(c) for c in range(w) if int(g[0, c]) == MAROON}
    raise ValueError(f"unknown edge: {arrow_edge}")


def _pack_vector(vec: np.ndarray, *, toward_start: bool) -> np.ndarray:
    out = np.full_like(vec, BLACK)
    fixed_mask = (vec == GREY) | (vec == MAROON)
    out[fixed_mask] = vec[fixed_mask]

    movable = [int(v) for v in vec if int(v) not in (BLACK, GREY, MAROON)]
    free_positions = [i for i in range(len(vec)) if not bool(fixed_mask[i])]
    if toward_start:
        targets = free_positions[: len(movable)]
    else:
        targets = free_positions[len(free_positions) - len(movable) :]
    for pos, value in zip(targets, movable):
        out[int(pos)] = int(value)
    return out


def _transform_rows(g: np.ndarray, selected_rows: list[int], anchored_rows: set[int], *, toward_left: bool) -> np.ndarray:
    out = g.copy()
    for r in selected_rows:
        if int(r) in anchored_rows:
            continue
        out[int(r), :] = _pack_vector(g[int(r), :], toward_start=toward_left)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Compress arrow-marked rows or columns toward the arrow edge while freezing anchored lines.

    Input structure:
    - Black background with several separated solid colored rectangles in the interior.
    - Exactly one border carries grey three-pixel arrow markers; each arrow selects one row or one column.
    - The opposite parallel border carries maroon single-pixel anchor marks that identify fixed rows or columns.

    Rule:
    - Determine whether the arrows select rows (left/right edge) or columns (top/bottom edge).
    - For every selected line that is not anchored, collect its non-black non-marker pixels in their original order and
      repack them tightly against the arrow side, leaving grey arrow cells and maroon anchor cells in place.
    - Any selected line whose index is marked by a maroon anchor stays unchanged, and every unmarked line also stays unchanged.

    Key insight:
    - The arrows define both the affected lines and the packing direction, while the opposite-edge maroon marks carve out
      exception lines that must remain fixed instead of being compacted.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    edge, selected_lines = _detect_arrow_edge(g)
    anchored = _anchor_lines(g, edge)

    if edge == "left":
        return _transform_rows(g, selected_lines, anchored, toward_left=True)
    if edge == "right":
        return _transform_rows(g, selected_lines, anchored, toward_left=False)
    if edge == "top":
        return _transform_rows(g.T, selected_lines, anchored, toward_left=True).T
    if edge == "bottom":
        return _transform_rows(g.T, selected_lines, anchored, toward_left=False).T
    raise AssertionError("unreachable")


def _spaced_combinations(values: list[int], count: int, spacing: int) -> list[tuple[int, ...]]:
    return [
        combo
        for combo in itertools.combinations(values, count)
        if all(b - a >= spacing for a, b in zip(combo, combo[1:]))
    ]


def _sample_selected_rows(h: int) -> list[int] | None:
    candidates = list(range(2, h - 2))
    feasible = [
        combos
        for count in range(TARGET_LINE_COUNT_RANGE[0], TARGET_LINE_COUNT_RANGE[1] + 1)
        if (combos := _spaced_combinations(candidates, count, TARGET_LINE_SPACING))
    ]
    if not feasible:
        return None
    combos = random.choice(feasible)
    return list(random.choice(combos))


def _random_split(total: int, parts: int) -> list[int]:
    values = [0] * parts
    for _ in range(total):
        values[random.randrange(parts)] += 1
    return values


def _sample_intervals(start: int, end: int, count: int, min_width: int, max_width: int, gap: int) -> list[tuple[int, int]] | None:
    usable = end - start + 1
    if usable <= 0:
        return None

    for _ in range(100):
        widths = [random.randint(min_width, max_width) for _ in range(count)]
        minimum = sum(widths) + gap * (count - 1)
        if minimum > usable:
            continue
        extras = _random_split(usable - minimum, count + 1)
        intervals: list[tuple[int, int]] = []
        cursor = start + extras[0]
        for i, width in enumerate(widths):
            c0 = cursor
            c1 = c0 + width - 1
            intervals.append((int(c0), int(c1)))
            cursor = c1 + 1 + gap + extras[i + 1]
        return intervals
    return None


def _draw_arrows_left(g: np.ndarray, rows: list[int]) -> None:
    for r in rows:
        g[int(r), 0] = GREY
        g[int(r) - 1, 1] = GREY
        g[int(r) + 1, 1] = GREY


def _draw_arrows_right(g: np.ndarray, rows: list[int]) -> None:
    w = g.shape[1]
    for r in rows:
        g[int(r), w - 1] = GREY
        g[int(r) - 1, w - 2] = GREY
        g[int(r) + 1, w - 2] = GREY


def _draw_anchors_left(g: np.ndarray, rows: set[int]) -> None:
    for r in rows:
        g[int(r), 0] = MAROON


def _draw_anchors_right(g: np.ndarray, rows: set[int]) -> None:
    w = g.shape[1]
    for r in rows:
        g[int(r), w - 1] = MAROON


def _can_place_rect(occupied: np.ndarray, r0: int, r1: int, c0: int, c1: int, *, pad: int = 1) -> bool:
    h, w = occupied.shape
    rr0 = max(0, r0 - pad)
    rr1 = min(h - 1, r1 + pad)
    cc0 = max(0, c0 - pad)
    cc1 = min(w - 1, c1 + pad)
    return not bool(occupied[rr0 : rr1 + 1, cc0 : cc1 + 1].any())


def _place_extra_rectangles(g: np.ndarray, band_top: int, band_bottom: int, occupied: np.ndarray) -> None:
    h, w = g.shape
    for _ in range(random.randint(*EXTRA_RECT_COUNT_RANGE)):
        placed = False
        for _ in range(60):
            rect_h = random.randint(*EXTRA_RECT_HEIGHT_RANGE)
            rect_w = random.randint(*EXTRA_RECT_WIDTH_RANGE)
            r0 = random.randint(2, h - rect_h - 2)
            c0 = random.randint(2, w - rect_w - 2)
            r1 = r0 + rect_h - 1
            c1 = c0 + rect_w - 1
            if not (r1 < band_top - 1 or r0 > band_bottom + 1):
                continue
            if not _can_place_rect(occupied, r0, r1, c0, c1, pad=1):
                continue
            color = random.choice(OBJECT_COLORS)
            g[r0 : r1 + 1, c0 : c1 + 1] = int(color)
            occupied[r0 : r1 + 1, c0 : c1 + 1] = True
            placed = True
            break
        if not placed:
            continue


def _would_row_move(row: np.ndarray, *, toward_left: bool) -> bool:
    return not np.array_equal(row, _pack_vector(row, toward_start=toward_left))


def _line_changed(inp: np.ndarray, out: np.ndarray, index: int, *, axis: str) -> bool:
    if axis == "row":
        return not np.array_equal(inp[int(index), :], out[int(index), :])
    return not np.array_equal(inp[:, int(index)], out[:, int(index)])


def _build_horizontal_scene(h: int, w: int, *, toward_left: bool) -> tuple[np.ndarray, list[int], set[int]] | None:
    selected_rows = _sample_selected_rows(h)
    if selected_rows is None:
        return None

    if len(selected_rows) < 2:
        return None

    anchor_rows = set(random.sample(selected_rows, random.randint(1, len(selected_rows) - 1)))
    band_top = min(selected_rows)
    band_bottom = max(selected_rows)

    interval_count = random.randint(*INTERVAL_COUNT_RANGE)
    intervals = _sample_intervals(
        2,
        w - 3,
        interval_count,
        INTERVAL_WIDTH_RANGE[0],
        INTERVAL_WIDTH_RANGE[1],
        INTERVAL_GAP,
    )
    if intervals is None:
        return None

    g = np.full((h, w), BLACK, dtype=int)
    interval_colors = random.sample(OBJECT_COLORS, len(intervals))
    for color, (c0, c1) in zip(interval_colors, intervals):
        g[band_top : band_bottom + 1, c0 : c1 + 1] = int(color)

    occupied = g != BLACK
    _place_extra_rectangles(g, band_top, band_bottom, occupied)

    if toward_left:
        _draw_arrows_left(g, selected_rows)
        _draw_anchors_right(g, anchor_rows)
    else:
        _draw_arrows_right(g, selected_rows)
        _draw_anchors_left(g, anchor_rows)

    return g, selected_rows, anchor_rows


def _generated_case_is_valid(inp: np.ndarray, selected: list[int], anchors: set[int], edge: str) -> bool:
    try:
        out = main(inp)
    except Exception:
        return False

    if np.array_equal(inp, out):
        return False

    axis = "row" if edge in ("left", "right") else "col"
    toward_left = edge in ("left", "top")

    moved_lines = [idx for idx in selected if idx not in anchors and _line_changed(inp, out, idx, axis=axis)]
    if not moved_lines:
        return False

    frozen_lines = [idx for idx in anchors if not _line_changed(inp, out, idx, axis=axis)]
    if not frozen_lines:
        return False

    if axis == "row":
        if not any(_would_row_move(inp[idx, :], toward_left=toward_left) for idx in anchors):
            return False
    else:
        if not any(_would_row_move(inp[:, idx], toward_left=toward_left) for idx in anchors):
            return False

    return True


def _gen_attempt() -> np.ndarray | None:
    direction = random.choice(("left", "right", "top", "bottom"))
    base_h = random.randint(*GRID_HEIGHT_RANGE)
    base_w = random.randint(*GRID_WIDTH_RANGE)

    horizontal_edge = "left" if direction in ("left", "top") else "right"
    built = _build_horizontal_scene(base_h, base_w, toward_left=(horizontal_edge == "left"))
    if built is None:
        return None
    base, selected_rows, anchor_rows = built

    if direction in ("left", "right"):
        grid = base
        selected = selected_rows
        anchors = anchor_rows
        edge = direction
    else:
        grid = base.T
        selected = selected_rows
        anchors = anchor_rows
        edge = direction

    if not _generated_case_is_valid(grid, selected, anchors, edge):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_GENERATION_ATTEMPTS} attempts")
