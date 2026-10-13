from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _components_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = map(int, mask.shape)
    seen = np.zeros((h, w), dtype=bool)
    components: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                coords.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            components.append(np.array(coords, dtype=int))

    return components


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _touches_edge(coords: np.ndarray, h: int, w: int) -> bool:
    return bool(
        np.any(coords[:, 0] == 0)
        or np.any(coords[:, 0] == h - 1)
        or np.any(coords[:, 1] == 0)
        or np.any(coords[:, 1] == w - 1)
    )


def _extract_summary(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = map(int, g.shape)
    bg = _get_bg(g)

    slot_bboxes: list[tuple[int, int, int, int]] = []
    for coords in _components_mask(g == bg):
        if _touches_edge(coords, h, w):
            continue
        slot_bboxes.append(_bbox_coords(coords))

    assert slot_bboxes, "expected enclosed background slots"
    row_starts = sorted({box[0] for box in slot_bboxes})
    col_starts = sorted({box[1] for box in slot_bboxes})
    assert len(slot_bboxes) == len(row_starts) * len(col_starts), "slots must form a full table"

    slot_by_anchor = {(r0, c0): (r0, c0, r1, c1) for r0, c0, r1, c1 in slot_bboxes}
    assert len(slot_by_anchor) == len(slot_bboxes), "slot anchors must be unique"

    summary = np.full((len(row_starts), len(col_starts)), bg, dtype=int)
    for out_r, r0 in enumerate(row_starts):
        for out_c, c0 in enumerate(col_starts):
            assert (r0, c0) in slot_by_anchor, "missing slot in table layout"
            sr0, sc0, sr1, sc1 = slot_by_anchor[(r0, c0)]
            window = g[sr0 : sr1 + 1, sc0 : sc1 + 1]
            colors = np.unique(window[window != bg])
            assert colors.size == 1, "each slot must contain exactly one object color"
            summary[out_r, out_c] = int(colors[0])

    return summary


def _candidate_for_step(summary: np.ndarray, step: int) -> tuple[np.ndarray, np.ndarray] | None:
    rows, cols = map(int, summary.shape)
    base = np.empty(cols, dtype=int)

    for k in range(cols):
        diagonal = [int(summary[r, (k + step * r) % cols]) for r in range(rows)]
        counts = Counter(diagonal).most_common()
        if len(counts) > 1 and counts[0][1] == counts[1][1]:
            return None
        base[k] = int(counts[0][0])

    candidate = np.stack([np.roll(base, step * r) for r in range(rows)], axis=0)
    mismatches = np.argwhere(candidate != summary)
    return candidate, mismatches


def _repair_summary(summary: np.ndarray) -> np.ndarray:
    solutions: dict[tuple[tuple[int, int], bytes], np.ndarray] = {}

    for step in (-1, 1):
        result = _candidate_for_step(summary, step)
        if result is None:
            continue
        candidate, mismatches = result
        if len(mismatches) == 1:
            solutions[(candidate.shape, candidate.tobytes())] = candidate

    if len(solutions) == 1:
        return next(iter(solutions.values()))
    if len(solutions) > 1:
        candidates = list(solutions.values())
        first = candidates[0]
        if all(np.array_equal(first, other) for other in candidates[1:]):
            return first
    raise AssertionError("expected a unique one-cell repair on the summary matrix")


def main(grid: np.ndarray) -> np.ndarray:
    """Compress the slot table to a color matrix and repair the single broken cyclic diagonal.

    Input structure:
    - The most common color is both the outer background and the background of every enclosed slot.
    - The non-edge background components form a complete rectangular table of slots.
    - Each slot contains exactly one monochrome object whose color is the slot's summary value.

    Rule:
    - Find all enclosed background slots and order them by their bounding-box row and column positions.
    - Read the unique object color inside each slot to form the summary matrix.
    - Exactly one summary cell breaks the rule that each row is the previous row cyclically shifted by one step in a fixed direction; change that cell to the required color.

    Key insight:
    - The anomaly is only visible after compressing the large picture into its slot-color summary, where the intended pattern becomes diagonal constancy under wraparound.
    """
    summary = _extract_summary(np.asarray(grid, dtype=int))
    return _repair_summary(summary)


_MAX_GEN_ATTEMPTS = 500
_ROWS_RANGE = (3, 4)
_COLS_RANGE = (3, 4)
_SLOT_SIZE_RANGE = (4, 5)
_MARGIN_RANGE = (1, 2)
_PALETTE = tuple(range(10))

# Small connected shapes without holes; all are rendered strictly inside a slot.
_SHAPES = (
    ((0, 0),),
    ((0, 0), (0, 1)),
    ((0, 0), (1, 0)),
    ((0, 0), (0, 1), (0, 2)),
    ((0, 0), (1, 0), (2, 0)),
    ((0, 0), (0, 1), (1, 0)),
    ((0, 0), (0, 1), (1, 1)),
    ((0, 0), (1, 0), (1, 1)),
    ((0, 1), (1, 0), (1, 1)),
    ((0, 0), (0, 1), (1, 0), (1, 1)),
)


def _sample_table_layout() -> tuple[list[int], list[int], tuple[int, int, int, int]] | None:
    rows = random.randint(*_ROWS_RANGE)
    cols = random.randint(*_COLS_RANGE)
    slot_heights = [random.randint(*_SLOT_SIZE_RANGE) for _ in range(rows)]
    slot_widths = [random.randint(*_SLOT_SIZE_RANGE) for _ in range(cols)]
    margins = tuple(random.randint(*_MARGIN_RANGE) for _ in range(4))
    top, bottom, left, right = margins

    table_h = sum(slot_heights) + rows + 1
    table_w = sum(slot_widths) + cols + 1
    if top + table_h + bottom > 30 or left + table_w + right > 30:
        return None
    return slot_heights, slot_widths, margins


def _draw_table(
    grid: np.ndarray,
    wall_color: int,
    top: int,
    left: int,
    slot_heights: list[int],
    slot_widths: list[int],
) -> list[list[tuple[int, int, int, int]]]:
    table_h = sum(slot_heights) + len(slot_heights) + 1
    table_w = sum(slot_widths) + len(slot_widths) + 1

    rr = top
    grid[rr, left : left + table_w] = wall_color
    for height in slot_heights:
        rr += height + 1
        grid[rr, left : left + table_w] = wall_color

    cc = left
    grid[top : top + table_h, cc] = wall_color
    for width in slot_widths:
        cc += width + 1
        grid[top : top + table_h, cc] = wall_color

    slots: list[list[tuple[int, int, int, int]]] = []
    r = top + 1
    for height in slot_heights:
        row_slots: list[tuple[int, int, int, int]] = []
        c = left + 1
        for width in slot_widths:
            row_slots.append((r, c, height, width))
            c += width + 1
        slots.append(row_slots)
        r += height + 1
    return slots


def _fit_shapes(inner_h: int, inner_w: int) -> list[tuple[tuple[int, int], ...]]:
    fits = []
    for shape in _SHAPES:
        max_r = max(r for r, _ in shape) + 1
        max_c = max(c for _, c in shape) + 1
        if max_r <= inner_h and max_c <= inner_w:
            fits.append(shape)
    return fits


def _place_object(grid: np.ndarray, slot: tuple[int, int, int, int], color: int) -> None:
    r0, c0, height, width = slot
    inner_h = height - 2
    inner_w = width - 2
    fits = _fit_shapes(inner_h, inner_w)
    assert fits, "slot interior must support at least one object shape"

    shape = random.choice(fits)
    shape_h = max(r for r, _ in shape) + 1
    shape_w = max(c for _, c in shape) + 1
    start_r = r0 + 1 + random.randint(0, inner_h - shape_h)
    start_c = c0 + 1 + random.randint(0, inner_w - shape_w)
    for dr, dc in shape:
        grid[start_r + dr, start_c + dc] = color


def _build_correct_summary(rows: int, cols: int, colors: list[int], step: int) -> np.ndarray:
    base = np.array(random.sample(colors, cols), dtype=int)
    return np.stack([np.roll(base, step * r) for r in range(rows)], axis=0)


def _gen_attempt() -> tuple[np.ndarray, np.ndarray] | None:
    layout = _sample_table_layout()
    if layout is None:
        return None
    slot_heights, slot_widths, margins = layout
    top, bottom, left, right = margins
    rows = len(slot_heights)
    cols = len(slot_widths)

    bg, wall = random.sample(_PALETTE, 2)
    object_colors = [color for color in _PALETTE if color not in {bg, wall}]
    correct = _build_correct_summary(rows, cols, object_colors, random.choice((-1, 1)))

    broken = correct.copy()
    bad_r = random.randrange(rows)
    bad_c = random.randrange(cols)
    wrong_choices = [color for color in object_colors if color not in set(correct[0].tolist()) and color != int(correct[bad_r, bad_c])]
    if not wrong_choices:
        wrong_choices = [color for color in object_colors if color != int(correct[bad_r, bad_c])]
    broken[bad_r, bad_c] = random.choice(wrong_choices)

    table_h = sum(slot_heights) + rows + 1
    table_w = sum(slot_widths) + cols + 1
    grid = np.full((top + table_h + bottom, left + table_w + right), bg, dtype=int)
    slots = _draw_table(grid, wall, top, left, slot_heights, slot_widths)

    for r in range(rows):
        for c in range(cols):
            _place_object(grid, slots[r][c], int(broken[r, c]))

    try:
        solved = main(grid)
    except Exception:
        return None
    if not np.array_equal(solved, correct):
        return None
    return grid, correct


def generate_input() -> np.ndarray:
    for _ in range(_MAX_GEN_ATTEMPTS):
        attempt = _gen_attempt()
        if attempt is not None:
            grid, _expected = attempt
            return grid
    raise RuntimeError("failed to generate a valid input")
