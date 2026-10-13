from __future__ import annotations

import random

import numpy as np


BG = 0
BLUE = 1
RED = 2
GREEN = 3
OP_MARKERS = (BLUE, RED, GREEN)
PATTERN_COLORS = (4, 5, 6, 7, 8, 9)

GRID_WIDTH_RANGE = (14, 28)
ROW_COUNT_RANGE = (3, 6)
MARGIN_RANGE = (0, 2)
PATTERN_WIDTH_MIN = 5
PATTERN_WIDTH_MAX = 9
MAX_GENERATION_ATTEMPTS = 400


def _apply_operation(segment: np.ndarray, op: int) -> np.ndarray:
    if op == RED:
        return segment[::-1].copy()
    if op == BLUE:
        return segment.copy()
    if op == GREEN:
        return np.roll(segment, 2)
    raise AssertionError(f"unknown operation marker: {op}")


def _parse_scene(grid: np.ndarray) -> tuple[np.ndarray, list[dict[str, object]], list[int]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"
    h, w = map(int, g.shape)
    assert w >= 6, "grid is too narrow for left marker, pattern, and right key"

    active_rows: list[dict[str, object]] = []
    for r in range(h):
        row = g[r]
        if not np.any(row != BG):
            continue

        op = int(row[0])
        key_color = int(row[-1])
        assert op in OP_MARKERS, "every non-empty row must start with a valid operation marker"
        assert key_color != BG, "every seed row must contain a right-edge key color"

        interior = row[1:-1]
        non_bg_cols = np.flatnonzero(interior != BG)
        assert len(non_bg_cols) > 0, "every seed row must contain an interior pattern"

        left = int(non_bg_cols[0]) + 1
        right = int(non_bg_cols[-1]) + 1
        assert left >= 2, "pattern must be separated from the operation marker"
        assert right <= w - 3, "pattern must leave room before the right-edge key column"

        segment = row[left : right + 1].copy()
        pattern_colors = set(map(int, np.unique(segment))) - {BG}
        assert len(pattern_colors) == 1, "each seed row pattern must be monochrome"
        pattern_color = pattern_colors.pop()

        residue = row.copy()
        residue[0] = BG
        residue[-1] = BG
        residue[left : right + 1] = BG
        assert not np.any(residue != BG), "non-background cells may appear only in the marker, pattern span, and key slot"

        active_rows.append(
            {
                "row": int(r),
                "op": op,
                "left": left,
                "right": right,
                "segment": segment,
                "pattern_color": pattern_color,
                "key_color": key_color,
            }
        )

    assert active_rows, "expected at least one seed row"
    for prev, cur in zip(active_rows, active_rows[1:]):
        assert int(cur["row"]) - int(prev["row"]) >= 2, "seed rows must be separated by a blank row"

    pattern_colors = [int(info["pattern_color"]) for info in active_rows]
    key_order = [int(g[r, -1]) for r in range(h) if int(g[r, -1]) != BG]
    assert len(set(pattern_colors)) == len(pattern_colors), "pattern colors must uniquely identify the seed rows"
    assert len(key_order) == len(active_rows), "right-edge key column must have one color per seed row"
    assert len(set(key_order)) == len(key_order), "right-edge key colors must be unique"
    assert set(key_order) == set(pattern_colors), "right-edge key colors must permute the seed-row pattern colors"
    return g, active_rows, key_order


def _solve(grid: np.ndarray) -> np.ndarray:
    g, active_rows, key_order = _parse_scene(grid)
    out = np.full_like(g, BG)

    rows_by_color = {int(info["pattern_color"]): info for info in active_rows}
    target_rows = [int(info["row"]) for info in active_rows]

    for target_row, color in zip(target_rows, key_order):
        info = rows_by_color[int(color)]
        left = int(info["left"])
        right = int(info["right"])
        op = int(info["op"])
        segment = np.asarray(info["segment"], dtype=int)

        out[target_row, 0] = op
        out[target_row, -1] = int(color)
        out[target_row, left : right + 1] = _apply_operation(segment, op)

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reorder the seed rows by the right-edge color key and then apply each row's left-edge operation marker.

    Input structure:
    - The background is black.
    - Several active rows are separated by one blank background row.
    - Each active row has a left-edge operation marker, a monochrome interior pattern occupying middle columns, and a single right-edge key pixel.
    - The interior pattern colors are unique across rows, and the right-edge key pixels form a top-to-bottom permutation of those pattern colors.

    Rule:
    - Read the non-black pixels in the last column from top to bottom to obtain the target row-color order.
    - Match each color in that order to the unique seed row whose interior pattern uses the same color.
    - Rebuild the same-size canvas by placing those matched rows into the existing seed-row slots from top to bottom, keeping one blank row between slots.
    - On each placed row, transform only its interior pattern span: red reverses the span, blue leaves it unchanged, and green cyclically shifts the span two cells to the right.

    Key insight:
    - The rightmost column is a permutation key for row order, while the leftmost marker chooses an independent row-local transformation; both signals are required.
    """
    return _solve(grid)


def _sample_derangement(items: list[int], rng: random.Random) -> list[int]:
    if len(items) < 2:
        return items[:]

    for _ in range(200):
        candidate = items[:]
        rng.shuffle(candidate)
        if all(a != b for a, b in zip(items, candidate)):
            return candidate
    raise RuntimeError("failed to sample a derangement")


def _sample_mask(width: int, op: int, used_masks: set[tuple[int, ...]], rng: random.Random) -> tuple[int, ...] | None:
    for _ in range(300):
        inner = [1 if rng.random() < 0.58 else 0 for _ in range(width - 2)]
        mask = tuple([1, *inner, 1])
        if sum(mask) < 3:
            continue
        if 0 not in inner:
            continue
        if mask in used_masks:
            continue

        binary = np.array(mask, dtype=int)
        if op == RED and np.array_equal(binary, binary[::-1]):
            continue
        if op == GREEN and np.array_equal(binary, np.roll(binary, 2)):
            continue
        return mask
    return None


def _segment_left(grid_width: int, width: int, rng: random.Random) -> int:
    max_left = grid_width - width - 2
    assert max_left >= 2, "grid must leave a background gap on both sides of the pattern"

    centered = (grid_width - width) // 2
    lo = max(2, centered - 2)
    hi = min(max_left, centered + 2)
    assert lo <= hi, "pattern placement window must be non-empty"
    return rng.randint(lo, hi)


def _render_segment(mask: tuple[int, ...], color: int) -> np.ndarray:
    segment = np.full(len(mask), BG, dtype=int)
    for idx, bit in enumerate(mask):
        if bit:
            segment[idx] = int(color)
    return segment


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    row_count = rng.randint(*ROW_COUNT_RANGE)
    top_margin = rng.randint(*MARGIN_RANGE)
    bottom_margin = rng.randint(*MARGIN_RANGE)
    height = top_margin + (2 * row_count - 1) + bottom_margin

    width = rng.randint(*GRID_WIDTH_RANGE)
    width_hi = min(PATTERN_WIDTH_MAX, width - 6)
    if width_hi < PATTERN_WIDTH_MIN:
        return None

    ops = [RED, BLUE, GREEN] + [rng.choice(OP_MARKERS) for _ in range(row_count - 3)]
    rng.shuffle(ops)
    colors = rng.sample(PATTERN_COLORS, row_count)

    used_masks: set[tuple[int, ...]] = set()
    row_specs: list[dict[str, object]] = []
    for op, color in zip(ops, colors):
        width_i = rng.randint(PATTERN_WIDTH_MIN, width_hi)
        mask = _sample_mask(width_i, int(op), used_masks, rng)
        if mask is None:
            return None
        used_masks.add(mask)
        row_specs.append({"op": int(op), "color": int(color), "mask": mask})

    local_color_order = [int(spec["color"]) for spec in row_specs]
    key_order = _sample_derangement(local_color_order, rng)

    grid = np.full((height, width), BG, dtype=int)
    for idx, spec in enumerate(row_specs):
        row = top_margin + 2 * idx
        mask = spec["mask"]
        left = _segment_left(width, len(mask), rng)
        segment = _render_segment(mask, int(spec["color"]))

        grid[row, 0] = int(spec["op"])
        grid[row, left : left + len(segment)] = segment
        grid[row, -1] = int(key_order[idx])

    try:
        parsed_grid, active_rows, parsed_order = _parse_scene(grid)
        assert parsed_grid.shape == grid.shape
        assert parsed_order == key_order
        assert {int(info["op"]) for info in active_rows} == {RED, BLUE, GREEN}
        for info in active_rows:
            op = int(info["op"])
            if op in (RED, GREEN):
                transformed = _apply_operation(np.asarray(info["segment"], dtype=int), op)
                assert not np.array_equal(transformed, np.asarray(info["segment"], dtype=int))

        out = _solve(grid)
        assert out.shape == grid.shape
        assert not np.array_equal(out, grid)
        return grid
    except AssertionError:
        return None


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_GENERATION_ATTEMPTS} attempts")
