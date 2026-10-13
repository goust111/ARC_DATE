from __future__ import annotations

import random

import numpy as np


BLACK = 0
YELLOW = 4
ORANGE = 7

ROW_COUNT_RANGE = (2, 4)
COL_COUNT_RANGE = (2, 4)
INNER_HEIGHT_RANGE = (7, 11)
INNER_WIDTH_RANGE = (7, 11)
SPLIT_CHOICES = (2, 3, 4)
GEN_RETRY_LIMIT = 400


def _boundary_rows_cols(grid: np.ndarray) -> tuple[list[int], list[int]]:
    rows = [int(r) for r in range(grid.shape[0]) if np.all(grid[r, :] != BLACK)]
    cols = [int(c) for c in range(grid.shape[1]) if np.all(grid[:, c] != BLACK)]
    assert len(rows) >= 2 and len(cols) >= 2, "expected a framed rectangular partition"
    return rows, cols


def _solid_rect_bbox(region: np.ndarray) -> tuple[int, int, int, int, int]:
    coords = np.argwhere(region != BLACK)
    assert coords.size > 0, "each region must contain one object"
    r0, c0 = map(int, coords.min(axis=0))
    r1, c1 = map(int, coords.max(axis=0))
    color = int(region[r0, c0])
    patch = region[r0 : r1 + 1, c0 : c1 + 1]
    assert np.all(patch == color), "region object must be a solid rectangle"
    assert int(coords.shape[0]) == patch.size, "region must contain exactly one object"
    return r0, c0, r1, c1, color


def _resolve_parts(a: int, b: int) -> int:
    counts = [int(v) for v in (a, b) if int(v) >= 2]
    if not counts:
        return 1
    assert len(counts) == 1 or len(set(counts)) == 1, "conflicting marker counts on opposite borders"
    return int(counts[0])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Split each interior rectangle according to the marker count on its bordering separator lines.

    Horizontal-line yellow markers control how many vertical slices are produced.
    Vertical-line orange markers control how many horizontal slices are produced.
    Split pieces stay inside the same region with 1-pixel black gaps and are centered
    in the region. Regions with no adjacent markers are unchanged.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    row_bounds, col_bounds = _boundary_rows_cols(g)
    row_marker_counts = {r: int(np.count_nonzero(g[r, :] == YELLOW)) for r in row_bounds}
    col_marker_counts = {c: int(np.count_nonzero(g[:, c] == ORANGE)) for c in col_bounds}

    out = g.copy()

    for rb in range(len(row_bounds) - 1):
        top = int(row_bounds[rb])
        bottom = int(row_bounds[rb + 1])
        for cb in range(len(col_bounds) - 1):
            left = int(col_bounds[cb])
            right = int(col_bounds[cb + 1])

            region = g[top + 1 : bottom, left + 1 : right]
            inner_h, inner_w = map(int, region.shape)
            obj_r0, obj_c0, obj_r1, obj_c1, color = _solid_rect_bbox(region)
            obj_h = int(obj_r1 - obj_r0 + 1)
            obj_w = int(obj_c1 - obj_c0 + 1)

            col_parts = _resolve_parts(row_marker_counts[top], row_marker_counts[bottom])
            row_parts = _resolve_parts(col_marker_counts[left], col_marker_counts[right])
            if col_parts == 1 and row_parts == 1:
                continue

            assert obj_w % col_parts == 0, "object width must divide evenly into vertical slices"
            assert obj_h % row_parts == 0, "object height must divide evenly into horizontal slices"

            piece_w = int(obj_w // col_parts)
            piece_h = int(obj_h // row_parts)
            out_h = int(obj_h + row_parts - 1)
            out_w = int(obj_w + col_parts - 1)
            assert out_h <= inner_h and out_w <= inner_w, "split pieces must stay inside their region"

            out_region = out[top + 1 : bottom, left + 1 : right]
            out_region[region != BLACK] = BLACK

            start_r = int((inner_h - out_h) // 2)
            start_c = int((inner_w - out_w) // 2)
            for part_r in range(row_parts):
                rr = int(start_r + part_r * (piece_h + 1))
                for part_c in range(col_parts):
                    cc = int(start_c + part_c * (piece_w + 1))
                    out_region[rr : rr + piece_h, cc : cc + piece_w] = color

    return out


def _valid_masks(boundary_count: int) -> list[tuple[int, ...]]:
    masks: list[tuple[int, ...]] = []
    cell_count = int(boundary_count - 1)
    for bits in range(1, 1 << boundary_count):
        mask = tuple((bits >> i) & 1 for i in range(boundary_count))
        if any(mask[i] and mask[i + 1] for i in range(boundary_count - 1)):
            continue
        affected = [1 if mask[i] or mask[i + 1] else 0 for i in range(cell_count)]
        if 0 < sum(affected) < cell_count:
            masks.append(mask)
    assert masks
    return masks


def _marker_counts_from_mask(mask: tuple[int, ...]) -> list[int]:
    counts = [0] * len(mask)
    for i, flag in enumerate(mask):
        if flag:
            counts[i] = int(random.choice(SPLIT_CHOICES))
    return counts


def _band_parts(boundary_counts: list[int]) -> list[int]:
    return [_resolve_parts(boundary_counts[i], boundary_counts[i + 1]) for i in range(len(boundary_counts) - 1)]


def _arith_positions(length: int, count: int) -> list[int]:
    assert count >= 2 and length >= count
    max_step = int((length - 1) // (count - 1))
    steps = [d for d in range(2, max_step + 1)]
    step = int(random.choice(steps or [1]))
    slack = int(length - 1 - step * (count - 1))
    start = int(random.randint(0, slack))
    return [int(start + i * step) for i in range(count)]


def _choose_object_size(inner_h: int, inner_w: int, col_parts: int, row_parts: int) -> tuple[int, int]:
    heights = [
        obj_h
        for obj_h in range(row_parts, inner_h - 1)
        if obj_h % row_parts == 0 and obj_h + row_parts - 1 <= inner_h
    ]
    widths = [
        obj_w
        for obj_w in range(col_parts, inner_w - 1)
        if obj_w % col_parts == 0 and obj_w + col_parts - 1 <= inner_w
    ]

    candidates = [(h, w) for h in heights for w in widths if h * w >= 4]
    if not candidates:
        candidates = [(h, w) for h in heights for w in widths]
    assert candidates, "region must admit at least one valid rectangle"
    return random.choice(candidates)


def _adjacent_color_grid(rows: int, cols: int, palette: list[int]) -> list[list[int]]:
    colors: list[list[int]] = []
    for r in range(rows):
        cur_row: list[int] = []
        for c in range(cols):
            forbidden = set()
            if r > 0:
                forbidden.add(colors[r - 1][c])
            if c > 0:
                forbidden.add(cur_row[c - 1])
            choices = [color for color in palette if color not in forbidden] or palette
            cur_row.append(int(random.choice(choices)))
        colors.append(cur_row)
    return colors


def _build_candidate() -> np.ndarray:
    cell_rows = int(random.randint(*ROW_COUNT_RANGE))
    cell_cols = int(random.randint(*COL_COUNT_RANGE))

    inner_heights = [int(random.randint(*INNER_HEIGHT_RANGE)) for _ in range(cell_rows)]
    inner_widths = [int(random.randint(*INNER_WIDTH_RANGE)) for _ in range(cell_cols)]

    row_bounds = [0]
    for h in inner_heights:
        row_bounds.append(int(row_bounds[-1] + h + 1))
    col_bounds = [0]
    for w in inner_widths:
        col_bounds.append(int(col_bounds[-1] + w + 1))

    height = int(row_bounds[-1] + 1)
    width = int(col_bounds[-1] + 1)

    h_mask = random.choice(_valid_masks(cell_rows + 1))
    v_mask = random.choice(_valid_masks(cell_cols + 1))
    h_counts = _marker_counts_from_mask(h_mask)
    v_counts = _marker_counts_from_mask(v_mask)
    row_parts = _band_parts(h_counts)
    col_parts = _band_parts(v_counts)

    colors = [c for c in range(10) if c not in {BLACK, YELLOW, ORANGE}]
    h_color, v_color = random.sample(colors, 2)
    object_palette = [c for c in colors if c not in {h_color, v_color}]
    object_colors = _adjacent_color_grid(cell_rows, cell_cols, object_palette)

    grid = np.full((height, width), BLACK, dtype=int)
    grid[row_bounds, :] = h_color
    grid[:, col_bounds] = v_color

    for idx, count in enumerate(h_counts):
        if count < 2:
            continue
        seg = random.randrange(cell_cols)
        c0 = int(col_bounds[seg] + 1)
        positions = _arith_positions(inner_widths[seg], count)
        grid[row_bounds[idx], [c0 + pos for pos in positions]] = YELLOW

    for idx, count in enumerate(v_counts):
        if count < 2:
            continue
        seg = random.randrange(cell_rows)
        r0 = int(row_bounds[seg] + 1)
        positions = _arith_positions(inner_heights[seg], count)
        grid[[r0 + pos for pos in positions], col_bounds[idx]] = ORANGE

    for r in range(cell_rows):
        top = int(row_bounds[r])
        bottom = int(row_bounds[r + 1])
        inner_h = int(bottom - top - 1)
        for c in range(cell_cols):
            left = int(col_bounds[c])
            right = int(col_bounds[c + 1])
            inner_w = int(right - left - 1)

            obj_h, obj_w = _choose_object_size(inner_h, inner_w, row_parts[r], col_parts[c])
            obj_r = int(random.randint(1, inner_h - obj_h - 1))
            obj_c = int(random.randint(1, inner_w - obj_w - 1))
            color = int(object_colors[r][c])
            grid[top + obj_r : top + obj_r + obj_h, left + obj_c : left + obj_c + obj_w] = color

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        grid = _build_candidate()
        try:
            out = main(grid)
        except AssertionError:
            continue
        if np.array_equal(out, grid):
            continue
        return grid
    raise RuntimeError("failed to generate a valid input")
