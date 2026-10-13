from __future__ import annotations

import random
from collections import Counter, deque

import numpy as np


BLACK = 0
GREY = 5
NON_SPECIAL_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
GEN_ATTEMPTS = 400


def _runs(flags: list[bool]) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start = None
    for idx, flag in enumerate(flags):
        if flag and start is None:
            start = idx
        elif not flag and start is not None:
            runs.append((start, idx))
            start = None
    if start is not None:
        runs.append((start, len(flags)))
    return runs


def _compress_upper(grid: np.ndarray, sep_row: int) -> np.ndarray:
    top = np.asarray(grid[:sep_row], dtype=int)
    assert top.size > 0, "expected a non-empty upper region"

    active_rows = [bool(np.any(top[r] != GREY)) for r in range(top.shape[0])]
    active_cols = [bool(np.any(top[:, c] != GREY)) for c in range(top.shape[1])]
    row_bands = _runs(active_rows)
    col_bands = _runs(active_cols)
    assert row_bands and col_bands, "failed to find separator-delimited regions"

    compact = np.zeros((len(row_bands), len(col_bands)), dtype=int)
    for i, (r0, r1) in enumerate(row_bands):
        for j, (c0, c1) in enumerate(col_bands):
            block = top[r0:r1, c0:c1]
            colors = {int(v) for v in block.flat if int(v) != GREY}
            assert len(colors) == 1, "each upper partition cell must be monochromatic"
            color = next(iter(colors))
            assert color not in (BLACK, GREY)
            compact[i, j] = color
    return compact


def _consistent_period_mapping(
    grid: np.ndarray, start_row: int, period: int
) -> dict[int, int] | None:
    mapping: dict[int, int] = {}
    for r in range(start_row, grid.shape[0]):
        for c in range(grid.shape[1]):
            color = int(grid[r, c])
            if color in (BLACK, GREY):
                continue
            residue = (r + c) % period
            if residue in mapping and mapping[residue] != color:
                return None
            mapping[residue] = color
    return mapping


def main(grid: np.ndarray) -> np.ndarray:
    """
    Compress the separator-partitioned top region and use its column count to complete a diagonal cycle below.

    Input structure:
    - The grid has a black background with a top region and a bottom region separated by one full grey row.
    - The top region is partitioned by grey horizontal and vertical bars into monochromatic rectangular cells.
    - The bottom region contains a diagonal periodic color pattern with some black holes.

    Rule:
    - Find the bottom-most full grey row; this is the separator between the two regions.
    - Compress the top region by collapsing each separator-delimited rectangle to one pixel, preserving row/column order.
    - Let C be the number of columns in that compact color grid. In the bottom region, infer the residue-to-color mapping for (row + col) mod C from the visible non-black pixels, then fill every black hole with its residue color.
    - Keep the top region and the grey separator row unchanged.

    Key insight:
    - The top region does not directly transform into the output; its compressed geometry provides the period length needed to decode the diagonal pattern below.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    full_grey_rows = [r for r in range(g.shape[0]) if np.all(g[r] == GREY)]
    assert full_grey_rows, "expected at least one full grey separator row"
    sep_row = max(full_grey_rows)
    assert 0 < sep_row < g.shape[0] - 1, "separator must split upper and lower regions"

    compact = _compress_upper(g, sep_row)
    period = int(compact.shape[1])
    assert 2 <= period <= 4, "upper compact width must be the encoded period"

    mapping = _consistent_period_mapping(g, sep_row + 1, period)
    assert mapping is not None and len(mapping) == period, "lower region must reveal every residue color"

    out = g.copy()
    for r in range(sep_row + 1, g.shape[0]):
        for c in range(g.shape[1]):
            if out[r, c] == BLACK:
                out[r, c] = mapping[(r + c) % period]
    return out


def _random_partition(total: int, parts: int, min_part: int, max_part: int) -> list[int] | None:
    if total < parts * min_part or total > parts * max_part:
        return None
    vals = [min_part] * parts
    remaining = total - parts * min_part
    capacities = [max_part - min_part] * parts
    while remaining > 0:
        choices = [i for i, cap in enumerate(capacities) if cap > 0]
        if not choices:
            return None
        idx = random.choice(choices)
        vals[idx] += 1
        capacities[idx] -= 1
        remaining -= 1
    random.shuffle(vals)
    return vals


def _contiguous(mask: np.ndarray) -> bool:
    pts = np.argwhere(mask)
    if len(pts) == 0:
        return False
    start = tuple(map(int, pts[0]))
    q = deque([start])
    seen = {start}
    h, w = mask.shape
    while q:
        r, c = q.popleft()
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                q.append((nr, nc))
    return len(seen) == int(mask.sum())


def _irregular_hole_patch(h: int, w: int) -> np.ndarray:
    patch = np.ones((h, w), dtype=bool)
    if random.random() < 0.5:
        candidates = [
            (r, c)
            for r in range(h)
            for c in range(w)
            if (r in (0, h - 1) or c in (0, w - 1))
            and not (r in (0, h - 1) and c in (0, w - 1))
        ]
        if candidates:
            random.shuffle(candidates)
            removals = random.randint(1, min(2, len(candidates)))
            removed = 0
            for r, c in candidates:
                if removed >= removals or patch.sum() <= 4:
                    break
                patch[r, c] = False
                if _contiguous(patch):
                    removed += 1
                else:
                    patch[r, c] = True
    return patch


def _place_holes(lower_h: int, width: int, period: int, sep_row: int) -> np.ndarray | None:
    hole_count = random.randint(2, 4)
    hole_mask = np.zeros((lower_h, width), dtype=bool)
    boxes: list[tuple[int, int, int, int]] = []
    max_h = min(4, lower_h - 1)
    max_w = min(5, width - 1)
    if max_h < 2 or max_w < 2:
        return None

    for _ in range(hole_count):
        placed = False
        for _ in range(80):
            h = random.randint(2, max_h)
            w = random.randint(2, max_w)
            r0 = random.randint(0, lower_h - h)
            c0 = random.randint(0, width - w)
            box = (r0, c0, r0 + h - 1, c0 + w - 1)
            if any(
                not (box[2] + 1 < br0 or br1 + 1 < box[0] or box[3] + 1 < bc0 or bc1 + 1 < box[1])
                for br0, bc0, br1, bc1 in boxes
            ):
                continue
            patch = _irregular_hole_patch(h, w)
            hole_mask[r0 : r0 + h, c0 : c0 + w] |= patch
            boxes.append(box)
            placed = True
            break
        if not placed:
            return None

    for residue in range(period):
        visible = False
        covered_cells: list[tuple[int, int]] = []
        for lr in range(lower_h):
            gr = sep_row + 1 + lr
            for c in range(width):
                if (gr + c) % period != residue:
                    continue
                if hole_mask[lr, c]:
                    covered_cells.append((lr, c))
                else:
                    visible = True
        if not visible:
            if not covered_cells:
                return None
            lr, c = random.choice(covered_cells)
            hole_mask[lr, c] = False

    hole_residues = {
        (sep_row + 1 + lr + c) % period
        for lr, c in zip(*np.where(hole_mask))
    }
    if hole_mask.sum() < max(4, period) or len(hole_residues) < 2:
        return None
    return hole_mask


def _choose_lower_palette(top_colors: set[int], period: int) -> list[int]:
    available = [c for c in NON_SPECIAL_COLORS if c not in top_colors]
    if len(available) >= period and random.random() < 0.7:
        return random.sample(available, period)
    return random.sample(NON_SPECIAL_COLORS, period)


def _build_expected(
    rows: int,
    width: int,
    top_h: int,
    row_heights: list[int],
    col_widths: list[int],
    region_colors: np.ndarray,
    lower_colors: list[int],
) -> np.ndarray:
    lower_h = rows - top_h - 1
    out = np.full((rows, width), BLACK, dtype=int)

    top = np.full((top_h, width), GREY, dtype=int)
    r = 0
    for i, band_h in enumerate(row_heights):
        c = 0
        for j, band_w in enumerate(col_widths):
            top[r : r + band_h, c : c + band_w] = int(region_colors[i, j])
            c += band_w
            if j + 1 < len(col_widths):
                top[:, c] = GREY
                c += 1
        r += band_h
        if i + 1 < len(row_heights):
            top[r, :] = GREY
            r += 1
    out[:top_h] = top
    out[top_h, :] = GREY

    period = len(lower_colors)
    for lr in range(lower_h):
        gr = top_h + 1 + lr
        for c in range(width):
            out[gr, c] = lower_colors[(gr + c) % period]
    return out


def _make_case() -> tuple[np.ndarray, np.ndarray] | None:
    period = random.randint(2, 4)
    region_rows = random.randint(2, 4)
    top_h = random.randint(max(5, 2 * region_rows - 1), 7)
    color_row_total = top_h - (region_rows - 1)
    row_heights = _random_partition(color_row_total, region_rows, 1, 3)
    if row_heights is None:
        return None

    col_widths = [random.randint(2, 4) for _ in range(period)]
    width = sum(col_widths) + (period - 1)
    lower_h = random.randint(8, 10)
    rows = top_h + 1 + lower_h

    palette_size = random.randint(2, min(5, len(NON_SPECIAL_COLORS)))
    top_palette = random.sample(NON_SPECIAL_COLORS, palette_size)
    region_colors = np.array(
        [[random.choice(top_palette) for _ in range(period)] for _ in range(region_rows)],
        dtype=int,
    )
    if len(set(map(int, region_colors.flat))) < 2:
        return None

    lower_colors = _choose_lower_palette(set(map(int, region_colors.flat)), period)
    random.shuffle(lower_colors)
    expected = _build_expected(
        rows, width, top_h, row_heights, col_widths, region_colors, lower_colors
    )

    holes = _place_holes(lower_h, width, period, top_h)
    if holes is None:
        return None

    inp = expected.copy()
    inp[top_h + 1 :, :][holes] = BLACK

    compact = _compress_upper(inp, top_h)
    if compact.shape[1] != period:
        return None

    mapping = _consistent_period_mapping(inp, top_h + 1, period)
    if mapping is None or len(mapping) != period:
        return None

    visible_colors = {int(v) for v in inp[top_h + 1 :, :].flat if int(v) not in (BLACK, GREY)}
    if visible_colors != set(lower_colors):
        return None

    for wrong_period in (2, 3, 4):
        if wrong_period == period:
            continue
        wrong_mapping = _consistent_period_mapping(inp, top_h + 1, wrong_period)
        if wrong_mapping is not None and len(wrong_mapping) == wrong_period:
            return None

    solved = main(inp)
    if not np.array_equal(solved, expected):
        return None
    if np.array_equal(inp, expected):
        return None
    return inp, expected


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        case = _make_case()
        if case is not None:
            return case[0]
    raise RuntimeError("failed to generate a valid puzzle instance")
