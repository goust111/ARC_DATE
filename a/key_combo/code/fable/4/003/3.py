from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


ARC_COLORS = tuple(range(10))
NEIGHBORS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _background_color(g: np.ndarray) -> int:
    values, counts = np.unique(g, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _has_same_color_edge(g: np.ndarray, color: int) -> bool:
    mask = g == int(color)
    return bool(np.any(mask[:-1, :] & mask[1:, :]) or np.any(mask[:, :-1] & mask[:, 1:]))


def _barrier_color(g: np.ndarray, bg: int) -> int:
    colors = [int(c) for c in np.unique(g) if int(c) != int(bg)]
    candidates = [color for color in colors if _has_same_color_edge(g, color)]
    assert len(candidates) == 1, "expected one non-background color to form the barrier"
    return candidates[0]


def _label_regions(passable: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
    height, width = map(int, passable.shape)
    labels = np.full((height, width), -1, dtype=int)
    regions: list[np.ndarray] = []

    for r in range(height):
        for c in range(width):
            if not passable[r, c] or labels[r, c] != -1:
                continue
            label = len(regions)
            q: deque[tuple[int, int]] = deque([(r, c)])
            labels[r, c] = label
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for dr, dc in NEIGHBORS4:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < height and 0 <= nc < width and passable[nr, nc] and labels[nr, nc] == -1:
                        labels[nr, nc] = label
                        q.append((nr, nc))
            regions.append(np.array(coords, dtype=int))
    return labels, regions


def _solve(grid: np.ndarray) -> tuple[np.ndarray, int, int, set[int], np.ndarray, dict[int, int]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"
    assert max(g.shape) <= 30, "ARC grids may not exceed 30x30"

    bg = _background_color(g)
    barrier = _barrier_color(g, bg)
    palette = {int(c) for c in np.unique(g)} - {bg, barrier}
    assert len(palette) >= 2, "expected a shared dot palette"

    labels, regions = _label_regions(g != barrier)
    assert len(regions) >= 2, "barrier must split the canvas into several regions"

    out = g.copy()
    fill_by_region: dict[int, int] = {}
    for region_id, coords in enumerate(regions):
        rr = coords[:, 0]
        cc = coords[:, 1]
        colors_here = {int(c) for c in np.unique(g[rr, cc])}
        assert bg in colors_here, "each region must include background pixels"
        dot_colors = colors_here - {bg, barrier}
        missing = palette - dot_colors
        assert len(missing) == 1, "each region must miss exactly one palette color"
        fill = int(next(iter(missing)))
        fill_by_region[region_id] = fill
        out[rr, cc] = fill

    height, width = map(int, g.shape)
    for r, c in np.argwhere(g == barrier):
        adjacent_regions: list[int] = []
        for dr, dc in NEIGHBORS4:
            nr = int(r) + dr
            nc = int(c) + dc
            if 0 <= nr < height and 0 <= nc < width:
                region_id = int(labels[nr, nc])
                if region_id >= 0 and region_id not in adjacent_regions:
                    adjacent_regions.append(region_id)

        if len(adjacent_regions) < 2:
            continue
        counts = Counter(fill_by_region[region_id] for region_id in adjacent_regions)
        merge_colors = [int(color) for color, count in counts.items() if count >= 2]
        assert len(merge_colors) <= 1, "a barrier pixel cannot request two merge colors"
        if merge_colors:
            out[int(r), int(c)] = merge_colors[0]

    return out, bg, barrier, palette, labels, fill_by_region


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each barrier-separated region with its globally absent dot color, then remove same-fill walls.

    Input structure:
    - A single connected foreground barrier color splits the background into several 4-connected regions.
    - The remaining non-background foreground pixels are isolated colored dots inside those regions.
    - Across the whole grid, the dots form one shared palette, and every region is missing exactly one color from it.

    Rule:
    - Collect the full dot palette from the whole grid, excluding the background and the barrier color.
    - For each non-barrier region, find the one palette color absent from that region and repaint all its background and dot pixels with that color.
    - For any barrier pixel touching two or more regions that receive the same fill color, repaint that barrier pixel with the shared fill color; leave all other barrier pixels unchanged.

    Key insight:
    - A region's output color never appears inside that region, so the missing color can only be identified by comparing the region against the cross-region dot palette.
    """
    return _solve(grid)[0]


# --------------- generator ---------------

GEN_ATTEMPTS = 600
CELL_COUNT_RANGE = (2, 4)
CELL_SIZE_RANGE = (4, 6)
PALETTE_SIZES = (3, 4)
MAX_EXTRA_DOTS = 2


def _bands(sizes: list[int]) -> tuple[int, list[tuple[int, int]], list[int]]:
    bands: list[tuple[int, int]] = []
    separators: list[int] = []
    pos = 0
    for idx, size in enumerate(sizes):
        bands.append((pos, pos + int(size) - 1))
        pos += int(size)
        if idx != len(sizes) - 1:
            separators.append(pos)
            pos += 1
    return pos, bands, separators


def _adjacent_cell_pairs(rows: int, cols: int) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    pairs: list[tuple[tuple[int, int], tuple[int, int]]] = []
    for r in range(rows):
        for c in range(cols):
            if r + 1 < rows:
                pairs.append(((r, c), (r + 1, c)))
            if c + 1 < cols:
                pairs.append(((r, c), (r, c + 1)))
    return pairs


def _sample_missing_colors(rows: int, cols: int, palette: list[int]) -> dict[tuple[int, int], int] | None:
    pairs = _adjacent_cell_pairs(rows, cols)
    for _ in range(200):
        missing = {(r, c): int(random.choice(palette)) for r in range(rows) for c in range(cols)}
        used = set(missing.values())
        if len(used) < 2:
            continue
        has_same = any(missing[a] == missing[b] for a, b in pairs)
        has_diff = any(missing[a] != missing[b] for a, b in pairs)
        if has_same and has_diff:
            return missing
    return None


def _sample_spaced_positions(height: int, width: int, count: int) -> list[tuple[int, int]] | None:
    cells = [(r, c) for r in range(height) for c in range(width)]
    for _ in range(60):
        random.shuffle(cells)
        chosen: list[tuple[int, int]] = []
        for r, c in cells:
            if all(abs(r - rr) + abs(c - cc) > 1 for rr, cc in chosen):
                chosen.append((r, c))
                if len(chosen) == count:
                    return chosen
    return None


def _paint_dots(
    grid: np.ndarray,
    row_bands: list[tuple[int, int]],
    col_bands: list[tuple[int, int]],
    palette: list[int],
    missing: dict[tuple[int, int], int],
) -> bool:
    for cell_r, (r0, r1) in enumerate(row_bands):
        for cell_c, (c0, c1) in enumerate(col_bands):
            absent = int(missing[(cell_r, cell_c)])
            dot_colors = [int(color) for color in palette if int(color) != absent]
            extra_count = random.randint(0, MAX_EXTRA_DOTS)
            dot_colors.extend(random.choice(dot_colors) for _ in range(extra_count))
            random.shuffle(dot_colors)

            positions = _sample_spaced_positions(r1 - r0 + 1, c1 - c0 + 1, len(dot_colors))
            if positions is None:
                return False
            for (dr, dc), color in zip(positions, dot_colors):
                grid[r0 + dr, c0 + dc] = int(color)
    return True


def _expected_from_layout(
    grid: np.ndarray,
    row_bands: list[tuple[int, int]],
    col_bands: list[tuple[int, int]],
    row_seps: list[int],
    col_seps: list[int],
    missing: dict[tuple[int, int], int],
    barrier: int,
) -> np.ndarray:
    out = grid.copy()
    for cell_r, (r0, r1) in enumerate(row_bands):
        for cell_c, (c0, c1) in enumerate(col_bands):
            out[r0 : r1 + 1, c0 : c1 + 1] = int(missing[(cell_r, cell_c)])

    for cell_r, (r0, r1) in enumerate(row_bands):
        for cell_c, sep_c in enumerate(col_seps):
            left = int(missing[(cell_r, cell_c)])
            right = int(missing[(cell_r, cell_c + 1)])
            if left == right:
                out[r0 : r1 + 1, sep_c] = left

    for sep_r_idx, sep_r in enumerate(row_seps):
        for cell_c, (c0, c1) in enumerate(col_bands):
            top = int(missing[(sep_r_idx, cell_c)])
            bottom = int(missing[(sep_r_idx + 1, cell_c)])
            if top == bottom:
                out[sep_r, c0 : c1 + 1] = top

    out[np.asarray(grid == barrier) & np.asarray(out == grid)] = barrier
    return out


def _layout_checks(
    grid: np.ndarray,
    expected: np.ndarray,
    row_bands: list[tuple[int, int]],
    col_bands: list[tuple[int, int]],
    palette: list[int],
    missing: dict[tuple[int, int], int],
    barrier: int,
) -> bool:
    dot_palette = {int(c) for c in np.unique(grid)} - {_background_color(grid), int(barrier)}
    if dot_palette != set(map(int, palette)):
        return False

    removed_wall = bool(np.any((grid == barrier) & (expected != barrier)))
    kept_wall = bool(np.any((grid == barrier) & (expected == barrier)))
    if not removed_wall or not kept_wall:
        return False

    if np.array_equal(grid, expected):
        return False

    for cell_r, (r0, r1) in enumerate(row_bands):
        for cell_c, (c0, c1) in enumerate(col_bands):
            absent = int(missing[(cell_r, cell_c)])
            patch = grid[r0 : r1 + 1, c0 : c1 + 1]
            if np.any(patch == absent):
                return False
            if not np.all(expected[r0 : r1 + 1, c0 : c1 + 1] == absent):
                return False
    return True


def _gen_attempt() -> np.ndarray | None:
    row_count = random.randint(*CELL_COUNT_RANGE)
    col_count = random.randint(*CELL_COUNT_RANGE)
    row_sizes = [random.randint(*CELL_SIZE_RANGE) for _ in range(row_count)]
    col_sizes = [random.randint(*CELL_SIZE_RANGE) for _ in range(col_count)]
    height, row_bands, row_seps = _bands(row_sizes)
    width, col_bands, col_seps = _bands(col_sizes)
    if max(height, width) > 30:
        return None

    bg = random.choice(ARC_COLORS)
    barrier = random.choice([color for color in ARC_COLORS if color != bg])
    palette_size = random.choice(PALETTE_SIZES)
    palette = random.sample([color for color in ARC_COLORS if color not in {bg, barrier}], palette_size)

    missing = _sample_missing_colors(row_count, col_count, palette)
    if missing is None:
        return None

    grid = np.full((height, width), int(bg), dtype=int)
    if row_seps:
        grid[row_seps, :] = int(barrier)
    if col_seps:
        grid[:, col_seps] = int(barrier)

    if not _paint_dots(grid, row_bands, col_bands, palette, missing):
        return None

    expected = _expected_from_layout(grid, row_bands, col_bands, row_seps, col_seps, missing, barrier)
    if not _layout_checks(grid, expected, row_bands, col_bands, palette, missing, barrier):
        return None

    try:
        solved = main(grid)
    except Exception:
        return None
    if not np.array_equal(solved, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid absent-color barrier input")
