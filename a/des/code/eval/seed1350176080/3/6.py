from __future__ import annotations

from collections import deque
import random

import numpy as np


def _background_color(grid: np.ndarray) -> int:
    colors, counts = np.unique(grid, return_counts=True)
    return int(colors[int(np.argmax(counts))])


def _runs(indices: np.ndarray) -> list[tuple[int, int]]:
    values = [int(x) for x in indices]
    assert values, "expected at least one active index"
    out: list[tuple[int, int]] = []
    start = prev = values[0]
    for value in values[1:]:
        if value == prev + 1:
            prev = value
            continue
        out.append((start, prev))
        start = prev = value
    out.append((start, prev))
    return out


def _neighbors4(r: int, c: int, h: int, w: int):
    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        rr, cc = r + dr, c + dc
        if 0 <= rr < h and 0 <= cc < w:
            yield rr, cc


def _mask_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    h, w = mask.shape
    start = tuple(map(int, coords[0]))
    seen = {start}
    q = deque([start])
    while q:
        r, c = q.popleft()
        for rr, cc in _neighbors4(r, c, h, w):
            if not mask[rr, cc] or (rr, cc) in seen:
                continue
            seen.add((rr, cc))
            q.append((rr, cc))
    return len(seen) == int(mask.sum())


def _cells_connected(cells: set[tuple[int, int]]) -> bool:
    if not cells:
        return False
    start = next(iter(cells))
    seen = {start}
    q = deque([start])
    while q:
        r, c = q.popleft()
        for rr, cc in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)):
            if (rr, cc) in cells and (rr, cc) not in seen:
                seen.add((rr, cc))
                q.append((rr, cc))
    return len(seen) == len(cells)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Overlay horizontally arranged incomplete copies of one template, then recolor each template pixel by the marker above the copy where that pixel is missing.

    Input structure:
    - The dominant color is background.
    - Several equal-size copies of one foreground template are laid out side by side, separated by background columns.
    - Each copy uses the same foreground color but has one connected missing region replaced by background.
    - One row directly above the copies contains one unique marker pixel per copy, centered over that copy.

    Rule:
    - Recover the copy rectangles and the shared foreground template.
    - For each template pixel, find the unique copy where that position is missing.
    - Output one template-sized grid where each template pixel is recolored with that copy's marker color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "grid must be 2D"

    bg = _background_color(g)
    colors, counts = np.unique(g, return_counts=True)
    count_by_color = {int(color): int(count) for color, count in zip(colors, counts)}

    marker_colors = [color for color, count in count_by_color.items() if color != bg and count == 1]
    assert len(marker_colors) >= 2, "expected at least two unique marker colors"
    marker_set = set(marker_colors)

    marker_entries: list[tuple[int, int, int]] = []
    for color in marker_colors:
        coords = np.argwhere(g == color)
        assert len(coords) == 1
        r, c = map(int, coords[0])
        marker_entries.append((r, c, int(color)))

    marker_rows = {r for r, _c, _color in marker_entries}
    assert len(marker_rows) == 1, "marker pixels must share a row"
    marker_row = next(iter(marker_rows))
    marker_entries.sort(key=lambda item: item[1])

    fg_colors = [color for color in count_by_color if color != bg and color not in marker_set]
    assert len(fg_colors) == 1, f"expected one shared foreground color, got {fg_colors}"
    fg = int(fg_colors[0])

    fg_coords = np.argwhere(g == fg)
    assert len(fg_coords) > 0, "expected foreground pixels"
    r0 = int(fg_coords[:, 0].min())
    r1 = int(fg_coords[:, 0].max())
    assert marker_row == r0 - 1, "markers must sit directly above the copies"

    band = g[r0 : r1 + 1]
    active_cols = np.where(np.any(band == fg, axis=0))[0]
    intervals = _runs(active_cols)
    assert len(intervals) == len(marker_entries), "copy count must match marker count"
    widths = {int(c1 - c0 + 1) for c0, c1 in intervals}
    assert len(widths) == 1, "all copies must have the same width"

    copies: list[np.ndarray] = []
    ordered_marker_colors: list[int] = []
    for (c0, c1), (mr, mc, color) in zip(intervals, marker_entries):
        assert c0 <= mc <= c1, "marker must lie above its copy"
        assert mr == marker_row
        patch = band[:, c0 : c1 + 1]
        assert np.all((patch == bg) | (patch == fg)), "copies may only contain background and the shared foreground color"
        copies.append(patch)
        ordered_marker_colors.append(int(color))

    copy_stack = np.stack([copy == fg for copy in copies], axis=0)
    template = np.any(copy_stack, axis=0)
    missing_counts = np.sum(template[None, :, :] & ~copy_stack, axis=0)
    assert np.all(missing_counts[template] == 1), "each template pixel must be missing from exactly one copy"

    out = np.full(template.shape, bg, dtype=int)
    for color, present in zip(ordered_marker_colors, copy_stack):
        out[template & ~present] = int(color)
    return out


# --------------- generator ---------------

PALETTE = tuple(range(10))
COPY_COUNT_CHOICES = (2, 3, 4)
HEIGHT_RANGE = (5, 8)
WIDTH_CHOICES = (5, 7, 9)
GAP_RANGE = (1, 3)
TOP_MARGIN_RANGE = (0, 2)
BOTTOM_MARGIN_RANGE = (0, 2)
SIDE_MARGIN_RANGE = (0, 2)
MAX_GRID_WIDTH = 30
MIN_TEMPLATE_DENSITY = 0.58
MAX_TEMPLATE_DENSITY = 0.82
MIN_LINE_CELLS = 3
MIN_REGION_AREA = 2
TEMPLATE_RETRIES = 120
PARTITION_RETRIES = 180
GENERATION_RETRIES = 400


def _touches_boundary(mask: np.ndarray, r: int, c: int) -> bool:
    h, w = mask.shape
    if r in (0, h - 1) or c in (0, w - 1):
        return True
    return any(not mask[rr, cc] for rr, cc in _neighbors4(r, c, h, w))


def _template_target_bounds(h: int, w: int, n_regions: int) -> tuple[int, int] | None:
    total = h * w
    lo = max(
        n_regions * MIN_REGION_AREA,
        MIN_LINE_CELLS * h,
        MIN_LINE_CELLS * w,
        int(np.ceil(total * MIN_TEMPLATE_DENSITY)),
    )
    hi = min(total - 1, int(np.floor(total * MAX_TEMPLATE_DENSITY)))
    if lo > hi:
        return None
    return lo, hi


def _build_template_mask(h: int, w: int, n_regions: int) -> np.ndarray | None:
    bounds = _template_target_bounds(h, w, n_regions)
    if bounds is None:
        return None
    lo, hi = bounds

    for _ in range(TEMPLATE_RETRIES):
        target = random.randint(lo, hi)
        mask = np.ones((h, w), dtype=bool)
        while int(mask.sum()) > target:
            boundary = [
                (int(r), int(c))
                for r, c in np.argwhere(mask)
                if _touches_boundary(mask, int(r), int(c))
                and int(mask[int(r)].sum()) > MIN_LINE_CELLS
                and int(mask[:, int(c)].sum()) > MIN_LINE_CELLS
            ]
            if not boundary:
                break
            random.shuffle(boundary)
            removed = False
            for r, c in boundary:
                mask[r, c] = False
                if _mask_connected(mask):
                    removed = True
                    break
                mask[r, c] = True
            if not removed:
                break

        if int(mask.sum()) != target:
            continue
        if np.any(mask.sum(axis=0) < MIN_LINE_CELLS) or np.any(mask.sum(axis=1) < MIN_LINE_CELLS):
            continue
        if not _mask_connected(mask):
            continue
        return mask
    return None


def _seed_cells(mask: np.ndarray, n_regions: int) -> list[tuple[int, int]] | None:
    cells = [tuple(map(int, coord)) for coord in np.argwhere(mask)]
    if len(cells) < n_regions:
        return None
    first = random.choice(cells)
    seeds = [first]
    remaining = set(cells)
    remaining.remove(first)

    while len(seeds) < n_regions and remaining:
        ranked = []
        for cell in remaining:
            dist = min(abs(cell[0] - sr) + abs(cell[1] - sc) for sr, sc in seeds)
            ranked.append((dist, cell))
        best = max(dist for dist, _cell in ranked)
        pool = [cell for dist, cell in ranked if dist >= best - 1]
        chosen = random.choice(pool)
        seeds.append(chosen)
        remaining.remove(chosen)

    return seeds if len(seeds) == n_regions else None


def _grow_partition(mask: np.ndarray, n_regions: int) -> list[set[tuple[int, int]]] | None:
    h, w = mask.shape
    cell_set = {tuple(map(int, coord)) for coord in np.argwhere(mask)}
    seeds = _seed_cells(mask, n_regions)
    if seeds is None:
        return None

    regions = [{seed} for seed in seeds]
    owner = {seed: idx for idx, seed in enumerate(seeds)}
    frontiers = []
    for seed in seeds:
        frontier = {nbr for nbr in _neighbors4(seed[0], seed[1], h, w) if nbr in cell_set and nbr not in owner}
        frontiers.append(frontier)

    while len(owner) < len(cell_set):
        choices = [idx for idx in range(n_regions) if frontiers[idx]]
        if not choices:
            return None
        min_size = min(len(regions[idx]) for idx in choices)
        eligible = [idx for idx in choices if len(regions[idx]) <= min_size + 1]
        idx = random.choice(eligible)
        cell = random.choice(tuple(frontiers[idx]))
        frontiers[idx].discard(cell)
        if cell in owner:
            continue

        owner[cell] = idx
        regions[idx].add(cell)
        for nbr in _neighbors4(cell[0], cell[1], h, w):
            if nbr in cell_set and nbr not in owner:
                frontiers[idx].add(nbr)

    return regions


def _valid_partition(mask: np.ndarray, regions: list[set[tuple[int, int]]]) -> bool:
    row_totals = mask.sum(axis=1)
    col_totals = mask.sum(axis=0)
    for region in regions:
        if len(region) < MIN_REGION_AREA or not _cells_connected(region):
            return False
        reg_mask = np.zeros_like(mask, dtype=bool)
        for r, c in region:
            reg_mask[r, c] = True
        if np.any(reg_mask.sum(axis=1) == row_totals):
            return False
        if np.any(reg_mask.sum(axis=0) == col_totals):
            return False
    return True


def _partition_template(mask: np.ndarray, n_regions: int) -> list[set[tuple[int, int]]] | None:
    for _ in range(PARTITION_RETRIES):
        regions = _grow_partition(mask, n_regions)
        if regions is None:
            continue
        if _valid_partition(mask, regions):
            return regions
    return None


def _expected_output(mask: np.ndarray, regions: list[set[tuple[int, int]]], region_colors: list[int], bg: int) -> np.ndarray:
    out = np.full(mask.shape, bg, dtype=int)
    for color, region in zip(region_colors, regions):
        for r, c in region:
            out[r, c] = int(color)
    return out


def _render_input(
    mask: np.ndarray,
    regions: list[set[tuple[int, int]]],
    fg: int,
    marker_colors: list[int],
    bg: int,
) -> np.ndarray | None:
    h, w = mask.shape
    n_copies = len(regions)
    top_margin = random.randint(*TOP_MARGIN_RANGE)
    bottom_margin = random.randint(*BOTTOM_MARGIN_RANGE)
    left_margin = random.randint(*SIDE_MARGIN_RANGE)
    right_margin = random.randint(*SIDE_MARGIN_RANGE)
    gaps = [random.randint(*GAP_RANGE) for _ in range(n_copies - 1)]

    width = left_margin + right_margin + n_copies * w + sum(gaps)
    height = top_margin + 1 + h + bottom_margin
    if width > MAX_GRID_WIDTH:
        return None

    grid = np.full((height, width), bg, dtype=int)
    row0 = top_margin + 1
    centers: list[int] = []
    c0 = left_margin
    for idx, region in enumerate(regions):
        copy = mask.copy()
        for r, c in region:
            copy[r, c] = False
        assert np.all(copy.sum(axis=0) > 0), "each generated copy must retain every template column"

        patch = grid[row0 : row0 + h, c0 : c0 + w]
        patch[copy] = fg

        center = c0 + w // 2
        centers.append(center)
        grid[row0 - 1, center] = marker_colors[idx]
        c0 += w + (gaps[idx] if idx < len(gaps) else 0)

    return grid


def _gen_attempt() -> np.ndarray | None:
    n_copies = random.choice(COPY_COUNT_CHOICES)
    h = random.randint(*HEIGHT_RANGE)
    w = random.choice(WIDTH_CHOICES)

    colors = list(PALETTE)
    bg, fg = random.sample(colors, 2)
    marker_colors = random.sample([color for color in colors if color not in {bg, fg}], n_copies)

    mask = _build_template_mask(h, w, n_copies)
    if mask is None:
        return None
    regions = _partition_template(mask, n_copies)
    if regions is None:
        return None

    grid = _render_input(mask, regions, fg, marker_colors, bg)
    if grid is None:
        return None

    expected = _expected_output(mask, regions, marker_colors, bg)
    if not np.array_equal(main(grid), expected):
        return None
    if int(np.count_nonzero(grid == bg)) <= max(int(np.count_nonzero(grid == fg)), len(marker_colors)):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 6")
