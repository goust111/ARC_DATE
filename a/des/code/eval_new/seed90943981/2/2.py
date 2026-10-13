from collections import Counter, deque
import random

import numpy as np


WALL = 9
GEN_RETRIES = 400
SIZE_CHOICES = (3, 4)
SMALL_GRID_RANGE = (5, 6)
LARGE_GRID_RANGE = (4, 5)
TEMPLATE_COUNT_RANGE = (2, 4)
TARGET_REGION_COUNT_RANGE = (4, 9)

_PATTERN_LIBRARY = {
    3: [
        np.array([[1, 0, 0], [1, 1, 0], [0, 1, 1]], dtype=int),
        np.array([[1, 1, 1], [0, 1, 0], [0, 1, 1]], dtype=int),
        np.array([[0, 1, 1], [1, 1, 0], [1, 0, 0]], dtype=int),
        np.array([[1, 0, 1], [1, 1, 0], [0, 1, 0]], dtype=int),
        np.array([[0, 1, 0], [1, 1, 1], [0, 0, 1]], dtype=int),
    ],
    4: [
        np.array([[1, 0, 0, 0], [1, 1, 1, 0], [0, 1, 1, 1], [0, 0, 1, 0]], dtype=int),
        np.array([[0, 1, 0, 0], [1, 1, 1, 0], [0, 1, 1, 1], [0, 0, 0, 1]], dtype=int),
        np.array([[1, 1, 0, 0], [0, 1, 1, 0], [1, 1, 1, 1], [0, 0, 1, 0]], dtype=int),
        np.array([[0, 0, 1, 0], [1, 1, 1, 0], [0, 1, 1, 1], [1, 0, 0, 0]], dtype=int),
    ],
}


def _most_common(values: np.ndarray) -> int:
    counts = Counter(map(int, values.reshape(-1)))
    best = max(counts.values())
    winners = [value for value, count in counts.items() if count == best]
    assert len(winners) == 1, "expected a unique most common color"
    return int(winners[0])


def _border_color(grid: np.ndarray) -> int:
    border = np.concatenate((grid[0, :], grid[-1, :], grid[1:-1, 0], grid[1:-1, -1]))
    return _most_common(border)


def _neighbors4(r: int, c: int, h: int, w: int):
    if r > 0:
        yield r - 1, c
    if r + 1 < h:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < w:
        yield r, c + 1


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = map(int, mask.shape)
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((int(rr), int(cc)))
                for nr, nc in _neighbors4(int(rr), int(cc), h, w):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((int(nr), int(nc)))
            comps.append(coords)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _c in coords]
    cols = [c for _r, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _template_anchor(shape: tuple[int, int]) -> tuple[int, int]:
    return shape[0] // 2, shape[1] // 2


def main(grid: np.ndarray) -> np.ndarray:
    """
    Tile each seeded partition with its same-color legend template while clipping to the wall-bounded region.

    Input structure:
    - The background is the most common color.
    - Maroon walls, including an outer border, split the grid into closed non-wall regions.
    - In one corner, several small rectangular legend boxes each contain a 3x3 or 4x4 single-color template pattern.
    - Every non-legend region contains exactly one seed pixel whose color matches one legend template.

    Rule:
    - Keep the wall layout and the legend boxes unchanged.
    - For each seeded non-legend region, use the seed color to select the matching legend template.
    - Align the template's center anchor with the seed pixel, then repeat the template periodically in both directions.
    - Paint only inside that region; cells where the tiled template is background stay background.

    Key insight:
    - Color chooses the template, but the seed position fixes the tiling phase, so the same template can land differently in different regions.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _most_common(g)
    wall = _border_color(g)
    assert wall != bg, "border walls must differ from the background"

    templates: dict[int, np.ndarray] = {}
    targets: list[tuple[list[tuple[int, int]], int, tuple[int, int]]] = []

    for coords in _components(g != wall):
        colors = [int(g[r, c]) for r, c in coords if int(g[r, c]) != bg]
        assert colors, "every region must contain at least one non-background marker or template"
        palette = set(colors)
        assert len(palette) == 1, "each region must encode exactly one non-background color"
        color = int(next(iter(palette)))

        color_coords = [(r, c) for r, c in coords if int(g[r, c]) == color]
        if len(color_coords) == 1:
            targets.append((coords, color, color_coords[0]))
            continue

        r0, c0, r1, c1 = _bbox(coords)
        h, w = int(r1 - r0 + 1), int(c1 - c0 + 1)
        assert len(coords) == h * w, "legend templates must live in rectangular subframes"
        pattern = np.full((h, w), bg, dtype=int)
        for r, c in color_coords:
            pattern[r - r0, c - c0] = color
        anchor_r, anchor_c = _template_anchor(pattern.shape)
        assert int(pattern[anchor_r, anchor_c]) == color, "template anchor must be colored"
        assert color not in templates, "template colors must be unique"
        templates[color] = pattern

    assert templates, "expected at least one legend template"
    assert targets, "expected at least one seeded target region"

    out = g.copy()
    for coords, color, seed in targets:
        assert color in templates, "seed color must match a legend template"
        pattern = templates[color]
        ph, pw = map(int, pattern.shape)
        anchor_r, anchor_c = _template_anchor(pattern.shape)
        sr, sc = seed
        for r, c in coords:
            pr = (int(r) - int(sr) + anchor_r) % ph
            pc = (int(c) - int(sc) + anchor_c) % pw
            out[int(r), int(c)] = int(pattern[pr, pc])
    return out


def _legend_layout(template_count: int) -> list[tuple[int, int]]:
    if template_count == 2:
        return random.choice(
            [
                [(0, 0), (0, 1)],
                [(0, 0), (1, 0)],
            ]
        )
    if template_count == 3:
        return random.choice(
            [
                [(0, 0), (0, 1), (0, 2)],
                [(0, 0), (1, 0), (2, 0)],
            ]
        )
    assert template_count == 4
    return [(0, 0), (0, 1), (1, 0), (1, 1)]


def _sample_pattern(unit: int) -> np.ndarray:
    mask = random.choice(_PATTERN_LIBRARY[unit]).copy()
    mask = np.rot90(mask, random.randrange(4))
    if random.choice([False, True]):
        mask = np.fliplr(mask)
    if random.choice([False, True]):
        mask = np.flipud(mask)
    anchor_r, anchor_c = _template_anchor(mask.shape)
    assert int(mask[anchor_r, anchor_c]) == 1
    assert int(mask.sum()) >= 3
    assert int(mask.sum()) < mask.size
    return mask


def _partition_cells(cells: list[tuple[int, int]], n_rows: int, n_cols: int) -> list[set[tuple[int, int]]]:
    unassigned = set(cells)
    groups: list[set[tuple[int, int]]] = []
    size_choices = [1, 2, 2, 3, 3, 4]

    while unassigned:
        start = random.choice(tuple(unassigned))
        target_size = random.choice([s for s in size_choices if s <= len(unassigned)])
        group = {start}
        unassigned.remove(start)

        while len(group) < target_size:
            frontier = []
            for r, c in group:
                for nr, nc in _neighbors4(int(r), int(c), n_rows, n_cols):
                    cell = (int(nr), int(nc))
                    if cell in unassigned and cell not in frontier:
                        frontier.append(cell)
            if not frontier:
                break
            nxt = random.choice(frontier)
            group.add(nxt)
            unassigned.remove(nxt)
        groups.append(group)
    return groups


def _cell_origin(cell_r: int, cell_c: int, unit: int) -> tuple[int, int]:
    step = unit + 1
    return 1 + cell_r * step, 1 + cell_c * step


def _render_partition(
    unit: int,
    n_rows: int,
    n_cols: int,
    legend_cells: list[tuple[int, int]],
    groups: list[set[tuple[int, int]]],
    bg: int,
) -> np.ndarray:
    h = n_rows * unit + (n_rows + 1)
    w = n_cols * unit + (n_cols + 1)
    grid = np.full((h, w), WALL, dtype=int)

    for cell_r in range(n_rows):
        for cell_c in range(n_cols):
            r0, c0 = _cell_origin(cell_r, cell_c, unit)
            grid[r0 : r0 + unit, c0 : c0 + unit] = bg

    owner: dict[tuple[int, int], int] = {}
    for group_id, group in enumerate(groups):
        for cell in group:
            owner[cell] = group_id

    legend_set = set(legend_cells)
    for cell_r in range(n_rows):
        for cell_c in range(n_cols):
            cell = (cell_r, cell_c)
            if cell in legend_set:
                continue
            r0, c0 = _cell_origin(cell_r, cell_c, unit)
            if cell_r + 1 < n_rows:
                down = (cell_r + 1, cell_c)
                if down not in legend_set and owner[cell] == owner[down]:
                    grid[r0 + unit, c0 : c0 + unit] = bg
            if cell_c + 1 < n_cols:
                right = (cell_r, cell_c + 1)
                if right not in legend_set and owner[cell] == owner[right]:
                    grid[r0 : r0 + unit, c0 + unit] = bg
    return grid


def _color_legend(
    grid: np.ndarray,
    unit: int,
    legend_cells: list[tuple[int, int]],
    template_colors: list[int],
    bg: int,
) -> dict[int, np.ndarray]:
    templates: dict[int, np.ndarray] = {}
    for cell, color in zip(legend_cells, template_colors, strict=False):
        mask = _sample_pattern(unit)
        pattern = np.where(mask == 1, int(color), int(bg))
        cell_r, cell_c = cell
        r0, c0 = _cell_origin(cell_r, cell_c, unit)
        grid[r0 : r0 + unit, c0 : c0 + unit] = pattern
        templates[int(color)] = pattern
    return templates


def _empty_regions(grid: np.ndarray, bg: int) -> list[list[tuple[int, int]]]:
    regions = []
    for coords in _components(grid != WALL):
        if all(int(grid[r, c]) == bg for r, c in coords):
            regions.append(coords)
    return regions


def _unique_dominant_bg(grid: np.ndarray, bg: int) -> bool:
    counts = Counter(map(int, grid.reshape(-1)))
    best = max(counts.values())
    winners = [color for color, count in counts.items() if count == best]
    return winners == [int(bg)]


def _gen_attempt() -> np.ndarray | None:
    unit = random.choice(SIZE_CHOICES)
    if unit == 3:
        n_rows = random.randint(*SMALL_GRID_RANGE)
        n_cols = random.randint(*SMALL_GRID_RANGE)
    else:
        n_rows = random.randint(*LARGE_GRID_RANGE)
        n_cols = random.randint(*LARGE_GRID_RANGE)

    template_count = random.randint(*TEMPLATE_COUNT_RANGE)
    legend_cells = _legend_layout(template_count)
    if max(r for r, _c in legend_cells) >= n_rows or max(c for _r, c in legend_cells) >= n_cols:
        return None

    bg_choices = [color for color in range(10) if color != WALL]
    bg = random.choice(bg_choices)
    template_pool = [color for color in range(10) if color not in (bg, WALL)]
    if len(template_pool) < template_count:
        return None
    template_colors = random.sample(template_pool, template_count)

    all_cells = [(r, c) for r in range(n_rows) for c in range(n_cols)]
    legend_set = set(legend_cells)
    target_cells = [cell for cell in all_cells if cell not in legend_set]
    if len(target_cells) < 8:
        return None

    groups = _partition_cells(target_cells, n_rows, n_cols)
    if not any(len(group) > 1 for group in groups):
        return None

    region_count = len(groups)
    lo, hi = TARGET_REGION_COUNT_RANGE
    if not (lo <= region_count <= hi):
        return None
    if region_count < template_count + 1:
        return None

    grid = _render_partition(unit, n_rows, n_cols, legend_cells, groups, bg)
    _color_legend(grid, unit, legend_cells, template_colors, bg)

    regions = _empty_regions(grid, bg)
    if len(regions) != region_count:
        return None

    color_jobs = template_colors.copy()
    while len(color_jobs) < len(regions):
        color_jobs.append(random.choice(template_colors))
    random.shuffle(color_jobs)

    for coords, color in zip(regions, color_jobs, strict=False):
        seed = random.choice(coords)
        grid[seed] = int(color)

    if not _unique_dominant_bg(grid, bg):
        return None

    try:
        out = main(grid)
    except AssertionError:
        return None

    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
