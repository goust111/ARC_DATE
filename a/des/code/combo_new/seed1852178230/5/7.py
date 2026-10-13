from __future__ import annotations

import random
from collections import defaultdict

import numpy as np


BLACK = 0
MAX_GENERATION_ATTEMPTS = 400


def _neighbors4(r: int, c: int, h: int, w: int) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    if r > 0:
        out.append((r - 1, c))
    if r + 1 < h:
        out.append((r + 1, c))
    if c > 0:
        out.append((r, c - 1))
    if c + 1 < w:
        out.append((r, c + 1))
    return out


def _label_components(mask: np.ndarray) -> tuple[np.ndarray, list[list[tuple[int, int]]]]:
    h, w = mask.shape
    labels = np.zeros((h, w), dtype=int)
    components: list[list[tuple[int, int]]] = []
    next_id = 1

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or labels[r, c] != 0:
                continue
            stack = [(r, c)]
            labels[r, c] = next_id
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc, h, w):
                    if mask[nr, nc] and labels[nr, nc] == 0:
                        labels[nr, nc] = next_id
                        stack.append((nr, nc))
            components.append(coords)
            next_id += 1

    return labels, components


def _detect_palette(grid: np.ndarray) -> tuple[int, list[int]]:
    h, w = grid.shape
    best: tuple[tuple[int, int], int, list[int]] | None = None
    max_band_width = min(4, w - 1)

    for band_width in range(1, max_band_width + 1):
        start = w - band_width
        sep_col = start - 1
        if sep_col < 0 or not np.all(grid[:, sep_col] == BLACK):
            continue
        if not np.any(grid[:, :sep_col] != BLACK):
            continue

        band = grid[:, start:]
        labels, comps = _label_components(band != BLACK)
        if len(comps) < 2:
            continue

        blocks: list[tuple[int, int, int]] = []
        ok = True
        for comp in comps:
            rows = [r for r, _ in comp]
            cols = [c for _, c in comp]
            if min(cols) != 0 or max(cols) != band_width - 1:
                ok = False
                break
            colors = {int(band[r, c]) for r, c in comp}
            if len(colors) != 1:
                ok = False
                break
            blocks.append((min(rows), max(rows), int(next(iter(colors)))))

        if not ok:
            continue

        blocks.sort()
        for i in range(len(blocks) - 1):
            if blocks[i][1] + 1 >= blocks[i + 1][0]:
                ok = False
                break
        if not ok:
            continue

        palette_colors = [color for _r0, _r1, color in blocks]
        score = (len(palette_colors), -band_width)
        if best is None or score > best[0]:
            best = (score, start, palette_colors)

    assert best is not None, "failed to detect the right-side palette"
    return best[1], best[2]


def _analyze_map(map_grid: np.ndarray) -> tuple[np.ndarray, dict[int, int], dict[int, int], list[tuple[int, list[int]]]]:
    labels, comps = _label_components(map_grid != BLACK)
    assert comps, "expected at least one enclosed region"

    areas = {idx: len(comp) for idx, comp in enumerate(comps, start=1)}
    adjacency = {idx: set() for idx in range(1, len(comps) + 1)}
    h, w = map_grid.shape

    for r in range(h):
        for c in range(w):
            if map_grid[r, c] != BLACK:
                continue
            if 0 < c < w - 1:
                left = int(labels[r, c - 1])
                right = int(labels[r, c + 1])
                if left and right and left != right:
                    adjacency[left].add(right)
                    adjacency[right].add(left)
            if 0 < r < h - 1:
                up = int(labels[r - 1, c])
                down = int(labels[r + 1, c])
                if up and down and up != down:
                    adjacency[up].add(down)
                    adjacency[down].add(up)

    degrees = {idx: len(neighbors) for idx, neighbors in adjacency.items()}
    groups: dict[int, list[int]] = defaultdict(list)
    for region_id, degree in degrees.items():
        groups[degree].append(region_id)

    ordered = sorted(
        groups.items(),
        key=lambda item: (-max(areas[region_id] for region_id in item[1]), item[0]),
    )
    return labels, areas, degrees, ordered


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor map regions by grouping them according to adjacency degree and ordering those groups by area.

    Input structure:
    - Black grid lines partition the main map into enclosed non-black regions.
    - The regions' input colors are arbitrary distractors and may repeat across different regions.
    - A narrow palette sits on the right side as a vertical stack of separated color blocks.

    Rule:
    - Ignore the regions' original colors and identify every enclosed region in the map area.
    - Compute each region's adjacency degree: how many other regions share a black wall segment with it.
    - Group regions by equal adjacency degree, then sort the degree-groups by the largest region area inside each group, from largest to smallest.
    - Refill every region in the first group with the top palette color, the second group with the next palette color, and so on.
    - Keep all black grid lines and the right-side palette unchanged.

    Key insight:
    The visible region colors are noise; the real label of each region is its position inside the hidden region-adjacency graph, and the palette order is unlocked only after ranking degree-groups by their largest member.
    """
    g = np.asarray(grid, dtype=int)
    palette_start, palette_colors = _detect_palette(g)
    map_grid = g[:, :palette_start]
    labels, areas, _degrees, ordered_groups = _analyze_map(map_grid)

    assert len(ordered_groups) <= len(palette_colors), "palette must provide enough colors"

    out = g.copy()
    for palette_color, (_degree, region_ids) in zip(palette_colors, ordered_groups, strict=False):
        for region_id in region_ids:
            out[:, :palette_start][labels == region_id] = int(palette_color)

    return out


def _grow_partition(rows: int, cols: int, n_regions: int) -> np.ndarray | None:
    all_cells = [(r, c) for r in range(rows) for c in range(cols)]
    seeds = random.sample(all_cells, n_regions)
    owner = np.full((rows, cols), -1, dtype=int)
    frontiers = [set() for _ in range(n_regions)]
    sizes = [1] * n_regions

    def add_frontier(region_id: int, r: int, c: int) -> None:
        for nr, nc in _neighbors4(r, c, rows, cols):
            if owner[nr, nc] == -1:
                frontiers[region_id].add((nr, nc))

    for region_id, (r, c) in enumerate(seeds):
        owner[r, c] = region_id
    for region_id, (r, c) in enumerate(seeds):
        add_frontier(region_id, r, c)

    unassigned = rows * cols - n_regions
    while unassigned > 0:
        active = [region_id for region_id in range(n_regions) if frontiers[region_id]]
        if not active:
            return None

        if random.random() < 0.55:
            smallest = min(sizes[region_id] for region_id in active)
            choices = [region_id for region_id in active if sizes[region_id] == smallest]
            region_id = random.choice(choices)
        else:
            region_id = random.choice(active)

        cell = random.choice(tuple(frontiers[region_id]))
        frontiers[region_id].discard(cell)
        r, c = cell
        if owner[r, c] != -1:
            continue

        owner[r, c] = region_id
        sizes[region_id] += 1
        unassigned -= 1
        add_frontier(region_id, r, c)

    return owner


def _rasterize_partition(partition: np.ndarray, region_colors: dict[int, int], cell_size: int) -> np.ndarray:
    rows, cols = partition.shape
    h = rows * cell_size + rows + 1
    w = cols * cell_size + cols + 1
    grid = np.zeros((h, w), dtype=int)

    for r in range(rows):
        for c in range(cols):
            color = int(region_colors[int(partition[r, c])])
            r0 = 1 + r * (cell_size + 1)
            c0 = 1 + c * (cell_size + 1)
            grid[r0 : r0 + cell_size, c0 : c0 + cell_size] = color

    for r in range(rows):
        for c in range(cols - 1):
            left_id = int(partition[r, c])
            right_id = int(partition[r, c + 1])
            if left_id == right_id:
                r0 = 1 + r * (cell_size + 1)
                wall_col = (c + 1) * (cell_size + 1)
                grid[r0 : r0 + cell_size, wall_col] = int(region_colors[left_id])

    for r in range(rows - 1):
        for c in range(cols):
            top_id = int(partition[r, c])
            bottom_id = int(partition[r + 1, c])
            if top_id == bottom_id:
                wall_row = (r + 1) * (cell_size + 1)
                c0 = 1 + c * (cell_size + 1)
                grid[wall_row, c0 : c0 + cell_size] = int(region_colors[top_id])

    for r in range(1, rows):
        for c in range(1, cols):
            ids = {
                int(partition[r - 1, c - 1]),
                int(partition[r - 1, c]),
                int(partition[r, c - 1]),
                int(partition[r, c]),
            }
            if len(ids) == 1:
                grid[r * (cell_size + 1), c * (cell_size + 1)] = int(region_colors[next(iter(ids))])

    return grid


def _build_palette(height: int, palette_colors: list[int], palette_width: int) -> np.ndarray | None:
    n = len(palette_colors)
    min_needed = n + max(0, n - 1)
    if height < min_needed:
        return None

    band = np.zeros((height, palette_width), dtype=int)
    heights = [1] * n
    extra = height - min_needed

    for _ in range(extra):
        idx = random.randrange(2 * n + 1)
        if idx < n and heights[idx] < 3:
            heights[idx] += 1

    used = sum(heights) + (n - 1)
    free_rows = height - used
    margins = [0] * (n + 1)
    for _ in range(free_rows):
        margins[random.randrange(n + 1)] += 1

    row = margins[0]
    for i, color in enumerate(palette_colors):
        band[row : row + heights[i], :] = int(color)
        row += heights[i]
        if i + 1 < n:
            row += 1 + margins[i + 1]
    return band


def _expected_output(inp: np.ndarray, palette_start: int, labels: np.ndarray, ordered_groups: list[tuple[int, list[int]]], palette_colors: list[int]) -> np.ndarray:
    out = inp.copy()
    for color, (_degree, region_ids) in zip(palette_colors, ordered_groups, strict=False):
        for region_id in region_ids:
            out[:, :palette_start][labels == region_id] = int(color)
    return out


def _gen_attempt() -> np.ndarray | None:
    rows = random.randint(4, 6)
    cols = random.randint(4, 6)
    cell_size = random.randint(2, 3)
    n_regions = random.randint(5, min(8, rows * cols - 1))
    partition = _grow_partition(rows, cols, n_regions)
    if partition is None:
        return None

    provisional_colors = {region_id: 1 + (region_id % 9) for region_id in range(n_regions)}
    map_grid = _rasterize_partition(partition, provisional_colors, cell_size)
    labels, areas, _degrees, ordered_groups = _analyze_map(map_grid)

    group_count = len(ordered_groups)
    if group_count < 2 or group_count > 5:
        return None
    if all(len(region_ids) == 1 for _degree, region_ids in ordered_groups):
        return None

    group_maxima = [max(areas[region_id] for region_id in region_ids) for _degree, region_ids in ordered_groups]
    if len(set(group_maxima)) != len(group_maxima):
        return None

    available_colors = list(range(1, 10))
    random.shuffle(available_colors)
    palette_colors = available_colors[:group_count]
    remaining = available_colors[group_count:]

    pool_size = random.randint(2, min(4, len(remaining)))
    input_pool = remaining[:pool_size]
    region_fill_colors = [random.choice(input_pool) for _ in range(n_regions)]
    if len(set(region_fill_colors)) < 2:
        return None
    if len(set(region_fill_colors)) == n_regions:
        return None

    region_colors = {region_id: region_fill_colors[region_id] for region_id in range(n_regions)}
    map_grid = _rasterize_partition(partition, region_colors, cell_size)
    labels, areas, _degrees, ordered_groups = _analyze_map(map_grid)

    group_maxima = [max(areas[region_id] for region_id in region_ids) for _degree, region_ids in ordered_groups]
    if len(set(group_maxima)) != len(group_maxima):
        return None

    palette_width = random.randint(2, 3)
    palette = _build_palette(map_grid.shape[0], palette_colors, palette_width)
    if palette is None:
        return None

    inp = np.concatenate([map_grid, palette], axis=1)
    if inp.shape[0] > 30 or inp.shape[1] > 30:
        return None

    try:
        palette_start, detected_palette = _detect_palette(inp)
    except AssertionError:
        return None
    if palette_start != map_grid.shape[1]:
        return None
    if detected_palette[:group_count] != palette_colors:
        return None

    expected = _expected_output(inp, palette_start, labels, ordered_groups, palette_colors)
    try:
        actual = main(inp)
    except AssertionError:
        return None

    if np.array_equal(actual, inp):
        return None
    if not np.array_equal(actual, expected):
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid ARC input after repeated attempts")
