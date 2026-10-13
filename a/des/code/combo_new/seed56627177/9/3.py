from __future__ import annotations

from collections import Counter
import random

import numpy as np


def _dominant_color(values: np.ndarray) -> int:
    counts = Counter(map(int, np.asarray(values).reshape(-1)))
    ranked = counts.most_common()
    assert ranked, "expected at least one color"
    if len(ranked) > 1:
        assert ranked[0][1] > ranked[1][1], "expected a unique dominant color"
    return int(ranked[0][0])


def _contiguous_runs(indices: list[int]) -> list[tuple[int, int]]:
    if not indices:
        return []
    runs = []
    start = prev = int(indices[0])
    for idx in indices[1:]:
        idx = int(idx)
        if idx == prev + 1:
            prev = idx
            continue
        runs.append((start, prev))
        start = prev = idx
    runs.append((start, prev))
    return runs


def _spans(length: int, runs: list[tuple[int, int]]) -> list[tuple[int, int]]:
    spans = []
    start = 0
    for r0, r1 in runs:
        assert start < r0, "separator bands must leave nonempty payload spans"
        spans.append((start, r0 - 1))
        start = r1 + 1
    assert start < length, "separator bands must leave a final payload span"
    spans.append((start, length - 1))
    return spans


def _find_partition(g: np.ndarray) -> tuple[int, list[tuple[int, int]], list[tuple[int, int]], int]:
    h, w = map(int, g.shape)
    candidates = []
    for color in map(int, np.unique(g)):
        full_rows = [r for r in range(h) if np.all(g[r, :] == color)]
        full_cols = [c for c in range(w) if np.all(g[:, c] == color)]
        if not full_rows or not full_cols:
            continue
        row_runs = _contiguous_runs(full_rows)
        col_runs = _contiguous_runs(full_cols)
        try:
            row_spans = _spans(h, row_runs)
            col_spans = _spans(w, col_runs)
        except AssertionError:
            continue
        row_sizes = {r1 - r0 + 1 for r0, r1 in row_spans}
        col_sizes = {c1 - c0 + 1 for c0, c1 in col_spans}
        if len(row_sizes) != 1 or len(col_sizes) != 1:
            continue
        cell_size = row_sizes.pop()
        if cell_size != col_sizes.pop():
            continue
        if cell_size < 3:
            continue
        score = len(full_rows) + len(full_cols)
        candidates.append((score, color, row_spans, col_spans, cell_size))

    assert candidates, "no valid separator partition found"
    candidates.sort(key=lambda item: item[0], reverse=True)
    best_score = candidates[0][0]
    best = [item for item in candidates if item[0] == best_score]
    assert len(best) == 1, "expected a unique separator color"
    _, separator, row_spans, col_spans, cell_size = best[0]
    return int(separator), row_spans, col_spans, int(cell_size)


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    assert coords.size > 0, "expected a nonempty mask"
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return mask[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1].astype(bool)


def _mask_signature(mask: np.ndarray) -> tuple[str, ...]:
    cropped = _crop_mask(mask)
    return tuple("".join("1" if cell else "0" for cell in row) for row in cropped)


def _extract_templates(legend: np.ndarray, bg: int, separator: int) -> dict[int, np.ndarray]:
    colors = [int(c) for c in np.unique(legend) if int(c) not in {bg, separator}]
    assert len(colors) >= 2, "legend must define at least two templates"

    templates: dict[int, np.ndarray] = {}
    for color in colors:
        mask = legend == color
        templates[color] = _crop_mask(mask)
    return templates


def _quadrant_rotation(seed_r: int, seed_c: int, tile_size: int) -> int:
    center = (tile_size - 1) / 2.0
    assert seed_r != center and seed_c != center, "seed must lie strictly inside one quadrant"
    if seed_r < center and seed_c < center:
        return 0
    if seed_r < center and seed_c > center:
        return 1
    if seed_r > center and seed_c > center:
        return 2
    return 3


def _centered_tile(tile_size: int, mask: np.ndarray, color: int, bg: int) -> np.ndarray:
    h, w = map(int, mask.shape)
    assert h <= tile_size and w <= tile_size, "template must fit in tile"
    out = np.full((tile_size, tile_size), int(bg), dtype=int)
    r0 = (tile_size - h) // 2
    c0 = (tile_size - w) // 2
    out[r0 : r0 + h, c0 : c0 + w][mask] = int(color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill seeded separator-grid cells with rotated color-matched templates from the legend cell.

    Input structure:
    - A dominant background color fills most cells.
    - One separator color forms complete rows and columns that partition the grid into equal square tiles.
    - The top-left tile is a legend tile containing several differently colored template shapes in their original orientation.
    - Every other tile is either pure background or contains exactly one non-background, non-separator seed pixel.

    Rule:
    - Detect the separator partition and crop one template mask per legend color from the top-left tile.
    - For each seeded tile, use the seed color to choose the matching template.
    - Use the seed pixel's quadrant relative to the tile center to choose the rotation: upper-left=0, upper-right=90 clockwise, lower-right=180, lower-left=270.
    - Replace that seeded tile with the chosen rotated template centered on background; leave the legend, empty tiles, and separator lines unchanged.

    Key insight:
    - Each seed pixel encodes two independent instructions at once: its color selects the template shape, and its position quadrant selects the template rotation.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a nonempty 2D grid"

    separator, row_spans, col_spans, tile_size = _find_partition(g)
    bg = _dominant_color(g[g != separator])

    legend_r0, legend_r1 = row_spans[0]
    legend_c0, legend_c1 = col_spans[0]
    legend = g[legend_r0 : legend_r1 + 1, legend_c0 : legend_c1 + 1]
    templates = _extract_templates(legend, bg, separator)

    out = g.copy()
    for tile_r, (r0, r1) in enumerate(row_spans):
        for tile_c, (c0, c1) in enumerate(col_spans):
            if tile_r == 0 and tile_c == 0:
                continue
            tile = g[r0 : r1 + 1, c0 : c1 + 1]
            payload = (tile != bg) & (tile != separator)
            count = int(payload.sum())
            if count == 0:
                continue
            assert count == 1, "each non-legend tile must contain exactly one seed pixel"
            seed_r, seed_c = np.argwhere(payload)[0]
            seed_r, seed_c = int(seed_r), int(seed_c)
            seed_color = int(tile[seed_r, seed_c])
            assert seed_color in templates, "seed color must exist in the legend"

            rotation = _quadrant_rotation(seed_r, seed_c, tile_size)
            rotated = np.rot90(templates[seed_color], -rotation)
            out[r0 : r1 + 1, c0 : c1 + 1] = _centered_tile(tile_size, rotated, seed_color, bg)

    return out


# --------------- generator ---------------

GEN_ATTEMPTS = 400
GEN_GRID_SHAPES = ((2, 3), (3, 2), (3, 3))
GEN_TILE_SIZE_CHOICES = (7, 9)
GEN_TEMPLATE_COUNT_RANGE = (2, 4)
GEN_TEMPLATE_PIXEL_RANGE = (3, 5)
GEN_COLOR_POOL = tuple(range(10))
GEN_SLOT_SIZE = 3


def _rotation_variants(mask: np.ndarray) -> list[tuple[str, ...]]:
    return [_mask_signature(np.rot90(mask, -k)) for k in range(4)]


def _random_connected_mask(size: int, n_cells: int) -> np.ndarray:
    start = (random.randrange(size), random.randrange(size))
    filled = {start}
    while len(filled) < n_cells:
        frontier = []
        for r, c in filled:
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                rr, cc = r + dr, c + dc
                if 0 <= rr < size and 0 <= cc < size and (rr, cc) not in filled:
                    frontier.append((rr, cc))
        if not frontier:
            break
        filled.add(random.choice(frontier))

    mask = np.zeros((size, size), dtype=bool)
    for r, c in filled:
        mask[r, c] = True
    return mask


def _random_template_mask(used_signatures: set[tuple[str, ...]]) -> tuple[np.ndarray, np.ndarray] | None:
    for _ in range(120):
        n_cells = random.randint(*GEN_TEMPLATE_PIXEL_RANGE)
        slot_mask = _random_connected_mask(GEN_SLOT_SIZE, n_cells)
        variants = _rotation_variants(slot_mask)
        if len(set(variants)) != 4:
            continue
        canonical = min(variants)
        if canonical in used_signatures:
            continue
        used_signatures.add(canonical)
        return slot_mask, _crop_mask(slot_mask)
    return None


def _slot_origins(tile_size: int) -> list[tuple[int, int]]:
    edge = tile_size - GEN_SLOT_SIZE
    return [(0, 0), (0, edge), (edge, 0), (edge, edge)]


def _quadrant_seed_position(tile_size: int, quadrant: int) -> tuple[int, int]:
    mid = tile_size // 2
    top_rows = list(range(0, max(1, mid - 1)))
    bottom_rows = list(range(mid + 2, tile_size))
    left_cols = list(range(0, max(1, mid - 1)))
    right_cols = list(range(mid + 2, tile_size))
    if not top_rows:
        top_rows = list(range(0, mid))
    if not bottom_rows:
        bottom_rows = list(range(mid + 1, tile_size))
    if not left_cols:
        left_cols = list(range(0, mid))
    if not right_cols:
        right_cols = list(range(mid + 1, tile_size))

    if quadrant == 0:
        return random.choice(top_rows), random.choice(left_cols)
    if quadrant == 1:
        return random.choice(top_rows), random.choice(right_cols)
    if quadrant == 2:
        return random.choice(bottom_rows), random.choice(right_cols)
    return random.choice(bottom_rows), random.choice(left_cols)


def _tile_origin(tile_r: int, tile_c: int, tile_size: int) -> tuple[int, int]:
    return tile_r * (tile_size + 1), tile_c * (tile_size + 1)


def _render_expected(
    grid: np.ndarray,
    assignments: list[tuple[int, int, int, int]],
    cropped_templates: dict[int, np.ndarray],
    bg: int,
    tile_size: int,
) -> np.ndarray:
    out = grid.copy()
    for tile_r, tile_c, color, quadrant in assignments:
        r0, c0 = _tile_origin(tile_r, tile_c, tile_size)
        rotated = np.rot90(cropped_templates[color], -quadrant)
        out[r0 : r0 + tile_size, c0 : c0 + tile_size] = _centered_tile(tile_size, rotated, color, bg)
    return out


def _gen_attempt() -> np.ndarray | None:
    tile_rows, tile_cols = random.choice(GEN_GRID_SHAPES)
    tile_size = random.choice(GEN_TILE_SIZE_CHOICES)
    height = tile_rows * tile_size + (tile_rows - 1)
    width = tile_cols * tile_size + (tile_cols - 1)
    if max(height, width) > 30:
        return None

    bg, separator = random.sample(GEN_COLOR_POOL, 2)
    max_templates = min(GEN_TEMPLATE_COUNT_RANGE[1], 4)
    template_count = random.randint(GEN_TEMPLATE_COUNT_RANGE[0], max_templates)
    template_colors = random.sample([c for c in GEN_COLOR_POOL if c not in {bg, separator}], template_count)

    grid = np.full((height, width), int(bg), dtype=int)
    for r in range(tile_size, height, tile_size + 1):
        grid[r, :] = int(separator)
    for c in range(tile_size, width, tile_size + 1):
        grid[:, c] = int(separator)

    used_signatures: set[tuple[str, ...]] = set()
    cropped_templates: dict[int, np.ndarray] = {}
    legend = np.full((tile_size, tile_size), int(bg), dtype=int)
    slots = _slot_origins(tile_size)
    random.shuffle(slots)
    for color, (sr, sc) in zip(template_colors, slots, strict=False):
        shape = _random_template_mask(used_signatures)
        if shape is None:
            return None
        slot_mask, cropped = shape
        legend[sr : sr + GEN_SLOT_SIZE, sc : sc + GEN_SLOT_SIZE][slot_mask] = int(color)
        cropped_templates[int(color)] = cropped

    legend_r0, legend_c0 = _tile_origin(0, 0, tile_size)
    grid[legend_r0 : legend_r0 + tile_size, legend_c0 : legend_c0 + tile_size] = legend

    target_tiles = [(tr, tc) for tr in range(tile_rows) for tc in range(tile_cols) if (tr, tc) != (0, 0)]
    max_seeds = len(target_tiles)
    min_seeds = max(4, template_count)
    if max_seeds < min_seeds:
        return None
    if max_seeds > min_seeds and random.random() < 0.7:
        seed_count = random.randint(min_seeds, max_seeds - 1)
    else:
        seed_count = max_seeds

    seed_tiles = random.sample(target_tiles, seed_count)

    colors = template_colors[:]
    while len(colors) < seed_count:
        colors.append(random.choice(template_colors))
    random.shuffle(colors)

    quadrants = [0, 1, 2, 3]
    while len(quadrants) < seed_count:
        quadrants.append(random.randrange(4))
    random.shuffle(quadrants)

    assignments = []
    for (tile_r, tile_c), color, quadrant in zip(seed_tiles, colors, quadrants, strict=False):
        local_r, local_c = _quadrant_seed_position(tile_size, quadrant)
        r0, c0 = _tile_origin(tile_r, tile_c, tile_size)
        grid[r0 + local_r, c0 + local_c] = int(color)
        assignments.append((tile_r, tile_c, int(color), int(quadrant)))

    expected = _render_expected(grid, assignments, cropped_templates, int(bg), tile_size)
    actual = main(grid)
    if not np.array_equal(actual, expected):
        return None

    separator2, row_spans, col_spans, cell_size2 = _find_partition(grid)
    if separator2 != separator or cell_size2 != tile_size:
        return None
    if len(row_spans) != tile_rows or len(col_spans) != tile_cols:
        return None
    if np.array_equal(grid, expected):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid separator-template ARC input")
