from __future__ import annotations

import random
from collections import Counter

import numpy as np


BACKGROUND_COLOR = 0
SEPARATOR_COLOR = 5
SEED_COLOR = 2
TEMPLATE_COLOR_POOL = (1, 3, 4, 6, 7, 8, 9)
MIN_TEMPLATE_COUNT = 3
MAX_TEMPLATE_COUNT = 5
TILE_SIDE_MIN = 3
TILE_SIDE_MAX = 6
MAX_GENERATION_ATTEMPTS = 300


def _spans(length: int, separators: list[int]) -> list[tuple[int, int]]:
    """Return half-open spans between one-cell separator lines."""
    spans: list[tuple[int, int]] = []
    start = 0
    for sep in sorted(int(x) for x in separators):
        assert start < sep < length, "separator lines must leave non-empty regions"
        spans.append((start, sep))
        start = sep + 1
    assert start < length, "separator lines must leave a final non-empty region"
    spans.append((start, length))
    return spans


def _separator_layout(grid: np.ndarray) -> tuple[int, list[tuple[int, int]], list[tuple[int, int]]]:
    g = np.asarray(grid, dtype=int)
    height, width = map(int, g.shape)

    solid_rows: dict[int, list[int]] = {}
    for r in range(height):
        if bool(np.all(g[r] == g[r, 0])):
            solid_rows.setdefault(int(g[r, 0]), []).append(int(r))

    solid_cols: dict[int, list[int]] = {}
    for c in range(width):
        if bool(np.all(g[:, c] == g[0, c])):
            solid_cols.setdefault(int(g[0, c]), []).append(int(c))

    valid_layouts = []
    for color in sorted(set(solid_rows) & set(solid_cols)):
        try:
            row_spans = _spans(height, solid_rows[color])
            col_spans = _spans(width, solid_cols[color])
        except AssertionError:
            continue
        if len(row_spans) < 2 or len(col_spans) < 1:
            continue
        row_heights = {r1 - r0 for r0, r1 in row_spans}
        col_widths = {c1 - c0 for c0, c1 in col_spans}
        if len(row_heights) == 1 and len(col_widths) == 1:
            valid_layouts.append((int(color), row_spans, col_spans))

    assert len(valid_layouts) == 1, "expected one equal-tile separator layout"
    return valid_layouts[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Replace each seeded subregion with the bottom template indexed by its seed count.

    Input structure:
    - The background is black.
    - Full grey separator rows and columns divide the grid into equal rectangular tiles.
    - The bottom tile row is a template strip: each tile contains a colored pattern on black.
    - Every tile above the template strip contains only black background plus some red single-pixel seeds.

    Rule:
    - Identify the equal-sized tile layout from the full grey separator lines.
    - Count the red seed pixels in each non-template tile.
    - Select the bottom template whose left-to-right index matches that count, and copy that full template tile into the region.
    - Keep the separator lines and the bottom template strip unchanged.

    Key insight:
    - The bottom row is a lookup table, and only the number of seed pixels matters; their positions are irrelevant noise.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    separator, row_spans, col_spans = _separator_layout(g)
    template_r0, template_r1 = row_spans[-1]
    data_row_spans = row_spans[:-1]
    assert data_row_spans, "expected at least one seeded tile row above the templates"

    tile_h = template_r1 - template_r0
    tile_w = col_spans[0][1] - col_spans[0][0]

    templates = [g[template_r0:template_r1, c0:c1].copy() for c0, c1 in col_spans]
    template_count = len(templates)
    assert template_count >= 1, "expected at least one template"
    assert all(tile.shape == (tile_h, tile_w) for tile in templates), "template sizes must match tile size"

    data_values: list[int] = []
    for r0, r1 in data_row_spans:
        for c0, c1 in col_spans:
            data_values.extend(int(x) for x in g[r0:r1, c0:c1].reshape(-1))
    counts = Counter(data_values)
    assert counts, "expected non-empty seeded tile area"
    background = int(counts.most_common(1)[0][0])
    seed_colors = [color for color in counts if color != background]
    assert len(seed_colors) == 1, "expected exactly one seed color in the non-template tiles"
    seed_color = int(seed_colors[0])
    assert separator not in {background, seed_color}, "separator must differ from seeded tile colors"

    out = g.copy()
    for r0, r1 in data_row_spans:
        for c0, c1 in col_spans:
            tile = g[r0:r1, c0:c1]
            tile_colors = {int(x) for x in np.unique(tile)}
            assert tile_colors.issubset({background, seed_color}), "seeded tiles may only contain background and seeds"
            seed_count = int(np.sum(tile == seed_color))
            assert 1 <= seed_count <= template_count, "seed count must index one template"
            out[r0:r1, c0:c1] = templates[seed_count - 1]

    return out


def _add_mask(pool: dict[bytes, np.ndarray], mask: np.ndarray) -> None:
    mask = np.asarray(mask, dtype=bool)
    filled = int(mask.sum())
    if 1 < filled < mask.size:
        pool[mask.tobytes()] = mask.copy()


def _template_masks(height: int, width: int) -> list[np.ndarray]:
    r_idx, c_idx = np.indices((height, width))
    cy = height // 2
    cx = width // 2
    pool: dict[bytes, np.ndarray] = {}

    diag = np.zeros((height, width), dtype=bool)
    for i in range(min(height, width)):
        diag[i, i] = True
    _add_mask(pool, diag)

    anti = np.zeros((height, width), dtype=bool)
    for i in range(min(height, width)):
        anti[i, width - 1 - i] = True
    _add_mask(pool, anti)
    _add_mask(pool, diag | anti)

    plus = np.zeros((height, width), dtype=bool)
    plus[cy, :] = True
    plus[:, cx] = True
    _add_mask(pool, plus)

    border = np.zeros((height, width), dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True
    _add_mask(pool, border)

    _add_mask(pool, (r_idx + c_idx) % 2 == 0)
    _add_mask(pool, (r_idx + c_idx) % 2 == 1)

    _add_mask(pool, c_idx % 2 == 0)
    _add_mask(pool, c_idx % 2 == 1)
    _add_mask(pool, r_idx % 2 == 0)
    _add_mask(pool, r_idx % 2 == 1)

    top_left = np.zeros((height, width), dtype=bool)
    top_left[0, :] = True
    top_left[:, 0] = True
    _add_mask(pool, top_left)

    top_right = np.zeros((height, width), dtype=bool)
    top_right[0, :] = True
    top_right[:, -1] = True
    _add_mask(pool, top_right)

    bottom_left = np.zeros((height, width), dtype=bool)
    bottom_left[-1, :] = True
    bottom_left[:, 0] = True
    _add_mask(pool, bottom_left)

    bottom_right = np.zeros((height, width), dtype=bool)
    bottom_right[-1, :] = True
    bottom_right[:, -1] = True
    _add_mask(pool, bottom_right)

    t_up = np.zeros((height, width), dtype=bool)
    t_up[0, :] = True
    t_up[:, cx] = True
    _add_mask(pool, t_up)

    t_down = np.zeros((height, width), dtype=bool)
    t_down[-1, :] = True
    t_down[:, cx] = True
    _add_mask(pool, t_down)

    t_left = np.zeros((height, width), dtype=bool)
    t_left[:, 0] = True
    t_left[cy, :] = True
    _add_mask(pool, t_left)

    t_right = np.zeros((height, width), dtype=bool)
    t_right[:, -1] = True
    t_right[cy, :] = True
    _add_mask(pool, t_right)

    corners = np.zeros((height, width), dtype=bool)
    corners[0, 0] = True
    corners[0, -1] = True
    corners[-1, 0] = True
    corners[-1, -1] = True
    _add_mask(pool, corners)

    center_block = np.zeros((height, width), dtype=bool)
    r0 = max(0, cy - 1)
    r1 = min(height, cy + 2)
    c0 = max(0, cx - 1)
    c1 = min(width, cx + 2)
    center_block[r0:r1, c0:c1] = True
    _add_mask(pool, center_block)

    radius = max(1.0, min(height, width) / 2.0 - 0.5)
    diamond = (np.abs(r_idx - (height - 1) / 2.0) + np.abs(c_idx - (width - 1) / 2.0)) <= radius
    _add_mask(pool, diamond)

    return list(pool.values())


def _expected_output(
    grid: np.ndarray,
    tile_height: int,
    tile_width: int,
    templates: list[np.ndarray],
    counts_by_col: list[int],
) -> np.ndarray:
    out = grid.copy()
    for col_idx, seed_count in enumerate(counts_by_col):
        c0 = col_idx * (tile_width + 1)
        out[0:tile_height, c0 : c0 + tile_width] = templates[seed_count - 1]
    return out


def _generate_attempt(rng: random.Random) -> np.ndarray | None:
    tile_height = rng.randint(TILE_SIDE_MIN, TILE_SIDE_MAX)
    tile_width = rng.randint(TILE_SIDE_MIN, TILE_SIDE_MAX)

    max_by_width = min(MAX_TEMPLATE_COUNT, 31 // (tile_width + 1))
    max_by_area = min(MAX_TEMPLATE_COUNT, tile_height * tile_width - 1)
    template_max = min(max_by_width, max_by_area)
    if template_max < MIN_TEMPLATE_COUNT:
        return None
    template_count = rng.randint(MIN_TEMPLATE_COUNT, template_max)

    mask_pool = _template_masks(tile_height, tile_width)
    if len(mask_pool) < template_count:
        return None
    rng.shuffle(mask_pool)
    chosen_masks = mask_pool[:template_count]

    template_colors = rng.sample(list(TEMPLATE_COLOR_POOL), template_count)
    templates: list[np.ndarray] = []
    for mask, color in zip(chosen_masks, template_colors):
        tile = np.full((tile_height, tile_width), BACKGROUND_COLOR, dtype=int)
        tile[mask] = int(color)
        templates.append(tile)

    counts_by_col = list(range(1, template_count + 1))
    rng.shuffle(counts_by_col)

    height = 2 * tile_height + 1
    width = template_count * tile_width + (template_count - 1)
    if max(height, width) > 30:
        return None

    grid = np.full((height, width), BACKGROUND_COLOR, dtype=int)
    grid[tile_height, :] = SEPARATOR_COLOR
    for col_idx in range(template_count - 1):
        sep_col = (col_idx + 1) * tile_width + col_idx
        grid[:, sep_col] = SEPARATOR_COLOR

    for col_idx, tile in enumerate(templates):
        c0 = col_idx * (tile_width + 1)
        grid[tile_height + 1 :, c0 : c0 + tile_width] = tile

    for col_idx, seed_count in enumerate(counts_by_col):
        c0 = col_idx * (tile_width + 1)
        positions = rng.sample(range(tile_height * tile_width), seed_count)
        for pos in positions:
            rr, cc = divmod(pos, tile_width)
            grid[rr, c0 + cc] = SEED_COLOR

    expected = _expected_output(grid, tile_height, tile_width, templates, counts_by_col)
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generate_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
