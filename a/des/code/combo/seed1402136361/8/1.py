from __future__ import annotations

from collections import Counter
import random

import numpy as np


MAX_GENERATION_ATTEMPTS = 400
MAX_GRID_SIDE = 30
COLOR_POOL = tuple(range(10))


def _dominant_color(grid: np.ndarray) -> int:
    colors, counts = np.unique(grid, return_counts=True)
    return int(colors[np.argmax(counts)])


def _content_bands(separator_mask: np.ndarray) -> list[tuple[int, int]]:
    bands: list[tuple[int, int]] = []
    start: int | None = None
    for idx, is_separator in enumerate(separator_mask.tolist()):
        if is_separator:
            if start is not None:
                bands.append((start, idx))
                start = None
        elif start is None:
            start = idx
    if start is not None:
        bands.append((start, len(separator_mask)))
    return bands


def _unique_mode(values: np.ndarray) -> int:
    counts = Counter(map(int, values.reshape(-1)))
    ranked = counts.most_common()
    assert ranked, "expected at least one value"
    if len(ranked) > 1:
        assert ranked[0][1] > ranked[1][1], "consensus color must be unique"
    return int(ranked[0][0])


def _extract_tiles(grid: np.ndarray, bg: int) -> np.ndarray:
    row_bands = _content_bands(np.all(grid == bg, axis=1))
    col_bands = _content_bands(np.all(grid == bg, axis=0))
    assert row_bands and col_bands, "expected separator-delimited tiles"

    heights = {int(r1 - r0) for r0, r1 in row_bands}
    widths = {int(c1 - c0) for c0, c1 in col_bands}
    assert len(heights) == 1 and len(widths) == 1, "tiles must share one size"

    tile_h = heights.pop()
    tile_w = widths.pop()
    assert tile_h > 0 and tile_w > 0
    assert len(row_bands) * len(col_bands) >= 3, "need enough copies for majority voting"

    tiles = np.empty((len(row_bands), len(col_bands), tile_h, tile_w), dtype=int)
    for br, (r0, r1) in enumerate(row_bands):
        for bc, (c0, c1) in enumerate(col_bands):
            tiles[br, bc] = grid[r0:r1, c0:c1]
    return tiles


def main(grid: np.ndarray) -> np.ndarray:
    """
    Find the unique anomalous tile in a separator-delimited repeated-tile array.

    The consensus tile is recovered by taking the unique majority color at each
    local coordinate across all tiles. The output is a same-size mask: anomalous
    colors at differing coordinates, background everywhere else.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _dominant_color(g)
    tiles = _extract_tiles(g, bg)
    block_rows, block_cols, tile_h, tile_w = tiles.shape

    consensus = np.empty((tile_h, tile_w), dtype=int)
    for r in range(tile_h):
        for c in range(tile_w):
            consensus[r, c] = _unique_mode(tiles[:, :, r, c])

    diff_mask = tiles != consensus
    diff_counts = diff_mask.reshape(block_rows, block_cols, tile_h * tile_w).sum(axis=2)
    max_diff = int(diff_counts.max())
    assert max_diff > 0, "expected exactly one anomalous tile"

    anomalous_tiles = np.argwhere(diff_counts == max_diff)
    assert anomalous_tiles.shape[0] == 1, "anomalous tile must be unique"
    bad_r, bad_c = map(int, anomalous_tiles[0])

    local_diff = diff_mask[bad_r, bad_c]
    out = np.full((tile_h, tile_w), bg, dtype=int)
    out[local_diff] = tiles[bad_r, bad_c][local_diff]
    return out


def _ensure_row_col_coverage(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    for r in range(h):
        if not bool(mask[r].any()):
            mask[r, random.randrange(w)] = True
    for c in range(w):
        if not bool(mask[:, c].any()):
            mask[random.randrange(h), c] = True
    return mask


def _target_fill_count(h: int, w: int) -> int:
    area = h * w
    lower = max(h, w)
    upper = max(lower, min(area - 1, int(np.floor(0.45 * area))))
    return random.randint(lower, upper)


def _random_mask_scatter(h: int, w: int, target: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    for r in range(h):
        mask[r, random.randrange(w)] = True
    for c in range(w):
        mask[random.randrange(h), c] = True
    while int(mask.sum()) < target:
        mask[random.randrange(h), random.randrange(w)] = True
    return _ensure_row_col_coverage(mask)


def _random_mask_strokes(h: int, w: int, target: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    n_strokes = random.randint(2, 4)
    for _ in range(n_strokes):
        if (random.random() < 0.5 and w > 1) or h == 1:
            r = random.randrange(h)
            c0 = random.randrange(w)
            c1 = random.randrange(c0, w)
            mask[r, c0 : c1 + 1] = True
        else:
            c = random.randrange(w)
            r0 = random.randrange(h)
            r1 = random.randrange(r0, h)
            mask[r0 : r1 + 1, c] = True
    mask = _ensure_row_col_coverage(mask)
    while int(mask.sum()) < target:
        r = random.randrange(h)
        c = random.randrange(w)
        if bool(mask[r, c]):
            continue
        mask[r, c] = True
    return mask


def _random_mask_growth(h: int, w: int, target: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    seeds = [(random.randrange(h), random.randrange(w)) for _ in range(random.randint(1, 3))]
    for r, c in seeds:
        mask[r, c] = True

    while int(mask.sum()) < target:
        frontier = np.argwhere(mask)
        r, c = map(int, frontier[random.randrange(len(frontier))])
        neighbors = []
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr = r + dr
            nc = c + dc
            if 0 <= nr < h and 0 <= nc < w:
                neighbors.append((nr, nc))
        random.shuffle(neighbors)
        placed = False
        for nr, nc in neighbors:
            if not bool(mask[nr, nc]):
                mask[nr, nc] = True
                placed = True
                break
        if not placed:
            mask[random.randrange(h), random.randrange(w)] = True

    return _ensure_row_col_coverage(mask)


def _build_motif(h: int, w: int, bg: int) -> tuple[np.ndarray, list[int]]:
    target = _target_fill_count(h, w)
    style = random.choice(["scatter", "strokes", "growth"])
    if style == "scatter":
        mask = _random_mask_scatter(h, w, target)
    elif style == "strokes":
        mask = _random_mask_strokes(h, w, target)
    else:
        mask = _random_mask_growth(h, w, target)

    motif = np.full((h, w), bg, dtype=int)
    color_style = random.choice(["single", "pattern", "patch"])
    if color_style == "single":
        palette = random.sample([c for c in COLOR_POOL if c != bg], 1)
    else:
        palette_size = random.randint(2, min(4, int(mask.sum())))
        palette = random.sample([c for c in COLOR_POOL if c != bg], palette_size)

    coords = [tuple(map(int, rc)) for rc in np.argwhere(mask)]
    if color_style == "single":
        color = palette[0]
        for r, c in coords:
            motif[r, c] = color
    elif color_style == "pattern":
        bias = random.randrange(len(palette))
        for r, c in coords:
            motif[r, c] = palette[(2 * r + c + bias) % len(palette)]
    else:
        random.shuffle(coords)
        for color, (r, c) in zip(palette, coords):
            motif[r, c] = color
        for r, c in coords[len(palette) :]:
            motif[r, c] = random.choice(palette)

    return motif, sorted(set(map(int, motif[motif != bg])))


def _cluster_cells(h: int, w: int, count: int) -> list[tuple[int, int]]:
    chosen = {(random.randrange(h), random.randrange(w))}
    while len(chosen) < count:
        r, c = random.choice(list(chosen))
        neighbors = []
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr = r + dr
            nc = c + dc
            if 0 <= nr < h and 0 <= nc < w:
                neighbors.append((nr, nc))
        random.shuffle(neighbors)
        for cell in neighbors:
            chosen.add(cell)
            if len(chosen) == count:
                break
        if len(chosen) < count and len(chosen) == h * w:
            break
        if len(chosen) < count and random.random() < 0.2:
            chosen.add((random.randrange(h), random.randrange(w)))
    return list(chosen)


def _pick_diff_cells(h: int, w: int) -> list[tuple[int, int]]:
    area = h * w
    max_diff = max(1, min(area - 1, max(2, area // 4)))
    count = random.randint(1, max_diff)
    if random.random() < 0.5:
        cells = random.sample([(r, c) for r in range(h) for c in range(w)], count)
    else:
        cells = _cluster_cells(h, w, count)
    return [(int(r), int(c)) for r, c in cells]


def _make_anomalous_tile(motif: np.ndarray, bg: int, motif_colors: list[int]) -> np.ndarray:
    h, w = motif.shape
    anomaly = motif.copy()
    diff_cells = _pick_diff_cells(h, w)

    extra_colors = [c for c in COLOR_POOL if c != bg and c not in motif_colors]
    mode = random.choice(["swap_existing", "paint_new", "mixed"])

    for r, c in diff_cells:
        old = int(anomaly[r, c])
        if mode == "swap_existing":
            choices = [color for color in motif_colors if color != old]
            if not choices:
                choices = extra_colors[:]
        elif mode == "paint_new":
            choices = extra_colors[:]
        else:
            choices = [color for color in motif_colors + extra_colors if color != old]
        assert choices, "need at least one visible anomaly color"
        anomaly[r, c] = random.choice(choices)

    return anomaly


def _build_grid(
    motif: np.ndarray,
    anomaly: np.ndarray,
    block_rows: int,
    block_cols: int,
    sep_h: int,
    sep_w: int,
    outer_border: bool,
    bg: int,
) -> tuple[np.ndarray, np.ndarray]:
    tile_h, tile_w = motif.shape
    if outer_border:
        height = block_rows * tile_h + (block_rows + 1) * sep_h
        width = block_cols * tile_w + (block_cols + 1) * sep_w
        row_base = sep_h
        col_base = sep_w
    else:
        height = block_rows * tile_h + (block_rows - 1) * sep_h
        width = block_cols * tile_w + (block_cols - 1) * sep_w
        row_base = 0
        col_base = 0

    grid = np.full((height, width), bg, dtype=int)
    bad_index = random.randrange(block_rows * block_cols)
    bad_row, bad_col = divmod(bad_index, block_cols)

    for br in range(block_rows):
        for bc in range(block_cols):
            r0 = row_base + br * (tile_h + sep_h)
            c0 = col_base + bc * (tile_w + sep_w)
            tile = anomaly if (br, bc) == (bad_row, bad_col) else motif
            grid[r0 : r0 + tile_h, c0 : c0 + tile_w] = tile

    out = np.full((tile_h, tile_w), bg, dtype=int)
    diff = anomaly != motif
    out[diff] = anomaly[diff]
    return grid, out


def _dominant_is_unique(grid: np.ndarray, bg: int) -> bool:
    colors, counts = np.unique(grid, return_counts=True)
    top = int(counts.max())
    winners = [int(color) for color, count in zip(colors, counts) if int(count) == top]
    return len(winners) == 1 and winners[0] == bg


def _generate_attempt() -> np.ndarray | None:
    bg = random.randrange(10)
    block_rows = random.randint(2, 4)
    block_cols = random.randint(2, 4)
    tile_h = random.randint(3, 7)
    tile_w = random.randint(3, 7)
    sep_h = random.randint(1, 2)
    sep_w = random.randint(1, 2)
    outer_border = bool(random.getrandbits(1))

    motif, motif_colors = _build_motif(tile_h, tile_w, bg)
    anomaly = _make_anomalous_tile(motif, bg, motif_colors)
    if np.array_equal(anomaly, motif):
        return None

    grid, expected = _build_grid(
        motif,
        anomaly,
        block_rows,
        block_cols,
        sep_h,
        sep_w,
        outer_border,
        bg,
    )

    if grid.shape[0] > MAX_GRID_SIDE or grid.shape[1] > MAX_GRID_SIDE:
        return None
    if not _dominant_is_unique(grid, bg):
        return None
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
