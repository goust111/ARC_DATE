from __future__ import annotations

import random
from itertools import combinations

import numpy as np


BLACK = 0
GREY = 5
ORANGE = 7
PARTITION_COLORS = (1, 2, 3, 4, 6, 8, 9)


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_bbox(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = map(int, bbox)
    return np.asarray(grid[r0 : r1 + 1, c0 : c1 + 1], dtype=int).copy()


def _connected_components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    components: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(int(r), int(c))]
            seen[r, c] = True
            coords = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            components.append(np.array(coords, dtype=int))
    return components


def _find_monochromatic_components(grid: np.ndarray, bg: int = BLACK) -> list[dict]:
    out = []
    for color in sorted(set(map(int, grid.ravel())) - {int(bg)}):
        for coords in _connected_components(grid == int(color)):
            rr = coords[:, 0]
            cc = coords[:, 1]
            out.append(
                {
                    "color": int(color),
                    "coords": coords,
                    "bbox": (int(rr.min()), int(cc.min()), int(rr.max()), int(cc.max())),
                    "area": int(coords.shape[0]),
                }
            )
    return out


def _extract_color_matrix(top: np.ndarray) -> np.ndarray:
    comps = _find_monochromatic_components(top, bg=BLACK)
    assert comps, "expected a colored partition above the grey separator"

    union = np.zeros_like(top, dtype=bool)
    for comp in comps:
        r0, c0, r1, c1 = comp["bbox"]
        rect_area = (r1 - r0 + 1) * (c1 - c0 + 1)
        assert comp["area"] == rect_area, "each partition cell must be a dense rectangle"
        union[comp["coords"][:, 0], comp["coords"][:, 1]] = True

    outer = _bbox(union)
    assert outer is not None
    r0, c0, r1, c1 = outer
    assert np.all(top[r0 : r1 + 1, c0 : c1 + 1] != BLACK), "partition must tile one rectangle without gaps"

    row_starts = sorted({int(comp["bbox"][0]) for comp in comps})
    col_starts = sorted({int(comp["bbox"][1]) for comp in comps})
    n_rows = len(row_starts)
    n_cols = len(col_starts)
    assert len(comps) == n_rows * n_cols, "partition must have one rectangle per row/column band intersection"

    row_ends: dict[int, int] = {}
    col_ends: dict[int, int] = {}
    for comp in comps:
        br0, bc0, br1, bc1 = map(int, comp["bbox"])
        row_ends.setdefault(br0, br1)
        col_ends.setdefault(bc0, bc1)
        assert row_ends[br0] == br1, "all rectangles in one partition row must share height"
        assert col_ends[bc0] == bc1, "all rectangles in one partition column must share width"

    matrix = np.zeros((n_rows, n_cols), dtype=int)
    for i, r in enumerate(row_starts):
        for j, c in enumerate(col_starts):
            color = int(top[r, c])
            assert color != BLACK, "sample positions must hit partition cells"
            matrix[i, j] = color
    return matrix


def _extract_tile(lower: np.ndarray) -> np.ndarray:
    fg_mask = lower != BLACK
    comps = _connected_components(fg_mask)
    assert len(comps) == 1, "expected exactly one lower foreground pattern"
    coords = comps[0]
    colors = {int(lower[r, c]) for r, c in coords}
    assert len(colors) == 1, "lower pattern must be monochrome"
    bbox = _bbox(fg_mask)
    assert bbox is not None
    tile = _crop_bbox(lower, bbox)
    assert 3 <= int((tile != BLACK).sum()) <= 5, "tile foreground size must be between 3 and 5"
    assert 2 <= tile.shape[0] <= 4 and 2 <= tile.shape[1] <= 4, "tile bbox must be between 2x2 and 4x4"
    return tile


def _render_output(color_matrix: np.ndarray, tile: np.ndarray) -> np.ndarray:
    fg_colors = sorted(set(map(int, tile.ravel())) - {BLACK})
    assert len(fg_colors) == 1, "tile must have exactly one foreground color"
    tile_fg = int(fg_colors[0])
    p, q = map(int, tile.shape)
    n, m = map(int, color_matrix.shape)
    out = np.zeros((n * p, m * q), dtype=int)
    mask = tile == tile_fg
    for i in range(n):
        for j in range(m):
            block = out[i * p : (i + 1) * p, j * q : (j + 1) * q]
            block[mask] = int(color_matrix[i, j])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Downsample the upper rectangular partition into a color matrix and use it to tile a cropped lower motif.

    Input structure:
    - Black background with exactly one full-width grey horizontal separator.
    - Above the separator is a tight rectangular partition made of monochrome rectangles; rectangles in one row share height and rectangles in one column share width.
    - Below the separator is one small 4-connected monochrome motif on black background.

    Rule:
    - Find all upper partition rectangles, sort their unique top-left row and column coordinates, and sample the grid at those coordinates to build the downsampled color matrix.
    - Crop the lower motif to its bounding box to obtain a tile whose foreground is the motif color and whose background stays black.
    - Create an output grid with one recolored tile per color-matrix entry; each tile foreground pixel becomes that entry's color and each tile background pixel remains black.

    Key insight:
    - The upper partition is not resized directly: it is first reduced to a color atlas, and that atlas simultaneously determines both the output tiling dimensions and the recoloring of every copied tile.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    grey_rows = [r for r in range(g.shape[0]) if np.all(g[r, :] == GREY)]
    assert len(grey_rows) == 1, "expected exactly one full-width grey separator"
    sep = int(grey_rows[0])
    assert 0 < sep < g.shape[0] - 1, "separator must split the grid into two non-empty regions"

    top = g[:sep, :]
    lower = g[sep + 1 :, :]
    color_matrix = _extract_color_matrix(top)
    tile = _extract_tile(lower)
    return _render_output(color_matrix, tile)


def _tile_library() -> dict[tuple[int, int], list[np.ndarray]]:
    library: dict[tuple[int, int], list[np.ndarray]] = {}
    for p in range(2, 5):
        for q in range(2, 5):
            options = []
            cells = [(r, c) for r in range(p) for c in range(q)]
            for area in range(3, min(5, p * q - 1) + 1):
                for chosen in combinations(cells, area):
                    mask = np.zeros((p, q), dtype=bool)
                    for r, c in chosen:
                        mask[r, c] = True
                    if len(_connected_components(mask)) != 1:
                        continue
                    bb = _bbox(mask)
                    assert bb is not None
                    if bb != (0, 0, p - 1, q - 1):
                        continue
                    if area == p * q:
                        continue
                    options.append(mask.astype(int))
            if options:
                library[(p, q)] = options
    return library


TILE_LIBRARY = _tile_library()
TILE_SHAPES = tuple(sorted(TILE_LIBRARY))


def _choose_partition_colors(n: int, m: int) -> np.ndarray:
    palette_size = random.randint(3, min(len(PARTITION_COLORS), max(3, min(6, n * m))))
    palette = random.sample(PARTITION_COLORS, palette_size)
    matrix = np.full((n, m), -1, dtype=int)
    for r in range(n):
        for c in range(m):
            banned = set()
            if r > 0:
                banned.add(int(matrix[r - 1, c]))
            if c > 0:
                banned.add(int(matrix[r, c - 1]))
            candidates = [color for color in palette if color not in banned]
            if not candidates:
                return _choose_partition_colors(n, m)
            matrix[r, c] = random.choice(candidates)
    if len(set(map(int, matrix.ravel()))) < 2:
        return _choose_partition_colors(n, m)
    return matrix


def _choose_band_sizes(k: int, lo: int = 2, hi: int = 5) -> list[int]:
    vals = [random.randint(lo, hi) for _ in range(k)]
    if k >= 2 and len(set(vals)) == 1:
        idx = random.randrange(k)
        vals[idx] = lo if vals[idx] > lo else hi
    return vals


def _place_partition(grid: np.ndarray, top: int, left: int, row_heights: list[int], col_widths: list[int], matrix: np.ndarray) -> None:
    r = int(top)
    for i, h in enumerate(row_heights):
        c = int(left)
        for j, w in enumerate(col_widths):
            grid[r : r + h, c : c + w] = int(matrix[i, j])
            c += w
        r += h


def _expected_from_latent(matrix: np.ndarray, tile_mask: np.ndarray) -> np.ndarray:
    tile = np.where(tile_mask > 0, ORANGE, BLACK).astype(int)
    return _render_output(matrix, tile)


def _validate_instance(
    grid: np.ndarray,
    matrix: np.ndarray,
    tile_mask: np.ndarray,
    row_heights: list[int],
    col_widths: list[int],
) -> None:
    assert grid.ndim == 2
    assert max(grid.shape) <= 30
    assert len(set(row_heights)) > 1 or len(set(col_widths)) > 1, "partition sizes must vary on at least one axis"
    assert len(set(map(int, matrix.ravel()))) >= 2, "color matrix must use at least two colors"

    for r in range(matrix.shape[0]):
        for c in range(matrix.shape[1]):
            if r + 1 < matrix.shape[0]:
                assert int(matrix[r, c]) != int(matrix[r + 1, c]), "adjacent partition rows must stay separable"
            if c + 1 < matrix.shape[1]:
                assert int(matrix[r, c]) != int(matrix[r, c + 1]), "adjacent partition columns must stay separable"

    assert tile_mask.shape[0] in range(2, 5) and tile_mask.shape[1] in range(2, 5)
    assert 3 <= int(tile_mask.sum()) <= 5
    assert int(tile_mask.sum()) < int(np.prod(tile_mask.shape)), "tile must be non-solid so cropping and background matter"
    assert len(_connected_components(tile_mask.astype(bool))) == 1
    assert _bbox(tile_mask.astype(bool)) == (0, 0, tile_mask.shape[0] - 1, tile_mask.shape[1] - 1)

    expected = _expected_from_latent(matrix, tile_mask)
    actual = main(grid)
    assert np.array_equal(actual, expected), "solver output must match latent construction"
    assert max(actual.shape) <= 30
    assert actual.shape == (matrix.shape[0] * tile_mask.shape[0], matrix.shape[1] * tile_mask.shape[1])
    assert np.any(actual == BLACK), "output should preserve tile background"
    assert len(set(map(int, actual.ravel())) - {BLACK}) >= 2, "output must visibly depend on recoloring"

    # Each output block must be the same tile mask but recolored by the sampled partition color.
    p, q = tile_mask.shape
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            block = actual[i * p : (i + 1) * p, j * q : (j + 1) * q]
            assert np.array_equal(block != BLACK, tile_mask.astype(bool))
            assert set(map(int, block[tile_mask.astype(bool)])) == {int(matrix[i, j])}


def generate_input() -> np.ndarray:
    for _ in range(500):
        n = random.randint(2, 5)
        m = random.randint(2, 5)
        row_heights = _choose_band_sizes(n)
        col_widths = _choose_band_sizes(m)
        part_h = sum(row_heights)
        part_w = sum(col_widths)

        tile_h, tile_w = random.choice(TILE_SHAPES)
        tile_mask = random.choice(TILE_LIBRARY[(tile_h, tile_w)])
        matrix = _choose_partition_colors(n, m)

        top_margin = random.randint(1, 3)
        bottom_margin = random.randint(1, 3)
        left_margin = random.randint(1, 3)
        right_margin = random.randint(1, 3)
        gap_above_sep = random.randint(1, 2)
        gap_below_sep = random.randint(1, 2)
        lower_h = tile_h + random.randint(2, 5)
        lower_w = tile_w + random.randint(2, 7)

        height = top_margin + part_h + gap_above_sep + 1 + gap_below_sep + lower_h + bottom_margin
        width = max(left_margin + part_w + right_margin, lower_w + 2)
        if height > 30 or width > 30:
            continue

        grid = np.zeros((height, width), dtype=int)
        sep = top_margin + part_h + gap_above_sep
        grid[sep, :] = GREY

        part_left = random.randint(1, width - part_w - 1)
        _place_partition(grid, top_margin, part_left, row_heights, col_widths, matrix)

        tile = np.where(tile_mask > 0, ORANGE, BLACK).astype(int)
        lower_top = sep + 1 + gap_below_sep + random.randint(0, lower_h - tile_h)
        lower_left = random.randint(1, width - tile_w - 1)
        tile_patch = tile.copy()
        grid[lower_top : lower_top + tile_h, lower_left : lower_left + tile_w][tile_patch == ORANGE] = ORANGE

        try:
            _validate_instance(grid, matrix, tile_mask, row_heights, col_widths)
        except AssertionError:
            continue
        return grid

    raise RuntimeError("failed to generate a valid ARC instance")
