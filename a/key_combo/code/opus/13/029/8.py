import numpy as np


BG = 0
SEP = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8)
DIRS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
RNG = np.random.default_rng()
MAX_GENERATION_ATTEMPTS = 400


def _row_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start = None
    for idx, value in enumerate(mask.tolist()):
        if value and start is None:
            start = idx
        elif not value and start is not None:
            runs.append((start, idx))
            start = None
    if start is not None:
        runs.append((start, len(mask)))
    return runs


def _extract_band_tiles(band: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sep_cols = [c for c in range(band.shape[1]) if np.all(band[:, c] == SEP)]
    assert len(sep_cols) == 2, "each band must contain exactly two separator columns"
    s1, s2 = map(int, sep_cols)
    w1 = s1
    w2 = s2 - s1 - 1
    assert w1 == w2 and w1 >= 3, "band tiles must have the same width"
    start3 = s2 + 1
    end3 = start3 + w1
    assert end3 <= band.shape[1], "third tile leaves band bounds"

    trailing = band[:, end3:]
    assert trailing.size == 0 or np.all(trailing == BG), "only black padding may follow the third tile"

    tile1 = band[:, :w1]
    tile2 = band[:, s1 + 1 : s2]
    tile3 = band[:, start3:end3]
    assert tile1.shape == tile2.shape == tile3.shape
    return tile1, tile2, tile3


def _infer_mapping(src: np.ndarray, dst: np.ndarray) -> dict[int, int]:
    src_mask = src != BG
    dst_mask = dst != BG
    assert np.array_equal(src_mask, dst_mask), "example pair must share the same non-black layout"

    mapping = {BG: BG}
    for color in sorted(set(map(int, src[src_mask].flatten()))):
        dest_values = sorted(set(map(int, dst[src == color].flatten())))
        assert len(dest_values) == 1, "every source color must map consistently"
        mapping[color] = int(dest_values[0])

    mapped = [mapping[color] for color in mapping if color != BG]
    assert len(mapped) == len(set(mapped)), "mapping must be one-to-one"
    return mapping


def _apply_mapping(tile: np.ndarray, mapping: dict[int, int]) -> np.ndarray:
    out = np.zeros_like(tile)
    for src_color, dst_color in mapping.items():
        out[tile == int(src_color)] = int(dst_color)
    return out


def _solve_band(tile1: np.ndarray, tile2: np.ndarray, tile3: np.ndarray) -> np.ndarray:
    mask = tile1 != BG
    assert np.array_equal(mask, tile3 != BG), "the third tile must reuse the same non-black layout"

    mapping = _infer_mapping(tile1, tile2)
    changed = [color for color, mapped in mapping.items() if color != BG and color != mapped]
    k = len(changed)
    assert 2 <= k <= 4, "effective mapping size must define a 2x-4x scale"

    third_colors = set(map(int, tile3[mask].flatten()))
    assert third_colors <= {color for color in mapping if color != BG}, "third tile uses an unknown source color"

    recolored = _apply_mapping(tile3, mapping)
    return np.kron(recolored, np.ones((k, k), dtype=int))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Infer a color mapping from the first two tiles in each band, then enlarge-and-recolor the third tile.

    Input structure:
    - Black background contains one or two horizontal bands separated by full gray rows.
    - Each band holds three equal-height tiles, with two full gray separator columns inside the band.
    - The first two tiles share the same non-black layout and define a one-to-one color substitution.

    Rule:
    - Split the grid into bands with gray separator rows, then split each band into three tiles with gray separator columns.
    - For each band, read the source-to-destination color mapping from matching positions in tile 1 and tile 2.
    - Count the non-identity color pairs K, recolor tile 3 with that mapping, and expand every pixel of tile 3 into a solid KxK block.
    - Stack the enlarged results from top band to bottom band without separator rows.

    Key insight:
    The example color mapping does double duty: it changes tile-3 colors and its number of actual color changes determines the enlargement factor.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert max(g.shape) <= 30, "ARC grids must stay within 30x30"

    band_rows = _row_runs(~np.all(g == SEP, axis=1))
    assert 1 <= len(band_rows) <= 2, "expected one or two horizontal bands"

    outputs = []
    for r0, r1 in band_rows:
        band = g[r0:r1, :]
        tile1, tile2, tile3 = _extract_band_tiles(band)
        outputs.append(_solve_band(tile1, tile2, tile3))

    widths = {band.shape[1] for band in outputs}
    assert len(widths) == 1, "stacked outputs must share one width"
    out = np.vstack(outputs)
    assert max(out.shape) <= 30, "output grid exceeds ARC size limits"
    return out


def _random_connected_mask(h: int, w: int, area: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    seed = (int(RNG.integers(h)), int(RNG.integers(w)))
    mask[seed] = True

    while int(mask.sum()) < area:
        frontier: set[tuple[int, int]] = set()
        for r, c in np.argwhere(mask):
            for dr, dc in DIRS4:
                nr, nc = int(r + dr), int(c + dc)
                if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc]:
                    frontier.add((nr, nc))
        assert frontier, "frontier vanished during mask growth"
        pick = list(frontier)[int(RNG.integers(len(frontier)))]
        mask[pick] = True

    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    if len(rows) < 2 or len(cols) < 2:
        raise ValueError("mask is too degenerate")
    return mask


def _choose_mapping(src_colors: list[int], k: int) -> dict[int, int]:
    mapping = {int(color): int(color) for color in src_colors}
    moved = [int(color) for color in RNG.choice(np.array(src_colors, dtype=int), size=k, replace=False)]
    RNG.shuffle(moved)

    if k == 2:
        a, b = moved
        mapping[a], mapping[b] = b, a
    elif k == 3:
        a, b, c = moved
        mapping[a], mapping[b], mapping[c] = b, c, a
    elif RNG.random() < 0.5:
        a, b, c, d = moved
        mapping[a], mapping[b], mapping[c], mapping[d] = b, c, d, a
    else:
        a, b, c, d = moved
        mapping[a], mapping[b], mapping[c], mapping[d] = b, a, d, c

    return mapping


def _paint_tile(mask: np.ndarray, colors: list[int]) -> np.ndarray:
    coords = [tuple(map(int, rc)) for rc in np.argwhere(mask)]
    fill = list(map(int, colors))
    while len(fill) < len(coords):
        fill.append(int(RNG.choice(np.array(colors, dtype=int))))
    RNG.shuffle(fill)

    tile = np.zeros(mask.shape, dtype=int)
    for (r, c), color in zip(coords, fill):
        tile[r, c] = int(color)
    return tile


def _choose_third_palette(src_colors: list[int], changed_colors: list[int]) -> list[int]:
    if len(src_colors) > 2 and RNG.random() < 0.65:
        size = int(RNG.integers(2, len(src_colors)))
    else:
        size = int(RNG.integers(2, len(src_colors) + 1))

    subset = {int(RNG.choice(np.array(changed_colors, dtype=int)))}
    remaining = [int(color) for color in src_colors if color not in subset]
    if size > 1:
        chosen = RNG.choice(np.array(remaining, dtype=int), size=size - 1, replace=False)
        subset.update(map(int, chosen.tolist()))
    return list(subset)


def _build_band(h: int, w: int, k: int) -> tuple[np.ndarray, np.ndarray]:
    max_area = h * w - 1
    min_area = max(k + 1, 5)
    area = int(RNG.integers(min_area, max_area + 1))
    mask = _random_connected_mask(h, w, area)

    max_colors = min(len(PALETTE), area, 5)
    n_src = int(RNG.integers(k, max_colors + 1))
    src_colors = list(map(int, RNG.choice(np.array(PALETTE, dtype=int), size=n_src, replace=False).tolist()))
    mapping = _choose_mapping(src_colors, k)
    changed_colors = [color for color in src_colors if mapping[color] != color]

    tile1 = _paint_tile(mask, src_colors)
    tile2 = _apply_mapping(tile1, mapping)
    third_palette = _choose_third_palette(src_colors, changed_colors)
    tile3 = _paint_tile(mask, third_palette)

    band_width = 3 * w + 2
    band = np.zeros((h, band_width), dtype=int)
    band[:, w] = SEP
    band[:, 2 * w + 1] = SEP
    band[:, :w] = tile1
    band[:, w + 1 : 2 * w + 1] = tile2
    band[:, 2 * w + 2 : 3 * w + 2] = tile3

    out = _solve_band(tile1, tile2, tile3)
    return band, out


def _choose_specs() -> list[tuple[int, int, int]]:
    if RNG.random() < 0.45:
        return [
            (
                int(RNG.integers(3, 5)),
                int(RNG.integers(3, 5)),
                int(RNG.integers(2, 5)),
            )
        ]

    if RNG.random() < 0.4:
        specs = [
            (int(RNG.integers(3, 5)), 4, 3),
            (int(RNG.integers(3, 5)), 3, 4),
        ]
        if RNG.random() < 0.5:
            specs.reverse()
        return specs

    while True:
        h1 = int(RNG.integers(3, 5))
        h2 = int(RNG.integers(3, 5))
        w = int(RNG.integers(3, 5))
        k = int(RNG.integers(2, 5))
        if k * (h1 + h2) <= 30:
            return [(h1, w, k), (h2, w, k)]


def _assemble_input(bands: list[np.ndarray]) -> np.ndarray:
    height = sum(int(band.shape[0]) for band in bands) + len(bands) - 1
    width = max(int(band.shape[1]) for band in bands)
    grid = np.zeros((height, width), dtype=int)

    r = 0
    for idx, band in enumerate(bands):
        h, w = band.shape
        grid[r : r + h, :w] = band
        r += h
        if idx + 1 < len(bands):
            grid[r, :] = SEP
            r += 1
    return grid


def _attempt_generate() -> tuple[np.ndarray, np.ndarray] | None:
    specs = _choose_specs()
    if len({w * k for _h, w, k in specs}) != 1:
        return None

    bands: list[np.ndarray] = []
    outputs: list[np.ndarray] = []
    for h, w, k in specs:
        try:
            band, out = _build_band(h, w, k)
        except (AssertionError, ValueError):
            return None
        bands.append(band)
        outputs.append(out)

    if sum(int(out.shape[0]) for out in outputs) > 30:
        return None

    grid = _assemble_input(bands)
    expected = np.vstack(outputs)
    if max(grid.shape) > 30 or max(expected.shape) > 30:
        return None

    try:
        observed = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(observed, expected):
        return None
    return grid, expected


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        built = _attempt_generate()
        if built is not None:
            grid, _expected = built
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 8")
