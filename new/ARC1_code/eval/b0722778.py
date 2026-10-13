from __future__ import annotations

import numpy as np

from arc_utils import get_bg, paste


def _foreground_runs(mask: np.ndarray, axis: int) -> list[tuple[int, int]]:
    occupied = mask.any(axis=axis)
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for idx, value in enumerate(list(map(bool, occupied)) + [False]):
        if value and start is None:
            start = idx
        elif not value and start is not None:
            runs.append((start, idx))
            start = None
    return runs


def _color_mapping(src: np.ndarray, dst: np.ndarray) -> dict[int, int] | None:
    mapping: dict[int, int] = {}
    reverse: dict[int, int] = {}
    for src_color, dst_color in zip(src.reshape(-1), dst.reshape(-1)):
        s = int(src_color)
        d = int(dst_color)
        if s in mapping and mapping[s] != d:
            return None
        if d in reverse and reverse[d] != s:
            return None
        mapping[s] = d
        reverse[d] = s
    return mapping


def _recolor(tile: np.ndarray, mapping: dict[int, int]) -> np.ndarray:
    out = np.empty_like(tile)
    for color in np.unique(tile):
        color = int(color)
        assert color in mapping, "tile colors must be covered by the learned mapping"
        out[tile == color] = mapping[color]
    return out


def _solve_band(tiles: list[np.ndarray]) -> np.ndarray:
    assert len(tiles) == 3, "expected three candidate tiles in each band"
    candidates: list[np.ndarray] = []
    for src_idx in range(3):
        for dst_idx in range(3):
            if src_idx == dst_idx:
                continue
            apply_idx = 3 - src_idx - dst_idx
            mapping = _color_mapping(tiles[src_idx], tiles[dst_idx])
            if mapping is None:
                continue
            if set(map(int, tiles[apply_idx].reshape(-1))) <= set(mapping):
                candidates.append(_recolor(tiles[apply_idx], mapping))

    assert len(candidates) == 1, "each band must have one unambiguous tile mapping"
    return candidates[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use the matching pair of tiles in each band as a color-substitution example and apply it to the remaining tile.

    Input structure:
    - Background rows separate horizontal bands of small same-size colored tiles.
    - Each band contains exactly three non-background tile regions in the same columns.
    - Within a band, exactly two tiles have the same equality pattern up to color renaming.

    Rule:
    - For each band, find the ordered pair of same-pattern tiles whose source colors cover the third tile.
    - Learn the color mapping from that source tile to the matching destination tile.
    - Recolor the third tile by that mapping and place it in the corresponding output band.

    Key insight:
    The duplicated tile pattern is a miniature example pair that defines how to translate colors for the odd tile out.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = int(get_bg(g))
    foreground = g != bg

    row_runs = _foreground_runs(foreground, axis=1)
    col_runs = _foreground_runs(foreground, axis=0)
    assert row_runs, "expected at least one foreground band"
    assert len(col_runs) == 3, "expected three tile columns"

    tile_widths = {c1 - c0 for c0, c1 in col_runs}
    assert len(tile_widths) == 1, "tile columns must have equal widths"
    tile_width = tile_widths.pop()
    assert tile_width > 0, "tile width must be positive"

    out = np.full((g.shape[0], tile_width), bg, dtype=int)
    for r0, r1 in row_runs:
        tiles = [g[r0:r1, c0:c1] for c0, c1 in col_runs]
        assert all(tile.shape == tiles[0].shape for tile in tiles), "tiles in a band must share shape"
        out[r0:r1, :] = _solve_band(tiles)

    return out

# --------------- generator (o5x) ---------------
# Tile pattern and palette configuration.
_GEN_TILE_SHAPE = (2, 2)
_GEN_COLORS = np.arange(1, 10, dtype=int)
_GEN_PATTERNS = tuple(
    np.array(bits, dtype=int).reshape(_GEN_TILE_SHAPE)
    for bits in (
        (0, 0, 0, 1),
        (0, 0, 1, 0),
        (0, 0, 1, 1),
        (0, 1, 0, 0),
        (0, 1, 0, 1),
        (0, 1, 1, 0),
        (0, 1, 1, 1),
    )
)
# Canvas layout and band-count bounds.
_GEN_MIN_BANDS, _GEN_MAX_BANDS_EXCLUSIVE = 3, 5
_GEN_BG_COLOR, _GEN_SEPARATOR_ROWS = 0, 1
_GEN_BAND_STRIDE = _GEN_TILE_SHAPE[0] + _GEN_SEPARATOR_ROWS
_GEN_INPUT_WIDTH, _GEN_OUTPUT_WIDTH, _GEN_TILE_COLS = 9, _GEN_TILE_SHAPE[1], (0, 3, 7)
# Color-mapping mode and validation retry limits.
_GEN_COLOR_PAIR_SIZE, _GEN_MAPPING_MODE_COUNT = 2, 2
_GEN_RETRY_LIMIT = 300
_GEN_RNG = np.random.default_rng()

def _gen_tile(pattern: np.ndarray, color0: int, color1: int) -> np.ndarray:
    return np.where(pattern == 0, int(color0), int(color1)).astype(int)


def _gen_color_mapping(rng: np.random.Generator) -> tuple[int, int, int, int]:
    src = rng.choice(_GEN_COLORS, size=_GEN_COLOR_PAIR_SIZE, replace=False)
    remaining = _GEN_COLORS[~np.isin(_GEN_COLORS, src)]
    if bool(rng.integers(_GEN_MAPPING_MODE_COUNT)):
        dst = rng.choice(remaining, size=_GEN_COLOR_PAIR_SIZE, replace=False)
    else:
        dst = np.empty(_GEN_COLOR_PAIR_SIZE, dtype=int)
        shared_src_i = int(rng.integers(_GEN_COLOR_PAIR_SIZE))
        shared_dst_i = int(rng.integers(_GEN_COLOR_PAIR_SIZE))
        new_color = int(rng.choice(remaining))
        dst[shared_dst_i] = src[shared_src_i]
        dst[1 - shared_dst_i] = new_color
    return int(src[0]), int(src[1]), int(dst[0]), int(dst[1])


def _gen_band_tiles(rng: np.random.Generator) -> tuple[tuple[np.ndarray, ...], np.ndarray]:
    pattern_idxs = rng.choice(len(_GEN_PATTERNS), size=_GEN_COLOR_PAIR_SIZE, replace=False)
    matched_pattern = _GEN_PATTERNS[int(pattern_idxs[0])]
    odd_pattern = _GEN_PATTERNS[int(pattern_idxs[1])]
    color_a, color_b, color_c, color_d = _gen_color_mapping(rng)

    output_tile = _gen_tile(odd_pattern, color_c, color_d)
    odd_tile = _gen_tile(odd_pattern, color_a, color_b)
    src_tile = _gen_tile(matched_pattern, color_a, color_b)
    dst_tile = _gen_tile(matched_pattern, color_c, color_d)
    return (src_tile, dst_tile, odd_tile), output_tile


def _gen_attempt() -> np.ndarray | None:
    rng = _GEN_RNG
    num_bands = int(rng.integers(_GEN_MIN_BANDS, _GEN_MAX_BANDS_EXCLUSIVE))
    height = _GEN_BAND_STRIDE * num_bands - _GEN_SEPARATOR_ROWS
    inp = np.full((height, _GEN_INPUT_WIDTH), _GEN_BG_COLOR, dtype=int)
    out = np.full((height, _GEN_OUTPUT_WIDTH), _GEN_BG_COLOR, dtype=int)

    for band_i in range(num_bands):
        r0 = band_i * _GEN_BAND_STRIDE
        tiles, output_tile = _gen_band_tiles(rng)
        paste(out, output_tile, r0, 0)

        positions = rng.permutation(len(_GEN_TILE_COLS))
        for tile, pos_i in zip(tiles, positions):
            paste(inp, tile, r0, _GEN_TILE_COLS[int(pos_i)])

    try:
        if np.array_equal(main(inp), out):
            return inp
    except AssertionError:
        return None
    return None


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRY_LIMIT):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("failed to generate a valid b0722778 input")
