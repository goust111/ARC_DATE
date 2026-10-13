from __future__ import annotations

from collections import Counter
import itertools
import random

import numpy as np


BG = 0
PALETTE = tuple(range(1, 10))
MAX_ATTEMPTS = 400


def get_bg(grid: np.ndarray) -> int:
    """Mirror arc_utils.get_bg: return the most common color."""
    return int(Counter(map(int, np.asarray(grid).ravel())).most_common(1)[0][0])


def colors_present(grid: np.ndarray, *, exclude: set[int] | None = None) -> list[int]:
    vals = set(map(int, np.asarray(grid).ravel()))
    if exclude is not None:
        vals -= set(map(int, exclude))
    return sorted(vals)


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def crop_bbox(grid: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    return np.asarray(grid)[r0 : r1 + 1, c0 : c1 + 1].copy()


def components_mask(mask: np.ndarray) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _split_bands(grid: np.ndarray, bg: int) -> list[tuple[int, int]]:
    bands: list[tuple[int, int]] = []
    start: int | None = None
    for r in range(grid.shape[0]):
        is_sep = bool(np.all(grid[r] == bg))
        if is_sep:
            if start is not None:
                bands.append((start, r - 1))
                start = None
        elif start is None:
            start = r
    if start is not None:
        bands.append((start, grid.shape[0] - 1))
    return bands


def _band_tiles(band: np.ndarray, bg: int) -> list[tuple[tuple[int, int, int, int], np.ndarray]]:
    tiles: list[tuple[tuple[int, int, int, int], np.ndarray]] = []
    for coords in components_mask(band != bg):
        box = bbox_coords(coords)
        tiles.append((box, crop_bbox(band, box)))
    tiles.sort(key=lambda item: (item[0][1], item[0][0]))
    return tiles


def _derive_mapping(left: np.ndarray, right: np.ndarray, bg: int) -> dict[int, int]:
    assert left.shape == right.shape, "paired tiles must have matching shapes"
    left_mask = left != bg
    right_mask = right != bg
    assert np.array_equal(left_mask, right_mask), "paired tiles must share the same non-background mask"

    mapping: dict[int, int] = {}
    for src in colors_present(left, exclude={bg}):
        vals = np.unique(right[left == src])
        vals = [int(v) for v in vals if int(v) != bg]
        assert len(vals) == 1, "each source color must map consistently"
        mapping[int(src)] = vals[0]

    assert set(mapping.keys()) == set(colors_present(left, exclude={bg}))
    assert set(mapping.values()) == set(colors_present(right, exclude={bg}))
    return mapping


def _apply_mapping(tile: np.ndarray, mapping: dict[int, int]) -> np.ndarray:
    out = np.asarray(tile, dtype=int).copy()
    for src, dst in mapping.items():
        out[tile == int(src)] = int(dst)
    return out


def main(input_grid: np.ndarray) -> np.ndarray:
    """
    Learn each left-to-right tile pair as a color dictionary, then relay-translate the query tile until no dictionary still matches its palette.

    Input structure:
    - Full background rows split the canvas into horizontal bands.
    - Every non-query band contains two separated tiles laid out left-to-right; their non-background shapes are identical but their colors differ.
    - The remaining band contains exactly one query tile, and the vertical order of all bands is arbitrary.

    Rule:
    - For each paired band, crop the left and right tiles and infer a color replacement dictionary from corresponding cells.
    - Start from the cropped query tile. If one dictionary's source palette exactly equals the tile's current non-background palette, recolor the whole tile with that dictionary.
    - Keep chaining dictionaries this way until no source palette matches the current palette, then output only the final translated query tile.

    Key insight:
    - The usable dictionary order is determined by palette equality between one pair's output colors and the next pair's input colors, not by the bands' top-to-bottom placement.
    """
    g = np.asarray(input_grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    bg = get_bg(g)
    bands = _split_bands(g, bg)
    assert bands, "expected at least one non-empty band"

    dictionaries: dict[frozenset[int], dict[int, int]] = {}
    query_tile: np.ndarray | None = None
    for r0, r1 in bands:
        band = g[r0 : r1 + 1, :]
        tiles = _band_tiles(band, bg)
        if len(tiles) == 1:
            assert query_tile is None, "expected exactly one query band"
            query_tile = tiles[0][1]
        else:
            assert len(tiles) == 2, "each example band must contain exactly two tiles"
            mapping = _derive_mapping(tiles[0][1], tiles[1][1], bg)
            src_palette = frozenset(mapping.keys())
            assert src_palette not in dictionaries, "source palettes must uniquely identify dictionaries"
            dictionaries[src_palette] = mapping

    assert query_tile is not None, "missing query tile"
    current = query_tile.copy()
    seen: set[frozenset[int]] = set()
    while True:
        palette = frozenset(colors_present(current, exclude={bg}))
        assert palette not in seen, "dictionary chaining should terminate without cycles"
        seen.add(palette)
        mapping = dictionaries.get(palette)
        if mapping is None:
            return current
        current = _apply_mapping(current, mapping)


def _random_palette_sequence(rng: random.Random, chain_len: int, palette_size: int) -> list[frozenset[int]]:
    all_sets = [frozenset(combo) for combo in itertools.combinations(PALETTE, palette_size)]
    rng.shuffle(all_sets)
    return all_sets[: chain_len + 1]


def _random_mapping(
    src_palette: frozenset[int], dst_palette: frozenset[int], rng: random.Random
) -> dict[int, int]:
    src = list(src_palette)
    dst = list(dst_palette)
    rng.shuffle(src)
    rng.shuffle(dst)
    mapping = {int(s): int(d) for s, d in zip(src, dst)}
    assert set(mapping.keys()) == set(src_palette)
    assert set(mapping.values()) == set(dst_palette)
    return mapping


def _make_tile(palette: frozenset[int], rng: random.Random) -> np.ndarray:
    h = rng.randint(3, 5)
    w = rng.randint(4, 7)
    colors = list(palette)
    k = len(colors)
    seed_cells = rng.sample([(r, c) for r in range(h) for c in range(w)], k)
    tile = np.zeros((h, w), dtype=int)
    for r in range(h):
        for c in range(w):
            distances = [abs(r - sr) + abs(c - sc) for sr, sc in seed_cells]
            tile[r, c] = int(colors[int(np.argmin(distances))])
    return tile


def _pack_pair(left: np.ndarray, right: np.ndarray, rng: random.Random) -> np.ndarray:
    assert left.shape == right.shape
    gap = rng.randint(2, 4)
    h, w = left.shape
    band = np.full((h, 2 * w + gap), BG, dtype=int)
    band[:, :w] = left
    band[:, w + gap : w + gap + w] = right
    return band


def _assemble_grid(bands: list[np.ndarray], rng: random.Random) -> np.ndarray:
    height = sum(int(b.shape[0]) for b in bands) + len(bands) - 1
    width = max(int(b.shape[1]) for b in bands)
    extra_width = rng.randint(0, max(0, 30 - width))
    width += extra_width
    grid = np.full((height, width), BG, dtype=int)
    r = 0
    for index, band in enumerate(bands):
        h, w = map(int, band.shape)
        c0 = rng.randint(0, width - w)
        grid[r : r + h, c0 : c0 + w] = band
        r += h
        if index != len(bands) - 1:
            r += 1
    return grid


def _expected_query_output(query_tile: np.ndarray, mappings: list[dict[int, int]]) -> np.ndarray:
    out = np.asarray(query_tile, dtype=int).copy()
    for mapping in mappings:
        out = _apply_mapping(out, mapping)
    return out


def _generate_case(rng: random.Random) -> tuple[np.ndarray, np.ndarray]:
    chain_len = rng.randint(2, 4)
    palette_size = rng.choice((2, 3))
    palettes = _random_palette_sequence(rng, chain_len, palette_size)
    mappings = [_random_mapping(palettes[i], palettes[i + 1], rng) for i in range(chain_len)]

    query_tile = _make_tile(palettes[0], rng)
    expected = _expected_query_output(query_tile, mappings)

    bands: list[np.ndarray] = []
    for i in range(chain_len):
        left = _make_tile(palettes[i], rng)
        right = _apply_mapping(left, mappings[i])
        bands.append(_pack_pair(left, right, rng))
    bands.append(query_tile.copy())
    rng.shuffle(bands)

    grid = _assemble_grid(bands, rng)
    assert max(grid.shape) <= 30
    assert max(expected.shape) <= 30

    source_palettes = [frozenset(mapping.keys()) for mapping in mappings]
    assert len(set(source_palettes)) == len(source_palettes)
    assert len(mappings) >= 2
    assert frozenset(colors_present(query_tile, exclude={BG})) == source_palettes[0]

    solved = main(grid)
    assert np.array_equal(solved, expected), "generated case does not satisfy the intended semantics"
    return grid, expected


def generate_input() -> np.ndarray:
    for seed in range(MAX_ATTEMPTS):
        rng = random.Random(random.randrange(10**9) + seed)
        try:
            grid, _expected = _generate_case(rng)
            return grid
        except AssertionError:
            continue
    raise RuntimeError("failed to generate a valid non-degenerate puzzle instance")
