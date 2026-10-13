from __future__ import annotations

import random
from typing import Sequence

import numpy as np


BG = 0
FG_COLORS = tuple(range(1, 10))
MAX_GENERATION_ATTEMPTS = 400


def _background_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _intervals(mask: np.ndarray) -> list[tuple[int, int]]:
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return []
    splits = np.where(np.diff(idx) > 1)[0] + 1
    groups = np.split(idx, splits)
    return [(int(group[0]), int(group[-1])) for group in groups]


def _extract_bands(grid: np.ndarray, bg: int) -> list[dict]:
    row_spans = _intervals(np.any(grid != bg, axis=1))
    bands = []
    for r0, r1 in row_spans:
        band = grid[r0 : r1 + 1, :]
        col_spans = _intervals(np.any(band != bg, axis=0))
        assert len(col_spans) == 3, "each band must contain exactly three tiles"
        widths = {c1 - c0 + 1 for c0, c1 in col_spans}
        assert len(widths) == 1, "all tiles in a band must share one width"
        tiles = [band[:, c0 : c1 + 1].copy() for c0, c1 in col_spans]
        bands.append(
            {
                "rows": (r0, r1),
                "cols": col_spans,
                "tiles": tiles,
            }
        )
    assert bands, "expected at least one content band"
    return bands


def _axis_signatures(tile: np.ndarray, axis: int) -> list[tuple[int, ...]]:
    if axis == 0:
        return [tuple(map(int, row)) for row in tile]
    return [tuple(map(int, col)) for col in tile.T]


def _apply_permutation(tile: np.ndarray, axis: int, perm: Sequence[int]) -> np.ndarray:
    order = np.asarray(perm, dtype=int)
    if axis == 0:
        return tile[order, :].copy()
    return tile[:, order].copy()


def _permutation_if_match(left: np.ndarray, right: np.ndarray, axis: int) -> tuple[int, ...] | None:
    if left.shape != right.shape:
        return None
    left_sig = _axis_signatures(left, axis)
    right_sig = _axis_signatures(right, axis)
    if len(set(left_sig)) != len(left_sig):
        return None
    if set(left_sig) != set(right_sig):
        return None
    pos = {sig: idx for idx, sig in enumerate(left_sig)}
    perm = tuple(pos[sig] for sig in right_sig)
    if perm == tuple(range(len(perm))):
        return None
    if not np.array_equal(_apply_permutation(left, axis, perm), right):
        return None
    return perm


def _relation_candidates(tiles: Sequence[np.ndarray]) -> list[dict]:
    out = []
    for left_idx in range(3):
        for right_idx in range(left_idx + 1, 3):
            for axis in (0, 1):
                perm = _permutation_if_match(tiles[left_idx], tiles[right_idx], axis)
                if perm is None:
                    continue
                other_idx = next(idx for idx in range(3) if idx not in (left_idx, right_idx))
                out.append(
                    {
                        "pair": (left_idx, right_idx),
                        "axis": axis,
                        "perm": perm,
                        "other": other_idx,
                    }
                )
    return out


def _unique_relation(tiles: Sequence[np.ndarray]) -> dict:
    candidates = _relation_candidates(tiles)
    assert len(candidates) == 1, "expected one unique demonstration pair"
    return candidates[0]


def _solve_with_metadata(grid: np.ndarray) -> tuple[np.ndarray, list[dict]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _background_color(g)
    bands = _extract_bands(g, bg)

    out = g.copy()
    metadata = []
    for band in bands:
        relation = _unique_relation(band["tiles"])
        other = relation["other"]
        transformed = _apply_permutation(band["tiles"][other], relation["axis"], relation["perm"])
        r0, r1 = band["rows"]
        c0, c1 = band["cols"][other]
        out[r0 : r1 + 1, c0 : c1 + 1] = transformed
        metadata.append(
            {
                "rows": band["rows"],
                "cols": band["cols"],
                "axis": relation["axis"],
                "pair": relation["pair"],
                "perm": relation["perm"],
                "other": other,
                "before": band["tiles"][other],
                "after": transformed,
            }
        )
    return out, metadata


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use the matched tile pair in each horizontal band to infer a row or column permutation, then apply it to the remaining tile.

    Input structure:
    - Background-only separator rows split the grid into horizontal bands.
    - Each band contains three same-sized multicolor rectangular tiles at aligned column positions.
    - In each band, exactly one pair of tiles has the same set of rows or the same set of columns, but in a different order.

    Rule:
    - In every band, find the unique matching pair and determine whether their relation is a row permutation or a column permutation.
    - Read the permutation from the left member of that pair to the right member of that pair.
    - Apply the same permutation on the same axis to the third tile, leaving the rest of the grid unchanged.

    Key insight:
    - The matching pair teaches a geometric reordering, not a color mapping; the untouched tile in the same band must undergo the same permutation along the inferred axis.
    """
    out, _metadata = _solve_with_metadata(grid)
    return out


def _random_non_identity_perm(length: int) -> tuple[int, ...]:
    perm = list(range(length))
    while True:
        random.shuffle(perm)
        if any(idx != value for idx, value in enumerate(perm)):
            return tuple(perm)


def _random_tile(height: int, width: int) -> np.ndarray:
    while True:
        palette_size = random.randint(3, min(6, len(FG_COLORS)))
        palette = random.sample(FG_COLORS, palette_size)
        tile = np.random.choice(palette, size=(height, width)).astype(int)
        if len(np.unique(tile)) >= 3:
            return tile


def _make_band_tiles(height: int, width: int, axis: int, pair: tuple[int, int]) -> tuple[list[np.ndarray], tuple[int, ...]]:
    perm_length = height if axis == 0 else width
    for _ in range(200):
        perm = _random_non_identity_perm(perm_length)
        src = _random_tile(height, width)
        dst = _apply_permutation(src, axis, perm)
        if _permutation_if_match(src, dst, axis) != perm:
            continue
        if _permutation_if_match(src, dst, 1 - axis) is not None:
            continue

        other_idx = next(idx for idx in range(3) if idx not in pair)
        other = _random_tile(height, width)
        transformed_other = _apply_permutation(other, axis, perm)
        if np.array_equal(other, transformed_other):
            continue
        if np.array_equal(other, src) or np.array_equal(other, dst):
            continue
        if np.array_equal(transformed_other, src) or np.array_equal(transformed_other, dst):
            continue

        tiles = [None, None, None]
        tiles[pair[0]] = src
        tiles[pair[1]] = dst
        tiles[other_idx] = other
        candidates = _relation_candidates(tiles)
        if len(candidates) != 1:
            continue
        relation = candidates[0]
        if relation["pair"] != pair or relation["axis"] != axis or relation["perm"] != perm:
            continue
        return [tile.copy() for tile in tiles], perm
    raise RuntimeError("failed to generate a valid band")


def _band_axes(band_count: int) -> list[int]:
    axes = [0, 1]
    while len(axes) < band_count:
        axes.append(random.randint(0, 1))
    random.shuffle(axes)
    return axes


def _specs_for_grid() -> tuple[int, int, int, int, int, int, list[dict]]:
    band_count = random.randint(2, 4)
    tile_h = random.randint(3, 5)
    tile_w = random.randint(3, 6)
    row_gap = random.randint(1, 2)
    col_gap = random.randint(1, 3)
    top_margin = random.randint(1, 2)
    left_margin = random.randint(1, 2)

    height = top_margin + band_count * tile_h + (band_count - 1) * row_gap + top_margin
    width = left_margin + 3 * tile_w + 2 * col_gap + left_margin
    assert height <= 30 and width <= 30

    pairs = [(0, 1), (0, 2), (1, 2)]
    specs = []
    for axis in _band_axes(band_count):
        pair = random.choice(pairs)
        tiles, perm = _make_band_tiles(tile_h, tile_w, axis, pair)
        specs.append({"axis": axis, "pair": pair, "perm": perm, "tiles": tiles})
    return tile_h, tile_w, row_gap, col_gap, top_margin, left_margin, specs


def _build_grid_from_specs(
    tile_h: int,
    tile_w: int,
    row_gap: int,
    col_gap: int,
    top_margin: int,
    left_margin: int,
    specs: Sequence[dict],
) -> np.ndarray:
    band_count = len(specs)
    height = top_margin + band_count * tile_h + (band_count - 1) * row_gap + top_margin
    width = left_margin + 3 * tile_w + 2 * col_gap + left_margin
    grid = np.full((height, width), BG, dtype=int)

    col_starts = [
        left_margin,
        left_margin + tile_w + col_gap,
        left_margin + 2 * (tile_w + col_gap),
    ]
    for band_idx, spec in enumerate(specs):
        r0 = top_margin + band_idx * (tile_h + row_gap)
        for tile_idx, c0 in enumerate(col_starts):
            grid[r0 : r0 + tile_h, c0 : c0 + tile_w] = spec["tiles"][tile_idx]
    return grid


def _validate_generated_grid(grid: np.ndarray, specs: Sequence[dict]) -> None:
    input_bands = _extract_bands(grid, BG)
    analyzed_out, metadata = _solve_with_metadata(grid)
    direct_out = main(grid)
    assert np.array_equal(analyzed_out, direct_out)
    assert len(metadata) == len(specs)

    output_bands = _extract_bands(analyzed_out, BG)
    for band_idx, (spec, meta, out_band) in enumerate(zip(specs, metadata, output_bands)):
        assert meta["axis"] == spec["axis"]
        assert meta["pair"] == spec["pair"]
        assert meta["perm"] == spec["perm"]
        assert not np.array_equal(meta["before"], meta["after"])
        expected_after = _apply_permutation(meta["before"], meta["axis"], meta["perm"])
        assert np.array_equal(meta["after"], expected_after)
        for idx in meta["pair"]:
            assert np.array_equal(out_band["tiles"][idx], spec["tiles"][idx])
        assert np.array_equal(out_band["tiles"][meta["other"]], expected_after)
        assert len(_relation_candidates(input_bands[band_idx]["tiles"])) == 1


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        try:
            tile_h, tile_w, row_gap, col_gap, top_margin, left_margin, specs = _specs_for_grid()
            grid = _build_grid_from_specs(tile_h, tile_w, row_gap, col_gap, top_margin, left_margin, specs)
            _validate_generated_grid(grid, specs)
        except (AssertionError, RuntimeError):
            continue
        return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
