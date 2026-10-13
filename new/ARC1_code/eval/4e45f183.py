from __future__ import annotations

from collections import Counter
from typing import Callable

import numpy as np

from arc_utils import Color, colors_present, paste
import random


CoordMap = Callable[[int, int], tuple[int, int]]

SLOTS = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1))
D4_MAPS: tuple[CoordMap, ...] = (
    lambda r, c: (r, c),
    lambda r, c: (-c, r),
    lambda r, c: (-r, -c),
    lambda r, c: (c, -r),
    lambda r, c: (r, -c),
    lambda r, c: (-r, c),
    lambda r, c: (c, r),
    lambda r, c: (-c, -r),
)


def _transform_mask(mask: np.ndarray, coord_map: CoordMap) -> np.ndarray:
    h, w = mask.shape
    assert h == w and h % 2 == 1, "fragment masks must be odd square tiles"
    center = h // 2

    out = np.zeros_like(mask, dtype=bool)
    for r, c in np.argwhere(mask):
        nr, nc = coord_map(int(r) - center, int(c) - center)
        out[nr + center, nc + center] = True
    return out


def _slot_from_mask(mask: np.ndarray) -> tuple[int, int]:
    coords = np.argwhere(mask)
    assert coords.size, "only nonblank masks have slots"
    center = mask.shape[0] // 2
    row_sum = int((coords[:, 0] - center).sum())
    col_sum = int((coords[:, 1] - center).sum())
    return (
        0 if row_sum == 0 else (-1 if row_sum < 0 else 1),
        0 if col_sum == 0 else (-1 if col_sum < 0 else 1),
    )


def _unique_mask(candidates: list[np.ndarray], slot: tuple[int, int]) -> np.ndarray:
    assert candidates, f"no fragment can generate slot {slot}"
    reference = candidates[0]
    assert all(np.array_equal(reference, candidate) for candidate in candidates), (
        f"inconsistent symmetry reconstructions for slot {slot}"
    )
    return reference


def main(grid: np.ndarray) -> np.ndarray:
    """
    Sort the 5x5 marker fragments into the 3x3 slots indicated by their orientations and complete any missing symmetric fragments.

    Input structure
    The grid is a 19x19 canvas split by black separator rows and columns into a 3x3 array of 5x5 tiles.
    Inside the tiles there are exactly two nonblack colors: a frequent fill color and a less frequent marker color.
    Nonblank marker masks are fragments whose centroid direction relative to the 5x5 center names one of the nine slots.

    Rule
    Classify each nonblank marker mask by the signs of its row and column offsets from the tile center: top/center/bottom and left/center/right.
    For every output slot, collect all dihedral symmetries of the known fragments that would land in that slot, assert they agree, and use that mask.
    Rebuild the same separated 3x3 tile canvas with each completed mask in its classified slot.

    Key insight
    The input is a shuffled library of symmetric tile fragments; each fragment's centroid direction is its target position.
    """
    g = np.asarray(grid, dtype=int)
    assert g.shape == (19, 19), "expected a 19x19 separated tile canvas"

    separators = (0, 6, 12, 18)
    assert np.all(g[list(separators), :] == Color.BLACK), "separator rows must be black"
    assert np.all(g[:, list(separators)] == Color.BLACK), "separator columns must be black"

    fg_colors = colors_present(g, exclude={Color.BLACK})
    assert len(fg_colors) == 2, "expected exactly two nonblack tile colors"
    counts = Counter(int(v) for v in g.ravel() if int(v) != Color.BLACK)
    assert counts[fg_colors[0]] != counts[fg_colors[1]], "fill and marker colors must be frequency-distinct"
    fill_color, marker_color = (color for color, _ in counts.most_common())

    known: dict[tuple[int, int], np.ndarray] = {}
    for tile_r in range(3):
        for tile_c in range(3):
            r0 = 1 + 6 * tile_r
            c0 = 1 + 6 * tile_c
            tile = g[r0 : r0 + 5, c0 : c0 + 5]
            assert set(map(int, np.unique(tile))).issubset({fill_color, marker_color})

            mask = tile == marker_color
            if not mask.any():
                continue
            slot = _slot_from_mask(mask)
            assert slot in SLOTS
            if slot in known:
                assert np.array_equal(known[slot], mask), f"duplicate slot {slot} must have the same mask"
            known[slot] = mask
    assert known, "at least one nonblank fragment is required"

    completed: dict[tuple[int, int], np.ndarray] = {}
    for slot in SLOTS:
        candidates = []
        for src_slot, src_mask in known.items():
            for coord_map in D4_MAPS:
                if coord_map(*src_slot) == slot:
                    candidates.append(_transform_mask(src_mask, coord_map))
        completed[slot] = _unique_mask(candidates, slot)

    out = np.zeros_like(g)
    for tile_r, slot_r in enumerate((-1, 0, 1)):
        for tile_c, slot_c in enumerate((-1, 0, 1)):
            tile = np.full((5, 5), fill_color, dtype=int)
            tile[completed[(slot_r, slot_c)]] = marker_color
            r0 = 1 + 6 * tile_r
            c0 = 1 + 6 * tile_c
            out[r0 : r0 + 5, c0 : c0 + 5] = tile
    return out

# --------------- generator (o5x) ---------------

# Grid/tile geometry.
_GEN_TILE_SIZE = 5
_GEN_TILE_SHAPE = (_GEN_TILE_SIZE, _GEN_TILE_SIZE)
_GEN_TILE_STRIDE, _GEN_TILE_OFFSET, _GEN_GRID_TILES = 6, 1, 3
_GEN_CANVAS_SHAPE = (19, 19)
_GEN_SLOT_VALUES = (-1, 0, 1)
_GEN_SLOTS = SLOTS
_GEN_TILE_POSITIONS = tuple((r, c) for r in range(_GEN_GRID_TILES) for c in range(_GEN_GRID_TILES))

# Color palette.
_GEN_COLOR_RANGE = range(1, 10)

# Corner and edge seed bitspaces.
_GEN_CORNER_BITS_MIN = 1
_GEN_CORNER_BITS_MAX = 7
_GEN_CORNER_BIT_GROUPS = (
    ((0, 0),),
    ((0, 1), (1, 0)),
    ((1, 1),),
)
_GEN_EDGE_FREE_ROWS = range(2)
_GEN_EDGE_FREE_COLS = (0, 1, 2)
_GEN_EDGE_BITS_MIN = 1
_GEN_EDGE_BITS_STOP = 1 << (len(_GEN_EDGE_FREE_ROWS) * len(_GEN_EDGE_FREE_COLS))
_GEN_EDGE_MIN_CELLS = 2
_GEN_EDGE_MAX_CELLS = 6

# D4 orbit sources.
_GEN_D4_MAPS = D4_MAPS
_GEN_CENTER_SLOT, _GEN_CORNER_SRC_SLOT, _GEN_EDGE_SRC_SLOT = (0, 0), (-1, -1), (-1, 0)

# Shuffle/retry limits.
_GEN_MAX_FIXED_TILES = 3
_GEN_RETRY_LIMIT = 300


def _gen_edge_bit_cell_count(bits: int) -> int:
    cells = 0
    bit_i = 0
    for _ in _GEN_EDGE_FREE_ROWS:
        for c in _GEN_EDGE_FREE_COLS:
            if bits & (1 << bit_i):
                cells += 1 if c == _GEN_TILE_SIZE // 2 else 2
            bit_i += 1
    return cells


_GEN_EDGE_BIT_CHOICES = tuple(
    bits
    for bits in range(_GEN_EDGE_BITS_MIN, _GEN_EDGE_BITS_STOP)
    if _GEN_EDGE_MIN_CELLS <= _gen_edge_bit_cell_count(bits) <= _GEN_EDGE_MAX_CELLS
)


def _gen_tile_origin(tile_r: int, tile_c: int) -> tuple[int, int]:
    return _GEN_TILE_OFFSET + _GEN_TILE_STRIDE * tile_r, _GEN_TILE_OFFSET + _GEN_TILE_STRIDE * tile_c


def _gen_tile_view(grid: np.ndarray, tile_r: int, tile_c: int) -> np.ndarray:
    r0, c0 = _gen_tile_origin(tile_r, tile_c)
    return grid[r0 : r0 + _GEN_TILE_SIZE, c0 : c0 + _GEN_TILE_SIZE]


def _gen_transform_mask(mask: np.ndarray, coord_map: CoordMap) -> np.ndarray:
    center = mask.shape[0] // 2
    out = np.zeros_like(mask, dtype=bool)
    for r, c in np.argwhere(mask):
        nr, nc = coord_map(int(r) - center, int(c) - center)
        out[nr + center, nc + center] = True
    return out


def _gen_corner_seed() -> np.ndarray:
    bits = random.randint(_GEN_CORNER_BITS_MIN, _GEN_CORNER_BITS_MAX)
    mask = np.zeros(_GEN_TILE_SHAPE, dtype=bool)
    for bit_i, cells in enumerate(_GEN_CORNER_BIT_GROUPS):
        if bits & (1 << bit_i):
            for r, c in cells:
                mask[r, c] = True
    return mask


def _gen_edge_seed() -> np.ndarray:
    bits = random.choice(_GEN_EDGE_BIT_CHOICES)
    mask = np.zeros(_GEN_TILE_SHAPE, dtype=bool)
    bit_i = 0
    for r in _GEN_EDGE_FREE_ROWS:
        for c in _GEN_EDGE_FREE_COLS:
            if bits & (1 << bit_i):
                mask[r, c] = True
                mask[r, _GEN_TILE_SIZE - 1 - c] = True
            bit_i += 1
    return mask


def _gen_orbit_mask(seed: np.ndarray, src_slot: tuple[int, int], dst_slot: tuple[int, int]) -> np.ndarray:
    for coord_map in _GEN_D4_MAPS:
        if coord_map(*src_slot) == dst_slot:
            return _gen_transform_mask(seed, coord_map)
    raise AssertionError(f"no D4 map from {src_slot} to {dst_slot}")


def _gen_slot_masks(corner_seed: np.ndarray, edge_seed: np.ndarray) -> dict[tuple[int, int], np.ndarray]:
    masks: dict[tuple[int, int], np.ndarray] = {}
    center_mask = np.zeros(_GEN_TILE_SHAPE, dtype=bool)
    for coord_map in _GEN_D4_MAPS:
        center_mask |= _gen_transform_mask(corner_seed, coord_map)

    for slot in _GEN_SLOTS:
        if slot == _GEN_CENTER_SLOT:
            masks[slot] = center_mask
        elif slot[0] != 0 and slot[1] != 0:
            masks[slot] = _gen_orbit_mask(corner_seed, _GEN_CORNER_SRC_SLOT, slot)
        else:
            masks[slot] = _gen_orbit_mask(edge_seed, _GEN_EDGE_SRC_SLOT, slot)
    return masks


def _gen_build_output(slot_masks: dict[tuple[int, int], np.ndarray], fill_color: int, marker_color: int) -> np.ndarray:
    out = np.zeros(_GEN_CANVAS_SHAPE, dtype=int)
    for tile_r, slot_r in enumerate(_GEN_SLOT_VALUES):
        for tile_c, slot_c in enumerate(_GEN_SLOT_VALUES):
            tile = np.full(_GEN_TILE_SHAPE, fill_color, dtype=int)
            tile[slot_masks[(slot_r, slot_c)]] = marker_color
            paste(out, tile, *_gen_tile_origin(tile_r, tile_c))
    return out


def _gen_shuffled_input(out: np.ndarray) -> np.ndarray | None:
    sources = list(_GEN_TILE_POSITIONS)
    random.shuffle(sources)
    if sum(src == dst for src, dst in zip(sources, _GEN_TILE_POSITIONS)) > _GEN_MAX_FIXED_TILES:
        return None

    inp = np.zeros_like(out)
    for (dst_r, dst_c), (src_r, src_c) in zip(_GEN_TILE_POSITIONS, sources):
        paste(inp, _gen_tile_view(out, src_r, src_c), *_gen_tile_origin(dst_r, dst_c))
    return inp


def _gen_attempt() -> np.ndarray | None:
    fill_color, marker_color = random.sample(_GEN_COLOR_RANGE, 2)
    corner_seed = _gen_corner_seed()
    edge_seed = _gen_edge_seed()
    slot_masks = _gen_slot_masks(corner_seed, edge_seed)
    out = _gen_build_output(slot_masks, fill_color, marker_color)
    inp = _gen_shuffled_input(out)
    if inp is None or not np.array_equal(main(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid 4e45f183 input")
