from __future__ import annotations

import itertools
import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
YELLOW = 4
GREY = 5

DIRECTIONS = ("up", "down", "left", "right")
SIZES = (3, 4)


def _bbox_of_true(mask: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(mask)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _mask_from_bitmask(size: int, bitmask: int) -> np.ndarray:
    mask = np.zeros((size, size), dtype=bool)
    for idx in range(size * size):
        if (bitmask >> idx) & 1:
            mask[idx // size, idx % size] = True
    return mask


def _valid_pattern_mask(mask: np.ndarray) -> bool:
    size = int(mask.shape[0])
    if mask.shape != (size, size):
        return False

    fg_count = int(mask.sum())
    if not (2 <= fg_count <= 5):
        return False

    fg_rows = np.flatnonzero(mask.any(axis=1))
    fg_cols = np.flatnonzero(mask.any(axis=0))
    if fg_rows.size <= 1 or fg_cols.size <= 1:
        return False

    r0, c0, r1, c1 = _bbox_of_true(mask)
    bbox_area = (r1 - r0 + 1) * (c1 - c0 + 1)
    if bbox_area - fg_count < 2:
        return False

    bg_count = size * size - fg_count
    return bg_count > 0


def _expand_complement_tiling(mask: np.ndarray) -> np.ndarray:
    source = mask.astype(int)
    tile = (~mask).astype(int)
    return np.kron(source, tile) * BLUE


def _compress_line_to_edge(values: np.ndarray, toward_start: bool) -> np.ndarray:
    src = np.asarray(values, dtype=int)
    line = np.zeros_like(src)
    sources = np.flatnonzero(src == BLUE)
    count = int(sources.size)
    if toward_start:
        finals = np.arange(count, dtype=int)
    else:
        finals = np.arange(src.size - count, src.size, dtype=int)

    if count:
        line[finals] = BLUE
        final_set = {int(idx) for idx in finals}
        for idx in sources:
            if int(idx) not in final_set:
                line[int(idx)] = RED
    return line


def _compress_expanded_grid(expanded: np.ndarray, direction: str) -> np.ndarray:
    g = np.asarray(expanded, dtype=int)
    out = np.zeros_like(g)

    if direction == "left":
        for r in range(g.shape[0]):
            out[r, :] = _compress_line_to_edge(g[r, :], toward_start=True)
    elif direction == "right":
        for r in range(g.shape[0]):
            out[r, :] = _compress_line_to_edge(g[r, :], toward_start=False)
    elif direction == "up":
        for c in range(g.shape[1]):
            out[:, c] = _compress_line_to_edge(g[:, c], toward_start=True)
    elif direction == "down":
        for c in range(g.shape[1]):
            out[:, c] = _compress_line_to_edge(g[:, c], toward_start=False)
    else:
        raise AssertionError(f"unknown direction: {direction}")

    return out


def _find_separator(grid: np.ndarray) -> tuple[str, int]:
    row_hits = [r for r in range(grid.shape[0]) if np.all(grid[r, :] == GREY)]
    col_hits = [c for c in range(grid.shape[1]) if np.all(grid[:, c] == GREY)]
    assert len(row_hits) + len(col_hits) == 1, "expected exactly one grey separator line"

    if row_hits:
        return "horizontal", int(row_hits[0])
    return "vertical", int(col_hits[0])


def _direction_from_marker(marker_region: np.ndarray) -> str:
    marker_cells = np.argwhere(marker_region == YELLOW)
    assert marker_cells.shape == (1, 2), "expected exactly one yellow marker"
    size = int(marker_region.shape[0])
    mr, mc = map(int, marker_cells[0])

    is_top = mr == 0 and 0 < mc < size - 1
    is_bottom = mr == size - 1 and 0 < mc < size - 1
    is_left = mc == 0 and 0 < mr < size - 1
    is_right = mc == size - 1 and 0 < mr < size - 1
    assert sum([is_top, is_bottom, is_left, is_right]) == 1, "marker must sit on one non-corner edge"

    if is_top:
        return "up"
    if is_bottom:
        return "down"
    if is_left:
        return "left"
    return "right"


def _parse_input(grid: np.ndarray) -> tuple[np.ndarray, str]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert set(np.unique(g)).issubset({BLACK, BLUE, YELLOW, GREY})

    sep_kind, sep_idx = _find_separator(g)
    if sep_kind == "vertical":
        region_a = g[:, :sep_idx]
        region_b = g[:, sep_idx + 1 :]
    else:
        region_a = g[:sep_idx, :]
        region_b = g[sep_idx + 1 :, :]

    assert region_a.shape == region_b.shape
    assert region_a.shape[0] == region_a.shape[1]
    size = int(region_a.shape[0])
    assert size in SIZES

    regions = [region_a, region_b]
    pattern_candidates = [reg for reg in regions if np.any(reg == BLUE)]
    marker_candidates = [reg for reg in regions if np.any(reg == YELLOW)]
    assert len(pattern_candidates) == 1 and len(marker_candidates) == 1

    pattern_region = pattern_candidates[0]
    marker_region = marker_candidates[0]
    assert set(np.unique(pattern_region)).issubset({BLACK, BLUE})
    assert set(np.unique(marker_region)).issubset({BLACK, YELLOW})

    mask = pattern_region == BLUE
    assert _valid_pattern_mask(mask)
    direction = _direction_from_marker(marker_region)
    return mask, direction


def main(grid: np.ndarray) -> np.ndarray:
    """Expand the blue pattern by stamping its complement, then compress the expanded pixels toward the yellow marker's edge and mark vacated cells red.

    Input structure:
    - One grey full row or full column splits the input into two square `N x N` regions, where `N` is 3 or 4.
    - One region is a black/blue pattern: black background with 2-5 blue pixels, not all in one row or one column.
    - The other region is a black marker panel with exactly one yellow pixel on a non-corner edge cell.

    Rule:
    - Read the blue pattern as a binary `N x N` mask and its black cells as the complementary tile mask.
    - Build an `(N*N) x (N*N)` expansion: every blue source cell receives one `N x N` tile whose blue cells are exactly the pattern's original black positions; every black source cell receives an all-black block.
    - Use the yellow marker's edge position to choose the compression direction: top/bottom/left/right means compress up/down/left/right.
    - Compress every row or every column of the expanded grid toward that edge; final occupied cells stay blue, and original blue cells that become empty turn red as swept trail.

    Key insight:
    - The small pattern determines the content of each stamped tile, while the separate yellow edge marker determines only how the fully expanded result is packed, so the answer is the composition of two independent clues in sequence.
    """
    mask, direction = _parse_input(grid)
    expanded = _expand_complement_tiling(mask)
    return _compress_expanded_grid(expanded, direction)


def _output_signature(grid: np.ndarray) -> tuple[tuple[int, int], bytes]:
    return tuple(map(int, grid.shape)), grid.tobytes()


def _enumerate_unique_specs() -> dict[str, dict[int, list[tuple[int, int, str]]]]:
    by_signature: dict[tuple[tuple[int, int], bytes], list[tuple[int, int, str]]] = {}

    for size in SIZES:
        cell_count = size * size
        for fg_count in range(2, 6):
            for cells in itertools.combinations(range(cell_count), fg_count):
                bitmask = 0
                for idx in cells:
                    bitmask |= 1 << idx
                mask = _mask_from_bitmask(size, bitmask)
                if not _valid_pattern_mask(mask):
                    continue

                expanded = _expand_complement_tiling(mask)
                for direction in DIRECTIONS:
                    compressed = _compress_expanded_grid(expanded, direction)
                    if not np.any(compressed == RED):
                        continue
                    by_signature.setdefault(_output_signature(compressed), []).append((size, bitmask, direction))

    unique_specs = [specs[0] for specs in by_signature.values() if len(specs) == 1]
    assert unique_specs, "expected at least one unique pattern-direction spec"

    pools = {direction: {size: [] for size in SIZES} for direction in DIRECTIONS}
    for spec in unique_specs:
        pools[spec[2]][spec[0]].append(spec)

    for direction in DIRECTIONS:
        for size in SIZES:
            specs = pools[direction][size]
            assert specs, f"expected non-empty pool for {(direction, size)}"
            random.shuffle(specs)
    return pools


_SPEC_POOLS = _enumerate_unique_specs()
_SPEC_INDEX = {direction: {size: 0 for size in SIZES} for direction in DIRECTIONS}
_COMBO_ORDER = [(direction, size) for direction in DIRECTIONS for size in SIZES]
random.shuffle(_COMBO_ORDER)
_GENERATION_COUNT = 0


def _next_combo() -> tuple[str, int]:
    global _GENERATION_COUNT
    if _GENERATION_COUNT > 0 and _GENERATION_COUNT % len(_COMBO_ORDER) == 0:
        random.shuffle(_COMBO_ORDER)
    direction, size = _COMBO_ORDER[_GENERATION_COUNT % len(_COMBO_ORDER)]
    _GENERATION_COUNT += 1
    return direction, size


def _next_spec(direction: str, size: int) -> tuple[int, int, str]:
    specs = _SPEC_POOLS[direction][size]
    idx = _SPEC_INDEX[direction][size]
    spec = specs[idx % len(specs)]
    _SPEC_INDEX[direction][size] = idx + 1
    return spec


def _marker_region(size: int, direction: str) -> np.ndarray:
    region = np.zeros((size, size), dtype=int)
    edge_pos = random.randint(1, size - 2)
    if direction == "up":
        region[0, edge_pos] = YELLOW
    elif direction == "down":
        region[size - 1, edge_pos] = YELLOW
    elif direction == "left":
        region[edge_pos, 0] = YELLOW
    elif direction == "right":
        region[edge_pos, size - 1] = YELLOW
    else:
        raise AssertionError(f"unknown direction: {direction}")
    return region


def _assemble_input(mask: np.ndarray, direction: str) -> np.ndarray:
    size = int(mask.shape[0])
    pattern_region = np.where(mask, BLUE, BLACK).astype(int)
    marker_region = _marker_region(size, direction)

    separator_vertical = bool(random.getrandbits(1))
    pattern_first = bool(random.getrandbits(1))

    if separator_vertical:
        parts = [pattern_region, marker_region] if pattern_first else [marker_region, pattern_region]
        return np.concatenate([parts[0], np.full((size, 1), GREY, dtype=int), parts[1]], axis=1)

    parts = [pattern_region, marker_region] if pattern_first else [marker_region, pattern_region]
    return np.concatenate([parts[0], np.full((1, size), GREY, dtype=int), parts[1]], axis=0)


def _oracle_from_spec(size: int, bitmask: int, direction: str) -> np.ndarray:
    mask = _mask_from_bitmask(size, bitmask)
    expanded = _expand_complement_tiling(mask)
    return _compress_expanded_grid(expanded, direction)


def generate_input() -> np.ndarray:
    direction, size = _next_combo()
    _, bitmask, _ = _next_spec(direction, size)
    mask = _mask_from_bitmask(size, bitmask)
    grid = _assemble_input(mask, direction)

    expected = _oracle_from_spec(size, bitmask, direction)
    actual = main(grid)
    assert np.array_equal(actual, expected)
    assert max(grid.shape) <= 30 and max(expected.shape) <= 30
    return grid
