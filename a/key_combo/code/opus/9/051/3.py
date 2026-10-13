from __future__ import annotations

import random
from functools import lru_cache

import numpy as np


BLACK = 0
BLUE = 1
GREEN = 3
GREY = 5

PRIMITIVES: dict[str, tuple[tuple[int, int], ...]] = {
    "square": ((0, 0), (0, 1), (1, 0), (1, 1)),
    "hbar": ((0, 0), (0, 1), (0, 2)),
    "vbar": ((0, 0), (1, 0), (2, 0)),
}

GRID_HEIGHT_RANGE = (13, 18)
GRID_WIDTH_RANGE = (13, 18)
COMPONENT_COUNT_CHOICES = (2, 3, 4)
COMPONENT_TILE_COUNT_CHOICES = (2, 3, 4, 5)
LOCAL_CANVAS = 11
MAX_COMPONENT_SPAN = 8
GENERATE_RETRIES = 400


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _neighbors8(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return tuple((r + dr, c + dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if dr or dc)


def _gray_components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    components: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            components.append(sorted(comp))
    return components


def _tile_from_anchor(shape: str, r0: int, c0: int) -> tuple[tuple[int, int], ...]:
    return tuple(sorted((r0 + dr, c0 + dc) for dr, dc in PRIMITIVES[shape]))


def _tile_kind(tile: tuple[tuple[int, int], ...]) -> str:
    rows = sorted({r for r, _ in tile})
    cols = sorted({c for _, c in tile})
    h = rows[-1] - rows[0] + 1
    w = cols[-1] - cols[0] + 1
    if (h, w) == (2, 2):
        return "square"
    if (h, w) == (1, 3):
        return "hbar"
    if (h, w) == (3, 1):
        return "vbar"
    raise AssertionError("unexpected primitive shape")


def _tile_family(tile: tuple[tuple[int, int], ...]) -> str:
    return "square" if _tile_kind(tile) == "square" else "bar"


def _tile_top_left(tile: tuple[tuple[int, int], ...]) -> tuple[int, int]:
    return min(r for r, _ in tile), min(c for _, c in tile)


def _tile_parity(tile: tuple[tuple[int, int], ...]) -> tuple[int, int]:
    r0, c0 = _tile_top_left(tile)
    return r0 % 2, c0 % 2


def _touches_union(tile: tuple[tuple[int, int], ...], occupied: set[tuple[int, int]]) -> bool:
    return any(nb in occupied for cell in tile for nb in _neighbors4(*cell))


def _adjacent_tiles(a: tuple[tuple[int, int], ...], b: tuple[tuple[int, int], ...]) -> bool:
    bset = set(b)
    return any(nb in bset for cell in a for nb in _neighbors4(*cell))


def _component_bbox(cells: set[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _normalize_cells(cells: set[tuple[int, int]]) -> tuple[set[tuple[int, int]], tuple[int, int]]:
    r0, c0, _r1, _c1 = _component_bbox(cells)
    shifted = {(r - r0, c - c0) for r, c in cells}
    return shifted, (r0, c0)


def _decompose_component_unique(cells: list[tuple[int, int]]) -> list[tuple[tuple[int, int], ...]]:
    cellset = frozenset((int(r), int(c)) for r, c in cells)

    def _covering_tiles(cell: tuple[int, int], remaining: frozenset[tuple[int, int]]) -> list[tuple[tuple[int, int], ...]]:
        r, c = cell
        out: set[tuple[tuple[int, int], ...]] = set()
        for r0 in (r - 1, r):
            for c0 in (c - 1, c):
                tile = _tile_from_anchor("square", r0, c0)
                if all(rc in remaining for rc in tile):
                    out.add(tile)
        for c0 in (c - 2, c - 1, c):
            tile = _tile_from_anchor("hbar", r, c0)
            if all(rc in remaining for rc in tile):
                out.add(tile)
        for r0 in (r - 2, r - 1, r):
            tile = _tile_from_anchor("vbar", r0, c)
            if all(rc in remaining for rc in tile):
                out.add(tile)
        return sorted(out)

    @lru_cache(maxsize=None)
    def _search(remaining: frozenset[tuple[int, int]]) -> tuple[tuple[tuple[tuple[int, int], ...], ...], ...]:
        if not remaining:
            return ((),)
        cell = min(remaining)
        solutions: list[tuple[tuple[tuple[int, int], ...], ...]] = []
        for tile in _covering_tiles(cell, remaining):
            next_remaining = frozenset(remaining.difference(tile))
            for suffix in _search(next_remaining):
                solutions.append((tile,) + suffix)
                if len(solutions) >= 2:
                    return tuple(solutions)
        return tuple(solutions)

    solutions = _search(cellset)
    assert len(solutions) == 1, "gray component must have a unique primitive decomposition"
    return list(solutions[0])


def _extract_primitives(grid: np.ndarray) -> tuple[tuple[int, int], list[list[tuple[tuple[int, int], ...]]]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    palette = set(map(int, np.unique(g)))
    assert palette <= {BLACK, GREEN, GREY}, "unexpected colors"

    marker_coords = np.argwhere(g == GREEN)
    assert marker_coords.shape == (1, 2), "expected exactly one green marker pixel"
    marker = (int(marker_coords[0, 0]), int(marker_coords[0, 1]))

    components = _gray_components(g == GREY)
    assert 2 <= len(components) <= 4, "expected 2-4 gray composite shapes"
    decomposed = [_decompose_component_unique(comp) for comp in components]
    return marker, decomposed


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decompose each gray composite shape into primitive tiles, then keep only the tiles whose top-left parity matches the green marker.

    Input structure:
    - The grid is mostly black background with 2-4 disconnected gray composite shapes.
    - Each gray composite is made by edge-joining 2x2 squares and/or 1x3 or 3x1 rectangles, but the primitive boundaries are not drawn.
    - Exactly one green marker pixel sits away from the gray shapes and provides the reference parity.

    Rule:
    - Uniquely decompose every gray connected component into non-overlapping primitive tiles of sizes 2x2, 1x3, or 3x1.
    - Read the green marker coordinate `(r_m, c_m)` and its parity pair `(r_m % 2, c_m % 2)`.
    - For each primitive tile, take its top-left corner `(r_t, c_t)` from the tile's minimum row and minimum column.
    - If `(r_t % 2, c_t % 2)` matches the marker parity, recolor the whole tile blue; otherwise erase the whole tile to black.
    - Erase the green marker itself to black.

    Key insight:
    - The parity test applies to recovered primitive tiles, not to whole gray blobs and not to individual pixels, so decomposition determines which neighboring pixels survive together.
    """
    marker, components = _extract_primitives(grid)
    parity = (marker[0] % 2, marker[1] % 2)

    out = np.zeros_like(np.asarray(grid, dtype=int))
    for tiles in components:
        for tile in tiles:
            if _tile_parity(tile) == parity:
                for r, c in tile:
                    out[r, c] = BLUE
    return out


def _component_to_array(cells: set[tuple[int, int]]) -> np.ndarray:
    normalized, _ = _normalize_cells(cells)
    r0, c0, r1, c1 = _component_bbox(normalized)
    arr = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=int)
    for r, c in normalized:
        arr[r, c] = GREY
    return arr


def _component_good_classes(
    tiles: list[tuple[tuple[int, int], ...]]
) -> list[tuple[int, int]]:
    good: list[tuple[int, int]] = []
    n = len(tiles)
    adj_pairs = [(i, j) for i in range(n) for j in range(i + 1, n) if _adjacent_tiles(tiles[i], tiles[j])]
    for parity in ((0, 0), (0, 1), (1, 0), (1, 1)):
        matches = [i for i, tile in enumerate(tiles) if _tile_parity(tile) == parity]
        k = len(matches)
        if not (1 <= k < n and abs(2 * k - n) <= 1):
            continue
        matched = set(matches)
        if any((i in matched) != (j in matched) for i, j in adj_pairs):
            good.append(parity)
    return good


def _enumerate_component_additions(
    occupied: set[tuple[int, int]],
) -> list[tuple[str, int, int, tuple[tuple[int, int], ...]]]:
    options: list[tuple[str, int, int, tuple[tuple[int, int], ...]]] = []
    for shape in random.sample(list(PRIMITIVES), k=len(PRIMITIVES)):
        cells0 = PRIMITIVES[shape]
        h = max(r for r, _ in cells0) + 1
        w = max(c for _, c in cells0) + 1
        for r0 in range(LOCAL_CANVAS - h + 1):
            for c0 in range(LOCAL_CANVAS - w + 1):
                tile = _tile_from_anchor(shape, r0, c0)
                tile_set = set(tile)
                if tile_set & occupied:
                    continue
                if occupied and not _touches_union(tile, occupied):
                    continue
                bbox_cells = occupied | tile_set
                br0, bc0, br1, bc1 = _component_bbox(bbox_cells)
                if br1 - br0 + 1 > MAX_COMPONENT_SPAN or bc1 - bc0 + 1 > MAX_COMPONENT_SPAN:
                    continue
                options.append((shape, r0, c0, tile))
    return options


def _generate_component() -> tuple[np.ndarray, list[tuple[tuple[int, int], ...]], tuple[int, int]] | None:
    tile_count = random.choice(COMPONENT_TILE_COUNT_CHOICES)
    occupied: set[tuple[int, int]] = set()

    for idx in range(tile_count):
        options = _enumerate_component_additions(occupied)
        if not options:
            return None
        if idx == 0:
            options = [opt for opt in options if opt[1] in (4, 5) and opt[2] in (4, 5)]
        if not options:
            return None
        shape_weights = {"square": 3, "hbar": 2, "vbar": 2}
        weights = [shape_weights[shape] for shape, _r0, _c0, _tile in options]
        shape, r0, c0, tile = random.choices(options, weights=weights, k=1)[0]
        del shape, r0, c0
        occupied.update(tile)

    normalized_cells, _ = _normalize_cells(occupied)
    try:
        tiles = _decompose_component_unique(sorted(normalized_cells))
    except AssertionError:
        return None
    if len(tiles) != tile_count:
        return None
    families = {_tile_family(tile) for tile in tiles}
    if families != {"square", "bar"}:
        return None
    good = _component_good_classes(tiles)
    if not good:
        return None
    arr = _component_to_array(normalized_cells)
    return arr, tiles, random.choice(good)


def _dilate8(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    out = mask.copy()
    coords = np.argwhere(mask)
    for r, c in coords:
        rr = int(r)
        cc = int(c)
        for nr, nc in _neighbors8(rr, cc):
            if 0 <= nr < h and 0 <= nc < w:
                out[nr, nc] = True
    return out


def _placement_candidates(
    occupied: np.ndarray,
    patch: np.ndarray,
    required_parity: tuple[int, int],
) -> list[tuple[int, int]]:
    h, w = occupied.shape
    ph, pw = patch.shape
    blocked = _dilate8(occupied)
    mask = patch == GREY
    candidates: list[tuple[int, int]] = []
    for r0 in range(h - ph + 1):
        if r0 % 2 != required_parity[0]:
            continue
        for c0 in range(w - pw + 1):
            if c0 % 2 != required_parity[1]:
                continue
            if np.any(blocked[r0 : r0 + ph, c0 : c0 + pw] & mask):
                continue
            candidates.append((r0, c0))
    random.shuffle(candidates)
    return candidates


def _place_patch(grid: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    mask = patch == GREY
    sub = grid[r0 : r0 + patch.shape[0], c0 : c0 + patch.shape[1]]
    sub[mask] = GREY


def _marker_candidates(grid: np.ndarray, parity: tuple[int, int]) -> list[tuple[int, int]]:
    h, w = grid.shape
    candidates: list[tuple[int, int]] = []
    gray = grid == GREY
    for r in range(1, h - 1):
        if r % 2 != parity[0]:
            continue
        for c in range(1, w - 1):
            if c % 2 != parity[1] or grid[r, c] != BLACK:
                continue
            if any(gray[nr, nc] for nr, nc in _neighbors8(r, c)):
                continue
            if any(grid[nr, nc] != BLACK for nr, nc in _neighbors4(r, c)):
                continue
            candidates.append((r, c))
    random.shuffle(candidates)
    return candidates


def _semantic_checks(inp: np.ndarray) -> None:
    marker, components = _extract_primitives(inp)
    marker_parity = (marker[0] % 2, marker[1] % 2)
    out = main(inp)

    total_gray_tiles = 0
    has_adjacent_opposite_fate = False
    families_by_fate = {True: set(), False: set()}
    for tiles in components:
        total_gray_tiles += len(tiles)
        kept = [tile for tile in tiles if _tile_parity(tile) == marker_parity]
        erased = [tile for tile in tiles if _tile_parity(tile) != marker_parity]
        assert 1 <= len(kept) < len(tiles)
        assert abs(len(kept) - len(erased)) <= 1
        for tile in kept:
            families_by_fate[True].add(_tile_family(tile))
        for tile in erased:
            families_by_fate[False].add(_tile_family(tile))
        for i in range(len(tiles)):
            for j in range(i + 1, len(tiles)):
                if _adjacent_tiles(tiles[i], tiles[j]) and (_tile_parity(tiles[i]) == marker_parity) != (_tile_parity(tiles[j]) == marker_parity):
                    has_adjacent_opposite_fate = True

    assert total_gray_tiles >= 4
    assert {"square", "bar"} <= (families_by_fate[True] | families_by_fate[False])
    assert families_by_fate[True] & families_by_fate[False]
    assert has_adjacent_opposite_fate
    assert 0 < int(np.count_nonzero(out == BLUE)) < int(np.count_nonzero(inp == GREY))
    assert not np.array_equal(inp, out)


def generate_input() -> np.ndarray:
    for _ in range(GENERATE_RETRIES):
        h = random.randint(*GRID_HEIGHT_RANGE)
        w = random.randint(*GRID_WIDTH_RANGE)
        grid = np.zeros((h, w), dtype=int)
        marker_parity = random.choice(((0, 0), (0, 1), (1, 0), (1, 1)))

        component_count = random.choice(COMPONENT_COUNT_CHOICES)
        components: list[tuple[np.ndarray, list[tuple[tuple[int, int], ...]], tuple[int, int]]] = []
        for _idx in range(component_count):
            comp = _generate_component()
            if comp is None:
                break
            components.append(comp)
        if len(components) != component_count:
            continue

        occupied = np.zeros((h, w), dtype=bool)
        placed = True
        components.sort(key=lambda item: int(np.count_nonzero(item[0] == GREY)), reverse=True)
        for patch, _tiles, local_keep_parity in components:
            required_parity = (
                marker_parity[0] ^ local_keep_parity[0],
                marker_parity[1] ^ local_keep_parity[1],
            )
            candidates = _placement_candidates(occupied, patch, required_parity)
            if not candidates:
                placed = False
                break
            r0, c0 = candidates[0]
            _place_patch(grid, patch, r0, c0)
            occupied |= grid == GREY
        if not placed:
            continue

        marker_options = _marker_candidates(grid, marker_parity)
        if not marker_options:
            continue
        mr, mc = marker_options[0]
        grid[mr, mc] = GREEN

        try:
            _semantic_checks(grid)
        except AssertionError:
            continue
        return grid

    raise RuntimeError("failed to generate a valid ARC input")
