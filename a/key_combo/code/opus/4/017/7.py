from __future__ import annotations

import random

import numpy as np


BLACK = 0
GREEN = 3
TILE_COLOR_POOL = (1, 2, 4, 5, 6, 7, 8, 9)

GEN_RETRIES = 400
GRID_SIDE_RANGE = (10, 12)
TOKEN_COUNTS = (2, 3, 4)
TILE_SIDES = (3, 4)
OFFSETS = (0, 1)


def _neighbors4(shape: tuple[int, int], r: int, c: int):
    h, w = map(int, shape)
    if r > 0:
        yield r - 1, c
    if r + 1 < h:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < w:
        yield r, c + 1


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    """4-connected components of a boolean mask, following arc_utils (r, c) conventions."""
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r, c in np.argwhere(mask):
        rr = int(r)
        cc = int(c)
        if seen[rr, cc]:
            continue
        stack = [(rr, cc)]
        seen[rr, cc] = True
        coords: list[tuple[int, int]] = []
        while stack:
            cr, cc = stack.pop()
            coords.append((int(cr), int(cc)))
            for nr, nc in _neighbors4(mask.shape, cr, cc):
                if mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    stack.append((int(nr), int(nc)))
        comps.append(coords)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rs = [r for r, _ in coords]
    cs = [c for _, c in coords]
    return min(rs), min(cs), max(rs), max(cs)


def _crop_bbox(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = map(int, bbox)
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _paste(dst: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    h, w = map(int, patch.shape)
    dst[int(r0) : int(r0) + h, int(c0) : int(c0) + w] = patch


def _rotate_clockwise(tile: np.ndarray, quarter_turns: int) -> np.ndarray:
    return np.rot90(tile, -int(quarter_turns))


def _extract_tile_and_token_count(grid: np.ndarray) -> tuple[np.ndarray, int]:
    fg_components = _components(grid != BLACK)
    multi = [comp for comp in fg_components if len(comp) > 1]
    assert multi, "expected one multi-pixel tile"

    areas = sorted((len(comp) for comp in multi), reverse=True)
    assert len(areas) == 1 or areas[0] > areas[1], "largest tile must be unique"
    tile_coords = max(multi, key=len)
    tile = _crop_bbox(grid, _bbox(tile_coords))
    assert tile.shape[0] == tile.shape[1], "tile bbox must be square"

    token_count = 0
    for comp in fg_components:
        if len(comp) != 1:
            continue
        r, c = comp[0]
        assert int(grid[r, c]) == GREEN, "every singleton foreground pixel must be a green token"
        token_count += 1

    assert token_count in TOKEN_COUNTS, "token count must determine NxN output size"
    return tile, token_count


def _expected_output(tile: np.ndarray, n: int) -> np.ndarray:
    side = int(tile.shape[0])
    out = np.full((side * int(n), side * int(n)), BLACK, dtype=int)
    for i in range(int(n)):
        for j in range(int(n)):
            rot = _rotate_clockwise(tile, (i + j) % 4)
            _paste(out, rot, i * side, j * side)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Count the isolated green tokens, tile the square sprite into an NxN grid, and rotate each copy by its tile coordinates.

    Input structure:
    - The input is a black background grid containing one multicolor square tile near the upper-left corner.
    - That tile is the unique largest 4-connected non-black component and occupies a square bbox of side 3 or 4.
    - Elsewhere, the remaining non-black pixels are isolated green singleton components.

    Rule:
    - Find the largest multi-pixel 4-connected foreground component and crop its square bbox to obtain the base tile.
    - Count the isolated green singleton components; call this count N.
    - Produce an (N*S) x (N*S) black grid, where S is the tile side length, partitioned into an NxN array of SxS slots.
    - In slot (i, j), place the base tile rotated clockwise by ((i + j) mod 4) quarter-turns.

    Key insight:
    - The green dots provide only the global tiling density, while each tile's local diagonal position encodes its own rotation.
    """
    g = np.asarray(grid, dtype=int)
    tile, n = _extract_tile_and_token_count(g)
    return _expected_output(tile, n)


def _all_rotations_distinct(tile: np.ndarray) -> bool:
    return len({_rotate_clockwise(tile, k).tobytes() for k in range(4)}) == 4


def _make_tile() -> np.ndarray:
    for _ in range(200):
        side = random.choice(TILE_SIDES)
        palette = random.sample(TILE_COLOR_POOL, random.randint(2, min(4, side + 1)))
        tile = np.array(
            [[random.choice(palette) for _ in range(side)] for _ in range(side)],
            dtype=int,
        )
        if len(set(map(int, tile.ravel()))) < 2:
            continue
        if not _all_rotations_distinct(tile):
            continue
        return tile
    raise RuntimeError("failed to generate a non-symmetric tile")


def _can_place_token(grid: np.ndarray, r: int, c: int) -> bool:
    if int(grid[r, c]) != BLACK:
        return False
    h, w = grid.shape
    for rr in range(max(0, r - 1), min(h, r + 2)):
        for cc in range(max(0, c - 1), min(w, c + 2)):
            if int(grid[rr, cc]) != BLACK:
                return False
    return True


def _place_tokens(grid: np.ndarray, n: int) -> bool:
    h, w = grid.shape
    cells = [(r, c) for r in range(h) for c in range(w)]
    random.shuffle(cells)
    placed = 0
    for r, c in cells:
        if not _can_place_token(grid, r, c):
            continue
        grid[r, c] = GREEN
        placed += 1
        if placed == int(n):
            return True
    return False


def _validate_scene(grid: np.ndarray, tile: np.ndarray, token_count: int) -> bool:
    try:
        extracted_tile, counted = _extract_tile_and_token_count(grid)
    except Exception:
        return False
    if counted != int(token_count):
        return False
    if not np.array_equal(extracted_tile, tile):
        return False
    expected = _expected_output(tile, token_count)
    try:
        actual = main(grid)
    except Exception:
        return False
    if not np.array_equal(actual, expected):
        return False
    if max(grid.shape) > 30 or max(expected.shape) > 30:
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    n = random.choice(TOKEN_COUNTS)
    tile = _make_tile()
    side = int(tile.shape[0])
    out_side = side * int(n)
    if out_side > 30:
        return None

    rows = random.randint(*GRID_SIDE_RANGE)
    cols = random.randint(*GRID_SIDE_RANGE)
    grid = np.full((rows, cols), BLACK, dtype=int)

    tile_r0 = random.choice(OFFSETS)
    tile_c0 = random.choice(OFFSETS)
    if tile_r0 + side > rows or tile_c0 + side > cols:
        return None
    _paste(grid, tile, tile_r0, tile_c0)

    if not _place_tokens(grid, n):
        return None
    if not _validate_scene(grid, tile, n):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 7")
