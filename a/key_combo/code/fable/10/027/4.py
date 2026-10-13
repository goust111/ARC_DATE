from __future__ import annotations

import random

import numpy as np


BACKGROUND = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
PINK = 6
ORANGE = 7
TEAL = 8
MAROON = 9

TILE_SIZE = 3
MACRO_SIZE = 3
GRID_SIZE = MACRO_SIZE * TILE_SIZE + (MACRO_SIZE - 1)
SEPARATORS = (3, 7)
PALETTE = (BLUE, GREEN, PINK, ORANGE, TEAL, MAROON)
LOCAL_CELLS = [(r, c) for r in range(TILE_SIZE) for c in range(TILE_SIZE)]
MACRO_CELLS = [(r, c) for r in range(MACRO_SIZE) for c in range(MACRO_SIZE)]
GEN_RETRIES = 500


def _bands(separators: list[int], size: int) -> list[tuple[int, int]]:
    starts = [0] + [s + 1 for s in separators]
    ends = separators + [size]
    bands = list(zip(starts, ends))
    assert all(lo < hi for lo, hi in bands)
    return bands


def _band_index(value: int, bands: list[tuple[int, int]]) -> int:
    hits = [idx for idx, (lo, hi) in enumerate(bands) if lo <= value < hi]
    assert len(hits) == 1
    return hits[0]


def _tile_slice(br: int, bc: int) -> tuple[slice, slice]:
    r0 = br * (TILE_SIZE + 1)
    c0 = bc * (TILE_SIZE + 1)
    return slice(r0, r0 + TILE_SIZE), slice(c0, c0 + TILE_SIZE)


def _step_kind(src: tuple[int, int], dst: tuple[int, int]) -> str:
    dr = dst[0] - src[0]
    dc = dst[1] - src[1]
    return "diag" if dr != 0 and dc != 0 else "card"


def _neighbors(pos: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = pos
    out = []
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            nr, nc = r + dr, c + dc
            if 0 <= nr < MACRO_SIZE and 0 <= nc < MACRO_SIZE:
                out.append((nr, nc))
    random.shuffle(out)
    return out


def _parse_layout(grid: np.ndarray) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    h, w = map(int, grid.shape)
    row_seps = [r for r in range(h) if np.all(grid[r, :] == GREY)]
    col_seps = [c for c in range(w) if np.all(grid[:, c] == GREY)]
    assert len(row_seps) == len(col_seps) == 2
    row_bands = _bands(row_seps, h)
    col_bands = _bands(col_seps, w)
    assert len(row_bands) == len(col_bands) == MACRO_SIZE
    assert {hi - lo for lo, hi in row_bands} == {TILE_SIZE}
    assert {hi - lo for lo, hi in col_bands} == {TILE_SIZE}

    non_sep = np.ones_like(grid, dtype=bool)
    non_sep[row_seps, :] = False
    non_sep[:, col_seps] = False
    assert not np.any(grid[non_sep] == GREY)
    return row_bands, col_bands


def _extract_tile(
    grid: np.ndarray,
    row_bands: list[tuple[int, int]],
    col_bands: list[tuple[int, int]],
    br: int,
    bc: int,
) -> np.ndarray:
    r0, r1 = row_bands[br]
    c0, c1 = col_bands[bc]
    tile = grid[r0:r1, c0:c1]
    assert tile.shape == (TILE_SIZE, TILE_SIZE)
    return tile


def _yellow_local(tile: np.ndarray) -> tuple[int, int]:
    yellow = np.argwhere(tile == YELLOW)
    assert yellow.shape == (1, 2)
    return int(yellow[0, 0]), int(yellow[0, 1])


def _follow_chain(grid: np.ndarray) -> tuple[list[tuple[int, int]], np.ndarray]:
    row_bands, col_bands = _parse_layout(grid)
    red_coords = np.argwhere(grid == RED)
    assert red_coords.shape == (1, 2)

    for br, bc in MACRO_CELLS:
        tile = _extract_tile(grid, row_bands, col_bands, br, bc)
        _yellow_local(tile)

    red_r, red_c = map(int, red_coords[0])
    start = (_band_index(red_r, row_bands), _band_index(red_c, col_bands))
    start_tile = _extract_tile(grid, row_bands, col_bands, *start)
    assert np.count_nonzero(start_tile == RED) == 1

    seen: set[tuple[int, int]] = set()
    path: list[tuple[int, int]] = []
    pos = start
    while True:
        assert pos not in seen, "pointer chain must not cycle"
        seen.add(pos)
        path.append(pos)
        tile = _extract_tile(grid, row_bands, col_bands, *pos)
        yr, yc = _yellow_local(tile)
        dr, dc = yr - 1, yc - 1
        if (dr, dc) == (0, 0):
            return path, tile.copy()
        nr, nc = pos[0] + dr, pos[1] + dc
        assert 0 <= nr < MACRO_SIZE and 0 <= nc < MACRO_SIZE
        pos = (nr, nc)


def main(grid: np.ndarray) -> np.ndarray:
    """Follow the yellow local-coordinate pointer chain from the red start tile and crop out the terminal tile.

    Input structure:
    - Full grey separator rows and columns split the canvas into a 3x3 macro-grid of 3x3 tiles.
    - Every tile contains exactly one yellow marker pixel inside a small colored pattern.
    - Exactly one tile also contains a red pixel; that tile is the unique chain start.

    Rule:
    - Read each tile's yellow local position `(r, c)` as the macro displacement `(r-1, c-1)`.
    - Start at the tile containing the red pixel and repeatedly move to the neighboring tile named by the current yellow displacement.
    - Stop when the current tile's yellow pixel is at the tile center, then output that terminal tile's 3x3 contents.

    Key insight:
    - The yellow pixel is not the answer itself; its local coordinates encode a pointer, and only the chain reached from the unique red start is semantically relevant.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape == (GRID_SIZE, GRID_SIZE)
    _, terminal = _follow_chain(g)
    return terminal


def _blank_grid() -> np.ndarray:
    grid = np.full((GRID_SIZE, GRID_SIZE), BACKGROUND, dtype=int)
    grid[list(SEPARATORS), :] = GREY
    grid[:, list(SEPARATORS)] = GREY
    return grid


def _path_mode_ok(path: list[tuple[int, int]], mode: str) -> bool:
    kinds = {_step_kind(a, b) for a, b in zip(path, path[1:])}
    if mode == "mixed":
        return kinds == {"card", "diag"}
    if mode == "card_only":
        return kinds == {"card"}
    if mode == "diag_only":
        return kinds == {"diag"}
    raise ValueError(mode)


def _search_path(
    path: list[tuple[int, int]],
    target_nodes: int,
    mode: str,
) -> list[tuple[int, int]] | None:
    if len(path) == target_nodes:
        return path.copy() if _path_mode_ok(path, mode) else None

    current = path[-1]
    for nxt in _neighbors(current):
        if nxt in path:
            continue
        kind = _step_kind(current, nxt)
        if mode == "card_only" and kind != "card":
            continue
        if mode == "diag_only" and kind != "diag":
            continue
        path.append(nxt)
        found = _search_path(path, target_nodes, mode)
        if found is not None:
            return found
        path.pop()
    return None


def _sample_chain_path() -> list[tuple[int, int]] | None:
    modes = [("mixed", (4, 6)), ("card_only", (3, 6)), ("diag_only", (3, 5))]
    mode, (lo, hi) = random.choice(modes)
    node_count = random.randint(lo, hi)
    starts = MACRO_CELLS[:]
    random.shuffle(starts)
    for start in starts:
        found = _search_path([start], node_count, mode)
        if found is not None:
            return found
    return None


def _pointer_local_from_step(src: tuple[int, int], dst: tuple[int, int]) -> tuple[int, int]:
    return dst[0] - src[0] + 1, dst[1] - src[1] + 1


def _signature(tile: np.ndarray) -> bytes:
    return tile.astype(np.int8, copy=False).tobytes()


def _random_tile(
    yellow_local: tuple[int, int],
    *,
    is_start: bool,
    min_pattern: int,
    max_pattern: int,
) -> np.ndarray:
    tile = np.zeros((TILE_SIZE, TILE_SIZE), dtype=int)
    tile[yellow_local] = YELLOW
    occupied = {yellow_local}

    if is_start:
        red_pos = random.choice([cell for cell in LOCAL_CELLS if cell not in occupied])
        tile[red_pos] = RED
        occupied.add(red_pos)

    pattern_count = random.randint(min_pattern, max_pattern)
    slots = random.sample([cell for cell in LOCAL_CELLS if cell not in occupied], k=pattern_count)
    colors = random.sample(PALETTE, k=pattern_count)
    for pos, color in zip(slots, colors):
        tile[pos] = color
    return tile


def _build_tiles(path: list[tuple[int, int]]) -> dict[tuple[int, int], np.ndarray] | None:
    chain_set = set(path)
    free = [pos for pos in MACRO_CELLS if pos not in chain_set]
    if len(path) < 3 or len(free) < 2:
        return None

    min_centers = 1
    max_centers = min(2, len(free) - 1)
    if max_centers < min_centers:
        return None
    extra_centers = set(random.sample(free, k=random.randint(min_centers, max_centers)))

    pointer_locals: dict[tuple[int, int], tuple[int, int]] = {}
    for src, dst in zip(path, path[1:]):
        pointer_locals[src] = _pointer_local_from_step(src, dst)
    pointer_locals[path[-1]] = (1, 1)

    for pos in free:
        if pos in extra_centers:
            pointer_locals[pos] = (1, 1)
            continue
        moves = []
        chain_moves = []
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                nr, nc = pos[0] + dr, pos[1] + dc
                if 0 <= nr < MACRO_SIZE and 0 <= nc < MACRO_SIZE:
                    move = (dr + 1, dc + 1)
                    moves.append(move)
                    if (nr, nc) in chain_set:
                        chain_moves.append(move)
        pool = chain_moves if chain_moves and random.random() < 0.8 else moves
        pointer_locals[pos] = random.choice(pool)

    tiles: dict[tuple[int, int], np.ndarray] = {}
    seen: set[bytes] = set()
    start = path[0]
    terminal = path[-1]
    for pos in MACRO_CELLS:
        for _ in range(100):
            is_start = pos == start
            min_pattern = 2 if pos in (start, terminal) else 1
            max_pattern = 3
            tile = _random_tile(pointer_locals[pos], is_start=is_start, min_pattern=min_pattern, max_pattern=max_pattern)
            sig = _signature(tile)
            if sig in seen:
                continue
            seen.add(sig)
            tiles[pos] = tile
            break
        else:
            return None
    return tiles


def _render_grid(tiles: dict[tuple[int, int], np.ndarray]) -> np.ndarray:
    grid = _blank_grid()
    for (br, bc), tile in tiles.items():
        rs, cs = _tile_slice(br, bc)
        grid[rs, cs] = tile
    return grid


def _center_tile_count(tiles: dict[tuple[int, int], np.ndarray]) -> int:
    return sum(int(tile[1, 1] == YELLOW) for tile in tiles.values())


def _semantic_checks(grid: np.ndarray, path: list[tuple[int, int]], tiles: dict[tuple[int, int], np.ndarray]) -> bool:
    try:
        solved = main(grid)
        traced_path, terminal = _follow_chain(grid)
    except Exception:
        return False

    if traced_path != path:
        return False
    if not np.array_equal(solved, terminal):
        return False
    if not np.array_equal(terminal, tiles[path[-1]]):
        return False
    if np.count_nonzero(grid == RED) != 1:
        return False
    if sum(np.count_nonzero(tile == YELLOW) for tile in tiles.values()) != len(tiles):
        return False
    if len({_signature(tile) for tile in tiles.values()}) != len(tiles):
        return False
    if _center_tile_count(tiles) < 2:
        return False

    free = [pos for pos in MACRO_CELLS if pos not in set(path)]
    if not any(tiles[pos][1, 1] == YELLOW for pos in free):
        return False
    if not any(tiles[pos][1, 1] != YELLOW for pos in free):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    path = _sample_chain_path()
    if path is None:
        return None
    tiles = _build_tiles(path)
    if tiles is None:
        return None
    grid = _render_grid(tiles)
    if not _semantic_checks(grid, path, tiles):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid pointer-chain ARC input")
