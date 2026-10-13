from __future__ import annotations

from collections import deque
import random

import numpy as np


BLACK = 0
RED = 2
YELLOW = 4
GREY = 5

Coord = tuple[int, int]


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _neighbors8(r: int, c: int):
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            yield r + dr, c + dc


def _normalize_shape(shape: tuple[Coord, ...]) -> tuple[Coord, ...]:
    min_r = min(r for r, _ in shape)
    min_c = min(c for _, c in shape)
    return tuple(sorted((r - min_r, c - min_c) for r, c in shape))


def _transform_shape(shape: tuple[Coord, ...], rotations: int, flip: bool) -> tuple[Coord, ...]:
    pts = list(shape)
    if flip:
        pts = [(r, -c) for r, c in pts]
    for _ in range(rotations % 4):
        pts = [(c, -r) for r, c in pts]
    return _normalize_shape(tuple(pts))


def _build_shape_bank() -> dict[int, list[tuple[Coord, ...]]]:
    bases = [
        ((0, 0),),
        ((0, 0), (0, 1)),
        ((0, 0), (0, 1), (0, 2)),
        ((0, 0), (1, 0), (1, 1)),
        ((0, 0), (0, 1), (0, 2), (0, 3)),
        ((0, 0), (0, 1), (1, 0), (1, 1)),
        ((0, 0), (1, 0), (2, 0), (2, 1)),
        ((0, 0), (0, 1), (0, 2), (1, 1)),
        ((0, 1), (0, 2), (1, 0), (1, 1)),
    ]
    by_area: dict[int, set[tuple[Coord, ...]]] = {}
    for base in bases:
        area = len(base)
        by_area.setdefault(area, set())
        for rot in range(4):
            for flip in (False, True):
                by_area[area].add(_transform_shape(base, rot, flip))
    return {area: sorted(shapes) for area, shapes in by_area.items()}


SHAPES_BY_AREA = _build_shape_bank()


def _shape_size(shape: tuple[Coord, ...]) -> tuple[int, int]:
    return max(r for r, _ in shape) + 1, max(c for _, c in shape) + 1


def _shift_shape(shape: tuple[Coord, ...], top: int, left: int) -> tuple[Coord, ...]:
    return tuple((top + r, left + c) for r, c in shape)


def _paint(grid: np.ndarray, coords: set[Coord] | tuple[Coord, ...], color: int) -> None:
    for r, c in coords:
        grid[r, c] = color


def _red_components(grid: np.ndarray) -> list[tuple[Coord, ...]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    blocks: list[tuple[Coord, ...]] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or grid[r, c] != RED:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[Coord] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if not _in_bounds(grid.shape, nr, nc):
                        continue
                    if seen[nr, nc] or grid[nr, nc] != RED:
                        continue
                    seen[nr, nc] = True
                    q.append((nr, nc))
            blocks.append(tuple(sorted(coords)))
    return blocks


def _adjacent_lit_cells(block: tuple[Coord, ...] | set[Coord], lit: set[Coord]) -> set[Coord]:
    out: set[Coord] = set()
    for r, c in block:
        for nr in range(r - 1, r + 2):
            for nc in range(c - 1, c + 2):
                if (nr, nc) in lit:
                    out.add((nr, nc))
    return out


def _touches_lit(block: tuple[Coord, ...], lit: set[Coord]) -> bool:
    return bool(_adjacent_lit_cells(block, lit))


def _illuminate_background(grid: np.ndarray, lit: set[Coord], sources: tuple[Coord, ...], radius: int) -> None:
    h, w = grid.shape
    for r, c in sources:
        for nr in range(max(0, r - radius), min(h, r + radius + 1)):
            for nc in range(max(0, c - radius), min(w, c + radius + 1)):
                if grid[nr, nc] == BLACK:
                    lit.add((nr, nc))


def _simulate_fire(grid: np.ndarray) -> tuple[list[tuple[Coord, ...]], set[int], set[Coord]]:
    ignition = np.argwhere(grid == YELLOW)
    if ignition.shape != (1, 2):
        raise ValueError("expected exactly one ignition pixel")

    lit: set[Coord] = set()
    seed = (int(ignition[0, 0]), int(ignition[0, 1]))
    _illuminate_background(grid, lit, (seed,), 1)

    blocks = _red_components(grid)
    burned: set[int] = set()
    while True:
        new_ids = [i for i, block in enumerate(blocks) if i not in burned and _touches_lit(block, lit)]
        if not new_ids:
            break
        for i in new_ids:
            burned.add(i)
            _illuminate_background(grid, lit, blocks[i], len(blocks[i]))
    return blocks, burned, lit


def _find_block_index(blocks: list[tuple[Coord, ...]], target: tuple[Coord, ...] | set[Coord]) -> int | None:
    target_set = set(target)
    for i, block in enumerate(blocks):
        if set(block) == target_set:
            return i
    return None


def _can_place_fuel(
    grid: np.ndarray,
    coords: tuple[Coord, ...],
    *,
    avoid_cells: set[Coord],
    forbidden_neighbor_colors: set[int],
) -> bool:
    h, w = grid.shape
    coord_set = set(coords)
    for r, c in coords:
        if not _in_bounds((h, w), r, c):
            return False
        if grid[r, c] != BLACK or (r, c) in avoid_cells:
            return False
    for r, c in coords:
        for nr, nc in _neighbors8(r, c):
            if not _in_bounds((h, w), nr, nc) or (nr, nc) in coord_set:
                continue
            if int(grid[nr, nc]) in forbidden_neighbor_colors:
                return False
    return True


def _random_shape(area: int | None = None) -> tuple[Coord, ...]:
    if area is None:
        area = random.randint(1, 4)
    return random.choice(SHAPES_BY_AREA[int(area)])


def _try_add_burnable_block(grid: np.ndarray, lit: set[Coord], max_tries: int = 700) -> bool:
    h, w = grid.shape
    for _ in range(max_tries):
        shape = _random_shape()
        sh, sw = _shape_size(shape)
        top = random.randint(0, h - sh)
        left = random.randint(0, w - sw)
        coords = _shift_shape(shape, top, left)
        if not _can_place_fuel(
            grid,
            coords,
            avoid_cells=lit,
            forbidden_neighbor_colors={RED, YELLOW, GREY},
        ):
            continue
        if not _touches_lit(coords, lit):
            continue
        test = grid.copy()
        _paint(test, coords, RED)
        blocks, burned, _ = _simulate_fire(test)
        idx = _find_block_index(blocks, coords)
        if idx is None or idx not in burned:
            continue
        _paint(grid, coords, RED)
        return True
    return False


def _try_add_blocked_candidate(
    grid: np.ndarray,
    max_tries: int = 1200,
) -> tuple[set[Coord], set[Coord]] | None:
    _, _, lit = _simulate_fire(grid)
    h, w = grid.shape
    for _ in range(max_tries):
        shape = _random_shape()
        sh, sw = _shape_size(shape)
        top = random.randint(0, h - sh)
        left = random.randint(0, w - sw)
        coords = _shift_shape(shape, top, left)
        if not _can_place_fuel(
            grid,
            coords,
            avoid_cells=set(),
            forbidden_neighbor_colors={RED, YELLOW},
        ):
            continue
        wall_cells = _adjacent_lit_cells(coords, lit)
        if not wall_cells or len(wall_cells) > 8:
            continue

        blocked = grid.copy()
        _paint(blocked, coords, RED)
        _paint(blocked, wall_cells, GREY)
        blocks, burned, _ = _simulate_fire(blocked)
        idx = _find_block_index(blocks, coords)
        if idx is None or idx in burned:
            continue
        if len(burned) < 2 or len(burned) == len(blocks):
            continue

        unblocked = blocked.copy()
        _paint(unblocked, wall_cells, BLACK)
        blocks2, burned2, _ = _simulate_fire(unblocked)
        idx2 = _find_block_index(blocks2, coords)
        if idx2 is None or idx2 not in burned2:
            continue

        _paint(grid, coords, RED)
        _paint(grid, wall_cells, GREY)
        return set(coords), set(wall_cells)
    return None


def _try_add_safe_blocks(grid: np.ndarray, count: int, max_tries: int = 1200) -> int:
    added = 0
    for _ in range(max_tries):
        if added >= count:
            break
        blocks, burned, lit = _simulate_fire(grid)
        h, w = grid.shape
        shape = _random_shape()
        sh, sw = _shape_size(shape)
        top = random.randint(0, h - sh)
        left = random.randint(0, w - sw)
        coords = _shift_shape(shape, top, left)
        if not _can_place_fuel(
            grid,
            coords,
            avoid_cells=lit,
            forbidden_neighbor_colors={RED, YELLOW, GREY},
        ):
            continue
        if _touches_lit(coords, lit):
            continue
        test = grid.copy()
        _paint(test, coords, RED)
        blocks2, burned2, _ = _simulate_fire(test)
        idx = _find_block_index(blocks2, coords)
        if idx is None or idx in burned2:
            continue
        if len(burned2) != len(burned):
            continue
        _paint(grid, coords, RED)
        added += 1
    return added


def _add_extra_walls(grid: np.ndarray, count: int, max_tries: int = 400) -> None:
    _, _, lit = _simulate_fire(grid)
    h, w = grid.shape
    placed = 0
    for _ in range(max_tries):
        if placed >= count:
            return
        r = random.randrange(h)
        c = random.randrange(w)
        if grid[r, c] != BLACK or (r, c) in lit:
            continue
        ok = True
        for nr, nc in _neighbors8(r, c):
            if _in_bounds((h, w), nr, nc) and grid[nr, nc] == YELLOW:
                ok = False
                break
        if not ok:
            continue
        grid[r, c] = GREY
        placed += 1


def main(grid: np.ndarray) -> np.ndarray:
    """Erase exactly the red fuel blocks reached by an invisible background-light cascade.

    Input structure:
    - The grid has a black background, several small red 4-connected fuel blocks, some grey wall pixels, and exactly one yellow ignition pixel.
    - Each red block has area 1 to 4 and remains a separate connected component.

    Rule:
    - The yellow ignition lights only black background cells within Chebyshev distance 1.
    - Any red block with at least one pixel 8-neighbor-adjacent to a lit background cell ignites as a whole; an ignited block then lights black background cells within Chebyshev distance equal to its own area.
    - Repeat that relay until no new block ignites, then erase only the ignited red blocks to black while keeping surviving red blocks, grey walls, and the yellow ignition unchanged.

    Key insight:
    - The output never shows the fire itself, only which red blocks vanished, so the solver must reconstruct the hidden relay: light spreads only through black background, block size sets the next lighting radius, and grey walls can sever the chain by removing background cells that would otherwise carry it onward.
    """
    g = np.asarray(grid, dtype=int)
    blocks, burned, _ = _simulate_fire(g)
    out = g.copy()
    for i in burned:
        for r, c in blocks[i]:
            out[r, c] = BLACK
    return out


def _attempt_generate() -> np.ndarray | None:
    h = random.randint(18, 28)
    w = random.randint(18, 28)
    grid = np.zeros((h, w), dtype=int)

    seed_r = random.randint(2, h - 3)
    seed_c = random.randint(2, w - 3)
    grid[seed_r, seed_c] = YELLOW

    chain_len = random.randint(3, 5)
    for _ in range(chain_len):
        _, _, lit = _simulate_fire(grid)
        if not _try_add_burnable_block(grid, lit):
            return None

    blocked = _try_add_blocked_candidate(grid)
    if blocked is None:
        return None
    blocked_coords, wall_cells = blocked

    safe_target = random.randint(1, 3)
    safe_added = _try_add_safe_blocks(grid, safe_target)
    if safe_added == 0:
        return None

    _add_extra_walls(grid, random.randint(0, 3))

    blocks, burned, _ = _simulate_fire(grid)
    total_blocks = len(blocks)
    if total_blocks < 5:
        return None
    if not (2 <= len(burned) < total_blocks):
        return None

    blocked_idx = _find_block_index(blocks, blocked_coords)
    if blocked_idx is None or blocked_idx in burned:
        return None

    wallless = grid.copy()
    _paint(wallless, wall_cells, BLACK)
    blocks2, burned2, _ = _simulate_fire(wallless)
    blocked_idx2 = _find_block_index(blocks2, blocked_coords)
    if blocked_idx2 is None or blocked_idx2 not in burned2:
        return None

    survivors = total_blocks - len(burned)
    if survivors < 2:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
