from __future__ import annotations

import random
from typing import Iterable

import numpy as np


BG = 0
SEED = 3
BLOCKER = 5

TILE_COUNT = 3
TILE_HEIGHT_OPTIONS = (4, 5, 6)
GRID_WIDTH_RANGE = (12, 18)
BLOCKER_COUNT_RANGE = (3, 6)
MAX_GENERATION_ATTEMPTS = 1200

SPIRAL_DIRECTIONS = [(-1, 0), (0, 1), (1, 0), (0, -1)]
MIN_PAINTED_CELLS = 8
MAX_PAINTED_CELLS = 80


def in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _tile_ranges(shape: tuple[int, int]) -> list[tuple[int, int]]:
    height, _ = shape
    assert height % TILE_COUNT == 0, "grid must split into three equal horizontal tiles"
    tile_h = height // TILE_COUNT
    return [(idx * tile_h, (idx + 1) * tile_h) for idx in range(TILE_COUNT)]


def _local_blockers(tile: np.ndarray) -> set[tuple[int, int]]:
    return {tuple(map(int, rc)) for rc in np.argwhere(tile == BLOCKER)}


def _is_diagonal_symmetric(blockers: Iterable[tuple[int, int]]) -> bool:
    cells = {tuple(map(int, cell)) for cell in blockers}
    return cells == {(c, r) for r, c in cells}


def _split_metadata(grid: np.ndarray) -> tuple[int, list[tuple[int, int]], list[set[tuple[int, int]]], int]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    height, width = g.shape
    assert height % TILE_COUNT == 0, "grid height must be divisible by 3"
    tile_h = height // TILE_COUNT

    seeds: list[tuple[int, int]] = []
    blockers_by_tile: list[set[tuple[int, int]]] = []
    symmetric_tiles: list[int] = []
    for tile_idx, (r0, r1) in enumerate(_tile_ranges(g.shape)):
        tile = g[r0:r1, :]
        seed_coords = np.argwhere(tile == SEED)
        assert seed_coords.shape == (1, 2), "each tile must contain exactly one green seed"
        sr, sc = map(int, seed_coords[0])
        seeds.append((r0 + sr, sc))

        blockers = _local_blockers(tile)
        blockers_by_tile.append(blockers)
        if _is_diagonal_symmetric(blockers):
            symmetric_tiles.append(tile_idx)

    asymmetric_tiles = [idx for idx in range(TILE_COUNT) if idx not in symmetric_tiles]
    assert len(asymmetric_tiles) == 1, "expected exactly one asymmetrical blocker tile"
    return tile_h, seeds, blockers_by_tile, asymmetric_tiles[0]


def _spiral_trace(seed: tuple[int, int], shape: tuple[int, int]) -> list[tuple[int, int, bool]]:
    r, c = seed
    height, width = shape
    max_delta = max(r, height - 1 - r, c, width - 1 - c)
    max_leg_length = 2 * (max_delta + 1)

    trace: list[tuple[int, int, bool]] = []
    direction_index = 0
    leg_length = 2
    saw_oob = False
    while leg_length <= max_leg_length:
        dr, dc = SPIRAL_DIRECTIONS[direction_index]
        for _ in range(leg_length):
            r += dr
            c += dc
            if not in_bounds(shape, r, c):
                saw_oob = True
                continue
            trace.append((r, c, saw_oob))
        direction_index = (direction_index + 1) % len(SPIRAL_DIRECTIONS)
        if direction_index in (0, 2):
            leg_length += 2
    return trace


def _draw_spiral(
    out: np.ndarray,
    seed: tuple[int, int],
    blockers: set[tuple[int, int]],
) -> np.ndarray:
    r, c = seed
    for nr, nc, _ in _spiral_trace(seed, out.shape):
        if (nr, nc) in blockers:
            break
        if out[nr, nc] == BG:
            out[nr, nc] = SEED
        r, c = nr, nc
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Activate the seed in the only diagonally asymmetric strip and let it paint a clipped spiral across the full grid.

    Input structure:
    - The grid has a black background and is split into three equal horizontal tiles.
    - Each tile contains exactly one green seed pixel and several gray blocker pixels.
    - Exactly two tiles have blocker layouts equal to their own local-coordinate transpose; one tile does not.

    Rule:
    - Inspect only the gray blocker coordinates inside each tile, using the tile's own `(r, c)` coordinates.
    - Keep the seed from the unique non-symmetric tile and erase the seeds in the other two tiles.
    - From the kept seed, follow an unbounded spiral with directions up, right, down, left and leg lengths `2, 2, 4, 4, 6, 6, ...`; paint only in-bounds black cells green, skip out-of-bounds steps while still counting them, and stop immediately before the first gray blocker reached anywhere in the full grid.

    Key insight:
    - The start point is chosen by per-tile diagonal-transpose symmetry, but the stopping point is chosen by the global blocker arrangement, so both stages are necessary.
    """
    g = np.asarray(grid, dtype=int)
    tile_h, seeds, blockers_by_tile, active_tile = _split_metadata(g)
    del tile_h  # parity with generator metadata; the rule only needs the split.

    out = g.copy()
    for tile_idx, (r, c) in enumerate(seeds):
        if tile_idx != active_tile:
            out[r, c] = BG

    blockers = {tuple(map(int, rc)) for rc in np.argwhere(g == BLOCKER)}
    return _draw_spiral(out, seeds[active_tile], blockers)


def _seed_row_options(tile_idx: int, tile_h: int, height: int) -> list[int]:
    r0 = tile_idx * tile_h
    center = height // 2
    candidates = [r0 + offset for offset in range(tile_h)]
    distances = sorted({abs(r - center) for r in candidates})
    keep = distances[: min(2, len(distances))]
    return [r for r in candidates if abs(r - center) in keep]


def _choose_seeds(tile_h: int, width: int, asym_tile: int) -> list[tuple[int, int]]:
    height = tile_h * TILE_COUNT
    center_cols = [c for c in range(width) if abs(c - width // 2) <= 2]
    if not center_cols:
        center_cols = list(range(width))

    seeds: list[tuple[int, int]] = []
    for tile_idx in range(TILE_COUNT):
        row_choices = _seed_row_options(tile_idx, tile_h, height)
        if tile_idx != asym_tile and len(row_choices) > 1:
            row_choices = row_choices[:]
        row = random.choice(row_choices)
        col = random.choice(center_cols)
        seeds.append((row, col))
    return seeds


def _global_to_local(cell: tuple[int, int], tile_h: int) -> tuple[int, tuple[int, int]]:
    r, c = cell
    return r // tile_h, (r % tile_h, c)


def _symmetric_closure(cells: Iterable[tuple[int, int]], tile_h: int) -> set[tuple[int, int]] | None:
    closure: set[tuple[int, int]] = set()
    for r, c in cells:
        if not (0 <= r < tile_h and 0 <= c < tile_h):
            return None
        closure.add((r, c))
        closure.add((c, r))
    return closure


def _available_zone_cells(tile_h: int, forbidden: set[tuple[int, int]]) -> list[tuple[int, int]]:
    return [
        (r, c)
        for r in range(tile_h)
        for c in range(tile_h)
        if (r, c) not in forbidden
    ]


def _generate_symmetric_set(
    tile_h: int,
    count: int,
    forbidden: set[tuple[int, int]],
    required: set[tuple[int, int]] | None = None,
) -> set[tuple[int, int]] | None:
    required = set() if required is None else set(required)
    must = _symmetric_closure(required, tile_h)
    if must is None or len(must) > count or must & forbidden:
        return None

    all_diagonals = [(d, d) for d in range(tile_h) if (d, d) not in forbidden]
    all_pairs = [
        ((r, c), (c, r))
        for r in range(tile_h)
        for c in range(r + 1, tile_h)
        if (r, c) not in forbidden and (c, r) not in forbidden
    ]

    must_pairs = {tuple(sorted(((r, c), (c, r)))) for r, c in must if r != c}
    used_diagonals = {(r, c) for r, c in must if r == c}
    pair_pool = [pair for pair in all_pairs if tuple(sorted(pair)) not in must_pairs]
    diag_pool = [cell for cell in all_diagonals if cell not in used_diagonals]

    remaining = count - len(must)
    valid_compositions = []
    for pair_count in range(len(pair_pool) + 1):
        diag_count = remaining - 2 * pair_count
        if 0 <= diag_count <= len(diag_pool):
            valid_compositions.append((pair_count, diag_count))
    if not valid_compositions:
        return None

    for _ in range(80):
        pair_count, diag_count = random.choice(valid_compositions)
        chosen = set(must)
        if pair_count:
            for pair in random.sample(pair_pool, pair_count):
                chosen.update(pair)
        if diag_count:
            chosen.update(random.sample(diag_pool, diag_count))
        if len(chosen) == count and _is_diagonal_symmetric(chosen):
            return chosen
    return None


def _generate_asymmetric_set(
    tile_h: int,
    count: int,
    forbidden: set[tuple[int, int]],
    required: set[tuple[int, int]] | None = None,
) -> set[tuple[int, int]] | None:
    required = set() if required is None else set(required)
    if any(not (0 <= r < tile_h and 0 <= c < tile_h) for r, c in required):
        return None
    if required & forbidden:
        return None

    pool = [cell for cell in _available_zone_cells(tile_h, forbidden) if cell not in required]
    if len(required) > count or len(pool) < count - len(required):
        return None

    for _ in range(120):
        chosen = set(required)
        if count > len(required):
            chosen.update(random.sample(pool, count - len(required)))
        if not _is_diagonal_symmetric(chosen):
            return chosen
    return None


def _painted_tile_indices(seed: tuple[int, int], blockers: set[tuple[int, int]], shape: tuple[int, int], tile_h: int) -> set[int]:
    seen = {seed[0] // tile_h}
    for r, c, _ in _spiral_trace(seed, shape):
        if (r, c) in blockers:
            break
        seen.add(r // tile_h)
    return seen


def _first_hit(seed: tuple[int, int], blockers: set[tuple[int, int]], shape: tuple[int, int]) -> tuple[int, tuple[int, int], bool] | None:
    for idx, (r, c, saw_oob) in enumerate(_spiral_trace(seed, shape)):
        if (r, c) in blockers:
            return idx, (r, c), saw_oob
    return None


def _candidate_trace_cells(
    seed: tuple[int, int],
    shape: tuple[int, int],
    tile_h: int,
    asym_tile: int,
    seeds: list[tuple[int, int]],
    desired_target: str,
    desired_oob: bool,
) -> list[tuple[int, tuple[int, int], int, tuple[int, int], bool]]:
    trace = _spiral_trace(seed, shape)
    seed_set = set(seeds)
    candidates = []
    for idx, (r, c, saw_oob) in enumerate(trace):
        if idx < MIN_PAINTED_CELLS or idx > MAX_PAINTED_CELLS:
            continue
        tile_idx, local = _global_to_local((r, c), tile_h)
        if c >= tile_h or (r, c) in seed_set:
            continue
        if desired_target == "same_tile" and tile_idx != asym_tile:
            continue
        if desired_target == "other_tile" and tile_idx == asym_tile:
            continue
        if saw_oob != desired_oob:
            continue

        painted_tiles = {seed[0] // tile_h}
        for pr, pc, _ in trace[:idx]:
            painted_tiles.add(pr // tile_h)
        if len(painted_tiles) < 2:
            continue

        if tile_idx != asym_tile:
            closure = _symmetric_closure({local}, tile_h)
            if closure is None:
                continue
            early_cells = {trace[j][:2] for j in range(idx)}
            partner_hits_early = False
            for rr, cc in closure:
                global_partner = (tile_idx * tile_h + rr, cc)
                if global_partner in early_cells:
                    partner_hits_early = True
                    break
                if global_partner in seed_set:
                    partner_hits_early = True
                    break
            if partner_hits_early:
                continue

        candidates.append((idx, (r, c), tile_idx, local, saw_oob))
    return candidates


def _build_counts(asym_tile: int, target_tile: int) -> list[int]:
    asym_count = random.randint(*BLOCKER_COUNT_RANGE)
    counts = [0] * TILE_COUNT
    counts[asym_tile] = asym_count

    symmetric_tiles = [idx for idx in range(TILE_COUNT) if idx != asym_tile]
    same_count_tile = target_tile if target_tile in symmetric_tiles else random.choice(symmetric_tiles)
    counts[same_count_tile] = asym_count

    other_tile = next(idx for idx in symmetric_tiles if idx != same_count_tile)
    counts[other_tile] = random.randint(*BLOCKER_COUNT_RANGE)
    return counts


def _construct_blockers(
    shape: tuple[int, int],
    tile_h: int,
    seeds: list[tuple[int, int]],
    asym_tile: int,
    hit_idx: int,
    hit_cell: tuple[int, int],
    hit_tile: int,
    hit_local: tuple[int, int],
) -> list[set[tuple[int, int]]] | None:
    trace = _spiral_trace(seeds[asym_tile], shape)
    early_global = {trace[idx][:2] for idx in range(hit_idx)}
    counts = _build_counts(asym_tile, hit_tile)
    blockers_by_tile: list[set[tuple[int, int]] | None] = [None] * TILE_COUNT

    for tile_idx in range(TILE_COUNT):
        local_forbidden = {local for global_cell in early_global if _global_to_local(global_cell, tile_h)[0] == tile_idx for local in [_global_to_local(global_cell, tile_h)[1]]}
        seed_r, seed_c = seeds[tile_idx]
        local_forbidden.add((seed_r % tile_h, seed_c))

        required: set[tuple[int, int]] = set()
        if tile_idx == hit_tile:
            required.add(hit_local)

        if tile_idx == asym_tile:
            blockers = _generate_asymmetric_set(tile_h, counts[tile_idx], local_forbidden, required)
        else:
            blockers = _generate_symmetric_set(tile_h, counts[tile_idx], local_forbidden, required)
        if blockers is None:
            return None
        blockers_by_tile[tile_idx] = blockers

    return [set(blockers) for blockers in blockers_by_tile]


def _render_grid(
    tile_h: int,
    width: int,
    seeds: list[tuple[int, int]],
    blockers_by_tile: list[set[tuple[int, int]]],
) -> np.ndarray:
    grid = np.zeros((tile_h * TILE_COUNT, width), dtype=int)
    for tile_idx, (r, c) in enumerate(seeds):
        grid[r, c] = SEED
        row_offset = tile_idx * tile_h
        for lr, lc in blockers_by_tile[tile_idx]:
            grid[row_offset + lr, lc] = BLOCKER
    return grid


def _oracle_output(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    tile_h, seeds, _blockers_by_tile, active_tile = _split_metadata(g)
    out = g.copy()
    for tile_idx, (r, c) in enumerate(seeds):
        if tile_idx != active_tile:
            out[r, c] = BG
    blockers = {tuple(map(int, rc)) for rc in np.argwhere(g == BLOCKER)}
    for r, c, _ in _spiral_trace(seeds[active_tile], g.shape):
        if (r, c) in blockers:
            break
        if out[r, c] == BG:
            out[r, c] = SEED
    return out


def _validate_generated(
    grid: np.ndarray,
    desired_target: str,
    desired_oob: bool,
) -> bool:
    if not isinstance(grid, np.ndarray) or grid.ndim != 2 or max(grid.shape) > 30:
        return False

    try:
        tile_h, seeds, blockers_by_tile, asym_tile = _split_metadata(grid)
    except Exception:
        return False

    counts = [len(blockers) for blockers in blockers_by_tile]
    if not all(BLOCKER_COUNT_RANGE[0] <= count <= BLOCKER_COUNT_RANGE[1] for count in counts):
        return False
    if counts.count(counts[asym_tile]) < 2:
        return False

    blocker_cells = {tuple(map(int, rc)) for rc in np.argwhere(grid == BLOCKER)}
    first_hit = _first_hit(seeds[asym_tile], blocker_cells, grid.shape)
    if first_hit is None:
        return False
    hit_idx, hit_cell, saw_oob = first_hit
    if not (MIN_PAINTED_CELLS <= hit_idx <= MAX_PAINTED_CELLS):
        return False

    hit_tile, hit_local = _global_to_local(hit_cell, tile_h)
    if desired_target == "same_tile" and hit_tile != asym_tile:
        return False
    if desired_target == "other_tile" and hit_tile == asym_tile:
        return False
    if saw_oob != desired_oob:
        return False

    if len(_painted_tile_indices(seeds[asym_tile], blocker_cells, grid.shape, tile_h)) < 2:
        return False
    if hit_tile != asym_tile and hit_local[0] != hit_local[1]:
        partner = (hit_tile * tile_h + hit_local[1], hit_local[0])
        trace_prefix = {cell for _idx, cell, _flag in [(idx, rc[:2], rc[2]) for idx, rc in enumerate(_spiral_trace(seeds[asym_tile], grid.shape))][:hit_idx]}
        if partner in trace_prefix:
            return False

    try:
        solved = main(grid)
        oracle = _oracle_output(grid)
    except Exception:
        return False
    if not np.array_equal(solved, oracle):
        return False

    green_in = int(np.count_nonzero(grid == SEED))
    green_out = int(np.count_nonzero(solved == SEED))
    if green_out <= green_in:
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    tile_h = random.choice(TILE_HEIGHT_OPTIONS)
    width = random.randint(*GRID_WIDTH_RANGE)
    height = tile_h * TILE_COUNT

    asym_tile = random.randrange(TILE_COUNT)
    seeds = _choose_seeds(tile_h, width, asym_tile)
    desired_target = random.choice(("same_tile", "other_tile"))
    desired_oob = random.choice((False, True))

    candidates = _candidate_trace_cells(
        seeds[asym_tile],
        (height, width),
        tile_h,
        asym_tile,
        seeds,
        desired_target,
        desired_oob,
    )
    if not candidates:
        return None

    random.shuffle(candidates)
    for hit_idx, hit_cell, hit_tile, hit_local, _ in candidates[:40]:
        blockers_by_tile = _construct_blockers(
            (height, width),
            tile_h,
            seeds,
            asym_tile,
            hit_idx,
            hit_cell,
            hit_tile,
            hit_local,
        )
        if blockers_by_tile is None:
            continue
        grid = _render_grid(tile_h, width, seeds, blockers_by_tile)
        if _validate_generated(grid, desired_target, desired_oob):
            return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 4")
