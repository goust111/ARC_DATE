from __future__ import annotations

import itertools
import random

import numpy as np


BG = 0
TILE_SIZE = 4
TILE_ROWS = 4
TILE_COLS = 4
PARITY_SLOTS = ((0, 0), (0, 1), (1, 0), (1, 1))
ALL_SCHEDULES = tuple(itertools.permutations(range(4)))
NON_BLACK_COLORS = tuple(range(1, 10))
RETRY_LIMIT = 500


def _parity_index(i: int, j: int) -> int:
    return (i % 2) * 2 + (j % 2)


def _rotate(tile: np.ndarray, k: int) -> np.ndarray:
    return np.rot90(tile, k).copy()


def _split_tiles(grid: np.ndarray) -> list[list[np.ndarray]]:
    return [
        [
            grid[
                i * TILE_SIZE : (i + 1) * TILE_SIZE,
                j * TILE_SIZE : (j + 1) * TILE_SIZE,
            ].copy()
            for j in range(TILE_COLS)
        ]
        for i in range(TILE_ROWS)
    ]


def _assemble_output(base: np.ndarray, schedule: tuple[int, int, int, int]) -> np.ndarray:
    out = np.zeros((TILE_ROWS * TILE_SIZE, TILE_COLS * TILE_SIZE), dtype=int)
    for i in range(TILE_ROWS):
        for j in range(TILE_COLS):
            k = schedule[_parity_index(i, j)]
            out[
                i * TILE_SIZE : (i + 1) * TILE_SIZE,
                j * TILE_SIZE : (j + 1) * TILE_SIZE,
            ] = _rotate(base, k)
    return out


def _infer_candidate(
    grid: np.ndarray, schedule: tuple[int, int, int, int]
) -> tuple[np.ndarray, np.ndarray] | None:
    tiles = _split_tiles(grid)
    normalized_by_parity: list[list[np.ndarray]] = [[] for _ in range(4)]

    for i in range(TILE_ROWS):
        for j in range(TILE_COLS):
            p = _parity_index(i, j)
            normalized_by_parity[p].append(_rotate(tiles[i][j], -schedule[p]))

    base = np.zeros((TILE_SIZE, TILE_SIZE), dtype=int)
    for r in range(TILE_SIZE):
        for c in range(TILE_SIZE):
            seen = {
                int(tile[r, c])
                for group in normalized_by_parity
                for tile in group
                if int(tile[r, c]) != BG
            }
            if len(seen) > 1:
                return None
            if seen:
                base[r, c] = next(iter(seen))

    predicted_tiles = [_rotate(base, schedule[p]) for p in range(4)]
    for p in range(4):
        pred = predicted_tiles[p]
        exact_match = False
        for tile in normalized_by_parity[p]:
            observed = _rotate(tile, schedule[p])
            if np.any((observed != BG) & (observed != pred)):
                return None
            if np.array_equal(observed, pred):
                exact_match = True
        if not exact_match:
            return None

    out = _assemble_output(base, schedule)
    for i in range(TILE_ROWS):
        for j in range(TILE_COLS):
            block = grid[
                i * TILE_SIZE : (i + 1) * TILE_SIZE,
                j * TILE_SIZE : (j + 1) * TILE_SIZE,
            ]
            target = out[
                i * TILE_SIZE : (i + 1) * TILE_SIZE,
                j * TILE_SIZE : (j + 1) * TILE_SIZE,
            ]
            if np.any((block != BG) & (block != target)):
                return None
    return base, out


def _candidate_outputs(grid: np.ndarray) -> list[np.ndarray]:
    found: dict[bytes, np.ndarray] = {}
    for schedule in ALL_SCHEDULES:
        candidate = _infer_candidate(grid, schedule)
        if candidate is None:
            continue
        _, out = candidate
        found.setdefault(out.tobytes(), out)
    return list(found.values())


def _infer_pure_translation(grid: np.ndarray) -> np.ndarray | None:
    tile = np.zeros((TILE_SIZE, TILE_SIZE), dtype=int)
    for r in range(TILE_SIZE):
        for c in range(TILE_SIZE):
            seen = {
                int(grid[rr, cc])
                for rr in range(r, grid.shape[0], TILE_SIZE)
                for cc in range(c, grid.shape[1], TILE_SIZE)
                if int(grid[rr, cc]) != BG
            }
            if len(seen) > 1:
                return None
            if seen:
                tile[r, c] = next(iter(seen))
    out = np.tile(tile, (TILE_ROWS, TILE_COLS))
    if np.any((grid != BG) & (grid != out)):
        return None
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Recover blacked-out tiles by inferring one base 4x4 tile and the hidden 2x2 rotation schedule.

    Input structure:
    - The grid is a 4x4 array of equal 4x4 subtiles.
    - Every subtile is a rotated copy of one asymmetric multicolor base tile that may itself contain black cells.
    - Some subtiles are fully blacked out and some others are partially blacked out.

    Rule:
    - Treat each subtile position `(i, j)` as using a rotation determined only by `(i mod 2, j mod 2)`.
    - Test the four-way rotation schedules, unrotate visible evidence back to a common frame, and infer the unique base tile whose non-black cells agree across all subtiles.
    - Reapply the inferred schedule to rebuild the full 4x4 tiling and replace every black occluded cell with its scheduled tile value.

    Key insight:
    - Black is ambiguous because it can mean either true tile content or occlusion, so the solver must normalize rotations first and use the repeated parity schedule to tell persistent black structure from erased pixels.
    """
    g = np.asarray(grid, dtype=int)
    assert g.shape == (TILE_ROWS * TILE_SIZE, TILE_COLS * TILE_SIZE)
    outputs = _candidate_outputs(g)
    assert outputs, "expected at least one consistent rotation schedule"
    assert len(outputs) == 1, "the visible evidence must determine a unique completed grid"
    return outputs[0]


def _all_rotations_distinct(tile: np.ndarray) -> bool:
    rotations = [_rotate(tile, k) for k in range(4)]
    return len({rot.tobytes() for rot in rotations}) == 4


def _sample_base_tile() -> np.ndarray | None:
    color_count = random.randint(2, 3)
    colors = random.sample(NON_BLACK_COLORS, color_count)
    nonzero_count = random.randint(8, 12)
    positions = random.sample(range(TILE_SIZE * TILE_SIZE), nonzero_count)

    tile = np.zeros((TILE_SIZE, TILE_SIZE), dtype=int)
    assignments = colors[:]
    while len(assignments) < nonzero_count:
        assignments.append(random.choice(colors))
    random.shuffle(assignments)

    for idx, color in zip(positions, assignments):
        tile[idx // TILE_SIZE, idx % TILE_SIZE] = color

    if np.count_nonzero(tile == BG) < 3:
        return None
    if not _all_rotations_distinct(tile):
        return None
    if any(np.count_nonzero(tile[r, :]) == 0 for r in range(TILE_SIZE)):
        return None
    if any(np.count_nonzero(tile[:, c]) == 0 for c in range(TILE_SIZE)):
        return None
    if len(set(int(v) for v in tile.ravel() if int(v) != BG)) != color_count:
        return None
    return tile


def _parity_positions() -> list[list[tuple[int, int]]]:
    groups = [[] for _ in range(4)]
    for i in range(TILE_ROWS):
        for j in range(TILE_COLS):
            groups[_parity_index(i, j)].append((i, j))
    return groups


def _connected_mask() -> np.ndarray:
    area = random.randint(2, 6)
    start = (random.randrange(TILE_SIZE), random.randrange(TILE_SIZE))
    cells = {start}
    while len(cells) < area:
        frontier = []
        for r, c in cells:
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                rr, cc = r + dr, c + dc
                if 0 <= rr < TILE_SIZE and 0 <= cc < TILE_SIZE and (rr, cc) not in cells:
                    frontier.append((rr, cc))
        if not frontier:
            break
        cells.add(random.choice(frontier))
    mask = np.zeros((TILE_SIZE, TILE_SIZE), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    return mask


def _choose_mask_for_tile(tile: np.ndarray) -> np.ndarray | None:
    nonzero = tile != BG
    for _ in range(40):
        mask = _connected_mask()
        if not np.any(mask & nonzero):
            continue
        if np.all(mask | ~nonzero):
            continue
        masked = tile.copy()
        masked[mask] = BG
        if np.array_equal(masked, tile):
            continue
        if np.all(masked == BG):
            continue
        return mask
    return None


def _counts_per_parity(positions: list[tuple[int, int]]) -> list[int]:
    counts = [0, 0, 0, 0]
    for i, j in positions:
        counts[_parity_index(i, j)] += 1
    return counts


def _reserve_visible_positions() -> set[tuple[int, int]]:
    protected = set()
    for group in _parity_positions():
        protected.add(random.choice(group))
    return protected


def _apply_occlusions(
    out: np.ndarray, schedule: tuple[int, int, int, int]
) -> np.ndarray | None:
    del schedule
    inp = out.copy()
    positions = [(i, j) for i in range(TILE_ROWS) for j in range(TILE_COLS)]
    protected = _reserve_visible_positions()
    full_count = random.randint(2, 4)
    partial_count = random.randint(2, 3)

    full_tiles: list[tuple[int, int]] = []
    full_parity = [0, 0, 0, 0]
    choices = [pos for pos in positions if pos not in protected]
    random.shuffle(choices)
    for pos in choices:
        p = _parity_index(*pos)
        if full_parity[p] >= 2:
            continue
        full_tiles.append(pos)
        full_parity[p] += 1
        if len(full_tiles) == full_count:
            break
    if len(full_tiles) != full_count:
        return None

    remaining = [pos for pos in positions if pos not in protected and pos not in full_tiles]
    random.shuffle(remaining)
    partial_tiles = remaining[:partial_count]
    if len(partial_tiles) != partial_count:
        return None

    masks: dict[tuple[int, int], np.ndarray] = {}
    for i, j in partial_tiles:
        tile = out[
            i * TILE_SIZE : (i + 1) * TILE_SIZE,
            j * TILE_SIZE : (j + 1) * TILE_SIZE,
        ]
        mask = _choose_mask_for_tile(tile)
        if mask is None:
            return None
        masks[(i, j)] = mask

    for i, j in full_tiles:
        inp[
            i * TILE_SIZE : (i + 1) * TILE_SIZE,
            j * TILE_SIZE : (j + 1) * TILE_SIZE,
        ] = BG
    for (i, j), mask in masks.items():
        block = inp[
            i * TILE_SIZE : (i + 1) * TILE_SIZE,
            j * TILE_SIZE : (j + 1) * TILE_SIZE,
        ]
        block[mask] = BG

    if any(
        np.array_equal(
            inp[
                i * TILE_SIZE : (i + 1) * TILE_SIZE,
                j * TILE_SIZE : (j + 1) * TILE_SIZE,
            ],
            out[
                i * TILE_SIZE : (i + 1) * TILE_SIZE,
                j * TILE_SIZE : (j + 1) * TILE_SIZE,
            ],
        )
        for i, j in full_tiles + partial_tiles
    ):
        return None

    return inp


def _validate_case(inp: np.ndarray, out: np.ndarray) -> bool:
    if _infer_pure_translation(inp) is not None:
        return False

    outputs = _candidate_outputs(inp)
    if len(outputs) != 1:
        return False
    if not np.array_equal(outputs[0], out):
        return False

    exact_per_parity = [0, 0, 0, 0]
    changed_tiles = 0
    for i in range(TILE_ROWS):
        for j in range(TILE_COLS):
            p = _parity_index(i, j)
            in_tile = inp[
                i * TILE_SIZE : (i + 1) * TILE_SIZE,
                j * TILE_SIZE : (j + 1) * TILE_SIZE,
            ]
            out_tile = out[
                i * TILE_SIZE : (i + 1) * TILE_SIZE,
                j * TILE_SIZE : (j + 1) * TILE_SIZE,
            ]
            if np.array_equal(in_tile, out_tile):
                exact_per_parity[p] += 1
            else:
                changed_tiles += 1
    if min(exact_per_parity) < 1:
        return False
    if changed_tiles < 4:
        return False
    if not np.array_equal(main(inp), out):
        return False
    return True


def _generate_case() -> tuple[np.ndarray, np.ndarray] | None:
    base = _sample_base_tile()
    if base is None:
        return None
    schedule = tuple(random.sample(range(4), 4))
    out = _assemble_output(base, schedule)
    inp = _apply_occlusions(out, schedule)
    if inp is None:
        return None
    if not _validate_case(inp, out):
        return None
    return inp, out


def generate_input() -> np.ndarray:
    for _ in range(RETRY_LIMIT):
        case = _generate_case()
        if case is not None:
            return case[0]
    raise RuntimeError("failed to generate a valid rotating-tile occlusion puzzle")
