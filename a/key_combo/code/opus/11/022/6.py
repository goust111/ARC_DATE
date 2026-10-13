import random

import numpy as np


BACKGROUND = 0
CROSS_COLOR = 3
DOT_COLOR = 4
HALF_MIN = 5
HALF_MAX = 10
MAX_GENERATION_ATTEMPTS = 400

OVERLAP_TEMPLATES = (
    ((0, 1), (0, 2), (1, 3)),
    ((0, 1), (2, 3), (0, 3)),
    ((0, 2), (1, 2), (1, 3)),
    ((0, 1, 2), (1, 3), (0, 2, 3)),
    ((0, 1), (0, 2, 3), (1, 2, 3)),
)
EXTRA_SUPPORTS = ((0, 3), (1, 2), (0, 1, 2, 3), (0, 1, 3), (0, 2, 3))


def _find_cross_center(grid: np.ndarray) -> tuple[int, int]:
    rows = [r for r in range(grid.shape[0]) if np.all(grid[r, :] == CROSS_COLOR)]
    cols = [c for c in range(grid.shape[1]) if np.all(grid[:, c] == CROSS_COLOR)]
    assert len(rows) == 1, "expected exactly one full green row"
    assert len(cols) == 1, "expected exactly one full green column"
    return rows[0], cols[0]


def _folded_quadrants(grid: np.ndarray, center_r: int, center_c: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ul = grid[:center_r, :center_c]
    ur = np.fliplr(grid[:center_r, center_c + 1 :])
    ll = np.flipud(grid[center_r + 1 :, :center_c])
    lr = np.rot90(grid[center_r + 1 :, center_c + 1 :], 2)
    assert ul.shape == ur.shape == ll.shape == lr.shape, "four quadrants must fold to the same size"
    return ul, ur, ll, lr


def main(grid: np.ndarray) -> np.ndarray:
    """Fold the four dot quadrants onto the upper-left quadrant, then erase odd Chebyshev shells.

    Input structure
    - The input is an odd-sized square grid on a black background.
    - One full green row and one full green column cross at the grid center.
    - The four open quadrants around that cross contain sparse yellow dots and nothing else.

    Rule
    - Locate the green cross intersection and split the grid into the four quadrants that exclude the cross itself.
    - Keep the upper-left quadrant as the target frame, mirror the upper-right horizontally, mirror the lower-left vertically, and rotate the lower-right by 180 degrees.
    - Overlay those four layers transparently: a target cell is yellow iff any folded layer places a yellow dot there.
    - In the folded upper-left frame, keep only cells whose Chebyshev distance from coordinate (0,0), namely `max(row, col)`, is even; erase odd-distance cells to black.

    Key insight
    - The green cross does double duty: it fixes both the folding axes and the origin of the alternating-distance filter, so neither the fold nor the parity mask can be skipped.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape[0] == g.shape[1], "expected a square 2D grid"
    assert g.shape[0] % 2 == 1, "expected an odd side length"
    assert set(np.unique(g)).issubset({BACKGROUND, CROSS_COLOR, DOT_COLOR}), "unexpected colors in grid"

    center_r, center_c = _find_cross_center(g)
    assert center_r == g.shape[0] // 2 and center_c == g.shape[1] // 2, "cross must be centered"

    ul, ur, ll, lr = _folded_quadrants(g, center_r, center_c)
    folded = np.where((ul == DOT_COLOR) | (ur == DOT_COLOR) | (ll == DOT_COLOR) | (lr == DOT_COLOR), DOT_COLOR, BACKGROUND)

    rows, cols = np.indices(folded.shape)
    out = folded.copy()
    out[np.maximum(rows, cols) % 2 == 1] = BACKGROUND
    return out


def _raw_local_from_fold_position(quadrant_id: int, half: int, pos: tuple[int, int]) -> tuple[int, int]:
    r, c = pos
    if quadrant_id == 0:
        return r, c
    if quadrant_id == 1:
        return r, half - 1 - c
    if quadrant_id == 2:
        return half - 1 - r, c
    return half - 1 - r, half - 1 - c


def _build_grid(half: int, quadrant_positions: dict[int, set[tuple[int, int]]]) -> np.ndarray:
    side = 2 * half + 1
    grid = np.full((side, side), BACKGROUND, dtype=int)
    grid[half, :] = CROSS_COLOR
    grid[:, half] = CROSS_COLOR

    for r, c in quadrant_positions[0]:
        grid[r, c] = DOT_COLOR
    for r, c in quadrant_positions[1]:
        grid[r, half + 1 + c] = DOT_COLOR
    for r, c in quadrant_positions[2]:
        grid[half + 1 + r, c] = DOT_COLOR
    for r, c in quadrant_positions[3]:
        grid[half + 1 + r, half + 1 + c] = DOT_COLOR
    return grid


def _visible_even_positions(half: int, transformed_sets: dict[int, set[tuple[int, int]]]) -> set[tuple[int, int]]:
    composite = set().union(*transformed_sets.values())
    return {pos for pos in composite if max(pos) % 2 == 0}


def _validate_generated(grid: np.ndarray) -> bool:
    if not isinstance(grid, np.ndarray) or grid.ndim != 2:
        return False
    if grid.shape[0] != grid.shape[1] or grid.shape[0] % 2 != 1 or max(grid.shape) > 30:
        return False
    if set(np.unique(grid)) != {BACKGROUND, CROSS_COLOR, DOT_COLOR}:
        return False

    half = grid.shape[0] // 2
    center_r, center_c = _find_cross_center(grid)
    if (center_r, center_c) != (half, half):
        return False

    raw_sets = {
        0: {tuple(map(int, rc)) for rc in np.argwhere(grid[:half, :half] == DOT_COLOR)},
        1: {tuple(map(int, rc)) for rc in np.argwhere(grid[:half, half + 1 :] == DOT_COLOR)},
        2: {tuple(map(int, rc)) for rc in np.argwhere(grid[half + 1 :, :half] == DOT_COLOR)},
        3: {tuple(map(int, rc)) for rc in np.argwhere(grid[half + 1 :, half + 1 :] == DOT_COLOR)},
    }
    if not all(2 <= len(points) <= 4 for points in raw_sets.values()):
        return False
    if len({tuple(sorted(points)) for points in raw_sets.values()}) != 4:
        return False

    transformed_sets = {
        qid: {_raw_local_from_fold_position(qid, half, pos) for pos in points}
        for qid, points in raw_sets.items()
    }
    if len({tuple(sorted(points)) for points in transformed_sets.values()}) != 4:
        return False

    composite = set().union(*transformed_sets.values())
    visible = _visible_even_positions(half, transformed_sets)
    if len(visible) < 3:
        return False
    if len(composite - visible) < 2:
        return False
    if len({r for r, _ in visible}) < 2 or len({c for _, c in visible}) < 2:
        return False

    overlap_positions = {pos for pos in composite if sum(pos in transformed_sets[qid] for qid in range(4)) >= 2}
    if len(overlap_positions) < 3:
        return False
    if not (visible & overlap_positions):
        return False

    single_quadrant_views = [{pos for pos in transformed_sets[qid] if max(pos) % 2 == 0} for qid in range(4)]
    if any(view == visible for view in single_quadrant_views):
        return False
    if not any(pos in visible and pos not in transformed_sets[0] and any(pos in transformed_sets[qid] for qid in (1, 2, 3)) for pos in composite):
        return False

    return np.array_equal(main(grid), _positions_to_grid(half, visible))


def _positions_to_grid(half: int, positions: set[tuple[int, int]]) -> np.ndarray:
    out = np.full((half, half), BACKGROUND, dtype=int)
    for r, c in positions:
        out[r, c] = DOT_COLOR
    return out


def _gen_attempt() -> np.ndarray | None:
    half = random.randint(HALF_MIN, HALF_MAX)
    all_positions = [(r, c) for r in range(half) for c in range(half)]
    even_positions = [pos for pos in all_positions if max(pos) % 2 == 0]
    odd_positions = [pos for pos in all_positions if max(pos) % 2 == 1]
    edge_even = [pos for pos in even_positions if 0 in pos or half - 1 in pos]
    edge_odd = [pos for pos in odd_positions if 0 in pos or half - 1 in pos]
    if len(even_positions) < 4 or len(odd_positions) < 2:
        return None

    overlap_positions: list[tuple[int, int]] = []
    for pool in (edge_even or even_positions, even_positions, edge_odd or odd_positions):
        choices = [pos for pos in pool if pos not in overlap_positions]
        overlap_positions.append(random.choice(choices))
    if random.random() < 0.55:
        overlap_positions.append(random.choice([pos for pos in all_positions if pos not in overlap_positions]))

    supports = list(random.choice(OVERLAP_TEMPLATES))
    if len(overlap_positions) == 4:
        supports.append(random.choice(EXTRA_SUPPORTS))

    transformed_sets = {qid: set() for qid in range(4)}
    for pos, support in zip(overlap_positions, supports):
        for quadrant_id in support:
            transformed_sets[quadrant_id].add(pos)

    target_counts: dict[int, int] = {}
    for quadrant_id in range(4):
        minimum = max(2, len(transformed_sets[quadrant_id]))
        if minimum > 4:
            return None
        target_counts[quadrant_id] = random.randint(minimum, 4)

    for quadrant_id in range(4):
        need = target_counts[quadrant_id] - len(transformed_sets[quadrant_id])
        if need <= 0:
            continue
        pool = [pos for pos in all_positions if pos not in transformed_sets[quadrant_id]]
        random.shuffle(pool)
        for pos in pool[:need]:
            transformed_sets[quadrant_id].add(pos)
        if len(transformed_sets[quadrant_id]) != target_counts[quadrant_id]:
            return None

    if len({tuple(sorted(points)) for points in transformed_sets.values()}) != 4:
        return None

    raw_sets = {
        quadrant_id: {_raw_local_from_fold_position(quadrant_id, half, pos) for pos in points}
        for quadrant_id, points in transformed_sets.items()
    }
    grid = _build_grid(half, raw_sets)
    return grid if _validate_generated(grid) else None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid sample after {MAX_GENERATION_ATTEMPTS} attempts")
