from __future__ import annotations

import numpy as np


PANEL_SIZE = 7
SEPARATOR_COLOR = 5
LEFT_INPUT_COLOR = 2
RIGHT_INPUT_COLOR = 1
LEFT_ONLY_COLOR = 2
RIGHT_ONLY_COLOR = 1
BOTH_COLOR = 4
PRIORITY_ORDER = (LEFT_ONLY_COLOR, RIGHT_ONLY_COLOR, BOTH_COLOR)
ALL_SLOPES = ((1, 1), (-1, -1), (-1, 1), (1, -1))
POS_SLOPES = ((1, 1), (-1, -1))
NEG_SLOPES = ((-1, 1), (1, -1))
GEN_RETRIES = 400
BASE_PATTERNS = (
    (
        ((2, 6), LEFT_ONLY_COLOR),
        ((1, 3), LEFT_ONLY_COLOR),
        ((4, 6), RIGHT_ONLY_COLOR),
        ((3, 3), RIGHT_ONLY_COLOR),
        ((0, 2), BOTH_COLOR),
        ((0, 6), BOTH_COLOR),
    ),
    (
        ((2, 0), RIGHT_ONLY_COLOR),
        ((2, 1), BOTH_COLOR),
        ((2, 6), LEFT_ONLY_COLOR),
        ((3, 0), RIGHT_ONLY_COLOR),
        ((3, 6), BOTH_COLOR),
        ((4, 0), LEFT_ONLY_COLOR),
    ),
    (
        ((0, 6), BOTH_COLOR),
        ((3, 3), RIGHT_ONLY_COLOR),
        ((3, 6), RIGHT_ONLY_COLOR),
        ((4, 6), BOTH_COLOR),
        ((6, 2), LEFT_ONLY_COLOR),
        ((6, 3), LEFT_ONLY_COLOR),
    ),
    (
        ((0, 5), RIGHT_ONLY_COLOR),
        ((1, 0), LEFT_ONLY_COLOR),
        ((1, 5), BOTH_COLOR),
        ((3, 2), LEFT_ONLY_COLOR),
        ((3, 4), RIGHT_ONLY_COLOR),
        ((6, 1), BOTH_COLOR),
    ),
    (
        ((0, 3), RIGHT_ONLY_COLOR),
        ((0, 5), BOTH_COLOR),
        ((1, 6), LEFT_ONLY_COLOR),
        ((2, 5), BOTH_COLOR),
        ((3, 2), RIGHT_ONLY_COLOR),
        ((3, 6), LEFT_ONLY_COLOR),
    ),
)
COUNT_CHOICES = np.array([4, 5, 6, 7, 8], dtype=int)
COUNT_PROBS = np.array([0.25, 0.35, 0.25, 0.10, 0.05], dtype=float)


def _expand_seeds(
    seed_grid: np.ndarray,
    *,
    step: int = 3,
    slopes: tuple[tuple[int, int], ...] = ALL_SLOPES,
    priority_order: tuple[int, int, int] = PRIORITY_ORDER,
) -> np.ndarray:
    rank = {color: idx for idx, color in enumerate(priority_order)}
    out = np.zeros_like(seed_grid, dtype=int)
    best = np.full(seed_grid.shape, -1, dtype=int)
    height, width = seed_grid.shape

    for r, c in np.argwhere(seed_grid != 0):
        color = int(seed_grid[r, c])
        if rank[color] > best[r, c]:
            out[r, c] = color
            best[r, c] = rank[color]

        for dr, dc in slopes:
            rr = int(r) + dr * step
            cc = int(c) + dc * step
            while 0 <= rr < height and 0 <= cc < width:
                if rank[color] > best[rr, cc]:
                    out[rr, cc] = color
                    best[rr, cc] = rank[color]
                rr += dr * step
                cc += dc * step

    return out


def _seed_grid_from_input(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    height, width = g.shape
    assert width % 2 == 1, "expected a center separator column"

    mid = width // 2
    separator = g[:, mid]
    assert np.all(separator == separator[0]), "expected a solid separator column"

    left = g[:, :mid]
    right = g[:, mid + 1 :]
    assert left.shape == right.shape, "expected equal-size side panels"

    left_mask = left != 0
    right_mask = right != 0

    seed = np.zeros(left.shape, dtype=int)
    seed[left_mask & ~right_mask] = LEFT_ONLY_COLOR
    seed[right_mask & ~left_mask] = RIGHT_ONLY_COLOR
    seed[left_mask & right_mask] = BOTH_COLOR
    return seed


def _contribution_sets(
    seed_grid: np.ndarray,
    *,
    step: int = 3,
    slopes: tuple[tuple[int, int], ...] = ALL_SLOPES,
) -> list[list[set[int]]]:
    height, width = seed_grid.shape
    contrib = [[set() for _ in range(width)] for _ in range(height)]

    for r, c in np.argwhere(seed_grid != 0):
        color = int(seed_grid[r, c])
        contrib[int(r)][int(c)].add(color)
        for dr, dc in slopes:
            rr = int(r) + dr * step
            cc = int(c) + dc * step
            while 0 <= rr < height and 0 <= cc < width:
                contrib[rr][cc].add(color)
                rr += dr * step
                cc += dc * step

    return contrib


def _seed_counts(seed_grid: np.ndarray) -> tuple[int, int, int]:
    left_only = int(np.count_nonzero(seed_grid == LEFT_ONLY_COLOR))
    right_only = int(np.count_nonzero(seed_grid == RIGHT_ONLY_COLOR))
    both = int(np.count_nonzero(seed_grid == BOTH_COLOR))
    return left_only, right_only, both


def _render_input(seed_grid: np.ndarray) -> np.ndarray:
    height, width = seed_grid.shape
    grid = np.zeros((height, width * 2 + 1), dtype=int)
    grid[:, width] = SEPARATOR_COLOR
    left = grid[:, :width]
    right = grid[:, width + 1 :]
    left[(seed_grid == LEFT_ONLY_COLOR) | (seed_grid == BOTH_COLOR)] = LEFT_INPUT_COLOR
    right[(seed_grid == RIGHT_ONLY_COLOR) | (seed_grid == BOTH_COLOR)] = RIGHT_INPUT_COLOR
    return grid


def _apply_symmetry(seed_grid: np.ndarray, code: int) -> np.ndarray:
    code = int(code) % 8
    if code == 0:
        return seed_grid.copy()
    if code == 1:
        return np.rot90(seed_grid, 1)
    if code == 2:
        return np.rot90(seed_grid, 2)
    if code == 3:
        return np.rot90(seed_grid, 3)
    if code == 4:
        return np.fliplr(seed_grid)
    if code == 5:
        return np.flipud(seed_grid)
    if code == 6:
        return seed_grid.T.copy()
    return np.fliplr(seed_grid).T.copy()


def _base_seed_grid() -> np.ndarray:
    seed = np.zeros((PANEL_SIZE, PANEL_SIZE), dtype=int)
    pattern = BASE_PATTERNS[int(np.random.randint(0, len(BASE_PATTERNS)))]
    for (r, c), color in pattern:
        seed[r, c] = int(color)
    return _apply_symmetry(seed, int(np.random.randint(0, 8)))


def _pad_seed_grid(seed_grid: np.ndarray) -> np.ndarray:
    target = int(np.random.choice([7, 8, 9]))
    if target == seed_grid.shape[0]:
        return seed_grid
    out = np.zeros((target, target), dtype=int)
    r0 = int(np.random.randint(0, target - seed_grid.shape[0] + 1))
    c0 = int(np.random.randint(0, target - seed_grid.shape[1] + 1))
    out[r0 : r0 + seed_grid.shape[0], c0 : c0 + seed_grid.shape[1]] = seed_grid
    return out


def _pick_target_count() -> int:
    idx = int(np.random.choice(len(COUNT_CHOICES), p=COUNT_PROBS))
    return int(COUNT_CHOICES[idx])


def _augment_seed_grid(seed_grid: np.ndarray) -> np.ndarray:
    seed = seed_grid.copy()
    target_left = _pick_target_count()
    target_right = _pick_target_count()

    while True:
        left_only, right_only, both = _seed_counts(seed)
        left_total = left_only + both
        right_total = right_only + both

        category_choices: list[int] = []
        if left_total < target_left:
            category_choices.append(LEFT_ONLY_COLOR)
        if right_total < target_right:
            category_choices.append(RIGHT_ONLY_COLOR)
        if left_total < target_left and right_total < target_right:
            category_choices.extend([BOTH_COLOR, BOTH_COLOR])
        if not category_choices:
            return seed

        free = np.argwhere(seed == 0)
        if free.size == 0:
            return seed

        r, c = free[int(np.random.randint(0, len(free)))]
        color = int(category_choices[int(np.random.randint(0, len(category_choices)))])
        seed[int(r), int(c)] = color


def _seed_mutations(seed_grid: np.ndarray) -> list[np.ndarray]:
    variants: list[np.ndarray] = []
    for r, c in np.argwhere(seed_grid != 0):
        color = int(seed_grid[r, c])
        if color == BOTH_COLOR:
            for replacement in (LEFT_ONLY_COLOR, RIGHT_ONLY_COLOR, 0):
                mutated = seed_grid.copy()
                mutated[int(r), int(c)] = replacement
                variants.append(mutated)
        else:
            mutated = seed_grid.copy()
            mutated[int(r), int(c)] = 0
            variants.append(mutated)
    return variants


def _seed_is_semantically_valid(seed_grid: np.ndarray) -> bool:
    left_only, right_only, both = _seed_counts(seed_grid)
    left_total = left_only + both
    right_total = right_only + both

    if not (4 <= left_total <= 8 and 4 <= right_total <= 8):
        return False
    if min(left_only, right_only, both) < 1:
        return False

    out = _expand_seeds(seed_grid)
    if set(map(int, np.unique(out))) != {0, 1, 2, 4}:
        return False

    non_black = int(np.count_nonzero(out))
    if not (8 <= non_black <= 24):
        return False

    contrib = _contribution_sets(seed_grid)
    conflicts = [
        frozenset(contrib[r][c])
        for r in range(seed_grid.shape[0])
        for c in range(seed_grid.shape[1])
        if len(contrib[r][c]) >= 2
    ]
    if len(conflicts) < 2:
        return False
    if frozenset((LEFT_ONLY_COLOR, RIGHT_ONLY_COLOR)) not in conflicts:
        return False
    if not any(BOTH_COLOR in colors for colors in conflicts):
        return False

    if np.array_equal(out, seed_grid):
        return False
    if np.array_equal(out, _expand_seeds(seed_grid, slopes=POS_SLOPES)):
        return False
    if np.array_equal(out, _expand_seeds(seed_grid, slopes=NEG_SLOPES)):
        return False
    if np.array_equal(out, _expand_seeds(seed_grid, step=1)):
        return False
    if np.array_equal(out, _expand_seeds(seed_grid, priority_order=(RIGHT_ONLY_COLOR, LEFT_ONLY_COLOR, BOTH_COLOR))):
        return False
    if np.array_equal(out, _expand_seeds(seed_grid, priority_order=(LEFT_ONLY_COLOR, BOTH_COLOR, RIGHT_ONLY_COLOR))):
        return False

    for mutated in _seed_mutations(seed_grid):
        if np.array_equal(out, _expand_seeds(mutated)):
            return False

    rendered = _render_input(seed_grid)
    if not np.array_equal(_seed_grid_from_input(rendered), seed_grid):
        return False
    if not np.array_equal(main(rendered), out):
        return False
    return True


def main(grid: np.ndarray) -> np.ndarray:
    """
    Compare two separated masks, classify each disagreement type, then spread those seed colors along every third diagonal position.

    Input structure:
    - The grid has a solid separator column in the middle, with equal-size left and right panels.
    - The left panel is a black background with sparse red foreground pixels, and the right panel is a black background with sparse blue foreground pixels.
    - Foreground overlap is judged by aligned panel coordinates after ignoring the separator.

    Rule:
    - Remove the separator and compare aligned panel cells.
    - Mark left-only foreground cells red, right-only foreground cells blue, and cells occupied on both sides yellow.
    - Treat every non-black marked cell as a seed, then repeat that seed color along both diagonal slopes at offsets spaced by 3 cells in both directions.
    - When several propagated colors claim one output cell, keep yellow over blue and blue over red.

    Key insight:
    - The pixelwise comparison only creates the seed layer; the real output comes from extending each seed through its modulo-3 diagonal congruence classes and resolving overlaps by color priority.
    """
    seed_grid = _seed_grid_from_input(grid)
    return _expand_seeds(seed_grid)


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        seed = _pad_seed_grid(_augment_seed_grid(_base_seed_grid()))
        if _seed_is_semantically_valid(seed):
            return _render_input(seed)
    raise RuntimeError("failed to generate a valid ARC input after retries")
