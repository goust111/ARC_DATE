from __future__ import annotations

import random
from typing import Iterable

import numpy as np


BLACK = 0
RED = 2
GREEN = 3
YELLOW = 4

DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIAG_DIRS = [(-1, -1), (-1, 1), (1, -1), (1, 1)]

CENTER_MARGIN = 3
GRID_MIN = 15
GRID_MAX = 22
CLEAR_DISTANCE_RANGE = (5, 7)
BLOCK_DISTANCE_RANGE = (5, 7)
EXTRA_CENTER_PROB = 0.45
EXTRA_DISTRACTOR_RANGE = (1, 4)
PLACEMENT_ATTEMPTS = 250
GENERATION_ATTEMPTS = 500


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _neighbors4(r: int, c: int) -> Iterable[tuple[int, int]]:
    for dr, dc in DIR4:
        yield r + dr, c + dc


def _red_components(grid: np.ndarray) -> list[tuple[tuple[int, int], ...]]:
    mask = grid == RED
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[tuple[tuple[int, int], ...]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if _in_bounds(grid.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(tuple(sorted(cells)))
    return comps


def _is_micro_bar(cells: tuple[tuple[int, int], ...]) -> bool:
    if len(cells) != 2:
        return False
    (r1, c1), (r2, c2) = cells
    return abs(r1 - r2) + abs(c1 - c2) == 1


def _micro_bars(grid: np.ndarray) -> set[frozenset[tuple[int, int]]]:
    return {
        frozenset(comp)
        for comp in _red_components(grid)
        if _is_micro_bar(comp)
    }


def _required_bars(center: tuple[int, int]) -> tuple[frozenset[tuple[int, int]], ...]:
    r, c = center
    return (
        frozenset(((r - 3, c), (r - 2, c))),
        frozenset(((r + 2, c), (r + 3, c))),
        frozenset(((r, c - 3), (r, c - 2))),
        frozenset(((r, c + 2), (r, c + 3))),
    )


def _plus_cells(center: tuple[int, int]) -> set[tuple[int, int]]:
    cells = set()
    for bar in _required_bars(center):
        cells.update(bar)
    return cells


def _gap_cells(center: tuple[int, int]) -> tuple[tuple[int, int], ...]:
    r, c = center
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _center_valid_window(shape: tuple[int, int], center: tuple[int, int]) -> bool:
    r, c = center
    return CENTER_MARGIN <= r < shape[0] - CENTER_MARGIN and CENTER_MARGIN <= c < shape[1] - CENTER_MARGIN


def _is_implied_center(
    grid: np.ndarray,
    bars: set[frozenset[tuple[int, int]]],
    center: tuple[int, int],
) -> bool:
    if not _center_valid_window(grid.shape, center):
        return False
    r, c = center
    if int(grid[r, c]) != BLACK:
        return False
    if any(int(grid[rr, cc]) != BLACK for rr, cc in _gap_cells(center)):
        return False
    return all(bar in bars for bar in _required_bars(center))


def _detect_centers(grid: np.ndarray) -> list[tuple[int, int]]:
    bars = _micro_bars(grid)
    centers = []
    for r in range(CENTER_MARGIN, grid.shape[0] - CENTER_MARGIN):
        for c in range(CENTER_MARGIN, grid.shape[1] - CENTER_MARGIN):
            if _is_implied_center(grid, bars, (r, c)):
                centers.append((r, c))
    return centers


def _diag_interior(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    r1, c1 = a
    r2, c2 = b
    dr = r2 - r1
    dc = c2 - c1
    if dr == 0 or abs(dr) != abs(dc):
        return []
    sr = 1 if dr > 0 else -1
    sc = 1 if dc > 0 else -1
    return [(r1 + k * sr, c1 + k * sc) for k in range(1, abs(dr))]


def _rule_output(grid: np.ndarray, centers: list[tuple[int, int]]) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    for r, c in centers:
        out[r, c] = YELLOW
    for i, a in enumerate(centers):
        for b in centers[i + 1 :]:
            interior = _diag_interior(a, b)
            if not interior:
                continue
            if all(int(grid[r, c]) == BLACK for r, c in interior):
                for r, c in interior:
                    if out[r, c] == BLACK:
                        out[r, c] = GREEN
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Infer missing plus centers from four aligned micro-bars, then connect diagonally aligned centers across clear diagonal gaps.

    Input structure:
    - The grid has a black background and red foreground pixels.
    - Every red component is a two-pixel horizontal or vertical micro-bar.
    - Some quartets of bars imply a plus centered on a missing black cell: one vertical bar above, one below, one horizontal bar left, and one right, each with a one-cell black gap to the center.

    Rule:
    - Detect every black cell whose four cardinal directions contain those four required red micro-bars and mark that center yellow.
    - Keep all original red bars unchanged, including distractor bars that do not belong to any implied plus.
    - For each pair of yellow centers, if they lie on the same 45-degree diagonal and every strict interior cell between them is black in the input, color those interior cells green.
    - Do not overwrite the yellow endpoints.

    Key insight:
    - The important objects are not explicit pluses but the missing centers implied by four separate bars; only those inferred centers participate in diagonal connection, while stray bars also act as blockers.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    centers = _detect_centers(g)
    return _rule_output(g, centers)


def _center_pair_ranges(
    size: int,
    clear_step: int,
    clear_sign: int,
    block_step: int,
    block_sign: int,
) -> tuple[int, int] | None:
    lo = max(
        CENTER_MARGIN,
        CENTER_MARGIN - clear_sign * clear_step,
        CENTER_MARGIN - block_sign * block_step,
    )
    hi = min(
        size - CENTER_MARGIN - 1,
        size - CENTER_MARGIN - 1 - clear_sign * clear_step,
        size - CENTER_MARGIN - 1 - block_sign * block_step,
    )
    if lo > hi:
        return None
    return lo, hi


def _plus_patterns_are_separate(centers: list[tuple[int, int]]) -> bool:
    red_cells: set[tuple[int, int]] = set()
    for center in centers:
        for cell in _plus_cells(center):
            if cell in red_cells:
                return False
            for nr, nc in _neighbors4(*cell):
                if (nr, nc) in red_cells:
                    return False
        red_cells.update(_plus_cells(center))
    return True


def _pick_core_centers(shape: tuple[int, int]) -> tuple[list[tuple[int, int]], tuple[tuple[int, int], tuple[int, int]], tuple[tuple[int, int], tuple[int, int]]] | None:
    h, w = shape
    for _ in range(PLACEMENT_ATTEMPTS):
        clear_step = random.randint(*CLEAR_DISTANCE_RANGE)
        block_step = random.randint(*BLOCK_DISTANCE_RANGE)
        clear_dir = random.choice(DIAG_DIRS)
        block_dir = random.choice([(clear_dir[0], -clear_dir[1]), (-clear_dir[0], clear_dir[1])])

        row_range = _center_pair_ranges(h, clear_step, clear_dir[0], block_step, block_dir[0])
        col_range = _center_pair_ranges(w, clear_step, clear_dir[1], block_step, block_dir[1])
        if row_range is None or col_range is None:
            continue

        base = (
            random.randint(*row_range),
            random.randint(*col_range),
        )
        clear_center = (base[0] + clear_dir[0] * clear_step, base[1] + clear_dir[1] * clear_step)
        blocked_center = (base[0] + block_dir[0] * block_step, base[1] + block_dir[1] * block_step)
        centers = [base, clear_center, blocked_center]
        if not _plus_patterns_are_separate(centers):
            continue
        return centers, (base, clear_center), (base, blocked_center)
    return None


def _pick_extra_center(shape: tuple[int, int], centers: list[tuple[int, int]]) -> tuple[int, int] | None:
    h, w = shape
    candidates = [
        (r, c)
        for r in range(CENTER_MARGIN, h - CENTER_MARGIN)
        for c in range(CENTER_MARGIN, w - CENTER_MARGIN)
    ]
    random.shuffle(candidates)
    for cand in candidates:
        if cand in centers:
            continue
        if any(_diag_interior(cand, other) for other in centers):
            continue
        trial = centers + [cand]
        if _plus_patterns_are_separate(trial):
            return cand
    return None


def _bar_can_place(
    shape: tuple[int, int],
    cells: frozenset[tuple[int, int]],
    red_cells: set[tuple[int, int]],
    protected_black: set[tuple[int, int]],
) -> bool:
    for r, c in cells:
        if not _in_bounds(shape, r, c):
            return False
        if (r, c) in red_cells or (r, c) in protected_black:
            return False
    for r, c in cells:
        for nr, nc in _neighbors4(r, c):
            if (nr, nc) in red_cells and (nr, nc) not in cells:
                return False
    return True


def _place_blocker_bar(
    shape: tuple[int, int],
    path: list[tuple[int, int]],
    red_cells: set[tuple[int, int]],
    protected_black: set[tuple[int, int]],
) -> frozenset[tuple[int, int]] | None:
    path_cells = path[:]
    random.shuffle(path_cells)
    for r, c in path_cells:
        candidates = [
            frozenset(((r, c), (r, c - 1))),
            frozenset(((r, c), (r, c + 1))),
            frozenset(((r, c), (r - 1, c))),
            frozenset(((r, c), (r + 1, c))),
        ]
        random.shuffle(candidates)
        for bar in candidates:
            if _bar_can_place(shape, bar, red_cells, protected_black):
                return bar
    return None


def _place_random_distractors(
    shape: tuple[int, int],
    red_cells: set[tuple[int, int]],
    protected_black: set[tuple[int, int]],
    count: int,
) -> list[frozenset[tuple[int, int]]]:
    bars: list[frozenset[tuple[int, int]]] = []
    for _ in range(count):
        placed = False
        for _ in range(PLACEMENT_ATTEMPTS):
            if random.random() < 0.5:
                r = random.randint(0, shape[0] - 1)
                c = random.randint(0, shape[1] - 2)
                bar = frozenset(((r, c), (r, c + 1)))
            else:
                r = random.randint(0, shape[0] - 2)
                c = random.randint(0, shape[1] - 1)
                bar = frozenset(((r, c), (r + 1, c)))
            if _bar_can_place(shape, bar, red_cells, protected_black):
                bars.append(bar)
                red_cells.update(bar)
                placed = True
                break
        if not placed:
            break
    return bars


def _bars_to_grid(shape: tuple[int, int], bars: Iterable[frozenset[tuple[int, int]]]) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    for bar in bars:
        for r, c in bar:
            grid[r, c] = RED
    return grid


def _scene_signature(grid: np.ndarray) -> tuple[
    list[tuple[int, int]],
    list[tuple[tuple[int, int], tuple[int, int], bool, list[tuple[int, int]]]],
    set[frozenset[tuple[int, int]]],
]:
    centers = sorted(_detect_centers(grid))
    bars = _micro_bars(grid)
    pairs = []
    for i, a in enumerate(centers):
        for b in centers[i + 1 :]:
            interior = _diag_interior(a, b)
            if not interior:
                continue
            blocked = any(int(grid[r, c]) == RED for r, c in interior)
            pairs.append((a, b, blocked, interior))
    return centers, pairs, bars


def _validate_scene(grid: np.ndarray, intended_centers: list[tuple[int, int]]) -> bool:
    if grid.ndim != 2 or max(grid.shape) > 30:
        return False
    comps = _red_components(grid)
    if not comps or not all(_is_micro_bar(comp) for comp in comps):
        return False

    centers, pairs, bars = _scene_signature(grid)
    if centers != sorted(intended_centers):
        return False
    if len(centers) < 3:
        return False
    if len(pairs) != 2:
        return False

    clear_pairs = [entry for entry in pairs if not entry[2]]
    blocked_pairs = [entry for entry in pairs if entry[2]]
    if len(clear_pairs) != 1 or len(blocked_pairs) != 1:
        return False

    used_bars: set[frozenset[tuple[int, int]]] = set()
    for center in centers:
        used_bars.update(_required_bars(center))
    if not used_bars.issubset(bars):
        return False

    distractor_bars = bars - used_bars
    if not distractor_bars:
        return False
    distractor_cells = {cell for bar in distractor_bars for cell in bar}
    if not any(cell in distractor_cells for cell in blocked_pairs[0][3]):
        return False

    expected = grid.copy()
    for r, c in centers:
        expected[r, c] = YELLOW
    for r, c in clear_pairs[0][3]:
        expected[r, c] = GREEN
    if not np.array_equal(main(grid), expected):
        return False
    return True


def _generate_once() -> np.ndarray | None:
    shape = (random.randint(GRID_MIN, GRID_MAX), random.randint(GRID_MIN, GRID_MAX))
    core = _pick_core_centers(shape)
    if core is None:
        return None

    centers, clear_pair, blocked_pair = core
    if random.random() < EXTRA_CENTER_PROB:
        extra = _pick_extra_center(shape, centers)
        if extra is not None:
            centers.append(extra)

    bars = []
    red_cells: set[tuple[int, int]] = set()
    for center in centers:
        for bar in _required_bars(center):
            bars.append(bar)
            red_cells.update(bar)

    protected_black = set(centers)
    for center in centers:
        protected_black.update(_gap_cells(center))
    protected_black.update(_diag_interior(*clear_pair))

    blocker = _place_blocker_bar(shape, _diag_interior(*blocked_pair), red_cells, protected_black)
    if blocker is None:
        return None
    bars.append(blocker)
    red_cells.update(blocker)

    extra_distractors = _place_random_distractors(
        shape,
        red_cells,
        protected_black,
        random.randint(*EXTRA_DISTRACTOR_RANGE),
    )
    bars.extend(extra_distractors)

    grid = _bars_to_grid(shape, bars)
    return grid if _validate_scene(grid, centers) else None


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _generate_once()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid implied-plus connector scene")
