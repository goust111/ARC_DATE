from __future__ import annotations

import random

import numpy as np


BG = 0
GREEN = 3
SEED_COLORS = (1, 2, 4, 5, 6, 7, 8, 9)
GEN_ATTEMPTS = 300

QUADRANT_RULES: dict[str, dict[str, tuple[int, int] | tuple[tuple[int, int], ...]]] = {
    "ul": {"diag": (-1, -1), "turns": ((-1, 0), (0, -1))},
    "ur": {"diag": (-1, 1), "turns": ((0, 1), (-1, 0))},
    "ll": {"diag": (1, -1), "turns": ((0, -1), (1, 0))},
    "lr": {"diag": (1, 1), "turns": ((1, 0), (0, 1))},
}


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _find_cross(grid: np.ndarray) -> tuple[int, int, int]:
    h, w = grid.shape
    row_candidates: list[tuple[int, int]] = []
    col_candidates: list[tuple[int, int]] = []

    for r in range(h):
        color = int(grid[r, 0])
        if color != BG and np.all(grid[r, :] == color):
            row_candidates.append((r, color))

    for c in range(w):
        color = int(grid[0, c])
        if color != BG and np.all(grid[:, c] == color):
            col_candidates.append((c, color))

    matches = [
        (color, r, c)
        for r, color in row_candidates
        for c, col_color in col_candidates
        if color == col_color
    ]
    assert len(matches) == 1, "expected one full green cross"
    return matches[0]


def _quadrant(r: int, c: int, mid_r: int, mid_c: int) -> str:
    if r < mid_r and c < mid_c:
        return "ul"
    if r < mid_r and c > mid_c:
        return "ur"
    if r > mid_r and c < mid_c:
        return "ll"
    if r > mid_r and c > mid_c:
        return "lr"
    raise AssertionError("seed lies on the cross")


def _simulate(grid: np.ndarray, *, return_bends: bool = False):
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    green, mid_r, mid_c = _find_cross(g)

    out = g.copy()
    h, w = out.shape
    seeds = sorted(
        [
            (int(r), int(c))
            for r, c in np.argwhere((out != BG) & (out != green))
        ]
    )

    bends = 0
    for seed_r, seed_c in seeds:
        color = int(out[seed_r, seed_c])
        quad = _quadrant(seed_r, seed_c, mid_r, mid_c)
        diag_dr, diag_dc = QUADRANT_RULES[quad]["diag"]
        turn_dirs = QUADRANT_RULES[quad]["turns"]
        assert isinstance(diag_dr, int) and isinstance(diag_dc, int)
        assert isinstance(turn_dirs, tuple)

        cur_r, cur_c = seed_r, seed_c
        dr, dc = diag_dr, diag_dc
        diagonal_phase = True

        while True:
            next_r = cur_r + dr
            next_c = cur_c + dc
            if not _in_bounds((h, w), next_r, next_c):
                break

            next_value = int(out[next_r, next_c])
            if next_value == BG:
                out[next_r, next_c] = color
                cur_r, cur_c = next_r, next_c
                continue

            if next_value == green:
                break

            if not diagonal_phase:
                break

            turned = False
            for turn_dr, turn_dc in turn_dirs:
                turn_r = cur_r + turn_dr
                turn_c = cur_c + turn_dc
                if _in_bounds((h, w), turn_r, turn_c) and int(out[turn_r, turn_c]) == BG:
                    out[turn_r, turn_c] = color
                    cur_r, cur_c = turn_r, turn_c
                    dr, dc = turn_dr, turn_dc
                    diagonal_phase = False
                    bends += 1
                    turned = True
                    break

            if not turned:
                break

    if return_bends:
        return out, bends
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Cast each colored seed toward its quadrant corner, bending blocked rays into outward cardinal tails.

    Input structure
    - The background is black, and one full green row plus one full green column form the central cross that splits the canvas into four quadrants.
    - Each quadrant contains one or two isolated single-pixel seeds in distinct non-black, non-green colors.

    Rule
    - Process the seeds in reading order from top to bottom, then left to right.
    - Each seed first paints a diagonal ray away from the green cross: upper-left seeds go northwest, upper-right seeds northeast, lower-left seeds southwest, and lower-right seeds southeast.
    - If the next diagonal cell is already occupied by another colored seed or trail, the ray turns into that quadrant's outward cardinal side (up, right, down, left in clockwise quadrant order); if that side is blocked it tries the other outward side, and after turning it continues straight until the next obstruction or the grid edge.

    Key insight
    - The green cross supplies the hidden direction label for every seed, while the final bent shapes depend on the accumulated colored blockers left by earlier rays.
    """
    return _simulate(grid)


def _quadrant_cells(shape: tuple[int, int], quad: str) -> list[tuple[int, int]]:
    h, w = shape
    mid_r = h // 2
    mid_c = w // 2
    if quad == "ul":
        rows = range(1, mid_r)
        cols = range(1, mid_c)
    elif quad == "ur":
        rows = range(1, mid_r)
        cols = range(mid_c + 1, w - 1)
    elif quad == "ll":
        rows = range(mid_r + 1, h - 1)
        cols = range(1, mid_c)
    else:
        assert quad == "lr"
        rows = range(mid_r + 1, h - 1)
        cols = range(mid_c + 1, w - 1)
    return [(r, c) for r in rows for c in cols]


def _sample_pair(shape: tuple[int, int], quad: str) -> list[tuple[int, int]] | None:
    cells = _quadrant_cells(shape, quad)
    cell_set = set(cells)
    diag_dr, diag_dc = QUADRANT_RULES[quad]["diag"]
    assert isinstance(diag_dr, int) and isinstance(diag_dc, int)

    for _ in range(200):
        outer = random.choice(cells)
        candidates: list[tuple[int, int]] = []
        for gap in range(2, 7):
            inner = (outer[0] - diag_dr * gap, outer[1] - diag_dc * gap)
            if inner in cell_set:
                candidates.append(inner)
        if candidates:
            inner = random.choice(candidates)
            return [outer, inner]
    return None


def _validate_input(grid: np.ndarray) -> bool:
    try:
        g = np.asarray(grid, dtype=int)
        if g.ndim != 2:
            return False
        h, w = g.shape
        if h % 2 != 0 or w % 2 != 0 or max(h, w) > 30:
            return False

        green, mid_r, mid_c = _find_cross(g)
        if green != GREEN:
            return False

        seeds = [(int(r), int(c)) for r, c in np.argwhere((g != BG) & (g != green))]
        if len(seeds) < 4 or len(seeds) > 8:
            return False

        seed_colors = [int(g[r, c]) for r, c in seeds]
        if len(set(seed_colors)) != len(seed_colors):
            return False

        counts = {quad: 0 for quad in QUADRANT_RULES}
        for r, c in seeds:
            counts[_quadrant(r, c, mid_r, mid_c)] += 1
        if any(count not in (1, 2) for count in counts.values()):
            return False
        if sum(count == 2 for count in counts.values()) < 1:
            return False

        out, bends = _simulate(g, return_bends=True)
        if bends < 1:
            return False
        if np.array_equal(out, g):
            return False
        if not np.array_equal(main(g), out):
            return False
        return True
    except Exception:
        return False


def _gen_attempt() -> np.ndarray | None:
    h = random.randrange(12, 31, 2)
    w = random.randrange(12, 31, 2)
    grid = np.full((h, w), BG, dtype=int)

    mid_r = h // 2
    mid_c = w // 2
    grid[mid_r, :] = GREEN
    grid[:, mid_c] = GREEN

    quadrants = ["ul", "ur", "ll", "lr"]
    paired_quadrants = set(random.sample(quadrants, random.randint(1, 4)))
    seed_total = 4 + len(paired_quadrants)
    colors = random.sample(SEED_COLORS, seed_total)
    color_idx = 0

    for quad in quadrants:
        if quad in paired_quadrants:
            positions = _sample_pair((h, w), quad)
            if positions is None:
                return None
        else:
            positions = [random.choice(_quadrant_cells((h, w), quad))]

        for r, c in positions:
            if int(grid[r, c]) != BG:
                return None
            grid[r, c] = colors[color_idx]
            color_idx += 1

    if _validate_input(grid):
        return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
