from __future__ import annotations

from collections import Counter
from itertools import product
import random

import numpy as np


BG = 0
PATH = 5
VOTE_COLORS = (1, 2, 3, 4, 6, 8, 9)

GRID_H_RANGE = (17, 28)
GRID_W_RANGE = (17, 28)
HORIZONTAL_COUNT_RANGE = (2, 4)
VERTICAL_COUNT_RANGE = (2, 4)
AXIS_MARGIN = 2
MIN_AXIS_SEPARATION = 3
PALETTE_SIZE_CHOICES = (2, 3)
GREY_PROB_CHOICES = (0.15, 0.25, 0.35)
FORCE_NEUTRAL_PATH_PROB = 0.4
FORCE_TIE_INTERSECTION_PROB = 0.5
MAX_GENERATION_ATTEMPTS = 400

DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]
PERP_DIRS = {
    (-1, 0): [(0, -1), (0, 1)],
    (1, 0): [(0, -1), (0, 1)],
    (0, 1): [(-1, 0), (1, 0)],
    (0, -1): [(-1, 0), (1, 0)],
}


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _is_marker(value: int) -> bool:
    return value not in (BG, PATH)


def _majority_by_order(colors: list[int]) -> int | None:
    if not colors:
        return None

    counts = Counter(int(color) for color in colors)
    best = max(counts.values())
    winners = {color for color, count in counts.items() if count == best}
    for color in colors:
        if color in winners:
            return int(color)
    raise AssertionError("unreachable majority selection state")


def _has_top_tie(colors: list[int]) -> bool:
    if not colors:
        return False
    counts = Counter(colors)
    best = max(counts.values())
    return sum(count == best for count in counts.values()) > 1


def _scene_masks(path_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    left = np.zeros_like(path_mask, dtype=bool)
    right = np.zeros_like(path_mask, dtype=bool)
    up = np.zeros_like(path_mask, dtype=bool)
    down = np.zeros_like(path_mask, dtype=bool)

    left[:, 1:] = path_mask[:, :-1]
    right[:, :-1] = path_mask[:, 1:]
    up[1:, :] = path_mask[:-1, :]
    down[:-1, :] = path_mask[1:, :]

    horizontal = path_mask & (left | right)
    vertical = path_mask & (up | down)
    intersections = path_mask & horizontal & vertical
    return horizontal, vertical, intersections


def _arm_vote(grid: np.ndarray, arm_r: int, arm_c: int, dr: int, dc: int) -> int | None:
    if not _in_bounds(grid.shape, arm_r, arm_c):
        return None

    direct = int(grid[arm_r, arm_c])
    if _is_marker(direct):
        return direct
    if direct != PATH:
        return None

    for sdr, sdc in PERP_DIRS[(dr, dc)]:
        rr = arm_r + sdr
        cc = arm_c + sdc
        if _in_bounds(grid.shape, rr, cc) and _is_marker(int(grid[rr, cc])):
            return int(grid[rr, cc])
    return None


def _intersection_colors(grid: np.ndarray, intersections: np.ndarray) -> np.ndarray:
    colors = np.full(grid.shape, PATH, dtype=int)
    for r, c in np.argwhere(intersections):
        ordered_votes: list[int] = []
        for dr, dc in DIRS:
            vote = _arm_vote(grid, int(r + dr), int(c + dc), dr, dc)
            if vote is not None:
                ordered_votes.append(int(vote))
        picked = _majority_by_order(ordered_votes)
        if picked is not None:
            colors[int(r), int(c)] = int(picked)
    return colors


def _row_runs(mask: np.ndarray):
    h, w = mask.shape
    for r in range(h):
        c = 0
        while c < w:
            if not bool(mask[r, c]):
                c += 1
                continue
            c0 = c
            while c < w and bool(mask[r, c]):
                c += 1
            yield int(r), int(c0), int(c - 1)


def _col_runs(mask: np.ndarray):
    h, w = mask.shape
    for c in range(w):
        r = 0
        while r < h:
            if not bool(mask[r, c]):
                r += 1
                continue
            r0 = r
            while r < h and bool(mask[r, c]):
                r += 1
            yield int(c), int(r0), int(r - 1)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Color intersections by nearby votes, then color each straight path by the majority of its colored intersections.

    Input structure
    - Background is black.
    - Grey pixels form 1-cell-wide horizontal and vertical straight paths.
    - Colored vote markers sit next to non-intersection path cells.
    Rule
    - Each intersection collects at most one vote from each direction in up/right/down/left order.
    - A direction votes via a marker on the arm next to that intersection.
    - The intersection takes the majority vote color, breaking ties by first appearance in up/right/down/left order.
    - Each horizontal or vertical path recolors its non-intersection grey pixels to the majority color of its colored intersections.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    path_mask = g == PATH
    assert np.any(path_mask), "expected at least one grey path"

    horizontal, vertical, intersections = _scene_masks(path_mask)
    cross_colors = _intersection_colors(g, intersections)

    out = g.copy()
    recolored_cross = intersections & (cross_colors != PATH)
    out[recolored_cross] = cross_colors[recolored_cross]

    for r, c0, c1 in _row_runs(horizontal):
        ordered = [
            int(cross_colors[r, c])
            for c in range(c0, c1 + 1)
            if bool(intersections[r, c]) and int(cross_colors[r, c]) != PATH
        ]
        picked = _majority_by_order(ordered)
        if picked is None:
            continue
        for c in range(c0, c1 + 1):
            if bool(path_mask[r, c]) and not bool(intersections[r, c]):
                out[r, c] = int(picked)

    for c, r0, r1 in _col_runs(vertical):
        ordered = [
            int(cross_colors[r, c])
            for r in range(r0, r1 + 1)
            if bool(intersections[r, c]) and int(cross_colors[r, c]) != PATH
        ]
        picked = _majority_by_order(ordered)
        if picked is None:
            continue
        for r in range(r0, r1 + 1):
            if bool(path_mask[r, c]) and not bool(intersections[r, c]):
                out[r, c] = int(picked)

    return out


def _sample_positions(lo: int, hi: int, count: int, min_sep: int) -> list[int] | None:
    if hi - lo + 1 < count:
        return None
    for _ in range(120):
        values = sorted(random.sample(range(lo, hi + 1), count))
        if all(b - a >= min_sep for a, b in zip(values, values[1:])):
            return values
    return None


def _line_majority(values: list[int]) -> int | None:
    colored = [int(value) for value in values if int(value) != PATH]
    if not colored:
        return PATH
    counts = Counter(colored)
    best = max(counts.values())
    winners = [color for color, count in counts.items() if count == best]
    if len(winners) != 1:
        return None
    return int(winners[0])


def _sample_color_matrix(nh: int, nv: int, palette: list[int]) -> np.ndarray | None:
    need_neutral_path = random.random() < FORCE_NEUTRAL_PATH_PROB
    grey_prob = random.choice(GREY_PROB_CHOICES)
    neutral_axis: tuple[str, int] | None = None
    if need_neutral_path:
        if random.random() < 0.5:
            neutral_axis = ("row", random.randrange(nh))
        else:
            neutral_axis = ("col", random.randrange(nv))

    for _ in range(200):
        matrix = np.full((nh, nv), PATH, dtype=int)
        for r in range(nh):
            for c in range(nv):
                if neutral_axis == ("row", r) or neutral_axis == ("col", c):
                    continue
                if random.random() < grey_prob:
                    continue
                matrix[r, c] = random.choice(palette)

        if not np.any(matrix != PATH):
            continue
        if nh * nv > 1 and not np.any(matrix == PATH):
            continue

        row_majorities = [_line_majority(matrix[r, :].tolist()) for r in range(nh)]
        col_majorities = [_line_majority(matrix[:, c].tolist()) for c in range(nv)]
        if any(value is None for value in row_majorities + col_majorities):
            continue
        if need_neutral_path and PATH not in row_majorities + col_majorities:
            continue

        row_colored = np.sum(matrix != PATH, axis=1)
        col_colored = np.sum(matrix != PATH, axis=0)
        if int(max(np.max(row_colored), np.max(col_colored))) < 2:
            continue

        used_colors = {int(value) for value in matrix.reshape(-1) if int(value) != PATH}
        if len(used_colors) == 1 and random.random() < 0.5:
            continue
        return matrix
    return None


def _assignment_result(assignment: tuple[int | None, ...]) -> int | None:
    ordered = [int(color) for color in assignment if color is not None]
    return _majority_by_order(ordered)


def _sample_assignment(target: int, palette: list[int], require_tie: bool) -> tuple[int | None, ...] | None:
    tie_options: list[tuple[int | None, ...]] = []
    plain_options: list[tuple[int | None, ...]] = []
    for assignment in product([None] + palette, repeat=4):
        ordered = [int(color) for color in assignment if color is not None]
        if not ordered:
            continue
        if _assignment_result(assignment) != int(target):
            continue
        if _has_top_tie(ordered):
            tie_options.append(tuple(assignment))
        else:
            plain_options.append(tuple(assignment))

    if require_tie:
        options = tie_options
    elif plain_options and tie_options and random.random() < 0.2:
        options = tie_options
    else:
        options = plain_options or tie_options

    if not options:
        return None
    return random.choice(options)


def _build_input(
    height: int,
    width: int,
    rows: list[int],
    cols: list[int],
    matrix: np.ndarray,
    palette: list[int],
) -> np.ndarray | None:
    grid = np.full((height, width), BG, dtype=int)

    h_start = max(0, min(cols) - random.randint(1, 2))
    h_end = min(width - 1, max(cols) + random.randint(1, 2))
    v_start = max(0, min(rows) - random.randint(1, 2))
    v_end = min(height - 1, max(rows) + random.randint(1, 2))

    if h_start >= min(cols) or h_end <= max(cols) or v_start >= min(rows) or v_end <= max(rows):
        return None

    for r in rows:
        grid[r, h_start : h_end + 1] = PATH
    for c in cols:
        grid[v_start : v_end + 1, c] = PATH

    colored_intersections = [(r_idx, c_idx) for r_idx in range(len(rows)) for c_idx in range(len(cols)) if int(matrix[r_idx, c_idx]) != PATH]
    if not colored_intersections:
        return None

    require_tie = random.random() < FORCE_TIE_INTERSECTION_PROB and len(palette) >= 2
    tie_target = random.choice(colored_intersections) if require_tie else None

    occupied_markers: set[tuple[int, int]] = set()
    for r_idx, r in enumerate(rows):
        for c_idx, c in enumerate(cols):
            target = int(matrix[r_idx, c_idx])
            if target == PATH:
                continue

            assignment = _sample_assignment(
                target=target,
                palette=palette,
                require_tie=tie_target == (r_idx, c_idx),
            )
            if assignment is None:
                return None

            for direction_index, color in enumerate(assignment):
                if color is None:
                    continue
                dr, dc = DIRS[direction_index]
                arm_r = r + dr
                arm_c = c + dc
                if not _in_bounds(grid.shape, arm_r, arm_c) or int(grid[arm_r, arm_c]) != PATH:
                    return None

                candidates = []
                for sdr, sdc in PERP_DIRS[(dr, dc)]:
                    rr = arm_r + sdr
                    cc = arm_c + sdc
                    if not _in_bounds(grid.shape, rr, cc):
                        continue
                    if int(grid[rr, cc]) != BG:
                        continue
                    if (rr, cc) in occupied_markers:
                        continue
                    candidates.append((rr, cc))

                if not candidates:
                    return None
                rr, cc = random.choice(candidates)
                grid[rr, cc] = int(color)
                occupied_markers.add((rr, cc))

    return grid


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(*GRID_H_RANGE)
    width = random.randint(*GRID_W_RANGE)
    nh = random.randint(*HORIZONTAL_COUNT_RANGE)
    nv = random.randint(*VERTICAL_COUNT_RANGE)
    if height < 2 * AXIS_MARGIN + 1 or width < 2 * AXIS_MARGIN + 1:
        return None

    rows = _sample_positions(AXIS_MARGIN, height - AXIS_MARGIN - 1, nh, MIN_AXIS_SEPARATION)
    cols = _sample_positions(AXIS_MARGIN, width - AXIS_MARGIN - 1, nv, MIN_AXIS_SEPARATION)
    if rows is None or cols is None:
        return None

    palette = random.sample(list(VOTE_COLORS), k=random.choice(PALETTE_SIZE_CHOICES))
    matrix = _sample_color_matrix(nh, nv, palette)
    if matrix is None:
        return None

    grid = _build_input(height, width, rows, cols, matrix, palette)
    if grid is None:
        return None

    try:
        out = main(grid)
    except AssertionError:
        return None

    if np.array_equal(out, grid):
        return None

    path_mask = grid == PATH
    horizontal, vertical, intersections = _scene_masks(path_mask)
    if int(np.sum(intersections)) != nh * nv:
        return None
    if not np.any(out[intersections] != PATH):
        return None
    if not np.any((out != grid) & path_mask & ~intersections):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
