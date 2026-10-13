from __future__ import annotations

import random
from typing import Iterable

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4

GRID_HEIGHT_RANGE = (11, 16)
GRID_WIDTH_OPTIONS = [12, 14, 16, 18]
SEED_COUNT_OPTIONS = [2, 3]
RED_COUNT_OPTIONS = [1, 2]
HOLE_COUNT_OPTIONS = [2, 3]
MAX_GENERATION_ATTEMPTS = 400


def _mirror_col(width: int, col: int) -> int:
    return int(width - 1 - col)


def _seed_positions(grid: np.ndarray) -> list[tuple[int, int]]:
    return sorted((int(r), int(c)) for r, c in np.argwhere(grid == GREEN))


def _complete_blue_symmetry(grid: np.ndarray) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    height, width = out.shape
    for r in range(height):
        for c in range(width // 2):
            mc = _mirror_col(width, c)
            left = int(out[r, c])
            right = int(out[r, mc])
            if left == BLUE and right == BLACK:
                out[r, mc] = BLUE
            elif right == BLUE and left == BLACK:
                out[r, c] = BLUE
    return out


def _cast_rays(base: np.ndarray, obstacle_grid: np.ndarray | None = None) -> np.ndarray:
    obstacle = np.asarray(base if obstacle_grid is None else obstacle_grid, dtype=int)
    out = np.asarray(base, dtype=int).copy()
    height, _width = out.shape

    for seed_r, seed_c in _seed_positions(out):
        for r in range(seed_r - 1, -1, -1):
            if int(obstacle[r, seed_c]) != BLACK:
                break
            out[r, seed_c] = YELLOW
    return out


def _first_blocker_row(grid: np.ndarray, seed_r: int, seed_c: int) -> int | None:
    for r in range(seed_r - 1, -1, -1):
        if int(grid[r, seed_c]) != BLACK:
            return int(r)
    return None


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete the nearly mirrored blue obstacle pattern, then let bottom seeds shoot upward until the repaired obstacles stop them.

    Input structure:
    - Black is the background and the grid width is even, so every column has a left-right mirror partner.
    - Blue pixels form an almost vertically mirrored pattern in the upper half, with a few one-sided holes where one mirror cell is blue and the other is black.
    - Red pixels are isolated mid-grid obstacles, and green pixels are seed markers near the bottom.

    Rule:
    - First repair the blue pattern by left-right mirror completion: for each row and each mirrored column pair, if one side is blue and the other side is black, fill the black side with blue.
    - After that repair, cast an upward ray from every green seed in its own column; every black cell on the ray becomes yellow.
    - Stop a ray before the first non-black cell in the repaired grid, or at the top boundary if no obstacle exists; keep blue, red, and green cells unchanged.

    Key insight:
    - The yellow result is controlled by the repaired obstacle landscape, so some rays stop early only because symmetry completion created new blue blockers.
    """
    g = np.asarray(grid, dtype=int)
    completed = _complete_blue_symmetry(g)
    return _cast_rays(completed, obstacle_grid=completed)


def _choose_seed_columns(width: int, count: int) -> list[int] | None:
    cols = list(range(1, width - 1))
    random.shuffle(cols)
    chosen: list[int] = []
    for c in cols:
        if any(abs(c - old) < 2 for old in chosen):
            continue
        if _mirror_col(width, c) in chosen:
            continue
        chosen.append(c)
        if len(chosen) == count:
            return sorted(chosen)
    return None


def _sample_distinct(values: Iterable[int], count: int) -> list[int] | None:
    pool = list(values)
    if len(pool) < count:
        return None
    return random.sample(pool, count)


def _build_mode_plan(seed_count: int) -> list[str]:
    if seed_count == 2:
        other = random.choices(["red", "blue", "top"], weights=[4, 3, 3], k=1)
        return ["hole", *other]

    if random.random() < 0.75:
        other = random.sample(["red", "blue", "top"], 2)
    else:
        other = random.choices(["red", "blue", "top"], weights=[4, 3, 3], k=2)
    random.shuffle(other)
    return ["hole", *other]


def _blocked_rows_by_mode(height: int, modes: list[str]) -> dict[int, int]:
    top_max = height // 2
    blue_needed = sum(mode in {"hole", "blue"} for mode in modes)
    red_needed = sum(mode == "red" for mode in modes)

    blue_rows = _sample_distinct(range(1, top_max + 1), blue_needed)
    red_rows = _sample_distinct(range(top_max + 1, height - 3), red_needed)
    if blue_rows is None or red_rows is None:
        raise ValueError("not enough rows to assign blockers")

    random.shuffle(blue_rows)
    random.shuffle(red_rows)
    plan: dict[int, int] = {}
    blue_idx = 0
    red_idx = 0
    for idx, mode in enumerate(modes):
        if mode in {"hole", "blue"}:
            plan[idx] = int(blue_rows[blue_idx])
            blue_idx += 1
        elif mode == "red":
            plan[idx] = int(red_rows[red_idx])
            red_idx += 1
    return plan


def _can_use_blue_cell(
    row: int,
    col: int,
    allowed_seed_blue: set[tuple[int, int]],
    reserved_seed_cols: set[int],
    height: int,
) -> bool:
    if not (0 <= row < height):
        return False
    if col in reserved_seed_cols and (row, col) not in allowed_seed_blue:
        return False
    return True


def _add_blue_pair(
    blue: np.ndarray,
    cells: Iterable[tuple[int, int]],
    allowed_seed_blue: set[tuple[int, int]],
    reserved_seed_cols: set[int],
) -> bool:
    height, width = blue.shape
    full_cells: set[tuple[int, int]] = set()
    for row, col in cells:
        row = int(row)
        col = int(col)
        mirror = _mirror_col(width, col)
        for rr, cc in ((row, col), (row, mirror)):
            if not _can_use_blue_cell(rr, cc, allowed_seed_blue, reserved_seed_cols, height):
                return False
            full_cells.add((rr, cc))

    for row, col in full_cells:
        blue[row, col] = True
    return True


def _random_object_cells(height: int, width: int, left_limit: int) -> list[tuple[int, int]]:
    shape = random.choice(["h1", "h2", "h3", "v2", "v3", "b2"])
    top_max = height // 2

    if shape.startswith("h"):
        length = int(shape[1])
        row = random.randint(1, top_max)
        col = random.randint(1, left_limit - length)
        return [(row, col + dc) for dc in range(length)]

    if shape.startswith("v"):
        length = int(shape[1])
        row = random.randint(1, top_max - length + 1)
        col = random.randint(1, left_limit - 1)
        return [(row + dr, col) for dr in range(length)]

    row = random.randint(1, top_max - 1)
    col = random.randint(1, left_limit - 2)
    return [(row + dr, col + dc) for dr in range(2) for dc in range(2)]


def _blue_pair_candidates(
    blue: np.ndarray,
    excluded_cols: set[int],
    protected_pairs: set[tuple[int, int]],
) -> list[tuple[int, int]]:
    height, width = blue.shape
    pairs: list[tuple[int, int]] = []
    for row in range(height):
        for left_col in range(width // 2):
            right_col = _mirror_col(width, left_col)
            if left_col in excluded_cols or right_col in excluded_cols:
                continue
            if (row, left_col) in protected_pairs:
                continue
            if blue[row, left_col] and blue[row, right_col]:
                pairs.append((row, left_col))
    return pairs


def _isolated_red_ok(grid: np.ndarray, row: int, col: int) -> bool:
    height, width = grid.shape
    if not (0 <= row < height and 0 <= col < width):
        return False
    if int(grid[row, col]) != BLACK:
        return False
    for rr in range(max(0, row - 1), min(height, row + 2)):
        for cc in range(max(0, col - 1), min(width, col + 2)):
            if (rr, cc) == (row, col):
                continue
            if int(grid[rr, cc]) != BLACK:
                return False
    return True


def _compose_input(
    blue: np.ndarray,
    hole_specs: list[tuple[int, int, str]],
    red_cells: list[tuple[int, int]],
    seed_cols: list[int],
    seed_row: int,
) -> np.ndarray:
    height, width = blue.shape
    grid = np.zeros((height, width), dtype=int)
    grid[blue] = BLUE

    for row, left_col, side in hole_specs:
        target_col = left_col if side == "left" else _mirror_col(width, left_col)
        grid[row, target_col] = BLACK

    for row, col in red_cells:
        grid[row, col] = RED
    for col in seed_cols:
        grid[seed_row, col] = GREEN
    return grid


def _semantic_checks(
    inp: np.ndarray,
    full_blue: np.ndarray,
    modes: list[str],
    seed_cols: list[int],
    blocker_rows: dict[int, int],
    special_seed_idx: int,
    special_hole_row: int,
) -> bool:
    height, width = inp.shape
    seed_row = height - 2
    completed = _complete_blue_symmetry(inp)

    if not np.array_equal(completed == BLUE, full_blue):
        return False
    if np.count_nonzero(inp == RED) not in RED_COUNT_OPTIONS:
        return False
    if np.count_nonzero(inp == GREEN) != len(seed_cols):
        return False
    if not np.all(inp[height - 1, :] == BLACK):
        return False

    hole_count = 0
    for row in range(height):
        for left_col in range(width // 2):
            right_col = _mirror_col(width, left_col)
            left = int(inp[row, left_col])
            right = int(inp[row, right_col])
            if (left, right) in {(BLUE, BLACK), (BLACK, BLUE)}:
                hole_count += 1
    if hole_count not in HOLE_COUNT_OPTIONS:
        return False

    raw_rays = _cast_rays(inp, obstacle_grid=inp)
    fixed_rays = main(inp)
    if np.array_equal(raw_rays, fixed_rays):
        return False

    special_col = seed_cols[special_seed_idx]
    fixed_hit = _first_blocker_row(completed, seed_row, special_col)
    raw_hit = _first_blocker_row(inp, seed_row, special_col)
    if fixed_hit != special_hole_row:
        return False
    if raw_hit == fixed_hit:
        return False
    if int(inp[special_hole_row, special_col]) != BLACK or int(completed[special_hole_row, special_col]) != BLUE:
        return False

    for idx, (mode, col) in enumerate(zip(modes, seed_cols)):
        blocker = _first_blocker_row(completed, seed_row, col)
        if mode == "top":
            if blocker is not None:
                return False
            continue

        expected_row = blocker_rows[idx]
        if blocker != expected_row:
            return False
        color = int(completed[blocker, col])
        if mode == "red" and color != RED:
            return False
        if mode == "blue" and color != BLUE:
            return False
        if mode == "hole":
            if color != BLUE or int(inp[blocker, col]) != BLACK:
                return False

    for row, col in np.argwhere(inp == RED):
        r = int(row)
        c = int(col)
        for rr in range(max(0, r - 1), min(height, r + 2)):
            for cc in range(max(0, c - 1), min(width, c + 2)):
                if (rr, cc) == (r, c):
                    continue
                if int(inp[rr, cc]) != BLACK:
                    return False

    return True


def _attempt_generate() -> np.ndarray | None:
    height = random.randint(*GRID_HEIGHT_RANGE)
    width = random.choice(GRID_WIDTH_OPTIONS)
    seed_count = random.choice(SEED_COUNT_OPTIONS)
    seed_cols = _choose_seed_columns(width, seed_count)
    if seed_cols is None:
        return None

    seed_row = height - 2
    modes = _build_mode_plan(seed_count)
    special_seed_idx = 0
    blocker_rows = _blocked_rows_by_mode(height, modes)

    blue = np.zeros((height, width), dtype=bool)
    reserved_seed_cols = set(seed_cols)
    allowed_seed_blue: set[tuple[int, int]] = set()
    protected_pairs: set[tuple[int, int]] = set()

    special_col = seed_cols[special_seed_idx]
    special_hole_row = blocker_rows[special_seed_idx]
    special_left = min(special_col, _mirror_col(width, special_col))
    protected_pairs.add((special_hole_row, special_left))

    for idx, (mode, col) in enumerate(zip(modes, seed_cols)):
        if mode not in {"hole", "blue"}:
            continue
        row = blocker_rows[idx]
        mirror = _mirror_col(width, col)
        allowed_seed_blue.add((row, col))
        allowed_seed_blue.add((row, mirror))
        blue[row, col] = True
        blue[row, mirror] = True
        protected_pairs.add((row, min(col, mirror)))

    left_limit = width // 2 - 1
    object_target = random.randint(4, 7)
    for _ in range(object_target * 8):
        if int(np.count_nonzero(blue)) >= object_target * 4:
            break
        cells = _random_object_cells(height, width, left_limit)
        _add_blue_pair(blue, cells, allowed_seed_blue, reserved_seed_cols)

    hole_total = random.choice(HOLE_COUNT_OPTIONS)
    extra_holes_needed = hole_total - 1
    excluded_for_extra_holes = set(seed_cols)

    for _ in range(40):
        candidates = _blue_pair_candidates(blue, excluded_for_extra_holes, protected_pairs)
        if len(candidates) >= extra_holes_needed:
            break
        row = random.randint(1, height // 2)
        left_col_choices = [
            c
            for c in range(1, width // 2)
            if c not in excluded_for_extra_holes and _mirror_col(width, c) not in excluded_for_extra_holes
        ]
        if not left_col_choices:
            return None
        left_col = random.choice(left_col_choices)
        if (row, left_col) in protected_pairs:
            continue
        if not _add_blue_pair(blue, [(row, left_col)], allowed_seed_blue, reserved_seed_cols):
            continue

    extra_candidates = _blue_pair_candidates(blue, excluded_for_extra_holes, protected_pairs)
    if len(extra_candidates) < extra_holes_needed:
        return None

    chosen_extra_holes = random.sample(extra_candidates, extra_holes_needed)
    hole_specs: list[tuple[int, int, str]] = [(special_hole_row, special_left, "left" if special_col == special_left else "right")]
    for row, left_col in chosen_extra_holes:
        hole_specs.append((int(row), int(left_col), random.choice(["left", "right"])))

    red_cells: list[tuple[int, int]] = []
    base_grid = _compose_input(blue, hole_specs, red_cells, seed_cols, seed_row)

    for idx, (mode, col) in enumerate(zip(modes, seed_cols)):
        if mode != "red":
            continue
        row = blocker_rows[idx]
        if not _isolated_red_ok(base_grid, row, col):
            return None
        red_cells.append((row, col))
        base_grid[row, col] = RED

    red_total = random.choice(RED_COUNT_OPTIONS)
    while len(red_cells) < red_total:
        candidates = [
            (row, col)
            for row in range(height // 2, height - 3)
            for col in range(1, width - 1)
            if col not in seed_cols
        ]
        random.shuffle(candidates)
        placed = False
        for row, col in candidates:
            if _isolated_red_ok(base_grid, row, col):
                red_cells.append((row, col))
                base_grid[row, col] = RED
                placed = True
                break
        if not placed:
            return None

    inp = _compose_input(blue, hole_specs, red_cells, seed_cols, seed_row)
    if not _semantic_checks(inp, blue, modes, seed_cols, blocker_rows, special_seed_idx, special_hole_row):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: failed to build a valid symmetry-ray puzzle")
