from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass

import numpy as np


BLACK = 0
BLUE = 1
SEED_COLORS = (2, 3, 4, 5, 6, 7, 8, 9)
MAX_ATTEMPTS = 400


@dataclass(frozen=True)
class Cup:
    r0: int
    c0: int
    height: int
    width: int

    @property
    def r1(self) -> int:
        return self.r0 + self.height - 1

    @property
    def c1(self) -> int:
        return self.c0 + self.width - 1

    @property
    def interior_rows(self) -> slice:
        return slice(self.r0 + 1, self.r1)

    @property
    def interior_cols(self) -> slice:
        return slice(self.c0 + 1, self.c1)

    def catches_column(self, col: int) -> bool:
        return self.c0 < col < self.c1

    def as_bbox(self) -> tuple[int, int, int, int]:
        return self.r0, self.c0, self.r1, self.c1


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(coords)
    return comps


def _cup_patch(height: int, width: int) -> np.ndarray:
    patch = np.zeros((height, width), dtype=int)
    patch[1:, 0] = BLUE
    patch[1:, -1] = BLUE
    patch[-1, :] = BLUE
    return patch


def _extract_cups(grid: np.ndarray) -> list[Cup]:
    cups: list[Cup] = []
    for coords in _components(grid == BLUE):
        blue_r0, c0, r1, c1 = _bbox(coords)
        if blue_r0 == 0:
            continue
        r0 = blue_r0 - 1
        height = r1 - r0 + 1
        width = c1 - c0 + 1
        if height not in (3, 4) or width not in (3, 4, 5):
            continue
        patch = grid[r0 : r1 + 1, c0 : c1 + 1]
        if np.array_equal(patch, _cup_patch(height, width)):
            cups.append(Cup(r0, c0, height, width))
    cups.sort(key=lambda cup: (cup.r0, cup.c0))
    return cups


def _seed_pixels(grid: np.ndarray) -> list[tuple[int, int, int]]:
    coords = np.argwhere((grid != BLACK) & (grid != BLUE))
    seeds = [(int(r), int(c), int(grid[r, c])) for r, c in coords]
    seeds.sort()
    return seeds


def main(input_grid: np.ndarray) -> np.ndarray:
    """
    Extend top seed rays downward until they either reach the floor or get captured by a blue cup outline.

    Input structure:
    - The background is black, with 2 to 4 non-blue singleton seeds near the top rows.
    - Blue foreground components are upright U-shaped cup outlines with black interiors and open tops.
    - Seed columns are distinct, and any cup that aligns with a seed contains that seed column strictly between its walls.

    Rule:
    - For each non-black, non-blue seed, scan downward in the same column.
    - Paint ordinary black cells with the seed color until the scan reaches a cup opening.
    - When the scan reaches the opening row of a cup whose walls straddle that column, fill that cup's interior black cells with the seed color and stop that ray.
    - If no cup captures the ray, continue painting straight down to the bottom; blue outline pixels and original seeds remain unchanged.

    Key insight:
    - A seed first behaves like a one-dimensional downward ray, but a hit cup converts that ray into a two-dimensional interior fill and simultaneously blocks further propagation.
    """
    grid = np.asarray(input_grid, dtype=int)
    out = grid.copy()
    cups = _extract_cups(grid)
    seeds = _seed_pixels(grid)

    for seed_r, seed_c, color in seeds:
        for row in range(seed_r + 1, grid.shape[0]):
            captured = next((cup for cup in cups if cup.r0 == row and cup.catches_column(seed_c)), None)
            if captured is not None:
                interior = out[captured.interior_rows, captured.interior_cols]
                interior[interior == BLACK] = color
                break
            if out[row, seed_c] == BLACK:
                out[row, seed_c] = color
            else:
                break

    return out


def _sample_spaced_columns(width: int, count: int, min_gap: int = 4) -> list[int] | None:
    cols = list(range(1, width - 1))
    for _ in range(80):
        random.shuffle(cols)
        chosen: list[int] = []
        for col in cols:
            if all(abs(col - other) >= min_gap for other in chosen):
                chosen.append(col)
                if len(chosen) == count:
                    chosen.sort()
                    return chosen
    return None


def _interval_candidates_for_seed(seed_col: int, all_seed_cols: set[int], width: int, grid_w: int) -> list[tuple[int, int]]:
    options: list[tuple[int, int]] = []
    left_min = max(0, seed_col - (width - 2))
    left_max = min(seed_col - 1, grid_w - width)
    for left in range(left_min, left_max + 1):
        right = left + width - 1
        cols = set(range(left, right + 1))
        if seed_col not in cols:
            continue
        if any(other in cols for other in all_seed_cols if other != seed_col):
            continue
        options.append((left, right))
    random.shuffle(options)
    return options


def _interval_candidates_without_seed(all_seed_cols: set[int], width: int, grid_w: int) -> list[tuple[int, int]]:
    options: list[tuple[int, int]] = []
    for left in range(0, grid_w - width + 1):
        right = left + width - 1
        cols = set(range(left, right + 1))
        if cols & all_seed_cols:
            continue
        options.append((left, right))
    random.shuffle(options)
    return options


def _interval_conflicts(interval: tuple[int, int], chosen: list[tuple[int, int]]) -> bool:
    left, right = interval
    for other_left, other_right in chosen:
        if not (right + 1 < other_left or other_right + 1 < left):
            return True
    return False


def _choose_intervals(specs: list[int | None], grid_w: int, all_seed_cols: list[int]) -> list[tuple[int, int]] | None:
    seed_set = set(all_seed_cols)
    candidates: list[list[tuple[int, int]]] = []
    for spec in specs:
        options: list[tuple[int, int]] = []
        widths = [3, 4, 5]
        random.shuffle(widths)
        for width in widths:
            if spec is None:
                options.extend(_interval_candidates_without_seed(seed_set, width, grid_w))
            else:
                options.extend(_interval_candidates_for_seed(spec, seed_set, width, grid_w))
        unique = list(dict.fromkeys(options))
        if not unique:
            return None
        candidates.append(unique)

    order = sorted(range(len(specs)), key=lambda idx: len(candidates[idx]))
    chosen: list[tuple[int, int] | None] = [None] * len(specs)
    occupied: list[tuple[int, int]] = []

    def backtrack(pos: int) -> bool:
        if pos == len(order):
            return True
        idx = order[pos]
        for interval in candidates[idx]:
            if _interval_conflicts(interval, occupied):
                continue
            chosen[idx] = interval
            occupied.append(interval)
            if backtrack(pos + 1):
                return True
            occupied.pop()
            chosen[idx] = None
        return False

    if not backtrack(0):
        return None
    return [interval for interval in chosen if interval is not None]


def _draw_cup(grid: np.ndarray, cup: Cup) -> None:
    grid[cup.r0 + 1 : cup.r1 + 1, cup.c0] = BLUE
    grid[cup.r0 + 1 : cup.r1 + 1, cup.c1] = BLUE
    grid[cup.r1, cup.c0 : cup.c1 + 1] = BLUE


def _build_input(height: int, width: int, seeds: list[tuple[int, int, int]], cups: list[Cup]) -> np.ndarray:
    grid = np.zeros((height, width), dtype=int)
    for row, col, color in seeds:
        grid[row, col] = color
    for cup in cups:
        _draw_cup(grid, cup)
    return grid


def _expected_from_metadata(height: int, width: int, seeds: list[tuple[int, int, int]], cups: list[Cup]) -> np.ndarray:
    out = _build_input(height, width, seeds, cups)
    ordered_cups = sorted(cups, key=lambda cup: (cup.r0, cup.c0))
    for seed_r, seed_c, color in sorted(seeds):
        target = next((cup for cup in ordered_cups if cup.catches_column(seed_c)), None)
        if target is None:
            out[seed_r + 1 :, seed_c] = color
            continue
        out[seed_r + 1 : target.r0, seed_c] = color
        interior = out[target.interior_rows, target.interior_cols]
        interior[interior == BLACK] = color
    return out


def _validate_case(height: int, width: int, seeds: list[tuple[int, int, int]], cups: list[Cup], inp: np.ndarray) -> bool:
    if max(inp.shape) > 30:
        return False

    detected = _extract_cups(inp)
    if sorted(cup.as_bbox() for cup in detected) != sorted(cup.as_bbox() for cup in cups):
        return False

    hit_cups = []
    for cup in cups:
        aligned = [seed for seed in seeds if cup.catches_column(seed[1])]
        if len(aligned) > 1:
            return False
        if aligned:
            hit_cups.append((cup, aligned[0]))
    if not hit_cups or len(hit_cups) == len(cups):
        return False

    bottom_seeds = [seed for seed in seeds if all(not cup.catches_column(seed[1]) for cup in cups)]
    if not bottom_seeds:
        return False

    expected = _expected_from_metadata(height, width, seeds, cups)
    solved = main(inp)
    if not np.array_equal(solved, expected):
        return False

    for cup, seed in hit_cups:
        interior = solved[cup.interior_rows, cup.interior_cols]
        if interior.size == 0 or not np.all(interior == seed[2]):
            return False
        if cup.r1 + 1 < height and solved[cup.r1 + 1, seed[1]] == seed[2]:
            return False

    for cup in cups:
        if any(cup is hit for hit, _seed in hit_cups):
            continue
        interior = solved[cup.interior_rows, cup.interior_cols]
        if np.any(interior != BLACK):
            return False

    for seed_r, seed_c, color in bottom_seeds:
        if not np.all(solved[seed_r + 1 :, seed_c] == color):
            return False

    return True


def _generate_attempt() -> np.ndarray | None:
    height = random.randint(13, 18)
    width = random.randint(16, 22)
    n_seeds = random.randint(2, 4)
    n_cups = random.randint(2, 4)

    seed_cols = _sample_spaced_columns(width, n_seeds)
    if seed_cols is None:
        return None

    max_hits = min(n_seeds - 1, n_cups - 1)
    if max_hits < 1:
        return None
    n_hits = random.randint(1, max_hits)
    hit_seed_indices = random.sample(range(n_seeds), n_hits)
    specs: list[int | None] = [seed_cols[idx] for idx in hit_seed_indices]
    specs.extend([None] * (n_cups - n_hits))
    intervals = _choose_intervals(specs, width, seed_cols)
    if intervals is None:
        return None

    seed_rows = [random.randint(0, 1) for _ in range(n_seeds)]
    if len(set(seed_rows)) == 1 and random.random() < 0.7:
        seed_rows[random.randrange(n_seeds)] = 1 - seed_rows[0]
    seed_colors = random.sample(SEED_COLORS, n_seeds)
    seeds = [(seed_rows[i], seed_cols[i], seed_colors[i]) for i in range(n_seeds)]

    cups: list[Cup] = []
    top_min = max(4, height // 3)
    for left, right in intervals:
        cup_width = right - left + 1
        cup_height = random.choice((3, 4))
        top_max = height - cup_height - 1
        if top_min > top_max:
            return None
        cup_top = random.randint(top_min, top_max)
        cups.append(Cup(cup_top, left, cup_height, cup_width))

    inp = _build_input(height, width, seeds, cups)
    if not _validate_case(height, width, seeds, cups, inp):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_ATTEMPTS} attempts")
