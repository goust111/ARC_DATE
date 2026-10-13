from __future__ import annotations

import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3

GRID_H_RANGE = (10, 14)
GRID_W_RANGE = (10, 14)
SOURCE_COUNT_RANGE = (3, 5)
GENERATION_RETRIES = 400

MODE_WEIGHTS = (
    ("none", 0.35),
    ("pair", 0.35),
    ("triple", 0.15),
    ("double_pair", 0.15),
)


def _shadow_counts(shape: tuple[int, int], sources: list[tuple[int, int]]) -> np.ndarray:
    counts = np.zeros(shape, dtype=int)
    h, w = shape
    for r, c in sources:
        rr, cc = r - 1, c + 1
        while 0 <= rr < h and 0 <= cc < w:
            counts[rr, cc] += 1
            rr -= 1
            cc += 1
    return counts


def _render_from_sources(shape: tuple[int, int], sources: list[tuple[int, int]]) -> np.ndarray:
    grid = np.full(shape, BLACK, dtype=int)
    for r, c in sources:
        grid[r, c] = BLUE

    counts = _shadow_counts(shape, sources)
    out = grid.copy()
    bg = grid == BLACK
    out[bg & (counts == 1)] = RED
    out[bg & (counts >= 2)] = GREEN
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Cast a diagonal shadow from each blue pixel and recolor shared shadow cells by overlap count.

    Input structure:
    - The grid is mostly black.
    - Three to five isolated blue pixels appear inside the canvas.
    - The blue pixels are distinct sources, and any overlap happens only on the diagonal shadow tails they induce.

    Rule:
    - For each blue pixel at `(r, c)`, trace the diagonal shadow cells `(r-1, c+1)`, `(r-2, c+2)`, ... until the ray leaves the grid.
    - Count how many rays visit every traced cell.
    - Keep the original blue source pixels unchanged.
    - Recolor covered background cells with count `1` to red and count `>= 2` to green.
    - Leave all uncovered cells black.

    Key insight:
    - The output color depends on shadow depth rather than distance: cells on a shared anti-diagonal tail of multiple blue sources turn green.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    sources = [tuple(map(int, rc)) for rc in np.argwhere(g == BLUE)]
    counts = _shadow_counts(g.shape, sources)

    out = g.copy()
    background = g == BLACK
    out[background & (counts == 1)] = RED
    out[background & (counts >= 2)] = GREEN
    return out


def _is_isolated(candidate: tuple[int, int], points: list[tuple[int, int]]) -> bool:
    r, c = candidate
    for rr, cc in points:
        if max(abs(r - rr), abs(c - cc)) <= 1:
            return False
    return True


def _candidate_pool(h: int, w: int) -> list[tuple[int, int]]:
    return [(r, c) for r in range(2, h - 1) for c in range(1, w - 2)]


def _choose_mode(k: int) -> str:
    choices: list[tuple[str, float]] = []
    for mode, weight in MODE_WEIGHTS:
        if mode == "triple" and k < 4:
            continue
        if mode == "double_pair" and k < 4:
            continue
        choices.append((mode, weight))
    names = [name for name, _ in choices]
    weights = [weight for _, weight in choices]
    return random.choices(names, weights=weights, k=1)[0]


def _pick_same_sum_group(
    candidates: list[tuple[int, int]],
    existing: list[tuple[int, int]],
    used_sums: set[int],
    size: int,
) -> list[tuple[int, int]] | None:
    used_diffs = {r - c for r, c in existing}
    sums = list({r + c for r, c in candidates} - used_sums)
    random.shuffle(sums)

    for total in sums:
        pts = [p for p in candidates if p[0] + p[1] == total]
        random.shuffle(pts)
        chosen: list[tuple[int, int]] = []
        local_diffs = set(used_diffs)
        for p in pts:
            d = p[0] - p[1]
            if d in local_diffs:
                continue
            if not _is_isolated(p, existing + chosen):
                continue
            chosen.append(p)
            local_diffs.add(d)
            if len(chosen) == size:
                return chosen
    return None


def _pick_unique_sum_points(
    candidates: list[tuple[int, int]],
    existing: list[tuple[int, int]],
    used_sums: set[int],
    need: int,
) -> list[tuple[int, int]] | None:
    chosen: list[tuple[int, int]] = []
    used_diffs = {r - c for r, c in existing}
    local_sums = set(used_sums)
    pool = candidates.copy()
    random.shuffle(pool)

    for p in pool:
        total = p[0] + p[1]
        diff = p[0] - p[1]
        if total in local_sums or diff in used_diffs:
            continue
        if not _is_isolated(p, existing + chosen):
            continue
        chosen.append(p)
        local_sums.add(total)
        used_diffs.add(diff)
        if len(chosen) == need:
            return chosen
    return None


def _generate_sources(h: int, w: int, k: int) -> list[tuple[int, int]] | None:
    candidates = _candidate_pool(h, w)
    mode = _choose_mode(k)
    sources: list[tuple[int, int]] = []
    used_sums: set[int] = set()

    group_sizes: list[int]
    if mode == "none":
        group_sizes = []
    elif mode == "pair":
        group_sizes = [2]
    elif mode == "triple":
        group_sizes = [3]
    else:
        group_sizes = [2, 2]

    for size in group_sizes:
        group = _pick_same_sum_group(candidates, sources, used_sums, size)
        if group is None:
            return None
        sources.extend(group)
        used_sums.add(group[0][0] + group[0][1])

    extra = _pick_unique_sum_points(candidates, sources, used_sums, k - len(sources))
    if extra is None:
        return None
    sources.extend(extra)
    return sources


def _is_valid_input(grid: np.ndarray) -> bool:
    pts = [tuple(map(int, rc)) for rc in np.argwhere(grid == BLUE)]
    if not (3 <= len(pts) <= 5):
        return False

    seen_diffs: set[int] = set()
    for r, c in pts:
        if r <= 0 or c <= 0:
            return False
        diff = r - c
        if diff in seen_diffs:
            return False
        seen_diffs.add(diff)
        for rr, cc in pts:
            if (r, c) != (rr, cc) and max(abs(r - rr), abs(c - cc)) <= 1:
                return False
    return True


def _branch_signature(grid: np.ndarray) -> tuple[int, int]:
    out = main(grid)
    green = int(np.count_nonzero(out == GREEN))
    red = int(np.count_nonzero(out == RED))
    return green, red


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        h = random.randint(*GRID_H_RANGE)
        w = random.randint(*GRID_W_RANGE)
        k = random.randint(*SOURCE_COUNT_RANGE)

        sources = _generate_sources(h, w, k)
        if sources is None:
            continue

        grid = np.full((h, w), BLACK, dtype=int)
        for r, c in sources:
            grid[r, c] = BLUE

        if not _is_valid_input(grid):
            continue

        green_count, red_count = _branch_signature(grid)
        if red_count == 0:
            continue

        has_shared_sum = len({r + c for r, c in sources}) < len(sources)
        if has_shared_sum and green_count == 0:
            continue
        if (not has_shared_sum) and green_count != 0:
            continue

        return grid

    raise RuntimeError("failed to generate a valid input")
