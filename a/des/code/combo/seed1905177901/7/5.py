import random

import numpy as np


BACKGROUND = 5
COLOR_POOL = (0, 1, 2, 3, 4, 6, 7, 8, 9)

GRID_MIN = 10
GRID_MAX = 18
PAIR_COUNT_MIN = 3
PAIR_COUNT_MAX = 6
GEN_ATTEMPTS = 500
PAIR_ATTEMPTS = 250
MIN_STRAIGHT_SPAN = 2
MIN_ELBOW_SPAN = 2
MAX_FILL_RATIO = 0.38
OVERLAP_PROB = 0.35

_RNG = random.Random()


def _bg_color(grid: np.ndarray) -> int:
    values = np.asarray(grid, dtype=int).ravel()
    return int(np.bincount(values, minlength=10).argmax())


def _unique_cells(cells: list[tuple[int, int]]) -> list[tuple[int, int]]:
    return list(dict.fromkeys(cells))


def _path_cells(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    (r0, c0), (r1, c1) = a, b
    if r0 == r1:
        step = 1 if c1 >= c0 else -1
        return [(r0, c) for c in range(c0, c1 + step, step)]
    if c0 == c1:
        step = 1 if r1 >= r0 else -1
        return [(r, c0) for r in range(r0, r1 + step, step)]

    if c0 < c1:
        left = (r0, c0)
        right = (r1, c1)
    else:
        left = (r1, c1)
        right = (r0, c0)

    cells = [(left[0], c) for c in range(left[1], right[1] + 1)]
    step = 1 if right[0] >= left[0] else -1
    cells.extend((r, right[1]) for r in range(left[0], right[0] + step, step))
    return _unique_cells(cells)


def _pair_specs(grid: np.ndarray) -> tuple[int, list[dict[str, object]]]:
    g = np.asarray(grid, dtype=int)
    bg = _bg_color(g)
    specs: list[dict[str, object]] = []

    for color in (int(c) for c in np.unique(g) if int(c) != bg):
        points = [tuple(map(int, pt)) for pt in np.argwhere(g == color)]
        assert len(points) == 2, "each foreground color must appear exactly twice"
        cells = _path_cells(points[0], points[1])
        rows = [points[0][0], points[1][0]]
        cols = [points[0][1], points[1][1]]
        specs.append(
            {
                "color": color,
                "points": points,
                "cells": cells,
                "length": len(cells),
                "rmin": min(rows),
                "cmin": min(cols),
                "rmax": max(rows),
                "cmax": max(cols),
            }
        )

    specs.sort(
        key=lambda spec: (
            -int(spec["length"]),
            int(spec["rmin"]),
            int(spec["cmin"]),
            int(spec["rmax"]),
            int(spec["cmax"]),
        )
    )
    return bg, specs


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect matching color pairs with a straight segment or an L-shaped elbow.

    Pairs are processed by decreasing path size. Later paths overwrite earlier
    ones, so overlaps are resolved in favor of the smaller path.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    _bg, specs = _pair_specs(g)
    for spec in specs:
        color = int(spec["color"])
        for r, c in spec["cells"]:
            out[r, c] = color
    return out


def _balanced_branch_types(n_pairs: int) -> list[str]:
    counts = {"row": 1, "col": 1, "elbow": 1}
    branches = ["row", "col", "elbow"]
    while len(branches) < n_pairs:
        min_count = min(counts.values())
        choices = [branch for branch, count in counts.items() if count == min_count]
        branch = _RNG.choice(choices)
        counts[branch] += 1
        branches.append(branch)
    _RNG.shuffle(branches)
    elbows = [branch for branch in branches if branch == "elbow"]
    others = [branch for branch in branches if branch != "elbow"]
    _RNG.shuffle(elbows)
    _RNG.shuffle(others)
    return elbows + others


def _required_rows(branches: list[str]) -> int:
    return sum(1 if branch == "row" else 2 for branch in branches)


def _required_cols(branches: list[str]) -> int:
    return sum(1 if branch == "col" else 2 for branch in branches)


def _sample_endpoints(
    branch: str, height: int, width: int, used_rows: set[int], used_cols: set[int]
) -> tuple[tuple[int, int], tuple[int, int]] | None:
    rows = [r for r in range(1, height - 1) if r not in used_rows]
    cols = [c for c in range(1, width - 1) if c not in used_cols]

    if branch == "row":
        if len(rows) < 1 or len(cols) < 2:
            return None
        r = _RNG.choice(rows)
        c0, c1 = sorted(_RNG.sample(cols, 2))
        if c1 - c0 < MIN_STRAIGHT_SPAN:
            return None
        return (r, c0), (r, c1)

    if branch == "col":
        if len(rows) < 2 or len(cols) < 1:
            return None
        r0, r1 = sorted(_RNG.sample(rows, 2))
        if r1 - r0 < MIN_STRAIGHT_SPAN:
            return None
        c = _RNG.choice(cols)
        return (r0, c), (r1, c)

    if len(rows) < 2 or len(cols) < 2:
        return None
    r0, r1 = sorted(_RNG.sample(rows, 2))
    c0, c1 = sorted(_RNG.sample(cols, 2))
    if r1 - r0 < MIN_ELBOW_SPAN or c1 - c0 < MIN_ELBOW_SPAN:
        return None
    if _RNG.random() < 0.5:
        return (r0, c0), (r1, c1)
    return (r1, c0), (r0, c1)


def _overlap_cells(
    cells: list[tuple[int, int]], path_counts: dict[tuple[int, int], int]
) -> list[tuple[int, int]]:
    return [cell for cell in cells if cell in path_counts]


def _choose_pair(
    branch: str,
    height: int,
    width: int,
    used_rows: set[int],
    used_cols: set[int],
    used_lengths: set[int],
    path_counts: dict[tuple[int, int], int],
    overlap_mode: str,
) -> dict[str, object] | None:
    best_overlap = None
    best_no_overlap = None

    for _ in range(PAIR_ATTEMPTS):
        endpoints = _sample_endpoints(branch, height, width, used_rows, used_cols)
        if endpoints is None:
            continue

        cells = _path_cells(*endpoints)
        length = len(cells)
        if length in used_lengths:
            continue

        overlap = _overlap_cells(cells, path_counts)
        if overlap:
            if len(overlap) != 1 or path_counts[overlap[0]] != 1:
                continue
            candidate = {
                "branch": branch,
                "points": endpoints,
                "cells": cells,
                "length": length,
                "overlap_cell": overlap[0],
            }
            best_overlap = candidate
            if overlap_mode in {"force", "prefer"}:
                return candidate
            continue

        candidate = {
            "branch": branch,
            "points": endpoints,
            "cells": cells,
            "length": length,
            "overlap_cell": None,
        }
        best_no_overlap = candidate
        if overlap_mode == "forbid":
            return candidate

    if overlap_mode == "force":
        return best_overlap
    if overlap_mode == "prefer":
        return best_overlap if best_overlap is not None else best_no_overlap
    return best_no_overlap


def _gen_attempt() -> np.ndarray | None:
    height = _RNG.randint(GRID_MIN, GRID_MAX)
    width = _RNG.randint(GRID_MIN, GRID_MAX)

    branches = None
    for _ in range(30):
        pair_count = _RNG.randint(PAIR_COUNT_MIN, min(PAIR_COUNT_MAX, len(COLOR_POOL)))
        trial = _balanced_branch_types(pair_count)
        if _required_rows(trial) <= height - 2 and _required_cols(trial) <= width - 2:
            branches = trial
            break
    if branches is None:
        return None

    need_overlap = len(branches) >= 4 and _RNG.random() < OVERLAP_PROB
    used_rows: set[int] = set()
    used_cols: set[int] = set()
    used_lengths: set[int] = set()
    path_counts: dict[tuple[int, int], int] = {}
    pairs: list[dict[str, object]] = []
    overlap_used = False

    for idx, branch in enumerate(branches):
        if need_overlap and not overlap_used:
            if idx >= 2:
                overlap_mode = "force" if idx == len(branches) - 1 else "prefer"
            else:
                overlap_mode = "forbid"
        else:
            overlap_mode = "forbid"

        pair = _choose_pair(
            branch,
            height,
            width,
            used_rows,
            used_cols,
            used_lengths,
            path_counts,
            overlap_mode,
        )
        if pair is None:
            return None

        endpoints = pair["points"]
        rows = {int(endpoints[0][0]), int(endpoints[1][0])}
        cols = {int(endpoints[0][1]), int(endpoints[1][1])}
        used_rows |= rows
        used_cols |= cols
        used_lengths.add(int(pair["length"]))
        for cell in pair["cells"]:
            path_counts[cell] = path_counts.get(cell, 0) + 1
        overlap_used = overlap_used or pair["overlap_cell"] is not None
        pairs.append(pair)

    if need_overlap and not overlap_used:
        return None

    colors = _RNG.sample(COLOR_POOL, len(pairs))
    grid = np.full((height, width), BACKGROUND, dtype=int)
    for color, pair in zip(colors, pairs):
        for r, c in pair["points"]:
            grid[r, c] = color

    try:
        out = main(grid)
    except AssertionError:
        return None

    if np.array_equal(out, grid):
        return None
    if int(np.count_nonzero(out != BACKGROUND)) > int(height * width * MAX_FILL_RATIO):
        return None
    if len({int(c) for c in np.unique(grid) if int(c) != BACKGROUND}) != len(pairs):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input: failed after {GEN_ATTEMPTS} attempts")
