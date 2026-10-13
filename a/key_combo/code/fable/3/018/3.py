from __future__ import annotations

from collections import Counter
import random

import numpy as np


GRAY = 5
BACKGROUND = 0
PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]
DIAGONAL_DIRS = [(-1, -1), (-1, 1), (1, -1), (1, 1)]


def _sign(x: int) -> int:
    return (int(x) > 0) - (int(x) < 0)


def _get_bg(grid: np.ndarray) -> int:
    counts = Counter(int(v) for v in grid.ravel())
    candidates = [c for c in counts if c != GRAY]
    return int(max(candidates, key=lambda c: counts[c]))


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components(mask: np.ndarray, connectivity: int = 4) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros(mask.shape, dtype=bool)
    if connectivity == 4:
        dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    else:
        dirs = [
            (dr, dc)
            for dr in (-1, 0, 1)
            for dc in (-1, 0, 1)
            if not (dr == 0 and dc == 0)
        ]
    comps: list[np.ndarray] = []
    for sr, sc in np.argwhere(mask):
        sr, sc = int(sr), int(sc)
        if seen[sr, sc]:
            continue
        seen[sr, sc] = True
        stack = [(sr, sc)]
        coords: list[tuple[int, int]] = []
        while stack:
            r, c = stack.pop()
            coords.append((r, c))
            for dr, dc in dirs:
                rr, cc = r + dr, c + dc
                if 0 <= rr < h and 0 <= cc < w and bool(mask[rr, cc]) and not seen[rr, cc]:
                    seen[rr, cc] = True
                    stack.append((rr, cc))
        comps.append(np.array(coords, dtype=int))
    return comps


def _rect_border_cells(r0: int, c0: int, r1: int, c1: int) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    for c in range(int(c0), int(c1) + 1):
        cells.add((int(r0), int(c)))
        cells.add((int(r1), int(c)))
    for r in range(int(r0) + 1, int(r1)):
        cells.add((int(r), int(c0)))
        cells.add((int(r), int(c1)))
    return cells


def _rect_all_cells(r0: int, c0: int, r1: int, c1: int) -> list[tuple[int, int]]:
    return [(r, c) for r in range(int(r0), int(r1) + 1) for c in range(int(c0), int(c1) + 1)]


def _find_container(grid: np.ndarray, color: int, bg: int) -> dict:
    containers: list[dict] = []
    for comp in _components(grid == int(color), connectivity=4):
        if comp.size == 0:
            continue
        r0, c0, r1, c1 = _bbox(comp)
        if r1 - r0 < 2 or c1 - c0 < 2:
            continue
        border = _rect_border_cells(r0, c0, r1, c1)
        comp_set = {(int(r), int(c)) for r, c in comp}
        if comp_set != border:
            continue
        interior = grid[r0 + 1 : r1, c0 + 1 : c1]
        if interior.size == 0:
            continue
        if not bool(np.logical_or(interior == int(bg), interior == GRAY).all()):
            continue
        containers.append({"box": (r0, c0, r1, c1), "border": border})
    assert len(containers) == 1, "each endpoint color must have one bordered container"
    return containers[0]


def _chessboard_distance_to_border(pt: tuple[int, int], border: set[tuple[int, int]]) -> int:
    r, c = pt
    return min(max(abs(int(r) - br), abs(int(c) - bc)) for br, bc in border)


def _fuel_cells_row_major(grid: np.ndarray, box: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = box
    return [
        (r, c)
        for r in range(int(r0) + 1, int(r1))
        for c in range(int(c0) + 1, int(c1))
        if int(grid[r, c]) == GRAY
    ]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Draw fuel-limited diagonal links from the endpoint nearest its same-colored container.

    Input structure:
    - A dominant background contains several distinct endpoint colors plus gray fuel cells.
    - For each endpoint color, exactly two endpoint pixels lie on one strict 45-degree diagonal.
    - The same color also forms one rectangular border whose interior contains only gray fuel and background.
    - The two endpoints are at different chessboard distances from that color's container border.

    Rule:
    - For each color, let k be the number of background cells strictly between its diagonal endpoints.
    - Count the gray cells inside the same-colored container as fuel f.
    - Starting next to the endpoint closer to the container, paint min(k, f) diagonal gap cells toward the other endpoint.
    - Repaint the first min(k, f) gray cells in that container, scanning top-to-bottom then left-to-right, with the endpoint color.

    Key insight:
    - The gray container cells are expendable ink: their count caps the line length, and the nearer endpoint fixes which side of the diagonal is drawn first.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = _get_bg(g)
    assert bg != GRAY, "gray is reserved for fuel, not background"

    out = g.copy()
    colors = [int(c) for c in np.unique(g) if int(c) not in (int(bg), GRAY)]
    assert colors, "expected at least one endpoint color"

    for color in colors:
        container = _find_container(g, color, bg)
        border = container["border"]
        endpoints = [
            (int(r), int(c))
            for r, c in np.argwhere(g == int(color))
            if (int(r), int(c)) not in border
        ]
        assert len(endpoints) == 2, "each color must have exactly two non-container endpoints"
        p0, p1 = endpoints

        dr = p1[0] - p0[0]
        dc = p1[1] - p0[1]
        assert dr != 0 and abs(dr) == abs(dc), "endpoints must define a strict diagonal"
        k = abs(dr) - 1
        assert k >= 1, "endpoint gap must contain at least one background cell"

        d0 = _chessboard_distance_to_border(p0, border)
        d1 = _chessboard_distance_to_border(p1, border)
        assert d0 != d1, "one endpoint must be strictly nearer to the container"
        near, far = (p0, p1) if d0 < d1 else (p1, p0)
        step_r = _sign(far[0] - near[0])
        step_c = _sign(far[1] - near[1])

        fuel = _fuel_cells_row_major(g, container["box"])
        used = min(k, len(fuel))
        for i in range(1, used + 1):
            rr = near[0] + i * step_r
            cc = near[1] + i * step_c
            assert int(g[rr, cc]) == int(bg), "diagonal gap must be background in the input"
            assert int(out[rr, cc]) == int(bg), "diagonal paths must not cross"
            out[rr, cc] = int(color)
        for r, c in fuel[:used]:
            assert int(out[r, c]) == GRAY, "fuel cells must be unused gray pixels"
            out[r, c] = int(color)

    return out


def _has_reserved_near(cells: list[tuple[int, int]], reserved: np.ndarray, padding: int = 1) -> bool:
    h, w = reserved.shape
    for r, c in cells:
        for rr in range(max(0, r - padding), min(h, r + padding + 1)):
            for cc in range(max(0, c - padding), min(w, c + padding + 1)):
                if bool(reserved[rr, cc]):
                    return True
    return False


def _mark(cells: list[tuple[int, int]], reserved: np.ndarray) -> None:
    for r, c in cells:
        reserved[int(r), int(c)] = True


def _dims_for_branch(branch: str, gap: int) -> list[tuple[int, int]]:
    dims: list[tuple[int, int]] = []
    for h in range(3, 8):
        for w in range(3, 8):
            capacity = (h - 2) * (w - 2)
            if branch == "surplus" and capacity >= gap + 1:
                dims.append((h, w))
            elif branch == "exact" and capacity >= gap:
                dims.append((h, w))
            elif branch in ("limited", "zero") and capacity >= 1:
                dims.append((h, w))
    random.shuffle(dims)
    return dims


def _fuel_count(branch: str, gap: int, capacity: int) -> int:
    if branch == "zero":
        return 0
    if branch == "limited":
        return random.randint(1, min(gap - 1, capacity))
    if branch == "exact":
        return int(gap)
    return random.randint(gap + 1, capacity)


def _choose_gray_cells(
    r0: int, c0: int, r1: int, c1: int, count: int
) -> list[tuple[int, int]]:
    interior = [(r, c) for r in range(r0 + 1, r1) for c in range(c0 + 1, c1)]
    if count == 0:
        return []
    return random.sample(interior, int(count))


def _candidate_line(
    h: int, w: int, gap: int, reserved: np.ndarray
) -> tuple[list[tuple[int, int]], tuple[int, int], tuple[int, int]] | None:
    steps = int(gap) + 1
    for _ in range(500):
        dr, dc = random.choice(DIAGONAL_DIRS)
        r0 = random.randint(1, h - 2)
        c0 = random.randint(1, w - 2)
        r1 = r0 + dr * steps
        c1 = c0 + dc * steps
        if not (1 <= r1 < h - 1 and 1 <= c1 < w - 1):
            continue
        cells = [(r0 + i * dr, c0 + i * dc) for i in range(steps + 1)]
        if _has_reserved_near(cells, reserved, padding=1):
            continue
        return cells, cells[0], cells[-1]
    return None


def _place_one(
    grid: np.ndarray,
    reserved: np.ndarray,
    color: int,
    branch: str,
    prefer_first_near: bool,
) -> dict | None:
    h, w = grid.shape
    gap_values = list(range(2, 8))
    random.shuffle(gap_values)

    for gap in gap_values:
        line = _candidate_line(h, w, gap, reserved)
        if line is None:
            continue
        line_cells, p0, p1 = line
        local_reserved = reserved.copy()
        _mark(line_cells, local_reserved)

        for box_h, box_w in _dims_for_branch(branch, gap):
            capacity = (box_h - 2) * (box_w - 2)
            fuel_n = _fuel_count(branch, gap, capacity)
            positions = [
                (r, c)
                for r in range(1, h - box_h)
                for c in range(1, w - box_w)
            ]
            random.shuffle(positions)
            for r0, c0 in positions[:220]:
                r1, c1 = r0 + box_h - 1, c0 + box_w - 1
                rect_cells = _rect_all_cells(r0, c0, r1, c1)
                if _has_reserved_near(rect_cells, local_reserved, padding=1):
                    continue
                border = _rect_border_cells(r0, c0, r1, c1)
                d0 = _chessboard_distance_to_border(p0, border)
                d1 = _chessboard_distance_to_border(p1, border)
                if d0 == d1 or abs(d0 - d1) < 2 or min(d0, d1) > 8:
                    continue
                if (d0 < d1) != bool(prefer_first_near):
                    continue

                gray_cells = _choose_gray_cells(r0, c0, r1, c1, fuel_n)
                for r, c in border:
                    grid[r, c] = int(color)
                for r, c in gray_cells:
                    grid[r, c] = GRAY
                grid[p0[0], p0[1]] = int(color)
                grid[p1[0], p1[1]] = int(color)

                _mark(line_cells, reserved)
                _mark(rect_cells, reserved)
                return {
                    "color": int(color),
                    "p0": p0,
                    "p1": p1,
                    "gap": int(gap),
                    "border": border,
                    "box": (r0, c0, r1, c1),
                    "gray_cells": sorted(gray_cells),
                    "branch": branch,
                }
    return None


def _expected_from_specs(inp: np.ndarray, specs: list[dict]) -> np.ndarray:
    out = inp.copy()
    for spec in specs:
        color = int(spec["color"])
        p0 = spec["p0"]
        p1 = spec["p1"]
        d0 = _chessboard_distance_to_border(p0, spec["border"])
        d1 = _chessboard_distance_to_border(p1, spec["border"])
        near, far = (p0, p1) if d0 < d1 else (p1, p0)
        step_r = _sign(far[0] - near[0])
        step_c = _sign(far[1] - near[1])
        used = min(int(spec["gap"]), len(spec["gray_cells"]))
        for i in range(1, used + 1):
            out[near[0] + i * step_r, near[1] + i * step_c] = color
        for r, c in spec["gray_cells"][:used]:
            out[r, c] = color
    return out


def _branch_plan(n: int) -> list[str]:
    branches = ["limited", "surplus"]
    if n >= 3:
        branches.append(random.choice(["zero", "exact"]))
    while len(branches) < n:
        branches.append(random.choice(["limited", "surplus", "exact", "zero"]))
    random.shuffle(branches)
    return branches


def _generate_attempt() -> np.ndarray | None:
    h = random.randint(20, 30)
    w = random.randint(20, 30)
    n_objects = random.randint(2, 4)
    colors = random.sample(PALETTE, n_objects)
    branches = _branch_plan(n_objects)
    near_preferences = [True, False]
    while len(near_preferences) < n_objects:
        near_preferences.append(bool(random.getrandbits(1)))
    random.shuffle(near_preferences)

    grid = np.full((h, w), BACKGROUND, dtype=int)
    reserved = np.zeros((h, w), dtype=bool)
    specs: list[dict] = []
    for color, branch, near_pref in zip(colors, branches, near_preferences):
        spec = _place_one(grid, reserved, color, branch, near_pref)
        if spec is None:
            return None
        specs.append(spec)

    expected = _expected_from_specs(grid, specs)
    try:
        solved = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(solved, expected):
        return None
    if not {"limited", "surplus"}.issubset({spec["branch"] for spec in specs}):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(600):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid fuel-limited diagonal-link input")
