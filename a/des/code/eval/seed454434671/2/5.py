from __future__ import annotations

from itertools import combinations
import random

import numpy as np


BG = 0
WALL = 5
MARKER_COLORS = (1, 2, 3, 4)
DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]

GRID_MIN = 17
GRID_MAX = 27
EXTRA_OPENINGS_MIN = 1
EXTRA_OPENINGS_MAX = 7
GEN_ATTEMPTS = 500

SCENARIOS = (
    ("invisible_pair", "visible_pair", "line_triple"),
    ("visible_pair", "elbow_triple", "tee_quad"),
    ("invisible_pair", "elbow_triple", "tee_quad"),
    ("visible_pair", "line_triple", "tee_quad"),
    ("invisible_pair", "visible_pair", "line_triple", "tee_quad"),
)

PATTERN_PRIORITY = {
    "tee_quad": 0,
    "line_triple": 1,
    "elbow_triple": 2,
    "visible_pair": 3,
    "invisible_pair": 4,
}


def _in_bounds(h: int, w: int, r: int, c: int) -> bool:
    return 0 <= r < h and 0 <= c < w


def _neighbors8(r: int, c: int) -> list[tuple[int, int]]:
    out = []
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            out.append((r + dr, c + dc))
    return out


def _visible_same_count(grid: np.ndarray, r: int, c: int, color: int) -> int:
    h, w = grid.shape
    seen = 0
    for dr, dc in DIRS:
        nr, nc = r + dr, c + dc
        while _in_bounds(h, w, nr, nc) and int(grid[nr, nc]) != WALL:
            if int(grid[nr, nc]) == color:
                seen += 1
            nr += dr
            nc += dc
    return seen


def _available_fill_cells(grid: np.ndarray, r: int, c: int) -> list[tuple[int, int]]:
    h, w = grid.shape
    cells: list[tuple[int, int]] = []
    for dr, dc in DIRS:
        nr, nc = r + dr, c + dc
        if _in_bounds(h, w, nr, nc) and int(grid[nr, nc]) == BG:
            cells.append((nr, nc))
    return cells


def main(grid: np.ndarray) -> np.ndarray:
    """
    In a grey-walled maze, count how many same-color markers each marker can
    see along straight lines, then fill that many adjacent black cells using
    up/right/down/left priority.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    out = g.copy()

    claims: list[tuple[int, int, int]] = []
    for r in range(h):
        for c in range(w):
            color = int(g[r, c])
            if color in (BG, WALL):
                continue
            visible = _visible_same_count(g, r, c, color)
            for nr, nc in _available_fill_cells(g, r, c)[:visible]:
                claims.append((nr, nc, color))

    for r, c, color in claims:
        out[r, c] = color
    return out


def _random_odd(rng: random.Random, lo: int, hi: int) -> int:
    start = lo if lo % 2 == 1 else lo + 1
    stop = hi if hi % 2 == 1 else hi - 1
    return rng.randrange(start, stop + 1, 2)


def _carve_maze(h: int, w: int, rng: random.Random) -> np.ndarray:
    grid = np.full((h, w), WALL, dtype=int)
    start = (rng.randrange(1, h, 2), rng.randrange(1, w, 2))
    stack = [start]
    visited = {start}
    grid[start] = BG

    while stack:
        r, c = stack[-1]
        nxt: list[tuple[int, int, int, int]] = []
        for dr, dc in [(-2, 0), (0, 2), (2, 0), (0, -2)]:
            rr, cc = r + dr, c + dc
            if 1 <= rr < h - 1 and 1 <= cc < w - 1 and (rr, cc) not in visited:
                nxt.append((rr, cc, r + dr // 2, c + dc // 2))
        if not nxt:
            stack.pop()
            continue
        rr, cc, wr, wc = rng.choice(nxt)
        visited.add((rr, cc))
        grid[wr, wc] = BG
        grid[rr, cc] = BG
        stack.append((rr, cc))

    return grid


def _add_extra_openings(grid: np.ndarray, rng: random.Random) -> None:
    h, w = grid.shape
    candidates: list[tuple[int, int]] = []
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            if int(grid[r, c]) != WALL:
                continue
            if r % 2 == 0 and c % 2 == 1:
                if int(grid[r - 1, c]) == BG and int(grid[r + 1, c]) == BG:
                    candidates.append((r, c))
            elif r % 2 == 1 and c % 2 == 0:
                if int(grid[r, c - 1]) == BG and int(grid[r, c + 1]) == BG:
                    candidates.append((r, c))
    rng.shuffle(candidates)
    openings = rng.randint(EXTRA_OPENINGS_MIN, EXTRA_OPENINGS_MAX)
    for r, c in candidates[:openings]:
        grid[r, c] = BG


def _segment_data(grid: np.ndarray) -> dict[str, object]:
    h, w = grid.shape
    row_id = -np.ones((h, w), dtype=int)
    col_id = -np.ones((h, w), dtype=int)
    row_segments: list[list[tuple[int, int]]] = []
    col_segments: list[list[tuple[int, int]]] = []

    for r in range(h):
        c = 0
        while c < w:
            if int(grid[r, c]) != BG:
                c += 1
                continue
            seg: list[tuple[int, int]] = []
            sid = len(row_segments)
            while c < w and int(grid[r, c]) == BG:
                row_id[r, c] = sid
                seg.append((r, c))
                c += 1
            row_segments.append(seg)

    for c in range(w):
        r = 0
        while r < h:
            if int(grid[r, c]) != BG:
                r += 1
                continue
            seg: list[tuple[int, int]] = []
            sid = len(col_segments)
            while r < h and int(grid[r, c]) == BG:
                col_id[r, c] = sid
                seg.append((r, c))
                r += 1
            col_segments.append(seg)

    degree: dict[tuple[int, int], int] = {}
    open_cells: list[tuple[int, int]] = []
    for r in range(h):
        for c in range(w):
            if int(grid[r, c]) != BG:
                continue
            open_cells.append((r, c))
            degree[(r, c)] = sum(
                1
                for dr, dc in DIRS
                if _in_bounds(h, w, r + dr, c + dc) and int(grid[r + dr, c + dc]) == BG
            )

    return {
        "row_id": row_id,
        "col_id": col_id,
        "segments": row_segments + col_segments,
        "degree": degree,
        "open_cells": open_cells,
    }


def _arm_cells(grid: np.ndarray, r: int, c: int, dr: int, dc: int) -> list[tuple[int, int]]:
    h, w = grid.shape
    arm: list[tuple[int, int]] = []
    nr, nc = r + dr, c + dc
    while _in_bounds(h, w, nr, nc) and int(grid[nr, nc]) == BG:
        arm.append((nr, nc))
        nr += dr
        nc += dc
    return arm


def _mark_forbidden(forbidden: set[tuple[int, int]], cells: list[tuple[int, int]]) -> None:
    for r, c in cells:
        forbidden.add((r, c))
        for nr, nc in _neighbors8(r, c):
            forbidden.add((nr, nc))


def _sample_line_group(
    segments: list[list[tuple[int, int]]],
    degree: dict[tuple[int, int], int],
    forbidden: set[tuple[int, int]],
    count: int,
    min_degree: int,
    rng: random.Random,
) -> list[tuple[int, int]] | None:
    order = list(range(len(segments)))
    rng.shuffle(order)
    for sid in order:
        seg = segments[sid]
        usable = [
            (idx, cell)
            for idx, cell in enumerate(seg)
            if cell not in forbidden and degree[cell] >= min_degree
        ]
        if len(usable) < count:
            continue
        combos = [
            [cell for _, cell in combo]
            for combo in combinations(usable, count)
            if all(combo[i + 1][0] - combo[i][0] >= 2 for i in range(count - 1))
        ]
        if combos:
            return rng.choice(combos)
    return None


def _sample_invisible_pair(
    open_cells: list[tuple[int, int]],
    row_id: np.ndarray,
    col_id: np.ndarray,
    forbidden: set[tuple[int, int]],
    rng: random.Random,
) -> list[tuple[int, int]] | None:
    usable = [cell for cell in open_cells if cell not in forbidden]
    if len(usable) < 2:
        return None
    for _ in range(300):
        a, b = rng.sample(usable, 2)
        if row_id[a] == row_id[b] or col_id[a] == col_id[b]:
            continue
        if max(abs(a[0] - b[0]), abs(a[1] - b[1])) <= 1:
            continue
        return [a, b]
    rng.shuffle(usable)
    for i, a in enumerate(usable):
        for b in usable[i + 1 :]:
            if row_id[a] == row_id[b] or col_id[a] == col_id[b]:
                continue
            if max(abs(a[0] - b[0]), abs(a[1] - b[1])) <= 1:
                continue
            return [a, b]
    return None


def _sample_elbow_triple(
    grid: np.ndarray,
    open_cells: list[tuple[int, int]],
    degree: dict[tuple[int, int], int],
    forbidden: set[tuple[int, int]],
    rng: random.Random,
) -> list[tuple[int, int]] | None:
    centers = [cell for cell in open_cells if cell not in forbidden and degree[cell] >= 2]
    rng.shuffle(centers)
    for r, c in centers:
        arms = [_arm_cells(grid, r, c, dr, dc) for dr, dc in DIRS]
        v_dirs = [d for d in (0, 2) if len(arms[d]) >= 2]
        h_dirs = [d for d in (1, 3) if len(arms[d]) >= 2]
        if not v_dirs or not h_dirs:
            continue
        rng.shuffle(v_dirs)
        rng.shuffle(h_dirs)
        for vd in v_dirs:
            v_opts = [cell for cell in arms[vd][1:] if cell not in forbidden and degree[cell] >= 1]
            if not v_opts:
                continue
            for hd in h_dirs:
                h_opts = [cell for cell in arms[hd][1:] if cell not in forbidden and degree[cell] >= 1]
                if not h_opts:
                    continue
                return [(r, c), rng.choice(v_opts), rng.choice(h_opts)]
    return None


def _sample_tee_quad(
    grid: np.ndarray,
    open_cells: list[tuple[int, int]],
    degree: dict[tuple[int, int], int],
    forbidden: set[tuple[int, int]],
    rng: random.Random,
) -> list[tuple[int, int]] | None:
    templates = [
        ((0, 2, 1), (2, 2, 1)),
        ((0, 2, 3), (2, 2, 1)),
        ((1, 3, 0), (2, 2, 1)),
        ((1, 3, 2), (2, 2, 1)),
    ]
    centers = [cell for cell in open_cells if cell not in forbidden and degree[cell] >= 3]
    rng.shuffle(centers)
    for r, c in centers:
        arms = [_arm_cells(grid, r, c, dr, dc) for dr, dc in DIRS]
        local_templates = templates[:]
        rng.shuffle(local_templates)
        for dirs, needs in local_templates:
            chosen: list[tuple[int, int]] = []
            ok = True
            for d, need in zip(dirs, needs):
                if len(arms[d]) < 2:
                    ok = False
                    break
                opts = [cell for cell in arms[d][1:] if cell not in forbidden and degree[cell] >= need]
                if not opts:
                    ok = False
                    break
                chosen.append(rng.choice(opts))
            if ok:
                return [(r, c)] + chosen
    return None


def _pattern_cells(
    pattern: str,
    grid: np.ndarray,
    data: dict[str, object],
    forbidden: set[tuple[int, int]],
    rng: random.Random,
) -> tuple[list[tuple[int, int]], list[int]] | None:
    segments = data["segments"]
    degree = data["degree"]
    open_cells = data["open_cells"]
    row_id = data["row_id"]
    col_id = data["col_id"]

    if pattern == "visible_pair":
        cells = _sample_line_group(segments, degree, forbidden, count=2, min_degree=1, rng=rng)
        if cells is None:
            return None
        return cells, [1, 1]
    if pattern == "line_triple":
        cells = _sample_line_group(segments, degree, forbidden, count=3, min_degree=2, rng=rng)
        if cells is None:
            return None
        return cells, [2, 2, 2]
    if pattern == "invisible_pair":
        cells = _sample_invisible_pair(open_cells, row_id, col_id, forbidden, rng)
        if cells is None:
            return None
        return cells, [0, 0]
    if pattern == "elbow_triple":
        cells = _sample_elbow_triple(grid, open_cells, degree, forbidden, rng)
        if cells is None:
            return None
        return cells, [2, 1, 1]
    if pattern == "tee_quad":
        cells = _sample_tee_quad(grid, open_cells, degree, forbidden, rng)
        if cells is None:
            return None
        return cells, [3, 2, 2, 1]
    raise ValueError(f"unknown pattern: {pattern}")


def _validate_layout(
    grid: np.ndarray,
    placements: list[tuple[int, str, list[tuple[int, int]], list[int]]],
) -> bool:
    claims: dict[tuple[int, int], int] = {}
    for color, _pattern, cells, expected_counts in placements:
        for (r, c), expected in zip(cells, expected_counts):
            actual = _visible_same_count(grid, r, c, color)
            if actual != expected:
                return False
            targets = _available_fill_cells(grid, r, c)
            if len(targets) < actual:
                return False
            for cell in targets[:actual]:
                if cell in claims:
                    return False
                claims[cell] = color
    return bool(claims)


def _build_candidate(rng: random.Random) -> np.ndarray | None:
    h = _random_odd(rng, GRID_MIN, GRID_MAX)
    w = _random_odd(rng, GRID_MIN, GRID_MAX)
    base = _carve_maze(h, w, rng)
    _add_extra_openings(base, rng)
    data = _segment_data(base)

    scenario = list(rng.choice(SCENARIOS))
    colors = rng.sample(list(MARKER_COLORS), len(scenario))
    jobs = list(zip(colors, scenario))
    jobs.sort(key=lambda item: PATTERN_PRIORITY[item[1]])

    placements: list[tuple[int, str, list[tuple[int, int]], list[int]]] = []
    forbidden: set[tuple[int, int]] = set()
    for color, pattern in jobs:
        picked = _pattern_cells(pattern, base, data, forbidden, rng)
        if picked is None:
            return None
        cells, expected = picked
        placements.append((color, pattern, cells, expected))
        _mark_forbidden(forbidden, cells)

    grid = base.copy()
    for color, _pattern, cells, _expected in placements:
        for r, c in cells:
            grid[r, c] = color

    if not _validate_layout(grid, placements):
        return None

    # Reject overly uniform examples so count branches stay diverse.
    counts = {count for _, _, _, expected in placements for count in expected}
    if len(counts) < 3:
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _build_candidate(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid maze-marker input")
