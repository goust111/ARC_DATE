from __future__ import annotations

import random

import numpy as np


GRID_SIZE = 16
BG = 0
TL_COLOR = 2
TR_COLOR = 1
BL_COLOR = 3
BR_COLOR = 4
WALL = 5
MARKER = 6

EDGES = ("top", "bottom", "left", "right")
MAX_GENERATION_ATTEMPTS = 400


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _neighbors8(r: int, c: int):
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            yield r + dr, c + dc


def _edge_direction(pos: tuple[int, int], shape: tuple[int, int]) -> tuple[int, int]:
    r, c = pos
    h, w = shape
    options = []
    if r == 0:
        options.append((1, 0))
    if r == h - 1:
        options.append((-1, 0))
    if c == 0:
        options.append((0, 1))
    if c == w - 1:
        options.append((0, -1))
    assert len(options) == 1, "marker must lie on exactly one edge"
    return options[0]


def _quadrant_color(grid: np.ndarray, r: int, c: int) -> int:
    h, w = grid.shape
    top = r < h // 2
    left = c < w // 2
    if top and left:
        return int(grid[0, 0])
    if top and not left:
        return int(grid[0, w - 1])
    if not top and left:
        return int(grid[h - 1, 0])
    return int(grid[h - 1, w - 1])


def _wall_segment_bounds(grid: np.ndarray, hit_r: int, hit_c: int, dr: int, dc: int) -> tuple[int, int]:
    h, w = grid.shape
    if dr != 0:
        lo = hi = hit_c
        while lo - 1 >= 0 and int(grid[hit_r, lo - 1]) == WALL:
            lo -= 1
        while hi + 1 < w and int(grid[hit_r, hi + 1]) == WALL:
            hi += 1
        return lo, hi

    lo = hi = hit_r
    while lo - 1 >= 0 and int(grid[lo - 1, hit_c]) == WALL:
        lo -= 1
    while hi + 1 < h and int(grid[hi + 1, hit_c]) == WALL:
        hi += 1
    return lo, hi


def _trace(grid: np.ndarray) -> tuple[np.ndarray, dict]:
    g = np.asarray(grid, dtype=int)
    marker_cells = np.argwhere(g == MARKER)
    assert len(marker_cells) == 1, "expected exactly one marker"
    start = tuple(map(int, marker_cells[0]))
    dr, dc = _edge_direction(start, g.shape)

    out = g.copy()
    stack = [start]
    seen_starts: set[tuple[int, int]] = set()
    painted: list[tuple[int, int]] = []
    hit_segments: list[tuple[int, int, int]] = []
    branch_starts: list[tuple[int, int]] = []

    while stack:
        sr, sc = stack.pop()
        if (sr, sc) in seen_starts:
            continue
        seen_starts.add((sr, sc))

        cr, cc = sr, sc
        while True:
            nr, nc = cr + dr, cc + dc
            if not _in_bounds(g.shape, nr, nc):
                break
            nxt = int(g[nr, nc])
            if nxt == BG:
                out[nr, nc] = _quadrant_color(g, nr, nc)
                painted.append((nr, nc))
                cr, cc = nr, nc
                continue
            if nxt != WALL:
                break

            lo, hi = _wall_segment_bounds(g, nr, nc, dr, dc)
            if dr != 0:
                hit_segments.append((nr, lo, hi))
                endpoints = ((nr, lo), (nr, hi))
            else:
                hit_segments.append((nc, lo, hi))
                endpoints = ((lo, nc), (hi, nc))

            for er, ec in endpoints:
                br, bc = er + dr, ec + dc
                if _in_bounds(g.shape, br, bc) and int(g[br, bc]) == BG:
                    branch_starts.append((br, bc))
                    stack.append((br, bc))
            break

    quadrants = {_quadrant_color(g, r, c) for r, c in painted}
    stats = {
        "painted": painted,
        "painted_count": len(painted),
        "hit_count": len(hit_segments),
        "hit_segments": hit_segments,
        "branch_starts": branch_starts,
        "seen_start_count": len(seen_starts),
        "quadrant_colors": quadrants,
    }
    return out, stats


def _components(mask: np.ndarray) -> tuple[list[list[tuple[int, int]]], np.ndarray]:
    h, w = mask.shape
    comp_id = -np.ones((h, w), dtype=int)
    comps: list[list[tuple[int, int]]] = []
    next_id = 0
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or comp_id[r, c] != -1:
                continue
            stack = [(r, c)]
            comp_id[r, c] = next_id
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and comp_id[nr, nc] == -1:
                        comp_id[nr, nc] = next_id
                        stack.append((nr, nc))
            comps.append(coords)
            next_id += 1
    return comps, comp_id


def _validate_wall_components(grid: np.ndarray) -> int:
    comps, comp_id = _components(grid == WALL)
    assert 4 <= len(comps) <= 6, "expected 4-6 wall segments"

    for coords in comps:
        rows = {r for r, _ in coords}
        cols = {c for _, c in coords}
        assert len(coords) in range(2, 6), "wall length must be 2-5"
        assert len(rows) == 1 or len(cols) == 1, "wall must be straight"
        if len(rows) == 1:
            row = next(iter(rows))
            span = sorted(c for _, c in coords)
            assert span[-1] - span[0] + 1 == len(coords), "horizontal wall must be contiguous"
            assert 0 < span[0] and span[-1] < grid.shape[1] - 1, "wall endpoints must have side openings"
            assert 0 < row < grid.shape[0] - 1, "wall must stay off the border"
        else:
            col = next(iter(cols))
            span = sorted(r for r, _ in coords)
            assert span[-1] - span[0] + 1 == len(coords), "vertical wall must be contiguous"
            assert 0 < span[0] and span[-1] < grid.shape[0] - 1, "wall endpoints must have side openings"
            assert 0 < col < grid.shape[1] - 1, "wall must stay off the border"

    for r in range(grid.shape[0]):
        for c in range(grid.shape[1]):
            if comp_id[r, c] == -1:
                continue
            for nr, nc in _neighbors8(r, c):
                if _in_bounds(grid.shape, nr, nc):
                    other = comp_id[nr, nc]
                    assert other in (-1, comp_id[r, c]), "wall segments must not touch"
    return len(comps)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Trace the edge marker inward, branch at wall-segment openings, and recolor every traversed background cell by its quadrant legend color.

    Input structure:
    - The grid is an even-sized black background with four fixed corner legend pixels.
    - A single magenta marker lies on one non-corner edge cell and sets the inward ray direction.
    - Several disjoint grey wall segments lie inside the grid, each as a short straight horizontal or vertical run.

    Rule:
    - Split the grid into four equal quadrants and use the four corner pixels as the quadrant colors.
    - From the marker, move one step at a time inward; every black cell visited by a ray is recolored with the legend color of that cell's quadrant.
    - When the next cell is a grey wall, stop that ray, recover the contiguous wall segment in the direction perpendicular to travel, and start new same-direction rays from the in-bounds black cells immediately beyond the wall's two endpoints.
    - Keep processing every branch until it would leave the grid or its next cell is not black.

    Key insight:
    - The branching geometry comes from the wall openings, but the color of each painted cell comes only from that cell's absolute quadrant, so a single ray can change color as it crosses the grid midpoint.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = g.shape
    assert h % 2 == 0 and w % 2 == 0, "grid must split evenly into quadrants"
    assert int(g[0, 0]) == TL_COLOR
    assert int(g[0, w - 1]) == TR_COLOR
    assert int(g[h - 1, 0]) == BL_COLOR
    assert int(g[h - 1, w - 1]) == BR_COLOR

    out, _stats = _trace(g)
    return out


def _map_coord(edge: str, r: int, c: int) -> tuple[int, int]:
    last = GRID_SIZE - 1
    if edge == "top":
        return r, c
    if edge == "bottom":
        return last - r, c
    if edge == "left":
        return c, r
    assert edge == "right"
    return c, last - r


def _interval_separated(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return a[1] + 2 <= b[0] or b[1] + 2 <= a[0]


def _interval_candidates(parent: int, blocked: set[int]) -> list[tuple[int, int]]:
    candidates = []
    for lo in range(1, GRID_SIZE - 1):
        for hi in range(lo + 1, min(GRID_SIZE - 2, lo + 4) + 1):
            if not (lo <= parent <= hi):
                continue
            if any(lo <= b <= hi for b in blocked):
                continue
            candidates.append((lo, hi))
    random.shuffle(candidates)
    return candidates


def _choose_intervals_for_row(parents: list[int], selected: list[int]) -> dict[int, tuple[int, int]] | None:
    ordered = sorted(selected)
    blocked_map = {p: set(parents) - {p} for p in ordered}
    cand_map = {p: _interval_candidates(p, blocked_map[p]) for p in ordered}
    placed: dict[int, tuple[int, int]] = {}

    def backtrack(idx: int) -> bool:
        if idx == len(ordered):
            return True
        parent = ordered[idx]
        for interval in cand_map[parent]:
            if all(_interval_separated(interval, other) for other in placed.values()):
                placed[parent] = interval
                if backtrack(idx + 1):
                    return True
                del placed[parent]
        return False

    return placed if backtrack(0) else None


def _build_canonical_plan(total_segments: int) -> dict | None:
    level3_count = total_segments - 3
    for _ in range(200):
        start_col = random.randint(2, GRID_SIZE - 3)
        row1 = random.randint(2, 3)
        row2 = random.randint(row1 + 3, 7)
        row3 = random.randint(max(row2 + 3, 9), 12)

        root_candidates = _interval_candidates(start_col, set())
        if not root_candidates:
            continue
        root = random.choice(root_candidates)
        level2_parents = [root[0], root[1]]
        level2 = _choose_intervals_for_row(level2_parents, level2_parents)
        if level2 is None:
            continue

        level2_children = []
        for parent in sorted(level2):
            lo, hi = level2[parent]
            level2_children.extend([lo, hi])
        if len(set(level2_children)) != 4:
            continue
        level2_children = sorted(level2_children)

        selected_level3 = random.sample(level2_children, level3_count)
        level3 = _choose_intervals_for_row(level2_children, selected_level3)
        if level3 is None:
            continue

        return {
            "start_col": start_col,
            "rows": (row1, row2, row3),
            "root": root,
            "level2": {p: level2[p] for p in sorted(level2)},
            "level3": {p: level3[p] for p in sorted(level3)},
        }
    return None


def _render_grid(edge: str, plan: dict) -> np.ndarray:
    grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
    grid[0, 0] = TL_COLOR
    grid[0, GRID_SIZE - 1] = TR_COLOR
    grid[GRID_SIZE - 1, 0] = BL_COLOR
    grid[GRID_SIZE - 1, GRID_SIZE - 1] = BR_COLOR

    mr, mc = _map_coord(edge, 0, plan["start_col"])
    grid[mr, mc] = MARKER

    row1, row2, row3 = plan["rows"]
    segments = [(row1, plan["root"])]
    segments.extend((row2, interval) for interval in plan["level2"].values())
    segments.extend((row3, interval) for interval in plan["level3"].values())

    for row, (lo, hi) in segments:
        for c in range(lo, hi + 1):
            rr, cc = _map_coord(edge, row, c)
            grid[rr, cc] = WALL
    return grid


def _is_valid_input(grid: np.ndarray) -> bool:
    try:
        assert grid.shape == (GRID_SIZE, GRID_SIZE)
        assert int(grid[0, 0]) == TL_COLOR
        assert int(grid[0, GRID_SIZE - 1]) == TR_COLOR
        assert int(grid[GRID_SIZE - 1, 0]) == BL_COLOR
        assert int(grid[GRID_SIZE - 1, GRID_SIZE - 1]) == BR_COLOR

        marker_cells = np.argwhere(grid == MARKER)
        assert len(marker_cells) == 1
        mr, mc = map(int, marker_cells[0])
        assert (mr in {0, GRID_SIZE - 1}) ^ (mc in {0, GRID_SIZE - 1})

        wall_count = _validate_wall_components(grid)
        out, stats = _trace(grid)
        assert stats["hit_count"] == wall_count, "every wall should participate in the branch tree"
        assert stats["hit_count"] >= 2, "must hit at least two walls"
        assert stats["seen_start_count"] >= 3, "must create meaningful branching"
        assert len(stats["quadrant_colors"]) >= 2, "painted path must cross at least two quadrants"
        assert stats["painted_count"] >= 12, "output must not be trivial"

        assert np.array_equal(main(grid), out)
        protected = (grid != BG)
        assert np.array_equal(out[protected], grid[protected]), "non-background structure must be preserved"
        assert np.any(out != grid), "transformation must paint at least one cell"
        return True
    except AssertionError:
        return False


def _gen_attempt() -> np.ndarray | None:
    edge = random.choice(EDGES)
    total_segments = random.randint(4, 6)
    plan = _build_canonical_plan(total_segments)
    if plan is None:
        return None
    grid = _render_grid(edge, plan)
    return grid if _is_valid_input(grid) else None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input for opus/12/005/4")
