from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass

import numpy as np


BG = 0
YELLOW = 4
GREY = 5
PATH_COLORS = [1, 2, 3, 6, 7, 8, 9]
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
GEN_RETRIES = 500


@dataclass(frozen=True)
class RectSpec:
    index: int
    r0: int
    c0: int
    r1: int
    c1: int
    cell_r: int
    cell_c: int

    @property
    def h(self) -> int:
        return self.r1 - self.r0 + 1

    @property
    def w(self) -> int:
        return self.c1 - self.c0 + 1


@dataclass
class PathPlan:
    color: int
    marker: tuple[int, int]
    start: tuple[int, int]
    end: tuple[int, int]
    side: str
    rect_index: int
    points: list[tuple[int, int]]
    cells: list[tuple[int, int]]


def _in_bounds(h: int, w: int, r: int, c: int) -> bool:
    return 0 <= r < h and 0 <= c < w


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _mode_color(grid: np.ndarray) -> int:
    vals, cnts = np.unique(grid, return_counts=True)
    return int(vals[int(cnts.argmax())])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if _in_bounds(h, w, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _find_rectangles(grid: np.ndarray) -> list[RectSpec]:
    rects: list[RectSpec] = []
    for idx, comp in enumerate(_components(grid == GREY)):
        rs = [r for r, _ in comp]
        cs = [c for _, c in comp]
        r0, r1 = min(rs), max(rs)
        c0, c1 = min(cs), max(cs)
        patch = grid[r0 : r1 + 1, c0 : c1 + 1]
        assert np.all(patch == GREY), "grey component must be a solid rectangle"
        rects.append(RectSpec(idx, r0, c0, r1, c1, -1, -1))
    rects.sort(key=lambda rect: (rect.r0, rect.c0))
    return [RectSpec(i, rect.r0, rect.c0, rect.r1, rect.c1, rect.cell_r, rect.cell_c) for i, rect in enumerate(rects)]


def _adjacent_rectangle_side(
    pos: tuple[int, int], rects: list[RectSpec]
) -> tuple[int, str] | None:
    r, c = pos
    hits: list[tuple[int, str]] = []
    for rect in rects:
        if r == rect.r0 - 1 and rect.c0 <= c <= rect.c1:
            hits.append((rect.index, "top"))
        if r == rect.r1 + 1 and rect.c0 <= c <= rect.c1:
            hits.append((rect.index, "bottom"))
        if c == rect.c0 - 1 and rect.r0 <= r <= rect.r1:
            hits.append((rect.index, "left"))
        if c == rect.c1 + 1 and rect.r0 <= r <= rect.r1:
            hits.append((rect.index, "right"))
    assert len(hits) <= 1, "path endpoint must be adjacent to at most one rectangle side"
    return hits[0] if hits else None


def _walkable_path_pixel(value: int, bg: int) -> bool:
    return value not in (bg, YELLOW, GREY)


def _next_step(
    grid: np.ndarray,
    pos: tuple[int, int],
    direction: tuple[int, int],
    path_color: int,
    bg: int,
) -> tuple[tuple[int, int], tuple[int, int]] | None:
    h, w = grid.shape
    r, c = pos
    dr, dc = direction
    value = int(grid[r, c])

    fr, fc = r + dr, c + dc
    if value != path_color:
        if _in_bounds(h, w, fr, fc) and _walkable_path_pixel(int(grid[fr, fc]), bg):
            return (fr, fc), direction
        return None

    if _in_bounds(h, w, fr, fc):
        fv = int(grid[fr, fc])
        if fv == path_color:
            return (fr, fc), direction
        if _walkable_path_pixel(fv, bg):
            br, bc = fr + dr, fc + dc
            if _in_bounds(h, w, br, bc) and int(grid[br, bc]) == path_color:
                return (fr, fc), direction

    back = (-dr, -dc)
    turns: list[tuple[tuple[int, int], tuple[int, int]]] = []
    for ndr, ndc in DIRS4:
        if (ndr, ndc) == back or (ndr, ndc) == direction:
            continue
        nr, nc = r + ndr, c + ndc
        if _in_bounds(h, w, nr, nc) and int(grid[nr, nc]) == path_color:
            turns.append(((nr, nc), (ndr, ndc)))
    if len(turns) == 1:
        return turns[0]
    return None


def _trace_path(
    grid: np.ndarray,
    start: tuple[int, int],
    direction: tuple[int, int],
    path_color: int,
    rects: list[RectSpec],
    bg: int,
) -> tuple[int, str]:
    pos = start
    state_seen = {(start, direction)}
    for _ in range(grid.size * 4):
        nxt = _next_step(grid, pos, direction, path_color, bg)
        if nxt is None:
            hit = _adjacent_rectangle_side(pos, rects)
            assert hit is not None, "traced path must end beside a grey rectangle"
            return hit
        pos, direction = nxt
        state = (pos, direction)
        assert state not in state_seen, "path tracing loop detected"
        state_seen.add(state)
    raise AssertionError("path tracing exceeded step budget")


def _fill_half(out: np.ndarray, rect: RectSpec, side: str, color: int) -> None:
    if side == "top":
        depth = max(1, rect.h // 2)
        out[rect.r0 : rect.r0 + depth, rect.c0 : rect.c1 + 1] = color
    elif side == "bottom":
        depth = max(1, rect.h // 2)
        out[rect.r1 - depth + 1 : rect.r1 + 1, rect.c0 : rect.c1 + 1] = color
    elif side == "left":
        depth = max(1, rect.w // 2)
        out[rect.r0 : rect.r1 + 1, rect.c0 : rect.c0 + depth] = color
    else:
        depth = max(1, rect.w // 2)
        out[rect.r0 : rect.r1 + 1, rect.c1 - depth + 1 : rect.c1 + 1] = color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Trace colored paths through hidden crossings and fill the destination grey rectangles from the reached side.

    Input structure:
    - A black background contains several single-pixel-wide colored orthogonal paths.
    - Each path starts next to a yellow border marker and terminates beside a small solid grey rectangle.
    - Different-colored paths may cross at one shared pixel, where only one of the two colors is visible.

    Rule:
    - Start from each yellow border marker and follow the adjacent path color.
    - At an off-color crossing pixel, keep moving straight through the crossing instead of turning.
    - At a normal corner, turn only when there is exactly one same-color continuation other than the cell you came from.
    - When the path ends next to a grey rectangle, detect whether it reaches the top, bottom, left, or right side.
    - Fill half of that rectangle from the reached side using the path color; multiple arrivals can fill different sides of the same rectangle.

    Key insight:
    - The visible crossing color belongs to only one path, so the hidden path must be recovered by preserving direction through crossings rather than by following the visible color locally.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _mode_color(g)
    rects = _find_rectangles(g)
    rect_by_index = {rect.index: rect for rect in rects}

    markers = [(int(r), int(c)) for r, c in np.argwhere(g == YELLOW) if r in (0, g.shape[0] - 1) or c in (0, g.shape[1] - 1)]
    markers.sort()
    assert markers, "expected at least one yellow border marker"

    out = g.copy()
    for mr, mc in markers:
        starts: list[tuple[int, int]] = []
        for nr, nc in _neighbors4(mr, mc):
            if _in_bounds(*g.shape, nr, nc) and _walkable_path_pixel(int(g[nr, nc]), bg):
                starts.append((nr, nc))
        assert len(starts) == 1, "each yellow marker must touch exactly one path start"
        start = starts[0]
        direction = (start[0] - mr, start[1] - mc)
        path_color = int(g[start])
        rect_index, side = _trace_path(g, start, direction, path_color, rects, bg)
        _fill_half(out, rect_by_index[rect_index], side, path_color)
    return out


def _cell_rect(cell_r: int, cell_c: int, pitch: int, cell_size: int, index: int) -> RectSpec:
    r0 = cell_r * pitch + 1
    c0 = cell_c * pitch + 1
    r1 = r0 + cell_size - 1
    c1 = c0 + cell_size - 1
    return RectSpec(index, r0, c0, r1, c1, cell_r, cell_c)


def _sample_rectangles(
    cell_rows: int, cell_cols: int, pitch: int, cell_size: int, rng: random.Random
) -> list[RectSpec] | None:
    interior = [(r, c) for r in range(1, cell_rows - 1) for c in range(1, cell_cols - 1)]
    if not interior:
        return None

    want_two = len(interior) >= 3 and rng.random() < 0.45
    rect_cells = [rng.choice(interior)]
    if want_two:
        rng.shuffle(interior)
        for cand in interior:
            dr = abs(cand[0] - rect_cells[0][0])
            dc = abs(cand[1] - rect_cells[0][1])
            if not (dr == 0 and dc == 0) and not (dr + dc == 1):
                rect_cells.append(cand)
                break
        if len(rect_cells) == 1:
            want_two = False

    rects = [_cell_rect(cr, cc, pitch, cell_size, idx) for idx, (cr, cc) in enumerate(rect_cells)]
    return rects


def _choose_assignments(rects: list[RectSpec], rng: random.Random) -> list[tuple[int, str]]:
    sides = ["top", "bottom", "left", "right"]
    if len(rects) == 1:
        count = rng.choice([2, 2, 3, 3, 4])
        chosen = rng.sample(sides, count)
        return [(0, side) for side in chosen]

    patterns = [(2, 1), (1, 2), (2, 2)]
    counts = rng.choice(patterns)
    assignments: list[tuple[int, str]] = []
    for rect_index, count in enumerate(counts):
        chosen = rng.sample(sides, count)
        assignments.extend((rect_index, side) for side in chosen)
    rng.shuffle(assignments)
    return assignments


def _side_endpoint(rect: RectSpec, side: str, rng: random.Random) -> tuple[int, int]:
    if side == "top":
        c = rng.choice(list(range(rect.c0 + 1, rect.c1)))
        return rect.r0 - 1, c
    if side == "bottom":
        c = rng.choice(list(range(rect.c0 + 1, rect.c1)))
        return rect.r1 + 1, c
    if side == "left":
        r = rng.choice(list(range(rect.r0 + 1, rect.r1)))
        return r, rect.c0 - 1
    r = rng.choice(list(range(rect.r0 + 1, rect.r1)))
    return r, rect.c1 + 1


def _rasterize(points: list[tuple[int, int]]) -> list[tuple[int, int]]:
    cells = [points[0]]
    for (r0, c0), (r1, c1) in zip(points, points[1:], strict=False):
        dr = 0 if r1 == r0 else (1 if r1 > r0 else -1)
        dc = 0 if c1 == c0 else (1 if c1 > c0 else -1)
        r, c = r0, c0
        while (r, c) != (r1, c1):
            r += dr
            c += dc
            cells.append((r, c))
    return cells


def _path_kind(path: PathPlan, cell: tuple[int, int]) -> str:
    idx = path.cells.index(cell)
    if idx == 0 or idx == len(path.cells) - 1:
        return "END"
    pr, pc = path.cells[idx - 1]
    nr, nc = path.cells[idx + 1]
    if pr == nr:
        return "H"
    if pc == nc:
        return "V"
    return "TURN"


def _validate_paths(
    paths: list[PathPlan],
    rects: list[RectSpec],
    h: int,
    w: int,
) -> tuple[bool, int]:
    rect_cells = {
        (r, c)
        for rect in rects
        for r in range(rect.r0, rect.r1 + 1)
        for c in range(rect.c0, rect.c1 + 1)
    }
    marker_cells = {path.marker for path in paths}
    path_sets = [set(path.cells) for path in paths]

    for path in paths:
        if len(path.cells) != len(set(path.cells)):
            return False, 0
        if len(path.points) < 3:
            return False, 0
        if not all(_in_bounds(h, w, r, c) for r, c in path.cells):
            return False, 0
        if any(cell in rect_cells for cell in path.cells):
            return False, 0
        if path.marker in rect_cells or path.marker in marker_cells - {path.marker}:
            return False, 0
        if path.start != path.cells[0] or path.end != path.cells[-1]:
            return False, 0
        hit = _adjacent_rectangle_side(path.end, rects)
        if hit != (path.rect_index, path.side):
            return False, 0

    occ: dict[tuple[int, int], list[int]] = {}
    for idx, path in enumerate(paths):
        for cell in path.cells:
            occ.setdefault(cell, []).append(idx)

    crossing_cells: set[tuple[int, int]] = set()
    for cell, owners in occ.items():
        if len(owners) > 2:
            return False, 0
        if len(owners) == 2:
            kind_a = _path_kind(paths[owners[0]], cell)
            kind_b = _path_kind(paths[owners[1]], cell)
            if {kind_a, kind_b} != {"H", "V"}:
                return False, 0
            crossing_cells.add(cell)

    for idx, path in enumerate(paths):
        for cell in path.cells:
            for nr, nc in _neighbors4(*cell):
                owners = occ.get((nr, nc), [])
                for other_idx in owners:
                    if other_idx == idx:
                        continue
                    if (nr, nc) in path_sets[idx] or cell in path_sets[other_idx]:
                        continue
                    return False, 0

    for path in paths:
        adj = [(nr, nc) for nr, nc in _neighbors4(*path.marker) if _in_bounds(h, w, nr, nc) and (nr, nc) in set(path.cells)]
        if adj != [path.start]:
            return False, 0
        if path.marker[0] not in (0, h - 1) and path.marker[1] not in (0, w - 1):
            return False, 0

    return True, len(crossing_cells)


def _candidate_routes(
    end: tuple[int, int],
    side: str,
    rect_index: int,
    color: int,
    h: int,
    w: int,
    lane_rows: list[int],
    lane_cols: list[int],
    rng: random.Random,
) -> list[PathPlan]:
    end_orientation = "H" if side in ("top", "bottom") else "V"
    candidates: list[PathPlan] = []
    edges = ["top", "bottom", "left", "right"]
    rng.shuffle(edges)

    for edge in edges:
        if edge == "top":
            starts = [((0, col), (1, col), "V") for col in lane_cols]
        elif edge == "bottom":
            starts = [((h - 1, col), (h - 2, col), "V") for col in lane_cols]
        elif edge == "left":
            starts = [((row, 0), (row, 1), "H") for row in lane_rows]
        else:
            starts = [((row, w - 1), (row, w - 2), "H") for row in lane_rows]
        rng.shuffle(starts)

        for marker, start, start_orientation in starts:
            points: list[tuple[int, int]] | None = None
            if start_orientation != end_orientation:
                turn = (end[0], start[1]) if start_orientation == "V" else (start[0], end[1])
                if turn != start and turn != end:
                    points = [start, turn, end]
            elif start_orientation == "V":
                if start[1] == end[1]:
                    continue
                mids = lane_rows[:]
                rng.shuffle(mids)
                for mid_r in mids:
                    points = [start, (mid_r, start[1]), (mid_r, end[1]), end]
                    break
            else:
                if start[0] == end[0]:
                    continue
                mids = lane_cols[:]
                rng.shuffle(mids)
                for mid_c in mids:
                    points = [start, (start[0], mid_c), (end[0], mid_c), end]
                    break

            if points is None:
                continue

            cells = _rasterize(points)
            candidates.append(
                PathPlan(
                    color=color,
                    marker=marker,
                    start=start,
                    end=end,
                    side=side,
                    rect_index=rect_index,
                    points=points,
                    cells=cells,
                )
            )
    return candidates


def _draw_scene(h: int, w: int, rects: list[RectSpec], paths: list[PathPlan], rng: random.Random) -> np.ndarray:
    grid = np.full((h, w), BG, dtype=int)
    for rect in rects:
        grid[rect.r0 : rect.r1 + 1, rect.c0 : rect.c1 + 1] = GREY
    order = list(range(len(paths)))
    rng.shuffle(order)
    for idx in order:
        path = paths[idx]
        for r, c in path.cells:
            grid[r, c] = path.color
    for path in paths:
        grid[path.marker] = YELLOW
    return grid


def _build_scene(rng: random.Random) -> np.ndarray | None:
    cell_size = rng.choice([4, 6])
    pitch = cell_size + 1
    max_cells = 5 if cell_size == 4 else 4
    cell_rows = rng.randint(3, max_cells)
    cell_cols = rng.randint(3, max_cells)
    h = cell_rows * pitch + 1
    w = cell_cols * pitch + 1

    rects = _sample_rectangles(cell_rows, cell_cols, pitch, cell_size, rng)
    if rects is None:
        return None

    assignments = _choose_assignments(rects, rng)
    if len(assignments) < 2:
        return None

    lane_rows = [k * pitch for k in range(1, cell_rows)]
    lane_cols = [k * pitch for k in range(1, cell_cols)]
    colors = rng.sample(PATH_COLORS, len(assignments))
    endpoints = [_side_endpoint(rects[rect_index], side, rng) for rect_index, side in assignments]

    paths: list[PathPlan] = []
    required_crossings = 1 if len(assignments) <= 3 else rng.choice([1, 2])
    current_crossings = 0

    for (rect_index, side), end, color in zip(assignments, endpoints, colors, strict=False):
        candidates = _candidate_routes(end, side, rect_index, color, h, w, lane_rows, lane_cols, rng)
        scored: list[tuple[int, int, float, PathPlan]] = []
        for cand in candidates:
            ok, crossing_count = _validate_paths(paths + [cand], rects, h, w)
            if not ok:
                continue
            gain = crossing_count - current_crossings
            turns = len(cand.points) - 2
            scored.append((1 if gain > 0 else 0, crossing_count, turns, rng.random(), cand))
        if not scored:
            return None
        if current_crossings < required_crossings:
            scored.sort(key=lambda item: (item[0], item[1], item[2], item[3]), reverse=True)
        else:
            scored.sort(key=lambda item: (item[1], item[2], item[3]), reverse=True)
        best = scored[0]
        paths.append(best[4])
        current_crossings = best[1]

    ok, crossing_count = _validate_paths(paths, rects, h, w)
    if not ok or crossing_count < required_crossings:
        return None

    grid = _draw_scene(h, w, rects, paths, rng)
    out = main(grid)
    if np.array_equal(grid, out):
        return None
    if len({tuple(path.marker) for path in paths}) != len(paths):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        rng = random.Random()
        grid = _build_scene(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
