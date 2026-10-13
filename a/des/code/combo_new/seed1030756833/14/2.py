from __future__ import annotations

import random

import numpy as np


Coord = tuple[int, int]

BG = 0
DIR4: tuple[Coord, ...] = ((1, 0), (-1, 0), (0, 1), (0, -1))
COLOR_POOL = list(range(1, 10))
GENERATION_ATTEMPTS = 600


def _colors_in_order(g: np.ndarray) -> list[int]:
    seen = {BG}
    colors: list[int] = []
    for raw in g.ravel():
        color = int(raw)
        if color not in seen:
            seen.add(color)
            colors.append(color)
    return colors


def _components(mask: np.ndarray) -> list[list[Coord]]:
    h, w = mask.shape
    seen = np.zeros(mask.shape, dtype=bool)
    comps: list[list[Coord]] = []
    for r0, c0 in np.argwhere(mask):
        r = int(r0)
        c = int(c0)
        if seen[r, c]:
            continue
        stack = [(r, c)]
        seen[r, c] = True
        comp: list[Coord] = []
        while stack:
            pr, pc = stack.pop()
            comp.append((pr, pc))
            for dr, dc in DIR4:
                nr = pr + dr
                nc = pc + dc
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    stack.append((nr, nc))
        comps.append(comp)
    comps.sort(key=lambda pts: min(pts))
    return comps


def _is_edge(p: Coord, h: int, w: int) -> bool:
    r, c = p
    return r == 0 or c == 0 or r == h - 1 or c == w - 1


def _endpoint_and_outward(comp: list[Coord], h: int, w: int) -> tuple[Coord, Coord]:
    cells = set(comp)
    candidates: list[Coord] = []
    for r, c in comp:
        deg = sum((r + dr, c + dc) in cells for dr, dc in DIR4)
        if deg == 1 and not _is_edge((r, c), h, w):
            candidates.append((r, c))

    assert len(candidates) == 1, "each path fragment must have exactly one interior open endpoint"
    endpoint = candidates[0]
    er, ec = endpoint
    neighbor = None
    for dr, dc in DIR4:
        cand = (er + dr, ec + dc)
        if cand in cells:
            neighbor = cand
            break
    assert neighbor is not None, "open endpoint must touch its own fragment"
    nr, nc = neighbor
    return endpoint, (er - nr, ec - nc)


def _step_toward(a: int, b: int) -> int:
    if b > a:
        return 1
    if b < a:
        return -1
    return 0


def _path_cells(start: Coord, stop: Coord, first_axis: str) -> list[Coord]:
    r0, c0 = start
    r1, c1 = stop
    cells: list[Coord] = []

    if first_axis == "h":
        dc = _step_toward(c0, c1)
        if dc == 0:
            cells.append((r0, c0))
        else:
            for c in range(c0, c1 + dc, dc):
                cells.append((r0, c))
        dr = _step_toward(r0, r1)
        if dr != 0:
            for r in range(r0 + dr, r1 + dr, dr):
                cells.append((r, c1))
        return cells

    assert first_axis == "v"
    dr = _step_toward(r0, r1)
    if dr == 0:
        cells.append((r0, c0))
    else:
        for r in range(r0, r1 + dr, dr):
            cells.append((r, c0))
    dc = _step_toward(c0, c1)
    if dc != 0:
        for c in range(c0 + dc, c1 + dc, dc):
            cells.append((r1, c))
    return cells


def _path_options(start: Coord, stop: Coord) -> list[tuple[str, list[Coord]]]:
    if start[0] == stop[0]:
        return [("h", _path_cells(start, stop, "h"))]
    if start[1] == stop[1]:
        return [("v", _path_cells(start, stop, "v"))]
    return [("h", _path_cells(start, stop, "h")), ("v", _path_cells(start, stop, "v"))]


def _direction_score(path: list[Coord], start: Coord, stop: Coord, outward: dict[Coord, Coord]) -> int:
    if len(path) < 2:
        return 0
    sr, sc = start
    tr, tc = stop
    p1r, p1c = path[1]
    pnr, pnc = path[-2]
    score = 0
    if (p1r - sr, p1c - sc) == outward[start]:
        score += 1
    if (pnr - tr, pnc - tc) == outward[stop]:
        score += 1
    return score


def _select_connection(
    original: np.ndarray, canvas: np.ndarray, color: int
) -> tuple[list[Coord], str, int]:
    h, w = original.shape
    comps = _components(original == color)
    assert len(comps) == 2, "each active color must have exactly two path fragments"

    endpoints: list[Coord] = []
    outward: dict[Coord, Coord] = {}
    for comp in comps:
        endpoint, out_step = _endpoint_and_outward(comp, h, w)
        endpoints.append(endpoint)
        outward[endpoint] = out_step

    start, stop = sorted(endpoints)
    best: tuple[int, int, int, str, list[Coord]] | None = None
    for axis, path in _path_options(start, stop):
        direction = _direction_score(path, start, stop, outward)
        conflicts = sum(
            int(canvas[r, c]) not in (BG, color)
            for r, c in path
        )
        axis_rank = 0 if axis == "h" else 1
        key = (-direction, conflicts, axis_rank, axis, path)
        if best is None or key[:3] < best[:3]:
            best = key

    assert best is not None
    _neg_direction, conflicts, _axis_rank, axis, path = best
    return path, axis, conflicts


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete same-colored broken paths with tunneled one-bend orthogonal connectors.

    Input structure:
    - The background is black. Two to four non-black colors appear as one-pixel-wide path fragments.
    - Each active color forms exactly two 4-connected components, and each component has one interior open endpoint.
    - Existing fragments are horizontal/vertical strokes, possibly with right-angle bends, and different colors may lie on future connector crossings.

    Rule:
    - Process colors in the order of their first row-major appearance in the input.
    - For each color, connect the two interior open endpoints with the better shortest L path: prefer the bend whose first and last steps continue the visible fragment directions, then the bend with fewer occupied crossings.
    - Draw the chosen connector only on black cells; if the connector crosses an already colored cell, that cell keeps its previous color and the new path tunnels underneath it.

    Key insight:
    - The task is both path completion and layering: endpoints determine the missing L-shaped route, while crossing cells preserve the color that was already present or drawn earlier.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    for color in _colors_in_order(g):
        path, _axis, _conflicts = _select_connection(g, out, color)
        for r, c in path:
            if int(out[r, c]) in (BG, color):
                out[r, c] = color
    return out


def _dist_to_border(p: Coord, direction: Coord, h: int, w: int) -> int:
    r, c = p
    dr, dc = direction
    if dr == 1:
        return h - 1 - r
    if dr == -1:
        return r
    if dc == 1:
        return w - 1 - c
    assert dc == -1
    return c


def _component_from_endpoint(
    endpoint: Coord,
    outward: Coord,
    h: int,
    w: int,
    rng: random.Random,
    may_bend: bool,
) -> list[Coord]:
    back = (-outward[0], -outward[1])
    first_dist = _dist_to_border(endpoint, back, h, w)
    cells = [endpoint]

    if may_bend and first_dist >= 3 and rng.random() < 0.45:
        leg = rng.randint(1, min(first_dist - 1, 4))
        cur = endpoint
        for _ in range(leg):
            cur = (cur[0] + back[0], cur[1] + back[1])
            cells.append(cur)
        perps = [(back[1], -back[0]), (-back[1], back[0])]
        rng.shuffle(perps)
        turn = next((d for d in perps if _dist_to_border(cur, d, h, w) > 0), perps[0])
        for _ in range(_dist_to_border(cur, turn, h, w)):
            cur = (cur[0] + turn[0], cur[1] + turn[1])
            cells.append(cur)
        return cells

    cur = endpoint
    for _ in range(first_dist):
        cur = (cur[0] + back[0], cur[1] + back[1])
        cells.append(cur)
    return cells


def _can_place(cells: list[Coord], grid: np.ndarray, reserved: np.ndarray) -> bool:
    h, w = grid.shape
    unique = set(cells)
    if len(unique) != len(cells):
        return False
    for r, c in cells:
        if not (0 <= r < h and 0 <= c < w):
            return False
        if int(grid[r, c]) != BG or bool(reserved[r, c]):
            return False
    for r, c in cells:
        for dr, dc in DIR4:
            nr = r + dr
            nc = c + dc
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in unique:
                if int(grid[nr, nc]) != BG:
                    return False
    return True


def _paint(cells: list[Coord], grid: np.ndarray, color: int) -> None:
    for r, c in cells:
        grid[r, c] = color


def _endpoint_outward_from_path(path: list[Coord], endpoint: Coord) -> Coord:
    if endpoint == path[0]:
        nr, nc = path[1]
    else:
        nr, nc = path[-2]
    er, ec = endpoint
    return nr - er, nc - ec


def _place_component(
    grid: np.ndarray,
    reserved: np.ndarray,
    endpoint: Coord,
    outward: Coord,
    color: int,
    rng: random.Random,
    may_bend: bool,
) -> bool:
    h, w = grid.shape
    for bend in ([may_bend, False] if may_bend else [False]):
        cells = _component_from_endpoint(endpoint, outward, h, w, rng, bend)
        if _can_place(cells, grid, reserved):
            _paint(cells, grid, color)
            return True
    return False


def _place_crossing_pair(grid: np.ndarray, reserved: np.ndarray, colors: list[int], rng: random.Random) -> bool:
    h, w = grid.shape
    color_a, color_b = colors[:2]

    for _ in range(80):
        cr = rng.randint(6, h - 7)
        cc = rng.randint(6, w - 7)
        c_left = rng.randint(2, cc - 3)
        c_right = rng.randint(cc + 3, w - 3)
        r_a_stop = rng.randint(cr + 2, h - 3)
        r_b_start = rng.randint(2, cr - 3)
        r_b_stop = rng.randint(cr + 2, h - 3)
        c_b_stop = rng.randint(2, cc - 3)

        trial = grid.copy()
        trial_reserved = reserved.copy()
        trial_reserved[cr, cc] = True

        a_start = (cr, c_left)
        a_stop = (r_a_stop, c_right)
        a_path = _path_cells(a_start, a_stop, "h")
        b_start = (r_b_start, cc)
        b_stop = (r_b_stop, c_b_stop)
        b_path = _path_cells(b_start, b_stop, "v")

        specs = [
            (color_a, a_start, _endpoint_outward_from_path(a_path, a_start)),
            (color_a, a_stop, _endpoint_outward_from_path(a_path, a_stop)),
            (color_b, b_start, _endpoint_outward_from_path(b_path, b_start)),
            (color_b, b_stop, _endpoint_outward_from_path(b_path, b_stop)),
        ]
        rng.shuffle(specs)

        ok = True
        for color, endpoint, outward in specs:
            if not _place_component(trial, trial_reserved, endpoint, outward, color, rng, may_bend=True):
                ok = False
                break
        if ok:
            grid[:, :] = trial
            reserved[:, :] = trial_reserved
            return True
    return False


def _place_extra_color(grid: np.ndarray, reserved: np.ndarray, color: int, rng: random.Random) -> bool:
    h, w = grid.shape
    for _ in range(220):
        p0 = (rng.randint(2, h - 3), rng.randint(2, w - 3))
        p1 = (rng.randint(2, h - 3), rng.randint(2, w - 3))
        if abs(p0[0] - p1[0]) + abs(p0[1] - p1[1]) < 6:
            continue
        start, stop = sorted([p0, p1])
        options = _path_options(start, stop)
        axis, path = rng.choice(options)

        outward_start = _endpoint_outward_from_path(path, start)
        outward_stop = _endpoint_outward_from_path(path, stop)
        cells0 = _component_from_endpoint(start, outward_start, h, w, rng, may_bend=True)
        cells1 = _component_from_endpoint(stop, outward_stop, h, w, rng, may_bend=True)
        connector_body = set(path[1:-1])
        if connector_body & (set(cells0) | set(cells1)):
            continue

        trial = grid.copy()
        if not _can_place(cells0, trial, reserved):
            continue
        _paint(cells0, trial, color)
        if not _can_place(cells1, trial, reserved):
            continue
        _paint(cells1, trial, color)
        grid[:, :] = trial
        return True
    return False


def _trace_solution(inp: np.ndarray) -> tuple[np.ndarray, int, set[str]]:
    out = inp.copy()
    tunnels = 0
    axes: set[str] = set()
    for color in _colors_in_order(inp):
        path, axis, _conflicts = _select_connection(inp, out, color)
        axes.add(axis)
        for r, c in path:
            if int(out[r, c]) not in (BG, color):
                tunnels += 1
            elif int(out[r, c]) == BG:
                out[r, c] = color
    return out, tunnels, axes


def _valid_input(grid: np.ndarray, expected_colors: int) -> bool:
    try:
        colors = _colors_in_order(grid)
        if len(colors) != expected_colors:
            return False
        for color in colors:
            _select_connection(grid, grid.copy(), color)
        traced, tunnels, axes = _trace_solution(grid)
        if tunnels < 1:
            return False
        if not {"h", "v"}.issubset(axes):
            return False
        if not np.array_equal(main(grid), traced):
            return False
        if int(np.sum(grid == BG)) <= grid.size // 2:
            return False
    except AssertionError:
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    h = rng.randint(17, 26)
    w = rng.randint(17, 26)
    n_colors = rng.randint(2, 4)
    colors = rng.sample(COLOR_POOL, n_colors)
    grid = np.full((h, w), BG, dtype=int)
    reserved = np.zeros((h, w), dtype=bool)

    if not _place_crossing_pair(grid, reserved, colors, rng):
        return None
    for color in colors[2:]:
        if not _place_extra_color(grid, reserved, color, rng):
            return None

    if not _valid_input(grid, n_colors):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {GENERATION_ATTEMPTS} attempts")
