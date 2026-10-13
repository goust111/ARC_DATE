from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
WIRE = 5
DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]  # U, R, D, L
TOKEN_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]

# Asymmetric and safe at distance 2 from a wire endpoint: all edge-center cells are empty.
BASE_TOKEN = np.array(
    [
        [1, 0, 1],
        [0, 1, 0],
        [1, 0, 0],
    ],
    dtype=int,
)


def _get_bg(grid: np.ndarray) -> int:
    vals, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(vals[np.argmax(counts)])


def _neighbors4(r: int, c: int):
    for dr, dc in DIRS:
        yield r + dr, c + dc


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _wire_candidate_score(grid: np.ndarray, color: int) -> int | None:
    comps = _components(grid == color)
    if not comps:
        return None
    total = 0
    for comp in comps:
        pts = set(comp)
        if len(pts) < 2:
            return None
        degs = []
        for r, c in pts:
            deg = sum((nr, nc) in pts for nr, nc in _neighbors4(r, c))
            degs.append(deg)
        if max(degs) > 2:
            return None
        if sum(deg == 1 for deg in degs) != 2:
            return None
        total += len(pts)
    return total


def _find_wire_color(grid: np.ndarray, bg: int) -> int:
    colors = [int(v) for v in np.unique(grid) if int(v) != bg]
    scored = []
    for color in colors:
        score = _wire_candidate_score(grid, color)
        if score is not None:
            scored.append((score, color))
    assert scored, "expected a wire color"
    scored.sort()
    return int(scored[-1][1])


def _extract_token(
    grid: np.ndarray, center: tuple[int, int], bg: int, wire_color: int
) -> tuple[int, np.ndarray] | None:
    cr, cc = center
    h, w = grid.shape
    if not (1 <= cr < h - 1 and 1 <= cc < w - 1):
        return None
    patch = np.asarray(grid[cr - 1 : cr + 2, cc - 1 : cc + 2], dtype=int)
    colors = {int(v) for v in np.unique(patch) if int(v) not in (bg, wire_color)}
    if len(colors) != 1:
        return None
    color = next(iter(colors))
    if np.any((patch != bg) & (patch != color)):
        return None
    for pr in range(3):
        for pc in range(3):
            if patch[pr, pc] != color:
                continue
            rr = cr - 1 + pr
            cc2 = cc - 1 + pc
            for nr, nc in _neighbors4(rr, cc2):
                if 0 <= nr < h and 0 <= nc < w and int(grid[nr, nc]) == wire_color:
                    return None
    return int(color), patch


def _trace_path(comp: list[tuple[int, int]], start: tuple[int, int]) -> list[tuple[int, int]]:
    pts = set(comp)
    path = [start]
    prev: tuple[int, int] | None = None
    cur = start
    while True:
        nxt = [nb for nb in _neighbors4(*cur) if nb in pts and nb != prev]
        if not nxt:
            break
        assert len(nxt) == 1, "wire must be a simple path"
        prev, cur = cur, nxt[0]
        path.append(cur)
    assert len(path) == len(pts), "wire path should cover the whole component"
    return path


def _dir_index(vec: tuple[int, int]) -> int:
    return DIRS.index(tuple(map(int, vec)))


def _paste_token(out: np.ndarray, patch: np.ndarray, center: tuple[int, int], color: int) -> None:
    cr, cc = center
    for pr in range(3):
        for pc in range(3):
            if int(patch[pr, pc]) != color:
                continue
            out[cr - 1 + pr, cc - 1 + pc] = color


def main(grid: np.ndarray) -> np.ndarray:
    """Transmit each endpoint token to the opposite end of its grey wire, rotating by the net turn from start heading to end heading.

    Input structure:
    - Black background contains several one-cell-thick grey orthogonal wire polylines that do not touch each other.
    - Exactly one endpoint of each wire has a monochrome asymmetric 3x3 token centered two cells outside that endpoint.
    - Token colors are distinct from the wire color and from one another.

    Rule:
    - For each wire, identify the endpoint that has the source token and trace the wire to the opposite endpoint.
    - Let the start heading be the first wire step away from the source endpoint, and the end heading be the final step into the destination endpoint.
    - Copy the 3x3 token to the point two cells beyond the destination endpoint, rotating it by the turn angle from the start heading to the end heading; keep the original wire and source token unchanged.

    Key insight:
    - The token behaves like a rigid marker attached to the travel direction, so only the difference between the first and last segment directions determines the copied token's rotation.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    wire_color = _find_wire_color(g, bg)
    out = g.copy()

    for comp in _components(g == wire_color):
        pts = set(comp)
        endpoints = []
        for r, c in comp:
            deg = sum((nr, nc) in pts for nr, nc in _neighbors4(r, c))
            if deg == 1:
                endpoints.append((r, c))
        assert len(endpoints) == 2, "each wire must have two endpoints"

        source_info = None
        for ep in endpoints:
            nbrs = [nb for nb in _neighbors4(*ep) if nb in pts]
            assert len(nbrs) == 1
            nr, nc = nbrs[0]
            travel_dir = (nr - ep[0], nc - ep[1])
            center = (ep[0] - 2 * travel_dir[0], ep[1] - 2 * travel_dir[1])
            token = _extract_token(g, center, bg, wire_color)
            if token is not None:
                source_info = (ep, token[0], token[1])
                break

        assert source_info is not None, "each wire needs one source token"
        src_ep, token_color, token_patch = source_info
        path = _trace_path(comp, src_ep)
        dst_ep = path[-1]

        start_dir = (path[1][0] - path[0][0], path[1][1] - path[0][1])
        end_dir = (path[-1][0] - path[-2][0], path[-1][1] - path[-2][1])
        turn_steps = (_dir_index(end_dir) - _dir_index(start_dir)) % 4
        rotated = np.rot90(token_patch, k=(-turn_steps) % 4)
        dst_center = (dst_ep[0] + 2 * end_dir[0], dst_ep[1] + 2 * end_dir[1])
        _paste_token(out, rotated, dst_center, token_color)

    return out


def _rotate_cw_point(point: tuple[int, int], n: int) -> tuple[int, int]:
    r, c = point
    return c, n - 1 - r


def _rotate_points(points: list[tuple[int, int]], n: int, turns: int) -> list[tuple[int, int]]:
    out = list(points)
    for _ in range(turns % 4):
        out = [_rotate_cw_point(pt, n) for pt in out]
    return out


def _draw_segment(cells: list[tuple[int, int]], a: tuple[int, int], b: tuple[int, int]) -> None:
    if a[0] == b[0]:
        step = 1 if b[1] >= a[1] else -1
        for c in range(a[1], b[1] + step, step):
            cells.append((a[0], c))
    else:
        step = 1 if b[0] >= a[0] else -1
        for r in range(a[0], b[0] + step, step):
            cells.append((r, a[1]))


def _path_from_waypoints(pts: list[tuple[int, int]]) -> list[tuple[int, int]]:
    cells: list[tuple[int, int]] = []
    for i in range(len(pts) - 1):
        seg: list[tuple[int, int]] = []
        _draw_segment(seg, pts[i], pts[i + 1])
        if i:
            seg = seg[1:]
        cells.extend(seg)
    return cells


def _canonical_waypoints(n: int, rel_turn: int) -> list[tuple[int, int]]:
    a = 3
    b = n - 4
    m = (a + b) // 2
    if rel_turn == 0:
        return [(m, a), (m, b)]
    if rel_turn == 1:
        return [(a, a), (a, b), (b, b)]
    if rel_turn == 2:
        return [(a, a), (a, b), (b, b), (b, a)]
    if rel_turn == 3:
        return [(b, a), (b, b), (a, b)]
    raise ValueError(rel_turn)


def _turn_dir(vec: tuple[int, int], steps: int) -> tuple[int, int]:
    idx = _dir_index(vec)
    return DIRS[(idx + steps) % 4]


def _build_local_wire(n: int) -> tuple[np.ndarray, tuple[int, int], tuple[int, int]]:
    start_dir = random.choice(DIRS)
    rel_turn = random.randrange(4)
    base = _canonical_waypoints(n, rel_turn)
    turns_from_east = _dir_index(start_dir) - _dir_index((0, 1))
    waypoints = _rotate_points(base, n, turns_from_east)
    cells = _path_from_waypoints(waypoints)

    local = np.full((n, n), BG, dtype=int)
    for r, c in cells:
        local[r, c] = WIRE

    src_ep = waypoints[0]
    dst_ep = waypoints[-1]
    return local, src_ep, dst_ep


def _stamp_token(
    grid: np.ndarray, center: tuple[int, int], color: int, patch: np.ndarray
) -> None:
    cr, cc = center
    for pr in range(3):
        for pc in range(3):
            if patch[pr, pc]:
                grid[cr - 1 + pr, cc - 1 + pc] = color


def _box_candidates(
    occ: np.ndarray, size: int, pad: int
) -> list[tuple[int, int]]:
    h, w = occ.shape
    out = []
    for r0 in range(h - size + 1):
        for c0 in range(w - size + 1):
            rr0 = max(0, r0 - pad)
            cc0 = max(0, c0 - pad)
            rr1 = min(h, r0 + size + pad)
            cc1 = min(w, c0 + size + pad)
            if not occ[rr0:rr1, cc0:cc1].any():
                out.append((r0, c0))
    return out


def _place_boxes(shape: tuple[int, int], sizes: list[int], pad: int) -> list[tuple[int, int]] | None:
    occ = np.zeros(shape, dtype=bool)
    placements: list[tuple[int, int] | None] = [None] * len(sizes)
    order = sorted(range(len(sizes)), key=lambda i: -sizes[i])
    for idx in order:
        size = sizes[idx]
        candidates = _box_candidates(occ, size, pad)
        if not candidates:
            return None
        r0, c0 = random.choice(candidates)
        placements[idx] = (r0, c0)
        occ[r0 : r0 + size, c0 : c0 + size] = True
    return [pt for pt in placements if pt is not None]


def _make_case() -> np.ndarray | None:
    count = random.choices([1, 2, 3], weights=[2, 4, 3])[0]
    if count == 1:
        grid_h = random.randint(18, 30)
        grid_w = random.randint(18, 30)
    elif count == 2:
        grid_h = random.randint(20, 30)
        grid_w = random.randint(20, 30)
    else:
        grid_h = random.randint(27, 30)
        grid_w = random.randint(27, 30)
    sizes = [random.randint(9, 10) for _ in range(count)]
    placements = _place_boxes((grid_h, grid_w), sizes, pad=1)
    if placements is None:
        return None

    grid = np.full((grid_h, grid_w), BG, dtype=int)
    colors = random.sample(TOKEN_COLORS, count)

    for size, (r0, c0), color in zip(sizes, placements, colors):
        local, src_ep, _dst_ep = _build_local_wire(size)
        token_turn = random.randrange(4)
        token_patch = np.rot90(BASE_TOKEN, k=(-token_turn) % 4)

        grid[r0 : r0 + size, c0 : c0 + size] = np.maximum(
            grid[r0 : r0 + size, c0 : c0 + size], local
        )

        wire_cells = np.argwhere(local == WIRE)
        pts = {(int(r), int(c)) for r, c in wire_cells}
        nbrs = [nb for nb in _neighbors4(*src_ep) if nb in pts]
        assert len(nbrs) == 1
        travel_dir = (nbrs[0][0] - src_ep[0], nbrs[0][1] - src_ep[1])
        src_center = (src_ep[0] - 2 * travel_dir[0], src_ep[1] - 2 * travel_dir[1])
        _stamp_token(grid, (r0 + src_center[0], c0 + src_center[1]), color, token_patch)

    try:
        out = main(grid)
    except Exception:
        return None

    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(300):
        grid = _make_case()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
