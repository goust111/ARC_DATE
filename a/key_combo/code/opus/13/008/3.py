from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
MOVER = 3
WALL = 5
DEST = 7
ANCHOR = 8
BLOCK_SIZE = 2

DIAG_DIRS = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

GEN_ATTEMPTS = 600


def _find_unique_pixel(grid: np.ndarray, color: int) -> tuple[int, int]:
    pts = np.argwhere(grid == int(color))
    assert pts.shape == (1, 2), f"expected one pixel of color {color}"
    return int(pts[0, 0]), int(pts[0, 1])


def _find_square_top_left(grid: np.ndarray, color: int) -> tuple[int, int]:
    pts = np.argwhere(grid == int(color))
    assert pts.shape == (4, 2), f"expected a 2x2 square of color {color}"
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    assert int(r1 - r0) == 1 and int(c1 - c0) == 1, f"color {color} must form a 2x2 square"
    patch = grid[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]
    assert bool((patch == int(color)).all())
    return int(r0), int(c0)


def _full_arm_path(anchor: tuple[int, int], dr: int, dc: int, shape: tuple[int, int]) -> list[tuple[int, int]]:
    """Enumerate one staircase arm with horizontal-then-vertical steps of length two."""
    h, w = map(int, shape)
    r, c = map(int, anchor)
    path: list[tuple[int, int]] = []
    while True:
        for _ in range(2):
            c += int(dc)
            if not (0 <= r < h and 0 <= c < w):
                return path
            path.append((int(r), int(c)))
        for _ in range(2):
            r += int(dr)
            if not (0 <= r < h and 0 <= c < w):
                return path
            path.append((int(r), int(c)))


def _paint_arm(out: np.ndarray, anchor: tuple[int, int], dr: int, dc: int) -> None:
    h, w = out.shape
    r, c = map(int, anchor)
    while True:
        stopped = False
        for _ in range(2):
            nc = int(c + dc)
            if not (0 <= r < h and 0 <= nc < w) or int(out[r, nc]) == WALL:
                stopped = True
                break
            c = nc
            out[r, c] = WALL
        if stopped:
            break
        for _ in range(2):
            nr = int(r + dr)
            if not (0 <= nr < h and 0 <= c < w) or int(out[nr, c]) == WALL:
                stopped = True
                break
            r = nr
            out[r, c] = WALL
        if stopped:
            break


def _build_wall_grid(grid: np.ndarray) -> np.ndarray:
    out = np.array(grid, dtype=int, copy=True)
    anchor = _find_unique_pixel(out, ANCHOR)
    for dr, dc in DIAG_DIRS:
        _paint_arm(out, anchor, dr, dc)
    out[anchor] = ANCHOR
    return out


def _feasible_positions(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    open_mask = (grid != WALL) & (grid != ANCHOR)
    h, w = open_mask.shape
    feasible = np.zeros((h - 1, w - 1), dtype=bool)
    for r in range(h - 1):
        for c in range(w - 1):
            feasible[r, c] = bool(open_mask[r : r + BLOCK_SIZE, c : c + BLOCK_SIZE].all())
    return feasible, open_mask


def _slide_once(feasible: np.ndarray, pos: tuple[int, int], dr: int, dc: int) -> tuple[int, int] | None:
    h2, w2 = feasible.shape
    r, c = map(int, pos)
    rr, cc = r, c
    moved = False
    while True:
        nr = int(rr + dr)
        nc = int(cc + dc)
        if not (0 <= nr < h2 and 0 <= nc < w2 and bool(feasible[nr, nc])):
            break
        rr, cc = nr, nc
        moved = True
    if not moved:
        return None
    return int(rr), int(cc)


def _slide_neighbors(feasible: np.ndarray, pos: tuple[int, int]) -> list[tuple[int, int]]:
    nbrs: set[tuple[int, int]] = set()
    for dr, dc in DIR4:
        nxt = _slide_once(feasible, pos, dr, dc)
        if nxt is not None:
            nbrs.add(nxt)
    return sorted(nbrs)


def _slide_bfs(feasible: np.ndarray, start: tuple[int, int]) -> dict[tuple[int, int], int]:
    dist = {tuple(map(int, start)): 0}
    queue = deque([tuple(map(int, start))])
    while queue:
        cur = queue.popleft()
        for nxt in _slide_neighbors(feasible, cur):
            if nxt not in dist:
                dist[nxt] = int(dist[cur] + 1)
                queue.append(nxt)
    return dist


def _best_blocked_stop(dist: dict[tuple[int, int], int], dest: tuple[int, int]) -> tuple[int, int]:
    return max(
        dist,
        key=lambda pos: (
            int(dist[pos]),
            -abs(int(pos[0]) - int(dest[0])) - abs(int(pos[1]) - int(dest[1])),
            -int(pos[0]),
            -int(pos[1]),
        ),
    )


def _label_components(mask: np.ndarray) -> tuple[np.ndarray, list[list[tuple[int, int]]]]:
    labels = np.full(mask.shape, -1, dtype=int)
    comps: list[list[tuple[int, int]]] = []
    h, w = mask.shape
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or int(labels[r, c]) != -1:
                continue
            label = len(comps)
            queue = deque([(int(r), int(c))])
            labels[r, c] = label
            comp: list[tuple[int, int]] = []
            while queue:
                rr, cc = queue.popleft()
                comp.append((int(rr), int(cc)))
                for dr, dc in DIR4:
                    nr = int(rr + dr)
                    nc = int(cc + dc)
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and int(labels[nr, nc]) == -1:
                        labels[nr, nc] = label
                        queue.append((nr, nc))
            comps.append(comp)
    return labels, comps


def _corner_profile(pos: tuple[int, int], feasible_shape: tuple[int, int]) -> tuple[int, int]:
    r, c = map(int, pos)
    h2, w2 = map(int, feasible_shape)
    corners = [(0, 0), (0, w2 - 1), (h2 - 1, 0), (h2 - 1, w2 - 1)]
    dists = [abs(r - rr) + abs(c - cc) for rr, cc in corners]
    best = min(range(4), key=lambda idx: dists[idx])
    return int(best), int(dists[best])


def _analyze_layout(base_grid: np.ndarray) -> dict | None:
    rendered = _build_wall_grid(base_grid)
    feasible, open_mask = _feasible_positions(rendered)
    if feasible.shape[0] < 2 or feasible.shape[1] < 2 or int(feasible.sum()) < 36:
        return None

    f_labels, f_comps = _label_components(feasible)
    o_labels, _ = _label_components(open_mask)

    states = [tuple(map(int, rc)) for rc in np.argwhere(feasible)]
    neighbors = {state: _slide_neighbors(feasible, state) for state in states}
    degree = {state: len(neighbors[state]) for state in states}
    if max(degree.values(), default=0) < 2:
        return None

    return {
        "rendered": rendered,
        "feasible": feasible,
        "open_mask": open_mask,
        "f_labels": f_labels,
        "f_comps": f_comps,
        "o_labels": o_labels,
        "states": states,
        "neighbors": neighbors,
        "degree": degree,
    }


def _pick_reachable_positions(data: dict) -> tuple[tuple[int, int], tuple[int, int]] | None:
    feasible = data["feasible"]
    states = [s for s in data["states"] if data["degree"][s] >= 2 and _corner_profile(s, feasible.shape)[1] <= 4]
    random.shuffle(states)
    for start in states:
        dist = _slide_bfs(feasible, start)
        if len(dist) < 8 or max(dist.values(), default=0) < 2:
            continue
        start_corner, _ = _corner_profile(start, feasible.shape)
        candidates = [
            state
            for state, d in dist.items()
            if state != start and d >= 2 and _corner_profile(state, feasible.shape)[0] != start_corner
        ]
        if not candidates:
            continue
        dest = max(
            candidates,
            key=lambda state: (
                min(4, _corner_profile(state, feasible.shape)[1]) == _corner_profile(state, feasible.shape)[1],
                int(dist[state]),
                abs(int(state[0]) - int(start[0])) + abs(int(state[1]) - int(start[1])),
            ),
        )
        return start, dest
    return None


def _pick_blocked_positions(data: dict) -> tuple[tuple[int, int], tuple[int, int]] | None:
    feasible = data["feasible"]
    f_labels = data["f_labels"]
    o_labels = data["o_labels"]
    states = [s for s in data["states"] if data["degree"][s] >= 2 and _corner_profile(s, feasible.shape)[1] <= 4]
    random.shuffle(states)
    for start in states:
        dist = _slide_bfs(feasible, start)
        if len(dist) < 8 or max(dist.values(), default=0) < 2:
            continue
        final = _best_blocked_stop(dist, start)
        start_f = int(f_labels[start])
        start_open = int(o_labels[start])
        start_corner, _ = _corner_profile(start, feasible.shape)
        candidates = []
        for dest in data["states"]:
            if int(f_labels[dest]) == start_f:
                continue
            if int(o_labels[dest]) != start_open:
                continue
            dest_corner, dest_corner_dist = _corner_profile(dest, feasible.shape)
            if dest_corner == start_corner or dest_corner_dist > 4:
                continue
            gap = abs(int(dest[0]) - int(final[0])) + abs(int(dest[1]) - int(final[1]))
            if 1 <= gap <= 8:
                candidates.append((gap, dest))
        if not candidates:
            continue
        candidates.sort(key=lambda item: (item[0], item[1][0], item[1][1]))
        return start, candidates[0][1]
    return None


def main(grid: np.ndarray) -> np.ndarray:
    """
    Generate four staircase walls from a teal anchor, then slide a 2x2 green block through the resulting corridor system.

    Input structure:
    - The background is black.
    - There is exactly one teal anchor pixel and 4-8 scattered gray stopper pixels on the four staircase rays.
    - One green 2x2 solid square is the mover, and one orange 2x2 solid square is the destination.

    Rule:
    - From the teal anchor, extend four gray staircase arms toward the four diagonal quadrants.
    - Each arm alternates 2 horizontal cells then 2 vertical cells, stopping before the first existing gray stopper or the grid boundary.
    - Treat the completed gray staircases and the teal anchor as walls; the green 2x2 block can only occupy 2x2 all-open positions.
    - The block slides in straight lines until its next step would hit a wall or leave the grid; if the orange 2x2 position is reachable, the green block replaces it, otherwise the green block moves to the farthest slide-reachable stop and its start becomes black.

    Key insight:
    - The corridor geometry is not explicit in the input: it appears only after expanding the anchor into four staircase walls, and that derived wall layout determines the 2x2 sliding graph.
    """
    g = np.array(grid, dtype=int, copy=True)
    rendered = _build_wall_grid(g)

    start = _find_square_top_left(g, MOVER)
    dest = _find_square_top_left(g, DEST)
    feasible, _ = _feasible_positions(rendered)
    assert bool(feasible[start]), "mover must start inside a valid 2x2 corridor position"

    dist = _slide_bfs(feasible, start)
    final = dest if dest in dist else _best_blocked_stop(dist, dest)

    out = rendered.copy()
    sr, sc = start
    fr, fc = final
    out[sr : sr + BLOCK_SIZE, sc : sc + BLOCK_SIZE] = BG
    out[fr : fr + BLOCK_SIZE, fc : fc + BLOCK_SIZE] = MOVER
    return out


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(17, 24)
    w = random.randint(17, 24)
    anchor = (random.randint(5, h - 6), random.randint(5, w - 6))

    arm_paths = {(dr, dc): _full_arm_path(anchor, dr, dc, (h, w)) for dr, dc in DIAG_DIRS}
    if min(len(path) for path in arm_paths.values()) < 8:
        return None

    stop_pixels: set[tuple[int, int]] = set()
    for path in arm_paths.values():
        full_step_cells = [idx for idx in range(3, len(path), 4)]
        if len(full_step_cells) < 2:
            return None
        primary_idx = random.choice(full_step_cells[1:])
        stop_pixels.add(path[primary_idx])
        later_steps = [idx for idx in full_step_cells if idx > primary_idx + 3]
        if later_steps and random.random() < 0.45:
            stop_pixels.add(path[random.choice(later_steps)])

    if not (4 <= len(stop_pixels) <= 8):
        return None

    base = np.zeros((h, w), dtype=int)
    base[anchor] = ANCHOR
    for r, c in stop_pixels:
        if abs(r - anchor[0]) + abs(c - anchor[1]) <= 1:
            return None
        base[r, c] = WALL

    data = _analyze_layout(base)
    if data is None:
        return None

    want_blocked = bool(random.getrandbits(1))
    picked = _pick_blocked_positions(data) if want_blocked else _pick_reachable_positions(data)
    if picked is None:
        picked = _pick_reachable_positions(data) if want_blocked else _pick_blocked_positions(data)
    if picked is None:
        return None

    start, dest = picked
    if start == dest:
        return None

    inp = base.copy()
    sr, sc = start
    dr, dc = dest
    inp[sr : sr + BLOCK_SIZE, sc : sc + BLOCK_SIZE] = MOVER
    inp[dr : dr + BLOCK_SIZE, dc : dc + BLOCK_SIZE] = DEST

    out = main(inp)
    if np.array_equal(out, inp):
        return None

    rendered = _build_wall_grid(inp)
    dist = _slide_bfs(_feasible_positions(rendered)[0], start)
    reached = dest in dist
    if want_blocked == reached:
        return None
    if int((rendered == WALL).sum()) - len(stop_pixels) < 12:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid staircase-corridor sample")
