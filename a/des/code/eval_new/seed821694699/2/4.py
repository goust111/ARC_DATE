"""Self-contained ARC-style solution and generator for the gate diffusion puzzle."""

import numpy as np
from collections import deque
from typing import Deque, Dict, List, Tuple

BLACK = 0
MAROON = 9
GRAY = 5
RED = 2


def _inside(h: int, w: int, r: int, c: int) -> bool:
    """Check whether a coordinate is inside a grid."""
    return 0 <= r < h and 0 <= c < w


def _extract_transform(grid: np.ndarray) -> Dict[int, int]:
    """
    Extract the 2xN legend from the bottom two rows.
    Any pair of non-special colors in the same column becomes one map entry.
    """
    h, w = grid.shape
    if h < 2:
        return {}
    src_row = h - 2
    dst_row = h - 1
    transform: Dict[int, int] = {}
    for c in range(w):
        src = int(grid[src_row, c])
        dst = int(grid[dst_row, c])
        if src in (BLACK, GRAY, MAROON) or dst in (BLACK, GRAY, MAROON):
            continue
        transform[src] = dst
    return transform


def _component_ids(grid: np.ndarray) -> Tuple[np.ndarray, List[List[Tuple[int, int]]], List[bool], List[bool]]:
    """
    Label black/red components in the grid.
    Returns component ids, cell lists, touches-border flags, and has-red flags.
    """
    h, w = grid.shape
    passable = (grid == BLACK) | (grid == RED)
    comp = -np.ones((h, w), dtype=np.int32)
    cells: List[List[Tuple[int, int]]] = []
    touches_border: List[bool] = []
    has_red: List[bool] = []

    comp_id = 0
    for r in range(h):
        for c in range(w):
            if not passable[r, c] or comp[r, c] != -1:
                continue
            q: Deque[Tuple[int, int]] = deque([(r, c)])
            comp[r, c] = comp_id
            cur_cells: List[Tuple[int, int]] = []
            tb = False
            red_present = False

            while q:
                cr, cc = q.popleft()
                cur_cells.append((cr, cc))
                if grid[cr, cc] == RED:
                    red_present = True
                if cr == 0 or cr == h - 1 or cc == 0 or cc == w - 1:
                    tb = True
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if not _inside(h, w, nr, nc):
                        continue
                    if comp[nr, nc] != -1:
                        continue
                    if not passable[nr, nc]:
                        continue
                    comp[nr, nc] = comp_id
                    q.append((nr, nc))

            cells.append(cur_cells)
            touches_border.append(tb)
            has_red.append(red_present)
            comp_id += 1

    return comp, cells, touches_border, has_red


def _color_at_distance(transform: Dict[int, int], steps: int, start: int) -> int | None:
    """Apply one-step mapping `steps` times; return None if chain stops."""
    color = start
    for _ in range(steps):
        if color not in transform:
            return None
        color = transform[color]
    return color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Problem:
        Expand a signal from the red room through maroon doors, by room graph distance.
    Input:
        np.ndarray with ARC integer colors: black background, gray walls, maroon doors, and two-row legend bottom-left.
    Procedure:
        1) find black/red room components not touching border, 2) build room adjacency via maroon doors,
        3) BFS from the room containing the red source, 4) fill each reached room by iterative legend mapping.
    Return:
        A grid with only room interiors recolored; walls and maroon doors remain unchanged.
    """

    g = np.array(grid, dtype=int, copy=True)
    h, w = g.shape
    transform = _extract_transform(g)

    comp, comp_cells, comp_touches_border, comp_has_red = _component_ids(g)
    comp_to_room: Dict[int, int] = {}
    room_cells: List[List[Tuple[int, int]]] = []
    room_has_red: List[bool] = []

    for idx, cells in enumerate(comp_cells):
        if comp_touches_border[idx]:
            continue
        comp_to_room[idx] = len(room_cells)
        room_cells.append(cells)
        room_has_red.append(comp_has_red[idx])

    room_n = len(room_cells)
    if room_n == 0:
        return g

    source_room = None
    for rid, red_present in enumerate(room_has_red):
        if red_present:
            source_room = rid
            break

    if source_room is None:
        return g

    adj: List[set[int]] = [set() for _ in range(room_n)]
    for r in range(h):
        for c in range(w):
            if g[r, c] != MAROON:
                continue
            neighbors = set()
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = r + dr, c + dc
                if not _inside(h, w, nr, nc):
                    continue
                cid = int(comp[nr, nc])
                if cid in comp_to_room:
                    neighbors.add(comp_to_room[cid])
            if len(neighbors) < 2:
                continue
            nlist = list(neighbors)
            for i in range(len(nlist)):
                for j in range(i + 1, len(nlist)):
                    a, b = nlist[i], nlist[j]
                    adj[a].add(b)
                    adj[b].add(a)

    room_dist = [-1] * room_n
    room_dist[source_room] = 0
    q: Deque[int] = deque([source_room])
    while q:
        cur = q.popleft()
        for nxt in adj[cur]:
            if room_dist[nxt] != -1:
                continue
            room_dist[nxt] = room_dist[cur] + 1
            q.append(nxt)

    out = g.copy()
    for rid, dist in enumerate(room_dist):
        if dist < 0:
            continue
        color = RED if dist == 0 else _color_at_distance(transform, dist, RED)
        if color is None:
            continue
        for rr, cc in room_cells[rid]:
            if out[rr, cc] in (BLACK, RED):
                out[rr, cc] = color

    return out


def generate_input() -> np.ndarray:
    """
    Generate a random valid instance (size <= 30) with:
    - black room cells, gray walls, maroon single-cell doors,
    - exactly one red source,
    - two-row transformation legend at the bottom-left.
    """
    rng = np.random.default_rng()

    n_rows = int(rng.integers(2, 4))
    n_cols = int(rng.integers(2, 4))
    row_h = [int(rng.integers(3, 7)) for _ in range(n_rows)]
    col_w = [int(rng.integers(3, 7)) for _ in range(n_cols)]

    row_edges = [0]
    for i, v in enumerate(row_h):
        row_edges.append(row_edges[-1] + v + 1)
    col_edges = [0]
    for wv in col_w:
        col_edges.append(col_edges[-1] + wv + 1)

    room_h = row_edges[-1] + 1
    room_w = col_edges[-1] + 1
    chain_len = int(rng.integers(2, 6))

    width = max(room_w, chain_len + 2)
    height = room_h + 2

    if width > 30 or height > 30:
        width = min(30, width)
        height = min(30, height)

    g = np.full((height, width), BLACK, dtype=int)

    # Draw room boundaries (gray walls).
    for r in range(room_h):
        for c in range(room_w):
            if r in row_edges or c in col_edges:
                g[r, c] = GRAY

    # Place exactly one maroon door on each shared internal vertical boundary.
    for jr in range(n_rows):
        y0, y1 = row_edges[jr] + 1, row_edges[jr + 1] - 1
        for k in range(1, n_cols):
            cx = col_edges[k]
            if width <= cx or y0 > y1:
                continue
            dy = int(rng.integers(y0, y1 + 1))
            g[dy, cx] = MAROON

    # Place exactly one maroon door on each shared internal horizontal boundary.
    for jr in range(1, n_rows):
        rx = row_edges[jr]
        for kc in range(n_cols):
            x0, x1 = col_edges[kc] + 1, col_edges[kc + 1] - 1
            if x0 > x1 or x0 < 0 or x1 >= width:
                continue
            dx = int(rng.integers(x0, x1 + 1))
            g[rx, dx] = MAROON

    # Put a single source in a random room.
    sr = int(rng.integers(0, n_rows))
    sc = int(rng.integers(0, n_cols))
    sy0, sy1 = row_edges[sr] + 1, row_edges[sr + 1] - 1
    sx0, sx1 = col_edges[sc] + 1, col_edges[sc + 1] - 1
    rr = int(rng.integers(sy0, sy1 + 1))
    cc = int(rng.integers(sx0, sx1 + 1))
    g[rr, cc] = RED

    # Construct a finite color chain in the legend.
    candidates = [1, 3, 4, 6, 7, 8]
    chain_len = max(2, chain_len)
    chain_len = min(chain_len, len(candidates) + 1)

    first = [RED]
    rest = list(rng.choice(candidates, size=chain_len - 1, replace=False))
    first.extend(int(x) for x in rest)

    # Ensure chain behavior: first L-1 entries map forward, last maps to a random color.
    second = []
    for i in range(chain_len - 1):
        second.append(int(first[i + 1]))
    second.append(int(rng.choice([x for x in candidates if x not in [RED]])))

    leg_r0 = height - 2
    leg_r1 = height - 1
    for i in range(chain_len):
        c = int(first[i])
        d = int(second[i])
        g[leg_r0, i] = c
        g[leg_r1, i] = d

    return g


if __name__ == "__main__":
    # Minimal smoke test when run directly.
    inp = generate_input()
    out = main(inp)
    print(inp.shape, out.shape, int((inp != out).sum()))
