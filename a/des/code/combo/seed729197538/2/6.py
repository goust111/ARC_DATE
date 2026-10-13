import heapq
import random
from collections import deque

import numpy as np


BG = 0
RING = 5
SEP = 9
USABLE_COLORS = (1, 2, 3, 4, 6, 7, 8)
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

MAX_GRID_SIDE = 30
MIN_PANEL_SIDE = 8
MAX_PANEL_SIDE = 12
GEN_ATTEMPTS = 400
PICK_ATTEMPTS = 200


def _neighbors4(r: int, c: int, h: int, w: int):
    for dr, dc in DIR4:
        rr, cc = r + dr, c + dc
        if 0 <= rr < h and 0 <= cc < w:
            yield rr, cc


def _bands(indices: np.ndarray) -> list[tuple[int, int]]:
    if indices.size == 0:
        return []
    bands: list[tuple[int, int]] = []
    start = prev = int(indices[0])
    for value in indices[1:]:
        value = int(value)
        if value != prev + 1:
            bands.append((start, prev))
            start = value
        prev = value
    bands.append((start, prev))
    return bands


def _panel_bands(grid: np.ndarray) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    row_bands = _bands(np.where(~np.all(grid == SEP, axis=1))[0])
    col_bands = _bands(np.where(~np.all(grid == SEP, axis=0))[0])
    return row_bands, col_bands


def _find_endpoints(mask: np.ndarray) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    h, w = mask.shape
    endpoints: list[tuple[tuple[int, int], tuple[int, int]]] = []
    for r, c in np.argwhere(mask):
        r = int(r)
        c = int(c)
        neighbors = [(rr, cc) for rr, cc in _neighbors4(r, c, h, w) if mask[rr, cc]]
        if len(neighbors) == 1:
            rr, cc = neighbors[0]
            endpoints.append(((r, c), (r - rr, c - cc)))
    return endpoints


def _on_forward_ray(cell: tuple[int, int], endpoint: tuple[int, int], direction: tuple[int, int]) -> bool:
    dr = int(cell[0] - endpoint[0])
    dc = int(cell[1] - endpoint[1])
    if direction[0] != 0:
        return dc == 0 and dr * int(direction[0]) >= 1
    return dr == 0 and dc * int(direction[1]) >= 1


def _completion_path(mask: np.ndarray, walkable: np.ndarray) -> list[tuple[int, int]]:
    endpoints = _find_endpoints(mask)
    assert len(endpoints) == 2, "expected exactly two ring endpoints"
    (e1, d1), (e2, d2) = endpoints
    start = (e1[0] + d1[0], e1[1] + d1[1])
    goal = (e2[0] + d2[0], e2[1] + d2[1])
    h, w = mask.shape

    for r, c in (start, goal):
        assert 0 <= r < h and 0 <= c < w, "completion path must stay in bounds"
        assert walkable[r, c], "completion path start/end must land on the black gap"

    if start == goal:
        return [start]

    def off_ray(cell: tuple[int, int]) -> int:
        return 0 if (_on_forward_ray(cell, e1, d1) or _on_forward_ray(cell, e2, d2)) else 1

    pq: list[tuple[int, int, int, int, int, int]] = [(0, 0, off_ray(start), start[0], start[1], -1)]
    best: dict[tuple[int, int, int], tuple[int, int, int]] = {(start[0], start[1], -1): (0, 0, off_ray(start))}
    parent: dict[tuple[int, int, int], tuple[int, int, int]] = {}
    goal_state: tuple[int, int, int] | None = None

    while pq:
        steps, turns, stray, r, c, prev_dir = heapq.heappop(pq)
        state = (r, c, prev_dir)
        if best.get(state) != (steps, turns, stray):
            continue
        if (r, c) == goal:
            goal_state = state
            break

        dir_order = sorted(
            range(len(DIR4)),
            key=lambda idx: abs(r + DIR4[idx][0] - goal[0]) + abs(c + DIR4[idx][1] - goal[1]),
        )
        for dir_idx in dir_order:
            dr, dc = DIR4[dir_idx]
            nr, nc = r + dr, c + dc
            if not (0 <= nr < h and 0 <= nc < w) or not walkable[nr, nc]:
                continue
            next_cost = (
                steps + 1,
                turns + int(prev_dir != -1 and prev_dir != dir_idx),
                stray + off_ray((nr, nc)),
            )
            next_state = (nr, nc, dir_idx)
            if next_cost < best.get(next_state, (10**9, 10**9, 10**9)):
                best[next_state] = next_cost
                parent[next_state] = state
                heapq.heappush(pq, (*next_cost, nr, nc, dir_idx))

    assert goal_state is not None, "failed to reconstruct the missing arc"

    path: list[tuple[int, int]] = []
    cur = goal_state
    while True:
        path.append((cur[0], cur[1]))
        if cur not in parent:
            break
        cur = parent[cur]
    path.reverse()

    assert path[0] == start and path[-1] == goal
    prev = e1
    seen: set[tuple[int, int]] = set()
    for cell in path:
        assert abs(cell[0] - prev[0]) + abs(cell[1] - prev[1]) == 1
        assert cell not in seen and walkable[cell]
        seen.add(cell)
        prev = cell
    assert abs(e2[0] - prev[0]) + abs(e2[1] - prev[1]) == 1
    return path


def _outside_mask(closed_ring: np.ndarray) -> np.ndarray:
    h, w = closed_ring.shape
    outside = np.zeros((h, w), dtype=bool)
    q = deque()
    for r in range(h):
        for c in (0, w - 1):
            if not closed_ring[r, c] and not outside[r, c]:
                outside[r, c] = True
                q.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if not closed_ring[r, c] and not outside[r, c]:
                outside[r, c] = True
                q.append((r, c))

    while q:
        r, c = q.popleft()
        for rr, cc in _neighbors4(r, c, h, w):
            if closed_ring[rr, cc] or outside[rr, cc]:
                continue
            outside[rr, cc] = True
            q.append((rr, cc))
    return outside


def _distance_from_ring(closed_ring: np.ndarray) -> np.ndarray:
    h, w = closed_ring.shape
    dist = np.full((h, w), -1, dtype=int)
    q = deque()
    for r, c in np.argwhere(closed_ring):
        r = int(r)
        c = int(c)
        dist[r, c] = 0
        q.append((r, c))

    while q:
        r, c = q.popleft()
        for rr, cc in _neighbors4(r, c, h, w):
            if dist[rr, cc] != -1:
                continue
            dist[rr, cc] = dist[r, c] + 1
            q.append((rr, cc))
    return dist


def _solve_panel(panel: np.ndarray) -> np.ndarray:
    ring_mask = panel == RING
    path = _completion_path(ring_mask, panel == BG)
    closed_ring = ring_mask.copy()
    for r, c in path:
        closed_ring[r, c] = True

    outside = _outside_mask(closed_ring)
    inside = (~closed_ring) & (~outside)
    dist = _distance_from_ring(closed_ring)

    fragments = []
    for r, c in np.argwhere((panel != BG) & (panel != RING)):
        r = int(r)
        c = int(c)
        if closed_ring[r, c]:
            continue
        fragments.append((r, c, int(panel[r, c]), bool(inside[r, c])))

    inside_fragments = [frag for frag in fragments if frag[3]]
    outside_fragments = [frag for frag in fragments if not frag[3]]
    chosen = inside_fragments if len(inside_fragments) > len(outside_fragments) else outside_fragments
    assert chosen, "one side must contribute the output color"
    color = min(chosen, key=lambda frag: (int(dist[frag[0], frag[1]]), frag[0], frag[1]))[2]

    out = np.full(panel.shape, BG, dtype=int)
    out[closed_ring] = color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    row_bands, col_bands = _panel_bands(g)
    assert row_bands and col_bands, "expected at least one panel"
    for r0, r1 in row_bands:
        for c0, c1 in col_bands:
            out[r0 : r1 + 1, c0 : c1 + 1] = _solve_panel(g[r0 : r1 + 1, c0 : c1 + 1])
    return out


def _layout_options() -> list[tuple[int, int]]:
    return [(1, 2), (2, 1), (1, 3), (3, 1), (2, 2), (2, 3), (3, 2), (3, 3)]


def _rectangle_ring(h: int, w: int) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    outer_h = random.randint(6, h - 2)
    outer_w = random.randint(6, w - 2)
    r0 = random.randint(1, h - outer_h - 1)
    c0 = random.randint(1, w - outer_w - 1)
    r1 = r0 + outer_h - 1
    c1 = c0 + outer_w - 1

    mask = np.zeros((h, w), dtype=bool)
    mask[r0, c0 : c1 + 1] = True
    mask[r1, c0 : c1 + 1] = True
    mask[r0 : r1 + 1, c0] = True
    mask[r0 : r1 + 1, c1] = True
    return mask, (r0, c0, r1, c1)


def _gap_cells(bbox: tuple[int, int, int, int]) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    height = r1 - r0 + 1
    width = c1 - c0 + 1
    gap_type = random.choice(("straight", "corner"))
    cells: set[tuple[int, int]] = set()

    if gap_type == "straight":
        side = random.choice(("top", "bottom", "left", "right"))
        if side in ("top", "bottom"):
            max_len = min(3, width - 4)
            gap_len = random.randint(1, max_len)
            offset = random.randint(2, width - gap_len - 2)
            row = r0 if side == "top" else r1
            for dc in range(gap_len):
                cells.add((row, c0 + offset + dc))
        else:
            max_len = min(3, height - 4)
            gap_len = random.randint(1, max_len)
            offset = random.randint(2, height - gap_len - 2)
            col = c0 if side == "left" else c1
            for dr in range(gap_len):
                cells.add((r0 + offset + dr, col))
        return cells

    corner = random.choice(("tl", "tr", "bl", "br"))
    horiz = random.randint(1, min(3, width - 3))
    vert = random.randint(1, min(3, height - 3))

    if corner == "tl":
        for dc in range(horiz + 1):
            cells.add((r0, c0 + dc))
        for dr in range(vert + 1):
            cells.add((r0 + dr, c0))
    elif corner == "tr":
        for dc in range(horiz + 1):
            cells.add((r0, c1 - dc))
        for dr in range(vert + 1):
            cells.add((r0 + dr, c1))
    elif corner == "bl":
        for dc in range(horiz + 1):
            cells.add((r1, c0 + dc))
        for dr in range(vert + 1):
            cells.add((r1 - dr, c0))
    else:
        for dc in range(horiz + 1):
            cells.add((r1, c1 - dc))
        for dr in range(vert + 1):
            cells.add((r1 - dr, c1))
    return cells


def _manhattan(a: tuple[int, int], b: tuple[int, int]) -> int:
    return abs(int(a[0] - b[0])) + abs(int(a[1] - b[1]))


def _pick_positions(
    candidates: list[tuple[int, int]],
    count: int,
    dist_map: np.ndarray,
    *,
    require_unique_min: bool,
) -> list[tuple[int, int]] | None:
    if count == 0:
        return []
    if len(candidates) < count:
        return None
    for _ in range(PICK_ATTEMPTS):
        picked = random.sample(candidates, count)
        if any(_manhattan(a, b) <= 1 for i, a in enumerate(picked) for b in picked[i + 1 :]):
            continue
        dists = [int(dist_map[r, c]) for r, c in picked]
        if require_unique_min and dists.count(min(dists)) != 1:
            continue
        return picked
    return None


def _sample_colors(count: int, *, distinct: bool) -> list[int]:
    if count == 0:
        return []
    if distinct:
        return random.sample(USABLE_COLORS, count)
    palette = random.sample(USABLE_COLORS, random.randint(1, min(3, len(USABLE_COLORS))))
    return [random.choice(palette) for _ in range(count)]


def _make_panel(h: int, w: int) -> tuple[np.ndarray, np.ndarray] | None:
    full_ring, bbox = _rectangle_ring(h, w)
    missing = _gap_cells(bbox)
    open_ring = full_ring.copy()
    for r, c in missing:
        open_ring[r, c] = False

    if len(_find_endpoints(open_ring)) != 2:
        return None

    outside = _outside_mask(full_ring)
    inside = (~full_ring) & (~outside)
    dist = _distance_from_ring(full_ring)

    inside_candidates = [(int(r), int(c)) for r, c in np.argwhere(inside)]
    outside_candidates = [(int(r), int(c)) for r, c in np.argwhere((~full_ring) & (~inside))]

    winner_side = random.choice(("inside", "outside"))
    if winner_side == "inside":
        count_pair = random.choice([(1, 0), (2, 0), (2, 1), (3, 0)])
    else:
        count_pair = random.choice([(0, 1), (0, 2), (1, 1), (0, 3), (1, 2)])
    inside_count, outside_count = count_pair

    inside_positions = _pick_positions(
        inside_candidates,
        inside_count,
        dist,
        require_unique_min=(winner_side == "inside" and inside_count > 1),
    )
    outside_positions = _pick_positions(
        outside_candidates,
        outside_count,
        dist,
        require_unique_min=(winner_side == "outside" and outside_count > 1),
    )
    if inside_positions is None or outside_positions is None:
        return None

    inside_colors = _sample_colors(inside_count, distinct=inside_count > 1)
    outside_colors = _sample_colors(outside_count, distinct=outside_count > 1)

    panel = np.full((h, w), BG, dtype=int)
    panel[open_ring] = RING
    for (r, c), color in zip(inside_positions, inside_colors):
        panel[r, c] = color
    for (r, c), color in zip(outside_positions, outside_colors):
        panel[r, c] = color

    chosen_positions = inside_positions if winner_side == "inside" else outside_positions
    chosen_colors = inside_colors if winner_side == "inside" else outside_colors
    chosen_idx = min(
        range(len(chosen_positions)),
        key=lambda idx: (int(dist[chosen_positions[idx][0], chosen_positions[idx][1]]), chosen_positions[idx][0], chosen_positions[idx][1]),
    )
    target_color = chosen_colors[chosen_idx]

    expected = np.full((h, w), BG, dtype=int)
    expected[full_ring] = target_color

    if not np.array_equal(_solve_panel(panel), expected):
        return None
    return panel, expected


def _gen_attempt() -> np.ndarray | None:
    rows, cols = random.choice(_layout_options())
    max_panel_h = min(MAX_PANEL_SIDE, (MAX_GRID_SIDE - (rows - 1)) // rows)
    max_panel_w = min(MAX_PANEL_SIDE, (MAX_GRID_SIDE - (cols - 1)) // cols)
    if max_panel_h < MIN_PANEL_SIDE or max_panel_w < MIN_PANEL_SIDE:
        return None

    panel_h = random.randint(MIN_PANEL_SIDE, max_panel_h)
    panel_w = random.randint(MIN_PANEL_SIDE, max_panel_w)
    grid_h = rows * panel_h + (rows - 1)
    grid_w = cols * panel_w + (cols - 1)

    grid = np.full((grid_h, grid_w), BG, dtype=int)
    expected = grid.copy()
    for split_r in range(panel_h, grid_h, panel_h + 1):
        grid[split_r, :] = SEP
        expected[split_r, :] = SEP
    for split_c in range(panel_w, grid_w, panel_w + 1):
        grid[:, split_c] = SEP
        expected[:, split_c] = SEP

    for pr in range(rows):
        for pc in range(cols):
            made = _make_panel(panel_h, panel_w)
            if made is None:
                return None
            panel, solved = made
            r0 = pr * (panel_h + 1)
            c0 = pc * (panel_w + 1)
            grid[r0 : r0 + panel_h, c0 : c0 + panel_w] = panel
            expected[r0 : r0 + panel_h, c0 : c0 + panel_w] = solved

    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
