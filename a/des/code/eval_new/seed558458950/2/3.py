import heapq
import random
from collections import defaultdict, deque
from typing import Dict, Iterable, List, Set, Tuple

import numpy as np

BG_COLOR_A = 0
BG_COLOR_B = 1
CHANNEL_COLOR = 8
TURN_COLOR = 9
RED = 2
GREEN = 3
YELLOW = 4
MAGENTA = 5
LIGHT_COLORS = [RED, GREEN, YELLOW, MAGENTA]

DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
LEFT_TURN = {
    (-1, 0): (0, -1),
    (0, -1): (1, 0),
    (1, 0): (0, 1),
    (0, 1): (-1, 0),
}
RIGHT_TURN = {
    (-1, 0): (0, 1),
    (0, 1): (1, 0),
    (1, 0): (0, -1),
    (0, -1): (-1, 0),
}
NEXT_COLOR = {
    RED: GREEN,
    GREEN: YELLOW,
    YELLOW: MAGENTA,
    MAGENTA: RED,
}


def _in_bounds(r: int, c: int, h: int, w: int) -> bool:
    return 0 <= r < h and 0 <= c < w


def _build_checkerboard(h: int, w: int) -> np.ndarray:
    parity = random.randint(0, 1)
    grid = np.empty((h, w), dtype=int)
    for r in range(h):
        base = (r + parity) % 2
        for c in range(w):
            grid[r, c] = BG_COLOR_A if (c + base) % 2 == 0 else BG_COLOR_B
    return grid


def _segment_cells_valid(
    cells: List[Tuple[int, int]],
    axis: str,
    graph: Dict[Tuple[int, int], Set[Tuple[int, int]]],
    h: int,
    w: int,
) -> bool:
    if len(cells) < 2:
        return False
    for i in range(len(cells) - 1):
        r0, c0 = cells[i]
        r1, c1 = cells[i + 1]
        if abs(r0 - r1) + abs(c0 - c1) != 1:
            return False
        if not (0 <= r1 < h and 0 <= c1 < w):
            return False

    axis_is_horizontal = axis == "h"
    for i, (r, c) in enumerate(cells):
        prev_neighbors = graph.get((r, c), set())
        at_end = i == 0 or i == len(cells) - 1

        if at_end:
            if prev_neighbors:
                return False
        else:
            if len(prev_neighbors) == 2:
                drs = {nr - r for nr, nc in prev_neighbors}
                dcs = {nc - c for nr, nc in prev_neighbors}
                if axis_is_horizontal:
                    if drs != {-1, 1} or dcs != {0}:
                        return False
                else:
                    if drs != {0} or dcs != {-1, 1}:
                        return False
            elif prev_neighbors:
                return False

        if i > 0:
            prev = cells[i - 1]
            if prev in graph.get((r, c), set()):
                return False

    return True


def _add_segment(
    cells: List[Tuple[int, int]],
    axis: str,
    graph: Dict[Tuple[int, int], Set[Tuple[int, int]]],
    h: int,
    w: int,
) -> bool:
    if not _segment_cells_valid(cells, axis, graph, h, w):
        return False
    for (r0, c0), (r1, c1) in zip(cells, cells[1:]):
        graph.setdefault((r0, c0), set()).add((r1, c1))
        graph.setdefault((r1, c1), set()).add((r0, c0))
    return True


def _random_straight_segment(h: int, w: int, axis: str) -> List[Tuple[int, int]]:
    if axis == "h":
        max_len = max(5, min(w, w // 2 + 2))
        if max_len < 5:
            return []
        length = random.randint(5, max_len)
        if length > w:
            length = w
        r = random.randint(0, h - 1)
        c0 = random.randint(0, w - length)
        return [(r, c0 + k) for k in range(length)]

    max_len = max(5, min(h, h // 2 + 2))
    if max_len < 5:
        return []
    length = random.randint(5, max_len)
    if length > h:
        length = h
    c = random.randint(0, w - 1)
    r0 = random.randint(0, h - length)
    return [(r0 + k, c) for k in range(length)]


def _collect_components(
    graph: Dict[Tuple[int, int], Set[Tuple[int, int]]],
) -> Dict[Tuple[int, int], int]:
    comp_id = {}
    cid = 0
    for cell in graph:
        if cell in comp_id:
            continue
        q = deque([cell])
        comp_id[cell] = cid
        while q:
            x = q.popleft()
            for nb in graph[x]:
                if nb not in comp_id:
                    comp_id[nb] = cid
                    q.append(nb)
        cid += 1
    return comp_id


def _resolve_ties(out: np.ndarray, arrivals: Dict[Tuple[int, int], Dict]) -> np.ndarray:
    tie_cells = {p for p, d in arrivals.items() if len(d["events"]) > 1}
    if not tie_cells:
        return out

    for pos, data in arrivals.items():
        unique_events = []
        seen = set()
        for ev in data["events"]:
            if ev not in seen:
                unique_events.append(ev)
                seen.add(ev)
        data["events"] = unique_events

    tie_cells = {p for p, d in arrivals.items() if len(d["events"]) > 1}
    tie_cells_sorted = sorted(tie_cells)
    visited = set()

    for start in tie_cells_sorted:
        if start in visited:
            continue

        comp = []
        q = deque([start])
        visited.add(start)
        while q:
            r, c = q.popleft()
            comp.append((r, c))
            for dr, dc in DIRS:
                nb = (r + dr, c + dc)
                if nb in tie_cells and nb not in visited:
                    visited.add(nb)
                    q.append(nb)

        if not comp:
            continue

        # seed ids are based on source id, color and incoming direction
        seed_by_key: Dict[Tuple[int, int, int, int], int] = {}
        seed_color: Dict[int, int] = {}
        seeds_by_cell: Dict[Tuple[int, int], List[int]] = defaultdict(list)

        for cell in sorted(comp):
            for source_id, color, dr, dc in arrivals[cell]["events"]:
                key = (source_id, color, dr, dc)
                if key not in seed_by_key:
                    sid = len(seed_by_key)
                    seed_by_key[key] = sid
                    seed_color[sid] = color
                else:
                    sid = seed_by_key[key]
                if sid not in seeds_by_cell[cell]:
                    seeds_by_cell[cell].append(sid)

        dist: Dict[Tuple[int, int], int] = {}
        owners: Dict[Tuple[int, int], Set[int]] = {}
        frontier = deque()

        for cell, sids in seeds_by_cell.items():
            dist[cell] = 0
            owner_set = set(sids)
            owners[cell] = owner_set
            frontier.append(cell)

        comp_set = set(comp)
        while frontier:
            r, c = frontier.popleft()
            d = dist[(r, c)]
            cell_owners = owners[(r, c)]

            for dr, dc in DIRS:
                nb = (r + dr, c + dc)
                if nb not in comp_set:
                    continue

                nd = d + 1
                if nb not in dist:
                    dist[nb] = nd
                    owners[nb] = set(cell_owners)
                    frontier.append(nb)
                    continue

                if nd < dist[nb]:
                    dist[nb] = nd
                    owners[nb] = set(cell_owners)
                    frontier.append(nb)
                elif nd == dist[nb]:
                    before = len(owners[nb])
                    owners[nb].update(cell_owners)
                    if len(owners[nb]) > before:
                        frontier.append(nb)

        for cell in comp:
            choices = owners[cell]
            sid = min(choices)
            out[cell] = seed_color[sid]

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Propagate signal colors through a channel network and split at color-shifting intersections.

    Input structure:
    - Background is a checkerboard made of two colors.
    - Single-cell grey/turn channels form one-cell-wide orthogonal paths and crosses.
    - Source pixels are red, green, yellow, or magenta on background cells touching a channel endpoint.
    - Some cross cells are orange (turning junctions); all other cross/through cells are channel grey.

    Rule:
    - Each source starts at its neighboring channel cell and moves with the line direction through channel cells.
    - At normal intersections (grey) the signal keeps going straight; at turning intersections (orange) it splits left and right.
    - Every moved signal recolors visited channel cells with its current color; at orange it recolors that orange cell with pre-split color and continues with next color.
    - Color cycle at every turn is red->green->yellow->magenta->red.
    - If two signals reach a cell with different intersection counts, the smaller count wins; equal counts are split by nearest-front partition from their incoming sides.

    Key insight:
    - This is signal expansion on a constrained orthogonal graph where channel cell type controls straight/turn behavior and tie competition is resolved by arrival-time layers and side-based splits."""

    grid_in = np.array(grid, dtype=int)
    h, w = grid_in.shape
    out = grid_in.copy()

    bg_colors = {BG_COLOR_A, BG_COLOR_B}
    channel_mask = (grid_in == CHANNEL_COLOR) | (grid_in == TURN_COLOR)
    if not np.any(channel_mask):
        return out

    light_positions = np.argwhere(np.isin(grid_in, LIGHT_COLORS))
    sources: List[Tuple[int, int, int, int, int, int]] = []

    for sr, sc in light_positions:
        color = int(grid_in[sr, sc])
        for dr, dc in DIRS:
            nr, nc = sr + dr, sc + dc
            if _in_bounds(nr, nc, h, w) and channel_mask[nr, nc]:
                sources.append((nr, nc, dr, dc, color, len(sources)))

    if not sources:
        return out

    neighbors: Dict[Tuple[int, int], Set[Tuple[int, int]]] = {}
    is_turn: Dict[Tuple[int, int], bool] = {}
    is_intersection: Dict[Tuple[int, int], bool] = {}

    for r in range(h):
        for c in range(w):
            if not channel_mask[r, c]:
                continue
            neigh = set()
            for dr, dc in DIRS:
                nr, nc = r + dr, c + dc
                if _in_bounds(nr, nc, h, w) and channel_mask[nr, nc]:
                    neigh.add((nr, nc))
            neighbors[(r, c)] = neigh
            is_turn[(r, c)] = grid_in[r, c] == TURN_COLOR
            is_intersection[(r, c)] = is_turn[(r, c)] or len(neigh) == 4

    arrivals: Dict[Tuple[int, int], Dict] = {}
    best_state: Dict[Tuple[int, int, int, int, int], int] = {}

    pq: List[Tuple[int, int, int, int, int, int, int]] = []
    for r, c, dr, dc, color, source_id in sources:
        d0 = 1 if is_intersection.get((r, c), False) else 0
        heapq.heappush(pq, (d0, source_id, r, c, dr, dc, color))

    while pq:
        dist, source_id, r, c, dr, dc, color = heapq.heappop(pq)
        key = (r, c, dr, dc, color)
        prev = best_state.get(key)
        if prev is not None and dist > prev:
            continue
        if prev is None:
            best_state[key] = dist

        rec = (source_id, color, dr, dc)
        cell = (r, c)
        if cell not in arrivals:
            arrivals[cell] = {"dist": dist, "events": [rec]}
        else:
            if dist < arrivals[cell]["dist"]:
                arrivals[cell] = {"dist": dist, "events": [rec]}
            elif dist == arrivals[cell]["dist"]:
                if rec not in arrivals[cell]["events"]:
                    arrivals[cell]["events"].append(rec)
            else:
                continue

        if is_turn.get((r, c), False):
            branch_color = NEXT_COLOR[color]
            for ndr, ndc in (LEFT_TURN[(dr, dc)], RIGHT_TURN[(dr, dc)]):
                nr, nc = r + ndr, c + ndc
                if not _in_bounds(nr, nc, h, w) or not channel_mask[nr, nc]:
                    continue
                nd = dist + (1 if is_intersection.get((nr, nc), False) else 0)
                next_key = (nr, nc, ndr, ndc, branch_color)
                if best_state.get(next_key, 10**9) <= nd:
                    continue
                heapq.heappush(pq, (nd, source_id, nr, nc, ndr, ndc, branch_color))
        else:
            nr, nc = r + dr, c + dc
            if not _in_bounds(nr, nc, h, w) or not channel_mask[nr, nc]:
                continue
            nd = dist + (1 if is_intersection.get((nr, nc), False) else 0)
            next_key = (nr, nc, dr, dc, color)
            if best_state.get(next_key, 10**9) <= nd:
                continue
            heapq.heappush(pq, (nd, source_id, nr, nc, dr, dc, color))

    # Apply unique arrivals first.
    for (r, c), data in arrivals.items():
        events = data["events"]
        if len(events) == 1:
            _, color, _, _ = events[0]
            out[r, c] = color

    return _resolve_ties(out, arrivals)


def generate_input() -> np.ndarray:
    while True:
        h = random.randint(16, 28)
        w = random.randint(16, 28)
        graph: Dict[Tuple[int, int], Set[Tuple[int, int]]] = {}

        segment_attempts = random.randint(2, 4)
        for _ in range(segment_attempts):
            axis = random.choice(["h", "v"])
            placed = False
            for _ in range(80):
                cells = _random_straight_segment(h, w, axis)
                if not cells:
                    break
                if _add_segment(cells, axis, graph, h, w):
                    placed = True
                    break

            if not placed:
                continue

        # Add one dedicated straight lane for strong tie scenarios.
        duel_endpoints: List[Tuple[int, int]] = []
        if random.random() < 0.75:
            axis = random.choice(["h", "v"]);
            placed = False
            for _ in range(60):
                cells = _random_straight_segment(h, w, axis)
                if not cells:
                    break
                if len(cells) < 9:
                    continue
                if any(cell in graph for cell in cells):
                    continue
                if _segment_cells_valid(cells, axis, graph, h, w):
                    _add_segment(cells, axis, graph, h, w)
                    duel_endpoints = [cells[0], cells[-1]]
                    placed = True
                    break
            if not placed:
                duel_endpoints = []

        # Validation guards.
        if len(graph) < 18:
            continue

        degs = {cell: len(graph[cell]) for cell in graph}
        if any(d == 3 for d in degs.values()) or all(d == 0 for d in degs.values()):
            continue

        endpoints = [cell for cell, d in degs.items() if d == 1]
        if len(endpoints) < 2:
            continue

        crosses = [cell for cell, d in degs.items() if d == 4]

        base = _build_checkerboard(h, w)
        channel_cells = sorted(graph.keys())
        for r, c in channel_cells:
            base[r, c] = CHANNEL_COLOR

        cross_set = set(crosses)
        random.shuffle(crosses)
        if crosses:
            if random.random() < 0.8:
                orange_count = random.randint(1, len(crosses))
            else:
                orange_count = 0
            for cell in crosses[:orange_count]:
                base[cell] = TURN_COLOR

        comp_id = _collect_components(graph)
        cross_comps = {
            comp_id[cell]
            for cell in crosses
            if cell in comp_id
        }

        endpoint_infos = []
        background = {BG_COLOR_A, BG_COLOR_B}
        for idx, (r, c) in enumerate(endpoints):
            opts = []
            for dr, dc in DIRS:
                nr, nc = r + dr, c + dc
                if _in_bounds(nr, nc, h, w) and base[nr, nc] in background:
                    opts.append((nr, nc))
            if opts:
                endpoint_infos.append((idx, (r, c), opts))

        if len(endpoint_infos) < 1:
            continue

        if len(endpoint_infos) < 1:
            continue

        n_sources = random.randint(1, min(4, len(endpoint_infos)))
        selected_idx: Set[int] = set()

        # Ensure some branch-heavy behavior.
        if duel_endpoints:
            duel_idx = {
                k for k, (_idx, ep, _opts) in enumerate(endpoint_infos) if ep in duel_endpoints
            }
            if random.random() < 0.6:
                selected_idx.update(duel_idx)

        if cross_comps and random.random() < 0.75:
            eligible = {
                k
                for k, (_, ep, _) in enumerate(endpoint_infos)
                if comp_id.get(ep, -1) in cross_comps
            }
            if eligible:
                selected_idx.add(random.choice(list(eligible)))

        n_sources = max(n_sources, len(selected_idx))
        while len(selected_idx) < n_sources:
            selected_idx.add(random.randrange(len(endpoint_infos)))

        chosen = list(selected_idx)
        random.shuffle(chosen)
        colors = LIGHT_COLORS.copy()
        random.shuffle(colors)

        used = set()
        placed_ok = 0
        for i, sel in enumerate(chosen):
            _, opts = endpoint_infos[sel][0], endpoint_infos[sel][2]
            color = colors[i % len(colors)]
            random.shuffle(opts)
            pos = None
            for candidate in opts:
                if candidate not in used:
                    pos = candidate
                    break
            if pos is None:
                break
            r, c = pos
            base[r, c] = color
            used.add(pos)
            placed_ok += 1

        if placed_ok == 0:
            continue

        return base
