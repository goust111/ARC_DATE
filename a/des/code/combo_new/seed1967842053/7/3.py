import random
from collections import deque

import numpy as np


GREY = 5


def _neighbors(r, c, h, w):
    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        nr, nc = r + dr, c + dc
        if 0 <= nr < h and 0 <= nc < w:
            yield nr, nc


def _background_color(grid):
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _components_of_grey(grid):
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    comp_id = {}
    for r in range(h):
        for c in range(w):
            if grid[r, c] != GREY or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp = []
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                comp_id[(cr, cc)] = len(comps)
                for nr, nc in _neighbors(cr, cc, h, w):
                    if grid[nr, nc] == GREY and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps, comp_id


def main(grid: np.ndarray) -> np.ndarray:
    """Color pipe segments selected by matching endpoint markers, then remove valves.

    Input structure:
    A mostly background grid contains a one-cell-wide orthogonal grey pipe
    network.  Some pipe cells are replaced by distinct colored valve pixels,
    and some true dead-end pipe tips have a one-cell colored marker just outside
    the pipe on adjacent background.  Marker colors match valve colors.

    Rule:
    Treat colored valve pixels as cuts in the grey pipe and flood-fill the grey
    pipe components on each side without crossing any valve.  For each valve,
    any adjacent grey component that has a dead-end marker of the same color is
    recolored to that valve color.  Valve pixels themselves become background,
    markers stay in place, and unclaimed grey pipe remains grey.

    Key insight:
    A valve is both the local divider and the candidate color source; the
    endpoint marker decides which side inherits that color, so the two valve
    sides must be considered independently.
    """
    arr = np.array(grid, copy=True)
    h, w = arr.shape
    bg = _background_color(arr)

    colored = [
        (r, c)
        for r in range(h)
        for c in range(w)
        if int(arr[r, c]) not in (bg, GREY)
    ]

    valves = set()
    for r, c in colored:
        grey_n = sum(1 for nr, nc in _neighbors(r, c, h, w) if arr[nr, nc] == GREY)
        if grey_n >= 2:
            valves.add((r, c))

    markers = set(colored) - valves
    comps, comp_id = _components_of_grey(arr)
    assignments = {}

    core = np.zeros((h, w), dtype=bool)
    core[arr == GREY] = True
    for r, c in valves:
        core[r, c] = True

    marker_by_pos = {pos: int(arr[pos]) for pos in markers}

    for vr, vc in valves:
        color = int(arr[vr, vc])
        side_ids = set()
        for nr, nc in _neighbors(vr, vc, h, w):
            if (nr, nc) in comp_id:
                side_ids.add(comp_id[(nr, nc)])

        for cid in side_ids:
            touches_matching_marker = False
            for pr, pc in comps[cid]:
                core_degree = sum(1 for nr, nc in _neighbors(pr, pc, h, w) if core[nr, nc])
                if core_degree > 1:
                    continue
                for mr, mc in _neighbors(pr, pc, h, w):
                    if marker_by_pos.get((mr, mc)) == color:
                        touches_matching_marker = True
                        break
                if touches_matching_marker:
                    break
            if touches_matching_marker:
                assignments.setdefault(cid, color)

    out = arr.copy()
    for cid, color in assignments.items():
        for r, c in comps[cid]:
            out[r, c] = color
    for r, c in valves:
        out[r, c] = bg
    return out


def _random_tree(rows, cols):
    start = (random.randrange(rows), random.randrange(cols))
    seen = {start}
    frontier = []

    def add_frontier(cell):
        r, c = cell
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) not in seen:
                frontier.append((cell, (nr, nc)))

    add_frontier(start)
    edges = []
    degree = {start: 0}
    while frontier and len(seen) < rows * cols:
        idx = random.randrange(len(frontier))
        a, b = frontier.pop(idx)
        if b in seen:
            continue
        seen.add(b)
        degree.setdefault(a, 0)
        degree.setdefault(b, 0)
        degree[a] += 1
        degree[b] += 1
        edges.append((a, b))
        add_frontier(b)
    return edges, degree


def _pipe_components_for_generator(pipe, valves):
    h, w = pipe.shape
    open_pipe = pipe.copy()
    for r, c in valves:
        open_pipe[r, c] = False
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    comp_id = {}
    for r in range(h):
        for c in range(w):
            if not open_pipe[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp = []
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                comp_id[(cr, cc)] = len(comps)
                for nr, nc in _neighbors(cr, cc, h, w):
                    if open_pipe[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps, comp_id


def generate_input() -> np.ndarray:
    for _ in range(200):
        node_rows = random.randint(4, 7)
        node_cols = random.randint(4, 7)
        h = random.randint(2 * node_rows + 1, 30)
        w = random.randint(2 * node_cols + 1, 30)
        top = random.randint(1, h - 2 * (node_rows - 1) - 2)
        left = random.randint(1, w - 2 * (node_cols - 1) - 2)

        edges, degree = _random_tree(node_rows, node_cols)
        leaves = [cell for cell, deg in degree.items() if deg == 1]
        if len(leaves) < 4 or max(degree.values(), default=0) < 3:
            continue

        grid = np.zeros((h, w), dtype=int)
        pipe = np.zeros((h, w), dtype=bool)
        node_pos = {
            (r, c): (top + 2 * r, left + 2 * c)
            for r in range(node_rows)
            for c in range(node_cols)
        }

        for pos in node_pos.values():
            pipe[pos] = True
        edge_cells = []
        for a, b in edges:
            ar, ac = node_pos[a]
            br, bc = node_pos[b]
            mid = ((ar + br) // 2, (ac + bc) // 2)
            pipe[mid] = True
            edge_cells.append((mid, a, b))

        valve_count = random.randint(2, min(5, len(edge_cells)))
        chosen_edges = random.sample(edge_cells, valve_count)
        valves = [mid for mid, _, _ in chosen_edges]
        colors = random.sample([1, 2, 3, 4, 6, 7, 8, 9], valve_count)

        comps, comp_id = _pipe_components_for_generator(pipe, set(valves))
        valve_sides = []
        for valve in valves:
            r, c = valve
            side_ids = {
                comp_id[(nr, nc)]
                for nr, nc in _neighbors(r, c, h, w)
                if (nr, nc) in comp_id
            }
            if len(side_ids) < 2:
                break
            valve_sides.append(side_ids)
        else:
            leaf_markers = {}
            for leaf in leaves:
                lr, lc = node_pos[leaf]
                nr, nc = next(other for edge in edges for other in edge if leaf in edge and other != leaf)
                dr = leaf[0] - nr
                dc = leaf[1] - nc
                marker = (lr + dr, lc + dc)
                if not (0 <= marker[0] < h and 0 <= marker[1] < w):
                    continue
                if pipe[marker]:
                    continue
                cid = comp_id.get((lr, lc))
                if cid is not None:
                    leaf_markers.setdefault(cid, []).append(marker)

            assignments = {}
            used_markers = set()
            valve_order = list(range(valve_count))
            random.shuffle(valve_order)
            for vi in valve_order:
                candidate_sides = [
                    cid
                    for cid in valve_sides[vi]
                    if cid in leaf_markers and cid not in assignments
                ]
                random.shuffle(candidate_sides)
                if not candidate_sides:
                    continue
                take = 1 if random.random() < 0.7 else min(2, len(candidate_sides))
                for cid in candidate_sides[:take]:
                    choices = [m for m in leaf_markers[cid] if m not in used_markers]
                    if not choices:
                        continue
                    marker = random.choice(choices)
                    assignments[cid] = (colors[vi], marker)
                    used_markers.add(marker)

            if len(assignments) < max(1, valve_count // 2):
                continue

            grid[pipe] = GREY
            for (r, c), color in zip(valves, colors):
                grid[r, c] = color
            for color, marker in assignments.values():
                grid[marker] = color
            return grid

    return generate_input()
