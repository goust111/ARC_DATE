# Combo variant of 332f06d7 (index 8)
# combo_spec: D4/1+D5/3
# Name: L形移动体+对接停靠

from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np
from scipy.ndimage import label as scipy_label


"""
Design analysis for the fused variant
=====================================

a) Concrete form and behavior trace
   - Original: a solid black n×n square slides through blue n-wide corridor windows toward a solid red n×n square.
   - Fused variant: the mover is a rotated L polyomino inside an n×n bbox, with n in {2,3,4}. The red terminal
     object is a docking frame whose outer bbox is (n+2)×(n+2). Inside that red frame, the river-colored cavity is
     the same rotated L shape, and the frame opens toward the corridor only along the side where that L touches the
     boundary.
   - Small trace for n=3, base L = left column + bottom row:
       original move-right test at anchor (r,c+1): require all 9 cells in the 3×3 window to be traversable.
       fused move-right test at anchor (r,c+1): require only the 5 L cells to be traversable.
       original goal: black square covers the red square.
       fused goal: black L reaches anchor (dock_r,dock_c) and occupies only the blue L cavity, leaving the red frame
       visible around and inside the non-cavity cells.
     The difference is therefore present in both dynamics and terminal condition: movement is shape-specific, and the
     destination is a shape-matched docking cavity rather than a replaceable solid block.

b) Complexity calibration
   - Only one mover family is used: L. That keeps the fusion visually readable and avoids introducing too many latent
     categories at once.
   - Rotation still varies, so the task is clearly different from the original square task and from a single fixed
     template. The same orientation is reused consistently by mover, corridor sweep, and docking cavity.
   - n stays small and square-bbox based, preserving the original task's physical scale and making placement search
     tractable.

c) Implicit assumption audit against the original main()
   - "Find solid same-sized squares" breaks: the black mover is no longer a filled rectangle, and the red target is
     no longer solid at all.
   - "Destination top-left equals red bbox top-left" breaks: the docking target is the cavity anchor, which is always
     offset by +1,+1 from the frame bbox.
   - "Destination color is traversable" breaks: red frame pixels are walls, only the blue cavity and mouth are
     traversable.
   - "Feasible placement means the whole n×n bbox is traversable" breaks: only the occupied L cells should be checked.
   - "Erase / paint the full bbox" breaks: doing so would destroy the hole structure that carries the new rule.
   - New fused assumptions introduced here:
     1. exactly one black connected component exists and its normalized shape is one of the 4 rotations of an L;
     2. exactly one red connected docking frame exists and its cavity matches the black L exactly;
     3. the docking frame bbox is n+2 square and the cavity anchor is at bbox offset +1,+1.
     The generator below constructs inputs to guarantee all three.

d) Placement legality
   - The generator works in placement-coordinate space: every path anchor is a top-left position of the L bbox.
   - The blue corridor is rendered as the union of translated copies of the same L shape, so every intended anchor
     remains in-bounds by construction.
   - The docking frame is painted only after the corridor exists, and only if its whole (n+2)×(n+2) patch contains
     background/river pixels. This prevents overlap with unrelated objects.
   - Stuck cases remove only the non-spine cells of the L along a straight run. The river stays 4-connected, but the
     full L placement becomes impossible past the gate.

e) Visual observability
   - A human sees three reinforcing cues across examples:
     1. the mover is visibly L-shaped rather than square;
     2. the corridor width matters relative to that exact L silhouette;
     3. the red goal is a larger docking frame with an L-shaped blue notch, and reachable outputs keep that frame
        visible while the black L nests inside it.
   - No hidden sorting by color value is used as the conceptual rule. Black is "the moving shape" and red is "the
     frame" because those roles are visually explicit.

f) Visual cleanliness
   - Only the original four colors are used.
   - The frame is the only red object, the mover is the only black object, and the channel remains a single connected
     blue region before objects are overpainted.
   - The final black docked shape does not overwrite the entire frame; it occupies only the cavity, which makes the
     docking outcome easy to read.

g) Color / change inferability
   - Color semantics stay stable: green background, blue traversable space, black mover, red docking frame.
   - The important change is structural, not a hidden recoloring rule: the solver follows the exact shape geometry.

h) Quantifiable non-degeneration checks used after implementation
   - Compare the fused solver against a rectangle oracle that treats the mover as a full n×n square on the same input.
     Because the corridor and dock cavity are L-specific, many samples should yield a different reachable target.
   - Measure the number of red pixels inside the docking frame's inner n×n area. It must be positive in every sample;
     otherwise the target would collapse back to the older square-cavity docking variant.

i) Generator-to-solver contract review
   - _solve_core assumes the mover's bbox is square and its normalized cells match one of the 4 rotated L masks.
   - _find_matching_dock_frame assumes the red component's bbox is exactly n+2 and equals one generated frame mask.
   - BFS assumes the start anchor is feasible under the L-mask occupancy test.
   - generate_input() verifies all of these by construction and by running main() as an oracle before accepting a grid.
"""


_BG, _RIVER, _MOVER, _DEST = 3, 1, 0, 2
_DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
_SIDES = ("top", "bottom", "left", "right")


def _largest_component_of_color(grid: np.ndarray, color: int):
    mask = grid == int(color)
    if not bool(mask.any()):
        return None
    labeled, num = scipy_label(mask.astype(int))
    if int(num) == 0:
        return None
    best = None
    best_sz = -1
    for idx in range(1, int(num) + 1):
        coords = np.argwhere(labeled == idx)
        if int(coords.shape[0]) > best_sz:
            best = coords
            best_sz = int(coords.shape[0])
    assert best is not None
    r0 = int(best[:, 0].min())
    c0 = int(best[:, 1].min())
    r1 = int(best[:, 0].max())
    c1 = int(best[:, 1].max())
    rel = tuple(sorted((int(r - r0), int(c - c0)) for r, c in best.tolist()))
    return {
        "r0": r0,
        "c0": c0,
        "r1": r1,
        "c1": c1,
        "h": int(r1 - r0 + 1),
        "w": int(c1 - c0 + 1),
        "coords": rel,
    }


def _swap_axis(axis: str) -> str:
    return "v" if axis == "h" else "h"


def _rotate_coords(coords: set[tuple[int, int]], n: int) -> set[tuple[int, int]]:
    return {(int(c), int(n - 1 - r)) for r, c in coords}


def _shape_config(n: int, turns: int):
    mask = {(r, 0) for r in range(n)} | {(n - 1, c) for c in range(n)}
    remove = {(r, 0) for r in range(n - 1)}
    gate_axis = "h"
    for _ in range(turns % 4):
        mask = _rotate_coords(mask, n)
        remove = _rotate_coords(remove, n)
        gate_axis = _swap_axis(gate_axis)
    return {
        "n": n,
        "coords": tuple(sorted(mask)),
        "remove": tuple(sorted(remove)),
        "gate_axis": gate_axis,
    }


def _occupied_sides(coords: tuple[tuple[int, int], ...], n: int) -> set[str]:
    sides = set()
    for r, c in coords:
        if r == 0:
            sides.add("top")
        if r == n - 1:
            sides.add("bottom")
        if c == 0:
            sides.add("left")
        if c == n - 1:
            sides.add("right")
    return sides


def _dock_edges_for_shape(coords: tuple[tuple[int, int], ...], n: int) -> list[str]:
    sides = _occupied_sides(coords, n)
    edges = []
    if "bottom" in sides:
        edges.append("top")
    if "top" in sides:
        edges.append("bottom")
    if "right" in sides:
        edges.append("left")
    if "left" in sides:
        edges.append("right")
    return [edge for edge in edges if _approach_works(coords, n, edge)]


def _paint_shape(grid: np.ndarray, anchor: tuple[int, int], coords, color: int) -> None:
    r0, c0 = anchor
    for dr, dc in coords:
        grid[int(r0 + dr), int(c0 + dc)] = int(color)


def _approach_works(coords: tuple[tuple[int, int], ...], n: int, edge: str) -> bool:
    H = W = n + 8
    grid = np.full((H, W), _BG, dtype=int)
    if edge == "top":
        dock = (1, 3)
        prev = (2, 3)
    elif edge == "bottom":
        dock = (H - n - 1, 3)
        prev = (H - n - 2, 3)
    elif edge == "left":
        dock = (3, 1)
        prev = (3, 2)
    else:
        dock = (3, W - n - 1)
        prev = (3, W - n - 2)
    _paint_shape(grid, prev, coords, _RIVER)
    _paint_shape(grid, dock, coords, _RIVER)
    _paint_dock_frame(grid, dock[0], dock[1], coords, n, edge, _DEST)
    feasible = _compute_feasible(grid == _RIVER, coords, n)
    return bool(feasible[prev[0], prev[1]] and feasible[dock[0], dock[1]])


def _frame_mask_from_shape(shape_coords: tuple[tuple[int, int], ...], n: int, gap_side: str) -> np.ndarray:
    m = n + 2
    mask = np.zeros((m, m), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True

    shape_set = set(shape_coords)
    for r in range(n):
        for c in range(n):
            if (r, c) not in shape_set:
                mask[int(r + 1), int(c + 1)] = True

    if gap_side == "top":
        for r, c in shape_coords:
            if r == 0:
                mask[0, int(c + 1)] = False
    elif gap_side == "bottom":
        for r, c in shape_coords:
            if r == n - 1:
                mask[int(n + 1), int(c + 1)] = False
    elif gap_side == "left":
        for r, c in shape_coords:
            if c == 0:
                mask[int(r + 1), 0] = False
    else:
        for r, c in shape_coords:
            if c == n - 1:
                mask[int(r + 1), int(n + 1)] = False

    return mask


def _frame_spec_for_edge(H: int, W: int, dock_r: int, dock_c: int, n: int, edge: str) -> tuple[int, int, str]:
    if edge == "top":
        return 0, int(dock_c - 1), "bottom"
    if edge == "bottom":
        return int(H - (n + 2)), int(dock_c - 1), "top"
    if edge == "left":
        return int(dock_r - 1), 0, "right"
    return int(dock_r - 1), int(W - (n + 2)), "left"


def _paint_dock_frame(
    grid: np.ndarray,
    dock_r: int,
    dock_c: int,
    shape_coords: tuple[tuple[int, int], ...],
    n: int,
    edge: str,
    color: int,
) -> None:
    H, W = grid.shape
    fr, fc, gap_side = _frame_spec_for_edge(H, W, dock_r, dock_c, n, edge)
    mask = _frame_mask_from_shape(shape_coords, n, gap_side)
    patch = grid[fr : fr + n + 2, fc : fc + n + 2]
    patch[mask] = int(color)

    for dr, dc in shape_coords:
        grid[int(fr + 1 + dr), int(fc + 1 + dc)] = _RIVER
    if gap_side == "top":
        for dr, dc in shape_coords:
            if dr == 0:
                grid[int(fr), int(fc + 1 + dc)] = _RIVER
    elif gap_side == "bottom":
        for dr, dc in shape_coords:
            if dr == n - 1:
                grid[int(fr + n + 1), int(fc + 1 + dc)] = _RIVER
    elif gap_side == "left":
        for dr, dc in shape_coords:
            if dc == 0:
                grid[int(fr + 1 + dr), int(fc)] = _RIVER
    else:
        for dr, dc in shape_coords:
            if dc == n - 1:
                grid[int(fr + 1 + dr), int(fc + n + 1)] = _RIVER


def _find_matching_dock_frame(grid: np.ndarray, shape_coords: tuple[tuple[int, int], ...], n: int):
    frame = _largest_component_of_color(grid, _DEST)
    assert frame is not None
    assert int(frame["h"]) == int(frame["w"]) == int(n + 2)
    r0 = int(frame["r0"])
    c0 = int(frame["c0"])
    patch = grid[r0 : int(frame["r1"]) + 1, c0 : int(frame["c1"]) + 1]
    for gap_side in _SIDES:
        if np.array_equal(patch == _DEST, _frame_mask_from_shape(shape_coords, n, gap_side)):
            return {
                "r0": r0,
                "c0": c0,
                "gap_side": gap_side,
                "dock_r": int(r0 + 1),
                "dock_c": int(c0 + 1),
            }
    raise AssertionError("No matching docking frame found")


def _compute_feasible(trav: np.ndarray, coords: tuple[tuple[int, int], ...], n: int) -> np.ndarray:
    H, W = trav.shape
    h2 = H - n + 1
    w2 = W - n + 1
    feasible = np.zeros((h2, w2), dtype=bool)
    for r in range(h2):
        for c in range(w2):
            ok = True
            for dr, dc in coords:
                if not bool(trav[int(r + dr), int(c + dc)]):
                    ok = False
                    break
            feasible[int(r), int(c)] = ok
    return feasible


def _bfs_placement(feasible: np.ndarray, sr: int, sc: int) -> np.ndarray:
    h2, w2 = feasible.shape
    dist = -np.ones((h2, w2), dtype=int)
    dist[int(sr), int(sc)] = 0
    q = deque([(int(sr), int(sc))])
    while q:
        r, c = q.popleft()
        for dr, dc in _DIR4:
            rr = int(r + dr)
            cc = int(c + dc)
            if 0 <= rr < h2 and 0 <= cc < w2 and bool(feasible[rr, cc]) and int(dist[rr, cc]) == -1:
                dist[rr, cc] = int(dist[int(r), int(c)] + 1)
                q.append((rr, cc))
    return dist


def _pick_target(dist: np.ndarray, feasible: np.ndarray, sr: int, sc: int, dr: int, dc: int):
    h2, w2 = feasible.shape
    if 0 <= dr < h2 and 0 <= dc < w2 and bool(feasible[int(dr), int(dc)]) and int(dist[int(dr), int(dc)]) >= 0:
        return int(dr), int(dc), True
    best = -1
    tr, tc = int(sr), int(sc)
    for r in range(h2):
        for c in range(w2):
            if int(dist[int(r), int(c)]) > best:
                best = int(dist[int(r), int(c)])
                tr, tc = int(r), int(c)
    return tr, tc, False


def _solve_core(grid: np.ndarray):
    g = np.array(grid, dtype=int)
    bg = int(Counter([int(g[0, 0]), int(g[0, -1]), int(g[-1, 0]), int(g[-1, -1])]).most_common(1)[0][0])

    mover = _largest_component_of_color(g, _MOVER)
    assert mover is not None
    assert int(mover["h"]) == int(mover["w"])
    n = int(mover["h"])
    assert n >= 2
    shape_coords = tuple(mover["coords"])
    valid_shapes = {tuple(_shape_config(n, turns)["coords"]) for turns in range(4)}
    assert shape_coords in valid_shapes

    frame = _find_matching_dock_frame(g, shape_coords, n)
    sr, sc = int(mover["r0"]), int(mover["c0"])
    dr, dc = int(frame["dock_r"]), int(frame["dock_c"])

    assert bool((g == _RIVER).any())
    river = _RIVER
    trav = (g == int(river)) | (g == _MOVER)
    feasible = _compute_feasible(trav, shape_coords, n)
    assert bool(feasible[int(sr), int(sc)])

    dist = _bfs_placement(feasible, sr, sc)
    tr, tc, dest_ok = _pick_target(dist, feasible, sr, sc, dr, dc)

    out = g.copy()
    _paint_shape(out, (sr, sc), shape_coords, river)
    _paint_shape(out, (tr, tc), shape_coords, _MOVER)
    return out, {
        "bg": bg,
        "river": river,
        "n": n,
        "shape_coords": shape_coords,
        "start": (sr, sc),
        "dest": (dr, dc),
        "target": (tr, tc),
        "dist": dist,
        "feasible": feasible,
        "dest_ok": dest_ok,
    }


def main(grid: np.ndarray) -> np.ndarray:
    return _solve_core(grid)[0]


def _hamiltonian_walk(rows: int, cols: int, start: tuple[int, int], end: tuple[int, int]):
    total = rows * cols
    best_path = None
    best_len = 0
    for _attempt in range(80):
        visited = {start}
        path = [start]
        cur = start
        while True:
            if cur == end and len(path) >= total * 0.7:
                if len(path) > best_len:
                    best_path = list(path)
                    best_len = len(path)
                break
            neighbors = []
            for dr, dc in _DIR4:
                nr = int(cur[0] + dr)
                nc = int(cur[1] + dc)
                if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) not in visited:
                    neighbors.append((nr, nc))
            if not neighbors:
                break
            valid = []
            for nb in neighbors:
                if nb == end:
                    valid.append(nb)
                    continue
                test_visited = visited | {nb}
                seen = {end}
                q = deque([end])
                while q:
                    cr, cc = q.popleft()
                    for dr2, dc2 in _DIR4:
                        nr2 = int(cr + dr2)
                        nc2 = int(cc + dc2)
                        if 0 <= nr2 < rows and 0 <= nc2 < cols and (nr2, nc2) not in test_visited and (nr2, nc2) not in seen:
                            seen.add((nr2, nc2))
                            q.append((nr2, nc2))
                for dr2, dc2 in _DIR4:
                    ar = int(nb[0] + dr2)
                    ac = int(nb[1] + dc2)
                    if 0 <= ar < rows and 0 <= ac < cols and (ar, ac) not in test_visited and (ar, ac) in seen:
                        valid.append(nb)
                        break
            if not valid:
                break
            random.shuffle(valid)
            nxt = valid[0]
            visited.add(nxt)
            path.append(nxt)
            cur = nxt
        if best_path and best_len >= total * 0.85:
            break
    return best_path


def _compute_offsets(G: int, n: int, num_coarse: int):
    if num_coarse == 1:
        return [1]
    first = 1
    last = G - n - 1
    if last < first:
        return None
    step = (last - first) / (num_coarse - 1)
    return [int(round(first + i * step)) for i in range(num_coarse)]


def _expand_walk(walk, offsets_r, offsets_c):
    path = [(int(offsets_r[walk[0][0]]), int(offsets_c[walk[0][1]]))]
    for cr, cc in walk[1:]:
        tr = int(offsets_r[cr])
        tc = int(offsets_c[cc])
        r, c = path[-1]
        while (r, c) != (tr, tc):
            if r < tr:
                r += 1
            elif r > tr:
                r -= 1
            elif c < tc:
                c += 1
            else:
                c -= 1
            path.append((int(r), int(c)))
    return path


def _count_turns(path) -> int:
    turns = 0
    for i in range(1, len(path) - 1):
        a = path[i - 1]
        b = path[i]
        c = path[i + 1]
        d1 = (int(b[0] - a[0]), int(b[1] - a[1]))
        d2 = (int(c[0] - b[0]), int(c[1] - b[1]))
        if d1 != d2:
            turns += 1
    return turns


def _straight_runs(path):
    if len(path) < 2:
        return []
    runs = []
    start = 0
    cur_axis = "h" if path[0][0] == path[1][0] else "v"
    for i in range(1, len(path) - 1):
        next_axis = "h" if path[i][0] == path[i + 1][0] else "v"
        if next_axis != cur_axis:
            runs.append({"start": start, "end": i, "axis": cur_axis, "length": int(i - start + 1)})
            start = i
            cur_axis = next_axis
    runs.append({"start": start, "end": len(path) - 1, "axis": cur_axis, "length": int(len(path) - start)})
    return runs


def _render_river(grid: np.ndarray, path, shape_coords) -> None:
    for anchor in path:
        _paint_shape(grid, anchor, shape_coords, _RIVER)


def _river_connected(grid: np.ndarray) -> bool:
    mask = grid == _RIVER
    if not bool(mask.any()):
        return False
    _labeled, num = scipy_label(mask.astype(int))
    return int(num) == 1


def _extend_path_to_dock(path, h2: int, w2: int, shape_coords: tuple[tuple[int, int], ...], n: int):
    used = set(path)
    r, c = path[-1]
    edges = _dock_edges_for_shape(shape_coords, n)
    random.shuffle(edges)
    candidates = []
    for edge in edges:
        rr, cc = int(r), int(c)
        seq = []
        while True:
            if edge == "top":
                if rr == 1:
                    break
                rr -= 1
            elif edge == "bottom":
                if rr == h2 - 2:
                    break
                rr += 1
            elif edge == "left":
                if cc == 1:
                    break
                cc -= 1
            else:
                if cc == w2 - 2:
                    break
                cc += 1
            pos = (int(rr), int(cc))
            if pos in used:
                seq = None
                break
            seq.append(pos)
        if seq:
            candidates.append((len(seq), path + seq, edge))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0])
    return candidates[0][1], candidates[0][2]


def _anchor_edge(anchor: tuple[int, int], h2: int, w2: int):
    r, c = anchor
    edges = []
    if r == 1:
        edges.append("top")
    if r == h2 - 2:
        edges.append("bottom")
    if c == 1:
        edges.append("left")
    if c == w2 - 2:
        edges.append("right")
    return edges[0] if len(edges) == 1 else None


def _rect_oracle_result(grid: np.ndarray):
    g = np.array(grid, dtype=int)
    mover = _largest_component_of_color(g, _MOVER)
    frame = _largest_component_of_color(g, _DEST)
    assert mover is not None and frame is not None
    n = int(mover["h"])
    sr, sc = int(mover["r0"]), int(mover["c0"])
    dr, dc = int(frame["r0"] + 1), int(frame["c0"] + 1)
    trav = (g == _RIVER) | (g == _MOVER)
    feasible = np.zeros((g.shape[0] - n + 1, g.shape[1] - n + 1), dtype=bool)
    for r in range(feasible.shape[0]):
        for c in range(feasible.shape[1]):
            feasible[r, c] = bool(trav[r : r + n, c : c + n].all())
    if not bool(feasible[sr, sc]):
        return {"start_ok": False, "dest_ok": False, "target": (sr, sc)}
    dist = _bfs_placement(feasible, sr, sc)
    tr, tc, dest_ok = _pick_target(dist, feasible, sr, sc, dr, dc)
    return {"start_ok": True, "dest_ok": dest_ok, "target": (tr, tc)}


def _gen_attempt():
    reached = random.random() < 0.55
    n = random.choice([2, 3, 4])
    configs = [_shape_config(n, turns) for turns in range(4)]

    min_size = max(12, 4 * n + 2)
    if min_size > 20:
        return None
    G = random.randint(min_size, 20)
    H = W = G
    h2 = H - n + 1
    w2 = W - n + 1

    max_coarse = max(3, min(6, (G - 2) // max(2, n)))
    coarse_dim = random.randint(3, max_coarse)
    offsets_r = _compute_offsets(G, n, coarse_dim)
    offsets_c = _compute_offsets(G, n, coarse_dim)
    if offsets_r is None or offsets_c is None:
        return None

    edge_cells = []
    for c in range(1, coarse_dim - 1):
        edge_cells.append((0, c))
        edge_cells.append((coarse_dim - 1, c))
    for r in range(1, coarse_dim - 1):
        edge_cells.append((r, 0))
        edge_cells.append((r, coarse_dim - 1))
    if not edge_cells:
        return None
    random.shuffle(edge_cells)
    start_coarse = edge_cells[0]
    far = sorted(edge_cells, key=lambda e: -(abs(e[0] - start_coarse[0]) + abs(e[1] - start_coarse[1])))
    end_coarse = far[0] if far[0] != start_coarse else far[1]

    walk = _hamiltonian_walk(coarse_dim, coarse_dim, start_coarse, end_coarse)
    if walk is None or len(walk) < 4:
        return None

    path = _expand_walk(walk, offsets_r, offsets_c)
    if len(path) < 12 or _count_turns(path) < 3 or len(set(path)) != len(path):
        return None

    mover_anchor = path[0]
    dock_anchor = path[-1]
    dock_edge = _anchor_edge(dock_anchor, h2, w2)
    if dock_edge is None:
        return None
    if min(mover_anchor[0], h2 - 1 - mover_anchor[0], mover_anchor[1], w2 - 1 - mover_anchor[1]) != 1:
        return None

    runs = _straight_runs(path)
    viable = []
    for config in configs:
        if dock_edge not in _dock_edges_for_shape(config["coords"], n):
            continue
        if reached:
            viable.append(config)
            continue
        for run in runs:
            if run["axis"] == config["gate_axis"] and run["length"] >= 5:
                viable.append(config)
                break
    if not viable:
        return None
    chosen = random.choice(viable)

    if dock_edge == "top" and dock_anchor[0] != 1:
        return None
    if dock_edge == "bottom" and dock_anchor[0] != h2 - 2:
        return None
    if dock_edge == "left" and dock_anchor[1] != 1:
        return None
    if dock_edge == "right" and dock_anchor[1] != w2 - 2:
        return None

    grid = np.full((H, W), _BG, dtype=int)
    _render_river(grid, path, chosen["coords"])

    if not reached:
        runs = _straight_runs(path)
        viable_after = []
        for run in runs:
            if run["axis"] != chosen["gate_axis"]:
                continue
            if run["length"] < 5:
                continue
            if run["start"] < len(path) // 5:
                continue
            if run["end"] > len(path) - 4:
                continue
            viable_after.append(run)
        if not viable_after:
            return None
        run = random.choice(viable_after)
        gate_len = random.randint(2, min(4, run["length"] - 2))
        gate_start = random.randint(run["start"] + 1, run["end"] - gate_len)
        for idx in range(gate_start, gate_start + gate_len):
            _paint_shape(grid, path[idx], chosen["remove"], _BG)
        if not _river_connected(grid):
            return None

    mask = grid != _BG
    if np.any(mask, axis=1).sum() / H < 0.55 or np.any(mask, axis=0).sum() / W < 0.55:
        return None

    fr, fc, _gap_side = _frame_spec_for_edge(H, W, dock_anchor[0], dock_anchor[1], n, dock_edge)
    if not (0 <= fr and 0 <= fc and fr + n + 2 <= H and fc + n + 2 <= W):
        return None
    frame_patch = grid[fr : fr + n + 2, fc : fc + n + 2]
    if not bool(np.isin(frame_patch, (_BG, _RIVER)).all()):
        return None

    _paint_shape(grid, mover_anchor, chosen["coords"], _MOVER)
    _paint_dock_frame(grid, dock_anchor[0], dock_anchor[1], chosen["coords"], n, dock_edge, _DEST)

    out, info = _solve_core(grid)
    if reached != bool(info["dest_ok"]):
        return None
    if not reached and tuple(info["target"]) == tuple(dock_anchor):
        return None

    expected = grid.copy()
    _paint_shape(expected, mover_anchor, chosen["coords"], _RIVER)
    _paint_shape(expected, info["target"], chosen["coords"], _MOVER)
    if not np.array_equal(out, expected):
        return None

    rect_info = _rect_oracle_result(grid)
    inner_red = int((grid[fr + 1 : fr + n + 1, fc + 1 : fc + n + 1] == _DEST).sum())
    metadata = {
        "n": n,
        "reached": reached,
        "rect_diff": (not rect_info["start_ok"]) or bool(rect_info["dest_ok"]) != bool(info["dest_ok"]) or tuple(rect_info["target"]) != tuple(info["target"]),
        "inner_red": inner_red,
    }
    if metadata["inner_red"] <= 0:
        return None
    return grid, metadata


def generate_input():
    for _ in range(500):
        result = _gen_attempt()
        if result is not None:
            return result[0]
    raise RuntimeError("Failed to generate valid input after 500 attempts")
