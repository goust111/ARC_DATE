# Combo variant of 332f06d7 (index 33)
# combo_spec: D4/1+D5/3+D8/3
# Name: L形移动体+对接停靠+形状线索方向

from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np
from scipy.ndimage import label as scipy_label

from arc_utils import find_components


"""
Design analysis for the fused variant
=====================================

a) Concrete behavior and trace
   - Original task: a solid n x n square moves through blue space, and destination is another
     solid n x n square. Reachability depends on whether every cell of the translated square
     stays inside the blue corridor.
   - Fused variant:
     1. The mover body is an L polyomino inside an n x n bbox, with n in {2,3,4}.
     2. One extra same-colored tip pixel sticks out on one side of that L body. The tip is
        input-only and indicates the mover's initial outgoing direction.
     3. The destination is a red docking frame whose inner cavity is an L-shaped blue notch
        matching the mover body orientation. The frame touches one grid edge and has a mouth
        on the inward side so the mover can dock into the cavity.
     4. Motion uses the exact L footprint, not the full n x n bbox. If the dock anchor is
        reachable, the mover body slides there and nests into the cavity; otherwise it stops
        at the farthest reachable placement. The tip disappears in the output.
   - Small trace:
     * Suppose n=3 and the body is the canonical bottom-left L:
       {(0,0),(1,0),(2,0),(2,1),(2,2)}.
     * Original logic would test a full 3 x 3 window. The fused logic tests only those
       5 occupied cells, so an apparent 3-wide blue strip can still fail if the upper arm
       cell needed by the L is missing.
     * If the tip points right, the start placement must have exactly one feasible neighbor
       to the right. The destination frame might touch the top edge, with a bottom-side mouth
       leading into a bottom-left L cavity. The mover can only dock if both the corridor is
       shape-passable and the cavity orientation matches the mover body.

b) Complexity calibration
   - Only one body family is used: L shapes. This keeps the category count low while still
     making the geometry visibly different from the original square task.
   - The only marker is a single protrusion pixel. It is visually salient but does not create
     additional moving geometry in the output.
   - Docks always use the same rigid construction: an (n+2) x (n+2) red frame/support with an
     internal blue L notch. This is large enough to be obvious and restrictive enough that the
     docking condition is readable across a few examples.

c) Hidden assumptions from the original main() that break
   - "Both key objects are solid squares" is false. The mover is an L plus a tip, and the
     destination is a structured frame.
   - "Sort by color to decide mover vs destination" is invalid. The generator randomizes object
     colors, so the solver must detect the arrow-marked L component and the dock structurally.
   - "Full n x n window must be traversable" is invalid. Only the occupied L cells matter.
   - "Destination color is passable" is invalid. Red dock pixels are walls; only river and the
     mover component itself are traversable during the search.
   - "Erase / paint entire bbox" is invalid. The output must erase the full marked mover
     component (body plus tip) but only paint the L body at the terminal position.

d) Placement legality enforced by the generator
   - The river is generated from translated copies of the exact L body, so every intended step
     is in-bounds by construction.
   - The start placement is accepted only when it has exactly one feasible neighboring placement,
     making the tip direction unambiguous.
   - The dock frame is built from a verified mask at an anchor one cell in from a grid edge.
     Its cavity orientation always matches the mover body.
   - Stuck cases carve shape-specific chokepoints by deleting one arm pattern from consecutive
     placements on a straight run. The river stays pixel-connected, but the full L placement
     becomes impossible beyond the gate.

e) Visual observability
   - Humans can see three simultaneous cues: the mover is the irregular object with a tip, the
     destination is the frame with a matching L notch, and the mover body orientation matches
     the notch shape. The tip shows the initial slide direction without relying on color order.
   - Reachable outputs visibly end with the black/random-colored L nested inside the red frame.
     Stuck outputs leave the frame empty and stop the mover earlier on the river.

f) Visual cleanliness
   - Only four semantic regions remain: background, river, mover color, dock color.
   - The tip is a single pixel on existing river, so it does not clutter the scene.
   - The dock is a single connected red component with a clean blue cavity, rather than several
     separate markers.

g) Fusion-behavior audit targets
   - Rectangle-baseline disagreement: many generated instances should have a different terminal
     position if one incorrectly uses a full n x n square instead of the L body.
   - Color-order disagreement: because mover/dock colors are randomized, "smaller color moves"
     should fail regularly. This prevents silent regression to the original identification rule.

h) New solver assumptions introduced here
   - Exactly one non-river component is an L body plus exactly one side tip pixel.
   - Exactly one other non-river component is a valid dock mask whose cavity matches that L
     orientation and size.
   - The river is the most common non-background color; generator area ratios enforce this.
   - The mover start anchor is feasible and the dock anchor lies within placement bounds.
   The generator below explicitly constructs inputs satisfying these assumptions.
"""


_BG = 3
_RIVER = 1
_OBJECT_COLORS = [0, 2, 4, 5, 6, 7, 8, 9]
_DIR4 = {"U": (-1, 0), "D": (1, 0), "L": (0, -1), "R": (0, 1)}
_OPPOSITE = {"U": "D", "D": "U", "L": "R", "R": "L"}
_SIDE_TO_DIR = {"top": "U", "bottom": "D", "left": "L", "right": "R"}
_DIR_TO_SIDE = {v: k for k, v in _SIDE_TO_DIR.items()}
_OPPOSITE_SIDE = {"top": "bottom", "bottom": "top", "left": "right", "right": "left"}


def _rotate_coords(coords: set[tuple[int, int]], n: int) -> set[tuple[int, int]]:
    return {(int(c), int(n - 1 - r)) for r, c in coords}


def _canonical_l(n: int) -> set[tuple[int, int]]:
    return {(r, 0) for r in range(n)} | {(n - 1, c) for c in range(n)}


def _shape_config(n: int, turns: int) -> dict:
    coords = _canonical_l(n)
    remove = {(r, 0) for r in range(n - 1)}
    gate_axis = "h"
    for _ in range(turns % 4):
        coords = _rotate_coords(coords, n)
        remove = _rotate_coords(remove, n)
        gate_axis = "v" if gate_axis == "h" else "h"
    return {
        "n": int(n),
        "coords": tuple(sorted(coords)),
        "remove": tuple(sorted(remove)),
        "gate_axis": gate_axis,
    }


def _all_shape_configs(n: int) -> list[dict]:
    return [_shape_config(n, turns) for turns in range(4)]


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


def _paint_shape(grid: np.ndarray, anchor: tuple[int, int], coords, color: int) -> None:
    r0, c0 = anchor
    for dr, dc in coords:
        grid[int(r0 + dr), int(c0 + dc)] = int(color)


def _compute_feasible(trav: np.ndarray, coords, n: int) -> np.ndarray:
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
        for dr, dc in _DIR4.values():
            rr = int(r + dr)
            cc = int(c + dc)
            if 0 <= rr < h2 and 0 <= cc < w2 and bool(feasible[rr, cc]) and int(dist[rr, cc]) == -1:
                dist[rr, cc] = int(dist[int(r), int(c)] + 1)
                q.append((int(rr), int(cc)))
    return dist


def _pick_terminal(dist: np.ndarray, feasible: np.ndarray, sr: int, sc: int, tr0: int, tc0: int):
    h2, w2 = feasible.shape
    if 0 <= tr0 < h2 and 0 <= tc0 < w2 and bool(feasible[int(tr0), int(tc0)]) and int(dist[int(tr0), int(tc0)]) >= 0:
        return int(tr0), int(tc0), True
    best = -1
    tr, tc = int(sr), int(sc)
    for r in range(h2):
        for c in range(w2):
            if int(dist[int(r), int(c)]) > best:
                best = int(dist[int(r), int(c)])
                tr, tc = int(r), int(c)
    return tr, tc, False


def _dock_mask(n: int, coords: tuple[tuple[int, int], ...], frame_edge: str) -> np.ndarray:
    m = n + 2
    mask = np.zeros((m, m), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    coord_set = set(coords)
    for r in range(n):
        for c in range(n):
            if (r, c) not in coord_set:
                mask[1 + r, 1 + c] = True
    mouth_side = _OPPOSITE_SIDE[frame_edge]
    if mouth_side == "top":
        for r, c in coord_set:
            if r == 0:
                mask[0, 1 + c] = False
    elif mouth_side == "bottom":
        for r, c in coord_set:
            if r == n - 1:
                mask[m - 1, 1 + c] = False
    elif mouth_side == "left":
        for r, c in coord_set:
            if c == 0:
                mask[1 + r, 0] = False
    else:
        for r, c in coord_set:
            if c == n - 1:
                mask[1 + r, m - 1] = False
    return mask


def _dock_anchor_for_edge(H: int, W: int, n: int, frame_edge: str) -> tuple[int, int]:
    if frame_edge == "top":
        return 1, -1
    if frame_edge == "bottom":
        return H - n - 1, -1
    if frame_edge == "left":
        return -1, 1
    return -1, W - n - 1


def _valid_frame_edges(coords: tuple[tuple[int, int], ...], n: int) -> list[str]:
    sides = _occupied_sides(coords, n)
    edges = []
    for side in sides:
        edges.append(_OPPOSITE_SIDE[side])
    return edges


def _find_arrow_l_info(comp: dict) -> dict | None:
    coords = {(int(r), int(c)) for r, c in comp["coords"]}
    if len(coords) < 6:
        return None

    matches: list[dict] = []
    for tip_r, tip_c in coords:
        body = coords - {(tip_r, tip_c)}
        rows = [r for r, _ in body]
        cols = [c for _, c in body]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        n = int(r1 - r0 + 1)
        if n != int(c1 - c0 + 1) or n < 2:
            continue
        if len(body) != 2 * n - 1:
            continue
        rel = {(int(r - r0), int(c - c0)) for r, c in body}
        matched = None
        for cfg in _all_shape_configs(n):
            if rel != set(cfg["coords"]):
                continue
            direction = None
            if tip_r == r0 - 1 and c0 <= tip_c <= c1:
                direction = "U"
            elif tip_r == r1 + 1 and c0 <= tip_c <= c1:
                direction = "D"
            elif tip_c == c0 - 1 and r0 <= tip_r <= r1:
                direction = "L"
            elif tip_c == c1 + 1 and r0 <= tip_r <= r1:
                direction = "R"
            if direction is None:
                continue
            if not any((tip_r + dr, tip_c + dc) in body for dr, dc in _DIR4.values()):
                continue
            matched = cfg
            matches.append(
                {
                    "col": int(comp["color"]),
                    "r0": int(r0),
                    "c0": int(c0),
                    "n": int(n),
                    "dir": direction,
                    "tip": (int(tip_r), int(tip_c)),
                    "body_coords_abs": set(body),
                    "shape_coords": tuple(sorted(matched["coords"])),
                    "full_coords_abs": set(coords),
                }
            )
            break
    if len(matches) != 1:
        return None
    return matches[0]


def _find_dock_info(g: np.ndarray, bg: int, river: int) -> list[dict]:
    docks: list[dict] = []
    seen = set()
    for comp in find_components(g, bg=bg, connectivity=4, monochromatic=True, return_format="dict"):
        col = int(comp["color"])
        if col == int(river):
            continue
        coords = {(int(r), int(c)) for r, c in comp["coords"]}
        rows = [r for r, _ in coords]
        cols = [c for _, c in coords]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        h = int(r1 - r0 + 1)
        w = int(c1 - c0 + 1)
        if h != w or h < 4:
            continue
        n = h - 2
        patch = g[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]
        for cfg in _all_shape_configs(n):
            for edge in _valid_frame_edges(cfg["coords"], n):
                if np.array_equal(patch == int(col), _dock_mask(n, cfg["coords"], edge)):
                    key = (int(col), int(r0), int(c0), int(n), tuple(sorted(cfg["coords"])), edge)
                    if key in seen:
                        continue
                    seen.add(key)
                    docks.append(
                        {
                            "col": int(col),
                            "r0": int(r0),
                            "c0": int(c0),
                            "n": int(n),
                            "shape_coords": tuple(sorted(cfg["coords"])),
                            "frame_edge": edge,
                            "dock_r": int(r0 + 1),
                            "dock_c": int(c0 + 1),
                        }
                    )
    return docks


def _solve_core(grid: np.ndarray):
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(Counter([int(g[0, 0]), int(g[0, -1]), int(g[-1, 0]), int(g[-1, -1])]).most_common(1)[0][0])

    counts = Counter(map(int, g.flatten()))
    river = None
    for col, _ct in counts.most_common():
        if int(col) != int(bg):
            river = int(col)
            break
    assert river is not None

    movers: list[dict] = []
    for comp in find_components(g, bg=bg, connectivity=4, monochromatic=True, return_format="dict"):
        if int(comp["color"]) in (int(bg), int(river)):
            continue
        info = _find_arrow_l_info(comp)
        if info is not None:
            movers.append(info)
    docks = _find_dock_info(g, bg, river)

    pairs = [
        (mover, dock)
        for mover in movers
        for dock in docks
        if int(mover["n"]) == int(dock["n"])
        and tuple(mover["shape_coords"]) == tuple(dock["shape_coords"])
        and int(mover["col"]) != int(dock["col"])
    ]
    uniq = {}
    for mover, dock in pairs:
        key = (
            int(mover["r0"]),
            int(mover["c0"]),
            int(dock["r0"]),
            int(dock["c0"]),
            tuple(mover["shape_coords"]),
        )
        uniq[key] = (mover, dock)
    pairs = list(uniq.values())
    assert len(pairs) == 1
    mover, dock = pairs[0]

    n = int(mover["n"])
    sr, sc = int(mover["r0"]), int(mover["c0"])
    dr, dc = int(dock["dock_r"]), int(dock["dock_c"])
    shape_coords = tuple(mover["shape_coords"])

    trav = (g == int(river)) | (g == int(mover["col"]))
    feasible = _compute_feasible(trav, shape_coords, n)
    assert bool(feasible[int(sr), int(sc)])

    dist = _bfs_placement(feasible, sr, sc)
    tr, tc, dest_ok = _pick_terminal(dist, feasible, sr, sc, dr, dc)

    out = g.copy()
    for r, c in mover["full_coords_abs"]:
        out[int(r), int(c)] = int(river)
    _paint_shape(out, (tr, tc), shape_coords, int(mover["col"]))
    return out, {
        "bg": bg,
        "river": river,
        "mover": mover,
        "dock": dock,
        "shape_coords": shape_coords,
        "start": (sr, sc),
        "target": (dr, dc),
        "terminal": (tr, tc),
        "feasible": feasible,
        "dist": dist,
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
            for dr, dc in _DIR4.values():
                nr, nc = cur[0] + dr, cur[1] + dc
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
                    for dr2, dc2 in _DIR4.values():
                        nr2, nc2 = cr + dr2, cc + dc2
                        if 0 <= nr2 < rows and 0 <= nc2 < cols and (nr2, nc2) not in test_visited and (nr2, nc2) not in seen:
                            seen.add((nr2, nc2))
                            q.append((nr2, nc2))
                for dr2, dc2 in _DIR4.values():
                    ar, ac = nb[0] + dr2, nb[1] + dc2
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
        d1 = (path[i][0] - path[i - 1][0], path[i][1] - path[i - 1][1])
        d2 = (path[i + 1][0] - path[i][0], path[i + 1][1] - path[i][1])
        if d1 != d2:
            turns += 1
    return turns


def _straight_runs(path):
    if len(path) < 2:
        return []
    runs = []
    start = 0
    axis = "h" if path[0][0] == path[1][0] else "v"
    for i in range(1, len(path) - 1):
        next_axis = "h" if path[i][0] == path[i + 1][0] else "v"
        if next_axis != axis:
            runs.append({"start": start, "end": i, "axis": axis, "length": i - start + 1})
            start = i
            axis = next_axis
    runs.append({"start": start, "end": len(path) - 1, "axis": axis, "length": len(path) - start})
    return runs


def _extend_path_to_dock(path, h2: int, w2: int, frame_edges: list[str]):
    used = set(path)
    r, c = path[-1]
    candidates = []
    for edge in frame_edges:
        if edge == "top":
            if not (1 <= c <= w2 - 2):
                continue
            target = (1, c)
        elif edge == "bottom":
            if not (1 <= c <= w2 - 2):
                continue
            target = (h2 - 2, c)
        elif edge == "left":
            if not (1 <= r <= h2 - 2):
                continue
            target = (r, 1)
        else:
            if not (1 <= r <= h2 - 2):
                continue
            target = (r, w2 - 2)
        if target == (r, c):
            continue
        seq = []
        rr, cc = r, c
        ok = True
        while (rr, cc) != target:
            if rr < target[0]:
                rr += 1
            elif rr > target[0]:
                rr -= 1
            elif cc < target[1]:
                cc += 1
            else:
                cc -= 1
            pos = (int(rr), int(cc))
            if pos in used:
                ok = False
                break
            seq.append(pos)
        if ok and seq:
            candidates.append((len(seq), path + seq, edge))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0])
    _dist, new_path, edge = candidates[0]
    return new_path, edge


def _extend_to_dock(grid: np.ndarray, pr: int, pc: int, n: int, frame_edges: list[str]):
    H, W = grid.shape
    candidates = []
    random.shuffle(frame_edges)
    for edge in frame_edges:
        if edge == "top":
            dock_r, dock_c = 1, pc
            strip = grid[min(pr, dock_r) : max(pr, dock_r) + n, pc : pc + n]
        elif edge == "bottom":
            dock_r, dock_c = H - n - 1, pc
            strip = grid[min(pr, dock_r) : max(pr, dock_r) + n, pc : pc + n]
        elif edge == "left":
            dock_r, dock_c = pr, 1
            strip = grid[pr : pr + n, min(pc, dock_c) : max(pc, dock_c) + n]
        else:
            dock_r, dock_c = pr, W - n - 1
            strip = grid[pr : pr + n, min(pc, dock_c) : max(pc, dock_c) + n]

        if not (0 <= dock_r <= H - n and 0 <= dock_c <= W - n):
            continue
        fr, fc = dock_r - 1, dock_c - 1
        if not (0 <= fr and 0 <= fc and fr + n + 2 <= H and fc + n + 2 <= W):
            continue
        patch = grid[fr : fr + n + 2, fc : fc + n + 2]
        if not bool(np.isin(strip, (_BG, _RIVER)).all()):
            continue
        if not bool(np.isin(patch, (_BG, _RIVER)).all()):
            continue
        dist = abs(dock_r - pr) + abs(dock_c - pc)
        if dist == 0:
            continue
        candidates.append((dist, dock_r, dock_c, edge))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0])
    _dist, dock_r, dock_c, edge = candidates[0]
    if dock_r != pr:
        grid[min(pr, dock_r) : max(pr, dock_r) + n, pc : pc + n] = _RIVER
    if dock_c != pc:
        grid[pr : pr + n, min(pc, dock_c) : max(pc, dock_c) + n] = _RIVER
    return int(dock_r), int(dock_c), edge


def _render_river(grid: np.ndarray, path, shape_coords) -> None:
    for anchor in path:
        _paint_shape(grid, anchor, shape_coords, _RIVER)


def _river_connected(grid: np.ndarray) -> bool:
    mask = grid == _RIVER
    if not bool(mask.any()):
        return False
    _labeled, num = scipy_label(mask.astype(int))
    return int(num) == 1


def _neighbor_dirs(feasible: np.ndarray, r: int, c: int) -> list[str]:
    out = []
    for name, (dr, dc) in _DIR4.items():
        rr, cc = int(r + dr), int(c + dc)
        if 0 <= rr < feasible.shape[0] and 0 <= cc < feasible.shape[1] and bool(feasible[rr, cc]):
            out.append(name)
    return out


def _choose_tip_cell(grid: np.ndarray, r0: int, c0: int, coords, direction: str):
    pts = list(coords)
    center = (sum(r for r, _ in pts) / len(pts), sum(c for _, c in pts) / len(pts))
    ranked = []
    for dr, dc in coords:
        if direction == "U":
            tr, tc = r0 + dr - 1, c0 + dc
            side_ok = all((rr != dr or cc != dc) for rr, cc in coords if rr == dr - 1 and cc == dc)
        elif direction == "D":
            tr, tc = r0 + dr + 1, c0 + dc
            side_ok = all((rr != dr or cc != dc) for rr, cc in coords if rr == dr + 1 and cc == dc)
        elif direction == "L":
            tr, tc = r0 + dr, c0 + dc - 1
            side_ok = all((rr != dr or cc != dc) for rr, cc in coords if rr == dr and cc == dc - 1)
        else:
            tr, tc = r0 + dr, c0 + dc + 1
            side_ok = all((rr != dr or cc != dc) for rr, cc in coords if rr == dr and cc == dc + 1)
        if not side_ok:
            continue
        if 0 <= tr < grid.shape[0] and 0 <= tc < grid.shape[1] and int(grid[tr, tc]) == _RIVER:
            score = abs(dr - center[0]) + abs(dc - center[1])
            ranked.append((score, tr, tc))
    if not ranked:
        return None
    ranked.sort(key=lambda x: x[0])
    return int(ranked[0][1]), int(ranked[0][2])


def _paint_dock(grid: np.ndarray, dock_r: int, dock_c: int, n: int, shape_coords, frame_edge: str, color: int) -> None:
    mask = _dock_mask(n, shape_coords, frame_edge)
    fr = int(dock_r - 1)
    fc = int(dock_c - 1)
    patch = grid[fr : fr + n + 2, fc : fc + n + 2]
    patch[mask] = int(color)


def _shape_rect_result(grid: np.ndarray):
    try:
        out, info = _solve_core(grid)
    except AssertionError:
        return None
    mover = info["mover"]
    dock = info["dock"]
    n = int(mover["n"])
    g = np.array(grid, dtype=int)
    trav = (g == int(info["river"])) | (g == int(mover["col"]))
    feasible = np.zeros((g.shape[0] - n + 1, g.shape[1] - n + 1), dtype=bool)
    for r in range(feasible.shape[0]):
        for c in range(feasible.shape[1]):
            feasible[r, c] = bool(trav[r : r + n, c : c + n].all())
    if not bool(feasible[int(mover["r0"]), int(mover["c0"])]):
        return {"start_ok": False, "target": (int(mover["r0"]), int(mover["c0"])), "dest_ok": False}
    dist = _bfs_placement(feasible, int(mover["r0"]), int(mover["c0"]))
    tr, tc, dest_ok = _pick_terminal(dist, feasible, int(mover["r0"]), int(mover["c0"]), int(dock["dock_r"]), int(dock["dock_c"]))
    return {"start_ok": True, "target": (tr, tc), "dest_ok": dest_ok, "out": out}


def fusion_metrics(grid: np.ndarray) -> dict:
    _, info = _solve_core(grid)
    rect = _shape_rect_result(grid)
    mover_col = int(info["mover"]["col"])
    dock_col = int(info["dock"]["col"])
    return {
        "rect_diff": (not rect["start_ok"]) or rect["dest_ok"] != bool(info["dest_ok"]) or tuple(rect["target"]) != tuple(info["terminal"]),
        "color_order_wrong_for_original": mover_col > dock_col,
        "initial_dir": info["mover"]["dir"],
        "frame_edge": info["dock"]["frame_edge"],
        "dest_reached": bool(info["dest_ok"]),
    }


def _gen_attempt():
    reached = random.random() < 0.55
    n = random.choice([2, 3, 4])
    min_size = max(12, 4 * n + 2)
    if min_size > 20:
        return None
    G = random.randint(min_size, 20)
    H = W = G
    h2 = H - n + 1
    w2 = W - n + 1

    coarse_dim = random.randint(3, min(6, max(3, (G - 2) // max(2, n))))
    offsets_r = _compute_offsets(G, n, coarse_dim)
    offsets_c = _compute_offsets(G, n, coarse_dim)
    if offsets_r is None or offsets_c is None:
        return None

    edge_cells = set()
    for c in range(coarse_dim):
        edge_cells.add((0, c))
        edge_cells.add((coarse_dim - 1, c))
    for r in range(1, coarse_dim - 1):
        edge_cells.add((r, 0))
        edge_cells.add((r, coarse_dim - 1))
    edge_cells = list(edge_cells)
    random.shuffle(edge_cells)
    start_coarse = edge_cells[0]
    far = sorted(edge_cells, key=lambda e: -(abs(e[0] - start_coarse[0]) + abs(e[1] - start_coarse[1])))
    end_coarse = far[0] if far[0] != start_coarse else far[1]

    walk = _hamiltonian_walk(coarse_dim, coarse_dim, start_coarse, end_coarse)
    if walk is None or len(walk) < 4:
        return None

    path = _expand_walk(walk, offsets_r, offsets_c)
    if len(path) < 12 or len(set(path)) != len(path) or _count_turns(path) < 3:
        return None

    config = random.choice(_all_shape_configs(n))
    mover_anchor = path[0]
    if min(mover_anchor[0], h2 - 1 - mover_anchor[0], mover_anchor[1], w2 - 1 - mover_anchor[1]) < 1:
        return None

    grid = np.full((H, W), _BG, dtype=int)
    _render_river(grid, path, config["coords"])
    dock_spec = _extend_to_dock(grid, int(path[-1][0]), int(path[-1][1]), n, _valid_frame_edges(config["coords"], n))
    if dock_spec is None:
        return None
    dock_anchor = (int(dock_spec[0]), int(dock_spec[1]))
    frame_edge = dock_spec[2]

    if not reached:
        runs = _straight_runs(path)
        viable = []
        for run in runs:
            if run["axis"] != config["gate_axis"]:
                continue
            if run["length"] < 5:
                continue
            if run["start"] < len(path) // 5:
                continue
            if run["end"] > len(path) - 3:
                continue
            viable.append(run)
        if not viable:
            return None
        run = random.choice(viable)
        gate_len = random.randint(2, min(4, run["length"] - 2))
        gate_start = random.randint(run["start"] + 1, run["end"] - gate_len)
        for idx in range(gate_start, gate_start + gate_len):
            _paint_shape(grid, path[idx], config["remove"], _BG)
        if not _river_connected(grid):
            return None

    mask = grid != _BG
    if np.any(mask, axis=1).sum() / H < 0.55 or np.any(mask, axis=0).sum() / W < 0.55:
        return None

    trav = grid == _RIVER
    feasible = _compute_feasible(trav, config["coords"], n)
    if not bool(feasible[int(mover_anchor[0]), int(mover_anchor[1])]):
        return None
    start_dirs = _neighbor_dirs(feasible, int(mover_anchor[0]), int(mover_anchor[1]))
    if len(start_dirs) != 1:
        return None
    tip = _choose_tip_cell(grid, int(mover_anchor[0]), int(mover_anchor[1]), config["coords"], start_dirs[0])
    if tip is None:
        return None

    mover_col, dock_col = random.sample(_OBJECT_COLORS, 2)
    _paint_shape(grid, mover_anchor, config["coords"], mover_col)
    grid[int(tip[0]), int(tip[1])] = int(mover_col)
    _paint_dock(grid, int(dock_anchor[0]), int(dock_anchor[1]), n, config["coords"], frame_edge, dock_col)

    try:
        out, info = _solve_core(grid)
    except AssertionError:
        return None
    if tuple(info["target"]) != tuple(dock_anchor):
        return None
    if tuple(info["mover"]["tip"]) != tuple(tip):
        return None
    if info["mover"]["dir"] != start_dirs[0]:
        return None

    expected = grid.copy()
    for r, c in info["mover"]["full_coords_abs"]:
        expected[int(r), int(c)] = _RIVER
    _paint_shape(expected, info["terminal"], config["coords"], mover_col)
    if not np.array_equal(out, expected):
        return None

    try:
        metrics = fusion_metrics(grid)
    except AssertionError:
        return None
    if not metrics["rect_diff"]:
        return None
    return grid


def generate_input():
    for _ in range(500):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("Failed to generate valid input after 500 attempts")
