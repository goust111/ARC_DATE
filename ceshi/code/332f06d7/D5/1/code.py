# Variant of 332f06d7
# Dimension: D5 — 终点条件 (dst)
# Change: 同尺寸红色实心方块位于通道末端（接触网格边缘），可达时方块覆盖目标位置... → 距离编码终点

# Design analysis for the D5 variant
# ==================================
# spec_variant:
#   core_preserved:
#     - Grid stays square, 10..20 per side, with green background in the corners.
#     - A blue winding river is still the only traversable corridor.
#     - A black n×n square still slides by feasible n×n top-left placements on the river.
#     - Chokepoints narrower than n can still make the square stop before the river end.
#   D5_change:
#     - The edge target is no longer a solid red stop-square.
#     - It becomes an n×n red coding block that contains some yellow pixels.
#     - The number of yellow pixels is the requested movement distance in placement steps.
#     - The mover advances min(encoded_steps, accessible_route_length) steps along the river route.
#     - Therefore the red/yellow block is a distance marker, not inherently the stopping square.
#   visual_realization:
#     - Colors stay semantically stable across examples:
#       green=background, blue=river, black=mover, red=destination block, yellow=step code.
#     - The coding block still touches exactly one grid edge like the old destination,
#       so humans can align it with the previous "river end" role.
#     - Yellow cells are painted as a compact prefix from the touched edge inward, so the count
#       is visually salient instead of looking like random noise.
#   behavior_trace:
#     - Original: if the river remained wide enough, the black square slid all the way onto the
#       red square; otherwise it slid to the last feasible placement before the chokepoint.
#     - Variant: first decode k = number of yellow cells inside the edge block.
#       Then follow the same placement graph toward the river end:
#         * if the reachable route has length L and k < L, stop k steps early;
#         * if k == L, land exactly on the end point;
#         * if k > L, cap at the last reachable point.
#     - Example: with n=3 and a reachable route of 7 placements to the edge block,
#       an old instance would always end on the red block.
#       In this variant, 4 yellow pixels make the mover stop after 4 placements even though the
#       river still continues for 3 more placements.
#   complexity_calibration:
#     - The new signal uses only one extra color and one bounded degree of freedom: count 1..n^2-1.
#     - n is restricted to 2..4 because n=1 cannot visibly host an internal count.
#     - The coding block remains the only multicolor square touching an edge, which avoids ambiguity.
#     - The route logic stays one-dimensional enough for humans because the river is still a winding
#       corridor without branches in placement space.
#   implicit_assumption_audit_against_old_main:
#     - Old solid-square detection for both mover and destination no longer holds:
#       the destination is now mixed red/yellow, so the solver must detect only the mover as a
#       solid monochrome square and detect the target as a multicolor square touching an edge.
#     - Old color-order sorting by numeric id is not valid for the variant design requirement.
#       The new solver uses geometry/role cues instead: solid interior square for the mover,
#       edge-touching mixed-color square for the code block.
#     - Old target selection "reachable destination -> land on dest else farthest reachable" is
#       replaced by "choose the route end exactly as before, then clamp by encoded step count".
#     - Old traversable mask used {river, mover, dest}; the variant must also include yellow code
#       cells because they occupy traversable destination area.
#   placement_legality:
#     - Generator keeps mover and coding block non-overlapping and inside bounds.
#     - The coding block always touches exactly one edge, not a corner, so its role is legible.
#     - The mover still starts near an edge but not touching it, matching the original feel.
#     - The river region beneath mover/target is painted first, then those objects are overlaid,
#       so removing the mover restores the river exactly.
#   observability_and_cleanliness:
#     - Reachable samples are biased toward k < route length, so many examples visibly stop before
#       the river end; this prevents silent regression to the original "always cover red" rule.
#     - Some samples still use k >= route length, preserving the cap behavior and preventing the
#       task from collapsing into "always move a short fixed amount".
#     - Only one mixed-color object exists, and it is isolated at the corridor end, so the grid
#       remains easy to parse.
#   new_solver_assumptions_checked_against_generator:
#     - Exactly one solid non-background square of side n exists: the mover.
#     - Exactly one connected component over {red,yellow} has bbox n×n and touches one edge:
#       the coding block.
#     - The blue river is still the dominant non-background color by area.
#     - The feasible-placement graph from the mover is non-empty and has at least one move.

from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np
from scipy.ndimage import label as scipy_label

from arc_utils import bbox_h, bbox_w, find_components


_BG, _RIVER, _MOVER, _DEST, _CODE = 3, 1, 0, 2, 4
_DIR4 = [(1, 0), (-1, 0), (0, 1), (0, -1)]


def _corner_bg(g: np.ndarray) -> int:
    return int(Counter([int(g[0, 0]), int(g[0, -1]), int(g[-1, 0]), int(g[-1, -1])]).most_common(1)[0][0])


def _touch_count(H: int, W: int, r0: int, c0: int, n: int) -> int:
    return int(r0 == 0) + int(r0 + n == H) + int(c0 == 0) + int(c0 + n == W)


def _edge_side(H: int, W: int, r0: int, c0: int, n: int) -> str:
    if r0 == 0:
        return "top"
    if r0 + n == H:
        return "bottom"
    if c0 == 0:
        return "left"
    return "right"


def _detect_mover(g: np.ndarray, bg: int) -> dict:
    solids: list[dict] = []
    for comp in find_components(g, bg=bg, connectivity=4, monochromatic=True, return_format="dict"):
        col = int(comp["color"])
        b = comp["bbox"]
        if bbox_h(b) != bbox_w(b):
            continue
        r0, c0, r1, c1 = map(int, b)
        patch = g[r0 : r1 + 1, c0 : c1 + 1]
        if not bool((patch == col).all()):
            continue
        solids.append({"color": col, "r0": r0, "c0": c0, "n": int(bbox_h(b))})
    assert solids
    solids.sort(key=lambda s: (-int(s["n"]), int(s["r0"]), int(s["c0"])))
    mover = solids[0]
    assert int(mover["color"]) == int(_MOVER)
    assert int(mover["n"]) >= 2
    return mover


def _detect_target(g: np.ndarray, bg: int, n: int) -> dict:
    target_comps = find_components(
        g,
        bg=bg,
        connectivity=4,
        monochromatic=False,
        colors=[_DEST, _CODE],
        return_format="dict",
    )
    matches = []
    H, W = g.shape
    for comp in target_comps:
        b = comp["bbox"]
        if bbox_h(b) != n or bbox_w(b) != n:
            continue
        r0, c0, r1, c1 = map(int, b)
        if _touch_count(H, W, r0, c0, n) != 1:
            continue
        patch = g[r0 : r1 + 1, c0 : c1 + 1]
        if not bool(np.isin(patch, [_DEST, _CODE]).all()):
            continue
        steps = int((patch == _CODE).sum())
        if not (1 <= steps < n * n):
            continue
        matches.append({"r0": r0, "c0": c0, "n": n, "steps": steps, "edge": _edge_side(H, W, r0, c0, n)})
    assert len(matches) == 1
    return matches[0]


def _bfs_placement(feasible: np.ndarray, sr: int, sc: int) -> tuple[np.ndarray, np.ndarray]:
    h2, w2 = feasible.shape
    dist = -np.ones((h2, w2), dtype=int)
    parent = -np.ones((h2, w2, 2), dtype=int)
    dist[sr, sc] = 0
    q = deque([(sr, sc)])
    while q:
        r, c = q.popleft()
        for dr, dc in _DIR4:
            rr, cc = r + dr, c + dc
            if 0 <= rr < h2 and 0 <= cc < w2 and feasible[rr, cc] and dist[rr, cc] == -1:
                dist[rr, cc] = dist[r, c] + 1
                parent[rr, cc] = (r, c)
                q.append((rr, cc))
    return dist, parent


def _recover_path(parent: np.ndarray, sr: int, sc: int, tr: int, tc: int) -> list[tuple[int, int]]:
    path = [(int(tr), int(tc))]
    r, c = int(tr), int(tc)
    while not (r == sr and c == sc):
        pr, pc = map(int, parent[r, c])
        assert not (pr < 0 or pc < 0)
        path.append((pr, pc))
        r, c = pr, pc
    path.reverse()
    return path


def _analyze_scene(grid: np.ndarray) -> dict:
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = _corner_bg(g)

    mover = _detect_mover(g, bg)
    n = int(mover["n"])
    sr, sc = int(mover["r0"]), int(mover["c0"])

    river = None
    for col, _ct in Counter(map(int, g.flatten())).most_common():
        if int(col) in (int(bg), int(_MOVER), int(_DEST), int(_CODE)):
            continue
        river = int(col)
        break
    assert river == _RIVER

    target = _detect_target(g, bg, n)
    dr, dc = int(target["r0"]), int(target["c0"])
    steps = int(target["steps"])

    trav = np.isin(g, [_RIVER, _MOVER, _DEST, _CODE])
    h2, w2 = H - n + 1, W - n + 1
    assert not (h2 <= 0 or w2 <= 0)
    assert not (not (0 <= sr < h2 and 0 <= sc < w2))

    feasible = np.zeros((h2, w2), dtype=bool)
    for r in range(h2):
        for c in range(w2):
            if bool(trav[r : r + n, c : c + n].all()):
                feasible[r, c] = True
    assert bool(feasible[sr, sc])

    dist, parent = _bfs_placement(feasible, sr, sc)
    dest_ok = bool(0 <= dr < h2 and 0 <= dc < w2 and feasible[dr, dc] and dist[dr, dc] >= 0)

    if dest_ok:
        goal_r, goal_c = dr, dc
    else:
        best_key = None
        goal_r, goal_c = sr, sc
        for r in range(h2):
            for c in range(w2):
                if int(dist[r, c]) < 0:
                    continue
                key = (int(dist[r, c]), -abs(int(r - dr)) - abs(int(c - dc)))
                if best_key is None or key > best_key:
                    best_key = key
                    goal_r, goal_c = r, c

    route = _recover_path(parent, sr, sc, goal_r, goal_c)
    accessible_len = len(route) - 1
    final_idx = min(int(steps), int(accessible_len))
    tr, tc = route[final_idx]

    return {
        "grid": g,
        "H": H,
        "W": W,
        "bg": bg,
        "river": river,
        "mover": mover,
        "target": target,
        "feasible": feasible,
        "dist": dist,
        "dest_ok": dest_ok,
        "goal": (goal_r, goal_c),
        "route": route,
        "accessible_len": accessible_len,
        "final": (int(tr), int(tc)),
    }


def _variant_metrics(grid: np.ndarray) -> dict:
    info = _analyze_scene(grid)
    encoded = int(info["target"]["steps"])
    accessible = int(info["accessible_len"])
    return {
        "encoded_steps": encoded,
        "accessible_len": accessible,
        "dest_reachable": bool(info["dest_ok"]),
        "stops_short": bool(encoded < accessible),
        "hits_route_end": bool(encoded >= accessible),
        "lands_on_target": bool(info["dest_ok"] and encoded >= accessible),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide a solid square through a winding channel by a distance encoded inside the edge marker.

    Input structure:
    - bg dominates the corners, and a river color forms a winding corridor.
    - One solid black n×n square is the mover.
    - One red n×n block touches the grid edge; some cells inside it are yellow and encode a step count.

    Rule:
    - The mover follows the same feasible-placement river logic as in the original task.
    - Count the yellow cells in the edge block to get the requested number of steps.
    - Move that many placement steps toward the corridor end, capped by the farthest reachable point.
    - Replace the mover's starting square with river color; the coded edge block stays unless the mover lands on it.

    Key insight: The red edge block is no longer the stopping condition by itself — its yellow cell count is.
    """
    info = _analyze_scene(grid)
    g = info["grid"]
    mover = info["mover"]
    river = int(info["river"])
    sr, sc = int(mover["r0"]), int(mover["c0"])
    n = int(mover["n"])
    tr, tc = map(int, info["final"])

    out = g.copy()
    out[sr : sr + n, sc : sc + n] = river
    out[tr : tr + n, tc : tc + n] = int(mover["color"])
    return out


def _hamiltonian_walk(rows: int, cols: int, start: tuple[int, int], end: tuple[int, int]) -> list[tuple[int, int]] | None:
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
                    for dr2, dc2 in _DIR4:
                        nr2, nc2 = cr + dr2, cc + dc2
                        if 0 <= nr2 < rows and 0 <= nc2 < cols and (nr2, nc2) not in test_visited and (nr2, nc2) not in seen:
                            seen.add((nr2, nc2))
                            q.append((nr2, nc2))
                for dr2, dc2 in _DIR4:
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


def _render_channel(grid: np.ndarray, path: list[tuple[int, int]], n: int, offsets_r: list[int], offsets_c: list[int]) -> list[tuple[int, int]]:
    pixel_pos = []
    for cr, cc in path:
        pr, pc = offsets_r[cr], offsets_c[cc]
        pixel_pos.append((pr, pc))
        grid[pr : pr + n, pc : pc + n] = _RIVER
    for i in range(len(path) - 1):
        pr1, pc1 = pixel_pos[i]
        pr2, pc2 = pixel_pos[i + 1]
        if pr1 == pr2:
            c_lo, c_hi = min(pc1, pc2), max(pc1, pc2) + n
            grid[pr1 : pr1 + n, c_lo:c_hi] = _RIVER
        else:
            r_lo, r_hi = min(pr1, pr2), max(pr1, pr2) + n
            grid[r_lo:r_hi, pc1 : pc1 + n] = _RIVER
    return pixel_pos


def _add_chokepoint(grid: np.ndarray, pixel_pos_i: tuple[int, int], pixel_pos_j: tuple[int, int], n: int, direction: str) -> None:
    H, W = grid.shape
    pr1, pc1 = pixel_pos_i
    pr2, pc2 = pixel_pos_j
    narrow_w = random.randint(1, n - 1)
    remove = n - narrow_w
    side = random.randint(0, 1)

    if direction == "h":
        c_lo = min(pc1, pc2) + n
        c_hi = max(pc1, pc2)
        if c_hi <= c_lo:
            c_lo, c_hi = min(pc1, pc2), max(pc1, pc2) + n
        hold_len = random.randint(1, min(6, max(1, c_hi - c_lo - 2 * remove)))
        total_len = 2 * remove + hold_len
        c_mid = (c_lo + c_hi) // 2
        c_start = max(c_lo, c_mid - total_len // 2)
        for step in range(total_len):
            c = c_start + step
            if not (0 <= c < W):
                continue
            if step < remove:
                depth = step + 1
            elif step >= total_len - remove:
                depth = total_len - step
            else:
                depth = remove
            for k in range(depth):
                rr = pr1 + k if side == 0 else pr1 + n - 1 - k
                if 0 <= rr < H:
                    grid[rr, c] = _BG
    else:
        r_lo = min(pr1, pr2) + n
        r_hi = max(pr1, pr2)
        if r_hi <= r_lo:
            r_lo, r_hi = min(pr1, pr2), max(pr1, pr2) + n
        hold_len = random.randint(1, min(6, max(1, r_hi - r_lo - 2 * remove)))
        total_len = 2 * remove + hold_len
        r_mid = (r_lo + r_hi) // 2
        r_start = max(r_lo, r_mid - total_len // 2)
        for step in range(total_len):
            r = r_start + step
            if not (0 <= r < H):
                continue
            if step < remove:
                depth = step + 1
            elif step >= total_len - remove:
                depth = total_len - step
            else:
                depth = remove
            for k in range(depth):
                cc = pc1 + k if side == 0 else pc1 + n - 1 - k
                if 0 <= cc < W:
                    grid[r, cc] = _BG


def _river_connected(grid: np.ndarray) -> bool:
    mask = np.isin(grid, [_RIVER, _MOVER, _DEST, _CODE])
    if not mask.any():
        return False
    _labeled, num = scipy_label(mask.astype(int))
    return int(num) == 1


def _compute_offsets(G: int, n: int, num_coarse: int) -> list[int] | None:
    if num_coarse == 1:
        return [1]
    first = 1
    last = G - n - 1
    if last < first:
        return None
    step = (last - first) / (num_coarse - 1)
    return [int(round(first + i * step)) for i in range(num_coarse)]


def _extend_to_edge(grid: np.ndarray, pr: int, pc: int, n: int) -> tuple[int, int] | None:
    H, W = grid.shape
    candidates = []
    if all(grid[r, c] in (_BG, _RIVER) for r in range(pr) for c in range(pc, pc + n)):
        candidates.append((0, pc, pr))
    if all(grid[r, c] in (_BG, _RIVER) for r in range(pr + n, H) for c in range(pc, pc + n)):
        candidates.append((H - n, pc, H - (pr + n)))
    if all(grid[r, c] in (_BG, _RIVER) for r in range(pr, pr + n) for c in range(pc)):
        candidates.append((pr, 0, pc))
    if all(grid[r, c] in (_BG, _RIVER) for r in range(pr, pr + n) for c in range(pc + n, W)):
        candidates.append((pr, W - n, W - (pc + n)))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[2])
    return int(candidates[0][0]), int(candidates[0][1])


def _target_fill_order(n: int, edge: str) -> list[tuple[int, int]]:
    cells = [(r, c) for r in range(n) for c in range(n)]
    if edge == "top":
        cells.sort(key=lambda p: (p[0], p[1]))
    elif edge == "bottom":
        cells.sort(key=lambda p: (-p[0], p[1]))
    elif edge == "left":
        cells.sort(key=lambda p: (p[1], p[0]))
    else:
        cells.sort(key=lambda p: (-p[1], p[0]))
    return cells


def _paint_target(grid: np.ndarray, r0: int, c0: int, n: int, steps: int, edge: str) -> None:
    patch = np.full((n, n), _DEST, dtype=int)
    for rr, cc in _target_fill_order(n, edge)[:steps]:
        patch[rr, cc] = _CODE
    grid[r0 : r0 + n, c0 : c0 + n] = patch


def _manual_expected(grid: np.ndarray) -> np.ndarray:
    info = _analyze_scene(grid)
    out = info["grid"].copy()
    mover = info["mover"]
    sr, sc = int(mover["r0"]), int(mover["c0"])
    n = int(mover["n"])
    tr, tc = map(int, info["final"])
    out[sr : sr + n, sc : sc + n] = int(info["river"])
    out[tr : tr + n, tc : tc + n] = int(mover["color"])
    return out


def _gen_attempt() -> np.ndarray | None:
    reached = random.random() < 0.55
    n = random.choice([2, 3, 4])

    min_size = max(10, 5 * n)
    if min_size > 20:
        return None
    G = random.randint(min_size, 20)
    H = W = G

    min_spacing = n + 1
    max_coarse = max(3, (G - 2) // min_spacing)
    coarse_dim = random.randint(3, min(6, max_coarse))

    offsets_r = _compute_offsets(G, n, coarse_dim)
    offsets_c = _compute_offsets(G, n, coarse_dim)
    if offsets_r is None or offsets_c is None:
        return None
    for i in range(len(offsets_r) - 1):
        if offsets_r[i + 1] - offsets_r[i] < n or offsets_c[i + 1] - offsets_c[i] < n:
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
    if len(set(cr for cr, _ in walk)) < coarse_dim or len(set(cc for _, cc in walk)) < coarse_dim:
        return None

    grid = np.full((H, W), _BG, dtype=int)
    pixel_pos = _render_channel(grid, walk, n, offsets_r, offsets_c)

    mask = (grid != _BG)
    if np.any(mask, axis=1).sum() / H < 0.70 or np.any(mask, axis=0).sum() / W < 0.70:
        return None

    mover_r, mover_c = pixel_pos[0]
    dest_r, dest_c = pixel_pos[-1]

    moved = _extend_to_edge(grid, dest_r, dest_c, n)
    if moved is None:
        return None
    new_dest_r, new_dest_c = moved
    if new_dest_r != dest_r:
        r_lo, r_hi = min(new_dest_r, dest_r), max(new_dest_r, dest_r) + n
        grid[r_lo:r_hi, dest_c : dest_c + n] = _RIVER
    if new_dest_c != dest_c:
        c_lo, c_hi = min(new_dest_c, dest_c), max(new_dest_c, dest_c) + n
        grid[dest_r : dest_r + n, c_lo:c_hi] = _RIVER
    dest_r, dest_c = new_dest_r, new_dest_c
    if _touch_count(H, W, dest_r, dest_c, n) != 1:
        return None

    touching = (mover_r == 0 or mover_r + n == H or mover_c == 0 or mover_c + n == W)
    if touching:
        if mover_r == 0:
            grid[0, mover_c : mover_c + n] = _RIVER
            mover_r = 1
        elif mover_r + n == H:
            grid[H - 1, mover_c : mover_c + n] = _RIVER
            mover_r = H - n - 1
        elif mover_c == 0:
            grid[mover_r : mover_r + n, 0] = _RIVER
            mover_c = 1
        else:
            grid[mover_r : mover_r + n, W - 1] = _RIVER
            mover_c = W - n - 1
    if mover_r == 0 or mover_r + n == H or mover_c == 0 or mover_c + n == W:
        return None
    if min(mover_r, H - (mover_r + n), mover_c, W - (mover_c + n)) > 2:
        return None

    grid[mover_r : mover_r + n, mover_c : mover_c + n] = _RIVER
    grid[dest_r : dest_r + n, dest_c : dest_c + n] = _RIVER

    if not reached:
        num_choke = random.randint(1, min(3, len(walk) - 2))
        min_seg = max(1, len(walk) // 4)
        available = list(range(min_seg, len(walk) - 1))
        random.shuffle(available)
        for seg_idx in available[:num_choke]:
            pr1, pc1 = pixel_pos[seg_idx]
            pr2, pc2 = pixel_pos[min(seg_idx + 1, len(pixel_pos) - 1)]
            direction = "h" if pr1 == pr2 else "v"
            _add_chokepoint(grid, (pr1, pc1), (pr2, pc2), n, direction)
        if not _river_connected(grid):
            return None

    edge = _edge_side(H, W, dest_r, dest_c, n)
    grid[mover_r : mover_r + n, mover_c : mover_c + n] = _MOVER
    _paint_target(grid, dest_r, dest_c, n, 1, edge)

    try:
        info0 = _analyze_scene(grid)
    except Exception:
        return None

    if reached and not bool(info0["dest_ok"]):
        return None
    if not reached and bool(info0["dest_ok"]):
        return None
    if int(info0["accessible_len"]) < 1:
        return None

    max_code = n * n - 1
    accessible = int(info0["accessible_len"])
    if accessible == 1:
        steps = random.randint(1, max_code)
    else:
        if random.random() < 0.75:
            steps = random.randint(1, min(max_code, accessible - 1))
        else:
            low = min(accessible, max_code)
            steps = random.randint(low, max_code)

    _paint_target(grid, dest_r, dest_c, n, steps, edge)

    try:
        info = _analyze_scene(grid)
    except Exception:
        return None

    if int(info["target"]["steps"]) != steps:
        return None
    expected = _manual_expected(grid)
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None

    metrics = _variant_metrics(grid)
    if not reached and not bool(metrics["hits_route_end"]) and accessible <= 2:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("Failed to generate valid input after 400 attempts")
