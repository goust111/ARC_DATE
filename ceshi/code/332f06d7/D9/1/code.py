# Variant of 332f06d7
# Dimension: D9 — 多方块移动（基数关系） (multi_step)
# Change: 1:1单一移动体到单一目标... → 一对多移动

from __future__ import annotations

from collections import Counter, defaultdict, deque
import random

import numpy as np
from scipy.ndimage import label as scipy_label

from arc_utils import bbox_h, bbox_w, find_components


# Design analysis
# a) Pixel concretization:
#    The scene stays close to the source task: green background, one blue river,
#    one black n×n mover square, and now 2-3 red n×n target squares of the same size.
#    All red squares sit on the same river. In the output the black square has advanced
#    through the river, erasing every red square it managed to visit. Unvisited red
#    squares remain red. The observable change is therefore not a new shape language,
#    but a repeated consume-and-continue behavior on the same channel.
#    Small trace: in the original task, BFS is run once from the mover and a single
#    destination cell is chosen. Here BFS is rerun after each visited target. Example:
#    mover -> target A reachable, so current position becomes A; then BFS from A finds
#    target B; after B, BFS from B either finds target C or stops at the farthest point
#    before a choke. The difference is cumulative state: multiple targets can disappear
#    before the final black position is drawn.
# b) Complexity calibration:
#    Using 2-3 same-sized red targets keeps the change visible without introducing a
#    second independent cue such as color remapping. More targets would make the river
#    visually busy; only two would still show the cardinality change. The order rule is
#    "go to the nearest remaining target by river-placement distance", which is visible
#    from the layout and does not rely on hidden color ids.
# c) Hidden assumptions from the original solver:
#    - The original `cand.sort(key=lambda s: s["col"])` assumed exactly one mover color
#      and one destination color, with color ids encoding roles. That is invalid now.
#      The new solver instead infers roles from multiplicity: one singleton square color
#      is the mover, the repeated square color is the target family.
#    - The original kept only one destination `(dr, dc)`. That fails once there are
#      several red squares. The new solver stores a list of target placements and loops.
#    - The original output only cleared the start square. Now every visited target must
#      be cleared to river before the final mover is painted.
# d) Placement legality:
#    The generator still paints the river first, then overwrites selected river-aligned
#    n×n placements with mover/target colors. Candidate target indices are spaced apart
#    along the coarse walk, and the generator rejects any sample where solid-square
#    detection merges targets or where the solver sees fewer than the intended targets.
# e) Visual observability:
#    The ordering uses shortest-path distance on the river, not color-id ordering.
#    Across examples, humans can see the black square consume red squares one after
#    another along the channel; in blocked cases it consumes a prefix and stops before
#    the next red square.
# f) Visual cleanliness:
#    Only one overlay layer exists on the river: solid squares. All targets share one
#    color and one size, so the grid remains easy to parse. Chokepoints are inserted
#    only after at least one target, avoiding clutter near the start.
# g) Color-rule inferability:
#    No new color mapping is introduced. Black remains the mover, blue the river, red
#    the target family. The only new rule is repeated visitation.
# h) Generator-side control variable:
#    `stop_after` explicitly controls how many red squares are reachable before a choke.
#    Samples are rejected unless solver metadata confirms the intended reachable prefix.
# i) New solver assumptions and how generation enforces them:
#    - Exactly one same-sized singleton-color square family exists for the mover.
#    - Exactly one repeated same-sized square family exists for the targets.
#    - At least two targets exist.
#    - Nearest-target order is unambiguous enough that the solver visits the intended
#      prefix. The generator validates this by comparing solver metadata to the chosen
#      target list and rejects ambiguous layouts.


_BG, _RIVER, _MOVER, _DEST = 3, 1, 0, 2


def _bfs_placement(feasible: np.ndarray, sr: int, sc: int) -> np.ndarray:
    h2, w2 = feasible.shape
    dist = -np.ones((h2, w2), dtype=int)
    dist[sr, sc] = 0
    q = deque([(sr, sc)])
    while q:
        r, c = q.popleft()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            rr, cc = r + dr, c + dc
            if 0 <= rr < h2 and 0 <= cc < w2 and feasible[rr, cc] and dist[rr, cc] == -1:
                dist[rr, cc] = dist[r, c] + 1
                q.append((rr, cc))
    return dist


def _collect_squares(g: np.ndarray, bg: int) -> list[dict]:
    squares: list[dict] = []
    for comp in find_components(g, bg=bg, connectivity=4, monochromatic=True, return_format="dict"):
        col = int(comp["color"])
        b = comp["bbox"]
        if bbox_h(b) != bbox_w(b):
            continue
        patch = g[int(b[0]) : int(b[2]) + 1, int(b[1]) : int(b[3]) + 1]
        if not bool((patch == col).all()):
            continue
        squares.append({"col": col, "r0": int(b[0]), "c0": int(b[1]), "n": int(bbox_h(b))})
    return squares


def _choose_square_family(squares: list[dict]) -> tuple[dict, list[dict]]:
    by_n: dict[int, list[dict]] = defaultdict(list)
    for sq in squares:
        by_n[int(sq["n"])].append(sq)

    candidates: list[tuple[int, int, dict, list[dict]]] = []
    for nside, group in by_n.items():
        by_color: dict[int, list[dict]] = defaultdict(list)
        for sq in group:
            by_color[int(sq["col"])].append(sq)
        if len(group) < 3 or len(by_color) != 2:
            continue
        mover_colors = [col for col, members in by_color.items() if len(members) == 1]
        target_colors = [col for col, members in by_color.items() if len(members) >= 2]
        if len(mover_colors) != 1 or len(target_colors) != 1:
            continue
        mover = by_color[mover_colors[0]][0]
        targets = sorted(by_color[target_colors[0]], key=lambda s: (int(s["r0"]), int(s["c0"])))
        candidates.append((len(group), int(nside), mover, targets))

    assert candidates
    _count, _nside, mover, targets = max(candidates, key=lambda item: (item[0], item[1]))
    return mover, targets


def _solve(grid: np.ndarray, *, return_meta: bool = False):
    g = np.array(grid, dtype=int)
    H, W = g.shape

    bg = int(Counter([int(g[0, 0]), int(g[0, -1]), int(g[-1, 0]), int(g[-1, -1])]).most_common(1)[0][0])
    squares = _collect_squares(g, bg)
    mover, targets = _choose_square_family(squares)
    assert len(targets) >= 2

    n = int(mover["n"])
    sr, sc = int(mover["r0"]), int(mover["c0"])
    mover_col = int(mover["col"])
    target_col = int(targets[0]["col"])

    counts = Counter(map(int, g.flatten()))
    river = None
    for col, _ct in counts.most_common():
        if col in (bg, mover_col, target_col):
            continue
        river = int(col)
        break
    assert river is not None

    trav = (g == river) | (g == mover_col) | (g == target_col)
    h2 = H - n + 1
    w2 = W - n + 1
    assert h2 > 0 and w2 > 0
    assert 0 <= sr < h2 and 0 <= sc < w2

    feasible = np.zeros((h2, w2), dtype=bool)
    for r in range(h2):
        for c in range(w2):
            if bool(trav[r : r + n, c : c + n].all()):
                feasible[r, c] = True
    assert feasible[sr, sc]

    remaining = [dict(t) for t in targets]
    visited: list[dict] = []
    current = (sr, sc)
    final_pos = current
    stopped_early = False

    while remaining:
        dist = _bfs_placement(feasible, current[0], current[1])
        reachable: list[tuple[int, int, int, int]] = []
        for idx, tgt in enumerate(remaining):
            tr, tc = int(tgt["r0"]), int(tgt["c0"])
            if 0 <= tr < h2 and 0 <= tc < w2 and feasible[tr, tc] and dist[tr, tc] >= 0:
                reachable.append((int(dist[tr, tc]), tr, tc, idx))

        if not reachable:
            best_d = -1
            tr, tc = current
            for r in range(h2):
                for c in range(w2):
                    if int(dist[r, c]) > best_d:
                        best_d = int(dist[r, c])
                        tr, tc = int(r), int(c)
            final_pos = (tr, tc)
            stopped_early = True
            break

        reachable.sort(key=lambda item: (item[0], abs(item[1] - current[0]) + abs(item[2] - current[1]), item[1], item[2]))
        _d, tr, tc, chosen_idx = reachable[0]
        chosen = remaining.pop(chosen_idx)
        visited.append(chosen)
        current = (tr, tc)
        final_pos = current

    out = g.copy()
    out[sr : sr + n, sc : sc + n] = river
    for tgt in visited:
        tr, tc = int(tgt["r0"]), int(tgt["c0"])
        out[tr : tr + n, tc : tc + n] = river
    fr, fc = final_pos
    out[fr : fr + n, fc : fc + n] = mover_col

    if not return_meta:
        return out

    meta = {
        "n": n,
        "river": river,
        "mover_color": mover_col,
        "target_color": target_col,
        "start": (sr, sc),
        "final_pos": final_pos,
        "total_targets": len(targets),
        "visited_count": len(visited),
        "visited_positions": [(int(t["r0"]), int(t["c0"])) for t in visited],
        "all_target_positions": [(int(t["r0"]), int(t["c0"])) for t in targets],
        "stopped_early": stopped_early,
    }
    return out, meta


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide one solid square through a winding channel while sequentially visiting several same-sized targets.

    Input structure:
    - The corners reveal the background color. A river color forms a connected corridor.
    - One solid NxN mover square sits on the river, plus multiple same-sized target squares sharing one color.

    Rule:
    - Treat the mover and every target as valid river occupancy, then search in placement space for NxN positions.
    - From its current position, the mover goes to the nearest remaining target that its full NxN footprint can reach.
    - After reaching a target, that target is considered consumed and the mover continues toward the next nearest target.
    - If no remaining target is reachable, the mover stops at the farthest still-reachable placement; its start and all consumed targets become river.

    Key insight:
    - The river is unchanged by intermediate visits, so the task is a repeated BFS over the same feasible-placement map.
    """
    return _solve(grid, return_meta=False)


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
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
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
                    for dr2, dc2 in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        nr2, nc2 = cr + dr2, cc + dc2
                        if 0 <= nr2 < rows and 0 <= nc2 < cols and (nr2, nc2) not in test_visited and (nr2, nc2) not in seen:
                            seen.add((nr2, nc2))
                            q.append((nr2, nc2))
                for dr2, dc2 in ((1, 0), (-1, 0), (0, 1), (0, -1)):
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


def _render_channel(grid: np.ndarray, path, n: int, offsets_r: list[int], offsets_c: list[int]):
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


def _add_chokepoint(grid: np.ndarray, pixel_pos_i: tuple[int, int], pixel_pos_j: tuple[int, int], n: int, direction: str):
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
    mask = (grid == _RIVER) | (grid == _MOVER) | (grid == _DEST)
    if not mask.any():
        return False
    _labeled, num = scipy_label(mask.astype(int))
    return num == 1


def _compute_offsets(G: int, n: int, num_coarse: int):
    if num_coarse == 1:
        return [1]
    first = 1
    last = G - n - 1
    if last < first:
        return None
    step = (last - first) / (num_coarse - 1)
    return [int(round(first + i * step)) for i in range(num_coarse)]


def _extend_to_edge(grid: np.ndarray, pr: int, pc: int, n: int):
    H, W = grid.shape
    candidates = []
    for direction in ("top", "bottom", "left", "right"):
        if direction == "top":
            dist = pr
            if dist == 0:
                candidates.append((0, pc, 0))
            elif all(grid[r, c] in (_BG, _RIVER) for r in range(pr) for c in range(pc, pc + n) if c < W):
                candidates.append((0, pc, dist))
        elif direction == "bottom":
            dist = H - (pr + n)
            if dist == 0:
                candidates.append((pr, pc, 0))
            elif all(grid[r, c] in (_BG, _RIVER) for r in range(pr + n, H) for c in range(pc, pc + n) if c < W):
                candidates.append((H - n, pc, dist))
        elif direction == "left":
            dist = pc
            if dist == 0:
                candidates.append((pr, 0, 0))
            elif all(grid[r, c] in (_BG, _RIVER) for r in range(pr, pr + n) for c in range(pc) if r < H):
                candidates.append((pr, 0, dist))
        else:
            dist = W - (pc + n)
            if dist == 0:
                candidates.append((pr, pc, 0))
            elif all(grid[r, c] in (_BG, _RIVER) for r in range(pr, pr + n) for c in range(pc + n, W) if r < H):
                candidates.append((pr, W - n, dist))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[2])
    return candidates[0][:2]


def _sample_target_indices(path_len: int, num_targets: int):
    last_idx = path_len - 1
    if num_targets < 2 or last_idx < 4:
        return None
    picks = []
    prev = 0
    for k in range(1, num_targets):
        ideal = int(round(k * last_idx / num_targets))
        lo = max(prev + 2, ideal - 2)
        hi = min(last_idx - 2 * (num_targets - k), ideal + 2)
        if lo > hi:
            return None
        idx = random.randint(lo, hi)
        picks.append(idx)
        prev = idx
    if last_idx - picks[-1] < 2:
        return None
    return picks


def _gen_attempt():
    all_reached = random.random() < 0.5
    n = random.choice([1, 2, 3, 4]) if all_reached else random.choice([2, 3, 4])

    min_size = max(10, 5 * n)
    if min_size > 20:
        return None
    G = random.randint(min_size, 20)
    H = W = G

    min_spacing = n + 1
    max_coarse = max(3, (G - 2) // min_spacing)
    coarse_dim = random.randint(3, min(6, max_coarse))
    coarse_rows = coarse_cols = coarse_dim

    offsets_r = _compute_offsets(G, n, coarse_rows)
    offsets_c = _compute_offsets(G, n, coarse_cols)
    if offsets_r is None or offsets_c is None:
        return None
    if any(offsets_r[i + 1] - offsets_r[i] < n for i in range(len(offsets_r) - 1)):
        return None
    if any(offsets_c[i + 1] - offsets_c[i] < n for i in range(len(offsets_c) - 1)):
        return None

    edge_cells = set()
    for c in range(coarse_cols):
        edge_cells.add((0, c))
        edge_cells.add((coarse_rows - 1, c))
    for r in range(1, coarse_rows - 1):
        edge_cells.add((r, 0))
        edge_cells.add((r, coarse_cols - 1))
    edge_cells = list(edge_cells)
    random.shuffle(edge_cells)
    start_coarse = edge_cells[0]
    far = sorted(edge_cells, key=lambda e: -(abs(e[0] - start_coarse[0]) + abs(e[1] - start_coarse[1])))
    end_coarse = far[0] if far[0] != start_coarse else far[1]

    walk = _hamiltonian_walk(coarse_rows, coarse_cols, start_coarse, end_coarse)
    if walk is None or len(walk) < 5:
        return None
    if len(set(cr for cr, _ in walk)) < coarse_rows or len(set(cc for _, cc in walk)) < coarse_cols:
        return None

    grid = np.full((H, W), _BG, dtype=int)
    pixel_pos = _render_channel(grid, walk, n, offsets_r, offsets_c)

    mask = grid != _BG
    if np.any(mask, axis=1).sum() / H < 0.70 or np.any(mask, axis=0).sum() / W < 0.70:
        return None

    mover_r, mover_c = pixel_pos[0]
    dest_r, dest_c = pixel_pos[-1]

    result = _extend_to_edge(grid, dest_r, dest_c, n)
    if result is None:
        return None
    new_dest_r, new_dest_c = result
    if new_dest_r != dest_r:
        r_lo, r_hi = min(new_dest_r, dest_r), max(new_dest_r, dest_r) + n
        grid[r_lo:r_hi, dest_c : dest_c + n] = _RIVER
    if new_dest_c != dest_c:
        c_lo, c_hi = min(new_dest_c, dest_c), max(new_dest_c, dest_c) + n
        grid[dest_r : dest_r + n, c_lo:c_hi] = _RIVER
    dest_r, dest_c = new_dest_r, new_dest_c
    if not (dest_r == 0 or dest_r + n == H or dest_c == 0 or dest_c + n == W):
        return None

    touching = mover_r == 0 or mover_r + n == H or mover_c == 0 or mover_c + n == W
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

    num_targets = 3 if len(pixel_pos) >= 8 and random.random() < 0.5 else 2
    interior_target_indices = _sample_target_indices(len(pixel_pos), num_targets)
    if interior_target_indices is None:
        if num_targets == 3:
            num_targets = 2
            interior_target_indices = _sample_target_indices(len(pixel_pos), num_targets)
        if interior_target_indices is None:
            return None

    target_positions = [pixel_pos[idx] for idx in interior_target_indices] + [(dest_r, dest_c)]
    choke_anchor_indices = interior_target_indices + [len(pixel_pos) - 1]

    grid[mover_r : mover_r + n, mover_c : mover_c + n] = _RIVER
    for tr, tc in target_positions:
        grid[tr : tr + n, tc : tc + n] = _RIVER

    stop_after = num_targets
    if not all_reached:
        stop_after = random.randint(1, num_targets - 1)
        prev_anchor = choke_anchor_indices[stop_after - 1]
        next_anchor = choke_anchor_indices[stop_after]
        if next_anchor - prev_anchor < 1:
            return None
        seg_idx = random.randint(prev_anchor, next_anchor - 1)
        pr1, pc1 = pixel_pos[seg_idx]
        pr2, pc2 = pixel_pos[seg_idx + 1]
        direction = "h" if pr1 == pr2 else "v"
        _add_chokepoint(grid, (pr1, pc1), (pr2, pc2), n, direction)
        if not _river_connected(grid):
            return None

    grid[mover_r : mover_r + n, mover_c : mover_c + n] = _MOVER
    for tr, tc in target_positions:
        grid[tr : tr + n, tc : tc + n] = _DEST

    try:
        output, meta = _solve(grid, return_meta=True)
    except Exception:
        return None

    intended_positions = target_positions
    if meta["total_targets"] != num_targets:
        return None
    if meta["all_target_positions"] != sorted(intended_positions, key=lambda p: (p[0], p[1])):
        return None
    if meta["visited_positions"] != intended_positions[: meta["visited_count"]]:
        return None
    if all_reached:
        if meta["visited_count"] != num_targets or meta["stopped_early"]:
            return None
    else:
        if meta["visited_count"] != stop_after or not meta["stopped_early"]:
            return None
    if not np.array_equal(main(grid), output):
        return None
    return grid


def generate_input():
    for _ in range(400):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("Failed to generate valid input after 400 attempts")
