# Combo variant of 332f06d7 (index 38)
# combo_spec: D3/0+D5/0+D17/1
# Name: 连续轨迹+多目标选择+一对多目标

from __future__ import annotations

from collections import Counter, defaultdict, deque
import random

import numpy as np
from scipy.ndimage import label as scipy_label

from arc_utils import bbox_h, bbox_w, find_components


# Design analysis for the fused variant
# a) Concrete behavior and trace
#    - Input keeps the original river mechanics: a blue winding corridor on green background and one black n x n mover.
#    - The destination side is now fused in two ways.
#      1. There are multiple target squares on the river instead of one.
#      2. Targets come in multiple colors, but exactly one color is selected by an isolated same-colored bar in the
#         background.
#    - The selected color appears on 2-3 same-sized target squares placed in order along the main corridor. Those are
#      the only targets the mover should conceptually visit. Other same-sized target squares of different colors sit on
#      short side branches as reachable distractors.
#    - Output leaves a continuous black trail. Example: mover at A, selected-color targets at C and F on the main
#      corridor, a yellow distractor on a side spur near D. If F is reachable, the output paints every placement from
#      A to F black, covering C on the way but leaving the yellow spur target unchanged. If a chokepoint appears
#      between C and F, the trail runs from A through C and stops at the farthest main-corridor placement before the
#      choke; F stays in its target color and the yellow distractor still stays untouched.
#    - Relative to the base task, three stages change together: target identification uses a visible color cue, the
#      destination role expands from one square to several same-colored squares in order, and rendering changes from a
#      teleport-style final square to a full black path.
# b) Complexity calibration
#    - The object vocabulary stays small: one mover, 2-3 selected targets of one repeated color, 1-2 distractor
#      targets of distinct colors, and one tiny selector bar.
#    - The branch structure is limited to short side spurs so the main corridor still visually dominates and the visit
#      order remains legible. This avoids turning the task into arbitrary graph search.
#    - Repeating the selected color on both the selector bar and multiple targets gives strong support for the color
#      matching rule without relying on hidden numeric color ordering.
# c) Implicit assumptions broken from the base solver
#    - Base code assumed exactly two relevant same-sized squares and identified roles by sorting colors. That is false
#      here: there are many same-sized targets, some distractors, and only the black mover is unique by role.
#    - Base traversability used only river + mover + one destination. Here every target square must count as traversable
#      because targets occupy legal river placements, while the selector bar must not count as traversable because it is
#      isolated in the background.
#    - Base rendering wrote only source and final rectangles. That fails once the output must show the entire movement
#      trail. The fused solver therefore stores BFS parents and reconstructs the path to the final stop.
#    - Base logic had no need to distinguish "chosen targets" from "reachable but irrelevant distractors". The fused
#      solver assumes exactly one target color has an extra non-square component; the generator guarantees that.
# d) Placement legality
#    - All targets use the mover's size n and are placed on legal n x n river placements.
#    - Selected targets lie on the main corridor in increasing corridor order; distractors are placed only on branch tips
#      that are not on that main path.
#    - The selector bar is placed on a background patch with a full one-cell background moat so it cannot merge with the
#      river or any target.
#    - Chokepoints are inserted only on the main corridor after at least one selected target and before at least one
#      later selected target, preserving the intended reachable-prefix behavior.
# e) Visual observability
#    - A human can infer the rule from a few examples because the selected color is explicitly marked by the background
#      bar, that same color appears on multiple main-corridor targets, and the output black trail consumes exactly that
#      color's corridor prefix while leaving wrong-color branch targets untouched.
#    - No invisible ordering by numeric color id is used anywhere in the rule.
# f) Visual cleanliness
#    - The main corridor remains a single winding ribbon; side branches are short and sparse.
#    - The selector stays off the river, so there is no ambiguity between "path object" and "selection cue".
#    - Only one output color is added by motion, namely the mover's black trail, which keeps the final image readable.
# g) Color-change inferability
#    - The only color relation is equality: the selector bar color matches the chosen targets. Distractor colors do not
#      change. Black trail encodes movement, not a hidden color mapping.
#
# New solver assumptions and generator guarantees
# - Exactly one solid black n x n mover exists.
# - Exactly one target color has both square target components and one extra non-square selector component.
# - Selected-color targets are laid out along one main corridor so that the farthest reachable placement from the start
#   also represents the correct sequential stopping point when a later selected target is blocked.
# - Side-branch distractors are short enough that they do not become the farthest reachable placement in blocked cases.


_BG, _RIVER, _MOVER = 3, 1, 0
_TARGET_COLORS = [2, 4, 5, 6, 7, 8, 9]


def _is_full_square(grid: np.ndarray, comp: dict) -> bool:
    b = comp["bbox"]
    if bbox_h(b) != bbox_w(b):
        return False
    col = int(comp["color"])
    patch = grid[int(b[0]) : int(b[2]) + 1, int(b[1]) : int(b[3]) + 1]
    return bool((patch == col).all())


def _collect_square_components(grid: np.ndarray, bg: int) -> tuple[list[dict], dict[int, list[dict]], Counter]:
    comps = find_components(grid, bg=bg, connectivity=4, monochromatic=True, return_format="dict")
    squares: list[dict] = []
    by_color: dict[int, list[dict]] = defaultdict(list)
    counts: Counter = Counter()
    for comp in comps:
        col = int(comp["color"])
        counts[col] += 1
        by_color[col].append(comp)
        if not _is_full_square(grid, comp):
            continue
        b = comp["bbox"]
        squares.append({"col": col, "r0": int(b[0]), "c0": int(b[1]), "n": int(bbox_h(b))})
    return squares, by_color, counts


def _build_feasible_mask(trav: np.ndarray, n: int) -> np.ndarray:
    H, W = trav.shape
    h2, w2 = H - n + 1, W - n + 1
    feasible = np.zeros((h2, w2), dtype=bool)
    for r in range(h2):
        for c in range(w2):
            if bool(trav[r : r + n, c : c + n].all()):
                feasible[r, c] = True
    return feasible


def _bfs_with_parent(feasible: np.ndarray, sr: int, sc: int):
    h2, w2 = feasible.shape
    dist = -np.ones((h2, w2), dtype=int)
    parent_r = -np.ones((h2, w2), dtype=int)
    parent_c = -np.ones((h2, w2), dtype=int)
    q = deque([(sr, sc)])
    dist[sr, sc] = 0
    while q:
        r, c = q.popleft()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            rr, cc = r + dr, c + dc
            if 0 <= rr < h2 and 0 <= cc < w2 and feasible[rr, cc] and dist[rr, cc] == -1:
                dist[rr, cc] = dist[r, c] + 1
                parent_r[rr, cc] = r
                parent_c[rr, cc] = c
                q.append((rr, cc))
    return dist, parent_r, parent_c


def _reconstruct_path(parent_r: np.ndarray, parent_c: np.ndarray, sr: int, sc: int, tr: int, tc: int):
    path = []
    r, c = tr, tc
    while True:
        path.append((int(r), int(c)))
        if r == sr and c == sc:
            break
        pr = int(parent_r[r, c])
        pc = int(parent_c[r, c])
        assert pr >= 0 and pc >= 0
        r, c = pr, pc
    path.reverse()
    return path


def _find_farthest(dist: np.ndarray, sr: int, sc: int) -> tuple[int, int]:
    best = -1
    tr, tc = sr, sc
    for r in range(dist.shape[0]):
        for c in range(dist.shape[1]):
            cand = int(dist[r, c])
            if cand > best:
                best = cand
                tr, tc = int(r), int(c)
    return tr, tc


def _solve_core(grid: np.ndarray):
    g = np.array(grid, dtype=int)
    bg = int(Counter([int(g[0, 0]), int(g[0, -1]), int(g[-1, 0]), int(g[-1, -1])]).most_common(1)[0][0])
    squares, by_color, _component_counts = _collect_square_components(g, bg)

    movers = [sq for sq in squares if int(sq["col"]) == _MOVER]
    assert len(movers) == 1
    mover = movers[0]
    n = int(mover["n"])
    sr, sc = int(mover["r0"]), int(mover["c0"])

    targets = [sq for sq in squares if int(sq["col"]) != _MOVER and int(sq["n"]) == n]
    assert len(targets) >= 3
    target_colors = {int(sq["col"]) for sq in targets}

    selected_colors = []
    for col in sorted(target_colors):
        comps = by_color[int(col)]
        if any(not _is_full_square(g, comp) for comp in comps):
            selected_colors.append(int(col))
    assert len(selected_colors) == 1
    selected_color = int(selected_colors[0])
    selected_targets = [dict(sq) for sq in targets if int(sq["col"]) == selected_color]
    assert len(selected_targets) >= 2

    counts = Counter(map(int, g.flatten()))
    river = None
    for col, _ct in counts.most_common():
        if int(col) == bg or int(col) == _MOVER or int(col) in target_colors:
            continue
        river = int(col)
        break
    assert river is not None

    trav = (g == river)
    trav[sr : sr + n, sc : sc + n] = True
    for sq in targets:
        rr, cc = int(sq["r0"]), int(sq["c0"])
        trav[rr : rr + n, cc : cc + n] = True

    feasible = _build_feasible_mask(trav, n)
    assert bool(feasible[sr, sc])
    dist, parent_r, parent_c = _bfs_with_parent(feasible, sr, sc)

    reachable_selected = []
    unreachable_selected = []
    for sq in selected_targets:
        rr, cc = int(sq["r0"]), int(sq["c0"])
        ok = 0 <= rr < feasible.shape[0] and 0 <= cc < feasible.shape[1] and bool(feasible[rr, cc]) and int(dist[rr, cc]) >= 0
        sq2 = dict(sq)
        sq2["dist"] = int(dist[rr, cc]) if ok else -1
        (reachable_selected if ok else unreachable_selected).append(sq2)

    reachable_selected.sort(key=lambda sq: (int(sq["dist"]), int(sq["r0"]), int(sq["c0"])))
    unreachable_selected.sort(key=lambda sq: (int(sq["r0"]), int(sq["c0"])))

    if len(reachable_selected) == len(selected_targets):
        final_target = reachable_selected[-1]
        tr, tc = int(final_target["r0"]), int(final_target["c0"])
    else:
        tr, tc = _find_farthest(dist, sr, sc)

    path = _reconstruct_path(parent_r, parent_c, sr, sc, tr, tc)

    out = g.copy()
    for r, c in path:
        out[r : r + n, c : c + n] = _MOVER

    return out, {
        "bg": bg,
        "river": int(river),
        "n": n,
        "mover": mover,
        "targets": [dict(sq) for sq in targets],
        "selected_color": int(selected_color),
        "selected_targets": [dict(sq) for sq in selected_targets],
        "reachable_selected": reachable_selected,
        "unreachable_selected": unreachable_selected,
        "dist": dist,
        "path": path,
        "final_pos": (int(tr), int(tc)),
        "trail_area": int(np.count_nonzero(out == _MOVER)),
        "start_area": int(n * n),
    }


def main(grid: np.ndarray) -> np.ndarray:
    return _solve_core(np.array(grid, dtype=int))[0]


def _bfs_placement(feasible: np.ndarray, sr: int, sc: int) -> np.ndarray:
    dist = -np.ones(feasible.shape, dtype=int)
    dist[sr, sc] = 0
    q = deque([(sr, sc)])
    while q:
        r, c = q.popleft()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            rr, cc = r + dr, c + dc
            if 0 <= rr < feasible.shape[0] and 0 <= cc < feasible.shape[1] and feasible[rr, cc] and dist[rr, cc] == -1:
                dist[rr, cc] = dist[r, c] + 1
                q.append((rr, cc))
    return dist


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
        total_len = 2 * remove + max(1, hold_len)
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
        total_len = 2 * remove + max(1, hold_len)
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
    mask = grid == _RIVER
    if not mask.any():
        return False
    _labels, num = scipy_label(mask.astype(int))
    return bool(num == 1)


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
    return int(candidates[0][0]), int(candidates[0][1])


def _sample_main_target_indices(path_len: int, count: int) -> list[int] | None:
    if path_len < 5 or count < 2:
        return None
    last_idx = path_len - 1
    pool = list(range(2, last_idx - 1))
    random.shuffle(pool)
    chosen = [last_idx]
    for idx in pool:
        if all(abs(idx - old) >= 2 for old in chosen):
            chosen.append(idx)
            if len(chosen) == count:
                return sorted(chosen)
    return None


def _coarse_neighbors(cell: tuple[int, int], rows: int, cols: int):
    r, c = cell
    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        rr, cc = r + dr, c + dc
        if 0 <= rr < rows and 0 <= cc < cols:
            yield (rr, cc)


def _find_branch_paths(main_walk: list[tuple[int, int]], rows: int, cols: int, desired: int) -> list[list[tuple[int, int]]]:
    used = set(main_walk)
    branches: list[list[tuple[int, int]]] = []
    attach_indices = list(range(2, len(main_walk) - 2))
    random.shuffle(attach_indices)

    for attach_idx in attach_indices:
        if len(branches) >= desired:
            break
        attach = main_walk[attach_idx]
        first_steps = [nb for nb in _coarse_neighbors(attach, rows, cols) if nb not in used]
        random.shuffle(first_steps)
        picked = None
        for first in first_steps:
            if any(nb in used and nb != attach for nb in _coarse_neighbors(first, rows, cols)):
                continue
            path = [attach, first]
            local_used = used | {first}
            second_steps = [nb for nb in _coarse_neighbors(first, rows, cols) if nb not in local_used]
            random.shuffle(second_steps)
            for second in second_steps:
                if any(nb in local_used and nb != first for nb in _coarse_neighbors(second, rows, cols)):
                    continue
                path.append(second)
                break
            picked = path
            break
        if picked is None:
            continue
        branches.append(picked)
        used.update(picked[1:])
    return branches


def _find_selector_slot(grid: np.ndarray) -> tuple[int, int, int, int] | None:
    H, W = grid.shape
    shapes = [(1, 2), (2, 1)]
    random.shuffle(shapes)
    positions = [(r, c) for r in range(1, H - 1) for c in range(1, W - 1)]
    random.shuffle(positions)
    for h, w in shapes:
        for r, c in positions:
            if r + h >= H or c + w >= W:
                continue
            if not bool((grid[r : r + h, c : c + w] == _BG).all()):
                continue
            rr0, rr1 = r - 1, r + h + 1
            cc0, cc1 = c - 1, c + w + 1
            if rr0 < 0 or cc0 < 0 or rr1 > H or cc1 > W:
                continue
            if bool((grid[rr0:rr1, cc0:cc1] == _BG).all()):
                return r, c, h, w
    return None


def _gen_attempt() -> np.ndarray | None:
    all_selected_reachable = random.random() < 0.5
    n = random.choice([1, 2, 3, 4]) if all_selected_reachable else random.choice([2, 3, 4])

    min_size = max(10, 5 * n)
    if min_size > 20:
        return None
    G = random.randint(min_size, 20)
    H = W = G

    min_spacing = n + 1
    max_coarse = max(4, (G - 2) // min_spacing)
    coarse_dim = random.randint(4, min(6, max_coarse))
    offsets_r = _compute_offsets(G, n, coarse_dim)
    offsets_c = _compute_offsets(G, n, coarse_dim)
    if offsets_r is None or offsets_c is None:
        return None
    if any(offsets_r[i + 1] - offsets_r[i] < n for i in range(len(offsets_r) - 1)):
        return None
    if any(offsets_c[i + 1] - offsets_c[i] < n for i in range(len(offsets_c) - 1)):
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

    main_walk = _hamiltonian_walk(coarse_dim, coarse_dim, start_coarse, end_coarse)
    if main_walk is None or len(main_walk) < 7:
        return None

    selected_count = 3 if len(main_walk) >= 10 and random.random() < 0.7 else 2
    selected_indices = _sample_main_target_indices(len(main_walk), selected_count)
    if selected_indices is None:
        return None

    branch_count = random.randint(1, 2)
    branches = _find_branch_paths(main_walk, coarse_dim, coarse_dim, branch_count)
    if len(branches) < 1:
        return None

    grid = np.full((H, W), _BG, dtype=int)
    main_pixel = _render_channel(grid, main_walk, n, offsets_r, offsets_c)
    branch_pixels = []
    for branch in branches:
        branch_pixels.append(_render_channel(grid, branch, n, offsets_r, offsets_c))

    if np.any(grid != _BG, axis=1).sum() / H < 0.65 or np.any(grid != _BG, axis=0).sum() / W < 0.65:
        return None

    mover_r, mover_c = main_pixel[0]
    end_r, end_c = main_pixel[-1]
    final_edge = _extend_to_edge(grid, end_r, end_c, n)
    if final_edge is None:
        return None
    new_end_r, new_end_c = final_edge
    if new_end_r != end_r:
        r_lo, r_hi = min(new_end_r, end_r), max(new_end_r, end_r) + n
        grid[r_lo:r_hi, end_c : end_c + n] = _RIVER
    if new_end_c != end_c:
        c_lo, c_hi = min(new_end_c, end_c), max(new_end_c, end_c) + n
        grid[end_r : end_r + n, c_lo:c_hi] = _RIVER
    main_pixel = list(main_pixel)
    main_pixel[-1] = (new_end_r, new_end_c)

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

    if not all_selected_reachable:
        selected_prefix = random.randint(1, len(selected_indices) - 1)
        left_idx = selected_indices[selected_prefix - 1]
        right_idx = selected_indices[selected_prefix]
        choke_candidates = list(range(left_idx, right_idx))
        random.shuffle(choke_candidates)
        added = False
        for seg_idx in choke_candidates:
            before = grid.copy()
            pr1, pc1 = main_pixel[seg_idx]
            pr2, pc2 = main_pixel[seg_idx + 1]
            direction = "h" if pr1 == pr2 else "v"
            _add_chokepoint(grid, (pr1, pc1), (pr2, pc2), n, direction)
            if _river_connected(grid):
                added = True
                break
            grid = before
        if not added:
            return None

    selected_color = random.choice(_TARGET_COLORS)
    other_colors = [col for col in _TARGET_COLORS if col != selected_color]
    random.shuffle(other_colors)

    target_specs = []
    for idx in selected_indices:
        rr, cc = main_pixel[idx]
        target_specs.append({"kind": "selected", "col": selected_color, "r": rr, "c": cc, "main_idx": idx})

    for branch_pix, color in zip(branch_pixels, other_colors):
        rr, cc = branch_pix[-1]
        target_specs.append({"kind": "distractor", "col": int(color), "r": rr, "c": cc, "main_idx": -1})

    for spec in target_specs:
        rr, cc = int(spec["r"]), int(spec["c"])
        grid[rr : rr + n, cc : cc + n] = int(spec["col"])

    marker = _find_selector_slot(grid)
    if marker is None:
        return None
    mr, mc, mh, mw = marker
    grid[mr : mr + mh, mc : mc + mw] = int(selected_color)
    grid[mover_r : mover_r + n, mover_c : mover_c + n] = _MOVER

    try:
        out, info = _solve_core(grid)
    except Exception:
        return None

    selected_targets = sorted(
        [spec for spec in target_specs if spec["kind"] == "selected"],
        key=lambda spec: int(spec["main_idx"]),
    )
    distractors = [spec for spec in target_specs if spec["kind"] == "distractor"]
    path_set = set(info["path"])

    reachable_selected_count = 0
    for spec in selected_targets:
        if (int(spec["r"]), int(spec["c"])) in path_set:
            reachable_selected_count += 1
        else:
            break
    if any((int(spec["r"]), int(spec["c"])) in path_set for spec in selected_targets[reachable_selected_count:]):
        return None

    if all_selected_reachable:
        if reachable_selected_count != len(selected_targets):
            return None
    else:
        if not (1 <= reachable_selected_count < len(selected_targets)):
            return None

    for spec in selected_targets[:reachable_selected_count]:
        rr, cc = int(spec["r"]), int(spec["c"])
        if not bool((out[rr : rr + n, cc : cc + n] == _MOVER).all()):
            return None
    for spec in selected_targets[reachable_selected_count:]:
        rr, cc = int(spec["r"]), int(spec["c"])
        if not bool((out[rr : rr + n, cc : cc + n] == int(spec["col"])).all()):
            return None

    distractor_preserved = 0
    for spec in distractors:
        rr, cc = int(spec["r"]), int(spec["c"])
        if (rr, cc) in path_set:
            return None
        if bool((out[rr : rr + n, cc : cc + n] == int(spec["col"])).all()):
            distractor_preserved += 1
    if distractor_preserved < 1:
        return None

    if info["trail_area"] <= info["start_area"]:
        return None

    if not np.array_equal(main(grid), out):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(800):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("Failed to generate valid input after 800 attempts")
