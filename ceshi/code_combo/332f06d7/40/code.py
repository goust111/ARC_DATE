# Combo variant of 332f06d7 (index 40)
# combo_spec: D3/2+D7/1
# Name: 渐变色轨迹+裁剪通道内部
#
# Design analysis
# ===============
# a) Concrete behavior and trace
# - Input remains the original scene type: green background, one blue winding channel,
#   one black n x n mover square, and one red n x n destination square.
# - The motion logic is still "slide an n x n window through the traversable channel by
#   BFS on feasible top-left placements". The key fusion is in output rendering.
# - First compute the exact placement path from the mover start to the chosen stop:
#   destination if reachable, otherwise the farthest reachable feasible placement.
# - Every vacated placement on that realized path is recolored with a 3-band distance
#   gradient: start-near = 4, middle = 8, end-near = 6. The final stop remains black.
# - After that full-size rendered output is produced, crop to the tight bounding box of
#   all non-background pixels, removing the outer green border.
# - Small trace in placement coordinates for n=2:
#   path = [(2,2),(2,3),(3,3),(4,3)], target = (4,3).
#   Original output: first placement restored to blue, final placement black, full grid kept.
#   Combo output: placements (2,2),(2,3),(3,3) become 4,8,6 respectively, (4,3) stays black,
#   then the outer green border is cropped away. The crop makes the gradient occupy a larger
#   fraction of the visible output, so the distance encoding is easier to see.
#
# b) Complexity calibration
# - The gradient uses exactly three fixed colors. That is enough to expose distance coding
#   without introducing an arbitrary per-step palette that would be hard to infer.
# - The generator rejects short-motion cases where fewer than 3 vacated placements exist,
#   because those would hide part of the gradient and weaken the D3 signal.
# - The crop is always active: the channel is generated strictly inside a positive green
#   margin, so the output is smaller than the input in both dimensions.
#
# c) Implicit-assumption audit against the original main()
# - Square detection by monochromatic connected components still works because inputs use
#   only {green, blue, black, red}; trail colors appear only in outputs.
# - River inference as the most common non-bg non-square color still works because blue is
#   the only such input color.
# - Original BFS distance alone was enough for "where does the square stop", but not for
#   "which exact placements were traversed". The fused variant therefore stores parents and
#   reconstructs the realized path. New assumption: the chosen target must be reachable from
#   the start in this BFS tree. generate_input() guarantees that because the target is either
#   the reachable destination or a reachable max-distance placement.
# - Original unconditional refill of the start square with blue would erase the trail, so
#   output rendering must write the trajectory first and then place the final black square.
# - D7 adds one more postcondition: the intended output is the tight crop of the rendered
#   non-background structure, not the full-size grid.
#
# d) Placement legality
# - The generator maps coarse walk cells to pixel positions with a positive outer margin, so
#   the whole channel lies strictly inside the grid and cropping is well-defined.
# - Mover and destination are placed on already-blue channel cells and never overlap.
# - Chokepoints are carved only from existing river cells, preserving the intended geometry.
#
# e) Visual observability
# - Across a few examples, a human sees two coupled effects at once:
#   1. the black block moves according to width constraints,
#   2. the traversed portion of the river is re-encoded by an ordered 4->8->6 trail,
#      and the surrounding green frame disappears.
# - The ordering signal is geometric distance along the path, not hidden numeric properties
#   of the input colors.
#
# f) Visual cleanliness
# - The trail is confined to the mover footprint along the actual path, so the untouched river
#   stays blue and readable.
# - Cropping removes empty exterior border instead of adding more marks, which makes the final
#   image denser but not noisier.
#
# g) Color-change inferability
# - Only three trail colors are introduced, always in the same path-order role:
#   start-near = 4, middle = 8, end-near = 6.
# - The generator enforces that all three colors are visible in the cropped output and that
#   the combo output differs from the D7-only cropped baseline, so the fusion cannot silently
#   degenerate back to "crop only".
#
# h) Quantified non-degeneration target used in validation
# - crop_delta = (input_h - output_h, input_w - output_w) must be positive in both dimensions.
# - trail_steps must be >= 3 and the cropped output must contain all three trail colors.
# - combo_vs_crop_baseline_diff = number of cells that differ between this combo output and the
#   plain D7 cropped output must be > 0. This proves the D3 component is active on the sample.
#
# i) New main() assumptions guaranteed by generate_input()
# - Inputs use only colors {0,1,2,3}; outputs may additionally use {4,8,6}.
# - The channel has at least one-cell green margin on all four sides.
# - The actual motion path has at least 3 vacated placements, so the 3-band trail is visible.
# - Neighbor order is fixed, so BFS parent reconstruction is deterministic.

from __future__ import annotations

from collections import Counter, defaultdict, deque
import random

import numpy as np
from scipy.ndimage import label as scipy_label

from arc_utils import bbox_h, bbox_w, find_components


_BG, _RIVER, _MOVER, _DEST = 3, 1, 0, 2
_TRAIL_COLORS = (4, 8, 6)
_DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _find_squares(g: np.ndarray, bg: int) -> list[dict]:
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


def _extract_scene(grid: np.ndarray) -> dict:
    g = np.array(grid, dtype=int)
    H, W = g.shape

    bg = int(Counter([int(g[0, 0]), int(g[0, -1]), int(g[-1, 0]), int(g[-1, -1])]).most_common(1)[0][0])
    squares = _find_squares(g, bg)
    assert len(squares) >= 2

    by_n: dict[int, list[dict]] = defaultdict(list)
    for sq in squares:
        by_n[int(sq["n"])].append(sq)
    nside = max(by_n.items(), key=lambda kv: (len(kv[1]), kv[0]))[0]
    cand = by_n[int(nside)]
    assert len(cand) >= 2
    cand.sort(key=lambda s: int(s["col"]))
    mover = cand[0]
    dest = cand[1]

    counts = Counter(map(int, g.flatten()))
    river = None
    for col, _ct in counts.most_common():
        if int(col) == int(bg):
            continue
        if int(col) in (int(mover["col"]), int(dest["col"])):
            continue
        river = int(col)
        break
    assert river is not None

    return {
        "grid": g,
        "shape": (H, W),
        "bg": bg,
        "river": int(river),
        "mover": mover,
        "dest": dest,
    }


def _build_feasible(trav: np.ndarray, n: int) -> np.ndarray:
    H, W = trav.shape
    h2 = H - n + 1
    w2 = W - n + 1
    assert h2 > 0 and w2 > 0
    feasible = np.zeros((h2, w2), dtype=bool)
    for r in range(h2):
        for c in range(w2):
            if bool(trav[r : r + n, c : c + n].all()):
                feasible[r, c] = True
    return feasible


def _bfs_with_parent(
    feasible: np.ndarray, sr: int, sc: int
) -> tuple[np.ndarray, dict[tuple[int, int], tuple[int, int] | None]]:
    h2, w2 = feasible.shape
    dist = -np.ones((h2, w2), dtype=int)
    dist[sr, sc] = 0
    parent: dict[tuple[int, int], tuple[int, int] | None] = {(sr, sc): None}
    q = deque([(sr, sc)])
    while q:
        r, c = q.popleft()
        for dr, dc in _DIR4:
            rr, cc = r + dr, c + dc
            if 0 <= rr < h2 and 0 <= cc < w2 and feasible[rr, cc] and dist[rr, cc] == -1:
                dist[rr, cc] = dist[r, c] + 1
                parent[(rr, cc)] = (r, c)
                q.append((rr, cc))
    return dist, parent


def _choose_target(
    dist: np.ndarray, feasible: np.ndarray, sr: int, sc: int, dr: int, dc: int
) -> tuple[tuple[int, int], bool]:
    h2, w2 = feasible.shape
    dest_ok = 0 <= dr < h2 and 0 <= dc < w2 and bool(feasible[dr, dc]) and int(dist[dr, dc]) >= 0
    if dest_ok:
        return (dr, dc), True

    best = -1
    tr, tc = sr, sc
    for r in range(h2):
        for c in range(w2):
            if int(dist[r, c]) > best:
                best = int(dist[r, c])
                tr, tc = r, c
    return (tr, tc), False


def _reconstruct_path(
    parent: dict[tuple[int, int], tuple[int, int] | None], target: tuple[int, int]
) -> list[tuple[int, int]]:
    path: list[tuple[int, int]] = []
    cur: tuple[int, int] | None = target
    while cur is not None:
        path.append(cur)
        cur = parent[cur]
    path.reverse()
    return path


def _solve_path_info(grid: np.ndarray) -> dict:
    scene = _extract_scene(grid)
    g = scene["grid"]
    mover = scene["mover"]
    dest = scene["dest"]
    river = int(scene["river"])

    n = int(mover["n"])
    sr, sc = int(mover["r0"]), int(mover["c0"])
    dr, dc = int(dest["r0"]), int(dest["c0"])

    trav = (g == river) | (g == int(mover["col"])) | (g == int(dest["col"]))
    feasible = _build_feasible(trav, n)
    assert 0 <= sr < feasible.shape[0] and 0 <= sc < feasible.shape[1]
    assert bool(feasible[sr, sc])

    dist, parent = _bfs_with_parent(feasible, sr, sc)
    target, reached_dest = _choose_target(dist, feasible, sr, sc, dr, dc)
    assert target in parent
    path = _reconstruct_path(parent, target)

    return {
        **scene,
        "n": n,
        "start": (sr, sc),
        "dest_pos": (dr, dc),
        "target": target,
        "reached_dest": reached_dest,
        "feasible": feasible,
        "dist": dist,
        "path": path,
    }


def _trail_color(step_idx: int, trail_steps: int) -> int:
    if trail_steps <= 1:
        return int(_TRAIL_COLORS[0])
    band = (step_idx * (len(_TRAIL_COLORS) - 1)) // (trail_steps - 1)
    return int(_TRAIL_COLORS[band])


def _render_baseline_full_output(info: dict) -> np.ndarray:
    out = np.array(info["grid"], dtype=int).copy()
    n = int(info["n"])
    sr, sc = info["start"]
    tr, tc = info["target"]
    out[sr : sr + n, sc : sc + n] = int(info["river"])
    out[tr : tr + n, tc : tc + n] = int(info["mover"]["col"])
    return out


def _render_combo_full_output(info: dict) -> np.ndarray:
    out = np.array(info["grid"], dtype=int).copy()
    n = int(info["n"])
    trail_steps = max(0, len(info["path"]) - 1)

    for step_idx, (r, c) in enumerate(info["path"][:-1]):
        out[r : r + n, c : c + n] = _trail_color(step_idx, trail_steps)

    tr, tc = info["target"]
    out[tr : tr + n, tc : tc + n] = int(info["mover"]["col"])
    return out


def _crop_to_non_bg(grid: np.ndarray, bg: int) -> np.ndarray:
    mask = grid != int(bg)
    assert bool(mask.any())
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    return grid[int(rows[0]) : int(rows[-1]) + 1, int(cols[0]) : int(cols[-1]) + 1]


def _trail_stats(info: dict, cropped_output: np.ndarray | None = None) -> dict:
    trail_steps = max(0, len(info["path"]) - 1)
    used = {_trail_color(step_idx, trail_steps) for step_idx in range(trail_steps)}
    visible = set()
    if cropped_output is not None:
        visible = set(map(int, np.unique(cropped_output))) & set(_TRAIL_COLORS)
    return {"trail_steps": trail_steps, "used_colors": used, "visible_colors": visible}


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide the square as in 332f06d7, leave a 3-band distance trail on vacated placements,
    then crop away the outer green background.
    """
    info = _solve_path_info(np.array(grid, dtype=int))
    full = _render_combo_full_output(info)
    return _crop_to_non_bg(full, int(info["bg"]))


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


def _render_channel(
    grid: np.ndarray, path: list[tuple[int, int]], n: int, offsets_r: list[int], offsets_c: list[int]
) -> list[tuple[int, int]]:
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


def _add_chokepoint(
    grid: np.ndarray, pixel_pos_i: tuple[int, int], pixel_pos_j: tuple[int, int], n: int, direction: str
) -> None:
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
    if not bool(mask.any()):
        return False
    _labeled, num = scipy_label(mask.astype(int))
    return int(num) == 1


def _compute_offsets(G: int, n: int, num_coarse: int, margin: int) -> list[int] | None:
    if num_coarse == 1:
        return [margin]
    first = margin
    last = G - n - margin
    if last < first:
        return None
    step = (last - first) / (num_coarse - 1)
    return [int(round(first + i * step)) for i in range(num_coarse)]


def _gen_attempt() -> np.ndarray | None:
    reached = random.random() < 0.5
    n = random.choice([1, 2, 3, 4]) if reached else random.choice([2, 3, 4])

    min_size = max(10, 5 * n)
    if min_size > 20:
        return None
    G = random.randint(min_size, 20)
    H = W = G

    max_margin = max(1, min(3, (G - n - 3) // 2))
    if max_margin < 1:
        return None
    margin = random.randint(1, max_margin)

    min_spacing = n + 1
    usable = G - 2 * margin
    max_coarse = max(3, usable // min_spacing)
    coarse_dim = random.randint(3, min(6, max_coarse))
    coarse_rows = coarse_cols = coarse_dim

    offsets_r = _compute_offsets(G, n, coarse_rows, margin)
    offsets_c = _compute_offsets(G, n, coarse_cols, margin)
    if offsets_r is None or offsets_c is None:
        return None
    for i in range(len(offsets_r) - 1):
        if offsets_r[i + 1] - offsets_r[i] < n:
            return None
    for i in range(len(offsets_c) - 1):
        if offsets_c[i + 1] - offsets_c[i] < n:
            return None

    edge_cells = set()
    for c in range(coarse_cols):
        edge_cells.add((0, c))
        edge_cells.add((coarse_rows - 1, c))
    for r in range(1, coarse_rows - 1):
        edge_cells.add((r, 0))
        edge_cells.add((r, coarse_cols - 1))
    edge_list = list(edge_cells)
    random.shuffle(edge_list)
    start_coarse = edge_list[0]
    far = sorted(edge_list, key=lambda e: -(abs(e[0] - start_coarse[0]) + abs(e[1] - start_coarse[1])))
    end_coarse = far[0] if far[0] != start_coarse else far[1]

    walk = _hamiltonian_walk(coarse_rows, coarse_cols, start_coarse, end_coarse)
    if walk is None or len(walk) < 4:
        return None
    if len({cr for cr, _ in walk}) < coarse_rows or len({cc for _, cc in walk}) < coarse_cols:
        return None

    grid = np.full((H, W), _BG, dtype=int)
    pixel_pos = _render_channel(grid, walk, n, offsets_r, offsets_c)

    mover_r, mover_c = pixel_pos[0]
    dest_r, dest_c = pixel_pos[-1]

    grid[mover_r : mover_r + n, mover_c : mover_c + n] = _RIVER
    grid[dest_r : dest_r + n, dest_c : dest_c + n] = _RIVER

    if not reached and n >= 2:
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

    grid[mover_r : mover_r + n, mover_c : mover_c + n] = _MOVER
    grid[dest_r : dest_r + n, dest_c : dest_c + n] = _DEST

    if set(map(int, np.unique(grid))) - {_BG, _RIVER, _MOVER, _DEST}:
        return None

    info = _solve_path_info(grid)
    if reached != bool(info["reached_dest"]):
        return None

    combo_full = _render_combo_full_output(info)
    combo_cropped = _crop_to_non_bg(combo_full, int(info["bg"]))
    base_full = _render_baseline_full_output(info)
    base_cropped = _crop_to_non_bg(base_full, int(info["bg"]))
    trail_info = _trail_stats(info, combo_cropped)

    if trail_info["trail_steps"] < len(_TRAIL_COLORS):
        return None
    if trail_info["used_colors"] != set(_TRAIL_COLORS):
        return None
    if trail_info["visible_colors"] != set(_TRAIL_COLORS):
        return None
    if combo_cropped.shape[0] >= H or combo_cropped.shape[1] >= W:
        return None
    if combo_cropped.shape != base_cropped.shape:
        return None
    if np.array_equal(combo_cropped, base_cropped):
        return None
    if not np.array_equal(main(grid), combo_cropped):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(300):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("Failed to generate valid input after 300 attempts")
