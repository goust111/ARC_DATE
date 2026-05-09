# Combo variant of 332f06d7 (index 14)
# combo_spec: D6/2+D19/2
# Name: 变色为目标色+颜色线索可达性
#
# Design analysis for the fused variant
# a) Concrete behavior and trace:
#    - Keep the original scene grammar: green background, one winding corridor, one black n×n mover,
#      one red n×n destination of the same size.
#    - The corridor now uses two blues:
#      - dark blue (1) = effectively wide enough for any n;
#      - light blue (8) = effectively width 1.
#    - Therefore a 1×1 mover may traverse both blues, while an n>=2 mover may only traverse dark blue.
#    - Reachable branch also changes: when the mover reaches the destination, it turns red instead of
#      overwriting the target as black.
#    - Small trace:
#      - Original task on a physically wide 2×2 corridor with no erased chokepoint: the black square
#        reaches the red target and the output target becomes black.
#      - Fused task on the same wide geometry but with the downstream suffix recolored light blue:
#        the 2×2 square stops at the last dark-blue placement before the light-blue suffix.
#      - If the mover is 1×1 on that same colored suffix, it still reaches the target, and the output
#        target remains red because arrival recolors the mover to the destination color.
#    - The two changes interact rather than sit side by side: color decides whether the arrival branch
#      is taken, and the arrival branch decides whether the mover becomes red.
# b) Complexity calibration:
#    - Only one extra input degree of freedom is added: corridor color state {dark blue, light blue}.
#    - Only one extra output rule is added: successful arrival recolors the mover to red.
#    - The clues repeat within each sample as long light-blue suffixes, so the rule is observable without
#      introducing many independent marker types.
# c) Hidden assumptions audit against the original main():
#    - Original `river = most common remaining color` is no longer valid because the corridor may use two
#      colors. The fused solver uses explicit corridor semantics instead of a single inferred river color.
#    - Original `trav = river ∪ mover ∪ dest` is invalid because not every river-colored pixel is equally
#      traversable anymore; traversability depends on mover size and corridor shade.
#    - Original final paint `terminal = mover color` is invalid because successful arrival must now paint
#      the destination color instead.
#    - Original size-pair selection by “most common square size” can become brittle once a second river
#      color exists. The fused solver first looks for the black mover and red destination explicitly, then
#      only falls back to the generic square-pair heuristic if needed.
# d) Placement legality:
#    - The generator first paints a fully wide, physically valid corridor, then recolors a downstream
#      suffix to light blue; recoloring cannot create overlaps or out-of-bounds placements.
#    - Mover and destination are placed on pre-cleared corridor blocks, same size, non-overlapping.
#    - The destination is extended to the boundary, and the mover stays near an edge but not touching it,
#      matching the original task family.
# e) Visual observability:
#    - From a few examples, a human can compare:
#      - large mover + all dark blue => reaches and ends red,
#      - large mover + light-blue suffix => stops before the suffix,
#      - 1×1 mover + light-blue suffix => still reaches and ends red.
#    - No invisible ordering such as “smaller color id wins” is needed to infer the fused rule.
# f) Visual cleanliness:
#    - The scene still has only five meaningful colors: green, dark blue, light blue, black, red.
#    - No overlays or extra symbols are stacked on top of the corridor; the clue is encoded directly in
#      corridor color, so the moved object and the channel remain easy to parse.
# g) Color-change inferability:
#    - Arrival always maps black -> red, and only on the successful-arrival branch.
#    - That recolor is structurally anchored by the red destination already present in the input.
# h) Degeneration prevention:
#    - For intended stuck samples with n>=2, the generator explicitly verifies that if light blue were
#      reinterpreted as dark blue, the destination would become reachable. So the sample differs from the
#      original task for a real semantic reason, not because of accidental geometry.
# i) New assumptions checked against the generator:
#    - The mover always starts on dark blue, so clearing its source back to dark blue is valid.
#    - Light-blue cells, when used, form a contiguous downstream suffix that explains the stop position.
#    - Reachable outputs contain no black mover; unreachable outputs keep a black mover and red target.

from __future__ import annotations

from collections import Counter, defaultdict, deque
import random

import numpy as np

from arc_utils import bbox_h, bbox_w, find_components


_BG = 3
_WIDE = 1
_NARROW = 8
_MOVER = 0
_DEST = 2


def _passable_mask(grid: np.ndarray, n: int, mover_col: int, dest_col: int) -> np.ndarray:
    mask = (grid == _WIDE) | (grid == mover_col) | (grid == dest_col)
    if n == 1:
        mask |= grid == _NARROW
    return mask


def _compute_feasible(grid: np.ndarray, n: int, mover_col: int, dest_col: int) -> np.ndarray:
    H, W = grid.shape
    h2, w2 = H - n + 1, W - n + 1
    assert not (h2 <= 0 or w2 <= 0)
    trav = _passable_mask(grid, n, mover_col, dest_col)
    feasible = np.zeros((h2, w2), dtype=bool)
    for r in range(h2):
        for c in range(w2):
            if bool(trav[r : r + n, c : c + n].all()):
                feasible[r, c] = True
    return feasible


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


def _dest_reachable(feasible: np.ndarray, dist: np.ndarray, dr: int, dc: int) -> bool:
    h2, w2 = feasible.shape
    return bool(0 <= dr < h2 and 0 <= dc < w2 and feasible[dr, dc] and dist[dr, dc] >= 0)


def _pick_terminal(dist: np.ndarray, feasible: np.ndarray, dr: int, dc: int, sr: int, sc: int) -> tuple[int, int]:
    if _dest_reachable(feasible, dist, dr, dc):
        return dr, dc
    best = -1
    tr, tc = sr, sc
    for r in range(feasible.shape[0]):
        for c in range(feasible.shape[1]):
            if int(dist[r, c]) > best:
                best = int(dist[r, c])
                tr, tc = r, c
    return tr, tc


def _detect_squares(grid: np.ndarray, bg: int) -> list[dict]:
    squares: list[dict] = []
    for comp in find_components(grid, bg=bg, connectivity=4, monochromatic=True, return_format="dict"):
        col = int(comp["color"])
        b = comp["bbox"]
        if bbox_h(b) != bbox_w(b):
            continue
        patch = grid[int(b[0]) : int(b[2]) + 1, int(b[1]) : int(b[3]) + 1]
        if not bool((patch == col).all()):
            continue
        squares.append({"col": col, "r0": int(b[0]), "c0": int(b[1]), "n": int(bbox_h(b))})
    return squares


def _pick_mover_and_dest(squares: list[dict]) -> tuple[dict, dict]:
    mover = next((sq for sq in squares if int(sq["col"]) == _MOVER), None)
    dest = next((sq for sq in squares if int(sq["col"]) == _DEST), None)
    if mover is not None and dest is not None and int(mover["n"]) == int(dest["n"]):
        return mover, dest

    by_n: dict[int, list[dict]] = defaultdict(list)
    for sq in squares:
        by_n[int(sq["n"])].append(sq)
    assert not (not by_n)
    nside = max(by_n.items(), key=lambda kv: (len(kv[1]), kv[0]))[0]
    cand = sorted(by_n[int(nside)], key=lambda sq: int(sq["col"]))
    assert not (len(cand) < 2)
    return cand[0], cand[1]


def _analyze_scene(grid: np.ndarray) -> dict:
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(Counter([int(g[0, 0]), int(g[0, -1]), int(g[-1, 0]), int(g[-1, -1])]).most_common(1)[0][0])

    squares = _detect_squares(g, bg)
    assert not (len(squares) < 2)
    mover, dest = _pick_mover_and_dest(squares)

    n = int(mover["n"])
    sr, sc = int(mover["r0"]), int(mover["c0"])
    dr, dc = int(dest["r0"]), int(dest["c0"])

    feasible = _compute_feasible(g, n, int(mover["col"]), int(dest["col"]))
    h2, w2 = feasible.shape
    assert not (not (0 <= sr < h2 and 0 <= sc < w2))
    assert not (not bool(feasible[sr, sc]))

    dist = _bfs_placement(feasible, sr, sc)
    reached = _dest_reachable(feasible, dist, dr, dc)
    tr, tc = _pick_terminal(dist, feasible, dr, dc, sr, sc)

    return {
        "grid": g,
        "H": H,
        "W": W,
        "bg": bg,
        "mover": mover,
        "dest": dest,
        "n": n,
        "start": (sr, sc),
        "terminal": (tr, tc),
        "reached": reached,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fused rule:
    - Dark blue is always traversable.
    - Light blue is traversable only for a 1x1 mover.
    - If the destination is reached, the mover adopts the destination color (red).
    - Otherwise it stops at the farthest reachable placement and stays black.
    - The source block is restored to dark blue.
    """
    info = _analyze_scene(grid)
    out = info["grid"].copy()

    n = int(info["n"])
    sr, sc = info["start"]
    tr, tc = info["terminal"]
    mover_col = int(info["mover"]["col"])
    dest_col = int(info["dest"]["col"])

    out[sr : sr + n, sc : sc + n] = _WIDE
    out[tr : tr + n, tc : tc + n] = dest_col if info["reached"] else mover_col
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
                        if (
                            0 <= nr2 < rows
                            and 0 <= nc2 < cols
                            and (nr2, nc2) not in test_visited
                            and (nr2, nc2) not in seen
                        ):
                            seen.add((nr2, nc2))
                            q.append((nr2, nc2))
                for dr2, dc2 in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ar, ac = nb[0] + dr2, nb[1] + dc2
                    if (
                        0 <= ar < rows
                        and 0 <= ac < cols
                        and (ar, ac) not in test_visited
                        and (ar, ac) in seen
                    ):
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


def _compute_offsets(G: int, n: int, num_coarse: int) -> list[int] | None:
    if num_coarse == 1:
        return [1]
    first = 1
    last = G - n - 1
    if last < first:
        return None
    step = (last - first) / (num_coarse - 1)
    return [int(round(first + i * step)) for i in range(num_coarse)]


def _paint_bridge(grid: np.ndarray, p1: tuple[int, int], p2: tuple[int, int], n: int, color: int) -> None:
    pr1, pc1 = p1
    pr2, pc2 = p2
    if pr1 == pr2:
        c_lo, c_hi = min(pc1, pc2), max(pc1, pc2) + n
        grid[pr1 : pr1 + n, c_lo:c_hi] = color
    else:
        r_lo, r_hi = min(pr1, pr2), max(pr1, pr2) + n
        grid[r_lo:r_hi, pc1 : pc1 + n] = color


def _render_channel(
    grid: np.ndarray, path: list[tuple[int, int]], n: int, offsets_r: list[int], offsets_c: list[int]
) -> list[tuple[int, int]]:
    pixel_pos = []
    for cr, cc in path:
        pr, pc = offsets_r[cr], offsets_c[cc]
        pixel_pos.append((pr, pc))
        grid[pr : pr + n, pc : pc + n] = _WIDE
    for i in range(len(pixel_pos) - 1):
        _paint_bridge(grid, pixel_pos[i], pixel_pos[i + 1], n, _WIDE)
    return pixel_pos


def _extend_to_edge(grid: np.ndarray, pr: int, pc: int, n: int) -> tuple[int, int] | None:
    H, W = grid.shape
    corridor_colors = {_WIDE, _NARROW}
    candidates = []
    for direction in ("top", "bottom", "left", "right"):
        if direction == "top":
            dist = pr
            if dist == 0:
                candidates.append((0, pc, 0))
                continue
            if all(grid[r, c] in corridor_colors | {_BG} for r in range(pr) for c in range(pc, pc + n) if c < W):
                candidates.append((0, pc, dist))
        elif direction == "bottom":
            dist = H - (pr + n)
            if dist == 0:
                candidates.append((pr, pc, 0))
                continue
            if all(
                grid[r, c] in corridor_colors | {_BG}
                for r in range(pr + n, H)
                for c in range(pc, pc + n)
                if c < W
            ):
                candidates.append((H - n, pc, dist))
        elif direction == "left":
            dist = pc
            if dist == 0:
                candidates.append((pr, 0, 0))
                continue
            if all(grid[r, c] in corridor_colors | {_BG} for r in range(pr, pr + n) for c in range(pc) if r < H):
                candidates.append((pr, 0, dist))
        else:
            dist = W - (pc + n)
            if dist == 0:
                candidates.append((pr, pc, 0))
                continue
            if all(
                grid[r, c] in corridor_colors | {_BG}
                for r in range(pr, pr + n)
                for c in range(pc + n, W)
                if r < H
            ):
                candidates.append((pr, W - n, dist))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[2])
    return int(candidates[0][0]), int(candidates[0][1])


def _color_suffix(
    grid: np.ndarray,
    pixel_pos: list[tuple[int, int]],
    cut: int,
    n: int,
    dest_pos: tuple[int, int],
) -> None:
    for idx in range(cut, len(pixel_pos)):
        pr, pc = pixel_pos[idx]
        grid[pr : pr + n, pc : pc + n] = _NARROW
    for idx in range(max(0, cut - 1), len(pixel_pos) - 1):
        _paint_bridge(grid, pixel_pos[idx], pixel_pos[idx + 1], n, _NARROW)
    last = pixel_pos[-1]
    if dest_pos != last:
        grid[dest_pos[0] : dest_pos[0] + n, dest_pos[1] : dest_pos[1] + n] = _NARROW
        _paint_bridge(grid, last, dest_pos, n, _NARROW)


def _geometry_grid(grid: np.ndarray) -> np.ndarray:
    geom = np.array(grid, copy=True)
    geom[geom == _NARROW] = _WIDE
    return geom


def _gen_attempt() -> np.ndarray | None:
    n = random.choice([1, 2, 3, 4])
    reached = True if n == 1 else random.random() < 0.5
    use_narrow_hint = (n == 1 and random.random() < 0.75) or (n >= 2 and not reached)

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
    edge_cells = list(edge_cells)
    random.shuffle(edge_cells)
    start_coarse = edge_cells[0]
    far = sorted(edge_cells, key=lambda e: -(abs(e[0] - start_coarse[0]) + abs(e[1] - start_coarse[1])))
    end_coarse = far[0] if far[0] != start_coarse else far[1]

    walk = _hamiltonian_walk(coarse_rows, coarse_cols, start_coarse, end_coarse)
    if walk is None or len(walk) < 4:
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
        grid[r_lo:r_hi, dest_c : dest_c + n] = _WIDE
    if new_dest_c != dest_c:
        c_lo, c_hi = min(new_dest_c, dest_c), max(new_dest_c, dest_c) + n
        grid[dest_r : dest_r + n, c_lo:c_hi] = _WIDE
    dest_r, dest_c = new_dest_r, new_dest_c
    if not (dest_r == 0 or dest_r + n == H or dest_c == 0 or dest_c + n == W):
        return None

    touching = mover_r == 0 or mover_r + n == H or mover_c == 0 or mover_c + n == W
    if touching:
        if mover_r == 0:
            grid[0, mover_c : mover_c + n] = _WIDE
            mover_r = 1
        elif mover_r + n == H:
            grid[H - 1, mover_c : mover_c + n] = _WIDE
            mover_r = H - n - 1
        elif mover_c == 0:
            grid[mover_r : mover_r + n, 0] = _WIDE
            mover_c = 1
        else:
            grid[mover_r : mover_r + n, W - 1] = _WIDE
            mover_c = W - n - 1
    if mover_r == 0 or mover_r + n == H or mover_c == 0 or mover_c + n == W:
        return None
    if min(mover_r, H - (mover_r + n), mover_c, W - (mover_c + n)) > 2:
        return None

    grid[mover_r : mover_r + n, mover_c : mover_c + n] = _WIDE
    grid[dest_r : dest_r + n, dest_c : dest_c + n] = _WIDE

    if use_narrow_hint:
        lo = max(1, len(pixel_pos) // 3)
        hi = max(lo, len(pixel_pos) - 2)
        cut = random.randint(lo, hi)
        _color_suffix(grid, pixel_pos, cut, n, (dest_r, dest_c))
        if not np.any(grid == _NARROW):
            return None
    elif np.any(grid == _NARROW):
        return None

    grid[mover_r : mover_r + n, mover_c : mover_c + n] = _MOVER
    grid[dest_r : dest_r + n, dest_c : dest_c + n] = _DEST

    feasible = _compute_feasible(grid, n, _MOVER, _DEST)
    if not feasible[mover_r, mover_c]:
        return None
    dist = _bfs_placement(feasible, mover_r, mover_c)
    dest_ok = _dest_reachable(feasible, dist, dest_r, dest_c)

    geom = _geometry_grid(grid)
    geo_feasible = _compute_feasible(geom, n, _MOVER, _DEST)
    if not geo_feasible[mover_r, mover_c]:
        return None
    geo_dist = _bfs_placement(geo_feasible, mover_r, mover_c)
    geo_ok = _dest_reachable(geo_feasible, geo_dist, dest_r, dest_c)

    if n == 1:
        if not dest_ok or not geo_ok:
            return None
    else:
        if reached and not dest_ok:
            return None
        if not reached and dest_ok:
            return None
        if use_narrow_hint and not geo_ok:
            return None
        if not use_narrow_hint and geo_ok != dest_ok:
            return None

    tr, tc = _pick_terminal(dist, feasible, dest_r, dest_c, mover_r, mover_c)
    output = grid.copy()
    output[mover_r : mover_r + n, mover_c : mover_c + n] = _WIDE
    output[tr : tr + n, tc : tc + n] = _DEST if dest_ok else _MOVER

    if not np.array_equal(main(grid), output):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("Failed to generate valid input after 400 attempts")
