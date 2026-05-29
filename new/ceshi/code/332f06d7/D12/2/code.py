# Variant of 332f06d7
# Dimension: D12 — 背景与通道的颜色关系 (meta)
# Change: 固定配色：背景=绿色(3)，通道=蓝色(1)，移动体=黑色(0)，目标=红色(2)... → 多色通道分段

from __future__ import annotations

from collections import Counter, defaultdict, deque
import random

import numpy as np
from arc_utils import bbox_h, bbox_w, find_components


# Design analysis for the D12 variant
# a) Concrete behavior:
#    The river is no longer a single-color corridor. Each input contains a main
#    corridor plus extra segments in a second river color. The mover may slide
#    only on the river color that directly surrounds its starting square. This
#    changes the placement-feasibility stage: the original solver treated one
#    globally chosen river color as traversable, while the variant must infer a
#    local "authorized water color" from the mover's neighborhood.
#    Small example: if the black square starts inside a blue segment and the
#    corridor later continues in teal, the old logic would still regard the
#    whole corridor as open if teal were chosen as the global river. The new
#    logic stops at the last fully blue placement before the teal segment.
# b) Complexity calibration:
#    Only two river colors are used in each sample. This is enough to make the
#    color rule visible while avoiding ambiguity from many competing colors.
#    Reachable cases keep the whole main corridor in the mover's water color and
#    add teal/other-color side branches; stuck cases recolor a downstream suffix
#    of the main corridor, so the color rule actively determines the stop point.
# c) Hidden assumptions from the original main():
#    - Sorting the two squares by color id is no longer acceptable; roles are
#      assigned geometrically instead: the destination touches the border and the
#      mover does not. The generator guarantees this.
#    - Choosing one global most-common river color no longer works because two
#      river colors are present. The replacement assumption is: among non-bg
#      colors touching the mover's 1-cell ring, one color dominates, and that is
#      the passable river color. The generator keeps other river colors away from
#      that ring.
#    - Filling the mover's original position with the river color remains valid
#      only if the square started on the authorized river color. The generator
#      explicitly paints that underlay before placing the mover.
# d) Placement legality:
#    The generator first builds a coarse placement walk, paints it as a full n×n
#    corridor, extends the destination to the border, then adds a second-color
#    suffix or side branches without overlapping the squares. Every produced grid
#    is checked by the solver oracle before acceptance.
# e) Visual observability:
#    The passable river color varies across samples and is inferred from visible
#    adjacency to the mover, not from color ids. The other river color appears in
#    repeated segments, so the "only one water color works" rule is visible in
#    2-3 examples.
# f) Visual cleanliness:
#    Extra-color segments are attached as clean branches or contiguous suffixes;
#    they do not overlap the mover or the destination, and they do not create
#    dense marker clutter.
# g) Color-rule inferability:
#    Each sample uses exactly two river colors, with the mover surrounded by only
#    one of them. Reachable samples show the mover ignoring other-color branches;
#    stuck samples show the mover halting exactly when the corridor changes to
#    the other color. That repeated structure makes the mapping human-inferable.


_BG, _MOVER, _DEST = 3, 0, 2
_CHANNEL_PALETTE = [1, 4, 5, 6, 7, 8, 9]
_DIR4 = [(1, 0), (-1, 0), (0, 1), (0, -1)]


def _touches_edge(square: dict, H: int, W: int) -> bool:
    r0, c0, n = int(square["r0"]), int(square["c0"]), int(square["n"])
    return r0 == 0 or c0 == 0 or r0 + n == H or c0 + n == W


def _detect_squares(g: np.ndarray, bg: int) -> list[dict]:
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


def _identify_roles(g: np.ndarray, bg: int) -> tuple[dict, dict]:
    H, W = g.shape
    by_n: dict[int, list[dict]] = defaultdict(list)
    for sq in _detect_squares(g, bg):
        by_n[int(sq["n"])].append(sq)

    options: list[tuple[int, int, list[dict], list[dict]]] = []
    for n, group in by_n.items():
        edge = [sq for sq in group if _touches_edge(sq, H, W)]
        inner = [sq for sq in group if not _touches_edge(sq, H, W)]
        if edge and inner:
            options.append((len(group), int(n), inner, edge))
    assert options

    _count, _n, inner, edge = max(options, key=lambda item: (item[0], item[1]))
    mover = min(inner, key=lambda sq: (sq["r0"] + sq["c0"], sq["r0"], sq["c0"]))
    dest = min(edge, key=lambda sq: (sq["r0"] + sq["c0"], sq["r0"], sq["c0"]))
    return mover, dest


def _passable_color(g: np.ndarray, mover: dict, dest: dict, bg: int) -> int:
    H, W = g.shape
    ignore = {bg, int(mover["col"]), int(dest["col"])}
    r0, c0, n = int(mover["r0"]), int(mover["c0"]), int(mover["n"])
    counts: Counter[int] = Counter()

    if r0 > 0:
        for c in range(c0, c0 + n):
            col = int(g[r0 - 1, c])
            if col not in ignore:
                counts[col] += 1
    if r0 + n < H:
        for c in range(c0, c0 + n):
            col = int(g[r0 + n, c])
            if col not in ignore:
                counts[col] += 1
    if c0 > 0:
        for r in range(r0, r0 + n):
            col = int(g[r, c0 - 1])
            if col not in ignore:
                counts[col] += 1
    if c0 + n < W:
        for r in range(r0, r0 + n):
            col = int(g[r, c0 + n])
            if col not in ignore:
                counts[col] += 1

    assert counts
    return int(counts.most_common(1)[0][0])


def _bfs_placement(feasible: np.ndarray, sr: int, sc: int) -> np.ndarray:
    h2, w2 = feasible.shape
    dist = -np.ones((h2, w2), dtype=int)
    dist[sr, sc] = 0
    q = deque([(sr, sc)])
    while q:
        r, c = q.popleft()
        for dr, dc in _DIR4:
            rr, cc = r + dr, c + dc
            if 0 <= rr < h2 and 0 <= cc < w2 and feasible[rr, cc] and dist[rr, cc] == -1:
                dist[rr, cc] = dist[r, c] + 1
                q.append((rr, cc))
    return dist


def _solve_core(grid: np.ndarray) -> tuple[np.ndarray, dict]:
    g = np.array(grid, dtype=int)
    H, W = g.shape

    bg = int(Counter([int(g[0, 0]), int(g[0, -1]), int(g[-1, 0]), int(g[-1, -1])]).most_common(1)[0][0])
    mover, dest = _identify_roles(g, bg)

    n = int(mover["n"])
    sr, sc = int(mover["r0"]), int(mover["c0"])
    dr, dc = int(dest["r0"]), int(dest["c0"])
    river = _passable_color(g, mover, dest, bg)

    trav = (g == river) | (g == int(mover["col"])) | (g == int(dest["col"]))
    h2, w2 = H - n + 1, W - n + 1
    assert h2 > 0 and w2 > 0
    assert 0 <= sr < h2 and 0 <= sc < w2

    feasible = np.zeros((h2, w2), dtype=bool)
    for r in range(h2):
        for c in range(w2):
            if bool(trav[r : r + n, c : c + n].all()):
                feasible[r, c] = True
    assert bool(feasible[sr, sc])

    dist = _bfs_placement(feasible, sr, sc)
    if 0 <= dr < h2 and 0 <= dc < w2 and feasible[dr, dc] and dist[dr, dc] >= 0:
        tr, tc = dr, dc
    else:
        best = -1
        tr, tc = sr, sc
        for r in range(h2):
            for c in range(w2):
                if int(dist[r, c]) > best:
                    best = int(dist[r, c])
                    tr, tc = r, c

    out = g.copy()
    out[sr : sr + n, sc : sc + n] = river
    out[tr : tr + n, tc : tc + n] = int(mover["col"])
    meta = {
        "bg": bg,
        "river": river,
        "mover": mover,
        "dest": dest,
        "target": (tr, tc),
        "dist": dist,
        "feasible": feasible,
    }
    return out, meta


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide a solid square along a multi-colored waterway to a same-sized border target.

    Input structure:
    - The corner-majority background surrounds a winding corridor made from two river colors.
    - Two same-sized solid squares sit on that corridor: an interior mover and a border-touching destination.

    Rule:
    - The mover can travel only through the river color that touches its starting square.
    - A placement is valid only when the full NxN mover footprint stays inside that authorized river color.
    - If the destination lies on the reachable authorized-water region, the mover replaces it; otherwise it stops at the farthest reachable placement.
    - The mover's start is restored to the authorized river color.

    Key insight:
    The second river color is a visual decoy: even a wide corridor segment becomes impassable when its color does not match the mover's starting water.
    """
    out, _meta = _solve_core(np.array(grid, dtype=int))
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


def _compute_offsets(G: int, n: int, num_coarse: int) -> list[int] | None:
    if num_coarse == 1:
        return [1]
    first = 1
    last = G - n - 1
    if last < first:
        return None
    step = (last - first) / (num_coarse - 1)
    return [int(round(first + i * step)) for i in range(num_coarse)]


def _paint_cell(grid: np.ndarray, pr: int, pc: int, n: int, color: int) -> None:
    grid[pr : pr + n, pc : pc + n] = color


def _paint_link(grid: np.ndarray, p1: tuple[int, int], p2: tuple[int, int], n: int, color: int) -> None:
    pr1, pc1 = p1
    pr2, pc2 = p2
    if pr1 == pr2:
        c_lo, c_hi = min(pc1, pc2), max(pc1, pc2) + n
        grid[pr1 : pr1 + n, c_lo:c_hi] = color
    else:
        r_lo, r_hi = min(pr1, pr2), max(pr1, pr2) + n
        grid[r_lo:r_hi, pc1 : pc1 + n] = color


def _paint_branch_link(grid: np.ndarray, p_main: tuple[int, int], p_branch: tuple[int, int], n: int, color: int) -> None:
    pr1, pc1 = p_main
    pr2, pc2 = p_branch
    if pr1 == pr2:
        if pc2 > pc1:
            grid[pr1 : pr1 + n, pc1 + n : pc2 + n] = color
        else:
            grid[pr1 : pr1 + n, pc2:pc1] = color
    else:
        if pr2 > pr1:
            grid[pr1 + n : pr2 + n, pc1 : pc1 + n] = color
        else:
            grid[pr2:pr1, pc1 : pc1 + n] = color


def _render_main_path(
    grid: np.ndarray,
    path: list[tuple[int, int]],
    n: int,
    offsets_r: list[int],
    offsets_c: list[int],
    color: int,
) -> list[tuple[int, int]]:
    pixel_pos = []
    for cr, cc in path:
        pr, pc = offsets_r[cr], offsets_c[cc]
        pixel_pos.append((pr, pc))
        _paint_cell(grid, pr, pc, n, color)
    for i in range(len(pixel_pos) - 1):
        _paint_link(grid, pixel_pos[i], pixel_pos[i + 1], n, color)
    return pixel_pos


def _extend_to_edge(grid: np.ndarray, pr: int, pc: int, n: int, river: int) -> tuple[int, int] | None:
    H, W = grid.shape
    candidates = []
    for direction in ["top", "bottom", "left", "right"]:
        if direction == "top":
            dist = pr
            if dist == 0:
                candidates.append((0, pc, 0))
                continue
            if all(grid[r, c] in (_BG, river) for r in range(pr) for c in range(pc, pc + n)):
                candidates.append((0, pc, dist))
        elif direction == "bottom":
            dist = H - (pr + n)
            if dist == 0:
                candidates.append((pr, pc, 0))
                continue
            if all(grid[r, c] in (_BG, river) for r in range(pr + n, H) for c in range(pc, pc + n)):
                candidates.append((H - n, pc, dist))
        elif direction == "left":
            dist = pc
            if dist == 0:
                candidates.append((pr, 0, 0))
                continue
            if all(grid[r, c] in (_BG, river) for r in range(pr, pr + n) for c in range(pc)):
                candidates.append((pr, 0, dist))
        else:
            dist = W - (pc + n)
            if dist == 0:
                candidates.append((pr, pc, 0))
                continue
            if all(grid[r, c] in (_BG, river) for r in range(pr, pr + n) for c in range(pc + n, W)):
                candidates.append((pr, W - n, dist))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[2])
    return int(candidates[0][0]), int(candidates[0][1])


def _add_side_branches(
    grid: np.ndarray,
    walk: list[tuple[int, int]],
    pixel_pos: list[tuple[int, int]],
    offsets_r: list[int],
    offsets_c: list[int],
    n: int,
    blocked: int,
) -> int:
    occupied = set(walk)
    candidates = []
    for idx in range(2, len(walk) - 2):
        r, c = walk[idx]
        for dr, dc in _DIR4:
            nr, nc = r + dr, c + dc
            if (nr, nc) in occupied:
                continue
            if not (0 <= nr < len(offsets_r) and 0 <= nc < len(offsets_c)):
                continue
            candidates.append((idx, (dr, dc), (nr, nc)))
    random.shuffle(candidates)

    made = 0
    for idx, direction, root in candidates:
        if root in occupied:
            continue
        main_pixel = pixel_pos[idx]
        root_pixel = (offsets_r[root[0]], offsets_c[root[1]])
        _paint_branch_link(grid, main_pixel, root_pixel, n, blocked)
        _paint_cell(grid, root_pixel[0], root_pixel[1], n, blocked)
        occupied.add(root)

        dr, dc = direction
        ext = (root[0] + dr, root[1] + dc)
        if random.random() < 0.5 and 0 <= ext[0] < len(offsets_r) and 0 <= ext[1] < len(offsets_c) and ext not in occupied:
            ext_pixel = (offsets_r[ext[0]], offsets_c[ext[1]])
            _paint_link(grid, root_pixel, ext_pixel, n, blocked)
            _paint_cell(grid, ext_pixel[0], ext_pixel[1], n, blocked)
            occupied.add(ext)

        made += 1
        if made >= random.randint(2, 4):
            break
    return made


def _force_mover_interior(grid: np.ndarray, pr: int, pc: int, n: int, river: int) -> tuple[int, int] | None:
    H, W = grid.shape
    mover_r, mover_c = pr, pc
    if mover_r == 0:
        grid[0, mover_c : mover_c + n] = river
        mover_r = 1
    elif mover_r + n == H:
        grid[H - 1, mover_c : mover_c + n] = river
        mover_r = H - n - 1
    elif mover_c == 0:
        grid[mover_r : mover_r + n, 0] = river
        mover_c = 1
    elif mover_c + n == W:
        grid[mover_r : mover_r + n, W - 1] = river
        mover_c = W - n - 1

    if mover_r <= 0 or mover_r + n >= H or mover_c <= 0 or mover_c + n >= W:
        return None
    if min(mover_r, H - (mover_r + n), mover_c, W - (mover_c + n)) > 2:
        return None
    return mover_r, mover_c


def _recolor_suffix(grid: np.ndarray, pixel_pos: list[tuple[int, int]], n: int, start_idx: int, color: int) -> None:
    for i in range(start_idx, len(pixel_pos)):
        _paint_cell(grid, pixel_pos[i][0], pixel_pos[i][1], n, color)
    for i in range(start_idx, len(pixel_pos) - 1):
        _paint_link(grid, pixel_pos[i], pixel_pos[i + 1], n, color)
    if start_idx > 0:
        _paint_link(grid, pixel_pos[start_idx - 1], pixel_pos[start_idx], n, color)


def _gen_attempt(reached: bool | None = None) -> np.ndarray | None:
    if reached is None:
        reached = random.random() < 0.5
    n = random.choice([1, 2, 3, 4])

    min_size = max(10, 5 * n)
    if min_size > 20:
        return None
    G = random.randint(min_size, 20)
    H = W = G

    allowed, blocked = random.sample(_CHANNEL_PALETTE, 2)

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
    if len(set(r for r, _ in walk)) < coarse_rows or len(set(c for _, c in walk)) < coarse_cols:
        return None

    grid = np.full((H, W), _BG, dtype=int)
    pixel_pos = _render_main_path(grid, walk, n, offsets_r, offsets_c, allowed)

    mover_r, mover_c = pixel_pos[0]
    dest_r, dest_c = pixel_pos[-1]

    edge_dest = _extend_to_edge(grid, dest_r, dest_c, n, allowed)
    if edge_dest is None:
        return None
    new_dest_r, new_dest_c = edge_dest
    if new_dest_r != dest_r:
        r_lo, r_hi = min(new_dest_r, dest_r), max(new_dest_r, dest_r) + n
        grid[r_lo:r_hi, dest_c : dest_c + n] = allowed
    if new_dest_c != dest_c:
        c_lo, c_hi = min(new_dest_c, dest_c), max(new_dest_c, dest_c) + n
        grid[dest_r : dest_r + n, c_lo:c_hi] = allowed
    dest_r, dest_c = new_dest_r, new_dest_c
    pixel_pos[-1] = (dest_r, dest_c)

    mover_fix = _force_mover_interior(grid, mover_r, mover_c, n, allowed)
    if mover_fix is None:
        return None
    mover_r, mover_c = mover_fix
    _paint_cell(grid, mover_r, mover_c, n, allowed)
    _paint_cell(grid, dest_r, dest_c, n, allowed)

    if reached:
        branches = _add_side_branches(grid, walk, pixel_pos, offsets_r, offsets_c, n, blocked)
        if branches < 2:
            return None
    else:
        cut_idx = random.randint(max(1, len(pixel_pos) // 3), len(pixel_pos) - 1)
        _recolor_suffix(grid, pixel_pos, n, cut_idx, blocked)
        if random.random() < 0.7:
            _add_side_branches(grid, walk, pixel_pos, offsets_r, offsets_c, n, blocked)
        if dest_r == pixel_pos[-1][0] and dest_c == pixel_pos[-1][1]:
            _paint_cell(grid, dest_r, dest_c, n, blocked)
        if cut_idx <= 1:
            return None

    grid[mover_r : mover_r + n, mover_c : mover_c + n] = _MOVER
    grid[dest_r : dest_r + n, dest_c : dest_c + n] = _DEST

    colors = sorted(set(int(v) for v in grid.flatten()))
    river_colors = [col for col in colors if col not in (_BG, _MOVER, _DEST)]
    if len(river_colors) < 2:
        return None

    try:
        out, meta = _solve_core(grid)
    except Exception:
        return None

    if meta["river"] != allowed:
        return None

    dest_ok = meta["target"] == (dest_r, dest_c)
    if reached != dest_ok:
        return None

    if reached and np.array_equal(out, grid):
        return None
    if not reached and meta["dist"][dest_r, dest_c] >= 0:
        return None

    return grid


def generate_input() -> np.ndarray:
    desired = random.random() < 0.5
    for _ in range(250):
        result = _gen_attempt(desired)
        if result is not None:
            return result
    for _ in range(250):
        result = _gen_attempt(None)
        if result is not None:
            return result
    raise RuntimeError("Failed to generate valid input after 500 attempts")
