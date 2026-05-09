# Variant of 332f06d7
# Dimension: D10 — 阻塞场景处理 (meta)
# Change: 不可达时（stuck）方块停在BFS可达的最远位置（距起点最远的可行放置点），目标方块保持不变... → 阻塞时方块缩小

from __future__ import annotations

from collections import Counter, defaultdict, deque
import random

import numpy as np

from arc_utils import bbox_h, bbox_w, find_components


# Design analysis for the D10 variant
# ===================================
# a) Concrete behavior + trace
#    - Input still shows one blue winding channel on a green background, plus one black
#      solid square and one red solid square of the same original size n.
#    - The red square is the edge-touching endpoint; the black square is near an edge
#      but does not touch it. This is visible, so the solver does not rely on color-id
#      sorting to decide their roles.
#    - The blue channel is no longer "passable for n or not passable at all". Instead,
#      the path is built in width stages: a wide prefix fits n, then a narrower suffix
#      fits n-1, then possibly n-2, etc. At each width drop the black square is forced
#      to shrink by exactly 1 before it can continue.
#    - Example trace: for n=3, suppose the path capacities are [3,3,3,2,2,1,1]. The
#      original logic would BFS only with size 3 and stop before the first 2-wide part.
#      The variant first explores all 3x3 placements, then shrinks to 2x2 at the first
#      frontier, continues, then shrinks to 1x1 at the next frontier, and finally reaches
#      the red square with a 1x1 black square anchored at the red square's top-left.
# b) Complexity calibration
#    - Sizes are kept in the small original range n in {2,3,4}; shrink counts are
#      therefore in {1..n-1}. This creates visible but limited branching.
#    - The channel remains a single winding object; no extra symbols or markers are added.
#      Visual evidence comes from repeated width drops and the smaller black output square.
# c) Hidden assumptions audited from the original main()
#    - Original assumption "mover/dest can be selected by color sort" is invalid here:
#      using color ids as an invisible ordering cue is disallowed. Replaced with the
#      visible edge-touching rule.
#    - Original assumption "one fixed size n is used for all feasible placements" is no
#      longer valid. The variant precomputes feasible windows for every size 1..n and runs
#      BFS layer by layer.
#    - Original assumption "if dest is unreachable, return the farthest n-placement" is
#      replaced by "shrink at a width frontier and continue". A size-1 fallback remains
#      only as a safety net if a malformed input is still unreachable.
#    - Original unconditional overwrite of the whole destination square would be wrong once
#      the mover shrinks. The output now paints only the final sxs black square, leaving the
#      uncovered part of the red square visible.
# d) Placement legality
#    - Generator first chooses a coarse self-avoiding walk, then assigns a capacity to each
#      path segment. River pixels are rendered as top-left-anchored squares/strips whose
#      width equals that local capacity. This guarantees that every intended placement is
#      in bounds and that narrower suffixes do not overlap illegally with unrelated objects.
#    - The destination keeps the original n x n size and is extended to a boundary; the
#      blue connector that reaches it only needs the final capacity, because the red square
#      itself is traversable.
# e) Visual observability
#    - The distinguishing signal is visible in both input and output: the input channel
#      narrows in stages, and the output black square may be smaller than the input one.
#      No color-id arithmetic or hidden ranking is required.
# f) Visual cleanliness
#    - Only three object types remain: background, river, and two solid squares. Width
#      changes are aligned to the channel, so the image still reads as one coherent river
#      instead of overlapping decorations.
# g) Color-change inferability
#    - No color remapping is introduced. Colors preserve their original roles; only shape
#      size changes.
# New solver assumptions guaranteed by the generator
#    - Exactly two same-sized solid squares are present.
#    - Exactly one of them touches the grid edge (destination).
#    - The river is the most common non-background color other than the two square colors.
#    - At least one shrink frontier exists, and after the planned shrink sequence the red
#      top-left placement becomes reachable.


_BG, _RIVER, _MOVER, _DEST = 3, 1, 0, 2
_DIR4 = [(1, 0), (-1, 0), (0, 1), (0, -1)]


def _touches_edge(r0: int, c0: int, n: int, H: int, W: int) -> bool:
    return r0 == 0 or c0 == 0 or r0 + n == H or c0 + n == W


def _dist_to_edge(r0: int, c0: int, n: int, H: int, W: int) -> int:
    return min(r0, c0, H - (r0 + n), W - (c0 + n))


def _find_square_roles(g: np.ndarray, bg: int) -> tuple[dict, dict]:
    H, W = g.shape
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

    assert len(squares) >= 2

    by_n: dict[int, list[dict]] = defaultdict(list)
    for sq in squares:
        by_n[int(sq["n"])].append(sq)

    best_pair = None
    best_key = None
    for n, group in by_n.items():
        edge = [sq for sq in group if _touches_edge(int(sq["r0"]), int(sq["c0"]), int(n), H, W)]
        inner = [sq for sq in group if not _touches_edge(int(sq["r0"]), int(sq["c0"]), int(n), H, W)]
        if not edge or not inner:
            continue
        mover = min(inner, key=lambda sq: _dist_to_edge(int(sq["r0"]), int(sq["c0"]), int(n), H, W))
        dest = min(edge, key=lambda sq: (int(sq["r0"]), int(sq["c0"])))
        key = (len(group), int(n))
        if best_key is None or key > best_key:
            best_key = key
            best_pair = (mover, dest)

    assert best_pair is not None
    return best_pair


def _feasible_mask(trav: np.ndarray, size: int) -> np.ndarray:
    H, W = trav.shape
    h2 = H - size + 1
    w2 = W - size + 1
    assert h2 > 0 and w2 > 0
    feasible = np.zeros((h2, w2), dtype=bool)
    for r in range(h2):
        for c in range(w2):
            if bool(trav[r : r + size, c : c + size].all()):
                feasible[r, c] = True
    return feasible


def _bfs_from_starts(feasible: np.ndarray, starts: set[tuple[int, int]]) -> np.ndarray:
    h2, w2 = feasible.shape
    dist = -np.ones((h2, w2), dtype=int)
    q: deque[tuple[int, int]] = deque()
    for r, c in sorted(starts):
        if 0 <= r < h2 and 0 <= c < w2 and bool(feasible[r, c]) and int(dist[r, c]) == -1:
            dist[r, c] = 0
            q.append((r, c))
    while q:
        r, c = q.popleft()
        for dr, dc in _DIR4:
            rr, cc = r + dr, c + dc
            if 0 <= rr < h2 and 0 <= cc < w2 and bool(feasible[rr, cc]) and int(dist[rr, cc]) == -1:
                dist[rr, cc] = int(dist[r, c]) + 1
                q.append((rr, cc))
    return dist


def _farthest_reachable(dist: np.ndarray) -> tuple[int, int, int]:
    best_d = -1
    br = bc = 0
    h2, w2 = dist.shape
    for r in range(h2):
        for c in range(w2):
            if int(dist[r, c]) > best_d:
                best_d = int(dist[r, c])
                br, bc = r, c
    return br, bc, best_d


def _collect_shrink_frontier(dist: np.ndarray, feasible_cur: np.ndarray, feasible_next: np.ndarray) -> set[tuple[int, int]]:
    h2, w2 = dist.shape
    h3, w3 = feasible_next.shape
    out: set[tuple[int, int]] = set()
    for r in range(h2):
        for c in range(w2):
            if int(dist[r, c]) < 0:
                continue
            for dr, dc in _DIR4:
                rr, cc = r + dr, c + dc
                if not (0 <= rr < h3 and 0 <= cc < w3 and bool(feasible_next[rr, cc])):
                    continue
                cur_ok = 0 <= rr < h2 and 0 <= cc < w2 and bool(feasible_cur[rr, cc])
                if not cur_ok:
                    out.add((r, c))
                    break
    return out


def _simulate_variant(g: np.ndarray, mover: dict, dest: dict, river: int) -> dict:
    H, W = g.shape
    n = int(mover["n"])
    sr, sc = int(mover["r0"]), int(mover["c0"])
    dr, dc = int(dest["r0"]), int(dest["c0"])

    trav = (g == river) | (g == int(mover["col"])) | (g == int(dest["col"]))
    feasible_by_size = {size: _feasible_mask(trav, size) for size in range(1, n + 1)}
    assert bool(feasible_by_size[n][sr, sc])

    size = n
    starts = {(sr, sc)}
    shrink_sizes: list[int] = []
    best_state = {"r": sr, "c": sc, "size": n, "dist": 0}

    while True:
        feasible = feasible_by_size[size]
        dist = _bfs_from_starts(feasible, starts)
        br, bc, best_d = _farthest_reachable(dist)
        if best_d >= best_state["dist"]:
            best_state = {"r": br, "c": bc, "size": size, "dist": best_d}

        if 0 <= dr < feasible.shape[0] and 0 <= dc < feasible.shape[1] and int(dist[dr, dc]) >= 0:
            return {
                "r": dr,
                "c": dc,
                "size": size,
                "reached": True,
                "shrink_sizes": shrink_sizes,
            }

        if size == 1:
            return {
                "r": int(best_state["r"]),
                "c": int(best_state["c"]),
                "size": int(best_state["size"]),
                "reached": False,
                "shrink_sizes": shrink_sizes,
            }

        next_feasible = feasible_by_size[size - 1]
        next_starts = _collect_shrink_frontier(dist, feasible, next_feasible)
        if not next_starts:
            next_starts = {(int(r), int(c)) for r, c in np.argwhere(dist >= 0)}
        shrink_sizes.append(size - 1)
        starts = next_starts
        size -= 1


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move a solid square along a blue corridor toward a same-sized red endpoint, shrinking when blocked.

    Input structure:
    - The background is the corner-majority color.
    - A blue winding river connects a black solid square to a red solid square of the same original size.
    - The red square touches the grid edge; the black square starts near an edge without touching it.

    Rule:
    - The black square slides through the blue corridor using 4-neighbor top-left placements.
    - It keeps its current size while that size still fits the river.
    - When a chokepoint is too narrow, the square shrinks by exactly 1 and continues from the blockage frontier.
    - The original black square location becomes river, and the final smaller black square is painted at the red square's top-left if reachable.

    Key insight:
    - The river is staged by width, so the correct solution is a layered BFS over sizes n, n-1, ..., 1 rather than a single fixed-size reachability check.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape

    bg = int(Counter([int(g[0, 0]), int(g[0, -1]), int(g[-1, 0]), int(g[-1, -1])]).most_common(1)[0][0])
    mover, dest = _find_square_roles(g, bg)

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

    sim = _simulate_variant(g, mover, dest, river)
    sr, sc, n = int(mover["r0"]), int(mover["c0"]), int(mover["n"])
    tr, tc, size = int(sim["r"]), int(sim["c"]), int(sim["size"])

    out = g.copy()
    out[sr : sr + n, sc : sc + n] = river
    out[tr : tr + size, tc : tc + size] = int(mover["col"])
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
                        if (
                            0 <= nr2 < rows
                            and 0 <= nc2 < cols
                            and (nr2, nc2) not in test_visited
                            and (nr2, nc2) not in seen
                        ):
                            seen.add((nr2, nc2))
                            q.append((nr2, nc2))
                for dr2, dc2 in _DIR4:
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


def _choose_shrink_segments(path_len: int, num_shrinks: int) -> list[int] | None:
    low = max(1, path_len // 5)
    high = path_len - 2
    if high < low or high - low + 1 < num_shrinks:
        return None
    choices = list(range(low, high + 1))
    random.shuffle(choices)
    picks: list[int] = []
    for idx in sorted(choices):
        if all(abs(idx - prev) >= 2 for prev in picks):
            picks.append(idx)
            if len(picks) == num_shrinks:
                break
    if len(picks) != num_shrinks:
        return None
    return sorted(picks)


def _render_variable_channel(
    grid: np.ndarray,
    path: list[tuple[int, int]],
    caps: list[int],
    offsets_r: list[int],
    offsets_c: list[int],
) -> list[tuple[int, int]]:
    pixel_pos: list[tuple[int, int]] = []
    for (cr, cc), cap in zip(path, caps):
        pr, pc = offsets_r[cr], offsets_c[cc]
        pixel_pos.append((pr, pc))
        grid[pr : pr + cap, pc : pc + cap] = _RIVER
    for i in range(len(path) - 1):
        pr1, pc1 = pixel_pos[i]
        pr2, pc2 = pixel_pos[i + 1]
        cap = min(caps[i], caps[i + 1])
        if pr1 == pr2:
            c_lo, c_hi = min(pc1, pc2), max(pc1, pc2) + cap
            grid[pr1 : pr1 + cap, c_lo:c_hi] = _RIVER
        else:
            r_lo, r_hi = min(pr1, pr2), max(pr1, pr2) + cap
            grid[r_lo:r_hi, pc1 : pc1 + cap] = _RIVER
    return pixel_pos


def _extend_dest_to_edge(
    grid: np.ndarray, pr: int, pc: int, n: int
) -> tuple[tuple[int, int] | None, str | None]:
    H, W = grid.shape
    candidates = []
    for direction in ["top", "bottom", "left", "right"]:
        if direction == "top":
            dist = pr
            if dist == 0:
                candidates.append(((0, pc), direction, 0))
                continue
            if all(grid[r, c] in (_BG, _RIVER) for r in range(pr) for c in range(pc, pc + n) if c < W):
                candidates.append(((0, pc), direction, dist))
        elif direction == "bottom":
            dist = H - (pr + n)
            if dist == 0:
                candidates.append(((pr, pc), direction, 0))
                continue
            if all(grid[r, c] in (_BG, _RIVER) for r in range(pr + n, H) for c in range(pc, pc + n) if c < W):
                candidates.append(((H - n, pc), direction, dist))
        elif direction == "left":
            dist = pc
            if dist == 0:
                candidates.append(((pr, 0), direction, 0))
                continue
            if all(grid[r, c] in (_BG, _RIVER) for r in range(pr, pr + n) for c in range(pc) if r < H):
                candidates.append(((pr, 0), direction, dist))
        else:
            dist = W - (pc + n)
            if dist == 0:
                candidates.append(((pr, pc), direction, 0))
                continue
            if all(grid[r, c] in (_BG, _RIVER) for r in range(pr, pr + n) for c in range(pc + n, W) if r < H):
                candidates.append(((pr, W - n), direction, dist))
    if not candidates:
        return None, None
    candidates.sort(key=lambda item: item[2])
    return candidates[0][0], candidates[0][1]


def _paint_dest_connector(
    grid: np.ndarray, pr: int, pc: int, dest_r: int, dest_c: int, final_size: int
) -> None:
    if dest_r != pr:
        r_lo, r_hi = min(dest_r, pr), max(dest_r, pr) + final_size
        grid[r_lo:r_hi, pc : pc + final_size] = _RIVER
    if dest_c != pc:
        c_lo, c_hi = min(dest_c, pc), max(dest_c, pc) + final_size
        grid[pr : pr + final_size, c_lo:c_hi] = _RIVER


def _gen_attempt() -> np.ndarray | None:
    n = random.choice([2, 3, 4])
    num_shrinks = random.randint(1, n - 1)
    final_size = n - num_shrinks

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

    edge_cells = []
    for c in range(1, coarse_cols - 1):
        edge_cells.append((0, c))
        edge_cells.append((coarse_rows - 1, c))
    for r in range(1, coarse_rows - 1):
        edge_cells.append((r, 0))
        edge_cells.append((r, coarse_cols - 1))
    if len(edge_cells) < 2:
        return None

    random.shuffle(edge_cells)
    start_coarse = edge_cells[0]
    far = sorted(edge_cells, key=lambda e: -(abs(e[0] - start_coarse[0]) + abs(e[1] - start_coarse[1])))
    end_coarse = far[0] if far[0] != start_coarse else far[1]

    walk = _hamiltonian_walk(coarse_rows, coarse_cols, start_coarse, end_coarse)
    if walk is None or len(walk) < max(5, 2 * num_shrinks + 3):
        return None
    if len(set(cr for cr, _ in walk)) < coarse_rows or len(set(cc for _, cc in walk)) < coarse_cols:
        return None

    choke_segments = _choose_shrink_segments(len(walk), num_shrinks)
    if choke_segments is None:
        return None

    caps = []
    for idx in range(len(walk)):
        drops = sum(1 for seg in choke_segments if seg < idx)
        caps.append(n - drops)

    grid = np.full((H, W), _BG, dtype=int)
    pixel_pos = _render_variable_channel(grid, walk, caps, offsets_r, offsets_c)

    mask = grid != _BG
    if np.any(mask, axis=1).sum() / H < 0.65 or np.any(mask, axis=0).sum() / W < 0.65:
        return None

    mover_r, mover_c = pixel_pos[0]
    dest_r, dest_c = pixel_pos[-1]

    dest_pos, _direction = _extend_dest_to_edge(grid, dest_r, dest_c, n)
    if dest_pos is None:
        return None
    new_dest_r, new_dest_c = dest_pos
    _paint_dest_connector(grid, dest_r, dest_c, new_dest_r, new_dest_c, final_size)
    dest_r, dest_c = new_dest_r, new_dest_c

    if not (dest_r == 0 or dest_r + n == H or dest_c == 0 or dest_c + n == W):
        return None
    if mover_r == 0 or mover_c == 0 or mover_r + n == H or mover_c + n == W:
        return None
    if _dist_to_edge(mover_r, mover_c, n, H, W) > 2:
        return None

    grid[mover_r : mover_r + n, mover_c : mover_c + n] = _RIVER
    grid[dest_r : dest_r + n, dest_c : dest_c + n] = _RIVER
    grid[mover_r : mover_r + n, mover_c : mover_c + n] = _MOVER
    grid[dest_r : dest_r + n, dest_c : dest_c + n] = _DEST

    bg = int(Counter([int(grid[0, 0]), int(grid[0, -1]), int(grid[-1, 0]), int(grid[-1, -1])]).most_common(1)[0][0])
    mover, dest = _find_square_roles(grid, bg)
    sim = _simulate_variant(grid, mover, dest, _RIVER)
    if not bool(sim["reached"]):
        return None
    if int(sim["size"]) >= n:
        return None
    if len(sim["shrink_sizes"]) != num_shrinks:
        return None

    output = grid.copy()
    output[mover_r : mover_r + n, mover_c : mover_c + n] = _RIVER
    final_r, final_c, size = int(sim["r"]), int(sim["c"]), int(sim["size"])
    output[final_r : final_r + size, final_c : final_c + size] = _MOVER

    if not np.array_equal(main(grid), output):
        return None
    return grid


def generate_input():
    for _ in range(400):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("Failed to generate valid input after 400 attempts")
