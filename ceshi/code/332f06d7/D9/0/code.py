# Variant of 332f06d7
# Dimension: D9 — 多方块移动（基数关系） (multi_step)
# Change: 1:1单一移动体到单一目标... → 多对一移动

from __future__ import annotations

from collections import Counter, defaultdict, deque
import random

import numpy as np

from arc_utils import bbox_h, bbox_w, find_components

# Design analysis for the D9 variant
# ----------------------------------
# concretization:
# - The blue corridor is no longer a single start-to-end river. It is a tree:
#   three colored n×n movers sit on three leaf branches, and one red n×n square
#   sits at the unique boundary leaf as the shared sink.
# - Output behavior must stay visible on the pixel grid. If every reachable
#   mover tried to overwrite the same red square, the colors would conflict.
#   Therefore the shared red square acts as an absorbing sink: a mover that can
#   reach it disappears into the red target, while a blocked mover remains
#   visible at the farthest feasible placement on its own branch.
# - Small example at the changed step:
#   original: one black square BFSes on {river, black, red}; if red reachable,
#   paint black at red.
#   variant: three movers BFS independently on {river, self, red}; each one is
#   solved separately; successful movers vanish, unsuccessful movers repaint
#   themselves at their branch-local stopping point.
#
# complexity calibration:
# - Always generate 3 movers, not 2, so "many-to-one" is visually obvious from
#   a single sample.
# - Keep all movers the same size as the target. Only the cardinality changes.
# - Use n in {2, 3}; n=1 would erase the chokepoint effect and collapse back to
#   trivial reachability, which would weaken the D9 signal.
# - Restrict mover colors to a small palette and do not encode any rule in the
#   color ids. Colors only distinguish multiple movers for human tracking.
#
# hidden assumptions audited from the original main():
# - Sorting two squares by color to decide mover/destination no longer works.
#   The new solver identifies the target structurally: it is the unique same-
#   sized solid square touching the border.
# - Treating all non-bg square colors as traversable would let one mover pass
#   through another mover's starting block. That is invalid for independent
#   motion, so each BFS only uses {river, current mover, target}.
# - The original single overwrite at the end would destroy multi-mover results.
#   The new output clears all starts first, then repaints only the stuck movers.
#
# placement legality:
# - Generator uses a fixed branch template in coarse placement space. Branches
#   are disjoint except at one junction, so movers never overlap and each has a
#   unique route to the shared trunk.
# - Chokepoints are inserted only on a mover's private last branch segment
#   before the junction. This guarantees that blocking one mover does not change
#   the reachability of the others.
# - The red target is extended to the actual grid edge; movers stay one pixel
#   away from the border so the border-touching test cleanly singles out the
#   target.
#
# visual observability / cleanliness:
# - Human-solvable clue: all interior same-sized colored squares behave by the
#   same motion rule and all point toward the same edge red square.
# - Every sample contains both reached and stuck movers, so the independence of
#   branch-local blocking is visible without needing many examples.
#
# new solver assumptions guaranteed by the generator:
# - There is exactly one size group containing at least 4 solid squares.
# - In that group, exactly one square touches the border.
# - All movers and the target have pairwise disjoint initial placements.
# - A mover's accessible region is a simple private branch plus the shared trunk,
#   so "farthest reachable point" stays unambiguous.


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide several same-sized squares through a branching channel toward one shared sink.

    Input structure:
    - The corner-majority color is background, and blue forms a connected river tree.
    - Several interior solid NxN squares of different colors are movers.
    - One same-sized red NxN square touches the border and serves as the common target.

    Rule:
    - Each mover is processed independently on the blue river while the other movers stay fixed obstacles.
    - A mover may occupy only placements where its full NxN footprint lies inside {river, itself, target}.
    - If it can reach the red square, it disappears into that shared target; otherwise it stops at its farthest reachable placement.
    - Every original mover position is turned back into river color.

    Key insight: All movers share one destination, but chokepoints on private branches block only the mover whose branch contains the narrow gate.
    """
    out, _meta = _solve_instance(grid)
    return out


def _touches_border(square: dict, H: int, W: int) -> bool:
    r0 = int(square["r0"])
    c0 = int(square["c0"])
    n = int(square["n"])
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


def _choose_problem_group(squares: list[dict], H: int, W: int) -> tuple[int, dict, list[dict]]:
    by_n: dict[int, list[dict]] = defaultdict(list)
    for sq in squares:
        by_n[int(sq["n"])].append(sq)

    candidates: list[tuple[int, int, list[dict], dict, list[dict]]] = []
    for nside, group in by_n.items():
        if len(group) < 3:
            continue
        border = [sq for sq in group if _touches_border(sq, H, W)]
        if len(border) != 1:
            continue
        target = border[0]
        movers = [sq for sq in group if sq is not target]
        if len(movers) < 2:
            continue
        candidates.append((len(group), int(nside), group, target, movers))

    assert candidates
    _count, nside, _group, target, movers = max(candidates, key=lambda item: (item[0], item[1]))
    return int(nside), target, movers


def _detect_river(g: np.ndarray, bg: int, square_colors: set[int]) -> int:
    counts = Counter(map(int, g.flatten()))
    for col, _ct in counts.most_common():
        if int(col) == int(bg):
            continue
        if int(col) in square_colors:
            continue
        return int(col)
    raise AssertionError("river color not found")


def _bfs_placement(feasible: np.ndarray, sr: int, sc: int) -> np.ndarray:
    h2, w2 = feasible.shape
    dist = -np.ones((h2, w2), dtype=int)
    dist[int(sr), int(sc)] = 0
    q = deque([(int(sr), int(sc))])
    while q:
        r, c = q.popleft()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            rr = int(r + dr)
            cc = int(c + dc)
            if 0 <= rr < h2 and 0 <= cc < w2 and bool(feasible[rr, cc]) and int(dist[rr, cc]) == -1:
                dist[rr, cc] = int(dist[int(r), int(c)] + 1)
                q.append((rr, cc))
    return dist


def _choose_stop(dist: np.ndarray, target_rc: tuple[int, int]) -> tuple[int, int]:
    dr, dc = target_rc
    best_key = None
    best_rc = None
    for r in range(dist.shape[0]):
        for c in range(dist.shape[1]):
            d = int(dist[r, c])
            if d < 0:
                continue
            manh = abs(int(r) - int(dr)) + abs(int(c) - int(dc))
            key = (d, -manh, -int(r), -int(c))
            if best_key is None or key > best_key:
                best_key = key
                best_rc = (int(r), int(c))
    assert best_rc is not None
    return best_rc


def _solve_instance(grid: np.ndarray) -> tuple[np.ndarray, dict]:
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(Counter([int(g[0, 0]), int(g[0, -1]), int(g[-1, 0]), int(g[-1, -1])]).most_common(1)[0][0])

    squares = _detect_squares(g, bg)
    assert len(squares) >= 4
    n, target, movers = _choose_problem_group(squares, H, W)
    square_colors = {int(sq["col"]) for sq in squares}
    river = _detect_river(g, bg, square_colors)

    tr = int(target["r0"])
    tc = int(target["c0"])
    h2 = int(H - n + 1)
    w2 = int(W - n + 1)
    assert h2 > 0 and w2 > 0

    out = g.copy()
    for mover in movers:
        sr = int(mover["r0"])
        sc = int(mover["c0"])
        out[sr : sr + n, sc : sc + n] = int(river)

    mover_meta = []
    for mover in movers:
        sr = int(mover["r0"])
        sc = int(mover["c0"])
        mover_col = int(mover["col"])
        trav = (g == int(river)) | (g == mover_col) | (g == int(target["col"]))

        feasible = np.zeros((h2, w2), dtype=bool)
        for r in range(h2):
            for c in range(w2):
                if bool(trav[r : r + n, c : c + n].all()):
                    feasible[r, c] = True
        assert bool(feasible[sr, sc])

        dist = _bfs_placement(feasible, sr, sc)
        reached = bool(0 <= tr < h2 and 0 <= tc < w2 and feasible[tr, tc] and int(dist[tr, tc]) >= 0)
        if reached:
            final_r, final_c = tr, tc
            status = "reached"
        else:
            final_r, final_c = _choose_stop(dist, (tr, tc))
            out[final_r : final_r + n, final_c : final_c + n] = mover_col
            status = "stuck"

        mover_meta.append(
            {
                "color": mover_col,
                "start": (sr, sc),
                "final": (int(final_r), int(final_c)),
                "status": status,
            }
        )

    meta = {
        "n": int(n),
        "bg": int(bg),
        "river": int(river),
        "target_color": int(target["col"]),
        "target": (tr, tc),
        "movers": mover_meta,
    }
    return out, meta


# --- Generator ---

_BG = 3
_RIVER = 1
_DEST = 2
_MOVER_PALETTE = [0, 4, 5, 6, 7, 8, 9]
_COARSE_DIM = 5

_BASE_TRUNK = [(2, 2), (2, 3), (1, 3), (1, 4)]
_BASE_BRANCHES = [
    [(2, 0), (2, 1), (2, 2)],
    [(0, 1), (1, 1), (1, 2), (2, 2)],
    [(4, 1), (3, 1), (3, 2), (2, 2)],
]


def _compute_offsets(G: int, n: int, num_coarse: int) -> list[int] | None:
    if num_coarse == 1:
        return [1]
    first = 1
    last = G - n - 1
    if last < first:
        return None
    step = (last - first) / (num_coarse - 1)
    values = [int(round(first + i * step)) for i in range(num_coarse)]
    for i in range(len(values) - 1):
        if values[i + 1] - values[i] < n:
            return None
    return values


def _rotate_coord(coord: tuple[int, int], turns: int, dim: int) -> tuple[int, int]:
    r, c = coord
    turns %= 4
    for _ in range(turns):
        r, c = c, dim - 1 - r
    return int(r), int(c)


def _rotate_path(path: list[tuple[int, int]], turns: int, dim: int) -> list[tuple[int, int]]:
    return [_rotate_coord(cell, turns, dim) for cell in path]


def _render_path(
    grid: np.ndarray,
    path: list[tuple[int, int]],
    n: int,
    offsets_r: list[int],
    offsets_c: list[int],
) -> list[tuple[int, int]]:
    pixel_pos = []
    for cr, cc in path:
        pr, pc = int(offsets_r[cr]), int(offsets_c[cc])
        pixel_pos.append((pr, pc))
        grid[pr : pr + n, pc : pc + n] = _RIVER
    for i in range(len(pixel_pos) - 1):
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
    grid: np.ndarray,
    pixel_pos_i: tuple[int, int],
    pixel_pos_j: tuple[int, int],
    n: int,
    direction: str,
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
        hold_len = random.randint(1, min(5, max(1, c_hi - c_lo - 2 * remove)))
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
        hold_len = random.randint(1, min(5, max(1, r_hi - r_lo - 2 * remove)))
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


def _extend_target_to_border(
    grid: np.ndarray,
    coarse_target: tuple[int, int],
    pixel_target: tuple[int, int],
    n: int,
) -> tuple[int, int]:
    H, W = grid.shape
    cr, cc = coarse_target
    pr, pc = pixel_target
    if cc == _COARSE_DIM - 1:
        grid[pr : pr + n, pc:W] = _RIVER
        return pr, W - n
    if cc == 0:
        grid[pr : pr + n, 0 : pc + n] = _RIVER
        return pr, 0
    if cr == 0:
        grid[0 : pr + n, pc : pc + n] = _RIVER
        return 0, pc
    assert cr == _COARSE_DIM - 1
    grid[pr:H, pc : pc + n] = _RIVER
    return H - n, pc


def _gen_attempt() -> np.ndarray | None:
    n = random.choice([2, 3])
    min_size = 14 if n == 2 else 17
    G = random.randint(min_size, 20)

    offsets_r = _compute_offsets(G, n, _COARSE_DIM)
    offsets_c = _compute_offsets(G, n, _COARSE_DIM)
    if offsets_r is None or offsets_c is None:
        return None

    turns = random.randint(0, 3)
    trunk = _rotate_path(_BASE_TRUNK, turns, _COARSE_DIM)
    branches = [_rotate_path(path, turns, _COARSE_DIM) for path in _BASE_BRANCHES]
    coarse_target = trunk[-1]

    grid = np.full((G, G), _BG, dtype=int)
    trunk_pixels = _render_path(grid, trunk, n, offsets_r, offsets_c)
    branch_pixels = [_render_path(grid, path, n, offsets_r, offsets_c) for path in branches]

    target_r0, target_c0 = _extend_target_to_border(grid, coarse_target, trunk_pixels[-1], n)

    mover_colors = random.sample(_MOVER_PALETTE, len(branches))
    blocked_count = random.choice([1, 2])
    blocked_ids = set(random.sample(range(len(branches)), blocked_count))
    expected_status = {}

    for idx, pixels in enumerate(branch_pixels):
        if idx in blocked_ids:
            pr1, pc1 = pixels[-2]
            pr2, pc2 = pixels[-1]
            direction = "h" if pr1 == pr2 else "v"
            _add_chokepoint(grid, (pr1, pc1), (pr2, pc2), n, direction)
            expected_status[mover_colors[idx]] = "stuck"
        else:
            expected_status[mover_colors[idx]] = "reached"

    for idx, pixels in enumerate(branch_pixels):
        sr, sc = pixels[0]
        grid[sr : sr + n, sc : sc + n] = mover_colors[idx]

    grid[target_r0 : target_r0 + n, target_c0 : target_c0 + n] = _DEST

    if any(grid[r, c] != _BG for r, c in ((0, 0), (0, G - 1), (G - 1, 0), (G - 1, G - 1))):
        return None

    try:
        output, meta = _solve_instance(grid)
    except Exception:
        return None

    if len(meta["movers"]) != 3:
        return None

    status_by_color = {item["color"]: item["status"] for item in meta["movers"]}
    if status_by_color != expected_status:
        return None

    reached = sum(item["status"] == "reached" for item in meta["movers"])
    stuck = sum(item["status"] == "stuck" for item in meta["movers"])
    if reached == 0 or stuck == 0:
        return None

    for mover in meta["movers"]:
        sr, sc = mover["start"]
        if mover["status"] == "stuck" and (sr, sc) == mover["final"]:
            return None

    # Ensure the output actually differs from the input in multiple branch-local places.
    delta = int(np.count_nonzero(output != grid))
    if delta < n * n * 3:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("Failed to generate valid input after 400 attempts")
