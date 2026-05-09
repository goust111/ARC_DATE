# Combo variant of 332f06d7 (index 2)
# combo_spec: D1/2+D2/1
# Name: 重力流动+渐变宽度

from __future__ import annotations

from collections import Counter, defaultdict, deque
import random

import numpy as np
from scipy.ndimage import label as scipy_label

from arc_utils import bbox_h, bbox_w, find_components


_BG, _RIVER, _MOVER, _DEST = 3, 1, 0, 2


# Design analysis for the combo variant.
# combo_variant_spec:
#   a_concretization_and_trace:
#     - Objects stay the same as the source task: green background, one blue traversable region,
#       one black n x n mover, one red n x n destination of the same size.
#     - The corridor is no longer interpreted as an undirected pipe. We reason on the placement
#       grid of the mover's top-left corner, but motion is gravity-biased:
#         1. If the placement directly below is feasible, the mover must go down.
#         2. Only when below is blocked may it spread left/right along the current shelf.
#         3. Upward motion is never legal, so elevated side pockets are not gravity-reachable.
#     - The final descent is a visibly tapered vertical shaft. One wall stays fixed while the
#       other steps inward over several rows. Reachable examples taper from width n+extra to n.
#       Stuck examples taper from width n+extra to some width in 1..n-1, so the mover stops at
#       the last gravity-reachable placement above the first too-narrow slice.
#     - Small trace with n=3 on the placement grid:
#       original task: any feasible neighbor in {up,down,left,right} is allowed, so if a side
#       branch goes upward then BFS can climb it.
#       combo task: start drops down a shaft, then drifts on a shelf only when blocked below,
#       then drops again into a taper. When the shaft widths look like 5,4,3,2,... the first
#       row whose local clearance falls below 3 blocks further falling. The black square stops
#       just above that point even if a thin blue spine still connects to the red square.
#   b_complexity_calibration:
#     - Every sample has at least two vertical drops and two horizontal shelves, so the visual
#       language is clearly "gravity cave", not a slightly modified winding corridor.
#     - Every sample also contains an elevated side alcove branching upward from the lower shelf.
#       This keeps gravity visually relevant and prevents the combo from collapsing into a pure
#       width-only problem.
#     - Width tapering is monotone and uses only a few parameters: side in {left,right}, extra
#       width in 1..3, and end width either n or 1..n-1. That keeps the pattern learnable.
#   c_implicit_assumption_audit_of_original_main:
#     - Detecting solid squares by monochromatic connected components is still valid.
#     - The original assumption "movement is undirected BFS on feasible placements" is invalid:
#       gravity removes upward motion and makes down take priority over sideways motion.
#     - The original blocked-case fallback "maximum BFS distance from start" is also invalid.
#       In the combo, the meaningful stop is the deepest gravity-reachable placement; among ties
#       we choose the one nearest the destination so the mover settles beside the taper bottleneck.
#     - The original solver used color ordering to distinguish the two squares. That is preserved
#       as a fallback, but the generator explicitly keeps black as mover and red as destination.
#   d_placement_legality:
#     - The generator first constructs an abstract placement graph, renders it to pixels by
#       stamping n x n windows, then repaints the final vertical shaft with a tapered width
#       profile and recomputes the true feasible-placement mask from the final pixels.
#     - Acceptance checks are based on the recomputed mask, so any accidental placements created
#       by overlaps or destroyed by the taper are caught before a sample is accepted.
#     - The mover and destination are placed on distinct feasible n x n windows and never overlap.
#   e_visual_observability:
#     - A human can see repeated "fall, shelf, fall" structure across examples and also see that
#       the last shaft becomes narrower row by row. In blocked examples the mover stops just above
#       the narrow segment while the red square remains below it.
#     - The elevated side alcove is visible but unchanged in the output, reinforcing that the move
#       rule is not simple connectivity through all blue pixels.
#   f_visual_cleanliness:
#     - There is still exactly one connected traversable blue region and only one black/red pair.
#       The branch is orthogonal and sparse, so it reads as a side pocket rather than clutter.
#     - The taper is applied only on the final vertical shaft, avoiding noisy width changes across
#       the whole cave.
#   g_change_inferability:
#     - No hidden color remapping is introduced. The transformation is inferable from geometry:
#       gravity-like descent plus a monotone narrowing that eventually becomes too tight.
#   new_main_assumptions_and_generator_guarantees:
#     - There is exactly one intended black mover and one intended red destination among the solid
#       same-sized squares; generator guarantees this.
#     - The destination window itself is always feasible, even in blocked cases. The break happens
#       in the connecting shaft above it, not inside the destination square.
#     - Every accepted sample satisfies gravity_reach_size < bfs_reach_size, so the gravity rule
#       is behaviorally active rather than decorative.
#     - Every accepted blocked sample also differs from the original task's "undirected BFS +
#       farthest reachable placement" outcome, so the fusion does not collapse back to the source.


def _detect_entities(g: np.ndarray) -> tuple[int, int, dict, dict]:
    h, w = g.shape
    bg = int(Counter([int(g[0, 0]), int(g[0, w - 1]), int(g[h - 1, 0]), int(g[h - 1, w - 1])]).most_common(1)[0][0])

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
    if len(squares) < 2:
        raise ValueError("need at least two solid squares")

    by_n: dict[int, list[dict]] = defaultdict(list)
    for sq in squares:
        by_n[int(sq["n"])].append(sq)
    nside = max(by_n.items(), key=lambda kv: (len(kv[1]), kv[0]))[0]
    cand = by_n[int(nside)]
    if len(cand) < 2:
        raise ValueError("need two same-sized squares")

    by_color = {int(s["col"]): s for s in cand}
    if _MOVER in by_color and _DEST in by_color:
        mover = by_color[_MOVER]
        dest = by_color[_DEST]
    else:
        cand = sorted(cand, key=lambda s: (int(s["col"]), int(s["r0"]), int(s["c0"])))
        mover, dest = cand[0], cand[1]

    counts = Counter(map(int, g.flatten()))
    river = None
    for col, _ in counts.most_common():
        if int(col) == int(bg):
            continue
        if int(col) in {int(mover["col"]), int(dest["col"])}:
            continue
        river = int(col)
        break
    if river is None:
        raise ValueError("missing river color")
    return bg, river, mover, dest


def _compute_feasible(g: np.ndarray, river: int, mover_col: int, dest_col: int, n: int) -> np.ndarray:
    trav = (g == int(river)) | (g == int(mover_col)) | (g == int(dest_col))
    h, w = g.shape
    h2, w2 = h - n + 1, w - n + 1
    feasible = np.zeros((h2, w2), dtype=bool)
    for r in range(h2):
        for c in range(w2):
            if bool(trav[r : r + n, c : c + n].all()):
                feasible[r, c] = True
    return feasible


def _gravity_neighbors(feasible: np.ndarray, r: int, c: int) -> list[tuple[int, int]]:
    h2, w2 = feasible.shape
    if r + 1 < h2 and bool(feasible[r + 1, c]):
        return [(int(r + 1), int(c))]
    out: list[tuple[int, int]] = []
    if c - 1 >= 0 and bool(feasible[r, c - 1]):
        out.append((int(r), int(c - 1)))
    if c + 1 < w2 and bool(feasible[r, c + 1]):
        out.append((int(r), int(c + 1)))
    return out


def _gravity_reachable(feasible: np.ndarray, sr: int, sc: int) -> np.ndarray:
    h2, w2 = feasible.shape
    reach = np.zeros((h2, w2), dtype=bool)
    q = deque([(int(sr), int(sc))])
    reach[int(sr), int(sc)] = True
    while q:
        r, c = q.popleft()
        for rr, cc in _gravity_neighbors(feasible, int(r), int(c)):
            if not bool(reach[rr, cc]):
                reach[rr, cc] = True
                q.append((int(rr), int(cc)))
    return reach


def _bfs_distance(feasible: np.ndarray, sr: int, sc: int) -> np.ndarray:
    h2, w2 = feasible.shape
    dist = -np.ones((h2, w2), dtype=int)
    q = deque([(int(sr), int(sc))])
    dist[int(sr), int(sc)] = 0
    while q:
        r, c = q.popleft()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            rr, cc = int(r + dr), int(c + dc)
            if 0 <= rr < h2 and 0 <= cc < w2 and bool(feasible[rr, cc]) and int(dist[rr, cc]) == -1:
                dist[rr, cc] = int(dist[int(r), int(c)] + 1)
                q.append((int(rr), int(cc)))
    return dist


def _select_gravity_stop(reach: np.ndarray, dest: tuple[int, int]) -> tuple[int, int]:
    dr, dc = map(int, dest)
    cells = np.argwhere(reach)
    if len(cells) == 0:
        raise ValueError("empty gravity reachable set")
    best_key = None
    best_pos = None
    for r, c in cells:
        rr = int(r)
        cc = int(c)
        key = (rr, -(abs(rr - dr) + abs(cc - dc)), -abs(cc - dc), -cc)
        if best_key is None or key > best_key:
            best_key = key
            best_pos = (rr, cc)
    assert best_pos is not None
    return best_pos


def _select_original_stop(dist: np.ndarray, start: tuple[int, int]) -> tuple[int, int]:
    sr, sc = map(int, start)
    best = -1
    tr, tc = sr, sc
    for r in range(dist.shape[0]):
        for c in range(dist.shape[1]):
            if int(dist[r, c]) > best:
                best = int(dist[r, c])
                tr, tc = int(r), int(c)
    return tr, tc


def main(grid: np.ndarray) -> np.ndarray:
    """
    Combo rule for 332f06d7:
    - motion is gravity-driven on the n x n placement grid;
    - the final shaft may gradually taper;
    - if the destination is not gravity-reachable, stop at the deepest reachable placement
      nearest the destination, which corresponds to the last spot before the taper becomes
      too narrow for the square.
    """
    g = np.array(grid, dtype=int)
    _bg, river, mover, dest = _detect_entities(g)
    n = int(mover["n"])
    sr, sc = int(mover["r0"]), int(mover["c0"])
    dr, dc = int(dest["r0"]), int(dest["c0"])

    feasible = _compute_feasible(g, river, int(mover["col"]), int(dest["col"]), n)
    if not bool(feasible[sr, sc]):
        raise ValueError("mover start is not traversable")

    reach = _gravity_reachable(feasible, sr, sc)
    if 0 <= dr < feasible.shape[0] and 0 <= dc < feasible.shape[1] and bool(reach[dr, dc]):
        tr, tc = int(dr), int(dc)
    else:
        tr, tc = _select_gravity_stop(reach, (dr, dc))

    out = g.copy()
    out[sr : sr + n, sc : sc + n] = int(river)
    out[tr : tr + n, tc : tc + n] = int(mover["col"])
    return out


def _add_vertical(feasible: np.ndarray, c: int, r0: int, r1: int) -> None:
    lo, hi = sorted((int(r0), int(r1)))
    feasible[lo : hi + 1, int(c)] = True


def _add_horizontal(feasible: np.ndarray, r: int, c0: int, c1: int) -> None:
    lo, hi = sorted((int(c0), int(c1)))
    feasible[int(r), lo : hi + 1] = True


def _render_from_placements(feasible: np.ndarray, n: int) -> np.ndarray:
    h2, w2 = feasible.shape
    grid = np.full((h2 + n - 1, w2 + n - 1), _BG, dtype=int)
    for r, c in np.argwhere(feasible):
        rr = int(r)
        cc = int(c)
        grid[rr : rr + n, cc : cc + n] = _RIVER
    return grid


def _river_connected(grid: np.ndarray) -> bool:
    mask = (grid == _RIVER) | (grid == _MOVER) | (grid == _DEST)
    if not mask.any():
        return False
    _labels, num = scipy_label(mask.astype(int))
    return int(num) == 1


def _profile_widths(body_len: int, start_w: int, end_w: int, taper_len: int) -> list[int]:
    assert body_len >= taper_len >= 1
    if taper_len == 1:
        lead = [int(end_w)]
    else:
        lead = []
        diff = int(start_w - end_w)
        for i in range(taper_len):
            w = int(start_w - (diff * i) // (taper_len - 1))
            lead.append(int(w))
        lead[-1] = int(end_w)
    return lead + [int(end_w)] * (body_len - taper_len)


def _apply_vertical_taper(
    grid: np.ndarray,
    pr: int,
    pc: int,
    n: int,
    dest_r: int,
    side: int,
    extra: int,
    end_w: int,
    taper_len: int,
) -> bool:
    h, w = grid.shape
    body_len = int(dest_r - pr)
    if body_len < 1:
        return False
    max_w = int(n + extra)
    if side < 0:
        c0 = int(pc - extra)
        c1 = int(pc + n)
    else:
        c0 = int(pc)
        c1 = int(pc + n + extra)
    if c0 < 0 or c1 > w:
        return False

    grid[pr:h, c0:c1] = _BG
    widths = _profile_widths(body_len, max_w, int(end_w), int(taper_len))
    for offset, r in enumerate(range(pr, dest_r)):
        cur_w = int(widths[offset])
        if side < 0:
            cc0 = int(pc + n - cur_w)
        else:
            cc0 = int(pc)
        grid[int(r), cc0 : cc0 + cur_w] = _RIVER
    grid[dest_r:h, pc : pc + n] = _RIVER
    return True


def _squares_overlap(a: tuple[int, int], b: tuple[int, int], n: int) -> bool:
    return abs(int(a[0]) - int(b[0])) < n and abs(int(a[1]) - int(b[1])) < n


def _build_abstract_layout(p: int) -> tuple[np.ndarray, tuple[int, int], tuple[int, int], dict] | None:
    feasible = np.zeros((p, p), dtype=bool)

    sr = 1
    sc = random.randint(2, p - 3)

    shelf1_hi = p - 8
    if shelf1_hi < 3:
        return None
    shelf1_r = random.randint(3, shelf1_hi)
    drop1_candidates = [c for c in range(2, p - 2) if abs(c - sc) >= 2]
    if not drop1_candidates:
        return None
    drop1_c = random.choice(drop1_candidates)
    shelf1_l = max(1, min(sc, drop1_c) - random.randint(0, 2))
    shelf1_rgt = min(p - 2, max(sc, drop1_c) + random.randint(0, 2))

    shelf2_lo = shelf1_r + 3
    shelf2_hi = p - 5
    if shelf2_lo > shelf2_hi:
        return None
    shelf2_r = random.randint(shelf2_lo, shelf2_hi)
    drop2_candidates = [c for c in range(2, p - 2) if abs(c - drop1_c) >= 2]
    if not drop2_candidates:
        return None
    drop2_c = random.choice(drop2_candidates)
    shelf2_l = max(1, min(drop1_c, drop2_c) - random.randint(0, 2))
    shelf2_rgt = min(p - 2, max(drop1_c, drop2_c) + random.randint(0, 2))

    dest = (p - 1, drop2_c)
    mover = (sr, sc)

    _add_vertical(feasible, sc, sr, shelf1_r)
    _add_horizontal(feasible, shelf1_r, shelf1_l, shelf1_rgt)
    _add_vertical(feasible, drop1_c, shelf1_r, shelf2_r)
    _add_horizontal(feasible, shelf2_r, shelf2_l, shelf2_rgt)
    _add_vertical(feasible, drop2_c, shelf2_r, dest[0])

    branch_candidates = [c for c in range(shelf2_l, shelf2_rgt + 1) if abs(c - drop2_c) >= 1]
    if not branch_candidates:
        return None
    branch_c = random.choice(branch_candidates)
    branch_top_lo = shelf1_r + 1
    branch_top_hi = shelf2_r - 2
    if branch_top_lo > branch_top_hi:
        return None
    branch_top = random.randint(branch_top_lo, branch_top_hi)
    _add_vertical(feasible, branch_c, branch_top, shelf2_r)

    if branch_c < drop2_c:
        branch_end_lo = 1
        branch_end_hi = max(1, branch_c - 1)
        branch_end = random.randint(branch_end_lo, branch_end_hi)
    else:
        branch_end_lo = min(p - 2, branch_c + 1)
        branch_end_hi = p - 2
        if branch_end_lo > branch_end_hi:
            return None
        branch_end = random.randint(branch_end_lo, branch_end_hi)
    _add_horizontal(feasible, branch_top, branch_c, branch_end)

    taper = {
        "tail_r": int(shelf2_r),
        "tail_c": int(drop2_c),
    }
    return feasible, mover, dest, taper


def _choose_taper_plan(grid: np.ndarray, tail_r: int, tail_c: int, n: int, reached: bool) -> dict | None:
    h, w = grid.shape
    dest_r = h - n
    body_len = int(dest_r - tail_r)
    if body_len < 4:
        return None

    candidates: list[dict] = []
    room_left = int(tail_c)
    room_right = int(w - (tail_c + n))
    for side, room in [(-1, room_left), (1, room_right)]:
        if room < 1:
            continue
        max_extra = min(int(room), max(1, min(3, n + 1)))
        for extra in range(1, max_extra + 1):
            start_w = int(n + extra)
            end_widths = [n] if reached else list(range(1, n))
            for end_w in end_widths:
                min_taper = max(3, start_w - end_w + 1)
                if body_len < min_taper:
                    continue
                for taper_len in range(min_taper, body_len + 1):
                    candidates.append(
                        {
                            "side": int(side),
                            "extra": int(extra),
                            "start_w": int(start_w),
                            "end_w": int(end_w),
                            "taper_len": int(taper_len),
                            "dest_r": int(dest_r),
                        }
                    )
    if not candidates:
        return None
    candidates.sort(key=lambda plan: (plan["start_w"] - plan["end_w"], plan["extra"], plan["taper_len"]), reverse=True)
    return random.choice(candidates[: min(10, len(candidates))])


def _analyze_instance(grid: np.ndarray) -> dict:
    g = np.array(grid, dtype=int)
    _bg, river, mover, dest = _detect_entities(g)
    n = int(mover["n"])
    sr, sc = int(mover["r0"]), int(mover["c0"])
    dr, dc = int(dest["r0"]), int(dest["c0"])
    feasible = _compute_feasible(g, river, int(mover["col"]), int(dest["col"]), n)
    if not bool(feasible[sr, sc]):
        raise ValueError("start not feasible")
    if not bool(feasible[dr, dc]):
        raise ValueError("dest not feasible")
    grav = _gravity_reachable(feasible, sr, sc)
    dist = _bfs_distance(feasible, sr, sc)
    final = (dr, dc) if bool(grav[dr, dc]) else _select_gravity_stop(grav, (dr, dc))
    original = (dr, dc) if int(dist[dr, dc]) >= 0 else _select_original_stop(dist, (sr, sc))
    return {
        "n": int(n),
        "start": (int(sr), int(sc)),
        "dest": (int(dr), int(dc)),
        "feasible": feasible,
        "gravity": grav,
        "dist": dist,
        "gravity_final": tuple(map(int, final)),
        "original_final": tuple(map(int, original)),
    }


def _accept_layout(grid: np.ndarray, reached: bool) -> dict | None:
    try:
        info = _analyze_instance(grid)
    except Exception:
        return None

    grav = info["gravity"]
    dist = info["dist"]
    dr, dc = info["dest"]
    sr, sc = info["start"]
    gravity_hits_dest = bool(grav[dr, dc])

    if bool(reached) != gravity_hits_dest:
        return None
    if not reached and int(dist[dr, dc]) >= 0:
        return None
    if int(np.sum(grav)) >= int(np.sum(dist >= 0)):
        return None
    if not reached and tuple(info["gravity_final"]) == tuple(info["original_final"]):
        return None

    out = grid.copy()
    n = int(info["n"])
    tr, tc = info["gravity_final"]
    out[sr : sr + n, sc : sc + n] = _RIVER
    out[tr : tr + n, tc : tc + n] = _MOVER
    if not np.array_equal(main(grid), out):
        return None

    meta = {
        "reached": bool(reached),
        "n": int(info["n"]),
        "gravity_cells": int(np.sum(grav)),
        "bfs_cells": int(np.sum(dist >= 0)),
        "gravity_final": tuple(map(int, info["gravity_final"])),
        "original_final": tuple(map(int, info["original_final"])),
        "dest": tuple(map(int, info["dest"])),
    }
    return meta


def _gen_attempt(return_meta: bool = False):
    reached = random.random() < 0.5
    n = random.choice([1, 2, 3, 4]) if reached else random.choice([2, 3, 4])

    p_min = 10
    p_max = 21 - n
    if p_min > p_max:
        return (None, None) if return_meta else None
    p = random.randint(p_min, min(16, p_max))

    layout = _build_abstract_layout(p)
    if layout is None:
        return (None, None) if return_meta else None
    abstract, mover, dest, taper = layout
    if _squares_overlap(mover, dest, n):
        return (None, None) if return_meta else None

    grid = _render_from_placements(abstract, n)
    if grid.shape[0] < 10 or grid.shape[0] > 20 or grid.shape[0] != grid.shape[1]:
        return (None, None) if return_meta else None

    plan = _choose_taper_plan(grid, int(taper["tail_r"]), int(taper["tail_c"]), n, reached)
    if plan is None:
        return (None, None) if return_meta else None
    ok = _apply_vertical_taper(
        grid,
        int(taper["tail_r"]),
        int(taper["tail_c"]),
        n,
        int(plan["dest_r"]),
        int(plan["side"]),
        int(plan["extra"]),
        int(plan["end_w"]),
        int(plan["taper_len"]),
    )
    if not ok:
        return (None, None) if return_meta else None

    river_ratio = float(np.mean(grid == _RIVER))
    if not (0.16 <= river_ratio <= 0.62):
        return (None, None) if return_meta else None
    corners = [int(grid[0, 0]), int(grid[0, -1]), int(grid[-1, 0]), int(grid[-1, -1])]
    if any(col != _BG for col in corners):
        return (None, None) if return_meta else None

    sr, sc = mover
    dr, dc = dest
    if not (0 <= sr <= grid.shape[0] - n and 0 <= sc <= grid.shape[1] - n):
        return (None, None) if return_meta else None
    if not (0 <= dr <= grid.shape[0] - n and 0 <= dc <= grid.shape[1] - n):
        return (None, None) if return_meta else None

    grid[sr : sr + n, sc : sc + n] = _MOVER
    grid[dr : dr + n, dc : dc + n] = _DEST
    if not _river_connected(grid):
        return (None, None) if return_meta else None

    accepted = _accept_layout(grid, reached)
    if accepted is None:
        return (None, None) if return_meta else None

    accepted.update(
        {
            "start": (int(sr), int(sc)),
            "taper_side": int(plan["side"]),
            "start_width": int(plan["start_w"]),
            "end_width": int(plan["end_w"]),
            "taper_len": int(plan["taper_len"]),
        }
    )
    if return_meta:
        return grid, accepted
    return grid


def generate_input():
    for _ in range(500):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("Failed to generate valid combo input after 500 attempts")
