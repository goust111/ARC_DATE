# Variant of 332f06d7
# Dimension: D9 — 多方块移动（基数关系） (multi_step)
# Change: 1:1单一移动体到单一目标... → 多对多独立移动

from __future__ import annotations

from collections import Counter, defaultdict, deque
import random

import numpy as np
from scipy.ndimage import label as scipy_label

from arc_utils import bbox_h, bbox_w, find_components


# Design analysis for the D9 variant
# a) Concrete behavior:
#    - The blue river is now a shared connected network with 2-3 horizontal lanes linked by one
#      vertical trunk. Every non-blue object is still an n×n solid square.
#    - Each pair uses one unique color that appears exactly twice: one interior square (the mover)
#      and one border-touching square (the target). The solver must pair them by visible color
#      equality, not by color id ordering.
#    - Old flow on one lane: black square -> red square. New flow on one lane: color X square ->
#      same-color edge square. Multiple such lanes coexist at once. A lane may contain a narrow
#      choke between the mover and the trunk, making that specific mover stop early while other
#      colors still move on the same blue network.
# b) Complexity calibration:
#    - Using 2-3 colors keeps the multiplicity visible without making matching ambiguous.
#    - All squares keep the same side length n, so the only new burden is multi-pair tracking.
#    - Targets always touch a border while movers never do, which provides a visible role cue.
# c) Old implicit assumptions that must change:
#    - Old code assumed exactly two squares of the relevant size and sorted them by numeric color.
#      That breaks immediately once several colors are present.
#    - Old traversability only admitted river + the chosen mover/destination colors. In the
#      variant, every colored square must be treated as empty river during path planning because
#      all pairs move independently and do not block one another.
#    - Old output wrote exactly one moved square. The variant must clear all mover starts first,
#      then place all final mover positions.
# d) Placement legality:
#    - Generator keeps one common square size n for all pairs.
#    - Lane rows are separated by at least one blank row, movers sit one cell away from a side
#      border, and targets sit flush with the opposite border. This prevents overlaps.
#    - Chokepoints are carved only on a lane's private segment between a mover and the trunk, so
#      they do not interfere with other pairs and remain solver-safe.
# e) Visual observability:
#    - Pairing uses same-color repetition, which is directly visible.
#    - Targets are the only same-color squares touching an outer border; movers are interior.
#    - Mixtures of reachable and stuck lanes make the new multi-pair rule visible across examples.
# f) Visual cleanliness:
#    - The network is a ladder/trunk layout rather than arbitrary overlap. Each lane has a single
#      colored mover and a single colored target, so the scene stays legible.
# g) Color inference:
#    - No hidden palette ordering is used. Colors only encode pair identity via equality, and the
#      generator uses a small subset of distinct colors with exactly one repeated pair each.


_BG = 3
_RIVER = 1
_PAIR_COLORS = [0, 2, 4, 5, 6, 7, 8, 9]


def _touches_edge(r: int, c: int, n: int, H: int, W: int) -> bool:
    return r == 0 or c == 0 or r + n == H or c + n == W


def _find_solid_squares(g: np.ndarray, bg: int) -> list[dict]:
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


def _build_feasible(trav: np.ndarray, n: int) -> np.ndarray:
    H, W = trav.shape
    h2, w2 = H - n + 1, W - n + 1
    assert not (h2 <= 0 or w2 <= 0)
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


def _extract_scene(grid: np.ndarray) -> tuple[np.ndarray, int, int, int, list[dict], np.ndarray]:
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(Counter([int(g[0, 0]), int(g[0, -1]), int(g[-1, 0]), int(g[-1, -1])]).most_common(1)[0][0])

    squares = _find_solid_squares(g, bg)
    assert not (len(squares) < 4)

    by_n: dict[int, list[dict]] = defaultdict(list)
    for sq in squares:
        by_n[int(sq["n"])].append(sq)

    best_n = None
    best_pairs: list[dict] = []
    for nside, group in by_n.items():
        by_col: dict[int, list[dict]] = defaultdict(list)
        for sq in group:
            by_col[int(sq["col"])].append(sq)

        pairs: list[dict] = []
        for col, comps in by_col.items():
            if len(comps) != 2:
                continue
            edge = [sq for sq in comps if _touches_edge(int(sq["r0"]), int(sq["c0"]), int(nside), H, W)]
            inner = [sq for sq in comps if not _touches_edge(int(sq["r0"]), int(sq["c0"]), int(nside), H, W)]
            if len(edge) != 1 or len(inner) != 1:
                continue
            mover = inner[0]
            target = edge[0]
            pairs.append(
                {
                    "col": int(col),
                    "sr": int(mover["r0"]),
                    "sc": int(mover["c0"]),
                    "tr": int(target["r0"]),
                    "tc": int(target["c0"]),
                    "n": int(nside),
                }
            )
        if best_n is None or (len(pairs), int(nside)) > (len(best_pairs), int(best_n)):
            best_n = int(nside)
            best_pairs = pairs

    assert not (best_n is None)
    assert not (len(best_pairs) < 2)

    square_cols = {int(pair["col"]) for pair in best_pairs}
    counts = Counter(map(int, g.flatten()))
    river = None
    for col, _ct in counts.most_common():
        if int(col) == int(bg):
            continue
        if int(col) in square_cols:
            continue
        river = int(col)
        break
    assert not (river is None)

    trav = g == int(river)
    for col in square_cols:
        trav |= g == int(col)
    feasible = _build_feasible(trav, int(best_n))

    for pair in best_pairs:
        assert not (not bool(feasible[int(pair["sr"]), int(pair["sc"])]))

    best_pairs.sort(key=lambda pair: (int(pair["sr"]), int(pair["sc"]), int(pair["tr"]), int(pair["tc"])))
    return g, bg, int(river), int(best_n), best_pairs, feasible


def _choose_final_position(feasible: np.ndarray, sr: int, sc: int, tr: int, tc: int) -> tuple[int, int]:
    dist = _bfs_placement(feasible, sr, sc)
    if 0 <= tr < feasible.shape[0] and 0 <= tc < feasible.shape[1] and feasible[tr, tc] and dist[tr, tc] >= 0:
        return int(tr), int(tc)

    best = -1
    out_r, out_c = int(sr), int(sc)
    for r in range(feasible.shape[0]):
        for c in range(feasible.shape[1]):
            if int(dist[r, c]) > best:
                best = int(dist[r, c])
                out_r, out_c = int(r), int(c)
    return out_r, out_c


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move several same-sized squares independently through one shared river network.

    Input structure:
    - The corner-majority background surrounds a connected blue channel network.
    - Each active color appears twice as equal-sized solid squares: one interior mover and one
      border-touching target of the same color.

    Rule:
    - For every color pair, the interior square slides through the blue network toward its
      same-color border square.
    - Motion is computed on n×n placements: a move is legal only when the whole square fits inside
      blue pixels or any colored square location. Different colors do not block one another.
    - If the matching target is reachable, the mover ends there; otherwise it stops at its farthest
      reachable placement. All original mover positions turn into river.

    Key insight:
    - The old one-pair river search must now be repeated for each visible color match, while the
      traversable mask treats every colored square as passable so the pairs behave independently.
    """
    g, _bg, river, n, pairs, feasible = _extract_scene(grid)

    out = g.copy()
    finals: list[tuple[int, int, int]] = []
    for pair in pairs:
        sr, sc = int(pair["sr"]), int(pair["sc"])
        tr, tc = int(pair["tr"]), int(pair["tc"])
        out[sr : sr + n, sc : sc + n] = int(river)
        fr, fc = _choose_final_position(feasible, sr, sc, tr, tc)
        finals.append((fr, fc, int(pair["col"])))

    for fr, fc, col in finals:
        out[fr : fr + n, fc : fc + n] = int(col)
    return out


def _river_connected(grid: np.ndarray, colors: list[int]) -> bool:
    mask = grid == _RIVER
    for col in colors:
        mask |= grid == int(col)
    labeled, num = scipy_label(mask.astype(int))
    return bool(mask.any()) and int(num) == 1


def _sample_lane_rows(G: int, n: int, k: int) -> list[int] | None:
    candidates = list(range(1, G - n))
    for _ in range(200):
        rows = sorted(random.sample(candidates, k))
        if rows[-1] + n > G - 1:
            continue
        if all(rows[i + 1] - rows[i] >= n + 1 for i in range(k - 1)):
            return rows
    return None


def _add_horizontal_chokepoint(grid: np.ndarray, row: int, mover_c: int, trunk_c: int, n: int) -> bool:
    lo = min(mover_c, trunk_c) + n + 1
    hi = max(mover_c, trunk_c) - 1
    if hi < lo:
        return False
    gate_w = random.randint(1, min(3, hi - lo + 1))
    gate_c = random.randint(lo, hi - gate_w + 1)
    narrow_h = random.randint(1, n - 1)
    cut = n - narrow_h
    from_top = random.randint(0, 1) == 0
    if from_top:
        grid[row : row + cut, gate_c : gate_c + gate_w] = _BG
    else:
        grid[row + n - cut : row + n, gate_c : gate_c + gate_w] = _BG
    return True


def _plan_statuses(k: int) -> set[int]:
    if random.random() < 0.3:
        return set()
    max_stuck = max(1, min(2, k - 1))
    count = random.randint(1, max_stuck)
    return set(random.sample(range(k), count))


def _analyze_pairs(grid: np.ndarray) -> list[dict]:
    _g, _bg, _river, n, pairs, feasible = _extract_scene(grid)
    analysis: list[dict] = []
    for pair in pairs:
        fr, fc = _choose_final_position(feasible, int(pair["sr"]), int(pair["sc"]), int(pair["tr"]), int(pair["tc"]))
        analysis.append(
            {
                "col": int(pair["col"]),
                "start": (int(pair["sr"]), int(pair["sc"])),
                "target": (int(pair["tr"]), int(pair["tc"])),
                "final": (int(fr), int(fc)),
                "reached": (int(fr), int(fc)) == (int(pair["tr"]), int(pair["tc"])),
                "n": int(n),
            }
        )
    return analysis


def _expected_output(grid: np.ndarray, analysis: list[dict]) -> np.ndarray:
    out = np.array(grid, copy=True)
    for item in analysis:
        sr, sc = item["start"]
        n = int(item["n"])
        out[sr : sr + n, sc : sc + n] = _RIVER
    for item in analysis:
        fr, fc = item["final"]
        n = int(item["n"])
        out[fr : fr + n, fc : fc + n] = int(item["col"])
    return out


def _gen_attempt() -> np.ndarray | None:
    n = random.choice([2, 3])
    k = random.choice([2, 3])
    width_min = 4 * n + 5
    height_min = 2 + k * n + (k - 1)
    G_min = max(12, width_min, height_min)
    if G_min > 20:
        return None
    G = random.randint(G_min, 20)

    rows = _sample_lane_rows(G, n, k)
    if rows is None:
        return None

    trunk_left = n + 3
    trunk_right = G - 2 * n - 3
    if trunk_left > trunk_right:
        return None
    trunk_c = random.randint(trunk_left, trunk_right)

    colors = random.sample(_PAIR_COLORS, k)
    sides = [random.choice(["left", "right"]) for _ in range(k)]
    if len(set(sides)) == 1:
        sides[0] = "left"
        sides[-1] = "right"
    stuck_idx = _plan_statuses(k)

    grid = np.full((G, G), _BG, dtype=int)
    top = rows[0]
    bottom = rows[-1] + n
    grid[top:bottom, trunk_c : trunk_c + n] = _RIVER

    pairs = []
    for idx, row in enumerate(rows):
        grid[row : row + n, :] = _RIVER
        target_left = sides[idx] == "left"
        target_c = 0 if target_left else G - n
        mover_c = G - n - 1 if target_left else 1

        if idx in stuck_idx:
            if not _add_horizontal_chokepoint(grid, int(row), int(mover_c), int(trunk_c), int(n)):
                return None

        pairs.append({"row": int(row), "mover_c": int(mover_c), "target_c": int(target_c), "col": int(colors[idx])})

    for pair in pairs:
        row = int(pair["row"])
        mover_c = int(pair["mover_c"])
        target_c = int(pair["target_c"])
        col = int(pair["col"])
        grid[row : row + n, mover_c : mover_c + n] = col
        grid[row : row + n, target_c : target_c + n] = col

    if not _river_connected(grid, colors):
        return None

    analysis = _analyze_pairs(grid)
    if len(analysis) != k:
        return None
    reached = sum(int(item["reached"]) for item in analysis)
    moved = sum(int(item["final"] != item["start"]) for item in analysis)
    if moved < k:
        return None
    if reached == 0 or reached == k:
        if k >= 3:
            return None

    if not np.array_equal(main(grid), _expected_output(grid, analysis)):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("Failed to generate valid input after 400 attempts")
