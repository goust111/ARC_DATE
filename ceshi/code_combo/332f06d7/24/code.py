# Combo variant of 332f06d7 (index 24)
# combo_spec: D11/1+D12/2
# Name: 多色分段河流+管道网络

"""
Design analysis for the combo variant
=====================================

This fusion keeps the original "slide an n x n square by BFS over feasible placements"
mechanism, but combines two orthogonal changes:

1. D11/1 color filtering:
   the river is no longer one uniformly traversable color. Several river colors form one
   connected pipe picture, and the mover can travel only on the river color visibly
   supporting its start square.

2. D12/2 network topology:
   the pipe is no longer a single corridor. It is a small network with a branch and a loop,
   so the motion space is graph-like rather than a simple line.

a) Concretization and behavior trace
------------------------------------
Small example idea:
- A black 2x2 square starts on a yellow pipe segment.
- A red 2x2 destination sits on the border.
- The visible pipe picture contains:
  - a yellow trunk,
  - a yellow loop before the end,
  - a blue spur branching away,
  - and, in stuck cases, the final tail to the destination is blue rather than yellow.

Original logic on the same picture would conceptually do:
- pick one global river color,
- allow the whole pipe component,
- BFS through every river pixel regardless of segment color.

Combo logic does:
- explicitly find the black mover square and red destination square,
- infer each square's local support color from adjacent river pixels,
- replace the two squares conceptually by those support colors,
- allow only the mover's support color,
- BFS over feasible n x n placements inside that one-color subnetwork.

Difference:
- In reachable cases, the mover follows the correct-colored route through a real network,
  ignoring differently colored branches.
- In stuck cases, the full pipe network is still visually connected and even the old
  color-agnostic rule would reach the destination, but the combo rule stops at the last
  feasible placement before the final color transition.

b) Complexity calibration
-------------------------
- River colors are restricted to a tiny palette {1,4,8}. Only 2 or 3 appear per sample.
- The network shape is restricted to two coarse templates, varied only by rotation/mirror.
- The mover support color varies across samples, so the rule is not tied to a fixed color id.
- A differently colored spur is always present, so reachable samples do not collapse back to
  the single-color original task.
- In stuck samples, the destination tail is color-blocked rather than geometry-blocked, while
  the mover's own allowed subnetwork still contains a branch and a loop. This makes both
  dimensions simultaneously active.

c) Implicit assumption audit against the original main()
-------------------------------------------------------
Original assumptions that break:
- "The most common non-bg color is the river color."
  Invalid now because several river colors exist.
  Fix: infer mover/destination support colors locally and build a support grid.
- "Sorting square-like components by color is safe."
  Unsafe now because monochrome river patches may accidentally look square-ish.
  Fix: detect the mover as the unique solid black square and the destination as the unique
  solid red square.
- "All river pixels are traversable."
  Invalid now because only one visible river color is allowed for this mover.
- "Filling the origin with one global river color is correct."
  Invalid now because the origin must be restored with the mover's local support color.

Assumptions kept and re-guaranteed by the generator:
- Background still dominates the four corners.
- Exactly one solid black square and one solid red square of equal size exist.
- The mover start placement is feasible in its own support color.
- The destination touches at least one river side cell, so its support color is inferable.

New combo-specific assumptions and generator guarantees:
- The mover-reachable one-color subnetwork contains a branch and a loop, not merely the
  all-colors pipe picture. The generator measures this directly from feasible placements.
- In stuck samples, color-agnostic traversal can still reach the destination. Therefore the
  observed failure is genuinely caused by color filtering, not by a hidden geometric choke.

d) Placement legality
---------------------
- Coarse anchor offsets are spaced by at least n+1, so adjacency appears only where an edge
  is explicitly painted.
- Nodes are painted as n x n solid squares and edges only fill the gap between adjacent nodes.
- The destination is extended straight to one border using only background or same-color cells.
- The mover is placed on the start node after the network is painted, so it never overlaps the
  destination or corrupts support-color inference.

e) Visual observability
-----------------------
- A human can see that the black square sits on one river color and that other colors exist in
  the same pipe network.
- The mover-reachable route itself contains a loop/branch, so the network topology is not just
  decorative background.
- In stuck examples the destination lies beyond a visible color change on the final tail, while
  the full network remains connected. This is a strong clue that color, not mere geometry, is
  the blocking factor.

f) Visual cleanliness
---------------------
- Only four object roles are present: green background, multicolor pipe, black mover, red goal.
- The network uses compact templates instead of random dense graphs, so crossings and branches
  stay legible.
- Extra color variation is limited to a short spur and, in stuck cases, the destination tail.

g) Inferability of the change
-----------------------------
- The rule is structural: "the mover can move only on the pipe color it starts on."
- This is supported by repeated evidence because the mover support color changes between samples.
- The network dimension is also structural: visible branches/loops recur across samples in
  varying orientations, so the solver cannot rely on one fixed layout.
"""

from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np

from arc_utils import bbox_h, bbox_w, find_components


_BG = 3
_MOVER = 0
_DEST = 2
_RIVER_COLORS = [1, 4, 8]
_DIR4 = [(1, 0), (-1, 0), (0, 1), (0, -1)]


def _find_solid_square(grid: np.ndarray, color: int, bg: int) -> tuple[int, int, int]:
    comps = find_components(grid, bg=bg, colors=[color], connectivity=4, monochromatic=True, return_format="dict")
    squares: list[tuple[int, int, int]] = []
    for comp in comps:
        b = comp["bbox"]
        if bbox_h(b) != bbox_w(b):
            continue
        r0, c0, r1, c1 = map(int, b)
        patch = grid[r0 : r1 + 1, c0 : c1 + 1]
        if not bool((patch == int(color)).all()):
            continue
        squares.append((r0, c0, int(bbox_h(b))))
    assert len(squares) == 1
    return squares[0]


def _infer_support_color(grid: np.ndarray, r0: int, c0: int, n: int, forbidden: set[int]) -> int:
    H, W = grid.shape
    counts: Counter[int] = Counter()
    for c in range(c0, c0 + n):
        for rr in (r0 - 1, r0 + n):
            if 0 <= rr < H and 0 <= c < W:
                col = int(grid[rr, c])
                if col not in forbidden:
                    counts[col] += 1
    for r in range(r0, r0 + n):
        for cc in (c0 - 1, c0 + n):
            if 0 <= r < H and 0 <= cc < W:
                col = int(grid[r, cc])
                if col not in forbidden:
                    counts[col] += 1
    assert counts
    return int(counts.most_common(1)[0][0])


def _build_feasible(mask: np.ndarray, n: int) -> np.ndarray:
    H, W = mask.shape
    h2, w2 = H - n + 1, W - n + 1
    assert h2 > 0 and w2 > 0
    feasible = np.zeros((h2, w2), dtype=bool)
    for r in range(h2):
        for c in range(w2):
            feasible[r, c] = bool(mask[r : r + n, c : c + n].all())
    return feasible


def _bfs_placement(feasible: np.ndarray, sr: int, sc: int) -> np.ndarray:
    h2, w2 = feasible.shape
    dist = -np.ones((h2, w2), dtype=int)
    dist[int(sr), int(sc)] = 0
    q = deque([(int(sr), int(sc))])
    while q:
        r, c = q.popleft()
        for dr, dc in _DIR4:
            rr, cc = int(r + dr), int(c + dc)
            if 0 <= rr < h2 and 0 <= cc < w2 and bool(feasible[rr, cc]) and int(dist[rr, cc]) == -1:
                dist[rr, cc] = int(dist[r, c] + 1)
                q.append((rr, cc))
    return dist


def _pick_farthest(dist: np.ndarray, sr: int, sc: int) -> tuple[int, int]:
    best = -1
    tr, tc = int(sr), int(sc)
    for r in range(dist.shape[0]):
        for c in range(dist.shape[1]):
            if int(dist[r, c]) > best:
                best = int(dist[r, c])
                tr, tc = int(r), int(c)
    return tr, tc


def _reachable_topology_metrics(feasible: np.ndarray, sr: int, sc: int) -> tuple[int, int, int]:
    if not bool(feasible[int(sr), int(sc)]):
        return 0, 0, 0
    seen = {(int(sr), int(sc))}
    q = deque([(int(sr), int(sc))])
    nodes: list[tuple[int, int]] = []
    while q:
        r, c = q.popleft()
        nodes.append((int(r), int(c)))
        for dr, dc in _DIR4:
            rr, cc = int(r + dr), int(c + dc)
            if 0 <= rr < feasible.shape[0] and 0 <= cc < feasible.shape[1] and bool(feasible[rr, cc]) and (rr, cc) not in seen:
                seen.add((rr, cc))
                q.append((rr, cc))

    node_set = set(nodes)
    branch_count = 0
    edge_count = 0
    for r, c in nodes:
        deg = 0
        for dr, dc in _DIR4:
            rr, cc = int(r + dr), int(c + dc)
            if (rr, cc) in node_set:
                deg += 1
        if deg >= 3:
            branch_count += 1
        for dr, dc in [(1, 0), (0, 1)]:
            rr, cc = int(r + dr), int(c + dc)
            if (rr, cc) in node_set:
                edge_count += 1
    cycle_rank = int(edge_count - len(nodes) + 1)
    return branch_count, cycle_rank, len(nodes)


def _river_colors(grid: np.ndarray, bg: int) -> list[int]:
    return sorted(int(col) for col in np.unique(grid) if int(col) not in {bg, _MOVER, _DEST})


def _analyze(grid: np.ndarray) -> dict:
    g = np.array(grid, dtype=int)
    H, W = g.shape
    assert H == W

    bg = int(Counter([int(g[0, 0]), int(g[0, -1]), int(g[-1, 0]), int(g[-1, -1])]).most_common(1)[0][0])
    mover_r, mover_c, n = _find_solid_square(g, _MOVER, bg)
    dest_r, dest_c, n2 = _find_solid_square(g, _DEST, bg)
    assert n == n2

    forbidden = {bg, _MOVER, _DEST}
    mover_support = _infer_support_color(g, mover_r, mover_c, n, forbidden)
    dest_support = _infer_support_color(g, dest_r, dest_c, n, forbidden)

    support = np.array(g, copy=True)
    support[mover_r : mover_r + n, mover_c : mover_c + n] = mover_support
    support[dest_r : dest_r + n, dest_c : dest_c + n] = dest_support

    allowed_mask = support == mover_support
    all_mask = support != bg
    feasible = _build_feasible(allowed_mask, n)
    all_feasible = _build_feasible(all_mask, n)
    assert bool(feasible[mover_r, mover_c])
    assert bool(all_feasible[mover_r, mover_c])

    dist = _bfs_placement(feasible, mover_r, mover_c)
    all_dist = _bfs_placement(all_feasible, mover_r, mover_c)

    dest_ok = bool(feasible[dest_r, dest_c]) and int(dist[dest_r, dest_c]) >= 0
    all_dest_ok = bool(all_feasible[dest_r, dest_c]) and int(all_dist[dest_r, dest_c]) >= 0

    if dest_ok:
        tr, tc = int(dest_r), int(dest_c)
    else:
        tr, tc = _pick_farthest(dist, mover_r, mover_c)

    return {
        "grid": g,
        "bg": bg,
        "n": n,
        "mover_r": mover_r,
        "mover_c": mover_c,
        "dest_r": dest_r,
        "dest_c": dest_c,
        "mover_support": mover_support,
        "dest_support": dest_support,
        "support": support,
        "feasible": feasible,
        "dist": dist,
        "all_feasible": all_feasible,
        "all_dist": all_dist,
        "dest_ok": dest_ok,
        "all_dest_ok": all_dest_ok,
        "target_r": tr,
        "target_c": tc,
        "river_colors": _river_colors(g, bg),
    }


def main(grid: np.ndarray) -> np.ndarray:
    info = _analyze(np.array(grid, dtype=int))
    out = np.array(info["grid"], copy=True)
    n = int(info["n"])
    sr, sc = int(info["mover_r"]), int(info["mover_c"])
    tr, tc = int(info["target_r"]), int(info["target_c"])
    out[sr : sr + n, sc : sc + n] = int(info["mover_support"])
    out[tr : tr + n, tc : tc + n] = _MOVER
    return out


def _compute_offsets(G: int, n: int, num_coarse: int) -> list[int] | None:
    if num_coarse == 1:
        return [1]
    first = 1
    last = G - n - 1
    if last < first:
        return None
    step = (last - first) / (num_coarse - 1)
    vals = [int(round(first + i * step)) for i in range(num_coarse)]
    for i in range(len(vals) - 1):
        if vals[i + 1] - vals[i] < n + 1:
            return None
    return vals


def _canonical_templates() -> dict[int, dict]:
    return {
        4: {
            "path": [(1, 0), (1, 1), (1, 2), (2, 2), (2, 3)],
            "cycle_edges": [((1, 1), (2, 1)), ((2, 1), (3, 1)), ((3, 1), (3, 2)), ((3, 2), (2, 2))],
            "spur_edges": [((1, 2), (0, 2))],
            "gate": ((2, 2), (2, 3)),
        },
        5: {
            "path": [(1, 0), (1, 1), (1, 2), (2, 2), (2, 3), (3, 3), (3, 4)],
            "cycle_edges": [((1, 1), (2, 1)), ((2, 1), (3, 1)), ((3, 1), (3, 2)), ((3, 2), (3, 3))],
            "spur_edges": [((1, 2), (0, 2))],
            "gate": ((3, 3), (3, 4)),
        },
    }


def _transform_point(pt: tuple[int, int], size: int, rot: int, flip: bool) -> tuple[int, int]:
    r, c = int(pt[0]), int(pt[1])
    if flip:
        c = int(size - 1 - c)
    for _ in range(rot % 4):
        r, c = int(c), int(size - 1 - r)
    return (r, c)


def _transform_points(points: list[tuple[int, int]], size: int, rot: int, flip: bool) -> list[tuple[int, int]]:
    return [_transform_point(pt, size, rot, flip) for pt in points]


def _transform_edge(edge: tuple[tuple[int, int], tuple[int, int]], size: int, rot: int, flip: bool) -> tuple[tuple[int, int], tuple[int, int]]:
    a = _transform_point(edge[0], size, rot, flip)
    b = _transform_point(edge[1], size, rot, flip)
    return (a, b)


def _render_colored_network(
    grid: np.ndarray,
    node_colors: dict[tuple[int, int], int],
    edge_colors: dict[tuple[tuple[int, int], tuple[int, int]], int],
    n: int,
    offsets_r: list[int],
    offsets_c: list[int],
) -> dict[tuple[int, int], tuple[int, int]]:
    pos = {(r, c): (int(offsets_r[r]), int(offsets_c[c])) for r, c in node_colors}
    for node, color in node_colors.items():
        pr, pc = pos[node]
        grid[pr : pr + n, pc : pc + n] = int(color)
    for edge, color in edge_colors.items():
        a, b = edge
        pr1, pc1 = pos[a]
        pr2, pc2 = pos[b]
        if pr1 == pr2:
            c_lo = min(pc1, pc2) + n
            c_hi = max(pc1, pc2)
            if c_hi > c_lo:
                grid[pr1 : pr1 + n, c_lo:c_hi] = int(color)
        else:
            r_lo = min(pr1, pr2) + n
            r_hi = max(pr1, pr2)
            if r_hi > r_lo:
                grid[r_lo:r_hi, pc1 : pc1 + n] = int(color)
    return pos


def _extend_to_edge(grid: np.ndarray, pr: int, pc: int, n: int, color: int) -> tuple[int, int] | None:
    H, W = grid.shape
    candidates: list[tuple[int, int, int, str]] = []
    if all(grid[r, c] in (_BG, color) for r in range(pr) for c in range(pc, pc + n)):
        candidates.append((0, pc, pr, "top"))
    if all(grid[r, c] in (_BG, color) for r in range(pr + n, H) for c in range(pc, pc + n)):
        candidates.append((H - n, pc, H - (pr + n), "bottom"))
    if all(grid[r, c] in (_BG, color) for r in range(pr, pr + n) for c in range(pc)):
        candidates.append((pr, 0, pc, "left"))
    if all(grid[r, c] in (_BG, color) for r in range(pr, pr + n) for c in range(pc + n, W)):
        candidates.append((pr, W - n, W - (pc + n), "right"))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[2])
    nr, nc, _dist, direction = candidates[0]
    if direction in ("top", "bottom"):
        r_lo, r_hi = min(nr, pr), max(nr, pr) + n
        grid[r_lo:r_hi, pc : pc + n] = int(color)
    else:
        c_lo, c_hi = min(nc, pc), max(nc, pc) + n
        grid[pr : pr + n, c_lo:c_hi] = int(color)
    return int(nr), int(nc)


def _edge_key(a: tuple[int, int], b: tuple[int, int]) -> tuple[tuple[int, int], tuple[int, int]]:
    return tuple(sorted((a, b)))


def _build_legacy_output(info: dict) -> np.ndarray:
    out = np.array(info["grid"], copy=True)
    n = int(info["n"])
    sr, sc = int(info["mover_r"]), int(info["mover_c"])
    dr, dc = int(info["dest_r"]), int(info["dest_c"])
    if bool(info["all_feasible"][dr, dc]) and int(info["all_dist"][dr, dc]) >= 0:
        tr, tc = int(dr), int(dc)
    else:
        tr, tc = _pick_farthest(info["all_dist"], sr, sc)
    out[sr : sr + n, sc : sc + n] = int(info["mover_support"])
    out[tr : tr + n, tc : tc + n] = _MOVER
    return out


def _gen_attempt() -> np.ndarray | None:
    reached = random.random() < 0.55
    n = random.choice([2, 3])
    template_size = random.choice([4, 5]) if n == 2 else 4
    min_G = (template_size - 1) * (n + 1) + n + 2
    if min_G > 20:
        return None
    G = random.randint(max(12, min_G), 20)

    offsets = _compute_offsets(G, n, template_size)
    if offsets is None:
        return None

    template = _canonical_templates()[template_size]
    rot = random.randrange(4)
    flip = bool(random.randrange(2))

    path = _transform_points(template["path"], template_size, rot, flip)
    cycle_edges = [_transform_edge(edge, template_size, rot, flip) for edge in template["cycle_edges"]]
    spur_edges = [_transform_edge(edge, template_size, rot, flip) for edge in template["spur_edges"]]
    gate = _transform_edge(template["gate"], template_size, rot, flip)

    mover_color, alt1, alt2 = random.sample(_RIVER_COLORS, 3)
    tail_color = mover_color if reached else alt1
    spur_color = alt1 if reached else alt2

    node_colors: dict[tuple[int, int], int] = {}
    edge_colors: dict[tuple[tuple[int, int], tuple[int, int]], int] = {}

    for node in path[:-1]:
        node_colors[node] = mover_color
    node_colors[path[-1]] = tail_color

    for a, b in zip(path, path[1:]):
        edge_colors[_edge_key(a, b)] = mover_color if (a, b) != gate and (b, a) != gate else tail_color

    for a, b in cycle_edges:
        edge_colors[_edge_key(a, b)] = mover_color
        node_colors.setdefault(a, mover_color)
        node_colors.setdefault(b, mover_color)

    for a, b in spur_edges:
        edge_colors[_edge_key(a, b)] = spur_color
        node_colors.setdefault(a, mover_color)
        node_colors[b] = spur_color

    grid = np.full((G, G), _BG, dtype=int)
    coarse_to_pixel = _render_colored_network(grid, node_colors, edge_colors, n, offsets, offsets)

    mover_r, mover_c = coarse_to_pixel[path[0]]
    dest_anchor_r, dest_anchor_c = coarse_to_pixel[path[-1]]
    dest_pos = _extend_to_edge(grid, dest_anchor_r, dest_anchor_c, n, tail_color)
    if dest_pos is None:
        return None
    dest_r, dest_c = dest_pos

    if mover_r == 0 or mover_r + n == G or mover_c == 0 or mover_c + n == G:
        return None
    if not (dest_r == 0 or dest_r + n == G or dest_c == 0 or dest_c + n == G):
        return None

    river_area = int(np.isin(grid, _RIVER_COLORS).sum())
    if not (int(0.18 * G * G) <= river_area <= int(0.68 * G * G)):
        return None

    grid[mover_r : mover_r + n, mover_c : mover_c + n] = _MOVER
    grid[dest_r : dest_r + n, dest_c : dest_c + n] = _DEST

    try:
        info = _analyze(grid)
    except Exception:
        return None

    if len(info["river_colors"]) < 2:
        return None
    if int(info["mover_support"]) != int(mover_color):
        return None
    if int(info["dest_support"]) != int(tail_color):
        return None
    if bool(info["dest_ok"]) != bool(reached):
        return None
    if not bool(info["all_dest_ok"]):
        return None

    branch_count, cycle_rank, reachable_nodes = _reachable_topology_metrics(info["feasible"], mover_r, mover_c)
    if branch_count < 1 or cycle_rank < 1 or reachable_nodes < 8:
        return None

    if not reached:
        if int((info["dist"] == int(info["dist"].max())).sum()) != 1:
            return None
        legacy_out = _build_legacy_output(info)
        combo_out = main(grid)
        if np.array_equal(combo_out, legacy_out):
            return None
    else:
        if int(info["target_r"]) != int(dest_r) or int(info["target_c"]) != int(dest_c):
            return None

    if not np.array_equal(main(grid), main(np.array(grid, copy=True))):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(500):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("Failed to generate valid input after 500 attempts")
