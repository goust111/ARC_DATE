# Variant of 332f06d7
# Dimension: D11 — 通道结构复杂度 (obstacle)
# Change: 单条蜿蜒走廊，>=3个转弯，无分叉，通过粗粒度Hamiltonian walk生成... → 环形通道

from __future__ import annotations

from collections import Counter, defaultdict, deque
import random

import numpy as np

from arc_utils import bbox_h, bbox_w, find_components


"""
Design analysis for the D11 loop-channel variant
===============================================

a) Concrete behavior and where it changes
-----------------------------------------
- Pixels still contain:
  - background 3 at all four corners,
  - a blue river/channel,
  - one black solid n×n square (the mover),
  - one red solid n×n square (the destination), same size as the mover.
- The visible change is that the channel is no longer a single corridor. It contains one
  rectangular loop, so there are two distinct river routes from the mover side to the
  destination side.
- The new rule is not "take any feasible n×n route". Instead, the mover first chooses the
  shortest reference route traced by its top-left corner through the visible blue pixels.
  Then it advances along that chosen route while its whole n×n body still fits. If the
  chosen route contains a narrow gate, it stops on that branch even when the longer branch
  of the loop would still allow arrival.

- Small execution trace:
  - Original logic: build all feasible n×n placements, BFS on that placement graph, so any
    alternate passable branch can be used.
  - Variant logic: BFS on raw visible river pixels from mover top-left to destination
    top-left, which picks the shorter branch of the loop; only afterward check whole-body
    fit along that chosen branch.
  - Difference: a short-but-narrow branch can now dominate the decision, producing a stop
    before the gate even though a longer wide branch exists.

b) Complexity calibration
-------------------------
- The loop is always a single clear rectangular ring with two stems: one stem to the mover,
  one stem to the destination. This keeps the visual rule legible in 2-4 examples.
- Only one special branch event is introduced: the shorter branch may contain one gate of
  width 1..n-1. That is enough to make the loop matter without turning the task into a
  general maze search problem.
- The generator rejects equal-length branch layouts so the "shortest branch" decision stays
  visible and deterministic.

c) Hidden assumptions from the original solver that must change
---------------------------------------------------------------
- Original code sorted candidate squares by color id. That is not allowed here because
  color ids are not a visible cue. The new solver instead identifies the destination as the
  only same-sized square touching the border; the mover is the same-sized square that does
  not touch the border and is closest to a border.
- Original code assumed the final location should be chosen by BFS over feasible n×n
  placements. That silently bakes in the old "reroute through any open branch" behavior.
  In the loop variant this would be wrong for stuck examples, because it would incorrectly
  use the longer branch.
- Original code's "farthest reachable feasible placement" fallback also belongs to the old
  global-BFS interpretation. Here the stop position must be the last feasible point on the
  chosen shortest reference branch.

d) Placement legality guaranteed by the generator
-------------------------------------------------
- The generator places a coarse rectangular ring on an (n+1)-spaced lattice so squares,
  stems, and loop edges never overlap illegally.
- The destination square always touches exactly one border and never a corner.
- The mover square is always one cell in from a border, so solver role assignment is
  visible without using color order.
- In stuck cases, only one connector on the short branch is narrowed; the long branch stays
  n-wide and remains feasible for the original placement-BFS logic.

e) Visual observability
-----------------------
- The loop is visible in the input itself: two blue routes connect the mover side to the
  destination side.
- Training pairs can show both outcomes:
  - reachable: mover reaches the border-touching target through the shorter branch;
  - loop-blocked: mover stops before the short-branch gate while the destination remains.
- This makes the loop structure matter in the observed input/output relation.

f) Visual cleanliness
---------------------
- There are only four colors and one connected river component.
- The gate is carved from the lower/right part of a connector, so the loop remains visually
  continuous while the n×n body fit is visibly compromised.
- Objects do not overlap and the channel remains easy to parse as a ring plus two stems.

g) Color inference
------------------
- No color remapping exists.
- Colors keep their original roles: black mover, red destination, blue channel, green
  background. The solver does not rely on black<red or any other invisible id ordering.

New solver assumptions introduced here
--------------------------------------
- Exactly one same-sized square touches the border: that one is the destination.
- Exactly one same-sized square of that size does not touch the border and serves as the
  mover.
- A raw visible river path exists between mover top-left and destination top-left.
- In generated stuck cases, the raw shortest branch is unique, and the longer branch stays
  feasible under the old full-body placement BFS, proving the variant is not collapsing
  back to the original task.
"""


_BG, _RIVER, _MOVER, _DEST = 3, 1, 0, 2
_DIR4 = [(0, 1), (1, 0), (0, -1), (-1, 0)]


def _touches_border(r0: int, c0: int, n: int, H: int, W: int) -> bool:
    return r0 == 0 or c0 == 0 or r0 + n == H or c0 + n == W


def _border_distance(r0: int, c0: int, n: int, H: int, W: int) -> int:
    return min(r0, c0, H - (r0 + n), W - (c0 + n))


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


def _choose_mover_and_dest(squares: list[dict], H: int, W: int) -> tuple[dict, dict]:
    by_n: dict[int, list[dict]] = defaultdict(list)
    for sq in squares:
        by_n[int(sq["n"])].append(sq)

    candidates: list[tuple[int, int, int, dict, dict]] = []
    for n, group in by_n.items():
        touching = [sq for sq in group if _touches_border(int(sq["r0"]), int(sq["c0"]), int(n), H, W)]
        interior = [sq for sq in group if not _touches_border(int(sq["r0"]), int(sq["c0"]), int(n), H, W)]
        if touching and interior:
            dest = min(
                touching,
                key=lambda sq: (_border_distance(int(sq["r0"]), int(sq["c0"]), int(n), H, W), int(sq["r0"]), int(sq["c0"])),
            )
            mover = min(
                interior,
                key=lambda sq: (_border_distance(int(sq["r0"]), int(sq["c0"]), int(n), H, W), int(sq["r0"]), int(sq["c0"])),
            )
            candidates.append((len(group), int(n), -_border_distance(int(mover["r0"]), int(mover["c0"]), int(n), H, W), mover, dest))

    assert candidates
    candidates.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)
    return candidates[0][3], candidates[0][4]


def _shortest_reference_path(mask: np.ndarray, start: tuple[int, int], goal: tuple[int, int]) -> list[tuple[int, int]] | None:
    H, W = mask.shape
    sr, sc = start
    gr, gc = goal
    if not (0 <= sr < H and 0 <= sc < W and 0 <= gr < H and 0 <= gc < W):
        return None
    if not (mask[sr, sc] and mask[gr, gc]):
        return None

    prev: dict[tuple[int, int], tuple[int, int] | None] = {(sr, sc): None}
    q = deque([(sr, sc)])
    while q:
        r, c = q.popleft()
        if (r, c) == (gr, gc):
            break
        for dr, dc in _DIR4:
            rr, cc = r + dr, c + dc
            if 0 <= rr < H and 0 <= cc < W and bool(mask[rr, cc]) and (rr, cc) not in prev:
                prev[(rr, cc)] = (r, c)
                q.append((rr, cc))

    if (gr, gc) not in prev:
        return None

    path: list[tuple[int, int]] = []
    cur: tuple[int, int] | None = (gr, gc)
    while cur is not None:
        path.append(cur)
        cur = prev[cur]
    path.reverse()
    return path


def _reference_guide_mask(trav: np.ndarray, start: tuple[int, int], goal: tuple[int, int]) -> np.ndarray:
    H, W = trav.shape
    guide = np.zeros_like(trav, dtype=bool)
    for r in range(H):
        for c in range(W):
            if not bool(trav[r, c]):
                continue
            above_open = (r == 0) or (not bool(trav[r - 1, c]))
            left_open = (c == 0) or (not bool(trav[r, c - 1]))
            diag_open = (r == 0) or (c == 0) or (not bool(trav[r - 1, c - 1]))
            if above_open or left_open or diag_open:
                guide[r, c] = True
    guide[start] = True
    guide[goal] = True
    return guide


def _placement_fits(trav: np.ndarray, r: int, c: int, n: int) -> bool:
    H, W = trav.shape
    if not (0 <= r <= H - n and 0 <= c <= W - n):
        return False
    return bool(trav[r : r + n, c : c + n].all())


def _last_feasible_on_path(trav: np.ndarray, path: list[tuple[int, int]], n: int) -> tuple[int, int]:
    assert path
    last = path[0]
    for r, c in path:
        if _placement_fits(trav, int(r), int(c), int(n)):
            last = (int(r), int(c))
        else:
            break
    return last


def _solve_state(grid: np.ndarray) -> tuple[np.ndarray, dict]:
    g = np.array(grid, dtype=int)
    H, W = g.shape

    bg = int(Counter([int(g[0, 0]), int(g[0, -1]), int(g[-1, 0]), int(g[-1, -1])]).most_common(1)[0][0])
    squares = _find_solid_squares(g, bg)
    assert len(squares) >= 2

    mover, dest = _choose_mover_and_dest(squares, H, W)
    n = int(mover["n"])
    sr, sc = int(mover["r0"]), int(mover["c0"])
    dr, dc = int(dest["r0"]), int(dest["c0"])

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

    trav = (g == int(river)) | (g == int(mover["col"])) | (g == int(dest["col"]))
    guide = _reference_guide_mask(trav, (sr, sc), (dr, dc))
    path = _shortest_reference_path(guide, (sr, sc), (dr, dc))
    assert path is not None
    tr, tc = _last_feasible_on_path(trav, path, n)

    out = g.copy()
    out[sr : sr + n, sc : sc + n] = int(river)
    out[tr : tr + n, tc : tc + n] = int(mover["col"])

    info = {
        "bg": bg,
        "river": river,
        "n": n,
        "start": (sr, sc),
        "dest": (dr, dc),
        "final": (tr, tc),
        "trav": trav,
        "guide": guide,
        "path_len": len(path),
        "reached": (tr, tc) == (dr, dc),
    }
    return out, info


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide a solid square through a looped channel toward a same-sized border target.

    Input structure:
    - The corners show the background color, while blue pixels form one connected river with
      a visible loop and two stems.
    - Two same-sized solid squares sit on that river: a black mover inside the border and a
      red destination square touching the border.

    Rule:
    - Track the mover by following the shortest visible route from the mover's top-left
      corner to the destination's top-left corner through river-colored pixels.
    - Advance the full n×n mover along that chosen route only while every covered pixel stays
      inside the traversable set {river, mover, destination}.
    - If the whole chosen route is wide enough, the mover replaces the destination.
      Otherwise it stops at the last feasible point on that shorter branch, and its original
      location turns back into river.

    Key insight: In this variant the loop matters because branch choice happens before
    whole-body fit is checked, so a short branch with a narrow gate can beat a longer
    passable branch and still trap the mover.
    """
    out, _info = _solve_state(np.array(grid, dtype=int))
    return out


def _bfs_placement(feasible: np.ndarray, sr: int, sc: int) -> np.ndarray:
    h2, w2 = feasible.shape
    dist = -np.ones((h2, w2), dtype=int)
    dist[sr, sc] = 0
    q = deque([(sr, sc)])
    while q:
        r, c = q.popleft()
        for dr, dc in _DIR4:
            rr, cc = r + dr, c + dc
            if 0 <= rr < h2 and 0 <= cc < w2 and bool(feasible[rr, cc]) and int(dist[rr, cc]) == -1:
                dist[rr, cc] = int(dist[r, c] + 1)
                q.append((rr, cc))
    return dist


def _placement_dist(trav: np.ndarray, n: int, start: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    H, W = trav.shape
    h2, w2 = H - n + 1, W - n + 1
    feasible = np.zeros((h2, w2), dtype=bool)
    for r in range(h2):
        for c in range(w2):
            feasible[r, c] = bool(trav[r : r + n, c : c + n].all())
    sr, sc = start
    assert feasible[sr, sc]
    return feasible, _bfs_placement(feasible, sr, sc)


def _paint_block(grid: np.ndarray, pos: tuple[int, int], n: int, color: int = _RIVER) -> None:
    r, c = pos
    grid[r : r + n, c : c + n] = color


def _paint_segment(grid: np.ndarray, p1: tuple[int, int], p2: tuple[int, int], n: int, color: int = _RIVER) -> None:
    r1, c1 = p1
    r2, c2 = p2
    if r1 == r2:
        c_lo, c_hi = min(c1, c2), max(c1, c2) + n
        grid[r1 : r1 + n, c_lo:c_hi] = color
    else:
        r_lo, r_hi = min(r1, r2), max(r1, r2) + n
        grid[r_lo:r_hi, c1 : c1 + n] = color


def _carve_horizontal_gate_preserve_top(grid: np.ndarray, left: tuple[int, int], right: tuple[int, int], n: int) -> bool:
    r, c1 = left
    _r2, c2 = right
    c_lo = min(c1, c2) + n
    c_hi = max(c1, c2)
    if c_hi <= c_lo:
        return False
    narrow_w = random.randint(1, n - 1)
    grid[r + narrow_w : r + n, c_lo:c_hi] = _BG
    return True


def _make_case() -> np.ndarray | None:
    reached = random.random() < 0.5
    n = random.choice([2, 3, 4])
    stride = n + 1

    valid_shapes: list[tuple[int, int, int, int, int]] = []
    for rh in [3, 4]:
        for rw in [3, 4]:
            for ls in [1, 2]:
                for ts in [1, 2]:
                    base_r = ts * stride
                    base_c = 1 + ls * stride
                    min_h = base_r + (rh - 1) * stride + n + 1
                    min_w = base_c + (rw - 1) * stride + n + 1
                    if max(min_h, min_w) <= 20:
                        valid_shapes.append((rh, rw, ls, ts, max(min_h, min_w)))
    if not valid_shapes:
        return None

    rh, rw, ls, ts, gmin = random.choice(valid_shapes)
    G = random.randint(max(10, gmin), 20)
    grid = np.full((G, G), _BG, dtype=int)

    base_r = ts * stride
    base_c = 1 + ls * stride

    def pix(cr: int, cc: int) -> tuple[int, int]:
        return base_r + cr * stride, base_c + cc * stride

    ring_nodes: list[tuple[int, int]] = []
    for cc in range(rw):
        ring_nodes.append((0, cc))
    for rr in range(1, rh):
        ring_nodes.append((rr, rw - 1))
    for cc in range(rw - 2, -1, -1):
        ring_nodes.append((rh - 1, cc))
    for rr in range(rh - 2, 0, -1):
        ring_nodes.append((rr, 0))

    for node in ring_nodes:
        _paint_block(grid, pix(*node), n)
    for i in range(len(ring_nodes)):
        _paint_segment(grid, pix(*ring_nodes[i]), pix(*ring_nodes[(i + 1) % len(ring_nodes)]), n)

    a_node = (1, 0)
    b_node = (0, rw - 2)
    mover = (pix(*a_node)[0], 1)
    dest = (0, pix(*b_node)[1])
    _paint_block(grid, mover, n)
    _paint_block(grid, dest, n)
    _paint_segment(grid, mover, pix(*a_node), n)
    _paint_segment(grid, dest, pix(*b_node), n)

    gate_added = False
    if not reached and n >= 2:
        top_edge_choices = list(range(0, b_node[1]))
        random.shuffle(top_edge_choices)
        for edge_idx in top_edge_choices:
            if _carve_horizontal_gate_preserve_top(grid, pix(0, edge_idx), pix(0, edge_idx + 1), n):
                gate_added = True
                break
        if not gate_added:
            return None

    grid[mover[0] : mover[0] + n, mover[1] : mover[1] + n] = _MOVER
    grid[dest[0] : dest[0] + n, dest[1] : dest[1] + n] = _DEST

    if Counter([int(grid[0, 0]), int(grid[0, -1]), int(grid[-1, 0]), int(grid[-1, -1])]).most_common(1)[0][0] != _BG:
        return None
    if not _touches_border(dest[0], dest[1], n, G, G):
        return None
    if _touches_border(mover[0], mover[1], n, G, G):
        return None

    river_area = int((grid == _RIVER).sum() + (grid == _MOVER).sum() + (grid == _DEST).sum())
    if not (0.18 <= river_area / float(G * G) <= 0.62):
        return None

    out, info = _solve_state(grid)
    trav = info["trav"]
    feasible, dist = _placement_dist(trav, n, info["start"])
    dr, dc = info["dest"]
    original_reaches = bool(feasible[dr, dc]) and int(dist[dr, dc]) >= 0

    if reached:
        if not info["reached"]:
            return None
    else:
        if info["reached"]:
            return None
        if not original_reaches:
            return None

    expected = grid.copy()
    sr, sc = info["start"]
    tr, tc = info["final"]
    expected[sr : sr + n, sc : sc + n] = _RIVER
    expected[tr : tr + n, tc : tc + n] = _MOVER
    if not np.array_equal(out, expected):
        return None

    return grid


def generate_input():
    for _ in range(400):
        result = _make_case()
        if result is not None:
            return result
    raise RuntimeError("Failed to generate valid input after 400 attempts")
