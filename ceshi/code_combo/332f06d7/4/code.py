# Combo variant of 332f06d7 (index 4)
# combo_spec: D2/2+D4/0
# Name: 多级门控+矩形移动体

from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np

from arc2_solver_select.arc_utils import bbox_h, bbox_w, find_components


"""
Design analysis for combo variant D2/2 + D4/0
==============================================

This fusion keeps the original high-level story "a token slides through one blue
channel toward a red target", but changes two coupled assumptions at once:

1. The token is a fixed-orientation h x w rectangle with h != w.
2. The river contains several gates of different visible widths, and the relevant
   width depends on segment direction:
   - horizontal travel cares about corridor height >= h
   - vertical travel cares about corridor width >= w

a) Concretization and behavior trace
------------------------------------
Concrete picture used by the generator:
- One black solid rectangle and one red solid rectangle have the same shape.
- A single blue zig-zag corridor follows five segments:
  horizontal -> vertical -> horizontal -> vertical -> horizontal.
- Every segment is initially painted wider than necessary, then a local gate is
  carved into that segment to produce one of several visible widths.
- Earlier gates are still passable for the rectangle, while the last gate is
  either exactly tight (reachable case) or one pixel too small (stuck case).

Small trace with h=2, w=4:
- Original task would use one scalar n and ask whether every corridor cross
  section is at least 2x2 or 4x4. That assumption no longer makes sense.
- In the fused task:
  - the first horizontal gate may have visible height 3, so the 2-high token
    passes it
  - the first vertical gate may have visible width 5, so the 4-wide token
    passes it
  - a later horizontal gate may narrow to height 2, so it still passes
  - the final horizontal gate may narrow to height 1, so the 2-high token stops
    at the last feasible placement before that gate
- Difference from the original:
  - pass/fail is not controlled by a single scalar size
  - multiple gates matter in sequence, not just one choke point
  - horizontal and vertical narrowings test different rectangle dimensions

b) Complexity calibration
-------------------------
- Only one mover-target pair is used. This keeps the scene readable and avoids
  ambiguous matching.
- Rectangle shapes come from a small family:
  (2x3), (3x2), (2x4), (4x2), (3x4), (4x3).
- The path always has five segments and five visible gates, so there is repeated
  evidence for "multi-level gating" without combinatorial clutter.
- Gate widths are drawn from only a few levels around the required dimensions:
  h+1, h, h-1 on horizontal segments and w+1, w on vertical segments.

c) Hidden-assumption audit of the original main()
------------------------------------------------
Original assumptions that break here:
- "bbox_h == bbox_w" for the moving token and the destination.
  The fused task needs solid rectangles, not just squares.
- One scalar side length n is enough.
  Here feasibility must use windows of shape h x w.
- A single bottleneck width determines the outcome.
  Here several gates exist, and different segments constrain different axes.
- Reaching the destination can be reasoned about with one generic width check.
  Here the same fixed rectangle interacts anisotropically with the river shape.

What still survives:
- Traversability is still "river + mover + destination".
- BFS over feasible top-left placements still solves the task correctly.
  The direction-specific rule is encoded geometrically in the traversable mask:
  a horizontal pinch removes height from feasible h x w windows, while a vertical
  pinch removes width.

d) Placement legality
---------------------
- The generator uses one monotone zig-zag placement backbone with ample spacing.
- Start is near a border but not touching; destination touches one border.
- Segments are long enough that local gates can be carved away from turns.
- All accepted samples are verified by running the solver and by checking staged
  reachability: anchors before the last gate must remain reachable.

e) Visual observability
-----------------------
- The mover is visibly rectangular in every sample because h != w always holds.
- The corridor itself shows different horizontal-vs-vertical thickness.
- Multiple gates of visibly different widths appear on one path, so examples
  show that the token can pass some tight places but not others.
- No human-invisible ordering is used. Colors only identify mover/target/river.

f) Visual cleanliness
---------------------
- Only the original four colors are used.
- There is one connected river and one mover-target pair.
- Gates are cut from the same corridor instead of adding extra markers.
- Random flips/rotations add variety without changing the core structure.

g) New solver assumptions and generator guarantees
--------------------------------------------------
Solver assumptions:
- exactly one black solid rectangle and one red solid rectangle share a shape
- both are axis-aligned solid blocks
- the most common non-background non-token color is the river

Generator guarantees:
- exactly one such black/red pair exists
- their shared shape is non-square
- the river is the only other non-background structure
- early gates are passable for the intended rectangle
- the last gate alone controls reachable vs stuck in the canonical template
"""


_BG, _RIVER, _MOVER, _DEST = 3, 1, 0, 2
_RECT_SHAPES = [(2, 3), (3, 2), (2, 4), (4, 2), (3, 4), (4, 3)]
_DIR4 = [(1, 0), (-1, 0), (0, 1), (0, -1)]


def _background_color(grid: np.ndarray) -> int:
    corners = [int(grid[0, 0]), int(grid[0, -1]), int(grid[-1, 0]), int(grid[-1, -1])]
    return int(Counter(corners).most_common(1)[0][0])


def _solid_rectangles(grid: np.ndarray, bg: int) -> list[dict]:
    rects: list[dict] = []
    for comp in find_components(grid, bg=bg, connectivity=4, monochromatic=True, return_format="dict"):
        color = int(comp["color"])
        r0, c0, r1, c1 = map(int, comp["bbox"])
        h = int(bbox_h(comp["bbox"]))
        w = int(bbox_w(comp["bbox"]))
        patch = grid[r0 : r1 + 1, c0 : c1 + 1]
        if bool((patch == color).all()):
            rects.append({"col": color, "r0": r0, "c0": c0, "h": h, "w": w})
    return rects


def _choose_token_pair(rects: list[dict]) -> tuple[dict, dict]:
    movers = [rect for rect in rects if int(rect["col"]) == _MOVER]
    dests = [rect for rect in rects if int(rect["col"]) == _DEST]
    pairs: list[tuple[int, dict, dict]] = []
    for mover in movers:
        for dest in dests:
            if int(mover["h"]) == int(dest["h"]) and int(mover["w"]) == int(dest["w"]):
                area = int(mover["h"]) * int(mover["w"])
                pairs.append((area, mover, dest))
    assert pairs
    pairs.sort(key=lambda item: (item[0], item[1]["h"], item[1]["w"]))
    _area, mover, dest = pairs[-1]
    assert int(mover["h"]) != int(mover["w"])
    return mover, dest


def _river_color(grid: np.ndarray, bg: int, token_colors: set[int]) -> int:
    counts = Counter(map(int, grid.flatten()))
    for color, _ct in counts.most_common():
        if int(color) == int(bg):
            continue
        if int(color) in token_colors:
            continue
        return int(color)
    raise AssertionError("river color not found")


def _feasible_placements(trav: np.ndarray, h: int, w: int) -> np.ndarray:
    H, W = trav.shape
    h2 = H - h + 1
    w2 = W - w + 1
    assert h2 > 0 and w2 > 0
    feasible = np.zeros((h2, w2), dtype=bool)
    for r in range(h2):
        for c in range(w2):
            feasible[r, c] = bool(trav[r : r + h, c : c + w].all())
    return feasible


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


def _farthest_reachable(dist: np.ndarray, sr: int, sc: int) -> tuple[int, int]:
    best = -1
    tr, tc = sr, sc
    for r in range(dist.shape[0]):
        for c in range(dist.shape[1]):
            if int(dist[r, c]) > best:
                best = int(dist[r, c])
                tr, tc = r, c
    return int(tr), int(tc)


def _solve_on_traversable(
    trav: np.ndarray,
    h: int,
    w: int,
    sr: int,
    sc: int,
    dr: int,
    dc: int,
) -> tuple[tuple[int, int], np.ndarray, np.ndarray, bool]:
    feasible = _feasible_placements(trav, h, w)
    assert feasible[sr, sc]
    dist = _bfs_placement(feasible, sr, sc)
    dest_ok = (
        0 <= dr < feasible.shape[0]
        and 0 <= dc < feasible.shape[1]
        and bool(feasible[dr, dc])
        and int(dist[dr, dc]) >= 0
    )
    if dest_ok:
        target = (int(dr), int(dc))
    else:
        target = _farthest_reachable(dist, sr, sc)
    return target, feasible, dist, bool(dest_ok)


def _solve_detected_case(grid: np.ndarray) -> tuple[dict, dict, int, tuple[int, int], np.ndarray]:
    g = np.array(grid, dtype=int)
    bg = _background_color(g)
    rects = _solid_rectangles(g, bg)
    mover, dest = _choose_token_pair(rects)
    river = _river_color(g, bg, {int(mover["col"]), int(dest["col"])})
    h = int(mover["h"])
    w = int(mover["w"])
    sr, sc = int(mover["r0"]), int(mover["c0"])
    dr, dc = int(dest["r0"]), int(dest["c0"])
    trav = (g == river) | (g == int(mover["col"])) | (g == int(dest["col"]))
    (tr, tc), _feasible, dist, _dest_ok = _solve_on_traversable(trav, h, w, sr, sc, dr, dc)
    return mover, dest, river, (int(tr), int(tc)), dist


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide a solid rectangle through one blue channel with several directional gates.

    The token never rotates.
    - On horizontal travel, the relevant clearance is the river's height versus
      the rectangle's height.
    - On vertical travel, the relevant clearance is the river's width versus the
      rectangle's width.

    The solver does not hand-code those two cases separately. Instead it performs
    BFS over feasible h x w placements; the anisotropic gates are encoded by the
    geometry of the traversable mask itself.
    """
    g = np.array(grid, dtype=int)
    mover, _dest, river, (tr, tc), _dist = _solve_detected_case(g)
    h = int(mover["h"])
    w = int(mover["w"])
    sr, sc = int(mover["r0"]), int(mover["c0"])

    out = g.copy()
    out[sr : sr + h, sc : sc + w] = int(river)
    out[tr : tr + h, tc : tc + w] = int(mover["col"])
    return out


def _paint_h(grid: np.ndarray, r: int, c0: int, c1: int, band_h: int) -> None:
    left = min(c0, c1)
    right = max(c0, c1)
    grid[r : r + band_h, left:right] = _RIVER


def _paint_v(grid: np.ndarray, r0: int, r1: int, c: int, band_w: int) -> None:
    top = min(r0, r1)
    bottom = max(r0, r1)
    grid[top:bottom, c : c + band_w] = _RIVER


def _gate_h(grid: np.ndarray, r: int, c0: int, c1: int, band_h: int, gate_h: int) -> bool:
    if gate_h >= band_h:
        return True
    left = min(c0, c1)
    right = max(c0, c1)
    lo = left + 1
    hi = right - 1
    if hi <= lo:
        return False
    span = hi - lo
    gate_len = max(1, min(3, span))
    start = lo + (span - gate_len) // 2
    grid[r + gate_h : r + band_h, start : start + gate_len] = _BG
    return True


def _gate_v(grid: np.ndarray, r0: int, r1: int, c: int, band_w: int, gate_w: int) -> bool:
    if gate_w >= band_w:
        return True
    top = min(r0, r1)
    bottom = max(r0, r1)
    lo = top + 1
    hi = bottom - 1
    if hi <= lo:
        return False
    span = hi - lo
    gate_len = max(1, min(3, span))
    start = lo + (span - gate_len) // 2
    grid[start : start + gate_len, c + gate_w : c + band_w] = _BG
    return True


def _apply_transform(arr: np.ndarray, code: int) -> np.ndarray:
    if code == 0:
        return arr.copy()
    if code == 1:
        return np.rot90(arr, 1)
    if code == 2:
        return np.rot90(arr, 2)
    if code == 3:
        return np.rot90(arr, 3)
    if code == 4:
        return np.fliplr(arr)
    if code == 5:
        return np.flipud(arr)
    if code == 6:
        return np.rot90(np.fliplr(arr), 1)
    if code == 7:
        return np.rot90(np.flipud(arr), 1)
    raise AssertionError("bad transform code")


def _transform_shape(h: int, w: int, code: int) -> tuple[int, int]:
    if code in (1, 3, 6, 7):
        return int(w), int(h)
    return int(h), int(w)


def _gate_specs(h: int, w: int, reached: bool) -> list[dict]:
    return [
        {"dir": "h", "band": h + 2, "gate": h + 1},
        {"dir": "v", "band": w + 2, "gate": w + 1},
        {"dir": "h", "band": h + 1, "gate": h},
        {"dir": "v", "band": w + 1, "gate": w},
        {"dir": "h", "band": h + 2, "gate": h if reached else h - 1},
    ]


def _canonical_anchors(G: int, h: int, w: int) -> list[tuple[int, int]] | None:
    c_start = 1
    c1 = random.randint(4, 5)
    min_c2 = c1 + 4
    max_c2 = G - w - 4
    if max_c2 < min_c2:
        return None
    c2 = random.randint(min_c2, max_c2)

    r_high = random.randint(1, 2)
    r_mid_lo = r_high + 4
    r_mid_hi = G - h - 6
    if r_mid_hi < r_mid_lo:
        return None
    r_mid = random.randint(r_mid_lo, r_mid_hi)
    r_low = G - h - 2

    dest_c = G - w
    anchors = [
        (r_low, c_start),
        (r_low, c1),
        (r_high, c1),
        (r_high, c2),
        (r_mid, c2),
        (r_mid, dest_c),
    ]
    return anchors


def _build_grid_with_gates(G: int, h: int, w: int, reached: bool) -> tuple[np.ndarray, list[tuple[int, int]], list[dict]] | None:
    anchors = _canonical_anchors(G, h, w)
    if anchors is None:
        return None

    grid = np.full((G, G), _BG, dtype=int)
    specs = _gate_specs(h, w, reached)

    for idx, spec in enumerate(specs):
        (r0, c0) = anchors[idx]
        (r1, c1) = anchors[idx + 1]
        if spec["dir"] == "h":
            _paint_h(grid, r0, c0, c1 + w, int(spec["band"]))
            ok = _gate_h(grid, r0, c0, c1 + w, int(spec["band"]), int(spec["gate"]))
        else:
            _paint_v(grid, r0, r1 + h, c0, int(spec["band"]))
            ok = _gate_v(grid, r0, r1 + h, c0, int(spec["band"]), int(spec["gate"]))
        if not ok:
            return None

    return grid, anchors, specs


def _stage_reachability(
    grid: np.ndarray,
    h: int,
    w: int,
    start: tuple[int, int],
    checkpoints: list[tuple[int, int]],
) -> tuple[np.ndarray, np.ndarray]:
    trav = (grid == _RIVER) | (grid == _MOVER) | (grid == _DEST)
    feasible = _feasible_placements(trav, h, w)
    sr, sc = start
    assert feasible[sr, sc]
    dist = _bfs_placement(feasible, sr, sc)
    hits = np.array([int(dist[r, c]) >= 0 for r, c in checkpoints], dtype=bool)
    return hits, dist


def _swapped_shape_reachability(grid: np.ndarray) -> dict:
    g = np.array(grid, dtype=int)
    bg = _background_color(g)
    rects = _solid_rectangles(g, bg)
    mover, dest = _choose_token_pair(rects)
    river = _river_color(g, bg, {int(mover["col"]), int(dest["col"])})
    h = int(mover["h"])
    w = int(mover["w"])
    sr, sc = int(mover["r0"]), int(mover["c0"])
    dr, dc = int(dest["r0"]), int(dest["c0"])
    trav = (g == river) | (g == int(mover["col"])) | (g == int(dest["col"]))

    actual_target, _actual_feasible, actual_dist, actual_dest_ok = _solve_on_traversable(trav, h, w, sr, sc, dr, dc)
    swapped = {"actual_dest_ok": bool(actual_dest_ok), "actual_steps": int(actual_dist[actual_target])}
    try:
        swapped_target, _swap_feasible, swap_dist, swap_dest_ok = _solve_on_traversable(trav, w, h, sr, sc, dr, dc)
        swapped["swapped_dest_ok"] = bool(swap_dest_ok)
        swapped["swapped_steps"] = int(swap_dist[swapped_target])
    except Exception:
        swapped["swapped_dest_ok"] = False
        swapped["swapped_steps"] = -1
    return swapped


def _gen_attempt(return_meta: bool = False) -> np.ndarray | tuple[np.ndarray, dict] | None:
    reached = random.random() < 0.5
    h, w = random.choice(_RECT_SHAPES)

    min_g = max(17, h + 12, w + 13)
    if min_g > 20:
        return None
    G = random.randint(min_g, 20)

    built = _build_grid_with_gates(G, h, w, reached)
    if built is None:
        return None
    grid, anchors, specs = built

    mover_r, mover_c = anchors[0]
    dest_r, dest_c = anchors[-1]
    grid[mover_r : mover_r + h, mover_c : mover_c + w] = _MOVER
    grid[dest_r : dest_r + h, dest_c : dest_c + w] = _DEST

    checkpoints = anchors[1:]
    reach_hits, dist = _stage_reachability(grid, h, w, anchors[0], checkpoints)
    if not bool(np.all(reach_hits[:4])):
        return None
    if bool(reach_hits[4]) != bool(reached):
        return None

    expected = main(grid)
    mover, dest, _river, (tr, tc), solver_dist = _solve_detected_case(grid)
    if reached:
        if (tr, tc) != (dest_r, dest_c):
            return None
    else:
        if (tr, tc) == (dest_r, dest_c):
            return None
        if int(solver_dist[anchors[4]]) < 0:
            return None

    if not np.array_equal(main(grid), expected):
        return None

    code = random.randint(0, 7)
    final_grid = _apply_transform(grid, code)
    final_shape = _transform_shape(h, w, code)

    if not return_meta:
        return final_grid

    meta = {
        "reached": bool(reached),
        "grid_size": int(G),
        "base_shape": (int(h), int(w)),
        "final_shape": tuple(map(int, final_shape)),
        "gate_signature": [(spec["dir"], int(spec["gate"])) for spec in specs],
        "distinct_gate_widths": len({int(spec["gate"]) for spec in specs}),
        "passed_gates": int(np.sum(reach_hits)),
        "transform_code": int(code),
    }
    meta.update(_swapped_shape_reachability(final_grid))
    return final_grid, meta


def generate_input() -> np.ndarray:
    for _ in range(400):
        result = _gen_attempt(return_meta=False)
        if result is not None:
            return result
    raise RuntimeError("Failed to generate valid input after 400 attempts")
