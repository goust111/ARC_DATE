# Combo variant of 332f06d7 (index 42)
# combo_spec: D2/2+D4/0+D9/2
# Name: 多级门控+矩形移动体+非正方形网格

from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np

from arc_utils import bbox_h, bbox_w, find_components


"""
Design analysis for combo D2/2 + D4/0 + D9/2
=============================================

source_task:
  before:
    - one black n x n square and one red n x n square
    - one blue winding corridor on a square grid
    - one scalar width threshold n decides every passage
  after:
    - one black h x w rectangle and one red h x w rectangle, with h != w
    - a non-square grid
    - a five-stage snake corridor whose horizontal stages and vertical stages have
      different visible widths
    - several gate levels appear in the same sample; a stuck sample passes some
      gates and then fails at a later one

a_concretization_and_trace:
  concrete_visual_plan:
    - The mover and destination are solid rectangles from a small non-square family.
    - The river is a single H-V-H-V-H snake. Horizontal strips have independently
      chosen heights; vertical strips have independently chosen widths.
    - Example with mover 2 x 4 on a 15 x 19 grid:
      stage_0 horizontal height = 4
      stage_1 vertical width = 5
      stage_2 horizontal height = 3
      stage_3 vertical width = 3
      stage_4 horizontal height = 2
      Here the rectangle can pass stages 0, 1, 2 because 4 >= 2, 5 >= 4, 3 >= 2,
      but it fails at stage 3 because vertical travel needs width >= 4 and this gate
      is only 3 wide. It therefore stops at the farthest feasible placement in the
      upper part of the snake instead of reaching the red target on the right edge.
  old_logic_vs_new_logic:
    - original main() assumed one square side n and one global feasible-window size
    - new main() must detect a solid rectangle, keep its orientation fixed, and test
      h x w windows instead of n x n windows
    - original generator only needed one chokepoint; new generator must ensure several
      visible gate levels exist in one corridor and that only a later gate blocks the
      mover in stuck cases
  why_this_is_a_true_fusion:
    - D4 changes the object from square to rectangle, which makes horizontal and
      vertical clearance requirements different
    - D2 adds multiple gate levels instead of one bottleneck
    - D9 makes the whole scene non-square so the long horizontal and vertical runs
      are visually asymmetric and not reducible to the original square-grid task

b_complexity_calibration:
  - Only one mover/target pair is used, so there is no ambiguous matching problem.
  - Rectangle shapes come from a small repeated family:
    (2,3), (3,2), (2,4), (4,2), (3,4), (4,3)
  - The corridor always has exactly five stages and four turns. That is enough to
    show multi-level gating without producing visually chaotic mazes.
  - Stage widths are sampled from only three passable levels per axis:
    horizontal heights in {h, h+1, h+2}
    vertical widths in {w, w+1, w+2}
    plus one optional failing level in {1..h-1} or {1..w-1}.
    This keeps the visual rule discrete and observable.

c_implicit_assumption_audit_of_original_main:
  - "bbox_h == bbox_w" is no longer valid; solid non-square rectangles must be kept.
  - "choose the most common side length n" is no longer valid; the relevant key is the
    exact ordered shape (h, w), and mover/dest are identified by their visible colors.
  - "grid is square" is no longer valid; all loops must use the actual H and W.
  - "one scalar width n governs all segments" is no longer valid; feasibility still
    comes from whole-window BFS, but the generator must encode anisotropic gate widths
    so that horizontal and vertical failures happen for different reasons.
  - "one chokepoint is enough" is no longer valid; the solver still runs one BFS, but
    the generator must guarantee several visible gate levels so the fused behavior is
    inferable from examples.

d_placement_legality:
  - The snake path is specified by six top-left anchor positions with gaps larger than
    the mover size, so the turn boxes at adjacent anchors do not accidentally merge
    across an entire blocked stage.
  - Every painted corridor rectangle is clipped by construction to stay in-bounds.
  - The mover start and target footprints are explicitly painted as full h x w blocks
    before recoloring them black/red, so both tokens are always solid legal objects.
  - The generator oracle-checks every sample by recomputing reachability and verifying
    main(input) equals the intended output.

e_visual_observability:
  - The mover is visibly rectangular in every sample because h != w always holds.
  - The river itself exposes the rule: horizontal and vertical stages have visibly
    different thicknesses, and those thicknesses vary by stage.
  - In stuck samples, the mover is not blocked immediately. It passes earlier wider
    gates and only stops deeper in the snake, which makes the "multi-level" aspect
    directly visible from input/output pairs.
  - No hidden sorting by raw color ids is used beyond the semantic role colors that
    are directly visible to a human: black = mover, red = destination.

f_visual_cleanliness:
  - Only the original four colors are kept.
  - The scene is one connected river with one mover and one target; there are no extra
    markers, labels, or decorative objects.
  - The five-stage snake is structured and sparse enough that each gate can be seen
    without crowding.

g_new_main_assumptions_and_generator_guarantees:
  - Assumption: exactly one black solid rectangle and one red solid rectangle share the
    same non-square shape.
    Guarantee: generator places exactly one such pair.
  - Assumption: the river is the most common non-background color besides black/red.
    Guarantee: generator uses only one river color and paints most non-background area
    with it.
  - Assumption: the black start position is itself a feasible h x w placement.
    Guarantee: the first horizontal stage is always passable.
  - Assumption: non-square behavior matters.
    Guarantee: generator only uses h != w and H != W.
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
            rects.append({"color": color, "r0": r0, "c0": c0, "h": h, "w": w})
    return rects


def _choose_token_pair(rects: list[dict]) -> tuple[dict, dict]:
    movers = [rect for rect in rects if int(rect["color"]) == _MOVER and int(rect["h"]) != int(rect["w"])]
    dests = [rect for rect in rects if int(rect["color"]) == _DEST and int(rect["h"]) != int(rect["w"])]
    assert movers and dests

    shared_shapes = sorted(
        set((int(rect["h"]), int(rect["w"])) for rect in movers)
        & set((int(rect["h"]), int(rect["w"])) for rect in dests)
    )
    assert shared_shapes

    shape = max(shared_shapes, key=lambda hw: (hw[0] * hw[1], hw[0] + hw[1], hw[0]))
    mover = next(rect for rect in movers if (int(rect["h"]), int(rect["w"])) == shape)
    dest = next(rect for rect in dests if (int(rect["h"]), int(rect["w"])) == shape)
    return mover, dest


def _river_color(grid: np.ndarray, bg: int) -> int:
    counts = Counter(map(int, grid.flatten()))
    for color, _count in counts.most_common():
        if int(color) not in (int(bg), _MOVER, _DEST):
            return int(color)
    raise AssertionError("river color not found")


def _feasible_placements(trav: np.ndarray, h: int, w: int) -> np.ndarray:
    H, W = trav.shape
    h2, w2 = H - h + 1, W - w + 1
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
    return tr, tc


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide a solid rectangle through a multi-gate winding corridor on a non-square grid.

    The rectangle never rotates. Horizontal stages need height >= h; vertical stages
    need width >= w. If all gates are passable, the mover covers the red rectangle.
    Otherwise it stops at the farthest reachable placement before the first failing gate.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape

    bg = _background_color(g)
    mover, dest = _choose_token_pair(_solid_rectangles(g, bg))
    river = _river_color(g, bg)

    h = int(mover["h"])
    w = int(mover["w"])
    sr, sc = int(mover["r0"]), int(mover["c0"])
    dr, dc = int(dest["r0"]), int(dest["c0"])

    trav = (g == river) | (g == _MOVER) | (g == _DEST)
    h2, w2 = H - h + 1, W - w + 1
    assert h2 > 0 and w2 > 0
    assert 0 <= sr < h2 and 0 <= sc < w2

    feasible = _feasible_placements(trav, h, w)
    assert feasible[sr, sc]
    dist = _bfs_placement(feasible, sr, sc)

    if 0 <= dr < h2 and 0 <= dc < w2 and feasible[dr, dc] and dist[dr, dc] >= 0:
        tr, tc = dr, dc
    else:
        tr, tc = _farthest_reachable(dist, sr, sc)

    out = g.copy()
    out[sr : sr + h, sc : sc + w] = river
    out[tr : tr + h, tc : tc + w] = _MOVER
    return out


def _paint_h(grid: np.ndarray, r: int, c0: int, c1: int, band_h: int, mover_w: int) -> None:
    left, right = sorted((c0, c1))
    grid[r : r + band_h, left : right + mover_w] = _RIVER


def _paint_v(grid: np.ndarray, r0: int, r1: int, c: int, band_w: int, mover_h: int) -> None:
    top, bottom = sorted((r0, r1))
    grid[top : bottom + mover_h, c : c + band_w] = _RIVER


def _river_connected(grid: np.ndarray) -> bool:
    mask = grid == _RIVER
    if not bool(mask.any()):
        return False
    start = tuple(map(int, np.argwhere(mask)[0]))
    seen = {start}
    q = deque([start])
    while q:
        r, c = q.popleft()
        for dr, dc in _DIR4:
            rr, cc = r + dr, c + dc
            if 0 <= rr < grid.shape[0] and 0 <= cc < grid.shape[1] and bool(mask[rr, cc]) and (rr, cc) not in seen:
                seen.add((rr, cc))
                q.append((rr, cc))
    return len(seen) == int(mask.sum())


def _solve_state(grid: np.ndarray) -> dict:
    bg = _background_color(grid)
    mover, dest = _choose_token_pair(_solid_rectangles(grid, bg))
    river = _river_color(grid, bg)
    h = int(mover["h"])
    w = int(mover["w"])
    sr, sc = int(mover["r0"]), int(mover["c0"])
    dr, dc = int(dest["r0"]), int(dest["c0"])
    trav = (grid == river) | (grid == _MOVER) | (grid == _DEST)
    feasible = _feasible_placements(trav, h, w)
    dist = _bfs_placement(feasible, sr, sc)
    dest_ok = 0 <= dr < feasible.shape[0] and 0 <= dc < feasible.shape[1] and feasible[dr, dc] and dist[dr, dc] >= 0
    tr, tc = (dr, dc) if dest_ok else _farthest_reachable(dist, sr, sc)
    return {
        "mover": mover,
        "dest": dest,
        "river": river,
        "feasible": feasible,
        "dist": dist,
        "dest_ok": bool(dest_ok),
        "target": (int(tr), int(tc)),
    }


def _expected_output(grid: np.ndarray) -> np.ndarray:
    state = _solve_state(grid)
    mover = state["mover"]
    river = int(state["river"])
    tr, tc = state["target"]
    h = int(mover["h"])
    w = int(mover["w"])
    sr, sc = int(mover["r0"]), int(mover["c0"])
    out = grid.copy()
    out[sr : sr + h, sc : sc + w] = river
    out[tr : tr + h, tc : tc + w] = _MOVER
    return out


def _axis_swapped_meta(meta: dict) -> dict:
    gate_sequence = []
    for gate in meta["gate_sequence"]:
        gate_sequence.append(
            {
                "axis": "v" if gate["axis"] == "h" else "h",
                "width": int(gate["width"]),
                "req": int(gate["req"]),
                "req_swapped": int(gate["req_swapped"]),
                "passable": bool(gate["passable"]),
                "passable_swapped": bool(gate["passable_swapped"]),
            }
        )
    meta2 = dict(meta)
    meta2["shape"] = (int(meta["shape"][1]), int(meta["shape"][0]))
    meta2["grid_shape"] = (int(meta["grid_shape"][1]), int(meta["grid_shape"][0]))
    meta2["gate_sequence"] = gate_sequence
    return meta2


def _post_transform(arr: np.ndarray, code: int) -> np.ndarray:
    if code == 0:
        return arr.copy()
    if code == 1:
        return np.fliplr(arr)
    if code == 2:
        return np.flipud(arr)
    if code == 3:
        return np.rot90(arr, 2)
    raise AssertionError("bad transform code")


def _build_case() -> tuple[np.ndarray, np.ndarray, dict] | None:
    h, w = random.choice(_RECT_SHAPES)
    assert h != w

    gap_up = random.randint(h + 1, h + 2)
    gap_down = random.randint(h + 1, h + 2)
    len1 = random.randint(4, 5)
    len2 = random.randint(4, 5)
    len3 = random.randint(4, 5)

    r4 = 1
    r0 = r4 + gap_up
    r2 = r0 + gap_down

    c0 = 1
    c1 = c0 + len1
    c3 = c1 + len2
    c5 = c3 + len3

    max_horiz = h + 2
    H = r2 + max_horiz + 1
    W = c5 + w
    if not (10 <= H <= 20 and 10 <= W <= 20 and H != W):
        return None

    horiz_extras = random.sample([0, 1, 2], 3)
    vert_extras = random.sample([0, 1, 2], 2)
    seg_widths = [
        h + horiz_extras[0],
        w + vert_extras[0],
        h + horiz_extras[1],
        w + vert_extras[1],
        h + horiz_extras[2],
    ]

    reached = random.random() < 0.5
    blocked_segment = None
    if not reached:
        blocked_segment = random.choice([2, 3, 4])
        if blocked_segment % 2 == 0:
            seg_widths[blocked_segment] = random.randint(1, h - 1)
        else:
            seg_widths[blocked_segment] = random.randint(1, w - 1)

    if len(set(int(x) for x in seg_widths)) < 3:
        return None

    grid = np.full((H, W), _BG, dtype=int)
    points = [(r0, c0), (r0, c1), (r2, c1), (r2, c3), (r4, c3), (r4, c5)]

    for r, c in points:
        grid[r : r + h, c : c + w] = _RIVER

    _paint_h(grid, points[0][0], points[0][1], points[1][1], seg_widths[0], w)
    _paint_v(grid, points[1][0], points[2][0], points[1][1], seg_widths[1], h)
    _paint_h(grid, points[2][0], points[2][1], points[3][1], seg_widths[2], w)
    _paint_v(grid, points[3][0], points[4][0], points[3][1], seg_widths[3], h)
    _paint_h(grid, points[4][0], points[4][1], points[5][1], seg_widths[4], w)

    mover_r, mover_c = points[0]
    dest_r, dest_c = points[-1]
    grid[mover_r : mover_r + h, mover_c : mover_c + w] = _RIVER
    grid[dest_r : dest_r + h, dest_c : dest_c + w] = _RIVER

    if not _river_connected(grid):
        return None

    inp = grid.copy()
    inp[mover_r : mover_r + h, mover_c : mover_c + w] = _MOVER
    inp[dest_r : dest_r + h, dest_c : dest_c + w] = _DEST

    state = _solve_state(inp)
    if not bool(state["feasible"][mover_r, mover_c]):
        return None
    if bool(state["dest_ok"]) != bool(reached):
        return None

    expected = _expected_output(inp)
    if not np.array_equal(main(inp), expected):
        return None

    gate_sequence = []
    reqs = [h, w, h, w, h]
    reqs_swapped = [w, h, w, h, w]
    for idx, width in enumerate(seg_widths):
        axis = "h" if idx % 2 == 0 else "v"
        gate_sequence.append(
            {
                "axis": axis,
                "width": int(width),
                "req": int(reqs[idx]),
                "req_swapped": int(reqs_swapped[idx]),
                "passable": bool(width >= reqs[idx]),
                "passable_swapped": bool(width >= reqs_swapped[idx]),
            }
        )

    meta = {
        "shape": (int(h), int(w)),
        "grid_shape": (int(H), int(W)),
        "reached": bool(reached),
        "blocked_segment": blocked_segment,
        "gate_sequence": gate_sequence,
    }

    if random.random() < 0.5:
        inp = np.rot90(inp, 1)
        expected = np.rot90(expected, 1)
        meta = _axis_swapped_meta(meta)

    code = random.randint(0, 3)
    inp = _post_transform(inp, code)
    expected = _post_transform(expected, code)
    meta["grid_shape"] = tuple(map(int, inp.shape))

    if inp.shape[0] == inp.shape[1]:
        return None
    if np.array_equal(inp, expected):
        return None
    if not np.array_equal(main(inp), expected):
        return None
    return inp, expected, meta


def _generate_case_with_meta() -> tuple[np.ndarray, dict]:
    for _ in range(300):
        built = _build_case()
        if built is not None:
            inp, _expected, meta = built
            return inp, meta
    raise RuntimeError("Failed to generate valid input after 300 attempts")


def generate_input() -> np.ndarray:
    return _generate_case_with_meta()[0]
