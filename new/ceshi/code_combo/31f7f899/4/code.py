# Combo variant of 31f7f899 (index 4)
# combo_spec: D3/1+D5/2
# Name: 十字形精灵+旋转放入

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

SOLVER_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(SOLVER_DIR) not in sys.path:
    sys.path.append(str(SOLVER_DIR))

from arc_utils import bbox_h, find_components, get_bg


# Design analysis for this combo variant
# a) Concrete behavior and trace:
#    Seed task objects are centered vertical rectangles that intersect the base row across
#    their middle, so "shape reuse" and "slot anchor" are both read from the same centered
#    geometry. Here each source object is instead a monochrome plus sprite whose bottom tip
#    touches the base row. For arm length a=2, the input sprite relative to its base-row
#    anchor is:
#      vertical spine: (-4,0),(-3,0),(-2,0),(-1,0),(0,0)
#      horizontal arm: (-2,-2),(-2,-1),(-2,0),(-2,1),(-2,2)
#    If anchors are [4, 14, 24], slot colors are [red, green, yellow], and input arm
#    lengths are [4, 1, 3], the seed logic would sort by size and paste the raw shapes back
#    upright into those three slots. The combo logic also sorts by size, but before pasting
#    it rotates each chosen sprite around its base-row anchor using (dr, dc) -> (dc, -dr).
#    The a=2 example above becomes a sideways cross whose left tip lies on the slot anchor:
#      base-row arm: (0,0),(0,1),(0,2),(0,3),(0,4)
#      vertical arm: (-2,2),(-1,2),(0,2),(1,2),(2,2)
#    So the visible differences are coupled: objects are plus-shaped rather than bars, and
#    the sorted object placed into each slot is rotated before recoloring to that slot color.
# b) Complexity calibration:
#    Arm lengths come from a small distinct set {1,2,3,4}. This keeps the sorting cue clear
#    while providing enough variation that 2-4 examples visibly support the rule. The plus
#    silhouettes are larger than 1xN bars and contain both vertical and horizontal strokes,
#    making the D3 source-shape change obvious. Anchors are spaced with one guaranteed blank
#    column between neighboring objects in both input and output, avoiding accidental merges.
# c) Hidden-assumption audit against the seed solver:
#    - Base-row detection still assumes one non-bg color dominates some row. Generator keeps
#      a full-width baseline, so the assumption remains valid.
#    - Component extraction still assumes isolated monochrome objects. Generator enforces
#      separation, so find_components(..., monochromatic=True) remains valid.
#    - The seed anchor heuristic uses the median column among component cells on the base row.
#      That would fail if an object touched the base row in multiple columns, so the generator
#      specifically makes each input cross touch the base row at exactly one cell: its bottom
#      tip. This keeps the slot identity well defined.
#    - bbox_h was previously the bar height. For these bottom-anchored crosses, bbox_h is
#      2*a+1, so arm length is recovered as (bbox_h-1)//2. This is a new assumption: objects
#      must be canonical plus sprites, not arbitrary blobs.
#    - The seed placement loop reused stored coordinates unchanged. That assumption is broken:
#      the combo variant must rotate the stored relative coordinates before placement.
# d) Placement legality:
#    Generator is output-first in the sense that slot anchors are chosen to satisfy both the
#    rotated output footprint and the upright input footprint for one shared set of anchors.
#    For slot i with input arm p_i and sorted output arm s_i:
#      consecutive anchor gaps >= max(p_i + p_{i+1} + 2, 2*s_i + 2)
#    This guarantees one blank column between neighbors in the input and in the rotated
#    output. Vertical bounds are handled by choosing odd grid sizes with base_r >= 2*max_arm.
#    Every candidate is oracle-checked with main(inp) == expected_output.
# e) Visual observability:
#    A human sees the same baseline and fixed slot colors as in the seed task, but the
#    movable objects are now crosses with varying arm lengths. Across examples, the shorter
#    crosses migrate to the leftmost slots exactly as before, yet the outputs are no longer
#    upright: each reassigned cross lies sideways because the anchor is the tip on the base
#    row, not the cross center. This makes the fusion readable from a few examples.
# f) New solver assumptions and how the generator guarantees them:
#    - Every non-baseline component is a single canonical plus sprite with odd bbox height.
#    - Every such sprite has exactly one cell on the base row, used as the slot anchor.
#    - Arm lengths are distinct, so sorting is unambiguous.
#    - Shared anchors leave enough horizontal room for both upright and rotated placements.
#    The generator explicitly constructs only objects with those properties and rejects any
#    candidate that violates the oracle or collapses to the no-rotation variant.


def _cross_rel(arm: int) -> list[tuple[int, int]]:
    a = int(arm)
    pts = {(int(dr), 0) for dr in range(-2 * a, 1)}
    pts.update({(-a, int(dc)) for dc in range(-a, a + 1)})
    return sorted(pts)


def _rotate_rel(rel: list[tuple[int, int]]) -> list[tuple[int, int]]:
    return [(int(dc), int(-dr)) for dr, dc in rel]


def _extract_objects(grid: np.ndarray) -> tuple[int, int, int, list[dict[str, object]]]:
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

    best = None  # (count, row, color)
    for r in range(H):
        vals, counts = np.unique(g[r, :], return_counts=True)
        for v, cnt in zip(vals, counts):
            v = int(v)
            cnt = int(cnt)
            if v == bg:
                continue
            if best is None or cnt > best[0]:
                best = (cnt, int(r), int(v))
    assert best is not None
    _cnt, base_r, base_col = best

    objects: list[dict[str, object]] = []
    for comp in find_components(g, bg=bg, connectivity=4, monochromatic=True, return_format="dict"):
        col = int(comp["color"])
        if col in {bg, int(base_col)}:
            continue

        pts = comp["coords"]
        base_cols = sorted(int(c) for r, c in pts if int(r) == int(base_r))
        assert len(base_cols) == 1
        anchor = int(base_cols[0])

        height = int(bbox_h(comp["bbox"]))
        assert height % 2 == 1 and height >= 3
        arm = int((height - 1) // 2)

        rel = sorted((int(r) - int(base_r), int(c) - int(anchor)) for r, c in pts)
        objects.append(
            {
                "color": int(col),
                "anchor": int(anchor),
                "arm": int(arm),
                "size": int(len(rel)),
                "rel": rel,
            }
        )

    return int(bg), int(base_r), int(base_col), objects


def _solve(grid: np.ndarray, *, rotate: bool) -> np.ndarray:
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg, base_r, base_col, objects = _extract_objects(g)

    slots = sorted(objects, key=lambda o: (int(o["anchor"]), int(o["color"])))
    shapes = sorted(objects, key=lambda o: (int(o["arm"]), int(o["size"]), int(o["anchor"]), int(o["color"])))

    out = np.full((H, W), int(bg), dtype=int)
    out[int(base_r), :] = int(base_col)

    for slot, shape in zip(slots, shapes):
        slot_col = int(slot["color"])
        slot_anchor = int(slot["anchor"])
        rel_coords = _rotate_rel(shape["rel"]) if rotate else list(shape["rel"])
        for dr, dc in rel_coords:
            rr = int(base_r) + int(dr)
            cc = int(slot_anchor) + int(dc)
            if 0 <= rr < H and 0 <= cc < W:
                out[rr, cc] = int(slot_col)

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Sort bottom-anchored plus sprites by arm length, then rotate each chosen sprite
    by 90 degrees around its base-row tip before placing it into the fixed left-to-right
    slots and recoloring it to the destination slot color.
    """
    return _solve(grid, rotate=True)


def _solve_without_rotation(grid: np.ndarray) -> np.ndarray:
    return _solve(grid, rotate=False)


def _rand_composition(total: int, parts: int) -> list[int]:
    return [int(x) for x in np.random.multinomial(int(total), np.full(int(parts), 1.0 / int(parts)))]


def _non_identity_perm(k: int) -> np.ndarray:
    perm = np.random.permutation(k)
    if np.array_equal(perm, np.arange(k)):
        a, b = np.random.choice(k, size=2, replace=False)
        perm[a], perm[b] = perm[b], perm[a]
    return perm


def _slot_anchors(n: int, input_arms: list[int], output_arms: list[int]) -> list[int]:
    k = len(input_arms)
    if k != len(output_arms):
        raise ValueError("arm lists must have the same length")

    min_left = int(input_arms[0])
    min_right = int(max(input_arms[-1], 2 * output_arms[-1]))
    min_gaps = [
        int(max(input_arms[i] + input_arms[i + 1] + 2, 2 * output_arms[i] + 2))
        for i in range(k - 1)
    ]

    used = int(min_left + min_right + 1 + sum(min_gaps))
    extra = int(n - used)
    if extra < 0:
        raise ValueError("crosses do not fit")

    extras = _rand_composition(extra, k + 1)
    anchors = [int(min_left + extras[0])]
    for i, gap in enumerate(min_gaps):
        anchors.append(int(anchors[-1] + gap + extras[i + 1]))
    return anchors


def _draw_scene(
    n: int,
    base_r: int,
    base_color: int,
    bg: int,
    anchors: list[int],
    colors: list[int],
    arms: list[int],
    *,
    rotate: bool,
) -> np.ndarray:
    grid = np.full((n, n), int(bg), dtype=int)
    grid[base_r, :] = int(base_color)
    for anchor, color, arm in zip(anchors, colors, arms):
        rel = _cross_rel(int(arm))
        if rotate:
            rel = _rotate_rel(rel)
        for dr, dc in rel:
            rr = int(base_r) + int(dr)
            cc = int(anchor) + int(dc)
            if not (0 <= rr < n and 0 <= cc < n):
                raise ValueError("placement out of bounds")
            grid[rr, cc] = int(color)
    return grid


def _rotation_delta_metric(grid: np.ndarray) -> int:
    return int(np.sum(main(grid) != _solve_without_rotation(grid)))


def _gen_attempt() -> np.ndarray | None:
    bg = 8
    base_color = 6

    arm_pool = [1, 2, 3, 4]
    k = int(np.random.randint(3, 5))
    sorted_arms = sorted(int(a) for a in np.random.choice(arm_pool, size=k, replace=False))
    perm = _non_identity_perm(k)
    input_arms = [int(sorted_arms[i]) for i in perm]

    max_arm = int(max(sorted_arms))
    min_n = max(4 * max_arm + 1, input_arms[0] + max(input_arms[-1], 2 * sorted_arms[-1]) + 1)
    min_gaps = [
        int(max(input_arms[i] + input_arms[i + 1] + 2, 2 * sorted_arms[i] + 2))
        for i in range(k - 1)
    ]
    min_n = max(int(min_n), int(input_arms[0] + sum(min_gaps) + max(input_arms[-1], 2 * sorted_arms[-1]) + 1))

    candidate_ns = [n for n in range(max(17, min_n), 41) if n % 2 == 1]
    if not candidate_ns:
        return None
    n = int(np.random.choice(candidate_ns))
    base_r = n // 2

    try:
        anchors = _slot_anchors(n, input_arms, sorted_arms)
    except ValueError:
        return None

    colors = [int(c) for c in np.random.choice([1, 2, 3, 4, 5, 7, 9], size=k, replace=False)]

    try:
        inp = _draw_scene(n, base_r, base_color, bg, anchors, colors, input_arms, rotate=False)
        out = _draw_scene(n, base_r, base_color, bg, anchors, colors, sorted_arms, rotate=True)
    except ValueError:
        return None

    if np.array_equal(inp, out):
        return None
    if not np.array_equal(main(inp), out):
        return None
    if np.array_equal(_solve_without_rotation(inp), out):
        return None
    if _rotation_delta_metric(inp) <= 0:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(300):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input failed after 300 attempts")
