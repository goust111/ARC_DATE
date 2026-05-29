# Combo variant of 31f7f899 (index 5)
# combo_spec: D3/0+D13/0
# Name: 非对称条+单向柱状图

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[4]
_SOLVER_DIR = _ROOT / "solvers"
if str(_SOLVER_DIR) not in sys.path:
    sys.path.append(str(_SOLVER_DIR))

from arc_utils import find_components, get_bg, paste


# Design analysis for the combo variant
# a) Concrete behavior and trace:
#    The original task uses solid vertical rectangles centered on the baseline, so a
#    height-7 bar on base row 4 occupies rows 1..7 and looks vertically symmetric.
#    This combo turns every object into a one-sided histogram column: it still
#    touches the baseline row, but it extends only upward. On a 9x9 grid with
#    `base_r = 4`, three width-1 bars with top extents [4, 1, 3] occupy rows
#    [0..4], [3..4], [1..4]. Input keeps those shapes attached to their original
#    colored slots; output sorts the column shapes by top extent so the visible
#    upward reaches become [1, 3, 4] from left to right while slot colors stay
#    fixed. The key difference from the source task is geometric: there are no
#    mirrored lower halves, so the scene reads as a standard bar chart instead of a
#    symmetric totem around the baseline.
# b) Complexity calibration:
#    The fusion must stay visually obvious without adding too many degrees of
#    freedom. Generator therefore keeps the original slot structure, a single global
#    width per scene (1 or 3), and 3-5 bars with distinct upward extents. The only
#    new structural freedom is one-sidedness: every bar has bottom extent 0 and
#    positive top extent. Distinct heights make the sorting pattern legible from a
#    few examples, while the width remains low-cardinality so slots are still easy
#    to compare.
# c) Hidden-assumption audit of the original main():
#    - Base-row detection via the most frequent non-bg color on a row still works:
#      the full-width baseline remains the only long monochrome run.
#    - Component extraction still works because each bar is still a single
#      monochrome connected component.
#    - The original `bbox_h(...)` ranking key is no longer the intended semantic
#      quantity. On legal combo inputs `bbox_h == top_extent + 1`, so numerically it
#      would coincide, but the fused rule is specifically "sort by upward reach in a
#      one-sided histogram". The solver therefore measures `top_extent` directly.
#    - Anchor recovery from base-row pixels still works because every bar occupies
#      the baseline row and has odd width, so its center column is stable.
#    - Repainting by relative coordinates remains valid because moving a whole
#      component preserves the one-sided silhouette.
#    New solver-side assumptions introduced here:
#    - every non-baseline bar touches the baseline row;
#    - all legal bars satisfy `bottom_extent == 0`;
#    - the scene contains at least one bar, and generator guarantees 3-5.
# d) Placement legality:
#    Bars are rendered as solid rectangles of shape `(top_extent + 1, width)` with
#    their bottom row aligned to the baseline. Because the board size is odd and
#    `top_extent <= base_r`, no bar can extend above row 0. Slots are spaced with at
#    least one background column between rectangles, so neither input nor output can
#    create overlap even after silhouettes are reassigned to different slots.
# e) Visual observability:
#    A human seeing 2-4 examples should notice two stable cues: all non-baseline
#    mass lies above the baseline, and the column heights become left-to-right
#    sorted in the output while colors stay bound to x-slots. This combo is mostly a
#    geometric fusion rather than a rank-disambiguation fusion, so the generator's
#    anti-degeneracy check focuses on histogramness itself: every sampled bar must be
#    strictly asymmetric with zero lower tail, making the one-sided bar-chart cue
#    unmistakable.


def _find_base(grid: np.ndarray, bg: int) -> tuple[int, int]:
    best = None  # (count, row, color)
    H = int(grid.shape[0])
    for r in range(H):
        vals, counts = np.unique(grid[r, :], return_counts=True)
        for v, cnt in zip(vals, counts):
            v = int(v)
            cnt = int(cnt)
            if v == bg:
                continue
            if best is None or cnt > best[0]:
                best = (cnt, int(r), int(v))
    assert best is not None
    _cnt, base_r, base_col = best
    return int(base_r), int(base_col)


def _extract_objects(grid: np.ndarray, base_r: int, base_col: int, bg: int) -> list[dict[str, object]]:
    objects: list[dict[str, object]] = []
    for comp in find_components(grid, bg=bg, connectivity=4, monochromatic=True, return_format="dict"):
        col = int(comp["color"])
        if col in {int(bg), int(base_col)}:
            continue

        pts = [(int(r), int(c)) for r, c in comp["coords"]]
        rows = [r for r, _c in pts]
        base_cols = sorted(c for r, c in pts if r == int(base_r))
        if base_cols:
            anchor = int(base_cols[len(base_cols) // 2])
        else:
            cols_sorted = sorted(c for _r, c in pts)
            anchor = int(cols_sorted[len(cols_sorted) // 2])

        top_extent = int(base_r - min(rows))
        bottom_extent = int(max(rows) - base_r)
        rel = [(r - int(base_r), c - int(anchor)) for r, c in pts]
        objects.append(
            {
                "color": int(col),
                "anchor": int(anchor),
                "top": int(top_extent),
                "bottom": int(bottom_extent),
                "size": int(len(rel)),
                "rel": rel,
            }
        )
    return objects


def main(grid: np.ndarray) -> np.ndarray:
    """
    Sort one-sided vertical bars by their upward extent above the baseline, then
    redistribute those silhouettes across fixed left-to-right colored slots.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))
    base_r, base_col = _find_base(g, bg)

    objects = _extract_objects(g, int(base_r), int(base_col), int(bg))
    assert objects

    slots = sorted(objects, key=lambda o: (int(o["anchor"]), int(o["color"])))
    shapes = sorted(
        objects,
        key=lambda o: (
            int(o["top"]),
            int(o["size"]),
            int(o["anchor"]),
            int(o["color"]),
        ),
    )

    out = np.full((H, W), int(bg), dtype=int)
    out[int(base_r), :] = int(base_col)

    for slot, shape in zip(slots, shapes):
        slot_col = int(slot["color"])
        slot_anchor = int(slot["anchor"])
        for dr, dc in shape["rel"]:
            rr = int(base_r) + int(dr)
            cc = int(slot_anchor) + int(dc)
            if 0 <= rr < H and 0 <= cc < W:
                out[rr, cc] = int(slot_col)

    return out


def _rand_composition(total: int, parts: int) -> list[int]:
    return [int(x) for x in np.random.multinomial(int(total), np.full(int(parts), 1.0 / int(parts)))]


def _slot_lefts(n: int, k: int, width: int) -> list[int]:
    used = int(k * width + (k - 1))
    extra = int(n - used)
    if extra < 0:
        raise ValueError("bars do not fit")
    gaps = _rand_composition(extra, k + 1)
    lefts: list[int] = []
    c = int(gaps[0])
    for i in range(k):
        lefts.append(int(c))
        c += int(width)
        if i < k - 1:
            c += 1 + int(gaps[i + 1])
    return lefts


def _draw_scene(
    n: int,
    base_r: int,
    base_color: int,
    bg: int,
    lefts: list[int],
    width: int,
    colors: list[int],
    top_extents: list[int],
) -> np.ndarray:
    grid = np.full((n, n), int(bg), dtype=int)
    grid[base_r, :] = int(base_color)
    for left, color, top in zip(lefts, colors, top_extents):
        patch = np.full((int(top) + 1, int(width)), int(color), dtype=int)
        paste(grid, patch, int(base_r - top), int(left))
    return grid


def _sample_tops(k: int, max_top: int) -> list[int] | None:
    if int(max_top) < int(k):
        return None
    tops = sorted(int(x) for x in np.random.choice(np.arange(1, int(max_top) + 1), size=int(k), replace=False))
    if tops[-1] - tops[0] < 2:
        return None
    return tops


def _gen_attempt() -> np.ndarray | None:
    n = int(np.random.choice(np.arange(9, 18, 2)))
    base_r = int(n // 2)
    bg = 8
    base_color = 6

    max_count = min(5, int(base_r))
    if max_count < 3:
        return None
    k = int(np.random.randint(3, max_count + 1))

    feasible_widths = [1]
    if n - 3 * k > 3 and n >= 4 * k - 1:
        feasible_widths.append(3)
    width = int(np.random.choice(feasible_widths))

    lefts = _slot_lefts(n, k, width)
    colors = [int(c) for c in np.random.choice([1, 2, 3, 4, 5, 7, 9], size=k, replace=False)]

    tops_sorted = _sample_tops(k, int(base_r))
    if tops_sorted is None:
        return None

    out = _draw_scene(n, base_r, base_color, bg, lefts, width, colors, tops_sorted)

    perm = np.random.permutation(k)
    if np.array_equal(perm, np.arange(k)):
        a, b = np.random.choice(k, size=2, replace=False)
        perm[a], perm[b] = perm[b], perm[a]
    tops_input = [int(tops_sorted[int(i)]) for i in perm]
    inp = _draw_scene(n, base_r, base_color, bg, lefts, width, colors, tops_input)

    if np.array_equal(inp, out):
        return None

    objects = _extract_objects(inp, int(base_r), int(base_color), int(bg))
    if len(objects) != k:
        return None
    if any(int(obj["bottom"]) != 0 for obj in objects):
        return None
    if sorted(int(obj["top"]) for obj in objects) != [int(x) for x in tops_sorted]:
        return None

    if not np.array_equal(main(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(300):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input failed after 300 attempts")
