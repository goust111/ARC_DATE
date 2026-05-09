# Variant of 31f7f899
# Dimension: D9 — 条形数量与颜色约束 (src)
# Change: 3-5个条形，每个使用不同的非背景非基线色，从 {1,2,3,4,5,7,9} 中无放回选取... → 包含干扰色块

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

SOLVER_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(SOLVER_DIR) not in sys.path:
    sys.path.append(str(SOLVER_DIR))

from arc_utils import bbox_h, bbox_w, find_components, get_bg, paste


# Design analysis for the D9 variant
# ==================================
# a) Concrete behavior and trace
#    - The original task contains only one meaningful object family: vertical solid
#      bars centered on a full-width horizontal baseline. Every non-bg / non-baseline
#      component is therefore a sortable bar.
#    - This variant adds extra monochrome "noise blocks" that are not part of the
#      sorting. They sit in the top/bottom margins, use colors different from all bar
#      colors, and stay unchanged from input to output.
#    - Small trace:
#      Input may contain 4 bars crossing the baseline plus 2 floating 2x2 blocks in
#      the corners. Original logic would collect all 6 components, sort them together,
#      and move the corner blocks into bar slots. The variant must first filter:
#      components crossing the baseline and matching the solid centered-rectangle bar
#      geometry are sortable bars; everything else is copied through unchanged.
# b) Complexity calibration
#    - The distractors are visible in every generated sample (at least one, usually
#      two or three), so the variant is observable by eye.
#    - Distractor geometry is kept small (1x2, 2x1, 2x2) to avoid creating a second
#      complex task. The core challenge stays the same: sort bar heights. The added
#      difficulty is only the foreground filtering step.
#    - Bar heights remain distinct odd numbers, which keeps the ranking unambiguous.
# c) Original implicit assumptions that break
#    - Iterating over all monochrome components and treating each as a slot/object
#      relied on "every component is a bar". This is false once distractors exist.
#    - Using every component anchor to define slots would create fake slots at noise
#      positions. Using every component height would also let noise blocks enter the
#      sorted-shape list.
#    - The new solver therefore introduces an explicit bar predicate. It assumes true
#      bars:
#      1. intersect the baseline row,
#      2. are vertically centered on that baseline,
#      3. form a solid rectangle.
#      The generator is built to guarantee exactly those properties.
# d) Placement legality
#    - To keep distractors disjoint from both input and output bars, the generator
#      limits bar heights to leave a top and bottom margin of at least two rows.
#      Distractors are placed entirely inside those margins.
#    - Because bars only move among the original slot columns and never leave the
#      central activity band, margin distractors never overlap sorted outputs.
# e) Visual observability
#    - Across 2-4 examples, humans can see that the corner/edge blocks never move
#      while only the baseline-crossing bars get permuted by height. That difference
#      is visible without needing hidden metadata or color mappings.

BAR_COLORS = [1, 2, 3, 4, 5, 7, 9]
DISTRACTOR_SHAPES = [(1, 2), (2, 1), (2, 2)]


def _base_info(g: np.ndarray, bg: int) -> tuple[int, int]:
    best = None  # (count, row, color)
    for r in range(g.shape[0]):
        vals, counts = np.unique(g[r, :], return_counts=True)
        for v, cnt in zip(vals, counts):
            v = int(v)
            cnt = int(cnt)
            if v == bg:
                continue
            if best is None or cnt > best[0]:
                best = (cnt, int(r), int(v))
    assert best is not None
    return int(best[1]), int(best[2])


def _is_sortable_bar(comp: dict[str, object], base_r: int) -> bool:
    pts = np.array(comp["coords"], dtype=int)
    r0, c0, r1, c1 = [int(x) for x in comp["bbox"]]
    height = int(bbox_h((r0, c0, r1, c1)))
    width = int(bbox_w((r0, c0, r1, c1)))
    area = int(comp["area"])

    if not np.any(pts[:, 0] == int(base_r)):
        return False
    if r0 + r1 != 2 * int(base_r):
        return False
    if area != height * width:
        return False

    base_cols = sorted(int(c) for r, c in pts if int(r) == int(base_r))
    return base_cols == list(range(int(c0), int(c1) + 1))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Variant rule:
    - Ignore foreground distractor blocks that do not match the baseline-crossing bar
      geometry.
    - Sort only the true bars by height and redistribute them across the original
      left-to-right slots, preserving each slot's original color.
    - Preserve both the baseline and the distractor blocks.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))
    base_r, base_col = _base_info(g, bg)

    bars: list[dict[str, object]] = []
    distractors: list[dict[str, object]] = []
    for comp in find_components(g, bg=bg, connectivity=4, monochromatic=True, return_format="dict"):
        col = int(comp["color"])
        if col in {int(bg), int(base_col)}:
            continue

        pts = np.array(comp["coords"], dtype=int)
        if _is_sortable_bar(comp, base_r):
            r0, c0, r1, c1 = [int(x) for x in comp["bbox"]]
            anchor = int((c0 + c1) // 2)
            rel = [(int(r) - int(base_r), int(c) - int(anchor)) for r, c in pts]
            bars.append(
                {
                    "color": int(col),
                    "anchor": int(anchor),
                    "height": int(bbox_h((r0, c0, r1, c1))),
                    "size": int(len(rel)),
                    "rel": rel,
                }
            )
        else:
            distractors.append({"color": int(col), "coords": pts})

    assert len(bars) >= 1

    slots = sorted(bars, key=lambda o: (int(o["anchor"]), int(o["color"])))
    shapes = sorted(bars, key=lambda o: (int(o["height"]), int(o["size"]), int(o["anchor"]), int(o["color"])))

    out = np.full((H, W), int(bg), dtype=int)
    out[int(base_r), :] = int(base_col)

    for comp in distractors:
        out[tuple(np.array(comp["coords"]).T)] = int(comp["color"])

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


def _draw_bars(
    n: int,
    base_r: int,
    base_color: int,
    bg: int,
    lefts: list[int],
    width: int,
    colors: list[int],
    heights: list[int],
) -> np.ndarray:
    grid = np.full((n, n), int(bg), dtype=int)
    grid[base_r, :] = int(base_color)
    for left, color, height in zip(lefts, colors, heights):
        patch = np.full((int(height), int(width)), int(color), dtype=int)
        paste(grid, patch, int(base_r - height // 2), int(left))
    return grid


def _place_distractors(n: int, base_r: int, max_height: int, colors: list[int]) -> list[dict[str, int]]:
    top_h = int(base_r - max_height // 2)
    bottom_start = int(base_r + max_height // 2 + 1)
    bottom_h = int(n - bottom_start)
    occupancy = np.zeros((n, n), dtype=bool)
    distractors: list[dict[str, int]] = []

    for color in colors:
        placed = False
        for _ in range(80):
            h, w = DISTRACTOR_SHAPES[int(np.random.randint(len(DISTRACTOR_SHAPES)))]
            bands: list[tuple[int, int]] = []
            if h <= top_h:
                bands.append((0, top_h))
            if h <= bottom_h:
                bands.append((bottom_start, n))
            if not bands:
                break
            band_r0, band_r1 = bands[int(np.random.randint(len(bands)))]
            r = int(np.random.randint(band_r0, band_r1 - h + 1))
            c = int(np.random.randint(0, n - w + 1))
            if occupancy[r : r + h, c : c + w].any():
                continue
            occupancy[r : r + h, c : c + w] = True
            distractors.append({"color": int(color), "r": int(r), "c": int(c), "h": int(h), "w": int(w)})
            placed = True
            break
        if not placed:
            raise ValueError("failed to place distractor")

    return distractors


def _add_distractors(grid: np.ndarray, distractors: list[dict[str, int]]) -> np.ndarray:
    out = np.array(grid, copy=True)
    for d in distractors:
        patch = np.full((int(d["h"]), int(d["w"])), int(d["color"]), dtype=int)
        paste(out, patch, int(d["r"]), int(d["c"]))
    return out


def _gen_attempt() -> np.ndarray | None:
    n = int(np.random.choice(np.arange(9, 18, 2)))
    base_r = n // 2
    bg = 8
    base_color = 6

    height_choices = list(range(1, min(n - 4, 9) + 1, 2))
    max_count = min(5, len(height_choices), (n + 1) // 2)
    if max_count < 3:
        return None
    k = int(np.random.randint(3, max_count + 1))

    feasible_widths = [1]
    if n - 3 * k > 3 and n >= 4 * k - 1:
        feasible_widths.append(3)
    width = int(np.random.choice(feasible_widths))

    lefts = _slot_lefts(n, k, width)
    colors = [int(c) for c in np.random.choice(BAR_COLORS, size=k, replace=False)]

    sorted_heights = sorted(int(h) for h in np.random.choice(height_choices, size=k, replace=False))
    perm = np.random.permutation(k)
    if np.array_equal(perm, np.arange(k)):
        a, b = np.random.choice(k, size=2, replace=False)
        perm[a], perm[b] = perm[b], perm[a]
    in_heights = [int(sorted_heights[i]) for i in perm]

    distractor_pool = [int(c) for c in range(10) if c not in {bg, base_color, *colors}]
    max_distractors = min(3, len(distractor_pool))
    if max_distractors <= 0:
        return None
    min_distractors = 1 if max_distractors == 1 else 2
    dcount = int(np.random.randint(min_distractors, max_distractors + 1))
    distractor_colors = [int(c) for c in np.random.choice(distractor_pool, size=dcount, replace=False)]
    distractors = _place_distractors(n, base_r, max(sorted_heights), distractor_colors)

    out = _add_distractors(
        _draw_bars(n, base_r, base_color, bg, lefts, width, colors, sorted_heights),
        distractors,
    )
    inp = _add_distractors(
        _draw_bars(n, base_r, base_color, bg, lefts, width, colors, in_heights),
        distractors,
    )

    if np.array_equal(inp, out):
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
