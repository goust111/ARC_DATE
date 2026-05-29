# Variant of fc7cae8d
# Dimension: D4 — 变换类型 (transform)
# Change: 对齐后裁掉引导边（crop），输出纯核心矩形 → 保留引导边并重着色

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc2_solver_select.arc_utils import crop_bbox, find_components, get_bg, paste, random_free_location


"""
Design analysis for the D4 variant
==================================

Reference-task baseline:
- Source task `20270e3b` keeps the docking pixels after alignment and recolors them into the body.
- This variant applies the same structural idea here: after the correct D4 alignment is found, the guide edges are
  not removed; they are absorbed into the rectangle by recoloring them to the rectangle fill color.

1) Concrete behavior trace
- Original aligned object example:
    guide-top color 7, guide-left color 3, corner 0
    0 7 7 7
    3 4 4 9
    3 4 8 4
- Original solver matched frame sides, then sliced away row 0 and col 0, yielding:
    4 4 9
    4 8 4
- Variant solver matches the same alignment, but instead of slicing it recolors the kept guide row/col and the corner
  to fill color 4, yielding:
    4 4 4 4
    4 4 4 9
    4 4 8 4
- The changed step is only the post-alignment action; input construction and D4 search stay the same.

2) Complexity calibration
- The object still has two colored guides on adjacent sides, so the D4 reasoning remains necessary.
- The output difference is visually obvious because the kept guides add exactly one row and one column to the cropped
  core, producing a larger rectangle with a new fill-colored border on the controlled sides.
- Internal patterns remain random and asymmetric enough that the central rectangle is not reduced to a trivial solid
  block.

3) Hidden-assumption audit of the original solver
- `detect_controls_frame(...)` depends on frame controls still being mostly solid on exactly two sides. This remains
  true; generator preserves one horizontal and one vertical control edge.
- `extract_center_object(...)` depends on the object being the largest non-border component. This remains true because
  the object is always placed strictly away from the frame.
- `guide_edges(...)` depends on fill being the dominant object color and guide rows/cols differing from fill except for
  at most one corner cell. This still holds in the input; recoloring happens only in the output stage.
- The old crop slice was the only place that relied on the obsolete "guides disappear by deletion" rule. In the
  variant it is invalid because it would erase the very pixels whose preservation is now the point of the task.
- The new recolor step assumes the two guide sides are adjacent and unique. Generator guarantees exactly one horizontal
  and one vertical guide, so recoloring one row and one column is well-defined.

4) Placement legality
- Generator still constructs an aligned object first, applies an inverse D4 transform for the input object, paints the
  frame controls, and uses `random_free_location(..., border=1)` so the object does not touch the frame or exceed the
  canvas.
- Because the object bbox contains the guide-corner background cell, `extract_center_object` can recover the whole
  object exactly after placement.

5) Visual observability
- Across samples, the controlled sides vary among the four corners, so the retained border appears on different sides.
- The output is always larger than the original crop result by one row and one column, which makes the "do not crop"
  inversion visible from a few examples rather than only inferable from code-level reasoning.
"""


def _transform(arr: np.ndarray, k: int, flip: bool) -> np.ndarray:
    out = np.rot90(arr, int(k))
    return np.fliplr(out) if flip else out


def _inverse_transform(arr: np.ndarray, k: int, flip: bool) -> np.ndarray:
    out = np.fliplr(arr) if flip else arr
    return np.rot90(out, -int(k))


def _dominant_fill(arr: np.ndarray) -> int:
    return int(Counter(map(int, arr.flatten())).most_common(1)[0][0])


def _detect_controls_frame(arr: np.ndarray, bg: int) -> list[tuple[str, int]]:
    sides = {
        "top": arr[0, 1:-1],
        "bottom": arr[-1, 1:-1],
        "left": arr[1:-1, 0],
        "right": arr[1:-1, -1],
    }
    cand: list[tuple[str, int, int, int]] = []
    for side, values in sides.items():
        vals = list(map(int, values))
        non_bg = [x for x in vals if x != int(bg)]
        if not non_bg:
            continue
        color, count = Counter(non_bg).most_common(1)[0]
        if int(count) >= len(vals) - 1:
            cand.append((side, int(color), int(count), int(len(vals))))
    cand.sort(key=lambda t: (t[2] / t[3], t[2]), reverse=True)
    return [(side, color) for side, color, _count, _length in cand[:2]]


def _extract_center_object(arr: np.ndarray, bg: int) -> np.ndarray:
    height, width = arr.shape
    comps = find_components(arr, bg=int(bg), connectivity=4, monochromatic=False, return_format="dict")
    best = None
    for comp in comps:
        r0, c0, r1, c1 = map(int, comp["bbox"])
        if bool(r0 == 0 or c0 == 0 or r1 == height - 1 or c1 == width - 1):
            continue
        area = int(comp["area"])
        if best is None or area > best[0]:
            best = (area, (r0, c0, r1, c1))
    if best is None:
        return arr.copy()
    _area, bbox = best
    return crop_bbox(arr, bbox)


def _guide_edges(arr: np.ndarray) -> dict[str, int]:
    fill = _dominant_fill(arr)
    edges = {
        "top": arr[0, :],
        "bottom": arr[-1, :],
        "left": arr[:, 0],
        "right": arr[:, -1],
    }
    out: dict[str, int] = {}
    for side, values in edges.items():
        vals = list(map(int, values))
        non_fill = [x for x in vals if x != int(fill)]
        if not non_fill:
            continue
        color, count = Counter(non_fill).most_common(1)[0]
        if int(count) >= len(vals) - 1:
            out[side] = int(color)
    return out


def _recolor_guides_to_fill(arr: np.ndarray, desired_sides: dict[str, int]) -> np.ndarray:
    fill = _dominant_fill(arr)
    out = np.array(arr, copy=True)
    if "top" in desired_sides:
        out[0, :] = int(fill)
    if "bottom" in desired_sides:
        out[-1, :] = int(fill)
    if "left" in desired_sides:
        out[:, 0] = int(fill)
    if "right" in desired_sides:
        out[:, -1] = int(fill)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Align the central rectangle to the frame controls via a dihedral transform.

    Variant rule:
    - Find the D4 transform whose two guide-edge colors match the frame-control sides.
    - Keep the 1px guides instead of cropping them.
    - Recolor those guides, plus their shared corner, to the rectangle's dominant fill color.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    desired = {side: color for side, color in _detect_controls_frame(g, bg)}
    obj = _extract_center_object(g, bg)

    for k in range(4):
        base = np.rot90(obj, int(k))
        for flip in (False, True):
            transformed = np.fliplr(base) if flip else base
            guides = _guide_edges(transformed)
            if all(side in guides and int(guides[side]) == int(color) for side, color in desired.items()):
                return _recolor_guides_to_fill(transformed, desired)

    return obj


def _build_core(h: int, w: int, fill: int, pattern_colors: list[int]) -> np.ndarray:
    core = np.full((h, w), int(fill), dtype=int)
    area = h * w
    max_nonfill = max(len(pattern_colors), int(area * 0.35))
    target_nonfill = random.randint(max(len(pattern_colors), area // 10), max_nonfill)
    painted = 0

    for color in pattern_colors:
        r = random.randrange(h)
        c = random.randrange(w)
        core[r, c] = int(color)
        painted += 1

    while painted < target_nonfill:
        color = random.choice(pattern_colors)
        shape_kind = random.choice(["dot", "line", "rect"])
        if shape_kind == "dot":
            cells = [(random.randrange(h), random.randrange(w))]
        elif shape_kind == "line":
            line_len = random.randint(2, min(5, max(h, w)))
            if random.random() < 0.5:
                rr = random.randrange(h)
                c0 = random.randrange(max(1, w - line_len + 1))
                cells = [(rr, c0 + dc) for dc in range(line_len)]
            else:
                r0 = random.randrange(max(1, h - line_len + 1))
                cc = random.randrange(w)
                cells = [(r0 + dr, cc) for dr in range(line_len)]
        else:
            mh = random.randint(1, min(3, h))
            mw = random.randint(1, min(3, w))
            r0 = random.randrange(h - mh + 1)
            c0 = random.randrange(w - mw + 1)
            cells = [(r0 + dr, c0 + dc) for dr in range(mh) for dc in range(mw)]

        remaining = target_nonfill - painted
        if remaining <= 0:
            break
        fill_cells = [(r, c) for r, c in cells if core[r, c] == fill]
        if len(fill_cells) > remaining:
            fill_cells = random.sample(fill_cells, remaining)
        for r, c in fill_cells:
            core[r, c] = int(color)
        painted += len(fill_cells)
        for r, c in cells:
            if core[r, c] != fill:
                continue
            if painted >= target_nonfill:
                break
            core[r, c] = int(color)
            painted += 1

    return core


def _attach_guides(core: np.ndarray, guide_sides: tuple[str, str], guide_colors: dict[str, int]) -> np.ndarray:
    h, w = core.shape
    has_top = "top" in guide_sides
    has_bottom = "bottom" in guide_sides
    has_left = "left" in guide_sides
    has_right = "right" in guide_sides
    out = np.zeros((h + has_top + has_bottom, w + has_left + has_right), dtype=int)
    r0 = 1 if has_top else 0
    c0 = 1 if has_left else 0
    paste(out, core, r0, c0)

    if has_top:
        out[0, c0 : c0 + w] = int(guide_colors["top"])
    if has_bottom:
        out[-1, c0 : c0 + w] = int(guide_colors["bottom"])
    if has_left:
        out[r0 : r0 + h, 0] = int(guide_colors["left"])
    if has_right:
        out[r0 : r0 + h, -1] = int(guide_colors["right"])

    if has_top and has_left:
        out[0, 0] = 0
    if has_top and has_right:
        out[0, -1] = 0
    if has_bottom and has_left:
        out[-1, 0] = 0
    if has_bottom and has_right:
        out[-1, -1] = 0
    return out


def _oracle_output(aligned_object: np.ndarray, guide_sides: tuple[str, str]) -> np.ndarray:
    desired = {side: 1 for side in guide_sides}
    return _recolor_guides_to_fill(aligned_object, desired)


def _paint_frame(grid: np.ndarray, horiz_side: str, vert_side: str, ctrl_h: int, ctrl_v: int) -> None:
    n = grid.shape[0]
    if horiz_side == "top":
        grid[0, :] = int(ctrl_h)
    else:
        grid[-1, :] = int(ctrl_h)
    if vert_side == "left":
        grid[:, 0] = int(ctrl_v)
    else:
        grid[:, -1] = int(ctrl_v)

    for side in ("top", "bottom", "left", "right"):
        if side in (horiz_side, vert_side):
            continue
        if side in ("top", "bottom"):
            r = 0 if side == "top" else n - 1
            coords = [(r, c) for c in range(1, n - 1)]
        else:
            c = 0 if side == "left" else n - 1
            coords = [(r, c) for r in range(1, n - 1)]
        min_count = int(np.ceil(0.35 * len(coords)))
        max_count = int(np.floor(0.60 * len(coords)))
        noise_count = random.randint(min_count, max_count)
        for r, c in random.sample(coords, noise_count):
            grid[r, c] = 5


def _gen_attempt() -> np.ndarray | None:
    n = random.randint(22, 29)
    horiz_side, vert_side = random.choice(
        [("top", "left"), ("top", "right"), ("bottom", "left"), ("bottom", "right")]
    )

    all_colors = list(range(1, 10))
    ctrl_h, ctrl_v = random.sample([c for c in all_colors if c != 5], 2)
    remaining = [c for c in all_colors if c not in {5, ctrl_h, ctrl_v}]
    fill = random.choice(remaining)
    remaining = [c for c in remaining if c != fill]
    n_pattern_colors = random.randint(2, min(4, len(remaining)))
    pattern_colors = random.sample(remaining, n_pattern_colors)

    max_core_h = min(20, n - 3)
    max_core_w = min(20, n - 3)
    if max_core_h < 5 or max_core_w < 5:
        return None
    core_h = random.randint(5, max_core_h)
    core_w = random.randint(5, max_core_w)
    core = _build_core(core_h, core_w, fill, pattern_colors)

    guide_sides = (horiz_side, vert_side)
    guide_colors = {horiz_side: ctrl_h, vert_side: ctrl_v}
    aligned_object = _attach_guides(core, guide_sides, guide_colors)
    expected = _oracle_output(aligned_object, guide_sides)

    k = random.randrange(4)
    flip = bool(random.randrange(2))
    input_object = _inverse_transform(aligned_object, k, flip)

    grid = np.zeros((n, n), dtype=int)
    _paint_frame(grid, horiz_side, vert_side, ctrl_h, ctrl_v)

    try:
        r0, c0 = random_free_location(grid, input_object, transparent=0, border=1)
    except ValueError:
        return None
    paste(grid, input_object, r0, c0, transparent=0)

    try:
        result = main(grid)
    except Exception:
        return None
    if not np.array_equal(result, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(300):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate input for fc7cae8d D4 variant")
