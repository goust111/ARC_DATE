# Variant of fc7cae8d
# Dimension: D10 — 干扰项与噪声 (meta)
# Change: 噪声边有35-60%密度的 color 5 随机像素，无其他干扰... → 内部干扰矩形

from __future__ import annotations

from collections import Counter
from pathlib import Path
import random
import sys

import numpy as np

ROOT = Path("/home/lixinyu/snap/new/arc2_solver_select")
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from arc_utils import crop_bbox, find_components, get_bg, paste, random_free_location


# Design analysis for the D10 variant:
# a) Concrete behavior trace:
#    Original input had one interior object, so `extract_center_object()` could safely pick the
#    largest non-border component. Here the input keeps the same two frame controls, but the
#    interior now contains 1 true guided rectangle plus 1-2 bare distractor rectangles.
#    Example: if the frame controls are `top=2` and `left=6`, the true source might be a rotated
#    object whose visible guide sides are `right=2` and `bottom=6`, while a larger bare rectangle
#    sits elsewhere. Original logic would crop the largest rectangle and fail or return nonsense;
#    the variant must instead scan every interior component, test all dihedral transforms, and
#    select the unique component whose boundary guide colors can align to the frame controls.
# b) Complexity calibration:
#    The distractors are always present, use rectangular filled bodies like the target core, and
#    one distractor is often comparable to or larger than the true source component. This keeps the
#    visual change obvious and breaks the original "largest component" shortcut.
# c) Implicit assumption audit against the original solver:
#    - `detect_controls_frame()` still works because the frame rule is unchanged: exactly two solid
#      control edges, no new border clutter.
#    - `extract_center_object()` is no longer valid because "largest interior object" is false.
#    - `guide_edges()` remains valid only if control colors appear exclusively on real guide edges.
#      The generator therefore forbids distractors and the target core interior from using either
#      control color.
# d) Placement legality:
#    Every placed object uses `random_free_location(..., border=1, padding=1)` so it stays off the
#    border controls, never overlaps another object, and remains a separate 4-connected component.
# e) Visual observability:
#    Across examples, humans can now see several interior rectangles. Only one rectangle carries the
#    control colors on two sides; the others are plainly unguided distractors. That makes the D10
#    change visible from input-output pairs instead of being a hidden implementation detail.
#
# New solver assumptions introduced by this variant and guaranteed by the generator:
# - Exactly one interior component admits a dihedral transform whose guide colors match the two
#   frame controls.
# - Distractors never use the frame-control colors, so they cannot fake a valid guide match.
# - The frame still has exactly one horizontal and one vertical control edge.


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
        col, count = Counter(non_bg).most_common(1)[0]
        if int(count) >= len(vals) - 1:
            cand.append((side, int(col), int(count), int(len(vals))))
    cand.sort(key=lambda t: (t[2] / t[3], t[2]), reverse=True)
    return [(side, col) for side, col, _count, _length in cand[:2]]


def _solid_edge_colors(arr: np.ndarray) -> dict[str, int]:
    edges = {
        "top": arr[0, :],
        "bottom": arr[-1, :],
        "left": arr[:, 0],
        "right": arr[:, -1],
    }
    out: dict[str, int] = {}
    for side, values in edges.items():
        vals = list(map(int, values))
        col, count = Counter(vals).most_common(1)[0]
        if int(count) >= len(vals) - 1:
            out[side] = int(col)
    return out


def _find_target_component(g: np.ndarray, bg: int, desired: dict[str, int]) -> np.ndarray:
    H, W = g.shape
    comps = find_components(g, bg=int(bg), connectivity=4, monochromatic=False, return_format="dict")
    best: tuple[int, np.ndarray] | None = None

    for comp in comps:
        r0, c0, r1, c1 = map(int, comp["bbox"])
        if r0 == 0 or c0 == 0 or r1 == H - 1 or c1 == W - 1:
            continue

        obj = crop_bbox(g, (r0, c0, r1, c1))
        for k in range(4):
            base = np.rot90(obj, int(k))
            for flip in (False, True):
                transformed = np.fliplr(base) if flip else base
                edge_map = _solid_edge_colors(transformed)
                if any(side not in edge_map or int(edge_map[side]) != int(col) for side, col in desired.items()):
                    continue

                top = 1 if "top" in desired else 0
                bottom = int(transformed.shape[0]) - (1 if "bottom" in desired else 0)
                left = 1 if "left" in desired else 0
                right = int(transformed.shape[1]) - (1 if "right" in desired else 0)
                core = transformed[int(top) : int(bottom), int(left) : int(right)]
                exact_bonus = 10000 if set(edge_map) == set(desired) else 0
                score = exact_bonus + int(comp["area"])
                if best is None or score > best[0]:
                    best = (score, core)

    if best is None:
        raise ValueError("no guided interior component matches the frame controls")
    return best[1]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Variant rule:
    - The frame still gives one horizontal and one vertical control side.
    - Multiple interior rectangles may appear.
    - Only the true source rectangle has two guide edges whose colors match the controls.
    - Apply the dihedral transform that aligns those guides to the frame sides, then crop guides.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    desired = {side: col for side, col in _detect_controls_frame(g, bg)}
    return _find_target_component(g, bg, desired)


def _transform(arr: np.ndarray, k: int, flip: bool) -> np.ndarray:
    out = np.rot90(arr, int(k))
    return np.fliplr(out) if flip else out


def _inverse_transform(arr: np.ndarray, k: int, flip: bool) -> np.ndarray:
    out = np.fliplr(arr) if flip else arr
    return np.rot90(out, -int(k))


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
            if random.random() < 0.5:
                line_len = random.randint(2, min(5, w))
                rr = random.randrange(h)
                c0 = random.randrange(max(1, w - line_len + 1))
                cells = [(rr, c0 + dc) for dc in range(line_len)]
            else:
                line_len = random.randint(2, min(5, h))
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
            if core[r, c] != fill or painted >= target_nonfill:
                continue
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


def _choose_core_dims(n: int) -> tuple[int, int] | None:
    max_core_h = min(14, n - 7)
    max_core_w = min(14, n - 7)
    if max_core_h < 5 or max_core_w < 5:
        return None
    return random.randint(5, max_core_h), random.randint(5, max_core_w)


def _choose_distractor_dims(
    n: int, source_h: int, source_w: int, want_large: bool, current_count: int
) -> tuple[int, int] | None:
    if current_count == 0 and want_large:
        min_h = min(n - 6, max(4, source_h - 1))
        min_w = min(n - 6, max(4, source_w - 1))
        max_h = min(n - 6, max(source_h + 2, min_h))
        max_w = min(n - 6, max(source_w + 2, min_w))
    else:
        min_h = 4
        min_w = 4
        max_h = min(n - 6, max(6, source_h + 1))
        max_w = min(n - 6, max(6, source_w + 1))

    if min_h > max_h or min_w > max_w:
        return None
    return random.randint(min_h, max_h), random.randint(min_w, max_w)


def _gen_attempt() -> np.ndarray | None:
    n = random.randint(24, 30)
    horiz_side, vert_side = random.choice(
        [("top", "left"), ("top", "right"), ("bottom", "left"), ("bottom", "right")]
    )

    all_colors = list(range(1, 10))
    ctrl_h, ctrl_v = random.sample(all_colors, 2)
    source_pool = [c for c in all_colors if c not in {ctrl_h, ctrl_v}]
    fill = random.choice(source_pool)
    pattern_pool = [c for c in source_pool if c != fill]
    n_pattern_colors = random.randint(2, min(4, len(pattern_pool)))
    pattern_colors = random.sample(pattern_pool, n_pattern_colors)

    dims = _choose_core_dims(n)
    if dims is None:
        return None
    core_h, core_w = dims
    core = _build_core(core_h, core_w, fill, pattern_colors)

    guide_colors = {horiz_side: ctrl_h, vert_side: ctrl_v}
    aligned_object = _attach_guides(core, (horiz_side, vert_side), guide_colors)
    k = random.randrange(4)
    flip = bool(random.randrange(2))
    input_object = _inverse_transform(aligned_object, k, flip)

    grid = np.zeros((n, n), dtype=int)
    _paint_frame(grid, horiz_side, vert_side, ctrl_h, ctrl_v)

    try:
        r0, c0 = random_free_location(grid, input_object, transparent=0, border=1, padding=1)
    except ValueError:
        return None
    paste(grid, input_object, r0, c0, transparent=0)

    n_distractors = random.randint(1, 2)
    placed_distractors = 0
    want_large = random.random() < 0.7
    source_obj_h, source_obj_w = input_object.shape

    for idx in range(n_distractors):
        dims = _choose_distractor_dims(n, source_obj_h, source_obj_w, want_large, idx)
        if dims is None:
            return None
        dh, dw = dims

        distractor_pool = [c for c in all_colors if c not in {ctrl_h, ctrl_v}]
        d_fill = random.choice(distractor_pool)
        d_pattern_pool = [c for c in distractor_pool if c != d_fill]
        d_pattern_count = random.randint(2, min(4, len(d_pattern_pool)))
        distractor = _build_core(dh, dw, d_fill, random.sample(d_pattern_pool, d_pattern_count))

        try:
            dr, dc = random_free_location(grid, distractor, transparent=0, border=1, padding=1)
        except ValueError:
            return None
        paste(grid, distractor, dr, dc, transparent=0)
        placed_distractors += 1

    if placed_distractors != n_distractors:
        return None
    if not np.array_equal(main(grid), core):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate input for fc7cae8d D10 variant")
