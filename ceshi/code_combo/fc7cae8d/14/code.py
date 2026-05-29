# Combo variant of fc7cae8d (index 14)
# combo_spec: D2/0+D4/0
# Name: 可变引导宽度+保留重着色

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc2_solver_select.arc_utils import crop_bbox, find_components, get_bg, paste, random_free_location


"""
Design analysis for the combo variant
=====================================

This file fuses two changes onto the original `fc7cae8d` task:
- D2/0: guide strips are no longer fixed at 1px; both adjacent guide sides have width 2 or 3.
- D4/0: after the correct D4 alignment, those guide strips are kept and recolored to the core fill
  instead of being cropped away.

1) Concrete behavior trace
- Original aligned object (top+left guides, width 1):
    0 7 7 7
    3 4 4 9
    3 4 8 4
  Original solver aligns this object, then removes row 0 and col 0, yielding:
    4 4 9
    4 8 4
- Combo aligned object with width 2:
    0 0 7 7 7
    0 0 7 7 7
    3 3 4 4 9
    3 3 4 8 4
  The solver still aligns by matching guide colors to frame-control sides, but now it must first infer
  that the guide width is 2. The output keeps the whole 5x5 bbox and recolors the 2-row top strip, the
  2-col left strip, and the 2x2 corner block to fill color 4:
    4 4 4 4 4
    4 4 4 4 4
    4 4 4 4 9
    4 4 4 8 4
- Width 3 behaves the same way, except that three rows and three columns are absorbed instead of one.
- The key fused difference is that width inference is now part of the reasoning chain, and the inferred
  width controls recoloring depth rather than crop depth.

2) Complexity calibration
- The width only takes values in {2, 3}. That adds a real latent parameter without exploding the search.
- Colors still provide the unique side correspondence, so the task remains structurally close to the
  original rather than turning into generic object parsing.
- The output size visibly depends on guide width: relative to the cropped core, height grows by `t` on the
  guided horizontal side and width grows by `t` on the guided vertical side. This makes the combo behavior
  easy to distinguish from the baseline crop task.

3) Hidden-assumption audit of the original solver
- `detect_controls_frame(...)`: assumes exactly one horizontal and one vertical frame control remain mostly
  solid. Generator keeps that invariant unchanged.
- `extract_center_object(...)`: assumes the interior object is the largest non-border multicolor component.
  Still valid because the object is placed away from the border and nothing else is placed inside.
- Original `guide_edges(...)`: assumed the entire outermost row/col was the guide and therefore only worked
  for width 1. That assumption is broken by D2/0 and is replaced here by exact strip-geometry checking for
  candidate widths 2 and 3.
- Original crop slice `t[top:bottom, left:right]`: assumed guide pixels disappear by deletion. D4/0 breaks
  that assumption; the new post-alignment step recolors the absorbed strips and their corner instead.
- New combo-specific assumption: once the correct transform is chosen, the guide geometry is a clean L-shape
  of width 2 or 3 with a same-size background corner square. Generator guarantees that exact geometry.

4) Placement legality
- The generator builds the aligned object first, then applies an inverse D4 transform to get the input
  object, just like the original task.
- Canvas size is chosen so that `core_h + guide_w <= n - 2` and `core_w + guide_w <= n - 2`; this keeps
  `random_free_location(..., border=1)` feasible and prevents border contact.
- The object bbox includes the background corner square, so `crop_bbox(...)` recovers the full object after
  placement without losing the guide geometry.

5) Visual observability
- Across samples, the controlled corner varies and guide width varies between 2 and 3. Humans can see both
  that the colored strips point to the frame controls and that the output keeps those strips as extra body.
- Because the absorbed border thickness changes between examples, a solver that blindly crops 1 row/col or
  blindly recolors 1 row/col would fail visibly.

6) Visual cleanliness
- The grid contains only three visual layers: frame controls on the border, one interior object, and sparse
  border noise on the uncontrolled frame sides. No overlapping interior objects are introduced.
- Guide colors are reserved away from fill/pattern colors, so the strips remain visually legible before they
  are absorbed in the output.

7) Color-change inferability
- Recoloring is structural, not arbitrary: exactly the strips whose colors match the controlling frame edges
  are converted into the dominant fill of the inner rectangle.
- The generator caps the number of pattern colors and excludes control colors from the core, so humans can
  infer that the strips merge into the rectangle rather than being remapped by a free color permutation.

8) Anti-degeneration check used during validation
- A useful metric is `(output.shape[0] - core.shape[0], output.shape[1] - core.shape[1])`, which must equal
  `(guide_w, guide_w)` because there is exactly one horizontal and one vertical guide side.
- Sampling should show both `(2, 2)` and `(3, 3)`. If every sample collapsed to one width or to `(1, 1)`,
  the combo would have regressed toward the original task.

9) New solver assumptions to mirror in the generator
- The width on the horizontal and vertical guide is the same scalar `guide_w`.
- The guide corner is a clean `guide_w x guide_w` background block.
- The core region contains no background pixels and no control colors, so its dominant color is the fill.
- Exactly one candidate transform/width pair is intended to satisfy the strip checks; generator enforces
  clean guide geometry and reserves colors to keep accidental matches unlikely.
"""


GUIDE_WIDTHS = (2, 3)


def _transform(arr: np.ndarray, k: int, flip: bool) -> np.ndarray:
    out = np.rot90(arr, int(k))
    return np.fliplr(out) if flip else out


def _inverse_transform(arr: np.ndarray, k: int, flip: bool) -> np.ndarray:
    out = np.fliplr(arr) if flip else arr
    return np.rot90(out, -int(k))


def _dominant_color(arr: np.ndarray) -> int:
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


def _guide_layout(
    shape: tuple[int, int], guide_sides: dict[str, int], guide_w: int
) -> tuple[tuple[int, int, int, int], tuple[slice, slice] | None, tuple[slice, slice] | None, tuple[slice, slice]]:
    height, width = map(int, shape)
    guide_w = int(guide_w)
    has_top = "top" in guide_sides
    has_bottom = "bottom" in guide_sides
    has_left = "left" in guide_sides
    has_right = "right" in guide_sides
    if sum((has_top, has_bottom)) != 1 or sum((has_left, has_right)) != 1:
        raise ValueError("guide layout requires exactly one horizontal and one vertical side")

    r0 = guide_w if has_top else 0
    r1 = height - guide_w if has_bottom else height
    c0 = guide_w if has_left else 0
    c1 = width - guide_w if has_right else width
    if r0 >= r1 or c0 >= c1:
        raise ValueError("guide width too large for object")

    horizontal = None
    if has_top:
        horizontal = (slice(0, guide_w), slice(c0, c1))
    if has_bottom:
        horizontal = (slice(height - guide_w, height), slice(c0, c1))

    vertical = None
    if has_left:
        vertical = (slice(r0, r1), slice(0, guide_w))
    if has_right:
        vertical = (slice(r0, r1), slice(width - guide_w, width))

    if has_top and has_left:
        corner = (slice(0, guide_w), slice(0, guide_w))
    elif has_top and has_right:
        corner = (slice(0, guide_w), slice(width - guide_w, width))
    elif has_bottom and has_left:
        corner = (slice(height - guide_w, height), slice(0, guide_w))
    else:
        corner = (slice(height - guide_w, height), slice(width - guide_w, width))

    return (r0, r1, c0, c1), horizontal, vertical, corner


def _match_aligned_object(
    arr: np.ndarray, bg: int, guide_sides: dict[str, int], guide_w: int
) -> tuple[int, tuple[int, int, int, int]] | None:
    try:
        core_box, horizontal, vertical, corner = _guide_layout(arr.shape, guide_sides, guide_w)
    except ValueError:
        return None

    r0, r1, c0, c1 = core_box
    core = arr[r0:r1, c0:c1]
    if core.size == 0:
        return None
    if np.any(core == int(bg)):
        return None
    for ctrl_color in guide_sides.values():
        if np.any(core == int(ctrl_color)):
            return None

    if "top" in guide_sides:
        if not np.all(arr[horizontal] == int(guide_sides["top"])):
            return None
    if "bottom" in guide_sides:
        if not np.all(arr[horizontal] == int(guide_sides["bottom"])):
            return None
    if "left" in guide_sides:
        if not np.all(arr[vertical] == int(guide_sides["left"])):
            return None
    if "right" in guide_sides:
        if not np.all(arr[vertical] == int(guide_sides["right"])):
            return None
    if not np.all(arr[corner] == int(bg)):
        return None

    fill = _dominant_color(core)
    return fill, core_box


def _recolor_guides_to_fill(arr: np.ndarray, guide_sides: dict[str, int], guide_w: int, fill: int) -> np.ndarray:
    out = np.array(arr, copy=True)
    guide_w = int(guide_w)
    if "top" in guide_sides:
        out[:guide_w, :] = int(fill)
    if "bottom" in guide_sides:
        out[-guide_w:, :] = int(fill)
    if "left" in guide_sides:
        out[:, :guide_w] = int(fill)
    if "right" in guide_sides:
        out[:, -guide_w:] = int(fill)
    return out


def _crop_core(arr: np.ndarray, guide_sides: dict[str, int], guide_w: int) -> np.ndarray:
    core_box, _horizontal, _vertical, _corner = _guide_layout(arr.shape, guide_sides, guide_w)
    r0, r1, c0, c1 = core_box
    return arr[r0:r1, c0:c1]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Align the central rectangle to the frame controls via a dihedral transform.

    Combo rule:
    - Detect which frame sides are controlled by the two border colors.
    - Try each D4 transform of the interior object and each candidate guide width in {2, 3}.
    - The correct aligned object has two adjacent uniform guide strips of that width and a same-size
      background corner square where those strips would overlap.
    - Keep those strips and recolor them to the dominant fill of the core instead of cropping them.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    desired = {side: color for side, color in _detect_controls_frame(g, bg)}
    obj = _extract_center_object(g, bg)

    for k in range(4):
        base = np.rot90(obj, int(k))
        for flip in (False, True):
            transformed = np.fliplr(base) if flip else base
            for guide_w in GUIDE_WIDTHS:
                match = _match_aligned_object(transformed, bg, desired, guide_w)
                if match is None:
                    continue
                fill, _core_box = match
                return _recolor_guides_to_fill(transformed, desired, guide_w, fill)

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


def _attach_guides(
    core: np.ndarray, guide_sides: tuple[str, str], guide_colors: dict[str, int], guide_w: int
) -> np.ndarray:
    height, width = core.shape
    has_top = "top" in guide_sides
    has_bottom = "bottom" in guide_sides
    has_left = "left" in guide_sides
    has_right = "right" in guide_sides
    out = np.zeros(
        (height + (guide_w if has_top or has_bottom else 0), width + (guide_w if has_left or has_right else 0)),
        dtype=int,
    )
    r0 = guide_w if has_top else 0
    c0 = guide_w if has_left else 0
    paste(out, core, r0, c0)

    if has_top:
        out[:guide_w, c0 : c0 + width] = int(guide_colors["top"])
    if has_bottom:
        out[-guide_w:, c0 : c0 + width] = int(guide_colors["bottom"])
    if has_left:
        out[r0 : r0 + height, :guide_w] = int(guide_colors["left"])
    if has_right:
        out[r0 : r0 + height, -guide_w:] = int(guide_colors["right"])
    return out


def _oracle_output(aligned_object: np.ndarray, guide_sides: tuple[str, str], guide_w: int) -> np.ndarray:
    desired = {side: 1 for side in guide_sides}
    fill = _dominant_color(_crop_core(aligned_object, desired, guide_w))
    return _recolor_guides_to_fill(aligned_object, desired, guide_w, fill)


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
    n = random.randint(24, 30)
    guide_w = random.choice(GUIDE_WIDTHS)
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

    max_core_h = min(18, n - guide_w - 2)
    max_core_w = min(18, n - guide_w - 2)
    min_core = 6
    if max_core_h < min_core or max_core_w < min_core:
        return None
    core_h = random.randint(min_core, max_core_h)
    core_w = random.randint(min_core, max_core_w)
    core = _build_core(core_h, core_w, fill, pattern_colors)

    guide_sides = (horiz_side, vert_side)
    guide_colors = {horiz_side: ctrl_h, vert_side: ctrl_v}
    aligned_object = _attach_guides(core, guide_sides, guide_colors, guide_w)
    expected = _oracle_output(aligned_object, guide_sides, guide_w)

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
    if result.shape != (core_h + guide_w, core_w + guide_w):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate input for fc7cae8d combo variant 14")
