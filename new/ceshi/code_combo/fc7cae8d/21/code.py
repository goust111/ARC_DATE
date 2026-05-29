# Combo variant of fc7cae8d (index 21)
# combo_spec: D3/2+D4/0
# Name: 三控制边+保留重着色

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc2_solver_select.arc_utils import crop_bbox, find_components, get_bg, paste, random_free_location


"""
Design analysis for combo variant D3/2 + D4/0
=============================================

This combo changes two parts of the baseline task at the same time:

- D3/2: the frame now has three solid control sides instead of one horizontal plus one vertical side.
- D4/0: after the correct dihedral alignment is found, the guide edges are kept and recolored to the body fill instead
  of being cropped away.

1) Concrete behavior trace
- Baseline aligned object with two guides, for example top=7 and left=3:
    0 7 7 7
    3 4 4 9
    3 4 8 4
  Baseline output crops the guides and returns:
    4 4 9
    4 8 4

- Combo aligned object with three guides, for example missing only the bottom side:
    0 7 7 7 0
    3 4 4 9 6
    3 4 8 4 6
  Here top=7, left=3, right=6, and bottom has no guide.

- Old post-alignment logic would crop every guided side and return the 2x3 core:
    4 4 9
    4 8 4

- New combo logic keeps those three guide sides and recolors them to the dominant fill 4:
    4 4 4 4 4
    4 4 4 9 4
    4 4 8 4 4
  So the output is larger than the core on exactly three directions and unchanged on the missing side.

2) Complexity calibration
- Three distinct control colors constrain orientation more strongly than the baseline, so the task still requires a
  true D4 search instead of a local crop heuristic.
- Keeping three recolored guide sides creates a visibly asymmetric output shape. The missing side is now detectable
  from output size alone, which makes the combo behavior visually different from both the baseline and the D4-only
  two-side recolor variant.
- The core keeps multiple interior pattern colors, but the outer border of the core stays fill-colored so the missing
  side does not accidentally masquerade as a guide.

3) Hidden-assumption audit against the original solver
- `detect_controls_frame(...)` in the original code assumed exactly two solid sides were relevant. That is invalid now.
  The combo solver must collect all solid frame sides and expect three of them.
- `guide_edges(...)` in the original code accepted at most one mismatching corner cell. That fails for a three-guide
  side such as top when both top-left and top-right corners are background. The combo version allows up to two corner
  mismatches on a detected guide side.
- The original crop slice assumed every matched control side should be deleted. D4/0 reverses that rule, so the new
  post-alignment step must recolor those sides instead.
- The original code did not care whether the un-guided side of the object had decorative colors touching the border.
  With three guides that would risk false-positive side detection, so the generator now guarantees that the core border
  itself stays fill-colored and only the added guide strips carry control colors.

4) Placement legality
- The generator constructs an aligned object first, computes its oracle output directly from that aligned object, then
  applies an inverse D4 transform to get the input object.
- The frame is painted before placement, and `random_free_location(..., border=1)` keeps the object away from the
  frame, so frame controls and object guides never overlap.
- Object sizes are chosen conservatively so the transformed object still fits with a one-cell moat inside the canvas.

5) Visual observability
- Across samples, any one of the four sides may be the single missing side. Humans can see that the output always
  keeps three extra 1px extensions and omits one side.
- The three control colors are distinct and repeated along full frame sides, giving a clear cue that the three matching
  guide sides on the internal object determine the orientation.

6) Visual cleanliness
- There is only one internal object, one outer frame, and one sparse noise side. No region mixes multiple annotation
  systems on top of each other.
- Sparse noise is confined to the lone non-control border side, so it is visually separable from the three solid
  controls.

7) Color-change inferability
- Only guide pixels change color, and they all change to the object's dominant fill color after alignment.
- The new color is not arbitrary: it is the already-visible fill color occupying most of the rectangle.

8) Anti-degeneracy target
- A valid generated sample must have exactly three detected frame controls and exactly three detected object guides in
  the matched orientation.
- The combo output must differ from the old crop result by retaining three 1px side extensions; if those extensions
  disappeared, the sample would collapse back toward the baseline task.

9) New solver assumptions and generator guarantees
- Assumption: the largest non-border component is the target object. Guarantee: generator places exactly one such
  object away from the frame.
- Assumption: fill is the dominant object color. Guarantee: the core remains mostly fill-colored and guide colors are
  each limited to one side strip.
- Assumption: the missing side of the aligned object is truly unguided. Guarantee: the core border is left as fill so
  `_guide_edges(...)` does not misread pattern colors as a guide.
"""


SIDES = ("top", "bottom", "left", "right")


def _transform(arr: np.ndarray, k: int, flip: bool) -> np.ndarray:
    out = np.rot90(arr, int(k))
    return np.fliplr(out) if flip else out


def _inverse_transform(arr: np.ndarray, k: int, flip: bool) -> np.ndarray:
    out = np.fliplr(arr) if flip else arr
    return np.rot90(out, -int(k))


def _dominant_fill(arr: np.ndarray) -> int:
    return int(Counter(map(int, arr.flatten())).most_common(1)[0][0])


def _detect_controls_frame(arr: np.ndarray, bg: int) -> dict[str, int]:
    sides = {
        "top": arr[0, 1:-1],
        "bottom": arr[-1, 1:-1],
        "left": arr[1:-1, 0],
        "right": arr[1:-1, -1],
    }
    found: dict[str, int] = {}
    for side, values in sides.items():
        vals = list(map(int, values))
        non_bg = [x for x in vals if x != int(bg)]
        if not non_bg:
            continue
        color, count = Counter(non_bg).most_common(1)[0]
        if int(count) >= len(vals) - 1:
            found[side] = int(color)
    return found


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
        return np.array(arr, copy=True)
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
        if int(count) >= len(vals) - 2:
            out[side] = int(color)
    return out


def _match_transform(obj: np.ndarray, desired: dict[str, int]) -> np.ndarray | None:
    if len(desired) != 3:
        return None
    for k in range(4):
        base = np.rot90(obj, int(k))
        for flip in (False, True):
            transformed = np.fliplr(base) if flip else base
            guides = _guide_edges(transformed)
            if len(guides) != len(desired):
                continue
            if all(side in guides and int(guides[side]) == int(color) for side, color in desired.items()):
                return transformed
    return None


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
    Align the central object to three frame-side controls via a dihedral transform.

    Combo rule:
    - The frame has exactly three solid control sides, each with its own color.
    - The object has three matching guide sides and one missing side.
    - After alignment, keep those three guide strips and recolor them to the object's dominant fill color.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    desired = _detect_controls_frame(g, bg)
    obj = _extract_center_object(g, bg)
    if len(desired) != 3:
        return obj
    matched = _match_transform(obj, desired)
    if matched is None:
        return obj
    return _recolor_guides_to_fill(matched, desired)


def _build_core(h: int, w: int, fill: int, pattern_colors: list[int]) -> np.ndarray:
    core = np.full((h, w), int(fill), dtype=int)
    interior_h = h - 2
    interior_w = w - 2
    interior_area = interior_h * interior_w
    max_nonfill = max(len(pattern_colors), int(interior_area * 0.40))
    target_nonfill = random.randint(max(len(pattern_colors), interior_area // 8), max_nonfill)
    painted = 0

    for color in pattern_colors:
        r = random.randrange(1, h - 1)
        c = random.randrange(1, w - 1)
        if core[r, c] == fill:
            core[r, c] = int(color)
            painted += 1

    while painted < target_nonfill:
        color = random.choice(pattern_colors)
        shape_kind = random.choice(("dot", "line", "rect"))
        if shape_kind == "dot":
            cells = [(random.randrange(1, h - 1), random.randrange(1, w - 1))]
        elif shape_kind == "line":
            if random.random() < 0.5:
                line_len = random.randint(2, min(5, interior_w))
                rr = random.randrange(1, h - 1)
                c0 = random.randrange(1, w - line_len)
                cells = [(rr, c0 + dc) for dc in range(line_len)]
            else:
                line_len = random.randint(2, min(5, interior_h))
                r0 = random.randrange(1, h - line_len)
                cc = random.randrange(1, w - 1)
                cells = [(r0 + dr, cc) for dr in range(line_len)]
        else:
            mh = random.randint(1, min(3, interior_h))
            mw = random.randint(1, min(3, interior_w))
            r0 = random.randrange(1, h - mh)
            c0 = random.randrange(1, w - mw)
            cells = [(r0 + dr, c0 + dc) for dr in range(mh) for dc in range(mw)]

        remaining = target_nonfill - painted
        if remaining <= 0:
            break
        fresh = [(r, c) for r, c in cells if core[r, c] == fill]
        if len(fresh) > remaining:
            fresh = random.sample(fresh, remaining)
        for r, c in fresh:
            core[r, c] = int(color)
        painted += len(fresh)

    return core


def _attach_guides(core: np.ndarray, guide_sides: tuple[str, ...], guide_colors: dict[str, int]) -> np.ndarray:
    guide_set = set(guide_sides)
    h, w = core.shape
    has_top = "top" in guide_set
    has_bottom = "bottom" in guide_set
    has_left = "left" in guide_set
    has_right = "right" in guide_set
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


def _oracle_output(aligned_object: np.ndarray, guide_sides: tuple[str, ...]) -> np.ndarray:
    desired = {side: 1 for side in guide_sides}
    return _recolor_guides_to_fill(aligned_object, desired)


def _paint_frame(grid: np.ndarray, control_colors: dict[str, int], noise_color: int = 5) -> None:
    n = grid.shape[0]
    for side, color in control_colors.items():
        if side == "top":
            grid[0, :] = int(color)
        elif side == "bottom":
            grid[-1, :] = int(color)
        elif side == "left":
            grid[:, 0] = int(color)
        else:
            grid[:, -1] = int(color)

    missing_side = next(side for side in SIDES if side not in control_colors)
    if missing_side in ("top", "bottom"):
        r = 0 if missing_side == "top" else n - 1
        coords = [(r, c) for c in range(1, n - 1)]
    else:
        c = 0 if missing_side == "left" else n - 1
        coords = [(r, c) for r in range(1, n - 1)]

    min_count = int(np.ceil(0.35 * len(coords)))
    max_count = int(np.floor(0.60 * len(coords)))
    noise_count = random.randint(min_count, max_count)
    for r, c in random.sample(coords, noise_count):
        grid[r, c] = int(noise_color)


def _gen_attempt() -> np.ndarray | None:
    n = random.randint(24, 30)
    missing_side = random.choice(SIDES)
    guide_sides = tuple(side for side in SIDES if side != missing_side)

    all_colors = list(range(1, 10))
    control_palette = [c for c in all_colors if c != 5]
    control_values = random.sample(control_palette, 3)
    control_colors = {side: color for side, color in zip(guide_sides, control_values)}

    remaining = [c for c in all_colors if c not in set(control_values) | {5}]
    fill = random.choice(remaining)
    remaining = [c for c in remaining if c != fill]
    n_pattern_colors = random.randint(2, min(4, len(remaining)))
    pattern_colors = random.sample(remaining, n_pattern_colors)

    max_core_h = min(18, n - 6)
    max_core_w = min(18, n - 6)
    if max_core_h < 6 or max_core_w < 6:
        return None
    core_h = random.randint(6, max_core_h)
    core_w = random.randint(6, max_core_w)
    core = _build_core(core_h, core_w, fill, pattern_colors)

    aligned_object = _attach_guides(core, guide_sides, control_colors)
    expected = _oracle_output(aligned_object, guide_sides)

    k = random.randrange(4)
    flip = bool(random.randrange(2))
    input_object = _inverse_transform(aligned_object, k, flip)

    grid = np.zeros((n, n), dtype=int)
    _paint_frame(grid, control_colors)

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
    for _ in range(400):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate input for fc7cae8d combo variant 21")
