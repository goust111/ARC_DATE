# Combo variant of fc7cae8d (index 41)
# combo_spec: D2/0+D4/0+D8/1
# Name: 可变宽+重着色+双层裁剪

from __future__ import annotations

from collections import Counter
from pathlib import Path
import random
import sys

import numpy as np

ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/arc2_solver_select")
if str(ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(ARC_UTILS_DIR))

from arc_utils import crop_bbox, find_components, get_bg, paste, random_free_location


"""
Design analysis for the combo variant
=====================================

a) Concrete behavior and trace
- Baseline fc7cae8d has one guide layer of fixed width 1. The solver finds the D4 transform whose two guide colors
  match the frame controls, then crops those guides away and returns the bare core rectangle.
- This combo replaces that object by a nested two-layer staircase-L structure on the same adjacent sides:
  an outer guide of width w1 in the frame-control colors and an inner guide of width w2 in two different colors.
  Both widths vary independently in {2,3}.
- Small aligned example for target sides top+left, with w1=2, w2=3, fill=4:
    rows 0..1:      bg-corner 2x2, then outer-top color across the rest
    rows 2..4:      outer-left color at cols 0..1, bg 3x3 inner-corner, then inner-top color
    rows 5..end:    outer-left color at cols 0..1, inner-left color at cols 2..4, then the multicolor core
- Old rule on that aligned object would try to remove a single 1px guide and would be structurally wrong.
- New rule:
  1. detect the correct D4 transform from the outer guide colors matching the frame controls;
  2. infer w1 and w2 from the nested layout;
  3. delete only the inner guide layer (compressing away those rows/cols);
  4. keep the outer layer, recolor it to the dominant core fill, and recolor its outer corner hole too.
- Output is therefore a rectangle larger than the bare core by w1 on the controlled sides, but smaller than the
  input object by the removed w2 layer.

b) Complexity calibration
- Both widths are chosen only from {2,3}; this keeps inference finite and repeated across samples instead of turning
  into unconstrained numeric guessing.
- The outer and inner layers use different colors and different actions. If they shared colors, the split between
  "keep/recolor" and "delete" would become visually ambiguous.
- The core remains an asymmetric patterned rectangle so the D4 alignment step is still meaningful and visible.

c) Implicit-assumption audit of the original main()
- Original `guide_edges(...)` assumed a single almost-solid 1px guide row/column on the object's bbox edge. This
  breaks because the new object has thick guides plus an extra inner hole, so edge rows contain larger structured gaps.
- Original `return t[top:bottom, left:right]` assumed removing one outer row/col solves the task. This breaks because
  the kept outer layer is now part of the answer while the removed layer is the inner one.
- Original code never had to identify guide width. Here width inference is essential; the solver now explicitly tests
  all `(w1,w2)` possibilities and validates the entire nested layout before producing an output.
- Original logic implicitly treated the output as a direct crop of the aligned object. That is no longer true: the
  result is a compressed recomposition after deleting interior guide bands.

d) Placement legality
- Generator builds the aligned object first, then applies an inverse D4 transform to get the input object.
- The object remains one 4-connected component because each guide branch touches the next inner branch and reaches the
  core, even though the outer and inner corners contain background holes.
- Placement uses `random_free_location(..., transparent=0, border=1)` so the object never touches the frame or leaves
  the canvas.

e) Visual observability
- Human solvers can see three stable cues from a few examples: the object has two nested guide layers, the outer layer
  uses the frame-control colors, and the output keeps one thick border while removing another.
- Width matters visibly because output size differs from the bare core by 2 or 3 cells, not a hidden single-pixel tweak.

f) Visual cleanliness
- There is only one interior object and its nested guides are all attached to the same rectangle, so the grid does not
  become cluttered with auxiliary legends or overlapping markers.
- The two background corner holes are the only interior empty regions; they clearly separate outer and inner layers.

g) Color-change inferability
- The recolor is not an arbitrary palette map. The only recolored cells are exactly the preserved outer guide layer,
  and they always become the dominant core fill. The frame-control colors therefore act as positional docking cues,
  not as output colors.

h) Quantitative anti-degeneration target
- For every generated sample, `w1 >= 2` and `w2 >= 2`, so it cannot collapse to the baseline 1px behavior.
- The aligned input object area exceeds output area by `w2 * (core_h + core_w + w1 + w2)` plus the deleted inner-corner
  square, while the output still exceeds the bare core by the preserved outer layer. This guarantees the combo's two
  key actions both materially affect the sample.

i) New assumptions introduced by the new solver, and how the generator guarantees them
- `_detect_controls_frame(...)` assumes exactly one horizontal and one vertical frame side are nearly solid controls.
  Generator preserves the old frame construction and keeps control colors distinct.
- `_parse_aligned_candidate(...)` assumes widths are in {2,3}, the outer layer uses the control colors, the inner layer
  is uniform on each side with colors distinct from bg and the controls, the two corner holes are background, and the
  core rectangle itself has no background holes. Generator constructs exactly that family.
- The transform search assumes only one candidate passes full structural validation. Distinct control colors plus the
  strongly constrained nested layout make accidental false positives extremely unlikely, and generator rejects any sample
  where `main(input)` does not equal the oracle output.
"""


GUIDE_WIDTHS = (2, 3)
FRAME_NOISE_COLOR = 5
ORIENTATIONS = [
    ("top", "left"),
    ("top", "right"),
    ("bottom", "left"),
    ("bottom", "right"),
]


def _transform(arr: np.ndarray, k: int, flip: bool) -> np.ndarray:
    out = np.rot90(arr, int(k))
    return np.fliplr(out) if flip else out


def _inverse_transform(arr: np.ndarray, k: int, flip: bool) -> np.ndarray:
    out = np.fliplr(arr) if flip else arr
    return np.rot90(out, -int(k))


def _dominant_color(arr: np.ndarray) -> int:
    vals = list(map(int, arr.flatten()))
    return int(Counter(vals).most_common(1)[0][0])


def _dihedral_signature(arr: np.ndarray) -> tuple[tuple[int, int], bytes]:
    return tuple(map(int, arr.shape)), arr.tobytes()


def _is_dihedrally_asymmetric(arr: np.ndarray) -> bool:
    seen = set()
    for k in range(4):
        base = np.rot90(arr, int(k))
        seen.add(_dihedral_signature(base))
        seen.add(_dihedral_signature(np.fliplr(base)))
    return len(seen) == 8


def _detect_controls_frame(arr: np.ndarray, bg: int) -> list[tuple[str, int]]:
    height, width = arr.shape
    sides = {
        "top": arr[0, 1 : width - 1],
        "bottom": arr[height - 1, 1 : width - 1],
        "left": arr[1 : height - 1, 0],
        "right": arr[1 : height - 1, width - 1],
    }
    candidates: list[tuple[str, int, int, int]] = []
    for side, values in sides.items():
        vals = list(map(int, values))
        non_bg = [x for x in vals if x != int(bg)]
        if not non_bg:
            continue
        color, count = Counter(non_bg).most_common(1)[0]
        if int(count) >= len(vals) - 1:
            candidates.append((side, int(color), int(count), int(len(vals))))
    candidates.sort(key=lambda t: (t[2] / t[3], t[2]), reverse=True)
    return [(side, color) for side, color, _count, _length in candidates[:2]]


def _extract_center_object(arr: np.ndarray, bg: int) -> np.ndarray:
    height, width = arr.shape
    comps = find_components(arr, bg=int(bg), connectivity=4, monochromatic=False, return_format="dict")
    best: tuple[int, tuple[int, int, int, int]] | None = None
    for comp in comps:
        r0, c0, r1, c1 = map(int, comp["bbox"])
        if r0 == 0 or c0 == 0 or r1 == height - 1 or c1 == width - 1:
            continue
        area = int(comp["area"])
        if best is None or area > best[0]:
            best = (area, (r0, c0, r1, c1))
    if best is None:
        return arr.copy()
    return crop_bbox(arr, best[1])


def _control_sides(ctrls: list[tuple[str, int]]) -> tuple[str, int, str, int] | None:
    horiz = [(side, color) for side, color in ctrls if side in ("top", "bottom")]
    vert = [(side, color) for side, color in ctrls if side in ("left", "right")]
    if len(horiz) != 1 or len(vert) != 1:
        return None
    return horiz[0][0], int(horiz[0][1]), vert[0][0], int(vert[0][1])


def _layout_slices(
    shape: tuple[int, int], horiz_side: str, vert_side: str, w1: int, w2: int
) -> dict[str, tuple[slice, slice]] | None:
    height, width = map(int, shape)
    if height <= w1 + w2 or width <= w1 + w2:
        return None

    if horiz_side == "top":
        outer_h_rows = slice(0, w1)
        inner_h_rows = slice(w1, w1 + w2)
        core_rows = slice(w1 + w2, height)
        outer_v_rows = slice(w1, height)
        inner_v_rows = slice(w1 + w2, height)
    else:
        outer_h_rows = slice(height - w1, height)
        inner_h_rows = slice(height - w1 - w2, height - w1)
        core_rows = slice(0, height - w1 - w2)
        outer_v_rows = slice(0, height - w1)
        inner_v_rows = slice(0, height - w1 - w2)

    if vert_side == "left":
        outer_v_cols = slice(0, w1)
        inner_v_cols = slice(w1, w1 + w2)
        core_cols = slice(w1 + w2, width)
        outer_h_cols = slice(w1, width)
        inner_h_cols = slice(w1 + w2, width)
    else:
        outer_v_cols = slice(width - w1, width)
        inner_v_cols = slice(width - w1 - w2, width - w1)
        core_cols = slice(0, width - w1 - w2)
        outer_h_cols = slice(0, width - w1)
        inner_h_cols = slice(0, width - w1 - w2)

    if core_rows.stop - core_rows.start < 4 or core_cols.stop - core_cols.start < 4:
        return None

    return {
        "outer_h": (outer_h_rows, outer_h_cols),
        "inner_h": (inner_h_rows, inner_h_cols),
        "outer_v": (outer_v_rows, outer_v_cols),
        "inner_v": (inner_v_rows, inner_v_cols),
        "outer_corner": (outer_h_rows, outer_v_cols),
        "inner_corner": (inner_h_rows, inner_v_cols),
        "core": (core_rows, core_cols),
    }


def _slice_view(arr: np.ndarray, region: tuple[slice, slice]) -> np.ndarray:
    rows, cols = region
    return arr[rows, cols]


def _uniform_color(arr: np.ndarray) -> int | None:
    vals = set(map(int, arr.flatten()))
    if len(vals) != 1:
        return None
    return int(next(iter(vals)))


def _compress_keep_outer(arr: np.ndarray, horiz_side: str, vert_side: str, w1: int, w2: int) -> np.ndarray:
    height, width = arr.shape
    if horiz_side == "top":
        keep_rows = list(range(0, w1)) + list(range(w1 + w2, height))
    else:
        keep_rows = list(range(0, height - w1 - w2)) + list(range(height - w1, height))
    if vert_side == "left":
        keep_cols = list(range(0, w1)) + list(range(w1 + w2, width))
    else:
        keep_cols = list(range(0, width - w1 - w2)) + list(range(width - w1, width))

    out = arr[np.ix_(keep_rows, keep_cols)].copy()
    return out


def _recolor_outer_to_fill(out: np.ndarray, horiz_side: str, vert_side: str, w1: int, fill: int) -> np.ndarray:
    arr = np.array(out, copy=True)
    if horiz_side == "top":
        arr[:w1, :] = int(fill)
    else:
        arr[-w1:, :] = int(fill)
    if vert_side == "left":
        arr[:, :w1] = int(fill)
    else:
        arr[:, -w1:] = int(fill)
    return arr


def _parse_aligned_candidate(
    arr: np.ndarray, bg: int, horiz_side: str, ctrl_h: int, vert_side: str, ctrl_v: int, w1: int, w2: int
) -> np.ndarray | None:
    layout = _layout_slices(arr.shape, horiz_side, vert_side, w1, w2)
    if layout is None:
        return None

    outer_h = _slice_view(arr, layout["outer_h"])
    inner_h = _slice_view(arr, layout["inner_h"])
    outer_v = _slice_view(arr, layout["outer_v"])
    inner_v = _slice_view(arr, layout["inner_v"])
    outer_corner = _slice_view(arr, layout["outer_corner"])
    inner_corner = _slice_view(arr, layout["inner_corner"])
    core = _slice_view(arr, layout["core"])

    if _uniform_color(outer_h) != int(ctrl_h):
        return None
    if _uniform_color(outer_v) != int(ctrl_v):
        return None
    if not np.all(outer_corner == int(bg)):
        return None
    if not np.all(inner_corner == int(bg)):
        return None
    if np.any(core == int(bg)):
        return None

    inner_h_color = _uniform_color(inner_h)
    inner_v_color = _uniform_color(inner_v)
    if inner_h_color is None or inner_v_color is None:
        return None
    if inner_h_color in {int(bg), int(ctrl_h), int(ctrl_v)}:
        return None
    if inner_v_color in {int(bg), int(ctrl_h), int(ctrl_v)}:
        return None
    if inner_h_color == inner_v_color:
        return None

    fill = _dominant_color(core)
    if fill in {int(bg), int(ctrl_h), int(ctrl_v), int(inner_h_color), int(inner_v_color)}:
        return None

    compressed = _compress_keep_outer(arr, horiz_side, vert_side, w1, w2)
    return _recolor_outer_to_fill(compressed, horiz_side, vert_side, w1, fill)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Find the D4 transform whose outer guide layer matches the frame controls.
    Keep the outer guide layer by recoloring it to the core fill, and delete the
    inner guide layer by compressing those rows/cols away.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    ctrls = _detect_controls_frame(g, bg)
    sides = _control_sides(ctrls)
    obj = _extract_center_object(g, bg)
    if sides is None:
        return obj

    horiz_side, ctrl_h, vert_side, ctrl_v = sides
    for k in range(4):
        base = np.rot90(obj, int(k))
        for flip in (False, True):
            transformed = np.fliplr(base) if flip else base
            for w1 in GUIDE_WIDTHS:
                for w2 in GUIDE_WIDTHS:
                    out = _parse_aligned_candidate(transformed, bg, horiz_side, ctrl_h, vert_side, ctrl_v, w1, w2)
                    if out is not None:
                        return out
    return obj


def _build_core(h: int, w: int, fill: int, pattern_colors: list[int]) -> np.ndarray:
    area = h * w
    max_nonfill = max(len(pattern_colors) + 5, int(area * 0.33))
    min_nonfill = max(len(pattern_colors) + 4, area // 9)

    for _ in range(200):
        core = np.full((h, w), int(fill), dtype=int)
        target_nonfill = random.randint(min_nonfill, max_nonfill)
        painted = 0

        for color in pattern_colors:
            r = random.randrange(h)
            c = random.randrange(w)
            if core[r, c] == int(fill):
                painted += 1
            core[r, c] = int(color)

        while painted < target_nonfill:
            color = int(random.choice(pattern_colors))
            kind = random.choice(["dot", "line", "rect", "el"])
            if kind == "dot":
                cells = [(random.randrange(h), random.randrange(w))]
            elif kind == "line":
                if random.random() < 0.5:
                    line_len = random.randint(2, min(5, w))
                    rr = random.randrange(h)
                    c0 = random.randrange(w - line_len + 1)
                    cells = [(rr, c0 + dc) for dc in range(line_len)]
                else:
                    line_len = random.randint(2, min(5, h))
                    r0 = random.randrange(h - line_len + 1)
                    cc = random.randrange(w)
                    cells = [(r0 + dr, cc) for dr in range(line_len)]
            elif kind == "rect":
                rh = random.randint(1, min(3, h))
                rw = random.randint(1, min(3, w))
                r0 = random.randrange(h - rh + 1)
                c0 = random.randrange(w - rw + 1)
                cells = [(r0 + dr, c0 + dc) for dr in range(rh) for dc in range(rw)]
            else:
                r0 = random.randrange(max(1, h - 2))
                c0 = random.randrange(max(1, w - 2))
                ah = random.randint(2, min(4, h - r0))
                aw = random.randint(2, min(4, w - c0))
                cells = [(r0 + dr, c0) for dr in range(ah)] + [(r0, c0 + dc) for dc in range(aw)]

            for r, c in cells:
                if core[r, c] == int(fill):
                    painted += 1
                core[r, c] = int(color)
                if painted >= target_nonfill:
                    break

        # Force a small asymmetric signature near one corner.
        marker_color = int(random.choice(pattern_colors))
        corner = random.choice(
            [
                (0, 0),
                (0, max(0, w - 2)),
                (max(0, h - 2), 0),
                (max(0, h - 2), max(0, w - 2)),
            ]
        )
        mr, mc = corner
        for rr, cc in [
            (mr, mc),
            (min(h - 1, mr + 1), mc),
            (mr, min(w - 1, mc + 1)),
        ]:
            core[rr, cc] = marker_color

        if _is_dihedrally_asymmetric(core):
            return core

    raise RuntimeError("failed to build asymmetric core")


def _attach_nested_guides(
    core: np.ndarray,
    horiz_side: str,
    vert_side: str,
    ctrl_h: int,
    ctrl_v: int,
    inner_h_color: int,
    inner_v_color: int,
    w1: int,
    w2: int,
    bg: int = 0,
) -> np.ndarray:
    core_h, core_w = core.shape
    height = core_h + w1 + w2
    width = core_w + w1 + w2
    out = np.full((height, width), int(bg), dtype=int)
    layout = _layout_slices(out.shape, horiz_side, vert_side, w1, w2)
    if layout is None:
        raise ValueError("invalid nested guide layout")

    paste(out, core, layout["core"][0].start, layout["core"][1].start)
    out[layout["outer_h"]] = int(ctrl_h)
    out[layout["outer_v"]] = int(ctrl_v)
    out[layout["inner_h"]] = int(inner_h_color)
    out[layout["inner_v"]] = int(inner_v_color)
    out[layout["outer_corner"]] = int(bg)
    out[layout["inner_corner"]] = int(bg)
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
            grid[r, c] = int(FRAME_NOISE_COLOR)


def _oracle_output(aligned_object: np.ndarray, horiz_side: str, vert_side: str, w1: int, w2: int) -> np.ndarray:
    layout = _layout_slices(aligned_object.shape, horiz_side, vert_side, w1, w2)
    assert layout is not None
    core = _slice_view(aligned_object, layout["core"])
    fill = _dominant_color(core)
    compressed = _compress_keep_outer(aligned_object, horiz_side, vert_side, w1, w2)
    return _recolor_outer_to_fill(compressed, horiz_side, vert_side, w1, fill)


def _gen_attempt() -> np.ndarray | None:
    n = random.randint(26, 32)
    horiz_side, vert_side = random.choice(ORIENTATIONS)
    w1 = random.choice(GUIDE_WIDTHS)
    w2 = random.choice(GUIDE_WIDTHS)

    all_colors = list(range(1, 10))
    ctrl_h, ctrl_v = random.sample([c for c in all_colors if c != FRAME_NOISE_COLOR], 2)
    remaining = [c for c in all_colors if c not in {FRAME_NOISE_COLOR, ctrl_h, ctrl_v}]
    fill = random.choice(remaining)
    remaining = [c for c in remaining if c != fill]
    inner_h_color, inner_v_color = random.sample(remaining, 2)
    remaining = [c for c in remaining if c not in {inner_h_color, inner_v_color}]
    pattern_count = random.randint(2, min(4, len(remaining)))
    pattern_colors = random.sample(remaining, pattern_count)

    max_core_h = min(15, n - (w1 + w2) - 4)
    max_core_w = min(15, n - (w1 + w2) - 4)
    if max_core_h < 5 or max_core_w < 5:
        return None
    core_h = random.randint(5, max_core_h)
    core_w = random.randint(5, max_core_w)
    core = _build_core(core_h, core_w, fill, pattern_colors)

    aligned_object = _attach_nested_guides(
        core, horiz_side, vert_side, ctrl_h, ctrl_v, inner_h_color, inner_v_color, w1, w2
    )
    expected = _oracle_output(aligned_object, horiz_side, vert_side, w1, w2)

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
    for _ in range(400):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate input for fc7cae8d combo variant 41")
