# Combo variant of 89565ca0 (index 4)
# combo_spec: D3/0+D4/0
# Name: 实心矩形带洞+无噪声

from __future__ import annotations

import random

import numpy as np

from arc_utils import bbox, colors_present, crop_bbox, get_bg, label_mask


# ---------------------------------------------------------------------------
# Design analysis for the D3/0 + D4/0 combo
#
# a) Concrete behavior + trace
# - Original task objects were sparse 1px wall frames. The solver cropped each
#   color, reconstructed horizontal/vertical walls, then counted enclosed black
#   regions. Example: a 2x2 room frame with one divider produced 2 rooms.
# - In this combo, each object is instead a dense colored rectangle whose
#   interior contains several black rectangular holes. Example: a solid red
#   9x11 block with three separated black windows now maps to a red bar of
#   length 3. The counted primitive changes from "background room bounded by
#   walls" to "black hole carved inside a filled body".
# - D4/0 removes the old scattered pad/noise color entirely. So the output can
#   no longer be padded by an inferred non-black noise color; it must use the
#   visible background color, which remains black.
#
# b) Complexity calibration
# - The D3 object change is visually strong: thin line frames become opaque
#   blocks with cut-out holes. That difference is obvious across 2-4 examples.
# - To avoid replacing the original task with a trivial one-object counting
#   exercise, each puzzle still contains 3-5 objects and hole counts span 1-6.
# - Ties are intentionally avoided. Once noise and overlap-based ambiguity are
#   removed, equal counts would need an extra visible tie-break rule unrelated
#   to the requested combo. Distinct counts keep the rule single and legible.
#
# c) Implicit-assumption audit against the old main()
# - Old `_reconstruct_walls()` assumed sparse rectilinear evidence. For filled
#   rectangles, most rows/cols are dense foreground, so wall reconstruction
#   would misread the whole bbox as wall structure.
# - Old `_count_enclosed_regions(~walls)` assumed the signal was connected
#   background separated by colored lines. Here the useful structure is black
#   components inside a dense colored bbox, so counting must inspect background
#   holes directly.
# - Old `pad_colors == 1` depended on one extra non-black noise color with zero
#   rooms. D4/0 breaks that premise completely, so the new solver uses black as
#   padding and treats every non-black color as a real object color.
#
# d) Placement legality
# - The generator places rectangles with at least one cell of clearance, so
#   bboxes are disjoint, stay in-bounds, and remain recoverable from visible
#   pixels alone.
# - Holes are carved strictly inside each rectangle, never touching the outer
#   border and never touching each other, so each hole remains a distinct
#   4-connected black rectangle.
#
# e) Visual observability
# - A human can directly compare input and output and see that "more black holes
#   inside the colored block -> longer output bar". No hidden use of color ids,
#   noise frequency, or invisible metadata is required.
# - Black right-padding is also human-visible because it matches the canvas
#   background rather than an unseen reserved color.
#
# f) Visual cleanliness
# - No overlap and no noise keep the image uncluttered. Each object is a single
#   solid block, each hole is rectangular, and background remains empty black.
# - This matters here because overlapping filled rectangles would hide holes and
#   make the D3 substitution much harder to infer from few examples.
#
# g) Change inferability + new main() assumptions
# - Colors are preserved from input to output. The only transformation is:
#   object color -> bar color, hole count -> bar length, background -> padding.
# - New solver assumptions:
#   1. every non-black color corresponds to one rectangle-with-holes object;
#   2. each object's visible bbox has an intact colored outer border;
#   3. black components strictly inside that bbox are rectangular holes.
# - Generator guarantees those assumptions by constructing one object per color,
#   disallowing overlap, keeping margins around holes, and validating each
#   sample against `main()` before returning it.
# ---------------------------------------------------------------------------


# --------------- solver ---------------

_MIN_OBJECT_SIDE = 6
_MIN_VISIBLE_FILL_RATIO = 0.5


def _count_internal_rectangles(bg_mask: np.ndarray) -> int:
    labels, n_labels = label_mask(bg_mask, connectivity=4)
    h, w = bg_mask.shape
    holes = 0
    for k in range(1, n_labels + 1):
        comp = labels == k
        coords = np.argwhere(comp)
        assert coords.size > 0
        if (
            np.any(coords[:, 0] == 0)
            or np.any(coords[:, 0] == h - 1)
            or np.any(coords[:, 1] == 0)
            or np.any(coords[:, 1] == w - 1)
        ):
            continue
        b = bbox(comp)
        assert b is not None
        r0, c0, r1, c1 = b
        rect = comp[r0 : r1 + 1, c0 : c1 + 1]
        if bool(rect.all()) and int(rect.size) == int(comp.sum()):
            holes += 1
    return int(holes)


def _looks_like_solid_rectangle(mask: np.ndarray) -> bool:
    h, w = mask.shape
    if h < _MIN_OBJECT_SIDE or w < _MIN_OBJECT_SIDE:
        return False
    if not bool(mask[0, :].all() and mask[h - 1, :].all() and mask[:, 0].all() and mask[:, w - 1].all()):
        return False
    return float(mask.mean()) >= _MIN_VISIBLE_FILL_RATIO


def _object_stats(g: np.ndarray, color: int, bg: int) -> tuple[int, tuple[int, int, int, int]]:
    b = bbox(g == int(color))
    assert b is not None
    mask = crop_bbox(g == int(color), b).astype(bool)
    assert _looks_like_solid_rectangle(mask)
    bg_mask = crop_bbox(g == int(bg), b).astype(bool)
    holes = _count_internal_rectangles(bg_mask)
    assert holes >= 1
    return int(holes), b


def main(grid: np.ndarray) -> np.ndarray:
    """
    Convert each colored solid rectangle-with-holes into a horizontal bar whose length is its number of internal holes.

    Input structure:
    - Black background with several non-overlapping axis-aligned filled rectangles, each in its own non-black color.
    - Every real object is a dense colored rectangle whose interior contains one or more black rectangular holes.
    - There is no scattered noise color and no damaged object pixels.

    Rule:
    - For each non-black color, crop its visible bbox and confirm it behaves like a solid filled rectangle with an intact colored border.
    - Inside that bbox, count the 4-connected black components that stay away from the bbox edge and exactly fill their own rectangular bbox.
    - Sort object colors by that hole count in ascending order.
    - Output one left-aligned row per object, using the object's color repeated `count` times and black to pad to the maximum count.

    Key insight:
    - The meaningful topology now comes from black voids inside opaque colored blocks, not from reconstructing sparse wall grids; once the outer rectangle is identified, the answer is just the number of internal rectangular holes.
    """
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    assert bg == 0

    objects: list[tuple[int, int, tuple[int, int, int, int]]] = []
    for color in colors_present(g, exclude={bg}):
        holes, b = _object_stats(g, int(color), bg)
        objects.append((int(color), int(holes), b))

    assert objects
    ordered = sorted(objects, key=lambda item: (item[1], item[2][0], item[2][1]))
    width = max(holes for _color, holes, _b in ordered)

    out = np.full((len(ordered), width), bg, dtype=int)
    for r, (color, holes, _b) in enumerate(ordered):
        out[r, :holes] = int(color)
    return out


# --------------- generator ---------------

_RECT_COUNT_OPTIONS = (3, 4, 5)
_HOLE_COUNT_RANGE = (1, 6)
_STYLE_BY_COUNT = {
    5: {
        "grid_h": (29, 30),
        "grid_w": (29, 30),
        "obj_max_h": 12,
        "obj_max_w": 13,
        "outer_margin": (1, 2),
        "hole_size": (2, 2),
        "gutter": (1, 1),
    },
    4: {
        "grid_h": (27, 30),
        "grid_w": (27, 30),
        "obj_max_h": 14,
        "obj_max_w": 16,
        "outer_margin": (1, 2),
        "hole_size": (2, 3),
        "gutter": (1, 2),
    },
    3: {
        "grid_h": (23, 29),
        "grid_w": (24, 30),
        "obj_max_h": 16,
        "obj_max_w": 19,
        "outer_margin": (2, 3),
        "hole_size": (2, 3),
        "gutter": (1, 2),
    },
}
_PLACEMENT_GAP = 1
_PLACEMENT_RETRY = 500
_GEN_RETRIES = 400
_COUNT_TARGET_RETRIES = 220
_RECT_COUNT_CURSOR = 0


def _hole_layouts(target_holes: int) -> list[list[int]]:
    layouts = {
        1: [[1]],
        2: [[2], [1, 1]],
        3: [[1, 2], [2, 1]],
        4: [[2, 2], [1, 2, 1]],
        5: [[2, 3], [3, 2], [2, 2, 1]],
        6: [[2, 2, 2], [3, 3]],
    }
    return layouts[int(target_holes)]


def _build_rectangle_with_holes(
    target_holes: int,
    max_h: int,
    max_w: int,
    *,
    outer_margin_range: tuple[int, int],
    hole_size_range: tuple[int, int],
    gutter_range: tuple[int, int],
) -> np.ndarray | None:
    for _ in range(100):
        layout = random.choice(_hole_layouts(target_holes))
        top = random.randint(*outer_margin_range)
        bottom = random.randint(*outer_margin_range)
        left = random.randint(*outer_margin_range)
        right = random.randint(*outer_margin_range)

        row_gap_count = max(0, len(layout) - 1)
        row_gaps = [random.randint(*gutter_range) for _ in range(row_gap_count)]
        row_specs: list[tuple[list[int], list[int], int, int]] = []
        inner_w = 0
        inner_h = 0
        for holes_in_row in layout:
            hole_h = random.randint(*hole_size_range)
            widths = [random.randint(*hole_size_range) for _ in range(holes_in_row)]
            gaps = [random.randint(*gutter_range) for _ in range(max(0, holes_in_row - 1))]
            row_w = sum(widths) + sum(gaps)
            row_specs.append((widths, gaps, hole_h, row_w))
            inner_w = max(inner_w, row_w)
            inner_h += hole_h
        inner_h += sum(row_gaps)

        h = top + inner_h + bottom
        w = left + inner_w + right
        if h > max_h or w > max_w:
            continue

        body = np.ones((h, w), dtype=bool)
        r = top
        for row_idx, (widths, gaps, hole_h, row_w) in enumerate(row_specs):
            slack = inner_w - row_w
            row_left = left + (0 if slack == 0 else random.randint(0, slack))
            c = row_left
            for hole_idx, hole_w in enumerate(widths):
                body[r : r + hole_h, c : c + hole_w] = False
                c += hole_w
                if hole_idx < len(gaps):
                    c += gaps[hole_idx]
            if row_idx < len(row_gaps):
                r += hole_h + row_gaps[row_idx]
            else:
                r += hole_h

        if not _looks_like_solid_rectangle(body):
            continue
        if _count_internal_rectangles(~body) != target_holes:
            continue
        return body
    return None


def _rectangles_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (
        ar1 + gap < br0
        or br1 + gap < ar0
        or ac1 + gap < bc0
        or bc1 + gap < ac0
    )


def _place_rectangles(rects: list[np.ndarray], grid_h: int, grid_w: int) -> list[tuple[int, int]] | None:
    positions: list[tuple[int, int] | None] = [None] * len(rects)
    placed_boxes: list[tuple[int, int, int, int]] = []
    order = sorted(range(len(rects)), key=lambda i: -(rects[i].shape[0] * rects[i].shape[1]))

    for idx in order:
        h, w = rects[idx].shape
        placed = False
        for _ in range(_PLACEMENT_RETRY):
            r0 = random.randint(0, grid_h - h)
            c0 = random.randint(0, grid_w - w)
            box = (r0, c0, r0 + h - 1, c0 + w - 1)
            if any(_rectangles_overlap(box, other, _PLACEMENT_GAP) for other in placed_boxes):
                continue
            positions[idx] = (r0, c0)
            placed_boxes.append(box)
            placed = True
            break
        if not placed:
            return None

    assert all(pos is not None for pos in positions)
    return [(int(r), int(c)) for r, c in positions]


def _render_rectangles(
    rects: list[tuple[int, int, np.ndarray]],
    positions: list[tuple[int, int]],
    grid_h: int,
    grid_w: int,
) -> np.ndarray:
    grid = np.zeros((grid_h, grid_w), dtype=int)
    for (color, _holes, body), (r0, c0) in zip(rects, positions):
        rs, cs = np.where(body)
        grid[rs + r0, cs + c0] = int(color)
    return grid


def _next_rect_count() -> int:
    global _RECT_COUNT_CURSOR
    rect_count = int(_RECT_COUNT_OPTIONS[_RECT_COUNT_CURSOR % len(_RECT_COUNT_OPTIONS)])
    _RECT_COUNT_CURSOR += 1
    return rect_count


def _gen_attempt(rect_count: int | None = None) -> np.ndarray | None:
    if rect_count is None:
        rect_count = random.choice(_RECT_COUNT_OPTIONS)
    hole_counts = random.sample(list(range(_HOLE_COUNT_RANGE[0], _HOLE_COUNT_RANGE[1] + 1)), rect_count)
    colors = random.sample(list(range(1, 10)), rect_count)
    style = _STYLE_BY_COUNT[rect_count]
    h_min, h_max = style["grid_h"]
    w_min, w_max = style["grid_w"]
    obj_max_h = int(style["obj_max_h"])
    obj_max_w = int(style["obj_max_w"])

    rects: list[tuple[int, int, np.ndarray]] = []
    for color, target_holes in zip(colors, hole_counts):
        body = _build_rectangle_with_holes(
            target_holes,
            obj_max_h,
            obj_max_w,
            outer_margin_range=style["outer_margin"],
            hole_size_range=style["hole_size"],
            gutter_range=style["gutter"],
        )
        if body is None:
            return None
        rects.append((int(color), int(target_holes), body))

    max_obj_h = max(body.shape[0] for _color, _holes, body in rects)
    max_obj_w = max(body.shape[1] for _color, _holes, body in rects)
    h_lo = max(h_min, max_obj_h + 2)
    w_lo = max(w_min, max_obj_w + 2)
    if h_lo > h_max or w_lo > w_max:
        return None
    grid_h = random.randint(h_lo, h_max)
    grid_w = random.randint(w_lo, w_max)

    positions = _place_rectangles([body for _color, _holes, body in rects], grid_h, grid_w)
    if positions is None:
        return None

    grid = _render_rectangles(rects, positions, grid_h, grid_w)

    try:
        output = main(grid)
    except Exception:
        return None

    ordered = sorted(
        [(holes, pos[0], pos[1], color) for (color, holes, _body), pos in zip(rects, positions)]
    )
    width = max(hole_counts)
    expected = np.zeros((rect_count, width), dtype=int)
    for r, (holes, _r0, _c0, color) in enumerate(ordered):
        expected[r, :holes] = int(color)

    return grid if np.array_equal(output, expected) else None


def generate_input() -> np.ndarray:
    preferred_rect_count = _next_rect_count()
    for _ in range(_COUNT_TARGET_RETRIES):
        result = _gen_attempt(preferred_rect_count)
        if result is not None:
            return result
    for _ in range(max(0, _GEN_RETRIES - _COUNT_TARGET_RETRIES)):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {_GEN_RETRIES} attempts")
