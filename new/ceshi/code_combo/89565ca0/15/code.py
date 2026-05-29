# Combo variant of 89565ca0 (index 15)
# combo_spec: D3/0+D4/0+D9/0
# Name: 实心矩形带洞+无噪声+无重叠

from __future__ import annotations

import random

import numpy as np

from arc_utils import bbox, colors_present, crop_bbox, get_bg, label_mask


# ---------------------------------------------------------------------------
# Design analysis for combo D3/0 + D4/0 + D9/0
#
# a) Concrete behavior + trace
# - Original behavior: each non-black color was a thin rectilinear wall network.
#   The solver first reconstructed damaged walls from sparse evidence, then
#   counted enclosed background rooms, then used overlap order to break ties.
# - Combo behavior: each non-black color is now exactly one solid axis-aligned
#   rectangle whose interior contains 1-6 black rectangular holes. Example:
#   an 11x12 green frame with one divider originally implied 2 rooms after wall
#   reconstruction; the combo version instead shows an 11x12 solid green block
#   with two separate black windows, so the count is read directly as 2 holes.
# - Difference: the counted primitive changes from "background chambers bounded
#   by walls" to "background holes carved out of a filled rectangle". Noise and
#   overlap are both removed, so the visible topology alone determines output.
#
# b) Complexity calibration
# - Removing both noise and overlap simplifies the original task a lot, so the
#   replacement object must stay visually distinctive. Dense rectangles with
#   multiple interior holes are easy to tell apart from line frames at a glance.
# - To avoid underconstrained samples, puzzles still contain 3-5 objects and
#   hole counts span 1-6. One tied pair is allowed in some samples so the new
#   visible tiebreak rule is exercised without making every example ambiguous.
#
# c) Implicit-assumption audit against the old main()
# - Old bbox->wall reconstruction assumed each object color formed sparse wall
#   evidence. That fails here because a filled rectangle makes almost every row
#   and column dense, so reconstruction would collapse the whole bbox.
# - Old "rooms == enclosed background after removing walls" no longer matches
#   the topology. We must instead count black connected components strictly
#   inside each colored bbox and verify they are rectangular holes.
# - Old pad/noise-color detection assumed one non-object color existed. D4/0
#   removes that color entirely, so every non-black color must be treated as a
#   real object and output padding must revert to the black background.
# - Old equal-count ordering depended on overlaps. D9/0 removes overlaps, so
#   ties are now resolved by a visible row-major bbox order: upper objects
#   first, and for the same row, left objects first.
#
# d) Placement legality
# - Generator places each rectangle using its full bbox and requires at least a
#   one-cell clearance between bboxes. This guarantees no overlap, no touching,
#   no clipping, and a recoverable bbox for every color.
# - Holes are carved strictly inside the rectangle with positive margins, so no
#   hole can leak to the boundary and be mistaken for exterior background.
#
# e) Visual observability
# - From 2-4 ARC examples, a human can see that the inputs are solid colored
#   blocks with black cut-outs and that output row lengths match hole counts.
# - The tie rule is also visible because it depends only on object position, not
#   on hidden metadata like color id or generation order.
#
# f) Visual cleanliness
# - There is no noise layer and no stacking. Each region of the grid belongs to
#   background, one rectangle body, or one hole, so objects stay easy to parse.
# - Hole sizes and outer margins are kept above 1 cell, which prevents tiny
#   specks from looking accidental.
#
# g) Change inferability
# - Colors are preserved from input to output with no remapping. The only color
#   absent from output bars is black, which now serves as ordinary padding.
# - New solver assumptions:
#   1. each non-black color corresponds to exactly one rectangle bbox;
#   2. inside that bbox only {background, object_color} appear;
#   3. every counted black component is rectangular and interior.
#   The generator enforces all three properties explicitly.
# ---------------------------------------------------------------------------


# --------------- solver ---------------

_MIN_OBJECT_SIDE = 6


def _count_internal_rectangles(bg_mask: np.ndarray) -> int:
    labels, n_labels = label_mask(bg_mask, connectivity=4)
    h, w = bg_mask.shape
    holes = 0
    for k in range(1, n_labels + 1):
        comp = labels == k
        coords = np.argwhere(comp)
        assert coords.size > 0
        touches_edge = (
            np.any(coords[:, 0] == 0)
            or np.any(coords[:, 0] == h - 1)
            or np.any(coords[:, 1] == 0)
            or np.any(coords[:, 1] == w - 1)
        )
        assert not touches_edge
        b = bbox(comp)
        assert b is not None
        r0, c0, r1, c1 = b
        rect = comp[r0 : r1 + 1, c0 : c1 + 1]
        assert bool(rect.all()) and int(rect.size) == int(comp.sum())
        holes += 1
    return int(holes)


def _looks_like_solid_rectangle(mask: np.ndarray) -> bool:
    h, w = mask.shape
    if h < _MIN_OBJECT_SIDE or w < _MIN_OBJECT_SIDE:
        return False
    return bool(mask[0, :].all() and mask[h - 1, :].all() and mask[:, 0].all() and mask[:, w - 1].all())


def _object_stats(g: np.ndarray, color: int, bg: int) -> tuple[int, tuple[int, int, int, int]]:
    b = bbox(g == int(color))
    assert b is not None
    crop = crop_bbox(g, b)
    color_mask = crop == int(color)
    if not _looks_like_solid_rectangle(color_mask):
        raise AssertionError("object is not a solid bordered rectangle")
    assert set(int(v) for v in np.unique(crop)) <= {int(bg), int(color)}
    holes = _count_internal_rectangles(crop == int(bg))
    assert holes > 0
    return int(holes), b


def main(grid: np.ndarray) -> np.ndarray:
    """
    Convert each colored solid rectangle-with-holes into a horizontal bar whose length is its number of internal holes.

    Input structure:
    - Black background with several non-overlapping axis-aligned filled rectangles in distinct non-black colors.
    - Each real object contains one or more black rectangular holes strictly inside its colored bbox.
    - There is no extra noise color and no overlap layer; every non-black color belongs to one visible object.

    Rule:
    - For each non-background color, take its visible bbox and confirm that it forms a filled rectangle with an intact colored border.
    - Inside that bbox, count the 4-connected black components that do not touch the bbox edge and are perfect rectangles.
    - Treat that hole count as the object's score.
    - Sort objects by increasing hole count; ties are broken by row-major bbox position, then write one left-aligned row per object padded with black.

    Key insight:
    - Once noise and overlap disappear, the problem reduces to counting interior voids inside solid colored blocks rather than reconstructing damaged wall graphs.
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

_RECT_COUNT_RANGE = (3, 5)
_HOLE_COUNT_RANGE = (1, 6)
_TIE_PROBABILITY = 0.5
_GRID_SIZE_BY_COUNT = {
    5: (28, 36, 28, 36, 15, 16),
    4: (24, 34, 24, 34, 15, 17),
    3: (20, 32, 20, 32, 16, 18),
}
_OUTER_MARGIN_RANGE = (1, 2)
_HOLE_SIZE_RANGE = (2, 3)
_GUTTER_RANGE = (1, 2)
_PLACEMENT_GAP = 1
_PLACEMENT_RETRY = 800
_GEN_RETRIES = 500
_CANVAS_MARGIN = 1
_ROW_COUNT_BY_RECTS = {
    3: (1, 3),
    4: (2, 3),
    5: (2, 4),
}


def _hole_layouts(target_holes: int) -> list[list[int]]:
    layouts = {
        1: [[1]],
        2: [[2], [1, 1]],
        3: [[1, 2], [2, 1]],
        4: [[2, 2], [1, 2, 1], [2, 1, 1]],
        5: [[2, 3], [3, 2], [2, 2, 1], [2, 1, 2], [1, 2, 2]],
        6: [[2, 2, 2], [3, 3], [2, 2, 1, 1], [2, 1, 2, 1]],
    }
    return layouts[int(target_holes)]


def _build_rectangle_with_holes(target_holes: int, max_h: int, max_w: int) -> np.ndarray | None:
    for _ in range(100):
        layout = random.choice(_hole_layouts(target_holes))
        top = random.randint(*_OUTER_MARGIN_RANGE)
        bottom = random.randint(*_OUTER_MARGIN_RANGE)
        left = random.randint(*_OUTER_MARGIN_RANGE)
        right = random.randint(*_OUTER_MARGIN_RANGE)

        row_gap_count = max(0, len(layout) - 1)
        row_gaps = [random.randint(*_GUTTER_RANGE) for _ in range(row_gap_count)]
        row_specs: list[tuple[list[int], list[int], int, int]] = []
        inner_w = 0
        inner_h = 0
        for holes_in_row in layout:
            hole_h = random.randint(*_HOLE_SIZE_RANGE)
            widths = [random.randint(*_HOLE_SIZE_RANGE) for _ in range(holes_in_row)]
            gaps = [random.randint(*_GUTTER_RANGE) for _ in range(max(0, holes_in_row - 1))]
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
            r += hole_h
            if row_idx < len(row_gaps):
                r += row_gaps[row_idx]

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


def _row_partition(n_items: int, row_count: int) -> list[int]:
    counts = [1] * row_count
    for _ in range(n_items - row_count):
        counts[random.randrange(row_count)] += 1
    random.shuffle(counts)
    return counts


def _place_rectangles(
    rects: list[np.ndarray],
    h_min: int,
    h_max: int,
    w_min: int,
    w_max: int,
) -> tuple[int, int, list[tuple[int, int]]] | None:
    if not rects:
        return h_min, w_min, []

    for _ in range(_PLACEMENT_RETRY):
        row_lo, row_hi = _ROW_COUNT_BY_RECTS[len(rects)]
        row_count = random.randint(row_lo, row_hi)
        counts = _row_partition(len(rects), row_count)

        order = list(range(len(rects)))
        random.shuffle(order)
        rows: list[list[int]] = []
        offset = 0
        for count in counts:
            row = order[offset : offset + count]
            offset += count
            row.sort(key=lambda idx: rects[idx].shape[1], reverse=True)
            rows.append(row)

        intra_gaps = [
            [random.randint(_PLACEMENT_GAP, _GUTTER_RANGE[1]) for _ in range(max(0, len(row) - 1))]
            for row in rows
        ]
        row_heights = [max(rects[idx].shape[0] for idx in row) for row in rows]
        row_widths = [
            sum(rects[idx].shape[1] for idx in row) + sum(gaps)
            for row, gaps in zip(rows, intra_gaps)
        ]
        inter_row_gaps = [random.randint(_PLACEMENT_GAP, _GUTTER_RANGE[1]) for _ in range(max(0, row_count - 1))]

        pack_h = sum(row_heights) + sum(inter_row_gaps)
        pack_w = max(row_widths)
        grid_h_lo = max(h_min, pack_h + 2 * _CANVAS_MARGIN)
        grid_w_lo = max(w_min, pack_w + 2 * _CANVAS_MARGIN)
        if grid_h_lo > h_max or grid_w_lo > w_max:
            continue

        grid_h = random.randint(grid_h_lo, h_max)
        grid_w = random.randint(grid_w_lo, w_max)
        start_r = random.randint(_CANVAS_MARGIN, grid_h - pack_h - _CANVAS_MARGIN)
        start_c = random.randint(_CANVAS_MARGIN, grid_w - pack_w - _CANVAS_MARGIN)

        positions: list[tuple[int, int] | None] = [None] * len(rects)
        r = start_r
        for row_idx, (row, gaps, row_h, row_w) in enumerate(zip(rows, intra_gaps, row_heights, row_widths)):
            row_c = start_c + random.randint(0, pack_w - row_w)
            c = row_c
            for idx_in_row, idx in enumerate(row):
                h, w = rects[idx].shape
                top_slack = row_h - h
                r0 = r + (0 if top_slack == 0 else random.randint(0, top_slack))
                positions[idx] = (r0, c)
                c += w
                if idx_in_row < len(gaps):
                    c += gaps[idx_in_row]
            r += row_h
            if row_idx < len(inter_row_gaps):
                r += inter_row_gaps[row_idx]

        assert all(pos is not None for pos in positions)
        placed_boxes = [
            (r0, c0, r0 + rects[idx].shape[0] - 1, c0 + rects[idx].shape[1] - 1)
            for idx, (r0, c0) in enumerate(positions)
        ]
        assert all(
            not _rectangles_overlap(a, b, _PLACEMENT_GAP)
            for i, a in enumerate(placed_boxes)
            for b in placed_boxes[i + 1 :]
        )
        return grid_h, grid_w, [(int(r0), int(c0)) for r0, c0 in positions]
    return None


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


def _choose_hole_counts(rect_count: int, allow_tie: bool) -> list[int]:
    available = list(range(_HOLE_COUNT_RANGE[0], _HOLE_COUNT_RANGE[1] + 1))
    if allow_tie and rect_count >= 3:
        chosen = random.sample(available, rect_count - 1)
        return chosen + [random.choice(chosen)]
    return random.sample(available, rect_count)


def _gen_attempt(rect_count: int, allow_tie: bool) -> np.ndarray | None:
    hole_counts = _choose_hole_counts(rect_count, allow_tie=allow_tie)
    random.shuffle(hole_counts)
    rect_colors = random.sample(list(range(1, 10)), rect_count)

    cfg = _GRID_SIZE_BY_COUNT[rect_count]
    h_min, h_max, w_min, w_max, obj_max_h, obj_max_w = cfg

    rects: list[tuple[int, int, np.ndarray]] = []
    for color, target_holes in zip(rect_colors, hole_counts):
        body = _build_rectangle_with_holes(target_holes, obj_max_h, obj_max_w)
        if body is None:
            return None
        rects.append((int(color), int(target_holes), body))

    placement = _place_rectangles(
        [body for _color, _holes, body in rects],
        h_min=h_min,
        h_max=h_max,
        w_min=w_min,
        w_max=w_max,
    )
    if placement is None:
        return None
    grid_h, grid_w, positions = placement

    grid = _render_rectangles(rects, positions, grid_h, grid_w)

    try:
        output = main(grid)
    except Exception:
        return None

    frame_info = sorted(
        [
            (holes, pos[0], pos[1], color)
            for (color, holes, _body), pos in zip(rects, positions)
        ]
    )
    width = max(hole_counts)
    expected = np.zeros((rect_count, width), dtype=int)
    for r, (holes, _r0, _c0, color) in enumerate(frame_info):
        expected[r, :holes] = int(color)

    return grid if np.array_equal(output, expected) else None


def generate_input() -> np.ndarray:
    rect_count = random.randint(*_RECT_COUNT_RANGE)
    allow_tie = random.random() < _TIE_PROBABILITY
    for _ in range(_GEN_RETRIES):
        result = _gen_attempt(rect_count=rect_count, allow_tie=allow_tie)
        if result is not None:
            return result
    for fallback_count in [count for count in range(_RECT_COUNT_RANGE[0], _RECT_COUNT_RANGE[1] + 1) if count != rect_count]:
        for _ in range(max(50, _GEN_RETRIES // 4)):
            result = _gen_attempt(rect_count=fallback_count, allow_tie=allow_tie)
            if result is not None:
                return result
    raise RuntimeError(f"Failed to generate valid input after {_GEN_RETRIES} attempts")
