# Combo variant of 89565ca0 (index 31)
# combo_spec: D3/0+D5/0+D7/1
# Name: 实心带洞+垂直色条+反向映射

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from arc_utils import bbox, colors_present, crop_bbox, get_bg, label_mask


# ---------------------------------------------------------------------------
# Design analysis for combo D3/0 + D5/0 + D7/1
#
# a) Concrete behavior + trace
# - Original 89565ca0 input objects were sparse colored wall grids. Solver logic
#   rebuilt walls, counted enclosed rooms, then drew one horizontal bar per
#   object whose row length equaled that count.
# - D3/0 replaces the source object itself: each real object is now a solid
#   colored rectangle containing several black rectangular holes. The counted
#   primitive is therefore "internal black holes" rather than "background rooms
#   bounded by recovered walls".
# - D5/0 rotates the destination encoding into columns. D7/1 then reverses the
#   count-to-size mapping. Example:
#   three objects have hole counts [1, 3, 4], sorted ascending by count.
#   Original task output would have been horizontal lengths [1, 3, 4].
#   D3-only would still output horizontal lengths [1, 3, 4], but derived from
#   hole counts.
#   This combo instead sets max_holes = 4 and column heights
#   [4, 2, 1] = [4-1+1, 4-3+1, 4-4+1], rendered as bottom-aligned vertical bars.
# - So the fusion changes both the topology being measured and the spatial
#   orientation / direction of the output encoding.
#
# b) Complexity calibration
# - Allowing opaque rectangles to overlap would hide holes and make D3 hard to
#   infer, so the generator removes overlap entirely and keeps a small clearance.
# - Hole counts remain in the familiar 1..6 range with 3-5 objects, but counts
#   are sampled without replacement and with span >= 2. That keeps the inverse
#   mapping observable and prevents ambiguous tie handling.
# - The output still uses only object colors plus one noise/pad color, so the
#   representation is visually different from the original but not cluttered.
#
# c) Implicit-assumption audit against the original main()
# - Original `_reconstruct_walls()` and `_cover_order()` assumed sparse line
#   structures and possible overlap-based tiebreaks. Those assumptions fail for
#   filled rectangles, so they are removed entirely.
# - Original `rooms > 0` vs `rooms == 0` still survives conceptually, but the
#   measured quantity is now "valid internal holes inside a dense rectangle".
# - Original `out[r, :rooms] = color` assumed horizontal direct mapping. The
#   combo needs two coordinated changes: use inverse height
#   `max_holes - holes + 1`, and place it as a bottom-aligned column instead of
#   a left-aligned row. If either change is omitted, the behavior collapses to a
#   simpler single-variant task.
# - The new solver assumes each true object has an intact colored outer border,
#   enough visible fill to look like one solid rectangle, and holes that remain
#   black rectangular components not touching the bbox edge. The generator
#   explicitly protects the outer border, forbids overlap, and never writes
#   noise into holes.
#
# d) Placement legality
# - Rectangles are placed fully inside the canvas with at least one cell of
#   clearance, so visible bboxes are recoverable and never merge.
# - Holes are carved strictly inside each rectangle. Noise can land on the body
#   or background but not on outer borders or in holes, so counting remains
#   well-posed.
#
# e) Visual observability
# - Humans can directly see the source-side change because the inputs are solid
#   blocks with black cut-outs, not thin frames.
# - They can also compare outputs and see two repeated regularities:
#   fewer holes -> taller column, more holes -> shorter column; columns are read
#   left-to-right in ascending hole-count order.
# - No sorting uses hidden facts like color ids. Generator avoids ties so the
#   visible count alone determines order.
#
# f) Visual cleanliness
# - No overlap and only one noise color keep the input clean.
# - Each output column contains one color plus pad color, and bars are all
#   bottom-aligned, so the transformed output remains compact and legible.
#
# g) Change inferability
# - There is no color remapping. Object colors persist into output columns.
# - The only changed correspondences are structural and count-based: identify
#   holes, sort by hole count, then encode the inverse of that count vertically.
# ---------------------------------------------------------------------------


_MIN_VISIBLE_FILL_RATIO = 0.5
_MIN_OBJECT_SIDE = 6


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


def _object_stats(g: np.ndarray, color: int, bg: int) -> tuple[int, tuple[int, int, int, int] | None]:
    b = bbox(g == int(color))
    if b is None:
        return 0, None
    mask = crop_bbox(g == int(color), b).astype(bool)
    if not _looks_like_solid_rectangle(mask):
        return 0, b
    bg_mask = crop_bbox(g == int(bg), b).astype(bool)
    holes = _count_internal_rectangles(bg_mask)
    return int(holes), b


def _inverse_bar_height(holes: int, max_holes: int) -> int:
    return int(max_holes - holes + 1)


def _render_vertical_bars(ordered: list[tuple[int, int, tuple[int, int, int, int] | None]], pad: int) -> np.ndarray:
    max_holes = max(holes for _color, holes, _b in ordered)
    heights = [_inverse_bar_height(holes, max_holes) for _color, holes, _b in ordered]
    out_h = max(heights)
    out_w = len(ordered)
    out = np.full((out_h, out_w), int(pad), dtype=int)
    for c, ((color, _holes, _b), height) in enumerate(zip(ordered, heights)):
        out[out_h - height :, c] = int(color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Convert each colored rectangle-with-holes into a bottom-aligned vertical bar whose height is the inverse of that object's hole count within the scene.

    Input structure:
    - Black background with several non-overlapping axis-aligned colored filled rectangles.
    - Each real object has an intact colored outer border and one or more black rectangular holes strictly inside its bbox.
    - One extra non-black color appears only as scattered noise/padding pixels on the background or on rectangle bodies.

    Rule:
    - For each non-background color, take its visible bbox and keep it only if it behaves like a dense solid rectangle with an intact border.
    - Inside each valid bbox, count the 4-connected black components that do not touch the bbox edge and are themselves perfect rectangles; this is that color's hole count.
    - The color with zero valid holes is the noise/pad color. Sort the real object colors by increasing hole count.
    - Let max_holes be the largest hole count among the real objects. Output one column per object, in sorted order, using a bottom-aligned vertical bar of height `max_holes - holes + 1` and filling all remaining cells with the noise color.

    Key insight:
    - The task is no longer about reconstructing sparse walls. The stable signal is the number of black voids carved out of each solid rectangle, and the destination encoding deliberately reverses that count while rotating it into vertical columns.
    """
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    assert bg == 0

    counts: list[tuple[int, int, tuple[int, int, int, int] | None]] = []
    for color in colors_present(g, exclude={bg}):
        holes, b = _object_stats(g, int(color), bg)
        counts.append((int(color), int(holes), b))

    drawings = [(color, holes, b) for color, holes, b in counts if holes > 0]
    pad_colors = [color for color, holes, _b in counts if holes == 0]
    assert drawings
    assert len(pad_colors) == 1

    pad = int(pad_colors[0])
    ordered = sorted(drawings, key=lambda item: (item[1], item[2][0], item[2][1]))
    return _render_vertical_bars(ordered, pad)


# --------------- generator ---------------

_RECT_COUNT_RANGE = (3, 5)
_HOLE_COUNT_RANGE = (1, 6)
_MIN_COUNT_SPAN = 2
_GRID_SIZE_BY_COUNT = {
    # Five rectangles need noticeably more room; the original cap made this
    # branch almost unreachable and under-covered in generated examples.
    5: (34, 36, 34, 36, 14, 15),
    4: (27, 32, 27, 32, 15, 17),
    3: (21, 29, 21, 30, 16, 19),
}
_OUTER_MARGIN_RANGE = (2, 3)
_HOLE_SIZE_RANGE = (2, 3)
_GUTTER_RANGE = (1, 2)
_PLACEMENT_GAP = 1
_PLACEMENT_RETRY = 500
_NOISE_FRACTION_RANGE = (0.04, 0.09)
_BODY_DAMAGE_FRACTION = 0.12
_GEN_RETRIES = 400


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


def _build_rectangle_with_holes(target_holes: int, max_h: int, max_w: int) -> np.ndarray | None:
    for _ in range(80):
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
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    grid = np.zeros((grid_h, grid_w), dtype=int)
    body_mask = np.zeros((grid_h, grid_w), dtype=bool)
    hole_mask = np.zeros((grid_h, grid_w), dtype=bool)
    protected_border = np.zeros((grid_h, grid_w), dtype=bool)

    for (color, _holes, body), (r0, c0) in zip(rects, positions):
        h, w = body.shape
        rs, cs = np.where(body)
        grid[rs + r0, cs + c0] = int(color)
        body_mask[rs + r0, cs + c0] = True

        holes = ~body
        hrs, hcs = np.where(holes)
        hole_mask[hrs + r0, hcs + c0] = True

        protected_border[r0, c0 : c0 + w] = True
        protected_border[r0 + h - 1, c0 : c0 + w] = True
        protected_border[r0 : r0 + h, c0] = True
        protected_border[r0 : r0 + h, c0 + w - 1] = True

    return grid, body_mask, hole_mask, protected_border


def _scatter_noise(
    grid: np.ndarray,
    body_mask: np.ndarray,
    hole_mask: np.ndarray,
    protected_border: np.ndarray,
    noise_color: int,
) -> None:
    h, w = grid.shape
    total_cells = h * w
    noise_count = int(total_cells * random.uniform(*_NOISE_FRACTION_RANGE))

    free_bg = (grid == 0) & (~hole_mask)
    bg_rs, bg_cs = np.where(free_bg)
    bg_take = min(len(bg_rs), max(1, noise_count // 2))
    if bg_take > 0:
        chosen = np.random.permutation(len(bg_rs))[:bg_take]
        grid[bg_rs[chosen], bg_cs[chosen]] = int(noise_color)

    damage_mask = body_mask & (~protected_border)
    damage_rs, damage_cs = np.where(damage_mask)
    max_damage = min(
        len(damage_rs),
        noise_count - bg_take,
        int(_BODY_DAMAGE_FRACTION * max(1, len(damage_rs))),
    )
    if max_damage > 0:
        chosen = np.random.permutation(len(damage_rs))[:max_damage]
        grid[damage_rs[chosen], damage_cs[chosen]] = int(noise_color)


def _sample_hole_counts(rect_count: int) -> list[int] | None:
    available = list(range(_HOLE_COUNT_RANGE[0], _HOLE_COUNT_RANGE[1] + 1))
    if rect_count > len(available):
        return None
    for _ in range(50):
        hole_counts = random.sample(available, rect_count)
        if max(hole_counts) - min(hole_counts) >= _MIN_COUNT_SPAN:
            return hole_counts
    return None


def _expected_output(hole_counts: list[int], rect_colors: list[int], noise_color: int) -> np.ndarray:
    ordered = sorted(zip(hole_counts, rect_colors), key=lambda item: item[0])
    max_holes = max(hole_counts)
    heights = [_inverse_bar_height(holes, max_holes) for holes, _color in ordered]
    out = np.full((max(heights), len(ordered)), int(noise_color), dtype=int)
    for c, ((holes, color), height) in enumerate(zip(ordered, heights)):
        out[-height:, c] = int(color)
    return out


def _gen_attempt() -> np.ndarray | None:
    rect_count = random.randint(*_RECT_COUNT_RANGE)
    hole_counts = _sample_hole_counts(rect_count)
    if hole_counts is None:
        return None
    colors = random.sample(list(range(1, 10)), rect_count + 1)
    rect_colors = colors[:rect_count]
    noise_color = colors[-1]

    cfg = _GRID_SIZE_BY_COUNT[rect_count]
    h_min, h_max, w_min, w_max, obj_max_h, obj_max_w = cfg

    rects: list[tuple[int, int, np.ndarray]] = []
    for color, target_holes in zip(rect_colors, hole_counts):
        body = _build_rectangle_with_holes(target_holes, obj_max_h, obj_max_w)
        if body is None:
            return None
        rects.append((int(color), int(target_holes), body))

    max_obj_h = max(body.shape[0] for _color, _holes, body in rects)
    max_obj_w = max(body.shape[1] for _color, _holes, body in rects)
    grid_h = random.randint(max(h_min, max_obj_h + 1, h_max - 2), h_max)
    grid_w = random.randint(max(w_min, max_obj_w + 1, w_max - 2), w_max)

    positions = _place_rectangles([body for _color, _holes, body in rects], grid_h, grid_w)
    if positions is None:
        return None

    grid, body_mask, hole_mask, protected_border = _render_rectangles(rects, positions, grid_h, grid_w)
    _scatter_noise(grid, body_mask, hole_mask, protected_border, noise_color)

    try:
        output = main(grid)
    except Exception:
        return None

    expected = _expected_output(hole_counts, rect_colors, noise_color)
    return grid if np.array_equal(output, expected) else None


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("Failed to generate valid input for combo 89565ca0/31")
