# Variant of 89565ca0
# Dimension: D3 — 源对象：彩色线框架 (src)
# Change: 轴对齐的1像素厚彩色线绘制的分区框架，包含水平和垂直线段形成网格状房间结构... → 实心彩色矩形带内部洞

from __future__ import annotations

import random

import numpy as np

from arc_utils import bbox, colors_present, crop_bbox, get_bg, label_mask


# ---------------------------------------------------------------------------
# Design analysis for the D3 variant
#
# a) Concrete behavior + trace
# - Original objects were sparse 1px wall networks; room counting worked by
#   reconstructing horizontal/vertical wall segments from row/column density.
# - The new objects are dense colored rectangles whose interiors contain several
#   black rectangular holes. Example trace:
#   original crop: a colored outline with one internal divider -> background
#   splits into 2 enclosed rooms.
#   variant crop: a solid colored block with 2 separate black windows ->
#   count 2 internal black components instead.
# - The output remains "one horizontal color bar per object, length = count",
#   but the counted primitive is now a hole rather than a line-bounded room.
#
# b) Complexity calibration
# - If the new rectangles were allowed to overlap like the old frames, large
#   opaque areas would hide each other's holes and make the D3 change hard to
#   infer from 2-4 examples.
# - To keep the variation visible but still nontrivial, each puzzle has 3-5
#   rectangles, each with 1-6 holes, scattered noise of one extra color, and
#   distinct hole counts to avoid ambiguous tie-breaking.
#
# c) Implicit-assumption audit against the old main()
# - Old `_reconstruct_walls()` assumed true structure is sparse and rectilinear.
#   With filled rectangles, almost every row/column is dense, so that logic
#   would collapse the whole bbox into "walls" and miscount.
# - Old `_count_enclosed_regions(~walls)` counted non-wall regions. In the
#   variant, the signal is instead the set of black components *inside* a
#   mostly filled bbox.
# - Old `_cover_order()` depended on overlaps between equal-count frames. That
#   hidden assumption is intentionally removed here; generator avoids ties and
#   avoids overlaps so the D3 change stays observable.
#
# d) Placement legality
# - Rectangles are placed without overlap and with a 1-cell clearance, so every
#   bbox is recoverable from visible color alone.
# - Noise may appear on background and on rectangle interiors, but never on a
#   rectangle's outer border and never inside holes; therefore bbox recovery and
#   hole counting remain stable.
#
# e) Visual observability
# - Human solvers can directly see "solid colored blocks with black cut-outs",
#   which is visibly different from thin line drawings.
# - Sorting never depends on color ids. Objects are sorted only by visible hole
#   count, and generator uses distinct counts so examples expose a single rule.
#
# f) Visual cleanliness
# - No object overlap, no hole overlap, and only one noise color. This prevents
#   crowded multi-layer regions that would obscure which black areas are actual
#   holes.
#
# g) Color-change inferability
# - Colors are preserved from input to output; there is no hidden remapping.
#   The only non-object color is the scattered pad/noise color.
# ---------------------------------------------------------------------------


# --------------- solver ---------------

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


def main(grid: np.ndarray) -> np.ndarray:
    """
    Convert each colored solid rectangle-with-holes into a horizontal bar whose length is its number of internal holes.

    Input structure:
    - Black background with several non-overlapping axis-aligned colored filled rectangles.
    - Each real object contains one or more black rectangular holes strictly inside its colored bbox.
    - One extra non-black color appears only as scattered noise/padding pixels on the background or on object bodies.

    Rule:
    - For each non-background color, take its visible bbox and test whether it behaves like a dense filled rectangle with an intact outer border.
    - Inside each valid bbox, count the 4-connected black components that stay away from the bbox edge and are themselves perfect rectangles.
    - The color with zero such holes is the noise/pad color.
    - Sort the real object colors by hole count and output one left-aligned row per color, padded with the noise color.

    Key insight:
    - The important topology is no longer sparse walls but black voids carved out of a solid color block; the object's hole count is stable once the outer rectangle is identified.
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
    width = max(holes for _color, holes, _b in ordered)

    out = np.full((len(ordered), width), pad, dtype=int)
    for r, (color, holes, _b) in enumerate(ordered):
        out[r, :holes] = int(color)
    return out


# --------------- generator ---------------

_RECT_COUNT_RANGE = (3, 5)
_HOLE_COUNT_RANGE = (1, 6)
_GRID_SIZE_BY_COUNT = {
    5: (28, 30, 28, 30, 15, 16),
    4: (25, 30, 25, 30, 15, 17),
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


def _gen_attempt() -> np.ndarray | None:
    rect_count = random.randint(*_RECT_COUNT_RANGE)
    hole_counts = random.sample(list(range(_HOLE_COUNT_RANGE[0], _HOLE_COUNT_RANGE[1] + 1)), rect_count)
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

    ordered = sorted(zip(hole_counts, rect_colors), key=lambda item: item[0])
    width = max(hole_counts)
    expected = np.full((rect_count, width), noise_color, dtype=int)
    for r, (holes, color) in enumerate(ordered):
        expected[r, :holes] = int(color)

    return grid if np.array_equal(output, expected) else None


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {_GEN_RETRIES} attempts")
