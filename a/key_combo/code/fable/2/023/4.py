from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import random

import numpy as np


BBox = tuple[int, int, int, int]


@dataclass(frozen=True)
class Rect:
    name: str
    color: int
    box: BBox


def _get_bg(grid: np.ndarray) -> int:
    """ARC-style background = most common color."""
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _bbox(mask: np.ndarray) -> BBox | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _colors_present(grid: np.ndarray, exclude: set[int] | None = None) -> list[int]:
    colors = set(map(int, np.unique(grid)))
    if exclude is not None:
        colors -= set(map(int, exclude))
    return sorted(colors)


def _area(box: BBox) -> int:
    r0, c0, r1, c1 = box
    return int(r1 - r0 + 1) * int(c1 - c0 + 1)


def _strictly_contains(box: BBox, r: int, c: int) -> bool:
    r0, c0, r1, c1 = box
    return int(r0) < int(r) < int(r1) and int(c0) < int(c) < int(c1)


def _bbox_distance(box: BBox, r: int, c: int) -> int:
    r0, c0, r1, c1 = box
    return max(r0 - r, 0, r - r1) + max(c0 - c, 0, c - c1)


def _draw_frame(grid: np.ndarray, box: BBox, color: int) -> None:
    r0, c0, r1, c1 = box
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def _visible_frame_box(grid: np.ndarray, color: int) -> BBox | None:
    coords = np.argwhere(grid == color)
    if coords.size == 0:
        return None
    box = _bbox(grid == color)
    if box is None:
        return None

    r0, c0, r1, c1 = box
    seen_top = False
    seen_bottom = False
    seen_left = False
    seen_right = False

    for r_raw, c_raw in coords:
        r = int(r_raw)
        c = int(c_raw)
        on_border = False
        if r == r0:
            seen_top = True
            on_border = True
        if r == r1:
            seen_bottom = True
            on_border = True
        if c == c0:
            seen_left = True
            on_border = True
        if c == c1:
            seen_right = True
            on_border = True
        if not on_border:
            return None

    if seen_top and seen_bottom and seen_left and seen_right:
        return box
    return None


def _parse_scene(grid: np.ndarray) -> tuple[int, dict[int, BBox], int]:
    bg = _get_bg(grid)
    frame_boxes: dict[int, BBox] = {}
    marker_colors: list[int] = []

    for color in _colors_present(grid, exclude={bg}):
        box = _visible_frame_box(grid, color)
        if box is None:
            marker_colors.append(color)
        else:
            frame_boxes[int(color)] = box

    assert frame_boxes, "expected at least one frame color"
    assert len(marker_colors) == 1, "expected exactly one marker color"
    marker_color = int(marker_colors[0])
    assert np.any(grid == marker_color), "expected marker pixels"
    return bg, frame_boxes, marker_color


def _apply_rule(grid: np.ndarray, bg: int, frame_boxes: dict[int, BBox], marker_color: int) -> np.ndarray:
    out = np.array(grid, copy=True)
    for r_raw, c_raw in np.argwhere(grid == marker_color):
        r = int(r_raw)
        c = int(c_raw)
        owners = [
            (_area(box), int(color))
            for color, box in frame_boxes.items()
            if _strictly_contains(box, r, c)
        ]
        if not owners:
            out[r, c] = bg
            continue

        owners.sort()
        if len(owners) >= 2:
            assert owners[0][0] < owners[1][0], "marker ownership is ambiguous"
        out[r, c] = owners[0][1]
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each marker with the smallest reconstructed rectangle that truly contains it.

    Input structure:
    - The dominant color is the background.
    - Several other colors each show visible fragments of one hollow axis-aligned rectangle frame.
    - One dedicated marker color appears only as scattered single pixels, some inside real frame interiors and some outside every frame.

    Rule:
    - For each frame color, reconstruct its full rectangle from the bbox of all visible pixels of that color.
    - For each marker pixel, collect every reconstructed rectangle whose strict interior contains it and choose the one with smallest area.
    - Recolor the marker to that rectangle's color, or erase it to the background if no reconstructed interior contains it; keep every frame fragment unchanged.

    Key insight:
    - A frame can still own a marker even when the nearby border segment is missing in the visible input, because the true rectangle is defined by the color's full visible bbox, not only by the locally seen fragment.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg, frame_boxes, marker_color = _parse_scene(g)
    return _apply_rule(g, bg, frame_boxes, marker_color)


PALETTE = list(range(10))
GRID_H_RANGE = (18, 28)
GRID_W_RANGE = (18, 28)
OUTER_SIDE_RANGE = (10, 16)
INNER_MIN_SIDE = 5
CORE_MIN_SIDE = 4
MIN_NEST_GAP = 2
OUTER_MARGIN = 3
OUTSIDE_MARKER_COUNT = (1, 3)
GENERATION_ATTEMPTS = 600


def _sample_outer_box(h: int, w: int) -> BBox:
    outer_h = random.randint(OUTER_SIDE_RANGE[0], min(OUTER_SIDE_RANGE[1], h - 2 * OUTER_MARGIN))
    outer_w = random.randint(OUTER_SIDE_RANGE[0], min(OUTER_SIDE_RANGE[1], w - 2 * OUTER_MARGIN))
    r0 = random.randint(OUTER_MARGIN, h - OUTER_MARGIN - outer_h)
    c0 = random.randint(OUTER_MARGIN, w - OUTER_MARGIN - outer_w)
    return int(r0), int(c0), int(r0 + outer_h - 1), int(c0 + outer_w - 1)


def _sample_nested_box(container: BBox, min_side: int, gap: int) -> BBox | None:
    r0, c0, r1, c1 = container
    avail_h = (r1 - r0 + 1) - 2 * gap
    avail_w = (c1 - c0 + 1) - 2 * gap
    if avail_h < min_side or avail_w < min_side:
        return None

    box_h = random.randint(min_side, avail_h)
    box_w = random.randint(min_side, avail_w)
    nr0 = random.randint(r0 + gap, r1 - gap - box_h + 1)
    nc0 = random.randint(c0 + gap, c1 - gap - box_w + 1)
    return int(nr0), int(nc0), int(nr0 + box_h - 1), int(nc0 + box_w - 1)


def _sample_occluder(shape: tuple[int, int], outer: BBox) -> tuple[BBox, tuple[int, int], str]:
    h, w = shape
    r0, c0, r1, c1 = outer
    outer_h = r1 - r0 + 1
    outer_w = c1 - c0 + 1
    side = random.choice(["left", "right", "top", "bottom"])

    if side == "left":
        width = random.randint(3, min(6, c0 + 1))
        overlap_h = random.randint(4, outer_h - 2)
        rr0 = random.randint(r0 + 1, r1 - overlap_h)
        rr1 = rr0 + overlap_h - 1
        box = (rr0, c0 - width + 1, rr1, c0)
        ghost = (random.randint(rr0 + 1, rr1 - 1), c0 + 1)
        return box, ghost, side

    if side == "right":
        width = random.randint(3, min(6, w - c1))
        overlap_h = random.randint(4, outer_h - 2)
        rr0 = random.randint(r0 + 1, r1 - overlap_h)
        rr1 = rr0 + overlap_h - 1
        box = (rr0, c1, rr1, c1 + width - 1)
        ghost = (random.randint(rr0 + 1, rr1 - 1), c1 - 1)
        return box, ghost, side

    if side == "top":
        height = random.randint(3, min(6, r0 + 1))
        overlap_w = random.randint(4, outer_w - 2)
        cc0 = random.randint(c0 + 1, c1 - overlap_w)
        cc1 = cc0 + overlap_w - 1
        box = (r0 - height + 1, cc0, r0, cc1)
        ghost = (r0 + 1, random.randint(cc0 + 1, cc1 - 1))
        return box, ghost, side

    height = random.randint(3, min(6, h - r1))
    overlap_w = random.randint(4, outer_w - 2)
    cc0 = random.randint(c0 + 1, c1 - overlap_w)
    cc1 = cc0 + overlap_w - 1
    box = (r1, cc0, r1 + height - 1, cc1)
    ghost = (r1 - 1, random.randint(cc0 + 1, cc1 - 1))
    return box, ghost, side


def _interior_cells(box: BBox) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = box
    return [(r, c) for r in range(r0 + 1, r1) for c in range(c0 + 1, c1)]


def _far_from_markers(coord: tuple[int, int], used: list[tuple[int, int]], sep: int = 1) -> bool:
    r, c = coord
    return all(max(abs(r - ur), abs(c - uc)) > sep for ur, uc in used)


def _pick_cell(
    candidates: list[tuple[int, int]],
    used: list[tuple[int, int]],
    *,
    exclude_boxes: list[BBox] | None = None,
) -> tuple[int, int] | None:
    exclude_boxes = exclude_boxes or []
    random.shuffle(candidates)
    for coord in candidates:
        if not _far_from_markers(coord, used):
            continue
        if any(_strictly_contains(box, coord[0], coord[1]) for box in exclude_boxes):
            continue
        return coord
    return None


def _outside_candidates(grid: np.ndarray, bg: int, boxes: list[BBox]) -> list[tuple[int, int]]:
    choices: list[tuple[int, int]] = []
    for r in range(grid.shape[0]):
        for c in range(grid.shape[1]):
            if grid[r, c] != bg:
                continue
            if any(_strictly_contains(box, r, c) for box in boxes):
                continue
            distance = min(_bbox_distance(box, r, c) for box in boxes)
            if 1 <= distance <= 5:
                choices.append((r, c))
    return choices


def _ghost_gap_visible(grid: np.ndarray, owner: Rect, ghost: tuple[int, int], side: str) -> bool:
    r, c = ghost
    r0, c0, r1, c1 = owner.box
    if side == "left":
        return int(grid[r, c0]) != owner.color
    if side == "right":
        return int(grid[r, c1]) != owner.color
    if side == "top":
        return int(grid[r0, c]) != owner.color
    return int(grid[r1, c]) != owner.color


def _marker_bbox_has_interior(markers: list[tuple[int, int]]) -> bool:
    rows = [r for r, _ in markers]
    cols = [c for _, c in markers]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    return any(r0 < r < r1 and c0 < c < c1 for r, c in markers)


def _truth_output(grid: np.ndarray, bg: int, marker_color: int, rects: list[Rect]) -> np.ndarray:
    frame_boxes = {rect.color: rect.box for rect in rects}
    return _apply_rule(grid, bg, frame_boxes, marker_color)


def _generate_case() -> tuple[np.ndarray, dict] | None:
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    bg = random.choice(PALETTE)

    outer_box = _sample_outer_box(h, w)
    inner_box = _sample_nested_box(outer_box, INNER_MIN_SIDE, MIN_NEST_GAP)
    if inner_box is None:
        return None

    use_core = random.random() < 0.6
    core_box = None
    if use_core:
        core_box = _sample_nested_box(inner_box, CORE_MIN_SIDE, MIN_NEST_GAP)

    needed_colors = 4 + int(core_box is not None)
    usable = [color for color in PALETTE if color != bg]
    if len(usable) < needed_colors:
        return None
    colors = random.sample(usable, needed_colors)
    marker_color = colors[0]
    rect_colors = colors[1:]

    outer = Rect("outer", rect_colors[0], outer_box)
    inner = Rect("inner", rect_colors[1], inner_box)
    occluder_box, ghost_coord, ghost_side = _sample_occluder((h, w), outer_box)
    occluder = Rect("occluder", rect_colors[2], occluder_box)

    rects = [outer, inner, occluder]
    if core_box is not None:
        rects.append(Rect("core", rect_colors[3], core_box))

    grid = np.full((h, w), bg, dtype=int)
    _draw_frame(grid, outer.box, outer.color)
    if core_box is not None:
        _draw_frame(grid, rects[-1].box, rects[-1].color)
    _draw_frame(grid, inner.box, inner.color)
    _draw_frame(grid, occluder.box, occluder.color)

    used_markers: list[tuple[int, int]] = []
    markers: list[tuple[int, int]] = []

    if grid[ghost_coord] != bg:
        return None
    markers.append(ghost_coord)
    used_markers.append(ghost_coord)

    deepest = rects[-1] if rects[-1].name == "core" else inner
    deep_coord = _pick_cell(_interior_cells(deepest.box), used_markers)
    if deep_coord is None:
        return None
    markers.append(deep_coord)
    used_markers.append(deep_coord)

    if deepest.name == "core":
        inner_only = _pick_cell(_interior_cells(inner.box), used_markers, exclude_boxes=[deepest.box])
        if inner_only is None:
            return None
        markers.append(inner_only)
        used_markers.append(inner_only)

    if random.random() < 0.7:
        occ_coord = _pick_cell(_interior_cells(occluder.box), used_markers)
        if occ_coord is not None and not _strictly_contains(outer.box, occ_coord[0], occ_coord[1]):
            markers.append(occ_coord)
            used_markers.append(occ_coord)

    outside = _outside_candidates(grid, bg, [rect.box for rect in rects])
    outside_needed = random.randint(*OUTSIDE_MARKER_COUNT)
    outside_markers: list[tuple[int, int]] = []
    for _ in range(outside_needed):
        coord = _pick_cell(outside, used_markers)
        if coord is None:
            return None
        outside_markers.append(coord)
        markers.append(coord)
        used_markers.append(coord)
        outside = [cell for cell in outside if cell != coord]

    if not _marker_bbox_has_interior(markers):
        return None

    for r, c in markers:
        if grid[r, c] != bg:
            return None
        grid[r, c] = marker_color

    meta = {
        "bg": bg,
        "marker_color": marker_color,
        "rects": rects,
        "ghost_coord": ghost_coord,
        "ghost_side": ghost_side,
        "outside_markers": outside_markers,
        "deep_coord": deep_coord,
    }
    return grid, meta


def _semantic_checks(grid: np.ndarray, meta: dict) -> bool:
    bg = int(meta["bg"])
    marker_color = int(meta["marker_color"])
    rects: list[Rect] = list(meta["rects"])
    ghost_coord = tuple(meta["ghost_coord"])
    ghost_side = str(meta["ghost_side"])

    if grid.ndim != 2 or max(grid.shape) > 30:
        return False
    other_counts = [np.count_nonzero(grid == color) for color in np.unique(grid) if int(color) != bg]
    if other_counts and np.count_nonzero(grid == bg) <= max(other_counts):
        return False

    true_boxes = {rect.color: rect.box for rect in rects}
    for rect in rects:
        visible_box = _visible_frame_box(grid, rect.color)
        if visible_box != rect.box:
            return False

    try:
        parsed_bg, parsed_boxes, parsed_marker = _parse_scene(grid)
    except AssertionError:
        return False

    if parsed_bg != bg or parsed_marker != marker_color or parsed_boxes != true_boxes:
        return False

    if not _ghost_gap_visible(grid, next(rect for rect in rects if rect.name == "outer"), ghost_coord, ghost_side):
        return False

    owners_by_marker: dict[tuple[int, int], list[Rect]] = {}
    for r_raw, c_raw in np.argwhere(grid == marker_color):
        coord = (int(r_raw), int(c_raw))
        owners = [rect for rect in rects if _strictly_contains(rect.box, coord[0], coord[1])]
        owners_by_marker[coord] = owners
        owner_areas = sorted(_area(rect.box) for rect in owners)
        if len(owner_areas) >= 2 and owner_areas[0] == owner_areas[1]:
            return False

    if not any(len(owners) == 0 for owners in owners_by_marker.values()):
        return False
    if not any(len(owners) >= 2 for owners in owners_by_marker.values()):
        return False
    if len(owners_by_marker.get(ghost_coord, [])) != 1:
        return False
    if owners_by_marker[ghost_coord][0].name != "outer":
        return False

    expected = _truth_output(grid, bg, marker_color, rects)
    try:
        actual = main(grid)
    except AssertionError:
        return False
    return np.array_equal(actual, expected)


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        generated = _generate_case()
        if generated is None:
            continue
        grid, meta = generated
        if _semantic_checks(grid, meta):
            return grid
    raise RuntimeError("failed to generate a valid ghost-rectangle ownership input")
