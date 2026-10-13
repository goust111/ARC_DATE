import random
from typing import Dict, List, Optional, Tuple

import numpy as np


BG = 0
YELLOW = 4
GREY = 5
MAGENTA = 6
ORANGE = 7

RECTANGLE_COLORS = (1, 2, 3, 8, 9)
SIDE_NAMES = ("top", "bottom", "left", "right")

GEN_RETRIES = 400
PAIR_COUNT = 2
PAIR_GAP = 2
PLACEMENT_MARGIN = 1
BACK_SIZE_RANGE = (5, 8)
PROTRUSION_RANGE = (2, 4)
MAX_OVERLAP_THICKNESS = 3
EXTRA_RECT_PROBABILITY = 0.55
EXTRA_RECT_SIZE_RANGE = (3, 6)


def _mode_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _bbox(mask: np.ndarray) -> Tuple[int, int, int, int]:
    coords = np.argwhere(mask)
    assert len(coords) > 0, "bbox requires at least one pixel"
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _bbox_intersection(
    box_a: Tuple[int, int, int, int],
    box_b: Tuple[int, int, int, int],
) -> Optional[Tuple[int, int, int, int]]:
    r0 = max(box_a[0], box_b[0])
    c0 = max(box_a[1], box_b[1])
    r1 = min(box_a[2], box_b[2])
    c1 = min(box_a[3], box_b[3])
    if r0 > r1 or c0 > c1:
        return None
    return r0, c0, r1, c1


def _parse_legend(row: np.ndarray, bg: int) -> Dict[int, int]:
    legend: Dict[int, int] = {}
    c = 0
    while c < len(row):
        if int(row[c]) == bg:
            c += 1
            continue
        assert c + 1 < len(row), "legend entries must be two cells wide"
        source = int(row[c])
        marker = int(row[c + 1])
        assert source not in {bg, GREY, YELLOW, ORANGE, MAGENTA}
        assert marker in {YELLOW, ORANGE}, "legend marker must be yellow or orange"
        assert source not in legend, "each occluder color must have one legend entry"
        legend[source] = marker
        c += 2
    assert legend, "expected at least one legend entry"
    return legend


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover hidden rectangle overlaps according to a bottom legend that maps each occluder color to either erase-or-fuse behavior.

    Input structure:
    - The background is a dominant black field, with a full grey separator row above a one-row bottom legend.
    - Above the separator are several differently colored solid rectangles; some are partially hidden because a front rectangle overwrote the overlap area.
    - Each hidden rectangle still exposes enough of its outline that its original axis-aligned bounding rectangle can be inferred from its visible pixels.
    - Every legend entry is a two-cell horizontal pair: an occluder color followed by a yellow or orange marker.

    Rule:
    - Infer one rectangle bbox per visible non-special scene color by taking the tight bbox of that color's visible pixels.
    - For every pair of inferred rectangles whose bboxes overlap, inspect the overlap patch in the input; if it is uniformly one rectangle's color, that rectangle is the front occluder and the other is the hidden back rectangle.
    - Use the legend entry for the occluder color: yellow means restore the overlap to the back rectangle's color, while orange means recolor the overlap to magenta.
    - Leave all non-overlap pixels, the grey separator, and the legend unchanged.

    Key insight:
    - The visible pixels already reveal each hidden rectangle's full bbox, and the bottom legend does not identify the hidden color at all; it only tells how to reinterpret the overlap based on whichever rectangle currently sits on top.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape[0] >= 3 and g.shape[1] >= 4, "expected a 2D ARC grid"

    separator_rows = [r for r in range(g.shape[0]) if np.all(g[r] == GREY)]
    assert len(separator_rows) == 1, "expected exactly one full grey separator row"
    separator_row = separator_rows[0]
    assert 1 <= separator_row < g.shape[0] - 1, "separator must split the scene from a bottom legend row"

    scene = g[:separator_row]
    legend_row = g[separator_row + 1]
    bg = _mode_color(scene)
    legend = _parse_legend(legend_row, bg)

    scene_colors = [
        int(color)
        for color in np.unique(scene)
        if int(color) not in {bg, GREY, YELLOW, ORANGE, MAGENTA}
    ]
    assert len(scene_colors) >= 2, "expected at least two scene rectangles"

    rectangles: Dict[int, Tuple[int, int, int, int]] = {}
    for color in scene_colors:
        rectangles[color] = _bbox(scene == color)

    out = g.copy()
    painted = np.zeros_like(scene, dtype=bool)

    for i, color_a in enumerate(scene_colors):
        for color_b in scene_colors[i + 1 :]:
            overlap = _bbox_intersection(rectangles[color_a], rectangles[color_b])
            if overlap is None:
                continue

            r0, c0, r1, c1 = overlap
            patch = scene[r0 : r1 + 1, c0 : c1 + 1]
            patch_colors = set(map(int, np.unique(patch)))
            assert patch_colors <= {color_a, color_b}, "overlap patch must contain only the two rectangle colors"
            assert len(patch_colors) == 1, "overlap patch must reveal a unique front rectangle"

            front = int(next(iter(patch_colors)))
            back = color_b if front == color_a else color_a
            assert front in legend, "every front occluder color must appear in the legend"

            overlap_mask = np.zeros_like(scene, dtype=bool)
            overlap_mask[r0 : r1 + 1, c0 : c1 + 1] = True
            assert not np.any(painted & overlap_mask), "overlap regions must not stack on the same cells"

            if legend[front] == YELLOW:
                out[r0 : r1 + 1, c0 : c1 + 1] = back
            else:
                out[r0 : r1 + 1, c0 : c1 + 1] = MAGENTA
            painted |= overlap_mask

    return out


def _make_pair_geometry(
    rng: random.Random,
    back_color: int,
    front_color: int,
    strategy: int,
    max_h: int,
    max_w: int,
) -> Optional[dict]:
    for _ in range(200):
        side = rng.choice(SIDE_NAMES)
        back_h = rng.randint(BACK_SIZE_RANGE[0], min(BACK_SIZE_RANGE[1], max_h - 2))
        back_w = rng.randint(BACK_SIZE_RANGE[0], min(BACK_SIZE_RANGE[1], max_w))

        if back_h < 5 or back_w < 5:
            continue

        if side in {"top", "bottom"}:
            overlap_w = rng.randint(2, back_w - 2)
            overlap_h = rng.randint(2, min(MAX_OVERLAP_THICKNESS, back_h - 1))
            protrusion = rng.randint(*PROTRUSION_RANGE)
            front_h = overlap_h + protrusion
            col_offset = rng.randint(1, back_w - overlap_w - 1)

            if side == "top":
                back_box = (front_h - overlap_h, 0, front_h - overlap_h + back_h - 1, back_w - 1)
                front_box = (0, col_offset, front_h - 1, col_offset + overlap_w - 1)
            else:
                back_box = (0, 0, back_h - 1, back_w - 1)
                front_box = (back_h - overlap_h, col_offset, back_h - overlap_h + front_h - 1, col_offset + overlap_w - 1)
        else:
            overlap_h = rng.randint(2, back_h - 2)
            overlap_w = rng.randint(2, min(MAX_OVERLAP_THICKNESS, back_w - 1))
            protrusion = rng.randint(*PROTRUSION_RANGE)
            front_w = overlap_w + protrusion
            row_offset = rng.randint(1, back_h - overlap_h - 1)

            if side == "left":
                back_box = (0, front_w - overlap_w, back_h - 1, front_w - overlap_w + back_w - 1)
                front_box = (row_offset, 0, row_offset + overlap_h - 1, front_w - 1)
            else:
                back_box = (0, 0, back_h - 1, back_w - 1)
                front_box = (row_offset, back_w - overlap_w, row_offset + overlap_h - 1, back_w - overlap_w + front_w - 1)

        footprint_h = max(back_box[2], front_box[2]) + 1
        footprint_w = max(back_box[3], front_box[3]) + 1
        if footprint_h > max_h or footprint_w > max_w:
            continue

        overlap_box = _bbox_intersection(back_box, front_box)
        assert overlap_box is not None
        return {
            "back_color": back_color,
            "front_color": front_color,
            "strategy": strategy,
            "side": side,
            "back_box_local": back_box,
            "front_box_local": front_box,
            "overlap_box_local": overlap_box,
            "footprint": (footprint_h, footprint_w),
        }
    return None


def _place_in_region(
    rng: random.Random,
    region: Tuple[int, int, int, int],
    footprint: Tuple[int, int],
) -> Optional[Tuple[int, int]]:
    r0, c0, r1, c1 = region
    height = r1 - r0 + 1
    width = c1 - c0 + 1
    fh, fw = footprint
    if fh > height or fw > width:
        return None

    row_lo = r0 + PLACEMENT_MARGIN
    row_hi = r1 - fh - PLACEMENT_MARGIN + 1
    col_lo = c0 + PLACEMENT_MARGIN
    col_hi = c1 - fw - PLACEMENT_MARGIN + 1
    if row_lo > row_hi or col_lo > col_hi:
        return None

    rows = list(range(row_lo, row_hi + 1))
    cols = list(range(col_lo, col_hi + 1))
    rng.shuffle(rows)
    rng.shuffle(cols)
    for row in rows:
        for col in cols:
            return row, col
    return None


def _draw_box(grid: np.ndarray, box: Tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = box
    grid[r0 : r1 + 1, c0 : c1 + 1] = color


def _translate_box(box: Tuple[int, int, int, int], dr: int, dc: int) -> Tuple[int, int, int, int]:
    return box[0] + dr, box[1] + dc, box[2] + dr, box[3] + dc


def _mark_occupied(occupied: np.ndarray, box: Tuple[int, int, int, int]) -> None:
    r0, c0, r1, c1 = box
    rr0 = max(0, r0 - PLACEMENT_MARGIN)
    cc0 = max(0, c0 - PLACEMENT_MARGIN)
    rr1 = min(occupied.shape[0] - 1, r1 + PLACEMENT_MARGIN)
    cc1 = min(occupied.shape[1] - 1, c1 + PLACEMENT_MARGIN)
    occupied[rr0 : rr1 + 1, cc0 : cc1 + 1] = True


def _find_extra_position(
    rng: random.Random,
    occupied: np.ndarray,
    rect_h: int,
    rect_w: int,
) -> Optional[Tuple[int, int]]:
    h, w = occupied.shape
    rows = list(range(1, h - rect_h))
    cols = list(range(1, w - rect_w))
    rng.shuffle(rows)
    rng.shuffle(cols)
    for r in rows:
        for c in cols:
            test_box = (r, c, r + rect_h - 1, c + rect_w - 1)
            rr0 = max(0, test_box[0] - PLACEMENT_MARGIN)
            cc0 = max(0, test_box[1] - PLACEMENT_MARGIN)
            rr1 = min(h - 1, test_box[2] + PLACEMENT_MARGIN)
            cc1 = min(w - 1, test_box[3] + PLACEMENT_MARGIN)
            if not occupied[rr0 : rr1 + 1, cc0 : cc1 + 1].any():
                return r, c
    return None


def _build_legend_row(rng: random.Random, width: int, entries: List[Tuple[int, int]]) -> Optional[np.ndarray]:
    row = np.full(width, BG, dtype=int)
    order = entries[:]
    rng.shuffle(order)

    cursor = rng.randint(0, 2)
    for index, (color, marker) in enumerate(order):
        if cursor + 1 >= width:
            return None
        row[cursor] = color
        row[cursor + 1] = marker
        if index == len(order) - 1:
            continue
        cursor += 2 + rng.randint(1, 3)
    return row


def _gen_attempt(rng: random.Random) -> Optional[np.ndarray]:
    scene_h = rng.randint(16, 24)
    scene_w = rng.randint(22, 28)
    total_h = scene_h + 2
    if total_h > 30:
        return None

    left_w = (scene_w - PAIR_GAP) // 2
    right_w = scene_w - PAIR_GAP - left_w
    if min(left_w, right_w) < 10:
        return None

    colors = list(RECTANGLE_COLORS)
    rng.shuffle(colors)

    pair_specs = [
        (colors[0], colors[1], YELLOW),
        (colors[2], colors[3], ORANGE),
    ]
    if rng.random() < 0.5:
        pair_specs.reverse()

    regions = [
        (0, 0, scene_h - 1, left_w - 1),
        (0, left_w + PAIR_GAP, scene_h - 1, scene_w - 1),
    ]
    if rng.random() < 0.5:
        pair_specs = pair_specs[::-1]

    pairs = []
    for (back_color, front_color, strategy), region in zip(pair_specs, regions):
        region_h = region[2] - region[0] + 1
        region_w = region[3] - region[1] + 1
        pair = _make_pair_geometry(
            rng,
            back_color=back_color,
            front_color=front_color,
            strategy=strategy,
            max_h=region_h - 2,
            max_w=region_w - 2,
        )
        if pair is None:
            return None
        origin = _place_in_region(rng, region, pair["footprint"])
        if origin is None:
            return None
        pair["origin"] = origin
        pairs.append(pair)

    scene = np.full((scene_h, scene_w), BG, dtype=int)
    expected_scene = np.full((scene_h, scene_w), BG, dtype=int)
    occupied = np.zeros_like(scene, dtype=bool)

    for pair in pairs:
        origin_r, origin_c = pair["origin"]
        back_box = _translate_box(pair["back_box_local"], origin_r, origin_c)
        front_box = _translate_box(pair["front_box_local"], origin_r, origin_c)
        overlap_box = _translate_box(pair["overlap_box_local"], origin_r, origin_c)

        pair["back_box"] = back_box
        pair["front_box"] = front_box
        pair["overlap_box"] = overlap_box

        _draw_box(scene, back_box, pair["back_color"])
        _draw_box(expected_scene, back_box, pair["back_color"])
        _mark_occupied(
            occupied,
            (
                origin_r,
                origin_c,
                origin_r + pair["footprint"][0] - 1,
                origin_c + pair["footprint"][1] - 1,
            ),
        )

    extra_color = colors[4]
    if rng.random() < EXTRA_RECT_PROBABILITY:
        rect_h = rng.randint(*EXTRA_RECT_SIZE_RANGE)
        rect_w = rng.randint(*EXTRA_RECT_SIZE_RANGE)
        extra_origin = _find_extra_position(rng, occupied, rect_h, rect_w)
        if extra_origin is not None:
            extra_box = (
                extra_origin[0],
                extra_origin[1],
                extra_origin[0] + rect_h - 1,
                extra_origin[1] + rect_w - 1,
            )
            _draw_box(scene, extra_box, extra_color)
            _draw_box(expected_scene, extra_box, extra_color)
            _mark_occupied(occupied, extra_box)

    for pair in pairs:
        _draw_box(scene, pair["front_box"], pair["front_color"])
        _draw_box(expected_scene, pair["front_box"], pair["front_color"])
        if pair["strategy"] == YELLOW:
            _draw_box(expected_scene, pair["overlap_box"], pair["back_color"])
        else:
            _draw_box(expected_scene, pair["overlap_box"], MAGENTA)

    separator = np.full((1, scene_w), GREY, dtype=int)
    legend_entries = [(pair["front_color"], pair["strategy"]) for pair in pairs]
    legend_row = _build_legend_row(rng, scene_w, legend_entries)
    if legend_row is None:
        return None

    grid = np.vstack([scene, separator, legend_row[None, :]])
    expected = np.vstack([expected_scene, separator, legend_row[None, :]])

    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid overlap-repair ARC sample")
