from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_UTILS_DIR) not in sys.path:
    sys.path.append(str(_UTILS_DIR))

from arc_utils import Color, bbox_union, find_components, get_bg, in_bounds, paste


BBox = tuple[int, int, int, int]
Direction = tuple[int, int]

DECORATIONS: tuple[tuple[tuple[int, int], int], ...] = (
    ((-1, -1), Color.BLUE),
    ((-1, 1), Color.RED),
    ((1, -1), Color.GREEN),
    ((1, 1), Color.YELLOW),
)

DIRS: tuple[str, ...] = ("left", "right", "up", "down")
DIR_VECTORS: dict[str, Direction] = {
    "left": (0, -1),
    "right": (0, 1),
    "up": (-1, 0),
    "down": (1, 0),
}

SHAPES: tuple[tuple[int, int], ...] = ((2, 2), (2, 3))
GRID_MIN = 14
GRID_MAX = 26
PAIR_COUNTS = (2, 3)
STEM_LENGTHS = (1, 2, 3, 4)
ANCHOR_GAPS = (1, 2, 3)
GEN_ATTEMPTS = 400
PAIR_PLACEMENT_ATTEMPTS = 200


def _bbox_h(box: BBox) -> int:
    return int(box[2] - box[0] + 1)


def _bbox_w(box: BBox) -> int:
    return int(box[3] - box[1] + 1)


def _boxes_overlap(a: BBox, b: BBox) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _expand_box(box: BBox, pad: int, shape: tuple[int, int]) -> BBox:
    h, w = shape
    return (
        max(0, box[0] - pad),
        max(0, box[1] - pad),
        min(h - 1, box[2] + pad),
        min(w - 1, box[3] + pad),
    )


def _translate_box(box: BBox, dr: int, dc: int) -> BBox:
    return (box[0] + dr, box[1] + dc, box[2] + dr, box[3] + dc)


def _component_is_solid_rectangle(comp: dict) -> bool:
    box = comp["bbox"]
    return int(comp["area"]) == _bbox_h(box) * _bbox_w(box)


def _coords_mask(shape: tuple[int, int], coords: np.ndarray) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    mask[coords[:, 0], coords[:, 1]] = True
    return mask


def _find_body_box(mask: np.ndarray) -> BBox:
    row_counts = mask.sum(axis=1)
    col_counts = mask.sum(axis=0)
    rows = np.where(row_counts >= 2)[0]
    cols = np.where(col_counts >= 2)[0]
    assert rows.size >= 2 and cols.size >= 2, "body must be at least 2x2"

    r0, r1 = int(rows[0]), int(rows[-1])
    c0, c1 = int(cols[0]), int(cols[-1])
    assert np.array_equal(rows, np.arange(r0, r1 + 1))
    assert np.array_equal(cols, np.arange(c0, c1 + 1))
    assert np.all(mask[r0 : r1 + 1, c0 : c1 + 1]), "body must be a solid rectangle"

    body_mask = np.zeros_like(mask)
    body_mask[r0 : r1 + 1, c0 : c1 + 1] = True
    assert np.any(mask & ~body_mask), "moving object must include a stem"
    return r0, c0, r1, c1


def _stem_info(stem_mask: np.ndarray, body_box: BBox) -> tuple[Direction, int]:
    r0, c0, r1, c1 = body_box
    rows, cols = np.where(stem_mask)
    assert rows.size > 0, "expected a non-empty stem"

    if rows.size == 1:
        row = int(rows[0])
        col = int(cols[0])
        if r0 <= row <= r1 and col == c0 - 1:
            return (0, -1), 1
        if r0 <= row <= r1 and col == c1 + 1:
            return (0, 1), 1
        if c0 <= col <= c1 and row == r0 - 1:
            return (-1, 0), 1
        if c0 <= col <= c1 and row == r1 + 1:
            return (1, 0), 1
        raise AssertionError("single-cell stem must touch one body side")

    unique_rows = sorted(set(map(int, rows)))
    unique_cols = sorted(set(map(int, cols)))

    if len(unique_rows) == 1:
        row = unique_rows[0]
        seg = sorted(map(int, cols))
        assert seg == list(range(seg[0], seg[-1] + 1)), "horizontal stem must be contiguous"
        assert r0 <= row <= r1, "horizontal stem must attach to a body row"
        if seg[-1] == c0 - 1:
            return (0, -1), len(seg)
        if seg[0] == c1 + 1:
            return (0, 1), len(seg)
        raise AssertionError("horizontal stem must extend from one body side")

    if len(unique_cols) == 1:
        col = unique_cols[0]
        seg = sorted(map(int, rows))
        assert seg == list(range(seg[0], seg[-1] + 1)), "vertical stem must be contiguous"
        assert c0 <= col <= c1, "vertical stem must attach to a body column"
        if seg[-1] == r0 - 1:
            return (-1, 0), len(seg)
        if seg[0] == r1 + 1:
            return (1, 0), len(seg)
        raise AssertionError("vertical stem must extend from one body side")

    raise AssertionError("stem must be a straight one-pixel-wide segment")


def _match_anchor(body_box: BBox, direction: Direction, stem_len: int, anchors: list[dict]) -> tuple[dict, int]:
    r0, c0, r1, c1 = body_box
    body_h, body_w = _bbox_h(body_box), _bbox_w(body_box)
    dr, dc = direction
    candidates: list[tuple[int, dict]] = []

    for anchor in anchors:
        box = anchor["bbox"]
        if _bbox_h(box) != body_h or _bbox_w(box) != body_w:
            continue

        ar0, ac0, ar1, ac1 = box
        if dr == 0:
            if (ar0, ar1) != (r0, r1):
                continue
            shift = ac0 - c1 - 1 if dc == 1 else c0 - ac1 - 1
        else:
            if (ac0, ac1) != (c0, c1):
                continue
            shift = ar0 - r1 - 1 if dr == 1 else r0 - ar1 - 1

        if shift >= stem_len:
            candidates.append((int(shift), anchor))

    assert candidates, "each moving body must have a matching anchor in stem direction"
    best_shift = min(shift for shift, _anchor in candidates)
    best = [anchor for shift, anchor in candidates if shift == best_shift]
    assert len(best) == 1, "matching anchor must be unique"
    return best[0], best_shift


def _paint_box(out: np.ndarray, box: BBox, color: int) -> None:
    r0, c0, r1, c1 = box
    patch = np.full((r1 - r0 + 1, c1 - c0 + 1), int(color), dtype=int)
    paste(out, patch, r0, c0)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide each stemmed rectangle onto its aligned anchor, erase the stems, then decorate the merged outline corners.

    Input structure:
    - The grid has black background and 2-3 independent pairs of foreground objects.
    - In each pair, one object is a solid rectangular anchor and the other is a same-sized rectangular body with a straight one-pixel stem extending from one side.
    - The anchor lies in the stem direction, aligned so that after the body slides along that direction the two rectangles become one larger rectangle.

    Rule:
    - Split foreground components into solid rectangular anchors and non-rectangular moving objects.
    - For each moving object, isolate its rectangular body, read the stem direction, erase the whole moving object, and translate only the body along the stem direction until it becomes edge-adjacent to its aligned same-sized anchor.
    - Keep anchors unchanged, then place outer diagonal corner markers around the merged body+anchor bounding box: top-left blue, top-right red, bottom-left green, bottom-right yellow.

    Key insight:
    - The decoration positions cannot be read directly from the input: they depend on the post-slide merged bounding box, so the slide must be simulated before the corner markers can be placed.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"
    bg = int(get_bg(g))

    comps = [
        {**comp, "id": idx}
        for idx, comp in enumerate(
            find_components(g, bg=bg, connectivity=4, monochromatic=True, return_format="dict")
        )
    ]
    assert comps, "expected foreground components"

    anchors = [comp for comp in comps if _component_is_solid_rectangle(comp)]
    movers = [comp for comp in comps if not _component_is_solid_rectangle(comp)]
    assert anchors and movers and len(anchors) == len(movers), "expected paired anchors and movers"

    out = g.copy()
    for mover in movers:
        coords = mover["coords"]
        out[coords[:, 0], coords[:, 1]] = bg

    unused_anchors = anchors[:]
    merged_boxes: list[BBox] = []

    for mover in movers:
        mask = _coords_mask(g.shape, mover["coords"])
        body_box = _find_body_box(mask)
        body_mask = np.zeros_like(mask)
        r0, c0, r1, c1 = body_box
        body_mask[r0 : r1 + 1, c0 : c1 + 1] = True
        direction, stem_len = _stem_info(mask & ~body_mask, body_box)
        anchor, shift = _match_anchor(body_box, direction, stem_len, unused_anchors)
        unused_anchors = [candidate for candidate in unused_anchors if candidate["id"] != anchor["id"]]

        dr, dc = direction
        moved_box = _translate_box(body_box, dr * shift, dc * shift)
        mr0, mc0, mr1, mc1 = moved_box
        assert 0 <= mr0 <= mr1 < g.shape[0] and 0 <= mc0 <= mc1 < g.shape[1]
        assert np.all(out[mr0 : mr1 + 1, mc0 : mc1 + 1] == bg), "moved body must land on empty space"
        _paint_box(out, moved_box, int(mover["color"]))

        merged = bbox_union([moved_box, anchor["bbox"]])
        assert merged is not None
        merged_boxes.append(merged)

    for box in merged_boxes:
        r0, c0, r1, c1 = box
        corner_coords = (
            (r0 - 1, c0 - 1, Color.BLUE),
            (r0 - 1, c1 + 1, Color.RED),
            (r1 + 1, c0 - 1, Color.GREEN),
            (r1 + 1, c1 + 1, Color.YELLOW),
        )
        for rr, cc, color in corner_coords:
            if in_bounds(out.shape, rr, cc):
                out[rr, cc] = int(color)

    return out


def _choose_directions(pair_count: int) -> list[str]:
    if pair_count == 2 and random.random() < 0.7:
        dirs = [random.choice(("left", "right")), random.choice(("up", "down"))]
        random.shuffle(dirs)
        return dirs
    return random.sample(DIRS, k=pair_count)


def _build_pair_layout(
    top: int,
    left: int,
    h: int,
    w: int,
    direction_name: str,
    stem_len: int,
    gap: int,
    anchor_color: int,
    body_color: int,
) -> dict:
    direction = DIR_VECTORS[direction_name]
    total_shift = stem_len + gap

    if direction_name == "right":
        moved_box = (top, left, top + h - 1, left + w - 1)
        anchor_box = (top, left + w, top + h - 1, left + 2 * w - 1)
        input_body_box = _translate_box(moved_box, 0, -total_shift)
        stem_row = top + random.randrange(h)
        stem_box = (
            stem_row,
            input_body_box[3] + 1,
            stem_row,
            input_body_box[3] + stem_len,
        )
        merged_box = (top, left, top + h - 1, left + 2 * w - 1)
    elif direction_name == "left":
        anchor_box = (top, left, top + h - 1, left + w - 1)
        moved_box = (top, left + w, top + h - 1, left + 2 * w - 1)
        input_body_box = _translate_box(moved_box, 0, total_shift)
        stem_row = top + random.randrange(h)
        stem_box = (
            stem_row,
            input_body_box[1] - stem_len,
            stem_row,
            input_body_box[1] - 1,
        )
        merged_box = (top, left, top + h - 1, left + 2 * w - 1)
    elif direction_name == "down":
        moved_box = (top, left, top + h - 1, left + w - 1)
        anchor_box = (top + h, left, top + 2 * h - 1, left + w - 1)
        input_body_box = _translate_box(moved_box, -total_shift, 0)
        stem_col = left + random.randrange(w)
        stem_box = (
            input_body_box[2] + 1,
            stem_col,
            input_body_box[2] + stem_len,
            stem_col,
        )
        merged_box = (top, left, top + 2 * h - 1, left + w - 1)
    else:
        anchor_box = (top, left, top + h - 1, left + w - 1)
        moved_box = (top + h, left, top + 2 * h - 1, left + w - 1)
        input_body_box = _translate_box(moved_box, total_shift, 0)
        stem_col = left + random.randrange(w)
        stem_box = (
            input_body_box[0] - stem_len,
            stem_col,
            input_body_box[0] - 1,
            stem_col,
        )
        merged_box = (top, left, top + 2 * h - 1, left + w - 1)

    return {
        "direction_name": direction_name,
        "direction": direction,
        "shape": (h, w),
        "stem_len": stem_len,
        "gap": gap,
        "total_shift": total_shift,
        "anchor_color": anchor_color,
        "body_color": body_color,
        "anchor_box": anchor_box,
        "moved_box": moved_box,
        "input_body_box": input_body_box,
        "stem_box": stem_box,
        "merged_box": merged_box,
    }


def _all_boxes_in_bounds(shape: tuple[int, int], boxes: list[BBox]) -> bool:
    h, w = shape
    for r0, c0, r1, c1 in boxes:
        if not (0 <= r0 <= r1 < h and 0 <= c0 <= c1 < w):
            return False
    return True


def _candidate_union_positions(shape: tuple[int, int], h: int, w: int, direction_name: str, total_shift: int) -> tuple[range, range] | None:
    H, W = shape
    if direction_name in {"left", "right"}:
        row_range = range(1, H - h)
        if direction_name == "right":
            col_lo = max(1, total_shift)
            col_hi = W - 2 * w - 1
        else:
            col_lo = 1
            col_hi = W - 2 * w - total_shift
        if col_lo > col_hi:
            return None
        return row_range, range(col_lo, col_hi + 1)

    col_range = range(1, W - w)
    if direction_name == "down":
        row_lo = max(1, total_shift)
        row_hi = H - 2 * h - 1
    else:
        row_lo = 1
        row_hi = H - 2 * h - total_shift
    if row_lo > row_hi:
        return None
    return range(row_lo, row_hi + 1), col_range


def _reserve_boxes(pair: dict, shape: tuple[int, int]) -> tuple[BBox, BBox]:
    input_footprint = bbox_union([pair["anchor_box"], pair["input_body_box"], pair["stem_box"]])
    assert input_footprint is not None
    return (
        _expand_box(input_footprint, 1, shape),
        _expand_box(pair["merged_box"], 1, shape),
    )


def _render_pair(grid: np.ndarray, pair: dict) -> None:
    _paint_box(grid, pair["anchor_box"], pair["anchor_color"])
    _paint_box(grid, pair["input_body_box"], pair["body_color"])
    _paint_box(grid, pair["stem_box"], pair["body_color"])


def _sample_shapes(pair_count: int) -> list[tuple[int, int]]:
    shapes = [random.choice(SHAPES) for _ in range(pair_count)]
    if pair_count >= 2 and random.random() < 0.6:
        shapes[0] = (2, 2)
        shapes[1] = (2, 3)
        random.shuffle(shapes)
    return shapes


def _generate_attempt() -> np.ndarray | None:
    pair_count = random.choice(PAIR_COUNTS)
    side_lo = GRID_MIN + 3 * (pair_count - 2)
    H = random.randint(side_lo, GRID_MAX)
    W = random.randint(side_lo, GRID_MAX)
    shape = (H, W)

    colors = random.sample(range(1, 10), 2 * pair_count)
    directions = _choose_directions(pair_count)
    shapes = _sample_shapes(pair_count)

    pair_specs = []
    for idx in range(pair_count):
        pair_specs.append(
            {
                "shape": shapes[idx],
                "direction_name": directions[idx],
                "stem_len": random.choice(STEM_LENGTHS),
                "gap": random.choice(ANCHOR_GAPS),
                "anchor_color": colors[2 * idx],
                "body_color": colors[2 * idx + 1],
            }
        )

    pair_specs.sort(
        key=lambda spec: (
            spec["shape"][0] * spec["shape"][1],
            spec["stem_len"] + spec["gap"],
            spec["direction_name"] in {"left", "right"},
        ),
        reverse=True,
    )

    placed_pairs: list[dict] = []
    input_reserves: list[BBox] = []
    output_reserves: list[BBox] = []

    for spec in pair_specs:
        h, w = spec["shape"]
        total_shift = spec["stem_len"] + spec["gap"]
        ranges = _candidate_union_positions(shape, h, w, spec["direction_name"], total_shift)
        if ranges is None:
            return None
        row_range, col_range = ranges
        if len(row_range) == 0 or len(col_range) == 0:
            return None

        placed = False
        for _ in range(PAIR_PLACEMENT_ATTEMPTS):
            top = random.choice(row_range)
            left = random.choice(col_range)
            pair = _build_pair_layout(
                top,
                left,
                h,
                w,
                spec["direction_name"],
                spec["stem_len"],
                spec["gap"],
                spec["anchor_color"],
                spec["body_color"],
            )
            boxes = [pair["anchor_box"], pair["input_body_box"], pair["stem_box"], pair["merged_box"]]
            if not _all_boxes_in_bounds(shape, boxes):
                continue

            input_reserve, output_reserve = _reserve_boxes(pair, shape)
            if any(_boxes_overlap(input_reserve, other) for other in input_reserves):
                continue
            if any(_boxes_overlap(output_reserve, other) for other in output_reserves):
                continue

            placed_pairs.append(pair)
            input_reserves.append(input_reserve)
            output_reserves.append(output_reserve)
            placed = True
            break

        if not placed:
            return None

    grid = np.zeros(shape, dtype=int)
    for pair in placed_pairs:
        _render_pair(grid, pair)

    out = main(grid)
    assert out.shape == grid.shape
    assert not np.array_equal(out, grid), "transformation must be non-trivial"
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("generate_input failed after repeated placement attempts")
