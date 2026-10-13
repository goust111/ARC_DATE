from __future__ import annotations

from collections import defaultdict, deque
import random

import numpy as np


Grid = np.ndarray
BBox = tuple[int, int, int, int]


def _sign(x: float) -> int:
    return 1 if x > 0 else -1 if x < 0 else 0


def _round_half_up(x: float) -> int:
    return int(np.floor(x + 0.5))


def _get_bg(grid: Grid) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _bbox_from_coords(coords: np.ndarray) -> BBox:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _bbox_center(bbox: BBox) -> tuple[float, float]:
    r0, c0, r1, c1 = bbox
    return (r0 + r1) / 2.0, (c0 + c1) / 2.0


def _bbox_intersects(a: BBox, b: BBox, gap: int = 0) -> bool:
    return not (
        a[2] + gap < b[0]
        or b[2] + gap < a[0]
        or a[3] + gap < b[1]
        or b[3] + gap < a[1]
    )


def _find_components(grid: Grid, bg: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[dict] = []

    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) == bg:
                continue

            color = int(grid[r, c])
            q = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []

            while q:
                cr, cc = q.popleft()
                cells.append((cr, cc))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = cr + dr, cc + dc
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if seen[nr, nc] or int(grid[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    q.append((nr, nc))

            coords = np.asarray(cells, dtype=int)
            components.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": _bbox_from_coords(coords),
                    "area": int(len(cells)),
                }
            )

    return components


def _normalized_component(component: dict) -> tuple[set[tuple[int, int]], int, int]:
    r0, c0, r1, c1 = map(int, component["bbox"])
    coords = {
        (int(r) - r0, int(c) - c0)
        for r, c in np.asarray(component["coords"], dtype=int)
    }
    return coords, r1 - r0 + 1, c1 - c0 + 1


def _completion_candidates(comp_a: dict, comp_b: dict) -> list[dict]:
    a_cells, ha, wa = _normalized_component(comp_a)
    b_cells, hb, wb = _normalized_component(comp_b)
    a_center = ((ha - 1) / 2.0, (wa - 1) / 2.0)

    candidates: list[dict] = []
    for dr in range(-hb, ha + 1):
        for dc in range(-wb, wa + 1):
            shifted_b = {(r + dr, c + dc) for r, c in b_cells}
            if a_cells & shifted_b:
                continue

            union = a_cells | shifted_b
            rows = [r for r, _ in union]
            cols = [c for _, c in union]
            r0, r1 = min(rows), max(rows)
            c0, c1 = min(cols), max(cols)
            rect_h = r1 - r0 + 1
            rect_w = c1 - c0 + 1
            if rect_h * rect_w != len(union):
                continue

            b_center = (dr + (hb - 1) / 2.0, dc + (wb - 1) / 2.0)
            diff_r = b_center[0] - a_center[0]
            diff_c = b_center[1] - a_center[1]
            if abs(diff_c) > abs(diff_r):
                axis = "h"
                main_diff = diff_c
                perp_diff = diff_r
            elif abs(diff_r) > abs(diff_c):
                axis = "v"
                main_diff = diff_r
                perp_diff = diff_c
            else:
                axis = "tie"
                main_diff = 0.0
                perp_diff = 0.0

            candidates.append(
                {
                    "rect_h": rect_h,
                    "rect_w": rect_w,
                    "axis": axis,
                    "main_diff": main_diff,
                    "perp_diff": perp_diff,
                    "dr": dr,
                    "dc": dc,
                }
            )

    return candidates


def _best_completion(comp_a: dict, comp_b: dict) -> tuple[int, int]:
    candidates = _completion_candidates(comp_a, comp_b)
    assert candidates, "same-color fragments must admit a rectangular completion"

    center_a = _bbox_center(comp_a["bbox"])
    center_b = _bbox_center(comp_b["bbox"])
    orig_diff_r = center_b[0] - center_a[0]
    orig_diff_c = center_b[1] - center_a[1]

    if abs(orig_diff_c) > abs(orig_diff_r):
        preferred_axis = "h"
        preferred_sign = _sign(orig_diff_c)
    elif abs(orig_diff_r) > abs(orig_diff_c):
        preferred_axis = "v"
        preferred_sign = _sign(orig_diff_r)
    else:
        preferred_axis = None
        preferred_sign = 0

    if preferred_axis is not None:
        axis_filtered = [cand for cand in candidates if cand["axis"] == preferred_axis]
        if axis_filtered:
            candidates = axis_filtered

        sign_filtered = [
            cand
            for cand in candidates
            if _sign(float(cand["main_diff"])) == preferred_sign
        ]
        if sign_filtered:
            candidates = sign_filtered

    def score(cand: dict) -> tuple[float, float, float, float, int, int, int]:
        main_mag = abs(float(cand["main_diff"]))
        perp_mag = abs(float(cand["perp_diff"]))
        dominance = main_mag - perp_mag
        return (
            dominance,
            -perp_mag,
            -main_mag,
            -abs(int(cand["dr"])) - abs(int(cand["dc"])),
            -int(cand["rect_h"]),
            -int(cand["rect_w"]),
            -int(cand["dr"]) * 100 - int(cand["dc"]),
        )

    best = max(candidates, key=score)
    return int(best["rect_h"]), int(best["rect_w"])


def _target_top_left(bbox_a: BBox, bbox_b: BBox, rect_h: int, rect_w: int) -> tuple[int, int]:
    center_a = _bbox_center(bbox_a)
    center_b = _bbox_center(bbox_b)
    mid_r = (center_a[0] + center_b[0]) / 2.0
    mid_c = (center_a[1] + center_b[1]) / 2.0
    top = _round_half_up(mid_r - (rect_h - 1) / 2.0)
    left = _round_half_up(mid_c - (rect_w - 1) / 2.0)
    return top, left


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pair scattered same-color fragments, merge each pair into a full rectangle, and place that rectangle at the pair midpoint.

    Input structure:
    - The background is grey.
    - Each non-background color appears as exactly two separated 4-connected fragments.
    - For each color, translating the two fragments toward one another in one cardinal direction makes their union a solid axis-aligned rectangle.

    Rule:
    - Group fragments by color and solve each same-color pair independently.
    - Infer the unique rectangular completion by finding a relative translation whose non-overlapping union is a filled rectangle; this determines whether the pair merges horizontally or vertically and what the finished rectangle size is.
    - Compute the midpoint of the two fragment bounding-box centers.
    - Erase the original fragments and draw the completed rectangle centered on that midpoint, rounding the top-left placement to the nearest grid cell.

    Key insight:
    - The fragments must first be matched and geometrically completed into a rectangle, and only then can the final location be recovered from the midpoint of their original positions.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _get_bg(g)
    h, w = g.shape

    components = _find_components(g, bg)
    by_color: dict[int, list[dict]] = defaultdict(list)
    for comp in components:
        by_color[int(comp["color"])].append(comp)

    out = np.full_like(g, bg)
    for color, parts in by_color.items():
        assert len(parts) == 2, "each foreground color must have exactly two fragments"
        rect_h, rect_w = _best_completion(parts[0], parts[1])
        top, left = _target_top_left(parts[0]["bbox"], parts[1]["bbox"], rect_h, rect_w)
        assert 0 <= top <= h - rect_h and 0 <= left <= w - rect_w, "completed rectangle must stay in bounds"
        region = out[top : top + rect_h, left : left + rect_w]
        assert np.all(region == bg), "completed rectangles must not overlap"
        region[:] = int(color)

    return out


# --------------- generator (att2) ---------------

BG = 5
PALETTE = [0, 1, 2, 3, 4, 6, 7, 8, 9]

GRID_H_RANGE = (18, 28)
GRID_W_RANGE = (18, 28)
PAIR_COUNT_RANGE = (2, 4)

FINAL_THICK_RANGE = (2, 5)
PART_RANGE = (1, 4)

INPUT_GAP_RANGE = (2, 8)
INPUT_JITTER = 2
INPUT_BOX_GAP = 1
OUTPUT_BOX_GAP = 1

PAIR_LOCAL_ATTEMPTS = 300
MAX_GENERATION_ATTEMPTS = 400


def _sample_pair_specs() -> list[dict]:
    count = random.randint(*PAIR_COUNT_RANGE)
    colors = random.sample(PALETTE, count)

    axes = ["h", "v"] * (count // 2)
    if count % 2 == 1:
        axes.append(random.choice(["h", "v"]))
    random.shuffle(axes)

    specs: list[dict] = []
    for color, axis in zip(colors, axes):
        if axis == "h":
            height = random.randint(*FINAL_THICK_RANGE)
            left_w = random.randint(*PART_RANGE)
            right_w = random.randint(*PART_RANGE)
            while right_w == left_w:
                right_w = random.randint(*PART_RANGE)
            specs.append(
                {
                    "color": color,
                    "axis": axis,
                    "a_shape": (height, left_w),
                    "b_shape": (height, right_w),
                    "rect_shape": (height, left_w + right_w),
                }
            )
        else:
            width = random.randint(*FINAL_THICK_RANGE)
            top_h = random.randint(*PART_RANGE)
            bottom_h = random.randint(*PART_RANGE)
            while bottom_h == top_h:
                bottom_h = random.randint(*PART_RANGE)
            specs.append(
                {
                    "color": color,
                    "axis": axis,
                    "a_shape": (top_h, width),
                    "b_shape": (bottom_h, width),
                    "rect_shape": (top_h + bottom_h, width),
                }
            )

    random.shuffle(specs)
    return specs


def _boxes_clear(boxes: list[BBox], candidate: BBox, gap: int) -> bool:
    return all(not _bbox_intersects(candidate, box, gap=gap) for box in boxes)


def _place_pair(
    grid_h: int,
    grid_w: int,
    spec: dict,
    input_boxes: list[BBox],
    output_boxes: list[BBox],
) -> dict | None:
    ha, wa = spec["a_shape"]
    hb, wb = spec["b_shape"]
    rect_h, rect_w = spec["rect_shape"]

    for _ in range(PAIR_LOCAL_ATTEMPTS):
        gap = random.randint(*INPUT_GAP_RANGE)
        sign = random.choice([-1, 1])

        if spec["axis"] == "h":
            drow = random.randint(-INPUT_JITTER, INPUT_JITTER)
            row_lo = max(0, -drow)
            row_hi = min(grid_h - ha, grid_h - hb - drow)
            if row_lo > row_hi:
                continue
            ra = random.randint(row_lo, row_hi)
            rb = ra + drow

            if sign > 0:
                col_lo, col_hi = 0, grid_w - (wa + gap + wb)
                if col_lo > col_hi:
                    continue
                ca = random.randint(col_lo, col_hi)
                cb = ca + wa + gap
            else:
                col_lo, col_hi = wb + gap, grid_w - wa
                if col_lo > col_hi:
                    continue
                ca = random.randint(col_lo, col_hi)
                cb = ca - wb - gap
        else:
            dcol = random.randint(-INPUT_JITTER, INPUT_JITTER)
            col_lo = max(0, -dcol)
            col_hi = min(grid_w - wa, grid_w - wb - dcol)
            if col_lo > col_hi:
                continue
            ca = random.randint(col_lo, col_hi)
            cb = ca + dcol

            if sign > 0:
                row_lo, row_hi = 0, grid_h - (ha + gap + hb)
                if row_lo > row_hi:
                    continue
                ra = random.randint(row_lo, row_hi)
                rb = ra + ha + gap
            else:
                row_lo, row_hi = hb + gap, grid_h - ha
                if row_lo > row_hi:
                    continue
                ra = random.randint(row_lo, row_hi)
                rb = ra - hb - gap

        bbox_a = (ra, ca, ra + ha - 1, ca + wa - 1)
        bbox_b = (rb, cb, rb + hb - 1, cb + wb - 1)
        if _bbox_intersects(bbox_a, bbox_b, gap=INPUT_BOX_GAP):
            continue
        if not _boxes_clear(input_boxes, bbox_a, gap=INPUT_BOX_GAP):
            continue
        if not _boxes_clear(input_boxes, bbox_b, gap=INPUT_BOX_GAP):
            continue

        top, left = _target_top_left(bbox_a, bbox_b, rect_h, rect_w)
        out_bbox = (top, left, top + rect_h - 1, left + rect_w - 1)
        if top < 0 or left < 0 or out_bbox[2] >= grid_h or out_bbox[3] >= grid_w:
            continue
        if not _boxes_clear(output_boxes, out_bbox, gap=OUTPUT_BOX_GAP):
            continue

        return {
            "color": spec["color"],
            "a_bbox": bbox_a,
            "b_bbox": bbox_b,
            "out_bbox": out_bbox,
        }

    return None


def _paint_box(grid: Grid, bbox: BBox, color: int) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0 : r1 + 1, c0 : c1 + 1] = int(color)


def _gen_attempt() -> np.ndarray | None:
    grid_h = random.randint(*GRID_H_RANGE)
    grid_w = random.randint(*GRID_W_RANGE)
    specs = _sample_pair_specs()

    input_boxes: list[BBox] = []
    output_boxes: list[BBox] = []
    placements: list[dict] = []

    for spec in specs:
        placed = _place_pair(grid_h, grid_w, spec, input_boxes, output_boxes)
        if placed is None:
            return None
        placements.append(placed)
        input_boxes.extend([placed["a_bbox"], placed["b_bbox"]])
        output_boxes.append(placed["out_bbox"])

    inp = np.full((grid_h, grid_w), BG, dtype=int)
    expected = np.full((grid_h, grid_w), BG, dtype=int)
    for placed in placements:
        color = int(placed["color"])
        _paint_box(inp, placed["a_bbox"], color)
        _paint_box(inp, placed["b_bbox"], color)
        _paint_box(expected, placed["out_bbox"], color)

    try:
        actual = main(inp)
    except AssertionError:
        return None
    return inp if np.array_equal(actual, expected) else None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid midpoint-merge input")
