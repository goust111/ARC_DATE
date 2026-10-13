from __future__ import annotations

import itertools
import random
import sys
from pathlib import Path

import numpy as np

_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_UTILS_DIR) not in sys.path:
    sys.path.append(str(_UTILS_DIR))

from arc_utils import find_components


BLACK = 0
GREEN = 3
BRIDGE = 8
RECT_COLORS = (1, 2, 4, 5, 6, 7)
SOURCE_QUADRANTS = (0, 1, 2, 3)  # TL, TR, BL, BR
MAX_GENERATION_ATTEMPTS = 400


def _find_cross_axes(grid: np.ndarray) -> tuple[int, int]:
    rows = np.where(np.all(grid == GREEN, axis=1))[0]
    cols = np.where(np.all(grid == GREEN, axis=0))[0]
    assert len(rows) == 1 and len(cols) == 1, "expected one full green row and one full green column"
    return int(rows[0]), int(cols[0])


def _rect_from_bbox(bbox: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = map(int, bbox)
    return r0, c0, r1, c1


def _paint_rect(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = _rect_from_bbox(bbox)
    grid[r0 : r1 + 1, c0 : c1 + 1] = int(color)


def _reflect_bbox(
    bbox: tuple[int, int, int, int],
    axis_r: int,
    axis_c: int,
    flip_rows: bool,
    flip_cols: bool,
) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = _rect_from_bbox(bbox)
    if flip_rows:
        r0, r1 = 2 * axis_r - r1, 2 * axis_r - r0
    if flip_cols:
        c0, c1 = 2 * axis_c - c1, 2 * axis_c - c0
    return int(r0), int(c0), int(r1), int(c1)


def _family_bboxes(
    bbox: tuple[int, int, int, int], axis_r: int, axis_c: int
) -> list[tuple[int, int, int, int]]:
    seen = set()
    boxes = []
    for flip_rows in (False, True):
        for flip_cols in (False, True):
            reflected = _reflect_bbox(bbox, axis_r, axis_c, flip_rows, flip_cols)
            if reflected not in seen:
                seen.add(reflected)
                boxes.append(reflected)
    return sorted(boxes)


def _extract_input_rectangles(
    grid: np.ndarray, axis_r: int, axis_c: int
) -> list[dict[str, int | tuple[int, int, int, int]]]:
    comps = find_components(grid, bg=BLACK, connectivity=4, monochromatic=True)
    rects = []
    for comp in comps:
        color = int(comp["color"])
        if color == GREEN:
            continue
        bbox = _rect_from_bbox(comp["bbox"])
        r0, c0, r1, c1 = bbox
        assert r1 < axis_r or r0 > axis_r
        assert c1 < axis_c or c0 > axis_c
        assert int(comp["area"]) == (r1 - r0 + 1) * (c1 - c0 + 1), "component must be a solid rectangle"
        assert np.all(grid[r0 : r1 + 1, c0 : c1 + 1] == color), "rectangle bbox must be uniformly filled"
        rects.append({"color": color, "bbox": bbox})

    assert len(rects) >= 1, "expected at least one non-green rectangle"
    colors = [int(rect["color"]) for rect in rects]
    assert len(colors) == len(set(colors)), "each input rectangle color must be unique"
    rects.sort(key=lambda rect: (rect["bbox"][0], rect["bbox"][1]))
    return rects


def _horizontal_candidate(
    base: np.ndarray,
    box_a: tuple[int, int, int, int],
    box_b: tuple[int, int, int, int],
) -> dict[str, int] | None:
    left, right = sorted((box_a, box_b), key=lambda box: (box[1], box[0]))
    if left[3] >= right[1]:
        return None

    row0 = max(left[0], right[0])
    row1 = min(left[2], right[2])
    overlap = row1 - row0 + 1
    if overlap < 3:
        return None

    gap = base[row0 : row1 + 1, left[3] + 1 : right[1]]
    if gap.size and not np.isin(gap, (BLACK, GREEN)).all():
        return None

    return {
        "gap": right[1] - left[3] - 1,
        "kind": 0,
        "r0": row0 + 1,
        "r1": row1 - 1,
        "c0": left[3],
        "c1": right[1],
    }


def _vertical_candidate(
    base: np.ndarray,
    box_a: tuple[int, int, int, int],
    box_b: tuple[int, int, int, int],
) -> dict[str, int] | None:
    top, bottom = sorted((box_a, box_b), key=lambda box: (box[0], box[1]))
    if top[2] >= bottom[0]:
        return None

    col0 = max(top[1], bottom[1])
    col1 = min(top[3], bottom[3])
    overlap = col1 - col0 + 1
    if overlap < 3:
        return None

    gap = base[top[2] + 1 : bottom[0], col0 : col1 + 1]
    if gap.size and not np.isin(gap, (BLACK, GREEN)).all():
        return None

    return {
        "gap": bottom[0] - top[2] - 1,
        "kind": 1,
        "r0": top[2],
        "r1": bottom[0],
        "c0": col0 + 1,
        "c1": col1 - 1,
    }


def _best_bridge(
    base: np.ndarray, boxes: list[tuple[int, int, int, int]]
) -> dict[str, int] | None:
    candidates = []
    for box_a, box_b in itertools.combinations(boxes, 2):
        horizontal = _horizontal_candidate(base, box_a, box_b)
        if horizontal is not None:
            candidates.append(horizontal)
        vertical = _vertical_candidate(base, box_a, box_b)
        if vertical is not None:
            candidates.append(vertical)

    if not candidates:
        return None
    return min(
        candidates,
        key=lambda cand: (cand["gap"], cand["kind"], cand["r0"], cand["c0"], cand["r1"], cand["c1"]),
    )


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete each colored rectangle by cross-axis reflection, then bridge the nearest reflected pair of the same color.

    Input structure:
    - The grid has a black background with one full green row and one full green column forming a cross.
    - Two or three other colors each form one solid rectangle placed wholly inside distinct quadrants.
    - No input rectangle touches the green axes, and every non-green foreground component is rectangular.

    Rule:
    - Detect the green cross row and column and treat them as horizontal and vertical mirror axes.
    - For each non-green rectangle, stamp its row reflection, column reflection, and double reflection in addition to the original, using later input rectangles to overwrite earlier ones where reflected copies collide.
    - For each color family of reflected copies, find all horizontal or vertical facing pairs whose gap contains only black or green cells and whose overlap across the gap is at least three cells.
    - Draw one cyan bridge for that color on the pair with the smallest gap; the bridge spans from rectangle edge to rectangle edge and uses the overlap shrunk by one cell at both ends.

    Key insight:
    The bridges are determined only after symmetry completion, because the mirrored copies create same-color facing pairs whose cross-axis gaps depend on the rectangle's distance from the axes.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    axis_r, axis_c = _find_cross_axes(g)
    rects = _extract_input_rectangles(g, axis_r, axis_c)

    out = np.zeros_like(g)
    out[axis_r, :] = GREEN
    out[:, axis_c] = GREEN

    families: dict[int, list[tuple[int, int, int, int]]] = {}
    for rect in rects:
        color = int(rect["color"])
        boxes = _family_bboxes(rect["bbox"], axis_r, axis_c)
        families[color] = boxes
        for bbox in boxes:
            r0, c0, r1, c1 = bbox
            assert 0 <= r0 <= r1 < g.shape[0] and 0 <= c0 <= c1 < g.shape[1], "reflection must stay in bounds"
            _paint_rect(out, bbox, color)

    reflected = out.copy()
    for color in sorted(families):
        bridge = _best_bridge(reflected, families[color])
        if bridge is None:
            continue
        if bridge["kind"] == 0:
            out[bridge["r0"] : bridge["r1"] + 1, bridge["c0"] : bridge["c1"] + 1] = BRIDGE
        else:
            out[bridge["r0"] : bridge["r1"] + 1, bridge["c0"] : bridge["c1"] + 1] = BRIDGE

    return out


def _bbox_overlaps(
    box_a: tuple[int, int, int, int], box_b: tuple[int, int, int, int], padding: int = 0
) -> bool:
    return not (
        box_a[2] + padding < box_b[0]
        or box_b[2] + padding < box_a[0]
        or box_a[3] + padding < box_b[1]
        or box_b[3] + padding < box_a[1]
    )


def _source_bbox_for_quadrant(
    canonical_bbox: tuple[int, int, int, int], axis_r: int, axis_c: int, quadrant: int
) -> tuple[int, int, int, int]:
    return _reflect_bbox(
        canonical_bbox,
        axis_r,
        axis_c,
        flip_rows=quadrant in (2, 3),
        flip_cols=quadrant in (1, 3),
    )


def _sample_dims(max_h: int, max_w: int) -> list[tuple[int, int]]:
    dims = []
    for h in range(2, min(5, max_h) + 1):
        for w in range(2, min(5, max_w) + 1):
            if h * w >= 6 and max(h, w) >= 3:
                dims.append((h, w))
    random.shuffle(dims)
    return dims


def _canonical_bbox_candidates(axis_r: int, axis_c: int) -> list[tuple[int, int, int, int]]:
    candidates = []
    for h, w in _sample_dims(axis_r - 1, axis_c - 1):
        max_row_gap = axis_r - h
        max_col_gap = axis_c - w
        if max_row_gap < 1 or max_col_gap < 1:
            continue

        modes = ["horizontal", "vertical", "balanced"]
        random.shuffle(modes)
        for mode in modes:
            if mode == "horizontal":
                row_choices = list(range(2, max_row_gap + 1)) or [1]
                col_choices = list(range(1, min(2, max_col_gap) + 1))
            elif mode == "vertical":
                row_choices = list(range(1, min(2, max_row_gap) + 1))
                col_choices = list(range(2, max_col_gap + 1)) or [1]
            else:
                shared = list(range(1, min(max_row_gap, max_col_gap) + 1))
                if not shared:
                    continue
                gap = random.choice(shared)
                row_choices = [gap]
                col_choices = [gap]

            if not row_choices or not col_choices:
                continue

            random.shuffle(row_choices)
            random.shuffle(col_choices)
            for row_gap in row_choices:
                for col_gap in col_choices:
                    r1 = axis_r - 1 - row_gap
                    c1 = axis_c - 1 - col_gap
                    r0 = r1 - h + 1
                    c0 = c1 - w + 1
                    if r0 < 0 or c0 < 0:
                        continue
                    candidates.append((r0, c0, r1, c1))
    random.shuffle(candidates)
    return candidates


def _build_input() -> np.ndarray | None:
    height = random.randrange(15, 22, 2)
    width = random.randrange(15, 22, 2)
    axis_r = height // 2
    axis_c = width // 2
    rect_count = random.choice((2, 3))

    canonical_boxes: list[tuple[int, int, int, int]] = []
    occupied_families: list[list[tuple[int, int, int, int]]] = []
    for candidate in _canonical_bbox_candidates(axis_r, axis_c):
        family_boxes = _family_bboxes(candidate, axis_r, axis_c)
        if any(
            _bbox_overlaps(new_box, old_box, padding=1)
            for new_box in family_boxes
            for old_family in occupied_families
            for old_box in old_family
        ):
            continue
        canonical_boxes.append(candidate)
        occupied_families.append(family_boxes)
        if len(canonical_boxes) == rect_count:
            break
    if len(canonical_boxes) != rect_count:
        return None

    quadrants = random.sample(SOURCE_QUADRANTS, rect_count)
    colors = random.sample(RECT_COLORS, rect_count)

    grid = np.zeros((height, width), dtype=int)
    grid[axis_r, :] = GREEN
    grid[:, axis_c] = GREEN

    for canonical_bbox, quadrant, color in zip(canonical_boxes, quadrants, colors):
        source_bbox = _source_bbox_for_quadrant(canonical_bbox, axis_r, axis_c, quadrant)
        _paint_rect(grid, source_bbox, color)

    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _build_input()
        if grid is None:
            continue
        try:
            out = main(grid)
        except Exception:
            continue
        if np.array_equal(out, grid):
            continue
        if not np.any(out == BRIDGE):
            continue
        return grid
    raise RuntimeError("failed to generate a valid puzzle input")
