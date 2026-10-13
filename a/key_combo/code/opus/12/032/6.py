from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


BLACK = 0
GREEN = 3
YELLOW = 4
TEAL = 8

DIR_ORDER = (
    ("top", "row", -1),
    ("bottom", "row", 1),
    ("left", "col", -1),
    ("right", "col", 1),
)

GEN_RETRIES = 400
GEN_HEIGHT_RANGE = (18, 24)
GEN_WIDTH_RANGE = (18, 24)
GEN_COLORS = (TEAL,)


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _find_components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                pts.append((rr, cc))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _centered_span(lo: int, hi: int, length: int) -> tuple[int, int]:
    center2 = int(lo) + int(hi)
    left = (center2 - (int(length) - 1)) // 2
    right = (center2 + (int(length) - 1)) // 2
    return int(left), int(right)


def _contiguous_span(indices: np.ndarray) -> tuple[int, int]:
    assert indices.size > 0
    lo = int(indices[0])
    hi = int(indices[-1])
    assert np.array_equal(indices, np.arange(lo, hi + 1))
    return lo, hi


def _object_info(coords: np.ndarray) -> dict:
    bbox = _bbox_coords(coords)
    rows: dict[int, tuple[int, int]] = {}
    cols: dict[int, tuple[int, int]] = {}
    for r in np.unique(coords[:, 0]):
        row_cols = np.sort(coords[coords[:, 0] == r, 1])
        rows[int(r)] = (int(row_cols[0]), int(row_cols[-1]))
    for c in np.unique(coords[:, 1]):
        col_rows = np.sort(coords[coords[:, 1] == c, 0])
        cols[int(c)] = (int(col_rows[0]), int(col_rows[-1]))
    r0, c0, r1, c1 = bbox
    return {
        "coords": coords,
        "bbox": bbox,
        "rows": rows,
        "cols": cols,
        "center_key": (r0 + r1, c0 + c1),
    }


def _detect_corridors(grid: np.ndarray, objects: list[dict]) -> tuple[np.ndarray, set[int], set[int], list[tuple]]:
    h, w = grid.shape
    corridor_mask = np.zeros_like(grid, dtype=bool)
    records: list[tuple] = []

    for r in range(h):
        row_items = []
        for idx, obj in enumerate(objects):
            if r in obj["rows"]:
                c0, c1 = obj["rows"][r]
                row_items.append((c0, c1, idx))
        row_items.sort()
        for i in range(len(row_items) - 1):
            left_c0, left_c1, left_idx = row_items[i]
            right_c0, right_c1, right_idx = row_items[i + 1]
            if left_idx == right_idx or left_c1 + 1 >= right_c0:
                continue
            if np.all(grid[r, left_c1 + 1 : right_c0] == BLACK):
                corridor_mask[r, left_c1 + 1 : right_c0] = True
                records.append(("row", int(r), int(left_c1 + 1), int(right_c0 - 1), left_idx, right_idx))

    for c in range(w):
        col_items = []
        for idx, obj in enumerate(objects):
            if c in obj["cols"]:
                r0, r1 = obj["cols"][c]
                col_items.append((r0, r1, idx))
        col_items.sort()
        for i in range(len(col_items) - 1):
            top_r0, top_r1, top_idx = col_items[i]
            bot_r0, bot_r1, bot_idx = col_items[i + 1]
            if top_idx == bot_idx or top_r1 + 1 >= bot_r0:
                continue
            if np.all(grid[top_r1 + 1 : bot_r0, c] == BLACK):
                corridor_mask[top_r1 + 1 : bot_r0, c] = True
                records.append(("col", int(c), int(top_r1 + 1), int(bot_r0 - 1), top_idx, bot_idx))

    corridor_rows = {int(r) for r in np.where(corridor_mask.any(axis=1))[0]}
    corridor_cols = {int(c) for c in np.where(corridor_mask.any(axis=0))[0]}
    return corridor_mask, corridor_rows, corridor_cols, records


def _edge_runs(mask: np.ndarray, bbox: tuple[int, int, int, int]) -> dict[str, tuple[int, int, int]]:
    r0, c0, r1, c1 = bbox
    top_c0, top_c1 = _contiguous_span(np.flatnonzero(mask[r0, :]))
    bot_c0, bot_c1 = _contiguous_span(np.flatnonzero(mask[r1, :]))
    left_r0, left_r1 = _contiguous_span(np.flatnonzero(mask[:, c0]))
    right_r0, right_r1 = _contiguous_span(np.flatnonzero(mask[:, c1]))
    return {
        "top": (r0, top_c0, top_c1),
        "bottom": (r1, bot_c0, bot_c1),
        "left": (c0, left_r0, left_r1),
        "right": (c1, right_r0, right_r1),
    }


def _direction_stats(
    grid_shape: tuple[int, int],
    runs: dict[str, tuple[int, int, int]],
    direction: str,
    corridor_rows: set[int],
    corridor_cols: set[int],
    use_barriers: bool,
) -> dict[str, int | str]:
    axis_name, axis_kind, step = {
        "top": ("top", "row", -1),
        "bottom": ("bottom", "row", 1),
        "left": ("left", "col", -1),
        "right": ("right", "col", 1),
    }[direction]
    source_axis, span_lo, span_hi = runs[axis_name]
    length = int(span_hi) - int(span_lo) + 1
    next_length = length - 2
    distance = 1
    h, w = grid_shape
    while next_length > 0:
        next_axis = int(source_axis) + int(step) * distance
        if axis_kind == "row" and not (0 <= next_axis < h):
            return {"reason": "edge", "steps": distance - 1}
        if axis_kind == "col" and not (0 <= next_axis < w):
            return {"reason": "edge", "steps": distance - 1}
        if use_barriers:
            if axis_kind == "row" and next_axis in corridor_rows:
                return {"reason": "barrier", "steps": distance - 1}
            if axis_kind == "col" and next_axis in corridor_cols:
                return {"reason": "barrier", "steps": distance - 1}
        distance += 1
        next_length -= 2
    return {"reason": "complete", "steps": distance - 1}


def _transform(grid: np.ndarray, *, use_barriers: bool) -> tuple[np.ndarray, dict]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _get_bg(g)
    assert bg == BLACK
    assert not np.any(g == GREEN)
    assert not np.any(g == YELLOW)

    comps = _find_components(g == TEAL)
    assert 3 <= len(comps) <= 4
    objects = [_object_info(coords) for coords in comps]
    objects.sort(key=lambda obj: obj["center_key"])

    corridor_mask, corridor_rows, corridor_cols, corridor_records = _detect_corridors(g, objects)
    out = g.copy()
    out[corridor_mask] = GREEN
    base = out.copy()

    stats = {
        "objects": [],
        "corridor_rows": corridor_rows,
        "corridor_cols": corridor_cols,
        "corridor_records": corridor_records,
    }

    for obj in objects:
        mask = np.zeros_like(g, dtype=bool)
        coords = obj["coords"]
        mask[coords[:, 0], coords[:, 1]] = True
        runs = _edge_runs(mask, obj["bbox"])
        obj_stats = {"bbox": obj["bbox"], "directions": {}}

        for direction, axis_kind, step in DIR_ORDER:
            source_axis, span_lo, span_hi = runs[direction]
            source_len = int(span_hi) - int(span_lo) + 1
            next_len = source_len - 2
            distance = 1
            while next_len > 0:
                next_axis = int(source_axis) + int(step) * distance
                if axis_kind == "row":
                    if not (0 <= next_axis < g.shape[0]):
                        obj_stats["directions"][direction] = {"reason": "edge", "steps": distance - 1}
                        break
                    if use_barriers and next_axis in corridor_rows:
                        obj_stats["directions"][direction] = {"reason": "barrier", "steps": distance - 1}
                        break
                    lo, hi = _centered_span(span_lo, span_hi, next_len)
                    lo = max(0, lo)
                    hi = min(g.shape[1] - 1, hi)
                    if lo <= hi:
                        row = next_axis
                        mask_black = base[row, lo : hi + 1] == BLACK
                        out[row, lo : hi + 1][mask_black] = YELLOW
                else:
                    if not (0 <= next_axis < g.shape[1]):
                        obj_stats["directions"][direction] = {"reason": "edge", "steps": distance - 1}
                        break
                    if use_barriers and next_axis in corridor_cols:
                        obj_stats["directions"][direction] = {"reason": "barrier", "steps": distance - 1}
                        break
                    lo, hi = _centered_span(span_lo, span_hi, next_len)
                    lo = max(0, lo)
                    hi = min(g.shape[0] - 1, hi)
                    if lo <= hi:
                        col = next_axis
                        mask_black = base[lo : hi + 1, col] == BLACK
                        out[lo : hi + 1, col][mask_black] = YELLOW
                distance += 1
                next_len -= 2
            else:
                obj_stats["directions"][direction] = {"reason": "complete", "steps": distance - 1}
        stats["objects"].append(obj_stats)

    return out, stats


def main(grid: np.ndarray) -> np.ndarray:
    """
    Mark unobstructed object-to-object corridors green, then stop each teal object's yellow shrinking extrapolations at those corridor rows and columns.

    Input structure:
    - The grid is a black background containing 3 or 4 separate teal 4-connected rounded objects.
    - Each object has contiguous top, bottom, left, and right edge runs of length at least 3.
    - Some pairs of objects share unobstructed horizontal or vertical black gaps along common rows or columns.

    Rule:
    - First detect every object-to-object row corridor and column corridor whose entire interior span is black, and recolor those corridor cells green.
    - Then, for each teal object in top-to-bottom and left-to-right bbox-center order, extend its top, bottom, left, and right edge runs outward in yellow, shrinking each copied run by 2 cells per step.
    - A direction stops before drawing a step if the next row belongs to any green corridor row, if the next column belongs to any green corridor column, or if that step would leave the grid; yellow is only drawn onto cells that were black after the green corridors were placed.
    - Teal objects and green corridor cells remain unchanged in the output.

    Key insight:
    - The green corridors are not an end result by themselves; their global row and column network acts as a barrier field that selectively truncates otherwise standard shrinking four-way extrapolations.
    """
    out, _stats = _transform(grid, use_barriers=True)
    return out


def _shape_catalog() -> list[np.ndarray]:
    rounded5 = np.array(
        [
            [0, 1, 1, 1, 0],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [0, 1, 1, 1, 0],
        ],
        dtype=int,
    )
    return [
        np.ones((3, 3), dtype=int),
        np.ones((4, 4), dtype=int),
        np.ones((4, 5), dtype=int),
        np.ones((5, 4), dtype=int),
        np.ones((5, 5), dtype=int),
        rounded5,
    ]


def _top_run_len(mask: np.ndarray) -> int:
    return int(np.count_nonzero(mask[0]))


def _bbox_gap_ok(a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int = 1) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return (
        ar1 + gap < br0
        or br1 + gap < ar0
        or ac1 + gap < bc0
        or bc1 + gap < ac0
    )


def _paste_mask(grid: np.ndarray, mask: np.ndarray, top: int, left: int, color: int) -> None:
    h, w = mask.shape
    grid[top : top + h, left : left + w][mask.astype(bool)] = color


def _bbox_from_pos(mask: np.ndarray, top: int, left: int) -> tuple[int, int, int, int]:
    h, w = mask.shape
    return int(top), int(left), int(top + h - 1), int(left + w - 1)


def _random_shape(min_top_run: int = 3) -> np.ndarray:
    choices = [mask for mask in _shape_catalog() if _top_run_len(mask) >= min_top_run]
    return random.choice(choices).copy()


def _build_candidate() -> np.ndarray | None:
    h = random.randint(*GEN_HEIGHT_RANGE)
    w = random.randint(*GEN_WIDTH_RANGE)
    grid = np.zeros((h, w), dtype=int)

    mask_a = _random_shape()
    mask_b = _random_shape()
    mask_c = _random_shape(min_top_run=5)
    extra = random.random() < 0.4
    mask_d = _random_shape() if extra else None

    ha, wa = mask_a.shape
    hb, wb = mask_b.shape
    hc, wc = mask_c.shape

    gap_h = random.randint(1, 3)
    gap_v = 1
    b_top_offset = max(0, ha - hb)
    c_left_offset = wa - wc

    min_left_shift = min(0, c_left_offset)
    total_w = max(wa, wa + gap_h + wb, c_left_offset + wc) - min_left_shift
    total_h = max(ha, b_top_offset + hb, ha + gap_v + hc)
    if total_h + 3 > h or total_w + 3 > w:
        return None

    top_a = random.randint(1, h - total_h - 1)
    left_a = random.randint(1 - min_left_shift, w - total_w - 1)
    top_b = top_a + b_top_offset
    left_b = left_a + wa + gap_h
    top_c = top_a + ha + gap_v
    left_c = left_a + c_left_offset

    placements = [
        (mask_a, top_a, left_a),
        (mask_b, top_b, left_b),
        (mask_c, top_c, left_c),
    ]
    bboxes = [_bbox_from_pos(mask, r, c) for mask, r, c in placements]
    if not all(_bbox_gap_ok(bboxes[i], bboxes[j]) for i in range(3) for j in range(i + 1, 3)):
        return None

    if mask_d is not None:
        hd, wd = mask_d.shape
        for _ in range(60):
            top_d = random.randint(1, h - hd - 1)
            left_d = random.randint(1, w - wd - 1)
            bbox_d = _bbox_from_pos(mask_d, top_d, left_d)
            if all(_bbox_gap_ok(bbox_d, bbox) for bbox in bboxes):
                placements.append((mask_d, top_d, left_d))
                bboxes.append(bbox_d)
                break
        else:
            return None

    for mask, top, left in placements:
        _paste_mask(grid, mask, top, left, TEAL)
    return grid


def _valid_generated(grid: np.ndarray) -> bool:
    try:
        out, stats = _transform(grid, use_barriers=True)
        out_no_barrier, _ = _transform(grid, use_barriers=False)
    except AssertionError:
        return False

    if len(stats["corridor_records"]) < 2:
        return False
    if np.array_equal(out, out_no_barrier):
        return False

    reasons = [d["reason"] for obj in stats["objects"] for d in obj["directions"].values()]
    if "barrier" not in reasons:
        return False
    if "complete" not in reasons:
        return False

    green_count = int(np.count_nonzero(out == GREEN))
    yellow_count = int(np.count_nonzero(out == YELLOW))
    if green_count < 2 or yellow_count < 2:
        return False
    if green_count + yellow_count > (grid.shape[0] * grid.shape[1]) // 3:
        return False

    comps = _find_components(grid == TEAL)
    if len(comps) not in (3, 4):
        return False
    for coords in comps:
        r0, c0, r1, c1 = _bbox_coords(coords)
        mask = np.zeros_like(grid, dtype=bool)
        mask[coords[:, 0], coords[:, 1]] = True
        top = np.flatnonzero(mask[r0, :])
        bot = np.flatnonzero(mask[r1, :])
        left = np.flatnonzero(mask[:, c0])
        right = np.flatnonzero(mask[:, c1])
        if min(len(top), len(bot), len(left), len(right)) < 3:
            return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _build_candidate()
        if grid is not None and _valid_generated(grid):
            return grid
    raise RuntimeError("failed to generate a valid corridor-barrier extrapolation input")
