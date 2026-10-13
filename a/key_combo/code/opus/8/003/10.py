from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import find_components, get_bg


PALETTE = (1, 2, 3, 4, 6)
SEPARATOR_COLOR = 5
ARC_MAX = 30
GEN_RETRY_LIMIT = 400
LOCAL_RETRY_LIMIT = 200


def _rect_from_dims(color: int, r: int, c: int, short: int, long: int, vertical: bool) -> dict:
    input_h, input_w = (long, short) if vertical else (short, long)
    return {
        "color": int(color),
        "r": int(r),
        "c": int(c),
        "short": int(short),
        "long": int(long),
        "vertical": bool(vertical),
        "input_h": int(input_h),
        "input_w": int(input_w),
        "norm_h": int(short),
        "norm_w": int(long),
    }


def _bbox(rect: dict, mode: str) -> tuple[int, int, int, int]:
    h = rect["input_h"] if mode == "input" else rect["norm_h"]
    w = rect["input_w"] if mode == "input" else rect["norm_w"]
    return rect["r"], rect["c"], rect["r"] + h - 1, rect["c"] + w - 1


def _boxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _fits(rect: dict, shape: tuple[int, int], mode: str) -> bool:
    h, w = shape
    r0, c0, r1, c1 = _bbox(rect, mode)
    return 0 <= r0 <= r1 < h and 0 <= c0 <= c1 < w


def _can_add(rect: dict, rects: list[dict], shape: tuple[int, int]) -> bool:
    if not _fits(rect, shape, "input") or not _fits(rect, shape, "norm"):
        return False
    for other in rects:
        if _boxes_overlap(_bbox(rect, "input"), _bbox(other, "input")):
            return False
        if _boxes_overlap(_bbox(rect, "norm"), _bbox(other, "norm")):
            return False
    return True


def _render_rectangles(rects: list[dict], shape: tuple[int, int], mode: str, bg: int = 0) -> np.ndarray:
    out = np.full(shape, int(bg), dtype=int)
    occupied = np.zeros(shape, dtype=bool)
    h_key = "input_h" if mode == "input" else "norm_h"
    w_key = "input_w" if mode == "input" else "norm_w"
    for rect in sorted(rects, key=lambda x: (x["r"], x["c"], x["color"])):
        r0, c0 = rect["r"], rect["c"]
        hh, ww = rect[h_key], rect[w_key]
        mask = np.zeros(shape, dtype=bool)
        mask[r0 : r0 + hh, c0 : c0 + ww] = True
        assert not np.any(occupied & mask)
        out[mask] = rect["color"]
        occupied |= mask
    return out


def _split_panels(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    h, w = g.shape
    assert w % 2 == 1 and max(g.shape) <= ARC_MAX

    sep = w // 2
    sep_color = int(g[0, sep])
    assert sep_color != bg and np.all(g[:, sep] == sep_color)
    left = g[:, :sep]
    right = g[:, sep + 1 :]
    assert left.shape == right.shape
    return left, right, bg


def _extract_rectangles(panel: np.ndarray, bg: int) -> list[dict]:
    components = find_components(panel, bg=bg, connectivity=4, return_format="dict")
    rects: list[dict] = []
    for obj in components:
        r0, c0, r1, c1 = obj["bbox"]
        hh = r1 - r0 + 1
        ww = c1 - c0 + 1
        color = int(obj["color"])
        assert int(obj["area"]) == hh * ww
        assert np.all(panel[r0 : r1 + 1, c0 : c1 + 1] == color)
        short = min(hh, ww)
        long = max(hh, ww)
        rects.append(
            {
                "color": color,
                "r": int(r0),
                "c": int(c0),
                "input_h": int(hh),
                "input_w": int(ww),
                "norm_h": int(short),
                "norm_w": int(long),
            }
        )
    return sorted(rects, key=lambda x: (x["r"], x["c"], x["color"]))


def _render_normalized(rects: list[dict], shape: tuple[int, int], bg: int) -> np.ndarray:
    out = np.full(shape, int(bg), dtype=int)
    occupied = np.zeros(shape, dtype=bool)
    for rect in rects:
        r0, c0 = rect["r"], rect["c"]
        hh, ww = rect["norm_h"], rect["norm_w"]
        mask = np.zeros(shape, dtype=bool)
        mask[r0 : r0 + hh, c0 : c0 + ww] = True
        assert not np.any(occupied & mask)
        out[mask] = rect["color"]
        occupied |= mask
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Normalize each panel's rectangles to horizontal and overlay only the non-conflicting right-panel rectangles onto the left panel.

    Input structure:
    - The grid has black background and a solid grey separator column splitting it into two equal-size side panels.
    - Each panel contains 2-4 non-overlapping solid monochrome rectangles in colors chosen from the non-grey palette.
    - Some rectangles are tall (height > width), so rotating them to horizontal changes their occupied footprint.

    Rule:
    - Remove the separator and treat each panel independently.
    - In both panels, replace every rectangle by its left-top-anchored horizontal form: keep it unchanged when width >= height, otherwise swap height and width so the rectangle expands rightward instead of downward.
    - Use the normalized left panel as the output canvas.
    - For each normalized right-panel rectangle, if any of its cells overlap normalized left-panel foreground cells at the same coordinates, discard it; otherwise copy it onto the canvas in its own color.

    Key insight:
    - Overlap must be tested after orientation normalization, because swapping a tall rectangle's height and width changes which coordinates are occupied.
    """
    left, right, bg = _split_panels(np.asarray(grid, dtype=int))
    shape = left.shape

    left_rects = _extract_rectangles(left, bg)
    right_rects = _extract_rectangles(right, bg)

    out = _render_normalized(left_rects, shape, bg)
    base_mask = out != bg
    added_mask = np.zeros(shape, dtype=bool)
    for rect in right_rects:
        r0, c0 = rect["r"], rect["c"]
        hh, ww = rect["norm_h"], rect["norm_w"]
        mask = np.zeros(shape, dtype=bool)
        mask[r0 : r0 + hh, c0 : c0 + ww] = True
        if np.any(base_mask & mask):
            continue
        assert not np.any(added_mask & mask)
        out[mask] = rect["color"]
        added_mask |= mask
    return out


def _sample_dims() -> tuple[int, int]:
    short = random.randint(2, 4)
    long = random.randint(max(4, short + 1), 6)
    return short, long


def _sample_free_rect(
    shape: tuple[int, int],
    color: int,
    rects: list[dict],
    *,
    vertical: bool | None = None,
    require_conflict: bool | None = None,
    left_norm_boxes: list[tuple[int, int, int, int]] | None = None,
) -> dict | None:
    height, width = shape
    for _ in range(LOCAL_RETRY_LIMIT):
        short, long = _sample_dims()
        use_vertical = random.choice((False, True)) if vertical is None else bool(vertical)
        rect = _rect_from_dims(color, 0, 0, short, long, use_vertical)
        max_r = height - rect["input_h"]
        max_c = width - rect["input_w"]
        if max_r < 0 or max_c < 0:
            continue
        rect["r"] = random.randint(0, max_r)
        rect["c"] = random.randint(0, max_c)
        if not _can_add(rect, rects, shape):
            continue
        if require_conflict is not None:
            assert left_norm_boxes is not None
            conflict = any(_boxes_overlap(_bbox(rect, "norm"), box) for box in left_norm_boxes)
            if conflict != require_conflict:
                continue
        return rect
    return None


def _build_case() -> np.ndarray | None:
    panel_h = random.randint(10, 13)
    panel_w = random.randint(10, 14)
    if max(panel_h, 2 * panel_w + 1) > ARC_MAX:
        return None

    special_short, special_long = _sample_dims()
    anchor_short, anchor_long = _sample_dims()
    shape = (panel_h, panel_w)

    anchor_c_min = special_short
    anchor_c_max = min(panel_w - anchor_long, panel_w - (special_long - special_short))
    anchor_r_max = panel_h - max(anchor_short, special_long)
    if anchor_c_min > anchor_c_max or anchor_r_max < 0:
        return None

    left_colors = random.sample(PALETTE, random.randint(2, 4))
    right_colors = random.sample(PALETTE, random.randint(2, 4))

    anchor_r = random.randint(0, anchor_r_max)
    anchor_c = random.randint(anchor_c_min, anchor_c_max)
    left_rects = [_rect_from_dims(left_colors[0], anchor_r, anchor_c, anchor_short, anchor_long, False)]

    special_rect = _rect_from_dims(
        right_colors[0],
        anchor_r,
        anchor_c - special_short,
        special_short,
        special_long,
        True,
    )
    if not _fits(special_rect, shape, "input") or not _fits(special_rect, shape, "norm"):
        return None

    reserved_input_box = _bbox(special_rect, "input")
    for _ in range(LOCAL_RETRY_LIMIT):
        target_left = random.randint(2, min(4, len(left_colors)))
        trial = list(left_rects)
        ok = True
        need_vertical = target_left >= 3
        for idx in range(1, target_left):
            force_vertical = True if need_vertical and idx == 1 else None
            rect = _sample_free_rect(shape, left_colors[idx], trial, vertical=force_vertical)
            if rect is None:
                ok = False
                break
            if _boxes_overlap(_bbox(rect, "norm"), reserved_input_box):
                ok = False
                break
            trial.append(rect)
        if ok:
            left_rects = trial
            break
    else:
        return None

    left_norm_boxes = [_bbox(rect, "norm") for rect in left_rects]
    special_input_conflict = any(_boxes_overlap(_bbox(special_rect, "input"), box) for box in left_norm_boxes)
    special_norm_conflict = any(_boxes_overlap(_bbox(special_rect, "norm"), box) for box in left_norm_boxes)
    if special_input_conflict or not special_norm_conflict:
        return None

    right_rects = [special_rect]
    accepted_rect = _sample_free_rect(
        shape,
        right_colors[1],
        right_rects,
        require_conflict=False,
        left_norm_boxes=left_norm_boxes,
    )
    if accepted_rect is None:
        return None
    right_rects.append(accepted_rect)

    target_right = random.randint(2, min(4, len(right_colors)))
    for idx in range(2, target_right):
        require_conflict = random.choice((False, True))
        rect = _sample_free_rect(
            shape,
            right_colors[idx],
            right_rects,
            require_conflict=require_conflict,
            left_norm_boxes=left_norm_boxes,
        )
        if rect is None:
            return None
        right_rects.append(rect)

    left_panel = _render_rectangles(left_rects, shape, "input", bg=0)
    right_panel = _render_rectangles(right_rects, shape, "input", bg=0)
    grid = np.full((panel_h, panel_w * 2 + 1), 0, dtype=int)
    grid[:, :panel_w] = left_panel
    grid[:, panel_w] = SEPARATOR_COLOR
    grid[:, panel_w + 1 :] = right_panel

    out = main(grid)
    if out.shape != shape:
        return None

    right_conflicts = [any(_boxes_overlap(_bbox(rect, "norm"), box) for box in left_norm_boxes) for rect in right_rects]
    if not any(right_conflicts) or not any(not flag for flag in right_conflicts):
        return None
    changed = [
        any(_boxes_overlap(_bbox(rect, "input"), box) for box in left_norm_boxes)
        != any(_boxes_overlap(_bbox(rect, "norm"), box) for box in left_norm_boxes)
        for rect in right_rects
    ]
    if not any(changed):
        return None

    expected = _render_normalized(_extract_rectangles(left_panel, 0), shape, 0)
    for rect in _extract_rectangles(right_panel, 0):
        r0, c0 = rect["r"], rect["c"]
        hh, ww = rect["norm_h"], rect["norm_w"]
        region = expected[r0 : r0 + hh, c0 : c0 + ww]
        if np.any(region != 0):
            continue
        expected[r0 : r0 + hh, c0 : c0 + ww] = rect["color"]
    if not np.array_equal(out, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        grid = _build_case()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
