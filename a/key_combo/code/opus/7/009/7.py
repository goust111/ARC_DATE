from __future__ import annotations

import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_UTILS_DIR) not in sys.path:
    sys.path.append(str(_UTILS_DIR))

from arc_utils import find_components, get_bg


BG = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
ORANGE = 7

LEFT_COLORS = [BLUE, RED, GREEN, YELLOW]
MAX_ATTEMPTS = 400


def _normalized_dims(h: int, w: int) -> tuple[int, int]:
    return min(h, w), max(h, w)


def _find_separator(grid: np.ndarray) -> int:
    candidates = [c for c in range(grid.shape[1]) if np.all(grid[:, c] == GREY)]
    assert len(candidates) == 1, "expected exactly one full-height grey separator"
    return int(candidates[0])


def _extract_rectangles(panel: np.ndarray) -> list[dict]:
    rects = []
    for comp in find_components(panel, bg=get_bg(panel), connectivity=4, monochromatic=True):
        color = int(comp["color"])
        if color not in LEFT_COLORS:
            continue
        r0, c0, r1, c1 = map(int, comp["bbox"])
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        assert int(comp["area"]) == h * w, "left objects must be solid rectangles"
        norm_h, norm_w = _normalized_dims(h, w)
        rects.append(
            {
                "bbox": (r0, c0, r1, c1),
                "color": color,
                "h": h,
                "w": w,
                "norm_h": norm_h,
                "norm_w": norm_w,
            }
        )
    rects.sort(key=lambda item: (item["bbox"][0], item["bbox"][1]))
    assert 3 <= len(rects) <= 4, "expected 3-4 source rectangles"
    return rects


def _frame_from_component(panel: np.ndarray, comp: dict) -> dict | None:
    r0, c0, r1, c1 = map(int, comp["bbox"])
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 4 or w < 3 or int(comp["area"]) != 2 * h + 2 * w - 4:
        return None

    rel = comp["coords"] - np.array([r0, c0], dtype=int)
    mask = np.zeros((h, w), dtype=bool)
    mask[rel[:, 0], rel[:, 1]] = True
    border = np.zeros((h, w), dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True
    if not np.array_equal(mask, border):
        return None

    marker_row = r1 - 1
    marker_cols = [c for c in range(c0 + 1, c1) if int(panel[marker_row, c]) == ORANGE]
    if not marker_cols:
        return None

    return {
        "bbox": (r0, c0, r1, c1),
        "inner_top": r0 + 1,
        "inner_left": c0 + 1,
        "inner_h": h - 2,
        "inner_w": w - 2,
        "marker_row": marker_row,
        "marker_cols": [c - (c0 + 1) for c in marker_cols],
    }


def _extract_frames(panel: np.ndarray) -> list[dict]:
    frames = []
    for comp in find_components(
        panel,
        bg=get_bg(panel),
        connectivity=4,
        monochromatic=True,
        colors=[GREY],
    ):
        frame = _frame_from_component(panel, comp)
        if frame is not None:
            frames.append(frame)
    frames.sort(key=lambda item: (item["bbox"][0], item["bbox"][1]))
    assert 3 <= len(frames) <= 4, "expected 3-4 frame slots"
    return frames


def _pair_indices_by_width(widths: list[int], frame_widths: list[int]) -> list[int] | None:
    queues: dict[int, list[int]] = defaultdict(list)
    for idx, width in enumerate(frame_widths):
        queues[int(width)].append(idx)

    perm = []
    for width in widths:
        bucket = queues[int(width)]
        if not bucket:
            return None
        perm.append(bucket.pop(0))

    if any(bucket for bucket in queues.values()):
        return None
    return perm


def main(input_grid: np.ndarray) -> np.ndarray:
    """
    Rotate scattered rectangles to their horizontal form, then embed each into the frame slot with the matching inner width.

    Input structure:
    - A black background grid is split by a full-height grey separator into a left source panel and a right slot panel.
    - The left panel contains 3-4 solid colored rectangles in blue, red, green, or yellow, with mixed horizontal and vertical orientations.
    - The right panel contains 3-4 grey hollow rectangular frames; each frame has orange marker pixels on the interior bottom row.

    Rule:
    - Normalize every left-panel rectangle to a horizontal rectangle: if its height exceeds its width, rotate it so the new size is min(h,w) by max(h,w).
    - Measure each frame by its inner width and match rectangles to frames with the same normalized width; when several rectangles share a width, preserve their top-to-bottom order and pair them with same-width frames in their top-to-bottom order.
    - Clear the left panel to background, then place each normalized rectangle into its matched frame with left edges aligned and with its bottom edge one row above the orange marker row.
    - Keep the separator, frames, and orange markers unchanged.

    Key insight:
    - Vertical source rectangles must be reinterpreted by their post-rotation width; their original narrow width does not identify the correct frame.
    """
    grid = np.asarray(input_grid, dtype=int)
    sep = _find_separator(grid)
    left = grid[:, :sep]
    right = grid[:, sep + 1 :]

    rects = _extract_rectangles(left)
    frames = _extract_frames(right)
    assert len(rects) == len(frames), "rectangle/frame counts must match"

    perm = _pair_indices_by_width(
        [rect["norm_w"] for rect in rects],
        [frame["inner_w"] for frame in frames],
    )
    assert perm is not None, "normalized widths must match frame inner widths"

    out = grid.copy()
    out[:, :sep] = BG

    for rect, frame_idx in zip(rects, perm):
        frame = frames[frame_idx]
        assert rect["norm_w"] == frame["inner_w"]
        assert rect["norm_h"] + 1 <= frame["inner_h"]

        top = int(frame["marker_row"] - rect["norm_h"])
        left_col = int(sep + 1 + frame["inner_left"])
        out[top : top + rect["norm_h"], left_col : left_col + rect["norm_w"]] = rect["color"]

    return out


def _random_partition(total: int, parts: int) -> list[int]:
    values = [0] * parts
    for _ in range(total):
        values[random.randrange(parts)] += 1
    return values


def _stack_positions(heights: list[int], total_h: int, min_gap: int = 1) -> list[int]:
    base = sum(heights) + min_gap * (len(heights) - 1)
    if base > total_h:
        raise ValueError("objects do not fit vertically")

    extras = _random_partition(total_h - base, len(heights) + 1)
    tops = []
    cur = extras[0]
    for idx, height in enumerate(heights):
        tops.append(cur)
        cur += height
        if idx != len(heights) - 1:
            cur += min_gap + extras[idx + 1]
    return tops


def _sample_norm_widths(count: int) -> list[int]:
    choices = [4, 5, 6, 7]
    while True:
        if random.random() < 0.6:
            distinct = random.sample(choices, random.randint(2, min(3, count)))
            widths = [random.choice(distinct) for _ in range(count)]
            if len(set(widths)) == count:
                continue
        else:
            widths = random.sample(choices, count)
        if len(set(widths)) >= 2:
            random.shuffle(widths)
            return widths


def _sample_marker_cols(inner_w: int) -> list[int]:
    size = random.randint(1, inner_w)
    return sorted(random.sample(range(inner_w), size))


def _render_input(spec: dict) -> np.ndarray:
    grid = np.full((spec["H"], spec["W"]), BG, dtype=int)
    grid[:, spec["sep"]] = GREY

    for rect in spec["rects"]:
        grid[
            rect["top"] : rect["top"] + rect["h"],
            rect["left"] : rect["left"] + rect["w"],
        ] = rect["color"]

    for frame in spec["frames"]:
        r0 = frame["top"]
        c0 = spec["sep"] + 1 + frame["left"]
        h = frame["outer_h"]
        w = frame["outer_w"]
        grid[r0, c0 : c0 + w] = GREY
        grid[r0 + h - 1, c0 : c0 + w] = GREY
        grid[r0 : r0 + h, c0] = GREY
        grid[r0 : r0 + h, c0 + w - 1] = GREY
        marker_row = r0 + h - 2
        for rel_c in frame["marker_cols"]:
            grid[marker_row, c0 + 1 + rel_c] = ORANGE

    return grid


def _render_output(spec: dict) -> np.ndarray:
    out = np.full((spec["H"], spec["W"]), BG, dtype=int)
    out[:, spec["sep"]] = GREY

    for frame in spec["frames"]:
        r0 = frame["top"]
        c0 = spec["sep"] + 1 + frame["left"]
        h = frame["outer_h"]
        w = frame["outer_w"]
        out[r0, c0 : c0 + w] = GREY
        out[r0 + h - 1, c0 : c0 + w] = GREY
        out[r0 : r0 + h, c0] = GREY
        out[r0 : r0 + h, c0 + w - 1] = GREY
        marker_row = r0 + h - 2
        for rel_c in frame["marker_cols"]:
            out[marker_row, c0 + 1 + rel_c] = ORANGE

    perm = _pair_indices_by_width(
        [rect["norm_w"] for rect in spec["rects"]],
        [frame["inner_w"] for frame in spec["frames"]],
    )
    assert perm is not None

    for rect, frame_idx in zip(spec["rects"], perm):
        frame = spec["frames"][frame_idx]
        top = frame["top"] + frame["outer_h"] - 2 - rect["norm_h"]
        left = spec["sep"] + 1 + frame["left"] + 1
        out[top : top + rect["norm_h"], left : left + rect["norm_w"]] = rect["color"]

    return out


def _sample_spec() -> dict | None:
    count = random.randint(3, 4)
    colors = random.sample(LEFT_COLORS, count)
    norm_widths = _sample_norm_widths(count)
    norm_heights = [random.randint(1, 3) for _ in range(count)]

    vertical_flags = [True, False] + [random.choice([True, False]) for _ in range(count - 2)]
    random.shuffle(vertical_flags)
    vertical_flags = vertical_flags[:count]
    if not any(vertical_flags) or not any(not flag for flag in vertical_flags):
        return None

    rects = []
    for idx in range(count):
        short = norm_heights[idx]
        long = norm_widths[idx]
        vertical = vertical_flags[idx]
        h, w = (long, short) if vertical else (short, long)
        rects.append(
            {
                "color": colors[idx],
                "h": h,
                "w": w,
                "norm_h": short,
                "norm_w": long,
                "vertical": vertical,
            }
        )

    width_to_max_height: dict[int, int] = defaultdict(int)
    width_counts: Counter[int] = Counter(norm_widths)
    for rect in rects:
        width_to_max_height[rect["norm_w"]] = max(
            width_to_max_height[rect["norm_w"]],
            rect["norm_h"],
        )

    frames = []
    for width in norm_widths:
        base_h = width_to_max_height[width] if width_counts[width] > 1 else 0
        needed_h = max(base_h, next(rect["norm_h"] for rect in rects if rect["norm_w"] == width))
        inner_h = needed_h + 1 + random.randint(0, 2)
        frames.append(
            {
                "inner_w": width,
                "inner_h": inner_h,
                "outer_w": width + 2,
                "outer_h": inner_h + 2,
                "marker_cols": _sample_marker_cols(width),
            }
        )

    random.shuffle(frames)
    perm = _pair_indices_by_width(
        [rect["norm_w"] for rect in rects],
        [frame["inner_w"] for frame in frames],
    )
    if perm is None or perm == list(range(count)):
        return None

    left_w = max(rect["w"] for rect in rects) + random.randint(2, 4)
    right_w = max(frame["outer_w"] for frame in frames) + random.randint(3, 5)
    H_min = max(
        sum(rect["h"] for rect in rects) + (count - 1),
        sum(frame["outer_h"] for frame in frames) + (count - 1),
    )
    H = H_min + random.randint(0, 4)
    W = left_w + 1 + right_w
    if H > 30 or W > 30:
        return None

    rect_tops = _stack_positions([rect["h"] for rect in rects], H, min_gap=1)
    frame_tops = _stack_positions([frame["outer_h"] for frame in frames], H, min_gap=1)

    for rect, top in zip(rects, rect_tops):
        rect["top"] = top
        rect["left"] = random.randint(0, left_w - rect["w"])

    for frame, top in zip(frames, frame_tops):
        frame["top"] = top
        frame["left"] = random.randint(1, right_w - frame["outer_w"])

    return {"H": H, "W": W, "sep": left_w, "rects": rects, "frames": frames}


def _spec_is_semantic(spec: dict) -> bool:
    rects = spec["rects"]
    frames = spec["frames"]
    norm_widths = [rect["norm_w"] for rect in rects]
    frame_widths = [frame["inner_w"] for frame in frames]
    raw_widths = [rect["w"] for rect in rects]

    if sorted(norm_widths) != sorted(frame_widths):
        return False
    if sorted(raw_widths) == sorted(frame_widths):
        return False
    if not any(rect["vertical"] for rect in rects):
        return False
    if not any(not rect["vertical"] for rect in rects):
        return False
    if any(rect["w"] in set(frame_widths) for rect in rects if rect["vertical"]):
        return False

    perm = _pair_indices_by_width(norm_widths, frame_widths)
    if perm is None or perm == list(range(len(rects))):
        return False

    grid = _render_input(spec)
    expected = _render_output(spec)
    if int(get_bg(grid)) != BG:
        return False
    if np.array_equal(grid, expected):
        return False
    if not np.array_equal(main(grid), expected):
        return False
    if max(grid.shape) > 30 or max(expected.shape) > 30:
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        spec = _sample_spec()
        if spec is None:
            continue
        if _spec_is_semantic(spec):
            return _render_input(spec)
    raise RuntimeError(f"failed to generate a valid input after {MAX_ATTEMPTS} attempts")
