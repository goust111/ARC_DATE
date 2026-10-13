from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import bbox_h, bbox_w, crop_bbox, find_components, get_bg, paste


BG = 0
COLOR_POOL = (1, 2, 3, 4, 5, 6, 7, 8, 9)
GRID_SIDE_RANGE = (15, 23)
FRAME_COUNT_CHOICES = (3, 4)
GEN_RETRIES = 500
PLACEMENT_RETRIES = 200
SIZE_RETRIES = 200


def _frame_patch(height: int, width: int, color: int, bg: int = BG) -> np.ndarray:
    patch = np.full((int(height), int(width)), int(bg), dtype=int)
    patch[0, :] = int(color)
    patch[-1, :] = int(color)
    patch[:, 0] = int(color)
    patch[:, -1] = int(color)
    return patch


def _is_rect_frame(patch: np.ndarray, color: int, bg: int) -> bool:
    height, width = map(int, patch.shape)
    if height < 3 or width < 3:
        return False
    if not np.all((patch == int(bg)) | (patch == int(color))):
        return False
    if not np.all(patch[0, :] == int(color)):
        return False
    if not np.all(patch[-1, :] == int(color)):
        return False
    if not np.all(patch[:, 0] == int(color)):
        return False
    if not np.all(patch[:, -1] == int(color)):
        return False
    if height > 2 and width > 2 and not np.all(patch[1:-1, 1:-1] == int(bg)):
        return False
    return True


def _detect_frames(grid: np.ndarray) -> tuple[int, list[dict]]:
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    comps = find_components(g, bg=bg, connectivity=4, monochromatic=True, return_format="dict")

    frames: list[dict] = []
    for comp in comps:
        color = int(comp["color"])
        box = tuple(map(int, comp["bbox"]))
        patch = crop_bbox(g, box)
        height = int(bbox_h(box))
        width = int(bbox_w(box))
        perimeter = 2 * (height + width) - 4
        if int(comp["area"]) != perimeter:
            continue
        if not _is_rect_frame(patch, color, bg):
            continue
        frames.append(
            {
                "color": color,
                "bbox": box,
                "r0": int(box[0]),
                "c0": int(box[1]),
                "r1": int(box[2]),
                "c1": int(box[3]),
                "h": height,
                "w": width,
                "area": int(height * width),
            }
        )

    return bg, frames


def _sorted_frames(frames: list[dict]) -> list[dict]:
    return sorted(
        frames,
        key=lambda frame: (-int(frame["area"]), -int(frame["h"]), -int(frame["w"]), int(frame["r0"]), int(frame["c0"])),
    )


def _nested_positions(
    frames: list[dict],
    *,
    anchor: str = "lower_left",
) -> list[dict]:
    assert frames, "expected at least one frame"
    placed: list[dict] = []
    for i, frame in enumerate(frames):
        if i == 0:
            r0 = int(frame["r0"])
            c0 = int(frame["c0"])
        else:
            prev = placed[-1]
            if anchor == "lower_left":
                r0 = int(prev["r1"]) - int(frame["h"])
                c0 = int(prev["c0"]) + 1
            elif anchor == "upper_left":
                r0 = int(prev["r0"]) + 1
                c0 = int(prev["c0"]) + 1
            else:
                raise ValueError(f"unknown anchor {anchor}")
        placed.append(
            {
                "color": int(frame["color"]),
                "h": int(frame["h"]),
                "w": int(frame["w"]),
                "area": int(frame["area"]),
                "r0": int(r0),
                "c0": int(c0),
                "r1": int(r0 + int(frame["h"]) - 1),
                "c1": int(c0 + int(frame["w"]) - 1),
            }
        )
    return placed


def _render_positions(shape: tuple[int, int], positions: list[dict], bg: int) -> np.ndarray:
    out = np.full(shape, int(bg), dtype=int)
    height, width = map(int, shape)
    for frame in positions:
        if not (0 <= int(frame["r0"]) and 0 <= int(frame["c0"]) and int(frame["r1"]) < height and int(frame["c1"]) < width):
            raise ValueError("nested frame placement left the canvas")
        patch = _frame_patch(int(frame["h"]), int(frame["w"]), int(frame["color"]), bg)
        paste(out, patch, int(frame["r0"]), int(frame["c0"]), transparent=int(bg))
    return out


def _nest_frames(
    shape: tuple[int, int],
    frames: list[dict],
    bg: int,
    *,
    anchor: str = "lower_left",
) -> tuple[np.ndarray, list[dict]]:
    positions = _nested_positions(frames, anchor=anchor)
    return _render_positions(shape, positions, bg), positions


def _row_endpoint_stats(grid: np.ndarray, bg: int) -> dict:
    same_rows = 0
    diff_rows = 0
    fillable_rows = 0
    for row in range(int(grid.shape[0])):
        cols = np.flatnonzero(grid[row] != int(bg))
        if cols.size == 0:
            continue
        left = int(cols[0])
        right = int(cols[-1])
        left_color = int(grid[row, left])
        right_color = int(grid[row, right])
        if left_color == right_color:
            same_rows += 1
            if not np.all(grid[row, left : right + 1] == left_color):
                fillable_rows += 1
        else:
            diff_rows += 1
    return {"same_rows": same_rows, "diff_rows": diff_rows, "fillable_rows": fillable_rows}


def _fill_same_end_rows(grid: np.ndarray, bg: int) -> np.ndarray:
    out = np.array(grid, dtype=int, copy=True)
    for row in range(int(out.shape[0])):
        cols = np.flatnonzero(out[row] != int(bg))
        if cols.size == 0:
            continue
        left = int(cols[0])
        right = int(cols[-1])
        left_color = int(out[row, left])
        right_color = int(out[row, right])
        if left_color == right_color:
            out[row, left : right + 1] = left_color
    return out


def _oracle_output(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg, frames = _detect_frames(g)
    assert 3 <= len(frames) <= 4, "expected 3 or 4 rectangular frames"
    ranked = _sorted_frames(frames)
    nested, _ = _nest_frames(g.shape, ranked, bg, anchor="lower_left")
    return _fill_same_end_rows(nested, bg)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reorder hollow rectangular frames by area, dock them by lower-left inner-corner alignment, then fill rows whose visible endpoints share a color.

    Input structure:
    - A black background contains 3 or 4 non-overlapping single-color hollow rectangular frames.
    - Each foreground object is a one-pixel-thick empty rectangle border with a unique color.
    - The frames are initially scattered and their bounding-box areas are all distinct.

    Rule:
    - Detect every rectangular frame and compute its bounding-box area `height x width`.
    - Sort the frames from largest area to smallest area.
    - Copy the largest frame unchanged into the output.
    - For each later frame, place it relative to the previous placed frame so its left border touches the previous frame's inner left wall and its bottom border touches the previous frame's inner bottom wall.
    - After all frames are redrawn in that order, scan each row of the result; if the leftmost and rightmost non-background pixels on that row have the same color, fill the entire interval between them with that color.
    - Leave rows with different endpoint colors, or no foreground at all, unchanged.

    Key insight:
    - The decisive structure is not the input scatter but the area-sorted lower-left docking arrangement, because that redraw changes which frame colors become the two visible row endpoints and therefore which rows are eligible for filling.
    """
    return _oracle_output(grid)


def _rects_separate(rects: list[tuple[int, int, int, int]], candidate: tuple[int, int, int, int]) -> bool:
    r0, c0, r1, c1 = candidate
    for rr0, cc0, rr1, cc1 in rects:
        if not (r1 < rr0 - 1 or rr1 < r0 - 1 or c1 < cc0 - 1 or cc1 < c0 - 1):
            return False
    return True


def _sample_sizes(frame_count: int) -> list[tuple[int, int]] | None:
    for _ in range(SIZE_RETRIES):
        h0 = random.randint(7, 10)
        w0 = random.randint(6, 9)
        sizes = [(h0, w0)]
        areas = {h0 * w0}

        h1_choices = [h for h in range(4, h0 - 1)]
        w1_choices = [w for w in range(w0, min(12, w0 + 2) + 1)]
        second = [(h, w) for h in h1_choices for w in w1_choices if h * w < h0 * w0 and h * w not in areas]
        if not second:
            continue
        h1, w1 = random.choice(second)
        sizes.append((h1, w1))
        areas.add(h1 * w1)

        if frame_count >= 3:
            third_candidates = []
            for h in range(3, min(11, h1 + 1) + 1):
                for w in range(4, min(12, w1 + 2) + 1):
                    area = h * w
                    if area >= h1 * w1 or area in areas:
                        continue
                    third_candidates.append((h, w))
            if not third_candidates:
                continue
            h2, w2 = random.choice(third_candidates)
            sizes.append((h2, w2))
            areas.add(h2 * w2)

        if frame_count >= 4:
            fourth_candidates = []
            prev_h, prev_w = sizes[-1]
            prev_area = prev_h * prev_w
            for h in range(3, min(10, prev_h + 1) + 1):
                for w in range(4, min(12, prev_w + 2) + 1):
                    area = h * w
                    if area >= prev_area or area in areas:
                        continue
                    fourth_candidates.append((h, w))
            if not fourth_candidates:
                continue
            sizes.append(random.choice(fourth_candidates))

        widths = [w for _, w in sizes]
        heights = [h for h, _ in sizes]
        protrudes_right = any(width + i > widths[0] for i, width in enumerate(widths[1:], start=1))
        leaves_same_rows = any(heights[0] > height + i for i, height in enumerate(heights[1:], start=1))
        if protrudes_right and leaves_same_rows:
            return sizes
    return None


def _choose_canvas_and_anchor(sizes: list[tuple[int, int]]) -> tuple[int, int, int, int] | None:
    max_height_need = max(i + h for i, (h, _w) in enumerate(sizes))
    max_width_need = max(i + w for i, (_h, w) in enumerate(sizes))
    min_side = max(GRID_SIDE_RANGE[0], max_height_need, max_width_need)
    max_side = GRID_SIDE_RANGE[1]
    if min_side > max_side:
        return None

    height = random.randint(min_side, max_side)
    width = random.randint(min_side, max_side)

    bottom_min = max(i + h - 1 for i, (h, _w) in enumerate(sizes))
    left_max = width - max(i + w for i, (_h, w) in enumerate(sizes))
    if left_max < 0:
        return None

    bottom = random.randint(bottom_min, height - 1)
    left = random.randint(0, left_max)
    top = bottom - sizes[0][0] + 1
    return height, width, top, left


def _scatter_positions(
    shape: tuple[int, int],
    sizes: list[tuple[int, int]],
    largest_origin: tuple[int, int],
    nested_positions: list[dict],
) -> list[tuple[int, int]] | None:
    height, width = map(int, shape)
    positions = [largest_origin]
    rects = [
        (
            int(largest_origin[0]),
            int(largest_origin[1]),
            int(largest_origin[0] + sizes[0][0] - 1),
            int(largest_origin[1] + sizes[0][1] - 1),
        )
    ]

    for index in range(1, len(sizes)):
        frame_h, frame_w = sizes[index]
        target = (int(nested_positions[index]["r0"]), int(nested_positions[index]["c0"]))
        candidates = [
            (r0, c0)
            for r0 in range(0, height - frame_h + 1)
            for c0 in range(0, width - frame_w + 1)
            if (r0, c0) != target
        ]
        random.shuffle(candidates)

        picked = None
        for r0, c0 in candidates:
            rect = (r0, c0, r0 + frame_h - 1, c0 + frame_w - 1)
            if not _rects_separate(rects, rect):
                continue
            picked = (r0, c0)
            rects.append(rect)
            positions.append(picked)
            break
        if picked is None:
            return None

    return positions


def _render_input(shape: tuple[int, int], sizes: list[tuple[int, int]], colors: list[int], positions: list[tuple[int, int]]) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    for (height, width), color, (r0, c0) in zip(sizes, colors, positions):
        patch = _frame_patch(height, width, color, BG)
        paste(grid, patch, int(r0), int(c0), transparent=BG)
    return grid


def _alternative_output(
    frames: list[dict],
    shape: tuple[int, int],
    bg: int,
    *,
    sort_desc: bool,
    anchor: str,
) -> np.ndarray | None:
    ordered = _sorted_frames(frames) if sort_desc else list(reversed(_sorted_frames(frames)))
    try:
        nested, _ = _nest_frames(shape, ordered, bg, anchor=anchor)
    except Exception:
        return None
    return _fill_same_end_rows(nested, bg)


def _semantic_checks(grid: np.ndarray) -> bool:
    g = np.asarray(grid, dtype=int)
    bg, frames = _detect_frames(g)
    if len(frames) not in (3, 4):
        return False
    if len({int(frame["area"]) for frame in frames}) != len(frames):
        return False

    ranked = _sorted_frames(frames)
    try:
        nested, positions = _nest_frames(g.shape, ranked, bg, anchor="lower_left")
    except Exception:
        return False
    stats = _row_endpoint_stats(nested, bg)
    if stats["same_rows"] < 2 or stats["diff_rows"] < 1 or stats["fillable_rows"] < 2:
        return False

    out = _fill_same_end_rows(nested, bg)
    if np.array_equal(out, nested):
        return False
    if not np.array_equal(main(g), out):
        return False

    direct_fill = _fill_same_end_rows(g, bg)
    if np.array_equal(direct_fill, out):
        return False

    alt_sort = _alternative_output(frames, g.shape, bg, sort_desc=False, anchor="lower_left")
    if alt_sort is not None and np.array_equal(alt_sort, out):
        return False

    alt_anchor = _alternative_output(frames, g.shape, bg, sort_desc=True, anchor="upper_left")
    if alt_anchor is not None and np.array_equal(alt_anchor, out):
        return False

    for index in range(1, len(positions)):
        if positions[index]["c1"] <= positions[index - 1]["c1"]:
            continue
        break
    else:
        return False

    return True


def _generate_one() -> np.ndarray | None:
    frame_count = random.choice(FRAME_COUNT_CHOICES)
    sizes = _sample_sizes(frame_count)
    if sizes is None:
        return None

    canvas = _choose_canvas_and_anchor(sizes)
    if canvas is None:
        return None
    height, width, top, left = canvas

    colors = random.sample(COLOR_POOL, frame_count)
    frame_specs = [
        {"color": color, "r0": 0, "c0": 0, "h": h, "w": w, "area": h * w}
        for (h, w), color in zip(sizes, colors)
    ]
    frame_specs[0]["r0"] = int(top)
    frame_specs[0]["c0"] = int(left)
    ranked = _sorted_frames(frame_specs)
    try:
        _, nested_positions = _nest_frames((height, width), ranked, BG, anchor="lower_left")
    except Exception:
        return None

    largest_origin = (int(top), int(left))
    positions = _scatter_positions((height, width), sizes, largest_origin, nested_positions)
    if positions is None:
        return None

    grid = _render_input((height, width), sizes, colors, positions)
    if not _semantic_checks(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _generate_one()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid nested-frame input")
