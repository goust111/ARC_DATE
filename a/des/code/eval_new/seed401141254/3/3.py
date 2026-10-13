import random
from collections import Counter

import numpy as np


SIDE_TO_DIR = {
    "top": (-1, 0),
    "bottom": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}


def _get_bg(grid: np.ndarray) -> int:
    counts = Counter(map(int, np.asarray(grid, dtype=int).ravel()))
    return int(max(counts.items(), key=lambda kv: kv[1])[0])


def _find_false_run(mask: list[bool]):
    if all(mask):
        return None
    false_idx = [i for i, value in enumerate(mask) if not value]
    if not false_idx:
        return None
    if false_idx[0] == 0 or false_idx[-1] == len(mask) - 1:
        return "invalid"
    start = false_idx[0]
    end = false_idx[-1] + 1
    if false_idx != list(range(start, end)):
        return "invalid"
    return (start, end)


def _gap_cells(bbox, side: str, gap_pos: int, gap_width: int):
    r0, c0, r1, c1 = map(int, bbox)
    cells = []
    if side == "top":
        row = r0
        for offset in range(gap_width):
            cells.append((row, c0 + 1 + gap_pos + offset))
    elif side == "bottom":
        row = r1
        for offset in range(gap_width):
            cells.append((row, c0 + 1 + gap_pos + offset))
    elif side == "left":
        col = c0
        for offset in range(gap_width):
            cells.append((r0 + 1 + gap_pos + offset, col))
    else:
        col = c1
        for offset in range(gap_width):
            cells.append((r0 + 1 + gap_pos + offset, col))
    return cells


def _detect_frames(grid: np.ndarray, bg: int):
    g = np.asarray(grid, dtype=int)
    frames = []
    for color in map(int, np.unique(g)):
        if color == bg:
            continue
        coords = np.argwhere(g == color)
        if coords.size == 0:
            continue

        r0, c0 = map(int, coords.min(axis=0))
        r1, c1 = map(int, coords.max(axis=0))
        if r1 - r0 < 2 or c1 - c0 < 2:
            continue

        window = g[r0 : r1 + 1, c0 : c1 + 1]
        local = np.argwhere(window == color)
        if any(0 < int(rr) < window.shape[0] - 1 and 0 < int(cc) < window.shape[1] - 1 for rr, cc in local):
            continue
        inner = window[1:-1, 1:-1]
        if inner.size == 0 or not bool(np.all(inner == bg)):
            continue

        side_masks = {
            "top": [int(g[r0, c]) == color for c in range(c0, c1 + 1)],
            "bottom": [int(g[r1, c]) == color for c in range(c0, c1 + 1)],
            "left": [int(g[r, c0]) == color for r in range(r0, r1 + 1)],
            "right": [int(g[r, c1]) == color for r in range(r0, r1 + 1)],
        }

        gap_side = None
        gap_run = None
        valid = True
        for side, mask in side_masks.items():
            run = _find_false_run(mask)
            if run is None:
                continue
            if run == "invalid":
                valid = False
                break
            if gap_side is not None:
                valid = False
                break
            gap_side = side
            gap_run = run
        if not valid or gap_side is None:
            continue

        start, end = gap_run
        gap_width = end - start
        if gap_width < 1:
            continue

        if gap_side in {"top", "bottom"}:
            gap_pos = start - 1
        else:
            gap_pos = start - 1
        if gap_pos < 0:
            continue

        gap_cells = _gap_cells((r0, c0, r1, c1), gap_side, gap_pos, gap_width)
        if any(int(g[r, c]) != bg for r, c in gap_cells):
            continue

        frames.append(
            {
                "color": color,
                "bbox": (r0, c0, r1, c1),
                "gap_side": gap_side,
                "gap_pos": gap_pos,
                "gap_width": gap_width,
                "gap_cells": gap_cells,
                "direction": SIDE_TO_DIR[gap_side],
            }
        )

    frames.sort(key=lambda frame: (frame["bbox"][0], frame["bbox"][1], frame["color"]))
    return frames


def _trace_ray(grid: np.ndarray, bg: int, frame: dict):
    g = np.asarray(grid, dtype=int)
    height, width = g.shape
    dr, dc = frame["direction"]
    gap_cells = frame["gap_cells"]

    path_cells = []
    step = 1
    while True:
        lane = [(r + dr * step, c + dc * step) for r, c in gap_cells]
        if any(not (0 <= r < height and 0 <= c < width) for r, c in lane):
            return path_cells, [], False

        hit_cells = [(r, c) for r, c in lane if int(g[r, c]) != bg]
        if hit_cells:
            marker_cells = []
            for r, c in hit_cells:
                ir = r + dr
                ic = c + dc
                if 0 <= ir < height and 0 <= ic < width:
                    marker_cells.append((ir, ic))
            return path_cells, marker_cells, True

        path_cells.extend(lane)
        step += 1


def main(grid: np.ndarray) -> np.ndarray:
    """
    Cast same-colored rays outward from the gaps of hollow rectangular frames and mark the inside of frames they hit.

    Input structure:
    - The most common color is the background.
    - Every other color forms one hollow axis-aligned rectangular frame with thickness 1.
    - Each frame has exactly one contiguous gap on one side, while the frame interior stays background.

    Rule:
    - Find the frame bbox and identify which side contains the gap and how wide that gap is.
    - Emit a ray from the gap outward, perpendicular to the gapped side, with the same width and color as the frame.
    - Draw the ray through background cells until the first step where any ray lane would touch another frame wall; stop one cell before that wall.
    - When a wall is hit, paint the immediately interior cells behind the struck wall with the ray color, aligned with the hit lane.

    Key insight:
    - The missing border segment defines both the ray direction and the ray thickness, and wall hits create a one-cell-deep projection inside the struck frame.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    out = g.copy()

    for frame in _detect_frames(g, bg):
        path_cells, marker_cells, _hit = _trace_ray(g, bg, frame)
        for r, c in path_cells:
            out[r, c] = int(frame["color"])
        for r, c in marker_cells:
            out[r, c] = int(frame["color"])

    return out


def _segments(span: int):
    result = []
    for width in range(1, min(3, span) + 1):
        for start in range(0, span - width + 1):
            result.append((start, width))
    return result


def _non_overlapping_segment_pairs(span: int):
    segments = _segments(span)
    pairs = []
    for seg_a in segments:
        a_cells = set(range(seg_a[0], seg_a[0] + seg_a[1]))
        for seg_b in segments:
            b_cells = set(range(seg_b[0], seg_b[0] + seg_b[1]))
            if a_cells.isdisjoint(b_cells):
                pairs.append((seg_a, seg_b))
    return pairs


def _draw_frame(grid: np.ndarray, bg: int, frame: dict) -> None:
    r0, c0, r1, c1 = map(int, frame["bbox"])
    color = int(frame["color"])
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color
    for r, c in _gap_cells(frame["bbox"], frame["gap_side"], frame["gap_pos"], frame["gap_width"]):
        grid[r, c] = bg


def _shift_frame(frame: dict, dr: int, dc: int, color: int):
    r0, c0, r1, c1 = map(int, frame["bbox"])
    return {
        "bbox": (r0 + dr, c0 + dc, r1 + dr, c1 + dc),
        "gap_side": frame["gap_side"],
        "gap_pos": int(frame["gap_pos"]),
        "gap_width": int(frame["gap_width"]),
        "color": int(color),
    }


def _random_partition(total: int, slots: int, rng: random.Random):
    parts = [0] * slots
    for _ in range(total):
        parts[rng.randrange(slots)] += 1
    return parts


def _make_horizontal_pair_scene(rng: random.Random, two_way: bool):
    if two_way:
        height = rng.randint(6, 8)
        inner = height - 2
        seg_left, seg_right = rng.choice(_non_overlapping_segment_pairs(inner))
        left_width = rng.randint(4, 7)
        right_width = rng.randint(4, 7)
        corridor = rng.randint(2, 6)
        left_frame = {
            "bbox": (0, 0, height - 1, left_width - 1),
            "gap_side": "right",
            "gap_pos": seg_left[0],
            "gap_width": seg_left[1],
        }
        right_x = left_width + corridor
        right_frame = {
            "bbox": (0, right_x, height - 1, right_x + right_width - 1),
            "gap_side": "left",
            "gap_pos": seg_right[0],
            "gap_width": seg_right[1],
        }
        return {
            "width": right_x + right_width,
            "height": height,
            "frames": [left_frame, right_frame],
        }

    height = rng.randint(4, 8)
    inner = height - 2
    source_seg = rng.choice(_segments(inner))
    target_seg = rng.choice(_segments(inner))
    left_width = rng.randint(4, 7)
    right_width = rng.randint(4, 7)
    corridor = rng.randint(2, 6)
    direction = rng.choice([-1, 1])
    left_frame = {
        "bbox": (0, 0, height - 1, left_width - 1),
        "gap_pos": target_seg[0],
        "gap_width": target_seg[1],
    }
    right_x = left_width + corridor
    right_frame = {
        "bbox": (0, right_x, height - 1, right_x + right_width - 1),
        "gap_pos": source_seg[0],
        "gap_width": source_seg[1],
    }
    if direction == 1:
        left_frame["gap_side"] = "right"
        right_frame["gap_side"] = "right"
    else:
        left_frame["gap_side"] = "left"
        right_frame["gap_side"] = "left"
    return {
        "width": right_x + right_width,
        "height": height,
        "frames": [left_frame, right_frame],
    }


def _make_horizontal_miss_scene(rng: random.Random):
    height = rng.randint(4, 8)
    width = rng.randint(4, 8)
    seg = rng.choice(_segments(height - 2))
    return {
        "width": width,
        "height": height,
        "frames": [
            {
                "bbox": (0, 0, height - 1, width - 1),
                "gap_side": rng.choice(["left", "right"]),
                "gap_pos": seg[0],
                "gap_width": seg[1],
            }
        ],
    }


def _make_vertical_pair_scene(rng: random.Random, two_way: bool):
    if two_way:
        width = rng.randint(6, 8)
        inner = width - 2
        seg_top, seg_bottom = rng.choice(_non_overlapping_segment_pairs(inner))
        top_height = rng.randint(4, 7)
        bottom_height = rng.randint(4, 7)
        corridor = rng.randint(2, 6)
        top_frame = {
            "bbox": (0, 0, top_height - 1, width - 1),
            "gap_side": "bottom",
            "gap_pos": seg_top[0],
            "gap_width": seg_top[1],
        }
        bottom_y = top_height + corridor
        bottom_frame = {
            "bbox": (bottom_y, 0, bottom_y + bottom_height - 1, width - 1),
            "gap_side": "top",
            "gap_pos": seg_bottom[0],
            "gap_width": seg_bottom[1],
        }
        return {
            "width": width,
            "height": bottom_y + bottom_height,
            "frames": [top_frame, bottom_frame],
        }

    width = rng.randint(4, 8)
    inner = width - 2
    source_seg = rng.choice(_segments(inner))
    target_seg = rng.choice(_segments(inner))
    top_height = rng.randint(4, 7)
    bottom_height = rng.randint(4, 7)
    corridor = rng.randint(2, 6)
    direction = rng.choice([-1, 1])
    top_frame = {
        "bbox": (0, 0, top_height - 1, width - 1),
        "gap_pos": target_seg[0],
        "gap_width": target_seg[1],
    }
    bottom_y = top_height + corridor
    bottom_frame = {
        "bbox": (bottom_y, 0, bottom_y + bottom_height - 1, width - 1),
        "gap_pos": source_seg[0],
        "gap_width": source_seg[1],
    }
    if direction == 1:
        top_frame["gap_side"] = "bottom"
        bottom_frame["gap_side"] = "bottom"
    else:
        top_frame["gap_side"] = "top"
        bottom_frame["gap_side"] = "top"
    return {
        "width": width,
        "height": bottom_y + bottom_height,
        "frames": [top_frame, bottom_frame],
    }


def _make_vertical_miss_scene(rng: random.Random):
    height = rng.randint(4, 8)
    width = rng.randint(4, 8)
    seg = rng.choice(_segments(width - 2))
    return {
        "width": width,
        "height": height,
        "frames": [
            {
                "bbox": (0, 0, height - 1, width - 1),
                "gap_side": rng.choice(["top", "bottom"]),
                "gap_pos": seg[0],
                "gap_width": seg[1],
            }
        ],
    }


def _build_horizontal_grid(rng: random.Random, scenes: list[dict], bg: int, colors: list[int]):
    count = len(scenes)
    min_height = sum(scene["height"] for scene in scenes) + (count - 1)
    max_width = max(scene["width"] for scene in scenes)
    if min_height > 30 or max_width > 30:
        return None

    total_height = min_height + rng.randint(0, 30 - min_height)
    total_width = max_width + rng.randint(0, 30 - max_width)
    extra_height = total_height - min_height
    parts = _random_partition(extra_height, 2 + count + (count - 1), rng)
    top_margin = parts[0]
    band_extra = parts[2 : 2 + count]
    gap_extra = parts[2 + count :]

    grid = np.full((total_height, total_width), bg, dtype=int)
    color_iter = iter(colors)
    current_row = top_margin
    for idx, scene in enumerate(scenes):
        band_h = scene["height"] + band_extra[idx]
        y_shift = current_row + rng.randint(0, band_extra[idx])
        x_shift = rng.randint(0, total_width - scene["width"])
        for frame in scene["frames"]:
            placed = _shift_frame(frame, y_shift, x_shift, next(color_iter))
            _draw_frame(grid, bg, placed)
        current_row += band_h
        if idx < count - 1:
            current_row += 1 + gap_extra[idx]
    return grid


def _build_vertical_grid(rng: random.Random, scenes: list[dict], bg: int, colors: list[int]):
    count = len(scenes)
    min_width = sum(scene["width"] for scene in scenes) + (count - 1)
    max_height = max(scene["height"] for scene in scenes)
    if min_width > 30 or max_height > 30:
        return None

    total_width = min_width + rng.randint(0, 30 - min_width)
    total_height = max_height + rng.randint(0, 30 - max_height)
    extra_width = total_width - min_width
    parts = _random_partition(extra_width, 2 + count + (count - 1), rng)
    left_margin = parts[0]
    band_extra = parts[2 : 2 + count]
    gap_extra = parts[2 + count :]

    grid = np.full((total_height, total_width), bg, dtype=int)
    color_iter = iter(colors)
    current_col = left_margin
    for idx, scene in enumerate(scenes):
        band_w = scene["width"] + band_extra[idx]
        x_shift = current_col + rng.randint(0, band_extra[idx])
        y_shift = rng.randint(0, total_height - scene["height"])
        for frame in scene["frames"]:
            placed = _shift_frame(frame, y_shift, x_shift, next(color_iter))
            _draw_frame(grid, bg, placed)
        current_col += band_w
        if idx < count - 1:
            current_col += 1 + gap_extra[idx]
    return grid


def _gen_attempt(rng: random.Random):
    mode = rng.choice(["horizontal", "vertical"])
    two_way = bool(rng.randrange(2))
    if mode == "horizontal":
        scenes = [
            _make_horizontal_pair_scene(rng, two_way=two_way),
            _make_horizontal_miss_scene(rng),
        ]
        if rng.random() < 0.5:
            scenes.append(_make_horizontal_miss_scene(rng))
        rng.shuffle(scenes)
    else:
        scenes = [
            _make_vertical_pair_scene(rng, two_way=two_way),
            _make_vertical_miss_scene(rng),
        ]
        if rng.random() < 0.5:
            scenes.append(_make_vertical_miss_scene(rng))
        rng.shuffle(scenes)

    frame_count = sum(len(scene["frames"]) for scene in scenes)
    palette = list(range(10))
    bg = rng.choice(palette)
    palette.remove(bg)
    colors = rng.sample(palette, frame_count)

    if mode == "horizontal":
        grid = _build_horizontal_grid(rng, scenes, bg, colors)
    else:
        grid = _build_vertical_grid(rng, scenes, bg, colors)
    if grid is None:
        return None

    if grid.ndim != 2 or max(grid.shape) > 30:
        return None
    if _get_bg(grid) != bg:
        return None

    frames = _detect_frames(grid, bg)
    if len(frames) != frame_count:
        return None

    hit_count = 0
    miss_count = 0
    for frame in frames:
        path_cells, marker_cells, hit = _trace_ray(grid, bg, frame)
        if not path_cells and not hit:
            return None
        if hit:
            if not marker_cells:
                return None
            hit_count += 1
        else:
            miss_count += 1

    if hit_count == 0 or miss_count == 0:
        return None

    out = main(grid)
    if not isinstance(out, np.ndarray) or out.shape != grid.shape:
        return None
    if not np.any(out != grid):
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(300):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input after 300 attempts")
