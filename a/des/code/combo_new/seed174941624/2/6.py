from __future__ import annotations

import random
from collections import Counter

import numpy as np


DIRS: dict[str, tuple[int, int]] = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
GEN_ATTEMPTS = 500
MAX_COUNT = 8
OBJECT_COUNT_RANGE = (2, 4)


def _mode_color(grid: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(values[int(np.argmax(counts))])


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                if rr > 0 and mask[rr - 1, cc] and not seen[rr - 1, cc]:
                    seen[rr - 1, cc] = True
                    stack.append((rr - 1, cc))
                if rr + 1 < h and mask[rr + 1, cc] and not seen[rr + 1, cc]:
                    seen[rr + 1, cc] = True
                    stack.append((rr + 1, cc))
                if cc > 0 and mask[rr, cc - 1] and not seen[rr, cc - 1]:
                    seen[rr, cc - 1] = True
                    stack.append((rr, cc - 1))
                if cc + 1 < w and mask[rr, cc + 1] and not seen[rr, cc + 1]:
                    seen[rr, cc + 1] = True
                    stack.append((rr, cc + 1))
            comps.append(np.asarray(cells, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _frame_bbox(coords: np.ndarray) -> tuple[int, int, int, int] | None:
    r0, c0, r1, c1 = _bbox(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return None
    if int(coords.shape[0]) != 2 * (h + w) - 4:
        return None
    rows = coords[:, 0]
    cols = coords[:, 1]
    if not bool(np.all((rows == r0) | (rows == r1) | (cols == c0) | (cols == c1))):
        return None
    return r0, c0, r1, c1


def _find_separator(grid: np.ndarray, bg: int) -> tuple[int, int]:
    h, w = grid.shape
    candidates: list[tuple[int, int, int]] = []
    for c in range(1, w - 1):
        col = grid[:, c]
        color = int(col[0])
        if color == bg or not bool(np.all(col == color)):
            continue
        if not bool(np.any(grid[:, :c] != bg)) or not bool(np.any(grid[:, c + 1 :] != bg)):
            continue
        support = int(np.sum(grid[:, max(0, c - 1)] == bg)) + int(np.sum(grid[:, min(w - 1, c + 1)] == bg))
        candidates.append((support, c, color))
    assert candidates, "expected a full-height separator column"
    _, col, color = max(candidates)
    return int(col), int(color)


def _pointer_direction(marker_coords: np.ndarray, box: tuple[int, int, int, int]) -> str:
    r0, c0, r1, c1 = box
    rows = marker_coords[:, 0]
    cols = marker_coords[:, 1]
    if bool(np.all(rows == r0)):
        return "up"
    if bool(np.all(rows == r1)):
        return "down"
    if bool(np.all(cols == c0)):
        return "left"
    if bool(np.all(cols == c1)):
        return "right"

    scores = {
        "up": int(np.sum(rows == r0)),
        "down": int(np.sum(rows == r1)),
        "left": int(np.sum(cols == c0)),
        "right": int(np.sum(cols == c1)),
    }
    return max(scores.items(), key=lambda item: (item[1], item[0]))[0]


def _object_patch(height: int, width: int, base_color: int, marker_color: int, direction: str) -> np.ndarray:
    patch = np.full((height, width), int(base_color), dtype=int)
    if direction in ("up", "down"):
        cols = [width // 2 - 1, width // 2]
        row = 0 if direction == "up" else height - 1
        patch[row, cols] = int(marker_color)
    else:
        rows = [height // 2 - 1, height // 2]
        col = 0 if direction == "left" else width - 1
        patch[rows, col] = int(marker_color)
    return patch


def _paint_object(grid: np.ndarray, top: int, left: int, patch: np.ndarray) -> None:
    h, w = patch.shape
    grid[top : top + h, left : left + w] = patch


def _paint_frame(
    grid: np.ndarray,
    top: int,
    left: int,
    inner_h: int,
    inner_w: int,
    frame_color: int,
    marker_color: int,
    marker_count: int,
) -> None:
    h = inner_h + 2
    w = inner_w + 2
    grid[top, left : left + w] = int(frame_color)
    grid[top + h - 1, left : left + w] = int(frame_color)
    grid[top : top + h, left] = int(frame_color)
    grid[top : top + h, left + w - 1] = int(frame_color)

    cells = [(r, c) for r in range(top + 1, top + h - 1) for c in range(left + 1, left + w - 1)]
    marked = random.sample(cells, marker_count)
    for r, c in marked:
        grid[r, c] = int(marker_color)


def _rect_conflicts(
    rect: tuple[int, int, int, int], others: list[tuple[int, int, int, int]], gap: int = 1
) -> bool:
    r0, c0, r1, c1 = rect
    for or0, oc0, or1, oc1 in others:
        separated = r1 + gap < or0 or or1 + gap < r0 or c1 + gap < oc0 or oc1 + gap < c0
        if not separated:
            return True
    return False


def _sample_object_spec(height: int, left_width: int, direction: str, base_color: int) -> dict | None:
    if direction in ("up", "down"):
        height_choices = [v for v in range(3, min(6, height) + 1)]
        width_choices = [v for v in range(4, min(8, left_width) + 1) if v % 2 == 0]
        if not height_choices or not width_choices:
            return None
        obj_h = random.choice(height_choices)
        obj_w = random.choice(width_choices)
        axis_room = height - obj_h
    else:
        height_choices = [v for v in range(4, min(8, height) + 1) if v % 2 == 0]
        width_choices = [v for v in range(3, min(6, left_width) + 1)]
        if not height_choices or not width_choices:
            return None
        obj_h = random.choice(height_choices)
        obj_w = random.choice(width_choices)
        axis_room = left_width - obj_w

    if axis_room < 1:
        return None

    clipped = random.random() < 0.5
    if clipped:
        step_cap = min(axis_room, 4, MAX_COUNT - 1)
        if step_cap < 1:
            return None
        actual_shift = random.randint(1, step_cap)
        desired_shift = random.randint(actual_shift + 1, min(MAX_COUNT, actual_shift + 3))
    else:
        step_cap = min(axis_room, 5, MAX_COUNT)
        if step_cap < 1:
            return None
        desired_shift = random.randint(1, step_cap)
        actual_shift = desired_shift

    return {
        "color": int(base_color),
        "direction": direction,
        "height": int(obj_h),
        "width": int(obj_w),
        "distance": int(desired_shift),
        "shift": int(actual_shift),
        "clipped": bool(clipped),
    }


def _object_candidates(
    spec: dict, height: int, left_width: int, used_input: list[tuple[int, int, int, int]], used_output: list[tuple[int, int, int, int]]
) -> list[dict]:
    obj_h = int(spec["height"])
    obj_w = int(spec["width"])
    shift = int(spec["shift"])
    direction = str(spec["direction"])
    clipped = bool(spec["clipped"])
    dr, dc = DIRS[direction]

    if direction == "up":
        out_rows = [0] if clipped else list(range(0, height - obj_h - shift + 1))
        out_cols = list(range(0, left_width - obj_w + 1))
    elif direction == "down":
        out_rows = [height - obj_h] if clipped else list(range(shift, height - obj_h + 1))
        out_cols = list(range(0, left_width - obj_w + 1))
    elif direction == "left":
        out_rows = list(range(0, height - obj_h + 1))
        out_cols = [0] if clipped else list(range(0, left_width - obj_w - shift + 1))
    else:
        out_rows = list(range(0, height - obj_h + 1))
        out_cols = [left_width - obj_w] if clipped else list(range(shift, left_width - obj_w + 1))

    candidates: list[dict] = []
    for out_r in out_rows:
        for out_c in out_cols:
            in_r = int(out_r - dr * shift)
            in_c = int(out_c - dc * shift)
            if not (0 <= in_r and 0 <= in_c and in_r + obj_h <= height and in_c + obj_w <= left_width):
                continue
            in_rect = (in_r, in_c, in_r + obj_h - 1, in_c + obj_w - 1)
            out_rect = (out_r, out_c, out_r + obj_h - 1, out_c + obj_w - 1)
            if _rect_conflicts(in_rect, used_input, gap=1):
                continue
            if _rect_conflicts(out_rect, used_output, gap=1):
                continue
            candidates.append(
                {
                    "input_top": int(in_r),
                    "input_left": int(in_c),
                    "output_top": int(out_r),
                    "output_left": int(out_c),
                    "input_rect": in_rect,
                    "output_rect": out_rect,
                }
            )
    random.shuffle(candidates)
    return candidates


def _sample_frame_spec(base_color: int, marker_color: int, distance: int) -> dict:
    choices = [(ih, iw) for ih in range(2, 5) for iw in range(2, 6) if ih * iw >= distance]
    inner_h, inner_w = random.choice(choices)
    return {
        "color": int(base_color),
        "marker": int(marker_color),
        "distance": int(distance),
        "inner_h": int(inner_h),
        "inner_w": int(inner_w),
        "height": int(inner_h + 2),
        "width": int(inner_w + 2),
    }


def _frame_candidates(
    spec: dict, height: int, width: int, sep_col: int, used_frames: list[tuple[int, int, int, int]]
) -> list[tuple[int, int, int, int]]:
    frame_h = int(spec["height"])
    frame_w = int(spec["width"])
    row_start = 0
    row_stop = height - frame_h + 1
    if row_stop <= row_start:
        return []

    col_min = sep_col + 2 if sep_col + 2 + frame_w <= width else sep_col + 1
    col_max = width - frame_w
    if col_min > col_max:
        return []

    candidates: list[tuple[int, int, int, int]] = []
    for top in range(row_start, row_stop):
        for left in range(col_min, col_max + 1):
            rect = (top, left, top + frame_h - 1, left + frame_w - 1)
            if _rect_conflicts(rect, used_frames, gap=1):
                continue
            candidates.append(rect)
    random.shuffle(candidates)
    return candidates


def main(grid: np.ndarray) -> np.ndarray:
    """
    Translate each left-side rectangle by the marker count stored in its same-colored counter frame.

    Input structure:
    - The background is the most common color.
    - A one-cell-wide full-height vertical separator splits the grid into a left object area and a right counter area.
    - The left area contains disconnected filled rectangles, each with a two-cell marker strip on one edge inside the rectangle.
    - The right area contains small one-cell-thick rectangular counter frames whose border color matches one left rectangle, and whose interiors contain marker pixels.

    Rule:
    - For each left rectangle, read the side touched by its internal two-cell strip to determine the movement direction.
    - Find the right-side counter frame with the same border color as that rectangle and count its interior marker pixels.
    - Move the whole rectangle, including its strip, by that many cells in the indicated direction.
    - If the requested move would leave the left area or cross the separator, truncate it so the rectangle stops at the boundary.
    - Leave the separator and the entire right area unchanged.

    Key insight:
    - The puzzle splits the motion command into two linked encodings: edge position gives direction, while the same-colored counter frame supplies the distance.
    """
    g = np.asarray(grid, dtype=int)
    bg = _mode_color(g)
    h, w = g.shape

    sep_col, sep_color = _find_separator(g, bg)

    frame_counts: dict[int, int] = {}
    right = g[:, sep_col + 1 :]
    for color in np.unique(right):
        color = int(color)
        if color in {bg, sep_color}:
            continue
        for coords in _components(right == color):
            bbox = _frame_bbox(coords)
            if bbox is None:
                continue
            r0, c0, r1, c1 = bbox
            interior = right[r0 + 1 : r1, c0 + 1 : c1]
            frame_counts[color] = int(np.sum((interior != bg) & (interior != color)))

    left_mask = np.zeros((h, w), dtype=bool)
    left_mask[:, :sep_col] = g[:, :sep_col] != bg
    objects = _components(left_mask)

    moves: list[tuple[np.ndarray, int, int]] = []
    for coords in objects:
        if coords.size == 0:
            continue
        values = [int(g[int(r), int(c)]) for r, c in coords]
        counts = Counter(values)
        if len(counts) < 2:
            continue
        base_color = int(max(counts.items(), key=lambda item: (item[1], item[0]))[0])
        non_base = [color for color in counts if color != base_color]
        marker_color = int(max(non_base, key=lambda color: counts[color]))

        box = _bbox(coords)
        marker_coords = np.asarray([(int(r), int(c)) for r, c in coords if int(g[int(r), int(c)]) == marker_color], dtype=int)
        direction = _pointer_direction(marker_coords, box)
        distance = int(frame_counts.get(base_color, 0))

        r0, c0, r1, c1 = box
        if direction == "up":
            actual = min(distance, r0)
        elif direction == "down":
            actual = min(distance, h - 1 - r1)
        elif direction == "left":
            actual = min(distance, c0)
        else:
            actual = min(distance, sep_col - 1 - c1)

        dr, dc = DIRS[direction]
        moves.append((coords, int(dr * actual), int(dc * actual)))

    out = g.copy()
    for coords, _dr, _dc in moves:
        out[coords[:, 0], coords[:, 1]] = int(bg)
    for coords, dr, dc in moves:
        for r, c in coords:
            out[int(r) + dr, int(c) + dc] = int(g[int(r), int(c)])
    return out


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(16, 28)
    width = random.randint(20, 30)
    if width < 19:
        return None

    sep_col = random.randint(8, width - 8)
    left_width = sep_col
    right_width = width - sep_col - 1
    if left_width < 8 or right_width < 6:
        return None

    obj_count = random.randint(*OBJECT_COUNT_RANGE)
    if obj_count > 4:
        return None

    palette = random.sample(range(10), obj_count + 3)
    bg = int(palette[0])
    sep_color = int(palette[1])
    marker_color = int(palette[2])
    object_colors = [int(color) for color in palette[3:]]

    directions = random.sample(list(DIRS), obj_count)
    object_specs = []
    for color, direction in zip(object_colors, directions):
        spec = _sample_object_spec(height, left_width, direction, color)
        if spec is None:
            return None
        object_specs.append(spec)

    used_input: list[tuple[int, int, int, int]] = []
    used_output: list[tuple[int, int, int, int]] = []
    placed_objects: list[dict] = []
    for spec in sorted(object_specs, key=lambda item: (item["height"] * item["width"], item["distance"]), reverse=True):
        candidates = _object_candidates(spec, height, left_width, used_input, used_output)
        if not candidates:
            return None
        choice = random.choice(candidates)
        spec = spec.copy()
        spec.update(choice)
        used_input.append(choice["input_rect"])
        used_output.append(choice["output_rect"])
        placed_objects.append(spec)

    frame_specs = [_sample_frame_spec(spec["color"], marker_color, spec["distance"]) for spec in placed_objects]
    used_frames: list[tuple[int, int, int, int]] = []
    placed_frames: list[dict] = []
    for frame in sorted(frame_specs, key=lambda item: (item["height"] * item["width"], item["distance"]), reverse=True):
        candidates = _frame_candidates(frame, height, width, sep_col, used_frames)
        if not candidates:
            return None
        top, left, bottom, right = random.choice(candidates)
        frame = frame.copy()
        frame.update({"top": int(top), "left": int(left), "bottom": int(bottom), "right": int(right)})
        used_frames.append((top, left, bottom, right))
        placed_frames.append(frame)

    grid = np.full((height, width), bg, dtype=int)
    grid[:, sep_col] = int(sep_color)
    for frame in placed_frames:
        _paint_frame(
            grid,
            int(frame["top"]),
            int(frame["left"]),
            int(frame["inner_h"]),
            int(frame["inner_w"]),
            int(frame["color"]),
            marker_color,
            int(frame["distance"]),
        )

    expected = grid.copy()
    for spec in placed_objects:
        patch = _object_patch(int(spec["height"]), int(spec["width"]), int(spec["color"]), marker_color, str(spec["direction"]))
        _paint_object(grid, int(spec["input_top"]), int(spec["input_left"]), patch)
        expected[int(spec["input_top"]) : int(spec["input_top"]) + int(spec["height"]), int(spec["input_left"]) : int(spec["input_left"]) + int(spec["width"])] = int(bg)
    for spec in placed_objects:
        patch = _object_patch(int(spec["height"]), int(spec["width"]), int(spec["color"]), marker_color, str(spec["direction"]))
        _paint_object(expected, int(spec["output_top"]), int(spec["output_left"]), patch)

    if _mode_color(grid) != bg:
        return None
    if np.array_equal(grid, expected):
        return None
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_ATTEMPTS} attempts")
