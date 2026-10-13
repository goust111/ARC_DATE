from __future__ import annotations

from collections import Counter
import random

import numpy as np


BACKGROUND = 0
FRAME = 5
YELLOW = 4
ARM_COLORS = (1, 2, 3, 7, 9)
MAX_GENERATION_ATTEMPTS = 500


def _frame_cells(center: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = center
    return [(rr, cc) for rr in range(r - 1, r + 2) for cc in range(c - 1, c + 2)]


def _plus_cells(center: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = center
    return [(r, c), (r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _draw_frame(grid: np.ndarray, center: tuple[int, int], inner_color: int) -> None:
    r, c = center
    grid[r - 1 : r + 2, c - 1 : c + 2] = FRAME
    grid[r, c] = int(inner_color)


def _draw_plus(grid: np.ndarray, center: tuple[int, int], arm_color: int) -> None:
    r, c = center
    grid[r, c] = YELLOW
    grid[r - 1, c] = int(arm_color)
    grid[r + 1, c] = int(arm_color)
    grid[r, c - 1] = int(arm_color)
    grid[r, c + 1] = int(arm_color)


def _find_frames(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    frames: list[dict] = []
    seen_centers: set[tuple[int, int]] = set()
    for r0 in range(h - 2):
        for c0 in range(w - 2):
            patch = g[r0 : r0 + 3, c0 : c0 + 3]
            if patch.shape != (3, 3):
                continue
            border = patch.copy()
            border[1, 1] = FRAME
            inner = int(patch[1, 1])
            if inner in (BACKGROUND, YELLOW, FRAME):
                continue
            if np.all(border == FRAME):
                center = (r0 + 1, c0 + 1)
                if center in seen_centers:
                    continue
                seen_centers.add(center)
                frames.append(
                    {
                        "color": inner,
                        "center": center,
                        "bbox": (r0, c0, r0 + 2, c0 + 2),
                    }
                )
    return sorted(frames, key=lambda f: (f["center"][0], f["center"][1]))


def _find_markers(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    markers: list[dict] = []
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            if int(g[r, c]) != YELLOW:
                continue
            arms = [int(g[r - 1, c]), int(g[r + 1, c]), int(g[r, c - 1]), int(g[r, c + 1])]
            if len(set(arms)) != 1:
                continue
            arm = arms[0]
            if arm not in ARM_COLORS:
                continue
            markers.append({"center": (r, c), "color": arm, "cells": _plus_cells((r, c))})
    return sorted(markers, key=lambda m: (m["center"][0], m["center"][1]))


def _bridge_axis(low0: int, high0: int, low1: int, high1: int, ctr0: int, ctr1: int) -> int:
    start = max(low0, low1)
    end = min(high0, high1)
    assert start <= end, "chosen bridge direction must have an overlapping corridor"
    choices = list(range(start, end + 1))
    return min(choices, key=lambda v: (abs(v - ctr0) + abs(v - ctr1), v))


def _bridge_cells_between(frame_a: dict, frame_b: dict) -> tuple[set[tuple[int, int]], str]:
    ra, ca = map(int, frame_a["center"])
    rb, cb = map(int, frame_b["center"])
    r0a, c0a, r1a, c1a = map(int, frame_a["bbox"])
    r0b, c0b, r1b, c1b = map(int, frame_b["bbox"])

    if abs(ra - rb) <= abs(ca - cb):
        row = _bridge_axis(r0a, r1a, r0b, r1b, ra, rb)
        if ca < cb:
            c_start, c_end = c1a + 1, c0b - 1
        else:
            c_start, c_end = c1b + 1, c0a - 1
        assert c_start <= c_end, "horizontal bridge gap must be non-empty"
        return {(row, c) for c in range(c_start, c_end + 1)}, "horizontal"

    col = _bridge_axis(c0a, c1a, c0b, c1b, ca, cb)
    if ra < rb:
        r_start, r_end = r1a + 1, r0b - 1
    else:
        r_start, r_end = r1b + 1, r0a - 1
    assert r_start <= r_end, "vertical bridge gap must be non-empty"
    return {(r, col) for r in range(r_start, r_end + 1)}, "vertical"


def _analyze_scene(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    frames = _find_frames(g)
    markers = _find_markers(g)
    frame_by_color = {int(frame["color"]): frame for frame in frames}
    assert len(frame_by_color) == len(frames), "frame inner colors must be unique"

    counts = Counter(int(marker["color"]) for marker in markers)
    valid_colors = {color for color, count in counts.items() if count >= 2}
    kept = [marker for marker in markers if int(marker["color"]) in valid_colors]
    kept.sort(key=lambda marker: marker["center"][1])
    erased = [marker for marker in markers if int(marker["color"]) not in valid_colors]
    return {
        "frames": frames,
        "markers": markers,
        "counts": counts,
        "valid_colors": valid_colors,
        "kept_markers": kept,
        "erased_markers": erased,
        "frame_by_color": frame_by_color,
    }


def _transform_from_scene(grid: np.ndarray, scene: dict) -> np.ndarray:
    out = np.array(grid, dtype=int, copy=True)
    for marker in scene["erased_markers"]:
        for r, c in marker["cells"]:
            out[int(r), int(c)] = BACKGROUND

    for left, right in zip(scene["kept_markers"], scene["kept_markers"][1:]):
        src_color = int(left["color"])
        dst_color = int(right["color"])
        if src_color == dst_color:
            continue
        assert src_color in scene["frame_by_color"] and dst_color in scene["frame_by_color"]
        cells, _orientation = _bridge_cells_between(
            scene["frame_by_color"][src_color], scene["frame_by_color"][dst_color]
        )
        for r, c in cells:
            out[int(r), int(c)] = src_color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Filter repeated-color yellow-centered plus markers into a control sequence, erase the rest, and bridge matching framed squares.

    Input structure:
    - Black background contains several grey 3x3 framed squares in the upper region; each frame has a unique 1x1 inner color.
    - The lower region contains yellow-centered plus markers whose four orthogonal arms share one non-black, non-yellow color.
    - Marker center columns are distinct, and some arm colors repeat while others appear only once as distractors.

    Rule:
    - Detect every valid plus marker, count how many markers use each arm color, and keep only colors that appear on at least two markers.
    - Erase every marker whose arm color is not kept.
    - Sort the kept markers by center-column from left to right and read their arm colors as a control sequence.
    - For each consecutive color pair (ci, ci+1), find the framed squares whose inner colors are ci and ci+1.
    - Compare the squares' center row difference and center column difference: if row difference is smaller or equal, draw a horizontal 1-cell bridge across the gap; otherwise draw a vertical 1-cell bridge. The bridge color is ci.

    Key insight:
    - The repeated arm colors identify which plus markers are real instructions, and only after filtering them can their left-to-right placement be decoded into the square-bridging order.
    """
    scene = _analyze_scene(np.asarray(grid, dtype=int))
    return _transform_from_scene(grid, scene)


def _sequence_two_colors(colors: list[int], length: int) -> list[int]:
    start = random.randrange(2)
    return [int(colors[(start + i) % 2]) for i in range(length)]


def _sequence_three_colors(colors: list[int]) -> list[int]:
    a, b, c = colors
    options = [
        [a, b, c, b, a, b, c],
        [c, b, a, b, c, b, a],
    ]
    return [int(v) for v in random.choice(options)]


def _candidate_frame_centers(height: int, width: int) -> list[tuple[int, int]]:
    upper = height // 2
    rows = sorted({2, max(2, upper - 3)})
    cols = list(range(2, width - 1, 4))
    return [(r, c) for r in rows for c in cols]


def _layout_valid_frames(height: int, width: int, valid_colors: list[int]) -> tuple[dict[int, tuple[int, int]], set[tuple[int, int]], set[str]] | None:
    upper = height // 2
    row_top = 2
    row_bottom = upper - 3
    cols = list(range(2, width - 1, 4))
    if row_bottom - row_top < 3 or len(cols) < 3:
        return None

    if len(valid_colors) == 2:
        orientation = random.choice(["horizontal", "vertical"])
        if orientation == "horizontal":
            c0, c1 = random.sample(cols, 2)
            row = random.choice([row_top, row_bottom])
            centers = [(row, min(c0, c1)), (row, max(c0, c1))]
            used_orientations = {"horizontal"}
        else:
            col = random.choice(cols)
            centers = [(row_top, col), (row_bottom, col)]
            used_orientations = {"vertical"}
        return {int(color): center for color, center in zip(valid_colors, centers)}, set(), used_orientations

    template = random.choice(["hv", "vh"])
    c_left, c_mid, c_right = random.sample(cols, 3)
    c_left, c_mid, c_right = sorted((c_left, c_mid, c_right))
    if template == "hv":
        centers = [(row_top, c_left), (row_top, c_mid), (row_bottom, c_mid)]
    else:
        centers = [(row_top, c_left), (row_bottom, c_left), (row_bottom, c_mid)]
    mapping = {int(color): center for color, center in zip(valid_colors, centers)}
    dummy_frames = {color: {"color": color, "center": ctr, "bbox": (ctr[0] - 1, ctr[1] - 1, ctr[0] + 1, ctr[1] + 1)} for color, ctr in mapping.items()}
    blocked_bridge_cells: set[tuple[int, int]] = set()
    for frame_a, frame_b in ((valid_colors[0], valid_colors[1]), (valid_colors[1], valid_colors[2])):
        cells, _ = _bridge_cells_between(dummy_frames[frame_a], dummy_frames[frame_b])
        blocked_bridge_cells |= cells
    return mapping, blocked_bridge_cells, {"horizontal", "vertical"}


def _pick_extra_frame_centers(
    all_centers: list[tuple[int, int]],
    used_centers: set[tuple[int, int]],
    blocked_bridge_cells: set[tuple[int, int]],
    count: int,
) -> list[tuple[int, int]] | None:
    random_centers = all_centers[:]
    random.shuffle(random_centers)
    picked: list[tuple[int, int]] = []
    for center in random_centers:
        if center in used_centers:
            continue
        bbox_cells = set(_frame_cells(center))
        if bbox_cells & blocked_bridge_cells:
            continue
        picked.append(center)
        used_centers.add(center)
        if len(picked) == count:
            return picked
    return None


def _assign_marker_rows(height: int, columns: list[int]) -> list[int] | None:
    split = height // 2
    row_choices = list(range(split + 1, height - 1))
    order = list(range(len(columns)))
    random.shuffle(order)
    order.sort(key=lambda idx: abs(columns[idx] - (columns[0] + columns[-1]) / 2))

    occupied: set[tuple[int, int]] = set()
    rows: list[int | None] = [None] * len(columns)

    def backtrack(pos: int) -> bool:
        if pos == len(order):
            return True
        idx = order[pos]
        random_rows = row_choices[:]
        random.shuffle(random_rows)
        for row in random_rows:
            cells = set(_plus_cells((row, columns[idx])))
            if cells & occupied:
                continue
            occupied.update(cells)
            rows[idx] = row
            if backtrack(pos + 1):
                return True
            occupied.difference_update(cells)
            rows[idx] = None
        return False

    if not backtrack(0):
        return None
    return [int(r) for r in rows]


def _build_input_instance() -> tuple[np.ndarray, dict] | None:
    height = random.randint(18, 24)
    width = random.randint(19, 24)
    valid_color_count = random.choices([2, 3], weights=[2, 3], k=1)[0]

    if valid_color_count == 3:
        marker_sequence_len = 7
        invalid_marker_count = 1
        square_count = random.choice([4, 5])
    else:
        options: list[tuple[int, int, int]] = []
        for marker_sequence_len in range(4, 8):
            for invalid_marker_count in range(1, 4):
                total_markers = marker_sequence_len + invalid_marker_count
                if not (6 <= total_markers <= 8):
                    continue
                for square_count in (4, 5):
                    if square_count >= valid_color_count + invalid_marker_count:
                        options.append((marker_sequence_len, invalid_marker_count, square_count))
        marker_sequence_len, invalid_marker_count, square_count = random.choice(options)

    square_colors = random.sample(list(ARM_COLORS), square_count)
    valid_colors = random.sample(square_colors, valid_color_count)
    other_square_colors = [color for color in square_colors if color not in valid_colors]
    invalid_colors = random.sample(other_square_colors, invalid_marker_count)

    layout = _layout_valid_frames(height, width, valid_colors)
    if layout is None:
        return None
    valid_frame_centers, blocked_bridge_cells, used_orientations = layout

    ordered_valid_colors = list(valid_frame_centers)
    if valid_color_count == 3:
        marker_sequence = _sequence_three_colors(ordered_valid_colors)
    else:
        marker_sequence = _sequence_two_colors(ordered_valid_colors, marker_sequence_len)

    frame_centers = _candidate_frame_centers(height, width)
    used_centers = set(valid_frame_centers.values())
    extra_needed = square_count - valid_color_count
    extra_centers = _pick_extra_frame_centers(frame_centers, used_centers, blocked_bridge_cells, extra_needed)
    if extra_centers is None:
        return None

    frame_positions = dict(valid_frame_centers)
    for color, center in zip(other_square_colors, extra_centers):
        frame_positions[int(color)] = center

    total_markers = marker_sequence_len + invalid_marker_count
    slot_columns = sorted(random.sample(range(2, width - 2), total_markers))
    invalid_slots = set(random.sample(range(total_markers), invalid_marker_count))
    marker_specs: list[tuple[int, int]] = []
    seq_idx = 0
    invalid_iter = iter(random.sample(invalid_colors, len(invalid_colors)))
    for idx in range(total_markers):
        if idx in invalid_slots:
            marker_specs.append((slot_columns[idx], next(invalid_iter)))
        else:
            marker_specs.append((slot_columns[idx], marker_sequence[seq_idx]))
            seq_idx += 1

    marker_rows = _assign_marker_rows(height, [col for col, _color in marker_specs])
    if marker_rows is None:
        return None

    grid = np.zeros((height, width), dtype=int)
    for color, center in frame_positions.items():
        _draw_frame(grid, center, color)
    for row, (col, color) in zip(marker_rows, marker_specs):
        _draw_plus(grid, (row, col), color)

    metadata = {
        "valid_color_count": valid_color_count,
        "expected_orientations": used_orientations,
        "square_count": square_count,
    }
    return grid, metadata


def _validate_generated_instance(grid: np.ndarray, metadata: dict) -> bool:
    g = np.asarray(grid, dtype=int)
    scene = _analyze_scene(g)

    if not (4 <= len(scene["frames"]) <= 5):
        return False
    if not (6 <= len(scene["markers"]) <= 8):
        return False
    if len(scene["valid_colors"]) != int(metadata["valid_color_count"]):
        return False
    if len(scene["valid_colors"]) not in (2, 3):
        return False
    if len({marker["center"][1] for marker in scene["markers"]}) != len(scene["markers"]):
        return False
    if set(scene["valid_colors"]) - set(scene["frame_by_color"]):
        return False
    if not scene["erased_markers"]:
        return False
    if any(count < 2 for color, count in scene["counts"].items() if color in scene["valid_colors"]):
        return False
    if any(count != 1 for color, count in scene["counts"].items() if color not in scene["valid_colors"]):
        return False

    orientations: set[str] = set()
    for left, right in zip(scene["kept_markers"], scene["kept_markers"][1:]):
        if int(left["color"]) == int(right["color"]):
            return False
        cells, orientation = _bridge_cells_between(
            scene["frame_by_color"][int(left["color"])],
            scene["frame_by_color"][int(right["color"])],
        )
        if not cells:
            return False
        orientations.add(orientation)
    if not orientations.issuperset(set(metadata["expected_orientations"])):
        return False

    out = main(g)
    expected = _transform_from_scene(g, scene)
    if not np.array_equal(out, expected):
        return False
    if np.array_equal(out, g):
        return False

    top_changed = bool(np.any(out[: g.shape[0] // 2] != g[: g.shape[0] // 2]))
    bottom_changed = bool(np.any(out[g.shape[0] // 2 :] != g[g.shape[0] // 2 :]))
    if not (top_changed and bottom_changed):
        return False

    for marker in scene["erased_markers"]:
        for r, c in marker["cells"]:
            if int(out[r, c]) != BACKGROUND:
                return False
    for marker in scene["kept_markers"]:
        r, c = marker["center"]
        if int(out[r, c]) != YELLOW:
            return False
        arm = int(marker["color"])
        if [int(out[r - 1, c]), int(out[r + 1, c]), int(out[r, c - 1]), int(out[r, c + 1])] != [arm] * 4:
            return False

    return True


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        built = _build_input_instance()
        if built is None:
            continue
        grid, metadata = built
        if _validate_generated_instance(grid, metadata):
            return grid
    raise RuntimeError("failed to generate a valid ARC instance")
