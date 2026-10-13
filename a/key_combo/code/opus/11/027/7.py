from __future__ import annotations

import random
from collections import defaultdict

import numpy as np


_ORTHO_DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
_DIAG_DIRS = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
_COLOR_POOL = tuple(range(1, 10))
_GRID_MIN = 13
_GRID_MAX = 23
_GEN_RETRIES = 400


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(values[np.argmax(counts)])


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _cross_input_cells(center: tuple[int, int], arm_len: int) -> set[tuple[int, int]]:
    r, c = center
    cells = {(r, c)}
    for step in range(1, arm_len + 1):
        cells.add((r - step, c))
        cells.add((r + step, c))
        cells.add((r, c - step))
        cells.add((r, c + step))
    return cells


def _cross_pattern(
    center: tuple[int, int], arm_len: int, arm_color: int, center_color: int
) -> dict[tuple[int, int], tuple[int, str]]:
    r, c = center
    pattern: dict[tuple[int, int], tuple[int, str]] = {(r, c): (center_color, "center")}
    for step in range(1, arm_len + 2):
        pattern[(r - step, c)] = (arm_color, "arm")
        pattern[(r + step, c)] = (arm_color, "arm")
        pattern[(r, c - step)] = (arm_color, "arm")
        pattern[(r, c + step)] = (arm_color, "arm")
    for dr, dc in _DIAG_DIRS:
        pattern[(r + dr, c + dc)] = (center_color, "center")
    return pattern


def _render_output(
    shape: tuple[int, int],
    crosses: list[dict[str, int | tuple[int, int]]],
    *,
    use_diagonals: bool = True,
    use_orthogonal_extension: bool = True,
    high_priority_wins: bool = True,
) -> np.ndarray:
    out = np.zeros(shape, dtype=int)
    ordered = sorted(crosses, key=lambda cross: int(cross["arm_len"]), reverse=high_priority_wins)
    for cross in ordered:
        center = tuple(cross["center"])
        arm_len = int(cross["arm_len"])
        arm_color = int(cross["arm_color"])
        center_color = int(cross["center_color"])
        pattern = {(center[0], center[1]): (center_color, "center")}
        ortho_limit = arm_len + 1 if use_orthogonal_extension else arm_len
        for step in range(1, ortho_limit + 1):
            pattern[(center[0] - step, center[1])] = (arm_color, "arm")
            pattern[(center[0] + step, center[1])] = (arm_color, "arm")
            pattern[(center[0], center[1] - step)] = (arm_color, "arm")
            pattern[(center[0], center[1] + step)] = (arm_color, "arm")
        if use_diagonals:
            for dr, dc in _DIAG_DIRS:
                pattern[(center[0] + dr, center[1] + dc)] = (center_color, "center")
        for (r, c), (color, _role) in pattern.items():
            if _in_bounds(shape, r, c):
                out[r, c] = color
    return out


def _ray_length(grid: np.ndarray, r: int, c: int, dr: int, dc: int, color: int) -> int:
    steps = 0
    rr = r + dr
    cc = c + dc
    while _in_bounds(grid.shape, rr, cc) and int(grid[rr, cc]) == color:
        steps += 1
        rr += dr
        cc += dc
    return steps


def _detect_crosses(grid: np.ndarray) -> list[dict[str, int | tuple[int, int] | set[tuple[int, int]]]]:
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    crosses = []
    claimed: set[tuple[int, int]] = set()

    for r in range(g.shape[0]):
        for c in range(g.shape[1]):
            center_color = int(g[r, c])
            if center_color == bg:
                continue
            neighbor_colors = []
            ok = True
            for dr, dc in _ORTHO_DIRS:
                rr = r + dr
                cc = c + dc
                if not _in_bounds(g.shape, rr, cc):
                    ok = False
                    break
                val = int(g[rr, cc])
                if val == bg or val == center_color:
                    ok = False
                    break
                neighbor_colors.append(val)
            if not ok or len(set(neighbor_colors)) != 1:
                continue

            arm_color = neighbor_colors[0]
            lengths = [
                _ray_length(g, r, c, -1, 0, arm_color),
                _ray_length(g, r, c, 1, 0, arm_color),
                _ray_length(g, r, c, 0, -1, arm_color),
                _ray_length(g, r, c, 0, 1, arm_color),
            ]
            if min(lengths) < 1 or len(set(lengths)) != 1:
                continue
            arm_len = lengths[0]

            expected = _cross_input_cells((r, c), arm_len)
            for rr in range(r - arm_len, r + arm_len + 1):
                for cc in range(c - arm_len, c + arm_len + 1):
                    if not _in_bounds(g.shape, rr, cc):
                        ok = False
                        break
                    val = int(g[rr, cc])
                    if (rr, cc) in expected:
                        if (rr, cc) == (r, c):
                            if val != center_color:
                                ok = False
                                break
                        elif val != arm_color:
                            ok = False
                            break
                    elif val != bg:
                        ok = False
                        break
                if not ok:
                    break
            if not ok or expected & claimed:
                continue

            claimed |= expected
            crosses.append(
                {
                    "center": (r, c),
                    "arm_len": arm_len,
                    "arm_color": arm_color,
                    "center_color": center_color,
                    "input_cells": expected,
                }
            )

    all_non_bg = {(int(r), int(c)) for r, c in np.argwhere(g != bg)}
    used = set().union(*(cross["input_cells"] for cross in crosses)) if crosses else set()
    assert used == all_non_bg, "all non-background pixels must belong to detected crosses"
    lengths = [int(cross["arm_len"]) for cross in crosses]
    assert 2 <= len(crosses) <= 3, "expected 2 or 3 crosses"
    assert len(lengths) == len(set(lengths)), "cross arm lengths must be distinct"
    return crosses


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand every two-color cross and resolve the new overlaps by original arm length priority.

    Input structure:
    - The grid is a black background containing 2 or 3 separate two-color crosses.
    - Each cross has a one-pixel center of one color and four equal-length orthogonal arms
      of a second color.
    - Different crosses use different color pairs, and their original shapes do not overlap.

    Rule:
    - Detect each cross and recover its original arm length.
    - Extend the arm color one more step beyond each of the four arm tips, and paint the
      center color onto the four diagonal neighbors of the center.
    - If expanded pixels from multiple crosses land on the same cell, keep the one from the
      cross with the longer original arm length.
    - Leave all other cells unchanged from background.

    Key insight:
    - The visible winner at an overlap is decided by a hidden geometric property of the input
      object: the pre-expansion arm length, not drawing order or color identity.
    """
    g = np.asarray(grid, dtype=int)
    crosses = _detect_crosses(g)
    simple_crosses = [
        {
            "center": tuple(cross["center"]),
            "arm_len": int(cross["arm_len"]),
            "arm_color": int(cross["arm_color"]),
            "center_color": int(cross["center_color"]),
        }
        for cross in crosses
    ]
    return _render_output(g.shape, simple_crosses, high_priority_wins=True)


def _shift_cross(cross: dict[str, int | tuple[int, int]], dr: int, dc: int) -> dict[str, int | tuple[int, int]]:
    r, c = tuple(cross["center"])
    return {
        "center": (r + dr, c + dc),
        "arm_len": int(cross["arm_len"]),
        "arm_color": int(cross["arm_color"]),
        "center_color": int(cross["center_color"]),
    }


def _min_manhattan(a: set[tuple[int, int]], b: set[tuple[int, int]]) -> int:
    return min(abs(r1 - r2) + abs(c1 - c2) for r1, c1 in a for r2, c2 in b)


def _input_bbox(center: tuple[int, int], arm_len: int) -> tuple[int, int, int, int]:
    r, c = center
    return (r - arm_len, c - arm_len, r + arm_len, c + arm_len)


def _bboxes_disjoint(box_a: tuple[int, int, int, int], box_b: tuple[int, int, int, int]) -> bool:
    return box_a[2] < box_b[0] or box_b[2] < box_a[0] or box_a[3] < box_b[1] or box_b[3] < box_a[1]


def _relative_options(len_a: int, len_b: int) -> list[dict[str, object]]:
    anchor_input = _cross_input_cells((0, 0), len_a)
    anchor_pattern = _cross_pattern((0, 0), len_a, 1, 2)
    anchor_box = _input_bbox((0, 0), len_a)
    options = []
    for dr in range(-8, 9):
        for dc in range(-8, 9):
            if (dr, dc) == (0, 0):
                continue
            other_input = {(r + dr, c + dc) for r, c in _cross_input_cells((0, 0), len_b)}
            if anchor_input & other_input:
                continue
            if not _bboxes_disjoint(anchor_box, _input_bbox((dr, dc), len_b)):
                continue
            distance = _min_manhattan(anchor_input, other_input)
            if not 2 <= distance <= 4:
                continue
            other_pattern = {
                (r + dr, c + dc): (color, role)
                for (r, c), (color, role) in _cross_pattern((0, 0), len_b, 3, 4).items()
            }
            overlap = set(anchor_pattern) & set(other_pattern)
            if not overlap:
                continue
            role_pairs = {
                tuple(sorted((anchor_pattern[pos][1], other_pattern[pos][1]))) for pos in overlap
            }
            options.append(
                {
                    "offset": (dr, dc),
                    "distance": distance,
                    "overlap_count": len(overlap),
                    "role_pairs": role_pairs,
                }
            )
    return options


_RELATIVE_CACHE = {
    (len_a, len_b): _relative_options(len_a, len_b)
    for len_a in (1, 2, 3)
    for len_b in (1, 2, 3)
    if len_a != len_b
}


def _pick_cluster(count: int) -> list[dict[str, int | tuple[int, int]]]:
    lengths = random.sample((1, 2, 3), count)
    lengths.sort(reverse=True)
    colors = random.sample(_COLOR_POOL, count * 2)

    base_crosses = []
    for idx, arm_len in enumerate(lengths):
        base_crosses.append(
            {
                "center": (0, 0),
                "arm_len": arm_len,
                "arm_color": colors[2 * idx],
                "center_color": colors[2 * idx + 1],
            }
        )

    options_by_mode: dict[int, dict[str, list[tuple[int, int]]]] = {}
    for arm_len in lengths[1:]:
        rels = _RELATIVE_CACHE[(lengths[0], arm_len)]
        arm_arm = [tuple(opt["offset"]) for opt in rels if opt["role_pairs"] == {("arm", "arm")}]
        center_mix = [
            tuple(opt["offset"])
            for opt in rels
            if any("center" in pair for pair in opt["role_pairs"])
        ]
        options_by_mode[arm_len] = {"arm_arm": arm_arm, "center_mix": center_mix, "any": [tuple(opt["offset"]) for opt in rels]}

    anchor = dict(base_crosses[0])
    placed = [anchor]

    if count == 2:
        second = dict(base_crosses[1])
        branch = random.choice(["arm_arm", "center_mix"])
        offsets = options_by_mode[int(second["arm_len"])][branch]
        if not offsets:
            offsets = options_by_mode[int(second["arm_len"])]["any"]
        second_offset = random.choice(offsets)
        placed.append(_shift_cross(second, second_offset[0], second_offset[1]))
        return placed

    second = dict(base_crosses[1])
    third = dict(base_crosses[2])
    second_offsets = options_by_mode[int(second["arm_len"])]["arm_arm"] or options_by_mode[int(second["arm_len"])]["any"]
    third_pref = random.choice(["center_mix", "arm_arm"])
    third_offsets = options_by_mode[int(third["arm_len"])][third_pref] or options_by_mode[int(third["arm_len"])]["any"]
    random.shuffle(second_offsets)
    random.shuffle(third_offsets)

    for off2 in second_offsets:
        cross2 = _shift_cross(second, off2[0], off2[1])
        cells2 = _cross_input_cells(tuple(cross2["center"]), int(cross2["arm_len"]))
        for off3 in third_offsets:
            cross3 = _shift_cross(third, off3[0], off3[1])
            cells3 = _cross_input_cells(tuple(cross3["center"]), int(cross3["arm_len"]))
            if cells2 & cells3:
                continue
            if _min_manhattan(cells2, cells3) < 2:
                continue
            box2 = _input_bbox(tuple(cross2["center"]), int(cross2["arm_len"]))
            box3 = _input_bbox(tuple(cross3["center"]), int(cross3["arm_len"]))
            if not _bboxes_disjoint(box2, box3):
                continue
            placed = [anchor, cross2, cross3]
            if _cluster_overlap_load(placed) <= 5:
                return placed
    raise RuntimeError("failed to assemble a valid 3-cross cluster")


def _cluster_overlap_load(crosses: list[dict[str, int | tuple[int, int]]]) -> int:
    counts: defaultdict[tuple[int, int], int] = defaultdict(int)
    for cross in crosses:
        pattern = _cross_pattern(
            tuple(cross["center"]),
            int(cross["arm_len"]),
            int(cross["arm_color"]),
            int(cross["center_color"]),
        )
        for pos in pattern:
            counts[pos] += 1
    return sum(count > 1 for count in counts.values())


def _translate_cluster(
    crosses: list[dict[str, int | tuple[int, int]]]
) -> tuple[tuple[int, int], list[dict[str, int | tuple[int, int]]]]:
    min_r = min(int(cross["center"][0]) - int(cross["arm_len"]) - 1 for cross in crosses)
    max_r = max(int(cross["center"][0]) + int(cross["arm_len"]) + 1 for cross in crosses)
    min_c = min(int(cross["center"][1]) - int(cross["arm_len"]) - 1 for cross in crosses)
    max_c = max(int(cross["center"][1]) + int(cross["arm_len"]) + 1 for cross in crosses)
    cluster_h = max_r - min_r + 1
    cluster_w = max_c - min_c + 1

    height_lo = max(_GRID_MIN, cluster_h + 2)
    width_lo = max(_GRID_MIN, cluster_w + 2)
    height = random.randint(height_lo, _GRID_MAX)
    width = random.randint(width_lo, _GRID_MAX)
    shift_r = random.randint(1 - min_r, height - 2 - max_r)
    shift_c = random.randint(1 - min_c, width - 2 - max_c)
    shifted = [_shift_cross(cross, shift_r, shift_c) for cross in crosses]
    return (height, width), shifted


def _build_input(shape: tuple[int, int], crosses: list[dict[str, int | tuple[int, int]]]) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    for cross in crosses:
        center = tuple(cross["center"])
        arm_len = int(cross["arm_len"])
        arm_color = int(cross["arm_color"])
        center_color = int(cross["center_color"])
        r, c = center
        grid[r, c] = center_color
        for step in range(1, arm_len + 1):
            grid[r - step, c] = arm_color
            grid[r + step, c] = arm_color
            grid[r, c - step] = arm_color
            grid[r, c + step] = arm_color
    return grid


def _overlap_analysis(
    crosses: list[dict[str, int | tuple[int, int]]]
) -> tuple[int, set[tuple[str, str]]]:
    occupancy: defaultdict[tuple[int, int], list[tuple[int, str]]] = defaultdict(list)
    for idx, cross in enumerate(crosses):
        pattern = _cross_pattern(
            tuple(cross["center"]),
            int(cross["arm_len"]),
            int(cross["arm_color"]),
            int(cross["center_color"]),
        )
        for pos, (_color, role) in pattern.items():
            occupancy[pos].append((idx, role))

    overlap_cells = 0
    role_pairs: set[tuple[str, str]] = set()
    for entries in occupancy.values():
        if len(entries) > 1:
            overlap_cells += 1
            for i in range(len(entries)):
                for j in range(i + 1, len(entries)):
                    role_pairs.add(tuple(sorted((entries[i][1], entries[j][1]))))
    return overlap_cells, role_pairs


def _validate_generated(shape: tuple[int, int], crosses: list[dict[str, int | tuple[int, int]]]) -> bool:
    inp = _build_input(shape, crosses)
    out = main(inp)
    detected = _detect_crosses(inp)
    if len(detected) != len(crosses):
        return False

    true_lengths = sorted(int(cross["arm_len"]) for cross in crosses)
    found_lengths = sorted(int(cross["arm_len"]) for cross in detected)
    if true_lengths != found_lengths:
        return False

    overlap_cells, role_pairs = _overlap_analysis(crosses)
    if overlap_cells < 1 or overlap_cells > 5:
        return False
    if not any("center" in pair for pair in role_pairs):
        return False

    if np.array_equal(inp, out):
        return False
    if np.array_equal(out, _render_output(shape, crosses, use_diagonals=False, high_priority_wins=True)):
        return False
    if np.array_equal(out, _render_output(shape, crosses, use_orthogonal_extension=False, high_priority_wins=True)):
        return False
    if np.array_equal(out, _render_output(shape, crosses, high_priority_wins=False)):
        return False

    visible_added_diagonals = 0
    visible_extended_arms = 0
    for cross in crosses:
        r, c = tuple(cross["center"])
        center_color = int(cross["center_color"])
        arm_color = int(cross["arm_color"])
        arm_len = int(cross["arm_len"])
        if any(out[r + dr, c + dc] == center_color for dr, dc in _DIAG_DIRS):
            visible_added_diagonals += 1
        if any(
            out[r + dr * (arm_len + 1), c + dc * (arm_len + 1)] == arm_color
            for dr, dc in _ORTHO_DIRS
        ):
            visible_extended_arms += 1
    if visible_added_diagonals < 2 or visible_extended_arms < len(crosses):
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRIES):
        count = random.choice([2, 2, 3, 3])
        try:
            cluster = _pick_cluster(count)
        except RuntimeError:
            continue
        shape, shifted = _translate_cluster(cluster)
        if _validate_generated(shape, shifted):
            return _build_input(shape, shifted)
    raise RuntimeError("failed to generate a valid layered-cross input")
