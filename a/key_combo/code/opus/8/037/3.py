from __future__ import annotations

import random
from typing import Iterable

import numpy as np


BACKGROUND = 0
WALL = 5
MARKER_COLORS = (1, 2, 3, 4, 7)

HEIGHT_RANGE = (20, 30)
WIDTH_RANGE = (22, 30)
RADIUS_RANGE = (4, 6)
DIAMOND_COUNT_CHOICES = (2, 3)
PAIR_GAP_RANGE = (2, 4)
VERTICAL_GAP_RANGE = (2, 4)
MARKER_COUNT_RANGE = (3, 5)
MARKER_MIN_SPACING = 3
GENERATION_RETRIES = 500
TIE_PROBABILITY = 0.35


def _round_half_up_average(a: int, b: int) -> int:
    return (int(a) + int(b) + 1) // 2


def _manhattan(a: tuple[int, int], b: tuple[int, int]) -> int:
    return abs(int(a[0]) - int(b[0])) + abs(int(a[1]) - int(b[1]))


def _diamond_outline(center_r: int, center_c: int, radius: int) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    for dr in range(-radius, radius + 1):
        row = center_r + dr
        dc = radius - abs(dr)
        if dc == 0:
            cells.add((row, center_c))
        else:
            cells.add((row, center_c - dc))
            cells.add((row, center_c + dc))
    return cells


def _diamond_contains(center_r: int, center_c: int, radius: int, r: int, c: int) -> bool:
    return abs(int(r) - int(center_r)) + abs(int(c) - int(center_c)) <= int(radius)


def _diamond_row_walls(diamond: dict[str, object], row: int) -> list[tuple[int, int]]:
    radius = int(diamond["radius"])
    center_r = int(diamond["r"])
    center_c = int(diamond["c"])
    dr = abs(int(row) - center_r)
    if dr > radius:
        return []
    dc = radius - dr
    idx = int(diamond["id"])
    if dc == 0:
        return [(center_c, idx)]
    return [(center_c - dc, idx), (center_c + dc, idx)]


def _grid_row_intervals(grid: np.ndarray) -> list[dict[str, int]]:
    intervals: list[dict[str, int]] = []
    for r in range(grid.shape[0]):
        wall_cols = [int(c) for c in np.flatnonzero(grid[r] == WALL)]
        for left, right in zip(wall_cols, wall_cols[1:]):
            if right - left <= 1:
                continue
            intervals.append(
                {
                    "row": int(r),
                    "left": int(left),
                    "right": int(right),
                    "center_c": _round_half_up_average(left, right),
                }
            )
    return intervals


def _intervals_from_diamonds(height: int, diamonds: Iterable[dict[str, object]]) -> list[dict[str, int]]:
    intervals: list[dict[str, int]] = []
    for row in range(int(height)):
        wall_points: list[tuple[int, int]] = []
        for diamond in diamonds:
            wall_points.extend(_diamond_row_walls(diamond, row))
        wall_points.sort()
        for (left, left_id), (right, right_id) in zip(wall_points, wall_points[1:]):
            if right - left <= 1:
                continue
            intervals.append(
                {
                    "row": int(row),
                    "left": int(left),
                    "right": int(right),
                    "center_c": _round_half_up_average(left, right),
                    "left_id": int(left_id),
                    "right_id": int(right_id),
                }
            )
    return intervals


def _marker_winner(
    markers: list[dict[str, int]],
    center: tuple[int, int],
) -> tuple[int, bool]:
    ranked: list[tuple[int, int, int, int]] = []
    for idx, marker in enumerate(markers):
        ranked.append(
            (
                _manhattan(center, (marker["r"], marker["c"])),
                int(marker["r"]),
                int(marker["c"]),
                idx,
            )
        )
    ranked.sort()
    best_dist = ranked[0][0]
    tie = sum(item[0] == best_dist for item in ranked) > 1
    return int(ranked[0][3]), tie


def _solve_from_grid(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    markers = [
        {"r": int(r), "c": int(c), "color": int(g[r, c])}
        for r, c in np.argwhere((g != BACKGROUND) & (g != WALL))
    ]
    assert markers, "expected singleton marker pixels"

    for interval in _grid_row_intervals(g):
        winner_idx, _ = _marker_winner(markers, (interval["row"], interval["center_c"]))
        fill_color = int(markers[winner_idx]["color"])
        row = int(interval["row"])
        left = int(interval["left"])
        right = int(interval["right"])
        segment = out[row, left + 1 : right]
        segment[segment == BACKGROUND] = fill_color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each diamond-bounded row interval with the color of the nearest marker to that interval's center.

    Input structure:
    - A black background contains 2-3 hollow grey diamond outlines made of diagonal walls.
    - Several singleton marker pixels of distinct non-grey colors lie outside every diamond.
    - Some rows are crossed by walls from multiple diamonds, so one row can contain multiple closed intervals.

    Rule:
    - For each row, collect every grey wall pixel and sort their column positions.
    - Every background segment strictly between two adjacent wall pixels is a closed scanline interval; open prefixes/suffixes at the row edges are ignored.
    - For each closed interval, take the center point at that row and at the half-up rounded average of the two wall columns, then choose the marker with minimum Manhattan distance to that center.
    - Break equal distances by smaller marker row, then smaller marker column, and recolor only the background cells of that interval with the chosen marker's color.
    - Keep the grey wall pixels and the original marker pixels unchanged.

    Key insight:
    - The diamonds matter only as row-dependent wall generators: every row must be scanned independently, and each interval's color comes from a nearest-marker comparison at that interval center rather than from a per-diamond or global fill color.
    """
    return _solve_from_grid(grid)


def _sample_pair_diamonds(height: int, width: int) -> list[dict[str, object]] | None:
    radius_a = random.randint(*RADIUS_RANGE)
    radius_b = random.randint(*RADIUS_RANGE)
    gap = random.randint(*PAIR_GAP_RANGE)
    pair_width = (2 * radius_a + 1) + gap + (2 * radius_b + 1)
    if pair_width > width - 2:
        return None

    row_min = max(radius_a, radius_b)
    row_max = min(height - 1 - radius_a, height - 1 - radius_b)
    if row_min > row_max:
        return None
    base_row = random.randint(row_min, row_max)

    left_margin = random.randint(1, width - pair_width - 1)
    center_a = left_margin + radius_a
    center_b = left_margin + (2 * radius_a + 1) + gap + radius_b

    return [
        {"id": 0, "r": base_row, "c": center_a, "radius": radius_a},
        {"id": 1, "r": base_row, "c": center_b, "radius": radius_b},
    ]


def _sample_three_diamonds(height: int, width: int) -> list[dict[str, object]] | None:
    pair = _sample_pair_diamonds(height, width)
    if pair is None:
        return None

    radius_c = random.randint(*RADIUS_RANGE)
    pair_row = int(pair[0]["r"])
    pair_top = min(int(d["r"]) - int(d["radius"]) for d in pair)
    pair_bottom = max(int(d["r"]) + int(d["radius"]) for d in pair)
    pair_left = min(int(d["c"]) - int(d["radius"]) for d in pair)
    pair_right = max(int(d["c"]) + int(d["radius"]) for d in pair)
    gap_v = random.randint(*VERTICAL_GAP_RANGE)

    row_choices: list[int] = []
    above = pair_top - gap_v - radius_c
    below = pair_bottom + gap_v + radius_c
    if radius_c <= above <= height - 1 - radius_c:
        row_choices.append(above)
    if radius_c <= below <= height - 1 - radius_c:
        row_choices.append(below)
    if not row_choices:
        return None

    row_c = random.choice(row_choices)
    center_targets = [
        int((int(pair[0]["c"]) + int(pair[1]["c"])) // 2),
        pair_left + radius_c,
        pair_right - radius_c,
    ]
    col_options = [
        col
        for col in center_targets
        if radius_c <= int(col) <= width - 1 - radius_c
    ]
    if not col_options:
        return None

    random.shuffle(col_options)
    for col_c in col_options:
        overlap = False
        for diamond in pair:
            if (
                abs(int(row_c) - int(diamond["r"])) <= radius_c + int(diamond["radius"])
                and abs(int(col_c) - int(diamond["c"])) <= radius_c + int(diamond["radius"])
            ):
                if _diamond_outline(int(row_c), int(col_c), radius_c) & _diamond_outline(
                    int(diamond["r"]), int(diamond["c"]), int(diamond["radius"])
                ):
                    overlap = True
                    break
        if not overlap:
            pair.append({"id": 2, "r": row_c, "c": int(col_c), "radius": radius_c})
            return pair
    return None


def _sample_diamonds(height: int, width: int) -> list[dict[str, object]] | None:
    count = random.choice(DIAMOND_COUNT_CHOICES)
    if count == 2:
        diamonds = _sample_pair_diamonds(height, width)
    else:
        diamonds = _sample_three_diamonds(height, width)
    if diamonds is None:
        return None
    for diamond in diamonds:
        diamond["outline"] = _diamond_outline(
            int(diamond["r"]), int(diamond["c"]), int(diamond["radius"])
        )
    return diamonds


def _scene_bbox(diamonds: list[dict[str, object]]) -> tuple[int, int, int, int]:
    r0 = min(int(d["r"]) - int(d["radius"]) for d in diamonds)
    c0 = min(int(d["c"]) - int(d["radius"]) for d in diamonds)
    r1 = max(int(d["r"]) + int(d["radius"]) for d in diamonds)
    c1 = max(int(d["c"]) + int(d["radius"]) for d in diamonds)
    return r0, c0, r1, c1


def _is_valid_marker_cell(
    r: int,
    c: int,
    height: int,
    width: int,
    diamonds: list[dict[str, object]],
    used: set[tuple[int, int]],
) -> bool:
    if not (0 <= r < height and 0 <= c < width):
        return False
    if (r, c) in used:
        return False
    for ur, uc in used:
        if _manhattan((r, c), (ur, uc)) < MARKER_MIN_SPACING:
            return False
    for diamond in diamonds:
        if _diamond_contains(int(diamond["r"]), int(diamond["c"]), int(diamond["radius"]), r, c):
            return False
    return True


def _snap_to_valid(
    target: tuple[int, int],
    height: int,
    width: int,
    diamonds: list[dict[str, object]],
    used: set[tuple[int, int]],
) -> tuple[int, int] | None:
    candidates: list[tuple[int, float, int, int]] = []
    for r in range(height):
        for c in range(width):
            if not _is_valid_marker_cell(r, c, height, width, diamonds, used):
                continue
            distance = _manhattan((r, c), target)
            candidates.append((distance, random.random(), r, c))
    if not candidates:
        return None
    candidates.sort()
    _, _, r, c = candidates[0]
    return int(r), int(c)


def _place_tie_markers(
    intervals: list[dict[str, int]],
    diamonds: list[dict[str, object]],
    height: int,
    width: int,
    used: set[tuple[int, int]],
) -> tuple[list[tuple[int, int]], str] | None:
    between = [interval for interval in intervals if interval["left_id"] != interval["right_id"]]
    interior = [interval for interval in intervals if interval["left_id"] == interval["right_id"]]

    modes = ["row", "col"]
    random.shuffle(modes)
    for mode in modes:
        if mode == "row":
            pool = between or intervals
            random.shuffle(pool)
            for interval in pool:
                r = int(interval["row"])
                c = int(interval["center_c"])
                for d in range(3, 6):
                    options = [((r - d, c), (r, c - d)), ((r - d, c), (r, c + d))]
                    random.shuffle(options)
                    for a, b in options:
                        if _is_valid_marker_cell(a[0], a[1], height, width, diamonds, used) and _is_valid_marker_cell(
                            b[0], b[1], height, width, diamonds, used | {a}
                        ):
                            return [a, b], "row"
        else:
            pool = interior[:]
            random.shuffle(pool)
            for interval in pool:
                if interval["left_id"] != interval["right_id"]:
                    continue
                diamond = diamonds[int(interval["left_id"])]
                r = int(interval["row"])
                c = int(interval["center_c"])
                d = int(diamond["radius"]) + 2
                a = (r, c - d)
                b = (r, c + d)
                if _is_valid_marker_cell(a[0], a[1], height, width, diamonds, used) and _is_valid_marker_cell(
                    b[0], b[1], height, width, diamonds, used | {a}
                ):
                    return [a, b], "col"
    return None


def _sample_markers(
    height: int,
    width: int,
    diamonds: list[dict[str, object]],
    intervals: list[dict[str, int]],
    want_tie: bool,
) -> tuple[list[dict[str, int]], str | None] | None:
    bbox_r0, bbox_c0, bbox_r1, bbox_c1 = _scene_bbox(diamonds)
    used: set[tuple[int, int]] = set()
    positions: list[tuple[int, int]] = []
    tie_mode: str | None = None

    if want_tie:
        tie_positions = _place_tie_markers(intervals, diamonds, height, width, used)
        if tie_positions is None:
            return None
        cells, tie_mode = tie_positions
        for cell in cells:
            positions.append(cell)
            used.add(cell)

    count = random.randint(*MARKER_COUNT_RANGE)
    anchor_targets = [
        (bbox_r0 - 3, bbox_c0 + 1),
        (bbox_r1 + 3, (bbox_c0 + bbox_c1) // 2),
        (bbox_r0 + 1, bbox_c1 + 4),
        ((bbox_r0 + bbox_r1) // 2, bbox_c0 - 4),
        (bbox_r1 - 1, bbox_c1 + 3),
        (bbox_r0 - 4, (bbox_c0 + bbox_c1) // 2 + 2),
        (bbox_r1 + 4, bbox_c0 + 2),
    ]
    random.shuffle(anchor_targets)

    while len(positions) < count and anchor_targets:
        target = anchor_targets.pop()
        snapped = _snap_to_valid(target, height, width, diamonds, used)
        if snapped is None:
            continue
        positions.append(snapped)
        used.add(snapped)

    while len(positions) < count:
        random_target = (random.randint(0, height - 1), random.randint(0, width - 1))
        snapped = _snap_to_valid(random_target, height, width, diamonds, used)
        if snapped is None:
            return None
        positions.append(snapped)
        used.add(snapped)

    colors = random.sample(MARKER_COLORS, len(positions))
    markers = [
        {"r": int(r), "c": int(c), "color": int(color)}
        for (r, c), color in zip(positions, colors)
    ]
    return markers, tie_mode


def _render_input(
    height: int,
    width: int,
    diamonds: list[dict[str, object]],
    markers: list[dict[str, int]],
) -> np.ndarray:
    grid = np.zeros((height, width), dtype=int)
    for diamond in diamonds:
        for r, c in diamond["outline"]:
            grid[r, c] = WALL
    for marker in markers:
        grid[int(marker["r"]), int(marker["c"])] = int(marker["color"])
    return grid


def _metadata_output(
    height: int,
    width: int,
    diamonds: list[dict[str, object]],
    markers: list[dict[str, int]],
    grid: np.ndarray,
) -> tuple[np.ndarray, list[dict[str, int]]]:
    out = grid.copy()
    annotated: list[dict[str, int]] = []
    for interval in _intervals_from_diamonds(height, diamonds):
        winner_idx, tie = _marker_winner(markers, (interval["row"], interval["center_c"]))
        fill_color = int(markers[winner_idx]["color"])
        row = int(interval["row"])
        for col in range(int(interval["left"]) + 1, int(interval["right"])):
            if out[row, col] == BACKGROUND:
                out[row, col] = fill_color
        annotated.append(
            {
                **interval,
                "winner": int(winner_idx),
                "color": fill_color,
                "tie": int(tie),
            }
        )
    return out, annotated


def _semantic_checks(
    height: int,
    width: int,
    diamonds: list[dict[str, object]],
    markers: list[dict[str, int]],
    annotated: list[dict[str, int]],
    want_tie: bool,
    tie_mode: str | None,
    grid: np.ndarray,
    expected: np.ndarray,
) -> bool:
    if not (2 <= len(diamonds) <= 3):
        return False
    if not (3 <= len(markers) <= 5):
        return False
    if len({marker["color"] for marker in markers}) != len(markers):
        return False
    if not np.any(expected != grid):
        return False

    overlap_rows = set()
    for row in range(height):
        ids = set()
        for diamond in diamonds:
            if _diamond_row_walls(diamond, row):
                ids.add(int(diamond["id"]))
        if len(ids) >= 2:
            overlap_rows.add(row)
    if not overlap_rows:
        return False

    if not any(item["left_id"] != item["right_id"] for item in annotated):
        return False
    if len({item["winner"] for item in annotated}) < 2:
        return False

    row_to_colors: dict[int, set[int]] = {}
    for item in annotated:
        row_to_colors.setdefault(int(item["row"]), set()).add(int(item["color"]))
    if not any(len(colors) >= 2 for colors in row_to_colors.values()):
        return False

    varied_interior = False
    for diamond in diamonds:
        colors = {
            int(item["color"])
            for item in annotated
            if item["left_id"] == diamond["id"] == item["right_id"]
        }
        if len(colors) >= 2:
            varied_interior = True
            break
    if not varied_interior:
        return False

    tie_rows = [item for item in annotated if item["tie"]]
    if want_tie and not tie_rows:
        return False
    if not want_tie and tie_rows:
        return False
    if want_tie and tie_mode == "col":
        if not any(item["tie"] for item in annotated if item["left_id"] == item["right_id"]):
            return False

    if _grid_row_intervals(grid) != [
        {k: item[k] for k in ("row", "left", "right", "center_c")} for item in annotated
    ]:
        return False

    solver_out = main(grid)
    if not np.array_equal(solver_out, expected):
        return False
    if solver_out.shape != (height, width) or max(solver_out.shape) > 30:
        return False
    return True


def _generate_one() -> np.ndarray | None:
    height = random.randint(*HEIGHT_RANGE)
    width = random.randint(*WIDTH_RANGE)
    diamonds = _sample_diamonds(height, width)
    if diamonds is None:
        return None

    intervals = _intervals_from_diamonds(height, diamonds)
    if not intervals:
        return None

    want_tie = random.random() < TIE_PROBABILITY
    marker_result = _sample_markers(height, width, diamonds, intervals, want_tie)
    if marker_result is None:
        return None
    markers, tie_mode = marker_result

    grid = _render_input(height, width, diamonds, markers)
    expected, annotated = _metadata_output(height, width, diamonds, markers, grid)
    if not _semantic_checks(
        height, width, diamonds, markers, annotated, want_tie, tie_mode, grid, expected
    ):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _generate_one()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
