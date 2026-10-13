from __future__ import annotations

import itertools
import random

import numpy as np


BG = 0
BASELINE = 1
RED = 2
GREEN = 3
YELLOW = 4
SHELF = 5

DISTANCE_BY_COLOR = {
    RED: 3,
    GREEN: 4,
    YELLOW: 5,
}


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _find_baseline_row(g: np.ndarray) -> int:
    candidates: list[int] = []
    allowed = {BASELINE, *DISTANCE_BY_COLOR}
    for r in range(g.shape[0]):
        row = g[r]
        if np.all(row != BG) and set(map(int, row)).issubset(allowed) and BASELINE in row:
            candidates.append(int(r))
    assert candidates, "expected one full-width baseline row"
    return candidates[-1]


def _scan_shelves(g: np.ndarray) -> dict[int, list[tuple[int, int]]]:
    shelves: dict[int, list[tuple[int, int]]] = {}
    h, w = g.shape
    for r in range(h):
        segments: list[tuple[int, int]] = []
        c = 0
        while c < w:
            if int(g[r, c]) != SHELF:
                c += 1
                continue
            c0 = c
            while c + 1 < w and int(g[r, c + 1]) == SHELF:
                c += 1
            segments.append((int(c0), int(c)))
            c += 1
        if segments:
            shelves[int(r)] = segments
    return shelves


def _segment_containing(
    shelves: dict[int, list[tuple[int, int]]], row: int, col: int
) -> tuple[int, int] | None:
    for c0, c1 in shelves.get(int(row), []):
        if c0 <= int(col) <= c1:
            return int(c0), int(c1)
    return None


def _collect_markers(g: np.ndarray, baseline_row: int) -> list[tuple[int, int]]:
    markers: list[tuple[int, int]] = []
    for c, color in enumerate(map(int, g[baseline_row])):
        if color in DISTANCE_BY_COLOR:
            markers.append((int(c), int(color)))
    assert 2 <= len(markers) <= 3, "expected two or three distinct markers"
    assert len({color for _, color in markers}) == len(markers), "markers must use distinct colors"
    return markers


def _trace_marker(
    g: np.ndarray,
    shelves: dict[int, list[tuple[int, int]]],
    start_r: int,
    start_c: int,
    color: int,
) -> dict:
    h, w = g.shape
    distance = DISTANCE_BY_COLOR[int(color)]
    painted: set[tuple[int, int]] = set()
    hit_rows: list[int] = []
    spread_intervals: list[tuple[int, int, int]] = []
    branch_starts: set[tuple[int, int]] = set()
    seen_starts: set[tuple[int, int]] = set()
    stack: list[tuple[int, int]] = [(int(start_r), int(start_c))]

    while stack:
        r, c = stack.pop()
        if (r, c) in seen_starts:
            continue
        seen_starts.add((r, c))
        if _in_bounds(g.shape, r, c) and int(g[r, c]) != SHELF:
            painted.add((r, c))

        cur_r = int(r)
        for _ in range(distance):
            next_r = cur_r - 1
            if next_r < 0:
                break
            if int(g[next_r, c]) == SHELF:
                segment = _segment_containing(shelves, next_r, c)
                assert segment is not None, "shelf must be horizontal"
                left_shelf, right_shelf = segment
                left_fill = max(int(c - distance), int(left_shelf - 1))
                right_fill = min(int(c + distance), int(right_shelf + 1))
                for x in range(left_fill, right_fill + 1):
                    painted.add((cur_r, int(x)))
                hit_rows.append(int(next_r))
                spread_intervals.append((int(cur_r), int(left_fill), int(right_fill)))
                if left_shelf - 1 >= 0 and c - distance <= left_shelf - 1:
                    branch_starts.add((int(cur_r), int(left_shelf - 1)))
                    stack.append((int(cur_r), int(left_shelf - 1)))
                if right_shelf + 1 < w and c + distance >= right_shelf + 1:
                    branch_starts.add((int(cur_r), int(right_shelf + 1)))
                    stack.append((int(cur_r), int(right_shelf + 1)))
                break
            cur_r = int(next_r)
            painted.add((cur_r, c))

    return {
        "painted": painted,
        "hit_rows": hit_rows,
        "spread_intervals": spread_intervals,
        "branch_starts": branch_starts,
    }


def _paint_marker(
    out: np.ndarray,
    g: np.ndarray,
    shelves: dict[int, list[tuple[int, int]]],
    start_r: int,
    start_c: int,
    color: int,
) -> None:
    distance = DISTANCE_BY_COLOR[int(color)]
    stack: list[tuple[int, int]] = [(int(start_r), int(start_c))]
    seen_starts: set[tuple[int, int]] = set()
    origin = (int(start_r), int(start_c))

    def paint(r: int, c: int) -> None:
        if not _in_bounds(out.shape, r, c) or int(g[r, c]) == SHELF:
            return
        if (r, c) == origin or int(out[r, c]) in (BG, int(color)):
            out[r, c] = int(color)

    while stack:
        r, c = stack.pop()
        if (r, c) in seen_starts:
            continue
        seen_starts.add((r, c))
        paint(int(r), int(c))

        cur_r = int(r)
        for _ in range(distance):
            next_r = cur_r - 1
            if next_r < 0:
                break
            if int(g[next_r, c]) == SHELF:
                segment = _segment_containing(shelves, next_r, c)
                assert segment is not None, "shelf must be horizontal"
                left_shelf, right_shelf = segment
                left_fill = max(int(c - distance), int(left_shelf - 1))
                right_fill = min(int(c + distance), int(right_shelf + 1))
                for x in range(left_fill, right_fill + 1):
                    paint(int(cur_r), int(x))
                if left_shelf - 1 >= 0 and c - distance <= left_shelf - 1:
                    stack.append((int(cur_r), int(left_shelf - 1)))
                if right_shelf + 1 < out.shape[1] and c + distance >= right_shelf + 1:
                    stack.append((int(cur_r), int(right_shelf + 1)))
                break
            cur_r = int(next_r)
            paint(int(cur_r), int(c))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fire color-coded upward rays from baseline markers, let them spread under shelves, and continue the cascade from shelf overflows.

    Input structure:
    - The background is black, the bottom control row is a full-width blue baseline,
    - Two or three distinct red/green/yellow marker pixels are embedded in that baseline,
    - One-pixel-thick grey horizontal shelf segments sit above the baseline on otherwise empty rows.

    Rule:
    - Each marker launches a vertical upward ray in its own color.
    - The marker color encodes both the maximum vertical travel and the left/right spread width under any shelf:
      red -> 3, green -> 4, yellow -> 5.
    - A ray paints background cells until it either exhausts its travel budget or sees a shelf directly above;
      in the shelf case it stops one row below the shelf, paints horizontally under that shelf across its color-coded width,
      clipped to at most one cell beyond each shelf end, then launches new upward rays from the reached overflow cells.
    - Repeat the same shelf-hit behavior for every new branch, keep all original shelves and the baseline, and when two markers
      try to claim the same painted pixel, preserve the color from the shorter-range marker by processing markers in distance order.

    Key insight:
    - A single visible marker color controls two independent geometric parameters at once: vertical reach and horizontal shelf spread,
      so the final cascade tree can only be reconstructed by combining both uses of the same color code.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    baseline_row = _find_baseline_row(g)
    shelves = _scan_shelves(g)
    markers = _collect_markers(g, baseline_row)
    markers.sort(key=lambda item: (DISTANCE_BY_COLOR[item[1]], item[0]))

    for col, color in markers:
        _paint_marker(out, g, shelves, baseline_row, col, color)

    return out


def _sample_marker_colors() -> list[int]:
    if random.random() < 0.4:
        return sorted(random.sample(list(DISTANCE_BY_COLOR), 2), key=DISTANCE_BY_COLOR.get)
    return [RED, GREEN, YELLOW]


def _sample_centers(count: int, width: int) -> list[int] | None:
    candidates = list(range(3, width - 3))
    for _ in range(200):
        cols = sorted(random.sample(candidates, count))
        if all(b - a >= 8 for a, b in zip(cols, cols[1:])):
            return cols
    return None


def _first_shelf_rows(baseline_row: int, colors: list[int]) -> list[int] | None:
    options = []
    for color in colors:
        distance = DISTANCE_BY_COLOR[color]
        options.append([baseline_row - steps - 1 for steps in range(1, distance)])

    best: tuple[int, int, tuple[int, ...]] | None = None
    for rows in itertools.product(*options):
        if len(set(rows)) != len(rows):
            continue
        ordered = sorted(rows)
        gaps = [b - a for a, b in zip(ordered, ordered[1:])]
        min_gap = min(gaps) if gaps else 99
        score = (int(min_gap), int(sum(gaps)))
        if best is None or score > best[:2]:
            best = (score[0], score[1], tuple(map(int, rows)))
    if best is None:
        return None
    return list(best[2])


def _first_geometry_candidates(
    center: int,
    distance: int,
    lane_left: int,
    lane_right: int,
) -> list[tuple[int, int, int, int]]:
    candidates: list[tuple[int, int, int, int]] = []
    for left_reach in range(1, distance + 1):
        for right_reach in range(1, distance + 1):
            shelf_width = left_reach + right_reach - 1
            if not (3 <= shelf_width <= 5):
                continue
            left_overflow = center - left_reach
            right_overflow = center + right_reach
            if left_overflow < lane_left or right_overflow > lane_right:
                continue
            shelf_left = left_overflow + 1
            shelf_right = right_overflow - 1
            candidates.append(
                (
                    int(shelf_left),
                    int(shelf_right),
                    int(left_overflow),
                    int(right_overflow),
                )
            )
    return candidates


def _extra_geometry_candidates(
    center: int,
    lane_left: int,
    lane_right: int,
    banned_cols: set[int] | None = None,
) -> list[tuple[int, int]]:
    banned_cols = banned_cols or set()
    candidates: list[tuple[int, int]] = []
    for width in range(3, 6):
        left_min = max(int(lane_left), int(center - width + 1))
        left_max = min(int(center), int(lane_right - width + 1))
        for shelf_left in range(left_min, left_max + 1):
            shelf_right = int(shelf_left + width - 1)
            if any(shelf_left <= col <= shelf_right for col in banned_cols):
                continue
            candidates.append((int(shelf_left), int(shelf_right)))
    return candidates


def _extra_shelf_row_candidates(
    start_row: int,
    distance: int,
    used_rows: set[int],
) -> list[int]:
    rows: list[int] = []
    for steps in range(1, distance):
        row = int(start_row - steps - 1)
        if row <= 0 or row in used_rows:
            continue
        rows.append(row)
    return rows


def _place_shelf(g: np.ndarray, row: int, left: int, right: int) -> None:
    g[int(row), int(left) : int(right) + 1] = SHELF


def _build_case() -> np.ndarray | None:
    colors = _sample_marker_colors()
    random.shuffle(colors)
    marker_count = len(colors)

    height = random.randint(17, 24)
    width_lo = 14 if marker_count == 2 else 23
    width_hi = 22 if marker_count == 2 else 29
    width = random.randint(width_lo, width_hi)
    baseline_row = height - 1

    centers = _sample_centers(marker_count, width)
    if centers is None:
        return None

    first_rows = _first_shelf_rows(baseline_row, colors)
    if first_rows is None:
        return None

    lanes = [(int(center - 3), int(center + 3)) for center in centers]
    g = np.zeros((height, width), dtype=int)
    g[baseline_row, :] = BASELINE

    marker_specs: list[dict] = []
    used_rows = set()
    for center, color, first_row, lane in zip(centers, colors, first_rows, lanes):
        lane_left = lane[0] + 1
        lane_right = lane[1] - 1
        if lane_right - lane_left < 4:
            return None

        geom = _first_geometry_candidates(center, DISTANCE_BY_COLOR[color], lane_left, lane_right)
        if not geom:
            return None
        shelf_left, shelf_right, left_overflow, right_overflow = random.choice(geom)
        _place_shelf(g, first_row, shelf_left, shelf_right)
        used_rows.add(int(first_row))
        g[baseline_row, center] = int(color)
        marker_specs.append(
            {
                "center": int(center),
                "color": int(color),
                "distance": int(DISTANCE_BY_COLOR[color]),
                "lane_left": int(lane_left),
                "lane_right": int(lane_right),
                "first_row": int(first_row),
                "left_overflow": int(left_overflow),
                "right_overflow": int(right_overflow),
            }
        )

    extra_budget = 1 if marker_count == 3 else random.choice([1, 1, 2])
    preferred = sorted(marker_specs, key=lambda item: (-item["distance"], random.random()))
    extras_added = 0

    for spec in preferred:
        if extras_added >= extra_budget:
            break
        start_row = spec["first_row"] + 1
        sibling_cols = {spec["left_overflow"], spec["right_overflow"]}
        branch_order = [spec["left_overflow"], spec["right_overflow"]]
        random.shuffle(branch_order)

        for branch_col in branch_order:
            row_candidates = _extra_shelf_row_candidates(start_row, spec["distance"], used_rows)
            if not row_candidates:
                continue
            random.shuffle(row_candidates)

            banned = sibling_cols - {branch_col}
            geom_candidates = _extra_geometry_candidates(
                branch_col,
                spec["lane_left"],
                spec["lane_right"],
                banned_cols=banned,
            )
            if not geom_candidates:
                continue
            random.shuffle(geom_candidates)

            placed = False
            for row in row_candidates:
                for shelf_left, shelf_right in geom_candidates:
                    if any(int(g[row, c]) != BG for c in range(shelf_left, shelf_right + 1)):
                        continue
                    _place_shelf(g, row, shelf_left, shelf_right)
                    used_rows.add(int(row))
                    extras_added += 1
                    placed = True
                    break
                if placed:
                    break
            if placed:
                break

    shelves = _scan_shelves(g)
    shelf_count = sum(len(v) for v in shelves.values())
    if not (2 <= shelf_count <= 4):
        return None

    if not _validate_generated_case(g):
        return None
    return g


def _validate_generated_case(g: np.ndarray) -> bool:
    try:
        baseline_row = _find_baseline_row(g)
        markers = _collect_markers(g, baseline_row)
        shelves = _scan_shelves(g)
        shelf_count = sum(len(v) for v in shelves.values())
        if not (2 <= shelf_count <= 4):
            return False

        traces: list[dict] = []
        for col, color in markers:
            trace = _trace_marker(g, shelves, baseline_row, col, color)
            if len(trace["hit_rows"]) < 1:
                return False
            if len(trace["branch_starts"]) < 2:
                return False
            if any(r <= 1 for r, _ in trace["branch_starts"]):
                return False
            traces.append(trace)

        total_hits = sum(len(trace["hit_rows"]) for trace in traces)
        if total_hits <= len(markers):
            return False

        for i in range(len(traces)):
            for j in range(i + 1, len(traces)):
                if traces[i]["painted"] & traces[j]["painted"]:
                    return False

        output = main(g)
        if output.shape != g.shape:
            return False
        if np.array_equal(output, g):
            return False

        color_pixel_counts = {
            color: int(np.count_nonzero(output == color)) - int(np.count_nonzero(g == color))
            for _col, color in markers
        }
        if any(count <= 4 for count in color_pixel_counts.values()):
            return False

        horizontal_spreads = 0
        for trace in traces:
            for _row, left, right in trace["spread_intervals"]:
                if right - left + 1 >= 4:
                    horizontal_spreads += 1
        if horizontal_spreads < len(markers):
            return False

        return True
    except AssertionError:
        return False


def generate_input() -> np.ndarray:
    for _ in range(500):
        grid = _build_case()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid ray-cascade input")
