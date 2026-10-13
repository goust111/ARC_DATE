from __future__ import annotations

import random
from collections import Counter, defaultdict

import numpy as np


BG = 0
YELLOW = 4
ORANGE = 7
EDGE_COLORS = {YELLOW, ORANGE}
PAIR_COLORS = [1, 2, 3, 6, 8, 9]
STRUCT4_DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, np.asarray(grid).ravel())).most_common(1)[0][0])


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def bbox_h(bbox: tuple[int, int, int, int]) -> int:
    return int(bbox[2] - bbox[0] + 1)


def bbox_w(bbox: tuple[int, int, int, int]) -> int:
    return int(bbox[3] - bbox[1] + 1)


def find_components(
    grid: np.ndarray,
    *,
    bg: int,
    colors: list[int] | None = None,
    connectivity: int = 4,
) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    if connectivity != 4:
        raise ValueError("this solver uses 4-connectivity only")

    if colors is None:
        colors = sorted(set(map(int, g.ravel())) - {int(bg)})

    seen = np.zeros(g.shape, dtype=bool)
    out: list[dict] = []
    for color in colors:
        mask = g == int(color)
        for r0, c0 in np.argwhere(mask):
            if seen[int(r0), int(c0)]:
                continue
            stack = [(int(r0), int(c0))]
            coords: list[tuple[int, int]] = []
            seen[int(r0), int(c0)] = True
            while stack:
                r, c = stack.pop()
                coords.append((r, c))
                for dr, dc in STRUCT4_DIRS:
                    rr = r + dr
                    cc = c + dc
                    if 0 <= rr < g.shape[0] and 0 <= cc < g.shape[1] and not seen[rr, cc] and mask[rr, cc]:
                        seen[rr, cc] = True
                        stack.append((rr, cc))
            arr = np.array(coords, dtype=int)
            out.append(
                {
                    "color": int(color),
                    "coords": arr,
                    "bbox": bbox_coords(arr),
                    "area": int(arr.shape[0]),
                }
            )
    return out


def _detect_band_width(grid: np.ndarray, side: str) -> int:
    g = np.asarray(grid, dtype=int)
    H, W = g.shape
    if side == "left":
        cols = range(W)
    elif side == "right":
        cols = range(W - 1, -1, -1)
    else:
        raise ValueError("side must be left or right")

    width = 0
    for c in cols:
        col = g[:, int(c)]
        if np.all(np.isin(col, list(EDGE_COLORS))):
            width += 1
        else:
            break
    return int(width)


def _assert_solid_square(grid: np.ndarray, color: int, component: dict) -> tuple[int, int, int, int]:
    bbox = tuple(map(int, component["bbox"]))
    r0, c0, r1, c1 = bbox
    assert bbox_h(bbox) == bbox_w(bbox) in {2, 3}
    assert component["area"] == bbox_h(bbox) * bbox_w(bbox)
    assert np.all(grid[r0 : r1 + 1, c0 : c1 + 1] == int(color))
    return bbox


def _pair_geometry(
    upper: tuple[int, int, int, int],
    lower: tuple[int, int, int, int],
) -> tuple[tuple[int, int], tuple[int, int], int]:
    ur0, uc0, ur1, uc1 = map(int, upper)
    lr0, lc0, lr1, lc1 = map(int, lower)
    row_gap = int(lr0 - ur1)
    assert row_gap > 0

    if row_gap == int(lc0 - uc1):
        return (ur1, uc1), (lr0, lc0), 1
    if row_gap == int(uc0 - lc1):
        return (ur1, uc0), (lr0, lc1), -1
    raise AssertionError("same-color rectangles must be diagonally aligned by facing corners")


def _solve(grid: np.ndarray) -> tuple[np.ndarray, dict]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = int(get_bg(g))
    assert bg == BG

    left_w = _detect_band_width(g, "left")
    right_w = _detect_band_width(g, "right")
    assert left_w in {1, 2} and right_w in {1, 2}
    assert left_w == right_w
    assert left_w + right_w < g.shape[1]

    pair_colors = sorted(set(map(int, g.ravel())) - {bg} - EDGE_COLORS)
    assert 2 <= len(pair_colors) <= 3

    components = find_components(g, bg=bg, colors=pair_colors, connectivity=4)
    by_color: dict[int, list[dict]] = defaultdict(list)
    for comp in components:
        by_color[int(comp["color"])].append(comp)
    assert set(by_color) == set(pair_colors)

    out = g.copy()
    row_dirs: dict[int, set[int]] = defaultdict(set)
    pair_meta = []
    painted_line_cells = set()

    for color in pair_colors:
        parts = by_color[int(color)]
        assert len(parts) == 2
        boxes = [_assert_solid_square(g, int(color), part) for part in parts]
        boxes.sort(key=lambda b: (b[0], b[1]))
        upper, lower = boxes
        start, end, slope = _pair_geometry(upper, lower)

        dr = 1
        dc = 1 if slope == 1 else -1
        line_cells = []
        r, c = int(start[0] + dr), int(start[1] + dc)
        while (r, c) != end:
            assert g[r, c] == bg
            out[r, c] = int(color)
            assert (r, c) not in painted_line_cells
            painted_line_cells.add((r, c))
            line_cells.append((r, c))
            r += dr
            c += dc

        for row in range(int(start[0]), int(end[0]) + 1):
            row_dirs[int(row)].add(int(slope))

        pair_meta.append(
            {
                "color": int(color),
                "upper": upper,
                "lower": lower,
                "start": start,
                "end": end,
                "slope": int(slope),
                "rows": set(range(int(start[0]), int(end[0]) + 1)),
                "line_cells": tuple(line_cells),
            }
        )

    band_w = left_w
    left_slice = g[:, :band_w].copy()
    right_slice = g[:, g.shape[1] - band_w :].copy()
    for row, dirs in row_dirs.items():
        if dirs == {1}:
            out[row, g.shape[1] - band_w :] = left_slice[row]
            out[row, :band_w] = bg
        elif dirs == {-1}:
            out[row, :band_w] = right_slice[row]
            out[row, g.shape[1] - band_w :] = bg
        else:
            assert dirs == {1, -1}

    meta = {
        "band_w": int(band_w),
        "row_dirs": {int(r): set(map(int, dirs)) for r, dirs in row_dirs.items()},
        "pairs": pair_meta,
    }
    return out, meta


def main(grid: np.ndarray) -> np.ndarray:
    """
    Draw each same-color diagonal connector, then use its slope-coded row set to migrate edge-band slices across the grid.

    Input structure:
    - The background is black, with a full-height left edge band and right edge band made only of yellow and orange cells.
    - There are 2 or 3 foreground colors other than the edge-band colors, and each such color forms exactly two solid 2x2 or 3x3 squares.
    - Each same-color square pair is diagonally offset so the upper square's lower facing corner and the lower square's upper facing corner lie on one 45-degree diagonal.

    Rule:
    - For each same-color pair, keep both squares and draw the open diagonal segment between their facing corners using that pair's color.
    - Record every row touched by that diagonal, including the endpoint rows inside the two squares.
    - A NW-SE connector moves that row's left edge-band slice to the same row on the right edge and erases the original left slice; a NE-SW connector does the symmetric right-to-left move.
    - If a row is touched by both connector slopes, neither edge band moves on that row.

    Key insight:
    - The diagonal is both a geometric connector and a control signal: its slope determines migration direction, while its vertical span selects which edge-band rows are active.
    """
    return _solve(grid)[0]


def _expanded(cells: set[tuple[int, int]], shape: tuple[int, int]) -> set[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    H, W = shape
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr = r + dr
                cc = c + dc
                if 0 <= rr < H and 0 <= cc < W:
                    out.add((rr, cc))
    return out


def _square_cells(r0: int, c0: int, size: int) -> set[tuple[int, int]]:
    return {(r, c) for r in range(r0, r0 + size) for c in range(c0, c0 + size)}


def _build_pair(
    H: int,
    W: int,
    band_w: int,
    slope: int,
    interval: tuple[int, int],
    color: int,
    blocked: set[tuple[int, int]],
) -> dict | None:
    s_row, e_row = map(int, interval)
    d = int(e_row - s_row)
    assert d >= 3
    content_left = band_w + 2
    content_right = W - band_w - 3
    mid = W // 2

    for _ in range(200):
        upper_size = random.choice([2, 3])
        lower_size = random.choice([2, 3])
        upper_r0 = s_row - upper_size + 1
        lower_r0 = e_row
        lower_r1 = lower_r0 + lower_size - 1
        if upper_r0 < 1 or lower_r1 > H - 2:
            continue

        if slope == 1:
            c_min = content_left
            c_max = min(mid - 3, content_right - (upper_size + lower_size + d - 2))
            if c_min > c_max:
                continue
            upper_c0 = random.randint(c_min, c_max)
            lower_c0 = upper_c0 + upper_size - 1 + d
            lower_c1 = lower_c0 + lower_size - 1
            if lower_c1 > content_right:
                continue
            start = (s_row, upper_c0 + upper_size - 1)
            end = (e_row, lower_c0)
        else:
            c_min = max(content_left + d + lower_size - 1, mid + 1)
            c_max = content_right - (upper_size - 1)
            if c_min > c_max:
                continue
            upper_c0 = random.randint(c_min, c_max)
            lower_c1 = upper_c0 - d
            lower_c0 = lower_c1 - lower_size + 1
            if lower_c0 < content_left:
                continue
            start = (s_row, upper_c0)
            end = (e_row, lower_c1)

        upper_cells = _square_cells(upper_r0, upper_c0, upper_size)
        lower_cells = _square_cells(lower_r0, lower_c0, lower_size)
        line_cells: list[tuple[int, int]] = []
        for step in range(1, d):
            line_cells.append((s_row + step, start[1] + step * slope))
        all_cells = set(upper_cells) | set(lower_cells) | set(line_cells)
        if any(c < band_w or c >= W - band_w for _, c in all_cells):
            continue
        if _expanded(all_cells, (H, W)) & blocked:
            continue
        return {
            "color": int(color),
            "slope": int(slope),
            "upper": (upper_r0, upper_c0, upper_r0 + upper_size - 1, upper_c0 + upper_size - 1),
            "lower": (lower_r0, lower_c0, lower_r0 + lower_size - 1, lower_c0 + lower_size - 1),
            "start": start,
            "end": end,
            "rows": set(range(s_row, e_row + 1)),
            "line_cells": tuple(line_cells),
            "cells": all_cells,
        }
    return None


def _interval_overlap(a: tuple[int, int], b: tuple[int, int]) -> int:
    return max(0, min(a[1], b[1]) - max(a[0], b[0]) + 1)


def _choose_second_interval(
    H: int,
    first: tuple[int, int],
    require_conflict: bool,
) -> tuple[int, int] | None:
    min_start = 2
    max_end = H - 3
    for _ in range(200):
        length = random.randint(4, 7)
        if require_conflict:
            start = random.randint(max(min_start, first[0] - 2), max(min_start, first[0] - 1))
            end_lo = max(start + 3, first[0] + 1)
            end_hi = min(first[1] - 1, max_end)
            if end_lo > end_hi:
                continue
            end = random.randint(end_lo, end_hi)
        else:
            side = random.choice(["above", "below"])
            if side == "above":
                end = random.randint(min_start + 3, max(first[0] - 2, min_start + 3))
                start = end - (length - 1)
            else:
                start = random.randint(min(first[1] + 2, max_end - 3), max_end - 3)
                end = start + (length - 1)
        if start < min_start or end > max_end:
            continue
        overlap = _interval_overlap(first, (start, end))
        if require_conflict:
            if overlap == 0 or overlap >= (first[1] - first[0] + 1) or overlap >= (end - start + 1):
                continue
        else:
            if overlap != 0:
                continue
        return int(start), int(end)
    return None


def _choose_third_interval(H: int, existing: list[tuple[int, int]]) -> tuple[int, int] | None:
    min_start = 2
    max_end = H - 3
    for _ in range(300):
        length = random.randint(4, 7)
        start = random.randint(min_start, max_end - (length - 1))
        end = start + length - 1
        interval = (int(start), int(end))
        if all(_interval_overlap(interval, other) <= 1 for other in existing):
            return interval
    return None


def _random_band(height: int, width: int) -> np.ndarray:
    patterns = [(YELLOW,), (ORANGE,)] if width == 1 else [
        (YELLOW, YELLOW),
        (YELLOW, ORANGE),
        (ORANGE, YELLOW),
        (ORANGE, ORANGE),
    ]
    out = np.empty((height, width), dtype=int)
    prev = None
    for r in range(height):
        choices = patterns[:]
        if prev in choices and len(choices) > 1 and random.random() < 0.7:
            choices.remove(prev)
        pat = random.choice(choices)
        out[r] = pat
        prev = pat
    if not np.any(out == YELLOW):
        out[random.randrange(height), random.randrange(width)] = YELLOW
    if not np.any(out == ORANGE):
        out[random.randrange(height), random.randrange(width)] = ORANGE
    return out


def _render_from_meta(grid: np.ndarray, pairs: list[dict]) -> np.ndarray:
    out = np.array(grid, dtype=int, copy=True)
    row_dirs: dict[int, set[int]] = defaultdict(set)
    for pair in pairs:
        for r, c in pair["line_cells"]:
            out[int(r), int(c)] = int(pair["color"])
        for row in pair["rows"]:
            row_dirs[int(row)].add(int(pair["slope"]))

    band_w = _detect_band_width(grid, "left")
    left_slice = grid[:, :band_w].copy()
    right_slice = grid[:, grid.shape[1] - band_w :].copy()
    for row, dirs in row_dirs.items():
        if dirs == {1}:
            out[row, grid.shape[1] - band_w :] = left_slice[row]
            out[row, :band_w] = BG
        elif dirs == {-1}:
            out[row, :band_w] = right_slice[row]
            out[row, grid.shape[1] - band_w :] = BG
    return out


def _validate_generated(grid: np.ndarray, pairs: list[dict], require_conflict: bool) -> bool:
    if grid.ndim != 2 or max(grid.shape) > 30:
        return False
    slopes = {int(pair["slope"]) for pair in pairs}
    if slopes != {1, -1}:
        return False

    row_dirs: dict[int, set[int]] = defaultdict(set)
    for pair in pairs:
        for row in pair["rows"]:
            row_dirs[int(row)].add(int(pair["slope"]))

    exclusive_pos = [row for row, dirs in row_dirs.items() if dirs == {1}]
    exclusive_neg = [row for row, dirs in row_dirs.items() if dirs == {-1}]
    conflict_rows = [row for row, dirs in row_dirs.items() if dirs == {1, -1}]
    inactive_rows = [row for row in range(grid.shape[0]) if row not in row_dirs]

    if not exclusive_pos or not exclusive_neg or not inactive_rows:
        return False
    if require_conflict and not conflict_rows:
        return False

    band_w = _detect_band_width(grid, "left")
    if band_w != _detect_band_width(grid, "right"):
        return False
    if band_w not in {1, 2}:
        return False

    left = grid[:, :band_w].copy()
    right = grid[:, grid.shape[1] - band_w :].copy()
    for row in exclusive_pos:
        if np.array_equal(left[row], right[row]):
            return False
    for row in exclusive_neg:
        if np.array_equal(left[row], right[row]):
            return False

    expected = _render_from_meta(grid, pairs)
    actual = main(grid)
    if not np.array_equal(expected, actual):
        return False

    solved, parsed = _solve(grid)
    if not np.array_equal(actual, solved):
        return False
    parsed_dirs = {row: frozenset(dirs) for row, dirs in parsed["row_dirs"].items()}
    meta_dirs = {row: frozenset(dirs) for row, dirs in row_dirs.items()}
    return parsed_dirs == meta_dirs


def _gen_attempt() -> np.ndarray | None:
    H = random.randint(17, 24)
    W = random.randint(18, 26)
    band_w = random.choices([1, 2], weights=[2, 3])[0]
    n_pairs = random.choices([2, 3], weights=[3, 2])[0]
    require_conflict = random.random() < 0.6

    first_interval = (
        random.randint(3, max(3, H // 2 - 1)),
        random.randint(max(7, H // 2 + 1), H - 3),
    )
    if first_interval[1] - first_interval[0] < 3:
        return None

    second_interval = _choose_second_interval(H, first_interval, require_conflict)
    if second_interval is None:
        return None

    intervals = [first_interval, second_interval]
    slopes = [1, -1]
    if n_pairs == 3:
        third_interval = _choose_third_interval(H, intervals)
        if third_interval is None:
            return None
        intervals.append(third_interval)
        slopes.append(random.choice([1, -1]))

    colors = random.sample(PAIR_COLORS, n_pairs)
    grid = np.zeros((H, W), dtype=int)
    left_band = _random_band(H, band_w)
    right_band = _random_band(H, band_w)
    grid[:, :band_w] = left_band
    grid[:, W - band_w :] = right_band

    pairs: list[dict] = []
    blocked: set[tuple[int, int]] = set()
    for interval, slope, color in zip(intervals, slopes, colors):
        pair = _build_pair(H, W, band_w, int(slope), interval, int(color), blocked)
        if pair is None:
            return None
        pairs.append(pair)
        blocked |= _expanded(pair["cells"], (H, W))
        ur0, uc0, ur1, uc1 = pair["upper"]
        lr0, lc0, lr1, lc1 = pair["lower"]
        grid[ur0 : ur1 + 1, uc0 : uc1 + 1] = int(color)
        grid[lr0 : lr1 + 1, lc0 : lc1 + 1] = int(color)

    row_dirs: dict[int, set[int]] = defaultdict(set)
    for pair in pairs:
        for row in pair["rows"]:
            row_dirs[int(row)].add(int(pair["slope"]))
    for row, dirs in row_dirs.items():
        if dirs in ({1}, {-1}) and np.array_equal(grid[row, :band_w], grid[row, W - band_w :]):
            if dirs == {1}:
                grid[row, W - band_w :] = np.where(grid[row, :band_w] == YELLOW, ORANGE, YELLOW)
            else:
                grid[row, :band_w] = np.where(grid[row, W - band_w :] == YELLOW, ORANGE, YELLOW)

    if not _validate_generated(grid, pairs, require_conflict):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(500):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 3")
