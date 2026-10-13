from __future__ import annotations

import random
from collections import defaultdict

import numpy as np


BG = 0
ANCHOR = 9
PATH_COLORS = (1, 2, 3, 4, 5, 6, 7, 8)
COUNT_OPTIONS = ((1, 2), (2, 1), (2, 2), (1, 3), (3, 1))
GRID_H_RANGE = (12, 18)
GRID_W_RANGE = (12, 18)
MIN_SPACING = 3
MAX_RETRIES = 400


def _bg_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _perimeter_coords(h: int, w: int) -> list[tuple[int, int]]:
    if h == 1:
        return [(0, c) for c in range(w)]
    if w == 1:
        return [(r, 0) for r in range(h)]

    coords: list[tuple[int, int]] = []
    coords.extend((0, c) for c in range(w))
    coords.extend((r, w - 1) for r in range(1, h - 1))
    coords.extend((h - 1, c) for c in range(w - 1, -1, -1))
    coords.extend((r, 0) for r in range(h - 2, 0, -1))
    return coords


def _scan_priority(grid: np.ndarray, bg: int) -> list[int]:
    order: list[int] = []
    seen: set[int] = set()
    for r, c in _perimeter_coords(*grid.shape):
        value = int(grid[r, c])
        if value in (bg, ANCHOR) or value in seen:
            continue
        seen.add(value)
        order.append(value)
    return order


def _row_major_colors(grid: np.ndarray, bg: int) -> list[int]:
    order: list[int] = []
    seen: set[int] = set()
    h, w = grid.shape
    for r in range(h):
        for c in range(w):
            value = int(grid[r, c])
            if value in (bg, ANCHOR) or value in seen:
                continue
            seen.add(value)
            order.append(value)
    return order


def _runs(indices: list[int]) -> list[tuple[int, int]]:
    if not indices:
        return []

    result: list[tuple[int, int]] = []
    start = prev = indices[0]
    for value in indices[1:]:
        if value == prev + 1:
            prev = value
            continue
        result.append((start, prev))
        start = prev = value
    result.append((start, prev))
    return result


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete anchored gaps in monochrome orthogonal paths, then resolve overlaps
    using the clockwise order of border markers as a global priority list.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = _bg_color(g)

    priority = _scan_priority(g, bg)
    color_order = priority[:]
    for color in _row_major_colors(g, bg):
        if color not in color_order:
            color_order.append(color)
    rank = {color: idx for idx, color in enumerate(color_order)}

    interior = g.copy()
    interior[0, :] = bg
    interior[h - 1, :] = bg
    interior[:, 0] = bg
    interior[:, w - 1] = bg

    claims: dict[tuple[int, int], set[int]] = defaultdict(set)
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            value = int(interior[r, c])
            if value not in (bg, ANCHOR):
                claims[(r, c)].add(value)

    for color in color_order:
        mask = interior == color

        for r in range(1, h - 1):
            positions = [int(c) for c in np.flatnonzero(mask[r])]
            for (_, left_end), (right_start, _) in zip(_runs(positions), _runs(positions)[1:]):
                if right_start <= left_end + 1:
                    continue
                gap = interior[r, left_end + 1 : right_start]
                if np.any(gap == ANCHOR):
                    for c in range(left_end + 1, right_start):
                        claims[(r, c)].add(color)

        for c in range(1, w - 1):
            positions = [int(r) for r in np.flatnonzero(mask[:, c])]
            for (_, top_end), (bottom_start, _) in zip(_runs(positions), _runs(positions)[1:]):
                if bottom_start <= top_end + 1:
                    continue
                gap = interior[top_end + 1 : bottom_start, c]
                if np.any(gap == ANCHOR):
                    for r in range(top_end + 1, bottom_start):
                        claims[(r, c)].add(color)

    out = np.full_like(g, bg)
    for (r, c), colors in claims.items():
        winner = min(colors, key=lambda color: rank.get(color, len(rank)))
        out[r, c] = winner
    return out


def _pick_spaced_positions(low: int, high: int, count: int, gap: int) -> list[int] | None:
    values = list(range(low, high + 1))
    if count <= 0 or len(values) < count:
        return None

    for _ in range(200):
        picks = sorted(random.sample(values, count))
        if all(b - a >= gap for a, b in zip(picks, picks[1:])):
            return picks
    return None


def _merge_intervals(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    if not intervals:
        return []
    merged = [intervals[0]]
    for start, end in sorted(intervals)[1:]:
        prev_start, prev_end = merged[-1]
        if start <= prev_end + 1:
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))
    return merged


def _build_anchored_intervals(
    start: int,
    end: int,
    all_crosses: list[int],
    owned_crosses: list[int],
) -> list[tuple[int, int]] | None:
    if not owned_crosses:
        return []

    index_of = {value: idx for idx, value in enumerate(all_crosses)}
    raw: list[tuple[int, int]] = []
    for cross in sorted(owned_crosses):
        idx = index_of[cross]
        left_limit = start + 1 if idx == 0 else all_crosses[idx - 1] + 1
        right_limit = end - 1 if idx == len(all_crosses) - 1 else all_crosses[idx + 1] - 1
        max_left = min(2, cross - left_limit)
        max_right = min(2, right_limit - cross)
        if max_left <= 0 and max_right <= 0:
            return None

        left_ext = random.randint(0, max_left) if max_left > 0 else 0
        right_ext = random.randint(0, max_right) if max_right > 0 else 0
        if left_ext == 0 and right_ext == 0:
            if max_left > 0 and max_right > 0:
                if random.random() < 0.5:
                    left_ext = 1
                else:
                    right_ext = 1
            elif max_left > 0:
                left_ext = 1
            else:
                right_ext = 1
        raw.append((cross - left_ext, cross + right_ext))
    return _merge_intervals(raw)


def _positions_from_intervals(intervals: list[tuple[int, int]]) -> set[int]:
    positions: set[int] = set()
    for start, end in intervals:
        positions.update(range(start, end + 1))
    return positions


def _choose_unanchored_gap(path: dict) -> tuple[int, int] | None:
    start = path["start"]
    end = path["end"]
    crosses = set(path["crosses"])
    blocked: set[int] = set(crosses)
    for lo, hi in path["anchored"]:
        blocked.update(range(lo - 1, hi + 2))
    for lo, hi in path["unanchored"]:
        blocked.update(range(lo - 1, hi + 2))

    stretches: list[tuple[int, int]] = []
    cur_start: int | None = None
    for pos in range(start + 1, end):
        if pos in blocked:
            if cur_start is not None:
                stretches.append((cur_start, pos - 1))
                cur_start = None
            continue
        if cur_start is None:
            cur_start = pos
    if cur_start is not None:
        stretches.append((cur_start, end - 1))

    stretches = [stretch for stretch in stretches if stretch[0] <= stretch[1]]
    if not stretches:
        return None

    lo, hi = random.choice(stretches)
    max_len = min(3, hi - lo + 1)
    gap_len = random.randint(1, max_len)
    gap_start = random.randint(lo, hi - gap_len + 1)
    return (gap_start, gap_start + gap_len - 1)


def _path_coord(path: dict, axis_pos: int) -> tuple[int, int]:
    if path["orient"] == "H":
        return (path["fixed"], axis_pos)
    return (axis_pos, path["fixed"])


def _full_axis_positions(path: dict) -> range:
    return range(path["start"], path["end"] + 1)


def _gap_positions(path: dict, include_unanchored: bool) -> set[int]:
    positions = _positions_from_intervals(path["anchored"])
    if include_unanchored:
        positions |= _positions_from_intervals(path["unanchored"])
    return positions


def _sample_perimeter_indices(length: int, count: int) -> list[int] | None:
    if count > length:
        return None
    for _ in range(200):
        picks = sorted(random.sample(range(length), count))
        ok = True
        for i, a in enumerate(picks):
            b = picks[(i + 1) % len(picks)]
            dist = (b - a) % length
            if 0 < dist < 2:
                ok = False
                break
        if ok:
            return picks
    return None


def _render_input(
    h: int,
    w: int,
    paths: list[dict],
    priority: list[int],
) -> np.ndarray | None:
    grid = np.full((h, w), BG, dtype=int)

    for path in paths:
        hidden = _gap_positions(path, include_unanchored=True)
        for axis_pos in _full_axis_positions(path):
            if axis_pos in hidden:
                continue
            r, c = _path_coord(path, axis_pos)
            if grid[r, c] != BG:
                return None
            grid[r, c] = path["color"]

    for path in paths:
        for lo, hi in path["anchored"]:
            cells = []
            for axis_pos in range(lo, hi + 1):
                r, c = _path_coord(path, axis_pos)
                if grid[r, c] == BG:
                    cells.append((r, c))
            if not cells:
                return None
            ar, ac = random.choice(cells)
            grid[ar, ac] = ANCHOR

    perimeter = _perimeter_coords(h, w)
    pick_indices = _sample_perimeter_indices(len(perimeter), len(priority))
    if pick_indices is None:
        return None
    for idx, color in zip(pick_indices, priority):
        r, c = perimeter[idx]
        if grid[r, c] != BG:
            return None
        grid[r, c] = color

    return grid


def _expected_output(
    h: int,
    w: int,
    paths: list[dict],
    priority: list[int],
) -> np.ndarray:
    rank = {color: idx for idx, color in enumerate(priority)}
    claims: dict[tuple[int, int], set[int]] = defaultdict(set)

    for path in paths:
        blocked = _positions_from_intervals(path["unanchored"])
        for axis_pos in _full_axis_positions(path):
            if axis_pos in blocked:
                continue
            claims[_path_coord(path, axis_pos)].add(path["color"])

    out = np.full((h, w), BG, dtype=int)
    for (r, c), colors in claims.items():
        out[r, c] = min(colors, key=lambda color: rank[color])
    return out


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    n_h, n_v = random.choice(COUNT_OPTIONS)

    rows = _pick_spaced_positions(2, h - 3, n_h, MIN_SPACING)
    cols = _pick_spaced_positions(2, w - 3, n_v, MIN_SPACING)
    if rows is None or cols is None:
        return None

    colors = random.sample(PATH_COLORS, n_h + n_v)
    priority = random.sample(colors, len(colors))
    rank = {color: idx for idx, color in enumerate(priority)}

    paths: list[dict] = []
    h_paths: list[dict] = []
    v_paths: list[dict] = []

    for i, row in enumerate(rows):
        start = random.randint(1, cols[0] - 1)
        end = random.randint(cols[-1] + 1, w - 2)
        path = {
            "color": colors[i],
            "orient": "H",
            "fixed": row,
            "start": start,
            "end": end,
            "crosses": cols[:],
            "owned": [],
            "anchored": [],
            "unanchored": [],
        }
        paths.append(path)
        h_paths.append(path)

    for j, col in enumerate(cols):
        start = random.randint(1, rows[0] - 1)
        end = random.randint(rows[-1] + 1, h - 2)
        path = {
            "color": colors[n_h + j],
            "orient": "V",
            "fixed": col,
            "start": start,
            "end": end,
            "crosses": rows[:],
            "owned": [],
            "anchored": [],
            "unanchored": [],
        }
        paths.append(path)
        v_paths.append(path)

    crossings = [(hi, vi) for hi in range(n_h) for vi in range(n_v)]
    overwrite_cross, keep_cross = random.sample(crossings, 2)
    assignments: dict[tuple[int, int], str] = {}

    for hi, vi in crossings:
        h_path = h_paths[hi]
        v_path = v_paths[vi]
        if (hi, vi) == overwrite_cross:
            assignments[(hi, vi)] = "H" if rank[h_path["color"]] < rank[v_path["color"]] else "V"
        elif (hi, vi) == keep_cross:
            assignments[(hi, vi)] = "H" if rank[h_path["color"]] > rank[v_path["color"]] else "V"
        else:
            assignments[(hi, vi)] = random.choice(("H", "V"))

    for hi, vi in crossings:
        if assignments[(hi, vi)] == "H":
            h_paths[hi]["owned"].append(v_paths[vi]["fixed"])
        else:
            v_paths[vi]["owned"].append(h_paths[hi]["fixed"])

    for path in paths:
        anchored = _build_anchored_intervals(
            path["start"],
            path["end"],
            sorted(path["crosses"]),
            sorted(path["owned"]),
        )
        if anchored is None:
            return None
        path["anchored"] = anchored

    must_unanchored = random.choice(paths)
    for path in random.sample(paths, len(paths)):
        need_gap = (path is must_unanchored) or (not path["anchored"])
        if need_gap:
            gap = _choose_unanchored_gap(path)
            if gap is None:
                return None
            path["unanchored"].append(gap)

    if sum(len(path["unanchored"]) for path in paths) < 1:
        return None
    if not any(path["anchored"] for path in paths):
        return None
    if not all(path["anchored"] or path["unanchored"] for path in paths):
        return None

    grid = _render_input(h, w, paths, priority)
    if grid is None:
        return None

    expected = _expected_output(h, w, paths, priority)
    actual = main(grid)
    if not np.array_equal(actual, expected):
        return None

    overwrite_seen = False
    keep_seen = False
    for hi, vi in crossings:
        hr = h_paths[hi]["fixed"]
        vc = v_paths[vi]["fixed"]
        h_color = h_paths[hi]["color"]
        v_color = v_paths[vi]["color"]
        winner = expected[hr, vc]
        visible = grid[hr, vc]
        owner = assignments[(hi, vi)]
        owner_color = h_color if owner == "H" else v_color
        other_color = v_color if owner == "H" else h_color
        if owner_color == winner and visible == other_color:
            overwrite_seen = True
        if winner == visible and winner == other_color:
            keep_seen = True
    if not overwrite_seen or not keep_seen:
        return None

    anchored_fill_seen = np.any((grid == BG) & (expected != BG)) or np.any((grid == ANCHOR) & (expected != BG))
    unanchored_bg_seen = False
    for path in paths:
        for lo, hi in path["unanchored"]:
            for axis_pos in range(lo, hi + 1):
                r, c = _path_coord(path, axis_pos)
                if expected[r, c] == BG:
                    unanchored_bg_seen = True
                    break
            if unanchored_bg_seen:
                break
        if unanchored_bg_seen:
            break
    if not anchored_fill_seen or not unanchored_bg_seen:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
