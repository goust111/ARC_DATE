from __future__ import annotations

import itertools
import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GRAY = 5
ORANGE = 7

COUNT_TO_COLOR = {1: BLUE, 2: GREEN, 3: ORANGE}
GEN_RETRIES = 500


def _find_marker(grid: np.ndarray) -> tuple[int, int]:
    cells = np.argwhere(grid == RED)
    assert cells.shape == (1, 2), "expected exactly one red edge marker"
    r, c = map(int, cells[0])
    assert r == 0, "the marker must sit on the top edge"
    return r, c


def _extract_walls(grid: np.ndarray) -> list[dict]:
    rows = sorted({int(r) for r in np.argwhere(grid == GRAY)[:, 0]})
    assert 2 <= len(rows) <= 3, "expected 2 or 3 horizontal wall rows"

    walls = []
    for row in rows:
        cols = np.flatnonzero(grid[row] == GRAY)
        assert cols.size > 0
        c0 = int(cols.min())
        c1 = int(cols.max())
        assert c1 - c0 + 1 >= 6, "wall span must be wide enough to matter"
        segment = grid[row, c0 : c1 + 1]
        assert np.all((segment == BLACK) | (segment == GRAY)), "wall rows may only contain wall cells and holes"
        holes = tuple(int(c0 + i) for i, value in enumerate(segment) if int(value) == BLACK)
        assert 1 <= len(holes) <= 3, "each wall must expose 1 to 3 holes"
        walls.append({"row": row, "c0": c0, "c1": c1, "holes": holes})
    return walls


def _find_frame(grid: np.ndarray) -> dict:
    coords = np.argwhere(grid == YELLOW)
    assert coords.size > 0, "expected one yellow frame"
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    r0 = int(r0)
    c0 = int(c0)
    r1 = int(r1)
    c1 = int(c1)
    assert r1 - r0 >= 3 and c1 - c0 >= 3, "frame must have a non-empty interior"
    assert np.all(grid[r0 : r1 + 1, c0] == YELLOW)
    assert np.all(grid[r0 : r1 + 1, c1] == YELLOW)
    assert np.all(grid[r1, c0 : c1 + 1] == YELLOW)
    top = grid[r0, c0 : c1 + 1]
    assert int(top[0]) == YELLOW and int(top[-1]) == YELLOW
    assert np.all((top == YELLOW) | (top == BLACK))
    top_holes = tuple(c0 + i for i, value in enumerate(top) if i not in (0, len(top) - 1) and int(value) == BLACK)
    assert 2 <= len(top_holes) <= 4, "frame top must expose 2 to 4 entry holes"
    assert np.all(grid[r0 + 1 : r1, c0 + 1 : c1] == BLACK), "frame interior must be empty in the input"
    return {
        "r0": r0,
        "c0": c0,
        "r1": r1,
        "c1": c1,
        "top_holes": top_holes,
        "inner_h": r1 - r0 - 1,
        "inner_w": c1 - c0 - 1,
    }


def _trace_branches(shape: tuple[int, int], start_c: int, walls: list[dict], frame: dict) -> tuple[set[tuple[int, int]], list[tuple[int, int]], list[dict]]:
    height, width = shape
    assert 0 <= start_c < width
    wall_by_row = {wall["row"]: wall for wall in walls}
    red_cells: set[tuple[int, int]] = set()
    arrivals_by_col: dict[int, int] = {}
    wall_stats = [
        {"row": wall["row"], "hit_columns": set(), "hit_counts": set(), "spawn_count": 0}
        for wall in walls
    ]
    wall_stats_by_row = {item["row"]: item for item in wall_stats}

    stack = [(0, start_c, 0)]
    seen_counts: dict[tuple[int, int], int] = {}
    while stack:
        row, col, count = stack.pop()
        key = (int(row), int(col))
        prev = seen_counts.get(key)
        assert prev in (None, count), "merged branch states would make counts ambiguous"
        if prev == count:
            continue
        seen_counts[key] = int(count)

        red_cells.add(key)
        cur_row = int(row)
        while True:
            next_row = cur_row + 1
            if next_row >= height:
                break

            if next_row == frame["r0"] and frame["c0"] <= col <= frame["c1"]:
                if col in frame["top_holes"]:
                    red_cells.add((next_row, col))
                    prev_count = arrivals_by_col.get(col)
                    assert prev_count in (None, count), "entry column must have a unique wall-count label"
                    arrivals_by_col[col] = int(count)
                break

            wall = wall_by_row.get(next_row)
            if wall is not None and wall["c0"] <= col <= wall["c1"] and col not in wall["holes"]:
                stat = wall_stats_by_row[next_row]
                stat["hit_columns"].add(int(col))
                stat["hit_counts"].add(int(count))
                next_count = count + 1
                for gap_col in wall["holes"]:
                    red_cells.add((next_row, gap_col))
                    prev_count = seen_counts.get((next_row, gap_col))
                    assert prev_count in (None, next_count), "wall-spawned branches must stay unambiguous"
                    stack.append((next_row, gap_col, next_count))
                stat["spawn_count"] += len(wall["holes"])
                break

            red_cells.add((next_row, col))
            cur_row = next_row

    arrivals = sorted((int(col), int(count)) for col, count in arrivals_by_col.items())
    return red_cells, arrivals, wall_stats


def _render_solution(grid: np.ndarray, red_cells: set[tuple[int, int]], arrivals: list[tuple[int, int]], frame: dict) -> np.ndarray:
    out = np.array(grid, copy=True)
    for r, c in red_cells:
        if int(out[r, c]) in (BLACK, RED):
            out[r, c] = RED

    stripe_count = len(arrivals)
    assert 2 <= stripe_count <= 4, "expected 2 to 4 arriving branches"
    assert frame["inner_w"] % stripe_count == 0, "frame interior must split evenly into stripes"
    stripe_w = frame["inner_w"] // stripe_count
    for idx, (_col, wall_count) in enumerate(arrivals):
        assert wall_count in COUNT_TO_COLOR, "wall-count color mapping is only defined for 1..3 walls"
        color = COUNT_TO_COLOR[wall_count]
        c_start = frame["c0"] + 1 + idx * stripe_w
        out[frame["r0"] + 1 : frame["r1"], c_start : c_start + stripe_w] = color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Trace branching rays through broken horizontal walls, then fill the bottom frame with count-coded entry-order stripes.

    Input structure:
    - The background is black.
    - A single red marker sits on the top edge.
    - Two or three one-row gray walls appear below it, each spanning a wide interval and containing 1 to 3 single-cell holes.
    - A hollow yellow rectangle sits below the last wall, and its top edge contains 2 to 4 holes.

    Rule:
    - Shoot the red marker straight downward, painting traversed black cells red.
    - When a ray hits a gray wall cell, stop that ray and start new downward rays from every hole of that same wall; each spawned branch increases its wall-cross count by one.
    - When a branch reaches a hole in the yellow frame's top edge, record its entry column and wall-cross count, paint the hole red, and stop the ray before drawing inside the frame.
    - Sort the arriving branches by entry column, divide the frame interior into equally wide vertical stripes, and color each stripe by that branch's wall-cross count: 1 -> blue, 2 -> green, 3 -> orange.

    Key insight:
    - The wall-branching process determines three things at once: which frame holes are reached, the left-to-right order of those arrivals, and the stripe colors through the number of wall hits each surviving branch accumulated.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    _marker_r, marker_c = _find_marker(g)
    walls = _extract_walls(g)
    frame = _find_frame(g)
    red_cells, arrivals, _wall_stats = _trace_branches(g.shape, marker_c, walls, frame)
    return _render_solution(g, red_cells, arrivals, frame)


def _rand_hole_subset(rng: random.Random, choices: list[int], min_count: int, max_count: int) -> tuple[int, ...]:
    max_count = min(max_count, len(choices))
    if min_count == max_count:
        count = min_count
    elif max_count >= 3 and rng.random() < 0.6:
        count = min(3, max_count)
    elif max_count >= 2 and rng.random() < 0.75:
        count = max(2, min_count)
    else:
        count = rng.randint(min_count, max_count)
    valid = [
        combo
        for combo in itertools.combinations(choices, count)
        if all(b - a >= 2 for a, b in zip(combo, combo[1:]))
    ]
    if not valid:
        valid = list(itertools.combinations(choices, count))
    assert valid
    return tuple(int(v) for v in rng.choice(valid))


def _choose_rows(rng: random.Random, wall_count: int) -> tuple[list[int], int, int]:
    wall_rows = [rng.randint(3, 4)]
    while len(wall_rows) < wall_count:
        wall_rows.append(wall_rows[-1] + rng.randint(3, 4))
    frame_top = wall_rows[-1] + rng.randint(4, 5)
    frame_h = rng.randint(5, 7)
    return wall_rows, frame_top, frame_h


def _sample_intervals_for_slice(
    cols: list[int],
    active: dict[int, int],
    left_idx: int,
    right_idx: int,
    width: int,
    grid_w: int,
) -> list[tuple[int, int]]:
    hit_cols = cols[left_idx : right_idx + 1]
    if len({active[c] for c in hit_cols}) != 1:
        return []
    left_neighbor = cols[left_idx - 1] if left_idx > 0 else None
    right_neighbor = cols[right_idx + 1] if right_idx + 1 < len(cols) else None
    c0_min = (left_neighbor + 1) if left_neighbor is not None else 1
    c0_max = hit_cols[0]
    out = []
    for c0 in range(c0_min, c0_max + 1):
        c1 = c0 + width - 1
        if c1 < hit_cols[-1]:
            continue
        if c1 > (right_neighbor - 1 if right_neighbor is not None else grid_w - 2):
            continue
        if not all(not (c0 <= col <= c1) or col in hit_cols for col in cols):
            continue
        out.append((int(c0), int(c1)))
    return out


def _step_active_through_wall(active: dict[int, int], wall: dict) -> dict[int, int] | None:
    outside = {col: count for col, count in active.items() if not (wall["c0"] <= col <= wall["c1"])}
    hit = {col: count for col, count in active.items() if wall["c0"] <= col <= wall["c1"] and col not in wall["holes"]}
    pass_holes = {col: count for col, count in active.items() if wall["c0"] <= col <= wall["c1"] and col in wall["holes"]}
    if not hit:
        return None
    if pass_holes:
        return None
    hit_counts = {int(v) for v in hit.values()}
    if len(hit_counts) != 1:
        return None
    next_count = hit_counts.pop() + 1
    new_active = dict(outside)
    for gap_col in wall["holes"]:
        if gap_col in new_active:
            return None
        new_active[int(gap_col)] = int(next_count)
    return new_active


def _build_next_wall(rng: random.Random, active: dict[int, int], grid_w: int) -> tuple[dict, dict[int, int]] | None:
    cols = sorted(active)
    width_options = list(range(max(10, grid_w - 8), grid_w - 2))
    rng.shuffle(width_options)

    slice_specs = [(i, j) for i in range(len(cols)) for j in range(i, len(cols))]
    rng.shuffle(slice_specs)

    for width in width_options:
        for i, j in slice_specs:
            intervals = _sample_intervals_for_slice(cols, active, i, j, width, grid_w)
            if not intervals:
                continue
            rng.shuffle(intervals)
            for c0, c1 in intervals[:6]:
                hit_cols = cols[i : j + 1]
                available = [c for c in range(c0, c1 + 1) if c not in hit_cols]
                if not available:
                    continue
                for _ in range(8):
                    min_holes = 2 if len(available) >= 2 else 1
                    holes = _rand_hole_subset(rng, available, min_holes, min(3, len(available)))
                    wall = {"c0": int(c0), "c1": int(c1), "holes": holes}
                    next_active = _step_active_through_wall(active, wall)
                    if next_active is None:
                        continue
                    if len(next_active) > 5:
                        continue
                    return wall, next_active
    return None


def _choose_rectangle(
    rng: random.Random,
    final_active: dict[int, int],
    grid_w: int,
    prefer_mixed: bool,
    prefer_non_monotone: bool,
) -> tuple[dict, list[tuple[int, int]]] | None:
    items = sorted((int(col), int(count)) for col, count in final_active.items())
    max_k = min(4, len(items))
    subset_specs_by_k: dict[int, list[list[tuple[int, int]]]] = {k: [] for k in range(2, max_k + 1)}
    for k in range(2, max_k + 1):
        for subset in itertools.combinations(items, k):
            counts = [count for _col, count in subset]
            if prefer_mixed and len(set(counts)) < 2:
                continue
            if prefer_non_monotone and k >= 3:
                if counts == sorted(counts) or counts == sorted(counts, reverse=True):
                    continue
            subset_specs_by_k[k].append(list(subset))

    candidate_ks = [k for k in range(max_k, 1, -1) if subset_specs_by_k.get(k)]
    if not candidate_ks:
        return None

    for k in candidate_ks:
        rng.shuffle(subset_specs_by_k[k])
    for subset in list(itertools.chain.from_iterable(subset_specs_by_k[k] for k in candidate_ks)):
        cols = [col for col, _count in subset]
        k = len(subset)
        span = cols[-1] - cols[0] + 1
        inner_min = max(span, 2 * k)
        inner_candidates = [w for w in range(inner_min, grid_w - 2) if w % k == 0]
        rng.shuffle(inner_candidates)
        for inner_w in inner_candidates:
            c0_lo = max(1, cols[-1] - inner_w)
            c0_hi = min(cols[0] - 1, grid_w - inner_w - 3)
            if c0_lo > c0_hi:
                continue
            c0 = rng.randint(c0_lo, c0_hi)
            c1 = c0 + inner_w + 1
            if c1 >= grid_w - 1:
                continue
            if any(col in (c0, c1) for col, _count in items):
                continue
            return {"c0": int(c0), "c1": int(c1), "top_holes": tuple(cols)}, subset
    return None


def _render_input(
    height: int,
    width: int,
    marker_c: int,
    walls: list[dict],
    frame: dict,
) -> np.ndarray:
    grid = np.zeros((height, width), dtype=int)
    grid[0, marker_c] = RED
    for wall in walls:
        row = wall["row"]
        grid[row, wall["c0"] : wall["c1"] + 1] = GRAY
        for gap_col in wall["holes"]:
            grid[row, gap_col] = BLACK

    r0, c0, r1, c1 = frame["r0"], frame["c0"], frame["r1"], frame["c1"]
    grid[r0, c0 : c1 + 1] = YELLOW
    grid[r1, c0 : c1 + 1] = YELLOW
    grid[r0 : r1 + 1, c0] = YELLOW
    grid[r0 : r1 + 1, c1] = YELLOW
    for col in frame["top_holes"]:
        grid[r0, col] = BLACK
    return grid


def _validate_generated_case(grid: np.ndarray, expected_arrivals: list[tuple[int, int]]) -> bool:
    try:
        _marker_r, marker_c = _find_marker(grid)
        walls = _extract_walls(grid)
        frame = _find_frame(grid)
        red_cells, arrivals, wall_stats = _trace_branches(grid.shape, marker_c, walls, frame)
        out = _render_solution(grid, red_cells, arrivals, frame)
    except AssertionError:
        return False

    if arrivals != expected_arrivals:
        return False
    if len(arrivals) < 2 or len(arrivals) > 4:
        return False
    if len({count for _col, count in arrivals}) < 2:
        return False
    if frame["inner_w"] % len(arrivals) != 0:
        return False
    for stat in wall_stats:
        if not stat["hit_columns"]:
            return False
        if len(stat["hit_counts"]) != 1:
            return False
    if np.count_nonzero(out == RED) <= 3:
        return False

    stripe_w = frame["inner_w"] // len(arrivals)
    expected_colors = [COUNT_TO_COLOR[count] for _col, count in arrivals]
    observed = []
    for idx in range(len(arrivals)):
        c_start = frame["c0"] + 1 + idx * stripe_w
        patch = out[frame["r0"] + 1 : frame["r1"], c_start : c_start + stripe_w]
        if len(set(map(int, patch.flatten()))) != 1:
            return False
        observed.append(int(patch[0, 0]))
    return observed == expected_colors


def _attempt_generate() -> np.ndarray | None:
    rng = random.Random()
    wall_count = 3 if rng.random() < 0.7 else 2
    wall_rows, frame_top, frame_h = _choose_rows(rng, wall_count)

    max_width = 27
    min_width = 18
    width = rng.randint(min_width, max_width)

    first_width = rng.randint(max(10, width - 8), width - 3)
    first_c0 = rng.randint(1, width - first_width - 1)
    first_c1 = first_c0 + first_width - 1
    marker_c = rng.randint(first_c0 + 1, first_c1 - 1)
    first_min_holes = 3 if first_c1 - first_c0 + 1 >= 5 else 2
    first_holes = _rand_hole_subset(
        rng,
        [c for c in range(first_c0, first_c1 + 1) if c != marker_c],
        first_min_holes,
        3,
    )

    walls = [{"row": wall_rows[0], "c0": first_c0, "c1": first_c1, "holes": first_holes}]
    active = {int(col): 1 for col in first_holes}

    for row in wall_rows[1:]:
        built = _build_next_wall(rng, active, width)
        if built is None:
            return None
        wall, active = built
        walls.append({"row": row, "c0": wall["c0"], "c1": wall["c1"], "holes": wall["holes"]})

    prefer_mixed = True
    prefer_non_monotone = rng.random() < 0.6
    rect_choice = _choose_rectangle(rng, active, width, prefer_mixed, prefer_non_monotone)
    if rect_choice is None:
        return None
    rect_spec, expected_arrivals = rect_choice

    height = frame_top + frame_h + rng.randint(1, 3)
    if height > 30:
        return None

    frame = {
        "r0": frame_top,
        "c0": rect_spec["c0"],
        "r1": frame_top + frame_h - 1,
        "c1": rect_spec["c1"],
        "top_holes": rect_spec["top_holes"],
    }
    if frame["r1"] >= height:
        return None

    grid = _render_input(height, width, marker_c, walls, frame)
    if not _validate_generated_case(grid, expected_arrivals):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid wall-ray stripe puzzle")
