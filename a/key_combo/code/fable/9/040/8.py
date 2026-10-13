from __future__ import annotations

import random
from collections import Counter

import numpy as np


BG = 0
OBSTACLE = 5
GROUP_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
MAX_RETRIES = 400


def _get_bg(grid: np.ndarray) -> int:
    return Counter(map(int, grid.ravel())).most_common(1)[0][0]


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _extract_groups(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    assert bg == BG, "expected black background"

    groups: list[dict] = []
    for color in sorted(int(c) for c in np.unique(g) if int(c) not in (bg, OBSTACLE)):
        coords = [tuple(map(int, rc)) for rc in np.argwhere(g == color)]
        assert len(coords) == 4, "each group must have exactly four input markers"
        rows = [r for r, _ in coords]
        cols = [c for _, c in coords]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        assert r1 > r0 and c1 > c0 and r1 - r0 == c1 - c0, "group bbox must be a square"

        corners = {(r0, c0), (r0, c1), (r1, c0), (r1, c1)}
        corner_markers = corners & set(coords)
        inner_markers = [rc for rc in coords if rc not in corners]
        missing = list(corners - set(coords))
        assert len(corner_markers) == 3, "expected three corner markers"
        assert len(inner_markers) == 1, "expected one inner marker"
        assert len(missing) == 1, "expected one missing corner"

        inner_r, inner_c = inner_markers[0]
        assert r0 < inner_r < r1 and c0 < inner_c < c1, "inner marker must lie strictly inside the square"
        side_cmp = 2 * inner_c - (c0 + c1)
        assert side_cmp != 0, "inner marker may not sit on the vertical center line"
        direction = -1 if side_cmp < 0 else 1

        groups.append(
            {
                "color": color,
                "bbox": (r0, c0, r1, c1),
                "missing": missing[0],
                "inner": inner_markers[0],
                "dir": direction,
            }
        )

    assert groups, "expected at least one colored marker group"
    return groups


def _trace_path(grid: np.ndarray, group: dict, override_dir: int | None = None) -> dict:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    color = int(group["color"])
    direction = int(group["dir"] if override_dir is None else override_dir)
    r, c = map(int, group["missing"])

    path = [(r, c)]
    turns = 0
    while True:
        if r == 0:
            return {"path": path, "turns": turns, "status": "top"}

        above = int(g[r - 1, c])
        if above == BG or above == color:
            r -= 1
            path.append((r, c))
            continue

        if above == OBSTACLE:
            nc = c + direction
            if not _in_bounds((h, w), r, nc) or int(g[r, nc]) != BG or (r, nc) in path:
                return {"path": path, "turns": turns, "status": "blocked"}
            c = nc
            path.append((r, c))
            turns += 1
            continue

        return {"path": path, "turns": turns, "status": "blocked"}


def _all_paths(grid: np.ndarray, groups: list[dict]) -> list[dict]:
    traces = [_trace_path(grid, group) | {"group": group} for group in groups]
    used: set[tuple[int, int]] = set()
    for trace in traces:
        for cell in trace["path"]:
            assert cell not in used, "paths must be non-overlapping"
            used.add(cell)
    return traces


def main(grid: np.ndarray) -> np.ndarray:
    """Grow same-color upward paths from each three-corner square's missing corner, with detours chosen by the inner marker's left/right side.

    Input structure:
    - The background is black, with scattered gray obstacle pixels.
    - Each non-gray foreground color appears as one group of four markers.
    - In each group, three markers occupy three corners of a square and the fourth
      lies strictly inside that square, away from the vertical center line.

    Rule:
    - Complete each group's square conceptually; the missing fourth corner is that
      group's path start, even though that cell is background in the input.
    - Paint the start cell and grow a path upward in the same color.
    - If the cell above is background, move up one cell.
    - If the cell above is gray, sidestep one cell left when the inner marker is
      in the square's left half, or right when it is in the right half, then keep
      trying to go upward from the new position.
    - Stop when the sideways target is unavailable or when the path reaches row 0.
      Existing markers and gray obstacles stay unchanged.

    Key insight:
    - The omitted corner is the real control pixel, while the seemingly stray inner
      marker is the per-group switch that decides whether obstacle detours go left
      or right.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"
    groups = _extract_groups(g)
    traces = _all_paths(g, groups)

    out = g.copy()
    for trace in traces:
        color = int(trace["group"]["color"])
        for r, c in trace["path"]:
            out[r, c] = color
    return out


def _split_turns(total: int) -> list[int]:
    if total <= 1:
        return [total]
    if total == 2:
        return random.choice([[2], [1, 1]])
    return random.choice([[3], [1, 2], [2, 1], [1, 1, 1]])


def _choose_layout(group_count: int) -> tuple[int, list[tuple[int, int]]]:
    for _ in range(200):
        if group_count == 2:
            widths = [random.randint(6, 9) for _ in range(group_count)]
            gaps = [random.randint(1, 2) for _ in range(group_count + 1)]
        elif group_count == 3:
            widths = [random.randint(5, 7) for _ in range(group_count)]
            gaps = [random.randint(1, 2) for _ in range(group_count + 1)]
        else:
            widths = [random.randint(5, 6) for _ in range(group_count)]
            gaps = [1] * (group_count + 1)
        width = sum(widths) + sum(gaps)
        if width > 30:
            continue

        bands = []
        c = gaps[0]
        for i, band_w in enumerate(widths):
            bands.append((c, c + band_w - 1))
            c += band_w + gaps[i + 1]
        return width, bands
    raise RuntimeError("failed to choose a valid band layout")


def _pick_inner_marker(r0: int, c0: int, side: int, direction: int) -> tuple[int, int] | None:
    r1, c1 = r0 + side, c0 + side
    inner_rows = list(range(r0 + 1, r1))
    if direction < 0:
        inner_cols = [c for c in range(c0 + 1, c1) if 2 * c < c0 + c1]
    else:
        inner_cols = [c for c in range(c0 + 1, c1) if 2 * c > c0 + c1]
    if not inner_rows or not inner_cols:
        return None
    return random.choice(inner_rows), random.choice(inner_cols)


def _render_group_markers(group: dict) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = group["bbox"]
    missing = tuple(group["missing"])
    markers = {(r0, c0), (r0, c1), (r1, c0), (r1, c1)}
    markers.remove(missing)
    markers.add(tuple(group["inner"]))
    return markers


def _plan_group(
    shape: tuple[int, int],
    band: tuple[int, int],
    color: int,
    direction: int,
    must_block: bool,
    used_input: set[tuple[int, int]],
    used_paths: set[tuple[int, int]],
    frame_boxes: list[tuple[int, int, int, int]],
) -> dict | None:
    h, w = shape
    left, right = band
    band_width = right - left + 1

    for _ in range(300):
        side_hi = 4 if band_width >= 6 else 3
        side = random.randint(3, side_hi)
        max_shift = min(3, band_width - side)
        if max_shift < 1:
            continue
        total_shift = random.randint(1, max_shift)
        turn_runs = _split_turns(total_shift)

        r0_min = max(5, len(turn_runs) + 1)
        r0_max = h - side - 2
        if r0_min > r0_max:
            continue
        r0 = random.randint(r0_min, r0_max)
        successful_ups = random.randint(max(2, len(turn_runs)), r0 - 1) if must_block else r0

        if len(turn_runs) > successful_ups:
            continue

        step_ids = sorted(random.sample(range(successful_ups), k=len(turn_runs)))
        turn_map = dict(zip(step_ids, turn_runs))

        if direction < 0:
            start_lo = left + total_shift
            start_hi = right - side
            if start_lo > start_hi:
                continue
            start_c = random.randint(start_lo, start_hi)
            c0 = start_c
            missing = (r0, c0)
        else:
            start_lo = left + side
            start_hi = right - total_shift
            if start_lo > start_hi:
                continue
            start_c = random.randint(start_lo, start_hi)
            c0 = start_c - side
            missing = (r0, start_c)

        bbox = (r0, c0, r0 + side, c0 + side)
        inner = _pick_inner_marker(r0, c0, side, direction)
        if inner is None:
            continue

        if any(not (bbox[2] < b[0] or b[2] < bbox[0] or bbox[3] < b[1] or b[3] < bbox[1]) for b in frame_boxes):
            continue

        group = {"color": color, "bbox": bbox, "missing": missing, "inner": inner, "dir": direction}
        markers = _render_group_markers(group)
        if any(cell in used_input or cell in used_paths for cell in markers):
            continue

        local_input = set(markers)
        local_path = {missing}
        local_obstacles: set[tuple[int, int]] = set()
        r, c = missing
        path = [missing]
        ok = True

        for step in range(successful_ups):
            run = turn_map.get(step, 0)
            for _ in range(run):
                obstacle = (r - 1, c)
                side_cell = (r, c + direction)
                if not _in_bounds(shape, *obstacle) or not _in_bounds(shape, *side_cell):
                    ok = False
                    break
                if obstacle in used_input or obstacle in used_paths or obstacle in local_input or obstacle in local_path:
                    ok = False
                    break
                if side_cell in used_input or side_cell in used_paths or side_cell in local_input or side_cell in local_obstacles:
                    ok = False
                    break
                local_obstacles.add(obstacle)
                c += direction
                local_path.add((r, c))
                path.append((r, c))
            if not ok:
                break

            up_cell = (r - 1, c)
            if not _in_bounds(shape, *up_cell):
                ok = False
                break
            if up_cell in used_input or up_cell in used_paths or up_cell in local_input or up_cell in local_obstacles:
                ok = False
                break
            r -= 1
            local_path.add((r, c))
            path.append((r, c))

        if not ok:
            continue

        if must_block:
            above = (r - 1, c)
            if not _in_bounds(shape, *above):
                continue
            if above in used_input or above in used_paths or above in local_input or above in local_path:
                continue
            local_obstacles.add(above)

            side_block = (r, c + direction)
            if _in_bounds(shape, *side_block):
                if side_block in used_input or side_block in used_paths or side_block in local_input or side_block in local_path:
                    continue
                local_obstacles.add(side_block)

        group["planned_path"] = path
        group["planned_obstacles"] = sorted(local_obstacles)
        return group

    return None


def _safe_extra_cells(
    shape: tuple[int, int],
    groups: list[dict],
    occupied: set[tuple[int, int]],
) -> list[tuple[int, int]]:
    h, w = shape
    unsafe = set(occupied)
    for group in groups:
        r0, c0, r1, c1 = group["bbox"]
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                unsafe.add((r, c))
        for r, c in group["planned_path"]:
            unsafe.add((r, c))
            if r > 0:
                unsafe.add((r - 1, c))
            if c > 0:
                unsafe.add((r, c - 1))
            if c + 1 < w:
                unsafe.add((r, c + 1))

    cells = [(r, c) for r in range(h) for c in range(w) if (r, c) not in unsafe]
    random.shuffle(cells)
    return cells


def _validate_candidate(grid: np.ndarray) -> bool:
    try:
        groups = _extract_groups(grid)
        traces = _all_paths(grid, groups)
    except Exception:
        return False

    if len(groups) < 2:
        return False
    dirs = {int(group["dir"]) for group in groups}
    if dirs != {-1, 1}:
        return False

    statuses = Counter(trace["status"] for trace in traces)
    if statuses["top"] == 0 or statuses["blocked"] == 0:
        return False
    if any(trace["turns"] == 0 for trace in traces):
        return False

    boxes = [group["bbox"] for group in groups]
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            a = boxes[i]
            b = boxes[j]
            if not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1]):
                return False

    for trace in traces:
        group = trace["group"]
        flipped = _trace_path(grid, group, override_dir=-int(group["dir"]))
        if flipped["path"] == trace["path"] and flipped["status"] == trace["status"]:
            return False

    out = main(grid)
    if np.array_equal(out, grid):
        return False
    return True


def _attempt_generate() -> np.ndarray | None:
    group_count = random.randint(2, 4)
    width, bands = _choose_layout(group_count)
    height = random.randint(14, 22)
    shape = (height, width)

    colors = random.sample(GROUP_COLORS, group_count)
    directions = [-1, 1]
    while len(directions) < group_count:
        directions.append(random.choice([-1, 1]))
    random.shuffle(directions)
    blocked_idx = random.randrange(group_count)

    groups: list[dict] = []
    used_input: set[tuple[int, int]] = set()
    used_paths: set[tuple[int, int]] = set()
    frame_boxes: list[tuple[int, int, int, int]] = []

    order = list(range(group_count))
    random.shuffle(order)
    for i in order:
        group = _plan_group(
            shape=shape,
            band=bands[i],
            color=colors[i],
            direction=directions[i],
            must_block=(i == blocked_idx),
            used_input=used_input,
            used_paths=used_paths,
            frame_boxes=frame_boxes,
        )
        if group is None:
            return None
        groups.append(group)
        used_input.update(_render_group_markers(group))
        used_input.update(group["planned_obstacles"])
        used_paths.update(group["planned_path"])
        frame_boxes.append(group["bbox"])

    grid = np.zeros(shape, dtype=int)
    for group in groups:
        for r, c in group["planned_obstacles"]:
            grid[r, c] = OBSTACLE
        for r, c in _render_group_markers(group):
            grid[r, c] = group["color"]

    extras = _safe_extra_cells(shape, groups, used_input)
    extra_count = min(len(extras), random.randint(4, 12))
    for r, c in extras[:extra_count]:
        grid[r, c] = OBSTACLE

    if not _validate_candidate(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid puzzle after {MAX_RETRIES} attempts")
