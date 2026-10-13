from __future__ import annotations

import random
from collections import deque

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREY = 5
TEAL = 8

FRAME_MIN_INNER = 3
FRAME_MIN_OUTER = FRAME_MIN_INNER + 2
GEN_ATTEMPTS = 400


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _connected_components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _find_frames(grid: np.ndarray) -> list[dict[str, int]]:
    frames: list[dict[str, int]] = []
    for comp in _connected_components(grid == BLUE):
        r0, c0, r1, c1 = _bbox(comp)
        if r1 - r0 + 1 < FRAME_MIN_OUTER or c1 - c0 + 1 < FRAME_MIN_OUTER:
            continue

        missing: list[tuple[int, int]] = []
        valid = True
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                on_border = r in (r0, r1) or c in (c0, c1)
                if on_border:
                    if grid[r, c] != BLUE:
                        missing.append((r, c))
                elif grid[r, c] != BLACK:
                    valid = False
                    break
            if not valid:
                break
        if not valid or len(missing) != 1:
            continue

        gap_r, gap_c = missing[0]
        if gap_r != r1 or not (c0 < gap_c < c1):
            continue

        border_area = 2 * ((r1 - r0 + 1) + (c1 - c0 + 1)) - 4 - 1
        if len(comp) != border_area:
            continue

        frames.append(
            {
                "r0": r0,
                "c0": c0,
                "r1": r1,
                "c1": c1,
                "gap_r": gap_r,
                "gap_c": gap_c,
            }
        )
    frames.sort(key=lambda f: (f["r0"], f["c0"], f["r1"], f["c1"]))
    return frames


def _trace_single_path(grid: np.ndarray, start_c: int) -> dict[str, object]:
    h, w = grid.shape
    r = h - 1
    c = int(start_c)
    coords = [(r, c)]
    detours: list[dict[str, object]] = []

    while True:
        if r == 0:
            break
        above = int(grid[r - 1, c])
        if above in (BLUE, GREY):
            if c + 1 >= w or int(grid[r, c + 1]) in (BLUE, GREY):
                break
            detours.append({"from": (r, c), "to": (r, c + 1), "reason": above})
            c += 1
            coords.append((r, c))
        else:
            r -= 1
            coords.append((r, c))
    return {"start_c": int(start_c), "coords": coords, "detours": detours}


def _simulate_paths(grid: np.ndarray) -> tuple[np.ndarray, list[dict[str, object]]]:
    out = np.array(grid, dtype=int, copy=True)
    starts = [int(c) for c in np.where(out[-1] == RED)[0]]
    paths = [_trace_single_path(out, start_c) for start_c in starts]
    for path in paths:
        for r, c in path["coords"]:
            out[r, c] = RED
    return out, paths


def _analyze(grid: np.ndarray) -> dict[str, object]:
    frames = _find_frames(grid)
    path_grid, paths = _simulate_paths(grid)
    frame_infos: list[dict[str, object]] = []
    for frame in frames:
        gap = (frame["gap_r"], frame["gap_c"])
        blocked = int(path_grid[gap]) == RED
        stopper = "blocked"
        if not blocked:
            rr = frame["gap_r"] + 1
            while rr < grid.shape[0]:
                val = int(path_grid[rr, frame["gap_c"]])
                if val == BLUE:
                    stopper = "frame"
                    break
                if val == RED:
                    stopper = "path"
                    break
                rr += 1
            else:
                stopper = "bottom"
        hit_paths = [p for p in paths if gap in set(p["coords"])]
        frame_infos.append(
            {
                **frame,
                "blocked": blocked,
                "stopper": stopper,
                "gap": gap,
                "hit_paths": hit_paths,
            }
        )
    return {"frames": frame_infos, "paths": paths, "path_grid": path_grid}


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow red detouring paths, use any path-hit bottom gaps as plugs, then flood rectangles and spill only from unplugged gaps.

    Input structure:
    - The grid has a black background, 2-4 blue hollow rectangles in the upper region, and each rectangle has exactly one one-cell gap on its bottom side.
    - Several red start pixels sit on the bottom row, and gray obstacle pixels appear between the starts and the rectangles.
    - Rectangle interiors are black, blue border pixels are structural blockers, and different rectangles do not overlap or touch.

    Rule:
    - From each bottom-row red start, repeatedly try to move upward one cell; if the cell above is gray or blue, move right one cell instead, and stop if that right move is impossible.
    - Any rectangle whose bottom-gap cell becomes red is considered plugged; then fill every rectangle interior black cell with teal.
    - For each unplugged rectangle, recolor the gap cell teal and continue straight downward, painting only black cells teal until a blue border, a red path, or the grid bottom edge is reached.
    - Keep all original blue border pixels and gray obstacle pixels unchanged, and preserve all red path pixels.

    Key insight:
    - The red growth paths are not decoration: obstacle-driven right detours decide which rectangle gaps become sealed, and only the unsealed gaps are allowed to leak teal downward.
    """
    g = np.asarray(grid, dtype=int)
    out, _paths = _simulate_paths(g)
    frames = _find_frames(g)

    for frame in frames:
        r0, c0, r1, c1 = frame["r0"], frame["c0"], frame["r1"], frame["c1"]
        interior = out[r0 + 1 : r1, c0 + 1 : c1]
        interior[interior == BLACK] = TEAL

    for frame in frames:
        gap_r, gap_c = frame["gap_r"], frame["gap_c"]
        if out[gap_r, gap_c] == RED:
            continue
        out[gap_r, gap_c] = TEAL
        rr = gap_r + 1
        while rr < out.shape[0]:
            if out[rr, gap_c] in (BLUE, RED):
                break
            if out[rr, gap_c] == BLACK:
                out[rr, gap_c] = TEAL
            rr += 1

    return out


def _frame_bbox(frame: dict[str, int]) -> tuple[int, int, int, int]:
    return frame["r0"], frame["c0"], frame["r1"], frame["c1"]


def _boxes_clear(box: tuple[int, int, int, int], others: list[tuple[int, int, int, int]], pad: int = 1) -> bool:
    r0, c0, r1, c1 = box
    for or0, oc0, or1, oc1 in others:
        if not (r1 + pad < or0 or or1 + pad < r0 or c1 + pad < oc0 or oc1 + pad < c0):
            return False
    return True


def _draw_frame(grid: np.ndarray, frame: dict[str, int]) -> None:
    r0, c0, r1, c1 = frame["r0"], frame["c0"], frame["r1"], frame["c1"]
    grid[r0, c0 : c1 + 1] = BLUE
    grid[r1, c0 : c1 + 1] = BLUE
    grid[r0 : r1 + 1, c0] = BLUE
    grid[r0 : r1 + 1, c1] = BLUE
    grid[r1, frame["gap_c"]] = BLACK


def _gap_column(frame: dict[str, int]) -> int:
    return int(frame["gap_c"])


def _random_frame(
    h: int,
    w: int,
    others: list[tuple[int, int, int, int]],
    row_range: tuple[int, int],
    col_range: tuple[int, int],
) -> dict[str, int] | None:
    for _ in range(60):
        fh = random.randint(5, 7)
        fw = random.randint(5, 8)
        r0_lo, r0_hi = row_range
        c0_lo, c0_hi = col_range
        if r0_hi < r0_lo or c0_hi < c0_lo:
            return None
        r0_hi = min(r0_hi, h - fh - 2)
        c0_hi = min(c0_hi, w - fw - 1)
        if r0_hi < r0_lo or c0_hi < c0_lo:
            continue
        r0 = random.randint(r0_lo, r0_hi)
        c0 = random.randint(c0_lo, c0_hi)
        frame = {"r0": r0, "c0": c0, "r1": r0 + fh - 1, "c1": c0 + fw - 1}
        box = _frame_bbox(frame)
        if not _boxes_clear(box, others, pad=1):
            continue
        frame["gap_c"] = random.randint(frame["c0"] + 1, frame["c1"] - 1)
        return frame
    return None


def _cells_adjacent_to_boxes(shape: tuple[int, int], boxes: list[tuple[int, int, int, int]]) -> set[tuple[int, int]]:
    h, w = shape
    blocked: set[tuple[int, int]] = set()
    for r0, c0, r1, c1 in boxes:
        for r in range(r0 - 1, r1 + 2):
            for c in range(c0 - 1, c1 + 2):
                if 0 <= r < h and 0 <= c < w:
                    blocked.add((r, c))
    return blocked


def _plan_shift_rows(
    h: int,
    target_row: int,
    shifts: int,
    reserved_rows: set[int] | None = None,
) -> list[int] | None:
    reserved_rows = reserved_rows or set()
    candidates = [r for r in range(target_row + 3, h - 1) if r not in reserved_rows]
    if len(candidates) < shifts:
        return None
    return sorted(random.sample(candidates, shifts), reverse=True)


def _guided_obstacles_to_column(
    shape: tuple[int, int],
    frames: list[dict[str, int]],
    existing: set[tuple[int, int]],
    start_c: int,
    target_c: int,
    target_row: int,
    reserved_rows: set[int] | None = None,
) -> set[tuple[int, int]] | None:
    shifts = target_c - start_c
    if shifts < 0:
        return None
    rows = _plan_shift_rows(shape[0], target_row, shifts, reserved_rows=reserved_rows)
    if rows is None:
        return None
    frame_block = _cells_adjacent_to_boxes(shape, [_frame_bbox(f) for f in frames])
    cur_c = start_c
    added: set[tuple[int, int]] = set()
    for cur_r in rows:
        obs = (cur_r - 1, cur_c)
        right = (cur_r, cur_c + 1)
        if cur_c + 1 >= shape[1]:
            return None
        if obs in existing or obs in added or obs in frame_block:
            return None
        if right in existing or right in added or right in frame_block:
            return None
        added.add(obs)
        cur_c += 1
    return added if cur_c == target_c else None


def _scatter_safe_obstacles(
    grid: np.ndarray,
    frames: list[dict[str, int]],
    path_sensitive: set[tuple[int, int]],
    keep_clear_cols: set[int],
    existing_obstacles: set[tuple[int, int]],
) -> None:
    h, w = grid.shape
    blocked = _cells_adjacent_to_boxes(grid.shape, [_frame_bbox(f) for f in frames]) | path_sensitive
    candidates: list[tuple[int, int]] = []
    for r in range(1, h - 1):
        for c in range(w):
            cell = (r, c)
            if c in keep_clear_cols:
                continue
            if cell in blocked or cell in existing_obstacles:
                continue
            if grid[cell] != BLACK:
                continue
            candidates.append(cell)
    random.shuffle(candidates)
    for cell in candidates[: random.randint(1, min(5, len(candidates))) if candidates else 0]:
        grid[cell] = GREY


def _place_extra_frames(
    h: int,
    w: int,
    frames: list[dict[str, int]],
    target_count: int,
) -> bool:
    while len(frames) < target_count:
        extra = _random_frame(
            h,
            w,
            [_frame_bbox(f) for f in frames],
            (1, max(2, h // 2)),
            (1, w - 6),
        )
        if extra is None:
            return False
        frames.append(extra)
    return True


def _construct_attempt() -> tuple[np.ndarray, dict[str, object]] | None:
    scenario = random.choices(
        ["bottom", "path_stop", "frame_stop"], weights=[3, 3, 2], k=1
    )[0]
    target_frames = random.randint(3, 4) if scenario == "frame_stop" else random.randint(2, 4)
    if scenario == "frame_stop":
        h = random.randint(18, 22)
        w = random.randint(18, 24)
    else:
        h = random.randint(15, 20)
        w = random.randint(14, 22 if target_frames >= 4 else 18)

    frames: list[dict[str, int]] = []

    if scenario == "frame_stop":
        open_upper = _random_frame(h, w, [], (1, 3), (w // 3, w - 9))
        if open_upper is None:
            return None
        frames.append(open_upper)

        lower_h = random.randint(5, 7)
        lower_w = random.randint(5, 8)
        gap_c = _gap_column(open_upper)
        lower_r0_lo = open_upper["r1"] + 3
        lower_r0_hi = min(h - lower_h - 2, open_upper["r1"] + 7)
        lower_c0_lo = max(1, gap_c - (lower_w - 2))
        lower_c0_hi = min(gap_c - 1, w - lower_w - 1)
        if lower_r0_lo > lower_r0_hi or lower_c0_lo > lower_c0_hi:
            return None
        lower = {
            "r0": random.randint(lower_r0_lo, lower_r0_hi),
            "c0": random.randint(lower_c0_lo, lower_c0_hi),
        }
        lower["r1"] = lower["r0"] + lower_h - 1
        lower["c1"] = lower["c0"] + lower_w - 1
        lower["gap_c"] = random.randint(lower["c0"] + 1, lower["c1"] - 1)
        if not _boxes_clear(_frame_bbox(lower), [_frame_bbox(open_upper)], pad=1):
            return None
        frames.append(lower)

        blocked = _random_frame(h, w, [_frame_bbox(f) for f in frames], (2, 5), (1, w // 3))
        if blocked is None:
            return None
        frames.append(blocked)
        role_blocked = blocked
        role_open = open_upper
    else:
        left_first = random.choice([True, False])
        blocked_col_range = (1, max(2, w // 2 - 5)) if left_first else (w // 2, w - 6)
        open_col_range = (w // 2, w - 6) if left_first else (1, max(2, w // 2 - 5))
        blocked = _random_frame(h, w, [], (2, 5), blocked_col_range)
        if blocked is None:
            return None
        frames.append(blocked)
        open_frame = _random_frame(h, w, [_frame_bbox(blocked)], (1, 5), open_col_range)
        if open_frame is None:
            return None
        frames.append(open_frame)
        role_blocked = blocked
        role_open = open_frame

    if not _place_extra_frames(h, w, frames, target_frames):
        return None

    grid = np.zeros((h, w), dtype=int)
    for frame in frames:
        _draw_frame(grid, frame)

    obstacle_cells: set[tuple[int, int]] = set()
    starts: list[int] = []

    blocked_gap = role_blocked["gap_c"]
    max_blocked_shifts = min(3, blocked_gap, h - role_blocked["r1"] - 4)
    if max_blocked_shifts < 1:
        return None
    blocked_shifts = random.randint(1, max_blocked_shifts)
    blocked_start = blocked_gap - blocked_shifts
    blocked_obs = _guided_obstacles_to_column(
        grid.shape,
        frames,
        obstacle_cells,
        blocked_start,
        blocked_gap,
        role_blocked["r1"],
    )
    if blocked_obs is None:
        return None
    obstacle_cells |= blocked_obs
    starts.append(blocked_start)

    if scenario == "path_stop":
        gap_c = role_open["gap_c"]
        max_path_shifts = min(3, gap_c, h - role_open["r1"] - 5)
        if max_path_shifts < 1:
            return None
        path_shifts = random.randint(1, max_path_shifts)
        stop_start = gap_c - path_shifts
        cross_row = random.randint(role_open["r1"] + 3, min(h - 3, role_open["r1"] + 6))
        stop_obs = _guided_obstacles_to_column(
            grid.shape,
            frames,
            obstacle_cells,
            stop_start,
            gap_c,
            cross_row,
            reserved_rows={role_blocked["r1"] + 3},
        )
        if stop_obs is None:
            return None
        leave_obstacle = (cross_row - 1, gap_c)
        if leave_obstacle in obstacle_cells or leave_obstacle in stop_obs:
            return None
        if leave_obstacle in _cells_adjacent_to_boxes(grid.shape, [_frame_bbox(f) for f in frames]):
            return None
        stop_obs.add(leave_obstacle)
        obstacle_cells |= stop_obs
        starts.append(stop_start)
    else:
        safe_cols = [c for c in range(w) if c not in starts and c > max(f["gap_c"] for f in frames)]
        if not safe_cols:
            safe_cols = [c for c in range(w) if c not in starts and c < min(f["gap_c"] for f in frames)]
        if not safe_cols:
            return None
        starts.append(random.choice(safe_cols))

    desired_start_count = random.randint(2, 4)
    extra_start_pool = [c for c in range(w) if c not in starts and c > max(f["gap_c"] for f in frames)]
    random.shuffle(extra_start_pool)
    while len(starts) < desired_start_count and extra_start_pool:
        starts.append(extra_start_pool.pop())

    for r, c in obstacle_cells:
        grid[r, c] = GREY
    for c in starts:
        grid[h - 1, c] = RED

    analysis_before_scatter = _analyze(grid)
    path_sensitive: set[tuple[int, int]] = set()
    for path in analysis_before_scatter["paths"]:
        for r, c in path["coords"]:
            path_sensitive.add((r, c))
            if r > 0:
                path_sensitive.add((r - 1, c))
            if c + 1 < w:
                path_sensitive.add((r, c + 1))

    keep_clear_cols = {role_open["gap_c"], role_blocked["gap_c"]}
    if scenario == "frame_stop":
        keep_clear_cols.add(role_open["gap_c"])
    _scatter_safe_obstacles(grid, frames, path_sensitive, keep_clear_cols, obstacle_cells)
    return grid, {"scenario": scenario, "blocked_gap": (role_blocked["r1"], role_blocked["gap_c"]), "open_gap": (role_open["r1"], role_open["gap_c"])}


def _validate_generated(grid: np.ndarray, meta: dict[str, object]) -> bool:
    frames = _find_frames(grid)
    if not (2 <= len(frames) <= 4):
        return False
    start_count = int(np.sum(grid[-1] == RED))
    if not (2 <= start_count <= 4):
        return False

    analysis = _analyze(grid)
    frame_by_gap = {tuple(info["gap"]): info for info in analysis["frames"]}
    blocked_gap = tuple(meta["blocked_gap"])
    open_gap = tuple(meta["open_gap"])
    if blocked_gap not in frame_by_gap or open_gap not in frame_by_gap:
        return False

    blocked_info = frame_by_gap[blocked_gap]
    open_info = frame_by_gap[open_gap]
    if not blocked_info["blocked"] or open_info["blocked"]:
        return False
    if sum(1 for info in analysis["frames"] if info["blocked"]) < 1:
        return False
    if sum(1 for info in analysis["frames"] if not info["blocked"]) < 1:
        return False

    hit_paths = blocked_info["hit_paths"]
    if not hit_paths:
        return False
    if not any(any(detour["reason"] == GREY for detour in path["detours"]) for path in hit_paths):
        return False

    no_grey = grid.copy()
    no_grey[no_grey == GREY] = BLACK
    no_grey_analysis = _analyze(no_grey)
    no_grey_by_gap = {tuple(info["gap"]): info for info in no_grey_analysis["frames"]}
    if blocked_gap not in no_grey_by_gap or no_grey_by_gap[blocked_gap]["blocked"]:
        return False

    expected_stopper = meta["scenario"]
    stopper_map = {"bottom": "bottom", "path_stop": "path", "frame_stop": "frame"}
    if open_info["stopper"] != stopper_map[expected_stopper]:
        return False

    out = main(grid)
    if out.shape != grid.shape or np.array_equal(out, grid):
        return False
    if TEAL not in set(map(int, np.unique(out))):
        return False
    if RED not in set(map(int, np.unique(out))):
        return False

    status_signature = tuple(sorted((info["gap_c"], int(info["blocked"]), str(info["stopper"])) for info in analysis["frames"]))
    if len(set(status_signature)) == 1:
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        built = _construct_attempt()
        if built is None:
            continue
        grid, meta = built
        if _validate_generated(grid, meta):
            return grid
    raise RuntimeError("failed to generate a valid path-plug flood puzzle")
