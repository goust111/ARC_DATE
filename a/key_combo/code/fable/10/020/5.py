from __future__ import annotations

from collections import Counter

import numpy as np


BG = 0
OUT_RED = 2
OBSTACLE = 3
OUT_YELLOW = 4
MARKER = 5
FRAME = 6

CARDINALS = [(-1, 0), (1, 0), (0, -1), (0, 1)]

GEN_MIN_RECT = 6
GEN_MAX_RECT = 11
GEN_MAX_ATTEMPTS = 400
GEN_MAX_NOISE = 6


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, np.asarray(grid).ravel())).most_common(1)[0][0])


def in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _sign(value: int) -> int:
    if value > 0:
        return 1
    if value < 0:
        return -1
    return 0


def _mask_path_summary(free: np.ndarray) -> dict[str, object]:
    """Summarize all shortest monotone paths on a local down-right boolean corridor mask."""
    h, w = free.shape
    forward = [[0] * w for _ in range(h)]
    backward = [[0] * w for _ in range(h)]

    if free[0, 0]:
        forward[0][0] = 1
    for r in range(h):
        for c in range(w):
            if not free[r, c] or (r == 0 and c == 0):
                continue
            total = 0
            if r > 0:
                total += forward[r - 1][c]
            if c > 0:
                total += forward[r][c - 1]
            forward[r][c] = total

    if free[h - 1, w - 1]:
        backward[h - 1][w - 1] = 1
    for r in range(h - 1, -1, -1):
        for c in range(w - 1, -1, -1):
            if not free[r, c] or (r == h - 1 and c == w - 1):
                continue
            total = 0
            if r + 1 < h:
                total += backward[r + 1][c]
            if c + 1 < w:
                total += backward[r][c + 1]
            backward[r][c] = total

    total_paths = int(forward[h - 1][w - 1])
    union = np.zeros_like(free, dtype=bool)
    forced = np.zeros_like(free, dtype=bool)
    if total_paths:
        for r in range(h):
            for c in range(w):
                if not free[r, c]:
                    continue
                left = int(forward[r][c])
                right = int(backward[r][c])
                if left and right:
                    union[r, c] = True
                    if left * right == total_paths:
                        forced[r, c] = True

    return {
        "total_paths": total_paths,
        "union": union,
        "forced": forced,
    }


def _find_doorways(grid: np.ndarray, bg: int) -> list[dict[str, tuple[int, int] | int]]:
    doors: list[dict[str, tuple[int, int] | int]] = []
    h, w = grid.shape

    for r in range(1, h - 1):
        for c in range(1, w - 1):
            center_color = int(grid[r, c])
            if center_color == bg:
                continue

            gaps: list[tuple[int, int]] = []
            for dr, dc in CARDINALS:
                rr, cc = r + dr, c + dc
                if int(grid[rr, cc]) == bg:
                    gaps.append((dr, dc))
            if len(gaps) != 1:
                continue

            gap_dir = gaps[0]
            frame_color = None
            ok = True
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    rr, cc = r + dr, c + dc
                    value = int(grid[rr, cc])
                    if (dr, dc) == gap_dir:
                        ok &= value == bg
                        continue
                    if value == bg:
                        ok = False
                        continue
                    if frame_color is None:
                        frame_color = value
                    elif value != frame_color:
                        ok = False
            if not ok or frame_color is None or frame_color == center_color:
                continue

            ext = (r + 2 * gap_dir[0], c + 2 * gap_dir[1])
            if not in_bounds(grid.shape, *ext):
                continue
            if int(grid[ext]) != bg:
                continue

            doors.append(
                {
                    "marker": (r, c),
                    "gap": (r + gap_dir[0], c + gap_dir[1]),
                    "exterior": ext,
                    "marker_color": center_color,
                    "frame_color": int(frame_color),
                }
            )

    assert len(doors) == 2, "expected exactly two doorway frames"
    marker_colors = {int(door["marker_color"]) for door in doors}
    frame_colors = {int(door["frame_color"]) for door in doors}
    assert len(marker_colors) == 1 and len(frame_colors) == 1, "doorway styles must match"
    return sorted(doors, key=lambda door: tuple(door["exterior"]))  # stable only


def _local_free_mask(grid: np.ndarray, bg: int, start: tuple[int, int], end: tuple[int, int]) -> np.ndarray:
    row_step = _sign(end[0] - start[0])
    col_step = _sign(end[1] - start[1])
    dh = abs(end[0] - start[0])
    dw = abs(end[1] - start[1])
    free = np.zeros((dh + 1, dw + 1), dtype=bool)

    for i in range(dh + 1):
        r = start[0] + row_step * i
        for j in range(dw + 1):
            c = start[1] + col_step * j
            free[i, j] = int(grid[r, c]) == bg

    assert free[0, 0] and free[-1, -1], "door exteriors must stay on background"
    return free


def _paint_local_mask(
    out: np.ndarray,
    mask: np.ndarray,
    start: tuple[int, int],
    end: tuple[int, int],
    color: int,
) -> None:
    row_step = _sign(end[0] - start[0])
    col_step = _sign(end[1] - start[1])

    for i, j in np.argwhere(mask):
        r = start[0] + row_step * int(i)
        c = start[1] + col_step * int(j)
        out[r, c] = color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Mark the cells that can be walked through and the cells that must be walked through between two doorway frames.

    Input structure:
    - The grid has a dominant background, two matching 3x3 doorway frames, and one center marker inside each frame.
    - Each marker has exactly one background 4-neighbor: the doorway gap, with one more background cell outside it as the exterior doorway cell.
    - Additional foreground obstacle cells may lie between the two exterior doorway cells.

    Rule:
    - Treat every non-background input cell as a blocker, except that the two doorway gaps remain background openings.
    - Between the two exterior doorway cells, allow only shortest monotone 4-neighbor walks that always step toward the other exterior cell.
    - Paint every background cell on at least one such walk yellow, then repaint the cells on every such walk red; both doorway gaps are mandatory and therefore also red.

    Key insight:
    - The output separates the union of all valid shortest monotone paths from their intersection, so only bottlenecks shared by every route become red.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = get_bg(g)
    doors = _find_doorways(g, bg)

    start = tuple(doors[0]["exterior"])
    end = tuple(doors[1]["exterior"])
    free = _local_free_mask(g, bg, start, end)
    summary = _mask_path_summary(free)
    assert int(summary["total_paths"]) > 0, "doorway exteriors must be connected by at least one monotone path"

    out = g.copy()
    _paint_local_mask(out, np.asarray(summary["union"], dtype=bool), start, end, OUT_YELLOW)
    _paint_local_mask(out, np.asarray(summary["forced"], dtype=bool), start, end, OUT_RED)
    for door in doors:
        gap = tuple(door["gap"])
        out[gap] = OUT_RED
    return out


def _checkpoint_chain(
    dh: int, dw: int, bottlenecks: int, rng: np.random.Generator
) -> list[tuple[int, int]]:
    points: list[tuple[int, int]] = []
    for _ in range(200):
        points.clear()
        if bottlenecks == 1:
            i = int(rng.integers(1, dh))
            j = int(rng.integers(1, dw))
            points.append((i, j))
        else:
            i1 = int(rng.integers(1, dh - 1))
            j1 = int(rng.integers(1, dw - 1))
            i2 = int(rng.integers(i1 + 1, dh))
            j2 = int(rng.integers(j1 + 1, dw))
            points.extend([(i1, j1), (i2, j2)])
        if all(0 < i < dh and 0 < j < dw for i, j in points):
            return points
    raise RuntimeError("failed to sample bottlenecks")


def _base_free_mask(dh: int, dw: int, checkpoints: list[tuple[int, int]]) -> np.ndarray:
    free = np.ones((dh + 1, dw + 1), dtype=bool)
    for checkpoint in checkpoints:
        diag = checkpoint[0] + checkpoint[1]
        i0 = max(0, diag - dw)
        i1 = min(dh, diag)
        for i in range(i0, i1 + 1):
            j = diag - i
            if (i, j) != checkpoint:
                free[i, j] = False
    return free


def _semantic_ok(
    free: np.ndarray,
    summary: dict[str, object],
    checkpoints: list[tuple[int, int]],
    min_obstacles: int,
) -> bool:
    total_paths = int(summary["total_paths"])
    union = np.asarray(summary["union"], dtype=bool)
    forced = np.asarray(summary["forced"], dtype=bool)
    if total_paths < 2:
        return False
    if np.count_nonzero(~free) < min_obstacles:
        return False

    internal_forced = forced.copy()
    internal_forced[0, 0] = False
    internal_forced[-1, -1] = False
    if np.count_nonzero(internal_forced) < len(checkpoints):
        return False
    for point in checkpoints:
        if not forced[point]:
            return False

    yellow = union & ~forced
    if np.count_nonzero(yellow) < 6:
        return False

    before = union[: checkpoints[0][0] + 1, : checkpoints[0][1] + 1].copy()
    before[0, 0] = False
    before[checkpoints[0]] = False
    if not np.any(before):
        return False

    after = union[checkpoints[-1][0] :, checkpoints[-1][1] :].copy()
    after[-1, -1] = False
    after[0, 0] = False
    if not np.any(after):
        return False

    return True


def _add_noise_blockers(
    free: np.ndarray, checkpoints: list[tuple[int, int]], rng: np.random.Generator
) -> tuple[np.ndarray, dict[str, object]]:
    summary = _mask_path_summary(free)
    protected = {(0, 0), (free.shape[0] - 1, free.shape[1] - 1), *checkpoints}
    target = int(rng.integers(1, GEN_MAX_NOISE + 1))
    added = 0

    while added < target:
        union = np.asarray(summary["union"], dtype=bool)
        forced = np.asarray(summary["forced"], dtype=bool)
        candidates = [
            (int(r), int(c))
            for r, c in np.argwhere(union)
            if free[r, c] and not forced[r, c] and (int(r), int(c)) not in protected
        ]
        if not candidates:
            break
        rng.shuffle(candidates)

        accepted = False
        for r, c in candidates:
            trial = free.copy()
            trial[r, c] = False
            trial_summary = _mask_path_summary(trial)
            if not _semantic_ok(trial, trial_summary, checkpoints, min_obstacles=len(checkpoints) + 1):
                continue
            if np.array_equal(np.asarray(trial_summary["union"]), union) and np.array_equal(
                np.asarray(trial_summary["forced"]), forced
            ):
                continue
            free = trial
            summary = trial_summary
            added += 1
            accepted = True
            break
        if not accepted:
            break

    return free, summary


def _door_frames_for_signs(
    dh: int, dw: int, row_sign: int, col_sign: int, rng: np.random.Generator
) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int], tuple[int, int]]:
    start_dir_options = [(row_sign, 0), (0, col_sign)]
    end_dir_options = [(-row_sign, 0), (0, -col_sign)]
    start_dir = start_dir_options[int(rng.integers(0, 2))]
    end_dir = end_dir_options[int(rng.integers(0, 2))]
    return (0, 0), (row_sign * dh, col_sign * dw), start_dir, end_dir


def _frame_cells(marker: tuple[int, int], gap_dir: tuple[int, int]) -> tuple[list[tuple[int, int]], tuple[int, int]]:
    cells: list[tuple[int, int]] = []
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            if (dr, dc) == gap_dir:
                continue
            cells.append((marker[0] + dr, marker[1] + dc))
    gap = (marker[0] + gap_dir[0], marker[1] + gap_dir[1])
    return cells, gap


def _layout_and_render(
    free: np.ndarray,
    summary: dict[str, object],
    row_sign: int,
    col_sign: int,
    start_dir: tuple[int, int],
    end_dir: tuple[int, int],
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    dh, dw = free.shape[0] - 1, free.shape[1] - 1

    def map_local(i: int, j: int) -> tuple[int, int]:
        return row_sign * i, col_sign * j

    start_ext = (0, 0)
    end_ext = (row_sign * dh, col_sign * dw)
    start_marker = (start_ext[0] - 2 * start_dir[0], start_ext[1] - 2 * start_dir[1])
    end_marker = (end_ext[0] - 2 * end_dir[0], end_ext[1] - 2 * end_dir[1])
    start_frame, start_gap = _frame_cells(start_marker, start_dir)
    end_frame, end_gap = _frame_cells(end_marker, end_dir)

    obstacle_cells = [map_local(int(r), int(c)) for r, c in np.argwhere(~free)]
    all_points = obstacle_cells + start_frame + end_frame + [start_gap, end_gap, start_marker, end_marker, start_ext, end_ext]
    rows = [p[0] for p in all_points]
    cols = [p[1] for p in all_points]
    min_r, max_r = min(rows), max(rows)
    min_c, max_c = min(cols), max(cols)

    span_h = max_r - min_r + 1
    span_w = max_c - min_c + 1
    top = int(rng.integers(1, 5))
    bottom = int(rng.integers(1, 5))
    left = int(rng.integers(1, 5))
    right = int(rng.integers(1, 5))
    while span_h + top + bottom > 30:
        if top > bottom and top > 1:
            top -= 1
        elif bottom > 1:
            bottom -= 1
    while span_w + left + right > 30:
        if left > right and left > 1:
            left -= 1
        elif right > 1:
            right -= 1

    shift_r = top - min_r
    shift_c = left - min_c
    grid = np.full((span_h + top + bottom, span_w + left + right), BG, dtype=int)

    def put(point: tuple[int, int], color: int) -> tuple[int, int]:
        rr = point[0] + shift_r
        cc = point[1] + shift_c
        assert in_bounds(grid.shape, rr, cc)
        grid[rr, cc] = color
        return rr, cc

    for point in obstacle_cells:
        put(point, OBSTACLE)
    for point in start_frame + end_frame:
        put(point, FRAME)
    put(start_marker, MARKER)
    put(end_marker, MARKER)

    output = grid.copy()
    actual_start = (start_ext[0] + shift_r, start_ext[1] + shift_c)
    actual_end = (end_ext[0] + shift_r, end_ext[1] + shift_c)
    _paint_local_mask(output, np.asarray(summary["union"], dtype=bool), actual_start, actual_end, OUT_YELLOW)
    _paint_local_mask(output, np.asarray(summary["forced"], dtype=bool), actual_start, actual_end, OUT_RED)
    output[start_gap[0] + shift_r, start_gap[1] + shift_c] = OUT_RED
    output[end_gap[0] + shift_r, end_gap[1] + shift_c] = OUT_RED

    return grid, output


def _attempt(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray] | None:
    dh = int(rng.integers(GEN_MIN_RECT, GEN_MAX_RECT + 1))
    dw = int(rng.integers(GEN_MIN_RECT, GEN_MAX_RECT + 1))
    row_sign = int(rng.choice([-1, 1]))
    col_sign = int(rng.choice([-1, 1]))
    bottlenecks = 1 if rng.random() < 0.45 else 2

    checkpoints = _checkpoint_chain(dh, dw, bottlenecks, rng)
    free = _base_free_mask(dh, dw, checkpoints)
    free, summary = _add_noise_blockers(free, checkpoints, rng)
    if not _semantic_ok(free, summary, checkpoints, min_obstacles=len(checkpoints) + 2):
        return None

    _start_ext, _end_ext, start_dir, end_dir = _door_frames_for_signs(dh, dw, row_sign, col_sign, rng)
    inp, expected = _layout_and_render(free, summary, row_sign, col_sign, start_dir, end_dir, rng)

    if not np.array_equal(main(inp), expected):
        return None
    if np.count_nonzero(inp == MARKER) != 2:
        return None
    if not (np.any(expected == OUT_RED) and np.any(expected == OUT_YELLOW)):
        return None
    return inp, expected


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_MAX_ATTEMPTS):
        result = _attempt(rng)
        if result is None:
            continue
        return result[0]
    raise RuntimeError("failed to generate a valid doorway-corridor input")
