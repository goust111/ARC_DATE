from __future__ import annotations

from collections import deque

import numpy as np


BG = 0
RED = 2
ORANGE = 7
CYAN = 8
MAROON = 9

NEIGHBORS_4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
NEIGHBORS_8 = [
    (dr, dc)
    for dr in (-1, 0, 1)
    for dc in (-1, 0, 1)
    if not (dr == 0 and dc == 0)
]

GEN_ATTEMPTS = 400


def _mirror_col(col: int, axis_sum: int) -> int:
    return axis_sum - col


def _bfs_path(
    mask: np.ndarray,
    start: tuple[int, int],
    goal: tuple[int, int],
    blocked_edge: tuple[tuple[int, int], tuple[int, int]] | None = None,
) -> list[tuple[int, int]] | None:
    h, w = mask.shape
    q = deque([start])
    parent = {start: None}

    while q:
        r, c = q.popleft()
        if (r, c) == goal:
            break
        for dr, dc in NEIGHBORS_4:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < h and 0 <= nc < w and mask[nr, nc]):
                continue
            if blocked_edge is not None:
                a, b = blocked_edge
                if ((r, c) == a and (nr, nc) == b) or ((r, c) == b and (nr, nc) == a):
                    continue
            nxt = (nr, nc)
            if nxt not in parent:
                parent[nxt] = (r, c)
                q.append(nxt)

    if goal not in parent:
        return None

    path = []
    cur = goal
    while cur is not None:
        path.append(cur)
        cur = parent[cur]
    return path[::-1]


def _unique_path(
    mask: np.ndarray, start: tuple[int, int], goal: tuple[int, int]
) -> tuple[bool, list[tuple[int, int]] | None]:
    path = _bfs_path(mask, start, goal)
    if path is None:
        return False, None

    for a, b in zip(path, path[1:]):
        if _bfs_path(mask, start, goal, blocked_edge=(a, b)) is not None:
            return False, path
    return True, path


def _forbidden_mask(grid: np.ndarray) -> np.ndarray:
    forbidden = np.zeros(grid.shape, dtype=bool)
    rows, cols = np.where(grid == ORANGE)
    for r, c in zip(rows, cols):
        forbidden[r, c] = True
        for dr, dc in NEIGHBORS_8:
            nr, nc = r + dr, c + dc
            if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1]:
                forbidden[nr, nc] = True
    return forbidden


def _safe_track_mask(grid: np.ndarray, passable_colors: set[int]) -> np.ndarray:
    forbidden = _forbidden_mask(grid)
    return np.isin(grid, sorted(passable_colors)) & (~forbidden)


def _endpoint_positions(grid: np.ndarray) -> list[tuple[int, int]]:
    points = [tuple(map(int, p)) for p in np.argwhere(grid == RED)]
    assert len(points) == 2, f"expected exactly two endpoints, got {len(points)}"
    return points


def _symmetry_fits(grid: np.ndarray, hole_color: int, axis_sum: int) -> tuple[bool, int]:
    h, w = grid.shape
    evidence = 0
    for r in range(h):
        for c in range(w):
            a = int(grid[r, c])
            mc = _mirror_col(c, axis_sum)
            b = int(grid[r, mc]) if 0 <= mc < w else BG
            if a == hole_color or b == hole_color:
                continue
            if a != b:
                return False, 0
            if a != BG or b != BG:
                evidence += 1
    if evidence == 0:
        return False, 0

    holes = np.argwhere(grid == hole_color)
    if len(holes) == 0:
        return False, 0
    for r, c in holes:
        mc = _mirror_col(int(c), axis_sum)
        if not (0 <= mc < w):
            return False, 0
        if int(grid[int(r), mc]) == hole_color:
            return False, 0
    return True, evidence


def _find_hole_color_and_axis(grid: np.ndarray) -> tuple[int, int]:
    g = np.asarray(grid, dtype=int)
    colors = [int(c) for c in np.unique(g) if c != BG]
    candidates = []
    for hole_color in colors:
        for axis_sum in range(0, 2 * g.shape[1] - 1):
            ok, evidence = _symmetry_fits(g, hole_color, axis_sum)
            if ok:
                candidates.append((hole_color, axis_sum, evidence))
    assert len(candidates) == 1, f"expected a unique hole-color/axis pair, got {candidates}"
    hole_color, axis_sum, _ = candidates[0]
    return hole_color, axis_sum


def _repair_by_reflection(grid: np.ndarray, hole_color: int, axis_sum: int) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    rows, cols = np.where(out == hole_color)
    for r, c in zip(rows, cols):
        mc = _mirror_col(int(c), axis_sum)
        assert 0 <= mc < out.shape[1], "hole mirror must stay inside the grid"
        mirrored = int(out[int(r), mc])
        assert mirrored != hole_color, "hole mirror must be visible"
        out[int(r), int(c)] = mirrored
    assert not np.any(out == hole_color), "all maroon occlusions must be resolved"
    return out


def _expected_output_from_completed(
    completed: np.ndarray, path_cells: list[tuple[int, int]]
) -> np.ndarray:
    out = completed.copy()
    for r, c in path_cells:
        if out[r, c] == CYAN:
            out[r, c] = RED
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Repair the maroon-occluded symmetric track map, then mark the unique obstacle-safe route between the red endpoints.

    Input structure:
    - A black background contains a cyan 4-connected track network with exactly two red endpoint cells embedded in it.
    - Orange obstacle pixels lie near the track, and maroon pixels overwrite some hidden cyan, orange, or black cells.
    - Ignoring the maroon cells reveals a left-right reflection pattern around one vertical axis, which may be on a column or between two columns and need not be centered in the full canvas.

    Rule:
    - Identify maroon as the occluding color and infer the unique reflection axis for the visible non-maroon pattern.
    - Replace each maroon cell with the color at its reflected counterpart to recover the full symmetric track-and-obstacle layout.
    - Mark every orange cell and its 8-neighborhood as forbidden, then search only the remaining cyan/red cells for the unique 4-connected path between the two red endpoints.
    - Recolor the cyan cells on that unique safe path to red, leaving all other cells unchanged.

    Key insight:
    - Reflection repair restores two opposite kinds of hidden information at once: extra cyan track needed by the final route and extra orange obstacle influence that removes the misleading detours.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    hole_color, axis_sum = _find_hole_color_and_axis(g)
    repaired = _repair_by_reflection(g, hole_color, axis_sum)

    start, goal = _endpoint_positions(repaired)
    safe_mask = _safe_track_mask(repaired, {CYAN, RED})
    unique, path = _unique_path(safe_mask, start, goal)
    assert unique and path is not None, "expected a unique safe path after repair"

    out = repaired.copy()
    for r, c in path:
        if out[r, c] == CYAN:
            out[r, c] = RED
    return out


def _build_pattern(rng: np.random.Generator):
    odd_axis = bool(rng.integers(0, 2))
    if odd_axis:
        half = int(rng.integers(5, 9))
        width = 2 * half + 1
    else:
        half = int(rng.integers(5, 9))
        width = 2 * half

    inner_left_max = (width - 3) // 2
    outer_left_max = inner_left_max - 2
    outer_left = int(rng.integers(1, outer_left_max + 1))
    inner_left = int(rng.integers(outer_left + 2, inner_left_max + 1))
    axis_sum_local = width - 1
    outer_right = _mirror_col(outer_left, axis_sum_local)
    inner_right = _mirror_col(inner_left, axis_sum_local)

    top_margin = int(rng.integers(1, 3))
    bottom_margin = int(rng.integers(1, 3))
    loop_height = int(rng.integers(4, 8))
    loops_down = bool(rng.integers(0, 2))
    if loops_down:
        main_row = top_margin
        loop_row = main_row + loop_height
    else:
        loop_row = top_margin
        main_row = loop_row + loop_height
    height = main_row + bottom_margin + 1
    if loops_down:
        height = max(height, loop_row + bottom_margin + 1)

    block_row = main_row + 2 if loops_down else main_row - 2
    hidden_left = bool(rng.integers(0, 2))

    completed = np.zeros((height, width), dtype=int)
    completed[main_row, :] = CYAN
    completed[main_row, 0] = RED
    completed[main_row, -1] = RED

    r0, r1 = sorted((main_row, loop_row))
    for col in (outer_left, inner_left, inner_right, outer_right):
        completed[r0 : r1 + 1, col] = CYAN
    completed[loop_row, outer_left : inner_left + 1] = CYAN
    completed[loop_row, inner_right : outer_right + 1] = CYAN

    obstacle_left = (block_row, outer_left - 1)
    obstacle_right = (block_row, outer_right + 1)
    completed[obstacle_left] = ORANGE
    completed[obstacle_right] = ORANGE

    path_cells = [(main_row, c) for c in range(width)]
    input_pattern = completed.copy()

    if hidden_left:
        input_pattern[obstacle_left] = MAROON
        track_candidates = [c for c in range(inner_left, max(inner_left + 1, width // 2))]
        track_cols = rng.choice(track_candidates, size=min(len(track_candidates), int(rng.integers(1, 3))), replace=False)
        for c in np.atleast_1d(track_cols):
            input_pattern[main_row, int(c)] = MAROON
    else:
        input_pattern[obstacle_right] = MAROON
        right_start = (width // 2) + (1 if width % 2 == 1 else 0)
        track_candidates = [c for c in range(max(right_start, 1), inner_right + 1)]
        track_cols = rng.choice(track_candidates, size=min(len(track_candidates), int(rng.integers(1, 3))), replace=False)
        for c in np.atleast_1d(track_cols):
            input_pattern[main_row, int(c)] = MAROON

    isolated_black_candidates = []
    for r in range(height):
        for c in range(width):
            if completed[r, c] != BG or input_pattern[r, c] == MAROON:
                continue
            mc = _mirror_col(c, axis_sum_local)
            if not (0 <= mc < width):
                continue
            if completed[r, mc] != BG or input_pattern[r, mc] == MAROON:
                continue
            if hidden_left and c > axis_sum_local // 2:
                continue
            if not hidden_left and c < (axis_sum_local + 1) // 2:
                continue
            touches_foreground = False
            for dr, dc in NEIGHBORS_4:
                nr, nc = r + dr, c + dc
                if 0 <= nr < height and 0 <= nc < width and completed[nr, nc] != BG:
                    touches_foreground = True
                    break
            if not touches_foreground:
                isolated_black_candidates.append((r, c))
    if isolated_black_candidates and bool(rng.integers(0, 2)):
        black_hole = isolated_black_candidates[int(rng.integers(0, len(isolated_black_candidates)))]
        input_pattern[black_hole] = MAROON

    return completed, input_pattern, path_cells


def _embed_pattern(
    rng: np.random.Generator,
    completed: np.ndarray,
    input_pattern: np.ndarray,
    path_cells: list[tuple[int, int]],
):
    ph, pw = completed.shape

    max_left = min(6, 30 - pw)
    max_right = min(6, 30 - pw)
    left_margin = int(rng.integers(0, max_left + 1))
    right_margin = int(rng.integers(0, max_right + 1))
    if left_margin == right_margin and left_margin < max_left and bool(rng.integers(0, 2)):
        right_margin += 1

    max_top = min(5, 30 - ph)
    max_bottom = min(5, 30 - ph)
    top_margin = int(rng.integers(0, max_top + 1))
    bottom_margin = int(rng.integers(0, max_bottom + 1))

    full_h = ph + top_margin + bottom_margin
    full_w = pw + left_margin + right_margin
    assert full_h <= 30 and full_w <= 30

    completed_full = np.zeros((full_h, full_w), dtype=int)
    input_full = np.zeros((full_h, full_w), dtype=int)
    completed_full[top_margin : top_margin + ph, left_margin : left_margin + pw] = completed
    input_full[top_margin : top_margin + ph, left_margin : left_margin + pw] = input_pattern

    path_full = [(top_margin + r, left_margin + c) for r, c in path_cells]
    return completed_full, input_full, path_full


def _validate_instance(
    input_grid: np.ndarray,
    completed_grid: np.ndarray,
    expected_output: np.ndarray,
    path_cells: list[tuple[int, int]],
) -> bool:
    if not np.any(input_grid == MAROON):
        return False

    try:
        hole_color, axis_sum = _find_hole_color_and_axis(input_grid)
    except AssertionError:
        return False
    if hole_color != MAROON:
        return False

    repaired = _repair_by_reflection(input_grid, hole_color, axis_sum)
    if not np.array_equal(repaired, completed_grid):
        return False

    endpoints = _endpoint_positions(completed_grid)
    final_safe = _safe_track_mask(completed_grid, {CYAN, RED})
    unique, path = _unique_path(final_safe, endpoints[0], endpoints[1])
    if not unique or path is None:
        return False
    if path != path_cells:
        return False

    pre_safe = _safe_track_mask(input_grid, {CYAN, RED, MAROON})
    pre_unique, pre_path = _unique_path(pre_safe, endpoints[0], endpoints[1])
    if pre_path is None or pre_unique:
        return False

    if not any(input_grid[r, c] == MAROON and completed_grid[r, c] == CYAN for r, c in path_cells):
        return False
    if not np.any((input_grid == MAROON) & (completed_grid == ORANGE)):
        return False

    if not np.array_equal(main(input_grid), expected_output):
        return False
    return True


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_ATTEMPTS):
        completed, input_pattern, path_cells = _build_pattern(rng)
        completed_full, input_full, path_full = _embed_pattern(rng, completed, input_pattern, path_cells)
        expected = _expected_output_from_completed(completed_full, path_full)
        if _validate_instance(input_full, completed_full, expected, path_full):
            return input_full
    raise RuntimeError(f"failed to generate a valid instance after {GEN_ATTEMPTS} attempts")
