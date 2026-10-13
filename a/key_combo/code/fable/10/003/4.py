from __future__ import annotations

import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
PINK = 6
ORANGE = 7
TEAL = 8
MAROON = 9

STRUCTURAL_COLORS = {BLACK, GREEN, GREY, TEAL}
PALETTE_COLOR_POOL = (BLUE, RED, YELLOW, PINK, ORANGE, MAROON)

MAX_GENERATION_ATTEMPTS = 400
MAIN_LEN_RANGE = (11, 19)
CROSS_LEN_RANGE = (10, 18)
DETOUR_OPTIONS = (1, 2, 3)
PALETTE_LEN_OPTIONS = (2, 3, 4, 5)
RUN_LEN_OPTIONS = (1, 2, 3, 4, 5)
EXTRA_TEAL_OPTIONS = (0, 1, 2, 3, 4)

DIR_TOP = (-1, 0)
DIR_RIGHT = (0, 1)
DIR_BOTTOM = (1, 0)
DIR_LEFT = (0, -1)


def in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = map(int, shape)
    return 0 <= r < h and 0 <= c < w


def neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _orthogonals(direction: tuple[int, int]) -> tuple[tuple[int, int], tuple[int, int]]:
    dr, dc = direction
    return (-dc, dr), (dc, -dr)


def _on_border(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = map(int, shape)
    return r == 0 or c == 0 or r == h - 1 or c == w - 1


def _start_info(grid: np.ndarray) -> tuple[tuple[int, int], tuple[int, int], str]:
    starts = np.argwhere(grid == GREEN)
    assert starts.shape[0] == 1, "expected exactly one green start pixel"
    r, c = map(int, starts[0])
    h, w = map(int, grid.shape)
    if r == 0:
        return (r, c), DIR_BOTTOM, "bottom"
    if r == h - 1:
        return (r, c), DIR_TOP, "top"
    if c == 0:
        return (r, c), DIR_RIGHT, "right"
    if c == w - 1:
        return (r, c), DIR_LEFT, "left"
    raise AssertionError("green start must lie on the border")


def _trace_path(
    grid: np.ndarray,
    start: tuple[int, int],
    direction: tuple[int, int],
    goal_edge: str,
) -> tuple[list[tuple[int, int]], int]:
    h, w = map(int, grid.shape)
    r, c = start
    dr, dc = direction
    path: list[tuple[int, int]] = []
    turns = 0
    seen_states = set()

    for _ in range(h * w * 4):
        state = (r, c, dr, dc)
        assert state not in seen_states, "path looped before reaching the opposite boundary"
        seen_states.add(state)

        fr, fc = r + dr, c + dc
        assert in_bounds(grid.shape, fr, fc), "path stepped out of bounds"
        if int(grid[fr, fc]) == TEAL:
            candidates = []
            for ndr, ndc in _orthogonals((dr, dc)):
                nr, nc = r + ndr, c + ndc
                if in_bounds(grid.shape, nr, nc) and int(grid[nr, nc]) == BLACK:
                    candidates.append((ndr, ndc, nr, nc))
            assert len(candidates) == 1, "a teal trigger must force exactly one turn"
            dr, dc, fr, fc = candidates[0]
            turns += 1
        else:
            assert int(grid[fr, fc]) == BLACK, "path may only travel through black cells"

        r, c = fr, fc
        path.append((r, c))
        if goal_edge == "top" and r == 0:
            return path, turns
        if goal_edge == "bottom" and r == h - 1:
            return path, turns
        if goal_edge == "left" and c == 0:
            return path, turns
        if goal_edge == "right" and c == w - 1:
            return path, turns

    raise AssertionError("path did not reach the opposite boundary")


def _find_marker(grid: np.ndarray) -> tuple[int, int]:
    coords = np.argwhere(grid == GREY)
    assert coords.size > 0, "expected a grey length marker"
    cols = np.unique(coords[:, 1])
    assert cols.size == 1, "grey marker must be a one-column segment"
    col = int(cols[0])
    rows = np.sort(coords[:, 0].astype(int))
    assert np.array_equal(rows, np.arange(rows.size)), "grey marker must be top-aligned and contiguous"
    return col, int(rows.size)


def _component_cells(mask: np.ndarray) -> list[tuple[int, int]]:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return []
    start = tuple(map(int, coords[0]))
    stack = [start]
    seen = {start}
    out = [start]
    while stack:
        r, c = stack.pop()
        for nr, nc in neighbors4(r, c):
            if in_bounds(mask.shape, nr, nc) and bool(mask[nr, nc]) and (nr, nc) not in seen:
                seen.add((nr, nc))
                stack.append((nr, nc))
                out.append((nr, nc))
    return out


def _find_palette(grid: np.ndarray) -> list[int]:
    mask = ~np.isin(grid, list(STRUCTURAL_COLORS))
    cells = _component_cells(mask)
    assert cells, "expected a border palette"
    assert len(cells) == int(mask.sum()), "palette colors must form a single component"

    rows = sorted({r for r, _ in cells})
    cols = sorted({c for _, c in cells})
    h, w = map(int, grid.shape)

    if len(cols) == 1:
        col = cols[0]
        assert col in (0, w - 1), "vertical palette must sit on a side border"
        assert rows == list(range(rows[0], rows[-1] + 1)), "palette cells must be contiguous"
        return [int(grid[r, col]) for r in rows]

    if len(rows) == 1:
        row = rows[0]
        assert row in (0, h - 1), "horizontal palette must sit on a border"
        assert cols == list(range(cols[0], cols[-1] + 1)), "palette cells must be contiguous"
        return [int(grid[row, c]) for c in cols]

    raise AssertionError("palette must be a one-cell-wide border strip")


def _paint_path(grid: np.ndarray, path: list[tuple[int, int]], palette: list[int], run_len: int) -> np.ndarray:
    out = np.array(grid, copy=True)
    for idx, (r, c) in enumerate(path):
        out[r, c] = int(palette[(idx // run_len) % len(palette)])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Trace the teal-triggered route across the grid and recolor its mileage using cyclic palette runs.

    Input structure:
    - Black background contains scattered teal trigger pixels, one green border start, one ordered border palette strip, and one top-aligned grey vertical marker segment.
    - The green start lies on a border and its inward direction is perpendicular to that border.
    - The palette cells form a single one-cell-wide ordered strip on a border; the grey marker is a single vertical segment whose height is visible.

    Rule:
    - Starting from the green pixel, move straight inward through black cells.
    - Whenever the next forward cell is teal, do not enter it; instead turn to the unique black side cell among the two perpendicular options.
    - Continue until the path reaches the opposite border, then recolor the visited path cells after the start in path order.
    - Read the palette strip in its border order and cycle through those colors, painting each color for exactly h consecutive path cells where h is the grey marker height.
    - Keep the teal triggers, palette strip, grey marker, and green start unchanged.

    Key insight:
    - The palette is applied along one-dimensional path arclength, so bends do not restart the color runs; only traveled step count matters.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    start, direction, goal_edge = _start_info(g)
    _marker_col, run_len = _find_marker(g)
    palette = _find_palette(g)
    path, _turns = _trace_path(g, start, direction, goal_edge)
    return _paint_path(g, path, palette, run_len)


def _transform_coord(
    coord: tuple[int, int], canonical_shape: tuple[int, int], start_edge: str
) -> tuple[int, int]:
    r, c = coord
    main_len, _cross_len = canonical_shape
    if start_edge == "top":
        return r, c
    if start_edge == "bottom":
        return main_len - 1 - r, c
    if start_edge == "left":
        return c, r
    if start_edge == "right":
        return c, main_len - 1 - r
    raise AssertionError(f"unknown start edge: {start_edge}")


def _final_shape(canonical_shape: tuple[int, int], start_edge: str) -> tuple[int, int]:
    main_len, cross_len = canonical_shape
    if start_edge in ("top", "bottom"):
        return main_len, cross_len
    return cross_len, main_len


def _sample_turn_rows(main_len: int, detours: int) -> list[int] | None:
    options = list(range(2, main_len - 2))
    for _ in range(80):
        rows = sorted(random.sample(options, detours))
        if all(b - a >= 2 for a, b in zip(rows, rows[1:])):
            return rows
    return None


def _sample_columns(cross_len: int, detours: int) -> list[int] | None:
    options = list(range(2, cross_len - 2))
    for _ in range(120):
        cols = [random.choice(options)]
        ok = True
        for _idx in range(detours):
            next_choices = [c for c in options if abs(c - cols[-1]) >= 2]
            if not next_choices:
                ok = False
                break
            cols.append(random.choice(next_choices))
        if ok:
            return cols
    return None


def _build_canonical_path(
    main_len: int, cross_len: int, detours: int
) -> tuple[list[tuple[int, int]], set[tuple[int, int]]] | None:
    rows = _sample_turn_rows(main_len, detours)
    cols = _sample_columns(cross_len, detours)
    if rows is None or cols is None:
        return None

    path = [(0, cols[0])]
    teal_cells: set[tuple[int, int]] = set()
    cur_row = 0
    cur_col = cols[0]

    for turn_row, next_col in zip(rows, cols[1:]):
        for r in range(cur_row + 1, turn_row + 1):
            path.append((r, cur_col))

        step = 1 if next_col > cur_col else -1
        teal_cells.add((turn_row + 1, cur_col))
        teal_cells.add((turn_row, cur_col - step))
        for c in range(cur_col + step, next_col + step, step):
            path.append((turn_row, c))

        teal_cells.add((turn_row, next_col + step))
        teal_cells.add((turn_row - 1, next_col))
        cur_row = turn_row
        cur_col = next_col

    for r in range(cur_row + 1, main_len):
        path.append((r, cur_col))

    if len(path) != len(set(path)):
        return None
    if teal_cells & set(path):
        return None
    if any(not in_bounds((main_len, cross_len), r, c) for r, c in teal_cells):
        return None
    return path, teal_cells


def _path_dirs(path: list[tuple[int, int]], start_dir: tuple[int, int]) -> list[tuple[int, int]]:
    dirs = []
    prev = start_dir
    for i in range(len(path)):
        if i + 1 < len(path):
            r0, c0 = path[i]
            r1, c1 = path[i + 1]
            prev = (r1 - r0, c1 - c0)
        dirs.append(prev)
    return dirs


def _safe_extra_teal_cells(
    shape: tuple[int, int],
    path: list[tuple[int, int]],
    teal_cells: set[tuple[int, int]],
    start: tuple[int, int],
    direction: tuple[int, int],
) -> list[tuple[int, int]]:
    forbidden = set(path) | set(teal_cells) | {start}
    h, w = shape
    cur = start
    dr, dc = direction
    for idx, cell in enumerate(path):
        fr, fc = cur[0] + dr, cur[1] + dc
        if (fr, fc) not in teal_cells:
            forbidden.add((fr, fc))
        else:
            for ndr, ndc in _orthogonals((dr, dc)):
                nr, nc = cur[0] + ndr, cur[1] + ndc
                if in_bounds(shape, nr, nc):
                    forbidden.add((nr, nc))
        if idx + 1 < len(path):
            nr, nc = path[idx + 1]
            dr, dc = nr - cell[0], nc - cell[1]
        cur = cell

    candidates = []
    for r in range(h):
        for c in range(w):
            if (r, c) in forbidden:
                continue
            if any(abs(r - rr) + abs(c - cc) <= 1 for rr, cc in forbidden):
                continue
            candidates.append((r, c))
    return candidates


def _find_palette_slot(
    shape: tuple[int, int],
    reserved: set[tuple[int, int]],
    palette_len: int,
) -> tuple[int, list[int]] | None:
    h, w = shape
    sides = [0, w - 1]
    random.shuffle(sides)
    choices: list[tuple[int, list[int]]] = []
    for col in sides:
        rows = [r for r in range(h) if (r, col) not in reserved]
        for start_idx in range(0, h - palette_len + 1):
            segment = list(range(start_idx, start_idx + palette_len))
            if all(r in rows for r in segment):
                choices.append((col, segment))
    if not choices:
        return None
    return random.choice(choices)


def _find_marker_col(
    shape: tuple[int, int],
    reserved: set[tuple[int, int]],
    run_len: int,
    palette_col: int,
) -> int | None:
    h, w = shape
    edge_pref = [c for c in (1, w - 2) if 0 <= c < w and c != palette_col]
    inner = [c for c in range(w) if c not in edge_pref and c != palette_col]
    random.shuffle(edge_pref)
    random.shuffle(inner)
    for col in edge_pref + inner:
        if run_len > h:
            return None
        if all((r, col) not in reserved for r in range(run_len)):
            return col
    return None


def _choose_run_and_palette(path_len: int) -> tuple[int, int] | None:
    wrap_first = random.random() < 0.65
    combos = []
    for run_len in RUN_LEN_OPTIONS:
        for palette_len in PALETTE_LEN_OPTIONS:
            if path_len <= run_len:
                continue
            cycle = run_len * palette_len
            if wrap_first and path_len > cycle:
                combos.append((run_len, palette_len))
            if (not wrap_first) and run_len < path_len <= cycle:
                combos.append((run_len, palette_len))
    if not combos:
        for run_len in RUN_LEN_OPTIONS:
            for palette_len in PALETTE_LEN_OPTIONS:
                if path_len > run_len:
                    combos.append((run_len, palette_len))
    return random.choice(combos) if combos else None


def _validate_instance(
    grid: np.ndarray,
    expected_output: np.ndarray,
    intended_path: list[tuple[int, int]],
    intended_turns: int,
    palette: list[int],
    run_len: int,
) -> bool:
    try:
        start, direction, goal_edge = _start_info(grid)
        marker_col, parsed_run_len = _find_marker(grid)
        parsed_palette = _find_palette(grid)
        traced_path, traced_turns = _trace_path(grid, start, direction, goal_edge)
    except AssertionError:
        return False

    if parsed_run_len != run_len:
        return False
    if parsed_palette != palette:
        return False
    if traced_path != intended_path:
        return False
    if traced_turns != intended_turns:
        return False
    if traced_turns < 2:
        return False

    out = main(grid)
    if not np.array_equal(out, expected_output):
        return False

    if len({out[r, c] for r, c in traced_path}) < 2:
        return False
    if marker_col < 0:
        return False
    return True


def _attempt_generate() -> np.ndarray | None:
    start_edge = random.choice(("top", "bottom", "left", "right"))
    main_len = random.randint(*MAIN_LEN_RANGE)
    cross_len = random.randint(*CROSS_LEN_RANGE)
    if start_edge in ("left", "right"):
        if max(main_len, cross_len) > 30:
            return None
    if min(main_len, cross_len) < 8:
        return None

    detour_candidates = [d for d in DETOUR_OPTIONS if main_len >= 2 * d + 7 and cross_len >= 2 * d + 5]
    if not detour_candidates:
        return None
    detours = random.choice(detour_candidates)
    built = _build_canonical_path(main_len, cross_len, detours)
    if built is None:
        return None
    canonical_path, canonical_teal = built

    canonical_shape = (main_len, cross_len)
    final_shape = _final_shape(canonical_shape, start_edge)
    transformed = [_transform_coord(cell, canonical_shape, start_edge) for cell in canonical_path]
    start = transformed[0]
    path = transformed[1:]
    teal_cells = {_transform_coord(cell, canonical_shape, start_edge) for cell in canonical_teal}
    h, w = final_shape

    if len(path) < 8:
        return None
    chosen = _choose_run_and_palette(len(path))
    if chosen is None:
        return None
    run_len, palette_len = chosen
    palette = random.sample(PALETTE_COLOR_POOL, palette_len)

    direction = {
        "top": DIR_BOTTOM,
        "bottom": DIR_TOP,
        "left": DIR_RIGHT,
        "right": DIR_LEFT,
    }[start_edge]

    reserved = set(path) | set(teal_cells) | {start}
    slot = _find_palette_slot(final_shape, reserved, palette_len)
    if slot is None:
        return None
    palette_col, palette_rows = slot
    for r in palette_rows:
        reserved.add((r, palette_col))

    marker_col = _find_marker_col(final_shape, reserved, run_len, palette_col)
    if marker_col is None:
        return None

    grid = np.full(final_shape, BLACK, dtype=int)
    for r, c in teal_cells:
        grid[r, c] = TEAL
    grid[start] = GREEN
    for r, color in zip(palette_rows, palette):
        grid[r, palette_col] = int(color)
    grid[:run_len, marker_col] = GREY

    extras = _safe_extra_teal_cells(final_shape, path, teal_cells, start, direction)
    random.shuffle(extras)
    for r, c in extras[: random.choice(EXTRA_TEAL_OPTIONS)]:
        grid[r, c] = TEAL

    expected_output = _paint_path(grid, path, palette, run_len)
    intended_turns = detours * 2
    if not _validate_instance(grid, expected_output, path, intended_turns, palette, run_len):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid path-mileage recoloring instance")
