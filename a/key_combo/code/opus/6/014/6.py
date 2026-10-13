from __future__ import annotations

import random

import numpy as np


BG = 0
CONTENT_COLORS = (1, 2, 3, 4)
BAR = 5
CONTROL = 9
GEN_ATTEMPTS = 500


def _manhattan(a: tuple[int, int], b: tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _find_marker(
    grid: np.ndarray,
) -> tuple[tuple[int, int], tuple[int, int], set[tuple[int, int]], tuple[str, int]]:
    grey = [tuple(map(int, rc)) for rc in np.argwhere(grid == BAR)]
    maroon = [tuple(map(int, rc)) for rc in np.argwhere(grid == CONTROL)]
    assert len(grey) == 3, "expected exactly one 3-pixel grey bar"
    assert len(maroon) == 1, "expected exactly one control pixel"

    rows = {r for r, _ in grey}
    cols = {c for _, c in grey}
    bar_cells: set[tuple[int, int]]

    if len(rows) == 1:
        row = next(iter(rows))
        ordered = sorted(c for _, c in grey)
        assert ordered[1] == ordered[0] + 1 and ordered[2] == ordered[1] + 1
        center = (row, ordered[1])
        bar_cells = {(row, ordered[0]), center, (row, ordered[2])}
        bar_axis = "horizontal"
    elif len(cols) == 1:
        col = next(iter(cols))
        ordered = sorted(r for r, _ in grey)
        assert ordered[1] == ordered[0] + 1 and ordered[2] == ordered[1] + 1
        center = (ordered[1], col)
        bar_cells = {(ordered[0], col), center, (ordered[2], col)}
        bar_axis = "vertical"
    else:
        raise AssertionError("grey marker must be a straight 3-pixel line")

    control = maroon[0]
    assert _manhattan(center, control) == 1, "control must touch bar center"
    step = (center[0] - control[0], center[1] - control[1])
    assert step in {(-1, 0), (1, 0), (0, -1), (0, 1)}

    if bar_axis == "horizontal":
        assert step[1] == 0, "horizontal bar must launch a vertical ray"
        active = ("col", center[1])
    else:
        assert step[0] == 0, "vertical bar must launch a horizontal ray"
        active = ("row", center[0])
    return center, control, bar_cells, active


def _content_coords(grid: np.ndarray) -> np.ndarray:
    return np.argwhere(np.isin(grid, CONTENT_COLORS))


def _sym(shape: tuple[int, int], r: int, c: int) -> tuple[int, int]:
    h, w = shape
    return h - 1 - int(r), w - 1 - int(c)


def _expected_output(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    _, control, bar_cells, active = _find_marker(g)
    out = g.copy()
    axis, index = active

    for r, c in _content_coords(g):
        rr, cc = _sym(g.shape, int(r), int(c))
        if g[rr, cc] != BG:
            continue
        if axis == "row" and rr == index:
            out[rr, cc] = int(g[r, c])
        if axis == "col" and cc == index:
            out[rr, cc] = int(g[r, c])

    for r, c in bar_cells | {control}:
        out[r, c] = BG
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Selectively complete only the ray-activated line under 180-degree central symmetry.

    Input structure:
    - The grid is an odd-sized black background with colored content pixels in colors 1..4.
    - A unique grey 3-pixel straight bar and an adjacent maroon control pixel form a marker.
    - The colored content is incomplete under 180-degree rotation, and some missing counterparts lie on the marker-selected line while others lie elsewhere.

    Rule:
    - Find the bar center and cast a ray from that center in the direction away from the maroon control pixel.
    - A horizontal ray activates only its row; a vertical ray activates only its column.
    - For every non-background content pixel, compute its 180-degree counterpart `(H-1-r, W-1-c)`.
    - If that counterpart is background and lies on the activated row or column, paint it with the source color; otherwise leave it blank.
    - Erase the grey bar and the maroon control pixel to background.

    Key insight:
    - The task is not global symmetry completion: central symmetry is applied only where the ray marker activates one specific row or one specific column.
    """
    return _expected_output(np.asarray(grid, dtype=int))


def _place_marker(
    grid: np.ndarray,
    ray_orientation: str,
) -> tuple[tuple[str, int], tuple[str, int], set[tuple[int, int]], tuple[int, int]]:
    h, w = grid.shape
    mid_r = (h - 1) // 2
    mid_c = (w - 1) // 2

    if ray_orientation == "horizontal":
        active_row = random.randint(mid_r + 2, h - 2)
        side = random.choice(("left", "right"))
        center_c = 1 if side == "left" else w - 2
        center = (active_row, center_c)
        control = (active_row, 0 if side == "left" else w - 1)
        bar_cells = {(active_row - 1, center_c), center, (active_row + 1, center_c)}
        active = ("row", active_row)
        source_line = ("row", h - 1 - active_row)
    else:
        active_col = random.randint(mid_c + 2, w - 2)
        side = random.choice(("top", "bottom"))
        center_r = 1 if side == "top" else h - 2
        center = (center_r, active_col)
        control = (0 if side == "top" else h - 1, active_col)
        bar_cells = {(center_r, active_col - 1), center, (center_r, active_col + 1)}
        active = ("col", active_col)
        source_line = ("col", w - 1 - active_col)

    for r, c in bar_cells:
        grid[r, c] = BAR
    grid[control] = CONTROL
    return active, source_line, bar_cells | {control}, center


def _active_candidates(
    shape: tuple[int, int],
    source_line: tuple[str, int],
    forbidden: set[tuple[int, int]],
) -> list[tuple[int, int]]:
    h, w = shape
    axis, index = source_line
    out: list[tuple[int, int]] = []

    if axis == "row":
        r = index
        for c in range(w):
            src = (r, c)
            dst = _sym(shape, r, c)
            if src in forbidden or dst in forbidden:
                continue
            out.append(src)
    else:
        c = index
        upper_rows = list(range((h - 1) // 2))
        for r in upper_rows:
            src = (r, c)
            dst = _sym(shape, r, c)
            if src in forbidden or dst in forbidden:
                continue
            out.append(src)
    random.shuffle(out)
    return out


def _decoy_candidates(
    shape: tuple[int, int],
    source_line: tuple[str, int],
    active: tuple[str, int],
    forbidden: set[tuple[int, int]],
) -> list[tuple[int, int]]:
    h, w = shape
    upper_rows = list(range((h - 1) // 2))
    out: list[tuple[int, int]] = []
    src_axis, src_index = source_line
    active_axis, active_index = active

    for r in upper_rows:
        for c in range(w):
            if (src_axis == "row" and r == src_index) or (src_axis == "col" and c == src_index):
                continue
            src = (r, c)
            dst = _sym(shape, r, c)
            if src in forbidden or dst in forbidden:
                continue
            if active_axis == "row" and dst[0] == active_index:
                continue
            if active_axis == "col" and dst[1] == active_index:
                continue
            out.append(src)
    random.shuffle(out)
    return out


def _sample_color_plan(active_capacity: int) -> tuple[list[int], dict[int, int], dict[int, int]] | None:
    color_count = random.randint(2, 4)
    colors = random.sample(list(CONTENT_COLORS), color_count)
    totals = {color: random.randint(3, 5) for color in colors}
    if active_capacity < 2:
        return None

    focus = random.choice(colors)
    active_counts = {color: 0 for color in colors}
    active_counts[focus] = 1
    other_choices = [color for color in colors if color != focus]
    active_counts[random.choice(other_choices)] = 1

    extra_budget = random.randint(0, min(4, active_capacity - 2))
    while extra_budget > 0:
        options = [
            color
            for color in colors
            if active_counts[color] < min(3, totals[color] - (1 if color == focus else 0))
        ]
        if not options:
            break
        color = random.choice(options)
        active_counts[color] += 1
        extra_budget -= 1

    if active_counts[focus] >= totals[focus]:
        return None
    if sum(active_counts.values()) < 2:
        return None
    return colors, totals, active_counts


def _paint_sources(
    grid: np.ndarray,
    colors: list[int],
    totals: dict[int, int],
    active_counts: dict[int, int],
    active_cells: list[tuple[int, int]],
    decoy_cells: list[tuple[int, int]],
) -> bool:
    need_active = sum(active_counts.values())
    need_decoy = sum(totals[color] - active_counts[color] for color in colors)
    if len(active_cells) < need_active or len(decoy_cells) < need_decoy:
        return False

    random.shuffle(active_cells)
    random.shuffle(decoy_cells)
    a_idx = 0
    d_idx = 0

    for color in colors:
        for _ in range(active_counts[color]):
            r, c = active_cells[a_idx]
            grid[r, c] = color
            a_idx += 1
        for _ in range(totals[color] - active_counts[color]):
            r, c = decoy_cells[d_idx]
            grid[r, c] = color
            d_idx += 1
    return True


def _global_completion(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    _, control, bar_cells, _ = _find_marker(g)
    out = g.copy()
    for r, c in _content_coords(g):
        rr, cc = _sym(g.shape, int(r), int(c))
        if g[rr, cc] == BG:
            out[rr, cc] = int(g[r, c])
    for r, c in bar_cells | {control}:
        out[r, c] = BG
    return out


def _semantic_checks(inp: np.ndarray, out: np.ndarray) -> bool:
    g = np.asarray(inp, dtype=int)
    solved = np.asarray(out, dtype=int)
    center, control, bar_cells, active = _find_marker(g)
    del center

    if np.any(solved == BAR) or np.any(solved == CONTROL):
        return False
    if g.shape != solved.shape or g.ndim != 2:
        return False

    content = _content_coords(g)
    colors = sorted({int(g[r, c]) for r, c in content})
    if not (2 <= len(colors) <= 4):
        return False
    if any(np.count_nonzero(g == color) < 3 or np.count_nonzero(g == color) > 8 for color in colors):
        return False
    if content.size == 0 or np.any(content[:, 0] >= (g.shape[0] - 1) // 2):
        return False

    active_axis, active_index = active
    active_missing: list[tuple[int, int, int]] = []
    inactive_missing: list[tuple[int, int, int]] = []
    color_modes: dict[int, set[str]] = {color: set() for color in colors}

    for r, c in content:
        rr, cc = _sym(g.shape, int(r), int(c))
        if g[rr, cc] != BG:
            continue
        color = int(g[r, c])
        if (active_axis == "row" and rr == active_index) or (active_axis == "col" and cc == active_index):
            active_missing.append((rr, cc, color))
            color_modes[color].add("active")
        else:
            inactive_missing.append((rr, cc, color))
            color_modes[color].add("inactive")

    if len(active_missing) < 2 or len(inactive_missing) < 1:
        return False
    if not any(modes == {"active", "inactive"} for modes in color_modes.values()):
        return False

    expected = _expected_output(g)
    if not np.array_equal(expected, solved):
        return False

    added = (solved != BG) & (g == BG)
    if int(np.count_nonzero(added)) != len(active_missing):
        return False

    global_completed = _global_completion(g)
    if np.array_equal(global_completed, solved):
        return False
    extra_from_global = (global_completed != solved) & (solved == BG)
    if int(np.count_nonzero(extra_from_global)) < len(inactive_missing):
        return False

    for r, c, color in active_missing:
        if solved[r, c] != color:
            return False
    for r, c, _ in inactive_missing:
        if solved[r, c] != BG:
            return False

    for r, c in bar_cells | {control}:
        if solved[r, c] != BG:
            return False
    return not np.array_equal(g, solved)


def _gen_attempt() -> np.ndarray | None:
    h = random.choice((11, 13, 15))
    w = random.choice((11, 13, 15))
    grid = np.zeros((h, w), dtype=int)

    ray_orientation = random.choice(("horizontal", "vertical"))
    active, source_line, forbidden, _ = _place_marker(grid, ray_orientation)

    active_cells = _active_candidates(grid.shape, source_line, forbidden)
    decoy_cells = _decoy_candidates(grid.shape, source_line, active, forbidden)
    plan = _sample_color_plan(len(active_cells))
    if plan is None:
        return None
    colors, totals, active_counts = plan

    if not _paint_sources(grid, colors, totals, active_counts, active_cells, decoy_cells):
        return None

    try:
        out = main(grid)
    except Exception:
        return None
    if not _semantic_checks(grid, out):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 6")
