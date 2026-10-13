from __future__ import annotations

import random

import numpy as np


BG = 0
PILLAR = 5
GROUND = 9
TOP_COLORS = (1, 2, 3, 4, 6, 7, 8)

HEIGHT_MIN = 2
HEIGHT_MAX = 10
TOWER_COUNT_MIN = 3
TOWER_COUNT_MAX = 6
HEADROOM_MIN = 1
HEADROOM_MAX = 3
GAP_MIN = 2
GAP_MAX = 5
SIDE_MARGIN_MIN = 1
SIDE_MARGIN_MAX = 3
LAYOUT_RETRIES = 300
PERMUTATION_RETRIES = 300


def _sorted_towers(towers: list[dict]) -> list[dict]:
    return sorted(towers, key=lambda tower: (-int(tower["height"]), int(tower["col"])))


def _append_segment(path: list[tuple[int, int]], segment: list[tuple[int, int]]) -> None:
    if not segment:
        return
    if path and path[-1] == segment[0]:
        path.extend(segment[1:])
    else:
        path.extend(segment)


def _axis_line(r0: int, c0: int, r1: int, c1: int) -> list[tuple[int, int]]:
    assert (r0 == r1) != (c0 == c1) or (r0 == r1 and c0 == c1)
    if r0 == r1:
        step = 1 if c1 >= c0 else -1
        return [(int(r0), int(c)) for c in range(int(c0), int(c1) + step, step)]
    step = 1 if r1 >= r0 else -1
    return [(int(r), int(c0)) for r in range(int(r0), int(r1) + step, step)]


def _blocking_towers_between(towers: list[dict], src: dict, dst: dict) -> list[dict]:
    lo = min(int(src["col"]), int(dst["col"]))
    hi = max(int(src["col"]), int(dst["col"]))
    base_row = int(src["top_row"])
    blockers = []
    for tower in towers:
        col = int(tower["col"])
        if lo < col < hi and int(tower["top_row"]) <= base_row:
            blockers.append(tower)
    return blockers


def _ray_path(towers: list[dict], src: dict, dst: dict) -> list[tuple[int, int]]:
    src_col = int(src["col"])
    dst_col = int(dst["col"])
    base_row = int(src["top_row"])
    if src_col == dst_col:
        return [(base_row, src_col)]

    step = 1 if dst_col > src_col else -1
    blockers = _blocking_towers_between(towers, src, dst)
    if not blockers:
        return _axis_line(base_row, src_col, base_row, dst_col)

    first_blocker = min(blockers, key=lambda tower: step * int(tower["col"]))
    turn_col = int(first_blocker["col"]) - step
    detour_row = min(int(tower["top_row"]) for tower in blockers) - 1
    assert detour_row >= 0, "detour needs one free row above the tallest blocker"

    path: list[tuple[int, int]] = []
    _append_segment(path, _axis_line(base_row, src_col, base_row, turn_col))
    _append_segment(path, _axis_line(base_row, turn_col, detour_row, turn_col))
    _append_segment(path, _axis_line(detour_row, turn_col, detour_row, dst_col))
    _append_segment(path, _axis_line(detour_row, dst_col, base_row, dst_col))
    return path


def _render_input(towers: list[dict], height: int, width: int) -> np.ndarray:
    grid = np.full((int(height), int(width)), BG, dtype=int)
    grid[-1, :] = GROUND
    for tower in towers:
        top_row = int(tower["top_row"])
        col = int(tower["col"])
        color = int(tower["color"])
        grid[top_row, col] = color
        if top_row + 1 < height - 1:
            grid[top_row + 1 : height - 1, col] = PILLAR
    return grid


def _parse_towers(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    height, width = map(int, g.shape)
    assert height >= HEIGHT_MIN + 2
    assert width >= TOWER_COUNT_MIN + 2
    assert np.all(g[-1, :] == GROUND), "bottom row must be the maroon ground line"

    towers: list[dict] = []
    for col in range(width):
        column = g[:-1, col]
        non_bg_rows = np.flatnonzero(column != BG)
        if len(non_bg_rows) == 0:
            continue

        top_row = int(non_bg_rows[0])
        top_color = int(column[top_row])
        assert top_color not in (BG, PILLAR, GROUND)
        assert bool(np.all(column[top_row + 1 :] == PILLAR)), "tower body must be grey"
        towers.append(
            {
                "col": int(col),
                "top_row": int(top_row),
                "height": int(height - 1 - top_row),
                "color": int(top_color),
            }
        )

    assert len(towers) >= 2, "expected at least two towers"
    assert len({int(tower["color"]) for tower in towers}) == len(towers), "tower-top colors must be unique"

    expected = _render_input(towers, height, width)
    assert np.array_equal(expected, g), "input must contain only towers on black background plus the ground line"
    return towers


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect tower tops in descending height order.

    Each tower sends a ray to the next shorter tower using its own top color.
    Rays travel horizontally at the sender's top row. If a taller intermediate tower
    blocks that row, the ray rises one cell above the tallest blocker in the span,
    crosses there, then descends at the target column.
    """
    g = np.asarray(grid, dtype=int)
    towers = _parse_towers(g)
    ordered = _sorted_towers(towers)
    structural = g != BG

    out = g.copy()
    for src, dst in zip(ordered, ordered[1:]):
        color = int(src["color"])
        source_cell = (int(src["top_row"]), int(src["col"]))
        for r, c in _ray_path(towers, src, dst):
            if structural[int(r), int(c)] and (int(r), int(c)) != source_cell:
                continue
            out[int(r), int(c)] = color
    return out


def _path_cells_on_background(towers: list[dict], height: int, width: int) -> list[set[tuple[int, int]]]:
    structural = _render_input(towers, height, width) != BG
    cells_per_ray: list[set[tuple[int, int]]] = []
    ordered = _sorted_towers(towers)
    for src, dst in zip(ordered, ordered[1:]):
        cells = {
            (int(r), int(c))
            for r, c in _ray_path(towers, src, dst)
            if not structural[int(r), int(c)]
        }
        cells_per_ray.append(cells)
    return cells_per_ray


def _layout_is_clear(towers: list[dict], height: int, width: int) -> bool:
    seen: set[tuple[int, int]] = set()
    for cells in _path_cells_on_background(towers, height, width):
        if seen & cells:
            return False
        seen |= cells
    return True


def _sample_columns(count: int) -> tuple[list[int], int]:
    left_margin = random.randint(SIDE_MARGIN_MIN, SIDE_MARGIN_MAX)
    right_margin = random.randint(SIDE_MARGIN_MIN, SIDE_MARGIN_MAX)
    cols = [left_margin]
    for _ in range(count - 1):
        cols.append(cols[-1] + random.randint(GAP_MIN, GAP_MAX))
    width = cols[-1] + right_margin + 1
    return cols, width


def _sample_towers(needs_detour: bool, want_mixed_directions: bool) -> tuple[list[dict], int, int] | None:
    count = random.randint(TOWER_COUNT_MIN, TOWER_COUNT_MAX)
    heights = sorted(random.sample(range(HEIGHT_MIN, HEIGHT_MAX + 1), count), reverse=True)
    headroom = random.randint(HEADROOM_MIN, HEADROOM_MAX)
    grid_h = max(heights) + headroom + 1
    cols, grid_w = _sample_columns(count)
    colors = random.sample(list(TOP_COLORS), count)

    for _ in range(PERMUTATION_RETRIES):
        slot_order = random.sample(range(count), count)
        towers = []
        for rank, slot in enumerate(slot_order):
            tower_height = int(heights[rank])
            towers.append(
                {
                    "col": int(cols[slot]),
                    "top_row": int(grid_h - 1 - tower_height),
                    "height": int(tower_height),
                    "color": int(colors[rank]),
                }
            )

        ordered = _sorted_towers(towers)
        obstructed = [bool(_blocking_towers_between(towers, src, dst)) for src, dst in zip(ordered, ordered[1:])]
        if needs_detour != any(obstructed):
            continue

        directions = [1 if int(dst["col"]) > int(src["col"]) else -1 for src, dst in zip(ordered, ordered[1:])]
        if want_mixed_directions and len(set(directions)) < 2:
            continue

        if not _layout_is_clear(towers, grid_h, grid_w):
            continue
        return towers, grid_h, grid_w
    return None


def _gen_attempt() -> np.ndarray | None:
    needs_detour = bool(random.getrandbits(1))
    want_mixed_directions = bool(random.getrandbits(1))

    layout = _sample_towers(needs_detour, want_mixed_directions)
    if layout is None and want_mixed_directions:
        layout = _sample_towers(needs_detour, False)
    if layout is None:
        return None

    towers, grid_h, grid_w = layout
    inp = _render_input(towers, grid_h, grid_w)

    try:
        parsed = _parse_towers(inp)
    except AssertionError:
        return None
    if len(parsed) != len(towers):
        return None

    out = main(inp)
    if out.shape != inp.shape:
        return None
    if not np.array_equal(inp[inp != BG], out[inp != BG]):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(LAYOUT_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid signal-tower puzzle")
