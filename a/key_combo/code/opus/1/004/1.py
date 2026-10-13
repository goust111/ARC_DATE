from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
SOURCE = 1
RAIL_COLORS = (2, 3, 4)
GEN_ATTEMPTS = 500


def _collect_rail_components(grid: np.ndarray) -> list[tuple[int, list[tuple[int, int]]]]:
    h, w = grid.shape
    mask = (grid != BG) & (grid != SOURCE)
    seen = np.zeros((h, w), dtype=bool)
    components: list[tuple[int, list[tuple[int, int]]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            color = int(grid[r, c])
            q: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        if dr == 0 and dc == 0:
                            continue
                        nr = rr + dr
                        nc = cc + dc
                        if not (0 <= nr < h and 0 <= nc < w):
                            continue
                        if seen[nr, nc] or int(grid[nr, nc]) != color:
                            continue
                        seen[nr, nc] = True
                        q.append((nr, nc))
            components.append((color, cells))
    return components


def _analyze_rail(color: int, cells: list[tuple[int, int]], shape: tuple[int, int]) -> dict:
    h, w = shape
    assert 3 <= len(cells) <= 6
    ordered = sorted((int(r), int(c)) for r, c in cells)
    rows = [r for r, _ in ordered]
    cols = [c for _, c in ordered]
    assert len(set(rows)) == len(ordered)
    assert len(set(cols)) == len(ordered)
    assert min(rows) >= 1 and max(rows) <= h - 2
    assert max(rows) - min(rows) + 1 == len(ordered)
    dc = ordered[1][1] - ordered[0][1]
    assert dc in (-1, 1)
    for i in range(len(ordered) - 1):
        r0, c0 = ordered[i]
        r1, c1 = ordered[i + 1]
        assert r1 - r0 == 1
        assert c1 - c0 == dc
    index_map = {cell: i for i, cell in enumerate(ordered)}
    return {
        "color": int(color),
        "cells": ordered,
        "slope": int(dc),
        "index_map": index_map,
        "col_min": int(min(cols)),
        "col_max": int(max(cols)),
    }


def _parse_input(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    assert h <= 30 and w <= 30

    source_cols = sorted(int(c) for c in np.flatnonzero(g[0] == SOURCE))
    assert 1 <= len(source_cols) <= 2
    assert int(np.count_nonzero(g == SOURCE)) == len(source_cols)
    if len(source_cols) == 2:
        assert source_cols[1] - source_cols[0] >= 3

    source_mask = np.zeros_like(g, dtype=bool)
    for c in source_cols:
        source_mask[0, c] = True

    raw_components = _collect_rail_components(g)
    assert 2 <= len(raw_components) <= 4

    rails: list[dict] = []
    rail_by_cell: dict[tuple[int, int], int] = {}
    for color, cells in raw_components:
        rail = _analyze_rail(color, cells, g.shape)
        rid = len(rails)
        for cell in rail["cells"]:
            assert cell not in rail_by_cell
            rail_by_cell[cell] = rid
        rails.append(rail)

    for (r, c), rid in rail_by_cell.items():
        assert not (r == 1 and c in source_cols), "rail directly below a source would violate source preservation"
        for nr in range(r - 1, r + 2):
            for nc in range(c - 1, c + 2):
                other = rail_by_cell.get((nr, nc))
                if other is not None and other != rid:
                    raise AssertionError("rails must not touch")

    assert any(
        rail["col_min"] <= sc <= rail["col_max"]
        for rail in rails
        for sc in source_cols
    )

    return {
        "grid": g,
        "h": h,
        "w": w,
        "sources": source_cols,
        "source_mask": source_mask,
        "rails": rails,
        "rail_by_cell": rail_by_cell,
    }


def _simulate(
    grid: np.ndarray,
    *,
    recolor: bool = True,
    extra_step: bool = True,
    stop_on_occupied: bool = True,
    parsed: dict | None = None,
    return_meta: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict]:
    info = _parse_input(grid) if parsed is None else parsed
    g = info["grid"]
    h = info["h"]
    w = info["w"]
    sources = info["sources"]
    source_mask = info["source_mask"]
    rails = info["rails"]
    rail_by_cell = info["rail_by_cell"]

    out = g.copy()
    source_meta: list[dict] = []
    used_rails: set[int] = set()

    for source_col in sources:
        r = 0
        c = int(source_col)
        current_color = SOURCE
        collisions: list[int] = []
        collision_colors: list[int] = []
        stop_reason = "bottom"

        while True:
            if r == h - 1:
                if out[r, c] == BG:
                    out[r, c] = int(current_color)
                stop_reason = "bottom"
                break

            nr = r + 1
            rail_id = rail_by_cell.get((nr, c))
            if rail_id is not None:
                rail = rails[rail_id]
                used_rails.add(int(rail_id))
                if recolor:
                    current_color = int(rail["color"])
                if not source_mask[r, c]:
                    out[r, c] = int(current_color)

                hit_idx = rail["index_map"][(nr, c)]
                collisions.append(int(rail_id))
                collision_colors.append(int(rail["color"]))

                last_r = r
                last_c = c
                for rr, cc in rail["cells"][hit_idx:]:
                    last_r = int(rr)
                    last_c = int(cc)
                    out[last_r, last_c] = int(current_color)

                if extra_step:
                    exit_r = last_r + 1
                    exit_c = last_c + int(rail["slope"])
                    if not (0 <= exit_r < h and 0 <= exit_c < w):
                        stop_reason = "edge"
                        r, c = last_r, last_c
                        break
                    if stop_on_occupied and out[exit_r, exit_c] != BG:
                        stop_reason = "occupied"
                        r, c = last_r, last_c
                        break
                    out[exit_r, exit_c] = int(current_color)
                    r, c = int(exit_r), int(exit_c)
                    continue

                r, c = int(last_r), int(last_c)
                continue

            if stop_on_occupied and out[nr, c] != BG:
                stop_reason = "occupied"
                break

            r = int(nr)
            out[r, c] = int(current_color)

        source_meta.append(
            {
                "col": int(source_col),
                "collisions": collisions,
                "collision_colors": collision_colors,
                "stop_reason": stop_reason,
                "final": (int(r), int(c)),
            }
        )

    if return_meta:
        return out, {"sources": source_meta, "used_rails": used_rails, "rails": rails}
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Simulate diagonal rails that recolor and redirect falling water streams.

    Input structure:
    - The grid has black background, one or two blue source pixels on the top row, and 2-4 isolated diagonal rail segments inside the grid.
    - Each rail is a monochromatic 45-degree line of length 3-6, does not touch the top or bottom edge, and uses a non-black, non-blue color.

    Rule:
    - Process the blue sources from left to right. Each source emits a vertical water stream that paints background cells in its current color.
    - When the stream is directly above a rail pixel from the original input, recolor that stopping cell to the rail color, then slide down along the rail's row-increasing diagonal direction, painting the traversed diagonal cells and one extra diagonal exit cell.
    - After leaving a rail, the stream resumes vertical falling in the new column with its latest color, and it stops before any already-painted non-background cell or at the bottom edge. Original source pixels stay blue and original rail pixels keep their rail color.

    Key insight:
    - The diagonal decoration is actually a fixed deflection device: geometry chooses the next column, while the rail color becomes the stream's new color for all later segments.
    """
    return _simulate(np.asarray(grid, dtype=int))


def _rail_cells(hit_r: int, hit_c: int, slope: int, length: int, hit_index: int) -> list[tuple[int, int]]:
    return [
        (int(hit_r + offset), int(hit_c + slope * offset))
        for offset in range(-hit_index, length - hit_index)
    ]


def _rail_conflicts(cells: list[tuple[int, int]], occupied: set[tuple[int, int]]) -> bool:
    cell_set = set(cells)
    for r, c in cells:
        if (r, c) in occupied:
            return True
        for nr in range(r - 1, r + 2):
            for nc in range(c - 1, c + 2):
                if (nr, nc) in occupied and (nr, nc) not in cell_set:
                    return True
    return False


def _choose_color(prev_color: int) -> int:
    options = [color for color in RAIL_COLORS if color != prev_color]
    return int(random.choice(options))


def _try_place_rail(
    *,
    h: int,
    w: int,
    current_row: int,
    current_col: int,
    occupied: set[tuple[int, int]],
    color: int,
    slope_options: list[int],
    rem_options: list[int],
    exit_col_range: tuple[int, int] | None = None,
    min_hit_row: int | None = None,
    max_hit_row: int | None = None,
) -> dict | None:
    lengths = list(range(3, 7))
    random.shuffle(lengths)
    for slope in slope_options:
        pair_order = []
        for length in lengths:
            random_rems = [rem for rem in rem_options if 1 <= rem <= length]
            random.shuffle(random_rems)
            for rem in random_rems:
                pair_order.append((length, rem))
        random.shuffle(pair_order)
        for length, rem in pair_order:
            hit_index = int(length - rem)
            exit_col = int(current_col + slope * rem)
            if not (0 <= exit_col < w):
                continue
            if exit_col_range is not None:
                lo, hi = exit_col_range
                if not (lo <= exit_col <= hi):
                    continue

            cells_cols = [int(current_col + slope * offset) for offset in range(-hit_index, length - hit_index)]
            if min(cells_cols) < 1 or max(cells_cols) > w - 2:
                continue

            row_min = max(int(current_row + 2), int(1 + hit_index))
            row_max = min(int(h - 2 - (length - 1 - hit_index)), int(h - 1 - rem))
            if min_hit_row is not None:
                row_min = max(row_min, int(min_hit_row))
            if max_hit_row is not None:
                row_max = min(row_max, int(max_hit_row))
            if row_min > row_max:
                continue

            hit_rows = list(range(row_min, row_max + 1))
            random.shuffle(hit_rows)
            for hit_r in hit_rows:
                if any((rr, current_col) in occupied for rr in range(current_row + 1, hit_r + 1)):
                    continue
                cells = _rail_cells(hit_r, current_col, slope, length, hit_index)
                if _rail_conflicts(cells, occupied):
                    continue
                exit_cell = (int(cells[-1][0] + 1), int(cells[-1][1] + slope))
                if not (0 <= exit_cell[0] < h and 0 <= exit_cell[1] < w):
                    continue
                if exit_cell in occupied:
                    continue
                return {
                    "color": int(color),
                    "cells": cells,
                    "slope": int(slope),
                    "exit": exit_cell,
                }
    return None


def _render_grid(h: int, w: int, source_cols: list[int], rails: list[dict]) -> np.ndarray:
    grid = np.zeros((h, w), dtype=int)
    for c in source_cols:
        grid[0, int(c)] = SOURCE
    for rail in rails:
        for r, c in rail["cells"]:
            grid[int(r), int(c)] = int(rail["color"])
    return grid


def _build_chain() -> np.ndarray | None:
    h = random.randint(15, 22)
    w = random.randint(15, 24)
    source_col = random.randint(4, w - 5)
    current_row = 0
    current_col = source_col
    prev_color = SOURCE
    occupied: set[tuple[int, int]] = set()
    rails: list[dict] = []
    n_collisions = random.choice([2, 3])
    first_slope = random.choice([-1, 1])

    for i in range(n_collisions):
        slope_pref = first_slope if i % 2 == 0 else -first_slope
        color = _choose_color(prev_color)
        rail = _try_place_rail(
            h=h,
            w=w,
            current_row=current_row,
            current_col=current_col,
            occupied=occupied,
            color=color,
            slope_options=[slope_pref, -slope_pref],
            rem_options=[2, 3, 4],
        )
        if rail is None:
            return None
        rails.append(rail)
        occupied.update(rail["cells"])
        current_row, current_col = rail["exit"]
        prev_color = color

    return _render_grid(h, w, [source_col], rails)


def _build_parallel() -> np.ndarray | None:
    h = random.randint(16, 22)
    w = random.randint(18, 24)
    mid = w // 2
    left_source = random.randint(4, mid - 4)
    right_source = random.randint(mid + 3, w - 5)
    if right_source - left_source < 5:
        return None

    occupied: set[tuple[int, int]] = set()
    rails: list[dict] = []

    plans = [
        {
            "source_col": left_source,
            "n": random.choice([1, 2]),
            "first_slope": -1,
            "exit_range": (1, mid - 2),
        },
        {
            "source_col": right_source,
            "n": random.choice([1, 2]),
            "first_slope": 1,
            "exit_range": (mid + 1, w - 2),
        },
    ]

    for plan in plans:
        current_row = 0
        current_col = int(plan["source_col"])
        prev_color = SOURCE
        for i in range(int(plan["n"])):
            slope_pref = int(plan["first_slope"] if i % 2 == 0 else -plan["first_slope"])
            color = _choose_color(prev_color)
            rail = _try_place_rail(
                h=h,
                w=w,
                current_row=current_row,
                current_col=current_col,
                occupied=occupied,
                color=color,
                slope_options=[slope_pref, -slope_pref],
                rem_options=[2, 3, 4],
                exit_col_range=plan["exit_range"],
            )
            if rail is None:
                return None
            rails.append(rail)
            occupied.update(rail["cells"])
            current_row, current_col = rail["exit"]
            prev_color = color

    return _render_grid(h, w, [left_source, right_source], rails)


def _build_block() -> np.ndarray | None:
    h = random.randint(17, 23)
    w = random.randint(16, 23)
    left_source = random.randint(3, w - 9)
    distance = random.randint(3, 5)
    right_source = left_source + distance
    if right_source > w - 4:
        return None

    occupied: set[tuple[int, int]] = set()
    rails: list[dict] = []

    color1 = _choose_color(SOURCE)
    first_rail = _try_place_rail(
        h=h,
        w=w,
        current_row=0,
        current_col=left_source,
        occupied=occupied,
        color=color1,
        slope_options=[1],
        rem_options=[distance],
        min_hit_row=3,
        max_hit_row=7,
    )
    if first_rail is None:
        return None
    rails.append(first_rail)
    occupied.update(first_rail["cells"])

    current_row, current_col = first_rail["exit"]
    color2 = _choose_color(color1)
    second_rail = _try_place_rail(
        h=h,
        w=w,
        current_row=current_row + 1,
        current_col=current_col,
        occupied=occupied,
        color=color2,
        slope_options=[-1, 1],
        rem_options=[2, 3, 4],
    )
    if second_rail is None:
        return None
    rails.append(second_rail)
    occupied.update(second_rail["cells"])

    return _render_grid(h, w, [left_source, right_source], rails)


def _validate_generated(grid: np.ndarray, scenario: str) -> bool:
    try:
        parsed = _parse_input(grid)
        out, meta = _simulate(grid, parsed=parsed, return_meta=True)
    except AssertionError:
        return False

    if np.array_equal(out, grid):
        return False
    if int(np.count_nonzero(out != grid)) < 8:
        return False
    if int(np.count_nonzero(out != grid)) > (grid.shape[0] * grid.shape[1]) // 2:
        return False

    sources = meta["sources"]
    all_collision_count = sum(len(src["collisions"]) for src in sources)
    if all_collision_count == 0:
        return False

    used_slopes = {meta["rails"][rid]["slope"] for rid in meta["used_rails"]}
    if len(meta["used_rails"]) >= 2 and used_slopes != {-1, 1}:
        return False

    wrong_no_recolor = _simulate(grid, parsed=parsed, recolor=False)
    wrong_no_exit = _simulate(grid, parsed=parsed, extra_step=False)
    if np.array_equal(out, wrong_no_recolor):
        return False
    if np.array_equal(out, wrong_no_exit):
        return False

    if scenario == "chain":
        if len(sources) != 1:
            return False
        if len(sources[0]["collisions"]) < 2:
            return False
        if len(set(sources[0]["collision_colors"])) < 2:
            return False

    elif scenario == "parallel":
        if len(sources) != 2:
            return False
        if any(len(src["collisions"]) < 1 for src in sources):
            return False
        if any(src["stop_reason"] == "occupied" for src in sources):
            return False

    elif scenario == "block":
        if len(sources) != 2:
            return False
        if len(sources[0]["collisions"]) < 2:
            return False
        if sources[1]["stop_reason"] != "occupied":
            return False
        if len(sources[1]["collisions"]) != 0:
            return False
        if sources[1]["final"][0] < 2:
            return False
        wrong_ignore_overlap = _simulate(grid, parsed=parsed, stop_on_occupied=False)
        if np.array_equal(out, wrong_ignore_overlap):
            return False

    else:
        return False

    return True


def generate_input() -> np.ndarray:
    builders = {
        "chain": _build_chain,
        "parallel": _build_parallel,
        "block": _build_block,
    }
    scenario_weights = [0.35, 0.35, 0.30]
    scenarios = list(builders.keys())

    for _ in range(GEN_ATTEMPTS):
        scenario = random.choices(scenarios, weights=scenario_weights, k=1)[0]
        grid = builders[scenario]()
        if grid is None:
            continue
        if _validate_generated(grid, scenario):
            return grid

    raise RuntimeError("failed to generate a valid diagonal-rail cascade puzzle")
