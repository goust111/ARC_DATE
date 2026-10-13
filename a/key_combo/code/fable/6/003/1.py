from __future__ import annotations

from collections import Counter
import random
from typing import Any

import numpy as np


GEN_ATTEMPTS = 500
PALETTE = list(range(10))


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components(mask: np.ndarray, connectivity: int = 4) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    if int(connectivity) == 8:
        steps = [
            (-1, -1),
            (-1, 0),
            (-1, 1),
            (0, -1),
            (0, 1),
            (1, -1),
            (1, 0),
            (1, 1),
        ]
    else:
        steps = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or not bool(mask[r, c]):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                cells.append((int(rr), int(cc)))
                for dr, dc in steps:
                    nr = int(rr + dr)
                    nc = int(cc + dc)
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and bool(mask[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _parse_basin_component(grid: np.ndarray, color: int, bg: int) -> dict[str, int] | None:
    comps = _components(grid == int(color), connectivity=4)
    if len(comps) != 1:
        return None
    coords = comps[0]
    r0, c0, r1, c1 = _bbox_coords(coords)
    if r1 - r0 + 1 < 3 or c1 - c0 + 1 < 3:
        return None

    cells = {(int(r), int(c)) for r, c in coords}
    if any((r1, c) not in cells for c in range(c0, c1 + 1)):
        return None
    if any((r, c0) not in cells for r in range(r0, r1 + 1)):
        return None
    if any((r, c1) not in cells for r in range(r0, r1 + 1)):
        return None

    for r in range(r0, r1):
        for c in range(c0 + 1, c1):
            if (r, c) in cells:
                return None
            if int(grid[r, c]) != int(bg):
                return None
    local = grid[r0 : r1 + 1, c0 : c1 + 1]
    if not np.all(local[-1, :] == int(color)):
        return None
    if not np.all(local[:, 0] == int(color)) or not np.all(local[:, -1] == int(color)):
        return None
    if np.any(local[:-1, 1:-1] != int(bg)):
        return None

    return {
        "top": int(r0),
        "left": int(c0),
        "bottom": int(r1),
        "right": int(c1),
    }


def _is_horizontal_segment_component(coords: np.ndarray) -> bool:
    rows = sorted({int(r) for r, _c in coords})
    cols = sorted(int(c) for _r, c in coords)
    if len(rows) != 1:
        return False
    return cols == list(range(cols[0], cols[-1] + 1))


def _color_row_runs(grid: np.ndarray, color: int) -> dict[int, list[tuple[int, int]]]:
    h, w = grid.shape
    runs: dict[int, list[tuple[int, int]]] = {}
    for r in range(h):
        c = 0
        while c < w:
            if int(grid[r, c]) != int(color):
                c += 1
                continue
            c0 = int(c)
            while c + 1 < w and int(grid[r, c + 1]) == int(color):
                c += 1
            runs.setdefault(int(r), []).append((int(c0), int(c)))
            c += 1
    return runs


def _parse_scene(grid: np.ndarray) -> dict[str, Any]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    bg = _get_bg(g)

    top_sources = [(0, int(c), int(g[0, c])) for c in np.where(g[0] != int(bg))[0]]
    assert len(top_sources) >= 2, "expected multiple top-row sources"
    source_cols = [c for _r, c, _color in top_sources]
    assert len(set(source_cols)) == len(source_cols), "source columns must be distinct"
    source_colors = [color for _r, _c, color in top_sources]
    assert len(set(source_colors)) == len(source_colors), "source colors must be distinct"

    for _r, c, color in top_sources:
        assert int(np.count_nonzero(g == int(color))) == 1, "each source color must appear once"
        assert int(g[0, c]) == int(color)

    non_bg = [int(c) for c in np.unique(g) if int(c) != int(bg)]
    remaining = [int(c) for c in non_bg if int(c) not in set(source_colors)]
    assert len(remaining) == 2, "expected shelf color and basin color besides the sources"

    basin_color = None
    basin = None
    for color in remaining:
        basin_info = _parse_basin_component(g, color, bg)
        if basin_info is not None:
            basin_color = int(color)
            basin = basin_info

    assert basin_color is not None and basin is not None, "failed to identify the basin color"
    shelf_candidates = [int(color) for color in remaining if int(color) != int(basin_color)]
    assert len(shelf_candidates) == 1, "failed to identify the shelf color"
    shelf_color = int(shelf_candidates[0])

    shelf_rows = _color_row_runs(g, int(shelf_color))
    assert shelf_rows, "expected at least one shelf"
    for row in shelf_rows:
        shelf_rows[row].sort()

    sources = [{"row": 0, "col": int(c), "color": int(color)} for _r, c, color in top_sources]
    sources.sort(key=lambda item: int(item["col"]))
    source_by_col = {int(item["col"]): int(item["color"]) for item in sources}

    return {
        "grid": g,
        "shape": (h, w),
        "bg": int(bg),
        "sources": sources,
        "source_by_col": source_by_col,
        "shelf_color": int(shelf_color),
        "shelf_rows": shelf_rows,
        "basin_color": int(basin_color),
        "basin": basin,
    }


def _dedupe_traces(traces: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[tuple[int, tuple[tuple[int, int], ...]]] = set()
    for trace in traces:
        key = (int(trace["source_color"]), tuple((int(r), int(c)) for r, c in trace["cells"]))
        if key in seen:
            continue
        seen.add(key)
        out.append({"source_color": int(trace["source_color"]), "cells": list(key[1])})
    return out


def _dedupe_forward_traces(traces: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[tuple[int, tuple[tuple[int, int], ...], bool]] = set()
    for trace in traces:
        key = (
            int(trace["source_color"]),
            tuple((int(r), int(c)) for r, c in trace["cells"]),
            bool(trace["reached_basin"]),
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(
            {
                "source_color": int(trace["source_color"]),
                "cells": list(key[1]),
                "reached_basin": bool(key[2]),
            }
        )
    return out


def _trace_to(scene: dict[str, Any], end_row: int, col: int, memo: dict[tuple[int, int], list[dict[str, Any]]]) -> list[dict[str, Any]]:
    key = (int(end_row), int(col))
    if key in memo:
        return memo[key]

    g = scene["grid"]
    h, w = scene["shape"]
    bg = int(scene["bg"])
    source_by_col = scene["source_by_col"]
    shelf_rows = scene["shelf_rows"]

    traces: list[dict[str, Any]] = []
    if not (0 <= int(col) < int(w) and 1 <= int(end_row) < int(h)):
        memo[key] = traces
        return traces
    if int(g[int(end_row), int(col)]) != int(bg):
        memo[key] = traces
        return traces

    source_color = source_by_col.get(int(col))
    if source_color is not None and np.all(g[1 : int(end_row) + 1, int(col)] == int(bg)):
        cells = [(0, int(col))]
        cells.extend((int(r), int(col)) for r in range(1, int(end_row) + 1))
        traces.append({"source_color": int(source_color), "cells": cells})

    for shelf_row, segs in shelf_rows.items():
        s = int(shelf_row)
        if s >= int(end_row) or s < 1:
            continue
        if not np.all(g[s : int(end_row) + 1, int(col)] == int(bg)):
            continue
        for a, b in segs:
            if int(col) == int(a - 1) or int(col) == int(b + 1):
                for impact_col in range(int(a), int(b) + 1):
                    lo = min(int(impact_col), int(col))
                    hi = max(int(impact_col), int(col))
                    if not np.all(g[s - 1, lo : hi + 1] == int(bg)):
                        continue
                    upper = _trace_to(scene, s - 1, int(impact_col), memo)
                    for prev in upper:
                        cells = list(prev["cells"])
                        step = 1 if int(col) > int(impact_col) else -1
                        for x in range(int(impact_col) + step, int(col) + step, step):
                            cells.append((int(s - 1), int(x)))
                        for r in range(int(s), int(end_row) + 1):
                            cells.append((int(r), int(col)))
                        traces.append({"source_color": int(prev["source_color"]), "cells": cells})

    memo[key] = _dedupe_traces(traces)
    return memo[key]


def _solve_unique_trace(scene: dict[str, Any]) -> dict[str, Any]:
    g = scene["grid"]
    basin = scene["basin"]
    bg = int(scene["bg"])

    all_traces: list[dict[str, Any]] = []
    memo: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for entry_col in range(int(basin["left"]) + 1, int(basin["right"])):
        if not np.all(
            g[int(basin["top"]) : int(basin["bottom"]), int(entry_col)] == int(bg)
        ):
            continue
        for prefix in _trace_to(scene, int(basin["top"]), int(entry_col), memo):
            cells = list(prefix["cells"])
            for r in range(int(basin["top"]) + 1, int(basin["bottom"])):
                cells.append((int(r), int(entry_col)))
            all_traces.append({"source_color": int(prefix["source_color"]), "cells": cells})

    all_traces = _dedupe_traces(all_traces)
    assert len(all_traces) == 1, "the basin must admit exactly one reverse-trace branch"
    return all_traces[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reverse-trace the only waterfall branch that actually feeds the bottom basin.

    Input structure:
    - The dominant background contains several distinct single-cell sources on the
      top row, one shared shelf color forming horizontal bars, and one unique
      basin color forming an upward-opening U shape near the bottom.
    - Every shelf is a 1-cell-thick horizontal segment, and the basin interior is
      empty background.

    Rule:
    - Imagine each source falling straight down; when a stream lands on a shelf it
      spreads along the shelf top and can overflow one cell past either end, creating
      two child branches.
    - Exactly one root-to-leaf branch from the full cascade enters the basin interior
      and then falls to the basin floor.
    - Keep the scene unchanged except for painting only the background cells visited
      by that unique basin-reaching branch in the winning source color.

    Key insight:
    - Rather than simulating every source forward, solve the puzzle by starting from
      the basin mouth and uniquely reversing each overflow decision back through the
      ancestor shelves to its source.
    """
    scene = _parse_scene(grid)
    trace = _solve_unique_trace(scene)

    out = scene["grid"].copy()
    color = int(trace["source_color"])
    for r, c in trace["cells"]:
        out[int(r), int(c)] = int(color)
    return out


def _find_shelf_segment(shelf_rows: dict[int, list[tuple[int, int]]], row: int, col: int) -> tuple[int, int] | None:
    for a, b in shelf_rows.get(int(row), []):
        if int(a) <= int(col) <= int(b):
            return (int(a), int(b))
    return None


def _enumerate_forward_paths(scene: dict[str, Any]) -> list[dict[str, Any]]:
    g = scene["grid"]
    h, _w = scene["shape"]
    bg = int(scene["bg"])
    shelf_rows = scene["shelf_rows"]
    basin = scene["basin"]

    out: list[dict[str, Any]] = []

    def dfs(row: int, col: int, source_color: int, cells: list[tuple[int, int]]) -> None:
        rr = int(row)
        cc = int(col)
        path = list(cells)

        while True:
            if rr + 1 >= int(h):
                out.append(
                    {
                        "source_color": int(source_color),
                        "cells": list(path),
                        "reached_basin": False,
                    }
                )
                return

            below = int(g[rr + 1, cc])
            if below == int(bg):
                rr += 1
                path.append((int(rr), int(cc)))
                if (
                    int(basin["top"]) <= int(rr) < int(basin["bottom"])
                    and int(basin["left"]) < int(cc) < int(basin["right"])
                    and int(rr) == int(basin["bottom"]) - 1
                ):
                    out.append(
                        {
                            "source_color": int(source_color),
                            "cells": list(path),
                            "reached_basin": True,
                        }
                    )
                    return
                continue

            seg = _find_shelf_segment(shelf_rows, rr + 1, cc)
            if seg is None:
                out.append(
                    {
                        "source_color": int(source_color),
                        "cells": list(path),
                        "reached_basin": False,
                    }
                )
                return

            a, b = seg
            for overflow in (int(a - 1), int(b + 1)):
                branch = list(path)
                step = 1 if int(overflow) > int(cc) else -1
                for x in range(int(cc) + step, int(overflow) + step, step):
                    branch.append((int(rr), int(x)))
                dfs(int(rr), int(overflow), int(source_color), branch)
            return

    for source in scene["sources"]:
        dfs(int(source["row"]), int(source["col"]), int(source["color"]), [(0, int(source["col"]))])
    return _dedupe_forward_traces(out)


def _output_from_trace(scene: dict[str, Any], trace: dict[str, Any]) -> np.ndarray:
    out = scene["grid"].copy()
    color = int(trace["source_color"])
    for r, c in trace["cells"]:
        out[int(r), int(c)] = int(color)
    return out


def _zone_cols(left: int, right: int) -> list[int]:
    if int(left) > int(right):
        return []
    return list(range(int(left), int(right) + 1))


def _draw_basin(grid: np.ndarray, basin: dict[str, int], color: int) -> None:
    grid[int(basin["top"]) : int(basin["bottom"]) + 1, int(basin["left"])] = int(color)
    grid[int(basin["top"]) : int(basin["bottom"]) + 1, int(basin["right"])] = int(color)
    grid[int(basin["bottom"]), int(basin["left"]) : int(basin["right"]) + 1] = int(color)


def _add_shelf(
    grid: np.ndarray,
    bg: int,
    shelf_color: int,
    used_rows: set[int],
    row: int,
    a: int,
    b: int,
) -> bool:
    if int(row) in used_rows or int(row) <= 0 or int(a) < 1 or int(b) >= grid.shape[1] - 1 or int(a) >= int(b):
        return False
    if not np.all(grid[int(row), int(a) : int(b) + 1] == int(bg)):
        return False
    if int(row) - 1 > 0 and np.any(grid[int(row) - 1, int(a) : int(b) + 1] == int(shelf_color)):
        return False
    if int(row) + 1 < grid.shape[0] and np.any(grid[int(row) + 1, int(a) : int(b) + 1] == int(shelf_color)):
        return False
    grid[int(row), int(a) : int(b) + 1] = int(shelf_color)
    used_rows.add(int(row))
    return True


def _build_target_chain(
    rng: random.Random,
    h: int,
    w: int,
    basin: dict[str, int],
) -> dict[str, Any] | None:
    target_col = rng.randint(int(basin["left"]) + 1, int(basin["right"]) - 1)
    depth = rng.randint(2, 4)

    end_row = int(basin["top"])
    down_col = int(target_col)
    shelves: list[dict[str, int | str]] = []

    for _ in range(depth):
        gap = rng.randint(3, 5)
        shelf_row = int(end_row - gap)
        if shelf_row < 3:
            return None

        opts: list[tuple[str, int, int, int]] = []
        if int(down_col) <= int(w) - 4:
            a = int(down_col + 1)
            max_b = min(int(w - 2), int(a + 3))
            for b in range(int(a + 1), int(max_b) + 1):
                for up_col in range(int(a), int(b) + 1):
                    opts.append(("left", int(a), int(b), int(up_col)))
        if int(down_col) >= 3:
            b = int(down_col - 1)
            min_a = max(1, int(b - 3))
            for a in range(int(min_a), int(b)):
                for up_col in range(int(a), int(b) + 1):
                    opts.append(("right", int(a), int(b), int(up_col)))
        if not opts:
            return None

        side, a, b, up_col = rng.choice(opts)
        sibling_col = int(b + 1) if side == "left" else int(a - 1)
        shelves.append(
            {
                "row": int(shelf_row),
                "a": int(a),
                "b": int(b),
                "up_col": int(up_col),
                "down_col": int(down_col),
                "sibling_col": int(sibling_col),
                "side": side,
            }
        )
        end_row = int(shelf_row - 1)
        down_col = int(up_col)

    if down_col <= int(basin["left"]) - 3 or down_col >= int(basin["right"]) + 3:
        return None
    return {
        "source_col": int(down_col),
        "target_col": int(target_col),
        "shelves": list(reversed(shelves)),
    }


def _draw_target_chain(
    grid: np.ndarray,
    bg: int,
    shelf_color: int,
    used_rows: set[int],
    plan: dict[str, Any],
) -> bool:
    for shelf in plan["shelves"]:
        if not _add_shelf(
            grid,
            int(bg),
            int(shelf_color),
            used_rows,
            int(shelf["row"]),
            int(shelf["a"]),
            int(shelf["b"]),
        ):
            return False
    return True


def _build_decoy_chain(
    rng: random.Random,
    grid: np.ndarray,
    bg: int,
    shelf_color: int,
    used_rows: set[int],
    source_col: int,
    zone_left: int,
    zone_right: int,
    basin: dict[str, int],
    min_depth: int,
    max_depth: int,
) -> bool:
    depth = rng.randint(int(min_depth), int(max_depth))
    row_cursor = 0
    col_cursor = int(source_col)
    placed = 0

    for _ in range(depth):
        row_choices = [
            r
            for r in range(int(row_cursor) + 3, int(basin["top"]) - 1)
            if int(r) not in used_rows
        ]
        if not row_choices:
            break
        shelf_row = rng.choice(row_choices)

        attempts: list[tuple[int, int, int]] = []
        for a in range(max(int(zone_left) + 1, int(col_cursor) - 3), int(col_cursor) + 1):
            for b in range(int(col_cursor), min(int(zone_right) - 1, int(col_cursor) + 3) + 1):
                if int(a) >= int(b):
                    continue
                if not (int(a) <= int(col_cursor) <= int(b)):
                    continue
                left_over = int(a - 1)
                right_over = int(b + 1)
                if int(basin["left"]) < int(left_over) < int(basin["right"]):
                    continue
                if int(basin["left"]) < int(right_over) < int(basin["right"]):
                    continue
                next_col = rng.choice([left_over, right_over])
                attempts.append((int(a), int(b), int(next_col)))
        if not attempts:
            break
        a, b, next_col = rng.choice(attempts)
        if not _add_shelf(grid, int(bg), int(shelf_color), used_rows, int(shelf_row), int(a), int(b)):
            continue
        row_cursor = int(shelf_row - 1)
        col_cursor = int(next_col)
        placed += 1

    return placed >= int(min_depth)


def _pick_palette(rng: random.Random, source_count: int) -> dict[str, Any]:
    colors = list(PALETTE)
    rng.shuffle(colors)
    bg = int(colors.pop())
    shelf = int(colors.pop())
    basin = int(colors.pop())
    sources = [int(colors.pop()) for _ in range(source_count)]
    return {"bg": bg, "shelf": shelf, "basin": basin, "sources": sources}


def _generate_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(20, 28)
    w = rng.randint(20, 28)

    basin_outer_w = rng.randint(5, 7)
    basin_h = rng.randint(4, 6)
    basin_left_low = 5
    basin_left_high = w - basin_outer_w - 6
    if basin_left_low > basin_left_high:
        return None
    basin_left = rng.randint(int(basin_left_low), int(basin_left_high))
    basin = {
        "left": int(basin_left),
        "right": int(basin_left + basin_outer_w - 1),
        "bottom": int(h - rng.randint(1, 2)),
    }
    basin["top"] = int(basin["bottom"] - basin_h + 1)
    if int(basin["top"]) < 10:
        return None

    left_zone = _zone_cols(1, int(basin["left"]) - 4)
    right_zone = _zone_cols(int(basin["right"]) + 4, int(w - 2))
    available_zones = [zone for zone in (left_zone, right_zone) if len(zone) >= 4]
    if not available_zones:
        return None

    extra_source_count = rng.randint(1, min(3, len(left_zone) // 4 + len(right_zone) // 4))
    source_count = 1 + extra_source_count
    palette = _pick_palette(rng, source_count)

    grid = np.full((h, w), int(palette["bg"]), dtype=int)
    _draw_basin(grid, basin, int(palette["basin"]))

    plan = _build_target_chain(rng, h, w, basin)
    if plan is None:
        return None

    used_rows: set[int] = set()
    if not _draw_target_chain(grid, int(palette["bg"]), int(palette["shelf"]), used_rows, plan):
        return None

    target_source_col = int(plan["source_col"])
    if int(grid[0, target_source_col]) != int(palette["bg"]):
        return None
    grid[0, target_source_col] = int(palette["sources"][0])

    candidate_source_cols: list[tuple[str, int]] = []
    for side, zone in (("left", left_zone), ("right", right_zone)):
        candidate_cols = [c for c in zone if abs(int(c) - int(target_source_col)) >= 4]
        rng.shuffle(candidate_cols)
        for c in candidate_cols:
            candidate_source_cols.append((side, int(c)))
    rng.shuffle(candidate_source_cols)

    if len(candidate_source_cols) < extra_source_count:
        return None

    decoy_shelf_seen = False
    for idx in range(extra_source_count):
        side, source_col = candidate_source_cols[idx]
        color = int(palette["sources"][idx + 1])
        if int(grid[0, source_col]) != int(palette["bg"]):
            return None
        grid[0, source_col] = int(color)
        zone = left_zone if side == "left" else right_zone
        min_depth = 1 if idx == 0 else 0
        max_depth = 2 if idx == 0 else 1
        ok = _build_decoy_chain(
            rng,
            grid,
            int(palette["bg"]),
            int(palette["shelf"]),
            used_rows,
            int(source_col),
            int(zone[0]),
            int(zone[-1]),
            basin,
            min_depth=min_depth,
            max_depth=max_depth,
        )
        if idx == 0 and not ok:
            return None
        decoy_shelf_seen = decoy_shelf_seen or ok

    if not decoy_shelf_seen:
        return None

    scene = _parse_scene(grid)
    forward = _enumerate_forward_paths(scene)
    basin_paths = [trace for trace in forward if bool(trace.get("reached_basin"))]
    if len(basin_paths) != 1:
        return None

    target_trace = _solve_unique_trace(scene)
    expected = _output_from_trace(scene, basin_paths[0])
    if not np.array_equal(_output_from_trace(scene, target_trace), expected):
        return None
    if not np.array_equal(main(grid), expected):
        return None

    basin_source_color = int(basin_paths[0]["source_color"])
    if basin_source_color != int(palette["sources"][0]):
        return None

    if len(scene["shelf_rows"]) < 3:
        return None
    if len(scene["sources"]) < 2:
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _generate_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid reverse-waterfall basin puzzle")
