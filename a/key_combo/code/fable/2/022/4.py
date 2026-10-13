import random
from collections import Counter

import numpy as np


BLACK = 0
PINK = 6
TEAL = 8
DEFLECTORS = {PINK, TEAL}
TURN_BY_COLOR = {PINK: 1, TEAL: -1}
SAND_COLORS = [1, 2, 3, 4, 5, 7, 9]

HEIGHT_RANGE = (18, 26)
WIDTH_RANGE = (14, 24)
ROUTE_COUNT_CHOICES = (2, 3)
TURN_GAP_RANGE = (2, 4)
ROUTE_GAP_MIN = 2
SIDE_MARGIN_MIN = 2
ATTEMPT_LIMIT = 400


def _sand_positions(grid: np.ndarray) -> list[tuple[int, int, int]]:
    coords = np.argwhere(~np.isin(grid, [BLACK, PINK, TEAL]))
    return [(int(r), int(c), int(grid[r, c])) for r, c in coords]


def _simulate(
    grid: np.ndarray,
    *,
    direction_map: dict[int, int] | None = None,
    low_to_high: bool = True,
    ignore_deflectors: bool = False,
    turn_limit: int | None = None,
    strict: bool = False,
) -> tuple[np.ndarray, list[dict]] | None:
    direction_map = TURN_BY_COLOR if direction_map is None else direction_map
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2:
        if strict:
            raise AssertionError("grid must be 2D")
        return None

    h, w = g.shape
    sands = _sand_positions(g)
    if not sands:
        if strict:
            raise AssertionError("expected at least one sand grain")
        return None

    order = sorted(sands, key=lambda item: (-item[0], item[1]) if low_to_high else (item[0], item[1]))
    out = np.zeros_like(g)
    out[g == PINK] = PINK
    out[g == TEAL] = TEAL
    traces: list[dict] = []

    for start_r, start_c, color in order:
        r, c = start_r, start_c
        turns: list[int] = []
        turn_points: list[tuple[int, int]] = []

        while True:
            if not (0 <= r < h and 0 <= c < w):
                if strict:
                    raise AssertionError("grain moved out of bounds")
                return None

            if out[r, c] != BLACK:
                if strict:
                    raise AssertionError("grain occupies non-empty cell")
                return None

            if r == h - 1:
                break

            below = int(out[r + 1, c])
            if ignore_deflectors:
                if below not in (BLACK, PINK, TEAL):
                    break
                r += 1
                continue

            if below in direction_map:
                if turn_limit is not None and len(turns) >= turn_limit:
                    break
                dc = int(direction_map[below])
                nr = r + 1
                nc = c + dc
                if not (0 <= nc < w):
                    if strict:
                        raise AssertionError("redirected grain left the grid")
                    return None
                if out[nr, nc] != BLACK:
                    if strict:
                        raise AssertionError("redirected grain entered occupied cell")
                    return None
                turns.append(dc)
                turn_points.append((nr, c))
                r, c = nr, nc
                continue

            if below != BLACK:
                break

            r += 1

        out[r, c] = color
        traces.append(
            {
                "start": (start_r, start_c),
                "final": (r, c),
                "color": color,
                "turn_dirs": tuple(turns),
                "turn_points": tuple(turn_points),
            }
        )

    return out, traces


def main(grid: np.ndarray) -> np.ndarray:
    """Drop colored sand grains through pink/cyan deflector pixels until they settle on the ground or earlier grains.

    Input structure:
    - A black background contains several colored sand grains, with grain colors different from pink and cyan.
    - Pink and cyan single-pixel deflectors are fixed in place inside the canvas.
    - The bottom edge of the canvas acts as the ground.

    Rule:
    - Remove the input sand grains conceptually and simulate them one by one from lower starting rows to higher starting rows.
    - Each grain falls straight down in its current column; if the cell directly below is pink, the grain moves to the lower-right adjacent cell, and if it is cyan, the grain moves to the lower-left adjacent cell.
    - After every redirect the grain continues falling in the new column, possibly chaining through multiple deflectors; it stops when the cell below is the ground or an already settled grain.
    - Deflectors stay unchanged, and the output is the final settled sand arrangement together with those deflectors.

    Key insight:
    - The deflectors form a directional field, so a grain's landing column depends on the full encountered deflector chain, while earlier settled grains become new local ground for later grains.
    """
    result = _simulate(np.asarray(grid, dtype=int), strict=True)
    assert result is not None
    out, _ = result
    return out


def _route_positions(dirs: list[int]) -> list[int]:
    positions = [0]
    cur = 0
    for dc in dirs:
        cur += dc
        positions.append(cur)
    return positions


def _sample_partition(total: int, parts: int) -> list[int]:
    out = [0] * parts
    for _ in range(total):
        out[random.randrange(parts)] += 1
    return out


def _sample_special_dirs() -> list[int]:
    length = random.randint(2, 4)
    dirs = [random.choice([-1, 1]) for _ in range(length)]
    if len(set(dirs)) == 1:
        dirs[random.randrange(length)] *= -1
    return dirs


def _sample_regular_dirs() -> list[int]:
    length = random.randint(1, 3)
    dirs = [random.choice([-1, 1]) for _ in range(length)]
    if length >= 2 and random.random() < 0.35:
        dirs[random.randrange(length)] *= -1
    return dirs


def _sample_turn_rows(height: int, turn_count: int, first_row_min: int) -> list[int] | None:
    remaining_min = TURN_GAP_RANGE[0] * (turn_count - 1)
    max_first = min(10, height - 4 - remaining_min)
    if max_first < first_row_min:
        return None
    first = random.randint(first_row_min, max_first)
    rows = [first]
    cur = first
    for idx in range(turn_count - 1):
        turns_left = turn_count - idx - 2
        max_gap = min(TURN_GAP_RANGE[1], height - 4 - cur - TURN_GAP_RANGE[0] * turns_left)
        if max_gap < TURN_GAP_RANGE[0]:
            return None
        gap = random.randint(TURN_GAP_RANGE[0], max_gap)
        cur += gap
        rows.append(cur)
    return rows


def _build_routes() -> list[dict]:
    route_count = random.choice(ROUTE_COUNT_CHOICES)
    routes = [{"dirs": _sample_special_dirs(), "grain_count": random.randint(2, 3), "stack_route": True}]
    while len(routes) < route_count:
        routes.append(
            {
                "dirs": _sample_regular_dirs(),
                "grain_count": random.randint(1, 2),
                "stack_route": False,
            }
        )

    if all(sum(route["dirs"]) == 0 for route in routes):
        routes[-1]["dirs"] = [random.choice([-1, 1])]

    random.shuffle(routes)
    return routes


def _place_route_columns(routes: list[dict]) -> int | None:
    for route in routes:
        positions = _route_positions(route["dirs"])
        route["positions"] = positions
        route["min_pos"] = min(positions)
        route["max_pos"] = max(positions)
        route["span"] = route["max_pos"] - route["min_pos"] + 1

    min_width = sum(route["span"] for route in routes)
    min_width += SIDE_MARGIN_MIN * 2 + ROUTE_GAP_MIN * (len(routes) - 1)
    if min_width > WIDTH_RANGE[1]:
        return None

    width = random.randint(max(WIDTH_RANGE[0], min_width), WIDTH_RANGE[1])
    slack = width - min_width
    extras = _sample_partition(slack, len(routes) + 1)
    left_margin = SIDE_MARGIN_MIN + extras[0]
    gaps = [ROUTE_GAP_MIN + extras[idx + 1] for idx in range(len(routes) - 1)]

    cursor = left_margin
    for idx, route in enumerate(routes):
        corridor_left = cursor
        route["start_col"] = corridor_left - route["min_pos"]
        cursor = corridor_left + route["span"]
        if idx < len(routes) - 1:
            cursor += gaps[idx]
    return width


def _populate_rows_and_grains(routes: list[dict], height: int) -> bool:
    total_grains = sum(route["grain_count"] for route in routes)
    first_row_min = max(5, total_grains + 1)

    for route in routes:
        rows = _sample_turn_rows(height, len(route["dirs"]), first_row_min)
        if rows is None:
            return False
        route["turn_rows"] = rows

    top_limit = min(route["turn_rows"][0] for route in routes) - 2
    if top_limit + 1 < total_grains:
        return False

    available_rows = random.sample(range(top_limit + 1), total_grains)
    available_colors = random.sample(SAND_COLORS, total_grains)
    random.shuffle(available_rows)
    random.shuffle(available_colors)

    row_idx = 0
    color_idx = 0
    for route in routes:
        route_rows = sorted(available_rows[row_idx : row_idx + route["grain_count"]], reverse=True)
        route_colors = available_colors[color_idx : color_idx + route["grain_count"]]
        route["grain_rows"] = route_rows
        route["grain_colors"] = route_colors
        row_idx += route["grain_count"]
        color_idx += route["grain_count"]
    return True


def _render_grid(height: int, width: int, routes: list[dict]) -> np.ndarray | None:
    grid = np.zeros((height, width), dtype=int)

    for route in routes:
        col = route["start_col"]
        used = set()
        for row, dc in zip(route["turn_rows"], route["dirs"]):
            if (row, col) in used or grid[row, col] != BLACK:
                return None
            color = PINK if dc == 1 else TEAL
            grid[row, col] = color
            used.add((row, col))
            col += dc
        route["final_col"] = col

    for route in routes:
        for row, color in zip(route["grain_rows"], route["grain_colors"]):
            if grid[row, route["start_col"]] != BLACK:
                return None
            grid[row, route["start_col"]] = color

    return grid


def _semantic_ok(grid: np.ndarray) -> bool:
    result = _simulate(grid, strict=True)
    if result is None:
        return False
    out, traces = result

    sand_rows = [trace["start"][0] for trace in traces]
    if len(set(sand_rows)) != len(sand_rows):
        return False

    turn_dirs = [dc for trace in traces for dc in trace["turn_dirs"]]
    if not turn_dirs or 1 not in turn_dirs or -1 not in turn_dirs:
        return False
    if max(len(trace["turn_dirs"]) for trace in traces) < 2:
        return False
    if not any(len(set(trace["turn_dirs"])) == 2 for trace in traces if trace["turn_dirs"]):
        return False
    if not any(trace["start"][1] != trace["final"][1] for trace in traces):
        return False

    final_cols = [trace["final"][1] for trace in traces]
    counts = Counter(final_cols)
    if max(counts.values()) < 2:
        return False
    if len(counts) < 2:
        return False

    reverse_result = _simulate(grid, low_to_high=False)
    if reverse_result is None or np.array_equal(out, reverse_result[0]):
        return False

    straight_result = _simulate(grid, ignore_deflectors=True)
    if straight_result is not None and np.array_equal(out, straight_result[0]):
        return False

    first_turn_result = _simulate(grid, turn_limit=1)
    if first_turn_result is not None and np.array_equal(out, first_turn_result[0]):
        return False

    swapped_result = _simulate(grid, direction_map={PINK: -1, TEAL: 1})
    if swapped_result is not None and np.array_equal(out, swapped_result[0]):
        return False

    if np.array_equal(grid, out):
        return False

    return True


def _gen_attempt() -> np.ndarray | None:
    routes = _build_routes()
    width = _place_route_columns(routes)
    if width is None:
        return None

    height = random.randint(*HEIGHT_RANGE)
    if not _populate_rows_and_grains(routes, height):
        return None

    grid = _render_grid(height, width, routes)
    if grid is None:
        return None

    if not _semantic_ok(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(ATTEMPT_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid sand-deflector input")
