import random

import numpy as np


BG = 0
COLOR_POOL = tuple(range(1, 10))
GENERATION_ATTEMPTS = 500
GRID_HEIGHT_RANGE = (12, 28)
GRID_WIDTH_RANGE = (12, 28)
LAYOUT_OPTIONS = ((1, 1), (1, 2), (2, 1), (2, 2), (2, 3), (3, 2), (3, 3))
POSITION_MARGIN = 2
MIN_POSITION_GAP = 3


def _background_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _runs_1d(mask: np.ndarray) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for idx, value in enumerate(mask.tolist()):
        if value:
            if start is None:
                start = idx
        elif start is not None:
            runs.append((start, idx - 1))
            start = None
    if start is not None:
        runs.append((start, len(mask) - 1))
    return runs


def _strict_majority_color(values: np.ndarray) -> int | None:
    colors, counts = np.unique(values, return_counts=True)
    if len(colors) == 0:
        return None
    order = np.argsort(-counts)
    best_idx = int(order[0])
    if len(order) > 1 and int(counts[best_idx]) == int(counts[int(order[1])]):
        return None
    return int(colors[best_idx])


def _extract_paths(grid: np.ndarray) -> tuple[int, list[dict[str, int]], list[dict[str, int]]]:
    g = np.asarray(grid, dtype=int)
    bg = _background_color(g)
    non_bg = g != bg
    height, width = g.shape

    horizontals: list[dict[str, int]] = []
    for row in range(height):
        for c0, c1 in _runs_1d(non_bg[row]):
            length = c1 - c0 + 1
            if length < 2:
                continue
            segment = g[row, c0 : c1 + 1]
            color = _strict_majority_color(segment)
            if color is None:
                continue
            if int(np.count_nonzero(segment == color)) < 2:
                continue
            horizontals.append(
                {
                    "row": int(row),
                    "c0": int(c0),
                    "c1": int(c1),
                    "length": int(length),
                    "color": int(color),
                }
            )

    verticals: list[dict[str, int]] = []
    for col in range(width):
        for r0, r1 in _runs_1d(non_bg[:, col]):
            length = r1 - r0 + 1
            if length < 2:
                continue
            segment = g[r0 : r1 + 1, col]
            color = _strict_majority_color(segment)
            if color is None:
                continue
            if int(np.count_nonzero(segment == color)) < 2:
                continue
            verticals.append(
                {
                    "col": int(col),
                    "r0": int(r0),
                    "r1": int(r1),
                    "length": int(length),
                    "color": int(color),
                }
            )

    return bg, horizontals, verticals


def _path_maps(
    shape: tuple[int, int], horizontals: list[dict[str, int]], verticals: list[dict[str, int]]
) -> tuple[np.ndarray, np.ndarray]:
    h_map = np.full(shape, -1, dtype=int)
    v_map = np.full(shape, -1, dtype=int)

    for idx, path in enumerate(horizontals):
        h_map[path["row"], path["c0"] : path["c1"] + 1] = idx
    for idx, path in enumerate(verticals):
        v_map[path["r0"] : path["r1"] + 1, path["col"]] = idx

    return h_map, v_map


def _apply_rule(
    grid: np.ndarray, horizontals: list[dict[str, int]], verticals: list[dict[str, int]]
) -> tuple[np.ndarray, list[dict[str, int]]]:
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    h_map, v_map = _path_maps(g.shape, horizontals, verticals)
    intersections: list[dict[str, int]] = []

    rows, cols = np.nonzero((h_map >= 0) & (v_map >= 0))
    for row, col in zip(rows.tolist(), cols.tolist()):
        h_path = horizontals[int(h_map[row, col])]
        v_path = verticals[int(v_map[row, col])]
        h_len = int(h_path["length"])
        v_len = int(v_path["length"])
        if h_len == v_len:
            winner_color = int(g[row, col])
            winner = -1
        elif h_len > v_len:
            winner_color = int(h_path["color"])
            winner = 0
        else:
            winner_color = int(v_path["color"])
            winner = 1
        out[row, col] = winner_color
        intersections.append(
            {
                "row": int(row),
                "col": int(col),
                "winner": int(winner),
                "winner_color": int(winner_color),
                "h_len": h_len,
                "v_len": v_len,
            }
        )

    return out, intersections


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each crossing pixel to the color of the longer straight path that passes through it.

    Input structure:
    - The background is black.
    - Non-background pixels form single-cell-wide straight horizontal and vertical paths.
    - When two paths cross, the shared pixel shows one of the two path colors, so a straight
      path can contain a few foreign-colored pixels at its crossings.

    Rule:
    - Recover every complete horizontal path and every complete vertical path as maximal
      contiguous non-background runs in rows and columns.
    - Treat a pixel as a crossing when it belongs to one recovered horizontal path and one
      recovered vertical path.
    - Compare the total lengths of the two full paths through each crossing.
    - Recolor the crossing pixel to the color of the longer path; leave all non-crossing
      pixels unchanged.

    Key insight:
    The visible color at a crossing is not reliable locally: the solver must first recover
    the full straight paths and compare their global lengths before deciding that one pixel.
    """
    g = np.asarray(grid, dtype=int)
    _bg, horizontals, verticals = _extract_paths(g)
    out, _intersections = _apply_rule(g, horizontals, verticals)
    return out


def _sample_spaced_positions(
    limit: int, count: int, rng: random.Random, *, margin: int = POSITION_MARGIN, gap: int = MIN_POSITION_GAP
) -> list[int] | None:
    candidates = list(range(margin, limit - margin))
    if len(candidates) < count:
        return None
    for _ in range(200):
        picks = sorted(rng.sample(candidates, count))
        if all(b - a >= gap for a, b in zip(picks, picks[1:])):
            return picks
    return None


def _draw_scene(
    shape: tuple[int, int],
    horizontals: list[dict[str, int]],
    verticals: list[dict[str, int]],
    visible_colors: dict[tuple[int, int], int],
) -> np.ndarray:
    grid = np.full(shape, BG, dtype=int)
    for path in horizontals:
        grid[path["row"], path["c0"] : path["c1"] + 1] = path["color"]
    for path in verticals:
        grid[path["r0"] : path["r1"] + 1, path["col"]] = path["color"]
    for (row, col), color in visible_colors.items():
        grid[row, col] = int(color)
    return grid


def _expected_output(
    grid: np.ndarray, horizontals: list[dict[str, int]], verticals: list[dict[str, int]]
) -> tuple[np.ndarray, list[dict[str, int]]]:
    return _apply_rule(grid, horizontals, verticals)


def _path_signature(path: dict[str, int]) -> tuple[int, ...]:
    if "row" in path:
        return (path["row"], path["c0"], path["c1"], path["length"], path["color"])
    return (path["col"], path["r0"], path["r1"], path["length"], path["color"])


def _background_is_dominant(grid: np.ndarray) -> bool:
    values, counts = np.unique(grid, return_counts=True)
    top_idx = int(np.argmax(counts))
    return int(values[top_idx]) == BG


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    height = rng.randint(*GRID_HEIGHT_RANGE)
    width = rng.randint(*GRID_WIDTH_RANGE)
    n_h, n_v = rng.choice(LAYOUT_OPTIONS)

    rows = _sample_spaced_positions(height, n_h, rng)
    cols = _sample_spaced_positions(width, n_v, rng)
    if rows is None or cols is None:
        return None

    if n_h + n_v > len(COLOR_POOL):
        return None
    colors = rng.sample(COLOR_POOL, n_h + n_v)
    h_colors = colors[:n_h]
    v_colors = colors[n_h:]

    horizontals: list[dict[str, int]] = []
    for row, color in zip(rows, h_colors):
        left_choices = list(range(1, cols[0]))
        right_choices = list(range(cols[-1] + 1, width - 1))
        if not left_choices or not right_choices:
            return None
        c0 = rng.choice(left_choices)
        c1 = rng.choice(right_choices)
        horizontals.append(
            {
                "row": int(row),
                "c0": int(c0),
                "c1": int(c1),
                "length": int(c1 - c0 + 1),
                "color": int(color),
            }
        )

    verticals: list[dict[str, int]] = []
    for col, color in zip(cols, v_colors):
        top_choices = list(range(1, rows[0]))
        bottom_choices = list(range(rows[-1] + 1, height - 1))
        if not top_choices or not bottom_choices:
            return None
        r0 = rng.choice(top_choices)
        r1 = rng.choice(bottom_choices)
        verticals.append(
            {
                "col": int(col),
                "r0": int(r0),
                "r1": int(r1),
                "length": int(r1 - r0 + 1),
                "color": int(color),
            }
        )

    winners: list[int] = []
    visible_colors: dict[tuple[int, int], int] = {}
    expected_visible_change = False
    for h_path in horizontals:
        for v_path in verticals:
            h_len = int(h_path["length"])
            v_len = int(v_path["length"])
            if h_len == v_len:
                return None
            row = int(h_path["row"])
            col = int(v_path["col"])
            winner_color = int(h_path["color"]) if h_len > v_len else int(v_path["color"])
            loser_color = int(v_path["color"]) if h_len > v_len else int(h_path["color"])
            winners.append(0 if h_len > v_len else 1)
            if rng.random() < 0.5:
                visible_color = winner_color
            else:
                visible_color = loser_color
                expected_visible_change = True
            visible_colors[(row, col)] = int(visible_color)

    if not winners:
        return None
    if len(winners) > 1 and (all(w == 0 for w in winners) or all(w == 1 for w in winners)):
        return None

    if not expected_visible_change:
        row = int(horizontals[0]["row"])
        col = int(verticals[0]["col"])
        first_h = horizontals[0]
        first_v = verticals[0]
        if int(first_h["length"]) > int(first_v["length"]):
            visible_colors[(row, col)] = int(first_v["color"])
        else:
            visible_colors[(row, col)] = int(first_h["color"])

    grid = _draw_scene((height, width), horizontals, verticals, visible_colors)
    if not _background_is_dominant(grid):
        return None

    expected, intersections = _expected_output(grid, horizontals, verticals)
    if not intersections:
        return None
    if np.array_equal(grid, expected):
        return None

    _bg, parsed_h, parsed_v = _extract_paths(grid)
    if len(parsed_h) != len(horizontals) or len(parsed_v) != len(verticals):
        return None
    if {_path_signature(path) for path in parsed_h} != {_path_signature(path) for path in horizontals}:
        return None
    if {_path_signature(path) for path in parsed_v} != {_path_signature(path) for path in verticals}:
        return None
    if not np.array_equal(main(grid), expected):
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {GENERATION_ATTEMPTS} attempts")
