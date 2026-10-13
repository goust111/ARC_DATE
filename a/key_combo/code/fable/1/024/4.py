from __future__ import annotations

import numpy as np


# Lightweight helpers aligned with the shared arc_utils conventions.
BG = 0
PALETTE = tuple(range(1, 10))

GRID_H_RANGE = (10, 25)  # upper-exclusive
GRID_W_RANGE = (10, 25)  # upper-exclusive
COLOR_COUNT_RANGE = (2, 6)  # upper-exclusive
MAX_INTERSECTIONS = 10
MAX_ATTEMPTS = 500


def get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(values[np.argmax(counts)])


def colors_present(grid: np.ndarray, *, exclude: set[int] | None = None) -> list[int]:
    colors = set(map(int, np.unique(np.asarray(grid, dtype=int))))
    if exclude is not None:
        colors -= set(map(int, exclude))
    return sorted(colors)


def _is_border(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return r == 0 or c == 0 or r == h - 1 or c == w - 1


def _collect_vertical_pairs(grid: np.ndarray, bg: int) -> dict[int, list[int]]:
    h, w = grid.shape
    by_color: dict[int, list[int]] = {}
    for c in range(w):
        color = int(grid[0, c])
        if color != bg and int(grid[h - 1, c]) == color:
            by_color.setdefault(color, []).append(int(c))
    return {color: sorted(cols) for color, cols in by_color.items()}


def _collect_horizontal_pairs(grid: np.ndarray, bg: int) -> dict[int, list[int]]:
    h, w = grid.shape
    by_color: dict[int, list[int]] = {}
    for r in range(h):
        color = int(grid[r, 0])
        if color != bg and int(grid[r, w - 1]) == color:
            by_color.setdefault(color, []).append(int(r))
    return {color: sorted(rows) for color, rows in by_color.items()}


def _paint_intersections(
    out: np.ndarray,
    verticals: dict[int, list[int]],
    horizontals: dict[int, list[int]],
) -> np.ndarray:
    h, w = out.shape
    shared = sorted(set(verticals) & set(horizontals))
    for color in shared:
        for c in verticals[color]:
            for r in horizontals[color]:
                if 0 < r < h - 1 and 0 < c < w - 1:
                    out[int(r), int(c)] = int(color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Keep the boundary markers and reveal only the intersections of same-color implied border lines.

    Input structure:
    - A black-background grid has all foreground pixels on the outer border.
    - Matching same-color markers on the top and bottom border imply vertical lines; matching same-color markers on the left and right border imply horizontal lines.
    - Additional border pixels may fail to form any valid pair and act as distractors.

    Rule:
    - Detect every column whose top and bottom border cells share the same non-background color, and every row whose left and right border cells share the same non-background color.
    - Never draw the implied lines themselves.
    - Only when a vertical and a horizontal implied line have the same color, paint their crossing cell with that color; preserve all original border pixels.

    Key insight:
    - The border markers define invisible intermediate geometry, and the output depends only on color-matched crossings of those unseen row and column lines.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "input must be a 2D grid"

    bg = get_bg(g)
    out = g.copy()
    verticals = _collect_vertical_pairs(g, bg)
    horizontals = _collect_horizontal_pairs(g, bg)
    return _paint_intersections(out, verticals, horizontals)


def _sample_line_count(rng: np.random.Generator) -> int:
    return int(rng.choice(np.array([1, 1, 2, 2, 3], dtype=int)))


def _sample_axis_positions(length: int, count: int, rng: np.random.Generator) -> list[int]:
    if count == 0:
        return []
    candidates = np.arange(1, length - 1, dtype=int)
    if count > len(candidates):
        raise ValueError("not enough interior border slots")

    for min_gap in (2, 1, 0):
        for _ in range(80):
            picks = sorted(map(int, rng.choice(candidates, size=count, replace=False)))
            if all(b - a > min_gap for a, b in zip(picks, picks[1:])):
                return picks
    raise ValueError("failed to sample spaced positions")


def _side_cell(shape: tuple[int, int], side: str, pos: int) -> tuple[int, int]:
    h, w = shape
    if side == "top":
        return 0, int(pos)
    if side == "bottom":
        return h - 1, int(pos)
    if side == "left":
        return int(pos), 0
    if side == "right":
        return int(pos), w - 1
    raise ValueError(f"unknown side: {side}")


def _empty_side_map() -> dict[str, set[int]]:
    return {"top": set(), "bottom": set(), "left": set(), "right": set()}


def _place_extra_marks(
    grid: np.ndarray,
    specs: list[dict],
    side_maps: dict[int, dict[str, set[int]]],
    rng: np.random.Generator,
) -> bool:
    h, w = grid.shape
    side_lengths = {"top": w, "bottom": w, "left": h, "right": h}
    opposite = {"top": "bottom", "bottom": "top", "left": "right", "right": "left"}
    sides = np.array(["top", "bottom", "left", "right"], dtype=object)

    for spec in specs:
        color = int(spec["color"])
        for _ in range(int(spec["n_extras"])):
            placed = False
            for _ in range(120):
                side = str(rng.choice(sides))
                pos = int(rng.integers(1, side_lengths[side] - 1))
                if pos in side_maps[color][opposite[side]]:
                    continue
                r, c = _side_cell(grid.shape, side, pos)
                if int(grid[r, c]) != BG:
                    continue
                grid[r, c] = color
                side_maps[color][side].add(pos)
                placed = True
                break
            if not placed:
                return False
    return True


def _expected_from_spec(grid: np.ndarray, specs: list[dict]) -> np.ndarray:
    out = np.array(grid, copy=True)
    for spec in specs:
        color = int(spec["color"])
        for c in spec["vertical_cols"]:
            for r in spec["horizontal_rows"]:
                out[int(r), int(c)] = color
    return out


def _sample_specs(rng: np.random.Generator) -> list[dict] | None:
    n_colors = int(rng.integers(*COLOR_COUNT_RANGE))
    colors = list(map(int, rng.choice(np.array(PALETTE, dtype=int), size=n_colors, replace=False)))

    n_cross = int(rng.integers(1, n_colors))
    cross_idx = set(map(int, rng.choice(np.arange(n_colors), size=n_cross, replace=False)))

    specs: list[dict] = []
    total_extras = 0
    for idx, color in enumerate(colors):
        if idx in cross_idx:
            mode = "cross"
            n_vertical = _sample_line_count(rng)
            n_horizontal = _sample_line_count(rng)
        else:
            mode = str(rng.choice(np.array(["vert", "horiz", "noise"], dtype=object), p=[0.35, 0.35, 0.30]))
            n_vertical = _sample_line_count(rng) if mode == "vert" else 0
            n_horizontal = _sample_line_count(rng) if mode == "horiz" else 0

        if mode == "noise":
            n_extras = int(rng.integers(2, 5))
        else:
            n_extras = int(rng.choice(np.array([0, 1, 1, 2], dtype=int)))
        total_extras += n_extras

        specs.append(
            {
                "color": int(color),
                "mode": mode,
                "n_vertical": int(n_vertical),
                "n_horizontal": int(n_horizontal),
                "n_extras": int(n_extras),
                "vertical_cols": [],
                "horizontal_rows": [],
            }
        )

    if total_extras == 0:
        chosen = specs[int(rng.integers(0, len(specs)))]
        chosen["n_extras"] = 1

    n_intersections = sum(int(spec["n_vertical"]) * int(spec["n_horizontal"]) for spec in specs)
    if not (1 <= n_intersections <= MAX_INTERSECTIONS):
        return None
    return specs


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()
    specs = _sample_specs(rng)
    if specs is None:
        return None

    total_vertical = sum(int(spec["n_vertical"]) for spec in specs)
    total_horizontal = sum(int(spec["n_horizontal"]) for spec in specs)

    h_lo = max(GRID_H_RANGE[0], total_horizontal + 6)
    w_lo = max(GRID_W_RANGE[0], total_vertical + 6)
    if h_lo >= GRID_H_RANGE[1] or w_lo >= GRID_W_RANGE[1]:
        return None

    h = int(rng.integers(h_lo, GRID_H_RANGE[1]))
    w = int(rng.integers(w_lo, GRID_W_RANGE[1]))

    try:
        vertical_positions = _sample_axis_positions(w, total_vertical, rng)
        horizontal_positions = _sample_axis_positions(h, total_horizontal, rng)
    except ValueError:
        return None

    v_cursor = 0
    h_cursor = 0
    for spec in specs:
        nv = int(spec["n_vertical"])
        nh = int(spec["n_horizontal"])
        spec["vertical_cols"] = vertical_positions[v_cursor : v_cursor + nv]
        spec["horizontal_rows"] = horizontal_positions[h_cursor : h_cursor + nh]
        v_cursor += nv
        h_cursor += nh

    grid = np.full((h, w), BG, dtype=int)
    side_maps = {int(spec["color"]): _empty_side_map() for spec in specs}

    for spec in specs:
        color = int(spec["color"])
        for c in spec["vertical_cols"]:
            grid[0, int(c)] = color
            grid[h - 1, int(c)] = color
            side_maps[color]["top"].add(int(c))
            side_maps[color]["bottom"].add(int(c))
        for r in spec["horizontal_rows"]:
            grid[int(r), 0] = color
            grid[int(r), w - 1] = color
            side_maps[color]["left"].add(int(r))
            side_maps[color]["right"].add(int(r))

    if not _place_extra_marks(grid, specs, side_maps, rng):
        return None

    expected = _expected_from_spec(grid, specs)
    produced = main(grid)
    if not np.array_equal(produced, expected):
        return None

    bg = get_bg(grid)
    if bg != BG:
        return None
    if len(colors_present(grid, exclude={BG})) < 2:
        return None

    fg = np.argwhere(grid != BG)
    if fg.size == 0 or not all(_is_border(grid.shape, int(r), int(c)) for r, c in fg):
        return None

    added = int(np.count_nonzero((grid == BG) & (expected != BG)))
    planned = sum(len(spec["vertical_cols"]) * len(spec["horizontal_rows"]) for spec in specs)
    if added != planned or added <= 0:
        return None

    detected_verticals = _collect_vertical_pairs(grid, BG)
    detected_horizontals = _collect_horizontal_pairs(grid, BG)
    planned_verticals = {
        int(spec["color"]): tuple(map(int, spec["vertical_cols"]))
        for spec in specs
        if spec["vertical_cols"]
    }
    planned_horizontals = {
        int(spec["color"]): tuple(map(int, spec["horizontal_rows"]))
        for spec in specs
        if spec["horizontal_rows"]
    }
    if {k: tuple(v) for k, v in detected_verticals.items()} != planned_verticals:
        return None
    if {k: tuple(v) for k, v in detected_horizontals.items()} != planned_horizontals:
        return None

    if np.count_nonzero(expected != BG) > int(0.28 * h * w):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_ATTEMPTS} attempts")
