from __future__ import annotations

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
ORANGE = 7

SHORT_MAX = 3
MEDIUM_MAX = 6
MAX_SEED_DIVIDER_GAP = 10

GRID_MIN = 17
GRID_MAX = 24
SIDE_COUNT_MIN = 2
SIDE_COUNT_MAX = 4
GEN_ATTEMPTS = 800


def _length_color(length: int) -> int:
    assert length >= 1, "every valid seed must produce a non-empty ray"
    if length <= SHORT_MAX:
        return GREEN
    if length <= MEDIUM_MAX:
        return YELLOW
    return ORANGE


def _length_band(length: int) -> str:
    if length <= SHORT_MAX:
        return "short"
    if length <= MEDIUM_MAX:
        return "medium"
    return "long"


def _divider_info(grid: np.ndarray) -> tuple[str, int]:
    grey_coords = np.argwhere(grid == GREY)
    assert len(grey_coords) > 0, "expected a grey divider"
    grey_rows = np.unique(grey_coords[:, 0])
    grey_cols = np.unique(grey_coords[:, 1])

    if len(grey_rows) == 1:
        divider = int(grey_rows[0])
        assert len(grey_coords) == grid.shape[1], "horizontal divider must span the full width"
        assert np.all(grid[divider, :] == GREY), "horizontal divider must be complete"
        return "horizontal", divider

    if len(grey_cols) == 1:
        divider = int(grey_cols[0])
        assert len(grey_coords) == grid.shape[0], "vertical divider must span the full height"
        assert np.all(grid[:, divider] == GREY), "vertical divider must be complete"
        return "vertical", divider

    raise AssertionError("expected exactly one straight grey divider")


def _ray_cells(
    shape: tuple[int, int],
    orientation: str,
    divider: int,
    row: int,
    col: int,
    color: int,
) -> tuple[list[tuple[int, int]], int, int, str]:
    height, width = map(int, shape)

    if orientation == "horizontal":
        side = "above" if row < divider else "below"
        divider_gap = abs(row - divider) - 1
        if color == RED:
            dr = 1 if side == "above" else -1
            length = divider_gap
        else:
            dr = -1 if side == "above" else 1
            length = row if side == "above" else height - 1 - row
        cells = [(row + step * dr, col) for step in range(length + 1)]
        return cells, length, divider_gap, side

    side = "left" if col < divider else "right"
    divider_gap = abs(col - divider) - 1
    if color == RED:
        dc = 1 if side == "left" else -1
        length = divider_gap
    else:
        dc = -1 if side == "left" else 1
        length = col if side == "left" else width - 1 - col
    cells = [(row, col + step * dc) for step in range(length + 1)]
    return cells, length, divider_gap, side


def _analyze_instance(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert 1 <= g.shape[0] <= 30 and 1 <= g.shape[1] <= 30, "grid must satisfy ARC size limits"
    assert set(map(int, np.unique(g))) <= {BLACK, BLUE, RED, GREY}, "unexpected input colors"

    orientation, divider = _divider_info(g)
    seed_coords = sorted(map(tuple, np.argwhere((g == BLUE) | (g == RED)).tolist()))
    assert seed_coords, "expected at least one red or blue seed"

    for i, (r0, c0) in enumerate(seed_coords):
        for r1, c1 in seed_coords[i + 1 :]:
            assert max(abs(r0 - r1), abs(c0 - c1)) > 1, "seed pixels must stay isolated"

    if orientation == "horizontal":
        used_columns = [c for _r, c in seed_coords]
        assert len(set(used_columns)) == len(used_columns), "horizontal instances need unique seed columns"
    else:
        used_rows = [r for r, _c in seed_coords]
        assert len(set(used_rows)) == len(used_rows), "vertical instances need unique seed rows"

    side_counts: dict[str, int] = {}
    occupied: set[tuple[int, int]] = set()
    seeds: list[dict] = []

    for row, col in seed_coords:
        color = int(g[row, col])
        ray, length, divider_gap, side = _ray_cells(g.shape, orientation, divider, row, col, color)
        assert 1 <= divider_gap <= MAX_SEED_DIVIDER_GAP, "seed-to-divider distance must be in 1..10"
        assert length >= 1, "every seed must create a non-empty ray"

        for rr, cc in ray[1:]:
            assert g[rr, cc] == BLACK, "ray corridors must be empty in the input"
        for cell in ray:
            assert cell not in occupied, "rays must be non-overlapping"
            occupied.add(cell)

        side_counts[side] = side_counts.get(side, 0) + 1
        seeds.append(
            {
                "row": int(row),
                "col": int(col),
                "color": color,
                "side": side,
                "ray": ray,
                "length": int(length),
                "divider_gap": int(divider_gap),
                "band": _length_band(length),
                "paint_color": _length_color(length),
            }
        )

    if orientation == "horizontal":
        side_names = ("above", "below")
    else:
        side_names = ("left", "right")
    for side in side_names:
        assert SIDE_COUNT_MIN <= side_counts.get(side, 0) <= SIDE_COUNT_MAX, "each side needs 2-4 seeds"

    return {
        "grid": g,
        "orientation": orientation,
        "divider": int(divider),
        "seeds": seeds,
        "side_counts": side_counts,
    }


def _render_output(info: dict) -> np.ndarray:
    out = info["grid"].copy()
    for seed in info["seeds"]:
        for row, col in seed["ray"]:
            out[row, col] = int(seed["paint_color"])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor attraction and repulsion rays by distance band from a full grey divider.

    Input structure:
    - The grid has a black background and one complete grey row or column spanning the canvas.
    - Red and blue single-pixel seeds lie on both sides of the divider, separated so their rays do not overlap.
    - Horizontal dividers use distinct seed columns; vertical dividers use distinct seed rows.

    Rule:
    - Detect whether the grey divider is horizontal or vertical.
    - From each red seed, draw a ray perpendicular to the divider toward it, stopping one cell before the grey line.
    - From each blue seed, draw a ray perpendicular to the divider away from it, stopping at the outer grid edge.
    - Let L be the number of drawn ray cells beyond the seed itself; recolor the whole ray including the seed to green for L=1..3, yellow for L=4..6, and orange for L>=7.
    - Keep the grey divider unchanged and leave all other black cells unchanged.

    Key insight:
    - Seed color determines only whether the ray is attracted to or repelled from the divider, while the final visible color depends only on the ray length band.
    """
    return _render_output(_analyze_instance(grid))


def _side_color_list(count: int, rng: np.random.Generator) -> list[int]:
    colors = [RED, BLUE]
    while len(colors) < count:
        colors.append(int(rng.choice([RED, BLUE])))
    rng.shuffle(colors)
    return colors


def _candidate_axis_positions(side: str, divider: int, limit: int) -> np.ndarray:
    if side in {"above", "left"}:
        low = max(0, divider - MAX_SEED_DIVIDER_GAP - 1)
        high = divider - 1
        return np.arange(low, high, dtype=int)
    low = divider + 2
    high = min(limit, divider + MAX_SEED_DIVIDER_GAP + 2)
    return np.arange(low, high, dtype=int)


def _color_safe_axis_positions(
    orientation: str,
    side: str,
    divider: int,
    limit: int,
    color: int,
) -> np.ndarray:
    positions = _candidate_axis_positions(side, divider, limit)
    if color == RED:
        return positions
    if orientation == "horizontal":
        if side == "above":
            return positions[positions >= 1]
        return positions[positions <= limit - 2]
    if side == "left":
        return positions[positions >= 1]
    return positions[positions <= limit - 2]


def _build_candidate_grid(rng: np.random.Generator) -> np.ndarray | None:
    orientation = "horizontal" if rng.random() < 0.5 else "vertical"
    height = int(rng.integers(GRID_MIN, GRID_MAX + 1))
    width = int(rng.integers(GRID_MIN, GRID_MAX + 1))
    grid = np.zeros((height, width), dtype=int)

    if orientation == "horizontal":
        divider = int(rng.integers(5, height - 5))
        grid[divider, :] = GREY
        side_names = ("above", "below")
        orth_pool = np.arange(width, dtype=int)
    else:
        divider = int(rng.integers(5, width - 5))
        grid[:, divider] = GREY
        side_names = ("left", "right")
        orth_pool = np.arange(height, dtype=int)

    side_counts = {
        side_names[0]: int(rng.integers(SIDE_COUNT_MIN, SIDE_COUNT_MAX + 1)),
        side_names[1]: int(rng.integers(SIDE_COUNT_MIN, SIDE_COUNT_MAX + 1)),
    }
    total_seeds = side_counts[side_names[0]] + side_counts[side_names[1]]
    if total_seeds > len(orth_pool):
        return None

    chosen_orth = rng.choice(orth_pool, size=total_seeds, replace=False)
    cursor = 0

    for side in side_names:
        count = side_counts[side]
        orth_values = chosen_orth[cursor : cursor + count]
        cursor += count
        colors = _side_color_list(count, rng)
        for orth_value, color in zip(orth_values, colors):
            axis_positions = _color_safe_axis_positions(
                orientation,
                side,
                divider,
                height if orientation == "horizontal" else width,
                color,
            )
            if len(axis_positions) == 0:
                return None
            axis = int(rng.choice(axis_positions))
            if orientation == "horizontal":
                row, col = axis, int(orth_value)
            else:
                row, col = int(orth_value), axis
            if grid[row, col] != BLACK:
                return None
            grid[row, col] = int(color)

    return grid


def _validate_generated_instance(grid: np.ndarray) -> bool:
    try:
        info = _analyze_instance(grid)
    except AssertionError:
        return False
    seeds = info["seeds"]
    side_names = ("above", "below") if info["orientation"] == "horizontal" else ("left", "right")

    for side in side_names:
        side_colors = {seed["color"] for seed in seeds if seed["side"] == side}
        if side_colors != {BLUE, RED}:
            return False

    bands = {seed["band"] for seed in seeds}
    if bands != {"short", "medium", "long"}:
        return False

    by_color = {
        RED: {seed["band"] for seed in seeds if seed["color"] == RED},
        BLUE: {seed["band"] for seed in seeds if seed["color"] == BLUE},
    }
    if len(by_color[RED]) < 2 and len(by_color[BLUE]) < 2:
        return False

    shared_band = by_color[RED] & by_color[BLUE]
    if not shared_band:
        return False

    blue_counterexamples = [
        seed
        for seed in seeds
        if seed["color"] == BLUE and _length_band(seed["divider_gap"]) != seed["band"]
    ]
    if not blue_counterexamples:
        return False

    out = _render_output(info)
    if not np.array_equal(out, main(grid)):
        return False
    visible_output_colors = set(map(int, np.unique(out))) - {BLACK, GREY}
    if visible_output_colors != {GREEN, YELLOW, ORANGE}:
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        rng = np.random.default_rng()
        grid = _build_candidate_grid(rng)
        if grid is None:
            continue
        if _validate_generated_instance(grid):
            return grid
    raise RuntimeError("failed to generate a valid divider-ray recoloring instance")
