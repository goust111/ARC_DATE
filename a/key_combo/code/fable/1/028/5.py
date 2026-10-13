from __future__ import annotations

import random

import numpy as np


# Follow the shared arc_utils conventions: grids use grid[r, c] and coords are (r, c).
_BG = 0
_PALETTE = tuple(range(1, 10))

# Generator bounds.
_GEN_HEIGHT_RANGE = (9, 16)
_GEN_WALL_COUNT_CHOICES = (3, 4, 5)
_GEN_WALL_COUNT_WEIGHTS = (4, 4, 2)
_GEN_LEFT_GAP_RANGE = (1, 4)
_GEN_BETWEEN_GAP_RANGE = (2, 4)
_GEN_RIGHT_GAP_RANGE = (2, 4)
_GEN_DOOR_COUNT_CHOICES = (3, 4)
_GEN_MAX_ATTEMPTS = 300


def _dominant_color(values: np.ndarray) -> int:
    colors, counts = np.unique(values, return_counts=True)
    max_count = int(counts.max())
    winners = colors[counts == max_count]
    assert len(winners) == 1, "expected a unique dominant color"
    return int(winners[0])


def _paint_horizontal(out: np.ndarray, row: int, c0: int, c1: int, color: int) -> None:
    if c0 > c1:
        return
    out[int(row), int(c0) : int(c1) + 1] = int(color)


def _paint_vertical(out: np.ndarray, col: int, r0: int, r1: int, color: int) -> None:
    lo, hi = sorted((int(r0), int(r1)))
    out[lo : hi + 1, int(col)] = int(color)


def _extract_structure(grid: np.ndarray) -> tuple[np.ndarray, int, dict, list[dict]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    H, W = g.shape

    bg = _dominant_color(g)

    particle_rows = np.flatnonzero(g[:, 0] != bg)
    assert particle_rows.size == 1, "expected exactly one left-edge particle"
    particle_row = int(particle_rows[0])
    assert np.count_nonzero(g[:, 0] != bg) == 1, "left edge should contain only the particle"
    particle = {"row": particle_row, "col": 0, "color": int(g[particle_row, 0])}

    wall_cols = [c for c in range(1, W) if np.all(g[:, c] != bg)]
    assert wall_cols, "expected at least one full-column wall"
    assert all(wall_cols[i + 1] - wall_cols[i] >= 2 for i in range(len(wall_cols) - 1)), "wall columns must be non-adjacent"

    allowed_cols = {0, *wall_cols}
    for r, c in np.argwhere(g != bg):
        assert int(c) in allowed_cols, "unexpected foreground outside particle and wall columns"

    walls = []
    for c in wall_cols:
        column = g[:, c]
        wall_color = _dominant_color(column)
        door_rows = [int(r) for r in range(H) if int(column[r]) != wall_color]
        door_colors = [int(column[r]) for r in door_rows]
        assert len(door_rows) >= 2, "each wall should contain several doors"
        assert len(set(door_colors)) == len(door_colors), "door colors must be unique within a wall"
        assert all(color != wall_color for color in door_colors), "door color must differ from wall color"
        walls.append(
            {
                "col": int(c),
                "wall_color": wall_color,
                "doors": {int(r): int(column[r]) for r in door_rows},
            }
        )

    return g, bg, particle, walls


def _trace_chain(particle: dict, walls: list[dict]) -> tuple[list[tuple], list[dict], tuple[int, int, int]]:
    current_row = int(particle["row"])
    current_col = int(particle["col"])
    current_color = int(particle["color"])

    segments: list[tuple] = []
    visits: list[dict] = []
    for wall in walls:
        wall_col = int(wall["col"])
        matches = [row for row, color in wall["doors"].items() if int(color) == current_color]
        assert len(matches) == 1, "each wall must have exactly one door matching the current key color"
        target_row = int(matches[0])
        move = "stay" if target_row == current_row else ("down" if target_row > current_row else "up")

        segments.append(("h", current_row, current_col, wall_col - 1, current_color))
        segments.append(("v", wall_col - 1, current_row, target_row, current_color))
        segments.append(("p", target_row, wall_col, current_color))

        distractor_rows = [row for row in wall["doors"] if int(row) != target_row]
        visits.append(
            {
                "arrival_row": current_row,
                "arrival_color": current_color,
                "target_row": target_row,
                "move": move,
                "wall_col": wall_col,
                "wall_color": int(wall["wall_color"]),
                "door_rows": sorted(int(row) for row in wall["doors"]),
                "has_closer_distractor": any(
                    abs(int(row) - current_row) < abs(target_row - current_row) for row in distractor_rows
                ),
            }
        )

        current_row = target_row
        current_col = wall_col + 1
        current_color = int(wall["wall_color"])

    return segments, visits, (current_row, current_col, current_color)


def _render_output(grid: np.ndarray, particle: dict, walls: list[dict]) -> tuple[np.ndarray, list[dict]]:
    out = np.asarray(grid, dtype=int).copy()
    segments, visits, (final_row, final_col, final_color) = _trace_chain(particle, walls)
    for segment in segments:
        kind = segment[0]
        if kind == "h":
            _paint_horizontal(out, segment[1], segment[2], segment[3], segment[4])
        elif kind == "v":
            _paint_vertical(out, segment[1], segment[2], segment[3], segment[4])
        else:
            _, r, c, color = segment
            out[int(r), int(c)] = int(color)

    _paint_horizontal(out, final_row, final_col, out.shape[1] - 1, final_color)
    return out, visits


def main(grid: np.ndarray) -> np.ndarray:
    """
    Trace the particle through color-matched wall doors and draw its segmented trail.

    Input structure:
    - The background holds several non-adjacent full-height vertical wall columns.
    - Each wall is mostly one body color, with several one-cell door colors embedded in that column.
    - A single particle pixel sits on the left edge and carries the initial key color.

    Rule:
    - Move the particle from left to right; before each wall, stop in the column immediately to its left.
    - In that adjacent column, move vertically to the unique door whose color matches the particle's current color, then cross that door cell.
    - Draw every traversed segment in the color currently carried by the particle; after crossing a wall, replace the particle's color with that wall's body color, and keep all walls unchanged.

    Key insight:
    - Door colors choose where the particle crosses, while wall body colors become the next keys, so later choices depend causally on earlier walls.
    """
    g, _bg, particle, walls = _extract_structure(grid)
    out, _visits = _render_output(g, particle, walls)
    return out


def _sample_wall_columns(num_walls: int) -> tuple[int, list[int]] | None:
    left_gap = random.randint(*_GEN_LEFT_GAP_RANGE)
    right_gap = random.randint(*_GEN_RIGHT_GAP_RANGE)
    columns = [1 + left_gap]
    for _ in range(num_walls - 1):
        columns.append(columns[-1] + 1 + random.randint(*_GEN_BETWEEN_GAP_RANGE))
    width = columns[-1] + 1 + right_gap
    if width > 30:
        return None
    return width, columns


def _movement_modes(num_walls: int) -> list[str]:
    base = ["up", "down", "stay"]
    if num_walls == 1:
        return [random.choice(base)]
    if num_walls == 2:
        modes = ["up", "down"]
        random.shuffle(modes)
        return modes

    modes = base.copy()
    while len(modes) < num_walls:
        modes.append(random.choice(base))
    random.shuffle(modes)
    return modes


def _pick_target_row(height: int, current_row: int, mode: str, future_modes: list[str]) -> int | None:
    interior = list(range(1, height - 1))
    if mode == "stay":
        candidates = [current_row]
    elif mode == "up":
        candidates = [row for row in interior if row < current_row]
    else:
        candidates = [row for row in interior if row > current_row]
    if not candidates:
        return None

    preferred = [row for row in candidates if abs(row - current_row) >= 2]
    if preferred:
        candidates = preferred
    random.shuffle(candidates)

    need_up = "up" in future_modes
    need_down = "down" in future_modes
    for row in candidates:
        if need_up and row <= 1:
            continue
        if need_down and row >= height - 2:
            continue
        return int(row)

    return int(candidates[0]) if candidates else None


def _choose_spaced_rows(height: int, count: int, required_row: int, forced_row: int | None) -> list[int] | None:
    chosen = [int(required_row)]
    if forced_row is not None and forced_row != required_row:
        if not (1 <= forced_row < height - 1):
            return None
        if abs(int(forced_row) - int(required_row)) < 2:
            return None
        chosen.append(int(forced_row))

    interior = list(range(1, height - 1))
    while len(chosen) < count:
        options = [
            row
            for row in interior
            if row not in chosen and all(abs(int(row) - int(other)) >= 2 for other in chosen)
        ]
        if not options:
            return None
        chosen.append(int(random.choice(options)))

    return sorted(chosen)


def _build_input() -> np.ndarray | None:
    height = random.randrange(*_GEN_HEIGHT_RANGE)
    num_walls = random.choices(_GEN_WALL_COUNT_CHOICES, weights=_GEN_WALL_COUNT_WEIGHTS, k=1)[0]
    sampled = _sample_wall_columns(num_walls)
    if sampled is None:
        return None
    width, wall_cols = sampled

    chain_colors = random.sample(_PALETTE, num_walls + 1)
    initial_color = int(chain_colors[0])
    wall_colors = [int(color) for color in chain_colors[1:]]
    distractor_pool = [int(color) for color in _PALETTE if color not in chain_colors]

    grid = np.full((height, width), _BG, dtype=int)
    current_row = random.randint(2, height - 3)
    grid[current_row, 0] = initial_color

    current_key = initial_color
    modes = _movement_modes(num_walls)
    for index, wall_col in enumerate(wall_cols):
        wall_color = wall_colors[index]
        mode = modes[index]
        target_row = _pick_target_row(height, current_row, mode, modes[index + 1 :])
        if target_row is None:
            return None

        door_count = random.choice(_GEN_DOOR_COUNT_CHOICES)
        forced_row = None
        if mode != "stay" and abs(target_row - current_row) >= 3:
            step = -1 if mode == "up" else 1
            nearer = current_row + step
            if random.random() < 0.85:
                forced_row = nearer

        door_rows = _choose_spaced_rows(height, door_count, target_row, forced_row)
        if door_rows is None:
            return None

        if len(distractor_pool) < door_count - 1:
            return None
        distractor_colors = random.sample(distractor_pool, door_count - 1)
        random.shuffle(distractor_colors)

        grid[:, wall_col] = wall_color
        remaining_rows = [row for row in door_rows if row != target_row]
        grid[target_row, wall_col] = current_key
        for row, color in zip(remaining_rows, distractor_colors):
            grid[int(row), wall_col] = int(color)

        current_row = target_row
        current_key = wall_color

    return grid


def _validate_generated_input(grid: np.ndarray) -> None:
    g, bg, particle, walls = _extract_structure(grid)
    assert bg == _BG
    assert max(g.shape) <= 30
    assert len(walls) >= 3, "generator should create a nontrivial chain"
    assert np.count_nonzero(g != bg) == 1 + g.shape[0] * len(walls), "unexpected extra foreground"

    chain_palette = [int(particle["color"]), *[int(wall["wall_color"]) for wall in walls]]
    assert len(set(chain_palette)) == len(chain_palette), "trail segments should stay visually distinct"

    out, visits = _render_output(g, particle, walls)
    moves = {visit["move"] for visit in visits}
    assert {"up", "down", "stay"} <= moves, "generation should cover all vertical branches within one sample"
    assert any(visit["has_closer_distractor"] for visit in visits if visit["move"] != "stay"), "need a nearer wrong door as a counterexample"
    assert np.count_nonzero((g == bg) & (out != bg)) > 0, "output should add a visible trail"

    for prev, curr in zip(visits, visits[1:]):
        assert int(curr["arrival_color"]) == int(prev["wall_color"]), "later keys must come from earlier wall colors"


def generate_input() -> np.ndarray:
    for _ in range(_GEN_MAX_ATTEMPTS):
        grid = _build_input()
        if grid is None:
            continue
        try:
            _validate_generated_input(grid)
            return grid
        except AssertionError:
            continue
    raise RuntimeError("failed to generate a valid key-door causal-chain puzzle")
