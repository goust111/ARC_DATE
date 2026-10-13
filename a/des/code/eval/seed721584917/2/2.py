from __future__ import annotations

import random

import numpy as np


_RNG = random.Random()
_GEN_RETRIES = 500
_COLORS = tuple(range(10))
_WALL_COLOR = 5
_PRIORITY_ORDER = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)
_PRIORITY_RANK = {color: rank for rank, color in enumerate(_PRIORITY_ORDER)}
_DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))
_RING_COORDS = (
    (0, 0),
    (0, 1),
    (0, 2),
    (1, 0),
    (1, 2),
    (2, 0),
    (2, 1),
    (2, 2),
)


def _bg_color(grid: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(values[np.argmax(counts)])


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _is_tower_window(window: np.ndarray, bg: int) -> bool:
    if window.shape != (3, 3):
        return False
    emit = int(window[0, 0])
    if emit == bg:
        return False
    if any(int(window[r, c]) != emit for r, c in _RING_COORDS):
        return False
    priority = int(window[1, 1])
    return priority not in (bg, emit)


def _tower_cells(top: tuple[int, int]) -> set[tuple[int, int]]:
    r0, c0 = top
    return {(r0 + dr, c0 + dc) for dr in range(3) for dc in range(3)}


def _detect_towers(grid: np.ndarray, bg: int) -> tuple[list[dict[str, object]], set[tuple[int, int]]]:
    g = np.asarray(grid, dtype=int)
    candidates: list[dict[str, object]] = []
    for r0 in range(g.shape[0] - 2):
        for c0 in range(g.shape[1] - 2):
            window = g[r0 : r0 + 3, c0 : c0 + 3]
            if not _is_tower_window(window, bg):
                continue
            candidates.append(
                {
                    "top": (r0, c0),
                    "center": (r0 + 1, c0 + 1),
                    "emit": int(window[0, 0]),
                    "priority": int(window[1, 1]),
                }
            )

    towers: list[dict[str, object]] = []
    occupied: set[tuple[int, int]] = set()
    for candidate in candidates:
        cells = _tower_cells(candidate["top"])
        if cells & occupied:
            continue
        towers.append(candidate)
        occupied |= cells

    assert towers, "expected at least one signal tower"
    return towers, occupied


def _ray_cells(grid: np.ndarray, bg: int, center: tuple[int, int], direction: tuple[int, int]) -> list[tuple[int, int]]:
    dr, dc = direction
    r = center[0] + 2 * dr
    c = center[1] + 2 * dc
    cells: list[tuple[int, int]] = []
    while _in_bounds(grid.shape, r, c) and int(grid[r, c]) == bg:
        cells.append((r, c))
        r += dr
        c += dc
    return cells


def main(grid: np.ndarray) -> np.ndarray:
    """
    Find 3x3 signal towers, cast their four rays through background cells, and
    resolve ray overlaps using the towers' center-color priority order.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _bg_color(g)
    towers, _tower_cells_used = _detect_towers(g, bg)

    hits: dict[tuple[int, int], list[int]] = {}
    for tower_index, tower in enumerate(towers):
        center = tower["center"]
        for direction in _DIRS:
            for cell in _ray_cells(g, bg, center, direction):
                hits.setdefault(cell, []).append(tower_index)

    out = g.copy()
    for cell, tower_indices in hits.items():
        emit_colors = {int(towers[i]["emit"]) for i in tower_indices}
        if len(emit_colors) == 1:
            out[cell] = next(iter(emit_colors))
            continue

        best_rank = max(_PRIORITY_RANK[int(towers[i]["priority"])] for i in tower_indices)
        winner_colors = {
            int(towers[i]["emit"])
            for i in tower_indices
            if _PRIORITY_RANK[int(towers[i]["priority"])] == best_rank
        }
        assert len(winner_colors) == 1, "ray overlap must have a unique winning emission color"
        out[cell] = next(iter(winner_colors))

    return out


def _max_spaced_count(lo: int, hi: int, min_gap: int) -> int:
    if hi < lo:
        return 0
    return 1 + (hi - lo) // min_gap


def _sample_spaced_positions(lo: int, hi: int, count: int, min_gap: int) -> list[int] | None:
    values = list(range(lo, hi + 1))
    if count > len(values):
        return None
    for _ in range(300):
        picks = sorted(_RNG.sample(values, count))
        if all(b - a >= min_gap for a, b in zip(picks, picks[1:])):
            return picks
    return None


def _path_until_blocked(
    shape: tuple[int, int],
    occupied: set[tuple[int, int]],
    center: tuple[int, int],
    direction: tuple[int, int],
) -> list[tuple[int, int]]:
    dr, dc = direction
    r = center[0] + 2 * dr
    c = center[1] + 2 * dc
    cells: list[tuple[int, int]] = []
    while _in_bounds(shape, r, c) and (r, c) not in occupied:
        cells.append((r, c))
        r += dr
        c += dc
    return cells


def _direction_name(direction: tuple[int, int]) -> str:
    return {
        (-1, 0): "up",
        (1, 0): "down",
        (0, -1): "left",
        (0, 1): "right",
    }[direction]


def _place_wall_segment(
    occupied: set[tuple[int, int]],
    shape: tuple[int, int],
    anchor: tuple[int, int],
    ray_direction: tuple[int, int],
) -> set[tuple[int, int]] | None:
    if ray_direction[0] == 0:
        options = [
            {(anchor[0], anchor[1])},
            {(anchor[0] - 1, anchor[1]), (anchor[0], anchor[1]), (anchor[0] + 1, anchor[1])},
        ]
    else:
        options = [
            {(anchor[0], anchor[1])},
            {(anchor[0], anchor[1] - 1), (anchor[0], anchor[1]), (anchor[0], anchor[1] + 1)},
        ]

    _RNG.shuffle(options)
    for segment in options:
        if all(_in_bounds(shape, r, c) and (r, c) not in occupied for r, c in segment):
            return segment
    return None


def _render_input(
    height: int,
    width: int,
    bg: int,
    towers: list[dict[str, int | tuple[int, int]]],
    walls: set[tuple[int, int]],
) -> np.ndarray:
    grid = np.full((height, width), bg, dtype=int)
    for r, c in walls:
        grid[r, c] = _WALL_COLOR
    for tower in towers:
        r, c = tower["center"]
        emit = tower["emit"]
        priority = tower["priority"]
        grid[r - 1 : r + 2, c - 1 : c + 2] = emit
        grid[r, c] = priority
    return grid


def _choose_tower_colors(n_towers: int, bg: int, scenario: str) -> tuple[list[int], list[int], tuple[int, int]]:
    usable = [color for color in _COLORS if color not in (bg, _WALL_COLOR)]
    pair = tuple(_RNG.sample(range(n_towers), 2))

    if scenario == "same_color_cross":
        emit_palette = _RNG.sample(usable, _RNG.randint(2, min(3, n_towers)))
        shared_color = emit_palette[0]
        emits = [_RNG.choice(emit_palette) for _ in range(n_towers)]
        emits[pair[0]] = shared_color
        emits[pair[1]] = shared_color
    else:
        emit_palette = _RNG.sample(usable, _RNG.randint(max(2, n_towers - 1), n_towers))
        emits = [_RNG.choice(emit_palette) for _ in range(n_towers)]
        if emits[pair[0]] == emits[pair[1]]:
            alternatives = [color for color in emit_palette if color != emits[pair[0]]]
            emits[pair[1]] = _RNG.choice(alternatives)

    priority_pool = [color for color in usable if color not in set(emit_palette)]
    if len(priority_pool) < n_towers:
        raise ValueError("not enough distinct priority colors")
    priorities = _RNG.sample(priority_pool, n_towers)
    _RNG.shuffle(priorities)

    if priorities[pair[0]] == priorities[pair[1]]:
        raise AssertionError("priority colors must be unique")
    return emits, priorities, pair


def _build_walls(
    shape: tuple[int, int],
    towers: list[dict[str, int | tuple[int, int]]],
    primary_pair: tuple[int, int],
    scenario: str,
) -> tuple[set[tuple[int, int]], dict[str, object]]:
    occupied: set[tuple[int, int]] = set()
    for tower in towers:
        occupied |= _tower_cells((tower["center"][0] - 1, tower["center"][1] - 1))

    walls: set[tuple[int, int]] = set()
    info: dict[str, object] = {"blocked_tower": None, "blocked_direction": None, "blocked_anchor": None}
    if scenario != "wall_block":
        return walls, info

    blockers = [i for i in range(len(towers)) if i not in primary_pair]
    if not blockers:
        blockers = [primary_pair[0]]
    _RNG.shuffle(blockers)

    for blocker in blockers:
        directions = list(_DIRS)
        _RNG.shuffle(directions)
        center = towers[blocker]["center"]
        for direction in directions:
            path = _path_until_blocked(shape, occupied, center, direction)
            if len(path) < 2:
                continue
            candidate_cells = path[1:-1] or path[:-1]
            _RNG.shuffle(candidate_cells)
            for anchor in candidate_cells:
                segment = _place_wall_segment(occupied | walls, shape, anchor, direction)
                if segment is None:
                    continue
                walls |= segment
                info = {
                    "blocked_tower": blocker,
                    "blocked_direction": _direction_name(direction),
                    "blocked_anchor": anchor,
                }
                return walls, info

    raise RuntimeError("failed to place a blocking wall")


def _cross_cells(a: dict[str, int | tuple[int, int]], b: dict[str, int | tuple[int, int]]) -> list[tuple[int, int]]:
    ar, ac = a["center"]
    br, bc = b["center"]
    return [(ar, bc), (br, ac)]


def _winner_color(a: dict[str, int | tuple[int, int]], b: dict[str, int | tuple[int, int]]) -> int:
    if int(a["emit"]) == int(b["emit"]):
        return int(a["emit"])
    if _PRIORITY_RANK[int(a["priority"])] > _PRIORITY_RANK[int(b["priority"])]:
        return int(a["emit"])
    return int(b["emit"])


def _gen_attempt() -> np.ndarray | None:
    scenario = _RNG.choice(("priority_cross", "same_color_cross", "wall_block"))
    height = _RNG.randint(13, 20)
    width = _RNG.randint(13, 20)
    bg = _RNG.choice([color for color in _COLORS if color != _WALL_COLOR])

    lo_r, hi_r = 2, height - 3
    lo_c, hi_c = 2, width - 3
    max_rows = _max_spaced_count(lo_r, hi_r, 4)
    max_cols = _max_spaced_count(lo_c, hi_c, 4)
    n_towers_max = min(4, max_rows, max_cols)
    if n_towers_max < 3:
        return None
    n_towers = _RNG.randint(3, n_towers_max)

    rows = _sample_spaced_positions(lo_r, hi_r, n_towers, 4)
    cols = _sample_spaced_positions(lo_c, hi_c, n_towers, 4)
    if rows is None or cols is None:
        return None
    _RNG.shuffle(cols)

    try:
        emits, priorities, primary_pair = _choose_tower_colors(n_towers, bg, scenario)
    except ValueError:
        return None

    towers: list[dict[str, int | tuple[int, int]]] = []
    for row, col, emit, priority in zip(rows, cols, emits, priorities):
        towers.append({"center": (row, col), "emit": emit, "priority": priority})

    walls, wall_info = _build_walls((height, width), towers, primary_pair, scenario)
    grid = _render_input(height, width, bg, towers, walls)

    primary_a = towers[primary_pair[0]]
    primary_b = towers[primary_pair[1]]
    cross_cells = _cross_cells(primary_a, primary_b)
    if any(int(grid[r, c]) != bg for r, c in cross_cells):
        return None

    expected_winner = _winner_color(primary_a, primary_b)
    out = main(grid)
    if not isinstance(out, np.ndarray) or out.shape != grid.shape:
        return None

    if scenario == "same_color_cross":
        if int(primary_a["emit"]) != int(primary_b["emit"]):
            return None
    else:
        if int(primary_a["emit"]) == int(primary_b["emit"]):
            return None

    if any(int(out[r, c]) != expected_winner for r, c in cross_cells):
        return None

    if scenario == "wall_block":
        if wall_info["blocked_tower"] is None:
            return None
        blocker = towers[int(wall_info["blocked_tower"])]
        direction = {
            "up": (-1, 0),
            "down": (1, 0),
            "left": (0, -1),
            "right": (0, 1),
        }[str(wall_info["blocked_direction"])]
        anchor_r, anchor_c = wall_info["blocked_anchor"]
        before_anchor = (anchor_r - direction[0], anchor_c - direction[1])
        if int(out[anchor_r, anchor_c]) != _WALL_COLOR:
            return None
        if _in_bounds(grid.shape, *before_anchor) and int(grid[before_anchor]) == bg:
            if int(out[before_anchor]) != int(blocker["emit"]):
                return None

    if int(np.count_nonzero(out != grid)) < min(height, width):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid signal-tower input")
