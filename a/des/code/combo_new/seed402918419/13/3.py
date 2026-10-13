import random
from typing import Dict, Iterable, List, Sequence, Set, Tuple

import numpy as np


Coord = Tuple[int, int]
Tower = Tuple[int, int, int]

DIRS: Dict[str, Coord] = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
DIR_ORDER = ("up", "down", "left", "right")
PALETTE = [1, 2, 3, 4, 5, 6, 7, 8, 9]

GRID_H_RANGE = (12, 24)
GRID_W_RANGE = (12, 24)
TOWER_COUNT_RANGE = (3, 5)
PALETTE_SIZE_RANGE = (4, 6)
MAX_TOTAL_WALLS = 7
MIN_SPACING = 3
MAX_GENERATION_ATTEMPTS = 400
PLACEMENT_RETRIES = 40


def _inside(shape: Tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _neighbors4(r: int, c: int) -> Iterable[Coord]:
    for dr, dc in DIRS.values():
        yield r + dr, c + dc


def _axis_of_direction(name: str) -> str:
    return "horizontal" if name in {"left", "right"} else "vertical"


def _other_color(palette: Sequence[int], blocked: Set[int]) -> int:
    options = [color for color in palette if color not in blocked]
    assert options, "palette exhausted"
    return random.choice(options)


def _sample_spaced_positions(lo: int, hi: int, count: int, min_gap: int) -> List[int] | None:
    values = list(range(lo, hi + 1))
    random.shuffle(values)
    chosen: List[int] = []
    for value in values:
        if all(abs(value - prev) >= min_gap for prev in chosen):
            chosen.append(value)
            if len(chosen) == count:
                return sorted(chosen)
    return None


def _parse_scene(grid: np.ndarray) -> Tuple[List[Tower], Set[Coord], Dict[Coord, Coord], List[Tuple[Coord, Coord]]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = g.shape
    nonblack = g != 0
    coords = [tuple(map(int, rc)) for rc in np.argwhere(nonblack)]
    assert coords, "expected at least one non-black object"

    neighbor_map: Dict[Coord, List[Coord]] = {}
    for r, c in coords:
        neighbors = []
        for rr, cc in _neighbors4(r, c):
            if 0 <= rr < h and 0 <= cc < w and nonblack[rr, cc]:
                neighbors.append((rr, cc))
        neighbor_map[(r, c)] = neighbors

    towers: List[Tower] = []
    tower_cells: Set[Coord] = set()
    wall_next: Dict[Coord, Coord] = {}
    walls: List[Tuple[Coord, Coord]] = []
    seen: Set[Coord] = set()

    for pos in coords:
        degree = len(neighbor_map[pos])
        if degree == 0:
            towers.append((pos[0], pos[1], int(g[pos])))
            tower_cells.add(pos)
            continue

        assert degree == 1, "non-black pixels must be isolated towers or 2-pixel walls"
        partner = neighbor_map[pos][0]
        assert len(neighbor_map[partner]) == 1 and neighbor_map[partner][0] == pos, "wall cells must form isolated pairs"
        assert int(g[pos]) != int(g[partner]), "wall cells must use two different colors"
        wall_next[pos] = partner
        if pos not in seen:
            seen.add(pos)
            seen.add(partner)
            walls.append((pos, partner))

    towers.sort(key=lambda item: (item[0], item[1]))
    assert towers, "expected at least one signal tower"
    return towers, tower_cells, wall_next, walls


def _trace_rays(
    grid: np.ndarray,
    towers: Sequence[Tower],
    tower_cells: Set[Coord],
    wall_next: Dict[Coord, Coord],
    collect_stats: bool = False,
) -> Tuple[np.ndarray, Dict[str, object]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    out = g.copy()
    stats: Dict[str, object] = {
        "matched_walls": 0,
        "blocked_walls": 0,
        "overwritten_cells": 0,
        "painted_cells": 0,
        "wall_axes": set(),
    }

    for r0, c0, start_color in towers:
        for name in DIR_ORDER:
            dr, dc = DIRS[name]
            current = start_color
            r, c = r0 + dr, c0 + dc
            while 0 <= r < h and 0 <= c < w:
                pos = (r, c)
                if pos in tower_cells:
                    break

                if pos in wall_next:
                    partner = wall_next[pos]
                    axis = "horizontal" if partner[0] == r else "vertical"
                    if collect_stats:
                        cast = stats["wall_axes"]
                        assert isinstance(cast, set)
                        cast.add(axis)
                    if partner == (r + dr, c + dc) and current == int(g[pos]):
                        if collect_stats:
                            stats["matched_walls"] = int(stats["matched_walls"]) + 1
                        current = int(g[partner])
                        r += 2 * dr
                        c += 2 * dc
                        continue
                    if collect_stats:
                        stats["blocked_walls"] = int(stats["blocked_walls"]) + 1
                    break

                assert int(g[pos]) == 0, "only towers and walls may occupy non-background cells"
                if int(out[pos]) != 0:
                    if collect_stats:
                        stats["overwritten_cells"] = int(stats["overwritten_cells"]) + 1
                else:
                    if collect_stats:
                        stats["painted_cells"] = int(stats["painted_cells"]) + 1
                out[pos] = current
                r += dr
                c += dc

    return out, stats


def main(grid: np.ndarray) -> np.ndarray:
    """Broadcast each tower color along four rays, letting ordered 2-pixel filter walls recolor matching rays that pass through.

    Input structure:
    - The background is black.
    - Some non-black cells are isolated single pixels; these are the signal towers.
    - Some non-black cells form isolated orthogonally adjacent pairs of two different colors; these are filter walls.

    Rule:
    - Process signal towers in row-major order and cast four rays from each tower: up, down, left, and right.
    - A ray paints every black cell on its straight path with its current color and stops at the grid boundary or when it reaches another tower.
    - When the next obstacle is a 2-pixel wall aligned with the ray, the first wall pixel on that path is the input color and the second is the output color.
    - If the ray color matches the input color, the ray skips across the wall and continues beyond it using the output color; otherwise it stops before the wall.
    - Later rays overwrite earlier ray paint where they cross, but towers and wall pixels themselves stay unchanged.

    Key insight:
    - The wall pair is directional along the travel axis: whichever wall pixel the ray encounters first is the required input color, and its partner becomes the new color after a successful pass.
    """
    g = np.asarray(grid, dtype=int)
    towers, tower_cells, wall_next, _walls = _parse_scene(g)
    out, _ = _trace_rays(g, towers, tower_cells, wall_next, collect_stats=False)
    return out


def _line_cells(shape: Tuple[int, int], tower: Coord, direction: str) -> List[Coord]:
    h, w = shape
    dr, dc = DIRS[direction]
    cells: List[Coord] = []
    r, c = tower[0] + dr, tower[1] + dc
    while 0 <= r < h and 0 <= c < w:
        cells.append((r, c))
        r += dr
        c += dc
    return cells


def _pair_allowed(
    pair: Tuple[Coord, Coord],
    direction: str,
    tower_rows: Set[int],
    tower_cols: Set[int],
    source: Coord,
) -> bool:
    first, second = pair
    axis = _axis_of_direction(direction)
    if axis == "horizontal":
        forbidden_cols = tower_cols - {source[1]}
        return first[1] not in forbidden_cols and second[1] not in forbidden_cols
    forbidden_rows = tower_rows - {source[0]}
    return first[0] not in forbidden_rows and second[0] not in forbidden_rows


def _can_place_pair(shape: Tuple[int, int], occupied: Set[Coord], first: Coord, second: Coord) -> bool:
    if not (_inside(shape, *first) and _inside(shape, *second)):
        return False
    if first in occupied or second in occupied:
        return False
    if abs(first[0] - second[0]) + abs(first[1] - second[1]) != 1:
        return False

    pair = {first, second}
    for r, c in pair:
        for rr, cc in _neighbors4(r, c):
            if not _inside(shape, rr, cc):
                continue
            if (rr, cc) in occupied and (rr, cc) not in pair:
                return False
    return True


def _place_sequence(
    grid: np.ndarray,
    occupied: Set[Coord],
    tower: Tower,
    direction: str,
    events: Sequence[str],
    palette: Sequence[int],
    tower_rows: Set[int],
    tower_cols: Set[int],
) -> bool:
    if not events:
        return True

    line = _line_cells(grid.shape, (tower[0], tower[1]), direction)
    if len(line) < 2:
        return False

    last_extra = 1 if events[-1] == "match" else 0
    for _ in range(PLACEMENT_RETRIES):
        temp_occupied = set(occupied)
        cursor = 1  # keep one black cell between tower and the first wall
        current = tower[2]
        planned: List[Tuple[Coord, Coord, int, int]] = []
        ok = True

        for idx_event, event in enumerate(events):
            remaining = len(events) - idx_event - 1
            max_idx = len(line) - 2 - 3 * remaining - last_extra
            candidates: List[int] = []
            for idx in range(cursor, max_idx + 1):
                first = line[idx]
                second = line[idx + 1]
                if not _pair_allowed((first, second), direction, tower_rows, tower_cols, (tower[0], tower[1])):
                    continue
                if not _can_place_pair(grid.shape, temp_occupied, first, second):
                    continue
                candidates.append(idx)

            if not candidates:
                ok = False
                break

            chosen = random.choice(candidates)
            first = line[chosen]
            second = line[chosen + 1]
            if event == "match":
                in_color = current
                out_color = _other_color(palette, {current})
                current = out_color
            else:
                in_color = _other_color(palette, {current})
                out_color = _other_color(palette, {in_color})
            planned.append((first, second, in_color, out_color))
            temp_occupied.add(first)
            temp_occupied.add(second)
            cursor = chosen + 3

        if not ok:
            continue

        for first, second, in_color, out_color in planned:
            grid[first] = in_color
            grid[second] = out_color
            occupied.add(first)
            occupied.add(second)
        return True

    return False


def _choose_specs(towers: Sequence[Tower], shape: Tuple[int, int]) -> List[Dict[str, object]]:
    specs: List[Dict[str, object]] = []
    for tower in towers:
        for direction in DIR_ORDER:
            specs.append(
                {
                    "tower": tower,
                    "direction": direction,
                    "axis": _axis_of_direction(direction),
                    "length": len(_line_cells(shape, (tower[0], tower[1]), direction)),
                }
            )
    random.shuffle(specs)
    return specs


def _valid_sample(grid: np.ndarray) -> bool:
    try:
        towers, tower_cells, wall_next, walls = _parse_scene(grid)
        out, stats = _trace_rays(grid, towers, tower_cells, wall_next, collect_stats=True)
    except AssertionError:
        return False

    if len(towers) < 3 or len(walls) < 2:
        return False
    if int(stats["matched_walls"]) < 1 or int(stats["blocked_walls"]) < 1:
        return False
    if int(stats["overwritten_cells"]) < 1:
        return False
    wall_axes = stats["wall_axes"]
    assert isinstance(wall_axes, set)
    if wall_axes != {"horizontal", "vertical"}:
        return False
    if int(stats["painted_cells"]) < 12:
        return False
    return out.shape == grid.shape and max(grid.shape) <= 30


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    n_towers = random.randint(*TOWER_COUNT_RANGE)

    rows = _sample_spaced_positions(2, h - 3, n_towers, MIN_SPACING)
    cols = _sample_spaced_positions(2, w - 3, n_towers, MIN_SPACING)
    if rows is None or cols is None:
        return None

    cols_perm = cols[:]
    random.shuffle(cols_perm)
    palette = random.sample(PALETTE, random.randint(*PALETTE_SIZE_RANGE))
    tower_colors = [random.choice(palette) for _ in range(n_towers)]
    if len(set(tower_colors)) == 1:
        tower_colors[-1] = _other_color(palette, {tower_colors[0]})

    towers: List[Tower] = []
    grid = np.zeros((h, w), dtype=int)
    occupied: Set[Coord] = set()
    for r, c, color in zip(rows, cols_perm, tower_colors):
        grid[r, c] = color
        occupied.add((r, c))
        towers.append((r, c, color))

    tower_rows = {tower[0] for tower in towers}
    tower_cols = {tower[1] for tower in towers}
    specs = _choose_specs(towers, grid.shape)

    horiz_long = [spec for spec in specs if spec["axis"] == "horizontal" and int(spec["length"]) >= 6]
    vert_single = [spec for spec in specs if spec["axis"] == "vertical" and int(spec["length"]) >= 4]
    if not horiz_long or not vert_single:
        return None

    used: Set[Tuple[int, int, str]] = set()
    mandatory_h = random.choice(horiz_long)
    key_h = (mandatory_h["tower"][0], mandatory_h["tower"][1], mandatory_h["direction"])
    used.add(key_h)
    if not _place_sequence(
        grid,
        occupied,
        mandatory_h["tower"],
        mandatory_h["direction"],
        ("match", "mismatch"),
        palette,
        tower_rows,
        tower_cols,
    ):
        return None

    vert_choices = [
        spec
        for spec in vert_single
        if (spec["tower"][0], spec["tower"][1], spec["direction"]) not in used
    ]
    if not vert_choices:
        return None
    mandatory_v = random.choice(vert_choices)
    used.add((mandatory_v["tower"][0], mandatory_v["tower"][1], mandatory_v["direction"]))
    vertical_events = random.choice((("match",), ("mismatch",)))
    if not _place_sequence(
        grid,
        occupied,
        mandatory_v["tower"],
        mandatory_v["direction"],
        vertical_events,
        palette,
        tower_rows,
        tower_cols,
    ):
        return None

    total_walls = 3
    for spec in specs:
        key = (spec["tower"][0], spec["tower"][1], spec["direction"])
        if key in used or total_walls >= MAX_TOTAL_WALLS:
            continue
        roll = random.random()
        if roll < 0.18:
            events: Sequence[str] = ("match",)
        elif roll < 0.36:
            events = ("mismatch",)
        elif roll < 0.48 and int(spec["length"]) >= 7:
            events = random.choice((("match", "match"), ("match", "mismatch")))
        else:
            continue

        if _place_sequence(
            grid,
            occupied,
            spec["tower"],
            spec["direction"],
            events,
            palette,
            tower_rows,
            tower_cols,
        ):
            used.add(key)
            total_walls += len(events)

    if not _valid_sample(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid signal-tower filter-wall sample")
