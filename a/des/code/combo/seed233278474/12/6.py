from __future__ import annotations

from collections import deque
import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
ORANGE = 7
AZURE = 8
MAROON = 9

TOWER_COLORS = (BLUE, RED, YELLOW)
INPUT_COLORS = {BLACK, BLUE, RED, YELLOW, GREY, MAROON}
DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _perpendiculars(dr: int, dc: int) -> tuple[tuple[int, int], tuple[int, int]]:
    return (dc, -dr), (-dc, dr)


def _tower_cells(grid: np.ndarray) -> list[tuple[int, int, int]]:
    towers: list[tuple[int, int, int]] = []
    for r, c in np.argwhere(np.isin(grid, TOWER_COLORS)):
        towers.append((int(r), int(c), int(grid[r, c])))
    return towers


def _mix_color(covered: tuple[bool, bool, bool]) -> int:
    blue_hit, red_hit, yellow_hit = covered
    count = int(blue_hit) + int(red_hit) + int(yellow_hit)
    if count == 0:
        return BLACK
    if count == 3:
        return GREY
    if count == 1:
        if blue_hit:
            return BLUE
        if red_hit:
            return RED
        return YELLOW
    if blue_hit and red_hit:
        return GREEN
    if red_hit and yellow_hit:
        return ORANGE
    return AZURE


def _simulate(grid: np.ndarray) -> tuple[dict[int, np.ndarray], dict[str, int]]:
    """Propagate one state per (cell, direction, color) to avoid prism loops."""
    g = np.asarray(grid, dtype=int)
    masks = {color: np.zeros(g.shape, dtype=bool) for color in TOWER_COLORS}
    queue: deque[tuple[int, int, int, int, int]] = deque()
    seen: set[tuple[int, int, int, int, int]] = set()

    towers = _tower_cells(g)
    for r, c, color in towers:
        for dr, dc in DIRS:
            queue.append((r + dr, c + dc, dr, dc, color))

    prism_hits = 0
    wall_hits = 0
    split_count = 0

    while queue:
        r, c, dr, dc, color = queue.popleft()
        if not _in_bounds(g.shape, r, c):
            continue

        state = (r, c, dr, dc, color)
        if state in seen:
            continue
        seen.add(state)

        cell = int(g[r, c])
        if cell == MAROON:
            wall_hits += 1
            continue
        if cell == GREY:
            prism_hits += 1
            for ndr, ndc in _perpendiculars(dr, dc):
                split_count += 1
                queue.append((r + ndr, c + ndc, ndr, ndc, color))
            continue

        if cell == BLACK:
            masks[color][r, c] = True
        queue.append((r + dr, c + dc, dr, dc, color))

    return masks, {
        "prism_hits": prism_hits,
        "wall_hits": wall_hits,
        "split_count": split_count,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Broadcast colored rays from red/blue/yellow towers, split them at grey prisms,
    stop them at maroon walls, and mix overlapping colors on background cells.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert set(map(int, np.unique(g))) <= INPUT_COLORS, "unexpected input colors"

    towers = _tower_cells(g)
    tower_colors = [color for _r, _c, color in towers]
    assert 2 <= len(towers) <= 3, "expected 2-3 one-cell towers"
    assert len(set(tower_colors)) == len(towers), "tower colors must be distinct"

    masks, _stats = _simulate(g)
    out = g.copy()
    blue_mask = masks[BLUE]
    red_mask = masks[RED]
    yellow_mask = masks[YELLOW]

    for r, c in np.argwhere(g == BLACK):
        rr = int(r)
        cc = int(c)
        out[rr, cc] = _mix_color(
            (bool(blue_mask[rr, cc]), bool(red_mask[rr, cc]), bool(yellow_mask[rr, cc]))
        )
    return out


# --------------- generator ---------------

GRID_MIN = 15
GRID_MAX = 18
JUNCTION_MARGIN = 5
PRISM_STEPS = (2, 3)
TOWER_STEPS = (2, 3, 4)
WALL_STEPS = (2, 3, 4)
MAX_FILL_FRACTION = 0.55
MAX_GENERATION_ATTEMPTS = 500


def _object_free(
    pos: tuple[int, int], shape: tuple[int, int], object_positions: set[tuple[int, int]]
) -> bool:
    r, c = pos
    if not _in_bounds(shape, r, c):
        return False
    for orow, ocol in object_positions:
        if max(abs(orow - r), abs(ocol - c)) <= 1:
            return False
    return True


def _ray_cells(origin: tuple[int, int], direction: tuple[int, int], steps: int) -> list[tuple[int, int]]:
    r0, c0 = origin
    dr, dc = direction
    return [(r0 + dr * k, c0 + dc * k) for k in range(1, steps + 1)]


def _line_distance(a: tuple[int, int], b: tuple[int, int]) -> int:
    if a[0] == b[0]:
        return abs(a[1] - b[1])
    return abs(a[0] - b[0])


def _path_clear(
    cells: list[tuple[int, int]], shape: tuple[int, int], object_positions: set[tuple[int, int]]
) -> bool:
    for cell in cells:
        if not _in_bounds(shape, cell[0], cell[1]) or cell in object_positions:
            return False
    return True


def _add_object(
    grid: np.ndarray,
    object_positions: set[tuple[int, int]],
    pos: tuple[int, int],
    color: int,
) -> None:
    grid[pos] = color
    object_positions.add(pos)


def _candidate_specs(
    junction: tuple[int, int],
    axis: str,
    side: str,
) -> list[dict[str, object]]:
    jr, jc = junction
    candidates: list[dict[str, object]] = []

    prism_steps = list(PRISM_STEPS)
    tower_steps = list(TOWER_STEPS)
    random.shuffle(prism_steps)
    random.shuffle(tower_steps)

    if axis == "h":
        prism_dir = (0, -1) if side == "left" else (0, 1)
        toward_junction = (0, 1) if side == "left" else (0, -1)
        away_from_junction = (0, -1) if side == "left" else (0, 1)
        source_sides = ["up", "down"]
        random.shuffle(source_sides)
        for prism_step in prism_steps:
            prism = (jr + prism_dir[0] * prism_step, jc + prism_dir[1] * prism_step)
            for source_side in source_sides:
                for tower_step in tower_steps:
                    tower = (jr - tower_step, prism[1]) if source_side == "up" else (jr + tower_step, prism[1])
                    to_prism = (1, 0) if source_side == "up" else (-1, 0)
                    candidates.append(
                        {
                            "axis": axis,
                            "side": side,
                            "tower": tower,
                            "prism": prism,
                            "to_prism": to_prism,
                            "toward_junction": toward_junction,
                            "away_from_junction": away_from_junction,
                        }
                    )
    else:
        prism_dir = (-1, 0) if side == "up" else (1, 0)
        toward_junction = (1, 0) if side == "up" else (-1, 0)
        away_from_junction = (-1, 0) if side == "up" else (1, 0)
        source_sides = ["left", "right"]
        random.shuffle(source_sides)
        for prism_step in prism_steps:
            prism = (jr + prism_dir[0] * prism_step, jc + prism_dir[1] * prism_step)
            for source_side in source_sides:
                for tower_step in tower_steps:
                    tower = (prism[0], jc - tower_step) if source_side == "left" else (prism[0], jc + tower_step)
                    to_prism = (0, 1) if source_side == "left" else (0, -1)
                    candidates.append(
                        {
                            "axis": axis,
                            "side": side,
                            "tower": tower,
                            "prism": prism,
                            "to_prism": to_prism,
                            "toward_junction": toward_junction,
                            "away_from_junction": away_from_junction,
                        }
                    )
    random.shuffle(candidates)
    return candidates


def _spec_fits(
    spec: dict[str, object],
    junction: tuple[int, int],
    shape: tuple[int, int],
    object_positions: set[tuple[int, int]],
) -> bool:
    tower = spec["tower"]
    prism = spec["prism"]
    to_prism = spec["to_prism"]
    toward_junction = spec["toward_junction"]

    assert isinstance(tower, tuple)
    assert isinstance(prism, tuple)
    assert isinstance(to_prism, tuple)
    assert isinstance(toward_junction, tuple)

    if not _object_free(tower, shape, object_positions):
        return False
    if not _object_free(prism, shape, object_positions | {tower}):
        return False

    tower_gap = _line_distance(tower, prism) - 1
    prism_gap = _line_distance(prism, junction) - 1
    if tower_gap < 1 or prism_gap < 1:
        return False

    if not _path_clear(_ray_cells(tower, to_prism, tower_gap), shape, object_positions):
        return False
    if not _path_clear(_ray_cells(prism, toward_junction, prism_gap), shape, object_positions):
        return False
    return True


def _reserve_cells(specs: list[dict[str, object]], junction: tuple[int, int]) -> set[tuple[int, int]]:
    reserved = {junction}
    for spec in specs:
        tower = spec["tower"]
        prism = spec["prism"]
        to_prism = spec["to_prism"]
        toward_junction = spec["toward_junction"]
        assert isinstance(tower, tuple)
        assert isinstance(prism, tuple)
        assert isinstance(to_prism, tuple)
        assert isinstance(toward_junction, tuple)
        reserved.update(_ray_cells(tower, to_prism, _line_distance(tower, prism) - 1))
        reserved.update(_ray_cells(prism, toward_junction, _line_distance(prism, junction)))
    return reserved


def _try_place_wall(
    grid: np.ndarray,
    object_positions: set[tuple[int, int]],
    reserved: set[tuple[int, int]],
    origin: tuple[int, int],
    direction: tuple[int, int],
) -> bool:
    wall_steps = list(WALL_STEPS)
    random.shuffle(wall_steps)
    for steps in wall_steps:
        cells = _ray_cells(origin, direction, steps)
        wall = cells[-1]
        path = cells[:-1]
        if any(cell in reserved or cell in object_positions for cell in cells):
            continue
        if not _path_clear(path, grid.shape, object_positions):
            continue
        if not _object_free(wall, grid.shape, object_positions):
            continue
        _add_object(grid, object_positions, wall, MAROON)
        return True
    return False


def _coverage_counts(masks: dict[int, np.ndarray]) -> np.ndarray:
    return np.stack([masks[BLUE], masks[RED], masks[YELLOW]], axis=0).sum(axis=0)


def _oracle_accepts(
    grid: np.ndarray,
    junction: tuple[int, int],
    primary_count: int,
    want_three: bool,
) -> bool:
    try:
        out = main(grid)
    except Exception:
        return False

    masks, stats = _simulate(grid)
    counts = _coverage_counts(masks)
    jr, jc = junction
    changed = int(np.sum((grid == BLACK) & (counts > 0)))
    two_mix = int(np.sum(counts == 2))
    three_mix = int(np.sum(counts == 3))

    if stats["prism_hits"] < primary_count:
        return False
    if stats["wall_hits"] == 0 or stats["split_count"] == 0:
        return False
    if changed < 12 or changed > int(MAX_FILL_FRACTION * grid.size):
        return False
    if counts[jr, jc] != (3 if want_three else 2):
        return False
    if want_three and three_mix == 0:
        return False
    if not want_three and two_mix == 0:
        return False
    return not np.array_equal(out, grid)


def _placement_templates(want_three: bool) -> list[tuple[str, str]]:
    if not want_three:
        templates = [("h", random.choice(("left", "right"))), ("v", random.choice(("up", "down")))]
        random.shuffle(templates)
        return templates

    if random.random() < 0.5:
        templates = [("h", "left"), ("h", "right"), ("v", random.choice(("up", "down")))]
    else:
        templates = [("h", random.choice(("left", "right"))), ("v", "up"), ("v", "down")]
    random.shuffle(templates)
    return templates


def _build_attempt(want_three: bool) -> np.ndarray | None:
    height = random.randint(GRID_MIN, GRID_MAX)
    width = random.randint(GRID_MIN, GRID_MAX)
    junction = (
        random.randint(JUNCTION_MARGIN, height - 1 - JUNCTION_MARGIN),
        random.randint(JUNCTION_MARGIN, width - 1 - JUNCTION_MARGIN),
    )

    grid = np.full((height, width), BLACK, dtype=int)
    object_positions: set[tuple[int, int]] = set()
    templates = _placement_templates(want_three)
    colors = random.sample(list(TOWER_COLORS), len(templates))

    specs: list[dict[str, object]] = []
    for color, (axis, side) in zip(colors, templates):
        placed = False
        for spec in _candidate_specs(junction, axis, side):
            if not _spec_fits(spec, junction, grid.shape, object_positions):
                continue
            spec["color"] = color
            specs.append(spec)
            _add_object(grid, object_positions, spec["tower"], color)
            _add_object(grid, object_positions, spec["prism"], GREY)
            placed = True
            break
        if not placed:
            return None

    reserved = _reserve_cells(specs, junction)

    for spec in specs:
        if not _try_place_wall(
            grid,
            object_positions,
            reserved,
            spec["prism"],
            spec["away_from_junction"],
        ):
            return None

    for spec in random.sample(specs, len(specs)):
        extra_dirs = [direction for direction in DIRS if direction != spec["to_prism"]]
        random.shuffle(extra_dirs)
        for direction in extra_dirs[: random.randint(1, 2)]:
            _try_place_wall(grid, object_positions, reserved, spec["tower"], direction)

    if not _oracle_accepts(grid, junction, len(specs), want_three):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        want_three = random.random() < 0.5
        grid = _build_attempt(want_three)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid sample after {MAX_GENERATION_ATTEMPTS} attempts")
