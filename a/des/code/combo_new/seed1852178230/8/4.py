import random
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np


BG = 0
WALL = 5
TOWER_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
DIR4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
MAX_GEN_RETRIES = 400


Coord = Tuple[int, int]
Wall = Dict[str, object]


def _components(mask: np.ndarray) -> List[List[Coord]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: List[List[Coord]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: List[Coord] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for dr, dc in DIR4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _detect_walls(grid: np.ndarray) -> List[Wall]:
    walls: List[Wall] = []
    for comp in _components(grid == WALL):
        rows = sorted({r for r, _ in comp})
        cols = sorted({c for _, c in comp})
        if len(rows) == 1:
            row = rows[0]
            c0, c1 = cols[0], cols[-1]
            assert c1 - c0 + 1 == len(comp), "gray wall must be a contiguous horizontal segment"
            walls.append(
                {
                    "orientation": "H",
                    "row": row,
                    "c0": c0,
                    "c1": c1,
                    "cells": tuple((row, c) for c in range(c0, c1 + 1)),
                }
            )
        else:
            assert len(cols) == 1, "gray wall must be straight"
            col = cols[0]
            r0, r1 = rows[0], rows[-1]
            assert r1 - r0 + 1 == len(comp), "gray wall must be a contiguous vertical segment"
            walls.append(
                {
                    "orientation": "V",
                    "col": col,
                    "r0": r0,
                    "r1": r1,
                    "cells": tuple((r, col) for r in range(r0, r1 + 1)),
                }
            )
    walls.sort(
        key=lambda wall: (
            wall["cells"][0][0],  # type: ignore[index]
            wall["cells"][0][1],  # type: ignore[index]
        )
    )
    return walls


def _wall_lookup(walls: Sequence[Wall]) -> Dict[Coord, int]:
    lookup: Dict[Coord, int] = {}
    for idx, wall in enumerate(walls):
        for cell in wall["cells"]:  # type: ignore[index]
            lookup[cell] = idx
    return lookup


def _first_nonblack(grid: np.ndarray, r: int, c: int, dr: int, dc: int) -> Optional[Tuple[int, int, int]]:
    h, w = grid.shape
    rr, cc = r + dr, c + dc
    while 0 <= rr < h and 0 <= cc < w:
        value = int(grid[rr, cc])
        if value != BG:
            return rr, cc, value
        rr += dr
        cc += dc
    return None


def _collect_wall_signals(
    grid: np.ndarray,
    walls: Sequence[Wall],
) -> Tuple[List[List[int]], bool, bool]:
    wall_by_cell = _wall_lookup(walls)
    towers = [tuple(map(int, pos)) for pos in np.argwhere((grid != BG) & (grid != WALL))]
    towers.sort()

    signals: List[List[int]] = [[] for _ in walls]
    seen: List[Set[int]] = [set() for _ in walls]
    saw_tower_block = False
    saw_boundary_miss = False

    for r, c in towers:
        color = int(grid[r, c])
        for dr, dc in DIR4:
            hit = _first_nonblack(grid, r, c, dr, dc)
            if hit is None:
                saw_boundary_miss = True
                continue
            rr, cc, value = hit
            if value == WALL:
                wall_idx = wall_by_cell[(rr, cc)]
                if color not in seen[wall_idx]:
                    seen[wall_idx].add(color)
                    signals[wall_idx].append(color)
            else:
                saw_tower_block = True

    return signals, saw_tower_block, saw_boundary_miss


def main(grid: np.ndarray) -> np.ndarray:
    """
    Turn each gray receiver wall into a color-striped rectangle determined by the towers that can see it.

    Input structure:
    - The background is black.
    - Gray pixels form separate 4-connected horizontal or vertical wall segments.
    - Other non-black pixels are isolated one-cell signal towers in distinct colors.

    Rule:
    - From every tower, cast rays up, down, left, and right until the first non-black cell.
    - A direction contributes the tower's color only when that first non-black cell belongs to a gray wall; if the ray reaches another tower first or leaves the grid, that direction does nothing.
    - For each wall, collect distinct received colors in row-major first-appearance order of the towers in the input.
    - Repaint a horizontal wall as a rectangle that keeps the wall width and extends downward one row per received color, filling the rows from top to bottom by that order.
    - Repaint a vertical wall as a rectangle that keeps the wall height and extends rightward one column per received color, filling the columns from left to right by that order.
    - Towers themselves remain unchanged.

    Key insight:
    - The puzzle is a visibility problem first and a repainting problem second: you must infer which towers are the first visible object along each cardinal ray, then use the count and order of those visible colors to choose each wall's stripe thickness and band sequence.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = g.shape

    walls = _detect_walls(g)
    signals, _saw_tower_block, _saw_boundary_miss = _collect_wall_signals(g, walls)

    out = g.copy()
    out[g == WALL] = BG
    for wall, colors in zip(walls, signals):
        cells: Tuple[Coord, ...] = wall["cells"]  # type: ignore[assignment]
        if not colors:
            for r, c in cells:
                out[r, c] = WALL
            continue

        if wall["orientation"] == "H":
            row = int(wall["row"])
            c0 = int(wall["c0"])
            c1 = int(wall["c1"])
            assert row + len(colors) <= h, "horizontal stripe expansion exceeds grid"
            for i, color in enumerate(colors):
                out[row + i, c0 : c1 + 1] = int(color)
        else:
            col = int(wall["col"])
            r0 = int(wall["r0"])
            r1 = int(wall["r1"])
            assert col + len(colors) <= w, "vertical stripe expansion exceeds grid"
            for i, color in enumerate(colors):
                out[r0 : r1 + 1, col + i] = int(color)

    tower_mask = (g != BG) & (g != WALL)
    out[tower_mask] = g[tower_mask]
    return out


def _mark_rect(mask: np.ndarray, rect: Tuple[int, int, int, int], moat: int) -> None:
    h, w = mask.shape
    r0, c0, r1, c1 = rect
    rr0 = max(0, r0 - moat)
    cc0 = max(0, c0 - moat)
    rr1 = min(h - 1, r1 + moat)
    cc1 = min(w - 1, c1 + moat)
    mask[rr0 : rr1 + 1, cc0 : cc1 + 1] = True


def _rect_is_clear(mask: np.ndarray, rect: Tuple[int, int, int, int], moat: int) -> bool:
    h, w = mask.shape
    r0, c0, r1, c1 = rect
    rr0 = max(0, r0 - moat)
    cc0 = max(0, c0 - moat)
    rr1 = min(h - 1, r1 + moat)
    cc1 = min(w - 1, c1 + moat)
    return not bool(mask[rr0 : rr1 + 1, cc0 : cc1 + 1].any())


def _paint_wall(grid: np.ndarray, wall: Wall) -> None:
    for r, c in wall["cells"]:  # type: ignore[index]
        grid[r, c] = WALL


def _build_output_mask(h: int, w: int, walls: Sequence[Wall]) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    for wall in walls:
        r0, c0, r1, c1 = wall["output_rect"]  # type: ignore[index]
        mask[r0 : r1 + 1, c0 : c1 + 1] = True
    return mask


def _place_walls(h: int, w: int, specs: Sequence[Tuple[str, int]]) -> Optional[List[Wall]]:
    reserved = np.zeros((h, w), dtype=bool)
    walls: List[Wall] = []
    for orientation, depth in specs:
        placed = False
        for _ in range(250):
            length = random.randint(3, 8)
            if orientation == "H":
                if h < depth + 6 or w < length + 6:
                    continue
                row = random.randint(2, h - depth - 3)
                c0 = random.randint(2, w - length - 3)
                c1 = c0 + length - 1
                rect = (row, c0, row + depth - 1, c1)
                if not _rect_is_clear(reserved, rect, moat=1):
                    continue
                wall: Wall = {
                    "orientation": "H",
                    "row": row,
                    "c0": c0,
                    "c1": c1,
                    "depth": depth,
                    "cells": tuple((row, c) for c in range(c0, c1 + 1)),
                    "output_rect": rect,
                }
            else:
                if h < length + 6 or w < depth + 6:
                    continue
                r0 = random.randint(2, h - length - 3)
                col = random.randint(2, w - depth - 3)
                r1 = r0 + length - 1
                rect = (r0, col, r1, col + depth - 1)
                if not _rect_is_clear(reserved, rect, moat=1):
                    continue
                wall = {
                    "orientation": "V",
                    "col": col,
                    "r0": r0,
                    "r1": r1,
                    "depth": depth,
                    "cells": tuple((r, col) for r in range(r0, r1 + 1)),
                    "output_rect": rect,
                }

            _mark_rect(reserved, rect, moat=1)
            walls.append(wall)
            placed = True
            break
        if not placed:
            return None
    return walls


def _tower_is_isolated(grid: np.ndarray, r: int, c: int, output_mask: np.ndarray) -> bool:
    h, w = grid.shape
    if output_mask[r, c] or int(grid[r, c]) != BG:
        return False
    for rr in range(max(0, r - 1), min(h, r + 2)):
        for cc in range(max(0, c - 1), min(w, c + 2)):
            if int(grid[rr, cc]) != BG:
                return False
    return True


def _iter_primary_candidates(wall: Wall, h: int, w: int) -> Iterable[Tuple[Coord, Tuple[int, int]]]:
    sides = ["up", "down", "left", "right"]
    random.shuffle(sides)

    if wall["orientation"] == "H":
        row = int(wall["row"])
        c0 = int(wall["c0"])
        c1 = int(wall["c1"])
        depth = int(wall["depth"])
        cols = list(range(c0, c1 + 1))
        random.shuffle(cols)
        left_cols = list(range(0, c0 - 1))
        right_cols = list(range(c1 + 2, w))
        up_rows = list(range(0, row - 1))
        down_rows = list(range(row + depth + 1, h))
        random.shuffle(left_cols)
        random.shuffle(right_cols)
        random.shuffle(up_rows)
        random.shuffle(down_rows)
        for side in sides:
            if side == "up":
                for col in cols:
                    for tower_row in up_rows:
                        yield (tower_row, col), (1, 0)
            elif side == "down":
                for col in cols:
                    for tower_row in down_rows:
                        yield (tower_row, col), (-1, 0)
            elif side == "left":
                for tower_col in left_cols:
                    yield (row, tower_col), (0, 1)
            else:
                for tower_col in right_cols:
                    yield (row, tower_col), (0, -1)
    else:
        col = int(wall["col"])
        r0 = int(wall["r0"])
        r1 = int(wall["r1"])
        depth = int(wall["depth"])
        rows = list(range(r0, r1 + 1))
        random.shuffle(rows)
        left_cols = list(range(0, col - 1))
        right_cols = list(range(col + depth + 1, w))
        up_rows = list(range(0, r0 - 1))
        down_rows = list(range(r1 + 2, h))
        random.shuffle(left_cols)
        random.shuffle(right_cols)
        random.shuffle(up_rows)
        random.shuffle(down_rows)
        for side in sides:
            if side == "up":
                for tower_row in up_rows:
                    yield (tower_row, col), (1, 0)
            elif side == "down":
                for tower_row in down_rows:
                    yield (tower_row, col), (-1, 0)
            elif side == "left":
                for row in rows:
                    for tower_col in left_cols:
                        yield (row, tower_col), (0, 1)
            else:
                for row in rows:
                    for tower_col in right_cols:
                        yield (row, tower_col), (0, -1)


def _corridor_cells(pos: Coord, direction: Tuple[int, int], hit: Tuple[int, int, int]) -> Set[Coord]:
    r, c = pos
    dr, dc = direction
    rr, cc = r + dr, c + dc
    cells: Set[Coord] = set()
    while (rr, cc) != (hit[0], hit[1]):
        cells.add((rr, cc))
        rr += dr
        cc += dc
    return cells


def _valid_primary_candidate(
    grid: np.ndarray,
    walls: Sequence[Wall],
    wall_lookup: Dict[Coord, int],
    wall_id: int,
    pos: Coord,
    direction: Tuple[int, int],
    color: int,
) -> Optional[Set[Coord]]:
    temp = grid.copy()
    temp[pos] = color
    target_hit = _first_nonblack(temp, pos[0], pos[1], direction[0], direction[1])
    if target_hit is None or target_hit[2] != WALL:
        return None
    if wall_lookup[(target_hit[0], target_hit[1])] != wall_id:
        return None

    for other_dir in DIR4:
        hit = _first_nonblack(temp, pos[0], pos[1], other_dir[0], other_dir[1])
        if other_dir == direction:
            continue
        if hit is not None and hit[2] == WALL:
            return None
    return _corridor_cells(pos, direction, target_hit)


def _place_primary_tower(
    grid: np.ndarray,
    walls: Sequence[Wall],
    wall_lookup: Dict[Coord, int],
    wall_id: int,
    color: int,
    output_mask: np.ndarray,
    reserved_corridors: Set[Coord],
) -> Optional[Dict[str, object]]:
    h, w = grid.shape
    for pos, direction in _iter_primary_candidates(walls[wall_id], h, w):
        r, c = pos
        if pos in reserved_corridors:
            continue
        if not _tower_is_isolated(grid, r, c, output_mask):
            continue
        corridor = _valid_primary_candidate(grid, walls, wall_lookup, wall_id, pos, direction, color)
        if corridor is None:
            continue
        grid[r, c] = color
        reserved_corridors.update(corridor)
        return {"pos": pos, "dir": direction, "color": color, "wall_id": wall_id}
    return None


def _distribute_signal_counts(wall_count: int) -> List[int]:
    total = random.randint(wall_count + 1, min(6, wall_count * 3))
    counts = [1] * wall_count
    for _ in range(total - wall_count):
        choices = [i for i, count in enumerate(counts) if count < 3]
        counts[random.choice(choices)] += 1
    return counts


def _expected_signal_sets(assignments: Sequence[Dict[str, object]], wall_count: int) -> List[Set[int]]:
    expected = [set() for _ in range(wall_count)]
    for info in assignments:
        expected[int(info["wall_id"])].add(int(info["color"]))
    return expected


def _validate_expected_sets(
    grid: np.ndarray,
    walls: Sequence[Wall],
    expected: Sequence[Set[int]],
) -> Tuple[bool, bool, bool]:
    signals, saw_tower_block, saw_boundary_miss = _collect_wall_signals(grid, walls)
    actual = [set(colors) for colors in signals]
    return actual == list(expected), saw_tower_block, saw_boundary_miss


def _try_add_blocker(
    grid: np.ndarray,
    walls: Sequence[Wall],
    primary: Sequence[Dict[str, object]],
    output_mask: np.ndarray,
    reserved_corridors: Set[Coord],
    extra_colors: Sequence[int],
    expected: Sequence[Set[int]],
) -> Optional[np.ndarray]:
    h, w = grid.shape
    tower_positions = [info["pos"] for info in primary]
    shuffled = list(range(len(primary)))
    random.shuffle(shuffled)
    for color in extra_colors:
        for idx in shuffled:
            base_r, base_c = primary[idx]["pos"]  # type: ignore[index]
            intended = primary[idx]["dir"]  # type: ignore[index]
            dirs = list(DIR4)
            random.shuffle(dirs)
            for dr, dc in dirs:
                if (dr, dc) == intended:
                    continue
                distances = list(range(2, max(h, w)))
                random.shuffle(distances)
                for dist in distances:
                    r = base_r + dr * dist
                    c = base_c + dc * dist
                    if not (0 <= r < h and 0 <= c < w):
                        continue
                    pos = (r, c)
                    if pos in reserved_corridors:
                        continue
                    if not _tower_is_isolated(grid, r, c, output_mask):
                        continue

                    clear = True
                    for step in range(1, dist):
                        rr = base_r + dr * step
                        cc = base_c + dc * step
                        if int(grid[rr, cc]) != BG:
                            clear = False
                            break
                    if not clear:
                        continue

                    temp = grid.copy()
                    temp[r, c] = color
                    valid_sets, saw_tower_block, saw_boundary_miss = _validate_expected_sets(temp, walls, expected)
                    if valid_sets and saw_tower_block and saw_boundary_miss:
                        return temp
    return None


def _gen_attempt() -> Optional[np.ndarray]:
    h = random.randint(16, 30)
    w = random.randint(16, 30)
    wall_count = random.randint(2, 4)
    counts = _distribute_signal_counts(wall_count)
    if max(counts) == 1:
        counts[random.randrange(wall_count)] += 1

    orientations = ["H", "V"] + [random.choice(("H", "V")) for _ in range(wall_count - 2)]
    specs = list(zip(orientations, counts))
    specs.sort(key=lambda item: item[1], reverse=True)
    walls = _place_walls(h, w, specs)
    if walls is None:
        return None

    grid = np.full((h, w), BG, dtype=int)
    for wall in walls:
        _paint_wall(grid, wall)

    output_mask = _build_output_mask(h, w, walls)
    wall_lookup = _wall_lookup(walls)
    total_primary = sum(counts)
    extra_pool = list(random.sample(TOWER_COLORS, total_primary + 2))
    primary_colors = extra_pool[:total_primary]
    blocker_colors = extra_pool[total_primary:]

    assignments = [wall_id for wall_id, count in enumerate(counts) for _ in range(count)]
    random.shuffle(assignments)

    reserved_corridors: Set[Coord] = set()
    primary: List[Dict[str, object]] = []
    for color, wall_id in zip(primary_colors, assignments):
        info = _place_primary_tower(grid, walls, wall_lookup, wall_id, color, output_mask, reserved_corridors)
        if info is None:
            return None
        primary.append(info)

    expected = _expected_signal_sets(primary, wall_count)
    valid_sets, saw_tower_block, saw_boundary_miss = _validate_expected_sets(grid, walls, expected)
    if not valid_sets or not saw_boundary_miss:
        return None

    if not saw_tower_block:
        blocked = _try_add_blocker(grid, walls, primary, output_mask, reserved_corridors, blocker_colors, expected)
        if blocked is None:
            return None
        grid = blocked

    valid_sets, saw_tower_block, saw_boundary_miss = _validate_expected_sets(grid, walls, expected)
    if not valid_sets or not saw_tower_block or not saw_boundary_miss:
        return None

    signals, _, _ = _collect_wall_signals(grid, walls)
    if any(len(colors) == 0 for colors in signals):
        return None
    if max(len(colors) for colors in signals) < 2:
        return None

    out = main(grid)
    if out.shape != grid.shape:
        return None
    if max(out.shape) > 30:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid signal-wall puzzle")
