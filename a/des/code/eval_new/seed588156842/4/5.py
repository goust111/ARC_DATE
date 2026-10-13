from __future__ import annotations

import random
from collections import Counter

import numpy as np


BLACK = 0
GREY = 5
TOWER_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
MAX_GENERATION_ATTEMPTS = 400
GRID_SIDE_MIN = 16
GRID_SIDE_MAX = 26
EXTRA_TOWER_PROB = 0.45


def _arm_length(grid: np.ndarray, color: int, row: int, col: int, dr: int, dc: int) -> int:
    height, width = grid.shape
    length = 0
    rr, cc = row + dr, col + dc
    while 0 <= rr < height and 0 <= cc < width and int(grid[rr, cc]) == color:
        length += 1
        rr += dr
        cc += dc
    return length


def _extract_towers(grid: np.ndarray) -> list[dict[str, int]]:
    towers: list[dict[str, int]] = []
    for color in sorted(int(v) for v in np.unique(grid) if int(v) not in (BLACK, GREY)):
        coords = np.argwhere(grid == color)
        if coords.size == 0:
            continue

        row_counts = Counter(int(r) for r, _c in coords)
        col_counts = Counter(int(c) for _r, c in coords)
        center_r = max(row_counts, key=lambda r: (row_counts[r], -r))
        center_c = max(col_counts, key=lambda c: (col_counts[c], -c))

        assert int(grid[center_r, center_c]) == color
        assert all(int(r) == center_r or int(c) == center_c for r, c in coords)

        up = _arm_length(grid, color, center_r, center_c, -1, 0)
        down = _arm_length(grid, color, center_r, center_c, 1, 0)
        left = _arm_length(grid, color, center_r, center_c, 0, -1)
        right = _arm_length(grid, color, center_r, center_c, 0, 1)

        assert len(coords) == up + down + left + right + 1
        towers.append(
            {
                "color": color,
                "row": center_r,
                "col": center_c,
                "up": up,
                "down": down,
                "left": left,
                "right": right,
                "r0": center_r - up,
                "r1": center_r + down,
                "c0": center_c - left,
                "c1": center_c + right,
            }
        )
    return towers


def _directional_strength(tower: dict[str, int], row: int, col: int) -> int:
    vertical = None
    if row < tower["row"]:
        vertical = tower["up"]
    elif row > tower["row"]:
        vertical = tower["down"]

    horizontal = None
    if col < tower["col"]:
        horizontal = tower["left"]
    elif col > tower["col"]:
        horizontal = tower["right"]

    if vertical is None:
        assert horizontal is not None
        return horizontal
    if horizontal is None:
        return vertical
    return min(vertical, horizontal)


def _solve(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    towers = _extract_towers(g)
    out = g.copy()
    black_pixels = np.argwhere(g == BLACK)

    for row_raw, col_raw in black_pixels:
        row = int(row_raw)
        col = int(col_raw)
        covering = [
            tower
            for tower in towers
            if tower["r0"] <= row <= tower["r1"] and tower["c0"] <= col <= tower["c1"]
        ]
        if not covering:
            continue
        if len(covering) == 1:
            out[row, col] = covering[0]["color"]
            continue

        scored = [
            (_directional_strength(tower, row, col), tower["color"])
            for tower in covering
        ]
        best_strength = max(score for score, _color in scored)
        winners = [color for score, color in scored if score == best_strength]
        if len(winners) == 1:
            out[row, col] = winners[0]

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each background area with the color of the signal tower that covers it most strongly.

    Input structure:
    - The background is black.
    - Each non-grey foreground color is a single monochrome cross-shaped signal tower:
      one center cell with one contiguous arm in each cardinal direction.
    - Grey pixels are walls and may appear as single cells or short line segments.

    Rule:
    - Read every tower's visible up/down/left/right arm lengths from its center cell.
    - Those four reaches define the tower's coverage rectangle.
    - For each black pixel inside exactly one coverage rectangle, paint it with that tower's color.
    - For overlap pixels, compare each tower's directional reach toward that pixel:
      on a shared row or column use the corresponding arm length, and in a quadrant use
      the limiting reach given by the smaller of the relevant vertical and horizontal arms.
      Paint the unique strongest tower; if the best strengths tie, keep the pixel black.
    - Keep grey walls and the original tower pixels unchanged.

    Key insight:
    The cross itself encodes two things at once: its arm lengths define the rectangle that
    can be painted, and the same directional arm lengths resolve which tower wins in overlaps.
    """
    return _solve(grid)


def _cross_points(tower: dict[str, int]) -> set[tuple[int, int]]:
    row = tower["row"]
    col = tower["col"]
    points = {(row, col)}
    for step in range(1, tower["up"] + 1):
        points.add((row - step, col))
    for step in range(1, tower["down"] + 1):
        points.add((row + step, col))
    for step in range(1, tower["left"] + 1):
        points.add((row, col - step))
    for step in range(1, tower["right"] + 1):
        points.add((row, col + step))
    return points


def _draw_grid(height: int, width: int, towers: list[dict[str, int]], walls: set[tuple[int, int]]) -> np.ndarray:
    grid = np.full((height, width), BLACK, dtype=int)
    for row, col in walls:
        grid[row, col] = GREY
    for tower in towers:
        for row, col in _cross_points(tower):
            assert grid[row, col] == BLACK
            grid[row, col] = tower["color"]
    return grid


def _branch_stats(grid: np.ndarray) -> Counter:
    towers = _extract_towers(grid)
    stats: Counter[str] = Counter()
    for row_raw, col_raw in np.argwhere(grid == BLACK):
        row = int(row_raw)
        col = int(col_raw)
        covering = [
            tower
            for tower in towers
            if tower["r0"] <= row <= tower["r1"] and tower["c0"] <= col <= tower["c1"]
        ]
        if not covering:
            stats["uncovered"] += 1
        elif len(covering) == 1:
            stats["single"] += 1
        else:
            strengths = [_directional_strength(tower, row, col) for tower in covering]
            best = max(strengths)
            if sum(strength == best for strength in strengths) == 1:
                stats["winner"] += 1
            else:
                stats["tie"] += 1
    return stats


def _mode_ok(grid: np.ndarray, mode: str) -> bool:
    stats = _branch_stats(grid)
    if stats["single"] < 8:
        return False
    if stats["winner"] + stats["tie"] < 4:
        return False
    if stats["uncovered"] < 12:
        return False
    if mode == "winner":
        return stats["winner"] >= 4
    if mode == "tie":
        return stats["tie"] >= 4
    raise ValueError(mode)


def _arms_with_min(target: int) -> tuple[int, int]:
    extra = random.randint(0, 2)
    if random.random() < 0.5:
        return target, target + extra
    return target + extra, target


def _map_arms(toward_vertical: int, toward_horizontal: int, away_vertical: int, away_horizontal: int, row_sign: int, col_sign: int) -> tuple[int, int, int, int]:
    if row_sign > 0:
        up, down = away_vertical, toward_vertical
    else:
        up, down = toward_vertical, away_vertical

    if col_sign > 0:
        left, right = away_horizontal, toward_horizontal
    else:
        left, right = toward_horizontal, away_horizontal
    return up, down, left, right


def _make_pair(mode: str, height: int, width: int, colors: tuple[int, int]) -> list[dict[str, int]] | None:
    for _ in range(120):
        row_sign = random.choice((-1, 1))
        col_sign = random.choice((-1, 1))

        if mode == "tie":
            strength_a = strength_b = random.randint(2, 4)
        elif mode == "winner":
            low = random.randint(2, 3)
            high = low + random.randint(1, 2)
            if random.random() < 0.5:
                strength_a, strength_b = high, low
            else:
                strength_a, strength_b = low, high
        else:
            raise ValueError(mode)

        a_vertical, a_horizontal = _arms_with_min(strength_a)
        b_vertical, b_horizontal = _arms_with_min(strength_b)
        overlap_h = random.randint(2, min(4, a_vertical))
        overlap_w = random.randint(2, min(4, a_horizontal))

        row_gap = b_vertical + a_vertical - overlap_h + 1
        col_gap = b_horizontal + a_horizontal - overlap_w + 1

        a_up, a_down, a_left, a_right = _map_arms(
            toward_vertical=a_vertical,
            toward_horizontal=a_horizontal,
            away_vertical=random.randint(1, 4),
            away_horizontal=random.randint(1, 4),
            row_sign=row_sign,
            col_sign=col_sign,
        )
        b_up, b_down, b_left, b_right = _map_arms(
            toward_vertical=b_vertical,
            toward_horizontal=b_horizontal,
            away_vertical=random.randint(1, 4),
            away_horizontal=random.randint(1, 4),
            row_sign=-row_sign,
            col_sign=-col_sign,
        )

        row_choices = []
        for row_a in range(height):
            row_b = row_a + row_sign * row_gap
            if not (a_up <= row_a < height - a_down):
                continue
            if not (0 <= row_b - b_up and row_b + b_down < height):
                continue
            row_choices.append(row_a)

        col_choices = []
        for col_a in range(width):
            col_b = col_a + col_sign * col_gap
            if not (a_left <= col_a < width - a_right):
                continue
            if not (0 <= col_b - b_left and col_b + b_right < width):
                continue
            col_choices.append(col_a)

        if not row_choices or not col_choices:
            continue

        row_a = random.choice(row_choices)
        col_a = random.choice(col_choices)
        row_b = row_a + row_sign * row_gap
        col_b = col_a + col_sign * col_gap

        towers = [
            {
                "color": colors[0],
                "row": row_a,
                "col": col_a,
                "up": a_up,
                "down": a_down,
                "left": a_left,
                "right": a_right,
            },
            {
                "color": colors[1],
                "row": row_b,
                "col": col_b,
                "up": b_up,
                "down": b_down,
                "left": b_left,
                "right": b_right,
            },
        ]

        points0 = _cross_points(towers[0])
        points1 = _cross_points(towers[1])
        if points0 & points1:
            continue
        return towers
    return None


def _sample_extra_tower(height: int, width: int, color: int) -> dict[str, int]:
    up = random.randint(1, 5)
    down = random.randint(1, 5)
    left = random.randint(1, 5)
    right = random.randint(1, 5)
    row = random.randint(up, height - down - 1)
    col = random.randint(left, width - right - 1)
    return {
        "color": color,
        "row": row,
        "col": col,
        "up": up,
        "down": down,
        "left": left,
        "right": right,
    }


def _can_add_tower(tower: dict[str, int], towers: list[dict[str, int]]) -> bool:
    new_points = _cross_points(tower)
    for existing in towers:
        if new_points & _cross_points(existing):
            return False
    return True


def _wall_anchor(tower: dict[str, int], direction: str) -> tuple[int, int]:
    row = tower["row"]
    col = tower["col"]
    if direction == "up":
        return row - tower["up"] - 1, col
    if direction == "down":
        return row + tower["down"] + 1, col
    if direction == "left":
        return row, col - tower["left"] - 1
    if direction == "right":
        return row, col + tower["right"] + 1
    raise ValueError(direction)


def _wall_segment(height: int, width: int, tower: dict[str, int], direction: str, occupied: set[tuple[int, int]]) -> set[tuple[int, int]]:
    row, col = _wall_anchor(tower, direction)
    if not (0 <= row < height and 0 <= col < width):
        return set()

    if direction in {"up", "down"}:
        candidates = []
        for radius in random.sample((1, 0), 2):
            cells = {(row, col + dc) for dc in range(-radius, radius + 1)}
            if all(0 <= cc < width for _rr, cc in cells) and not (cells & occupied):
                candidates.append(cells)
        return candidates[0] if candidates else set()

    candidates = []
    for radius in random.sample((1, 0), 2):
        cells = {(row + dr, col) for dr in range(-radius, radius + 1)}
        if all(0 <= rr < height for rr, _cc in cells) and not (cells & occupied):
            candidates.append(cells)
    return candidates[0] if candidates else set()


def _place_walls(height: int, width: int, towers: list[dict[str, int]]) -> set[tuple[int, int]]:
    occupied = set().union(*(_cross_points(tower) for tower in towers))
    walls: set[tuple[int, int]] = set()
    directions = ["up", "down", "left", "right"]

    for tower in towers:
        random.shuffle(directions)
        for direction in directions:
            if random.random() < 0.65:
                segment = _wall_segment(height, width, tower, direction, occupied | walls)
                if segment:
                    walls |= segment

    if walls:
        return walls

    for tower in towers:
        for direction in ("up", "down", "left", "right"):
            segment = _wall_segment(height, width, tower, direction, occupied | walls)
            if segment:
                walls |= segment
                return walls
    return walls


def _gen_attempt() -> np.ndarray | None:
    mode = random.choice(("winner", "tie"))
    height = random.randint(GRID_SIDE_MIN, GRID_SIDE_MAX)
    width = random.randint(GRID_SIDE_MIN, GRID_SIDE_MAX)

    color_count = 3 if random.random() < EXTRA_TOWER_PROB else 2
    colors = tuple(random.sample(TOWER_COLORS, color_count))
    pair = _make_pair(mode, height, width, colors[:2])
    if pair is None:
        return None

    towers = pair[:]
    if color_count == 3:
        for _ in range(80):
            extra = _sample_extra_tower(height, width, colors[2])
            if not _can_add_tower(extra, towers):
                continue
            candidate = towers + [extra]
            grid = _draw_grid(height, width, candidate, set())
            if _mode_ok(grid, mode):
                towers = candidate
                break

    walls = _place_walls(height, width, towers)
    if not walls:
        return None

    grid = _draw_grid(height, width, towers, walls)

    try:
        parsed = _extract_towers(grid)
    except AssertionError:
        return None
    if len(parsed) != len(towers):
        return None
    if not _mode_ok(grid, mode):
        return None

    out = _solve(grid)
    if np.array_equal(out, grid):
        return None
    if not np.any(out == GREY):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid signal-tower input")
