from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import random

import numpy as np


WALL_COLOR = 5
PALETTE = tuple(range(10))
MAX_GENERATION_RETRIES = 500
CHAIN_PATTERNS = (
    ("P", "E", "E"),
    ("E", "E", "P"),
    ("P", "E", "E", "P"),
    ("P", "E", "E", "P", "P", "E", "E"),
    ("E", "E", "P", "P", "E", "E", "P"),
)


@dataclass(frozen=True)
class Room:
    row_idx: int
    col_idx: int
    r0: int
    r1: int
    c0: int
    c1: int
    pattern_cells: tuple[tuple[int, int, int], ...]

    @property
    def is_empty(self) -> bool:
        return not self.pattern_cells


@dataclass(frozen=True)
class Layout:
    bg: int
    wall: int
    wall_rows: tuple[int, ...]
    wall_cols: tuple[int, ...]
    rooms: dict[tuple[int, int], Room]


def _largest_gap_threshold(counts: np.ndarray) -> float:
    values = sorted(set(map(int, counts.tolist())))
    assert len(values) >= 2, "expected at least two wall-count levels"
    best_gap = -1
    threshold = 0.0
    for left, right in zip(values, values[1:]):
        gap = right - left
        if gap > best_gap:
            best_gap = gap
            threshold = (left + right) / 2.0
    return threshold


def _line_indices(counts: np.ndarray) -> tuple[int, ...]:
    threshold = _largest_gap_threshold(counts)
    lines = tuple(int(i) for i, count in enumerate(counts.tolist()) if float(count) > threshold)
    assert len(lines) >= 2
    return lines


def _analyze_layout(grid: np.ndarray) -> Layout:
    g = np.asarray(grid, dtype=int)
    bg = int(Counter(map(int, g.flatten())).most_common(1)[0][0])

    non_bg_counts = Counter(map(int, g.flatten()))
    non_bg_counts.pop(bg)
    assert non_bg_counts, "expected a wall color"
    wall = int(max(non_bg_counts.items(), key=lambda item: item[1])[0])

    wall_mask = g == wall
    wall_rows = _line_indices(wall_mask.sum(axis=1))
    wall_cols = _line_indices(wall_mask.sum(axis=0))
    assert wall_rows[0] == 0 and wall_rows[-1] == g.shape[0] - 1
    assert wall_cols[0] == 0 and wall_cols[-1] == g.shape[1] - 1

    rooms: dict[tuple[int, int], Room] = {}
    for row_idx, (top, bottom) in enumerate(zip(wall_rows, wall_rows[1:])):
        if bottom - top <= 1:
            continue
        for col_idx, (left, right) in enumerate(zip(wall_cols, wall_cols[1:])):
            if right - left <= 1:
                continue
            sub = g[top + 1 : bottom, left + 1 : right]
            if sub.size == 0 or np.any(sub == wall):
                continue
            pattern_cells = []
            for rr in range(top + 1, bottom):
                for cc in range(left + 1, right):
                    color = int(g[rr, cc])
                    if color not in (bg, wall):
                        pattern_cells.append((rr, cc, color))
            rooms[(row_idx, col_idx)] = Room(
                row_idx=row_idx,
                col_idx=col_idx,
                r0=top + 1,
                r1=bottom - 1,
                c0=left + 1,
                c1=right - 1,
                pattern_cells=tuple(pattern_cells),
            )
    assert rooms, "no rooms found"
    return Layout(bg=bg, wall=wall, wall_rows=wall_rows, wall_cols=wall_cols, rooms=rooms)


def _vertical_door(grid: np.ndarray, bg: int, wall_col: int, r0: int, r1: int) -> tuple[int, int] | None:
    rows = [r for r in range(r0, r1 + 1) if int(grid[r, wall_col]) == bg]
    if not rows:
        return None
    assert len(rows) == 1, "expected exactly one door on a vertical wall segment"
    return (int(rows[0]), int(wall_col))


def _horizontal_door(grid: np.ndarray, bg: int, wall_row: int, c0: int, c1: int) -> tuple[int, int] | None:
    cols = [c for c in range(c0, c1 + 1) if int(grid[wall_row, c]) == bg]
    if not cols:
        return None
    assert len(cols) == 1, "expected exactly one door on a horizontal wall segment"
    return (int(wall_row), int(cols[0]))


def _neighbors_with_doors(grid: np.ndarray, layout: Layout, room: Room) -> list[tuple[str, Room, tuple[int, int]]]:
    out: list[tuple[str, Room, tuple[int, int]]] = []

    right = layout.rooms.get((room.row_idx, room.col_idx + 1))
    if right is not None:
        door = _vertical_door(grid, layout.bg, layout.wall_cols[room.col_idx + 1], max(room.r0, right.r0), min(room.r1, right.r1))
        if door is not None:
            out.append(("right", right, door))

    left = layout.rooms.get((room.row_idx, room.col_idx - 1))
    if left is not None:
        door = _vertical_door(grid, layout.bg, layout.wall_cols[room.col_idx], max(room.r0, left.r0), min(room.r1, left.r1))
        if door is not None:
            out.append(("left", left, door))

    down = layout.rooms.get((room.row_idx + 1, room.col_idx))
    if down is not None:
        door = _horizontal_door(grid, layout.bg, layout.wall_rows[room.row_idx + 1], max(room.c0, down.c0), min(room.c1, down.c1))
        if door is not None:
            out.append(("down", down, door))

    up = layout.rooms.get((room.row_idx - 1, room.col_idx))
    if up is not None:
        door = _horizontal_door(grid, layout.bg, layout.wall_rows[room.row_idx], max(room.c0, up.c0), min(room.c1, up.c1))
        if door is not None:
            out.append(("up", up, door))

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Mirror each room's pattern through its door into the adjacent empty room.
    """
    g = np.asarray(grid, dtype=int)
    layout = _analyze_layout(g)
    out = g.copy()

    for room in layout.rooms.values():
        if room.is_empty:
            continue

        empty_neighbors = [
            (side, neighbor, door)
            for side, neighbor, door in _neighbors_with_doors(g, layout, room)
            if neighbor.is_empty
        ]
        assert len(empty_neighbors) == 1, "each patterned room must have exactly one empty-door neighbor"
        side, target, door = empty_neighbors[0]

        for rr, cc, color in room.pattern_cells:
            if side in ("left", "right"):
                mirrored_r = rr
                mirrored_c = 2 * door[1] - cc
            else:
                mirrored_r = 2 * door[0] - rr
                mirrored_c = cc
            if target.r0 <= mirrored_r <= target.r1 and target.c0 <= mirrored_c <= target.c1:
                out[mirrored_r, mirrored_c] = color

    return out


def _connected_shape(height: int, width: int, area: int, seed: tuple[int, int], rng: random.Random) -> set[tuple[int, int]]:
    cells = {seed}
    while len(cells) < area:
        frontier = []
        for rr, cc in cells:
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = rr + dr, cc + dc
                if 0 <= nr < height and 0 <= nc < width and (nr, nc) not in cells:
                    frontier.append((nr, nc))
        if not frontier:
            break
        cells.add(rng.choice(frontier))
    return cells


def _pattern_area_limit(height: int, width: int) -> int:
    return max(2, min(height * width - 1, (height * width * 2) // 3, 12))


def _paint_pattern(
    grid: np.ndarray,
    room: Room,
    side: str,
    bg: int,
    wall: int,
    rng: random.Random,
) -> None:
    height = room.r1 - room.r0 + 1
    width = room.c1 - room.c0 + 1
    max_area = _pattern_area_limit(height, width)
    area = rng.randint(2, max_area)

    if side == "right":
        seed = (rng.randrange(height), rng.randrange(max(1, min(width, 2))))
    elif side == "left":
        seed = (rng.randrange(height), width - 1 - rng.randrange(max(1, min(width, 2))))
    elif side == "down":
        seed = (rng.randrange(max(1, min(height, 2))), rng.randrange(width))
    else:
        seed = (height - 1 - rng.randrange(max(1, min(height, 2))), rng.randrange(width))

    if rng.random() < 0.45:
        shape = _connected_shape(height, width, area, seed, rng)
    else:
        seed = (rng.randrange(height), rng.randrange(width))
        shape = _connected_shape(height, width, area, seed, rng)

    colors = [c for c in PALETTE if c not in (bg, wall)]
    rng.shuffle(colors)
    palette = colors[: rng.randint(1, min(3, len(colors)))]
    for rr, cc in shape:
        grid[room.r0 + rr, room.c0 + cc] = rng.choice(palette)


def _room_sequence_target_side(sequence: tuple[str, ...], idx: int) -> str | None:
    if sequence[idx] != "P":
        return None
    left_empty = idx > 0 and sequence[idx - 1] == "E"
    right_empty = idx + 1 < len(sequence) and sequence[idx + 1] == "E"
    assert left_empty != right_empty, "pattern room must have exactly one empty neighbor"
    return "left" if left_empty else "right"


def _build_chain_grid(rng: random.Random) -> np.ndarray:
    sequence = rng.choice(CHAIN_PATTERNS)
    horizontal = rng.random() < 0.5

    bg = rng.choice([c for c in PALETTE if c != WALL_COLOR])
    wall = WALL_COLOR

    if horizontal:
        room_height = rng.randint(6, 9)
        width_lo, width_hi = (5, 8) if len(sequence) <= 4 else (4, 5)
        room_widths = [rng.randint(width_lo, width_hi) for _ in sequence]
        total_width = sum(room_widths) + len(sequence) + 1
        if total_width > 42:
            raise ValueError("grid too wide")
        wall_rows = (0, room_height + 1)
        wall_cols = [0]
        col = 0
        for width in room_widths:
            col += width + 1
            wall_cols.append(col)
        grid = np.full((room_height + 2, total_width), bg, dtype=int)
    else:
        room_width = rng.randint(6, 9)
        height_lo, height_hi = (5, 8) if len(sequence) <= 4 else (4, 5)
        room_heights = [rng.randint(height_lo, height_hi) for _ in sequence]
        total_height = sum(room_heights) + len(sequence) + 1
        if total_height > 42:
            raise ValueError("grid too tall")
        wall_cols = (0, room_width + 1)
        wall_rows = [0]
        row = 0
        for height in room_heights:
            row += height + 1
            wall_rows.append(row)
        grid = np.full((total_height, room_width + 2), bg, dtype=int)

    grid[list(wall_rows), :] = wall
    grid[:, list(wall_cols)] = wall

    if horizontal:
        for wall_col in wall_cols[1:-1]:
            door_row = rng.randint(1, grid.shape[0] - 2)
            grid[door_row, wall_col] = bg
    else:
        for wall_row in wall_rows[1:-1]:
            door_col = rng.randint(1, grid.shape[1] - 2)
            grid[wall_row, door_col] = bg

    layout = _analyze_layout(grid)
    chain_rooms = []
    if horizontal:
        for idx in range(len(sequence)):
            chain_rooms.append(layout.rooms[(0, idx)])
    else:
        for idx in range(len(sequence)):
            chain_rooms.append(layout.rooms[(idx, 0)])

    for idx, state in enumerate(sequence):
        if state != "P":
            continue
        side = _room_sequence_target_side(sequence, idx)
        assert side is not None
        if not horizontal:
            side = {"left": "up", "right": "down"}[side]
        _paint_pattern(grid, chain_rooms[idx], side, bg, wall, rng)

    return grid


def _validate_generated_grid(grid: np.ndarray) -> bool:
    try:
        layout = _analyze_layout(grid)
        counts = Counter(map(int, grid.flatten()))
        if counts[layout.bg] != max(counts.values()):
            return False
        if any(color != layout.bg and counts[color] > counts[layout.wall] for color in counts):
            return False

        target_keys: set[tuple[int, int]] = set()
        patterned = 0
        for room in layout.rooms.values():
            neighbors = _neighbors_with_doors(grid, layout, room)
            empty_neighbors = [(side, neighbor, door) for side, neighbor, door in neighbors if neighbor.is_empty]
            if room.is_empty:
                continue
            patterned += 1
            if len(empty_neighbors) != 1:
                return False
            target_key = (empty_neighbors[0][1].row_idx, empty_neighbors[0][1].col_idx)
            if target_key in target_keys:
                return False
            target_keys.add(target_key)
        if patterned == 0:
            return False

        out = main(grid)
        if np.array_equal(out, grid):
            return False
        return True
    except (AssertionError, KeyError, ValueError):
        return False


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_RETRIES):
        try:
            grid = _build_chain_grid(rng)
        except ValueError:
            continue
        if _validate_generated_grid(grid):
            return grid
    raise RuntimeError("failed to generate a valid room-projection puzzle")
