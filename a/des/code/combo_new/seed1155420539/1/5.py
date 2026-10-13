from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
WALL = 5
ARC_COLORS = tuple(c for c in range(10) if c not in (BG, WALL))
NEIGHBORS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
GEN_ATTEMPTS = 300


def _room_components(grid: np.ndarray) -> tuple[np.ndarray, list[dict[str, int]]]:
    """Label enclosed non-wall components and summarize their vote color/count."""
    h, w = grid.shape
    wall = grid == WALL
    seen = np.zeros((h, w), dtype=bool)
    labels = np.full((h, w), -1, dtype=int)
    rooms: list[dict[str, int]] = []

    for sr in range(h):
        for sc in range(w):
            if wall[sr, sc] or seen[sr, sc]:
                continue

            q = deque([(sr, sc)])
            seen[sr, sc] = True
            coords: list[tuple[int, int]] = []
            touches_edge = False

            while q:
                r, c = q.popleft()
                coords.append((r, c))
                if r == 0 or c == 0 or r == h - 1 or c == w - 1:
                    touches_edge = True
                for dr, dc in NEIGHBORS4:
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < h and 0 <= cc < w and not wall[rr, cc] and not seen[rr, cc]:
                        seen[rr, cc] = True
                        q.append((rr, cc))

            if touches_edge:
                continue

            votes = [int(grid[r, c]) for r, c in coords if int(grid[r, c]) not in (BG, WALL)]
            assert votes, "each enclosed room must contain colored vote pixels"
            vote_colors = set(votes)
            assert len(vote_colors) == 1, "all vote pixels in a room must have one color"

            room_id = len(rooms)
            for r, c in coords:
                labels[r, c] = room_id
            rooms.append({"color": int(votes[0]), "count": int(len(votes))})

    assert rooms, "expected at least one enclosed room"
    return labels, rooms


def _junction_mask(wall: np.ndarray) -> np.ndarray:
    h, w = wall.shape
    junctions = np.zeros((h, w), dtype=bool)
    for r in range(h):
        for c in range(w):
            if not wall[r, c]:
                continue
            degree = 0
            for dr, dc in NEIGHBORS4:
                rr, cc = r + dr, c + dc
                if 0 <= rr < h and 0 <= cc < w and wall[rr, cc]:
                    degree += 1
            junctions[r, c] = degree >= 3
    return junctions


def _winning_color(room_ids: set[int], rooms: list[dict[str, int]]) -> int | None:
    ids = sorted(room_ids)
    if not ids:
        return None
    assert len(ids) <= 2, "a straight wall segment may touch at most two rooms"
    if len(ids) == 1:
        return int(rooms[ids[0]]["color"])

    a, b = ids
    ca = int(rooms[a]["count"])
    cb = int(rooms[b]["count"])
    if ca == cb:
        return WALL
    return int(rooms[a]["color"] if ca > cb else rooms[b]["color"])


def _color_segment(
    out: np.ndarray,
    labels: np.ndarray,
    rooms: list[dict[str, int]],
    coords: list[tuple[int, int]],
    horizontal: bool,
) -> None:
    h, w = labels.shape
    room_ids: set[int] = set()

    for r, c in coords:
        if horizontal:
            neighbors = ((r - 1, c), (r + 1, c))
        else:
            neighbors = ((r, c - 1), (r, c + 1))
        for rr, cc in neighbors:
            if 0 <= rr < h and 0 <= cc < w and labels[rr, cc] >= 0:
                room_ids.add(int(labels[rr, cc]))

    color = _winning_color(room_ids, rooms)
    if color is None:
        return
    for r, c in coords:
        out[r, c] = color


def _paint_horizontal_segments(out: np.ndarray, wall: np.ndarray, junctions: np.ndarray, labels: np.ndarray, rooms: list[dict[str, int]]) -> None:
    h, w = wall.shape
    for r in range(h):
        c = 0
        while c < w:
            if not wall[r, c]:
                c += 1
                continue

            c0 = c
            while c < w and wall[r, c]:
                c += 1
            c1 = c - 1
            if c1 - c0 + 1 < 2:
                continue

            k = c0
            while k <= c1:
                while k <= c1 and junctions[r, k]:
                    k += 1
                start = k
                while k <= c1 and not junctions[r, k]:
                    k += 1
                end = k - 1
                if start <= end:
                    _color_segment(out, labels, rooms, [(r, cc) for cc in range(start, end + 1)], True)


def _paint_vertical_segments(out: np.ndarray, wall: np.ndarray, junctions: np.ndarray, labels: np.ndarray, rooms: list[dict[str, int]]) -> None:
    h, w = wall.shape
    for c in range(w):
        r = 0
        while r < h:
            if not wall[r, c]:
                r += 1
                continue

            r0 = r
            while r < h and wall[r, c]:
                r += 1
            r1 = r - 1
            if r1 - r0 + 1 < 2:
                continue

            k = r0
            while k <= r1:
                while k <= r1 and junctions[k, c]:
                    k += 1
                start = k
                while k <= r1 and not junctions[k, c]:
                    k += 1
                end = k - 1
                if start <= end:
                    _color_segment(out, labels, rooms, [(rr, c) for rr in range(start, end + 1)], False)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor every grey wall segment by the vote count of its adjacent rectangular rooms.

    Input structure:
    - The background is black and the walls are one-pixel-wide grey horizontal
      and vertical lines forming closed rectangular rooms.
    - Each enclosed room contains one or more non-black, non-grey vote pixels.
    - All vote pixels in a room have the same color, while different rooms may
      share a color and have different vote counts.

    Rule:
    - Find each enclosed non-wall room and record its vote color and number of
      vote pixels.
    - Split the grey wall network into straight wall segments between grey
      junction pixels; junction pixels touching three or more wall directions
      remain grey.
    - A segment touching one room takes that room's vote color. A segment shared
      by two rooms takes the color of the room with more vote pixels, and remains
      grey if the two counts tie.
    - Clear all vote pixels from the room interiors back to black.

    Key insight:
    The colored pixels are indirect votes for nearby walls, and shared walls are
    decided by comparing the two adjacent rooms rather than by copying pixels in
    place.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    wall = g == WALL
    labels, rooms = _room_components(g)
    junctions = _junction_mask(wall)

    out = np.where(wall, WALL, BG).astype(int)
    _paint_horizontal_segments(out, wall, junctions, labels, rooms)
    _paint_vertical_segments(out, wall, junctions, labels, rooms)
    out[junctions] = WALL
    return out


def _wall_positions(margin_before: int, sizes: list[int]) -> list[int]:
    positions = [margin_before]
    pos = margin_before
    for size in sizes:
        pos += size + 1
        positions.append(pos)
    return positions


def _draw_wall_grid(
    room_heights: list[int],
    room_widths: list[int],
    margins: tuple[int, int, int, int],
) -> tuple[np.ndarray, list[int], list[int]]:
    top, bottom, left, right = margins
    wall_rows = _wall_positions(top, room_heights)
    wall_cols = _wall_positions(left, room_widths)
    height = wall_rows[-1] + 1 + bottom
    width = wall_cols[-1] + 1 + right

    grid = np.full((height, width), BG, dtype=int)
    r0, r1 = wall_rows[0], wall_rows[-1]
    c0, c1 = wall_cols[0], wall_cols[-1]
    for r in wall_rows:
        grid[r, c0 : c1 + 1] = WALL
    for c in wall_cols:
        grid[r0 : r1 + 1, c] = WALL
    return grid, wall_rows, wall_cols


def _room_boxes(wall_rows: list[int], wall_cols: list[int]) -> list[tuple[int, int, int, int]]:
    boxes = []
    for i in range(len(wall_rows) - 1):
        for j in range(len(wall_cols) - 1):
            boxes.append((wall_rows[i] + 1, wall_cols[j] + 1, wall_rows[i + 1] - 1, wall_cols[j + 1] - 1))
    return boxes


def _place_votes(grid: np.ndarray, boxes: list[tuple[int, int, int, int]]) -> None:
    room_count = len(boxes)
    max_area = min((r1 - r0 + 1) * (c1 - c0 + 1) for r0, c0, r1, c1 in boxes)
    count_pool = list(range(1, min(max_area, 14) + 1))
    if len(count_pool) < room_count:
        raise ValueError("rooms are too small for unique vote counts")
    counts = random.sample(count_pool, room_count)
    random.shuffle(counts)

    color_count = random.randint(2, min(5, len(ARC_COLORS), room_count))
    palette = random.sample(list(ARC_COLORS), color_count)
    colors = [random.choice(palette) for _ in range(room_count)]
    if len(set(colors)) == 1 and room_count > 1:
        colors[random.randrange(room_count)] = random.choice([c for c in ARC_COLORS if c != colors[0]])

    for box, color, count in zip(boxes, colors, counts):
        r0, c0, r1, c1 = box
        cells = [(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)]
        for r, c in random.sample(cells, count):
            grid[r, c] = int(color)


def _gen_attempt() -> np.ndarray | None:
    room_rows, room_cols = random.choice(((2, 2), (2, 3), (3, 2), (3, 3)))
    room_heights = [random.randint(4, 7) for _ in range(room_rows)]
    room_widths = [random.randint(4, 7) for _ in range(room_cols)]
    margins = tuple(random.randint(0, 2) for _ in range(4))

    grid, wall_rows, wall_cols = _draw_wall_grid(room_heights, room_widths, margins)
    if max(grid.shape) > 30:
        return None

    boxes = _room_boxes(wall_rows, wall_cols)
    try:
        _place_votes(grid, boxes)
    except ValueError:
        return None

    try:
        out = main(grid)
    except Exception:
        return None
    if out.shape != grid.shape or max(out.shape) > 30:
        return None
    if np.array_equal(out, grid):
        return None
    if np.any((out != BG) & (out != WALL) & (grid != WALL)):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid seed1155420539 task 5 input")
