from __future__ import annotations

import random
from typing import Iterable

import numpy as np


BG = 0
RED = 2
GREY = 5
PATTERN_COLORS = (1, 3, 4, 6, 7, 8, 9)
GEN_RETRIES = 400

DIRS = ("up", "down", "left", "right")
DIR_TO_DELTA = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}

PATTERN_TEMPLATES = [
    {(0, 0)},
    {(0, 0), (0, 1)},
    {(0, 0), (1, 0)},
    {(0, 0), (0, 1), (1, 0)},
    {(0, 0), (0, 1), (1, 1)},
    {(0, 0), (1, 0), (1, 1)},
    {(0, 1), (1, 0), (1, 1)},
    {(0, 0), (0, 1), (1, 0), (1, 1)},
    {(0, 0), (0, 1), (0, 2)},
    {(0, 0), (1, 0), (2, 0)},
    {(0, 0), (0, 1), (0, 2), (1, 1)},
    {(0, 1), (1, 0), (1, 1), (1, 2)},
    {(0, 0), (1, 0), (1, 1), (2, 1)},
    {(0, 1), (1, 1), (1, 0), (2, 0)},
    {(0, 0), (1, 0), (2, 0), (2, 1)},
    {(0, 1), (1, 1), (2, 1), (2, 0)},
]


def _room_spans(line_mask: np.ndarray) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    start: int | None = None
    for idx, is_line in enumerate(list(map(bool, line_mask)) + [True]):
        if not is_line and start is None:
            start = idx
        elif is_line and start is not None:
            spans.append((start, idx))
            start = None
    return spans


def _extract_layout(grid: np.ndarray) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    row_lines = np.all(grid == GREY, axis=1)
    col_lines = np.all(grid == GREY, axis=0)
    row_spans = _room_spans(row_lines)
    col_spans = _room_spans(col_lines)
    assert row_spans and col_spans, "expected at least one room"

    room_h = row_spans[0][1] - row_spans[0][0]
    room_w = col_spans[0][1] - col_spans[0][0]
    assert room_h >= 3 and room_w >= 3, "rooms must be large enough for arrows"
    assert all((r1 - r0) == room_h for r0, r1 in row_spans), "room heights must match"
    assert all((c1 - c0) == room_w for c0, c1 in col_spans), "room widths must match"
    return row_spans, col_spans


def _arrow_cells(direction: str, room_h: int, room_w: int) -> set[tuple[int, int]]:
    mr = room_h // 2
    mc = room_w // 2
    if direction == "up":
        return {(0, mc), (1, mc - 1), (1, mc + 1)}
    if direction == "down":
        return {(room_h - 1, mc), (room_h - 2, mc - 1), (room_h - 2, mc + 1)}
    if direction == "left":
        return {(mr, 0), (mr - 1, 1), (mr + 1, 1)}
    if direction == "right":
        return {(mr, room_w - 1), (mr - 1, room_w - 2), (mr + 1, room_w - 2)}
    raise ValueError(direction)


def _detect_arrow(room: np.ndarray) -> tuple[str | None, set[tuple[int, int]]]:
    red_cells = {tuple(map(int, coord)) for coord in np.argwhere(room == RED)}
    if not red_cells:
        return None, set()
    room_h, room_w = map(int, room.shape)
    matches = [direction for direction in DIRS if red_cells == _arrow_cells(direction, room_h, room_w)]
    assert len(matches) <= 1, "ambiguous room arrow"
    assert matches, "unexpected red pixels inside room"
    direction = matches[0]
    return direction, _arrow_cells(direction, room_h, room_w)


def _neighbors(room: tuple[int, int], n_rows: int, n_cols: int) -> list[tuple[tuple[int, int], str]]:
    r, c = room
    out: list[tuple[tuple[int, int], str]] = []
    if r > 0:
        out.append(((r - 1, c), "up"))
    if r + 1 < n_rows:
        out.append(((r + 1, c), "down"))
    if c > 0:
        out.append(((r, c - 1), "left"))
    if c + 1 < n_cols:
        out.append(((r, c + 1), "right"))
    return out


def _direction_between(src: tuple[int, int], dst: tuple[int, int]) -> str:
    dr = dst[0] - src[0]
    dc = dst[1] - src[1]
    for direction, (want_dr, want_dc) in DIR_TO_DELTA.items():
        if (dr, dc) == (want_dr, want_dc):
            return direction
    raise ValueError(f"rooms are not adjacent: {src} -> {dst}")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Copy each arrow-marked room's pattern into the adjacent room indicated by the arrow, then clear the source room.

    Input structure:
    - The background is black.
    - Grey single-pixel full rows and columns partition the grid into equal rectangular rooms.
    - Inside each room, cells are either black background, a small non-black/non-grey pattern, or a red three-pixel arrow placed on one room edge.

    Rule:
    - Recover the room row/column layout from the full grey separator lines.
    - In each room with a red edge arrow, read its direction from the three-pixel arrowhead shape.
    - Copy every non-background pattern pixel from that source room into the adjacent target room at the same relative room coordinates, then clear the entire source room contents including the arrow.
    - Apply all copies simultaneously: non-arrow rooms keep their original patterns unless overwritten by an incoming copy, and incoming copied pixels override any existing non-background target pixels.

    Key insight:
    - The red edge marker is not decoration; it is a transport instruction on top of a room grid, and the move is a simultaneous room-to-room copy using room-local coordinates.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    row_spans, col_spans = _extract_layout(g)
    room_h = row_spans[0][1] - row_spans[0][0]
    room_w = col_spans[0][1] - col_spans[0][0]

    moves: list[tuple[tuple[int, int], list[tuple[int, int, int]]]] = []
    source_rooms: set[tuple[int, int]] = set()

    for room_r, (r0, r1) in enumerate(row_spans):
        for room_c, (c0, c1) in enumerate(col_spans):
            room = g[r0:r1, c0:c1]
            direction, arrow_cells = _detect_arrow(room)
            if direction is None:
                continue

            dr, dc = DIR_TO_DELTA[direction]
            target_room = (room_r + dr, room_c + dc)
            assert 0 <= target_room[0] < len(row_spans) and 0 <= target_room[1] < len(col_spans), (
                "arrow must point to an adjacent room"
            )

            payload: list[tuple[int, int, int]] = []
            for rr in range(room_h):
                for cc in range(room_w):
                    if (rr, cc) in arrow_cells:
                        continue
                    color = int(room[rr, cc])
                    if color != BG:
                        payload.append((rr, cc, color))

            source_rooms.add((room_r, room_c))
            moves.append((target_room, payload))

    assert moves, "expected at least one arrow-marked room"

    out = g.copy()
    for room_r, room_c in source_rooms:
        r0, r1 = row_spans[room_r]
        c0, c1 = col_spans[room_c]
        out[r0:r1, c0:c1] = BG

    for (target_r, target_c), payload in moves:
        r0, _r1 = row_spans[target_r]
        c0, _c1 = col_spans[target_c]
        for rr, cc, color in payload:
            out[r0 + rr, c0 + cc] = color

    return out


def _oriented_cells(base_cells: Iterable[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    cells = [tuple(map(int, cell)) for cell in base_cells]
    if random.random() < 0.5:
        cells = [(c, r) for r, c in cells]
    max_r = max(r for r, _c in cells)
    max_c = max(c for _r, c in cells)
    if random.random() < 0.5:
        cells = [(max_r - r, c) for r, c in cells]
    if random.random() < 0.5:
        cells = [(r, max_c - c) for r, c in cells]
    min_r = min(r for r, _c in cells)
    min_c = min(c for _r, c in cells)
    norm = sorted((r - min_r, c - min_c) for r, c in cells)
    return tuple(norm)


def _sample_pattern_cells(
    room_h: int,
    room_w: int,
    forbidden: set[tuple[int, int]],
    preferred: set[tuple[int, int]] | None = None,
    require_overlap: bool = False,
) -> set[tuple[int, int]] | None:
    preferred = set() if preferred is None else set(preferred)
    for _ in range(200):
        shape = _oriented_cells(random.choice(PATTERN_TEMPLATES))
        shape_h = 1 + max(r for r, _c in shape)
        shape_w = 1 + max(c for _r, c in shape)
        if shape_h > room_h or shape_w > room_w:
            continue
        off_r = random.randint(0, room_h - shape_h)
        off_c = random.randint(0, room_w - shape_w)
        cells = {(off_r + r, off_c + c) for r, c in shape}
        if cells & forbidden:
            continue
        if require_overlap and not (cells & preferred):
            continue
        return cells

    allowed = [(r, c) for r in range(room_h) for c in range(room_w) if (r, c) not in forbidden]
    if not allowed:
        return None
    if require_overlap:
        overlap_cells = [cell for cell in preferred if cell not in forbidden]
        if not overlap_cells:
            return None
        return {random.choice(overlap_cells)}
    return {random.choice(allowed)}


def _build_layout() -> tuple[np.ndarray, list[tuple[int, int]], list[tuple[int, int]], int, int]:
    n_rows = random.randint(2, 4)
    n_cols = random.randint(2, 4)
    outer_frame = random.random() < 0.5

    row_sizes = [
        room_h
        for room_h in (5, 7, 9)
        if n_rows * room_h + (n_rows + 1 if outer_frame else n_rows - 1) <= 30
    ]
    col_sizes = [
        room_w
        for room_w in (5, 7, 9)
        if n_cols * room_w + (n_cols + 1 if outer_frame else n_cols - 1) <= 30
    ]
    assert row_sizes and col_sizes
    room_h = random.choice(row_sizes)
    room_w = random.choice(col_sizes)

    row_spans: list[tuple[int, int]] = []
    col_spans: list[tuple[int, int]] = []

    pos = 1 if outer_frame else 0
    for idx in range(n_rows):
        row_spans.append((pos, pos + room_h))
        pos += room_h
        if idx != n_rows - 1 or outer_frame:
            pos += 1
    height = pos

    pos = 1 if outer_frame else 0
    for idx in range(n_cols):
        col_spans.append((pos, pos + room_w))
        pos += room_w
        if idx != n_cols - 1 or outer_frame:
            pos += 1
    width = pos

    grid = np.full((height, width), BG, dtype=int)
    if outer_frame:
        grid[0, :] = GREY
        grid[-1, :] = GREY
        grid[:, 0] = GREY
        grid[:, -1] = GREY

    for r0, r1 in row_spans[:-1]:
        grid[r1, :] = GREY
    for c0, c1 in col_spans[:-1]:
        grid[:, c1] = GREY

    return grid, row_spans, col_spans, room_h, room_w


def _gen_attempt() -> np.ndarray | None:
    grid, row_spans, col_spans, room_h, room_w = _build_layout()
    n_rows = len(row_spans)
    n_cols = len(col_spans)
    rooms = [(r, c) for r in range(n_rows) for c in range(n_cols)]

    max_sources = min(4, len(rooms) - 1)
    source_rooms: set[tuple[int, int]] = set()
    target_rooms: set[tuple[int, int]] = set()
    incoming_count: dict[tuple[int, int], int] = {}
    moves: list[tuple[tuple[int, int], tuple[int, int], str]] = []

    if random.random() < 0.5:
        multi_target_candidates = [room for room in rooms if len(_neighbors(room, n_rows, n_cols)) >= 2]
        if multi_target_candidates:
            target = random.choice(multi_target_candidates)
            neighbor_rooms = [neighbor for neighbor, _direction in _neighbors(target, n_rows, n_cols)]
            sources = random.sample(neighbor_rooms, 2)
            for src in sources:
                direction = _direction_between(src, target)
                moves.append((src, target, direction))
                source_rooms.add(src)
            target_rooms.add(target)
            incoming_count[target] = 2

    extra_limit = max_sources - len(source_rooms)
    extra_count = random.randint(0 if source_rooms else 1, max(1, extra_limit)) if extra_limit > 0 else 0
    for _ in range(extra_count):
        options: list[tuple[tuple[int, int], list[tuple[tuple[int, int], str]]]] = []
        for src in rooms:
            if src in source_rooms or src in target_rooms:
                continue
            target_options = [
                (neighbor, direction)
                for neighbor, direction in _neighbors(src, n_rows, n_cols)
                if neighbor not in source_rooms and neighbor not in target_rooms
            ]
            if target_options:
                options.append((src, target_options))
        if not options:
            break
        src, target_options = random.choice(options)
        target, direction = random.choice(target_options)
        moves.append((src, target, direction))
        source_rooms.add(src)
        target_rooms.add(target)
        incoming_count[target] = 1

    if not moves:
        return None

    room_patterns: dict[tuple[int, int], tuple[int, set[tuple[int, int]]]] = {}
    incoming_cells: dict[tuple[int, int], set[tuple[int, int]]] = {}
    incoming_colors: dict[tuple[int, int], set[int]] = {}

    for src, target, direction in moves:
        forbidden = _arrow_cells(direction, room_h, room_w) | incoming_cells.get(target, set())
        cells = _sample_pattern_cells(room_h, room_w, forbidden)
        if not cells:
            return None
        color = random.choice(PATTERN_COLORS)
        room_patterns[src] = (color, cells)
        incoming_cells.setdefault(target, set()).update(cells)
        incoming_colors.setdefault(target, set()).add(color)

    if target_rooms and random.random() < 0.7:
        target = random.choice(list(target_rooms))
        overlap = random.random() < 0.5
        incoming_here = incoming_cells.get(target, set())
        forbidden = set() if overlap else set(incoming_here)
        cells = _sample_pattern_cells(
            room_h,
            room_w,
            forbidden=forbidden,
            preferred=incoming_here,
            require_overlap=overlap,
        )
        if not cells:
            return None
        color_choices = [color for color in PATTERN_COLORS if color not in incoming_colors.get(target, set())]
        if not color_choices:
            color_choices = list(PATTERN_COLORS)
        room_patterns[target] = (random.choice(color_choices), cells)

    other_rooms = [room for room in rooms if room not in source_rooms and room not in room_patterns]
    extra_static = random.randint(0, min(2, len(other_rooms)))
    for room in random.sample(other_rooms, extra_static):
        cells = _sample_pattern_cells(room_h, room_w, forbidden=set())
        if not cells:
            return None
        room_patterns[room] = (random.choice(PATTERN_COLORS), cells)

    for room, (color, cells) in room_patterns.items():
        r0, _r1 = row_spans[room[0]]
        c0, _c1 = col_spans[room[1]]
        for rr, cc in cells:
            grid[r0 + rr, c0 + cc] = color

    for src, _target, direction in moves:
        r0, _r1 = row_spans[src[0]]
        c0, _c1 = col_spans[src[1]]
        for rr, cc in _arrow_cells(direction, room_h, room_w):
            grid[r0 + rr, c0 + cc] = RED

    try:
        out = main(grid)
    except AssertionError:
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid room-copy ARC input")
