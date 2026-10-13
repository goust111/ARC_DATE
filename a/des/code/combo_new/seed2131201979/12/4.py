from __future__ import annotations

import random
from collections import deque

import numpy as np


BLACK = 0
GRAY = 5
PALETTE_POOL = [1, 2, 3, 4, 6, 7, 8, 9]
MAX_GENERATION_ATTEMPTS = 400


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _find_legend(bottom_row: np.ndarray) -> tuple[int, int, list[int]]:
    cols = np.flatnonzero(bottom_row != BLACK)
    assert cols.size > 0, "expected a palette on the bottom row"

    runs: list[tuple[int, int]] = []
    start = int(cols[0])
    prev = int(cols[0])
    for col in map(int, cols[1:]):
        if col != prev + 1:
            runs.append((start, prev))
            start = col
        prev = col
    runs.append((start, prev))

    pal_start, pal_end = max(runs, key=lambda run: run[1] - run[0])
    colors = [int(v) for v in bottom_row[pal_start : pal_end + 1]]
    assert len(colors) == len(set(colors)), "palette colors must be distinct"
    return pal_start, pal_end, colors


def _wall_rows_and_cols(body: np.ndarray, wall: int) -> tuple[set[int], set[int]]:
    row_counts = np.sum(body == wall, axis=1)
    col_counts = np.sum(body == wall, axis=0)
    assert row_counts.max() > 0 and col_counts.max() > 0, "expected visible walls"

    row_threshold = max(3, int(np.ceil(0.7 * int(row_counts.max()))))
    col_threshold = max(3, int(np.ceil(0.7 * int(col_counts.max()))))

    wall_rows = {int(r) for r, count in enumerate(row_counts) if int(count) >= row_threshold}
    wall_cols = {int(c) for c, count in enumerate(col_counts) if int(count) >= col_threshold}
    assert wall_rows and wall_cols, "failed to infer wall guides"
    return wall_rows, wall_cols


def _door_mask(body: np.ndarray, wall: int) -> np.ndarray:
    h, w = body.shape
    wall_rows, wall_cols = _wall_rows_and_cols(body, wall)
    doors = np.zeros((h, w), dtype=bool)

    for r in range(1, h - 1):
        for c in range(1, w - 1):
            if int(body[r, c]) != BLACK:
                continue

            is_horizontal = (
                r in wall_rows
                and c not in wall_cols
                and int(body[r, c - 1]) == wall
                and int(body[r, c + 1]) == wall
                and int(body[r - 1, c]) == BLACK
                and int(body[r + 1, c]) == BLACK
            )
            is_vertical = (
                c in wall_cols
                and r not in wall_rows
                and int(body[r - 1, c]) == wall
                and int(body[r + 1, c]) == wall
                and int(body[r, c - 1]) == BLACK
                and int(body[r, c + 1]) == BLACK
            )
            if is_horizontal or is_vertical:
                doors[r, c] = True

    assert doors.any(), "expected at least one door"
    return doors


def _black_components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each rectangular room from the bottom palette using its total door count, then color every door by the later of its two room colors.

    Input structure:
    - Background is black.
    - One-pixel-thick grey horizontal and vertical wall segments partition the upper part of the grid into rectangular rooms.
    - Doors are single black gaps inside grey wall segments, and the bottom row contains a contiguous left-to-right color legend with distinct colors.

    Rule:
    - Read the bottom-row legend in left-to-right order so its first color means "1 door", second means "2 doors", and so on.
    - Detect every door gap, temporarily restore those gaps to grey, and find the enclosed black room interiors separated by the walls.
    - Count how many detected doors touch each room, fill the room interior with the legend color at that door-count position, and then recolor each door pixel with the later legend color of the two rooms it connects.

    Key insight:
    - The door gaps must be interpreted twice: first as countable openings that choose each room's palette entry, then as shared boundary cells whose final color is determined by comparing the two adjacent room colors in legend order.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape[0] >= 2, "expected a 2D grid with a bottom palette row"

    _, _, legend = _find_legend(g[-1])
    body = g[:-1].copy()

    body_colors = {int(v) for v in np.unique(body)} - {BLACK}
    assert len(body_colors) == 1, "expected exactly one wall color above the palette"
    wall = body_colors.pop()

    doors = _door_mask(body, wall)
    sealed = body.copy()
    sealed[doors] = wall

    room_mask = sealed == BLACK
    comps = _black_components(room_mask)
    h, w = sealed.shape

    room_id = -np.ones((h, w), dtype=int)
    rooms: list[list[tuple[int, int]]] = []
    for comp in comps:
        touches_boundary = any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in comp)
        if touches_boundary:
            continue
        idx = len(rooms)
        rooms.append(comp)
        for r, c in comp:
            room_id[r, c] = idx

    assert rooms, "expected enclosed rooms"

    room_doors = [0] * len(rooms)
    door_neighbors: dict[tuple[int, int], tuple[int, int]] = {}
    for r, c in zip(*np.where(doors)):
        neighbors = {
            int(room_id[nr, nc])
            for nr, nc in _neighbors4(int(r), int(c))
            if 0 <= nr < h and 0 <= nc < w and int(room_id[nr, nc]) >= 0
        }
        assert len(neighbors) == 2, "each door must separate exactly two rooms"
        a, b = sorted(neighbors)
        room_doors[a] += 1
        room_doors[b] += 1
        door_neighbors[(int(r), int(c))] = (a, b)

    assert max(room_doors) <= len(legend), "palette is too short for the observed door counts"

    room_colors = [legend[count - 1] for count in room_doors]
    room_ranks = [count for count in room_doors]

    out = g.copy()
    for idx, comp in enumerate(rooms):
        color = room_colors[idx]
        for r, c in comp:
            out[r, c] = color

    for (r, c), (a, b) in door_neighbors.items():
        out[r, c] = room_colors[a] if room_ranks[a] >= room_ranks[b] else room_colors[b]

    return out


def _snake_rooms(rows: int, cols: int) -> list[tuple[int, int]]:
    order: list[tuple[int, int]] = []
    for r in range(rows):
        cols_iter = range(cols) if r % 2 == 0 else range(cols - 1, -1, -1)
        for c in cols_iter:
            order.append((r, c))
    return order


def _segment_positions(points: list[int], count: int, rng: random.Random) -> list[int]:
    if count == 0:
        return []
    if count == 1:
        return [rng.choice(points)]

    pairs = [(a, b) for a in points for b in points if a < b and b - a >= 2]
    assert pairs, "segment is too short for two separated doors"
    a, b = rng.choice(pairs)
    return [a, b]


def _room_graph(rows: int, cols: int):
    segments = []
    pair_to_idx: dict[frozenset[tuple[int, int]], int] = {}
    incident: dict[tuple[int, int], list[int]] = {(r, c): [] for r in range(rows) for c in range(cols)}

    for r in range(rows):
        for c in range(cols - 1):
            idx = len(segments)
            a = (r, c)
            b = (r, c + 1)
            segments.append({"rooms": (a, b), "orientation": "v", "count": 0})
            pair_to_idx[frozenset((a, b))] = idx
            incident[a].append(idx)
            incident[b].append(idx)

    for r in range(rows - 1):
        for c in range(cols):
            idx = len(segments)
            a = (r, c)
            b = (r + 1, c)
            segments.append({"rooms": (a, b), "orientation": "h", "count": 0})
            pair_to_idx[frozenset((a, b))] = idx
            incident[a].append(idx)
            incident[b].append(idx)

    return segments, pair_to_idx, incident


def _apply_segment_counts(
    grid: np.ndarray,
    row_bounds: list[int],
    col_bounds: list[int],
    segments: list[dict],
    rng: random.Random,
) -> None:
    for seg in segments:
        count = int(seg["count"])
        if count == 0:
            continue
        (r0, c0), (r1, c1) = seg["rooms"]
        if seg["orientation"] == "v":
            wall_col = col_bounds[c0 + 1]
            rows = list(range(row_bounds[r0] + 1, row_bounds[r0 + 1]))
            for rr in _segment_positions(rows, count, rng):
                grid[rr, wall_col] = BLACK
        else:
            wall_row = row_bounds[r0 + 1]
            cols = list(range(col_bounds[c0] + 1, col_bounds[c0 + 1]))
            for cc in _segment_positions(cols, count, rng):
                grid[wall_row, cc] = BLACK


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    target_max = rng.randint(3, 5)

    if target_max == 5:
        choices = [(r, c) for r in range(2, 5) for c in range(2, 5) if r * c >= 6]
        room_rows, room_cols = rng.choice(choices)
    else:
        room_rows = rng.randint(2, 4)
        room_cols = rng.randint(2, 4)

    heights = [rng.randint(3, 5) for _ in range(room_rows)]
    widths = [rng.randint(3, 5) for _ in range(room_cols)]

    layout_h = sum(heights) + room_rows + 1
    layout_w = sum(widths) + room_cols + 1

    top_margin = rng.randint(0, 2)
    left_margin = rng.randint(0, 2)
    right_margin = rng.randint(0, 2)
    gap_to_legend = rng.randint(0, 2)

    total_h = top_margin + layout_h + gap_to_legend + 1
    total_w = left_margin + layout_w + right_margin
    if total_h > 30 or total_w > 30:
        return None

    grid = np.full((total_h, total_w), BLACK, dtype=int)

    row_bounds = [top_margin]
    for height in heights:
        row_bounds.append(row_bounds[-1] + height + 1)

    col_bounds = [left_margin]
    for width in widths:
        col_bounds.append(col_bounds[-1] + width + 1)

    top = row_bounds[0]
    bottom = row_bounds[-1]
    left = col_bounds[0]
    right = col_bounds[-1]

    for r in row_bounds:
        grid[r, left : right + 1] = GRAY
    for c in col_bounds:
        grid[top : bottom + 1, c] = GRAY

    segments, pair_to_idx, incident = _room_graph(room_rows, room_cols)
    snake = _snake_rooms(room_rows, room_cols)
    for a, b in zip(snake, snake[1:]):
        seg_idx = pair_to_idx[frozenset((a, b))]
        segments[seg_idx]["count"] += 1

    counts = {(r, c): 0 for r in range(room_rows) for c in range(room_cols)}
    for seg in segments:
        if seg["count"]:
            a, b = seg["rooms"]
            counts[a] += seg["count"]
            counts[b] += seg["count"]

    focus_candidates = [
        room
        for room, seg_ids in incident.items()
        if 2 * len(seg_ids) >= target_max
    ]
    if not focus_candidates:
        return None
    focus = rng.choice(focus_candidates)

    while counts[focus] < target_max:
        options = []
        for seg_idx in incident[focus]:
            seg = segments[seg_idx]
            if seg["count"] >= 2:
                continue
            a, b = seg["rooms"]
            other = b if a == focus else a
            if counts[other] >= target_max:
                continue
            options.append(seg_idx)
        if not options:
            return None
        seg_idx = rng.choice(options)
        seg = segments[seg_idx]
        seg["count"] += 1
        a, b = seg["rooms"]
        counts[a] += 1
        counts[b] += 1

    extra_steps = rng.randint(0, max(1, len(segments) // 3))
    for _ in range(extra_steps):
        options = []
        for idx, seg in enumerate(segments):
            if seg["count"] >= 2:
                continue
            a, b = seg["rooms"]
            if counts[a] >= target_max or counts[b] >= target_max:
                continue
            options.append(idx)
        if not options:
            break
        seg_idx = rng.choice(options)
        seg = segments[seg_idx]
        seg["count"] += 1
        a, b = seg["rooms"]
        counts[a] += 1
        counts[b] += 1

    room_counts = list(counts.values())
    if min(room_counts) < 1 or max(room_counts) != target_max:
        return None
    if len(set(room_counts)) < 3 or 1 not in room_counts:
        return None

    used_orientations = {seg["orientation"] for seg in segments if seg["count"] > 0}
    if used_orientations != {"h", "v"}:
        return None

    _apply_segment_counts(grid, row_bounds, col_bounds, segments, rng)

    palette = rng.sample(PALETTE_POOL, target_max)
    start_col = rng.randint(0, total_w - target_max)
    grid[-1, start_col : start_col + target_max] = palette

    if (grid[:-1] == GRAY).sum() == 0:
        return None

    try:
        out = main(grid)
    except Exception:
        return None

    if out.shape != grid.shape:
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
