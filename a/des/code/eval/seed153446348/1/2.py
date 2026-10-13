import random

import numpy as np


BG = 0
WALL = 5
LEGEND_COLORS = (2, 3, 4, 1, 6)

ROOM_SIZE_RANGE = (2, 5)
MARGIN_RANGE = (1, 3)
GEN_RETRIES = 400


def _background_color(grid: np.ndarray) -> int:
    counts = np.bincount(np.asarray(grid, dtype=int).ravel())
    return int(np.argmax(counts))


def _find_legend(grid: np.ndarray, bg: int) -> list[tuple[int, int]]:
    h, w = map(int, grid.shape)
    candidates: list[list[tuple[int, int]]] = []
    for start_r, col, step in ((0, 0, 1), (0, w - 1, 1), (h - 1, 0, -1), (h - 1, w - 1, -1)):
        if int(grid[start_r, col]) == bg:
            continue
        coords: list[tuple[int, int]] = []
        r = start_r
        while 0 <= r < h and int(grid[r, col]) != bg:
            coords.append((int(r), int(col)))
            r += step
        colors = [int(grid[rr, cc]) for rr, cc in coords]
        if len(coords) >= 2 and len(set(colors)) == len(colors):
            candidates.append(sorted(coords))
    if not candidates:
        raise ValueError("legend not found")
    return max(candidates, key=len)


def _legend_mask(shape: tuple[int, int], coords: list[tuple[int, int]]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    for r, c in coords:
        mask[r, c] = True
    return mask


def _extract_structure(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    bg = _background_color(g)
    legend_coords = _find_legend(g, bg)
    legend = [int(g[r, c]) for r, c in sorted(legend_coords)]
    lmask = _legend_mask(g.shape, legend_coords)

    non_bg = g[(g != bg) & (~lmask)]
    if non_bg.size == 0:
        raise ValueError("wall color not found")
    wall = int(np.bincount(non_bg).argmax())

    wall_mask = (g == wall) & (~lmask)
    points = np.argwhere(wall_mask)
    if points.size == 0:
        raise ValueError("no wall pixels")
    r0, c0 = points.min(axis=0)
    r1, c1 = points.max(axis=0)

    sub = wall_mask[r0 : r1 + 1, c0 : c1 + 1]
    row_counts = sub.sum(axis=1)
    col_counts = sub.sum(axis=0)
    row_threshold = max(3, (sub.shape[1] + 1) // 2)
    col_threshold = max(3, (sub.shape[0] + 1) // 2)

    hwalls = [int(r0 + i) for i, cnt in enumerate(row_counts) if int(cnt) >= row_threshold]
    vwalls = [int(c0 + j) for j, cnt in enumerate(col_counts) if int(cnt) >= col_threshold]
    if len(hwalls) < 2 or len(vwalls) < 2:
        raise ValueError("not enough wall lines")

    rooms: list[tuple[int, int, int, int]] = []
    for ra, rb in zip(hwalls, hwalls[1:]):
        if rb - ra <= 1:
            continue
        for ca, cb in zip(vwalls, vwalls[1:]):
            if cb - ca <= 1:
                continue
            interior = g[ra + 1 : rb, ca + 1 : cb]
            if interior.size and np.all(interior == bg):
                rooms.append((int(ra), int(ca), int(rb), int(cb)))

    return {
        "bg": bg,
        "wall": wall,
        "legend": legend,
        "legend_coords": legend_coords,
        "hwalls": hwalls,
        "vwalls": vwalls,
        "rooms": rooms,
    }


def _count_unit_gaps(values: np.ndarray, wall: int) -> int:
    gaps = 0
    run = 0
    for value in np.asarray(values, dtype=int):
        if int(value) == wall:
            if run == 1:
                gaps += 1
            run = 0
        else:
            run += 1
    if run == 1:
        gaps += 1
    return int(gaps)


def _room_door_count(grid: np.ndarray, room: tuple[int, int, int, int], wall: int) -> int:
    r0, c0, r1, c1 = room
    return int(
        _count_unit_gaps(grid[r0, c0 + 1 : c1], wall)
        + _count_unit_gaps(grid[r1, c0 + 1 : c1], wall)
        + _count_unit_gaps(grid[r0 + 1 : r1, c0], wall)
        + _count_unit_gaps(grid[r0 + 1 : r1, c1], wall)
    )


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each rectangular room by the color indexed by its single-pixel door count.
    """
    g = np.asarray(grid, dtype=int)
    info = _extract_structure(g)
    legend = info["legend"]
    wall = info["wall"]

    out = g.copy()
    for room in info["rooms"]:
        door_count = _room_door_count(g, room, wall)
        if 0 <= door_count < len(legend):
            r0, c0, r1, c1 = room
            out[r0 + 1 : r1, c0 + 1 : c1] = int(legend[door_count])
    return out


def _wall_positions(start: int, sizes: list[int]) -> list[int]:
    pos = [int(start)]
    for size in sizes:
        pos.append(int(pos[-1] + size + 1))
    return pos


def _build_segments(
    nrows: int,
    ncols: int,
    wall_rows: list[int],
    wall_cols: list[int],
    room_heights: list[int],
    room_widths: list[int],
) -> tuple[dict, dict, dict]:
    segments: dict = {}
    room_to_segments = {(r, c): [] for r in range(nrows) for c in range(ncols)}
    line_to_segments: dict = {}

    for i in range(nrows + 1):
        for j in range(ncols):
            sid = ("h", i, j)
            rooms = []
            if i > 0:
                rooms.append((i - 1, j))
            if i < nrows:
                rooms.append((i, j))
            segments[sid] = {
                "kind": "h",
                "row": int(wall_rows[i]),
                "start": int(wall_cols[j] + 1),
                "length": int(room_widths[j]),
                "rooms": tuple(rooms),
                "line_key": ("h", i),
            }
            line_to_segments.setdefault(("h", i), []).append(sid)
            for room in rooms:
                room_to_segments[room].append(sid)

    for i in range(nrows):
        for j in range(ncols + 1):
            sid = ("v", i, j)
            rooms = []
            if j > 0:
                rooms.append((i, j - 1))
            if j < ncols:
                rooms.append((i, j))
            segments[sid] = {
                "kind": "v",
                "col": int(wall_cols[j]),
                "start": int(wall_rows[i] + 1),
                "length": int(room_heights[i]),
                "rooms": tuple(rooms),
                "line_key": ("v", j),
            }
            line_to_segments.setdefault(("v", j), []).append(sid)
            for room in rooms:
                room_to_segments[room].append(sid)

    return segments, room_to_segments, line_to_segments


def _door_counts(segments: dict, states: dict, nrows: int, ncols: int) -> dict:
    counts = {(r, c): 0 for r in range(nrows) for c in range(ncols)}
    for sid, state in states.items():
        if state != 1:
            continue
        for room in segments[sid]["rooms"]:
            counts[room] += 1
    return counts


def _draw_walls(grid: np.ndarray, wall_rows: list[int], wall_cols: list[int]) -> None:
    left, right = wall_cols[0], wall_cols[-1]
    top, bottom = wall_rows[0], wall_rows[-1]
    for row in wall_rows:
        grid[row, left : right + 1] = WALL
    for col in wall_cols:
        grid[top : bottom + 1, col] = WALL


def _carve_gap(grid: np.ndarray, seg: dict, gap_len: int, rng: random.Random) -> None:
    start = int(seg["start"])
    length = int(seg["length"])
    gap_len = min(int(gap_len), length)
    offset = rng.randint(0, length - gap_len)
    if seg["kind"] == "h":
        row = int(seg["row"])
        grid[row, start + offset : start + offset + gap_len] = BG
    else:
        col = int(seg["col"])
        grid[start + offset : start + offset + gap_len, col] = BG


def _place_legend(grid: np.ndarray, colors: tuple[int, ...], corner: str) -> None:
    h, w = map(int, grid.shape)
    col = 0 if corner.endswith("l") else w - 1
    if corner.startswith("t"):
        rows = list(range(len(colors)))
    else:
        rows = list(range(h - len(colors), h))
    for row, color in zip(rows, colors):
        grid[int(row), int(col)] = int(color)


def _valid_layout(
    grid: np.ndarray,
    wall_rows: list[int],
    wall_cols: list[int],
    expected_counts: dict,
    legend: tuple[int, ...],
) -> bool:
    try:
        info = _extract_structure(grid)
    except Exception:
        return False
    if tuple(info["legend"]) != tuple(legend):
        return False
    if info["wall"] != WALL:
        return False
    if info["hwalls"] != wall_rows or info["vwalls"] != wall_cols:
        return False
    detected = []
    for i in range(len(wall_rows) - 1):
        for j in range(len(wall_cols) - 1):
            room = (wall_rows[i], wall_cols[j], wall_rows[i + 1], wall_cols[j + 1])
            detected.append(_room_door_count(grid, room, WALL))
    expected = [expected_counts[(i, j)] for i in range(len(wall_rows) - 1) for j in range(len(wall_cols) - 1)]
    if detected != expected:
        return False

    solved = main(grid)
    for i in range(len(wall_rows) - 1):
        for j in range(len(wall_cols) - 1):
            r0, c0, r1, c1 = wall_rows[i], wall_cols[j], wall_rows[i + 1], wall_cols[j + 1]
            fill = int(legend[expected_counts[(i, j)]])
            if not np.all(solved[r0 + 1 : r1, c0 + 1 : c1] == fill):
                return False
    return True


def _generate_attempt(rng: random.Random) -> np.ndarray | None:
    target_k = rng.choice((2, 3, 4))
    legend = LEGEND_COLORS[: target_k + 1]
    corner = rng.choice(("tl", "tr", "bl", "br"))

    layouts = {
        2: ((2, 3), (3, 2), (3, 3)),
        3: ((2, 3), (3, 2), (3, 3)),
        4: ((3, 3),),
    }
    nrows, ncols = rng.choice(layouts[target_k])
    room_heights = [rng.randint(*ROOM_SIZE_RANGE) for _ in range(nrows)]
    room_widths = [rng.randint(*ROOM_SIZE_RANGE) for _ in range(ncols)]

    top_margin = rng.randint(*MARGIN_RANGE)
    bottom_margin = rng.randint(*MARGIN_RANGE)
    left_margin = rng.randint(*MARGIN_RANGE)
    right_margin = rng.randint(*MARGIN_RANGE)
    if corner.endswith("l"):
        left_margin = max(left_margin, 2)
    else:
        right_margin = max(right_margin, 2)

    maze_h = sum(room_heights) + nrows + 1
    maze_w = sum(room_widths) + ncols + 1
    h = top_margin + maze_h + bottom_margin
    w = left_margin + maze_w + right_margin

    wall_rows = _wall_positions(top_margin, room_heights)
    wall_cols = _wall_positions(left_margin, room_widths)

    segments, room_to_segments, line_to_segments = _build_segments(
        nrows, ncols, wall_rows, wall_cols, room_heights, room_widths
    )

    corners = [(0, 0), (0, ncols - 1), (nrows - 1, 0), (nrows - 1, ncols - 1)]
    high_room = rng.choice(corners)
    low_room = (nrows - 1 - high_room[0], ncols - 1 - high_room[1])

    states = {sid: None for sid in segments}
    for sid in room_to_segments[low_room]:
        states[sid] = 0

    high_segments = list(room_to_segments[high_room])
    rng.shuffle(high_segments)
    for sid in high_segments[:target_k]:
        if states[sid] == 0:
            return None
        states[sid] = 1
    for sid in high_segments[target_k:]:
        if states[sid] is None:
            states[sid] = 0

    door_prob = {2: 0.28, 3: 0.34, 4: 0.38}[target_k]
    for sid in segments:
        if states[sid] is None:
            states[sid] = 1 if rng.random() < door_prob else 0

    counts = _door_counts(segments, states, nrows, ncols)
    values = list(counts.values())
    if max(values) != target_k or any(v > target_k for v in values):
        return None
    if set(range(target_k + 1)) - set(values):
        return None

    wide_gap_lengths: dict = {}
    if rng.random() < 0.5:
        blocked_rooms = {high_room, low_room}
        candidates = []
        for sid, seg in segments.items():
            if states[sid] != 0 or int(seg["length"]) < 2:
                continue
            if any(room in blocked_rooms for room in seg["rooms"]):
                continue
            same_line = line_to_segments[seg["line_key"]]
            if any(states[other] == 1 for other in same_line):
                continue
            candidates.append(sid)
        if candidates:
            sid = rng.choice(candidates)
            wide_gap_lengths[sid] = rng.randint(2, min(3, int(segments[sid]["length"])))

    grid = np.full((h, w), BG, dtype=int)
    _draw_walls(grid, wall_rows, wall_cols)
    for sid, state in states.items():
        if state == 1:
            _carve_gap(grid, segments[sid], 1, rng)
    for sid, gap_len in wide_gap_lengths.items():
        _carve_gap(grid, segments[sid], gap_len, rng)
    _place_legend(grid, legend, corner)

    if not _valid_layout(grid, wall_rows, wall_cols, counts, legend):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _generate_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid room-and-doors puzzle")
