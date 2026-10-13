import random
from collections import Counter, deque

import numpy as np


BLACK = 0
GREY = 5
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
COLOR_POOL = (1, 2, 3, 4, 6, 7, 8, 9)
GEN_RETRIES = 400
MAX_BLOCK_WIDTH = 6


def _components_mask(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for dr, dc in DIR4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(coords)
    return comps


def _touches_edge(coords: list[tuple[int, int]], shape: tuple[int, int]) -> bool:
    h, w = shape
    return any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in coords)


def _parse_sample_row(row: np.ndarray) -> dict[int, int]:
    row = np.asarray(row, dtype=int)
    mapping: dict[int, int] = {}
    c = 0
    while c < row.size:
        if int(row[c]) == BLACK:
            c += 1
            continue
        color = int(row[c])
        start = c
        while c < row.size and int(row[c]) == color:
            c += 1
        mapping[c - start] = color
    assert mapping, "expected at least one sample block"
    return mapping


def _infer_wall_color(body: np.ndarray) -> int:
    vals = body[body != BLACK]
    assert vals.size, "expected non-black wall pixels above the sample row"
    return int(Counter(vals.tolist()).most_common(1)[0][0])


def _validate_room_candidate(
    body: np.ndarray,
    comp: set[tuple[int, int]],
    r0: int,
    c0: int,
    r1: int,
    c1: int,
    wall_color: int,
) -> tuple[int, int, int, int, int] | None:
    h, w = body.shape
    if not (0 < r0 <= r1 < h - 1 and 0 < c0 <= c1 < w - 1):
        return None

    rect = {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)}
    if not rect <= comp:
        return None

    extras = comp - rect
    if not extras:
        return None

    top_cols: list[int] = []
    bottom_cols: list[int] = []
    left_rows: list[int] = []
    right_rows: list[int] = []

    for r, c in extras:
        matches = 0
        if r == r0 - 1 and c0 <= c <= c1:
            top_cols.append(c)
            matches += 1
        if r == r1 + 1 and c0 <= c <= c1:
            bottom_cols.append(c)
            matches += 1
        if c == c0 - 1 and r0 <= r <= r1:
            left_rows.append(r)
            matches += 1
        if c == c1 + 1 and r0 <= r <= r1:
            right_rows.append(r)
            matches += 1
        if matches != 1:
            return None

    for seq in (top_cols, bottom_cols, left_rows, right_rows):
        seq.sort()
        if any(b - a == 1 for a, b in zip(seq, seq[1:])):
            return None

    top_set = set(top_cols)
    bottom_set = set(bottom_cols)
    left_set = set(left_rows)
    right_set = set(right_rows)

    for c in range(c0, c1 + 1):
        if c in top_set:
            if int(body[r0 - 1, c]) != BLACK or int(body[r0 - 2, c]) != wall_color:
                return None
        elif int(body[r0 - 1, c]) != wall_color:
            return None

        if c in bottom_set:
            if int(body[r1 + 1, c]) != BLACK or int(body[r1 + 2, c]) != wall_color:
                return None
        elif int(body[r1 + 1, c]) != wall_color:
            return None

    for r in range(r0, r1 + 1):
        if r in left_set:
            if int(body[r, c0 - 1]) != BLACK or int(body[r, c0 - 2]) != wall_color:
                return None
        elif int(body[r, c0 - 1]) != wall_color:
            return None

        if r in right_set:
            if int(body[r, c1 + 1]) != BLACK or int(body[r, c1 + 2]) != wall_color:
                return None
        elif int(body[r, c1 + 1]) != wall_color:
            return None

    for rr, cc in ((r0 - 1, c0 - 1), (r0 - 1, c1 + 1), (r1 + 1, c0 - 1), (r1 + 1, c1 + 1)):
        if int(body[rr, cc]) != wall_color:
            return None

    return r0, c0, r1, c1, len(extras)


def _infer_room(body: np.ndarray, coords: list[tuple[int, int]], wall_color: int) -> tuple[int, int, int, int, int] | None:
    comp = {(int(r), int(c)) for r, c in coords}
    rows = [r for r, _ in comp]
    cols = [c for _, c in comp]
    br0, br1 = min(rows), max(rows)
    bc0, bc1 = min(cols), max(cols)

    candidates: list[tuple[int, int, int, int, int]] = []
    for top_trim in (0, 1):
        for bottom_trim in (0, 1):
            for left_trim in (0, 1):
                for right_trim in (0, 1):
                    r0 = br0 + top_trim
                    r1 = br1 - bottom_trim
                    c0 = bc0 + left_trim
                    c1 = bc1 - right_trim
                    cand = _validate_room_candidate(body, comp, r0, c0, r1, c1, wall_color)
                    if cand is not None:
                        candidates.append(cand)

    if not candidates:
        return None

    candidates.sort(
        key=lambda item: (
            -((item[2] - item[0] + 1) * (item[3] - item[1] + 1)),
            item[4],
            item[0],
            item[1],
        )
    )
    return candidates[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each walled room with the sample color whose block width matches its window count.

    Input structure:
    - The background is black.
    - Above the bottom sample row, grey wall segments enclose several rectangular rooms made of black pixels.
    - Some wall cells are replaced by isolated one-pixel black gaps acting as room windows.
    - The last row is a color legend made of separated solid blocks whose widths are 1, 2, 3, ... pixels.

    Rule:
    - Read the bottom legend and build a mapping from block width to block color.
    - For each room, identify its rectangular black interior and count the black wall-gap pixels on its boundary.
    - Find the legend block whose width equals that window count.
    - Recolor the room interior with that legend color while leaving grey walls, windows, and the legend row unchanged.

    Key insight:
    - The important hidden attribute is not the room size but the number of wall gaps, and that count indexes the legend indirectly through block width.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert g.shape[0] >= 3 and g.shape[1] >= 3

    legend = _parse_sample_row(g[-1])
    body = g[:-1]
    wall_color = _infer_wall_color(body)

    out = g.copy()
    rooms_found = 0
    for coords in _components_mask(body == BLACK):
        if _touches_edge(coords, body.shape):
            continue
        room = _infer_room(body, coords, wall_color)
        if room is None:
            continue
        r0, c0, r1, c1, window_count = room
        assert window_count in legend, f"missing legend block for width {window_count}"
        out[r0 : r1 + 1, c0 : c1 + 1] = legend[window_count]
        rooms_found += 1

    assert rooms_found, "expected at least one room"
    return out


def _room_capacity(height: int, width: int) -> int:
    return 2 * ((width + 1) // 2) + 2 * ((height + 1) // 2)


def _sample_row_width(max_width: int) -> int:
    return max_width * (max_width + 1) // 2 + (max_width - 1)


def _neighbor_conflict(
    assigned: dict[tuple[int, int], list[tuple[str, int]]],
    room_row: int,
    room_col: int,
    n_rows: int,
    n_cols: int,
    side: str,
    idx: int,
) -> bool:
    if side == "left" and room_col > 0:
        return ("right", idx) in assigned.get((room_row, room_col - 1), [])
    if side == "right" and room_col + 1 < n_cols:
        return ("left", idx) in assigned.get((room_row, room_col + 1), [])
    if side == "top" and room_row > 0:
        return ("bottom", idx) in assigned.get((room_row - 1, room_col), [])
    if side == "bottom" and room_row + 1 < n_rows:
        return ("top", idx) in assigned.get((room_row + 1, room_col), [])
    return False


def _pick_windows(
    room: dict,
    window_count: int,
    assigned: dict[tuple[int, int], list[tuple[str, int]]],
    n_rows: int,
    n_cols: int,
    rng: random.Random,
) -> list[tuple[str, int]] | None:
    side_lengths = {
        "top": room["w"],
        "bottom": room["w"],
        "left": room["h"],
        "right": room["h"],
    }
    for _ in range(300):
        chosen: list[tuple[str, int]] = []
        used = {side: set() for side in side_lengths}
        while len(chosen) < window_count:
            candidates: list[tuple[str, int]] = []
            for side, length in side_lengths.items():
                for idx in range(length):
                    if idx in used[side] or idx - 1 in used[side] or idx + 1 in used[side]:
                        continue
                    if _neighbor_conflict(assigned, room["row"], room["col"], n_rows, n_cols, side, idx):
                        continue
                    candidates.append((side, idx))
            if not candidates:
                break
            choice = rng.choice(candidates)
            chosen.append(choice)
            used[choice[0]].add(choice[1])
        if len(chosen) == window_count:
            return chosen
    return None


def _assign_all_windows(
    rooms: list[dict],
    window_counts: dict[tuple[int, int], int],
    n_rows: int,
    n_cols: int,
    rng: random.Random,
) -> dict[tuple[int, int], list[tuple[str, int]]] | None:
    order = rooms[:]
    rng.shuffle(order)
    order.sort(key=lambda room: window_counts[(room["row"], room["col"])], reverse=True)

    assigned: dict[tuple[int, int], list[tuple[str, int]]] = {}
    for room in order:
        k = window_counts[(room["row"], room["col"])]
        windows = _pick_windows(room, k, assigned, n_rows, n_cols, rng)
        if windows is None:
            return None
        assigned[(room["row"], room["col"])] = windows
    return assigned


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    n_rows = rng.randint(1, 3)
    n_cols = rng.randint(1, 3)
    if n_rows * n_cols < 2:
        return None

    row_heights = [rng.randint(2, 5) for _ in range(n_rows)]
    col_widths = [rng.randint(2, 6) for _ in range(n_cols)]

    layout_h = sum(row_heights) + 2 * (n_rows + 1)
    layout_w = sum(col_widths) + 2 * (n_cols + 1)
    if layout_h + 3 > 30:
        return None

    rooms: list[dict] = []
    caps: list[int] = []
    for i, hh in enumerate(row_heights):
        for j, ww in enumerate(col_widths):
            cap = _room_capacity(hh, ww)
            caps.append(cap)
            rooms.append({"row": i, "col": j, "h": hh, "w": ww, "cap": cap})

    max_width = min(MAX_BLOCK_WIDTH, max(caps))
    if max_width < 3:
        return None
    legend_max = rng.randint(3, max_width)
    legend_w = _sample_row_width(legend_max)

    min_grid_w = max(layout_w + 2, legend_w + 2)
    min_grid_h = layout_h + 3
    if min_grid_w > 30 or min_grid_h > 30:
        return None

    grid_w = rng.randint(min_grid_w, min(30, min_grid_w + 4))
    grid_h = rng.randint(min_grid_h, min(30, min_grid_h + 4))
    layout_top = rng.randint(1, grid_h - layout_h - 2)
    layout_left = rng.randint(1, grid_w - layout_w - 1)
    sample_start = rng.randint(1, grid_w - legend_w - 1)

    grid = np.full((grid_h, grid_w), BLACK, dtype=int)

    legend_colors = rng.sample(COLOR_POOL, legend_max)
    cur = sample_start
    for width, color in enumerate(legend_colors, start=1):
        grid[-1, cur : cur + width] = color
        cur += width
        if width != legend_max:
            cur += 1

    row_pairs: list[tuple[int, int]] = []
    cursor = layout_top
    for i in range(n_rows + 1):
        row_pairs.append((cursor, cursor + 1))
        cursor += 2
        if i < n_rows:
            cursor += row_heights[i]

    col_pairs: list[tuple[int, int]] = []
    cursor = layout_left
    for j in range(n_cols + 1):
        col_pairs.append((cursor, cursor + 1))
        cursor += 2
        if j < n_cols:
            cursor += col_widths[j]

    layout_bottom = row_pairs[-1][1]
    layout_right = col_pairs[-1][1]

    for a, b in row_pairs:
        grid[a, layout_left : layout_right + 1] = GREY
        grid[b, layout_left : layout_right + 1] = GREY
    for a, b in col_pairs:
        grid[layout_top : layout_bottom + 1, a] = GREY
        grid[layout_top : layout_bottom + 1, b] = GREY

    room_lookup: dict[tuple[int, int], dict] = {}
    for room in rooms:
        i, j = room["row"], room["col"]
        r0 = row_pairs[i][1] + 1
        r1 = row_pairs[i + 1][0] - 1
        c0 = col_pairs[j][1] + 1
        c1 = col_pairs[j + 1][0] - 1
        room["r0"] = r0
        room["r1"] = r1
        room["c0"] = c0
        room["c1"] = c1
        room_lookup[(i, j)] = room

    anchor_candidates = [room for room in rooms if room["cap"] >= legend_max]
    if not anchor_candidates:
        return None
    anchor = rng.choice(anchor_candidates)

    window_counts: dict[tuple[int, int], int] = {}
    for room in rooms:
        upper = min(legend_max, room["cap"])
        key = (room["row"], room["col"])
        if room is anchor:
            window_counts[key] = legend_max
        else:
            window_counts[key] = rng.randint(1, upper)

    if len(set(window_counts.values())) < 2:
        alternatives = [
            room
            for room in rooms
            if room is not anchor and min(legend_max, room["cap"]) >= 2
        ]
        if not alternatives:
            return None
        room = rng.choice(alternatives)
        key = (room["row"], room["col"])
        options = [k for k in range(1, min(legend_max, room["cap"]) + 1) if k != window_counts[key]]
        if not options:
            return None
        window_counts[key] = rng.choice(options)

    assignments = _assign_all_windows(rooms, window_counts, n_rows, n_cols, rng)
    if assignments is None:
        return None

    for (i, j), windows in assignments.items():
        room = room_lookup[(i, j)]
        top_wall = row_pairs[i][1]
        bottom_wall = row_pairs[i + 1][0]
        left_wall = col_pairs[j][1]
        right_wall = col_pairs[j + 1][0]
        for side, idx in windows:
            if side == "top":
                grid[top_wall, room["c0"] + idx] = BLACK
            elif side == "bottom":
                grid[bottom_wall, room["c0"] + idx] = BLACK
            elif side == "left":
                grid[room["r0"] + idx, left_wall] = BLACK
            else:
                grid[room["r0"] + idx, right_wall] = BLACK

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is None:
            continue
        try:
            out = main(grid)
        except AssertionError:
            continue
        if np.array_equal(out, grid):
            continue
        return grid
    raise RuntimeError("failed to generate a valid room-window input")
