from __future__ import annotations

from collections import Counter, defaultdict, deque
import random

import numpy as np


BG = 0
WALL = 8
ARROW = 9
PALETTE = (1, 2, 3, 4, 5, 6, 7)

DIRS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}

ARROW_TAILS = {
    "up": ((1, -1), (1, 1)),
    "down": ((-1, -1), (-1, 1)),
    "left": ((-1, 1), (1, 1)),
    "right": ((-1, -1), (1, -1)),
}

ROOM_LAYOUTS = ((1, 2), (1, 3), (2, 2), (2, 3), (3, 2))
ROOM_SIZE_RANGE = (6, 9)
MAX_GRID_SIDE = 30
GEN_RETRIES = 400


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for dr, dc in DIRS.values():
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _room_slices(grid: np.ndarray) -> list[tuple[int, int, int, int]]:
    h, w = grid.shape
    rooms: list[tuple[int, int, int, int]] = []
    for comp in _components(grid != WALL):
        if any(r in (0, h - 1) or c in (0, w - 1) for r, c in comp):
            continue
        rows = [r for r, _ in comp]
        cols = [c for _, c in comp]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        if (r1 - r0 + 1) * (c1 - c0 + 1) != len(comp):
            continue
        rooms.append((r0, r1, c0, c1))
    rooms.sort()
    assert rooms, "expected at least one enclosed room"
    return rooms


def _detect_arrow(room: np.ndarray) -> tuple[str, tuple[int, int]]:
    pts = {tuple(map(int, pt)) for pt in np.argwhere(room == ARROW)}
    assert len(pts) == 3, "each room must contain exactly one 3-pixel arrow"
    matches: list[tuple[str, tuple[int, int]]] = []
    for name, tails in ARROW_TAILS.items():
        for hr, hc in pts:
            cand = {(hr, hc)}
            cand.update((hr + dr, hc + dc) for dr, dc in tails)
            if cand == pts:
                matches.append((name, (hr, hc)))
    assert len(matches) == 1, "arrow orientation must be unambiguous"
    return matches[0]


def _push_pixels(room: np.ndarray, direction: str) -> tuple[np.ndarray, list[int], int]:
    h, w = room.shape
    out = np.full((h, w), BG, dtype=int)
    placed_colors: list[int] = []

    pixels = [
        (r, c, int(room[r, c]))
        for r in range(h)
        for c in range(w)
        if int(room[r, c]) not in (BG, ARROW)
    ]

    if direction in ("up", "down"):
        lanes: defaultdict[int, list[tuple[int, int]]] = defaultdict(list)
        for r, c, color in pixels:
            lanes[c].append((r, color))
        for c in sorted(lanes):
            items = sorted(lanes[c], key=lambda item: item[0], reverse=direction == "down")
            for idx, (_row, color) in enumerate(items):
                rr = idx if direction == "up" else h - 1 - idx
                out[rr, c] = color
                placed_colors.append(color)
    else:
        lanes = defaultdict(list)
        for r, c, color in pixels:
            lanes[r].append((c, color))
        for r in sorted(lanes):
            items = sorted(lanes[r], key=lambda item: item[0], reverse=direction == "right")
            for idx, (_col, color) in enumerate(items):
                cc = idx if direction == "left" else w - 1 - idx
                out[r, cc] = color
                placed_colors.append(color)

    return out, placed_colors, len(pixels)


def _dominant_color(colors: list[int]) -> int | None:
    if not colors:
        return None
    counts = Counter(colors)
    best = max(counts.values())
    for color in colors:
        if counts[color] == best:
            return color
    return None


def _draw_line(room: np.ndarray, head: tuple[int, int], direction: str, length: int, color: int | None) -> None:
    if length <= 0 or color is None:
        return
    h, w = room.shape
    dr, dc = DIRS[direction]
    rr, cc = head
    step_r, step_c = -dr, -dc
    for step in range(length):
        nr = rr + step * step_r
        nc = cc + step * step_c
        if not (0 <= nr < h and 0 <= nc < w):
            break
        room[nr, nc] = color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Push each room's colored pixels toward its arrow wall, then replace the
    maroon arrow with a reverse-direction line whose length equals that wall
    pile size and whose color is the pile's majority color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "grid must be 2D"

    out = g.copy()
    for r0, r1, c0, c1 in _room_slices(g):
        room = g[r0 : r1 + 1, c0 : c1 + 1]
        direction, head = _detect_arrow(room)
        packed, placed_colors, wall_total = _push_pixels(room, direction)
        _draw_line(packed, head, direction, wall_total, _dominant_color(placed_colors))
        out[r0 : r1 + 1, c0 : c1 + 1] = packed

    return out


def _balanced_prefix(values: tuple[object, ...], n: int) -> list[object]:
    bag: list[object] = []
    while len(bag) < n:
        chunk = list(values)
        random.shuffle(chunk)
        bag.extend(chunk)
    return bag[:n]


def _arrow_cells(head: tuple[int, int], direction: str) -> set[tuple[int, int]]:
    hr, hc = head
    pts = {(hr, hc)}
    pts.update((hr + dr, hc + dc) for dr, dc in ARROW_TAILS[direction])
    return pts


def _line_capacity(h: int, w: int, head: tuple[int, int], direction: str) -> int:
    hr, hc = head
    if direction == "up":
        return h - hr
    if direction == "down":
        return hr + 1
    if direction == "left":
        return w - hc
    return hc + 1


def _sample_counts(total: int, n_colors: int) -> list[int] | None:
    if n_colors == 1:
        return [total]
    if total < n_colors:
        return None
    if n_colors == 2:
        lo = total // 2 + 1
        if lo > total - 1:
            return None
        dominant = random.randint(lo, total - 1)
        return [dominant, total - dominant]

    lo = max(2, total // 3 + 1)
    hi = total - 2
    if lo > hi:
        return None
    for _ in range(50):
        dominant = random.randint(lo, hi)
        rem = total - dominant
        a = random.randint(1, rem - 1)
        b = rem - a
        if dominant > max(a, b):
            others = [a, b]
            random.shuffle(others)
            return [dominant] + others
    return None


def _choose_head(h: int, w: int) -> tuple[int, int]:
    row_candidates = list(range(max(1, h // 2 - 1), min(h - 1, h // 2 + 2)))
    col_candidates = list(range(max(1, w // 2 - 1), min(w - 1, w // 2 + 2)))
    return random.choice(row_candidates), random.choice(col_candidates)


def _candidate_positions(
    h: int,
    w: int,
    arrow_cells: set[tuple[int, int]],
    head: tuple[int, int],
    direction: str,
) -> list[tuple[int, int]]:
    hr, hc = head
    positions = []
    for r in range(h):
        for c in range(w):
            if (r, c) in arrow_cells:
                continue
            if direction in ("up", "down") and c == hc:
                continue
            if direction in ("left", "right") and r == hr:
                continue
            positions.append((r, c))
    return positions


def _place_room(
    grid: np.ndarray,
    top: int,
    left: int,
    h: int,
    w: int,
    direction: str,
    truncate: bool,
) -> bool:
    room = grid[top : top + h, left : left + w]
    head = _choose_head(h, w)
    arrow_pts = _arrow_cells(head, direction)
    if not all(0 <= r < h and 0 <= c < w for r, c in arrow_pts):
        return False

    capacity = _line_capacity(h, w, head, direction)
    positions = _candidate_positions(h, w, arrow_pts, head, direction)
    max_pixels = len(positions)
    if max_pixels < 2:
        return False

    if truncate and max_pixels > capacity:
        lo = capacity + 1
        hi = min(max_pixels, capacity + 4)
    else:
        lo = 2
        hi = min(max_pixels, capacity)
        truncate = False
    if lo > hi:
        return False

    total_pixels = random.randint(lo, hi)
    max_color_count = 3 if total_pixels >= 4 else min(2, total_pixels)
    n_colors = random.choices(
        range(1, max_color_count + 1),
        weights=[2, 4, 3][:max_color_count],
        k=1,
    )[0]
    counts = _sample_counts(total_pixels, n_colors)
    if counts is None:
        return False

    colors = random.sample(PALETTE, n_colors)
    pixels = random.sample(positions, total_pixels)
    color_stream: list[int] = []
    for color, count in zip(colors, counts):
        color_stream.extend([color] * count)
    random.shuffle(color_stream)

    room[:, :] = BG
    for r, c in arrow_pts:
        room[r, c] = ARROW
    for (r, c), color in zip(pixels, color_stream):
        room[r, c] = color
    return True


def _gen_attempt() -> np.ndarray | None:
    n_room_rows, n_room_cols = random.choice(ROOM_LAYOUTS)
    heights = [random.randint(*ROOM_SIZE_RANGE) for _ in range(n_room_rows)]
    widths = [random.randint(*ROOM_SIZE_RANGE) for _ in range(n_room_cols)]
    h = sum(heights) + n_room_rows + 1
    w = sum(widths) + n_room_cols + 1
    if h > MAX_GRID_SIDE or w > MAX_GRID_SIDE:
        return None

    grid = np.full((h, w), BG, dtype=int)

    row_edges = [0]
    for height in heights:
        row_edges.append(row_edges[-1] + 1 + height)
    col_edges = [0]
    for width in widths:
        col_edges.append(col_edges[-1] + 1 + width)

    for rr in row_edges:
        grid[rr, :] = WALL
    for cc in col_edges:
        grid[:, cc] = WALL

    n_rooms = n_room_rows * n_room_cols
    directions = _balanced_prefix(tuple(DIRS), n_rooms)
    truncates = _balanced_prefix((False, True), n_rooms)

    idx = 0
    for i in range(n_room_rows):
        for j in range(n_room_cols):
            top = row_edges[i] + 1
            left = col_edges[j] + 1
            room_h = heights[i]
            room_w = widths[j]
            if not _place_room(
                grid,
                top,
                left,
                room_h,
                room_w,
                str(directions[idx]),
                bool(truncates[idx]),
            ):
                return None
            idx += 1

    try:
        out = main(grid)
    except AssertionError:
        return None
    if not np.any(out != grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_RETRIES} attempts")
