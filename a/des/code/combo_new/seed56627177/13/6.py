from __future__ import annotations

from collections import deque
import random

import numpy as np


BLACK = 0
WALL = 3
PALETTE_COLORS = (1, 2, 4, 5, 6, 7, 8, 9)
MIN_ROOM_SIDE = 4
MAX_ROOM_SIDE = 6
MAX_ATTEMPTS = 400


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []

    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((int(cr), int(cc)))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        q.append((int(nr), int(nc)))
            comps.append(comp)

    return comps


def _touches_boundary(comp: list[tuple[int, int]], shape: tuple[int, int]) -> bool:
    h, w = shape
    return any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in comp)


def _palette_priority(grid: np.ndarray) -> dict[int, int]:
    palette: list[int] = []
    seen: set[int] = set()
    for value in grid[0].tolist():
        color = int(value)
        if color in (BLACK, WALL) or color in seen:
            continue
        seen.add(color)
        palette.append(color)
    assert palette, "expected at least one palette color on the top row"
    return {color: i for i, color in enumerate(palette)}


def _distance_map(room_mask: np.ndarray, seed: tuple[int, int]) -> np.ndarray:
    h, w = room_mask.shape
    dist = np.full((h, w), -1, dtype=int)
    sr, sc = seed
    q = deque([(int(sr), int(sc))])
    dist[sr, sc] = 0

    while q:
        r, c = q.popleft()
        nd = int(dist[r, c] + 1)
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and bool(room_mask[nr, nc]) and int(dist[nr, nc]) == -1:
                dist[nr, nc] = nd
                q.append((int(nr), int(nc)))

    return dist


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow each room's colored seed pixels until the whole room is filled, using the top palette to break equal-distance ties.

    Input structure:
    - The background is black.
    - Green one-pixel-thick walls partition the grid into several enclosed rectangular rooms.
    - Each room contains one or two non-black, non-green seed pixels.
    - The top row, outside the walls, contains a left-to-right palette whose distinct colors exactly cover the seed colors.

    Rule:
    - Keep the walls and the palette unchanged.
    - For each enclosed room, expand every seed simultaneously through 4-neighbor steps until all black cells in that room are reached.
    - A cell belongs to the seed that reaches it in fewer expansion steps.
    - If two seeds reach a cell in the same number of steps, color that cell with whichever seed color appears earlier in the top palette.

    Key insight:
    - The fill is a multi-source shortest-path partition inside each enclosed room, and the top palette provides the visible tie-break priority between same-distance seeds.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    priority = _palette_priority(g)

    room_candidates = g != WALL
    rooms = [comp for comp in _components(room_candidates) if not _touches_boundary(comp, g.shape)]
    assert rooms, "expected at least one enclosed room"

    out = g.copy()
    for comp in rooms:
        room_mask = np.zeros(g.shape, dtype=bool)
        for r, c in comp:
            room_mask[r, c] = True

        seeds = [(r, c, int(g[r, c])) for r, c in comp if int(g[r, c]) not in (BLACK, WALL)]
        assert 1 <= len(seeds) <= 2, "each room must contain one or two seeds"
        for _r, _c, color in seeds:
            assert color in priority, "every seed color must appear in the palette"

        if len(seeds) == 1:
            fill_color = int(seeds[0][2])
            for r, c in comp:
                out[r, c] = fill_color
            continue

        dist_maps = [_distance_map(room_mask, (r, c)) for r, c, _color in seeds]
        for r, c in comp:
            best_dist = min(int(dist[r, c]) for dist in dist_maps)
            tied_colors = [
                int(color)
                for dist, (_sr, _sc, color) in zip(dist_maps, seeds, strict=False)
                if int(dist[r, c]) == best_dist
            ]
            out[r, c] = min(tied_colors, key=lambda color: priority[color])

    return out


def _room_ranges(sizes: list[int], start_wall: int) -> tuple[list[tuple[int, int]], list[int]]:
    rooms: list[tuple[int, int]] = []
    walls = [start_wall]
    cursor = start_wall
    for size in sizes:
        r0 = cursor + 1
        r1 = r0 + size - 1
        rooms.append((r0, r1))
        cursor = r1 + 1
        walls.append(cursor)
    return rooms, walls


def _choose_grid_counts() -> tuple[int, int]:
    rows = random.choices([2, 3, 4], weights=[4, 4, 1], k=1)[0]
    cols = random.choices([2, 3, 4], weights=[4, 4, 1], k=1)[0]
    return rows, cols


def _seed_counts(room_count: int) -> list[int]:
    if room_count == 1:
        return [random.choice([1, 2])]
    counts = [1] * room_count
    two_count = random.randint(1, room_count - 1)
    for idx in random.sample(range(room_count), two_count):
        counts[idx] = 2
    return counts


def _two_seed_styles(two_room_count: int) -> list[str]:
    if two_room_count == 0:
        return []
    if two_room_count == 1:
        return [random.choice(["even", "odd"])]
    styles = ["even", "odd"]
    while len(styles) < two_room_count:
        styles.append(random.choice(["even", "odd"]))
    random.shuffle(styles)
    return styles


def _pick_two_seed_positions(
    r0: int,
    r1: int,
    c0: int,
    c1: int,
    parity: str,
) -> list[tuple[int, int]]:
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    need_even = parity == "even"
    orientations = ["horizontal", "vertical"]
    random.shuffle(orientations)

    for orientation in orientations:
        span = w if orientation == "horizontal" else h
        diffs = [d for d in range(1, span) if (d % 2 == 0) == need_even]
        if not diffs:
            continue
        diff = random.choice(diffs)
        if orientation == "horizontal":
            row = random.randint(r0, r1)
            left = random.randint(c0, c1 - diff)
            pts = [(row, left), (row, left + diff)]
        else:
            col = random.randint(c0, c1)
            top = random.randint(r0, r1 - diff)
            pts = [(top, col), (top + diff, col)]
        random.shuffle(pts)
        return pts

    raise AssertionError("room size must support the requested two-seed parity")


def _expected_output(
    inp: np.ndarray,
    rooms: list[dict],
    priority: dict[int, int],
) -> np.ndarray:
    out = inp.copy()
    for room in rooms:
        r0, r1 = room["rows"]
        c0, c1 = room["cols"]
        seeds = room["seeds"]
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                dists = [(abs(r - sr) + abs(c - sc), color) for sr, sc, color in seeds]
                best = min(dist for dist, _color in dists)
                tied = [color for dist, color in dists if dist == best]
                out[r, c] = min(tied, key=lambda color: priority[color])
    return out


def _build_layout() -> tuple[np.ndarray, list[dict], list[int]]:
    room_rows, room_cols = _choose_grid_counts()
    heights = [random.randint(MIN_ROOM_SIDE, MAX_ROOM_SIDE) for _ in range(room_rows)]
    widths = [random.randint(MIN_ROOM_SIDE, MAX_ROOM_SIDE) for _ in range(room_cols)]

    total_h = 1 + sum(heights) + room_rows + 1
    total_w = sum(widths) + room_cols + 1
    assert total_h <= 30 and total_w <= 30

    grid = np.full((total_h, total_w), BLACK, dtype=int)
    row_ranges, wall_rows = _room_ranges(heights, 1)
    col_ranges, wall_cols = _room_ranges(widths, 0)

    for wr in wall_rows:
        grid[wr, :] = WALL
    for wc in wall_cols:
        grid[1:, wc] = WALL

    rooms: list[dict] = []
    for rr in row_ranges:
        for cc in col_ranges:
            rooms.append({"rows": rr, "cols": cc, "seeds": []})

    seed_counts = _seed_counts(len(rooms))
    total_seed_slots = sum(seed_counts)
    palette_size = random.randint(3, min(6, len(PALETTE_COLORS), total_seed_slots))
    palette = random.sample(list(PALETTE_COLORS), palette_size)

    room_colors: list[list[int]] = []
    for count in seed_counts:
        if count == 1:
            room_colors.append([random.choice(palette)])
        else:
            room_colors.append(random.sample(palette, 2))

    used_colors = {color for colors in room_colors for color in colors}
    unused = [color for color in palette if color not in used_colors]
    for color in unused:
        slots = [
            (room_idx, seed_idx)
            for room_idx, colors in enumerate(room_colors)
            for seed_idx in range(len(colors))
            if len(colors) == 1 or colors[1 - seed_idx] != color
        ]
        room_idx, seed_idx = random.choice(slots)
        room_colors[room_idx][seed_idx] = color

    two_seed_styles = _two_seed_styles(sum(1 for count in seed_counts if count == 2))
    style_idx = 0
    for room, count, colors in zip(rooms, seed_counts, room_colors, strict=False):
        r0, r1 = room["rows"]
        c0, c1 = room["cols"]
        if count == 1:
            pos = (random.randint(r0, r1), random.randint(c0, c1))
            room["seeds"] = [(pos[0], pos[1], colors[0])]
        else:
            pts = _pick_two_seed_positions(r0, r1, c0, c1, two_seed_styles[style_idx])
            style_idx += 1
            room["seeds"] = [(r, c, color) for (r, c), color in zip(pts, colors, strict=False)]
        for r, c, color in room["seeds"]:
            grid[r, c] = color

    start = random.randint(0, total_w - palette_size)
    grid[0, start : start + palette_size] = np.array(palette, dtype=int)

    return grid, rooms, palette


def _valid_structure(grid: np.ndarray, rooms: list[dict], palette: list[int]) -> bool:
    if grid.ndim != 2 or max(grid.shape) > 30:
        return False
    if any(color in (BLACK, WALL) for color in palette):
        return False
    if len(palette) != len(set(palette)):
        return False
    used = {color for room in rooms for _r, _c, color in room["seeds"]}
    if used != set(palette):
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid, rooms, palette = _build_layout()
        if not _valid_structure(grid, rooms, palette):
            continue

        priority = {color: idx for idx, color in enumerate(palette)}
        expected = _expected_output(grid, rooms, priority)
        try:
            actual = main(grid)
        except AssertionError:
            continue
        if np.array_equal(actual, expected):
            return grid

    raise RuntimeError("failed to generate a valid input for puzzle 6")
