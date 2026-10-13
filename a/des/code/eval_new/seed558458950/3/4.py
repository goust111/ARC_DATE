"""Propagate room-level item colors across door-connected room groups and recolor each room floor by group vote.
Input structure:
- Grid uses a dominant floor color (bg) and a grey wall color.
- One-pixel-thick axis-aligned wall lines partition the grid into rectangular rooms.
- Some wall pixels are replaced by bg to represent one-cell door openings between adjacent rooms.
- Each room contains zero, one, or two small connected objects made of 2-3 bg-connected pixels in red/blue/green.

Rule:
1) Identify each room as the 4-connected area of non-wall pixels when doors are temporarily treated as walls.
2) For each room, choose its dominant item color (the non-bg, non-wall color with highest pixel count; tie broken by red then blue then green).
3) Build room groups via door-connected components: rooms linked by doors belong to the same group.
4) For each group, take the majority of its rooms' dominant colors; recolor only that group's room floor pixels (bg) with the group color. If a group has no items, keep bg.
5) Walls and all item pixels remain unchanged.

Key insight:
Separate the geometry (rooms and connectivity) from the symbols: first recover room identity by closing every possible door, then merge rooms through doors, then apply a group-level majority vote.
"""

from __future__ import annotations

from collections import Counter
from collections import deque
from dataclasses import dataclass
import numpy as np

COLOR_ORDER = (2, 3, 4)

def _color_to_order(color: int) -> int:
    try:
        return COLOR_ORDER.index(color)
    except ValueError:
        return len(COLOR_ORDER)


def _flood_rooms(
    passable: np.ndarray,
) -> tuple[np.ndarray, list[list[tuple[int, int]]], list[Counter]]:
    h, w = passable.shape
    room_id = -np.ones((h, w), dtype=int)
    rooms: list[list[tuple[int, int]]] = []
    room_counts: list[Counter] = []
    next_id = 0

    for sr in range(h):
        for sc in range(w):
            if not passable[sr, sc] or room_id[sr, sc] != -1:
                continue

            q: deque[tuple[int, int]] = deque([(sr, sc)])
            room_id[sr, sc] = next_id
            cells: list[tuple[int, int]] = []
            counts = Counter()

            while q:
                r, c = q.popleft()
                cells.append((r, c))

                for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if not passable[nr, nc]:
                        continue
                    if room_id[nr, nc] != -1:
                        continue
                    room_id[nr, nc] = next_id
                    q.append((nr, nc))

            room_counts.append(counts)
            rooms.append(cells)
            next_id += 1

    return room_id, rooms, room_counts


class _DSU:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        p = self.parent
        while p[x] != x:
            p[x] = p[p[x]]
            x = p[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra = self.find(a)
        rb = self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def _dominant_color_for_room(
    cell_color_counter: Counter,
    bg: int,
) -> int | None:
    del bg
    if not cell_color_counter:
        return None

    max_count = max(cell_color_counter.values())
    ties = [c for c, cnt in cell_color_counter.items() if cnt == max_count]
    ties.sort(key=_color_to_order)
    return ties[0]


def _is_door(
    grid: np.ndarray,
    wall_mask: np.ndarray,
    r: int,
    c: int,
    bg: int,
    bg_mask: np.ndarray,
) -> bool:
    h, w = grid.shape
    if grid[r, c] != bg:
        return False
    if r == 0 or c == 0 or r == h - 1 or c == w - 1:
        return False

    up = wall_mask[r - 1, c]
    dn = wall_mask[r + 1, c]
    lf = wall_mask[r, c - 1]
    rt = wall_mask[r, c + 1]

    # Vertical wall normally has wall pixels above and below with bg/object on both sides.
    if up and dn and not lf and not rt:
        return True
    # Horizontal wall normally has wall pixels left and right with bg/object above and below.
    if lf and rt and not up and not dn:
        return True
    return False


def main(grid: np.ndarray) -> np.ndarray:
    """Execute the room-group recoloring rule."""

    if grid.size == 0:
        return grid.copy()

    colors, counts = np.unique(grid, return_counts=True)
    bg = colors[np.argmax(counts)]

    non_bg_colors = colors[colors != bg]
    if non_bg_colors.size == 0:
        return grid.copy()

    # Assume the most frequent non-bg color is wall color; this is stable for the generator
    # and typical ARC instances with wall-dominant backgrounds.
    wall_color = non_bg_colors[np.argmax(counts[colors != bg])]

    wall_mask = grid == wall_color
    bg_mask = grid == bg

    h, w = grid.shape
    door_mask = np.zeros((h, w), dtype=bool)
    for r in range(h):
        for c in range(w):
            if _is_door(grid, wall_mask, r, c, bg, bg_mask):
                door_mask[r, c] = True

    # Closed-view for room detection: doors are temporarily blocked.
    passable = ~(wall_mask | door_mask)

    room_id, rooms, room_object_counts = _flood_rooms(passable)
    room_num = len(rooms)
    if room_num == 0:
        return grid.copy()

    room_colors: list[int | None] = []
    for rid in range(room_num):
        counts_for_room = Counter()
        for r, c in rooms[rid]:
            pix = grid[r, c]
            if pix != bg and pix != wall_color:
                counts_for_room[pix] += 1
        # keep only room items; wall_color already blocked out by passability,
        # but guard for oddball inputs.
        room_colors.append(_dominant_color_for_room(counts_for_room, bg))

    dsu = _DSU(room_num)

    # Merge room components through each door.
    for r in range(h):
        for c in range(w):
            if not door_mask[r, c]:
                continue

            if r > 0 and r < h - 1:
                if wall_mask[r - 1, c] and wall_mask[r + 1, c]:
                    left = room_id[r, c - 1]
                    right = room_id[r, c + 1]
                    if left != -1 and right != -1:
                        dsu.union(left, right)
            if c > 0 and c < w - 1:
                if wall_mask[r, c - 1] and wall_mask[r, c + 1]:
                    up = room_id[r - 1, c]
                    dn = room_id[r + 1, c]
                    if up != -1 and dn != -1:
                        dsu.union(up, dn)

    group_cells: dict[int, list[int]] = {}
    for rid in range(room_num):
        group = dsu.find(rid)
        group_cells.setdefault(group, []).append(rid)

    group_color: dict[int, int | None] = {}
    for group, rids in group_cells.items():
        cnt = Counter()
        for rid in rids:
            c = room_colors[rid]
            if c is not None:
                cnt[c] += 1

        if not cnt:
            group_color[group] = None
            continue

        max_count = max(cnt.values())
        ties = [c for c, num in cnt.items() if num == max_count]
        ties.sort(key=_color_to_order)
        group_color[group] = ties[0]

    out = grid.copy()
    for rid, cells in enumerate(rooms):
        g = dsu.find(rid)
        fill_color = group_color.get(g)
        if fill_color is None:
            continue
        for r, c in cells:
            if out[r, c] == bg:
                out[r, c] = fill_color

    return out


def _random_connected_shape(rng: np.random.Generator, size: int) -> list[tuple[int, int]]:
    assert size in (2, 3)
    pts = [(0, 0)]
    while len(pts) < size:
        x, y = pts[rng.integers(0, len(pts))]
        dx, dy = [(1, 0), (-1, 0), (0, 1), (0, -1)][rng.integers(4)]
        nx, ny = x + dx, y + dy
        if (nx, ny) not in pts:
            pts.append((nx, ny))

    min_x = min(p[0] for p in pts)
    min_y = min(p[1] for p in pts)
    norm = [(x - min_x, y - min_y) for x, y in pts]
    norm.sort()
    return norm


def _place_object(
    rng: np.random.Generator,
    grid: np.ndarray,
    color: int,
    room: tuple[int, int, int, int],
    bg: int,
) -> None:
    r0, r1, c0, c1 = room
    size = int(rng.choice([2, 3]))
    shape = _random_connected_shape(rng, size)
    sh = [p[0] for p in shape]
    sw = [p[1] for p in shape]
    h_obj = max(sh) - min(sh) + 1
    w_obj = max(sw) - min(sw) + 1

    if r0 + h_obj - 1 > r1 or c0 + w_obj - 1 > c1:
        return

    max_r = r1 - h_obj + 1
    max_c = c1 - w_obj + 1
    for _ in range(40):
        sr = int(rng.integers(r0, max_r + 1))
        sc = int(rng.integers(c0, max_c + 1))
        cells = [(sr + dr, sc + dc) for dr, dc in shape]

        if any(grid[r, c] != bg for r, c in cells):
            continue
        for r, c in cells:
            grid[r, c] = color
        return


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()

    # Fixed item colors (red/blue/green roles) and floor/wall colors.
    bg = 0
    wall = 8
    items = (2, 3, 4)

    while True:
        cols = int(rng.integers(2, 5))
        rows = int(rng.integers(2, 5))

        col_widths = [2] * cols
        row_heights = [2] * rows

        max_col_extra = (30 - (cols + 1)) - 2 * cols
        max_row_extra = (30 - (rows + 1)) - 2 * rows
        if max_col_extra < 0 or max_row_extra < 0:
            continue

        for _ in range(int(rng.integers(0, max_col_extra + 1))):
            col_widths[rng.integers(cols)] += 1
        for _ in range(int(rng.integers(0, max_row_extra + 1))):
            row_heights[rng.integers(rows)] += 1

        width = sum(col_widths) + cols + 1
        height = sum(row_heights) + rows + 1
        if width > 30 or height > 30:
            continue

        # Build wall skeleton.
        x_lines = [0]
        cur = 0
        for width_part in col_widths:
            cur += width_part + 1
            x_lines.append(cur)

        y_lines = [0]
        cur = 0
        for h_part in row_heights:
            cur += h_part + 1
            y_lines.append(cur)

        grid = np.full((height, width), bg, dtype=int)

        # Outer boundary and all internal partition lines.
        for x in x_lines:
            grid[:, x] = wall
        for y in y_lines:
            grid[y, :] = wall

        # Create 0~2 door openings on each internal segment.
        for x in x_lines[1:-1]:
            possible_rows = [r for r in range(1, height - 1) if r not in y_lines]
            if not possible_rows:
                continue

            n_doors = int(rng.choice([0, 1, 2], p=[0.45, 0.45, 0.10]))
            if len(possible_rows) == 1:
                n_doors = min(n_doors, 1)
            rng.shuffle(possible_rows)
            for r in possible_rows[:n_doors]:
                grid[r, x] = bg

        for y in y_lines[1:-1]:
            possible_cols = [c for c in range(1, width - 1) if c not in x_lines]
            if not possible_cols:
                continue

            n_doors = int(rng.choice([0, 1, 2], p=[0.45, 0.45, 0.10]))
            if len(possible_cols) == 1:
                n_doors = min(n_doors, 1)
            rng.shuffle(possible_cols)
            for c in possible_cols[:n_doors]:
                grid[y, c] = bg

        # Place objects.
        for i in range(len(y_lines) - 1):
            r0 = y_lines[i] + 1
            r1 = y_lines[i + 1] - 1
            if r0 > r1:
                continue
            for j in range(len(x_lines) - 1):
                c0 = x_lines[j] + 1
                c1 = x_lines[j + 1] - 1
                if c0 > c1:
                    continue

                room_area = (r1 - r0 + 1) * (c1 - c0 + 1)
                if room_area < 2:
                    continue

                n_items = int(rng.choice([0, 1, 2], p=[0.28, 0.44, 0.28]))
                if room_area < 3:
                    n_items = min(n_items, 1)

                for item_idx in range(n_items):
                    # Keep branch variety: sometimes two items can share color, sometimes not.
                    if item_idx == 0:
                        color = int(rng.choice(items))
                        room_colors = [color]
                    else:
                        if rng.random() < 0.55:
                            color = int(rng.choice(items))
                        else:
                            alternatives = [c for c in items if c != room_colors[0]]
                            color = int(rng.choice(alternatives)) if alternatives else int(rng.choice(items))
                        room_colors.append(color)

                    _place_object(rng, grid, color, (r0, r1, c0, c1), bg)

        # Ensure at least one room and a non-trivial room structure exists.
        if len(np.unique(grid)) < 2:
            continue
        if len(x_lines) < 3 or len(y_lines) < 3:
            continue

        return grid
