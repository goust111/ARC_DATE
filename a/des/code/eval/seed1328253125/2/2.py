from collections import deque
import random

import numpy as np


BLACK = 0
MAROON = 9
MARKER_COLORS = (1, 2, 3, 4, 5, 6, 7, 8)

GEN_ATTEMPTS = 400
ROOM_ROW_RANGE = (2, 5)
ROOM_COL_RANGE = (2, 5)
ROOM_HEIGHT_RANGE = (2, 6)
ROOM_WIDTH_RANGE = (2, 6)
MARGIN_RANGE = (1, 5)
PALETTE_MARGIN_RANGE = (2, 6)
PALETTE_EXTRA_MAX = 3
EXTRA_DOOR_PROB = 0.3


class UnionFind:
    def __init__(self, size):
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra = self.find(a)
        rb = self.find(b)
        if ra == rb:
            return False
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1
        return True


def _neighbors4(r, c):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _find_doors(grid):
    h, w = grid.shape
    door_mask = np.zeros((h, w), dtype=bool)
    doors = []
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            if int(grid[r, c]) != BLACK:
                continue
            vertical_gap = (
                int(grid[r - 1, c]) == MAROON
                and int(grid[r + 1, c]) == MAROON
                and int(grid[r, c - 1]) != MAROON
                and int(grid[r, c + 1]) != MAROON
            )
            horizontal_gap = (
                int(grid[r, c - 1]) == MAROON
                and int(grid[r, c + 1]) == MAROON
                and int(grid[r - 1, c]) != MAROON
                and int(grid[r + 1, c]) != MAROON
            )
            if vertical_gap == horizontal_gap:
                continue
            door_mask[r, c] = True
            doors.append((r, c, "vertical" if vertical_gap else "horizontal"))
    return door_mask, doors


def _extract_rooms(grid):
    h, w = grid.shape
    door_mask, doors = _find_doors(grid)
    candidate = (grid != MAROON) & ~door_mask
    seen = np.zeros((h, w), dtype=bool)
    labels = -np.ones((h, w), dtype=int)
    rooms = []

    for r in range(h):
        for c in range(w):
            if seen[r, c] or not candidate[r, c]:
                continue
            queue = deque([(r, c)])
            seen[r, c] = True
            coords = []
            touches_boundary = False
            while queue:
                rr, cc = queue.popleft()
                coords.append((rr, cc))
                if rr == 0 or rr == h - 1 or cc == 0 or cc == w - 1:
                    touches_boundary = True
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and candidate[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))

            if touches_boundary:
                continue

            room_id = len(rooms)
            arr = np.array(coords, dtype=int)
            labels[arr[:, 0], arr[:, 1]] = room_id
            colors = {int(grid[rr, cc]) for rr, cc in coords if int(grid[rr, cc]) != BLACK}
            assert len(colors) <= 1, "each room may contain at most one marker color"
            rooms.append(
                {
                    "coords": arr,
                    "marker": None if not colors else next(iter(colors)),
                }
            )

    assert rooms, "expected at least one enclosed room"
    return rooms, labels, doors


def _region_colors(grid):
    rooms, labels, doors = _extract_rooms(grid)
    uf = UnionFind(len(rooms))
    for r, c, orientation in doors:
        if orientation == "vertical":
            a = int(labels[r, c - 1])
            b = int(labels[r, c + 1])
        else:
            a = int(labels[r - 1, c])
            b = int(labels[r + 1, c])
        if a >= 0 and b >= 0 and a != b:
            uf.union(a, b)

    groups = {}
    for room_id, room in enumerate(rooms):
        root = uf.find(room_id)
        groups.setdefault(root, []).append(room_id)

    room_fill = [None] * len(rooms)
    for room_ids in groups.values():
        markers = {rooms[room_id]["marker"] for room_id in room_ids if rooms[room_id]["marker"] is not None}
        assert len(markers) == 1, "each connected room region must contain exactly one marker color"
        fill_color = next(iter(markers))
        for room_id in room_ids:
            room_fill[room_id] = fill_color
    return rooms, room_fill


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each door-connected room region with the color of its unique marker.
    """
    g = np.asarray(grid, dtype=int)
    rooms, room_fill = _region_colors(g)

    out = g.copy()
    for room, fill_color in zip(rooms, room_fill):
        coords = room["coords"]
        mask = out[coords[:, 0], coords[:, 1]] == BLACK
        out[coords[mask, 0], coords[mask, 1]] = fill_color
    return out


def _room_spans(offset, sizes):
    spans = []
    walls = [offset]
    cursor = offset
    for size in sizes:
        start = cursor + 1
        end = start + size - 1
        spans.append((start, end))
        cursor = end + 1
        walls.append(cursor)
    return spans, walls


def _room_id(row_index, col_index, cols):
    return row_index * cols + col_index


def _build_grid(rng):
    room_rows = rng.randrange(*ROOM_ROW_RANGE)
    room_cols = rng.randrange(*ROOM_COL_RANGE)
    heights = [rng.randrange(*ROOM_HEIGHT_RANGE) for _ in range(room_rows)]
    widths = [rng.randrange(*ROOM_WIDTH_RANGE) for _ in range(room_cols)]

    palette_side = rng.choice(("top", "bottom"))
    top_margin = rng.randrange(*PALETTE_MARGIN_RANGE) if palette_side == "top" else rng.randrange(*MARGIN_RANGE)
    bottom_margin = rng.randrange(*PALETTE_MARGIN_RANGE) if palette_side == "bottom" else rng.randrange(*MARGIN_RANGE)
    left_margin = rng.randrange(*MARGIN_RANGE)
    right_margin = rng.randrange(*MARGIN_RANGE)

    plan_h = sum(heights) + room_rows + 1
    plan_w = sum(widths) + room_cols + 1
    h = top_margin + plan_h + bottom_margin
    w = left_margin + plan_w + right_margin

    grid = np.zeros((h, w), dtype=int)
    row_spans, wall_rows = _room_spans(top_margin, heights)
    col_spans, wall_cols = _room_spans(left_margin, widths)
    top_wall = wall_rows[0]
    bottom_wall = wall_rows[-1]
    left_wall = wall_cols[0]
    right_wall = wall_cols[-1]

    for wall_r in wall_rows:
        grid[wall_r, left_wall : right_wall + 1] = MAROON
    for wall_c in wall_cols:
        grid[top_wall : bottom_wall + 1, wall_c] = MAROON

    edges = []
    for i in range(room_rows):
        for j in range(room_cols - 1):
            edges.append(("vertical", i, j))
    for i in range(room_rows - 1):
        for j in range(room_cols):
            edges.append(("horizontal", i, j))

    room_count = room_rows * room_cols
    max_regions = min(4, room_count - 1)
    target_regions = rng.randint(1, max_regions)

    rng.shuffle(edges)
    uf = UnionFind(room_count)
    selected = []
    selected_set = set()
    components_left = room_count
    for edge in edges:
        if components_left == target_regions:
            break
        orientation, i, j = edge
        if orientation == "vertical":
            a = _room_id(i, j, room_cols)
            b = _room_id(i, j + 1, room_cols)
        else:
            a = _room_id(i, j, room_cols)
            b = _room_id(i + 1, j, room_cols)
        if uf.union(a, b):
            selected.append(edge)
            selected_set.add(edge)
            components_left -= 1

    assert components_left == target_regions

    for edge in edges:
        if edge in selected_set or rng.random() >= EXTRA_DOOR_PROB:
            continue
        orientation, i, j = edge
        if orientation == "vertical":
            a = _room_id(i, j, room_cols)
            b = _room_id(i, j + 1, room_cols)
        else:
            a = _room_id(i, j, room_cols)
            b = _room_id(i + 1, j, room_cols)
        if uf.find(a) != uf.find(b):
            continue
        selected.append(edge)
        selected_set.add(edge)

    for orientation, i, j in selected:
        if orientation == "vertical":
            wall_c = wall_cols[j + 1]
            row_start, row_end = row_spans[i]
            door_r = rng.randint(row_start, row_end)
            grid[door_r, wall_c] = BLACK
        else:
            wall_r = wall_rows[i + 1]
            col_start, col_end = col_spans[j]
            door_c = rng.randint(col_start, col_end)
            grid[wall_r, door_c] = BLACK

    roots = []
    for room_id in range(room_count):
        root = uf.find(room_id)
        if root not in roots:
            roots.append(root)

    region_colors = rng.sample(list(MARKER_COLORS), len(roots))
    for root, color in zip(roots, region_colors):
        region_rooms = [room_id for room_id in range(room_count) if uf.find(room_id) == root]
        marker_room = rng.choice(region_rooms)
        room_r, room_c = divmod(marker_room, room_cols)
        rr0, rr1 = row_spans[room_r]
        cc0, cc1 = col_spans[room_c]
        marker_r = rng.randint(rr0, rr1)
        marker_c = rng.randint(cc0, cc1)
        grid[marker_r, marker_c] = color

    extra_count = rng.randint(0, min(PALETTE_EXTRA_MAX, len(MARKER_COLORS) - len(region_colors)))
    extras = rng.sample([c for c in MARKER_COLORS if c not in region_colors], extra_count)
    palette = list(region_colors) + extras
    rng.shuffle(palette)
    palette_row = 0 if palette_side == "top" else h - 1
    start_c = rng.randint(0, w - len(palette))
    grid[palette_row, start_c : start_c + len(palette)] = palette

    return grid, room_count, len(roots)


def _valid_generated_grid(grid, expected_rooms, expected_regions):
    rooms, room_fill = _region_colors(grid)
    if len(rooms) != expected_rooms:
        return False
    if len(set(room_fill)) != expected_regions:
        return False
    if np.array_equal(main(grid), grid):
        return False
    return True


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid, expected_rooms, expected_regions = _build_grid(rng)
        if _valid_generated_grid(grid, expected_rooms, expected_regions):
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
