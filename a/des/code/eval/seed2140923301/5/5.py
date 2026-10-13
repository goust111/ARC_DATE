import math
import random
from collections import Counter, defaultdict, deque

import numpy as np


DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
WALL_COLOR = 0
SEPARATOR_COLOR = 9

# --------------- generator config ---------------
CELL_SIZE_OPTIONS = (3, 4)
COARSE_ROWS_RANGE = (4, 7)
COARSE_COLS_RANGE = (4, 6)
GROUP_COUNT_RANGE = (2, 4)
ROOM_EXTRA_RANGE = (2, 5)
EXTRA_DOOR_PROB = 0.35
GEN_ATTEMPTS = 500
PALETTE = tuple(range(1, 9))


def _get_bg(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _find_separator(grid: np.ndarray, bg: int) -> tuple[int, int]:
    h, w = grid.shape
    candidates = []
    for c in range(w - 1):
        value = int(grid[0, c])
        if value == bg:
            continue
        if np.all(grid[:, c] == value) and np.all(grid[:, c + 1 :] == bg):
            candidates.append((c, value))
    assert len(candidates) == 1, "expected one separator column with blank output zone"
    return candidates[0]


def _detect_doors(left: np.ndarray, bg: int, wall: int) -> list[tuple[int, int, str]]:
    h, w = left.shape
    doors = []

    def passable(r: int, c: int) -> bool:
        return int(left[r, c]) != wall

    for r in range(1, h - 1):
        for c in range(1, w - 1):
            if int(left[r, c]) != bg:
                continue
            across_vertical_wall = (
                passable(r, c - 1)
                and passable(r, c + 1)
                and int(left[r - 1, c]) == wall
                and int(left[r + 1, c]) == wall
            )
            across_horizontal_wall = (
                passable(r - 1, c)
                and passable(r + 1, c)
                and int(left[r, c - 1]) == wall
                and int(left[r, c + 1]) == wall
            )
            if across_vertical_wall ^ across_horizontal_wall:
                doors.append((r, c, "v" if across_vertical_wall else "h"))
    return doors


def _label_rooms(left: np.ndarray, bg: int, wall: int, doors: list[tuple[int, int, str]]):
    closed = left != wall
    for r, c, _ in doors:
        closed[r, c] = False

    h, w = left.shape
    labels = np.full((h, w), -1, dtype=int)
    rooms = []
    room_id = 0
    for r in range(h):
        for c in range(w):
            if not closed[r, c] or labels[r, c] != -1:
                continue
            queue = deque([(r, c)])
            labels[r, c] = room_id
            coords = []
            while queue:
                rr, cc = queue.popleft()
                coords.append((rr, cc))
                for dr, dc in DIR4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and closed[nr, nc] and labels[nr, nc] == -1:
                        labels[nr, nc] = room_id
                        queue.append((nr, nc))

            arr = np.asarray(coords, dtype=int)
            values = left[arr[:, 0], arr[:, 1]]
            marker_coords = [
                (int(rr), int(cc))
                for rr, cc in coords
                if int(left[rr, cc]) not in (bg, wall)
            ]
            assert len(marker_coords) == 1, "each room must contain exactly one marker"
            rooms.append(
                {
                    "coords": arr,
                    "area": int(np.count_nonzero(values == bg)),
                    "marker": marker_coords[0],
                    "color": int(left[marker_coords[0]]),
                }
            )
            room_id += 1

    assert rooms, "expected at least one room"
    return labels, rooms


class _DSU:
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n
        self.components = n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> bool:
        ra = self.find(a)
        rb = self.find(b)
        if ra == rb:
            return False
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1
        self.components -= 1
        return True


def _room_shape(coords: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    r0 = int(rows.min())
    r1 = int(rows.max())
    c0 = int(cols.min())
    c1 = int(cols.max())
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    mask[rows - r0, cols - c0] = True
    return mask, (r0, c0, r1, c1)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Group rooms by door connectivity, choose the largest room in each group,
    and copy its cropped room shape into the blank output zone on the right.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = g.shape
    bg = _get_bg(g)
    sep_col, sep_color = _find_separator(g, bg)

    left = g[:, :sep_col]
    non_bg_left = [int(v) for v in left.reshape(-1) if int(v) != bg]
    wall_color = Counter(non_bg_left).most_common(1)[0][0]
    assert wall_color != sep_color, "wall color must differ from separator color"

    doors = _detect_doors(left, bg, wall_color)
    labels, rooms = _label_rooms(left, bg, wall_color, doors)

    dsu = _DSU(len(rooms))
    for r, c, orient in doors:
        if orient == "v":
            a = int(labels[r, c - 1])
            b = int(labels[r, c + 1])
        else:
            a = int(labels[r - 1, c])
            b = int(labels[r + 1, c])
        assert a >= 0 and b >= 0 and a != b, "door must connect two distinct rooms"
        dsu.union(a, b)

    groups = defaultdict(list)
    for rid in range(len(rooms)):
        groups[dsu.find(rid)].append(rid)

    ordered_groups = []
    for group_room_ids in groups.values():
        group_colors = {rooms[rid]["color"] for rid in group_room_ids}
        assert len(group_colors) == 1, "all rooms in a door-connected group must share one color"
        color = next(iter(group_colors))
        first_marker = min((rooms[rid]["marker"] for rid in group_room_ids), key=lambda rc: (rc[0], rc[1]))
        largest_room_id = sorted(
            group_room_ids,
            key=lambda rid: (-rooms[rid]["area"], rooms[rid]["marker"][0], rooms[rid]["marker"][1]),
        )[0]
        shape, _bbox = _room_shape(rooms[largest_room_id]["coords"])
        ordered_groups.append((first_marker, color, shape))

    ordered_groups.sort(key=lambda item: (item[0][0], item[0][1]))

    out = g.copy()
    current_row = 0
    output_col = sep_col + 1
    for _marker, color, shape in ordered_groups:
        sh, sw = shape.shape
        assert current_row + sh <= h, "output area is too short for stacked shapes"
        assert output_col + sw <= w, "output area is too narrow for room shape"
        view = out[current_row : current_row + sh, output_col : output_col + sw]
        view[shape] = int(color)
        current_row += sh + 1
    return out


def _cell_start(index: int, cell_size: int) -> int:
    return 1 + index * (cell_size + 1)


def _room_mask(
    room_cells: set[tuple[int, int]],
    coarse_rows: int,
    coarse_cols: int,
    cell_size: int,
) -> np.ndarray:
    step = cell_size + 1
    h = coarse_rows * cell_size + (coarse_rows + 1)
    w = coarse_cols * cell_size + (coarse_cols + 1)
    mask = np.zeros((h, w), dtype=bool)
    for r, c in room_cells:
        r0 = _cell_start(r, cell_size)
        c0 = _cell_start(c, cell_size)
        mask[r0 : r0 + cell_size, c0 : c0 + cell_size] = True
    for r, c in room_cells:
        if (r, c + 1) in room_cells:
            r0 = _cell_start(r, cell_size)
            cwall = _cell_start(c, cell_size) + cell_size
            mask[r0 : r0 + cell_size, cwall] = True
        if (r + 1, c) in room_cells:
            rwall = _cell_start(r, cell_size) + cell_size
            c0 = _cell_start(c, cell_size)
            mask[rwall, c0 : c0 + cell_size] = True
    return mask


def _partition_rooms(coarse_rows: int, coarse_cols: int, room_count: int):
    cells = [(r, c) for r in range(coarse_rows) for c in range(coarse_cols)]
    seeds = random.sample(cells, room_count)
    owner = np.full((coarse_rows, coarse_cols), -1, dtype=int)
    room_cells = [set() for _ in range(room_count)]
    for rid, (r, c) in enumerate(seeds):
        owner[r, c] = rid
        room_cells[rid].add((r, c))

    unassigned = {cell for cell in cells if cell not in seeds}
    while unassigned:
        frontier = []
        for r, c in list(unassigned):
            neighbors = []
            for dr, dc in DIR4:
                nr, nc = r + dr, c + dc
                if 0 <= nr < coarse_rows and 0 <= nc < coarse_cols and owner[nr, nc] != -1:
                    neighbors.append(int(owner[nr, nc]))
            if neighbors:
                frontier.append((r, c, neighbors))
        if not frontier:
            return None
        r, c, neighbors = random.choice(frontier)
        weights = [1.0 / max(1, len(room_cells[nid])) for nid in neighbors]
        chosen = random.choices(neighbors, weights=weights, k=1)[0]
        owner[r, c] = chosen
        room_cells[chosen].add((r, c))
        unassigned.remove((r, c))
    return owner, room_cells


def _pair_boundaries(owner: np.ndarray):
    coarse_rows, coarse_cols = owner.shape
    boundaries = defaultdict(list)
    for r in range(coarse_rows):
        for c in range(coarse_cols - 1):
            a = int(owner[r, c])
            b = int(owner[r, c + 1])
            if a != b:
                boundaries[tuple(sorted((a, b)))].append(("v", r, c))
    for r in range(coarse_rows - 1):
        for c in range(coarse_cols):
            a = int(owner[r, c])
            b = int(owner[r + 1, c])
            if a != b:
                boundaries[tuple(sorted((a, b)))].append(("h", r, c))
    return boundaries


def _door_cell(boundary: tuple[str, int, int], cell_size: int) -> tuple[int, int]:
    orient, r, c = boundary
    if orient == "v":
        r0 = _cell_start(r, cell_size)
        c0 = _cell_start(c, cell_size) + cell_size
        offset = random.randint(1, cell_size - 2)
        return r0 + offset, c0
    r0 = _cell_start(r, cell_size) + cell_size
    c0 = _cell_start(c, cell_size)
    offset = random.randint(1, cell_size - 2)
    return r0, c0 + offset


def _marker_candidates(mask: np.ndarray) -> list[tuple[int, int]]:
    h, w = mask.shape
    interior = []
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            if not mask[r, c]:
                continue
            if all(mask[r + dr, c + dc] for dr, dc in DIR4):
                interior.append((r, c))
    if interior:
        return interior
    return [tuple(map(int, pt)) for pt in np.argwhere(mask)]


def _build_expected(
    grid: np.ndarray,
    sep_col: int,
    ordered_groups: list[tuple[tuple[int, int], int, np.ndarray]],
) -> np.ndarray:
    out = grid.copy()
    row = 0
    for _first_marker, color, shape in ordered_groups:
        h, w = shape.shape
        view = out[row : row + h, sep_col + 1 : sep_col + 1 + w]
        view[shape] = int(color)
        row += h + 1
    return out


def _gen_attempt() -> np.ndarray | None:
    cell_size = random.choice(CELL_SIZE_OPTIONS)
    coarse_rows = random.randint(*COARSE_ROWS_RANGE)
    coarse_cols = random.randint(*COARSE_COLS_RANGE)
    group_count = random.randint(*GROUP_COUNT_RANGE)
    max_rooms = min(coarse_rows * coarse_cols, group_count + ROOM_EXTRA_RANGE[1])
    min_rooms = min(max_rooms, group_count + ROOM_EXTRA_RANGE[0])
    if min_rooms < group_count + 1:
        return None
    room_count = random.randint(min_rooms, max_rooms)

    partition = _partition_rooms(coarse_rows, coarse_cols, room_count)
    if partition is None:
        return None
    owner, room_cells = partition

    boundaries = _pair_boundaries(owner)
    if len(boundaries) < room_count - group_count:
        return None

    dsu = _DSU(room_count)
    chosen_boundaries = []
    room_pairs = list(boundaries.keys())
    random.shuffle(room_pairs)
    for pair in room_pairs:
        if dsu.components <= group_count:
            break
        a, b = pair
        if dsu.union(a, b):
            chosen_boundaries.append(random.choice(boundaries[pair]))
    if dsu.components != group_count:
        return None

    extras = []
    for pair in room_pairs:
        a, b = pair
        if dsu.find(a) == dsu.find(b) and random.random() < EXTRA_DOOR_PROB:
            extras.append(random.choice(boundaries[pair]))
    chosen_boundaries.extend(extras)

    roots = sorted({dsu.find(rid) for rid in range(room_count)})
    if len(roots) > len(PALETTE) - 1:
        return None
    bg = random.choice(PALETTE)
    marker_palette = [color for color in PALETTE if color != bg]
    random.shuffle(marker_palette)
    root_to_color = {root: marker_palette[idx] for idx, root in enumerate(roots)}

    room_masks = []
    room_areas = []
    markers = []
    for rid, cells in enumerate(room_cells):
        mask = _room_mask(cells, coarse_rows, coarse_cols, cell_size)
        room_masks.append(mask)
        room_areas.append(int(mask.sum()))
        markers.append(random.choice(_marker_candidates(mask)))

    groups = defaultdict(list)
    for rid in range(room_count):
        groups[dsu.find(rid)].append(rid)

    if not any(len(room_ids) >= 2 for room_ids in groups.values()):
        return None
    for room_ids in groups.values():
        areas = sorted((room_areas[rid] for rid in room_ids), reverse=True)
        if len(areas) >= 2 and areas[0] == areas[1]:
            return None

    earliest_rows = []
    ordered_groups = []
    for root, room_ids in groups.items():
        color = root_to_color[root]
        first_marker = min((markers[rid] for rid in room_ids), key=lambda rc: (rc[0], rc[1]))
        largest_rid = max(room_ids, key=lambda rid: room_areas[rid])
        shape, _bbox = _room_shape(np.argwhere(room_masks[largest_rid]))
        ordered_groups.append((first_marker, color, shape))
        earliest_rows.append(first_marker[0])
    if len(set(earliest_rows)) != len(earliest_rows):
        return None

    ordered_groups.sort(key=lambda item: (item[0][0], item[0][1]))
    maze_h = coarse_rows * cell_size + (coarse_rows + 1)
    maze_w = coarse_cols * cell_size + (coarse_cols + 1)
    stacked_h = sum(shape.shape[0] for _marker, _color, shape in ordered_groups) + len(ordered_groups) - 1
    max_shape_w = max(shape.shape[1] for _marker, _color, shape in ordered_groups)
    if stacked_h > maze_h:
        return None

    preferred_out_w = math.ceil((maze_w + 1) / 2)
    if max_shape_w > preferred_out_w + 2:
        return None

    maze = np.full((maze_h, maze_w), WALL_COLOR, dtype=int)
    for cells in room_cells:
        for r, c in cells:
            r0 = _cell_start(r, cell_size)
            c0 = _cell_start(c, cell_size)
            maze[r0 : r0 + cell_size, c0 : c0 + cell_size] = bg
        for r, c in cells:
            if (r, c + 1) in cells:
                r0 = _cell_start(r, cell_size)
                cwall = _cell_start(c, cell_size) + cell_size
                maze[r0 : r0 + cell_size, cwall] = bg
            if (r + 1, c) in cells:
                rwall = _cell_start(r, cell_size) + cell_size
                c0 = _cell_start(c, cell_size)
                maze[rwall, c0 : c0 + cell_size] = bg

    for boundary in chosen_boundaries:
        dr, dc = _door_cell(boundary, cell_size)
        maze[dr, dc] = bg

    for rid, marker in enumerate(markers):
        maze[marker] = root_to_color[dsu.find(rid)]

    bg_maze = int(np.count_nonzero(maze == bg))
    black_maze = int(np.count_nonzero(maze == WALL_COLOR))
    needed_bg_cells = max(0, black_maze - bg_maze + 1)
    bg_dominance_w = math.ceil(needed_bg_cells / maze_h) if needed_bg_cells else 0
    out_w = max(preferred_out_w, max_shape_w + random.randint(1, 3), bg_dominance_w)
    if out_w > maze_w:
        return None

    grid = np.full((maze_h, maze_w + 1 + out_w), bg, dtype=int)
    grid[:, :maze_w] = maze
    grid[:, maze_w] = SEPARATOR_COLOR

    expected = _build_expected(grid, maze_w, ordered_groups)
    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None

    counts = Counter(map(int, grid.reshape(-1)))
    if counts[bg] <= max(count for color, count in counts.items() if color != bg):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid maze instance")
