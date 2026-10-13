import random
from collections import deque

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
MAGENTA = 6
ORANGE = 7
AZURE = 8

FILL_SEQUENCE = (RED, ORANGE, YELLOW, BLUE, MAGENTA, AZURE)
NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
MAX_RETRIES = 300


def _neighbors4(r: int, c: int):
    for dr, dc in NEIGHBORS4:
        yield r + dr, c + dc


def _components(mask: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
    h, w = mask.shape
    comp_id = np.full((h, w), -1, dtype=int)
    comps: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or int(comp_id[r, c]) != -1:
                continue
            idx = len(comps)
            q = deque([(r, c)])
            comp_id[r, c] = idx
            coords: list[tuple[int, int]] = []

            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and int(comp_id[nr, nc]) == -1:
                        comp_id[nr, nc] = idx
                        q.append((nr, nc))

            comps.append(np.array(coords, dtype=int))

    return comp_id, comps


def _door_candidates(grid: np.ndarray) -> list[tuple[int, int, int]]:
    h, w = grid.shape
    doors: list[tuple[int, int, int]] = []
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            if int(grid[r, c]) != BLACK:
                continue

            up = int(grid[r - 1, c])
            down = int(grid[r + 1, c])
            left = int(grid[r, c - 1])
            right = int(grid[r, c + 1])

            if up == GREEN and down == GREEN and left != GREEN and right != GREEN:
                doors.append((r, c, 0))
            elif left == GREEN and right == GREEN and up != GREEN and down != GREEN:
                doors.append((r, c, 1))
    return doors


def main(grid: np.ndarray) -> np.ndarray:
    """
    Color the rooms of a wall-separated chain starting from the red-marked endpoint.

    Input structure:
    - Background is black.
    - Thin green wall pixels partition the black area into enclosed rooms.
    - Adjacent rooms may be linked by a single black wall-gap door, and all doors together form one non-branching room chain.
    - Exactly one endpoint room contains a red marker pixel.

    Rule:
    - Detect each single-pixel wall gap that acts as a door and temporarily close it to recover the individual rooms.
    - Find the room containing the red marker, then traverse the room-adjacency chain through the doors.
    - Fill the first room red, then cycle orange, yellow, blue, magenta, azure for later rooms, covering the marker as part of the room fill.
    - Preserve every green wall pixel and leave every black door pixel black.

    Key insight:
    - The black interiors are connected through doors, so the solver must first infer those missing wall pixels, close them to separate the rooms, and only then walk the resulting room graph from the marked endpoint.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    door_candidates = _door_candidates(g)
    sealed = g.copy()
    for r, c, _ in door_candidates:
        sealed[r, c] = GREEN

    comp_id, comps = _components(sealed != GREEN)

    red_pixels = np.argwhere(g == RED)
    assert len(red_pixels) == 1, "expected exactly one red marker"
    sr, sc = map(int, red_pixels[0])
    start_id = int(comp_id[sr, sc])
    assert start_id >= 0, "red marker must lie inside an open room"

    adjacency = [set() for _ in comps]
    for r, c, orientation in door_candidates:
        if orientation == 0:
            a = (r, c - 1)
            b = (r, c + 1)
        else:
            a = (r - 1, c)
            b = (r + 1, c)

        left_id = int(comp_id[a])
        right_id = int(comp_id[b])
        if left_id >= 0 and right_id >= 0 and left_id != right_id:
            adjacency[left_id].add(right_id)
            adjacency[right_id].add(left_id)

    if adjacency[start_id]:
        assert len(adjacency[start_id]) == 1, "the marked room must be a chain endpoint"

    order: list[int] = []
    seen: set[int] = set()
    prev = -1
    cur = start_id

    while True:
        assert cur not in seen, "room graph must not cycle"
        assert len(adjacency[cur]) <= 2, "room graph must be a chain"
        seen.add(cur)
        order.append(cur)

        nxt = [nid for nid in adjacency[cur] if nid != prev]
        assert len(nxt) <= 1, "room graph must not branch"
        if not nxt:
            break
        prev, cur = cur, nxt[0]

    out = g.copy()
    for idx, room_id in enumerate(order):
        color = int(FILL_SEQUENCE[idx % len(FILL_SEQUENCE)])
        coords = comps[room_id]
        out[coords[:, 0], coords[:, 1]] = color

    return out


def _induced_options(
    cell: tuple[int, int],
    used: set[tuple[int, int]],
    rows: int,
    cols: int,
) -> list[tuple[int, int]]:
    r, c = cell
    options: list[tuple[int, int]] = []
    for nr, nc in _neighbors4(r, c):
        if not (0 <= nr < rows and 0 <= nc < cols):
            continue
        if (nr, nc) in used:
            continue
        ok = True
        for ar, ac in _neighbors4(nr, nc):
            if (ar, ac) in used and (ar, ac) != cell:
                ok = False
                break
        if ok:
            options.append((nr, nc))
    return options


def _dfs_induced_path(
    rng: random.Random,
    path: list[tuple[int, int]],
    used: set[tuple[int, int]],
    rows: int,
    cols: int,
    target_len: int,
) -> list[tuple[int, int]] | None:
    if len(path) == target_len:
        return list(path)

    cur = path[-1]
    scored: list[tuple[int, float, tuple[int, int]]] = []
    for nxt in _induced_options(cur, used, rows, cols):
        new_used = used | {nxt}
        if len(path) + 1 < target_len:
            future = _induced_options(nxt, new_used, rows, cols)
            if not future:
                continue
            score = len(future)
        else:
            score = 0
        scored.append((score, rng.random(), nxt))

    scored.sort(key=lambda item: (-item[0], item[1]))
    for _score, _noise, nxt in scored:
        path.append(nxt)
        used.add(nxt)
        result = _dfs_induced_path(rng, path, used, rows, cols, target_len)
        if result is not None:
            return result
        path.pop()
        used.remove(nxt)
    return None


def _generate_induced_path(
    rng: random.Random,
    rows: int,
    cols: int,
    target_len: int,
) -> list[tuple[int, int]] | None:
    starts = [(r, c) for r in range(rows) for c in range(cols)]
    rng.shuffle(starts)
    for start in starts:
        result = _dfs_induced_path(rng, [start], {start}, rows, cols, target_len)
        if result is not None:
            return result
    return None


def _partition_lengths(rng: random.Random, total: int, parts: int) -> list[int]:
    cuts = sorted(rng.sample(range(1, total), parts - 1))
    prev = 0
    lengths: list[int] = []
    for cut in cuts + [total]:
        lengths.append(cut - prev)
        prev = cut
    return lengths


def _cell_origin(cell_r: int, cell_c: int, cell_size: int) -> tuple[int, int]:
    return cell_r * (cell_size + 1) + 1, cell_c * (cell_size + 1) + 1


def _apply_random_symmetry(grid: np.ndarray, rng: random.Random) -> np.ndarray:
    out = np.array(grid, copy=True)
    k = rng.randrange(4)
    if k:
        out = np.rot90(out, k)
    if rng.random() < 0.5:
        out = np.fliplr(out)
    return np.array(out, dtype=int, copy=False)


def _pad_with_background(grid: np.ndarray, rng: random.Random) -> np.ndarray:
    h, w = grid.shape
    spare_h = 30 - h
    spare_w = 30 - w
    assert spare_h >= 2 and spare_w >= 2

    top = rng.randint(1, min(3, spare_h - 1))
    bottom = rng.randint(1, min(3, spare_h - top))
    left = rng.randint(1, min(3, spare_w - 1))
    right = rng.randint(1, min(3, spare_w - left))

    out = np.full((h + top + bottom, w + left + right), BLACK, dtype=int)
    out[top : top + h, left : left + w] = grid
    return out


def _render_rooms(
    path: list[tuple[int, int]],
    room_ids: list[int],
    room_count: int,
    cell_size: int,
    rng: random.Random,
) -> np.ndarray:
    rows = [r for r, _ in path]
    cols = [c for _, c in path]
    min_r, max_r = min(rows), max(rows)
    min_c, max_c = min(cols), max(cols)
    shifted = [(r - min_r, c - min_c) for r, c in path]

    h = (max_r - min_r + 1) * (cell_size + 1) + 1
    w = (max_c - min_c + 1) * (cell_size + 1) + 1
    grid = np.full((h, w), BLACK, dtype=int)

    room_of_cell = {cell: int(room_ids[idx]) for idx, cell in enumerate(shifted)}
    anchor_pixels = {room: [] for room in range(room_count)}

    for idx, (cell_r, cell_c) in enumerate(shifted):
        room = int(room_ids[idx])
        r0, c0 = _cell_origin(cell_r, cell_c, cell_size)
        r1 = r0 + cell_size - 1
        c1 = c0 + cell_size - 1

        grid[r0 - 1, c0 - 1 : c1 + 2] = GREEN
        grid[r1 + 1, c0 - 1 : c1 + 2] = GREEN
        grid[r0 - 1 : r1 + 2, c0 - 1] = GREEN
        grid[r0 - 1 : r1 + 2, c1 + 1] = GREEN

        for rr in range(r0 + 1, r1):
            for cc in range(c0 + 1, c1):
                anchor_pixels[room].append((rr, cc))

    for cell_r, cell_c in shifted:
        room = room_of_cell[(cell_r, cell_c)]
        r0, c0 = _cell_origin(cell_r, cell_c, cell_size)
        r1 = r0 + cell_size - 1
        c1 = c0 + cell_size - 1

        right = (cell_r, cell_c + 1)
        if right in room_of_cell:
            boundary_c = c1 + 1
            if int(room_of_cell[right]) == room:
                grid[r0 : r1 + 1, boundary_c] = BLACK
            else:
                door_r = rng.randint(r0 + 1, r1 - 1)
                grid[door_r, boundary_c] = BLACK

        down = (cell_r + 1, cell_c)
        if down in room_of_cell:
            boundary_r = r1 + 1
            if int(room_of_cell[down]) == room:
                grid[boundary_r, c0 : c1 + 1] = BLACK
            else:
                door_c = rng.randint(c0 + 1, c1 - 1)
                grid[boundary_r, door_c] = BLACK

    start_room = 0 if rng.random() < 0.5 else room_count - 1
    marker_r, marker_c = rng.choice(anchor_pixels[start_room])
    grid[marker_r, marker_c] = RED

    return grid


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    wrap_branch = rng.random() < 0.5

    if wrap_branch:
        coarse_rows = rng.randint(5, 6)
        coarse_cols = rng.randint(5, 6)
        room_count = rng.randint(7, 8)
    else:
        coarse_rows = rng.randint(4, 6)
        coarse_cols = rng.randint(4, 6)
        room_count = rng.randint(3, 6)

    cell_cap = 27 // max(coarse_rows, coarse_cols) - 1
    cell_size = rng.randint(3, min(4, cell_cap))

    max_path_len = min(coarse_rows * coarse_cols // 2 + 2, 11)
    if room_count > max_path_len:
        return None

    extra = rng.randint(0, min(3, max_path_len - room_count))
    path_len = room_count + extra

    path = _generate_induced_path(rng, coarse_rows, coarse_cols, path_len)
    if path is None:
        return None

    lengths = _partition_lengths(rng, path_len, room_count)
    room_ids: list[int] = []
    for room, length in enumerate(lengths):
        room_ids.extend([room] * length)

    base = _render_rooms(path, room_ids, room_count, cell_size, rng)
    base = _apply_random_symmetry(base, rng)
    grid = _pad_with_background(base, rng)

    if max(grid.shape) > 30:
        return None
    if np.count_nonzero(grid == RED) != 1:
        return None

    out = main(grid)
    if out.shape != grid.shape:
        return None
    if not np.any(out != grid):
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid room-chain puzzle")
