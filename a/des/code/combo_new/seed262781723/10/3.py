import random

import numpy as np


BG = 0
WALL = 1
SOURCE = 2
PATH = 3
YELLOW = 4
ORANGE = 7

DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]
CYCLE = [SOURCE, YELLOW, ORANGE]

GEN_ATTEMPTS = 500
ASSEMBLY_ATTEMPTS = 160
ROOM_ATTEMPTS = 40
PACK_ATTEMPTS = 160
MAX_PATHS = 3
MAX_BENDS = 5
BIG_CANVAS = 30


def _neighbors4(r: int, c: int, h: int, w: int) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for dr, dc in DIRS:
        rr, cc = r + dr, c + dc
        if 0 <= rr < h and 0 <= cc < w:
            out.append((rr, cc))
    return out


def _find_rooms(grid: np.ndarray) -> tuple[list[tuple[int, int, int, int]], dict[tuple[int, int], int]]:
    h, w = grid.shape
    mask = grid == WALL
    seen = np.zeros((h, w), dtype=bool)
    rooms: list[tuple[int, int, int, int]] = []
    wall_to_room: dict[tuple[int, int], int] = {}

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc, h, w):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))

            rows = [p[0] for p in comp]
            cols = [p[1] for p in comp]
            r0, r1 = min(rows), max(rows)
            c0, c1 = min(cols), max(cols)
            if r1 - r0 < 2 or c1 - c0 < 2:
                continue

            expected = set()
            for rr in range(r0, r1 + 1):
                expected.add((rr, c0))
                expected.add((rr, c1))
            for cc in range(c0, c1 + 1):
                expected.add((r0, cc))
                expected.add((r1, cc))

            comp_set = set(comp)
            if comp_set != expected:
                continue

            room_id = len(rooms)
            rooms.append((r0, c0, r1, c1))
            for cell in comp_set:
                wall_to_room[cell] = room_id

    return rooms, wall_to_room


def _touching_room_ids(
    cell: tuple[int, int], wall_to_room: dict[tuple[int, int], int], h: int, w: int
) -> set[int]:
    ids: set[int] = set()
    r, c = cell
    for nr, nc in _neighbors4(r, c, h, w):
        room_id = wall_to_room.get((nr, nc))
        if room_id is not None:
            ids.add(room_id)
    return ids


def _trace_path(grid: np.ndarray, start: tuple[int, int]) -> tuple[list[tuple[int, int]], int]:
    h, w = grid.shape
    prev: tuple[int, int] | None = None
    cur = start
    cells = [start]
    seen = {start}
    dirs: list[tuple[int, int]] = []

    while True:
        next_cells = []
        for nr, nc in _neighbors4(cur[0], cur[1], h, w):
            if prev is not None and (nr, nc) == prev:
                continue
            if int(grid[nr, nc]) == PATH:
                next_cells.append((nr, nc))

        if prev is None:
            assert len(next_cells) == 1, "source must start exactly one path"
        else:
            assert len(next_cells) <= 1, "path must not branch"

        if not next_cells:
            break

        nxt = next_cells[0]
        assert nxt not in seen, "path must be a simple chain"
        dirs.append((nxt[0] - cur[0], nxt[1] - cur[1]))
        prev, cur = cur, nxt
        seen.add(cur)
        cells.append(cur)

    assert dirs, "path must contain at least one segment"
    bends = sum(dirs[i] != dirs[i - 1] for i in range(1, len(dirs)))
    return cells, bends


def main(grid: np.ndarray) -> np.ndarray:
    """
    Follow each source-marked path, change color at every bend, and fill the destination room with the arriving signal color.

    Input structure:
    - Black background.
    - Blue hollow rectangular rooms, each a one-pixel-thick closed rectangle.
    - One-pixel-wide green orthogonal paths connecting room walls.
    - Each path has a red source pixel at one endpoint next to its starting room.

    Rule:
    - Start each signal at the red source with color red.
    - Trace the connected green path from that source to its far endpoint.
    - Every time the path turns 90 degrees, advance the color cycle
      red -> yellow -> orange -> red.
    - Fill the interior of the room touched by the far endpoint with the final color.
      Keep the blue walls, green path pixels, and red source pixels unchanged.

    Key insight:
    The transformation depends only on the number of direction changes along each
    simple orthogonal path; the endpoint room inherits the bend-count color modulo 3.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    rooms, wall_to_room = _find_rooms(g)
    assert rooms, "expected at least one room"

    out = g.copy()
    used_path_cells: set[tuple[int, int]] = set()
    for start_arr in np.argwhere(g == SOURCE):
        start = (int(start_arr[0]), int(start_arr[1]))
        start_ids = _touching_room_ids(start, wall_to_room, h, w)
        assert len(start_ids) == 1, "source must touch exactly one start room"

        path_cells, bends = _trace_path(g, start)
        assert not (used_path_cells & set(path_cells)), "paths must be disjoint"
        used_path_cells.update(path_cells)

        end = path_cells[-1]
        end_ids = _touching_room_ids(end, wall_to_room, h, w)
        assert len(end_ids) == 1, "path endpoint must touch exactly one destination room"
        end_room = rooms[next(iter(end_ids))]

        fill_color = CYCLE[bends % len(CYCLE)]
        r0, c0, r1, c1 = end_room
        out[r0 + 1 : r1, c0 + 1 : c1] = fill_color

    return out


def _bbox_from_points(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [p[0] for p in points]
    cols = [p[1] for p in points]
    return min(rows), min(cols), max(rows), max(cols)


def _boxes_separate(a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int = 0) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return ar1 + pad < br0 or br1 + pad < ar0 or ac1 + pad < bc0 or bc1 + pad < ac0


def _attach_room_bbox(
    endpoint: tuple[int, int], room_dir: int, outer_h: int, outer_w: int, offset: int
) -> tuple[int, int, int, int]:
    r, c = endpoint
    if room_dir == 0:  # north
        r0 = r - outer_h
        c0 = c - offset
    elif room_dir == 1:  # east
        r0 = r - offset
        c0 = c + 1
    elif room_dir == 2:  # south
        r0 = r + 1
        c0 = c - offset
    else:  # west
        r0 = r - offset
        c0 = c - outer_w
    return r0, c0, r0 + outer_h - 1, c0 + outer_w - 1


def _room_wall_cells(bbox: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    cells: list[tuple[int, int]] = []
    for r in range(r0, r1 + 1):
        cells.append((r, c0))
        cells.append((r, c1))
    for c in range(c0 + 1, c1):
        cells.append((r0, c))
        cells.append((r1, c))
    return cells


def _path_clear_of_box(
    path_cells: list[tuple[int, int]], endpoint_idx: int, bbox: tuple[int, int, int, int], pad: int = 1
) -> bool:
    r0, c0, r1, c1 = bbox
    for i, (r, c) in enumerate(path_cells):
        if i == endpoint_idx:
            continue
        if r0 - pad <= r <= r1 + pad and c0 - pad <= c <= c1 + pad:
            return False
    return True


def _build_path(dirs: list[int], lengths: list[int]) -> list[tuple[int, int]] | None:
    cells = [(0, 0)]
    used = {(0, 0)}
    for dir_idx, seg_len in zip(dirs, lengths):
        dr, dc = DIRS[dir_idx]
        for _ in range(seg_len):
            cr, cc = cells[-1]
            nr, nc = cr + dr, cc + dc
            if (nr, nc) in used:
                return None
            for ar, ac in ((nr - 1, nc), (nr + 1, nc), (nr, nc - 1), (nr, nc + 1)):
                if (ar, ac) in used and (ar, ac) != (cr, cc):
                    return None
            cells.append((nr, nc))
            used.add((nr, nc))
    return cells


def _make_assembly(remainder: int) -> np.ndarray | None:
    bend_options = [b for b in range(MAX_BENDS + 1) if b % 3 == remainder]
    for _ in range(ASSEMBLY_ATTEMPTS):
        bends = random.choice(bend_options)
        dirs = [random.randrange(4)]
        for _ in range(bends):
            dirs.append((dirs[-1] + random.choice((-1, 1))) % 4)
        min_len = 3 if bends == 0 else 2
        lengths = [random.randint(min_len, 4) for _ in dirs]
        path_cells = _build_path(dirs, lengths)
        if path_cells is None:
            continue

        path_box = _bbox_from_points(path_cells)
        if path_box[2] - path_box[0] > 10 or path_box[3] - path_box[1] > 10:
            continue

        start_dir = (dirs[0] + 2) % 4
        end_dir = dirs[-1]
        for _ in range(ROOM_ATTEMPTS):
            sh = random.randint(4, 6)
            sw = random.randint(4, 6)
            eh = random.randint(4, 6)
            ew = random.randint(4, 6)
            soff = random.randint(1, sh - 2) if start_dir in (1, 3) else random.randint(1, sw - 2)
            eoff = random.randint(1, eh - 2) if end_dir in (1, 3) else random.randint(1, ew - 2)
            start_box = _attach_room_bbox(path_cells[0], start_dir, sh, sw, soff)
            end_box = _attach_room_bbox(path_cells[-1], end_dir, eh, ew, eoff)

            if not _boxes_separate(start_box, end_box, pad=1):
                continue
            if not _path_clear_of_box(path_cells, 0, start_box):
                continue
            if not _path_clear_of_box(path_cells, len(path_cells) - 1, end_box):
                continue

            total_r0 = min(path_box[0], start_box[0], end_box[0]) - 1
            total_c0 = min(path_box[1], start_box[1], end_box[1]) - 1
            total_r1 = max(path_box[2], start_box[2], end_box[2]) + 1
            total_c1 = max(path_box[3], start_box[3], end_box[3]) + 1
            h = total_r1 - total_r0 + 1
            w = total_c1 - total_c0 + 1
            if h > 18 or w > 18:
                continue

            arr = np.full((h, w), BG, dtype=int)
            for box in (start_box, end_box):
                for r, c in _room_wall_cells(box):
                    rr = r - total_r0
                    cc = c - total_c0
                    if arr[rr, cc] != BG:
                        break
                    arr[rr, cc] = WALL
                else:
                    continue
                break
            else:
                for i, (r, c) in enumerate(path_cells):
                    rr = r - total_r0
                    cc = c - total_c0
                    if arr[rr, cc] != BG:
                        break
                    arr[rr, cc] = SOURCE if i == 0 else PATH
                else:
                    if np.count_nonzero(arr == SOURCE) != 1:
                        continue
                    try:
                        out = main(arr)
                    except Exception:
                        continue
                    if np.array_equal(out, arr):
                        continue
                    return arr
    return None


def _mark_forbidden(mask: np.ndarray, coords: np.ndarray) -> None:
    h, w = mask.shape
    for r, c in coords:
        rr = int(r)
        cc = int(c)
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr, nc = rr + dr, cc + dc
                if 0 <= nr < h and 0 <= nc < w:
                    mask[nr, nc] = True


def _pack_assemblies(assemblies: list[np.ndarray]) -> np.ndarray | None:
    canvas = np.full((BIG_CANVAS, BIG_CANVAS), BG, dtype=int)
    forbidden = np.zeros_like(canvas, dtype=bool)

    order = sorted(assemblies, key=lambda arr: arr.shape[0] * arr.shape[1], reverse=True)
    for arr in order:
        ah, aw = arr.shape
        mask = arr != BG
        coords = np.argwhere(mask)
        placed = False
        for _ in range(PACK_ATTEMPTS):
            r0 = random.randint(0, BIG_CANVAS - ah)
            c0 = random.randint(0, BIG_CANVAS - aw)
            if np.any(mask & forbidden[r0 : r0 + ah, c0 : c0 + aw]):
                continue
            sub = canvas[r0 : r0 + ah, c0 : c0 + aw]
            sub[mask] = arr[mask]
            shifted = coords + np.array([r0, c0])
            _mark_forbidden(forbidden, shifted)
            placed = True
            break
        if not placed:
            return None

    used = np.argwhere(canvas != BG)
    if used.size == 0:
        return None
    r0, c0 = used.min(axis=0)
    r1, c1 = used.max(axis=0)
    top = random.randint(0, min(3, int(r0)))
    left = random.randint(0, min(3, int(c0)))
    bottom = random.randint(0, min(3, BIG_CANVAS - 1 - int(r1)))
    right = random.randint(0, min(3, BIG_CANVAS - 1 - int(c1)))
    return canvas[r0 - top : r1 + bottom + 1, c0 - left : c1 + right + 1]


def _sample_remainders(path_count: int) -> list[int]:
    if path_count == 1:
        return [random.randrange(3)]
    values = [0, 1, 2]
    random.shuffle(values)
    return values[:path_count]


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        path_count = random.randint(1, MAX_PATHS)
        assemblies: list[np.ndarray] = []
        ok = True
        for remainder in _sample_remainders(path_count):
            assembly = _make_assembly(remainder)
            if assembly is None:
                ok = False
                break
            assemblies.append(assembly)
        if not ok:
            continue

        grid = _pack_assemblies(assemblies)
        if grid is None:
            continue
        if max(grid.shape) > 30:
            continue

        try:
            out = main(grid)
        except Exception:
            continue
        if np.array_equal(out, grid):
            continue
        return grid

    raise RuntimeError("failed to generate a valid input")
