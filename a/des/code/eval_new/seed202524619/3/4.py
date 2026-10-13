import numpy as np
from collections import deque


PALETTE = [2, 7, 4, 3, 1]
BG_COLOR = 0
WALL_COLOR = 5
MARKER_COLORS = [6, 8, 9, 10, 11, 12]


def main(grid: np.ndarray) -> np.ndarray:
    """Fill rectangular rooms separated by wall segments with a cyclic palette following their door chain order.
Input structure:
- black is background; grey-form wall cells form a one-pixel-thick closed partition into rectangular rooms.
- between two adjacent rooms one wall pixel is missing (kept black), this is the door linking the rooms.
- exactly one room contains a single-color marker pixel that designates the start room.
- rooms are linked by doors into a simple linear chain.
Rule:
- detect wall and door cells, split non-wall, non-door area into rectangular room components,
  build the room adjacency graph from doors, then walk the chain from the start room.
- recolor the first room with red, the second with orange, then yellow, green, blue,
  repeating red→orange→yellow→green→blue if rooms exceed five.
- keep walls and door pixels unchanged; marker pixels are overwritten by the room fill color.
Key insight:
- the chain is defined by wall gaps (doors), not by 2-D room positions; reading rooms in that graph order is essential.
"""

    arr = np.array(grid, dtype=int)
    H, W = arr.shape

    # Infer background and wall.
    colors, counts = np.unique(arr, return_counts=True)
    bg = BG_COLOR if BG_COLOR in colors else int(colors[np.argmax(counts)])

    non_bg = [(int(c), int(n)) for c, n in zip(colors, counts) if int(c) != bg]
    if not non_bg:
        return arr.copy()
    non_bg.sort(key=lambda x: x[1], reverse=True)
    wall = non_bg[0][0]

    wall_mask = arr == wall

    # A door is a non-wall cell with a pair of opposite neighbors that are both wall.
    door_mask = np.zeros((H, W), dtype=bool)
    for r in range(H):
        for c in range(W):
            if wall_mask[r, c]:
                continue
            if r > 0 and r + 1 < H and wall_mask[r - 1, c] and wall_mask[r + 1, c]:
                door_mask[r, c] = True
                continue
            if c > 0 and c + 1 < W and wall_mask[r, c - 1] and wall_mask[r, c + 1]:
                door_mask[r, c] = True

    floor_mask = (~wall_mask) & (~door_mask)

    # Flood-fill rectangular room candidates.
    visited = np.zeros((H, W), dtype=bool)
    comp_id = -np.ones((H, W), dtype=int)
    rooms = {}
    comp = 0
    steps = ((1, 0), (-1, 0), (0, 1), (0, -1))

    for r0 in range(H):
        for c0 in range(W):
            if not floor_mask[r0, c0] or visited[r0, c0]:
                continue

            q = deque([(r0, c0)])
            visited[r0, c0] = True
            coords = []
            min_r = min_c = 10**9
            max_r = max_c = -10**9

            while q:
                r, c = q.popleft()
                coords.append((r, c))
                if r < min_r:
                    min_r = r
                if r > max_r:
                    max_r = r
                if c < min_c:
                    min_c = c
                if c > max_c:
                    max_c = c

                for dr, dc in steps:
                    nr, nc = r + dr, c + dc
                    if nr < 0 or nr >= H or nc < 0 or nc >= W:
                        continue
                    if visited[nr, nc] or not floor_mask[nr, nc]:
                        continue
                    visited[nr, nc] = True
                    q.append((nr, nc))

            area = len(coords)
            bh = max_r - min_r + 1
            bw = max_c - min_c + 1
            if area != bh * bw:
                continue

            room_coords = np.array(coords, dtype=int)
            for rr, cc in room_coords:
                comp_id[rr, cc] = comp

            rooms[comp] = {
                "coords": room_coords,
                "bbox": (min_r, min_c, bh, bw),
                "area": area,
            }
            comp += 1

    if not rooms:
        return arr.copy()

    adjacency = {cid: set() for cid in rooms}
    for r, c in zip(*np.where(door_mask)):
        linked = []
        if r > 0 and r + 1 < H:
            a = comp_id[r - 1, c]
            b = comp_id[r + 1, c]
            if a != -1 and b != -1 and a != b:
                linked.append((a, b))
        if c > 0 and c + 1 < W:
            a = comp_id[r, c - 1]
            b = comp_id[r, c + 1]
            if a != -1 and b != -1 and a != b:
                linked.append((a, b))
        for a, b in linked:
            adjacency.setdefault(a, set()).add(b)
            adjacency.setdefault(b, set()).add(a)

    # Locate start room from the unique marker cell.
    marker_mask = (arr != bg) & (arr != wall) & (~door_mask)
    start_room = -1
    marker_cells = np.argwhere(marker_mask)
    if marker_cells.size > 0:
        r0, c0 = marker_cells[0]
        start_room = int(comp_id[r0, c0])

    if start_room not in rooms:
        start_room = next(iter(rooms))

    # Walk the chain.
    def walk(start, first):
        if start not in rooms or first not in rooms:
            return [start]
        path = [start, first]
        prev = start
        cur = first
        while True:
            candidates = [x for x in sorted(adjacency.get(cur, ())) if x != prev and x not in path]
            if not candidates:
                break
            nxt = candidates[0]
            path.append(nxt)
            prev, cur = cur, nxt
        return path

    neighbors = sorted(adjacency.get(start_room, ()))
    if len(neighbors) == 0:
        order = [start_room] if start_room in rooms else []
    elif len(neighbors) == 1:
        order = walk(start_room, neighbors[0])
    else:
        p1 = walk(start_room, neighbors[0])
        p2 = walk(start_room, neighbors[1])
        def path_key(path):
            return tuple((rooms[x]["bbox"][0], rooms[x]["bbox"][1]) for x in path if x in rooms)

        if len(p1) == len(rooms) and len(p2) != len(rooms):
            order = p1
        elif len(p2) == len(rooms) and len(p1) != len(rooms):
            order = p2
        elif len(p1) >= len(p2):
            order = p1
        else:
            order = p2
        if path_key(p1) > path_key(p2):
            order = p2

        if len(order) < len(rooms):
            for k in sorted(rooms):
                if k not in order:
                    order.append(k)

    out = arr.copy()
    for i, rid in enumerate(order):
        if rid not in rooms:
            continue
        color = PALETTE[i % len(PALETTE)]
        rr, cc = rooms[rid]["coords"].T
        out[rr, cc] = color

    return out


def _bounds(rect, pad=0):
    r, c, h, w = rect
    return (r - pad, c - pad, r + h - 1 + pad, c + w - 1 + pad)


def _intersects(a, b, pad=0):
    ar1, ac1, ar2, ac2 = _bounds(a, pad)
    br1, bc1, br2, bc2 = _bounds(b, pad)
    return not (ar2 < br1 or br2 < ar1 or ac2 < bc1 or bc2 < ac1)


def _valid_candidate(candidate, rooms, prev_idx):
    for idx, room in enumerate(rooms):
        if _intersects(candidate, room, pad=0):
            return False
        if idx != prev_idx and _intersects(candidate, room, pad=1):
            return False
    return True


def generate_input() -> np.ndarray:
    """Generate a wall-partition chain puzzle input with valid random geometry.

    Input generation uses a room chain made from rectangular rooms with one-cell-thick walls
    and exactly one-cell black doors between consecutive rooms. Room count, segment sizes,
    geometry turns and marker placement are randomized. The marker room is always one chain
    endpoint for deterministic start-room extraction.
    """

    rng = np.random.default_rng()
    attempts = 500

    for _ in range(attempts):
        n_rooms = int(rng.integers(2, 8))

        first_h = int(rng.integers(3, 7))
        first_w = int(rng.integers(3, 7))
        rooms = [(0, 0, first_h, first_w)]

        prev_dir = -1
        opposite = {0: 1, 1: 0, 2: 3, 3: 2}

        ok = True
        for _ in range(1, n_rooms):
            prev = rooms[-1]
            prev_h, prev_w = prev[2], prev[3]
            dirs = [0, 1, 2, 3]
            if prev_dir in opposite:
                back = opposite[prev_dir]
                dirs = [d for d in dirs if d != back]
            rng.shuffle(dirs)

            placed = False
            for d in dirs:
                if d == 3:  # right
                    nh = prev_h
                    nw = int(rng.integers(3, 7))
                    nr = prev[0]
                    nc = prev[1] + prev_w + 1
                elif d == 2:  # left
                    nh = prev_h
                    nw = int(rng.integers(3, 7))
                    nr = prev[0]
                    nc = prev[1] - nw - 1
                elif d == 1:  # down
                    nh = int(rng.integers(3, 7))
                    nw = prev_w
                    nr = prev[0] + prev_h + 1
                    nc = prev[1]
                else:  # up
                    nh = int(rng.integers(3, 7))
                    nw = prev_w
                    nr = prev[0] - nh - 1
                    nc = prev[1]

                cand = (nr, nc, nh, nw)
                if _valid_candidate(cand, rooms, len(rooms) - 1):
                    rooms.append(cand)
                    prev_dir = d
                    placed = True
                    break

            if not placed:
                ok = False
                break

        if not ok:
            continue

        min_r = min(r for r, _, _, _ in rooms)
        max_r = max(r + h - 1 for r, _, h, _ in rooms)
        min_c = min(c for _, c, _, _ in rooms)
        max_c = max(c + w - 1 for _, c, _, w in rooms)

        H = (max_r - min_r + 1) + 2
        W = (max_c - min_c + 1) + 2
        if H > 30 or W > 30:
            continue

        sr = 1 - min_r
        sc = 1 - min_c
        shifted = [(r + sr, c + sc, h, w) for r, c, h, w in rooms]

        grid = np.full((H, W), BG_COLOR, dtype=int)

        for r, c, h, w in shifted:
            t, b = r - 1, r + h
            l, rr = c - 1, c + w
            grid[t, l:rr + 1] = WALL_COLOR
            grid[b, l:rr + 1] = WALL_COLOR
            grid[t : b + 1, l] = WALL_COLOR
            grid[t : b + 1, rr] = WALL_COLOR

        for i in range(len(shifted) - 1):
            r1, c1, h1, w1 = shifted[i]
            r2, c2, h2, w2 = shifted[i + 1]
            if r1 == r2:
                wall_col = (c1 + w1) if c2 > c1 else (c2 + w2)
                door_r = int(rng.integers(r1 + 1, r1 + h1 - 1))
                grid[door_r, wall_col] = BG_COLOR
            else:
                wall_row = (r1 + h1) if r2 > r1 else (r2 + h2)
                door_c = int(rng.integers(c1 + 1, c1 + w1 - 1))
                grid[wall_row, door_c] = BG_COLOR

        start_room_idx = int(rng.choice([0, len(shifted) - 1]))
        sr0, sc0, sh, sw = shifted[start_room_idx]
        marker_r = int(rng.integers(sr0, sr0 + sh))
        marker_c = int(rng.integers(sc0, sc0 + sw))
        marker_color = int(rng.choice(MARKER_COLORS))
        grid[marker_r, marker_c] = marker_color

        return grid.astype(int)

    # Fallback to a tiny valid sample if random search keeps failing.
    return np.array(
        [
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 5, 5, 5, 5, 0],
            [0, 6, 5, 0, 5, 0, 0],
            [0, 5, 0, 5, 6, 0, 0],
            [0, 5, 5, 5, 5, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
        ],
        dtype=int,
    )
