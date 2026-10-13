"""Draw color rays through adjacent rooms determined by per-room marker counts.

Input structure:
- The background color is black. Rooms form a rectangular grid of equal-sized
  interior cells separated by one-pixel-thick gray walls.
- Every room contains 1 to 4 marker pixels, all sharing one room-specific color
  chosen from visible non-gray colors.
- Wall cells are shared boundaries; room interiors contain only background and
  markers.

Rule:
- Count marker pixels in each room, N=1..4, and map N to a door side:
  1=top, 2=right, 3=bottom, 4=left.
- Open one wall pixel at the midpoint of that side (replace with background).
  If the chosen side is on the outer boundary, switch to the opposite side.
- Emit a one-cell-wide ray starting from that door cell and moving into the
  neighboring room in the chosen direction, with the room's marker color.
  Stop before any gray wall or grid edge.
- Leave all other wall, background, and original marker pixels unchanged.

Key insight:
- Room size is uniform and boundaries are explicit walls, so room-level logic can be
  done by component-level extraction, and the only geometry needed for a ray is the
  exact doorway midpoint and whether it has an adjacent room.
"""

import numpy as np


_BG_COLOR = 0


def _connected_components(mask: np.ndarray) -> list[np.ndarray]:
    """Return list of point arrays (r, c) for connected components in mask."""
    h, w = mask.shape
    visited = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    stack: list[tuple[int, int]] = []

    for sr in range(h):
        for sc in range(w):
            if not mask[sr, sc] or visited[sr, sc]:
                continue
            stack.append((sr, sc))
            visited[sr, sc] = True
            pts: list[tuple[int, int]] = []

            while stack:
                r, c = stack.pop()
                pts.append((r, c))
                if r > 0 and mask[r - 1, c] and not visited[r - 1, c]:
                    visited[r - 1, c] = True
                    stack.append((r - 1, c))
                if r + 1 < h and mask[r + 1, c] and not visited[r + 1, c]:
                    visited[r + 1, c] = True
                    stack.append((r + 1, c))
                if c > 0 and mask[r, c - 1] and not visited[r, c - 1]:
                    visited[r, c - 1] = True
                    stack.append((r, c - 1))
                if c + 1 < w and mask[r, c + 1] and not visited[r, c + 1]:
                    visited[r, c + 1] = True
                    stack.append((r, c + 1))

            comps.append(np.array(pts, dtype=np.int64))

    return comps


def _bbox(pts: np.ndarray) -> tuple[int, int, int, int]:
    rs = pts[:, 0]
    cs = pts[:, 1]
    return int(rs.min()), int(rs.max()), int(cs.min()), int(cs.max())


def _most_frequent_except(values: np.ndarray, forbidden: set[int]) -> int | None:
    vals, counts = np.unique(values, return_counts=True)
    best_val = None
    best_count = -1
    for v, c in zip(vals, counts):
        iv = int(v)
        if iv in forbidden:
            continue
        if c > best_count:
            best_count = int(c)
            best_val = iv
    return best_val


def _opposite(direction: tuple[int, int]) -> tuple[int, int]:
    return (-direction[0], -direction[1])


def _direction_from_count(count: int) -> tuple[int, int] | None:
    mapping = {
        1: (-1, 0),
        2: (0, 1),
        3: (1, 0),
        4: (0, -1),
    }
    return mapping.get(count)


def _door_and_neighbor_start(
    rmin: int,
    rmax: int,
    cmin: int,
    cmax: int,
    direction: tuple[int, int],
    wall: int,
    h: int,
    w: int,
    grid: np.ndarray,
) -> tuple[tuple[int, int] | None, tuple[int, int] | None]:
    mid_r = (rmin + rmax) // 2
    mid_c = (cmin + cmax) // 2

    dr, dc = direction
    if dr == -1 and dc == 0:
        door = (rmin - 1, mid_c)
        start = (rmin - 2, mid_c)
    elif dr == 1 and dc == 0:
        door = (rmax + 1, mid_c)
        start = (rmax + 2, mid_c)
    elif dr == 0 and dc == -1:
        door = (mid_r, cmin - 1)
        start = (mid_r, cmin - 2)
    elif dr == 0 and dc == 1:
        door = (mid_r, cmax + 1)
        start = (mid_r, cmax + 2)
    else:
        return None, None

    sr, sc = start
    if not (0 <= sr < h and 0 <= sc < w):
        return None, None

    dr0, dc0 = door
    if not (0 <= dr0 < h and 0 <= dc0 < w):
        return None, None
    if grid[dr0, dc0] != wall:
        return None, None
    if grid[sr, sc] == wall:
        return None, None

    return (dr0, dc0), (sr, sc)


def main(grid: np.ndarray) -> np.ndarray:
    """Apply room-door rays according to per-room marker counts."""
    inp = np.asarray(grid)
    out = inp.copy()

    colors, counts = np.unique(inp, return_counts=True)
    bg = int(colors[np.argmax(counts)]) if colors.size > 0 else _BG_COLOR
    if bg not in colors:
        bg = _BG_COLOR

    wall = _most_frequent_except(inp.ravel(), {bg})
    if wall is None:
        return out

    h, w = inp.shape
    non_wall = inp != wall
    comps = _connected_components(non_wall)
    if not comps:
        return out

    for comp in comps:
        rmin, rmax, cmin, cmax = _bbox(comp)
        room_vals = inp[comp[:, 0], comp[:, 1]]
        room_marker_mask = (room_vals != bg) & (room_vals != wall)
        marker_values = room_vals[room_marker_mask]
        if marker_values.size == 0:
            continue

        room_color = _most_frequent_except(marker_values, {wall, bg})
        if room_color is None:
            continue

        count = int(marker_values.size)
        if count < 1 or count > 4:
            continue

        direction = _direction_from_count(count)
        if direction is None:
            continue

        door, start = _door_and_neighbor_start(rmin, rmax, cmin, cmax, direction, wall, h, w, inp)
        if door is None or start is None:
            direction = _opposite(direction)
            door, start = _door_and_neighbor_start(rmin, rmax, cmin, cmax, direction, wall, h, w, inp)
        if door is None or start is None:
            continue

        out[door] = bg

        dr, dc = direction
        sr, sc = start
        while 0 <= sr < h and 0 <= sc < w:
            if out[sr, sc] == wall:
                break
            if out[sr, sc] == bg:
                out[sr, sc] = room_color
            sr += dr
            sc += dc

    return out


def _rand_room_colors(rng: np.random.Generator, room_count: int) -> list[int]:
    """Return distinct visible room-marker colors."""
    palette = [2, 3, 4, 6, 7, 8, 9, 1]
    room_colors = palette.copy()
    rng.shuffle(room_colors)
    return room_colors[:room_count]


def generate_input() -> np.ndarray:
    """Generate random ARC-like inputs that satisfy room-based ray rules."""
    rng = np.random.default_rng()

    while True:
        n_rows = int(rng.integers(1, 5))
        n_cols = int(rng.integers(1, 5))
        room_count = n_rows * n_cols
        if 1 <= room_count <= 8:
            break

    wall = 5
    room_colors = _rand_room_colors(rng, room_count)

    while True:
        room_h = int(rng.integers(3, 8))
        room_w = int(rng.integers(3, 8))
        h = n_rows * room_h + (n_rows + 1)
        w = n_cols * room_w + (n_cols + 1)
        if h <= 30 and w <= 30:
            break

    grid = np.full((h, w), _BG_COLOR, dtype=np.int64)

    # Draw gray wall grid.
    for r in range(h):
        if r % (room_h + 1) == 0:
            grid[r, :] = wall
    for c in range(w):
        if c % (room_w + 1) == 0:
            grid[:, c] = wall

    idx = 0
    for rr in range(n_rows):
        for cc in range(n_cols):
            top = rr * (room_h + 1) + 1
            left = cc * (room_w + 1) + 1
            cells = [(r, c) for r in range(top, top + room_h) for c in range(left, left + room_w)]
            n_markers = int(rng.integers(1, 5))
            sampled = rng.choice(len(cells), size=n_markers, replace=False)
            color = room_colors[idx]
            idx += 1
            for p in sampled:
                r, c = cells[int(p)]
                grid[r, c] = color

    return grid
