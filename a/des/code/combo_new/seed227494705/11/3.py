import random
import numpy as np


def _components(mask):
    h, w = mask.shape
    seen = np.zeros((h, w), bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = [(r, c)]
            seen[r, c] = True
            pts = []
            while q:
                x, y = q.pop()
                pts.append((x, y))
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < h and 0 <= ny < w and mask[nx, ny] and not seen[nx, ny]:
                        seen[nx, ny] = True
                        q.append((nx, ny))
            comps.append(pts)
    return comps


def main(grid: np.ndarray) -> np.ndarray:
    """Repaint each room with the original floor color of the room across its door.

    Input structure: grey one-pixel walls partition a black-framed plan into
    rectangular rooms filled with solid non-grey floor colors; black one-cell
    gaps in shared walls are doors.
    Rule: identify room floor components and door gaps joining two rooms. A
    one-door room takes the opposite room's original color; a two-door room takes
    the original color of the smaller adjacent room. All repainting is simultaneous.
    Key insight: wall gaps are directional color-transfer portals, and double
    doors need an area-based neighbor choice.
    """
    grey = 5
    out = grid.copy()
    comps = _components((grid != 0) & (grid != grey))
    rooms = []
    cell_to_room = {}
    for idx, pts in enumerate(comps):
        color = int(grid[pts[0]])
        rooms.append({"pts": pts, "color": color, "area": len(pts), "neigh": set()})
        for p in pts:
            cell_to_room[p] = idx
    for r in range(grid.shape[0]):
        for c in range(grid.shape[1]):
            if grid[r, c] != 0:
                continue
            pairs = []
            if (r - 1, c) in cell_to_room and (r + 1, c) in cell_to_room:
                pairs.append((cell_to_room[(r - 1, c)], cell_to_room[(r + 1, c)]))
            if (r, c - 1) in cell_to_room and (r, c + 1) in cell_to_room:
                pairs.append((cell_to_room[(r, c - 1)], cell_to_room[(r, c + 1)]))
            for a, b in pairs:
                if a != b:
                    rooms[a]["neigh"].add(b)
                    rooms[b]["neigh"].add(a)
    for room in rooms:
        neigh = list(room["neigh"])
        if not neigh:
            continue
        if len(neigh) == 1:
            new_color = rooms[neigh[0]]["color"]
        else:
            chosen = min(neigh, key=lambda i: rooms[i]["area"])
            new_color = rooms[chosen]["color"]
        for r, c in room["pts"]:
            out[r, c] = new_color
    return out


def generate_input() -> np.ndarray:
    rows, cols = random.choice([(2, 2), (2, 3), (3, 2)])
    rh, rw = random.randint(4, 5), random.randint(4, 5)
    h, w = rows * rh + rows + 1, cols * rw + cols + 1
    grid = np.zeros((h, w), dtype=int)
    for r in range(0, h, rh + 1):
        grid[r, :] = 5
    for c in range(0, w, rw + 1):
        grid[:, c] = 5
    colors = random.sample([1, 2, 3, 4, 6, 7, 8, 9], rows * cols)
    k = 0
    for i in range(rows):
        for j in range(cols):
            r1, c1 = 1 + i * (rh + 1), 1 + j * (rw + 1)
            grid[r1:r1 + rh, c1:c1 + rw] = colors[k]
            k += 1
    possible = []
    for i in range(rows):
        for j in range(cols - 1):
            wall_c = (j + 1) * (rw + 1)
            possible.append((1 + i * (rh + 1) + random.randint(1, rh - 2), wall_c))
    for i in range(rows - 1):
        for j in range(cols):
            wall_r = (i + 1) * (rh + 1)
            possible.append((wall_r, 1 + j * (rw + 1) + random.randint(1, rw - 2)))
    random.shuffle(possible)
    for r, c in possible[:max(rows * cols - 1, 2)]:
        grid[r, c] = 0
    return grid
