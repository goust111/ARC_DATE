import random
import numpy as np


BG = 0
GREY = 5


def _components(mask):
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells = []
            while stack:
                cr, cc = stack.pop()
                cells.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(cells)
    return comps


def _bbox(cells):
    rs = [p[0] for p in cells]
    cs = [p[1] for p in cells]
    return min(rs), min(cs), max(rs), max(cs)


def _rot(mask, k):
    return np.rot90(mask, -k)


def main(grid: np.ndarray) -> np.ndarray:
    """Stamp edge color templates into grey-partitioned rooms using marker quadrant.

    Input structure: Black background is split by full grey row and column
    dividers into rectangular rooms. Small monochrome template sprites sit on
    the outer edge, and each room contains one singleton marker whose color
    matches one template.
    Rule: Read the same-colored edge template, rotate it according to the
    marker's quadrant in its room, align the corresponding template corner to
    the marker, stamp it into the room, and then erase the marker pixel. Grey
    dividers and edge templates are preserved.
    Key insight: Marker color selects the template while marker position inside
    the local room selects the rotation and anchor corner.
    """
    out = grid.copy()
    h, w = grid.shape
    templates = {}
    for color in [int(x) for x in np.unique(grid) if x not in (BG, GREY)]:
        comps = _components(grid == color)
        for comp in comps:
            if not any(r in (0, h - 1) or c in (0, w - 1) for r, c in comp):
                continue
            r0, c0, r1, c1 = _bbox(comp)
            if r1 - r0 <= 2 and c1 - c0 <= 2 and len(comp) > 1:
                m = np.zeros((3, 3), dtype=bool)
                for r, c in comp:
                    m[r - r0, c - c0] = True
                templates[color] = m
                break

    grey_rows = [r for r in range(h) if np.count_nonzero(grid[r] == GREY) >= max(3, w // 2)]
    grey_cols = [c for c in range(w) if np.count_nonzero(grid[:, c] == GREY) >= max(3, h // 2)]
    row_edges = [-1] + grey_rows + [h]
    col_edges = [-1] + grey_cols + [w]

    for i in range(len(row_edges) - 1):
        for j in range(len(col_edges) - 1):
            r0, r1 = row_edges[i] + 1, row_edges[i + 1] - 1
            c0, c1 = col_edges[j] + 1, col_edges[j + 1] - 1
            if r1 - r0 < 2 or c1 - c0 < 2:
                continue
            marks = []
            for r in range(r0, r1 + 1):
                for c in range(c0, c1 + 1):
                    col = int(grid[r, c])
                    if col in templates:
                        edge_neighbor = r in (0, h - 1) or c in (0, w - 1)
                        if not edge_neighbor:
                            marks.append((r, c, col))
            if len(marks) != 1:
                continue
            mr, mc, col = marks[0]
            top = mr <= (r0 + r1) / 2
            left = mc <= (c0 + c1) / 2
            if top and left:
                k, ar, ac = 0, mr, mc
            elif top and not left:
                k, ar, ac = 1, mr, mc - 2
            elif (not top) and (not left):
                k, ar, ac = 2, mr - 2, mc - 2
            else:
                k, ar, ac = 3, mr - 2, mc
            mask = _rot(templates[col], k)
            for rr in range(3):
                for cc in range(3):
                    if mask[rr, cc]:
                        nr, nc = ar + rr, ac + cc
                        if r0 <= nr <= r1 and c0 <= nc <= c1 and out[nr, nc] == BG:
                            out[nr, nc] = col
            out[mr, mc] = BG
    return out


def generate_input() -> np.ndarray:
    colors = random.sample([1, 2, 3, 4, 6, 7, 8, 9], 4)
    h = random.choice([17, 19, 21, 23])
    w = random.choice([17, 19, 21, 23])
    grid = np.zeros((h, w), dtype=int)
    shapes = [
        [(0, 0), (1, 0), (2, 0), (2, 1)],
        [(0, 0), (0, 1), (0, 2), (1, 1), (2, 1)],
        [(0, 0), (1, 0), (1, 1), (1, 2)],
        [(0, 1), (1, 1), (2, 1), (2, 0), (2, 2)],
    ]
    for idx, (col, shape) in enumerate(zip(colors, shapes)):
        c0 = 1 + idx * 4
        if c0 + 2 >= w:
            break
        for dr, dc in shape:
            grid[dr, c0 + dc] = col
    gr = [h // 2]
    gc = [w // 2]
    if random.random() < 0.45:
        gr = sorted([h // 3, 2 * h // 3])
    if random.random() < 0.45:
        gc = sorted([w // 3, 2 * w // 3])
    for r in gr:
        grid[r, :] = GREY
    for c in gc:
        grid[:, c] = GREY
    row_edges = [-1] + gr + [h]
    col_edges = [-1] + gc + [w]
    qpos = [(0.25, 0.25), (0.25, 0.75), (0.75, 0.75), (0.75, 0.25)]
    rooms = []
    for i in range(len(row_edges) - 1):
        for j in range(len(col_edges) - 1):
            r0, r1 = row_edges[i] + 1, row_edges[i + 1] - 1
            c0, c1 = col_edges[j] + 1, col_edges[j + 1] - 1
            if r1 - r0 >= 4 and c1 - c0 >= 4:
                rooms.append((r0, c0, r1, c1))
    random.shuffle(rooms)
    for idx, room in enumerate(rooms[: min(len(rooms), 6)]):
        r0, c0, r1, c1 = room
        col = random.choice(colors)
        qr, qc = random.choice(qpos)
        mr = max(r0 + 2, min(r1 - 2, int(round(r0 + qr * (r1 - r0)))))
        mc = max(c0 + 2, min(c1 - 2, int(round(c0 + qc * (c1 - c0)))))
        grid[mr, mc] = col
    return grid
