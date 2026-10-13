import random
import numpy as np


DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]


def _segments(lines, limit):
    return [(lines[i] + 1, lines[i + 1] - 1) for i in range(len(lines) - 1) if lines[i] + 1 <= lines[i + 1] - 1]


def main(grid: np.ndarray) -> np.ndarray:
    out = np.array(grid, copy=True)
    h, w = out.shape
    sep_rows = [r for r in range(h) if np.all(out[r] == 5)]
    sep_cols = [c for c in range(w) if np.all(out[:, c] == 5)]
    row_segs = _segments(sep_rows, h)
    col_segs = _segments(sep_cols, w)
    if not row_segs or not col_segs:
        return out
    start = None
    for tr, (r0, r1) in enumerate(row_segs):
        for tc, (c0, c1) in enumerate(col_segs):
            tile = out[r0:r1 + 1, c0:c1 + 1]
            pts = np.argwhere(tile == 3)
            if pts.size:
                sr, sc = map(int, pts[0])
                start = (tr, tc, sr, sc)
    if start is None:
        return out
    tr, tc, sr, sc = start
    dir_map = {(0, 1): 0, (1, 2): 1, (2, 1): 2, (1, 0): 3}
    d = dir_map.get((sr, sc))
    if d is None:
        return out
    visited = set()
    while True:
        dr, dc = DIRS[d]
        tr, tc = tr + dr, tc + dc
        if not (0 <= tr < len(row_segs) and 0 <= tc < len(col_segs)) or (tr, tc) in visited:
            break
        visited.add((tr, tc))
        r0, r1 = row_segs[tr]
        c0, c1 = col_segs[tc]
        tile = out[r0:r1 + 1, c0:c1 + 1]
        vals = set(int(x) for x in tile.flatten()) - {0}
        if not vals:
            out[r0 + sr, c0 + sc] = 3
        elif 2 in vals:
            break
        elif 6 in vals:
            d = (d + 1) % 4
        elif 8 in vals:
            d = (d - 1) % 4
        else:
            break
    return out


def generate_input() -> np.ndarray:
    g = np.zeros((17, 17), dtype=int)
    g[::4, :] = 5
    g[:, ::4] = 5
    cells = [(r, c) for r in range(4) for c in range(4)]
    for _attempt in range(200):
        start = random.choice(cells)
        path = [start]
        while len(path) < random.randint(5, 8):
            r, c = path[-1]
            options = []
            for d, (dr, dc) in enumerate(DIRS):
                nr, nc = r + dr, c + dc
                if 0 <= nr < 4 and 0 <= nc < 4 and (nr, nc) not in path:
                    if len(path) < 2 or (nr, nc) != path[-2]:
                        options.append((nr, nc))
            if not options:
                break
            path.append(random.choice(options))
        if len(path) >= 4:
            break
    d0 = DIRS.index((path[1][0] - path[0][0], path[1][1] - path[0][1]))
    sub = {0: (0, 1), 1: (1, 2), 2: (2, 1), 3: (1, 0)}[d0]
    r0, c0 = path[0]
    g[1 + 4 * r0 + sub[0], 1 + 4 * c0 + sub[1]] = 3
    turn_counts = {6: 0, 8: 0}
    for i in range(1, len(path) - 1):
        prev_d = DIRS.index((path[i][0] - path[i - 1][0], path[i][1] - path[i - 1][1]))
        next_d = DIRS.index((path[i + 1][0] - path[i][0], path[i + 1][1] - path[i][1]))
        color = 0
        if next_d == (prev_d + 1) % 4:
            color = 6
        elif next_d == (prev_d - 1) % 4:
            color = 8
        if color:
            r, c = path[i]
            g[2 + 4 * r, 2 + 4 * c] = color
            turn_counts[color] += 1
    r, c = path[-1]
    g[2 + 4 * r, 2 + 4 * c] = 2
    free = [cell for cell in cells if cell not in path]
    random.shuffle(free)
    while turn_counts[6] < 2 and free:
        r, c = free.pop()
        g[2 + 4 * r, 2 + 4 * c] = 6
        turn_counts[6] += 1
    while turn_counts[8] < 1 and free:
        r, c = free.pop()
        g[2 + 4 * r, 2 + 4 * c] = 8
        turn_counts[8] += 1
    return g
