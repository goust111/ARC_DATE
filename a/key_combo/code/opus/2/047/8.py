import numpy as np


def _move_left(grid, colors):
    h, w = grid.shape
    out = np.zeros_like(grid)
    for r in range(h):
        row = [(c, int(grid[r, c])) for c in range(w) if grid[r, c] not in (0, 8)]
        row.sort(key=lambda x: x[0])
        for k, (_, col) in enumerate(row):
            out[r, k] = col
    return out


def _move_right(grid, colors):
    h, w = grid.shape
    out = np.zeros_like(grid)
    for r in range(h):
        row = [(c, int(grid[r, c])) for c in range(w) if grid[r, c] not in (0, 8)]
        row.sort(key=lambda x: x[0])
        for k, (_, col) in enumerate(row):
            out[r, w - 1 - k] = col
    return out


def _move_up(grid, colors):
    h, w = grid.shape
    out = np.zeros_like(grid)
    for c in range(w):
        col = [(r, int(grid[r, c])) for r in range(h) if grid[r, c] not in (0, 8)]
        col.sort(key=lambda x: x[0])
        for k, (_, color) in enumerate(col):
            out[k, c] = color
    return out


def _move_down(grid, colors):
    h, w = grid.shape
    out = np.zeros_like(grid)
    for c in range(w):
        col = [(r, int(grid[r, c])) for r in range(h) if grid[r, c] not in (0, 8)]
        col.sort(key=lambda x: x[0])
        for k, (_, color) in enumerate(col):
            out[h - 1 - k, c] = color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, copy=True)
    h, w = g.shape
    marker = np.argwhere(g == 8)
    if len(marker) == 0:
        return g
    mr, mc = map(int, marker[0])

    if mr == 0:
        packed = _move_up(g, None)
        direction = "up"
    elif mr == h - 1:
        packed = _move_down(g, None)
        direction = "down"
    elif mc == 0:
        packed = _move_left(g, None)
        direction = "left"
    elif mc == w - 1:
        packed = _move_right(g, None)
        direction = "right"
    else:
        return g

    # keep marker
    packed[mr, mc] = 8

    # redraw lines from same-color pairs
    out = packed.copy()
    colors = [c for c in np.unique(packed) if c not in (0, 8)]
    coords = {int(c): np.argwhere(packed == c) for c in colors}

    # horizontals first, then verticals (vertical has priority at cross)
    for c in colors:
        pts = coords[c]
        if len(pts) != 2:
            continue
        (r0, c0), (r1, c1) = map(tuple, pts[:2])
        if r0 == r1:
            cc0, cc1 = sorted((c0, c1))
            out[r0, cc0 : cc1 + 1] = c

    for c in colors:
        pts = coords[c]
        if len(pts) != 2:
            continue
        (r0, c0), (r1, c1) = map(tuple, pts[:2])
        if c0 == c1:
            rr0, rr1 = sorted((r0, r1))
            out[rr0 : rr1 + 1, c0] = c

    return out


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    h = w = 10
    dirs = ["up", "down", "left", "right"]
    direction = str(rng.choice(dirs))
    palette = [1, 2, 3, 4, 7]

    while True:
        grid = np.zeros((h, w), dtype=np.int64)
        if direction == "up":
            marker = (0, int(rng.integers(0, w)))
        elif direction == "down":
            marker = (h - 1, int(rng.integers(0, w)))
        elif direction == "left":
            marker = (int(rng.integers(0, h)), 0)
        else:
            marker = (int(rng.integers(0, h)), w - 1)
        grid[marker] = 8

        colors = list(rng.choice(palette, size=4, replace=False))
        used = set([(int(marker[0]), int(marker[1]))])
        pairs = []
        for c in colors:
            pts = []
            for _ in range(100):
                a = (int(rng.integers(0, h)), int(rng.integers(0, w)))
                b = (int(rng.integers(0, h)), int(rng.integers(0, w)))
                if a == b or a in used or b in used or a[0] == b[0] and a[1] == b[1]:
                    continue
                if a[0] == b[0] or a[1] == b[1]:
                    continue
                pts = [a, b]
                break
            if not pts:
                break
            used.add(a)
            used.add(b)
            grid[a] = c
            grid[b] = c
            pairs.append((c, a, b))
        if len(pairs) != len(colors):
            continue

        def compress_coords(color_pairs):
            cells = {}
            if direction in ("left", "right"):
                for ccol, a, b in color_pairs:
                    for rr, cc in (a, b):
                        cells[(rr, cc)] = None
                    for r in range(h):
                        row_cells = [x for x in cells if x[0] == r]
                        row_cells.sort(key=lambda x: x[1])
            else:
                pass

        # simulate compression and pairing after transform
        tmp = grid.copy()
        if direction == "left":
            tmp = _move_left(tmp, None)
        elif direction == "right":
            tmp = _move_right(tmp, None)
        elif direction == "up":
            tmp = _move_up(tmp, None)
        else:
            tmp = _move_down(tmp, None)

        align = 0
        for c in colors:
            ps = np.argwhere(tmp == c)
            if len(ps) < 2:
                continue
            if ps[0, 0] == ps[1, 0] or ps[0, 1] == ps[1, 1]:
                align += 1
        if align >= 2 and align < len(colors):
            return grid

