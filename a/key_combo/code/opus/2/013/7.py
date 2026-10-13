import numpy as np

RNG = np.random.default_rng(707)


def main(grid: np.ndarray) -> np.ndarray:
    """Task 7: slide to cup and fill ray from active opening.

Input:
    A gray cup-like 3-sided frame with one opening and one blue object.
Output:
    Object moved to contact, cavity filled by distance color, rays emit from active opening.

Reasoning:
    Infer the open side from wall-side pixel counts in cup bbox,
    move object by shortest wall distance, move direction opposite becomes active opening.

Complexity:
    O(H*W*object_cells).
    """

    g = np.array(grid, dtype=int)
    h, w = g.shape
    out = g.copy()

    wall_pos = list(zip(*np.where(g == 5)))
    if not wall_pos:
        return out
    rs = [p[0] for p in wall_pos]
    cs = [p[1] for p in wall_pos]
    top, bottom = min(rs), max(rs)
    left, right = min(cs), max(cs)

    open_counts = {
        "up": int(np.sum(g[top, left : right + 1] == 5)),
        "down": int(np.sum(g[bottom, left : right + 1] == 5)),
        "left": int(np.sum(g[top : bottom + 1, left] == 5)),
        "right": int(np.sum(g[top : bottom + 1, right] == 5)),
    }
    open_side = min(open_counts, key=open_counts.get)

    obj = [(r, c) for r in range(h) for c in range(w) if g[r, c] == 1]
    if not obj:
        return out

    def contact_steps(dr, dc, cells):
        best = None
        for r, c in cells:
            rr, cc = r + dr, c + dc
            d = 0
            while 0 <= rr < h and 0 <= cc < w and g[rr, cc] != 5:
                rr += dr
                cc += dc
                d += 1
            if 0 <= rr < h and 0 <= cc < w and g[rr, cc] == 5:
                if best is None or d - 1 < best:
                    best = d - 1
        return best if best is not None else 10 ** 9

    options = [(-1, 0, "up"), (1, 0, "down"), (0, -1, "left"), (0, 1, "right")]
    best_d = 10 ** 9
    best_dir = "up"
    best_vec = (-1, 0)
    for dr, dc, name in options:
        d = contact_steps(dr, dc, obj)
        if d < best_d:
            best_d = d
            best_dir = name
            best_vec = (dr, dc)

    if best_d == 10 ** 9:
        return out

    dr, dc = best_vec
    d = best_d
    color_map = {0: 2, 1: 2, 2: 3, 3: 4}
    fill_color = color_map.get(d, 7)

    for r, c in obj:
        out[r, c] = 0
    moved = []
    for r, c in obj:
        nr, nc = r + d * dr, c + d * dc
        if 0 <= nr < h and 0 <= nc < w:
            moved.append((nr, nc))
            out[nr, nc] = 1

    opposite = {"up": "down", "down": "up", "left": "right", "right": "left"}
    active = opposite[best_dir]

    ir1, ir2 = top + 1, bottom - 1
    ic1, ic2 = left + 1, right - 1

    # fill cavity interior
    for rr in range(ir1, ir2 + 1):
        for cc in range(ic1, ic2 + 1):
            if out[rr, cc] != 5:
                out[rr, cc] = fill_color

    if active == "right":
        for rr in range(ir1, ir2 + 1):
            cc = ic2 + 1
            while cc < w and out[rr, cc] != 5:
                if out[rr, cc] != 1:
                    out[rr, cc] = fill_color
                cc += 1
    elif active == "left":
        for rr in range(ir1, ir2 + 1):
            cc = ic1 - 1
            while cc >= 0 and out[rr, cc] != 5:
                if out[rr, cc] != 1:
                    out[rr, cc] = fill_color
                cc -= 1
    elif active == "down":
        for cc in range(ic1, ic2 + 1):
            rr = ir2 + 1
            while rr < h and out[rr, cc] != 5:
                if out[rr, cc] != 1:
                    out[rr, cc] = fill_color
                rr += 1
    else:
        for cc in range(ic1, ic2 + 1):
            rr = ir1 - 1
            while rr >= 0 and out[rr, cc] != 5:
                if out[rr, cc] != 1:
                    out[rr, cc] = fill_color
                rr -= 1

    for rr, cc in moved:
        out[rr, cc] = 1

    return out


def _draw_cup(grid: np.ndarray, top: int, left: int, size: int, open_side: str) -> None:
    bottom = top + size - 1
    right = left + size - 1
    grid[top : bottom + 1, left] = 5
    grid[top : bottom + 1, right] = 5
    if open_side != "up":
        grid[top, left : right + 1] = 5
    if open_side != "down":
        grid[bottom, left : right + 1] = 5
    if open_side != "left":
        for r in range(top, bottom + 1):
            grid[r, left] = 5
    if open_side != "right":
        for r in range(top, bottom + 1):
            grid[r, right] = 5


def generate_input() -> np.ndarray:
    h = 14
    w = 14
    g = np.zeros((h, w), dtype=int)

    open_side = str(RNG.choice(["up", "down", "left", "right"]))
    size = 8
    top = 2
    left = 3
    _draw_cup(g, top, left, size, open_side)

    closed = [s for s in ["up", "down", "left", "right"] if s != open_side]
    move_side = str(RNG.choice(closed))

    if move_side == "left":
        sr = int(RNG.integers(top + 1, top + size - 2))
        sc = int(RNG.integers(1, left))
        direction = (0, 1)
    elif move_side == "right":
        sr = int(RNG.integers(top + 1, top + size - 2))
        sc = int(RNG.integers(left + size + 1, w - 1))
        direction = (0, -1)
    elif move_side == "up":
        sr = int(RNG.integers(0, top))
        sc = int(RNG.integers(left + 1, left + size - 1))
        direction = (1, 0)
    else:
        sr = int(RNG.integers(top + size, h))
        sc = int(RNG.integers(left + 1, left + size - 1))
        direction = (-1, 0)

    obj_sz = int(RNG.integers(3, 6))
    obj = {(sr, sc)}
    obj_list = [(sr, sc)]
    tries = 0
    while len(obj) < obj_sz and tries < 5000:
        tries += 1
        r, c = obj_list[int(RNG.integers(len(obj_list)))]
        rr = r + int(RNG.choice([-1, 0, 1]))
        cc = c + int(RNG.choice([-1, 0, 1]))
        if abs(rr - r) + abs(cc - c) != 1:
            continue
        if 0 <= rr < h and 0 <= cc < w and g[rr, cc] == 0 and not (top <= rr <= top + size - 1 and left <= cc <= left + size - 1):
            if (rr, cc) not in obj:
                obj.add((rr, cc))
                obj_list.append((rr, cc))

    if len(obj) < 3:
        # fallback safe tiny object
        if sc + 1 < w and g[sr, sc + 1] == 0 and not (top <= sr <= top + size - 1 and left + 1 <= sc + 1 <= left + size - 1):
            obj.add((sr, sc + 1))
        if len(obj) < 3 and sc - 1 >= 0 and g[sr, sc - 1] == 0:
            obj.add((sr, sc - 1))

    for r, c in obj:
        g[r, c] = 1

    return g
