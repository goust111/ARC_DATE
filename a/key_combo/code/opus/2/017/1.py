import numpy as np

_RNG = np.random.default_rng(3101)


def _components(mask: np.ndarray):
    h, w = mask.shape
    vis = np.zeros_like(mask, dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or vis[r, c]:
                continue
            stack = [(r, c)]
            vis[r, c] = True
            comp = []
            while stack:
                x, y = stack.pop()
                comp.append((x, y))
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < h and 0 <= ny < w and mask[nx, ny] and not vis[nx, ny]:
                        vis[nx, ny] = True
                        stack.append((nx, ny))
            comps.append(comp)
    return comps


def _bbox(coords):
    rs = [p[0] for p in coords]
    cs = [p[1] for p in coords]
    return min(rs), max(rs), min(cs), max(cs)


def _draw_hollow_ring(grid: np.ndarray, r0, c0, r1, c1, color: int):
    grid[r0, c0:c1 + 1] = color
    grid[r1, c0:c1 + 1] = color
    grid[r0:r1 + 1, c0] = color
    grid[r0:r1 + 1, c1] = color


def _find_tail_and_depth(pts, h, w):
    depth = len(pts)
    sizes = [len(p) for p in pts]
    outer = pts[int(np.argmax(sizes))]
    outer_set = set(outer)
    r0, r1, c0, c1 = _bbox(outer)
    cm = (c0 + c1) // 2
    rm = (r0 + r1) // 2
    direction = None
    tail = None
    if (r0 - 1, cm) in outer_set:
        direction = (-1, 0)
        tail = (r0 - 1, cm)
    elif (r1 + 1, cm) in outer_set:
        direction = (1, 0)
        tail = (r1 + 1, cm)
    elif (rm, c0 - 1) in outer_set:
        direction = (0, -1)
        tail = (rm, c0 - 1)
    elif (rm, c1 + 1) in outer_set:
        direction = (0, 1)
        tail = (rm, c1 + 1)

    if direction is None:
        return None
    return depth, direction, tail


def main(grid: np.ndarray):
    """Task 1 transform.

    1) Find each non-zero color and split into connected frame components.
    2) Component count of a color gives ejection depth.
    3) Remove all frame pixels, then draw depth-long tail trail from detected spike.
    4) Keep only generated trails (and unrelated colors) in output.
    """

    g = np.array(grid, dtype=int)
    out = g.copy()

    for color in [int(v) for v in np.unique(g) if v != 0]:
        comps = _components(g == color)
        if not comps:
            continue

        for comp in comps:
            for r, c in comp:
                out[r, c] = 0

        info = _find_tail_and_depth(comps, *out.shape)
        if info is None:
            continue
        depth, (dr, dc), (sr, sc) = info
        for k in range(depth):
            rr = sr + dr * k
            cc = sc + dc * k
            if 0 <= rr < out.shape[0] and 0 <= cc < out.shape[1]:
                out[rr, cc] = color

    return out


def _dilate(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    out = np.zeros_like(mask, dtype=bool)
    for r in range(h):
        for c in range(w):
            if not mask[r, c]:
                continue
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    rr = r + dr
                    cc = c + dc
                    if 0 <= rr < h and 0 <= cc < w:
                        out[rr, cc] = True
    return out


def _place_group(g: np.ndarray, occupied: np.ndarray, color: int, depth: int, trials: int = 120) -> bool:
    h, w = g.shape
    min_side = 4 * (depth - 1) + 5
    max_side = h - 2
    if min_side > max_side:
        return False

    directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]
    for _ in range(trials):
        side = int(_RNG.choice(np.arange(min_side, max_side + 1, 2)))
        r0 = int(_RNG.integers(1, h - side))
        c0 = int(_RNG.integers(1, w - side))
        r1 = r0 + side - 1
        c1 = c0 + side - 1

        temp = np.zeros_like(g, dtype=bool)
        for lv in range(depth):
            rr0 = r0 + 2 * lv
            cc0 = c0 + 2 * lv
            rr1 = r1 - 2 * lv
            cc1 = c1 - 2 * lv
            if rr0 >= rr1 or cc0 >= cc1:
                break
            _draw_hollow_ring(temp, rr0, cc0, rr1, cc1, True)

        any_pixel = temp.any()
        if not any_pixel:
            continue

        dir_dr, dir_dc = directions[int(_RNG.integers(len(directions)))]
        if dir_dr == -1:
            tail = (r0 - 1, (c0 + c1) // 2)
        elif dir_dr == 1:
            tail = (r1 + 1, (c0 + c1) // 2)
        elif dir_dc == -1:
            tail = ((r0 + r1) // 2, c0 - 1)
        else:
            tail = ((r0 + r1) // 2, c1 + 1)

        if not (0 <= tail[0] < h and 0 <= tail[1] < w):
            continue
        temp[tail] = True

        if np.any(occupied & _dilate(temp)):
            continue

        g[temp] = color
        occupied |= _dilate(temp)
        return True

    return False


def generate_input():
    """Task 1 input generator.

    1) Create a 15x15 grid.
    2) Place 2-3 groups; each group is same color nested hollow frames with different depths.
    3) Give each outer frame one spike pixel in its chosen direction.
    """

    h, w = 15, 15
    g = np.zeros((h, w), dtype=int)
    occupied = np.zeros((h, w), dtype=bool)

    group_count = int(_RNG.integers(2, 4))
    colors = [2, 3, 4, 1]
    _RNG.shuffle(colors)
    depths = list(range(1, group_count + 1))
    _RNG.shuffle(depths)

    for i in range(group_count):
        color = colors[i]
        _place_group(g, occupied, color, depths[i])

    return g


if __name__ == "__main__":
    print(main(generate_input()))
