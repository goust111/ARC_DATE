import numpy as np

_RNG = np.random.default_rng(3102)


def _component_mask(mask: np.ndarray):
    h, w = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or visited[r, c]:
                continue
            stack = [(r, c)]
            visited[r, c] = True
            comp = []
            while stack:
                x, y = stack.pop()
                comp.append((x, y))
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < h and 0 <= ny < w and mask[nx, ny] and not visited[nx, ny]:
                        visited[nx, ny] = True
                        stack.append((nx, ny))
            comps.append(comp)
    return comps


def _traverse_path(grid: np.ndarray):
    h, w = grid.shape
    mask = grid != 0
    comps = _component_mask(mask)
    if not comps:
        return []

    path_coords = comps[0]
    if len(path_coords) == 0:
        return []

    path_set = set(path_coords)
    deg = {}
    for r, c in path_coords:
        cnt = 0
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if (nr, nc) in path_set:
                cnt += 1
        deg[(r, c)] = cnt

    endpoints = [p for p in path_coords if deg.get(p, 0) == 1]
    if len(endpoints) == 0:
        return []

    endpoints = sorted(endpoints, key=lambda x: (x[0], x[1]))
    start = endpoints[0]

    ordered = []
    prev = (-1, -1)
    cur = start
    while True:
        ordered.append(cur)
        neighbors = []
        r, c = cur
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nb = (r + dr, c + dc)
            if nb in path_set and nb != prev:
                neighbors.append(nb)
        if not neighbors:
            break
        nxt = neighbors[0]
        if nxt in ordered:
            break
        prev, cur = cur, nxt

    return ordered


def main(grid: np.ndarray):
    """Task 2 transform.

    1) Locate the 2-endpoint snake in the left area and traverse its color sequence.
    2) Extract first appearance order of non-black colors along traversal.
    3) Map order to three panels and choose one object of each target color.
    4) Render selected objects onto a 15x3 canvas, overlaying panel 3 over panel 2 over panel 1.
    """

    g = np.array(grid, dtype=int)
    h, w = g.shape
    left_w = 9
    panel_w = 3
    path = _traverse_path(g[:, :left_w])

    if not path:
        return np.zeros((h, panel_w), dtype=int)

    order = []
    for r, c in path:
        color = int(g[r, c])
        if color != 0 and color not in order:
            order.append(color)

    while len(order) < 3:
        missing = [1, 2, 3]
        for x in order:
            if x in missing:
                missing.remove(x)
        if not missing:
            break
        order.append(missing[0])

    out = np.zeros((h, panel_w), dtype=int)
    for i in range(3):
        if i >= len(order):
            continue
        c = order[i]
        c0 = left_w + i * 4
        panel = g[:, c0:c0 + panel_w]
        pos = np.argwhere(panel == c)
        for rr, cc in pos:
            out[rr, int(cc)] = c

    return out


def _build_snake(h, w):
    path = []
    for r in range(h):
        if r % 2 == 0:
            for c in range(w):
                path.append((r, c))
        else:
            for c in range(w - 1, -1, -1):
                path.append((r, c))
    return path


def generate_input():
    """Problem 2 input generator.

    1) Build a fixed 15x9 left snake region and color it with a random 1-appearance order.
    2) Build 3 right panels (width 3) separated by gray bars.
    3) Put one object per panel and color, same local offsets across panels.
    """

    h, w = 15, 20
    grid = np.zeros((h, w), dtype=int)

    colors = [1, 2, 3]
    perm = list(colors)
    _RNG.shuffle(perm)

    base_path = _build_snake(h, 9)
    # random segmentation for first-appearance order
    cuts = sorted(_RNG.choice(np.arange(1, len(base_path)), size=2, replace=False).tolist())
    segments = [cuts[0], cuts[1] - cuts[0], len(base_path) - cuts[1]]

    idx = 0
    for i in range(3):
        color = perm[i]
        for _ in range(segments[i]):
            r, c = base_path[idx]
            grid[r, c] = color
            idx += 1

    for r in range(len(base_path) - idx):
        rr, cc = base_path[idx + r]
        grid[rr, cc] = perm[min(i, 2)]

    # two vertical gray separators: between 3 panels
    grid[:, 12] = 5
    grid[:, 16] = 5

    # three objects per panel, same local offsets
    panel_starts = [9, 13, 17]
    local = {
        1: (1, 0),
        2: (6, 0),
        3: (11, 0),
    }
    for i, c0 in enumerate(panel_starts):
        for color in colors:
            lr, lc = local[color]
            r0 = lr
            c1 = lc
            for dr in [0, 1]:
                for dc in [0, 1]:
                    rr = r0 + dr
                    cc = c0 + c1 + dc
                    if 0 <= rr < h and 0 <= cc < w:
                        grid[rr, cc] = color

    return grid


if __name__ == "__main__":
    sample = generate_input()
    print(main(sample))
