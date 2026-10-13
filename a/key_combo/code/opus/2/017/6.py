import numpy as np

_RNG = np.random.default_rng(3106)


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


def _has_hole(g: np.ndarray, comp):
    rs = [p[0] for p in comp]
    cs = [p[1] for p in comp]
    r0, r1 = min(rs), max(rs)
    c0, c1 = min(cs), max(cs)

    patch = g[r0:r1 + 1, c0:c1 + 1] == 2
    h, w = patch.shape
    vis = np.zeros_like(patch, dtype=bool)

    def _bfs(start):
        stack = [start]
        vis[start[0], start[1]] = True
        while stack:
            x, y = stack.pop()
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nx, ny = x + dx, y + dy
                if 0 <= nx < h and 0 <= ny < w and not patch[nx, ny] and not vis[nx, ny]:
                    vis[nx, ny] = True
                    stack.append((nx, ny))

    # mark zero area connected to border as outside
    for x in range(h):
        if not patch[x, 0] and not vis[x, 0]:
            _bfs((x, 0))
        if not patch[x, w - 1] and not vis[x, w - 1]:
            _bfs((x, w - 1))
    for y in range(w):
        if not patch[0, y] and not vis[0, y]:
            _bfs((0, y))
        if not patch[h - 1, y] and not vis[h - 1, y]:
            _bfs((h - 1, y))

    for x in range(h):
        for y in range(w):
            if not patch[x, y] and not vis[x, y]:
                return True
    return False


def main(grid: np.ndarray):
    """Task 6 transform.

    1) Split the red objects into connected components and test each for internal black holes.
    2) Count hole-bearing vs non-hole objects to find the minority class.
    3) Mirror only minority-class objects across the selected symmetry axis.
    4) Draw mirrored copies (cyan/oange) while original red objects remain unchanged.
    """

    g = np.array(grid, dtype=int)
    h, w = g.shape
    out = g.copy()

    mask = g == 2
    comps = _components(mask)
    hole_group = []
    solid_group = []

    for comp in comps:
        if _has_hole(g, comp):
            hole_group.append(comp)
        else:
            solid_group.append(comp)

    H = len(hole_group)
    S = len(solid_group)

    if H == S:
        return out

    if H < S:
        minority = hole_group
        color = 8
        for comp in minority:
            for r, c in comp:
                rr = r
                cc = (w - 1) - c
                out[rr, cc] = color
    else:
        minority = solid_group
        color = 7
        for comp in minority:
            for r, c in comp:
                rr = (h - 1) - r
                cc = c
                out[rr, cc] = color

    return out


def _draw_object(g, cells, rng):
    for r, c in cells:
        g[r, c] = 2


def _random_walk_cells(size, width=15, height=15, start=None, rng=None):
    rng = rng or _RNG
    if start is None:
        start = (rng.integers(1, height - 1), rng.integers(1, width - 1))
    current = set([start])
    frontier = [start]
    dirs = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    while len(current) < size:
        r, c = rng.choice(frontier)
        dr, dc = dirs[int(rng.integers(4))]
        nr, nc = r + dr, c + dc
        if (nr, nc) in current:
            continue
        if nr <= 0 or nr >= height - 1 or nc <= 0 or nc >= width - 1:
            continue
        current.add((nr, nc))
        frontier.append((nr, nc))
    return list(current)


def _ring3x3(start_r, start_c):
    cells = []
    for dc in range(3):
        cells.append((start_r, start_c + dc))
        cells.append((start_r + 2, start_c + dc))
    for dr in (1,):
        cells.append((start_r + dr, start_c))
        cells.append((start_r + dr, start_c + 2))
    return sorted(set(cells))


def generate_input():
    """Task 6 input generator.

    1) Build 4-6 disjoint red objects in a 15x15 canvas.
    2) Split objects into hole-bearing and solid categories with unequal counts.
    3) Ensure objects do not touch and use random 4-neighbor connectivity sizes 4-10.
    """

    h, w = 15, 15
    g = np.zeros((h, w), dtype=int)
    occ = np.zeros_like(g, dtype=bool)

    total = int(_RNG.integers(4, 7))
    while True:
        holes = int(_RNG.integers(1, total))
        solids = total - holes
        if holes != solids:
            break
    if holes < 0 or solids < 0:
        holes = total // 2
        solids = total - holes

    targets = []
    for _ in range(holes):
        targets.append(8)  # marker only, overwritten

    for i in range(total):
        need_hole = i < holes
        for _ in range(160):
            if need_hole:
                size = int(_RNG.integers(8, 11))
                if size > 10:
                    size = 10
                if size < 8:
                    size = 8
                r0 = int(_RNG.integers(1, h - 3))
                c0 = int(_RNG.integers(1, w - 3))
                cells = _ring3x3(r0, c0)
                extra = size - len(cells)
                if extra > 0:
                    candidates = [
                        (r0, c0 + 1), (r0 + 1, c0), (r0 + 1, c0 + 1), (r0 + 1, c0 + 2),
                        (r0 + 2, c0 + 1),
                    ]
                    _RNG.shuffle(candidates)
                    for p in candidates[:extra]:
                        if p not in cells:
                            cells.append(p)
            else:
                size = int(_RNG.integers(4, 11))
                sr = int(_RNG.integers(1, h - 1))
                sc = int(_RNG.integers(1, w - 1))
                if occ[sr, sc] or any(abs(sr - x) <= 1 and abs(sc - y) <= 1 for x, y in np.argwhere(occ)):
                    continue
                cells = _random_walk_cells(size, width=w, height=h, start=(sr, sc), rng=_RNG)

            if not cells:
                continue
            rr = min(r for r, c in cells)
            rc = max(r for r, c in cells)
            cc = min(c for r, c in cells)
            cd = max(c for r, c in cells)
            if rr <= 0 or rc >= h - 1 or cc <= 0 or cd >= w - 1:
                continue
            if np.any(occ[max(0, rr - 1):min(h, rc + 2), max(0, cc - 1):min(w, cd + 2)]):
                continue

            for rr, cc in cells:
                if rr < 0 or rr >= h or cc < 0 or cc >= w:
                    break
                g[rr, cc] = 2
                occ[rr, cc] = True
            break

    return g


if __name__ == "__main__":
    sample = generate_input()
    print(main(sample))
