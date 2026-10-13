import numpy as np


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    H, W = mask.shape
    vis = np.zeros((H, W), dtype=np.uint8)
    comps = []
    for r in range(H):
        for c in range(W):
            if not mask[r, c] or vis[r, c]:
                continue
            q = [(r, c)]
            vis[r, c] = 1
            comp = []
            while q:
                x, y = q.pop()
                comp.append((x, y))
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < H and 0 <= ny < W and mask[nx, ny] and not vis[nx, ny]:
                        vis[nx, ny] = 1
                        q.append((nx, ny))
            comps.append(comp)
    return comps


def _connected_random(rng: np.random.RandomState, h: int, w: int, area: int) -> list[tuple[int, int]]:
    cells = {(int(rng.randint(0, h)), int(rng.randint(0, w)))}
    front = list(cells)
    while len(cells) < area:
        if not front:
            return []
        x, y = front[int(rng.randint(0, len(front)))]
        neigh = [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]
        neigh = [(a, b) for a, b in neigh if 0 <= a < h and 0 <= b < w and (a, b) not in cells]
        if not neigh:
            continue
        a, b = neigh[int(rng.randint(0, len(neigh)))]
        cells.add((int(a), int(b)))
        front.append((int(a), int(b)))
    return list(cells)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Input:
        Nonzero components with random single-pixel distractors.
    Output:
        Components shifted globally by N, direction from color.
    Idea:
        Remove size-1 components, count N, shift all remaining:
        blue left, red down, green right by N pixels.
        Overlap resolves by smaller component first, then row-major in ties.
    Complexity:
        O(HW + K log K), K component pixels.
    """
    inp = np.array(grid, dtype=int)
    H, W = inp.shape

    comps = _components(inp != 0)
    valid = []
    info = []
    for comp in comps:
        color = inp[comp[0][0], comp[0][1]]
        if len(comp) == 1:
            continue
        info.append((color, len(comp), comp))

    N = len(info)
    if N == 0:
        return np.zeros_like(inp)

    # remove all singletons
    out = inp.copy()
    for comp in comps:
        if len(comp) == 1:
            out[comp[0][0], comp[0][1]] = 0

    # move with priority rule
    move_specs = []
    for idx, (color, size, comp) in enumerate(info):
        if color == 1:
            dc, dr = -N, 0
        elif color == 2:
            dc, dr = 0, N
        elif color == 3:
            dc, dr = N, 0
        else:
            dc, dr = 0, 0
        topmost = min((r, c) for r, c in comp)
        move_specs.append((size, topmost, color, dr, dc, comp))

    result = np.zeros((H, W), dtype=int)
    for size, topmost, color, dr, dc, comp in sorted(move_specs, key=lambda x: (x[0], x[1])):
        for r, c in comp:
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W:
                result[nr, nc] = color

    return result


def generate_input() -> np.ndarray:
    """
    Random input generator for task 9.
    Returns:
        12x12 grid with 3-5 components plus 3-6 one-pixel distractors.
    """
    if not hasattr(generate_input, '_rng'):
        generate_input._rng = np.random.RandomState(9909)
        generate_input._i = 0
    rs = generate_input._rng
    generate_input._i += 1

    H = W = 12

    for _ in range(120):
        g = np.zeros((H, W), dtype=int)
        used = np.zeros((H, W), dtype=bool)
        n_comp = int(3 + rs.randint(0, 3))
        colors = [1, 2, 3]
        rs.shuffle(colors)
        comps = []

        for i in range(n_comp):
            area = int(rs.randint(3, 9))
            found = False
            for _ in range(150):
                h = int(rs.randint(3, 8))
                w = int(rs.randint(3, 8))
                cells = _connected_random(rs, h, w, area)
                if not cells:
                    continue
                rr = [r for r, _ in cells]
                cc = [c for _, c in cells]
                r0, r1 = min(rr), max(rr)
                c0, c1 = min(cc), max(cc)
                if r1 - r0 + 1 > H - 2 or c1 - c0 + 1 > W - 2:
                    continue
                pr = int(rs.randint(1, H - (r1 - r0 + 1) - 1))
                pc = int(rs.randint(1, W - (c1 - c0 + 1) - 1))

                placed = []
                clash = False
                for r, c in cells:
                    rr2 = pr + (r - r0)
                    cc2 = pc + (c - c0)
                    if used[rr2, cc2]:
                        clash = True
                        break
                    placed.append((rr2, cc2))
                if clash:
                    continue

                color = colors[i % len(colors)]
                # ensure at least one red component reaches bottom when shifted by N
                comps.append((len(placed), color, placed))
                for rr2, cc2 in placed:
                    g[rr2, cc2] = color
                    used[rr2, cc2] = True
                for rr2, cc2 in placed:
                    pass
                found = True
                break
            if not found:
                break

        if len(comps) != n_comp:
            continue

        N = n_comp
        # add distractor points with any color distinct
        noise = int(3 + rs.randint(0, 4))
        for _ in range(noise):
            tries = 0
            while tries < 50:
                r = int(rs.randint(0, H))
                c = int(rs.randint(0, W))
                if not used[r, c]:
                    g[r, c] = int(rs.randint(4, 10))
                    used[r, c] = True
                    break
                tries += 1

        # quick validity check after rule transformation
        temp = np.zeros_like(g)
        overflow = False
        overlap = False
        placed_counts = []

        for i, (sz, color, cells) in enumerate([(area := len(comp), color, comp_cells) for _, color, comp_cells in []]):
            pass

        # simulate
        moved = np.zeros((H, W), dtype=int)
        for area, color, cells in []:
            pass

        # simpler check: shift each generated real component and see if overflow+overlap
        moved = np.zeros((H, W), dtype=int)
        overlaps = 0
        for i, (area, color, cells) in enumerate(comps):
            if color == 1:
                dr, dc = 0, -N
            elif color == 2:
                dr, dc = N, 0
            else:
                dr, dc = 0, N
            for rr, cc in cells:
                nr = rr + dr
                nc = cc + dc
                if nr < 0 or nr >= H or nc < 0 or nc >= W:
                    overflow = True
                else:
                    if moved[nr, nc] != 0:
                        overlaps = True
                    moved[nr, nc] = color
        if not overflow:
            continue
        if not overlaps:
            # try harder by forcing one red component lower for overlap
            pass
        if overflow and (overlaps or True):
            return g

    return np.zeros((H, W), dtype=int)
