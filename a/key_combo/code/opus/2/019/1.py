import numpy as np


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    H, W = mask.shape
    vis = np.zeros((H, W), dtype=np.uint8)
    out = []
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
            out.append(comp)
    return out


def _tile_component(cells: list[tuple[int, int]]) -> list[tuple[list[tuple[int, int]], int]] | None:
    if not cells:
        return []
    rs = [r for r, _ in cells]
    cs = [c for _, c in cells]
    r0, c0 = min(rs), min(cs)
    r1, c1 = max(rs), max(cs)
    h, w = r1 - r0 + 1, c1 - c0 + 1

    idx = {}
    id_rc = {}
    for i, (r, c) in enumerate(cells):
        id_rc[i] = (r, c)
        idx[(r - r0, c - c0)] = i

    mask = np.zeros((h, w), dtype=np.uint8)
    for r, c in cells:
        mask[r - r0, c - c0] = 1

    total = len(cells)
    full = (1 << total) - 1

    cand_by_idx = [[] for _ in range(total)]

    for rr in range(h - 1):
        for cc in range(w - 1):
            pts = [(rr, cc), (rr + 1, cc), (rr, cc + 1), (rr + 1, cc + 1)]
            if all(mask[a, b] for a, b in pts):
                bits = 0
                for p in pts:
                    bits |= 1 << idx[p]
                for p in pts:
                    cand_by_idx[idx[p]].append((bits, 8))

    for rr in range(h):
        for cc in range(w - 2):
            pts = [(rr, cc), (rr, cc + 1), (rr, cc + 2)]
            if all(mask[a, b] for a, b in pts):
                bits = 0
                for p in pts:
                    bits |= 1 << idx[p]
                for p in pts:
                    cand_by_idx[idx[p]].append((bits, 2))

    for rr in range(h - 2):
        for cc in range(w):
            pts = [(rr, cc), (rr + 1, cc), (rr + 2, cc)]
            if all(mask[a, b] for a, b in pts):
                bits = 0
                for p in pts:
                    bits |= 1 << idx[p]
                for p in pts:
                    cand_by_idx[idx[p]].append((bits, 2))

    memo = {}

    def dfs(used: int):
        if used in memo:
            return memo[used]
        if used == full:
            return []
        rem = full & ~used
        i = (rem & -rem).bit_length() - 1
        for bits, color in cand_by_idx[i]:
            if bits & used:
                continue
            nxt = used | bits
            if nxt == used:
                continue
            tail = dfs(nxt)
            if tail is not None:
                memo[used] = [(bits, color)] + tail
                return memo[used]
        memo[used] = None
        return None

    sol = dfs(0)
    if sol is None:
        return None
    out = []
    for bits, color in sol:
        pix = []
        b = bits
        while b:
            lb = b & -b
            j = lb.bit_length() - 1
            pix.append(id_rc[j])
            b ^= lb
        out.append((pix, color))
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Input:
        Input grid with several gray(5) shapes each tagged by a convex ear pixel.
    Output:
        Grid after shape mirroring, ear removal, contour-like decomposition and recolor.
    Idea:
        For each component, mirror gray cells across the side opposite the ear, remove all
        ear pixels, tile resulting pixels by 2x2 or 1x3/3x1 primitives.
    Complexity:
        O(A * P), A=active shape area, P=recursion state count in tiling.
    """
    inp = np.array(grid, dtype=int)
    H, W = inp.shape
    out = np.zeros((H, W), dtype=int)

    comps = _components(inp != 0)
    for comp in comps:
        gray = [(r, c) for r, c in comp if inp[r, c] == 5]
        if not gray:
            continue
        ears = [(r, c) for r, c in comp if inp[r, c] != 5]
        if not ears:
            continue

        gr = [r for r, _ in gray]
        gc = [c for _, c in gray]
        rmin, rmax = min(gr), max(gr)
        cmin, cmax = min(gc), max(gc)

        ear_r, ear_c = ears[0]
        if ear_c < cmin:
            side = 'L'
        elif ear_c > cmax:
            side = 'R'
        elif ear_r < rmin:
            side = 'U'
        else:
            side = 'D'

        def mirror(pt):
            r, c = pt
            if side == 'L':
                return r, 2 * cmax - c + 1
            if side == 'R':
                return r, 2 * cmin - c - 1
            if side == 'U':
                return 2 * rmax - r + 1, c
            return 2 * rmin - r - 1, c

        target = set()
        for p in gray:
            target.add(p)
            mp = mirror(p)
            if 0 <= mp[0] < H and 0 <= mp[1] < W:
                target.add(mp)

        # decompose each connected target piece
        mask = np.zeros((H, W), dtype=np.uint8)
        for r, c in target:
            mask[r, c] = 1
        for piece in _components(mask):
            sol = _tile_component(piece)
            if sol is None:
                continue
            for cells, color in sol:
                for rr, cc in cells:
                    out[rr, cc] = color

    return out


def _rand_connected_coords(rng: np.random.RandomState, h: int, w: int, target: int) -> list[tuple[int, int]] | None:
    for _ in range(90):
        rs, cs = rng.randint(0, h), rng.randint(0, w)
        comp = {(int(rs), int(cs))}
        frontier = [(int(rs), int(cs))]
        while len(comp) < target:
            if not frontier:
                break
            x, y = frontier[rng.randint(0, len(frontier))]
            cand = []
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nx, ny = x + dx, y + dy
                if 0 <= nx < h and 0 <= ny < w and (nx, ny) not in comp:
                    cand.append((nx, ny))
            if not cand:
                continue
            nx, ny = cand[rng.randint(0, len(cand))]
            comp.add((nx, ny))
            frontier.append((nx, ny))
        if len(comp) == target:
            return sorted(comp)
    return None


def _tileable_cells(cells: list[tuple[int, int]]) -> bool:
    if not cells:
        return False
    rr = [r for r, _ in cells]
    cc = [c for _, c in cells]
    H, W = max(rr) + 1, max(cc) + 1
    tmp = np.zeros((H, W), dtype=np.uint8)
    for r, c in cells:
        tmp[r, c] = 1
    for p in _components(tmp):
        if _tile_component(p) is None:
            return False
    return True


def generate_input() -> np.ndarray:
    """
    Random generator for task 1.
    Returns:
        A 30x30 grid containing 2-4 convex-ear constructions.
    """
    if not hasattr(generate_input, '_rng'):
        generate_input._rng = np.random.RandomState(1301)
        generate_input._i = 0
    rs = generate_input._rng
    generate_input._i += 1

    size = 30
    ear_colors = [1, 2, 3]

    for _ in range(200):
        g = np.zeros((size, size), dtype=int)
        n_shapes = int(2 + rs.randint(0, 3))
        ok_all = True

        for si in range(n_shapes):
            ear_color = ear_colors[min(si, 2)]
            found = False
            for _ in range(300):
                h = int(rs.randint(3, 5))
                w = int(rs.randint(3, 5))
                target = int(rs.randint(5, 10))
                raw = _rand_connected_coords(rs, h, w, target)
                if raw is None:
                    continue

                rmin, rmax = min(r for r, _ in raw), max(r for r, _ in raw)
                cmin, cmax = min(c for _, c in raw), max(c for _, c in raw)
                side = int(rs.randint(0, 4))
                if side == 0 and cmin == 0:
                    continue
                if side == 1 and cmax == w - 1:
                    continue
                if side == 2 and rmin == 0:
                    continue
                if side == 3 and rmax == h - 1:
                    continue

                if side == 0:
                    ear = (int(rs.randint(rmin, rmax + 1)), cmin - 1)
                    mirror = lambda p: (p[0], 2 * cmax - p[1] + 1)
                elif side == 1:
                    ear = (int(rs.randint(rmin, rmax + 1)), cmax + 1)
                    mirror = lambda p: (p[0], 2 * cmin - p[1] - 1)
                elif side == 2:
                    ear = (rmin - 1, int(rs.randint(cmin, cmax + 1)))
                    mirror = lambda p: (2 * rmax - p[0] + 1, p[1])
                else:
                    ear = (rmax + 1, int(rs.randint(cmin, cmax + 1)))
                    mirror = lambda p: (2 * rmin - p[0] - 1, p[1])

                raw_set = set(raw)
                if ear in raw_set:
                    continue

                target_set = set(raw_set)
                for p in raw_set:
                    mp = mirror(p)
                    if 0 <= mp[0] < h and 0 <= mp[1] < w:
                        target_set.add(mp)

                r0, c0 = min(x for x, _ in target_set), min(y for _, y in target_set)
                target_norm = {(x - r0, y - c0) for x, y in target_set}
                raw_norm = {(x - r0, y - c0) for x, y in raw_set}
                # keep a construction where mirror has a valid-looking paired shape
                # (heuristic: accept this shape directly to avoid heavy exact tiling search in generator)

                hr = max(x for x, _ in target_set) - min(x for x, _ in target_set)
                wc = max(y for _, y in target_set) - min(y for _, y in target_set)
                candidates_r = list(range(2, size - hr - 2))
                candidates_c = list(range(2, size - wc - 2))
                if not candidates_r or not candidates_c:
                    continue

                for _ in range(80):
                    off_r = int(rs.choice(candidates_r)) - min(x for x, _ in raw_set)
                    off_c = int(rs.choice(candidates_c)) - min(y for _, y in raw_set)

                    all_cells = [(r + off_r, c + off_c) for r, c in raw_set]
                    ear_cell = (ear[0] + off_r, ear[1] + off_c)
                    if any(r < 2 or r >= size - 2 or c < 2 or c >= size - 2 for r, c in all_cells):
                        continue
                    if g[ear_cell] != 0:
                        continue
                    if any(g[r, c] != 0 for r, c in all_cells):
                        continue

                    for r, c in all_cells:
                        g[r, c] = 5
                    g[ear_cell] = ear_color
                    found = True
                    break
                if found:
                    break

            if not found:
                ok_all = False
                break

        if ok_all:
            if np.count_nonzero(g == 5) >= 20:
                return g

    return np.zeros((size, size), dtype=int)
