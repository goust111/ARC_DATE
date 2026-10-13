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


def _contour(comp: list[tuple[int, int]], grid: np.ndarray) -> list[tuple[int, int]]:
    H, W = grid.shape
    s = set(comp)
    out = []
    for r, c in comp:
        color = grid[r, c]
        edge = False
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if nr < 0 or nr >= H or nc < 0 or nc >= W:
                edge = True
                break
            if (nr, nc) not in s and grid[nr, nc] != color:
                edge = True
                break
        if edge:
            out.append((r, c))
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Input:
        Filled non-self-overlapping colored components.
    Output:
        Only contour pixels remain, shifted downward by contour size.
    Idea:
        For each original component, keep its 4-neighbour boundary.
        Shift that boundary by perimeter P rows and layer by larger-perimeter-first order.
    Complexity:
        O(HW + C log C), C total component pixels.
    """
    inp = np.array(grid, dtype=int)
    H, W = inp.shape
    out = np.zeros((H, W), dtype=int)

    parts = []
    for comp in _components(inp != 0):
        color = inp[comp[0][0], comp[0][1]]
        border = _contour(comp, inp)
        if border:
            parts.append((len(border), color, border))

    parts.sort(key=lambda x: x[0], reverse=True)
    for perim, color, border in parts:
        for r, c in border:
            nr = r + perim
            if 0 <= nr < H:
                out[nr, c] = color
    return out


def _perim(cells: set[tuple[int, int]]) -> int:
    p = 0
    for r, c in cells:
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            if (r + dr, c + dc) not in cells:
                p += 1
    return p


def _rotated(shape, k):
    if k == 0:
        return set(shape)
    s = set(shape)
    if k == 1:
        return {(y, -x) for x, y in s}
    if k == 2:
        return {(-x, -y) for x, y in s}
    return {(-y, x) for x, y in s}


def generate_input() -> np.ndarray:
    """
    Random generator for task 4.
    Returns:
        A compact 18x18 arrangement of several irregular components.
    """
    if not hasattr(generate_input, '_rng'):
        generate_input._rng = np.random.RandomState(4404)
        generate_input._i = 0
    rs = generate_input._rng

    H = W = 18
    templates = [
        {(0, 0), (0, 1), (1, 0), (1, 1), (1, 2), (2, 1)},
        {(0, 0), (1, 0), (2, 0), (2, 1), (3, 1), (3, 2), (2, 2)},
        {(0, 1), (1, 0), (1, 1), (1, 2), (2, 0), (2, 2)},
        {(0, 0), (0, 1), (0, 2), (1, 1), (2, 1), (2, 2)},
        {(0, 0), (1, 0), (2, 0), (2, 1), (1, 2)},
    ]
    colors = [1, 2, 3, 4, 5, 6]

    for _case in range(300):
        g = np.zeros((H, W), dtype=int)
        used = np.zeros((H, W), dtype=bool)
        n = 3 + int(rs.randint(0, 3))
        candidates = []

        for i in range(n):
            placed = False
            for _attempt in range(250):
                base = list(templates[rs.randint(0, len(templates))])
                cells = _rotated(base, int(rs.randint(0, 4)))
                rr = [r for r, _ in cells]
                cc = [c for _, c in cells]
                h = max(rr) - min(rr) + 1
                w = max(cc) - min(cc) + 1
                ph = int(rs.randint(1, H - h - 2))
                pw = int(rs.randint(1, W - w - 2))
                tr = set((r - min(rr) + ph, c - min(cc) + pw) for r, c in cells)

                if any((r < 2 or c < 2 or r >= H - 2 or c >= W - 2) for r, c in tr):
                    continue
                if any(used[r, c] for r, c in tr):
                    continue

                for r, c in tr:
                    used[r, c] = True
                    g[r, c] = colors[i % len(colors)]
                candidates.append(tr)
                placed = True
                break
            if not placed:
                break

        if len(candidates) >= 3:
            return g

    raise RuntimeError("failed to generate a valid task-4 input")
