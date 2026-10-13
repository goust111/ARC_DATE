import random
import numpy as np


def _find_period_1d(values, max_p: int) -> int | None:
    n = len(values)
    for p in range(1, max_p + 1):
        ok = True
        for i in range(n - p):
            a = int(values[i])
            b = int(values[i + p])
            if a != 0 and b != 0 and a != b:
                ok = False
                break
        if ok:
            return p
    return None


def _inpaint_region(upper: np.ndarray) -> np.ndarray:
    up = np.array(upper, dtype=int, copy=True)
    h, w = up.shape

    pr = _find_period_1d(up[:, 0].tolist(), 4)  # rough estimate by vertical consistency
    pc = None
    for p in range(1, 5):
        ok = True
        for j in range(w - p):
            col0 = up[:, j]
            col1 = up[:, j + p]
            for a, b in zip(col0, col1):
                if int(a) != 0 and int(b) != 0 and int(a) != int(b):
                    ok = False
                    break
            if not ok:
                break
        if ok:
            pc = p
            break

    # recompute row period with stronger condition
    pr = None
    for p in range(1, 5):
        ok = True
        for i in range(h - p):
            for j in range(w):
                a = int(up[i, j])
                b = int(up[i + p, j])
                if a != 0 and b != 0 and a != b:
                    ok = False
                    break
            if not ok:
                break
        if ok:
            pr = p
            break

    vecs = []
    if pr is not None and pr >= 1:
        vecs.append((pr, 0))
    if pc is not None and pc >= 1:
        vecs.append((0, pc))
    if not vecs:
        return up

    visited = np.zeros((h, w), dtype=bool)

    def _fill_from(start_r: int, start_c: int):
        stack = [(start_r, start_c)]
        seen = set([(start_r, start_c)])
        colors = set()
        while stack:
            r, c = stack.pop()
            v = int(up[r, c])
            if v != 0:
                colors.add(v)
            for dr, dc in vecs:
                for k in (-1, 1):
                    nr, nc = r + k * dr, c + k * dc
                    if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc]:
                        visited[nr, nc] = True
                        stack.append((nr, nc))
                        seen.add((nr, nc))
        if len(colors) == 1:
            color = next(iter(colors))
            for rr, cc in seen:
                up[rr, cc] = color

    for r in range(h):
        for c in range(w):
            if up[r, c] == 0 and not visited[r, c]:
                visited[r, c] = True
                _fill_from(r, c)

    return up


def main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int, copy=False)
    h, w = g.shape
    sep_rows = [int(r) for r in np.where(np.all(g == 5, axis=1))[0]]
    if not sep_rows:
        return g.copy()
    sep = sep_rows[0]

    upper = g[:sep, :]
    lower = g[sep + 1 :, :]

    repaired_upper = _inpaint_region(upper)

    col_counts = [int(np.count_nonzero(repaired_upper[:, c])) for c in range(w)]
    order = np.argsort(col_counts)
    reordered = lower[:, order]

    out = np.zeros_like(g)
    out[:sep, :] = repaired_upper
    out[sep, :] = 5
    out[sep + 1 :, :] = reordered
    return out


def generate_input() -> np.ndarray:
    upper_h = 6
    lower_h = 3
    w = 5
    h = upper_h + 1 + lower_h
    sep = upper_h
    g = np.zeros((h, w), dtype=int)

    g[sep, :] = 5

    ph = random.randint(2, 4)
    pw = random.randint(2, 4)
    palette = [1, 2, 3]
    tile = np.random.choice(palette, size=(ph, pw)).tolist()
    for r in range(upper_h):
        for c in range(w):
            g[r, c] = tile[r % ph][c % pw]

    black_cells = set()
    while len(black_cells) < random.randint(3, 5):
        rr = random.randrange(upper_h)
        cc = random.randrange(w)
        black_cells.add((rr, cc))
    for rr, cc in black_cells:
        g[rr, cc] = 0

    lower_colors = random.sample([4, 6, 7, 8, 9], w)
    for c in range(w):
        height = random.randint(1, 3)
        for rr in range(h - 1, h - 1 - height, -1):
            g[rr, c] = lower_colors[c]

    return g
