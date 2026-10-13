import random
from collections import deque
import numpy as np


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(comp: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rs = [r for r, _ in comp]
    cs = [c for _, c in comp]
    return min(rs), max(rs), min(cs), max(cs)


def main(grid: np.ndarray) -> np.ndarray:
    h, w = grid.shape
    div = max(range(w), key=lambda c: int(np.sum(grid[:, c] == 5)))
    left_mask = grid[:, :div] != 0
    right_mask = grid[:, div + 1 :] != 0
    left_mask &= grid[:, :div] != 5
    right_mask &= grid[:, div + 1 :] != 5

    left = _components(left_mask)
    right = _components(right_mask)
    stamp = max(right, key=len)
    stamp = [(r, c + div + 1) for r, c in stamp]
    target_size = len(stamp)
    templ = next(comp for comp in left if len(comp) == target_size)

    tr0, tr1, tc0, tc1 = _bbox(templ)
    sr0, sr1, sc0, sc1 = _bbox(stamp)
    th, tw = tr1 - tr0 + 1, tc1 - tc0 + 1
    sh, sw = sr1 - sr0 + 1, sc1 - sc0 + 1
    out = np.zeros((th * sh, tw * sw), dtype=int)
    stamp_arr = grid[sr0 : sr1 + 1, sc0 : sc1 + 1]

    template_cells = {(r, c) for r, c in templ}
    for rr in range(th):
        for cc in range(tw):
            if (tr0 + rr, tc0 + cc) in template_cells:
                block = out[rr * sh : (rr + 1) * sh, cc * sw : (cc + 1) * sw]
                block[stamp_arr != 0] = stamp_arr[stamp_arr != 0]
    return out


def _shape(n: int) -> set[tuple[int, int]]:
    cells = {(0, 0)}
    while len(cells) < n:
        r, c = random.choice(tuple(cells))
        dr, dc = random.choice(((1, 0), (-1, 0), (0, 1), (0, -1)))
        cells.add((r + dr, c + dc))
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    return {(r - min_r, c - min_c) for r, c in cells}


def _place(grid: np.ndarray, shape: set[tuple[int, int]], rows: range, cols: range, color: int) -> bool:
    sh = max(r for r, _ in shape) + 1
    sw = max(c for _, c in shape) + 1
    spots = [(r, c) for r in rows if r + sh <= rows.stop for c in cols if c + sw <= cols.stop]
    random.shuffle(spots)
    for r0, c0 in spots:
        cells = [(r0 + r, c0 + c) for r, c in shape]
        pad = [(r, c) for rr, cc in cells for r in range(rr - 1, rr + 2) for c in range(cc - 1, cc + 2)]
        if all(0 <= r < grid.shape[0] and 0 <= c < grid.shape[1] and grid[r, c] == 0 for r, c in pad):
            for r, c in cells:
                grid[r, c] = color
            return True
    return False


def generate_input() -> np.ndarray:
    for _ in range(300):
        h, w, div = 10, 20, 10
        grid = np.zeros((h, w), dtype=int)
        grid[:, div] = 5
        sizes = random.sample([3, 4, 5, 6], random.randint(3, 4))
        match = random.choice(sizes)
        colors = random.sample([1, 2, 3, 7, 9], len(sizes))
        ok = True
        for size, color in zip(sizes, colors):
            ok &= _place(grid, _shape(size), range(1, h - 1), range(1, div - 1), color)
        ok &= _place(grid, _shape(match), range(1, h - 1), range(div + 2, w - 1), 4)
        if ok:
            return grid
    return grid
