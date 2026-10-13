import random
import numpy as np


def main(grid: np.ndarray) -> np.ndarray:
    out = grid.copy()
    pts = np.argwhere(grid == 5)
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)

    mr = mc = None
    h, w = grid.shape
    for r in range(h - 1):
        for c in range(w - 1):
            block = grid[r:r + 2, c:c + 2]
            vals = block.ravel().tolist()
            if all(v not in (0, 5) for v in vals) and len(set(vals)) == 4:
                if r + 1 < r0 - 1 or r > r1 + 1 or c + 1 < c0 - 1 or c > c1 + 1:
                    mr, mc = r, c
                    break
        if mr is not None:
            break

    block = grid[mr:mr + 2, mc:mc + 2]
    colors = {
        "tl": int(block[1, 1]),
        "tr": int(block[1, 0]),
        "bl": int(block[0, 1]),
        "br": int(block[0, 0]),
    }
    rm = (r0 + r1) // 2
    cm = (c0 + c1) // 2
    for r, c in pts:
        if r <= rm and c <= cm:
            out[r, c] = colors["tl"]
        elif r <= rm and c > cm:
            out[r, c] = colors["tr"]
        elif r > rm and c <= cm:
            out[r, c] = colors["bl"]
        else:
            out[r, c] = colors["br"]
    return out


def _connected(cells):
    cells = set(cells)
    stack = [next(iter(cells))]
    seen = {stack[0]}
    while stack:
        r, c = stack.pop()
        for nr, nc in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)):
            if (nr, nc) in cells and (nr, nc) not in seen:
                seen.add((nr, nc))
                stack.append((nr, nc))
    return len(seen) == len(cells)


def generate_input() -> np.ndarray:
    for _ in range(1000):
        h = random.randint(12, 15)
        w = random.randint(12, 15)
        bh, bw = random.choice([(4, 6), (6, 4), (6, 6)])
        r0 = random.randint(2, h - bh - 3)
        c0 = random.randint(2, w - bw - 3)
        all_cells = [(r, c) for r in range(bh) for c in range(bw)]
        target = random.randint(15, min(25, bh * bw - 2))
        cells = set(all_cells)
        order = all_cells[:]
        random.shuffle(order)
        for cell in order:
            if len(cells) <= target:
                break
            trial = cells - {cell}
            rows = [r for r, _ in trial]
            cols = [c for _, c in trial]
            if min(rows) == 0 and max(rows) == bh - 1 and min(cols) == 0 and max(cols) == bw - 1 and _connected(trial):
                cells = trial
        if len(cells) < 15 or len(cells) == bh * bw:
            continue
        grid = np.zeros((h, w), dtype=int)
        for r, c in cells:
            grid[r0 + r, c0 + c] = 5

        spots = []
        for r in range(h - 1):
            for c in range(w - 1):
                if r + 1 < r0 - 1 or r > r0 + bh or c + 1 < c0 - 1 or c > c0 + bw:
                    spots.append((r, c))
        if not spots:
            continue
        mr, mc = random.choice(spots)
        colors = random.sample([1, 2, 3, 4, 6, 7, 8, 9], 4)
        grid[mr:mr + 2, mc:mc + 2] = np.array(colors).reshape(2, 2)
        return grid
    raise RuntimeError("failed to generate input")
