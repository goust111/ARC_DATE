import random
import numpy as np


def main(grid: np.ndarray) -> np.ndarray:
    h, w = grid.shape
    guide_row = max(range(h), key=lambda r: int(np.sum(grid[r] == 5)))
    guide_cols = np.where(grid[guide_row] == 5)[0]
    lo, hi = int(guide_cols.min()), int(guide_cols.max())
    target_row = guide_row - 1
    out = np.zeros_like(grid)
    out[guide_row, lo : hi + 1] = grid[guide_row, lo : hi + 1]

    by_color: dict[int, list[tuple[int, int]]] = {}
    for r in range(guide_row):
        for c in range(w):
            color = int(grid[r, c])
            if color not in (0, 5):
                if grid[guide_row, c] == 5:
                    out[target_row, c] = color
                    by_color.setdefault(color, []).append((target_row, c))

    for color, pts in by_color.items():
        if len(pts) == 2:
            c1, c2 = sorted((pts[0][1], pts[1][1]))
            for c in range(c1, c2 + 1):
                if out[target_row, c] == 0:
                    out[target_row, c] = color
            out[target_row, c1] = color
            out[target_row, c2] = color
    return out


def generate_input() -> np.ndarray:
    colors = [1, 2, 3, 4, 7]
    for _ in range(300):
        h = random.randint(13, 16)
        w = random.randint(13, 17)
        row = random.randint(6, h - 5)
        grid = np.zeros((h, w), dtype=int)
        lo, hi = 1, w - 2
        grid[row, lo : hi + 1] = 5
        gaps = sorted(random.sample(range(lo + 1, hi), 3))
        for c in gaps:
            grid[row, c] = 0
        solid = [c for c in range(lo, hi + 1) if c not in gaps]

        pair_count = random.randint(4, 5)
        chosen = random.sample(colors, pair_count)
        specs = [
            (random.sample(solid, 2), chosen[0]),
            ([random.choice(solid), gaps[0]], chosen[1]),
            (gaps[1:3], chosen[2]),
        ]
        free_cols = [c for c in range(lo, hi + 1) if c not in {x for cs, _ in specs for x in cs}]
        for color in chosen[3:]:
            if len(free_cols) < 2:
                break
            cs = random.sample(free_cols, 2)
            free_cols = [c for c in free_cols if c not in cs]
            specs.append((cs, color))

        used = set()
        ok = True
        for cols, color in specs:
            for c in cols:
                r = random.randint(0, row - 2)
                if (r, c) in used:
                    ok = False
                    break
                used.add((r, c))
                grid[r, c] = color
        if ok:
            return grid
    return grid
