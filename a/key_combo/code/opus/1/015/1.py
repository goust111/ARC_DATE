import random
import numpy as np


def main(grid: np.ndarray) -> np.ndarray:
    out = grid.copy()
    h, w = grid.shape
    stripes = []
    for r in range(h):
        vals = set(grid[r].tolist())
        if len(vals) == 1 and grid[r, 0] != 0:
            stripes.append(int(grid[r, 0]))
        else:
            break
    start = len(stripes)
    marker_rows = range(start, h)
    marker_positions = {
        c: [(r, col) for r in marker_rows for col in range(w) if grid[r, col] == c]
        for c in stripes
    }
    for color in stripes:
        for r, c in sorted(marker_positions[color]):
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and out[nr, nc] == 0:
                        out[nr, nc] = color
    return out


def generate_input() -> np.ndarray:
    colors = [1, 2, 3, 4, 6, 7, 8, 9]
    for _ in range(300):
        h = random.randint(14, 18)
        w = random.randint(13, 18)
        k = random.randint(3, 5)
        stripe_colors = random.sample(colors, k)
        grid = np.zeros((h, w), dtype=int)
        for r, color in enumerate(stripe_colors):
            grid[r, :] = color
        rows = list(range(k + 1, h - 1))
        cols = list(range(1, w - 1))
        positions = []
        for color in stripe_colors:
            positions.append((random.choice(rows), random.choice(cols), color))
        for _i in range(random.randint(1, 3)):
            base = random.choice(positions)
            color = random.choice(stripe_colors)
            br, bc, _ = base
            candidates = [
                (br + dr, bc + dc)
                for dr in range(-2, 3)
                for dc in range(-2, 3)
                if max(abs(dr), abs(dc)) <= 2 and (dr or dc)
            ]
            random.shuffle(candidates)
            for r, c in candidates:
                if r in rows and c in cols and all((r, c) != (pr, pc) for pr, pc, _ in positions):
                    positions.append((r, c, color))
                    break
        while len(positions) < random.randint(6, 8):
            r, c = random.choice(rows), random.choice(cols)
            if all((r, c) != (pr, pc) for pr, pc, _ in positions):
                positions.append((r, c, random.choice(stripe_colors)))
        overlap_pairs = 0
        for i, (r1, c1, _) in enumerate(positions):
            for r2, c2, _ in positions[i + 1:]:
                if max(abs(r1 - r2), abs(c1 - c2)) <= 2:
                    overlap_pairs += 1
        if overlap_pairs >= 2:
            for r, c, color in positions:
                grid[r, c] = color
            return grid
    return generate_input()
