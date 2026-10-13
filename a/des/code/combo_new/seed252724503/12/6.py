import numpy as np


BLACK = 0


def _components_by_color(grid):
    comps = []
    for color in np.unique(grid):
        if color == BLACK:
            continue
        mask = grid == color
        rows, cols = mask.shape
        seen = np.zeros_like(mask, dtype=bool)
        for r in range(rows):
            for c in range(cols):
                if not mask[r, c] or seen[r, c]:
                    continue
                stack = [(r, c)]
                seen[r, c] = True
                comp = []
                while stack:
                    rr, cc = stack.pop()
                    comp.append((rr, cc))
                    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        nr, nc = rr + dr, cc + dc
                        if 0 <= nr < rows and 0 <= nc < cols and mask[nr, nc] and not seen[nr, nc]:
                            seen[nr, nc] = True
                            stack.append((nr, nc))
                comps.append((int(color), comp))
    return comps


def main(grid: np.ndarray) -> np.ndarray:
    """
    Swap region colors by pairing the smallest area with the largest, the second smallest with the second largest, and so on.

    Input structure:
    - Black background with an even number of disconnected colored regions.
    - Every region uses a distinct color.
    - Region areas are all different.

    Rule:
    - Sort the regions by area from smallest to largest.
    - Pair the first with the last, the second with the second last, and continue inward.
    - Keep each region's shape and position unchanged, but recolor every pair by swapping their colors.

    Key insight:
    - The recoloring relation is not keyed by the visible hue itself; the hidden ordering comes from region area.
    """
    out = np.zeros_like(grid)
    comps = sorted(_components_by_color(grid), key=lambda item: len(item[1]))
    mapped = [comps[-1 - i][0] for i in range(len(comps))]
    for idx, (_, comp) in enumerate(comps):
        color = mapped[idx]
        for r, c in comp:
            out[r, c] = color
    return out


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    grid = np.zeros((24, 24), dtype=int)
    n = int(rng.choice([4, 6]))
    colors = list(rng.choice([1, 2, 3, 4, 5, 6, 7, 8, 9], size=n, replace=False))
    used = np.zeros_like(grid, dtype=bool)
    dims = [(2, 2), (2, 3), (2, 4), (3, 3), (3, 4), (4, 4)]
    rng.shuffle(dims)
    for color, (h, w) in zip(colors, dims[:n]):
        for _attempt in range(200):
            r0 = int(rng.integers(1, 24 - h - 1))
            c0 = int(rng.integers(1, 24 - w - 1))
            if used[r0 - 1 : r0 + h + 1, c0 - 1 : c0 + w + 1].any():
                continue
            grid[r0 : r0 + h, c0 : c0 + w] = int(color)
            used[r0 - 1 : r0 + h + 1, c0 - 1 : c0 + w + 1] = True
            break
    return grid
