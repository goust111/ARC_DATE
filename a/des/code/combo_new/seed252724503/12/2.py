import numpy as np


BLACK = 0
BLUE = 1


def _flood(grid, start):
    rows, cols = grid.shape
    color = int(grid[start])
    seen = {start}
    stack = [start]
    cells = []
    while stack:
        r, c = stack.pop()
        cells.append((r, c))
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) not in seen and grid[nr, nc] != BLUE:
                seen.add((nr, nc))
                stack.append((nr, nc))
    return cells, color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Flood-fill each blue enclosure or chamber from its seed pixel using the seed's color.

    Input structure:
    - Black background with several blue rectangular outlines, optionally containing extra blue interior walls.
    - Each enclosure contains exactly one non-blue seed pixel of some color.
    - Blue pixels are the only barriers; all other cells inside a chamber are black.

    Rule:
    - Keep every blue wall pixel unchanged.
    - For each seed, flood-fill from that seed through all non-blue cells reachable by four-neighbor moves.
    - Recolor the whole reachable chamber, including the seed position itself, to the seed's color.

    Key insight:
    - The seed does not just choose the fill color; its position chooses which chamber gets filled when blue walls split an enclosure.
    """
    out = grid.copy()
    seeds = np.argwhere((grid != BLACK) & (grid != BLUE))
    for seed in seeds:
        cells, color = _flood(grid, tuple(seed))
        for r, c in cells:
            out[r, c] = color
    return out


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    grid = np.zeros((24, 24), dtype=int)
    used = np.zeros_like(grid, dtype=bool)
    room_count = int(rng.integers(2, 5))
    colors = list(rng.choice([2, 3, 4, 6, 7, 8, 9], size=room_count, replace=False))
    for color in colors:
        for _attempt in range(200):
            h = int(rng.integers(6, 10))
            w = int(rng.integers(6, 10))
            r0 = int(rng.integers(1, grid.shape[0] - h - 1))
            c0 = int(rng.integers(1, grid.shape[1] - w - 1))
            if used[r0 - 1 : r0 + h + 1, c0 - 1 : c0 + w + 1].any():
                continue
            grid[r0 : r0 + h, c0 : c0 + w] = BLUE
            grid[r0 + 1 : r0 + h - 1, c0 + 1 : c0 + w - 1] = BLACK
            chamber_cells = [(r, c) for r in range(r0 + 1, r0 + h - 1) for c in range(c0 + 1, c0 + w - 1)]
            if rng.random() < 0.6 and h >= 7 and w >= 7:
                if rng.random() < 0.5:
                    cc = c0 + w // 2
                    gap = r0 + 1 + int(rng.integers(0, h - 2))
                    for r in range(r0 + 1, r0 + h - 1):
                        if r != gap:
                            grid[r, cc] = BLUE
                    chamber_cells = [(r, c) for r, c in chamber_cells if c < cc or c > cc]
                else:
                    rr = r0 + h // 2
                    gap = c0 + 1 + int(rng.integers(0, w - 2))
                    for c in range(c0 + 1, c0 + w - 1):
                        if c != gap:
                            grid[rr, c] = BLUE
                    chamber_cells = [(r, c) for r, c in chamber_cells if r < rr or r > rr]
            rr, cc = chamber_cells[int(rng.integers(0, len(chamber_cells)))]
            grid[rr, cc] = int(color)
            used[r0 - 1 : r0 + h + 1, c0 - 1 : c0 + w + 1] = True
            break
    return grid
