import random
import numpy as np


def _core_from_grid(grid):
    body = grid[1:, :]
    if body.shape[1] == 3:
        return body.copy()
    best_start = 0
    best_score = -1
    for start in range(body.shape[1] - 2):
        sub = body[:, start:start + 3]
        score = int(np.count_nonzero(sub))
        if score > best_score:
            best_score = score
            best_start = start
    return body[:, best_start:best_start + 3].copy()


def main(grid: np.ndarray) -> np.ndarray:
    grid = np.asarray(grid)
    core = _core_from_grid(grid)
    out = np.zeros((9, 9), dtype=grid.dtype)
    for r, c in np.argwhere(core == 2):
        r = int(r)
        c = int(c)
        br, bc = r * 3, c * 3
        center_r, center_c = br + 1, bc + 1
        distances = [
            (center_r, 0),
            (8 - center_c, 1),
            (8 - center_r, 2),
            (center_c, 3),
        ]
        _, turns = min(distances, key=lambda x: (x[0], x[1]))
        stamp = np.rot90(core, -turns)
        out[br:br + 3, bc:bc + 3] = stamp
    return out


def _non_symmetric(core):
    return not any(np.array_equal(core, np.rot90(core, k)) for k in (1, 2, 3))


def generate_input() -> np.ndarray:
    rng = random.Random()
    while True:
        grid = np.zeros((4, 4), dtype=int)
        guide = rng.choice([1, 8, 9])
        grid[0, :] = guide
        colors = [3, 4, 6, 7]
        corners = [(0, 0), (0, 2), (2, 0), (2, 2)]
        controls = [rng.choice(corners)]
        non_corners = [(r, c) for r in range(3) for c in range(3) if (r, c) not in corners]
        controls.append(rng.choice(non_corners))
        if rng.random() < 0.45:
            extra = rng.choice([(r, c) for r in range(3) for c in range(3) if (r, c) not in controls])
            controls.append(extra)
        core = np.zeros((3, 3), dtype=int)
        for r, c in controls:
            core[r, c] = 2
        for r in range(3):
            for c in range(3):
                if core[r, c] == 0 and rng.random() < 0.55:
                    core[r, c] = rng.choice(colors)
        if np.count_nonzero(core) >= 4 and _non_symmetric(core):
            grid[1:, :3] = core
            return grid
