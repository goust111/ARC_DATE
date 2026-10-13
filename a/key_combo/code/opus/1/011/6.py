import random
from collections import deque

import numpy as np


def _components(mask):
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp = []
            while q:
                x, y = q.popleft()
                comp.append((x, y))
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < h and 0 <= ny < w and mask[nx, ny] and not seen[nx, ny]:
                        seen[nx, ny] = True
                        q.append((nx, ny))
            comps.append(comp)
    return comps


def _panel_objects(panel):
    objs = []
    for color in sorted(set(map(int, panel.ravel())) - {0, 5}):
        for comp in _components(panel == color):
            cr = sum(r for r, _ in comp) / len(comp)
            cc = sum(c for _, c in comp) / len(comp)
            objs.append({"coords": comp, "color": color, "centroid": (cr, cc)})
    return sorted(objs, key=lambda o: o["centroid"][0])


def main(grid: np.ndarray) -> np.ndarray:
    grid = np.array(grid, copy=True)
    sep_cols = [c for c in range(grid.shape[1]) if np.all(grid[:, c] == 5)]
    bounds = [-1] + sep_cols + [grid.shape[1]]
    panels = [grid[:, bounds[i] + 1:bounds[i + 1]] for i in range(3)]
    selected = []
    for idx, panel in enumerate(panels):
        selected.append(_panel_objects(panel)[idx])
    targets = [selected[1]["centroid"][0], selected[2]["centroid"][0], selected[0]["centroid"][0]]
    out = np.zeros(panels[0].shape, dtype=int)
    for obj, tr in zip(selected, targets):
        sr = obj["centroid"][0]
        shift = int(round(tr - sr))
        for r, c in obj["coords"]:
            nr = r + shift
            if 0 <= nr < out.shape[0] and 0 <= c < out.shape[1]:
                out[nr, c] = obj["color"]
    return out


def _try_put(panel, cells, color):
    if all(0 <= r < panel.shape[0] and 0 <= c < panel.shape[1] and panel[r, c] == 0 for r, c in cells):
        for r, c in cells:
            panel[r, c] = color
        return True
    return False


def generate_input() -> np.ndarray:
    grid = np.zeros((10, 29), dtype=int)
    grid[:, 9] = 5
    grid[:, 19] = 5
    palette = [1, 2, 3, 4, 6, 7, 8, 9]
    colors = []
    for _ in range(3):
        colors.extend(random.sample(palette, 3))
    shapes = [
        [(0, 0), (0, 1), (1, 0)],
        [(0, 0), (1, 0), (1, 1), (2, 1)],
        [(0, 1), (1, 0), (1, 1), (1, 2)],
    ]
    for pidx, start in enumerate([0, 10, 20]):
        panel = grid[:, start:start + 9]
        rows = random.sample([1, 4, 7], 3)
        for j, base_r in enumerate(rows):
            shape = random.choice(shapes)
            for _ in range(100):
                base_c = random.randint(1, 5)
                cells = [(base_r + r, base_c + c) for r, c in shape]
                if _try_put(panel, cells, colors[pidx * 3 + j]):
                    break
    return grid
