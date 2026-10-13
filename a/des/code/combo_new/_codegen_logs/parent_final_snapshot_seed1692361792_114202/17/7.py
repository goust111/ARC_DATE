import random
from collections import deque
import numpy as np

BLACK = 0
GREY = 5


def components(mask: np.ndarray):
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if mask[r, c] and not seen[r, c]:
                q = deque([(r, c)])
                seen[r, c] = True
                cells = []
                while q:
                    x, y = q.popleft()
                    cells.append((x, y))
                    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < h and 0 <= ny < w and mask[nx, ny] and not seen[nx, ny]:
                            seen[nx, ny] = True
                            q.append((nx, ny))
                comps.append(cells)
    return comps


def main(grid: np.ndarray) -> np.ndarray:
    grid = np.array(grid, dtype=int)
    s = 3
    corners = {
        "tl": (slice(0, s), slice(0, s)),
        "tr": (slice(0, s), slice(-s, None)),
        "bl": (slice(-s, None), slice(0, s)),
        "br": (slice(-s, None), slice(-s, None)),
    }
    templates = []
    occupied = np.zeros_like(grid, dtype=bool)
    for name, (rs, cs) in corners.items():
        patch = grid[rs, cs].copy()
        occupied[rs, cs] = True
        theme_values = [x for x in np.unique(patch) if x not in (BLACK, GREY)]
        assert theme_values, "each corner template must contain a non-grey theme color"
        theme = theme_values[0]
        score = int(np.count_nonzero(patch == theme))
        templates.append((score, patch))
    templates.sort(key=lambda x: x[0])
    order = [tpl for _, tpl in templates]
    out = grid.copy()
    objs = []
    for cells in components((grid != BLACK) & (~occupied)):
        vals = {grid[r, c] for r, c in cells}
        if vals == {BLACK}:
            continue
        rs = [r for r, _ in cells]
        cs = [c for _, c in cells]
        r0, r1, c0, c1 = min(rs), max(rs), min(cs), max(cs)
        objs.append((len(cells), r0, r1, c0, c1))
    areas = sorted({a for a, *_ in objs})
    rank = {a: i for i, a in enumerate(areas)}
    for area, r0, r1, c0, c1 in objs:
        tpl = order[rank[area]]
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                out[r, c] = tpl[(r - r0) % s, (c - c0) % s]
    return out


def _generate_once() -> np.ndarray:
    h = w = random.randint(16, 22)
    grid = np.zeros((h, w), dtype=int)
    themes = random.sample([1, 2, 3, 4, 6, 7, 8, 9], 4)
    patterns = [
        np.array([[themes[0], GREY, GREY], [GREY, GREY, GREY], [GREY, GREY, GREY]]),
        np.array([[themes[1], GREY, themes[1]], [GREY, GREY, GREY], [themes[1], GREY, GREY]]),
        np.array([[themes[2], GREY, themes[2]], [GREY, themes[2], GREY], [themes[2], GREY, GREY]]),
        np.array([[themes[3], GREY, themes[3]], [GREY, themes[3], GREY], [themes[3], GREY, themes[3]]]),
    ]
    grid[:3, :3] = patterns[0]
    grid[:3, -3:] = patterns[1]
    grid[-3:, :3] = patterns[2]
    grid[-3:, -3:] = patterns[3]
    sizes = [(2, 2), (2, 3), (3, 3), (3, 4)]
    colors = themes[:]
    random.shuffle(colors)
    for idx, (rh, rw) in enumerate(sizes):
        for _ in range(random.randint(1, 2)):
            r0 = random.randint(4, h - rh - 5)
            c0 = random.randint(4, w - rw - 5)
            if np.any(grid[r0 : r0 + rh, c0 : c0 + rw] != BLACK):
                continue
            grid[r0 : r0 + rh, c0 : c0 + rw] = colors[idx]
    return grid


def generate_input() -> np.ndarray:
    for _ in range(500):
        grid = _generate_once()
        try:
            out = main(grid)
        except Exception:
            continue
        if out.ndim == 2 and max(out.shape) <= 30:
            return grid
    raise RuntimeError("failed to generate valid input")
