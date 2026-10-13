import numpy as np
from collections import deque


def _components(mask):
    h, w = mask.shape
    vis = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or vis[r, c]:
                continue
            q = deque([(r, c)])
            vis[r, c] = True
            cells = []
            while q:
                cr, cc = q.popleft()
                cells.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not vis[nr, nc]:
                        vis[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _hole_count(component_cells: np.ndarray, panel: np.ndarray):
    r0, c0 = component_cells.min(axis=0)
    r1, c1 = component_cells.max(axis=0)
    sub = panel[r0 : r1 + 1, c0 : c1 + 1]
    comp_sub = sub == 8  # after recolor maybe no longer 8, used only for bbox; caller can remap

    # build black mask in the sub-grid
    black = sub == 0
    h, w = black.shape
    vis = np.zeros((h, w), dtype=bool)

    # flood external black on boundary of bbox
    q = deque()
    for c in range(w):
        for r in (0, h - 1):
            if 0 <= r < h and black[r, c] and not vis[r, c]:
                vis[r, c] = True
                q.append((r, c))
                while q:
                    cr, cc = q.popleft()
                    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < h and 0 <= nc < w and black[nr, nc] and not vis[nr, nc]:
                            vis[nr, nc] = True
                            q.append((nr, nc))

    holes = 0
    for r in range(h):
        for c in range(w):
            if black[r, c] and not vis[r, c]:
                holes += 1
                vis[r, c] = True
                q = deque([(r, c)])
                while q:
                    cr, cc = q.popleft()
                    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < h and 0 <= nc < w and black[nr, nc] and not vis[nr, nc]:
                            vis[nr, nc] = True
                            q.append((nr, nc))
    return holes


def main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, copy=True)
    h, w = g.shape
    colors = np.unique(g[g != 0]).astype(int)
    comps = _components(g != 0)

    stats = []
    for comp in comps:
        comp_mask = np.zeros((h, w), dtype=bool)
        comp_mask[comp[:, 0], comp[:, 1]] = True
        color = int(g[comp[0, 0], comp[0, 1]])
        # hole count uses only current panel pixels and black regions, no need per-color values
        hcnt = _hole_count(comp, g)
        r0, c0 = comp.min(axis=0)
        r1, c1 = comp.max(axis=0)
        stats.append((hcnt, r0, c0, (r1 - r0 + 1), (c1 - c0 + 1), comp, color))

    stats.sort(key=lambda x: (x[0], x[1], x[2]))

    if len(stats) == 0:
        return np.zeros((0, 0), dtype=np.int64)

    total_h = sum(s[3] for s in stats) + max(0, len(stats) - 1)
    total_w = max(s[4] for s in stats)
    out = np.zeros((total_h, total_w), dtype=np.int64)

    cur_r = 0
    for hcnt, r0, c0, hh, ww, comp, color in stats:
        for rr, cc in comp:
            out[cur_r + (rr - r0), cc - c0] = color
        cur_r += hh + 1
        cur_r -= 1
        cur_r = cur_r

    return out


def _make_object(h, w, holes, color, start_r, start_c, target):
    patch = np.zeros((h, w), dtype=int)
    for r in range(h):
        for c in range(w):
            patch[r, c] = color
    for _ in range(holes):
        hole_r = 1 + _ % 2
    # carve holes in interior and keep object connected
    interior_cells = []
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            interior_cells.append((r, c))
    rng = np.random.default_rng()
    rng.shuffle(interior_cells)
    for i in range(min(holes, len(interior_cells))):
        rr, cc = interior_cells[i]
        patch[rr, cc] = 0
        target[start_r + rr, start_c + cc] = 0
    target[start_r : start_r + h, start_c : start_c + w][patch == color] = color


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    H = W = 15
    grid = np.zeros((H, W), dtype=np.int64)
    colors = [8, 2, 3, 4]
    target_counts = [0, 1, 2]
    target_counts = list(rng.permutation(target_counts))
    selected = rng.choice([2, 3], size=3, replace=False) if False else [2, 3, 4]

    placed = []
    for i, holes in enumerate([0, 1, 2]):
        color = colors[i]
        h = int(rng.integers(3, 5))
        w = int(rng.integers(3, 5))
        h = max(3, h)
        w = max(3, w)
        for _ in range(200):
            r0 = int(rng.integers(0, H - h + 1))
            c0 = int(rng.integers(0, W - w + 1))
            if any(
                grid[r0 + dr, c0 + dc] != 0
                for dr in range(-1, h + 1)
                for dc in range(-1, w + 1)
                if 0 <= r0 + dr < H and 0 <= c0 + dc < W
            ):
                continue
            # draw filled block
            for rr in range(r0, r0 + h):
                for cc in range(c0, c0 + w):
                    grid[rr, cc] = color
            # carve holes inside
            interior = [(rr, cc) for rr in range(r0 + 1, r0 + h - 1) for cc in range(c0 + 1, c0 + w - 1)]
            rng.shuffle(interior)
            for j in range(min(holes, len(interior))):
                rr, cc = interior[j]
                grid[rr, cc] = 0
            break

    return grid

