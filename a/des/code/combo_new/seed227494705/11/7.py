import random
import numpy as np


def _components(mask):
    h, w = mask.shape
    seen = np.zeros((h, w), bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = [(r, c)]
            seen[r, c] = True
            pts = []
            while q:
                x, y = q.pop()
                pts.append((x, y))
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < h and 0 <= ny < w and mask[nx, ny] and not seen[nx, ny]:
                        seen[nx, ny] = True
                        q.append((nx, ny))
            comps.append(pts)
    return comps


def main(grid: np.ndarray) -> np.ndarray:
    """Slide each stamp along a grey guide and redraw it with the color marker at the far end.

    Input structure: black background with horizontal or vertical grey guide
    segments; one endpoint touches a multi-pixel colored stamp and the opposite
    endpoint touches a single-pixel color marker.
    Rule: erase stamps and markers, keep guides, translate each stamp from its
    guide endpoint to the opposite endpoint while preserving shape, and recolor
    the translated stamp with the destination marker color.
    Key insight: the guide line controls position transfer, while the endpoint
    marker supplies the stamp's output color.
    """
    grey = 5
    out = grid.copy()
    comps = _components((grid != 0) & (grid != grey))
    comp_id = {}
    for i, comp in enumerate(comps):
        for p in comp:
            comp_id[p] = i
    for comp in comps:
        for r, c in comp:
            out[r, c] = 0
    for rail in _components(grid == grey):
        rs = [p[0] for p in rail]
        cs = [p[1] for p in rail]
        if len(set(rs)) == 1:
            r, c1, c2 = rs[0], min(cs), max(cs)
            endpoints = [((r, c1), (0, -1)), ((r, c2), (0, 1))]
        elif len(set(cs)) == 1:
            c, r1, r2 = cs[0], min(rs), max(rs)
            endpoints = [((r1, c), (-1, 0)), ((r2, c), (1, 0))]
        else:
            continue
        found = []
        for ep, d in endpoints:
            p = (ep[0] + d[0], ep[1] + d[1])
            if p in comp_id:
                found.append((ep, d, comp_id[p]))
        if len(found) != 2:
            continue
        a, b = found
        ca, cb = comps[a[2]], comps[b[2]]
        if len(ca) == 1 and len(cb) > 1:
            marker, stamp = a, b
        elif len(cb) == 1 and len(ca) > 1:
            marker, stamp = b, a
        else:
            continue
        color = int(grid[comps[marker[2]][0]])
        source_anchor = (stamp[0][0] + stamp[1][0], stamp[0][1] + stamp[1][1])
        target_anchor = (marker[0][0] + marker[1][0], marker[0][1] + marker[1][1])
        dr, dc = target_anchor[0] - source_anchor[0], target_anchor[1] - source_anchor[1]
        for r, c in comps[stamp[2]]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < out.shape[0] and 0 <= nc < out.shape[1]:
                out[nr, nc] = color
    return out


def generate_input() -> np.ndarray:
    grid = np.zeros((24, 24), dtype=int)
    rails = random.randint(2, 3)
    for i in range(rails):
        horizontal = random.random() < 0.5
        if horizontal:
            r = 4 + i * 7
            c1, c2 = 5, random.randint(14, 19)
            grid[r, c1:c2 + 1] = 5
            stamp_side = random.choice(["left", "right"])
            marker_color = random.choice([2, 3, 4, 6, 7, 8, 9])
            stamp_color = random.choice([x for x in [1, 2, 3, 4, 6, 7, 8] if x != marker_color])
            shape = [(0, 0), (1, 0), (1, 1), (2, 0)]
            if stamp_side == "left":
                sr, sc, mr, mc = r - 1, c1 - 3, r, c2 + 1
            else:
                sr, sc, mr, mc = r - 1, c2 + 1, r, c1 - 1
            for a, b in shape:
                grid[sr + a, sc + b] = stamp_color
            grid[mr, mc] = marker_color
        else:
            c = 4 + i * 7
            r1, r2 = 5, random.randint(14, 19)
            grid[r1:r2 + 1, c] = 5
            marker_color = random.choice([2, 3, 4, 6, 7, 8, 9])
            stamp_color = random.choice([x for x in [1, 2, 3, 4, 6, 7, 8] if x != marker_color])
            shape = [(0, 0), (0, 1), (1, 0), (2, 0)]
            if random.random() < 0.5:
                sr, sc, mr, mc = r1 - 3, c - 1, r2 + 1, c
            else:
                sr, sc, mr, mc = r2 + 1, c - 1, r1 - 1, c
            for a, b in shape:
                grid[sr + a, sc + b] = stamp_color
            grid[mr, mc] = marker_color
    return grid
