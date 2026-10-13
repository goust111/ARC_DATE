import numpy as np
from collections import deque

BG = 0


def _components(grid: np.ndarray, mask: np.ndarray):
    h, w = grid.shape
    vis = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if mask[r, c] and not vis[r, c]:
                q = deque([(r, c)])
                vis[r, c] = True
                pts = []
                while q:
                    rr, cc = q.popleft()
                    pts.append((rr, cc))
                    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                        nr, nc = rr + dr, cc + dc
                        if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not vis[nr, nc]:
                            vis[nr, nc] = True
                            q.append((nr, nc))
                color = int(grid[pts[0][0], pts[0][1]])
                comps.append((color, np.asarray(pts, dtype=np.int32)))
    return comps


def _bbox(pts: np.ndarray):
    r0, c0 = pts[:, 0].min(), pts[:, 1].min()
    r1, c1 = pts[:, 0].max(), pts[:, 1].max()
    return int(r0), int(c0), int(r1), int(c1)


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid)
    h, w = g.shape
    out = np.zeros_like(g)

    comps = _components(g, g != BG)
    boundary_rects = []

    # Identify solid boundary rectangles.
    for color, pts in comps:
        r0, c0, r1, c1 = _bbox(pts)
        if pts.shape[0] != (r1 - r0 + 1) * (c1 - c0 + 1):
            continue
        touch_border = (r0 == 0) or (c0 == 0) or (r1 == h - 1) or (c1 == w - 1)
        if not touch_border:
            continue
        # One side touches border.
        touches = 0
        side = None
        if r0 == 0 and c0 > 0 and c1 < w - 1:
            touches += 1
            side = "top"
        if r1 == h - 1 and c0 > 0 and c1 < w - 1:
            touches += 1
            side = "bottom"
        if c0 == 0 and r0 > 0 and r1 < h - 1:
            touches += 1
            side = "left"
        if c1 == w - 1 and r0 > 0 and r1 < h - 1:
            touches += 1
            side = "right"
        if touches == 1 and 2 <= (r1 - r0 + 1) <= 3:
            boundary_rects.append((color, side, r0, c0, r1, c1))

    # Count external sprites for each rectangle color.
    forbid = np.zeros_like(g, dtype=bool)
    for _color, _side, r0, c0, r1, c1 in boundary_rects:
        forbid[r0 : r1 + 1, c0 : c1 + 1] = True

    moved = []
    for color, side, r0, c0, r1, c1 in boundary_rects:
        comp_count = 0
        used = np.zeros_like(g, dtype=bool)
        for r in range(1, h - 1):
            for c in range(1, w - 1):
                if used[r, c] or g[r, c] != color or forbid[r, c]:
                    continue
                # connected component of same color in interior
                q = deque([(r, c)])
                used[r, c] = True
                touches_border = False
                while q:
                    rr, cc = q.popleft()
                    if rr == 0 or cc == 0 or rr == h - 1 or cc == w - 1:
                        touches_border = True
                    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                        nr, nc = rr + dr, cc + dc
                        if 0 <= nr < h and 0 <= nc < w and not used[nr, nc] and g[nr, nc] == color and not forbid[nr, nc]:
                            used[nr, nc] = True
                            q.append((nr, nc))
                if not touches_border:
                    comp_count += 1

        dr = dc = 0
        if side == "top":
            dr = comp_count
        elif side == "bottom":
            dr = -comp_count
        elif side == "left":
            dc = comp_count
        elif side == "right":
            dc = -comp_count

        moved.append((comp_count, side, color, r0, c0, r1, c1, dr, dc))

    # Draw lower side priority first, higher on top.
    def side_priority(side: str) -> int:
        return 1 if side in ("top", "left") else 0

    moved.sort(key=lambda x: (x[0], side_priority(x[1])))

    for _, _, color, r0, c0, r1, c1, dr, dc in moved:
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                nr, nc = r + dr, c + dc
                out[nr, nc] = color

    return out


def _choose_rect_area(rng: np.random.Generator, h: int, w: int, side: str):
    if side in ("top", "bottom"):
        height = int(rng.integers(2, 4))
        width = int(rng.integers(3, 6))
        r0 = 0 if side == "top" else h - height
        c0 = int(rng.integers(0, w - width))
        return r0, c0, r0 + height - 1, c0 + width - 1
    height = int(rng.integers(3, 6))
    width = int(rng.integers(2, 4))
    c0 = 0 if side == "left" else w - width
    r0 = int(rng.integers(0, h - height))
    return r0, c0, r0 + height - 1, c0 + width - 1


def _random_component(rng: np.random.Generator, h: int, w: int, size: int, occupied: np.ndarray):
    # Small explicit templates keep generation fast and non-rectangular.
    templates = {
        2: [
            [(0, 0), (1, 0)],
            [(0, 0), (0, 1)],
        ],
        3: [
            [(0, 0), (1, 0), (1, 1)],
            [(0, 1), (1, 0), (1, 1)],
            [(0, 0), (0, 1), (1, 0)],
        ],
        4: [
            [(0, 0), (1, 0), (1, 1), (2, 1)],
            [(0, 0), (0, 1), (1, 0), (1, 2)],
            [(0, 1), (1, 0), (1, 1), (2, 1)],
        ],
    }
    tmpl = templates.get(size, [ [(0,0)] ])
    for _ in range(250):
        mask = tmpl[int(rng.integers(0, len(tmpl)))]
        max_r = max(p[0] for p in mask)
        max_c = max(p[1] for p in mask)
        r0 = int(rng.integers(1, h - max_r - 1))
        c0 = int(rng.integers(1, w - max_c - 1))
        pts = [(r0 + dr, c0 + dc) for dr, dc in mask]
        if any(occupied[r, c] for r, c in pts):
            continue
        return pts
    # Deterministic fallback for impossible cases.
    for r0 in range(1, h - 1):
        for c0 in range(1, w - 1):
            if not occupied[r0, c0]:
                return [(r0, c0), (min(r0 + 1, h - 2), c0)]
    return None


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    h = 14
    w = 14
    grid = np.zeros((h, w), dtype=np.int64)

    colors = [1, 2, 3, 7]
    rng.shuffle(colors)
    n_rects = int(rng.integers(3, 5))
    sides = rng.choice(["top", "bottom", "left", "right"], size=n_rects, replace=False)
    occupied = np.zeros_like(grid, dtype=bool)

    rectangles = []
    for i in range(n_rects):
        side = str(sides[i])
        for _ in range(500):
            r0, c0, r1, c1 = _choose_rect_area(rng, h, w, side)
            if r0 < 0 or c0 < 0:
                continue
            if occupied[r0 : r1 + 1, c0 : c1 + 1].any():
                continue
            color = colors[i]
            grid[r0:r1 + 1, c0:c1 + 1] = color
            occupied[r0 : r1 + 1, c0 : c1 + 1] = True
            rectangles.append((side, color, r0, c0, r1, c1))
            break

    # place internal sprites
    for color, side in ((colors[i], rect[0]) for i, rect in enumerate(rectangles)):
        count = int(rng.integers(1, 4))
        for _ in range(count):
            size = int(rng.integers(2, 5))
            pts = _random_component(rng, h, w, size, occupied)
            if pts is None:
                continue
            # Make sure sprite is not a solid rectangle of its bbox.
            rs = np.array([p[0] for p in pts])
            cs = np.array([p[1] for p in pts])
            area = (rs.max() - rs.min() + 1) * (cs.max() - cs.min() + 1)
            if len(pts) == area:
                continue
            for rr, cc in pts:
                grid[rr, cc] = color
    return grid
