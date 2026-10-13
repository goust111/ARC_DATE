import numpy as np
from collections import deque


_BG = 0
_GRAY = 5

_generate_calls = 0


def _components(grid, color):
    h, w = grid.shape
    vis = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if grid[r, c] != color or vis[r, c]:
                continue
            q = deque([(r, c)])
            vis[r, c] = True
            comp = []
            while q:
                x, y = q.popleft()
                comp.append((x, y))
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < h and 0 <= ny < w and not vis[nx, ny] and grid[nx, ny] == color:
                        vis[nx, ny] = True
                        q.append((nx, ny))
            comps.append(comp)
    return comps


def _bbox(cells):
    rs = [r for r, c in cells]
    cs = [c for r, c in cells]
    return min(rs), min(cs), max(rs), max(cs)


def _perimeter_path(bbox):
    r0, c0, r1, c1 = bbox
    pts = []
    if r0 > r1 or c0 > c1:
        return pts
    for c in range(c0, c1 + 1):
        pts.append((r0, c))
    for r in range(r0 + 1, r1 + 1):
        pts.append((r, c1))
    for c in range(c1 - 1, c0 - 1, -1):
        pts.append((r1, c))
    for r in range(r1 - 1, r0, -1):
        pts.append((r, c0))
    return pts


def _obstacle_perimeter(pts):
    s = set(pts)
    p = 0
    for r, c in pts:
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            if (r + dr, c + dc) not in s:
                p += 1
    return p


def generate_input():
    global _generate_calls
    rng = np.random.RandomState(8000 + _generate_calls)
    _generate_calls += 1

    h, w = 22, 20
    grid = np.zeros((h, w), dtype=int)

    colors = [2, 1, 3, 4]
    rng.shuffle(colors)
    groups = 3
    cols = [2, 8, 14]

    for i, color in enumerate(colors[:groups]):
        x = cols[i]
        bar_len = 4 + rng.randint(0, 4)
        bar = [(r, x) for r in range(bar_len)]
        for p in bar:
            if 0 <= p[0] < h:
                grid[p] = color

        ob_h = 3
        ob_w = 3
        top = bar_len + 2
        left = max(0, x - 1)
        for rr in range(top, min(h - 1, top + ob_h - 1) + 1):
            for cc in range(left, min(w - 1, left + ob_w - 1) + 1):
                grid[rr, cc] = _GRAY

        # initial segment pixels
        seg_len = 4 + rng.randint(0, 5)
        cc = x
        r = top - 1
        for k in range(seg_len):
            rr = r + 1
            if rr >= h:
                break
            dc = -1 if k % 2 else 1
            cc = max(0, min(w - 1, cc + dc))
            grid[rr, cc] = color
            r = rr

    return grid


def _closest_obstacle(comp, obstacles):
    if not obstacles:
        return None
    s = set(comp)
    best = None
    bestd = 10 ** 9
    for ob in obstacles:
        for p in ob:
            for q in s:
                d = abs(p[0] - q[0]) + abs(p[1] - q[1])
                if d < bestd:
                    bestd = d
                    best = ob
    return best


def _build_group(grid, color, comp, obstacles):
    h, w = grid.shape
    comp_set = set(comp)
    bar = sorted([p for p in comp if p[1] == min(c for _, c in comp)], key=lambda x: x[0])
    tail_len = max(1, len(comp_set) - len(bar))

    ob = _closest_obstacle(comp, obstacles)
    if ob is None:
        return None

    obr0, obc0, obr1, obc1 = _bbox(ob)
    path = _perimeter_path((max(0, obr0 - 1), max(0, obc0 - 1), min(h - 1, obr1 + 1), min(w - 1, obc1 + 1)))
    if not path:
        return None
    if bar:
        start = bar[-1]
    else:
        start = comp[0]
    start_idx = min(range(len(path)), key=lambda i: abs(path[i][0] - start[0]) + abs(path[i][1] - start[1]))
    route = path[start_idx:] + path[:start_idx]
    if len(route) == 0:
        return None
    route = route[:tail_len]

    all_points = list(comp_set | set(route) | set(ob))
    r0, c0, r1, c1 = _bbox(all_points)
    hh = r1 - r0 + 1
    ww = c1 - c0 + 1
    canvas = np.zeros((hh, ww), dtype=int)
    for r, c in ob:
        canvas[r - r0, c - c0] = _GRAY
    for r, c in comp:
        canvas[r - r0, c - c0] = color
    for r, c in route:
        if 0 <= r - r0 < hh and 0 <= c - c0 < ww:
            canvas[r - r0, c - c0] = color

    return {
        "canvas": canvas,
        "height": hh,
        "width": ww,
        "key": (len(ob), _obstacle_perimeter(ob), c0),
        "min_c": c0,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """1) For each color, find connected bar+segment and nearby gray obstacle.
2) Rebuild segment as a fixed-length clockwise boundary walk around obstacle.
3) Sort groups by obstacle perimeter and concatenate bottom-aligned with 1-column gaps.
4) Keep transformed groups only; keep obstacle geometry copied through grouping."""

    g = np.array(grid)
    comps_by_color = {}
    for color in [1, 2, 3, 4]:
        comps = _components(g, color)
        if comps:
            comps_by_color[color] = comps

    obstacles = []
    for comp in _components(g, _GRAY):
        obstacles.append(comp)

    if not obstacles or not comps_by_color:
        return g

    groups = []
    for comps in comps_by_color.values():
        for comp in comps:
            gp = _build_group(g, int(g[comp[0]]), comp, obstacles)
            if gp is not None:
                groups.append(gp)

    if not groups:
        return g

    groups.sort(key=lambda x: (x["key"][1], x["key"][2]))
    total_w = sum(g["width"] for g in groups) + max(0, len(groups) - 1)
    max_h = max(g["height"] for g in groups)
    # cap final size to safe bound
    if total_w > 30:
        total_w = min(total_w, 30)

    out = np.zeros((max_h, total_w), dtype=int)
    x = 0
    for gp in groups:
        if x >= total_w:
            break
        ww = min(gp["width"], total_w - x)
        hh = gp["height"]
        y = max_h - hh
        out[y : y + hh, x : x + ww] = gp["canvas"][:, :ww]
        x += ww + 1

    return out


if __name__ == "__main__":
    print(main(generate_input()).shape)
