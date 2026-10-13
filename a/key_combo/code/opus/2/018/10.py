import numpy as np


_BG = 0
_BLUE = 1
_RED = 2
_GREEN = 3
_YELLOW = 4
_ORANGE = 7
_BROWN = 9

_generate_calls = 0


def _is_symmetric(mask, axis, axis_pos):
    h, w = mask.shape
    pts = set(zip(*np.where(mask == _BLUE)))
    ok = True
    if axis == "H":
        for r, c in pts:
            if (2 * axis_pos - r, c) not in pts:
                ok = False
                break
    else:
        for r, c in pts:
            if (r, 2 * axis_pos - c) not in pts:
                ok = False
                break
    return ok


def _find_axis(mask, axis):
    h, w = mask.shape
    if axis == "H":
        best = None
        bestv = None
        for r in range(h):
            ok = _is_symmetric(mask, "H", r)
            if ok:
                best = r
                break
        if best is None:
            best = h // 2
        return best
    else:
        best = None
        for c in range(w):
            if _is_symmetric(mask, "V", c):
                best = c
                break
        if best is None:
            best = w // 2
        return best


def _connected(mask):
    h, w = mask.shape
    vis = np.zeros((h, w), dtype=bool)
    cells = []
    for r in range(h):
        for c in range(w):
            if mask[r, c] != _BLUE or vis[r, c]:
                continue
            comp = []
            stack = [(r, c)]
            vis[r, c] = True
            while stack:
                x, y = stack.pop()
                comp.append((x, y))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nx, ny = x + dr, y + dc
                    if 0 <= nx < h and 0 <= ny < w and not vis[nx, ny] and mask[nx, ny] == _BLUE:
                        vis[nx, ny] = True
                        stack.append((nx, ny))
            cells.append(comp)
    return cells


def generate_input():
    global _generate_calls
    rng = np.random.RandomState(12000 + _generate_calls)
    _generate_calls += 1

    h = 15
    w = 15
    g = np.zeros((h, w), dtype=int)

    # blue symmetric object centered near middle
    axis_r = 7
    axis_c = 7
    # draw a plus-like symmetric 4-connected object
    for d in range(-2, 3):
        g[axis_r + d, axis_c] = _BLUE
        g[axis_r, axis_c + d] = _BLUE

    # small non-symmetric red object in one quadrant
    # quadrant chosen by random
    q = rng.randint(0, 4)
    base = [(3, 1), (3, 2), (4, 2)]
    if q == 1:
        base = [(axis_r - 3, axis_c - 1), (axis_r - 2, axis_c - 1), (axis_r - 2, axis_c - 2)]
    elif q == 2:
        base = [(axis_r + 1, axis_c - 3), (axis_r + 2, axis_c - 3), (axis_r + 2, axis_c - 2)]
    elif q == 3:
        base = [(axis_r + 1, axis_c + 1), (axis_r + 1, axis_c + 2), (axis_r + 2, axis_c + 2)]

    # keep strictly in one quadrant
    for y, x in base:
        if 0 <= y < h and 0 <= x < w:
            g[y, x] = _RED

    colors = [_GREEN, _YELLOW, _ORANGE, _BROWN]
    rng.shuffle(colors)
    count = 4 + rng.randint(0, 5)
    for i in range(count):
        r, c = rng.randint(0, h), rng.randint(0, w)
        if g[r, c] != 0:
            continue
        g[r, c] = colors[i % 4]

    return g


def main(grid: np.ndarray) -> np.ndarray:
    """1) Locate symmetric blue body and recover horizontal/vertical axes.
2) Mirror-expand red object across both axes, creating four-quadrant pattern.
3) Keep only scattered singleton pixels covered by this mirror pattern.
4) Apply fixed color swap to kept singletons and output transformed board."""

    g = np.array(grid)
    h, w = g.shape

    blue_mask = g == _BLUE
    comps = _connected(blue_mask)
    if not comps:
        return g
    # assume one blue object
    comp = set(comps[0])
    minr = min(r for r, c in comp)
    maxr = max(r for r, c in comp)
    minc = min(c for r, c in comp)
    maxc = max(c for r, c in comp)

    blue_bbox = g[minr : maxr + 1, minc : maxc + 1]

    # detect symmetry axes by direct reflection checks
    hr, hc = blue_bbox.shape
    h_axis = _find_axis((blue_bbox == _BLUE), "H") + minr
    v_axis = _find_axis((blue_bbox == _BLUE), "V") + minc

    red_mask = g == _RED
    rr, cc = np.where(red_mask)
    mirrored = set()
    for y, x in zip(rr, cc):
        mirrored.add((y, x))
        my = int(2 * h_axis - y)
        mx = int(2 * v_axis - x)
        mirrored.add((my, x))
        mirrored.add((y, mx))
        mirrored.add((my, mx))

    out = np.zeros_like(g)

    # keep mirrored red pattern
    for y, x in mirrored:
        if 0 <= y < h and 0 <= x < w:
            out[y, x] = _RED

    # preserve blue object
    out[blue_mask] = _BLUE

    # swap surviving singleton pixels in original singleton set if covered by mask
    for y, x in zip(*np.where(g != 0)):
        color = int(g[y, x])
        if color not in (_GREEN, _YELLOW, _ORANGE, _BROWN):
            continue
        if (y, x) not in mirrored:
            continue
        if color == _GREEN:
            out[y, x] = _YELLOW
        elif color == _YELLOW:
            out[y, x] = _GREEN
        elif color == _ORANGE:
            out[y, x] = _BROWN
        elif color == _BROWN:
            out[y, x] = _ORANGE

    return out


if __name__ == "__main__":
    print(main(generate_input()).shape)
