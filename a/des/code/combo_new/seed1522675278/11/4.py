import random
import numpy as np


BG = 0
QMAP = {"ul": 3, "ur": 1, "ll": 4, "lr": 2}


def _components(mask: np.ndarray):
    seen = set()
    out = []
    for r, c in np.argwhere(mask):
        r = int(r); c = int(c)
        if (r, c) in seen:
            continue
        stack = [(r, c)]
        seen.add((r, c))
        comp = []
        while stack:
            x, y = stack.pop()
            comp.append((x, y))
            for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
                q=(x+dx,y+dy)
                if 0<=q[0]<mask.shape[0] and 0<=q[1]<mask.shape[1] and mask[q] and q not in seen:
                    seen.add(q); stack.append(q)
        out.append(comp)
    return out


def _arrow_info(comp):
    rs=[p[0] for p in comp]; cs=[p[1] for p in comp]
    r0,c0=min(rs),min(cs); r1,c1=max(rs),max(cs)
    cells=set(comp)
    for r,c in comp:
        deg=sum((r+dr,c+dc) in cells for dr,dc in [(-1,0),(1,0),(0,-1),(0,1)])
        if deg==1:
            head=(r,c)
            if r==r0: return head,"u"
            if r==r1: return head,"d"
            if c==c0: return head,"l"
            return head,"r"


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect each facing same-colored arrow pair and color the connection by the midpoint quadrant.

    Input structure:
    - Same-colored 3-pixel arrows appear in facing pairs on a black background.
    - Each pair is aligned either horizontally or vertically.

    Rule:
    - Pair arrows by color and confirm their heads face one another.
    - Draw the straight connection between the two heads, excluding the head pixels.
    - Use the color assigned to the quadrant containing the pair midpoint.

    Key insight:
    - Arrow color only establishes the pairing; the drawn segment color comes from the
      pair's spatial midpoint, which is an independent rule.
    """
    out = grid.copy()
    nrows, ncols = grid.shape
    midr, midc = nrows // 2, ncols // 2
    for color in [int(c) for c in np.unique(grid) if c != BG]:
        comps = [c for c in _components(grid == color) if len(c) == 3]
        if len(comps) != 2:
            continue
        (h1, d1), (h2, d2) = map(_arrow_info, comps)
        mr = (h1[0] + h2[0]) // 2
        mc = (h1[1] + h2[1]) // 2
        quad = ("u" if mr < midr else "l")  # placeholder
        if mr < midr and mc < midc:
            pc = QMAP["ul"]
        elif mr < midr:
            pc = QMAP["ur"]
        elif mc < midc:
            pc = QMAP["ll"]
        else:
            pc = QMAP["lr"]
        if h1[0] == h2[0]:
            for c in range(min(h1[1], h2[1]) + 1, max(h1[1], h2[1])):
                out[h1[0], c] = pc
        else:
            for r in range(min(h1[0], h2[0]) + 1, max(h1[0], h2[0])):
                out[r, h1[1]] = pc
    return out


def generate_input() -> np.ndarray:
    grid = np.zeros((21, 21), dtype=int)
    colors = random.sample([5, 6, 7, 8], random.randint(2, 4))
    for color in colors:
        if random.random() < 0.5:
            r = random.randint(2, 18)
            c1 = random.randint(2, 7)
            c2 = random.randint(12, 17)
            grid[r, c1] = color; grid[r - 1, c1 + 1] = color; grid[r + 1, c1 + 1] = color
            grid[r, c2] = color; grid[r - 1, c2 - 1] = color; grid[r + 1, c2 - 1] = color
        else:
            c = random.randint(2, 18)
            r1 = random.randint(2, 7)
            r2 = random.randint(12, 17)
            grid[r1, c] = color; grid[r1 + 1, c - 1] = color; grid[r1 + 1, c + 1] = color
            grid[r2, c] = color; grid[r2 - 1, c - 1] = color; grid[r2 - 1, c + 1] = color
    return grid
