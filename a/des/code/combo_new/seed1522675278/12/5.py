import random
import numpy as np


BG = 0
BLUE = 1
RED = 2
GREY = 5


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match framed sample objects to equal-sized template holes and paste only their interiors.

    Input structure:
    - A grey divider splits a left template area from right sample objects.
    - The left side contains a large blue shape with rectangular black holes.
    - Each sample on the right is a red rectangular frame whose outer size matches one hole.

    Rule:
    - Measure every hole size in the blue template.
    - Match each hole to the sample whose red outer frame has the same size.
    - Remove the red frame and place only the sample interior into the hole, leaving a
      one-pixel black margin around it.
    - Return only the left template area.

    Key insight:
    - Matching uses the sample's full framed size, but placement uses only the interior
      payload after stripping that frame.
    """
    divider = int(np.where(np.all(grid == GREY, axis=0))[0][0])
    left = grid[:, :divider].copy()
    blue_pts = np.argwhere(left == BLUE)
    r0, c0 = blue_pts.min(axis=0)
    r1, c1 = blue_pts.max(axis=0)
    holes = []
    mask = left[r0 + 1 : r1, c0 + 1 : c1] == BG
    for comp in _components(mask):
        rs = [p[0] for p in comp]
        cs = [p[1] for p in comp]
        hr = max(rs) - min(rs) + 1
        wc = max(cs) - min(cs) + 1
        holes.append((r0 + 1 + min(rs), hr, c0 + 1 + min(cs), wc))
    samples = {}
    for comp in _red_rects(grid[:, divider + 1 :]):
        r, h, c, w = comp
        inner = grid[r + 1 : r + h - 1, divider + 1 + c + 1 : divider + 1 + c + w - 1]
        samples[(h, w)] = inner.copy()
    out = grid[:, :divider].copy()
    for r, h, c, w in holes:
        out[r + 1 : r + h - 1, c + 1 : c + w - 1] = samples[(h, w)]
    return out


def _components(mask: np.ndarray):
    seen = set()
    out = []
    H, W = mask.shape
    for r in range(H):
        for c in range(W):
            if not mask[r, c] or (r, c) in seen:
                continue
            stack = [(r, c)]
            seen.add((r, c))
            comp = []
            while stack:
                x, y = stack.pop()
                comp.append((x, y))
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < H and 0 <= ny < W and mask[nx, ny] and (nx, ny) not in seen:
                        seen.add((nx, ny))
                        stack.append((nx, ny))
            out.append(comp)
    return out


def _red_rects(grid: np.ndarray):
    rects = []
    for r in range(grid.shape[0] - 2):
        for c in range(grid.shape[1] - 2):
            if grid[r, c] != RED:
                continue
            for h in range(4, 8):
                for w in range(4, 8):
                    if r + h > grid.shape[0] or c + w > grid.shape[1]:
                        continue
                    if np.all(grid[r, c : c + w] == RED) and np.all(grid[r + h - 1, c : c + w] == RED) and np.all(grid[r : r + h, c] == RED) and np.all(grid[r : r + h, c + w - 1] == RED):
                        rects.append((r, h, c, w))
    uniq = []
    for x in rects:
        if x not in uniq:
            uniq.append(x)
    return uniq


def generate_input() -> np.ndarray:
    H, W = 24, 24
    grid = np.zeros((H, W), dtype=int)
    divider = 12
    grid[:, divider] = GREY
    grid[2:20, 1:11] = BLUE
    holes = []
    used = []
    for _ in range(random.randint(2, 3)):
        h = random.randint(4, 6)
        w = random.randint(4, 6)
        for _ in range(200):
            r = random.randint(3, 19 - h)
            c = random.randint(2, 10 - w)
            if all(r + h + 1 <= rr or rr + hh + 1 <= r or c + w + 1 <= cc or cc + ww + 1 <= c for rr, hh, cc, ww in used):
                used.append((r, h, c, w))
                grid[r : r + h, c : c + w] = BG
                holes.append((h, w))
                break
    right_used = []
    colors = random.sample([3, 4, 6, 7, 8, 9], len(holes))
    for (h, w), color in zip(holes, colors):
        for _ in range(200):
            r = random.randint(1, H - h - 1)
            c = random.randint(divider + 2, W - w - 1)
            if all(r + h <= rr or rr + hh <= r or c + w <= cc or cc + ww <= c for rr, hh, cc, ww in right_used):
                right_used.append((r, h, c, w))
                grid[r, c : c + w] = RED
                grid[r + h - 1, c : c + w] = RED
                grid[r : r + h, c] = RED
                grid[r : r + h, c + w - 1] = RED
                pat = np.random.randint(0, 2, (h - 2, w - 2))
                if pat.sum() in (0, pat.size):
                    pat[0, 0] = 1
                    pat[-1, -1] = 0
                grid[r + 1 : r + h - 1, c + 1 : c + w - 1][pat == 1] = color
                break
    return grid
