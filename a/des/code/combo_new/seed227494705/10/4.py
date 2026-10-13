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


def _bbox(pts):
    rs = [p[0] for p in pts]
    cs = [p[1] for p in pts]
    return min(rs), max(rs), min(cs), max(cs)


def main(grid: np.ndarray) -> np.ndarray:
    """Mirror arrow-selected source rectangles into grey target frames.

    Input structure: black background with grey hollow frames, a red arrow
    touching one side of each frame, and a same-size multicolor source rectangle
    beyond the arrow.
    Rule: remove all red arrows, find the source component in the direction the
    arrow points away from its frame, mirror the source left-right for horizontal
    arrows or top-bottom for vertical arrows, and stamp it into the frame interior.
    Key insight: the arrow encodes both where to search for the source and which
    axis to mirror before transfer.
    """
    out = grid.copy()
    out[out == 2] = 0
    grey = 5
    source_mask = (grid != 0) & (grid != grey) & (grid != 2)
    sources = _components(source_mask)
    frames = []
    for comp in _components(grid == grey):
        r1, r2, c1, c2 = _bbox(comp)
        if r2 - r1 >= 3 and c2 - c1 >= 3 and np.all(grid[r1, c1:c2 + 1] == grey) and np.all(grid[r2, c1:c2 + 1] == grey):
            frames.append((r1, r2, c1, c2))
    for r1, r2, c1, c2 in frames:
        checks = [((-1, 0), "up"), ((1, 0), "down"), ((0, -1), "left"), ((0, 1), "right")]
        direction = None
        for (dr, dc), name in checks:
            rr1, rr2 = max(0, r1 - 4), min(grid.shape[0], r2 + 5)
            cc1, cc2 = max(0, c1 - 4), min(grid.shape[1], c2 + 5)
            red = np.argwhere(grid[rr1:rr2, cc1:cc2] == 2)
            for ar, ac in red:
                ar += rr1
                ac += cc1
                if name == "up" and ar < r1 and c1 <= ac <= c2:
                    direction = (-1, 0)
                if name == "down" and ar > r2 and c1 <= ac <= c2:
                    direction = (1, 0)
                if name == "left" and ac < c1 and r1 <= ar <= r2:
                    direction = (0, -1)
                if name == "right" and ac > c2 and r1 <= ar <= r2:
                    direction = (0, 1)
            if direction:
                break
        if not direction:
            continue
        ih, iw = r2 - r1 - 1, c2 - c1 - 1
        candidates = []
        for comp in sources:
            sr1, sr2, sc1, sc2 = _bbox(comp)
            if sr2 - sr1 + 1 == ih and sc2 - sc1 + 1 == iw:
                center = ((sr1 + sr2) / 2, (sc1 + sc2) / 2)
                frame_center = ((r1 + r2) / 2, (c1 + c2) / 2)
                dot = (center[0] - frame_center[0]) * direction[0] + (center[1] - frame_center[1]) * direction[1]
                if dot > 0:
                    candidates.append((dot, comp))
        if not candidates:
            continue
        comp = min(candidates, key=lambda x: x[0])[1]
        sr1, sr2, sc1, sc2 = _bbox(comp)
        block = grid[sr1:sr2 + 1, sc1:sc2 + 1]
        block = np.fliplr(block) if direction[1] else np.flipud(block)
        out[r1 + 1:r2, c1 + 1:c2] = block
    return out


def _draw_arrow(grid, side, r1, r2, c1, c2):
    mr, mc = (r1 + r2) // 2, (c1 + c2) // 2
    if side == "up":
        pts = [(r1 - 1, mc), (r1 - 2, mc), (r1 - 3, mc), (r1 - 3, mc - 1), (r1 - 3, mc + 1)]
    elif side == "down":
        pts = [(r2 + 1, mc), (r2 + 2, mc), (r2 + 3, mc), (r2 + 3, mc - 1), (r2 + 3, mc + 1)]
    elif side == "left":
        pts = [(mr, c1 - 1), (mr, c1 - 2), (mr, c1 - 3), (mr - 1, c1 - 3), (mr + 1, c1 - 3)]
    else:
        pts = [(mr, c2 + 1), (mr, c2 + 2), (mr, c2 + 3), (mr - 1, c2 + 3), (mr + 1, c2 + 3)]
    for r, c in pts:
        if 0 <= r < grid.shape[0] and 0 <= c < grid.shape[1]:
            grid[r, c] = 2


def generate_input() -> np.ndarray:
    grid = np.zeros((24, 24), dtype=int)
    ih, iw = random.choice([(3, 4), (4, 3), (4, 5)])
    r, c = random.randint(7, 11), random.randint(7, 11)
    grid[r:r + ih + 2, c] = 5
    grid[r:r + ih + 2, c + iw + 1] = 5
    grid[r, c:c + iw + 2] = 5
    grid[r + ih + 1, c:c + iw + 2] = 5
    side = random.choice(["up", "down", "left", "right"])
    _draw_arrow(grid, side, r, r + ih + 1, c, c + iw + 1)
    if side == "up":
        sr, sc = r - 3 - ih - 1, c + 1
    elif side == "down":
        sr, sc = r + ih + 5, c + 1
    elif side == "left":
        sr, sc = r + 1, c - 3 - iw - 1
    else:
        sr, sc = r + 1, c + iw + 5
    colors = [random.choice([1, 3, 4, 6, 7, 8, 9]) for _ in range(ih * iw)]
    k = 0
    for a in range(ih):
        for b in range(iw):
            if 0 <= sr + a < grid.shape[0] and 0 <= sc + b < grid.shape[1]:
                grid[sr + a, sc + b] = colors[k]
            k += 1
    return grid
