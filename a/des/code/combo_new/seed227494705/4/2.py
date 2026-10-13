import random
import numpy as np


BG = 0


def _components(mask):
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells = []
            while stack:
                cr, cc = stack.pop()
                cells.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(cells)
    return comps


def _bbox(cells):
    rs = [r for r, _ in cells]
    cs = [c for _, c in cells]
    return min(rs), min(cs), max(rs), max(cs)


def _adjacent(a, b):
    bs = set(b)
    for r, c in a:
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            if (r + dr, c + dc) in bs:
                return True
    return False


def _leader_info(cells):
    r0, c0, r1, c1 = _bbox(cells)
    if (r1 - r0, c1 - c0, len(cells)) != (2, 2, 8):
        return None
    s = {(r - r0, c - c0) for r, c in cells}
    miss = ({(r, c) for r in range(3) for c in range(3)} - s)
    if len(miss) != 1:
        return None
    m = next(iter(miss))
    dirs = {(0, 1): (-1, 0), (2, 1): (1, 0), (1, 0): (0, -1), (1, 2): (0, 1)}
    if m not in dirs:
        return None
    return dirs[m], (r0 + m[0], c0 + m[1])


def main(grid: np.ndarray) -> np.ndarray:
    """Merge same-colored fragments toward the notched leader fragment.

    Input structure: On a black background, each foreground color appears as
    two or three separated 4-connected fragments. Exactly one fragment of a
    color is a nearly solid 3x3 block with one edge-midpoint indentation.
    Rule: For each color, use the indentation direction as the translation
    direction. Slide every non-leader fragment in that direction until it
    becomes 4-adjacent to the growing same-color union, then fill the leader's
    missing indentation pixel.
    Key insight: The active direction is encoded by the leader fragment's
    unique concave notch, not by the fragments' absolute positions.
    """
    out = grid.copy()
    h, w = grid.shape
    for col in [int(x) for x in np.unique(grid) if x != BG]:
        comps = _components(grid == col)
        leader = None
        info = None
        for comp in comps:
            info = _leader_info(comp)
            if info is not None:
                leader = comp
                break
        if leader is None:
            continue
        direction, fill_cell = info
        union = set(leader)
        for comp in comps:
            if comp == leader:
                continue
            for r, c in comp:
                out[r, c] = BG
        for comp in sorted([c for c in comps if c != leader], key=lambda x: len(x), reverse=True):
            moved = list(comp)
            for _ in range(max(h, w)):
                if _adjacent(moved, union):
                    break
                nxt = [(r + direction[0], c + direction[1]) for r, c in moved]
                if any(r < 0 or r >= h or c < 0 or c >= w for r, c in nxt):
                    break
                blockers = {(r, c) for r, c in nxt if out[r, c] not in (BG, col)}
                if blockers:
                    break
                moved = nxt
            for r, c in moved:
                out[r, c] = col
                union.add((r, c))
        out[fill_cell] = col
    return out


def generate_input() -> np.ndarray:
    h = w = random.choice([18, 20, 22, 24, 26])
    grid = np.zeros((h, w), dtype=int)
    colors = random.sample([1, 2, 3, 4, 6, 7, 8, 9], random.randint(2, 4))
    dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    placements = [(5, 5), (5, w - 8), (h - 8, 5), (h - 8, w - 8)]
    random.shuffle(placements)
    missing_for_dir = {(-1, 0): (0, 1), (1, 0): (2, 1), (0, -1): (1, 0), (0, 1): (1, 2)}
    for col, (r0, c0) in zip(colors, placements):
        d = random.choice(dirs)
        miss = missing_for_dir[d]
        for rr in range(3):
            for cc in range(3):
                if (rr, cc) != miss:
                    grid[r0 + rr, c0 + cc] = col
        # Put trailing fragments opposite the movement direction, so sliding
        # along d brings them into the leader.
        br, bc = r0 - d[0] * random.randint(4, 6), c0 - d[1] * random.randint(4, 6)
        frag = [(0, 0), (0, 1)] if d[0] else [(0, 0), (1, 0)]
        for dr, dc in frag:
            if 1 <= br + dr < h - 1 and 1 <= bc + dc < w - 1:
                grid[br + dr, bc + dc] = col
        if random.random() < 0.6:
            br2, bc2 = br - d[0] * 3, bc - d[1] * 3
            if 1 <= br2 < h - 1 and 1 <= bc2 < w - 1:
                grid[br2, bc2] = col
    return grid
