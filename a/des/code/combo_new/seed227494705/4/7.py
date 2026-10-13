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


def main(grid: np.ndarray) -> np.ndarray:
    """Open matching rooms by ray-casting from each door to its first key.

    Input structure: Colored one-pixel rectangular room walls have one missing
    door pixel on a side. Singleton colored keys sit in the outside background,
    with extra distractor keys present.
    Rule: For each room, find the door gap and scan outward perpendicular to
    that side until the first key pixel. If the key color matches the wall
    color, fill all interior background cells with that color and erase the key;
    otherwise leave both unchanged. Walls and door gaps remain fixed.
    Key insight: Door direction gives the spatial key-room pairing, and wall
    color decides whether that first key actually opens the room.
    """
    out = grid.copy()
    h, w = grid.shape
    for col in [int(x) for x in np.unique(grid) if x != BG]:
        comps = _components(grid == col)
        for comp in comps:
            if len(comp) < 6:
                continue
            r0, c0, r1, c1 = _bbox(comp)
            if r1 - r0 < 3 or c1 - c0 < 3:
                continue
            border = []
            for c in range(c0, c1 + 1):
                border.append((r0, c))
                border.append((r1, c))
            for r in range(r0 + 1, r1):
                border.append((r, c0))
                border.append((r, c1))
            missing = [p for p in border if grid[p] == BG]
            if len(missing) != 1:
                continue
            dr = dc = 0
            door = missing[0]
            if door[0] == r0:
                dr = -1
            elif door[0] == r1:
                dr = 1
            elif door[1] == c0:
                dc = -1
            else:
                dc = 1
            rr, cc = door[0] + dr, door[1] + dc
            key = None
            while 0 <= rr < h and 0 <= cc < w:
                if grid[rr, cc] != BG:
                    key = (rr, cc, int(grid[rr, cc]))
                    break
                rr += dr
                cc += dc
            if key and key[2] == col:
                for r in range(r0 + 1, r1):
                    for c in range(c0 + 1, c1):
                        if out[r, c] == BG:
                            out[r, c] = col
                out[key[0], key[1]] = BG
    return out


def generate_input() -> np.ndarray:
    h = w = random.choice([24, 26, 28])
    grid = np.zeros((h, w), dtype=int)
    colors = random.sample([1, 2, 3, 4, 6, 7, 8, 9], 3)
    rooms = [(4, 4), (4, 14), (14, 6)]
    for col, (r0, c0) in zip(colors, rooms):
        rh, rw = random.randint(5, 7), random.randint(5, 7)
        r1, c1 = r0 + rh - 1, c0 + rw - 1
        grid[r0, c0:c1 + 1] = col
        grid[r1, c0:c1 + 1] = col
        grid[r0:r1 + 1, c0] = col
        grid[r0:r1 + 1, c1] = col
        side = random.choice(["top", "bottom", "left", "right"])
        if side == "top":
            door = (r0, (c0 + c1) // 2)
            key = (max(0, r0 - random.randint(2, 4)), door[1])
        elif side == "bottom":
            door = (r1, (c0 + c1) // 2)
            key = (min(h - 1, r1 + random.randint(2, 4)), door[1])
        elif side == "left":
            door = ((r0 + r1) // 2, c0)
            key = (door[0], max(0, c0 - random.randint(2, 4)))
        else:
            door = ((r0 + r1) // 2, c1)
            key = (door[0], min(w - 1, c1 + random.randint(2, 4)))
        grid[door] = BG
        grid[key] = col if random.random() < 0.65 else random.choice([x for x in colors if x != col])
    for _ in range(3):
        r, c = random.randrange(h), random.randrange(w)
        if grid[r, c] == BG:
            grid[r, c] = random.choice(colors)
    return grid
