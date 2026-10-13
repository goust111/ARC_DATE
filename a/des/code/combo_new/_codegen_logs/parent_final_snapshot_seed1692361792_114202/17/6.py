import random
from collections import deque
import numpy as np

BLACK = 0
GREY = 5


def components(mask: np.ndarray):
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if mask[r, c] and not seen[r, c]:
                q = deque([(r, c)])
                seen[r, c] = True
                cells = []
                while q:
                    x, y = q.popleft()
                    cells.append((x, y))
                    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < h and 0 <= ny < w and mask[nx, ny] and not seen[nx, ny]:
                            seen[nx, ny] = True
                            q.append((nx, ny))
                comps.append(cells)
    return comps


def bbox(cells):
    rs = [r for r, _ in cells]
    cs = [c for _, c in cells]
    return min(rs), max(rs), min(cs), max(cs)


def main(grid: np.ndarray) -> np.ndarray:
    grid = np.array(grid, dtype=int)
    grey_cells = np.argwhere(grid == GREY)
    r0, c0 = grey_cells.min(axis=0)
    r1, c1 = grey_cells.max(axis=0)
    board = grid[r0 : r1 + 1, c0 : c1 + 1]
    markers = []
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            if grid[r, c] not in (BLACK, GREY):
                markers.append((r, c, grid[r, c]))
    pieces = []
    for cells in components((grid != BLACK) & (grid != GREY)):
        if any(r0 <= r <= r1 and c0 <= c <= c1 for r, c in cells):
            continue
        br0, br1, bc0, bc1 = bbox(cells)
        crop = grid[br0 : br1 + 1, bc0 : bc1 + 1].copy()
        vals = [grid[r, c] for r, c in cells]
        key = min(set(vals), key=lambda v: vals.count(v))
        base = max(set(vals), key=lambda v: vals.count(v))
        pieces.append((key, base, crop, (br0, br1, bc0, bc1)))
    out = grid.copy()
    for _, _, _, (br0, br1, bc0, bc1) in pieces:
        out[br0 : br1 + 1, bc0 : bc1 + 1] = BLACK
    for mr, mc, key in markers:
        match = [p for p in pieces if p[0] == key]
        if not match:
            continue
        _, base, crop, _ = match[0]
        key_pos = np.argwhere(crop == key)[0]
        for k in range(4):
            rot = np.rot90(crop, k)
            pos = np.argwhere(rot == key)[0]
            rr = mr - pos[0]
            cc = mc - pos[1]
            cells = np.argwhere(rot != BLACK)
            ok = True
            for r, c in cells:
                tr, tc = rr + r, cc + c
                if not (0 <= tr < grid.shape[0] and 0 <= tc < grid.shape[1]):
                    ok = False
                    break
                if out[tr, tc] != BLACK and (tr, tc) != (mr, mc):
                    ok = False
                    break
            if ok:
                rot = np.where(rot != BLACK, base, BLACK)
                for r, c in np.argwhere(rot != BLACK):
                    out[rr + r, cc + c] = base
                break
    return out


def piece_with_bump(base, marker, side, body_h, body_w, bump_d, bump_w):
    if side in ("up", "down"):
        arr = np.zeros((body_h + bump_d, body_w), dtype=int)
        if side == "up":
            arr[bump_d:, :] = base
            c0 = (body_w - bump_w) // 2
            arr[:bump_d, c0 : c0 + bump_w] = base
            arr[0, c0 + bump_w // 2] = marker
        else:
            arr[:body_h, :] = base
            c0 = (body_w - bump_w) // 2
            arr[body_h:, c0 : c0 + bump_w] = base
            arr[-1, c0 + bump_w // 2] = marker
    else:
        arr = np.zeros((body_h, body_w + bump_d), dtype=int)
        if side == "left":
            arr[:, bump_d:] = base
            r0 = (body_h - bump_w) // 2
            arr[r0 : r0 + bump_w, :bump_d] = base
            arr[r0 + bump_w // 2, 0] = marker
        else:
            arr[:, :body_w] = base
            r0 = (body_h - bump_w) // 2
            arr[r0 : r0 + bump_w, body_w:] = base
            arr[r0 + bump_w // 2, -1] = marker
    return arr


def generate_input() -> np.ndarray:
    h, w = random.randint(18, 24), random.randint(20, 28)
    grid = np.zeros((h, w), dtype=int)
    bh, bw = random.randint(8, 11), random.randint(8, 11)
    r0 = (h - bh) // 2
    c0 = (w - bw) // 2
    grid[r0 : r0 + bh, c0 : c0 + bw] = GREY
    marker_colors = random.sample([1, 2, 3, 4, 6, 7, 8, 9], random.randint(2, 4))
    for key in marker_colors:
        side = random.choice(["up", "down", "left", "right"])
        bump_d = random.randint(2, 3)
        bump_w = random.randint(2, 3)
        body_h, body_w = random.randint(3, 4), random.randint(3, 4)
        if side == "up":
            col = random.randint(c0 + 2, c0 + bw - bump_w - 2)
            grid[r0 : r0 + bump_d, col : col + bump_w] = BLACK
            grid[r0 + bump_d - 1, col + bump_w // 2] = key
        elif side == "down":
            col = random.randint(c0 + 2, c0 + bw - bump_w - 2)
            grid[r0 + bh - bump_d : r0 + bh, col : col + bump_w] = BLACK
            grid[r0 + bh - bump_d, col + bump_w // 2] = key
        elif side == "left":
            row = random.randint(r0 + 2, r0 + bh - bump_w - 2)
            grid[row : row + bump_w, c0 : c0 + bump_d] = BLACK
            grid[row + bump_w // 2, c0 + bump_d - 1] = key
        else:
            row = random.randint(r0 + 2, r0 + bh - bump_w - 2)
            grid[row : row + bump_w, c0 + bw - bump_d : c0 + bw] = BLACK
            grid[row + bump_w // 2, c0 + bw - bump_d] = key
        base = random.choice([v for v in [1, 2, 3, 4, 6, 7, 8, 9] if v != key])
        piece = piece_with_bump(base, key, side, body_h, body_w, bump_d, bump_w)
        pr = random.choice([1, h - piece.shape[0] - 1])
        pc = random.randint(1, w - piece.shape[1] - 1)
        if side in ("left", "right"):
            pr = random.randint(1, h - piece.shape[0] - 1)
            pc = random.choice([1, w - piece.shape[1] - 1])
        grid[pr : pr + piece.shape[0], pc : pc + piece.shape[1]][piece != BLACK] = piece[piece != BLACK]
    return grid
