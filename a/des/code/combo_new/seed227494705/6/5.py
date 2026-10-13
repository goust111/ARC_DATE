import random
import numpy as np


BG = 0
GREY = 5


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


def _norm_mask(cells):
    r0, c0, r1, c1 = _bbox(cells)
    m = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in cells:
        m[r - r0, c - c0] = True
    return m


def _match_rot(piece, hole):
    for k in range(4):
        rot = np.rot90(piece, k)
        if rot.shape == hole.shape and np.array_equal(rot, hole):
            return rot
    return None


def main(grid: np.ndarray) -> np.ndarray:
    """Place rotated jigsaw pieces into matching grey-frame slots.

    Input structure: A large grey solid rectangle contains black irregular
    slots. Around it are monochrome colored pieces whose shapes match the slots
    up to 90-degree rotations.
    Rule: Extract every internal black slot, match it to the unique same-shaped
    colored piece under rotation, clear the piece's original cells, and fill the
    slot with that piece color. Grey frame pixels remain unchanged.
    Key insight: Slot-piece pairing is by exact contour shape under rotation,
    not by area alone.
    """
    out = grid.copy()
    grey = max(_components(grid == GREY), key=len)
    gr0, gc0, gr1, gc1 = _bbox(grey)
    holes = []
    sub = grid[gr0:gr1 + 1, gc0:gc1 + 1] == BG
    for comp in _components(sub):
        shifted = [(r + gr0, c + gc0) for r, c in comp]
        if not any(r in (gr0, gr1) or c in (gc0, gc1) for r, c in shifted):
            holes.append(shifted)
    pieces = []
    for col in [int(x) for x in np.unique(grid) if x not in (BG, GREY)]:
        for comp in _components(grid == col):
            pieces.append((col, comp, _norm_mask(comp)))
    for hole in holes:
        hm = _norm_mask(hole)
        for col, comp, pm in pieces:
            if _match_rot(pm, hm) is not None:
                for r, c in comp:
                    out[r, c] = BG
                for r, c in hole:
                    out[r, c] = col
                break
    return out


def generate_input() -> np.ndarray:
    h = w = 28
    grid = np.zeros((h, w), dtype=int)
    grid[6:21, 7:22] = GREY
    shapes = [[(0, 0), (1, 0), (1, 1), (2, 1)], [(0, 1), (1, 0), (1, 1), (1, 2)], [(0, 0), (0, 1), (1, 1), (2, 1), (2, 2)]]
    holes_at = [(9, 10), (9, 16), (15, 13)]
    colors = random.sample([1, 2, 3, 4, 6, 7, 8, 9], 3)
    for shape, (r0, c0), col, idx in zip(shapes, holes_at, colors, range(3)):
        for dr, dc in shape:
            grid[r0 + dr, c0 + dc] = BG
        m = np.zeros((4, 4), dtype=bool)
        for dr, dc in shape:
            m[dr, dc] = True
        rot = np.rot90(m, random.randrange(4))
        pr, pc = 2 + idx * 8, 2
        for rr, cc in zip(*np.where(rot)):
            grid[pr + rr, pc + cc] = col
    return grid
