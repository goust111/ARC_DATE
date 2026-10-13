import random
import numpy as np


BG = 0
GREY = 5
TAG_COLORS = {2: 0, 3: 1, 1: 2, 4: 3}


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
    """Fill left-frame gaps with matching right sprites and erase controls.

    Input structure: A full grey vertical divider splits the grid. Left of it
    are colored rectangular hollow frames, each with a short missing border
    segment and a singleton rotation-control pixel inside. Right of it are
    small same-colored pieces.
    Rule: For each frame, match the right piece by frame color, use the control
    color as the rotation instruction, restore the missing border segment with
    the frame color, erase the control pixel, and remove all right-side pieces.
    The grey divider remains fixed.
    Key insight: Color establishes the left/right pairing, while the internal
    control marker supplies the orientation needed to complete the frame gap.
    """
    out = grid.copy()
    h, w = grid.shape
    div_cols = [c for c in range(w) if np.count_nonzero(grid[:, c] == GREY) == h]
    div = div_cols[0] if div_cols else w // 2
    for col in [int(x) for x in np.unique(grid) if x not in (BG, GREY) and x not in TAG_COLORS]:
        comps = _components(grid == col)
        for comp in comps:
            r0, c0, r1, c1 = _bbox(comp)
            if c1 >= div or r1 - r0 < 3 or c1 - c0 < 3:
                continue
            border = []
            for c in range(c0, c1 + 1):
                border.append((r0, c))
                border.append((r1, c))
            for r in range(r0 + 1, r1):
                border.append((r, c0))
                border.append((r, c1))
            for r, c in border:
                if out[r, c] == BG:
                    out[r, c] = col
            for r in range(r0 + 1, r1):
                for c in range(c0 + 1, c1):
                    if int(out[r, c]) in TAG_COLORS:
                        out[r, c] = BG
        for comp in comps:
            if _bbox(comp)[1] > div:
                for r, c in comp:
                    out[r, c] = BG
    return out


def generate_input() -> np.ndarray:
    h, w = 20, 29
    grid = np.zeros((h, w), dtype=int)
    div = w // 2
    grid[:, div] = GREY
    frame_colors = random.sample([6, 7, 8, 9], 3)
    tag_colors = [2, 3, 1, 4]
    starts = [(2, 2), (9, 2), (2, 8)]
    for col, (r0, c0) in zip(frame_colors, starts):
        fh, fw = random.randint(5, 7), random.randint(5, 6)
        r1, c1 = r0 + fh - 1, c0 + fw - 1
        grid[r0, c0:c1 + 1] = col
        grid[r1, c0:c1 + 1] = col
        grid[r0:r1 + 1, c0] = col
        grid[r0:r1 + 1, c1] = col
        side = random.choice(["top", "bottom", "left", "right"])
        if side in ("top", "bottom"):
            rr = r0 if side == "top" else r1
            gap_start = random.randint(c0 + 1, c1 - 2)
            grid[rr, gap_start:gap_start + 2] = BG
            piece = [(0, 0), (0, 1)]
        else:
            cc = c0 if side == "left" else c1
            gap_start = random.randint(r0 + 1, r1 - 2)
            grid[gap_start:gap_start + 2, cc] = BG
            piece = [(0, 0), (1, 0)]
        tr = random.choice([r0 + 1, r1 - 1])
        tc = random.choice([c0 + 1, c1 - 1])
        grid[tr, tc] = random.choice(tag_colors)
        pr = random.randint(2, h - 4)
        pc = random.randint(div + 2, w - 4)
        for dr, dc in piece:
            grid[pr + dr, pc + dc] = col
    return grid
