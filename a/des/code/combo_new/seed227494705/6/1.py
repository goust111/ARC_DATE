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


def _side(mask):
    rs, cs = np.where(mask)
    h, w = mask.shape
    if len(rs) == 0:
        return None
    if np.all(rs == 0):
        return "top"
    if np.all(rs == h - 1):
        return "bottom"
    if np.all(cs == 0):
        return "left"
    if np.all(cs == w - 1):
        return "right"
    return None


def _rotate_to_side(sprite_vals, band_color, target_side):
    for k in range(4):
        rot = np.rot90(sprite_vals, -k)
        side = _side(rot == band_color)
        if side == target_side:
            return rot
    return sprite_vals


def main(grid: np.ndarray) -> np.ndarray:
    """Embed outside multicolor blocks into matching band-marked frames.

    Input structure: Colored hollow rectangular frames contain a short internal
    band on one inner side. Outside the frames are compact multicolor blocks,
    each with an edge band whose color matches one frame's internal band.
    Rule: Pair a block to a frame by band color. Rotate the block until its band
    lies on the same side as the frame's internal band, clear the original block,
    and place it in the frame interior with the bands overlapping. Frames and
    background remain unchanged.
    Key insight: Band color selects the target frame, while band side supplies
    the rotation needed before insertion.
    """
    out = grid.copy()
    h, w = grid.shape
    frames = []
    frame_area = set()
    for col in [int(x) for x in np.unique(grid) if x not in (BG, GREY)]:
        for comp in _components(grid == col):
            r0, c0, r1, c1 = _bbox(comp)
            if r1 - r0 >= 4 and c1 - c0 >= 4:
                border = {(r0, c) for c in range(c0, c1 + 1)} | {(r1, c) for c in range(c0, c1 + 1)}
                border |= {(r, c0) for r in range(r0, r1 + 1)} | {(r, c1) for r in range(r0, r1 + 1)}
                if all(grid[p] == col for p in border):
                    frames.append((r0, c0, r1, c1, col))
                    frame_area |= {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)}
    sprites = []
    outside = (grid != BG) & (grid != GREY)
    for r, c in frame_area:
        outside[r, c] = False
    for comp in _components(outside):
        r0, c0, r1, c1 = _bbox(comp)
        vals = grid[r0:r1 + 1, c0:c1 + 1].copy()
        vals[vals == 0] = BG
        colors = set(int(grid[r, c]) for r, c in comp)
        sprites.append((comp, vals, colors))
    for r0, c0, r1, c1, fcol in frames:
        interior = grid[r0 + 1:r1, c0 + 1:c1]
        band_colors = [int(x) for x in np.unique(interior) if x not in (BG, fcol)]
        if not band_colors:
            continue
        bcol = band_colors[0]
        band_mask = interior == bcol
        target_side = _side(band_mask)
        match = next((s for s in sprites if bcol in s[2]), None)
        if match is None:
            continue
        comp, vals, _ = match
        for rr, cc in comp:
            out[rr, cc] = BG
        rot = _rotate_to_side(vals, bcol, target_side)
        ih, iw = r1 - r0 - 1, c1 - c0 - 1
        for rr in range(min(ih, rot.shape[0])):
            for cc in range(min(iw, rot.shape[1])):
                if rot[rr, cc] != BG:
                    out[r0 + 1 + rr, c0 + 1 + cc] = int(rot[rr, cc])
    return out


def generate_input() -> np.ndarray:
    h = w = 28
    grid = np.zeros((h, w), dtype=int)
    frame_cols = random.sample([1, 2, 3, 4, 6, 7, 8, 9], 3)
    band_cols = random.sample([1, 2, 3, 4, 6, 7, 8, 9], 3)
    positions = [(3, 3), (3, 13), (13, 5)]
    sides = ["top", "bottom", "left", "right"]
    for idx, (fcol, bcol, (r0, c0)) in enumerate(zip(frame_cols, band_cols, positions)):
        fh = fw = 6
        r1, c1 = r0 + fh - 1, c0 + fw - 1
        grid[r0, c0:c1 + 1] = fcol
        grid[r1, c0:c1 + 1] = fcol
        grid[r0:r1 + 1, c0] = fcol
        grid[r0:r1 + 1, c1] = fcol
        side = random.choice(sides)
        if side == "top":
            grid[r0 + 1, c0 + 2:c0 + 4] = bcol
        elif side == "bottom":
            grid[r1 - 1, c0 + 2:c0 + 4] = bcol
        elif side == "left":
            grid[r0 + 2:r0 + 4, c0 + 1] = bcol
        else:
            grid[r0 + 2:r0 + 4, c1 - 1] = bcol
        sprite = np.zeros((4, 4), dtype=int)
        sprite[0, 1:3] = bcol
        sprite[1:3, 1:3] = random.choice([x for x in frame_cols if x != fcol])
        sprite[3, 2] = fcol
        sprite = np.rot90(sprite, random.randrange(4))
        pr, pc = 20, 2 + idx * 8
        for rr in range(4):
            for cc in range(4):
                if sprite[rr, cc]:
                    grid[pr + rr, pc + cc] = sprite[rr, cc]
    return grid
