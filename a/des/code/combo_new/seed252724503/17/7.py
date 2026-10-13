import random

import numpy as np


BG = 0
PALETTE = [1, 2, 3, 4, 5, 6, 7, 8, 9]


def _components(mask: np.ndarray):
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells = []
            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(cells)
    return comps


def _bbox(cells):
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _shape_signature(cells):
    r0, c0, r1, c1 = _bbox(cells)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in cells:
        mask[r - r0, c - c0] = True
    return mask


def _hole_components(shape_mask: np.ndarray):
    h, w = shape_mask.shape
    outside = np.zeros_like(shape_mask, dtype=bool)
    stack = [(0, 0)]
    padded = np.pad(~shape_mask, 1, constant_values=True)
    outside = np.zeros_like(padded, dtype=bool)
    outside[0, 0] = True
    while stack:
        r, c = stack.pop()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < padded.shape[0] and 0 <= nc < padded.shape[1] and padded[nr, nc] and not outside[nr, nc]:
                outside[nr, nc] = True
                stack.append((nr, nc))
    holes = []
    seen = np.zeros_like(padded, dtype=bool)
    for r in range(1, padded.shape[0] - 1):
        for c in range(1, padded.shape[1] - 1):
            if padded[r, c] and not outside[r, c] and not seen[r, c]:
                stack = [(r, c)]
                seen[r, c] = True
                cells = []
                while stack:
                    rr, cc = stack.pop()
                    cells.append((rr - 1, cc - 1))
                    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        nr, nc = rr + dr, cc + dc
                        if padded[nr, nc] and not outside[nr, nc] and not seen[nr, nc]:
                            seen[nr, nc] = True
                            stack.append((nr, nc))
                holes.append(cells)
    return holes


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move each scattered fragment into the uniquely matching missing region of the central shape.

    Input structure:
    - The background is black and the largest foreground object is a single-color central shape.
    - That central shape contains several black missing regions fully enclosed inside its bounding box.
    - Smaller colored fragments are scattered around the outside and each fragment matches one missing region exactly without rotation or reflection.

    Rule:
    - Identify the enclosed holes inside the central shape and normalize each hole shape.
    - Match every outer fragment to the unique hole with the same shape.
    - Clear the fragment's original location and repaint the matched hole with the fragment's own color.

    Key insight:
    - Colors do not determine the destination; exact untranslated shape equality between fragments and holes does.
    """
    g = np.asarray(grid, dtype=int)
    comps = _components(g != BG)
    central = max(comps, key=len)
    central_bbox = _bbox(central)
    cr0, cc0, cr1, cc1 = central_bbox
    central_color = int(g[central[0]])
    central_mask = g[cr0 : cr1 + 1, cc0 : cc1 + 1] == central_color
    holes = []
    for cells in _hole_components(central_mask):
        abs_cells = [(cr0 + r, cc0 + c) for r, c in cells]
        holes.append({"cells": abs_cells, "sig": _shape_signature(abs_cells)})

    out = g.copy()
    for comp in comps:
        if comp is central:
            continue
        color = int(g[comp[0]])
        sig = _shape_signature(comp)
        match = next((hole for hole in holes if hole["sig"].shape == sig.shape and np.array_equal(hole["sig"], sig)), None)
        if match is None:
            raise ValueError("no matching hole for fragment")
        for r, c in comp:
            out[r, c] = BG
        for r, c in match["cells"]:
            out[r, c] = color
    return out


def _random_mask(rng: random.Random):
    h = rng.randint(2, 4)
    w = rng.randint(2, 4)
    target = rng.randint(2, min(5, h * w))
    cells = {(rng.randrange(h), rng.randrange(w))}
    while len(cells) < target:
        r, c = rng.choice(tuple(cells))
        dr, dc = rng.choice(((1, 0), (-1, 0), (0, 1), (0, -1)))
        nr, nc = r + dr, c + dc
        if 0 <= nr < h and 0 <= nc < w:
            cells.add((nr, nc))
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in cells:
        mask[r - r0, c - c0] = True
    return mask


def _placeable(occ: np.ndarray, top: int, left: int, h: int, w: int):
    return top >= 0 and left >= 0 and top + h <= occ.shape[0] and left + w <= occ.shape[1] and not np.any(
        occ[top : top + h, left : left + w]
    )


def _gen_attempt(rng: random.Random):
    h = rng.randint(18, 26)
    w = rng.randint(18, 26)
    grid = np.full((h, w), BG, dtype=int)
    occ = np.zeros_like(grid, dtype=bool)
    central_color = rng.choice(PALETTE)
    hole_count = rng.randint(2, 4)
    hole_masks = []
    used_shapes = []
    for _ in range(hole_count):
        for _ in range(50):
            mask = _random_mask(rng)
            if all(mask.shape != other.shape or not np.array_equal(mask, other) for other in used_shapes):
                used_shapes.append(mask)
                hole_masks.append(mask)
                break
        else:
            return None

    rect_h = rng.randint(8, 12)
    rect_w = rng.randint(8, 12)
    top = (h - rect_h) // 2
    left = (w - rect_w) // 2
    grid[top : top + rect_h, left : left + rect_w] = central_color
    occ[top : top + rect_h, left : left + rect_w] = True

    placed_holes = []
    for mask in hole_masks:
        mh, mw = mask.shape
        for _ in range(100):
            ht = rng.randint(top + 1, top + rect_h - mh - 1)
            hl = rng.randint(left + 1, left + rect_w - mw - 1)
            conflict = False
            for ot, ol, om in placed_holes:
                oh, ow = om.shape
                if not (ht + mh + 1 <= ot or ot + oh + 1 <= ht or hl + mw + 1 <= ol or ol + ow + 1 <= hl):
                    conflict = True
                    break
            if not conflict:
                view = grid[ht : ht + mh, hl : hl + mw]
                view[mask] = BG
                placed_holes.append((ht, hl, mask))
                break
        else:
            return None

    frag_colors = rng.sample([c for c in PALETTE if c != central_color], hole_count)
    for idx, (_, _, mask) in enumerate(placed_holes):
        mh, mw = mask.shape
        color = frag_colors[idx]
        placed = False
        for _ in range(200):
            ft = rng.randint(0, h - mh)
            fl = rng.randint(0, w - mw)
            if not (ft + mh <= top - 1 or ft >= top + rect_h + 1 or fl + mw <= left - 1 or fl >= left + rect_w + 1):
                continue
            if not _placeable(occ, ft, fl, mh, mw):
                continue
            view = grid[ft : ft + mh, fl : fl + mw]
            view[mask] = color
            rr0 = max(0, ft - 1)
            cc0 = max(0, fl - 1)
            rr1 = min(h, ft + mh + 1)
            cc1 = min(w, fl + mw + 1)
            occ[rr0:rr1, cc0:cc1] = True
            placed = True
            break
        if not placed:
            return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(300):
        grid = _gen_attempt(random.Random())
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
