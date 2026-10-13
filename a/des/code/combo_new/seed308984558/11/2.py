from __future__ import annotations

import random

import numpy as np
from arc_utils import crop_bbox, find_components, paste


BG = 0


def _center(bbox: tuple[int, int, int, int]) -> tuple[float, float]:
    r0, c0, r1, c1 = bbox
    return (r0 + r1) / 2.0, (c0 + c1) / 2.0


def _grow_mask(h: int, w: int, area: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    r, c = random.randrange(h), random.randrange(w)
    mask[r, c] = True
    while int(mask.sum()) < area:
        frontier = []
        for rr, cc in np.argwhere(mask):
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = int(rr) + dr, int(cc) + dc
                if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc]:
                    frontier.append((nr, nc))
        nr, nc = random.choice(frontier)
        mask[nr, nc] = True
    return mask


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pair same-color mirrored fragments and slide each pair together into one symmetric figure.

    Input structure:
    - Black background with several small monochrome 4-connected objects.
    - Each non-black color appears exactly twice.
    - The two objects of a color are mirrored versions of one another and are aligned either horizontally or vertically.

    Rule:
    - Pair objects by color and detect whether each pair shares its row-center or its column-center.
    - Translate the pair along that alignment axis until the two bboxes just touch.
    - Place the touched pair so the combined figure is centered at the midpoint of the original pair centers.
    - Clear all unused cells to black.

    Key insight:
    - Color identifies the pairing, but the shared center row/column determines the assembly axis.
    """
    g = np.asarray(grid, dtype=int)
    out = np.zeros_like(g)
    comps = find_components(g, bg=BG, connectivity=4, monochromatic=True)
    by_color: dict[int, list[dict]] = {}
    for comp in comps:
        by_color.setdefault(int(comp["color"]), []).append(comp)
    for color, pair in by_color.items():
        a, b = pair
        ca = _center(a["bbox"])
        cb = _center(b["bbox"])
        pa = crop_bbox(g, a["bbox"])
        pb = crop_bbox(g, b["bbox"])
        if abs(ca[0] - cb[0]) < 1e-6:
            left, right = (a, b) if ca[1] <= cb[1] else (b, a)
            pl = crop_bbox(g, left["bbox"])
            pr = crop_bbox(g, right["bbox"])
            hl, wl = pl.shape
            hr, wr = pr.shape
            mid = (_center(left["bbox"])[1] + _center(right["bbox"])[1]) / 2.0
            new_c0 = int(round(mid - (wl + wr - 1) / 2.0))
            row = int(round(ca[0] - (hl - 1) / 2.0))
            paste(out, pl, row, new_c0, transparent=BG)
            paste(out, pr, int(round(ca[0] - (hr - 1) / 2.0)), new_c0 + wl, transparent=BG)
        else:
            top, bottom = (a, b) if ca[0] <= cb[0] else (b, a)
            pt = crop_bbox(g, top["bbox"])
            pb = crop_bbox(g, bottom["bbox"])
            ht, wt = pt.shape
            hb, wb = pb.shape
            mid = (_center(top["bbox"])[0] + _center(bottom["bbox"])[0]) / 2.0
            new_r0 = int(round(mid - (ht + hb - 1) / 2.0))
            col = int(round(ca[1] - (wt - 1) / 2.0))
            paste(out, pt, new_r0, col, transparent=BG)
            paste(out, pb, new_r0 + ht, int(round(ca[1] - (wb - 1) / 2.0)), transparent=BG)
        out[out == color] = color
    return out


def generate_input() -> np.ndarray:
    h = random.randint(16, 26)
    w = random.randint(16, 26)
    g = np.zeros((h, w), dtype=int)
    colors = random.sample([1, 2, 3, 4, 6, 7, 8, 9], random.randint(1, 3))
    occupied = np.zeros_like(g, dtype=bool)
    for color in colors:
        for _ in range(200):
            orient = random.choice(["h", "v"])
            mh = random.randint(2, 4)
            mw = random.randint(2, 3)
            area = random.randint(2, mh * mw)
            mask = _grow_mask(mh, mw, area)
            mate = np.fliplr(mask) if orient == "h" else np.flipud(mask)
            ph, pw = mask.shape
            gap = random.randint(2, 5)
            if orient == "h":
                row = random.randint(1, h - ph - 2)
                col = random.randint(1, w - (2 * pw + gap) - 2)
                pos = [(row, col, mask), (row, col + pw + gap, mate)]
            else:
                row = random.randint(1, h - (2 * ph + gap) - 2)
                col = random.randint(1, w - pw - 2)
                pos = [(row, col, mask), (row + ph + gap, col, mate)]
            ok = True
            for r0, c0, m in pos:
                view = occupied[r0 : r0 + m.shape[0], c0 : c0 + m.shape[1]]
                if np.any(view & m):
                    ok = False
                    break
            if not ok:
                continue
            for r0, c0, m in pos:
                g[r0 : r0 + m.shape[0], c0 : c0 + m.shape[1]][m] = color
                occupied[r0 : r0 + m.shape[0], c0 : c0 + m.shape[1]] |= m
            break
    return g
