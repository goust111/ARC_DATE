import random

import numpy as np


BG = 0
GREY = 5
PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]


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


def _center2(bbox):
    r0, c0, r1, c1 = bbox
    return r0 + r1, c0 + c1


def _mask_from_cells(cells):
    r0, c0, r1, c1 = _bbox(cells)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in cells:
        mask[r - r0, c - c0] = True
    return mask, (r0, c0, r1, c1)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use each small grey shape as a mask to cut the same shape from its aligned colored rectangle.

    Input structure:
    - The background is black and several large monochrome rectangles appear in distinct colors.
    - Several smaller grey connected shapes are placed elsewhere, each with a bounding box no larger than 5x5.
    - Every grey shape is aligned with exactly one large rectangle by sharing either the same center row or the same center column.

    Rule:
    - Pair each grey shape with its uniquely aligned rectangle.
    - Center the grey mask inside the rectangle's bounding box, remove that mask-shaped region from the rectangle, and recolor the grey shape itself with the rectangle's color.
    - Leave the rest of the scene unchanged.

    Key insight:
    - The small grey object is both a placement target and a shape template, while the large rectangle supplies both the color and the cut-out evidence.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    grey_components = _components(g == GREY)
    rectangles = []
    for color in np.unique(g):
        color = int(color)
        if color in (BG, GREY):
            continue
        comps = _components(g == color)
        for cells in comps:
            bbox = _bbox(cells)
            r0, c0, r1, c1 = bbox
            if len(cells) == (r1 - r0 + 1) * (c1 - c0 + 1):
                rectangles.append({"color": color, "bbox": bbox})

    for comp in grey_components:
        mask, small_bbox = _mask_from_cells(comp)
        small_center = _center2(small_bbox)
        matches = []
        for rect in rectangles:
            rect_center = _center2(rect["bbox"])
            if rect_center[0] == small_center[0] or rect_center[1] == small_center[1]:
                matches.append(rect)
        if not matches:
            raise ValueError("expected exactly one aligned rectangle")
        rect = min(
            matches,
            key=lambda item: min(
                abs(_center2(item["bbox"])[0] - small_center[0]) // 2,
                abs(_center2(item["bbox"])[1] - small_center[1]) // 2,
            ),
        )
        rr0, rc0, rr1, rc1 = rect["bbox"]
        mh, mw = mask.shape
        cut_top = (rr0 + rr1 - (mh - 1)) // 2
        cut_left = (rc0 + rc1 - (mw - 1)) // 2
        out[small_bbox[0] : small_bbox[2] + 1, small_bbox[1] : small_bbox[3] + 1][mask] = rect["color"]
        rect_view = out[cut_top : cut_top + mh, cut_left : cut_left + mw]
        rect_view[mask] = BG
    return out


def _random_mask(rng: random.Random):
    for _ in range(100):
        h = rng.choice([3, 5])
        w = rng.choice([3, 5])
        target = rng.randint(3, min(8, h * w - 1))
        cells = {(h // 2, w // 2)}
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
        if (r1 - r0) % 2 == 0 and (c1 - c0) % 2 == 0:
            mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
            for r, c in cells:
                mask[r - r0, c - c0] = True
            return mask
    raise RuntimeError("failed to build odd-sized mask")


def _fits(occ, top, left, h, w):
    return top >= 0 and left >= 0 and top + h <= occ.shape[0] and left + w <= occ.shape[1] and not np.any(
        occ[top : top + h, left : left + w]
    )


def _reserve(occ, top, left, h, w):
    rr0 = max(0, top - 1)
    cc0 = max(0, left - 1)
    rr1 = min(occ.shape[0], top + h + 1)
    cc1 = min(occ.shape[1], left + w + 1)
    occ[rr0:rr1, cc0:cc1] = True


def _gen_attempt(rng: random.Random):
    grid = np.full((26, 26), BG, dtype=int)
    occ = np.zeros_like(grid, dtype=bool)
    pair_count = rng.randint(2, 4)
    rect_centers_r = rng.sample([4, 10, 16, 22], pair_count)
    rect_centers_c = rng.sample([4, 10, 16, 22], pair_count)
    colors = rng.sample(PALETTE, pair_count)

    for idx in range(pair_count):
        mask = _random_mask(rng)
        mh, mw = mask.shape
        rect_h = rng.choice([7, 9])
        rect_w = rng.choice([7, 9])
        if rect_h < mh or rect_w < mw:
            return None
        center_r = rect_centers_r[idx]
        center_c = rect_centers_c[idx]
        rr0 = center_r - rect_h // 2
        rc0 = center_c - rect_w // 2
        if not _fits(occ, rr0, rc0, rect_h, rect_w):
            return None
        grid[rr0 : rr0 + rect_h, rc0 : rc0 + rect_w] = colors[idx]
        _reserve(occ, rr0, rc0, rect_h, rect_w)

        align_row = rng.random() < 0.5
        placed = False
        if align_row:
            for left in rng.sample(list(range(0, 24 - mw)), 24 - mw):
                top = center_r - mh // 2
                small_center_c2 = left + left + mw - 1
                if (
                    left + mw <= rc0 - 2
                    and all(small_center_c2 != 2 * c for c in rect_centers_c)
                    and _fits(occ, top, left, mh, mw)
                ):
                    grid[top : top + mh, left : left + mw][mask] = GREY
                    _reserve(occ, top, left, mh, mw)
                    placed = True
                    break
        else:
            for top in rng.sample(list(range(0, 26 - mh)), 26 - mh):
                left = center_c - mw // 2
                small_center_r2 = top + top + mh - 1
                if (
                    top + mh <= rr0 - 2
                    and all(small_center_r2 != 2 * r for r in rect_centers_r)
                    and _fits(occ, top, left, mh, mw)
                ):
                    grid[top : top + mh, left : left + mw][mask] = GREY
                    _reserve(occ, top, left, mh, mw)
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
