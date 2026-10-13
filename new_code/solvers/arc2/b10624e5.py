import numpy as np

from arc_utils import bbox_pts as _bbox, cc4 as _cc4, get_bg as _mode_color


def _find_dividers(g: np.ndarray, bg: int) -> tuple[int | None, int | None, int | None]:
    """
    Return (mid_row, mid_col, divider_color) if there is a full row and full col of the same
    non-bg color (quadrant divider), else (None, None, None).
    """
    h, w = g.shape
    full_rows = [(r, int(g[r, 0])) for r in range(h) if np.all(g[r, :] == g[r, 0]) and int(g[r, 0]) != bg]
    full_cols = [(c, int(g[0, c])) for c in range(w) if np.all(g[:, c] == g[0, c]) and int(g[0, c]) != bg]
    # prefer a color that appears in both a full row and full col
    for r, vr in full_rows:
        for c, vc in full_cols:
            if vr == vc:
                return int(r), int(c), int(vr)
    return None, None, None


def transform(grid: np.ndarray) -> np.ndarray:
    """
    Replicate a motif across quadrants with scaling and reflection keyed to red anchor blocks.

    Input structure:
    - Grid divided into four quadrants by a full-row and full-column divider of a single non-bg color.
    - Each quadrant contains a red (2) square anchor block of varying size.
    - The top-left quadrant also has a colored motif attached to its red anchor.

    Rule:
    - Use the top-left quadrant's motif as the source pattern.
    - For each other quadrant, scale the motif by the ratio of that quadrant's red block size to the source red block size.
    - Reflect horizontally for right-side quadrants, vertically for bottom quadrants (both for bottom-right).
    - Align the scaled/reflected motif so its red region overlaps the target anchor, then stamp non-bg pixels.

    Key insight: Red block size encodes the scale factor; quadrant position encodes the reflection axes.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    out = g.copy()

    bg = _mode_color(g)
    mid_r, mid_c, div_col = _find_dividers(g, bg)
    if mid_r is None or mid_c is None:
        return out

    # Find red anchor blocks (color 2) in each quadrant.
    red = _cc4(g == 2)
    if not red:
        return out

    # Quadrant key: (top/bottom, left/right)
    anchors: dict[tuple[int, int], tuple[int, int, int]] = {}
    # value: (r0,c0,size)
    for pts in red:
        r0, c0, r1, c1 = _bbox(pts)
        rr = (r0 + r1) // 2
        cc = (c0 + c1) // 2
        qb = 0 if rr < mid_r else 1
        qr = 0 if cc < mid_c else 1
        size = max(r1 - r0 + 1, c1 - c0 + 1)
        # take the largest if multiple in same quadrant (shouldn't happen)
        if (qb, qr) not in anchors or anchors[(qb, qr)][2] < size:
            anchors[(qb, qr)] = (int(r0), int(c0), int(size))

    if (0, 0) not in anchors:
        return out
    sr0, sc0, sL = anchors[(0, 0)]
    if sL <= 0:
        return out

    # Build a source patch around the TL motif: take the non-bg connected component (4-neigh)
    # in the TL quadrant that contains the TL red anchor.
    tl = g[0:mid_r, 0:mid_c]
    comps = _cc4(tl != bg)
    patch_bb = None
    for pts in comps:
        # component contains any red pixel?
        if any(tl[r, c] == 2 for r, c in pts):
            r0, c0, r1, c1 = _bbox(pts)
            patch_bb = (int(r0), int(c0), int(r1), int(c1))
            break
    if patch_bb is None:
        return out

    pr0, pc0, pr1, pc1 = patch_bb
    patch = tl[pr0 : pr1 + 1, pc0 : pc1 + 1].copy()

    def resize_nn(img: np.ndarray, new_h: int, new_w: int) -> np.ndarray:
        src_h, src_w = img.shape
        if new_h <= 0 or new_w <= 0:
            return img.copy()
        rr = (np.floor(np.arange(new_h) * src_h / new_h)).astype(int)
        cc = (np.floor(np.arange(new_w) * src_w / new_w)).astype(int)
        rr = np.clip(rr, 0, src_h - 1)
        cc = np.clip(cc, 0, src_w - 1)
        return img[rr[:, None], cc[None, :]]

    # For each quadrant anchor, paste a scaled/reflected version of the patch.
    for (qb, qr), (tr0, tc0, tL) in anchors.items():
        if (qb, qr) == (0, 0):
            continue
        if tL <= 0:
            continue

        patch_t = patch
        if qr == 1:
            patch_t = np.fliplr(patch_t)
        if qb == 1:
            patch_t = np.flipud(patch_t)

        ratio = float(tL) / float(sL)
        new_h = int(round(patch_t.shape[0] * ratio))
        new_w = int(round(patch_t.shape[1] * ratio))
        new_h = max(1, new_h)
        new_w = max(1, new_w)
        patch_s = resize_nn(patch_t, new_h, new_w)

        # Align scaled red bbox in patch_s to the target red bbox.
        rr_red, cc_red = np.where(patch_s == 2)
        if rr_red.size == 0:
            continue
        ar0, ac0 = int(rr_red.min()), int(cc_red.min())
        place_r0 = tr0 - ar0
        place_c0 = tc0 - ac0

        rr_p, cc_p = np.where(patch_s != bg)
        for rr0, cc0 in zip(rr_p.tolist(), cc_p.tolist(), strict=False):
            val = int(patch_s[rr0, cc0])
            rr_t = place_r0 + rr0
            cc_t = place_c0 + cc0
            if not (0 <= rr_t < h and 0 <= cc_t < w):
                continue
            if rr_t == mid_r or cc_t == mid_c:
                continue
            if out[rr_t, cc_t] == 2:
                continue
            out[rr_t, cc_t] = val

    return out
