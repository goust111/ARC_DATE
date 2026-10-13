from __future__ import annotations
from collections import Counter
import numpy as np
from arc_utils import bbox, bbox_coords, components_mask, crop_bbox, find_components, get_bg, paste

def main(grid: np.ndarray) -> np.ndarray:
    """
    Inflate a frame's hole-layout and fill each hole with its matching piece.

    Input structure:
    - bg = most common color. Largest component is a monochrome frame enclosing bg holes.
    - Outside pieces composed of k×k blocks; k = side of smallest solid square piece.

    Rule:
    - Extract connected hole regions inside the frame as slots.
    - Match each piece to a slot by comparing block-level shapes under rotation/flip.
    - Output the layout scaled by k, with pieces placed in slots; frame color as background.

    Key insight: Pieces are k×k-block-scaled versions of holes; downsampling reveals shape matches under all 8 orientations.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    # --- find the largest connected component of foreground (multicolor) ---
    comps = find_components(g, bg=bg, connectivity=4, monochromatic=False, return_format="dict")
    assert not (not comps)
    frame_comp = max(comps, key=lambda d: int(d["area"]))
    fr0, fc0, fr1, fc1 = map(int, frame_comp["bbox"])

    # The frame is monochrome; use its dominant color as output background.
    frame_cols = [int(g[int(r), int(c)]) for r, c in frame_comp["coords"]]
    bgc = int(Counter(frame_cols).most_common(1)[0][0])

    # --- layout = bg holes inside the frame bbox, tight-cropped ---
    sub = crop_bbox(g, (fr0, fc0, fr1, fc1))
    hole = sub == int(bg)
    hole_box = bbox(hole)
    if hole_box is None:
        return np.full((1, 1), int(bgc), dtype=int)
    lr0, lc0, lr1, lc1 = map(int, hole_box)
    layout = hole[lr0 : lr1 + 1, lc0 : lc1 + 1]

    # --- slot components in the layout ---
    slots = []
    for coords in components_mask(layout, connectivity=4):
        rr0, cc0, rr1, cc1 = bbox_coords(coords)
        rr0, cc0, rr1, cc1 = map(int, (rr0, cc0, rr1, cc1))
        h = int(rr1 - rr0 + 1)
        w = int(cc1 - cc0 + 1)
        sm = np.zeros((h, w), dtype=bool)
        sm[coords[:, 0] - rr0, coords[:, 1] - cc0] = True
        slots.append({"area": int(sm.sum()), "r0": int(rr0), "c0": int(cc0), "mask": sm})
    slots.sort(key=lambda s: (-int(s["area"]), int(s["r0"]), int(s["c0"])))

    if not slots:
        return np.full((1, 1), int(bgc), dtype=int)

    # --- pieces = non-singleton foreground components excluding the frame component ---
    pieces = []
    for comp in comps:
        if comp is frame_comp:
            continue
        coords = comp["coords"]
        if coords.size == 0 or int(coords.shape[0]) <= 1:
            continue
        pr0, pc0, pr1, pc1 = map(int, comp["bbox"])
        patch = crop_bbox(g, (pr0, pc0, pr1, pc1))
        ph, pw = map(int, patch.shape)
        pm = np.zeros((ph, pw), dtype=bool)
        pm[coords[:, 0] - pr0, coords[:, 1] - pc0] = True
        pieces.append({"area": int(pm.sum()), "mask": pm, "patch": patch})

    if not pieces:
        return np.full((1, 1), int(bgc), dtype=int)

    # --- infer k from the smallest solid square piece ---
    k = None
    for p in pieces:
        h, w = map(int, p["mask"].shape)
        if h == w and int(p["mask"].sum()) == int(h * w):
            k = int(h) if k is None else min(int(k), int(h))
    k = int(k or 1)

    def downsample_blocks(mask: np.ndarray) -> np.ndarray | None:
        hh, ww = mask.shape
        if hh % k != 0 or ww % k != 0:
            return None
        out_h, out_w = int(hh // k), int(ww // k)
        out = np.zeros((out_h, out_w), dtype=bool)
        for r in range(out_h):
            for c in range(out_w):
                block = mask[int(r * k) : int((r + 1) * k), int(c * k) : int((c + 1) * k)]
                if bool(block.all()):
                    out[r, c] = True
                elif bool(block.any()):
                    return None
        return out

    def dihedral_variants(
        patch: np.ndarray, mask: np.ndarray
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        base = []
        for kk in range(4):
            p = np.rot90(patch, int(kk))
            m = np.rot90(mask, int(kk))
            base.append((p, m))
            base.append((np.fliplr(p), np.fliplr(m)))
        uniq = []
        seen = set()
        for p, m in base:
            key = m.tobytes()
            if key in seen:
                continue
            seen.add(key)
            uniq.append((p, m))
        return uniq

    # Precompute all usable (piece,variant) candidates with their block masks.
    candidates = []
    for pi, p in enumerate(pieces):
        for vp, vm in dihedral_variants(p["patch"], p["mask"]):
            um = downsample_blocks(vm)
            if um is None:
                continue
            candidates.append({"pi": int(pi), "area": int(p["area"]), "um": um, "vp": vp, "vm": vm})

    used_piece: set[int] = set()
    placements: list[tuple[dict, dict]] = []
    for s in slots:
        need_area = int(s["area"]) * int(k) * int(k)
        found = None
        for cand in candidates:
            if int(cand["pi"]) in used_piece:
                continue
            if int(cand["area"]) != int(need_area):
                continue
            if cand["um"].shape != s["mask"].shape:
                continue
            if np.array_equal(cand["um"], s["mask"]):
                found = cand
                break
        if found is None:
            continue
        used_piece.add(int(found["pi"]))
        placements.append((s, found))

    outH = (int(layout.shape[0]) + 2) * int(k)
    outW = (int(layout.shape[1]) + 2) * int(k)
    out = np.full((outH, outW), int(bgc), dtype=int)

    for s, cand in placements:
        top = (int(s["r0"]) + 1) * int(k)
        left = (int(s["c0"]) + 1) * int(k)
        paste(out, cand["vp"], int(top), int(left), mask=cand["vm"], overwrite=True)

    return out



