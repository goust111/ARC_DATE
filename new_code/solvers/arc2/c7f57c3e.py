import numpy as np

from arc_utils import bbox_pts as _bbox_pts, cc8 as _cc8, get_bg as _mode_color


def _pick_anchor_bbox(arr: np.ndarray, bg: int) -> tuple[int, int, int, int] | None:
    """
    Pick a stable anchor bbox inside the sprite array.

    Primary heuristic: use the bbox of color 1 pixels (present across all observed variants/scales).
    Fallback: bbox of all non-background pixels.
    """
    if np.any(arr == 1):
        ys, xs = np.where(arr == 1)
        return int(ys.min()), int(xs.min()), int(ys.max()), int(xs.max())
    if np.any(arr != bg):
        ys, xs = np.where(arr != bg)
        return int(ys.min()), int(xs.min()), int(ys.max()), int(xs.max())
    return None


def _scale_nn(arr: np.ndarray, scale: int) -> np.ndarray:
    if scale <= 1:
        return arr.copy()
    return np.repeat(np.repeat(arr, scale, axis=0), scale, axis=1)


def transform(grid: np.ndarray) -> np.ndarray:
    """
    Swap every sprite to the opposite prototype while preserving scale and position.

    Input structure:
    - bg = most common color. Two distinct small sprite patterns (prototypes) appear at various scales.
    - Each sprite instance is a scaled copy of one of the two prototypes, placed on the grid.

    Rule:
    - Identify the two prototype patterns from the smallest-area sprites.
    - For each sprite, determine which prototype it matches and its integer scale factor.
    - Replace it with the other prototype at the same scale, aligned by an internal anchor region (color-1 pixels).

    Key insight: Every sprite toggles to the alternate prototype; the anchor sub-region inside each sprite pins the alignment so swapped sprites land in the correct position regardless of scale.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = _mode_color(g)

    comps = _cc8(g != bg)
    if len(comps) < 2:
        return g.copy()

    # Extract sprite records.
    sprites = []
    for pts in comps:
        r0, c0, r1, c1 = _bbox_pts(pts)
        arr = g[r0 : r1 + 1, c0 : c1 + 1].copy()
        sprites.append({"pts": pts, "bbox": (r0, c0, r1, c1), "arr": arr})

    # Determine two base prototypes from the smallest bbox area sprites.
    areas = [int((s["bbox"][2] - s["bbox"][0] + 1) * (s["bbox"][3] - s["bbox"][1] + 1)) for s in sprites]
    min_area = min(areas)
    base = [s for s, a in zip(sprites, areas, strict=False) if a == min_area]
    uniq: dict[tuple[int, ...], np.ndarray] = {}
    for s in base:
        key = tuple(s["arr"].ravel().tolist())
        if key not in uniq:
            uniq[key] = s["arr"]
    if len(uniq) < 2:
        # fallback: use two smallest distinct patterns overall
        uniq = {}
        for s in sorted(sprites, key=lambda x: (x["arr"].size, x["bbox"][0], x["bbox"][1])):
            key = tuple(s["arr"].ravel().tolist())
            if key not in uniq:
                uniq[key] = s["arr"]
            if len(uniq) == 2:
                break
    if len(uniq) < 2:
        return g.copy()

    proto_keys = list(uniq.keys())[:2]
    protos = [uniq[proto_keys[0]], uniq[proto_keys[1]]]

    anc0 = _pick_anchor_bbox(protos[0], bg)
    anc1 = _pick_anchor_bbox(protos[1], bg)
    if anc0 is None or anc1 is None:
        return g.copy()
    a0r0, a0c0, a0r1, a0c1 = anc0
    base_anc_h = a0r1 - a0r0 + 1
    base_anc_w = a0c1 - a0c0 + 1

    out = g.copy()

    for s in sprites:
        r0, c0, r1, c1 = s["bbox"]
        arr = s["arr"]
        anc = _pick_anchor_bbox(arr, bg)
        if anc is None:
            continue
        ar0, ac0, ar1, ac1 = anc
        inst_anc_h = ar1 - ar0 + 1
        inst_anc_w = ac1 - ac0 + 1
        # infer integer scale
        scale_h = max(1, inst_anc_h // max(1, base_anc_h))
        scale_w = max(1, inst_anc_w // max(1, base_anc_w))
        scale = int(max(scale_h, scale_w))

        # score which prototype matches this sprite when aligned by core bbox
        best_idx = 0
        best_score = -1
        for idx, p in enumerate(protos):
            sp = _scale_nn(p, scale)
            panc = _pick_anchor_bbox(p, bg)
            if panc is None:
                continue
            pr0, pc0, pr1, pc1 = panc
            pr0 *= scale
            pc0 *= scale
            # align anchor top-left to instance anchor top-left (global coords)
            top = (r0 + ar0) - pr0
            left = (c0 + ac0) - pc0

            # overlap score: count matching non-bg pixels where prototype expects non-bg
            score = 0
            ph, pw = sp.shape
            rr0 = max(0, top)
            cc0 = max(0, left)
            rr1 = min(h, top + ph)
            cc1 = min(w, left + pw)
            if rr0 >= rr1 or cc0 >= cc1:
                continue
            sub_g = g[rr0:rr1, cc0:cc1]
            sub_p = sp[(rr0 - top) : (rr1 - top), (cc0 - left) : (cc1 - left)]
            m = sub_p != bg
            if np.any(m):
                score = int(np.sum(sub_g[m] == sub_p[m]))
            if score > best_score:
                best_score = score
                best_idx = idx

        repl_idx = 1 - best_idx

        # Clear current sprite pixels.
        for rr, cc in s["pts"]:
            out[rr, cc] = bg

        # Place replacement.
        p = protos[repl_idx]
        sp = _scale_nn(p, scale)
        panc = _pick_anchor_bbox(p, bg)
        if panc is None:
            continue
        pr0, pc0, pr1, pc1 = panc
        pr0 *= scale
        pc0 *= scale
        top = (r0 + ar0) - pr0
        left = (c0 + ac0) - pc0

        ph, pw = sp.shape
        rr0 = max(0, top)
        cc0 = max(0, left)
        rr1 = min(h, top + ph)
        cc1 = min(w, left + pw)
        sub_p = sp[(rr0 - top) : (rr1 - top), (cc0 - left) : (cc1 - left)]
        m = sub_p != bg
        out_slice = out[rr0:rr1, cc0:cc1]
        out_slice[m] = sub_p[m]
        out[rr0:rr1, cc0:cc1] = out_slice

    return out
