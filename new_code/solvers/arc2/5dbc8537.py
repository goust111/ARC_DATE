from __future__ import annotations
from collections import Counter
import numpy as np
from arc_utils import bbox_coords, colors_present, components_mask, crop_bbox, find_components, get_bg, paste

def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill rectangular cavities in a template shape by placing source sprites into matching holes.

    Input structure:
    - bg = most common color. A dominant "base" color forms a large frame/template with rectangular gaps.
    - Small colored sprites sit outside the template region.

    Rule:
    - The base color (largest connected component excluding bg) defines the template bbox.
    - Each rectangular hole inside the template matches exactly one outside sprite by size.
    - Place each sprite into its matching cavity; output is the filled template bbox.

    Key insight: Sprite bounding boxes match cavity dimensions exactly — it's a jigsaw-fit puzzle where each piece has a unique slot.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

    # --- pick base color by largest CC size (excluding bg) ---
    best = None  # (max_cc_area, color)
    for col in colors_present(g, exclude={bg}):
        max_area = 0
        for coords in components_mask(g == int(col), connectivity=8):
            max_area = max(int(max_area), int(coords.shape[0]))
        cand = (int(max_area), int(col))
        if best is None or cand > best:
            best = cand

    assert not (best is None)
    base_col = int(best[1])

    # --- destination bbox = bbox of all base-color pixels ---
    base_coords = np.argwhere(g == int(base_col))
    assert not (base_coords.size == 0)
    r0, c0, r1, c1 = bbox_coords(base_coords)
    dst_bbox = (int(r0), int(c0), int(r1), int(c1))

    dst = crop_bbox(g, dst_bbox)
    dh, dw = map(int, dst.shape)
    hole_mask = dst != int(base_col)

    out = np.full((dh, dw), int(base_col), dtype=int)

    # --- source sprites = connected components of non-bg pixels outside dst_bbox ---
    masked = g.copy()
    masked[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1] = int(bg)
    sprite_comps = find_components(masked, bg=bg, connectivity=8, monochromatic=False, return_format="dict")

    sprites: list[dict] = []
    for comp in sprite_comps:
        b = tuple(map(int, comp["bbox"]))
        pr0, pc0, pr1, pc1 = b
        patch = crop_bbox(g, b)
        ph, pw = map(int, patch.shape)
        m = np.zeros((ph, pw), dtype=bool)
        coords = comp["coords"]
        m[coords[:, 0] - pr0, coords[:, 1] - pc0] = True
        sprites.append({"h": int(ph), "w": int(pw), "mask": m, "patch": patch})

    # Larger sprites first; search for a non-overlapping packing.
    sprites.sort(key=lambda s: (-int(s["h"] * s["w"]), -int(s["mask"].sum())))

    # --- candidate placements per sprite: any all-hole rectangle of same bbox ---
    candidates: list[list[tuple[int, int]]] = []
    areas = [int(s["h"] * s["w"]) for s in sprites]
    for sp in sprites:
        h, w = int(sp["h"]), int(sp["w"])
        pos: list[tuple[int, int]] = []
        if h <= dh and w <= dw:
            for rr0 in range(int(dh - h + 1)):
                for cc0 in range(int(dw - w + 1)):
                    if bool(hole_mask[rr0 : rr0 + h, cc0 : cc0 + w].all()):
                        pos.append((int(rr0), int(cc0)))
        candidates.append(pos)

    used = ~hole_mask  # base pixels are treated as already occupied
    best_score = -1
    best_place: list[tuple[int, int] | None] = [None] * len(sprites)
    cur_place: list[tuple[int, int] | None] = [None] * len(sprites)

    # --- DFS with pruning (suffix upper bounds) ---
    suffix = [0] * (len(sprites) + 1)
    for i in range(len(sprites) - 1, -1, -1):
        suffix[i] = int(suffix[i + 1]) + int(areas[i])

    def dfs(i: int, score: int) -> None:
        nonlocal best_score, best_place
        if int(score) + int(suffix[i]) <= int(best_score):
            return
        if i == len(sprites):
            if int(score) > int(best_score):
                best_score = int(score)
                best_place = cur_place.copy()
            return

        h, w = int(sprites[i]["h"]), int(sprites[i]["w"])
        for rr0, cc0 in candidates[i]:
            if bool(used[rr0 : rr0 + h, cc0 : cc0 + w].any()):
                continue
            used[rr0 : rr0 + h, cc0 : cc0 + w] = True
            cur_place[i] = (int(rr0), int(cc0))
            dfs(i + 1, int(score) + int(areas[i]))
            cur_place[i] = None
            used[rr0 : rr0 + h, cc0 : cc0 + w] = False

        # allow skipping (helps when no candidate fits)
        dfs(i + 1, int(score))

    dfs(0, 0)

    # --- paste according to best placement ---
    for i, pos in enumerate(best_place):
        if pos is None:
            continue
        rr0, cc0 = map(int, pos)
        sp = sprites[i]
        paste(out, sp["patch"], rr0, cc0, mask=sp["mask"], overwrite=True)

    return out



