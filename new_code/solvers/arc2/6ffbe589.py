from __future__ import annotations
from collections import Counter
import numpy as np
from arc_utils import colors_present, find_components, get_bg


def crop_components_near_largest(
    grid: np.ndarray,
    *,
    bg: int | None = None,
    margin: int = 3,
    connectivity: int = 8,
) -> tuple[np.ndarray, tuple[int, int, int, int], int]:
    """
    Local helper (only used by this solver): crop the cluster near the largest non-bg component.

    Returns:
        (crop, crop_bbox, bg) where crop_bbox is in the original grid coordinates (inclusive).
    """
    if bg is None:
        bg = get_bg(grid)
    bg = int(bg)

    comps = find_components(grid, bg=bg, connectivity=int(connectivity), monochromatic=False, return_format="dict")
    if not comps:
        H, W = grid.shape
        return grid.copy(), (0, 0, int(H - 1), int(W - 1)), int(bg)

    comps = sorted(comps, key=lambda d: int(d["area"]), reverse=True)
    mr0, mc0, mr1, mc1 = map(int, comps[0]["bbox"])

    H, W = grid.shape
    er0 = max(0, int(mr0 - margin))
    ec0 = max(0, int(mc0 - margin))
    er1 = min(int(H - 1), int(mr1 + margin))
    ec1 = min(int(W - 1), int(mc1 + margin))

    keep_mask = np.zeros((H, W), dtype=bool)
    for comp in comps:
        r0, c0, r1, c1 = map(int, comp["bbox"])
        if r1 < er0 or r0 > er1 or c1 < ec0 or c0 > ec1:
            continue
        coords = comp["coords"]
        keep_mask[coords[:, 0], coords[:, 1]] = True

    keep_coords = np.argwhere(keep_mask)
    if keep_coords.size == 0:
        return grid.copy(), (0, 0, int(H - 1), int(W - 1)), int(bg)

    r0, c0 = keep_coords.min(axis=0)
    r1, c1 = keep_coords.max(axis=0)
    r0, c0, r1, c1 = map(int, (r0, c0, r1, c1))

    crop = np.full((int(r1 - r0 + 1), int(c1 - c0 + 1)), int(bg), dtype=grid.dtype)
    sub = keep_mask[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]
    crop[sub] = grid[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1][sub]
    return crop, (int(r0), int(c0), int(r1), int(c1)), int(bg)


def rotate_color_layer(grid: np.ndarray, color: int, rot_k: int, *, bg: int | None = None) -> np.ndarray:
    """
    Local helper (only used by this solver): rotate a single color's mask inside a fixed-size grid.
    """
    if bg is None:
        bg = get_bg(grid)
    bg = int(bg)

    k = int(rot_k) % 4
    if k == 0:
        return grid

    col = int(color)
    mask = (grid == col).astype(int)
    rmask = np.rot90(mask, k)
    if rmask.shape != mask.shape:
        return grid

    out = grid.copy()
    out[out == col] = bg
    out[rmask == 1] = col
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Crop a board cluster and rotate its color layers based on outside cue sprites.

    Input structure:
    - bg = most common color.
    - A main multi-color board cluster (largest foreground region).
    - Small monochrome cue sprites outside the board, one per interior color.

    Rule:
    - Tight-crop around the board cluster.
    - For each color layer inside the crop, find the largest same-color cue outside.
    - Rotate that layer clockwise by (cue pixel count mod 4) × 90°.
    - No outside cue → no rotation.

    Key insight: Cue sprite size mod 4 encodes rotation amount for its matching color layer.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

    # --- STEP 1) Crop the "board" cluster (drop distant sample sprites) ---
    crop, (r0, c0, r1, c1), bg = crop_components_near_largest(g, bg=bg, margin=3, connectivity=8)

    # --- STEP 2) Determine the dominant (frame) color in the crop (not needed for the uniform rule) ---
    cnt = Counter(map(int, crop[crop != int(bg)].flatten()))
    if not cnt:
        return crop
    _main_col = int(cnt.most_common(1)[0][0])

    # --- STEP 3) For each layer, rotate according to the largest outside cue size ---
    colors = colors_present(crop, exclude={int(bg)})
    out = crop.copy()

    outside_grid = g.copy()
    outside_grid[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1] = int(bg)

    for col in colors:
        outside_comps = find_components(
            outside_grid,
            bg=int(bg),
            connectivity=8,
            monochromatic=True,
            colors=[int(col)],
            return_format="dict",
        )
        best_sz = max((int(d["area"]) for d in outside_comps), default=0)
        rot_k = (-int(best_sz)) % 4
        if rot_k:
            out = rotate_color_layer(out, int(col), int(rot_k), bg=int(bg))

    return out

