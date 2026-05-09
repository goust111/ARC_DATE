# Variant of 1818057f
# Dimension: D7 — 输出网格形状 (grid_shape)
# Change: 输出网格与输入网格尺寸完全相同(10x12到22x22)，原位变换... → 裁剪到'+'区域的外接矩形

from __future__ import annotations

import numpy as np
from arc_utils import bbox, crop_bbox, get_bg


# Design analysis for the D7 variant
# ----------------------------------
# spec:
#   original_behavior:
#     - Input has one foreground color on a dominant background.
#     - Any foreground pixel with foreground on all 4 orthogonal neighbors is a '+' center.
#     - Output recolors each such 5-cell cross to teal and keeps the full input canvas.
#   variant_core:
#     - Keep the same plus detection and recoloring rule.
#     - Change only the output framing: after recoloring, crop to the smallest bbox that
#       contains the union of all recolored 5-cell plus footprints.
#   concrete_trace:
#     - If two plus centers are at (4,5) and (8,9), the original solver recolors both
#       crosses in-place and returns the full HxW grid.
#     - The variant recolors the same cells first, then computes the bbox of those teal
#       footprint cells, e.g. rows 3..9 and cols 4..10, and returns only that subgrid.
#     - Foreground distractors outside that bbox disappear; distractors inside the bbox stay.
#   complexity_calibration:
#     - Crop is based on the transformed plus footprints, not on all foreground pixels.
#       This keeps D7 visually meaningful: the output is smaller, and some non-plus fg can
#       be cut away even though it exists in the input.
#     - Generator keeps multiple pluses common, but still allows a single-plus case.
#   implicit_assumption_audit:
#     - `bg = get_bg(g)` still holds because the generator keeps background dominant.
#     - `fg = first non-bg color` still holds because the input still uses exactly one fg color.
#     - Scanning the original grid for centers is still correct; recoloring must not affect
#       detection order, so centers are collected from `g`, not from the progressively edited output.
#     - The old `return out` relied on fixed-size output. This is the only solver stage that
#       must change for D7: now we must crop by the bbox of the recolored plus-mask.
#   placement_legality:
#     - Plus centers are kept at least 2 cells from the outer border, so the crop is strictly
#       smaller than the input in both dimensions.
#     - Centers keep Manhattan distance >= 4, preserving the original non-overlap behavior.
#     - Extra fg cells are added only when they do not create new plus centers.
#     - At least one fg distractor is forced outside the plus bbox so the crop rule is observable.
#   observability_and_tidiness:
#     - Every generated sample shrinks after solving.
#     - The crop rectangle is not the bbox of all fg: outside distractors vanish, while inside
#       non-plus pixels can remain, making the crop rule inferable from a few examples.


PLUS_DIRS = [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)]
ORTHO_DIRS = PLUS_DIRS[1:]


def _plus_centers(grid: np.ndarray, fg: int) -> list[tuple[int, int]]:
    H, W = grid.shape
    centers: list[tuple[int, int]] = []
    for r in range(1, H - 1):
        for c in range(1, W - 1):
            if int(grid[r, c]) != fg:
                continue
            if all(int(grid[r + dr, c + dc]) == fg for dr, dc in ORTHO_DIRS):
                centers.append((r, c))
    return centers


def _plus_mask_from_centers(shape: tuple[int, int], centers: list[tuple[int, int]]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    for r, c in centers:
        for dr, dc in PLUS_DIRS:
            mask[r + dr, c + dc] = True
    return mask


def _plus_mask(grid: np.ndarray, fg: int) -> np.ndarray:
    return _plus_mask_from_centers(grid.shape, _plus_centers(grid, fg))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Detect foreground '+' motifs, recolor them teal, then keep only their shared span.

    Input structure:
    - The most common color is the background.
    - Exactly one other color forms mixed lines, blobs, and embedded '+' motifs on that background.

    Rule:
    - A '+' center is any foreground cell whose 4 orthogonal neighbors are also foreground.
    - Recolor the center and its 4 orthogonal neighbors to teal (8).
    - After recoloring all such motifs, crop the result to the minimal bounding rectangle covering every recolored '+' cell.

    Key insight: The logical target is still the local 5-cell '+' footprint, but the output frame is no longer the whole canvas — it is the tight crop around the union of all highlighted plus footprints.
    """
    g = np.array(grid, dtype=int)
    out = g.copy()
    bg = int(get_bg(g))

    colors = sorted(set(map(int, g.flatten())) - {bg})
    if not colors:
        return out
    fg = int(colors[0])

    centers = _plus_centers(g, fg)
    if not centers:
        return out

    plus_cells = _plus_mask_from_centers(g.shape, centers)
    out[plus_cells] = 8

    box = bbox(plus_cells)
    assert box is not None
    return crop_bbox(out, box)


def _would_create_plus(grid: np.ndarray, r: int, c: int, fg: int) -> bool:
    """Check whether turning (r,c) into fg would create a plus at that cell or a neighbor."""
    H, W = grid.shape
    candidates = [(r, c), (r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]
    for cr, cc in candidates:
        if cr < 1 or cr >= H - 1 or cc < 1 or cc >= W - 1:
            continue
        good = True
        for dr, dc in PLUS_DIRS:
            nr, nc = cr + dr, cc + dc
            val = fg if (nr, nc) == (r, c) else int(grid[nr, nc])
            if val != fg:
                good = False
                break
        if good:
            return True
    return False


def _sample_centers(H: int, W: int, rng: np.random.Generator) -> list[tuple[int, int]]:
    max_k = min(9, max(1, (H - 4) * (W - 4) // 12))
    K = int(rng.integers(1, max_k + 1))
    centers: list[tuple[int, int]] = []
    for _ in range(K * 50):
        if len(centers) >= K:
            break
        cr = int(rng.integers(2, H - 2))
        cc = int(rng.integers(2, W - 2))
        if all(abs(cr - er) + abs(cc - ec) >= 4 for er, ec in centers):
            centers.append((cr, cc))
    return centers


def _add_arm_extensions(grid: np.ndarray, centers: list[tuple[int, int]], fg: int, rng: np.random.Generator) -> None:
    for cr, cc in centers:
        n_ext = int(rng.integers(1, 3))
        for idx in rng.choice(len(ORTHO_DIRS), size=n_ext, replace=False):
            dr, dc = ORTHO_DIRS[int(idx)]
            tip_r = cr + dr
            tip_c = cc + dc
            ext_len = int(rng.integers(1, 4))
            for step in range(1, ext_len + 1):
                nr = tip_r + dr * step
                nc = tip_c + dc * step
                if nr < 0 or nr >= grid.shape[0] or nc < 0 or nc >= grid.shape[1]:
                    break
                grid[nr, nc] = fg


def _try_place_cluster(
    grid: np.ndarray,
    cells: list[tuple[int, int]],
    bg: int,
    fg: int,
    forbid_mask: np.ndarray | None = None,
) -> bool:
    placed: list[tuple[int, int]] = []
    for r, c in cells:
        if not (0 <= r < grid.shape[0] and 0 <= c < grid.shape[1]):
            for rr, cc in placed:
                grid[rr, cc] = bg
            return False
        if forbid_mask is not None and bool(forbid_mask[r, c]):
            for rr, cc in placed:
                grid[rr, cc] = bg
            return False
        if int(grid[r, c]) == fg:
            continue
        if _would_create_plus(grid, r, c, fg):
            for rr, cc in placed:
                grid[rr, cc] = bg
            return False
        grid[r, c] = fg
        placed.append((r, c))
    return bool(placed)


def _place_outside_cluster(
    grid: np.ndarray,
    plus_box: tuple[int, int, int, int],
    bg: int,
    fg: int,
    rng: np.random.Generator,
) -> bool:
    r0, c0, r1, c1 = plus_box
    H, W = grid.shape
    outside = [(r, c) for r in range(H) for c in range(W) if not (r0 <= r <= r1 and c0 <= c <= c1)]
    rng.shuffle(outside)

    cluster_dirs = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    for ar, ac in outside:
        shapes = [
            [(ar, ac)],
            [(ar, ac), (ar, ac + 1)],
            [(ar, ac), (ar + 1, ac)],
        ]
        dr, dc = cluster_dirs[int(rng.integers(len(cluster_dirs)))]
        shapes.append([(ar + step * dr, ac + step * dc) for step in range(3)])
        rng.shuffle(shapes)
        for cells in shapes:
            if _try_place_cluster(grid, cells, bg, fg):
                return True
    return False


def _sprinkle_distractors(
    grid: np.ndarray,
    bg: int,
    fg: int,
    rng: np.random.Generator,
    target_density: float,
) -> None:
    total = grid.shape[0] * grid.shape[1]
    target_fg = int(total * target_density)
    current_fg = int(np.sum(grid != bg))

    bg_positions = list(zip(*np.where(grid == bg)))
    rng.shuffle(bg_positions)
    for r, c in bg_positions:
        if current_fg >= target_fg:
            break
        if not _would_create_plus(grid, int(r), int(c), fg):
            grid[r, c] = fg
            current_fg += 1


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()
    H = int(rng.integers(10, 23))
    W = int(rng.integers(10, 23))
    bg, fg = 2, 4

    grid = np.full((H, W), bg, dtype=int)
    centers = _sample_centers(H, W, rng)
    if not centers:
        return None

    for cr, cc in centers:
        for dr, dc in PLUS_DIRS:
            grid[cr + dr, cc + dc] = fg

    plus_cells = _plus_mask_from_centers(grid.shape, centers)
    plus_box = bbox(plus_cells)
    if plus_box is None:
        return None
    r0, c0, r1, c1 = plus_box

    # Force visible cropping on all four sides.
    if not (r0 >= 1 and c0 >= 1 and r1 <= H - 2 and c1 <= W - 2):
        return None

    _add_arm_extensions(grid, centers, fg, rng)

    if not _place_outside_cluster(grid, plus_box, bg, fg, rng):
        return None

    _sprinkle_distractors(grid, bg, fg, rng, target_density=float(rng.uniform(0.26, 0.41)))

    if int(np.sum(grid != bg)) >= (H * W) // 2:
        return None

    expected_centers = _plus_centers(grid, fg)
    if sorted(expected_centers) != sorted(centers):
        return None

    actual_plus_mask = _plus_mask(grid, fg)
    actual_box = bbox(actual_plus_mask)
    if actual_box is None or actual_box != plus_box:
        return None

    # Reject degenerate cases where every foreground pixel survives the crop.
    outside_fg = int(np.sum((grid != bg) & (~actual_plus_mask)))
    if outside_fg <= 0:
        return None
    rr0, cc0, rr1, cc1 = actual_box
    outside_crop_fg = int(np.sum(grid[:rr0, :] != bg))
    outside_crop_fg += int(np.sum(grid[rr1 + 1 :, :] != bg))
    outside_crop_fg += int(np.sum(grid[rr0 : rr1 + 1, :cc0] != bg))
    outside_crop_fg += int(np.sum(grid[rr0 : rr1 + 1, cc1 + 1 :] != bg))
    if outside_crop_fg <= 0:
        return None

    out = main(grid)
    if out.shape == grid.shape:
        return None
    if out.shape != (rr1 - rr0 + 1, cc1 - cc0 + 1):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input failed after 400 attempts")
