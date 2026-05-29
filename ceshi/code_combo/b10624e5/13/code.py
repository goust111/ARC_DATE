# Combo variant of b10624e5 (index 13)
# combo_spec: D5/3+D8/3
# Name: 对角线反射 + 对角线分割三角形区域

"""
design_analysis:
  variant_goal:
    keep:
      - single 27x27 canvas
      - one source motif containing a 2x2 red anchor
      - three target red anchors whose sizes encode nearest-neighbor scale
      - output-first generation with an oracle check
    change:
      - replace the row/column cross with two full diagonals of one color, creating top/right/bottom/left triangular regions
      - move the source from the old top-left quadrant to the top triangle
      - use one fixed main-diagonal reflection (transpose) for every target copy
      - stamp each target copy only inside its own triangle, so the geometry change is active in the paint step itself
  concrete_behavior_trace:
    example:
      - assume the top source patch has bbox 4x6, its red anchor sits near the lower-left of that bbox, and one green cell is at local offset (0,5)
      - old task:
          - the top-right target would mirror left-right, moving that green cell to (0,0)
          - the bottom-left target would mirror up-down, moving it to (3,5)
      - fused task:
          - every target first transposes the patch, so that green cell moves to (5,0)
          - the transformed patch is then scaled to the target anchor size and aligned to a red square inside a triangle region
          - any transformed pixel landing on a divider diagonal or outside the region mask is rejected
      - visible difference:
          - the source now lives in the top wedge rather than a rectangular quadrant
          - the copied motifs all share the same diagonal flip, and their legal footprint is shaped by the triangle masks
  complexity_calibration:
    preserved_freedom:
      - target anchor sizes still vary independently in 1..4
      - source motifs still use 3..5 non-background colors including red
      - scaling still obeys the exact roundtrip check
    added_constraints:
      - divider color is excluded from the motif palette so diagonal borders stay unambiguous
      - the source motif must cross the hidden old center column, making the D8 geometry visibly incompatible with quadrant reading
      - at least two target regions must differ from an orthogonal-reflection counterfactual, so the D5 change is not silent
      - generator prefers target placements whose copied motif crosses the hidden old center axis of that triangle
    reason:
      - without these filters, the puzzle could look too close to the original cross/quadrant version or to an unreflected copy task
  implicit_assumption_audit:
    - divider detection now assumes both diagonals are monochrome and share one non-background color; generator enforces that and never uses that color inside motifs
    - anchor classification uses the red square's center cell; generator keeps every red square strictly inside one triangle and off both diagonals
    - source extraction assumes the source is the unique non-divider component containing the top-region red anchor; generator keeps other regions input-empty except for their anchor squares
    - alignment by the scaled red bbox minimum corner still works after transpose because the source anchor remains a square block under diagonal reflection
    - target stamping assumes transformed non-background pixels are legal only where the destination region mask is true; generator places anchors only where the full transformed motif fits
  placement_legality:
    - the four region masks are disjoint except for the divider diagonals
    - exhaustive placement search is used for the source patch and every target anchor, and only placements with full in-bounds region-safe motifs are accepted
    - source bbox sides remain at most 6 before scaling and at most 12 after scaling, so there is ample space inside each triangle
  visual_observability:
    - the top source spans the hidden old center column, which makes the new triangular partition obvious from 2-3 examples
    - right/left/bottom targets are placed near the central part of their wedges so the copied motif shape, not just the red square, shows the new region geometry
    - colors are copied verbatim; no invisible ordering or color arithmetic is required
  cleanliness:
    - each triangle contains one anchor and at most one motif component
    - divider diagonals remain untouched, so region boundaries stay visually crisp
  color_reasoning:
    - the task uses direct color copying only
    - color 1 is reserved for the diagonals and never doubled as motif content
  generator_validation:
    - every sample is accepted only if main(input) exactly reconstructs the synthesized output
    - behavior metrics track how many target regions differ from an orthogonal-reflection counterfactual and how many placed motifs cross the hidden old mid axes
"""

import random

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, paste


BG = 4
DIV = 1
RED = 2
GRID_SIZE = 27
MID = GRID_SIZE // 2
SOURCE_ANCHOR_SIZE = 2
REGIONS = ("top", "right", "bottom", "left")
TARGET_REGIONS = ("right", "bottom", "left")
COLOR_POOL = [0, 3, 5, 6, 7, 8, 9]
BLOCK_DIMS = [(2, 2), (2, 3), (3, 2), (3, 3)]
BLOCK_PROB = 0.55
FILL_RATIO_MIN = 0.5
FILL_RATIO_MAX = 0.8
FREEFORM_H_RANGE = (3, 6)
FREEFORM_W_RANGE = (3, 6)
MOTIF_COLORS_MIN = 3
MOTIF_COLORS_MAX = 5
TARGET_SIZE_RANGE = (1, 4)
MAX_RETRIES = 600


def _build_region_masks(n: int) -> dict[str, np.ndarray]:
    rows = np.arange(n)[:, None]
    cols = np.arange(n)[None, :]
    return {
        "top": (rows < cols) & ((rows + cols) < n - 1),
        "right": (rows < cols) & ((rows + cols) > n - 1),
        "bottom": (rows > cols) & ((rows + cols) > n - 1),
        "left": (rows > cols) & ((rows + cols) < n - 1),
    }


REGION_MASKS = _build_region_masks(GRID_SIZE)


def _find_diagonal_divider_color(g: np.ndarray, bg: int) -> int:
    n = g.shape[0]
    main_vals = {int(g[i, i]) for i in range(n)}
    anti_vals = {int(g[i, n - 1 - i]) for i in range(n)}
    assert len(main_vals) == 1 and len(anti_vals) == 1, "both diagonals must be monochrome"
    div_main = next(iter(main_vals))
    div_anti = next(iter(anti_vals))
    assert div_main == div_anti and div_main != bg, "diagonals must share one non-background color"
    return div_main


def _classify_region(r: int, c: int, n: int) -> str:
    assert 0 <= r < n and 0 <= c < n
    assert r != c and r + c != n - 1, "divider cells do not belong to a triangle region"
    if r < c and r + c < n - 1:
        return "top"
    if r < c and r + c > n - 1:
        return "right"
    if r > c and r + c > n - 1:
        return "bottom"
    return "left"


def _resize_nn(img: np.ndarray, new_h: int, new_w: int) -> np.ndarray:
    src_h, src_w = img.shape
    rr = np.clip(np.floor(np.arange(new_h) * src_h / new_h).astype(int), 0, src_h - 1)
    cc = np.clip(np.floor(np.arange(new_w) * src_w / new_w).astype(int), 0, src_w - 1)
    return img[rr[:, None], cc[None, :]]


def _diag_reflect(img: np.ndarray) -> np.ndarray:
    return np.transpose(img).copy()


def _orthogonal_counterfactual(img: np.ndarray, region: str) -> np.ndarray:
    if region == "right":
        return np.fliplr(img).copy()
    if region == "bottom":
        return np.flipud(np.fliplr(img)).copy()
    if region == "left":
        return np.flipud(img).copy()
    return img.copy()


def _same_grid(a: np.ndarray, b: np.ndarray) -> bool:
    return a.shape == b.shape and np.array_equal(a, b)


def _transform_and_scale(patch: np.ndarray, source_size: int, target_size: int) -> np.ndarray:
    patch_t = _diag_reflect(patch)
    ratio = target_size / source_size
    new_h = max(1, int(round(patch_t.shape[0] * ratio)))
    new_w = max(1, int(round(patch_t.shape[1] * ratio)))
    return _resize_nn(patch_t, new_h, new_w)


def _crosses_hidden_axis(rr: np.ndarray, cc: np.ndarray, region: str) -> bool:
    if rr.size == 0:
        return False
    if region in ("top", "bottom"):
        return int(cc.min()) < MID < int(cc.max())
    return int(rr.min()) < MID < int(rr.max())


def _grow_4conn(h: int, w: int, seeds: set[tuple[int, int]], n_fill: int) -> set[tuple[int, int]]:
    filled = set(seeds)
    frontier = set()
    for r, c in seeds:
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in filled:
                frontier.add((nr, nc))
    while len(filled) < n_fill and frontier:
        cell = random.choice(list(frontier))
        filled.add(cell)
        frontier.discard(cell)
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = cell[0] + dr, cell[1] + dc
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in filled:
                frontier.add((nr, nc))
    return filled


def _assign_colors(
    filled: set[tuple[int, int]],
    red_cells: set[tuple[int, int]],
    h: int,
    w: int,
) -> np.ndarray | None:
    rows = {r for r, _ in filled}
    cols = {c for _, c in filled}
    if rows != set(range(h)) or cols != set(range(w)):
        return None
    non_red = [p for p in filled if p not in red_cells]
    random.shuffle(non_red)
    max_colors = min(MOTIF_COLORS_MAX, len(non_red) + 1)
    if max_colors < MOTIF_COLORS_MIN:
        return None
    n_colors = random.randint(MOTIF_COLORS_MIN, max_colors)
    pool = COLOR_POOL[:]
    random.shuffle(pool)
    extras = pool[: n_colors - 1]
    grid = np.full((h, w), BG, dtype=int)
    for r, c in red_cells:
        grid[r, c] = RED
    for i, (r, c) in enumerate(non_red):
        grid[r, c] = extras[i] if i < len(extras) else random.choice(extras)
    return grid


def _source_patch(needs_block: bool) -> tuple[np.ndarray, int, int] | None:
    s = SOURCE_ANCHOR_SIZE
    use_block = needs_block or random.random() < BLOCK_PROB
    if use_block:
        lh, lw = random.choice(BLOCK_DIMS)
        n_cells = lh * lw
        min_fill = max(3, int(np.ceil(FILL_RATIO_MIN * n_cells)))
        max_fill = max(min_fill, int(np.floor(FILL_RATIO_MAX * n_cells)))
        ar, ac = random.randrange(lh), random.randrange(lw)
        filled = _grow_4conn(lh, lw, {(ar, ac)}, random.randint(min_fill, max_fill))
        logical = _assign_colors(filled, {(ar, ac)}, lh, lw)
        if logical is None:
            return None
        patch = np.repeat(np.repeat(logical, s, axis=0), s, axis=1)
        return patch, ar * s, ac * s

    h = random.randint(*FREEFORM_H_RANGE)
    w = random.randint(*FREEFORM_W_RANGE)
    ar, ac = random.randint(0, h - s), random.randint(0, w - s)
    red_cells = {(ar + dr, ac + dc) for dr in range(s) for dc in range(s)}
    min_fill = max(len(red_cells) + 2, int(np.ceil(FILL_RATIO_MIN * h * w)))
    max_fill = int(np.floor(FILL_RATIO_MAX * h * w))
    if max_fill < min_fill:
        return None
    filled = _grow_4conn(h, w, red_cells, random.randint(min_fill, max_fill))
    patch = _assign_colors(filled, red_cells, h, w)
    if patch is None:
        return None
    return patch, ar, ac


def _sample_target_sizes() -> dict[str, int]:
    while True:
        target_sizes = {
            "top": SOURCE_ANCHOR_SIZE,
            "right": random.randint(*TARGET_SIZE_RANGE),
            "bottom": random.randint(*TARGET_SIZE_RANGE),
            "left": random.randint(*TARGET_SIZE_RANGE),
        }
        non_source = [target_sizes[r] for r in TARGET_REGIONS]
        if max(non_source) >= 3 and sum(size >= 2 for size in non_source) >= 2:
            return target_sizes


def _diag_change_count(patch: np.ndarray, target_sizes: dict[str, int]) -> int:
    changed = 0
    source_size = target_sizes["top"]
    for region in TARGET_REGIONS:
        size = target_sizes[region]
        new_patch = _transform_and_scale(patch, source_size, size)
        old_patch = _resize_nn(
            _orthogonal_counterfactual(patch, region),
            max(1, int(round(patch.shape[0] * size / source_size))),
            max(1, int(round(patch.shape[1] * size / source_size))),
        )
        if not _same_grid(new_patch, old_patch):
            changed += 1
    return changed


def _source_positions(patch: np.ndarray, require_axis_cross: bool) -> list[tuple[int, int]]:
    mask = REGION_MASKS["top"]
    h, w = patch.shape
    rr, cc = np.where(patch != BG)
    if rr.size == 0:
        return []
    preferred: list[tuple[int, int]] = []
    fallback: list[tuple[int, int]] = []
    for top in range(GRID_SIZE - h + 1):
        for left in range(GRID_SIZE - w + 1):
            tgt_r = rr + top
            tgt_c = cc + left
            if not np.all(mask[tgt_r, tgt_c]):
                continue
            entry = (int(top), int(left))
            if _crosses_hidden_axis(tgt_r, tgt_c, "top"):
                preferred.append(entry)
            else:
                fallback.append(entry)
    return preferred if preferred else (fallback if not require_axis_cross else [])


def _target_anchor_positions(
    patch_s: np.ndarray,
    region: str,
    size: int,
    require_axis_cross: bool,
) -> list[tuple[int, int]]:
    mask = REGION_MASKS[region]
    rr_fg, cc_fg = np.where(patch_s != BG)
    rr_red, cc_red = np.where(patch_s == RED)
    if rr_fg.size == 0 or rr_red.size == 0:
        return []
    red_r0, red_c0 = int(rr_red.min()), int(cc_red.min())
    preferred: list[tuple[int, int]] = []
    fallback: list[tuple[int, int]] = []
    for ar in range(GRID_SIZE - size + 1):
        for ac in range(GRID_SIZE - size + 1):
            if not np.all(mask[ar : ar + size, ac : ac + size]):
                continue
            place_r0 = ar - red_r0
            place_c0 = ac - red_c0
            tgt_r = rr_fg + place_r0
            tgt_c = cc_fg + place_c0
            if np.any(tgt_r < 0) or np.any(tgt_r >= GRID_SIZE) or np.any(tgt_c < 0) or np.any(tgt_c >= GRID_SIZE):
                continue
            if not np.all(mask[tgt_r, tgt_c]):
                continue
            entry = (int(ar), int(ac))
            if _crosses_hidden_axis(tgt_r, tgt_c, region):
                preferred.append(entry)
            else:
                fallback.append(entry)
    return preferred if preferred else (fallback if not require_axis_cross else [])


def _stamp_region(
    out: np.ndarray,
    patch_s: np.ndarray,
    anchor_r: int,
    anchor_c: int,
    region: str,
) -> None:
    rr_red, cc_red = np.where(patch_s == RED)
    place_r0 = anchor_r - int(rr_red.min())
    place_c0 = anchor_c - int(cc_red.min())
    rr_p, cc_p = np.where(patch_s != BG)
    mask = REGION_MASKS[region]
    for rr0, cc0 in zip(rr_p.tolist(), cc_p.tolist(), strict=False):
        rr_t = place_r0 + rr0
        cc_t = place_c0 + cc0
        if not (0 <= rr_t < GRID_SIZE and 0 <= cc_t < GRID_SIZE):
            continue
        if not mask[rr_t, cc_t]:
            continue
        if out[rr_t, cc_t] == RED:
            continue
        out[rr_t, cc_t] = int(patch_s[rr0, cc0])


def _extract_structure(
    grid: np.ndarray,
) -> tuple[int, int, dict[str, tuple[int, int, int]], tuple[int, int], np.ndarray]:
    g = np.asarray(grid, dtype=int)
    assert g.shape == (GRID_SIZE, GRID_SIZE), "expected 27x27 grid"
    bg = get_bg(g)
    div = _find_diagonal_divider_color(g, bg)

    red_comps = components_mask(g == RED, connectivity=4)
    assert red_comps, "grid must contain red anchors"
    anchors: dict[str, tuple[int, int, int]] = {}
    for pts in red_comps:
        r0, c0, r1, c1 = bbox_coords(pts)
        size_h = r1 - r0 + 1
        size_w = c1 - c0 + 1
        assert size_h == size_w, "red anchors must be square"
        assert len(pts) == size_h * size_w, "red anchors must be filled squares"
        rr = (r0 + r1) // 2
        cc = (c0 + c1) // 2
        region = _classify_region(int(rr), int(cc), GRID_SIZE)
        assert region not in anchors, "each triangle must contain exactly one anchor"
        anchors[region] = (int(r0), int(c0), int(size_h))

    assert set(anchors) == set(REGIONS), "all four triangle regions must contain one red anchor"

    src_r0, src_c0, src_size = anchors["top"]
    mask = (g != bg) & (g != div)
    comps = components_mask(mask, connectivity=4)
    patch_bb = None
    for pts in comps:
        if any(src_r0 <= r < src_r0 + src_size and src_c0 <= c < src_c0 + src_size for r, c in pts):
            patch_bb = bbox_coords(pts)
            break
    assert patch_bb is not None, "top region must contain the source motif component"
    pr0, pc0, pr1, pc1 = patch_bb
    patch = g[pr0 : pr1 + 1, pc0 : pc1 + 1].copy()
    return bg, div, anchors, (int(pr0), int(pc0)), patch


def behavior_metrics(grid: np.ndarray) -> dict[str, int]:
    bg, div, anchors, patch_origin, patch = _extract_structure(grid)
    del bg, div
    target_sizes = {region: anchors[region][2] for region in REGIONS}
    diag_diff = _diag_change_count(patch, target_sizes)

    src_r0, src_c0 = patch_origin
    src_patch_rr, src_patch_cc = np.where(patch != BG)
    source_crosses = _crosses_hidden_axis(src_patch_rr + src_r0, src_patch_cc + src_c0, "top")

    target_crosses = 0
    for region in TARGET_REGIONS:
        patch_s = _transform_and_scale(patch, anchors["top"][2], anchors[region][2])
        rr_red, cc_red = np.where(patch_s == RED)
        place_r0 = anchors[region][0] - int(rr_red.min())
        place_c0 = anchors[region][1] - int(cc_red.min())
        rr_fg, cc_fg = np.where(patch_s != BG)
        if _crosses_hidden_axis(rr_fg + place_r0, cc_fg + place_c0, region):
            target_crosses += 1

    return {
        "diag_counterfactual_diff": int(diag_diff),
        "source_crosses_hidden_axis": int(source_crosses),
        "target_crosses_hidden_axis": int(target_crosses),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Copy the top triangular motif into the other triangular regions using anchor-sized scaling and one diagonal mirror.

    Input structure:
    - The 27x27 board is split by two full diagonals of one non-background color, producing top/right/bottom/left triangles.
    - Each triangle contains exactly one filled red square anchor, with the top anchor embedded inside a larger colored motif.
    - The three non-top triangles show only their red anchors in the input, so the missing motif copies must be inferred.

    Rule:
    - Extract the top-region connected non-background, non-divider component containing the source red anchor.
    - Reflect that patch across the main diagonal, scale it by target_anchor_size / source_anchor_size with nearest-neighbor resize, and align its scaled red block to each target anchor.
    - Stamp every non-background pixel that lands inside the target's triangle, skipping divider cells implicitly via the region mask and never overwriting existing red anchor cells.

    Key insight: the red square still encodes the scale factor, but the positional reasoning is no longer rectangular quadrants;
    all three targets use the same 45-degree reflection and the triangular region masks determine where that reflected copy may exist.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    bg, div, anchors, patch_origin, patch = _extract_structure(g)
    del bg, div
    del patch_origin

    source_size = anchors["top"][2]
    assert source_size > 0, "source anchor must have positive size"

    for region in TARGET_REGIONS:
        tr0, tc0, target_size = anchors[region]
        patch_s = _transform_and_scale(patch, source_size, target_size)
        roundtrip = _resize_nn(patch_s, patch.shape[1], patch.shape[0])
        assert np.array_equal(roundtrip, _diag_reflect(patch)), "scaling must preserve all motif information"
        rr_red, cc_red = np.where(patch_s == RED)
        assert rr_red.size > 0, "scaled patch must contain red anchor pixels"
        _stamp_region(out, patch_s, tr0, tc0, region)

    return out


def _gen_attempt() -> np.ndarray | None:
    target_sizes = _sample_target_sizes()
    needs_block = any(target_sizes[region] % SOURCE_ANCHOR_SIZE != 0 for region in TARGET_REGIONS)
    source = _source_patch(needs_block)
    if source is None:
        return None
    patch, anchor_r_in_patch, anchor_c_in_patch = source

    if _same_grid(patch, _diag_reflect(patch)):
        return None
    if _diag_change_count(patch, target_sizes) < 2:
        return None

    source_positions = _source_positions(patch, require_axis_cross=True)
    if not source_positions:
        return None
    src_top, src_left = random.choice(source_positions)
    source_anchor = (src_top + anchor_r_in_patch, src_left + anchor_c_in_patch)

    scaled = {
        region: _transform_and_scale(patch, SOURCE_ANCHOR_SIZE, target_sizes[region])
        for region in TARGET_REGIONS
    }
    anchors: dict[str, tuple[int, int]] = {"top": source_anchor}
    target_crosses = 0
    for region in TARGET_REGIONS:
        candidates = _target_anchor_positions(
            scaled[region],
            region,
            target_sizes[region],
            require_axis_cross=True,
        )
        if not candidates:
            return None
        anchor = random.choice(candidates)
        anchors[region] = anchor
        rr_red, cc_red = np.where(scaled[region] == RED)
        rr_fg, cc_fg = np.where(scaled[region] != BG)
        place_r0 = anchor[0] - int(rr_red.min())
        place_c0 = anchor[1] - int(cc_red.min())
        if _crosses_hidden_axis(rr_fg + place_r0, cc_fg + place_c0, region):
            target_crosses += 1
    if target_crosses < 2:
        return None

    out = np.full((GRID_SIZE, GRID_SIZE), BG, dtype=int)
    for i in range(GRID_SIZE):
        out[i, i] = DIV
        out[i, GRID_SIZE - 1 - i] = DIV

    paste(out, patch, src_top, src_left, transparent=BG)
    for region in TARGET_REGIONS:
        ar, ac = anchors[region]
        size = target_sizes[region]
        paste(out, np.full((size, size), RED, dtype=int), ar, ac)
    for region in TARGET_REGIONS:
        _stamp_region(out, scaled[region], anchors[region][0], anchors[region][1], region)

    inp = np.full((GRID_SIZE, GRID_SIZE), BG, dtype=int)
    for i in range(GRID_SIZE):
        inp[i, i] = DIV
        inp[i, GRID_SIZE - 1 - i] = DIV
    paste(inp, patch, src_top, src_left, transparent=BG)
    for region in TARGET_REGIONS:
        ar, ac = anchors[region]
        size = target_sizes[region]
        paste(inp, np.full((size, size), RED, dtype=int), ar, ac)

    try:
        if not np.array_equal(main(inp), out):
            return None
        metrics = behavior_metrics(inp)
    except Exception:
        return None
    if metrics["diag_counterfactual_diff"] < 2:
        return None
    if metrics["source_crosses_hidden_axis"] != 1 or metrics["target_crosses_hidden_axis"] < 2:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid b10624e5 combo input")
