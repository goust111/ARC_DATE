# Combo variant of b10624e5 (index 16)
# combo_spec: D2/0+D11/3
# Name: 双重源合并 + 每象限独立源图案
#
# Design analysis for the combo variant
# -------------------------------------
# spec delta:
#   - Keep the 27x27 grid, yellow background 4, and the blue divider cross at row/col 13.
#   - Keep one red square anchor per quadrant, but every quadrant now already contains a local
#     source component attached to that anchor.
#   - The four local sources are not equivalent:
#       * TL and TR are shape sources: their non-red support uses one dummy color only.
#       * BL and BR are color sources: their support carries the final palette.
#   - Pairing is diagonal, not rowwise:
#       * TL shape + BR color
#       * TR shape + BL color
#   - Within each diagonal pair, both local patches are block-upscaled views of the same logical
#     bbox and the same logical red-anchor cell, but each quadrant may use its own anchor size.
#   - Output keeps all local sources from the input and adds one fused lobe per quadrant on the
#     opposite side of the same anchor. The fused lobe is the pair's color template masked by the
#     pair's shape template, then reflected across both axes into the diagonal partner.
#
# a) Concrete behavior and trace:
#    Original task:
#      TL alone already contained both shape and colors, so the solver extracted one pixel patch
#      and copied scaled/reflected versions into the other three quadrants.
#    This combo:
#      Each quadrant contains its own patch. Example with logical 2x3 cells:
#        TL shape logical = [[8, 4, 4], [8, 8, 2]] where 4 means bg holes after masking.
#        BR color logical = [[7, 6, 5], [7, 3, 2]].
#      The solver first normalizes TL and BR by their own anchor sizes back to logical cells,
#      merges them into [[7, 4, 4], [7, 3, 2]], then re-upscales to TL size and BR size and
#      stamps the doubly flipped copy near both TL and BR anchors. TR and BL do the same with
#      an independent second logical motif. The changed stage is therefore:
#        one-source extract/copy  ->  four extracts, two diagonal pairings, two source merges.
#
# b) Complexity calibration:
#    - The scene is visibly different from the original: all four quadrants are populated already,
#      and outputs gain four extra fused lobes instead of filling previously empty quadrants.
#    - The role split is low-branch and reusable across examples: top quadrants always provide
#      shape, bottom quadrants always provide color, and diagonal pairing is fixed.
#    - Logical motif sizes stay in {2x2, 2x3, 3x2}; this is enough variety to avoid triviality
#      while keeping placement and human inspection manageable.
#
# c) Hidden-assumption audit against the original solver:
#    - "Only TL contains a source component" breaks immediately; solver must read all quadrants.
#    - "Pixel colors in the source patch are final colors" is false for TL/TR; top patches carry
#      shape only, so their non-red color id is intentionally semantically meaningless.
#    - "Source scaling denominator is always TL anchor size" is false; each extracted patch must be
#      normalized by its own anchor size before pairwise merging.
#    - "Reflection is determined only by destination quadrant" is no longer sufficient. Imported
#      lobes come specifically from diagonal partners, so the relevant geometric relation is the
#      pair relation, which is always a two-axis flip.
#    - New solver assumption: each local source is a uniform block-upscaling of logical cells.
#      The generator enforces this by constructing every patch from logical grids.
#
# d) Placement legality:
#    - Local patches always use the bottom-right logical cell as the red anchor cell, so the local
#      source occupies the NW side of its anchor.
#    - Imported fused lobes are always diagonal flips of that logical patch, so they occupy the SE
#      side. The two lobes therefore only meet on the red anchor block.
#    - Anchor positions are sampled from the intersection that keeps both lobes fully inside the
#      13x13 quadrant. This avoids overlaps, divider contact, and out-of-bounds paste failures.
#
# e) Visual observability:
#    - Shape-source quadrants are immediately recognizable because they are single-color masks.
#    - Color-source quadrants are immediately recognizable because they are multi-color templates.
#    - The output fused lobe in TL clearly uses BR colors but TL holes, while TR uses BL colors but
#      TR holes. A human can infer both the role split and the diagonal pairing from a few examples.
#
# f) Visual cleanliness:
#    - Each quadrant still revolves around one anchor and at most two lobes.
#    - No legends, counters, or auxiliary markers are introduced.
#    - Motif sizes are capped so the NW local lobe and SE fused lobe remain separated and legible.
#
# g) Change inferability:
#    - No hidden ordering uses raw color ids. Colors are copied positionally from visible templates.
#    - The number of support colors is small, and the masked-out cells create direct evidence that
#      both the top shape patch and the diagonal bottom color patch are jointly required.

import os
import random
import sys

import numpy as np

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../.."))
_SOLVERS = os.path.join(_ROOT, "solvers")
if _SOLVERS not in sys.path:
    sys.path.append(_SOLVERS)

from arc_utils import bbox_coords, components_mask, get_bg, paste


BG = 4
DIV = 1
RED = 2
SHAPE_COLOR = 8
GRID_SIZE = 27
MID = 13
MAX_RETRIES = 500
TARGET_SIZE_RANGE = (1, 4)
COLOR_POOL = [0, 3, 5, 6, 7, 9]
LOGICAL_DIMS = [(2, 2), (2, 3), (3, 2)]
SHAPE_TOP_KEYS = ((0, 0), (0, 1))
COLOR_KEY_FOR_SHAPE = {
    (0, 0): (1, 1),
    (0, 1): (1, 0),
}
QUADS = {
    (0, 0): (0, 12, 0, 12),
    (0, 1): (0, 12, 14, 26),
    (1, 0): (14, 26, 0, 12),
    (1, 1): (14, 26, 14, 26),
}


def _find_dividers(g: np.ndarray, bg: int) -> tuple[int, int]:
    """Return the unique full-row/full-col divider crossing."""
    h, w = g.shape
    full_rows = [(r, int(g[r, 0])) for r in range(h) if np.all(g[r, :] == g[r, 0]) and int(g[r, 0]) != bg]
    full_cols = [(c, int(g[0, c])) for c in range(w) if np.all(g[:, c] == g[0, c]) and int(g[0, c]) != bg]
    matches = []
    for r, vr in full_rows:
        for c, vc in full_cols:
            if vr == vc:
                matches.append((int(r), int(c)))
    assert len(matches) == 1, "grid must have exactly one matching full-row/full-col divider"
    return matches[0]


def _quad_bounds(shape: tuple[int, int], mid_r: int, mid_c: int, key: tuple[int, int]) -> tuple[int, int, int, int]:
    h, w = shape
    qb, qr = key
    r0 = 0 if qb == 0 else mid_r + 1
    r1 = mid_r - 1 if qb == 0 else h - 1
    c0 = 0 if qr == 0 else mid_c + 1
    c1 = mid_c - 1 if qr == 0 else w - 1
    return int(r0), int(r1), int(c0), int(c1)


def _upscale_logical(logical: np.ndarray, block: int) -> np.ndarray:
    return np.repeat(np.repeat(logical, block, axis=0), block, axis=1)


def _diag_reflect(img: np.ndarray) -> np.ndarray:
    return np.flipud(np.fliplr(img))


def _red_signature(patch: np.ndarray) -> tuple[int, int, int, int]:
    rr, cc = np.where(patch == RED)
    assert rr.size > 0, "patch must contain red anchor pixels"
    return int(rr.min()), int(cc.min()), int(rr.max() - rr.min() + 1), int(cc.max() - cc.min() + 1)


def _extract_patch(
    g: np.ndarray,
    bg: int,
    mid_r: int,
    mid_c: int,
    key: tuple[int, int],
) -> np.ndarray:
    q_r0, q_r1, q_c0, q_c1 = _quad_bounds(g.shape, mid_r, mid_c, key)
    quad = g[q_r0 : q_r1 + 1, q_c0 : q_c1 + 1]
    comps = components_mask(quad != bg, connectivity=4)
    matches = [pts for pts in comps if any(quad[r, c] == RED for r, c in pts)]
    assert len(matches) == 1, "each quadrant must contain exactly one non-bg component with red"
    r0, c0, r1, c1 = bbox_coords(matches[0])
    return quad[r0 : r1 + 1, c0 : c1 + 1].copy()


def _logical_from_patch(patch: np.ndarray, block: int) -> np.ndarray:
    h, w = patch.shape
    assert h % block == 0 and w % block == 0, "patch dimensions must be multiples of anchor size"
    logical_h = h // block
    logical_w = w // block
    view = patch.reshape(logical_h, block, logical_w, block)
    logical = view[:, 0, :, 0].copy()
    assert np.all(view == logical[:, None, :, None]), "each logical cell must be a uniform block"
    return logical


def _merge_sources(shape_logical: np.ndarray, color_logical: np.ndarray, bg: int) -> np.ndarray:
    assert shape_logical.shape == color_logical.shape, "paired sources must share one logical bbox"
    assert _red_signature(shape_logical) == _red_signature(color_logical), "paired source anchors must align"
    merged = color_logical.copy()
    merged[shape_logical == bg] = bg
    return merged


def _stamp(
    out: np.ndarray,
    patch_s: np.ndarray,
    anchor_r: int,
    anchor_c: int,
    mid_r: int,
    mid_c: int,
    bg: int,
) -> None:
    rr_red, cc_red = np.where(patch_s == RED)
    assert rr_red.size > 0, "scaled patch must contain red anchor pixels"
    place_r0 = anchor_r - int(rr_red.min())
    place_c0 = anchor_c - int(cc_red.min())

    rr_p, cc_p = np.where(patch_s != bg)
    for rr0, cc0 in zip(rr_p.tolist(), cc_p.tolist(), strict=False):
        rr_t = place_r0 + rr0
        cc_t = place_c0 + cc0
        if not (0 <= rr_t < out.shape[0] and 0 <= cc_t < out.shape[1]):
            continue
        if rr_t == mid_r or cc_t == mid_c:
            continue
        if out[rr_t, cc_t] == RED:
            continue
        out[rr_t, cc_t] = int(patch_s[rr0, cc0])


def _anchors_by_quadrant(g: np.ndarray, mid_r: int, mid_c: int) -> dict[tuple[int, int], tuple[int, int, int]]:
    red = components_mask(g == RED, connectivity=4)
    assert red, "grid must contain red anchor blocks"

    anchors: dict[tuple[int, int], tuple[int, int, int]] = {}
    for pts in red:
        r0, c0, r1, c1 = bbox_coords(pts)
        rr = (r0 + r1) // 2
        cc = (c0 + c1) // 2
        qb = 0 if rr < mid_r else 1
        qr = 0 if cc < mid_c else 1
        size_h = r1 - r0 + 1
        size_w = c1 - c0 + 1
        assert size_h == size_w, "anchors must be square"
        key = (qb, qr)
        assert key not in anchors, "each quadrant must contain exactly one red anchor block"
        anchors[key] = (int(r0), int(c0), int(size_h))

    assert len(anchors) == 4, "all four quadrants must contain anchors"
    return anchors


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fuse diagonal shape/color sources after anchor-size normalization, then add the merged lobe back to both members.

    Input structure:
    - A single non-background full row and full column split the 27x27 grid into four quadrants.
    - Every quadrant contains one red square anchor block and one local anchor-containing source component.
    - The two top quadrants are shape sources: their local components use one dummy support color plus red.
    - The two bottom quadrants are color sources: their local components use several final colors plus red.
    - Diagonal quadrants form the two fusion pairs: TL with BR, and TR with BL.

    Rule:
    - Extract the anchor-containing component from each quadrant and use that quadrant's anchor size as the block size
      that converts the pixel patch back into a logical cell grid.
    - For each diagonal pair, take the top patch as a shape mask and the bottom patch as a color template. Keep the
      bottom colors only where the top mask is non-background, producing one merged logical motif for that pair.
    - For each member of the pair, upscale the merged logical motif to that member's anchor size, reflect it across
      both axes, align its red block to the destination anchor, and stamp the non-background pixels.

    Key insight:
    - No single quadrant is self-sufficient anymore. The solver must identify four local sources, separate shape from
      color by quadrant role, normalize each source by its own anchor size, then combine diagonal partners before the
      final reflected copy step.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    bg = get_bg(g)
    mid_r, mid_c = _find_dividers(g, bg)
    anchors = _anchors_by_quadrant(g, mid_r, mid_c)

    logicals: dict[tuple[int, int], np.ndarray] = {}
    for key, (_, _, size) in anchors.items():
        patch = _extract_patch(g, bg, mid_r, mid_c, key)
        logicals[key] = _logical_from_patch(patch, size)

    for shape_key in SHAPE_TOP_KEYS:
        color_key = COLOR_KEY_FOR_SHAPE[shape_key]
        merged = _merge_sources(logicals[shape_key], logicals[color_key], bg)
        merged_reflected = _diag_reflect(merged)
        for dst_key in (shape_key, color_key):
            tr0, tc0, tL = anchors[dst_key]
            patch_s = _upscale_logical(merged_reflected, tL)
            _stamp(out, patch_s, tr0, tc0, mid_r, mid_c, bg)

    return out


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


def _max_anchor_size(lh: int, lw: int) -> int:
    limit_h = 13 // (2 * lh - 1)
    limit_w = 13 // (2 * lw - 1)
    return min(TARGET_SIZE_RANGE[1], limit_h, limit_w)


def _build_shape_logical(lh: int, lw: int) -> np.ndarray | None:
    anchor = (lh - 1, lw - 1)
    total = lh * lw
    min_fill = max(3, int(np.ceil(0.55 * total)))
    max_fill = total - 1
    if max_fill < min_fill:
        return None
    for _ in range(200):
        filled = _grow_4conn(lh, lw, {anchor}, random.randint(min_fill, max_fill))
        if len(filled) == total:
            continue
        rows = {r for r, _ in filled}
        cols = {c for _, c in filled}
        if rows != set(range(lh)) or cols != set(range(lw)):
            continue
        logical = np.full((lh, lw), BG, dtype=int)
        for r, c in filled:
            logical[r, c] = SHAPE_COLOR
        logical[anchor] = RED
        return logical
    return None


def _build_color_logical(lh: int, lw: int) -> np.ndarray | None:
    anchor = (lh - 1, lw - 1)
    for _ in range(200):
        n_extra = random.randint(2, min(4, len(COLOR_POOL)))
        extras = random.sample(COLOR_POOL, n_extra)
        logical = np.empty((lh, lw), dtype=int)
        for r in range(lh):
            for c in range(lw):
                logical[r, c] = random.choice(extras)
        logical[anchor] = RED
        if len({int(logical[r, c]) for r in range(lh) for c in range(lw) if (r, c) != anchor}) < 2:
            continue
        return logical
    return None


def _logical_signature(arr: np.ndarray) -> tuple[int, ...]:
    return tuple(int(x) for x in arr.flatten())


def _choose_anchor(key: tuple[int, int], size: int, logical_shape: tuple[int, int]) -> tuple[int, int] | None:
    q_r0, q_r1, q_c0, q_c1 = QUADS[key]
    lh, lw = logical_shape
    patch_h = lh * size
    patch_w = lw * size
    r_lo = q_r0 + (patch_h - size)
    r_hi = q_r1 - patch_h + 1
    c_lo = q_c0 + (patch_w - size)
    c_hi = q_c1 - patch_w + 1
    if r_lo > r_hi or c_lo > c_hi:
        return None
    return random.randint(r_lo, r_hi), random.randint(c_lo, c_hi)


def _place_local_patch(out: np.ndarray, patch: np.ndarray, anchor_r: int, anchor_c: int) -> None:
    rr_red, cc_red = np.where(patch == RED)
    assert rr_red.size > 0
    place_r0 = anchor_r - int(rr_red.min())
    place_c0 = anchor_c - int(cc_red.min())
    paste(out, patch, place_r0, place_c0)


def _no_mask_output(inp: np.ndarray) -> np.ndarray:
    g = np.asarray(inp, dtype=int)
    out = g.copy()
    bg = get_bg(g)
    mid_r, mid_c = _find_dividers(g, bg)
    anchors = _anchors_by_quadrant(g, mid_r, mid_c)
    logicals = {
        key: _logical_from_patch(_extract_patch(g, bg, mid_r, mid_c, key), anchors[key][2])
        for key in anchors
    }
    for shape_key in SHAPE_TOP_KEYS:
        color_key = COLOR_KEY_FOR_SHAPE[shape_key]
        merged_reflected = _diag_reflect(logicals[color_key])
        for dst_key in (shape_key, color_key):
            tr0, tc0, tL = anchors[dst_key]
            patch_s = _upscale_logical(merged_reflected, tL)
            _stamp(out, patch_s, tr0, tc0, mid_r, mid_c, bg)
    return out


def _wrong_pair_output(inp: np.ndarray) -> np.ndarray:
    g = np.asarray(inp, dtype=int)
    out = g.copy()
    bg = get_bg(g)
    mid_r, mid_c = _find_dividers(g, bg)
    anchors = _anchors_by_quadrant(g, mid_r, mid_c)
    logicals = {
        key: _logical_from_patch(_extract_patch(g, bg, mid_r, mid_c, key), anchors[key][2])
        for key in anchors
    }
    wrong_color_for_shape = {
        (0, 0): (1, 0),
        (0, 1): (1, 1),
    }
    for shape_key in SHAPE_TOP_KEYS:
        color_key = wrong_color_for_shape[shape_key]
        merged = _merge_sources(logicals[shape_key], logicals[color_key], bg)
        merged_reflected = _diag_reflect(merged)
        for dst_key in (shape_key, COLOR_KEY_FOR_SHAPE[shape_key]):
            tr0, tc0, tL = anchors[dst_key]
            patch_s = _upscale_logical(merged_reflected, tL)
            _stamp(out, patch_s, tr0, tc0, mid_r, mid_c, bg)
    return out


def behavior_gap(inp: np.ndarray) -> int:
    actual = main(inp)
    no_mask_gap = int(np.count_nonzero(actual != _no_mask_output(inp)))
    try:
        wrong_pair_gap = int(np.count_nonzero(actual != _wrong_pair_output(inp)))
    except AssertionError:
        wrong_pair_gap = GRID_SIZE * GRID_SIZE
    return min(no_mask_gap, wrong_pair_gap)


def _gen_attempt() -> np.ndarray | None:
    pair_specs: dict[tuple[int, int], dict[str, object]] = {}
    merged_signatures: set[tuple[int, ...]] = set()

    for shape_key in SHAPE_TOP_KEYS:
        color_key = COLOR_KEY_FOR_SHAPE[shape_key]
        dims = random.choice(LOGICAL_DIMS)
        max_size = _max_anchor_size(*dims)
        if max_size < 1:
            return None

        shape_logical = _build_shape_logical(*dims)
        color_logical = _build_color_logical(*dims)
        if shape_logical is None or color_logical is None:
            return None

        merged = _merge_sources(shape_logical, color_logical, BG)
        if np.count_nonzero(merged == BG) == 0:
            return None
        if np.array_equal(merged, _diag_reflect(merged)):
            return None
        sig = _logical_signature(merged)
        if sig in merged_signatures:
            return None
        merged_signatures.add(sig)

        top_size = random.randint(1, max_size)
        bottom_size = random.randint(1, max_size)
        pair_specs[shape_key] = {
            "color_key": color_key,
            "dims": dims,
            "shape_logical": shape_logical,
            "color_logical": color_logical,
            "top_size": top_size,
            "bottom_size": bottom_size,
        }

    if all(int(spec["top_size"]) == int(spec["bottom_size"]) for spec in pair_specs.values()):
        return None

    anchors: dict[tuple[int, int], tuple[int, int]] = {}
    local_patches: dict[tuple[int, int], np.ndarray] = {}

    for shape_key in SHAPE_TOP_KEYS:
        spec = pair_specs[shape_key]
        color_key = spec["color_key"]  # type: ignore[assignment]
        dims = spec["dims"]  # type: ignore[assignment]
        top_size = int(spec["top_size"])
        bottom_size = int(spec["bottom_size"])
        shape_logical = spec["shape_logical"]  # type: ignore[assignment]
        color_logical = spec["color_logical"]  # type: ignore[assignment]

        local_patches[shape_key] = _upscale_logical(shape_logical, top_size)
        local_patches[color_key] = _upscale_logical(color_logical, bottom_size)  # type: ignore[index]

        anchor_top = _choose_anchor(shape_key, top_size, dims)
        anchor_bottom = _choose_anchor(color_key, bottom_size, dims)  # type: ignore[arg-type]
        if anchor_top is None or anchor_bottom is None:
            return None
        anchors[shape_key] = anchor_top
        anchors[color_key] = anchor_bottom  # type: ignore[index]

    inp = np.full((GRID_SIZE, GRID_SIZE), BG, dtype=int)
    inp[MID, :] = DIV
    inp[:, MID] = DIV

    for key, patch in local_patches.items():
        _place_local_patch(inp, patch, *anchors[key])

    try:
        out = main(inp)
    except Exception:
        return None

    if np.array_equal(out, inp):
        return None
    if behavior_gap(inp) <= 0:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid combo b10624e5 input")
