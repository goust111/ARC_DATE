# Combo variant of b10624e5 (index 46)
# combo_spec: D8/2+D13/1
# Name: 同心区域 + 图案环绕锚块

import random

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, paste


# Design analysis for combo D8/2 + D13/1
# ---------------------------------------
# a) Concrete behavior and trace:
#    - Original task:
#      A cross divider creates four 13x13 quadrants. The source motif sits in the
#      top-left quadrant, touches the red anchor, and is transferred to the other
#      three quadrants by scale + quadrant reflection.
#    - Fused task:
#      The cross disappears. Instead there is one hollow square frame in the middle
#      of the 27x27 grid, so the layout is "inner source region + outer ring".
#      The source now lives inside the frame, and the outer ring owns four target
#      anchors in the four corner sectors around that center.
#      The source motif is a 6x6 template whose central 2x2 red anchor is wrapped
#      by four colored side strips with one-cell background gaps. Because the strips
#      are disconnected from the anchor, the solver must recover the whole non-bg
#      bbox inside the center, not a red-connected component.
#      Example trace:
#      * center patch side colors (top,right,bottom,left) = (8,6,3,9)
#      * NE target size = 3, so the 6x6 surround template becomes 9x9 by NN resize
#      * NE lies to the right of center, so the surround template is mirrored LR:
#        side order becomes (8,9,3,6)
#      * SW lies below center, so it is mirrored UD:
#        side order becomes (3,6,8,9)
#      * SE lies diagonally across both axes, so it is mirrored both ways:
#        side order becomes (3,9,8,6)
#      This keeps the old reflection family visible while moving the whole copying
#      process into a concentric inner→outer geometry.
# b) Complexity calibration:
#    - The central square frame makes the D8 change obvious at a glance; the puzzle
#      no longer resembles four quadrants.
#    - The D13 change is also obvious because the source motif visibly surrounds the
#      anchor with four separated side pieces instead of touching one side only.
#    - To keep the task learnable, target anchor sizes are restricted to {2,3}; a
#      6x6 surround template then scales only to 6x6 or 9x9, which fits cleanly in
#      the four 9x9 outer corner regions and still shows nontrivial resizing.
#    - Size patterns are chosen from a small set with repeated values, so examples
#      provide both scale variety and same-scale comparisons across different flips.
# c) Hidden assumptions from the original solver that break:
#    - `_find_dividers` cannot look for one full row and one full column any more;
#      it must find one hollow square frame of a single non-bg color.
#    - Anchor classification cannot depend on TL/TR/BL/BR quadrants split by a cross;
#      it now depends on whether a red square lies inside the frame or in one of the
#      four outer corner sectors around that frame.
#    - Source extraction cannot use "the non-bg component containing red" because the
#      surround strips are deliberately disconnected from the anchor. The full center
#      bbox is the transferable object.
#    - The old roundtrip-resize assertion is invalid here: the 6->9 NN scaling of a
#      surround template with gaps is intentionally aliased, so lossless roundtrip is
#      no longer a valid invariant.
# d) Placement legality:
#    - The frame bbox is fixed to rows/cols 9..17, so the center interior is 7x7 and
#      every outer corner sector is 9x9.
#    - The source template is fixed to 6x6, which always fits in the 7x7 center.
#    - The only target scales are 1x and 1.5x, producing 6x6 and 9x9 templates, both
#      of which fit fully inside a 9x9 corner sector.
#    - Anchor positions are sampled only from ranges where the fully transformed patch
#      fits in its assigned corner sector, so there is no clipping, no overlap, and
#      no ambiguity about which target belongs to which outer location.
# e) Visual observability:
#    - A human immediately sees one colored frame in the middle, one complete motif in
#      the center, and four outer red anchors with no colored support in the input.
#    - The four side colors of the source template are all distinct, so LR and UD
#      reflections visibly swap different colors.
#    - Because outer targets occupy all four corner sectors, the set of transforms
#      {identity, LR, UD, both} is present in a single sample.
# f) Visual cleanliness:
#    - The center source is isolated inside the frame; the four targets are isolated in
#      four disjoint 9x9 corner sectors. No sector contains more than one motif.
#    - The frame color is excluded from motif colors, so the frame is never confused
#      with copied pattern pixels.
# g) Inference of change:
#    - No color arithmetic or hidden ordering is used. Colors only label visible sides
#      of the surround template. The rule is inferable from geometry: inner source,
#      outer targets, scale from anchor size, and reflections from outer corner sector.
# 9) New solver assumptions and generator guarantees:
#    - `main()` assumes exactly one hollow square frame of a single non-bg color.
#      `generate_input()` creates exactly one such frame and never reuses that color.
#    - `main()` assumes exactly five red square anchors: one inside the frame and one
#      in each outer corner sector. The generator always constructs that layout.
#    - `main()` assumes the center source template surrounds the red anchor from all
#      four sides and that its outer side colors are not mirror-symmetric. The
#      generator uses a fixed 6x6 surround construction with four distinct side colors.
#    - `main()` aligns targets by the top-left of the scaled red bbox. The generator
#      keeps the red anchor as a solid square in every scaled template, so this
#      alignment rule is stable across all emitted samples.


BG = 4
DIV = 1
RED = 2
GRID_SIZE = 27
FRAME_BBOX = (9, 9, 17, 17)  # inclusive hollow square frame
SOURCE_ANCHOR_SIZE = 2
TARGET_SIZE_OPTIONS = (2, 3)
COLOR_POOL = [0, 3, 5, 6, 7, 8, 9]
MAX_RETRIES = 400

REGION_KEYS = ("center", "nw", "ne", "sw", "se")
SIZE_PATTERNS = (
    {"center": 2, "nw": 2, "ne": 3, "sw": 3, "se": 2},
    {"center": 2, "nw": 3, "ne": 2, "sw": 2, "se": 3},
    {"center": 2, "nw": 2, "ne": 2, "sw": 3, "se": 3},
    {"center": 2, "nw": 3, "ne": 3, "sw": 2, "se": 2},
)


def _frame_border_mask(shape: tuple[int, int], bbox: tuple[int, int, int, int]) -> np.ndarray:
    """Boolean mask of a 1-cell-thick hollow rectangular frame."""
    r0, c0, r1, c1 = bbox
    mask = np.zeros(shape, dtype=bool)
    mask[r0, c0 : c1 + 1] = True
    mask[r1, c0 : c1 + 1] = True
    mask[r0 : r1 + 1, c0] = True
    mask[r0 : r1 + 1, c1] = True
    return mask


def _find_frame(g: np.ndarray, bg: int) -> tuple[int, int, int, int, int]:
    """Return (frame_color, r0, c0, r1, c1) for the unique hollow square frame."""
    candidates: list[tuple[int, int, int, int, int]] = []
    for color in sorted(set(map(int, g.flatten())) - {bg}):
        mask = g == color
        pts = np.argwhere(mask)
        if pts.size == 0:
            continue
        r0, c0, r1, c1 = bbox_coords(pts)
        if r1 - r0 < 4 or c1 - c0 < 4 or (r1 - r0) != (c1 - c0):
            continue
        if np.array_equal(mask, _frame_border_mask(g.shape, (r0, c0, r1, c1))):
            candidates.append((int(color), int(r0), int(c0), int(r1), int(c1)))
    assert len(candidates) == 1, "grid must contain exactly one hollow square frame"
    return candidates[0]


def _region_bounds(
    h: int, w: int, frame_bbox: tuple[int, int, int, int]
) -> dict[str, tuple[int, int, int, int]]:
    """Inclusive bounds for center + four outer corner sectors."""
    r0, c0, r1, c1 = frame_bbox
    return {
        "center": (r0 + 1, r1 - 1, c0 + 1, c1 - 1),
        "nw": (0, r0 - 1, 0, c0 - 1),
        "ne": (0, r0 - 1, c1 + 1, w - 1),
        "sw": (r1 + 1, h - 1, 0, c0 - 1),
        "se": (r1 + 1, h - 1, c1 + 1, w - 1),
    }


def _region_key(rr: int, cc: int, frame_bbox: tuple[int, int, int, int]) -> str:
    """Classify a point into the center or one of the four outer corner sectors."""
    r0, c0, r1, c1 = frame_bbox
    if r0 < rr < r1 and c0 < cc < c1:
        return "center"
    if rr < r0 and cc < c0:
        return "nw"
    if rr < r0 and cc > c1:
        return "ne"
    if rr > r1 and cc < c0:
        return "sw"
    if rr > r1 and cc > c1:
        return "se"
    raise AssertionError("red anchor must lie in the center or an outer corner sector")


def _resize_nn(img: np.ndarray, new_h: int, new_w: int) -> np.ndarray:
    """Nearest-neighbor resize supporting the 6->9 surround scaling used here."""
    src_h, src_w = img.shape
    if new_h <= 0 or new_w <= 0:
        return img.copy()
    rr = np.clip(np.floor(np.arange(new_h) * src_h / new_h).astype(int), 0, src_h - 1)
    cc = np.clip(np.floor(np.arange(new_w) * src_w / new_w).astype(int), 0, src_w - 1)
    return img[rr[:, None], cc[None, :]]


def _transform_patch(patch: np.ndarray, key: str) -> np.ndarray:
    """Reflect the surround template according to the outer corner's relation to center."""
    out = patch
    if key in ("ne", "se"):
        out = np.fliplr(out)
    if key in ("sw", "se"):
        out = np.flipud(out)
    return out


def _red_bbox(img: np.ndarray) -> tuple[int, int, int, int]:
    rr, cc = np.where(img == RED)
    assert rr.size > 0, "template must contain red anchor pixels"
    return int(rr.min()), int(cc.min()), int(rr.max()), int(cc.max())


def _surrounds_anchor(patch: np.ndarray, bg: int) -> bool:
    r0, c0, r1, c1 = _red_bbox(patch)
    top = np.any(patch[:r0, :] != bg)
    bottom = np.any(patch[r1 + 1 :, :] != bg)
    left = np.any(patch[:, :c0] != bg)
    right = np.any(patch[:, c1 + 1 :] != bg)
    return bool(top and bottom and left and right)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Copy one anchor-centered surround template from the inner region to the outer ring with size-coded scaling and radial reflections.

    Input structure:
    - The grid contains one hollow square frame of a single non-background color, creating a center region and an outer ring.
    - There are exactly five red square anchors: one inside the frame and one in each outer corner sector around it.
    - Only the center region contains the full motif, and that motif surrounds the red anchor on all four sides with background gaps.

    Rule:
    - Recover the whole non-background bbox from the center region as the source template.
    - For each outer anchor, reflect the template according to which corner sector it occupies relative to the center: right-side sectors flip LR, bottom sectors flip UD, bottom-right flips both.
    - Scale the transformed template by target_anchor_size / source_anchor_size.
    - Align the scaled template's red bbox to the target red anchor and stamp all non-background pixels except onto existing red pixels.

    Key insight: The transferable object is no longer a quadrant-local connected component but a center-anchored surround layout inside a concentric frame, so solving requires detecting the frame geometry and preserving the whole surround bbox around the anchor.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    out = g.copy()

    bg = get_bg(g)
    _frame_color, fr0, fc0, fr1, fc1 = _find_frame(g, bg)
    frame_bbox = (fr0, fc0, fr1, fc1)
    regions = _region_bounds(h, w, frame_bbox)

    red_components = components_mask(g == RED, connectivity=4)
    assert red_components, "grid must contain red anchors"

    anchors: dict[str, tuple[int, int, int]] = {}
    for pts in red_components:
        r0, c0, r1, c1 = bbox_coords(pts)
        assert (r1 - r0) == (c1 - c0), "red anchors must be square"
        rr = (r0 + r1) // 2
        cc = (c0 + c1) // 2
        key = _region_key(rr, cc, frame_bbox)
        size = r1 - r0 + 1
        assert key not in anchors, "each region must contain exactly one red anchor"
        anchors[key] = (int(r0), int(c0), int(size))

    assert set(anchors) == set(REGION_KEYS), "grid must contain one center anchor and four outer corner anchors"
    sr0, sc0, source_size = anchors["center"]
    assert source_size == SOURCE_ANCHOR_SIZE, "center anchor must be 2x2 in this combo"

    cr0, cr1, cc0, cc1 = regions["center"]
    center = g[cr0 : cr1 + 1, cc0 : cc1 + 1]
    center_pts = np.argwhere(center != bg)
    assert center_pts.size > 0, "center region must contain the source surround template"
    pr0, pc0, pr1, pc1 = bbox_coords(center_pts)
    patch = center[pr0 : pr1 + 1, pc0 : pc1 + 1].copy()

    assert _surrounds_anchor(patch, bg), "source template must surround the center anchor on all four sides"
    rr_red, cc_red = np.where(patch == RED)
    assert int(rr_red.min()) == sr0 - cr0 - pr0 and int(cc_red.min()) == sc0 - cc0 - pc0

    for key, (tr0, tc0, target_size) in anchors.items():
        if key == "center":
            continue
        patch_t = _transform_patch(patch, key)
        ratio = float(target_size) / float(source_size)
        new_h = max(1, int(round(patch_t.shape[0] * ratio)))
        new_w = max(1, int(round(patch_t.shape[1] * ratio)))
        patch_s = _resize_nn(patch_t, new_h, new_w)

        rr_red, cc_red = np.where(patch_s == RED)
        assert rr_red.size > 0, "scaled surround template must retain red anchor pixels"
        place_r0 = tr0 - int(rr_red.min())
        place_c0 = tc0 - int(cc_red.min())

        rr_p, cc_p = np.where(patch_s != bg)
        for rr0, cc0 in zip(rr_p.tolist(), cc_p.tolist(), strict=False):
            val = int(patch_s[rr0, cc0])
            rr_t = place_r0 + rr0
            cc_t = place_c0 + cc0
            if not (0 <= rr_t < h and 0 <= cc_t < w):
                continue
            if out[rr_t, cc_t] == RED:
                continue
            out[rr_t, cc_t] = val

    return out


def _source_patch() -> tuple[np.ndarray, int, int]:
    """Create a fixed-size surround template with four distinct visible sides."""
    top, bottom, left, right = random.sample(COLOR_POOL, 4)
    patch = np.full((6, 6), BG, dtype=int)

    patch[2:4, 2:4] = RED
    patch[0, 1:5] = top
    patch[5, 1:5] = bottom
    patch[1:5, 0] = left
    patch[1:5, 5] = right

    corners = {
        (0, 0): random.choice((BG, top, left)),
        (0, 5): random.choice((BG, top, right)),
        (5, 0): random.choice((BG, bottom, left)),
        (5, 5): random.choice((BG, bottom, right)),
    }
    for (r, c), val in corners.items():
        patch[r, c] = val

    assert _surrounds_anchor(patch, BG)
    assert not np.array_equal(patch, np.fliplr(patch))
    assert not np.array_equal(patch, np.flipud(patch))
    return patch, 2, 2


def _scaled_patch(patch: np.ndarray, key: str, target_size: int) -> np.ndarray:
    p = _transform_patch(patch, key)
    ratio = float(target_size) / float(SOURCE_ANCHOR_SIZE)
    h = max(1, int(round(p.shape[0] * ratio)))
    w = max(1, int(round(p.shape[1] * ratio)))
    return _resize_nn(p, h, w)


def _choose_anchor(region: tuple[int, int, int, int], patch_s: np.ndarray) -> tuple[int, int]:
    """Pick a red-anchor top-left position that keeps the whole scaled template in-region."""
    q_r0, q_r1, q_c0, q_c1 = region
    rr, cc = np.where(patch_s == RED)
    red_r0, red_c0 = int(rr.min()), int(cc.min())

    fit_r0 = q_r0 + red_r0
    fit_r1 = q_r1 - patch_s.shape[0] + 1 + red_r0
    fit_c0 = q_c0 + red_c0
    fit_c1 = q_c1 - patch_s.shape[1] + 1 + red_c0

    assert fit_r0 <= fit_r1 and fit_c0 <= fit_c1, "scaled surround template must fit its target sector"
    return random.randint(fit_r0, fit_r1), random.randint(fit_c0, fit_c1)


def _stamp(out: np.ndarray, patch_s: np.ndarray, anchor_r: int, anchor_c: int) -> None:
    rr_red, cc_red = np.where(patch_s == RED)
    place_r0 = anchor_r - int(rr_red.min())
    place_c0 = anchor_c - int(cc_red.min())

    src_rr, src_cc = np.where(patch_s != BG)
    tgt_rr = src_rr + place_r0
    tgt_cc = src_cc + place_c0

    valid = (
        (tgt_rr >= 0)
        & (tgt_rr < GRID_SIZE)
        & (tgt_cc >= 0)
        & (tgt_cc < GRID_SIZE)
    )
    tgt_rr, tgt_cc = tgt_rr[valid], tgt_cc[valid]
    src_rr, src_cc = src_rr[valid], src_cc[valid]
    not_red = out[tgt_rr, tgt_cc] != RED
    out[tgt_rr[not_red], tgt_cc[not_red]] = patch_s[src_rr[not_red], src_cc[not_red]]


def _draw_frame(out: np.ndarray, color: int) -> None:
    mask = _frame_border_mask(out.shape, FRAME_BBOX)
    out[mask] = color


def _target_sizes() -> dict[str, int]:
    return random.choice(SIZE_PATTERNS).copy()


def _gen_attempt() -> np.ndarray | None:
    target_sizes = _target_sizes()
    regions = _region_bounds(GRID_SIZE, GRID_SIZE, FRAME_BBOX)
    patch, anchor_r_in_patch, anchor_c_in_patch = _source_patch()

    scaled = {
        key: _scaled_patch(patch, key, target_sizes[key])
        for key in REGION_KEYS
        if key != "center"
    }

    anchors: dict[str, tuple[int, int]] = {}
    cr0, cr1, cc0, cc1 = regions["center"]
    tl_r0 = random.randint(cr0, cr1 - patch.shape[0] + 1)
    tl_c0 = random.randint(cc0, cc1 - patch.shape[1] + 1)
    anchors["center"] = (tl_r0 + anchor_r_in_patch, tl_c0 + anchor_c_in_patch)
    for key, patch_s in scaled.items():
        anchors[key] = _choose_anchor(regions[key], patch_s)

    out = np.full((GRID_SIZE, GRID_SIZE), BG, dtype=int)
    _draw_frame(out, DIV)
    paste(out, patch, tl_r0, tl_c0)
    for key in REGION_KEYS:
        if key == "center":
            continue
        size = target_sizes[key]
        ar, ac = anchors[key]
        paste(out, np.full((size, size), RED, dtype=int), ar, ac)
    for key, patch_s in scaled.items():
        _stamp(out, patch_s, *anchors[key])

    inp = out.copy()
    for key in ("nw", "ne", "sw", "se"):
        r0, r1, c0, c1 = regions[key]
        inp[r0 : r1 + 1, c0 : c1 + 1] = BG
        ar, ac = anchors[key]
        paste(inp, np.full((target_sizes[key], target_sizes[key]), RED, dtype=int), ar, ac)

    return inp if np.array_equal(main(inp), out) else None


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid b10624e5 combo D8/2+D13/1 input")
