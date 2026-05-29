# Variant of b10624e5
# Dimension: D13 — 图案与锚块的空间关系 (meta)
# Change: 图案紧邻锚块放置（与锚块共享边或角），图案的锚定点是锚块的特定角/边... → 图案在锚块内部（重叠）

import random

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, paste


"""
Design analysis for the D13 overlap variant
===========================================

a) Concrete behavior and trace
- Original task: the TL motif lived outside the red anchor, and the solver aligned each
  transformed copy by matching the transformed red sub-region back onto a target red block.
- Variant: the TL source is now a square anchor area whose visible pixels already mix red and
  non-red colors. The motif is not adjacent to the anchor; it is drawn on top of it.
- Concretely, TL contains one 2x2 square patch. One cell stays red, the other three cells are
  distinct non-red colors. The patch is therefore the visible result of "anchor + overlay".
- Input still keeps the other three quadrants simple: each has only a solid red NxN square.
- Output overlays the reflected/scaled TL patch directly on top of each target red square so the
  square bboxes coincide. This is the key procedural change: placement is by anchor bbox, not by
  transformed red pixels.
- Small trace:
  original logic on a 2x2 source with red at bottom-right would first find the transformed red
  cell and shift the copied patch so that that red cell lands on the target anchor's top-left.
  In the new variant the copied patch is already the anchor-sized overlay, so its own top-left
  must land on the target anchor's top-left. The difference is visible because the same patch
  lands in a different position inside the target square.

b) Complexity calibration
- Keeping the source as a full 2x2 colored square preserves the original "size controls scale"
  mechanic but avoids under-specifying the overlap rule.
- The source is forced to be asymmetric under LR and UD flips so quadrant reflections remain
  visually recoverable from 2-4 examples.
- Non-TL sizes are restricted to 2..4 instead of 1..4. Size 1 would collapse the whole overlap
  cue into one pixel and often erase the observable "inside the anchor" structure.

c) Hidden assumptions from the original solver that break here
- `components_mask(g == 2)` used to recover all anchors, including TL. That no longer works
  because the TL anchor is partially hidden by non-red overlay colors.
- The source patch used to be a larger non-bg component containing a red sub-region whose bbox
  could be aligned to target anchors. In this variant the whole source patch is the anchor area,
  so alignment by red sub-bbox is wrong.
- The original stamp step skipped existing red cells, which preserved visible target anchors under
  an adjacent motif. Here overlay colors must be allowed to overwrite red anchor pixels.
- The original roundtrip scaling assert was tied to the old generator's lossless-resize design.
  The new overlap patch is allowed to scale by ordinary nearest-neighbor without that constraint.

d) Placement legality
- Every visible object stays fully inside its quadrant; no clipping is used.
- TL has exactly one non-bg component, a 2x2 square, so the solver can recover the source bbox
  unambiguously.
- Each non-TL quadrant contains exactly one solid red square in the input, so target anchors are
  still easy to detect.

e) Visual observability
- The source patch always contains exactly one red cell at TL-input bottom-right, with three
  distinct non-red cells around it. This creates an easy-to-see orientation marker.
- At least one target anchor is larger than the source anchor, and at least two target sizes are
  present among TR/BL/BR, so examples show both reflection and scaling evidence.

f) Visual cleanliness
- Only one layer of overlap exists in any quadrant: a square overlay sitting exactly on one red
  square. There are no external decorations around the anchors, so the result remains legible.
- Divider color is excluded from motif colors so the cross separator remains visually unique.

g) Color change inferability
- There is no hidden color remapping. Colors are copied directly from the TL source patch.
- The palette is kept small: red plus three distinct non-red colors in the source patch.

Post-edit solver assumptions checked against the generator
---------------------------------------------------------
- `main()` assumes TL has exactly one non-bg component with a square bbox; the generator always
  produces one 2x2 component there.
- `main()` assumes each non-TL quadrant has exactly one solid red square; the generator enforces
  that directly.
- `main()` assumes the source patch contains both red and non-red cells; the generator always
  emits exactly one red cell and three non-red cells.
- `main()` places transformed patches by target square top-left. The generator constructs outputs
  with the same placement rule, so oracle-checking remains exact.
"""


BG = 4
DIV = 1
RED = 2
GRID_SIZE = 27
MID = 13
QUADS = {
    (0, 0): (0, 12, 0, 12),
    (0, 1): (0, 12, 14, 26),
    (1, 0): (14, 26, 0, 12),
    (1, 1): (14, 26, 14, 26),
}

SOURCE_SIZE = 2
TARGET_SIZE_RANGE = (2, 4)
COLOR_POOL = [0, 3, 5, 6, 7, 8, 9]
MAX_RETRIES = 300


def _find_dividers(g: np.ndarray, bg: int) -> tuple[int, int]:
    """Return the unique full-row/full-col divider intersection."""
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


def _resize_nn(img: np.ndarray, new_h: int, new_w: int) -> np.ndarray:
    """Nearest-neighbor resize."""
    src_h, src_w = img.shape
    rr = np.clip(np.floor(np.arange(new_h) * src_h / new_h).astype(int), 0, src_h - 1)
    cc = np.clip(np.floor(np.arange(new_w) * src_w / new_w).astype(int), 0, src_w - 1)
    return img[rr[:, None], cc[None, :]]


def _is_solid_red_square(g: np.ndarray, pts: np.ndarray) -> bool:
    """Whether a red component is a filled red square."""
    r0, c0, r1, c1 = bbox_coords(pts)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    return h == w and len(pts) == h * w and np.all(g[r0 : r1 + 1, c0 : c1 + 1] == RED)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Copy an internal anchor overlay across quadrants with scaling and reflection.

    Input structure:
    - A full row and full column of one non-background color split the grid into four quadrants.
    - The three non-top-left quadrants each contain one solid red square anchor.
    - The top-left quadrant contains one square non-background patch where a colored motif is
      already drawn on top of its red anchor, so red and non-red pixels share the same square area.

    Rule:
    - Treat the top-left square patch itself as the source anchor-plus-motif template.
    - For each other quadrant, reflect the source left/right for right quadrants and up/down for
      bottom quadrants, then resize it by the target-anchor-size / source-anchor-size ratio.
    - Place the transformed square directly over the target red anchor so their bboxes coincide,
      allowing the transformed non-red pixels to overwrite the red anchor beneath them.

    Key insight: anchor size still determines scale, but once the motif lives inside the anchor,
    placement is recovered from the whole square anchor bbox rather than from a separate attached
    motif or a transformed red sub-region.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    bg = get_bg(g)
    mid_r, mid_c = _find_dividers(g, bg)

    # Recover the three visible target anchors from solid red square components.
    anchors: dict[tuple[int, int], tuple[int, int, int]] = {}
    for pts in components_mask(g == RED, connectivity=4):
        r0, c0, r1, c1 = bbox_coords(pts)
        rr = (r0 + r1) // 2
        cc = (c0 + c1) // 2
        key = (0 if rr < mid_r else 1, 0 if cc < mid_c else 1)
        if key == (0, 0):
            continue
        if not _is_solid_red_square(g, pts):
            continue
        size = r1 - r0 + 1
        assert key not in anchors, "each non-TL quadrant must contain exactly one solid red square"
        anchors[key] = (int(r0), int(c0), int(size))

    assert set(anchors) == {(0, 1), (1, 0), (1, 1)}, "TR/BL/BR must each contain one red square anchor"

    # In TL the full visible square is the source patch, not a separate motif attached to a red block.
    tl = g[0:mid_r, 0:mid_c]
    comps = components_mask(tl != bg, connectivity=4)
    assert len(comps) == 1, "top-left quadrant must contain exactly one non-bg component"
    pr0, pc0, pr1, pc1 = bbox_coords(comps[0])
    patch = tl[pr0 : pr1 + 1, pc0 : pc1 + 1].copy()
    s_h, s_w = patch.shape
    assert s_h == s_w, "source overlap patch must be square"
    assert np.any(patch == RED), "source patch must keep some visible red anchor pixels"
    assert np.any((patch != RED) & (patch != bg)), "source patch must include overlay colors"
    anchors[(0, 0)] = (int(pr0), int(pc0), int(s_h))

    sr0, sc0, sL = anchors[(0, 0)]
    _ = (sr0, sc0)  # kept for parity with the original structure

    for (qb, qr), (tr0, tc0, tL) in anchors.items():
        if (qb, qr) == (0, 0):
            continue
        patch_t = patch
        if qr == 1:
            patch_t = np.fliplr(patch_t)
        if qb == 1:
            patch_t = np.flipud(patch_t)

        ratio = float(tL) / float(sL)
        new_h = max(1, int(round(patch_t.shape[0] * ratio)))
        new_w = max(1, int(round(patch_t.shape[1] * ratio)))
        patch_s = _resize_nn(patch_t, new_h, new_w)

        place_r0 = tr0
        place_c0 = tc0
        rr_p, cc_p = np.where(patch_s != bg)
        for rr0, cc0 in zip(rr_p.tolist(), cc_p.tolist(), strict=False):
            rr_t = place_r0 + rr0
            cc_t = place_c0 + cc0
            if rr_t == mid_r or cc_t == mid_c:
                continue
            out[rr_t, cc_t] = int(patch_s[rr0, cc0])

    return out


def _scaled_patch(patch: np.ndarray, qb: int, qr: int, target_size: int) -> np.ndarray:
    """Reflect and scale a square anchor-overlay patch to the target anchor size."""
    p = patch
    if qr:
        p = np.fliplr(p)
    if qb:
        p = np.flipud(p)
    return _resize_nn(p, target_size, target_size)


def _make_source_patch() -> np.ndarray:
    """Build an asymmetric 2x2 source overlay: three distinct colors plus one red corner."""
    colors = random.sample(COLOR_POOL, 3)
    patch = np.array(
        [
            [colors[0], colors[1]],
            [colors[2], RED],
        ],
        dtype=int,
    )
    assert not np.array_equal(patch, np.fliplr(patch))
    assert not np.array_equal(patch, np.flipud(patch))
    return patch


def _choose_sizes() -> dict[tuple[int, int], int]:
    """Choose target sizes with visible scaling evidence in each sample set."""
    while True:
        tr = random.randint(*TARGET_SIZE_RANGE)
        bl = random.randint(*TARGET_SIZE_RANGE)
        br = random.randint(*TARGET_SIZE_RANGE)
        vals = [tr, bl, br]
        if len(set(vals)) >= 2 and max(vals) > SOURCE_SIZE:
            return {
                (0, 0): SOURCE_SIZE,
                (0, 1): tr,
                (1, 0): bl,
                (1, 1): br,
            }


def _choose_anchor_top_left(qb: int, qr: int, size: int) -> tuple[int, int]:
    """Pick a top-left location for a size x size anchor fully inside its quadrant."""
    q_r0, q_r1, q_c0, q_c1 = QUADS[(qb, qr)]
    return (
        random.randint(q_r0, q_r1 - size + 1),
        random.randint(q_c0, q_c1 - size + 1),
    )


def _gen_attempt() -> np.ndarray | None:
    """Single output-first attempt."""
    target_sizes = _choose_sizes()
    patch = _make_source_patch()

    anchors: dict[tuple[int, int], tuple[int, int]] = {}
    tl_r0 = random.randint(0, MID - SOURCE_SIZE)
    tl_c0 = random.randint(0, MID - SOURCE_SIZE)
    anchors[(0, 0)] = (tl_r0, tl_c0)
    for key, size in target_sizes.items():
        if key == (0, 0):
            continue
        anchors[key] = _choose_anchor_top_left(key[0], key[1], size)

    out = np.full((GRID_SIZE, GRID_SIZE), BG, dtype=int)
    out[MID, :] = DIV
    out[:, MID] = DIV
    paste(out, patch, tl_r0, tl_c0)
    for key, size in target_sizes.items():
        if key == (0, 0):
            continue
        ar, ac = anchors[key]
        paste(out, np.full((size, size), RED, dtype=int), ar, ac)
        paste(out, _scaled_patch(patch, key[0], key[1], size), ar, ac)

    inp = np.full((GRID_SIZE, GRID_SIZE), BG, dtype=int)
    inp[MID, :] = DIV
    inp[:, MID] = DIV
    paste(inp, patch, tl_r0, tl_c0)
    for key, size in target_sizes.items():
        if key == (0, 0):
            continue
        ar, ac = anchors[key]
        paste(inp, np.full((size, size), RED, dtype=int), ar, ac)

    return inp if np.array_equal(main(inp), out) else None


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid b10624e5 D13-overlap input")
