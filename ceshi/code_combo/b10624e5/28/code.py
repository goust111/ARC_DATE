# Combo variant of b10624e5 (index 28)
# combo_spec: D7/2+D15/1
# Name: 象限独立裁剪拼接 + 每个象限既是源也是目标

from __future__ import annotations

import random
import sys

import numpy as np

try:
    from arc_utils import bbox_coords, components_mask, get_bg, paste
except ModuleNotFoundError:
    sys.path.append("/home/lixinyu/snap/new/solvers")
    from arc_utils import bbox_coords, components_mask, get_bg, paste


# Design analysis for combo D7/2 + D15/1
# a) Concrete behavior and trace
#    - Original task: only TL contains a full motif; TR/BL/BR show only red anchors. Output stays on the
#      same 27x27 canvas and copies the TL motif into the other three quadrants with scale + reflections.
#    - Fused task here:
#      1. all four quadrants contain their own local motif attached to their own red anchor,
#      2. each quadrant also receives one imported motif copied from the diagonally opposite quadrant,
#      3. after those four local solves are done, each quadrant is cropped independently and the four crops
#         are repacked into a 2x2 output with 1-cell gaps and a 1-cell outer border.
#    - Small trace with size 2 anchors:
#      * TL input tile can be a 2x2 logical block `[[a,b],[c,red]]`, i.e. a local motif living NW of the
#        anchor. BR may use the same NW-oriented template but different colors.
#      * BR -> TL differs in both row side and column side, so the copied patch is flipped both vertically
#        and horizontally; the imported motif therefore lands SE of TL's anchor.
#      * TL output quadrant now visibly contains the original NW motif plus a new SE motif from BR.
#      * Only after this local solve do we crop TL, TR, BL, BR separately and repack them, so the final
#        output no longer preserves the divider cross or the original absolute spacing.
# b) Complexity calibration
#    - "Every quadrant is both source and target" is implemented as diagonal pair exchange, not 4-to-4
#      broadcast. That keeps the interaction nontrivial but avoids unreadable three-way overlaps at one
#      anchor.
#    - The generator uses only two motif orientation families: TL/BR are NW-oriented, TR/BL are
#      NE-oriented. Because diagonal copying applies a double reflection, imported motifs land on the
#      opposite side of the same anchor, making the fusion visually obvious.
#    - Anchor sizes still vary in 1..4. At least one diagonal pair must have mismatched sizes so scaling
#      remains visible in generated samples.
# c) Implicit-assumption audit of the original solver
#    - Old assumption: TL is the unique source. Broken. The new solver must recover one motif component in
#      every quadrant and choose the diagonally opposite quadrant as the source for each target.
#    - Old assumption: non-TL quadrants contain only red anchors in the input. Broken. Component extraction
#      must now work independently inside each quadrant instead of globally privileging TL.
#    - Old assumption: output shape equals input shape. Broken by D7/2, so the final stage cannot return
#      the solved 27x27 board. It must crop solved quadrants and repack them.
#    - Old assumption: there is no need to keep per-quadrant solved content after stamping. Broken because
#      D7/2 depends on each quadrant's individual crop.
# d) Placement legality
#    - Each motif is built from a 2x2 logical template scaled by that quadrant's anchor size. This makes
#      every source->target resize lossless for integer target sizes 1..4.
#    - Anchors are placed in a central legal range that simultaneously fits the local motif and the
#      imported diagonal motif inside the same 13x13 quadrant. This prevents overlap with the divider and
#      avoids clipping.
#    - Local and imported motifs are designed to overlap only on the red anchor block, so color conflicts
#      remain interpretable even though the solver allows the imported layer to overwrite non-red pixels.
# e) Visual observability
#    - Humans can see four motifs in the input, not one, and each output crop contains a second motif that
#      matches the diagonally opposite input quadrant after scaling and double reflection.
#    - The packed output removes the divider cross entirely, so the solver must mentally map each packed
#      crop back to its original quadrant identity.
#    - No rule branch uses hidden numeric color-id ordering; only visible quadrant position, anchor size,
#      and motif geometry matter.
# f) Visual cleanliness
#    - Only one imported motif is added per quadrant.
#    - The two orientation families make local vs imported content sit on opposite sides of the anchor.
#    - The packed renderer keeps one background border and one-cell seams, so the four crops stay easy to
#      separate by eye.
# g) Fusion-behavior checks and new assumptions
#    - Generator metrics reject samples unless:
#      * all 4 quadrants change after fusion (`changed_quadrants == 4`),
#      * the packed output shape differs from 27x27,
#      * the fused output differs from a TL-only packed baseline.
#    - New solver assumption: every quadrant has exactly one red square anchor and exactly one non-bg
#      component containing it. The generator enforces both conditions directly.


BG = 4
DIV = 1
RED = 2
GRID_SIZE = 27
MID = 13
ANCHOR_SIZE_RANGE = (1, 4)
MAX_RETRIES = 400
COLOR_POOL = [3, 5, 6, 7, 8, 9]
ORDER = [(0, 0), (0, 1), (1, 0), (1, 1)]
OPPOSITE = {
    (0, 0): (1, 1),
    (0, 1): (1, 0),
    (1, 0): (0, 1),
    (1, 1): (0, 0),
}
ORIENTATION = {
    (0, 0): "nw",
    (1, 1): "nw",
    (0, 1): "ne",
    (1, 0): "ne",
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


def _quadrant_bounds(h: int, w: int, mid_r: int, mid_c: int) -> dict[tuple[int, int], tuple[int, int, int, int]]:
    return {
        (0, 0): (0, mid_r - 1, 0, mid_c - 1),
        (0, 1): (0, mid_r - 1, mid_c + 1, w - 1),
        (1, 0): (mid_r + 1, h - 1, 0, mid_c - 1),
        (1, 1): (mid_r + 1, h - 1, mid_c + 1, w - 1),
    }


def _resize_nn(img: np.ndarray, new_h: int, new_w: int) -> np.ndarray:
    src_h, src_w = img.shape
    assert src_h > 0 and src_w > 0
    rr = np.clip(np.floor(np.arange(new_h) * src_h / new_h).astype(int), 0, src_h - 1)
    cc = np.clip(np.floor(np.arange(new_w) * src_w / new_w).astype(int), 0, src_w - 1)
    return img[rr[:, None], cc[None, :]]


def _crop_non_bg(tile: np.ndarray, bg: int) -> np.ndarray:
    pts = np.argwhere(tile != bg)
    assert pts.size > 0, "crop expects at least one non-bg pixel"
    r0, c0, r1, c1 = bbox_coords(pts)
    return tile[r0 : r1 + 1, c0 : c1 + 1].copy()


def _transform_patch(
    patch: np.ndarray,
    src_key: tuple[int, int],
    dst_key: tuple[int, int],
    src_size: int,
    dst_size: int,
) -> np.ndarray:
    p = patch
    if src_key[1] != dst_key[1]:
        p = np.fliplr(p)
    if src_key[0] != dst_key[0]:
        p = np.flipud(p)
    new_h = max(1, int(round(p.shape[0] * float(dst_size) / float(src_size))))
    new_w = max(1, int(round(p.shape[1] * float(dst_size) / float(src_size))))
    scaled = _resize_nn(p, new_h, new_w)
    roundtrip = _resize_nn(scaled, p.shape[0], p.shape[1])
    assert np.array_equal(roundtrip, p), "scaling must preserve the logical motif exactly"
    return scaled


def _stamp_aligned(
    canvas: np.ndarray,
    patch: np.ndarray,
    anchor_r: int,
    anchor_c: int,
    bg: int,
) -> None:
    rr_red, cc_red = np.where(patch == RED)
    assert rr_red.size > 0, "scaled patch must contain the red anchor"
    place_r0 = int(anchor_r) - int(rr_red.min())
    place_c0 = int(anchor_c) - int(cc_red.min())
    h, w = canvas.shape
    rr_p, cc_p = np.where(patch != bg)
    for rr0, cc0 in zip(rr_p.tolist(), cc_p.tolist(), strict=False):
        tr = place_r0 + int(rr0)
        tc = place_c0 + int(cc0)
        if not (0 <= tr < h and 0 <= tc < w):
            continue
        if int(canvas[tr, tc]) == RED:
            continue
        canvas[tr, tc] = int(patch[rr0, cc0])


def _scan_scene(grid: np.ndarray) -> tuple[int, int, dict[tuple[int, int], dict[str, np.ndarray | int]]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = int(get_bg(g))
    mid_r, mid_c = _find_dividers(g, bg)
    bounds = _quadrant_bounds(h, w, mid_r, mid_c)
    info: dict[tuple[int, int], dict[str, np.ndarray | int]] = {}

    for key, (r0, r1, c0, c1) in bounds.items():
        q = g[r0 : r1 + 1, c0 : c1 + 1].copy()
        red_comps = components_mask(q == RED, connectivity=4)
        assert len(red_comps) == 1, "each quadrant must contain exactly one red anchor component"
        ar0, ac0, ar1, ac1 = bbox_coords(red_comps[0])
        size = int(ar1 - ar0 + 1)
        assert size == int(ac1 - ac0 + 1), "red anchor must be square"
        assert np.all(q[ar0 : ar1 + 1, ac0 : ac1 + 1] == RED), "red anchor must be filled"

        patch_bb = None
        for pts in components_mask(q != bg, connectivity=4):
            if any(int(q[r, c]) == RED for r, c in pts):
                patch_bb = bbox_coords(pts)
                break
        assert patch_bb is not None, "quadrant must contain a non-bg component attached to the anchor"
        pr0, pc0, pr1, pc1 = patch_bb
        patch = q[pr0 : pr1 + 1, pc0 : pc1 + 1].copy()
        info[key] = {
            "quadrant": q,
            "patch": patch,
            "anchor_r": int(ar0),
            "anchor_c": int(ac0),
            "size": int(size),
        }
    return bg, mid_r, info


def _solve_quadrants(grid: np.ndarray) -> tuple[int, dict[tuple[int, int], np.ndarray], dict[tuple[int, int], dict[str, np.ndarray | int]]]:
    bg, _mid_r, info = _scan_scene(grid)
    solved: dict[tuple[int, int], np.ndarray] = {}
    for key in ORDER:
        q = np.array(info[key]["quadrant"], dtype=int, copy=True)
        src_key = OPPOSITE[key]
        patch_s = _transform_patch(
            np.asarray(info[src_key]["patch"], dtype=int),
            src_key,
            key,
            int(info[src_key]["size"]),
            int(info[key]["size"]),
        )
        _stamp_aligned(q, patch_s, int(info[key]["anchor_r"]), int(info[key]["anchor_c"]), bg)
        solved[key] = q
    return bg, solved, info


def _pack_quadrants(quadrants: dict[tuple[int, int], np.ndarray], bg: int) -> np.ndarray:
    crops = {key: _crop_non_bg(tile, bg) for key, tile in quadrants.items()}
    top_h = max(int(crops[(0, 0)].shape[0]), int(crops[(0, 1)].shape[0]))
    bot_h = max(int(crops[(1, 0)].shape[0]), int(crops[(1, 1)].shape[0]))
    left_w = max(int(crops[(0, 0)].shape[1]), int(crops[(1, 0)].shape[1]))
    right_w = max(int(crops[(0, 1)].shape[1]), int(crops[(1, 1)].shape[1]))

    out_h = 1 + top_h + 1 + bot_h + 1
    out_w = 1 + left_w + 1 + right_w + 1
    out = np.full((out_h, out_w), int(bg), dtype=int)

    placements = {
        (0, 0): (1, 1),
        (0, 1): (1, 1 + left_w + 1),
        (1, 0): (1 + top_h + 1, 1),
        (1, 1): (1 + top_h + 1, 1 + left_w + 1),
    }
    for key in ORDER:
        rr, cc = placements[key]
        paste(out, crops[key], rr, cc)
    return out


def _baseline_tl_only_packed(grid: np.ndarray) -> np.ndarray:
    bg, _mid_r, info = _scan_scene(grid)
    solved = {key: np.array(info[key]["quadrant"], dtype=int, copy=True) for key in ORDER}
    for key in ORDER:
        if key == (0, 0):
            continue
        patch_s = _transform_patch(
            np.asarray(info[(0, 0)]["patch"], dtype=int),
            (0, 0),
            key,
            int(info[(0, 0)]["size"]),
            int(info[key]["size"]),
        )
        _stamp_aligned(solved[key], patch_s, int(info[key]["anchor_r"]), int(info[key]["anchor_c"]), bg)
    return _pack_quadrants(solved, bg)


def _fusion_metrics(grid: np.ndarray) -> dict[str, int]:
    bg, solved, info = _solve_quadrants(grid)
    changed_quadrants = 0
    for key in ORDER:
        before = _crop_non_bg(np.asarray(info[key]["quadrant"], dtype=int), bg)
        after = _crop_non_bg(solved[key], bg)
        if not np.array_equal(before, after):
            changed_quadrants += 1
    packed = _pack_quadrants(solved, bg)
    pair_size_mismatches = int(info[(0, 0)]["size"] != info[(1, 1)]["size"]) + int(
        info[(0, 1)]["size"] != info[(1, 0)]["size"]
    )
    baseline_equal = int(np.array_equal(packed, _baseline_tl_only_packed(grid)))
    return {
        "changed_quadrants": int(changed_quadrants),
        "out_h": int(packed.shape[0]),
        "out_w": int(packed.shape[1]),
        "pair_size_mismatches": int(pair_size_mismatches),
        "baseline_equal": int(baseline_equal),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Exchange diagonal quadrant motifs through anchor-coded scaling, then repack the solved quadrants.

    Input structure:
    - The grid is split into four 13x13 quadrants by one full divider row and one full divider column.
    - Every quadrant contains exactly one filled red square anchor.
    - Unlike the base task, every quadrant also contains its own colored motif as the same 4-connected
      component as that anchor.

    Rule:
    - Treat each quadrant as a target whose source is the diagonally opposite quadrant.
    - Copy that opposite motif into the target by reflecting across each differing quadrant axis and by
      resizing it according to target-anchor-size / source-anchor-size.
    - Keep the target's original motif, stamp the imported motif onto the same anchor, then crop each
      solved quadrant to its non-background bbox and repack the four crops into a bordered 2x2 layout.

    Key insight: quadrant identity is now needed twice: first to decide which opposite motif feeds a given
    anchor and which reflections apply, then again to place the independently cropped solved quadrants into
    the packed output order.
    """
    bg, solved, _info = _solve_quadrants(np.asarray(grid, dtype=int))
    return _pack_quadrants(solved, bg)


def _make_logical_patch(kind: str, colors: list[int]) -> np.ndarray:
    assert len(colors) == 3 and len(set(colors)) == 3
    if kind == "nw":
        return np.array([[colors[0], colors[1]], [colors[2], RED]], dtype=int)
    assert kind == "ne"
    return np.array([[colors[0], colors[1]], [RED, colors[2]]], dtype=int)


def _scaled_local_patch(kind: str, size: int, colors: list[int]) -> tuple[np.ndarray, int, int]:
    logical = _make_logical_patch(kind, colors)
    patch = np.repeat(np.repeat(logical, int(size), axis=0), int(size), axis=1)
    if kind == "nw":
        return patch, int(size), int(size)
    return patch, int(size), 0


def _anchor_range(size: int) -> tuple[int, int]:
    lo = int(size)
    hi = 13 - 2 * int(size)
    assert lo <= hi, "anchor central range must exist"
    return lo, hi


def _sample_sizes() -> dict[tuple[int, int], int]:
    for _ in range(200):
        sizes = {key: random.randint(*ANCHOR_SIZE_RANGE) for key in ORDER}
        if len(set(sizes.values())) < 2:
            continue
        if sizes[(0, 0)] == 4 and sizes[(0, 1)] == 4 and sizes[(1, 0)] == 4 and sizes[(1, 1)] == 4:
            continue
        if sizes[(0, 0)] == sizes[(1, 1)] and sizes[(0, 1)] == sizes[(1, 0)]:
            continue
        return sizes
    raise RuntimeError("failed to sample nondegenerate anchor sizes")


def _sample_color_triplets() -> dict[tuple[int, int], list[int]]:
    while True:
        triplets = {}
        signatures = set()
        ok = True
        for key in ORDER:
            cols = random.sample(COLOR_POOL, 3)
            sig = tuple(cols)
            if sig in signatures:
                ok = False
                break
            signatures.add(sig)
            triplets[key] = cols
        if ok:
            return triplets


def _build_input() -> np.ndarray:
    grid = np.full((GRID_SIZE, GRID_SIZE), BG, dtype=int)
    grid[MID, :] = DIV
    grid[:, MID] = DIV
    sizes = _sample_sizes()
    colors = _sample_color_triplets()

    bounds = {
        (0, 0): (0, 12, 0, 12),
        (0, 1): (0, 12, 14, 26),
        (1, 0): (14, 26, 0, 12),
        (1, 1): (14, 26, 14, 26),
    }

    for key in ORDER:
        size = int(sizes[key])
        patch, ar_rel, ac_rel = _scaled_local_patch(ORIENTATION[key], size, colors[key])
        q_r0, _q_r1, q_c0, _q_c1 = bounds[key]
        lo, hi = _anchor_range(size)
        anchor_r = q_r0 + random.randint(lo, hi)
        anchor_c = q_c0 + random.randint(lo, hi)
        paste(grid, patch, anchor_r - ar_rel, anchor_c - ac_rel)

    return grid


def _gen_attempt() -> np.ndarray | None:
    inp = _build_input()
    try:
        out = main(inp)
        metrics = _fusion_metrics(inp)
    except Exception:
        return None

    if metrics["changed_quadrants"] != 4:
        return None
    if metrics["pair_size_mismatches"] < 1:
        return None
    if metrics["baseline_equal"] != 0:
        return None
    if (metrics["out_h"], metrics["out_w"]) == (GRID_SIZE, GRID_SIZE):
        return None
    if out.shape[0] > GRID_SIZE or out.shape[1] > GRID_SIZE:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid b10624e5 combo input")
