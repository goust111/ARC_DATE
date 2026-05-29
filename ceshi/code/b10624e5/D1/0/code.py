# Variant of b10624e5
# Dimension: D1 — 匹配条件：红色锚块颜色匹配 (match_cond)
# Change: 所有象限的锚块共享同一颜色（红色），通过 s.anchor_color == d.anchor_color 配对；缩放因... → 多颜色锚块分组配对

import random
import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, paste


# ---------------------------------------------------------------------------
# Design analysis for the D1 variant
# ---------------------------------------------------------------------------
# design:
#   overview:
#     original_task:
#       TL had one red anchor+motif component; each non-TL quadrant had one red square anchor.
#       Solver only needed quadrant position for reflection and anchor size for scaling.
#     variant_goal:
#       Keep the quadrant divider, scale-by-anchor-size, and quadrant-based reflection unchanged.
#       Replace the single-color matching condition with color-group matching:
#       TL now acts as a "catalog" of source components, one per anchor color.
#       Each target quadrant contains exactly one colored square anchor, and the anchor color
#       decides which TL source component is copied there.
#   a_concretization_and_trace:
#     input:
#       - 27x27 grid with divider row/col at index 13.
#       - TL quadrant contains exactly 3 disconnected source components.
#       - Each source component contains one 2x2 anchor square of its own color
#         (sampled from {2,3,8,9}) plus an attached asymmetric motif.
#       - TR / BL / BR each contain exactly one standalone colored NxN anchor square.
#         Their colors are a permutation of the TL anchor colors.
#     output:
#       - For each target quadrant, find the TL component whose anchor color matches the
#         target anchor color, reflect it according to quadrant side, scale it by
#         target_size / source_size, align the colored anchor regions, and stamp it.
#     concrete_trace:
#       - Original: TR has a 3x3 red anchor, so solver always takes the unique TL red motif,
#         flips left-right, scales by 3/2, and pastes it.
#       - Variant: TR may have a 3x3 blue anchor while BL has a 4x4 red anchor.
#         Solver must first detect "TR is blue, so use the TL blue source patch",
#         and only then do the old reflection+scaling step. BL independently matches red.
#       - Difference: motif selection is no longer tied to "the only red component";
#         it is a two-step inference: match color group first, then apply the old geometry.
#   b_complexity_calibration:
#     - Use exactly 3 source colors per puzzle so the color-matching rule is visible.
#     - Use a random permutation of those 3 colors across TR/BL/BR so quadrant alone
#       cannot predict which motif should appear.
#     - Keep source motifs small (fixed 4x6 physical bbox) and disconnected to avoid clutter.
#     - Motifs are asymmetric because each logical 2x3 pattern has one missing corner,
#       so LR/UD reflections remain visually observable.
#   c_implicit_assumptions_review:
#     original_main_lines_that_break:
#       - components_mask(g == 2): assumes every anchor is red. Invalid now.
#       - assert top-left has the source red anchor: no longer a single anchor in TL.
#       - component containing any red pixel: source extraction must become per-color.
#       - np.where(patch_s == 2): alignment must use the matched anchor color, not red.
#       - if out[...] == 2: skip-over-anchor must use the target anchor color, not red.
#     new_solver_assumptions:
#       - Only target anchor colors appear as standalone square anchors outside TL.
#       - TL contains exactly one source anchor square for each target color.
#       - Each TL non-bg component contains exactly one such source anchor.
#     generator_guarantees:
#       - Non-TL input contains only anchor squares.
#       - TL motif fill colors come from a pool disjoint from the chosen anchor colors.
#       - TL source components are separated by at least one bg row/col.
#   d_placement_legality:
#     - TL source patches use fixed 4x6 bboxes and are placed into 3 of 4 preset slots
#       inside the 13x13 TL quadrant, so they never overlap or cross the divider.
#     - Target anchors are positioned with a fit-first strategy using the already-scaled,
#       already-reflected matched patch; full fit is preferred, anchor-only fit is allowed.
#     - Stamping still clips at grid bounds and skips the divider row/col.
#   e_visual_observability:
#     - Within one example, the same quadrant positions can host different motif identities
#       because target colors are permuted.
#     - Across 2-3 examples, users can see that "same anchor color => same motif family"
#       while size and reflection still follow the old rule.
#   f_cleanliness:
#     - Only TL contains multiple objects; other quadrants stay sparse in the input.
#     - Extra motif colors avoid divider color and the chosen anchor colors, so anchors remain
#       easy to isolate visually.
#   g_color_rule_inference:
#     - Color ids are not used for sorting or any hidden numeric ordering.
#     - The visible cue is direct equality of anchor colors between TL and target quadrants.


def _find_dividers(g: np.ndarray, bg: int) -> tuple[int, int]:
    """
    Return the unique full-row/full-col quadrant divider: (mid_row, mid_col).
    """
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


def _quadrant_key(mid_r: int, mid_c: int, r0: int, c0: int, r1: int, c1: int) -> tuple[int, int]:
    assert r1 < mid_r or r0 > mid_r, "component must stay within one row half"
    assert c1 < mid_c or c0 > mid_c, "component must stay within one col half"
    rr = (r0 + r1) // 2
    cc = (c0 + c1) // 2
    return (0 if rr < mid_r else 1, 0 if cc < mid_c else 1)


def _resize_nn(img: np.ndarray, new_h: int, new_w: int) -> np.ndarray:
    src_h, src_w = img.shape
    if new_h <= 0 or new_w <= 0:
        return img.copy()
    rr = np.clip(np.floor(np.arange(new_h) * src_h / new_h).astype(int), 0, src_h - 1)
    cc = np.clip(np.floor(np.arange(new_w) * src_w / new_w).astype(int), 0, src_w - 1)
    return img[rr[:, None], cc[None, :]]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Copy color-matched source motifs from the top-left catalog into the other quadrants.

    Input structure:
    - A single non-bg divider row and divider column split the grid into four 13x13 quadrants.
    - The top-left quadrant contains several disconnected source components; each has a square
      anchor block of a distinct color plus a colored motif attached to that anchor.
    - Each non-top-left quadrant contains exactly one square anchor block, and its color matches
      one of the source anchors in the top-left quadrant.

    Rule:
    - For each target quadrant, select the top-left source component whose anchor color matches
      the target anchor color.
    - Reflect that matched source component left-right for right quadrants, up-down for bottom
      quadrants, and both for the bottom-right quadrant.
    - Scale the reflected component by the ratio of target-anchor size to matched-source-anchor
      size, align the anchor-colored region onto the target anchor, then stamp non-bg pixels.

    Key insight: Quadrant position still determines the reflection and anchor size still
    determines the scale, but the motif identity now comes from visible anchor-color matching.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    out = g.copy()

    bg = get_bg(g)
    mid_r, mid_c = _find_dividers(g, bg)
    divider_color = int(g[mid_r, mid_c])

    square_parts: list[dict[str, int]] = []
    colors = sorted({int(v) for v in g.flatten()} - {int(bg), divider_color})
    for color in colors:
        for pts in components_mask(g == color, connectivity=4):
            r0, c0, r1, c1 = bbox_coords(pts)
            if r0 <= mid_r <= r1 or c0 <= mid_c <= c1:
                continue
            size_h = r1 - r0 + 1
            size_w = c1 - c0 + 1
            if size_h != size_w:
                continue
            if len(pts) != size_h * size_w:
                continue
            qb, qr = _quadrant_key(mid_r, mid_c, int(r0), int(c0), int(r1), int(c1))
            square_parts.append(
                {
                    "color": int(color),
                    "r0": int(r0),
                    "c0": int(c0),
                    "size": int(size_h),
                    "qb": int(qb),
                    "qr": int(qr),
                }
            )

    targets: dict[tuple[int, int], dict[str, int]] = {}
    target_colors: set[int] = set()
    for part in square_parts:
        key = (int(part["qb"]), int(part["qr"]))
        if key == (0, 0):
            continue
        assert key not in targets, "each non-top-left quadrant must contain exactly one anchor square"
        targets[key] = part
        target_colors.add(int(part["color"]))
    assert set(targets) == {(0, 1), (1, 0), (1, 1)}, "TR/BL/BR must each contain one target anchor"

    source_anchors: dict[int, dict[str, int]] = {}
    for part in square_parts:
        key = (int(part["qb"]), int(part["qr"]))
        if key != (0, 0):
            continue
        color = int(part["color"])
        if color not in target_colors:
            continue
        assert color not in source_anchors, "each target color must have exactly one TL source anchor"
        source_anchors[color] = part
    assert target_colors == set(source_anchors), "every target anchor color must exist in TL"

    tl = g[0:mid_r, 0:mid_c]
    tl_components = components_mask(tl != bg, connectivity=4)
    tl_entries: list[tuple[set[tuple[int, int]], tuple[int, int, int, int]]] = []
    for pts in tl_components:
        r0, c0, r1, c1 = bbox_coords(pts)
        tl_entries.append(({(int(r), int(c)) for r, c in pts}, (int(r0), int(c0), int(r1), int(c1))))

    source_patches: dict[int, tuple[np.ndarray, int]] = {}
    for color, anchor in source_anchors.items():
        anchor_pixel = (int(anchor["r0"]), int(anchor["c0"]))
        patch_bb = None
        for ptset, bb in tl_entries:
            if anchor_pixel in ptset:
                patch_bb = bb
                break
        assert patch_bb is not None, "each TL source anchor must belong to one non-bg component"
        pr0, pc0, pr1, pc1 = patch_bb
        patch = tl[pr0 : pr1 + 1, pc0 : pc1 + 1].copy()
        source_patches[int(color)] = (patch, int(anchor["size"]))

    for (qb, qr), target in targets.items():
        color = int(target["color"])
        tr0 = int(target["r0"])
        tc0 = int(target["c0"])
        t_size = int(target["size"])
        patch, s_size = source_patches[color]

        patch_t = patch
        if qr == 1:
            patch_t = np.fliplr(patch_t)
        if qb == 1:
            patch_t = np.flipud(patch_t)

        ratio = float(t_size) / float(s_size)
        new_h = max(1, int(round(patch_t.shape[0] * ratio)))
        new_w = max(1, int(round(patch_t.shape[1] * ratio)))
        patch_s = _resize_nn(patch_t, new_h, new_w)
        roundtrip = _resize_nn(patch_s, patch_t.shape[0], patch_t.shape[1])
        assert np.array_equal(roundtrip, patch_t), "scaling must preserve the matched motif"

        rr_anchor, cc_anchor = np.where(patch_s == color)
        assert rr_anchor.size > 0, "scaled patch must retain the matched anchor color"
        place_r0 = tr0 - int(rr_anchor.min())
        place_c0 = tc0 - int(cc_anchor.min())

        rr_p, cc_p = np.where(patch_s != bg)
        for rr0, cc0 in zip(rr_p.tolist(), cc_p.tolist(), strict=False):
            val = int(patch_s[rr0, cc0])
            rr_t = place_r0 + rr0
            cc_t = place_c0 + cc0
            if not (0 <= rr_t < h and 0 <= cc_t < w):
                continue
            if rr_t == mid_r or cc_t == mid_c:
                continue
            if out[rr_t, cc_t] == color:
                continue
            out[rr_t, cc_t] = val

    return out


# --------------- generator ---------------

BG = 4
DIV = 1
GRID_SIZE = 27
MID = 13
QUADS = {
    (0, 0): (0, 12, 0, 12),
    (0, 1): (0, 12, 14, 26),
    (1, 0): (14, 26, 0, 12),
    (1, 1): (14, 26, 14, 26),
}
TARGET_KEYS = [(0, 1), (1, 0), (1, 1)]

SOURCE_ANCHOR_SIZE = 2
TARGET_SIZE_RANGE = (1, 4)
ANCHOR_POOL = [2, 3, 8, 9]
EXTRA_POOL = [0, 5, 6, 7]
HOLE_CORNERS = [(0, 0), (0, 2), (1, 0), (1, 2)]
SOURCE_SLOTS = [(0, 0), (0, 7), (8, 0), (8, 7)]

MAX_RETRIES = 300


def _scaled_patch(patch: np.ndarray, qb: int, qr: int, target_size: int) -> np.ndarray:
    p = patch
    if qr:
        p = np.fliplr(p)
    if qb:
        p = np.flipud(p)
    ratio = target_size / SOURCE_ANCHOR_SIZE
    new_h = max(1, int(round(p.shape[0] * ratio)))
    new_w = max(1, int(round(p.shape[1] * ratio)))
    return _resize_nn(p, new_h, new_w)


def _make_source_patch(anchor_color: int, hole: tuple[int, int]) -> tuple[np.ndarray, int, int]:
    logical = np.full((2, 3), BG, dtype=int)
    filled = [(r, c) for r in range(2) for c in range(3) if (r, c) != hole]
    anchor_r, anchor_c = random.choice(filled)
    logical[anchor_r, anchor_c] = anchor_color

    extras = EXTRA_POOL[:]
    random.shuffle(extras)
    n_extra = random.randint(2, 3)
    palette = extras[:n_extra]
    other_cells = [(r, c) for r, c in filled if (r, c) != (anchor_r, anchor_c)]
    random.shuffle(other_cells)
    for idx, (r, c) in enumerate(other_cells):
        logical[r, c] = palette[idx % len(palette)]

    patch = np.repeat(np.repeat(logical, SOURCE_ANCHOR_SIZE, axis=0), SOURCE_ANCHOR_SIZE, axis=1)
    return patch, anchor_r * SOURCE_ANCHOR_SIZE, anchor_c * SOURCE_ANCHOR_SIZE


def _choose_anchor(qb: int, qr: int, size: int, patch_s: np.ndarray, anchor_color: int) -> tuple[int, int]:
    q_r0, q_r1, q_c0, q_c1 = QUADS[(qb, qr)]
    rr, cc = np.where(patch_s == anchor_color)
    red_r0, red_c0 = int(rr.min()), int(cc.min())

    fit_r0 = q_r0 + red_r0
    fit_r1 = q_r1 - patch_s.shape[0] + 1 + red_r0
    fit_c0 = q_c0 + red_c0
    fit_c1 = q_c1 - patch_s.shape[1] + 1 + red_c0

    anch_r0, anch_r1 = q_r0, q_r1 - size + 1
    anch_c0, anch_c1 = q_c0, q_c1 - size + 1

    r_lo, r_hi = max(fit_r0, anch_r0), min(fit_r1, anch_r1)
    c_lo, c_hi = max(fit_c0, anch_c0), min(fit_c1, anch_c1)
    if r_lo <= r_hi and c_lo <= c_hi:
        return random.randint(r_lo, r_hi), random.randint(c_lo, c_hi)
    return random.randint(anch_r0, anch_r1), random.randint(anch_c0, anch_c1)


def _stamp(out: np.ndarray, patch_s: np.ndarray, anchor_r: int, anchor_c: int, anchor_color: int) -> None:
    rr_anchor, cc_anchor = np.where(patch_s == anchor_color)
    place_r0 = anchor_r - int(rr_anchor.min())
    place_c0 = anchor_c - int(cc_anchor.min())

    src_rr, src_cc = np.where(patch_s != BG)
    tgt_rr = src_rr + place_r0
    tgt_cc = src_cc + place_c0

    valid = (
        (tgt_rr >= 0)
        & (tgt_rr < GRID_SIZE)
        & (tgt_cc >= 0)
        & (tgt_cc < GRID_SIZE)
        & (tgt_rr != MID)
        & (tgt_cc != MID)
    )
    tgt_rr, tgt_cc = tgt_rr[valid], tgt_cc[valid]
    src_rr, src_cc = src_rr[valid], src_cc[valid]
    not_anchor = out[tgt_rr, tgt_cc] != anchor_color
    out[tgt_rr[not_anchor], tgt_cc[not_anchor]] = patch_s[src_rr[not_anchor], src_cc[not_anchor]]


def _gen_attempt() -> np.ndarray | None:
    source_colors = random.sample(ANCHOR_POOL, 3)
    holes = random.sample(HOLE_CORNERS, 3)
    slots = random.sample(SOURCE_SLOTS, 3)

    source_specs: dict[int, dict[str, object]] = {}
    for color, hole, (r0, c0) in zip(source_colors, holes, slots, strict=False):
        patch, ar, ac = _make_source_patch(color, hole)
        source_specs[color] = {"patch": patch, "anchor_r": ar, "anchor_c": ac, "slot_r0": r0, "slot_c0": c0}

    target_sizes = {(0, 0): SOURCE_ANCHOR_SIZE}
    for key in TARGET_KEYS:
        target_sizes[key] = random.randint(*TARGET_SIZE_RANGE)
    if all(target_sizes[key] == SOURCE_ANCHOR_SIZE for key in TARGET_KEYS):
        force_key = random.choice(TARGET_KEYS)
        target_sizes[force_key] = random.choice([1, 3, 4])

    permuted_colors = random.sample(source_colors, len(source_colors))
    target_colors = {key: color for key, color in zip(TARGET_KEYS, permuted_colors, strict=False)}

    scaled = {}
    for key in TARGET_KEYS:
        color = target_colors[key]
        scaled[key] = _scaled_patch(source_specs[color]["patch"], key[0], key[1], target_sizes[key])

    anchors: dict[tuple[int, int], tuple[int, int]] = {}
    for color, spec in source_specs.items():
        anchors[(0, 0, color)] = (int(spec["slot_r0"]) + int(spec["anchor_r"]), int(spec["slot_c0"]) + int(spec["anchor_c"]))
    for key in TARGET_KEYS:
        color = target_colors[key]
        anchors[key] = _choose_anchor(key[0], key[1], target_sizes[key], scaled[key], color)

    out = np.full((GRID_SIZE, GRID_SIZE), BG, dtype=int)
    out[MID, :] = DIV
    out[:, MID] = DIV

    for spec in source_specs.values():
        paste(out, spec["patch"], int(spec["slot_r0"]), int(spec["slot_c0"]))
    for key in TARGET_KEYS:
        color = target_colors[key]
        size = target_sizes[key]
        ar, ac = anchors[key]
        paste(out, np.full((size, size), color, dtype=int), ar, ac)
    for key in TARGET_KEYS:
        color = target_colors[key]
        _stamp(out, scaled[key], anchors[key][0], anchors[key][1], color)

    inp = out.copy()
    for key in TARGET_KEYS:
        q_r0, q_r1, q_c0, q_c1 = QUADS[key]
        inp[q_r0 : q_r1 + 1, q_c0 : q_c1 + 1] = BG
        color = target_colors[key]
        size = target_sizes[key]
        ar, ac = anchors[key]
        paste(inp, np.full((size, size), color, dtype=int), ar, ac)

    return inp if np.array_equal(main(inp), out) else None


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid b10624e5 D1 input")
