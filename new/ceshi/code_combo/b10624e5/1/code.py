# Combo variant of b10624e5 (index 1)
# combo_spec: D1/0+D10/0
# Name: 多颜色锚块分组配对 + 锚块颜色编码变换类型

import random

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, paste


# ---------------------------------------------------------------------------
# Design analysis for the combo variant
# ---------------------------------------------------------------------------
# design:
#   overview:
#     original_task:
#       TL had exactly one red anchor+motif component. Each non-TL quadrant had one red square
#       anchor. Solver only needed one source patch, target size for scaling, and quadrant
#       position for reflection.
#     combo_goal:
#       Replace the single TL source with a 3-item TL catalog keyed by anchor color (D1),
#       and reuse the same anchor color as a transform code for targets (D10). The same visible
#       color now carries two meanings in sequence:
#         1. choose which TL source component to copy
#         2. choose how that matched source should be transformed
#       This is not two independent tweaks glued together; target processing is explicitly
#       color -> source match -> color-coded transform -> scale -> align.
#   a_concretization_and_trace:
#     input:
#       - 27x27 grid with divider row/col at index 13, making four 13x13 quadrants.
#       - TL contains 3 disconnected source components. Their anchor colors are fixed to
#         red(2), green(3), teal(8). Each anchor is a 2x2 solid square attached to its own motif.
#       - TR / BL / BR each contain exactly one standalone NxN anchor square, using those same
#         3 colors in a random permutation.
#     output:
#       - For a target quadrant with color 8, solver must first select the TL teal source patch.
#       - Then it must apply teal's transform code, rot90 clockwise.
#       - Then it scales by target_size / 2, aligns the teal anchor region, and stamps.
#     concrete_trace:
#       - Original: BR always copied the lone TL red source and always used rot180 because BR is
#         bottom-right.
#       - Combo: BR might instead contain a green 4x4 anchor. Solver must ignore BR position as
#         the source identity cue, match green to the TL green source patch, then apply green's
#         transform code (flip_ud), not BR's old rot180 rule, before scaling 4/2.
#       - Difference: source identity and transform type are both color-driven, but they operate
#         at different stages. Matching cannot be solved from geometry alone, and transform cannot
#         be solved from quadrant alone.
#   b_complexity_calibration:
#     - Use exactly 3 anchor colors and exactly one target of each color, so the codebook is
#       small enough for humans to infer from 2-3 examples.
#     - Keep TL source patches in fixed-size 4x6 physical boxes built from a 2x3 logical pattern.
#       This keeps scaling robust for target sizes 1..4 and keeps the visual layout clean.
#     - Force the 3 TL source patches to have distinct masked signatures, so color matching is
#       behaviorally necessary instead of redundant.
#     - Reject samples where color-coded transforms collapse to the old quadrant-based behavior.
#   c_implicit_assumptions_review:
#     original_main_lines_that_break:
#       - `components_mask(g == 2)` assumed all anchors were red. Now anchors can be 2/3/8.
#       - "the TL component containing red" assumed a single TL source. Now TL holds a catalog.
#       - Reflection by quadrant assumed transform came from position. Now transform comes from
#         target anchor color.
#       - Alignment on red pixels assumed every anchor used red. Now alignment must use the
#         matched anchor color.
#       - Skipping only red pixels during stamping is unsafe; generator now guarantees all anchor
#         colors are drawn from a dedicated codebook and motif paint avoids those colors.
#     new_solver_assumptions:
#       - Only colors {2,3,8} act as anchors.
#       - Each of those colors appears exactly once as a TL source anchor square and exactly once
#         as a non-TL target anchor square.
#       - Each TL source component contains exactly one anchor square of its own color.
#       - Motif colors never reuse anchor colors, so anchor squares stay isolatable.
#     generator_guarantees:
#       - Non-TL inputs contain only the target anchor squares.
#       - TL source components are disconnected and placed in separated slots.
#       - Source patches are asymmetric enough that their assigned transform is observable.
#   d_placement_legality:
#     - TL sources use three non-overlapping preset slots inside the 13x13 TL quadrant.
#     - Target anchors use a fit-first placement strategy driven by the already transformed and
#       scaled matched patch. Full fit is preferred; anchor-only fit is the fallback.
#     - Stamping still clips at grid bounds and skips divider cells, matching the original task.
#   e_visual_observability:
#     - Each sample shows all three anchor colors twice: once in TL as source labels and once in
#       non-TL as target labels. This makes the color-matching relation directly visible.
#     - The target colors are permuted across TR/BL/BR, so position alone cannot explain motif
#       identity. BR also never uses its old rot180 transform, so position alone cannot explain
#       geometry either.
#     - Because the same color appears in TL and in exactly one target quadrant, humans can infer
#       the color->transform rule by comparing each source with its output counterpart.
#   f_cleanliness:
#     - Only TL is dense; target quadrants stay sparse in the input and receive one overlay each
#       in the output.
#     - Motif fill colors exclude divider color and anchor colors, keeping the codebook visually
#       isolated and preventing accidental square-anchor decoys.
#   g_color_rule_inference:
#     - The color vocabulary is intentionally capped at three anchor colors:
#         red(2)   -> flip_lr
#         green(3) -> flip_ud
#         teal(8)  -> rot90 clockwise
#     - No numeric ordering of color ids is used. The inference cue is visible repetition:
#       same target color matches the same TL source color and always carries the same transform.


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
TARGET_KEYS = ((0, 1), (1, 0), (1, 1))

SOURCE_ANCHOR_SIZE = 2
ANCHOR_COLORS = (2, 3, 8)
TRANSFORM_BY_COLOR = {
    2: "flip_lr",
    3: "flip_ud",
    8: "rot90",
}
ORIGINAL_BY_QUADRANT = {
    (0, 1): "flip_lr",
    (1, 0): "flip_ud",
    (1, 1): "rot180",
}

TARGET_SIZE_RANGE = (1, 4)
EXTRA_POOL = [0, 5, 6, 7, 9]
HOLE_CORNERS = [(0, 0), (0, 2), (1, 0), (1, 2)]
SOURCE_SLOTS = [(0, 0), (0, 7), (8, 0), (8, 7)]
MAX_RETRIES = 500


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


def _apply_transform(patch: np.ndarray, op: str) -> np.ndarray:
    if op == "flip_lr":
        return np.fliplr(patch)
    if op == "flip_ud":
        return np.flipud(patch)
    if op == "rot180":
        return np.rot90(patch, 2)
    if op == "rot90":
        return np.rot90(patch, -1)
    raise ValueError(f"unknown transform {op}")


def _square_component(g: np.ndarray, pts: np.ndarray, color: int) -> tuple[int, int, int] | None:
    r0, c0, r1, c1 = bbox_coords(pts)
    side_r = r1 - r0 + 1
    side_c = c1 - c0 + 1
    if side_r != side_c:
        return None
    if not np.all(g[r0 : r1 + 1, c0 : c1 + 1] == color):
        return None
    return int(r0), int(c0), int(side_r)


def _mask_anchor_signature(patch: np.ndarray, anchor_color: int) -> tuple[tuple[int, ...], ...]:
    view = patch.copy()
    view[view == anchor_color] = BG
    return tuple(tuple(int(x) for x in row) for row in view.tolist())


def _mask_codebook_signature(patch: np.ndarray) -> tuple[tuple[int, ...], ...]:
    view = patch.copy()
    for color in ANCHOR_COLORS:
        view[view == color] = BG
    return tuple(tuple(int(x) for x in row) for row in view.tolist())


def _scaled_patch(patch: np.ndarray, color: int, target_size: int) -> np.ndarray:
    op = TRANSFORM_BY_COLOR[color]
    transformed = _apply_transform(patch, op)
    ratio = target_size / SOURCE_ANCHOR_SIZE
    new_h = max(1, int(round(transformed.shape[0] * ratio)))
    new_w = max(1, int(round(transformed.shape[1] * ratio)))
    return _resize_nn(transformed, new_h, new_w)


def _choose_anchor(qb: int, qr: int, size: int, patch_s: np.ndarray, anchor_color: int) -> tuple[int, int]:
    q_r0, q_r1, q_c0, q_c1 = QUADS[(qb, qr)]
    rr, cc = np.where(patch_s == anchor_color)
    anchor_r0, anchor_c0 = int(rr.min()), int(cc.min())

    fit_r0 = q_r0 + anchor_r0
    fit_r1 = q_r1 - patch_s.shape[0] + 1 + anchor_r0
    fit_c0 = q_c0 + anchor_c0
    fit_c1 = q_c1 - patch_s.shape[1] + 1 + anchor_c0

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
    tgt_rr = tgt_rr[valid]
    tgt_cc = tgt_cc[valid]
    src_rr = src_rr[valid]
    src_cc = src_cc[valid]
    keep = ~np.isin(out[tgt_rr, tgt_cc], np.array(ANCHOR_COLORS))
    out[tgt_rr[keep], tgt_cc[keep]] = patch_s[src_rr[keep], src_cc[keep]]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Replicate color-matched TL source motifs into the other quadrants, with anchor color
    controlling both which source to copy and which transform to apply.

    Input structure:
    - A single non-background full row and full column split the 27x27 board into four
      13x13 quadrants.
    - The top-left quadrant contains three disconnected source components; each has a 2x2
      square anchor of one code color and an attached motif.
    - Each non-top-left quadrant contains exactly one square anchor block. Its size gives the
      scale factor, and its color is one of the same code colors used in TL.

    Rule:
    - For each target quadrant, first find the TL source component whose anchor color matches
      the target anchor color.
    - Then read the transform from that same color:
      red(2)=flip left-right, green(3)=flip up-down, teal(8)=rotate 90 degrees clockwise.
    - Apply the color-selected transform to the matched source, scale it by
      target_anchor_size / source_anchor_size, align the transformed anchor-colored region onto
      the target anchor, and stamp non-background pixels while leaving divider cells and anchor
      cells unchanged.

    Key insight: One visible attribute, anchor color, now drives a two-step reasoning chain.
    It first selects the correct TL motif family, then specifies the geometric transform for
    that matched family; anchor size still independently controls scale.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    out = g.copy()

    bg = get_bg(g)
    mid_r, mid_c = _find_dividers(g, bg)

    square_parts: list[dict[str, int]] = []
    for color in ANCHOR_COLORS:
        for pts in components_mask(g == color, connectivity=4):
            square = _square_component(g, pts, color)
            if square is None:
                continue
            r0, c0, size = square
            r1 = r0 + size - 1
            c1 = c0 + size - 1
            if r0 <= mid_r <= r1 or c0 <= mid_c <= c1:
                continue
            qb, qr = _quadrant_key(mid_r, mid_c, r0, c0, r1, c1)
            square_parts.append(
                {
                    "color": int(color),
                    "r0": int(r0),
                    "c0": int(c0),
                    "size": int(size),
                    "qb": int(qb),
                    "qr": int(qr),
                }
            )

    targets: dict[tuple[int, int], dict[str, int]] = {}
    target_colors: set[int] = set()
    source_anchors: dict[int, dict[str, int]] = {}
    for part in square_parts:
        key = (int(part["qb"]), int(part["qr"]))
        color = int(part["color"])
        if key == (0, 0):
            assert color not in source_anchors, "each anchor color must appear once in TL"
            source_anchors[color] = part
        else:
            assert key not in targets, "each non-top-left quadrant must contain exactly one anchor square"
            targets[key] = part
            target_colors.add(color)

    assert set(targets) == set(TARGET_KEYS), "TR/BL/BR must each contain one target anchor"
    assert target_colors == set(ANCHOR_COLORS), "targets must use the full anchor color codebook"
    assert set(source_anchors) == set(ANCHOR_COLORS), "TL must contain one source anchor for each code color"

    tl = g[0:mid_r, 0:mid_c]
    tl_components = components_mask(tl != bg, connectivity=4)
    tl_entries: list[tuple[set[tuple[int, int]], tuple[int, int, int, int]]] = []
    for pts in tl_components:
        r0, c0, r1, c1 = bbox_coords(pts)
        tl_entries.append(({(int(r), int(c)) for r, c in pts}, (int(r0), int(c0), int(r1), int(c1))))
    assert len(tl_entries) == 3, "TL must contain exactly three disconnected source components"

    source_patches: dict[int, tuple[np.ndarray, int]] = {}
    for color, anchor in source_anchors.items():
        anchor_pixel = (int(anchor["r0"]), int(anchor["c0"]))
        patch_bb = None
        for ptset, bb in tl_entries:
            if anchor_pixel in ptset:
                patch_bb = bb
                break
        assert patch_bb is not None, "each TL source anchor must belong to one source component"
        pr0, pc0, pr1, pc1 = patch_bb
        patch = tl[pr0 : pr1 + 1, pc0 : pc1 + 1].copy()
        source_patches[color] = (patch, int(anchor["size"]))

    for key, target in targets.items():
        color = int(target["color"])
        tr0 = int(target["r0"])
        tc0 = int(target["c0"])
        t_size = int(target["size"])
        patch, s_size = source_patches[color]

        patch_t = _apply_transform(patch, TRANSFORM_BY_COLOR[color])
        ratio = float(t_size) / float(s_size)
        new_h = max(1, int(round(patch_t.shape[0] * ratio)))
        new_w = max(1, int(round(patch_t.shape[1] * ratio)))
        patch_s = _resize_nn(patch_t, new_h, new_w)
        roundtrip = _resize_nn(patch_s, patch_t.shape[0], patch_t.shape[1])
        assert np.array_equal(roundtrip, patch_t), "scaling must preserve the matched transformed motif"

        rr_anchor, cc_anchor = np.where(patch_s == color)
        assert rr_anchor.size > 0, "scaled patch must retain the matched anchor color"
        place_r0 = tr0 - int(rr_anchor.min())
        place_c0 = tc0 - int(cc_anchor.min())

        rr_p, cc_p = np.where(patch_s != bg)
        for rr0, cc0 in zip(rr_p.tolist(), cc_p.tolist(), strict=False):
            rr_t = place_r0 + rr0
            cc_t = place_c0 + cc0
            if not (0 <= rr_t < h and 0 <= cc_t < w):
                continue
            if rr_t == mid_r or cc_t == mid_c:
                continue
            if int(out[rr_t, cc_t]) in ANCHOR_COLORS:
                continue
            out[rr_t, cc_t] = int(patch_s[rr0, cc0])

    return out


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


def _target_sizes(target_colors: dict[tuple[int, int], int]) -> dict[tuple[int, int], int] | None:
    sizes = {(0, 0): SOURCE_ANCHOR_SIZE}
    for key in TARGET_KEYS:
        sizes[key] = random.randint(*TARGET_SIZE_RANGE)
    if all(sizes[key] == SOURCE_ANCHOR_SIZE for key in TARGET_KEYS):
        return None
    rot90_key = next(key for key, color in target_colors.items() if TRANSFORM_BY_COLOR[color] == "rot90")
    if sizes[rot90_key] == 1:
        return None
    return sizes


def _matching_visible(
    source_specs: dict[int, dict[str, object]],
    target_colors: dict[tuple[int, int], int],
    target_sizes: dict[tuple[int, int], int],
) -> bool:
    for key in TARGET_KEYS:
        color = target_colors[key]
        size = target_sizes[key]
        actual_sig = _mask_codebook_signature(_scaled_patch(source_specs[color]["patch"], color, size))
        alt_sigs = {
            _mask_codebook_signature(_scaled_patch(source_specs[other]["patch"], color, size))
            for other in ANCHOR_COLORS
            if other != color
        }
        if actual_sig in alt_sigs:
            return False
    return True


def _transform_visible(
    source_specs: dict[int, dict[str, object]],
    target_colors: dict[tuple[int, int], int],
    target_sizes: dict[tuple[int, int], int],
) -> bool:
    mismatches = 0
    for key in TARGET_KEYS:
        color = target_colors[key]
        size = target_sizes[key]
        patch = source_specs[color]["patch"]
        actual_sig = _mask_codebook_signature(_scaled_patch(patch, color, size))
        old_patch = _apply_transform(patch, ORIGINAL_BY_QUADRANT[key])
        ratio = size / SOURCE_ANCHOR_SIZE
        old_scaled = _resize_nn(
            old_patch,
            max(1, int(round(old_patch.shape[0] * ratio))),
            max(1, int(round(old_patch.shape[1] * ratio))),
        )
        old_sig = _mask_codebook_signature(old_scaled)
        if actual_sig != old_sig:
            mismatches += 1
    return mismatches >= 2


def _source_catalog_valid(source_specs: dict[int, dict[str, object]]) -> bool:
    signatures = set()
    for color, spec in source_specs.items():
        patch = spec["patch"]
        sig = _mask_codebook_signature(patch)
        signatures.add(sig)
        transformed_sig = _mask_codebook_signature(_apply_transform(patch, TRANSFORM_BY_COLOR[color]))
        if transformed_sig == sig:
            return False
    return len(signatures) == len(ANCHOR_COLORS)


def _gen_attempt() -> np.ndarray | None:
    holes = random.sample(HOLE_CORNERS, len(ANCHOR_COLORS))
    slots = random.sample(SOURCE_SLOTS, len(ANCHOR_COLORS))

    source_specs: dict[int, dict[str, object]] = {}
    for color, hole, (r0, c0) in zip(ANCHOR_COLORS, holes, slots, strict=False):
        patch, anchor_r, anchor_c = _make_source_patch(color, hole)
        source_specs[color] = {
            "patch": patch,
            "anchor_r": anchor_r,
            "anchor_c": anchor_c,
            "slot_r0": r0,
            "slot_c0": c0,
        }
    if not _source_catalog_valid(source_specs):
        return None

    permuted_colors = random.sample(list(ANCHOR_COLORS), len(ANCHOR_COLORS))
    target_colors = {key: color for key, color in zip(TARGET_KEYS, permuted_colors, strict=False)}
    target_sizes = _target_sizes(target_colors)
    if target_sizes is None:
        return None

    if not _matching_visible(source_specs, target_colors, target_sizes):
        return None
    if not _transform_visible(source_specs, target_colors, target_sizes):
        return None

    scaled = {
        key: _scaled_patch(source_specs[target_colors[key]]["patch"], target_colors[key], target_sizes[key])
        for key in TARGET_KEYS
    }

    anchors: dict[tuple[int, int] | tuple[int, int, int], tuple[int, int]] = {}
    for color, spec in source_specs.items():
        anchors[(0, 0, color)] = (
            int(spec["slot_r0"]) + int(spec["anchor_r"]),
            int(spec["slot_c0"]) + int(spec["anchor_c"]),
        )
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
    raise RuntimeError("failed to generate a valid combo b10624e5 input")
