# Combo variant of b10624e5 (index 8)
# combo_spec: D1/0+D15/0
# Name: 多颜色锚块配对 + 多源多目标一一配对

import random

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, paste


# ---------------------------------------------------------------------------
# Design analysis for the combo variant
# ---------------------------------------------------------------------------
# design:
#   a_concretization_and_trace:
#     input_scene:
#       - The 27x27 cross divider remains.
#       - Two source quadrants live in the left column: TL and BL.
#       - Each source quadrant contains one connected motif component with its own colored NxN
#         anchor square. The two source anchor colors are different and their motif shapes differ.
#       - Two target quadrants live in the right column: TR and BR.
#       - Each target quadrant contains only one colored square anchor, and each target color
#         matches exactly one source color.
#     behavior_trace:
#       - Original task: read the unique TL red source once, then copy it to every other quadrant.
#       - D1 alone would say: choose a source by matching anchor colors.
#       - D15 alone would say: there are two independent transfers instead of one source feeding all.
#       - Combo here: first match target -> source by visible anchor color equality; then, for that
#         matched pair only, reflect according to the relative source/target quadrants and scale by
#         target_size / matched_source_size. Because the two source anchors can have different sizes,
#         color matching also decides which scale denominator applies.
#       - Difference from the original is therefore not cosmetic: source identity, pairing count,
#         and the scale reference all depend on the chosen source.
#   b_complexity_calibration:
#     - Use exactly 2 source colors and 2 targets so the one-to-one structure is readable.
#     - Keep sources on the left and targets on the right to avoid clutter, but randomize top/bottom
#       placement of size ranks and target colors so row position is not a hidden pairing cue.
#     - Force the target-size ranking to disagree with the source-size ranking. This blocks a degenerate
#       size-only solver and makes anchor color genuinely necessary.
#     - Use a tiny library of asymmetric 3x3 logical motifs whose non-anchor colors form L-shapes,
#       so the anchor is the only filled square inside each source component.
#   c_implicit_assumptions_review:
#     original_assumptions_that_break:
#       - "all anchors are red" is invalid because anchors now use multiple colors.
#       - "the only source is TL" is invalid because BL can also be a source.
#       - "quadrant alone identifies the source motif" is invalid because target color selects source.
#       - "align using red pixels" is invalid because each pair aligns on its own anchor color.
#     combo_specific_solver_requirements:
#       - Each quadrant must contain exactly one non-bg connected component in the input.
#       - Inside each component there must be exactly one monochrome filled square, which is the anchor.
#       - Anchor colors are unique identifiers: each target color appears in exactly one source quadrant.
#     generator_guarantees:
#       - Sources use motif colors that exclude both source anchor colors, so the anchor color is unique
#         inside its component.
#       - Non-anchor color regions are connected L-shapes, not filled squares, so anchor detection is stable.
#       - Right-column targets contain only the colored anchor square.
#   d_placement_legality:
#     - Logical motifs are 3x3 and scale by anchor size 1..4, so every transformed patch is at most 12x12
#       and fits inside a 13x13 quadrant.
#     - Source patches are placed fully inside their quadrants.
#     - Target anchors are chosen with a fit-first strategy using the already transformed patch.
#     - Stamping still clips nothing important, skips the divider, and never overwrites the target anchor color.
#   e_visual_observability:
#     - Across examples, the same source row can map to TR or BR depending on the matched color, so users
#       can see that row is not the pairing rule.
#     - Because each source motif is visibly different and target colors repeat the source colors, the
#       one-to-one color pairing is inferable from 2-3 examples.
#     - Vertical reflection is observable whenever the matched source and target sit in different rows,
#       which happens under random size-rank placements.
#   f_cleanliness:
#     - Left column contains exactly one object per quadrant; right column inputs stay sparse.
#     - Divider color is never reused in motifs.
#     - Only two anchor colors appear, keeping the color mapping legible.
#   g_change_inferability:
#     - No hidden numeric ordering of color ids is used.
#     - Color equality is the pairing cue; size only affects scaling once the color-selected source is known.
#   h_post_main_assumptions_review:
#     - `main()` now assumes every quadrant has exactly one connected component and one anchor square.
#       `_gen_attempt()` constructs that explicitly.
#     - `main()` assumes the two source colors and two target colors match bijectively.
#       `_gen_attempt()` builds targets by reusing the source anchor colors exactly once each.
#     - `main()` assumes a wrong size-based pairing would differ from the intended pairing often enough
#       to make the combo non-degenerate. `_gen_attempt()` enforces opposite size-rank pairing so that
#       the color rule is behaviorally necessary.


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
SOURCE_KEYS = [(0, 0), (1, 0)]
TARGET_KEYS = [(0, 1), (1, 1)]
ANCHOR_POOL = [2, 3, 8, 9]
MOTIF_COLOR_POOL = [0, 2, 3, 5, 6, 7, 8, 9]
SOURCE_SIZE_CHOICES = [1, 2, 3]
TARGET_SIZE_CHOICES = [1, 2, 3, 4]
MAX_RETRIES = 400

TEMPLATES = [
    np.array([[-1, 1, 1], [2, 2, 1], [0, 2, 0]], dtype=int),
    np.array([[1, 1, -1], [1, 2, 2], [0, 2, 0]], dtype=int),
    np.array([[0, 1, -1], [1, 1, 2], [0, 2, 2]], dtype=int),
    np.array([[0, -1, 1], [2, 1, 1], [2, 2, 0]], dtype=int),
    np.array([[0, 2, -1], [1, 2, 2], [1, 1, 0]], dtype=int),
    np.array([[0, 2, 2], [1, 1, 2], [-1, 1, 0]], dtype=int),
]


def _find_dividers(g: np.ndarray, bg: int) -> tuple[int, int]:
    """Return the unique full-row/full-col quadrant divider: (mid_row, mid_col)."""
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


def _quadrant_bounds(key: tuple[int, int]) -> tuple[int, int, int, int]:
    q_r0, q_r1, q_c0, q_c1 = QUADS[key]
    return q_r0, q_r1 + 1, q_c0, q_c1 + 1


def _resize_nn(img: np.ndarray, new_h: int, new_w: int) -> np.ndarray:
    src_h, src_w = img.shape
    if new_h <= 0 or new_w <= 0:
        return img.copy()
    rr = np.clip(np.floor(np.arange(new_h) * src_h / new_h).astype(int), 0, src_h - 1)
    cc = np.clip(np.floor(np.arange(new_w) * src_w / new_w).astype(int), 0, src_w - 1)
    return img[rr[:, None], cc[None, :]]


def _canonical_signature(arr: np.ndarray) -> tuple:
    variants = []
    for t in (arr, np.fliplr(arr), np.flipud(arr), np.flipud(np.fliplr(arr))):
        variants.append((t.shape[0], t.shape[1], tuple(map(int, t.flatten()))))
    return min(variants)


def _extract_unique_component(quad: np.ndarray, bg: int) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    comps = components_mask(quad != bg, connectivity=4)
    assert len(comps) == 1, "each quadrant must contain exactly one non-bg component"
    r0, c0, r1, c1 = bbox_coords(comps[0])
    return quad[r0 : r1 + 1, c0 : c1 + 1].copy(), (int(r0), int(c0), int(r1), int(c1))


def _find_square_anchor(patch: np.ndarray, bg: int) -> dict[str, int]:
    candidates: list[dict[str, int]] = []
    for color in sorted(set(map(int, patch.flatten())) - {int(bg)}):
        for pts in components_mask(patch == color, connectivity=4):
            r0, c0, r1, c1 = bbox_coords(pts)
            size_h = int(r1 - r0 + 1)
            size_w = int(c1 - c0 + 1)
            if size_h != size_w:
                continue
            if len(pts) != size_h * size_w:
                continue
            candidates.append(
                {
                    "color": int(color),
                    "size": int(size_h),
                    "r0": int(r0),
                    "c0": int(c0),
                }
            )
    assert len(candidates) == 1, "each component must contain exactly one filled square anchor"
    return candidates[0]


def _transform_patch(
    patch: np.ndarray,
    src_key: tuple[int, int],
    dst_key: tuple[int, int],
    source_size: int,
    target_size: int,
) -> np.ndarray:
    p = patch
    if src_key[1] != dst_key[1]:
        p = np.fliplr(p)
    if src_key[0] != dst_key[0]:
        p = np.flipud(p)
    ratio = float(target_size) / float(source_size)
    new_h = max(1, int(round(p.shape[0] * ratio)))
    new_w = max(1, int(round(p.shape[1] * ratio)))
    patch_s = _resize_nn(p, new_h, new_w)
    roundtrip = _resize_nn(patch_s, p.shape[0], p.shape[1])
    assert np.array_equal(roundtrip, p), "scaling must preserve all motif information"
    return patch_s


def _stamp(
    out: np.ndarray,
    patch_s: np.ndarray,
    anchor_r: int,
    anchor_c: int,
    anchor_color: int,
    bg: int,
    mid_r: int,
    mid_c: int,
) -> None:
    rr_anchor, cc_anchor = np.where(patch_s == anchor_color)
    assert rr_anchor.size > 0, "transformed patch must retain the matched anchor color"
    place_r0 = int(anchor_r) - int(rr_anchor.min())
    place_c0 = int(anchor_c) - int(cc_anchor.min())
    src_rr, src_cc = np.where(patch_s != bg)
    for rr0, cc0 in zip(src_rr.tolist(), src_cc.tolist(), strict=False):
        rr_t = place_r0 + rr0
        cc_t = place_c0 + cc0
        if not (0 <= rr_t < out.shape[0] and 0 <= cc_t < out.shape[1]):
            continue
        if rr_t == mid_r or cc_t == mid_c:
            continue
        if out[rr_t, cc_t] == anchor_color:
            continue
        out[rr_t, cc_t] = int(patch_s[rr0, cc0])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Transfer two color-keyed source motifs to their matching targets inside the quadrant frame.

    Input structure:
    - One non-bg full row and one non-bg full column divide the grid into four quadrants.
    - Two quadrants contain full source components: a colored square anchor with an attached motif.
    - The other two quadrants contain only standalone colored square anchors, and each target color
      matches exactly one source anchor color.

    Rule:
    - In every quadrant, find the unique square anchor inside the unique non-bg component.
    - Classify a quadrant as a source when its component contains non-anchor pixels beyond the square.
    - Pair each target with the source of the same anchor color, reflect by the relative source/target
      row and column positions, scale by target_anchor_size / matched_source_anchor_size, then align
      the transformed anchor-colored region onto the target anchor and stamp the non-bg pixels.

    Key insight:
    - Anchor color decides which source feeds each target, and that same choice also fixes the source
      anchor size used in the scaling ratio, so color matching and geometric transfer are fused.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    bg = get_bg(g)
    mid_r, mid_c = _find_dividers(g, bg)

    info_by_key: dict[tuple[int, int], dict[str, object]] = {}
    for key in QUADS:
        q_r0, q_r1, q_c0, q_c1 = _quadrant_bounds(key)
        quad = g[q_r0:q_r1, q_c0:q_c1]
        patch, (pr0, pc0, _, _) = _extract_unique_component(quad, bg)
        anchor = _find_square_anchor(patch, bg)
        info_by_key[key] = {
            "patch": patch,
            "color": int(anchor["color"]),
            "size": int(anchor["size"]),
            "anchor_r0": q_r0 + pr0 + int(anchor["r0"]),
            "anchor_c0": q_c0 + pc0 + int(anchor["c0"]),
            "is_source": int(np.count_nonzero(patch != bg)) > int(anchor["size"]) * int(anchor["size"]),
        }

    sources = [key for key, info in info_by_key.items() if bool(info["is_source"])]
    targets = [key for key, info in info_by_key.items() if not bool(info["is_source"])]
    assert len(sources) == 2 and len(targets) == 2, "variant requires exactly two sources and two targets"

    source_by_color: dict[int, tuple[int, int]] = {}
    target_by_color: dict[int, tuple[int, int]] = {}
    for key in sources:
        color = int(info_by_key[key]["color"])
        assert color not in source_by_color, "source anchor colors must be distinct"
        source_by_color[color] = key
    for key in targets:
        color = int(info_by_key[key]["color"])
        assert color not in target_by_color, "target anchor colors must be distinct"
        target_by_color[color] = key
    assert set(source_by_color) == set(target_by_color), "target colors must bijectively match source colors"

    for color in sorted(source_by_color):
        src_key = source_by_color[color]
        dst_key = target_by_color[color]
        src_info = info_by_key[src_key]
        dst_info = info_by_key[dst_key]
        patch_s = _transform_patch(
            np.asarray(src_info["patch"], dtype=int),
            src_key,
            dst_key,
            int(src_info["size"]),
            int(dst_info["size"]),
        )
        _stamp(
            out,
            patch_s,
            int(dst_info["anchor_r0"]),
            int(dst_info["anchor_c0"]),
            int(color),
            bg,
            mid_r,
            mid_c,
        )

    return out


def _is_asymmetric(arr: np.ndarray) -> bool:
    return not (
        np.array_equal(arr, np.fliplr(arr))
        or np.array_equal(arr, np.flipud(arr))
        or np.array_equal(arr, np.flipud(np.fliplr(arr)))
    )


def _make_source_patch(anchor_color: int, anchor_size: int, template: np.ndarray, fill_colors: list[int]) -> tuple[np.ndarray, int, int]:
    logical = np.full(template.shape, BG, dtype=int)
    logical[template == -1] = anchor_color
    logical[template == 1] = fill_colors[0]
    logical[template == 2] = fill_colors[1]
    ar, ac = np.argwhere(template == -1)[0]
    patch = np.repeat(np.repeat(logical, anchor_size, axis=0), anchor_size, axis=1)
    return patch, int(ar * anchor_size), int(ac * anchor_size)


def _choose_patch_top_left(key: tuple[int, int], patch: np.ndarray) -> tuple[int, int]:
    q_r0, q_r1, q_c0, q_c1 = QUADS[key]
    max_r0 = q_r1 - patch.shape[0] + 1
    max_c0 = q_c1 - patch.shape[1] + 1
    assert max_r0 >= q_r0 and max_c0 >= q_c0, "source patch must fit inside its quadrant"
    return random.randint(q_r0, max_r0), random.randint(q_c0, max_c0)


def _choose_anchor(key: tuple[int, int], size: int, patch_s: np.ndarray, anchor_color: int) -> tuple[int, int]:
    q_r0, q_r1, q_c0, q_c1 = QUADS[key]
    rr, cc = np.where(patch_s == anchor_color)
    anchor_r0 = int(rr.min())
    anchor_c0 = int(cc.min())

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


def _gen_attempt() -> np.ndarray | None:
    source_colors = random.sample(ANCHOR_POOL, 2)
    source_sizes = sorted(random.sample(SOURCE_SIZE_CHOICES, 2))
    target_sizes = sorted(random.sample(TARGET_SIZE_CHOICES, 2))

    source_keys_by_rank = random.sample(SOURCE_KEYS, 2)
    source_size_by_key = {
        source_keys_by_rank[0]: source_sizes[0],
        source_keys_by_rank[1]: source_sizes[1],
    }
    source_color_by_key = {
        SOURCE_KEYS[0]: source_colors[0],
        SOURCE_KEYS[1]: source_colors[1],
    }
    source_key_by_color = {color: key for key, color in source_color_by_key.items()}
    color_small_source = source_color_by_key[source_keys_by_rank[0]]
    color_large_source = source_color_by_key[source_keys_by_rank[1]]

    target_keys_by_rank = random.sample(TARGET_KEYS, 2)
    target_size_by_key = {
        target_keys_by_rank[0]: target_sizes[0],
        target_keys_by_rank[1]: target_sizes[1],
    }
    target_color_by_key = {
        target_keys_by_rank[0]: color_large_source,
        target_keys_by_rank[1]: color_small_source,
    }

    extra_pool = [c for c in MOTIF_COLOR_POOL if c not in source_colors and c not in {BG, DIV}]
    template_sigs: set[tuple] = set()
    source_specs: dict[tuple[int, int], dict[str, object]] = {}
    for key in SOURCE_KEYS:
        choices = [template for template in TEMPLATES if _is_asymmetric(template)]
        random.shuffle(choices)
        chosen_template = None
        for template in choices:
            sig = _canonical_signature(template)
            if sig in template_sigs:
                continue
            chosen_template = template
            template_sigs.add(sig)
            break
        if chosen_template is None:
            return None
        fill_colors = random.sample(extra_pool, 2)
        if random.random() < 0.5:
            fill_colors.reverse()
        patch, ar, ac = _make_source_patch(
            source_color_by_key[key],
            source_size_by_key[key],
            chosen_template,
            fill_colors,
        )
        source_specs[key] = {
            "patch": patch,
            "anchor_r_in_patch": ar,
            "anchor_c_in_patch": ac,
            "size": source_size_by_key[key],
            "color": source_color_by_key[key],
        }

    pair_specs: list[dict[str, object]] = []
    for dst_key in TARGET_KEYS:
        color = target_color_by_key[dst_key]
        src_key = source_key_by_color[color]
        src_spec = source_specs[src_key]
        patch_s = _transform_patch(
            np.asarray(src_spec["patch"], dtype=int),
            src_key,
            dst_key,
            int(src_spec["size"]),
            target_size_by_key[dst_key],
        )
        pair_specs.append(
            {
                "src_key": src_key,
                "dst_key": dst_key,
                "color": color,
                "target_size": target_size_by_key[dst_key],
                "patch_s": patch_s,
            }
        )

    anchors: dict[tuple[int, int], tuple[int, int]] = {}
    source_patch_topleft: dict[tuple[int, int], tuple[int, int]] = {}
    for src_key, spec in source_specs.items():
        pr0, pc0 = _choose_patch_top_left(src_key, np.asarray(spec["patch"], dtype=int))
        source_patch_topleft[src_key] = (pr0, pc0)
        anchors[src_key] = (
            pr0 + int(spec["anchor_r_in_patch"]),
            pc0 + int(spec["anchor_c_in_patch"]),
        )

    for pair in pair_specs:
        dst_key = pair["dst_key"]
        anchors[dst_key] = _choose_anchor(
            dst_key,
            int(pair["target_size"]),
            np.asarray(pair["patch_s"], dtype=int),
            int(pair["color"]),
        )

    out = np.full((GRID_SIZE, GRID_SIZE), BG, dtype=int)
    out[MID, :] = DIV
    out[:, MID] = DIV

    for src_key, spec in source_specs.items():
        pr0, pc0 = source_patch_topleft[src_key]
        paste(out, np.asarray(spec["patch"], dtype=int), pr0, pc0)

    for dst_key in TARGET_KEYS:
        color = target_color_by_key[dst_key]
        size = target_size_by_key[dst_key]
        paste(out, np.full((size, size), color, dtype=int), anchors[dst_key][0], anchors[dst_key][1])

    for pair in pair_specs:
        _stamp(
            out,
            np.asarray(pair["patch_s"], dtype=int),
            anchors[pair["dst_key"]][0],
            anchors[pair["dst_key"]][1],
            int(pair["color"]),
            BG,
            MID,
            MID,
        )

    inp = out.copy()
    for dst_key in TARGET_KEYS:
        q_r0, q_r1, q_c0, q_c1 = QUADS[dst_key]
        inp[q_r0 : q_r1 + 1, q_c0 : q_c1 + 1] = BG
        color = target_color_by_key[dst_key]
        size = target_size_by_key[dst_key]
        paste(inp, np.full((size, size), color, dtype=int), anchors[dst_key][0], anchors[dst_key][1])

    if not np.array_equal(main(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid b10624e5 combo input")
