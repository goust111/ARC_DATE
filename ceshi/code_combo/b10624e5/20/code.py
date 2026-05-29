# Combo variant of b10624e5 (index 20)
# combo_spec: D1/0+D14/0
# Name: 干扰色块 + 多颜色锚块配对

import random

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, paste


# ---------------------------------------------------------------------------
# Design analysis for the combo variant
# ---------------------------------------------------------------------------
# design:
#   overview:
#     original_task:
#       The grid was split into four quadrants by one full divider row and one full
#       divider column. TL had a single red anchor+motif component; each other quadrant
#       had one red square anchor. The solver copied the TL motif to TR/BL/BR using:
#       target/source anchor-size ratio for scaling, and quadrant side for reflection.
#     combo_goal:
#       Keep the divider, reflection rule, and size-based scaling, but replace the
#       unique-red-anchor assumption with multi-color anchor matching (D1), and add
#       unrelated colored distractor blocks that may reuse anchor colors (D14).
#       The fused task is therefore:
#       1. find which colored squares are the real anchors,
#       2. match each target anchor to the TL source component with the same color,
#       3. ignore other same-colored blocks that are only distractors.
#   a_concretization_and_trace:
#     concrete_input:
#       - TL contains exactly 3 disconnected source components.
#       - Each source component has one 2x2 monochrome square anchor in a distinct color
#         from {2,3,8,9}, plus an attached asymmetric 4x6 motif.
#       - TR / BL / BR each contain exactly one standalone NxN anchor square, with the
#         three colors permuted across those quadrants.
#       - Additional domino / triomino distractors are scattered in TL and the target
#         quadrants. Some distractors intentionally use the same colors as anchors.
#     concrete_trace:
#       - Original logic: if TR has a red 3x3 square, solver always uses the unique TL red
#         source, flips LR, scales by 3/2, and pastes it.
#       - Combo logic: TR may have a teal 3x3 square, a separate teal 1x2 distractor, and
#         TL may also contain an isolated teal 1x3 distractor. Solver must first isolate
#         the real anchors by structure (filled squares), then match TR's teal square to
#         the TL teal anchor component, and only then perform the old reflection+scaling.
#       - Difference:
#         motif identity is no longer "the one red source"; it is determined by anchor
#         color equality, while unrelated same-colored blocks remain untouched.
#   b_complexity_calibration:
#     - Use exactly 3 source colors per puzzle so the color-matching relation is visible.
#     - Keep all three target colors distinct, one per non-TL quadrant, so color matching
#       is required in every example.
#     - Use fixed 4x6 source patches with one missing logical corner so LR / UD reflections
#       remain human-visible after scaling.
#     - Add at least one same-color distractor in every target quadrant, plus one in TL, so
#       the distractor dimension is consistently observable rather than sporadic.
#     - Distractors are non-square by construction. That keeps the task inferable: humans can
#       still discover that "solid square = anchor" instead of facing ambiguous duplicates.
#   c_implicit_assumptions_review:
#     original_main_lines_that_break:
#       - components_mask(g == 2): invalid, because anchors are no longer all red.
#       - "top-left quadrant must contain a red anchor": invalid, TL now has multiple colors.
#       - "component containing any red pixel": invalid, source extraction must become
#         per-target-color rather than per-red.
#       - np.where(patch_s == 2): invalid, alignment must use the matched anchor color.
#       - if out[...] == 2: invalid, anchor-preservation must use the target anchor color.
#     new_solver_assumptions:
#       - Real anchors are monochrome filled squares that stay within one quadrant.
#       - Each non-TL quadrant has exactly one such square anchor.
#       - TL has exactly one square source anchor for each target color.
#       - Each TL source anchor belongs to one larger non-bg connected component.
#       - Distractors may reuse anchor colors, but they are never filled squares.
#     generator_guarantees:
#       - Source motif colors are disjoint from the chosen anchor colors, so anchor-colored
#         square components remain unique despite distractors sharing those colors elsewhere.
#       - Distractors are 4-disconnected from anchors, motifs, each other, and the divider.
#       - Non-TL distractors never occupy the target motif footprint in the output.
#   d_placement_legality:
#     - TL source components occupy 3 of 4 preset slots inside the 13x13 TL quadrant, so
#       they never overlap and never cross the divider.
#     - Target anchors use the already-reflected, already-scaled matched patch to choose a
#       fit-first anchor position; full motif fit is preferred, anchor-only fit is allowed.
#     - Distractors are placed afterward using rejection sampling with a one-cell 4-neighbor
#       margin against all existing non-bg pixels in the final output.
#   e_visual_observability:
#     - Across examples, the same quadrant can host different motif families because target
#       anchor colors are permuted, so quadrant position alone is insufficient.
#     - Within one example, same-color distractors near anchors make it visible that color
#       alone is also insufficient; the square-anchor structure matters.
#     - Reflection remains observable because each source patch is asymmetric.
#   f_cleanliness:
#     - Only TL contains multiple source objects; other quadrants still stay sparse.
#     - Distractors are thin blocks instead of large blobs, so they add filtering difficulty
#       without hiding the anchor squares or making the output visually crowded.
#     - Divider color and background color are excluded from distractors.
#   g_rule_inference:
#     - No hidden ordering uses color ids; the operative cue is direct color equality between
#       source anchors and target anchors.
#     - The extra distractors do not participate in any transformation, so a human can infer:
#       "ignore loose blocks; only square anchors choose which TL motif to copy."


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
DISTRACTOR_SHAPES = [
    [(0, 0), (0, 1)],
    [(0, 0), (1, 0)],
    [(0, 0), (0, 1), (0, 2)],
    [(0, 0), (1, 0), (2, 0)],
]
MAX_RETRIES = 400


def _find_dividers(g: np.ndarray, bg: int) -> tuple[int, int]:
    """Return the unique full-row/full-col quadrant divider."""
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


def _collect_square_parts(
    g: np.ndarray, bg: int, divider_color: int, mid_r: int, mid_c: int
) -> list[dict[str, int]]:
    """Collect monochrome filled-square components that stay inside one quadrant."""
    square_parts: list[dict[str, int]] = []
    colors = sorted({int(v) for v in g.flatten()} - {int(bg), int(divider_color)})
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
    return square_parts


def main(grid: np.ndarray) -> np.ndarray:
    """
    Copy the color-matched TL source motif into each other quadrant while ignoring same-color distractors.

    Input structure:
    - One full non-bg row and one full non-bg column split the 27x27 grid into four 13x13 quadrants.
    - The top-left quadrant contains several disconnected source components; each source has a
      monochrome square anchor of its own color and a motif attached to it.
    - Each non-top-left quadrant contains exactly one monochrome square anchor whose color matches
      one TL source anchor, plus extra non-square distractor blocks that may reuse anchor colors.

    Rule:
    - Detect the real anchors as the filled square components, not as "all pixels of some color".
    - For each target quadrant, choose the TL source component whose anchor color matches the target
      anchor color.
    - Reflect that matched source by quadrant side, scale it by target-anchor size / source-anchor
      size, align the anchor-colored region onto the target square, then stamp non-bg pixels.

    Key insight: Quadrant position still controls reflection and anchor size still controls scale,
    but motif identity now comes from square-anchor color matching; loose same-colored blocks are noise.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    out = g.copy()

    bg = get_bg(g)
    mid_r, mid_c = _find_dividers(g, bg)
    divider_color = int(g[mid_r, mid_c])

    square_parts = _collect_square_parts(g, bg, divider_color, mid_r, mid_c)

    targets: dict[tuple[int, int], dict[str, int]] = {}
    target_colors: set[int] = set()
    for part in square_parts:
        key = (int(part["qb"]), int(part["qr"]))
        if key == (0, 0):
            continue
        assert key not in targets, "each non-top-left quadrant must contain exactly one square anchor"
        targets[key] = part
        target_colors.add(int(part["color"]))
    assert set(targets) == set(TARGET_KEYS), "TR/BL/BR must each contain one target anchor"
    assert len(target_colors) == len(TARGET_KEYS), "target anchor colors must be distinct"

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
        assert patch_bb is not None, "each TL source anchor must belong to one TL source component"
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
        assert np.array_equal(roundtrip, patch_t), "scaling must preserve the matched source motif"

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
    palette = extras[: random.randint(2, 3)]
    other_cells = [(r, c) for r, c in filled if (r, c) != (anchor_r, anchor_c)]
    random.shuffle(other_cells)
    for idx, (r, c) in enumerate(other_cells):
        logical[r, c] = palette[idx % len(palette)]

    patch = np.repeat(np.repeat(logical, SOURCE_ANCHOR_SIZE, axis=0), SOURCE_ANCHOR_SIZE, axis=1)
    return patch, anchor_r * SOURCE_ANCHOR_SIZE, anchor_c * SOURCE_ANCHOR_SIZE


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
    tgt_rr, tgt_cc = tgt_rr[valid], tgt_cc[valid]
    src_rr, src_cc = src_rr[valid], src_cc[valid]
    not_anchor = out[tgt_rr, tgt_cc] != anchor_color
    out[tgt_rr[not_anchor], tgt_cc[not_anchor]] = patch_s[src_rr[not_anchor], src_cc[not_anchor]]


def _can_place_cells(board: np.ndarray, cells: list[tuple[int, int]], qkey: tuple[int, int]) -> bool:
    q_r0, q_r1, q_c0, q_c1 = QUADS[qkey]
    cell_set = set(cells)
    for r, c in cells:
        if not (q_r0 <= r <= q_r1 and q_c0 <= c <= q_c1):
            return False
        if board[r, c] != BG:
            return False
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if (nr, nc) in cell_set:
                continue
            if 0 <= nr < GRID_SIZE and 0 <= nc < GRID_SIZE and board[nr, nc] != BG:
                return False
    return True


def _place_distractor(
    out: np.ndarray, inp: np.ndarray, qkey: tuple[int, int], color: int, shape: list[tuple[int, int]]
) -> bool:
    q_r0, q_r1, q_c0, q_c1 = QUADS[qkey]
    max_dr = max(r for r, _ in shape)
    max_dc = max(c for _, c in shape)
    if q_r0 + max_dr > q_r1 or q_c0 + max_dc > q_c1:
        return False
    for _ in range(80):
        r0 = random.randint(q_r0, q_r1 - max_dr)
        c0 = random.randint(q_c0, q_c1 - max_dc)
        cells = [(r0 + dr, c0 + dc) for dr, dc in shape]
        if not _can_place_cells(out, cells, qkey):
            continue
        for r, c in cells:
            out[r, c] = color
            inp[r, c] = color
        return True
    return False


def _add_distractors(
    out: np.ndarray,
    inp: np.ndarray,
    source_colors: list[int],
    target_colors: dict[tuple[int, int], int],
) -> bool:
    specs: list[tuple[tuple[int, int], int]] = [((0, 0), random.choice(source_colors))]
    specs.extend((key, target_colors[key]) for key in TARGET_KEYS)

    for qkey, color in specs:
        shape = random.choice(DISTRACTOR_SHAPES)
        if not _place_distractor(out, inp, qkey, color, shape):
            return False

    extra_count = random.randint(1, 2)
    extra_colors = source_colors + EXTRA_POOL
    for _ in range(extra_count):
        qkey = random.choice(list(QUADS))
        color = random.choice(extra_colors)
        shape = random.choice(DISTRACTOR_SHAPES)
        if not _place_distractor(out, inp, qkey, color, shape):
            return False
    return True


def _gen_attempt() -> np.ndarray | None:
    source_colors = random.sample(ANCHOR_POOL, 3)
    holes = random.sample(HOLE_CORNERS, 3)
    slots = random.sample(SOURCE_SLOTS, 3)

    source_specs: dict[int, dict[str, object]] = {}
    for color, hole, (r0, c0) in zip(source_colors, holes, slots, strict=False):
        patch, ar, ac = _make_source_patch(color, hole)
        source_specs[color] = {
            "patch": patch,
            "anchor_r": ar,
            "anchor_c": ac,
            "slot_r0": r0,
            "slot_c0": c0,
        }

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

    anchors: dict[object, tuple[int, int]] = {}
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

    if not _add_distractors(out, inp, source_colors, target_colors):
        return None

    try:
        solved = main(inp)
    except AssertionError:
        return None
    return inp if np.array_equal(solved, out) else None


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid b10624e5 combo input")
