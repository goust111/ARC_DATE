# Variant of b10624e5
# Dimension: D10 — 锚块颜色的角色 (meta)
# Change: 锚块统一为红色，颜色仅用于标识锚块身份... → 锚块颜色编码变换类型

import random

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, paste


# Design analysis for the D10 variant
# - Concrete behavior:
#   The 27x27 grid still has one divider row and one divider column crossing at (13,13).
#   The TL quadrant still contains the full source motif with a red 2x2 source anchor inside it.
#   The three target quadrants still contain only square anchors in the input, but those anchors
#   now use three different colors. Size keeps its old meaning (scaling ratio), while color gets
#   a new meaning:
#     red(2)   -> horizontal mirror
#     green(3) -> vertical mirror
#     teal(8)  -> 90-degree clockwise rotation
#   A concrete trace: in the original task, a TR anchor of size 3 always caused an LR reflection
#   because TR is on the right. In this variant, a TR anchor of size 3 can instead trigger any of
#   the three transforms; if it is teal, the motif rotates 90 degrees and then scales by 3/2.
#   This is the key behavioral difference: quadrant position no longer determines the transform.
# - Complexity calibration:
#   Only three transform colors are used, exactly one per non-TL quadrant in every sample, and
#   their placement is randomly permuted across quadrants. This is enough to visibly contradict
#   the old position-based rule without introducing an unbounded color vocabulary.
# - Hidden assumptions from the old solver that must change:
#   1. `components_mask(g == 2)` assumed every anchor was red. This would now miss green/teal
#      anchors, so anchor search must scan a fixed anchor-color codebook instead.
#   2. `if out[rr_t, cc_t] == 2: continue` only protected red anchors during stamping. In this
#      variant target anchors can be green or teal, so all anchor colors must be protected.
#   3. Quadrant position previously implied the transform. That logic must be replaced by a
#      color->transform lookup, otherwise the solver silently solves the old task instead.
# - Placement legality:
#   The generator places anchors only where the whole transformed+scaled motif fits when possible,
#   falling back to anchor-only placement if needed, matching the original permissive clipping
#   rule. Motif colors exclude the target-anchor colors, so color-coded anchors remain isolated
#   monochrome square components that the solver can recover unambiguously.
# - Visual observability:
#   Every sample uses all three target-anchor colors and randomly permutes them across TR/BL/BR.
#   This prevents a human from explaining the outputs with the old "right means LR, bottom means
#   UD" rule. The source motif is filtered to be asymmetric under LR, UD, 180, and rot90, and the
#   generator rejects samples whose realized target patches collapse back to the old behavior.
# - Visual cleanliness:
#   The grid still contains only one motif component per quadrant. Target quadrants remain sparse
#   in the input, and the output only overlays one transformed motif per anchor, preserving the
#   original task's tidy four-panel appearance.
# - Color-rule inferability:
#   The mapping uses only three recurring anchor colors, and those colors never appear as ordinary
#   motif paint in non-anchor roles. This keeps the "color encodes transform" signal explicit.


BG = 4
DIV = 1
SOURCE_ANCHOR = 2
GRID_SIZE = 27
MID = 13
QUADS = {
    (0, 0): (0, 12, 0, 12),
    (0, 1): (0, 12, 14, 26),
    (1, 0): (14, 26, 0, 12),
    (1, 1): (14, 26, 14, 26),
}

TRANSFORM_BY_COLOR = {
    2: "flip_lr",
    3: "flip_ud",
    8: "rot90",
}
ANCHOR_COLORS = tuple(sorted(TRANSFORM_BY_COLOR))
TARGET_KEYS = ((0, 1), (1, 0), (1, 1))
ORIGINAL_BY_QUADRANT = {
    (0, 1): "flip_lr",
    (1, 0): "flip_ud",
    (1, 1): "rot180",
}

SOURCE_ANCHOR_SIZE = 2
BLOCK_DIMS = [(2, 3), (3, 2), (3, 3)]
FILL_RATIO_MIN = 0.5
FILL_RATIO_MAX = 0.8
MOTIF_COLORS_MIN = 3
MOTIF_COLORS_MAX = 5
MOTIF_EXTRA_COLORS = [0, 5, 6, 7, 9]
TARGET_SIZE_RANGE = (1, 4)
MAX_RETRIES = 400


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


def _anchor_signature(patch: np.ndarray) -> tuple[tuple[int, ...], ...]:
    view = patch.copy()
    view[view == SOURCE_ANCHOR] = BG
    return tuple(tuple(int(x) for x in row) for row in view.tolist())


def _square_component(g: np.ndarray, pts: np.ndarray, color: int) -> tuple[int, int, int] | None:
    r0, c0, r1, c1 = bbox_coords(pts)
    side_r = r1 - r0 + 1
    side_c = c1 - c0 + 1
    if side_r != side_c:
        return None
    if not np.all(g[r0 : r1 + 1, c0 : c1 + 1] == color):
        return None
    return int(r0), int(c0), int(side_r)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Copy the TL motif into the other quadrants, with scale still read from anchor size and
    the geometric transform now read from anchor color.

    Input structure:
    - A single non-background full row and full column split the 27x27 board into four 13x13 quadrants.
    - The TL quadrant contains one connected colored motif with a red 2x2 source anchor embedded in it.
    - Each non-TL quadrant contains exactly one square anchor block; its side length gives the scale,
      and its color is one of the transform-code colors.

    Rule:
    - Extract the TL motif component containing the red source anchor.
    - For each other quadrant, look up the transform from the anchor color:
      red(2)=horizontal mirror, green(3)=vertical mirror, teal(8)=rotate 90 degrees clockwise.
    - Apply that transform to the source motif, scale it by target_anchor_size / source_anchor_size,
      align the transformed red region onto the target anchor, and stamp non-background pixels while
      leaving divider cells and anchor cells unchanged.

    Key insight: The anchor's size and color now split the old anchor job: size says how much to
    scale the copied motif, while color says which transform that quadrant should receive.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    out = g.copy()

    bg = get_bg(g)
    mid_r, mid_c = _find_dividers(g, bg)

    anchors: dict[tuple[int, int], tuple[int, int, int, int]] = {}
    for color in ANCHOR_COLORS:
        for pts in components_mask(g == color, connectivity=4):
            square = _square_component(g, pts, color)
            if square is None:
                continue
            r0, c0, size = square
            rr = r0 + size // 2
            cc = c0 + size // 2
            qb = 0 if rr < mid_r else 1
            qr = 0 if cc < mid_c else 1
            key = (qb, qr)
            assert key not in anchors, "each quadrant must contain exactly one anchor block"
            anchors[key] = (r0, c0, size, color)

    assert set(anchors) == {(0, 0), (0, 1), (1, 0), (1, 1)}, "must find one anchor in each quadrant"
    sr0, sc0, sL, scol = anchors[(0, 0)]
    assert scol == SOURCE_ANCHOR, "top-left source anchor must stay red"
    assert sL > 0, "top-left anchor must have positive size"

    tl = g[0:mid_r, 0:mid_c]
    patch_bb = None
    for pts in components_mask(tl != bg, connectivity=4):
        if any(tl[r, c] == SOURCE_ANCHOR for r, c in pts):
            r0, c0, r1, c1 = bbox_coords(pts)
            patch_bb = (int(r0), int(c0), int(r1), int(c1))
            break
    assert patch_bb is not None, "top-left quadrant must contain a motif component with the red source anchor"

    pr0, pc0, pr1, pc1 = patch_bb
    patch = tl[pr0 : pr1 + 1, pc0 : pc1 + 1].copy()

    for key, (tr0, tc0, tL, color) in anchors.items():
        if key == (0, 0):
            continue
        patch_t = _apply_transform(patch, TRANSFORM_BY_COLOR[color])
        ratio = float(tL) / float(sL)
        new_h = max(1, int(round(patch_t.shape[0] * ratio)))
        new_w = max(1, int(round(patch_t.shape[1] * ratio)))
        patch_s = _resize_nn(patch_t, new_h, new_w)
        roundtrip = _resize_nn(patch_s, patch_t.shape[0], patch_t.shape[1])
        assert np.array_equal(roundtrip, patch_t), "scaling must preserve all motif information"

        rr_red, cc_red = np.where(patch_s == SOURCE_ANCHOR)
        assert rr_red.size > 0, "transformed source patch must still contain the red source anchor"
        ar0, ac0 = int(rr_red.min()), int(cc_red.min())
        place_r0 = tr0 - ar0
        place_c0 = tc0 - ac0

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
        frontier.discard(cell)
        filled.add(cell)
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = cell[0] + dr, cell[1] + dc
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in filled:
                frontier.add((nr, nc))
    return filled


def _assign_colors(filled: set[tuple[int, int]], anchor_cell: tuple[int, int], h: int, w: int) -> np.ndarray | None:
    rows = {r for r, _ in filled}
    cols = {c for _, c in filled}
    if rows != set(range(h)) or cols != set(range(w)):
        return None
    non_anchor = [p for p in filled if p != anchor_cell]
    if len(non_anchor) < MOTIF_COLORS_MIN - 1:
        return None
    max_colors = min(MOTIF_COLORS_MAX, len(non_anchor) + 1)
    if max_colors < MOTIF_COLORS_MIN:
        return None
    n_colors = random.randint(MOTIF_COLORS_MIN, max_colors)
    extras = MOTIF_EXTRA_COLORS[:]
    random.shuffle(extras)
    palette = extras[: n_colors - 1]

    logical = np.full((h, w), BG, dtype=int)
    logical[anchor_cell] = SOURCE_ANCHOR
    random.shuffle(non_anchor)
    for i, (r, c) in enumerate(non_anchor):
        logical[r, c] = palette[i] if i < len(palette) else random.choice(palette)
    return logical


def _build_source_patch() -> tuple[np.ndarray, int, int] | None:
    logical_h, logical_w = random.choice(BLOCK_DIMS)
    n_cells = logical_h * logical_w
    min_fill = max(4, int(np.ceil(FILL_RATIO_MIN * n_cells)))
    max_fill = max(min_fill, int(np.floor(FILL_RATIO_MAX * n_cells)))
    anchor_cell = (random.randrange(logical_h), random.randrange(logical_w))
    filled = _grow_4conn(logical_h, logical_w, {anchor_cell}, random.randint(min_fill, max_fill))
    logical = _assign_colors(filled, anchor_cell, logical_h, logical_w)
    if logical is None:
        return None

    patch = np.repeat(np.repeat(logical, SOURCE_ANCHOR_SIZE, axis=0), SOURCE_ANCHOR_SIZE, axis=1)

    signatures = {
        op: _anchor_signature(_apply_transform(patch, op))
        for op in ("flip_lr", "flip_ud", "rot180", "rot90")
    }
    if len(set(signatures.values())) != len(signatures):
        return None

    anchor_r = anchor_cell[0] * SOURCE_ANCHOR_SIZE
    anchor_c = anchor_cell[1] * SOURCE_ANCHOR_SIZE
    return patch, anchor_r, anchor_c


def _scaled_patch(patch: np.ndarray, color: int, target_size: int) -> np.ndarray:
    op = TRANSFORM_BY_COLOR[color]
    p = _apply_transform(patch, op)
    ratio = target_size / SOURCE_ANCHOR_SIZE
    h = max(1, int(round(p.shape[0] * ratio)))
    w = max(1, int(round(p.shape[1] * ratio)))
    return _resize_nn(p, h, w)


def _choose_anchor(qb: int, qr: int, size: int, patch_s: np.ndarray) -> tuple[int, int]:
    q_r0, q_r1, q_c0, q_c1 = QUADS[(qb, qr)]
    rr, cc = np.where(patch_s == SOURCE_ANCHOR)
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


def _stamp(out: np.ndarray, patch_s: np.ndarray, anchor_r: int, anchor_c: int) -> None:
    rr_red, cc_red = np.where(patch_s == SOURCE_ANCHOR)
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
        & (tgt_rr != MID)
        & (tgt_cc != MID)
    )
    tgt_rr = tgt_rr[valid]
    tgt_cc = tgt_cc[valid]
    src_rr = src_rr[valid]
    src_cc = src_cc[valid]
    keep = ~np.isin(out[tgt_rr, tgt_cc], np.array(ANCHOR_COLORS))
    out[tgt_rr[keep], tgt_cc[keep]] = patch_s[src_rr[keep], src_cc[keep]]


def _target_sizes() -> dict[tuple[int, int], int]:
    while True:
        sizes = {(0, 0): SOURCE_ANCHOR_SIZE}
        for key in TARGET_KEYS:
            sizes[key] = random.randint(*TARGET_SIZE_RANGE)
        non_tl = [sizes[key] for key in TARGET_KEYS]
        if sum(size >= 2 for size in non_tl) >= 2 and any(size != SOURCE_ANCHOR_SIZE for size in non_tl):
            return sizes


def _variant_visible(patch: np.ndarray, target_specs: dict[tuple[int, int], tuple[int, int]]) -> bool:
    realized = {}
    for key, (size, color) in target_specs.items():
        realized[key] = _anchor_signature(_scaled_patch(patch, color, size))
    if len(set(realized.values())) != 3:
        return False

    differs_from_original = False
    for key, (size, color) in target_specs.items():
        new_sig = realized[key]
        old_patch = _apply_transform(patch, ORIGINAL_BY_QUADRANT[key])
        ratio = size / SOURCE_ANCHOR_SIZE
        old_scaled = _resize_nn(old_patch, max(1, int(round(old_patch.shape[0] * ratio))), max(1, int(round(old_patch.shape[1] * ratio))))
        old_sig = _anchor_signature(old_scaled)
        if new_sig != old_sig:
            differs_from_original = True
    return differs_from_original


def _gen_attempt() -> np.ndarray | None:
    source = _build_source_patch()
    if source is None:
        return None
    patch, anchor_r_in_patch, anchor_c_in_patch = source

    target_sizes = _target_sizes()
    target_colors = random.sample(list(ANCHOR_COLORS), k=3)
    target_specs = {
        key: (target_sizes[key], color)
        for key, color in zip(TARGET_KEYS, target_colors, strict=False)
    }

    rot90_key = next(key for key, (_size, color) in target_specs.items() if TRANSFORM_BY_COLOR[color] == "rot90")
    if target_specs[rot90_key][0] < 2:
        return None
    if not _variant_visible(patch, target_specs):
        return None

    scaled = {
        key: _scaled_patch(patch, color, size)
        for key, (size, color) in target_specs.items()
    }

    anchors: dict[tuple[int, int], tuple[int, int]] = {}
    tl_r0 = random.randint(0, MID - patch.shape[0])
    tl_c0 = random.randint(0, MID - patch.shape[1])
    anchors[(0, 0)] = (tl_r0 + anchor_r_in_patch, tl_c0 + anchor_c_in_patch)
    for key, patch_s in scaled.items():
        size, _color = target_specs[key]
        anchors[key] = _choose_anchor(key[0], key[1], size, patch_s)

    out = np.full((GRID_SIZE, GRID_SIZE), BG, dtype=int)
    out[MID, :] = DIV
    out[:, MID] = DIV
    paste(out, patch, tl_r0, tl_c0)
    for key, (size, color) in target_specs.items():
        ar, ac = anchors[key]
        paste(out, np.full((size, size), color, dtype=int), ar, ac)
    for key, patch_s in scaled.items():
        _stamp(out, patch_s, *anchors[key])

    inp = out.copy()
    for key, (size, color) in target_specs.items():
        q_r0, q_r1, q_c0, q_c1 = QUADS[key]
        inp[q_r0 : q_r1 + 1, q_c0 : q_c1 + 1] = BG
        ar, ac = anchors[key]
        paste(inp, np.full((size, size), color, dtype=int), ar, ac)

    return inp if np.array_equal(main(inp), out) else None


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid D10 b10624e5 input")
