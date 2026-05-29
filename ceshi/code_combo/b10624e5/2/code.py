# Combo variant of b10624e5 (index 2)
# combo_spec: D1/4+D5/0
# Name: 位置对齐匹配 + 旋转替代反射

import random

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, paste


# Design analysis for this combo variant
# --------------------------------------
# a) Concrete behavior trace:
#    - The original task always used the TL motif as the source, scaled by anchor-size ratio,
#      then chose a reflection from quadrant relation alone.
#    - This combo keeps the same source motif and scaling, but changes the control signal:
#      each target is classified by its red anchor's alignment with the TL red anchor.
#      same top edge -> "row" role, same left edge -> "col" role, neither -> "diag" role.
#    - The old D1/4 meaning would have mapped those three roles to horizontal / vertical /
#      double reflection. After fusion with D5/0, those roles instead map to rotations:
#      row -> 90 deg clockwise, col -> 270 deg clockwise, diag -> 180 deg.
#    - Example trace with a 4x6 asymmetric patch:
#      old row-aligned target: fliplr keeps a 4x6 footprint;
#      new row-aligned target: rot90 makes it 6x4 and moves colored mass below/left of the
#      red block. The same change happens for the col-aligned target with the opposite turn.
#
# b) Complexity calibration:
#    - The geometry of the original puzzle fixes the source in TL, so row-aligned / col-aligned /
#      diagonal targets still end up in TR / BL / BR respectively. That means a shortcut solver
#      could still key off quadrants. To keep the intended combo cue visible anyway, the generator
#      enforces exact top-edge and left-edge alignments so the row/col signal is obvious to humans.
#    - The generator also rejects samples where the rotation output matches the legacy reflection
#      output in TR or BL, so the rotation replacement is visually observable instead of nominal.
#
# c) Implicit assumptions review:
#    - Divider detection and TL-source extraction remain valid.
#    - The original loop assumed quadrant relation directly selected reflection axes; that branch
#      is replaced by an alignment classifier plus a role->rotation table.
#    - Red-anchor alignment by the transformed patch's red bbox still works because rotating a
#      square red block preserves a contiguous square red block.
#    - The roundtrip resize assertion still matters; generator output is accepted only when the
#      sampled source motif survives the chosen scale ratios.
#
# d) Placement legality:
#    - The generator uses output-first construction.
#    - It chooses TL anchor coordinates from the intersection of:
#      1) TL patch-fit bounds,
#      2) TR full-fit bounds with the same anchor row,
#      3) BL full-fit bounds with the same anchor column.
#    - All non-TL anchors are then placed from full-fit ranges, so motifs stay inside quadrants
#      without overlap or divider collisions.
#
# e) Visual observability:
#    - Accepted samples must satisfy:
#      1) TR anchor shares the TL anchor's top edge,
#      2) BL anchor shares the TL anchor's left edge,
#      3) the combo output differs from the legacy reflection output in both TR and BL.
#    - Non-square source patches are preferred so 90/270 degree turns are easy to spot.
#
# f) Cleanliness:
#    - Non-TL inputs still contain only red anchors, so each output motif appears in isolation.
#    - Full-fit placement avoids clipped fragments that would make the rotation cue harder to read.
#
# g) New solver assumptions introduced here:
#    - Exactly one red anchor per quadrant.
#    - TL is still the unique source quadrant.
#    - At least one non-source anchor is row-aligned, at least one is col-aligned, and the rest
#      are diagonal relative to the TL anchor when using anchor top-left coordinates.
#    - The generator enforces the strongest form of that assumption: one row role in TR, one col
#      role in BL, one diagonal role in BR, all using exact top-left alignment where applicable.


ROTATION_K_BY_RELATION = {
    "row": 3,   # 90 degrees clockwise; np.rot90 is CCW
    "col": 1,   # 270 degrees clockwise
    "diag": 2,  # 180 degrees
}

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
RELATION_BY_QUADRANT = {
    (0, 1): "row",
    (1, 0): "col",
    (1, 1): "diag",
}

SOURCE_ANCHOR_SIZE = 2
BLOCK_DIMS = [(2, 2), (2, 3), (3, 2), (3, 3)]
BLOCK_PROB = 0.5
FILL_RATIO_MIN = 0.5
FILL_RATIO_MAX = 0.8
FREEFORM_H_RANGE = (3, 6)
FREEFORM_W_RANGE = (3, 6)
MOTIF_COLORS_MIN = 3
MOTIF_COLORS_MAX = 5
COLOR_POOL = [0, 1, 3, 5, 6, 7, 8, 9]
TARGET_SIZE_RANGE = (1, 4)
MAX_RETRIES = 700


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


def _resize_nn(img: np.ndarray, new_h: int, new_w: int) -> np.ndarray:
    src_h, src_w = img.shape
    if new_h <= 0 or new_w <= 0:
        return img.copy()
    rr = np.clip(np.floor(np.arange(new_h) * src_h / new_h).astype(int), 0, src_h - 1)
    cc = np.clip(np.floor(np.arange(new_w) * src_w / new_w).astype(int), 0, src_w - 1)
    return img[rr[:, None], cc[None, :]]


def _transform_rotation(patch: np.ndarray, relation: str) -> np.ndarray:
    return np.rot90(patch, ROTATION_K_BY_RELATION[relation])


def _transform_legacy_reflection(patch: np.ndarray, relation: str) -> np.ndarray:
    if relation == "row":
        return np.fliplr(patch)
    if relation == "col":
        return np.flipud(patch)
    return np.flipud(np.fliplr(patch))


def _extract_anchors(g: np.ndarray, mid_r: int, mid_c: int) -> dict[tuple[int, int], tuple[int, int, int]]:
    red = components_mask(g == RED, connectivity=4)
    assert red, "grid must contain red (color 2) anchor blocks"

    anchors: dict[tuple[int, int], tuple[int, int, int]] = {}
    for pts in red:
        r0, c0, r1, c1 = bbox_coords(pts)
        rr = (r0 + r1) // 2
        cc = (c0 + c1) // 2
        qb = 0 if rr < mid_r else 1
        qr = 0 if cc < mid_c else 1
        size = max(r1 - r0 + 1, c1 - c0 + 1)
        key = (qb, qr)
        assert key not in anchors, "each quadrant must contain exactly one red anchor block"
        anchors[key] = (int(r0), int(c0), int(size))

    assert len(anchors) == 4, "grid must contain exactly four quadrant anchors"
    return anchors


def _extract_source_patch(g: np.ndarray, bg: int, mid_r: int, mid_c: int) -> np.ndarray:
    tl = g[0:mid_r, 0:mid_c]
    comps = components_mask(tl != bg, connectivity=4)
    patch_bb = None
    for pts in comps:
        if any(tl[r, c] == RED for r, c in pts):
            r0, c0, r1, c1 = bbox_coords(pts)
            patch_bb = (int(r0), int(c0), int(r1), int(c1))
            break
    assert patch_bb is not None, "top-left quadrant must contain a non-bg component with a red pixel"
    pr0, pc0, pr1, pc1 = patch_bb
    return tl[pr0 : pr1 + 1, pc0 : pc1 + 1].copy()


def _relation_from_alignment(src_anchor: tuple[int, int], dst_anchor: tuple[int, int]) -> str:
    same_row = int(dst_anchor[0]) == int(src_anchor[0])
    same_col = int(dst_anchor[1]) == int(src_anchor[1])
    if same_row and same_col:
        raise AssertionError("non-source anchor cannot share both row and column with the source anchor")
    if same_row:
        return "row"
    if same_col:
        return "col"
    return "diag"


def _scaled_patch(patch: np.ndarray, relation: str, target_size: int, *, use_rotation: bool) -> np.ndarray:
    transformed = _transform_rotation(patch, relation) if use_rotation else _transform_legacy_reflection(patch, relation)
    ratio = target_size / SOURCE_ANCHOR_SIZE
    new_h = max(1, int(round(transformed.shape[0] * ratio)))
    new_w = max(1, int(round(transformed.shape[1] * ratio)))
    return _resize_nn(transformed, new_h, new_w)


def _stamp(out: np.ndarray, patch_s: np.ndarray, anchor_r: int, anchor_c: int) -> None:
    rr_red, cc_red = np.where(patch_s == RED)
    assert rr_red.size > 0, "scaled source patch must contain red anchor pixels"
    place_r0 = int(anchor_r) - int(rr_red.min())
    place_c0 = int(anchor_c) - int(cc_red.min())

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
    not_red = out[tgt_rr, tgt_cc] != RED
    out[tgt_rr[not_red], tgt_cc[not_red]] = patch_s[src_rr[not_red], src_cc[not_red]]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Replicate the top-left motif by anchor-size scaling, but choose rotations from anchor alignment.

    Input structure:
    - A single full row and full column of one non-background color divide the board into four quadrants.
    - Every quadrant contains one red square anchor block; the top-left quadrant also contains the full colored motif.
    - The top-left motif is a single non-background component that includes its red anchor, while the other quadrants initially show only their red anchors.

    Rule:
    - Extract the top-left motif as the source patch.
    - For each other quadrant, compare its anchor's top-left corner to the top-left anchor's top-left corner:
      same row means the "row" target, same column means the "col" target, otherwise it is the diagonal target.
    - Scale the source patch by the target/source anchor-size ratio, then rotate it according to that role:
      row -> 90 degrees clockwise, col -> 270 degrees clockwise, diagonal -> 180 degrees.
    - Align the transformed patch's red block to the target anchor and stamp non-background pixels without overwriting divider cells or existing red pixels.

    Key insight: Anchor size still gives the scale factor, but the transform is no longer read from quadrant mirrors; it is inferred from visible row/column alignment with the source anchor.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    bg = get_bg(g)
    mid_r, mid_c = _find_dividers(g, bg)
    anchors = _extract_anchors(g, mid_r, mid_c)

    assert (0, 0) in anchors, "top-left quadrant must contain a red anchor block"
    sr0, sc0, sL = anchors[(0, 0)]
    assert sL > 0, "top-left red anchor must have positive size"
    patch = _extract_source_patch(g, bg, mid_r, mid_c)

    for key, (tr0, tc0, tL) in anchors.items():
        if key == (0, 0):
            continue
        relation = _relation_from_alignment((sr0, sc0), (tr0, tc0))
        patch_t = _transform_rotation(patch, relation)
        new_h = max(1, int(round(patch_t.shape[0] * float(tL) / float(sL))))
        new_w = max(1, int(round(patch_t.shape[1] * float(tL) / float(sL))))
        patch_s = _resize_nn(patch_t, new_h, new_w)
        roundtrip = _resize_nn(patch_s, patch_t.shape[0], patch_t.shape[1])
        assert np.array_equal(roundtrip, patch_t), "scaling must preserve all motif information"
        _stamp(out, patch_s, tr0, tc0)

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


def _assign_colors(
    filled: set[tuple[int, int]], red_cells: set[tuple[int, int]], h: int, w: int
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


def _full_fit_ranges(
    qkey: tuple[int, int], size: int, patch_s: np.ndarray
) -> tuple[int, int, int, int] | None:
    q_r0, q_r1, q_c0, q_c1 = QUADS[qkey]
    rr, cc = np.where(patch_s == RED)
    red_r0, red_c0 = int(rr.min()), int(cc.min())

    fit_r0 = max(q_r0 + red_r0, q_r0)
    fit_r1 = min(q_r1 - patch_s.shape[0] + 1 + red_r0, q_r1 - size + 1)
    fit_c0 = max(q_c0 + red_c0, q_c0)
    fit_c1 = min(q_c1 - patch_s.shape[1] + 1 + red_c0, q_c1 - size + 1)
    if fit_r0 > fit_r1 or fit_c0 > fit_c1:
        return None
    return fit_r0, fit_r1, fit_c0, fit_c1


def _build_output(
    patch: np.ndarray,
    tl_r0: int,
    tl_c0: int,
    target_sizes: dict[tuple[int, int], int],
    anchors: dict[tuple[int, int], tuple[int, int]],
    scaled: dict[tuple[int, int], np.ndarray],
) -> np.ndarray:
    out = np.full((GRID_SIZE, GRID_SIZE), BG, dtype=int)
    out[MID, :] = DIV
    out[:, MID] = DIV
    paste(out, patch, tl_r0, tl_c0)
    for key, size in target_sizes.items():
        if key == (0, 0):
            continue
        ar, ac = anchors[key]
        paste(out, np.full((size, size), RED, dtype=int), ar, ac)
    for key, patch_s in scaled.items():
        _stamp(out, patch_s, *anchors[key])
    return out


def _build_input(
    out: np.ndarray,
    target_sizes: dict[tuple[int, int], int],
    anchors: dict[tuple[int, int], tuple[int, int]],
) -> np.ndarray:
    inp = out.copy()
    for key, size in target_sizes.items():
        if key == (0, 0):
            continue
        q_r0, q_r1, q_c0, q_c1 = QUADS[key]
        inp[q_r0 : q_r1 + 1, q_c0 : q_c1 + 1] = BG
        ar, ac = anchors[key]
        paste(inp, np.full((size, size), RED, dtype=int), ar, ac)
    return inp


def _diff_stats(combo_out: np.ndarray, legacy_out: np.ndarray) -> dict[str, int]:
    stats: dict[str, int] = {}
    for key, relation in RELATION_BY_QUADRANT.items():
        r0, r1, c0, c1 = QUADS[key]
        q_combo = combo_out[r0 : r1 + 1, c0 : c1 + 1]
        q_legacy = legacy_out[r0 : r1 + 1, c0 : c1 + 1]
        diff = (q_combo != q_legacy) & (q_combo != RED) & (q_legacy != RED)
        stats[relation] = int(diff.sum())
    return stats


def _observable_combo(stats: dict[str, int]) -> bool:
    return stats["row"] > 0 and stats["col"] > 0


def _gen_attempt() -> np.ndarray | None:
    target_sizes = {
        (0, 0): SOURCE_ANCHOR_SIZE,
        (0, 1): random.randint(*TARGET_SIZE_RANGE),
        (1, 0): random.randint(*TARGET_SIZE_RANGE),
        (1, 1): random.randint(*TARGET_SIZE_RANGE),
    }

    needs_block = any(size % SOURCE_ANCHOR_SIZE != 0 for key, size in target_sizes.items() if key != (0, 0))
    source = _source_patch(needs_block)
    if source is None:
        return None
    patch, anchor_r_in_patch, anchor_c_in_patch = source

    if patch.shape[0] == patch.shape[1]:
        return None

    scaled_combo = {
        key: _scaled_patch(patch, RELATION_BY_QUADRANT[key], size, use_rotation=True)
        for key, size in target_sizes.items()
        if key != (0, 0)
    }
    scaled_legacy = {
        key: _scaled_patch(patch, RELATION_BY_QUADRANT[key], size, use_rotation=False)
        for key, size in target_sizes.items()
        if key != (0, 0)
    }

    row_fit = _full_fit_ranges((0, 1), target_sizes[(0, 1)], scaled_combo[(0, 1)])
    col_fit = _full_fit_ranges((1, 0), target_sizes[(1, 0)], scaled_combo[(1, 0)])
    diag_fit = _full_fit_ranges((1, 1), target_sizes[(1, 1)], scaled_combo[(1, 1)])
    if row_fit is None or col_fit is None or diag_fit is None:
        return None

    src_r_lo = max(anchor_r_in_patch, row_fit[0])
    src_r_hi = min(MID - patch.shape[0] + anchor_r_in_patch, row_fit[1])
    src_c_lo = max(anchor_c_in_patch, col_fit[2])
    src_c_hi = min(MID - patch.shape[1] + anchor_c_in_patch, col_fit[3])
    if src_r_lo > src_r_hi or src_c_lo > src_c_hi:
        return None

    src_r0 = random.randint(src_r_lo, src_r_hi)
    src_c0 = random.randint(src_c_lo, src_c_hi)
    tl_r0 = src_r0 - anchor_r_in_patch
    tl_c0 = src_c0 - anchor_c_in_patch

    anchors: dict[tuple[int, int], tuple[int, int]] = {
        (0, 0): (src_r0, src_c0),
        (0, 1): (src_r0, random.randint(row_fit[2], row_fit[3])),
        (1, 0): (random.randint(col_fit[0], col_fit[1]), src_c0),
        (1, 1): (random.randint(diag_fit[0], diag_fit[1]), random.randint(diag_fit[2], diag_fit[3])),
    }

    out_combo = _build_output(patch, tl_r0, tl_c0, target_sizes, anchors, scaled_combo)
    out_legacy = _build_output(patch, tl_r0, tl_c0, target_sizes, anchors, scaled_legacy)
    if not _observable_combo(_diff_stats(out_combo, out_legacy)):
        return None

    inp = _build_input(out_combo, target_sizes, anchors)

    # Final structural sanity checks for the intended row/column cue.
    if anchors[(0, 1)][0] != anchors[(0, 0)][0]:
        return None
    if anchors[(1, 0)][1] != anchors[(0, 0)][1]:
        return None
    if anchors[(1, 1)][0] == anchors[(0, 0)][0] or anchors[(1, 1)][1] == anchors[(0, 0)][1]:
        return None

    return inp if np.array_equal(main(inp), out_combo) else None


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid b10624e5 combo input")
