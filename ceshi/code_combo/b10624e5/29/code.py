# Combo variant of b10624e5 (index 29)
# combo_spec: D1/1+D9/1
# Name: L形锚块 + 锚块形状匹配配对

import itertools
import random

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, paste


# Design analysis for the combo variant
# -------------------------------------
# spec:
#   preserved_backbone:
#     - 27x27 grid, bg=4, one full-row/full-col divider at index 13, yielding
#       four 13x13 quadrants.
#     - TL quadrant contains the full source motif component.
#     - Non-TL input quadrants contain only one red anchor each.
#     - Output still mirrors by quadrant and scales before stamping.
#   fused_change:
#     - D9/1 replaces square anchors with irregular L anchors.
#     - D1/1 replaces "every red anchor receives a copy" with "only anchors whose
#       shape signature matches the TL anchor receive a copy".
#     - All anchors are uniform integer scalings of one 3x3 logical L prototype,
#       so the bbox side ratio is a valid and unambiguous scale factor even though
#       the visible anchor is non-rectangular.
#   concrete_behavior:
#     - Example: TL uses a bottom-left L. TR may show the same bottom-left L at
#       scale 2, BL a top-right L at scale 2, BR the same bottom-left L at scale 3.
#     - Original solver would have treated all three targets as generic red blocks
#       and pasted into every quadrant based only on size.
#     - Combo solver first normalizes each red component into a 3x3 logical mask,
#       keeps only exact shape matches to TL, then scales and reflects the TL motif.
#   complexity_calibration:
#     - Anchor families are limited to four corner-L signatures and scales 1..3.
#       This is enough to force shape recognition without creating a huge latent
#       rule table.
#     - Each sample guarantees both support and contrast: at least one matching
#       target and at least one non-matching distractor are present.
#     - A distractor is forced to share a scale with a matching target, so size
#       alone cannot explain which quadrants receive the motif.
#   hidden_assumption_audit:
#     - Original `size = max(...)` assumed the anchor class could be collapsed to
#       one scalar. That loses the new pairing signal, so the solver must retain a
#       normalized shape signature.
#     - Original "paste into every non-TL quadrant" is now wrong; shape filtering
#       must happen before scaling/stamping.
#     - Alignment by red bbox still works only because generator constrains every
#       anchor to be a clean integer scaling of the same 3x3 logical lattice.
#     - Round-trip nearest-neighbor recovery is safe because all generated scales
#       are integers and the source patch is built on that same lattice.
#   placement_legality:
#     - The source motif is 4x4 logical cells; with max scale 3 it becomes 12x12,
#       so it still fits inside one 13x13 quadrant.
#     - Each quadrant contains exactly one red component, and all non-red source
#       pixels are 4-connected to that anchor component.
#     - Placement helpers choose anchor positions that prefer full motif fit while
#       still allowing the bare anchor to fit if the scaled motif would clip.
#   observability_and_cleanliness:
#     - Because every anchor is red and every shape is some kind of L, the solver
#       must use visible geometry rather than color or quadrant order.
#     - The source motif always has asymmetric non-red frontier cells so mirrored
#       copies are visibly different across quadrants.
#     - Distractor quadrants stay uncluttered: they contain just one bare L anchor.


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

SOURCE_SCALE = 1
LOGICAL_ANCHOR_SIZE = 3
SOURCE_PATCH_SIZE = 4
TARGET_SCALE_RANGE = (1, 3)
COLOR_POOL = [0, 3, 5, 6, 7, 8, 9]
MAX_RETRIES = 500

L_SHAPES = {
    "bl": np.array(
        [
            [1, 0, 0],
            [1, 0, 0],
            [1, 1, 1],
        ],
        dtype=int,
    ),
    "br": np.array(
        [
            [0, 0, 1],
            [0, 0, 1],
            [1, 1, 1],
        ],
        dtype=int,
    ),
    "tl": np.array(
        [
            [1, 1, 1],
            [1, 0, 0],
            [1, 0, 0],
        ],
        dtype=int,
    ),
    "tr": np.array(
        [
            [1, 1, 1],
            [0, 0, 1],
            [0, 0, 1],
        ],
        dtype=int,
    ),
}
SIGNATURES = [tuple(tuple(int(v) for v in row) for row in arr.tolist()) for arr in L_SHAPES.values()]


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


def _resize_nn(img: np.ndarray, new_h: int, new_w: int) -> np.ndarray:
    src_h, src_w = img.shape
    if new_h <= 0 or new_w <= 0:
        return img.copy()
    rr = np.clip(np.floor(np.arange(new_h) * src_h / new_h).astype(int), 0, src_h - 1)
    cc = np.clip(np.floor(np.arange(new_w) * src_w / new_w).astype(int), 0, src_w - 1)
    return img[rr[:, None], cc[None, :]]


def _anchor_signature(pts: np.ndarray) -> tuple[tuple[int, ...], ...]:
    """Normalize one scaled L anchor into its 3x3 logical occupancy mask."""
    r0, c0, r1, c1 = bbox_coords(pts)
    h = int(r1 - r0 + 1)
    w = int(c1 - c0 + 1)
    assert h == w and h % LOGICAL_ANCHOR_SIZE == 0, "anchor bbox must be square with side divisible by 3"
    k = h // LOGICAL_ANCHOR_SIZE

    mask = np.zeros((h, w), dtype=bool)
    for r, c in pts.tolist():
        mask[int(r - r0), int(c - c0)] = True

    rows = []
    for br in range(LOGICAL_ANCHOR_SIZE):
        row = []
        for bc in range(LOGICAL_ANCHOR_SIZE):
            block = mask[br * k : (br + 1) * k, bc * k : (bc + 1) * k]
            full = bool(np.all(block))
            empty = not bool(np.any(block))
            assert full or empty, "anchors must be uniform on each logical block"
            row.append(1 if full else 0)
        rows.append(tuple(row))
    signature = tuple(rows)
    assert signature in SIGNATURES, "anchor must be one of the supported L prototypes"
    return signature


def _anchor_patch(signature: tuple[tuple[int, ...], ...], scale: int) -> np.ndarray:
    logical = np.array(signature, dtype=int)
    return np.kron(logical, np.full((scale, scale), RED, dtype=int))


def _build_source_patch(signature: tuple[tuple[int, ...], ...]) -> np.ndarray | None:
    """
    Create a 4x4 source motif whose red subshape is the selected 3x3 L prototype.
    Extra colored cells are chosen from the red frontier so the whole motif stays
    4-connected and its 4x4 bbox plus asymmetry are visible after reflection.
    """
    patch = np.full((SOURCE_PATCH_SIZE, SOURCE_PATCH_SIZE), BG, dtype=int)
    anchor = np.array(signature, dtype=int)
    patch[:LOGICAL_ANCHOR_SIZE, :LOGICAL_ANCHOR_SIZE][anchor == 1] = RED

    frontier = []
    for r in range(SOURCE_PATCH_SIZE):
        for c in range(SOURCE_PATCH_SIZE):
            if patch[r, c] != BG:
                continue
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < SOURCE_PATCH_SIZE and 0 <= nc < SOURCE_PATCH_SIZE and patch[nr, nc] == RED:
                    frontier.append((r, c))
                    break

    candidates = []
    for cells in itertools.combinations(frontier, 3):
        rows = {r for r, _ in cells}
        cols = {c for _, c in cells}
        if 3 not in rows or 3 not in cols:
            continue
        probe = patch.copy()
        for r, c in cells:
            probe[r, c] = 9
        if np.array_equal(probe, np.fliplr(probe)):
            continue
        if np.array_equal(probe, np.flipud(probe)):
            continue
        candidates.append(cells)

    if not candidates:
        return None

    extra_cells = random.choice(candidates)
    palette = COLOR_POOL[:]
    random.shuffle(palette)
    n_extra_colors = random.randint(2, 3)
    extra_colors = palette[:n_extra_colors]
    for idx, (r, c) in enumerate(extra_cells):
        patch[r, c] = extra_colors[idx % n_extra_colors]
    return patch


def _scaled_patch(patch: np.ndarray, qb: int, qr: int, target_scale: int) -> np.ndarray:
    p = patch
    if qr == 1:
        p = np.fliplr(p)
    if qb == 1:
        p = np.flipud(p)
    new_h = patch.shape[0] * target_scale
    new_w = patch.shape[1] * target_scale
    return _resize_nn(p, new_h, new_w)


def _choose_anchor_for_patch(qb: int, qr: int, patch_s: np.ndarray) -> tuple[int, int]:
    """Choose the red-anchor bbox top-left, preferring placements where the whole motif fits."""
    q_r0, q_r1, q_c0, q_c1 = QUADS[(qb, qr)]
    rr, cc = np.where(patch_s == RED)
    red_r0, red_c0 = int(rr.min()), int(cc.min())
    red_r1, red_c1 = int(rr.max()), int(cc.max())
    anchor_h = red_r1 - red_r0 + 1
    anchor_w = red_c1 - red_c0 + 1

    fit_r0 = q_r0 + red_r0
    fit_r1 = q_r1 - patch_s.shape[0] + 1 + red_r0
    fit_c0 = q_c0 + red_c0
    fit_c1 = q_c1 - patch_s.shape[1] + 1 + red_c0

    anch_r0, anch_r1 = q_r0, q_r1 - anchor_h + 1
    anch_c0, anch_c1 = q_c0, q_c1 - anchor_w + 1

    r_lo, r_hi = max(fit_r0, anch_r0), min(fit_r1, anch_r1)
    c_lo, c_hi = max(fit_c0, anch_c0), min(fit_c1, anch_c1)
    if r_lo <= r_hi and c_lo <= c_hi:
        return random.randint(r_lo, r_hi), random.randint(c_lo, c_hi)
    return random.randint(anch_r0, anch_r1), random.randint(anch_c0, anch_c1)


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
    Copy the TL motif only onto quadrants whose irregular red anchor has the same L signature.

    Input structure:
    - A single-color full row and full column divide the board into four quadrants.
    - Every quadrant contains exactly one red anchor, but anchors are no longer solid squares:
      each is a scaled 3x3 L-shaped prototype.
    - The top-left quadrant additionally contains one colored motif component attached to its red anchor.

    Rule:
    - Extract the TL motif as the source patch.
    - Normalize each red anchor into a 3x3 logical signature and keep only targets whose signature
      exactly matches the TL anchor's signature.
    - For each matching target, reflect the source patch by quadrant, scale it by the target/source
      anchor bbox ratio, align the scaled red region to the target anchor, and stamp all non-bg pixels.

    Key insight:
    - Red color alone no longer determines the destinations; the exact irregular anchor shape does.
      The anchor's bbox still encodes the integer scale once the matching shape class is identified.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    out = g.copy()

    bg = get_bg(g)
    mid_r, mid_c = _find_dividers(g, bg)

    red = components_mask(g == RED, connectivity=4)
    assert red, "grid must contain red anchors"

    anchors: dict[tuple[int, int], dict[str, object]] = {}
    for pts in red:
        r0, c0, r1, c1 = bbox_coords(pts)
        rr = (r0 + r1) // 2
        cc = (c0 + c1) // 2
        qb = 0 if rr < mid_r else 1
        qr = 0 if cc < mid_c else 1
        key = (qb, qr)
        assert key not in anchors, "each quadrant must contain exactly one red anchor"
        signature = _anchor_signature(pts)
        anchors[key] = {
            "r0": int(r0),
            "c0": int(c0),
            "side": int(r1 - r0 + 1),
            "signature": signature,
        }

    assert (0, 0) in anchors, "top-left quadrant must contain a red anchor"
    source_anchor = anchors[(0, 0)]
    source_side = int(source_anchor["side"])
    source_signature = source_anchor["signature"]

    tl = g[0:mid_r, 0:mid_c]
    comps = components_mask(tl != bg, connectivity=4)
    patch_bb = None
    for pts in comps:
        if any(tl[r, c] == RED for r, c in pts.tolist()):
            r0, c0, r1, c1 = bbox_coords(pts)
            patch_bb = (int(r0), int(c0), int(r1), int(c1))
            break
    assert patch_bb is not None, "top-left quadrant must contain a non-bg component with a red anchor"

    pr0, pc0, pr1, pc1 = patch_bb
    patch = tl[pr0 : pr1 + 1, pc0 : pc1 + 1].copy()

    for (qb, qr), anchor in anchors.items():
        if (qb, qr) == (0, 0):
            continue
        if anchor["signature"] != source_signature:
            continue

        ratio = int(anchor["side"]) // source_side
        assert ratio >= 1 and ratio * source_side == int(anchor["side"]), "target scale must be an integer"
        patch_s = _scaled_patch(patch, qb, qr, ratio)
        roundtrip = _resize_nn(patch_s, patch.shape[0], patch.shape[1])
        assert np.array_equal(roundtrip, _scaled_patch(patch, qb, qr, 1)), "scaling must preserve the motif information"

        rr_red, cc_red = np.where(patch_s == RED)
        assert rr_red.size > 0, "scaled patch must retain the red anchor"
        place_r0 = int(anchor["r0"]) - int(rr_red.min())
        place_c0 = int(anchor["c0"]) - int(cc_red.min())

        rr_p, cc_p = np.where(patch_s != bg)
        for rr0, cc0 in zip(rr_p.tolist(), cc_p.tolist(), strict=False):
            val = int(patch_s[rr0, cc0])
            rr_t = place_r0 + rr0
            cc_t = place_c0 + cc0
            if not (0 <= rr_t < h and 0 <= cc_t < w):
                continue
            if rr_t == mid_r or cc_t == mid_c:
                continue
            if out[rr_t, cc_t] == RED:
                continue
            out[rr_t, cc_t] = val

    return out


def _gen_attempt() -> np.ndarray | None:
    source_signature = random.choice(SIGNATURES)
    source_patch = _build_source_patch(source_signature)
    if source_patch is None:
        return None

    other_keys = [(0, 1), (1, 0), (1, 1)]
    matched_count = random.choice([1, 2])
    matched_keys = set(random.sample(other_keys, matched_count))
    distractor_keys = [key for key in other_keys if key not in matched_keys]
    assert distractor_keys, "generator must keep at least one distractor quadrant"

    quadrant_shapes: dict[tuple[int, int], tuple[tuple[int, ...], ...]] = {(0, 0): source_signature}
    distractor_choices = [sig for sig in SIGNATURES if sig != source_signature]
    for key in matched_keys:
        quadrant_shapes[key] = source_signature
    for key in distractor_keys:
        quadrant_shapes[key] = random.choice(distractor_choices)

    scales = {(0, 0): SOURCE_SCALE}
    for key in other_keys:
        scales[key] = random.randint(*TARGET_SCALE_RANGE)
    if all(scales[key] == SOURCE_SCALE for key in matched_keys):
        bump = random.choice(list(matched_keys))
        scales[bump] = random.choice([2, 3])
    shared_distractor = random.choice(distractor_keys)
    shared_match = random.choice(list(matched_keys))
    scales[shared_distractor] = scales[shared_match]

    scaled_patches = {
        key: _scaled_patch(source_patch, key[0], key[1], scales[key])
        for key in matched_keys
    }
    anchor_imgs = {
        key: _anchor_patch(quadrant_shapes[key], scales[key])
        for key in [(0, 0), *other_keys]
    }

    tl_r0 = random.randint(0, MID - source_patch.shape[0])
    tl_c0 = random.randint(0, MID - source_patch.shape[1])
    anchors: dict[tuple[int, int], tuple[int, int]] = {}
    for key in other_keys:
        if key in matched_keys:
            anchors[key] = _choose_anchor_for_patch(key[0], key[1], scaled_patches[key])
        else:
            anchors[key] = _choose_anchor_for_patch(key[0], key[1], anchor_imgs[key])

    out = np.full((GRID_SIZE, GRID_SIZE), BG, dtype=int)
    out[MID, :] = DIV
    out[:, MID] = DIV
    paste(out, source_patch, tl_r0, tl_c0)
    for key in other_keys:
        ar, ac = anchors[key]
        paste(out, anchor_imgs[key], ar, ac)
    for key in matched_keys:
        _stamp(out, scaled_patches[key], *anchors[key])

    inp = out.copy()
    for key in other_keys:
        q_r0, q_r1, q_c0, q_c1 = QUADS[key]
        inp[q_r0 : q_r1 + 1, q_c0 : q_c1 + 1] = BG
        ar, ac = anchors[key]
        paste(inp, anchor_imgs[key], ar, ac)

    return inp if np.array_equal(main(inp), out) else None


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid b10624e5 combo input")
