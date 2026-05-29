# Variant of b10624e5
# Dimension: D15 — 源与目标的基数关系 (meta)
# Change: 1个源象限（左上）对多个目标象限（1:many），源图案是唯一的... → 单一目标接收多个源
#
# Design analysis
# a) Concrete behavior:
#    - The divider and one-red-square-per-quadrant language from the original task stay intact.
#    - The inversion is implemented by making exactly one quadrant the sink: it contains only a red
#      4x4 anchor in the input, while the other three quadrants each contain a 4-connected colored
#      component attached to their own red 4x4 anchor.
#    - Each visible source component is an exact 2x enlargement of a smaller motif whose red part is
#      a 2x2 block. In the output, the sink quadrant receives the three motifs after shrinking them
#      back by 1/2 and packing them into the sink anchor's three corresponding 2x2 quarters. The
#      quarter whose global label matches the sink quadrant stays plain red.
#    - Small trace example: originally TL was extracted once and copied into TR/BL/BR. Here the flow
#      changes at the "replicate source to targets" stage: three quadrants now act as sources, each
#      is reduced rather than enlarged, and all three are stamped into one destination anchor. The
#      difference is therefore both combinatorial (3 sources -> 1 sink) and geometric (packing into
#      sub-anchor quarters instead of spreading to separate quadrants).
# b) Complexity calibration:
#    - The sink quadrant is not fixed; it is the unique quadrant that lacks a colored motif, so the
#      many:1 behavior is visible across examples.
#    - Only one packing template is used: a 2x2 subdivision of the sink's 4x4 anchor. This keeps
#      the rule learnable while still making the inversion obvious.
#    - Source motifs are distinct random patterns with 3-5 colors and asymmetric freeform fill, so
#      the merge is visually nontrivial and cannot be mistaken for recoloring or simple copying.
# c) Hidden assumptions reviewed against the original solver:
#    - The original solver assumes TL is the only source and that every other quadrant is anchor-only.
#      That no longer holds, so target detection must switch from hardcoded TL extraction to finding
#      the unique quadrant whose red component has no attached colors.
#    - The original scaling uses anchor-size ratios per quadrant. In the variant, the scale factor is
#      tied to the packing layout instead: every source is a 2x blow-up of the motif that will be
#      placed into a 2x2 anchor quarter. The solver therefore checks 2x -> 1x roundtrip instead of
#      target-size/source-size ratios.
#    - The original stamping loop safely wrote into separate quadrants. Here all reduced motifs land
#      in one quadrant, so generation must explicitly avoid motif-motif overlap; otherwise iteration
#      order would become an accidental hidden rule.
# d) Placement legality:
#    - Source patches are always fully contained in their own 13x13 quadrant.
#    - The sink anchor position is chosen from the intersection of all source-fit constraints, and a
#      conflict check rejects placements where the reduced motifs would overlap each other or leave
#      the sink quadrant.
#    - Output-first validation still guards the entire construction by requiring main(input) == output.
# e) Visual observability:
#    - In every sample the input already shows three enlarged motifs and one empty sink, while the
#      output shows those same motifs miniaturized and combined at the sink. A human can infer both
#      "many sources -> one target" and the quarter-based packing from only a few examples.
# f) Visual cleanliness:
#    - Divider color is excluded from motif colors to avoid confusion with the quadrant borders.
#    - Source anchors are biased toward the outer corner of each reduced patch so the packed mini
#      motifs spread away from one another, making the merged target readable instead of cluttered.
# g) Color-rule audit:
#    - No color remapping is used. Colors are preserved exactly through shrinking, so the task relies
#      on visible geometry only, not on hidden color-id arithmetic or ordering.

import random

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, paste


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
    """Nearest-neighbor resize."""
    src_h, src_w = img.shape
    assert new_h > 0 and new_w > 0
    rr = np.clip(np.floor(np.arange(new_h) * src_h / new_h).astype(int), 0, src_h - 1)
    cc = np.clip(np.floor(np.arange(new_w) * src_w / new_w).astype(int), 0, src_w - 1)
    return img[rr[:, None], cc[None, :]]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Merge three enlarged quadrant motifs into the single anchor-only quadrant by shrinking and packing them.

    Input structure:
    - The grid is split into four quadrants by one full row and one full column of the same non-bg color.
    - Every quadrant contains exactly one red square anchor block.
    - Exactly one quadrant contains only that red anchor; the other three quadrants each contain one colored
      4-connected component attached to their red anchor.

    Rule:
    - Treat the anchor-only quadrant as the destination.
    - In each source quadrant, extract the component containing the red anchor and shrink it by a factor of 2
      with nearest-neighbor resizing.
    - Subdivide the destination red anchor into four equal 2x2 quarters labeled by the global quadrant positions;
      place each shrunken source so its red 2x2 block lands on the matching quarter, then stamp the non-bg pixels.

    Key insight: The empty quadrant is the only sink, while the three visible motifs are already enlarged copies
    of the mini-motifs that must be packed back into the sink anchor according to quadrant identity.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    out = g.copy()

    bg = get_bg(g)
    mid_r, mid_c = _find_dividers(g, bg)

    red = components_mask(g == RED, connectivity=4)
    assert len(red) == 4, "grid must contain exactly four red anchors"

    anchors: dict[tuple[int, int], tuple[int, int, int]] = {}
    for pts in red:
        r0, c0, r1, c1 = bbox_coords(pts)
        size_r = r1 - r0 + 1
        size_c = c1 - c0 + 1
        assert size_r == size_c, "red anchors must be square"
        rr = (r0 + r1) // 2
        cc = (c0 + c1) // 2
        key = (0 if rr < mid_r else 1, 0 if cc < mid_c else 1)
        assert key not in anchors, "each quadrant must contain exactly one red anchor"
        anchors[key] = (int(r0), int(c0), int(size_r))

    source_patches: dict[tuple[int, int], np.ndarray] = {}
    target_key: tuple[int, int] | None = None

    for key, (q_r0, q_r1, q_c0, q_c1) in QUADS.items():
        quad = g[q_r0 : q_r1 + 1, q_c0 : q_c1 + 1]
        comps = components_mask(quad != bg, connectivity=4)
        patch = None
        for pts in comps:
            if any(quad[r, c] == RED for r, c in pts):
                r0, c0, r1, c1 = bbox_coords(pts)
                patch = quad[r0 : r1 + 1, c0 : c1 + 1].copy()
                break
        assert patch is not None, "every quadrant must contain a non-bg component with the red anchor"

        if np.all(patch == RED):
            assert target_key is None, "there must be exactly one destination quadrant"
            target_key = key
        else:
            source_patches[key] = patch

    assert target_key is not None and len(source_patches) == 3, "expected three sources and one destination"
    tr0, tc0, target_size = anchors[target_key]
    assert target_size % 2 == 0, "destination anchor size must be even"
    slot_size = target_size // 2

    for key, patch in source_patches.items():
        assert patch.shape[0] % 2 == 0 and patch.shape[1] % 2 == 0, "source patch must be a 2x enlargement"
        mini = _resize_nn(patch, patch.shape[0] // 2, patch.shape[1] // 2)
        roundtrip = _resize_nn(mini, patch.shape[0], patch.shape[1])
        assert np.array_equal(roundtrip, patch), "source patch must shrink losslessly by factor 2"

        rr_red, cc_red = np.where(mini == RED)
        assert rr_red.size > 0, "reduced source must keep a red anchor"
        red_r0, red_c0 = int(rr_red.min()), int(cc_red.min())
        red_r1, red_c1 = int(rr_red.max()), int(cc_red.max())
        assert red_r1 - red_r0 + 1 == slot_size and red_c1 - red_c0 + 1 == slot_size

        slot_r0 = tr0 + key[0] * slot_size
        slot_c0 = tc0 + key[1] * slot_size
        place_r0 = slot_r0 - red_r0
        place_c0 = slot_c0 - red_c0

        rr_p, cc_p = np.where(mini != bg)
        for rr0, cc0 in zip(rr_p.tolist(), cc_p.tolist(), strict=False):
            rr_t = place_r0 + rr0
            cc_t = place_c0 + cc0
            if not (0 <= rr_t < h and 0 <= cc_t < w):
                continue
            if rr_t == mid_r or cc_t == mid_c:
                continue
            if out[rr_t, cc_t] == RED:
                continue
            out[rr_t, cc_t] = int(mini[rr0, cc0])

    return out


# --------------- generator ---------------

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

TARGET_ANCHOR_SIZE = 4
MINI_ANCHOR_SIZE = TARGET_ANCHOR_SIZE // 2
SOURCE_SCALE = 2
BASE_H_RANGE = (3, 6)
BASE_W_RANGE = (3, 6)
FILL_RATIO_MIN = 0.5
FILL_RATIO_MAX = 0.8
MOTIF_COLORS_MIN = 3
MOTIF_COLORS_MAX = 5
COLOR_POOL = [0, 3, 5, 6, 7, 8, 9]
MAX_RETRIES = 400


def _grow_4conn(h: int, w: int, seeds: set[tuple[int, int]], n_fill: int) -> set[tuple[int, int]]:
    """Grow a 4-connected region from seeds until it has n_fill cells."""
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
    filled: set[tuple[int, int]],
    red_cells: set[tuple[int, int]],
    h: int,
    w: int,
) -> np.ndarray | None:
    """Color a connected region while ensuring the bbox uses all rows and cols."""
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
    palette = COLOR_POOL[:]
    random.shuffle(palette)
    extras = palette[: n_colors - 1]

    patch = np.full((h, w), BG, dtype=int)
    for r, c in red_cells:
        patch[r, c] = RED
    for idx, (r, c) in enumerate(non_red):
        patch[r, c] = extras[idx] if idx < len(extras) else random.choice(extras)
    return patch


def _preferred_anchor(h: int, w: int, key: tuple[int, int]) -> tuple[int, int]:
    """Place the 2x2 red anchor near the corner that makes the packed motif spread outward."""
    ar = 0 if key[0] == 1 else h - MINI_ANCHOR_SIZE
    ac = 0 if key[1] == 1 else w - MINI_ANCHOR_SIZE
    return ar, ac


def _base_patch(key: tuple[int, int]) -> tuple[np.ndarray, int, int] | None:
    """Build a reduced motif whose anchor corner depends on its quadrant label."""
    for _ in range(80):
        h = random.randint(*BASE_H_RANGE)
        w = random.randint(*BASE_W_RANGE)
        ar, ac = _preferred_anchor(h, w, key)
        red_cells = {(ar + dr, ac + dc) for dr in range(MINI_ANCHOR_SIZE) for dc in range(MINI_ANCHOR_SIZE)}
        min_fill = max(len(red_cells) + 2, int(np.ceil(FILL_RATIO_MIN * h * w)))
        max_fill = int(np.floor(FILL_RATIO_MAX * h * w))
        if max_fill < min_fill:
            continue

        filled = _grow_4conn(h, w, red_cells, random.randint(min_fill, max_fill))
        patch = _assign_colors(filled, red_cells, h, w)
        if patch is None:
            continue
        if len(set(map(int, patch.flatten())) - {BG}) < MOTIF_COLORS_MIN:
            continue
        return patch, ar, ac
    return None


def _upscale_source(base: np.ndarray, anchor_r: int, anchor_c: int) -> tuple[np.ndarray, int, int]:
    """Convert a reduced motif into the visible 2x source component."""
    patch = np.repeat(np.repeat(base, SOURCE_SCALE, axis=0), SOURCE_SCALE, axis=1)
    return patch, anchor_r * SOURCE_SCALE, anchor_c * SOURCE_SCALE


def _random_source_position(key: tuple[int, int], patch: np.ndarray) -> tuple[int, int]:
    """Choose a fully in-bounds source placement inside its quadrant."""
    q_r0, q_r1, q_c0, q_c1 = QUADS[key]
    max_r0 = q_r1 - patch.shape[0] + 1
    max_c0 = q_c1 - patch.shape[1] + 1
    assert q_r0 <= max_r0 and q_c0 <= max_c0
    return random.randint(q_r0, max_r0), random.randint(q_c0, max_c0)


def _can_pack_target(
    target_key: tuple[int, int],
    anchor_r: int,
    anchor_c: int,
    minis: dict[tuple[int, int], tuple[np.ndarray, int, int]],
) -> bool:
    """Check that all reduced motifs fit in the target quadrant without non-red overlap."""
    q_r0, q_r1, q_c0, q_c1 = QUADS[target_key]
    canvas = np.full((GRID_SIZE, GRID_SIZE), BG, dtype=int)
    paste(canvas, np.full((TARGET_ANCHOR_SIZE, TARGET_ANCHOR_SIZE), RED, dtype=int), anchor_r, anchor_c)

    for key, (mini, ar, ac) in minis.items():
        slot_r0 = anchor_r + key[0] * MINI_ANCHOR_SIZE
        slot_c0 = anchor_c + key[1] * MINI_ANCHOR_SIZE
        place_r0 = slot_r0 - ar
        place_c0 = slot_c0 - ac
        rr_p, cc_p = np.where(mini != BG)
        for rr0, cc0 in zip(rr_p.tolist(), cc_p.tolist(), strict=False):
            rr_t = place_r0 + rr0
            cc_t = place_c0 + cc0
            if not (q_r0 <= rr_t <= q_r1 and q_c0 <= cc_t <= q_c1):
                return False
            val = int(mini[rr0, cc0])
            cur = int(canvas[rr_t, cc_t])
            if val == RED:
                if cur != RED:
                    return False
                continue
            if cur != BG:
                return False
            canvas[rr_t, cc_t] = val
    return True


def _choose_target_anchor(
    target_key: tuple[int, int],
    minis: dict[tuple[int, int], tuple[np.ndarray, int, int]],
) -> tuple[int, int] | None:
    """Pick a target anchor position that can host all reduced motifs."""
    q_r0, q_r1, q_c0, q_c1 = QUADS[target_key]
    r_lo = q_r0
    r_hi = q_r1 - TARGET_ANCHOR_SIZE + 1
    c_lo = q_c0
    c_hi = q_c1 - TARGET_ANCHOR_SIZE + 1

    for key, (mini, ar, ac) in minis.items():
        slot_dr = key[0] * MINI_ANCHOR_SIZE
        slot_dc = key[1] * MINI_ANCHOR_SIZE
        r_lo = max(r_lo, q_r0 + ar - slot_dr)
        r_hi = min(r_hi, q_r1 - mini.shape[0] + 1 + ar - slot_dr)
        c_lo = max(c_lo, q_c0 + ac - slot_dc)
        c_hi = min(c_hi, q_c1 - mini.shape[1] + 1 + ac - slot_dc)

    if r_lo > r_hi or c_lo > c_hi:
        return None

    candidates = [(r, c) for r in range(r_lo, r_hi + 1) for c in range(c_lo, c_hi + 1)]
    random.shuffle(candidates)
    for anchor_r, anchor_c in candidates:
        if _can_pack_target(target_key, anchor_r, anchor_c, minis):
            return anchor_r, anchor_c
    return None


def _stamp(out: np.ndarray, patch: np.ndarray, anchor_r: int, anchor_c: int, red_r0: int, red_c0: int) -> None:
    """Stamp a patch aligned by its red bbox top-left, preserving existing red cells."""
    place_r0 = anchor_r - red_r0
    place_c0 = anchor_c - red_c0
    rr_p, cc_p = np.where(patch != BG)
    for rr0, cc0 in zip(rr_p.tolist(), cc_p.tolist(), strict=False):
        rr_t = place_r0 + rr0
        cc_t = place_c0 + cc0
        if out[rr_t, cc_t] == RED:
            continue
        out[rr_t, cc_t] = int(patch[rr0, cc0])


def _gen_attempt() -> tuple[np.ndarray, np.ndarray] | None:
    """Single output-first generation attempt."""
    target_key = random.choice(list(QUADS))
    source_keys = [key for key in QUADS if key != target_key]

    bases: dict[tuple[int, int], tuple[np.ndarray, int, int]] = {}
    seen = set()
    for key in source_keys:
        built = _base_patch(key)
        if built is None:
            return None
        base, ar, ac = built
        signature = tuple(base.flatten().tolist())
        if signature in seen:
            return None
        seen.add(signature)
        bases[key] = (base, ar, ac)

    sources: dict[tuple[int, int], tuple[np.ndarray, int, int]] = {}
    for key, (base, ar, ac) in bases.items():
        source_patch, src_ar, src_ac = _upscale_source(base, ar, ac)
        sources[key] = (source_patch, src_ar, src_ac)

    target_anchor = _choose_target_anchor(target_key, bases)
    if target_anchor is None:
        return None
    target_anchor_r, target_anchor_c = target_anchor

    source_positions: dict[tuple[int, int], tuple[int, int]] = {}
    for key, (patch, _ar, _ac) in sources.items():
        source_positions[key] = _random_source_position(key, patch)

    out = np.full((GRID_SIZE, GRID_SIZE), BG, dtype=int)
    out[MID, :] = DIV
    out[:, MID] = DIV

    for key, (patch, _src_ar, _src_ac) in sources.items():
        pr0, pc0 = source_positions[key]
        paste(out, patch, pr0, pc0)

    paste(out, np.full((TARGET_ANCHOR_SIZE, TARGET_ANCHOR_SIZE), RED, dtype=int), target_anchor_r, target_anchor_c)

    for key, (base, ar, ac) in bases.items():
        slot_r0 = target_anchor_r + key[0] * MINI_ANCHOR_SIZE
        slot_c0 = target_anchor_c + key[1] * MINI_ANCHOR_SIZE
        _stamp(out, base, slot_r0, slot_c0, ar, ac)

    inp = out.copy()
    q_r0, q_r1, q_c0, q_c1 = QUADS[target_key]
    inp[q_r0 : q_r1 + 1, q_c0 : q_c1 + 1] = BG
    paste(inp, np.full((TARGET_ANCHOR_SIZE, TARGET_ANCHOR_SIZE), RED, dtype=int), target_anchor_r, target_anchor_c)

    if not np.array_equal(main(inp), out):
        return None
    return inp, out


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        sample = _gen_attempt()
        if sample is not None:
            inp, _out = sample
            return inp
    raise RuntimeError("failed to generate a valid b10624e5 D15 variant input")
