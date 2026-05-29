# Combo variant of b10624e5 (index 34)
# combo_spec: D4/2+D15/2
# Name: 多源汇聚到单一目标 + 缩小图案
#
# Design analysis
# a) Concrete behavior:
#    - The original task has one visible source in TL and three anchor-only targets. This combo flips
#      the cardinality: exactly one quadrant is now the sink and the other three are visible sources.
#    - Each source patch is an enlarged copy of a smaller motif whose red anchor is always 2x2. The
#      visible source anchor is either 4x4 or 6x6, so the solver must infer a shrink factor of 1/2
#      or 1/3 from that quadrant's own red square size.
#    - The sink quadrant contains only a red 4x4 anchor in the input. In the output, that 4x4 block
#      is interpreted as four 2x2 quarters labeled by global quadrant identity. The three source
#      motifs are independently shrunk and packed into the three matching quarters; the sink's own
#      quarter stays plain red.
#    - Small trace: if TR, BL, BR are sources with anchor sizes 4, 6, 4, the original code would try
#      to copy one TL motif outward and would immediately rely on the wrong source count. Here the new
#      solver instead finds the unique anchor-only quadrant, shrinks TR by 1/2, BL by 1/3, BR by 1/2,
#      and merges all three around the sink anchor.
# b) Complexity calibration:
#    - The sink quadrant is not fixed, so the many:1 behavior is visible across examples.
#    - Shrink factors are restricted to two values, 1/2 and 1/3. That keeps the task learnable while
#      still making "per-source shrink" observable instead of collapsing to one universal factor.
#    - Every sample contains both source sizes 4 and 6, so the D4 change is active inside each puzzle
#      rather than only across a dataset.
# c) Hidden assumptions reviewed against the original solver:
#    - Hardcoding TL as the only source no longer works; the solver must identify the unique quadrant
#      whose red component is anchor-only.
#    - The original ratio logic compared one source anchor against several target anchors. Here each
#      source has its own visible anchor size, while the target slot size is fixed at 2x2 inside the
#      sink anchor. The shrink ratio therefore comes from source_anchor_size / 2.
#    - The original stamping loop wrote into separate quadrants. Now three transformed motifs land in
#      one quadrant, so generation must guarantee their packed footprints do not overlap.
# d) Placement legality:
#    - Base motifs are 3x3 to 4x4, so after enlarging by factor 3 they still fit inside a 13x13
#      quadrant.
#    - The target anchor position is chosen from the intersection of all fit constraints, and a
#      conflict check rejects any placement where packed mini-motifs overlap or leave the sink quadrant.
#    - Output-first validation enforces main(input) == output on every accepted sample.
# e) Visual observability:
#    - The input already shows three large attached motifs and one empty sink, while the output shows
#      those same motifs miniaturized and co-located. From a few examples, a human can infer both the
#      many:1 merge and the anchor-size-dependent shrinking.
# f) Visual cleanliness:
#    - Divider color is excluded from motif colors so the border is never confused with motif content.
#    - Source anchors are placed near the outward-facing corner of each mini motif, so when packed they
#      spread away from the center of the sink anchor instead of piling onto one another.
# g) Color-rule audit:
#    - No recoloring is used. Colors survive shrinking unchanged, so the task depends only on visible
#      geometry, source-vs-sink role, and anchor-size-based scaling.

import random

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, paste


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
SLOT_SIZE = TARGET_ANCHOR_SIZE // 2
SOURCE_FACTORS = (2, 3)
SOURCE_SIZES = tuple(SLOT_SIZE * factor for factor in SOURCE_FACTORS)
BASE_H_RANGE = (3, 4)
BASE_W_RANGE = (3, 4)
FILL_RATIO_MIN = 0.5
FILL_RATIO_MAX = 0.8
MOTIF_COLORS_MIN = 3
MOTIF_COLORS_MAX = 5
COLOR_POOL = [0, 3, 5, 6, 7, 8, 9]
MAX_RETRIES = 500


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


def _detect_anchors(g: np.ndarray, mid_r: int, mid_c: int) -> dict[tuple[int, int], tuple[int, int, int]]:
    """Return anchor map keyed by quadrant: (r0, c0, size)."""
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
    return anchors


def _extract_patch_with_red(quad: np.ndarray, bg: int) -> np.ndarray:
    """Extract the unique non-bg component in a quadrant that contains red."""
    comps = components_mask(quad != bg, connectivity=4)
    patch = None
    for pts in comps:
        if any(quad[r, c] == RED for r, c in pts):
            assert patch is None, "quadrant must not contain multiple red-bearing components"
            r0, c0, r1, c1 = bbox_coords(pts)
            patch = quad[r0 : r1 + 1, c0 : c1 + 1].copy()
    assert patch is not None, "every quadrant must contain a non-bg component with the red anchor"
    return patch


def main(grid: np.ndarray) -> np.ndarray:
    """
    Merge three enlarged source motifs into the unique anchor-only quadrant by shrinking each source separately.

    Input structure:
    - The grid is divided into four 13x13 quadrants by one full row and one full column of the same non-bg color.
    - Every quadrant contains exactly one red square anchor block.
    - Exactly one quadrant is anchor-only and uses a 4x4 red block; the other three quadrants each contain one
      colored 4-connected component attached to a 4x4 or 6x6 red anchor.

    Rule:
    - Treat the anchor-only quadrant as the sink.
    - In each source quadrant, extract the component containing the red anchor and shrink it with nearest-neighbor
      resizing so that its red square becomes 2x2. A 4x4 source anchor shrinks by 1/2; a 6x6 source anchor shrinks
      by 1/3.
    - Split the sink's 4x4 anchor into four 2x2 quarters labeled by global quadrant identity. Place each shrunken
      source so its 2x2 red block lands on the quarter matching that source quadrant, then stamp the non-bg pixels.

    Key insight: The only empty quadrant is the destination, and each visible source announces its own shrink factor
    through red-anchor size before all three mini-motifs are packed into one shared 4x4 anchor.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    out = g.copy()

    bg = get_bg(g)
    mid_r, mid_c = _find_dividers(g, bg)
    anchors = _detect_anchors(g, mid_r, mid_c)

    source_patches: dict[tuple[int, int], np.ndarray] = {}
    target_key: tuple[int, int] | None = None

    for key, (q_r0, q_r1, q_c0, q_c1) in QUADS.items():
        quad = g[q_r0 : q_r1 + 1, q_c0 : q_c1 + 1]
        patch = _extract_patch_with_red(quad, bg)
        if np.all(patch == RED):
            assert target_key is None, "there must be exactly one destination quadrant"
            target_key = key
        else:
            source_patches[key] = patch

    assert target_key is not None and len(source_patches) == 3, "expected three sources and one destination"
    tr0, tc0, target_size = anchors[target_key]
    assert target_size == TARGET_ANCHOR_SIZE, "destination anchor must be 4x4"

    for key, patch in source_patches.items():
        _sr0, _sc0, source_size = anchors[key]
        assert source_size in SOURCE_SIZES, "source anchor size must encode shrink factor"
        factor = source_size // SLOT_SIZE
        assert source_size == SLOT_SIZE * factor
        assert patch.shape[0] % factor == 0 and patch.shape[1] % factor == 0, "source patch must be an exact upscale"

        mini = _resize_nn(patch, patch.shape[0] // factor, patch.shape[1] // factor)
        roundtrip = _resize_nn(mini, patch.shape[0], patch.shape[1])
        assert np.array_equal(roundtrip, patch), "source patch must shrink losslessly"

        rr_red, cc_red = np.where(mini == RED)
        assert rr_red.size > 0, "reduced source must retain a red anchor"
        red_r0, red_c0 = int(rr_red.min()), int(cc_red.min())
        red_r1, red_c1 = int(rr_red.max()), int(cc_red.max())
        assert red_r1 - red_r0 + 1 == SLOT_SIZE and red_c1 - red_c0 + 1 == SLOT_SIZE

        slot_r0 = tr0 + key[0] * SLOT_SIZE
        slot_c0 = tc0 + key[1] * SLOT_SIZE
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
    """Bias the 2x2 red anchor toward the corner that makes the packed motif spread outward."""
    ar = 0 if key[0] == 1 else h - SLOT_SIZE
    ac = 0 if key[1] == 1 else w - SLOT_SIZE
    return ar, ac


def _base_patch(key: tuple[int, int]) -> tuple[np.ndarray, int, int] | None:
    """Build a reduced motif with a 2x2 red anchor."""
    for _ in range(100):
        h = random.randint(*BASE_H_RANGE)
        w = random.randint(*BASE_W_RANGE)
        ar, ac = _preferred_anchor(h, w, key)
        red_cells = {(ar + dr, ac + dc) for dr in range(SLOT_SIZE) for dc in range(SLOT_SIZE)}
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


def _upscale_source(base: np.ndarray, factor: int) -> np.ndarray:
    """Convert a reduced motif into a visible enlarged source component."""
    return np.repeat(np.repeat(base, factor, axis=0), factor, axis=1)


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

    for key, (mini, red_r0, red_c0) in minis.items():
        slot_r0 = anchor_r + key[0] * SLOT_SIZE
        slot_c0 = anchor_c + key[1] * SLOT_SIZE
        place_r0 = slot_r0 - red_r0
        place_c0 = slot_c0 - red_c0
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

    for key, (mini, red_r0, red_c0) in minis.items():
        slot_dr = key[0] * SLOT_SIZE
        slot_dc = key[1] * SLOT_SIZE
        r_lo = max(r_lo, q_r0 + red_r0 - slot_dr)
        r_hi = min(r_hi, q_r1 - mini.shape[0] + 1 + red_r0 - slot_dr)
        c_lo = max(c_lo, q_c0 + red_c0 - slot_dc)
        c_hi = min(c_hi, q_c1 - mini.shape[1] + 1 + red_c0 - slot_dc)

    if r_lo > r_hi or c_lo > c_hi:
        return None

    candidates = [(r, c) for r in range(r_lo, r_hi + 1) for c in range(c_lo, c_hi + 1)]
    random.shuffle(candidates)
    for anchor_r, anchor_c in candidates:
        if _can_pack_target(target_key, anchor_r, anchor_c, minis):
            return anchor_r, anchor_c
    return None


def _stamp(out: np.ndarray, patch: np.ndarray, anchor_r: int, anchor_c: int, red_r0: int, red_c0: int) -> None:
    """Stamp a patch aligned by the top-left of its red anchor."""
    place_r0 = anchor_r - red_r0
    place_c0 = anchor_c - red_c0
    rr_p, cc_p = np.where(patch != BG)
    for rr0, cc0 in zip(rr_p.tolist(), cc_p.tolist(), strict=False):
        rr_t = place_r0 + rr0
        cc_t = place_c0 + cc0
        if out[rr_t, cc_t] == RED:
            continue
        out[rr_t, cc_t] = int(patch[rr0, cc0])


def _choose_source_factors(source_keys: list[tuple[int, int]]) -> dict[tuple[int, int], int]:
    """Assign source scale factors while forcing both shrink ratios to appear."""
    heavy = set(random.sample(source_keys, k=random.choice([1, 2])))
    return {key: (3 if key in heavy else 2) for key in source_keys}


def _gen_attempt() -> tuple[np.ndarray, np.ndarray] | None:
    """Single output-first generation attempt."""
    target_key = random.choice(list(QUADS))
    source_keys = [key for key in QUADS if key != target_key]
    source_factors = _choose_source_factors(source_keys)

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

    minis = {key: (base, ar, ac) for key, (base, ar, ac) in bases.items()}
    target_anchor = _choose_target_anchor(target_key, minis)
    if target_anchor is None:
        return None
    target_anchor_r, target_anchor_c = target_anchor

    sources: dict[tuple[int, int], tuple[np.ndarray, int]] = {}
    for key, (base, _ar, _ac) in bases.items():
        factor = source_factors[key]
        patch = _upscale_source(base, factor)
        sources[key] = (patch, SLOT_SIZE * factor)

    source_positions: dict[tuple[int, int], tuple[int, int]] = {}
    for key, (patch, _size) in sources.items():
        source_positions[key] = _random_source_position(key, patch)

    out = np.full((GRID_SIZE, GRID_SIZE), BG, dtype=int)
    out[MID, :] = DIV
    out[:, MID] = DIV

    for key, (patch, _size) in sources.items():
        pr0, pc0 = source_positions[key]
        paste(out, patch, pr0, pc0)

    paste(out, np.full((TARGET_ANCHOR_SIZE, TARGET_ANCHOR_SIZE), RED, dtype=int), target_anchor_r, target_anchor_c)

    for key, (base, ar, ac) in bases.items():
        slot_r0 = target_anchor_r + key[0] * SLOT_SIZE
        slot_c0 = target_anchor_c + key[1] * SLOT_SIZE
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
    raise RuntimeError("failed to generate a valid b10624e5 combo input")
