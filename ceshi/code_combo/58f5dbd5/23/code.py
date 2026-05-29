# Combo variant of 58f5dbd5 (index 23)
# combo_spec: D3/0+D19/0
# Name: 多色motif+多对一映射

from __future__ import annotations

import itertools
import random

import numpy as np

from arc_utils import bbox_h, bbox_w, crop_bbox, find_components, get_bg, is_contiguous, paste


# Design analysis for the D3/0 + D19/0 combo
# -------------------------------------------
# a) Concrete pixel behavior
#    - The destination side remains a rectangular array of solid 5x5 monochrome blocks.
#    - The source side remains a separate rectangular array of small motifs, but each useful motif is now
#      a 3x3-bounded two-color sprite:
#      1. the primary color matches exactly one destination block color;
#      2. the decoration color is a shared third color reused across all useful motifs.
#    - Each block color now appears in 2 or 3 different useful motifs. All of them are centered onto the same
#      5x5 block and fused.
#    - Example trace on a red block:
#      original task: one red mask -> those cells become background.
#      D3 only: one {red + teal} motif -> red cells become background, teal cells become teal.
#      fused combo: three different {red + teal} motifs -> every red-coded cell across the set becomes background,
#      every teal-coded cell across the set becomes teal, and the final center pattern is the union of all motifs.
#      Difference: one source object now carries two roles, and several such objects combine onto one block.
#
# b) Complexity calibration
#    - A single shared decoration color keeps the color-role rule legible. Allowing each motif its own decoration
#      color would add too many latent correspondences for 2-4 ARC examples.
#    - The block count is tightened to 3-4. This is the smallest change that makes room for 2-3 motifs per block
#      within the original 18-22 by 18-24 canvas while preserving axis-separated regions.
#    - Distractor motifs are removed. The fusion is already harder than the base task; omitting distractors avoids
#      ambiguity and keeps the main novelty visible.
#
# c) Hidden-assumption audit against the original main()
#    - Original STEP 2 assumed useful motifs were monochrome and could be found by searching per block color.
#      That breaks because a useful motif is now multicolor; the solver must inspect multicolor components and then
#      infer which of the two colors is the matching block color.
#    - Original STEP 2 also assumed one motif per block color and kept just one candidate. That breaks because the
#      fused task needs 2-3 motifs per block color, so the solver must retain a list per color.
#    - Original STEP 3 changed every motif cell to background. That is no longer correct because decoration cells
#      must survive as a third color in the output.
#    - A new potential conflict appears when multiple motifs overlap with opposite roles. The generator avoids this by
#      guaranteeing that the union of all primary cells and the union of all decoration cells are disjoint.
#
# d) Placement legality
#    - Destination blocks keep the original 5x5 size, 1-cell internal gaps, and rectangular packing.
#    - Motif slots remain 3x3 with 1-cell gaps, but slot capacity expands to 8/9/10 so all repeated motifs fit.
#    - Destination and motif regions are still separated along one axis, preventing accidental merges between large
#      blocks and small motifs in multicolor component detection.
#
# e) Visual observability
#    - In the input, every useful motif visibly contains one block color plus one repeated decoration color.
#    - Each block color repeats 2-3 times among the motifs, making the multi-to-one relation directly visible.
#    - In the output, every block shows both holes and decoration-colored inlays, and additional motifs enlarge that
#      pattern beyond what a single source motif would produce.
#
# f) Visual neatness
#    - The shared decoration color is the only repeated non-block cue. This keeps the motif field dense but readable.
#    - Pattern-set sampling avoids fully filling the 3x3 center; some original block color remains visible, so the
#      transformed tiles do not collapse into noisy 3x3 patches.
#
# g) Color-change inferability
#    - The decoration color appears repeatedly in source motifs and again inside output blocks, giving a structural
#      clue that those cells are recolored rather than erased.
#    - No ordering based on hidden numeric color IDs is used; all matching follows visible color equality.
#
# h) Variant-validity metric used after implementation
#    - For generated samples we compare the fused output against:
#      1. a baseline that keeps only one motif per block color;
#      2. a baseline that erases all motif cells to background and ignores decoration.
#      Healthy samples should differ from both baselines on most trials.
#
# i) New solver assumptions and generator guarantees
#    - Solver assumption: every useful motif is a connected non-bg component with bbox <= 3x3 and exactly two colors,
#      with exactly one of those colors belonging to the destination-block palette.
#      Generator guarantee: all motifs satisfy this and no distractors violate it.
#    - Solver assumption: for a given block color, all useful motifs share one decoration color and their primary/deco
#      unions do not overlap. Generator guarantee: palette choice and pattern-set validation enforce both.
#    - Solver assumption: each block color has 2 or 3 useful motifs. Generator guarantee: the sampled slot payload is
#      exactly the repeated matching set, with no missing or extra same-primary motifs.

def component_patch(grid: np.ndarray, comp: dict, bg: int) -> np.ndarray:
    """Extract one connected component into its tight bbox while preserving cell colors."""
    r0, c0, r1, c1 = map(int, comp["bbox"])
    patch = np.full((r1 - r0 + 1, c1 - c0 + 1), int(bg), dtype=int)
    for r, c in comp["coords"]:
        patch[int(r) - r0, int(c) - c0] = int(grid[int(r), int(c)])
    return patch


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fuse several two-color motifs into one transformed block pattern and pack the transformed blocks.

    Input structure:
    - The most frequent color is the background.
    - Several destination objects are solid 5x5 monochrome blocks with distinct non-background colors.
    - In a separate motif region, each block color appears in 2-3 small 8-connected two-color motifs whose bbox is
      at most 3x3. One motif color matches the block; the other is a shared decoration color.

    Rule:
    - For each block color, collect all small two-color motifs that contain that block color.
    - Center every collected motif inside the matching 5x5 block. Across the whole set, cells marked by the matching
      color become background holes, while cells marked by the decoration color become that decoration color.
    - Pack all transformed 5x5 blocks into a compact output grid that preserves the original row/column order, with
      1-cell gaps and a 1-cell background border.

    Key insight:
    - The block is not paired with one stencil anymore. Several two-color motifs of the same primary color must be
      merged, and the two motif colors encode different actions: erase primary-colored cells, preserve decoration-
      colored cells as a visible third-color inlay.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    # STEP 1) Identify destination blocks: solid 5x5 monochrome squares.
    blocks: list[dict[str, int]] = []
    comps = find_components(g, bg=bg, connectivity=8, monochromatic=True, return_format="dict")
    for comp in comps:
        b = comp["bbox"]
        if bbox_h(b) != 5 or bbox_w(b) != 5:
            continue
        col = int(comp["color"])
        patch = crop_bbox(g, b)
        if bool((patch == int(col)).all()):
            r0, c0, _r1, _c1 = map(int, b)
            blocks.append({"col": int(col), "r0": int(r0), "c0": int(c0)})

    assert blocks

    # STEP 2) Collect every useful multicolor motif for each block color.
    block_colors = {int(b["col"]) for b in blocks}
    motif_candidates: dict[int, list[tuple[int, int, int, int, np.ndarray, np.ndarray, int]]] = {}
    comps_multi = find_components(g, bg=bg, connectivity=8, monochromatic=False, min_area=4, return_format="dict")
    for comp in comps_multi:
        b = comp["bbox"]
        if bbox_h(b) > 3 or bbox_w(b) > 3:
            continue
        if not (4 <= int(comp["area"]) <= 8):
            continue

        patch = component_patch(g, comp, bg)
        colors = sorted({int(v) for v in patch.flatten() if int(v) != int(bg)})
        if len(colors) != 2:
            continue

        matching = [int(c) for c in colors if int(c) in block_colors]
        if len(matching) != 1:
            continue

        match_color = int(matching[0])
        deco_color = int(colors[0] if int(colors[1]) == match_color else colors[1])
        primary_mask = patch == int(match_color)
        deco_mask = patch == int(deco_color)
        if not primary_mask.any() or not deco_mask.any():
            continue

        r0, c0, _r1, _c1 = map(int, b)
        motif_candidates.setdefault(match_color, []).append(
            (
                int(r0),
                int(c0),
                int(comp["area"]),
                int(patch.shape[0] * patch.shape[1]),
                primary_mask,
                deco_mask,
                int(deco_color),
            )
        )

    motifs_by_color: dict[int, list[tuple[np.ndarray, np.ndarray, int]]] = {}
    for block_color, candidates in motif_candidates.items():
        candidates.sort(key=lambda t: (t[0], t[1], t[2], t[3]))
        motifs_by_color[int(block_color)] = [
            (primary_mask, deco_mask, deco_color)
            for _r0, _c0, _area, _bbox_area, primary_mask, deco_mask, deco_color in candidates
        ]

    # STEP 3) Transform each 5x5 block by overlaying every matched motif of that color.
    transformed: dict[tuple[int, int], np.ndarray] = {}
    for b in blocks:
        col = int(b["col"])
        tile = np.full((5, 5), int(col), dtype=int)
        motifs = motifs_by_color.get(int(col), [])

        # Decoration is applied first, then primary holes win if a malformed input overlaps roles.
        for primary_mask, deco_mask, deco_color in motifs:
            mh, mw = map(int, primary_mask.shape)
            off_r = (5 - mh) // 2
            off_c = (5 - mw) // 2
            for rr, cc in np.argwhere(deco_mask):
                tr = int(off_r + int(rr))
                tc = int(off_c + int(cc))
                if 0 <= tr < 5 and 0 <= tc < 5:
                    tile[tr, tc] = int(deco_color)
        for primary_mask, _deco_mask, _deco_color in motifs:
            mh, mw = map(int, primary_mask.shape)
            off_r = (5 - mh) // 2
            off_c = (5 - mw) // 2
            for rr, cc in np.argwhere(primary_mask):
                tr = int(off_r + int(rr))
                tc = int(off_c + int(cc))
                if 0 <= tr < 5 and 0 <= tc < 5:
                    tile[tr, tc] = int(bg)

        transformed[(int(b["r0"]), int(b["c0"]))] = tile

    # STEP 4) Pack transformed 5x5 blocks into a compact output grid.
    rs = sorted({int(b["r0"]) for b in blocks})
    cs = sorted({int(b["c0"]) for b in blocks})
    r_index = {int(r): int(i) for i, r in enumerate(rs)}
    c_index = {int(c): int(j) for j, c in enumerate(cs)}

    out_h = int(1 + len(rs) * 5 + (len(rs) - 1) + 1)
    out_w = int(1 + len(cs) * 5 + (len(cs) - 1) + 1)
    out = np.full((out_h, out_w), int(bg), dtype=int)

    step = 6
    for b in blocks:
        r0 = int(b["r0"])
        c0 = int(b["c0"])
        i = int(r_index[int(r0)])
        j = int(c_index[int(c0)])
        rr = int(1 + i * step)
        cc = int(1 + j * step)
        out[rr : rr + 5, cc : cc + 5] = transformed[(int(r0), int(c0))]

    return out


# --------------- generator (o5x, combo variant) ---------------

CANVAS_HEIGHT_RANGE = (18, 22)
CANVAS_WIDTH_RANGE = (18, 24)
COLOR_COUNT = 10

DST_BLOCK_SIZE = 5
DST_BLOCK_STEP = 6
DST_LAYOUTS = {
    3: [(1, 3), (3, 1)],
    4: [(2, 2)],
}

MOTIF_SIZE = 3
MOTIF_STEP = 4
MOTIF_AREA_RANGE = (4, 8)
DECOR_AREA_RANGE = (1, 2)
MIN_PRIMARY_AREA = 3
MOTIF_LAYOUTS = {
    8: [(2, 4), (4, 2)],
    9: [(3, 3)],
    10: [(2, 5), (5, 2)],
}
MAX_EMPTY_SLOTS = 1

OUTER_MARGIN = 1
REGION_GAP = 1
OUTPUT_BORDER = 1
MOTIF_OFFSET_IN_BLOCK = 1
PRIMARY_UNION_AREA_RANGE = (5, 7)
DECOR_UNION_AREA_RANGE = (2, 4)
TOTAL_UNION_AREA_RANGE = (7, 8)

GENERATION_RETRIES = 500
PATTERN_SET_CACHE_TARGETS = {2: 256, 3: 128}
PATTERN_SET_CACHE_RETRIES = {2: 20000, 3: 120000}


def _spanned_size(count: int, item_size: int, step: int) -> int:
    """Return the span for count fixed-size items placed at a constant step."""
    return (count - 1) * step + item_size


def _make_patterns_by_area() -> dict[int, list[tuple[np.ndarray, np.ndarray]]]:
    """Precompute valid two-color motif patterns keyed by total occupied area."""
    min_area, max_area = MOTIF_AREA_RANGE
    patterns: dict[int, list[tuple[np.ndarray, np.ndarray]]] = {
        area: [] for area in range(min_area, max_area + 1)
    }
    for bits in itertools.product([False, True], repeat=MOTIF_SIZE * MOTIF_SIZE):
        shape = np.array(bits, dtype=bool).reshape(MOTIF_SIZE, MOTIF_SIZE)
        area = int(shape.sum())
        if area not in patterns:
            continue
        if not (shape[0].any() and shape[-1].any() and shape[:, 0].any() and shape[:, -1].any()):
            continue
        if not is_contiguous(shape.astype(int), background=0, connectivity=8):
            continue

        coords = [tuple(map(int, rc)) for rc in np.argwhere(shape)]
        max_decor = min(int(DECOR_AREA_RANGE[1]), area - 1)
        for decor_area in range(int(DECOR_AREA_RANGE[0]), max_decor + 1):
            for idxs in itertools.combinations(range(len(coords)), decor_area):
                deco = np.zeros_like(shape, dtype=bool)
                for idx in idxs:
                    r, c = coords[idx]
                    deco[r, c] = True
                primary = shape & ~deco
                if int(primary.sum()) < MIN_PRIMARY_AREA:
                    continue
                patterns[area].append((primary.copy(), deco.copy()))
    return patterns


PATTERNS_BY_AREA = _make_patterns_by_area()
ALL_PATTERNS = [
    (primary.copy(), deco.copy())
    for area in range(MOTIF_AREA_RANGE[0], MOTIF_AREA_RANGE[1] + 1)
    for primary, deco in PATTERNS_BY_AREA[area]
]


def _pattern_signature(pattern: tuple[np.ndarray, np.ndarray]) -> tuple[int, ...]:
    """Serialize a two-mask pattern for uniqueness checks."""
    primary, deco = pattern
    return tuple(int(v) for v in primary.astype(int).flatten()) + tuple(
        int(v) for v in deco.astype(int).flatten()
    )


def _union_pattern_roles(patterns: list[tuple[np.ndarray, np.ndarray]]) -> tuple[np.ndarray, np.ndarray]:
    """Return the union of primary masks and decoration masks across a set."""
    union_primary = np.zeros((MOTIF_SIZE, MOTIF_SIZE), dtype=bool)
    union_deco = np.zeros((MOTIF_SIZE, MOTIF_SIZE), dtype=bool)
    for primary, deco in patterns:
        union_primary |= primary
        union_deco |= deco
    return union_primary, union_deco


def _pattern_set_is_valid(patterns: list[tuple[np.ndarray, np.ndarray]]) -> bool:
    """Keep fused outputs visible, non-degenerate, and order-independent."""
    if len({_pattern_signature(pattern) for pattern in patterns}) != len(patterns):
        return False

    union_primary, union_deco = _union_pattern_roles(patterns)
    if np.any(union_primary & union_deco):
        return False

    primary_area = int(union_primary.sum())
    deco_area = int(union_deco.sum())
    changed_area = int((union_primary | union_deco).sum())
    if not (PRIMARY_UNION_AREA_RANGE[0] <= primary_area <= PRIMARY_UNION_AREA_RANGE[1]):
        return False
    if not (DECOR_UNION_AREA_RANGE[0] <= deco_area <= DECOR_UNION_AREA_RANGE[1]):
        return False
    if not (TOTAL_UNION_AREA_RANGE[0] <= changed_area <= TOTAL_UNION_AREA_RANGE[1]):
        return False

    for idx, (primary, deco) in enumerate(patterns):
        others = np.zeros((MOTIF_SIZE, MOTIF_SIZE), dtype=bool)
        for j, (other_primary, other_deco) in enumerate(patterns):
            if j == idx:
                continue
            others |= other_primary | other_deco
        if not np.any((primary | deco) & ~others):
            return False

    first_primary, first_deco = patterns[0]
    added_primary = int((union_primary & ~first_primary).sum())
    added_deco = int((union_deco & ~first_deco).sum())
    return added_primary >= 1 and added_deco >= 1


def _make_pattern_set_cache(count: int) -> list[list[tuple[np.ndarray, np.ndarray]]]:
    """Precompute a reusable bank of valid pattern sets for stable generation."""
    out: list[list[tuple[np.ndarray, np.ndarray]]] = []
    seen: set[tuple[tuple[int, ...], ...]] = set()
    target = int(PATTERN_SET_CACHE_TARGETS[count])
    retries = int(PATTERN_SET_CACHE_RETRIES[count])
    for _ in range(retries):
        patterns = [pattern for pattern in random.sample(ALL_PATTERNS, count)]
        if not _pattern_set_is_valid(patterns):
            continue
        sig = tuple(sorted(_pattern_signature(pattern) for pattern in patterns))
        if sig in seen:
            continue
        seen.add(sig)
        out.append([(primary.copy(), deco.copy()) for primary, deco in patterns])
        if len(out) >= target:
            break
    if not out:
        raise RuntimeError(f"failed to precompute valid pattern sets for count={count}")
    return out


VALID_PATTERN_SETS = {
    count: _make_pattern_set_cache(count) for count in sorted(PATTERN_SET_CACHE_TARGETS)
}


def _can_split(H: int, W: int, dh: int, dw: int, mh: int, mw: int) -> bool:
    """Return whether destination and motif regions can be separated by an axis."""
    return (
        dh + mh + 2 * OUTER_MARGIN + REGION_GAP <= H
        and max(dw, mw) + 2 * OUTER_MARGIN <= W
    ) or (
        dw + mw + 2 * OUTER_MARGIN + REGION_GAP <= W
        and max(dh, mh) + 2 * OUTER_MARGIN <= H
    )


def _placement_schemes(H: int, W: int, dh: int, dw: int, mh: int, mw: int) -> list[str]:
    """List feasible orderings for separated destination and motif regions."""
    schemes: list[str] = []
    if dh + mh + 2 * OUTER_MARGIN + REGION_GAP <= H and max(dw, mw) + 2 * OUTER_MARGIN <= W:
        schemes.extend(["dst_above", "motif_above"])
    if dw + mw + 2 * OUTER_MARGIN + REGION_GAP <= W and max(dh, mh) + 2 * OUTER_MARGIN <= H:
        schemes.extend(["dst_left", "motif_left"])
    return schemes


def _place_vertical(
    H: int, W: int, dh: int, dw: int, mh: int, mw: int, scheme: str
) -> tuple[int, int, int, int]:
    """Place vertically separated regions and return destination then motif origins."""
    if scheme == "dst_above":
        top_h, top_w, bot_h, bot_w = dh, dw, mh, mw
    else:
        top_h, top_w, bot_h, bot_w = mh, mw, dh, dw

    top_r = random.randint(OUTER_MARGIN, H - top_h - bot_h - OUTER_MARGIN - REGION_GAP)
    bot_r = random.randint(top_r + top_h + REGION_GAP, H - bot_h - OUTER_MARGIN)
    top_c = random.randint(OUTER_MARGIN, W - top_w - OUTER_MARGIN)
    bot_c = random.randint(OUTER_MARGIN, W - bot_w - OUTER_MARGIN)
    if scheme == "dst_above":
        return top_r, top_c, bot_r, bot_c
    return bot_r, bot_c, top_r, top_c


def _place_horizontal(
    H: int, W: int, dh: int, dw: int, mh: int, mw: int, scheme: str
) -> tuple[int, int, int, int]:
    """Place horizontally separated regions and return destination then motif origins."""
    if scheme == "dst_left":
        left_h, left_w, right_h, right_w = dh, dw, mh, mw
    else:
        left_h, left_w, right_h, right_w = mh, mw, dh, dw

    left_c = random.randint(OUTER_MARGIN, W - left_w - right_w - OUTER_MARGIN - REGION_GAP)
    right_c = random.randint(left_c + left_w + REGION_GAP, W - right_w - OUTER_MARGIN)
    left_r = random.randint(OUTER_MARGIN, H - left_h - OUTER_MARGIN)
    right_r = random.randint(OUTER_MARGIN, H - right_h - OUTER_MARGIN)
    if scheme == "dst_left":
        return left_r, left_c, right_r, right_c
    return right_r, right_c, left_r, left_c


def _place_regions(
    H: int, W: int, dh: int, dw: int, mh: int, mw: int
) -> tuple[int, int, int, int] | None:
    """Choose a valid separated placement for destination and motif regions."""
    schemes = _placement_schemes(H, W, dh, dw, mh, mw)
    if not schemes:
        return None
    scheme = random.choice(schemes)
    if scheme in {"dst_above", "motif_above"}:
        return _place_vertical(H, W, dh, dw, mh, mw, scheme)
    return _place_horizontal(H, W, dh, dw, mh, mw, scheme)


def _motifs_per_color_options(count: int) -> list[int]:
    """Return feasible matching-motif multiplicities for the sampled block count."""
    if count == 3:
        return [2, 3]
    return [2]


def _configs(H: int, W: int) -> list[tuple[int, int, int, int, int]]:
    """Enumerate layout configurations that fit in the sampled canvas."""
    out: list[tuple[int, int, int, int, int]] = []
    for count, dst_layouts in DST_LAYOUTS.items():
        for R, C in dst_layouts:
            dh = _spanned_size(R, DST_BLOCK_SIZE, DST_BLOCK_STEP)
            dw = _spanned_size(C, DST_BLOCK_SIZE, DST_BLOCK_STEP)
            for motifs_per_color in _motifs_per_color_options(count):
                filled = count * motifs_per_color
                for capacity, motif_layouts in MOTIF_LAYOUTS.items():
                    if not (filled <= capacity <= filled + MAX_EMPTY_SLOTS):
                        continue
                    for Mr, Mc in motif_layouts:
                        mh = _spanned_size(Mr, MOTIF_SIZE, MOTIF_STEP)
                        mw = _spanned_size(Mc, MOTIF_SIZE, MOTIF_STEP)
                        if _can_split(H, W, dh, dw, mh, mw):
                            out.append((R, C, Mr, Mc, motifs_per_color))
    return out


def _choose_config(H: int, W: int) -> tuple[int, int, int, int, int] | None:
    """Sample a valid layout while preserving count-first weighting."""
    configs = _configs(H, W)
    if not configs:
        return None
    counts = sorted({R * C for R, C, _Mr, _Mc, _k in configs})
    count = random.choice(counts)
    return random.choice([cfg for cfg in configs if cfg[0] * cfg[1] == count])


def _choose_palette(bg: int, count: int) -> tuple[int, list[int]]:
    """Pick one shared decoration color, then distinct destination colors."""
    palette = [c for c in range(COLOR_COUNT) if c != bg]
    random.shuffle(palette)
    decor_color = int(palette[0])
    block_colors = [int(c) for c in palette[1 : 1 + count]]
    return decor_color, block_colors


def _choose_pattern_set(count: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Sample 2-3 unique two-color motifs from the prevalidated cache."""
    cached = random.choice(VALID_PATTERN_SETS[count])
    return [(primary.copy(), deco.copy()) for primary, deco in cached]


def _paint_motif(
    grid: np.ndarray,
    r0: int,
    c0: int,
    primary_color: int,
    decor_color: int,
    bg: int,
    pattern: tuple[np.ndarray, np.ndarray],
) -> None:
    """Paint one two-color motif into a 3x3 slot."""
    primary_mask, deco_mask = pattern
    patch = np.full((MOTIF_SIZE, MOTIF_SIZE), int(bg), dtype=int)
    patch[primary_mask] = int(primary_color)
    patch[deco_mask] = int(decor_color)
    paste(grid, patch, r0, c0, mask=(primary_mask | deco_mask))


def _render_expected(
    bg: int,
    R: int,
    C: int,
    block_colors: list[int],
    motif_pattern_sets: dict[int, list[tuple[np.ndarray, np.ndarray]]],
    decor_color: int,
    use_all_motifs: bool = True,
    keep_decor: bool = True,
) -> np.ndarray:
    """Render the oracle output corresponding to the sampled block grid."""
    out_h = 2 * OUTPUT_BORDER + _spanned_size(R, DST_BLOCK_SIZE, DST_BLOCK_STEP)
    out_w = 2 * OUTPUT_BORDER + _spanned_size(C, DST_BLOCK_SIZE, DST_BLOCK_STEP)
    expected = np.full((out_h, out_w), int(bg), dtype=int)

    motif_slice = slice(MOTIF_OFFSET_IN_BLOCK, MOTIF_OFFSET_IN_BLOCK + MOTIF_SIZE)
    for i in range(R):
        for j in range(C):
            color = int(block_colors[i * C + j])
            tile = np.full((DST_BLOCK_SIZE, DST_BLOCK_SIZE), int(color), dtype=int)
            patterns = motif_pattern_sets[color]
            if not use_all_motifs:
                patterns = patterns[:1]
            union_primary, union_deco = _union_pattern_roles(patterns)
            sub = tile[motif_slice, motif_slice]
            if keep_decor:
                sub[union_deco] = int(decor_color)
            sub[union_primary] = int(bg)
            paste(
                expected,
                tile,
                OUTPUT_BORDER + i * DST_BLOCK_STEP,
                OUTPUT_BORDER + j * DST_BLOCK_STEP,
            )
    return expected


def _paint_destination_blocks(
    grid: np.ndarray, dst_r: int, dst_c: int, R: int, C: int, block_colors: list[int]
) -> None:
    """Paint the solid 5x5 destination blocks into the input grid."""
    for i in range(R):
        for j in range(C):
            color = int(block_colors[i * C + j])
            block = np.full((DST_BLOCK_SIZE, DST_BLOCK_SIZE), int(color), dtype=int)
            paste(grid, block, dst_r + i * DST_BLOCK_STEP, dst_c + j * DST_BLOCK_STEP)


def _slot_items(
    block_colors: list[int],
    motif_pattern_sets: dict[int, list[tuple[np.ndarray, np.ndarray]]],
) -> list[tuple[int, tuple[np.ndarray, np.ndarray]]]:
    """Build the shuffled motif payloads for occupied motif slots."""
    items: list[tuple[int, tuple[np.ndarray, np.ndarray]]] = []
    for color in block_colors:
        for pattern in motif_pattern_sets[color]:
            items.append((int(color), pattern))
    random.shuffle(items)
    return items


def _paint_motif_slots(
    grid: np.ndarray,
    motif_r: int,
    motif_c: int,
    Mr: int,
    Mc: int,
    block_colors: list[int],
    motif_pattern_sets: dict[int, list[tuple[np.ndarray, np.ndarray]]],
    decor_color: int,
    bg: int,
) -> None:
    """Randomly assign all useful motifs to visible motif slots."""
    slots = [(i, j) for i in range(Mr) for j in range(Mc)]
    random.shuffle(slots)
    slot_items = _slot_items(block_colors, motif_pattern_sets)
    for (sr, sc), (primary_color, pattern) in zip(slots, slot_items):
        _paint_motif(
            grid,
            motif_r + sr * MOTIF_STEP,
            motif_c + sc * MOTIF_STEP,
            int(primary_color),
            int(decor_color),
            int(bg),
            pattern,
        )


def _render_input(
    H: int,
    W: int,
    bg: int,
    R: int,
    C: int,
    Mr: int,
    Mc: int,
    block_colors: list[int],
    motif_pattern_sets: dict[int, list[tuple[np.ndarray, np.ndarray]]],
    decor_color: int,
    placement: tuple[int, int, int, int],
) -> np.ndarray:
    """Render the sampled destination blocks and motif slots into the input grid."""
    dst_r, dst_c, motif_r, motif_c = placement
    inp = np.full((H, W), int(bg), dtype=int)
    _paint_destination_blocks(inp, dst_r, dst_c, R, C, block_colors)
    _paint_motif_slots(
        inp,
        motif_r,
        motif_c,
        Mr,
        Mc,
        block_colors,
        motif_pattern_sets,
        decor_color,
        bg,
    )
    return inp


def _gen_attempt() -> np.ndarray | None:
    """Sample one input grid and reject it if the solver oracle disagrees."""
    H = random.randint(*CANVAS_HEIGHT_RANGE)
    W = random.randint(*CANVAS_WIDTH_RANGE)
    bg = random.randrange(COLOR_COUNT)

    config = _choose_config(H, W)
    if config is None:
        return None
    R, C, Mr, Mc, motifs_per_color = config

    dst_h = _spanned_size(R, DST_BLOCK_SIZE, DST_BLOCK_STEP)
    dst_w = _spanned_size(C, DST_BLOCK_SIZE, DST_BLOCK_STEP)
    motif_h = _spanned_size(Mr, MOTIF_SIZE, MOTIF_STEP)
    motif_w = _spanned_size(Mc, MOTIF_SIZE, MOTIF_STEP)
    placement = _place_regions(H, W, dst_h, dst_w, motif_h, motif_w)
    if placement is None:
        return None

    count = R * C
    decor_color, block_colors = _choose_palette(bg, count)
    motif_pattern_sets = {
        color: _choose_pattern_set(motifs_per_color) for color in block_colors
    }

    expected = _render_expected(bg, R, C, block_colors, motif_pattern_sets, decor_color)
    single_motif_baseline = _render_expected(
        bg,
        R,
        C,
        block_colors,
        motif_pattern_sets,
        decor_color,
        use_all_motifs=False,
    )
    erase_only_baseline = _render_expected(
        bg,
        R,
        C,
        block_colors,
        motif_pattern_sets,
        decor_color,
        keep_decor=False,
    )
    if np.array_equal(expected, single_motif_baseline):
        return None
    if np.array_equal(expected, erase_only_baseline):
        return None

    inp = _render_input(
        H,
        W,
        bg,
        R,
        C,
        Mr,
        Mc,
        block_colors,
        motif_pattern_sets,
        decor_color,
        placement,
    )

    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 58f5dbd5 combo input")
