# Combo variant of b10624e5 (index 41)
# combo_spec: D8/1+D15/0
# Name: 三分区/六分区 + 多源多目标一一配对

import random

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, paste


# Design analysis
# a) Concrete behavior and trace:
#    The fused task uses the six-region branch of D8: a 27x27 grid is split by one full
#    divider row and two full divider columns into a 2x3 layout. Every region has one red
#    square anchor. Exactly three regions are sources, meaning they already contain one
#    colored component attached to that anchor; the other three regions are targets and show
#    only the red square in the input.
#    Relative to the original task:
#    - Original: read one source from the top-left quadrant, then copy it to all other
#      quadrants using quadrant-based reflection and anchor-size scaling.
#    - Fused: read three source motifs, sort the three source anchors by size, sort the
#      three target anchors by size, pair small/medium/large source with small/medium/large
#      target, then transform each source only into its matched target.
#    - Spatial transform now depends on source->target divider crossings in the 2x3 layout:
#      crossing an odd number of vertical dividers toggles LR reflection, and changing row
#      bands toggles UD reflection.
#    Small concrete trace:
#    source rank 0 might live in region (0,2) with anchor size 1, source rank 1 in (1,0)
#    with size 2, and source rank 2 in (0,1) with size 3. Their matched targets then carry
#    sizes 2,3,4 in three different regions. If rank 1 maps from (1,0) to (0,1), the motif
#    flips both horizontally and vertically and scales by 3/2 before being aligned to the
#    target red square. This combines D8's richer partition geometry with D15's rank-based
#    source dispatch.
# b) Complexity calibration:
#    Three source motifs and three targets are enough to make the pairing rule visible
#    without creating combinatorial noise. Source anchor sizes are fixed to {1,2,3} and
#    target anchor sizes to {2,3,4}; both sets are distinct, which makes the small/medium/
#    large pairing unambiguous. Generator layouts are constrained so each sample shows both
#    same-row and cross-row transfers and also both odd-column-crossing and even-column-
#    crossing transfers; otherwise one reflection axis could become invisible in a sample.
# c) Hidden assumptions that break from the original solver:
#    - There is no single privileged top-left source anymore.
#    - There are two divider columns, so binary left/right quadrant logic is invalid.
#    - Copy destinations are not "all non-source regions from one source"; they are matched
#      one-to-one after ranking by visible anchor size.
#    - Reflection depends on relative source/target position, not absolute target position.
#    The new solver therefore detects all six anchors, extracts the anchor-connected patch in
#    each region, classifies source vs target by whether that patch contains non-red cells
#    beyond the anchor, then pairs ranks before stamping.
# d) Placement legality:
#    Region widths are 8, 8, and 9. To guarantee that the largest target size 4 still fits,
#    source motifs are generated only from logical block grids of width at most 2 and height
#    at most 3, then expanded by the source anchor size. Nearest-neighbor scaling from source
#    size S to target size T becomes exact block resizing, so every transformed motif can be
#    fully placed inside any region via anchor placement that prefers full-fit positions.
# e) Visual observability:
#    Every source motif is forced to be asymmetric under LR and UD reflection, and the three
#    source motifs are distinct even up to flips. This lets a human see which source fed a
#    target and whether the transfer crossed one vertical divider, two vertical dividers, or
#    the horizontal divider. Because ranks are reassigned to different regions each sample,
#    pairing cannot be reduced to a fixed positional rule.
# f) Visual cleanliness:
#    Each region contains only one anchor-connected component. Source regions contain exactly
#    one motif, target regions contain only the anchor in the input, and outputs add exactly
#    one transformed motif per target. Divider color is excluded from motif colors, and the
#    generator avoids clipping whenever a full-fit anchor position exists.
# g) Color inference:
#    No color-id arithmetic is used. Colors are copied literally within each matched source;
#    anchor size ranking is the only pairing cue, and relative region placement is the only
#    cue for reflection.
# 9) New solver assumptions and generator guarantees:
#    - `main()` assumes exactly six square red anchors, one per region; `generate_input()`
#      constructs exactly that.
#    - `main()` assumes exactly three source regions and three target regions; the generator
#      enforces it by clearing target regions back to background after output construction.
#    - `main()` assumes source anchor sizes and target anchor sizes are each pairwise distinct
#      so the rank pairing is unique; the generator fixes the size sets accordingly.
#    - `main()` assumes every source motif is a single anchor-connected component and that the
#      transformed motif still contains red cells; the block-based generator guarantees both.


BG = 4
DIV = 1
RED = 2
GRID_SIZE = 27
DIV_ROW = 13
DIV_COLS = (8, 17)
SOURCE_SIZES = (1, 2, 3)
TARGET_SIZES = (2, 3, 4)
BLOCK_DIMS = ((2, 2), (3, 2))
FILL_RATIO_MIN = 0.6
FILL_RATIO_MAX = 0.9
MOTIF_COLORS_MIN = 3
MOTIF_COLORS_MAX = 4
COLOR_POOL = [0, 3, 5, 6, 7, 8, 9]
MAX_RETRIES = 500


def _find_dividers(g: np.ndarray, bg: int) -> tuple[int, tuple[int, int]]:
    """Return the unique 2x3 divider layout: one full row and two full cols of one color."""
    h, w = g.shape
    full_rows = [(r, int(g[r, 0])) for r in range(h) if np.all(g[r, :] == g[r, 0]) and int(g[r, 0]) != bg]
    full_cols = [(c, int(g[0, c])) for c in range(w) if np.all(g[:, c] == g[0, c]) and int(g[0, c]) != bg]
    colors = sorted({v for _, v in full_rows} & {v for _, v in full_cols})
    matches = []
    for color in colors:
        rows = sorted(r for r, v in full_rows if v == color)
        cols = sorted(c for c, v in full_cols if v == color)
        if len(rows) == 1 and len(cols) == 2:
            matches.append((int(rows[0]), (int(cols[0]), int(cols[1]))))
    assert len(matches) == 1, "grid must have exactly one matching full-row/two-full-col divider"
    return matches[0]


def _region_bounds(
    h: int, w: int, div_row: int, div_cols: tuple[int, int]
) -> dict[tuple[int, int], tuple[int, int, int, int]]:
    """Inclusive bounds for each region in the 2x3 layout."""
    c1, c2 = div_cols
    row_ranges = [(0, div_row - 1), (div_row + 1, h - 1)]
    col_ranges = [(0, c1 - 1), (c1 + 1, c2 - 1), (c2 + 1, w - 1)]
    return {
        (rb, cb): (r0, r1, c0, c1_)
        for rb, (r0, r1) in enumerate(row_ranges)
        for cb, (c0, c1_) in enumerate(col_ranges)
    }


REGIONS = _region_bounds(GRID_SIZE, GRID_SIZE, DIV_ROW, DIV_COLS)
ALL_KEYS = tuple(sorted(REGIONS))


def _region_key(rr: int, cc: int, div_row: int, div_cols: tuple[int, int]) -> tuple[int, int]:
    """Map a point to its (row band, col band) region key."""
    rb = 0 if rr < div_row else 1
    if cc < div_cols[0]:
        cb = 0
    elif cc < div_cols[1]:
        cb = 1
    else:
        cb = 2
    return rb, cb


def _resize_nn(img: np.ndarray, new_h: int, new_w: int) -> np.ndarray:
    """Nearest-neighbor resize."""
    src_h, src_w = img.shape
    if new_h <= 0 or new_w <= 0:
        return img.copy()
    rr = np.clip(np.floor(np.arange(new_h) * src_h / new_h).astype(int), 0, src_h - 1)
    cc = np.clip(np.floor(np.arange(new_w) * src_w / new_w).astype(int), 0, src_w - 1)
    return img[rr[:, None], cc[None, :]]


def _extract_anchor_patch(
    g: np.ndarray,
    bg: int,
    regions: dict[tuple[int, int], tuple[int, int, int, int]],
    key: tuple[int, int],
    anchor_bbox: tuple[int, int, int, int],
) -> np.ndarray:
    """Extract the anchor-connected component within one region."""
    q_r0, q_r1, q_c0, q_c1 = regions[key]
    region = g[q_r0 : q_r1 + 1, q_c0 : q_c1 + 1]
    local_bbox = (
        anchor_bbox[0] - q_r0,
        anchor_bbox[1] - q_c0,
        anchor_bbox[2] - q_r0,
        anchor_bbox[3] - q_c0,
    )
    patch_bb = None
    for pts in components_mask(region != bg, connectivity=4):
        if any(local_bbox[0] <= r <= local_bbox[2] and local_bbox[1] <= c <= local_bbox[3] for r, c in pts):
            r0, c0, r1, c1 = bbox_coords(pts)
            patch_bb = (int(r0), int(c0), int(r1), int(c1))
            break
    assert patch_bb is not None, "anchor region must contain a non-bg component through that anchor"
    pr0, pc0, pr1, pc1 = patch_bb
    return region[pr0 : pr1 + 1, pc0 : pc1 + 1].copy()


def _transform_patch(
    patch: np.ndarray,
    src_key: tuple[int, int],
    dst_key: tuple[int, int],
    source_size: int,
    target_size: int,
) -> np.ndarray:
    """Reflect by divider crossings, then scale by target/source anchor size."""
    p = patch
    if abs(src_key[1] - dst_key[1]) % 2 == 1:
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
    bg: int,
    div_row: int,
    div_cols: tuple[int, int],
) -> None:
    """Stamp a transformed patch aligned by the top-left red pixel of its anchor."""
    rr_red, cc_red = np.where(patch_s == RED)
    assert rr_red.size > 0, "transformed patch must contain red anchor pixels"
    place_r0 = int(anchor_r) - int(rr_red.min())
    place_c0 = int(anchor_c) - int(cc_red.min())
    src_rr, src_cc = np.where(patch_s != bg)
    for rr0, cc0 in zip(src_rr.tolist(), src_cc.tolist(), strict=False):
        rr_t = place_r0 + rr0
        cc_t = place_c0 + cc0
        if not (0 <= rr_t < out.shape[0] and 0 <= cc_t < out.shape[1]):
            continue
        if rr_t == div_row or cc_t in div_cols:
            continue
        if out[rr_t, cc_t] == RED:
            continue
        out[rr_t, cc_t] = int(patch_s[rr0, cc0])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete three independent motif transfers across a six-region layout using rank pairing and divider-crossing reflections.

    Input structure:
    - The grid is split into six regions by one full-row divider and two full-column dividers of one non-background color.
    - Every region contains one red square anchor block.
    - Exactly three regions also contain an attached colored motif, while the other three contain only the red anchor.

    Rule:
    - Extract the anchor-connected motif from each source region.
    - Sort the three source anchors by size and sort the three target anchors by size; pair smallest with smallest, medium with medium, and largest with largest.
    - For each matched pair, reflect left-right if source and target are separated by an odd number of vertical divider crossings, reflect up-down if they lie in different row bands, then scale by target_size / source_size and align the transformed red region onto the target anchor.

    Key insight:
    - Anchor size now controls both correspondence and scale, while the 2x3 region layout controls reflection through divider-crossing parity instead of fixed quadrant labels.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    bg = get_bg(g)
    div_row, div_cols = _find_dividers(g, bg)
    regions = _region_bounds(g.shape[0], g.shape[1], div_row, div_cols)

    red = components_mask(g == RED, connectivity=4)
    assert len(red) == 6, "grid must contain exactly six red anchor blocks"

    anchors: dict[tuple[int, int], dict[str, object]] = {}
    for pts in red:
        r0, c0, r1, c1 = bbox_coords(pts)
        size_h = int(r1 - r0 + 1)
        size_w = int(c1 - c0 + 1)
        assert size_h == size_w, "anchor blocks must be square"
        rr = (r0 + r1) // 2
        cc = (c0 + c1) // 2
        key = _region_key(rr, cc, div_row, div_cols)
        assert key not in anchors, "each region must contain exactly one red anchor"
        patch = _extract_anchor_patch(g, bg, regions, key, (r0, c0, r1, c1))
        anchors[key] = {
            "anchor_r0": int(r0),
            "anchor_c0": int(c0),
            "size": size_h,
            "patch": patch,
            "is_source": int(np.count_nonzero(patch != bg)) > size_h * size_h,
        }

    assert set(anchors) == set(regions), "there must be exactly one anchor in every region"
    sources = [key for key, info in anchors.items() if bool(info["is_source"])]
    targets = [key for key, info in anchors.items() if not bool(info["is_source"])]
    assert len(sources) == 3 and len(targets) == 3, "variant requires exactly three sources and three targets"

    source_sizes = sorted(int(anchors[key]["size"]) for key in sources)
    target_sizes = sorted(int(anchors[key]["size"]) for key in targets)
    assert len(set(source_sizes)) == 3, "source anchors must have distinct sizes"
    assert len(set(target_sizes)) == 3, "target anchors must have distinct sizes"

    sources = sorted(sources, key=lambda key: (int(anchors[key]["size"]), key))
    targets = sorted(targets, key=lambda key: (int(anchors[key]["size"]), key))

    for src_key, dst_key in zip(sources, targets, strict=True):
        src_info = anchors[src_key]
        dst_info = anchors[dst_key]
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
            bg,
            div_row,
            div_cols,
        )

    return out


def _grow_4conn(h: int, w: int, seeds: set[tuple[int, int]], n_fill: int) -> set[tuple[int, int]]:
    """Grow a 4-connected set from the given seeds."""
    filled = set(seeds)
    frontier = set()
    for r, c in seeds:
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in filled:
                frontier.add((nr, nc))
    while len(filled) < n_fill and frontier:
        cell = random.choice(list(frontier))
        frontier.remove(cell)
        filled.add(cell)
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
    """Assign colors to a logical motif while keeping the bbox fully occupied by rows/cols."""
    rows = {r for r, _ in filled}
    cols = {c for _, c in filled}
    if rows != set(range(h)) or cols != set(range(w)):
        return None
    non_red = [p for p in filled if p not in red_cells]
    if len(non_red) < MOTIF_COLORS_MIN - 1:
        return None
    random.shuffle(non_red)
    max_colors = min(MOTIF_COLORS_MAX, len(non_red) + 1)
    if max_colors < MOTIF_COLORS_MIN:
        return None
    n_colors = random.randint(MOTIF_COLORS_MIN, max_colors)
    extras = COLOR_POOL[:]
    random.shuffle(extras)
    extras = extras[: n_colors - 1]
    logical = np.full((h, w), BG, dtype=int)
    for r, c in red_cells:
        logical[r, c] = RED
    for i, (r, c) in enumerate(non_red):
        logical[r, c] = extras[i] if i < len(extras) else random.choice(extras)
    return logical


def _canonical_signature(arr: np.ndarray) -> tuple:
    """Canonicalize up to LR/UD reflections so source motifs stay visibly distinct."""
    variants = []
    for t in (arr, np.fliplr(arr), np.flipud(arr), np.flipud(np.fliplr(arr))):
        variants.append((t.shape[0], t.shape[1], tuple(map(int, t.flatten()))))
    return min(variants)


def _is_observable(arr: np.ndarray) -> bool:
    """Reject motifs whose reflections would be visually ambiguous."""
    return not (
        np.array_equal(arr, np.fliplr(arr))
        or np.array_equal(arr, np.flipud(arr))
        or np.array_equal(arr, np.flipud(np.fliplr(arr)))
    )


def _source_patch(anchor_size: int, forbidden: set[tuple]) -> tuple[np.ndarray, int, int, tuple] | None:
    """Build a block-structured source motif that scales losslessly to the target sizes."""
    for _ in range(80):
        lh, lw = random.choice(BLOCK_DIMS)
        n_cells = lh * lw
        min_fill = max(4, int(np.ceil(FILL_RATIO_MIN * n_cells)))
        max_fill = max(min_fill, int(np.floor(FILL_RATIO_MAX * n_cells)))
        ar, ac = random.randrange(lh), random.randrange(lw)
        filled = _grow_4conn(lh, lw, {(ar, ac)}, random.randint(min_fill, max_fill))
        logical = _assign_colors(filled, {(ar, ac)}, lh, lw)
        if logical is None or not _is_observable(logical):
            continue
        sig = _canonical_signature(logical)
        if sig in forbidden:
            continue
        patch = np.repeat(np.repeat(logical, anchor_size, axis=0), anchor_size, axis=1)
        return patch, ar * anchor_size, ac * anchor_size, sig
    return None


def _choose_patch_top_left(key: tuple[int, int], patch: np.ndarray) -> tuple[int, int]:
    """Place a source patch fully inside its region."""
    q_r0, q_r1, q_c0, q_c1 = REGIONS[key]
    max_r0 = q_r1 - patch.shape[0] + 1
    max_c0 = q_c1 - patch.shape[1] + 1
    assert max_r0 >= q_r0 and max_c0 >= q_c0, "source patch must fit inside its region"
    return random.randint(q_r0, max_r0), random.randint(q_c0, max_c0)


def _choose_anchor(key: tuple[int, int], size: int, patch_s: np.ndarray) -> tuple[int, int]:
    """Pick an anchor position inside a target region, preferring full motif fit."""
    q_r0, q_r1, q_c0, q_c1 = REGIONS[key]
    rr, cc = np.where(patch_s == RED)
    red_r0, red_c0 = int(rr.min()), int(cc.min())

    fit_r0 = q_r0 + red_r0
    fit_r1 = q_r1 - patch_s.shape[0] + 1 + red_r0
    fit_c0 = q_c0 + red_c0
    fit_c1 = q_c1 - patch_s.shape[1] + 1 + red_c0

    anch_r0, anch_r1 = q_r0, q_r1 - size + 1
    anch_c0, anch_c1 = q_c0, q_c1 - size + 1

    r_lo, r_hi = max(fit_r0, anch_r0), min(fit_r1, anch_r1)
    c_lo, c_hi = max(fit_c0, anch_c0), min(fit_c1, anch_c1)
    assert anch_r0 <= anch_r1 and anch_c0 <= anch_c1, "anchor must fit inside its region"
    if r_lo <= r_hi and c_lo <= c_hi:
        return random.randint(r_lo, r_hi), random.randint(c_lo, c_hi)
    return random.randint(anch_r0, anch_r1), random.randint(anch_c0, anch_c1)


def _choose_layout() -> list[tuple[tuple[int, int], tuple[int, int]]]:
    """Choose three source->target region pairs with visible transform diversity."""
    while True:
        source_keys = random.sample(list(ALL_KEYS), 3)
        target_keys = [key for key in ALL_KEYS if key not in source_keys]
        if len({key[0] for key in source_keys}) != 2:
            continue
        if len({key[0] for key in target_keys}) != 2:
            continue
        random.shuffle(source_keys)
        random.shuffle(target_keys)
        pairs = list(zip(source_keys, target_keys, strict=True))
        row_diffs = {src[0] != dst[0] for src, dst in pairs}
        col_parities = {abs(src[1] - dst[1]) % 2 for src, dst in pairs}
        transforms = {(src[0] != dst[0], abs(src[1] - dst[1]) % 2 == 1) for src, dst in pairs}
        if row_diffs != {False, True}:
            continue
        if col_parities != {0, 1}:
            continue
        if len(transforms) < 3:
            continue
        random.shuffle(pairs)
        return pairs


def _gen_attempt() -> np.ndarray | None:
    """Build one output-first training instance and return the corresponding input."""
    pairs = _choose_layout()

    source_specs: dict[tuple[int, int], dict[str, object]] = {}
    target_size_by_key: dict[tuple[int, int], int] = {}
    pair_specs: list[dict[str, object]] = []
    forbidden: set[tuple] = set()

    for rank, (src_key, dst_key) in enumerate(pairs):
        built = _source_patch(SOURCE_SIZES[rank], forbidden)
        if built is None:
            return None
        patch, anchor_r_in_patch, anchor_c_in_patch, sig = built
        forbidden.add(sig)
        source_specs[src_key] = {
            "patch": patch,
            "anchor_r_in_patch": anchor_r_in_patch,
            "anchor_c_in_patch": anchor_c_in_patch,
            "size": SOURCE_SIZES[rank],
        }
        target_size_by_key[dst_key] = TARGET_SIZES[rank]

    for rank, (src_key, dst_key) in enumerate(pairs):
        spec = source_specs[src_key]
        patch_s = _transform_patch(
            np.asarray(spec["patch"], dtype=int),
            src_key,
            dst_key,
            int(spec["size"]),
            target_size_by_key[dst_key],
        )
        pair_specs.append(
            {
                "src_key": src_key,
                "dst_key": dst_key,
                "patch_s": patch_s,
                "source_size": int(spec["size"]),
                "target_size": target_size_by_key[dst_key],
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
        )

    out = np.full((GRID_SIZE, GRID_SIZE), BG, dtype=int)
    out[DIV_ROW, :] = DIV
    out[:, DIV_COLS[0]] = DIV
    out[:, DIV_COLS[1]] = DIV

    for src_key, spec in source_specs.items():
        pr0, pc0 = source_patch_topleft[src_key]
        paste(out, np.asarray(spec["patch"], dtype=int), pr0, pc0)

    for dst_key, size in target_size_by_key.items():
        paste(out, np.full((size, size), RED, dtype=int), anchors[dst_key][0], anchors[dst_key][1])

    for pair in pair_specs:
        _stamp(
            out,
            np.asarray(pair["patch_s"], dtype=int),
            anchors[pair["dst_key"]][0],
            anchors[pair["dst_key"]][1],
            BG,
            DIV_ROW,
            DIV_COLS,
        )

    inp = out.copy()
    for dst_key, size in target_size_by_key.items():
        q_r0, q_r1, q_c0, q_c1 = REGIONS[dst_key]
        inp[q_r0 : q_r1 + 1, q_c0 : q_c1 + 1] = BG
        paste(inp, np.full((size, size), RED, dtype=int), anchors[dst_key][0], anchors[dst_key][1])

    return inp if np.array_equal(main(inp), out) else None


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid b10624e5 combo input")
