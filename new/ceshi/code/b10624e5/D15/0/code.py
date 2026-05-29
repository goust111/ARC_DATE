# Variant of b10624e5
# Dimension: D15 — 源与目标的基数关系 (meta)
# Change: 1个源象限（左上）对多个目标象限（1:many），源图案是唯一的... → 多源多目标一一配对

import random

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, paste


# Design analysis for the D15 variant
# a) Concrete behavior:
#    The 27x27 frame and the cross divider stay unchanged. The visual change is that there are now
#    two source quadrants, each already containing a colored component attached to a red square anchor,
#    and two target quadrants that show only red anchors. The solver must extract both source motifs,
#    decide which target each one belongs to, then copy each source exactly once.
#    Small trace:
#    - Original flow: read TL patch once -> reflect/scale it into TR/BL/BR.
#    - New flow: read source A and source B -> sort sources by anchor size -> sort targets by anchor
#      size -> pair small-with-small and large-with-large -> for each pair reflect according to the
#      relative source/target quadrants and scale by target_size/source_size.
#    The changed step is therefore the source-selection and dispatch stage: one source no longer feeds
#    every target.
# b) Complexity calibration:
#    Pairing by visible size ranking is strong enough to be inferable from 2-3 examples, but still
#    different from the original task because the target's motif depends on which source anchor is
#    the smaller/larger one. To avoid ambiguity, the two source anchor sizes are distinct and the two
#    target anchor sizes are distinct. The two source motifs are also forced to differ even after flips.
# c) Hidden assumptions from the original main():
#    - `assert (0, 0) in anchors` and later TL-only extraction assumed a unique TL source.
#      This is no longer valid, so the new solver extracts the red-containing component for every
#      quadrant and classifies it as source vs target by whether that component contains non-red cells
#      beyond the anchor square.
#    - The old loop pasted one transformed copy into every non-TL quadrant. That would overwrite the
#      new one-to-one structure, so the new loop only iterates over paired source/target matches.
#    - Reflection previously depended only on absolute target position because the source was fixed.
#      With multiple sources, reflection must depend on the relative quadrant displacement between the
#      chosen source and its target.
# d) Placement legality:
#    The generator keeps sources in the left column and targets in the right column so the scene is
#    visually clean. Source patches are placed fully inside their quadrants. Target anchors are chosen
#    with a preference for positions where the whole transformed patch fits inside the target quadrant.
#    This guarantees non-overlap with divider lines and makes the produced output easy for the solver
#    to reconstruct.
# e) Visual observability:
#    The two source motifs are non-symmetric block patterns, and the row order of "small" vs "large"
#    is randomized independently for sources and targets. This means some samples map top->top while
#    others map bottom->top, making the size-based pairing visible instead of reducible to a fixed row
#    rule. At least one pair is also forced to undergo real scaling, so the anchor-size information is
#    doing visible work.
# f) Visual cleanliness:
#    Only one connected motif lives in each source quadrant, and targets are empty except for the red
#    anchor in the input. The divider color is not reused inside motifs, which avoids confusion between
#    object pixels and the cross separator.
# g) Color inference:
#    No hidden color-id ordering is used. Colors only decorate the motifs so that the two sources are
#    easy to tell apart. Pairing is controlled solely by visible anchor sizes.


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


def _quadrant_bounds(h: int, w: int, mid_r: int, mid_c: int, key: tuple[int, int]) -> tuple[int, int, int, int]:
    qb, qr = key
    r0 = 0 if qb == 0 else mid_r + 1
    r1 = mid_r if qb == 0 else h
    c0 = 0 if qr == 0 else mid_c + 1
    c1 = mid_c if qr == 0 else w
    return r0, r1, c0, c1


def _extract_anchor_patch(
    g: np.ndarray,
    bg: int,
    mid_r: int,
    mid_c: int,
    key: tuple[int, int],
    anchor_bbox: tuple[int, int, int, int],
) -> np.ndarray:
    h, w = g.shape
    q_r0, q_r1, q_c0, q_c1 = _quadrant_bounds(h, w, mid_r, mid_c, key)
    quad = g[q_r0:q_r1, q_c0:q_c1]
    local_bbox = (
        anchor_bbox[0] - q_r0,
        anchor_bbox[1] - q_c0,
        anchor_bbox[2] - q_r0,
        anchor_bbox[3] - q_c0,
    )
    patch_bb = None
    for pts in components_mask(quad != bg, connectivity=4):
        if any(local_bbox[0] <= r <= local_bbox[2] and local_bbox[1] <= c <= local_bbox[3] for r, c in pts):
            r0, c0, r1, c1 = bbox_coords(pts)
            patch_bb = (int(r0), int(c0), int(r1), int(c1))
            break
    assert patch_bb is not None, "quadrant with anchor must contain a non-bg component through that anchor"
    pr0, pc0, pr1, pc1 = patch_bb
    return quad[pr0 : pr1 + 1, pc0 : pc1 + 1].copy()


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


def _stamp(out: np.ndarray, patch_s: np.ndarray, anchor_r: int, anchor_c: int, bg: int, mid_r: int, mid_c: int) -> None:
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
        if rr_t == mid_r or cc_t == mid_c:
            continue
        if out[rr_t, cc_t] == RED:
            continue
        out[rr_t, cc_t] = int(patch_s[rr0, cc0])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete two independent source-to-target motif transfers inside the four-quadrant frame.

    Input structure:
    - The grid is split into four quadrants by one full-row and one full-column divider of a single non-bg color.
    - Every quadrant contains one red square anchor block, but exactly two quadrants also contain an attached colored motif.
    - The two source anchors have different sizes, and the two target anchors also have different sizes.

    Rule:
    - Extract the red-connected motif from each source quadrant.
    - Order the two source quadrants by anchor size and order the two target quadrants by anchor size; pair smallest with smallest and largest with largest.
    - For each pair, reflect horizontally when source and target lie in opposite columns, reflect vertically when they lie in opposite rows, then scale by target_size / source_size and align the transformed red region onto the target anchor.

    Key insight:
    - Anchor size now serves two purposes at once: it decides which source feeds which target, and it still provides the scale ratio used for that matched copy.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    h, w = g.shape

    bg = get_bg(g)
    mid_r, mid_c = _find_dividers(g, bg)

    red = components_mask(g == RED, connectivity=4)
    assert len(red) == 4, "grid must contain exactly four red anchor blocks"

    anchors: dict[tuple[int, int], dict[str, object]] = {}
    for pts in red:
        r0, c0, r1, c1 = bbox_coords(pts)
        rr = (r0 + r1) // 2
        cc = (c0 + c1) // 2
        qb = 0 if rr < mid_r else 1
        qr = 0 if cc < mid_c else 1
        size_h = int(r1 - r0 + 1)
        size_w = int(c1 - c0 + 1)
        assert size_h == size_w, "anchor blocks must be square"
        key = (qb, qr)
        assert key not in anchors, "each quadrant must contain exactly one red anchor block"
        patch = _extract_anchor_patch(g, bg, mid_r, mid_c, key, (r0, c0, r1, c1))
        anchors[key] = {
            "anchor_r0": int(r0),
            "anchor_c0": int(c0),
            "size": size_h,
            "patch": patch,
            "is_source": int(np.count_nonzero(patch != bg)) > size_h * size_h,
        }

    sources = [key for key, info in anchors.items() if bool(info["is_source"])]
    targets = [key for key, info in anchors.items() if not bool(info["is_source"])]
    assert len(sources) == 2 and len(targets) == 2, "variant requires exactly two source quadrants and two targets"

    source_sizes = sorted(int(anchors[key]["size"]) for key in sources)
    target_sizes = sorted(int(anchors[key]["size"]) for key in targets)
    assert source_sizes[0] != source_sizes[1], "source anchors must have distinct sizes"
    assert target_sizes[0] != target_sizes[1], "target anchors must have distinct sizes"

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
            mid_r,
            mid_c,
        )

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
SOURCE_KEYS = [(0, 0), (1, 0)]
TARGET_KEYS = [(0, 1), (1, 1)]
SOURCE_SIZE_CHOICES = [1, 2, 3]
TARGET_SIZE_CHOICES = [1, 2, 3, 4]
BLOCK_DIMS = [(2, 2), (2, 3), (3, 2), (3, 3)]
FILL_RATIO_MIN = 0.6
FILL_RATIO_MAX = 0.9
MOTIF_COLORS_MIN = 3
MOTIF_COLORS_MAX = 4
COLOR_POOL = [0, 3, 5, 6, 7, 8, 9]
MAX_RETRIES = 400


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
    variants = []
    for t in (arr, np.fliplr(arr), np.flipud(arr), np.flipud(np.fliplr(arr))):
        variants.append((t.shape[0], t.shape[1], tuple(map(int, t.flatten()))))
    return min(variants)


def _is_asymmetric(arr: np.ndarray) -> bool:
    return not (
        np.array_equal(arr, np.fliplr(arr))
        or np.array_equal(arr, np.flipud(arr))
        or np.array_equal(arr, np.flipud(np.fliplr(arr)))
    )


def _source_patch(anchor_size: int, forbidden: set[tuple]) -> tuple[np.ndarray, int, int, tuple] | None:
    for _ in range(80):
        lh, lw = random.choice(BLOCK_DIMS)
        n_cells = lh * lw
        min_fill = max(4, int(np.ceil(FILL_RATIO_MIN * n_cells)))
        max_fill = max(min_fill, int(np.floor(FILL_RATIO_MAX * n_cells)))
        ar, ac = random.randrange(lh), random.randrange(lw)
        filled = _grow_4conn(lh, lw, {(ar, ac)}, random.randint(min_fill, max_fill))
        logical = _assign_colors(filled, {(ar, ac)}, lh, lw)
        if logical is None:
            continue
        if not _is_asymmetric(logical):
            continue
        sig = _canonical_signature(logical)
        if sig in forbidden:
            continue
        patch = np.repeat(np.repeat(logical, anchor_size, axis=0), anchor_size, axis=1)
        return patch, ar * anchor_size, ac * anchor_size, sig
    return None


def _choose_patch_top_left(key: tuple[int, int], patch: np.ndarray) -> tuple[int, int]:
    q_r0, q_r1, q_c0, q_c1 = QUADS[key]
    max_r0 = q_r1 - patch.shape[0] + 1
    max_c0 = q_c1 - patch.shape[1] + 1
    assert max_r0 >= q_r0 and max_c0 >= q_c0, "source patch must fit inside its quadrant"
    return random.randint(q_r0, max_r0), random.randint(q_c0, max_c0)


def _choose_anchor(key: tuple[int, int], size: int, patch_s: np.ndarray) -> tuple[int, int]:
    q_r0, q_r1, q_c0, q_c1 = QUADS[key]
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
    if r_lo <= r_hi and c_lo <= c_hi:
        return random.randint(r_lo, r_hi), random.randint(c_lo, c_hi)
    return random.randint(anch_r0, anch_r1), random.randint(anch_c0, anch_c1)


def _gen_attempt() -> np.ndarray | None:
    source_sizes = sorted(random.sample(SOURCE_SIZE_CHOICES, 2))
    target_sizes = sorted(random.sample(TARGET_SIZE_CHOICES, 2))
    if source_sizes == target_sizes:
        return None

    small_source_key = random.choice(SOURCE_KEYS)
    large_source_key = SOURCE_KEYS[0] if small_source_key == SOURCE_KEYS[1] else SOURCE_KEYS[1]
    small_target_key = random.choice(TARGET_KEYS)
    large_target_key = TARGET_KEYS[0] if small_target_key == TARGET_KEYS[1] else TARGET_KEYS[1]

    rank_to_source_key = {0: small_source_key, 1: large_source_key}
    rank_to_target_key = {0: small_target_key, 1: large_target_key}
    source_size_by_key = {
        rank_to_source_key[0]: source_sizes[0],
        rank_to_source_key[1]: source_sizes[1],
    }
    target_size_by_key = {
        rank_to_target_key[0]: target_sizes[0],
        rank_to_target_key[1]: target_sizes[1],
    }

    forbidden: set[tuple] = set()
    source_specs: dict[tuple[int, int], dict[str, object]] = {}
    for rank in (0, 1):
        src_key = rank_to_source_key[rank]
        built = _source_patch(source_size_by_key[src_key], forbidden)
        if built is None:
            return None
        patch, anchor_r_in_patch, anchor_c_in_patch, sig = built
        forbidden.add(sig)
        source_specs[src_key] = {
            "patch": patch,
            "anchor_r_in_patch": anchor_r_in_patch,
            "anchor_c_in_patch": anchor_c_in_patch,
            "size": source_size_by_key[src_key],
        }

    pair_specs: list[dict[str, object]] = []
    for rank in (0, 1):
        src_key = rank_to_source_key[rank]
        dst_key = rank_to_target_key[rank]
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
        anchors[dst_key] = _choose_anchor(dst_key, int(pair["target_size"]), np.asarray(pair["patch_s"], dtype=int))

    out = np.full((GRID_SIZE, GRID_SIZE), BG, dtype=int)
    out[MID, :] = DIV
    out[:, MID] = DIV

    for src_key, spec in source_specs.items():
        pr0, pc0 = source_patch_topleft[src_key]
        paste(out, np.asarray(spec["patch"], dtype=int), pr0, pc0)

    for dst_key, size in target_size_by_key.items():
        paste(out, np.full((size, size), RED, dtype=int), anchors[dst_key][0], anchors[dst_key][1])

    for pair in pair_specs:
        _stamp(out, np.asarray(pair["patch_s"], dtype=int), anchors[pair["dst_key"]][0], anchors[pair["dst_key"]][1], BG, MID, MID)

    inp = out.copy()
    for dst_key, size in target_size_by_key.items():
        q_r0, q_r1, q_c0, q_c1 = QUADS[dst_key]
        inp[q_r0 : q_r1 + 1, q_c0 : q_c1 + 1] = BG
        paste(inp, np.full((size, size), RED, dtype=int), anchors[dst_key][0], anchors[dst_key][1])

    return inp if np.array_equal(main(inp), out) else None


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid b10624e5 D15 input")
