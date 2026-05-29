# Variant of b10624e5
# Dimension: D12 — 符号编码：图例或参数标记 (symbol_instructions)
# Change: 无符号编码——缩放因子和反射轴均从空间关系隐式推导... → 颜色映射图例定义图案重着色

import random

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, paste


# Design analysis
# ---------------
# spec_variant:
#   core_preserved:
#     - 27x27 grid with one full divider row and one full divider column crossing at the center.
#     - One red square anchor per quadrant; TL still owns the source motif.
#     - Non-TL quadrants still receive reflected + scaled copies keyed by anchor size and quadrant.
#   D12 concretization:
#     - A fixed legend sits in the bottom-left corner of the whole grid.
#     - The legend is a 2xK strip, K in {2,3}; row h-2 lists source colors, row h-1 lists target colors.
#     - Only non-red motif colors are remapped; bg, divider, and red anchor stay unchanged.
#   behavior_trace:
#     - Original flow: reflect/scale the TL patch, then stamp its colors directly beside the target anchor.
#     - Variant flow: reflect/scale the TL patch, then replace every non-red color c using the legend pair
#       c -> mapped(c), and only then stamp beside the target anchor.
#     - Example: if the TL motif uses {3, 8} and the legend bottom-left shows top row [3, 8],
#       bottom row [7, 0], then a reflected target copy changes every 3 to 7 and every 8 to 0.
#   complexity_calibration:
#     - The legend has only 2 or 3 pairs so humans can infer it from 2-3 examples.
#     - Source and target palettes are chosen disjointly, guaranteeing visible recoloring.
#     - The source motif is generated with 2x2 logical blocks so scaling remains lossless and the
#       recoloring stays visually legible after resize.
#   implicit_assumption_review:
#     - Original solver directly stamped patch colors; that would silently ignore D12, so recoloring is
#       now an explicit pre-stamp step.
#     - Original generator allowed divider color inside motifs; that would muddy the legend/divider roles,
#       so motif palettes now exclude divider, bg, and red.
#     - Original placement could use the entire BL quadrant; the legend now occupies a reserved corner,
#       so anchor placement enumerates only full-fit positions that avoid that zone.
#   placement_legality:
#     - The legend uses rows 25-26 and columns 0..K-1, with a one-cell moat above/right kept empty.
#     - BL anchor positions are searched exhaustively so the scaled target patch never overlaps the legend.
#     - All other quadrants also use full-fit anchor placements, keeping the output clean and solver-safe.
#   observability:
#     - TL keeps the unrecolored motif, so the source palette is always visible.
#     - Non-TL outputs always use the legend's target palette, making the source->target mapping explicit.
#     - At least two mapped colors appear in every motif, preventing the recoloring rule from degenerating.
#   solver_assumptions_added_by_variant:
#     - The legend is always the contiguous non-bg prefix in the last two rows, starting at column 0.
#     - Every non-red color present in the TL motif appears exactly once in the legend's source row.
#     - The legend itself contains no red or bg cells, and generate_input() enforces all of these.


def _find_dividers(g: np.ndarray, bg: int) -> tuple[int, int]:
    """
    Return the unique full-row/full-col quadrant divider: (mid_row, mid_col).
    """
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
    rr = np.clip(np.floor(np.arange(new_h) * src_h / new_h).astype(int), 0, src_h - 1)
    cc = np.clip(np.floor(np.arange(new_w) * src_w / new_w).astype(int), 0, src_w - 1)
    return img[rr[:, None], cc[None, :]]


def _read_legend(g: np.ndarray, bg: int, mid_r: int, mid_c: int) -> tuple[dict[int, int], np.ndarray]:
    h, w = g.shape
    src_r, tgt_r = h - 2, h - 1
    k = 0
    while k < mid_c and g[src_r, k] != bg and g[tgt_r, k] != bg:
        k += 1
    assert 2 <= k <= 3, "legend must contain 2 or 3 source->target pairs"
    assert src_r - 1 > mid_r, "legend needs a spacer row inside the bottom-left quadrant"
    assert np.all(g[src_r - 1, : min(mid_c, k + 1)] == bg), "legend must be separated from the motif area"
    if k < mid_c:
        assert g[src_r, k] == bg and g[tgt_r, k] == bg, "legend must be a contiguous strip from the left edge"

    mapping: dict[int, int] = {}
    legend_mask = np.zeros((h, w), dtype=bool)
    legend_mask[src_r : tgt_r + 1, :k] = True
    for c in range(k):
        src = int(g[src_r, c])
        tgt = int(g[tgt_r, c])
        assert src not in {bg, RED} and tgt not in {bg, RED}, "legend may only use non-bg, non-red colors"
        assert src not in mapping, "legend source colors must be unique"
        mapping[src] = tgt
    assert len(set(mapping.values())) == k, "legend target colors must be unique"
    return mapping, legend_mask


def _recolor_patch(patch: np.ndarray, mapping: dict[int, int], bg: int) -> np.ndarray:
    out = patch.copy()
    colors = {int(v) for v in np.unique(patch)} - {bg, RED}
    assert colors <= set(mapping), "legend must cover every non-red source color"
    for src, tgt in mapping.items():
        out[patch == src] = tgt
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Replicate the top-left motif across quadrants, but recolor copied pixels using a corner legend.

    Input structure:
    - Grid divided into four quadrants by one full non-bg row and one full non-bg column.
    - Each quadrant contains a red (2) square anchor block, with the top-left quadrant also holding the source motif.
    - The bottom-left corner contains a 2xK color legend: source colors on the upper row, target colors below.

    Rule:
    - Extract the top-left motif component that contains the TL red anchor.
    - For each other quadrant, reflect and scale that motif exactly as in the base task so its red region matches the target anchor.
    - Before stamping the transformed copy, replace every non-red motif color according to the legend's source->target pairs.

    Key insight: Anchor size and quadrant still control geometry, but the copied motif's palette is now explicitly encoded by the corner legend.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    out = g.copy()

    bg = get_bg(g)
    mid_r, mid_c = _find_dividers(g, bg)
    color_map, legend_mask = _read_legend(g, bg, mid_r, mid_c)

    red = components_mask(g == RED, connectivity=4)
    assert red, "grid must contain red anchor blocks"

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

    assert (0, 0) in anchors, "top-left quadrant must contain a red anchor block"
    _sr0, _sc0, sL = anchors[(0, 0)]

    tl = g[0:mid_r, 0:mid_c]
    comps = components_mask(tl != bg, connectivity=4)
    patch_bb = None
    for pts in comps:
        if any(tl[r, c] == RED for r, c in pts):
            r0, c0, r1, c1 = bbox_coords(pts)
            patch_bb = (int(r0), int(c0), int(r1), int(c1))
            break
    assert patch_bb is not None, "top-left quadrant must contain the source motif component"

    pr0, pc0, pr1, pc1 = patch_bb
    patch = tl[pr0 : pr1 + 1, pc0 : pc1 + 1].copy()

    for (qb, qr), (tr0, tc0, tL) in anchors.items():
        if (qb, qr) == (0, 0):
            continue

        patch_t = patch
        if qr == 1:
            patch_t = np.fliplr(patch_t)
        if qb == 1:
            patch_t = np.flipud(patch_t)

        ratio = float(tL) / float(sL)
        new_h = max(1, int(round(patch_t.shape[0] * ratio)))
        new_w = max(1, int(round(patch_t.shape[1] * ratio)))
        patch_s = _resize_nn(patch_t, new_h, new_w)
        roundtrip = _resize_nn(patch_s, patch_t.shape[0], patch_t.shape[1])
        assert np.array_equal(roundtrip, patch_t), "scaling must preserve all motif information"
        patch_s = _recolor_patch(patch_s, color_map, bg)

        rr_red, cc_red = np.where(patch_s == RED)
        assert rr_red.size > 0, "scaled source patch must contain red anchor pixels"
        ar0, ac0 = int(rr_red.min()), int(cc_red.min())
        place_r0 = tr0 - ar0
        place_c0 = tc0 - ac0

        rr_p, cc_p = np.where(patch_s != bg)
        for rr0, cc0 in zip(rr_p.tolist(), cc_p.tolist(), strict=False):
            val = int(patch_s[rr0, cc0])
            rr_t = place_r0 + rr0
            cc_t = place_c0 + cc0
            if not (0 <= rr_t < h and 0 <= cc_t < w):
                continue
            if rr_t == mid_r or cc_t == mid_c:
                continue
            if legend_mask[rr_t, cc_t]:
                continue
            if out[rr_t, cc_t] == RED:
                continue
            out[rr_t, cc_t] = val

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

SOURCE_ANCHOR_SIZE = 2
BLOCK_DIMS = [(2, 2), (2, 3), (3, 2), (3, 3)]
FILL_RATIO_MIN = 0.5
FILL_RATIO_MAX = 0.8
TARGET_SIZE_RANGE = (1, 4)
COLOR_POOL = [0, 3, 5, 6, 7, 8, 9]
LEGEND_PAIR_RANGE = (2, 3)
MAX_RETRIES = 300


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
    filled: set[tuple[int, int]],
    red_cells: set[tuple[int, int]],
    h: int,
    w: int,
    source_colors: list[int],
) -> np.ndarray | None:
    rows = {r for r, _ in filled}
    cols = {c for _, c in filled}
    if rows != set(range(h)) or cols != set(range(w)):
        return None

    non_red = [p for p in filled if p not in red_cells]
    if len(non_red) < len(source_colors):
        return None

    random.shuffle(non_red)
    grid = np.full((h, w), BG, dtype=int)
    for r, c in red_cells:
        grid[r, c] = RED

    for idx, (r, c) in enumerate(non_red):
        if idx < len(source_colors):
            grid[r, c] = source_colors[idx]
        else:
            grid[r, c] = random.choice(source_colors)
    return grid


def _source_patch(source_colors: list[int]) -> tuple[np.ndarray, int, int] | None:
    S = SOURCE_ANCHOR_SIZE
    lh, lw = random.choice(BLOCK_DIMS)
    n_cells = lh * lw
    min_fill = max(1 + len(source_colors), 3, int(np.ceil(FILL_RATIO_MIN * n_cells)))
    max_fill = max(min_fill, int(np.floor(FILL_RATIO_MAX * n_cells)))
    if max_fill > n_cells:
        max_fill = n_cells
    if min_fill > max_fill:
        return None

    ar, ac = random.randrange(lh), random.randrange(lw)
    filled = _grow_4conn(lh, lw, {(ar, ac)}, random.randint(min_fill, max_fill))
    logical = _assign_colors(filled, {(ar, ac)}, lh, lw, source_colors)
    if logical is None:
        return None
    patch = np.repeat(np.repeat(logical, S, axis=0), S, axis=1)
    return patch, ar * S, ac * S


def _scaled_patch(patch: np.ndarray, qb: int, qr: int, target_size: int) -> tuple[np.ndarray, np.ndarray]:
    p = patch
    if qr:
        p = np.fliplr(p)
    if qb:
        p = np.flipud(p)
    ratio = target_size / SOURCE_ANCHOR_SIZE
    h = max(1, int(round(p.shape[0] * ratio)))
    w = max(1, int(round(p.shape[1] * ratio)))
    return p, _resize_nn(p, h, w)


def _legend_masks(k: int) -> tuple[np.ndarray, np.ndarray]:
    occupied = np.zeros((GRID_SIZE, GRID_SIZE), dtype=bool)
    occupied[GRID_SIZE - 2 : GRID_SIZE, :k] = True

    blocked = occupied.copy()
    blocked[GRID_SIZE - 3 : GRID_SIZE, : min(MID, k + 1)] = True
    return occupied, blocked


def _place_legend(grid: np.ndarray, source_colors: list[int], target_colors: list[int]) -> None:
    k = len(source_colors)
    grid[GRID_SIZE - 2, :k] = np.array(source_colors, dtype=int)
    grid[GRID_SIZE - 1, :k] = np.array(target_colors, dtype=int)


def _choose_anchor(
    qb: int,
    qr: int,
    size: int,
    patch_s: np.ndarray,
    blocked: np.ndarray | None = None,
) -> tuple[int, int] | None:
    q_r0, q_r1, q_c0, q_c1 = QUADS[(qb, qr)]
    rr_red, cc_red = np.where(patch_s == RED)
    red_r0, red_c0 = int(rr_red.min()), int(cc_red.min())
    non_bg = np.argwhere(patch_s != BG)

    candidates: list[tuple[int, int]] = []
    for ar in range(q_r0, q_r1 - size + 2):
        for ac in range(q_c0, q_c1 - size + 2):
            place_r0 = ar - red_r0
            place_c0 = ac - red_c0
            ok = True
            for pr, pc in non_bg.tolist():
                tr = place_r0 + pr
                tc = place_c0 + pc
                if not (q_r0 <= tr <= q_r1 and q_c0 <= tc <= q_c1):
                    ok = False
                    break
                if blocked is not None and blocked[tr, tc]:
                    ok = False
                    break
            if ok:
                candidates.append((ar, ac))
    if not candidates:
        return None
    return random.choice(candidates)


def _stamp(out: np.ndarray, patch_s: np.ndarray, anchor_r: int, anchor_c: int, blocked: np.ndarray | None) -> None:
    rr_red, cc_red = np.where(patch_s == RED)
    place_r0 = anchor_r - int(rr_red.min())
    place_c0 = anchor_c - int(cc_red.min())

    rr_p, cc_p = np.where(patch_s != BG)
    for rr0, cc0 in zip(rr_p.tolist(), cc_p.tolist(), strict=False):
        rr_t = place_r0 + rr0
        cc_t = place_c0 + cc0
        if rr_t == MID or cc_t == MID:
            continue
        if blocked is not None and blocked[rr_t, cc_t]:
            continue
        if out[rr_t, cc_t] == RED:
            continue
        out[rr_t, cc_t] = int(patch_s[rr0, cc0])


def _gen_attempt() -> np.ndarray | None:
    n_pairs = random.randint(*LEGEND_PAIR_RANGE)
    palette = random.sample(COLOR_POOL, 2 * n_pairs)
    source_colors = palette[:n_pairs]
    target_colors = palette[n_pairs:]
    color_map = {src: tgt for src, tgt in zip(source_colors, target_colors, strict=True)}
    legend_occupied, legend_blocked = _legend_masks(n_pairs)

    target_sizes = {
        (0, 0): SOURCE_ANCHOR_SIZE,
        (0, 1): random.randint(*TARGET_SIZE_RANGE),
        (1, 0): random.randint(*TARGET_SIZE_RANGE),
        (1, 1): random.randint(*TARGET_SIZE_RANGE),
    }

    source = _source_patch(source_colors)
    if source is None:
        return None
    patch, anchor_r_in_patch, anchor_c_in_patch = source

    scaled_raw: dict[tuple[int, int], np.ndarray] = {}
    scaled_recolored: dict[tuple[int, int], np.ndarray] = {}
    for key, size in target_sizes.items():
        if key == (0, 0):
            continue
        patch_t, patch_s = _scaled_patch(patch, key[0], key[1], size)
        roundtrip = _resize_nn(patch_s, patch_t.shape[0], patch_t.shape[1])
        if not np.array_equal(roundtrip, patch_t):
            return None
        scaled_raw[key] = patch_s
        scaled_recolored[key] = _recolor_patch(patch_s, color_map, BG)

    anchors: dict[tuple[int, int], tuple[int, int]] = {}
    tl_r0 = random.randint(0, MID - patch.shape[0])
    tl_c0 = random.randint(0, MID - patch.shape[1])
    anchors[(0, 0)] = (tl_r0 + anchor_r_in_patch, tl_c0 + anchor_c_in_patch)

    for key, patch_s in scaled_raw.items():
        blocked = legend_blocked if key == (1, 0) else None
        anchor = _choose_anchor(key[0], key[1], target_sizes[key], patch_s, blocked)
        if anchor is None:
            return None
        anchors[key] = anchor

    out = np.full((GRID_SIZE, GRID_SIZE), BG, dtype=int)
    out[MID, :] = DIV
    out[:, MID] = DIV
    _place_legend(out, source_colors, target_colors)
    paste(out, patch, tl_r0, tl_c0)
    for key, size in target_sizes.items():
        if key == (0, 0):
            continue
        paste(out, np.full((size, size), RED, dtype=int), *anchors[key])
    for key, patch_s in scaled_recolored.items():
        blocked = legend_occupied if key == (1, 0) else None
        _stamp(out, patch_s, *anchors[key], blocked)

    inp = np.full((GRID_SIZE, GRID_SIZE), BG, dtype=int)
    inp[MID, :] = DIV
    inp[:, MID] = DIV
    _place_legend(inp, source_colors, target_colors)
    paste(inp, patch, tl_r0, tl_c0)
    for key, size in target_sizes.items():
        if key == (0, 0):
            continue
        paste(inp, np.full((size, size), RED, dtype=int), *anchors[key])

    if not np.array_equal(main(inp), out):
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid b10624e5 D12 input")
