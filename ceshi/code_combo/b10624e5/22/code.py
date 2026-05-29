# Combo variant of b10624e5 (index 22)
# combo_spec: D12/2+D16/3
# Name: 锚块颜色编码缩放因子 + 颜色映射图例

import random

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, paste


# Design analysis
# ---------------
# spec_variant:
#   core_preserved:
#     - The board stays 27x27 with one full divider row and one full divider column crossing at the center.
#     - TL still contains the only full source motif; TR / BL / BR still receive reflected copies.
#     - Reflection remains purely positional: right quadrants mirror left-right, bottom quadrants mirror up-down.
#   fused_concretization:
#     - The old size-based target anchors are replaced by same-size colored seed anchors in the non-TL quadrants.
#     - A bottom-left 2-row legend is made of exactly two horizontal runs. In each run, the upper color is a
#       source motif color and the lower color is its recolored target. The run length is either 2 or 3.
#     - The lower-row legend color also serves as an allowed non-TL anchor color. Matching an anchor color to its
#       legend run yields the scale factor: target-color-in-run-of-length-2 => 2x, run-of-length-3 => 3x.
#   behavior_trace:
#     - Original task: read target anchor size, use size ratio for scaling, then stamp the reflected motif unchanged.
#     - This combo: first read the legend as two source->target mappings, then read the same legend again to decode
#       target-color->scale from run lengths, then use each target anchor color to choose 2x or 3x scaling.
#     - Example: if the legend is top [5,5,8,8,8] and bottom [7,7,9,9,9], then 5->7 and 8->9 for recoloring,
#       while anchor color 7 means scale 2 and anchor color 9 means scale 3.
#   complexity_calibration:
#     - Only two scale factors are used, exactly once each in the legend, so the symbol system is small enough to learn.
#     - The source motif uses exactly two non-red colors, and both appear in the legend, so recoloring is visible and local.
#     - All non-TL input anchors use the same 2x2 seed size, preventing accidental fallback to the original size rule.
#   implicit_assumption_review:
#     - The original solver assumed every target anchor was red and its side length directly encoded scale; that is no
#       longer true, so anchor detection now depends on legend target colors and scaling depends on legend run lengths.
#     - The original solver aligned a scaled red region onto an equally scaled target block. Here the input anchor is
#       only a same-size seed, so the copied anchor region is expanded from that seed's top-left corner after scaling.
#     - The original D12 single-cell legend could be parsed by column count alone; this combo needs run parsing, because
#       repeated legend cells now encode both recoloring pairs and scale values.
#   placement_legality:
#     - The source patch is always a 4x4 pixel upsample of a 2x2 logical motif, so 3x scaling tops out at 12x12 and
#       still fits inside a 13x13 quadrant.
#     - The legend occupies the bottom-left corner plus a one-cell moat above/right; BL placements are enumerated against
#       that blocker, and BL is forced to use the 2x code so a full transformed patch can fit cleanly.
#     - Every target patch is placed by exhaustive candidate search that checks full in-quadrant fit and no legend overlap.
#   observability_and_neatness:
#     - The legend is compact and isolated; the source motif stays in TL with the unmapped palette, while copied motifs
#       switch to the target palette and also expand according to the target anchor color.
#     - Because both anchor colors appear among the three target quadrants, solvers get repeated evidence that color,
#       not seed size, controls 2x vs 3x growth.
#     - Keeping only two source colors and two target colors avoids a visually noisy grid while still exposing both layers.
#   solver_assumptions_added_by_variant:
#     - The legend is a contiguous prefix in the last two rows, split into exactly two constant-color runs of lengths 2 and 3.
#     - Each non-TL quadrant contains exactly one solid square seed anchor whose color is one of the legend target colors.
#     - The source motif contains every legend source color at least once, and generate_input() enforces all of these.


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
TARGET_SEED_SIZE = 2
LEGEND_SCALES = (2, 3)
LOGICAL_SHAPE = (2, 2)
FILL_CHOICES = (3, 4)
COLOR_POOL = [3, 5, 6, 7, 8, 9]
MAX_RETRIES = 400


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
    rr = np.clip(np.floor(np.arange(new_h) * src_h / new_h).astype(int), 0, src_h - 1)
    cc = np.clip(np.floor(np.arange(new_w) * src_w / new_w).astype(int), 0, src_w - 1)
    return img[rr[:, None], cc[None, :]]


def _parse_legend(g: np.ndarray, bg: int, mid_r: int, mid_c: int) -> tuple[dict[int, int], dict[int, int], np.ndarray]:
    h, _w = g.shape
    src_r, tgt_r = h - 2, h - 1
    assert src_r - 1 > mid_r, "legend must live below the divider with a spacer row"

    c = 0
    mapping: dict[int, int] = {}
    scale_map: dict[int, int] = {}
    legend_mask = np.zeros_like(g, dtype=bool)
    run_lengths: list[int] = []

    while c < mid_c and g[src_r, c] != bg and g[tgt_r, c] != bg:
        src = int(g[src_r, c])
        tgt = int(g[tgt_r, c])
        assert src not in {bg, RED, DIV} and tgt not in {bg, RED, DIV}, "legend colors must be non-bg, non-divider, non-red"
        c0 = c
        while c < mid_c and int(g[src_r, c]) == src and int(g[tgt_r, c]) == tgt:
            c += 1
        run_len = c - c0
        assert run_len in LEGEND_SCALES, "legend run lengths must encode 2x or 3x"
        assert src not in mapping, "legend source colors must be unique"
        assert tgt not in scale_map, "legend target colors must be unique"
        mapping[src] = tgt
        scale_map[tgt] = run_len
        run_lengths.append(run_len)
        legend_mask[src_r : tgt_r + 1, c0:c] = True

    assert sorted(run_lengths) == [2, 3], "legend must contain exactly one 2-cell run and one 3-cell run"
    assert np.all(g[src_r - 1, : min(mid_c, c + 1)] == bg), "legend must have a blank spacer row above it"
    if c < mid_c:
        assert g[src_r, c] == bg and g[tgt_r, c] == bg, "legend must form one contiguous prefix from the left edge"
    return mapping, scale_map, legend_mask


def _recolor_scaled_patch(raw_patch: np.ndarray, mapping: dict[int, int], anchor_color: int, bg: int) -> np.ndarray:
    out = raw_patch.copy()
    source_colors = {int(v) for v in np.unique(raw_patch)} - {bg, RED}
    assert source_colors == set(mapping), "legend must cover every non-red source color exactly"
    for src, tgt in mapping.items():
        out[raw_patch == src] = tgt
    out[raw_patch == RED] = anchor_color
    return out


def _reflect_patch(patch: np.ndarray, qb: int, qr: int) -> np.ndarray:
    out = patch
    if qr == 1:
        out = np.fliplr(out)
    if qb == 1:
        out = np.flipud(out)
    return out


def _is_solid_square(mask: np.ndarray, pts: np.ndarray) -> tuple[bool, tuple[int, int, int, int]]:
    r0, c0, r1, c1 = bbox_coords(pts)
    ok = (r1 - r0) == (c1 - c0) and np.all(mask[r0 : r1 + 1, c0 : c1 + 1])
    return bool(ok), (int(r0), int(c0), int(r1), int(c1))


def _extract_source_patch(tl: np.ndarray, bg: int) -> tuple[np.ndarray, tuple[int, int]]:
    comps = components_mask(tl != bg, connectivity=4)
    patch_bb = None
    for pts in comps:
        if any(int(tl[r, c]) == RED for r, c in pts):
            r0, c0, r1, c1 = bbox_coords(pts)
            patch_bb = (int(r0), int(c0), int(r1), int(c1))
            break
    assert patch_bb is not None, "top-left quadrant must contain the source motif component"
    pr0, pc0, pr1, pc1 = patch_bb
    patch = tl[pr0 : pr1 + 1, pc0 : pc1 + 1].copy()
    rr_red, cc_red = np.where(patch == RED)
    assert rr_red.size > 0, "source motif must contain a red source anchor"
    return patch, (int(rr_red.min()), int(cc_red.min()))


def _find_target_anchor(
    quad: np.ndarray,
    allowed_colors: set[int],
) -> tuple[int, int, int]:
    found: list[tuple[int, int, int]] = []
    for color in sorted(allowed_colors):
        comps = components_mask(quad == color, connectivity=4)
        for pts in comps:
            ok, (r0, c0, _r1, _c1) = _is_solid_square(quad == color, pts)
            if ok:
                found.append((int(r0), int(c0), int(color)))
    assert len(found) == 1, "each non-TL quadrant must contain exactly one square anchor seed"
    return found[0]


def _scaled_variant(
    patch: np.ndarray,
    qb: int,
    qr: int,
    factor: int,
    color_map: dict[int, int],
    anchor_color: int,
) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    patch_t = _reflect_patch(patch, qb, qr)
    new_h = patch_t.shape[0] * factor
    new_w = patch_t.shape[1] * factor
    scaled_raw = _resize_nn(patch_t, new_h, new_w)
    roundtrip = _resize_nn(scaled_raw, patch_t.shape[0], patch_t.shape[1])
    assert np.array_equal(roundtrip, patch_t), "scaling must preserve motif information"
    rr_red, cc_red = np.where(scaled_raw == RED)
    assert rr_red.size > 0, "scaled motif must retain the source anchor region"
    anchor_offset = (int(rr_red.min()), int(cc_red.min()))
    scaled_colored = _recolor_scaled_patch(scaled_raw, color_map, anchor_color, BG)
    return scaled_raw, scaled_colored, anchor_offset


def main(grid: np.ndarray) -> np.ndarray:
    """
    Copy the TL motif into the other quadrants using a legend-coded palette swap and a legend-coded scale chosen by anchor color.

    Input structure:
    - A single-color full row and full column split the 27x27 grid into four quadrants.
    - TL contains the source motif with a red 2x2 source anchor embedded inside it.
    - TR / BL / BR each contain only one same-size colored square seed anchor, and the bottom-left corner holds a 2-row run-length legend.

    Rule:
    - Read the bottom-left legend as two source-color -> target-color mappings, and also treat each lower-row run length as that target color's scale factor.
    - Extract the TL motif component containing the red anchor, reflect it by quadrant, scale it by the factor attached to the target anchor color, then recolor all non-red motif colors through the legend.
    - Replace the copied anchor region with the target anchor color, align that expanded anchor region to the seed anchor's top-left corner, and stamp the transformed motif while leaving the divider and legend untouched.

    Key insight: The legend carries two linked symbol systems at once: the lower-row color tells you how to recolor copied pixels and, via its run length, how much that quadrant should be scaled.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    out = g.copy()

    bg = get_bg(g)
    mid_r, mid_c = _find_dividers(g, bg)
    color_map, scale_map, legend_mask = _parse_legend(g, bg, mid_r, mid_c)

    tl = g[:mid_r, :mid_c]
    patch, _source_anchor_offset = _extract_source_patch(tl, bg)

    anchors: dict[tuple[int, int], tuple[int, int, int]] = {}
    for key in ((0, 1), (1, 0), (1, 1)):
        q_r0, q_r1, q_c0, q_c1 = QUADS[key]
        quad = g[q_r0 : q_r1 + 1, q_c0 : q_c1 + 1]
        r0, c0, color = _find_target_anchor(quad, set(scale_map))
        anchors[key] = (q_r0 + r0, q_c0 + c0, color)

    for (qb, qr), (seed_r0, seed_c0, anchor_color) in anchors.items():
        factor = scale_map[anchor_color]
        raw_patch, colored_patch, (anchor_r0, anchor_c0) = _scaled_variant(
            patch,
            qb,
            qr,
            factor,
            color_map,
            anchor_color,
        )
        place_r0 = seed_r0 - anchor_r0
        place_c0 = seed_c0 - anchor_c0
        rr_p, cc_p = np.where(raw_patch != bg)
        for rr0, cc0 in zip(rr_p.tolist(), cc_p.tolist(), strict=False):
            rr_t = place_r0 + rr0
            cc_t = place_c0 + cc0
            if not (0 <= rr_t < h and 0 <= cc_t < w):
                continue
            if rr_t == mid_r or cc_t == mid_c:
                continue
            if legend_mask[rr_t, cc_t]:
                continue
            out[rr_t, cc_t] = int(colored_patch[rr0, cc0])

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


def _build_source_patch(source_colors: list[int]) -> tuple[np.ndarray, int, int] | None:
    h, w = LOGICAL_SHAPE
    ar, ac = random.randrange(h), random.randrange(w)
    filled = _grow_4conn(h, w, {(ar, ac)}, random.choice(FILL_CHOICES))
    rows = {r for r, _ in filled}
    cols = {c for _, c in filled}
    if rows != set(range(h)) or cols != set(range(w)):
        return None

    non_red = [cell for cell in filled if cell != (ar, ac)]
    if len(non_red) < len(source_colors):
        return None
    random.shuffle(non_red)

    logical = np.full((h, w), BG, dtype=int)
    logical[ar, ac] = RED
    for idx, (r, c) in enumerate(non_red):
        logical[r, c] = source_colors[idx] if idx < len(source_colors) else random.choice(source_colors)

    patch = np.repeat(np.repeat(logical, SOURCE_ANCHOR_SIZE, axis=0), SOURCE_ANCHOR_SIZE, axis=1)
    return patch, ar * SOURCE_ANCHOR_SIZE, ac * SOURCE_ANCHOR_SIZE


def _legend_blockers(entries: list[tuple[int, int, int]]) -> tuple[np.ndarray, np.ndarray]:
    total_len = sum(scale for _src, _tgt, scale in entries)
    occupied = np.zeros((GRID_SIZE, GRID_SIZE), dtype=bool)
    occupied[GRID_SIZE - 2 : GRID_SIZE, :total_len] = True

    blocked = occupied.copy()
    blocked[GRID_SIZE - 3 : GRID_SIZE, : min(MID, total_len + 1)] = True
    return occupied, blocked


def _place_legend(grid: np.ndarray, entries: list[tuple[int, int, int]]) -> None:
    c = 0
    for src, tgt, scale in entries:
        grid[GRID_SIZE - 2, c : c + scale] = src
        grid[GRID_SIZE - 1, c : c + scale] = tgt
        c += scale


def _choose_anchor_position(
    qb: int,
    qr: int,
    raw_patch: np.ndarray,
    anchor_offset: tuple[int, int],
    blocked: np.ndarray | None = None,
) -> tuple[int, int] | None:
    q_r0, q_r1, q_c0, q_c1 = QUADS[(qb, qr)]
    non_bg = np.argwhere(raw_patch != BG)
    anchor_r0, anchor_c0 = anchor_offset

    candidates: list[tuple[int, int]] = []
    for ar in range(q_r0, q_r1 - TARGET_SEED_SIZE + 2):
        for ac in range(q_c0, q_c1 - TARGET_SEED_SIZE + 2):
            place_r0 = ar - anchor_r0
            place_c0 = ac - anchor_c0
            ok = True
            for pr, pc in non_bg.tolist():
                tr = place_r0 + int(pr)
                tc = place_c0 + int(pc)
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


def _stamp(out: np.ndarray, patch_colored: np.ndarray, anchor_pos: tuple[int, int], anchor_offset: tuple[int, int]) -> None:
    place_r0 = anchor_pos[0] - anchor_offset[0]
    place_c0 = anchor_pos[1] - anchor_offset[1]
    rr_p, cc_p = np.where(patch_colored != BG)
    for rr0, cc0 in zip(rr_p.tolist(), cc_p.tolist(), strict=False):
        rr_t = place_r0 + rr0
        cc_t = place_c0 + cc0
        if rr_t == MID or cc_t == MID:
            continue
        out[rr_t, cc_t] = int(patch_colored[rr0, cc0])


def _build_entries() -> tuple[list[tuple[int, int, int]], dict[int, int], dict[int, int]]:
    palette = random.sample(COLOR_POOL, 4)
    source_colors = palette[:2]
    target_colors = palette[2:]
    mapping_by_scale = {
        2: (source_colors[0], target_colors[0]),
        3: (source_colors[1], target_colors[1]),
    }
    entries = [(src, tgt, scale) for scale, (src, tgt) in mapping_by_scale.items()]
    random.shuffle(entries)
    color_map = {src: tgt for src, tgt, _scale in entries}
    scale_to_anchor_color = {scale: tgt for _src, tgt, scale in entries}
    return entries, color_map, scale_to_anchor_color


def _gen_attempt() -> np.ndarray | None:
    entries, color_map, scale_to_anchor_color = _build_entries()
    legend_occupied, legend_blocked = _legend_blockers(entries)

    source = _build_source_patch([src for src in color_map])
    if source is None:
        return None
    patch, anchor_r_in_patch, anchor_c_in_patch = source

    scale_plan_items = [2, 3, random.choice(LEGEND_SCALES)]
    random.shuffle(scale_plan_items)
    scale_plan = {
        (0, 1): scale_plan_items[0],
        (1, 1): scale_plan_items[1],
        (1, 0): 2,
    }
    if 3 not in scale_plan.values():
        return None

    transformed: dict[tuple[int, int], tuple[np.ndarray, np.ndarray, tuple[int, int], int]] = {}
    for key, scale in scale_plan.items():
        anchor_color = scale_to_anchor_color[scale]
        raw_patch, colored_patch, anchor_offset = _scaled_variant(
            patch,
            key[0],
            key[1],
            scale,
            color_map,
            anchor_color,
        )
        transformed[key] = (raw_patch, colored_patch, anchor_offset, anchor_color)

    anchors: dict[tuple[int, int], tuple[int, int]] = {}
    tl_r0 = random.randint(0, MID - patch.shape[0])
    tl_c0 = random.randint(0, MID - patch.shape[1])
    anchors[(0, 0)] = (tl_r0 + anchor_r_in_patch, tl_c0 + anchor_c_in_patch)

    for key, (raw_patch, _colored_patch, anchor_offset, _anchor_color) in transformed.items():
        blocked = legend_blocked if key == (1, 0) else None
        anchor = _choose_anchor_position(key[0], key[1], raw_patch, anchor_offset, blocked)
        if anchor is None:
            return None
        anchors[key] = anchor

    out = np.full((GRID_SIZE, GRID_SIZE), BG, dtype=int)
    out[MID, :] = DIV
    out[:, MID] = DIV
    _place_legend(out, entries)
    paste(out, patch, tl_r0, tl_c0)
    for key, (_raw_patch, _colored_patch, _anchor_offset, anchor_color) in transformed.items():
        paste(out, np.full((TARGET_SEED_SIZE, TARGET_SEED_SIZE), anchor_color, dtype=int), *anchors[key])
    for key, (_raw_patch, colored_patch, anchor_offset, _anchor_color) in transformed.items():
        q_r0, q_r1, q_c0, q_c1 = QUADS[key]
        place_r0 = anchors[key][0] - anchor_offset[0]
        place_c0 = anchors[key][1] - anchor_offset[1]
        rr_p, cc_p = np.where(colored_patch != BG)
        for rr0, cc0 in zip(rr_p.tolist(), cc_p.tolist(), strict=False):
            rr_t = place_r0 + rr0
            cc_t = place_c0 + cc0
            if not (q_r0 <= rr_t <= q_r1 and q_c0 <= cc_t <= q_c1):
                return None
            if legend_occupied[rr_t, cc_t]:
                return None
        _stamp(out, colored_patch, anchors[key], anchor_offset)

    inp = np.full((GRID_SIZE, GRID_SIZE), BG, dtype=int)
    inp[MID, :] = DIV
    inp[:, MID] = DIV
    _place_legend(inp, entries)
    paste(inp, patch, tl_r0, tl_c0)
    for key, scale in scale_plan.items():
        anchor_color = scale_to_anchor_color[scale]
        paste(inp, np.full((TARGET_SEED_SIZE, TARGET_SEED_SIZE), anchor_color, dtype=int), *anchors[key])

    # Anti-degeneracy: all target seeds are same size, and both coded colors must actually appear.
    seed_colors = {scale_to_anchor_color[scale_plan[key]] for key in scale_plan}
    if len(seed_colors) < 2:
        return None

    if not np.array_equal(main(inp), out):
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid b10624e5 combo input")
