# Variant of 58f5dbd5
# Dimension: D2 — 匹配条件：干扰项过滤 (match_cond)
# Change: 输入中包含干扰motif（颜色与任何方块不匹配），solver通过颜色匹配自然过滤... → 空间对齐匹配

from __future__ import annotations

import itertools
import random
from collections import defaultdict

import numpy as np

from arc_utils import (
    bbox_h,
    bbox_w,
    crop_bbox,
    find_components,
    get_bg,
    is_contiguous,
    paste,
)


def mask_from_coords(coords, shape: tuple[int, int]) -> np.ndarray:
    """Local helper: build a boolean mask with True at the given coords (in-bounds only)."""
    m = np.zeros(shape, dtype=bool)
    H, W = shape
    for r, c in coords:
        rr = int(r)
        cc = int(c)
        if 0 <= rr < H and 0 <= cc < W:
            m[rr, cc] = True
    return m


# Design analysis for the D2 variant
# ----------------------------------
# a) Concretization / behavior trace
# - The original task matched a small sprite to a 5x5 solid block by shared color.
# - In this variant, motif colors are decoupled from block colors. Matching is visible only
#   through placement:
#   * row mode: the motif region sits left/right of the block region; each aligned motif row
#     shares the block row center, and blocks/motifs are paired left-to-right inside that row.
#   * col mode: the motif region sits above/below the block region; each aligned motif column
#     shares the block column center, and blocks/motifs are paired top-to-bottom inside that
#     column.
# - Distractors live only in one extra unmatched motif row/column, so the solver must filter
#   them spatially rather than by color.
# - Example delta:
#   * original 2x3 case: block color 4 matched only motif color 4.
#   * new 2x3 col mode: the three block columns each align with one motif column; inside a
#     column, the upper/lower motifs pair to the upper/lower blocks regardless of color.
#   * distractor motifs may have the same colors as matching motifs, but they sit in the extra
#     non-aligned column and must be ignored.
#
# b) Complexity calibration
# - Keeping the original multi-block rectangular layouts preserves enough structure. Restricting
#   matching to pure row-or-column alignment would be ambiguous unless pairing within an aligned
#   group is also ordered, so the solver uses row-wise left-to-right or column-wise top-to-bottom
#   order inside each aligned band.
# - Only one extra distractor row/column is allowed. This keeps the visual rule obvious and
#   prevents many hard-to-disentangle free degrees.
#
# c) Implicit-assumption audit against the old main()
# - Old STEP 2 assumed "same color => same motif". This is invalid now because motif colors are
#   deliberately chosen from a disjoint palette.
# - Old per-color search also assumed one motif candidate per block color; repeated motif colors
#   would now collapse unrelated motifs together and lose information.
# - Carving and final packing remain valid because they depend only on the recovered motif masks,
#   not on how the match was established.
#
# d) Placement legality
# - The generator locks the aligned axis (shared row centers or shared column centers) and randomizes
#   only the separated axis, guaranteeing non-overlap and in-bounds placement.
# - Distractors are placed only in the trailing unmatched row/column so they never accidentally
#   satisfy the solver's alignment predicate.
#
# e) Visual observability
# - Matching motif colors never overlap block colors, so color matching is visibly impossible.
# - The aligned row/column bands repeat across several blocks in each sample, making the new cue
#   human-visible from a few examples.
#
# f) Cleanliness
# - There are only two coarse regions: block region and motif region. No element overlays another.
# - Extra distractors occupy a whole outer row/column of motif slots, which stays easy to read.
#
# g) Color inference
# - Colors no longer encode correspondence. Motif colors are allowed to repeat and are sampled from
#   colors not used by the blocks, providing explicit evidence that geometry, not color ids, drives
#   the task.


def _solid_5x5_block(comp: dict, grid: np.ndarray) -> bool:
    b = comp["bbox"]
    if bbox_h(b) != 5 or bbox_w(b) != 5:
        return False
    col = int(comp["color"])
    return bool((crop_bbox(grid, b) == int(col)).all())


def _extract_mask(comp: dict) -> np.ndarray:
    r0, c0, r1, c1 = map(int, comp["bbox"])
    rel_coords = [(int(r) - r0, int(c) - c0) for r, c in comp["coords"]]
    return mask_from_coords(rel_coords, (int(r1 - r0 + 1), int(c1 - c0 + 1)))


def _pair_by_alignment(
    blocks: list[dict[str, int]], motifs: list[dict[str, object]]
) -> dict[tuple[int, int], np.ndarray]:
    """
    Recover the motif matched to each block using spatial alignment only.

    New main() assumptions:
    - Every block belongs to exactly one rectangular block row and column layout.
    - Generator guarantees either:
      * row mode: each block row has exactly as many motifs on the same center row as blocks in it.
      * col mode: each block column has exactly as many motifs on the same center column as blocks in it.
    - Distractors never share those matched center rows/cols.
    """
    block_rows: dict[int, list[dict[str, int]]] = defaultdict(list)
    block_cols: dict[int, list[dict[str, int]]] = defaultdict(list)
    motif_rows: dict[int, list[dict[str, object]]] = defaultdict(list)
    motif_cols: dict[int, list[dict[str, object]]] = defaultdict(list)

    for b in blocks:
        block_rows[int(b["center_r"])].append(b)
        block_cols[int(b["center_c"])].append(b)
    for m in motifs:
        motif_rows[int(m["center_r"])].append(m)
        motif_cols[int(m["center_c"])].append(m)

    row_ok = all(
        len(motif_rows.get(int(rc), [])) == len(row_blocks)
        for rc, row_blocks in block_rows.items()
    )
    col_ok = all(
        len(motif_cols.get(int(cc), [])) == len(col_blocks)
        for cc, col_blocks in block_cols.items()
    )

    if row_ok and not col_ok:
        pairing: dict[tuple[int, int], np.ndarray] = {}
        for rc, row_blocks in block_rows.items():
            bs = sorted(row_blocks, key=lambda b: int(b["c0"]))
            ms = sorted(motif_rows[int(rc)], key=lambda m: int(m["c0"]))
            for b, m in zip(bs, ms):
                pairing[(int(b["r0"]), int(b["c0"]))] = np.array(m["mask"], dtype=bool)
        return pairing

    if col_ok and not row_ok:
        pairing = {}
        for cc, col_blocks in block_cols.items():
            bs = sorted(col_blocks, key=lambda b: int(b["r0"]))
            ms = sorted(motif_cols[int(cc)], key=lambda m: int(m["r0"]))
            for b, m in zip(bs, ms):
                pairing[(int(b["r0"]), int(b["c0"]))] = np.array(m["mask"], dtype=bool)
        return pairing

    # The generator avoids ambiguous dual-mode samples. This assert catches mismatched inputs.
    raise AssertionError("could not infer a unique row/column alignment mode")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Carve spatially aligned motifs into solid blocks and pack them into a compact grid.

    Input structure:
    - bg = most common color.
    - Several solid 5x5 monochrome blocks arranged on a rectangular block lattice.
    - A separate motif region contains 3x3 monochrome sprites whose colors do not identify targets;
      only shared row alignment or shared column alignment with the block lattice matters.

    Rule:
    - Match motifs to blocks by spatial alignment: motifs in the same row band pair to that block row
      left-to-right, or motifs in the same column band pair to that block column top-to-bottom.
    - Center each matched motif within its 5x5 block and recolor those motif cells to bg.
    - Pack all carved blocks into an output grid preserving the original block row/column order, with
      1-cell gaps between tiles and a 1-cell outer border.

    Key insight: The motif colors are a decoy; the real correspondence is conveyed by row/column
    alignment between the separated motif lattice and the block lattice.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    blocks: list[dict[str, int]] = []
    motifs: list[dict[str, object]] = []
    comps = find_components(g, bg=bg, connectivity=8, monochromatic=True, return_format="dict")
    for comp in comps:
        r0, c0, r1, c1 = map(int, comp["bbox"])
        if _solid_5x5_block(comp, g):
            blocks.append(
                {
                    "col": int(comp["color"]),
                    "r0": int(r0),
                    "c0": int(c0),
                    "center_r": int(r0 + 2),
                    "center_c": int(c0 + 2),
                }
            )
        elif bbox_h(comp["bbox"]) <= 3 and bbox_w(comp["bbox"]) <= 3:
            motifs.append(
                {
                    "r0": int(r0),
                    "c0": int(c0),
                    "center_r": int((r0 + r1) // 2),
                    "center_c": int((c0 + c1) // 2),
                    "mask": _extract_mask(comp),
                }
            )

    assert blocks
    pairing = _pair_by_alignment(blocks, motifs)

    carved: dict[tuple[int, int], np.ndarray] = {}
    for b in blocks:
        col = int(b["col"])
        tile = np.full((5, 5), int(col), dtype=int)
        mm = pairing[(int(b["r0"]), int(b["c0"]))]
        mh, mw = map(int, mm.shape)
        off_r = (5 - mh) // 2
        off_c = (5 - mw) // 2
        for rr, cc in np.argwhere(mm):
            tr = int(off_r + int(rr))
            tc = int(off_c + int(cc))
            if 0 <= tr < 5 and 0 <= tc < 5:
                tile[tr, tc] = int(bg)
        carved[(int(b["r0"]), int(b["c0"]))] = tile

    rs = sorted({int(b["r0"]) for b in blocks})
    cs = sorted({int(b["c0"]) for b in blocks})
    r_index = {int(r): int(i) for i, r in enumerate(rs)}
    c_index = {int(c): int(j) for j, c in enumerate(cs)}

    out_h = int(1 + len(rs) * 5 + (len(rs) - 1) + 1)
    out_w = int(1 + len(cs) * 5 + (len(cs) - 1) + 1)
    out = np.full((out_h, out_w), int(bg), dtype=int)

    step = 6
    for b in blocks:
        i = int(r_index[int(b["r0"])])
        j = int(c_index[int(b["c0"])])
        rr = int(1 + i * step)
        cc = int(1 + j * step)
        out[rr : rr + 5, cc : cc + 5] = carved[(int(b["r0"]), int(b["c0"]))]

    return out


# --------------- generator (D2 alignment variant) ---------------

CANVAS_HEIGHT_RANGE = (18, 22)
CANVAS_WIDTH_RANGE = (18, 24)
COLOR_COUNT = 10

DST_BLOCK_SIZE = 5
DST_BLOCK_STEP = 6
DST_LAYOUTS = {
    3: [(1, 3), (3, 1)],
    4: [(2, 2)],
    6: [(2, 3), (3, 2)],
}

MOTIF_SIZE = 3
MOTIF_MATCH_STEP = 6
MOTIF_FREE_STEP = 4
MOTIF_AREA_RANGE = (4, 8)

OUTER_MARGIN = 1
REGION_GAP = 1
OUTPUT_BORDER = 1
MOTIF_OFFSET_IN_BLOCK = 1

GENERATION_RETRIES = 400


def _spanned_size(count: int, item_size: int, step: int) -> int:
    """Return the span for count fixed-size items placed at a constant step."""
    return (count - 1) * step + item_size


def _make_masks_by_area() -> dict[int, list[np.ndarray]]:
    """Precompute valid 3x3 motif masks keyed by filled-cell area."""
    min_area, max_area = MOTIF_AREA_RANGE
    masks: dict[int, list[np.ndarray]] = {
        area: [] for area in range(min_area, max_area + 1)
    }
    for bits in itertools.product([False, True], repeat=MOTIF_SIZE * MOTIF_SIZE):
        m = np.array(bits, dtype=bool).reshape(MOTIF_SIZE, MOTIF_SIZE)
        area = int(m.sum())
        if area not in masks:
            continue
        if not (m[0].any() and m[-1].any() and m[:, 0].any() and m[:, -1].any()):
            continue
        if is_contiguous(m.astype(int), background=0, connectivity=8):
            masks[area].append(m)
    return masks


MASKS_BY_AREA = _make_masks_by_area()


def _random_mask() -> np.ndarray:
    min_area, max_area = MOTIF_AREA_RANGE
    area = random.randint(min_area, max_area)
    return random.choice(MASKS_BY_AREA[area]).copy()


def _choose_block_colors(bg: int, count: int) -> list[int]:
    palette = [c for c in range(COLOR_COUNT) if c != bg]
    random.shuffle(palette)
    return [int(c) for c in palette[:count]]


def _choose_motif_colors(bg: int, block_colors: list[int], total: int) -> list[int]:
    motif_palette = [c for c in range(COLOR_COUNT) if c != bg and c not in block_colors]
    assert motif_palette
    random.shuffle(motif_palette)
    k = random.randint(1, len(motif_palette))
    pool = motif_palette[:k]
    return [int(random.choice(pool)) for _ in range(total)]


def _render_expected(
    bg: int, R: int, C: int, block_colors: list[int], motif_masks: list[np.ndarray]
) -> np.ndarray:
    out_h = 2 * OUTPUT_BORDER + _spanned_size(R, DST_BLOCK_SIZE, DST_BLOCK_STEP)
    out_w = 2 * OUTPUT_BORDER + _spanned_size(C, DST_BLOCK_SIZE, DST_BLOCK_STEP)
    expected = np.full((out_h, out_w), bg, dtype=int)

    motif_slice = slice(MOTIF_OFFSET_IN_BLOCK, MOTIF_OFFSET_IN_BLOCK + MOTIF_SIZE)
    for i in range(R):
        for j in range(C):
            idx = i * C + j
            color = int(block_colors[idx])
            tile = np.full((DST_BLOCK_SIZE, DST_BLOCK_SIZE), color, dtype=int)
            tile[motif_slice, motif_slice][motif_masks[idx]] = bg
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
    for i in range(R):
        for j in range(C):
            color = int(block_colors[i * C + j])
            patch = np.full((DST_BLOCK_SIZE, DST_BLOCK_SIZE), color, dtype=int)
            paste(grid, patch, dst_r + i * DST_BLOCK_STEP, dst_c + j * DST_BLOCK_STEP)


def _paint_motif(grid: np.ndarray, r0: int, c0: int, color: int, mask: np.ndarray) -> None:
    patch = np.full((MOTIF_SIZE, MOTIF_SIZE), int(color), dtype=int)
    paste(grid, patch, r0, c0, mask=mask)


def _row_mode_sizes(R: int, C: int, extra_rows: int) -> tuple[int, int]:
    rows = R + extra_rows
    h = _spanned_size(rows, MOTIF_SIZE, MOTIF_MATCH_STEP)
    w = _spanned_size(C, MOTIF_SIZE, MOTIF_FREE_STEP)
    return int(h), int(w)


def _col_mode_sizes(R: int, C: int, extra_cols: int) -> tuple[int, int]:
    cols = C + extra_cols
    h = _spanned_size(R, MOTIF_SIZE, MOTIF_FREE_STEP)
    w = _spanned_size(cols, MOTIF_SIZE, MOTIF_MATCH_STEP)
    return int(h), int(w)


def _feasible_modes(H: int, W: int) -> list[dict[str, int | str]]:
    options: list[dict[str, int | str]] = []
    for count, layouts in DST_LAYOUTS.items():
        for R, C in layouts:
            dst_h = _spanned_size(R, DST_BLOCK_SIZE, DST_BLOCK_STEP)
            dst_w = _spanned_size(C, DST_BLOCK_SIZE, DST_BLOCK_STEP)

            for extra_rows in (0, 1):
                motif_h, motif_w = _row_mode_sizes(R, C, extra_rows)
                if (
                    max(dst_h, motif_h + 1) + 2 * OUTER_MARGIN <= H
                    and dst_w + motif_w + 2 * OUTER_MARGIN + REGION_GAP <= W
                ):
                    options.append(
                        {
                            "count": int(count),
                            "R": int(R),
                            "C": int(C),
                            "mode": "row",
                            "extra": int(extra_rows),
                        }
                    )

            for extra_cols in (0, 1):
                motif_h, motif_w = _col_mode_sizes(R, C, extra_cols)
                if (
                    dst_h + motif_h + 2 * OUTER_MARGIN + REGION_GAP <= H
                    and max(dst_w, motif_w + 1) + 2 * OUTER_MARGIN <= W
                ):
                    options.append(
                        {
                            "count": int(count),
                            "R": int(R),
                            "C": int(C),
                            "mode": "col",
                            "extra": int(extra_cols),
                        }
                    )
    return options


def _choose_config(H: int, W: int) -> dict[str, int | str] | None:
    configs = _feasible_modes(H, W)
    if not configs:
        return None

    counts = sorted({int(cfg["count"]) for cfg in configs})
    count = random.choice(counts)
    subset = [cfg for cfg in configs if int(cfg["count"]) == int(count)]

    with_distractors = [cfg for cfg in subset if int(cfg["extra"]) == 1]
    if with_distractors and random.random() < 0.65:
        subset = with_distractors
    return random.choice(subset)


def _place_row_mode(
    H: int, W: int, R: int, C: int, extra_rows: int
) -> tuple[int, int, int, int]:
    dst_h = _spanned_size(R, DST_BLOCK_SIZE, DST_BLOCK_STEP)
    dst_w = _spanned_size(C, DST_BLOCK_SIZE, DST_BLOCK_STEP)
    motif_h, motif_w = _row_mode_sizes(R, C, extra_rows)

    max_dst_r = min(
        H - OUTER_MARGIN - dst_h,
        H - OUTER_MARGIN - motif_h - 1,
    )
    assert max_dst_r >= OUTER_MARGIN
    dst_r = random.randint(OUTER_MARGIN, max_dst_r)
    motif_r = dst_r + 1

    total_w = dst_w + motif_w + REGION_GAP
    start_c = random.randint(OUTER_MARGIN, W - OUTER_MARGIN - total_w)
    if random.choice([True, False]):
        motif_c = start_c
        dst_c = start_c + motif_w + REGION_GAP
    else:
        dst_c = start_c
        motif_c = start_c + dst_w + REGION_GAP
    return int(dst_r), int(dst_c), int(motif_r), int(motif_c)


def _place_col_mode(
    H: int, W: int, R: int, C: int, extra_cols: int
) -> tuple[int, int, int, int]:
    dst_h = _spanned_size(R, DST_BLOCK_SIZE, DST_BLOCK_STEP)
    dst_w = _spanned_size(C, DST_BLOCK_SIZE, DST_BLOCK_STEP)
    motif_h, motif_w = _col_mode_sizes(R, C, extra_cols)

    max_dst_c = min(
        W - OUTER_MARGIN - dst_w,
        W - OUTER_MARGIN - motif_w - 1,
    )
    assert max_dst_c >= OUTER_MARGIN
    dst_c = random.randint(OUTER_MARGIN, max_dst_c)
    motif_c = dst_c + 1

    total_h = dst_h + motif_h + REGION_GAP
    start_r = random.randint(OUTER_MARGIN, H - OUTER_MARGIN - total_h)
    if random.choice([True, False]):
        motif_r = start_r
        dst_r = start_r + motif_h + REGION_GAP
    else:
        dst_r = start_r
        motif_r = start_r + dst_h + REGION_GAP
    return int(dst_r), int(dst_c), int(motif_r), int(motif_c)


def _paint_row_mode_motifs(
    grid: np.ndarray,
    motif_r: int,
    motif_c: int,
    R: int,
    C: int,
    extra_rows: int,
    match_masks: list[np.ndarray],
    motif_colors: list[int],
    distractor_masks: list[np.ndarray],
) -> None:
    idx = 0
    color_idx = 0
    for i in range(R):
        for j in range(C):
            _paint_motif(
                grid,
                motif_r + i * MOTIF_MATCH_STEP,
                motif_c + j * MOTIF_FREE_STEP,
                motif_colors[color_idx],
                match_masks[idx],
            )
            idx += 1
            color_idx += 1

    if extra_rows:
        for j in range(C):
            _paint_motif(
                grid,
                motif_r + R * MOTIF_MATCH_STEP,
                motif_c + j * MOTIF_FREE_STEP,
                motif_colors[color_idx],
                distractor_masks[j],
            )
            color_idx += 1


def _paint_col_mode_motifs(
    grid: np.ndarray,
    motif_r: int,
    motif_c: int,
    R: int,
    C: int,
    extra_cols: int,
    match_masks: list[np.ndarray],
    motif_colors: list[int],
    distractor_masks: list[np.ndarray],
) -> None:
    idx = 0
    color_idx = 0
    for i in range(R):
        for j in range(C):
            _paint_motif(
                grid,
                motif_r + i * MOTIF_FREE_STEP,
                motif_c + j * MOTIF_MATCH_STEP,
                motif_colors[color_idx],
                match_masks[idx],
            )
            idx += 1
            color_idx += 1

    if extra_cols:
        for i in range(R):
            _paint_motif(
                grid,
                motif_r + i * MOTIF_FREE_STEP,
                motif_c + C * MOTIF_MATCH_STEP,
                motif_colors[color_idx],
                distractor_masks[i],
            )
            color_idx += 1


def _render_input(
    H: int,
    W: int,
    bg: int,
    config: dict[str, int | str],
    block_colors: list[int],
    match_masks: list[np.ndarray],
    motif_colors: list[int],
    distractor_masks: list[np.ndarray],
) -> tuple[np.ndarray, dict[str, int | str]]:
    R = int(config["R"])
    C = int(config["C"])
    mode = str(config["mode"])
    extra = int(config["extra"])

    if mode == "row":
        dst_r, dst_c, motif_r, motif_c = _place_row_mode(H, W, R, C, extra)
    else:
        dst_r, dst_c, motif_r, motif_c = _place_col_mode(H, W, R, C, extra)

    inp = np.full((H, W), bg, dtype=int)
    _paint_destination_blocks(inp, dst_r, dst_c, R, C, block_colors)
    if mode == "row":
        _paint_row_mode_motifs(
            inp,
            motif_r,
            motif_c,
            R,
            C,
            extra,
            match_masks,
            motif_colors,
            distractor_masks,
        )
    else:
        _paint_col_mode_motifs(
            inp,
            motif_r,
            motif_c,
            R,
            C,
            extra,
            match_masks,
            motif_colors,
            distractor_masks,
        )
    meta = {
        "mode": mode,
        "extra": int(extra),
        "R": int(R),
        "C": int(C),
        "dst_r": int(dst_r),
        "dst_c": int(dst_c),
        "motif_r": int(motif_r),
        "motif_c": int(motif_c),
    }
    return inp, meta


def _gen_attempt(return_meta: bool = False):
    H = random.randint(*CANVAS_HEIGHT_RANGE)
    W = random.randint(*CANVAS_WIDTH_RANGE)
    bg = random.randrange(COLOR_COUNT)

    config = _choose_config(H, W)
    if config is None:
        return None
    R = int(config["R"])
    C = int(config["C"])
    count = int(config["count"])
    extra = int(config["extra"])

    block_colors = _choose_block_colors(bg, count)
    match_masks = [_random_mask() for _ in range(count)]
    distractor_count = C if str(config["mode"]) == "row" else R
    distractor_masks = [_random_mask() for _ in range(distractor_count)] if extra else []
    motif_total = count + (distractor_count if extra else 0)
    motif_colors = _choose_motif_colors(bg, block_colors, motif_total)

    expected = _render_expected(bg, R, C, block_colors, match_masks)
    inp, meta = _render_input(
        H,
        W,
        bg,
        config,
        block_colors,
        match_masks,
        motif_colors,
        distractor_masks,
    )

    if not np.array_equal(main(inp), expected):
        return None

    if not return_meta:
        return inp

    meta.update(
        {
            "bg": int(bg),
            "block_colors": tuple(int(c) for c in block_colors),
            "motif_colors": tuple(int(c) for c in motif_colors),
            "motif_palette_overlap": int(
                len(set(block_colors) & set(motif_colors))
            ),
            "distractor_count": int(distractor_count if extra else 0),
        }
    )
    return inp, meta


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 58f5dbd5 D2 input")


def sample_variant_metrics(n: int = 32) -> list[dict[str, int | str | tuple[int, ...]]]:
    """Small helper for auditing that the generator really exercises alignment-based matching."""
    out: list[dict[str, int | str | tuple[int, ...]]] = []
    tries = 0
    while len(out) < n and tries < n * 20:
        tries += 1
        sample = _gen_attempt(return_meta=True)
        if sample is None:
            continue
        _inp, meta = sample
        out.append(meta)
    return out
