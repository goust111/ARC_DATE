# Variant of edb79dae
# Dimension: D4 — 目标位置：框架内的匹配点 (dst)
# Change: 灰色框架内部的单色像素点，每个点的颜色指示应匹配哪个外部精灵... → 目标像素颜色不直接对应精灵颜色

import itertools
import random

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, paste, random_free_location


# Design analysis for the D4 variant
# ----------------------------------
# a) Concrete behavior:
#    - Original: a solid NxN color block inside the frame directly names the outside sprite color.
#    - Variant: the frame now contains "query colors" dst. A separate lookup legend at the top-left
#      decodes dst -> src, where src is the real outside sprite color. Then the existing overlapping
#      legend still maps src -> tgt for the final recolor.
#    - Small trace:
#        frame block color = 2
#        top-left lookup tile says 2 over 7  => dst 2 means use sprite color 7
#        external 3x4 legend says 7 over 4  => sprite 7 is finally recolored to 4
#        output therefore stamps shape(7) painted in 4, not shape(2).
#      The old solver would incorrectly use shape(2) directly.
# b) Complexity calibration:
#    - The new lookup is a left-to-right table of 2x2 tiles, each entry repeated by actual frame usage.
#    - dst colors are a derangement of the sprite-color set, so every query color is visibly "wrong"
#      if interpreted directly. This prevents degeneration back to the original task.
#    - Sprite masks are generated to be pairwise distinct, so choosing the wrong sprite is visually obvious.
# c) Old assumptions that break:
#    - Placeholder color == sprite color: no longer true.
#    - Placeholder color helps identify the src side of the external recolor legend: no longer safe.
#      We therefore infer src -> tgt from the fixed 7-vs-5 area asymmetry of the 3x4 legend.
#    - Sprite extraction must exclude the new top-left lookup legend pieces, otherwise those 1x2 bars
#      could be mistaken for sprites of the same color.
# d) Placement legality:
#    - The lookup legend occupies a reserved top-left rectangle with an extra blank row and blank columns
#      on its right, so it stays isolated from the frame and from other outside patches.
#    - The frame is sampled with the original size constraints but may not overlap that reserved lookup zone.
#    - All remaining outside patches still use the original non-overlap placement utility.
# e) Visual observability:
#    - The new rule is inferable from 2-4 examples because frame colors no longer match the visible sprites;
#      a compact legend explicitly bridges them. The legend is structural, not numeric-color-id based.
# f) Visual cleanliness:
#    - The lookup legend sits alone in the top-left, while sprite templates and recolor legends remain
#      elsewhere outside the frame. This avoids stacking several symbolic systems into one region.
# g) Color-change inferability:
#    - There are only two lookup stages: dst -> src from the top-left tiles, then src -> tgt from the
#      original overlapping legend. Both are repeated across multiple placements and colors.
# Main() assumptions guaranteed by generate_input():
#    - Each lookup tile is exactly 2 columns wide, 2 rows tall, and the whole lookup cluster is the
#      leftmost non-bg content touching row 0.
#    - dst -> src is a bijection, with zero fixed points.
#    - Every src has exactly one outside sprite bbox matching the common placeholder size.
#    - Every src also has exactly one overlapping recolor legend block.


def _find_frame(g: np.ndarray, bg: int) -> tuple[int, tuple[int, int, int, int]]:
    """
    Find a rectangular border component:
      - all bbox-border cells are this color,
      - bbox-interior is not entirely this color.
    Return (frame_color, (r0,c0,r1,c1)).
    """
    candidates = []
    colors = [int(c) for c in np.unique(g) if int(c) != bg]
    for col in colors:
        for pts in components_mask(g == col, connectivity=4):
            r0, c0, r1, c1 = bbox_coords(pts)
            if r1 - r0 < 2 or c1 - c0 < 2:
                continue
            sub = g[r0 : r1 + 1, c0 : c1 + 1]
            border = np.zeros(sub.shape, dtype=bool)
            border[0, :] = True
            border[-1, :] = True
            border[:, 0] = True
            border[:, -1] = True
            if not np.all(sub[border] == col):
                continue
            if np.all(sub[~border] == col):
                continue
            candidates.append((col, (r0, c0, r1, c1)))
    assert len(candidates) == 1, f"expected exactly one rectangular frame component, found {len(candidates)}"
    frame_col, bbox = candidates[0]
    return int(frame_col), bbox


def _bbox_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (ar1 < br0 or br1 < ar0 or ac1 < bc0 or bc1 < ac0)


def _bbox_inside(inner: tuple[int, int, int, int], outer: tuple[int, int, int, int]) -> bool:
    ir0, ic0, ir1, ic1 = inner
    or0, oc0, or1, oc1 = outer
    return or0 <= ir0 and or1 >= ir1 and oc0 <= ic0 and oc1 >= ic1


def _lookup_legend(g: np.ndarray, bg: int) -> tuple[dict[int, int], tuple[int, int, int, int]]:
    """
    Read the leftmost top-row legend cluster as 2x2 lookup tiles:
      [dst dst]
      [src src]

    The cluster is the first top-row non-bg run group; a gap > 2 ends the cluster,
    mirroring the reference legend_lookup style where the legend occupies the top-left.
    """
    h, w = g.shape
    top_pos = [int(c) for c in np.flatnonzero(g[0, :] != bg)]
    assert top_pos, "lookup legend must touch the top row"
    top_pos.sort()

    cols = [top_pos[0]]
    for c in top_pos[1:]:
        if c - cols[-1] <= 2:
            cols.append(c)
        else:
            break

    runs: list[list[int]] = []
    cur = [cols[0]]
    for c in cols[1:]:
        if c == cur[-1] + 1:
            cur.append(c)
        else:
            runs.append(cur)
            cur = [c]
    runs.append(cur)

    assert h >= 2, "lookup legend needs two rows"
    query_to_src: dict[int, int] = {}
    for run in runs:
        assert len(run) == 2 and run[1] == run[0] + 1, f"lookup run must be width 2, got {run}"
        c0 = run[0]
        tile = g[0:2, c0 : c0 + 2]
        assert tile.shape == (2, 2), f"bad lookup tile at col {c0}"
        top_vals = set(map(int, tile[0]))
        bot_vals = set(map(int, tile[1]))
        assert len(top_vals) == 1 and len(bot_vals) == 1, f"lookup tile at {c0} must have solid rows"
        dst = int(tile[0, 0])
        src = int(tile[1, 0])
        assert dst != bg and src != bg and dst != src, f"invalid lookup tile colors at {c0}"
        assert dst not in query_to_src, f"duplicate lookup key color {dst}"
        query_to_src[dst] = src
        if h > 2:
            assert np.all(g[2, c0 : c0 + 2] == bg), "lookup tiles must be isolated by a blank row"

    return query_to_src, (0, min(cols), 1, max(cols))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stamp recolored sprites into a framed region after decoding placeholder colors through a lookup legend.

    Input structure:
    - bg background, a rectangular single-color frame, and solid placeholder blocks inside the frame.
    - Outside the frame: source-color sprite templates, the original overlapping recolor legends, and a
      separate top-left lookup legend whose tiles say which source color each frame color refers to.

    Rule:
    - Find the frame and its interior placeholder blocks.
    - Decode each placeholder color dst through the lookup legend to get the real sprite color src.
    - Use src to locate the outside sprite shape, then use the original overlapping legend to recolor
      that sprite from src to its final output color tgt before stamping it into the frame.

    Key insight: The frame colors are only symbolic queries. They do not name the sprite directly; the
    top-left lookup legend must be read first, and only then can the old src-to-tgt recolor legend apply.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)

    frame_col, (fr0, fc0, fr1, fc1) = _find_frame(g, bg)
    query_to_src, lookup_bbox = _lookup_legend(g, bg)

    crop = g[fr0 : fr1 + 1, fc0 : fc1 + 1]
    h, w = crop.shape
    assert h >= 3 and w >= 3, f"frame too small for interior: {(h, w)}"

    inner_mask = np.zeros((h, w), dtype=bool)
    inner_mask[1:-1, 1:-1] = True
    placeholder_colors = set(map(int, np.unique(crop[inner_mask]))) - {bg, frame_col}
    assert placeholder_colors, "no placeholder colors found inside frame"
    assert set(query_to_src) == placeholder_colors, (
        f"lookup keys {sorted(query_to_src)} do not match frame colors {sorted(placeholder_colors)}"
    )

    placeholders: list[tuple[int, tuple[int, int, int, int]]] = []
    for col in sorted(placeholder_colors):
        for pts in components_mask((crop == col) & inner_mask, connectivity=4):
            r0, c0, r1, c1 = bbox_coords(pts)
            placeholders.append((col, (r0, c0, r1, c1)))

    phw: dict[int, tuple[int, int]] = {}
    for col in placeholder_colors:
        dims = [(b[2] - b[0] + 1, b[3] - b[1] + 1) for c, b in placeholders if c == col]
        assert dims, f"no placeholder component found for color {col}"
        unique_dims = sorted(set(dims))
        assert len(unique_dims) == 1, f"ambiguous placeholder sizes for color {col}: {unique_dims}"
        phw[col] = unique_dims[0]

    src_dims: dict[int, tuple[int, int]] = {}
    for dst, src in query_to_src.items():
        if src in src_dims:
            assert src_dims[src] == phw[dst], f"inconsistent placeholder sizes for mapped src color {src}"
        else:
            src_dims[src] = phw[dst]

    outside = np.ones(g.shape, dtype=bool)
    outside[fr0 : fr1 + 1, fc0 : fc1 + 1] = False

    comps = []
    colors = [int(c) for c in np.unique(g) if int(c) not in (bg, frame_col)]
    cid = 0
    for col in colors:
        for pts in components_mask((g == col) & outside, connectivity=4):
            r0, c0, r1, c1 = bbox_coords(pts)
            pts_tuples = [(int(r), int(c)) for r, c in pts]
            comps.append(
                {
                    "id": cid,
                    "col": int(col),
                    "pts": pts_tuples,
                    "bbox": (r0, c0, r1, c1),
                    "area": len(pts_tuples),
                }
            )
            cid += 1

    color_map: dict[int, int] = {}
    color_legend_ids: set[int] = set()
    mapped_sources = set(query_to_src.values())

    for i in range(len(comps)):
        for j in range(i + 1, len(comps)):
            ci = comps[i]
            cj = comps[j]
            if ci["col"] == cj["col"]:
                continue
            if not _bbox_overlap(ci["bbox"], cj["bbox"]):
                continue
            r0 = min(ci["bbox"][0], cj["bbox"][0])
            c0 = min(ci["bbox"][1], cj["bbox"][1])
            r1 = max(ci["bbox"][2], cj["bbox"][2])
            c1 = max(ci["bbox"][3], cj["bbox"][3])
            if (r1 - r0 + 1) > 8 or (c1 - c0 + 1) > 8:
                continue
            sub = g[r0 : r1 + 1, c0 : c1 + 1]
            sub_colors = set(map(int, np.unique(sub))) - {frame_col}
            if bg in sub_colors or len(sub_colors) != 2:
                continue

            # The original 3x4 legend has asymmetric component areas: src uses 7 cells, tgt uses 5 cells.
            assert ci["area"] != cj["area"], f"ambiguous equal-area legend pair: {ci['col']} and {cj['col']}"
            if ci["area"] > cj["area"]:
                src, tgt = ci["col"], cj["col"]
            else:
                src, tgt = cj["col"], ci["col"]

            if src not in mapped_sources:
                continue
            assert src not in color_map, f"duplicate legend mapping for sprite color {src}"
            color_map[src] = int(tgt)
            color_legend_ids.add(ci["id"])
            color_legend_ids.add(cj["id"])

    assert set(color_map) == mapped_sources, (
        f"recolor legends {sorted(color_map)} do not match sprite colors {sorted(mapped_sources)}"
    )

    sprite_coords: dict[int, tuple[tuple[tuple[int, int], ...], int, int]] = {}
    for src_col in sorted(mapped_sources):
        all_candidates = [
            c
            for c in comps
            if c["col"] == src_col and c["id"] not in color_legend_ids and not _bbox_inside(c["bbox"], lookup_bbox)
        ]
        assert all_candidates, f"no sprite candidate found for color {src_col}"
        want_h, want_w = src_dims[src_col]
        sized = [
            c
            for c in all_candidates
            if (c["bbox"][2] - c["bbox"][0] + 1) == want_h and (c["bbox"][3] - c["bbox"][1] + 1) == want_w
        ]
        assert len(sized) == 1, f"expected one size-matched sprite for color {src_col}, found {len(sized)}"
        base = sized[0]
        sr0, sc0, sr1, sc1 = base["bbox"]

        pts_set = set(base["pts"])
        for c in all_candidates:
            if c["id"] == base["id"]:
                continue
            r0, c0, r1, c1 = c["bbox"]
            assert sr0 <= r0 and sr1 >= r1 and sc0 <= c0 and sc1 >= c1, (
                f"extra sprite component for color {src_col} is outside the base bbox"
            )
            pts_set.update(c["pts"])

        coords = tuple(sorted((r - sr0, c - sc0) for r, c in pts_set))
        sprite_coords[src_col] = (coords, sr1 - sr0 + 1, sc1 - sc0 + 1)

    out = crop.copy()
    out[1:-1, 1:-1] = bg

    for dst_col, (pr0, pc0, pr1, pc1) in placeholders:
        src_col = query_to_src[dst_col]
        tgt_col = int(color_map[src_col])
        coords, sh, sw = sprite_coords[src_col]
        assert (pr1 - pr0 + 1, pc1 - pc0 + 1) == (sh, sw), (
            f"placeholder size does not match sprite for query color {dst_col}"
        )
        assert pr0 + sh <= h and pc0 + sw <= w, f"sprite for query color {dst_col} does not fit at placeholder"
        for dr, dc in coords:
            out[pr0 + dr, pc0 + dc] = tgt_col

    return out


# --------------- generator (o5x) ---------------

# === CFG: tunables ===
FRAME_COL = 5
SPRITE_N_RANGE = (3, 6)
GRID_ROWS_RANGE = (2, 4)
GRID_COLS_RANGE = (3, 5)
TOTAL_BLOCKS_RANGE = (6, 10)
MAX_EMPTY_SLOTS = 2
NUM_COLORS_RANGE = (2, 5)
SPRITE_FILL_MIN = 0.5
SPRITE_FILL_MAX = 0.9
MIN_FRAME_H = 13
MIN_FRAME_W = 18
MAX_FRAME_DIM = 21
FRAME_MARGIN_SLACK = 4
CANVAS_H_RANGE = (22, 31)
CANVAS_W_RANGE = (23, 31)
FRAME_POS_JITTER_PROB = 0.25
PATCH_PADDING = 1
GEN_RETRIES = 300
LOOKUP_TILE_W = 2
LOOKUP_GAP = 1
LOOKUP_H = 2
LOOKUP_BUFFER_RIGHT = 3
LOOKUP_BUFFER_BOTTOM = 1


def _legend(src: int, tgt: int) -> np.ndarray:
    """Build a 3x4 legend block mapping src color to tgt color."""
    return np.array(
        [[src, src, tgt, tgt], [src, src, src, tgt], [src, src, tgt, tgt]],
        dtype=int,
    )


def _lookup_patch(dst_cols: list[int], src_cols: list[int], bg: int) -> np.ndarray:
    """Build a top-left lookup legend of 2x2 tiles stacked as [dst][src]."""
    w = len(dst_cols) * (LOOKUP_TILE_W + LOOKUP_GAP) - LOOKUP_GAP
    patch = np.full((LOOKUP_H, w), bg, dtype=int)
    for i, (dst, src) in enumerate(zip(dst_cols, src_cols)):
        c0 = i * (LOOKUP_TILE_W + LOOKUP_GAP)
        patch[0, c0 : c0 + LOOKUP_TILE_W] = int(dst)
        patch[1, c0 : c0 + LOOKUP_TILE_W] = int(src)
    return patch


def _sprite_mask(n: int) -> np.ndarray:
    """Generate a random NxN binary mask with fill ratio in [SPRITE_FILL_MIN, SPRITE_FILL_MAX]."""
    lo = max((n * n + 1) // 2, int(np.ceil(SPRITE_FILL_MIN * n * n)))
    hi = min(int(np.floor(SPRITE_FILL_MAX * n * n)), n * n - 1)
    target = random.randint(lo, hi)

    mask = np.zeros((n, n), dtype=bool)
    c = random.randrange(n)
    for r in range(n):
        c = max(0, min(n - 1, c + random.choice([-1, 0, 1])))
        mask[r, c] = True
    mask[random.randrange(n), :] = True

    if random.random() < 0.5:
        mask = mask.T.copy()
    if random.random() < 0.5:
        mask = np.flipud(mask)
    if random.random() < 0.5:
        mask = np.fliplr(mask)

    while int(mask.sum()) < target:
        dilated = np.zeros_like(mask)
        dilated[1:, :] |= mask[:-1, :]
        dilated[:-1, :] |= mask[1:, :]
        dilated[:, 1:] |= mask[:, :-1]
        dilated[:, :-1] |= mask[:, 1:]
        frontier = list(zip(*np.where(dilated & ~mask)))
        r, c = random.choice(frontier)
        mask[r, c] = True
    return mask


def _unique_sprite_masks(src_cols: list[int], n: int) -> dict[int, np.ndarray]:
    """Generate pairwise distinct sprite masks so wrong lookup choices remain visually detectable."""
    masks: dict[int, np.ndarray] = {}
    seen: set[tuple[tuple[int, ...], ...]] = set()
    for src in src_cols:
        for _ in range(200):
            mask = _sprite_mask(n)
            key = tuple(tuple(int(v) for v in row) for row in mask.astype(int))
            if key not in seen:
                seen.add(key)
                masks[src] = mask
                break
        else:
            raise RuntimeError("failed to generate distinct sprite masks")
    return masks


def _colored_sprite(mask: np.ndarray, color: int, bg: int) -> np.ndarray:
    out = np.full(mask.shape, bg, dtype=int)
    out[mask] = int(color)
    return out


def _content_extent(n: int, grid_rows: int, grid_cols: int) -> tuple[int, int]:
    return (grid_rows * n + grid_rows - 1, grid_cols * n + grid_cols - 1)


def _choose_structure() -> tuple[int, int, int, int, int] | None:
    total = random.randint(*TOTAL_BLOCKS_RANGE)
    options = []
    for n in range(*SPRITE_N_RANGE):
        for rows in range(*GRID_ROWS_RANGE):
            for cols in range(*GRID_COLS_RANGE):
                ch, cw = _content_extent(n, rows, cols)
                base_fh = ch + 4
                base_fw = cw + 4
                slots = rows * cols
                empty = slots - total
                if base_fh <= MAX_FRAME_DIM and base_fw <= MAX_FRAME_DIM and 0 <= empty <= MAX_EMPTY_SLOTS:
                    options.append((n, rows, cols, empty, total))
    return random.choice(options) if options else None


def _choose_empty_slots(
    rows: int, cols: int, empty_count: int
) -> tuple[set[tuple[int, int]], list[tuple[int, int]]]:
    slots = [(r, c) for r in range(rows) for c in range(cols)]
    valid = []
    for combo in itertools.combinations(slots, empty_count):
        empty = set(combo)
        filled = [s for s in slots if s not in empty]
        if len({r for r, _ in filled}) == rows and len({c for _, c in filled}) == cols:
            valid.append(empty)
    empty_slots = random.choice(valid)
    filled_slots = [s for s in slots if s not in empty_slots]
    random.shuffle(filled_slots)
    return empty_slots, filled_slots


def _make_bordered_frame(h: int, w: int, bg: int) -> np.ndarray:
    frame = np.full((h, w), bg, dtype=int)
    frame[0, :] = frame[-1, :] = FRAME_COL
    frame[:, 0] = frame[:, -1] = FRAME_COL
    return frame


def _place_patch(
    grid: np.ndarray,
    reserve: np.ndarray,
    patch: np.ndarray,
    bg: int,
    *,
    transparent: bool,
) -> bool:
    blocker = np.full(patch.shape, FRAME_COL, dtype=int)
    try:
        r, c = random_free_location(reserve, blocker, transparent=bg, padding=PATCH_PADDING)
    except ValueError:
        return False
    paste(reserve, blocker, r, c)
    paste(grid, patch, r, c, transparent=bg if transparent else None)
    return True


def _derangement(items: list[int]) -> list[int]:
    """Return a shuffled copy with no fixed points; len(items) is always >= 2 here."""
    out = items[:]
    while True:
        random.shuffle(out)
        if all(a != b for a, b in zip(items, out)):
            return out


def _choose_frame_origin(
    h: int,
    w: int,
    frame_h: int,
    frame_w: int,
    reserved_bbox: tuple[int, int, int, int],
) -> tuple[int, int] | None:
    candidates: list[tuple[int, int]] = []
    trials = 60
    for t in range(trials):
        if t < 24:
            fr = random.choice([0, h - frame_h])
            fc = random.choice([0, w - frame_w])
            if h - frame_h > 2 and random.random() < FRAME_POS_JITTER_PROB:
                fr = random.randrange(0, h - frame_h + 1)
            if w - frame_w > 2 and random.random() < FRAME_POS_JITTER_PROB:
                fc = random.randrange(0, w - frame_w + 1)
        else:
            fr = random.randrange(0, h - frame_h + 1)
            fc = random.randrange(0, w - frame_w + 1)
        bbox = (fr, fc, fr + frame_h - 1, fc + frame_w - 1)
        if not _bbox_overlap(bbox, reserved_bbox):
            candidates.append((fr, fc))
    return random.choice(candidates) if candidates else None


def _gen_attempt() -> np.ndarray | None:
    struct = _choose_structure()
    if struct is None:
        return None
    n, rows, cols, empty_count, _ = struct

    bg = random.choice([c for c in range(10) if c != FRAME_COL])
    usable = [c for c in range(10) if c not in (bg, FRAME_COL)]

    k = random.randint(*NUM_COLORS_RANGE)
    src_cols = random.sample(usable, k)
    dst_cols = _derangement(src_cols)
    dst_to_src = dict(zip(dst_cols, src_cols))
    tgt_cols = [random.choice([c for c in usable if c != src]) for src in src_cols]
    color_to_tgt = dict(zip(src_cols, tgt_cols))
    masks = _unique_sprite_masks(src_cols, n)

    ch, cw = _content_extent(n, rows, cols)
    min_fh = max(MIN_FRAME_H, ch + 4)
    min_fw = max(MIN_FRAME_W, cw + 4)
    frame_h = random.randint(min_fh, min(MAX_FRAME_DIM, max(min_fh, ch + 2 + FRAME_MARGIN_SLACK)))
    frame_w = random.randint(min_fw, min(MAX_FRAME_DIM, max(min_fw, cw + 2 + FRAME_MARGIN_SLACK)))
    y_margin = frame_h - 2 - ch
    x_margin = frame_w - 2 - cw
    top_margin = random.randint(1, y_margin - 1)
    left_margin = random.randint(1, x_margin - 1)

    _, filled_slots = _choose_empty_slots(rows, cols, empty_count)
    cell_colors: dict[tuple[int, int], int] = {}
    for slot, col in zip(filled_slots[:k], dst_cols):
        cell_colors[slot] = col
    for slot in filled_slots[k:]:
        cell_colors[slot] = random.choice(dst_cols)

    input_frame = _make_bordered_frame(frame_h, frame_w, bg)
    output_frame = _make_bordered_frame(frame_h, frame_w, bg)
    for (rr, cc), dst in cell_colors.items():
        src = dst_to_src[dst]
        r0 = 1 + top_margin + rr * (n + 1)
        c0 = 1 + left_margin + cc * (n + 1)
        paste(input_frame, np.full((n, n), dst, dtype=int), r0, c0)
        paste(output_frame, _colored_sprite(masks[src], color_to_tgt[src], bg), r0, c0, transparent=bg)

    lookup_patch = _lookup_patch(dst_cols, src_cols, bg)
    lookup_reserve_h = LOOKUP_H + LOOKUP_BUFFER_BOTTOM
    lookup_reserve_w = lookup_patch.shape[1] + LOOKUP_BUFFER_RIGHT

    ext_area = k * (12 + n * n + 10) + lookup_reserve_h * lookup_reserve_w
    possible_dims = [
        (h, w)
        for h in range(*CANVAS_H_RANGE)
        for w in range(*CANVAS_W_RANGE)
        if h >= frame_h and w >= frame_w and h * w - frame_h * frame_w >= ext_area
    ]
    if not possible_dims:
        return None
    h, w = random.choice(possible_dims)
    grid = np.full((h, w), bg, dtype=int)
    reserve = np.full((h, w), bg, dtype=int)

    paste(grid, lookup_patch, 0, 0)
    paste(reserve, np.full((lookup_reserve_h, lookup_reserve_w), FRAME_COL, dtype=int), 0, 0)

    frfc = _choose_frame_origin(h, w, frame_h, frame_w, (0, 0, lookup_reserve_h - 1, lookup_reserve_w - 1))
    if frfc is None:
        return None
    fr0, fc0 = frfc
    paste(grid, input_frame, fr0, fc0)
    paste(reserve, np.full((frame_h, frame_w), FRAME_COL, dtype=int), fr0, fc0)

    patches: list[tuple[np.ndarray, bool]] = []
    for src, tgt in zip(src_cols, tgt_cols):
        patches.append((_colored_sprite(masks[src], src, bg), True))
        patches.append((_legend(src, tgt), False))
    random.shuffle(patches)
    patches.sort(key=lambda item: item[0].size, reverse=True)
    for patch, transparent in patches:
        if not _place_patch(grid, reserve, patch, bg, transparent=transparent):
            return None

    try:
        if not np.array_equal(main(grid), output_frame):
            return None
    except Exception:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        out = _gen_attempt()
        if out is not None:
            return out
    raise RuntimeError("failed to generate valid edb79dae D4 variant input")
