# Combo variant of edb79dae (index 2)
# combo_spec: D1/4+D4/3
# Name: 双重间接颜色映射

import itertools
import random

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, paste, random_free_location


# Design analysis for the combo variant
# -------------------------------------
# a) Concrete behavior and trace:
#    - Original chain: frame block color src -> same-color outside sprite -> recolor legend src->tgt.
#    - D4 alone inserts a lookup: dst -> src -> same-color sprite -> recolor to tgt.
#    - D1 alone flips sprite lookup: src -> complement(src) sprite -> recolor to tgt.
#    - This combo composes them instead of running them independently:
#         frame query color dst
#         -> top-left lookup legend decodes dst -> mid
#         -> complement relation picks actual sprite color comp(mid)
#         -> overlapping 3x4 legend recolors mid -> tgt
#      Example:
#         frame block = 6
#         top-left tile says 6 over 1, so dst 6 means mid 1
#         complement rule says mid 1 uses sprite color 2
#         outside 3x4 legend says 1 -> 8
#         output pastes the color-2 sprite shape and repaints it to 8
#      Relative to the original task, both the query color and the actual sprite color are displaced.
# b) Complexity calibration:
#    - The complement system is kept small and repeated: 1<->2, 3<->4, 6<->7.
#    - Each sample uses exactly four mid colors: one full complement pair plus one singleton from each
#      remaining pair. This keeps the mapping learnable while guaranteeing a visibly meaningful swap.
#    - The lookup permutation is constrained so every placement satisfies dst != mid and
#      dst != complement(mid). That prevents the combo from collapsing back to either single variant.
#    - Sprite masks are generated pairwise distinct, so a wrong intermediate inference changes shape visibly.
# c) Old assumptions that break and the needed fixes:
#    - "Placeholder color directly identifies the sprite": false twice over; frame color only names a lookup key.
#    - "Legend source can be inferred by checking which color appears in the frame": unsafe, because frame colors
#      are dst while the overlapping legends are keyed by mid. We therefore infer recolor legends by their fixed
#      7-vs-5 area asymmetry exactly as in the D4 reference variant.
#    - "Non-legend outside components of a color are sprites for that same semantic color": false, because the
#      semantic key is mid while the visible sprite color is complement(mid). Solver extraction is keyed by mid
#      but searches for outside components colored complement(mid).
#    - "Top-left small bars can be ignored": false once lookup tiles are introduced. Solver explicitly excludes
#      components lying inside the lookup bbox when gathering sprite candidates.
# d) Placement legality:
#    - The top-left lookup occupies a reserved rectangle with a blank buffer below and on its right.
#    - The frame is sampled so its bbox never overlaps that reserved lookup zone.
#    - External sprite patches and 3x4 legends still use the original non-overlap placement utility with padding.
#    - Sprite bbox size remains identical to the placeholder block size, so stamping stays top-left aligned.
# e) Visual observability:
#    - A human sees three distinct symbolic regions: frame queries, a top-left query->mid table, and scattered
#      outside templates/legends. The chain is structural rather than based on numeric color ids.
#    - Because every sample contains a full complement pair among the mids and every query color is "wrong" for
#      both the mid and the actual sprite, the two-step indirection is visible from 2-4 examples.
# f) Visual cleanliness:
#    - The lookup legend is isolated in the top-left corner, while sprite templates and recolor legends stay
#      elsewhere. This avoids stacking multiple cue types into one crowded region.
#    - Final recolor targets are drawn from {0,8,9} and kept separate from the six complement colors so the
#      output-color cue does not pollute either lookup stage.
# g) Generator guarantees for solver assumptions:
#    - Lookup tiles are exactly 2x2, touch row 0, and form the first top-row non-bg cluster.
#    - query_to_mid is a bijection over the four used colors, with no key equal to its mapped mid or sprite color.
#    - Every mid has exactly one outside complementary-color sprite bbox matching the placeholder size.
#    - Every mid also has exactly one overlapping recolor legend.


FRAME_COL = 5
COMPLEMENT_PAIRS = ((1, 2), (3, 4), (6, 7))
COMPLEMENT_MAP = {a: b for x, y in COMPLEMENT_PAIRS for a, b in ((x, y), (y, x))}
MID_COLOR_COUNT = 4
BG_CHOICES = (0, 8, 9)
TARGET_POOL = (0, 8, 9)

# === CFG: inherited structure tunables ===
SPRITE_N_RANGE = (3, 6)
GRID_ROWS_RANGE = (2, 4)
GRID_COLS_RANGE = (3, 5)
TOTAL_BLOCKS_RANGE = (6, 10)
MAX_EMPTY_SLOTS = 2
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
      [mid mid]
    """
    h, _ = g.shape
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
    query_to_mid: dict[int, int] = {}
    for run in runs:
        assert len(run) == 2 and run[1] == run[0] + 1, f"lookup run must be width 2, got {run}"
        c0 = run[0]
        tile = g[0:2, c0 : c0 + 2]
        top_vals = set(map(int, tile[0]))
        bot_vals = set(map(int, tile[1]))
        assert len(top_vals) == 1 and len(bot_vals) == 1, f"lookup tile at {c0} must have solid rows"
        dst = int(tile[0, 0])
        mid = int(tile[1, 0])
        assert dst != bg and mid != bg and dst != mid, f"invalid lookup tile colors at {c0}"
        assert dst not in query_to_mid, f"duplicate lookup key color {dst}"
        query_to_mid[dst] = mid
        if h > 2:
            assert np.all(g[2, c0 : c0 + 2] == bg), "lookup tiles must be isolated by a blank row"

    return query_to_mid, (0, min(cols), 1, max(cols))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill the frame with recolored sprites after decoding frame colors through a lookup table and then a complement rule.

    Input structure:
    - A background canvas contains one gray rectangular frame whose interior holds solid query-colored placeholder blocks.
    - Outside the frame are complementary-color sprite templates, the usual overlapping 3x4 recolor legends keyed by
      intermediate colors, and a separate top-left lookup legend whose 2x2 tiles decode query color -> intermediate color.

    Rule:
    - Detect the frame and collect every placeholder block inside it.
    - Read the top-left lookup legend to convert each frame color dst into an intermediate color mid.
    - Use the fixed complement relation on mid to choose the actual outside sprite color, then read the overlapping
      legend keyed by mid to find the final repaint color tgt.
    - Stamp that complementary-color sprite into each placeholder bbox and repaint its pixels to tgt.

    Key insight: Neither the frame color nor the sprite color is the direct semantic key by itself. The correct shape
    comes from a chain: query color -> lookup legend -> intermediate color -> complement swap -> sprite template,
    while the final color is still controlled by the legend attached to the intermediate color.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)

    frame_col, (fr0, fc0, fr1, fc1) = _find_frame(g, bg)
    query_to_mid, lookup_bbox = _lookup_legend(g, bg)

    crop = g[fr0 : fr1 + 1, fc0 : fc1 + 1]
    h, w = crop.shape
    assert h >= 3 and w >= 3, f"frame too small for interior: {(h, w)}"

    inner_mask = np.zeros((h, w), dtype=bool)
    inner_mask[1:-1, 1:-1] = True
    placeholder_colors = set(map(int, np.unique(crop[inner_mask]))) - {bg, frame_col}
    assert placeholder_colors, "no placeholder colors found inside frame"
    assert set(query_to_mid) == placeholder_colors, (
        f"lookup keys {sorted(query_to_mid)} do not match frame colors {sorted(placeholder_colors)}"
    )

    mid_colors = set(map(int, query_to_mid.values()))
    assert mid_colors <= set(COMPLEMENT_MAP), f"unsupported intermediate colors: {sorted(mid_colors - set(COMPLEMENT_MAP))}"
    assert len(mid_colors) == len(query_to_mid), "lookup must be bijective over the used colors"

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

    mid_dims: dict[int, tuple[int, int]] = {}
    for dst, mid in query_to_mid.items():
        sprite_col = COMPLEMENT_MAP[mid]
        assert dst != sprite_col, f"query color {dst} must differ from complementary sprite color {sprite_col}"
        if mid in mid_dims:
            assert mid_dims[mid] == phw[dst], f"inconsistent placeholder sizes for intermediate color {mid}"
        else:
            mid_dims[mid] = phw[dst]

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

            assert ci["area"] != cj["area"], f"ambiguous equal-area legend pair: {ci['col']} and {cj['col']}"
            if ci["area"] > cj["area"]:
                mid, tgt = ci["col"], cj["col"]
            else:
                mid, tgt = cj["col"], ci["col"]

            if mid not in mid_colors:
                continue
            assert mid not in color_map, f"duplicate legend mapping for intermediate color {mid}"
            color_map[mid] = int(tgt)
            color_legend_ids.add(ci["id"])
            color_legend_ids.add(cj["id"])

    assert set(color_map) == mid_colors, (
        f"recolor legends {sorted(color_map)} do not match intermediate colors {sorted(mid_colors)}"
    )

    sprite_coords: dict[int, tuple[tuple[tuple[int, int], ...], int, int]] = {}
    for mid_col in sorted(mid_colors):
        sprite_col = COMPLEMENT_MAP[mid_col]
        all_candidates = [
            c
            for c in comps
            if c["col"] == sprite_col and c["id"] not in color_legend_ids and not _bbox_inside(c["bbox"], lookup_bbox)
        ]
        assert all_candidates, f"no sprite candidate found for complementary color {sprite_col} of {mid_col}"
        want_h, want_w = mid_dims[mid_col]
        sized = [
            c
            for c in all_candidates
            if (c["bbox"][2] - c["bbox"][0] + 1) == want_h and (c["bbox"][3] - c["bbox"][1] + 1) == want_w
        ]
        assert len(sized) == 1, f"expected one size-matched complementary sprite for {mid_col}, found {len(sized)}"
        base = sized[0]
        sr0, sc0, sr1, sc1 = base["bbox"]

        pts_set = set(base["pts"])
        for c in all_candidates:
            if c["id"] == base["id"]:
                continue
            r0, c0, r1, c1 = c["bbox"]
            assert sr0 <= r0 and sr1 >= r1 and sc0 <= c0 and sc1 >= c1, (
                f"extra sprite component for complementary color {sprite_col} is outside the base bbox"
            )
            pts_set.update(c["pts"])

        coords = tuple(sorted((r - sr0, c - sc0) for r, c in pts_set))
        sprite_coords[mid_col] = (coords, sr1 - sr0 + 1, sc1 - sc0 + 1)

    out = crop.copy()
    out[1:-1, 1:-1] = bg

    for dst_col, (pr0, pc0, pr1, pc1) in placeholders:
        mid_col = query_to_mid[dst_col]
        tgt_col = int(color_map[mid_col])
        coords, sh, sw = sprite_coords[mid_col]
        assert (pr1 - pr0 + 1, pc1 - pc0 + 1) == (sh, sw), (
            f"placeholder size does not match sprite for query color {dst_col}"
        )
        assert pr0 + sh <= h and pc0 + sw <= w, f"sprite for query color {dst_col} does not fit at placeholder"
        for dr, dc in coords:
            out[pr0 + dr, pc0 + dc] = tgt_col

    return out


def _legend(mid: int, tgt: int) -> np.ndarray:
    """Build a 3x4 legend block mapping intermediate color to output color."""
    return np.array(
        [[mid, mid, tgt, tgt], [mid, mid, mid, tgt], [mid, mid, tgt, tgt]],
        dtype=int,
    )


def _lookup_patch(dst_cols: list[int], mid_cols: list[int], bg: int) -> np.ndarray:
    """Build a top-left lookup legend of 2x2 tiles stacked as [dst][mid]."""
    w = len(dst_cols) * (LOOKUP_TILE_W + LOOKUP_GAP) - LOOKUP_GAP
    patch = np.full((LOOKUP_H, w), bg, dtype=int)
    for i, (dst, mid) in enumerate(zip(dst_cols, mid_cols)):
        c0 = i * (LOOKUP_TILE_W + LOOKUP_GAP)
        patch[0, c0 : c0 + LOOKUP_TILE_W] = int(dst)
        patch[1, c0 : c0 + LOOKUP_TILE_W] = int(mid)
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


def _unique_sprite_masks(mid_cols: list[int], n: int) -> dict[int, np.ndarray]:
    """Generate pairwise distinct sprite masks keyed by intermediate color."""
    masks: dict[int, np.ndarray] = {}
    seen: set[tuple[tuple[int, ...], ...]] = set()
    for mid in mid_cols:
        for _ in range(200):
            mask = _sprite_mask(n)
            key = tuple(tuple(int(v) for v in row) for row in mask.astype(int))
            if key not in seen:
                seen.add(key)
                masks[mid] = mask
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


def _choose_mid_colors() -> list[int]:
    """Choose four intermediate colors with exactly one full complement pair present."""
    pair_idx = random.randrange(len(COMPLEMENT_PAIRS))
    primary_pair = list(COMPLEMENT_PAIRS[pair_idx])
    mids = primary_pair[:]
    for other_idx, pair in enumerate(COMPLEMENT_PAIRS):
        if other_idx == pair_idx:
            continue
        mids.append(random.choice(pair))
    random.shuffle(mids)
    assert len(mids) == MID_COLOR_COUNT
    return mids


def _query_colors(mid_cols: list[int]) -> list[int]:
    """
    Choose dst colors as a permutation of the mids such that each mapping uses two active indirections:
      dst != mid and dst != complement(mid)
    """
    options = []
    for perm in itertools.permutations(mid_cols):
        if all(dst != mid and dst != COMPLEMENT_MAP[mid] for dst, mid in zip(perm, mid_cols)):
            options.append(list(perm))
    assert options, f"no valid query permutation for mids {mid_cols}"
    return random.choice(options)


def _choose_frame_origin(
    h: int,
    w: int,
    frame_h: int,
    frame_w: int,
    reserved_bbox: tuple[int, int, int, int],
) -> tuple[int, int] | None:
    candidates: list[tuple[int, int]] = []
    for t in range(60):
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

    bg = random.choice(BG_CHOICES)
    tgt_choices = [c for c in TARGET_POOL if c != bg]
    assert len(tgt_choices) >= 2

    mid_cols = _choose_mid_colors()
    dst_cols = _query_colors(mid_cols)
    query_to_mid = dict(zip(dst_cols, mid_cols))
    tgt_cols = [random.choice(tgt_choices) for _ in mid_cols]
    color_to_tgt = dict(zip(mid_cols, tgt_cols))
    masks = _unique_sprite_masks(mid_cols, n)

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
    for slot, col in zip(filled_slots[:MID_COLOR_COUNT], dst_cols):
        cell_colors[slot] = col
    for slot in filled_slots[MID_COLOR_COUNT:]:
        cell_colors[slot] = random.choice(dst_cols)

    input_frame = _make_bordered_frame(frame_h, frame_w, bg)
    output_frame = _make_bordered_frame(frame_h, frame_w, bg)
    for (rr, cc), dst in cell_colors.items():
        mid = query_to_mid[dst]
        r0 = 1 + top_margin + rr * (n + 1)
        c0 = 1 + left_margin + cc * (n + 1)
        paste(input_frame, np.full((n, n), dst, dtype=int), r0, c0)
        paste(output_frame, _colored_sprite(masks[mid], color_to_tgt[mid], bg), r0, c0, transparent=bg)

    lookup_patch = _lookup_patch(dst_cols, mid_cols, bg)
    lookup_reserve_h = LOOKUP_H + LOOKUP_BUFFER_BOTTOM
    lookup_reserve_w = lookup_patch.shape[1] + LOOKUP_BUFFER_RIGHT

    ext_area = MID_COLOR_COUNT * (12 + n * n + 10) + lookup_reserve_h * lookup_reserve_w
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
    for mid in mid_cols:
        sprite_col = COMPLEMENT_MAP[mid]
        patches.append((_colored_sprite(masks[mid], sprite_col, bg), True))
        patches.append((_legend(mid, color_to_tgt[mid]), False))
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
    raise RuntimeError("failed to generate valid edb79dae combo input")
