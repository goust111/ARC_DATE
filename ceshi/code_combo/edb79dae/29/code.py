# Combo variant of edb79dae (index 29)
# combo_spec: D13/2+D2/3
# Name: 层级组合图例+位置编码查找

import itertools
import random

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, paste, random_free_location


# Design analysis for the combo variant
# -------------------------------------
# 1. Concrete fused rule
#    - Step 1 stays close to the source task: inside the gray frame, each solid NxN placeholder
#      color still chooses which same-colored external sprite shape gets stamped into that slot.
#    - Step 2 is replaced by a two-layer spatial legend parked at the canvas top-left:
#        left  panel: a 2x2 "base palette" of four solid 3x3 color swatches;
#        right panel: a 2x2 "modifier palette" of four 3x3 glyphs on background.
#    - A placed sprite first finds its own frame-relative region: top/bottom half x left/right half.
#      That region indexes *both* panels. The base panel contributes the default region color. The
#      modifier glyph then says whether to toggle the row half, the column half, both, or neither
#      before reading the final base swatch.
#    - Small trace:
#        original task: placeholder color 2 -> external sprite of color 2 -> legend tied to color 2
#        fused task: placeholder color 2 -> external sprite of color 2 -> region = upper-right
#                    -> modifier at upper-right says "flip row bit"
#                    -> read lower-right base swatch -> repaint with that color.
#      Difference: recolor is no longer a single color-key lookup nor a direct one-panel region map.
#
# 2. Complexity calibration
#    - Keep only four spatial regions, matching the natural 2x2 half-grid split from D2/3. This is
#      rich enough to show position coding clearly, but small enough for humans to track.
#    - Keep only four modifier glyph types:
#        dot        = keep region as-is
#        vertical   = toggle top/bottom half
#        horizontal = toggle left/right half
#        cross      = toggle both halves
#      The generator uses each glyph exactly once per sample and also requires the composed region
#      mapping to be a non-identity permutation, so the modifier layer visibly matters.
#
# 3. Implicit assumption audit
#    - Original `color_map[src] -> tgt` is invalid now. Recolor depends on each placement instance,
#      so `main()` must compute target color per placeholder, not per source color.
#    - Original task could ignore frame-relative row/column ranks. The fused rule depends on them, so
#      the generator must preserve at least one filled slot in every logical row, column, and region.
#    - The new solver assumes the top-left legend area is reserved and fixed-size. The generator
#      blocks that rectangle before placing the frame and the external sprite templates.
#    - The sprite extraction logic still assumes each placeholder color has exactly one NxN sprite
#      bbox outside the frame; generator keeps that invariant from the source task.
#
# 4. Placement legality
#    - Legend rectangle: fixed at rows 0..6, cols 0..14.
#    - Frame: placed with `random_free_location` against a reserve mask that already blocks the
#      legend rectangle, so there is no overlap.
#    - External sprites: placed afterward using the same reserve mask and 1-cell padding, so they
#      stay disjoint from the legend and the frame.
#
# 5. Visual observability and neatness
#    - The base palette is four large solid color squares; the modifier palette is four large glyph
#      blocks using one ink color on background. The two layers are spatially parallel and easy to
#      read side-by-side.
#    - The generator forces one source color to appear in multiple regions, so humans can see that
#      source-color identity alone no longer fixes the output color.
#    - Because the composed mapping is a non-identity permutation, the right panel cannot be ignored:
#      reading only the left panel gives the wrong recolor in the same sample.
#
# 6. New solver-side assumptions checked against generation
#    - `_read_base_panel` assumes each 3x3 swatch is solid and non-background. Generator constructs
#      exactly that.
#    - `_read_modifier_panel` assumes each 3x3 block matches one of four hard-coded masks. Generator
#      emits only those masks.
#    - Region lookup assumes at least a 2x2 logical placement grid after removing empty cells.
#      Generator enforces populated rows, columns, and all four half-regions.


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
PATCH_PADDING = 1
GEN_RETRIES = 300

PANEL_SIZE = 7
PANEL_BLOCK = 3
PANEL_GAP = 1
BASE_PANEL_R0 = 0
BASE_PANEL_C0 = 0
MOD_PANEL_R0 = 0
MOD_PANEL_C0 = PANEL_SIZE + 1
LEGEND_H = PANEL_SIZE
LEGEND_W = PANEL_SIZE * 2 + 1

REGIONS = [(0, 0), (0, 1), (1, 0), (1, 1)]
MOD_KEEP = (0, 0)
MOD_FLIP_COL = (0, 1)
MOD_FLIP_ROW = (1, 0)
MOD_FLIP_BOTH = (1, 1)
MODIFIER_PATTERNS: dict[tuple[int, int], np.ndarray] = {
    MOD_KEEP: np.array(
        [[0, 0, 0],
         [0, 1, 0],
         [0, 0, 0]],
        dtype=bool,
    ),
    MOD_FLIP_COL: np.array(
        [[0, 0, 0],
         [1, 1, 1],
         [0, 0, 0]],
        dtype=bool,
    ),
    MOD_FLIP_ROW: np.array(
        [[0, 1, 0],
         [0, 1, 0],
         [0, 1, 0]],
        dtype=bool,
    ),
    MOD_FLIP_BOTH: np.array(
        [[0, 1, 0],
         [1, 1, 1],
         [0, 1, 0]],
        dtype=bool,
    ),
}


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


def _half_index(rank: int, count: int) -> int:
    """Map an ordered row/column rank to 0 (first half) or 1 (second half)."""
    return 0 if int(rank) * 2 < int(count) else 1


def _block_origin(region: tuple[int, int]) -> tuple[int, int]:
    rr, cc = region
    return rr * (PANEL_BLOCK + PANEL_GAP), cc * (PANEL_BLOCK + PANEL_GAP)


def _extract_panel_block(panel: np.ndarray, region: tuple[int, int]) -> np.ndarray:
    r0, c0 = _block_origin(region)
    return panel[r0 : r0 + PANEL_BLOCK, c0 : c0 + PANEL_BLOCK]


def _read_base_panel(g: np.ndarray, bg: int) -> dict[tuple[int, int], int]:
    """Read the fixed 7x7 base color panel from the top-left corner."""
    assert g.shape[0] >= PANEL_SIZE and g.shape[1] >= LEGEND_W, "grid too small for combo legend"
    panel = g[BASE_PANEL_R0 : BASE_PANEL_R0 + PANEL_SIZE, BASE_PANEL_C0 : BASE_PANEL_C0 + PANEL_SIZE]
    assert np.all(panel[3, :] == bg) and np.all(panel[:, 3] == bg), "base panel must contain a background cross"
    mapping: dict[tuple[int, int], int] = {}
    for region in REGIONS:
        block = _extract_panel_block(panel, region)
        cols = set(map(int, np.unique(block)))
        assert len(cols) == 1, "base panel entries must be solid"
        col = int(next(iter(cols)))
        assert col != bg, "base panel entries cannot use the background color"
        mapping[region] = col
    return mapping


def _read_modifier_panel(g: np.ndarray, bg: int) -> dict[tuple[int, int], tuple[int, int]]:
    """Read the fixed 7x7 modifier glyph panel from the top-left legend area."""
    assert g.shape[0] >= PANEL_SIZE and g.shape[1] >= LEGEND_W, "grid too small for combo legend"
    assert np.all(g[:LEGEND_H, PANEL_SIZE] == bg), "legend panels must be separated by one background column"
    panel = g[MOD_PANEL_R0 : MOD_PANEL_R0 + PANEL_SIZE, MOD_PANEL_C0 : MOD_PANEL_C0 + PANEL_SIZE]
    assert np.all(panel[3, :] == bg) and np.all(panel[:, 3] == bg), "modifier panel must contain a background cross"
    mapping: dict[tuple[int, int], tuple[int, int]] = {}
    for region in REGIONS:
        block = _extract_panel_block(panel, region)
        mask = block != bg
        matched = [delta for delta, pat in MODIFIER_PATTERNS.items() if np.array_equal(mask, pat)]
        assert len(matched) == 1, "modifier entry does not match any allowed glyph"
        mapping[region] = matched[0]
    return mapping


def _compose_region(region: tuple[int, int], modifier: tuple[int, int]) -> tuple[int, int]:
    return region[0] ^ modifier[0], region[1] ^ modifier[1]


def _slot_region(slot: tuple[int, int], rows: int, cols: int) -> tuple[int, int]:
    rr, cc = slot
    return _half_index(rr, rows), _half_index(cc, cols)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stamp external sprites into the frame, then recolor them from a two-layer spatial legend.

    Input structure:
    - The canvas background contains one gray rectangular frame with solid colored placeholders inside.
    - The top-left corner contains a fixed legend rectangle made of two aligned 2x2 panels:
      a left base-color panel of solid swatches and a right modifier panel of glyphs.
    - Outside the frame and outside that reserved legend area: one sprite template per placeholder color.

    Rule:
    - Detect the frame, recover the placeholder grid, and use placeholder color to fetch the matching
      external sprite shape exactly as in the source task.
    - For each placement, determine its frame-relative quadrant (upper/lower half and left/right half).
      Use that quadrant once to read a modifier glyph, use it again as the default base-panel index,
      apply the modifier to that quadrant index, and finally read the target color from the base panel.
    - Stamp the selected sprite into the placeholder box and recolor all sprite pixels to that target color.

    Key insight: The left panel does not directly tell you each output color. Position chooses a base entry,
    but the right-panel glyph at the same position can redirect that lookup to another quadrant, so solving
    requires combining two position-dependent legend layers rather than reading a single map.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)

    frame_col, (fr0, fc0, fr1, fc1) = _find_frame(g, bg)
    assert fr0 >= LEGEND_H or fc0 >= LEGEND_W, "main frame overlaps the reserved combo legend area"
    base_map = _read_base_panel(g, bg)
    modifier_map = _read_modifier_panel(g, bg)

    crop = g[fr0 : fr1 + 1, fc0 : fc1 + 1]
    H, W = crop.shape
    assert H >= 3 and W >= 3, f"frame too small for interior: {(H, W)}"

    inner_mask = np.zeros((H, W), dtype=bool)
    inner_mask[1:-1, 1:-1] = True
    placeholder_colors = set(map(int, np.unique(crop[inner_mask]))) - {bg, frame_col}
    assert placeholder_colors, "no placeholder colors found inside frame"

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

    row_origins = sorted({b[0] for _, b in placeholders})
    col_origins = sorted({b[1] for _, b in placeholders})
    assert len(row_origins) >= 2 and len(col_origins) >= 2, "need at least a 2x2 placement grid"
    row_rank = {r0: i for i, r0 in enumerate(row_origins)}
    col_rank = {c0: i for i, c0 in enumerate(col_origins)}

    outside = np.ones(g.shape, dtype=bool)
    outside[fr0 : fr1 + 1, fc0 : fc1 + 1] = False
    outside[:LEGEND_H, :LEGEND_W] = False

    comps = []
    colors = [int(c) for c in np.unique(g) if int(c) not in (bg, frame_col)]
    for col in colors:
        for pts in components_mask((g == col) & outside, connectivity=4):
            r0, c0, r1, c1 = bbox_coords(pts)
            pts_tuples = [(int(r), int(c)) for r, c in pts]
            comps.append(
                {
                    "col": int(col),
                    "pts": pts_tuples,
                    "bbox": (r0, c0, r1, c1),
                }
            )

    sprite_coords: dict[int, tuple[tuple[tuple[int, int], ...], int, int]] = {}
    for src_col in sorted(placeholder_colors):
        all_candidates = [c for c in comps if c["col"] == src_col]
        assert all_candidates, f"no external sprite found for color {src_col}"
        want_h, want_w = phw[src_col]
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
            if c is base:
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

    for src_col, (pr0, pc0, pr1, pc1) in placeholders:
        assert src_col in sprite_coords, f"placeholder color {src_col} missing sprite"
        region = (_half_index(row_rank[pr0], len(row_origins)), _half_index(col_rank[pc0], len(col_origins)))
        final_region = _compose_region(region, modifier_map[region])
        tgt_col = base_map[final_region]
        coords, sh, sw = sprite_coords[src_col]
        assert (pr1 - pr0 + 1, pc1 - pc0 + 1) == (sh, sw), (
            f"placeholder size does not match sprite for color {src_col}"
        )
        assert pr0 + sh <= H and pc0 + sw <= W, f"sprite for color {src_col} does not fit at placeholder"
        for dr, dc in coords:
            out[pr0 + dr, pc0 + dc] = tgt_col

    return out


def _base_panel(base_colors: dict[tuple[int, int], int], bg: int) -> np.ndarray:
    panel = np.full((PANEL_SIZE, PANEL_SIZE), bg, dtype=int)
    for region, col in base_colors.items():
        r0, c0 = _block_origin(region)
        panel[r0 : r0 + PANEL_BLOCK, c0 : c0 + PANEL_BLOCK] = int(col)
    return panel


def _modifier_panel(region_to_delta: dict[tuple[int, int], tuple[int, int]], ink: int, bg: int) -> np.ndarray:
    panel = np.full((PANEL_SIZE, PANEL_SIZE), bg, dtype=int)
    for region, delta in region_to_delta.items():
        r0, c0 = _block_origin(region)
        mask = MODIFIER_PATTERNS[delta]
        panel[r0 : r0 + PANEL_BLOCK, c0 : c0 + PANEL_BLOCK][mask] = int(ink)
    return panel


def _combo_legend(
    base_colors: dict[tuple[int, int], int], region_to_delta: dict[tuple[int, int], tuple[int, int]], ink: int, bg: int
) -> np.ndarray:
    legend = np.full((LEGEND_H, LEGEND_W), bg, dtype=int)
    legend[:, :PANEL_SIZE] = _base_panel(base_colors, bg)
    legend[:, MOD_PANEL_C0 : MOD_PANEL_C0 + PANEL_SIZE] = _modifier_panel(region_to_delta, ink, bg)
    return legend


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


def _colored_sprite(mask: np.ndarray, color: int, bg: int) -> np.ndarray:
    out = np.full(mask.shape, bg, dtype=int)
    out[mask] = int(color)
    return out


def _content_extent(n: int, grid_rows: int, grid_cols: int) -> tuple[int, int]:
    return grid_rows * n + grid_rows - 1, grid_cols * n + grid_cols - 1


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
    """Choose empty slots while keeping every row, column, and region populated."""
    slots = [(r, c) for r in range(rows) for c in range(cols)]
    valid = []
    for combo in itertools.combinations(slots, empty_count):
        empty = set(combo)
        filled = [s for s in slots if s not in empty]
        if len({r for r, _ in filled}) != rows or len({c for _, c in filled}) != cols:
            continue
        regions = {_slot_region(s, rows, cols) for s in filled}
        if regions != set(REGIONS):
            continue
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
    grid: np.ndarray, reserve: np.ndarray, patch: np.ndarray, bg: int, *, transparent: bool
) -> bool:
    blocker = np.full(patch.shape, FRAME_COL, dtype=int)
    try:
        r, c = random_free_location(reserve, blocker, transparent=bg, padding=PATCH_PADDING)
    except ValueError:
        return False
    paste(reserve, blocker, r, c)
    paste(grid, patch, r, c, transparent=bg if transparent else None)
    return True


def _assign_cell_colors(
    filled_slots: list[tuple[int, int]], src_cols: list[int], rows: int, cols: int
) -> dict[tuple[int, int], int]:
    """Assign source colors, forcing one source color to span at least two regions."""
    region_slots: dict[tuple[int, int], list[tuple[int, int]]] = {region: [] for region in REGIONS}
    for slot in filled_slots:
        region_slots[_slot_region(slot, rows, cols)].append(slot)

    cell_colors: dict[tuple[int, int], int] = {}
    used_slots: set[tuple[int, int]] = set()

    bridge_color = random.choice(src_cols)
    bridge_regions = random.sample(REGIONS, 2)
    for region in bridge_regions:
        slot = random.choice(region_slots[region])
        cell_colors[slot] = bridge_color
        used_slots.add(slot)

    remaining_colors = [c for c in src_cols if c != bridge_color]
    free_slots = [s for s in filled_slots if s not in used_slots]
    random.shuffle(free_slots)
    assert len(free_slots) >= len(remaining_colors), "not enough slots to seed all source colors"
    for col, slot in zip(remaining_colors, free_slots):
        cell_colors[slot] = col
        used_slots.add(slot)

    for slot in filled_slots:
        if slot not in used_slots:
            cell_colors[slot] = random.choice(src_cols)
    return cell_colors


def _sample_modifier_map() -> dict[tuple[int, int], tuple[int, int]]:
    """Assign each region a distinct modifier and require a non-identity composed permutation."""
    deltas = list(MODIFIER_PATTERNS)
    for _ in range(80):
        random.shuffle(deltas)
        region_to_delta = {region: delta for region, delta in zip(REGIONS, deltas)}
        composed = {region: _compose_region(region, region_to_delta[region]) for region in REGIONS}
        values = list(composed.values())
        if len(set(values)) != 4:
            continue
        if all(composed[region] == region for region in REGIONS):
            continue
        return region_to_delta
    raise RuntimeError("failed to sample non-degenerate modifier map")


def _gen_attempt() -> np.ndarray | None:
    struct = _choose_structure()
    if struct is None:
        return None
    n, rows, cols, empty_count, total = struct
    if total < 4:
        return None

    bg = random.choice([c for c in range(10) if c != FRAME_COL])
    usable = [c for c in range(10) if c not in (bg, FRAME_COL)]

    k = random.randint(*NUM_COLORS_RANGE)
    src_cols = random.sample(usable, k)
    masks = {src: _sprite_mask(n) for src in src_cols}

    base_color_list = random.sample(usable, 4)
    base_colors = dict(zip(REGIONS, base_color_list))
    remaining_for_ink = [c for c in usable if c not in base_color_list]
    modifier_ink = random.choice(remaining_for_ink if remaining_for_ink else usable)
    region_to_delta = _sample_modifier_map()

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
    cell_colors = _assign_cell_colors(filled_slots, src_cols, rows, cols)

    input_frame = _make_bordered_frame(frame_h, frame_w, bg)
    output_frame = _make_bordered_frame(frame_h, frame_w, bg)
    for (rr, cc), src in cell_colors.items():
        r0 = 1 + top_margin + rr * (n + 1)
        c0 = 1 + left_margin + cc * (n + 1)
        paste(input_frame, np.full((n, n), src, dtype=int), r0, c0)
        region = _slot_region((rr, cc), rows, cols)
        target_region = _compose_region(region, region_to_delta[region])
        tgt = base_colors[target_region]
        paste(output_frame, _colored_sprite(masks[src], tgt, bg), r0, c0, transparent=bg)

    ext_area = LEGEND_H * LEGEND_W + k * (n * n + 8)
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

    legend = _combo_legend(base_colors, region_to_delta, modifier_ink, bg)
    paste(grid, legend, 0, 0)
    paste(reserve, np.full_like(legend, FRAME_COL), 0, 0)

    frame_blocker = np.full((frame_h, frame_w), FRAME_COL, dtype=int)
    try:
        fr0, fc0 = random_free_location(reserve, frame_blocker, transparent=bg, padding=PATCH_PADDING)
    except ValueError:
        return None
    paste(grid, input_frame, fr0, fc0)
    paste(reserve, frame_blocker, fr0, fc0)

    patches: list[tuple[np.ndarray, bool]] = [(_colored_sprite(masks[src], src, bg), True) for src in src_cols]
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
