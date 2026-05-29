# Variant of edb79dae
# Dimension: D2 — 第二步匹配条件：形状-颜色图例查找 (match_cond)
# Change: 放置后的精灵形状与左上角图例中的形状模式匹配（s2.shape == d2.shape），确定重着色的目标颜色... → 位置编码图例

import itertools
import random

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, paste, random_free_location


# Design analysis for the D2 variant
# ----------------------------------
# source_task: none
# original_task:
#   - A gray rectangular frame contains solid NxN placeholder blocks.
#   - Outside the frame there is one sprite template per placeholder color.
#   - A separate legend chooses the final recolor for each placed sprite.
# variant_core:
#   - Keep step 1 unchanged: placeholder color still selects which external sprite shape
#     gets stamped into the matching block inside the frame.
#   - Replace step 2 completely: the recolor is no longer tied to source color or shape.
#     Instead, a fixed 2x2 legend panel sits in the top-left corner of the whole canvas.
#     Its four entries encode colors for the four relative regions of the frame interior:
#     top-left, top-right, bottom-left, bottom-right.
# concrete_behavior:
#   - The legend is a 5x5 panel:
#       aa.bb
#       aa.bb
#       .....
#       cc.dd
#       cc.dd
#     where "." means background. The four 2x2 colored squares are the four target colors.
#   - The placeholder grid inside the frame still has 2-3 rows and 3-4 columns with 0-2
#     empty cells. The solver reconstructs row/column ranks from placeholder bbox positions.
#   - A placeholder belongs to the upper half if its row-rank is in the first half of the
#     occupied rows, and to the left half if its column-rank is in the first half of the
#     occupied columns. These two bits pick one legend entry.
# behavior_trace:
#   - Old recolor logic at one placement: "this sprite came from placeholder color 3, so find
#     the legend entry associated with color 3, then repaint it to that target color."
#   - New recolor logic at the same placement: "this sprite is being stamped into the frame's
#     upper-right region, so use the upper-right legend square's color, regardless of which
#     placeholder color supplied the shape."
#   - Visible consequence: the same source color can appear in multiple regions and will then
#     produce different output colors in different places.
# complexity_calibration:
#   - Use exactly four legend entries so the position rule is visible but not overparameterized.
#   - Generator guarantees every quadrant has at least one filled placeholder, so each legend
#     color is exercised in each sample.
#   - Generator also forces at least one placeholder color to appear in two different quadrants,
#     preventing accidental collapse back to a per-color recolor rule.
# implicit_assumption_audit:
#   - Original `color_map[src_col] -> tgt_col` assumed a unique legend mapping per source color.
#     That assumption is removed; recolor now depends on placeholder position, so the solver must
#     compute a target for each placement instance, not each source color.
#   - Placeholder row/column recovery assumes at least one filled cell in every logical row and
#     column. The generator enforces that.
#   - Region lookup assumes the top-left 5x5 area is reserved for the legend and does not overlap
#     the main frame. The generator reserves that area before placing the frame.
# placement_legality:
#   - The legend panel is fixed at canvas coordinate (0,0).
#   - The main frame is placed with `random_free_location` against a reserve mask that already
#     blocks the legend panel, so frame and legend never overlap.
#   - External sprite templates are placed afterward using the same reserve mask, so they stay
#     separate from both the legend and the frame.
# visual_observability:
#   - The four legend entries are simple 2x2 solid blocks with distinct colors, making the spatial
#     order obvious to a human.
#   - Output recolor differences can be seen immediately because the same source-colored placeholder
#     may turn into different colors when repeated across different quadrants.


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


def _read_position_legend(g: np.ndarray, bg: int) -> dict[tuple[int, int], int]:
    """Read the fixed 5x5 top-left position legend."""
    assert g.shape[0] >= 5 and g.shape[1] >= 5, "grid too small for top-left legend"
    panel = g[:5, :5]
    cross = np.ones((5, 5), dtype=bool)
    cross[:2, :2] = False
    cross[:2, 3:5] = False
    cross[3:5, :2] = False
    cross[3:5, 3:5] = False
    assert np.all(panel[cross] == bg), "position legend must have a background cross"

    mapping = {
        (0, 0): int(panel[0, 0]),
        (0, 1): int(panel[0, 3]),
        (1, 0): int(panel[3, 0]),
        (1, 1): int(panel[3, 3]),
    }
    for (rr, cc), col in mapping.items():
        r0 = 0 if rr == 0 else 3
        c0 = 0 if cc == 0 else 3
        block = panel[r0 : r0 + 2, c0 : c0 + 2]
        assert col != bg, "legend entry cannot use background color"
        assert len(set(map(int, np.unique(block)))) == 1, "legend entries must be solid 2x2 blocks"
    return mapping


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stamp external sprites into the frame, then recolor them by a spatial legend.

    Input structure:
    - The canvas background contains one gray rectangular frame with solid colored placeholders inside.
    - The top-left corner of the whole canvas contains a fixed 2x2 legend panel made of four colored 2x2 squares.
    - Outside the frame and outside that legend panel: one sprite template per placeholder color.

    Rule:
    - Detect the frame and recover the placeholder grid from the solid blocks inside it.
    - For each placeholder color, fetch the matching-color sprite template from outside the frame.
    - Stamp each sprite into its placeholder box, but choose the repaint color from the legend entry
      indexed by the placeholder's relative position within the frame (upper/lower half and left/right half).

    Key insight: The external sprite color still tells you which shape to place, but the final color is no
    longer keyed by source color; it is chosen by where that placement sits inside the framed grid.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)

    frame_col, (fr0, fc0, fr1, fc1) = _find_frame(g, bg)
    assert fr0 >= 5 or fc0 >= 5, "main frame overlaps the reserved top-left legend area"
    legend_map = _read_position_legend(g, bg)

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
    outside[:5, :5] = False

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
        row_side = _half_index(row_rank[pr0], len(row_origins))
        col_side = _half_index(col_rank[pc0], len(col_origins))
        tgt_col = legend_map[(row_side, col_side)]
        coords, sh, sw = sprite_coords[src_col]
        assert (pr1 - pr0 + 1, pc1 - pc0 + 1) == (sh, sw), (
            f"placeholder size does not match sprite for color {src_col}"
        )
        assert pr0 + sh <= H and pc0 + sw <= W, f"sprite for color {src_col} does not fit at placeholder"
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
PATCH_PADDING = 1
GEN_RETRIES = 300
LEGEND_SIZE = 5
LEGEND_BLOCK = 2


def _slot_region(slot: tuple[int, int], rows: int, cols: int) -> tuple[int, int]:
    rr, cc = slot
    return (_half_index(rr, rows), _half_index(cc, cols))


def _position_legend(region_to_tgt: dict[tuple[int, int], int], bg: int) -> np.ndarray:
    """Build the fixed 5x5 top-left legend panel."""
    panel = np.full((LEGEND_SIZE, LEGEND_SIZE), bg, dtype=int)
    panel[:LEGEND_BLOCK, :LEGEND_BLOCK] = region_to_tgt[(0, 0)]
    panel[:LEGEND_BLOCK, 3:5] = region_to_tgt[(0, 1)]
    panel[3:5, :LEGEND_BLOCK] = region_to_tgt[(1, 0)]
    panel[3:5, 3:5] = region_to_tgt[(1, 1)]
    return panel


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
    """Render a binary mask as a colored sprite on bg."""
    out = np.full(mask.shape, bg, dtype=int)
    out[mask] = int(color)
    return out


def _content_extent(n: int, grid_rows: int, grid_cols: int) -> tuple[int, int]:
    """Content area needed for grid_rows x grid_cols of NxN blocks with 1-cell gaps."""
    return (grid_rows * n + grid_rows - 1, grid_cols * n + grid_cols - 1)


def _choose_structure() -> tuple[int, int, int, int, int] | None:
    """Pick (n, rows, cols, empty_count, total) satisfying frame size constraints."""
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
    """Choose empty slots while keeping every row, column, and quadrant populated."""
    slots = [(r, c) for r in range(rows) for c in range(cols)]
    valid = []
    for combo in itertools.combinations(slots, empty_count):
        empty = set(combo)
        filled = [s for s in slots if s not in empty]
        if len({r for r, _ in filled}) != rows or len({c for _, c in filled}) != cols:
            continue
        regions = {_slot_region(s, rows, cols) for s in filled}
        if regions != {(0, 0), (0, 1), (1, 0), (1, 1)}:
            continue
        valid.append(empty)
    empty_slots = random.choice(valid)
    filled_slots = [s for s in slots if s not in empty_slots]
    random.shuffle(filled_slots)
    return empty_slots, filled_slots


def _make_bordered_frame(h: int, w: int, bg: int) -> np.ndarray:
    """Create an h x w frame filled with bg and a 1-cell FRAME_COL border."""
    frame = np.full((h, w), bg, dtype=int)
    frame[0, :] = frame[-1, :] = FRAME_COL
    frame[:, 0] = frame[:, -1] = FRAME_COL
    return frame


def _place_patch(
    grid: np.ndarray, reserve: np.ndarray, patch: np.ndarray, bg: int,
    *, transparent: bool,
) -> bool:
    """Find a free location for patch and paste it onto grid/reserve. Returns success."""
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
    region_slots: dict[tuple[int, int], list[tuple[int, int]]] = {(0, 0): [], (0, 1): [], (1, 0): [], (1, 1): []}
    for slot in filled_slots:
        region_slots[_slot_region(slot, rows, cols)].append(slot)

    cell_colors: dict[tuple[int, int], int] = {}
    used_slots: set[tuple[int, int]] = set()

    bridge_color = random.choice(src_cols)
    bridge_regions = random.sample(list(region_slots), 2)
    for reg in bridge_regions:
        slot = random.choice(region_slots[reg])
        cell_colors[slot] = bridge_color
        used_slots.add(slot)

    remaining_colors = [c for c in src_cols if c != bridge_color]
    free_slots = [s for s in filled_slots if s not in used_slots]
    random.shuffle(free_slots)
    assert len(free_slots) >= len(remaining_colors), "not enough slots to seed all source colors"
    for col, slot in zip(remaining_colors, free_slots):
        cell_colors[slot] = col
        used_slots.add(slot)

    free_slots = [s for s in filled_slots if s not in used_slots]
    for slot in free_slots:
        cell_colors[slot] = random.choice(src_cols)
    return cell_colors


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

    tgt_cols = random.sample(usable, 4)
    region_order = [(0, 0), (0, 1), (1, 0), (1, 1)]
    region_to_tgt = dict(zip(region_order, tgt_cols))

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
        tgt = region_to_tgt[_slot_region((rr, cc), rows, cols)]
        paste(output_frame, _colored_sprite(masks[src], tgt, bg), r0, c0, transparent=bg)

    ext_area = LEGEND_SIZE * LEGEND_SIZE + k * (n * n + 10)
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

    legend = _position_legend(region_to_tgt, bg)
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
    raise RuntimeError("failed to generate valid edb79dae D2 position-legend input")
