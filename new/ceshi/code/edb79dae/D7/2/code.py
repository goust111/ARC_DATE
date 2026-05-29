# Variant of edb79dae
# Dimension: D7 — 符号编码：图例的结构与位置 (symbol_instructions)
# Change: 左上角图例由多个条目组成，每个条目展示一个形状模式及其对应的输出颜色，形成形状到颜色的查找表... → 图例为嵌套编码（颜色数量编码而非形状）

import itertools
import random

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, paste, random_free_location

# ---------------------------------------------------------------------------
# Design analysis for the D7 variant
#
# spec_variant:
#   unchanged_core:
#     - keep the gray rectangular frame and the inside placeholder grid
#     - keep one external sprite per placeholder color
#     - keep output = frame crop with placeholders replaced by recolored sprites
#   d7_change:
#     - the legend becomes a fixed top-left strip instead of scattered two-color glyphs
#     - each legend entry is a monochrome 2x3 stencil using the first k cells of a
#       shared reading-order template, so larger codes strictly contain smaller codes
#     - the entry color is the output color; the entry area k is the code
#     - sprites are matched to legend entries by sprite pixel count, not by shape
#   concrete_pixel_behavior:
#     - example: a 4x4 cyan sprite with exactly 5 colored cells is placed inside the frame
#       wherever cyan placeholders appear; if the top-left legend contains an orange
#       5-pixel entry, the placed sprite becomes orange
#     - under the old logic, recoloring depended on finding a shape/color pair legend;
#       under the new logic, the shape can vary freely and only the count matters
#   complexity_calibration:
#     - use 2-5 placeholder colors and unique sprite areas chosen from 2..6
#     - this keeps the count vocabulary small enough to infer visually, while repeated
#       placeholder placements still provide support for each rule inside the frame
#     - sprites stay in 3x3 or 4x4 boxes so counting remains tractable
#   old_assumptions_that_break:
#     - old legend parsing assumed overlapping two-color glyphs; this variant removes
#       overlap entirely, so solver must read top-row legend components instead
#     - old sprite extraction wanted one component already spanning the full bbox;
#       sparse counted sprites may be disconnected, so solver must union all non-legend
#       outside components of a placeholder color
#   legality_and_clean_layout:
#     - generator reserves the top band for legend only, guaranteeing that every outside
#       component touching row 0 is a legend entry
#     - sprite masks always include opposite corners, so the union bbox matches the
#       placeholder NxN bbox even when the sprite is disconnected
#     - reserve masks prevent overlap among legend, frame, and external sprites
#   observability:
#     - human-solvable cue: count the colored cells in an external sprite, then find the
#       legend entry with the same number of pixels to obtain the output color
#     - nested legend symbols make "1/2/3/..." style counting explicit instead of
#       encoding arbitrary shapes
#   solver_assumptions_checked_against_generator:
#     - every placeholder color has exactly one external sprite union
#     - sprite areas are unique and exactly match the legend counts
#     - only legend entries occupy the top band, and every legend entry touches row 0
# ---------------------------------------------------------------------------


FRAME_COL = 5
SPRITE_N_RANGE = (3, 5)             # N in {3, 4}
GRID_ROWS_RANGE = (2, 4)
GRID_COLS_RANGE = (3, 5)
TOTAL_BLOCKS_RANGE = (6, 10)
MAX_EMPTY_SLOTS = 2
NUM_COLORS_RANGE = (2, 5)
SPRITE_AREA_CHOICES = (2, 3, 4, 5, 6)
MIN_FRAME_H = 13
MIN_FRAME_W = 18
MAX_FRAME_DIM = 21
FRAME_MARGIN_SLACK = 4
CANVAS_H_RANGE = (22, 31)
CANVAS_W_RANGE = (23, 31)
PATCH_PADDING = 1
GEN_RETRIES = 300

LEGEND_H = 2
LEGEND_ENTRY_W = 3
LEGEND_GAP = 1
TOP_BAND_H = LEGEND_H + 1
LEGEND_ORDER = ((0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2))


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


def _legend_entries(comps: list[dict]) -> list[tuple[int, int, int, int]]:
    """Return legend entries as (x_pos, out_color, pixel_count, comp_id), sorted by x."""
    entries = [
        (int(comp["bbox"][1]), int(comp["col"]), int(comp["area"]), int(comp["id"]))
        for comp in comps
        if comp["bbox"][0] == 0
    ]
    entries.sort(key=lambda item: item[0])
    return entries


def main(grid: np.ndarray) -> np.ndarray:
    """
    Replace frame placeholders with counted sprites and recolor them using a top-left count legend.

    Input structure:
    - bg background, one rectangular gray frame, and solid-color placeholder blocks inside the frame.
    - Outside the frame: one sprite per placeholder color plus a left-top legend made of compact single-color pixel groups.

    Rule:
    - Find each placeholder color and collect the matching-color sprite outside the frame.
    - Measure how many colored pixels that sprite contains, ignoring its exact shape.
    - Use the top-left legend to map that pixel count to an output color, then stamp the sprite into every matching placeholder slot.

    Key insight: The legend no longer identifies shapes at all; its nested dot counts are the lookup key, so sprite area determines the recoloring.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)

    frame_col, (fr0, fc0, fr1, fc1) = _find_frame(g, bg)

    crop = g[fr0 : fr1 + 1, fc0 : fc1 + 1]
    h, w = crop.shape
    assert h >= 3 and w >= 3, f"frame too small for interior: {(h, w)}"

    inner_mask = np.zeros((h, w), dtype=bool)
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

    legend = _legend_entries(comps)
    assert legend, "count legend must occupy the top row"
    legend_ids = {entry[3] for entry in legend}

    color_map: dict[int, int] = {}
    for _x, out_col, pixel_count, _cid in legend:
        assert pixel_count not in color_map, f"duplicate legend count {pixel_count}"
        color_map[pixel_count] = int(out_col)

    sprite_data: dict[int, tuple[tuple[tuple[int, int], ...], int, int, int]] = {}
    for src_col in sorted(placeholder_colors):
        all_candidates = [c for c in comps if c["col"] == src_col and c["id"] not in legend_ids]
        assert all_candidates, f"no sprite found for placeholder color {src_col}"

        pts_set = set()
        r0 = min(c["bbox"][0] for c in all_candidates)
        c0 = min(c["bbox"][1] for c in all_candidates)
        r1 = max(c["bbox"][2] for c in all_candidates)
        c1 = max(c["bbox"][3] for c in all_candidates)
        for comp in all_candidates:
            pts_set.update(comp["pts"])

        want_h, want_w = phw[src_col]
        assert (r1 - r0 + 1, c1 - c0 + 1) == (want_h, want_w), (
            f"sprite union bbox does not match placeholder size for color {src_col}"
        )
        coords = tuple(sorted((r - r0, c - c0) for r, c in pts_set))
        sprite_data[src_col] = (coords, want_h, want_w, len(coords))

    sprite_counts = {area for _coords, _sh, _sw, area in sprite_data.values()}
    assert sprite_counts == set(color_map), (
        f"legend counts {sorted(color_map)} do not match sprite areas {sorted(sprite_counts)}"
    )

    out = crop.copy()
    out[1:-1, 1:-1] = bg

    for src_col, (pr0, pc0, pr1, pc1) in placeholders:
        coords, sh, sw, area = sprite_data[src_col]
        tgt_col = int(color_map[area])
        assert (pr1 - pr0 + 1, pc1 - pc0 + 1) == (sh, sw), (
            f"placeholder size does not match sprite for color {src_col}"
        )
        for dr, dc in coords:
            out[pr0 + dr, pc0 + dc] = tgt_col

    return out


def _legend_patch(entries: list[tuple[int, int]], bg: int) -> np.ndarray:
    """Build the top-left nested legend from (pixel_count, target_color) pairs."""
    width = len(entries) * LEGEND_ENTRY_W + max(0, len(entries) - 1) * LEGEND_GAP
    patch = np.full((LEGEND_H, width), bg, dtype=int)
    for idx, (pixel_count, color) in enumerate(entries):
        c0 = idx * (LEGEND_ENTRY_W + LEGEND_GAP)
        for rr, cc in LEGEND_ORDER[:pixel_count]:
            patch[rr, c0 + cc] = int(color)
    return patch


def _sprite_mask(n: int, area: int) -> np.ndarray:
    """Generate an NxN sparse mask with exact area and full NxN union bbox."""
    assert 2 <= area <= min(len(LEGEND_ORDER), n * n), f"unsupported area {area} for n={n}"
    mask = np.zeros((n, n), dtype=bool)

    corner_pairs = [
        ((0, 0), (n - 1, n - 1)),
        ((0, n - 1), (n - 1, 0)),
    ]
    a, b = random.choice(corner_pairs)
    chosen = {a, b}
    candidates = [(r, c) for r in range(n) for c in range(n) if (r, c) not in chosen]
    random.shuffle(candidates)
    while len(chosen) < area:
        chosen.add(candidates.pop())

    for r, c in chosen:
        mask[r, c] = True
    return mask


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
    grid: np.ndarray, reserve: np.ndarray, patch: np.ndarray, bg: int,
    *, transparent: bool,
) -> bool:
    blocker = np.full(patch.shape, FRAME_COL, dtype=int)
    try:
        r, c = random_free_location(reserve, blocker, transparent=bg, padding=PATCH_PADDING)
    except ValueError:
        return False
    paste(reserve, blocker, r, c)
    paste(grid, patch, r, c, transparent=bg if transparent else None)
    return True


def _gen_attempt() -> np.ndarray | None:
    struct = _choose_structure()
    if struct is None:
        return None
    n, rows, cols, empty_count, _ = struct

    bg = random.choice([c for c in range(10) if c != FRAME_COL])
    usable = [c for c in range(10) if c not in (bg, FRAME_COL)]

    k = random.randint(*NUM_COLORS_RANGE)
    src_cols = random.sample(usable, k)
    tgt_cols = random.sample(usable, k)
    areas = random.sample(SPRITE_AREA_CHOICES, k)

    color_to_tgt = dict(zip(src_cols, tgt_cols))
    masks = {src: _sprite_mask(n, area) for src, area in zip(src_cols, areas)}
    legend_items = list(zip(areas, tgt_cols))
    random.shuffle(legend_items)
    legend_patch = _legend_patch(legend_items, bg)
    legend_w = legend_patch.shape[1]

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
    cell_colors = {}
    for slot, col in zip(filled_slots[:k], src_cols):
        cell_colors[slot] = col
    for slot in filled_slots[k:]:
        cell_colors[slot] = random.choice(src_cols)

    input_frame = _make_bordered_frame(frame_h, frame_w, bg)
    output_frame = _make_bordered_frame(frame_h, frame_w, bg)
    for (rr, cc), src in cell_colors.items():
        r0 = 1 + top_margin + rr * (n + 1)
        c0 = 1 + left_margin + cc * (n + 1)
        paste(input_frame, np.full((n, n), src, dtype=int), r0, c0)
        paste(output_frame, _colored_sprite(masks[src], color_to_tgt[src], bg), r0, c0, transparent=bg)

    sprite_budget = k * (n * n + 6)
    possible_dims = [
        (h, w)
        for h in range(*CANVAS_H_RANGE)
        for w in range(*CANVAS_W_RANGE)
        if h >= frame_h + TOP_BAND_H
        and w >= max(frame_w, legend_w)
        and h * w - frame_h * frame_w - TOP_BAND_H * w >= sprite_budget
    ]
    if not possible_dims:
        return None
    h, w = random.choice(possible_dims)
    grid = np.full((h, w), bg, dtype=int)
    reserve = np.full((h, w), bg, dtype=int)
    reserve[:TOP_BAND_H, :] = FRAME_COL

    paste(grid, legend_patch, 0, 0, transparent=bg)

    frame_blocker = np.full((frame_h, frame_w), FRAME_COL, dtype=int)
    try:
        fr0, fc0 = random_free_location(reserve, frame_blocker, transparent=bg, padding=PATCH_PADDING)
    except ValueError:
        return None
    paste(grid, input_frame, fr0, fc0)
    paste(reserve, frame_blocker, fr0, fc0)

    patches: list[np.ndarray] = []
    for src in src_cols:
        patches.append(_colored_sprite(masks[src], src, bg))
    random.shuffle(patches)
    patches.sort(key=lambda patch: patch.size, reverse=True)
    for patch in patches:
        if not _place_patch(grid, reserve, patch, bg, transparent=True):
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
    raise RuntimeError("failed to generate valid edb79dae D7 variant input")
