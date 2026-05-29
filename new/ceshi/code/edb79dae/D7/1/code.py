# Variant of edb79dae
# Dimension: D7 — 符号编码：图例的结构与位置 (symbol_instructions)
# Change: 左上角图例由多个条目组成，每个条目展示一个形状模式及其对应的输出颜色，形成形状到颜色的查找表... → 图例分散在网格边缘

import itertools
import random

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, paste, random_free_location


# ---------------------------------------------------------------------------
# Design analysis for the D7 variant
#
# concretization:
#   - The framed area, placeholder grid, sprite templates, and recoloring goal stay
#     the same as the base task.
#   - Only the legend mechanism changes. Each mapping entry is split into two edge
#     fragments:
#       1) a source fragment on the top or left edge: a same-color attachment bar
#          plus the sprite silhouette beneath / beside it,
#       2) a target fragment on the opposite edge: a plain color bar.
#   - Vertical legend entries use top+bottom pairing at the same column span.
#     Horizontal legend entries use left+right pairing at the same row span.
#   - The solver must gather fragments from all four outer edges and reconstruct
#     src->tgt mappings by matching opposite-edge spans, instead of reading one
#     compact legend block.
#
# behavior trace:
#   - Original logic: find compact overlapping two-color legend blobs outside the
#     frame, infer src/tgt directly from each local blob, then recolor sprites.
#   - Variant logic: read all top source bars and bottom target bars, pair them by
#     shared width/span; read all left source bars and right target bars, pair them
#     by shared height/span; union both sets into the full mapping table.
#   - The difference is visible: legend information is no longer locally complete.
#     A top fragment alone shows the source symbol but not the output color; the
#     bottom/right fragment alone shows a color but not which symbol it belongs to.
#
# complexity calibration:
#   - Each sample uses 2-5 placeholder colors as before, but the legend entries are
#     split across both vertical and horizontal pairings so the variant is observable.
#   - Pairing is constrained to one of two simple geometric mechanisms
#     (top<->bottom, left<->right). This keeps the task readable while still
#     requiring aggregation.
#
# implicit-assumption audit of the original solver:
#   - Original legend detection assumed legend parts form small overlapping blobs.
#     That assumption breaks completely here, so the pairwise bbox-overlap search
#     must be replaced.
#   - Original sprite extraction excluded legend components via ids discovered during
#     local-pair detection. Here legend fragments are edge-aligned regions, so the
#     solver must build an explicit legend mask and exclude those whole edge zones
#     before searching for sprite templates.
#   - Original frame finding assumes no other component looks like a hollow rectangle.
#     The generator therefore avoids creating frame-like edge symbols and frame-like
#     sprite masks.
#
# placement legality:
#   - Edge legend fragments occupy reserved border zones and never overlap the frame.
#   - External sprite templates are placed only in the remaining free area and are
#     forbidden from touching the outer boundary, so every boundary-touching object
#     belongs to the legend.
#   - Vertical legend slots stay away from left/right corners; horizontal slots stay
#     away from top/bottom corners. This prevents edge fragments from colliding.
#
# visual observability / neatness:
#   - The same entry is visibly split: source-symbol tags appear on top/left edges,
#     recolor bars appear on bottom/right edges.
#   - Because pairing uses aligned spans on opposite edges, humans can infer the
#     grouping from a few examples without relying on hidden color-id ordering.
#   - The frame remains visually dominant in the center, while legends sit cleanly
#     on the canvas boundary and sprite templates float in the remaining background.
#
# color-change inferability:
#   - Target colors are still only 2-5 distinct mappings, and each mapping is
#     supported by repeated placeholder occurrences inside the frame.
#   - The color change is not arbitrary: the edge legend explicitly links a source
#     symbol tag to a target color strip.
#
# new main() assumptions and generator guarantees:
#   - Every top source run has exactly one matching bottom target run with the same
#     column span; every left source run has exactly one matching right target run
#     with the same row span.
#   - Source fragment widths/heights match the placeholder sprite size for that
#     source color. The generator enforces this by using the same global N.
#   - No non-legend object touches the outer grid boundary. The generator reserves
#     the whole boundary before placing free sprites.
# ---------------------------------------------------------------------------


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
CANVAS_H_RANGE = (28, 38)
CANVAS_W_RANGE = (28, 38)
PATCH_PADDING = 1
GEN_RETRIES = 400


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


def _edge_runs(arr: np.ndarray, blocked: set[int]) -> list[tuple[int, int, int]]:
    runs: list[tuple[int, int, int]] = []
    i = 0
    vals = [int(x) for x in np.asarray(arr).tolist()]
    while i < len(vals):
        col = vals[i]
        if col in blocked:
            i += 1
            continue
        j = i
        while j + 1 < len(vals) and vals[j + 1] == col:
            j += 1
        runs.append((i, j, col))
        i = j + 1
    return runs


def _is_rectangular_border_mask(mask: np.ndarray) -> bool:
    if mask.ndim != 2 or min(mask.shape) < 3:
        return False
    border = np.zeros(mask.shape, dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True
    return bool(np.all(mask[border]) and not np.all(mask[~border]))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stamp recolored sprites into a framed region using split edge legends.

    Input structure:
    - Background plus one rectangular gray frame with solid placeholder blocks inside.
    - Outside the frame: one sprite template per placeholder color, and legend fragments spread across the grid edges.
    - Each legend entry is split across opposite edges: the top/left fragment shows the source symbol, while the opposite bottom/right fragment shows only the output color.

    Rule:
    - Find the frame and the placeholder blocks inside it.
    - Reconstruct the color mapping by pairing top with bottom spans and left with right spans along the outer boundary.
    - For each placeholder color, find the matching sprite template outside the frame, stamp it into each placeholder box, and recolor it using the reconstructed edge legend.

    Key insight: No single legend fragment is complete by itself; the solver must aggregate boundary fragments from multiple edges, pair opposite spans into full legend entries, and then apply those mappings to the placed sprites.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)

    frame_col, (fr0, fc0, fr1, fc1) = _find_frame(g, bg)

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

    blocked = {bg, frame_col}
    top_runs = _edge_runs(g[0, :], blocked)
    bottom_runs = _edge_runs(g[-1, :], blocked)
    left_runs = _edge_runs(g[:, 0], blocked)
    right_runs = _edge_runs(g[:, -1], blocked)

    bottom_by_span = {(c0, c1): col for c0, c1, col in bottom_runs}
    right_by_span = {(r0, r1): col for r0, r1, col in right_runs}
    assert len(bottom_by_span) == len(bottom_runs), "duplicate bottom legend spans"
    assert len(right_by_span) == len(right_runs), "duplicate right legend spans"

    color_map: dict[int, int] = {}
    legend_mask = np.zeros(g.shape, dtype=bool)

    for c0, c1, src in top_runs:
        assert src in placeholder_colors, f"top legend source color {src} is not a placeholder"
        want_h, want_w = phw[src]
        assert c1 - c0 + 1 == want_w, f"top legend width mismatch for color {src}"
        assert (c0, c1) in bottom_by_span, f"missing bottom legend pair for top span {(c0, c1)}"
        tgt = int(bottom_by_span[(c0, c1)])
        assert src not in color_map, f"duplicate legend mapping for placeholder color {src}"
        color_map[src] = tgt
        legend_mask[0 : want_h + 1, c0 : c1 + 1] = True
        legend_mask[g.shape[0] - 1 : g.shape[0], c0 : c1 + 1] = True

    for r0, r1, src in left_runs:
        assert src in placeholder_colors, f"left legend source color {src} is not a placeholder"
        want_h, want_w = phw[src]
        assert r1 - r0 + 1 == want_h, f"left legend height mismatch for color {src}"
        assert (r0, r1) in right_by_span, f"missing right legend pair for left span {(r0, r1)}"
        tgt = int(right_by_span[(r0, r1)])
        assert src not in color_map, f"duplicate legend mapping for placeholder color {src}"
        color_map[src] = tgt
        legend_mask[r0 : r1 + 1, 0 : want_w + 1] = True
        legend_mask[r0 : r1 + 1, g.shape[1] - 1 : g.shape[1]] = True

    assert set(color_map) == placeholder_colors, (
        f"legend mappings {sorted(color_map)} do not match placeholders {sorted(placeholder_colors)}"
    )

    outside = np.ones(g.shape, dtype=bool)
    outside[fr0 : fr1 + 1, fc0 : fc1 + 1] = False
    outside &= ~legend_mask

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
                }
            )
            cid += 1

    sprite_coords: dict[int, tuple[tuple[tuple[int, int], ...], int, int]] = {}
    for src_col in sorted(placeholder_colors):
        all_candidates = [c for c in comps if c["col"] == src_col]
        assert all_candidates, f"no sprite candidate found for color {src_col}"
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

    for src_col, (pr0, pc0, pr1, pc1) in placeholders:
        assert src_col in sprite_coords and src_col in color_map, f"placeholder color {src_col} missing sprite or mapping"
        tgt_col = int(color_map[src_col])
        coords, sh, sw = sprite_coords[src_col]
        assert (pr1 - pr0 + 1, pc1 - pc0 + 1) == (sh, sw), (
            f"placeholder size does not match sprite for color {src_col}"
        )
        off_r = pr0
        off_c = pc0
        for dr, dc in coords:
            out[off_r + dr, off_c + dc] = tgt_col

    return out


def _sprite_mask(n: int) -> np.ndarray:
    while True:
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

        if not _is_rectangular_border_mask(mask):
            return mask


def _colored_sprite(mask: np.ndarray, color: int, bg: int) -> np.ndarray:
    out = np.full(mask.shape, bg, dtype=int)
    out[mask] = int(color)
    return out


def _top_source_patch(mask: np.ndarray, color: int, bg: int) -> np.ndarray:
    n = mask.shape[0]
    patch = np.full((n + 1, n), bg, dtype=int)
    patch[0, :] = int(color)
    patch[1:, :][mask] = int(color)
    patch[-1, 0] = bg
    return patch


def _left_source_patch(mask: np.ndarray, color: int, bg: int) -> np.ndarray:
    n = mask.shape[0]
    patch = np.full((n, n + 1), bg, dtype=int)
    patch[:, 0] = int(color)
    patch[:, 1:][mask] = int(color)
    patch[0, -1] = bg
    return patch


def _target_bar(length: int, color: int, *, vertical: bool) -> np.ndarray:
    if vertical:
        return np.full((1, length), int(color), dtype=int)
    return np.full((length, 1), int(color), dtype=int)


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


def _slot_starts(limit: int, span: int, count: int, margin: int) -> list[int] | None:
    if count == 0:
        return []
    need = count * span + (count - 1)
    extra = limit - 2 * margin - need
    if extra < 0:
        return None
    offset = margin + random.randint(0, extra)
    return [offset + i * (span + 1) for i in range(count)]


def _place_patch(
    grid: np.ndarray, reserve: np.ndarray, patch: np.ndarray, bg: int
) -> bool:
    blocker = np.full(patch.shape, FRAME_COL, dtype=int)
    try:
        r, c = random_free_location(reserve, blocker, transparent=bg, padding=PATCH_PADDING)
    except ValueError:
        return False
    paste(reserve, blocker, r, c)
    paste(grid, patch, r, c, transparent=bg)
    return True


def _legend_entry_layout(
    src_cols: list[int], n: int, h: int, w: int
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]] | None:
    corner_clear = n + 2
    if len(src_cols) < 2:
        return None
    vcount = random.randint(1, len(src_cols) - 1)
    hcount = len(src_cols) - vcount
    top_starts = _slot_starts(w, n, vcount, corner_clear)
    left_starts = _slot_starts(h, n, hcount, corner_clear)
    if top_starts is None or left_starts is None:
        return None
    order = src_cols[:]
    random.shuffle(order)
    vertical = [(order[i], top_starts[i]) for i in range(vcount)]
    horizontal = [(order[vcount + i], left_starts[i]) for i in range(hcount)]
    return vertical, horizontal


def _gen_attempt() -> np.ndarray | None:
    struct = _choose_structure()
    if struct is None:
        return None
    n, rows, cols, empty_count, _ = struct

    bg = random.choice([c for c in range(10) if c != FRAME_COL])
    usable = [c for c in range(10) if c not in (bg, FRAME_COL)]

    k = random.randint(*NUM_COLORS_RANGE)
    src_cols = random.sample(usable, k)
    tgt_cols = [random.choice([c for c in usable if c != src]) for src in src_cols]
    color_to_tgt = dict(zip(src_cols, tgt_cols))
    masks = {src: _sprite_mask(n) for src in src_cols}

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

    edge_band = n + 1
    corner_clear = n + 2
    vcount_min = 1
    hcount_min = 1
    min_canvas_h = max(
        CANVAS_H_RANGE[0],
        frame_h + 2 * edge_band,
        2 * corner_clear + hcount_min * n,
    )
    min_canvas_w = max(
        CANVAS_W_RANGE[0],
        frame_w + 2 * edge_band,
        2 * corner_clear + vcount_min * n,
    )
    ext_area = k * (2 * n * n + 8 * n + 8)
    possible_dims = [
        (h, w)
        for h in range(min_canvas_h, CANVAS_H_RANGE[1])
        for w in range(min_canvas_w, CANVAS_W_RANGE[1])
        if h * w - frame_h * frame_w >= ext_area
    ]
    if not possible_dims:
        return None
    h, w = random.choice(possible_dims)

    layout = _legend_entry_layout(src_cols, n, h, w)
    if layout is None:
        return None
    vertical_entries, horizontal_entries = layout

    grid = np.full((h, w), bg, dtype=int)
    reserve = np.full((h, w), bg, dtype=int)
    reserve[0, :] = FRAME_COL
    reserve[-1, :] = FRAME_COL
    reserve[:, 0] = FRAME_COL
    reserve[:, -1] = FRAME_COL

    fr_lo = edge_band
    fr_hi = h - frame_h - edge_band
    fc_lo = edge_band
    fc_hi = w - frame_w - edge_band
    if fr_lo > fr_hi or fc_lo > fc_hi:
        return None
    fr0 = random.randint(fr_lo, fr_hi)
    fc0 = random.randint(fc_lo, fc_hi)
    paste(grid, input_frame, fr0, fc0)
    paste(reserve, np.full((frame_h, frame_w), FRAME_COL, dtype=int), fr0, fc0)

    for src, c0 in vertical_entries:
        top_patch = _top_source_patch(masks[src], src, bg)
        bottom_patch = _target_bar(n, color_to_tgt[src], vertical=True)
        paste(grid, top_patch, 0, c0, transparent=bg)
        paste(grid, bottom_patch, h - 1, c0)
        paste(reserve, np.full(top_patch.shape, FRAME_COL, dtype=int), 0, c0)
        paste(reserve, np.full(bottom_patch.shape, FRAME_COL, dtype=int), h - 1, c0)

    for src, r0 in horizontal_entries:
        left_patch = _left_source_patch(masks[src], src, bg)
        right_patch = _target_bar(n, color_to_tgt[src], vertical=False)
        paste(grid, left_patch, r0, 0, transparent=bg)
        paste(grid, right_patch, r0, w - 1)
        paste(reserve, np.full(left_patch.shape, FRAME_COL, dtype=int), r0, 0)
        paste(reserve, np.full(right_patch.shape, FRAME_COL, dtype=int), r0, w - 1)

    sprite_patches = [(_colored_sprite(masks[src], src, bg), src) for src in src_cols]
    random.shuffle(sprite_patches)
    for patch, _src in sprite_patches:
        if not _place_patch(grid, reserve, patch, bg):
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
