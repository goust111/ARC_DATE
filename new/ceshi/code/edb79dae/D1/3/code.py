# Variant of edb79dae
# Dimension: D1 — 第一步匹配条件：颜色匹配 (match_cond)
# Change: 精灵颜色 == 框架内目标像素颜色（s.color == d.color），通过颜色一致性将外部精灵与内部单色像素配对... → 连接器匹配

import itertools
import random

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, paste, random_free_location


# ---------------------------------------------------------------------------
# Design analysis for this D1 variant
# ---------------------------------------------------------------------------
#
# a) Concrete behavior and trace
# - Original task: each solid placeholder block inside the frame was identified by
#   its own color, and the solver searched outside the frame for a sprite of that
#   same color.
# - Variant: the placeholder block color no longer chooses the sprite. Each
#   placeholder block has a tiny connector glyph directly above its top-left
#   corner. Every external sprite template also has a connector glyph directly
#   above its sprite box. The matched sprite is the one whose connector matches
#   both in color and in normalized shape.
# - The legend still maps placeholder-color -> output-color exactly as before, so
#   recoloring is still driven by the source-color legend, not by the connector.
# - Small trace:
#   old logic:
#     frame contains a 4x4 red block -> find outside red sprite -> stamp it ->
#     use legend red->yellow to recolor the stamped shape yellow.
#   new logic:
#     frame contains a 4x4 red block with, say, an orange T connector above it ->
#     ignore sprite body colors -> find the outside sprite whose connector is also
#     orange T -> stamp that sprite body into the red block location -> still use
#     legend red->yellow to recolor the stamped shape yellow.
#   difference:
#     the source color still controls recoloring, but no longer controls shape
#     selection. Shape selection is now a multi-attribute connector lookup.
#
# b) Complexity calibration
# - To make the connector rule observable, connector colors and connector shapes
#   are both reused:
#   * at least one connector color is shared by multiple sprite types
#   * at least one connector shape is shared by multiple sprite types
# - Therefore neither color-only nor shape-only matching is sufficient.
# - The connector vocabulary is intentionally small (three fixed glyphs) so that
#   human solvers can infer the rule from a few examples instead of facing a large
#   unconstrained glyph space.
#
# c) Old implicit assumptions that must change
# - The original `main()` collected `placeholder_colors` from all non-bg colors in
#   the frame interior. That is no longer valid because connector colors now also
#   appear inside the frame.
# - The original sprite search filtered outside components by `component.color ==
#   placeholder_color`. That assumption is the exact D1 property being changed, so
#   it must be removed.
# - The original generator could keep 1-cell row gaps because nothing occupied the
#   gaps. This variant needs a 2-row band above each placeholder row so every
#   block can carry a visible connector without colliding with the row above.
#
# d) Placement legality
# - Every placeholder block remains a solid NxN rectangle.
# - The connector above a placeholder always occupies the exact 2-row band
#   immediately above that block and is confined to the block's width.
# - Every external sprite patch has height N+2 and width N: 2 connector rows on
#   top, NxN sprite box below. This makes extraction deterministic for the solver.
# - The generator still uses `random_free_location` with padding so frame, legends,
#   and external sprite patches do not overlap.
#
# e) Visual observability
# - The new behavior is visible because the inside-frame blocks now carry small
#   colored connector glyphs, and the external sprites do too.
# - Sprite body colors are chosen from a pool disjoint from placeholder colors, so
#   the old color-match heuristic is visibly wrong and cannot accidentally explain
#   the examples.
#
# f) Visual cleanliness
# - Connectors live in dedicated 2-row bands above placeholders and above sprite
#   boxes rather than being mixed inside the bodies. This keeps the scene legible:
#   block, connector, sprite, and legend occupy distinct visual roles.
#
# g) Color-change inferability
# - Recoloring still follows the original legend mechanism. The legend is kept in
#   the same fixed 3x4 two-color form, so the only conceptual novelty is the
#   connector-based sprite lookup.
#
# h) Reference-task calibration (20270e3b)
# - The reference task also uses connector shape+placement as the decisive cue.
#   This variant borrows the same level of exactness for connector comparison:
#   connectors are matched by normalized occupied-cell pattern plus connector
#   color. The docking/translation behavior from the reference task is not
#   imported; only the connector-identification idea is reused.
#
# i) New solver assumptions and generator guarantees
# - Solver assumption: inside-frame placeholders are exactly the large solid
#   rectangles; generator guarantee: connector glyphs are smaller non-solid shapes.
# - Solver assumption: the 2-row band above each placeholder contains exactly one
#   connector glyph; generator guarantee: row gaps are 2 and only filled slots get
#   connectors.
# - Solver assumption: each connector key (color + normalized shape) identifies
#   exactly one sprite template; generator guarantee: connector pairs are unique
#   per placeholder color.
# - Solver assumption: beneath an external connector lies an NxN sprite box;
#   generator guarantee: every sprite patch is built with that exact geometry.


CONNECTOR_SHAPES: dict[str, tuple[tuple[int, int], ...]] = {
    "L": ((0, 0), (1, 0), (1, 1)),
    "J": ((0, 1), (1, 0), (1, 1)),
    "T": ((0, 1), (1, 0), (1, 1), (1, 2)),
}


def _shape_bbox(shape: tuple[tuple[int, int], ...]) -> tuple[int, int]:
    return max(r for r, _ in shape) + 1, max(c for _, c in shape) + 1


def _normalize_coords(coords: list[tuple[int, int]] | tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    r0 = min(r for r, _ in coords)
    c0 = min(c for _, c in coords)
    return tuple(sorted((int(r - r0), int(c - c0)) for r, c in coords))


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


def _find_placeholders(
    crop: np.ndarray, bg: int, frame_col: int
) -> tuple[list[tuple[int, tuple[int, int, int, int], tuple[int, tuple[tuple[int, int], ...]]]], dict[int, tuple[int, int]]]:
    H, W = crop.shape
    inner_mask = np.zeros((H, W), dtype=bool)
    inner_mask[1:-1, 1:-1] = True

    placeholders: list[tuple[int, tuple[int, int, int, int], tuple[int, tuple[tuple[int, int], ...]]]] = []
    for col in [int(c) for c in np.unique(crop) if int(c) not in (bg, frame_col)]:
        for pts in components_mask((crop == col) & inner_mask, connectivity=4):
            r0, c0, r1, c1 = bbox_coords(pts)
            h = r1 - r0 + 1
            w = c1 - c0 + 1
            sub = crop[r0 : r1 + 1, c0 : c1 + 1]
            if h < 3 or w < 3:
                continue
            if not np.all(sub == col):
                continue
            assert r0 >= 3, "placeholder too close to frame top for connector band"
            band = crop[r0 - 2 : r0, c0 : c1 + 1]
            band_pts = [(int(r), int(c)) for r, c in np.argwhere((band != bg) & (band != frame_col))]
            assert band_pts, f"missing connector above placeholder color {col}"
            band_colors = {int(band[r, c]) for r, c in band_pts}
            assert len(band_colors) == 1, f"expected one connector color above placeholder color {col}, found {band_colors}"
            conn_shape = _normalize_coords(band_pts)
            placeholders.append((int(col), (r0, c0, r1, c1), (int(next(iter(band_colors))), conn_shape)))

    assert placeholders, "no placeholders found"
    phw: dict[int, tuple[int, int]] = {}
    for col in sorted({src for src, _, _ in placeholders}):
        dims = [(b[2] - b[0] + 1, b[3] - b[1] + 1) for src, b, _ in placeholders if src == col]
        keys = [k for src, _, k in placeholders if src == col]
        unique_dims = sorted(set(dims))
        unique_keys = sorted(set(keys))
        assert len(unique_dims) == 1, f"ambiguous placeholder sizes for color {col}: {unique_dims}"
        assert len(unique_keys) == 1, f"ambiguous connector keys for placeholder color {col}: {unique_keys}"
        phw[col] = unique_dims[0]
    return placeholders, phw


def _bbox_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (ar1 < br0 or br1 < ar0 or ac1 < bc0 or bc1 < ac0)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stamp recolored sprites into a framed region using connector-marked placeholders and a color legend.

    Input structure:
    - Background, one rectangular frame, and inside the frame solid placeholder blocks.
    - Each placeholder has a tiny connector glyph directly above it; outside the frame there are sprite templates
      with their own connectors, plus the same small color legend used for output recoloring.

    Rule:
    - Find the frame, then detect the solid placeholder blocks and read the connector above each block.
    - Match each placeholder to the outside sprite whose connector has the same color and normalized shape.
    - Stamp only the sprite body into the placeholder position, then recolor that stamped body using the legend
      mapping keyed by the placeholder block color.

    Key insight: The placeholder color no longer chooses the sprite shape. The connector glyph is now the selection
    key, while the original legend still supplies the final recoloring.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)

    frame_col, (fr0, fc0, fr1, fc1) = _find_frame(g, bg)
    crop = g[fr0 : fr1 + 1, fc0 : fc1 + 1]
    H, W = crop.shape
    assert H >= 3 and W >= 3, f"frame too small for interior: {(H, W)}"

    placeholders, phw = _find_placeholders(crop, bg, frame_col)
    placeholder_colors = {src for src, _, _ in placeholders}
    key_for_src = {src: key for src, _, key in placeholders}
    assert len(key_for_src) == len(placeholder_colors), "placeholder connector keys must be unique per source color"
    key_to_src = {key: src for src, key in key_for_src.items()}
    assert len(key_to_src) == len(key_for_src), "connector key collision across source colors"

    outside = np.ones(g.shape, dtype=bool)
    outside[fr0 : fr1 + 1, fc0 : fc1 + 1] = False

    comps = []
    cid = 0
    for col in [int(c) for c in np.unique(g) if int(c) not in (bg, frame_col)]:
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
    legend_ids: set[int] = set()
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
            if (ci["col"] in placeholder_colors) ^ (cj["col"] in placeholder_colors):
                src = ci["col"] if ci["col"] in placeholder_colors else cj["col"]
                tgt = cj["col"] if src == ci["col"] else ci["col"]
            else:
                assert ci["area"] != cj["area"], f"ambiguous equal-area legend pair: {ci['col']} and {cj['col']}"
                if ci["area"] > cj["area"]:
                    src, tgt = ci["col"], cj["col"]
                else:
                    src, tgt = cj["col"], ci["col"]
            assert src in placeholder_colors, f"legend source color {src} is not a placeholder"
            assert src not in color_map, f"duplicate legend mapping for placeholder color {src}"
            color_map[src] = int(tgt)
            legend_ids.add(ci["id"])
            legend_ids.add(cj["id"])
    assert set(color_map) == placeholder_colors, (
        f"legend mappings {sorted(color_map)} do not match placeholders {sorted(placeholder_colors)}"
    )

    sprite_coords: dict[int, tuple[tuple[tuple[int, int], ...], int, int]] = {}
    for comp in comps:
        if comp["id"] in legend_ids:
            continue
        bh = comp["bbox"][2] - comp["bbox"][0] + 1
        bw = comp["bbox"][3] - comp["bbox"][1] + 1
        if bh != 2 or bw > 3:
            continue
        key = (int(comp["col"]), _normalize_coords(comp["pts"]))
        if key not in key_to_src:
            continue
        src = key_to_src[key]
        want_h, want_w = phw[src]
        cr0, cc0, _cr1, _cc1 = comp["bbox"]
        br0 = cr0 + 2
        bc0 = cc0
        assert br0 + want_h <= g.shape[0] and bc0 + want_w <= g.shape[1], f"sprite body for {src} is out of bounds"
        assert np.all(outside[br0 : br0 + want_h, bc0 : bc0 + want_w]), f"sprite body for {src} overlaps frame"
        body = g[br0 : br0 + want_h, bc0 : bc0 + want_w]
        body_pts = tuple(sorted((int(r), int(c)) for r, c in np.argwhere(body != bg)))
        assert body_pts, f"missing sprite body beneath connector for placeholder color {src}"
        assert src not in sprite_coords, f"duplicate sprite found for placeholder color {src}"
        sprite_coords[src] = (body_pts, want_h, want_w)
    assert set(sprite_coords) == placeholder_colors, (
        f"sprite keys {sorted(sprite_coords)} do not match placeholders {sorted(placeholder_colors)}"
    )

    out = crop.copy()
    out[1:-1, 1:-1] = bg
    for src_col, (pr0, pc0, pr1, pc1), _key in placeholders:
        tgt_col = int(color_map[src_col])
        coords, sh, sw = sprite_coords[src_col]
        assert (pr1 - pr0 + 1, pc1 - pc0 + 1) == (sh, sw), (
            f"placeholder size does not match sprite for color {src_col}"
        )
        for dr, dc in coords:
            out[pr0 + dr, pc0 + dc] = tgt_col
    return out


# --------------- generator ---------------

# === CFG: tunables ===
FRAME_COL = 5
SPRITE_N_RANGE = (3, 6)
GRID_ROWS_RANGE = (2, 4)
GRID_COLS_RANGE = (3, 5)
TOTAL_BLOCKS_RANGE = (6, 10)
MAX_EMPTY_SLOTS = 2
NUM_COLORS_RANGE = (3, 4)
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
GEN_RETRIES = 400

ROW_GAP = 2
COL_GAP = 1
TOP_MARGIN_MIN = 2
BOTTOM_MARGIN_MIN = 1
SIDE_MARGIN_MIN = 1


def _legend(src: int, tgt: int) -> np.ndarray:
    return np.array(
        [[src, src, tgt, tgt], [src, src, src, tgt], [src, src, tgt, tgt]],
        dtype=int,
    )


def _sprite_mask(n: int) -> np.ndarray:
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


def _sprite_patch(mask: np.ndarray, body_col: int, ctrl_col: int, shape: tuple[tuple[int, int], ...], bg: int) -> np.ndarray:
    n = mask.shape[0]
    patch = np.full((n + 2, n), bg, dtype=int)
    for r, c in shape:
        patch[r, c] = int(ctrl_col)
    paste(patch, _colored_sprite(mask, body_col, bg), 2, 0, transparent=bg)
    return patch


def _content_extent(n: int, grid_rows: int, grid_cols: int) -> tuple[int, int]:
    return (grid_rows * n + (grid_rows - 1) * ROW_GAP, grid_cols * n + (grid_cols - 1) * COL_GAP)


def _choose_structure() -> tuple[int, int, int, int, int] | None:
    total = random.randint(*TOTAL_BLOCKS_RANGE)
    options = []
    for n in range(*SPRITE_N_RANGE):
        for rows in range(*GRID_ROWS_RANGE):
            for cols in range(*GRID_COLS_RANGE):
                ch, cw = _content_extent(n, rows, cols)
                base_fh = ch + TOP_MARGIN_MIN + BOTTOM_MARGIN_MIN + 2
                base_fw = cw + 2 * SIDE_MARGIN_MIN + 2
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


def _sample_connector_keys(ctrl_cols: list[int], k: int) -> list[tuple[int, tuple[tuple[int, int], ...]]] | None:
    all_pairs = [(ctrl, CONNECTOR_SHAPES[name]) for ctrl in ctrl_cols for name in sorted(CONNECTOR_SHAPES)]
    for _ in range(200):
        chosen = random.sample(all_pairs, k)
        cols = [ctrl for ctrl, _ in chosen]
        shapes = [shape for _, shape in chosen]
        if len(set(cols)) == len(cols) and k > len(ctrl_cols):
            continue
        if len(set(cols)) == k:
            continue
        if len(set(shapes)) == k:
            continue
        return chosen
    return None


def _gen_attempt() -> np.ndarray | None:
    struct = _choose_structure()
    if struct is None:
        return None
    n, rows, cols, empty_count, _total = struct

    bg = random.choice([c for c in range(10) if c != FRAME_COL])
    usable = [c for c in range(10) if c not in (bg, FRAME_COL)]

    k = random.randint(*NUM_COLORS_RANGE)
    src_cols = random.sample(usable, k)
    ctrl_pool = [c for c in usable if c not in src_cols]
    if len(ctrl_pool) < 2:
        return None
    ctrl_cols = random.sample(ctrl_pool, 2)
    conn_keys = _sample_connector_keys(ctrl_cols, k)
    if conn_keys is None:
        return None

    body_pool = [c for c in usable if c not in src_cols and c not in ctrl_cols]
    if not body_pool:
        return None
    body_cols = {src: random.choice(body_pool) for src in src_cols}

    tgt_cols = [random.choice([c for c in usable if c != src]) for src in src_cols]
    color_to_tgt = dict(zip(src_cols, tgt_cols))
    src_to_key = dict(zip(src_cols, conn_keys))
    masks = {src: _sprite_mask(n) for src in src_cols}

    ch, cw = _content_extent(n, rows, cols)
    min_fh = max(MIN_FRAME_H, ch + TOP_MARGIN_MIN + BOTTOM_MARGIN_MIN + 2)
    min_fw = max(MIN_FRAME_W, cw + 2 * SIDE_MARGIN_MIN + 2)
    max_fh = min(MAX_FRAME_DIM, max(min_fh, ch + TOP_MARGIN_MIN + BOTTOM_MARGIN_MIN + FRAME_MARGIN_SLACK + 2))
    max_fw = min(MAX_FRAME_DIM, max(min_fw, cw + 2 * SIDE_MARGIN_MIN + FRAME_MARGIN_SLACK + 2))
    if min_fh > max_fh or min_fw > max_fw:
        return None
    frame_h = random.randint(min_fh, max_fh)
    frame_w = random.randint(min_fw, max_fw)
    y_margin = frame_h - 2 - ch
    x_margin = frame_w - 2 - cw
    if y_margin < TOP_MARGIN_MIN + BOTTOM_MARGIN_MIN or x_margin < 2 * SIDE_MARGIN_MIN:
        return None
    top_margin = random.randint(TOP_MARGIN_MIN, y_margin - BOTTOM_MARGIN_MIN)
    left_margin = random.randint(SIDE_MARGIN_MIN, x_margin - SIDE_MARGIN_MIN)

    _empty_slots, filled_slots = _choose_empty_slots(rows, cols, empty_count)
    cell_colors: dict[tuple[int, int], int] = {}
    for slot, col in zip(filled_slots[:k], src_cols):
        cell_colors[slot] = col
    for slot in filled_slots[k:]:
        cell_colors[slot] = random.choice(src_cols)

    input_frame = _make_bordered_frame(frame_h, frame_w, bg)
    output_frame = _make_bordered_frame(frame_h, frame_w, bg)
    for (rr, cc), src in cell_colors.items():
        r0 = 1 + top_margin + rr * (n + ROW_GAP)
        c0 = 1 + left_margin + cc * (n + COL_GAP)
        paste(input_frame, np.full((n, n), src, dtype=int), r0, c0)
        ctrl_col, shape = src_to_key[src]
        for dr, dc in shape:
            input_frame[r0 - 2 + dr, c0 + dc] = int(ctrl_col)
        paste(output_frame, _colored_sprite(masks[src], color_to_tgt[src], bg), r0, c0, transparent=bg)

    ext_area = k * (12 + n * (n + 2) + 10)
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

    fr0 = random.choice([0, h - frame_h])
    fc0 = random.choice([0, w - frame_w])
    if h - frame_h > 2 and random.random() < FRAME_POS_JITTER_PROB:
        fr0 = random.randrange(0, h - frame_h + 1)
    if w - frame_w > 2 and random.random() < FRAME_POS_JITTER_PROB:
        fc0 = random.randrange(0, w - frame_w + 1)
    paste(grid, input_frame, fr0, fc0)
    paste(reserve, np.full((frame_h, frame_w), FRAME_COL, dtype=int), fr0, fc0)

    patches: list[tuple[np.ndarray, bool]] = []
    for src in src_cols:
        ctrl_col, shape = src_to_key[src]
        patches.append((_sprite_patch(masks[src], body_cols[src], ctrl_col, shape, bg), True))
        patches.append((_legend(src, color_to_tgt[src]), False))
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
    raise RuntimeError("failed to generate valid edb79dae input")
