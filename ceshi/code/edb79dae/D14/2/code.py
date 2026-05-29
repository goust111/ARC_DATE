# Variant of edb79dae
# Dimension: D14 — 精灵放置时的对齐方式 (meta)
# Change: 精灵形状以目标像素位置为中心放置，替换该像素及周围区域... → 精灵以锚点像素对齐（锚点由特殊颜色标记）
#
# Design analysis
# spec:
#   preserved_core:
#     - Keep the single rectangular frame, one external sprite template per source color,
#       and one external 3x4 two-color legend per source color.
#     - Keep the legend semantics: source color selects the sprite shape; legend maps that
#       source color to the output color.
#   D14_change:
#     - Inside the frame, placeholders are no longer solid NxN blocks. Each placement site is
#       now a single colored target pixel.
#     - Each external sprite contains exactly one special anchor pixel in the frame color (5).
#       When a sprite is placed, that anchor pixel must land on the matching-color target pixel.
#     - After placement, the full sprite footprint, including the anchor cell, is recolored to
#       the legend-mapped target color.
#   concrete_behavior_trace:
#     - Old behavior in the original code: a color-3 placeholder occupying a whole 4x4 box was
#       replaced by the color-3 sprite by aligning sprite bbox top-left to placeholder bbox
#       top-left. Internal structure did not affect placement.
#     - New behavior here: the color-3 placeholder is a single pixel at (r,c) inside a reserved
#       4x4 slot; the external color-3 sprite has one gray anchor at offset (1,3). Placement
#       uses top-left = (r-1, c-3), so the same sprite can extend asymmetrically around the
#       target pixel. Two sprites of the same bbox size can now land differently if their anchor
#       offsets differ.
#   complexity_calibration:
#     - The anchor must be visually inferable from 2-4 examples, so the generator forces the
#       external sprite to visibly contain one gray cell and the internal placeholder to be only
#       one colored pixel.
#     - To avoid ambiguity, each sprite body remains 4-connected after removing the anchor, and
#       its source-colored body still spans every row/column of its bbox. This makes sprite
#       extraction from the input reliable.
#     - Anchor offsets vary across colors whenever possible, and at least one non-central anchor
#       is required. Otherwise the variant would collapse back toward center/bbox alignment.
#   implicit_assumption_audit:
#     - Original placeholder extraction assumed connected solid blocks and recorded bboxes. That
#       is invalid now; the solver must read individual colored pixels as targets.
#     - Original sprite matching relied on placeholder bbox size to find the correct external
#       sprite. That also breaks now because placeholders are 1x1. The new solver instead finds
#       the unique non-legend same-color component whose bbox contains exactly one gray anchor
#       and only {bg, src, gray}.
#     - Original stamping assumed offset = placeholder bbox top-left. The new stamping must use
#       offset = target_pixel - anchor_offset.
#   placement_legality:
#     - Generation still reserves a regular grid of NxN slots with 1-cell gaps inside the frame.
#       Each target pixel is placed inside its slot at the sprite's anchor offset, so the final
#       placed sprite remains inside that slot and cannot overlap neighboring slots.
#     - Outside patches are still placed with background padding, so sprites, legends, and the
#       frame stay disjoint and solver parsing stays clean.
#   visual_cleanliness_and_inference:
#     - Input remains readable: sparse single-pixel targets inside the frame, dense sprite
#       templates outside, and compact legend glyphs separate from both.
#     - Color changes remain inferable from the same legend structure as the base task; D14 only
#       changes where the sprite lands, not how recoloring is decided.

import itertools
import random

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, paste, random_free_location


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


def main(grid: np.ndarray) -> np.ndarray:
    """
    Place recolored sprites into a framed region by matching anchor-marked templates to target pixels.

    Input structure:
    - A background grid with one rectangular frame and single-pixel colored targets inside the frame.
    - Outside the frame: one sprite template per target color, where each sprite includes exactly one
      gray anchor pixel, plus a small two-color legend that maps source colors to output colors.

    Rule:
    - Find all colored target pixels in the frame interior.
    - For each target color, locate the matching-color sprite outside the frame and recover its gray
      anchor position, then align that anchor to every same-color target pixel.
    - Recolor every placed sprite to the legend-defined output color while preserving the frame border
      and the background holes inside each sprite bbox.

    Key insight: The outside sprite is not aligned by its bbox anymore; the lone gray pixel inside the
    sprite acts as an anchor, and the legend still provides the final recoloring.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)

    frame_col, (fr0, fc0, fr1, fc1) = _find_frame(g, bg)

    crop = g[fr0 : fr1 + 1, fc0 : fc1 + 1]
    H, W = crop.shape
    assert H >= 3 and W >= 3, f"frame too small for interior: {(H, W)}"

    inner_mask = np.zeros((H, W), dtype=bool)
    inner_mask[1:-1, 1:-1] = True
    target_mask = inner_mask & (crop != bg) & (crop != frame_col)
    target_pts = [(int(crop[r, c]), int(r), int(c)) for r, c in np.argwhere(target_mask)]
    placeholder_colors = {col for col, _, _ in target_pts}
    assert target_pts, "no colored target pixels found inside frame"

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
            if bg in sub_colors:
                continue
            if len(sub_colors) != 2:
                continue
            cset = set(sub_colors)
            if ci["col"] not in cset or cj["col"] not in cset:
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

            assert src in placeholder_colors, f"legend source color {src} is not a target color"
            assert src not in color_map, f"duplicate legend mapping for target color {src}"
            color_map[src] = int(tgt)
            legend_ids.add(ci["id"])
            legend_ids.add(cj["id"])
    assert set(color_map) == placeholder_colors, (
        f"legend mappings {sorted(color_map)} do not match targets {sorted(placeholder_colors)}"
    )

    sprite_data: dict[int, tuple[tuple[tuple[int, int], ...], tuple[int, int], tuple[int, int]]] = {}
    for src_col in sorted(placeholder_colors):
        candidates = [c for c in comps if c["col"] == src_col and c["id"] not in legend_ids]
        assert len(candidates) == 1, f"expected one external sprite for color {src_col}, found {len(candidates)}"
        base = candidates[0]
        sr0, sc0, sr1, sc1 = base["bbox"]
        sub = g[sr0 : sr1 + 1, sc0 : sc1 + 1]
        sub_colors = set(map(int, np.unique(sub)))
        assert sub_colors <= {bg, frame_col, src_col}, f"sprite bbox for color {src_col} contains extra colors"
        anchor_pts = np.argwhere(sub == frame_col)
        assert anchor_pts.shape[0] == 1, f"sprite for color {src_col} must contain exactly one anchor pixel"
        coords = tuple(sorted((int(r), int(c)) for r, c in np.argwhere(sub != bg)))
        assert coords, f"sprite for color {src_col} is empty"
        anchor = tuple(map(int, anchor_pts[0]))
        sprite_data[src_col] = (coords, anchor, (sr1 - sr0 + 1, sc1 - sc0 + 1))

    out = crop.copy()
    out[1:-1, 1:-1] = bg

    for src_col, pr, pc in target_pts:
        assert src_col in sprite_data and src_col in color_map, f"target color {src_col} missing sprite or legend"
        coords, anchor, _ = sprite_data[src_col]
        ar, ac = anchor
        tgt_col = int(color_map[src_col])
        for dr, dc in coords:
            rr = pr + dr - ar
            cc = pc + dc - ac
            assert 1 <= rr < H - 1 and 1 <= cc < W - 1, f"placed sprite for color {src_col} leaves frame interior"
            assert out[rr, cc] in (bg, tgt_col), f"sprite overlap conflict at {(rr, cc)}"
            out[rr, cc] = tgt_col

    return out


# --------------- generator (o5x) ---------------

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


def _legend(src: int, tgt: int) -> np.ndarray:
    """Build a 3x4 legend block mapping src color to tgt color."""
    return np.array(
        [[src, src, tgt, tgt], [src, src, src, tgt], [src, src, tgt, tgt]],
        dtype=int,
    )


def _is_4_connected(mask: np.ndarray) -> bool:
    """Check whether all True cells belong to one 4-connected component."""
    if not mask.any():
        return False
    return len(components_mask(mask, connectivity=4)) == 1


def _sprite_occupancy(n: int) -> np.ndarray | None:
    """Generate a connected NxN sprite footprint that spans every row and column."""
    lo = max((n * n + 1) // 2, int(np.ceil(SPRITE_FILL_MIN * n * n)))
    hi = min(int(np.floor(SPRITE_FILL_MAX * n * n)), n * n)
    if hi < lo:
        return None
    target = random.randint(lo, hi)

    for _ in range(80):
        mask = np.ones((n, n), dtype=bool)
        cells = [(r, c) for r in range(n) for c in range(n)]
        random.shuffle(cells)
        for r, c in cells:
            if int(mask.sum()) <= target:
                break
            trial = mask.copy()
            trial[r, c] = False
            if _is_4_connected(trial) and np.all(trial.any(axis=0)) and np.all(trial.any(axis=1)):
                mask = trial
        if int(mask.sum()) == target and _is_4_connected(mask) and np.all(mask.any(axis=0)) and np.all(mask.any(axis=1)):
            return mask
    return None


def _anchor_candidates(mask: np.ndarray) -> list[tuple[int, int]]:
    """Return occupied cells whose removal keeps the source-colored body connected and full-span."""
    n, _ = mask.shape
    center = (n // 2, n // 2)
    preferred = []
    fallback = []
    for r, c in zip(*np.where(mask), strict=False):
        body = mask.copy()
        body[r, c] = False
        if not body.any():
            continue
        if not (_is_4_connected(body) and np.all(body.any(axis=0)) and np.all(body.any(axis=1))):
            continue
        cell = (int(r), int(c))
        if cell != center:
            preferred.append(cell)
        else:
            fallback.append(cell)
    return preferred + fallback


def _colored_sprite(mask: np.ndarray, color: int, anchor: tuple[int, int], bg: int, anchor_col: int) -> np.ndarray:
    """Render an occupied mask as a source-colored sprite with one special anchor pixel."""
    out = np.full(mask.shape, bg, dtype=int)
    out[mask] = int(color)
    ar, ac = anchor
    out[ar, ac] = int(anchor_col)
    return out


def _colored_output(mask: np.ndarray, color: int, bg: int) -> np.ndarray:
    """Render the final recolored sprite footprint."""
    out = np.full(mask.shape, bg, dtype=int)
    out[mask] = int(color)
    return out


def _content_extent(n: int, grid_rows: int, grid_cols: int) -> tuple[int, int]:
    """Content area needed for grid_rows x grid_cols of NxN slots with 1-cell gaps."""
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
    """Choose which grid slots are empty, ensuring every row and column has at least one filled slot."""
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
    """Create an h x w frame filled with bg and a 1-cell FRAME_COL border."""
    frame = np.full((h, w), bg, dtype=int)
    frame[0, :] = frame[-1, :] = FRAME_COL
    frame[:, 0] = frame[:, -1] = FRAME_COL
    return frame


def _place_patch(
    grid: np.ndarray, reserve: np.ndarray, patch: np.ndarray, bg: int, *, transparent: bool
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

    sprite_masks: dict[int, np.ndarray] = {}
    anchor_by_color: dict[int, tuple[int, int]] = {}
    used_anchors: set[tuple[int, int]] = set()
    for src in src_cols:
        mask = _sprite_occupancy(n)
        if mask is None:
            return None
        candidates = _anchor_candidates(mask)
        if not candidates:
            return None
        fresh = [a for a in candidates if a not in used_anchors]
        if fresh:
            anchor = random.choice(fresh)
        else:
            anchor = random.choice(candidates)
        sprite_masks[src] = mask
        anchor_by_color[src] = anchor
        used_anchors.add(anchor)

    if len(set(anchor_by_color.values())) < min(2, k):
        return None
    if all(anchor == (n // 2, n // 2) for anchor in anchor_by_color.values()):
        return None

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
    for slot, col in zip(filled_slots[:k], src_cols, strict=False):
        cell_colors[slot] = col
    for slot in filled_slots[k:]:
        cell_colors[slot] = random.choice(src_cols)

    input_frame = _make_bordered_frame(frame_h, frame_w, bg)
    output_frame = _make_bordered_frame(frame_h, frame_w, bg)
    for (rr, cc), src in cell_colors.items():
        r0 = 1 + top_margin + rr * (n + 1)
        c0 = 1 + left_margin + cc * (n + 1)
        ar, ac = anchor_by_color[src]
        input_frame[r0 + ar, c0 + ac] = src
        paste(output_frame, _colored_output(sprite_masks[src], color_to_tgt[src], bg), r0, c0, transparent=bg)

    ext_area = k * (12 + n * n + 10)
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
    for src, tgt in zip(src_cols, tgt_cols, strict=False):
        patches.append((_colored_sprite(sprite_masks[src], src, anchor_by_color[src], bg, FRAME_COL), True))
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
    raise RuntimeError("failed to generate valid edb79dae D14 input")
