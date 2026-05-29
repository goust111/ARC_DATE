# Variant of edb79dae
# Dimension: D10 — 多步处理顺序 (multi_step)
# Change: 三步流程：(1)识别框架和图例→(2)按颜色匹配精灵到框架内像素并放置形状→(3)按图例重着色所有放置的精灵... → 先重着色再放置

import itertools
import random

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, paste, random_free_location


# Design analysis for the D10 variant
# -----------------------------------
# concrete variant:
# - The frame still contains a regular grid of solid NxN placeholder blocks.
# - In the original task those blocks used the source/sprite colors, so the solver could place first
#   by color match and recolor later.
# - Here the frame blocks use the final/target colors instead. A direct color match to the outside
#   sprites is therefore impossible.
# - Each outside legend is changed from a color-color glyph into an NxN two-color square:
#     target target target ...
#     target + source-colored sprite mask painted on top
#   The larger-area color draws the exact sprite silhouette; the smaller-area color is the target.
# - The outside loose sprite still uses its source color on background.
#
# behavior trace:
# - original chain:
#     frame block color=3 -> find outside sprite color=3 -> stamp shape -> recolor via legend 3->7
# - new chain:
#     outside sprite shape/color=3 -> find legend with same shape -> infer target color 7 ->
#     recolor sprite to 7 -> place it into every frame block colored 7
# - This changes the causal dependency: placement now depends on the recolored result, not on the
#   original sprite color.
#
# complexity calibration:
# - Target colors are pairwise distinct and disjoint from source colors, so the new matching stage is
#   visible and unambiguous.
# - Sprite masks are also pairwise distinct, so a legend must be read by shape rather than by color id.
# - The sprite fill ratio is forced above 50%, making the source-colored legend shape strictly larger
#   than the target-colored remainder; this keeps legend parsing deterministic.
#
# implicit assumptions audited:
# - Original code assumed placeholder colors == outside sprite colors. That no longer holds, so all
#   sprite lookup code must switch from color-keyed matching to shape-keyed recolor + target-keyed placement.
# - Original legend decoding used placeholder membership / area heuristics. Here the legend source is
#   identified as the larger area inside a full rectangle, and the target is the smaller-area complement.
# - New solver assumes each target color maps to exactly one sprite shape, each legend square is a full
#   non-bg rectangle with exactly two colors, and each sprite shape appears once outside the frame.
#   The generator enforces all three conditions.
#
# placement legality / visual clarity:
# - Outside patches are still separated by background with padding via random_free_location.
# - Legend squares have no internal background, so they stand out as compact lookup keys.
# - Frame placeholders remain a clean grid with 1-cell gaps, making the recolor-then-place rule
#   visually recoverable from a few examples.


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


def _shape_signature(points: list[tuple[int, int]] | np.ndarray, r0: int, c0: int) -> tuple[tuple[int, int], ...]:
    return tuple(sorted((int(r) - int(r0), int(c) - int(c0)) for r, c in points))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor outside sprites from shape-coded legends, then stamp them into a framed region by final color.

    Input structure:
    - bg background, a rectangular single-color frame, and solid placeholder blocks inside the frame.
    - Outside the frame: source-colored sprite shapes and compact two-color legend squares whose larger color
      draws the same sprite silhouette while the smaller color indicates the destination color.

    Rule:
    - Find the frame and the target-colored placeholder blocks inside it.
    - Read each legend square to map sprite shape -> target color, then use that legend to logically recolor
      the matching outside sprite before any placement step.
    - Place each recolored sprite into every same-target-color placeholder location inside the frame.

    Key insight: The frame no longer stores source colors, so color matching cannot start the process; the
    legend's shape match must be resolved first, and only the recolored result can be used to choose placement sites.
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

    outside = np.ones(g.shape, dtype=bool)
    outside[fr0 : fr1 + 1, fc0 : fc1 + 1] = False

    # Legend squares are full non-bg rectangles outside the frame with exactly two colors.
    legend_cells = np.zeros(g.shape, dtype=bool)
    shape_to_target: dict[tuple[tuple[tuple[int, int], ...], int, int], int] = {}
    for pts in components_mask((g != bg) & outside, connectivity=4):
        r0, c0, r1, c1 = bbox_coords(pts)
        sub = g[r0 : r1 + 1, c0 : c1 + 1]
        h, w = sub.shape
        if len(pts) != h * w:
            continue
        cols = sorted(set(map(int, np.unique(sub))) - {bg, frame_col})
        if len(cols) != 2:
            continue
        areas = {col: int(np.sum(sub == col)) for col in cols}
        ordered = sorted(cols, key=lambda col: areas[col], reverse=True)
        src_col, tgt_col = ordered
        assert areas[src_col] > areas[tgt_col], "legend source/target areas must be distinguishable"
        sig = _shape_signature(np.argwhere(sub == src_col), 0, 0)
        key = (sig, h, w)
        assert key not in shape_to_target, f"duplicate legend shape detected for bbox {(r0, c0, r1, c1)}"
        shape_to_target[key] = int(tgt_col)
        legend_cells[r0 : r1 + 1, c0 : c1 + 1] = True

    assert shape_to_target, "no shape-coded legends found outside the frame"

    # External source sprites are the remaining outside colored components.
    sprite_candidates: dict[tuple[tuple[tuple[int, int], ...], int, int], tuple[tuple[tuple[int, int], ...], int, int]] = {}
    seen_targets: set[int] = set()
    colors = [int(c) for c in np.unique(g) if int(c) not in (bg, frame_col)]
    for col in colors:
        for pts in components_mask((g == col) & outside & ~legend_cells, connectivity=4):
            r0, c0, r1, c1 = bbox_coords(pts)
            h, w = r1 - r0 + 1, c1 - c0 + 1
            sig = _shape_signature(pts, r0, c0)
            key = (sig, h, w)
            if key not in shape_to_target:
                continue
            tgt = int(shape_to_target[key])
            assert tgt not in seen_targets, f"multiple sprites matched target color {tgt}"
            sprite_candidates[key] = (sig, h, w)
            seen_targets.add(tgt)

    target_to_sprite: dict[int, tuple[tuple[tuple[int, int], ...], int, int]] = {}
    for key, tgt in shape_to_target.items():
        assert key in sprite_candidates, f"no outside sprite matched legend shape for target color {tgt}"
        target_to_sprite[int(tgt)] = sprite_candidates[key]

    assert set(target_to_sprite) == placeholder_colors, (
        f"recolored sprite targets {sorted(target_to_sprite)} do not match placeholders {sorted(placeholder_colors)}"
    )

    out = crop.copy()
    out[1:-1, 1:-1] = bg

    for tgt_col, (pr0, pc0, pr1, pc1) in placeholders:
        coords, sh, sw = target_to_sprite[int(tgt_col)]
        assert (pr1 - pr0 + 1, pc1 - pc0 + 1) == (sh, sw), (
            f"placeholder size does not match recolored sprite for target color {tgt_col}"
        )
        off_r = pr0
        off_c = pc0
        for dr, dc in coords:
            out[off_r + dr, off_c + dc] = int(tgt_col)

    return out


# --------------- generator (o5x) ---------------

# === CFG: tunables ===
FRAME_COL = 5
SPRITE_N_RANGE = (3, 6)
GRID_ROWS_RANGE = (2, 4)
GRID_COLS_RANGE = (3, 5)
TOTAL_BLOCKS_RANGE = (6, 10)
MAX_EMPTY_SLOTS = 2
NUM_COLORS_RANGE = (2, 4)           # capped so src/tgt sets can be distinct and disjoint
SPRITE_FILL_MIN = 0.56              # strict majority so legend shape area > target remainder
SPRITE_FILL_MAX = 0.82
MIN_FRAME_H = 13
MIN_FRAME_W = 18
MAX_FRAME_DIM = 21
FRAME_MARGIN_SLACK = 4
CANVAS_H_RANGE = (22, 31)
CANVAS_W_RANGE = (23, 31)
FRAME_POS_JITTER_PROB = 0.25
PATCH_PADDING = 1
GEN_RETRIES = 300


def _legend_patch(mask: np.ndarray, src: int, tgt: int) -> np.ndarray:
    """Build an NxN shape legend: target-color base with the source-colored sprite silhouette on top."""
    out = np.full(mask.shape, int(tgt), dtype=int)
    out[mask] = int(src)
    return out


def _sprite_mask(n: int) -> np.ndarray:
    """Generate a random connected-ish NxN binary mask with a strict area majority."""
    lo = max(n * n // 2 + 1, int(np.ceil(SPRITE_FILL_MIN * n * n)))
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
    if len(usable) < 2 * k:
        return None
    src_cols = random.sample(usable, k)
    tgt_pool = [c for c in usable if c not in src_cols]
    tgt_cols = random.sample(tgt_pool, k)
    color_to_tgt = dict(zip(src_cols, tgt_cols))

    masks: dict[int, np.ndarray] = {}
    seen_sigs: set[tuple[tuple[int, int], ...]] = set()
    for src in src_cols:
        for _ in range(100):
            mask = _sprite_mask(n)
            sig = tuple(sorted(map(tuple, np.argwhere(mask))))
            if sig not in seen_sigs:
                seen_sigs.add(sig)
                masks[src] = mask
                break
        else:
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
    for slot, src in zip(filled_slots[:k], src_cols):
        cell_colors[slot] = color_to_tgt[src]
    for slot in filled_slots[k:]:
        src = random.choice(src_cols)
        cell_colors[slot] = color_to_tgt[src]

    input_frame = _make_bordered_frame(frame_h, frame_w, bg)
    output_frame = _make_bordered_frame(frame_h, frame_w, bg)
    for (rr, cc), tgt in cell_colors.items():
        r0 = 1 + top_margin + rr * (n + 1)
        c0 = 1 + left_margin + cc * (n + 1)
        paste(input_frame, np.full((n, n), int(tgt), dtype=int), r0, c0)
        src = next(src for src, mapped in color_to_tgt.items() if mapped == tgt)
        paste(output_frame, _colored_sprite(masks[src], tgt, bg), r0, c0, transparent=bg)

    ext_area = k * (2 * n * n + 8)
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
    for src, tgt in zip(src_cols, tgt_cols):
        patches.append((_colored_sprite(masks[src], src, bg), True))
        patches.append((_legend_patch(masks[src], src, tgt), False))
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
    raise RuntimeError("failed to generate valid edb79dae D10 variant input")
