# Combo variant of edb79dae (index 20)
# combo_spec: D10/0+D2/2
# Name: 先重着色再放置+颜色映射图例

import itertools
import random

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, paste, random_free_location


# Design analysis for the combo variant
# -------------------------------------
# concretization:
# - Keep the original high-level scene split: one gray rectangular frame on a background canvas, plus
#   several external sprite templates.
# - Reverse the causal chain from the original task. The frame interior no longer stores source colors;
#   it stores final target colors. Each solid NxN block inside the frame says "a recolored sprite of this
#   target color belongs here".
# - Replace the old overlap-style legend with a fixed Kx4 color table at the upper-left corner. Each row
#   is [src, src, tgt, tgt]. The left half names the original sprite color, the right half names the
#   color after recoloring.
# - External loose sprites still use source colors, one sprite template per source color, on background.
#
# behavior trace:
# - Original task:
#     frame block color=3 -> find outside sprite color=3 -> place shape -> use legend to recolor to 7.
# - This combo:
#     legend row [3,3,7,7] -> source sprite color 3 is logically recolored to 7 ->
#     only then can it be matched to frame blocks colored 7 -> place that 7-colored sprite shape there.
# - The important difference is that frame matching is now keyed by the post-legend color, not by the
#   sprite's visible source color.
# - Small concrete trace:
#     loose sprite color 2 + legend row [2,2,8,8] + frame block color 8
#     -> recolor that sprite to 8 conceptually
#     -> then stamp the sprite shape into every 8-colored frame slot.
#   Under the original rule the first lookup would have started from the frame's color 8, which now fails
#   on purpose because no loose sprite of color 8 exists.
#
# complexity calibration:
# - Force 2-4 color pairs so src/tgt sets can be pairwise distinct and disjoint inside the 8 usable
#   non-bg, non-frame colors. This prevents ambiguity and makes the inversion obvious.
# - Every target color appears at least once inside the frame, and every source color has exactly one
#   outside sprite. Repeated placements reuse colors, so the lookup table has visible support.
# - Target colors are all distinct. Otherwise two different source sprites could collapse onto one target
#   placeholder color and the "recolor first, then place by new color" chain would become ambiguous.
# - Shape uniqueness is intentionally not required in this combo: the table removes shape-based legend
#   lookup, so the key constraint is color-pair uniqueness and target-color uniqueness.
#
# implicit assumptions audited:
# - The original solver assumed frame placeholder colors were the same keys used to find external sprites.
#   That fails here, so sprite lookup must be driven by legend inversion: target -> source.
# - The original legend detector searched for overlapping compact glyphs. That assumption is intentionally
#   broken. The new solver instead reads a top-left rectangular Kx4 color table.
# - The new solver assumes the top-left external non-bg object is exactly that legend table, its rows are
#   unique in the source column, and its target column matches the set of frame placeholder colors.
#   The generator enforces all three conditions.
# - The new solver also assumes each source color has exactly one sprite bbox matching the corresponding
#   placeholder size, possibly split into several 4-components inside the same bbox. The generator keeps
#   one sprite per source color and never places unrelated same-color patches outside.
# - The new solver additionally assumes target colors do not need a second-stage disambiguator. Generator
#   side we therefore forbid duplicate target rows and keep source/target color sets disjoint.
#
# placement legality / observability / cleanliness:
# - The legend is fixed at the upper-left corner with background around it, so humans can immediately read
#   it as a lookup table instead of mistaking it for another sprite.
# - The frame and all sprites are placed with reserve-mask padding, so no objects overlap or fuse.
# - Source colors do not appear inside the frame, and target colors do not appear on loose sprites. This
#   makes the reversed dependency visually recoverable from a few examples and blocks degeneration back to
#   the original "match by current frame color" behavior.
# - Because the frame holds 6-10 placements but only 2-4 target colors, every sample contains repeated
#   support for the color table instead of a one-off mapping that would be hard to infer from few examples.


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


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor outside sprites through a color table before placing them into the framed region.

    Input structure:
    - A background canvas contains one gray rectangular frame whose interior holds solid NxN target-color placeholders.
    - Outside the frame are source-colored sprite templates and, in the upper-left corner, a compact Kx4 color
      lookup table with rows of the form [src, src, tgt, tgt].

    Rule:
    - Find the frame and collect the placeholder blocks inside it; their colors are the post-recolor target colors.
    - Read the upper-left table to map each source sprite color to its new target color, then invert that mapping so
      each target placeholder color identifies which source-colored external sprite must be recolored first.
    - For every target-colored placeholder, place the corresponding sprite shape into that bbox and draw it using the
      target color rather than the sprite's original source color.

    Key insight: Matching cannot start from the loose sprite colors anymore, because the frame stores only the final
    colors. The legend must be used first to convert source colors into target colors, and only those target colors can
    be used to choose placement sites.
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
    for tgt_col in sorted(placeholder_colors):
        for pts in components_mask((crop == tgt_col) & inner_mask, connectivity=4):
            r0, c0, r1, c1 = bbox_coords(pts)
            placeholders.append((tgt_col, (r0, c0, r1, c1)))

    phw: dict[int, tuple[int, int]] = {}
    for tgt_col in placeholder_colors:
        dims = [(b[2] - b[0] + 1, b[3] - b[1] + 1) for c, b in placeholders if c == tgt_col]
        assert dims, f"no placeholder component found for color {tgt_col}"
        unique_dims = sorted(set(dims))
        assert len(unique_dims) == 1, f"ambiguous placeholder sizes for color {tgt_col}: {unique_dims}"
        phw[tgt_col] = unique_dims[0]

    outside = np.ones(g.shape, dtype=bool)
    outside[fr0 : fr1 + 1, fc0 : fc1 + 1] = False

    outside_fg = np.argwhere(outside & (g != bg) & (g != frame_col))
    assert outside_fg.size > 0, "no external legend/sprites found"
    lr0, lc0 = min((int(r), int(c)) for r, c in outside_fg)
    legend_h = len(placeholder_colors)
    legend_w = 4
    assert lr0 + legend_h <= g.shape[0] and lc0 + legend_w <= g.shape[1], "legend table would exceed canvas bounds"
    legend = g[lr0 : lr0 + legend_h, lc0 : lc0 + legend_w]
    assert legend.shape == (legend_h, legend_w), f"unexpected legend shape {legend.shape}"
    assert np.all(legend[:, 0] == legend[:, 1]), "legend left half must be uniform per row"
    assert np.all(legend[:, 2] == legend[:, 3]), "legend right half must be uniform per row"
    assert not np.any((legend == bg) | (legend == frame_col)), "legend cannot contain bg or frame color"

    src_rows = [int(x) for x in legend[:, 0]]
    tgt_rows = [int(x) for x in legend[:, 2]]
    assert len(set(src_rows)) == legend_h, "duplicate source rows in legend"
    assert len(set(tgt_rows)) == legend_h, "duplicate target rows in legend"
    assert set(tgt_rows) == placeholder_colors, (
        f"legend target colors {sorted(set(tgt_rows))} do not match placeholders {sorted(placeholder_colors)}"
    )
    source_to_target = {src: tgt for src, tgt in zip(src_rows, tgt_rows)}
    target_to_source = {tgt: src for src, tgt in source_to_target.items()}

    legend_mask = np.zeros(g.shape, dtype=bool)
    legend_mask[lr0 : lr0 + legend_h, lc0 : lc0 + legend_w] = True

    sprite_coords: dict[int, tuple[tuple[tuple[int, int], ...], int, int]] = {}
    for tgt_col in sorted(placeholder_colors):
        src_col = int(target_to_source[tgt_col])
        want_h, want_w = phw[tgt_col]
        all_candidates = []
        for pts in components_mask((g == src_col) & outside & ~legend_mask, connectivity=4):
            r0, c0, r1, c1 = bbox_coords(pts)
            all_candidates.append(
                {
                    "pts": [(int(r), int(c)) for r, c in pts],
                    "bbox": (r0, c0, r1, c1),
                }
            )
        assert all_candidates, f"no non-legend sprite found for source color {src_col}"

        sized = [
            comp
            for comp in all_candidates
            if (comp["bbox"][2] - comp["bbox"][0] + 1, comp["bbox"][3] - comp["bbox"][1] + 1) == (want_h, want_w)
        ]
        assert len(sized) == 1, f"expected one size-matched sprite for source color {src_col}, found {len(sized)}"
        base = sized[0]
        sr0, sc0, sr1, sc1 = base["bbox"]

        pts_set = set(base["pts"])
        for comp in all_candidates:
            if comp is base:
                continue
            r0, c0, r1, c1 = comp["bbox"]
            assert sr0 <= r0 and r1 <= sr1 and sc0 <= c0 and c1 <= sc1, (
                f"extra sprite component for source color {src_col} is outside the base bbox"
            )
            pts_set.update(comp["pts"])

        coords = tuple(sorted((r - sr0, c - sc0) for r, c in pts_set))
        sprite_coords[tgt_col] = (coords, sr1 - sr0 + 1, sc1 - sc0 + 1)

    out = crop.copy()
    out[1:-1, 1:-1] = bg

    for tgt_col, (pr0, pc0, pr1, pc1) in placeholders:
        coords, sh, sw = sprite_coords[int(tgt_col)]
        assert (pr1 - pr0 + 1, pc1 - pc0 + 1) == (sh, sw), (
            f"placeholder size does not match recolored sprite for target color {tgt_col}"
        )
        assert pr0 + sh <= H and pc0 + sw <= W, f"sprite for target color {tgt_col} does not fit at placeholder"
        for dr, dc in coords:
            out[pr0 + dr, pc0 + dc] = int(tgt_col)

    return out


# --------------- generator (o5x) ---------------

# === CFG: tunables ===
FRAME_COL = 5
SPRITE_N_RANGE = (3, 6)
GRID_ROWS_RANGE = (2, 4)
GRID_COLS_RANGE = (3, 5)
TOTAL_BLOCKS_RANGE = (6, 10)
MAX_EMPTY_SLOTS = 2
NUM_COLORS_RANGE = (2, 4)           # disjoint src/tgt sets require 2*K <= 8 usable colors
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


def _legend_table(pairs: list[tuple[int, int]]) -> np.ndarray:
    """Build a Kx4 color lookup table with rows [src, src, tgt, tgt]."""
    return np.array([[src, src, tgt, tgt] for src, tgt in pairs], dtype=int)


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
    """Choose empty grid slots while keeping every row and column represented."""
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


def _choose_disjoint_target_colors(src_cols: list[int], usable: list[int]) -> list[int] | None:
    """Pick pairwise distinct target colors disjoint from the source color set."""
    tgt_pool = [c for c in usable if c not in src_cols]
    if len(tgt_pool) < len(src_cols):
        return None
    return random.sample(tgt_pool, len(src_cols))


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
    tgt_cols = _choose_disjoint_target_colors(src_cols, usable)
    if tgt_cols is None:
        return None
    color_to_tgt = dict(zip(src_cols, tgt_cols))
    target_to_src = {tgt: src for src, tgt in color_to_tgt.items()}
    assert len(target_to_src) == k, "target colors must be unique"

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
        src = target_to_src[int(tgt)]
        paste(output_frame, _colored_sprite(masks[src], tgt, bg), r0, c0, transparent=bg)

    legend_pairs = list(zip(src_cols, tgt_cols))
    random.shuffle(legend_pairs)
    legend = _legend_table(legend_pairs)
    legend_h, legend_w = legend.shape

    ext_area = legend_h * legend_w + k * (n * n + 12) + 20
    possible_dims = [
        (h, w)
        for h in range(*CANVAS_H_RANGE)
        for w in range(*CANVAS_W_RANGE)
        if h >= max(frame_h, legend_h)
        and w >= max(frame_w, legend_w)
        and (((h - frame_h) >= legend_h + PATCH_PADDING) or ((w - frame_w) >= legend_w + PATCH_PADDING))
        and h * w - frame_h * frame_w >= ext_area
    ]
    if not possible_dims:
        return None
    h, w = random.choice(possible_dims)
    grid = np.full((h, w), bg, dtype=int)
    reserve = np.full((h, w), bg, dtype=int)

    paste(grid, legend, 0, 0)
    paste(reserve, np.full((legend_h, legend_w), FRAME_COL, dtype=int), 0, 0)

    frame_blocker = np.full((frame_h, frame_w), FRAME_COL, dtype=int)
    try:
        fr0, fc0 = random_free_location(reserve, frame_blocker, transparent=bg, padding=PATCH_PADDING)
    except ValueError:
        return None
    paste(grid, input_frame, fr0, fc0)
    paste(reserve, frame_blocker, fr0, fc0)

    patches = [(_colored_sprite(masks[src], src, bg), True) for src in src_cols]
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
