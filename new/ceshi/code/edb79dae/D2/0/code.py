# Variant of edb79dae
# Dimension: D2 — 第二步匹配条件：形状-颜色图例查找 (match_cond)
# Change: 放置后的精灵形状与左上角图例中的形状模式匹配（s2.shape == d2.shape），确定重着色的目标颜色... → 按面积/像素数量查找图例
#
# Design analysis
# spec:
#   core_change:
#     - Step 1 stays close to the selected baseline: placeholder color still chooses
#       which external sprite shape is stamped into the frame.
#     - Step 2 is replaced completely. The recolor legend no longer carries any
#       source-color cue and does not resemble the sprite shape.
#     - Each legend entry is a fixed 5x6 two-color card. A shared "code color"
#       fills exactly N cells inside the left 5x5 zone; the other color on that
#       same card is the repaint target. N equals the sprite's foreground area.
#   behavior_trace:
#     - Original selected-baseline flow:
#       1. find placeholder color c
#       2. fetch same-color sprite outside the frame
#       3. find legend mentioning c
#       4. recolor to legend target
#     - New D2 flow:
#       1. find placeholder color c
#       2. fetch same-color sprite outside the frame
#       3. count sprite pixels, say area = 11
#       4. find the legend card whose shared code color appears 11 times
#       5. recolor to that card's other color
#     - Concrete difference:
#       If a red placeholder maps to an 11-pixel sprite, and the 11-dot card is
#       paired with yellow, then every red placeholder becomes the red sprite shape
#       recolored to yellow even though red never appears in the legend.
#   complexity_calibration:
#     - To keep the new rule visible, each sample uses 2-3 placeholder colors and
#       forces all sprite areas to be distinct. Otherwise counting would not decide
#       anything.
#     - Target colors are chosen disjoint from placeholder colors, so the legend
#       cannot accidentally be solved by the old color-pair heuristic.
#     - The legend cards all share one common code color. This creates a visually
#       coherent "counting legend" and gives the solver a stable parsing cue.
#   implicit_assumptions_review:
#     - _find_frame still assumes one rectangular border component; generator keeps
#       exactly one such frame.
#     - Placeholder extraction still assumes one bbox size per placeholder color;
#       generator keeps one global NxN sprite size.
#     - Sprite recovery still depends on at least one connected component spanning
#       the full NxN bbox. The sprite generator preserves the original seed pattern
#       (full row + one cell in every row), so this remains true.
#     - New legend parsing assumes the only outside fully filled 5x6 rectangles are
#       legend cards; generator ensures sprites always contain background and legends
#       are the only opaque rectangles of that size.
#     - New recoloring assumes sprite areas are unique across source colors; the
#       generator samples distinct area counts and validates the output with main().
#   placement_legality:
#     - The frame and every external patch are separated by at least one background
#       cell using the existing reserve-grid placement method.
#     - Legend cards are opaque rectangles; sprites are pasted transparently over
#       background. This prevents overlap and keeps binary legend components intact.
#   visual_observability:
#     - The new cards do not encode shape or source color. What repeats across cards
#       is only the code color and the fixed 5x6 layout, so the human-visible cue is
#       clearly "count these cells, then use the paired swatch color".
#     - K is capped at 3 so the counting legend stays readable and target colors can
#       remain disjoint from source colors.
#   cleanliness_and_color_inference:
#     - Legends are all the same size and internal template, so they read as one
#       legend family instead of random clutter.
#     - No color-id ordering is used. Matching is driven by visible count equality,
#       and target colors are directly shown on the legend cards.

import itertools
import random

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, paste, random_free_location


LEGEND_CODE_H = 5
LEGEND_CODE_W = 5
LEGEND_H = LEGEND_CODE_H
LEGEND_W = LEGEND_CODE_W + 1
LEGEND_ORDER = tuple((r, c) for r in range(LEGEND_CODE_H) for c in range(LEGEND_CODE_W))


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


def _extract_sprite_coords(
    comps: list[dict], placeholder_colors: set[int], phw: dict[int, tuple[int, int]]
) -> dict[int, tuple[tuple[tuple[int, int], ...], int, int]]:
    """Recover each source sprite as coords relative to its bbox."""
    sprite_coords: dict[int, tuple[tuple[tuple[int, int], ...], int, int]] = {}
    for src_col in sorted(placeholder_colors):
        all_candidates = [c for c in comps if c["col"] == src_col]
        assert all_candidates, f"no sprite found for color {src_col}"
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
    return sprite_coords


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stamp recolored sprites into a framed region using placeholder positions and an area legend.

    Input structure:
    - A background color, one gray rectangular frame, and solid placeholder blocks inside the frame.
    - Outside the frame are sprite templates in placeholder colors and several compact 5x6 legend cards.
      Every legend card uses one shared code color plus one target color.

    Rule:
    - Find the frame and the placeholder blocks.
    - For each placeholder color, recover the same-color sprite outside the frame and stamp that shape at
      every matching placeholder location.
    - Count the sprite's foreground pixels.
    - Recolor the stamped sprite using the target color from the legend card whose code-color cell count
      equals that sprite area.

    Key insight: The legend no longer keys off source color or shape. The repeated code color is only a
    numeric marker, so solving the second stage requires counting pixels and matching that count to a card.
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

    sprite_coords = _extract_sprite_coords(comps, placeholder_colors, phw)
    sprite_areas = {src_col: len(coords) for src_col, (coords, _sh, _sw) in sprite_coords.items()}
    assert len(set(sprite_areas.values())) == len(sprite_areas), "sprite areas must be unique"

    legend_candidates = []
    dense_outside = outside & (g != bg) & (g != frame_col)
    for pts in components_mask(dense_outside, connectivity=4):
        r0, c0, r1, c1 = bbox_coords(pts)
        if (r1 - r0 + 1, c1 - c0 + 1) != (LEGEND_H, LEGEND_W):
            continue
        sub = g[r0 : r1 + 1, c0 : c1 + 1]
        if np.any(sub == bg) or np.any(sub == frame_col):
            continue
        colors_here = set(map(int, np.unique(sub)))
        if len(colors_here) != 2:
            continue
        legend_candidates.append({"bbox": (r0, c0, r1, c1), "colors": colors_here, "sub": sub.copy()})

    assert len(legend_candidates) == len(placeholder_colors), (
        f"expected {len(placeholder_colors)} legend cards, found {len(legend_candidates)}"
    )
    common_colors = set.intersection(*(cand["colors"] for cand in legend_candidates))
    assert len(common_colors) == 1, f"expected one shared legend code color, found {sorted(common_colors)}"
    code_col = int(next(iter(common_colors)))
    assert code_col not in placeholder_colors, "legend code color must not also be a placeholder color"

    area_to_tgt: dict[int, int] = {}
    for cand in legend_candidates:
        code_area = int(np.sum(cand["sub"] == code_col))
        tgt_colors = [c for c in cand["colors"] if c != code_col]
        assert len(tgt_colors) == 1
        tgt = int(tgt_colors[0])
        assert code_area not in area_to_tgt, f"duplicate legend count {code_area}"
        area_to_tgt[code_area] = tgt

    assert set(area_to_tgt) == set(sprite_areas.values()), (
        f"legend counts {sorted(area_to_tgt)} do not match sprite areas {sorted(set(sprite_areas.values()))}"
    )

    out = crop.copy()
    out[1:-1, 1:-1] = bg

    for src_col, (pr0, pc0, pr1, pc1) in placeholders:
        assert src_col in sprite_coords, f"placeholder color {src_col} missing sprite"
        coords, sh, sw = sprite_coords[src_col]
        assert (pr1 - pr0 + 1, pc1 - pc0 + 1) == (sh, sw), (
            f"placeholder size does not match sprite for color {src_col}"
        )
        sprite_area = len(coords)
        assert sprite_area in area_to_tgt, f"no legend entry for sprite area {sprite_area}"
        tgt_col = int(area_to_tgt[sprite_area])
        assert pr0 + sh <= H and pc0 + sw <= W, f"sprite for color {src_col} does not fit at placeholder"
        for dr, dc in coords:
            rr, cc = pr0 + dr, pc0 + dc
            out[rr, cc] = tgt_col

    return out


# --------------- generator (o5x-D2) ---------------

# === CFG: tunables ===
FRAME_COL = 5
SPRITE_N_RANGE = (3, 6)
GRID_ROWS_RANGE = (2, 4)
GRID_COLS_RANGE = (3, 5)
TOTAL_BLOCKS_RANGE = (6, 10)
MAX_EMPTY_SLOTS = 2
NUM_COLORS_RANGE = (2, 3)
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


def _legend(area: int, tgt: int, code_col: int) -> np.ndarray:
    """Build a fixed 5x6 legend card: code-color cell count encodes the sprite area."""
    assert 0 < area <= LEGEND_CODE_H * LEGEND_CODE_W
    patch = np.full((LEGEND_H, LEGEND_W), int(tgt), dtype=int)
    for r, c in LEGEND_ORDER[:area]:
        patch[r, c] = int(code_col)
    return patch


def _sprite_area_bounds(n: int) -> tuple[int, int]:
    lo = max((n * n + 1) // 2, int(np.ceil(SPRITE_FILL_MIN * n * n)))
    hi = min(int(np.floor(SPRITE_FILL_MAX * n * n)), n * n - 1)
    return lo, hi


def _sprite_mask(n: int, target: int) -> np.ndarray:
    """Generate a random NxN binary mask with exactly `target` filled cells."""
    lo, hi = _sprite_area_bounds(n)
    assert lo <= target <= hi, f"target area {target} outside [{lo}, {hi}]"

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
    """Choose empty slots while keeping every row and column represented."""
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
    n, rows, cols, empty_count, _total = struct

    bg = random.choice([c for c in range(10) if c != FRAME_COL])
    legend_code_col = random.choice([c for c in range(10) if c not in (bg, FRAME_COL)])
    role_palette = [c for c in range(10) if c not in (bg, FRAME_COL, legend_code_col)]

    lo, hi = _sprite_area_bounds(n)
    possible_areas = list(range(lo, hi + 1))
    max_k = min(NUM_COLORS_RANGE[1], len(possible_areas), len(role_palette) // 2)
    if max_k < NUM_COLORS_RANGE[0]:
        return None
    k = random.randint(NUM_COLORS_RANGE[0], max_k)

    src_cols = random.sample(role_palette, k)
    tgt_pool = [c for c in role_palette if c not in src_cols]
    tgt_cols = random.sample(tgt_pool, k)
    areas = random.sample(possible_areas, k)

    color_to_tgt = dict(zip(src_cols, tgt_cols))
    color_to_area = dict(zip(src_cols, areas))
    masks = {src: _sprite_mask(n, color_to_area[src]) for src in src_cols}

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

    ext_area = k * (LEGEND_H * LEGEND_W + n * n + 12)
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
        patches.append((_colored_sprite(masks[src], src, bg), True))
        patches.append((_legend(color_to_area[src], color_to_tgt[src], legend_code_col), False))
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
    raise RuntimeError("failed to generate valid edb79dae D2 input")
