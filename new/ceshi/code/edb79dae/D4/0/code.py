# Variant of edb79dae
# Dimension: D4 — 目标位置：框架内的匹配点 (dst)
# Change: 灰色框架内部的单色像素点，每个点的颜色指示应匹配哪个外部精灵... → 目标为框架内的小型空洞区域

import itertools
import random

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, paste, random_free_location


# Design analysis for the D4 variant
# ----------------------------------
# spec:
#   preserved backbone:
#   - Keep the original three-stage story: a gray frame contains the destination area,
#     outside the frame sit one sprite per source color plus one legend per source color,
#     and the output is still the cropped frame only.
#   - Keep legend semantics unchanged: the 3x4 two-color glyph outside the frame maps
#     source sprite color -> final recolor.
#   concretized D4 change:
#   - Original inner targets were solid colored blocks; source color could be read
#     directly from the destination itself.
#   - Variant inner targets are rectangular bg cavities cut into the gray framed panel.
#     A sprite is selected by exact bbox size, then recolored via the legend.
#   - To make size-matching observable, each source color gets a unique bbox size
#     chosen from a small visible pool such as 2x3, 3x4, 4x2, 4x4. The same color
#     can repeat across multiple cavities, so the size clue is repeated within a sample.
#   behavior trace:
#   - Original step: a red 3x3 placeholder block inside the frame directly says
#     "place the red source sprite here".
#   - Variant step: a 3x4 bg hole inside the frame says only "some 3x4 sprite fits here".
#     The solver must inspect outside sprites, find the unique source color whose sprite
#     bbox is 3x4, then use that source color's legend pair to recolor the placed piece.
#   complexity calibration:
#   - This borrows the cavity-fit idea from 5dbc8537, but stays simpler than full exact
#     cover: the frame already provides separated cavities, so no search is needed once
#     bbox sizes are unique per source color.
#   hidden assumptions audited from the original solver:
#   - Original `placeholder_colors = unique(inner)` no longer works because the frame
#     interior no longer contains source-colored destination markers.
#   - Original placeholder bbox collection assumed non-bg components inside the frame.
#     In this variant the relevant objects are bg components, and they must be checked
#     to be full rectangles rather than arbitrary shapes.
#   - Original sprite lookup used "placeholder color + same bbox size". After D4, color
#     no longer exists at the destination, so sprite lookup must invert to
#     "cavity bbox size -> source color".
#   - Original output clearing `out[1:-1, 1:-1] = bg` would destroy the tray that makes
#     the cavities visible. The variant must preserve the tray and only fill the holes.
#   generator legality:
#   - Cavity rectangles are laid out in a grid with at least one tray cell between them,
#     so bg holes never merge.
#   - Every row and column has at least one cavity, matching the original balanced layout.
#   - Empty slots remain solid tray cells rather than extra bg holes, so the solver never
#     sees unmatched cavities.
#   visual observability:
#   - The frame interior visibly changes from "colored blocks" to "panel with cut-out
#     slots", which is easy to spot from a few examples.
#   - The size pool is intentionally small and repeated, so a human can infer that holes
#     and external sprites correspond by dimensions, while legends still explain recolor.
#   color inference:
#   - Recolor is still justified structurally by the outside 3x4 legends; no ordering by
#     color id or invisible tie-break is used.


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


def _rectangular_cavities(crop: np.ndarray, bg: int) -> list[tuple[int, int, int, int]]:
    """Return all bg cavities strictly inside the frame; each must be a full rectangle."""
    h, w = crop.shape
    inner_mask = np.zeros((h, w), dtype=bool)
    inner_mask[1:-1, 1:-1] = True

    cavities: list[tuple[int, int, int, int]] = []
    for pts in components_mask((crop == bg) & inner_mask, connectivity=4):
        r0, c0, r1, c1 = bbox_coords(pts)
        assert 0 < r0 <= r1 < h - 1 and 0 < c0 <= c1 < w - 1, "cavity must stay away from the frame border"
        sub = crop[r0 : r1 + 1, c0 : c1 + 1]
        assert int(len(pts)) == int(sub.size), "cavity must be a solid rectangle"
        assert np.all(sub == bg), "cavity bbox must contain only background"
        cavities.append((int(r0), int(c0), int(r1), int(c1)))

    assert cavities, "no inner cavities found"
    return sorted(cavities)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill framed cavities with size-matched sprites, then recolor them using the outside legend.

    Input structure:
    - bg background, one rectangular gray frame/panel, and inside it several
      rectangular bg cavities.
    - Outside the frame: one sprite per source color and one compact two-color legend block per
      source color.

    Rule:
    - Detect the frame and collect the bg cavities inside it.
    - Read the outside legends to obtain source-color -> output-color mappings.
    - Match each cavity to the unique outside sprite whose bounding box has the same size, stamp
      that sprite into the cavity, and recolor it with the legend's output color.

    Key insight: The destination no longer names the sprite by color; instead, each hole acts like
    a jigsaw slot, so bbox size identifies the source sprite and the legend only controls recolor.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)

    frame_col, (fr0, fc0, fr1, fc1) = _find_frame(g, bg)
    crop = g[fr0 : fr1 + 1, fc0 : fc1 + 1]
    h, w = crop.shape
    assert h >= 3 and w >= 3, f"frame too small for interior: {(h, w)}"

    cavities = _rectangular_cavities(crop, bg)
    cavity_dims = [(r1 - r0 + 1, c1 - c0 + 1) for r0, c0, r1, c1 in cavities]
    unique_cavity_dims = sorted(set(cavity_dims))

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
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    "area": int(len(pts_tuples)),
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
            if ci["col"] not in sub_colors or cj["col"] not in sub_colors:
                continue
            if ci["area"] == cj["area"]:
                continue

            if ci["area"] > cj["area"]:
                src, tgt = int(ci["col"]), int(cj["col"])
            else:
                src, tgt = int(cj["col"]), int(ci["col"])

            assert src not in color_map, f"duplicate legend mapping for source color {src}"
            color_map[src] = tgt
            legend_ids.add(int(ci["id"]))
            legend_ids.add(int(cj["id"]))

    assert color_map, "no legend mappings found"

    sprite_coords: dict[int, tuple[tuple[tuple[int, int], ...], int, int]] = {}
    for src_col in sorted(color_map):
        all_candidates = [c for c in comps if c["col"] == src_col and c["id"] not in legend_ids]
        assert all_candidates, f"no non-legend sprite found for color {src_col}"

        sized = [
            c
            for c in all_candidates
            if (c["bbox"][2] - c["bbox"][0] + 1, c["bbox"][3] - c["bbox"][1] + 1) in unique_cavity_dims
        ]
        assert len(sized) == 1, f"expected one size-matched sprite for color {src_col}, found {len(sized)}"
        base = sized[0]
        sr0, sc0, sr1, sc1 = base["bbox"]

        pts_set = set(base["pts"])
        for comp in all_candidates:
            if comp["id"] == base["id"]:
                continue
            r0, c0, r1, c1 = comp["bbox"]
            assert sr0 <= r0 and sr1 >= r1 and sc0 <= c0 and sc1 >= c1, (
                f"extra sprite component for color {src_col} is outside the base bbox"
            )
            pts_set.update(comp["pts"])

        coords = tuple(sorted((r - sr0, c - sc0) for r, c in pts_set))
        sprite_coords[src_col] = (coords, sr1 - sr0 + 1, sc1 - sc0 + 1)

    dims_to_src: dict[tuple[int, int], int] = {}
    for src_col, (_, sh, sw) in sprite_coords.items():
        dims = (int(sh), int(sw))
        assert dims in unique_cavity_dims, f"sprite size {dims} has no matching cavity"
        assert dims not in dims_to_src, f"ambiguous shared cavity size {dims}"
        dims_to_src[dims] = int(src_col)
    assert set(dims_to_src) == set(unique_cavity_dims), "not every cavity size has a matching sprite"

    out = crop.copy()
    for r0, c0, r1, c1 in cavities:
        dims = (r1 - r0 + 1, c1 - c0 + 1)
        src_col = dims_to_src[dims]
        tgt_col = int(color_map[src_col])
        coords, sh, sw = sprite_coords[src_col]
        assert dims == (sh, sw), f"cavity size {dims} does not match sprite size {(sh, sw)}"
        for dr, dc in coords:
            out[r0 + dr, c0 + dc] = tgt_col

    return out


# --------------- generator (d4 cavity) ---------------

FRAME_COL = 5
GRID_ROWS_RANGE = (2, 4)
GRID_COLS_RANGE = (3, 5)
TOTAL_CAVITIES_RANGE = (6, 10)
MAX_EMPTY_SLOTS = 2
NUM_COLORS_RANGE = (2, 4)
SIZE_POOL = [(2, 3), (3, 2), (3, 3), (2, 4), (4, 2), (3, 4), (4, 3), (4, 4)]
SPRITE_FILL_MIN = 0.45
SPRITE_FILL_MAX = 0.8
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


def _carve_path(mask: np.ndarray, start: tuple[int, int], end: tuple[int, int]) -> tuple[int, int]:
    """Carve a 4-connected Manhattan path from start to end."""
    r, c = start
    tr, tc = end
    mask[r, c] = True
    while (r, c) != (tr, tc):
        choices = []
        if r != tr:
            choices.append((r + (1 if tr > r else -1), c))
        if c != tc:
            choices.append((r, c + (1 if tc > c else -1)))
        r, c = random.choice(choices)
        mask[r, c] = True
    return r, c


def _sprite_mask(h: int, w: int) -> np.ndarray:
    """Generate a connected h x w binary mask whose bbox is exactly h x w."""
    area = h * w
    lo = max(4, int(np.ceil(SPRITE_FILL_MIN * area)))
    hi = min(int(np.floor(SPRITE_FILL_MAX * area)), area - 1)
    target = random.randint(lo, max(lo, hi))

    mask = np.zeros((h, w), dtype=bool)
    anchors = [
        (0, random.randrange(w)),
        (h - 1, random.randrange(w)),
        (random.randrange(h), 0),
        (random.randrange(h), w - 1),
    ]
    random.shuffle(anchors)
    cur = anchors[0]
    mask[cur] = True
    for nxt in anchors[1:]:
        cur = _carve_path(mask, cur, nxt)

    while int(mask.sum()) < target:
        frontier = set()
        for r, c in zip(*np.where(mask)):
            for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc]:
                    frontier.add((nr, nc))
        assert frontier
        rr, cc = random.choice(sorted(frontier))
        mask[rr, cc] = True

    assert mask[0, :].any() and mask[-1, :].any() and mask[:, 0].any() and mask[:, -1].any()
    return mask


def _colored_sprite(mask: np.ndarray, color: int, bg: int) -> np.ndarray:
    out = np.full(mask.shape, bg, dtype=int)
    out[mask] = int(color)
    return out


def _choose_structure() -> tuple[int, int, int, int] | None:
    """Pick (rows, cols, empty_count, total) consistent with the original frame density."""
    options = []
    for rows in range(*GRID_ROWS_RANGE):
        for cols in range(*GRID_COLS_RANGE):
            slots = rows * cols
            for total in range(TOTAL_CAVITIES_RANGE[0], TOTAL_CAVITIES_RANGE[1] + 1):
                empty = slots - total
                if 0 <= empty <= MAX_EMPTY_SLOTS:
                    options.append((rows, cols, empty, total))
    return random.choice(options) if options else None


def _choose_empty_slots(
    rows: int, cols: int, empty_count: int
) -> tuple[set[tuple[int, int]], list[tuple[int, int]]]:
    slots = [(r, c) for r in range(rows) for c in range(cols)]
    valid = []
    for combo in itertools.combinations(slots, empty_count):
        empty = set(combo)
        filled = [slot for slot in slots if slot not in empty]
        if len({r for r, _ in filled}) == rows and len({c for _, c in filled}) == cols:
            valid.append(empty)
    assert valid
    empty_slots = random.choice(valid)
    filled_slots = [slot for slot in slots if slot not in empty_slots]
    random.shuffle(filled_slots)
    return empty_slots, filled_slots


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


def _layout_cavities(
    rows: int,
    cols: int,
    cell_colors: dict[tuple[int, int], int],
    size_map: dict[int, tuple[int, int]],
) -> tuple[dict[tuple[int, int], tuple[int, int, int, int]], list[int], list[int], int, int]:
    """Compute centered cavity rectangles inside a regular grid of variable-size cells."""
    row_heights = []
    for r in range(rows):
        row_sizes = [size_map[cell_colors[(r, c)]][0] for c in range(cols) if (r, c) in cell_colors]
        assert row_sizes
        row_heights.append(max(row_sizes))

    col_widths = []
    for c in range(cols):
        col_sizes = [size_map[cell_colors[(r, c)]][1] for r in range(rows) if (r, c) in cell_colors]
        assert col_sizes
        col_widths.append(max(col_sizes))

    row_offsets = []
    cur = 0
    for rh in row_heights:
        row_offsets.append(cur)
        cur += rh + 1
    content_h = cur - 1

    col_offsets = []
    cur = 0
    for cw in col_widths:
        col_offsets.append(cur)
        cur += cw + 1
    content_w = cur - 1

    placements: dict[tuple[int, int], tuple[int, int, int, int]] = {}
    for (rr, cc), src in cell_colors.items():
        ch, cw = size_map[src]
        cell_r0 = row_offsets[rr]
        cell_c0 = col_offsets[cc]
        off_r = cell_r0 + (row_heights[rr] - ch) // 2
        off_c = cell_c0 + (col_widths[cc] - cw) // 2
        placements[(rr, cc)] = (off_r, off_c, ch, cw)

    return placements, row_heights, col_widths, content_h, content_w


def _gen_attempt() -> np.ndarray | None:
    struct = _choose_structure()
    if struct is None:
        return None
    rows, cols, empty_count, total = struct

    bg = random.choice([c for c in range(10) if c != FRAME_COL])
    usable = [c for c in range(10) if c not in (bg, FRAME_COL)]

    k = random.randint(NUM_COLORS_RANGE[0], min(NUM_COLORS_RANGE[1], total, len(SIZE_POOL)))
    src_cols = random.sample(usable, k)
    tgt_pool = usable[:]
    tgt_cols = [random.choice([c for c in tgt_pool if c != src]) for src in src_cols]
    color_to_tgt = dict(zip(src_cols, tgt_cols))

    size_choices = random.sample(SIZE_POOL, k)
    size_map = {src: dims for src, dims in zip(src_cols, size_choices)}
    masks = {src: _sprite_mask(*size_map[src]) for src in src_cols}

    _, filled_slots = _choose_empty_slots(rows, cols, empty_count)
    cell_colors: dict[tuple[int, int], int] = {}
    for slot, src in zip(filled_slots[:k], src_cols):
        cell_colors[slot] = src
    for slot in filled_slots[k:]:
        cell_colors[slot] = random.choice(src_cols)

    placements, _, _, content_h, content_w = _layout_cavities(rows, cols, cell_colors, size_map)

    min_fh = max(MIN_FRAME_H, content_h + 4)
    min_fw = max(MIN_FRAME_W, content_w + 4)
    max_fh = min(MAX_FRAME_DIM, content_h + 4 + FRAME_MARGIN_SLACK)
    max_fw = min(MAX_FRAME_DIM, content_w + 4 + FRAME_MARGIN_SLACK)
    if min_fh > max_fh or min_fw > max_fw:
        return None

    frame_h = random.randint(min_fh, max_fh)
    frame_w = random.randint(min_fw, max_fw)
    inner_h = frame_h - 2
    inner_w = frame_w - 2
    y_slack = inner_h - content_h
    x_slack = inner_w - content_w
    if y_slack < 2 or x_slack < 2:
        return None
    top_margin = random.randint(1, y_slack - 1)
    left_margin = random.randint(1, x_slack - 1)

    input_frame = np.full((frame_h, frame_w), FRAME_COL, dtype=int)
    output_frame = np.full((frame_h, frame_w), FRAME_COL, dtype=int)
    input_frame[0, :] = input_frame[-1, :] = FRAME_COL
    input_frame[:, 0] = input_frame[:, -1] = FRAME_COL
    output_frame[0, :] = output_frame[-1, :] = FRAME_COL
    output_frame[:, 0] = output_frame[:, -1] = FRAME_COL

    for slot, src in cell_colors.items():
        rr, cc, ch, cw = placements[slot]
        r0 = 1 + top_margin + rr
        c0 = 1 + left_margin + cc
        input_frame[r0 : r0 + ch, c0 : c0 + cw] = bg
        output_frame[r0 : r0 + ch, c0 : c0 + cw] = bg
        sprite = _colored_sprite(masks[src], color_to_tgt[src], bg)
        paste(output_frame, sprite, r0, c0, transparent=bg)

    patch_bbox_area = sum(h * w for h, w in size_map.values()) + 12 * k + 8 * k
    possible_dims = [
        (h, w)
        for h in range(*CANVAS_H_RANGE)
        for w in range(*CANVAS_W_RANGE)
        if h >= frame_h and w >= frame_w and h * w - frame_h * frame_w >= patch_bbox_area
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
        patches.append((_legend(src, tgt), False))
    random.shuffle(patches)
    patches.sort(key=lambda item: item[0].size, reverse=True)
    for patch, transparent in patches:
        if not _place_patch(grid, reserve, patch, bg, transparent=transparent):
            return None

    if get_bg(grid) != bg:
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
    raise RuntimeError("failed to generate valid edb79dae D4 input")
