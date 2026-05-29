# Combo variant of edb79dae (index 30)
# combo_spec: D14/0+D3/3
# Name: 左上角对齐+对角标记源
#
# Design analysis
# a) Concrete behavior:
#    - Original code assumes each inside target is an NxN solid placeholder block and each outside source is a full
#      same-color sprite with the same bbox. Placement is effectively bbox-to-bbox replacement.
#    - In this combo, the inside target must become a single colored anchor pixel, otherwise "top-left alignment"
#      is visually invisible. The outside source must become exactly two same-color pixels at opposite corners of a
#      bbox; those two pixels define an axis-aligned rectangle which expands to a full solid block during placement.
#    - Example: an outside red pair at (0,3) and (2,0) defines a 3x4 rectangle. If the matching red anchor inside
#      the frame is at (5,7), the output paints rows 5..7 and cols 7..10. A center-aligned counterfactual would
#      start earlier and produce a different footprint, so the alignment change is observable.
# b) Complexity calibration:
#    - Keep 2-4 colors and 6-10 anchor pixels on a regular grid. This gives repeated evidence per sample without
#      overwhelming the visual field.
#    - Rectangle sizes are limited to 2-4 in each dimension. This keeps them clearly visible, avoids trivial 1x1
#      degeneracy, and still fits 2-3 by 3-4 anchor grids inside a 13-21 by 18-21 frame.
# c) Hidden assumptions broken from the original solver:
#    - Placeholder bbox size no longer matches source bbox size; placeholder components are now single pixels.
#    - Source shapes are no longer dense or even connected; each source color is represented by exactly two
#      non-legend single-pixel components outside the frame.
#    - Placement anchor is no longer derived from placeholder bbox equality. It is the placeholder pixel itself,
#      and the pasted rectangle is offset only by that pixel.
# d) Placement legality:
#    - The generator uses a slot lattice with pitch (max_rect_h + 1, max_rect_w + 1), so every placed rectangle fits
#      completely inside its slot and leaves at least one background gap to neighbors.
#    - Outside patches reserve their full bbox even when only two diagonal pixels are visible, preventing overlaps.
# e) Visual observability:
#    - Humans can see the two-dot diagonal sources, infer the missing rectangle, and then notice that the rectangle's
#      top-left corner lands exactly on the inside colored anchor pixel.
#    - Colors are disjoint between source colors and recolor targets, so the legend is easy to read.
# f) Visual cleanliness:
#    - The frame interior contains only single anchor pixels in the input and non-overlapping solid rectangles in the
#      output. Outside the frame, each source is a sparse two-pixel marker plus one compact legend block.
# g) New generator-guaranteed assumptions used by main():
#    - Each placeholder component is exactly one pixel.
#    - For every placeholder color there is exactly one non-legend diagonal-marker pair of that color outside.
#    - Each such pair spans at least 2 rows and 2 cols, so expansion yields a proper rectangle.

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
    Reconstruct rectangles from diagonal markers, stamp them from anchor pixels inside a frame, then recolor them.

    Input structure:
    - The grid has a bg color, one rectangular gray frame, and single-pixel colored anchors inside that frame.
    - Outside the frame, each anchor color appears as two same-color pixels at opposite corners of a rectangle bbox.
    - Small overlapping two-color legend glyphs outside the frame map each anchor/source color to its output color.

    Rule:
    - Find the frame and collect every one-pixel anchor inside it.
    - For each anchor color, find the matching outside diagonal-marker pair and infer the full rectangle spanning the
      pair's bbox.
    - Place that inferred rectangle so its top-left corner lands on the anchor pixel.
    - Recolor the placed rectangle using the legend mapping for that source color.

    Key insight:
    - The outside source is not the visible two-pixel pattern itself; those two pixels only encode a rectangle's
      size. The inside colored pixel is the rectangle's anchor, and the legend changes only the final color.
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
    assert placeholder_colors, "no anchor colors found inside frame"

    anchors: list[tuple[int, int, int]] = []
    for col in sorted(placeholder_colors):
        for pts in components_mask((crop == col) & inner_mask, connectivity=4):
            assert len(pts) == 1, f"anchor color {col} must use single-pixel targets"
            r, c = map(int, pts[0])
            anchors.append((col, r, c))

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
            if (ci["col"] in placeholder_colors) ^ (cj["col"] in placeholder_colors):
                src = ci["col"] if ci["col"] in placeholder_colors else cj["col"]
                tgt = cj["col"] if src == ci["col"] else ci["col"]
            else:
                assert ci["area"] != cj["area"], f"ambiguous equal-area legend pair: {ci['col']} and {cj['col']}"
                if ci["area"] > cj["area"]:
                    src, tgt = ci["col"], cj["col"]
                else:
                    src, tgt = cj["col"], ci["col"]
            assert src in placeholder_colors, f"legend source color {src} is not an anchor color"
            assert src not in color_map, f"duplicate legend mapping for anchor color {src}"
            color_map[src] = int(tgt)
            legend_ids.add(ci["id"])
            legend_ids.add(cj["id"])

    assert set(color_map) == placeholder_colors, (
        f"legend mappings {sorted(color_map)} do not match anchors {sorted(placeholder_colors)}"
    )

    rect_hw: dict[int, tuple[int, int]] = {}
    for src_col in sorted(placeholder_colors):
        candidates = [c for c in comps if c["col"] == src_col and c["id"] not in legend_ids]
        assert len(candidates) == 2, f"expected exactly two diagonal-marker components for color {src_col}"
        assert all(c["area"] == 1 for c in candidates), f"diagonal markers for color {src_col} must be single pixels"
        pts = [cand["pts"][0] for cand in candidates]
        rs = [p[0] for p in pts]
        cs = [p[1] for p in pts]
        r0, r1 = min(rs), max(rs)
        c0, c1 = min(cs), max(cs)
        assert r0 < r1 and c0 < c1, f"markers for color {src_col} must occupy opposite rectangle corners"
        rect_hw[src_col] = (r1 - r0 + 1, c1 - c0 + 1)

    out = crop.copy()
    out[1:-1, 1:-1] = bg

    for src_col, pr, pc in anchors:
        assert src_col in rect_hw and src_col in color_map, f"anchor color {src_col} missing rectangle or color map"
        rh, rw = rect_hw[src_col]
        assert pr + rh <= H and pc + rw <= W, f"rectangle for color {src_col} does not fit from anchor {(pr, pc)}"
        out[pr : pr + rh, pc : pc + rw] = int(color_map[src_col])

    return out


# --------------- generator (combo D14/0 + D3/3) ---------------

FRAME_COL = 5
RECT_H_RANGE = (2, 5)               # 2-4 inclusive
RECT_W_RANGE = (2, 5)               # 2-4 inclusive
GRID_ROWS_RANGE = (2, 4)            # 2-3 inclusive
GRID_COLS_RANGE = (3, 5)            # 3-4 inclusive
TOTAL_ANCHORS_RANGE = (6, 10)       # inclusive
MAX_EMPTY_SLOTS = 2
NUM_COLORS_RANGE = (2, 4)           # inclusive, kept <=4 so src/tgt colors stay disjoint
MIN_FRAME_H = 13
MIN_FRAME_W = 18
MAX_FRAME_DIM = 21
FRAME_MARGIN_SLACK = 2
CANVAS_H_RANGE = (22, 31)
CANVAS_W_RANGE = (23, 31)
FRAME_POS_JITTER_PROB = 0.25
PATCH_PADDING = 1
GEN_RETRIES = 300


def _legend(src: int, tgt: int) -> np.ndarray:
    return np.array(
        [[src, src, tgt, tgt],
         [src, src, src, tgt],
         [src, src, tgt, tgt]],
        dtype=int,
    )


def _diagonal_pair_patch(h: int, w: int, color: int, bg: int) -> np.ndarray:
    patch = np.full((h, w), bg, dtype=int)
    if random.random() < 0.5:
        patch[0, 0] = int(color)
        patch[h - 1, w - 1] = int(color)
    else:
        patch[0, w - 1] = int(color)
        patch[h - 1, 0] = int(color)
    return patch


def _content_extent(max_h: int, max_w: int, rows: int, cols: int) -> tuple[int, int]:
    return (rows * max_h + rows - 1, cols * max_w + cols - 1)


def _choose_structure() -> tuple[int, int, int, int] | None:
    total = random.randint(*TOTAL_ANCHORS_RANGE)
    options = []
    for rows in range(*GRID_ROWS_RANGE):
        for cols in range(*GRID_COLS_RANGE):
            slots = rows * cols
            empty = slots - total
            if 0 <= empty <= MAX_EMPTY_SLOTS:
                options.append((rows, cols, empty, total))
    return random.choice(options) if options else None


def _choose_empty_slots(
    rows: int, cols: int, empty_count: int,
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
    grid: np.ndarray, reserve: np.ndarray, patch: np.ndarray, bg: int, *, transparent: bool,
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
    rows, cols, empty_count, _ = struct

    bg = random.choice([c for c in range(10) if c != FRAME_COL])
    usable = [c for c in range(10) if c not in (bg, FRAME_COL)]

    k = random.randint(*NUM_COLORS_RANGE)
    if 2 * k > len(usable):
        return None
    chosen = random.sample(usable, 2 * k)
    src_cols = chosen[:k]
    tgt_cols = chosen[k:]
    color_to_tgt = dict(zip(src_cols, tgt_cols))

    rect_specs = {
        src: (
            random.randint(*RECT_H_RANGE),
            random.randint(*RECT_W_RANGE),
        )
        for src in src_cols
    }
    max_h = max(h for h, _ in rect_specs.values())
    max_w = max(w for _, w in rect_specs.values())

    ch, cw = _content_extent(max_h, max_w, rows, cols)
    min_fh = max(MIN_FRAME_H, ch + 2)
    min_fw = max(MIN_FRAME_W, cw + 2)
    max_fh = min(MAX_FRAME_DIM, ch + 2 + FRAME_MARGIN_SLACK)
    max_fw = min(MAX_FRAME_DIM, cw + 2 + FRAME_MARGIN_SLACK)
    if min_fh > max_fh or min_fw > max_fw:
        return None
    frame_h = random.randint(min_fh, max_fh)
    frame_w = random.randint(min_fw, max_fw)
    y_margin = frame_h - 2 - ch
    x_margin = frame_w - 2 - cw
    top_margin = random.randint(0, y_margin)
    left_margin = random.randint(0, x_margin)

    _, filled_slots = _choose_empty_slots(rows, cols, empty_count)
    cell_colors: dict[tuple[int, int], int] = {}
    for slot, col in zip(filled_slots[:k], src_cols):
        cell_colors[slot] = col
    for slot in filled_slots[k:]:
        cell_colors[slot] = random.choice(src_cols)

    input_frame = _make_bordered_frame(frame_h, frame_w, bg)
    output_frame = _make_bordered_frame(frame_h, frame_w, bg)

    slot_h = max_h + 1
    slot_w = max_w + 1
    for (rr, cc), src in cell_colors.items():
        r0 = 1 + top_margin + rr * slot_h
        c0 = 1 + left_margin + cc * slot_w
        input_frame[r0, c0] = int(src)
        rh, rw = rect_specs[src]
        output_frame[r0 : r0 + rh, c0 : c0 + rw] = int(color_to_tgt[src])

    ext_area = sum((h * w) + 12 + 10 for h, w in rect_specs.values())
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
        rh, rw = rect_specs[src]
        patches.append((_diagonal_pair_patch(rh, rw, src, bg), True))
        patches.append((_legend(src, color_to_tgt[src]), False))
    random.shuffle(patches)
    patches.sort(key=lambda item: item[0].shape[0] * item[0].shape[1], reverse=True)
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
    raise RuntimeError("failed to generate valid edb79dae combo input")
