# Combo variant of edb79dae (index 21)
# combo_spec: D3/2+D8/0+D9/0
# Name: 精灵散布四周+保留源+全网格输出

# Design analysis for the fused variant:
# 1. Concrete trace:
#    Original behavior: frame interior contains solid placeholder blocks; each placeholder color finds its matching
#    sprite outside the frame; that sprite is effectively moved inward and recolored; the returned answer is cropped to
#    the frame. Fused behavior: the same matching and recoloring logic applies, but the outside sprites are scattered
#    around all four sides, they stay visible in the output, and the output keeps the entire canvas. Example: if the
#    red sprite is above the frame, blue left, green below, and orange right, then the output still shows those four
#    outside exemplars while also showing red/blue/green/orange-derived copies stamped inside the frame.
# 2. Complexity calibration:
#    The generator fixes the number of source colors to four. That is deliberate: it yields one source sprite per side,
#    so the surround layout is always obvious, but it avoids a palette explosion that would make the legend harder to
#    parse from only a few ARC examples. Placeholders repeat colors inside the frame so every source color has multiple
#    confirmations, not just a single one-off placement.
# 3. Implicit-assumption audit of the original solver:
#    - `crop = frame region` used to be both workspace and output. That is broken by D9/0, so the new solver only uses
#      the crop for analysis and writes results back into a full-grid copy.
#    - `outside` used to mean "temporary source area that disappears". D8/0 breaks that: the solver must clear only the
#      frame interior and must never erase outside pixels.
#    - Sprite lookup assumed the external source could be searched without directional structure. D3/2 does not change
#      the matching rule, but it removes the one-sided clustering shortcut, so the generator must guarantee visibility
#      on top/bottom/left/right while keeping exactly one usable NxN sprite bbox per source color.
#    - Legend detection relies on compact bg-free overlapping two-color glyphs. The generator preserves the original 3x4
#      legend exactly and isolates them in the four corners so they cannot be confused with side sprites.
#    - Placeholder-size matching assumes all instances of one source color share one bbox size. The generator keeps the
#      original shared-N design for all copies of a given color and uses the same N for every external sprite.
#    - The merge step for disconnected sprite pieces assumes every extra component of the chosen source color lies inside
#      the same outer bbox. The generator keeps one sprite template per source color, so this remains true.
# 4. Placement legality:
#    The frame is embedded with at least a 5-cell moat to every canvas edge. Legends occupy corner rectangles, and side
#    sprites occupy disjoint top/bottom/left/right bands aligned with the frame. Those bands are chosen so legends and
#    sprites cannot overlap, cannot clip, and remain separated by background.
# 5. Visual observability and neatness:
#    Object roles are spatially partitioned: legends in corners, source sprites on four sides, placeholders inside the
#    frame. Across input/output pairs only the frame interior changes, so a human can notice "outside stays, inside gets
#    copied+recolored" without disentangling overlapping annotations.
# 6. Inferability of color change:
#    Every recoloring is justified by an explicit legend block, and the target palette is a derangement of the four
#    source colors. That keeps the mapping visible while preventing identity recolors that would weaken the task signal.
# 7. New solver assumptions and generator guarantees:
#    - The solver expects every source color to appear in at least one placeholder and in exactly one non-legend sprite
#      bbox outside the frame; the generator enforces both.
#    - The solver expects full-grid output to differ from the original task structurally; the generator enforces this by
#      always leaving non-frame content in place and by placing sprites on all four sides so cropping would lose data.
#    - The solver expects the keep-source interpretation to be observable, not vacuous; the generator enforces that by
#      making every outside sprite remain unchanged and visible in the final whole-grid output.

import itertools
import random

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, paste


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
    Copy recolored sprites into a framed region while keeping the surrounding panorama intact.

    Input structure:
    - A bg canvas contains one grey rectangular frame whose interior holds solid colored placeholder blocks.
    - Outside the frame, one sprite template per placeholder color is scattered around the top, bottom, left, and right
      of the frame rather than being packed on one side.
    - Small two-color legend glyphs elsewhere outside the frame map each placeholder color to its output color.

    Rule:
    - Detect the frame, the placeholder colors inside it, the outside sprite shapes, and the legend color mappings.
    - For each placeholder block, copy the matching-color outside sprite into the placeholder bbox inside the frame.
    - Recolor the copied sprite with the legend's target color.
    - Return the entire grid: outside sprites and legends stay in place, and only the frame interior is rewritten.

    Key insight: Placeholder colors choose which external shape to duplicate, while the compact two-color legends choose
    the inside recoloring; because the full canvas is preserved, this is a copy-and-transform task, not a move-and-crop task.
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
    for col in sorted(placeholder_colors):
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

            assert src in placeholder_colors, f"legend source color {src} is not a placeholder"
            assert src not in color_map, f"duplicate legend mapping for placeholder color {src}"
            color_map[src] = int(tgt)
            legend_ids.add(ci["id"])
            legend_ids.add(cj["id"])

    assert set(color_map) == placeholder_colors, (
        f"legend mappings {sorted(color_map)} do not match placeholders {sorted(placeholder_colors)}"
    )

    sprite_coords: dict[int, tuple[tuple[tuple[int, int], ...], int, int]] = {}
    for src_col in sorted(placeholder_colors):
        all_candidates = [c for c in comps if c["col"] == src_col and c["id"] not in legend_ids]
        assert all_candidates, f"no non-legend sprite found for color {src_col}"
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

    out = g.copy()
    out[fr0 + 1 : fr1, fc0 + 1 : fc1] = bg

    for src_col, (pr0, pc0, pr1, pc1) in placeholders:
        assert src_col in sprite_coords and src_col in color_map, f"placeholder color {src_col} missing sprite or color mapping"
        tgt_col = int(color_map[src_col])
        coords, sh, sw = sprite_coords[src_col]
        assert (pr1 - pr0 + 1, pc1 - pc0 + 1) == (sh, sw), (
            f"placeholder size does not match sprite for color {src_col}"
        )
        off_r = fr0 + pr0
        off_c = fc0 + pc0
        for dr, dc in coords:
            rr, cc = off_r + dr, off_c + dc
            out[rr, cc] = tgt_col

    return out


FRAME_COL = 5
SPRITE_NS = (3, 4)
GRID_ROWS = (2, 3)
GRID_COLS = (3, 4)
TOTAL_BLOCKS_RANGE = (6, 10)
MAX_EMPTY_SLOTS = 2
NUM_COLORS = 4
SPRITE_FILL_MIN = 0.5
SPRITE_FILL_MAX = 0.9
MIN_FRAME_H = 13
MIN_FRAME_W = 18
MAX_FRAME_DIM = 20
FRAME_MARGIN_SLACK = 4
CANVAS_H_RANGE = (22, 31)
CANVAS_W_RANGE = (23, 31)
OUTER_MARGIN_MIN = 5
GEN_RETRIES = 300


def _legend(src: int, tgt: int) -> np.ndarray:
    return np.array(
        [
            [src, src, tgt, tgt],
            [src, src, src, tgt],
            [src, src, tgt, tgt],
        ],
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


def _content_extent(n: int, grid_rows: int, grid_cols: int) -> tuple[int, int]:
    return (grid_rows * n + grid_rows - 1, grid_cols * n + grid_cols - 1)


def _choose_structure() -> tuple[int, int, int, int, int] | None:
    options = []
    for total in range(TOTAL_BLOCKS_RANGE[0], TOTAL_BLOCKS_RANGE[1] + 1):
        for n in SPRITE_NS:
            for rows in GRID_ROWS:
                for cols in GRID_COLS:
                    slots = rows * cols
                    empty = slots - total
                    if not (0 <= empty <= MAX_EMPTY_SLOTS):
                        continue
                    ch, cw = _content_extent(n, rows, cols)
                    min_fh = max(MIN_FRAME_H, ch + 4)
                    min_fw = max(MIN_FRAME_W, cw + 4)
                    max_fh = min(MAX_FRAME_DIM, max(min_fh, ch + 2 + FRAME_MARGIN_SLACK))
                    max_fw = min(MAX_FRAME_DIM, max(min_fw, cw + 2 + FRAME_MARGIN_SLACK))
                    if min_fh > max_fh or min_fw > max_fw:
                        continue
                    if min_fh + 2 * OUTER_MARGIN_MIN > CANVAS_H_RANGE[1] - 1:
                        continue
                    if min_fw + 2 * OUTER_MARGIN_MIN > CANVAS_W_RANGE[1] - 1:
                        continue
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
    assert valid, "no valid empty-slot layout"
    empty_slots = random.choice(valid)
    filled_slots = [s for s in slots if s not in empty_slots]
    random.shuffle(filled_slots)
    return empty_slots, filled_slots


def _make_bordered_frame(h: int, w: int, bg: int) -> np.ndarray:
    frame = np.full((h, w), bg, dtype=int)
    frame[0, :] = frame[-1, :] = FRAME_COL
    frame[:, 0] = frame[:, -1] = FRAME_COL
    return frame


def _random_derangement(items: list[int]) -> list[int]:
    base = items[:]
    while True:
        perm = random.sample(base, len(base))
        if all(a != b for a, b in zip(base, perm)):
            return perm


def _rand_top_left(r0: int, r1: int, c0: int, c1: int) -> tuple[int, int]:
    assert r0 <= r1 and c0 <= c1, f"invalid placement box: {(r0, r1, c0, c1)}"
    return random.randint(r0, r1), random.randint(c0, c1)


def _gen_attempt() -> np.ndarray | None:
    struct = _choose_structure()
    if struct is None:
        return None
    n, rows, cols, empty_count, _ = struct

    bg = random.choice([c for c in range(10) if c != FRAME_COL])
    usable = [c for c in range(10) if c not in (bg, FRAME_COL)]
    src_cols = random.sample(usable, NUM_COLORS)
    tgt_cols = _random_derangement(src_cols)
    color_to_tgt = dict(zip(src_cols, tgt_cols))
    masks = {src: _sprite_mask(n) for src in src_cols}

    ch, cw = _content_extent(n, rows, cols)
    min_fh = max(MIN_FRAME_H, ch + 4)
    min_fw = max(MIN_FRAME_W, cw + 4)
    max_fh = min(MAX_FRAME_DIM, max(min_fh, ch + 2 + FRAME_MARGIN_SLACK))
    max_fw = min(MAX_FRAME_DIM, max(min_fw, cw + 2 + FRAME_MARGIN_SLACK))
    if min_fh > max_fh or min_fw > max_fw:
        return None
    frame_h = random.randint(min_fh, max_fh)
    frame_w = random.randint(min_fw, max_fw)

    y_margin = frame_h - 2 - ch
    x_margin = frame_w - 2 - cw
    if y_margin < 2 or x_margin < 2:
        return None
    top_margin = random.randint(1, y_margin - 1)
    left_margin = random.randint(1, x_margin - 1)

    _, filled_slots = _choose_empty_slots(rows, cols, empty_count)
    cell_colors: dict[tuple[int, int], int] = {}
    for slot, col in zip(filled_slots[:NUM_COLORS], src_cols):
        cell_colors[slot] = col
    for slot in filled_slots[NUM_COLORS:]:
        cell_colors[slot] = random.choice(src_cols)

    input_frame = _make_bordered_frame(frame_h, frame_w, bg)
    output_frame = _make_bordered_frame(frame_h, frame_w, bg)
    for (rr, cc), src in cell_colors.items():
        r0 = 1 + top_margin + rr * (n + 1)
        c0 = 1 + left_margin + cc * (n + 1)
        paste(input_frame, np.full((n, n), src, dtype=int), r0, c0)
        paste(output_frame, _colored_sprite(masks[src], color_to_tgt[src], bg), r0, c0, transparent=bg)

    h_min = max(CANVAS_H_RANGE[0], frame_h + 2 * OUTER_MARGIN_MIN)
    h_max = CANVAS_H_RANGE[1] - 1
    w_min = max(CANVAS_W_RANGE[0], frame_w + 2 * OUTER_MARGIN_MIN)
    w_max = CANVAS_W_RANGE[1] - 1
    if h_min > h_max or w_min > w_max:
        return None
    h = random.randint(h_min, h_max)
    w = random.randint(w_min, w_max)

    extra_h = h - frame_h - 2 * OUTER_MARGIN_MIN
    extra_w = w - frame_w - 2 * OUTER_MARGIN_MIN
    fr0 = OUTER_MARGIN_MIN + random.randint(0, extra_h)
    fc0 = OUTER_MARGIN_MIN + random.randint(0, extra_w)
    fr1 = fr0 + frame_h - 1
    fc1 = fc0 + frame_w - 1

    grid = np.full((h, w), bg, dtype=int)
    paste(grid, input_frame, fr0, fc0)

    legend_positions = ["tl", "tr", "bl", "br"]
    random.shuffle(legend_positions)
    for pos, src in zip(legend_positions, src_cols):
        patch = _legend(src, color_to_tgt[src])
        if pos == "tl":
            rr, cc = _rand_top_left(0, fr0 - 4, 0, fc0 - 5)
        elif pos == "tr":
            rr, cc = _rand_top_left(0, fr0 - 4, fc1 + 2, w - 4)
        elif pos == "bl":
            rr, cc = _rand_top_left(fr1 + 2, h - 3, 0, fc0 - 5)
        else:
            rr, cc = _rand_top_left(fr1 + 2, h - 3, fc1 + 2, w - 4)
        paste(grid, patch, rr, cc)

    side_to_src = dict(zip(["top", "bottom", "left", "right"], random.sample(src_cols, NUM_COLORS)))
    for side, src in side_to_src.items():
        patch = _colored_sprite(masks[src], src, bg)
        if side == "top":
            rr, cc = _rand_top_left(0, fr0 - n - 1, fc0, fc1 - n + 1)
        elif side == "bottom":
            rr, cc = _rand_top_left(fr1 + 2, h - n, fc0, fc1 - n + 1)
        elif side == "left":
            rr, cc = _rand_top_left(fr0, fr1 - n + 1, 0, fc0 - n - 1)
        else:
            rr, cc = _rand_top_left(fr0, fr1 - n + 1, fc1 + 2, w - n)
        paste(grid, patch, rr, cc, transparent=bg)

    expected = grid.copy()
    paste(expected, output_frame, fr0, fc0)

    try:
        if not np.array_equal(main(grid), expected):
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
