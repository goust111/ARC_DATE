# Combo variant of edb79dae (index 10)
# combo_spec: D8/2+D9/0
# Name: 移动轨迹+全网格输出
#
# Design analysis for the fused variant
# 1. Concrete behavior trace:
#    - Base task: a solid placeholder block inside the grey frame chooses a same-colored sprite
#      outside the frame, then only the frame crop is returned after recoloring the stamped shape.
#    - Fused task: the output keeps the full input canvas. The source sprites and legends remain
#      visible, the frame interior is rebuilt with legend-recolored sprite copies, and same-colored
#      trails are drawn from each outside source sprite toward all matching destinations in the frame.
#    - Small example: if a blue source sprite sits left of the frame and an entire frame row is made
#      of blue placeholder blocks, the old output hides the source side completely. The new output
#      keeps the source sprite, keeps the legend, and adds a blue corridor that visibly runs from
#      outside into that row of the frame before the row's blocks are replaced by the recolored shape.
# 2. Complexity calibration:
#    - A fully unconstrained many-to-many layout would make trails cross legends or each other and
#      create brittle matching assumptions.
#    - The generator therefore assigns one placeholder color per frame row (2-3 rows total) and
#      repeats that color across all columns in the row. Each color has repeated evidence, while each
#      trail family occupies its own horizontal lane.
# 3. Hidden assumptions broken by the combo:
#    - Old assumption: output can be cropped to the frame. D9/0 invalidates this, so `main()` now
#      edits a full-grid copy of the input instead of returning a crop.
#    - Old assumption: outside-frame sprites can disappear after matching. D8/2 invalidates this,
#      because the output must still show the original source region as the origin of the trail.
#    - Old assumption: only frame-interior cells change. The fused rule changes both frame-interior
#      cells and exterior background cells along the trajectory.
# 4. Placement legality:
#    - Legends live in a top band. Source sprites live in a left band, vertically aligned with the
#      placeholder row they control. The frame sits to the right of the sources and below the legends.
#    - This guarantees non-overlap, in-bounds placement, and clear empty corridors where trails can be
#      drawn without touching unrelated objects.
# 5. Visual observability:
#    - The new information outside the frame is meaningful rather than decorative: each trail starts
#      from a real source sprite and enters the exact row whose placeholders use that sprite's color.
#    - The legend still explains recoloring, so a human can infer both "shape source" and "final color".
# 6. Visual cleanliness:
#    - There are no branching mazes or dense overlays. Each color gets a dedicated horizontal lane,
#      legends are isolated above the frame, and the frame interior still uses the original tidy grid.
# 7. Color-change inferability:
#    - Only 2-3 source colors are used, each repeated several times in one row. Target colors are
#      distinct from source colors, so the legend remains the only recoloring mechanism.
# 8. New solver assumptions and generator guarantees:
#    - `main()` assumes one non-legend source sprite bbox per placeholder color; the generator creates
#      exactly one such sprite for each row color.
#    - `main()` assumes the straight trail path from a source sprite to its row only crosses background
#      outside the frame; the generator reserves that corridor.
#    - `main()` assumes legends are the only compact overlapping two-color regions without background;
#      the generator keeps legends isolated in the top strip.

from __future__ import annotations

import random

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, paste


FRAME_COL = 5

# --- generator configuration ---
SPRITE_N_RANGE = (3, 6)          # choose 3-5
ROWS_RANGE = (2, 4)              # choose 2-3
COLS_OPTIONS = (3, 4)
MIN_FRAME_H = 13
MIN_FRAME_W = 18
MAX_FRAME_DIM = 21
FRAME_MARGIN_SLACK = 3
SPRITE_FILL_MIN = 0.5
SPRITE_FILL_MAX = 0.9
CANVAS_H_RANGE = (22, 31)        # choose 22-30
CANVAS_W_RANGE = (23, 31)        # choose 23-30
LEGEND_H = 3
LEGEND_W = 4
LEGEND_GAP = 1
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
            candidates.append((int(col), (int(r0), int(c0), int(r1), int(c1))))
    assert len(candidates) == 1, f"expected exactly one rectangular frame component, found {len(candidates)}"
    return candidates[0]


def _bbox_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (ar1 < br0 or br1 < ar0 or ac1 < bc0 or bc1 < ac0)


def _legend(src: int, tgt: int) -> np.ndarray:
    """Build the original 3x4 legend glyph."""
    return np.array(
        [[src, src, tgt, tgt], [src, src, src, tgt], [src, src, tgt, tgt]],
        dtype=int,
    )


def _sprite_mask(n: int) -> np.ndarray:
    """Generate an NxN binary mask with at least one background cell."""
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
    """Render a sprite mask in one color on background."""
    out = np.full(mask.shape, bg, dtype=int)
    out[mask] = int(color)
    return out


def _content_extent(n: int, rows: int, cols: int) -> tuple[int, int]:
    """Content area for a rows x cols grid of NxN blocks with 1-cell gaps."""
    return (rows * n + rows - 1, cols * n + cols - 1)


def _make_bordered_frame(h: int, w: int, bg: int) -> np.ndarray:
    frame = np.full((h, w), bg, dtype=int)
    frame[0, :] = frame[-1, :] = FRAME_COL
    frame[:, 0] = frame[:, -1] = FRAME_COL
    return frame


def _path_points(start: tuple[int, int], end: tuple[int, int]) -> tuple[tuple[int, int], ...]:
    """Inclusive Manhattan path, horizontal first then vertical."""
    r, c = start
    tr, tc = end
    pts = [(int(r), int(c))]

    dc = 0 if tc == c else (1 if tc > c else -1)
    while c != tc:
        c += dc
        pts.append((int(r), int(c)))

    dr = 0 if tr == r else (1 if tr > r else -1)
    while r != tr:
        r += dr
        pts.append((int(r), int(c)))
    return tuple(pts)


def _center_of_bbox(bbox: tuple[int, int, int, int]) -> tuple[int, int]:
    r0, c0, r1, c1 = bbox
    return ((r0 + r1) // 2, (c0 + c1) // 2)


def _parse_problem(grid: np.ndarray) -> dict:
    """Parse the input into frame, placeholders, legends, and source sprites."""
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
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
            placeholders.append((int(col), (int(r0), int(c0), int(r1), int(c1))))
    placeholders.sort(key=lambda item: (item[1][0], item[1][1], item[0]))

    phw: dict[int, tuple[int, int]] = {}
    for col in placeholder_colors:
        dims = [(b[2] - b[0] + 1, b[3] - b[1] + 1) for c, b in placeholders if c == col]
        assert dims, f"no placeholder component found for color {col}"
        unique_dims = sorted(set(dims))
        assert len(unique_dims) == 1, f"ambiguous placeholder sizes for color {col}: {unique_dims}"
        phw[int(col)] = unique_dims[0]

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
                    "id": int(cid),
                    "col": int(col),
                    "pts": pts_tuples,
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
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
            if ci["col"] not in sub_colors or cj["col"] not in sub_colors:
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
            color_map[int(src)] = int(tgt)
            legend_ids.add(int(ci["id"]))
            legend_ids.add(int(cj["id"]))

    assert set(color_map) == placeholder_colors, (
        f"legend mappings {sorted(color_map)} do not match placeholders {sorted(placeholder_colors)}"
    )

    sprite_defs: dict[int, dict] = {}
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
        sprite_defs[int(src_col)] = {
            "coords": coords,
            "h": int(sr1 - sr0 + 1),
            "w": int(sc1 - sc0 + 1),
            "bbox": (int(sr0), int(sc0), int(sr1), int(sc1)),
            "start": _center_of_bbox((int(sr0), int(sc0), int(sr1), int(sc1))),
        }

    return {
        "grid": g,
        "bg": int(bg),
        "frame_col": int(frame_col),
        "frame_bbox": (int(fr0), int(fc0), int(fr1), int(fc1)),
        "placeholders": placeholders,
        "placeholder_colors": set(map(int, placeholder_colors)),
        "color_map": color_map,
        "sprite_defs": sprite_defs,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rebuild the framed arrangement on the full canvas and draw color-coded motion trails.

    Input structure:
    - Background plus one grey rectangular frame whose interior contains solid placeholder blocks.
    - Outside the frame: one sprite template per placeholder color and one 3x4 legend per color.
    - In this fused variant the source sprites sit in exterior lanes that can visually connect to the frame.

    Rule:
    - Identify the frame, the placeholder colors, the matching same-colored source sprites, and the legends.
    - Keep the entire grid in the output, not just the frame crop.
    - Clear the frame interior, draw same-colored trajectories from each source sprite to every matching
      destination inside the frame, then stamp the source sprite shape into each destination and recolor it
      according to the legend mapping.

    Key insight:
    - The outside sprite still determines the shape, the legend still determines the final color, and the
      new trail reveals the source-to-destination movement across the full grid, so the exterior region now
      carries essential information rather than being disposable.
    """
    parsed = _parse_problem(grid)
    g = parsed["grid"]
    bg = int(parsed["bg"])
    fr0, fc0, fr1, fc1 = parsed["frame_bbox"]
    placeholders = parsed["placeholders"]
    color_map = parsed["color_map"]
    sprite_defs = parsed["sprite_defs"]

    out = g.copy()
    out[fr0 + 1 : fr1, fc0 + 1 : fc1] = bg

    for src_col in sorted(parsed["placeholder_colors"]):
        start = sprite_defs[src_col]["start"]
        for col, (pr0, pc0, pr1, pc1) in placeholders:
            if col != src_col:
                continue
            end = _center_of_bbox((fr0 + pr0, fc0 + pc0, fr0 + pr1, fc0 + pc1))
            for rr, cc in _path_points(start, end):
                if fr0 <= rr <= fr1 and fc0 <= cc <= fc1:
                    out[rr, cc] = int(src_col)
                else:
                    assert out[rr, cc] in (bg, src_col), "trajectory intersects unrelated outside object"
                    out[rr, cc] = int(src_col)

    for src_col, (pr0, pc0, pr1, pc1) in placeholders:
        tgt_col = int(color_map[src_col])
        sprite = sprite_defs[src_col]
        coords = sprite["coords"]
        sh = int(sprite["h"])
        sw = int(sprite["w"])
        assert (pr1 - pr0 + 1, pc1 - pc0 + 1) == (sh, sw), (
            f"placeholder size does not match sprite for color {src_col}"
        )
        off_r = fr0 + pr0
        off_c = fc0 + pc0
        for dr, dc in coords:
            out[off_r + dr, off_c + dc] = tgt_col

    return out


def _choose_structure() -> tuple[int, int, int] | None:
    """Pick (n, rows, cols) with rows also equal to the number of source colors."""
    options = []
    for n in range(*SPRITE_N_RANGE):
        for rows in range(*ROWS_RANGE):
            for cols in COLS_OPTIONS:
                total = rows * cols
                ch, cw = _content_extent(n, rows, cols)
                base_fh = ch + 4
                base_fw = cw + 4
                if 6 <= total <= 10 and base_fh <= MAX_FRAME_DIM and base_fw <= MAX_FRAME_DIM:
                    options.append((int(n), int(rows), int(cols)))
    return random.choice(options) if options else None


def _build_case() -> np.ndarray | None:
    struct = _choose_structure()
    if struct is None:
        return None
    n, rows, cols = struct

    bg = random.choice([c for c in range(10) if c != FRAME_COL])
    usable = [c for c in range(10) if c not in (bg, FRAME_COL)]
    src_cols = random.sample(usable, rows)
    remaining = [c for c in usable if c not in src_cols]
    tgt_cols = random.sample(remaining, rows)
    color_to_tgt = dict(zip(src_cols, tgt_cols))
    masks = {src: _sprite_mask(n) for src in src_cols}

    ch, cw = _content_extent(n, rows, cols)
    min_fh = max(MIN_FRAME_H, ch + 4)
    min_fw = max(MIN_FRAME_W, cw + 4)
    max_fh = min(MAX_FRAME_DIM, ch + 2 + FRAME_MARGIN_SLACK)
    max_fw = min(MAX_FRAME_DIM, cw + 2 + FRAME_MARGIN_SLACK)
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

    pad_top = random.randint(0, 2)
    pad_left = random.randint(0, 1)
    trail_gap = random.randint(2, 4)
    fr0 = pad_top + LEGEND_H + 1
    fc0 = pad_left + n + trail_gap

    legend_total_w = rows * LEGEND_W + (rows - 1) * LEGEND_GAP
    min_h = fr0 + frame_h
    min_w = max(fc0 + frame_w, pad_left + legend_total_w)
    if min_h >= CANVAS_H_RANGE[1] or min_w >= CANVAS_W_RANGE[1]:
        return None

    h = random.randint(max(CANVAS_H_RANGE[0], min_h), CANVAS_H_RANGE[1] - 1)
    w = random.randint(max(CANVAS_W_RANGE[0], min_w), CANVAS_W_RANGE[1] - 1)
    grid = np.full((h, w), bg, dtype=int)

    input_frame = _make_bordered_frame(frame_h, frame_w, bg)
    expected = grid.copy()
    paste(grid, input_frame, fr0, fc0)
    paste(expected, input_frame, fr0, fc0)

    row_color: dict[int, int] = {rr: src_cols[rr] for rr in range(rows)}
    for rr in range(rows):
        src = row_color[rr]
        for cc in range(cols):
            r0 = 1 + top_margin + rr * (n + 1)
            c0 = 1 + left_margin + cc * (n + 1)
            paste(grid, np.full((n, n), src, dtype=int), fr0 + r0, fc0 + c0)
            paste(expected, _colored_sprite(masks[src], color_to_tgt[src], bg), fr0 + r0, fc0 + c0, transparent=bg)

    legend_r0 = pad_top
    legend_c0 = max(pad_left, min(w - legend_total_w, fc0 + (frame_w - legend_total_w) // 2))
    for idx, src in enumerate(src_cols):
        lc = legend_c0 + idx * (LEGEND_W + LEGEND_GAP)
        paste(grid, _legend(src, color_to_tgt[src]), legend_r0, lc)
        paste(expected, _legend(src, color_to_tgt[src]), legend_r0, lc)

    for rr, src in enumerate(src_cols):
        sprite_patch = _colored_sprite(masks[src], src, bg)
        pr0 = 1 + top_margin + rr * (n + 1)
        abs_r0 = fr0 + pr0
        paste(grid, sprite_patch, abs_r0, pad_left, transparent=bg)
        paste(expected, sprite_patch, abs_r0, pad_left, transparent=bg)

        src_bbox = (abs_r0, pad_left, abs_r0 + n - 1, pad_left + n - 1)
        start = _center_of_bbox(src_bbox)
        for cc in range(cols):
            dst_r0 = fr0 + 1 + top_margin + rr * (n + 1)
            dst_c0 = fc0 + 1 + left_margin + cc * (n + 1)
            end = _center_of_bbox((dst_r0, dst_c0, dst_r0 + n - 1, dst_c0 + n - 1))
            for tr, tc in _path_points(start, end):
                expected[tr, tc] = int(src)
        for cc in range(cols):
            dst_r0 = fr0 + 1 + top_margin + rr * (n + 1)
            dst_c0 = fc0 + 1 + left_margin + cc * (n + 1)
            paste(expected, _colored_sprite(masks[src], color_to_tgt[src], bg), dst_r0, dst_c0, transparent=bg)

    try:
        solved = main(grid)
    except Exception:
        return None
    if not np.array_equal(solved, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _build_case()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid edb79dae combo input")
