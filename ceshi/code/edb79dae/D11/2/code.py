# Variant of edb79dae
# Dimension: D11 — 灰色框架的结构 (meta)
# Change: 单一矩形灰色框架，框架线条均匀，内部为操作区域... → 框架内有内部分隔线

import random

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, paste


# Design analysis
# - concretization:
#   The outer frame is still a single gray rectangle, but a full-height gray divider now
#   splits the interior into exactly two chambers. Placeholder blocks stay fully inside one
#   chamber, so stamping a sprite cannot cross the divider. The left chamber and the right
#   chamber each have their own legends outside the frame, placed on the matching side of
#   the frame. The same placeholder color can therefore recolor differently on the two sides.
# - behavior trace:
#   In the original task, if color 2 appears twice inside the frame, both placements must use
#   the same legend mapping 2->X. In this variant, a left-side color-2 block follows a left
#   legend 2->X while a right-side color-2 block follows a right legend 2->Y. The sprite shape
#   source is still global by color, but recoloring is now (region, color) dependent.
# - complexity calibration:
#   The divider creates two repeated subcases instead of many arbitrary zones. That is enough
#   to force regional reasoning while keeping the legend vocabulary small and repeated. Each
#   region uses every placeholder color at least once, and at least one color maps to different
#   targets across the two regions, making the partition visually consequential.
# - implicit assumptions reviewed:
#   The original solver assumed one contiguous non-gray interior and one global color map.
#   Those assumptions break here. The new parser therefore:
#   1. finds walkable connected components inside the frame to recover the two chambers;
#   2. assigns each placeholder component to a chamber;
#   3. builds one legend map per chamber side instead of one global map;
#   4. preserves internal gray cells when clearing the output.
#   It still assumes every chamber is rectangular in practice, every placeholder bbox fits in
#   one chamber, and legends are placed entirely left or right of the frame; the generator
#   constructs inputs to guarantee those properties.
# - placement legality:
#   The generator uses a fixed side-strip layout: legends live only in the left/right strips,
#   sprites live only in one top/bottom band above or below the frame, and the frame sits
#   between them with a one-cell gap to the legend strips. Inside the frame, the divider sits
#   between chamber rectangles and all placeholder slots have a one-cell bg margin to the
#   chamber walls, so no sprite can touch or cross the divider after stamping.
# - visual observability:
#   Humans can see the new rule because the divider is explicit, the frame output preserves it,
#   and both sides contain repeated placeholder colors whose outputs differ. No hidden ordering
#   by color id is used to decide the recoloring; side membership comes only from visible
#   position relative to the divider and the side-local legends.
# - neatness:
#   Outside elements are separated by function: legends only in side strips, sprites only in a
#   horizontal band. This avoids mixed overlays near the frame and keeps the two regional rule
#   clusters legible.
# - color inference:
#   Each region gets a full legend set for the colors used there, with small fixed 3x4 glyphs.
#   That keeps recoloring structural and local: same placeholder color + same sprite shape, but
#   different side legend => different final color.


FRAME_COL = 5
SPRITE_N_RANGE = (3, 6)
GRID_ROWS_RANGE = (2, 4)
REGION_COL_OPTIONS = (1, 2)
NUM_COLORS_MIN = 2
NUM_COLORS_MAX = 3
MIN_FRAME_H = 13
MAX_FRAME_H = 21
MAX_FRAME_W = 20
SIDE_STRIP_W = 5
LEGEND_H = 3
LEGEND_W = 4
LEGEND_GAP = 1
SPRITE_GAP = 1
MIN_CANVAS_H = 22
MAX_CANVAS_H = 30
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


def _bbox_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (ar1 < br0 or br1 < ar0 or ac1 < bc0 or bc1 < ac0)


def _legend_side(
    bbox: tuple[int, int, int, int],
    frame_bbox: tuple[int, int, int, int],
) -> str:
    _, fc0, _, fc1 = frame_bbox
    _, c0, _, c1 = bbox
    if c1 < fc0:
        return "left"
    if c0 > fc1:
        return "right"
    raise AssertionError(f"legend bbox {bbox} is not fully left or right of frame {frame_bbox}")


def _extract_instance(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    frame_col, frame_bbox = _find_frame(g, bg)
    fr0, fc0, fr1, fc1 = frame_bbox
    crop = g[fr0 : fr1 + 1, fc0 : fc1 + 1]
    H, W = crop.shape
    assert H >= 3 and W >= 3, f"frame too small for interior: {(H, W)}"

    walkable = crop != frame_col
    region_pts = components_mask(walkable, connectivity=4)
    assert len(region_pts) == 2, f"expected exactly two frame chambers, found {len(region_pts)}"
    region_pts.sort(key=lambda pts: bbox_coords(pts)[1])
    regions = []
    region_by_cell = np.full(crop.shape, -1, dtype=int)
    for ridx, pts in enumerate(region_pts):
        r0, c0, r1, c1 = bbox_coords(pts)
        mask = np.zeros(crop.shape, dtype=bool)
        mask[tuple(pts.T)] = True
        region_by_cell[mask] = ridx
        regions.append(
            {
                "id": ridx,
                "side": "left" if ridx == 0 else "right",
                "bbox": (r0, c0, r1, c1),
                "mask": mask,
            }
        )
    assert np.all(region_by_cell[walkable] >= 0), "unassigned cells inside frame"

    placeholder_colors = set(map(int, np.unique(crop))) - {bg, frame_col}
    assert placeholder_colors, "no placeholder colors found inside frame"

    placeholders = []
    phw: dict[int, tuple[int, int]] = {}
    colors_per_side: dict[str, set[int]] = {"left": set(), "right": set()}
    for col in sorted(placeholder_colors):
        comps = components_mask(crop == col, connectivity=4)
        assert comps, f"no placeholder component found for color {col}"
        dims = []
        for pts in comps:
            r0, c0, r1, c1 = bbox_coords(pts)
            regs = {int(region_by_cell[int(r), int(c)]) for r, c in pts}
            assert len(regs) == 1 and -1 not in regs, f"placeholder color {col} crosses a divider"
            ridx = regs.pop()
            side = regions[ridx]["side"]
            dims.append((r1 - r0 + 1, c1 - c0 + 1))
            placeholders.append(
                {
                    "src_col": int(col),
                    "bbox": (r0, c0, r1, c1),
                    "region_id": ridx,
                    "side": side,
                }
            )
            colors_per_side[side].add(int(col))
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

    region_maps: dict[str, dict[int, int]] = {"left": {}, "right": {}}
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
            side = _legend_side((r0, c0, r1, c1), frame_bbox)
            assert src not in region_maps[side], f"duplicate legend mapping for {side} color {src}"
            region_maps[side][int(src)] = int(tgt)
            legend_ids.add(ci["id"])
            legend_ids.add(cj["id"])

    for side in ("left", "right"):
        assert region_maps[side].keys() == colors_per_side[side], (
            f"{side} legend mappings {sorted(region_maps[side])} do not match placeholders {sorted(colors_per_side[side])}"
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
        for extra in all_candidates:
            if extra["id"] == base["id"]:
                continue
            er0, ec0, er1, ec1 = extra["bbox"]
            assert sr0 <= er0 and sr1 >= er1 and sc0 <= ec0 and sc1 >= ec1, (
                f"extra sprite component for color {src_col} is outside the base bbox"
            )
            pts_set.update(extra["pts"])
        coords = tuple(sorted((r - sr0, c - sc0) for r, c in pts_set))
        sprite_coords[src_col] = (coords, sr1 - sr0 + 1, sc1 - sc0 + 1)

    return {
        "grid": g,
        "bg": bg,
        "frame_col": frame_col,
        "frame_bbox": frame_bbox,
        "crop": crop,
        "regions": regions,
        "region_by_cell": region_by_cell,
        "placeholders": placeholders,
        "region_maps": region_maps,
        "sprite_coords": sprite_coords,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stamp recolored sprites into a partitioned frame using placeholder positions and side-local legends.

    Input structure:
    - bg background, one rectangular gray outer frame, and a gray divider inside the frame that splits the interior into left/right chambers.
    - Inside each chamber: solid placeholder blocks. Outside the frame: one sprite template per placeholder color plus two legend groups, one on each side of the frame.

    Rule:
    - Find the frame, recover the two chambers from the internal gray divider, and locate the placeholder blocks inside each chamber.
    - For each placeholder color, locate the matching-color sprite outside the frame and stamp that sprite at every placeholder position.
    - Recolor the stamped sprite by the legend that belongs to the placeholder's chamber side, while preserving both the outer border and the internal divider.

    Key insight:
    - Shape selection is still global by placeholder color, but recoloring is regional: the left legends control left-chamber placements and the right legends control right-chamber placements.
    """
    info = _extract_instance(np.asarray(grid, dtype=int))
    crop = info["crop"]
    bg = info["bg"]
    frame_col = info["frame_col"]
    region_by_cell = info["region_by_cell"]

    out = crop.copy()
    out[crop != frame_col] = bg

    H, W = crop.shape
    for ph in info["placeholders"]:
        src_col = ph["src_col"]
        pr0, pc0, pr1, pc1 = ph["bbox"]
        side = ph["side"]
        tgt_col = int(info["region_maps"][side][src_col])
        coords, sh, sw = info["sprite_coords"][src_col]
        assert (pr1 - pr0 + 1, pc1 - pc0 + 1) == (sh, sw), (
            f"placeholder size does not match sprite for color {src_col}"
        )
        assert pr0 + sh <= H and pc0 + sw <= W, f"sprite for color {src_col} does not fit at placeholder"
        ridx = ph["region_id"]
        for dr, dc in coords:
            rr, cc = pr0 + dr, pc0 + dc
            assert int(region_by_cell[rr, cc]) == ridx, f"sprite for color {src_col} crosses the divider"
            out[rr, cc] = tgt_col

    return out


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
    lo = max((n * n + 1) // 2, int(np.ceil(0.50 * n * n)))
    hi = min(int(np.floor(0.90 * n * n)), n * n - 1)
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


def _spread_positions(span: int, size: int, count: int, gap: int) -> list[int]:
    need = count * size + max(0, count - 1) * gap
    assert need <= span, f"cannot fit {count} blocks of size {size} in span {span}"
    extra = span - need
    extras = [0] * (count + 1)
    for _ in range(extra):
        extras[random.randrange(count + 1)] += 1
    positions = []
    cur = extras[0]
    for idx in range(count):
        positions.append(cur)
        cur += size
        if idx < count - 1:
            cur += gap + extras[idx + 1]
        else:
            cur += extras[idx + 1]
    return positions


def _deranged_targets(src_cols: list[int], usable: list[int]) -> list[int]:
    options = [c for c in usable]
    for _ in range(200):
        tgts = random.sample(options, len(src_cols))
        if all(t != s for s, t in zip(src_cols, tgts)):
            return tgts
    raise RuntimeError("failed to draw distinct target colors")


def _region_dims(n: int, rows: int, cols: int) -> tuple[int, int]:
    return rows * n + (rows - 1) + 2, cols * n + (cols - 1) + 2


def _choose_structure() -> tuple[int, int, int, int] | None:
    options = []
    for n in range(*SPRITE_N_RANGE):
        for rows in range(*GRID_ROWS_RANGE):
            for left_cols in REGION_COL_OPTIONS:
                for right_cols in REGION_COL_OPTIONS:
                    total_slots = rows * (left_cols + right_cols)
                    if not (6 <= total_slots <= 10):
                        continue
                    region_h, left_w = _region_dims(n, rows, left_cols)
                    _, right_w = _region_dims(n, rows, right_cols)
                    frame_h = region_h + 2
                    frame_w = left_w + right_w + 3
                    if MIN_FRAME_H <= frame_h <= MAX_FRAME_H and frame_w <= MAX_FRAME_W:
                        options.append((n, rows, left_cols, right_cols))
    return random.choice(options) if options else None


def _build_partitioned_frame(frame_h: int, frame_w: int, sep_c: int, bg: int) -> np.ndarray:
    frame = np.full((frame_h, frame_w), bg, dtype=int)
    frame[0, :] = frame[-1, :] = FRAME_COL
    frame[:, 0] = frame[:, -1] = FRAME_COL
    frame[1:-1, sep_c] = FRAME_COL
    return frame


def _slot_coords(rows: int, cols: int) -> list[tuple[int, int]]:
    return [(r, c) for r in range(rows) for c in range(cols)]


def _assign_region_colors(rows: int, cols: int, src_cols: list[int]) -> dict[tuple[int, int], int]:
    slots = _slot_coords(rows, cols)
    assert len(slots) >= len(src_cols), "not enough region slots to show all colors"
    random.shuffle(slots)
    mapping = {}
    src_order = src_cols[:]
    random.shuffle(src_order)
    for slot, col in zip(slots[: len(src_order)], src_order):
        mapping[slot] = int(col)
    for slot in slots[len(src_order) :]:
        mapping[slot] = int(random.choice(src_cols))
    return mapping


def _variant_metrics(grid: np.ndarray) -> dict:
    info = _extract_instance(grid)
    crop = info["crop"]
    frame_col = info["frame_col"]
    inner_gray = int(np.sum((crop == frame_col)[1:-1, 1:-1]))
    left_map = info["region_maps"]["left"]
    right_map = info["region_maps"]["right"]
    shared = sorted(set(left_map) & set(right_map))
    differing = sum(int(left_map[c] != right_map[c]) for c in shared)
    return {
        "region_count": len(info["regions"]),
        "inner_gray_cells": inner_gray,
        "shared_color_count": len(shared),
        "differing_shared_mappings": differing,
    }


def _gen_attempt() -> np.ndarray | None:
    structure = _choose_structure()
    if structure is None:
        return None
    n, rows, left_cols, right_cols = structure

    bg = random.choice([c for c in range(10) if c != FRAME_COL])
    usable = [c for c in range(10) if c not in (bg, FRAME_COL)]

    k_max = min(NUM_COLORS_MAX, rows * left_cols, rows * right_cols)
    if k_max < NUM_COLORS_MIN:
        return None
    k = random.randint(NUM_COLORS_MIN, k_max)
    src_cols = random.sample(usable, k)
    left_tgts = _deranged_targets(src_cols, usable)
    for _ in range(200):
        right_tgts = _deranged_targets(src_cols, usable)
        if any(a != b for a, b in zip(left_tgts, right_tgts)):
            break
    else:
        return None
    region_maps = {
        "left": dict(zip(src_cols, left_tgts)),
        "right": dict(zip(src_cols, right_tgts)),
    }
    masks = {src: _sprite_mask(n) for src in src_cols}

    region_h, left_w = _region_dims(n, rows, left_cols)
    _, right_w = _region_dims(n, rows, right_cols)
    frame_h = region_h + 2
    frame_w = left_w + right_w + 3
    sep_c = 1 + left_w

    input_frame = _build_partitioned_frame(frame_h, frame_w, sep_c, bg)
    output_frame = _build_partitioned_frame(frame_h, frame_w, sep_c, bg)

    left_cells = _assign_region_colors(rows, left_cols, src_cols)
    right_cells = _assign_region_colors(rows, right_cols, src_cols)

    for (rr, cc), src in left_cells.items():
        r0 = 2 + rr * (n + 1)
        c0 = 2 + cc * (n + 1)
        paste(input_frame, np.full((n, n), src, dtype=int), r0, c0)
        paste(output_frame, _colored_sprite(masks[src], region_maps["left"][src], bg), r0, c0, transparent=bg)
    for (rr, cc), src in right_cells.items():
        r0 = 2 + rr * (n + 1)
        c0 = sep_c + 2 + cc * (n + 1)
        paste(input_frame, np.full((n, n), src, dtype=int), r0, c0)
        paste(output_frame, _colored_sprite(masks[src], region_maps["right"][src], bg), r0, c0, transparent=bg)

    min_h = max(MIN_CANVAS_H, frame_h + n + 1)
    if min_h > MAX_CANVAS_H:
        return None
    canvas_h = random.randint(min_h, MAX_CANVAS_H)
    canvas_w = frame_w + 2 * SIDE_STRIP_W
    if canvas_w > 30:
        return None

    band_h = n + 1
    slack = canvas_h - frame_h - band_h
    assert slack >= 0
    top_band = random.random() < 0.5
    extra_top = random.randint(0, slack)
    if top_band:
        frame_r = band_h + extra_top
    else:
        frame_r = extra_top
    frame_c = SIDE_STRIP_W

    grid = np.full((canvas_h, canvas_w), bg, dtype=int)
    paste(grid, input_frame, frame_r, frame_c)

    left_x = 0
    right_x = frame_c + frame_w + 1
    left_order = src_cols[:]
    right_order = src_cols[:]
    random.shuffle(left_order)
    random.shuffle(right_order)
    left_ys = _spread_positions(canvas_h, LEGEND_H, len(left_order), LEGEND_GAP)
    right_ys = _spread_positions(canvas_h, LEGEND_H, len(right_order), LEGEND_GAP)
    for y, src in zip(left_ys, left_order):
        paste(grid, _legend(src, region_maps["left"][src]), y, left_x)
    for y, src in zip(right_ys, right_order):
        paste(grid, _legend(src, region_maps["right"][src]), y, right_x)

    if top_band:
        band_r0, band_r1 = 0, frame_r - 1
    else:
        band_r0, band_r1 = frame_r + frame_h, canvas_h - 1
    band_height = band_r1 - band_r0 + 1
    if band_height < n:
        return None
    sprite_y = band_r0 + random.randint(0, band_height - n)
    sprite_xs = _spread_positions(frame_w, n, len(src_cols), SPRITE_GAP)
    sprite_order = src_cols[:]
    random.shuffle(sprite_order)
    for x_off, src in zip(sprite_xs, sprite_order):
        paste(grid, _colored_sprite(masks[src], src, bg), sprite_y, frame_c + x_off, transparent=bg)

    try:
        if not np.array_equal(main(grid), output_frame):
            return None
        metrics = _variant_metrics(grid)
        if metrics["region_count"] != 2 or metrics["inner_gray_cells"] <= 0:
            return None
        if metrics["differing_shared_mappings"] <= 0:
            return None
    except Exception:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        out = _gen_attempt()
        if out is not None:
            return out
    raise RuntimeError("failed to generate valid edb79dae D11 variant input")
