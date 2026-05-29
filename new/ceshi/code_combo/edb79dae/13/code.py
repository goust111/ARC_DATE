# Combo variant of edb79dae (index 13)
# combo_spec: D11/2+D5/2
# Name: 框架内分隔线+镜像翻转放置

import random

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, paste


# Design analysis
# - concretization:
#   The outer object is still one rectangular gray frame, but a full-height gray divider splits
#   the interior into exactly two chambers. Each chamber contains a 2x2 layout of 3x3 solid
#   placeholder blocks. Outside the frame there is still one sprite template per placeholder
#   color, plus legends; however legends are now chamber-local, with one legend stack on the
#   left side of the frame and one on the right side. Placement is no longer "copy sprite as-is":
#   the same sprite is mirrored according to the placeholder position relative to the center of
#   its own chamber. Top-left = unchanged, top-right = horizontal flip, bottom-left = vertical
#   flip, bottom-right = both flips.
# - behavior trace:
#   Example: suppose color 2 has an asymmetric 3x3 sprite and appears once in each slot of the
#   left chamber. In the original task, all four outputs would use the same orientation and the
#   same recolor. In the D11-only variant, the left and right chambers could recolor color 2
#   differently, but all placements inside one chamber would still share the same orientation.
#   In this combo, a left-chamber top-right block uses the left legend map for recolor and also
#   becomes the horizontal mirror of the sprite; a right-chamber bottom-left block uses the right
#   legend map and becomes the vertical mirror relative to the right-chamber center. The divider
#   therefore changes both "which legend applies" and "which center defines the reflection axes".
# - complexity calibration:
#   The generator fixes the interior to two repeated 2x2 chambers. That is small enough to keep
#   the visual grammar legible, but rich enough to expose all four chamber-local reflection cases
#   in every sample. It also forces one focus color to repeat in three slots per chamber so the
#   mirrored orientations are easy to compare by eye. Sprite masks are rejected if they are
#   horizontally or vertically symmetric, preventing silent collapse back to the original task.
# - implicit assumptions reviewed:
#   The original solver assumed one connected non-gray interior and one global legend map. That
#   breaks once the divider appears. The parser now assumes:
#   1. `crop != frame_col` has exactly two connected components, which are the chambers;
#   2. every placeholder component lies fully inside one chamber and never crosses the divider;
#   3. legends are fully left or fully right of the frame, so side membership is visible;
#   4. each placeholder color still has exactly one size-matched external sprite template;
#   5. chamber-local reflection is meaningful because chamber bboxes are rectangular and have
#      usable left/right and top/bottom halves. The generator constructs inputs to guarantee all
#      of these conditions.
# - placement legality:
#   Each chamber has one-cell horizontal padding and two-cell vertical padding around the 2x2
#   placeholder grid, so every 3x3 sprite stays strictly inside its chamber after any flip.
#   Legends use dedicated left/right strips; sprite templates use a separate top or bottom band.
#   These three functional zones do not overlap, and every pasted patch stays inside the canvas.
# - visual observability:
#   A human can infer the combo because the divider is explicit, both chambers preserve the same
#   repeated geometry, and the same source color appears multiple times with visibly different
#   orientations inside one chamber. At the same time, at least one shared color recolors
#   differently between the two legend stacks, so the divider has visible semantic meaning beyond
#   being a wall. No hidden sorting by color id or generation order is used.
# - neatness and inference:
#   The scene stays organized: legends in side strips, sprites in one horizontal band, frame in
#   the middle. Color changes remain structurally indicated by local legends, while orientation
#   changes remain structurally indicated by chamber position. The two signals reinforce rather
#   than obscure each other.


FRAME_COL = 5
SPRITE_N = 3
ROWS = 2
REGION_COLS = 2
INNER_GAP = 1
CHAMBER_TOP_PAD = 2
CHAMBER_SIDE_PAD = 1
LEGEND_H = 3
LEGEND_W = 4
LEGEND_GAP = 1
SPRITE_GAP = 1
SIDE_STRIP_W = LEGEND_W
CANVAS_H_RANGE = (22, 31)
CANVAS_W_OPTIONS = (29, 30)
NUM_COLORS_OPTIONS = (2,)
GEN_RETRIES = 400

CHAMBER_H = ROWS * SPRITE_N + (ROWS - 1) * INNER_GAP + 2 * CHAMBER_TOP_PAD
CHAMBER_W = REGION_COLS * SPRITE_N + (REGION_COLS - 1) * INNER_GAP + 2 * CHAMBER_SIDE_PAD
FRAME_H = CHAMBER_H + 2
FRAME_W = CHAMBER_W * 2 + 3
SEP_C = 1 + CHAMBER_W
FOCUS_SIGS = ((False, False), (False, True), (True, False))


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


def _region_flips(
    region_bbox: tuple[int, int, int, int],
    placeholder_bbox: tuple[int, int, int, int],
) -> tuple[bool, bool]:
    """
    Decide vertical/horizontal flips from the placeholder center relative to its chamber center.
    Equality stays on the non-flipped side.
    """
    rr0, rc0, rr1, rc1 = region_bbox
    pr0, pc0, pr1, pc1 = placeholder_bbox
    region_ctr_r = (rr0 + rr1) / 2.0
    region_ctr_c = (rc0 + rc1) / 2.0
    box_ctr_r = (pr0 + pr1) / 2.0
    box_ctr_c = (pc0 + pc1) / 2.0
    flip_v = box_ctr_r > region_ctr_r
    flip_h = box_ctr_c > region_ctr_c
    return flip_v, flip_h


def _mask_from_points(
    pts_set: set[tuple[int, int]],
    sr0: int,
    sc0: int,
    height: int,
    width: int,
) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    for r, c in pts_set:
        mask[r - sr0, c - sc0] = True
    return mask


def _orient_mask(mask: np.ndarray, flip_v: bool, flip_h: bool) -> np.ndarray:
    out = mask
    if flip_h:
        out = np.fliplr(out)
    if flip_v:
        out = np.flipud(out)
    return out


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

    sprite_masks: dict[int, np.ndarray] = {}
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
        sprite_masks[src_col] = _mask_from_points(pts_set, sr0, sc0, sr1 - sr0 + 1, sc1 - sc0 + 1)

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
        "sprite_masks": sprite_masks,
        "colors_per_side": colors_per_side,
    }


def _render_from_info(info: dict, *, local_region_flips: bool) -> np.ndarray:
    crop = info["crop"]
    bg = info["bg"]
    frame_col = info["frame_col"]
    region_by_cell = info["region_by_cell"]
    H, W = crop.shape

    out = crop.copy()
    out[crop != frame_col] = bg

    for ph in info["placeholders"]:
        src_col = ph["src_col"]
        pr0, pc0, pr1, pc1 = ph["bbox"]
        ridx = ph["region_id"]
        side = ph["side"]
        tgt_col = int(info["region_maps"][side][src_col])
        mask = info["sprite_masks"][src_col]
        sh, sw = mask.shape
        assert (pr1 - pr0 + 1, pc1 - pc0 + 1) == (sh, sw), (
            f"placeholder size does not match sprite for color {src_col}"
        )
        if local_region_flips:
            ref_bbox = info["regions"][ridx]["bbox"]
        else:
            ref_bbox = (0, 0, H - 1, W - 1)
        flip_v, flip_h = _region_flips(ref_bbox, ph["bbox"])
        oriented = _orient_mask(mask, flip_v, flip_h)
        for dr, dc in zip(*np.where(oriented)):
            rr, cc = pr0 + int(dr), pc0 + int(dc)
            assert int(region_by_cell[rr, cc]) == ridx, f"sprite for color {src_col} crosses the divider"
            out[rr, cc] = tgt_col
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill a partitioned frame with recolored sprites whose orientation is decided per chamber.

    Input structure:
    - A gray rectangular frame encloses solid placeholder blocks, and a gray divider splits the frame interior into left/right chambers.
    - Outside the frame there is one sprite template per placeholder color plus two legend groups, one fully left of the frame and one fully right of it.

    Rule:
    - Detect the frame, recover the two chambers from the internal divider, and assign every placeholder block to exactly one chamber.
    - For each placeholder color, find the matching external sprite template. Stamp it into each placeholder bbox, but first mirror it according to that bbox center relative to its own chamber center: right half means horizontal flip, bottom half means vertical flip.
    - Recolor the stamped sprite with the legend mapping from the placeholder's chamber side, while keeping both the outer border and the internal divider gray.

    Key insight:
    - The divider is not only a barrier for placement; it also defines two separate reference systems. The same placeholder color chooses the same base sprite globally, but chamber side chooses the recolor and chamber-local position chooses the mirrored orientation.
    """
    info = _extract_instance(np.asarray(grid, dtype=int))
    return _render_from_info(info, local_region_flips=True)


def _legend(src: int, tgt: int) -> np.ndarray:
    return np.array(
        [
            [src, src, tgt, tgt],
            [src, src, src, tgt],
            [src, src, tgt, tgt],
        ],
        dtype=int,
    )


def _raw_sprite_mask(n: int) -> np.ndarray:
    target = random.randint(max(4, (n * n + 1) // 2), n * n - 1)

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


def _sprite_mask(n: int) -> np.ndarray:
    for _ in range(120):
        mask = _raw_sprite_mask(n)
        if np.array_equal(mask, np.fliplr(mask)):
            continue
        if np.array_equal(mask, np.flipud(mask)):
            continue
        return mask
    raise RuntimeError(f"failed to build asymmetric {n}x{n} sprite")


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


def _build_partitioned_frame(bg: int) -> np.ndarray:
    frame = np.full((FRAME_H, FRAME_W), bg, dtype=int)
    frame[0, :] = frame[-1, :] = FRAME_COL
    frame[:, 0] = frame[:, -1] = FRAME_COL
    frame[1:-1, SEP_C] = FRAME_COL
    return frame


def _region_bbox(side: str) -> tuple[int, int, int, int]:
    if side == "left":
        return (1, 1, FRAME_H - 2, SEP_C - 1)
    if side == "right":
        return (1, SEP_C + 1, FRAME_H - 2, FRAME_W - 2)
    raise AssertionError(f"unknown side {side}")


def _slot_bbox(side: str, rr: int, cc: int) -> tuple[int, int, int, int]:
    base_c = 1 if side == "left" else SEP_C + 1
    r0 = 1 + CHAMBER_TOP_PAD + rr * (SPRITE_N + INNER_GAP)
    c0 = base_c + CHAMBER_SIDE_PAD + cc * (SPRITE_N + INNER_GAP)
    return r0, c0, r0 + SPRITE_N - 1, c0 + SPRITE_N - 1


def _slot_signature_map(side: str) -> dict[tuple[int, int], tuple[bool, bool]]:
    region_bbox = _region_bbox(side)
    sigs = {}
    for rr in range(ROWS):
        for cc in range(REGION_COLS):
            sigs[(rr, cc)] = _region_flips(region_bbox, _slot_bbox(side, rr, cc))
    return sigs


def _pick_region_targets(src_cols: list[int], usable: list[int]) -> list[int]:
    for _ in range(200):
        tgts = random.sample(usable, len(src_cols))
        if all(t != s for s, t in zip(src_cols, tgts)):
            return tgts
    raise RuntimeError("failed to draw valid target colors")


def _assign_side_cells(src_cols: list[int], side: str) -> dict[tuple[int, int], int]:
    slot_sigs = _slot_signature_map(side)
    by_sig = {sig: slot for slot, sig in slot_sigs.items()}
    focus = src_cols[0]
    mapping = {
        by_sig[(False, False)]: focus,
        by_sig[(False, True)]: focus,
        by_sig[(True, False)]: focus,
    }
    leftovers = [slot for slot in sorted(slot_sigs) if slot not in mapping]
    random.shuffle(leftovers)
    for idx, col in enumerate(src_cols[1:]):
        mapping[leftovers[idx]] = col
    for slot in leftovers[len(src_cols) - 1 :]:
        mapping[slot] = focus
    return mapping


def _variant_metrics(grid: np.ndarray) -> dict:
    info = _extract_instance(grid)
    local = _render_from_info(info, local_region_flips=True)
    global_ref = _render_from_info(info, local_region_flips=False)
    left_map = info["region_maps"]["left"]
    right_map = info["region_maps"]["right"]
    shared = sorted(set(left_map) & set(right_map))
    differing = sum(int(left_map[c] != right_map[c]) for c in shared)

    sigs_by_side = {"left": set(), "right": set()}
    sigs_by_color: dict[int, set[tuple[bool, bool]]] = {}
    for ph in info["placeholders"]:
        sig = _region_flips(info["regions"][ph["region_id"]]["bbox"], ph["bbox"])
        sigs_by_side[ph["side"]].add(sig)
        sigs_by_color.setdefault(ph["src_col"], set()).add(sig)

    return {
        "region_count": len(info["regions"]),
        "local_vs_global_delta": int(np.sum(local != global_ref)),
        "shared_color_count": len(shared),
        "differing_shared_mappings": differing,
        "left_sig_count": len(sigs_by_side["left"]),
        "right_sig_count": len(sigs_by_side["right"]),
        "max_color_sig_count": max(len(v) for v in sigs_by_color.values()),
    }


def _gen_attempt() -> np.ndarray | None:
    bg = random.choice([c for c in range(10) if c != FRAME_COL])
    usable = [c for c in range(10) if c not in (bg, FRAME_COL)]

    k = random.choice(NUM_COLORS_OPTIONS)
    if k > len(usable):
        return None
    src_cols = random.sample(usable, k)
    left_tgts = _pick_region_targets(src_cols, usable)
    for _ in range(200):
        right_tgts = _pick_region_targets(src_cols, usable)
        if any(a != b for a, b in zip(left_tgts, right_tgts)):
            break
    else:
        return None
    region_maps = {
        "left": dict(zip(src_cols, left_tgts)),
        "right": dict(zip(src_cols, right_tgts)),
    }
    masks = {src: _sprite_mask(SPRITE_N) for src in src_cols}

    input_frame = _build_partitioned_frame(bg)
    output_frame = _build_partitioned_frame(bg)

    left_cells = _assign_side_cells(src_cols, "left")
    right_cells = _assign_side_cells(src_cols, "right")
    for side, cells in (("left", left_cells), ("right", right_cells)):
        region_bbox = _region_bbox(side)
        for (rr, cc), src in cells.items():
            r0, c0, r1, c1 = _slot_bbox(side, rr, cc)
            paste(input_frame, np.full((SPRITE_N, SPRITE_N), src, dtype=int), r0, c0)
            flip_v, flip_h = _region_flips(region_bbox, (r0, c0, r1, c1))
            out_mask = _orient_mask(masks[src], flip_v, flip_h)
            paste(
                output_frame,
                _colored_sprite(out_mask, region_maps[side][src], bg),
                r0,
                c0,
                transparent=bg,
            )

    canvas_h = random.randrange(*CANVAS_H_RANGE)
    canvas_w = random.choice(CANVAS_W_OPTIONS)
    frame_c = canvas_w - FRAME_W - SIDE_STRIP_W
    assert frame_c in (4, 5)
    band_h = SPRITE_N + 1
    max_frame_r = canvas_h - FRAME_H
    if max_frame_r < band_h:
        return None
    top_band = random.random() < 0.5
    if top_band:
        frame_r = random.randint(band_h, max_frame_r)
        sprite_r0, sprite_r1 = 0, frame_r - SPRITE_N
    else:
        frame_r = random.randint(0, max_frame_r - band_h)
        sprite_r0, sprite_r1 = frame_r + FRAME_H, canvas_h - SPRITE_N
    if sprite_r0 > sprite_r1:
        return None

    grid = np.full((canvas_h, canvas_w), bg, dtype=int)
    paste(grid, input_frame, frame_r, frame_c)

    left_order = src_cols[:]
    right_order = src_cols[:]
    random.shuffle(left_order)
    random.shuffle(right_order)
    for y, src in zip(_spread_positions(canvas_h, LEGEND_H, len(left_order), LEGEND_GAP), left_order):
        paste(grid, _legend(src, region_maps["left"][src]), y, 0)
    right_x = frame_c + FRAME_W
    for y, src in zip(_spread_positions(canvas_h, LEGEND_H, len(right_order), LEGEND_GAP), right_order):
        paste(grid, _legend(src, region_maps["right"][src]), y, right_x)

    sprite_y = random.randint(sprite_r0, sprite_r1)
    sprite_order = src_cols[:]
    random.shuffle(sprite_order)
    for x_off, src in zip(_spread_positions(FRAME_W, SPRITE_N, len(sprite_order), SPRITE_GAP), sprite_order):
        paste(grid, _colored_sprite(masks[src], src, bg), sprite_y, frame_c + x_off, transparent=bg)

    try:
        if not np.array_equal(main(grid), output_frame):
            return None
        metrics = _variant_metrics(grid)
        if metrics["region_count"] != 2:
            return None
        if metrics["left_sig_count"] != 4 or metrics["right_sig_count"] != 4:
            return None
        if metrics["max_color_sig_count"] < 3:
            return None
        if metrics["differing_shared_mappings"] <= 0:
            return None
        if metrics["local_vs_global_delta"] <= 0:
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
