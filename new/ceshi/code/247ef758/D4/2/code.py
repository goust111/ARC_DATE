# Variant of 247ef758
# Dimension: D4 — 变换：印刻与擦除行为 (transform)
# Change: 擦除左侧原始精灵，在目标位置居中印刻精灵图案，像素超出框架内部时裁剪（clipping）... → 不擦除原始精灵（保留源）

from __future__ import annotations

from collections import defaultdict
import random

import numpy as np

from arc_utils import get_bg, label_mask


# Design analysis for the D4 keep_source variant
# ----------------------------------------------
# a) Concrete behavior trace
#    Input still has three regions:
#    - left strip: 2-3 mover sprites plus optional distractor sprites
#    - one solid divider column
#    - right full-height frame whose opposite edges carry mirrored color markers
#    Original behavior:
#    - detect each mover color on the left
#    - derive its target rows/cols from matching edge markers
#    - erase the left-side mover pixels to bg
#    - stamp the same sprite, centered at each row×col target inside the frame
#    Variant behavior:
#    - everything above stays the same except the erase step disappears
#    - the output now shows both the original left sprite and its copied stamps
#    Small trace:
#    - if a red 3-pixel sprite sits in cols 0..2 and red markers indicate one target
#      inside the frame, the old output has only the framed copy; the new output has
#      the original red sprite still present on the left plus that framed copy.
#
# b) Complexity calibration
#    We keep 2-3 movers, 1-2 row marks, 1-2 col marks, and optional distractor.
#    This preserves the original combinatorial structure. The new visual clue is
#    strong because every mover visibly survives on the left while also appearing
#    in the frame; this is enough to distinguish "copy" from "move" without adding
#    unrelated freedoms.
#
# c) Implicit-assumption audit of original main()
#    - frame detection via largest enclosed bg component: unchanged
#    - divider detection on the left: unchanged
#    - sprite extraction in left strip: unchanged
#    - target derivation from opposite-edge color equality: unchanged
#    - `out = g.copy()`: still correct, and now becomes essential because it keeps
#      the original sprites by default
#    - the original "erase originals of moved sprites" loop relied on the old D4
#      assumption that movers should disappear from the source strip. In this variant
#      that loop is wrong and must be removed. No later code requires those pixels to
#      be bg because stamping is restricted to the frame interior.
#    - bottom-to-top stamping order: unchanged; it still governs overlap inside the
#      frame only
#
# d) Placement legality
#    The left strip and frame interior are disjoint, so preserving the source cannot
#    create source/target overlap. Generator still ensures:
#    - all sprites fit inside the left strip
#    - mover target centers are valid interior cells
#    - row/col markers are not reused across mover colors
#    - main(input) runs successfully
#
# e) Visual observability
#    Humans can infer the new rule from 2-4 examples because the same colored shape
#    appears twice: once unchanged on the left and once copied into the frame. This
#    difference is visible without relying on hidden numeric properties.
#
# f) Visual cleanliness
#    We keep the original clean region separation. Preserved sources remain confined
#    to the left strip; copied stamps remain confined to the frame interior.
#
# g) Color inference
#    No new color mapping is introduced. Colors still identify which sprite matches
#    which edge markers.
#
# Additional generator-side checks for this variant
#    - at least one mover exists (already guaranteed: 2-3 movers)
#    - output differs from input because stamping occurs
#    - every mover source pixel remains unchanged in the output, so keep_source is
#      not merely intended but actually instantiated by generated samples


def largest_enclosed_component_bbox(mask: np.ndarray, *, connectivity: int = 4) -> tuple[int, int, int, int] | None:
    """
    Local helper (only used by this solver): bbox of largest connected component in
    `mask` that does NOT touch the grid border.
    """
    H, W = mask.shape
    best_area = -1
    best_bbox: tuple[int, int, int, int] | None = None

    lab, n = label_mask(mask.astype(bool), connectivity=int(connectivity))
    for k in range(1, int(n) + 1):
        coords = np.argwhere(lab == int(k))
        if coords.size == 0:
            continue
        rs = coords[:, 0]
        cs = coords[:, 1]
        if bool((rs == 0).any() or (rs == int(H - 1)).any() or (cs == 0).any() or (cs == int(W - 1)).any()):
            continue
        area = int(coords.shape[0])
        if area > best_area:
            best_area = area
            r0, c0 = coords.min(axis=0)
            r1, c1 = coords.max(axis=0)
            best_bbox = (int(r0), int(c0), int(r1), int(c1))

    return best_bbox


def main(grid: np.ndarray) -> np.ndarray:
    """
    Copy colored sprites from the left strip into a framed area at marker-defined targets.

    Input structure:
    - bg background; rectangular frame on right with colored markers on its edges;
      sprites in a narrow left strip separated by a constant-color divider column.

    Rule:
    - For each color, target rows/cols are where both opposite frame edges share
      that color. Targets are the Cartesian product.
    - Keep each source sprite in place and stamp it centered at every target.
      Higher sprites cover lower ones inside the frame.

    Key insight: One sprite replicates to a grid of row × column marker intersections,
    but unlike the original task it is copied rather than moved.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

    # --- frame interior: largest bg component not touching boundary ---
    inner_bbox = largest_enclosed_component_bbox(g == int(bg), connectivity=4)
    assert not (inner_bbox is None)
    ir0, ic0, ir1, ic1 = map(int, inner_bbox)
    fr0, fc0, fr1, fc1 = int(ir0 - 1), int(ic0 - 1), int(ir1 + 1), int(ic1 + 1)

    # --- divider columns (fixed) on the left side ---
    divider_mask = np.zeros((H, W), dtype=bool)
    for c in range(0, max(0, int(fc0))):
        colv = g[:, int(c)]
        non = colv[colv != int(bg)]
        if int(len(non)) >= int(H - 1) and len(set(map(int, non))) == 1:
            divider_mask[:, int(c)] = True

    left_cols = np.arange(W)[None, :] < int(fc0)

    # --- sprites: per-color 8-connected components in left area (excluding divider) ---
    sprites = []
    for col in sorted(set(map(int, g.flatten())) - {int(bg)}):
        mask = (g == int(col)) & left_cols & (~divider_mask)
        if not bool(mask.any()):
            continue
        slab, sn = label_mask(mask, connectivity=8)
        for sid in range(1, int(sn) + 1):
            pts = np.argwhere(slab == int(sid))
            r0, c0 = pts.min(axis=0)
            r1, c1 = pts.max(axis=0)
            r0, c0, r1, c1 = map(int, (r0, c0, r1, c1))
            m = slab[int(r0): int(r1) + 1, int(c0): int(c1) + 1] == int(sid)
            sprites.append({"color": int(col), "r0": int(r0), "c0": int(c0), "r1": int(r1), "c1": int(c1), "mask": m})

    move_colors = {int(s["color"]) for s in sprites}

    # --- marker-derived targets (restricted to colors that actually appear as sprites) ---
    row_marks: dict[int, list[int]] = defaultdict(list)
    col_marks: dict[int, list[int]] = defaultdict(list)
    for r in range(int(ir0), int(ir1) + 1):
        cl = int(g[int(r), int(fc0)])
        cr = int(g[int(r), int(fc1)])
        if cl == cr and cl != int(bg) and cl in move_colors:
            row_marks[int(cl)].append(int(r))
    for c in range(int(ic0), int(ic1) + 1):
        ct = int(g[int(fr0), int(c)])
        cb = int(g[int(fr1), int(c)])
        if ct == cb and ct != int(bg) and ct in move_colors:
            col_marks[int(ct)].append(int(c))

    targets: dict[int, list[tuple[int, int]]] = {}
    for col in set(row_marks.keys()) & set(col_marks.keys()):
        pts = {(int(r), int(c)) for r in row_marks[int(col)] for c in col_marks[int(col)]}
        targets[int(col)] = sorted(pts)

    out = g.copy()
    movers = [s for s in sprites if int(s["color"]) in targets]

    # Bottom-to-top ordering: lower sprites first, upper sprites last (covering).
    movers.sort(key=lambda s: (-((int(s["r0"]) + int(s["r1"])) / 2.0), int(s["c0"])))

    for s in movers:
        col = int(s["color"])
        cr = (int(s["r0"]) + int(s["r1"])) // 2
        cc = (int(s["c0"]) + int(s["c1"])) // 2
        pix = [(int(s["r0"]) + int(rr), int(s["c0"]) + int(cc2)) for rr, cc2 in np.argwhere(s["mask"])]

        for tr, tc in targets[int(col)]:
            dr = int(tr - cr)
            dc = int(tc - cc)
            for r, c in pix:
                rr = int(r + dr)
                cc3 = int(c + dc)
                if int(ir0) <= rr <= int(ir1) and int(ic0) <= cc3 <= int(ic1):
                    out[rr, cc3] = int(col)

    return out


def _pick_spaced(candidates: list[int], n: int, min_spacing: int) -> list[int] | None:
    """Pick up to n values from candidates with at least min_spacing between any two."""
    if not candidates:
        return None
    if n == 1:
        return [random.choice(candidates)]
    random.shuffle(candidates)
    picked: list[int] = []
    for v in candidates:
        if all(abs(v - p) >= min_spacing for p in picked):
            picked.append(v)
            if len(picked) == n:
                return picked
    return picked if picked else None


def _make_sprite(color: int, max_h: int, max_w: int) -> np.ndarray | None:
    """Generate a single-color 8-connected sprite cropped to its bounding box."""
    sh = min(random.choice([3, 3, 3, 4, 5]), max_h)
    sw = min(random.choice([3, 3, 3, 4, 5]), max_w)
    if sh < 1 or sw < 1:
        return None
    n_pixels = random.randint(3, min(9, sh * sw))
    pixels = {(sh // 2, sw // 2)}
    r, c = sh // 2, sw // 2
    for _ in range(n_pixels * 30):
        if len(pixels) >= n_pixels:
            break
        dr, dc = random.choice(
            [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]
        )
        nr, nc = r + dr, c + dc
        if 0 <= nr < sh and 0 <= nc < sw:
            pixels.add((nr, nc))
            r, c = nr, nc
    rows = [p[0] for p in pixels]
    cols = [p[1] for p in pixels]
    min_r, max_r = min(rows), max(rows)
    min_c, max_c = min(cols), max(cols)
    sprite = np.zeros((max_r - min_r + 1, max_c - min_c + 1), dtype=int)
    for pr, pc in pixels:
        sprite[pr - min_r, pc - min_c] = color
    return sprite


def _gen_attempt() -> np.ndarray | None:
    bg = 0

    # Dimensions — frame spans full grid height.
    interior_h = random.randint(8, 13)
    interior_w = random.randint(8, 13)
    H = interior_h + 2
    if not 10 <= H <= 15:
        return None
    sprite_area_w = random.randint(3, 5)
    frame_w = interior_w + 2
    W = sprite_area_w + 1 + frame_w
    if not 14 <= W <= 20:
        return None

    # Key positions.
    div_col = sprite_area_w
    fc0, fc1 = sprite_area_w + 1, W - 1
    ir0, ic0 = 1, fc0 + 1
    ir1, ic1 = H - 2, fc1 - 1

    # Assign distinct colors.
    all_colors = list(range(1, 10))
    random.shuffle(all_colors)
    n_movers = random.randint(2, 3)
    n_distractors = random.choice([0, 0, 1])
    idx = 0
    divider_color = all_colors[idx]
    idx += 1
    frame_base = all_colors[idx]
    idx += 1
    mover_colors = all_colors[idx:idx + n_movers]
    idx += n_movers
    distractor_colors = all_colors[idx:idx + n_distractors]

    # Generate mover sprites.
    sprites: list[np.ndarray] = []
    for mc in mover_colors:
        s = _make_sprite(mc, 5, sprite_area_w)
        if s is None:
            return None
        sprites.append(s)

    # Pick marker positions per mover (no two movers share a row or col marker).
    mover_targets: dict[int, tuple[list[int], list[int]]] = {}
    occupied_rows: set[int] = set()
    occupied_cols: set[int] = set()
    for mc, sp in zip(mover_colors, sprites):
        sh, sw = sp.shape
        min_row = ir0 + sh // 2
        max_row = ir1 - (sh - 1 - sh // 2)
        min_col = ic0 + sw // 2
        max_col = ic1 - (sw - 1 - sw // 2)
        if min_row > max_row or min_col > max_col:
            return None
        avail_rows = [r for r in range(min_row, max_row + 1) if r not in occupied_rows]
        row_marks = _pick_spaced(avail_rows, random.randint(1, 2), sh)
        if not row_marks:
            return None
        avail_cols = [c for c in range(min_col, max_col + 1) if c not in occupied_cols]
        col_marks = _pick_spaced(avail_cols, random.randint(1, 2), sw)
        if not col_marks:
            return None
        mover_targets[mc] = (sorted(row_marks), sorted(col_marks))
        occupied_rows.update(row_marks)
        occupied_cols.update(col_marks)

    # Build input grid: bg fill, then divider, frame, sprites.
    inp = np.full((H, W), bg, dtype=int)
    inp[:, div_col] = divider_color

    # Frame top/bottom edges (mirrored).
    top_edge = np.full(frame_w, frame_base, dtype=int)
    for mc, (_, cmarks) in mover_targets.items():
        for cc in cmarks:
            top_edge[cc - fc0] = mc
    inp[0, fc0:fc0 + frame_w] = top_edge
    inp[H - 1, fc0:fc0 + frame_w] = top_edge

    # Frame left/right edges (mirrored).
    for r in range(1, H - 1):
        val = frame_base
        for mc, (rmarks, _) in mover_targets.items():
            if r in rmarks:
                val = mc
                break
        inp[r, fc0] = val
        inp[r, fc1] = val

    # Stack sprites in left strip: movers top-to-bottom, then distractors.
    cur_row = 0
    mover_source_pixels: list[tuple[int, int, int]] = []
    all_items: list[tuple[int, np.ndarray, bool]] = [(mc, sp, False) for mc, sp in zip(mover_colors, sprites)]
    for dc_color in distractor_colors:
        ds = _make_sprite(dc_color, min(5, H - cur_row), sprite_area_w)
        if ds is None:
            return None
        all_items.append((dc_color, ds, True))

    for mc, sp, is_distractor in all_items:
        sh, sw = sp.shape
        if sw > sprite_area_w or cur_row + sh > H:
            return None
        c_off = random.randint(0, sprite_area_w - sw)
        for dr in range(sh):
            for dc in range(sw):
                if sp[dr, dc] != bg:
                    rr = cur_row + dr
                    cc = c_off + dc
                    inp[rr, cc] = sp[dr, dc]
                    if not is_distractor:
                        mover_source_pixels.append((rr, cc, mc))
        cur_row += sh

    # Validate: bg must be most common color so solver detects it correctly.
    vals, counts = np.unique(inp, return_counts=True)
    if int(vals[np.argmax(counts)]) != bg:
        return None

    try:
        output = main(inp)
    except Exception:
        return None

    # Stamping must actually add something new.
    if np.array_equal(inp, output):
        return None

    # Variant-specific guard: mover source pixels must survive unchanged.
    if not mover_source_pixels:
        return None
    for rr, cc, mc in mover_source_pixels:
        if int(output[rr, cc]) != int(mc):
            return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(300):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input: failed after 300 attempts")
