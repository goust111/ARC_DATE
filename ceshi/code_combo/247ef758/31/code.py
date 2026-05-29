# Combo variant of 247ef758 (index 31)
# combo_spec: D6/0+D9/2
# Name: 干扰精灵删除+裁剪输出
#
# Design analysis for the combo
# -----------------------------
# fusion_summary:
#   baseline task:
#     - parse single-color sprites from the left strip
#     - colors repeated on both frame side edges define valid target rows
#     - colors repeated on both frame top/bottom edges define valid target cols
#     - for each mover color, stamp its sprite at every row_mark x col_mark center
#     - lower input sprites stamp first, upper sprites stamp last and cover
#   combo change:
#     - D9/2: unmatched distractor sprites are erased to background in the full transformed canvas
#     - D6/0: after that full-canvas transform, return only the frame interior crop
#
# behavior_trace:
#   concrete_example:
#     - input has three left-strip sprites: red mover near the top, green mover below it,
#       blue distractor at the bottom. Only red and green appear as mirrored frame markers.
#     - original full-grid task:
#         erase red+green from the left, keep blue on the left, stamp red/green in the frame,
#         then return the whole canvas.
#     - fused variant:
#         erase red+green+blue from the left, stamp red/green in the frame exactly the same way,
#         then crop away the entire left strip, divider, and border so only the frame interior remains.
#     - difference:
#         compared with the original task, distractor blue is deleted rather than preserved.
#         compared with D6-alone, the final crop can match pixel-for-pixel, but the hidden full-canvas
#         state differs whenever a distractor exists. This combo is therefore semantically fused even
#         though D6 intentionally hides the left-strip result from the returned output.
#
# complexity_check:
#   - keep 2-3 mover colors and 1-2 row/col marks per mover, so the Cartesian-product stamping pattern
#     still has repeated evidence and is not reduced to a one-off placement.
#   - keep at most one distractor to avoid ambiguous "which unmatched object disappears" logic.
#   - bias generation toward one distractor in most samples so the D9 branch is exercised often,
#     even though the final crop hides that region.
#
# implicit_assumption_audit:
#   - largest_enclosed_component_bbox(g == bg):
#       assumes the frame interior is the largest background component that does not touch the border.
#       Generator preserves one full-height rectangular frame, so this remains valid.
#   - divider detection:
#       scans columns left of the frame for an almost full-height uniform non-bg column.
#       Generator keeps a single full-height divider column, so the parse stays unambiguous.
#   - sprite extraction:
#       assumes every non-divider 8-connected monochrome component left of the frame is a sprite.
#       Generator places only movers/distractor there and prevents overlap.
#   - marker discovery:
#       assumes mover colors, and only mover colors, appear on both opposite frame edges.
#       Generator never marks distractor colors and keeps mover row/col marks unique across colors.
#   - stamping legality:
#       assumes each chosen row/col mark lets the centered sprite fit within the frame interior.
#       Generator derives legal row/col ranges from the sprite bbox dimensions.
#   - new D9 assumption:
#       erasing every left-strip sprite is safe only because distractor colors never appear elsewhere.
#       Generator enforces distinct colors for bg, divider, frame, movers, distractor.
#   - new D6 assumption:
#       the intended answer is exactly the inferred interior bbox. Generator keeps a single frame,
#       so the crop boundary is unique.
#
# placement_legality:
#   - sprites are stacked top-to-bottom in the left strip with no overlap and no spill past the strip.
#   - same-color target rows/cols are spaced by at least the sprite height/width to avoid self-overlap.
#   - different mover colors never reuse row or col marks, which keeps target lattices separated.
#
# visual_observability:
#   - the crop rule is directly visible because outputs are always smaller and contain only frame interior.
#   - mover replication is still visible inside the crop because the raw interior starts blank.
#   - distractor deletion is latent after cropping; to keep the combo meaningful, generation strongly
#     prefers having a distractor, and validation compares the full hidden canvas against the old
#     "keep distractor" behavior to ensure D9 is genuinely active before the crop.
#
# cleanliness:
#   - the crop removes border markers and the sprite strip, so the final output is visually clean.
#   - inside the crop there are only stamped mover replicas on background; no extra decorations.

from __future__ import annotations

from collections import defaultdict
import random

import numpy as np

from arc_utils import get_bg, label_mask


def largest_enclosed_component_bbox(
    mask: np.ndarray, *, connectivity: int = 4
) -> tuple[int, int, int, int] | None:
    """
    Local helper: bbox of the largest connected component in `mask`
    that does NOT touch the grid border.
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
        if bool(
            (rs == 0).any()
            or (rs == int(H - 1)).any()
            or (cs == 0).any()
            or (cs == int(W - 1)).any()
        ):
            continue
        area = int(coords.shape[0])
        if area > best_area:
            best_area = area
            r0, c0 = coords.min(axis=0)
            r1, c1 = coords.max(axis=0)
            best_bbox = (int(r0), int(c0), int(r1), int(c1))

    return best_bbox


def _extract_scene(
    grid: np.ndarray,
) -> tuple[
    np.ndarray,
    int,
    tuple[int, int, int, int],
    tuple[int, int, int, int],
    list[dict[str, int | np.ndarray]],
    dict[int, list[tuple[int, int]]],
]:
    """
    Parse the fixed scene structure shared by the base task and this combo.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

    inner_bbox = largest_enclosed_component_bbox(g == int(bg), connectivity=4)
    assert inner_bbox is not None
    ir0, ic0, ir1, ic1 = map(int, inner_bbox)
    fr0, fc0, fr1, fc1 = int(ir0 - 1), int(ic0 - 1), int(ir1 + 1), int(ic1 + 1)

    divider_mask = np.zeros((H, W), dtype=bool)
    for c in range(0, max(0, int(fc0))):
        colv = g[:, int(c)]
        non = colv[colv != int(bg)]
        if int(len(non)) >= int(H - 1) and len(set(map(int, non))) == 1:
            divider_mask[:, int(c)] = True

    left_cols = np.arange(W)[None, :] < int(fc0)

    sprites: list[dict[str, int | np.ndarray]] = []
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
            sprites.append(
                {
                    "color": int(col),
                    "r0": int(r0),
                    "c0": int(c0),
                    "r1": int(r1),
                    "c1": int(c1),
                    "mask": slab[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1] == int(sid),
                }
            )

    sprite_colors = {int(s["color"]) for s in sprites}

    row_marks: dict[int, list[int]] = defaultdict(list)
    col_marks: dict[int, list[int]] = defaultdict(list)
    for r in range(int(ir0), int(ir1) + 1):
        cl = int(g[int(r), int(fc0)])
        cr = int(g[int(r), int(fc1)])
        if cl == cr and cl != int(bg) and cl in sprite_colors:
            row_marks[int(cl)].append(int(r))
    for c in range(int(ic0), int(ic1) + 1):
        ct = int(g[int(fr0), int(c)])
        cb = int(g[int(fr1), int(c)])
        if ct == cb and ct != int(bg) and ct in sprite_colors:
            col_marks[int(ct)].append(int(c))

    targets: dict[int, list[tuple[int, int]]] = {}
    for col in set(row_marks.keys()) & set(col_marks.keys()):
        targets[int(col)] = sorted(
            {(int(r), int(c)) for r in row_marks[int(col)] for c in col_marks[int(col)]}
        )

    return g, bg, (int(ir0), int(ic0), int(ir1), int(ic1)), (int(fr0), int(fc0), int(fr1), int(fc1)), sprites, targets


def _solve_full(
    grid: np.ndarray, *, erase_unmatched: bool
) -> tuple[np.ndarray, tuple[int, int, int, int], dict[str, int]]:
    """
    Full-canvas transform before the D6 crop.

    `erase_unmatched=True` implements the fused D9 rule.
    `erase_unmatched=False` mirrors the old behavior and is kept only for validation.
    """
    g, bg, inner_bbox, _frame_bbox, sprites, targets = _extract_scene(grid)
    ir0, ic0, ir1, ic1 = inner_bbox

    out = g.copy()
    movers = [s for s in sprites if int(s["color"]) in targets]
    erased = sprites if erase_unmatched else movers

    for s in erased:
        r0, c0 = int(s["r0"]), int(s["c0"])
        mask = np.asarray(s["mask"], dtype=bool)
        for rr, cc in np.argwhere(mask):
            out[int(r0) + int(rr), int(c0) + int(cc)] = int(bg)

    movers.sort(key=lambda s: (-((int(s["r0"]) + int(s["r1"])) / 2.0), int(s["c0"])))

    for s in movers:
        col = int(s["color"])
        cr = (int(s["r0"]) + int(s["r1"])) // 2
        cc = (int(s["c0"]) + int(s["c1"])) // 2
        mask = np.asarray(s["mask"], dtype=bool)
        pix = [(int(s["r0"]) + int(rr), int(s["c0"]) + int(cc2)) for rr, cc2 in np.argwhere(mask)]

        for tr, tc in targets[int(col)]:
            dr = int(tr - cr)
            dc = int(tc - cc)
            for r, c in pix:
                rr = int(r + dr)
                cc3 = int(c + dc)
                if int(ir0) <= rr <= int(ir1) and int(ic0) <= cc3 <= int(ic1):
                    out[rr, cc3] = int(col)

    meta = {
        "n_sprites": int(len(sprites)),
        "n_movers": int(len(movers)),
        "n_distractors": int(len(sprites) - len(movers)),
    }
    return out, inner_bbox, meta


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fused rule:
    1. erase every left-strip sprite, including unmatched distractors
    2. stamp only marker-matched movers into the frame interior
    3. return the tight crop of the frame interior
    """
    out, (ir0, ic0, ir1, ic1), _meta = _solve_full(grid, erase_unmatched=True)
    return out[int(ir0) : int(ir1) + 1, int(ic0) : int(ic1) + 1]


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
    max_pixels = min(9, sh * sw)
    if max_pixels < 3:
        return None
    n_pixels = random.randint(3, max_pixels)
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

    div_col = sprite_area_w
    fc0, fc1 = sprite_area_w + 1, W - 1
    ir0, ic0 = 1, fc0 + 1
    ir1, ic1 = H - 2, fc1 - 1

    all_colors = list(range(1, 10))
    random.shuffle(all_colors)
    n_movers = random.randint(2, 3)
    n_distractors = random.choice([1, 1, 1, 0])
    idx = 0
    divider_color = all_colors[idx]
    idx += 1
    frame_base = all_colors[idx]
    idx += 1
    mover_colors = all_colors[idx : idx + n_movers]
    idx += n_movers
    distractor_colors = all_colors[idx : idx + n_distractors]

    mover_sprites: list[np.ndarray] = []
    for mc in mover_colors:
        sprite = _make_sprite(mc, 5, sprite_area_w)
        if sprite is None:
            return None
        mover_sprites.append(sprite)

    used_h = sum(int(sp.shape[0]) for sp in mover_sprites)
    if n_distractors > 0 and used_h >= H:
        return None

    mover_targets: dict[int, tuple[list[int], list[int]]] = {}
    occupied_rows: set[int] = set()
    occupied_cols: set[int] = set()
    for mc, sp in zip(mover_colors, mover_sprites):
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

    inp = np.full((H, W), bg, dtype=int)
    inp[:, div_col] = divider_color

    top_edge = np.full(frame_w, frame_base, dtype=int)
    for mc, (_rmarks, cmarks) in mover_targets.items():
        for cc in cmarks:
            top_edge[cc - fc0] = mc
    inp[0, fc0 : fc0 + frame_w] = top_edge
    inp[H - 1, fc0 : fc0 + frame_w] = top_edge

    for r in range(1, H - 1):
        val = frame_base
        for mc, (rmarks, _cmarks) in mover_targets.items():
            if r in rmarks:
                val = mc
                break
        inp[r, fc0] = val
        inp[r, fc1] = val

    all_items: list[tuple[int, np.ndarray, bool]] = [
        (mc, sp, False) for mc, sp in zip(mover_colors, mover_sprites)
    ]
    cur_row = 0
    for dc_color in distractor_colors:
        ds = _make_sprite(dc_color, min(5, H - cur_row), sprite_area_w)
        if ds is None:
            return None
        all_items.append((dc_color, ds, True))

    for _color, sp, _is_distractor in all_items:
        sh, sw = sp.shape
        if sw > sprite_area_w or cur_row + sh > H:
            return None
        c_off = random.randint(0, sprite_area_w - sw)
        for dr in range(sh):
            for dc in range(sw):
                if sp[dr, dc] != bg:
                    inp[cur_row + dr, c_off + dc] = sp[dr, dc]
        cur_row += sh

    vals, counts = np.unique(inp, return_counts=True)
    if int(vals[np.argmax(counts)]) != bg:
        return None

    try:
        full_combo, inner_bbox, meta = _solve_full(inp, erase_unmatched=True)
        full_old, _inner_bbox_old, _meta_old = _solve_full(inp, erase_unmatched=False)
        output = main(inp)
    except Exception:
        return None

    out_ir0, out_ic0, out_ir1, out_ic1 = inner_bbox
    raw_interior = inp[int(out_ir0) : int(out_ir1) + 1, int(out_ic0) : int(out_ic1) + 1]
    combo_crop = full_combo[int(out_ir0) : int(out_ir1) + 1, int(out_ic0) : int(out_ic1) + 1]
    if not np.array_equal(output, combo_crop):
        return None
    if output.shape != (interior_h, interior_w):
        return None
    if output.shape[0] >= inp.shape[0] or output.shape[1] >= inp.shape[1]:
        return None
    if np.array_equal(output, raw_interior):
        return None
    if int(meta["n_movers"]) != n_movers:
        return None
    if int(meta["n_distractors"]) != n_distractors:
        return None

    # D9 validation must be done on the hidden full canvas, because D6 removes the left strip from output.
    if n_distractors > 0:
        if np.array_equal(full_combo, full_old):
            return None
        for dc_color in distractor_colors:
            if int(np.count_nonzero(full_combo == int(dc_color))) != 0:
                return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(300):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input: failed after 300 attempts")
