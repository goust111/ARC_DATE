# Combo variant of 247ef758 (index 3)
# combo_spec: D1/2+D3/2
# Name: 双重非对称边缘

from __future__ import annotations

from collections import defaultdict
import random

import numpy as np

from arc_utils import get_bg, label_mask


# Design analysis for combo D1/2 + D3/2
# --------------------------------------
# a) Concretization and behavior trace
# - The coarse layout stays the same as the base task:
#   left sprite strip | one full-height divider column | full-height rectangular frame on the right.
# - The fusion changes two different things at once:
#   1. Matching is asymmetric: row targets are decided by the LEFT edge only; column targets still require
#      DOUBLE confirmation from TOP and BOTTOM at the same column.
#   2. The frame itself is asymmetric: left/right and top/bottom no longer need to mirror each other.
# - Concrete mini-example:
#   color 4 has a sprite in the left strip.
#   frame row 5: left=4, right=frame_base.
#   frame columns: c=8 has top=4 and bottom=4; c=11 has top=4 and bottom=frame_base.
#   Base task:
#     row 5 is rejected because left/right are not both 4; c=11 is rejected because top/bottom are not both 4.
#     color 4 gets no target at all.
#   D3-only interpretation:
#     row 5 would still be rejected, but c=11 might be accepted if one used top/bottom independently.
#   Fused rule here:
#     row 5 is accepted from the left edge alone; c=8 is accepted by double confirmation; c=11 is rejected.
#     The sprite is stamped at (5, 8) only.
# - Visible difference:
#   the solver must infer both that the right edge is not the deciding row edge and that one-sided top/bottom
#   marks are merely noise for columns.
#
# b) Complexity calibration
# - Keep 2-3 movers and 0-1 distractors so the rule repeats across multiple colors without becoming crowded.
# - Each mover gets 1-2 valid left-edge rows and 1-2 valid double-confirmed columns.
# - The generator also forces at least one top-only mover-colored column and one bottom-only mover-colored column
#   somewhere in the sample. Those columns look tempting but must not be used.
# - Valid row marks and valid columns stay globally unique across movers, which keeps overlaps interpretable.
#
# c) Implicit assumption audit against the original main()
# - largest_enclosed_component_bbox():
#   still valid because the frame interior remains the largest enclosed bg component.
# - divider detection:
#   still valid because the divider remains a nearly full-height single-color column left of the frame.
# - sprite extraction:
#   still valid because sprites remain 8-connected monochrome blobs in the left strip.
# - Original row extraction used `cl == cr`; this assumption is broken by D1/2 and must be replaced with
#   `left edge alone decides rows`.
# - Original column extraction used `ct == cb`; this remains the deciding condition even though D3/2 breaks
#   top/bottom visual symmetry elsewhere. The asymmetry lives in the presence of extra one-sided edge marks,
#   not in the acceptance rule for target columns.
# - Stamping order and clipping assumptions remain unchanged. The generator spaces valid target rows/cols using
#   sprite bbox dimensions so centered placement fits inside the frame.
#
# d) Placement legality
# - Row marks are chosen from legal center rows computed from each sprite height.
# - Valid matched columns and one-sided noise columns are chosen from legal center columns computed from sprite width.
# - Valid columns are unique across movers; row marks are unique across movers.
# - Sprites are stacked top-to-bottom in the strip without overlap; samples that overflow are rejected.
#
# e) Visual observability
# - Left-only row marks are directly visible because the right edge stays base-colored at those rows.
# - One-sided top-only / bottom-only mover-colored columns are directly visible because they appear on only one
#   horizontal edge but do not produce extra stamped copies.
# - Therefore, from a few examples, a human can notice:
#   left edge matters for rows, both top and bottom matter for columns, and the frame itself is not mirrored.
#
# f) Cleanliness
# - Unused frame-edge cells stay frame_base.
# - Right-edge noise is avoided; the right side being mostly base makes the left-only row rule legible.
# - One-sided horizontal noise is limited to a small number of columns so the frame edge stays readable.
#
# g) New assumptions introduced by the fused main()
# - Every mover color that should move must appear at least once on the left edge and at least once on a
#   top-bottom matched column. The generator enforces this.
# - One-sided top/bottom noise may reuse mover colors, but must not reuse a matched column. The generator enforces
#   column uniqueness so valid and invalid cues remain separable.


def largest_enclosed_component_bbox(
    mask: np.ndarray, *, connectivity: int = 4
) -> tuple[int, int, int, int] | None:
    """
    Local helper (only used by this solver): bbox of largest connected component in `mask`
    that does NOT touch the grid border.
    """
    h, w = mask.shape
    best_area = -1
    best_bbox: tuple[int, int, int, int] | None = None

    lab, n = label_mask(mask.astype(bool), connectivity=int(connectivity))
    for k in range(1, int(n) + 1):
        coords = np.argwhere(lab == int(k))
        if coords.size == 0:
            continue
        rs = coords[:, 0]
        cs = coords[:, 1]
        if bool((rs == 0).any() or (rs == int(h - 1)).any() or (cs == 0).any() or (cs == int(w - 1)).any()):
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
    Move colored sprites from the left strip into a framed area at marker-defined targets.

    Fused rule:
    - Target rows: every row where the LEFT frame edge has the mover color.
    - Target columns: every column where TOP and BOTTOM both show the mover color.
    - Right-edge marks and one-sided top/bottom marks may exist but do not authorize targets.
    - Erase each moved sprite from the left strip and stamp it centered at every valid row x column target.
      Lower input sprites stamp first so upper ones overwrite them.
    """
    g = np.array(grid, dtype=int)
    h, w = g.shape
    bg = int(get_bg(g))

    inner_bbox = largest_enclosed_component_bbox(g == int(bg), connectivity=4)
    assert inner_bbox is not None
    ir0, ic0, ir1, ic1 = map(int, inner_bbox)
    fr0, fc0, fr1, fc1 = int(ir0 - 1), int(ic0 - 1), int(ir1 + 1), int(ic1 + 1)

    divider_mask = np.zeros((h, w), dtype=bool)
    for c in range(0, max(0, int(fc0))):
        colv = g[:, int(c)]
        non = colv[colv != int(bg)]
        if int(len(non)) >= int(h - 1) and len(set(map(int, non))) == 1:
            divider_mask[:, int(c)] = True

    left_cols = np.arange(w)[None, :] < int(fc0)

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
            m = slab[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1] == int(sid)
            sprites.append(
                {"color": int(col), "r0": int(r0), "c0": int(c0), "r1": int(r1), "c1": int(c1), "mask": m}
            )

    move_colors = {int(s["color"]) for s in sprites}

    row_marks: dict[int, list[int]] = defaultdict(list)
    col_marks: dict[int, list[int]] = defaultdict(list)

    for r in range(int(ir0), int(ir1) + 1):
        cl = int(g[int(r), int(fc0)])
        if cl != int(bg) and cl in move_colors:
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

    for s in movers:
        r0, c0 = int(s["r0"]), int(s["c0"])
        for rr, cc in np.argwhere(s["mask"]):
            out[int(r0) + int(rr), int(c0) + int(cc)] = int(bg)

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
    """Pick exactly n values from candidates with at least min_spacing between any two."""
    if n == 0:
        return []
    if len(candidates) < n:
        return None
    pool = list(candidates)
    random.shuffle(pool)
    picked: list[int] = []
    for v in pool:
        if all(abs(v - p) >= min_spacing for p in picked):
            picked.append(int(v))
            if len(picked) == n:
                return sorted(picked)
    return None


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


def _legal_row_range(ir0: int, ir1: int, sprite_h: int) -> tuple[int, int]:
    return int(ir0 + sprite_h // 2), int(ir1 - (sprite_h - 1 - sprite_h // 2))


def _legal_col_range(ic0: int, ic1: int, sprite_w: int) -> tuple[int, int]:
    return int(ic0 + sprite_w // 2), int(ic1 - (sprite_w - 1 - sprite_w // 2))


def _assign_column_plan(
    mover_colors: list[int],
    sprites: list[np.ndarray],
    ic0: int,
    ic1: int,
) -> dict[int, dict[str, list[int]]] | None:
    """
    For each mover color, assign:
    - matched: columns that appear on both top and bottom and therefore are real targets
    - top_only: columns that appear only on top
    - bottom_only: columns that appear only on bottom

    Sample-level guarantees:
    - every mover has at least one matched column, so every mover can move
    - at least one mover-colored top-only column exists
    - at least one mover-colored bottom-only column exists
    - all columns are globally unique across all three groups to keep cues legible
    """
    if len(mover_colors) < 2:
        return None

    by_color_sprite = {int(mc): sp for mc, sp in zip(mover_colors, sprites)}
    order = mover_colors[:]
    random.shuffle(order)
    top_noise_owner = int(order[0])
    bottom_noise_owner = int(order[1])
    extra_owner = int(order[2]) if len(order) >= 3 else None

    occupied_cols: set[int] = set()
    result: dict[int, dict[str, list[int]]] = {}

    for mc in mover_colors:
        sp = by_color_sprite[int(mc)]
        sw = int(sp.shape[1])
        min_col, max_col = _legal_col_range(int(ic0), int(ic1), sw)
        if min_col > max_col:
            return None

        matched_n = random.choice([1, 1, 2])
        top_only_n = 1 if int(mc) == top_noise_owner else 0
        bottom_only_n = 1 if int(mc) == bottom_noise_owner else 0
        if extra_owner is not None and int(mc) == extra_owner:
            extra_mode = random.choice(["none", "top", "bottom", "matched"])
            if extra_mode == "top":
                top_only_n += 1
            elif extra_mode == "bottom":
                bottom_only_n += 1
            elif extra_mode == "matched":
                matched_n += 1

        total_n = int(matched_n + top_only_n + bottom_only_n)
        avail_cols = [c for c in range(min_col, max_col + 1) if c not in occupied_cols]
        picked = _pick_spaced(avail_cols, total_n, max(1, sw))
        if picked is None:
            return None

        picked = list(picked)
        random.shuffle(picked)
        matched = sorted(picked[:matched_n])
        top_only = sorted(picked[matched_n : matched_n + top_only_n])
        bottom_only = sorted(picked[matched_n + top_only_n :])
        occupied_cols.update(matched)
        occupied_cols.update(top_only)
        occupied_cols.update(bottom_only)
        result[int(mc)] = {"matched": matched, "top_only": top_only, "bottom_only": bottom_only}

    if not any(result[int(mc)]["top_only"] for mc in mover_colors):
        return None
    if not any(result[int(mc)]["bottom_only"] for mc in mover_colors):
        return None

    return result


def _gen_attempt() -> np.ndarray | None:
    bg = 0

    interior_h = random.randint(8, 13)
    interior_w = random.randint(8, 13)
    h = interior_h + 2
    if not 10 <= h <= 15:
        return None
    sprite_area_w = random.randint(3, 5)
    frame_w = interior_w + 2
    w = sprite_area_w + 1 + frame_w
    if not 14 <= w <= 20:
        return None

    div_col = sprite_area_w
    fc0, fc1 = sprite_area_w + 1, w - 1
    ir0, ic0 = 1, fc0 + 1
    ir1, ic1 = h - 2, fc1 - 1

    all_colors = list(range(1, 10))
    random.shuffle(all_colors)
    n_movers = random.randint(2, 3)
    n_distractors = random.choice([0, 0, 1])
    idx = 0
    divider_color = int(all_colors[idx])
    idx += 1
    frame_base = int(all_colors[idx])
    idx += 1
    mover_colors = [int(v) for v in all_colors[idx : idx + n_movers]]
    idx += n_movers
    distractor_colors = [int(v) for v in all_colors[idx : idx + n_distractors]]

    sprites: list[np.ndarray] = []
    for mc in mover_colors:
        s = _make_sprite(int(mc), 5, sprite_area_w)
        if s is None:
            return None
        sprites.append(s)

    mover_rows: dict[int, list[int]] = {}
    occupied_rows: set[int] = set()
    for mc, sp in zip(mover_colors, sprites):
        sh, sw = sp.shape
        min_row, max_row = _legal_row_range(int(ir0), int(ir1), int(sh))
        min_col, max_col = _legal_col_range(int(ic0), int(ic1), int(sw))
        if min_row > max_row or min_col > max_col:
            return None
        avail_rows = [r for r in range(min_row, max_row + 1) if r not in occupied_rows]
        row_marks = _pick_spaced(avail_rows, random.randint(1, 2), max(1, int(sh)))
        if row_marks is None:
            return None
        mover_rows[int(mc)] = row_marks
        occupied_rows.update(row_marks)

    mover_cols = _assign_column_plan(mover_colors, sprites, ic0, ic1)
    if mover_cols is None:
        return None

    inp = np.full((h, w), bg, dtype=int)
    inp[:, div_col] = divider_color

    top_edge = np.full(frame_w, frame_base, dtype=int)
    bottom_edge = np.full(frame_w, frame_base, dtype=int)
    for mc in mover_colors:
        for cc in mover_cols[int(mc)]["matched"]:
            top_edge[int(cc - fc0)] = int(mc)
            bottom_edge[int(cc - fc0)] = int(mc)
        for cc in mover_cols[int(mc)]["top_only"]:
            top_edge[int(cc - fc0)] = int(mc)
        for cc in mover_cols[int(mc)]["bottom_only"]:
            bottom_edge[int(cc - fc0)] = int(mc)
    inp[0, fc0 : fc0 + frame_w] = top_edge
    inp[h - 1, fc0 : fc0 + frame_w] = bottom_edge

    inp[1 : h - 1, fc0] = frame_base
    inp[1 : h - 1, fc1] = frame_base
    for mc in mover_colors:
        for rr in mover_rows[int(mc)]:
            inp[int(rr), int(fc0)] = int(mc)

    cur_row = 0
    all_items: list[tuple[int, np.ndarray]] = [(int(mc), sp) for mc, sp in zip(mover_colors, sprites)]
    for dc_color in distractor_colors:
        ds = _make_sprite(int(dc_color), min(5, h - cur_row), sprite_area_w)
        if ds is None:
            return None
        all_items.append((int(dc_color), ds))

    for _, sp in all_items:
        sh, sw = sp.shape
        if sw > sprite_area_w or cur_row + sh > h:
            return None
        c_off = random.randint(0, sprite_area_w - sw)
        for dr in range(sh):
            for dc in range(sw):
                if int(sp[dr, dc]) != bg:
                    inp[cur_row + dr, c_off + dc] = int(sp[dr, dc])
        cur_row += sh

    vals, counts = np.unique(inp, return_counts=True)
    if int(vals[np.argmax(counts)]) != bg:
        return None

    left_only_rows = sum(
        1
        for r in range(int(ir0), int(ir1) + 1)
        if int(inp[int(r), int(fc0)]) in mover_colors and int(inp[int(r), int(fc1)]) != int(inp[int(r), int(fc0)])
    )
    top_only_cols = sum(
        1
        for c in range(int(ic0), int(ic1) + 1)
        if int(inp[0, int(c)]) in mover_colors and int(inp[h - 1, int(c)]) != int(inp[0, int(c)])
    )
    bottom_only_cols = sum(
        1
        for c in range(int(ic0), int(ic1) + 1)
        if int(inp[h - 1, int(c)]) in mover_colors and int(inp[0, int(c)]) != int(inp[h - 1, int(c)])
    )
    if left_only_rows == 0 or top_only_cols == 0 or bottom_only_cols == 0:
        return None

    try:
        output = main(inp)
    except Exception:
        return None
    if np.array_equal(inp, output):
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(400):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input: failed after 400 attempts")
