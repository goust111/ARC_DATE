# Variant of 247ef758
# Dimension: D1 — 匹配条件：边缘颜色匹配 (match_cond)
# Change: 精灵颜色 == 框架左/右边缘行标记颜色 且 == 上/下边缘列标记颜色（双对边镜像匹配），只有两对对边同时存在该颜色时... → 单对边匹配（仅行或仅列需双边确认）

from __future__ import annotations

from collections import defaultdict
import random

import numpy as np

from arc_utils import get_bg, label_mask


# Design analysis for the D1 variant
# ----------------------------------
# concretization:
# - Layout stays the same: left sprite strip | one full-height divider column | framed area on the right.
# - Movers are still single-color blobs in the left strip. The frame interior is still blank bg.
# - Column markers keep the original requirement: a mover color must appear on BOTH top and bottom edges at the
#   same interior column. That column participates in the Cartesian product.
# - Row markers change asymmetrically: a mover color only needs to appear on the LEFT frame edge at that row.
#   The right edge is no longer a confirmation edge and is intentionally kept as frame_base in generated tasks.
# - Output stays "erase movers from the strip, then stamp each mover centered at all valid row x column targets,
#   with lower sprites stamped first and upper sprites overwriting later."
#
# behavior trace on a concrete mini-example:
# - Suppose sprite color 4 exists on the left strip.
# - Frame edges show: left edge row 5 = 4, right edge row 5 = frame_base, top edge col 9 = 4, bottom edge col 9 = 4.
# - Original logic:
#   row 5 is rejected because left/right are not both 4, so color 4 gets no target and does not move.
# - Variant logic:
#   row 5 is accepted because the LEFT edge alone is 4; col 9 is accepted because top/bottom are both 4;
#   therefore target (5, 9) exists and sprite 4 is stamped there.
# - Visible difference:
#   movers now replicate even when the right edge does not mirror the left marker.
#
# complexity calibration:
# - Keep 2-3 movers and 0-1 distractors from the original task so multiple colors repeat the same asymmetric rule.
# - Keep row/col mark counts at 1-2 per mover and forbid sharing rows/cols between movers. This avoids ambiguous
#   overlaps while still producing multiple Cartesian-product placements.
# - Right edge is mostly frame_base so the left-only row rule is visually obvious rather than drowned in noise.
#
# implicit assumptions audit for main():
# - largest_enclosed_component_bbox() still assumes the frame interior is the largest enclosed bg component.
#   Generator preserves this by keeping the same frame geometry.
# - divider detection assumes a nearly full-height uniform non-bg column to the left of the frame.
#   Generator still creates exactly one such divider column.
# - sprite extraction assumes movers/distractors are 8-connected monochrome components in the left strip and that
#   divider cells are excluded. Generator preserves this unchanged.
# - target extraction is the only logic that depended on the old "opposite edges must match" invariant.
#   In the variant, row extraction must not inspect the right edge as a gate. Column extraction still must.
# - stamping assumes chosen targets are centered placements that fit within the interior or can be clipped safely.
#   Generator still spaces row/col marks using the sprite bbox dimensions to keep placements legal.
#
# placement legality:
# - Each mover receives unique row marks and unique col marks, spaced by at least the sprite height/width.
# - The legal row/col ranges are computed from the sprite center so every stamped copy fits inside the frame interior.
# - Sprites are stacked top-to-bottom in the left strip without overlap; distractors remain untouched.
#
# observability / cleanliness:
# - The asymmetry is visible directly in the input: row colors appear on the left edge only, while column colors
#   still appear mirrored on top and bottom.
# - Because the right edge is not used for row confirmation, generated examples do not collapse back to the
#   original task. Under the old solver, movers would fail to move; under the new solver, they replicate.
# - No color ordering or hidden numeric property is used; all reasoning is based on visible spatial/color matches.


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

    Variant rule:
    - Row marks: only the LEFT frame edge needs the mover color.
    - Column marks: TOP and BOTTOM frame edges still both need the mover color.
    - Targets are the Cartesian product of valid rows and valid columns per color.
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
    divider_color = all_colors[idx]
    idx += 1
    frame_base = all_colors[idx]
    idx += 1
    mover_colors = all_colors[idx : idx + n_movers]
    idx += n_movers
    distractor_colors = all_colors[idx : idx + n_distractors]

    sprites: list[np.ndarray] = []
    for mc in mover_colors:
        s = _make_sprite(mc, 5, sprite_area_w)
        if s is None:
            return None
        sprites.append(s)

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

    inp = np.full((h, w), bg, dtype=int)
    inp[:, div_col] = divider_color

    top_edge = np.full(frame_w, frame_base, dtype=int)
    for mc, (_, cmarks) in mover_targets.items():
        for cc in cmarks:
            top_edge[cc - fc0] = mc
    inp[0, fc0 : fc0 + frame_w] = top_edge
    inp[h - 1, fc0 : fc0 + frame_w] = top_edge

    left_edge = np.full(h - 2, frame_base, dtype=int)
    for mc, (rmarks, _) in mover_targets.items():
        for rr in rmarks:
            left_edge[rr - 1] = mc
    inp[1 : h - 1, fc0] = left_edge

    # Right edge is intentionally NOT a confirmation edge for row marks in this variant.
    inp[1 : h - 1, fc1] = frame_base

    cur_row = 0
    all_items: list[tuple[int, np.ndarray]] = [(mc, sp) for mc, sp in zip(mover_colors, sprites)]
    for dc_color in distractor_colors:
        ds = _make_sprite(dc_color, min(5, h - cur_row), sprite_area_w)
        if ds is None:
            return None
        all_items.append((dc_color, ds))

    for _, sp in all_items:
        sh, sw = sp.shape
        if sw > sprite_area_w or cur_row + sh > h:
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
        output = main(inp)
    except Exception:
        return None
    if np.array_equal(inp, output):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(300):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input: failed after 300 attempts")
