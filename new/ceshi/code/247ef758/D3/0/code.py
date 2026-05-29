# Variant of 247ef758
# Dimension: D3 — 目标位置：框架与标记结构 (dst)
# Change: 框架右侧为矩形框（单色边框+内部背景），边缘标记位于边框像素上，目标为行标记×列标记的笛卡尔积中心点，精灵印刻限制在框架内部
#         → 标记在内部而非边缘：框内第一条内侧引导列/行上出现彩色点，目标仍为同色行点 × 列点的交叉位置

from __future__ import annotations

from collections import defaultdict
import random

import numpy as np

from arc_utils import get_bg, label_mask


# ---------------------------------------------------------------------------
# Design analysis for the D3 variant
# ---------------------------------------------------------------------------
#
# a) Concrete behavior and trace
# - Original task: the right frame uses colored edge pixels. A row is "selected"
#   when the left/right border cells at that row share a mover color, and a
#   column is "selected" when the top/bottom border cells at that column share
#   that mover color.
# - Variant: the frame border is now uniform. Markers move one cell inward:
#   colored dots are placed on a fixed inner guide column and a fixed inner guide
#   row inside the frame. For a mover color X:
#     row marker   -> a dot of color X at (target_row, guide_col)
#     column marker-> a dot of color X at (guide_row, target_col)
#   The sprite is still stamped at every Cartesian-product intersection
#   (target_row, target_col), centered exactly as before.
# - Small trace:
#   Original at row r used g[r, fc0] == g[r, fc1] == X.
#   Variant at row r uses g[r, guide_col] == X instead. Likewise columns are
#   read from g[guide_row, c] instead of the top/bottom frame edges.
#   The movement/stamping stage is unchanged after target extraction.
#
# b) Complexity calibration
# - The rule still has 2-3 mover colors, 1-2 row marks, and 1-2 column marks
#   per mover. That preserves repeated evidence and keeps the task inferable.
# - Markers remain color-coded and re-used across multiple targets, so the
#   relationship is not a one-off lookup.
# - Dots are simpler than full internal lines, which keeps the image clean and
#   avoids accidental barriers inside the frame.
#
# c) Implicit-assumption audit of the original solver
# - largest_enclosed_component_bbox(g == bg): this relied on the frame interior
#   being one enclosed background component. Internal single-pixel dots do not
#   disconnect that component under 4-connectivity, so this still holds.
# - Divider detection and sprite extraction from the left strip do not depend on
#   edge markers, so they remain valid unchanged.
# - The original row/col extraction loop assumed marker colors lived exactly on
#   frame edges. That assumption is false in the variant, so only those loops
#   must change.
# - Stamping/clipping/cover order depend only on the derived targets, not on how
#   targets were encoded, so they remain valid.
#
# d) Placement legality
# - The generator reserves the first inner row/column of the frame as guide lanes
#   for marker dots.
# - Target centers are chosen with extra margin from those guide lanes so stamped
#   sprites do not overwrite the marker dots in the output.
# - No two mover colors share a marked row or marked column, preserving the
#   original unambiguous Cartesian-product decoding.
#
# e) Visual observability
# - The border is fully uniform in the variant, unlike the original examples.
# - All informative colored marks now sit inside the frame on a visible guide row
#   and guide column, so a human must look inside the box rather than on edges.
#
# f) Visual cleanliness
# - Only a single guide row and guide column contain markers; the rest of the
#   frame stays background until stamping.
# - Dots are used instead of thicker glyphs to avoid clutter near the top-left
#   of the frame.
#
# g) Color inferability
# - Colors still mean "this sprite color should be stamped at the rows/columns
#   marked with the same color". No extra color remapping is introduced.
#
# New solver assumptions introduced by the variant
# - The generator must keep the guide row/column inside the frame background
#   except for marker dots.
# - Marker dots must use mover colors only.
# - Target centers must be far enough from the guide lanes that output stamping
#   does not destroy those dots.


def largest_enclosed_component_bbox(
    mask: np.ndarray, *, connectivity: int = 4
) -> tuple[int, int, int, int] | None:
    """
    Local helper: bbox of the largest connected component in `mask`
    that does not touch the outer grid boundary.
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
        if bool(
            (rs == 0).any()
            or (rs == int(h - 1)).any()
            or (cs == 0).any()
            or (cs == int(w - 1)).any()
        ):
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
    Variant rule:
    - Find the right rectangular frame and the left sprite strip as before.
    - Read internal marker dots from the first inner guide column and row of the
      frame instead of reading colored pixels from the border.
    - For each mover color, stamp its sprite at every row-dot × col-dot
      intersection, clipping to the frame interior and preserving bottom-over-top
      overwrite order.
    """
    g = np.array(grid, dtype=int)
    h, w = g.shape
    bg = int(get_bg(g))

    inner_bbox = largest_enclosed_component_bbox(g == int(bg), connectivity=4)
    assert inner_bbox is not None
    ir0, ic0, ir1, ic1 = map(int, inner_bbox)
    fc0 = int(ic0 - 1)

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
                {
                    "color": int(col),
                    "r0": int(r0),
                    "c0": int(c0),
                    "r1": int(r1),
                    "c1": int(c1),
                    "mask": m,
                }
            )

    move_colors = {int(s["color"]) for s in sprites}

    guide_row = int(ir0 + 1)
    guide_col = int(ic0 + 1)
    assert guide_row <= int(ir1) and guide_col <= int(ic1)

    row_marks: dict[int, list[int]] = defaultdict(list)
    col_marks: dict[int, list[int]] = defaultdict(list)
    for r in range(int(ir0), int(ir1) + 1):
        val = int(g[int(r), int(guide_col)])
        if val != int(bg) and val in move_colors:
            row_marks[int(val)].append(int(r))
    for c in range(int(ic0), int(ic1) + 1):
        val = int(g[int(guide_row), int(c)])
        if val != int(bg) and val in move_colors:
            col_marks[int(val)].append(int(c))

    targets: dict[int, list[tuple[int, int]]] = {}
    for col in set(row_marks.keys()) & set(col_marks.keys()):
        pts = {
            (int(r), int(c))
            for r in row_marks[int(col)]
            for c in col_marks[int(col)]
        }
        targets[int(col)] = sorted(pts)

    out = g.copy()
    movers = [s for s in sprites if int(s["color"]) in targets]

    for s in movers:
        r0, c0 = int(s["r0"]), int(s["c0"])
        for rr, cc in np.argwhere(s["mask"]):
            out[int(r0) + int(rr), int(c0) + int(cc)] = int(bg)

    movers.sort(
        key=lambda s: (-((int(s["r0"]) + int(s["r1"])) / 2.0), int(s["c0"]))
    )

    for s in movers:
        col = int(s["color"])
        cr = (int(s["r0"]) + int(s["r1"])) // 2
        cc = (int(s["c0"]) + int(s["c1"])) // 2
        pix = [
            (int(s["r0"]) + int(rr), int(s["c0"]) + int(cc2))
            for rr, cc2 in np.argwhere(s["mask"])
        ]

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
    """Pick up to n values from candidates with at least min_spacing between them."""
    if not candidates:
        return None
    if n == 1:
        return [random.choice(candidates)]
    pool = list(candidates)
    random.shuffle(pool)
    picked: list[int] = []
    for v in pool:
        if all(abs(v - p) >= min_spacing for p in picked):
            picked.append(v)
            if len(picked) == n:
                return sorted(picked)
    return sorted(picked) if picked else None


def _make_sprite(color: int, max_h: int, max_w: int) -> np.ndarray | None:
    """Generate a single-color 8-connected sprite cropped to its bbox."""
    sh = min(random.choice([3, 3, 3, 4, 4, 5]), max_h)
    sw = min(random.choice([3, 3, 3, 4, 4, 5]), max_w)
    if sh < 1 or sw < 1:
        return None
    n_pixels = random.randint(3, min(9, sh * sw))
    pixels = {(sh // 2, sw // 2)}
    r, c = sh // 2, sw // 2
    for _ in range(n_pixels * 30):
        if len(pixels) >= n_pixels:
            break
        dr, dc = random.choice(
            [
                (0, 1),
                (0, -1),
                (1, 0),
                (-1, 0),
                (1, 1),
                (1, -1),
                (-1, 1),
                (-1, -1),
            ]
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
    guide_row = ir0 + 1
    guide_col = ic0 + 1
    if guide_row > ir1 or guide_col > ic1:
        return None

    all_colors = list(range(1, 10))
    random.shuffle(all_colors)
    n_movers = random.choice([2, 2, 3])
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
        top_extent = sh // 2
        bottom_extent = sh - 1 - top_extent
        left_extent = sw // 2
        right_extent = sw - 1 - left_extent

        min_row = max(ir0, guide_row + 1) + top_extent + 1
        max_row = ir1 - bottom_extent
        min_col = max(ic0, guide_col + 1) + left_extent + 1
        max_col = ic1 - right_extent
        if min_row > max_row or min_col > max_col:
            return None

        avail_rows = [
            r for r in range(min_row, max_row + 1) if r not in occupied_rows
        ]
        avail_cols = [
            c for c in range(min_col, max_col + 1) if c not in occupied_cols
        ]
        row_marks = _pick_spaced(avail_rows, random.randint(1, 2), max(2, sh))
        col_marks = _pick_spaced(avail_cols, random.randint(1, 2), max(2, sw))
        if not row_marks or not col_marks:
            return None

        mover_targets[mc] = (row_marks, col_marks)
        occupied_rows.update(row_marks)
        occupied_cols.update(col_marks)

    inp = np.full((h, w), bg, dtype=int)
    inp[:, div_col] = divider_color

    inp[0, fc0 : fc1 + 1] = frame_base
    inp[h - 1, fc0 : fc1 + 1] = frame_base
    inp[:, fc0] = frame_base
    inp[:, fc1] = frame_base

    for mc, (rmarks, cmarks) in mover_targets.items():
        for r in rmarks:
            inp[int(r), int(guide_col)] = int(mc)
        for c in cmarks:
            inp[int(guide_row), int(c)] = int(mc)

    cur_row = 0
    all_items: list[tuple[int, np.ndarray]] = [
        (mc, sp) for mc, sp in zip(mover_colors, sprites)
    ]
    for dc_color in distractor_colors:
        ds = _make_sprite(dc_color, min(5, h - cur_row), sprite_area_w)
        if ds is None:
            return None
        all_items.append((dc_color, ds))

    for _color, sp in all_items:
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
