# Combo variant of 247ef758 (index 22)
# combo_spec: D1/2+D3/2+D9/1
# Name: 非对称匹配+非对称框架+部分边缘干扰
#
# Design analysis
# a) Concrete behavior trace
#    - Base layout stays recognizable: left sprite strip | full-height divider | right rectangular frame.
#    - Fusion precedence is:
#      1. D1/2 changes the matching rule itself: rows are confirmed by the LEFT edge only; columns still need
#         a TOP/BOTTOM double confirmation at the same column.
#      2. D3/2 changes the visible frame coding without relaxing the column rule: top and bottom edges are built
#         independently, so they may disagree at many columns. Only the overlap of the two horizontal edges is real.
#      3. D9/1 adds a distractor color onto only part of the frame edge structure, creating additional false positives.
#    - Mini trace:
#      Suppose color 4 is a mover and color 6 is a distractor.
#      Left edge row 5 = 4, right edge row 5 = frame_base, so row 5 is valid for 4 under the fused rule.
#      Top edge has 4 at columns 8 and 11; bottom edge has 4 only at column 8.
#      Therefore column 8 is valid, column 11 is a decoy, and sprite 4 is stamped only at (5, 8).
#      If distractor color 6 appears only on the top edge at column 13, it still does not move because it lacks
#      a double-confirmed column and also receives no left-edge row mark.
#    - Difference from the original task:
#      original solver would reject row 5 because right edge is not 4; fused solver accepts it.
#      The asymmetric top-only/bottom-only columns are visible but ignored unless both horizontal edges agree.
#
# b) Complexity calibration
#    - Keep 2-3 movers so the asymmetric row rule repeats across multiple colors.
#    - Force at least one top-only mover decoy column and at least one bottom-only mover decoy column per sample,
#      otherwise the D3 contribution would be too weak visually.
#    - Force one distractor sprite whose color appears on a partial horizontal edge, so D9 is present in every sample.
#    - Horizontal marker columns are globally unique across all colors to avoid ambiguous shared targets.
#
# c) Implicit assumptions reviewed against `main()`
#    - `largest_enclosed_component_bbox()` still assumes the frame interior is the largest enclosed bg component.
#      Generator preserves the original rectangular frame geometry.
#    - Divider detection still assumes a nearly full-height uniform non-bg column to the left of the frame.
#      Generator keeps exactly one such divider column.
#    - Sprite extraction still assumes left-strip objects are monochrome 8-connected components. Preserved.
#    - The original row parser relied on `left == right`; that assumption is intentionally broken.
#      The fused solver must read only the left edge for row marks.
#    - The original column parser relied on mirrored top/bottom edges. In the fused variant the edges are not mirrored,
#      but the validity condition is still `top == bottom == mover_color`, so the solver must ignore top-only/bottom-only
#      cells rather than union them.
#    - Because distractor colors now appear on frame edges, the solver must also restrict accepted markers to colors that
#      really exist as left-strip sprites. The original code already did this via `move_colors`, and the fusion keeps it.
#
# d) Placement legality
#    - For every mover, valid row marks are chosen from the rows where the sprite can be centered without leaving the
#      frame interior. The same is done for valid columns and decoy columns using the sprite width.
#    - Row marks are globally unique across movers; valid and decoy columns are globally unique across every color.
#    - Left-strip sprites are stacked without overlap and only accepted when all sprites fit.
#
# e) Visual observability
#    - Humans can see the rule difference directly: row colors sit only on the left border, while horizontal borders
#      contain both real overlaps and single-sided decoys.
#    - The fused rule is inferable from a few examples because the same color can appear on top-only/bottom-only columns
#      yet produce output only at the columns where both horizontal edges agree.
#    - No hidden color ordering, sprite generation order, or numeric color id is used.
#
# f) Visual cleanliness
#    - The frame interior remains empty before stamping, so all signal lives on the frame edges and left strip.
#    - Right edge stays frame_base only; this keeps the row asymmetry easy to spot instead of mixing two row conventions.
#    - Decoy columns are limited to 1 per side per mover role, which keeps the border busy enough to matter but not noisy.
#
# g) New solver assumptions checked against the generator
#    - Every mover has at least one valid left-edge row mark and at least one valid double-confirmed column.
#    - Every sample has asymmetric horizontal evidence: at least one mover top-only column and at least one mover
#      bottom-only column.
#    - Every sample has a distractor color on a partial horizontal edge only, never on a valid left row mark.
#    - These guarantees ensure the solver's new asymmetric parsing assumptions are always exercised and never depend on
#      invisible generation-side state.

from __future__ import annotations

from collections import defaultdict
import random

import numpy as np

from arc_utils import get_bg, label_mask


def largest_enclosed_component_bbox(mask: np.ndarray, *, connectivity: int = 4) -> tuple[int, int, int, int] | None:
    """BBox of the largest connected component in `mask` that does not touch the border."""
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
    - Row marks: only the LEFT frame edge matters.
    - Column marks: TOP and BOTTOM must both show the mover color at the same column.
    - Top/bottom may disagree elsewhere; those single-sided columns are decoys.
    - Distractor colors may appear on partial edges but do not move unless they satisfy the full rule.
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
        cols = sorted(set(col_marks[int(col)]))
        pts = {(int(r), int(c)) for r in row_marks[int(col)] for c in cols}
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
    """Pick up to n values from candidates with at least `min_spacing` between picks."""
    if n == 0:
        return []
    if not candidates:
        return None
    if n == 1:
        return [random.choice(candidates)]
    candidates = list(candidates)
    random.shuffle(candidates)
    picked: list[int] = []
    for v in candidates:
        if all(abs(int(v) - int(p)) >= int(min_spacing) for p in picked):
            picked.append(int(v))
            if len(picked) == int(n):
                return sorted(picked)
    return sorted(picked) if picked else None


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


def _assign_mover_rows(
    mover_colors: list[int],
    sprites: list[np.ndarray],
    ir0: int,
    ir1: int,
) -> dict[int, list[int]] | None:
    occupied_rows: set[int] = set()
    result: dict[int, list[int]] = {}
    for mc, sp in zip(mover_colors, sprites):
        sh = int(sp.shape[0])
        min_row = int(ir0 + sh // 2)
        max_row = int(ir1 - (sh - 1 - sh // 2))
        if min_row > max_row:
            return None
        avail_rows = [r for r in range(min_row, max_row + 1) if r not in occupied_rows]
        row_marks = _pick_spaced(avail_rows, random.choice([1, 1, 2]), max(1, sh))
        if not row_marks:
            return None
        result[int(mc)] = sorted(int(r) for r in row_marks)
        occupied_rows.update(result[int(mc)])
    return result


def _assign_horizontal_marks(
    mover_colors: list[int],
    mover_sprites: list[np.ndarray],
    distractor_sprite: np.ndarray,
    ic0: int,
    ic1: int,
) -> tuple[dict[int, tuple[list[int], list[int]]], tuple[list[int], list[int]]] | None:
    """
    Build non-mirrored top/bottom edges while preserving a strict double-sided column rule.

    For movers:
    - `shared_cols` appear on both top and bottom and are the only valid targets.
    - `top_only` / `bottom_only` are visible decoys caused by the asymmetric frame.

    For the distractor:
    - only one side receives marks, so it never satisfies the column rule.
    """
    order = mover_colors[:]
    random.shuffle(order)
    roles: dict[int, tuple[int, int]] = {
        int(order[0]): (1, 0),
        int(order[1]): (0, 1),
    }
    if len(order) >= 3:
        roles[int(order[2])] = random.choice([(1, 0), (0, 1), (1, 1)])

    by_color_sprite = {int(mc): sp for mc, sp in zip(mover_colors, mover_sprites)}
    occupied_cols: set[int] = set()
    mover_marks: dict[int, tuple[list[int], list[int]]] = {}

    for mc in mover_colors:
        sp = by_color_sprite[int(mc)]
        sw = int(sp.shape[1])
        min_col = int(ic0 + sw // 2)
        max_col = int(ic1 - (sw - 1 - sw // 2))
        if min_col > max_col:
            return None

        shared_n = random.choice([1, 1, 2])
        extra_top_n, extra_bottom_n = roles[int(mc)]
        total_n = int(shared_n + extra_top_n + extra_bottom_n)
        avail_cols = [c for c in range(min_col, max_col + 1) if c not in occupied_cols]
        picked = _pick_spaced(avail_cols, total_n, max(1, sw))
        if not picked or len(picked) != total_n:
            return None

        picked = list(picked)
        random.shuffle(picked)
        shared_cols = sorted(int(v) for v in picked[:shared_n])
        top_only = sorted(int(v) for v in picked[shared_n : shared_n + extra_top_n])
        bottom_only = sorted(int(v) for v in picked[shared_n + extra_top_n :])
        top_cols = sorted(shared_cols + top_only)
        bottom_cols = sorted(shared_cols + bottom_only)
        mover_marks[int(mc)] = (top_cols, bottom_cols)
        occupied_cols.update(top_cols)
        occupied_cols.update(bottom_cols)

    dsw = int(distractor_sprite.shape[1])
    dmin_col = int(ic0 + dsw // 2)
    dmax_col = int(ic1 - (dsw - 1 - dsw // 2))
    if dmin_col > dmax_col:
        return None
    d_side = random.choice(["top", "bottom"])
    d_n = random.choice([1, 1, 2])
    d_avail = [c for c in range(dmin_col, dmax_col + 1) if c not in occupied_cols]
    d_cols = _pick_spaced(d_avail, d_n, max(1, dsw))
    if not d_cols or len(d_cols) != d_n:
        return None
    distractor_marks = (sorted(int(c) for c in d_cols), []) if d_side == "top" else ([], sorted(int(c) for c in d_cols))

    has_top_only = False
    has_bottom_only = False
    for top_cols, bottom_cols in mover_marks.values():
        top_set = set(map(int, top_cols))
        bottom_set = set(map(int, bottom_cols))
        if bool(top_set - bottom_set):
            has_top_only = True
        if bool(bottom_set - top_set):
            has_bottom_only = True
    if not (has_top_only and has_bottom_only):
        return None
    if not (distractor_marks[0] or distractor_marks[1]):
        return None
    return mover_marks, distractor_marks


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
    n_movers = random.choice([2, 2, 3])
    idx = 0
    divider_color = all_colors[idx]
    idx += 1
    frame_base = all_colors[idx]
    idx += 1
    mover_colors = all_colors[idx : idx + n_movers]
    idx += n_movers
    distractor_color = all_colors[idx]

    mover_sprites: list[np.ndarray] = []
    for mc in mover_colors:
        sp = _make_sprite(mc, 5, sprite_area_w)
        if sp is None:
            return None
        mover_sprites.append(sp)

    distractor_sprite = _make_sprite(distractor_color, 4, sprite_area_w)
    if distractor_sprite is None:
        return None

    mover_rows = _assign_mover_rows(mover_colors, mover_sprites, ir0, ir1)
    if mover_rows is None:
        return None

    horizontal_marks = _assign_horizontal_marks(
        mover_colors, mover_sprites, distractor_sprite, ic0, ic1
    )
    if horizontal_marks is None:
        return None
    mover_cols, distractor_cols = horizontal_marks

    inp = np.full((h, w), bg, dtype=int)
    inp[:, div_col] = divider_color

    top_edge = np.full(frame_w, frame_base, dtype=int)
    bottom_edge = np.full(frame_w, frame_base, dtype=int)
    for mc in mover_colors:
        top_cols, bottom_cols = mover_cols[int(mc)]
        for cc in top_cols:
            top_edge[int(cc - fc0)] = int(mc)
        for cc in bottom_cols:
            bottom_edge[int(cc - fc0)] = int(mc)
    for cc in distractor_cols[0]:
        top_edge[int(cc - fc0)] = int(distractor_color)
    for cc in distractor_cols[1]:
        bottom_edge[int(cc - fc0)] = int(distractor_color)
    inp[0, fc0 : fc0 + frame_w] = top_edge
    inp[h - 1, fc0 : fc0 + frame_w] = bottom_edge

    left_edge = np.full(h - 2, frame_base, dtype=int)
    for mc, rmarks in mover_rows.items():
        for rr in rmarks:
            left_edge[int(rr - 1)] = int(mc)
    inp[1 : h - 1, fc0] = left_edge
    inp[1 : h - 1, fc1] = frame_base

    cur_row = 0
    all_items: list[np.ndarray] = mover_sprites + [distractor_sprite]
    for sp in all_items:
        sh, sw = map(int, sp.shape)
        if sw > sprite_area_w or cur_row + sh > h:
            return None
        c_off = random.randint(0, sprite_area_w - sw)
        for dr in range(sh):
            for dc in range(sw):
                if int(sp[dr, dc]) != int(bg):
                    inp[int(cur_row + dr), int(c_off + dc)] = int(sp[dr, dc])
        cur_row += sh

    vals, counts = np.unique(inp, return_counts=True)
    if int(vals[np.argmax(counts)]) != int(bg):
        return None

    asym_mover_cols = 0
    valid_mover_cols = 0
    distractor_partial_cols = 0
    for c in range(int(ic0), int(ic1) + 1):
        top_v = int(inp[0, c])
        bottom_v = int(inp[h - 1, c])
        if top_v in mover_colors or bottom_v in mover_colors:
            if top_v == bottom_v and top_v in mover_colors:
                valid_mover_cols += 1
            elif top_v != bottom_v:
                asym_mover_cols += 1
        if top_v == int(distractor_color) or bottom_v == int(distractor_color):
            if not (top_v == bottom_v == int(distractor_color)):
                distractor_partial_cols += 1
    if valid_mover_cols == 0 or asym_mover_cols < 2 or distractor_partial_cols == 0:
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
