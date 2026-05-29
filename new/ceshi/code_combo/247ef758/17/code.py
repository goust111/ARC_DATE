# Combo variant of 247ef758 (index 17)
# combo_spec: D1/2+D3/2+D10/2
# Name: 三重边缘解析挑战

from __future__ import annotations

from collections import defaultdict
import random

import numpy as np

from arc_utils import get_bg, label_mask


# Design analysis for the combo variant
# -------------------------------------
# concretization:
# - Overall layout stays recognizable: left sprite strip | one full-height divider column | framed area on the right.
# - Movers are still monochrome 8-connected blobs placed in the left strip. The framed interior is still a blank bg area.
# - The fused edge logic is intentionally asymmetric in BOTH axes:
#   1. Row targets only read the LEFT edge. Same-color marks on the right edge are decoys.
#   2. Column targets read TOP and BOTTOM independently. A mover color can contribute columns from top-only,
#      bottom-only, or both; the final column set is the union.
# - D10-style fake markers are realized on the RIGHT edge: they reuse mover colors but appear without a left partner,
#   so they look tempting under the original symmetry assumption yet must be ignored by the combo solver.
#
# behavior trace on a concrete mini-example:
# - Suppose sprite color 4 exists on the left strip.
# - Edges show:
#   left edge row 5 = 4            -> real row mark
#   right edge row 8 = 4           -> fake row mark (single-sided decoy)
#   top edge col 10 = 4            -> valid target column
#   bottom edge col 13 = 4         -> another valid target column
# - Original logic would require left/right match for rows AND top/bottom match for columns.
#   Therefore row 5 is rejected, row 8 is rejected, col 10 is rejected, col 13 is rejected, so color 4 would not move.
# - Combo logic:
#   row set(4) = {5} because only left edge matters.
#   col set(4) = {10, 13} because top and bottom contribute independently.
#   Targets become {(5,10), (5,13)} and the sprite is stamped twice.
# - Visible difference:
#   the input contains mover-colored noise on the right edge and non-mirrored top/bottom columns, yet only the
#   left-supported rows and the union of top/bottom columns control replication.
#
# complexity calibration:
# - Keep 2-3 movers and 0-1 distractors as in the base task so the same fused rule repeats across multiple colors.
# - Force each sample to contain:
#   * at least one top-only column marker,
#   * at least one bottom-only column marker,
#   * at least one right-only fake row marker.
#   This prevents accidental regression to the original symmetric task.
# - Row and column locations remain globally unique across colors, which keeps overlaps readable even when several
#   movers replicate into the frame.
#
# implicit assumption audit for main():
# - `largest_enclosed_component_bbox(...)` still assumes the frame interior is the largest enclosed bg component.
#   The generator preserves the same frame geometry, so this remains valid.
# - Divider detection still assumes a nearly full-height uniform non-bg column to the left of the frame.
#   The generator still creates exactly one such divider column.
# - Sprite extraction still assumes left-strip monochrome 8-connected components separated from the divider.
#   Generator preserves that unchanged.
# - The original solver assumed row validity required `left == right` and column validity required `top == bottom`.
#   Both assumptions are broken here:
#   * right edge is now explicitly noisy for mover colors and must NOT gate row extraction,
#   * top and bottom are now independent signal sources and must be merged, not intersected.
# - The new solver therefore implicitly assumes fake mover-colored row marks, if present, appear ONLY on the right.
#   The generator guarantees this; it never places fake row marks on the left because left marks are real by rule.
# - The new solver also assumes any mover-colored top/bottom mark is semantically meaningful for columns.
#   The generator guarantees that by not placing mover-colored fake column marks that would conflict with D3.
#
# placement legality:
# - True row marks are sampled from center-safe rows for each sprite, so centered stamping fits inside the frame.
# - Column marks are sampled from center-safe columns and are unique across all top/bottom placements.
# - Fake right-edge row marks use different rows from every true row mark, so they never accidentally become real.
# - Sprites are stacked in the left strip without overlap; distractors remain absent from the frame markers.
#
# observability / cleanliness:
# - Humans can directly see that top and bottom edges disagree, yet output still uses both sets of columns.
# - Humans can also see mover-colored right-edge marks that never trigger placements, pushing them toward
#   "left edge is the authoritative row source" instead of a generic "any edge occurrence" hypothesis.
# - No hidden property such as numeric color ordering is used; only visible color/position relations matter.
#
# new_main_assumptions checked against generator:
# - Every moved color has at least one left row mark and at least one top-or-bottom column mark.
# - Right-only fake rows are disjoint from real left rows.
# - At least one mover gets a top-only column and at least one mover gets a bottom-only column in every sample.


def largest_enclosed_component_bbox(mask: np.ndarray, *, connectivity: int = 4) -> tuple[int, int, int, int] | None:
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


def _pick_spaced(candidates: list[int], n: int, min_spacing: int) -> list[int] | None:
    """Pick up to n values from candidates with at least min_spacing between any two."""
    if not candidates:
        return None
    if n == 1:
        return [random.choice(candidates)]
    candidates = list(candidates)
    random.shuffle(candidates)
    picked: list[int] = []
    for v in candidates:
        if all(abs(v - p) >= min_spacing for p in picked):
            picked.append(v)
            if len(picked) == n:
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


def _collect_combo_targets(
    g: np.ndarray,
    ir0: int,
    ic0: int,
    ir1: int,
    ic1: int,
    fr0: int,
    fr1: int,
    fc0: int,
    move_colors: set[int],
) -> dict[int, list[tuple[int, int]]]:
    """Row targets come from the left edge only; column targets are top/bottom union."""
    row_marks: dict[int, list[int]] = defaultdict(list)
    col_marks: dict[int, list[int]] = defaultdict(list)

    for r in range(int(ir0), int(ir1) + 1):
        cl = int(g[int(r), int(fc0)])
        if cl in move_colors:
            row_marks[int(cl)].append(int(r))

    for c in range(int(ic0), int(ic1) + 1):
        ct = int(g[int(fr0), int(c)])
        cb = int(g[int(fr1), int(c)])
        if ct in move_colors:
            col_marks[int(ct)].append(int(c))
        if cb in move_colors:
            col_marks[int(cb)].append(int(c))

    targets: dict[int, list[tuple[int, int]]] = {}
    for col in set(row_marks.keys()) & set(col_marks.keys()):
        cols = sorted(set(col_marks[int(col)]))
        pts = {(int(r), int(c)) for r in row_marks[int(col)] for c in cols}
        targets[int(col)] = sorted(pts)
    return targets


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move colored sprites from the left strip into a framed area at marker-defined targets.

    Combo rule:
    - Row marks: only the LEFT frame edge counts.
    - Column marks: TOP and BOTTOM edges each contribute independently; columns are unioned.
    - Right-edge mover-colored marks are decoys and must be ignored.
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
    targets = _collect_combo_targets(g, ir0, ic0, ir1, ic1, fr0, fr1, fc0, move_colors)

    out = g.copy()
    movers = [s for s in sprites if int(s["color"]) in targets]

    for s in movers:
        r0, c0 = int(s["r0"]), int(s["c0"])
        for rr, cc in np.argwhere(s["mask"]):
            out[int(r0) + int(rr), int(c0) + int(cc)] = int(bg)

    # Lower sprites stamp first; upper sprites overwrite later.
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


def _assign_true_rows(
    mover_colors: list[int],
    sprites: list[np.ndarray],
    ir0: int,
    ir1: int,
    ic0: int,
    ic1: int,
) -> dict[int, list[int]] | None:
    """Assign real row markers on the left edge only."""
    occupied_rows: set[int] = set()
    result: dict[int, list[int]] = {}
    for mc, sp in zip(mover_colors, sprites):
        sh, sw = sp.shape
        min_row = int(ir0 + sh // 2)
        max_row = int(ir1 - (sh - 1 - sh // 2))
        min_col = int(ic0 + sw // 2)
        max_col = int(ic1 - (sw - 1 - sw // 2))
        if min_row > max_row or min_col > max_col:
            return None
        avail_rows = [r for r in range(min_row, max_row + 1) if r not in occupied_rows]
        row_marks = _pick_spaced(avail_rows, random.randint(1, 2), max(1, sh))
        if not row_marks:
            return None
        result[int(mc)] = sorted(row_marks)
        occupied_rows.update(row_marks)
    return result


def _assign_column_marks(
    mover_colors: list[int],
    sprites: list[np.ndarray],
    ic0: int,
    ic1: int,
) -> dict[int, tuple[list[int], list[int]]] | None:
    """
    Assign asymmetric column markers.

    Returns:
      {color: (top_cols, bottom_cols)}

    Guarantees:
    - at least one top-only mover
    - at least one bottom-only mover
    - all chosen columns are globally unique across both edges
    """
    order = mover_colors[:]
    random.shuffle(order)
    roles: dict[int, str] = {int(order[0]): "top", int(order[1]): "bottom"}
    if len(order) >= 3:
        roles[int(order[2])] = random.choice(["both", "top", "bottom"])

    by_color_sprite = {int(mc): sp for mc, sp in zip(mover_colors, sprites)}
    occupied_cols: set[int] = set()
    result: dict[int, tuple[list[int], list[int]]] = {}

    for mc in mover_colors:
        sp = by_color_sprite[int(mc)]
        sw = int(sp.shape[1])
        min_col = int(ic0 + sw // 2)
        max_col = int(ic1 - (sw - 1 - sw // 2))
        if min_col > max_col:
            return None

        role = roles[int(mc)]
        if role == "top":
            top_n, bottom_n = random.choice([1, 2]), 0
        elif role == "bottom":
            top_n, bottom_n = 0, random.choice([1, 2])
        else:
            top_n, bottom_n = 1, 1

        total_n = int(top_n + bottom_n)
        avail_cols = [c for c in range(min_col, max_col + 1) if c not in occupied_cols]
        picked = _pick_spaced(avail_cols, total_n, max(1, sw))
        if not picked or len(picked) != total_n:
            return None

        picked = list(picked)
        random.shuffle(picked)
        top_cols = sorted(picked[:top_n])
        bottom_cols = sorted(picked[top_n:])
        occupied_cols.update(top_cols)
        occupied_cols.update(bottom_cols)
        result[int(mc)] = (top_cols, bottom_cols)

    if not any(len(top) > 0 and len(bottom) == 0 for top, bottom in result.values()):
        return None
    if not any(len(bottom) > 0 and len(top) == 0 for top, bottom in result.values()):
        return None
    return result


def _assign_fake_right_rows(
    mover_colors: list[int],
    sprites: list[np.ndarray],
    ir0: int,
    ir1: int,
    true_rows: dict[int, list[int]],
) -> dict[int, list[int]] | None:
    """
    Assign D10-style fake row markers on the RIGHT edge only.

    These reuse mover colors but must not overlap any true left-edge row mark.
    """
    by_color_sprite = {int(mc): sp for mc, sp in zip(mover_colors, sprites)}
    occupied_rows = {int(r) for rows in true_rows.values() for r in rows}
    result: dict[int, list[int]] = {int(mc): [] for mc in mover_colors}

    order = mover_colors[:]
    random.shuffle(order)
    n_fake_colors = random.randint(1, len(mover_colors))
    chosen = order[:n_fake_colors]

    for mc in chosen:
        sp = by_color_sprite[int(mc)]
        sh = int(sp.shape[0])
        min_row = int(ir0 + sh // 2)
        max_row = int(ir1 - (sh - 1 - sh // 2))
        if min_row > max_row:
            return None
        avail_rows = [r for r in range(min_row, max_row + 1) if r not in occupied_rows]
        fake_rows = _pick_spaced(avail_rows, 1, max(1, sh))
        if not fake_rows:
            return None
        result[int(mc)] = sorted(fake_rows)
        occupied_rows.update(fake_rows)

    if not any(rows for rows in result.values()):
        return None
    return result


def _count_marker_modes(
    mover_colors: list[int],
    true_rows: dict[int, list[int]],
    fake_right_rows: dict[int, list[int]],
    mover_cols: dict[int, tuple[list[int], list[int]]],
) -> tuple[int, int, int]:
    """Return (top_only_count, bottom_only_count, right_only_fake_count)."""
    top_only = 0
    bottom_only = 0
    for mc in mover_colors:
        top_cols, bottom_cols = mover_cols[int(mc)]
        top_only += len([c for c in top_cols if c not in bottom_cols])
        bottom_only += len([c for c in bottom_cols if c not in top_cols])
    fake_count = sum(len(fake_right_rows[int(mc)]) for mc in mover_colors)
    # `true_rows` is unused in the math, but keeping it in the signature clarifies the intended comparison:
    # fake rows are defined relative to the true left-edge row assignments.
    _ = true_rows
    return int(top_only), int(bottom_only), int(fake_count)


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

    true_rows = _assign_true_rows(mover_colors, sprites, ir0, ir1, ic0, ic1)
    if true_rows is None:
        return None

    mover_cols = _assign_column_marks(mover_colors, sprites, ic0, ic1)
    if mover_cols is None:
        return None

    fake_right_rows = _assign_fake_right_rows(mover_colors, sprites, ir0, ir1, true_rows)
    if fake_right_rows is None:
        return None

    top_only_count, bottom_only_count, fake_count = _count_marker_modes(
        mover_colors, true_rows, fake_right_rows, mover_cols
    )
    if top_only_count == 0 or bottom_only_count == 0 or fake_count == 0:
        return None

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
    inp[0, fc0 : fc0 + frame_w] = top_edge
    inp[h - 1, fc0 : fc0 + frame_w] = bottom_edge

    left_edge = np.full(h - 2, frame_base, dtype=int)
    right_edge = np.full(h - 2, frame_base, dtype=int)
    for mc in mover_colors:
        for rr in true_rows[int(mc)]:
            left_edge[int(rr - 1)] = int(mc)
        for rr in fake_right_rows[int(mc)]:
            right_edge[int(rr - 1)] = int(mc)
    inp[1 : h - 1, fc0] = left_edge
    inp[1 : h - 1, fc1] = right_edge

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
                    inp[cur_row + dr, c_off + dc] = int(sp[dr, dc])
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
    for _ in range(400):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input: failed after 400 attempts")
