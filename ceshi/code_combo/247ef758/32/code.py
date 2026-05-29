# Combo variant of 247ef758 (index 32)
# combo_spec: D1/2+D10/2
# Name: 单侧匹配+伪同色标记

from __future__ import annotations

from collections import defaultdict
import random

import numpy as np

from arc_utils import get_bg, label_mask


# Design analysis for the combo variant
# -------------------------------------
# concretization:
# - The global layout stays the same as the base task: left sprite strip | full-height divider | rectangular frame.
# - Movers are still monochrome 8-connected blobs in the left strip. The frame interior is still a bg rectangle.
# - Row matching changes to the D1 rule: a mover color only needs to appear on the LEFT frame edge at a row.
#   The right edge is no longer a confirmation edge for row targets.
# - Column matching keeps the original mirrored rule: a mover color must appear on BOTH top and bottom edges at
#   the same interior column.
# - D10 is fused into the frame construction: top/bottom edges also contain extra mover-colored marks that appear
#   on only one side. These are visually tempting pseudo column marks, but they must be ignored because columns
#   still require top/bottom confirmation.
#
# behavior trace on a concrete mini-example:
# - Suppose sprite color 4 is in the left strip.
# - Frame edges show:
#   left row 5 = 4, right row 5 = frame_base,
#   top col 9 = 4 and bottom col 9 = 4,
#   top col 12 = 4 and bottom col 12 = frame_base.
# - Original solver:
#   row 5 is rejected because left/right do not both equal 4, so sprite 4 does not move at all.
# - D1-only part of the combo:
#   row 5 is accepted from the left edge alone; col 9 is accepted because top/bottom mirror; target (5, 9) exists.
# - D10 part of the combo:
#   col 12 must still be rejected because it is only a top-side same-color hint, not a mirrored column marker.
# - Visible difference:
#   the sprite moves despite asymmetric row edges, but it stamps only at mirrored columns, not at one-sided
#   same-color column hints.
#
# complexity calibration:
# - Keep 2-3 movers and 0-1 distractors so the asymmetric row rule and the fake column marks repeat across colors.
# - Each mover still gets 1-2 valid row marks and 1-2 valid mirrored column marks, with no row/col reuse across
#   movers. This keeps target ownership readable.
# - Add 1..n_movers pseudo columns overall, each using a real mover color but placed on only top OR bottom. That
#   is enough to make the D10 effect visible without cluttering the frame.
#
# implicit assumptions audit for main():
# - largest_enclosed_component_bbox() assumes the frame interior is the largest enclosed bg component. The generator
#   preserves the same rectangular frame geometry and keeps the interior empty before stamping.
# - Divider detection assumes at least one almost-full-height uniform non-bg column to the left of the frame.
#   The generator still creates exactly one divider column and prevents same-color sprite columns from mimicking it.
# - Sprite extraction assumes left-strip components are monochrome and 8-connected. Generation preserves this.
# - Row-target extraction no longer depends on right-edge symmetry. The new assumption is stronger but explicit:
#   only the LEFT edge authorizes rows; right-edge colors are irrelevant to row validity.
# - Column-target extraction still depends on mirrored top/bottom equality. The generator guarantees pseudo
#   mover-colored column hints never accidentally mirror, so they remain safely ignorable.
# - Target construction assumes colors intersect across the two dimensions. Generation guarantees every mover has
#   at least one valid left row and one valid mirrored column, so every mover actually moves.
#
# placement legality:
# - Valid rows/cols are sampled from center positions where the sprite fits fully inside the frame interior.
# - Row and valid-column marks are unique across movers; pseudo columns also avoid all valid columns and each other.
# - Sprites are stacked top-to-bottom in the left strip without overlap; distractors remain outside the target map.
#
# observability / cleanliness / inferability:
# - Humans can see that the left edge, not the right edge, controls rows because movers replicate even when right
#   row markers are absent.
# - Humans can also see same-color marks on only one of top/bottom that do not create extra copies, revealing that
#   columns still require mirrored confirmation.
# - No hidden numeric ordering is used. All evidence is spatial and repeated across multiple movers, which keeps
#   the fusion learnable from a few samples rather than relying on a one-off coincidence.


def largest_enclosed_component_bbox(
    mask: np.ndarray, *, connectivity: int = 4
) -> tuple[int, int, int, int] | None:
    """Return the bbox of the largest connected component in mask that does not touch the border."""
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


def _extract_scene(grid: np.ndarray) -> tuple[int, tuple[int, int, int, int, int, int, int, int], list[dict]]:
    """
    Extract background, frame geometry, and left-strip sprites.

    Returns:
        bg,
        (fr0, fc0, fr1, fc1, ir0, ic0, ir1, ic1),
        sprites list with per-component metadata.
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
    sprites: list[dict] = []
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
            comp_mask = slab[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1] == int(sid)
            sprites.append(
                {
                    "color": int(col),
                    "r0": int(r0),
                    "c0": int(c0),
                    "r1": int(r1),
                    "c1": int(c1),
                    "mask": comp_mask,
                }
            )

    return bg, (fr0, fc0, fr1, fc1, ir0, ic0, ir1, ic1), sprites


def _build_targets(
    g: np.ndarray,
    move_colors: set[int],
    frame_geom: tuple[int, int, int, int, int, int, int, int],
) -> tuple[dict[int, list[int]], dict[int, list[int]], dict[int, list[tuple[int, int]]]]:
    """
    Build row marks, column marks, and Cartesian-product targets.

    Combo rule:
    - rows: left edge alone authorizes the row
    - cols: top and bottom must mirror
    """
    fr0, fc0, fr1, fc1, ir0, ic0, ir1, ic1 = map(int, frame_geom)

    row_marks: dict[int, list[int]] = defaultdict(list)
    col_marks: dict[int, list[int]] = defaultdict(list)

    for r in range(int(ir0), int(ir1) + 1):
        cl = int(g[int(r), int(fc0)])
        if cl in move_colors:
            row_marks[int(cl)].append(int(r))

    for c in range(int(ic0), int(ic1) + 1):
        ct = int(g[int(fr0), int(c)])
        cb = int(g[int(fr1), int(c)])
        if ct == cb and ct in move_colors:
            col_marks[int(ct)].append(int(c))

    targets: dict[int, list[tuple[int, int]]] = {}
    for col in set(row_marks.keys()) & set(col_marks.keys()):
        pts = {(int(r), int(c)) for r in row_marks[int(col)] for c in col_marks[int(col)]}
        targets[int(col)] = sorted(pts)

    return row_marks, col_marks, targets


def _analyze_fusion_features(grid: np.ndarray) -> dict[str, int]:
    """
    Measure whether the generated input actually exhibits the combo's key cues.

    Metrics:
    - left_only_row_marks: mover-colored left-edge rows whose right edge is different
    - mirrored_col_marks: mover-colored columns present on both top and bottom
    - pseudo_same_color_cols: mover-colored columns present on exactly one of top/bottom
    - target_points: total valid Cartesian-product targets under the combo solver
    """
    g = np.array(grid, dtype=int)
    _, frame_geom, sprites = _extract_scene(g)
    fr0, fc0, fr1, fc1, ir0, ic0, ir1, ic1 = map(int, frame_geom)
    move_colors = {int(s["color"]) for s in sprites}
    row_marks, col_marks, targets = _build_targets(g, move_colors, frame_geom)

    left_only_row_marks = 0
    for col, rows in row_marks.items():
        for r in rows:
            if int(g[int(r), int(fc1)]) != int(col):
                left_only_row_marks += 1

    mirrored_col_marks = sum(len(v) for v in col_marks.values())
    pseudo_same_color_cols = 0
    for c in range(int(ic0), int(ic1) + 1):
        ct = int(g[int(fr0), int(c)])
        cb = int(g[int(fr1), int(c)])
        top_ok = ct in move_colors
        bot_ok = cb in move_colors
        if top_ok ^ bot_ok:
            pseudo_same_color_cols += 1

    target_points = sum(len(v) for v in targets.values())
    return {
        "left_only_row_marks": int(left_only_row_marks),
        "mirrored_col_marks": int(mirrored_col_marks),
        "pseudo_same_color_cols": int(pseudo_same_color_cols),
        "target_points": int(target_points),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move left-strip sprites into the framed area.

    Fusion rule:
    - row match: left edge only
    - column match: top and bottom must both match
    - extra one-sided same-color top/bottom hints are ignored
    """
    g = np.array(grid, dtype=int)
    bg, frame_geom, sprites = _extract_scene(g)
    _, _, _, _, ir0, ic0, ir1, ic1 = map(int, frame_geom)

    move_colors = {int(s["color"]) for s in sprites}
    _, _, targets = _build_targets(g, move_colors, frame_geom)

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


def _pick_spaced(candidates: list[int], n: int, min_spacing: int) -> list[int] | None:
    """Pick up to n values from candidates with at least min_spacing between any two."""
    if not candidates:
        return None
    if n == 1:
        return [random.choice(candidates)]
    order = candidates[:]
    random.shuffle(order)
    picked: list[int] = []
    for v in order:
        if all(abs(v - p) >= min_spacing for p in picked):
            picked.append(v)
            if len(picked) == n:
                return sorted(picked)
    return sorted(picked) if picked else None


def _make_sprite(color: int, max_h: int, max_w: int) -> np.ndarray | None:
    """Generate one monochrome 8-connected sprite cropped to its bbox."""
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
        sp = _make_sprite(mc, 5, sprite_area_w)
        if sp is None:
            return None
        sprites.append(sp)

    mover_targets: dict[int, tuple[list[int], list[int]]] = {}
    legal_cols_by_color: dict[int, list[int]] = {}
    occupied_rows: set[int] = set()
    occupied_valid_cols: set[int] = set()
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

        legal_cols = list(range(min_col, max_col + 1))
        legal_cols_by_color[int(mc)] = legal_cols
        avail_cols = [c for c in legal_cols if c not in occupied_valid_cols]
        col_marks = _pick_spaced(avail_cols, random.randint(1, 2), sw)
        if not col_marks:
            return None

        mover_targets[int(mc)] = (sorted(row_marks), sorted(col_marks))
        occupied_rows.update(row_marks)
        occupied_valid_cols.update(col_marks)

    pseudo_edge_cols: dict[int, tuple[int, str]] = {}
    occupied_all_edge_cols = set(occupied_valid_cols)
    pseudo_owner_count = random.randint(1, n_movers)
    pseudo_owners = random.sample(mover_colors, pseudo_owner_count)
    side_order = ["top", "bottom"] * max(1, n_movers)
    random.shuffle(side_order)
    for i, mc in enumerate(pseudo_owners):
        legal_cols = legal_cols_by_color[int(mc)]
        avail = [c for c in legal_cols if c not in occupied_all_edge_cols]
        if not avail:
            return None
        pc = random.choice(avail)
        side = side_order[i]
        pseudo_edge_cols[int(mc)] = (int(pc), side)
        occupied_all_edge_cols.add(int(pc))

    inp = np.full((h, w), bg, dtype=int)
    inp[:, div_col] = divider_color

    top_edge = np.full(frame_w, frame_base, dtype=int)
    bottom_edge = np.full(frame_w, frame_base, dtype=int)
    for mc, (_, cmarks) in mover_targets.items():
        for cc in cmarks:
            top_edge[int(cc - fc0)] = int(mc)
            bottom_edge[int(cc - fc0)] = int(mc)
    for mc, (pc, side) in pseudo_edge_cols.items():
        if side == "top":
            top_edge[int(pc - fc0)] = int(mc)
        else:
            bottom_edge[int(pc - fc0)] = int(mc)
    inp[0, fc0 : fc0 + frame_w] = top_edge
    inp[h - 1, fc0 : fc0 + frame_w] = bottom_edge

    left_edge = np.full(h - 2, frame_base, dtype=int)
    for mc, (rmarks, _) in mover_targets.items():
        for rr in rmarks:
            left_edge[int(rr - 1)] = int(mc)
    inp[1 : h - 1, fc0] = left_edge

    # Right edge stays neutral so the left-only row rule is legible and not confounded by side-choice noise.
    inp[1 : h - 1, fc1] = frame_base

    cur_row = 0
    all_items: list[np.ndarray] = sprites[:]
    for dc_color in distractor_colors:
        ds = _make_sprite(dc_color, min(5, h - cur_row), sprite_area_w)
        if ds is None:
            return None
        all_items.append(ds)

    for sp in all_items:
        sh, sw = sp.shape
        if sw > sprite_area_w or cur_row + sh > h:
            return None
        c_off = random.randint(0, sprite_area_w - sw)
        for dr in range(sh):
            for dc in range(sw):
                if sp[dr, dc] != bg:
                    inp[int(cur_row + dr), int(c_off + dc)] = int(sp[dr, dc])
        cur_row += sh

    vals, counts = np.unique(inp, return_counts=True)
    if int(vals[np.argmax(counts)]) != bg:
        return None

    metrics = _analyze_fusion_features(inp)
    if metrics["left_only_row_marks"] <= 0:
        return None
    if metrics["mirrored_col_marks"] <= 0:
        return None
    if metrics["pseudo_same_color_cols"] <= 0:
        return None
    if metrics["target_points"] <= 0:
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
