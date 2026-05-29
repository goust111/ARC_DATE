# Variant of 247ef758
# Dimension: D6 — 输出网格形状 (grid_shape)
# Change: 输出与输入同尺寸，整个网格保留（包括左侧条带、分隔线、框架）... → 裁剪至框架（含边缘）

from __future__ import annotations

# Design analysis for the D6 variant
# ==================================
# reference_task_review:
#   requested: a251c730
#   solver_read: /home/lixinyu/snap/new_code/solvers/arc2/a251c730.py
#   relevant_mechanism: the reference task keeps its object-placement logic and
#     changes the final presentation by returning only the destination rectangle.
#     That is the baseline used here: preserve the 247ef758 stamping rule, crop
#     only at the end.
#
# variant_spec:
#   input_layout:
#     unchanged from the base task:
#       - left sprite strip
#       - one full-height divider column
#       - right frame whose top/bottom and left/right edges mirror marker colors
#   output_layout:
#     - return only the full frame rectangle [fr0:fr1, fc0:fc1], inclusive
#     - keep frame edges and their markers
#     - keep the interior stamping result
#     - remove the left strip and divider entirely by cropping, not by recoloring
#
# behavior_trace:
#   concrete_example:
#     - suppose input width is 18, divider is at col 4, frame spans cols 5..17
#     - original solver builds an 12x18 output, erases moved sprites from cols 0..3,
#       keeps divider/frame, then stamps sprites into the frame interior
#     - new variant performs the same erase+stamp work on the full grid, but returns
#       only cols 5..17, giving a 12x13 result
#   difference:
#     - original visible output still shows the source strip and divider
#     - variant visible output shows only the destination frame, so the task now
#       reads as "fill the frame" rather than "move inside the same canvas"
#
# complexity_calibration:
#   - the marker-matching rule stays unchanged, so there is no new ambiguity in
#     color matching, sprite ordering, or placement
#   - D6 is still visually obvious because sprite_area_w is always 3..5, so the
#     output width shrinks by 4..6 columns (strip + divider) on every sample
#   - edge markers are preserved, which keeps the row/column evidence visible
#
# implicit_assumption_review:
#   - largest_enclosed_component_bbox(...): still valid because the input keeps a
#     single bg-filled frame interior enclosed away from the border
#   - divider detection and left-area sprite extraction: still valid because the
#     input is unchanged; crop affects only the returned view
#   - row/col marker scans on frame edges: unchanged and still valid
#   - erase-originals step: harmless even though the erased region will later be
#     cropped away; it preserves the original bottom-to-top overwrite semantics
#   - final return value in the base task relied on "full grid is output"; this is
#     the one assumption that no longer holds, so the only solver change is to
#     slice the solved full-grid canvas down to the frame bbox
#
# placement_legality:
#   - generator still guarantees every chosen target center lets the sprite fit
#     completely inside the frame interior, so stamping never depends on clipping
#   - row/col markers are unique across movers, preventing ambiguous ownership
#   - sprites remain confined to the left strip and never overlap the divider/frame
#
# visual_observability:
#   - across 2-4 examples, the most visible change is structural: outputs are
#     narrower and contain only the framed region
#   - because markers remain on the border, the solver evidence is still readable
#   - optional distractors help show that the removed left strip is discarded as a
#     region, not solved object-by-object
#
# cleanliness_and_color_inference:
#   - no new color mapping is introduced
#   - the frame remains a clean rectangle with sparse edge markers and sparse
#     interior stamps; cropping reduces clutter rather than adding it

from collections import defaultdict
import random

import numpy as np

from arc_utils import get_bg, label_mask


def largest_enclosed_component_bbox(mask: np.ndarray, *, connectivity: int = 4) -> tuple[int, int, int, int] | None:
    """
    Local helper (only used by this solver): bbox of largest connected component in `mask`
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
        if bool((rs == 0).any() or (rs == int(H - 1)).any() or (cs == 0).any() or (cs == int(W - 1)).any()):
            continue
        area = int(coords.shape[0])
        if area > best_area:
            best_area = area
            r0, c0 = coords.min(axis=0)
            r1, c1 = coords.max(axis=0)
            best_bbox = (int(r0), int(c0), int(r1), int(c1))

    return best_bbox


def _frame_bounds(g: np.ndarray, bg: int) -> tuple[int, int, int, int, int, int, int, int]:
    """Return interior bbox and full frame bbox: ir0, ic0, ir1, ic1, fr0, fc0, fr1, fc1."""
    inner_bbox = largest_enclosed_component_bbox(g == int(bg), connectivity=4)
    assert inner_bbox is not None
    ir0, ic0, ir1, ic1 = map(int, inner_bbox)
    fr0, fc0, fr1, fc1 = int(ir0 - 1), int(ic0 - 1), int(ir1 + 1), int(ic1 + 1)
    return int(ir0), int(ic0), int(ir1), int(ic1), int(fr0), int(fc0), int(fr1), int(fc1)


def _solve_full_grid(grid: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """
    Base-task solver on the original full canvas.

    Returning the frame bbox separately lets `main` implement the D6 crop as a
    pure presentation change after all original placement logic has finished.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))
    ir0, ic0, ir1, ic1, fr0, fc0, fr1, fc1 = _frame_bounds(g, bg)

    divider_mask = np.zeros((H, W), dtype=bool)
    for c in range(0, max(0, int(fc0))):
        colv = g[:, int(c)]
        non = colv[colv != int(bg)]
        if int(len(non)) >= int(H - 1) and len(set(map(int, non))) == 1:
            divider_mask[:, int(c)] = True

    left_cols = np.arange(W)[None, :] < int(fc0)

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
            sprites.append({"color": int(col), "r0": int(r0), "c0": int(c0), "r1": int(r1), "c1": int(c1), "mask": m})

    move_colors = {int(s["color"]) for s in sprites}

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

    return out, (int(fr0), int(fc0), int(fr1), int(fc1))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move colored sprites from the left strip into the framed area, then crop the
    output down to the frame rectangle including its edge markers.
    """
    full_out, (fr0, fc0, fr1, fc1) = _solve_full_grid(grid)
    return full_out[int(fr0) : int(fr1) + 1, int(fc0) : int(fc1) + 1].copy()


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

    inp = np.full((H, W), bg, dtype=int)
    inp[:, div_col] = divider_color

    top_edge = np.full(frame_w, frame_base, dtype=int)
    for mc, (_, cmarks) in mover_targets.items():
        for cc in cmarks:
            top_edge[cc - fc0] = mc
    inp[0, fc0 : fc0 + frame_w] = top_edge
    inp[H - 1, fc0 : fc0 + frame_w] = top_edge

    for r in range(1, H - 1):
        val = frame_base
        for mc, (rmarks, _) in mover_targets.items():
            if r in rmarks:
                val = mc
                break
        inp[r, fc0] = val
        inp[r, fc1] = val

    cur_row = 0
    all_items: list[tuple[int, np.ndarray, bool]] = [(mc, sp, False) for mc, sp in zip(mover_colors, sprites)]
    for dc_color in distractor_colors:
        ds = _make_sprite(dc_color, min(5, H - cur_row), sprite_area_w)
        if ds is None:
            return None
        all_items.append((dc_color, ds, True))

    for _col, sp, _is_distractor in all_items:
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
        output = main(inp)
    except Exception:
        return None

    if output.shape != (H, frame_w):
        return None
    if output.shape == inp.shape:
        return None
    if np.array_equal(output, inp[:, fc0 : fc1 + 1]):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(300):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input: failed after 300 attempts")
