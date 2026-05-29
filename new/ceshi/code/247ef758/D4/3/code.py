# Variant of 247ef758
# Dimension: D4 — 变换：印刻与擦除行为 (transform)
# Change: 擦除左侧原始精灵，在目标位置居中印刻精灵图案，像素超出框架内部时裁剪（clipping）... → 印刻后填充外接矩形

from __future__ import annotations

"""
Design analysis for the D4 variant
==================================

source_task:
  reference_task: none
  action: skip

variant_spec:
  invariant_layout:
    - grid is still "left sprite strip | divider column | right frame"
    - frame markers still define target rows/cols by color matching on opposite edges
    - moved sprites are still erased from the left strip
    - stamping order is still bottom-to-top by original sprite position
  changed_transform:
    - original: translate only the sprite's occupied pixels; clip per pixel at frame interior
    - variant: translate the sprite exactly as before, then fill the translated sprite bbox as a solid
      rectangle in the sprite color; clipping applies to the rectangle's intersection with the frame interior
  visual consequence:
    - a sparse left-side blob becomes one or more solid rectangles inside the frame
    - the bbox size is inherited from the input sprite, so the frame still encodes where each sprite lands

behavior_trace:
  small_example:
    sprite_bbox: 3x4
    sprite_mask:
      - 1100
      - 0110
      - 0010
    original_output_at_target:
      - 1100
      - 0110
      - 0010
    variant_output_at_target:
      - 1111
      - 1111
      - 1111
  affected_pipeline_stage:
    - only the final painting step changes; target discovery, mover detection, erase step, and cover order stay the same

complexity_calibration:
  - If a mover sprite were already a full rectangle, the new rule would be visually indistinguishable from the
    original. The generator therefore requires movers to be sparse within their bbox, with at least two bbox
    cells missing when possible.
  - Filled rectangles occupy more area than sparse sprites, so row/col marker selection now enforces cross-mover
    clearance based on sprite bbox height/width. This keeps outputs readable and avoids accidental ambiguity.
  - The task still uses only 2-3 mover colors and 0-1 distractors, preserving the original low-entropy matching rule.

implicit_assumption_audit:
  - Frame discovery via "largest enclosed bg component" still depends only on the input; unchanged.
  - Divider detection assumes a near-full-height uniform non-bg column left of the frame; unchanged generator preserves that.
  - Sprite extraction assumes left-strip components are single-color 8-connected blobs; unchanged and reinforced by generator.
  - Target matching assumes mover colors are the only colors that appear mirrored on both relevant frame edges; unchanged.
  - The original stamping loop assumed "writing only occupied pixels" was sufficient. That assumption is replaced with
    "writing the translated bbox is required", so the solver must use bbox coordinates rather than foreground pixels.
  - Cover order remains valid because later movers simply overwrite earlier rectangles, just as they overwrote earlier pixels.

placement_legality:
  - Every mover target center is chosen so the full sprite bbox fits inside the frame interior before clipping.
    This keeps the new rectangles fully visible and makes the D4 change obvious instead of relying on edge clipping.
  - Row and column markers reserve clearance using bbox dimensions across all movers, not just within one mover.
  - Left-strip sprites remain non-overlapping and within the strip width.

observability_checks:
  - Human-solvable cue: the left object is a sparse blob, but the right-side copies are solid rectangles with the same
    footprint and color. This difference is visible in a single example and repeatable across examples.
  - No invisible attributes are used: ordering still depends on visible vertical position, not numeric color ids.
  - Colors do not change; only occupancy inside each copied bbox changes, so the rule is structurally inferable.

main/generator contract:
  - main() assumes mover sprites have at least one target color match and that the frame interior exists.
  - generate_input() guarantees these assumptions and additionally rejects any sample where the variant output equals
    the original sparse-stamping output, preventing silent regression to the source task.
"""

from collections import defaultdict
import random

import numpy as np

from arc_utils import get_bg, label_mask


def largest_enclosed_component_bbox(mask: np.ndarray, *, connectivity: int = 4) -> tuple[int, int, int, int] | None:
    """
    Local helper: bbox of the largest connected component in `mask`
    that does not touch the grid border.
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


def _extract_scene(grid: np.ndarray) -> tuple[np.ndarray, int, tuple[int, int, int, int], list[dict], dict[int, list[tuple[int, int]]]]:
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
            m = slab[int(r0): int(r1) + 1, int(c0): int(c1) + 1] == int(sid)
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

    return g, bg, (ir0, ic0, ir1, ic1), sprites, targets


def _solve(grid: np.ndarray, *, fill_bbox: bool) -> np.ndarray:
    g, bg, inner_bbox, sprites, targets = _extract_scene(grid)
    ir0, ic0, ir1, ic1 = map(int, inner_bbox)

    out = g.copy()
    movers = [s for s in sprites if int(s["color"]) in targets]

    for s in movers:
        r0, c0 = int(s["r0"]), int(s["c0"])
        for rr, cc in np.argwhere(s["mask"]):
            out[int(r0) + int(rr), int(c0) + int(cc)] = int(bg)

    movers.sort(key=lambda s: (-((int(s["r0"]) + int(s["r1"])) / 2.0), int(s["c0"])))

    for s in movers:
        col = int(s["color"])
        r0, c0, r1, c1 = map(int, (s["r0"], s["c0"], s["r1"], s["c1"]))
        cr = (r0 + r1) // 2
        cc = (c0 + c1) // 2
        pix = [(r0 + int(rr), c0 + int(cc2)) for rr, cc2 in np.argwhere(s["mask"])]

        for tr, tc in targets[int(col)]:
            dr = int(tr - cr)
            dc = int(tc - cc)
            if fill_bbox:
                rr0 = max(int(ir0), int(r0 + dr))
                rr1 = min(int(ir1), int(r1 + dr))
                cc0 = max(int(ic0), int(c0 + dc))
                cc1 = min(int(ic1), int(c1 + dc))
                if rr0 <= rr1 and cc0 <= cc1:
                    out[int(rr0): int(rr1) + 1, int(cc0): int(cc1) + 1] = int(col)
            else:
                for r, c in pix:
                    rr = int(r + dr)
                    cc3 = int(c + dc)
                    if int(ir0) <= rr <= int(ir1) and int(ic0) <= cc3 <= int(ic1):
                        out[rr, cc3] = int(col)

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Variant rule:
    - match target rows/cols from mirrored frame-edge colors
    - erase moved sprites from the left strip
    - copy each mover to every row×col target, but paint the translated bbox as a solid rectangle
    - keep original bottom-to-top overwrite order
    """
    return _solve(grid, fill_bbox=True)


def _pick_positions_with_clearance(candidates: list[int], n: int, span: int, occupied: list[tuple[int, int]]) -> list[int] | None:
    """Pick n centers so each new center stays clear of previous centers by bbox span."""
    if not candidates:
        return None
    order = candidates[:]
    random.shuffle(order)
    picked: list[int] = []
    local_occupied = occupied[:]
    for v in order:
        if all(abs(int(v) - int(p)) >= max(int(span), int(prev_span)) for p, prev_span in local_occupied):
            picked.append(int(v))
            local_occupied.append((int(v), int(span)))
            if len(picked) == int(n):
                return sorted(picked)
    return sorted(picked) if picked else None


def _make_sprite(color: int, max_h: int, max_w: int, *, require_sparse_bbox: bool) -> np.ndarray | None:
    """Generate a cropped 8-connected single-color sprite; movers must be visibly non-rectangular."""
    if max_h < 2 or max_w < 2:
        return None
    for _ in range(80):
        sh = min(random.choice([3, 3, 4, 4, 5]), max_h)
        sw = min(random.choice([3, 3, 4, 4, 5]), max_w)
        if sh < 2 or sw < 2:
            continue
        min_pixels = 4
        max_pixels = min(sh * sw - (2 if require_sparse_bbox and sh * sw >= 6 else 1), 9)
        if max_pixels < min_pixels:
            continue
        n_pixels = random.randint(min_pixels, max_pixels)
        pixels = {(sh // 2, sw // 2)}
        r, c = sh // 2, sw // 2
        for _ in range(n_pixels * 40):
            if len(pixels) >= n_pixels:
                break
            dr, dc = random.choice(
                [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]
            )
            nr, nc = r + dr, c + dc
            if 0 <= nr < sh and 0 <= nc < sw:
                pixels.add((nr, nc))
                r, c = nr, nc
        if len(pixels) < min_pixels:
            continue

        rows = [p[0] for p in pixels]
        cols = [p[1] for p in pixels]
        min_r, max_r = min(rows), max(rows)
        min_c, max_c = min(cols), max(cols)
        sprite_h = max_r - min_r + 1
        sprite_w = max_c - min_c + 1
        if sprite_h < 2 or sprite_w < 2:
            continue
        sprite = np.zeros((sprite_h, sprite_w), dtype=int)
        for pr, pc in pixels:
            sprite[pr - min_r, pc - min_c] = color

        bbox_area = int(sprite_h * sprite_w)
        filled = int(np.count_nonzero(sprite))
        missing = int(bbox_area - filled)
        if require_sparse_bbox and missing < (2 if bbox_area >= 6 else 1):
            continue
        return sprite
    return None


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
    mover_colors = all_colors[idx:idx + n_movers]
    idx += n_movers
    distractor_colors = all_colors[idx:idx + n_distractors]

    sprites: list[np.ndarray] = []
    for mc in mover_colors:
        s = _make_sprite(mc, 5, sprite_area_w, require_sparse_bbox=True)
        if s is None:
            return None
        sprites.append(s)

    mover_targets: dict[int, tuple[list[int], list[int]]] = {}
    occupied_rows: list[tuple[int, int]] = []
    occupied_cols: list[tuple[int, int]] = []
    for mc, sp in zip(mover_colors, sprites):
        sh, sw = sp.shape
        min_row = ir0 + sh // 2
        max_row = ir1 - (sh - 1 - sh // 2)
        min_col = ic0 + sw // 2
        max_col = ic1 - (sw - 1 - sw // 2)
        if min_row > max_row or min_col > max_col:
            return None
        row_count = random.randint(1, 2)
        col_count = random.randint(1, 2)
        avail_rows = list(range(min_row, max_row + 1))
        avail_cols = list(range(min_col, max_col + 1))
        row_marks = _pick_positions_with_clearance(avail_rows, row_count, sh, occupied_rows)
        if not row_marks or len(row_marks) != row_count:
            return None
        col_marks = _pick_positions_with_clearance(avail_cols, col_count, sw, occupied_cols)
        if not col_marks or len(col_marks) != col_count:
            return None
        mover_targets[mc] = (row_marks, col_marks)
        occupied_rows.extend((int(r), int(sh)) for r in row_marks)
        occupied_cols.extend((int(c), int(sw)) for c in col_marks)

    inp = np.full((h, w), bg, dtype=int)
    inp[:, div_col] = divider_color

    top_edge = np.full(frame_w, frame_base, dtype=int)
    for mc, (_, cmarks) in mover_targets.items():
        for cc in cmarks:
            top_edge[int(cc - fc0)] = int(mc)
    inp[0, fc0:fc0 + frame_w] = top_edge
    inp[h - 1, fc0:fc0 + frame_w] = top_edge

    for r in range(1, h - 1):
        val = frame_base
        for mc, (rmarks, _) in mover_targets.items():
            if int(r) in rmarks:
                val = mc
                break
        inp[r, fc0] = int(val)
        inp[r, fc1] = int(val)

    cur_row = 0
    all_items: list[tuple[int, np.ndarray]] = [(mc, sp) for mc, sp in zip(mover_colors, sprites)]
    for dc_color in distractor_colors:
        ds = _make_sprite(dc_color, min(5, h - cur_row), sprite_area_w, require_sparse_bbox=False)
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
                    inp[int(cur_row + dr), int(c_off + dc)] = int(sp[dr, dc])
        cur_row += sh

    vals, counts = np.unique(inp, return_counts=True)
    if int(vals[np.argmax(counts)]) != bg:
        return None

    try:
        output = main(inp)
        sparse_reference = _solve(inp, fill_bbox=False)
    except Exception:
        return None
    if np.array_equal(inp, output):
        return None
    if np.array_equal(output, sparse_reference):
        return None
    if int(np.count_nonzero(output != sparse_reference)) < 4:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(400):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input: failed after 400 attempts")
