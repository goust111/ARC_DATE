# Combo variant of 247ef758 (index 5)
# combo_spec: D2/1+D4/3
# Name: 空心精灵实心化印刻

from __future__ import annotations

"""
Design analysis for the combo variant
=====================================

source_task:
  reference_task: 247ef758
  invariant_backbone:
    - layout stays "left sprite strip | divider column | right frame"
    - mirrored edge markers on the frame still define target rows and target cols by color
    - each mover is erased from the left strip and replicated at every row_mark x col_mark target
    - overwrite order is still bottom-to-top by original vertical position

combo_change_summary:
  D2/1:
    - mover sprites are no longer arbitrary sparse blobs
    - they are hollow single-color components: a foreground ring encloses one or more bg cells
    - bbox still stays within 5x5, but the visible cue is now "outline / hollow shape"
  D4/3:
    - after centering at a target, we do not redraw only the occupied pixels
    - instead, we paint the whole translated bbox as a solid rectangle of the sprite color
  fused_effect:
    - the left side shows hollow outlines
    - the right-side copies become solid rectangles with the same bbox and color
    - the important inference is not merely "copy by color", but "copy location from markers, then fill the hollow shape"

behavior_trace:
  small_example:
    left_sprite_bbox_4x4:
      - 1111
      - 1001
      - 1001
      - 1111
    original_task_stamp:
      - 1111
      - 1001
      - 1001
      - 1111
    combo_stamp:
      - 1111
      - 1111
      - 1111
      - 1111
    difference:
      - the marker matching and centering are unchanged
      - only the occupancy inside the translated bbox changes: enclosed holes disappear
  pipeline_impact:
    - scene parsing is mostly unchanged because label_mask on the foreground still recovers each sprite component
    - the painting step must switch from foreground-pixel translation to bbox filling
    - the generator must ensure holes are genuinely present, otherwise D2/1 would be invisible

complexity_calibration:
  - Hollow sprites are restricted to small rectangular rings with bbox 3x3 to 5x5.
    This keeps the rule low-entropy and legible instead of introducing many unrelated hollow motifs.
  - Movers prefer hole area >= 2, so the solidification is visible even when a mover has only one target.
  - Frame marker counts remain 1-2 rows x 1-2 cols per mover, matching the source task.
  - Row/col target centers are spaced using bbox height/width, because filled rectangles occupy more area than outlines.
  - Distractors remain optional and color-distinct; they may also be hollow, but they never receive matching markers.

implicit_assumption_audit:
  - largest_enclosed_component_bbox(...) assumes the frame interior is the largest enclosed bg region.
    Hollow-sprite holes introduce extra enclosed bg regions, so the generator must keep each hole tiny relative to
    the frame interior. Using 3x3 to 5x5 rings guarantees this.
  - divider detection assumes a nearly full-height uniform non-bg column left of the frame. Preserved as-is.
  - sprite extraction assumes each color forms 8-connected components in the left strip. Rectangular rings satisfy this.
  - target matching assumes mover colors are the only colors mirrored on both relevant opposite frame edges. Preserved.
  - erase step only clears foreground pixels of movers, not their internal holes. This remains correct because holes
    are already bg in the input.
  - the transformed painting step now assumes "the intended output shape is the mover bbox, not the mover mask".
    The generator reinforces that by always using visibly hollow movers.
  - cover order still depends only on visible vertical position, never on color ids or hidden metadata.

placement_legality:
  - Every mover target center is chosen so the full sprite bbox fits inside the frame interior before clipping.
    This makes the solid rectangle fully visible and prevents the fusion cue from depending on edge truncation.
  - Target centers across different movers reserve clearance based on bbox span, reducing messy overlaps.
  - Left-strip sprites are stacked without overlap and kept within the strip width.
  - Distractors are added only if enough height remains for another hollow sprite.

visual_observability:
  - A human sees hollow outlines on the left and filled rectangles on the right in the same colors.
  - Because the fill happens exactly at marker-derived Cartesian-product centers, the two subrules support each other:
    markers explain where copies go; the left hollow shapes explain why the copied rectangles have those sizes.
  - The generator rejects samples where solid-bbox stamping is too similar to sparse stamping, preventing regression to
    the source task or to a visually weak variant.

tidiness_checks:
  - Only 2-3 movers are used, and target centers are spaced by bbox dimensions to keep the interior readable.
  - Hollow shapes are all from one compact family (rectangular rings), so the left strip looks coherent rather than noisy.
  - No invisible factors drive ordering or matching.

main/generator_contract:
  - main() assumes the frame interior exists, movers are detectable as left-strip connected components, and mirrored
    markers exist for mover colors.
  - generate_input() guarantees these assumptions and additionally guarantees:
    1. movers contain enclosed bg holes,
    2. the filled-bbox output differs materially from hollow-shape stamping,
    3. bg remains the most common color so get_bg(...) is stable.
"""

from collections import defaultdict
import random

import numpy as np

from arc_utils import get_bg, label_mask


def largest_enclosed_component_bbox(mask: np.ndarray, *, connectivity: int = 4) -> tuple[int, int, int, int] | None:
    """BBox of the largest connected component in `mask` that does not touch the grid border."""
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


def _extract_scene(
    grid: np.ndarray,
) -> tuple[np.ndarray, int, tuple[int, int, int, int], list[dict], dict[int, list[tuple[int, int]]]]:
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
                    out[int(rr0) : int(rr1) + 1, int(cc0) : int(cc1) + 1] = int(col)
            else:
                for r, c in pix:
                    rr = int(r + dr)
                    cc3 = int(c + dc)
                    if int(ir0) <= rr <= int(ir1) and int(ic0) <= cc3 <= int(ic1):
                        out[rr, cc3] = int(col)

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Combo rule:
    - match target rows/cols from mirrored frame-edge colors
    - erase moved hollow sprites from the left strip
    - center each sprite at every row x col target
    - paint the translated bbox as a solid rectangle, thereby filling any holes
    - keep original bottom-to-top overwrite order
    """
    return _solve(grid, fill_bbox=True)


def _pick_positions_with_clearance(
    candidates: list[int], n: int, span: int, occupied: list[tuple[int, int]]
) -> list[int] | None:
    """Pick target centers that stay clear of previously reserved spans."""
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


def _count_enclosed_hole_cells(mask: np.ndarray) -> int:
    """Count bg cells inside the cropped sprite bbox that belong to enclosed holes."""
    bg_inside = ~mask.astype(bool)
    lab, n = label_mask(bg_inside, connectivity=4)
    total = 0
    h, w = mask.shape
    for k in range(1, int(n) + 1):
        coords = np.argwhere(lab == int(k))
        if coords.size == 0:
            continue
        rs = coords[:, 0]
        cs = coords[:, 1]
        touches_border = bool(
            (rs == 0).any() or (rs == int(h - 1)).any() or (cs == 0).any() or (cs == int(w - 1)).any()
        )
        if not touches_border:
            total += int(coords.shape[0])
    return int(total)


def _make_rect_ring(color: int, h: int, w: int) -> np.ndarray:
    sprite = np.full((int(h), int(w)), int(color), dtype=int)
    sprite[1:-1, 1:-1] = 0
    return sprite


def _make_hollow_sprite(color: int, max_h: int, max_w: int, *, min_hole_area: int) -> np.ndarray | None:
    """
    Generate a cropped hollow sprite as a rectangular ring.

    Restricting the family to rectangular rings keeps the fusion visually legible:
    the left object is unmistakably hollow, and the filled result is unmistakably the bbox.
    """
    candidates: list[tuple[int, int]] = []
    upper_h = min(5, int(max_h))
    upper_w = min(5, int(max_w))
    for h in range(3, upper_h + 1):
        for w in range(3, upper_w + 1):
            hole_area = int((h - 2) * (w - 2))
            if hole_area >= int(min_hole_area):
                candidates.append((int(h), int(w)))
    if not candidates:
        return None

    # Favor medium sizes: big enough to show the hole, small enough to preserve bg-majority.
    candidates.sort(key=lambda hw: (abs(hw[0] * hw[1] - 12), -((hw[0] - 2) * (hw[1] - 2))))
    top = candidates[: min(6, len(candidates))]
    h, w = random.choice(top)
    return _make_rect_ring(int(color), int(h), int(w))


def _gen_attempt() -> np.ndarray | None:
    bg = 0

    interior_h = random.randint(9, 13)
    interior_w = random.randint(9, 13)
    h = interior_h + 2
    sprite_area_w = random.choice([3, 4, 4, 5])
    frame_w = interior_w + 2
    w = sprite_area_w + 1 + frame_w
    if not (10 <= h <= 15 and 14 <= w <= 20):
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
    remaining_colors = all_colors[idx:]

    sprites: list[np.ndarray] = []
    for mc in mover_colors:
        sp = _make_hollow_sprite(mc, 5, sprite_area_w, min_hole_area=2)
        if sp is None or _count_enclosed_hole_cells(sp == int(mc)) < 2:
            return None
        sprites.append(sp)

    if sum(int(sp.shape[0]) for sp in sprites) > int(h):
        return None

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
        row_marks = _pick_positions_with_clearance(list(range(min_row, max_row + 1)), row_count, sh, occupied_rows)
        col_marks = _pick_positions_with_clearance(list(range(min_col, max_col + 1)), col_count, sw, occupied_cols)
        if not row_marks or len(row_marks) != row_count:
            return None
        if not col_marks or len(col_marks) != col_count:
            return None
        mover_targets[int(mc)] = (row_marks, col_marks)
        occupied_rows.extend((int(r), int(sh)) for r in row_marks)
        occupied_cols.extend((int(c), int(sw)) for c in col_marks)

    distractor_sprites: list[tuple[int, np.ndarray]] = []
    remaining_height = int(h - sum(int(sp.shape[0]) for sp in sprites))
    if remaining_height >= 3 and remaining_colors and random.choice([0, 0, 1]) == 1:
        dc_color = int(random.choice(remaining_colors))
        ds = _make_hollow_sprite(dc_color, min(5, remaining_height), sprite_area_w, min_hole_area=1)
        if ds is not None:
            distractor_sprites.append((dc_color, ds))

    inp = np.full((h, w), bg, dtype=int)
    inp[:, div_col] = int(divider_color)

    top_edge = np.full(frame_w, int(frame_base), dtype=int)
    for mc, (_, cmarks) in mover_targets.items():
        for cc in cmarks:
            top_edge[int(cc - fc0)] = int(mc)
    inp[0, fc0 : fc0 + frame_w] = top_edge
    inp[h - 1, fc0 : fc0 + frame_w] = top_edge

    for r in range(1, h - 1):
        val = int(frame_base)
        for mc, (rmarks, _) in mover_targets.items():
            if int(r) in rmarks:
                val = int(mc)
                break
        inp[r, fc0] = int(val)
        inp[r, fc1] = int(val)

    cur_row = 0
    all_items: list[tuple[int, np.ndarray]] = [(int(mc), sp) for mc, sp in zip(mover_colors, sprites)] + distractor_sprites
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
        g, bg2, _, parsed_sprites, targets = _extract_scene(inp)
        assert bg2 == bg
    except Exception:
        return None

    if np.array_equal(inp, output):
        return None
    if np.array_equal(output, sparse_reference):
        return None

    visible_fill_gain = int(np.count_nonzero(output != sparse_reference))
    if visible_fill_gain < 4:
        return None

    mover_hole_cells = []
    for s in parsed_sprites:
        if int(s["color"]) not in targets:
            continue
        mover_hole_cells.append(_count_enclosed_hole_cells(np.array(s["mask"], dtype=bool)))
    if not mover_hole_cells or min(mover_hole_cells) < 2:
        return None

    # Ensure the filled result really looks like solid bboxes at target positions.
    for s in parsed_sprites:
        col = int(s["color"])
        if col not in targets:
            continue
        r0, c0, r1, c1 = map(int, (s["r0"], s["c0"], s["r1"], s["c1"]))
        cr = (r0 + r1) // 2
        cc = (c0 + c1) // 2
        for tr, tc in targets[col]:
            dr = int(tr - cr)
            dc = int(tc - cc)
            rr0 = int(r0 + dr)
            rr1 = int(r1 + dr)
            cc0 = int(c0 + dc)
            cc1 = int(c1 + dc)
            if not (ir0 <= rr0 <= rr1 <= ir1 and ic0 <= cc0 <= cc1 <= ic1):
                return None
            if not np.all(output[rr0 : rr1 + 1, cc0 : cc1 + 1] == int(col)):
                return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(600):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input: failed after 600 attempts")
