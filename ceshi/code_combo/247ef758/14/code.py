# Combo variant of 247ef758 (index 14)
# combo_spec: D2/0+D4/0+D14/1
# Name: 多色精灵+图例映射+重着色
#
# Design analysis
# ===============
# fused_rule:
#   - The overall scaffold stays the same as the original task:
#     left sprite strip | divider column | right rectangular frame.
#   - Each left-strip sprite is now a two-color 8-connected component.
#   - One color is a strict-majority "main color"; the other is a minority accent color.
#   - Frame edge markers do not use sprite colors directly. Instead, a compact two-row legend
#     in the top-left strip maps marker_color -> sprite_main_color.
#   - After legend lookup and target matching, the sprite is stamped with a fused recolor rule:
#       source main-color pixels -> frame_base
#       source accent-color pixels -> matched marker_color
#
# concrete_trace:
#   - Suppose a source sprite contains 5 cells of color S and 2 cells of color A.
#   - The legend shows M above S, so M -> S.
#   - The frame edges use M on matching rows/cols, not S.
#   - The solver must:
#       1. read the legend and decode M -> S
#       2. extract the whole two-color component, not two separate monochrome blobs
#       3. infer that S is the dominant color and therefore the matching key
#       4. stamp the same shape at every target, but recolor S-cells to frame_base and A-cells to M
#   - Difference from the original:
#       original used edge color == sprite color and copied the original monochrome color.
#       the fused version needs legend lookup, dominant-color inference, and a structured recolor.
#
# complexity_calibration:
#   - Keep only 2-3 movers, with 0-1 distractor and at most 3 total sprites below the legend.
#   - All source sprites share one accent color. This keeps the input readable while still forcing
#     dominant-color reasoning because only the majority color participates in legend matching.
#   - Marker colors are disjoint from mover main colors, divider, frame base, and source accent.
#   - The output recolor uses exactly two visible targets per mover family: frame_base and that
#     mover's marker color. This is strong enough to see, but not so unconstrained that examples
#     become noisy or underdetermined.
#
# hidden_assumption_audit:
#   - Original per-color sprite extraction breaks immediately for multicolor sprites.
#     Fix: label all non-bg left-strip cells at once, excluding divider and legend rows.
#   - Original edge color == sprite color assumption breaks because of the legend mapping.
#     Fix: collect row/col marks by marker color, then translate marker -> sprite main color.
#   - Original stamp color == sprite color assumption breaks because of recoloring.
#     Fix: keep the extracted full pattern, then rewrite pixel colors during stamping.
#   - Original generator could stack colored objects flush vertically. That would merge adjacent
#     multicolor sprites into one 8-connected component.
#     Fix: reserve at least one blank row between sprites, plus a blank row under the legend.
#
# placement_legality:
#   - Legend occupies rows 0 and 1 in the left strip; row 2 stays blank.
#   - Sprite stacking starts at row >= 3 and keeps one blank row between consecutive sprites.
#   - Target rows/cols are chosen only from positions where the full sprite bbox fits in the frame
#     interior. The solver still clips defensively, but the generator aims for full fits.
#   - Target bboxes are kept non-touching so the recolored output remains visually separable.
#
# visual_observability:
#   - Edge marker colors do not appear as source main colors, so the legend is necessary.
#   - The source accent color is shared, but output accent cells become each mover's marker color,
#     making the recolor rule visibly dependent on the matched legend key rather than on raw color ids.
#   - The output never reproduces source main colors inside the frame; it uses frame_base instead.
#     This gives repeated evidence for recoloring across multiple movers.
#
# cleanliness:
#   - The legend is compact and isolated from the sprite stack.
#   - The frame interior contains separated stamped copies rather than merged masses.
#   - The palette budget is controlled so the scene stays parseable in 2-4 training examples.
#
# generator_guarantees_for_new_solver_assumptions:
#   - Every extracted left-strip component uses exactly two colors and has a unique strict majority.
#   - Marker colors form a bijection to mover main colors through the legend.
#   - No legend color leaks into source sprites except through the later output recolor.
#   - At least one recolored marker-color pixel and at least one recolored frame-base pixel appear
#     inside the frame, so accepted samples really exercise the fused rule.

from __future__ import annotations

from collections import Counter, defaultdict
import random

import numpy as np

from arc_utils import get_bg, label_mask


def largest_enclosed_component_bbox(mask: np.ndarray, *, connectivity: int = 4) -> tuple[int, int, int, int] | None:
    """
    Local helper: bbox of the largest connected component in `mask`
    that does not touch the outer grid border.
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


def _dominant_color(vals: list[int]) -> int:
    counts = Counter(map(int, vals))
    return sorted(counts.items(), key=lambda kv: (-int(kv[1]), int(kv[0])))[0][0]


def _frame_base_color(g: np.ndarray, bg: int, fr0: int, fc0: int, fr1: int, fc1: int) -> int:
    border_vals: list[int] = []
    border_vals.extend(map(int, g[int(fr0), int(fc0) : int(fc1) + 1]))
    border_vals.extend(map(int, g[int(fr1), int(fc0) : int(fc1) + 1]))
    border_vals.extend(int(g[r, int(fc0)]) for r in range(int(fr0) + 1, int(fr1)))
    border_vals.extend(int(g[r, int(fc1)]) for r in range(int(fr0) + 1, int(fr1)))
    counts = Counter(v for v in border_vals if int(v) != int(bg))
    assert counts
    return int(counts.most_common(1)[0][0])


def _extract_legend_mapping(
    g: np.ndarray,
    *,
    bg: int,
    fc0: int,
    divider_mask: np.ndarray,
) -> dict[int, int]:
    """
    Legend format:
    - row 0: marker colors
    - row 1: corresponding sprite main colors
    - aligned non-bg columns define marker -> sprite_main mappings
    """
    legend: dict[int, int] = {}
    for c in range(int(fc0)):
        if bool(divider_mask[0, int(c)]):
            continue
        top = int(g[0, int(c)])
        bottom = int(g[1, int(c)])
        if top == int(bg) and bottom == int(bg):
            continue
        if top == int(bg) or bottom == int(bg):
            continue
        if top in legend and int(legend[top]) != int(bottom):
            raise AssertionError("inconsistent legend")
        legend[int(top)] = int(bottom)

    if not legend:
        raise AssertionError("missing legend")
    if len(set(map(int, legend.values()))) != len(legend):
        raise AssertionError("legend must be one-to-one")
    return legend


def _extract_sprites(
    g: np.ndarray,
    *,
    bg: int,
    fc0: int,
    divider_mask: np.ndarray,
) -> list[dict]:
    left_mask = np.arange(g.shape[1])[None, :] < int(fc0)
    legend_zone = np.zeros_like(g, dtype=bool)
    legend_zone[0:2, 0:int(fc0)] = True
    sprite_mask = (g != int(bg)) & left_mask & (~divider_mask) & (~legend_zone)
    slab, sn = label_mask(sprite_mask, connectivity=8)

    sprites: list[dict] = []
    for sid in range(1, int(sn) + 1):
        pts = np.argwhere(slab == int(sid))
        if pts.size == 0:
            continue
        r0, c0 = pts.min(axis=0)
        r1, c1 = pts.max(axis=0)
        r0, c0, r1, c1 = map(int, (r0, c0, r1, c1))
        comp = slab[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1] == int(sid)
        patch = np.where(comp, g[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1], int(bg))
        colors = [int(v) for v in patch[comp]]
        if not colors:
            continue
        sprites.append(
            {
                "main_color": int(_dominant_color(colors)),
                "r0": int(r0),
                "c0": int(c0),
                "r1": int(r1),
                "c1": int(c1),
                "pattern": patch.astype(int),
            }
        )
    return sprites


def _recolor_pattern(
    pattern: np.ndarray,
    *,
    bg: int,
    main_color: int,
    marker_color: int,
    frame_base: int,
) -> np.ndarray:
    out = np.array(pattern, dtype=int, copy=True)
    out[out == int(main_color)] = int(frame_base)
    out[(out != int(bg)) & (out != int(frame_base))] = int(marker_color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fused rule:
    - Decode marker_color -> sprite_main_color from the compact legend.
    - Extract each left-strip multicolor sprite as a whole component.
    - Match targets using the sprite's dominant color after legend lookup.
    - Stamp the sprite shape with recolor:
        main-color cells -> frame_base
        accent-color cells -> matching marker_color
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

    inner_bbox = largest_enclosed_component_bbox(g == int(bg), connectivity=4)
    assert inner_bbox is not None
    ir0, ic0, ir1, ic1 = map(int, inner_bbox)
    fr0, fc0, fr1, fc1 = int(ir0 - 1), int(ic0 - 1), int(ir1 + 1), int(ic1 + 1)
    frame_base = _frame_base_color(g, bg, fr0, fc0, fr1, fc1)

    divider_mask = np.zeros((H, W), dtype=bool)
    for c in range(0, max(0, int(fc0))):
        colv = g[:, int(c)]
        non = colv[colv != int(bg)]
        if int(len(non)) >= int(H - 1) and len(set(map(int, non))) == 1:
            divider_mask[:, int(c)] = True

    marker_to_sprite = _extract_legend_mapping(g, bg=int(bg), fc0=int(fc0), divider_mask=divider_mask)
    sprites = _extract_sprites(g, bg=int(bg), fc0=int(fc0), divider_mask=divider_mask)

    move_colors = {int(s["main_color"]) for s in sprites}
    valid_marker_to_sprite = {
        int(marker): int(sprite_color)
        for marker, sprite_color in marker_to_sprite.items()
        if int(sprite_color) in move_colors
    }

    row_marks: dict[int, list[int]] = defaultdict(list)
    col_marks: dict[int, list[int]] = defaultdict(list)
    for r in range(int(ir0), int(ir1) + 1):
        cl = int(g[int(r), int(fc0)])
        cr = int(g[int(r), int(fc1)])
        if cl == cr and cl in valid_marker_to_sprite:
            row_marks[int(valid_marker_to_sprite[cl])].append(int(r))
    for c in range(int(ic0), int(ic1) + 1):
        ct = int(g[int(fr0), int(c)])
        cb = int(g[int(fr1), int(c)])
        if ct == cb and ct in valid_marker_to_sprite:
            col_marks[int(valid_marker_to_sprite[ct])].append(int(c))

    targets: dict[int, list[tuple[int, int]]] = {}
    for sprite_color in set(row_marks.keys()) & set(col_marks.keys()):
        pts = {(int(r), int(c)) for r in row_marks[int(sprite_color)] for c in col_marks[int(sprite_color)]}
        targets[int(sprite_color)] = sorted(pts)

    marker_for_sprite = {
        int(sprite_color): int(marker)
        for marker, sprite_color in valid_marker_to_sprite.items()
        if int(sprite_color) in targets
    }

    out = g.copy()
    movers = [s for s in sprites if int(s["main_color"]) in targets]

    for s in movers:
        r0, c0 = int(s["r0"]), int(s["c0"])
        pat = np.array(s["pattern"], dtype=int)
        for rr, cc in np.argwhere(pat != int(bg)):
            out[int(r0) + int(rr), int(c0) + int(cc)] = int(bg)

    movers.sort(key=lambda s: (-((int(s["r0"]) + int(s["r1"])) / 2.0), int(s["c0"])))

    for s in movers:
        key = int(s["main_color"])
        marker_color = int(marker_for_sprite[int(key)])
        cr = (int(s["r0"]) + int(s["r1"])) // 2
        cc = (int(s["c0"]) + int(s["c1"])) // 2
        recolored = _recolor_pattern(
            np.array(s["pattern"], dtype=int),
            bg=int(bg),
            main_color=int(key),
            marker_color=int(marker_color),
            frame_base=int(frame_base),
        )
        pix = [(int(rr), int(cc2), int(recolored[int(rr), int(cc2)])) for rr, cc2 in np.argwhere(recolored != int(bg))]

        for tr, tc in targets[int(key)]:
            dr = int(tr - cr)
            dc = int(tc - cc)
            for rr0, cc0, color in pix:
                rr = int(s["r0"]) + int(rr0) + int(dr)
                cc3 = int(s["c0"]) + int(cc0) + int(dc)
                if int(ir0) <= rr <= int(ir1) and int(ic0) <= cc3 <= int(ic1):
                    out[rr, cc3] = int(color)

    return out


def _pick_spaced(candidates: list[int], n: int, min_spacing: int) -> list[int] | None:
    if not candidates:
        return None
    if n == 1:
        return [int(random.choice(candidates))]
    pool = list(candidates)
    random.shuffle(pool)
    picked: list[int] = []
    for v in pool:
        if all(abs(int(v) - int(p)) >= int(min_spacing) for p in picked):
            picked.append(int(v))
            if len(picked) == n:
                return picked
    return picked if picked else None


def _make_shape_mask(max_h: int, max_w: int) -> np.ndarray | None:
    sh = min(random.choice([2, 3, 3, 4]), int(max_h))
    sw = min(random.choice([2, 3, 3, 4, 5]), int(max_w))
    if sh < 1 or sw < 1:
        return None

    max_pixels = min(9, sh * sw)
    if max_pixels < 3:
        return None
    n_pixels = random.randint(3, max_pixels)

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

    rows = [p[0] for p in pixels]
    cols = [p[1] for p in pixels]
    min_r, max_r = min(rows), max(rows)
    min_c, max_c = min(cols), max(cols)
    mask = np.zeros((max_r - min_r + 1, max_c - min_c + 1), dtype=bool)
    for pr, pc in pixels:
        mask[pr - min_r, pc - min_c] = True
    return mask


def _sample_connected_subset(coords: list[tuple[int, int]], k: int) -> set[tuple[int, int]]:
    start = random.choice(coords)
    chosen = {start}
    frontier = [start]
    while len(chosen) < int(k):
        base = random.choice(frontier)
        nbrs = [
            (nr, nc)
            for nr in range(base[0] - 1, base[0] + 2)
            for nc in range(base[1] - 1, base[1] + 2)
            if (nr, nc) in coords and (nr, nc) not in chosen and not (nr == base[0] and nc == base[1])
        ]
        if nbrs:
            nxt = random.choice(nbrs)
            chosen.add(nxt)
            frontier.append(nxt)
            continue
        frontier.append(random.choice(list(chosen)))
        if len(frontier) > len(coords) * 4:
            break
    if len(chosen) < int(k):
        remaining = [p for p in coords if p not in chosen]
        random.shuffle(remaining)
        for p in remaining:
            if any(max(abs(p[0] - q[0]), abs(p[1] - q[1])) <= 1 for q in chosen):
                chosen.add(p)
                if len(chosen) >= int(k):
                    break
    return chosen


def _make_multicolor_sprite(primary: int, accent: int, max_h: int, max_w: int) -> np.ndarray | None:
    mask = _make_shape_mask(max_h, max_w)
    if mask is None:
        return None
    coords = [tuple(map(int, p)) for p in np.argwhere(mask)]
    area = len(coords)
    if area < 3:
        return None

    max_accent = min(3, (area - 1) // 2)
    if max_accent < 1:
        return None
    min_accent = 2 if area >= 5 and max_accent >= 2 else 1
    n_accent = random.randint(min_accent, max_accent)
    accent_cells = _sample_connected_subset(coords, n_accent)

    sprite = np.zeros(mask.shape, dtype=int)
    for r, c in coords:
        sprite[int(r), int(c)] = int(accent if (r, c) in accent_cells else primary)
    return sprite


def _stamp_bbox(sp: np.ndarray, tr: int, tc: int) -> tuple[int, int, int, int]:
    sh, sw = sp.shape
    r0 = int(tr - sh // 2)
    c0 = int(tc - sw // 2)
    return int(r0), int(c0), int(r0 + sh - 1), int(c0 + sw - 1)


def _bboxes_too_close(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    ar0, ac0, ar1, ac1 = map(int, a)
    br0, bc0, br1, bc1 = map(int, b)
    return not (ar1 + 1 < br0 or br1 + 1 < ar0 or ac1 + 1 < bc0 or bc1 + 1 < ac0)


def _pick_targets(
    sp: np.ndarray,
    ir0: int,
    ic0: int,
    ir1: int,
    ic1: int,
    occupied_rows: set[int],
    occupied_cols: set[int],
    existing_bboxes: list[tuple[int, int, int, int]],
) -> tuple[list[int], list[int], list[tuple[int, int, int, int]]] | None:
    sh, sw = sp.shape
    min_row = int(ir0 + sh // 2)
    max_row = int(ir1 - (sh - 1 - sh // 2))
    min_col = int(ic0 + sw // 2)
    max_col = int(ic1 - (sw - 1 - sw // 2))
    if min_row > max_row or min_col > max_col:
        return None

    avail_rows = [r for r in range(min_row, max_row + 1) if r not in occupied_rows]
    avail_cols = [c for c in range(min_col, max_col + 1) if c not in occupied_cols]
    if not avail_rows or not avail_cols:
        return None

    for _ in range(100):
        want_rows = random.randint(1, 2)
        want_cols = random.randint(1, 2)
        row_marks = _pick_spaced(avail_rows[:], want_rows, max(1, sh))
        col_marks = _pick_spaced(avail_cols[:], want_cols, max(1, sw))
        if not row_marks or not col_marks:
            continue

        new_boxes = [_stamp_bbox(sp, r, c) for r in row_marks for c in col_marks]
        ok = True
        for i in range(len(new_boxes)):
            for j in range(i + 1, len(new_boxes)):
                if _bboxes_too_close(new_boxes[i], new_boxes[j]):
                    ok = False
                    break
            if not ok:
                break
        if not ok:
            continue
        for box in new_boxes:
            if any(_bboxes_too_close(box, prev) for prev in existing_bboxes):
                ok = False
                break
        if ok:
            return sorted(row_marks), sorted(col_marks), new_boxes
    return None


def _sample_gaps(total_extra: int, n_slots: int) -> list[int]:
    gaps = [0] * int(n_slots)
    for _ in range(int(total_extra)):
        gaps[random.randrange(int(n_slots))] += 1
    return gaps


def _fusion_metric(
    inp: np.ndarray,
    out: np.ndarray,
    *,
    frame_base: int,
    marker_colors: list[int],
    source_accent: int,
    mover_colors: list[int],
) -> tuple[int, int, int, int]:
    g = np.array(inp, dtype=int)
    bg = int(get_bg(g))
    inner_bbox = largest_enclosed_component_bbox(g == int(bg), connectivity=4)
    assert inner_bbox is not None
    ir0, ic0, ir1, ic1 = map(int, inner_bbox)

    interior_mask = np.zeros_like(g, dtype=bool)
    interior_mask[int(ir0) : int(ir1) + 1, int(ic0) : int(ic1) + 1] = True
    changed = interior_mask & (g != out)

    marker_set = set(map(int, marker_colors))
    forbidden = set(map(int, mover_colors)) | {int(source_accent)}
    marker_pixels = int(np.count_nonzero(changed & np.isin(out, list(marker_set))))
    frame_pixels = int(np.count_nonzero(changed & (out == int(frame_base))))
    forbidden_pixels = int(np.count_nonzero(changed & np.isin(out, list(forbidden))))
    distinct_marker_colors = len(set(map(int, out[changed & np.isin(out, list(marker_set))].flatten())))
    return marker_pixels, frame_pixels, forbidden_pixels, distinct_marker_colors


def _gen_attempt() -> np.ndarray | None:
    bg = 0

    interior_h = random.randint(8, 13)
    interior_w = random.randint(8, 13)
    H = interior_h + 2
    if not 10 <= H <= 15:
        return None

    n_movers = random.randint(2, 3)
    n_distractors = 0 if int(n_movers) == 3 else random.choice([0, 0, 1])
    n_total = int(n_movers + n_distractors)

    min_sprite_area_w = max(3, 2 * int(n_movers) - 1)
    sprite_area_w = random.randint(min_sprite_area_w, 5)
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
    idx = 0
    divider_color = int(all_colors[idx])
    idx += 1
    frame_base = int(all_colors[idx])
    idx += 1
    source_accent = int(all_colors[idx])
    idx += 1
    mover_colors = [int(v) for v in all_colors[idx : idx + n_movers]]
    idx += n_movers
    marker_colors = [int(v) for v in all_colors[idx : idx + n_movers]]
    idx += n_movers
    distractor_colors = [int(v) for v in all_colors[idx : idx + n_distractors]]

    sprite_max_h = 4 if int(n_total) <= 2 else 3

    mover_sprites: list[np.ndarray] = []
    for mc in mover_colors:
        sp = _make_multicolor_sprite(int(mc), int(source_accent), sprite_max_h, sprite_area_w)
        if sp is None:
            return None
        mover_sprites.append(sp)

    mover_targets: dict[int, tuple[list[int], list[int]]] = {}
    occupied_rows: set[int] = set()
    occupied_cols: set[int] = set()
    occupied_target_boxes: list[tuple[int, int, int, int]] = []
    for mc, sp in zip(mover_colors, mover_sprites):
        picked = _pick_targets(sp, ir0, ic0, ir1, ic1, occupied_rows, occupied_cols, occupied_target_boxes)
        if picked is None:
            return None
        row_marks, col_marks, boxes = picked
        mover_targets[int(mc)] = (row_marks, col_marks)
        occupied_rows.update(map(int, row_marks))
        occupied_cols.update(map(int, col_marks))
        occupied_target_boxes.extend(boxes)

    all_items: list[tuple[int, np.ndarray, bool]] = [(int(mc), sp, False) for mc, sp in zip(mover_colors, mover_sprites)]
    for dc in distractor_colors:
        ds = _make_multicolor_sprite(int(dc), int(source_accent), sprite_max_h, sprite_area_w)
        if ds is None:
            return None
        all_items.append((int(dc), ds, True))

    start_row = 3
    min_needed_h = int(start_row + sum(int(sp.shape[0]) for _, sp, _ in all_items) + max(0, len(all_items) - 1))
    if min_needed_h > H:
        return None

    inp = np.full((H, W), bg, dtype=int)
    inp[:, int(div_col)] = int(divider_color)

    legend_needed_w = 2 * int(n_movers) - 1
    legend_offset = random.randint(0, int(sprite_area_w - legend_needed_w))
    legend_cols = [int(legend_offset + 2 * i) for i in range(int(n_movers))]
    for legend_col, mc, mk in zip(legend_cols, mover_colors, marker_colors):
        inp[0, int(legend_col)] = int(mk)
        inp[1, int(legend_col)] = int(mc)

    top_edge = np.full(frame_w, int(frame_base), dtype=int)
    for mc, mk in zip(mover_colors, marker_colors):
        _rmarks, cmarks = mover_targets[int(mc)]
        for cc in cmarks:
            top_edge[int(cc - fc0)] = int(mk)
    inp[0, int(fc0) : int(fc0 + frame_w)] = top_edge
    inp[H - 1, int(fc0) : int(fc0 + frame_w)] = top_edge

    for r in range(1, H - 1):
        val = int(frame_base)
        for mc, mk in zip(mover_colors, marker_colors):
            rmarks, _cmarks = mover_targets[int(mc)]
            if int(r) in rmarks:
                val = int(mk)
                break
        inp[int(r), int(fc0)] = int(val)
        inp[int(r), int(fc1)] = int(val)

    extra_rows = int(H - min_needed_h)
    gap_extras = _sample_gaps(extra_rows, len(all_items) + 1)
    cur_row = int(start_row + gap_extras[0])
    for idx_item, (_primary, sp, _is_distractor) in enumerate(all_items):
        sh, sw = sp.shape
        if sw > sprite_area_w or cur_row + sh > H:
            return None
        c_off = random.randint(0, int(sprite_area_w - sw))
        inp[cur_row : cur_row + sh, c_off : c_off + sw] = np.where(
            sp != int(bg),
            sp,
            inp[cur_row : cur_row + sh, c_off : c_off + sw],
        )
        cur_row += int(sh)
        if idx_item < len(all_items) - 1:
            cur_row += int(1 + gap_extras[idx_item + 1])

    vals, counts = np.unique(inp, return_counts=True)
    if int(vals[np.argmax(counts)]) != int(bg):
        return None

    try:
        output = main(inp)
    except Exception:
        return None

    if np.array_equal(inp, output):
        return None

    marker_pixels, frame_pixels, forbidden_pixels, distinct_marker_colors = _fusion_metric(
        inp,
        output,
        frame_base=int(frame_base),
        marker_colors=list(marker_colors),
        source_accent=int(source_accent),
        mover_colors=list(mover_colors),
    )
    if marker_pixels <= 0 or frame_pixels <= 0:
        return None
    if forbidden_pixels != 0:
        return None
    if distinct_marker_colors <= 0:
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(500):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input: failed after 500 attempts")
