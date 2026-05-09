# Variant of 247ef758
# Dimension: D14 — 框架基色与标记的区分方式 (meta)
# Change: 框架边缘由单一基色构成，标记是边缘上的异色像素，标记色==精灵色... → 标记色与精灵色不同（需查找映射）
#
# Design analysis
# ===============
# source_task_status:
#   requested_reference: 58490d8a
#   solver_code_available_in_workspace: false
#   note:
#     - The prompt requires reading the source-task solver first when available.
#     - This workspace does not contain a `58490d8a.py` solver under the allowed solver paths,
#       so the implementation below uses only the original 247ef758 solver/spec plus direct
#       reasoning about an explicit legend-lookup mechanism.
#
# variant_spec:
#   unchanged_backbone:
#     - Overall layout stays `left strip | divider column | right frame`.
#     - The frame interior is still the large enclosed background rectangle on the right.
#     - Sprites are still erased from the left and re-stamped at Cartesian products of
#       row-mark and column-mark positions.
#     - Stamping order is still bottom-to-top by original sprite position.
#   d14_change:
#     - Edge markers no longer reuse sprite colors.
#     - Each mover sprite keeps its own sprite color `S`.
#     - The frame edges use a separate marker color `M`.
#     - A small explicit legend in the left strip defines `M -> S`.
#   concrete_pixels:
#     - The legend occupies the top two rows of the left strip (excluding the divider).
#     - Row 0 contains marker colors.
#     - Row 1 contains the corresponding sprite colors directly underneath the matching
#       marker color columns.
#     - Example: if row0 has `[8, 0, 6]` and row1 has `[4, 0, 7]`, then `8 -> 4` and `6 -> 7`.
#     - The frame edges then use `8` and `6` as row/column markers, while the moved
#       sprites are still painted using `4` and `7`.
#
# behavior_trace:
#   original_local_step:
#     - Read edge color `c`.
#     - Assume `c` is both the marker identity and the sprite color.
#     - Use `c` directly to find the sprite and stamp it.
#   variant_local_step:
#     - Read edge color `m`.
#     - Look up `sprite_color = legend[m]`.
#     - Use the looked-up sprite color to choose which sprite to stamp.
#   concrete_difference:
#     - If the left sprite is color 4 and the matching edge marker is color 8,
#       the original solver would never move sprite 4 from edge color 8.
#     - The variant solver must decode `8 -> 4` from the legend first.
#
# complexity_calibration:
#   - Keep 2-3 movers as in the original task so the legend has repeated support.
#   - Use a bijection between marker colors and mover sprite colors to avoid ambiguity.
#   - Force marker colors to be distinct from sprite colors, divider color, frame base,
#     and distractor colors, so the D14 change is visually obvious.
#   - Keep the legend extremely small (two aligned rows) so the scene stays readable.
#
# implicit_assumption_audit_against_original_main:
#   - Original code collected sprites from every non-divider cell left of the frame.
#     This relied on "all meaningful colored things on the left are sprites".
#     That is no longer true because the legend also lives on the left.
#     Fix: exclude the top-two-row legend zone from sprite extraction.
#   - Original row/column mark collection assumed edge color == sprite color.
#     That is false now.
#     Fix: collect row/column marks by marker color, then translate through the legend.
#   - Original `move_colors` filter used sprite colors directly against edge colors.
#     That would drop all valid markers in the variant.
#     Fix: filter legend pairs by whether the mapped sprite color actually exists.
#   - Original erase/stamp logic only depends on the chosen sprite component and target
#     points, so it remains valid once the color-resolution step is corrected.
#
# placement_legality:
#   - The legend is restricted to rows 0 and 1 of the left strip, so it never overlaps
#     the divider or the right frame.
#   - Sprites start below the legend zone, so the solver can separate legend and sprites
#     deterministically.
#   - The generator still enforces that every stamped sprite center fits the frame interior.
#   - Row and column marker positions remain unique across movers, matching the original task.
#
# visual_observability:
#   - Across examples, viewers can see that edge marker colors do not exist as left sprites.
#   - The legend gives a direct repeated cue: top legend row = marker colors, bottom legend
#     row = sprite colors.
#   - Because the same marker color appears both in the legend and on the frame edges,
#     and the resulting stamped object uses the other color, the symbolic lookup is visible.
#
# visual_cleanliness:
#   - The legend is confined to one compact 2xk block in the top-left strip.
#   - The frame interior remains uncluttered except for the original stamping behavior.
#   - No extra overlays are introduced inside the frame, preserving the original composition.
#
# color_inference_quality:
#   - Only 2-3 legend pairs appear per grid, keeping the mapping small enough for humans.
#   - Each pair is spatially aligned, so the mapping rule is not hidden in raw color values.
#   - Marker colors always reappear on both opposite frame edges, giving supporting evidence
#     for each legend key.
#
# new_main_assumptions_and_generator_guarantees:
#   - Assumption: the legend occupies rows 0 and 1 in the left strip, with aligned non-bg
#     cells defining `marker -> sprite`.
#     Generator guarantee: every mover gets exactly one aligned legend column pair there.
#   - Assumption: actual movable sprites are below the legend zone.
#     Generator guarantee: sprite stacking begins at row 2.
#   - Assumption: marker colors form a one-to-one mapping to mover colors.
#     Generator guarantee: marker colors are sampled as a distinct set and paired bijectively.

from __future__ import annotations

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


def _extract_legend_mapping(
    g: np.ndarray,
    *,
    bg: int,
    fc0: int,
    divider_mask: np.ndarray,
) -> dict[int, int]:
    """
    Read the compact two-row legend in the left strip.

    Legend format:
    - row 0: marker colors
    - row 1: corresponding sprite colors
    - aligned columns define marker->sprite pairs
    """
    legend: dict[int, int] = {}
    for c in range(0, int(fc0)):
        if bool(divider_mask[0, int(c)]):
            continue
        top = int(g[0, int(c)])
        bottom = int(g[1, int(c)])
        if top == int(bg) and bottom == int(bg):
            continue
        if top == int(bg) or bottom == int(bg):
            continue
        if top in legend and int(legend[top]) != bottom:
            raise AssertionError("inconsistent legend key")
        legend[int(top)] = int(bottom)

    if not legend:
        raise AssertionError("missing legend")
    if len(set(map(int, legend.values()))) != len(legend):
        raise AssertionError("legend must be one-to-one")
    return legend


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move colored sprites from the left strip into a framed area at marker-defined targets.

    Variant rule:
    - The frame edges carry marker colors, not sprite colors.
    - A two-row legend in the top-left strip maps marker color -> sprite color.
    - After decoding the mapping, stamping is the same as the original task.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

    # --- frame interior: largest bg component not touching boundary ---
    inner_bbox = largest_enclosed_component_bbox(g == int(bg), connectivity=4)
    assert inner_bbox is not None
    ir0, ic0, ir1, ic1 = map(int, inner_bbox)
    fr0, fc0, fr1, fc1 = int(ir0 - 1), int(ic0 - 1), int(ir1 + 1), int(ic1 + 1)

    # --- divider columns (fixed) on the left side ---
    divider_mask = np.zeros((H, W), dtype=bool)
    for c in range(0, max(0, int(fc0))):
        colv = g[:, int(c)]
        non = colv[colv != int(bg)]
        if int(len(non)) >= int(H - 1) and len(set(map(int, non))) == 1:
            divider_mask[:, int(c)] = True

    left_cols = (np.arange(W)[None, :] < int(fc0))
    legend_zone = np.zeros((H, W), dtype=bool)
    legend_zone[0:2, 0:int(fc0)] = True

    # --- decode explicit legend before interpreting edge markers ---
    marker_to_sprite = _extract_legend_mapping(g, bg=int(bg), fc0=int(fc0), divider_mask=divider_mask)

    # --- sprites: per-color 8-connected components in left area (excluding divider and legend) ---
    sprites = []
    for col in sorted(set(map(int, g.flatten())) - {int(bg)}):
        mask = (g == int(col)) & left_cols & (~divider_mask) & (~legend_zone)
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
    valid_marker_to_sprite = {
        int(marker): int(sprite)
        for marker, sprite in marker_to_sprite.items()
        if int(sprite) in move_colors
    }

    # --- marker-derived targets: edge colors are legend keys, not sprite colors ---
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
    for col in set(row_marks.keys()) & set(col_marks.keys()):
        pts = {(int(r), int(c)) for r in row_marks[int(col)] for c in col_marks[int(col)]}
        targets[int(col)] = sorted(pts)

    out = g.copy()
    movers = [s for s in sprites if int(s["color"]) in targets]

    # Erase originals of moved sprites.
    for s in movers:
        r0, c0 = int(s["r0"]), int(s["c0"])
        for rr, cc in np.argwhere(s["mask"]):
            out[int(r0) + int(rr), int(c0) + int(cc)] = int(bg)

    # Bottom-to-top ordering: lower sprites first, upper sprites last (covering).
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


# ---------------------------------------------------------------------------
# generate_input
# ---------------------------------------------------------------------------
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

    # Dimensions — frame spans full grid height.
    interior_h = random.randint(8, 13)
    interior_w = random.randint(8, 13)
    H = interior_h + 2
    if not 10 <= H <= 15:
        return None

    n_movers = random.randint(2, 3)
    min_sprite_area_w = 2 * int(n_movers) - 1
    sprite_area_w = random.randint(min_sprite_area_w, 5)

    frame_w = interior_w + 2
    W = sprite_area_w + 1 + frame_w
    if not 14 <= W <= 20:
        return None

    # Key positions
    div_col = sprite_area_w
    fc0, fc1 = sprite_area_w + 1, W - 1
    ir0, ic0 = 1, fc0 + 1
    ir1, ic1 = H - 2, fc1 - 1

    # Assign distinct colors: divider, frame base, movers, marker palette, optional distractor.
    all_colors = list(range(1, 10))
    random.shuffle(all_colors)
    n_distractors = 0 if int(n_movers) == 3 else random.choice([0, 0, 1])
    idx = 0
    divider_color = all_colors[idx]
    idx += 1
    frame_base = all_colors[idx]
    idx += 1
    mover_colors = all_colors[idx : idx + n_movers]
    idx += n_movers
    marker_colors = all_colors[idx : idx + n_movers]
    idx += n_movers
    distractor_colors = all_colors[idx : idx + n_distractors]
    idx += n_distractors

    mover_pairs = list(zip(mover_colors, marker_colors))

    # Generate mover sprites.
    sprites: list[np.ndarray] = []
    for mc in mover_colors:
        s = _make_sprite(mc, 5, sprite_area_w)
        if s is None:
            return None
        sprites.append(s)

    # Pick marker positions per mover (no two movers share a row or col marker).
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
        mover_targets[int(mc)] = (sorted(row_marks), sorted(col_marks))
        occupied_rows.update(row_marks)
        occupied_cols.update(col_marks)

    # Build input grid: bg fill, then divider, legend, frame, sprites.
    inp = np.full((H, W), bg, dtype=int)
    inp[:, div_col] = divider_color

    # Legend: top row = marker colors, bottom row = sprite colors.
    legend_needed_w = 2 * int(n_movers) - 1
    legend_offset = random.randint(0, sprite_area_w - legend_needed_w)
    legend_cols = [legend_offset + 2 * i for i in range(int(n_movers))]
    for legend_col, (sprite_color, marker_color) in zip(legend_cols, mover_pairs):
        inp[0, int(legend_col)] = int(marker_color)
        inp[1, int(legend_col)] = int(sprite_color)

    # Frame top/bottom edges (mirrored) use marker colors.
    top_edge = np.full(frame_w, frame_base, dtype=int)
    for sprite_color, marker_color in mover_pairs:
        _, cmarks = mover_targets[int(sprite_color)]
        for cc in cmarks:
            top_edge[int(cc - fc0)] = int(marker_color)
    inp[0, fc0 : fc0 + frame_w] = top_edge
    inp[H - 1, fc0 : fc0 + frame_w] = top_edge

    # Frame left/right edges (mirrored) use marker colors.
    for r in range(1, H - 1):
        val = frame_base
        for sprite_color, marker_color in mover_pairs:
            rmarks, _ = mover_targets[int(sprite_color)]
            if r in rmarks:
                val = int(marker_color)
                break
        inp[r, fc0] = int(val)
        inp[r, fc1] = int(val)

    # Stack sprites in left strip below the two legend rows.
    cur_row = 2
    all_items: list[tuple[int, np.ndarray, bool]] = [(mc, sp, False) for mc, sp in zip(mover_colors, sprites)]
    for dc_color in distractor_colors:
        ds = _make_sprite(dc_color, min(5, H - cur_row), sprite_area_w)
        if ds is None:
            return None
        all_items.append((dc_color, ds, True))

    for _color, sp, _is_distractor in all_items:
        sh, sw = sp.shape
        if sw > sprite_area_w or cur_row + sh > H:
            return None
        c_off = random.randint(0, sprite_area_w - sw)
        for dr in range(sh):
            for dc in range(sw):
                if sp[dr, dc] != bg:
                    inp[cur_row + dr, c_off + dc] = sp[dr, dc]
        cur_row += sh

    # Validate: bg must be most common color so solver detects it correctly.
    vals, counts = np.unique(inp, return_counts=True)
    if int(vals[np.argmax(counts)]) != bg:
        return None

    # Run solver and verify output differs from input (movers actually moved).
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
