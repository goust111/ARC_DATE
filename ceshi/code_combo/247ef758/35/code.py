# Combo variant of 247ef758 (index 35)
# combo_spec: D4/2+D9/2
# Name: 保留源精灵+删除干扰

from __future__ import annotations

# Design analysis for the combined D4/2 + D9/2 variant
# -----------------------------------------------------
# a) Concrete behavior trace
#    Original task:
#    - left strip contains 2-3 mover sprites plus optional 0-1 distractor sprites
#    - a mover color appears as matching markers on both left/right frame edges and
#      top/bottom frame edges, which defines a Cartesian-product set of target cells
#    - solver erases movers from the left, stamps them into the frame, and leaves
#      distractors unchanged on the left
#    Combined variant:
#    - if a sprite color has row+col targets, it is a mover: keep the source pixels
#      on the left and also stamp copies into the frame interior
#    - if a sprite color has no complete target set, it is a distractor: erase it
#      completely from the left and stamp nothing
#    Small trace:
#    - red sprite has mirrored row+col markers; blue sprite has none
#    - original output: red disappears from the left and appears in the frame; blue
#      remains on the left
#    - combined output: red remains on the left and also appears in the frame; blue
#      is deleted to background
#    The visible contrast is therefore "matched => preserve+copy" versus
#    "unmatched => delete".
#
# b) Complexity calibration
#    We keep the original structural complexity:
#    - 2-3 movers
#    - 1-2 row markers and 1-2 col markers per mover
#    - 0-1 distractor
#    - unique colors for divider, frame base, movers, distractor
#    This is enough repeated evidence for a human to infer the rule while avoiding
#    ambiguity from many unmatched colors. The generator biases toward including one
#    distractor so the delete branch is visible in most samples.
#
# c) Implicit-assumption audit of the original main()
#    - largest enclosed background component = frame interior: still valid because
#      the generator preserves the full-height frame layout
#    - near-solid uniform left-side divider column exists: unchanged
#    - every non-divider connected component in the left strip is a sprite:
#      unchanged; generator places only movers/distractors there
#    - mover colors are exactly the colors with both mirrored row marks and mirrored
#      column marks: unchanged and now becomes the classifier between keep+copy and
#      delete
#    - original "erase originals of moved sprites" loop is invalid under D4/2 and
#      must be removed
#    - original "leave non-movers untouched" behavior is invalid under D9/2 and must
#      be replaced with an erase-nonmovers loop
#    - bottom-to-top ordering still applies only to stamped movers inside the frame;
#      source preservation does not interfere because source and frame are disjoint
#    - the loop over colors assumes sprite colors are unique and distinct from divider
#      and frame colors; generator enforces this to avoid accidental merging
#
# d) Placement legality
#    Generator guarantees:
#    - all sprites stay inside cols 0..divider-1 and are stacked without overlap
#    - mover target centers are chosen so each centered stamp fits inside the frame
#      interior without needing clipping on generated examples
#    - no two movers share a row mark or column mark, so target ownership is clear
#    - distractor colors never appear on frame markers, so delete-vs-copy is
#      unambiguous
#    - if a distractor is requested, there is enough remaining left-strip height to
#      place it
#
# e) Visual observability
#    Across 2-4 examples a human can detect two opposite behaviors:
#    - matched shape/color repeats in two places (left source + framed copies)
#    - unmatched shape/color disappears entirely
#    Because the classifier is visible on the frame edges, the rule is inferable from
#    pixel structure alone; no hidden ordering by numeric color values is used.
#
# f) Visual cleanliness
#    The layout stays clean because the left strip, divider, and frame remain
#    separated. Only movers stamp into the frame interior, and there is at most one
#    distractor to delete, so the left side does not become cluttered with many
#    special cases.
#
# g) Change inferability and new post-change assumptions
#    No extra color mapping is introduced: color identity still links a left sprite
#    to matching frame markers. New assumptions introduced by the combined solver:
#    - a sprite is preserved iff its color has both row and column marker sets
#    - a sprite is deleted iff its color lacks a complete target set
#    - the input contains no unrelated left-strip objects besides sprites
#    The generator explicitly constructs samples to satisfy all three assumptions and
#    then validates both branches:
#    - mover source pixels survive unchanged
#    - distractor pixels are erased to background and their color disappears

from collections import defaultdict
import random

import numpy as np

from arc_utils import get_bg, label_mask


def largest_enclosed_component_bbox(
    mask: np.ndarray, *, connectivity: int = 4
) -> tuple[int, int, int, int] | None:
    """
    Local helper (only used by this solver): bbox of largest connected component in
    `mask` that does NOT touch the grid border.
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
        if bool(
            (rs == 0).any()
            or (rs == int(H - 1)).any()
            or (cs == 0).any()
            or (cs == int(W - 1)).any()
        ):
            continue
        area = int(coords.shape[0])
        if area > best_area:
            best_area = area
            r0, c0 = coords.min(axis=0)
            r1, c1 = coords.max(axis=0)
            best_bbox = (int(r0), int(c0), int(r1), int(c1))

    return best_bbox


def _extract_left_sprites(
    g: np.ndarray, bg: int, fc0: int, divider_mask: np.ndarray
) -> list[dict[str, int | np.ndarray]]:
    """Return per-color 8-connected sprites from the left strip, excluding the divider."""
    W = int(g.shape[1])
    left_cols = np.arange(W)[None, :] < int(fc0)
    sprites: list[dict[str, int | np.ndarray]] = []
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
    return sprites


def _erase_sprite(out: np.ndarray, sprite: dict[str, int | np.ndarray], bg: int) -> None:
    """Paint one extracted sprite back to background at its source location."""
    r0, c0 = int(sprite["r0"]), int(sprite["c0"])
    mask = np.asarray(sprite["mask"], dtype=bool)
    for rr, cc in np.argwhere(mask):
        out[int(r0) + int(rr), int(c0) + int(cc)] = int(bg)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Combined rule:
    - marker-matched sprites are copied into the frame and kept at their source
    - unmatched sprites are deleted from the left strip
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

    # Frame interior = largest enclosed background component.
    inner_bbox = largest_enclosed_component_bbox(g == int(bg), connectivity=4)
    assert inner_bbox is not None
    ir0, ic0, ir1, ic1 = map(int, inner_bbox)
    fr0, fc0, fr1, fc1 = int(ir0 - 1), int(ic0 - 1), int(ir1 + 1), int(ic1 + 1)

    # Detect the solid divider column(s) to the left of the frame.
    divider_mask = np.zeros((H, W), dtype=bool)
    for c in range(0, max(0, int(fc0))):
        colv = g[:, int(c)]
        non = colv[colv != int(bg)]
        if int(len(non)) >= int(H - 1) and len(set(map(int, non))) == 1:
            divider_mask[:, int(c)] = True

    sprites = _extract_left_sprites(g, bg, fc0, divider_mask)
    sprite_colors = {int(s["color"]) for s in sprites}

    # Mirrored edge markers determine mover colors and target rows/cols.
    row_marks: dict[int, list[int]] = defaultdict(list)
    col_marks: dict[int, list[int]] = defaultdict(list)
    for r in range(int(ir0), int(ir1) + 1):
        cl = int(g[int(r), int(fc0)])
        cr = int(g[int(r), int(fc1)])
        if cl == cr and cl != int(bg) and cl in sprite_colors:
            row_marks[int(cl)].append(int(r))
    for c in range(int(ic0), int(ic1) + 1):
        ct = int(g[int(fr0), int(c)])
        cb = int(g[int(fr1), int(c)])
        if ct == cb and ct != int(bg) and ct in sprite_colors:
            col_marks[int(ct)].append(int(c))

    targets: dict[int, list[tuple[int, int]]] = {}
    for col in set(row_marks.keys()) & set(col_marks.keys()):
        pts = {(int(r), int(c)) for r in row_marks[int(col)] for c in col_marks[int(col)]}
        targets[int(col)] = sorted(pts)

    out = g.copy()
    movers = [s for s in sprites if int(s["color"]) in targets]
    distractors = [s for s in sprites if int(s["color"]) not in targets]

    # Combined behavior: movers keep their sources; distractors are deleted.
    for s in distractors:
        _erase_sprite(out, s, bg)

    # Lower sprites stamp first; higher sprites stamp later and cover them.
    movers.sort(key=lambda s: (-((int(s["r0"]) + int(s["r1"])) / 2.0), int(s["c0"])))

    for s in movers:
        col = int(s["color"])
        cr = (int(s["r0"]) + int(s["r1"])) // 2
        cc = (int(s["c0"]) + int(s["c1"])) // 2
        mask = np.asarray(s["mask"], dtype=bool)
        pix = [(int(s["r0"]) + int(rr), int(s["c0"]) + int(cc2)) for rr, cc2 in np.argwhere(mask)]

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
    if sh * sw < 3:
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

    # Dimensions: frame spans full grid height, matching the source task spec.
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

    # Distinct colors preserve unambiguous ownership of sprites and markers.
    all_colors = list(range(1, 10))
    random.shuffle(all_colors)
    n_movers = random.randint(2, 3)
    n_distractors = random.choice([1, 1, 1, 0])
    idx = 0
    divider_color = all_colors[idx]
    idx += 1
    frame_base = all_colors[idx]
    idx += 1
    mover_colors = all_colors[idx : idx + n_movers]
    idx += n_movers
    distractor_colors = all_colors[idx : idx + n_distractors]

    # Generate movers first because their shapes constrain legal target centers.
    mover_sprites: list[np.ndarray] = []
    for mc in mover_colors:
        s = _make_sprite(mc, 5, sprite_area_w)
        if s is None:
            return None
        mover_sprites.append(s)

    used_h = sum(int(sp.shape[0]) for sp in mover_sprites)
    remaining_h = H - used_h
    if n_distractors > 0 and remaining_h < 1:
        return None

    mover_targets: dict[int, tuple[list[int], list[int]]] = {}
    occupied_rows: set[int] = set()
    occupied_cols: set[int] = set()
    for mc, sp in zip(mover_colors, mover_sprites):
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
    mover_source_pixels: list[tuple[int, int, int]] = []
    distractor_source_pixels: list[tuple[int, int, int]] = []
    all_items: list[tuple[int, np.ndarray, bool]] = [
        (mc, sp, False) for mc, sp in zip(mover_colors, mover_sprites)
    ]
    for dc_color in distractor_colors:
        ds = _make_sprite(dc_color, min(5, H - cur_row), sprite_area_w)
        if ds is None:
            return None
        all_items.append((dc_color, ds, True))

    for color, sp, is_distractor in all_items:
        sh, sw = sp.shape
        if sw > sprite_area_w or cur_row + sh > H:
            return None
        c_off = random.randint(0, sprite_area_w - sw)
        for dr in range(sh):
            for dc in range(sw):
                if sp[dr, dc] != bg:
                    rr = cur_row + dr
                    cc = c_off + dc
                    inp[rr, cc] = sp[dr, dc]
                    if is_distractor:
                        distractor_source_pixels.append((rr, cc, color))
                    else:
                        mover_source_pixels.append((rr, cc, color))
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

    if not mover_source_pixels:
        return None
    for rr, cc, mc in mover_source_pixels:
        if int(output[rr, cc]) != int(mc):
            return None

    mover_visible_in_frame = False
    for mc in mover_colors:
        if bool(np.any(output[int(ir0) : int(ir1) + 1, int(ic0) : int(ic1) + 1] == int(mc))):
            mover_visible_in_frame = True
            break
    if not mover_visible_in_frame:
        return None

    for rr, cc, dc in distractor_source_pixels:
        if int(output[rr, cc]) != int(bg):
            return None
        if int(np.count_nonzero(output == int(dc))) != 0:
            return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(300):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input: failed after 300 attempts")
