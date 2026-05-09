# Combo variant of 247ef758 (index 34)
# combo_spec: D1/0+D3/1
# Name: 形状嵌合匹配+多框架

from __future__ import annotations

from collections import defaultdict
import random

import numpy as np

from arc_utils import get_bg, label_mask


# ---------------------------------------------------------------------------
# Design analysis for the combo variant
# ---------------------------------------------------------------------------
#
# combo_summary:
#   - Original 247ef758 has one right-side frame and uses edge colors to choose
#     target rows/cols for each left-strip sprite.
#   - D1/0 replaces color matching with shape matching: the frame edge contains
#     bg notches whose silhouettes match the sprite bbox mask.
#   - D3/1 replaces the single frame with several independent frames, so a
#     sprite must first determine *which* frame it belongs to and only then read
#     its target row/col marks inside that frame.
#
# concrete_layout:
#   - Keep the original global split: left sprite strip | full-height divider |
#     right canvas.
#   - The right canvas is now a 15x15 panel with 2-3 separate thick frames:
#       * top-left 7x7 frame
#       * top-right 7x7 frame
#       * bottom 7x15 wide frame
#   - Every frame is a solid base-colored rectangle with a carved 3-row-high bg
#     interior. The frame edges never use mover colors.
#   - Each mover is a 2x2 asymmetric L-shape. Its exact silhouette is carved as
#     bg on both left/right edges at one row mark and on both top/bottom edges
#     at one or more col marks.
#
# behavior_trace_small_example:
#   - Original:
#       red sprite on the left -> find red on both vertical edges of the single
#       frame and red on both horizontal edges -> stamp red at the Cartesian
#       product centers.
#   - Combo variant:
#       a green L-shape on the left no longer has any green pixels on frames.
#       Instead, one frame has that same L-shape cut into its left/right edges
#       at row r and into its top/bottom edges at cols c1,c2. The solver must:
#         1. enumerate all frames,
#         2. test which frame edge patches geometrically match the green sprite,
#         3. recover row mark r and col marks c1,c2 from that frame,
#         4. stamp the sprite at (r,c1) and (r,c2) inside only that frame.
#   - Difference from the original:
#       color is no longer the routing key; geometry is. Also the destination is
#       no longer a single privileged frame.
#
# complexity_calibration:
#   - Using 2x2 asymmetric templates keeps the notches visually obvious while
#     still fitting multiple frames inside the original 10-15 x 14-20 envelope.
#   - Three distinct L-shapes are enough for 2-3 movers without ambiguity, and
#     the optional distractor can use the remaining unused L-shape.
#   - The bottom wide frame always receives two column marks, so at least one
#     mover visibly replicates to multiple targets. This prevents the task from
#     degenerating into pure one-to-one routing.
#
# implicit_assumption_audit_against_original_main:
#   - "largest enclosed bg component == the only frame interior" breaks because
#     there are now multiple frames and the wide bottom frame can dominate the
#     enclosed area.
#   - "frame is one cell thick" breaks because notch matching needs a thick band.
#   - "edge markers are single colored pixels" breaks because the cue is now a
#     2x2 notch pattern cut into the frame.
#   - "a mover color appears on the frame edges" breaks completely; mover colors
#     are absent from the destination side, so the old color-based logic would
#     infer no valid targets.
#   - New solver assumptions introduced here:
#       * there is exactly one full-height uniform divider column;
#       * every frame base color is unique and appears only inside one frame;
#       * frame bboxes are 7 rows tall, with width 7 or 15, and their interior
#         is always the bbox shrunk by 2 cells on each side;
#       * mover silhouettes are unique normalized masks, so one frame cannot
#         accidentally match two movers.
#     The generator explicitly enforces all of these.
#
# placement_legality:
#   - Frames occupy disjoint rectangles separated by bg gaps.
#   - Notch patches are kept away from corners. The top small frames use a
#     single column mark, while the wide bottom frame uses two spaced column
#     marks so top/bottom notch patches do not overlap.
#   - Every sprite is generated from a 2x2 template and placed within the left
#     strip without overlap or out-of-bounds writes.
#
# observability_and_cleanliness:
#   - Humans can see three separable ideas directly in the pixels:
#       * multiple independent boxes on the right,
#       * same-shape cutouts repeated on opposite edges,
#       * the matched sprite being re-stamped only inside the corresponding box.
#   - Because mover colors never appear on frames, the new rule cannot be
#     mistaken for the original color-coded rule.
#   - Distinct frame base colors help segment the right canvas cleanly without
#     introducing any hidden numerical ordering.


_SPRITE_TEMPLATES = [
    np.array([[1, 1], [1, 0]], dtype=int),
    np.array([[1, 1], [0, 1]], dtype=int),
    np.array([[1, 0], [1, 1]], dtype=int),
    np.array([[0, 1], [1, 1]], dtype=int),
]

_FRAME_SPECS = [
    {"name": "top_left", "fr0": 0, "fc0": 5, "height": 7, "width": 7},
    {"name": "top_right", "fr0": 0, "fc0": 13, "height": 7, "width": 7},
    {"name": "bottom", "fr0": 8, "fc0": 5, "height": 7, "width": 15},
]


def _mask_signature(mask: np.ndarray) -> tuple[tuple[int, int], tuple[int, ...]]:
    arr = np.array(mask, dtype=int)
    return (int(arr.shape[0]), int(arr.shape[1])), tuple(map(int, arr.flatten()))


def _find_divider_mask(grid: np.ndarray, bg: int) -> tuple[np.ndarray, int]:
    g = np.array(grid, dtype=int)
    h, w = g.shape
    divider_mask = np.zeros((h, w), dtype=bool)
    divider_cols: list[int] = []
    for c in range(w):
        colv = g[:, int(c)]
        if bool((colv != int(bg)).all()) and len(set(map(int, colv))) == 1:
            divider_mask[:, int(c)] = True
            divider_cols.append(int(c))
    assert divider_cols
    return divider_mask, int(min(divider_cols))


def _find_frames(grid: np.ndarray, bg: int, divider_col: int) -> list[dict[str, int]]:
    g = np.array(grid, dtype=int)
    right = np.arange(g.shape[1])[None, :] > int(divider_col)
    frames: list[dict[str, int]] = []

    for col in sorted(set(map(int, g.flatten())) - {int(bg)}):
        mask = (g == int(col)) & right
        if not bool(mask.any()):
            continue
        coords = np.argwhere(mask)
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        r0, c0, r1, c1 = map(int, (r0, c0, r1, c1))
        height = int(r1 - r0 + 1)
        width = int(c1 - c0 + 1)
        if height != 7 or width not in (7, 15):
            continue
        if any(
            int(g[rr, cc]) != int(col)
            for rr, cc in ((r0, c0), (r0, c1), (r1, c0), (r1, c1))
        ):
            continue
        patch = g[r0 : r1 + 1, c0 : c1 + 1]
        nonbg = patch[patch != int(bg)]
        if nonbg.size == 0 or not bool((nonbg == int(col)).all()):
            continue
        if not bool(np.all(g[r0 + 2 : r1 - 1, c0 + 2 : c1 - 1] == int(bg))):
            continue
        frames.append(
            {
                "fr0": int(r0),
                "fc0": int(c0),
                "fr1": int(r1),
                "fc1": int(c1),
                "ir0": int(r0 + 2),
                "ic0": int(c0 + 2),
                "ir1": int(r1 - 2),
                "ic1": int(c1 - 2),
                "base": int(col),
            }
        )

    frames.sort(key=lambda d: (int(d["fr0"]), int(d["fc0"])))
    return frames


def _extract_sprites(grid: np.ndarray, bg: int, divider_mask: np.ndarray, divider_col: int) -> list[dict[str, object]]:
    g = np.array(grid, dtype=int)
    left_cols = np.arange(g.shape[1])[None, :] < int(divider_col)
    sprites: list[dict[str, object]] = []

    for col in sorted(set(map(int, g.flatten())) - {int(bg)}):
        mask = (g == int(col)) & left_cols & (~divider_mask)
        if not bool(mask.any()):
            continue
        lab, n = label_mask(mask, connectivity=8)
        for sid in range(1, int(n) + 1):
            pts = np.argwhere(lab == int(sid))
            if pts.size == 0:
                continue
            r0, c0 = pts.min(axis=0)
            r1, c1 = pts.max(axis=0)
            r0, c0, r1, c1 = map(int, (r0, c0, r1, c1))
            sprite_mask = lab[r0 : r1 + 1, c0 : c1 + 1] == int(sid)
            sprites.append(
                {
                    "color": int(col),
                    "r0": int(r0),
                    "c0": int(c0),
                    "r1": int(r1),
                    "c1": int(c1),
                    "mask": sprite_mask,
                    "sig": _mask_signature(sprite_mask),
                }
            )

    return sprites


def _center_candidates(lo: int, hi: int, span: int) -> list[int]:
    offset = (int(span) - 1) // 2
    start = int(lo + offset)
    end = int(hi - (int(span) - 1 - offset))
    if start > end:
        return []
    return list(range(int(start), int(end) + 1))


def _patch_matches_notch(patch: np.ndarray, mask: np.ndarray, bg: int, base: int) -> bool:
    if patch.shape != mask.shape:
        return False
    return bool(np.all(patch[mask] == int(bg)) and np.all(patch[~mask] == int(base)))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move left-strip sprites into one of several independent thick frames.

    Combo rule:
    - Detect multiple frame rectangles on the right.
    - For each frame, read row/col marks from bg notch shapes rather than colors.
    - Route each sprite to the unique frame whose opposite edge notches match its
      normalized silhouette, then stamp it at the row x col Cartesian product.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    divider_mask, divider_col = _find_divider_mask(g, bg)
    frames = _find_frames(g, bg, divider_col)
    assert len(frames) >= 2

    sprites = _extract_sprites(g, bg, divider_mask, divider_col)
    sig_to_colors: dict[tuple[tuple[int, int], tuple[int, ...]], list[int]] = defaultdict(list)
    for s in sprites:
        sig_to_colors[s["sig"]].append(int(s["color"]))

    targets: dict[int, list[tuple[int, int, int, int, int, int]]] = defaultdict(list)

    for frame in frames:
        fr0, fc0, fr1, fc1 = (int(frame[k]) for k in ("fr0", "fc0", "fr1", "fc1"))
        ir0, ic0, ir1, ic1 = (int(frame[k]) for k in ("ir0", "ic0", "ir1", "ic1"))
        base = int(frame["base"])

        for sig, cols in sig_to_colors.items():
            shape, flat = sig
            sh, sw = map(int, shape)
            mask = np.array(flat, dtype=int).reshape((sh, sw)).astype(bool)

            row_marks: list[int] = []
            for tr in _center_candidates(ir0, ir1, sh):
                top = int(tr - ((sh - 1) // 2))
                left_patch = g[top : top + sh, fc0 : fc0 + sw]
                right_patch = g[top : top + sh, fc1 - sw + 1 : fc1 + 1]
                if _patch_matches_notch(left_patch, mask, bg, base) and _patch_matches_notch(
                    right_patch, mask, bg, base
                ):
                    row_marks.append(int(tr))

            col_marks: list[int] = []
            for tc in _center_candidates(ic0, ic1, sw):
                left = int(tc - ((sw - 1) // 2))
                top_patch = g[fr0 : fr0 + sh, left : left + sw]
                bottom_patch = g[fr1 - sh + 1 : fr1 + 1, left : left + sw]
                if _patch_matches_notch(top_patch, mask, bg, base) and _patch_matches_notch(
                    bottom_patch, mask, bg, base
                ):
                    col_marks.append(int(tc))

            if not row_marks or not col_marks:
                continue
            for col in cols:
                for tr in row_marks:
                    for tc in col_marks:
                        targets[int(col)].append((int(tr), int(tc), ir0, ic0, ir1, ic1))

    out = g.copy()
    movers = [s for s in sprites if int(s["color"]) in targets]

    for s in movers:
        r0, c0 = int(s["r0"]), int(s["c0"])
        for rr, cc in np.argwhere(s["mask"]):
            out[int(r0) + int(rr), int(c0) + int(cc)] = int(bg)

    movers.sort(key=lambda s: (-((int(s["r0"]) + int(s["r1"])) / 2.0), int(s["c0"])))

    for s in movers:
        color = int(s["color"])
        cr = (int(s["r0"]) + int(s["r1"])) // 2
        cc = (int(s["c0"]) + int(s["c1"])) // 2
        pix = [(int(s["r0"]) + int(rr), int(s["c0"]) + int(cc2)) for rr, cc2 in np.argwhere(s["mask"])]
        for tr, tc, ir0, ic0, ir1, ic1 in targets[int(color)]:
            dr = int(tr - cr)
            dc = int(tc - cc)
            for r, c in pix:
                rr = int(r + dr)
                cc3 = int(c + dc)
                if int(ir0) <= rr <= int(ir1) and int(ic0) <= cc3 <= int(ic1):
                    out[rr, cc3] = int(color)

    return out


def _colorize(mask: np.ndarray, color: int) -> np.ndarray:
    return np.where(mask.astype(bool), int(color), 0).astype(int)


def _pick_spaced(candidates: list[int], count: int, min_spacing: int) -> list[int] | None:
    vals = list(map(int, candidates))
    random.shuffle(vals)
    picked: list[int] = []
    for v in vals:
        if all(abs(int(v) - int(p)) >= int(min_spacing) for p in picked):
            picked.append(int(v))
            if len(picked) == int(count):
                return sorted(picked)
    return None


def _frame_specs_for_attempt() -> list[dict[str, int]]:
    top_choice = random.choice([_FRAME_SPECS[0], _FRAME_SPECS[1]])
    if random.random() < 0.5:
        return [top_choice, _FRAME_SPECS[2]]
    return [_FRAME_SPECS[0], _FRAME_SPECS[1], _FRAME_SPECS[2]]


def _gen_attempt() -> np.ndarray | None:
    bg = 0
    h, w = 15, 20
    sprite_area_w = 4
    divider_col = sprite_area_w

    frame_specs = _frame_specs_for_attempt()
    n_frames = len(frame_specs)
    n_distractors = random.choice([0, 0, 1])
    if n_frames + n_distractors > len(_SPRITE_TEMPLATES):
        return None

    template_ids = random.sample(range(len(_SPRITE_TEMPLATES)), n_frames + n_distractors)
    mover_templates = [_SPRITE_TEMPLATES[i] for i in template_ids[:n_frames]]
    distractor_templates = [_SPRITE_TEMPLATES[i] for i in template_ids[n_frames:]]

    all_colors = list(range(1, 10))
    random.shuffle(all_colors)
    idx = 0
    divider_color = int(all_colors[idx]); idx += 1
    frame_bases = list(map(int, all_colors[idx : idx + n_frames])); idx += n_frames
    mover_colors = list(map(int, all_colors[idx : idx + n_frames])); idx += n_frames
    distractor_colors = list(map(int, all_colors[idx : idx + n_distractors])); idx += n_distractors

    frame_infos: list[dict[str, object]] = []
    for spec, base, mover_color, tmpl in zip(frame_specs, frame_bases, mover_colors, mover_templates):
        fr0 = int(spec["fr0"])
        fc0 = int(spec["fc0"])
        fr1 = int(fr0 + spec["height"] - 1)
        fc1 = int(fc0 + spec["width"] - 1)
        ir0, ic0 = int(fr0 + 2), int(fc0 + 2)
        ir1, ic1 = int(fr1 - 2), int(fc1 - 2)
        sh, sw = map(int, tmpl.shape)

        row_candidates = _center_candidates(ir0, ir1, sh)
        col_candidates = _center_candidates(ic0, ic1, sw)
        if not row_candidates or not col_candidates:
            return None

        row_marks = [int(random.choice(row_candidates))]
        if spec["name"] == "bottom":
            col_count = 2
            col_marks = _pick_spaced(col_candidates, col_count, sw)
            if col_marks is None:
                return None
        else:
            col_marks = [int(random.choice(col_candidates))]

        frame_infos.append(
            {
                "fr0": int(fr0),
                "fc0": int(fc0),
                "fr1": int(fr1),
                "fc1": int(fc1),
                "ir0": int(ir0),
                "ic0": int(ic0),
                "ir1": int(ir1),
                "ic1": int(ic1),
                "base": int(base),
                "mover": int(mover_color),
                "sprite": _colorize(tmpl, int(mover_color)),
                "mask": tmpl.astype(bool),
                "rows": list(map(int, row_marks)),
                "cols": list(map(int, col_marks)),
            }
        )

    inp = np.full((h, w), bg, dtype=int)
    inp[:, int(divider_col)] = int(divider_color)

    for info in frame_infos:
        fr0, fc0, fr1, fc1 = (int(info[k]) for k in ("fr0", "fc0", "fr1", "fc1"))
        ir0, ic0, ir1, ic1 = (int(info[k]) for k in ("ir0", "ic0", "ir1", "ic1"))
        base = int(info["base"])
        mask = np.array(info["mask"], dtype=bool)
        sh, sw = map(int, mask.shape)

        inp[fr0 : fr1 + 1, fc0 : fc1 + 1] = int(base)
        inp[ir0 : ir1 + 1, ic0 : ic1 + 1] = int(bg)

        for tr in info["rows"]:
            top = int(tr - ((sh - 1) // 2))
            inp[top : top + sh, fc0 : fc0 + sw][mask] = int(bg)
            inp[top : top + sh, fc1 - sw + 1 : fc1 + 1][mask] = int(bg)
        for tc in info["cols"]:
            left = int(tc - ((sw - 1) // 2))
            inp[fr0 : fr0 + sh, left : left + sw][mask] = int(bg)
            inp[fr1 - sh + 1 : fr1 + 1, left : left + sw][mask] = int(bg)

    items: list[np.ndarray] = [np.array(info["sprite"], dtype=int) for info in frame_infos]
    for dc_color, tmpl in zip(distractor_colors, distractor_templates):
        items.append(_colorize(tmpl, int(dc_color)))
    random.shuffle(items)

    heights = [int(sp.shape[0]) for sp in items]
    extra = int(h - sum(heights))
    if extra < 0:
        return None
    gaps = [0] * (len(items) + 1)
    for _ in range(extra):
        gaps[random.randrange(len(gaps))] += 1

    cur_row = int(gaps[0])
    for i, sp in enumerate(items):
        sh, sw = map(int, sp.shape)
        if sw > sprite_area_w or cur_row + sh > h:
            return None
        c_off = random.randint(0, int(sprite_area_w - sw))
        mask = sp != int(bg)
        inp[cur_row : cur_row + sh, c_off : c_off + sw][mask] = sp[mask]
        cur_row += int(sh + gaps[i + 1])

    vals, counts = np.unique(inp, return_counts=True)
    if int(vals[np.argmax(counts)]) != int(bg):
        return None

    divider_mask, found_divider = _find_divider_mask(inp, bg)
    if int(found_divider) != int(divider_col):
        return None
    if len(_find_frames(inp, bg, found_divider)) != int(n_frames):
        return None

    mover_colors_set = {int(info["mover"]) for info in frame_infos}
    if bool(np.isin(inp[:, divider_col + 1 :], list(mover_colors_set)).any()):
        return None

    try:
        out = main(inp)
    except Exception:
        return None
    if np.array_equal(inp, out):
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(1200):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input: failed after 1200 attempts")
