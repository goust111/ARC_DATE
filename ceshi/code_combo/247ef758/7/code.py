# Combo variant of 247ef758 (index 7)
# combo_spec: D3/2+D14/0
# Name: 条纹框架非对称标记
#
# Design analysis
# ---------------
# concrete_realization:
#   - Macro layout stays the same: left sprite strip | one full-height divider column |
#     one right rectangular frame with a bg-filled interior.
#   - The frame perimeter no longer has a single base color. It uses exactly the two mover
#     colors in alternating stripes. Regular edge pixels therefore already look like mover
#     colors.
#   - A marker is now a stripe break: at a chosen position, the edge cell is forced to the
#     opposite mover color from the one the alternation expects there.
#   - Left/right edges still work together to define target rows: a row is active only when
#     both vertical edges show the same anomalous color at that row.
#   - Top/bottom edges no longer need to mirror each other. A column is active for a color if
#     that color appears as an anomalous stripe break on the top edge or on the bottom edge.
# behavior_trace_small_example:
#   - Suppose mover colors are A and B.
#   - Old task: top/bottom were mostly one base color, so seeing A on both top and bottom at
#     c=7 directly meant "A-column at 7".
#   - D14 alone: top/bottom can be A/B/A/B stripes, so many cells are already A or B; a column
#     mark must be detected as "this A/B cell breaks the local alternation", not merely by color.
#   - D3 alone: even if top has A at c=7 and bottom has no A at c=7, c=7 still counts for A.
#   - Fused rule here: first recover the alternating expectation on each horizontal edge
#     separately, then collect top anomalies and bottom anomalies independently, union them by
#     color, and take row_marks(color) x col_marks(color).
# complexity_calibration:
#   - Use exactly two movers so the 2-color stripe is legible and the fitted alternation is
#     unambiguous.
#   - Keep row marks mirrored across left/right so only one axis becomes asymmetric; otherwise
#     the puzzle would become too unconstrained.
#   - Force one mover to have top-only horizontal marks and the other to have bottom-only marks,
#     so the asymmetry is always visible in a single sample.
#   - Keep horizontal anomaly columns globally unique to avoid ambiguous shared targets.
# implicit_assumption_audit_against_source_main:
#   - `largest_enclosed_component_bbox(...)` still works because the frame remains a closed
#     rectangle around one large bg component.
#   - Divider detection still works because the divider remains a nearly full-height uniform
#     non-bg column left of the frame.
#   - Left-side sprite extraction still works because it is restricted to columns left of the
#     frame; mover colors appearing on the frame do not interfere.
#   - The source row test `cl == cr and cl in move_colors` is no longer sufficient because
#     every regular stripe row would satisfy it. We must additionally require
#     `cl != expected_vertical_stripe`.
#   - The source column test `ct == cb and ct in move_colors` is broken in two ways:
#       1. regular stripe cells would be false positives
#       2. top-only / bottom-only marks would be missed entirely
#     Therefore the fused solver must detect anomalies on top and bottom separately.
# placement_legality:
#   - Generator only places row marks where the vertical stripe expects the opposite mover color,
#     so every marked row is a visible stripe break.
#   - Generator only places top/bottom marks where the corresponding horizontal stripe expects
#     the opposite mover color, again guaranteeing visible breaks.
#   - Target rows/cols are chosen with spacing based on sprite bbox size so stamped copies do not
#     collapse into one blob too often, and all chosen centers fit inside the frame interior.
# observability_and_cleanliness:
#   - The frame stays visually clean because only two colors appear on it, arranged in a strong
#     alternating pattern with a few local breaks.
#   - The fused behavior is visible from examples because some target columns are justified only
#     by top-edge breaks and others only by bottom-edge breaks; using the old mirrored rule would
#     miss them.
# new_main_assumptions_and_generator_guarantees:
#   - Solver assumes the frame perimeter uses exactly two non-bg colors; generator enforces
#     exactly two movers and uses only those colors on the perimeter.
#   - Solver assumes row marks are mirrored across left/right; generator writes identical
#     anomalies on both vertical edges.
#   - Solver assumes horizontal anomalies are sparse enough that the best alternating phase is
#     recoverable per edge; generator uses only 1-2 breaks per mover on each relevant edge.

from __future__ import annotations

from collections import defaultdict
import random

import numpy as np

from arc_utils import get_bg, label_mask


def largest_enclosed_component_bbox(mask: np.ndarray, *, connectivity: int = 4) -> tuple[int, int, int, int] | None:
    """BBox of the largest connected component in `mask` that does not touch the outer border."""
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


def _best_alternating_expectation(seqs: list[list[int]], colors: list[int]) -> list[int]:
    """Fit the lower-error 2-color alternation to one or more equal-length sequences."""
    assert len(colors) == 2
    length = len(seqs[0])
    best_phase = 0
    best_err = None
    for phase in (0, 1):
        err = 0
        for seq in seqs:
            assert len(seq) == length
            for i, v in enumerate(seq):
                if int(v) != int(colors[(i + phase) % 2]):
                    err += 1
        if best_err is None or err < best_err:
            best_err = err
            best_phase = phase
    return [int(colors[(i + best_phase) % 2]) for i in range(length)]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move left-strip sprites into the frame using striped-edge anomaly markers.

    Fused rule:
    - The frame perimeter alternates between the two mover colors.
    - A target row is a left/right mirrored stripe break.
    - A target column is any top-edge or bottom-edge stripe break of that color.
    - Stamp each moved sprite at every row x column target, bottom sprite first.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

    inner_bbox = largest_enclosed_component_bbox(g == int(bg), connectivity=4)
    assert inner_bbox is not None
    ir0, ic0, ir1, ic1 = map(int, inner_bbox)
    fr0, fc0, fr1, fc1 = int(ir0 - 1), int(ic0 - 1), int(ir1 + 1), int(ic1 + 1)

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
            m = slab[int(r0): int(r1) + 1, int(c0): int(c1) + 1] == int(sid)
            sprites.append({"color": int(col), "r0": int(r0), "c0": int(c0), "r1": int(r1), "c1": int(c1), "mask": m})

    move_colors = {int(s["color"]) for s in sprites}

    frame_colors = sorted(
        set(map(int, g[int(fr0), int(fc0): int(fc1) + 1]))
        | set(map(int, g[int(fr1), int(fc0): int(fc1) + 1]))
        | set(map(int, g[int(fr0): int(fr1) + 1, int(fc0)]))
        | set(map(int, g[int(fr0): int(fr1) + 1, int(fc1)]))
    )
    assert len(frame_colors) == 2

    left_seq = [int(g[int(r), int(fc0)]) for r in range(int(ir0), int(ir1) + 1)]
    right_seq = [int(g[int(r), int(fc1)]) for r in range(int(ir0), int(ir1) + 1)]
    top_seq = [int(g[int(fr0), int(c)]) for c in range(int(ic0), int(ic1) + 1)]
    bottom_seq = [int(g[int(fr1), int(c)]) for c in range(int(ic0), int(ic1) + 1)]

    expected_rows = _best_alternating_expectation([left_seq, right_seq], frame_colors)
    expected_top = _best_alternating_expectation([top_seq], frame_colors)
    expected_bottom = _best_alternating_expectation([bottom_seq], frame_colors)

    row_marks: dict[int, list[int]] = defaultdict(list)
    col_marks: dict[int, list[int]] = defaultdict(list)

    for offset, r in enumerate(range(int(ir0), int(ir1) + 1)):
        cl = int(g[int(r), int(fc0)])
        cr = int(g[int(r), int(fc1)])
        if cl == cr and cl in move_colors and cl != int(expected_rows[int(offset)]):
            row_marks[int(cl)].append(int(r))

    for offset, c in enumerate(range(int(ic0), int(ic1) + 1)):
        ct = int(g[int(fr0), int(c)])
        cb = int(g[int(fr1), int(c)])
        if ct in move_colors and ct != int(expected_top[int(offset)]):
            col_marks[int(ct)].append(int(c))
        if cb in move_colors and cb != int(expected_bottom[int(offset)]):
            col_marks[int(cb)].append(int(c))

    targets: dict[int, list[tuple[int, int]]] = {}
    for col in set(row_marks.keys()) & set(col_marks.keys()):
        cols = sorted(set(col_marks[int(col)]))
        pts = {(int(r), int(c)) for r in row_marks[int(col)] for c in cols}
        if pts:
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
    """Pick n values with pairwise spacing >= min_spacing using randomized greedy search."""
    if n == 0:
        return []
    if not candidates:
        return None
    candidates = list(candidates)
    for _ in range(12):
        random.shuffle(candidates)
        picked: list[int] = []
        for v in candidates:
            if all(abs(v - p) >= min_spacing for p in picked):
                picked.append(v)
                if len(picked) == n:
                    return sorted(picked)
    return None


def _make_sprite(color: int, max_h: int, max_w: int) -> np.ndarray | None:
    """Generate one 8-connected single-color sprite cropped to its bbox."""
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
        dr, dc = random.choice([(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)])
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


def _base_stripe_color(colors: list[int], pos: int, phase: int) -> int:
    return int(colors[(int(pos) + int(phase)) % 2])


def _assign_horizontal_marks(
    mover_colors: list[int],
    sprites: list[np.ndarray],
    ic0: int,
    ic1: int,
    fc0: int,
    top_phase: int,
    bottom_phase: int,
) -> dict[int, tuple[list[int], list[int]]] | None:
    """
    Assign horizontal stripe-break columns.

    Exactly one mover is top-only and the other is bottom-only, making the asymmetry explicit.
    """
    top_owner, bottom_owner = mover_colors[0], mover_colors[1]
    if random.random() < 0.5:
        top_owner, bottom_owner = bottom_owner, top_owner

    sprite_by_color = {int(mc): sp for mc, sp in zip(mover_colors, sprites)}
    occupied_cols: set[int] = set()
    result: dict[int, tuple[list[int], list[int]]] = {}

    for mc in mover_colors:
        sp = sprite_by_color[int(mc)]
        sw = int(sp.shape[1])
        min_col = int(ic0 + sw // 2)
        max_col = int(ic1 - (sw - 1 - sw // 2))
        if min_col > max_col:
            return None

        if int(mc) == int(top_owner):
            top_n, bottom_n = random.choice([1, 2]), 0
        else:
            top_n, bottom_n = 0, random.choice([1, 2])

        avail_top = [
            c
            for c in range(min_col, max_col + 1)
            if c not in occupied_cols and _base_stripe_color(mover_colors, c - fc0, top_phase) != int(mc)
        ]
        avail_bottom = [
            c
            for c in range(min_col, max_col + 1)
            if c not in occupied_cols and _base_stripe_color(mover_colors, c - fc0, bottom_phase) != int(mc)
        ]

        top_cols = _pick_spaced(avail_top, int(top_n), max(1, sw))
        if top_cols is None:
            return None
        occupied_cols.update(top_cols)

        avail_bottom = [c for c in avail_bottom if c not in occupied_cols]
        bottom_cols = _pick_spaced(avail_bottom, int(bottom_n), max(1, sw))
        if bottom_cols is None:
            return None
        occupied_cols.update(bottom_cols)

        result[int(mc)] = (top_cols, bottom_cols)

    if not any(len(top) > 0 and len(bottom) == 0 for top, bottom in result.values()):
        return None
    if not any(len(bottom) > 0 and len(top) == 0 for top, bottom in result.values()):
        return None
    return result


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
    n_movers = 2
    n_distractors = random.choice([0, 0, 1])
    idx = 0
    divider_color = all_colors[idx]
    idx += 1
    mover_colors = all_colors[idx: idx + n_movers]
    idx += n_movers
    distractor_colors = all_colors[idx: idx + n_distractors]

    sprites: list[np.ndarray] = []
    for mc in mover_colors:
        s = _make_sprite(mc, 5, sprite_area_w)
        if s is None:
            return None
        sprites.append(s)

    row_phase = random.randint(0, 1)
    top_phase = random.randint(0, 1)
    bottom_phase = random.randint(0, 1)

    mover_rows: dict[int, list[int]] = {}
    occupied_rows: set[int] = set()
    for mc, sp in zip(mover_colors, sprites):
        sh, sw = sp.shape
        min_row = int(ir0 + sh // 2)
        max_row = int(ir1 - (sh - 1 - sh // 2))
        min_col = int(ic0 + sw // 2)
        max_col = int(ic1 - (sw - 1 - sw // 2))
        if min_row > max_row or min_col > max_col:
            return None

        avail_rows = [
            r
            for r in range(min_row, max_row + 1)
            if r not in occupied_rows and _base_stripe_color(mover_colors, r, row_phase) != int(mc)
        ]
        row_marks = _pick_spaced(avail_rows, random.choice([1, 2]), max(1, sh))
        if row_marks is None:
            return None
        mover_rows[int(mc)] = row_marks
        occupied_rows.update(row_marks)

    mover_cols = _assign_horizontal_marks(mover_colors, sprites, ic0, ic1, fc0, top_phase, bottom_phase)
    if mover_cols is None:
        return None

    inp = np.full((H, W), bg, dtype=int)
    inp[:, div_col] = divider_color

    top_edge = np.array([_base_stripe_color(mover_colors, i, top_phase) for i in range(frame_w)], dtype=int)
    bottom_edge = np.array([_base_stripe_color(mover_colors, i, bottom_phase) for i in range(frame_w)], dtype=int)
    for mc in mover_colors:
        top_cols, bottom_cols = mover_cols[int(mc)]
        for cc in top_cols:
            top_edge[int(cc - fc0)] = int(mc)
        for cc in bottom_cols:
            bottom_edge[int(cc - fc0)] = int(mc)
    inp[0, fc0: fc0 + frame_w] = top_edge
    inp[H - 1, fc0: fc0 + frame_w] = bottom_edge

    for r in range(1, H - 1):
        val = _base_stripe_color(mover_colors, r, row_phase)
        for mc in mover_colors:
            if r in mover_rows[int(mc)]:
                val = int(mc)
                break
        inp[r, fc0] = val
        inp[r, fc1] = val

    cur_row = 0
    all_items: list[tuple[int, np.ndarray]] = [(mc, sp) for mc, sp in zip(mover_colors, sprites)]
    for dc_color in distractor_colors:
        ds = _make_sprite(dc_color, min(5, H - cur_row), sprite_area_w)
        if ds is None:
            return None
        all_items.append((dc_color, ds))

    for _color, sp in all_items:
        sh, sw = sp.shape
        if sw > sprite_area_w or cur_row + sh > H:
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

    expected_top = [_base_stripe_color(mover_colors, c - fc0, top_phase) for c in range(int(ic0), int(ic1) + 1)]
    expected_bottom = [_base_stripe_color(mover_colors, c - fc0, bottom_phase) for c in range(int(ic0), int(ic1) + 1)]
    top_only_breaks = 0
    bottom_only_breaks = 0
    for off, c in enumerate(range(int(ic0), int(ic1) + 1)):
        top_break = int(inp[0, c]) != int(expected_top[off])
        bottom_break = int(inp[H - 1, c]) != int(expected_bottom[off])
        if top_break and not bottom_break:
            top_only_breaks += 1
        if bottom_break and not top_break:
            bottom_only_breaks += 1
    if top_only_breaks == 0 or bottom_only_breaks == 0:
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
