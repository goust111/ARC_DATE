# Variant of 247ef758
# Dimension: D14 — 框架基色与标记的区分方式 (meta)
# Change: 框架边缘由单一基色构成，标记是边缘上的异色像素，标记色==精灵色
#   → 框架边缘改为由两种 mover 颜色交替形成的条纹；标记不再依赖“某像素颜色不同于基色”，
#     而是依赖“某位置打断了原本的交替规律”。解题者需要先恢复条纹的基准相位，再找异常位。
#
# Design analysis
# ---------------
# concrete_realization:
#   grid:
#     same macro layout as the source task:
#       left sprite strip | uniform divider column | right rectangular frame
#     bg remains 0 and the frame still encloses one large bg rectangle.
#   frame:
#     the whole perimeter uses exactly the two mover colors, alternating like A/B/A/B...
#     top==bottom and left==right are still mirrored.
#     row/col markers are encoded by parity breaks:
#       if a position was expected to be color B, we may force it to color A instead.
#       this yields a local "... A A A ..." run inside an otherwise alternating edge.
#     the observed color at the anomalous position is still the mover identity, but the key
#     distinction is that ordinary edge pixels also use mover colors. Therefore color equality
#     alone is insufficient; the solver must detect mismatch-to-pattern.
#   sprites:
#     exactly two movers, matching the two stripe colors. Optional distractor sprites keep the
#     original "present on the left, absent from the frame markers" role.
# behavior_trace:
#   original row logic:
#     read left/right edge pixels; if both equal some sprite color, that row is a mark.
#   variant row logic:
#     fit the best 2-color alternation to the left/right edges first.
#     a row is a mark only when both opposite edges equal each other AND both disagree with the
#     fitted alternating expectation at that row. The observed anomalous color chooses the mover.
#   difference:
#     in the variant, many non-marker edge pixels share mover colors and would be false positives
#     under the original rule; only parity-breaking positions count.
# complexity_calibration:
#   keep only two movers so the 2-color stripe is readable and unambiguous.
#   each mover gets 1-2 row marks and 1-2 col marks, so examples still show repetition and the
#   cartesian-product stamping remains visible.
#   distractors stay optional but are limited to at most one so the left strip remains legible.
# implicit_assumption_audit:
#   largest_enclosed_component_bbox(...) still works because the frame remains a closed perimeter.
#   divider detection still works because the divider is still a single full-height uniform column.
#   left-side sprite extraction by connected components still works even though mover colors also
#   appear on the frame, because extraction is restricted to columns left of the frame.
#   the original marker test:
#       cl == cr and cl in move_colors
#     is no longer valid, because every ordinary stripe position would satisfy it.
#     it must be replaced with:
#       cl == cr and cl != expected_alternating_color_at_that_position.
# placement_legality:
#   generator chooses row/col marks only where the base stripe expects the opposite mover color,
#   so every chosen mark is guaranteed to create a visible anomaly.
#   mark spacing still uses sprite bbox size to keep stamped copies from collapsing into one blob.
#   target centers are chosen so every stamped sprite fits inside the frame interior.
# observability_and_cleanliness:
#   humans can see two regular stripe colors on the frame and a small number of broken positions.
#   because only two edge colors exist, the "regular vs broken" contrast is strong across 2-4
#   training examples. The frame is visually cleaner than introducing extra tiny key objects.
# color_inference:
#   no hidden ordering by color id is used in the task rule.
#   the anomalous edge color is itself the mover identity; the solver may internally sort the
#   two stripe colors only to enumerate the two possible alternating phases.
# new_solver_assumptions_to_guarantee:
#   frame perimeter contains exactly two non-bg colors.
#   opposite edges are mirrored.
#   marks occur only at interior edge positions and only as single-cell substitutions, so the
#   underlying alternation phase remains globally recoverable by minimum mismatches.

from __future__ import annotations

from collections import defaultdict
import random

import numpy as np

from arc_utils import get_bg, label_mask


def largest_enclosed_component_bbox(mask: np.ndarray, *, connectivity: int = 4) -> tuple[int, int, int, int] | None:
    """
    Local helper: bbox of the largest connected component in `mask`
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
    Move colored sprites from the left strip into a framed area at stripe-break targets.

    Variant rule:
    - The frame perimeter alternates between the two mover colors.
    - Markers are not "special colors"; they are positions where the alternation is broken.
    - For each mover color, row marks are anomalous positions on both vertical edges with that
      observed color; col marks are analogous anomalies on the top/bottom edges.
    - As in the source task, stamp each sprite centered at the Cartesian product of its row and
      col marks. Lower sprites stamp first; upper sprites overwrite them.
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

    left_cols = (np.arange(W)[None, :] < int(fc0))

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
    expected_cols = _best_alternating_expectation([top_seq, bottom_seq], frame_colors)

    row_marks: dict[int, list[int]] = defaultdict(list)
    col_marks: dict[int, list[int]] = defaultdict(list)

    for offset, r in enumerate(range(int(ir0), int(ir1) + 1)):
        cl = int(g[int(r), int(fc0)])
        cr = int(g[int(r), int(fc1)])
        if cl == cr and cl != int(expected_rows[int(offset)]) and cl in move_colors:
            row_marks[int(cl)].append(int(r))

    for offset, c in enumerate(range(int(ic0), int(ic1) + 1)):
        ct = int(g[int(fr0), int(c)])
        cb = int(g[int(fr1), int(c)])
        if ct == cb and ct != int(expected_cols[int(offset)]) and ct in move_colors:
            col_marks[int(ct)].append(int(c))

    targets: dict[int, list[tuple[int, int]]] = {}
    for col in set(row_marks.keys()) & set(col_marks.keys()):
        pts = {(int(r), int(c)) for r in row_marks[int(col)] for c in col_marks[int(col)]}
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


def _base_stripe_color(colors: list[int], pos: int, phase: int) -> int:
    return int(colors[(int(pos) + int(phase)) % 2])


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
    col_phase = random.randint(0, 1)

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

        avail_rows = [
            r
            for r in range(min_row, max_row + 1)
            if r not in occupied_rows and _base_stripe_color(mover_colors, r, row_phase) != mc
        ]
        row_marks = _pick_spaced(avail_rows, random.randint(1, 2), sh)
        if not row_marks:
            return None

        avail_cols = [
            c
            for c in range(min_col, max_col + 1)
            if c not in occupied_cols and _base_stripe_color(mover_colors, c - fc0, col_phase) != mc
        ]
        col_marks = _pick_spaced(avail_cols, random.randint(1, 2), sw)
        if not col_marks:
            return None

        mover_targets[mc] = (sorted(row_marks), sorted(col_marks))
        occupied_rows.update(row_marks)
        occupied_cols.update(col_marks)

    inp = np.full((H, W), bg, dtype=int)
    inp[:, div_col] = divider_color

    top_edge = np.array([_base_stripe_color(mover_colors, i, col_phase) for i in range(frame_w)], dtype=int)
    for mc, (_, cmarks) in mover_targets.items():
        for cc in cmarks:
            top_edge[int(cc - fc0)] = int(mc)
    inp[0, fc0: fc0 + frame_w] = top_edge
    inp[H - 1, fc0: fc0 + frame_w] = top_edge

    for r in range(H):
        val = _base_stripe_color(mover_colors, r, row_phase)
        for mc, (rmarks, _) in mover_targets.items():
            if r in rmarks:
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

    for _, sp in all_items:
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
    if np.array_equal(inp, output):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(300):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input: failed after 300 attempts")
