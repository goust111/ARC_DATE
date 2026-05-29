# Combo variant of 3e6067c3 (index 19)
# combo_spec: D9/1+D1/4
# Name: 图例映射+互补色匹配
#
# Design analysis
# - concretization:
#   - The framed-square lattice and bridge geometry stay the same as the original task.
#   - The old single control row is replaced by two separate clue regions below the
#     squares:
#       1. a 2-row horizontal mapping legend `control_color -> frame_inner_color`;
#       2. a 2-column vertical pair legend whose each row is
#          `(start_control_color, end_control_color)`.
#   - Solving now requires two linked steps: translate both colors on each pair row
#     through the mapping legend, then find the corresponding start/end frames and draw
#     the bridge using the raw left control color.
# - behavior trace on a small example:
#   - Suppose the mapping legend says `2->7`, `6->4`, `3->9`.
#   - A pair row `[2, 6]` therefore means “connect the 7-filled frame to the 4-filled
#     frame” and paint that bridge with color 2.
#   - In the original task, the same visible colors would have meant “connect the
#     2-filled frame to the 6-filled frame”, and under the D9-only variant the row
#     would still match frames directly by 2 and 6. The fused variant changes both the
#     decoding step and the bridge color semantics.
# - complexity calibration:
#   - D9 removes sequential context, so the generator uses globally unique frame inner
#     colors to avoid unrelated ambiguity about which repeated square a legend row names.
#   - D1 adds an indirect color layer, but the mapping is still learnable because it is
#     shown explicitly as a small derangement over the actually used frame colors.
#   - Pair rows are shuffled away from path order when possible, so the lower block reads
#     as an explicit lookup table rather than a disguised walk sequence.
# - implicit-assumption audit against the original main():
#   - The original `ctrl_r` / `ctrl = [...]` logic assumed one informative row. That is
#     now false: the solver must detect both the vertical pair block and the separate
#     horizontal mapping legend.
#   - The original `ctrl` colors were used directly as frame-inner lookup keys. That is
#     false after D1: pair-legend colors first map through `control -> inner`.
#   - The original persistent `cur` traversal state is not valid after D9. Each pair row
#     is independent, so the solver resolves a fresh `(start_frame, end_frame)` for every
#     row instead of walking consecutive control tokens.
#   - The original `draw_bridge(..., a)` happened to use the same color for matching and
#     painting. In the fused rule, matching uses the translated inner color while painting
#     uses the raw left control color.
# - placement legality:
#   - Frames are still placed on a self-avoiding lattice path, so every requested bridge
#     is horizontal or vertical and has non-empty bridge cells.
#   - Bridge cells are required to be pairwise disjoint and to lie on background before
#     painting; invalid attempts are rejected.
#   - The mapping legend and the pair legend are both placed strictly below the square
#     field with vertical gaps between regions, so they never overlap frames or bridges.
# - visual observability / cleanliness:
#   - The two clue regions use different orientations on purpose: a 2-row horizontal map
#     for “color translation”, then a 2-column vertical table for “bridge requests”.
#     That keeps the fused rule legible in 2-4 examples instead of collapsing into one
#     visually ambiguous blob.
#   - No numeric ordering or hidden palette convention is used. The mapping is inferred
#     only from visible adjacency inside the legend.
# - new solver assumptions and generator guarantees:
#   - Solver assumption: below the frame field there is exactly one contiguous block of
#     rows with two adjacent pixels per row (the pair legend).
#     Generator guarantee: it constructs exactly one such block.
#   - Solver assumption: above that block there is one 2-row legend whose occupied
#     columns match and define a control->inner mapping.
#     Generator guarantee: it constructs exactly one such legend with at least 4 entries.
#   - Solver assumption: every translated pair names aligned frames.
#     Generator guarantee: every pair row comes from an actual edge of the underlying
#     lattice path.

from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np

from arc_utils import get_bg, in_bounds, label_mask, neighbors4


def _detect_frames(g: np.ndarray, border: int) -> list[dict]:
    lab, n = label_mask(g == int(border), connectivity=4)
    frames: list[dict] = []
    for cid in range(1, int(n) + 1):
        pts = np.argwhere(lab == int(cid))
        if int(pts.shape[0]) < 8:
            continue
        r0, c0 = pts.min(axis=0)
        r1, c1 = pts.max(axis=0)
        r0, c0, r1, c1 = map(int, (r0, c0, r1, c1))
        h = int(r1 - r0 + 1)
        w = int(c1 - c0 + 1)
        if h < 3 or w < 3:
            continue

        t = None
        inner_col = None
        for tt in range(1, int(min(h, w) // 2) + 1):
            inner = g[int(r0 + tt) : int(r1 - tt) + 1, int(c0 + tt) : int(c1 - tt) + 1]
            if inner.size == 0:
                continue
            if bool((inner == int(border)).any()):
                continue
            u = set(map(int, inner.flatten()))
            if len(u) != 1:
                continue

            ok = True
            if not bool((g[int(r0) : int(r0 + tt), int(c0) : int(c1) + 1] == int(border)).all()):
                ok = False
            if not bool((g[int(r1 - tt) + 1 : int(r1) + 1, int(c0) : int(c1) + 1] == int(border)).all()):
                ok = False
            if not bool((g[int(r0) : int(r1) + 1, int(c0) : int(c0 + tt)] == int(border)).all()):
                ok = False
            if not bool((g[int(r0) : int(r1) + 1, int(c1 - tt) + 1 : int(c1) + 1] == int(border)).all()):
                ok = False
            if not ok:
                continue

            t = int(tt)
            inner_col = int(next(iter(u)))
            break

        if t is None or inner_col is None:
            continue

        frames.append(
            {
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
                "t": int(t),
                "inner": int(inner_col),
                "inner_bbox": (int(r0 + t), int(c0 + t), int(r1 - t), int(c1 - t)),
            }
        )
    return frames


def _center(f: dict) -> tuple[int, int]:
    r0, c0, r1, c1 = map(int, f["bbox"])
    return int((r0 + r1) // 2), int((c0 + c1) // 2)


def _aligned(f1: dict, f2: dict) -> tuple[bool, bool]:
    a = f1["inner_bbox"]
    b = f2["inner_bbox"]
    same_row = bool(int(a[0]) == int(b[0]) and int(a[2]) == int(b[2]))
    same_col = bool(int(a[1]) == int(b[1]) and int(a[3]) == int(b[3]))
    return same_row, same_col


def _bridge_cells(fa: dict, fb: dict) -> set[tuple[int, int]]:
    (r0a, c0a, r1a, c1a) = map(int, fa["bbox"])
    (r0b, c0b, r1b, c1b) = map(int, fb["bbox"])
    ia = fa["inner_bbox"]
    ib = fb["inner_bbox"]
    if int(ia[0]) == int(ib[0]) and int(ia[2]) == int(ib[2]):
        rr0, rr1 = int(ia[0]), int(ia[2])
        if int(c1a) < int(c0b):
            cc0, cc1 = int(c1a + 1), int(c0b - 1)
        else:
            cc0, cc1 = int(c1b + 1), int(c0a - 1)
        if int(cc0) > int(cc1):
            return set()
        return {(int(r), int(c)) for r in range(int(rr0), int(rr1) + 1) for c in range(int(cc0), int(cc1) + 1)}
    if int(ia[1]) == int(ib[1]) and int(ia[3]) == int(ib[3]):
        cc0, cc1 = int(ia[1]), int(ia[3])
        if int(r1a) < int(r0b):
            rr0, rr1 = int(r1a + 1), int(r0b - 1)
        else:
            rr0, rr1 = int(r1b + 1), int(r0a - 1)
        if int(rr0) > int(rr1):
            return set()
        return {(int(r), int(c)) for r in range(int(rr0), int(rr1) + 1) for c in range(int(cc0), int(cc1) + 1)}
    return set()


def _draw_frame(g: np.ndarray, *, r0: int, c0: int, outer: int, t: int, border: int, inner_col: int) -> None:
    g[int(r0) : int(r0 + outer), int(c0) : int(c0 + outer)] = int(border)
    g[int(r0 + t) : int(r0 + outer - t), int(c0 + t) : int(c0 + outer - t)] = int(inner_col)


def _most_common_nonbg(g: np.ndarray, bg: int) -> int | None:
    cnt = Counter(map(int, g.flatten()))
    for col, _k in cnt.most_common():
        if int(col) != int(bg):
            return int(col)
    return None


def _find_pair_legend(g: np.ndarray, bg: int, min_row: int) -> tuple[list[int], int]:
    H, W = g.shape
    groups: list[tuple[list[int], int]] = []
    rows: list[int] = []
    cur_c0: int | None = None
    prev_r: int | None = None

    for r in range(int(min_row), int(H)):
        nz = [int(c) for c in range(int(W)) if int(g[int(r), int(c)]) != int(bg)]
        good = len(nz) == 2 and int(nz[1]) == int(nz[0] + 1)
        if good:
            c0 = int(nz[0])
            if rows and prev_r is not None and int(r) == int(prev_r + 1) and cur_c0 == int(c0):
                rows.append(int(r))
            else:
                if len(rows) >= 2 and cur_c0 is not None:
                    groups.append((rows[:], int(cur_c0)))
                rows = [int(r)]
                cur_c0 = int(c0)
            prev_r = int(r)
            continue

        if len(rows) >= 2 and cur_c0 is not None:
            groups.append((rows[:], int(cur_c0)))
        rows = []
        cur_c0 = None
        prev_r = None

    if len(rows) >= 2 and cur_c0 is not None:
        groups.append((rows[:], int(cur_c0)))

    assert groups
    return max(groups, key=lambda item: (len(item[0]), item[0][-1]))


def _find_mapping_rows(g: np.ndarray, bg: int, min_row: int, max_row: int) -> tuple[int, int, list[int]]:
    best: tuple[int, int, list[int]] | None = None
    for r in range(int(min_row), int(max_row)):
        if int(r + 1) >= int(max_row):
            break
        cols0 = [int(c) for c in range(int(g.shape[1])) if int(g[int(r), int(c)]) != int(bg)]
        cols1 = [int(c) for c in range(int(g.shape[1])) if int(g[int(r + 1), int(c)]) != int(bg)]
        if len(cols0) < 4 or cols0 != cols1:
            continue
        if any(int(cols0[i + 1] - cols0[i]) < 2 for i in range(int(len(cols0)) - 1)):
            continue
        best = (int(r), int(r + 1), cols0)
    assert best is not None
    return best


def _parse_control(g: np.ndarray, bg: int, min_row: int) -> tuple[dict[int, int], list[tuple[int, int]]]:
    pair_rows, pair_c0 = _find_pair_legend(g, int(bg), int(min_row))
    map_src_r, map_dst_r, map_cols = _find_mapping_rows(g, int(bg), int(min_row), int(pair_rows[0]))

    mapping: dict[int, int] = {}
    for c in map_cols:
        src = int(g[int(map_src_r), int(c)])
        dst = int(g[int(map_dst_r), int(c)])
        assert src != int(bg) and dst != int(bg)
        mapping[int(src)] = int(dst)
    assert len(mapping) == len(map_cols) and len(mapping) >= 4

    pairs = [(int(g[int(r), int(pair_c0)]), int(g[int(r), int(pair_c0 + 1)])) for r in pair_rows]
    assert len(pairs) >= 2
    return mapping, pairs


def _choose_pair(by: dict[int, list[dict]], start_inner: int, end_inner: int) -> tuple[dict, dict] | None:
    if start_inner not in by or end_inner not in by:
        return None
    best = None
    best_pair = None
    for fa in by[int(start_inner)]:
        for fb in by[int(end_inner)]:
            sr, sc = _aligned(fa, fb)
            al = 1 if (sr or sc) else 0
            r1_, c1_ = _center(fa)
            r2_, c2_ = _center(fb)
            d = abs(int(r1_) - int(r2_)) + abs(int(c1_) - int(c2_))
            cand = (int(al), -int(d))
            if best is None or cand > best:
                best = cand
                best_pair = (fa, fb)
    if best is None or best_pair is None or int(best[0]) != 1:
        return None
    return best_pair


def _draw_bridge(out: np.ndarray, fa: dict, fb: dict, color: int) -> None:
    H, W = out.shape
    (r0a, c0a, r1a, c1a) = map(int, fa["bbox"])
    (r0b, c0b, r1b, c1b) = map(int, fb["bbox"])
    ia = fa["inner_bbox"]
    sr, sc = _aligned(fa, fb)
    if sr:
        rr0, rr1 = int(ia[0]), int(ia[2])
        if int(c1a) < int(c0b):
            cc0, cc1 = int(c1a + 1), int(c0b - 1)
        else:
            cc0, cc1 = int(c1b + 1), int(c0a - 1)
        for r in range(int(rr0), int(rr1) + 1):
            for c in range(int(cc0), int(cc1) + 1):
                if 0 <= r < H and 0 <= c < W:
                    out[int(r), int(c)] = int(color)
        return
    if sc:
        cc0, cc1 = int(ia[1]), int(ia[3])
        if int(r1a) < int(r0b):
            rr0, rr1 = int(r1a + 1), int(r0b - 1)
        else:
            rr0, rr1 = int(r1b + 1), int(r0a - 1)
        for r in range(int(rr0), int(rr1) + 1):
            for c in range(int(cc0), int(cc1) + 1):
                if 0 <= r < H and 0 <= c < W:
                    out[int(r), int(c)] = int(color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect framed squares using a two-stage control system.

    1. A horizontal legend maps control colors to frame-inner colors.
    2. A vertical 2-column legend lists bridge requests row-by-row in control colors.
    3. Each request row is translated through the mapping legend, then the matching
       start/end frames are bridged using the raw start control color.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    cnt = Counter(map(int, g.flatten()))
    border = None
    for col, _k in cnt.most_common():
        if int(col) != int(bg):
            border = int(col)
            break
    assert border is not None

    frames = _detect_frames(g, int(border))
    assert frames

    by: dict[int, list[dict]] = defaultdict(list)
    for f in frames:
        by[int(f["inner"])].append(f)

    max_frame_r = max(int(f["bbox"][2]) for f in frames)
    mapping, ctrl_pairs = _parse_control(g, int(bg), int(max_frame_r + 1))

    out = g.copy()
    for src_ctrl, dst_ctrl in ctrl_pairs:
        assert int(src_ctrl) in mapping and int(dst_ctrl) in mapping
        src_inner = int(mapping[int(src_ctrl)])
        dst_inner = int(mapping[int(dst_ctrl)])
        chosen = _choose_pair(by, int(src_inner), int(dst_inner))
        assert chosen is not None
        _draw_bridge(out, chosen[0], chosen[1], int(src_ctrl))

    return out


def _solve_direct_pair_baseline(grid: np.ndarray) -> np.ndarray:
    """
    Baseline used only for generator-side anti-degeneration checks.

    It decodes the explicit pair legend but intentionally ignores the color-mapping
    legend, matching pair colors directly to frame inner colors. If this produces the
    same output as `main`, then the D1 half of the fusion did not matter visually.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    cnt = Counter(map(int, g.flatten()))
    border = None
    for col, _k in cnt.most_common():
        if int(col) != int(bg):
            border = int(col)
            break
    assert border is not None

    frames = _detect_frames(g, int(border))
    assert frames
    by: dict[int, list[dict]] = defaultdict(list)
    for f in frames:
        by[int(f["inner"])].append(f)

    max_frame_r = max(int(f["bbox"][2]) for f in frames)
    _mapping, ctrl_pairs = _parse_control(g, int(bg), int(max_frame_r + 1))

    out = g.copy()
    for src_ctrl, dst_ctrl in ctrl_pairs:
        chosen = _choose_pair(by, int(src_ctrl), int(dst_ctrl))
        if chosen is None:
            continue
        _draw_bridge(out, chosen[0], chosen[1], int(src_ctrl))
    return out


_DIGITS = [int(i) for i in range(10)]


def _shuffle_pair_slots(
    pair_slots: list[tuple[tuple[int, int], tuple[int, int]]],
    colors_by_slot: dict[tuple[int, int], int],
) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    if len(pair_slots) < 2:
        return pair_slots[:]
    idx = np.arange(int(len(pair_slots)), dtype=int)
    for _ in range(40):
        perm = np.random.permutation(idx)
        shuffled = [pair_slots[int(i)] for i in perm]
        if any(
            int(colors_by_slot[shuffled[i][1]]) != int(colors_by_slot[shuffled[i + 1][0]])
            for i in range(int(len(shuffled)) - 1)
        ):
            return shuffled
    return [pair_slots[int(i)] for i in np.random.permutation(idx)]


def _gen_attempt() -> np.ndarray | None:
    bg = int(np.random.choice(np.array(_DIGITS, dtype=int)))
    border = int(np.random.choice(np.array([d for d in _DIGITS if int(d) != int(bg)], dtype=int)))
    inner_palette = [int(d) for d in _DIGITS if int(d) not in (int(bg), int(border))]
    if len(inner_palette) < 6:
        return None

    t = int(np.random.choice(np.array([1, 1, 2], dtype=int)))
    inner = int(np.random.randint(1, 4))
    outer = int(inner + 2 * t)
    if not (3 <= int(outer) <= 7):
        return None

    R = int(np.random.randint(2, 6))
    C = int(np.random.randint(2, 6))
    max_frames = int(R * C)
    if max_frames < 4:
        return None
    n_frames = int(np.random.randint(4, int(min(7, max_frames, len(inner_palette))) + 1))

    path: list[tuple[int, int]] | None = None
    for _path_try in range(400):
        p = [(0, 0)]
        vis = {(0, 0)}
        ok = True
        for _ in range(int(n_frames) - 1):
            r, c = map(int, p[-1])
            opts = [
                (int(rr), int(cc))
                for rr, cc in neighbors4(int(r), int(c))
                if in_bounds((int(R), int(C)), int(rr), int(cc)) and (int(rr), int(cc)) not in vis
            ]
            if not opts:
                ok = False
                break
            nxt = opts[int(np.random.randint(0, len(opts)))]
            p.append((int(nxt[0]), int(nxt[1])))
            vis.add((int(nxt[0]), int(nxt[1])))
        if not ok:
            continue
        if len({r for r, _c in p}) < 2 or len({c for _r, c in p}) < 2:
            continue
        path = p
        break
    if path is None:
        return None

    row_gap = [int(np.random.randint(1, 6)) for _ in range(int(R) - 1)]
    col_gap = [int(np.random.randint(1, 6)) for _ in range(int(C) - 1)]
    top_margin = int(np.random.randint(1, 4))
    left_margin = int(np.random.randint(1, 4))

    r0s = [int(top_margin)]
    for k in range(int(R) - 1):
        r0s.append(int(r0s[-1] + outer + row_gap[int(k)]))
    c0s = [int(left_margin)]
    for k in range(int(C) - 1):
        c0s.append(int(c0s[-1] + outer + col_gap[int(k)]))

    slot_to_frame: dict[tuple[int, int], dict] = {}
    frames: list[dict] = []
    for rr, cc in path:
        r0 = int(r0s[int(rr)])
        c0 = int(c0s[int(cc)])
        r1 = int(r0 + outer - 1)
        c1 = int(c0 + outer - 1)
        f = {
            "bbox": (int(r0), int(c0), int(r1), int(c1)),
            "inner_bbox": (int(r0 + t), int(c0 + t), int(r1 - t), int(c1 - t)),
            "slot": (int(rr), int(cc)),
        }
        frames.append(f)
        slot_to_frame[(int(rr), int(cc))] = f

    max_r1 = max(int(f["bbox"][2]) for f in frames)
    max_c1 = max(int(f["bbox"][3]) for f in frames)

    pair_slots = list(zip(path, path[1:]))
    n_pairs = int(len(pair_slots))
    if n_pairs < 2:
        return None

    map_gap = int(np.random.randint(2, 5))
    pair_gap = int(np.random.randint(2, 5))
    map_src_r = int(max_r1 + map_gap)
    map_dst_r = int(map_src_r + 1)
    pair_r0 = int(map_dst_r + pair_gap)
    pair_c0 = int(np.random.randint(1, 4))
    bottom_margin = 1
    right_margin = int(np.random.randint(1, 5))

    chosen = np.random.choice(np.array(inner_palette, dtype=int), size=int(n_frames), replace=False)
    colors_by_slot = {slot: int(chosen[i]) for i, slot in enumerate(path)}
    used_inner_colors = [int(colors_by_slot[slot]) for slot in path]
    if len(set(used_inner_colors)) != int(n_frames):
        return None

    legend_order = [int(x) for x in np.random.permutation(np.array(used_inner_colors, dtype=int))]
    ctrl_to_inner = {
        int(legend_order[i]): int(legend_order[(i + 1) % len(legend_order)])
        for i in range(int(len(legend_order)))
    }
    if any(int(k) == int(v) for k, v in ctrl_to_inner.items()):
        return None
    inner_to_ctrl = {int(dst): int(src) for src, dst in ctrl_to_inner.items()}
    if len(inner_to_ctrl) != len(ctrl_to_inner):
        return None

    map_cols = [int(1 + 2 * i) for i in range(int(len(legend_order)))]
    H = int(pair_r0 + n_pairs + bottom_margin)
    W = int(max(max_c1 + 1 + right_margin, map_cols[-1] + 1 + right_margin, pair_c0 + 2 + right_margin))
    if not (7 <= int(H) <= 30 and 7 <= int(W) <= 30):
        return None

    out = np.full((int(H), int(W)), int(bg), dtype=int)
    for f in frames:
        rr, cc = map(int, f["slot"])
        r0, c0, _r1, _c1 = map(int, f["bbox"])
        _draw_frame(
            out,
            r0=int(r0),
            c0=int(c0),
            outer=int(outer),
            t=int(t),
            border=int(border),
            inner_col=int(colors_by_slot[(rr, cc)]),
        )

    bridge_cells_all: set[tuple[int, int]] = set()
    ok = True
    for a_slot, b_slot in pair_slots:
        fa = slot_to_frame[(int(a_slot[0]), int(a_slot[1]))]
        fb = slot_to_frame[(int(b_slot[0]), int(b_slot[1]))]
        cells = _bridge_cells(fa, fb)
        if not cells or cells & bridge_cells_all:
            ok = False
            break
        for r, c in cells:
            if not (0 <= int(r) < int(H) and 0 <= int(c) < int(W)):
                ok = False
                break
            if int(out[int(r), int(c)]) != int(bg):
                ok = False
                break
        if not ok:
            break
        paint = int(inner_to_ctrl[int(colors_by_slot[(int(a_slot[0]), int(a_slot[1]))])])
        for r, c in cells:
            out[int(r), int(c)] = int(paint)
        bridge_cells_all |= set(cells)
    if not ok:
        return None

    for src, cc in zip(legend_order, map_cols):
        out[int(map_src_r), int(cc)] = int(src)
        out[int(map_dst_r), int(cc)] = int(ctrl_to_inner[int(src)])

    legend_pairs = _shuffle_pair_slots(pair_slots, colors_by_slot)
    for i, (a_slot, b_slot) in enumerate(legend_pairs):
        a_inner = int(colors_by_slot[(int(a_slot[0]), int(a_slot[1]))])
        b_inner = int(colors_by_slot[(int(b_slot[0]), int(b_slot[1]))])
        out[int(pair_r0 + i), int(pair_c0)] = int(inner_to_ctrl[int(a_inner)])
        out[int(pair_r0 + i), int(pair_c0 + 1)] = int(inner_to_ctrl[int(b_inner)])

    inp = out.copy()
    for r, c in bridge_cells_all:
        inp[int(r), int(c)] = int(bg)

    if int(get_bg(inp)) != int(bg):
        return None
    if int(_most_common_nonbg(inp, int(bg)) or -1) != int(border):
        return None

    try:
        pred = main(inp)
    except Exception:
        return None
    if pred.shape != out.shape or not bool(np.array_equal(pred, out)):
        return None

    try:
        direct_pred = _solve_direct_pair_baseline(inp)
    except Exception:
        return None
    if bool(np.array_equal(direct_pred, out)):
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(500):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("Failed to generate valid input after 500 attempts")
