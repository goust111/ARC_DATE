# Variant of 3e6067c3
# Dimension: D1 — 匹配条件：颜色匹配方式 (match_cond)
# Change: 控制像素颜色与方块内部填充颜色相同 (s.colors == d.inner_squares.colors)... → 互补色匹配
#
# Design analysis
# - Concrete variant: the framed-square lattice and bridge geometry stay unchanged. The only
#   semantic change is that the bottom control row no longer names squares directly by their
#   inner fill color. Instead, a 2-row legend is inserted below the square field:
#   row A shows "control colors", row B directly underneath shows the corresponding square
#   inner colors. Columns align 1-to-1, so each vertical pair defines source -> destination.
#   The bottom-most signal row is still the control sequence. The solver must read the legend,
#   translate each control color into an inner-square color, then reuse the original alignment
#   and nearest-candidate logic. Bridges remain painted with the control color, which makes the
#   D1 change visible in the output instead of remaining a hidden lookup detail.
# - Small execution trace: if the legend contains 3 above 7 and 6 above 4, then a control row
#   segment [3, 6] means "start at the 7-filled frame, then go to the 4-filled frame", and the
#   bridge between them is painted 3. In the original task the same segment would have meant
#   "start at the 3-filled frame, then go to the 6-filled frame", so both matching and bridge
#   coloring semantics would collapse to the same color. The variant separates those roles.
# - Complexity calibration: the generator keeps the original row/column-band ambiguity so the
#   legend is not the only reasoning step. It also forces at least 3 distinct frame colors,
#   which prevents the variant from degenerating into a trivial 2-color swap that would be
#   visually weak across only a few examples.
# - Implicit assumptions in the original main():
#   1. `ctrl` colors are used as keys into `by[inner]`; this breaks immediately once control
#      colors differ from frame inner colors. The solver therefore now translates `ctrl_src`
#      through the legend into `ctrl_inner` before every lookup.
#   2. `assert cur["inner"] == a` previously compared the current frame inner color with the raw
#      control color. In the variant it must compare with the translated target color `a_inner`.
#   3. `draw_bridge(..., a)` previously used the same color for frame matching and for painting.
#      After D1 changes these are distinct; painting must use the raw control color `a_src`.
#   4. Control-row detection via "bottom-most row below the squares with >=2 non-bg pixels" is
#      still safe as long as all extra legend rows remain above the control row. The generator
#      guarantees exactly three informative rows below the squares: legend source, legend target,
#      control.
# - Placement legality: frames are still placed on a lattice path with at least one background
#   cell between frame borders; bridges occupy only the gaps between aligned inner bands. The
#   legend and control rows live strictly below the frame field, so they never overlap with
#   frames or bridges.
# - Visual observability / cleanliness: the legend uses aligned vertical color pairs at columns
#   1,3,5,..., and the control row keeps the same sparse rhythm as the original task. That makes
#   the new clue visually local and readable without introducing extra shapes or dense clutter.
# - Color-rule inferability: the mapping is explicit in the legend, uses only colors already
#   present in the frame/control system, and is a no-fixed-point permutation over the actually
#   used frame colors. Humans can infer "control color points to square color" directly from
#   the examples; no hidden numeric ordering is required.

from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np

from arc_utils import get_bg, in_bounds, label_mask, neighbors4


def _extract_frames(g: np.ndarray, border: int) -> list[dict]:
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


def _signal_rows(g: np.ndarray, bg: int, max_frame_r: int) -> list[int]:
    rows: list[int] = []
    for r in range(int(max_frame_r) + 1, int(g.shape[0])):
        if int((g[int(r)] != int(bg)).sum()) >= 2:
            rows.append(int(r))
    return rows


def _parse_legend_and_control(g: np.ndarray, bg: int, max_frame_r: int) -> tuple[dict[int, int], list[int], list[int]]:
    rows = _signal_rows(g, int(bg), int(max_frame_r))
    assert len(rows) >= 3
    legend_src_r, legend_dst_r, ctrl_r = map(int, rows[-3:])

    mapping: dict[int, int] = {}
    for c in range(int(g.shape[1])):
        src = int(g[int(legend_src_r), int(c)])
        dst = int(g[int(legend_dst_r), int(c)])
        if src == int(bg) and dst == int(bg):
            continue
        assert src != int(bg) and dst != int(bg)
        mapping[int(src)] = int(dst)

    assert len(mapping) >= 2
    ctrl_src = [int(g[int(ctrl_r), int(c)]) for c in range(int(g.shape[1])) if int(g[int(ctrl_r), int(c)]) != int(bg)]
    assert len(ctrl_src) >= 2
    ctrl_inner = [int(mapping[int(col)]) for col in ctrl_src]
    return mapping, ctrl_src, ctrl_inner


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect framed squares via colored bridges following a control-pixel sequence.

    Variant rule:
    - The control row names colors indirectly.
    - A 2-row legend below the square field maps control colors -> target inner colors.
    - Squares are matched using the translated target inner colors, while bridges are still
      painted using the raw control colors.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

    cnt = Counter(map(int, g.flatten()))
    border = None
    for col, _k in cnt.most_common():
        if int(col) != int(bg):
            border = int(col)
            break
    assert not (border is None)

    frames = _extract_frames(g, int(border))
    assert not (not frames)

    by: dict[int, list[dict]] = defaultdict(list)
    for f in frames:
        by[int(f["inner"])].append(f)

    max_frame_r = max(int(f["bbox"][2]) for f in frames)
    _mapping, ctrl_src, ctrl_inner = _parse_legend_and_control(g, int(bg), int(max_frame_r))

    def center(f: dict) -> tuple[int, int]:
        r0, c0, r1, c1 = map(int, f["bbox"])
        return int((r0 + r1) // 2), int((c0 + c1) // 2)

    def aligned(f1: dict, f2: dict) -> tuple[bool, bool]:
        a = f1["inner_bbox"]
        b = f2["inner_bbox"]
        same_row = bool(int(a[0]) == int(b[0]) and int(a[2]) == int(b[2]))
        same_col = bool(int(a[1]) == int(b[1]) and int(a[3]) == int(b[3]))
        return same_row, same_col

    start_inner = int(ctrl_inner[0])
    assert not (start_inner not in by)
    start_cands = by[start_inner]
    if len(start_cands) == 1:
        cur = start_cands[0]
    else:
        nxt_inner = int(ctrl_inner[1])
        nxt_cands = by.get(nxt_inner, [])
        best = None
        bestf = None
        for f in start_cands:
            for g2 in (nxt_cands if nxt_cands else [f]):
                sr, sc = aligned(f, g2)
                al = 1 if (sr or sc) else 0
                r1_, c1_ = center(f)
                r2_, c2_ = center(g2)
                d = abs(int(r1_) - int(r2_)) + abs(int(c1_) - int(c2_))
                cand = (int(al), -int(d))
                if best is None or cand > best:
                    best = cand
                    bestf = f
        cur = bestf

    out = g.copy()

    def draw_bridge(fa: dict, fb: dict, color: int) -> None:
        (r0a, c0a, r1a, c1a) = map(int, fa["bbox"])
        (r0b, c0b, r1b, c1b) = map(int, fb["bbox"])
        ia = fa["inner_bbox"]
        sr, sc = aligned(fa, fb)
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

    for i in range(int(len(ctrl_src)) - 1):
        a_src = int(ctrl_src[i])
        a_inner = int(ctrl_inner[i])
        b_inner = int(ctrl_inner[i + 1])
        assert int(cur["inner"]) == int(a_inner)
        if b_inner not in by:
            continue
        cands = by[b_inner]
        best = None
        bestf = None
        for f in cands:
            sr, sc = aligned(cur, f)
            al = 1 if (sr or sc) else 0
            r1_, c1_ = center(cur)
            r2_, c2_ = center(f)
            d = abs(int(r1_) - int(r2_)) + abs(int(c1_) - int(c2_))
            cand = (int(al), -int(d))
            if best is None or cand > best:
                best = cand
                bestf = f
        nxt = bestf
        draw_bridge(cur, nxt, int(a_src))
        cur = nxt

    return out


def _most_common_nonbg(g: np.ndarray, bg: int) -> int | None:
    cnt = Counter(map(int, g.flatten()))
    for col, _k in cnt.most_common():
        if int(col) != int(bg):
            return int(col)
    return None


def _draw_frame(g: np.ndarray, *, r0: int, c0: int, outer: int, t: int, border: int, inner_col: int) -> None:
    g[int(r0) : int(r0 + outer), int(c0) : int(c0 + outer)] = int(border)
    g[int(r0 + t) : int(r0 + outer - t), int(c0 + t) : int(c0 + outer - t)] = int(inner_col)


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


_DIGITS = [int(i) for i in range(10)]


def _gen_attempt() -> np.ndarray | None:
    bg = int(np.random.choice(np.array(_DIGITS, dtype=int)))
    border = int(np.random.choice(np.array([d for d in _DIGITS if int(d) != int(bg)], dtype=int)))
    inner_palette = [int(d) for d in _DIGITS if int(d) not in (int(bg), int(border))]
    if len(inner_palette) < 4:
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
    n_frames = int(np.random.randint(4, int(min(12, max_frames)) + 1))

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

    legend_gap = int(np.random.randint(2, 5))
    ctrl_gap = int(np.random.randint(2, 5))
    legend_src_r = int(max_r1 + legend_gap)
    legend_dst_r = int(legend_src_r + 1)
    ctrl_r = int(legend_dst_r + ctrl_gap)
    bottom_margin = 1
    right_margin = int(np.random.randint(1, 5))
    H = int(ctrl_r + 1 + bottom_margin)
    W = int(max_c1 + 1 + right_margin)

    if not (7 <= int(H) <= 30 and 7 <= int(W) <= 30):
        return None

    used_in_row: dict[int, set[int]] = defaultdict(set)
    used_in_col: dict[int, set[int]] = defaultdict(set)

    start_color = int(np.random.choice(np.array(inner_palette, dtype=int)))
    colors_by_slot: dict[tuple[int, int], int] = {(0, 0): int(start_color)}
    used_in_row[0].add(int(start_color))
    used_in_col[0].add(int(start_color))

    for rr, cc in path[1:]:
        rr = int(rr)
        cc = int(cc)
        forbid = set(used_in_row[int(rr)]) | set(used_in_col[int(cc)]) | {int(start_color)}
        choices = [int(x) for x in inner_palette if int(x) not in forbid]
        if not choices:
            return None
        col = int(np.random.choice(np.array(choices, dtype=int)))
        colors_by_slot[(int(rr), int(cc))] = int(col)
        used_in_row[int(rr)].add(int(col))
        used_in_col[int(cc)].add(int(col))

    if len(colors_by_slot) != len(path):
        return None

    used_inner_colors = sorted({int(v) for v in colors_by_slot.values()})
    if len(used_inner_colors) < 3:
        return None

    legend_order = [int(x) for x in np.random.permutation(np.array(used_inner_colors, dtype=int))]
    src_to_inner = {
        int(legend_order[i]): int(legend_order[(i + 1) % len(legend_order)])
        for i in range(int(len(legend_order)))
    }
    if any(int(k) == int(v) for k, v in src_to_inner.items()):
        return None
    inner_to_src = {int(dst): int(src) for src, dst in src_to_inner.items()}

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
    for a_slot, b_slot in zip(path, path[1:]):
        fa = slot_to_frame[(int(a_slot[0]), int(a_slot[1]))]
        fb = slot_to_frame[(int(b_slot[0]), int(b_slot[1]))]
        cells = _bridge_cells(fa, fb)
        if not cells:
            ok = False
            break
        if cells & bridge_cells_all:
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
        paint = int(inner_to_src[int(colors_by_slot[(int(a_slot[0]), int(a_slot[1]))])])
        for r, c in cells:
            out[int(r), int(c)] = int(paint)
        bridge_cells_all |= set(cells)

    if not ok:
        return None

    legend_cols = [int(1 + 2 * i) for i in range(int(len(legend_order)))]
    if int(legend_cols[-1]) >= int(W):
        return None
    for src, cc in zip(legend_order, legend_cols):
        out[int(legend_src_r), int(cc)] = int(src)
        out[int(legend_dst_r), int(cc)] = int(src_to_inner[int(src)])

    ctrl_colors = [int(inner_to_src[int(colors_by_slot[(int(r), int(c))])]) for r, c in path]
    if len(ctrl_colors) < 2:
        return None
    ctrl_cols = [int(1 + 2 * i) for i in range(int(len(ctrl_colors)))]
    if int(ctrl_cols[-1]) >= int(W):
        return None
    for col, cc in zip(ctrl_colors, ctrl_cols):
        out[int(ctrl_r), int(cc)] = int(col)

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
    if pred.shape != out.shape:
        return None
    if not bool(np.array_equal(pred, out)):
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(500):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("Failed to generate valid input after 500 attempts")
