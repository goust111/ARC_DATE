# Variant of 3e6067c3
# Dimension: D5 — 变换：桥梁绘制规则 (transform)
# Change: 用起始方块的内部颜色在两个方块之间绘制矩形桥梁，桥梁宽度等于内部填充区域的宽度... → 桥梁使用专用第三色

from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np

from arc_utils import get_bg, in_bounds, neighbors4, label_mask

# Design analysis for the D5 variant
# a) Concrete behavior:
#    The main framed-square scene is unchanged. The input now has a two-layer control
#    area below the squares:
#    - upper control row: the original visitation-order colors at columns 1,3,5,...
#    - lower control row: one bridge color per consecutive pair, placed at columns
#      2,4,6,... so each lower pixel sits visually between the two upper pixels that
#      define the edge it recolors.
#    Example: if the upper row is [2, 5, 7] and the lower row is [4, 6], the solver
#    still visits square 2 -> square 5 -> square 7, but draws the first bridge in 4
#    and the second bridge in 6. The original logic would have painted those bridges
#    in 2 and 5 respectively.
# b) Complexity calibration:
#    The variant keeps the original row/column-band matching and nearest-candidate
#    disambiguation, so object identity and path decoding do not become easier. The
#    new bridge row adds one extra visible control layer without introducing many
#    extra degrees of freedom: one color per edge, read left-to-right.
# c) Implicit-assumption audit:
#    The original main() assumed "control color = bridge color" in exactly one place:
#    draw_bridge(cur, nxt, a). That assumption no longer holds. The frame detection,
#    alignment test, start-frame selection, and next-frame selection still depend only
#    on square geometry and the upper control row, so they remain valid. The control
#    row detection must change because there are now two informative rows, and the
#    solver must not accidentally treat the lower bridge row as the visitation order.
# d) Placement legality:
#    The generator still samples a self-avoiding lattice path, uses _bridge_cells() to
#    guarantee non-overlapping rectangular bridges, and keeps the control area below
#    the lowest frame with a positive gap. The extra bridge-color row is adjacent to
#    the order row and never intersects the bridge drawing area.
# e) Visual observability:
#    The lower control pixels sit between the upper order pixels, so a human sees a
#    node-edge-node rhythm directly in the control area. Because every bridge color is
#    forced to differ from both endpoint square colors, the changed D5 behavior is
#    visible in every generated edge, not just in rare cases.
# f) Visual cleanliness:
#    The control rows are sparse and separated from the frame lattice. No additional
#    markings are overlaid on the squares themselves, so the scene remains legible.
# g) Color inference:
#    Bridge colors come from a small 3-color palette reused across edges within a
#    sample, but each edge chooses the member that is not equal to either endpoint.
#    This keeps the color rule structured and learnable from a few examples.
# h) New solver assumptions after modification:
#    - Below the square region there are exactly two non-empty control rows; the lower
#      one is immediately below the upper one.
#    - The upper row has one more colored pixel than the lower row.
#    - Each lower control pixel lies horizontally between the corresponding pair of
#      upper control pixels.
#    The generator enforces all three conditions explicitly.


def _extract_control_layers(g: np.ndarray, bg: int, max_frame_r: int) -> tuple[list[int], list[int]]:
    """Return (visit_colors, bridge_colors) from the two-row control area."""
    H, W = g.shape
    rows = [int(r) for r in range(int(max_frame_r) + 1, int(H)) if bool((g[int(r)] != int(bg)).any())]
    assert len(rows) >= 2
    seq_r = int(rows[-2])
    bridge_r = int(rows[-1])
    assert int(bridge_r) == int(seq_r + 1)

    seq_cols = [int(c) for c in range(int(W)) if int(g[int(seq_r), int(c)]) != int(bg)]
    bridge_cols = [int(c) for c in range(int(W)) if int(g[int(bridge_r), int(c)]) != int(bg)]
    assert len(seq_cols) >= 2
    assert len(bridge_cols) == len(seq_cols) - 1
    for i, bc in enumerate(bridge_cols):
        assert int(seq_cols[i]) < int(bc) < int(seq_cols[i + 1])

    ctrl = [int(g[int(seq_r), int(c)]) for c in seq_cols]
    bridge_ctrl = [int(g[int(bridge_r), int(c)]) for c in bridge_cols]
    return ctrl, bridge_ctrl


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect framed squares via colored bridges following a two-layer control area.

    Input structure:
    - bg = background. Border color forms equal-sized framed squares with inner fill colors.
    - Upper control row below the squares: colored pixels read left-to-right encode visitation order.
    - Lower control row directly under it: one pixel per consecutive pair, encoding bridge colors.

    Rule:
    - Start at the square matching the first upper-row color c0.
    - For each consecutive pair (ci, ci+1), find the ci+1-colored square aligned in the same row or
      column band, preferring nearest.
    - Draw the bridge between them using the i-th lower-row bridge color, not using ci or ci+1.
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

    assert not (not frames)

    by: dict[int, list[dict]] = defaultdict(list)
    for f in frames:
        by[int(f["inner"])].append(f)

    max_frame_r = max(int(f["bbox"][2]) for f in frames)
    ctrl, bridge_ctrl = _extract_control_layers(g, int(bg), int(max_frame_r))

    def center(f: dict) -> tuple[int, int]:
        r0, c0, r1, c1 = map(int, f["bbox"])
        return int((r0 + r1) // 2), int((c0 + c1) // 2)

    def aligned(f1: dict, f2: dict) -> tuple[bool, bool]:
        a = f1["inner_bbox"]
        b = f2["inner_bbox"]
        same_row = bool(int(a[0]) == int(b[0]) and int(a[2]) == int(b[2]))
        same_col = bool(int(a[1]) == int(b[1]) and int(a[3]) == int(b[3]))
        return same_row, same_col

    start_color = int(ctrl[0])
    assert not (start_color not in by)
    start_cands = by[start_color]
    if len(start_cands) == 1:
        cur = start_cands[0]
    else:
        nxt_color = int(ctrl[1])
        nxt_cands = by.get(nxt_color, [])
        best = None
        bestf = None
        for f in start_cands:
            for g2 in (nxt_cands if nxt_cands else [f]):
                sr, sc = aligned(f, g2)
                al = 1 if (sr or sc) else 0
                (r1_, c1_) = center(f)
                (r2_, c2_) = center(g2)
                d = abs(int(r1_) - int(r2_)) + abs(int(c1_) - int(c2_))
                cand = (int(al), -int(d))
                if best is None or cand > best:
                    best = cand
                    bestf = f
        cur = bestf
    assert cur is not None

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

    for i in range(int(len(ctrl)) - 1):
        a = int(ctrl[i])
        b = int(ctrl[i + 1])
        bridge_color = int(bridge_ctrl[i])
        assert int(cur["inner"]) == int(a)
        if b not in by:
            continue
        cands = by[b]
        best = None
        bestf = None
        for f in cands:
            sr, sc = aligned(cur, f)
            al = 1 if (sr or sc) else 0
            (r1_, c1_) = center(cur)
            (r2_, c2_) = center(f)
            d = abs(int(r1_) - int(r2_)) + abs(int(c1_) - int(c2_))
            cand = (int(al), -int(d))
            if best is None or cand > best:
                best = cand
                bestf = f
        nxt = bestf
        assert nxt is not None
        draw_bridge(cur, nxt, int(bridge_color))
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


def _choose_bridge_colors(ctrl_colors: list[int], inner_palette: list[int]) -> list[int] | None:
    palette = list(map(int, np.random.choice(np.array(inner_palette, dtype=int), size=3, replace=False)))
    bridge_colors: list[int] = []
    use_count: Counter[int] = Counter()
    for a, b in zip(ctrl_colors, ctrl_colors[1:]):
        choices = [int(col) for col in palette if int(col) not in (int(a), int(b))]
        if not choices:
            return None
        min_count = min(use_count.get(int(col), 0) for col in choices)
        balanced = [int(col) for col in choices if int(use_count.get(int(col), 0)) == int(min_count)]
        pick = int(balanced[int(np.random.randint(0, len(balanced)))])
        bridge_colors.append(int(pick))
        use_count[int(pick)] += 1
    if len(set(bridge_colors)) < 2:
        return None
    if len(bridge_colors) >= 4 and max(Counter(bridge_colors).values()) < 2:
        return None
    for col, a, b in zip(bridge_colors, ctrl_colors, ctrl_colors[1:]):
        if int(col) in (int(a), int(b)):
            return None
    return bridge_colors


_DIGITS = [int(i) for i in range(10)]


def _gen_attempt() -> np.ndarray | None:
    bg = int(np.random.choice(np.array(_DIGITS, dtype=int)))
    border = int(np.random.choice(np.array([d for d in _DIGITS if int(d) != int(bg)], dtype=int)))
    inner_palette = [int(d) for d in _DIGITS if int(d) not in (int(bg), int(border))]
    if len(inner_palette) < 3:
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
            cur = p[-1]
            r, c = map(int, cur)
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

    ctrl_gap = int(np.random.randint(2, 6))
    seq_r = int(max_r1 + ctrl_gap)
    bridge_r = int(seq_r + 1)
    bottom_margin = 1
    right_margin = int(np.random.randint(1, 5))
    H = int(bridge_r + 1 + bottom_margin)
    W = int(max_c1 + 1 + right_margin)

    if not (8 <= int(H) <= 30 and 7 <= int(W) <= 30):
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

    ctrl_colors = [int(colors_by_slot[(int(r), int(c))]) for r, c in path]
    if len(ctrl_colors) < 2:
        return None
    bridge_colors = _choose_bridge_colors(ctrl_colors, inner_palette)
    if bridge_colors is None:
        return None

    out = np.full((int(H), int(W)), int(bg), dtype=int)
    for f in frames:
        rr, cc = map(int, f["slot"])
        r0, c0, _r1, _c1 = map(int, f["bbox"])
        _draw_frame(out, r0=int(r0), c0=int(c0), outer=int(outer), t=int(t), border=int(border), inner_col=int(colors_by_slot[(rr, cc)]))

    bridge_cells_all: set[tuple[int, int]] = set()
    ok = True
    for idx, (a_slot, b_slot) in enumerate(zip(path, path[1:])):
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
        paint = int(bridge_colors[int(idx)])
        if int(paint) in (int(ctrl_colors[int(idx)]), int(ctrl_colors[int(idx) + 1])):
            ok = False
            break
        for r, c in cells:
            out[int(r), int(c)] = int(paint)
        bridge_cells_all |= set(cells)

    if not ok:
        return None

    ctrl_cols = [int(1 + 2 * i) for i in range(int(len(ctrl_colors)))]
    bridge_cols = [int(2 + 2 * i) for i in range(int(len(bridge_colors)))]
    if int(ctrl_cols[-1]) >= int(W):
        return None
    if bridge_cols and int(bridge_cols[-1]) >= int(W):
        return None
    for col, cc in zip(ctrl_colors, ctrl_cols):
        out[int(seq_r), int(cc)] = int(col)
    for col, cc in zip(bridge_colors, bridge_cols):
        out[int(bridge_r), int(cc)] = int(col)

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
