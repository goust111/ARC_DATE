# Variant of 3e6067c3
# Dimension: D1 — 匹配条件：颜色匹配方式 (match_cond)
# Change: 控制像素颜色与方块内部填充颜色相同 (s.colors == d.inner_squares.colors)... → 方块大小匹配
#
# Design analysis
# a) Concrete behavior:
#    The original task used equal-sized framed squares with different inner colors, and
#    single control pixels below them. This variant keeps the framed-square lattice and
#    bridge-drawing behavior, but all inner fills now share one color. Squares differ by
#    frame thickness, so their visible inner-square side lengths differ (1/3/5 cells for
#    outer size 7). The control row is replaced by separated horizontal color bars. A bar's
#    length matches the target square's inner side length; the bar color is only used as
#    the bridge paint color. Example: a length-5 bar means "pick the large-inner square",
#    then when connecting it to the next length-coded target, paint the bridge using that
#    bar's color. Compared with the original logic, the matching key changes from
#    `inner_color` to `measured_size`, while path order, alignment preference, and nearest
#    tie-breaking stay the same.
# b) Complexity calibration:
#    Using three visible size classes avoids degeneration into a trivial one-to-one coding.
#    Repeats are allowed and encouraged, but the generator constrains row/column usage so
#    that size collisions do not make the next step ambiguous under the solver's
#    align-first/nearest heuristic. Control bars have lengths 1/3/5, which is visually
#    distinct while still fitting inside ARC's <=30 width when path length is capped.
# c) Hidden-assumption audit of the original main():
#    - Grouping frames by `inner` color no longer holds, so grouping must switch to a
#      size feature extracted from each detected frame.
#    - The control row was read as isolated non-bg pixels; now it must be parsed as runs.
#    - `assert cur["inner"] == a` depended on color identity and must become a size check.
#    - `aligned()` previously required identical inner-bbox bands, which only holds when all
#      frames have equal size. With variable inner sizes, we instead align by shared center
#      line and bridge across the overlap of the two inner bands.
#    - `draw_bridge()` formerly used the full source band width/height; that would overpaint
#      outside a smaller target's opening. The variant uses the intersection band, i.e. the
#      common visible opening across the pair.
# d) Placement legality:
#    The generator keeps a lattice of non-overlapping 7x7 outer boxes with >=1 bg gap. It
#    computes bridges on the intended path, rejects overlaps/crossings, and verifies the
#    final input with `main(inp) == out`.
# e) Visual observability:
#    From a few examples, a human can see that all inner colors are identical, so color can
#    no longer explain which square is chosen. The only reliable cue is bar length versus
#    square opening size/thickness. Bar colors still vary, but they track bridge paint, not
#    object identity.
# f) Visual cleanliness:
#    Control bars live on a dedicated bottom row with bg gaps, and bridges only occupy the
#    corridor between neighboring lattice squares. This keeps frames, bridges, and controls
#    visually separable.
# g) Color inference:
#    No hidden color remapping is used. Colors only identify borders, the common inner fill,
#    and per-step bridge paint. Matching is purely by visible geometric size.
# 9) New main() assumptions and generator guarantees:
#    - Assumption: control items are monochrome runs separated by bg cells.
#      Guarantee: generator places one bar per step with a bg gap.
#    - Assumption: at least one frame exists for each control length.
#      Guarantee: control lengths are derived from the actual path frames.
#    - Assumption: when multiple frames share a size, the intended next frame is the best
#      aligned/nearest candidate.
#      Guarantee: generator enforces row/column size uniqueness and validates via oracle.
#    - Assumption: horizontally/vertically adjacent path squares have overlapping inner bands.
#      Guarantee: all outer boxes have the same size 7 and use centered nested frames.

from __future__ import annotations

from collections import Counter, defaultdict
import numpy as np

from arc_utils import get_bg, in_bounds, neighbors4, label_mask


OUTER = 7
SIZE_BY_T = {1: 5, 2: 3, 3: 1}
SIZE_VALUES = tuple(sorted(SIZE_BY_T.values()))
_DIGITS = [int(i) for i in range(10)]


def _parse_control_runs(row: np.ndarray, bg: int) -> list[dict]:
    runs: list[dict] = []
    c = 0
    W = int(row.shape[0])
    while c < W:
        if int(row[int(c)]) == int(bg):
            c += 1
            continue
        col = int(row[int(c)])
        c0 = int(c)
        while c < W and int(row[int(c)]) == int(col):
            c += 1
        runs.append({"color": int(col), "size": int(c - c0)})
    return runs


def _frame_center(f: dict) -> tuple[int, int]:
    r0, c0, r1, c1 = map(int, f["bbox"])
    return int((r0 + r1) // 2), int((c0 + c1) // 2)


def _aligned(f1: dict, f2: dict) -> tuple[bool, bool]:
    a = f1["inner_bbox"]
    b = f2["inner_bbox"]
    same_row = bool(
        int(a[0]) + int(a[2]) == int(b[0]) + int(b[2])
        and max(int(a[0]), int(b[0])) <= min(int(a[2]), int(b[2]))
    )
    same_col = bool(
        int(a[1]) + int(a[3]) == int(b[1]) + int(b[3])
        and max(int(a[1]), int(b[1])) <= min(int(a[3]), int(b[3]))
    )
    return same_row, same_col


def _bridge_cells(fa: dict, fb: dict) -> set[tuple[int, int]]:
    (r0a, c0a, r1a, c1a) = map(int, fa["bbox"])
    (r0b, c0b, r1b, c1b) = map(int, fb["bbox"])
    ia = fa["inner_bbox"]
    ib = fb["inner_bbox"]
    sr, sc = _aligned(fa, fb)
    if sr:
        rr0 = max(int(ia[0]), int(ib[0]))
        rr1 = min(int(ia[2]), int(ib[2]))
        if int(c1a) < int(c0b):
            cc0, cc1 = int(c1a + 1), int(c0b - 1)
        else:
            cc0, cc1 = int(c1b + 1), int(c0a - 1)
        if int(rr0) > int(rr1) or int(cc0) > int(cc1):
            return set()
        return {(int(r), int(c)) for r in range(int(rr0), int(rr1) + 1) for c in range(int(cc0), int(cc1) + 1)}
    if sc:
        cc0 = max(int(ia[1]), int(ib[1]))
        cc1 = min(int(ia[3]), int(ib[3]))
        if int(r1a) < int(r0b):
            rr0, rr1 = int(r1a + 1), int(r0b - 1)
        else:
            rr0, rr1 = int(r1b + 1), int(r0a - 1)
        if int(rr0) > int(rr1) or int(cc0) > int(cc1):
            return set()
        return {(int(r), int(c)) for r in range(int(rr0), int(rr1) + 1) for c in range(int(cc0), int(cc1) + 1)}
    return set()


def main(grid: np.ndarray) -> np.ndarray:
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

        inner_bbox = (int(r0 + t), int(c0 + t), int(r1 - t), int(c1 - t))
        size = int(inner_bbox[2] - inner_bbox[0] + 1)
        frames.append(
            {
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
                "t": int(t),
                "inner": int(inner_col),
                "size": int(size),
                "inner_bbox": inner_bbox,
            }
        )

    assert not (not frames)

    by_size: dict[int, list[dict]] = defaultdict(list)
    for f in frames:
        by_size[int(f["size"])].append(f)

    max_frame_r = max(int(f["bbox"][2]) for f in frames)
    ctrl_r = None
    for r in range(int(H - 1), int(max_frame_r), -1):
        if int((g[int(r)] != int(bg)).sum()) >= 2:
            ctrl_r = int(r)
            break
    assert not (ctrl_r is None)

    ctrl = _parse_control_runs(g[int(ctrl_r)], int(bg))
    assert not (len(ctrl) < 2)

    start_size = int(ctrl[0]["size"])
    assert not (start_size not in by_size)
    start_cands = by_size[start_size]
    if len(start_cands) == 1:
        cur = start_cands[0]
    else:
        nxt_size = int(ctrl[1]["size"])
        nxt_cands = by_size.get(nxt_size, [])
        best = None
        bestf = None
        for f in start_cands:
            for g2 in (nxt_cands if nxt_cands else [f]):
                sr, sc = _aligned(f, g2)
                al = 1 if (sr or sc) else 0
                (r1_, c1_) = _frame_center(f)
                (r2_, c2_) = _frame_center(g2)
                d = abs(int(r1_) - int(r2_)) + abs(int(c1_) - int(c2_))
                cand = (int(al), -int(d))
                if best is None or cand > best:
                    best = cand
                    bestf = f
        assert not (bestf is None)
        cur = bestf

    out = g.copy()

    def draw_bridge(fa: dict, fb: dict, color: int) -> None:
        cells = _bridge_cells(fa, fb)
        for r, c in cells:
            if 0 <= int(r) < H and 0 <= int(c) < W:
                out[int(r), int(c)] = int(color)

    for i in range(int(len(ctrl)) - 1):
        a = ctrl[int(i)]
        b = ctrl[int(i + 1)]
        assert int(cur["size"]) == int(a["size"])
        cands = by_size.get(int(b["size"]), [])
        assert not (not cands)
        best = None
        bestf = None
        for f in cands:
            sr, sc = _aligned(cur, f)
            al = 1 if (sr or sc) else 0
            (r1_, c1_) = _frame_center(cur)
            (r2_, c2_) = _frame_center(f)
            d = abs(int(r1_) - int(r2_)) + abs(int(c1_) - int(c2_))
            cand = (int(al), -int(d))
            if best is None or cand > best:
                best = cand
                bestf = f
        assert not (bestf is None)
        draw_bridge(cur, bestf, int(a["color"]))
        cur = bestf

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


def _gen_attempt() -> np.ndarray | None:
    bg = int(np.random.choice(np.array(_DIGITS, dtype=int)))
    border = int(np.random.choice(np.array([d for d in _DIGITS if int(d) != int(bg)], dtype=int)))
    inner_choices = [int(d) for d in _DIGITS if int(d) not in (int(bg), int(border))]
    if len(inner_choices) < 2:
        return None
    inner_col = int(np.random.choice(np.array(inner_choices, dtype=int)))
    ctrl_palette = [int(d) for d in _DIGITS if int(d) not in (int(bg), int(border), int(inner_col))]
    if len(ctrl_palette) < 5:
        return None

    R = int(np.random.randint(2, 4))
    C = int(np.random.randint(2, 4))
    max_frames = int(R * C)
    if max_frames < 4:
        return None
    n_frames = int(np.random.randint(4, int(min(5, max_frames)) + 1))

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

    row_gap = [int(np.random.randint(1, 4)) for _ in range(int(R) - 1)]
    col_gap = [int(np.random.randint(1, 4)) for _ in range(int(C) - 1)]
    top_margin = int(np.random.randint(1, 3))
    left_margin = int(np.random.randint(1, 3))

    r0s = [int(top_margin)]
    for k in range(int(R) - 1):
        r0s.append(int(r0s[-1] + OUTER + row_gap[int(k)]))
    c0s = [int(left_margin)]
    for k in range(int(C) - 1):
        c0s.append(int(c0s[-1] + OUTER + col_gap[int(k)]))

    slot_to_frame: dict[tuple[int, int], dict] = {}
    frames: list[dict] = []
    for rr, cc in path:
        r0 = int(r0s[int(rr)])
        c0 = int(c0s[int(cc)])
        r1 = int(r0 + OUTER - 1)
        c1 = int(c0 + OUTER - 1)
        f = {
            "bbox": (int(r0), int(c0), int(r1), int(c1)),
            "slot": (int(rr), int(cc)),
        }
        frames.append(f)
        slot_to_frame[(int(rr), int(cc))] = f

    max_r1 = max(int(f["bbox"][2]) for f in frames)
    max_c1 = max(int(f["bbox"][3]) for f in frames)

    size_by_slot: dict[tuple[int, int], int] = {}
    t_by_slot: dict[tuple[int, int], int] = {}
    used_in_row: dict[int, set[int]] = defaultdict(set)
    used_in_col: dict[int, set[int]] = defaultdict(set)
    prev_size = None
    for rr, cc in path:
        rr = int(rr)
        cc = int(cc)
        choices = [
            int(sz)
            for sz in SIZE_VALUES
            if int(sz) not in used_in_row[int(rr)]
            and int(sz) not in used_in_col[int(cc)]
            and int(sz) != int(prev_size if prev_size is not None else -1)
        ]
        if not choices:
            return None
        size = int(np.random.choice(np.array(choices, dtype=int)))
        size_by_slot[(int(rr), int(cc))] = int(size)
        t_by_slot[(int(rr), int(cc))] = int(next(t for t, s in SIZE_BY_T.items() if int(s) == int(size)))
        used_in_row[int(rr)].add(int(size))
        used_in_col[int(cc)].add(int(size))
        prev_size = int(size)

    counts = Counter(size_by_slot.values())
    if len(counts) < 2:
        return None
    if max(int(v) for v in counts.values()) < 2:
        return None

    ctrl_colors = [int(x) for x in np.random.choice(np.array(ctrl_palette, dtype=int), size=int(n_frames), replace=False)]
    ctrl_sizes = [int(size_by_slot[(int(r), int(c))]) for r, c in path]
    ctrl_spans: list[tuple[int, int, int]] = []
    cur_c = 1
    for col, size in zip(ctrl_colors, ctrl_sizes):
        ctrl_spans.append((int(cur_c), int(cur_c + size - 1), int(col)))
        cur_c += int(size + 1)
    ctrl_end = int(ctrl_spans[-1][1])

    ctrl_gap = int(np.random.randint(2, 5))
    ctrl_r = int(max_r1 + ctrl_gap)
    bottom_margin = 1
    right_margin = int(np.random.randint(1, 3))
    H = int(ctrl_r + 1 + bottom_margin)
    W = int(max(int(max_c1 + 1 + right_margin), int(ctrl_end + 1 + right_margin)))
    if not (7 <= int(H) <= 30 and 7 <= int(W) <= 30):
        return None

    out = np.full((int(H), int(W)), int(bg), dtype=int)
    for f in frames:
        rr, cc = map(int, f["slot"])
        r0, c0, r1, c1 = map(int, f["bbox"])
        t = int(t_by_slot[(int(rr), int(cc))])
        _draw_frame(out, r0=int(r0), c0=int(c0), outer=int(OUTER), t=int(t), border=int(border), inner_col=int(inner_col))
        f["t"] = int(t)
        f["inner"] = int(inner_col)
        f["size"] = int(size_by_slot[(int(rr), int(cc))])
        f["inner_bbox"] = (int(r0 + t), int(c0 + t), int(r1 - t), int(c1 - t))

    bridge_cells_all: set[tuple[int, int]] = set()
    ok = True
    for i, (a_slot, b_slot) in enumerate(zip(path, path[1:])):
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
        paint = int(ctrl_colors[int(i)])
        for r, c in cells:
            out[int(r), int(c)] = int(paint)
        bridge_cells_all |= set(cells)
    if not ok:
        return None

    for c0, c1, col in ctrl_spans:
        out[int(ctrl_r), int(c0) : int(c1) + 1] = int(col)

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
