# Variant of 3e6067c3
# Dimension: D13 — 桥梁与已有结构的交互 (meta)
# Change: 桥梁仅在背景色区域绘制，不覆盖任何已有的方块或先前桥梁... → 桥梁交叉时后者覆盖前者

from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np

from arc_utils import get_bg, in_bounds, label_mask

# Design analysis for the D13 variant
# a) Concrete behavior:
#    The original task already draws bridges in control-row order, but its generator only connected
#    adjacent lattice slots, so different bridges never shared pixels. D13 needs an actual ordered
#    overwrite. The generator therefore places some consecutive frames several lattice cells apart:
#    e.g. a horizontal bridge from slot (1,0) to (1,4), then later a vertical bridge from (0,2) to
#    (4,2). The crossing area is a 2x2 background patch inside an empty lattice cell. Under the old
#    rule the first bridge color would stay there; under D13 the later bridge repaints that patch.
# b) Complexity calibration:
#    Keep the framed-square detection unchanged and make inner colors globally unique. This removes
#    color-matching ambiguity so the only new reasoning load is draw order at overlaps. Each sample
#    has 5 path frames and at least one forced overlap; template symmetries vary the visual layout.
# c) Old assumptions that mattered:
#    `draw_bridge()` writes unconditionally. That was harmless before because the old generator
#    guaranteed bridge cells were all background. In D13 that same unconditional write becomes the
#    intended overwrite mechanism, but it is now unsafe unless the generator separately guarantees
#    that bridge cells never hit frame or control pixels.
# d) Placement legality:
#    Long bridges are allowed only across empty lattice rows/columns. The generator rejects any scene
#    where a bridge cell touches the frame/control scaffold, goes out of bounds, or fails to align.
# e) Visual observability:
#    Every generated scene contains a real overlap region and the overlap uses distinct source colors,
#    so the later repaint is visible from the output alone. The overlap is 2x2, not a single pixel.
# f) Visual cleanliness:
#    Only bridge-on-bridge overlap is permitted. Frames remain intact, the control row stays below
#    the scene, and there is at most one forced crossing, so the grid remains easy to parse by eye.
# g) Color inference:
#    No hidden numeric ordering is introduced. Bridge colors still come directly from the source
#    frame colors, exactly as in the original task.
# h) Variant validation target:
#    A scene is accepted only if at least one cell is painted by multiple bridges and the D13 output
#    differs from the old "paint only background cells" semantics on that same input.
# i) New solver-side assumptions and generator guarantees:
#    `main()` still assumes every consecutive control-color pair resolves to aligned frames. The
#    generator guarantees that by constructing the frame sequence directly from an aligned template.

_DIGITS = [int(i) for i in range(10)]
_OUTER = 4
_THICKNESS = 1
_INNER = _OUTER - 2 * _THICKNESS
_TEMPLATES: list[list[tuple[int, int]]] = [
    [(1, 0), (1, 4), (0, 4), (0, 2), (4, 2)],
    [(0, 2), (4, 2), (4, 0), (2, 0), (2, 4)],
]


def _most_common_nonbg(g: np.ndarray, bg: int) -> int | None:
    cnt = Counter(map(int, g.flatten()))
    for col, _k in cnt.most_common():
        if int(col) != int(bg):
            return int(col)
    return None


def _draw_frame(g: np.ndarray, *, r0: int, c0: int, outer: int, t: int, border: int, inner_col: int) -> None:
    g[int(r0) : int(r0 + outer), int(c0) : int(c0 + outer)] = int(border)
    g[int(r0 + t) : int(r0 + outer - t), int(c0 + t) : int(c0 + outer - t)] = int(inner_col)


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


def _paint_bridge(out: np.ndarray, fa: dict, fb: dict, color: int) -> None:
    # D13 semantics: bridges are painted strictly in control-sequence order, so later bridges
    # intentionally overwrite earlier bridge pixels when their cell sets overlap.
    for r, c in _bridge_cells(fa, fb):
        out[int(r), int(c)] = int(color)


def _old_semantics_paint(out: np.ndarray, fa: dict, fb: dict, color: int, bg: int) -> None:
    for r, c in _bridge_cells(fa, fb):
        if int(out[int(r), int(c)]) == int(bg):
            out[int(r), int(c)] = int(color)


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
    ctrl_r = None
    for r in range(int(H - 1), int(max_frame_r), -1):
        if int((g[int(r)] != int(bg)).sum()) >= 2:
            ctrl_r = int(r)
            break
    assert not (ctrl_r is None)

    ctrl = [int(g[int(ctrl_r), int(c)]) for c in range(int(W)) if int(g[int(ctrl_r), int(c)]) != int(bg)]
    assert not (len(ctrl) < 2)

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
                sr, sc = _aligned(f, g2)
                al = 1 if (sr or sc) else 0
                (r1_, c1_) = _center(f)
                (r2_, c2_) = _center(g2)
                d = abs(int(r1_) - int(r2_)) + abs(int(c1_) - int(c2_))
                cand = (int(al), -int(d))
                if best is None or cand > best:
                    best = cand
                    bestf = f
        cur = bestf

    out = g.copy()
    for i in range(int(len(ctrl)) - 1):
        a = int(ctrl[i])
        b = int(ctrl[i + 1])
        assert int(cur["inner"]) == int(a)
        if b not in by:
            continue
        cands = by[b]
        best = None
        bestf = None
        for f in cands:
            sr, sc = _aligned(cur, f)
            al = 1 if (sr or sc) else 0
            (r1_, c1_) = _center(cur)
            (r2_, c2_) = _center(f)
            d = abs(int(r1_) - int(r2_)) + abs(int(c1_) - int(c2_))
            cand = (int(al), -int(d))
            if best is None or cand > best:
                best = cand
                bestf = f
        nxt = bestf
        _paint_bridge(out, cur, nxt, int(a))
        cur = nxt

    return out


def _transform_slots(slots: list[tuple[int, int]], mode: int, size: int = 5) -> list[tuple[int, int]]:
    max_idx = int(size - 1)
    out: list[tuple[int, int]] = []
    for r, c in slots:
        r = int(r)
        c = int(c)
        if mode == 0:
            nr, nc = r, c
        elif mode == 1:
            nr, nc = r, int(max_idx - c)
        elif mode == 2:
            nr, nc = int(max_idx - r), c
        elif mode == 3:
            nr, nc = int(max_idx - r), int(max_idx - c)
        elif mode == 4:
            nr, nc = c, r
        elif mode == 5:
            nr, nc = c, int(max_idx - r)
        elif mode == 6:
            nr, nc = int(max_idx - c), r
        elif mode == 7:
            nr, nc = int(max_idx - c), int(max_idx - r)
        else:
            raise ValueError(mode)
        out.append((int(nr), int(nc)))
    return out


def _gen_attempt() -> np.ndarray | None:
    bg = int(np.random.choice(np.array(_DIGITS, dtype=int)))
    border = int(np.random.choice(np.array([d for d in _DIGITS if int(d) != int(bg)], dtype=int)))
    inner_palette = [int(d) for d in _DIGITS if int(d) not in (int(bg), int(border))]

    template = _TEMPLATES[int(np.random.randint(0, len(_TEMPLATES)))]
    slots = _transform_slots(template, int(np.random.randint(0, 8)))
    n_frames = len(slots)
    if len(inner_palette) < int(n_frames):
        return None
    colors = [int(x) for x in np.random.choice(np.array(inner_palette, dtype=int), size=int(n_frames), replace=False)]

    lattice_rows = 5
    lattice_cols = 5
    top_margin = int(np.random.randint(0, 2))
    left_margin = int(np.random.randint(0, 2))
    right_margin = int(np.random.randint(0, 2))
    row_gap = [int(np.random.randint(1, 3)) for _ in range(int(lattice_rows) - 1)]
    col_gap = [int(np.random.randint(1, 3)) for _ in range(int(lattice_cols) - 1)]
    ctrl_gap = 1
    bottom_margin = 1

    r0s = [int(top_margin)]
    for gap in row_gap:
        r0s.append(int(r0s[-1] + _OUTER + int(gap)))
    c0s = [int(left_margin)]
    for gap in col_gap:
        c0s.append(int(c0s[-1] + _OUTER + int(gap)))

    frames: list[dict] = []
    for idx, (rr, cc) in enumerate(slots):
        rr = int(rr)
        cc = int(cc)
        r0 = int(r0s[int(rr)])
        c0 = int(c0s[int(cc)])
        r1 = int(r0 + _OUTER - 1)
        c1 = int(c0 + _OUTER - 1)
        frames.append(
            {
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
                "inner_bbox": (int(r0 + _THICKNESS), int(c0 + _THICKNESS), int(r1 - _THICKNESS), int(c1 - _THICKNESS)),
                "inner": int(colors[int(idx)]),
                "slot": (int(rr), int(cc)),
            }
        )

    max_r1 = max(int(f["bbox"][2]) for f in frames)
    max_c1 = max(int(f["bbox"][3]) for f in frames)
    ctrl_r = int(max_r1 + ctrl_gap + 1)
    H = int(ctrl_r + 1 + bottom_margin)
    W = int(max(max_c1 + 1 + right_margin, 1 + 2 * (n_frames - 1) + 1))
    if not (7 <= int(H) <= 30 and 7 <= int(W) <= 30):
        return None

    scaffold = np.full((int(H), int(W)), int(bg), dtype=int)
    for f in frames:
        r0, c0, _r1, _c1 = map(int, f["bbox"])
        _draw_frame(
            scaffold,
            r0=int(r0),
            c0=int(c0),
            outer=int(_OUTER),
            t=int(_THICKNESS),
            border=int(border),
            inner_col=int(f["inner"]),
        )

    ctrl_cols = [int(1 + 2 * i) for i in range(int(n_frames))]
    if int(ctrl_cols[-1]) >= int(W):
        return None
    for col, color in zip(ctrl_cols, colors):
        scaffold[int(ctrl_r), int(col)] = int(color)

    out = scaffold.copy()
    old_semantics = scaffold.copy()
    paint_counts: Counter[tuple[int, int]] = Counter()
    for idx in range(int(n_frames) - 1):
        fa = frames[int(idx)]
        fb = frames[int(idx + 1)]
        cells = _bridge_cells(fa, fb)
        if not cells:
            return None
        if not (_aligned(fa, fb)[0] or _aligned(fa, fb)[1]):
            return None
        for r, c in cells:
            if not in_bounds((int(H), int(W)), int(r), int(c)):
                return None
            if int(scaffold[int(r), int(c)]) != int(bg):
                return None
        _paint_bridge(out, fa, fb, int(colors[int(idx)]))
        _old_semantics_paint(old_semantics, fa, fb, int(colors[int(idx)]), int(bg))
        for cell in cells:
            paint_counts[(int(cell[0]), int(cell[1]))] += 1

    overlap_pixels = sum(1 for count in paint_counts.values() if int(count) > 1)
    if overlap_pixels <= 0:
        return None
    if bool(np.array_equal(out, old_semantics)):
        return None

    inp = scaffold.copy()
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
    for _ in range(300):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("Failed to generate valid D13 input after 300 attempts")
