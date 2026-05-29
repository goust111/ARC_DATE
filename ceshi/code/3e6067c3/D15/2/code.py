# Variant of 3e6067c3
# Dimension: D15 — 连接路径的几何属性 (transform)
# Change: 桥梁是直线矩形条带，填充两个对齐方块之间的全部间隙空间... → 阶梯形桥梁

from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np

from arc_utils import get_bg, in_bounds, neighbors4, label_mask


# Design analysis for the D15 variant:
#
# a) Concretization + behavior trace
# - The original task places equal-sized framed squares and uses the bottom control row
#   to specify a visitation order. The old bridge filled an entire straight gap, which
#   only works when the two relevant squares share the same inner row band or column band.
# - In this variant, the next square can still be "to the left/right" or "above/below"
#   the current one, but its inner window is intentionally shifted. The output bridge is
#   therefore a monotone staircase corridor made from small axis-aligned blocks instead of
#   one straight rectangle.
# - Example: suppose two squares are left/right neighbors. Originally their inner rows were
#   identical, so the solver drew rows r..r+k-1 across every gap column. Here the right
#   square may start one row lower, so the bridge exits the left square, runs horizontally,
#   drops by one row inside the gap, runs horizontally again, and reaches the right square.
#   The control colors and bridge colors stay the same; only the gap geometry changes.
#
# b) Complexity calibration
# - If offsets were optional, many samples would degenerate back into straight bridges.
#   The generator therefore enforces a non-zero orthogonal offset on every path edge.
# - Repeated inner colors were useful in the original task because row/column alignment
#   disambiguated them. After removing exact alignment, that cue disappears. To keep D15
#   as the main novelty instead of introducing a second hidden tie-break rule, visited
#   frame colors are generated as globally unique.
# - The staircase is restricted to monotone left/right or up/down routing. There is only
#   one bridge style per sample, which keeps the task inferable from a few I/O pairs.
#
# c) Implicit assumptions audited against the original solver
# - Original `aligned()` assumed inner bboxes matched exactly in row or column; that is no
#   longer valid and would reject every intended bridge.
# - Original `draw_bridge()` filled a rectangular strip defined only by outer-gap bounds;
#   with shifted targets, that would either miss the target or overpaint the wrong rows.
# - Original candidate ranking preferred exact band alignment. The new ranking instead
#   prefers pairs whose outer boxes are side-adjacent (horizontal or vertical separation
#   with overlap on the orthogonal axis) and whose staircase corridor is valid.
# - The frame detector, control-row detector, and border/background inference remain valid.
#
# d) Placement legality
# - The generator still uses a logical lattice path, but each visited slot gets a small
#   per-frame jitter. Horizontal path edges require different row jitters; vertical edges
#   require different column jitters.
# - Lattice row/column spacing is padded so that even under worst-case jitter, frames do
#   not overlap and every bridge has enough empty space for a k-thick staircase corridor.
# - The generator constructs the output first, erases bridges to form the input, and then
#   validates `main(input) == output`.
#
# e) Visual observability
# - Every bridge edge has a visible orthogonal offset, so the new geometry is not an edge
#   case. A human sees the same color-to-color sequence as before, but the bridges now bend
#   through the gap in a staircase pattern.
#
# f) Cleanliness
# - Bridges are monotone and only occupy the gap between side-adjacent squares. The
#   generator rejects placements where bridges overlap each other or collide with frames.
#
# g) Color inferability
# - Bridge color still equals the source square/control color, exactly like the base task.
#   No ordering or routing depends on the numeric value of a color id.


def _interval_overlap(a0: int, a1: int, b0: int, b1: int) -> bool:
    return max(int(a0), int(b0)) <= min(int(a1), int(b1))


def _center(f: dict) -> tuple[int, int]:
    r0, c0, r1, c1 = map(int, f["bbox"])
    return int((r0 + r1) // 2), int((c0 + c1) // 2)


def _parse_scene(grid: np.ndarray) -> tuple[np.ndarray, int, int, list[dict], dict[int, list[dict]], list[int]]:
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

    cnt = Counter(map(int, g.flatten()))
    border = None
    for col, _k in cnt.most_common():
        if int(col) != int(bg):
            border = int(col)
            break
    assert border is not None

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

    assert frames

    by: dict[int, list[dict]] = defaultdict(list)
    for f in frames:
        by[int(f["inner"])].append(f)

    max_frame_r = max(int(f["bbox"][2]) for f in frames)
    ctrl_r = None
    for r in range(int(H - 1), int(max_frame_r), -1):
        if int((g[int(r)] != int(bg)).sum()) >= 2:
            ctrl_r = int(r)
            break
    assert ctrl_r is not None

    ctrl = [int(g[int(ctrl_r), int(c)]) for c in range(int(W)) if int(g[int(ctrl_r), int(c)]) != int(bg)]
    assert len(ctrl) >= 2

    return g, int(bg), int(border), frames, by, ctrl


def _inner_size(f: dict) -> int:
    ia = f["inner_bbox"]
    return int(ia[2] - ia[0] + 1)


def _side_relation(fa: dict, fb: dict) -> str | None:
    r0a, c0a, r1a, c1a = map(int, fa["bbox"])
    r0b, c0b, r1b, c1b = map(int, fb["bbox"])

    if int(c1a) < int(c0b) and _interval_overlap(int(r0a), int(r1a), int(r0b), int(r1b)):
        return "lr"
    if int(c1b) < int(c0a) and _interval_overlap(int(r0a), int(r1a), int(r0b), int(r1b)):
        return "rl"
    if int(r1a) < int(r0b) and _interval_overlap(int(c0a), int(c1a), int(c0b), int(c1b)):
        return "ud"
    if int(r1b) < int(r0a) and _interval_overlap(int(c0a), int(c1a), int(c0b), int(c1b)):
        return "du"
    return None


def _stair_positions_horizontal(start_r: int, start_c: int, end_r: int, end_c: int) -> list[tuple[int, int]]:
    assert int(start_c) <= int(end_c)
    positions = [(int(start_r), int(start_c))]
    r = int(start_r)
    c = int(start_c)
    delta_r = int(end_r - start_r)
    v_abs = abs(int(delta_r))
    v_sign = 0 if int(delta_r) == 0 else (1 if int(delta_r) > 0 else -1)
    h = int(end_c - start_c)
    v_done = 0
    for step in range(1, int(h) + 1):
        c += 1
        positions.append((int(r), int(c)))
        target_v = (int(step) * int(v_abs)) // int(h) if int(h) > 0 else int(v_abs)
        while int(v_done) < int(target_v):
            r += int(v_sign)
            v_done += 1
            positions.append((int(r), int(c)))
    while int(v_done) < int(v_abs):
        r += int(v_sign)
        v_done += 1
        positions.append((int(r), int(c)))
    assert (int(r), int(c)) == (int(end_r), int(end_c))
    return positions


def _stair_positions_vertical(start_r: int, start_c: int, end_r: int, end_c: int) -> list[tuple[int, int]]:
    assert int(start_r) <= int(end_r)
    positions = [(int(start_r), int(start_c))]
    r = int(start_r)
    c = int(start_c)
    delta_c = int(end_c - start_c)
    h_abs = abs(int(delta_c))
    h_sign = 0 if int(delta_c) == 0 else (1 if int(delta_c) > 0 else -1)
    v = int(end_r - start_r)
    h_done = 0
    for step in range(1, int(v) + 1):
        r += 1
        positions.append((int(r), int(c)))
        target_h = (int(step) * int(h_abs)) // int(v) if int(v) > 0 else int(h_abs)
        while int(h_done) < int(target_h):
            c += int(h_sign)
            h_done += 1
            positions.append((int(r), int(c)))
    while int(h_done) < int(h_abs):
        c += int(h_sign)
        h_done += 1
        positions.append((int(r), int(c)))
    assert (int(r), int(c)) == (int(end_r), int(end_c))
    return positions


def _bridge_cells(fa: dict, fb: dict) -> set[tuple[int, int]]:
    relation = _side_relation(fa, fb)
    if relation is None:
        return set()

    k = int(min(_inner_size(fa), _inner_size(fb)))
    if k <= 0:
        return set()

    r0a, c0a, r1a, c1a = map(int, fa["bbox"])
    r0b, c0b, r1b, c1b = map(int, fb["bbox"])
    ia = fa["inner_bbox"]
    ib = fb["inner_bbox"]

    if relation == "rl":
        return _bridge_cells(fb, fa)
    if relation == "du":
        return _bridge_cells(fb, fa)

    blocks: list[tuple[int, int]]
    if relation == "lr":
        start_r = int(ia[0])
        start_c = int(c1a + 1)
        end_r = int(ib[0])
        end_c = int(c0b - k)
        if int(start_c) > int(end_c):
            return set()
        blocks = _stair_positions_horizontal(int(start_r), int(start_c), int(end_r), int(end_c))
    else:
        start_r = int(r1a + 1)
        start_c = int(ia[1])
        end_r = int(r0b - k)
        end_c = int(ib[1])
        if int(start_r) > int(end_r):
            return set()
        blocks = _stair_positions_vertical(int(start_r), int(start_c), int(end_r), int(end_c))

    cells: set[tuple[int, int]] = set()
    for br, bc in blocks:
        for rr in range(int(br), int(br + k)):
            for cc in range(int(bc), int(bc + k)):
                cells.add((int(rr), int(cc)))
    return cells


def _candidate_score(cur: dict, cand: dict) -> tuple[int, int]:
    cells = _bridge_cells(cur, cand)
    ok = 1 if cells else 0
    r1, c1 = _center(cur)
    r2, c2 = _center(cand)
    d = abs(int(r1) - int(r2)) + abs(int(c1) - int(c2))
    return int(ok), -int(d)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect framed squares via staircase bridges following a control-pixel sequence.

    The control row still specifies the color order. For each consecutive pair, the next
    square is selected by color and side-adjacency/proximity, then the gap is filled by a
    monotone staircase bridge colored with the source color.
    """
    g, bg, _border, frames, by, ctrl = _parse_scene(grid)
    H, W = g.shape

    start_color = int(ctrl[0])
    assert start_color in by
    start_cands = by[start_color]
    if len(start_cands) == 1:
        cur = start_cands[0]
    else:
        nxt_color = int(ctrl[1])
        nxt_cands = by.get(nxt_color, [])
        best = None
        bestf = None
        for f in start_cands:
            if nxt_cands:
                cand_score = max((_candidate_score(f, g2) for g2 in nxt_cands), default=(0, -10**9))
            else:
                cand_score = (0, -10**9)
            if best is None or cand_score > best:
                best = cand_score
                bestf = f
        assert bestf is not None
        cur = bestf

    out = g.copy()

    for i in range(int(len(ctrl)) - 1):
        a = int(ctrl[i])
        b = int(ctrl[i + 1])
        assert int(cur["inner"]) == int(a)
        assert b in by
        cands = by[b]
        best = None
        bestf = None
        best_cells: set[tuple[int, int]] | None = None
        for f in cands:
            cells = _bridge_cells(cur, f)
            r1, c1 = _center(cur)
            r2, c2 = _center(f)
            d = abs(int(r1) - int(r2)) + abs(int(c1) - int(c2))
            cand = (1 if cells else 0, -int(d))
            if best is None or cand > best:
                best = cand
                bestf = f
                best_cells = cells
        assert bestf is not None
        assert best_cells
        for r, c in best_cells:
            if 0 <= int(r) < H and 0 <= int(c) < W:
                out[int(r), int(c)] = int(a)
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


_DIGITS = [int(i) for i in range(10)]


def _sample_path(R: int, C: int, n_frames: int) -> list[tuple[int, int]] | None:
    for _path_try in range(500):
        p = [(0, 0)]
        vis = {(0, 0)}
        ok = True
        for _ in range(int(n_frames) - 1):
            r, c = p[-1]
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
        return p
    return None


def _sample_jitters(path: list[tuple[int, int]], jitter_max: int) -> dict[tuple[int, int], tuple[int, int]] | None:
    jitters: dict[tuple[int, int], tuple[int, int]] = {}
    first = path[0]
    jitters[(int(first[0]), int(first[1]))] = (
        int(np.random.randint(0, int(jitter_max) + 1)),
        int(np.random.randint(0, int(jitter_max) + 1)),
    )
    for prev, cur in zip(path, path[1:]):
        pr, pc = map(int, prev)
        cr, cc = map(int, cur)
        prev_jr, prev_jc = jitters[(int(pr), int(pc))]
        if int(pr) == int(cr):
            jr_choices = [int(x) for x in range(int(jitter_max) + 1) if int(x) != int(prev_jr)]
            if not jr_choices:
                return None
            jr = int(np.random.choice(np.array(jr_choices, dtype=int)))
            jc = int(np.random.randint(0, int(jitter_max) + 1))
        else:
            jc_choices = [int(x) for x in range(int(jitter_max) + 1) if int(x) != int(prev_jc)]
            if not jc_choices:
                return None
            jr = int(np.random.randint(0, int(jitter_max) + 1))
            jc = int(np.random.choice(np.array(jc_choices, dtype=int)))
        jitters[(int(cr), int(cc))] = (int(jr), int(jc))
    return jitters


def _gen_attempt() -> np.ndarray | None:
    bg = int(np.random.choice(np.array(_DIGITS, dtype=int)))
    border = int(np.random.choice(np.array([d for d in _DIGITS if int(d) != int(bg)], dtype=int)))
    inner_palette = [int(d) for d in _DIGITS if int(d) not in (int(bg), int(border))]
    if len(inner_palette) < 4:
        return None

    t = int(np.random.choice(np.array([1, 1, 2], dtype=int)))
    inner = int(np.random.choice(np.array([1, 2], dtype=int)))
    outer = int(inner + 2 * t)
    k = int(inner)
    if not (3 <= int(outer) <= 6):
        return None

    R = int(np.random.randint(2, 5))
    C = int(np.random.randint(2, 5))
    max_frames = int(R * C)
    if max_frames < 4:
        return None
    n_frames = int(np.random.randint(4, int(min(8, max_frames)) + 1))

    path = _sample_path(int(R), int(C), int(n_frames))
    if path is None:
        return None

    jitter_max = 2
    jitters = _sample_jitters(path, int(jitter_max))
    if jitters is None:
        return None

    row_gap = [int(np.random.randint(int(k), int(k) + 4)) for _ in range(int(R) - 1)]
    col_gap = [int(np.random.randint(int(k), int(k) + 4)) for _ in range(int(C) - 1)]
    top_margin = int(np.random.randint(1, 4))
    left_margin = int(np.random.randint(1, 4))

    r0s = [int(top_margin)]
    for gap in row_gap:
        r0s.append(int(r0s[-1] + outer + int(gap) + jitter_max))
    c0s = [int(left_margin)]
    for gap in col_gap:
        c0s.append(int(c0s[-1] + outer + int(gap) + jitter_max))

    colors = list(np.random.permutation(np.array(inner_palette, dtype=int))[: int(n_frames)])
    if len(colors) != len(path):
        return None

    slot_to_frame: dict[tuple[int, int], dict] = {}
    frames: list[dict] = []
    colors_by_slot: dict[tuple[int, int], int] = {}
    for idx, (rr, cc) in enumerate(path):
        jr, jc = jitters[(int(rr), int(cc))]
        r0 = int(r0s[int(rr)] + int(jr))
        c0 = int(c0s[int(cc)] + int(jc))
        r1 = int(r0 + outer - 1)
        c1 = int(c0 + outer - 1)
        f = {
            "bbox": (int(r0), int(c0), int(r1), int(c1)),
            "inner_bbox": (int(r0 + t), int(c0 + t), int(r1 - t), int(c1 - t)),
            "slot": (int(rr), int(cc)),
            "t": int(t),
            "inner": int(colors[int(idx)]),
        }
        frames.append(f)
        slot_to_frame[(int(rr), int(cc))] = f
        colors_by_slot[(int(rr), int(cc))] = int(colors[int(idx)])

    max_r1 = max(int(f["bbox"][2]) for f in frames)
    max_c1 = max(int(f["bbox"][3]) for f in frames)
    ctrl_gap = int(np.random.randint(2, 6))
    ctrl_r = int(max_r1 + ctrl_gap)
    bottom_margin = 1
    right_margin = int(max(2, 2 * int(n_frames) + 2))
    H = int(ctrl_r + 1 + bottom_margin)
    W = int(max(max_c1 + 1 + right_margin, 2 * int(n_frames) + 3))
    if not (7 <= int(H) <= 30 and 7 <= int(W) <= 30):
        return None

    out = np.full((int(H), int(W)), int(bg), dtype=int)
    for f in frames:
        r0, c0, _r1, _c1 = map(int, f["bbox"])
        _draw_frame(
            out,
            r0=int(r0),
            c0=int(c0),
            outer=int(outer),
            t=int(t),
            border=int(border),
            inner_col=int(f["inner"]),
        )

    bridge_cells_all: set[tuple[int, int]] = set()
    ok = True
    stair_offsets: list[int] = []
    for a_slot, b_slot in zip(path, path[1:]):
        fa = slot_to_frame[(int(a_slot[0]), int(a_slot[1]))]
        fb = slot_to_frame[(int(b_slot[0]), int(b_slot[1]))]
        rel = _side_relation(fa, fb)
        if rel is None:
            ok = False
            break
        ia = fa["inner_bbox"]
        ib = fb["inner_bbox"]
        if rel in ("lr", "rl"):
            offset = abs(int(ia[0]) - int(ib[0]))
        else:
            offset = abs(int(ia[1]) - int(ib[1]))
        stair_offsets.append(int(offset))
        if int(offset) <= 0:
            ok = False
            break

        cells = _bridge_cells(fa, fb)
        if not cells or (cells & bridge_cells_all):
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
        paint = int(fa["inner"])
        for r, c in cells:
            out[int(r), int(c)] = int(paint)
        bridge_cells_all |= set(cells)

    if not ok or not stair_offsets or min(stair_offsets) <= 0:
        return None

    ctrl_colors = [int(colors_by_slot[(int(r), int(c))]) for r, c in path]
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
    for _ in range(600):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("Failed to generate valid input after 600 attempts")
