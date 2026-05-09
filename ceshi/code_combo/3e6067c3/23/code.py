# Combo variant of 3e6067c3 (index 23)
# combo_spec: D1/0+D3/1+D5/0
# Name: 边框匹配+左列控制+终点色桥梁

from __future__ import annotations

from collections import defaultdict

import numpy as np

from arc_utils import get_bg, in_bounds, label_mask, neighbors4


# Design analysis for the combo variant
# =====================================
# a) Concrete behavior and trace
# - Original task:
#   1. Detect framed squares under the assumption that all squares share one border color.
#   2. Read a control sequence from a bottom row, left-to-right.
#   3. Match each control color to a square's inner fill color.
#   4. Draw each bridge segment using the source square's color.
# - Combo task:
#   1. Each square is identified by its border color, so frame detection must enumerate every non-background color.
#   2. The control sequence sits in a dedicated left column, read top-to-bottom.
#   3. Each control pixel matches a square by border color, not by inner fill.
#   4. The bridge from A to B is painted with B's border color, not A's.
# - Small trace:
#   Suppose three squares share the same inner color 6, but their borders are 2, 5, 7. The left control column is [2, 5, 7].
#   Original logic would fail visually because 2/5/7 are not inner fills and because it expects a bottom row.
#   Combo logic reads the left column, picks border-2 -> border-5 -> border-7, and paints the bridges with colors 5 then 7.
#
# b) Complexity calibration
# - The generator uses one repeated inner color across all frames, so the border-vs-inner distinction is visible.
# - Border colors are unique along the path, so the control column has a clear one-to-one visual reference.
# - The left control strip is separated from the lattice by background columns, so the vertical encoding is explicit.
# - Path length is at least 4, which gives several bridge segments and repeated evidence for destination-colored painting.
#
# c) Implicit-assumption audit
# - Original `most common non-bg == border` no longer holds, because borders are multi-colored and the repeated inner color may be most common.
# - Original `label_mask(g == border)` no longer holds, because there is no single border color.
# - Original control-row scan below the frame field no longer holds; we must scan for a populated column left of all frames.
# - Original `by[f["inner"]]` and `assert cur["inner"] == a` no longer hold; matching must be keyed by `border`.
# - Original bridge coloring by `a` no longer holds; combo coloring must use the destination border `b`.
# - Geometry assumptions such as equal-sized frame rings and bridge bands remain valid and are preserved.
#
# d) Placement legality
# - Frames live on a lattice with at least one background cell between neighboring outer borders.
# - The control column is strictly left of the frame field, with at least two blank columns of separation.
# - Output is built first, bridges are erased to form the input, and the solver is validated against that virtual output.
# - Bridge cell sets are checked for overlap and collisions with existing frame pixels.
#
# e) Visual observability
# - A human sees the ordering clue in a vertical left-side strip, not in a bottom strip.
# - A human also sees that the strip colors match outer borders while interiors all repeat, so border matching is forced.
# - Because every bridge segment takes the destination border color, the bridge color "points toward" its endpoint across multiple examples.
#
# f) Visual cleanliness
# - Only three layers remain: frames, left control strip, and bridges.
# - The repeated inner color keeps the frame interiors quiet, leaving border colors and bridge colors as the salient clues.
# - The left strip is spatially isolated from the lattice, avoiding ambiguous merges.
#
# g) Change inferability
# - No hidden numeric ordering or color arithmetic is used.
# - Colors have small, structured roles: one background, one repeated inner fill, several unique border colors reused by control pixels and bridges.
# - The bridge color rule is inferable because every segment shares its color with the destination border, not the source border.
#
# h) Non-degeneration metric used after implementation
# - Metric 1: all frames share one inner color while path border colors are unique; therefore border matching is necessary.
# - Metric 2: every bridge segment has `destination border != source border`; therefore destination-colored output differs from source-colored output.
# - Metric 3: there is no usable bottom control row with >=2 non-background pixels below the frame field; the left column is the only sequence source.
#
# i) New solver assumptions and generator guarantees
# - Assumption: every valid frame is a single-color rectangular ring around a single-color non-background interior.
#   Guarantee: `_draw_frame()` constructs exactly that.
# - Assumption: at least one populated control column exists strictly left of all frames and contains >=2 non-background pixels.
#   Guarantee: generator reserves one such column and keeps other left-side columns empty.
# - Assumption: control colors correspond to frame border colors.
#   Guarantee: generator writes the path's border colors into the control column.
# - Assumption: consecutive path frames share either the same row band or the same column band.
#   Guarantee: the path is a 4-neighbor lattice path, so adjacent slots are axis-aligned neighbors.


def _detect_frames(g: np.ndarray, bg: int) -> list[dict]:
    frames: list[dict] = []
    colors = sorted(set(map(int, g.flatten())) - {int(bg)})
    for border_col in colors:
        lab, n = label_mask(g == int(border_col), connectivity=4)
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

            found = None
            for tt in range(1, int(min(h, w) // 2) + 1):
                inner = g[int(r0 + tt) : int(r1 - tt) + 1, int(c0 + tt) : int(c1 - tt) + 1]
                if inner.size == 0:
                    continue
                if bool((inner == int(border_col)).any()):
                    continue
                inner_vals = set(map(int, inner.flatten()))
                if len(inner_vals) != 1:
                    continue
                inner_col = int(next(iter(inner_vals)))
                if int(inner_col) == int(bg):
                    continue

                ring_area = int(h * w - (h - 2 * tt) * (w - 2 * tt))
                if int(pts.shape[0]) != int(ring_area):
                    continue

                ok = True
                if not bool((g[int(r0) : int(r0 + tt), int(c0) : int(c1) + 1] == int(border_col)).all()):
                    ok = False
                if not bool((g[int(r1 - tt) + 1 : int(r1) + 1, int(c0) : int(c1) + 1] == int(border_col)).all()):
                    ok = False
                if not bool((g[int(r0) : int(r1) + 1, int(c0) : int(c0 + tt)] == int(border_col)).all()):
                    ok = False
                if not bool((g[int(r0) : int(r1) + 1, int(c1 - tt) + 1 : int(c1) + 1] == int(border_col)).all()):
                    ok = False
                if not ok:
                    continue

                found = {
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    "t": int(tt),
                    "border": int(border_col),
                    "inner": int(inner_col),
                    "inner_bbox": (int(r0 + tt), int(c0 + tt), int(r1 - tt), int(c1 - tt)),
                }
                break

            if found is not None:
                frames.append(found)

    return frames


def _read_left_control_sequence(g: np.ndarray, bg: int, frames: list[dict]) -> list[int]:
    min_frame_c = min(int(f["bbox"][1]) for f in frames)
    ctrl_c = None
    for c in range(int(min_frame_c) - 1, -1, -1):
        if int((g[:, int(c)] != int(bg)).sum()) >= 2:
            ctrl_c = int(c)
            break
    assert not (ctrl_c is None)
    ctrl = [int(g[int(r), int(ctrl_c)]) for r in range(int(g.shape[0])) if int(g[int(r), int(ctrl_c)]) != int(bg)]
    assert not (len(ctrl) < 2)
    return ctrl


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect framed squares with destination-colored bridges.

    Combo rule:
    - Match control colors to frame border colors.
    - Read the control sequence from a left-side column, top-to-bottom.
    - Paint each bridge using the destination frame's border color.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

    frames = _detect_frames(g, int(bg))
    assert not (not frames)

    by: dict[int, list[dict]] = defaultdict(list)
    for f in frames:
        by[int(f["border"])].append(f)

    ctrl = _read_left_control_sequence(g, int(bg), frames)

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
                r1_, c1_ = center(f)
                r2_, c2_ = center(g2)
                d = abs(int(r1_) - int(r2_)) + abs(int(c1_) - int(c2_))
                cand = (int(al), -int(d))
                if best is None or cand > best:
                    best = cand
                    bestf = f
        assert not (bestf is None)
        cur = bestf

    out = g.copy()

    def draw_bridge(fa: dict, fb: dict, color: int) -> None:
        r0a, c0a, r1a, c1a = map(int, fa["bbox"])
        r0b, c0b, r1b, c1b = map(int, fb["bbox"])
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
        assert int(cur["border"]) == int(a)
        assert int(b) in by
        cands = by[b]
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
        assert not (bestf is None)
        nxt = bestf
        draw_bridge(cur, nxt, int(b))
        cur = nxt

    return out


def _draw_frame(g: np.ndarray, *, r0: int, c0: int, outer: int, t: int, border: int, inner_col: int) -> None:
    g[int(r0) : int(r0 + outer), int(c0) : int(c0 + outer)] = int(border)
    g[int(r0 + t) : int(r0 + outer - t), int(c0 + t) : int(c0 + outer - t)] = int(inner_col)


def _bridge_cells(fa: dict, fb: dict) -> set[tuple[int, int]]:
    r0a, c0a, r1a, c1a = map(int, fa["bbox"])
    r0b, c0b, r1b, c1b = map(int, fb["bbox"])
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
    avail = [int(d) for d in _DIGITS if int(d) != int(bg)]

    t = int(np.random.choice(np.array([1, 1, 2], dtype=int)))
    inner = int(np.random.randint(2, 4))
    outer = int(inner + 2 * t)
    if not (4 <= int(outer) <= 7):
        return None

    R = int(np.random.randint(2, 6))
    C = int(np.random.randint(2, 6))
    max_frames = int(R * C)
    if max_frames < 4:
        return None
    n_frames = int(np.random.randint(4, int(min(8, max_frames)) + 1))
    if len(avail) < int(n_frames + 1):
        return None

    path: list[tuple[int, int]] | None = None
    for _ in range(400):
        p = [(0, 0)]
        vis = {(0, 0)}
        ok = True
        for _step in range(int(n_frames) - 1):
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
    ctrl_c = int(np.random.randint(0, 3))
    ctrl_gap = int(np.random.randint(2, 6))
    left_margin = int(ctrl_c + ctrl_gap + 1)

    r0s = [int(top_margin)]
    for k in range(int(R) - 1):
        r0s.append(int(r0s[-1] + outer + row_gap[int(k)]))
    c0s = [int(left_margin)]
    for k in range(int(C) - 1):
        c0s.append(int(c0s[-1] + outer + col_gap[int(k)]))

    border_colors = [int(x) for x in np.random.choice(np.array(avail, dtype=int), size=int(n_frames), replace=False)]
    border_set = set(map(int, border_colors))
    inner_choices = [int(d) for d in avail if int(d) not in border_set]
    if not inner_choices:
        return None
    inner_col = int(np.random.choice(np.array(inner_choices, dtype=int)))

    slot_to_frame: dict[tuple[int, int], dict] = {}
    frames: list[dict] = []
    border_by_slot: dict[tuple[int, int], int] = {}
    for idx, (rr, cc) in enumerate(path):
        r0 = int(r0s[int(rr)])
        c0 = int(c0s[int(cc)])
        r1 = int(r0 + outer - 1)
        c1 = int(c0 + outer - 1)
        border_col = int(border_colors[int(idx)])
        f = {
            "bbox": (int(r0), int(c0), int(r1), int(c1)),
            "inner_bbox": (int(r0 + t), int(c0 + t), int(r1 - t), int(c1 - t)),
            "slot": (int(rr), int(cc)),
            "border": int(border_col),
            "inner": int(inner_col),
        }
        frames.append(f)
        slot_to_frame[(int(rr), int(cc))] = f
        border_by_slot[(int(rr), int(cc))] = int(border_col)

    max_r1 = max(int(f["bbox"][2]) for f in frames)
    max_c1 = max(int(f["bbox"][3]) for f in frames)

    ctrl_top = int(np.random.randint(0, 3))
    ctrl_rows = [int(ctrl_top + 2 * i) for i in range(int(n_frames))]
    bottom_margin = int(np.random.randint(1, 4))
    right_margin = int(np.random.randint(1, 5))
    H = int(max(int(max_r1 + 1), int(ctrl_rows[-1] + 1)) + bottom_margin)
    W = int(max_c1 + 1 + right_margin)
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
            border=int(f["border"]),
            inner_col=int(inner_col),
        )

    bridge_cells_all: set[tuple[int, int]] = set()
    ok = True
    for a_slot, b_slot in zip(path, path[1:]):
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
        paint = int(border_by_slot[(int(b_slot[0]), int(b_slot[1]))])
        for r, c in cells:
            out[int(r), int(c)] = int(paint)
        bridge_cells_all |= set(cells)
    if not ok:
        return None

    ctrl_colors = [int(border_by_slot[(int(r), int(c))]) for r, c in path]
    for row, col in zip(ctrl_rows, ctrl_colors):
        out[int(row), int(ctrl_c)] = int(col)

    inp = out.copy()
    for r, c in bridge_cells_all:
        inp[int(r), int(c)] = int(bg)

    if int(get_bg(inp)) != int(bg):
        return None

    frames_found = _detect_frames(inp, int(bg))
    if int(len(frames_found)) != int(len(frames)):
        return None
    if len({int(f["border"]) for f in frames_found}) != int(len(frames)):
        return None
    if {int(f["inner"]) for f in frames_found} != {int(inner_col)}:
        return None
    if set(ctrl_colors) != {int(f["border"]) for f in frames_found}:
        return None
    if int(inner_col) in set(ctrl_colors):
        return None

    min_frame_c = min(int(f["bbox"][1]) for f in frames)
    if any(int((inp[:, int(c)] != int(bg)).sum()) != 0 for c in range(int(ctrl_c + 1), int(min_frame_c))):
        return None
    tail_counts = (inp[int(max_r1 + 1) :, :] != int(bg)).sum(axis=1)
    if (int(tail_counts.max()) if int(tail_counts.size) else 0) >= 2:
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
