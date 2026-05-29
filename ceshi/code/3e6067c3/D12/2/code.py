# Variant of 3e6067c3
# Dimension: D12 — 数量关系：源-目标基数 (meta)
# Change: 控制序列定义唯一路径，每段桥梁是1:1的方块对连接... → 双向桥梁

from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np

from arc_utils import get_bg, in_bounds, neighbors4, label_mask


# Design analysis for the D12 variant
# a) Concrete behavior:
#    The original task paints one full-width bridge in the source color. Here the same
#    aligned gap is still the target region, but it is split into two parallel lanes.
#    Horizontal gaps become an upper lane in the source color and a lower lane in the
#    target color. Vertical gaps become a left lane in the source color and a right lane
#    in the target color. Example: if a 4x4-inner square of color 2 connects to a 4x4-inner
#    square of color 7 across the same row, the old output filled the whole 4-cell-high gap
#    with 2; the new output fills the top 2 rows with 2 and the bottom 2 rows with 7.
#    The path-selection logic does not change; only bridge rendering and output-first
#    generation change.
# b) Complexity calibration:
#    The split must be visually obvious, so generator only uses even inner widths (2 or 4).
#    Width 2 yields two one-cell lanes; width 4 yields two two-cell lanes. This keeps the
#    relation simple while ensuring the dual-lane effect is visible and not a degenerate copy
#    of the original single bridge.
# c) Hidden assumptions review:
#    - Frame detection assumes a uniform framed square with one solid inner color. Kept.
#    - Alignment logic assumes matching inner row-band or col-band. Kept.
#    - Original draw_bridge() assumed a single paint color could overwrite the whole band.
#      That is no longer harmless: it would destroy the second lane. The new code must split
#      the band first, then paint each half with a different color.
#    - Original generator allowed odd inner widths; that would make "half width" ambiguous.
#      The new generator forbids odd inner widths.
# d) Placement legality:
#    The path still moves on a lattice of orthogonally adjacent frame slots, so every segment
#    is aligned and bridgeable. Gaps remain at least one background cell wide. Because the
#    dual bridge occupies the same union of cells as the original bridge, the original
#    non-overlap checks still suffice; generator now checks both lane sets explicitly.
# e) Visual observability:
#    A human can infer the new rule from 2-3 examples because every segment shows a paired
#    stripe pattern instead of a single solid bridge, and the two stripe colors are exactly
#    the consecutive control colors. The top/left lane always corresponds to the current
#    control color and the bottom/right lane to the next control color.
# f) Visual cleanliness:
#    No extra markers are introduced. The two lanes partition the old bridge rectangle, so
#    the scene stays tidy: frames, control row, and bridges remain disjoint.
# g) Color-change inferability:
#    No hidden mapping is used. The bridge colors are copied directly from adjacent control
#    colors, and the geometric split supplies the structural clue for how those two colors
#    coexist on one connection.
# h) Variant metric used after testing:
#    For generated outputs, every bridge segment should contain exactly two colors and both
#    lane areas should be non-empty. We also track whether the sample set includes both
#    horizontal and vertical dual bridges.
# i) New solver assumptions audited against generator:
#    - Every bridgeable band has even thickness. Generator guarantees inner size in {2, 4}.
#    - Every consecutive path step is between orthogonal lattice neighbors. Generator
#      constructs a self-avoiding 4-neighbor path, so every chosen segment is aligned.
#    - Bridge bands do not overlap earlier bridge bands or frames. Generator validates that
#      before accepting a sample.


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect framed squares via paired colored bridges following a control-pixel sequence.

    Each consecutive control pair (a, b) selects the same source/target frames as in the
    base task, but the gap is split into two parallel lanes:
    - horizontal bridge: upper half in color a, lower half in color b
    - vertical bridge: left half in color a, right half in color b
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
            colors = set(map(int, inner.flatten()))
            if len(colors) != 1:
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
            inner_col = int(next(iter(colors)))
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

    def center(f: dict) -> tuple[int, int]:
        r0, c0, r1, c1 = map(int, f["bbox"])
        return int((r0 + r1) // 2), int((c0 + c1) // 2)

    def aligned(f1: dict, f2: dict) -> tuple[bool, bool]:
        a = f1["inner_bbox"]
        b = f2["inner_bbox"]
        same_row = bool(int(a[0]) == int(b[0]) and int(a[2]) == int(b[2]))
        same_col = bool(int(a[1]) == int(b[1]) and int(a[3]) == int(b[3]))
        return same_row, same_col

    def bridge_parts(fa: dict, fb: dict) -> tuple[set[tuple[int, int]], set[tuple[int, int]]]:
        (r0a, c0a, r1a, c1a) = map(int, fa["bbox"])
        (r0b, c0b, r1b, c1b) = map(int, fb["bbox"])
        ia = fa["inner_bbox"]
        ib = fb["inner_bbox"]
        if int(ia[0]) == int(ib[0]) and int(ia[2]) == int(ib[2]):
            rr0, rr1 = int(ia[0]), int(ia[2])
            height = int(rr1 - rr0 + 1)
            assert height % 2 == 0
            split = int(height // 2)
            if int(c1a) < int(c0b):
                cc0, cc1 = int(c1a + 1), int(c0b - 1)
            else:
                cc0, cc1 = int(c1b + 1), int(c0a - 1)
            if int(cc0) > int(cc1):
                return set(), set()
            top = {
                (int(r), int(c))
                for r in range(int(rr0), int(rr0 + split))
                for c in range(int(cc0), int(cc1) + 1)
            }
            bottom = {
                (int(r), int(c))
                for r in range(int(rr0 + split), int(rr1) + 1)
                for c in range(int(cc0), int(cc1) + 1)
            }
            return top, bottom
        if int(ia[1]) == int(ib[1]) and int(ia[3]) == int(ib[3]):
            cc0, cc1 = int(ia[1]), int(ia[3])
            width = int(cc1 - cc0 + 1)
            assert width % 2 == 0
            split = int(width // 2)
            if int(r1a) < int(r0b):
                rr0, rr1 = int(r1a + 1), int(r0b - 1)
            else:
                rr0, rr1 = int(r1b + 1), int(r0a - 1)
            if int(rr0) > int(rr1):
                return set(), set()
            left = {
                (int(r), int(c))
                for r in range(int(rr0), int(rr1) + 1)
                for c in range(int(cc0), int(cc0 + split))
            }
            right = {
                (int(r), int(c))
                for r in range(int(rr0), int(rr1) + 1)
                for c in range(int(cc0 + split), int(cc1) + 1)
            }
            return left, right
        return set(), set()

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
    assert cur is not None

    out = g.copy()

    def draw_bridge(fa: dict, fb: dict, color_a: int, color_b: int) -> None:
        cells_a, cells_b = bridge_parts(fa, fb)
        for r, c in cells_a:
            if 0 <= int(r) < int(H) and 0 <= int(c) < int(W):
                out[int(r), int(c)] = int(color_a)
        for r, c in cells_b:
            if 0 <= int(r) < int(H) and 0 <= int(c) < int(W):
                out[int(r), int(c)] = int(color_b)

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
        assert nxt is not None
        draw_bridge(cur, nxt, int(a), int(b))
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


def _bridge_parts(fa: dict, fb: dict) -> tuple[set[tuple[int, int]], set[tuple[int, int]]]:
    (r0a, c0a, r1a, c1a) = map(int, fa["bbox"])
    (r0b, c0b, r1b, c1b) = map(int, fb["bbox"])
    ia = fa["inner_bbox"]
    ib = fb["inner_bbox"]
    if int(ia[0]) == int(ib[0]) and int(ia[2]) == int(ib[2]):
        rr0, rr1 = int(ia[0]), int(ia[2])
        height = int(rr1 - rr0 + 1)
        if height % 2 != 0:
            return set(), set()
        split = int(height // 2)
        if int(c1a) < int(c0b):
            cc0, cc1 = int(c1a + 1), int(c0b - 1)
        else:
            cc0, cc1 = int(c1b + 1), int(c0a - 1)
        if int(cc0) > int(cc1):
            return set(), set()
        top = {
            (int(r), int(c))
            for r in range(int(rr0), int(rr0 + split))
            for c in range(int(cc0), int(cc1) + 1)
        }
        bottom = {
            (int(r), int(c))
            for r in range(int(rr0 + split), int(rr1) + 1)
            for c in range(int(cc0), int(cc1) + 1)
        }
        return top, bottom
    if int(ia[1]) == int(ib[1]) and int(ia[3]) == int(ib[3]):
        cc0, cc1 = int(ia[1]), int(ia[3])
        width = int(cc1 - cc0 + 1)
        if width % 2 != 0:
            return set(), set()
        split = int(width // 2)
        if int(r1a) < int(r0b):
            rr0, rr1 = int(r1a + 1), int(r0b - 1)
        else:
            rr0, rr1 = int(r1b + 1), int(r0a - 1)
        if int(rr0) > int(rr1):
            return set(), set()
        left = {
            (int(r), int(c))
            for r in range(int(rr0), int(rr1) + 1)
            for c in range(int(cc0), int(cc0 + split))
        }
        right = {
            (int(r), int(c))
            for r in range(int(rr0), int(rr1) + 1)
            for c in range(int(cc0 + split), int(cc1) + 1)
        }
        return left, right
    return set(), set()


_DIGITS = [int(i) for i in range(10)]


def _gen_attempt() -> np.ndarray | None:
    bg = int(np.random.choice(np.array(_DIGITS, dtype=int)))
    border = int(np.random.choice(np.array([d for d in _DIGITS if int(d) != int(bg)], dtype=int)))
    inner_palette = [int(d) for d in _DIGITS if int(d) not in (int(bg), int(border))]
    if len(inner_palette) < 3:
        return None

    t = int(np.random.choice(np.array([1, 1, 2], dtype=int)))
    inner = int(np.random.choice(np.array([2, 4], dtype=int)))
    outer = int(inner + 2 * t)
    if not (4 <= int(outer) <= 8):
        return None

    R = int(np.random.randint(2, 6))
    C = int(np.random.randint(2, 6))
    max_frames = int(R * C)
    if max_frames < 4:
        return None
    n_frames = int(np.random.randint(4, int(min(12, max_frames)) + 1))

    path: list[tuple[int, int]] | None = None
    for _path_try in range(600):
        p = [(0, 0)]
        vis = {(0, 0)}
        ok = True
        horiz = 0
        vert = 0
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
            if int(nxt[0]) == int(r):
                horiz += 1
            else:
                vert += 1
            p.append((int(nxt[0]), int(nxt[1])))
            vis.add((int(nxt[0]), int(nxt[1])))
        if not ok:
            continue
        if len({r for r, _c in p}) < 2 or len({c for _r, c in p}) < 2:
            continue
        if horiz == 0 or vert == 0:
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
    ctrl_r = int(max_r1 + ctrl_gap)
    bottom_margin = 1
    right_margin = int(np.random.randint(1, 5))
    H = int(ctrl_r + 1 + bottom_margin)
    W = int(max_c1 + 1 + right_margin)
    if not (8 <= int(H) <= 30 and 8 <= int(W) <= 30):
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
    saw_horizontal = False
    saw_vertical = False
    ok = True
    for a_slot, b_slot in zip(path, path[1:]):
        fa = slot_to_frame[(int(a_slot[0]), int(a_slot[1]))]
        fb = slot_to_frame[(int(b_slot[0]), int(b_slot[1]))]
        cells_a, cells_b = _bridge_parts(fa, fb)
        if not cells_a or not cells_b:
            ok = False
            break
        union = set(cells_a) | set(cells_b)
        if union & bridge_cells_all:
            ok = False
            break
        for r, c in union:
            if not (0 <= int(r) < int(H) and 0 <= int(c) < int(W)):
                ok = False
                break
            if int(out[int(r), int(c)]) != int(bg):
                ok = False
                break
        if not ok:
            break
        if int(fa["inner_bbox"][0]) == int(fb["inner_bbox"][0]):
            saw_horizontal = True
        if int(fa["inner_bbox"][1]) == int(fb["inner_bbox"][1]):
            saw_vertical = True
        color_a = int(colors_by_slot[(int(a_slot[0]), int(a_slot[1]))])
        color_b = int(colors_by_slot[(int(b_slot[0]), int(b_slot[1]))])
        for r, c in cells_a:
            out[int(r), int(c)] = int(color_a)
        for r, c in cells_b:
            out[int(r), int(c)] = int(color_b)
        bridge_cells_all |= union

    if not ok or not saw_horizontal or not saw_vertical:
        return None

    ctrl_colors = [int(colors_by_slot[(int(r), int(c))]) for r, c in path]
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

    bridge_dual_segments = 0
    for a_slot, b_slot in zip(path, path[1:]):
        fa = slot_to_frame[(int(a_slot[0]), int(a_slot[1]))]
        fb = slot_to_frame[(int(b_slot[0]), int(b_slot[1]))]
        cells_a, cells_b = _bridge_parts(fa, fb)
        colors_seen = {int(out[int(r), int(c)]) for r, c in set(cells_a) | set(cells_b)}
        if len(colors_seen) == 2 and colors_seen == {
            int(colors_by_slot[(int(a_slot[0]), int(a_slot[1]))]),
            int(colors_by_slot[(int(b_slot[0]), int(b_slot[1]))]),
        }:
            bridge_dual_segments += 1
    if bridge_dual_segments != int(len(path) - 1):
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(500):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("Failed to generate valid input after 500 attempts")
