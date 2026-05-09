# Variant of 3e6067c3
# Dimension: D9 — 符号指令：控制像素的语义 (symbol_instructions)
# Change: 控制像素表示按序访问的方块颜色，隐含定义连接路径（每对相邻像素之间画桥）
#   -> 图例式色彩映射：控制区改成 2 列 x K 行的 legend，每行左像素是起点方块颜色，
#      右像素是终点方块颜色；每一行独立指定一座要画的桥。
#
# Design analysis
# - requested_reference: 58490d8a
# - reference_solver_status:
#   - The workspace does not contain a readable solver file for `58490d8a` under the
#     provided/expected solver paths, so there was no solver code to read.
#   - I still checked that task id and only used its evaluation examples as a semantic
#     baseline for “legend lookup”: control information is grouped into an explicit
#     lookup/table region instead of being inferred from a single sequence.
# - concretization:
#   - Main scene stays the same: equal-sized framed squares on a lattice, bridges are
#     horizontal/vertical bands whose thickness matches the inner square thickness, and
#     bridge color is still the start square's inner color.
#   - The visible D9 change is only the control area below the squares:
#     old input had one sparse control row like `bg a bg b bg c ...`;
#     new input has a compact 2-column legend block:
#         row 0: [a b]
#         row 1: [d e]
#         row 2: [b c]
#     meaning “draw bridge a->b”, “draw bridge d->e”, “draw bridge b->c”.
#   - I concretize “2xN” as “2 columns by K rows”, because the prompt says each pair
#     has a left pixel and a right pixel. A 2-row-by-N-column layout would not have a
#     natural left/right within each pair.
# - behavior trace on a small example:
#   - Original: sequence `[2, 5, 7]` means draw 2->5 using color 2, then 5->7 using
#     color 5; the middle 5 is only implicit because it is adjacent in the sequence.
#   - Variant: legend rows `[[5,7],[2,5]]` mean exactly the same two bridges, but now
#     each bridge is explicitly named and row order does not define a traversal path.
#   - Difference: the solver no longer keeps a persistent “current frame”; it parses
#     independent `(start_color, end_color)` rows.
# - complexity calibration:
#   - The explicit-pair legend removes the original sequence context that helped
#     disambiguate repeated square colors. To avoid introducing unrelated ambiguity,
#     the generator now makes square inner colors globally unique.
#   - Repetition still exists where it matters visually: a color can appear on multiple
#     legend rows when a square participates in multiple bridges.
#   - Legend rows are shuffled away from the path order when possible, so the control
#     block visibly behaves like a lookup table rather than a walk trace.
# - old-code assumptions that break under D9:
#   - The old `ctrl_r` / `ctrl = [...]` logic assumes all control info lives on one row.
#     That is false for the new legend block.
#   - The old `start_color = ctrl[0]`, `cur = ...`, and loop over consecutive control
#     colors assume the control tokens define one continuous path. That is also false.
#   - The old duplicate-color handling relied on the previous current frame plus the next
#     token to disambiguate repeated colors. After switching to explicit pairs, that
#     assumption is no longer safe, so generator-side uniqueness is required.
#   - The frame detection and bridge-drawing geometry do remain valid; only control
#     decoding and traversal state need to change.
# - placement legality:
#   - The generator first builds a valid self-avoiding lattice path so every requested
#     bridge connects aligned neighboring path frames and has non-empty cells.
#   - Bridge cells must be pairwise disjoint and must lie on background in the output
#     canvas before painting; otherwise the attempt is rejected.
#   - The legend block is placed strictly below the lowest frame, uses exactly two
#     adjacent columns across contiguous rows, and has one background row underneath.
# - visual observability / cleanliness:
#   - The new rule is visible immediately because the control area changes from one thin
#     horizontal row to a stacked 2-column table.
#   - Legend rows are isolated from the frame field by a vertical gap, so they cannot be
#     mistaken for bridges or square interiors.
#   - Only two control columns are used, keeping the table compact and readable.
# - color inference:
#   - No numeric color ordering is used. Every bridge is inferred from visible color
#     equality between legend pixels and square fills.
# - new solver assumptions and generator guarantees:
#   - Solver assumption: below the frame field there is one contiguous run of rows, each
#     containing exactly two adjacent non-background pixels in the same columns.
#     Generator guarantee: it constructs exactly such a legend block.
#   - Solver assumption: every legend row names a pair of frames that are aligned in the
#     same row band or column band.
#     Generator guarantee: legend rows are sampled from actual edges of the lattice path.
#   - Solver assumption: start/end colors can be matched to the intended frames.
#     Generator guarantee: all frame inner colors are globally unique.

from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np

from arc_utils import get_bg, in_bounds, neighbors4, label_mask


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


def _find_legend_pairs(g: np.ndarray, bg: int, min_row: int) -> list[tuple[int, int]]:
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
    legend_rows, c0 = max(groups, key=lambda item: (len(item[0]), item[0][-1]))
    pairs = [(int(g[int(r), int(c0)]), int(g[int(r), int(c0 + 1)])) for r in legend_rows]
    assert len(pairs) >= 2
    return pairs


def _center(f: dict) -> tuple[int, int]:
    r0, c0, r1, c1 = map(int, f["bbox"])
    return int((r0 + r1) // 2), int((c0 + c1) // 2)


def _aligned(f1: dict, f2: dict) -> tuple[bool, bool]:
    a = f1["inner_bbox"]
    b = f2["inner_bbox"]
    same_row = bool(int(a[0]) == int(b[0]) and int(a[2]) == int(b[2]))
    same_col = bool(int(a[1]) == int(b[1]) and int(a[3]) == int(b[3]))
    return same_row, same_col


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect framed squares via explicit legend rows.

    Input structure:
    - The framed-square field is unchanged from the original task.
    - Below it sits a 2-column legend block; each row is `(start_color, end_color)`.

    Rule:
    - For each legend row, find the square with the start color and the square with the
      end color.
    - If they share a row band or column band, draw the bridge between them using the
      start color.
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

    frames = _detect_frames(g, int(border))
    assert frames

    by: dict[int, list[dict]] = defaultdict(list)
    for f in frames:
        by[int(f["inner"])].append(f)

    max_frame_r = max(int(f["bbox"][2]) for f in frames)
    pairs = _find_legend_pairs(g, int(bg), int(max_frame_r + 1))

    out = g.copy()

    def draw_bridge(fa: dict, fb: dict, color: int) -> None:
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

    for a, b in pairs:
        assert a in by and b in by
        best = None
        best_pair = None
        for fa in by[int(a)]:
            for fb in by[int(b)]:
                sr, sc = _aligned(fa, fb)
                al = 1 if (sr or sc) else 0
                (r1_, c1_) = _center(fa)
                (r2_, c2_) = _center(fb)
                d = abs(int(r1_) - int(r2_)) + abs(int(c1_) - int(c2_))
                cand = (int(al), -int(d))
                if best is None or cand > best:
                    best = cand
                    best_pair = (fa, fb)
        assert best is not None and best_pair is not None and int(best[0]) == 1
        draw_bridge(best_pair[0], best_pair[1], int(a))

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


def _shuffle_pairs(pair_slots: list[tuple[tuple[int, int], tuple[int, int]]], colors_by_slot: dict[tuple[int, int], int]) -> list[tuple[tuple[int, int], tuple[int, int]]]:
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
    if len(inner_palette) < 5:
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
    n_frames = int(np.random.randint(4, int(min(8, max_frames, len(inner_palette))) + 1))

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

    legend_gap = int(np.random.randint(2, 6))
    legend_r0 = int(max_r1 + legend_gap)
    legend_c0 = int(np.random.randint(1, 4))
    bottom_margin = 1
    right_margin = int(np.random.randint(1, 5))
    H = int(legend_r0 + n_pairs + bottom_margin)
    W = int(max(max_c1 + 1 + right_margin, legend_c0 + 2 + right_margin))
    if not (7 <= int(H) <= 30 and 7 <= int(W) <= 30):
        return None

    chosen = np.random.choice(np.array(inner_palette, dtype=int), size=int(n_frames), replace=False)
    colors_by_slot = {slot: int(chosen[i]) for i, slot in enumerate(path)}

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
        paint = int(colors_by_slot[(int(a_slot[0]), int(a_slot[1]))])
        for r, c in cells:
            out[int(r), int(c)] = int(paint)
        bridge_cells_all |= set(cells)
    if not ok:
        return None

    legend_pairs = _shuffle_pairs(pair_slots, colors_by_slot)
    for i, (a_slot, b_slot) in enumerate(legend_pairs):
        out[int(legend_r0 + i), int(legend_c0)] = int(colors_by_slot[(int(a_slot[0]), int(a_slot[1]))])
        out[int(legend_r0 + i), int(legend_c0 + 1)] = int(colors_by_slot[(int(b_slot[0]), int(b_slot[1]))])

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
