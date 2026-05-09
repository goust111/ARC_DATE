# Variant of 3e6067c3
# Dimension: D3 — 源对象：控制像素的编码形式 (src)
# Change: 底部区域的单行控制像素序列，每个像素代表一种颜色，从左到右编码访问顺序... → 对角线标记对指示连接

from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np

from arc_utils import get_bg, in_bounds, neighbors4, label_mask

# Design analysis for this D3 variant
# ===================================
# Reference-task baseline:
# - The prompt names `9385bd28` as the inspiration for the new control encoding.
# - In this workspace there is no standalone solver file for `9385bd28`; only
#   `/home/lixinyu/snap/new/ARC-data/ARC-AGI-2/data/evaluation/9385bd28.json`
#   exists. So the baseline below is inferred from visible I/O behavior there.
# - The relevant visible mechanism is a 2x2 diagonal marker: two colored pixels
#   occupy opposite corners of a 2x2 box, with the other two cells left as
#   background. The diagonal orientation itself is meaningful and visually obvious.
#
# a) Concrete behavior and trace
# - Original task:
#   1. Frames are detected exactly as before.
#   2. A bottom control row lists colors `c0,c1,c2,...`.
#   3. The solver walks that sequence and draws bridges for
#      `(c0->c1), (c1->c2), ...`.
# - New variant:
#   1. Frames are still detected exactly as before.
#   2. The bottom control area is now two rows tall, containing several disjoint
#      2x2 diagonal markers.
#   3. Each marker directly encodes one connection pair:
#      `\` diagonal => top-left is source color, bottom-right is target color.
#      `/` diagonal => top-right is source color, bottom-left is target color.
#   4. The solver reads each pair independently and draws the bridge colored by
#      the source color, exactly matching the original bridge-painting rule.
# - Small trace:
#   Original control row `[2,3,9,4]` means bridges `(2->3),(3->9),(9->4)`.
#   New control area may instead contain three markers:
#     `[[2,bg],[bg,3]]`, `[[bg,3],[9,bg]]`, `[[9,bg],[bg,4]]`.
#   Both produce the same three bridges, but the encoding changed from a single
#   linear sequence to explicit source-target pair objects.
#
# b) Complexity calibration
# - The variant must look different at a glance. A two-row band of repeated 2x2
#   diagonal markers is visually distinct from a single sparse control row.
# - The markers still use only two orientations, so the coding system remains
#   human-tractable.
# - To prevent direct-pair decoding from becoming ambiguous, inner colors are made
#   globally unique in generated examples. This is a necessary support change:
#   the original sequential walk could use "current frame" state to disambiguate
#   repeats, but a standalone pair marker cannot.
#
# c) Implicit-assumption review against original main()
# - Original `ctrl_r` scan assumed exactly one control row below frames. That is
#   no longer valid, so the new solver instead parses all non-background components
#   below the frames and keeps only size-2 diagonal 2x2 markers.
# - Original `ctrl = [...]` assumed one color per step and required a start state.
#   That no longer holds; each control object now already contains both endpoints.
# - Original start-frame selection logic depended on the first control color.
#   The new solver removes this stateful start logic entirely.
# - Original loop advanced `cur = nxt` after each bridge. That was safe because
#   the sequence encoding imposed an order. The new solver must not rely on order;
#   each marker is decoded independently.
#
# d) Placement legality
# - Frames are still placed on a lattice with at least one background cell between
#   neighboring boxes.
# - The bridge path remains a self-avoiding lattice path, so every intended bridge
#   is axis-aligned and uses the original bridge geometry.
# - Control markers are placed only below the frames, separated by at least one
#   blank row, and with at least one blank column between neighboring 2x2 boxes.
#   This keeps 8-connected marker components disjoint.
#
# e) Visual observability
# - A human can see that the bottom region is no longer "read colors left-to-right
#   as a row". Instead it is a series of little diagonal pair glyphs, each with two
#   colors. That directly exposes the D3 change in 2-3 examples.
#
# f) Visual cleanliness
# - The control area is kept to exactly two rows of markers plus one blank row
#   beneath, so it remains visually separate from the frame field.
# - Markers never overlap and bridges are still erased in the input, preserving the
#   original clean before/after contrast.
#
# g) Color-change inferability
# - No extra hidden color mapping is introduced. Marker colors are literal pointers
#   to source and target frames, and the bridge color still equals the source color.
#
# New solver assumptions introduced by this variant
# - Every intended control object below the frames is a size-2 diagonal component in
#   a 2x2 box. `generate_input()` guarantees this.
# - Every control color names exactly one frame in generated examples. The generator
#   guarantees global uniqueness of frame inner colors.
# - Every encoded pair links two frames sharing a row band or a column band. The
#   generator guarantees this by deriving pairs from consecutive cells on the path.


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect framed squares via colored bridges using diagonal pair markers.

    Input structure:
    - bg = background.
    - A border color forms equal-sized framed squares with inner fill colors.
    - Below the squares sits a two-row control area made of disjoint 2x2 diagonal
      markers. Each marker's two colors encode one source-target pair.

    Rule:
    - Parse every diagonal marker below the frames.
    - For each pair (src, dst), find the corresponding framed squares.
    - Draw the original style bridge between them, colored by src.
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

    def center(f: dict) -> tuple[int, int]:
        r0, c0, r1, c1 = map(int, f["bbox"])
        return int((r0 + r1) // 2), int((c0 + c1) // 2)

    def aligned(f1: dict, f2: dict) -> tuple[bool, bool]:
        a = f1["inner_bbox"]
        b = f2["inner_bbox"]
        same_row = bool(int(a[0]) == int(b[0]) and int(a[2]) == int(b[2]))
        same_col = bool(int(a[1]) == int(b[1]) and int(a[3]) == int(b[3]))
        return same_row, same_col

    def extract_ctrl_pairs() -> list[tuple[int, int]]:
        if int(max_frame_r + 1) >= int(H):
            return []
        sub = g[int(max_frame_r + 1) :, :]
        mask = sub != int(bg)
        lab2, n2 = label_mask(mask, connectivity=8)
        pairs: list[tuple[int, int, int]] = []
        for cid2 in range(1, int(n2) + 1):
            pts = np.argwhere(lab2 == int(cid2))
            if int(pts.shape[0]) != 2:
                continue
            r0, c0 = pts.min(axis=0)
            r1, c1 = pts.max(axis=0)
            r0, c0, r1, c1 = map(int, (r0, c0, r1, c1))
            if int(r1 - r0 + 1) != 2 or int(c1 - c0 + 1) != 2:
                continue
            rel = {(int(r - r0), int(c - c0)) for r, c in pts}
            if rel == {(0, 0), (1, 1)}:
                src = int(sub[int(r0), int(c0)])
                dst = int(sub[int(r1), int(c1)])
            elif rel == {(0, 1), (1, 0)}:
                src = int(sub[int(r0), int(c1)])
                dst = int(sub[int(r1), int(c0)])
            else:
                continue
            pairs.append((int(c0), int(src), int(dst)))
        pairs.sort()
        return [(int(src), int(dst)) for _c0, src, dst in pairs]

    ctrl_pairs = extract_ctrl_pairs()
    assert ctrl_pairs

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

    def best_pair_frames(src: int, dst: int) -> tuple[dict, dict] | None:
        if src not in by or dst not in by:
            return None
        best = None
        best_pair = None
        for fa in by[int(src)]:
            for fb in by[int(dst)]:
                sr, sc = aligned(fa, fb)
                al = 1 if (sr or sc) else 0
                (r1_, c1_) = center(fa)
                (r2_, c2_) = center(fb)
                d = abs(int(r1_) - int(r2_)) + abs(int(c1_) - int(c2_))
                cand = (int(al), -int(d))
                if best is None or cand > best:
                    best = cand
                    best_pair = (fa, fb)
        return best_pair

    for src, dst in ctrl_pairs:
        chosen = best_pair_frames(int(src), int(dst))
        if chosen is None:
            continue
        fa, fb = chosen
        draw_bridge(fa, fb, int(src))

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


def _extract_ctrl_pairs_from_grid(g: np.ndarray) -> list[tuple[int, int]]:
    bg = int(get_bg(g))
    cnt = Counter(map(int, g.flatten()))
    border = None
    for col, _k in cnt.most_common():
        if int(col) != int(bg):
            border = int(col)
            break
    if border is None:
        return []
    lab, n = label_mask(g == int(border), connectivity=4)
    max_frame_r = -1
    for cid in range(1, int(n) + 1):
        pts = np.argwhere(lab == int(cid))
        if pts.size == 0:
            continue
        r0, c0 = pts.min(axis=0)
        r1, c1 = pts.max(axis=0)
        if int(r1 - r0 + 1) >= 3 and int(c1 - c0 + 1) >= 3:
            max_frame_r = max(int(max_frame_r), int(r1))
    if int(max_frame_r + 1) >= int(g.shape[0]):
        return []
    sub = g[int(max_frame_r + 1) :, :]
    mask = sub != int(bg)
    lab2, n2 = label_mask(mask, connectivity=8)
    out: list[tuple[int, int, int]] = []
    for cid2 in range(1, int(n2) + 1):
        pts = np.argwhere(lab2 == int(cid2))
        if int(pts.shape[0]) != 2:
            continue
        r0, c0 = pts.min(axis=0)
        r1, c1 = pts.max(axis=0)
        r0, c0, r1, c1 = map(int, (r0, c0, r1, c1))
        if int(r1 - r0 + 1) != 2 or int(c1 - c0 + 1) != 2:
            continue
        rel = {(int(r - r0), int(c - c0)) for r, c in pts}
        if rel == {(0, 0), (1, 1)}:
            src = int(sub[int(r0), int(c0)])
            dst = int(sub[int(r1), int(c1)])
        elif rel == {(0, 1), (1, 0)}:
            src = int(sub[int(r0), int(c1)])
            dst = int(sub[int(r1), int(c0)])
        else:
            continue
        out.append((int(c0), int(src), int(dst)))
    out.sort()
    return [(int(src), int(dst)) for _c0, src, dst in out]


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
    n_frames = int(np.random.randint(4, int(min(len(inner_palette), max_frames)) + 1))

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

    n_pairs = int(len(path) - 1)
    ctrl_gap = int(np.random.randint(2, 6))
    ctrl_r0 = int(max_r1 + ctrl_gap)
    ctrl_r1 = int(ctrl_r0 + 1)
    bottom_margin = 1
    right_margin = int(np.random.randint(1, 5))
    min_ctrl_w = int(3 * n_pairs + 1)
    W = int(max(max_c1 + 1 + right_margin, min_ctrl_w))
    H = int(ctrl_r1 + 1 + bottom_margin)

    if not (7 <= int(H) <= 30 and 7 <= int(W) <= 30):
        return None

    chosen_colors = [int(x) for x in np.random.permutation(np.array(inner_palette, dtype=int))[: int(n_frames)]]
    colors_by_slot = {(int(slot[0]), int(slot[1])): int(col) for slot, col in zip(path, chosen_colors)}

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
        paint = int(colors_by_slot[(int(a_slot[0]), int(a_slot[1]))])
        for r, c in cells:
            out[int(r), int(c)] = int(paint)
        bridge_cells_all |= set(cells)
    if not ok:
        return None

    ctrl_pairs = [(int(colors_by_slot[a_slot]), int(colors_by_slot[b_slot])) for a_slot, b_slot in zip(path, path[1:])]
    if not ctrl_pairs:
        return None
    for i, (src, dst) in enumerate(ctrl_pairs):
        c0 = int(1 + 3 * i)
        if int(c0 + 1) >= int(W):
            return None
        if bool(np.random.randint(0, 2)):
            out[int(ctrl_r0), int(c0)] = int(src)
            out[int(ctrl_r1), int(c0 + 1)] = int(dst)
        else:
            out[int(ctrl_r0), int(c0 + 1)] = int(src)
            out[int(ctrl_r1), int(c0)] = int(dst)

    inp = out.copy()
    for r, c in bridge_cells_all:
        inp[int(r), int(c)] = int(bg)

    if int(get_bg(inp)) != int(bg):
        return None
    if int(_most_common_nonbg(inp, int(bg)) or -1) != int(border):
        return None
    if _extract_ctrl_pairs_from_grid(inp) != ctrl_pairs:
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
