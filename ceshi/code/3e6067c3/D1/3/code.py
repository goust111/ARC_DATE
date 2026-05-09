# Variant of 3e6067c3
# Dimension: D1 — 匹配条件：颜色匹配方式 (match_cond)
# Change: 控制像素颜色与方块内部填充颜色相同 (s.colors == d.inner_squares.colors)... → 内部像素数量匹配
#
# Design analysis
# a) Concrete behavior and trace
#    - Original task: each framed square has a solid inner fill color; the control row is a left-to-right
#      sequence of single colored pixels. Matching is color equality between control pixel and square fill.
#    - Variant: each framed square keeps the same outer border geometry, but its interior becomes background
#      plus exactly 1, 2, or 3 same-colored pixels ("dots"). The control row is no longer a sequence of
#      colors; it is a sequence of contiguous runs of one dedicated marker color, and the run lengths are
#      1/2/3. A run of length k selects a square whose interior contains k non-background pixels.
#    - Small trace:
#      original: control `[red][blue]` -> find red square then blue square -> draw a red bridge.
#      variant: control `[marker][marker marker]` -> find the 1-dot square then the 2-dot square ->
#      draw a bridge using the 1-dot square's own dot color.
#    - The changed logic sits exactly at the object-identification stage. Geometry, alignment preference,
#      nearest-candidate tie-break, bridge thickness, and output size stay unchanged.
#
# b) Complexity calibration
#    - The count alphabet is kept small: only 1, 2, 3 dots. This is large enough to create repetition,
#      so the solver still needs the original row/column-band disambiguation, but small enough that humans
#      can infer the rule from 2-3 examples.
#    - Dot colors are independent of counts and a single dedicated control-marker color is used for all
#      control groups. This prevents the scene from degenerating back into color matching.
#
# c) Old assumptions that break
#    - Original frame parsing assumed `len(unique(inner)) == 1` and used that color as both the square
#      identity and the bridge color. In the variant the inner region is mostly background, so that test
#      would reject every valid frame.
#    - Original control parsing read every non-background cell in the bottom row as one symbol. Here a
#      length-3 control token is three adjacent cells and must be parsed as one run with value 3.
#    - Original `assert cur["inner"] == a` relied on color identity. This becomes `assert cur["count"] == a`.
#    - Bridge painting cannot use the control token color anymore because the control row is monochrome.
#      The bridge must instead use the current square's dot color.
#
# d) Placement legality
#    - Frames remain axis-aligned border components and the path still walks between orthogonally adjacent
#      lattice slots, so bridge rectangles are unchanged and remain disjoint.
#    - The generator clears each inner region back to background before placing dots, guaranteeing that the
#      frame detector sees exactly one border ring and one interior dot color.
#    - Control groups are separated by at least one background cell, so run parsing is unambiguous.
#
# e) Visual observability
#    - Humans can see that the control row no longer uses many colors; instead it uses 1/2/3 identical
#      marker cells. The target squares likewise no longer have solid fills, only counted dots.
#    - Because bridge colors come from the dot colors while matching comes from dot counts, the examples
#      visibly decouple "which square is chosen" from "what color gets painted".
#
# f) Visual cleanliness
#    - Only three visual layers remain in the main region: border ring, sparse interior dots, and bridges.
#      Dots never overlap borders or bridges in the input, and bridges only fill inter-frame corridors.
#    - Unique paint colors per frame keep long bridge chains readable even though count labels repeat.
#
# g) Color-change inferability
#    - No hidden numeric ordering of colors is used. Control color is constant and semantically irrelevant
#      beyond marking the runs. Dot/bridge colors vary only to show which current square paints each bridge.
#
# Source-task note
#    - The prompt cites 8698868d as the inspiration for count-based symbolic cues. The repository segment
#      allowed for this task contains `8698868d.json` but not a solver file, so there was no source solver
#      implementation to inspect. The count-decoding mechanism below is therefore designed directly from the
#      allowed local assets and the variant specification.
#
# New assumptions review
#    - `main()` now assumes each valid frame interior contains exactly one non-background color and 1..3
#      pixels of that color. `generate_input()` enforces this by construction.
#    - `main()` assumes the control row below all frames contains contiguous non-background runs that encode
#      counts. `generate_input()` writes exactly such a row and leaves one blank row underneath, matching
#      the original scene layout.
#    - Candidate disambiguation still assumes the intended next square is the best aligned/nearest target
#      among same-count squares. The generator keeps count labels row/column-distinct on the lattice and
#      validates every sample by replaying `main()`, rejecting any ambiguous layout.

from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np

from arc_utils import get_bg, in_bounds, label_mask, neighbors4


def _most_common_nonbg(g: np.ndarray, bg: int) -> int | None:
    cnt = Counter(map(int, g.flatten()))
    for col, _k in cnt.most_common():
        if int(col) != int(bg):
            return int(col)
    return None


def _dot_offsets(inner: int, count: int) -> list[tuple[int, int]]:
    inner = int(inner)
    count = int(count)
    if inner == 2:
        patterns = {
            1: [(0, 0)],
            2: [(0, 0), (1, 1)],
            3: [(0, 0), (0, 1), (1, 1)],
        }
        return patterns[count]
    patterns = {
        1: [(1, 1)],
        2: [(0, 0), (2, 2)],
        3: [(0, 0), (1, 1), (2, 2)],
    }
    return patterns[count]


def _draw_frame(
    g: np.ndarray,
    *,
    r0: int,
    c0: int,
    outer: int,
    t: int,
    bg: int,
    border: int,
    paint: int,
    count: int,
) -> None:
    g[int(r0) : int(r0 + outer), int(c0) : int(c0 + outer)] = int(border)
    g[int(r0 + t) : int(r0 + outer - t), int(c0 + t) : int(c0 + outer - t)] = int(bg)
    inner = int(outer - 2 * t)
    for dr, dc in _dot_offsets(int(inner), int(count)):
        g[int(r0 + t + dr), int(c0 + t + dc)] = int(paint)


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


def _extract_frames(g: np.ndarray, bg: int, border: int) -> list[dict]:
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
        if h < 4 or w < 4:
            continue

        found = None
        for tt in range(1, int(min(h, w) // 2) + 1):
            inner = g[int(r0 + tt) : int(r1 - tt) + 1, int(c0 + tt) : int(c1 - tt) + 1]
            if inner.size == 0:
                continue
            if bool((inner == int(border)).any()):
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

            inner_mask = inner != int(bg)
            dot_count = int(inner_mask.sum())
            if dot_count < 1 or dot_count > 3:
                continue
            colors = set(map(int, inner[inner_mask].flatten()))
            if len(colors) != 1:
                continue
            paint = int(next(iter(colors)))
            found = {
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
                "t": int(tt),
                "count": int(dot_count),
                "paint": int(paint),
                "inner_bbox": (int(r0 + tt), int(c0 + tt), int(r1 - tt), int(c1 - tt)),
            }
            break

        if found is not None:
            frames.append(found)
    return frames


def _parse_control_counts(g: np.ndarray, bg: int, min_row: int) -> list[int]:
    H, W = g.shape
    ctrl_r = None
    for r in range(int(H - 1), int(min_row), -1):
        if int((g[int(r)] != int(bg)).sum()) >= 2:
            ctrl_r = int(r)
            break
    assert not (ctrl_r is None)

    counts: list[int] = []
    c = 0
    while int(c) < int(W):
        if int(g[int(ctrl_r), int(c)]) == int(bg):
            c += 1
            continue
        c0 = int(c)
        while int(c) < int(W) and int(g[int(ctrl_r), int(c)]) != int(bg):
            c += 1
        counts.append(int(c - c0))
    assert not (len(counts) < 2)
    return counts


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect framed squares via colored bridges following a control-count sequence.

    Matching changes from color equality to dot-count equality:
    - each square interior holds 1..3 non-background pixels of a single paint color
    - each control token is a contiguous run of 1..3 marker pixels
    - a token with length k selects a k-dot square
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

    frames = _extract_frames(g, int(bg), int(border))
    assert not (not frames)

    by_count: dict[int, list[dict]] = defaultdict(list)
    for f in frames:
        by_count[int(f["count"])].append(f)

    max_frame_r = max(int(f["bbox"][2]) for f in frames)
    ctrl = _parse_control_counts(g, int(bg), int(max_frame_r))

    def center(f: dict) -> tuple[int, int]:
        r0, c0, r1, c1 = map(int, f["bbox"])
        return int((r0 + r1) // 2), int((c0 + c1) // 2)

    def aligned(f1: dict, f2: dict) -> tuple[bool, bool]:
        a = f1["inner_bbox"]
        b = f2["inner_bbox"]
        same_row = bool(int(a[0]) == int(b[0]) and int(a[2]) == int(b[2]))
        same_col = bool(int(a[1]) == int(b[1]) and int(a[3]) == int(b[3]))
        return same_row, same_col

    start_count = int(ctrl[0])
    assert not (start_count not in by_count)
    start_cands = by_count[start_count]
    if len(start_cands) == 1:
        cur = start_cands[0]
    else:
        nxt_count = int(ctrl[1])
        nxt_cands = by_count.get(nxt_count, [])
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
    assert not (cur is None)

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
        assert int(cur["count"]) == int(a)
        if b not in by_count:
            continue
        cands = by_count[b]
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
        assert not (nxt is None)
        draw_bridge(cur, nxt, int(cur["paint"]))
        cur = nxt

    return out


_DIGITS = [int(i) for i in range(10)]


def _slot_count(rr: int, cc: int, offset: int) -> int:
    return int((int(rr) + 2 * int(cc) + int(offset)) % 3 + 1)


def _gen_attempt() -> np.ndarray | None:
    bg = int(np.random.choice(np.array(_DIGITS, dtype=int)))
    border = int(np.random.choice(np.array([d for d in _DIGITS if int(d) != int(bg)], dtype=int)))
    ctrl_color_choices = [d for d in _DIGITS if int(d) not in (int(bg), int(border))]
    if not ctrl_color_choices:
        return None
    ctrl_color = int(np.random.choice(np.array(ctrl_color_choices, dtype=int)))
    paint_palette = [int(d) for d in _DIGITS if int(d) not in (int(bg), int(border), int(ctrl_color))]
    if len(paint_palette) < 4:
        return None

    t = int(np.random.choice(np.array([1, 1, 2], dtype=int)))
    inner = int(np.random.choice(np.array([2, 3], dtype=int)))
    outer = int(inner + 2 * t)
    if not (4 <= int(outer) <= 7):
        return None

    R = int(np.random.randint(2, 4))
    C = int(np.random.randint(2, 4))
    max_frames = int(R * C)
    if max_frames < 4:
        return None
    n_frames = int(np.random.randint(4, int(min(7, max_frames)) + 1))

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

    row_gap = [int(np.random.randint(1, 5)) for _ in range(int(R) - 1)]
    col_gap = [int(np.random.randint(1, 5)) for _ in range(int(C) - 1)]
    top_margin = int(np.random.randint(1, 3))
    left_margin = int(np.random.randint(1, 3))

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

    offset = int(np.random.randint(0, 3))
    counts_by_slot = {(int(rr), int(cc)): int(_slot_count(int(rr), int(cc), int(offset))) for rr, cc in path}
    if len({int(v) for v in counts_by_slot.values()}) < 3:
        return None

    if len(paint_palette) < len(path):
        return None
    paints = np.random.permutation(np.array(paint_palette, dtype=int))[: len(path)]
    paint_by_slot = {(int(slot[0]), int(slot[1])): int(paints[i]) for i, slot in enumerate(path)}

    max_r1 = max(int(f["bbox"][2]) for f in frames)
    max_c1 = max(int(f["bbox"][3]) for f in frames)

    ctrl_gap = int(np.random.randint(2, 5))
    ctrl_r = int(max_r1 + ctrl_gap)
    bottom_margin = 1
    ctrl_counts = [int(counts_by_slot[(int(r), int(c))]) for r, c in path]
    ctrl_needed_w = int(1 + sum(ctrl_counts) + (len(ctrl_counts) - 1) + 1)
    right_margin = int(np.random.randint(1, 4))
    H = int(ctrl_r + 1 + bottom_margin)
    W = int(max(max_c1 + 1 + right_margin, ctrl_needed_w))

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
            bg=int(bg),
            border=int(border),
            paint=int(paint_by_slot[(rr, cc)]),
            count=int(counts_by_slot[(rr, cc)]),
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
        paint = int(paint_by_slot[(int(a_slot[0]), int(a_slot[1]))])
        for r, c in cells:
            out[int(r), int(c)] = int(paint)
        bridge_cells_all |= set(cells)
    if not ok:
        return None

    c = 1
    for count in ctrl_counts:
        if int(c + count) > int(W):
            return None
        out[int(ctrl_r), int(c) : int(c + count)] = int(ctrl_color)
        c += int(count + 1)

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
    for _ in range(800):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("Failed to generate valid input after 800 attempts")
