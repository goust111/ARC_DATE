# Combo variant of 3e6067c3 (index 13)
# combo_spec: D5/2+D7/0
# Name: 单像素桥梁+紧凑裁剪

from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np

from arc_utils import get_bg, in_bounds, label_mask, neighbors4

"""
Design analysis for combo D5/2 + D7/0.

a) Concrete behavior and trace
   - Base task: framed squares sit on a sparse lattice above a control row. The control
     pixels list inner colors in visitation order. The solver follows that order and
     connects consecutive aligned frames with a bridge colored by the current frame.
   - D5/2 changes the bridge geometry only. Original logic paints the whole inner band
     across the gap. Combo logic paints only the center row or center column of that band.
     Example: two horizontally aligned frames with inner rows 4..6 used to paint rows
     4,5,6 through the gap; now they paint only row 5.
   - D7/0 changes the returned canvas only. After drawing all thin bridges on the full
     input-sized canvas, the output is cropped to the minimal non-background rectangle
     strictly above the control row. This removes the control row and external margins.
   - The combined output therefore differs from the source task in two visible ways at
     once: bridges are thinner, and the finished scene is zoomed to the playfield.

b) Complexity calibration
   - To make D5 visible, the generator fixes the inner fill to 3x3. If the inner area
     were 1x1, a thick bridge and a single-pixel bridge would look identical.
   - To make D7 visible, the generator always keeps positive top/left/right margins and a
     positive gap above the control row, so the cropped output is strictly smaller.
   - The path still spans multiple lattice rows and columns, preserving the original
     alignment reasoning instead of collapsing to a trivial straight chain.

c) Implicit assumptions review
   - Frame detection by border-colored connected components still holds; neither D5 nor D7
     changes the input object vocabulary.
   - Candidate matching still relies on equal inner-row bands or equal inner-column bands,
     plus nearest-center tie-breaking. Thin bridges do not affect those coordinates.
   - The original `draw_bridge()` assumed every cell of the shared inner band should be
     filled. That assumption breaks under D5 and is replaced by a center-line cell set.
   - The original final `return out` assumed output shape equals input shape. D7 breaks
     that assumption, so the combo solver returns a crop computed after all bridges are
     drawn on full coordinates.
   - New combo-specific assumption: each aligned pair must have a unique center row/column.
     The generator guarantees this by fixing the inner size to 3.

d) Placement legality
   - Frames are placed on a lattice with at least one background cell gap between adjacent
     slots. Thin bridge cells are computed only inside those gaps.
   - Output-first construction rejects any sample whose bridges overlap frames, overlap one
     another, leave the canvas, or would fail exact solver reconstruction.
   - The crop is taken only from rows above the control row, so the control pixels never
     contaminate the output bbox.

e) Visual observability
   - Across examples, a human sees the same control-row semantics but thinner connections
     and the missing instruction row in the output. Those are direct pixel-level cues.
   - No hidden color ordering or numeric-color semantics are introduced. Colors keep the
     same role as in the source task.

f) Visual cleanliness
   - Thin bridges reduce clutter relative to the thick baseline, while the crop removes
     unused margins and the control strip. The result is visually cleaner, not noisier.

g) Post-hoc anti-degeneration metric
   - For D5: compare the combo output against the tight crop of the original thick-bridge
     reconstruction on the same input. The pixel Hamming distance must be positive.
   - For D7: compare input shape against combo output shape. Both height and width must
     shrink.
"""


def _most_common_nonbg(g: np.ndarray, bg: int) -> int | None:
    cnt = Counter(map(int, g.flatten()))
    for col, _k in cnt.most_common():
        if int(col) != int(bg):
            return int(col)
    return None


def _draw_frame(g: np.ndarray, *, r0: int, c0: int, outer: int, t: int, border: int, inner_col: int) -> None:
    g[int(r0) : int(r0 + outer), int(c0) : int(c0 + outer)] = int(border)
    g[int(r0 + t) : int(r0 + outer - t), int(c0 + t) : int(c0 + outer - t)] = int(inner_col)


def _aligned(f1: dict, f2: dict) -> tuple[bool, bool]:
    a = f1["inner_bbox"]
    b = f2["inner_bbox"]
    same_row = bool(int(a[0]) == int(b[0]) and int(a[2]) == int(b[2]))
    same_col = bool(int(a[1]) == int(b[1]) and int(a[3]) == int(b[3]))
    return same_row, same_col


def _center(f: dict) -> tuple[int, int]:
    r0, c0, r1, c1 = map(int, f["bbox"])
    return int((r0 + r1) // 2), int((c0 + c1) // 2)


def _center_bridge_cells(fa: dict, fb: dict) -> set[tuple[int, int]]:
    r0a, c0a, r1a, c1a = map(int, fa["bbox"])
    r0b, c0b, r1b, c1b = map(int, fb["bbox"])
    ia = fa["inner_bbox"]
    ib = fb["inner_bbox"]

    if int(ia[0]) == int(ib[0]) and int(ia[2]) == int(ib[2]):
        r = int((int(ia[0]) + int(ia[2])) // 2)
        if int(c1a) < int(c0b):
            cc0, cc1 = int(c1a + 1), int(c0b - 1)
        else:
            cc0, cc1 = int(c1b + 1), int(c0a - 1)
        if int(cc0) > int(cc1):
            return set()
        return {(int(r), int(c)) for c in range(int(cc0), int(cc1) + 1)}

    if int(ia[1]) == int(ib[1]) and int(ia[3]) == int(ib[3]):
        c = int((int(ia[1]) + int(ia[3])) // 2)
        if int(r1a) < int(r0b):
            rr0, rr1 = int(r1a + 1), int(r0b - 1)
        else:
            rr0, rr1 = int(r1b + 1), int(r0a - 1)
        if int(rr0) > int(rr1):
            return set()
        return {(int(r), int(c)) for r in range(int(rr0), int(rr1) + 1)}

    return set()


def _thick_bridge_cells(fa: dict, fb: dict) -> set[tuple[int, int]]:
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


def _tight_crop_scene(g: np.ndarray, bg: int, ctrl_r: int) -> np.ndarray:
    fg = np.argwhere(g[: int(ctrl_r), :] != int(bg))
    assert fg.size > 0
    r0, c0 = fg.min(axis=0)
    r1, c1 = fg.max(axis=0)
    r0, c0, r1, c1 = map(int, (r0, c0, r1, c1))
    return g[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1].copy()


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
    assert frames
    return frames


def _find_ctrl_row(g: np.ndarray, bg: int, frames: list[dict]) -> int:
    H = int(g.shape[0])
    max_frame_r = max(int(f["bbox"][2]) for f in frames)
    for r in range(int(H - 1), int(max_frame_r), -1):
        if int((g[int(r)] != int(bg)).sum()) >= 2:
            return int(r)
    raise AssertionError("control row not found")


def _choose_best(cur: dict, cands: list[dict]) -> dict:
    best = None
    bestf = None
    r1_, c1_ = _center(cur)
    for f in cands:
        sr, sc = _aligned(cur, f)
        al = 1 if (sr or sc) else 0
        r2_, c2_ = _center(f)
        d = abs(int(r1_) - int(r2_)) + abs(int(c1_) - int(c2_))
        cand = (int(al), -int(d))
        if best is None or cand > best:
            best = cand
            bestf = f
    assert bestf is not None
    return bestf


def _choose_start_frame(ctrl: list[int], by: dict[int, list[dict]]) -> dict:
    start_color = int(ctrl[0])
    assert start_color in by
    start_cands = by[start_color]
    if len(start_cands) == 1:
        return start_cands[0]

    nxt_color = int(ctrl[1])
    nxt_cands = by.get(nxt_color, [])
    best = None
    bestf = None
    for f in start_cands:
        for g2 in (nxt_cands if nxt_cands else [f]):
            sr, sc = _aligned(f, g2)
            al = 1 if (sr or sc) else 0
            r1_, c1_ = _center(f)
            r2_, c2_ = _center(g2)
            d = abs(int(r1_) - int(r2_)) + abs(int(c1_) - int(c2_))
            cand = (int(al), -int(d))
            if best is None or cand > best:
                best = cand
                bestf = f
    assert bestf is not None
    return bestf


def _solve_full(grid: np.ndarray, *, bridge_mode: str) -> tuple[np.ndarray, int, int]:
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

    border = _most_common_nonbg(g, int(bg))
    assert border is not None

    frames = _detect_frames(g, int(border))
    by: dict[int, list[dict]] = defaultdict(list)
    for f in frames:
        by[int(f["inner"])].append(f)

    ctrl_r = _find_ctrl_row(g, int(bg), frames)
    ctrl = [int(g[int(ctrl_r), int(c)]) for c in range(int(W)) if int(g[int(ctrl_r), int(c)]) != int(bg)]
    assert len(ctrl) >= 2

    cur = _choose_start_frame(ctrl, by)
    out = g.copy()

    cell_fn = _center_bridge_cells if bridge_mode == "thin" else _thick_bridge_cells
    for i in range(int(len(ctrl)) - 1):
        a = int(ctrl[i])
        b = int(ctrl[i + 1])
        assert int(cur["inner"]) == int(a)
        if b not in by:
            continue
        nxt = _choose_best(cur, by[b])
        for r, c in cell_fn(cur, nxt):
            if 0 <= int(r) < int(H) and 0 <= int(c) < int(W):
                out[int(r), int(c)] = int(a)
        cur = nxt

    return out, int(bg), int(ctrl_r)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect framed squares with one-pixel bridges, then tight-crop away the control row
    and all external margins.
    """
    full_out, bg, ctrl_r = _solve_full(np.array(grid, dtype=int), bridge_mode="thin")
    return _tight_crop_scene(full_out, int(bg), int(ctrl_r))


_DIGITS = [int(i) for i in range(10)]


def _gen_attempt() -> np.ndarray | None:
    bg = int(np.random.choice(np.array(_DIGITS, dtype=int)))
    border = int(np.random.choice(np.array([d for d in _DIGITS if int(d) != int(bg)], dtype=int)))
    inner_palette = [int(d) for d in _DIGITS if int(d) not in (int(bg), int(border))]
    if len(inner_palette) < 3:
        return None

    # Force D5 to be visible: with a 3x3 inner area, thick and thin bridges differ.
    t = int(np.random.choice(np.array([1, 2], dtype=int)))
    inner = 3
    outer = int(inner + 2 * t)

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

    ctrl_gap = int(np.random.randint(2, 6))
    ctrl_r = int(max_r1 + ctrl_gap)
    bottom_margin = int(np.random.randint(1, 3))
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

    full_out = np.full((int(H), int(W)), int(bg), dtype=int)
    for f in frames:
        rr, cc = map(int, f["slot"])
        r0, c0, _r1, _c1 = map(int, f["bbox"])
        _draw_frame(
            full_out,
            r0=int(r0),
            c0=int(c0),
            outer=int(outer),
            t=int(t),
            border=int(border),
            inner_col=int(colors_by_slot[(rr, cc)]),
        )

    bridge_cells_all: set[tuple[int, int]] = set()
    for a_slot, b_slot in zip(path, path[1:]):
        fa = slot_to_frame[(int(a_slot[0]), int(a_slot[1]))]
        fb = slot_to_frame[(int(b_slot[0]), int(b_slot[1]))]
        cells = _center_bridge_cells(fa, fb)
        if not cells:
            return None
        if cells & bridge_cells_all:
            return None
        for r, c in cells:
            if not (0 <= int(r) < int(H) and 0 <= int(c) < int(W)):
                return None
            if int(full_out[int(r), int(c)]) != int(bg):
                return None
        paint = int(colors_by_slot[(int(a_slot[0]), int(a_slot[1]))])
        for r, c in cells:
            full_out[int(r), int(c)] = int(paint)
        bridge_cells_all |= set(cells)

    ctrl_colors = [int(colors_by_slot[(int(r), int(c))]) for r, c in path]
    if len(ctrl_colors) < 2:
        return None
    ctrl_cols = [int(1 + 2 * i) for i in range(int(len(ctrl_colors)))]
    if int(ctrl_cols[-1]) >= int(W):
        return None
    for col, cc in zip(ctrl_colors, ctrl_cols):
        full_out[int(ctrl_r), int(cc)] = int(col)

    expected = _tight_crop_scene(full_out, int(bg), int(ctrl_r))
    if not (expected.shape[0] < full_out.shape[0] and expected.shape[1] < full_out.shape[1]):
        return None

    # Anti-degeneration check for D5: crop the thick-bridge oracle on the same scene and
    # require a visible pixel difference from the thin-bridge crop.
    thick_out = np.full((int(H), int(W)), int(bg), dtype=int)
    for f in frames:
        rr, cc = map(int, f["slot"])
        r0, c0, _r1, _c1 = map(int, f["bbox"])
        _draw_frame(
            thick_out,
            r0=int(r0),
            c0=int(c0),
            outer=int(outer),
            t=int(t),
            border=int(border),
            inner_col=int(colors_by_slot[(rr, cc)]),
        )
    for a_slot, b_slot in zip(path, path[1:]):
        fa = slot_to_frame[(int(a_slot[0]), int(a_slot[1]))]
        fb = slot_to_frame[(int(b_slot[0]), int(b_slot[1]))]
        paint = int(colors_by_slot[(int(a_slot[0]), int(a_slot[1]))])
        for r, c in _thick_bridge_cells(fa, fb):
            thick_out[int(r), int(c)] = int(paint)
    for col, cc in zip(ctrl_colors, ctrl_cols):
        thick_out[int(ctrl_r), int(cc)] = int(col)
    thick_crop = _tight_crop_scene(thick_out, int(bg), int(ctrl_r))
    if thick_crop.shape != expected.shape:
        return None
    if int(np.count_nonzero(thick_crop != expected)) <= 0:
        return None

    inp = full_out.copy()
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
    if pred.shape != expected.shape:
        return None
    if not bool(np.array_equal(pred, expected)):
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(500):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("Failed to generate valid input after 500 attempts")
