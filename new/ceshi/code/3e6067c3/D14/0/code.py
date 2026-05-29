# Variant of 3e6067c3
# Dimension: D14 — 控制像素的空间编码 (meta)
# Change: 控制像素等间距排列在一行中（间隔2像素），位于所有方块下方... → 控制像素间距编码额外信息
#
# Design analysis for this variant
# - Source-task review: none was provided, so there is no extra solver to inspect.
# - a) Concrete behavior:
#   The framed squares and the control row stay in the same overall places as the original task.
#   The only visible input-side change is that the control pixels are no longer equally spaced.
#   Consecutive control pixels separated by 2 columns encode a thin bridge for that step; separation
#   by 3 columns encodes a thick bridge for that step. The color sequence is still read left-to-right.
#   Output-side change: when moving from control color ci to ci+1, the bridge painted in color ci is
#   either a centered 1-cell-wide strip (thin) or the full original inner-band width (thick).
#   Small trace:
#   original: control columns [1,3,5] -> colors A,B,C -> draw full-width A bridge then full-width B bridge.
#   variant: control columns [1,3,6] -> gaps [2,3] -> colors A,B,C -> draw thin A bridge then thick B bridge.
#   The path-selection logic is unchanged; only bridge rendering depends on spatial spacing.
# - b) Complexity calibration:
#   Only two spacing classes are used: 2 and 3. That keeps the rule inferable while still visibly different.
#   To avoid degeneration, the generator forces every sample to contain both spacing classes, and it forces
#   square inner size to be 2 or 3 so thin vs thick bridges are visually distinguishable.
# - c) Old implicit assumptions that break under D14:
#   `ctrl = [...] if non-bg` in the original solver discards control columns and keeps only colors. That is
#   no longer enough because column differences now carry meaning, so the new solver must preserve positions.
#   `draw_bridge(..., color)` in the original solver always filled the full inner band. That was harmless only
#   because thickness was constant. Under D14, unconditional full-band filling would erase the thin/thick
#   distinction, so rendering must branch on each consecutive spacing.
#   The rest of `main()` still depends on the old invariants: all frames share compatible lattice alignment,
#   candidate squares are resolved by same-band preference plus proximity, and the control row is still the
#   bottom-most row below the squares with at least two colored pixels.
# - d) Placement legality:
#   The generator still constructs a self-avoiding path on a lattice, computes full-band bridge corridors,
#   and rejects any scene where those corridors overlap or hit non-background cells. Thin bridges are a subset
#   of those legal corridors, so they are automatically legal once the full corridor is legal.
#   Width is chosen as `max(scene_width, last_control_col + margin)` so variable control spacing never goes
#   out of bounds.
# - e) Visual observability:
#   In every generated sample there are both 1-gap and 2-gap blank runs between control pixels, and the
#   corresponding output contains both centered thin bridges and full-width bridges. A human can compare the
#   control-row spacing with the bridge widths across 2-3 examples and infer the mapping.
# - f) Visual cleanliness:
#   No new markers are added inside the square field. The only extra signal lives in the control row, and the
#   output still contains at most one bridge corridor per path edge. This keeps the scene readable.
# - g) Color-rule inferability:
#   Colors still serve exactly the original role: they identify visitation order and bridge color. No hidden
#   numeric ordering over color ids is introduced.
# - h) Variant-specific quantitative check:
#   After testing, we sample generated instances and measure how many control gaps decode as thin vs thick.
#   A valid variant should show both counts > 0 in essentially every sample, proving spacing is active.
# - i) New solver assumptions and generator guarantees:
#   The solver now assumes every consecutive control gap is either 2 or 3 columns in practice, interpreted as
#   thin vs thick. The generator guarantees that. The solver also assumes inner width >= 2 if thick/thin needs
#   to be visibly different; the generator enforces inner size in {2,3}.

from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np

from arc_utils import get_bg, in_bounds, neighbors4, label_mask

_DIGITS = [int(i) for i in range(10)]
_THIN_STEP = 2
_THICK_STEP = 3


def _mode_from_step(step: int) -> str:
    return "thick" if int(step) >= int(_THICK_STEP) else "thin"


def _most_common_nonbg(g: np.ndarray, bg: int) -> int | None:
    cnt = Counter(map(int, g.flatten()))
    for col, _k in cnt.most_common():
        if int(col) != int(bg):
            return int(col)
    return None


def _draw_frame(g: np.ndarray, *, r0: int, c0: int, outer: int, t: int, border: int, inner_col: int) -> None:
    g[int(r0) : int(r0 + outer), int(c0) : int(c0 + outer)] = int(border)
    g[int(r0 + t) : int(r0 + outer - t), int(c0 + t) : int(c0 + outer - t)] = int(inner_col)


def _collect_control_metadata(g: np.ndarray, bg: int, max_frame_r: int) -> tuple[int, list[int], list[int], list[str]]:
    H, W = map(int, g.shape)
    ctrl_r = None
    for r in range(int(H - 1), int(max_frame_r), -1):
        if int((g[int(r)] != int(bg)).sum()) >= 2:
            ctrl_r = int(r)
            break
    assert not (ctrl_r is None)

    ctrl_cols = [int(c) for c in range(int(W)) if int(g[int(ctrl_r), int(c)]) != int(bg)]
    ctrl = [int(g[int(ctrl_r), int(c)]) for c in ctrl_cols]
    assert not (len(ctrl) < 2)
    ctrl_modes = [_mode_from_step(int(ctrl_cols[i + 1]) - int(ctrl_cols[i])) for i in range(int(len(ctrl_cols)) - 1)]
    return int(ctrl_r), ctrl_cols, ctrl, ctrl_modes


def _bridge_span(fa: dict, fb: dict) -> tuple[str, int, int, int, int] | None:
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
            return None
        return "h", int(rr0), int(rr1), int(cc0), int(cc1)

    if int(ia[1]) == int(ib[1]) and int(ia[3]) == int(ib[3]):
        cc0, cc1 = int(ia[1]), int(ia[3])
        if int(r1a) < int(r0b):
            rr0, rr1 = int(r1a + 1), int(r0b - 1)
        else:
            rr0, rr1 = int(r1b + 1), int(r0a - 1)
        if int(rr0) > int(rr1):
            return None
        return "v", int(rr0), int(rr1), int(cc0), int(cc1)

    return None


def _bridge_cells(fa: dict, fb: dict, mode: str) -> set[tuple[int, int]]:
    span = _bridge_span(fa, fb)
    if span is None:
        return set()
    orient, rr0, rr1, cc0, cc1 = span
    if orient == "h":
        if mode == "thin":
            mid = int((rr0 + rr1) // 2)
            rr0, rr1 = int(mid), int(mid)
    else:
        if mode == "thin":
            mid = int((cc0 + cc1) // 2)
            cc0, cc1 = int(mid), int(mid)
    return {(int(r), int(c)) for r in range(int(rr0), int(rr1) + 1) for c in range(int(cc0), int(cc1) + 1)}


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect framed squares via colored bridges following a control-pixel sequence.

    Variant rule:
    - Left-to-right control colors still encode visitation order.
    - Consecutive control-pixel spacing now encodes bridge thickness:
      step 2 => thin centered bridge, step 3+ => thick full-band bridge.
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
    _ctrl_r, _ctrl_cols, ctrl, ctrl_modes = _collect_control_metadata(g, int(bg), int(max_frame_r))

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
        cur = bestf

    out = g.copy()

    def draw_bridge(fa: dict, fb: dict, color: int, mode: str) -> None:
        cells = _bridge_cells(fa, fb, mode)
        for r, c in cells:
            if 0 <= int(r) < int(H) and 0 <= int(c) < int(W):
                out[int(r), int(c)] = int(color)

    for i in range(int(len(ctrl)) - 1):
        a = int(ctrl[i])
        b = int(ctrl[i + 1])
        mode = str(ctrl_modes[i])
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
        draw_bridge(cur, nxt, int(a), mode)
        cur = nxt

    return out


def _gen_attempt() -> np.ndarray | None:
    bg = int(np.random.choice(np.array(_DIGITS, dtype=int)))
    border = int(np.random.choice(np.array([d for d in _DIGITS if int(d) != int(bg)], dtype=int)))
    inner_palette = [int(d) for d in _DIGITS if int(d) not in (int(bg), int(border))]
    if len(inner_palette) < 3:
        return None

    t = int(np.random.choice(np.array([1, 1, 2], dtype=int)))
    inner = int(np.random.randint(2, 4))  # force visible thin vs thick difference
    outer = int(inner + 2 * t)
    if not (4 <= int(outer) <= 7):
        return None

    R = int(np.random.randint(2, 6))
    C = int(np.random.randint(2, 6))
    max_frames = int(R * C)
    if max_frames < 4:
        return None
    n_frames = int(np.random.randint(4, int(min(8, max_frames)) + 1))

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

    bridge_modes = [str(np.random.choice(np.array(["thin", "thick"], dtype=object))) for _ in range(int(len(path)) - 1)]
    if "thin" not in bridge_modes or "thick" not in bridge_modes:
        return None

    ctrl_colors = [int(colors_by_slot[(int(r), int(c))]) for r, c in path]
    if len(ctrl_colors) < 2:
        return None

    ctrl_cols = [1]
    for mode in bridge_modes:
        step = int(_THICK_STEP if mode == "thick" else _THIN_STEP)
        ctrl_cols.append(int(ctrl_cols[-1] + step))

    ctrl_gap = int(np.random.randint(2, 6))
    ctrl_r = int(max_r1 + ctrl_gap)
    bottom_margin = 1
    right_margin = int(np.random.randint(1, 5))
    H = int(ctrl_r + 1 + bottom_margin)
    W = int(max(int(max_c1 + 1 + right_margin), int(ctrl_cols[-1] + 2)))

    if not (7 <= int(H) <= 30 and 7 <= int(W) <= 30):
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
        cells = _bridge_cells(fa, fb, bridge_modes[int(idx)])
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
        paint = int(colors_by_slot[(int(a_slot[0]), int(a_slot[1]))])
        for r, c in cells:
            out[int(r), int(c)] = int(paint)
        bridge_cells_all |= set(cells)

    if not ok:
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

    # Variant-activity guarantee: the same sample must genuinely use both spacing classes.
    _ctrl_r, ctrl_cols_check, _ctrl_check, ctrl_modes_check = _collect_control_metadata(inp, int(bg), int(max_r1))
    if ctrl_cols_check != ctrl_cols:
        return None
    if ctrl_modes_check != bridge_modes:
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(500):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("Failed to generate valid input after 500 attempts")
