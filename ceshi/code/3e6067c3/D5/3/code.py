# Variant of 3e6067c3
# Dimension: D5 — 变换：桥梁绘制规则 (transform)
# Change: 用起始方块的内部颜色在两个方块之间绘制矩形桥梁，桥梁宽度等于内部填充区域的宽度... → 渐变桥梁：起点色到终点色

from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np

from arc_utils import get_bg, in_bounds, neighbors4, label_mask


"""
Design analysis for the D5 variant
==================================

variant_spec:
  core_change:
    before: "Every bridge rectangle is painted uniformly with the source square color."
    after: "Every bridge rectangle is split along its travel axis: the half adjacent to the source square uses the source color, and the half adjacent to the destination square uses the destination color."
    directionality:
      - "Left-to-right bridge: left half = source color, right half = destination color."
      - "Right-to-left bridge: right half = source color, left half = destination color."
      - "Top-to-bottom bridge: top half = source color, bottom half = destination color."
      - "Bottom-to-top bridge: bottom half = source color, top half = destination color."
    midpoint_policy: "For odd bridge lengths, the midpoint stays with the source-side half via ceil(length/2)."

  a_concretization_and_trace:
    scene:
      - "Input still contains framed squares on a lattice-like layout and one control row underneath."
      - "Output still draws rectangular bridges occupying exactly the gap between aligned frames with inner-band thickness."
      - "The only visible D5 difference is internal bridge coloring: one bridge now contains two contiguous color blocks instead of a single solid color."
    example:
      - "Suppose a red square at columns 1..4 connects to a green square at columns 9..12, with bridge columns 5..8."
      - "Original output paints columns 5..8 all red."
      - "Variant output paints columns 5..6 red and columns 7..8 green."
      - "If the travel direction reverses, the red part must stay on the side touching the red frame, so spatial orientation matters."
    affected_pipeline_stage:
      - "Frame detection: unchanged."
      - "Control decoding and target selection: unchanged."
      - "Bridge drawing in main(): changed."
      - "Virtual output construction in generator: changed to match the new two-color bridge."

  b_complexity_calibration:
    - "This is not reduced to arbitrary per-cell gradients; each bridge has exactly two ordered color zones."
    - "The generator keeps multiple bridges per sample (path length >= 4), so the new rule repeats enough times for humans to spot it."
    - "Bridge gaps are forced to be at least 2 cells, preventing invisible degeneration into a single-cell bridge that cannot show both endpoint colors."
    - "Odd and even bridge lengths are both allowed, but the split rule is single and consistent."

  c_hidden_assumption_audit:
    - "Original draw_bridge(..., color) assumed one scalar color was sufficient for the whole rectangle. That assumption is invalid now."
    - "Original _bridge_cells() only encoded geometry, not direction. The variant requires direction because the source-side half depends on which endpoint is the current square."
    - "Original generator painted bridges with colors_by_slot[start]. That would silently collapse the variant back to the original task, so generator painting must also use endpoint-aware halves."
    - "Original geometry helpers were safe because orientation was irrelevant. In the variant, helpers must preserve whether the current frame is left/right or above/below the next frame."

  d_placement_legality:
    - "Frames remain axis-aligned, non-overlapping, and placed on lattice slots exactly as before."
    - "Bridge legality still means: aligned inner bands, gap rectangle inside bounds, no overlap with other bridges, and no collision with pre-existing non-background cells."
    - "The generator reuses the original output-first validation, so only inputs whose solver reconstruction exactly matches the intended output survive."

  e_visual_observability:
    - "A human can now see that bridges are not monochrome; each bridge has one color near the source frame and another near the target frame."
    - "Because the control row already specifies visit order, the two-color bridge can be inferred as 'follow sequence and blend from current color to next color'."
    - "Minimum bridge length 2 ensures both colors are visible within a single bridge."

  f_visual_cleanliness:
    - "The bridge still occupies a simple rectangle, so the grid does not become noisy."
    - "The split is along the long axis only, preserving a clean band instead of checkerboard or pixelwise noise."

  g_color_inference:
    - "No hidden numeric ordering of color ids is used."
    - "The only color logic is visible adjacency: the bridge side touching a square copies that square's inner color."
    - "There are not too many bridge states: each bridge uses exactly the two colors of the squares it connects."

  h_variant_validation_metric:
    - "A generated sample is non-degenerate if at least one bridge would differ from the original single-color solver."
    - "Operationally, with gap >= 2 and distinct endpoint colors, every bridge in accepted samples differs from the original behavior."

  i_new_main_assumptions_and_generator_guarantees:
    - "main() now assumes each usable bridge has a well-defined travel axis and an identifiable source-side half."
    - "generate_input() guarantees this by keeping the same row/column alignment rule and by constructing the visitation path explicitly."
    - "main() also behaves sensibly on length-1 bridges, but generate_input() avoids them so the D5 behavior remains observable."
"""


def _frame_center(frame: dict) -> tuple[int, int]:
    r0, c0, r1, c1 = map(int, frame["bbox"])
    return int((r0 + r1) // 2), int((c0 + c1) // 2)


def _aligned(f1: dict, f2: dict) -> tuple[bool, bool]:
    a = f1["inner_bbox"]
    b = f2["inner_bbox"]
    same_row = bool(int(a[0]) == int(b[0]) and int(a[2]) == int(b[2]))
    same_col = bool(int(a[1]) == int(b[1]) and int(a[3]) == int(b[3]))
    return same_row, same_col


def _bridge_spec(fa: dict, fb: dict) -> dict | None:
    (r0a, c0a, r1a, c1a) = map(int, fa["bbox"])
    (r0b, c0b, r1b, c1b) = map(int, fb["bbox"])
    ia = fa["inner_bbox"]
    ib = fb["inner_bbox"]

    if int(ia[0]) == int(ib[0]) and int(ia[2]) == int(ib[2]):
        rr0, rr1 = int(ia[0]), int(ia[2])
        if int(c1a) < int(c0b):
            return {
                "axis": "h",
                "rr0": int(rr0),
                "rr1": int(rr1),
                "cc0": int(c1a + 1),
                "cc1": int(c0b - 1),
                "start_positive": True,
            }
        if int(c1b) < int(c0a):
            return {
                "axis": "h",
                "rr0": int(rr0),
                "rr1": int(rr1),
                "cc0": int(c1b + 1),
                "cc1": int(c0a - 1),
                "start_positive": False,
            }

    if int(ia[1]) == int(ib[1]) and int(ia[3]) == int(ib[3]):
        cc0, cc1 = int(ia[1]), int(ia[3])
        if int(r1a) < int(r0b):
            return {
                "axis": "v",
                "rr0": int(r1a + 1),
                "rr1": int(r0b - 1),
                "cc0": int(cc0),
                "cc1": int(cc1),
                "start_positive": True,
            }
        if int(r1b) < int(r0a):
            return {
                "axis": "v",
                "rr0": int(r1b + 1),
                "rr1": int(r0a - 1),
                "cc0": int(cc0),
                "cc1": int(cc1),
                "start_positive": False,
            }

    return None


def _paint_rect(g: np.ndarray, spec: dict, color: int, *, part: str) -> None:
    if spec["axis"] == "h":
        if part == "start":
            length = int(spec["cc1"] - spec["cc0"] + 1)
            start_len = int((length + 1) // 2)
            if spec["start_positive"]:
                cc0 = int(spec["cc0"])
                cc1 = int(spec["cc0"] + start_len - 1)
            else:
                cc0 = int(spec["cc1"] - start_len + 1)
                cc1 = int(spec["cc1"])
        else:
            length = int(spec["cc1"] - spec["cc0"] + 1)
            start_len = int((length + 1) // 2)
            if spec["start_positive"]:
                cc0 = int(spec["cc0"] + start_len)
                cc1 = int(spec["cc1"])
            else:
                cc0 = int(spec["cc0"])
                cc1 = int(spec["cc1"] - start_len)
        if int(cc0) <= int(cc1):
            g[int(spec["rr0"]) : int(spec["rr1"]) + 1, int(cc0) : int(cc1) + 1] = int(color)
        return

    length = int(spec["rr1"] - spec["rr0"] + 1)
    start_len = int((length + 1) // 2)
    if part == "start":
        if spec["start_positive"]:
            rr0 = int(spec["rr0"])
            rr1 = int(spec["rr0"] + start_len - 1)
        else:
            rr0 = int(spec["rr1"] - start_len + 1)
            rr1 = int(spec["rr1"])
    else:
        if spec["start_positive"]:
            rr0 = int(spec["rr0"] + start_len)
            rr1 = int(spec["rr1"])
        else:
            rr0 = int(spec["rr0"])
            rr1 = int(spec["rr1"] - start_len)
    if int(rr0) <= int(rr1):
        g[int(rr0) : int(rr1) + 1, int(spec["cc0"]) : int(spec["cc1"]) + 1] = int(color)


def _paint_gradient_bridge(g: np.ndarray, fa: dict, fb: dict, start_color: int, end_color: int) -> bool:
    spec = _bridge_spec(fa, fb)
    if spec is None:
        return False
    if spec["axis"] == "h" and int(spec["cc0"]) > int(spec["cc1"]):
        return False
    if spec["axis"] == "v" and int(spec["rr0"]) > int(spec["rr1"]):
        return False
    _paint_rect(g, spec, int(start_color), part="start")
    _paint_rect(g, spec, int(end_color), part="end")
    return True


def _bridge_cells(fa: dict, fb: dict) -> set[tuple[int, int]]:
    spec = _bridge_spec(fa, fb)
    if spec is None:
        return set()
    if spec["axis"] == "h":
        if int(spec["cc0"]) > int(spec["cc1"]):
            return set()
        return {
            (int(r), int(c))
            for r in range(int(spec["rr0"]), int(spec["rr1"]) + 1)
            for c in range(int(spec["cc0"]), int(spec["cc1"]) + 1)
        }
    if int(spec["rr0"]) > int(spec["rr1"]):
        return set()
    return {
        (int(r), int(c))
        for r in range(int(spec["rr0"]), int(spec["rr1"]) + 1)
        for c in range(int(spec["cc0"]), int(spec["cc1"]) + 1)
    }


def _bridge_span_len(fa: dict, fb: dict) -> int:
    spec = _bridge_spec(fa, fb)
    if spec is None:
        return 0
    if spec["axis"] == "h":
        return max(0, int(spec["cc1"] - spec["cc0"] + 1))
    return max(0, int(spec["rr1"] - spec["rr0"] + 1))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect framed squares via ordered, direction-aware two-color bridges.

    Input structure:
    - bg = background.
    - Border color forms equal-sized framed squares with unique inner fill colors.
    - Control row below squares: colored pixels read left-to-right encode visitation order.

    Rule:
    - Start at the square matching the first control color c0.
    - For each consecutive pair (ci, ci+1), find the ci+1-colored square aligned in the same row or column band, preferring nearest.
    - Draw the bridge rectangle between the two frames, but split it along the travel axis:
      the half adjacent to the current frame uses ci, and the half adjacent to the next frame uses ci+1.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    _ = (int(H), int(W))
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
    for r in range(int(g.shape[0] - 1), int(max_frame_r), -1):
        if int((g[int(r)] != int(bg)).sum()) >= 2:
            ctrl_r = int(r)
            break
    assert not (ctrl_r is None)

    ctrl = [int(g[int(ctrl_r), int(c)]) for c in range(int(g.shape[1])) if int(g[int(ctrl_r), int(c)]) != int(bg)]
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
                (r1_, c1_) = _frame_center(f)
                (r2_, c2_) = _frame_center(g2)
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
            (r1_, c1_) = _frame_center(cur)
            (r2_, c2_) = _frame_center(f)
            d = abs(int(r1_) - int(r2_)) + abs(int(c1_) - int(c2_))
            cand = (int(al), -int(d))
            if best is None or cand > best:
                best = cand
                bestf = f
        nxt = bestf
        if nxt is None:
            continue
        _paint_gradient_bridge(out, cur, nxt, int(a), int(b))
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


_DIGITS = [int(i) for i in range(10)]


def _gen_attempt() -> np.ndarray | None:
    bg = int(np.random.choice(np.array(_DIGITS, dtype=int)))
    border = int(np.random.choice(np.array([d for d in _DIGITS if int(d) != int(bg)], dtype=int)))
    inner_palette = [int(d) for d in _DIGITS if int(d) not in (int(bg), int(border))]
    if len(inner_palette) < 3:
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

    # Keep every bridge long enough to display both source and destination colors.
    row_gap = [int(np.random.randint(2, 6)) for _ in range(int(R) - 1)]
    col_gap = [int(np.random.randint(2, 6)) for _ in range(int(C) - 1)]
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
    saw_gradient_bridge = False
    ok = True
    for a_slot, b_slot in zip(path, path[1:]):
        fa = slot_to_frame[(int(a_slot[0]), int(a_slot[1]))]
        fb = slot_to_frame[(int(b_slot[0]), int(b_slot[1]))]
        cells = _bridge_cells(fa, fb)
        if not cells:
            ok = False
            break
        if int(colors_by_slot[(int(a_slot[0]), int(a_slot[1]))]) == int(colors_by_slot[(int(b_slot[0]), int(b_slot[1]))]):
            ok = False
            break
        if int(_bridge_span_len(fa, fb)) < 2:
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
        if not _paint_gradient_bridge(
            out,
            fa,
            fb,
            int(colors_by_slot[(int(a_slot[0]), int(a_slot[1]))]),
            int(colors_by_slot[(int(b_slot[0]), int(b_slot[1]))]),
        ):
            ok = False
            break
        bridge_cells_all |= set(cells)
        saw_gradient_bridge = True

    if not ok or not saw_gradient_bridge:
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

    return inp


def generate_input() -> np.ndarray:
    for _ in range(500):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("Failed to generate valid input after 500 attempts")
