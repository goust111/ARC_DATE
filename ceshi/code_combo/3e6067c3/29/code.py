# Combo variant of 3e6067c3 (index 29)
# combo_spec: D2/2+D5/3+D8/1
# Name: 行列交替+渐变桥梁+反向处理

from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np

from arc_utils import get_bg, label_mask


"""
Design analysis for the combo variant
=====================================

Authoritative fused rule
- The top-level combo description is taken as authoritative where the per-dimension
  snippets are inconsistent: control pixels are read right-to-left, the resulting
  step index parity alternates the required alignment (odd step = same row band,
  even step = same column band), and each bridge is a source-to-target two-color
  gradient.

a) Concrete behavior and trace
- Suppose the visible control row from left to right is [E, D, C, B, A]. The solver
  reads it right-to-left as [A, B, C, D, E].
- Step 1 is therefore A->B and must use a same-row B candidate. Step 2 is B->C and
  must use a same-column C candidate. Step 3 returns to same-row, step 4 to
  same-column.
- In the generated scenes every non-start path color has two framed squares: one
  placed on the row that matches the intended odd step, and one placed on the column
  that would match the opposite parity. Reading the control row in the wrong
  direction changes the step numbering, which changes which duplicate square is
  selected.
- Each chosen straight bridge is split along its travel axis: the half touching the
  source frame uses the source color, and the half touching the destination frame
  uses the destination color.

b) Complexity calibration
- A unique-color path would make the row/column alternation invisible, so colors
- B/C/D/E are duplicated with structured decoys.
- Allowing arbitrary nearest-neighbor L bridges would add too much freedom for this
  fusion, so the generator keeps bridges straight and makes the disambiguation happen
  at target selection time.
- Only five control colors are used. This is enough to show four alternating steps
  without making the scene noisy.

c) Hidden assumption audit
- Original code assumed left-to-right control decoding. That no longer holds.
- Original code assumed a single alignment preference independent of step index. That
  breaks because alignment is now parity-dependent.
- Original bridge painting assumed a single solid color. That breaks because bridge
  direction now matters for the color split.
- The new solver assumes every processed pair has at least one candidate of the
  required alignment and that ties can be broken deterministically by distance and
  bbox order. The generator enforces this.

d) Placement legality
- Frames are still equal-sized, non-overlapping, and axis-aligned.
- The generator paints the full intended output first, checks every bridge rectangle
  against bounds and pre-existing non-background cells, then erases the bridge cells
  to form the input.
- Randomization is restricted to frame size, gaps, margins, colors, and axis-
  preserving reflections so the legality checks stay simple and reliable.

e) Visual observability
- Reverse reading is visible because the control row is written in the opposite
  direction of the actual traversal.
- Parity is visible because the chosen duplicate alternates between a horizontal
  continuation and a vertical continuation.
- The gradient is visible because every bridge gap is forced to have length >= 2,
  making both endpoint colors appear on the bridge itself.

f) Visual cleanliness
- The scene uses a fixed 3x4 slot skeleton: a staircase path plus one decoy per
  duplicated color. Bridges do not overlap frames or each other.
- The only duplicated information is color, not shape, which keeps the puzzle
  readable.

g) Color-rule inferability
- No numeric property of the color ids is used.
- A bridge always contains exactly the two colors of the squares it connects, and the
  side touching each frame matches that frame's inner color.
- Because the same two-zone bridge pattern repeats across multiple steps, humans can
  infer the gradient rule from a few examples.

h) Anti-degeneration rule
- A sample is rejected unless the fused output differs from a left-to-right baseline
  and from a right-to-left baseline that ignores the parity alignment rule. This
  ensures the reverse-reading and alternating-axis constraints are both active.

i) New solver assumptions and generator guarantees
- Step 1 is defined after right-to-left decoding. The generator writes the control row
  in the opposite left-to-right order.
- Odd steps need row-aligned targets and even steps need column-aligned targets. The
  staircase skeleton guarantees exactly one intended candidate per step while still
  providing a wrong-parity decoy of the same color.
- Bridges are always straight, axis-aligned rectangles with length at least 2, so the
  gradient split is observable and unambiguous.
"""


def _bbox_key(frame: dict) -> tuple[int, int, int, int]:
    return tuple(map(int, frame["bbox"]))


def _center(frame: dict) -> tuple[int, int]:
    r0, c0, r1, c1 = map(int, frame["bbox"])
    return int((r0 + r1) // 2), int((c0 + c1) // 2)


def _manhattan(frame_a: dict, frame_b: dict) -> int:
    ra, ca = _center(frame_a)
    rb, cb = _center(frame_b)
    return abs(int(ra) - int(rb)) + abs(int(ca) - int(cb))


def _aligned(frame_a: dict, frame_b: dict) -> tuple[bool, bool]:
    a = frame_a["inner_bbox"]
    b = frame_b["inner_bbox"]
    same_row = bool(int(a[0]) == int(b[0]) and int(a[2]) == int(b[2]))
    same_col = bool(int(a[1]) == int(b[1]) and int(a[3]) == int(b[3]))
    return same_row, same_col


def _extract_frames(grid: np.ndarray, border: int) -> list[dict]:
    lab, n = label_mask(grid == int(border), connectivity=4)
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

        thick = None
        inner_col = None
        for tt in range(1, int(min(h, w) // 2) + 1):
            inner = grid[int(r0 + tt) : int(r1 - tt) + 1, int(c0 + tt) : int(c1 - tt) + 1]
            if inner.size == 0:
                continue
            if bool((inner == int(border)).any()):
                continue
            vals = set(map(int, inner.flatten()))
            if len(vals) != 1:
                continue

            ok = True
            if not bool((grid[int(r0) : int(r0 + tt), int(c0) : int(c1) + 1] == int(border)).all()):
                ok = False
            if not bool((grid[int(r1 - tt) + 1 : int(r1) + 1, int(c0) : int(c1) + 1] == int(border)).all()):
                ok = False
            if not bool((grid[int(r0) : int(r1) + 1, int(c0) : int(c0 + tt)] == int(border)).all()):
                ok = False
            if not bool((grid[int(r0) : int(r1) + 1, int(c1 - tt) + 1 : int(c1) + 1] == int(border)).all()):
                ok = False
            if not ok:
                continue

            thick = int(tt)
            inner_col = int(next(iter(vals)))
            break

        if thick is None or inner_col is None:
            continue

        frames.append(
            {
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
                "t": int(thick),
                "inner": int(inner_col),
                "inner_bbox": (int(r0 + thick), int(c0 + thick), int(r1 - thick), int(c1 - thick)),
            }
        )
    return frames


def _control_colors(grid: np.ndarray, bg: int, frames: list[dict], *, reverse: bool) -> list[int]:
    max_frame_r = max(int(frame["bbox"][2]) for frame in frames)
    ctrl_r = None
    for r in range(int(grid.shape[0] - 1), int(max_frame_r), -1):
        if int((grid[int(r)] != int(bg)).sum()) >= 2:
            ctrl_r = int(r)
            break
    assert ctrl_r is not None

    if reverse:
        cols = range(int(grid.shape[1]) - 1, -1, -1)
    else:
        cols = range(int(grid.shape[1]))
    ctrl = [int(grid[int(ctrl_r), int(c)]) for c in cols if int(grid[int(ctrl_r), int(c)]) != int(bg)]
    assert len(ctrl) >= 2
    return ctrl


def _bridge_spec(frame_a: dict, frame_b: dict) -> dict | None:
    r0a, c0a, r1a, c1a = map(int, frame_a["bbox"])
    r0b, c0b, r1b, c1b = map(int, frame_b["bbox"])
    ia = frame_a["inner_bbox"]
    ib = frame_b["inner_bbox"]

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


def _bridge_cells(frame_a: dict, frame_b: dict) -> set[tuple[int, int]]:
    spec = _bridge_spec(frame_a, frame_b)
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


def _paint_half(grid: np.ndarray, spec: dict, color: int, *, part: str) -> None:
    if spec["axis"] == "h":
        length = int(spec["cc1"] - spec["cc0"] + 1)
        start_len = int((length + 1) // 2)
        if part == "start":
            if spec["start_positive"]:
                cc0 = int(spec["cc0"])
                cc1 = int(spec["cc0"] + start_len - 1)
            else:
                cc0 = int(spec["cc1"] - start_len + 1)
                cc1 = int(spec["cc1"])
        else:
            if spec["start_positive"]:
                cc0 = int(spec["cc0"] + start_len)
                cc1 = int(spec["cc1"])
            else:
                cc0 = int(spec["cc0"])
                cc1 = int(spec["cc1"] - start_len)
        if int(cc0) <= int(cc1):
            grid[int(spec["rr0"]) : int(spec["rr1"]) + 1, int(cc0) : int(cc1) + 1] = int(color)
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
        grid[int(rr0) : int(rr1) + 1, int(spec["cc0"]) : int(spec["cc1"]) + 1] = int(color)


def _paint_gradient_bridge(grid: np.ndarray, frame_a: dict, frame_b: dict, start_color: int, end_color: int) -> bool:
    spec = _bridge_spec(frame_a, frame_b)
    if spec is None:
        return False
    if spec["axis"] == "h" and int(spec["cc0"]) > int(spec["cc1"]):
        return False
    if spec["axis"] == "v" and int(spec["rr0"]) > int(spec["rr1"]):
        return False
    _paint_half(grid, spec, int(start_color), part="start")
    _paint_half(grid, spec, int(end_color), part="end")
    return True


def _pick_target(src: dict, cands: list[dict], *, want_row: bool) -> dict | None:
    filtered: list[dict] = []
    for cand in cands:
        same_row, same_col = _aligned(src, cand)
        if want_row and same_row:
            filtered.append(cand)
        if (not want_row) and same_col:
            filtered.append(cand)
    if not filtered:
        return None
    best = None
    bestf = None
    for cand in filtered:
        score = (int(_manhattan(src, cand)),) + _bbox_key(cand)
        if best is None or score < best:
            best = score
            bestf = cand
    return bestf


def _pick_start_frame(ctrl: list[int], by_color: dict[int, list[dict]], *, parity_rule: bool) -> dict:
    start_color = int(ctrl[0])
    start_cands = by_color[start_color]
    if len(start_cands) == 1 or len(ctrl) < 2:
        return min(start_cands, key=_bbox_key)

    next_color = int(ctrl[1])
    next_cands = by_color.get(int(next_color), [])
    best = None
    bestf = None
    for cand in start_cands:
        if parity_rule:
            nxt = _pick_target(cand, next_cands, want_row=True)
            if nxt is None:
                score = (1, 10**9) + _bbox_key(cand)
            else:
                score = (0, int(_manhattan(cand, nxt))) + _bbox_key(cand)
        else:
            if not next_cands:
                score = (1, 10**9) + _bbox_key(cand)
            else:
                nearest = min((int(_manhattan(cand, nxt)),) + _bbox_key(nxt) for nxt in next_cands)
                score = (0, int(nearest[0])) + _bbox_key(cand)
        if best is None or score < best:
            best = score
            bestf = cand
    assert bestf is not None
    return bestf


def _solve_with_mode(grid: np.ndarray, *, reverse_ctrl: bool, parity_rule: bool, gradient: bool) -> np.ndarray:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    cnt = Counter(map(int, g.flatten()))
    border = None
    for col, _k in cnt.most_common():
        if int(col) != int(bg):
            border = int(col)
            break
    assert border is not None

    frames = _extract_frames(g, int(border))
    assert frames

    by_color: dict[int, list[dict]] = defaultdict(list)
    for frame in frames:
        by_color[int(frame["inner"])].append(frame)

    ctrl = _control_colors(g, int(bg), frames, reverse=bool(reverse_ctrl))

    start_color = int(ctrl[0])
    assert start_color in by_color
    cur = _pick_start_frame(ctrl, by_color, parity_rule=bool(parity_rule))

    out = g.copy()
    for step_idx in range(1, int(len(ctrl))):
        color_a = int(ctrl[int(step_idx - 1)])
        color_b = int(ctrl[int(step_idx)])
        assert int(cur["inner"]) == int(color_a)
        cands = by_color.get(int(color_b), [])
        if not cands:
            continue

        if parity_rule:
            want_row = bool(step_idx % 2 == 1)
            nxt = _pick_target(cur, cands, want_row=bool(want_row))
        else:
            best = None
            nxt = None
            for cand in cands:
                score = (int(_manhattan(cur, cand)),) + _bbox_key(cand)
                if best is None or score < best:
                    best = score
                    nxt = cand
        if nxt is None:
            continue

        if gradient:
            ok = _paint_gradient_bridge(out, cur, nxt, int(color_a), int(color_b))
            assert ok
        else:
            for r, c in _bridge_cells(cur, nxt):
                out[int(r), int(c)] = int(color_a)
        cur = nxt

    return out


def main(grid: np.ndarray) -> np.ndarray:
    return _solve_with_mode(grid, reverse_ctrl=True, parity_rule=True, gradient=True)


def _most_common_nonbg(grid: np.ndarray, bg: int) -> int | None:
    cnt = Counter(map(int, grid.flatten()))
    for col, _k in cnt.most_common():
        if int(col) != int(bg):
            return int(col)
    return None


def _draw_frame(grid: np.ndarray, *, r0: int, c0: int, outer: int, t: int, border: int, inner_col: int) -> None:
    grid[int(r0) : int(r0 + outer), int(c0) : int(c0 + outer)] = int(border)
    grid[int(r0 + t) : int(r0 + outer - t), int(c0 + t) : int(c0 + outer - t)] = int(inner_col)


def _transform_slot(slot: tuple[int, int], mode: int) -> tuple[int, int]:
    r, c = map(int, slot)
    if int(mode) == 0:
        return int(r), int(c)
    if int(mode) == 1:
        return int(r), int(3 - c)
    if int(mode) == 2:
        return int(2 - r), int(c)
    return int(2 - r), int(3 - c)


def _slot_frame(slot: tuple[int, int], r0s: list[int], c0s: list[int], outer: int, t: int) -> dict:
    rr, cc = map(int, slot)
    r0 = int(r0s[int(rr)])
    c0 = int(c0s[int(cc)])
    r1 = int(r0 + outer - 1)
    c1 = int(c0 + outer - 1)
    return {
        "bbox": (int(r0), int(c0), int(r1), int(c1)),
        "inner_bbox": (int(r0 + t), int(c0 + t), int(r1 - t), int(c1 - t)),
        "t": int(t),
        "slot": (int(rr), int(cc)),
    }


_DIGITS = [int(i) for i in range(10)]
_BASE_LAYOUT = {
    "A": (0, 0),
    "B": (0, 1),
    "Cp": (0, 2),
    "Bp": (1, 0),
    "C": (1, 1),
    "D": (1, 2),
    "Ep": (1, 3),
    "Dp": (2, 1),
    "E": (2, 2),
}
_PATH_NAMES = ["A", "B", "C", "D", "E"]


def _gen_attempt() -> np.ndarray | None:
    bg = int(np.random.choice(np.array(_DIGITS, dtype=int)))
    border = int(np.random.choice(np.array([d for d in _DIGITS if int(d) != int(bg)], dtype=int)))
    inner_palette = [int(d) for d in _DIGITS if int(d) not in (int(bg), int(border))]
    if len(inner_palette) < 5:
        return None

    t = 1
    inner_side = int(np.random.choice(np.array([1, 2], dtype=int)))
    outer = int(inner_side + 2 * t)

    row_gap = [2, 2]
    col_gap = [3, 3, 3]
    top_margin = int(np.random.randint(1, 3))
    left_margin = int(np.random.randint(1, 3))

    r0s = [int(top_margin)]
    for gap in row_gap:
        r0s.append(int(r0s[-1] + outer + int(gap)))
    c0s = [int(left_margin)]
    for gap in col_gap:
        c0s.append(int(c0s[-1] + outer + int(gap)))

    mode = int(np.random.randint(0, 4))
    frames_by_name: dict[str, dict] = {}
    for name, slot in _BASE_LAYOUT.items():
        frames_by_name[name] = _slot_frame(_transform_slot(slot, int(mode)), r0s, c0s, int(outer), int(t))

    max_r1 = max(int(frame["bbox"][2]) for frame in frames_by_name.values())
    max_c1 = max(int(frame["bbox"][3]) for frame in frames_by_name.values())
    ctrl_gap = int(np.random.randint(2, 5))
    ctrl_r = int(max_r1 + ctrl_gap)
    bottom_margin = 1
    right_margin = int(np.random.randint(1, 3))
    H = int(ctrl_r + 1 + bottom_margin)
    W = int(max_c1 + 1 + right_margin)
    if not (10 <= int(H) <= 30 and 10 <= int(W) <= 30):
        return None

    path_colors = [int(x) for x in np.random.choice(np.array(inner_palette, dtype=int), size=5, replace=False)]
    color_by_name = {
        "A": int(path_colors[0]),
        "B": int(path_colors[1]),
        "Bp": int(path_colors[1]),
        "C": int(path_colors[2]),
        "Cp": int(path_colors[2]),
        "D": int(path_colors[3]),
        "Dp": int(path_colors[3]),
        "E": int(path_colors[4]),
        "Ep": int(path_colors[4]),
    }

    out = np.full((int(H), int(W)), int(bg), dtype=int)
    for name, frame in frames_by_name.items():
        r0, c0, r1, _c1 = map(int, frame["bbox"])
        _draw_frame(
            out,
            r0=int(r0),
            c0=int(c0),
            outer=int(r1 - r0 + 1),
            t=int(frame["t"]),
            border=int(border),
            inner_col=int(color_by_name[name]),
        )

    path_frames = [frames_by_name[name] for name in _PATH_NAMES]
    bridge_union: set[tuple[int, int]] = set()
    for idx in range(4):
        frame_a = path_frames[int(idx)]
        frame_b = path_frames[int(idx + 1)]
        cells = _bridge_cells(frame_a, frame_b)
        if not cells:
            return None
        if len(cells) < 2:
            return None
        if cells & bridge_union:
            return None
        for r, c in cells:
            if not (0 <= int(r) < int(H) and 0 <= int(c) < int(W)):
                return None
            if int(out[int(r), int(c)]) != int(bg):
                return None
        ok = _paint_gradient_bridge(
            out,
            frame_a,
            frame_b,
            int(color_by_name[_PATH_NAMES[int(idx)]]),
            int(color_by_name[_PATH_NAMES[int(idx + 1)]]),
        )
        if not ok:
            return None
        bridge_union |= set(cells)

    ctrl_colors_lr = list(reversed(path_colors))
    ctrl_cols = [int(1 + 2 * i) for i in range(len(ctrl_colors_lr))]
    if int(ctrl_cols[-1]) >= int(W):
        return None
    for col, color in zip(ctrl_cols, ctrl_colors_lr):
        out[int(ctrl_r), int(col)] = int(color)

    inp = out.copy()
    for r, c in bridge_union:
        inp[int(r), int(c)] = int(bg)

    if int(get_bg(inp)) != int(bg):
        return None
    if int(_most_common_nonbg(inp, int(bg)) or -1) != int(border):
        return None

    try:
        pred = main(inp)
        ltr = _solve_with_mode(inp, reverse_ctrl=False, parity_rule=True, gradient=True)
        no_parity = _solve_with_mode(inp, reverse_ctrl=True, parity_rule=False, gradient=True)
    except Exception:
        return None

    if pred.shape != out.shape:
        return None
    if not bool(np.array_equal(pred, out)):
        return None
    if bool(np.array_equal(pred, ltr)):
        return None
    if bool(np.array_equal(pred, no_parity)):
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(500):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("Failed to generate valid input after 500 attempts")
