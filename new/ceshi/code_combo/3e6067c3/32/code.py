# Combo variant of 3e6067c3 (index 32)
# combo_spec: D13/0+D8/1
# Name: 后绘覆盖前绘+反向处理

from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np

from arc_utils import get_bg, label_mask

"""
Design analysis for the combo variant

a) Concrete behavior and trace
- Keep the original visual vocabulary: same-sized framed squares above, one control row below.
- Each consecutive control-color pair still defines one bridge, and bridge color is still the left
  color of that pair.
- D13/0 contributes ordered overwrite: if two bridges share cells, the bridge painted later wins.
- The fusion spec changes the processing order itself: instead of walking pairs left-to-right with a
  running current frame, the solver resolves every pair independently and paints the pairs from the
  rightmost pair back to the leftmost pair.
- Example with five control colors [A,B,C,D,E]: the original pair order is AB, BC, CD, DE; this
  combo paints DE first, then CD, then BC, then AB. If AB is a long horizontal bridge and DE is a
  long vertical bridge crossing the same empty patch, the combo leaves the crossing colored A,
  because AB is painted last. In the original task the same crossing would end with color D.

b) Complexity calibration
- Colors inside frames are made globally unique in generated samples. That removes ambiguity about
  which frame a control color refers to, so the reasoning burden stays on geometric alignment plus
  order-sensitive overwrite.
- The scaffold uses only five frames and one forced crossing. That is enough to make the new rule
  visible without creating a visually noisy tangle of bridges.

c) Implicit-assumption audit
- The original `cur` state machine no longer holds: reverse-order painting means a pair cannot rely
  on the endpoint of the previous pair, so `main()` must stop threading a mutable current frame.
- The original `assert cur["inner"] == a` is invalid under independent reverse processing.
- The original generator assumed bridges never overlap. That would hide D13 entirely, so the combo
  generator must allow bridge-on-bridge overlap while still forbidding bridge-on-frame overlap.

d) Placement legality
- Every bridge must lie in a background-only corridor between aligned frames.
- Only bridge-bridge overlap is permitted; frames and the control row must remain untouched.
- The generator first renders the intended fused output, then erases all bridge cells to get input,
  and finally re-runs `main()` to guarantee exact reconstruction.

e) Visual observability
- Each accepted sample contains a real overlap region of at least 2x2 pixels, and the overlapping
  bridges use different colors.
- The generator rejects samples whose fused reverse-order output matches the left-to-right baseline,
  so the reversal is always visible from the I/O pair.

f) Visual cleanliness
- The scene stays sparse: five frames on a 5x5 lattice with small random gaps and only one forced
  crossing. Overlap happens in empty space, not through frame interiors or the control row.

g) Inferability
- There is no hidden numeric color rule. Bridge color still comes directly from the source frame.
- What changes is only the order in which the visible bridge segments are painted, and the overlap
  makes that order legible to a human from a few examples.

h) New solver-side assumptions and generator guarantees
- `main()` now assumes each control color names a frame and each pair has a best aligned match.
  The generator guarantees that by using a fixed aligned template with globally unique frame colors.
- `main()` also assumes overwriting is allowed on existing bridge pixels. The generator guarantees
  any such overwrite happens only on bridge cells, never on frames or the control row.
"""

_DIGITS = [int(i) for i in range(10)]
_OUTER = 4
_THICKNESS = 1
_TEMPLATES: list[list[tuple[int, int]]] = [
    [(1, 0), (1, 4), (0, 4), (0, 2), (4, 2)],
    [(0, 2), (4, 2), (4, 0), (2, 0), (2, 4)],
]


def _center(frame: dict) -> tuple[int, int]:
    r0, c0, r1, c1 = map(int, frame["bbox"])
    return int((r0 + r1) // 2), int((c0 + c1) // 2)


def _aligned(frame_a: dict, frame_b: dict) -> tuple[bool, bool]:
    a = frame_a["inner_bbox"]
    b = frame_b["inner_bbox"]
    same_row = bool(int(a[0]) == int(b[0]) and int(a[2]) == int(b[2]))
    same_col = bool(int(a[1]) == int(b[1]) and int(a[3]) == int(b[3]))
    return same_row, same_col


def _pair_order_reverse(n_ctrl: int) -> list[int]:
    return [int(i) for i in range(int(n_ctrl) - 2, -1, -1)]


def _pair_order_from_ends(n_ctrl: int) -> list[int]:
    order: list[int] = []
    left = 0
    right = int(n_ctrl) - 2
    while left <= right:
        order.append(int(left))
        left += 1
        if left <= right:
            order.append(int(right))
            right -= 1
    return order


def _most_common_nonbg(grid: np.ndarray, bg: int) -> int | None:
    cnt = Counter(map(int, grid.flatten()))
    for col, _k in cnt.most_common():
        if int(col) != int(bg):
            return int(col)
    return None


def _draw_frame(grid: np.ndarray, *, r0: int, c0: int, outer: int, t: int, border: int, inner_col: int) -> None:
    grid[int(r0) : int(r0 + outer), int(c0) : int(c0 + outer)] = int(border)
    grid[int(r0 + t) : int(r0 + outer - t), int(c0 + t) : int(c0 + outer - t)] = int(inner_col)


def _bridge_cells(frame_a: dict, frame_b: dict) -> set[tuple[int, int]]:
    r0a, c0a, r1a, c1a = map(int, frame_a["bbox"])
    r0b, c0b, r1b, c1b = map(int, frame_b["bbox"])
    ia = frame_a["inner_bbox"]
    ib = frame_b["inner_bbox"]
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


def _paint_bridge(grid: np.ndarray, frame_a: dict, frame_b: dict, color: int, *, overwrite: bool, bg: int) -> None:
    for r, c in _bridge_cells(frame_a, frame_b):
        if overwrite or int(grid[int(r), int(c)]) == int(bg):
            grid[int(r), int(c)] = int(color)


def _detect_frames(grid: np.ndarray, border: int) -> list[dict]:
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


def _best_pair(by_color: dict[int, list[dict]], color_a: int, color_b: int) -> tuple[dict, dict] | None:
    cands_a = by_color.get(int(color_a), [])
    cands_b = by_color.get(int(color_b), [])
    best = None
    best_pair = None
    for frame_a in cands_a:
        for frame_b in cands_b:
            if frame_a is frame_b:
                continue
            same_row, same_col = _aligned(frame_a, frame_b)
            aligned = 1 if (same_row or same_col) else 0
            cells = _bridge_cells(frame_a, frame_b)
            usable = 1 if cells else 0
            ra, ca = _center(frame_a)
            rb, cb = _center(frame_b)
            dist = abs(int(ra) - int(rb)) + abs(int(ca) - int(cb))
            cand = (int(usable), int(aligned), -int(dist))
            if best is None or cand > best:
                best = cand
                best_pair = (frame_a, frame_b)
    return best_pair


def _control_colors(grid: np.ndarray, bg: int, frames: list[dict]) -> list[int]:
    max_frame_r = max(int(frame["bbox"][2]) for frame in frames)
    ctrl_r = None
    for r in range(int(grid.shape[0] - 1), int(max_frame_r), -1):
        if int((grid[int(r)] != int(bg)).sum()) >= 2:
            ctrl_r = int(r)
            break
    assert ctrl_r is not None
    ctrl = [int(grid[int(ctrl_r), int(c)]) for c in range(int(grid.shape[1])) if int(grid[int(ctrl_r), int(c)]) != int(bg)]
    assert len(ctrl) >= 2
    return ctrl


def main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
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

    by_color: dict[int, list[dict]] = defaultdict(list)
    for frame in frames:
        by_color[int(frame["inner"])].append(frame)

    ctrl = _control_colors(g, int(bg), frames)

    out = g.copy()
    for idx in _pair_order_reverse(len(ctrl)):
        color_a = int(ctrl[int(idx)])
        color_b = int(ctrl[int(idx) + 1])
        pair = _best_pair(by_color, int(color_a), int(color_b))
        if pair is None:
            continue
        frame_a, frame_b = pair
        _paint_bridge(out, frame_a, frame_b, int(color_a), overwrite=True, bg=int(bg))
    return out


def _transform_slots(slots: list[tuple[int, int]], mode: int, size: int = 5) -> list[tuple[int, int]]:
    max_idx = int(size - 1)
    out: list[tuple[int, int]] = []
    for r, c in slots:
        r = int(r)
        c = int(c)
        if mode == 0:
            nr, nc = r, c
        elif mode == 1:
            nr, nc = r, int(max_idx - c)
        elif mode == 2:
            nr, nc = int(max_idx - r), c
        elif mode == 3:
            nr, nc = int(max_idx - r), int(max_idx - c)
        elif mode == 4:
            nr, nc = c, r
        elif mode == 5:
            nr, nc = c, int(max_idx - r)
        elif mode == 6:
            nr, nc = int(max_idx - c), r
        elif mode == 7:
            nr, nc = int(max_idx - c), int(max_idx - r)
        else:
            raise ValueError(mode)
        out.append((int(nr), int(nc)))
    return out


def _render_output(
    *,
    scaffold: np.ndarray,
    frames: list[dict],
    colors: list[int],
    order: list[int],
    overwrite: bool,
    bg: int,
) -> tuple[np.ndarray, list[set[tuple[int, int]]], Counter[tuple[int, int]]]:
    out = np.array(scaffold, dtype=int, copy=True)
    pair_cells: list[set[tuple[int, int]]] = []
    paint_counts: Counter[tuple[int, int]] = Counter()
    for idx in range(int(len(colors)) - 1):
        cells = _bridge_cells(frames[int(idx)], frames[int(idx) + 1])
        pair_cells.append(set(cells))
    for idx in order:
        _paint_bridge(out, frames[int(idx)], frames[int(idx) + 1], int(colors[int(idx)]), overwrite=overwrite, bg=int(bg))
        for cell in pair_cells[int(idx)]:
            paint_counts[(int(cell[0]), int(cell[1]))] += 1
    return out, pair_cells, paint_counts


def _gen_attempt() -> np.ndarray | None:
    bg = int(np.random.choice(np.array(_DIGITS, dtype=int)))
    border = int(np.random.choice(np.array([d for d in _DIGITS if int(d) != int(bg)], dtype=int)))
    inner_palette = [int(d) for d in _DIGITS if int(d) not in (int(bg), int(border))]

    template = _TEMPLATES[int(np.random.randint(0, len(_TEMPLATES)))]
    slots = _transform_slots(template, int(np.random.randint(0, 8)))
    n_frames = len(slots)
    if len(inner_palette) < int(n_frames):
        return None
    colors = [int(x) for x in np.random.choice(np.array(inner_palette, dtype=int), size=int(n_frames), replace=False)]

    lattice_rows = 5
    lattice_cols = 5
    top_margin = int(np.random.randint(0, 2))
    left_margin = int(np.random.randint(0, 2))
    right_margin = int(np.random.randint(0, 3))
    row_gap = [int(np.random.randint(1, 3)) for _ in range(int(lattice_rows) - 1)]
    col_gap = [int(np.random.randint(1, 3)) for _ in range(int(lattice_cols) - 1)]

    r0s = [int(top_margin)]
    for gap in row_gap:
        r0s.append(int(r0s[-1] + _OUTER + int(gap)))
    c0s = [int(left_margin)]
    for gap in col_gap:
        c0s.append(int(c0s[-1] + _OUTER + int(gap)))

    frames: list[dict] = []
    for idx, (rr, cc) in enumerate(slots):
        rr = int(rr)
        cc = int(cc)
        r0 = int(r0s[int(rr)])
        c0 = int(c0s[int(cc)])
        r1 = int(r0 + _OUTER - 1)
        c1 = int(c0 + _OUTER - 1)
        frames.append(
            {
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
                "inner_bbox": (int(r0 + _THICKNESS), int(c0 + _THICKNESS), int(r1 - _THICKNESS), int(c1 - _THICKNESS)),
                "t": int(_THICKNESS),
                "inner": int(colors[int(idx)]),
                "slot": (int(rr), int(cc)),
            }
        )

    max_r1 = max(int(frame["bbox"][2]) for frame in frames)
    max_c1 = max(int(frame["bbox"][3]) for frame in frames)
    ctrl_gap = int(np.random.randint(2, 5))
    ctrl_r = int(max_r1 + ctrl_gap)
    bottom_margin = 1
    ctrl_cols = [int(1 + 2 * i) for i in range(int(n_frames))]
    H = int(ctrl_r + 1 + bottom_margin)
    W = int(max(int(max_c1 + 1 + right_margin), int(ctrl_cols[-1] + 2)))
    if not (7 <= int(H) <= 30 and 7 <= int(W) <= 30):
        return None

    scaffold = np.full((int(H), int(W)), int(bg), dtype=int)
    for frame, color in zip(frames, colors):
        r0, c0, r1, _c1 = map(int, frame["bbox"])
        _draw_frame(scaffold, r0=int(r0), c0=int(c0), outer=int(r1 - r0 + 1), t=int(frame["t"]), border=int(border), inner_col=int(color))
    for col, color in zip(ctrl_cols, colors):
        scaffold[int(ctrl_r), int(col)] = int(color)

    pair_cells: list[set[tuple[int, int]]] = []
    for idx in range(int(n_frames) - 1):
        cells = _bridge_cells(frames[int(idx)], frames[int(idx) + 1])
        if not cells:
            return None
        for r, c in cells:
            if not (0 <= int(r) < int(H) and 0 <= int(c) < int(W)):
                return None
            if int(scaffold[int(r), int(c)]) != int(bg):
                return None
        pair_cells.append(set(cells))

    reverse_order = _pair_order_reverse(len(colors))
    ltr_order = [int(i) for i in range(int(len(colors)) - 1)]
    ends_order = _pair_order_from_ends(len(colors))

    fused_out, _, fused_counts = _render_output(
        scaffold=scaffold,
        frames=frames,
        colors=colors,
        order=reverse_order,
        overwrite=True,
        bg=int(bg),
    )
    ltr_out, _, _ = _render_output(
        scaffold=scaffold,
        frames=frames,
        colors=colors,
        order=ltr_order,
        overwrite=True,
        bg=int(bg),
    )
    ends_out, _, _ = _render_output(
        scaffold=scaffold,
        frames=frames,
        colors=colors,
        order=ends_order,
        overwrite=True,
        bg=int(bg),
    )
    no_overwrite_out, _, _ = _render_output(
        scaffold=scaffold,
        frames=frames,
        colors=colors,
        order=reverse_order,
        overwrite=False,
        bg=int(bg),
    )

    overlap_pixels = sum(1 for count in fused_counts.values() if int(count) > 1)
    if overlap_pixels <= 0:
        return None
    if bool(np.array_equal(fused_out, ltr_out)):
        return None
    if bool(np.array_equal(fused_out, ends_out)):
        return None
    if bool(np.array_equal(fused_out, no_overwrite_out)):
        return None

    bridge_union: set[tuple[int, int]] = set()
    for cells in pair_cells:
        bridge_union |= set(cells)

    inp = scaffold.copy()
    for r, c in bridge_union:
        inp[int(r), int(c)] = int(bg)

    if int(get_bg(inp)) != int(bg):
        return None
    if int(_most_common_nonbg(inp, int(bg)) or -1) != int(border):
        return None

    try:
        pred = main(inp)
    except Exception:
        return None
    if pred.shape != fused_out.shape:
        return None
    if not bool(np.array_equal(pred, fused_out)):
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(400):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("Failed to generate valid combo input after 400 attempts")
