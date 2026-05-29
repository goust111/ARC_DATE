# Combo variant of 3e6067c3 (index 30)
# combo_spec: D11/1+D8/1
# Name: 环形排列+反向处理

from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np

from arc_utils import get_bg, label_mask

"""
Design analysis for the combo variant

a) Concrete behavior and trace
- Original task: framed squares lie on an irregular orthogonal path. The control row is read left-to-right,
  consecutive control colors define consecutive bridges, and the path is open.
- D11 contributes a topology change: squares now occupy every perimeter slot of a coarse rectangle, so the
  objects form a visible ring and the final square must connect back to the first.
- The fusion description overrides the single D8 reference behavior: here the "reverse" effect is not
  alternating from both ends, but reading the bottom control row from right to left. The single-dimension
  D8 code is still useful because it highlights which original assumptions about control-sequence direction
  must be challenged.
- Concrete trace: suppose the clockwise ring colors are A,B,C,D,E,F and the visible control row is
  [F,E,D,C,B,A] from left to right. Original logic would read that row as F->E->D->C->B->A and then close
  A->F, so each ring edge would be colored by the "counterclockwise" source square. The fused rule must read
  the row right-to-left, recovering A->B->C->D->E->F and then close F->A. The geometry is the same ring, but
  every edge gets the opposite endpoint's color, so the visual result differs around the whole loop.

b) Complexity calibration
- The ring uses 6 or 8 squares on the perimeter of a 2x3, 3x2, 2x4, 4x2, or 3x3 coarse lattice. That keeps
  the cycle obvious without introducing many free placement degrees.
- Inner colors are globally distinct. This avoids ambiguous repeated-color matching, making the intended
  reasoning focus on topology plus reverse control reading.
- The generator rejects samples where left-to-right and right-to-left ring traversals would accidentally
  produce the same output, so the reverse rule remains visually observable.

c) Hidden-assumption review of the original main()
- Original consecutive-pair traversal assumed an open path and stopped after len(ctrl)-1 bridges. That fails
  for a ring, so the combo solver must add the closing bridge from the last visited frame back to the start.
- Original start/next selection ranked all same-colored aligned candidates by proximity. On a ring that is too
  weak because only the two ring neighbors are legal successors. The combo solver reconstructs the ring graph
  and follows only degree-2 ring adjacency.
- Original control-row extraction read non-background pixels left-to-right. The fusion changes that semantic:
  the visible row is still placed left-to-right in the grid, but the logical sequence is the reversed list.
- The single-dimension D8 variant showed that "control sequence handling" can invalidate traversal-state
  assumptions. Here stateful traversal still makes sense because reverse reading simply reverses the cyclic
  order rather than making pairs independent.

d) Placement legality
- Generator places frames only on coarse perimeter slots, leaving the center empty and guaranteeing a single
  degree-2 cycle after frame detection.
- Every neighboring slot pair must have a non-empty bridge corridor, remain inside bounds, and occupy only
  background cells in the input. Output-first validation rejects any collision or degenerate corridor.
- The control row is below all frames, uses odd columns 1,3,5,..., and the canvas width is expanded if needed.

e) Visual observability
- Across examples, the empty center and closing bridge make the ring topology obvious.
- The control row visibly lists colors in the opposite order from the bridge-color flow around the ring, which
  is the human-visible clue that the row is read from right to left.
- Because bridge colors differ between clockwise and counterclockwise traversals, reverse reading changes more
  than just an invisible execution order.

f) Visual cleanliness
- The scene contains only framed squares, background, a single control row, and axis-aligned bridges with the
  same thickness as each frame interior band.
- Adjacent ring bridges occupy only the gaps between neighboring frames, so they do not cover frame interiors
  or clutter the ring hole.

g) Change inferability
- No hidden ordering by numeric color id is used. Colors are matched only by visible equality between control
  pixels and square interiors.
- Distinct inner colors plus a single ring cycle provide enough structure for a human to infer that the
  control row is read in reverse to decide the bridge colors and traversal direction.

9) New assumptions introduced by the combo and how generation guarantees them
- `_ring_neighbors()` assumes the detected frames correspond exactly to a perimeter cycle of a coarse
  rectangle. Generator fills every perimeter slot and validates that the inferred graph is one cycle.
- `_choose_next()` assumes the next control color appears on exactly one legal ring neighbor. Generator uses
  globally distinct inner colors, so each step is unique.
- Reversing the control row only matters if the opposite traversal changes the output. Generator explicitly
  compares right-to-left output against left-to-right output and rejects equal cases.
- The closing bridge assumes the final and start frames are ring neighbors. Because traversal follows the full
  perimeter cycle, the last-to-first pair is always a valid adjacent edge.
"""


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect framed squares along a ring, reading the control row from right to left.

    Input structure:
    - Equal-sized framed squares sit on the perimeter of a coarse rectangle, leaving a hollow center.
    - A control row below the figure contains one full cyclic traversal, but the logical order is right-to-left.

    Rule:
    - Detect the ring of frames and read the visible control pixels from right to left.
    - Start at the frame whose inner color matches the first reversed control color.
    - Follow ring adjacency so each next control color selects the unique neighboring frame.
    - Draw each bridge using the current frame's color, then connect the final frame back to the start.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    border = _most_common_nonbg(g, bg)
    assert border is not None

    frames = _find_frames(g, int(border))
    assert frames

    ctrl = _control_colors(g, int(bg), frames, reverse=True)
    assert len(ctrl) >= 4

    neighbors = _ring_neighbors(g, frames, int(bg))
    start = _choose_start(frames, neighbors, ctrl)

    out = g.copy()
    prev: int | None = None
    cur = int(start)
    for next_color in ctrl[1:]:
        nxt = _choose_next(frames, neighbors, int(cur), prev, int(next_color))
        _draw_bridge(out, frames[int(cur)], frames[int(nxt)], int(frames[int(cur)]["inner"]))
        prev, cur = int(cur), int(nxt)

    _draw_bridge(out, frames[int(cur)], frames[int(start)], int(frames[int(cur)]["inner"]))
    return out


def _most_common_nonbg(g: np.ndarray, bg: int) -> int | None:
    cnt = Counter(map(int, g.flatten()))
    for col, _k in cnt.most_common():
        if int(col) != int(bg):
            return int(col)
    return None


def _find_frames(g: np.ndarray, border: int) -> list[dict]:
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
            vals = set(map(int, inner.flatten()))
            if len(vals) != 1:
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
            inner_col = int(next(iter(vals)))
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


def _aligned(f1: dict, f2: dict) -> tuple[bool, bool]:
    a = f1["inner_bbox"]
    b = f2["inner_bbox"]
    same_row = bool(int(a[0]) == int(b[0]) and int(a[2]) == int(b[2]))
    same_col = bool(int(a[1]) == int(b[1]) and int(a[3]) == int(b[3]))
    return same_row, same_col


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


def _center(f: dict) -> tuple[int, int]:
    r0, c0, r1, c1 = map(int, f["bbox"])
    return int((r0 + r1) // 2), int((c0 + c1) // 2)


def _control_colors(g: np.ndarray, bg: int, frames: list[dict], *, reverse: bool) -> list[int]:
    max_frame_r = max(int(f["bbox"][2]) for f in frames)
    ctrl_r = None
    for r in range(int(g.shape[0] - 1), int(max_frame_r), -1):
        if int((g[int(r)] != int(bg)).sum()) >= 2:
            ctrl_r = int(r)
            break
    assert ctrl_r is not None
    ctrl = [int(g[int(ctrl_r), int(c)]) for c in range(int(g.shape[1])) if int(g[int(ctrl_r), int(c)]) != int(bg)]
    assert len(ctrl) >= 2
    if reverse:
        ctrl = list(reversed(ctrl))
    return ctrl


def _perimeter_cycle(R: int, C: int) -> list[tuple[int, int]]:
    order: list[tuple[int, int]] = []
    for c in range(int(C)):
        order.append((0, int(c)))
    for r in range(1, int(R)):
        order.append((int(r), int(C - 1)))
    for c in range(int(C - 2), -1, -1):
        order.append((int(R - 1), int(c)))
    for r in range(int(R - 2), 0, -1):
        order.append((int(r), 0))
    return order


def _ring_neighbors(g: np.ndarray, frames: list[dict], bg: int) -> dict[int, list[int]]:
    row_vals = sorted({_center(f)[0] for f in frames})
    col_vals = sorted({_center(f)[1] for f in frames})
    row_index = {int(v): int(i) for i, v in enumerate(row_vals)}
    col_index = {int(v): int(i) for i, v in enumerate(col_vals)}
    R = len(row_vals)
    C = len(col_vals)
    order = _perimeter_cycle(int(R), int(C))
    assert len(order) == len(frames)

    slot_to_idx: dict[tuple[int, int], int] = {}
    for idx, f in enumerate(frames):
        rr, cc = _center(f)
        slot = (int(row_index[int(rr)]), int(col_index[int(cc)]))
        slot_to_idx[slot] = int(idx)

    assert set(slot_to_idx) == set(order)
    neighbors: dict[int, list[int]] = {int(i): [] for i in range(len(frames))}
    for sa, sb in zip(order, order[1:] + order[:1]):
        a = int(slot_to_idx[tuple(sa)])
        b = int(slot_to_idx[tuple(sb)])
        cells = _bridge_cells(frames[int(a)], frames[int(b)])
        assert cells and all(int(g[int(r), int(c)]) == int(bg) for r, c in cells)
        neighbors[int(a)].append(int(b))
        neighbors[int(b)].append(int(a))
    assert all(len(v) == 2 for v in neighbors.values())
    return neighbors


def _choose_start(frames: list[dict], neighbors: dict[int, list[int]], ctrl: list[int]) -> int:
    c0, c1, clast = int(ctrl[0]), int(ctrl[1]), int(ctrl[-1])
    candidates = [int(i) for i, f in enumerate(frames) if int(f["inner"]) == int(c0)]
    assert candidates
    if len(candidates) == 1:
        return int(candidates[0])

    good: list[int] = []
    for idx in candidates:
        nb_cols = sorted(int(frames[int(n)]["inner"]) for n in neighbors[int(idx)])
        if nb_cols == sorted([int(c1), int(clast)]):
            good.append(int(idx))
    assert len(good) == 1
    return int(good[0])


def _choose_next(frames: list[dict], neighbors: dict[int, list[int]], cur: int, prev: int | None, next_color: int) -> int:
    options = []
    for nb in neighbors[int(cur)]:
        if prev is not None and int(nb) == int(prev):
            continue
        if int(frames[int(nb)]["inner"]) == int(next_color):
            options.append(int(nb))

    if prev is None and not options:
        for nb in neighbors[int(cur)]:
            if int(frames[int(nb)]["inner"]) == int(next_color):
                options.append(int(nb))

    assert len(options) == 1
    return int(options[0])


def _draw_bridge(out: np.ndarray, fa: dict, fb: dict, color: int) -> None:
    H, W = map(int, out.shape)
    cells = _bridge_cells(fa, fb)
    assert cells
    for r, c in cells:
        if 0 <= int(r) < H and 0 <= int(c) < W:
            out[int(r), int(c)] = int(color)


def _draw_frame(g: np.ndarray, *, r0: int, c0: int, outer: int, t: int, border: int, inner_col: int) -> None:
    g[int(r0) : int(r0 + outer), int(c0) : int(c0 + outer)] = int(border)
    g[int(r0 + t) : int(r0 + outer - t), int(c0 + t) : int(c0 + outer - t)] = int(inner_col)


def _render_ring_output(
    *,
    shape: tuple[int, int],
    bg: int,
    border: int,
    frames: list[dict],
    traversal_slots: list[tuple[int, int]],
    ctrl_row_colors: list[int],
    ctrl_r: int,
    ctrl_cols: list[int],
) -> tuple[np.ndarray, set[tuple[int, int]]]:
    out = np.full(shape, int(bg), dtype=int)
    slot_to_frame = {tuple(f["slot"]): f for f in frames}
    for f in frames:
        r0, c0, r1, _c1 = map(int, f["bbox"])
        _draw_frame(
            out,
            r0=int(r0),
            c0=int(c0),
            outer=int(r1 - r0 + 1),
            t=int(f["t"]),
            border=int(border),
            inner_col=int(f["inner"]),
        )

    bridge_cells_all: set[tuple[int, int]] = set()
    ordered_frames = [slot_to_frame[tuple(slot)] for slot in traversal_slots]
    for fa, fb in zip(ordered_frames, ordered_frames[1:] + ordered_frames[:1]):
        cells = _bridge_cells(fa, fb)
        assert cells
        for r, c in cells:
            out[int(r), int(c)] = int(fa["inner"])
        bridge_cells_all |= set(cells)

    for color, col in zip(ctrl_row_colors, ctrl_cols):
        out[int(ctrl_r), int(col)] = int(color)

    return out, bridge_cells_all


_DIGITS = [int(i) for i in range(10)]


def _gen_attempt() -> np.ndarray | None:
    bg = int(np.random.choice(np.array(_DIGITS, dtype=int)))
    border = int(np.random.choice(np.array([d for d in _DIGITS if int(d) != int(bg)], dtype=int)))
    inner_palette = [int(d) for d in _DIGITS if int(d) not in (int(bg), int(border))]

    dims = [(2, 3), (3, 2), (2, 4), (4, 2), (3, 3)]
    R, C = dims[int(np.random.randint(0, len(dims)))]
    R = int(R)
    C = int(C)
    n_frames = int(2 * (R + C) - 4)
    if len(inner_palette) < int(n_frames):
        return None

    t = int(np.random.choice(np.array([1, 1, 2], dtype=int)))
    inner_side = int(np.random.randint(1, 4))
    outer = int(inner_side + 2 * t)
    if not (3 <= int(outer) <= 7):
        return None

    cycle = _perimeter_cycle(int(R), int(C))
    if len(cycle) != int(n_frames):
        return None

    row_gap = [int(np.random.randint(1, 5)) for _ in range(int(R) - 1)]
    col_gap = [int(np.random.randint(1, 5)) for _ in range(int(C) - 1)]
    top_margin = int(np.random.randint(1, 4))
    left_margin = int(np.random.randint(1, 4))

    r0s = [int(top_margin)]
    for gap in row_gap:
        r0s.append(int(r0s[-1] + outer + int(gap)))
    c0s = [int(left_margin)]
    for gap in col_gap:
        c0s.append(int(c0s[-1] + outer + int(gap)))

    direction = int(np.random.choice(np.array([1, -1], dtype=int)))
    start_idx = int(np.random.randint(0, int(n_frames)))
    traversal_slots = [cycle[(int(start_idx) + direction * k) % int(n_frames)] for k in range(int(n_frames))]

    ctrl_colors = [int(x) for x in np.random.permutation(np.array(inner_palette, dtype=int))[: int(n_frames)]]
    colors_by_slot = {tuple(slot): int(col) for slot, col in zip(traversal_slots, ctrl_colors)}

    frames: list[dict] = []
    for rr, cc in cycle:
        r0 = int(r0s[int(rr)])
        c0 = int(c0s[int(cc)])
        r1 = int(r0 + outer - 1)
        c1 = int(c0 + outer - 1)
        frames.append(
            {
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
                "inner_bbox": (int(r0 + t), int(c0 + t), int(r1 - t), int(c1 - t)),
                "t": int(t),
                "slot": (int(rr), int(cc)),
                "inner": int(colors_by_slot[(int(rr), int(cc))]),
            }
        )

    max_r1 = max(int(f["bbox"][2]) for f in frames)
    max_c1 = max(int(f["bbox"][3]) for f in frames)
    ctrl_gap = int(np.random.randint(2, 6))
    ctrl_r = int(max_r1 + ctrl_gap)
    bottom_margin = 1
    right_margin = int(np.random.randint(1, 5))
    ctrl_cols = [int(1 + 2 * i) for i in range(int(n_frames))]
    H = int(ctrl_r + 1 + bottom_margin)
    W = int(max(int(max_c1 + 1 + right_margin), int(ctrl_cols[-1] + 2)))
    if not (7 <= int(H) <= 30 and 7 <= int(W) <= 30):
        return None

    ctrl_row_colors = list(reversed(ctrl_colors))
    out, bridge_union = _render_ring_output(
        shape=(int(H), int(W)),
        bg=int(bg),
        border=int(border),
        frames=frames,
        traversal_slots=traversal_slots,
        ctrl_row_colors=ctrl_row_colors,
        ctrl_r=int(ctrl_r),
        ctrl_cols=ctrl_cols,
    )

    forward_slots = list(reversed(traversal_slots))
    forward_out, _ = _render_ring_output(
        shape=(int(H), int(W)),
        bg=int(bg),
        border=int(border),
        frames=frames,
        traversal_slots=forward_slots,
        ctrl_row_colors=ctrl_row_colors,
        ctrl_r=int(ctrl_r),
        ctrl_cols=ctrl_cols,
    )
    if bool(np.array_equal(out, forward_out)):
        return None

    inp = out.copy()
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
    if pred.shape != out.shape:
        return None
    if not bool(np.array_equal(pred, out)):
        return None

    try:
        ring_neighbors = _ring_neighbors(inp, _find_frames(inp, int(border)), int(bg))
    except Exception:
        return None
    if sorted(len(v) for v in ring_neighbors.values()) != [2] * int(n_frames):
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(600):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("Failed to generate valid input after 600 attempts")
