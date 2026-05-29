# Variant of 3e6067c3
# Dimension: D11 — 空间关系：方块排列拓扑 (meta)
# Change: 方块排列在不规则间距的网格格点上，桥梁仅在行/列对齐的方块之间绘制（正交方向）... → 环形排列

from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np

from arc_utils import get_bg, label_mask

"""
Design analysis for the D11 variant

a) Concrete behavior and trace
- Original: framed squares occupy an irregular lattice path. The control row lists colors in path order.
  `main()` looks for the next square with the requested color that is aligned in the same row/column band,
  then draws a straight bridge. The topology is an open self-avoiding path.
- Variant: the framed squares occupy every boundary slot of a coarse rectangle, leaving the center empty.
  That boundary forms a visible ring/cycle. The control row lists one full traversal of that ring, starting
  at an arbitrary square and going clockwise or counterclockwise. After drawing bridges for consecutive
  control colors, the solver also connects the last visited square back to the first, closing the loop.
- Small trace: if the ring order is A,B,C,D,E,F and control row is C,D,E,F,A,B, the original solver would
  only draw C->D->E->F->A->B and stop. The variant must additionally draw B->C, because the topology is a
  cycle rather than an open path. This is the core observable change.

b) Complexity calibration
- The squares are still equal-sized framed boxes and bridges are still axis-aligned solid rectangles, so the
  task remains readable.
- The change is not just a plain open-chain lookup: the control row gives a cyclic order, and the solver must
  use the second control color to choose which of the two ring directions to follow before closing the loop.
- The ring uses 6 or 8 squares so the hole in the middle and the closing edge are visually obvious while the
  control row still fits comfortably inside ARC-size grids.

c) Hidden assumptions from the original `main()`
- `aligned(f1, f2)` originally encoded the whole topology assumption: legal transitions only existed between
  same-row or same-column squares. That remains true locally for adjacent ring slots, but not globally for
  arbitrary equal colors elsewhere on the ring. Therefore the variant cannot choose the next frame by
  “aligned + nearest among all same-colored candidates” alone.
- Start selection in the original solver relied on proximity to the next same-colored candidate. That is no
  longer sufficient when the structure is cyclic. The variant instead uses ring adjacency: the start frame
  must have neighbors colored like the second and last control entries.
- The original loop stopped after `len(ctrl)-1` bridges. That silently assumes an open path. The variant must
  add one more bridge from the final frame back to the start frame.

d) Placement legality
- Generator places frames only on the perimeter slots of a small coarse lattice chosen from 2x3, 3x2, 2x4,
  4x2, or 3x3, so the perimeter length is always 6 or 8.
- Row and column gaps are still irregular, but every neighboring perimeter slot remains aligned in one axis.
- The bridge corridor between two adjacent slots is required to be non-empty, in bounds, and background-only
  in the input. Output-first validation checks all of this before accepting a sample.

e) Visual observability
- The center of the rectangle is empty, so the difference from the original lattice-path layout is visible
  from a few examples.
- The output usually forms a closed colored ring, including a bridge across the final gap back to the first
  control square. That closure is directly visible.

f) Visual cleanliness
- Only one object type occupies the scene above the control row: framed squares plus bridges of matching
  thickness. There are no extra symbols inside the ring hole.
- Adjacent bridges never overlap because each bridge occupies only the gap between two neighboring frames, not
  the frame interiors.

g) Color inferability
- No invisible numeric ordering is used. Colors are only matched by visible equality between control pixels
  and square interiors.
- The generator uses distinct inner colors so the intended cyclic visit order is easy to read from few-shot
  examples; the topology change is carried by the visible ring arrangement and the extra closing bridge.

9) New assumptions introduced by the variant and how generation guarantees them
- The inferred adjacency graph between frames must be a single degree-2 cycle. Generator enforces a full
  perimeter ring and validation rejects anything else.
- The first control color must identify a start frame. Generator makes the first color unique globally.
- The second control color must pick one of the two directions from the start frame. Generator uses distinct
  inner colors, so the two neighbors of the start are visibly different.
- Every consecutive control pair must correspond to adjacent ring frames. Generator builds control directly
  from a cyclic traversal of the perimeter slots.
"""


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect framed squares along a cyclic ring traversal.

    Input structure:
    - Equal-sized framed squares sit on the perimeter of a coarse rectangle, leaving a hollow center.
    - A control row below the figure lists one full traversal of the ring.

    Rule:
    - Find the start square matching the first control color.
    - Follow the ring to the neighbor whose inner color matches the next control pixel; once direction is set,
      keep moving to the matching neighbor that is not the immediately previous square.
    - Draw each bridge using the current square's color, and finally connect the last square back to the first.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    border = _most_common_nonbg(g, bg)
    assert border is not None

    frames = _find_frames(g, border)
    assert frames

    by: dict[int, list[int]] = defaultdict(list)
    for idx, f in enumerate(frames):
        by[int(f["inner"])].append(int(idx))

    max_frame_r = max(int(f["bbox"][2]) for f in frames)
    ctrl_r = None
    for r in range(int(g.shape[0] - 1), int(max_frame_r), -1):
        if int((g[int(r)] != int(bg)).sum()) >= 2:
            ctrl_r = int(r)
            break
    assert ctrl_r is not None

    ctrl = [int(g[int(ctrl_r), int(c)]) for c in range(int(g.shape[1])) if int(g[int(ctrl_r), int(c)]) != int(bg)]
    assert len(ctrl) >= 4

    neighbors = _ring_neighbors(g, frames, bg)
    start = _choose_start(frames, neighbors, ctrl)

    out = g.copy()
    prev: int | None = None
    cur = int(start)
    for next_color in ctrl[1:]:
        nxt = _choose_next(frames, neighbors, cur, prev, int(next_color))
        _draw_bridge(out, frames[cur], frames[nxt], int(frames[cur]["inner"]))
        prev, cur = int(cur), int(nxt)

    _draw_bridge(out, frames[cur], frames[start], int(frames[cur]["inner"]))
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
        cells = _bridge_cells(frames[a], frames[b])
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
        nb_cols = sorted(int(frames[n]["inner"]) for n in neighbors[int(idx)])
        if nb_cols == sorted([int(c1), int(clast)]):
            good.append(int(idx))
    assert len(good) == 1
    return int(good[0])


def _choose_next(
    frames: list[dict], neighbors: dict[int, list[int]], cur: int, prev: int | None, next_color: int
) -> int:
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


_DIGITS = [int(i) for i in range(10)]


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


def _build_ctrl_colors(inner_palette: list[int], n: int) -> list[int] | None:
    if len(inner_palette) < int(n) or n < 4:
        return None
    perm = np.random.permutation(np.array(inner_palette, dtype=int))
    return [int(x) for x in perm[: int(n)]]


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

    dims = [(2, 3), (3, 2), (2, 4), (4, 2), (3, 3)]
    R, C = dims[int(np.random.randint(0, len(dims)))]
    R = int(R)
    C = int(C)
    n_frames = int(2 * (R + C) - 4)
    if not (6 <= int(n_frames) <= 12):
        return None

    cycle = _perimeter_cycle(R, C)
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

    ctrl_colors = _build_ctrl_colors(inner_palette, int(n_frames))
    if ctrl_colors is None:
        return None

    colors_by_slot = {tuple(slot): int(col) for slot, col in zip(traversal_slots, ctrl_colors)}
    frames: list[dict] = []
    slot_to_frame: dict[tuple[int, int], dict] = {}
    for rr, cc in cycle:
        r0 = int(r0s[int(rr)])
        c0 = int(c0s[int(cc)])
        r1 = int(r0 + outer - 1)
        c1 = int(c0 + outer - 1)
        f = {
            "bbox": (int(r0), int(c0), int(r1), int(c1)),
            "inner_bbox": (int(r0 + t), int(c0 + t), int(r1 - t), int(c1 - t)),
            "slot": (int(rr), int(cc)),
            "inner": int(colors_by_slot[(int(rr), int(cc))]),
        }
        frames.append(f)
        slot_to_frame[(int(rr), int(cc))] = f

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
    if int(ctrl_cols[-1]) >= int(W):
        return None

    out = np.full((int(H), int(W)), int(bg), dtype=int)
    for f in frames:
        r0, c0, _r1, _c1 = map(int, f["bbox"])
        _draw_frame(
            out,
            r0=int(r0),
            c0=int(c0),
            outer=int(outer),
            t=int(t),
            border=int(border),
            inner_col=int(f["inner"]),
        )

    bridge_cells_all: set[tuple[int, int]] = set()
    ordered_frames = [slot_to_frame[tuple(slot)] for slot in traversal_slots]
    ok = True
    for fa, fb in zip(ordered_frames, ordered_frames[1:] + ordered_frames[:1]):
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
        paint = int(fa["inner"])
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

    ring_neighbors = _ring_neighbors(inp, _find_frames(inp, border), bg)
    if sorted(len(v) for v in ring_neighbors.values()) != [2] * int(n_frames):
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(600):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("Failed to generate valid input after 600 attempts")
