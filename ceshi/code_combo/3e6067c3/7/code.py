# Combo variant of 3e6067c3 (index 7)
# combo_spec: D2/2+D8/1
# Name: 行列交替对齐+反向处理

from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np

from arc_utils import get_bg, label_mask

"""
Design analysis for the combo variant
=====================================

This combo's prose spec is the authoritative rule:
- read the control sequence from right to left;
- traverse it step by step;
- odd-numbered steps must connect to a same-row target;
- even-numbered steps must connect to a same-column target.

The provided single-dimension notes for `D2/2` and `D8/1` do not match that prose.
For this file, the fusion description is treated as the actual target behavior.

a) Concrete behavior and trace
- Input still contains equal-sized framed squares plus one control row below them.
- Let the visible control row left-to-right be `[E, D, C, B, A]`. The solver must read
  it as `[A, B, C, D, E]`.
- Suppose `A->B` has both a same-row `B` and a same-column decoy `B'`.
  Step 1 is odd, so the solver must choose the row-aligned `B`.
- From that `B`, suppose `C` appears twice: one same-column real target and one same-row
  decoy. Step 2 is even, so the solver must choose the column-aligned `C`.
- The path therefore alternates row, column, row, column. Reading left-to-right or
  ignoring parity lands on decoys and produces visibly different bridges.

b) Complexity calibration
- Reverse reading alone can degenerate into "same bridges, different colors".
  Alternating direction alone can degenerate if every target color is unique.
- To keep both changes visible, the generator uses five path colors and duplicates each
  non-start target color once as an opposite-direction decoy.
- This keeps the palette small, gives repeated evidence for the row/column alternation,
  and makes wrong reading fall onto a different geometry instead of only recoloring.

c) Hidden assumptions from the original solver that no longer hold
- Original `ctrl` was read left-to-right. That must be reversed here.
- Original state update only needed "some aligned target"; now the allowed alignment
  depends on the step index parity, so the candidate filter changes every step.
- Original generator could rely on a single unambiguous path with mostly unique colors.
  Here repeated target colors are required, otherwise the alternating alignment rule is
  invisible.
- Original start disambiguation assumed the next step used the same generic alignment
  rule as later steps. Here start selection must already respect "step 1 = row".

d) Placement legality
- All frames stay on a sparse 6x6 slot lattice with at least one background cell gap.
- The generator first places all frames, then validates every intended bridge against
  frame occupancy, bounds, and previous bridges.
- Decoy frames are rejected if they sit on the intended bridge corridors.

e) Visual observability
- Each non-start path color appears exactly twice: one real continuation and one decoy.
- In the training pairs, the real bridge alternates horizontal/vertical while the decoy
  with the same color is available in the opposite orientation. This makes the parity
  rule directly visible.
- The control row is visibly reversed relative to the painted path order, so the
  right-to-left reading rule is also inferable.

f) Visual cleanliness
- Only nine framed squares are used: five path squares plus four decoys.
- Intended bridges never overlap each other and never cross frames or the control row.
- Decoys are spatially separated from the true corridors, so the scene remains readable.

g) Color-rule inferability
- Bridge colors are unchanged from the source task: each bridge uses the current control
  color at that step.
- There is no extra color remapping; the only challenge is choosing the right repeated
  target square under the fused constraints.

h) Generator-side anti-degeneration rule
- Every sample is rejected unless the combo output differs from:
  1. a left-to-right reader with the same odd/even direction schedule, and
  2. a right-to-left reader with the opposite odd/even schedule.
- This quantifies that both "reverse reading" and "alternating direction" are active.

i) New solver assumptions and how the generator guarantees them
- Assumption: the start color is unique. Generator makes only the later target colors
  repeat.
- Assumption: for each step, exactly one candidate of the target color matches the
  required row/column orientation. Generator assigns one real target and one opposite-
  orientation decoy for each step.
- Assumption: intended bridges are straight and non-empty. Generator uses a fixed sparse
  slot skeleton that preserves those corridors under axis-preserving reflections.
"""


_DIGITS = [int(i) for i in range(10)]
_PATH_SLOTS = [(0, 0), (0, 3), (3, 3), (3, 1), (5, 1)]
_DECOY_SLOTS = [(3, 0), (0, 5), (5, 3), (3, 5)]
_LATTICE_SIZE = 6


def _bbox_key(frame: dict) -> tuple[int, int, int, int]:
    return tuple(map(int, frame["bbox"]))


def _center(frame: dict) -> tuple[int, int]:
    r0, c0, r1, c1 = map(int, frame["bbox"])
    return int((r0 + r1) // 2), int((c0 + c1) // 2)


def _manhattan(frame_a: dict, frame_b: dict) -> int:
    ra, ca = _center(frame_a)
    rb, cb = _center(frame_b)
    return abs(int(ra) - int(rb)) + abs(int(ca) - int(cb))


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


def _aligned(frame_a: dict, frame_b: dict, mode: str) -> bool:
    a = frame_a["inner_bbox"]
    b = frame_b["inner_bbox"]
    if mode == "row":
        return bool(int(a[0]) == int(b[0]) and int(a[2]) == int(b[2]))
    return bool(int(a[1]) == int(b[1]) and int(a[3]) == int(b[3]))


def _bridge_cells(frame_a: dict, frame_b: dict, mode: str) -> set[tuple[int, int]]:
    r0a, c0a, r1a, c1a = map(int, frame_a["bbox"])
    r0b, c0b, r1b, c1b = map(int, frame_b["bbox"])
    ia = frame_a["inner_bbox"]
    ib = frame_b["inner_bbox"]

    if mode == "row":
        if not _aligned(frame_a, frame_b, "row"):
            return set()
        rr0, rr1 = int(ia[0]), int(ia[2])
        if int(c1a) < int(c0b):
            cc0, cc1 = int(c1a + 1), int(c0b - 1)
        else:
            cc0, cc1 = int(c1b + 1), int(c0a - 1)
        if int(cc0) > int(cc1):
            return set()
        return {(int(r), int(c)) for r in range(int(rr0), int(rr1) + 1) for c in range(int(cc0), int(cc1) + 1)}

    if not _aligned(frame_a, frame_b, "col"):
        return set()
    cc0, cc1 = int(ia[1]), int(ia[3])
    if int(r1a) < int(r0b):
        rr0, rr1 = int(r1a + 1), int(r0b - 1)
    else:
        rr0, rr1 = int(r1b + 1), int(r0a - 1)
    if int(rr0) > int(rr1):
        return set()
    return {(int(r), int(c)) for r in range(int(rr0), int(rr1) + 1) for c in range(int(cc0), int(cc1) + 1)}


def _draw_cells(grid: np.ndarray, cells: set[tuple[int, int]], color: int) -> None:
    H, W = grid.shape
    for r, c in cells:
        if 0 <= int(r) < int(H) and 0 <= int(c) < int(W):
            grid[int(r), int(c)] = int(color)


def _pick_target(cur: dict, cands: list[dict], mode: str) -> dict | None:
    best = None
    bestf = None
    for frame in cands:
        if frame is cur:
            continue
        if not _aligned(cur, frame, mode):
            continue
        cells = _bridge_cells(cur, frame, mode)
        if not cells:
            continue
        cand = (int(_manhattan(cur, frame)),) + _bbox_key(frame)
        if best is None or cand < best:
            best = cand
            bestf = frame
    return bestf


def _pick_start(start_cands: list[dict], next_cands: list[dict], first_mode: str) -> dict:
    if len(start_cands) == 1:
        return start_cands[0]

    best = None
    bestf = None
    for frame in start_cands:
        nxt = _pick_target(frame, next_cands, first_mode)
        usable = 1 if nxt is not None else 0
        nxt_key = (10**9, 10**9, 10**9, 10**9) if nxt is None else _bbox_key(nxt)
        nxt_dist = 10**9 if nxt is None else int(_manhattan(frame, nxt))
        cand = (-int(usable), int(nxt_dist), *nxt_key, *_bbox_key(frame))
        if best is None or cand < best:
            best = cand
            bestf = frame
    assert bestf is not None
    return bestf


def _solve(grid: np.ndarray, *, reverse: bool, odd_mode: str) -> np.ndarray:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    cnt = Counter(map(int, g.flatten()))
    border = None
    for col, _count in cnt.most_common():
        if int(col) != int(bg):
            border = int(col)
            break
    assert border is not None

    frames = _extract_frames(g, int(border))
    assert frames

    by_color: dict[int, list[dict]] = defaultdict(list)
    for frame in frames:
        by_color[int(frame["inner"])].append(frame)

    ctrl = _control_colors(g, int(bg), frames)
    seq = list(reversed(ctrl)) if reverse else list(ctrl)
    assert len(seq) >= 2

    first_mode = odd_mode
    other_mode = "col" if first_mode == "row" else "row"
    start_color = int(seq[0])
    next_color = int(seq[1])
    assert start_color in by_color
    cur = _pick_start(by_color[start_color], by_color.get(next_color, []), first_mode)

    out = g.copy()
    for idx in range(len(seq) - 1):
        color_a = int(seq[int(idx)])
        color_b = int(seq[int(idx) + 1])
        mode = first_mode if int(idx) % 2 == 0 else other_mode
        assert int(cur["inner"]) == int(color_a)
        nxt = _pick_target(cur, by_color.get(color_b, []), mode)
        assert nxt is not None
        _draw_cells(out, _bridge_cells(cur, nxt, mode), int(color_a))
        cur = nxt
    return out


def main(grid: np.ndarray) -> np.ndarray:
    return _solve(grid, reverse=True, odd_mode="row")


def _most_common_nonbg(grid: np.ndarray, bg: int) -> int | None:
    cnt = Counter(map(int, grid.flatten()))
    for col, _count in cnt.most_common():
        if int(col) != int(bg):
            return int(col)
    return None


def _draw_frame(grid: np.ndarray, *, r0: int, c0: int, outer: int, t: int, border: int, inner_col: int) -> None:
    grid[int(r0) : int(r0 + outer), int(c0) : int(c0 + outer)] = int(border)
    grid[int(r0 + t) : int(r0 + outer - t), int(c0 + t) : int(c0 + outer - t)] = int(inner_col)


def _frame_cells(frame: dict) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = map(int, frame["bbox"])
    return {(int(r), int(c)) for r in range(int(r0), int(r1) + 1) for c in range(int(c0), int(c1) + 1)}


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


def _transform_slot(slot: tuple[int, int], mode: int) -> tuple[int, int]:
    r, c = map(int, slot)
    if int(mode) == 0:
        return int(r), int(c)
    if int(mode) == 1:
        return int(r), int(_LATTICE_SIZE - 1 - c)
    if int(mode) == 2:
        return int(_LATTICE_SIZE - 1 - r), int(c)
    return int(_LATTICE_SIZE - 1 - r), int(_LATTICE_SIZE - 1 - c)


def _gen_attempt() -> np.ndarray | None:
    bg = int(np.random.choice(np.array(_DIGITS, dtype=int)))
    border = int(np.random.choice(np.array([d for d in _DIGITS if int(d) != int(bg)], dtype=int)))
    inner_palette = [int(d) for d in _DIGITS if int(d) not in (int(bg), int(border))]
    if len(inner_palette) < 5:
        return None

    t = 1
    outer = 3
    top_margin = int(np.random.randint(1, 3))
    left_margin = int(np.random.randint(1, 3))
    row_gap = [int(np.random.choice(np.array([1, 1, 2], dtype=int))) for _ in range(_LATTICE_SIZE - 1)]
    col_gap = [int(np.random.choice(np.array([1, 1, 2], dtype=int))) for _ in range(_LATTICE_SIZE - 1)]

    r0s = [int(top_margin)]
    for gap in row_gap:
        r0s.append(int(r0s[-1] + outer + int(gap)))
    c0s = [int(left_margin)]
    for gap in col_gap:
        c0s.append(int(c0s[-1] + outer + int(gap)))

    transform = int(np.random.randint(0, 4))
    path_slots = [_transform_slot(slot, int(transform)) for slot in _PATH_SLOTS]
    decoy_slots = [_transform_slot(slot, int(transform)) for slot in _DECOY_SLOTS]
    used_slots = path_slots + decoy_slots
    if len(set(used_slots)) != len(used_slots):
        return None

    frame_by_slot = {slot: _slot_frame(slot, r0s, c0s, int(outer), int(t)) for slot in used_slots}
    path_frames = [frame_by_slot[slot] for slot in path_slots]
    decoy_frames = [frame_by_slot[slot] for slot in decoy_slots]

    max_r1 = max(int(frame["bbox"][2]) for frame in frame_by_slot.values())
    max_c1 = max(int(frame["bbox"][3]) for frame in frame_by_slot.values())
    ctrl_gap = int(np.random.randint(2, 5))
    ctrl_r = int(max_r1 + ctrl_gap)
    bottom_margin = 1
    right_margin = int(np.random.randint(1, 4))
    H = int(ctrl_r + 1 + bottom_margin)
    W = int(max_c1 + 1 + right_margin)
    if not (12 <= int(H) <= 30 and 12 <= int(W) <= 30):
        return None

    path_colors = [int(x) for x in np.random.choice(np.array(inner_palette, dtype=int), size=len(path_frames), replace=False)]
    color_by_slot: dict[tuple[int, int], int] = {}
    for slot, color in zip(path_slots, path_colors):
        color_by_slot[slot] = int(color)
    for idx, slot in enumerate(decoy_slots, start=1):
        color_by_slot[slot] = int(path_colors[int(idx)])

    ctrl_cols = [int(1 + 2 * i) for i in range(len(path_colors))]
    if int(ctrl_cols[-1]) >= int(W):
        return None

    out = np.full((int(H), int(W)), int(bg), dtype=int)
    for slot in used_slots:
        frame = frame_by_slot[slot]
        r0, c0, r1, _c1 = map(int, frame["bbox"])
        _draw_frame(
            out,
            r0=int(r0),
            c0=int(c0),
            outer=int(r1 - r0 + 1),
            t=int(frame["t"]),
            border=int(border),
            inner_col=int(color_by_slot[slot]),
        )

    bridge_union: set[tuple[int, int]] = set()
    for idx in range(len(path_frames) - 1):
        mode = "row" if int(idx) % 2 == 0 else "col"
        cells = _bridge_cells(path_frames[int(idx)], path_frames[int(idx) + 1], mode)
        if not cells:
            return None
        if cells & bridge_union:
            return None
        for r, c in cells:
            if not (0 <= int(r) < int(H) and 0 <= int(c) < int(W)):
                return None
            if int(out[int(r), int(c)]) != int(bg):
                return None
        _draw_cells(out, cells, int(path_colors[int(idx)]))
        bridge_union |= set(cells)

    for frame in decoy_frames:
        if _frame_cells(frame) & bridge_union:
            return None

    ctrl_visible = list(reversed(path_colors))
    for col, color in zip(ctrl_cols, ctrl_visible):
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
        wrong_read = _solve(inp, reverse=False, odd_mode="row")
        wrong_parity = _solve(inp, reverse=True, odd_mode="col")
    except Exception:
        return None

    if pred.shape != out.shape:
        return None
    if not bool(np.array_equal(pred, out)):
        return None
    if bool(np.array_equal(pred, wrong_read)):
        return None
    if bool(np.array_equal(pred, wrong_parity)):
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(500):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("Failed to generate valid input after 500 attempts")
