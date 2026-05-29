# Variant of 3e6067c3
# Dimension: D2 — 匹配条件：对齐与消歧规则 (match_cond)
# Change: 当多个方块具有相同内部颜色时，优先选择行/列对齐且距离最近的方块... → 无对齐约束，纯最近邻

from __future__ import annotations

from collections import Counter, defaultdict
import itertools

import numpy as np

from arc_utils import get_bg, label_mask

"""
Design analysis for this D2 variant
==================================

Source-task note
- No `source_task` was provided, so there is no extra reference solver to inspect.

a) Concrete behavior and trace
- The original task contains equal-sized framed squares plus a control row below them.
- The output paints bridges whose color is the current control color, connecting the
  current square to the next square named by the next control color.
- Original disambiguation: if color `b` appears in multiple squares, prefer a `b`
  square aligned to the current square's inner row-band or inner col-band, then use
  nearest distance.
- Variant disambiguation: drop the alignment preference entirely. Among all `b`
  squares, pick the one with minimum Manhattan distance between square centers.
- Because the chosen target can now be diagonal relative to the current square, the
  bridge can no longer be only straight. This variant uses a deterministic L route:
  keep the current square's inner row-band as the horizontal leg, then use the target
  square's inner col-band as the vertical leg.
- Small trace:
  current=A at upper-left, target B-nearest at lower-right, decoy B' aligned on A's
  row but farther away. Original logic picks B' because alignment wins; this variant
  picks B because distance wins, so the output contains an L-shaped bridge instead of
  a straight bridge.

b) Complexity calibration
- If every control color were unique, D2 would be invisible. Therefore the generator
- duplicates each non-start path color exactly once: one intended path square and one
  decoy square.
- If all intended moves were still same-row or same-col, nearest-neighbor would look
  the same as the original. Therefore every path transition is diagonal in lattice
  coordinates, so L bridges are common and visually obvious.
- The number of colors stays small: four path colors total, with three duplicated.

c) Hidden assumptions in the original solver that break here
- `aligned()` used to be part of candidate scoring. That assumption is removed.
- `draw_bridge()` only handled pure horizontal or pure vertical bridges. That is no
  longer sufficient once the chosen target can be diagonal.
- The original generator relied on a 4-neighbor lattice path, which guaranteed that
  `_bridge_cells()` always had a straight same-row or same-col answer. That no longer
  holds, so generation must explicitly validate L bridges.

d) Placement legality
- Frames are still equal-sized and non-overlapping.
- The generator constructs the output first, paints all frames, then paints bridges.
- Any candidate sample is rejected if a bridge would leave the grid, overlap any
  frame, or overlap a previous bridge.
- Decoy frames are filtered so they do not sit on top of the intended bridge cells.

e) Visual observability
- Each sample contains repeated colors, but the intended next square is often diagonal
  while a farther decoy of the same color is aligned. This makes the new rule visible:
  the output bridge bends toward the nearest square rather than continuing straight.

f) Visual cleanliness
- The generator uses only four visited squares and three decoys on a sparse 5x5 slot
  lattice, which keeps the scene readable.
- Bridge overlap is disallowed, so the final image stays legible instead of becoming a
  pile of intersecting corridors.

g) Color-rule inferability
- Colors are not remapped. A bridge always takes the current control color, exactly as
  in the original task. The only changed clue is which repeated-color square is chosen.

h) Generator-side anti-degeneration rule
- For each transition `Pi -> P(i+1)`, the color of `P(i+1)` also appears on one decoy
  square aligned with `Pi` but farther away than `P(i+1)`.
- This guarantees that the new nearest-neighbor rule is active on the same input where
  the old aligned-first rule would prefer a different target.

i) New solver assumptions and generator guarantees
- Solver assumption: consecutive control colors are different, otherwise the nearest
  target of the same color could be the current square itself. The generator enforces
  unique path colors.
- Solver assumption: non-aligned frame pairs are fully separated in both row and
  column, so the fixed L route is well-defined. The generator enforces placement on a
  sparse lattice with positive gaps.
- Solver assumption: ties are rare. The solver breaks ties by bbox order, and the
  generator searches until the intended nearest choice is unique.
"""


def _bbox_key(f: dict) -> tuple[int, int, int, int]:
    return tuple(map(int, f["bbox"]))


def _center(f: dict) -> tuple[int, int]:
    r0, c0, r1, c1 = map(int, f["bbox"])
    return int((r0 + r1) // 2), int((c0 + c1) // 2)


def _manhattan(f1: dict, f2: dict) -> int:
    r1, c1 = _center(f1)
    r2, c2 = _center(f2)
    return abs(int(r1) - int(r2)) + abs(int(c1) - int(c2))


def _aligned(f1: dict, f2: dict) -> tuple[bool, bool]:
    a = f1["inner_bbox"]
    b = f2["inner_bbox"]
    same_row = bool(int(a[0]) == int(b[0]) and int(a[2]) == int(b[2]))
    same_col = bool(int(a[1]) == int(b[1]) and int(a[3]) == int(b[3]))
    return same_row, same_col


def _extract_frames(g: np.ndarray, border: int) -> list[dict]:
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


def _bridge_cells(fa: dict, fb: dict) -> set[tuple[int, int]]:
    (r0a, c0a, r1a, c1a) = map(int, fa["bbox"])
    (r0b, c0b, r1b, c1b) = map(int, fb["bbox"])
    ia = fa["inner_bbox"]
    ib = fb["inner_bbox"]
    sr, sc = _aligned(fa, fb)

    if sr:
        rr0, rr1 = int(ia[0]), int(ia[2])
        if int(c1a) < int(c0b):
            cc0, cc1 = int(c1a + 1), int(c0b - 1)
        else:
            cc0, cc1 = int(c1b + 1), int(c0a - 1)
        if int(cc0) > int(cc1):
            return set()
        return {(int(r), int(c)) for r in range(int(rr0), int(rr1) + 1) for c in range(int(cc0), int(cc1) + 1)}

    if sc:
        cc0, cc1 = int(ia[1]), int(ia[3])
        if int(r1a) < int(r0b):
            rr0, rr1 = int(r1a + 1), int(r0b - 1)
        else:
            rr0, rr1 = int(r1b + 1), int(r0a - 1)
        if int(rr0) > int(rr1):
            return set()
        return {(int(r), int(c)) for r in range(int(rr0), int(rr1) + 1) for c in range(int(cc0), int(cc1) + 1)}

    if int(c1a) < int(c0b):
        h_c0, h_c1 = int(c1a + 1), int(ib[3])
    elif int(c1b) < int(c0a):
        h_c0, h_c1 = int(ib[1]), int(c0a - 1)
    else:
        return set()

    if int(r1a) < int(r0b):
        v_r0, v_r1 = int(ia[0]), int(r0b - 1)
    elif int(r1b) < int(r0a):
        v_r0, v_r1 = int(r1b + 1), int(ia[2])
    else:
        return set()

    if int(h_c0) > int(h_c1) or int(v_r0) > int(v_r1):
        return set()

    cells = {
        (int(r), int(c))
        for r in range(int(ia[0]), int(ia[2]) + 1)
        for c in range(int(h_c0), int(h_c1) + 1)
    }
    cells |= {
        (int(r), int(c))
        for r in range(int(v_r0), int(v_r1) + 1)
        for c in range(int(ib[1]), int(ib[3]) + 1)
    }
    return cells


def _draw_cells(out: np.ndarray, cells: set[tuple[int, int]], color: int) -> None:
    H, W = out.shape
    for r, c in cells:
        if 0 <= int(r) < int(H) and 0 <= int(c) < int(W):
            out[int(r), int(c)] = int(color)


def _pick_nearest(src: dict, cands: list[dict]) -> dict:
    assert cands
    best = None
    bestf = None
    for f in cands:
        cand = (int(_manhattan(src, f)),) + _bbox_key(f)
        if best is None or cand < best:
            best = cand
            bestf = f
    assert bestf is not None
    return bestf


def _pick_start(start_cands: list[dict], next_cands: list[dict]) -> dict:
    if len(start_cands) == 1:
        return start_cands[0]
    if not next_cands:
        return min(start_cands, key=_bbox_key)

    best = None
    bestf = None
    for f in start_cands:
        nearest_next = min((int(_manhattan(f, g2)),) + _bbox_key(g2) for g2 in next_cands)
        cand = nearest_next + _bbox_key(f)
        if best is None or cand < best:
            best = cand
            bestf = f
    assert bestf is not None
    return bestf


def main(grid: np.ndarray) -> np.ndarray:
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

    frames = _extract_frames(g, int(border))
    assert frames

    by: dict[int, list[dict]] = defaultdict(list)
    for f in frames:
        by[int(f["inner"])].append(f)

    max_frame_r = max(int(f["bbox"][2]) for f in frames)
    ctrl_r = None
    for r in range(int(H - 1), int(max_frame_r), -1):
        if int((g[int(r)] != int(bg)).sum()) >= 2:
            ctrl_r = int(r)
            break
    assert ctrl_r is not None

    ctrl = [int(g[int(ctrl_r), int(c)]) for c in range(int(W)) if int(g[int(ctrl_r), int(c)]) != int(bg)]
    assert len(ctrl) >= 2

    start_color = int(ctrl[0])
    assert start_color in by
    cur = _pick_start(by[start_color], by.get(int(ctrl[1]), []))

    out = g.copy()
    for i in range(int(len(ctrl)) - 1):
        a = int(ctrl[i])
        b = int(ctrl[i + 1])
        assert int(cur["inner"]) == int(a)
        if b not in by:
            continue
        nxt = _pick_nearest(cur, by[b])
        cells = _bridge_cells(cur, nxt)
        _draw_cells(out, cells, int(a))
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


def _frame_cells(f: dict) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = map(int, f["bbox"])
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
        "slot": (int(rr), int(cc)),
    }


def _manhattan_slots(a: tuple[int, int], b: tuple[int, int], frame_by_slot: dict[tuple[int, int], dict]) -> int:
    return int(_manhattan(frame_by_slot[a], frame_by_slot[b]))


def _safe_decoy_side(cur: tuple[int, int], target: tuple[int, int], cand: tuple[int, int]) -> bool:
    cur_r, cur_c = map(int, cur)
    tar_r, tar_c = map(int, target)
    can_r, can_c = map(int, cand)
    if int(can_r) == int(cur_r):
        if int(tar_c) > int(cur_c):
            return int(can_c) < int(cur_c) or int(can_c) > int(tar_c)
        return int(can_c) > int(cur_c) or int(can_c) < int(tar_c)
    if int(can_c) == int(cur_c):
        if int(tar_r) > int(cur_r):
            return int(can_r) < int(cur_r) or int(can_r) > int(tar_r)
        return int(can_r) > int(cur_r) or int(can_r) < int(tar_r)
    return False


def _choose_path_and_bridges(
    H: int,
    W: int,
    bg: int,
    border: int,
    path_slots: list[tuple[int, int]],
    frame_by_slot: dict[tuple[int, int], dict],
    outer: int,
    t: int,
) -> set[tuple[int, int]] | None:
    temp = np.full((int(H), int(W)), int(bg), dtype=int)
    for slot in path_slots:
        f = frame_by_slot[slot]
        r0, c0, _r1, _c1 = map(int, f["bbox"])
        _draw_frame(temp, r0=int(r0), c0=int(c0), outer=int(outer), t=int(t), border=int(border), inner_col=1)

    bridge_cells_all: set[tuple[int, int]] = set()
    for a_slot, b_slot in zip(path_slots, path_slots[1:]):
        cells = _bridge_cells(frame_by_slot[a_slot], frame_by_slot[b_slot])
        if not cells:
            return None
        if cells & bridge_cells_all:
            return None
        for r, c in cells:
            if not (0 <= int(r) < int(H) and 0 <= int(c) < int(W)):
                return None
            if int(temp[int(r), int(c)]) != int(bg):
                return None
        for r, c in cells:
            temp[int(r), int(c)] = 2
        bridge_cells_all |= set(cells)
    return bridge_cells_all


def _choose_path_slots(
    H: int,
    W: int,
    bg: int,
    border: int,
    frame_by_slot: dict[tuple[int, int], dict],
    outer: int,
    t: int,
) -> tuple[list[tuple[int, int]], set[tuple[int, int]]] | None:
    rows_choices = list(itertools.combinations(range(5), 4))
    cols_choices = list(itertools.combinations(range(5), 4))
    np.random.shuffle(rows_choices)
    np.random.shuffle(cols_choices)

    attempts = 0
    for rows in rows_choices:
        row_list = [int(x) for x in rows]
        for cols in cols_choices:
            col_list = [int(x) for x in cols]
            perms = list(itertools.permutations(col_list))
            np.random.shuffle(perms)
            for perm in perms:
                attempts += 1
                if attempts > 240:
                    return None
                path_slots = [(int(row_list[i]), int(perm[i])) for i in range(4)]
                if any(int(path_slots[i][1]) == int(path_slots[i + 1][1]) for i in range(3)):
                    continue
                bridges = _choose_path_and_bridges(
                    int(H), int(W), int(bg), int(border), path_slots, frame_by_slot, int(outer), int(t)
                )
                if bridges is not None:
                    return path_slots, bridges
    return None


def _choose_decoy_slots(
    path_slots: list[tuple[int, int]],
    path_bridge_cells: set[tuple[int, int]],
    frame_by_slot: dict[tuple[int, int], dict],
) -> list[tuple[int, int]] | None:
    remaining = [slot for slot in frame_by_slot if slot not in set(path_slots)]
    blocked_slots = {
        slot
        for slot in remaining
        if _frame_cells(frame_by_slot[slot]) & path_bridge_cells
    }

    targets = [1, 2, 3]
    chosen: list[tuple[int, int] | None] = [None, None, None]

    def rec(idx: int, used: set[tuple[int, int]]) -> bool:
        if idx == len(targets):
            return True

        step = int(targets[idx])
        cur = path_slots[int(step - 1)]
        target = path_slots[int(step)]
        target_dist = _manhattan_slots(cur, target, frame_by_slot)
        cands = []
        for slot in remaining:
            if slot in used or slot in blocked_slots:
                continue
            same_row = bool(int(slot[0]) == int(cur[0]))
            same_col = bool(int(slot[1]) == int(cur[1]))
            if not (same_row or same_col):
                continue
            if not _safe_decoy_side(cur, target, slot):
                continue
            if _manhattan_slots(cur, slot, frame_by_slot) <= int(target_dist):
                continue
            cands.append(slot)

        cands.sort(
            key=lambda slot: (
                -int(_manhattan_slots(cur, slot, frame_by_slot)),
                int(frame_by_slot[slot]["bbox"][0]),
                int(frame_by_slot[slot]["bbox"][1]),
            )
        )
        for slot in cands:
            chosen[idx] = slot
            if rec(int(idx + 1), used | {slot}):
                return True
            chosen[idx] = None
        return False

    if not rec(0, set()):
        return None
    assert all(slot is not None for slot in chosen)
    return [slot for slot in chosen if slot is not None]


_DIGITS = [int(i) for i in range(10)]


def _gen_attempt() -> np.ndarray | None:
    bg = int(np.random.choice(np.array(_DIGITS, dtype=int)))
    border = int(np.random.choice(np.array([d for d in _DIGITS if int(d) != int(bg)], dtype=int)))
    inner_palette = [int(d) for d in _DIGITS if int(d) not in (int(bg), int(border))]
    if len(inner_palette) < 4:
        return None

    t = 1
    outer = 3
    path_len = 4
    R = 5
    C = 5

    row_gap = [int(np.random.randint(1, 3)) for _ in range(int(R) - 1)]
    col_gap = [int(np.random.randint(1, 3)) for _ in range(int(C) - 1)]
    top_margin = int(np.random.randint(1, 3))
    left_margin = int(np.random.randint(1, 3))

    r0s = [int(top_margin)]
    for gap in row_gap:
        r0s.append(int(r0s[-1] + outer + int(gap)))
    c0s = [int(left_margin)]
    for gap in col_gap:
        c0s.append(int(c0s[-1] + outer + int(gap)))

    max_r1 = int(r0s[-1] + outer - 1)
    max_c1 = int(c0s[-1] + outer - 1)
    ctrl_gap = int(np.random.randint(2, 4))
    ctrl_r = int(max_r1 + ctrl_gap)
    bottom_margin = 1
    right_margin = int(np.random.randint(1, 4))
    H = int(ctrl_r + 1 + bottom_margin)
    W = int(max_c1 + 1 + right_margin)
    if not (10 <= int(H) <= 30 and 10 <= int(W) <= 30):
        return None

    all_slots = [(int(r), int(c)) for r in range(int(R)) for c in range(int(C))]
    frame_by_slot = {slot: _slot_frame(slot, r0s, c0s, int(outer), int(t)) for slot in all_slots}

    chosen = _choose_path_slots(int(H), int(W), int(bg), int(border), frame_by_slot, int(outer), int(t))
    if chosen is None:
        return None
    path_slots, path_bridge_cells = chosen

    decoy_slots = _choose_decoy_slots(path_slots, path_bridge_cells, frame_by_slot)
    if decoy_slots is None:
        return None

    path_colors = list(map(int, np.random.choice(np.array(inner_palette, dtype=int), size=int(path_len), replace=False)))

    color_by_slot: dict[tuple[int, int], int] = {}
    for i, slot in enumerate(path_slots):
        color_by_slot[slot] = int(path_colors[int(i)])
    for i, slot in enumerate(decoy_slots, start=1):
        color_by_slot[slot] = int(path_colors[int(i)])

    out = np.full((int(H), int(W)), int(bg), dtype=int)
    used_slots = list(path_slots) + list(decoy_slots)
    for slot in used_slots:
        f = frame_by_slot[slot]
        r0, c0, _r1, _c1 = map(int, f["bbox"])
        _draw_frame(
            out,
            r0=int(r0),
            c0=int(c0),
            outer=int(outer),
            t=int(t),
            border=int(border),
            inner_col=int(color_by_slot[slot]),
        )

    bridge_cells_all: set[tuple[int, int]] = set()
    for i in range(int(path_len) - 1):
        a_slot = path_slots[int(i)]
        b_slot = path_slots[int(i + 1)]
        cells = _bridge_cells(frame_by_slot[a_slot], frame_by_slot[b_slot])
        if not cells:
            return None
        if cells & bridge_cells_all:
            return None
        for r, c in cells:
            if not (0 <= int(r) < int(H) and 0 <= int(c) < int(W)):
                return None
            if int(out[int(r), int(c)]) != int(bg):
                return None
        for r, c in cells:
            out[int(r), int(c)] = int(color_by_slot[a_slot])
        bridge_cells_all |= set(cells)

    ctrl_colors = [int(color_by_slot[slot]) for slot in path_slots]
    ctrl_cols = [int(1 + 2 * i) for i in range(int(len(ctrl_colors)))]
    if int(ctrl_cols[-1]) >= int(W):
        return None
    for col, col_val in zip(ctrl_cols, ctrl_colors):
        out[int(ctrl_r), int(col)] = int(col_val)

    inp = out.copy()
    for r, c in bridge_cells_all:
        inp[int(r), int(c)] = int(bg)

    if bool(np.random.randint(0, 2)):
        inp = np.fliplr(inp).copy()
        out = np.fliplr(out).copy()

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
