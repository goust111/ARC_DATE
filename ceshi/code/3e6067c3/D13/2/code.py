# Variant of 3e6067c3
# Dimension: D13 — 桥梁与已有结构的交互 (meta)
# Change: 桥梁仅在背景色区域绘制，不覆盖任何已有的方块或先前桥梁... → 桥梁避障绕行

from __future__ import annotations

from collections import Counter, defaultdict
import heapq

import numpy as np

from arc_utils import get_bg, in_bounds, label_mask

# Design analysis for the D13 variant
# ==================================
# a) Concrete behavior / trace
#    Original task: consecutive control colors select two framed squares that are aligned in the same row or
#    column band; the solver fills the whole straight gap rectangle between them.
#    Variant: the same source/target selection stays in place, but now some extra framed squares sit directly
#    in the straight corridor. The bridge therefore cannot be a raw rectangle any more. Instead it starts from
#    the source-side "port" cell just outside the frame, reaches the target-side port using only background
#    cells, and prefers minimum-turn paths. In generated examples this produces clean U-shaped detours around
#    a blocker frame. Small trace:
#      - original: A . . X . . B  -> fill every dot between A and B, overwriting X
#      - variant:  A . . X . . B  -> route above or below X, e.g. go up, cross, go down
#    The changed stage is only bridge drawing; frame parsing, control-row parsing, and source/target matching
#    remain the same.
#
# b) Complexity calibration
#    A full arbitrary maze would be too hard to infer from 2-4 ARC examples. To keep the new behavior visible
#    but still learnable, the generator uses:
#      - fixed 3x3 framed squares with 1-pixel inner color,
#      - 4-6 control-path squares,
#      - 1-2 blocker squares placed exactly on some straight row/column segment.
#    This guarantees repeated evidence for "straight when free, detour when blocked" without introducing many
#    unrelated degrees of freedom.
#
# c) Hidden assumptions in the original main() that no longer hold
#      - old draw_bridge() assumed every selected pair could be solved by one straight horizontal/vertical
#        rectangle derived from bbox endpoints; this fails once another framed square occupies the corridor.
#      - old code also assumed unconditional painting was harmless because only background was in the gap; in
#        the variant, unconditional painting would overwrite blocker squares or earlier bridges.
#      - candidate scoring based on row/column alignment is still valid because the generator preserves aligned
#        consecutive targets; only the rendering phase changes.
#    The new solver therefore replaces rectangle fill with a shortest path over background cells on the current
#    output grid, where all non-background cells are obstacles.
#
# d) Placement legality
#    The generator builds a coarse lattice, places path vertices on a simple rectilinear polyline, and optionally
#    places blocker frames on intermediate lattice cells of some segments. All frames are disjoint because each
#    slot maps to one 3x3 frame with a fixed pitch of 5. A bridge is accepted only if the same path planner used
#    by the solver can route it on the partially painted output. This preserves non-overlap, in-bounds behavior,
#    and solver/generator agreement.
#
# e) Visual observability
#    A human can see the difference immediately: some bridges are still straight, but others bend around an
#    obviously visible framed obstacle. The control row remains the same cue as in the seed task, so the only
#    new inference is "bridges avoid squares instead of crossing through them".
#
# f) Visual cleanliness
#    The scene contains only three element types: framed squares, thin colored bridges, and the control row.
#    Because bridges are 1-pixel wide and are forbidden from crossing non-background cells, clutter is limited.
#    The fixed pitch leaves two blank cells between adjacent frame boxes, which is enough room for neat U-turns.
#
# g) Color inference
#    Colors keep their original meaning: each bridge is colored by the source square in the control sequence.
#    No hidden numeric ordering is used. Colors are globally unique across all frames in generated samples, so
#    blocker squares cannot be mistaken for target repeats.
#
# h) Variant-specific metric used during validation
#    A sample is considered to exercise D13 if at least one consecutive pair has a blocked direct corridor,
#    forcing the planned path to be longer than the direct Manhattan corridor between its two ports. The test
#    script later measures how often this happens and whether the output differs from the old straight-bridge
#    behavior on the same input.
#
# i) New solver assumptions and how the generator guarantees them
#      - consecutive control targets remain aligned in one row or one column band;
#      - each target color identifies exactly one frame;
#      - a background-only path exists between source and target ports after earlier bridges are painted.
#    The generator enforces all three conditions before returning a sample.


_DIGITS = [int(i) for i in range(10)]
_DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]
_START_DIR = 4


def _most_common_nonbg(g: np.ndarray, bg: int) -> int | None:
    cnt = Counter(map(int, g.flatten()))
    for col, _k in cnt.most_common():
        if int(col) != int(bg):
            return int(col)
    return None


def _draw_frame(g: np.ndarray, *, r0: int, c0: int, outer: int, t: int, border: int, inner_col: int) -> None:
    g[int(r0) : int(r0 + outer), int(c0) : int(c0 + outer)] = int(border)
    g[int(r0 + t) : int(r0 + outer - t), int(c0 + t) : int(c0 + outer - t)] = int(inner_col)


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
    return frames


def _center(f: dict) -> tuple[int, int]:
    r0, c0, r1, c1 = map(int, f["bbox"])
    return int((r0 + r1) // 2), int((c0 + c1) // 2)


def _aligned(f1: dict, f2: dict) -> tuple[bool, bool]:
    a = f1["inner_bbox"]
    b = f2["inner_bbox"]
    same_row = bool(int(a[0]) == int(b[0]) and int(a[2]) == int(b[2]))
    same_col = bool(int(a[1]) == int(b[1]) and int(a[3]) == int(b[3]))
    return same_row, same_col


def _bridge_ports(fa: dict, fb: dict) -> tuple[tuple[int, int], tuple[int, int], str]:
    (r0a, c0a, r1a, c1a) = map(int, fa["bbox"])
    (r0b, c0b, r1b, c1b) = map(int, fb["bbox"])
    (cra, cca) = _center(fa)
    (crb, ccb) = _center(fb)
    sr, sc = _aligned(fa, fb)
    if sr:
        if int(c1a) < int(c0b):
            return (int(cra), int(c1a + 1)), (int(crb), int(c0b - 1)), "h"
        return (int(cra), int(c0a - 1)), (int(crb), int(c1b + 1)), "h"
    if sc:
        if int(r1a) < int(r0b):
            return (int(r1a + 1), int(cca)), (int(r0b - 1), int(ccb)), "v"
        return (int(r0a - 1), int(cca)), (int(r1b + 1), int(ccb)), "v"
    raise AssertionError("Consecutive frames must stay aligned in this variant")


def _direct_bridge_cells(fa: dict, fb: dict) -> list[tuple[int, int]]:
    start, goal, axis = _bridge_ports(fa, fb)
    cells: list[tuple[int, int]] = []
    if axis == "h":
        step = 1 if int(goal[1]) >= int(start[1]) else -1
        for c in range(int(start[1]), int(goal[1]) + step, step):
            cells.append((int(start[0]), int(c)))
        return cells
    step = 1 if int(goal[0]) >= int(start[0]) else -1
    for r in range(int(start[0]), int(goal[0]) + step, step):
        cells.append((int(r), int(start[1])))
    return cells


def _plan_bridge_cells(board: np.ndarray, fa: dict, fb: dict, bg: int) -> list[tuple[int, int]] | None:
    start, goal, axis = _bridge_ports(fa, fb)
    H, W = map(int, board.shape)
    if not in_bounds((int(H), int(W)), int(start[0]), int(start[1])):
        return None
    if not in_bounds((int(H), int(W)), int(goal[0]), int(goal[1])):
        return None
    if int(board[int(start[0]), int(start[1])]) != int(bg):
        return None
    if int(board[int(goal[0]), int(goal[1])]) != int(bg):
        return None

    if axis == "h":
        primary = 1 if int(goal[1]) > int(start[1]) else 3
        dir_order = [int(primary), 0, 2, int(3 if primary == 1 else 1)]
    else:
        primary = 2 if int(goal[0]) > int(start[0]) else 0
        dir_order = [int(primary), 1, 3, int(0 if primary == 2 else 2)]

    pq: list[tuple[int, int, int, int, int]] = [(0, 0, int(start[0]), int(start[1]), int(_START_DIR))]
    best: dict[tuple[int, int, int], tuple[int, int]] = {
        (int(start[0]), int(start[1]), int(_START_DIR)): (0, 0)
    }
    prev: dict[tuple[int, int, int], tuple[int, int, int] | None] = {
        (int(start[0]), int(start[1]), int(_START_DIR)): None
    }
    end_state: tuple[int, int, int] | None = None

    while pq:
        turns, steps, r, c, d = heapq.heappop(pq)
        state = (int(r), int(c), int(d))
        if best.get(state) != (int(turns), int(steps)):
            continue
        if int(r) == int(goal[0]) and int(c) == int(goal[1]):
            end_state = state
            break
        for nd in dir_order:
            dr, dc = _DIRS[int(nd)]
            rr = int(r + dr)
            cc = int(c + dc)
            if not in_bounds((int(H), int(W)), int(rr), int(cc)):
                continue
            if (int(rr), int(cc)) != (int(goal[0]), int(goal[1])) and int(board[int(rr), int(cc)]) != int(bg):
                continue
            nturns = int(turns + (0 if int(d) == int(_START_DIR) or int(d) == int(nd) else 1))
            nsteps = int(steps + 1)
            nstate = (int(rr), int(cc), int(nd))
            old = best.get(nstate)
            cand = (int(nturns), int(nsteps))
            if old is None or cand < old:
                best[nstate] = cand
                prev[nstate] = state
                heapq.heappush(pq, (int(nturns), int(nsteps), int(rr), int(cc), int(nd)))

    if end_state is None:
        return None

    rev: list[tuple[int, int]] = []
    cur: tuple[int, int, int] | None = end_state
    while cur is not None:
        rev.append((int(cur[0]), int(cur[1])))
        cur = prev[cur]
    rev.reverse()
    return rev


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
    assert not (border is None)

    frames = _detect_frames(g, int(border))
    assert not (not frames)

    by: dict[int, list[dict]] = defaultdict(list)
    for f in frames:
        by[int(f["inner"])].append(f)

    max_frame_r = max(int(f["bbox"][2]) for f in frames)
    ctrl_r = None
    for r in range(int(H - 1), int(max_frame_r), -1):
        if int((g[int(r)] != int(bg)).sum()) >= 2:
            ctrl_r = int(r)
            break
    assert not (ctrl_r is None)

    ctrl = [int(g[int(ctrl_r), int(c)]) for c in range(int(W)) if int(g[int(ctrl_r), int(c)]) != int(bg)]
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
                (r1_, c1_) = _center(f)
                (r2_, c2_) = _center(g2)
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
            (r1_, c1_) = _center(cur)
            (r2_, c2_) = _center(f)
            d = abs(int(r1_) - int(r2_)) + abs(int(c1_) - int(c2_))
            cand = (int(al), -int(d))
            if best is None or cand > best:
                best = cand
                bestf = f
        nxt = bestf
        assert nxt is not None
        cells = _plan_bridge_cells(out, cur, nxt, int(bg))
        assert cells is not None
        for r, c in cells:
            out[int(r), int(c)] = int(a)
        cur = nxt

    return out


def _segment_cells(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    (r0, c0) = map(int, a)
    (r1, c1) = map(int, b)
    if int(r0) == int(r1):
        step = 1 if int(c1) >= int(c0) else -1
        return [(int(r0), int(c)) for c in range(int(c0), int(c1) + step, step)]
    step = 1 if int(r1) >= int(r0) else -1
    return [(int(r), int(c0)) for r in range(int(r0), int(r1) + step, step)]


def _sample_path_slots(R: int, C: int, n_path: int) -> tuple[list[tuple[int, int]], list[list[tuple[int, int]]]] | None:
    for _ in range(300):
        start = (int(np.random.randint(0, int(R))), int(np.random.randint(0, int(C))))
        path = [start]
        used = {start}
        segments: list[list[tuple[int, int]]] = []
        last_axis = None
        ok = True
        for step_idx in range(int(n_path) - 1):
            opts: list[tuple[tuple[int, int], list[tuple[int, int]], int]] = []
            axes = [0, 1]
            if last_axis is not None and int(np.random.randint(0, 2)) == 0:
                axes = [1 - int(last_axis), int(last_axis)]
            for axis in axes:
                for sign in (-1, 1):
                    for dist in (1, 2, 3):
                        r, c = map(int, path[-1])
                        rr = int(r + (dist * sign if axis == 0 else 0))
                        cc = int(c + (dist * sign if axis == 1 else 0))
                        if not in_bounds((int(R), int(C)), int(rr), int(cc)):
                            continue
                        seg = _segment_cells((int(r), int(c)), (int(rr), int(cc)))
                        if any((int(sr), int(sc)) in used for sr, sc in seg[1:]):
                            continue
                        if int(step_idx) == int(n_path - 2):
                            rows = {int(x[0]) for x in path} | {int(rr)}
                            cols = {int(x[1]) for x in path} | {int(cc)}
                            if len(rows) < 2 or len(cols) < 2:
                                continue
                        opts.append(((int(rr), int(cc)), seg, int(axis)))
            if not opts:
                ok = False
                break
            endpoint, seg, axis = opts[int(np.random.randint(0, len(opts)))]
            path.append(endpoint)
            used |= {(int(sr), int(sc)) for sr, sc in seg[1:]}
            segments.append(seg)
            last_axis = int(axis)
        if not ok:
            continue
        if not any(len(seg) >= 3 for seg in segments):
            continue
        return path, segments
    return None


def _gen_attempt() -> np.ndarray | None:
    bg = int(np.random.choice(np.array(_DIGITS, dtype=int)))
    border = int(np.random.choice(np.array([d for d in _DIGITS if int(d) != int(bg)], dtype=int)))
    inner_palette = [int(d) for d in _DIGITS if int(d) not in (int(bg), int(border))]

    outer = 3
    t = 1
    R = int(np.random.randint(4, 6))
    C = int(np.random.randint(4, 6))
    n_path = int(np.random.randint(4, 7))

    sampled = _sample_path_slots(int(R), int(C), int(n_path))
    if sampled is None:
        return None
    path_slots, segments = sampled

    blocker_candidates = [seg[1:-1] for seg in segments if len(seg) >= 3]
    blocker_candidates = [cells for cells in blocker_candidates if cells]
    if not blocker_candidates:
        return None

    max_blockers = min(2, len(blocker_candidates), len(inner_palette) - int(n_path))
    if max_blockers < 1:
        return None
    n_blockers = int(np.random.randint(1, int(max_blockers) + 1))
    chosen_seg_ids = np.random.choice(np.arange(len(blocker_candidates)), size=int(n_blockers), replace=False)
    blocker_slots: list[tuple[int, int]] = []
    for seg_id in np.array(chosen_seg_ids, dtype=int).tolist():
        cells = blocker_candidates[int(seg_id)]
        blocker_slots.append(cells[int(np.random.randint(0, len(cells)))])

    all_slots = list(path_slots) + list(blocker_slots)
    if len(set(all_slots)) != len(all_slots):
        return None
    if len(all_slots) > len(inner_palette):
        return None

    top_margin = 2
    left_margin = 2
    pitch = 5
    ctrl_gap = int(np.random.randint(2, 5))
    right_margin = 2
    bottom_margin = 1

    frames: list[dict] = []
    slot_to_frame: dict[tuple[int, int], dict] = {}
    for rr, cc in all_slots:
        r0 = int(top_margin + int(rr) * int(pitch))
        c0 = int(left_margin + int(cc) * int(pitch))
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
    ctrl_r = int(max_r1 + ctrl_gap)
    H = int(ctrl_r + 1 + bottom_margin)
    W = int(max_c1 + 1 + right_margin)
    if not (7 <= int(H) <= 30 and 7 <= int(W) <= 30):
        return None

    color_pool = np.random.permutation(np.array(inner_palette, dtype=int)).tolist()
    path_colors = [int(color_pool[i]) for i in range(int(n_path))]
    blocker_colors = [int(color_pool[int(n_path) + i]) for i in range(int(n_blockers))]

    colors_by_slot: dict[tuple[int, int], int] = {}
    for slot, col in zip(path_slots, path_colors):
        colors_by_slot[(int(slot[0]), int(slot[1]))] = int(col)
    for slot, col in zip(blocker_slots, blocker_colors):
        colors_by_slot[(int(slot[0]), int(slot[1]))] = int(col)

    out = np.full((int(H), int(W)), int(bg), dtype=int)
    for f in frames:
        slot = (int(f["slot"][0]), int(f["slot"][1]))
        r0, c0, _r1, _c1 = map(int, f["bbox"])
        _draw_frame(
            out,
            r0=int(r0),
            c0=int(c0),
            outer=int(outer),
            t=int(t),
            border=int(border),
            inner_col=int(colors_by_slot[slot]),
        )

    bridge_cells_all: set[tuple[int, int]] = set()
    detour_count = 0
    for a_slot, b_slot in zip(path_slots, path_slots[1:]):
        fa = slot_to_frame[(int(a_slot[0]), int(a_slot[1]))]
        fb = slot_to_frame[(int(b_slot[0]), int(b_slot[1]))]
        direct = _direct_bridge_cells(fa, fb)
        blocked = any(int(out[int(r), int(c)]) != int(bg) for r, c in direct)
        cells = _plan_bridge_cells(out, fa, fb, int(bg))
        if cells is None:
            return None
        if any((int(r), int(c)) in bridge_cells_all for r, c in cells):
            return None
        if any(int(out[int(r), int(c)]) != int(bg) for r, c in cells):
            return None
        if blocked:
            if len(cells) <= len(direct):
                return None
            detour_count += 1
        paint = int(colors_by_slot[(int(a_slot[0]), int(a_slot[1]))])
        for r, c in cells:
            out[int(r), int(c)] = int(paint)
        bridge_cells_all |= {(int(r), int(c)) for r, c in cells}

    if detour_count < 1:
        return None

    ctrl_cols = [int(1 + 2 * i) for i in range(int(n_path))]
    if int(ctrl_cols[-1]) >= int(W):
        return None
    for col, cidx in zip(path_colors, ctrl_cols):
        out[int(ctrl_r), int(cidx)] = int(col)

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
