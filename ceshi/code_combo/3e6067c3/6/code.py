# Combo variant of 3e6067c3 (index 6)
# combo_spec: D2/1+D5/0
# Name: 最远优先+终点色桥梁

from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np

from arc_utils import get_bg, label_mask

"""
Design analysis for combo D2/1 + D5/0
=====================================

Spec-resolution note
- The provided D2/1 single-variant text says "strict row/column alternation", but the combo name and the
  fusion description both explicitly require "when multiple aligned candidates exist, choose the farthest
  instead of the nearest".
- This implementation follows the combo name plus the fusion description as the final target behavior,
  and treats the D2/1 prose as a mismatched reference note rather than the operative rule.

a) Concrete behavior and trace
- Visual scene stays close to the source task: equal-sized framed squares, one sparse control row below,
  and bridges drawn only in the gaps between frames.
- Original task behavior:
  from current square color A, the next control color B names a destination square. Among B-colored
  candidates aligned in the same row band or column band, the original solver prefers the nearest one.
  The bridge is then painted with the source color A.
- Combo behavior:
  keep the same aligned-candidate prerequisite, but reverse the distance preference: choose the farthest
  aligned B-square. Then paint the bridge with the destination color B.
- Small trace:
  suppose current square is color 2. The next control color is 7. There are exactly two aligned 7-squares:
  one one slot to the left and one three slots to the right. Old logic chooses the left 7 because it is
  nearer, and paints that short bridge with color 2. Combo logic chooses the right 7 because it is farther,
  and paints that longer bridge with color 7. This simultaneously changes both geometry and bridge color.

b) Complexity calibration
- If each target color appeared only once, the "choose farthest" rule would never activate. Therefore the
  generator gives every non-start path color exactly two instances: one nearer decoy and one farther true
  destination aligned with the current path square.
- If the farther target were only one cell farther, the visual difference would be weak. The generator
  requires every intended step to have lattice distance at least 2, and usually 3+, so long bridges are
  common.
- Path colors remain unique across path positions. This keeps the task readable while still forcing a real
  same-color disambiguation at every step.

c) Hidden assumptions broken in the original solver
- Original candidate scoring used `(aligned, -distance)`, which assumes nearest wins once alignment holds.
  In the combo, that must become "aligned first, then farthest".
- Original bridge painting used the source color `ctrl[i]`. In the combo, the bridge must use the chosen
  destination frame's inner color / `ctrl[i+1]`.
- Original generator relied on row/column color uniqueness to remove ambiguity. That would erase the combo,
  so the new generator intentionally creates aligned same-color pairs on each step.
- Start-frame logic in the original solver could get away with a single local heuristic. Here the solver
  traces from all possible starts of the first color and keeps any path that remains valid under the new
  farthest rule; the generator still makes the start unique for clarity.

d) Placement legality
- Generation happens on a discrete slot lattice first. Each path step reserves both the intended far target
  and its near decoy, and also reserves the empty slot corridor between current->target and current->decoy.
- Reserving both corridors prevents later frames from being dropped onto either the intended bridge or the
  old-nearest counterfactual bridge, so no frame/bridge overlap is introduced.
- After mapping slots to pixels, the sample is rejected if any bridge cell leaves bounds, touches a frame,
  or overlaps another bridge.

e) Visual observability
- Humans can see the new rule because each step presents a repeated destination color with two aligned
  options, and the output consistently connects to the farther one rather than the nearer one.
- The D5 part is also visible: every bridge segment matches the square it reaches, not the square it leaves.
- Because path colors are unique, these two clues do not get muddied by unrelated color repetition.

f) Visual cleanliness
- The scene uses only framed squares, empty bridge corridors, and one control row. No extra marker layer is
  added.
- Decoys are placed on the opposite side of the current frame from the true target, so the long intended
  bridge stays visually unobstructed instead of threading through another frame.
- Bridge overlap is explicitly rejected, keeping outputs readable.

g) Color inferability
- No color remapping is introduced: the control row still names frame colors, and the bridge color is simply
  copied from the destination frame.
- The color vocabulary stays small because only path colors are duplicated, and each duplication has a clear
  structural role: "near decoy vs far true destination".

h) Anti-degeneration metric used after implementation
- Metric 1: for each step, compare the farthest aligned candidate with the nearest aligned candidate for the
  next control color. The sample is non-degenerate when they differ. The generator is designed so this holds
  on every step.
- Metric 2: for each bridge, check whether source color != destination color. Because path colors are unique,
  every intended bridge visibly changes color under the D5 rule.

i) New solver assumptions and generator guarantees
- Solver assumption: for each step there is at least one aligned candidate of the next control color.
  Generator guarantees this by constructing the path first.
- Solver assumption: if multiple candidates tie on maximum distance, bbox order is an acceptable stable
  tie-break. Generator usually makes the far target unique, but the solver is deterministic either way.
- Solver assumption: the control row still lists the path colors in order. Generator emits exactly that.
- Solver assumption: bridge geometry is still straight row-band or column-band rectangles. Generator only
  creates aligned path transitions, so the original bridge rasterization shape remains valid.
"""


_DIGITS = [int(i) for i in range(10)]


def main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    h, w = g.shape
    bg = int(get_bg(g))
    border = _most_common_nonbg(g, int(bg))
    assert border is not None

    frames = _extract_frames(g, int(border))
    assert frames

    by: dict[int, list[dict]] = defaultdict(list)
    for f in frames:
        by[int(f["inner"])].append(f)

    ctrl_r = _find_control_row(g, int(bg), frames)
    ctrl = [int(g[int(ctrl_r), int(c)]) for c in range(int(w)) if int(g[int(ctrl_r), int(c)]) != int(bg)]
    assert len(ctrl) >= 2

    start_color = int(ctrl[0])
    assert start_color in by

    valid_paths: list[list[dict]] = []
    for start in sorted(by[start_color], key=_bbox_key):
        traced = _trace_path(start, ctrl, by, prefer="farthest")
        if traced is not None:
            valid_paths.append(traced)
    assert valid_paths

    path = min(valid_paths, key=_path_key)
    out = g.copy()
    for i in range(int(len(path)) - 1):
        cells = _bridge_cells(path[int(i)], path[int(i) + 1])
        assert cells
        dst_color = int(path[int(i) + 1]["inner"])
        for r, c in cells:
            if 0 <= int(r) < int(h) and 0 <= int(c) < int(w):
                out[int(r), int(c)] = int(dst_color)
    return out


def _most_common_nonbg(g: np.ndarray, bg: int) -> int | None:
    cnt = Counter(map(int, g.flatten()))
    for col, _k in cnt.most_common():
        if int(col) != int(bg):
            return int(col)
    return None


def _bbox_key(f: dict) -> tuple[int, int, int, int]:
    return tuple(map(int, f["bbox"]))


def _path_key(path: list[dict]) -> tuple[tuple[int, int, int, int], ...]:
    return tuple(_bbox_key(f) for f in path)


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
        hh = int(r1 - r0 + 1)
        ww = int(c1 - c0 + 1)
        if hh < 3 or ww < 3:
            continue

        thickness = None
        inner_col = None
        for tt in range(1, int(min(hh, ww) // 2) + 1):
            inner = g[int(r0 + tt) : int(r1 - tt) + 1, int(c0 + tt) : int(c1 - tt) + 1]
            if inner.size == 0:
                continue
            if bool((inner == int(border)).any()):
                continue
            uniq = set(map(int, inner.flatten()))
            if len(uniq) != 1:
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

            thickness = int(tt)
            inner_col = int(next(iter(uniq)))
            break

        if thickness is None or inner_col is None:
            continue

        frames.append(
            {
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
                "t": int(thickness),
                "inner": int(inner_col),
                "inner_bbox": (int(r0 + thickness), int(c0 + thickness), int(r1 - thickness), int(c1 - thickness)),
            }
        )
    return frames


def _find_control_row(g: np.ndarray, bg: int, frames: list[dict]) -> int:
    max_frame_r = max(int(f["bbox"][2]) for f in frames)
    for r in range(int(g.shape[0] - 1), int(max_frame_r), -1):
        if int((g[int(r)] != int(bg)).sum()) >= 2:
            return int(r)
    raise AssertionError("control row not found")


def _center(f: dict) -> tuple[int, int]:
    r0, c0, r1, c1 = map(int, f["bbox"])
    return int((r0 + r1) // 2), int((c0 + c1) // 2)


def _aligned(f1: dict, f2: dict) -> tuple[bool, bool]:
    a = f1["inner_bbox"]
    b = f2["inner_bbox"]
    same_row = bool(int(a[0]) == int(b[0]) and int(a[2]) == int(b[2]))
    same_col = bool(int(a[1]) == int(b[1]) and int(a[3]) == int(b[3]))
    return same_row, same_col


def _aligned_distance(f1: dict, f2: dict) -> int | None:
    same_row, same_col = _aligned(f1, f2)
    if not (same_row or same_col):
        return None
    r1, c1 = _center(f1)
    r2, c2 = _center(f2)
    return abs(int(r1) - int(r2)) + abs(int(c1) - int(c2))


def _choose_candidate(cur: dict, cands: list[dict], *, prefer: str) -> dict | None:
    scored: list[tuple[int, tuple[int, int, int, int], dict]] = []
    for f in cands:
        dist = _aligned_distance(cur, f)
        if dist is None or int(dist) == 0:
            continue
        scored.append((int(dist), _bbox_key(f), f))
    if not scored:
        return None
    if prefer == "farthest":
        scored.sort(key=lambda item: (-int(item[0]), item[1]))
    else:
        scored.sort(key=lambda item: (int(item[0]), item[1]))
    return scored[0][2]


def _trace_path(start: dict, ctrl: list[int], by: dict[int, list[dict]], *, prefer: str) -> list[dict] | None:
    path = [start]
    cur = start
    for i in range(int(len(ctrl)) - 1):
        nxt_color = int(ctrl[int(i) + 1])
        nxt = _choose_candidate(cur, by.get(int(nxt_color), []), prefer=prefer)
        if nxt is None:
            return None
        cur = nxt
        path.append(cur)
    return path


def _draw_frame(g: np.ndarray, *, r0: int, c0: int, outer: int, t: int, border: int, inner_col: int) -> None:
    g[int(r0) : int(r0 + outer), int(c0) : int(c0 + outer)] = int(border)
    g[int(r0 + t) : int(r0 + outer - t), int(c0 + t) : int(c0 + outer - t)] = int(inner_col)


def _bridge_cells(fa: dict, fb: dict) -> set[tuple[int, int]]:
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


def _segment_interior_slots(a: tuple[int, int], b: tuple[int, int]) -> set[tuple[int, int]]:
    ra, ca = map(int, a)
    rb, cb = map(int, b)
    out: set[tuple[int, int]] = set()
    if int(ra) == int(rb):
        lo, hi = sorted((int(ca), int(cb)))
        for cc in range(int(lo) + 1, int(hi)):
            out.add((int(ra), int(cc)))
    elif int(ca) == int(cb):
        lo, hi = sorted((int(ra), int(rb)))
        for rr in range(int(lo) + 1, int(hi)):
            out.add((int(rr), int(ca)))
    return out


def _segment_clear(a: tuple[int, int], b: tuple[int, int], blocked: set[tuple[int, int]]) -> bool:
    ra, ca = map(int, a)
    rb, cb = map(int, b)
    if not (int(ra) == int(rb) or int(ca) == int(cb)):
        return False
    if b in blocked:
        return False
    return not bool(_segment_interior_slots(a, b) & blocked)


def _pair_candidates(
    cur: tuple[int, int],
    *,
    shape: tuple[int, int],
    blocked: set[tuple[int, int]],
) -> list[tuple[str, tuple[int, int], tuple[int, int]]]:
    h, w = map(int, shape)
    r, c = map(int, cur)
    out: list[tuple[str, tuple[int, int], tuple[int, int]]] = []
    for axis in ("row", "col"):
        actuals: list[tuple[int, tuple[int, int]]] = []
        if axis == "row":
            for cc in range(int(w)):
                if int(cc) == int(c):
                    continue
                cand = (int(r), int(cc))
                dist = abs(int(cc) - int(c))
                if int(dist) < 2:
                    continue
                if not _segment_clear(cur, cand, blocked):
                    continue
                actuals.append((int(dist), cand))
        else:
            for rr in range(int(h)):
                if int(rr) == int(r):
                    continue
                cand = (int(rr), int(c))
                dist = abs(int(rr) - int(r))
                if int(dist) < 2:
                    continue
                if not _segment_clear(cur, cand, blocked):
                    continue
                actuals.append((int(dist), cand))
        np.random.shuffle(actuals)
        actuals.sort(key=lambda item: (-int(item[0]), item[1]))

        for dist, nxt in actuals:
            if axis == "row":
                dir_sign = 1 if int(nxt[1]) > int(c) else -1
                decoys = []
                for dec_d in range(1, int(dist)):
                    dec = (int(r), int(c - dir_sign * dec_d))
                    if not (0 <= int(dec[1]) < int(w)):
                        continue
                    if not _segment_clear(cur, dec, blocked):
                        continue
                    decoys.append((int(dec_d), dec))
            else:
                dir_sign = 1 if int(nxt[0]) > int(r) else -1
                decoys = []
                for dec_d in range(1, int(dist)):
                    dec = (int(r - dir_sign * dec_d), int(c))
                    if not (0 <= int(dec[0]) < int(h)):
                        continue
                    if not _segment_clear(cur, dec, blocked):
                        continue
                    decoys.append((int(dec_d), dec))
            np.random.shuffle(decoys)
            decoys.sort(key=lambda item: (int(item[0]), item[1]))
            for _ddec, dec in decoys:
                out.append((axis, nxt, dec))
    return out


def _search_layout(
    *,
    shape: tuple[int, int],
    path_len: int,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]], list[str]] | None:
    starts = [(int(r), int(c)) for r in range(int(shape[0])) for c in range(int(shape[1]))]
    np.random.shuffle(starts)

    def rec(
        path: list[tuple[int, int]],
        decoys: list[tuple[int, int]],
        axes: list[str],
        blocked: set[tuple[int, int]],
        max_dist_seen: int,
    ) -> tuple[list[tuple[int, int]], list[tuple[int, int]], list[str]] | None:
        if len(path) == int(path_len):
            rows = {int(r) for r, _c in path}
            cols = {int(c) for _r, c in path}
            if len(rows) < 2 or len(cols) < 2:
                return None
            if "row" not in axes or "col" not in axes:
                return None
            if int(max_dist_seen) < 3:
                return None
            return list(path), list(decoys), list(axes)

        cur = path[-1]
        pairs = _pair_candidates(cur, shape=shape, blocked=blocked)
        np.random.shuffle(pairs)
        pairs.sort(
            key=lambda item: (
                -(
                    abs(int(item[1][0]) - int(cur[0])) + abs(int(item[1][1]) - int(cur[1]))
                ),
                item[0],
                item[1],
                item[2],
            )
        )
        for axis, nxt, dec in pairs:
            step_dist = abs(int(nxt[0]) - int(cur[0])) + abs(int(nxt[1]) - int(cur[1]))
            new_blocked = set(blocked)
            new_blocked.add(nxt)
            new_blocked.add(dec)
            new_blocked |= _segment_interior_slots(cur, nxt)
            new_blocked |= _segment_interior_slots(cur, dec)
            path.append(nxt)
            decoys.append(dec)
            axes.append(axis)
            solved = rec(path, decoys, axes, new_blocked, max(int(max_dist_seen), int(step_dist)))
            if solved is not None:
                return solved
            axes.pop()
            decoys.pop()
            path.pop()
        return None

    for start in starts:
        solved = rec([start], [], [], {start}, 0)
        if solved is not None:
            return solved
    return None


def _gen_attempt() -> np.ndarray | None:
    bg = int(np.random.choice(np.array(_DIGITS, dtype=int)))
    border = int(np.random.choice(np.array([d for d in _DIGITS if int(d) != int(bg)], dtype=int)))
    inner_palette = [int(d) for d in _DIGITS if int(d) not in (int(bg), int(border))]
    if len(inner_palette) < 5:
        return None

    t = 1
    inner = 2
    outer = int(inner + 2 * t)

    slot_shape = (4, 5)
    path_len = 4
    layout = _search_layout(shape=slot_shape, path_len=path_len)
    if layout is None:
        return None
    path_slots, decoy_slots, _axes = layout

    row_gap = [1 for _ in range(int(slot_shape[0]) - 1)]
    col_gap = [1 for _ in range(int(slot_shape[1]) - 1)]
    top_margin = int(np.random.randint(1, 3))
    left_margin = int(np.random.randint(1, 3))

    r0s = [int(top_margin)]
    for k in range(int(slot_shape[0]) - 1):
        r0s.append(int(r0s[-1] + outer + row_gap[int(k)]))
    c0s = [int(left_margin)]
    for k in range(int(slot_shape[1]) - 1):
        c0s.append(int(c0s[-1] + outer + col_gap[int(k)]))

    path_colors = [int(x) for x in np.random.choice(np.array(inner_palette, dtype=int), size=int(path_len), replace=False)]

    frame_slots = list(path_slots) + list(decoy_slots)
    color_by_slot: dict[tuple[int, int], int] = {}
    for idx, slot in enumerate(path_slots):
        color_by_slot[(int(slot[0]), int(slot[1]))] = int(path_colors[int(idx)])
    for idx, slot in enumerate(decoy_slots):
        color_by_slot[(int(slot[0]), int(slot[1]))] = int(path_colors[int(idx) + 1])

    frames: list[dict] = []
    slot_to_frame: dict[tuple[int, int], dict] = {}
    for slot in frame_slots:
        rr, cc = map(int, slot)
        r0 = int(r0s[int(rr)])
        c0 = int(c0s[int(cc)])
        r1 = int(r0 + outer - 1)
        c1 = int(c0 + outer - 1)
        f = {
            "bbox": (int(r0), int(c0), int(r1), int(c1)),
            "inner_bbox": (int(r0 + t), int(c0 + t), int(r1 - t), int(c1 - t)),
            "slot": (int(rr), int(cc)),
            "inner": int(color_by_slot[(int(rr), int(cc))]),
        }
        frames.append(f)
        slot_to_frame[(int(rr), int(cc))] = f

    max_r1 = max(int(f["bbox"][2]) for f in frames)
    max_c1 = max(int(f["bbox"][3]) for f in frames)
    ctrl_gap = int(np.random.randint(2, 5))
    ctrl_r = int(max_r1 + ctrl_gap)
    bottom_margin = 1
    right_margin = int(np.random.randint(1, 4))
    height = int(ctrl_r + 1 + bottom_margin)
    width = int(max_c1 + 1 + right_margin)
    if not (7 <= int(height) <= 30 and 7 <= int(width) <= 30):
        return None

    out = np.full((int(height), int(width)), int(bg), dtype=int)
    for f in frames:
        r0, c0, _r1, _c1 = map(int, f["bbox"])
        _draw_frame(out, r0=int(r0), c0=int(c0), outer=int(outer), t=int(t), border=int(border), inner_col=int(f["inner"]))

    bridge_cells_all: set[tuple[int, int]] = set()
    for i in range(len(path_slots) - 1):
        fa = slot_to_frame[(int(path_slots[int(i)][0]), int(path_slots[int(i)][1]))]
        fb = slot_to_frame[(int(path_slots[int(i) + 1][0]), int(path_slots[int(i) + 1][1]))]
        cells = _bridge_cells(fa, fb)
        if not cells:
            return None
        if cells & bridge_cells_all:
            return None
        for r, c in cells:
            if not (0 <= int(r) < int(height) and 0 <= int(c) < int(width)):
                return None
            if int(out[int(r), int(c)]) != int(bg):
                return None
        paint = int(path_colors[int(i) + 1])
        for r, c in cells:
            out[int(r), int(c)] = int(paint)
        bridge_cells_all |= set(cells)

    ctrl_cols = [int(1 + 2 * i) for i in range(int(len(path_colors)))]
    if int(ctrl_cols[-1]) >= int(width):
        return None
    for col, cc in zip(path_colors, ctrl_cols):
        out[int(ctrl_r), int(cc)] = int(col)

    inp = out.copy()
    for r, c in bridge_cells_all:
        inp[int(r), int(c)] = int(bg)

    if int(get_bg(inp)) != int(bg):
        return None
    if int(_most_common_nonbg(inp, int(bg)) or -1) != int(border):
        return None

    by_color: dict[int, list[dict]] = defaultdict(list)
    for f in frames:
        by_color[int(f["inner"])].append(f)
    intended = _trace_path(slot_to_frame[(int(path_slots[0][0]), int(path_slots[0][1]))], path_colors, by_color, prefer="farthest")
    if intended is None:
        return None
    for i in range(len(path_colors) - 1):
        cur = intended[int(i)]
        nxt_color = int(path_colors[int(i) + 1])
        far = _choose_candidate(cur, by_color.get(int(nxt_color), []), prefer="farthest")
        near = _choose_candidate(cur, by_color.get(int(nxt_color), []), prefer="nearest")
        if far is None or near is None:
            return None
        if _bbox_key(far) == _bbox_key(near):
            return None
    if any(int(path_colors[int(i)]) == int(path_colors[int(i) + 1]) for i in range(len(path_colors) - 1)):
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
    for _ in range(800):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("Failed to generate valid input after 800 attempts")
