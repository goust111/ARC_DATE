# Combo variant of 3e6067c3 (index 11)
# combo_spec: D5/0+D6/1
# Name: 终点色桥梁+擦除控制行
#
# Design analysis for this combo variant
# -------------------------------------
# The combo-level description takes precedence over the standalone D6/1 snippet:
# here D6 contributes "erase the control row in the output", not "recolor visited
# frames". The fused task is therefore:
# - input: framed squares plus a bottom control-pixel sequence,
# - output: same framed squares, bridges inserted between consecutive control
#   colors, every bridge colored by its destination square, and the control row
#   cleared back to background.
#
# a) Concrete behavior trace:
# - Original 3e6067c3 uses control colors [c0,c1,c2,...] to choose a path of
#   aligned framed squares, then paints segment i with color ci and keeps the
#   control row visible.
# - In this combo, the geometry and path selection stay the same, but segment
#   (ci -> ci+1) is painted with ci+1 instead of ci. After all segments are
#   drawn, the entire control row is erased to background.
# - Example: if the input control row is [2,5,7] and the path is 2-square ->
#   5-square -> 7-square, the original output would show a 2-colored first
#   bridge, a 5-colored second bridge, and still display [2,5,7] below. The
#   combo output instead shows a 5-colored first bridge, a 7-colored second
#   bridge, and a blank background row where the control sequence used to be.
#
# b) Complexity calibration:
# - The generator keeps the original lattice/path structure so alignment-based
#   matching is still meaningful and the task does not collapse into arbitrary
#   nearest-neighbor linking.
# - To make "destination-colored" visually undeniable, adjacent path squares are
#   forced to have different inner colors by maintaining row/column uniqueness
#   along the lattice path. Thus every bridge segment disagrees with the source
#   color on generated samples.
# - The output control row is fully blank, so the solver cannot rely on any
#   residual colored pixels after rendering. The path information must come only
#   from the input control row.
#
# c) Hidden-assumption audit:
# - Border detection still assumes the shared frame color is the most common
#   non-background color in the input. The generator keeps that invariant.
# - Frame detection must happen on the input before any output-time erasure; the
#   solver therefore computes all frame geometry from the untouched input copy.
# - Start/candidate selection still depends only on frame geometry and the input
#   control sequence, so changing bridge colors does not affect matching.
# - `draw_bridge(..., color)` previously assumed `color == source inner`. The
#   combo changes that call-site to pass the destination inner color instead.
# - A new assumption is that the control row lies strictly below every frame, so
#   clearing it cannot destroy bridges or squares. The generator places `ctrl_r`
#   below `max_frame_r` with at least one background gap.
#
# d) Placement legality:
# - Frames are placed on a lattice with at least one background cell between
#   neighboring slots. Path steps are 4-neighbor lattice moves, so every bridge
#   occupies a clean rectangular gap between two aligned frames.
# - `_bridge_cells()` rejects non-aligned pairs and empty gaps; generation aborts
#   on any overlap, out-of-bounds cell, or failed bridge rectangle.
# - The control row uses odd columns `1,3,5,...` and the grid width is enlarged
#   when needed so control pixels never fall outside the grid.
#
# e) Visual observability:
# - Across 2-4 examples a human sees: input has a visible control sequence,
#   output has no control sequence, and each bridge matches its destination
#   square rather than its source square.
# - Because adjacent path colors are distinct and there are at least 3 bridge
#   segments, the bridge-color rule is supported by repeated evidence.
#
# f) Visual cleanliness:
# - No extra markers are introduced. The only visible changes are rectangular
#   bridges in existing gaps and a blanked control row below the objects.
# - Clearing the whole control row avoids leaving sparse colored hints that could
#   visually compete with the bridge evidence.
#
# g) Change inferability:
# - No new palette is introduced: bridge colors always come from already visible
#   square interiors.
# - The generator's non-degeneracy target is: every segment must satisfy
#   `source_color != destination_color`, so the combo output differs from a
#   "source-colored + erased-row" baseline on every generated sample.

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
import sys

import numpy as np

try:
    from arc_utils import get_bg, in_bounds, neighbors4, label_mask
except ModuleNotFoundError:
    ROOT = Path(__file__).resolve().parents[4]
    for rel in ("arc2_solver_select", "solvers"):
        cand = ROOT / rel
        if cand.exists():
            sys.path.append(str(cand))
    from arc_utils import get_bg, in_bounds, neighbors4, label_mask


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

        thickness = None
        inner_col = None
        for tt in range(1, int(min(h, w) // 2) + 1):
            inner = g[int(r0 + tt) : int(r1 - tt) + 1, int(c0 + tt) : int(c1 - tt) + 1]
            if inner.size == 0:
                continue
            if bool((inner == int(border)).any()):
                continue
            unique_inner = set(map(int, inner.flatten()))
            if len(unique_inner) != 1:
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
            inner_col = int(next(iter(unique_inner)))
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


def _center(frame: dict) -> tuple[int, int]:
    r0, c0, r1, c1 = map(int, frame["bbox"])
    return int((r0 + r1) // 2), int((c0 + c1) // 2)


def _aligned(f1: dict, f2: dict) -> tuple[bool, bool]:
    a = f1["inner_bbox"]
    b = f2["inner_bbox"]
    same_row = bool(int(a[0]) == int(b[0]) and int(a[2]) == int(b[2]))
    same_col = bool(int(a[1]) == int(b[1]) and int(a[3]) == int(b[3]))
    return same_row, same_col


def _choose_candidate(cur: dict, cands: list[dict]) -> dict | None:
    best = None
    bestf = None
    for frame in cands:
        sr, sc = _aligned(cur, frame)
        aligned_score = 1 if (sr or sc) else 0
        (r1_, c1_) = _center(cur)
        (r2_, c2_) = _center(frame)
        dist = abs(int(r1_) - int(r2_)) + abs(int(c1_) - int(c2_))
        cand = (int(aligned_score), -int(dist))
        if best is None or cand > best:
            best = cand
            bestf = frame
    return bestf


def _find_control_row(g: np.ndarray, bg: int, frames: list[dict]) -> int:
    h = int(g.shape[0])
    max_frame_r = max(int(frame["bbox"][2]) for frame in frames)
    for r in range(int(h - 1), int(max_frame_r), -1):
        if int((g[int(r)] != int(bg)).sum()) >= 2:
            return int(r)
    raise AssertionError("control row not found")


def _draw_bridge(out: np.ndarray, fa: dict, fb: dict, color: int) -> None:
    h, w = map(int, out.shape)
    (r0a, c0a, r1a, c1a) = map(int, fa["bbox"])
    (r0b, c0b, r1b, c1b) = map(int, fb["bbox"])
    inner_a = fa["inner_bbox"]
    same_row, same_col = _aligned(fa, fb)
    if same_row:
        rr0, rr1 = int(inner_a[0]), int(inner_a[2])
        if int(c1a) < int(c0b):
            cc0, cc1 = int(c1a + 1), int(c0b - 1)
        else:
            cc0, cc1 = int(c1b + 1), int(c0a - 1)
        for r in range(int(rr0), int(rr1) + 1):
            for c in range(int(cc0), int(cc1) + 1):
                if 0 <= int(r) < int(h) and 0 <= int(c) < int(w):
                    out[int(r), int(c)] = int(color)
        return
    if same_col:
        cc0, cc1 = int(inner_a[1]), int(inner_a[3])
        if int(r1a) < int(r0b):
            rr0, rr1 = int(r1a + 1), int(r0b - 1)
        else:
            rr0, rr1 = int(r1b + 1), int(r0a - 1)
        for r in range(int(rr0), int(rr1) + 1):
            for c in range(int(cc0), int(cc1) + 1):
                if 0 <= int(r) < int(h) and 0 <= int(c) < int(w):
                    out[int(r), int(c)] = int(color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect framed squares via colored bridges following a control-pixel sequence.

    Combo behavior:
    - segment color = destination square's inner color;
    - after drawing all bridges, the control row is erased to background.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    cnt = Counter(map(int, g.flatten()))
    border = None
    for col, _freq in cnt.most_common():
        if int(col) != int(bg):
            border = int(col)
            break
    assert not (border is None)

    frames = _find_frames(g, int(border))
    assert not (not frames)

    by: dict[int, list[dict]] = defaultdict(list)
    for frame in frames:
        by[int(frame["inner"])].append(frame)

    ctrl_r = _find_control_row(g, int(bg), frames)
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
        for frame in start_cands:
            for frame2 in (nxt_cands if nxt_cands else [frame]):
                sr, sc = _aligned(frame, frame2)
                aligned_score = 1 if (sr or sc) else 0
                (r1_, c1_) = _center(frame)
                (r2_, c2_) = _center(frame2)
                dist = abs(int(r1_) - int(r2_)) + abs(int(c1_) - int(c2_))
                cand = (int(aligned_score), -int(dist))
                if best is None or cand > best:
                    best = cand
                    bestf = frame
        cur = bestf

    assert not (cur is None)
    out = g.copy()

    for i in range(int(len(ctrl)) - 1):
        a = int(ctrl[i])
        b = int(ctrl[i + 1])
        assert int(cur["inner"]) == int(a)
        if b not in by:
            continue
        nxt = _choose_candidate(cur, by[b])
        if nxt is None:
            continue
        _draw_bridge(out, cur, nxt, int(b))
        cur = nxt

    out[int(ctrl_r), :] = int(bg)
    return out


def _most_common_nonbg(g: np.ndarray, bg: int) -> int | None:
    cnt = Counter(map(int, g.flatten()))
    for col, _freq in cnt.most_common():
        if int(col) != int(bg):
            return int(col)
    return None


def _draw_frame(g: np.ndarray, *, r0: int, c0: int, outer: int, t: int, border: int, inner_col: int) -> None:
    g[int(r0) : int(r0 + outer), int(c0) : int(c0 + outer)] = int(border)
    g[int(r0 + t) : int(r0 + outer - t), int(c0 + t) : int(c0 + outer - t)] = int(inner_col)


def _bridge_cells(fa: dict, fb: dict) -> set[tuple[int, int]]:
    (r0a, c0a, r1a, c1a) = map(int, fa["bbox"])
    (r0b, c0b, r1b, c1b) = map(int, fb["bbox"])
    inner_a = fa["inner_bbox"]
    inner_b = fb["inner_bbox"]
    if int(inner_a[0]) == int(inner_b[0]) and int(inner_a[2]) == int(inner_b[2]):
        rr0, rr1 = int(inner_a[0]), int(inner_a[2])
        if int(c1a) < int(c0b):
            cc0, cc1 = int(c1a + 1), int(c0b - 1)
        else:
            cc0, cc1 = int(c1b + 1), int(c0a - 1)
        if int(cc0) > int(cc1):
            return set()
        return {(int(r), int(c)) for r in range(int(rr0), int(rr1) + 1) for c in range(int(cc0), int(cc1) + 1)}
    if int(inner_a[1]) == int(inner_b[1]) and int(inner_a[3]) == int(inner_b[3]):
        cc0, cc1 = int(inner_a[1]), int(inner_a[3])
        if int(r1a) < int(r0b):
            rr0, rr1 = int(r1a + 1), int(r0b - 1)
        else:
            rr0, rr1 = int(r1b + 1), int(r0a - 1)
        if int(rr0) > int(rr1):
            return set()
        return {(int(r), int(c)) for r in range(int(rr0), int(rr1) + 1) for c in range(int(cc0), int(cc1) + 1)}
    return set()


_DIGITS = [int(i) for i in range(10)]


def _gen_attempt() -> np.ndarray | None:
    bg = int(np.random.choice(np.array(_DIGITS, dtype=int)))
    border = int(np.random.choice(np.array([d for d in _DIGITS if int(d) != int(bg)], dtype=int)))
    inner_palette = [int(d) for d in _DIGITS if int(d) not in (int(bg), int(border))]
    if len(inner_palette) < 4:
        return None

    t = int(np.random.choice(np.array([1, 1, 2], dtype=int)))
    inner = int(np.random.randint(1, 4))
    outer = int(inner + 2 * t)
    if not (3 <= int(outer) <= 7):
        return None

    rows = int(np.random.randint(2, 6))
    cols = int(np.random.randint(2, 6))
    max_slots = int(rows * cols)
    max_path = int(min(8, max_slots))
    if max_path < 4:
        return None
    n_frames = int(np.random.randint(4, max_path + 1))

    path: list[tuple[int, int]] | None = None
    for _ in range(400):
        candidate = [(0, 0)]
        visited = {(0, 0)}
        ok = True
        for _step in range(int(n_frames) - 1):
            r, c = map(int, candidate[-1])
            options = [
                (int(rr), int(cc))
                for rr, cc in neighbors4(int(r), int(c))
                if in_bounds((int(rows), int(cols)), int(rr), int(cc)) and (int(rr), int(cc)) not in visited
            ]
            if not options:
                ok = False
                break
            nxt = options[int(np.random.randint(0, len(options)))]
            candidate.append((int(nxt[0]), int(nxt[1])))
            visited.add((int(nxt[0]), int(nxt[1])))
        if not ok:
            continue
        if len({r for r, _c in candidate}) < 2 or len({c for _r, c in candidate}) < 2:
            continue
        path = candidate
        break
    if path is None:
        return None

    row_gap = [int(np.random.randint(1, 6)) for _ in range(int(rows) - 1)]
    col_gap = [int(np.random.randint(1, 6)) for _ in range(int(cols) - 1)]
    top_margin = int(np.random.randint(1, 4))
    left_margin = int(np.random.randint(1, 4))

    r0s = [int(top_margin)]
    for k in range(int(rows) - 1):
        r0s.append(int(r0s[-1] + outer + row_gap[int(k)]))
    c0s = [int(left_margin)]
    for k in range(int(cols) - 1):
        c0s.append(int(c0s[-1] + outer + col_gap[int(k)]))

    slot_to_frame: dict[tuple[int, int], dict] = {}
    frames: list[dict] = []
    for rr, cc in path:
        r0 = int(r0s[int(rr)])
        c0 = int(c0s[int(cc)])
        r1 = int(r0 + outer - 1)
        c1 = int(c0 + outer - 1)
        frame = {
            "bbox": (int(r0), int(c0), int(r1), int(c1)),
            "inner_bbox": (int(r0 + t), int(c0 + t), int(r1 - t), int(c1 - t)),
            "slot": (int(rr), int(cc)),
        }
        frames.append(frame)
        slot_to_frame[(int(rr), int(cc))] = frame

    max_r1 = max(int(frame["bbox"][2]) for frame in frames)
    max_c1 = max(int(frame["bbox"][3]) for frame in frames)

    ctrl_gap = int(np.random.randint(2, 6))
    ctrl_r = int(max_r1 + ctrl_gap)
    bottom_margin = 1
    right_margin = int(np.random.randint(1, 5))
    h = int(ctrl_r + 1 + bottom_margin)
    w = int(max(max_c1 + 1 + right_margin, 2 * n_frames + 1))
    if not (7 <= int(h) <= 30 and 7 <= int(w) <= 30):
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

    inp = np.full((int(h), int(w)), int(bg), dtype=int)
    out = np.full((int(h), int(w)), int(bg), dtype=int)
    for frame in frames:
        rr, cc = map(int, frame["slot"])
        r0, c0, _r1, _c1 = map(int, frame["bbox"])
        inner_col = int(colors_by_slot[(int(rr), int(cc))])
        _draw_frame(inp, r0=int(r0), c0=int(c0), outer=int(outer), t=int(t), border=int(border), inner_col=int(inner_col))
        _draw_frame(out, r0=int(r0), c0=int(c0), outer=int(outer), t=int(t), border=int(border), inner_col=int(inner_col))

    bridge_cells_all: set[tuple[int, int]] = set()
    for a_slot, b_slot in zip(path, path[1:]):
        fa = slot_to_frame[(int(a_slot[0]), int(a_slot[1]))]
        fb = slot_to_frame[(int(b_slot[0]), int(b_slot[1]))]
        cells = _bridge_cells(fa, fb)
        if not cells:
            return None
        if cells & bridge_cells_all:
            return None
        for r, c in cells:
            if not (0 <= int(r) < int(h) and 0 <= int(c) < int(w)):
                return None
            if int(out[int(r), int(c)]) != int(bg):
                return None
        paint = int(colors_by_slot[(int(b_slot[0]), int(b_slot[1]))])
        for r, c in cells:
            out[int(r), int(c)] = int(paint)
        bridge_cells_all |= set(cells)

    ctrl_colors = [int(colors_by_slot[(int(r), int(c))]) for r, c in path]
    ctrl_cols = [int(1 + 2 * i) for i in range(int(len(ctrl_colors)))]
    if int(ctrl_cols[-1]) >= int(w):
        return None
    for color, col in zip(ctrl_colors, ctrl_cols):
        inp[int(ctrl_r), int(col)] = int(color)

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

    if any(int(colors_by_slot[(int(a[0]), int(a[1]))]) == int(colors_by_slot[(int(b[0]), int(b[1]))]) for a, b in zip(path, path[1:])):
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(500):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("Failed to generate valid input after 500 attempts")
