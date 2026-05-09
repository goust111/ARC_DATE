# Combo variant of 3e6067c3 (index 33)
# combo_spec: D14/2+D6/1
# Name: 顶部控制行+擦除控制
#
# Design analysis for this combo
# a) Concrete behavior:
#    The source task reads an order from a bottom control row, then paints bridges between framed
#    squares while leaving the control row visible in the output. This combo changes two linked
#    stages at once: the order pixels now live on a sparse row above all squares, and that row is
#    erased after the bridges are drawn. A small example is:
#      input:  top row has colors [red, green, blue] as isolated pixels; below it sit framed
#              red/green/blue squares plus at least one unused decoy square.
#      output: red-to-green and green-to-blue bridges appear, but the top control row becomes all
#              background again.
#    The solver therefore must first find a nonstandard top control row, use it for path order,
#    and then deliberately remove the very evidence it just consumed.
# b) Complexity calibration:
#    If every square were on the path and there were no decoys, the task would visually collapse
#    into "top row disappears and bridges appear", which is still solvable but weaker. The
#    generator therefore keeps at least one non-path square so the scene still has untouched framed
#    objects that contrast with the traversed route.
# c) Hidden-assumption audit of the original main():
#    - Original control-row search assumes the order is below the frames. That fails here, so the
#      solver now scans the whole grid for a row of isolated single pixels and selects the topmost
#      such row.
#    - Original output copy preserved all source objects. That no longer holds; after drawing
#      bridges, the solver clears the detected control row back to background.
#    - Frame detection still assumes the input contains intact framed squares sharing one border
#      color. The generator preserves that.
#    - Candidate matching still assumes control colors identify frame interiors. The generator uses
#      unique path colors and keeps decoy colors out of the control row, so this remains stable.
# d) Placement legality:
#    The generator reserves vertical space so the control row is strictly above every frame, with at
#    least one full background row below it; this keeps each control pixel vertically isolated.
#    Frames occupy lattice slots with >=1 background cell between neighboring slots, and bridges are
#    drawn only in those gaps, so objects never overlap or leave the grid.
# e) Visual observability:
#    Across examples, humans can spot the same top sparse row in the input and notice that it is
#    absent in the output while bridges appear in the body of the scene. That makes both D14/2 and
#    the erase-control rule directly observable.
# f) Visual cleanliness:
#    The control row uses isolated pixels at columns 1,3,5,..., never touching frames or bridges.
#    Removing the row in the output reduces clutter instead of adding a new mark.
# g) Change inferability:
#    No hidden numeric ordering is used. The only cues are visible: top-row color order determines
#    bridge order, and the same row is blank in the output. This is inferable from a few examples.
# h) Spec-conflict resolution:
#    The provided single-variant reference for D6/1 recolors visited squares, but this combo's own
#    name and fusion description both explicitly require "erase control". That fusion-level spec is
#    treated as authoritative here, so D6/1 is realized as "the control source object is removed
#    from the output" rather than node recoloring.
# i) New solver assumptions guaranteed by the generator:
#    - Exactly one isolated multi-pixel control row exists, and it is the topmost such row.
#    - Every control color corresponds to exactly one path square.
#    - Consecutive path squares are lattice-adjacent, hence bridgeable by one straight gap band.

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
import sys

import numpy as np


_ROOT = Path(__file__).resolve()
for _parent in [_ROOT.parent] + list(_ROOT.parents):
    if (_parent / "arc_utils.py").exists():
        sys.path.insert(0, str(_parent))
        break

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

        t = None
        inner_col = None
        for tt in range(1, int(min(h, w) // 2) + 1):
            inner = g[int(r0 + tt) : int(r1 - tt) + 1, int(c0 + tt) : int(c1 - tt) + 1]
            if inner.size == 0 or bool((inner == int(border)).any()):
                continue
            values = set(map(int, inner.flatten()))
            if len(values) != 1:
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
            inner_col = int(next(iter(values)))
            break

        if t is None or inner_col is None:
            continue

        frames.append(
            {
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
                "inner_bbox": (int(r0 + t), int(c0 + t), int(r1 - t), int(c1 - t)),
                "inner": int(inner_col),
                "t": int(t),
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
    best_frame = None
    for frame in cands:
        same_row, same_col = _aligned(cur, frame)
        aligned = 1 if (same_row or same_col) else 0
        r1, c1 = _center(cur)
        r2, c2 = _center(frame)
        dist = abs(int(r1) - int(r2)) + abs(int(c1) - int(c2))
        cand = (int(aligned), -int(dist))
        if best is None or cand > best:
            best = cand
            best_frame = frame
    return best_frame


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


def _draw_bridge(out: np.ndarray, fa: dict, fb: dict, color: int) -> None:
    for r, c in _bridge_cells(fa, fb):
        out[int(r), int(c)] = int(color)


def _is_isolated_control_row(g: np.ndarray, r: int, bg: int) -> bool:
    H, W = g.shape
    cols = [int(c) for c in np.where(g[int(r)] != int(bg))[0].tolist()]
    if len(cols) < 2:
        return False
    for c in cols:
        if c - 1 >= 0 and int(g[int(r), int(c - 1)]) != int(bg):
            return False
        if c + 1 < W and int(g[int(r), int(c + 1)]) != int(bg):
            return False
        if r - 1 >= 0 and int(g[int(r - 1), int(c)]) != int(bg):
            return False
        if r + 1 < H and int(g[int(r + 1), int(c)]) != int(bg):
            return False
    return True


def _find_control_row(g: np.ndarray, bg: int) -> int | None:
    for r in range(int(g.shape[0])):
        if _is_isolated_control_row(g, int(r), int(bg)):
            return int(r)
    return None


def _extract_control_sequence(g: np.ndarray, ctrl_r: int, bg: int) -> list[int]:
    return [int(g[int(ctrl_r), int(c)]) for c in range(int(g.shape[1])) if int(g[int(ctrl_r), int(c)]) != int(bg)]


def _choose_start_frame(ctrl: list[int], by: dict[int, list[dict]]) -> dict:
    start_color = int(ctrl[0])
    assert start_color in by
    start_cands = by[start_color]
    if len(start_cands) == 1:
        return start_cands[0]

    nxt_color = int(ctrl[1])
    nxt_cands = by.get(nxt_color, [])
    best = None
    best_frame = None
    for frame in start_cands:
        peers = nxt_cands if nxt_cands else start_cands
        for other in peers:
            same_row, same_col = _aligned(frame, other)
            aligned = 1 if (same_row or same_col) else 0
            r1, c1 = _center(frame)
            r2, c2 = _center(other)
            dist = abs(int(r1) - int(r2)) + abs(int(c1) - int(c2))
            cand = (int(aligned), -int(dist))
            if best is None or cand > best:
                best = cand
                best_frame = frame
    assert best_frame is not None
    return best_frame


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect framed squares following the top control row, then erase that control row.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    cnt = Counter(map(int, g.flatten()))
    border = None
    for color, _count in cnt.most_common():
        if int(color) != int(bg):
            border = int(color)
            break
    assert border is not None

    frames = _find_frames(g, int(border))
    assert frames

    by: dict[int, list[dict]] = defaultdict(list)
    for frame in frames:
        by[int(frame["inner"])].append(frame)

    ctrl_r = _find_control_row(g, int(bg))
    assert ctrl_r is not None
    ctrl = _extract_control_sequence(g, int(ctrl_r), int(bg))
    assert len(ctrl) >= 2

    cur = _choose_start_frame(ctrl, by)
    out = g.copy()

    for i in range(int(len(ctrl)) - 1):
        a = int(ctrl[i])
        b = int(ctrl[i + 1])
        assert int(cur["inner"]) == int(a)
        if b not in by:
            continue
        nxt = _choose_candidate(cur, by[b])
        assert nxt is not None
        _draw_bridge(out, cur, nxt, int(a))
        cur = nxt

    out[int(ctrl_r), :] = int(bg)
    return out


def _most_common_nonbg(g: np.ndarray, bg: int) -> int | None:
    cnt = Counter(map(int, g.flatten()))
    for color, _count in cnt.most_common():
        if int(color) != int(bg):
            return int(color)
    return None


def _draw_frame(g: np.ndarray, *, r0: int, c0: int, outer: int, t: int, border: int, inner_col: int) -> None:
    g[int(r0) : int(r0 + outer), int(c0) : int(c0 + outer)] = int(border)
    g[int(r0 + t) : int(r0 + outer - t), int(c0 + t) : int(c0 + outer - t)] = int(inner_col)


_DIGITS = [int(i) for i in range(10)]


def _gen_attempt() -> np.ndarray | None:
    bg = int(np.random.choice(np.array(_DIGITS, dtype=int)))
    border = int(np.random.choice(np.array([d for d in _DIGITS if int(d) != int(bg)], dtype=int)))
    inner_palette = [int(d) for d in _DIGITS if int(d) not in (int(bg), int(border))]
    if len(inner_palette) < 5:
        return None

    t = int(np.random.choice(np.array([1, 1, 2], dtype=int)))
    inner = int(np.random.randint(1, 4))
    outer = int(inner + 2 * t)
    if not (3 <= int(outer) <= 7):
        return None

    R = int(np.random.randint(2, 6))
    C = int(np.random.randint(2, 6))
    max_slots = int(R * C)
    if max_slots < 5:
        return None

    max_path = int(min(7, max_slots - 1, len(inner_palette) - 1))
    if max_path < 4:
        return None
    n_path = int(np.random.randint(4, max_path + 1))

    path: list[tuple[int, int]] | None = None
    for _ in range(400):
        trial = [(0, 0)]
        seen = {(0, 0)}
        ok = True
        for _step in range(int(n_path) - 1):
            r, c = map(int, trial[-1])
            opts = [
                (int(rr), int(cc))
                for rr, cc in neighbors4(int(r), int(c))
                if in_bounds((int(R), int(C)), int(rr), int(cc)) and (int(rr), int(cc)) not in seen
            ]
            if not opts:
                ok = False
                break
            nxt = opts[int(np.random.randint(0, len(opts)))]
            trial.append((int(nxt[0]), int(nxt[1])))
            seen.add((int(nxt[0]), int(nxt[1])))
        if not ok:
            continue
        if len({r for r, _ in trial}) < 2 or len({c for _, c in trial}) < 2:
            continue
        path = trial
        break
    if path is None:
        return None

    spare_slots = [(int(r), int(c)) for r in range(int(R)) for c in range(int(C)) if (int(r), int(c)) not in set(path)]
    max_decoys = int(min(3, len(spare_slots), len(inner_palette) - n_path))
    if max_decoys < 1:
        return None
    n_decoys = int(np.random.randint(1, max_decoys + 1))
    decoy_idx = np.random.choice(len(spare_slots), size=int(n_decoys), replace=False)
    decoy_slots = [spare_slots[int(i)] for i in np.array(decoy_idx).tolist()]

    row_gap = [int(np.random.randint(1, 6)) for _ in range(int(R) - 1)]
    col_gap = [int(np.random.randint(1, 6)) for _ in range(int(C) - 1)]
    ctrl_top_pad = int(np.random.randint(0, 2))
    ctrl_gap = int(np.random.randint(1, 5))
    left_margin = int(np.random.randint(1, 4))

    ctrl_r = int(ctrl_top_pad)
    first_frame_r0 = int(ctrl_r + 2 + ctrl_gap)
    r0s = [int(first_frame_r0)]
    for k in range(int(R) - 1):
        r0s.append(int(r0s[-1] + outer + row_gap[int(k)]))
    c0s = [int(left_margin)]
    for k in range(int(C) - 1):
        c0s.append(int(c0s[-1] + outer + col_gap[int(k)]))

    slot_to_frame: dict[tuple[int, int], dict] = {}
    frames: list[dict] = []
    for rr, cc in list(path) + list(decoy_slots):
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

    path_colors = [int(x) for x in np.random.choice(np.array(inner_palette, dtype=int), size=int(n_path), replace=False).tolist()]
    remaining_colors = [int(c) for c in inner_palette if int(c) not in set(path_colors)]
    if len(remaining_colors) < len(decoy_slots):
        return None
    decoy_colors = [
        int(x) for x in np.random.choice(np.array(remaining_colors, dtype=int), size=int(len(decoy_slots)), replace=False).tolist()
    ]

    colors_by_slot: dict[tuple[int, int], int] = {}
    for slot, color in zip(path, path_colors):
        colors_by_slot[(int(slot[0]), int(slot[1]))] = int(color)
    for slot, color in zip(decoy_slots, decoy_colors):
        colors_by_slot[(int(slot[0]), int(slot[1]))] = int(color)

    ctrl_cols = [int(1 + 2 * i) for i in range(int(n_path))]
    right_margin = int(np.random.randint(1, 5))
    W = int(max(max_c1 + 1 + right_margin, ctrl_cols[-1] + 2))
    H = int(max_r1 + 2)
    if not (7 <= int(H) <= 30 and 7 <= int(W) <= 30):
        return None

    out = np.full((int(H), int(W)), int(bg), dtype=int)
    for frame in frames:
        rr, cc = map(int, frame["slot"])
        r0, c0, _, _ = map(int, frame["bbox"])
        _draw_frame(
            out,
            r0=int(r0),
            c0=int(c0),
            outer=int(outer),
            t=int(t),
            border=int(border),
            inner_col=int(colors_by_slot[(int(rr), int(cc))]),
        )

    bridge_cells_all: set[tuple[int, int]] = set()
    ok = True
    for a_slot, b_slot in zip(path, path[1:]):
        fa = slot_to_frame[(int(a_slot[0]), int(a_slot[1]))]
        fb = slot_to_frame[(int(b_slot[0]), int(b_slot[1]))]
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
        paint = int(colors_by_slot[(int(a_slot[0]), int(a_slot[1]))])
        for r, c in cells:
            out[int(r), int(c)] = int(paint)
        bridge_cells_all |= set(cells)
    if not ok:
        return None

    inp = out.copy()
    for r, c in bridge_cells_all:
        inp[int(r), int(c)] = int(bg)
    for color, col in zip(path_colors, ctrl_cols):
        inp[int(ctrl_r), int(col)] = int(color)

    if int(get_bg(inp)) != int(bg):
        return None
    if int(_most_common_nonbg(inp, int(bg)) or -1) != int(border):
        return None
    if _find_control_row(inp, int(bg)) != int(ctrl_r):
        return None
    if _find_control_row(out, int(bg)) is not None:
        return None

    try:
        pred = main(inp)
    except Exception:
        return None
    if pred.shape != out.shape or not bool(np.array_equal(pred, out)):
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(500):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("Failed to generate valid input after 500 attempts")
