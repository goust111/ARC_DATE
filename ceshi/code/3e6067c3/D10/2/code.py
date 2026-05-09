# Variant of 3e6067c3
# Dimension: D10 — 对象属性：边框颜色的角色 (meta)
# Change: 所有方块共享同一种边框颜色（最常见的非背景色），边框仅作结构分隔用... → 边框厚度编码优先级
#
# Design analysis for the D10 variant
# a) Concrete behavior:
#    - Keep the original scene grammar: framed squares sit above a control row, and the control colors still define
#      a visitation order whose bridges are painted in the predecessor's inner color.
#    - The changed part is ambiguity resolution. Inputs now contain both 1-pixel and 2-pixel border frames.
#      When a control color matches multiple candidate squares, the solver must still prefer row/column-band aligned
#      matches first, but among those aligned matches it now prefers the thicker border instead of the nearest center.
#    - Generator-wise, each sample includes at least two deliberate "forks": from the current square there is
#      a closer thin same-color decoy in one axis and a farther thick true target on the other axis. Original logic
#      would have gone to the closer decoy; the variant must go to the thick target.
#      Small trace example:
#      current A(red) at slot (0,0), target B(blue,t=2) at slot (0,2), decoy B'(blue,t=1) at slot (1,0).
#      Old score: both aligned, B' is nearer, so B' wins. New score: both aligned, B is thicker, so B wins.
# b) Complexity calibration:
#    - Thickness uses only two visible levels, {1,2}; that is enough to be visually obvious without adding hidden state.
#    - Only a small number of colors are duplicated, so the scene is still legible and the role of thickness can be
#      inferred from 2-3 examples.
# c) Hidden assumptions audited from the original solver:
#    - Border color inference still assumes one shared border color. That remains true.
#    - Frame extraction tested thicknesses from thin to thick. That remains valid because a 2-thick frame still fails
#      the tt=1 inner check (the peeled interior still contains border) and succeeds at tt=2.
#    - Alignment used equality of inner bboxes. With mixed thickness this is no longer automatic if outer boxes are
#      placed on a single lattice, so the generator now places frames on a shared inner-cell lattice instead.
#    - Proximity was previously the only tiebreak after alignment. That assumption is exactly what changes here.
# d) Placement legality:
#    - All frames are built around 1x1 inner cells on a 4x4 inner lattice with fixed 6-cell empty corridors.
#      Since max required clearance for two thick frames is t_a + t_b + 2 <= 6, frames never overlap and every
#      intended bridge has at least one background cell of width.
#    - Decoys are placed on the orthogonal axis from the current frame so they are aligned competitors but do not sit
#      inside the real bridge corridor.
# e) Visual observability:
#    - Thin frames are 3x3, thick frames are 5x5 around the same 1x1 core; humans can see the difference directly.
#    - Because the decoys are intentionally closer than the targets, the output bridge direction visibly reveals that
#      "thicker beats nearer".
# f) Visual neatness:
#    - Only one layer occupies any cell at a time: frames are separated, bridges are validated against occupancy,
#      and control pixels stay on a dedicated bottom row.
# g) Color inference:
#    - Colors are not re-mapped; they keep the original "bridge color = predecessor inner color" rule.
#      Thickness, not color id, drives the new disambiguation.
# h) Generator non-degeneracy check:
#    - Every generated puzzle is rejected unless the new route matches the intended path and the old nearest-only
#      route diverges on at least two control steps.
# i) New solver assumptions introduced here:
#    - Mixed-thickness inputs must still share the same inner-band lattice so aligned frames have identical inner bboxes.
#    - For any duplicated control color, the generator guarantees at least one aligned candidate and that the intended
#      candidate is uniquely thickest among the aligned competitors for that step.

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
import sys

import numpy as np

for _parent in Path(__file__).resolve().parents:
    _arc_utils = _parent / "solvers" / "arc_utils.py"
    if _arc_utils.exists():
        sys.path.insert(0, str(_parent / "solvers"))
        break

from arc_utils import get_bg, label_mask


def _most_common_nonbg(g: np.ndarray, bg: int) -> int | None:
    cnt = Counter(map(int, g.flatten()))
    for col, _k in cnt.most_common():
        if int(col) != int(bg):
            return int(col)
    return None


def _draw_frame(g: np.ndarray, *, inner_r: int, inner_c: int, t: int, border: int, inner_col: int) -> None:
    r0 = int(inner_r - t)
    c0 = int(inner_c - t)
    outer = int(1 + 2 * t)
    g[int(r0) : int(r0 + outer), int(c0) : int(c0 + outer)] = int(border)
    g[int(inner_r), int(inner_c)] = int(inner_col)


def _center(f: dict) -> tuple[int, int]:
    r0, c0, r1, c1 = map(int, f["bbox"])
    return int((r0 + r1) // 2), int((c0 + c1) // 2)


def _aligned(f1: dict, f2: dict) -> tuple[bool, bool]:
    a = f1["inner_bbox"]
    b = f2["inner_bbox"]
    same_row = bool(int(a[0]) == int(b[0]) and int(a[2]) == int(b[2]))
    same_col = bool(int(a[1]) == int(b[1]) and int(a[3]) == int(b[3]))
    return same_row, same_col


def _transition_score(fa: dict, fb: dict, *, prefer_thickness: bool) -> tuple[int, int, int]:
    sr, sc = _aligned(fa, fb)
    al = 1 if (sr or sc) else 0
    (r1_, c1_) = _center(fa)
    (r2_, c2_) = _center(fb)
    d = abs(int(r1_) - int(r2_)) + abs(int(c1_) - int(c2_))
    thick = int(fb["t"]) if prefer_thickness else 0
    return int(al), int(thick), -int(d)


def _pick_best_target(cur: dict, cands: list[dict], *, prefer_thickness: bool) -> dict:
    best = None
    bestf = None
    for f in cands:
        cand = _transition_score(cur, f, prefer_thickness=prefer_thickness)
        if best is None or cand > best:
            best = cand
            bestf = f
    assert bestf is not None
    return bestf


def _pick_start_frame(start_cands: list[dict], next_cands: list[dict], *, prefer_thickness: bool) -> dict:
    if len(start_cands) == 1:
        return start_cands[0]
    best = None
    bestf = None
    for f in start_cands:
        if next_cands:
            g2 = _pick_best_target(f, next_cands, prefer_thickness=prefer_thickness)
            cand = _transition_score(f, g2, prefer_thickness=prefer_thickness) + (int(f["t"]),)
        else:
            (rr, cc) = _center(f)
            cand = (0, int(f["t"]) if prefer_thickness else 0, -int(rr + cc), int(f["t"]))
        if best is None or cand > best:
            best = cand
            bestf = f
    assert bestf is not None
    return bestf


def _extract_frames(g: np.ndarray) -> tuple[int, int, list[dict]]:
    bg = int(get_bg(g))
    border = _most_common_nonbg(g, int(bg))
    assert border is not None

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
                "id": int(len(frames)),
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
                "t": int(t),
                "inner": int(inner_col),
                "inner_bbox": (int(r0 + t), int(c0 + t), int(r1 - t), int(c1 - t)),
            }
        )

    assert frames
    return int(bg), int(border), frames


def _find_control_sequence(g: np.ndarray, bg: int, frames: list[dict]) -> list[int]:
    H, W = g.shape
    max_frame_r = max(int(f["bbox"][2]) for f in frames)
    ctrl_r = None
    for r in range(int(H - 1), int(max_frame_r), -1):
        if int((g[int(r)] != int(bg)).sum()) >= 2:
            ctrl_r = int(r)
            break
    assert ctrl_r is not None
    ctrl = [int(g[int(ctrl_r), int(c)]) for c in range(int(W)) if int(g[int(ctrl_r), int(c)]) != int(bg)]
    assert len(ctrl) >= 2
    return ctrl


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


def _draw_bridge(out: np.ndarray, fa: dict, fb: dict, color: int) -> None:
    H, W = out.shape
    for r, c in _bridge_cells(fa, fb):
        if 0 <= int(r) < int(H) and 0 <= int(c) < int(W):
            out[int(r), int(c)] = int(color)


def _trace_route(frames: list[dict], ctrl: list[int], *, prefer_thickness: bool) -> list[int]:
    by: dict[int, list[dict]] = defaultdict(list)
    for f in frames:
        by[int(f["inner"])].append(f)

    start_color = int(ctrl[0])
    assert start_color in by
    next_cands = by.get(int(ctrl[1]), []) if len(ctrl) >= 2 else []
    cur = _pick_start_frame(by[start_color], next_cands, prefer_thickness=prefer_thickness)
    route = [int(cur["id"])]
    for b in ctrl[1:]:
        assert int(b) in by
        cur = _pick_best_target(cur, by[int(b)], prefer_thickness=prefer_thickness)
        route.append(int(cur["id"]))
    return route


def _count_local_thickness_forks(actual_frames: list[dict], frames: list[dict], ctrl: list[int]) -> int:
    by: dict[int, list[dict]] = defaultdict(list)
    for f in frames:
        by[int(f["inner"])].append(f)

    forks = 0
    for cur, nxt_color, intended_next in zip(actual_frames, ctrl[1:], actual_frames[1:]):
        cands = by[int(nxt_color)]
        best_new = _pick_best_target(cur, cands, prefer_thickness=True)
        best_old = _pick_best_target(cur, cands, prefer_thickness=False)
        if int(best_new["id"]) == int(intended_next["id"]) and int(best_old["id"]) != int(intended_next["id"]):
            forks += 1
    return int(forks)


def main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    bg, _border, frames = _extract_frames(g)
    ctrl = _find_control_sequence(g, int(bg), frames)

    by: dict[int, list[dict]] = defaultdict(list)
    for f in frames:
        by[int(f["inner"])].append(f)

    start_color = int(ctrl[0])
    assert start_color in by
    next_cands = by.get(int(ctrl[1]), []) if len(ctrl) >= 2 else []
    cur = _pick_start_frame(by[start_color], next_cands, prefer_thickness=True)

    out = g.copy()
    for i in range(int(len(ctrl)) - 1):
        a = int(ctrl[i])
        b = int(ctrl[i + 1])
        assert int(cur["inner"]) == int(a)
        if b not in by:
            continue
        nxt = _pick_best_target(cur, by[b], prefer_thickness=True)
        _draw_bridge(out, cur, nxt, int(a))
        cur = nxt
    return out


_DIGITS = [int(i) for i in range(10)]
_BASE_PATH = [(0, 0), (0, 2), (2, 2), (2, 3), (1, 3)]
_BASE_DECOYS = {1: (1, 0), 2: (0, 3)}


def _apply_transform(pt: tuple[int, int], size: int, mode: int) -> tuple[int, int]:
    r, c = map(int, pt)
    last = int(size - 1)
    if int(mode) == 0:
        return int(r), int(c)
    if int(mode) == 1:
        return int(r), int(last - c)
    if int(mode) == 2:
        return int(last - r), int(c)
    if int(mode) == 3:
        return int(last - r), int(last - c)
    if int(mode) == 4:
        return int(c), int(r)
    if int(mode) == 5:
        return int(c), int(last - r)
    if int(mode) == 6:
        return int(last - c), int(r)
    return int(last - c), int(last - r)


def _frame_from_slot(
    *,
    slot: tuple[int, int],
    t: int,
    inner_col: int,
    inner_r0s: list[int],
    inner_c0s: list[int],
    frame_id: int,
) -> dict:
    rr, cc = map(int, slot)
    inner_r = int(inner_r0s[int(rr)])
    inner_c = int(inner_c0s[int(cc)])
    return {
        "id": int(frame_id),
        "bbox": (int(inner_r - t), int(inner_c - t), int(inner_r + t), int(inner_c + t)),
        "t": int(t),
        "inner": int(inner_col),
        "inner_bbox": (int(inner_r), int(inner_c), int(inner_r), int(inner_c)),
        "slot": (int(rr), int(cc)),
    }


def _gen_attempt() -> np.ndarray | None:
    bg = int(np.random.choice(np.array(_DIGITS, dtype=int)))
    border_choices = [d for d in _DIGITS if int(d) != int(bg)]
    border = int(np.random.choice(np.array(border_choices, dtype=int)))
    inner_palette = [int(d) for d in _DIGITS if int(d) not in (int(bg), int(border))]
    if len(inner_palette) < len(_BASE_PATH):
        return None

    mode = int(np.random.randint(0, 8))
    grid_size = 4
    path_slots = [_apply_transform(pt, grid_size, mode) for pt in _BASE_PATH]
    decoy_slots = {int(k): _apply_transform(v, grid_size, mode) for k, v in _BASE_DECOYS.items()}

    all_slots = list(path_slots) + list(decoy_slots.values())
    if len(set(all_slots)) != len(all_slots):
        return None

    colors = [int(x) for x in np.random.choice(np.array(inner_palette, dtype=int), size=len(_BASE_PATH), replace=False)]
    actual_ts = [int(np.random.choice(np.array([1, 2], dtype=int))) for _ in path_slots]
    for idx in decoy_slots:
        actual_ts[int(idx)] = 2

    inner_r0s = [2, 9, 16, 23]
    inner_c0s = [2, 9, 16, 23]

    actual_frames: list[dict] = []
    frames: list[dict] = []
    for idx, slot in enumerate(path_slots):
        frame = _frame_from_slot(
            slot=slot,
            t=int(actual_ts[int(idx)]),
            inner_col=int(colors[int(idx)]),
            inner_r0s=inner_r0s,
            inner_c0s=inner_c0s,
            frame_id=len(frames),
        )
        actual_frames.append(frame)
        frames.append(frame)

    for idx, slot in decoy_slots.items():
        frame = _frame_from_slot(
            slot=slot,
            t=1,
            inner_col=int(colors[int(idx)]),
            inner_r0s=inner_r0s,
            inner_c0s=inner_c0s,
            frame_id=len(frames),
        )
        frames.append(frame)

    max_r1 = max(int(f["bbox"][2]) for f in frames)
    max_c1 = max(int(f["bbox"][3]) for f in frames)
    ctrl_r = int(max_r1 + 2)
    H = int(ctrl_r + 2)
    W = int(max(max_c1 + 3, 2 * len(actual_frames) + 2))
    if not (7 <= int(H) <= 30 and 7 <= int(W) <= 30):
        return None

    out = np.full((int(H), int(W)), int(bg), dtype=int)
    for f in frames:
        r0, c0, r1, c1 = map(int, f["bbox"])
        if r0 < 0 or c0 < 0 or r1 >= H or c1 >= W:
            return None
        if bool((out[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1] != int(bg)).any()):
            return None
        inner_r, inner_c = map(int, f["inner_bbox"][:2])
        _draw_frame(out, inner_r=int(inner_r), inner_c=int(inner_c), t=int(f["t"]), border=int(border), inner_col=int(f["inner"]))

    bridge_cells_all: set[tuple[int, int]] = set()
    for fa, fb in zip(actual_frames, actual_frames[1:]):
        cells = _bridge_cells(fa, fb)
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
            out[int(r), int(c)] = int(fa["inner"])
        bridge_cells_all |= set(cells)

    ctrl_colors = [int(f["inner"]) for f in actual_frames]
    ctrl_cols = [int(1 + 2 * i) for i in range(int(len(ctrl_colors)))]
    if int(ctrl_cols[-1]) >= int(W):
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

    intended_route = [int(f["id"]) for f in actual_frames]
    if _trace_route(frames, ctrl_colors, prefer_thickness=True) != intended_route:
        return None
    if _count_local_thickness_forks(actual_frames, frames, ctrl_colors) < 2:
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
