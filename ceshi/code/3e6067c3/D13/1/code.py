# Variant of 3e6067c3
# Dimension: D13 — 桥梁与已有结构的交互 (meta)
# Change: 桥梁仅在背景色区域绘制，不覆盖任何已有的方块或先前桥梁... → 交叉点使用混合色
#
# Design analysis
# ---------------
# yaml:
#   source_task:
#     id: 409aa875
#     solver_baseline:
#       - The reference solver first aggregates projectile landings into a
#         destination -> shooters map, then resolves multi-hit destinations with
#         a dedicated collision color.
#       - That order-independent "collect first, resolve later" pattern is the
#         correct baseline for this variant because the original bridge painter
#         wrote cells eagerly and implicitly assumed bridges never overlap.
#   concretization_and_trace:
#     scene:
#       - Input still contains equal-sized framed squares plus a control row below.
#       - Output still draws axis-aligned rectangular bridges between consecutive
#         control colors, using the current color for each bridge.
#       - New behavior only appears where two differently colored bridge
#         rectangles overlap on background cells: those overlap pixels become a
#         fixed special mix color 1 that is absent from the input.
#     small_trace:
#       - Example path template:
#           A(row1,col0) -> B(row1,col3): draw a long horizontal bridge in color a
#           ...
#           D(row0,col1) -> E(row3,col1): later draw a long vertical bridge in color d
#         The two rectangles overlap in the middle of the canvas.
#       - Original seed logic:
#           `out[r, c] = color` inside `draw_bridge()` assumed overlap never
#           happens; the generator separately rejected any reused bridge cell.
#       - Variant logic:
#           collect all bridge hits against the input background first; after all
#           segments are known, cells hit by one bridge keep that bridge color,
#           and cells hit by >=2 distinct bridge colors become mix color 1.
#   complexity_calibration:
#     - The mix color is fixed and global rather than pair-dependent. Humans only
#       need to infer one new rule: "bridge crossing => color 1".
#     - Generator always produces at least one long horizontal bridge and one long
#       vertical bridge that cross away from frame endpoints, so the new behavior
#       is visually obvious rather than a subtle single-pixel edge case.
#     - The rest of the task remains the original framed-square / control-sequence
#       puzzle. No unrelated object type or sorting rule is introduced.
#   implicit_assumption_audit_of_original_main:
#     - Frame detection and control-row extraction remain valid; D13 changes only
#       bridge interaction, not object parsing.
#     - The original `draw_bridge()` line-by-line unconditional writes relied on
#       the old generator guarantee that bridge cells are unique. That premise is
#       false here, so eager painting must be replaced by bridge-hit aggregation.
#     - The original loop over control pairs is still correct, but bridge
#       resolution must become order-independent or the overlap color would depend
#       on draw order instead of geometry.
#   placement_legality:
#     - Generator uses a fixed 5-frame lattice template whose control sequence
#       creates four aligned bridges:
#           long horizontal, short vertical, short horizontal, long vertical.
#       Only the first and last bridges intersect, and they intersect strictly on
#       background cells, never on frames or the control row.
#     - All frames share one border color and have unique inner colors, so the
#       solver's color lookup is unambiguous.
#     - Final acceptance validates:
#         1. main(input) == constructed_output
#         2. at least one mix-colored output cell exists
#         3. naive "skip occupied bridge cells" rendering differs from variant
#   visual_observability:
#     - Every generated pair visibly introduces color 1 only at bridge crossings.
#       The training examples therefore expose the D13 rule directly.
#   visual_cleanliness:
#     - Exactly one crossing region is intentionally created; accidental extra
#       overlaps are avoided by the template geometry.
#     - Mix color 1 never appears in the input, so it reads as a new output-only
#       marker instead of another ordinary square color.
#   color_inference:
#     - Only one new color behavior exists: overlap of differently colored
#       bridges produces 1. This is supported by repeated structure across all
#       samples and avoids underconstrained per-pair color mixing.
#   new_solver_assumptions:
#     - Collision color 1 is reserved and absent from input frames/background.
#     - Consecutive control colors correspond to aligned frames; generator
#       enforces this with the fixed path template.
#     - Bridge rectangles overlap only on cells that are background in the input.
#       Generator checks this before accepting a sample.

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[5]
for _p in (ROOT / "solvers", ROOT / "arc2_solver_select"):
    if str(_p) not in sys.path:
        sys.path.append(str(_p))

from arc_utils import get_bg, label_mask


MIX_COLOR = 1
_DIGITS = [int(i) for i in range(10)]
_BASE_PATH = [(1, 0), (1, 3), (0, 3), (0, 1), (3, 1)]
_LATTICE_N = 4


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
            if inner.size == 0 or bool((inner == int(border)).any()):
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

            t = int(tt)
            inner_col = int(next(iter(uniq)))
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


def _choose_next(cur: dict, target_color: int, by: dict[int, list[dict]]) -> dict | None:
    if int(target_color) not in by:
        return None
    best = None
    bestf = None
    for f in by[int(target_color)]:
        sr, sc = _aligned(cur, f)
        aligned_score = 1 if (sr or sc) else 0
        r1, c1 = _center(cur)
        r2, c2 = _center(f)
        dist = abs(int(r1) - int(r2)) + abs(int(c1) - int(c2))
        cand = (int(aligned_score), -int(dist))
        if best is None or cand > best:
            best = cand
            bestf = f
    return bestf


def _render_with_bridge_hits(base: np.ndarray, bg: int, bridge_hits: dict[tuple[int, int], set[int]]) -> np.ndarray:
    out = np.array(base, dtype=int, copy=True)
    for (r, c), colors in bridge_hits.items():
        if not colors:
            continue
        out[int(r), int(c)] = int(MIX_COLOR if len(colors) >= 2 else next(iter(colors)))
    return out


def _render_skip_overlap(base: np.ndarray, bg: int, path_frames: list[tuple[dict, dict, int]]) -> np.ndarray:
    out = np.array(base, dtype=int, copy=True)
    for fa, fb, color in path_frames:
        for r, c in _bridge_cells(fa, fb):
            if int(out[int(r), int(c)]) == int(bg):
                out[int(r), int(c)] = int(color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect framed squares via colored bridges following a control-pixel sequence.

    Variant rule:
    - Bridge geometry stays the same as the seed task.
    - If differently colored bridge rectangles cross on background cells, the
      crossing pixels become special mix color 1.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))
    border = _most_common_nonbg(g, int(bg))
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
    assert int(start_color) in by
    start_cands = by[int(start_color)]
    if len(start_cands) == 1:
        cur = start_cands[0]
    else:
        nxt_color = int(ctrl[1])
        best = None
        bestf = None
        for f in start_cands:
            g2 = _choose_next(f, int(nxt_color), by)
            if g2 is None:
                continue
            sr, sc = _aligned(f, g2)
            aligned_score = 1 if (sr or sc) else 0
            r1, c1 = _center(f)
            r2, c2 = _center(g2)
            dist = abs(int(r1) - int(r2)) + abs(int(c1) - int(c2))
            cand = (int(aligned_score), -int(dist))
            if best is None or cand > best:
                best = cand
                bestf = f
        assert not (bestf is None)
        cur = bestf

    bridge_hits: dict[tuple[int, int], set[int]] = defaultdict(set)
    for i in range(int(len(ctrl)) - 1):
        a = int(ctrl[i])
        b = int(ctrl[i + 1])
        assert int(cur["inner"]) == int(a)
        nxt = _choose_next(cur, int(b), by)
        if nxt is None:
            continue
        for r, c in _bridge_cells(cur, nxt):
            if 0 <= int(r) < int(H) and 0 <= int(c) < int(W) and int(g[int(r), int(c)]) == int(bg):
                bridge_hits[(int(r), int(c))].add(int(a))
        cur = nxt

    return _render_with_bridge_hits(g, int(bg), bridge_hits)


def _rotate_slot(slot: tuple[int, int], k: int) -> tuple[int, int]:
    r, c = map(int, slot)
    n = int(_LATTICE_N - 1)
    for _ in range(int(k) % 4):
        r, c = int(c), int(n - r)
    return int(r), int(c)


def _transform_path(path: list[tuple[int, int]]) -> list[tuple[int, int]]:
    flip = bool(np.random.randint(0, 2))
    rot = int(np.random.randint(0, 4))
    transformed = []
    for r, c in path:
        if flip:
            c = int(_LATTICE_N - 1 - c)
        transformed.append(_rotate_slot((int(r), int(c)), int(rot)))
    return transformed


def _build_template_frames(outer: int, t: int) -> tuple[list[dict], list[int], list[int]]:
    row_gap = [int(np.random.randint(1, 3)) for _ in range(int(_LATTICE_N) - 1)]
    col_gap = [int(np.random.randint(1, 3)) for _ in range(int(_LATTICE_N) - 1)]
    top_margin = int(np.random.randint(1, 3))
    left_margin = int(np.random.randint(1, 3))

    r0s = [int(top_margin)]
    for gap in row_gap:
        r0s.append(int(r0s[-1] + outer + int(gap)))
    c0s = [int(left_margin)]
    for gap in col_gap:
        c0s.append(int(c0s[-1] + outer + int(gap)))

    path = _transform_path(list(_BASE_PATH))
    frames: list[dict] = []
    for rr, cc in path:
        r0 = int(r0s[int(rr)])
        c0 = int(c0s[int(cc)])
        r1 = int(r0 + outer - 1)
        c1 = int(c0 + outer - 1)
        frames.append(
            {
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
                "inner_bbox": (int(r0 + t), int(c0 + t), int(r1 - t), int(c1 - t)),
                "slot": (int(rr), int(cc)),
            }
        )
    return frames, r0s, c0s


def _gen_attempt() -> np.ndarray | None:
    usable = [int(d) for d in _DIGITS if int(d) != int(MIX_COLOR)]
    bg = int(np.random.choice(np.array(usable, dtype=int)))
    border_choices = [int(d) for d in usable if int(d) != int(bg)]
    border = int(np.random.choice(np.array(border_choices, dtype=int)))
    inner_palette = [int(d) for d in usable if int(d) not in (int(bg), int(border))]
    if len(inner_palette) < len(_BASE_PATH):
        return None

    t = 1
    inner = int(np.random.randint(2, 4))
    outer = int(inner + 2 * t)

    frames, _r0s, _c0s = _build_template_frames(int(outer), int(t))
    max_r1 = max(int(f["bbox"][2]) for f in frames)
    max_c1 = max(int(f["bbox"][3]) for f in frames)
    ctrl_gap = int(np.random.randint(2, 4))
    ctrl_r = int(max_r1 + ctrl_gap)
    bottom_margin = 1
    right_margin = int(np.random.randint(1, 4))
    H = int(ctrl_r + 1 + bottom_margin)
    W = int(max_c1 + 1 + right_margin)
    if not (7 <= int(H) <= 30 and 7 <= int(W) <= 30):
        return None

    ctrl_colors = [int(x) for x in np.random.choice(np.array(inner_palette, dtype=int), size=len(frames), replace=False)]
    base = np.full((int(H), int(W)), int(bg), dtype=int)
    for frame, inner_col in zip(frames, ctrl_colors):
        r0, c0, _r1, _c1 = map(int, frame["bbox"])
        _draw_frame(base, r0=int(r0), c0=int(c0), outer=int(outer), t=int(t), border=int(border), inner_col=int(inner_col))
        frame["inner"] = int(inner_col)
        frame["t"] = int(t)

    ctrl_cols = [int(1 + 2 * i) for i in range(int(len(ctrl_colors)))]
    if int(ctrl_cols[-1]) >= int(W):
        return None
    for col, cc in zip(ctrl_colors, ctrl_cols):
        base[int(ctrl_r), int(cc)] = int(col)

    bridge_hits: dict[tuple[int, int], set[int]] = defaultdict(set)
    bridge_path: list[tuple[dict, dict, int]] = []
    for fa, fb, color in zip(frames, frames[1:], ctrl_colors[:-1]):
        cells = _bridge_cells(fa, fb)
        if not cells:
            return None
        for r, c in cells:
            if not (0 <= int(r) < int(H) and 0 <= int(c) < int(W)):
                return None
            if int(base[int(r), int(c)]) != int(bg):
                return None
            bridge_hits[(int(r), int(c))].add(int(color))
        bridge_path.append((fa, fb, int(color)))

    out = _render_with_bridge_hits(base, int(bg), bridge_hits)
    naive = _render_skip_overlap(base, int(bg), bridge_path)
    mix_count = int((out == int(MIX_COLOR)).sum())
    if mix_count <= 0:
        return None
    if bool(np.array_equal(out, naive)):
        return None

    if int(get_bg(base)) != int(bg):
        return None
    if int(_most_common_nonbg(base, int(bg)) or -1) != int(border):
        return None

    detected = _detect_frames(base, int(border))
    if len(detected) != len(frames):
        return None

    try:
        pred = main(base)
    except Exception:
        return None
    if pred.shape != out.shape or not bool(np.array_equal(pred, out)):
        return None
    return base


def generate_input() -> np.ndarray:
    for _ in range(500):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("Failed to generate valid input after 500 attempts")
