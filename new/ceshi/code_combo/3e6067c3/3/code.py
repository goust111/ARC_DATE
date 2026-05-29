# Combo variant of 3e6067c3 (index 3)
# combo_spec: D1/2+D4/0
# Name: 尺寸匹配+不等大方块

from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np

from arc_utils import get_bg, label_mask

"""
Design analysis for the combo variant
=====================================

a) Concrete behavior and trace
The original task used equal-sized framed squares with different inner colors and
single control pixels. The solver matched control color -> square inner color,
then drew a bridge with the shared inner-band thickness. This combo changes both
of those points at once:
- all squares now share one inner fill color, so color is no longer usable for
  target selection;
- squares have unequal outer sizes and unequal border thicknesses, so alignment
  must be recognized by shared row/column centerlines rather than identical
  inner bboxes, and bridge width must follow the smaller endpoint.

Concrete example:
- Suppose one square has a 1x1 opening and another has a 3x3 opening. Their
  control bars are length 1 and length 9, because matching is by visible inner
  area, not by color and not by side length.
- When a length-9 bar is followed by a length-1 bar, the solver must choose the
  1x1-opening square and draw the bridge using the bar color, but only through
  a 1-cell-thick corridor because the smaller endpoint is 1 cell wide.
- If two 1x1-opening squares exist with different border thicknesses, the same
  length-1 control bar still refers to both of them as one size class; the next
  step is resolved by alignment + nearest distance, not by outer size.

b) Complexity calibration
The scene uses a small discrete set of inner areas: 1 and 9, coming from 1x1
and 3x3 inner squares. This keeps the rule readable while making "area match"
visibly different from side-length match because the long control bar is length
9 rather than 3. Repetition is preserved: both size classes appear multiple
times along the path, and at least one size class is realized with multiple
outer sizes via different border thicknesses, so the fusion is not reducible to
matching on total frame size.

c) Hidden-assumption audit of the original main()
- Grouping by `inner` color is invalid because every frame has the same inner
  fill. We must group by measured inner area instead.
- Reading control as isolated pixels is invalid because the control items are
  monochrome runs whose lengths carry the meaning.
- The old equal-band alignment test fails once frames have unequal sizes. The
  combo solver aligns frames by shared doubled center coordinates and requires a
  non-empty bridge rectangle.
- The old bridge painter copied the source inner band width. That overpaints for
  smaller targets. The combo uses the smaller endpoint's inner bbox as the
  bridge cross-section.
- The old `assert cur["inner"] == a` no longer makes sense; it becomes an area
  check against the current control run length.

d) Placement legality
The generator places path squares on a coarse lattice of shared row/column
centers. All outer sizes are odd (3, 5, or 7), so row/column alignment by exact
centerline is realizable for every frame in the same lattice row/column. The
coarse spacing guarantees no overlap, at least one background cell between
frames, and non-empty bridge corridors.

e) Visual observability
Across a few examples, a human can see:
- all inner fills have the same color, so color cannot be the matching key;
- control items are 1-cell or 9-cell bars, strongly suggesting counting;
- squares with the same inner area can still have different outer footprints,
  so the relevant measured quantity is the opening area, not the overall frame.
This makes the fused rule recoverable from examples rather than hidden in code.

f) Visual cleanliness
Frames use one border color and one common inner color, bridges occupy only the
gaps between lattice neighbors, and control bars sit on a dedicated bottom row
with background gaps. That keeps the three visual layers separate.

g) Color inferability
No hidden color remapping is introduced. Border color marks the frame outline,
one common inner color marks every opening, and each control bar color is reused
directly as the paint color of the bridge for that step. Matching itself is
purely geometric: bar length equals inner area.

New solver assumptions and generator guarantees
----------------------------------------------
- Assumption: every control item is a solid horizontal run separated by at least
  one background cell.
  Guarantee: the generator builds exactly that control row.
- Assumption: aligned frames share the same doubled row center or doubled column
  center, and bridge corridors are non-empty.
  Guarantee: all frames are placed on a coarse odd-sized lattice.
- Assumption: the intended next frame is the unique best candidate under the
  solver's align-first / nearest heuristic.
  Guarantee: the generator keeps only layouts whose traced path matches the
  intended path exactly.
- Assumption: inner area, not outer size, is the relevant control feature.
  Guarantee: at least one repeated area class appears with multiple outer sizes,
  and the generator rejects samples that do not show this.
"""


_DIGITS = [int(i) for i in range(10)]
_INNER_SIZES = (1, 3)
_THICKNESSES = (1, 2)
_PATH_TEMPLATES = [
    [(0, 0), (0, 1), (0, 2), (1, 2)],
    [(0, 0), (0, 1), (1, 1), (1, 2)],
    [(0, 0), (0, 1), (1, 1), (1, 0)],
]
_SIZE_SEQUENCES = [
    (1, 1, 3, 3),
    (1, 3, 3, 1),
    (3, 1, 1, 3),
    (3, 3, 1, 1),
]
_T_SEQUENCE = (1, 1, 1, 2)


def _parse_control_runs(row: np.ndarray, bg: int) -> list[dict]:
    runs: list[dict] = []
    c = 0
    W = int(row.shape[0])
    while c < W:
        if int(row[int(c)]) == int(bg):
            c += 1
            continue
        color = int(row[int(c)])
        c0 = int(c)
        while c < W and int(row[int(c)]) == int(color):
            c += 1
        runs.append({"color": int(color), "length": int(c - c0)})
    return runs


def _most_common_nonbg(g: np.ndarray, bg: int) -> int | None:
    cnt = Counter(map(int, g.flatten()))
    for col, _count in cnt.most_common():
        if int(col) != int(bg):
            return int(col)
    return None


def _detect_frames(g: np.ndarray) -> tuple[int, int, list[dict]]:
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
        if h < 3 or w < 3 or h != w:
            continue

        found = None
        for t in range(1, int(min(h, w) // 2) + 1):
            inner = g[int(r0 + t) : int(r1 - t) + 1, int(c0 + t) : int(c1 - t) + 1]
            if inner.size == 0:
                continue
            if bool((inner == int(border)).any()):
                continue
            vals = set(map(int, inner.flatten()))
            if len(vals) != 1:
                continue

            ok = True
            if not bool((g[int(r0) : int(r0 + t), int(c0) : int(c1) + 1] == int(border)).all()):
                ok = False
            if not bool((g[int(r1 - t) + 1 : int(r1) + 1, int(c0) : int(c1) + 1] == int(border)).all()):
                ok = False
            if not bool((g[int(r0) : int(r1) + 1, int(c0) : int(c0 + t)] == int(border)).all()):
                ok = False
            if not bool((g[int(r0) : int(r1) + 1, int(c1 - t) + 1 : int(c1) + 1] == int(border)).all()):
                ok = False
            if not ok:
                continue

            inner_bbox = (int(r0 + t), int(c0 + t), int(r1 - t), int(c1 - t))
            inner_h = int(inner_bbox[2] - inner_bbox[0] + 1)
            inner_w = int(inner_bbox[3] - inner_bbox[1] + 1)
            if inner_h != inner_w:
                continue

            found = {
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
                "t": int(t),
                "outer": int(h),
                "inner": int(next(iter(vals))),
                "inner_bbox": inner_bbox,
                "inner_size": int(inner_h),
                "area": int(inner_h * inner_h),
                "center2": (int(r0 + r1), int(c0 + c1)),
            }
            break

        if found is not None:
            frames.append(found)

    assert frames
    return int(bg), int(border), frames


def _frames_by_area(frames: list[dict]) -> dict[int, list[dict]]:
    by: dict[int, list[dict]] = defaultdict(list)
    for f in frames:
        by[int(f["area"])].append(f)
    return by


def _alignment_kind(f1: dict, f2: dict) -> str | None:
    r1, c1 = map(int, f1["center2"])
    r2, c2 = map(int, f2["center2"])
    if int(r1) == int(r2) and int(c1) != int(c2):
        return "row"
    if int(c1) == int(c2) and int(r1) != int(r2):
        return "col"
    return None


def _bridge_rect(fa: dict, fb: dict) -> tuple[int, int, int, int] | None:
    kind = _alignment_kind(fa, fb)
    if kind is None:
        return None

    (r0a, c0a, r1a, c1a) = map(int, fa["bbox"])
    (r0b, c0b, r1b, c1b) = map(int, fb["bbox"])
    narrow = fa if int(fa["inner_size"]) <= int(fb["inner_size"]) else fb
    ib = narrow["inner_bbox"]

    if kind == "row":
        rr0, rr1 = int(ib[0]), int(ib[2])
        if int(c1a) < int(c0b):
            cc0, cc1 = int(c1a + 1), int(c0b - 1)
        else:
            cc0, cc1 = int(c1b + 1), int(c0a - 1)
        if int(rr0) > int(rr1) or int(cc0) > int(cc1):
            return None
        return int(rr0), int(rr1), int(cc0), int(cc1)

    cc0, cc1 = int(ib[1]), int(ib[3])
    if int(r1a) < int(r0b):
        rr0, rr1 = int(r1a + 1), int(r0b - 1)
    else:
        rr0, rr1 = int(r1b + 1), int(r0a - 1)
    if int(rr0) > int(rr1) or int(cc0) > int(cc1):
        return None
    return int(rr0), int(rr1), int(cc0), int(cc1)


def _bridge_cells(fa: dict, fb: dict) -> set[tuple[int, int]]:
    rect = _bridge_rect(fa, fb)
    if rect is None:
        return set()
    rr0, rr1, cc0, cc1 = map(int, rect)
    return {(int(r), int(c)) for r in range(int(rr0), int(rr1) + 1) for c in range(int(cc0), int(cc1) + 1)}


def _candidate_score(f1: dict, f2: dict) -> tuple[int, int]:
    aligned = 1 if _bridge_rect(f1, f2) is not None else 0
    r1, c1 = map(int, f1["center2"])
    r2, c2 = map(int, f2["center2"])
    d = abs(int(r1) - int(r2)) + abs(int(c1) - int(c2))
    return int(aligned), -int(d)


def _trace_path_frames(frames: list[dict], ctrl: list[dict]) -> list[dict]:
    by_area = _frames_by_area(frames)
    start_area = int(ctrl[0]["length"])
    assert start_area in by_area

    start_cands = by_area[int(start_area)]
    if len(start_cands) == 1:
        cur = start_cands[0]
    else:
        nxt_area = int(ctrl[1]["length"])
        nxt_cands = by_area.get(int(nxt_area), [])
        best = None
        bestf = None
        for f in start_cands:
            for g2 in (nxt_cands if nxt_cands else [f]):
                cand = _candidate_score(f, g2)
                if best is None or cand > best:
                    best = cand
                    bestf = f
        assert bestf is not None
        cur = bestf

    traced = [cur]
    for i in range(int(len(ctrl)) - 1):
        assert int(cur["area"]) == int(ctrl[int(i)]["length"])
        target_area = int(ctrl[int(i + 1)]["length"])
        cands = by_area.get(int(target_area), [])
        assert cands
        best = None
        bestf = None
        for candf in cands:
            cand = _candidate_score(cur, candf)
            if best is None or cand > best:
                best = cand
                bestf = candf
        assert bestf is not None
        traced.append(bestf)
        cur = bestf
    return traced


def _draw_bridge(out: np.ndarray, fa: dict, fb: dict, color: int) -> None:
    rect = _bridge_rect(fa, fb)
    if rect is None:
        return
    rr0, rr1, cc0, cc1 = map(int, rect)
    H, W = out.shape
    for r in range(int(rr0), int(rr1) + 1):
        for c in range(int(cc0), int(cc1) + 1):
            if 0 <= int(r) < H and 0 <= int(c) < W:
                out[int(r), int(c)] = int(color)


def _draw_frame(g: np.ndarray, *, r0: int, c0: int, outer: int, t: int, border: int, inner_col: int) -> None:
    g[int(r0) : int(r0 + outer), int(c0) : int(c0 + outer)] = int(border)
    g[int(r0 + t) : int(r0 + outer - t), int(c0 + t) : int(c0 + outer - t)] = int(inner_col)


def main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg, _border, frames = _detect_frames(g)

    max_frame_r = max(int(f["bbox"][2]) for f in frames)
    ctrl_r = None
    for r in range(int(H - 1), int(max_frame_r), -1):
        if int((g[int(r)] != int(bg)).sum()) >= 2:
            ctrl_r = int(r)
            break
    assert ctrl_r is not None

    ctrl = _parse_control_runs(g[int(ctrl_r)], int(bg))
    assert len(ctrl) >= 2

    traced = _trace_path_frames(frames, ctrl)

    out = g.copy()
    for i in range(int(len(ctrl)) - 1):
        assert int(traced[int(i)]["area"]) == int(ctrl[int(i)]["length"])
        _draw_bridge(out, traced[int(i)], traced[int(i + 1)], int(ctrl[int(i)]["color"]))

    return out


def _build_frames(
    path: list[tuple[int, int]],
    row_centers: list[int],
    col_centers: list[int],
    inner_by_slot: dict[tuple[int, int], int],
    t_by_slot: dict[tuple[int, int], int],
) -> tuple[list[dict], dict[tuple[int, int], dict]]:
    frames: list[dict] = []
    slot_to_frame: dict[tuple[int, int], dict] = {}
    for rr, cc in path:
        inner_size = int(inner_by_slot[(int(rr), int(cc))])
        t = int(t_by_slot[(int(rr), int(cc))])
        outer = int(inner_size + 2 * t)
        rad = int(outer // 2)
        r_ctr = int(row_centers[int(rr)])
        c_ctr = int(col_centers[int(cc)])
        r0 = int(r_ctr - rad)
        c0 = int(c_ctr - rad)
        r1 = int(r0 + outer - 1)
        c1 = int(c0 + outer - 1)
        f = {
            "bbox": (int(r0), int(c0), int(r1), int(c1)),
            "inner_bbox": (int(r0 + t), int(c0 + t), int(r1 - t), int(c1 - t)),
            "slot": (int(rr), int(cc)),
            "t": int(t),
            "outer": int(outer),
            "inner_size": int(inner_size),
            "area": int(inner_size * inner_size),
            "center2": (int(r0 + r1), int(c0 + c1)),
        }
        frames.append(f)
        slot_to_frame[(int(rr), int(cc))] = f
    return frames, slot_to_frame


def _scene_metrics(inp: np.ndarray) -> dict[str, int]:
    g = np.array(inp, dtype=int)
    bg, _border, frames = _detect_frames(g)
    max_frame_r = max(int(f["bbox"][2]) for f in frames)
    ctrl_r = None
    for r in range(int(g.shape[0] - 1), int(max_frame_r), -1):
        if int((g[int(r)] != int(bg)).sum()) >= 2:
            ctrl_r = int(r)
            break
    assert ctrl_r is not None
    ctrl = _parse_control_runs(g[int(ctrl_r)], int(bg))
    traced = _trace_path_frames(frames, ctrl)

    area_to_outer: dict[int, set[int]] = defaultdict(set)
    for f in frames:
        area_to_outer[int(f["area"])].add(int(f["outer"]))

    mixed_edges = 0
    for a, b in zip(traced, traced[1:]):
        if int(a["inner_size"]) != int(b["inner_size"]):
            mixed_edges += 1

    return {
        "frames": int(len(frames)),
        "distinct_areas": int(len({int(f["area"]) for f in frames})),
        "distinct_thicknesses": int(len({int(f["t"]) for f in frames})),
        "same_area_outer_diversity": int(sum(len(v) >= 2 for v in area_to_outer.values())),
        "long_bars": int(sum(int(item["length"]) == 9 for item in ctrl)),
        "mixed_edges": int(mixed_edges),
    }


def _apply_symmetry(slot: tuple[int, int], code: int) -> tuple[int, int]:
    r, c = map(int, slot)
    if int(code) == 0:
        return int(r), int(c)
    if int(code) == 1:
        return int(c), int(2 - r)
    if int(code) == 2:
        return int(2 - r), int(2 - c)
    if int(code) == 3:
        return int(2 - c), int(r)
    if int(code) == 4:
        return int(r), int(2 - c)
    if int(code) == 5:
        return int(2 - r), int(c)
    if int(code) == 6:
        return int(c), int(r)
    return int(2 - c), int(2 - r)


def _gen_attempt() -> np.ndarray | None:
    bg = int(np.random.choice(np.array(_DIGITS, dtype=int)))
    border = int(np.random.choice(np.array([d for d in _DIGITS if int(d) != int(bg)], dtype=int)))
    inner_choices = [int(d) for d in _DIGITS if int(d) not in (int(bg), int(border))]
    if len(inner_choices) < 2:
        return None
    inner_col = int(np.random.choice(np.array(inner_choices, dtype=int)))

    path_base = _PATH_TEMPLATES[int(np.random.randint(0, len(_PATH_TEMPLATES)))]
    sym = int(np.random.randint(0, 8))
    path = [_apply_symmetry(slot, int(sym)) for slot in path_base]
    pattern = _SIZE_SEQUENCES[int(np.random.randint(0, len(_SIZE_SEQUENCES)))]
    t_pattern = list(_T_SEQUENCE)
    if int(np.random.randint(0, 2)) == 1:
        t_pattern = list(reversed(t_pattern))

    inner_by_slot = {(int(rr), int(cc)): int(sz) for (rr, cc), sz in zip(path, pattern)}
    t_by_slot = {(int(rr), int(cc)): int(t) for (rr, cc), t in zip(path, t_pattern)}

    row_step = int(np.random.choice(np.array([8, 9], dtype=int)))
    col_step = int(np.random.choice(np.array([8, 9], dtype=int)))
    top_margin = int(np.random.randint(3, 5))
    left_margin = int(np.random.randint(3, 5))
    row_centers = [int(top_margin + row_step * rr) for rr in range(3)]
    col_centers = [int(left_margin + col_step * cc) for cc in range(3)]

    ctrl_palette = [int(d) for d in _DIGITS if int(d) not in (int(bg), int(border), int(inner_col))]
    if len(ctrl_palette) < len(path):
        return None
    ctrl_colors = [int(x) for x in np.random.choice(np.array(ctrl_palette, dtype=int), size=len(path), replace=False)]
    ctrl = [{"color": int(col), "length": int(sz * sz)} for col, sz in zip(ctrl_colors, pattern)]

    frames, slot_to_frame = _build_frames(path, row_centers, col_centers, inner_by_slot, t_by_slot)
    traced = _trace_path_frames(frames, ctrl)
    traced_slots = [tuple(map(int, f["slot"])) for f in traced]
    if traced_slots != [tuple(map(int, slot)) for slot in path]:
        return None

    max_r1 = max(int(f["bbox"][2]) for f in frames)
    max_c1 = max(int(f["bbox"][3]) for f in frames)
    ctrl_gap = int(np.random.randint(2, 5))
    ctrl_r = int(max_r1 + ctrl_gap)
    ctrl_spans: list[tuple[int, int, int]] = []
    cur_c = 1
    for item in ctrl:
        ctrl_spans.append((int(cur_c), int(cur_c + int(item["length"]) - 1), int(item["color"])))
        cur_c += int(item["length"] + 1)
    ctrl_end = int(ctrl_spans[-1][1])

    bottom_margin = 1
    right_margin = int(np.random.randint(1, 4))
    H = int(ctrl_r + 1 + bottom_margin)
    W = int(max(int(max_c1 + 1 + right_margin), int(ctrl_end + 1 + right_margin)))
    if not (7 <= int(H) <= 30 and 7 <= int(W) <= 30):
        return None

    out = np.full((int(H), int(W)), int(bg), dtype=int)
    for f in frames:
        rr, cc = map(int, f["slot"])
        r0, c0, _r1, _c1 = map(int, f["bbox"])
        _draw_frame(
            out,
            r0=int(r0),
            c0=int(c0),
            outer=int(f["outer"]),
            t=int(f["t"]),
            border=int(border),
            inner_col=int(inner_col),
        )
        f["inner"] = int(inner_col)
        slot_to_frame[(int(rr), int(cc))]["inner"] = int(inner_col)

    bridge_cells_all: set[tuple[int, int]] = set()
    for i, (a_slot, b_slot) in enumerate(zip(path, path[1:])):
        fa = slot_to_frame[(int(a_slot[0]), int(a_slot[1]))]
        fb = slot_to_frame[(int(b_slot[0]), int(b_slot[1]))]
        cells = _bridge_cells(fa, fb)
        if not cells or cells & bridge_cells_all:
            return None
        for r, c in cells:
            if not (0 <= int(r) < int(H) and 0 <= int(c) < int(W)):
                return None
            if int(out[int(r), int(c)]) != int(bg):
                return None
        for r, c in cells:
            out[int(r), int(c)] = int(ctrl_colors[int(i)])
        bridge_cells_all |= set(cells)

    for c0, c1, col in ctrl_spans:
        out[int(ctrl_r), int(c0) : int(c1) + 1] = int(col)

    inp = out.copy()
    for r, c in bridge_cells_all:
        inp[int(r), int(c)] = int(bg)

    if int(get_bg(inp)) != int(bg):
        return None
    if int(_most_common_nonbg(inp, int(bg)) or -1) != int(border):
        return None

    metrics = _scene_metrics(inp)
    if int(metrics["distinct_areas"]) < 2:
        return None
    if int(metrics["distinct_thicknesses"]) < 2:
        return None
    if int(metrics["same_area_outer_diversity"]) < 1:
        return None
    if int(metrics["long_bars"]) < 2:
        return None
    if int(metrics["mixed_edges"]) < 1:
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
    for _ in range(700):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("Failed to generate valid input after 700 attempts")
