# Variant of 3e6067c3
# Dimension: D7 — 网格形状：输出裁剪策略 (grid_shape)
# Change: 输出网格与输入相同大小，不做裁剪或扩展... → 分离输出为独立帧
#
# Design analysis
# a) Concrete behavior and trace
#    - The input is intentionally kept in the original format: a background field, several equal-sized
#      framed squares, and one control row below them. The control row still specifies a walk through
#      the colored inner squares, and bridges are still inferred exactly as in the source task.
#    - The D7 change is only in output organization. Instead of returning the whole original canvas
#      with all bridges painted back in place, we isolate every consecutive connection:
#        frame_i = {square for ctrl[i], square for ctrl[i+1], bridge colored ctrl[i]}.
#      Each frame is cropped to the bounding box of those two squares plus the bridge, then padded by
#      one background cell. All such frames are concatenated left-to-right, separated by a single
#      full-height divider column in the border color.
#    - Example trace:
#      Original behavior:
#        control = [2, 3, 9, 4]
#        connect 2->3 inside the full scene, then 3->9, then 9->4, and return the full input-sized
#        canvas with all three bridges present.
#      Variant behavior:
#        the same three bridges are inferred, but output becomes:
#          [crop of (2 square, 3 square, bridge color 2)] |divider|
#          [crop of (3 square, 9 square, bridge color 3)] |divider|
#          [crop of (9 square, 4 square, bridge color 9)]
#      The logical bridge construction is unchanged; only the final spatial packaging changes.
# b) Complexity calibration
#    - The variant must be visibly different from the source task, so the generator forces at least
#      three bridge segments (four visited squares). Multiple crops and divider columns make the D7
#      change obvious even when the local bridge geometry itself matches the original rule.
#    - We keep only one divider color (the border color) and one concatenation direction (horizontal)
#      so the solver does not need extra case analysis unrelated to D7.
# c) Review of old assumptions that break under D7
#    - Old `out = g.copy()` assumed output preserved all unrelated pixels. That is false now because
#      unrelated squares and the control row disappear from the output entirely.
#    - Old `draw_bridge()` painted directly into input coordinates. That is still useful for resolving
#      bridge cells, but the final output must be assembled from fresh cropped canvases instead of the
#      original coordinate frame.
#    - Old validation compared against a full-scene solved grid. Generator validation now compares
#      against the concatenated independent-frame output.
# d) Placement legality
#    - The input generator still uses the original lattice/path construction, so squares do not
#      overlap and each bridge spans only background cells between adjacent path neighbors.
#    - After the D7 crop step, each output frame is built from exactly two squares and one bridge, so
#      there is no overlap between unrelated objects inside a frame. Concatenation uses separator
#      columns, so adjacent output frames also remain visually disjoint.
# e) Visual observability
#    - From 2-3 examples a human can see that the output no longer preserves the input layout: the
#      same local bridge motif repeats as a row of separate panels. The divider columns make the new
#      frame structure explicit without relying on hidden numeric properties.
# f) Visual cleanliness
#    - Each frame is tight-cropped with a one-cell background margin, avoiding huge empty regions.
#      Full-height divider columns keep panels readable even when successive frames have different
#      aspect ratios.
# g) Color inference
#    - No new color mapping is introduced. The bridge color inside each frame stays the color of the
#      source square from the corresponding control pair, exactly as in the original task, so the only
#      new concept the solver must infer is panelization.
# Reference-task note
#    - The prompt cites 5545f144 as the `frame_separation` inspiration. In the provided workspace, a
#      repo-wide search finds `5545f144.json` but no corresponding solver `.py`, so there is no solver
#      implementation to read. This variant therefore uses the prompt's stated D7 mechanism directly
#      and keeps all non-D7 logic aligned with the original 3e6067c3 code.

from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np

from arc_utils import get_bg, in_bounds, neighbors4, label_mask


def _most_common_nonbg(g: np.ndarray, bg: int) -> int | None:
    cnt = Counter(map(int, g.flatten()))
    for col, _k in cnt.most_common():
        if int(col) != int(bg):
            return int(col)
    return None


def _draw_frame(g: np.ndarray, *, r0: int, c0: int, outer: int, t: int, border: int, inner_col: int) -> None:
    g[int(r0) : int(r0 + outer), int(c0) : int(c0 + outer)] = int(border)
    g[int(r0 + t) : int(r0 + outer - t), int(c0 + t) : int(c0 + outer - t)] = int(inner_col)


def _center(frame: dict) -> tuple[int, int]:
    r0, c0, r1, c1 = map(int, frame["bbox"])
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

        thick = None
        inner_col = None
        for tt in range(1, int(min(h, w) // 2) + 1):
            inner = g[int(r0 + tt) : int(r1 - tt) + 1, int(c0 + tt) : int(c1 - tt) + 1]
            if inner.size == 0:
                continue
            if bool((inner == int(border)).any()):
                continue
            inner_vals = set(map(int, inner.flatten()))
            if len(inner_vals) != 1:
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
            thick = int(tt)
            inner_col = int(next(iter(inner_vals)))
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


def _parse_scene(grid: np.ndarray) -> tuple[np.ndarray, int, int, list[dict], dict[int, list[dict]], list[int]]:
    g = np.array(grid, dtype=int)
    H, _W = g.shape
    bg = int(get_bg(g))
    border = _most_common_nonbg(g, int(bg))
    assert not (border is None)
    frames = _detect_frames(g, int(border))
    assert not (not frames)

    by: dict[int, list[dict]] = defaultdict(list)
    for frame in frames:
        by[int(frame["inner"])].append(frame)

    max_frame_r = max(int(frame["bbox"][2]) for frame in frames)
    ctrl_r = None
    for r in range(int(H - 1), int(max_frame_r), -1):
        if int((g[int(r)] != int(bg)).sum()) >= 2:
            ctrl_r = int(r)
            break
    assert not (ctrl_r is None)

    ctrl = [int(g[int(ctrl_r), int(c)]) for c in range(int(g.shape[1])) if int(g[int(ctrl_r), int(c)]) != int(bg)]
    assert not (len(ctrl) < 2)
    return g, int(bg), int(border), frames, by, ctrl


def _choose_start(by: dict[int, list[dict]], ctrl: list[int]) -> dict:
    start_color = int(ctrl[0])
    assert not (start_color not in by)
    start_cands = by[start_color]
    if len(start_cands) == 1:
        return start_cands[0]

    nxt_color = int(ctrl[1])
    nxt_cands = by.get(nxt_color, [])
    best = None
    best_frame = None
    for frame in start_cands:
        for other in (nxt_cands if nxt_cands else [frame]):
            same_row, same_col = _aligned(frame, other)
            aligned_score = 1 if (same_row or same_col) else 0
            r1, c1 = _center(frame)
            r2, c2 = _center(other)
            dist = abs(int(r1) - int(r2)) + abs(int(c1) - int(c2))
            cand = (int(aligned_score), -int(dist))
            if best is None or cand > best:
                best = cand
                best_frame = frame
    assert not (best_frame is None)
    return best_frame


def _resolve_segments(by: dict[int, list[dict]], ctrl: list[int]) -> list[tuple[dict, dict, int]]:
    cur = _choose_start(by, ctrl)
    segments: list[tuple[dict, dict, int]] = []
    for i in range(int(len(ctrl)) - 1):
        a = int(ctrl[i])
        b = int(ctrl[i + 1])
        assert int(cur["inner"]) == int(a)
        assert not (b not in by)
        best = None
        best_frame = None
        for frame in by[b]:
            same_row, same_col = _aligned(cur, frame)
            aligned_score = 1 if (same_row or same_col) else 0
            r1, c1 = _center(cur)
            r2, c2 = _center(frame)
            dist = abs(int(r1) - int(r2)) + abs(int(c1) - int(c2))
            cand = (int(aligned_score), -int(dist))
            if best is None or cand > best:
                best = cand
                best_frame = frame
        assert not (best_frame is None)
        segments.append((cur, best_frame, int(a)))
        cur = best_frame
    return segments


def _render_segment_frame(fa: dict, fb: dict, color: int, bg: int, border: int) -> np.ndarray:
    bridge = _bridge_cells(fa, fb)
    assert not (not bridge)
    r0 = min(int(fa["bbox"][0]), int(fb["bbox"][0]), min(int(r) for r, _c in bridge))
    c0 = min(int(fa["bbox"][1]), int(fb["bbox"][1]), min(int(c) for _r, c in bridge))
    r1 = max(int(fa["bbox"][2]), int(fb["bbox"][2]), max(int(r) for r, _c in bridge))
    c1 = max(int(fa["bbox"][3]), int(fb["bbox"][3]), max(int(c) for _r, c in bridge))

    pad = 1
    out = np.full((int(r1 - r0 + 1 + 2 * pad), int(c1 - c0 + 1 + 2 * pad)), int(bg), dtype=int)
    for frame in (fa, fb):
        fr0, fc0, fr1, _fc1 = map(int, frame["bbox"])
        outer = int(fr1 - fr0 + 1)
        _draw_frame(
            out,
            r0=int(fr0 - r0 + pad),
            c0=int(fc0 - c0 + pad),
            outer=int(outer),
            t=int(frame["t"]),
            border=int(border),
            inner_col=int(frame["inner"]),
        )
    for r, c in bridge:
        out[int(r - r0 + pad), int(c - c0 + pad)] = int(color)
    return out


def _assemble_segment_frames(segment_frames: list[np.ndarray], bg: int, divider: int) -> np.ndarray:
    assert not (not segment_frames)
    out_h = max(int(frame.shape[0]) for frame in segment_frames)
    out_w = sum(int(frame.shape[1]) for frame in segment_frames) + max(0, int(len(segment_frames) - 1))
    out = np.full((int(out_h), int(out_w)), int(bg), dtype=int)
    cursor = 0
    for i, frame in enumerate(segment_frames):
        h, w = map(int, frame.shape)
        out[: int(h), int(cursor) : int(cursor + w)] = frame
        cursor += int(w)
        if i + 1 < len(segment_frames):
            out[:, int(cursor)] = int(divider)
            cursor += 1
    return out


def _build_output_from_segments(segments: list[tuple[dict, dict, int]], bg: int, border: int) -> np.ndarray:
    return _assemble_segment_frames(
        [_render_segment_frame(fa, fb, int(color), int(bg), int(border)) for fa, fb, color in segments],
        int(bg),
        int(border),
    )


def main(grid: np.ndarray) -> np.ndarray:
    """
    Variant rule:
    - Infer the same bridge sequence as in 3e6067c3.
    - For each consecutive bridge, output only the two involved framed squares plus that bridge.
    - Crop each bridge scene tightly with one-cell background padding.
    - Concatenate the crops horizontally with one border-colored divider column between them.
    """
    _g, bg, border, _frames, by, ctrl = _parse_scene(grid)
    segments = _resolve_segments(by, ctrl)
    return _build_output_from_segments(segments, int(bg), int(border))


_DIGITS = [int(i) for i in range(10)]


def _gen_attempt() -> np.ndarray | None:
    bg = int(np.random.choice(np.array(_DIGITS, dtype=int)))
    border = int(np.random.choice(np.array([d for d in _DIGITS if int(d) != int(bg)], dtype=int)))
    inner_palette = [int(d) for d in _DIGITS if int(d) not in (int(bg), int(border))]
    if len(inner_palette) < 4:
        return None

    t = 1
    inner = int(np.random.randint(1, 3))
    outer = int(inner + 2 * t)
    if not (3 <= int(outer) <= 4):
        return None

    R = int(np.random.randint(2, 5))
    C = int(np.random.randint(2, 5))
    max_frames = int(R * C)
    if max_frames < 4:
        return None
    n_frames = int(np.random.randint(4, int(min(5, max_frames)) + 1))

    path: list[tuple[int, int]] | None = None
    for _path_try in range(400):
        p = [(0, 0)]
        vis = {(0, 0)}
        ok = True
        for _ in range(int(n_frames) - 1):
            r, c = map(int, p[-1])
            opts = [
                (int(rr), int(cc))
                for rr, cc in neighbors4(int(r), int(c))
                if in_bounds((int(R), int(C)), int(rr), int(cc)) and (int(rr), int(cc)) not in vis
            ]
            if not opts:
                ok = False
                break
            nxt = opts[int(np.random.randint(0, len(opts)))]
            p.append((int(nxt[0]), int(nxt[1])))
            vis.add((int(nxt[0]), int(nxt[1])))
        if not ok:
            continue
        if len({r for r, _c in p}) < 2 or len({c for _r, c in p}) < 2:
            continue
        path = p
        break
    if path is None:
        return None

    row_gap = [int(np.random.randint(1, 4)) for _ in range(int(R) - 1)]
    col_gap = [int(np.random.randint(1, 4)) for _ in range(int(C) - 1)]
    top_margin = int(np.random.randint(1, 3))
    left_margin = int(np.random.randint(1, 3))

    r0s = [int(top_margin)]
    for k in range(int(R) - 1):
        r0s.append(int(r0s[-1] + outer + row_gap[int(k)]))
    c0s = [int(left_margin)]
    for k in range(int(C) - 1):
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
            "t": int(t),
            "inner": -1,
            "inner_bbox": (int(r0 + t), int(c0 + t), int(r1 - t), int(c1 - t)),
            "slot": (int(rr), int(cc)),
        }
        frames.append(frame)
        slot_to_frame[(int(rr), int(cc))] = frame

    max_r1 = max(int(frame["bbox"][2]) for frame in frames)
    max_c1 = max(int(frame["bbox"][3]) for frame in frames)
    ctrl_gap = int(np.random.randint(2, 5))
    ctrl_r = int(max_r1 + ctrl_gap)
    bottom_margin = 1
    right_margin = int(np.random.randint(1, 4))
    H = int(ctrl_r + 1 + bottom_margin)
    W = int(max_c1 + 1 + right_margin)
    if not (7 <= int(H) <= 30 and 7 <= int(W) <= 30):
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
        chosen = int(np.random.choice(np.array(choices, dtype=int)))
        colors_by_slot[(int(rr), int(cc))] = int(chosen)
        used_in_row[int(rr)].add(int(chosen))
        used_in_col[int(cc)].add(int(chosen))
    if len(colors_by_slot) != len(path):
        return None

    for frame in frames:
        rr, cc = map(int, frame["slot"])
        frame["inner"] = int(colors_by_slot[(int(rr), int(cc))])

    solved_full = np.full((int(H), int(W)), int(bg), dtype=int)
    for frame in frames:
        r0, c0, r1, _c1 = map(int, frame["bbox"])
        _draw_frame(
            solved_full,
            r0=int(r0),
            c0=int(c0),
            outer=int(r1 - r0 + 1),
            t=int(t),
            border=int(border),
            inner_col=int(frame["inner"]),
        )

    bridge_cells_all: set[tuple[int, int]] = set()
    segments: list[tuple[dict, dict, int]] = []
    ok = True
    for a_slot, b_slot in zip(path, path[1:]):
        fa = slot_to_frame[(int(a_slot[0]), int(a_slot[1]))]
        fb = slot_to_frame[(int(b_slot[0]), int(b_slot[1]))]
        cells = _bridge_cells(fa, fb)
        if not cells:
            ok = False
            break
        if cells & bridge_cells_all:
            ok = False
            break
        for r, c in cells:
            if not (0 <= int(r) < int(H) and 0 <= int(c) < int(W)):
                ok = False
                break
            if int(solved_full[int(r), int(c)]) != int(bg):
                ok = False
                break
        if not ok:
            break
        color = int(colors_by_slot[(int(a_slot[0]), int(a_slot[1]))])
        for r, c in cells:
            solved_full[int(r), int(c)] = int(color)
        bridge_cells_all |= set(cells)
        segments.append((fa, fb, int(color)))
    if not ok:
        return None

    ctrl_colors = [int(colors_by_slot[(int(r), int(c))]) for r, c in path]
    ctrl_cols = [int(1 + 2 * i) for i in range(int(len(ctrl_colors)))]
    if int(ctrl_cols[-1]) >= int(W):
        return None
    for color, col in zip(ctrl_colors, ctrl_cols):
        solved_full[int(ctrl_r), int(col)] = int(color)

    out = _build_output_from_segments(segments, int(bg), int(border))
    if not (3 <= int(out.shape[0]) <= 30 and 3 <= int(out.shape[1]) <= 30):
        return None

    inp = solved_full.copy()
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
