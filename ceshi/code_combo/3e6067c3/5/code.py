# Combo variant of 3e6067c3 (index 5)
# combo_spec: D1/3+D8/1
# Name: 像素计数匹配+反向处理
#
# Design analysis
# a) Concrete behavior and trace
#    - Original task: each framed square has a solid interior color, and the bottom control row is read
#      left-to-right as a color sequence. Consecutive control symbols choose consecutive squares, and the
#      bridge between them is painted with the current square color.
#    - Combo task: each frame interior is mostly background plus 1..4 same-colored dots, so control tokens
#      no longer name colors directly. Instead, a contiguous run of k identical control pixels means
#      "choose a k-dot square". On top of that, the control row must be interpreted from right to left.
#    - Small trace:
#      input control row (left to right): `[####][#][###]`, where `#` is a fixed marker color.
#      Parsed left-to-right this is `[4,1,3]`, but the solver must reverse it to `[3,1,4]`.
#      It therefore starts from the 3-dot frame, then goes to a 1-dot frame, then to a 4-dot frame.
#      Each bridge is still painted with the current frame's dot color, not with the control marker color.
#    - The visible difference from the original task is therefore twofold:
#      1. The main-region identity cue is dot count instead of fill color.
#      2. The bottom row looks reversed relative to the actual bridge chain, so examples force the human
#         solver to infer right-to-left reading.
#
#    Note on the D8 reference: the supplied single-variant D8/1 file implements "from both ends inward",
#    but the combo description here explicitly says "read the control row from right to left" and
#    "connect in reverse order". This implementation follows the combo description, because that is the
#    direct specification for the fused task.
#
# b) Complexity calibration
#    - Counts are limited to 1..4. That is enough repetition to require geometric disambiguation, but still
#      easy for humans to count from a few examples.
#    - A single dedicated control-marker color is used for every run, so the task cannot collapse back into
#      ordinary color matching.
#    - Dot paint colors are unique per frame, which keeps bridge colors legible and makes reverse-order
#      effects visible even when the same undirected corridor set is reused.
#
# c) Old assumptions that break
#    - Original frame extraction assumed the full interior was a single solid color. That fails here because
#      the interior is mostly background with sparse dots.
#    - Original control parsing treated each non-background control cell as one token. Here a run of length
#      4 is one token with value 4.
#    - Original matching `cur["inner"] == control_color` no longer makes sense. Matching must use
#      `cur["count"] == control_count`.
#    - Original solver read control symbols left-to-right. Here the parsed run sequence must be reversed
#      before traversal.
#    - Original bridge paint came from the control symbol color. Here the control row is monochrome, so
#      bridge color must come from the current frame's dot color.
#
# d) Placement legality
#    - Frames remain axis-aligned border components with a clean rectangular inner box.
#    - The generator builds a self-avoiding lattice path and only uses bridges between orthogonally adjacent
#      lattice slots, so every bridge remains a straight row-band or column-band rectangle.
#    - Dot placement is restricted to fixed offsets inside a 3x3 inner region, so dots never touch borders.
#    - Control runs are separated by at least one background cell, which makes run parsing unambiguous.
#
# e) Visual observability
#    - A human can see that the control row is monochrome and grouped into runs of different lengths, so the
#      matching cue must be something like quantity rather than color.
#    - Because the visual chain in the main region follows the reverse of the left-to-right control layout,
#      examples can support the inference that the row must be read from right to left.
#    - The generator rejects samples where right-to-left and left-to-right processing produce the same
#      output, so the reverse-order rule is not a hidden no-op.
#
# f) Visual cleanliness
#    - Only three object types appear: border rings, sparse interior dots, and bridge rectangles.
#    - Bridges are generated disjointly, so the fused task stays readable; the reverse-order behavior is
#      shown by corridor color assignment rather than by messy overdraw.
#    - Control runs sit on their own row below the frames with a blank row underneath, preserving the
#      original scene organization.
#
# g) Change inferability
#    - No numeric meaning is attached to actual color values. Counts are visible through run lengths and dot
#      multiplicities only.
#    - Bridge colors remain structurally inferable because they always match the source frame's dot color.
#    - The right-to-left rule is tied to spatial order in the control row, not to any hidden sort key.
#
# 9) New assumptions review
#    - `main()` now assumes every valid frame interior contains exactly one non-background paint color and
#      1..4 pixels of that color. `generate_input()` enforces this directly.
#    - `main()` assumes the control row below all frames contains contiguous runs of one marker color.
#      `generate_input()` writes exactly that form.
#    - `main()` assumes the intended next square among same-count candidates is resolved by row/column-band
#      alignment and nearest-distance tie-breaks. The generator assigns counts from a small Latin-like
#      lattice pattern, validates by replaying `main()`, and rejects ambiguous layouts.
#    - `main()` assumes the right-to-left interpretation matters. The generator measures the cellwise
#      difference against the left-to-right variant and rejects zero-difference samples.

from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np

from arc_utils import get_bg, in_bounds, label_mask, neighbors4


_DIGITS = [int(i) for i in range(10)]


def _most_common_nonbg(grid: np.ndarray, bg: int) -> int | None:
    cnt = Counter(map(int, grid.flatten()))
    for col, _k in cnt.most_common():
        if int(col) != int(bg):
            return int(col)
    return None


def _dot_offsets(count: int) -> list[tuple[int, int]]:
    patterns = {
        1: [(1, 1)],
        2: [(0, 0), (2, 2)],
        3: [(0, 0), (1, 1), (2, 2)],
        4: [(0, 0), (0, 2), (2, 0), (2, 2)],
    }
    return patterns[int(count)]


def _draw_frame(
    grid: np.ndarray,
    *,
    r0: int,
    c0: int,
    outer: int,
    t: int,
    bg: int,
    border: int,
    paint: int,
    count: int,
) -> None:
    grid[int(r0) : int(r0 + outer), int(c0) : int(c0 + outer)] = int(border)
    grid[int(r0 + t) : int(r0 + outer - t), int(c0 + t) : int(c0 + outer - t)] = int(bg)
    for dr, dc in _dot_offsets(int(count)):
        grid[int(r0 + t + dr), int(c0 + t + dc)] = int(paint)


def _bridge_cells(frame_a: dict, frame_b: dict) -> set[tuple[int, int]]:
    r0a, c0a, r1a, c1a = map(int, frame_a["bbox"])
    r0b, c0b, r1b, c1b = map(int, frame_b["bbox"])
    ia = frame_a["inner_bbox"]
    ib = frame_b["inner_bbox"]
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


def _center(frame: dict) -> tuple[int, int]:
    r0, c0, r1, c1 = map(int, frame["bbox"])
    return int((r0 + r1) // 2), int((c0 + c1) // 2)


def _aligned(frame_a: dict, frame_b: dict) -> tuple[bool, bool]:
    a = frame_a["inner_bbox"]
    b = frame_b["inner_bbox"]
    same_row = bool(int(a[0]) == int(b[0]) and int(a[2]) == int(b[2]))
    same_col = bool(int(a[1]) == int(b[1]) and int(a[3]) == int(b[3]))
    return same_row, same_col


def _extract_frames(grid: np.ndarray, bg: int, border: int) -> list[dict]:
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
        if h < 5 or w < 5:
            continue

        found = None
        for tt in range(1, int(min(h, w) // 2) + 1):
            inner = grid[int(r0 + tt) : int(r1 - tt) + 1, int(c0 + tt) : int(c1 - tt) + 1]
            if inner.shape != (3, 3):
                continue
            if bool((inner == int(border)).any()):
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

            inner_mask = inner != int(bg)
            dot_count = int(inner_mask.sum())
            if dot_count < 1 or dot_count > 4:
                continue
            colors = set(map(int, inner[inner_mask].flatten()))
            if len(colors) != 1:
                continue
            found = {
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
                "t": int(tt),
                "count": int(dot_count),
                "paint": int(next(iter(colors))),
                "inner_bbox": (int(r0 + tt), int(c0 + tt), int(r1 - tt), int(c1 - tt)),
            }
            break

        if found is not None:
            frames.append(found)
    return frames


def _parse_control_counts(grid: np.ndarray, bg: int, min_row: int) -> list[int]:
    ctrl_r = None
    for r in range(int(grid.shape[0] - 1), int(min_row), -1):
        if int((grid[int(r)] != int(bg)).sum()) >= 2:
            ctrl_r = int(r)
            break
    assert ctrl_r is not None

    counts: list[int] = []
    c = 0
    while int(c) < int(grid.shape[1]):
        if int(grid[int(ctrl_r), int(c)]) == int(bg):
            c += 1
            continue
        c0 = int(c)
        while int(c) < int(grid.shape[1]) and int(grid[int(ctrl_r), int(c)]) != int(bg):
            c += 1
        counts.append(int(c - c0))
    assert len(counts) >= 2
    return counts


def _draw_bridge(out: np.ndarray, frame_a: dict, frame_b: dict, color: int) -> None:
    for r, c in _bridge_cells(frame_a, frame_b):
        if 0 <= int(r) < out.shape[0] and 0 <= int(c) < out.shape[1]:
            out[int(r), int(c)] = int(color)


def _solve_grid(grid: np.ndarray, *, read_right_to_left: bool) -> np.ndarray:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    cnt = Counter(map(int, g.flatten()))
    border = None
    for col, _k in cnt.most_common():
        if int(col) != int(bg):
            border = int(col)
            break
    assert border is not None

    frames = _extract_frames(g, int(bg), int(border))
    assert frames

    by_count: dict[int, list[dict]] = defaultdict(list)
    for frame in frames:
        by_count[int(frame["count"])].append(frame)

    max_frame_r = max(int(frame["bbox"][2]) for frame in frames)
    ctrl = _parse_control_counts(g, int(bg), int(max_frame_r))
    if bool(read_right_to_left):
        ctrl = list(reversed(ctrl))

    start_count = int(ctrl[0])
    assert start_count in by_count
    start_cands = by_count[start_count]
    if len(start_cands) == 1:
        cur = start_cands[0]
    else:
        nxt_count = int(ctrl[1])
        nxt_cands = by_count.get(nxt_count, [])
        best = None
        best_frame = None
        for frame in start_cands:
            for nxt in (nxt_cands if nxt_cands else [frame]):
                same_row, same_col = _aligned(frame, nxt)
                aligned = 1 if (same_row or same_col) else 0
                r1, c1 = _center(frame)
                r2, c2 = _center(nxt)
                dist = abs(int(r1) - int(r2)) + abs(int(c1) - int(c2))
                cand = (int(aligned), -int(dist))
                if best is None or cand > best:
                    best = cand
                    best_frame = frame
        cur = best_frame
    assert cur is not None

    out = g.copy()
    for i in range(int(len(ctrl)) - 1):
        a = int(ctrl[i])
        b = int(ctrl[i + 1])
        assert int(cur["count"]) == int(a)
        assert b in by_count

        best = None
        nxt_frame = None
        for frame in by_count[b]:
            same_row, same_col = _aligned(cur, frame)
            aligned = 1 if (same_row or same_col) else 0
            r1, c1 = _center(cur)
            r2, c2 = _center(frame)
            dist = abs(int(r1) - int(r2)) + abs(int(c1) - int(c2))
            cand = (int(aligned), -int(dist))
            if best is None or cand > best:
                best = cand
                nxt_frame = frame
        assert nxt_frame is not None
        _draw_bridge(out, cur, nxt_frame, int(cur["paint"]))
        cur = nxt_frame

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Solve the fused variant:
    - frames are matched by interior dot count (1..4)
    - control tokens are monochrome runs
    - the run sequence must be read from right to left
    """
    return _solve_grid(grid, read_right_to_left=True)


def _slot_count(rr: int, cc: int, offset: int) -> int:
    return int((int(rr) + int(cc) + int(offset)) % 4 + 1)


def _directional_difference(inp: np.ndarray) -> int:
    rev = _solve_grid(inp, read_right_to_left=True)
    ltr = _solve_grid(inp, read_right_to_left=False)
    return int(np.count_nonzero(rev != ltr))


def _gen_attempt() -> np.ndarray | None:
    bg = int(np.random.choice(np.array(_DIGITS, dtype=int)))
    border = int(np.random.choice(np.array([d for d in _DIGITS if int(d) != int(bg)], dtype=int)))
    ctrl_choices = [int(d) for d in _DIGITS if int(d) not in (int(bg), int(border))]
    if not ctrl_choices:
        return None
    ctrl_color = int(np.random.choice(np.array(ctrl_choices, dtype=int)))
    paint_palette = [int(d) for d in _DIGITS if int(d) not in (int(bg), int(border), int(ctrl_color))]
    if len(paint_palette) < 5:
        return None

    t = int(np.random.choice(np.array([1, 1, 2], dtype=int)))
    inner = 3
    outer = int(inner + 2 * t)

    dims = [(2, 3), (3, 2), (3, 3)]
    R, C = dims[int(np.random.randint(0, len(dims)))]
    max_frames = int(R * C)
    n_frames = int(np.random.randint(5, int(min(7, max_frames)) + 1))
    if int(n_frames) > int(len(paint_palette)):
        return None

    path: list[tuple[int, int]] | None = None
    for _ in range(500):
        start = (int(np.random.randint(0, R)), int(np.random.randint(0, C)))
        p = [start]
        seen = {start}
        ok = True
        for _step in range(int(n_frames) - 1):
            r, c = map(int, p[-1])
            opts = [
                (int(rr), int(cc))
                for rr, cc in neighbors4(int(r), int(c))
                if in_bounds((int(R), int(C)), int(rr), int(cc)) and (int(rr), int(cc)) not in seen
            ]
            if not opts:
                ok = False
                break
            nxt = opts[int(np.random.randint(0, len(opts)))]
            p.append((int(nxt[0]), int(nxt[1])))
            seen.add((int(nxt[0]), int(nxt[1])))
        if not ok:
            continue
        if len({int(r) for r, _c in p}) < 2 or len({int(c) for _r, c in p}) < 2:
            continue
        path = p
        break
    if path is None:
        return None

    row_gap = [int(np.random.randint(1, 5)) for _ in range(int(R) - 1)]
    col_gap = [int(np.random.randint(1, 5)) for _ in range(int(C) - 1)]
    top_margin = int(np.random.randint(1, 4))
    left_margin = int(np.random.randint(1, 4))

    r0s = [int(top_margin)]
    for gap in row_gap:
        r0s.append(int(r0s[-1] + outer + int(gap)))
    c0s = [int(left_margin)]
    for gap in col_gap:
        c0s.append(int(c0s[-1] + outer + int(gap)))

    frames: list[dict] = []
    slot_to_frame: dict[tuple[int, int], dict] = {}
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

    offset = int(np.random.randint(0, 4))
    counts_by_slot = {(int(rr), int(cc)): int(_slot_count(int(rr), int(cc), int(offset))) for rr, cc in path}
    path_counts = [int(counts_by_slot[(int(rr), int(cc))]) for rr, cc in path]
    if len(set(path_counts)) < 3:
        return None
    if 4 not in path_counts:
        return None
    if path_counts == list(reversed(path_counts)):
        return None

    paints = np.random.permutation(np.array(paint_palette, dtype=int))[: len(path)]
    paint_by_slot = {(int(slot[0]), int(slot[1])): int(paints[i]) for i, slot in enumerate(path)}

    max_r1 = max(int(frame["bbox"][2]) for frame in frames)
    max_c1 = max(int(frame["bbox"][3]) for frame in frames)
    ctrl_counts_ltr = list(reversed(path_counts))
    ctrl_gap = int(np.random.randint(2, 6))
    ctrl_r = int(max_r1 + ctrl_gap)
    ctrl_needed_w = int(1 + sum(ctrl_counts_ltr) + (len(ctrl_counts_ltr) - 1) + 1)
    bottom_margin = 1
    right_margin = int(np.random.randint(1, 5))
    H = int(ctrl_r + 1 + bottom_margin)
    W = int(max(int(max_c1 + 1 + right_margin), int(ctrl_needed_w)))
    if not (7 <= int(H) <= 30 and 7 <= int(W) <= 30):
        return None

    out = np.full((int(H), int(W)), int(bg), dtype=int)
    for frame in frames:
        rr, cc = map(int, frame["slot"])
        r0, c0, r1, _c1 = map(int, frame["bbox"])
        _draw_frame(
            out,
            r0=int(r0),
            c0=int(c0),
            outer=int(r1 - r0 + 1),
            t=int(t),
            bg=int(bg),
            border=int(border),
            paint=int(paint_by_slot[(int(rr), int(cc))]),
            count=int(counts_by_slot[(int(rr), int(cc))]),
        )

    bridge_union: set[tuple[int, int]] = set()
    ok = True
    for a_slot, b_slot in zip(path, path[1:]):
        frame_a = slot_to_frame[(int(a_slot[0]), int(a_slot[1]))]
        frame_b = slot_to_frame[(int(b_slot[0]), int(b_slot[1]))]
        cells = _bridge_cells(frame_a, frame_b)
        if not cells or bool(cells & bridge_union):
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
        paint = int(paint_by_slot[(int(a_slot[0]), int(a_slot[1]))])
        for r, c in cells:
            out[int(r), int(c)] = int(paint)
        bridge_union |= set(cells)
    if not ok:
        return None

    c = 1
    for count in ctrl_counts_ltr:
        if int(c + count) > int(W):
            return None
        out[int(ctrl_r), int(c) : int(c + count)] = int(ctrl_color)
        c += int(count + 1)

    inp = out.copy()
    for r, c in bridge_union:
        inp[int(r), int(c)] = int(bg)

    if int(get_bg(inp)) != int(bg):
        return None
    if int(_most_common_nonbg(inp, int(bg)) or -1) != int(border):
        return None

    try:
        pred = main(inp)
        diff_cells = _directional_difference(inp)
    except Exception:
        return None
    if pred.shape != out.shape:
        return None
    if not bool(np.array_equal(pred, out)):
        return None
    if int(diff_cells) <= 0:
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(800):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("Failed to generate valid input after 800 attempts")
