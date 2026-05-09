# Combo variant of 3e6067c3 (index 25)
# combo_spec: D1/3+D2/3+D8/1
# Name: 计数匹配+位置消歧+反向处理
#
# Design analysis
# a) Concrete behavior and trace
# - The original task has solid-color framed squares and a bottom control row of single colored pixels.
#   A left-to-right color sequence chooses squares, and each consecutive pair gets connected by a bridge
#   painted with the current square's color.
# - This combo changes three linked steps at once:
#   1. Matching uses interior dot counts, not fill colors. Each frame interior is mostly background and
#      contains exactly 1, 2, or 3 same-colored dots. The control row uses contiguous runs of one marker
#      color, and the run length selects the target dot count.
#   2. If several frames share that count, selection no longer depends on local alignment/proximity.
#      Frames are sorted globally by position (top-to-bottom, then left-to-right), and the first
#      unvisited matching frame is chosen.
#   3. The control sequence is read from right to left. In the generated scenes the visible left-to-right
#      control runs are the reverse of the true visitation order.
# - Small trace with counts `[a,b,a,c,a]` along the intended bridge path:
#   left-to-right control runs look like `[a,c,a,b,a]`, but the solver reads them right-to-left and
#   recovers `[a,b,a,c,a]`. Suppose count `b` appears both in the intended second frame and in a closer
#   decoy below the start frame. The combo picks the higher `b` frame because it is earlier in global
#   order, then continues to the next unvisited `a`, then the unique `c`, then the last `a`.
# - Compared with the original task, humans now have to infer three non-obvious changes together:
#   count the dots, ignore bridge-nearest decoys when counts repeat, and read the control row backwards.
#
# b) Complexity calibration
# - Counts stay in the small alphabet {1,2,3}; that is enough to force repetition without making the
#   scene numerically busy.
# - All control runs use one dedicated marker color absent from frame paints, so color matching cannot
#   accidentally solve the task.
# - Only one carefully placed count-decoy is added. This makes the new tie-break visible without flooding
#   the grid with many distractors.
# - The generator uses a fixed orthogonal skeleton plus random colors, margins, frame thickness, inner
#   size, and horizontal mirroring. This keeps samples diverse while preserving legality and a high
#   generation success rate.
#
# c) Implicit assumptions reviewed
# - Original frame parsing assumed a solid interior color. That breaks here because the interior is mostly
#   background, so parsing must accept sparse dots and extract both dot count and paint color.
# - Original control parsing treated every non-background pixel as one token. That breaks here because a
#   run of length 2 or 3 is one symbol.
# - Original candidate choice depended on alignment/proximity and had no visited set. That breaks here:
#   repeated counts require a global order plus visited tracking.
# - Original left-to-right iteration over the control row breaks because the combo spec explicitly
#   concretizes the processing change as right-to-left reading. Although the standalone D8 reference used
#   an alternating-ends scheme, this combo follows the fusion description, which is the binding rule.
# - The new solver assumes that, for the intended sequence, the first unvisited frame of each requested
#   count is bridge-connectable from the current frame. The generator guarantees this by construction.
#
# d) Placement legality
# - Frames sit on a regular 3x3 lattice with at least one background cell between neighboring borders.
# - The true bridge path follows orthogonally adjacent slots, so each bridge occupies a single empty
#   corridor and never overlaps a frame or the control row.
# - The decoy frame is placed off the true path but still on the lattice, keeping frame detection clean.
#
# e) Visual observability
# - A human can see monochrome control runs of varying lengths and frame interiors with sparse counted dots.
# - The first repeated-count decision is deliberately visible: a closer decoy exists, but the output bridge
#   goes to the earlier-by-position frame instead.
# - Because the control row is the reverse of the actual traversal, the first painted bridge color comes
#   from the rightmost control token's target rather than the leftmost token's target.
#
# f) Visual cleanliness
# - There are only four visible layers: background, border rings, sparse dots, and bridges.
# - The control row is isolated below the frames with one blank row underneath, matching the original task's
#   clean separation.
# - No bridge crossings are needed for this combo; the reverse-read effect is already visible because
#   reversing direction changes which frame color paints each corridor.
#
# g) Change inferability
# - No hidden color ordering is used. Count is directly visible from dots and control-run lengths.
# - Position disambiguation uses only visible top/down and left/right order.
# - Reverse reading is inferable because the control row visually lists the path in the opposite direction
#   from the bridges painted in the output.
#
# New solver assumptions and generator guarantees
# - Every valid frame interior has exactly one non-background paint color and 1..3 pixels of that color.
# - The control row below all frames contains contiguous non-background runs encoding 1..3.
# - At least one repeated-count transition has a local-nearest decoy that disagrees with global ordering.
# - Reading the same control runs left-to-right yields a different output from the intended right-to-left
#   rule. The generator validates all of these conditions before accepting a sample.

from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np

from arc_utils import get_bg, label_mask


_DIGITS = [int(i) for i in range(10)]
_PATH_SLOTS = [(0, 0), (0, 1), (1, 1), (1, 2), (2, 2)]
_DECOY_SLOT = (1, 0)


def _frame_order_key(frame: dict) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = map(int, frame["bbox"])
    return int(r0), int(c0), int(r1), int(c1)


def _most_common_nonbg(grid: np.ndarray, bg: int) -> int | None:
    cnt = Counter(map(int, grid.flatten()))
    for col, _k in cnt.most_common():
        if int(col) != int(bg):
            return int(col)
    return None


def _dot_offsets(inner: int, count: int) -> list[tuple[int, int]]:
    inner = int(inner)
    count = int(count)
    if inner == 2:
        patterns = {
            1: [(0, 0)],
            2: [(0, 0), (1, 1)],
            3: [(0, 0), (0, 1), (1, 1)],
        }
        return patterns[int(count)]
    patterns = {
        1: [(1, 1)],
        2: [(0, 0), (2, 2)],
        3: [(0, 0), (1, 1), (2, 2)],
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
    inner = int(outer - 2 * t)
    for dr, dc in _dot_offsets(int(inner), int(count)):
        grid[int(r0 + t + dr), int(c0 + t + dc)] = int(paint)


def _aligned(frame_a: dict, frame_b: dict) -> tuple[bool, bool]:
    a = frame_a["inner_bbox"]
    b = frame_b["inner_bbox"]
    same_row = bool(int(a[0]) == int(b[0]) and int(a[2]) == int(b[2]))
    same_col = bool(int(a[1]) == int(b[1]) and int(a[3]) == int(b[3]))
    return same_row, same_col


def _center(frame: dict) -> tuple[int, int]:
    r0, c0, r1, c1 = map(int, frame["bbox"])
    return int((r0 + r1) // 2), int((c0 + c1) // 2)


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
        return {
            (int(r), int(c))
            for r in range(int(rr0), int(rr1) + 1)
            for c in range(int(cc0), int(cc1) + 1)
        }
    if int(ia[1]) == int(ib[1]) and int(ia[3]) == int(ib[3]):
        cc0, cc1 = int(ia[1]), int(ia[3])
        if int(r1a) < int(r0b):
            rr0, rr1 = int(r1a + 1), int(r0b - 1)
        else:
            rr0, rr1 = int(r1b + 1), int(r0a - 1)
        if int(rr0) > int(rr1):
            return set()
        return {
            (int(r), int(c))
            for r in range(int(rr0), int(rr1) + 1)
            for c in range(int(cc0), int(cc1) + 1)
        }
    return set()


def _paint_bridge(grid: np.ndarray, frame_a: dict, frame_b: dict, color: int) -> None:
    for r, c in _bridge_cells(frame_a, frame_b):
        if 0 <= int(r) < grid.shape[0] and 0 <= int(c) < grid.shape[1]:
            grid[int(r), int(c)] = int(color)


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
        if h < 4 or w < 4:
            continue

        found = None
        for tt in range(1, int(min(h, w) // 2) + 1):
            inner = grid[int(r0 + tt) : int(r1 - tt) + 1, int(c0 + tt) : int(c1 - tt) + 1]
            if inner.size == 0:
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

            dot_mask = inner != int(bg)
            dot_count = int(dot_mask.sum())
            if dot_count < 1 or dot_count > 3:
                continue
            colors = set(map(int, inner[dot_mask].flatten()))
            if len(colors) != 1:
                continue
            paint = int(next(iter(colors)))
            found = {
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
                "t": int(tt),
                "count": int(dot_count),
                "paint": int(paint),
                "inner_bbox": (int(r0 + tt), int(c0 + tt), int(r1 - tt), int(c1 - tt)),
            }
            break

        if found is not None:
            frames.append(found)

    frames.sort(key=_frame_order_key)
    for idx, frame in enumerate(frames):
        frame["order"] = int(idx)
    return frames


def _control_counts_ltr(grid: np.ndarray, bg: int, min_row: int) -> list[int]:
    H, W = grid.shape
    ctrl_r = None
    for r in range(int(H - 1), int(min_row), -1):
        if int((grid[int(r)] != int(bg)).sum()) >= 2:
            ctrl_r = int(r)
            break
    assert not (ctrl_r is None)

    counts: list[int] = []
    c = 0
    while int(c) < int(W):
        if int(grid[int(ctrl_r), int(c)]) == int(bg):
            c += 1
            continue
        c0 = int(c)
        while int(c) < int(W) and int(grid[int(ctrl_r), int(c)]) != int(bg):
            c += 1
        counts.append(int(c - c0))
    assert not (len(counts) < 2)
    return counts


def _parse_scene(grid: np.ndarray) -> tuple[np.ndarray, int, int, list[dict], dict[int, list[dict]], list[int]]:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    border = _most_common_nonbg(g, int(bg))
    assert not (border is None)

    frames = _extract_frames(g, int(bg), int(border))
    assert not (not frames)

    by_count: dict[int, list[dict]] = defaultdict(list)
    for frame in frames:
        by_count[int(frame["count"])].append(frame)
    for count in by_count:
        by_count[int(count)].sort(key=_frame_order_key)

    max_frame_r = max(int(frame["bbox"][2]) for frame in frames)
    ctrl_ltr = _control_counts_ltr(g, int(bg), int(max_frame_r))
    return g, int(bg), int(border), frames, by_count, ctrl_ltr


def _select_global_first(by_count: dict[int, list[dict]], count: int, visited: set[int]) -> dict | None:
    for frame in by_count.get(int(count), []):
        order = int(frame["order"])
        if order not in visited:
            return frame
    return None


def _select_local_best(cur: dict, cands: list[dict]) -> dict | None:
    best = None
    bestf = None
    for frame in cands:
        if frame is cur:
            continue
        same_row, same_col = _aligned(cur, frame)
        aligned = 1 if (same_row or same_col) else 0
        r1, c1 = _center(cur)
        r2, c2 = _center(frame)
        dist = abs(int(r1) - int(r2)) + abs(int(c1) - int(c2))
        cand = (int(aligned), -int(dist), -int(frame["order"]))
        if best is None or cand > best:
            best = cand
            bestf = frame
    return bestf


def _solve(grid: np.ndarray, *, reverse_read: bool, use_global_order: bool) -> np.ndarray:
    g, _bg, _border, _frames, by_count, ctrl_ltr = _parse_scene(grid)
    ctrl = list(reversed(ctrl_ltr)) if bool(reverse_read) else list(ctrl_ltr)
    out = g.copy()

    start_count = int(ctrl[0])
    assert not (start_count not in by_count)

    if bool(use_global_order):
        visited: set[int] = set()
        cur = _select_global_first(by_count, int(start_count), visited)
        assert not (cur is None)
        visited.add(int(cur["order"]))
        for idx in range(int(len(ctrl)) - 1):
            cur_count = int(ctrl[idx])
            next_count = int(ctrl[idx + 1])
            assert int(cur["count"]) == int(cur_count)
            nxt = _select_global_first(by_count, int(next_count), visited)
            if nxt is None:
                continue
            _paint_bridge(out, cur, nxt, int(cur["paint"]))
            visited.add(int(nxt["order"]))
            cur = nxt
        return out

    start_cands = by_count[int(start_count)]
    if len(start_cands) == 1:
        cur = start_cands[0]
    else:
        next_count = int(ctrl[1])
        next_cands = by_count.get(int(next_count), [])
        best = None
        bestf = None
        for frame in start_cands:
            pool = next_cands if next_cands else [frame]
            for other in pool:
                if frame is other:
                    continue
                same_row, same_col = _aligned(frame, other)
                aligned = 1 if (same_row or same_col) else 0
                r1, c1 = _center(frame)
                r2, c2 = _center(other)
                dist = abs(int(r1) - int(r2)) + abs(int(c1) - int(c2))
                cand = (int(aligned), -int(dist), -int(frame["order"]))
                if best is None or cand > best:
                    best = cand
                    bestf = frame
        cur = bestf
    assert not (cur is None)

    for idx in range(int(len(ctrl)) - 1):
        cur_count = int(ctrl[idx])
        next_count = int(ctrl[idx + 1])
        assert int(cur["count"]) == int(cur_count)
        nxt = _select_local_best(cur, by_count.get(int(next_count), []))
        if nxt is None:
            continue
        _paint_bridge(out, cur, nxt, int(cur["paint"]))
        cur = nxt
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect framed squares using:
    - dot-count matching (1/2/3),
    - global position ordering for repeated counts,
    - right-to-left reading of the control runs.
    """
    return _solve(grid, reverse_read=True, use_global_order=True)


def _transform_slot(slot: tuple[int, int], mirror_h: bool) -> tuple[int, int]:
    r, c = map(int, slot)
    if bool(mirror_h):
        c = int(2 - c)
    return int(r), int(c)


def _build_frames(
    *,
    outer: int,
    t: int,
    mirror_h: bool,
    top_margin: int,
    left_margin: int,
    row_gaps: list[int],
    col_gaps: list[int],
) -> tuple[list[dict], dict[tuple[int, int], dict]]:
    r0s = [int(top_margin)]
    for gap in row_gaps:
        r0s.append(int(r0s[-1] + outer + int(gap)))
    c0s = [int(left_margin)]
    for gap in col_gaps:
        c0s.append(int(c0s[-1] + outer + int(gap)))

    all_slots = list(_PATH_SLOTS) + [_DECOY_SLOT]
    frames: list[dict] = []
    slot_to_frame: dict[tuple[int, int], dict] = {}
    for slot in all_slots:
        rr, cc = _transform_slot(slot, bool(mirror_h))
        r0 = int(r0s[int(rr)])
        c0 = int(c0s[int(cc)])
        r1 = int(r0 + outer - 1)
        c1 = int(c0 + outer - 1)
        frame = {
            "bbox": (int(r0), int(c0), int(r1), int(c1)),
            "inner_bbox": (int(r0 + t), int(c0 + t), int(r1 - t), int(c1 - t)),
            "t": int(t),
            "slot": (int(rr), int(cc)),
            "logical_slot": tuple(map(int, slot)),
        }
        frames.append(frame)
        slot_to_frame[(int(slot[0]), int(slot[1]))] = frame
    frames.sort(key=lambda f: _frame_order_key(f))
    return frames, slot_to_frame


def _gen_attempt() -> np.ndarray | None:
    bg = int(np.random.choice(np.array(_DIGITS, dtype=int)))
    border = int(np.random.choice(np.array([d for d in _DIGITS if int(d) != int(bg)], dtype=int)))
    ctrl_choices = [d for d in _DIGITS if int(d) not in (int(bg), int(border))]
    if not ctrl_choices:
        return None
    ctrl_color = int(np.random.choice(np.array(ctrl_choices, dtype=int)))
    paint_palette = [int(d) for d in _DIGITS if int(d) not in (int(bg), int(border), int(ctrl_color))]
    if len(paint_palette) < 6:
        return None

    t = int(np.random.choice(np.array([1, 1, 2], dtype=int)))
    inner = int(np.random.choice(np.array([2, 3], dtype=int)))
    outer = int(inner + 2 * t)
    if not (4 <= int(outer) <= 7):
        return None

    mirror_h = bool(int(np.random.randint(0, 2)))
    row_gaps = [1, int(np.random.randint(1, 4))]
    col_gaps = [int(np.random.randint(2, 5)), int(np.random.randint(1, 4))]
    special_a = _transform_slot(_PATH_SLOTS[0], bool(mirror_h))
    special_b = _transform_slot(_PATH_SLOTS[1], bool(mirror_h))
    special_col = int(min(int(special_a[1]), int(special_b[1])))
    col_gaps[int(special_col)] = int(np.random.randint(3, 6))

    top_margin = int(np.random.randint(1, 3))
    left_margin = int(np.random.randint(1, 3))
    right_margin = int(np.random.randint(1, 4))

    frames, slot_to_frame = _build_frames(
        outer=int(outer),
        t=int(t),
        mirror_h=bool(mirror_h),
        top_margin=int(top_margin),
        left_margin=int(left_margin),
        row_gaps=[int(x) for x in row_gaps],
        col_gaps=[int(x) for x in col_gaps],
    )

    count_perm = [int(x) for x in np.random.permutation(np.array([1, 2, 3], dtype=int))]
    repeated_count, decoy_count, unique_count = map(int, count_perm)
    count_by_logical_slot = {
        (0, 0): int(repeated_count),
        (0, 1): int(decoy_count),
        (1, 1): int(repeated_count),
        (1, 2): int(unique_count),
        (2, 2): int(repeated_count),
        (1, 0): int(decoy_count),
    }

    paints = [int(x) for x in np.random.permutation(np.array(paint_palette, dtype=int))[:6]]
    paint_by_logical_slot = {
        logical_slot: int(paint)
        for logical_slot, paint in zip(list(_PATH_SLOTS) + [_DECOY_SLOT], paints)
    }

    max_r1 = max(int(frame["bbox"][2]) for frame in frames)
    max_c1 = max(int(frame["bbox"][3]) for frame in frames)
    ctrl_gap = int(np.random.randint(2, 5))
    ctrl_r = int(max_r1 + ctrl_gap)
    bottom_margin = 1
    path_counts = [int(count_by_logical_slot[slot]) for slot in _PATH_SLOTS]
    ctrl_ltr_counts = list(reversed(path_counts))
    ctrl_needed_w = int(1 + sum(ctrl_ltr_counts) + len(ctrl_ltr_counts))
    H = int(ctrl_r + 1 + bottom_margin)
    W = int(max(max_c1 + 1 + right_margin, ctrl_needed_w + 1))
    if not (7 <= int(H) <= 30 and 7 <= int(W) <= 30):
        return None

    out = np.full((int(H), int(W)), int(bg), dtype=int)
    for frame in frames:
        logical_slot = tuple(map(int, frame["logical_slot"]))
        r0, c0, r1, _c1 = map(int, frame["bbox"])
        _draw_frame(
            out,
            r0=int(r0),
            c0=int(c0),
            outer=int(r1 - r0 + 1),
            t=int(frame["t"]),
            bg=int(bg),
            border=int(border),
            paint=int(paint_by_logical_slot[logical_slot]),
            count=int(count_by_logical_slot[logical_slot]),
        )

    bridge_cells_all: set[tuple[int, int]] = set()
    for slot_a, slot_b in zip(_PATH_SLOTS, _PATH_SLOTS[1:]):
        frame_a = slot_to_frame[tuple(map(int, slot_a))]
        frame_b = slot_to_frame[tuple(map(int, slot_b))]
        cells = _bridge_cells(frame_a, frame_b)
        if not cells:
            return None
        if cells & bridge_cells_all:
            return None
        for r, c in cells:
            if not (0 <= int(r) < int(H) and 0 <= int(c) < int(W)):
                return None
            if int(out[int(r), int(c)]) != int(bg):
                return None
        paint = int(paint_by_logical_slot[tuple(map(int, slot_a))])
        for r, c in cells:
            out[int(r), int(c)] = int(paint)
        bridge_cells_all |= set(cells)

    c = 1
    for count in ctrl_ltr_counts:
        if int(c + count) > int(W):
            return None
        out[int(ctrl_r), int(c) : int(c + count)] = int(ctrl_color)
        c += int(count + 1)

    inp = out.copy()
    for r, c in bridge_cells_all:
        inp[int(r), int(c)] = int(bg)

    if int(get_bg(inp)) != int(bg):
        return None
    if int(_most_common_nonbg(inp, int(bg)) or -1) != int(border):
        return None

    try:
        pred = main(inp)
        ltr_pred = _solve(inp, reverse_read=False, use_global_order=True)
        local_pred = _solve(inp, reverse_read=True, use_global_order=False)
    except Exception:
        return None

    if pred.shape != out.shape:
        return None
    if not bool(np.array_equal(pred, out)):
        return None
    if bool(np.array_equal(ltr_pred, out)):
        return None
    if bool(np.array_equal(local_pred, out)):
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(800):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("Failed to generate valid input after 800 attempts")
