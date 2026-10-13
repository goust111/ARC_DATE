from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


Coord = tuple[int, int]

BG = 0
YELLOW = 4
GEN_ATTEMPTS = 400
SIDE_NAMES = ("top", "right", "bottom", "left")


def _neighbors4(r: int, c: int) -> list[Coord]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _label_components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells: list[Coord] = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _parse_frame(grid: np.ndarray, border_color: int, coords: np.ndarray, bg: int) -> dict | None:
    r0 = int(coords[:, 0].min())
    c0 = int(coords[:, 1].min())
    r1 = int(coords[:, 0].max())
    c1 = int(coords[:, 1].max())
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h != w or h < 3:
        return None

    patch = grid[r0 : r1 + 1, c0 : c1 + 1]
    count = int(np.sum(patch == border_color))
    for thickness in range(1, h // 2 + 1):
        inner = patch[thickness : h - thickness, thickness : w - thickness]
        if inner.size == 0:
            continue
        if np.any(inner == border_color):
            continue
        inner_colors = {int(v) for v in inner.ravel()}
        if len(inner_colors) != 1:
            continue
        inner_color = next(iter(inner_colors))
        if inner_color in (border_color, bg):
            continue
        border_area = h * w - inner.shape[0] * inner.shape[1]
        if count != border_area:
            continue
        if not np.all(patch[:thickness, :] == border_color):
            continue
        if not np.all(patch[h - thickness :, :] == border_color):
            continue
        if not np.all(patch[:, :thickness] == border_color):
            continue
        if not np.all(patch[:, w - thickness :] == border_color):
            continue
        return {
            "bbox": (r0, c0, r1, c1),
            "outer_size": h,
            "thickness": thickness,
            "inner_color": inner_color,
            "inner_bbox": (r0 + thickness, c0 + thickness, r1 - thickness, c1 - thickness),
        }
    return None


def _detect_frames(grid: np.ndarray) -> tuple[int, list[dict]]:
    bg = _get_bg(grid)
    best: tuple[int, tuple[int, int], list[dict]] | None = None
    for color in sorted({int(v) for v in grid.ravel()} - {bg}):
        parsed: list[dict] = []
        for comp in _label_components(grid == color):
            frame = _parse_frame(grid, color, comp, bg)
            if frame is not None:
                parsed.append(frame)
        if not parsed:
            continue

        groups: dict[tuple[int, int], list[dict]] = {}
        for frame in parsed:
            key = (int(frame["outer_size"]), int(frame["thickness"]))
            groups.setdefault(key, []).append(frame)
        for key, frames in groups.items():
            inner_colors = [int(frame["inner_color"]) for frame in frames]
            if len(inner_colors) != len(set(inner_colors)):
                continue
            cand = (len(frames), color, key, frames)
            if best is None or cand[:2] > (len(best[2]), best[0]):
                best = (color, key, frames)

    assert best is not None, "expected framed squares"
    border_color, _key, frames = best
    assert len(frames) >= 4, "expected at least four framed squares"
    return int(border_color), frames


def _control_row(grid: np.ndarray, frames: list[dict], bg: int, border_color: int) -> tuple[int, list[int]]:
    max_r = max(int(frame["bbox"][2]) for frame in frames)
    ctrl_r = None
    for r in range(grid.shape[0] - 1, max_r + 1, -1):
        colors = [int(v) for v in grid[r] if int(v) != bg]
        if len(colors) >= 2:
            ctrl_r = r
            break
    assert ctrl_r is not None, "expected control row"
    if ctrl_r - 1 > max_r:
        assert np.all(grid[ctrl_r - 1] == bg), "expected a blank separator row"

    seq = [int(v) for v in grid[ctrl_r] if int(v) != bg]
    assert len(seq) >= 4, "expected loop control sequence"
    assert border_color not in seq and bg not in seq and YELLOW not in seq
    assert len(seq) == len(set(seq)), "control colors should be unique"
    return ctrl_r, seq


def _same_row_band(a: dict, b: dict) -> bool:
    ia = a["inner_bbox"]
    ib = b["inner_bbox"]
    return int(ia[0]) == int(ib[0]) and int(ia[2]) == int(ib[2])


def _same_col_band(a: dict, b: dict) -> bool:
    ia = a["inner_bbox"]
    ib = b["inner_bbox"]
    return int(ia[1]) == int(ib[1]) and int(ia[3]) == int(ib[3])


def _draw_bridge(out: np.ndarray, a: dict, b: dict, color: int) -> None:
    if _same_row_band(a, b):
        r0, _c0, r1, _c1 = map(int, a["inner_bbox"])
        a_left = int(a["bbox"][1])
        a_right = int(a["bbox"][3])
        b_left = int(b["bbox"][1])
        b_right = int(b["bbox"][3])
        if a_right < b_left:
            c_start, c_end = a_right + 1, b_left - 1
        else:
            c_start, c_end = b_right + 1, a_left - 1
        if c_start <= c_end:
            out[r0 : r1 + 1, c_start : c_end + 1] = color
        return

    if _same_col_band(a, b):
        r0a, _c0a, r1a, _c1a = map(int, a["bbox"])
        r0b, _c0b, r1b, _c1b = map(int, b["bbox"])
        _r0, c0, _r1, c1 = map(int, a["inner_bbox"])
        if r1a < r0b:
            r_start, r_end = r1a + 1, r0b - 1
        else:
            r_start, r_end = r1b + 1, r0a - 1
        if r_start <= r_end:
            out[r_start : r_end + 1, c0 : c1 + 1] = color
        return

    raise AssertionError("consecutive loop squares must share a row or column band")


def _fill_enclosed_background(grid: np.ndarray, bg: int) -> np.ndarray:
    out = grid.copy()
    h, w = out.shape
    seen = np.zeros((h, w), dtype=bool)
    q: deque[Coord] = deque()

    for r in range(h):
        for c in (0, w - 1):
            if out[r, c] == bg and not seen[r, c]:
                seen[r, c] = True
                q.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if out[r, c] == bg and not seen[r, c]:
                seen[r, c] = True
                q.append((r, c))

    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and out[nr, nc] == bg and not seen[nr, nc]:
                seen[nr, nc] = True
                q.append((nr, nc))

    enclosed = (out == bg) & ~seen
    assert np.any(enclosed), "expected a non-empty enclosed background region"
    out[enclosed] = YELLOW
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect the framed squares in the control-row order, close the hidden final bridge, and fill the enclosed background yellow.

    Input structure:
    - The background is black, and the main field contains several equal-sized framed squares sharing one frame color.
    - Each frame encloses a unique solid inner color, while a separated bottom control row lists some inner colors left-to-right.
    - Consecutive listed squares, including the final listed square back to the first, always share a row band or a column band.

    Rule:
    - Match each control color to the framed square with that inner color.
    - Draw a straight bridge in the frame color between every consecutive control pair, with bridge width equal to the squares' inner width.
    - Also bridge the last listed square back to the first to close the loop, then recolor every background cell unreachable from the grid boundary by 4-connectivity to yellow.

    Key insight:
    - The control row names a cyclic tour rather than an open path, so the missing closing bridge must be inferred before the enclosed region can be flood-filled.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = _get_bg(g)
    border_color, frames = _detect_frames(g)
    _ctrl_r, seq = _control_row(g, frames, bg, border_color)
    by_inner = {int(frame["inner_color"]): frame for frame in frames}
    assert len(by_inner) == len(frames)
    assert all(color in by_inner for color in seq)

    loop_frames = [by_inner[color] for color in seq]
    for a, b in zip(loop_frames, loop_frames[1:] + loop_frames[:1]):
        assert _same_row_band(a, b) or _same_col_band(a, b)

    out = g.copy()
    for a, b in zip(loop_frames, loop_frames[1:] + loop_frames[:1]):
        _draw_bridge(out, a, b, border_color)
    return _fill_enclosed_background(out, bg)


def _sample_partition(parts: int, total: int, low: int, high: int) -> list[int] | None:
    if parts * low > total or parts * high < total:
        return None
    values = []
    remaining = total
    for i in range(parts):
        left = parts - i - 1
        lo = max(low, remaining - left * high)
        hi = min(high, remaining - left * low)
        if lo > hi:
            return None
        value = random.randint(lo, hi)
        values.append(value)
        remaining -= value
    return values if remaining == 0 else None


def _cycle_positions(outer: int, top_extra: int, right_extra: int, bottom_extra: int, left_extra: int) -> list[tuple[str, int, int]]:
    gap_low = outer + 1
    gap_high = outer + 3

    top_segments = top_extra + 1
    bottom_segments = bottom_extra + 1
    right_segments = right_extra + 1
    left_segments = left_extra + 1

    x_low = max(top_segments * gap_low, bottom_segments * gap_low)
    x_high = min(top_segments * gap_high, bottom_segments * gap_high)
    y_low = max(right_segments * gap_low, left_segments * gap_low)
    y_high = min(right_segments * gap_high, left_segments * gap_high)
    if x_low > x_high or y_low > y_high:
        raise ValueError("no compatible cycle span")

    span_x = random.randint(x_low, x_high)
    span_y = random.randint(y_low, y_high)
    top_parts = _sample_partition(top_segments, span_x, gap_low, gap_high)
    bottom_parts = _sample_partition(bottom_segments, span_x, gap_low, gap_high)
    right_parts = _sample_partition(right_segments, span_y, gap_low, gap_high)
    left_parts = _sample_partition(left_segments, span_y, gap_low, gap_high)
    if any(parts is None for parts in (top_parts, bottom_parts, right_parts, left_parts)):
        raise ValueError("partition failure")

    pos: list[tuple[str, int, int]] = []

    c = 0
    pos.append(("top", 0, 0))
    for step in top_parts:
        c += step
        side = "top"
        pos.append((side, 0, c))

    r = 0
    for step in right_parts:
        r += step
        pos.append(("right", r, span_x))

    c = span_x
    for step in bottom_parts:
        c -= step
        pos.append(("bottom", span_y, c))

    r = span_y
    for i, step in enumerate(left_parts):
        r -= step
        if i < len(left_parts) - 1:
            pos.append(("left", r, 0))

    return pos


def _frame_sprite(outer: int, thickness: int, border_color: int, inner_color: int) -> np.ndarray:
    sprite = np.full((outer, outer), border_color, dtype=int)
    sprite[thickness : outer - thickness, thickness : outer - thickness] = inner_color
    return sprite


def _rects_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int], padding: int = 0) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (
        ar1 + padding < br0
        or br1 + padding < ar0
        or ac1 + padding < bc0
        or bc1 + padding < ac0
    )


def _candidate_distractors(
    occupied: list[tuple[int, int, int, int]],
    cycle_rect: tuple[int, int, int, int],
    aligned_rows: set[int],
    aligned_cols: set[int],
    outer: int,
    h: int,
    w: int,
    ctrl_r: int,
) -> list[tuple[int, int]]:
    candidates: list[tuple[int, int]] = []
    for r in range(0, ctrl_r - outer):
        for c in range(0, w - outer + 1):
            rect = (r, c, r + outer - 1, c + outer - 1)
            if r not in aligned_rows and c not in aligned_cols:
                continue
            if _rects_overlap(rect, cycle_rect, padding=0):
                continue
            if any(_rects_overlap(rect, other, padding=1) for other in occupied):
                continue
            candidates.append((r, c))
    return candidates


def _control_layout(width: int, count: int) -> list[int] | None:
    min_gap_sum = count - 1
    max_gap_sum = min(2 * (count - 1), width - count)
    if max_gap_sum < min_gap_sum:
        return None
    gap_sum = random.randint(min_gap_sum, max_gap_sum)
    extras = gap_sum - min_gap_sum
    gaps = [1] * (count - 1)
    for idx in random.sample(range(count - 1), extras):
        gaps[idx] += 1
    span = count + gap_sum
    start = random.randint(0, width - span)
    cols = [start]
    cur = start
    for gap in gaps:
        cur += gap + 1
        cols.append(cur)
    return cols


def _render_grid(
    outer: int,
    thickness: int,
    border_color: int,
    cycle_positions: list[tuple[str, int, int]],
    distractor_positions: list[tuple[int, int]],
    all_colors: list[int],
    final_h: int,
    final_w: int,
    top_offset: int,
    left_offset: int,
) -> np.ndarray:
    grid = np.full((final_h, final_w), BG, dtype=int)
    loop_inner_colors = all_colors[: len(cycle_positions)]
    distractor_colors = all_colors[len(cycle_positions) :]

    cycle_rows = []
    cycle_cols = []
    for (_, lr, lc), inner_color in zip(cycle_positions, loop_inner_colors):
        r = top_offset + lr
        c = left_offset + lc
        grid[r : r + outer, c : c + outer] = _frame_sprite(outer, thickness, border_color, inner_color)
        cycle_rows.append(r)
        cycle_cols.append(c)

    for (r, c), inner_color in zip(distractor_positions, distractor_colors):
        grid[r : r + outer, c : c + outer] = _frame_sprite(outer, thickness, border_color, inner_color)

    seq = loop_inner_colors[:]
    if random.random() < 0.5:
        seq.reverse()
    shift = random.randrange(len(seq))
    seq = seq[shift:] + seq[:shift]

    ctrl_r = final_h - 1
    ctrl_cols = _control_layout(final_w, len(seq))
    assert ctrl_cols is not None
    for c, color in zip(ctrl_cols, seq):
        grid[ctrl_r, c] = color

    assert np.all(grid[ctrl_r - 1] == BG)
    return grid


def _gen_attempt() -> np.ndarray | None:
    outer = random.randint(3, 5)
    thickness = random.choice([1] if outer == 3 else [1, 2] if outer >= 5 else [1])
    if outer - 2 * thickness < 1:
        return None

    extra_budget = random.randint(0, 2)
    extras = [0, 0, 0, 0]
    for idx in random.sample(range(4), extra_budget):
        extras[idx] = random.randint(1, 2 if extra_budget == 1 else 1)
    top_extra, right_extra, bottom_extra, left_extra = extras
    cycle_count = 4 + sum(extras)
    distractor_count = 2 if cycle_count <= 5 and random.random() < 0.5 else 1
    total_squares = cycle_count + distractor_count

    usable_colors = [1, 2, 3, 5, 6, 7, 8, 9]
    border_color = random.choice(usable_colors)
    inner_pool = [color for color in usable_colors if color != border_color]
    if total_squares > len(inner_pool):
        return None
    all_colors = random.sample(inner_pool, total_squares)

    try:
        cycle_positions = _cycle_positions(outer, top_extra, right_extra, bottom_extra, left_extra)
    except ValueError:
        return None

    local_h = max(r for _, r, _ in cycle_positions) + outer
    local_w = max(c for _, _, c in cycle_positions) + outer
    if local_h > 24 or local_w > 24:
        return None

    min_h = local_h + 2
    min_w = max(local_w, cycle_count * 2 - 1)
    max_h = min(30, local_h + 8)
    max_w = min(30, local_w + 8)
    if min_h > max_h or min_w > max_w:
        return None

    final_h = random.randint(min_h, max_h)
    final_w = random.randint(min_w, max_w)
    top_offset = random.randint(0, final_h - local_h - 2)
    left_offset = random.randint(0, final_w - local_w)
    ctrl_r = final_h - 1

    occupied: list[tuple[int, int, int, int]] = []
    cycle_rows: set[int] = set()
    cycle_cols: set[int] = set()
    for _, lr, lc in cycle_positions:
        r = top_offset + lr
        c = left_offset + lc
        occupied.append((r, c, r + outer - 1, c + outer - 1))
        cycle_rows.add(r)
        cycle_cols.add(c)
    cycle_rect = (
        min(rect[0] for rect in occupied),
        min(rect[1] for rect in occupied),
        max(rect[2] for rect in occupied),
        max(rect[3] for rect in occupied),
    )

    distractor_positions: list[tuple[int, int]] = []
    for _ in range(distractor_count):
        candidates = _candidate_distractors(
            occupied,
            cycle_rect,
            cycle_rows,
            cycle_cols,
            outer,
            final_h,
            final_w,
            ctrl_r,
        )
        if not candidates:
            return None
        choice = random.choice(candidates)
        distractor_positions.append(choice)
        occupied.append((choice[0], choice[1], choice[0] + outer - 1, choice[1] + outer - 1))

    grid = _render_grid(
        outer,
        thickness,
        border_color,
        cycle_positions,
        distractor_positions,
        all_colors,
        final_h,
        final_w,
        top_offset,
        left_offset,
    )

    try:
        out = main(grid)
    except AssertionError:
        return None

    if out.shape != grid.shape or max(out.shape) > 30:
        return None
    if not np.any(out == YELLOW):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_ATTEMPTS} attempts")
