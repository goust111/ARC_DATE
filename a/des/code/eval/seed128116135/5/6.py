import random
from dataclasses import dataclass

import numpy as np


BG = 0
RED = 2
GRAY = 5
ORANGE = 7
RECT_COLORS = (1, 3, 4, 6, 8, 9)
GEN_RETRIES = 400
PLACE_RETRIES = 200
MIN_LINE_LEN = 4


@dataclass(frozen=True)
class Rect:
    r0: int
    c0: int
    r1: int
    c1: int
    color: int


@dataclass(frozen=True)
class Line:
    orient: str
    fixed: int
    start: int
    end: int
    branch: int
    crossed_colors: tuple[int, ...]

    @property
    def endpoints(self) -> tuple[tuple[int, int], tuple[int, int]]:
        if self.orient == "h":
            return ((self.fixed, self.start), (self.fixed, self.end))
        return ((self.start, self.fixed), (self.end, self.fixed))


def _line_cells(line: Line) -> list[tuple[int, int]]:
    if line.orient == "h":
        return [(line.fixed, c) for c in range(line.start, line.end + 1)]
    return [(r, line.fixed) for r in range(line.start, line.end + 1)]


def _find_hidden_lines(grid: np.ndarray) -> list[Line]:
    red_pts = [tuple(map(int, pt)) for pt in np.argwhere(grid == RED)]
    row_to_cols: dict[int, list[int]] = {}
    col_to_rows: dict[int, list[int]] = {}
    for r, c in red_pts:
        row_to_cols.setdefault(r, []).append(c)
        col_to_rows.setdefault(c, []).append(r)
    for cols in row_to_cols.values():
        cols.sort()
    for rows in col_to_rows.values():
        rows.sort()

    lines: list[Line] = []
    seen_rows: set[int] = set()
    seen_cols: set[int] = set()
    used_pts: set[tuple[int, int]] = set()
    for r, cols in row_to_cols.items():
        if len(cols) != 2:
            continue
        ok = True
        for c in cols:
            if len(col_to_rows.get(c, [])) != 1:
                ok = False
                break
        if ok:
            lines.append(Line("h", r, cols[0], cols[1], -1, ()))
            seen_rows.add(r)
            used_pts.add((r, cols[0]))
            used_pts.add((r, cols[1]))
    for c, rows in col_to_rows.items():
        if len(rows) != 2:
            continue
        ok = True
        for r in rows:
            if len(row_to_cols.get(r, [])) != 1:
                ok = False
                break
        if ok:
            lines.append(Line("v", c, rows[0], rows[1], -1, ()))
            seen_cols.add(c)
            used_pts.add((rows[0], c))
            used_pts.add((rows[1], c))

    assert used_pts == set(red_pts), "every red marker must belong to exactly one line"
    assert len(seen_rows) + len(seen_cols) == len(lines)
    return lines


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover horizontal/vertical hidden lines from red marker pairs.

    The drawing color depends on how many distinct colored rectangles the segment
    overlaps: 0 -> gray, 1 -> that rectangle color, 2+ -> orange.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    for raw_line in _find_hidden_lines(g):
        colors = sorted(
            int(v)
            for v in np.unique(
                g[raw_line.fixed, raw_line.start : raw_line.end + 1]
                if raw_line.orient == "h"
                else g[raw_line.start : raw_line.end + 1, raw_line.fixed]
            )
            if int(v) not in (BG, RED)
        )
        if len(colors) == 0:
            draw_color = GRAY
        elif len(colors) == 1:
            draw_color = colors[0]
        else:
            draw_color = ORANGE
        for r, c in _line_cells(raw_line):
            out[r, c] = draw_color
    return out


def _rect_mask(rects: list[Rect], h: int, w: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    for rect in rects:
        mask[rect.r0 : rect.r1 + 1, rect.c0 : rect.c1 + 1] = True
    return mask


def _can_place_rect(rects: list[Rect], r0: int, c0: int, r1: int, c1: int) -> bool:
    for rect in rects:
        if not (r1 + 1 < rect.r0 or rect.r1 + 1 < r0 or c1 + 1 < rect.c0 or rect.c1 + 1 < c0):
            return False
    return True


def _place_rect(rects: list[Rect], h: int, w: int, color: int, r_rng: tuple[int, int], c_rng: tuple[int, int],
                hrng: tuple[int, int], wrng: tuple[int, int], rng: random.Random) -> bool:
    r_lo, r_hi = r_rng
    c_lo, c_hi = c_rng
    if r_lo > r_hi or c_lo > c_hi:
        return False
    for _ in range(PLACE_RETRIES):
        rh = rng.randint(hrng[0], hrng[1])
        cw = rng.randint(wrng[0], wrng[1])
        max_r0 = min(r_hi, h - rh - 1)
        max_c0 = min(c_hi, w - cw - 1)
        if max_r0 < r_lo or max_c0 < c_lo:
            continue
        r0 = rng.randint(r_lo, max_r0)
        c0 = rng.randint(c_lo, max_c0)
        r1 = r0 + rh - 1
        c1 = c0 + cw - 1
        if _can_place_rect(rects, r0, c0, r1, c1):
            rects.append(Rect(r0, c0, r1, c1, color))
            return True
    return False


def _build_rects(h: int, w: int, rng: random.Random) -> list[Rect] | None:
    colors = list(RECT_COLORS)
    rng.shuffle(colors)
    rects: list[Rect] = []

    row_band_top = rng.randint(2, max(2, h // 3))
    row_band_height = rng.randint(2, 4)
    row_gap = rng.randint(2, 4)
    left_width = rng.randint(2, 4)
    right_width = rng.randint(2, 4)
    c0 = rng.randint(1, max(1, w // 5))
    c1 = c0 + left_width - 1
    c2 = c1 + row_gap + 1
    c3 = c2 + right_width - 1
    if c3 >= w - 1:
        return None
    rects.append(Rect(row_band_top, c0, row_band_top + row_band_height - 1, c1, colors.pop()))
    rects.append(Rect(row_band_top, c2, row_band_top + row_band_height - 1, c3, colors.pop()))

    col_band_left = rng.randint(max(1, w // 2), max(1, w - 5))
    col_band_width = rng.randint(2, 4)
    col_gap = rng.randint(2, 4)
    top_height = rng.randint(2, 4)
    bottom_height = rng.randint(2, 4)
    r0 = rng.randint(max(row_band_top + row_band_height + 2, 1), max(row_band_top + row_band_height + 2, h // 2))
    r1 = r0 + top_height - 1
    r2 = r1 + col_gap + 1
    r3 = r2 + bottom_height - 1
    if r3 >= h - 1 or col_band_left + col_band_width - 1 >= w - 1:
        return None
    if not _can_place_rect(rects, r0, col_band_left, r1, col_band_left + col_band_width - 1):
        return None
    rects.append(Rect(r0, col_band_left, r1, col_band_left + col_band_width - 1, colors.pop()))
    if not _can_place_rect(rects, r2, col_band_left, r3, col_band_left + col_band_width - 1):
        return None
    rects.append(Rect(r2, col_band_left, r3, col_band_left + col_band_width - 1, colors.pop()))

    extra_count = rng.randint(0, min(2, len(colors)))
    for _ in range(extra_count):
        if not _place_rect(rects, h, w, colors.pop(), (1, h - 4), (1, w - 4), (2, 4), (2, 4), rng):
            break
    return rects


def _segments_from_intervals(intervals: list[tuple[int, int, int]], limit: int, fixed: int, orient: str) -> list[Line]:
    candidates: list[Line] = []

    def gap_before(idx: int) -> tuple[int, int]:
        if idx == 0:
            return (0, intervals[0][0] - 1)
        return (intervals[idx - 1][1] + 1, intervals[idx][0] - 1)

    def gap_after(idx: int) -> tuple[int, int]:
        if idx == len(intervals) - 1:
            return (intervals[idx][1] + 1, limit - 1)
        return (intervals[idx][1] + 1, intervals[idx + 1][0] - 1)

    if not intervals:
        if limit >= MIN_LINE_LEN:
            candidates.append(Line(orient, fixed, 0, limit - 1, 0, ()))
        return candidates

    gaps: list[tuple[int, int]] = []
    prev = 0
    for start, end, _color in intervals:
        gaps.append((prev, start - 1))
        prev = end + 1
    gaps.append((prev, limit - 1))

    for gs, ge in gaps:
        if ge - gs + 1 >= MIN_LINE_LEN:
            candidates.append(Line(orient, fixed, gs, ge, 0, ()))

    for i, (start, end, color) in enumerate(intervals):
        lg0, lg1 = gap_before(i)
        rg0, rg1 = gap_after(i)
        if lg0 <= lg1 and rg0 <= rg1:
            seg_start = lg1
            seg_end = rg0
            if seg_end - seg_start + 1 >= MIN_LINE_LEN:
                candidates.append(Line(orient, fixed, seg_start, seg_end, 1, (color,)))

    for i in range(len(intervals)):
        for j in range(i + 1, len(intervals)):
            lg0, lg1 = gap_before(i)
            rg0, rg1 = gap_after(j)
            if lg0 <= lg1 and rg0 <= rg1:
                seg_start = lg1
                seg_end = rg0
                if seg_end - seg_start + 1 >= MIN_LINE_LEN:
                    colors = tuple(intervals[k][2] for k in range(i, j + 1))
                    candidates.append(Line(orient, fixed, seg_start, seg_end, 2, colors))
    return candidates


def _candidate_lines(rects: list[Rect], h: int, w: int) -> list[Line]:
    candidates: list[Line] = []
    for r in range(h):
        intervals = sorted(
            [(rect.c0, rect.c1, rect.color) for rect in rects if rect.r0 <= r <= rect.r1],
            key=lambda item: item[0],
        )
        candidates.extend(_segments_from_intervals(intervals, w, r, "h"))
    for c in range(w):
        intervals = sorted(
            [(rect.r0, rect.r1, rect.color) for rect in rects if rect.c0 <= c <= rect.c1],
            key=lambda item: item[0],
        )
        candidates.extend(_segments_from_intervals(intervals, h, c, "v"))
    return candidates


def _line_conflicts(line: Line, chosen: list[Line]) -> bool:
    (r1, c1), (r2, c2) = line.endpoints
    new_rows = {r1, r2}
    new_cols = {c1, c2}
    new_cells = set(_line_cells(line))
    for other in chosen:
        (or1, oc1), (or2, oc2) = other.endpoints
        old_rows = {or1, or2}
        old_cols = {oc1, oc2}
        if new_rows & old_rows:
            return True
        if new_cols & old_cols:
            return True
        if new_cells & set(_line_cells(other)):
            return True
    return False


def _select_lines(candidates: list[Line], rng: random.Random) -> list[Line] | None:
    by_branch = {0: [], 1: [], 2: []}
    for line in candidates:
        by_branch[line.branch].append(line)
    for bucket in by_branch.values():
        rng.shuffle(bucket)
    chosen: list[Line] = []
    for branch in (2, 1, 0):
        picked = None
        for line in by_branch[branch]:
            if not _line_conflicts(line, chosen):
                picked = line
                break
        if picked is None:
            return None
        chosen.append(picked)

    extras = [line for line in candidates if line not in chosen]
    rng.shuffle(extras)
    extra_target = rng.randint(0, 1)
    for line in extras:
        if extra_target == 0:
            break
        if _line_conflicts(line, chosen):
            continue
        chosen.append(line)
        extra_target -= 1
    return chosen


def _paint_input(rects: list[Rect], lines: list[Line], h: int, w: int) -> np.ndarray:
    grid = np.full((h, w), BG, dtype=int)
    for rect in rects:
        grid[rect.r0 : rect.r1 + 1, rect.c0 : rect.c1 + 1] = rect.color
    for line in lines:
        for r, c in line.endpoints:
            grid[r, c] = RED
    return grid


def _line_branch_on_grid(grid: np.ndarray, line: Line) -> int:
    if line.orient == "h":
        vals = np.unique(grid[line.fixed, line.start : line.end + 1])
    else:
        vals = np.unique(grid[line.start : line.end + 1, line.fixed])
    colors = [int(v) for v in vals if int(v) not in (BG, RED)]
    return min(2, len(colors))


def _validate_generated(grid: np.ndarray) -> None:
    lines = _find_hidden_lines(grid)
    counts = {0: 0, 1: 0, 2: 0}
    for line in lines:
        counts[_line_branch_on_grid(grid, line)] += 1
    assert all(counts[k] >= 1 for k in (0, 1, 2))

    rect_colors = set(int(v) for v in np.unique(grid) if int(v) not in (BG, RED))
    for color in rect_colors:
        pts = np.argwhere(grid == color)
        r0, c0 = pts.min(axis=0)
        r1, c1 = pts.max(axis=0)
        patch = grid[r0 : r1 + 1, c0 : c1 + 1]
        assert np.all(patch == color), "each non-marker color must be one solid rectangle"


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(16, 24)
    w = rng.randint(16, 24)
    rects = _build_rects(h, w, rng)
    if rects is None:
        return None

    candidates = _candidate_lines(rects, h, w)
    candidates = [line for line in candidates if line.endpoints[0] != line.endpoints[1]]
    if not candidates:
        return None

    lines = _select_lines(candidates, rng)
    if lines is None:
        return None

    grid = _paint_input(rects, lines, h, w)
    try:
        _validate_generated(grid)
    except AssertionError:
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid ARC instance")
