from __future__ import annotations

import random
from collections import defaultdict

import numpy as np


BG = 0
BLUE = 1
RED = 2

GEN_HEIGHT_RANGE = (15, 18)
GEN_WIDTH_RANGE = (15, 18)
GEN_EXTRA_MARKERS = (1, 2)
GEN_RETRY_LIMIT = 500


def _neighbors8(r: int, c: int):
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            yield r + dr, c + dc


def _components8(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for nr, nc in _neighbors8(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _marker_groups(markers: np.ndarray):
    row_to_cols: dict[int, list[int]] = defaultdict(list)
    col_to_rows: dict[int, list[int]] = defaultdict(list)
    for r, c in np.argwhere(markers):
        row_to_cols[int(r)].append(int(c))
        col_to_rows[int(c)].append(int(r))
    for vals in row_to_cols.values():
        vals.sort()
    for vals in col_to_rows.values():
        vals.sort()
    return row_to_cols, col_to_rows


def _clear_horizontal(g: np.ndarray, row: int, c0: int, c1: int) -> bool:
    lo, hi = sorted((int(c0), int(c1)))
    return bool(np.all(g[row, lo : hi + 1] != RED))


def _clear_vertical(g: np.ndarray, col: int, r0: int, r1: int) -> bool:
    lo, hi = sorted((int(r0), int(r1)))
    return bool(np.all(g[lo : hi + 1, col] != RED))


def _paint_horizontal(out: np.ndarray, row: int, c0: int, c1: int):
    lo, hi = sorted((int(c0), int(c1)))
    assert np.all(out[row, lo : hi + 1] != RED)
    out[row, lo : hi + 1] = BLUE


def _paint_vertical(out: np.ndarray, col: int, r0: int, r1: int):
    lo, hi = sorted((int(r0), int(r1)))
    assert np.all(out[lo : hi + 1, col] != RED)
    out[lo : hi + 1, col] = BLUE


def _find_bypass_row(g: np.ndarray, row: int, current_col: int, target_col: int) -> int:
    h, _ = g.shape
    for delta in range(1, h):
        up = row - delta
        if (
            up >= 0
            and _clear_vertical(g, current_col, row, up)
            and _clear_vertical(g, target_col, row, up)
            and _clear_horizontal(g, up, current_col, target_col)
        ):
            return up
        down = row + delta
        if (
            down < h
            and _clear_vertical(g, current_col, row, down)
            and _clear_vertical(g, target_col, row, down)
            and _clear_horizontal(g, down, current_col, target_col)
        ):
            return down
    raise AssertionError("no horizontal detour row available")


def _find_bypass_col(g: np.ndarray, col: int, current_row: int, target_row: int) -> int:
    _, w = g.shape
    for delta in range(1, w):
        left = col - delta
        if (
            left >= 0
            and _clear_horizontal(g, current_row, col, left)
            and _clear_horizontal(g, target_row, col, left)
            and _clear_vertical(g, left, current_row, target_row)
        ):
            return left
        right = col + delta
        if (
            right < w
            and _clear_horizontal(g, current_row, col, right)
            and _clear_horizontal(g, target_row, col, right)
            and _clear_vertical(g, right, current_row, target_row)
        ):
            return right
    raise AssertionError("no vertical detour column available")


def _draw_horizontal_connection(out: np.ndarray, g: np.ndarray, row: int, c0: int, c1: int):
    step = 1 if c1 >= c0 else -1
    current = int(c0)
    while current != int(c1):
        nxt = current + step
        if g[row, nxt] != RED:
            out[row, nxt] = BLUE
            current = nxt
            continue

        bypass_row = _find_bypass_row(g, row, current, c1)
        _paint_vertical(out, current, row, bypass_row)
        _paint_horizontal(out, bypass_row, current, c1)
        _paint_vertical(out, c1, row, bypass_row)
        return


def _draw_vertical_connection(out: np.ndarray, g: np.ndarray, col: int, r0: int, r1: int):
    step = 1 if r1 >= r0 else -1
    current = int(r0)
    while current != int(r1):
        nxt = current + step
        if g[nxt, col] != RED:
            out[nxt, col] = BLUE
            current = nxt
            continue

        bypass_col = _find_bypass_col(g, col, current, r1)
        _paint_horizontal(out, current, col, bypass_col)
        _paint_vertical(out, bypass_col, current, r1)
        _paint_horizontal(out, r1, col, bypass_col)
        return


def _draw_connectors(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    markers = g == BLUE
    row_to_cols, col_to_rows = _marker_groups(markers)

    out = g.copy()
    for row, cols in row_to_cols.items():
        if len(cols) >= 2:
            _draw_horizontal_connection(out, g, row, cols[0], cols[-1])

    for col, rows in col_to_rows.items():
        if len(rows) >= 2:
            _draw_vertical_connection(out, g, col, rows[0], rows[-1])

    return out


def _remove_singletons(grid: np.ndarray) -> np.ndarray:
    out = grid.copy()
    for comp in _components8(out == BLUE):
        if len(comp) == 1:
            r, c = comp[0]
            out[r, c] = BG
    return out


def _transform(grid: np.ndarray, *, cleanup: bool) -> np.ndarray:
    out = _draw_connectors(grid)
    if cleanup:
        out = _remove_singletons(out)
    out[grid == RED] = RED
    return out


def _straight_fill(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    markers = g == BLUE
    row_to_cols, col_to_rows = _marker_groups(markers)
    out = g.copy()
    for row, cols in row_to_cols.items():
        if len(cols) >= 2:
            out[row, cols[0] : cols[-1] + 1] = BLUE
    for col, rows in col_to_rows.items():
        if len(rows) >= 2:
            out[rows[0] : rows[-1] + 1, col] = BLUE
    out[g == RED] = RED
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect aligned blue markers with obstacle-aware spans, then remove isolated blue singletons.

    Input structure:
    - Black (0) is the background.
    - Blue (1) cells are isolated marker pixels that define horizontal and vertical span endpoints.
    - Red (2) cells form one or two horizontal obstacle segments that may block those spans.

    Rule:
    - For every row containing at least two original blue markers, connect the leftmost and rightmost markers in blue.
    - For every column containing at least two original blue markers, connect the topmost and bottommost markers in blue.
    - If a horizontal connector would step into red, detour through the nearest clear bypass row (tie goes upward), then return at the target column; if a vertical connector would step into red, detour through the nearest clear bypass column (tie goes leftward), then return at the target row.
    - After all connectors are drawn, remove every 8-connected blue component of size 1; keep red obstacles unchanged.

    Key insight:
    - The row and column spans are computed from the original markers only, but any blocked span must reroute orthogonally around the obstacle before singleton cleanup is applied.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    return _transform(g, cleanup=True)


def _red_segments(grid: np.ndarray) -> list[tuple[int, int, int]]:
    g = np.asarray(grid, dtype=int)
    segments: list[tuple[int, int, int]] = []
    h, w = g.shape
    for r in range(h):
        c = 0
        while c < w:
            if g[r, c] != RED:
                c += 1
                continue
            c0 = c
            while c < w and g[r, c] == RED:
                c += 1
            segments.append((r, c0, c - 1))
    return segments


def _segment_blocked_pair_count(
    markers: np.ndarray, seg_row: int, seg_c0: int, seg_c1: int
) -> int:
    row_to_cols, col_to_rows = _marker_groups(markers)
    count = 0
    for row, cols in row_to_cols.items():
        if len(cols) >= 2 and row == seg_row:
            if max(cols[0], seg_c0) <= min(cols[-1], seg_c1) and cols[0] < seg_c1 and cols[-1] > seg_c0:
                if seg_c0 <= cols[-1] - 1 and seg_c1 >= cols[0] + 1:
                    count += 1
    for col, rows in col_to_rows.items():
        if len(rows) >= 2 and rows[0] < seg_row < rows[-1] and seg_c0 <= col <= seg_c1:
            count += 1
    return count


def _has_adjacent_markers(markers: np.ndarray) -> bool:
    h, w = markers.shape
    for r, c in np.argwhere(markers):
        for nr, nc in _neighbors8(int(r), int(c)):
            if 0 <= nr < h and 0 <= nc < w and markers[nr, nc] and (nr, nc) != (r, c):
                return True
    return False


def _validate_generated(grid: np.ndarray) -> bool:
    g = np.asarray(grid, dtype=int)
    markers = g == BLUE
    segments = _red_segments(g)
    row_to_cols, col_to_rows = _marker_groups(markers)

    if not (5 <= int(markers.sum()) <= 6):
        return False
    if _has_adjacent_markers(markers):
        return False
    if sum(len(cols) >= 2 for cols in row_to_cols.values()) < 2:
        return False
    if sum(len(rows) >= 2 for rows in col_to_rows.values()) < 2:
        return False
    if any(len(cols) > 2 for cols in row_to_cols.values()):
        return False
    if any(len(rows) > 2 for rows in col_to_rows.values()):
        return False
    if not (1 <= len(segments) <= 2):
        return False
    if any(c1 - c0 + 1 < 3 or c1 - c0 + 1 > 5 for _, c0, c1 in segments):
        return False
    if any(_segment_blocked_pair_count(markers, r, c0, c1) != 1 for r, c0, c1 in segments):
        return False

    try:
        straight = _straight_fill(g)
        preclean = _transform(g, cleanup=False)
        final = _transform(g, cleanup=True)
    except AssertionError:
        return False

    if np.array_equal(preclean, straight):
        return False
    if np.array_equal(final, preclean):
        return False
    if not np.array_equal(final[g == RED], g[g == RED]):
        return False

    removed = {
        (int(r), int(c))
        for r, c in np.argwhere((preclean == BLUE) & (final == BG))
    }
    if not removed:
        return False
    for comp in _components8(preclean == BLUE):
        comp_set = set(comp)
        if comp_set & removed and len(comp) != 1:
            return False

    detour_only = (preclean == BLUE) & (straight != BLUE)
    if not np.any(detour_only):
        return False
    return True


def _pick_layout() -> tuple[int, int, str | None, str | None]:
    h = random.randint(*GEN_HEIGHT_RANGE)
    w = random.randint(*GEN_WIDTH_RANGE)
    options = [
        ("up", None),
        ("down", None),
        (None, "left"),
        (None, "right"),
        ("up", "left"),
        ("up", "right"),
        ("down", "left"),
        ("down", "right"),
    ]
    return h, w, *random.choice(options)


def _vertical_obstacle_starts(c_left: int, c_right: int, width: int, blocked_col: int, length: int):
    starts = []
    for start in range(0, width - length + 1):
        end = start + length - 1
        if start <= blocked_col <= end and not (start <= (c_right if blocked_col == c_left else c_left) <= end):
            starts.append(start)
    return starts


def _choose_extras(
    grid: np.ndarray,
    reserved_rows: set[int],
    reserved_cols: set[int],
    preclean: np.ndarray,
) -> list[tuple[int, int]] | None:
    h, w = grid.shape
    blue_coords = [tuple(map(int, rc)) for rc in np.argwhere(preclean == BLUE)]
    red_coords = [tuple(map(int, rc)) for rc in np.argwhere(grid == RED)]
    candidates: list[tuple[int, int]] = []
    for r in range(h):
        for c in range(w):
            if grid[r, c] != BG or r in reserved_rows or c in reserved_cols:
                continue
            if any(max(abs(r - br), abs(c - bc)) <= 1 for br, bc in blue_coords):
                continue
            if any(max(abs(r - rr), abs(c - rc)) <= 1 for rr, rc in red_coords):
                continue
            candidates.append((r, c))

    random.shuffle(candidates)
    need = random.randint(*GEN_EXTRA_MARKERS)
    extras: list[tuple[int, int]] = []
    used_rows = set(reserved_rows)
    used_cols = set(reserved_cols)
    for r, c in candidates:
        if r in used_rows or c in used_cols:
            continue
        if any(max(abs(r - er), abs(c - ec)) <= 1 for er, ec in extras):
            continue
        extras.append((r, c))
        used_rows.add(r)
        used_cols.add(c)
        if len(extras) == need:
            return extras
    return None


def _gen_attempt() -> np.ndarray | None:
    h, w, h_branch, v_branch = _pick_layout()

    low_rows = list(range(0, min(3, h)))
    high_rows = list(range(max(0, h - 3), h))
    low_cols = list(range(0, min(3, w)))
    high_cols = list(range(max(0, w - 3), w))

    if h_branch == "down":
        blocked_row = 0
        other_row = random.choice([r for r in high_rows if r != blocked_row])
    elif h_branch == "up":
        blocked_row = h - 1
        other_row = random.choice([r for r in low_rows if r != blocked_row])
    else:
        blocked_row = None
        r0 = random.choice(low_rows)
        r1 = random.choice([r for r in high_rows if r != r0])
        other_row = None

    if blocked_row is None:
        r_top, r_bottom = sorted((r0, r1))
    else:
        r_top, r_bottom = sorted((blocked_row, other_row))

    if v_branch == "right":
        blocked_col = 0
        other_col = random.choice([c for c in high_cols if c != blocked_col])
    elif v_branch == "left":
        blocked_col = w - 1
        other_col = random.choice([c for c in low_cols if c != blocked_col])
    else:
        blocked_col = None
        c0 = random.choice(low_cols)
        c1 = random.choice([c for c in high_cols if c != c0])
        other_col = None

    if blocked_col is None:
        c_left, c_right = sorted((c0, c1))
    else:
        c_left, c_right = sorted((blocked_col, other_col))

    if c_right - c_left < 4 or r_bottom - r_top < 4:
        return None

    grid = np.zeros((h, w), dtype=int)
    corners = [(r_top, c_left), (r_top, c_right), (r_bottom, c_left), (r_bottom, c_right)]
    for r, c in corners:
        grid[r, c] = BLUE

    if h_branch is not None:
        row = blocked_row
        max_len = min(5, c_right - c_left - 1)
        if max_len < 3:
            return None
        length = random.randint(3, max_len)
        start = random.randint(c_left + 1, c_right - length)
        grid[row, start : start + length] = RED

    if v_branch is not None:
        col = blocked_col
        max_len = min(5, c_right - c_left)
        if max_len < 3:
            return None
        length = random.randint(3, max_len)
        if h_branch == "down":
            row_candidates = list(range(r_top + 3, r_bottom))
        elif h_branch == "up":
            row_candidates = list(range(r_top + 1, r_bottom - 2))
        else:
            row_candidates = list(range(r_top + 1, r_bottom))
        if not row_candidates:
            return None
        obs_row = random.choice(row_candidates)
        starts = _vertical_obstacle_starts(c_left, c_right, w, col, length)
        if not starts:
            return None
        start = random.choice(starts)
        grid[obs_row, start : start + length] = RED

    try:
        base_preclean = _transform(grid, cleanup=False)
    except AssertionError:
        return None

    extras = _choose_extras(grid, {r_top, r_bottom}, {c_left, c_right}, base_preclean)
    if extras is None:
        return None
    for r, c in extras:
        grid[r, c] = BLUE

    if not _validate_generated(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 4")
