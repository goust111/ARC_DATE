from __future__ import annotations

import random

import numpy as np

Coord = tuple[int, int]

BG = 0
RED = 2
GREEN = 3
INPUT_COLORS = [1, 4, 5, 6, 7, 8, 9]

MIN_SEGMENTS = 3
MAX_SEGMENTS = 6
MIN_GRID = 12
MAX_GRID = 24
MIN_GAP = 2
MAX_GAP = 4
MIN_DELTA = 2
MAX_DELTA = 6
GEN_ATTEMPTS = 400


def _components(mask: np.ndarray) -> list[list[Coord]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[Coord]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[Coord] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _straight_component_info(comp: list[Coord]) -> tuple[str, Coord, Coord] | None:
    rows = sorted({r for r, _ in comp})
    cols = sorted({c for _, c in comp})
    if len(rows) == 1:
        row = rows[0]
        if cols != list(range(cols[0], cols[-1] + 1)):
            return None
        return "h", (row, cols[0]), (row, cols[-1])
    if len(cols) == 1:
        col = cols[0]
        if rows != list(range(rows[0], rows[-1] + 1)):
            return None
        return "v", (rows[0], col), (rows[-1], col)
    return None


def _step_toward(start: Coord, end: Coord) -> Coord:
    dr = end[0] - start[0]
    dc = end[1] - start[1]
    if dr:
        return (1 if dr > 0 else -1, 0)
    if dc:
        return (0, 1 if dc > 0 else -1)
    raise AssertionError("expected distinct coordinates")


def _turn_sign(a: Coord, b: Coord) -> int:
    cross = a[0] * b[1] - a[1] * b[0]
    if cross > 0:
        return 1
    if cross < 0:
        return -1
    return 0


def _extract_segments(grid: np.ndarray) -> dict[int, dict[str, object]]:
    h, w = grid.shape
    segments: dict[int, dict[str, object]] = {}
    for color in sorted(int(v) for v in np.unique(grid) if int(v) != BG):
        comps = _components(grid == color)
        straight = []
        for comp in comps:
            info = _straight_component_info(comp)
            if info is not None and len(comp) >= 2:
                straight.append((comp, info))
        if not straight:
            continue
        assert len(straight) == 1, f"color {color} should have exactly one segment body"

        _comp, (orientation, lo, hi) = straight[0]
        if orientation == "h":
            r = lo[0]
            left = (r, lo[1] - 1) if lo[1] > 0 else None
            right = (r, hi[1] + 1) if hi[1] + 1 < w else None
            left_marked = left is not None and int(grid[left]) != BG
            right_marked = right is not None and int(grid[right]) != BG
            assert left_marked ^ right_marked, f"horizontal segment color {color} needs one marked side"
            if left_marked:
                marker = left
                plain = hi
            else:
                marker = right
                plain = lo
        else:
            c = lo[1]
            top = (lo[0] - 1, c) if lo[0] > 0 else None
            bottom = (hi[0] + 1, c) if hi[0] + 1 < h else None
            top_marked = top is not None and int(grid[top]) != BG
            bottom_marked = bottom is not None and int(grid[bottom]) != BG
            assert top_marked ^ bottom_marked, f"vertical segment color {color} needs one marked side"
            if top_marked:
                marker = top
                plain = hi
            else:
                marker = bottom
                plain = lo

        direction = _step_toward(plain, marker)
        segments[color] = {
            "color": color,
            "orientation": orientation,
            "plain": plain,
            "marker": marker,
            "direction": direction,
            "marker_color": int(grid[marker]),
        }
    assert segments, "expected at least one segment"
    return segments


def _paint_connector(out: np.ndarray, current: dict[str, object], nxt: dict[str, object]) -> None:
    cur_marker = current["marker"]
    nxt_plain = nxt["plain"]
    cur_dir = current["direction"]
    nxt_dir = nxt["direction"]
    orientation = current["orientation"]

    assert orientation == nxt["orientation"], "consecutive segments must be parallel"
    assert isinstance(cur_marker, tuple) and isinstance(nxt_plain, tuple)
    assert isinstance(cur_dir, tuple) and isinstance(nxt_dir, tuple)

    bridge_step = _step_toward(cur_marker, nxt_plain)
    assert _turn_sign(cur_dir, bridge_step) != 0
    assert _turn_sign(bridge_step, nxt_dir) != 0
    sign1 = _turn_sign(cur_dir, bridge_step)
    sign2 = _turn_sign(bridge_step, nxt_dir)
    assert sign1 == sign2, "connector should encode one turn direction"
    bridge_color = GREEN if sign1 > 0 else RED

    if orientation == "h":
        assert cur_marker[1] == nxt_plain[1], "horizontal segments must align by connector column"
        col = cur_marker[1]
        step = 1 if nxt_plain[0] > cur_marker[0] else -1
        for row in range(cur_marker[0] + step, nxt_plain[0], step):
            assert int(out[row, col]) == BG
            out[row, col] = bridge_color
        return

    assert orientation == "v"
    assert cur_marker[0] == nxt_plain[0], "vertical segments must align by connector row"
    row = cur_marker[0]
    step = 1 if nxt_plain[1] > cur_marker[1] else -1
    for col in range(cur_marker[1] + step, nxt_plain[1], step):
        assert int(out[row, col]) == BG
        out[row, col] = bridge_color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Chain the scattered segments by their endpoint markers, then color each
    perpendicular bridge by the turn direction of the resulting polyline.
    """
    g = np.asarray(grid, dtype=int)
    segments = _extract_segments(g)

    incoming = {color: 0 for color in segments}
    for seg in segments.values():
        marker_color = int(seg["marker_color"])
        if marker_color in segments:
            incoming[marker_color] += 1

    starts = [color for color, count in incoming.items() if count == 0]
    assert len(starts) == 1, "expected one segment with no incoming marker"

    out = g.copy()
    visited: set[int] = set()
    current_color = starts[0]

    while current_color not in visited:
        visited.add(current_color)
        current = segments[current_color]
        next_color = int(current["marker_color"])
        if next_color not in segments or next_color in visited:
            break
        nxt = segments[next_color]
        _paint_connector(out, current, nxt)
        current_color = next_color

    return out


def _sample_positions(count: int, limit: int) -> list[int] | None:
    if count == 1:
        if limit < 3:
            return None
        return [random.randint(1, limit - 2)]

    gaps = [random.randint(MIN_GAP, MAX_GAP) for _ in range(count - 1)]
    span = sum(gaps)
    lo = 1
    hi = limit - 2
    if hi - lo < span:
        return None
    start = random.randint(lo, hi - span)
    pos = [start]
    for gap in gaps:
        pos.append(pos[-1] + gap)
    return pos


def _sample_alternating_path(n_segments: int, limit: int, first_sign: int) -> list[int] | None:
    lo = 1
    hi = limit - 2
    if hi - lo < MIN_DELTA:
        return None
    for _ in range(80):
        vals = [random.randint(lo, hi)]
        sign = first_sign
        ok = True
        for _idx in range(n_segments):
            choices = []
            upper = min(MAX_DELTA, hi - lo)
            for delta in range(MIN_DELTA, upper + 1):
                nxt = vals[-1] + sign * delta
                if lo <= nxt <= hi:
                    choices.append(nxt)
            if not choices:
                ok = False
                break
            vals.append(random.choice(choices))
            sign *= -1
        if ok and len(set(vals)) >= 3:
            return vals
    return None


def _render_horizontal(
    height: int,
    width: int,
    body_colors: list[int],
    terminal_color: int,
    rows: list[int],
    cols: list[int],
) -> np.ndarray:
    grid = np.full((height, width), BG, dtype=int)
    n_segments = len(body_colors)
    for idx in range(n_segments):
        color = body_colors[idx]
        row = rows[idx]
        c0 = cols[idx]
        c1 = cols[idx + 1]
        grid[row, min(c0, c1) : max(c0, c1) + 1] = color
        grid[row, c1] = body_colors[idx + 1] if idx + 1 < n_segments else terminal_color
    return grid


def _render_vertical(
    height: int,
    width: int,
    body_colors: list[int],
    terminal_color: int,
    cols: list[int],
    rows: list[int],
) -> np.ndarray:
    grid = np.full((height, width), BG, dtype=int)
    n_segments = len(body_colors)
    for idx in range(n_segments):
        color = body_colors[idx]
        col = cols[idx]
        r0 = rows[idx]
        r1 = rows[idx + 1]
        grid[min(r0, r1) : max(r0, r1) + 1, col] = color
        grid[r1, col] = body_colors[idx + 1] if idx + 1 < n_segments else terminal_color
    return grid


def _gen_attempt() -> np.ndarray | None:
    n_segments = random.randint(MIN_SEGMENTS, MAX_SEGMENTS)
    orientation = random.choice(("h", "v"))
    height = random.randint(max(MIN_GRID, 2 * n_segments + 4), MAX_GRID)
    width = random.randint(max(MIN_GRID, 2 * n_segments + 4), MAX_GRID)
    colors = random.sample(INPUT_COLORS, n_segments + 1)
    body_colors = colors[:n_segments]
    terminal_color = colors[-1]

    if orientation == "h":
        rows_increasing = _sample_positions(n_segments, height)
        if rows_increasing is None:
            return None
        connector_down = random.choice((False, True))
        rows = rows_increasing if connector_down else list(reversed(rows_increasing))
        cols = _sample_alternating_path(n_segments, width, random.choice((-1, 1)))
        if cols is None:
            return None
        grid = _render_horizontal(height, width, body_colors, terminal_color, rows, cols)
    else:
        cols_increasing = _sample_positions(n_segments, width)
        if cols_increasing is None:
            return None
        connector_right = random.choice((False, True))
        cols = cols_increasing if connector_right else list(reversed(cols_increasing))
        rows = _sample_alternating_path(n_segments, height, random.choice((-1, 1)))
        if rows is None:
            return None
        grid = _render_vertical(height, width, body_colors, terminal_color, cols, rows)

    try:
        out = main(grid)
    except AssertionError:
        return None

    if out.shape != grid.shape:
        return None
    if not np.array_equal(out[grid != BG], grid[grid != BG]):
        return None
    if not bool(np.any(out == RED)) or not bool(np.any(out == GREEN)):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input failed after {GEN_ATTEMPTS} attempts")
