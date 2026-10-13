from __future__ import annotations

from itertools import product
from pathlib import Path
import sys

import numpy as np


_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import Color, find_components, get_bg  # noqa: E402


BG = Color.BLACK
BLUE = Color.BLUE
RED = Color.RED
GREEN = Color.GREEN
TEAL = Color.TEAL

MIN_SIZE = 13
MAX_SIZE_EXCL = 21
MIN_HITS = 3
MAX_HITS_EXCL = 6
MIN_LEN = 3
MAX_LEN = 7
MIN_GAP = 2
MAX_GAP = 4
MAX_BARS = 6
MAX_ATTEMPTS = 500


def _border_clockwise(shape: tuple[int, int]) -> list[tuple[int, int]]:
    h, w = map(int, shape)
    coords = [(0, c) for c in range(w)]
    coords.extend((r, w - 1) for r in range(1, h))
    if h > 1:
        coords.extend((h - 1, c) for c in range(w - 2, -1, -1))
    if w > 1:
        coords.extend((r, 0) for r in range(h - 2, 0, -1))
    return coords


def _primitive_period(seq: list[int] | tuple[int, ...]) -> int:
    n = len(seq)
    for period in range(1, n + 1):
        if n % period == 0 and all(seq[i] == seq[i % period] for i in range(n)):
            return period
    return n


def _primitive_cycle(seq: list[int] | tuple[int, ...]) -> tuple[int, ...]:
    period = _primitive_period(seq)
    return tuple(int(v) for v in seq[:period])


def _extract_marker(grid: np.ndarray) -> tuple[int, int]:
    starts = np.argwhere(grid == RED)
    assert starts.shape == (1, 2), "expected exactly one red marker"
    r, c = map(int, starts[0])
    assert r == 0, "marker must be on the top border"
    assert 0 < c < grid.shape[1] - 1, "marker must not be in a corner"
    return r, c


def _extract_bars(grid: np.ndarray) -> tuple[list[dict], dict[tuple[int, int], dict]]:
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    assert bg == BG

    bars: list[dict] = []
    cell_to_bar: dict[tuple[int, int], dict] = {}
    for comp in find_components(g, bg=bg, colors=[BLUE, GREEN], connectivity=4, return_format="dict"):
        r0, c0, r1, c1 = map(int, comp["bbox"])
        assert r0 == r1, "bars must be horizontal"
        assert 0 < r0 < g.shape[0] - 1, "bars must stay off the border"
        assert 0 < c0 <= c1 < g.shape[1] - 1, "bars must stay off the border"
        length = c1 - c0 + 1
        assert MIN_LEN <= length <= MAX_LEN, "bar length out of range"
        bar = {
            "row": r0,
            "start": c0,
            "end": c1,
            "length": length,
            "color": int(comp["color"]),
            "coords": comp["coords"],
        }
        bars.append(bar)
        for rr, cc in comp["coords"]:
            key = (int(rr), int(cc))
            assert key not in cell_to_bar, "bars must not overlap"
            cell_to_bar[key] = bar
    assert 3 <= len(bars) <= MAX_BARS, "bar count out of range"
    return bars, cell_to_bar


def _paint_background_only(inp: np.ndarray, out: np.ndarray, r: int, c: int) -> None:
    val = int(inp[r, c])
    assert val in (BG, RED), "path may only touch background or the marker"
    if val == BG:
        out[r, c] = RED


def _trace_path(grid: np.ndarray, out: np.ndarray | None = None) -> tuple[list[int], set[tuple[int, int]]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    _, lane = _extract_marker(g)
    bars, cell_to_bar = _extract_bars(g)
    _ = bars

    path_cells = {(0, lane)}
    sequence: list[int] = []
    row = 0

    def mark(rr: int, cc: int) -> None:
        assert 0 <= rr < h and 0 <= cc < w
        path_cells.add((rr, cc))
        if out is not None:
            _paint_background_only(g, out, rr, cc)

    while row < h - 2:
        next_row = row + 1
        front = int(g[next_row, lane])
        if front in (BLUE, GREEN):
            bar = cell_to_bar[(next_row, lane)]
            side_c = lane - 1 if front == BLUE else lane + 1
            end_row = int(bar["row"]) + 1

            assert 0 < side_c < w - 1, "detour leaves the interior"
            assert end_row <= h - 2, "detour would hit the bottom border"
            assert int(g[int(bar["row"]), side_c]) == BG, "detour side must be clear"

            mark(row, side_c)
            for rr in range(row + 1, end_row + 1):
                mark(rr, side_c)
            mark(end_row, lane)

            sequence.append(front)
            row = end_row
            continue

        assert front in (BG, RED), "path may only continue through background"
        mark(next_row, lane)
        row = next_row

    return sequence, path_cells


def _build_output(grid: np.ndarray) -> tuple[np.ndarray, list[int], set[tuple[int, int]]]:
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    marker = _extract_marker(g)
    sequence, path_cells = _trace_path(g, out)
    assert len(sequence) >= MIN_HITS, "expected at least three encounters"

    out[0, :] = TEAL
    out[-1, :] = TEAL
    out[:, 0] = TEAL
    out[:, -1] = TEAL
    out[marker] = RED

    idx = 0
    for coord in _border_clockwise(g.shape):
        if coord == marker:
            continue
        out[coord] = sequence[idx % len(sequence)]
        idx += 1

    return out, sequence, path_cells


def main(grid: np.ndarray) -> np.ndarray:
    """Repaint a border by cycling through the obstacle colors encountered by a red detour path.

    Input structure:
    - A black grid contains one red marker on the top border and several interior blue/green horizontal bars.
    - The bars are one-cell thick, do not touch each other, and the marker defines a downward travel lane.
    - No border frame is present in the input apart from the single red marker.

    Rule:
    - Draw a one-pixel teal border, but keep the red marker red.
    - Starting from the marker, trace a red path straight downward through the interior.
    - When the next cell is blue, detour one cell left around that bar; when it is green, detour one cell right; only black cells become red.
    - Record the encountered bar colors in order, then traverse the border clockwise from the top-left corner and replace each non-marker border cell with the next color from that recorded sequence, cycling as needed.

    Key insight:
    - The obstacle bars are both geometry and code: they determine the red path shape and simultaneously emit the color cycle that repaints the border.
    """
    return _build_output(np.asarray(grid, dtype=int))[0]


def _mark_forbidden(mask: np.ndarray, row: int, start: int, end: int) -> None:
    h, w = mask.shape
    r0 = max(0, row - 1)
    r1 = min(h - 1, row + 1)
    c0 = max(0, start - 1)
    c1 = min(w - 1, end + 1)
    mask[r0 : r1 + 1, c0 : c1 + 1] = True


def _place_bar(grid: np.ndarray, row: int, start: int, length: int, color: int) -> None:
    grid[row, start : start + length] = int(color)


def _choose_hit_sequence(count: int) -> list[int] | None:
    seq = [BLUE if np.random.randint(0, 2) == 0 else GREEN for _ in range(count)]
    if len(set(seq)) != 2:
        return None
    if _primitive_period(seq) != len(seq):
        return None
    return seq


def _choose_hit_rows(height: int, count: int) -> list[int] | None:
    max_last = height - 3
    latest_first = max_last - MIN_GAP * (count - 1)
    if latest_first < 2:
        return None

    first_hi = min(4, latest_first)
    cur = int(np.random.randint(2, first_hi + 1))
    rows = [cur]
    for idx in range(1, count):
        remaining = count - idx - 1
        gap_hi = min(MAX_GAP, max_last - cur - MIN_GAP * remaining)
        if gap_hi < MIN_GAP:
            return None
        cur += int(np.random.randint(MIN_GAP, gap_hi + 1))
        rows.append(cur)
    return rows


def _scenery_candidates(
    forbidden: np.ndarray,
    lane: int,
    preferred_rows: set[int],
) -> tuple[list[tuple[int, int, int]], list[tuple[int, int, int]]]:
    h, w = forbidden.shape
    preferred: list[tuple[int, int, int]] = []
    others: list[tuple[int, int, int]] = []

    for row in range(1, h - 1):
        for length in range(MIN_LEN, min(MAX_LEN, w - 2) + 1):
            for start in range(1, w - length):
                end = start + length - 1
                if start <= lane + 1 and end >= lane - 1:
                    continue
                if forbidden[row, start : end + 1].any():
                    continue
                target = (row, start, length)
                if row in preferred_rows:
                    preferred.append(target)
                else:
                    others.append(target)
    return preferred, others


def _bars_touch(bars: list[dict]) -> bool:
    occupied = set()
    halo = set()
    for bar in bars:
        row = int(bar["row"])
        for col in range(int(bar["start"]), int(bar["end"]) + 1):
            if (row, col) in halo:
                return True
            occupied.add((row, col))
        for rr in range(row - 1, row + 2):
            for cc in range(int(bar["start"]) - 1, int(bar["end"]) + 2):
                halo.add((rr, cc))
    return False


def _check_generated(grid: np.ndarray) -> bool:
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2 or min(g.shape) < MIN_SIZE or max(g.shape) > 30:
        return False

    try:
        marker = _extract_marker(g)
        bars, _ = _extract_bars(g)
        sequence, path_cells = _trace_path(g)
        out, seq2, _ = _build_output(g)
    except Exception:
        return False

    if sequence != seq2:
        return False
    if len(sequence) < MIN_HITS or len(set(sequence)) != 2:
        return False
    if _primitive_period(sequence) != len(sequence):
        return False
    if len(bars) <= len(sequence):
        return False
    if _bars_touch(bars):
        return False

    lane = int(marker[1])
    hit_bars = sorted(
        [bar for bar in bars if int(bar["start"]) <= lane <= int(bar["end"])],
        key=lambda bar: (int(bar["row"]), int(bar["start"])),
    )
    if [int(bar["color"]) for bar in hit_bars] != sequence:
        return False

    all_bars_rowwise = [int(bar["color"]) for bar in sorted(bars, key=lambda bar: (int(bar["row"]), int(bar["start"])))]
    if _primitive_cycle(all_bars_rowwise) == tuple(sequence):
        return False

    for bar in hit_bars:
        row = int(bar["row"])
        if int(bar["color"]) == BLUE:
            required = {(row - 1, lane - 1), (row, lane - 1), (row + 1, lane - 1), (row + 1, lane)}
            forbidden = (row, lane + 1)
        else:
            required = {(row - 1, lane + 1), (row, lane + 1), (row + 1, lane + 1), (row + 1, lane)}
            forbidden = (row, lane - 1)
        if not required <= path_cells:
            return False
        if forbidden in path_cells:
            return False

    border = _border_clockwise(g.shape)
    border_vals = [int(out[r, c]) for r, c in border if (r, c) != marker]
    if any(val == TEAL for val in border_vals):
        return False
    if any(val != sequence[i % len(sequence)] for i, val in enumerate(border_vals)):
        return False

    return True


def _generate_candidate() -> np.ndarray | None:
    hit_count = int(np.random.randint(MIN_HITS, MAX_HITS_EXCL))
    min_height = max(MIN_SIZE, 2 * hit_count + 3)
    height = int(np.random.randint(min_height, MAX_SIZE_EXCL))
    width = int(np.random.randint(MIN_SIZE, MAX_SIZE_EXCL))
    lane = int(np.random.randint(3, width - 3))

    hit_rows = _choose_hit_rows(height, hit_count)
    hit_sequence = _choose_hit_sequence(hit_count)
    if hit_rows is None or hit_sequence is None:
        return None

    grid = np.zeros((height, width), dtype=int)
    grid[0, lane] = RED

    forbidden = np.zeros((height, width), dtype=bool)
    hit_specs: list[tuple[int, int, int, int]] = []
    for row, color in zip(hit_rows, hit_sequence):
        max_len = min(MAX_LEN, (width - 1 - lane) if color == BLUE else lane)
        if max_len < MIN_LEN:
            return None
        length = int(np.random.randint(MIN_LEN, max_len + 1))
        start = lane if color == BLUE else lane - length + 1
        end = start + length - 1
        _place_bar(grid, row, start, length, color)
        _mark_forbidden(forbidden, row, start, end)
        hit_specs.append((row, start, length, color))

    preferred_rows = {
        row
        for a, b in zip(hit_rows, hit_rows[1:])
        for row in range(a + 1, b)
    }
    preferred, others = _scenery_candidates(forbidden, lane, preferred_rows)
    if not preferred and not others:
        return None

    max_scenery = min(2, MAX_BARS - hit_count)
    scenery_count = int(np.random.randint(1, max_scenery + 1))
    scenery_positions: list[tuple[int, int, int]] = []

    for idx in range(scenery_count):
        preferred, others = _scenery_candidates(forbidden, lane, preferred_rows)
        pool = preferred if idx == 0 and preferred else preferred + others
        if not pool:
            return None
        row, start, length = pool[int(np.random.randint(0, len(pool)))]
        scenery_positions.append((row, start, length))
        _mark_forbidden(forbidden, row, start, start + length - 1)

    assignments = list(product((BLUE, GREEN), repeat=scenery_count))
    order = np.random.permutation(len(assignments))
    for idx in order:
        candidate = grid.copy()
        for (row, start, length), color in zip(scenery_positions, assignments[int(idx)]):
            _place_bar(candidate, row, start, length, int(color))
        if _check_generated(candidate):
            return candidate

    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _generate_candidate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle input")
