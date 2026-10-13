from __future__ import annotations

from collections import Counter
import random

import numpy as np


BLUE = 1
RED = 2
PINK = 6
BG = 0

ORIENTS = ("down", "up", "right", "left")
SIDES = (-1, 1)


def _background(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = map(int, shape)
    return 0 <= int(r) < h and 0 <= int(c) < w


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components4(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    remaining = {(int(r), int(c)) for r, c in np.argwhere(mask)}
    comps: list[list[tuple[int, int]]] = []
    while remaining:
        seed = remaining.pop()
        stack = [seed]
        comp = [seed]
        while stack:
            r, c = stack.pop()
            for nr, nc in _neighbors4(r, c):
                if (nr, nc) in remaining:
                    remaining.remove((nr, nc))
                    stack.append((nr, nc))
                    comp.append((nr, nc))
        comps.append(sorted(comp))
    return comps


def _bar_records(grid: np.ndarray, bg: int, dr: int, dc: int) -> tuple[list[dict], dict[tuple[int, int], dict]]:
    mask = (grid == BLUE) | (grid == PINK)
    bars: list[dict] = []
    cell_to_bar: dict[tuple[int, int], dict] = {}

    for comp in _components4(mask):
        colors = {int(grid[r, c]) for r, c in comp}
        assert colors <= {BLUE, PINK}, "bars may contain only blue and pink"
        pinks = [(r, c) for r, c in comp if int(grid[r, c]) == PINK]
        assert len(pinks) == 1, "each bar must contain exactly one pink marker"
        assert len(comp) >= 4, "bar must be long enough for an off-center internal marker"
        pr, pc = pinks[0]

        if dr != 0:
            rows = {r for r, _c in comp}
            assert len(rows) == 1, "vertical travel requires horizontal bars"
            row = next(iter(rows))
            cols = sorted(c for _r, c in comp)
            c0, c1 = int(cols[0]), int(cols[-1])
            assert cols == list(range(c0, c1 + 1)), "bar cells must be contiguous"
            assert c0 < pc < c1, "pink marker must be internal to the bar"
            left_dist = pc - c0
            right_dist = c1 - pc
            assert left_dist != right_dist, "pink marker must be closer to exactly one end"
            side_c = c0 - 1 if left_dist < right_dist else c1 + 1
            record = {
                "id": len(bars),
                "bbox": (row, c0, row, c1),
                "side": (row, side_c),
            }
        else:
            cols = {c for _r, c in comp}
            assert len(cols) == 1, "horizontal travel requires vertical bars"
            col = next(iter(cols))
            rows = sorted(r for r, _c in comp)
            r0, r1 = int(rows[0]), int(rows[-1])
            assert rows == list(range(r0, r1 + 1)), "bar cells must be contiguous"
            assert r0 < pr < r1, "pink marker must be internal to the bar"
            top_dist = pr - r0
            bottom_dist = r1 - pr
            assert top_dist != bottom_dist, "pink marker must be closer to exactly one end"
            side_r = r0 - 1 if top_dist < bottom_dist else r1 + 1
            record = {
                "id": len(bars),
                "bbox": (r0, col, r1, col),
                "side": (side_r, col),
            }

        bars.append(record)
        for cell in comp:
            assert cell not in cell_to_bar
            cell_to_bar[cell] = record

    assert bars, "expected at least one obstacle bar"
    return bars, cell_to_bar


def _start_and_dir(grid: np.ndarray) -> tuple[int, int, int, int]:
    h, w = map(int, grid.shape)
    starts = np.argwhere(grid == RED)
    assert starts.shape == (1, 2), "expected one red border marker"
    r, c = map(int, starts[0])

    dirs: list[tuple[int, int]] = []
    if r == 0:
        dirs.append((1, 0))
    if r == h - 1:
        dirs.append((-1, 0))
    if c == 0:
        dirs.append((0, 1))
    if c == w - 1:
        dirs.append((0, -1))
    assert len(dirs) == 1, "red marker must be on exactly one non-corner border"
    dr, dc = dirs[0]
    return r, c, dr, dc


def main(grid: np.ndarray) -> np.ndarray:
    """
    Draw the red inward trace, detouring around each blue bar by the end selected by its pink marker.

    Input structure:
    - The dominant background contains one red marker on a non-corner border.
    - Inside the grid are several one-cell-thick blue bars perpendicular to the marker's inward direction.
    - Each bar contains exactly one internal pink cell, and that pink cell is closer to one bar end than the other.

    Rule:
    - Start at the red marker and move straight inward, coloring only background cells red.
    - When the current lane hits a bar, turn one cell before impact toward the bar end nearer to the pink marker.
    - Move parallel to the bar until one cell past that chosen end, step forward beside the endpoint, then resume the same inward direction from the shifted lane.
    - Existing blue and pink bar cells remain unchanged.

    Key insight:
    - The detour side is encoded by the pink cell's geometric position inside each bar, and the lane shift is cumulative across bars.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"
    assert max(g.shape) <= 30, "ARC grids must fit inside 30x30"
    bg = _background(g)
    assert bg not in (BLUE, RED, PINK), "background must be distinct from role colors"
    assert set(map(int, np.unique(g))) <= {bg, BLUE, RED, PINK}, "unexpected colors"

    r, c, dr, dc = _start_and_dir(g)
    bars, cell_to_bar = _bar_records(g, bg, dr, dc)

    out = g.copy()
    seen_bars: set[int] = set()

    def paint_cell(rr: int, cc: int) -> None:
        assert _in_bounds(g.shape, rr, cc), "trace must stay in bounds"
        assert int(g[rr, cc]) in (bg, RED), "trace may only paint background cells"
        out[rr, cc] = RED

    def paint_line(a: tuple[int, int], b: tuple[int, int]) -> None:
        ar, ac = a
        br, bc = b
        if ar == br:
            step = 1 if bc >= ac else -1
            for cc in range(ac, bc + step, step):
                paint_cell(ar, cc)
        elif ac == bc:
            step = 1 if br >= ar else -1
            for rr in range(ar, br + step, step):
                paint_cell(rr, ac)
        else:
            raise AssertionError("trace segments must be axis-aligned")

    paint_cell(r, c)
    while True:
        rr, cc = r + dr, c + dc
        hit: tuple[int, int] | None = None
        while _in_bounds(g.shape, rr, cc):
            if (rr, cc) in cell_to_bar:
                hit = (rr, cc)
                break
            assert int(g[rr, cc]) in (bg, RED), "non-bar objects may not block the lane"
            rr += dr
            cc += dc

        if hit is None:
            last = (rr - dr, cc - dc)
            if (r, c) != last:
                paint_line((r + dr, c + dc), last)
            break

        hr, hc = hit
        turn_start = (hr - dr, hc - dc)
        if (r, c) != turn_start:
            paint_line((r + dr, c + dc), turn_start)

        bar = cell_to_bar[hit]
        assert int(bar["id"]) not in seen_bars, "a bar should be encountered only once"
        seen_bars.add(int(bar["id"]))

        side = tuple(map(int, bar["side"]))
        assert _in_bounds(g.shape, side[0], side[1]), "chosen detour side must be in bounds"
        assert int(g[side]) == bg, "cell beyond the chosen endpoint must be open"
        turn_end = (side[0] - dr, side[1] - dc)
        assert _in_bounds(g.shape, turn_end[0], turn_end[1]), "pre-bar detour lane must be in bounds"

        paint_line(turn_start, turn_end)
        paint_cell(side[0], side[1])
        r, c = side

    assert seen_bars == {int(bar["id"]) for bar in bars}, "every bar must be hit in sequence"
    return out


# ---------------- generator ----------------

GEN_ATTEMPTS = 500
MIN_GAP = 3
MAX_GAP = 5
BAR_LENGTHS = (4, 5, 6, 7)
HIT_COUNT_RANGE = (3, 6)


def _shape(axis_len: int, lat_len: int, orient: str) -> tuple[int, int]:
    return (axis_len, lat_len) if orient in ("down", "up") else (lat_len, axis_len)


def _to_rc(axis: int, lane: int, shape: tuple[int, int], orient: str) -> tuple[int, int]:
    h, w = shape
    if orient == "down":
        return axis, lane
    if orient == "up":
        return h - 1 - axis, lane
    if orient == "right":
        return lane, axis
    if orient == "left":
        return lane, w - 1 - axis
    raise AssertionError("unknown orientation")


def _gen_levels(axis_len: int, n_hits: int) -> list[int] | None:
    max_last = axis_len - 3
    latest_first = max_last - MIN_GAP * (n_hits - 1)
    if latest_first < 3:
        return None
    first = random.randint(3, min(5, latest_first))
    levels = [first]
    cur = first
    for i in range(1, n_hits):
        remaining = n_hits - 1 - i
        max_gap = min(MAX_GAP, max_last - cur - MIN_GAP * remaining)
        if max_gap < MIN_GAP:
            return None
        cur += random.randint(MIN_GAP, max_gap)
        levels.append(cur)
    return levels


def _pink_options(start: int, end: int, sign: int) -> list[int]:
    opts = []
    for pos in range(start + 1, end):
        low_dist = pos - start
        high_dist = end - pos
        if sign < 0 and low_dist < high_dist:
            opts.append(pos)
        if sign > 0 and high_dist < low_dist:
            opts.append(pos)
    return opts


def _bar_choices(lane: int, lat_len: int, allowed_signs: tuple[int, ...]) -> list[dict]:
    choices: list[dict] = []
    for length in BAR_LENGTHS:
        max_start = lat_len - length - 2
        for start in range(2, max_start + 1):
            end = start + length - 1
            if not (start <= lane <= end):
                continue
            for sign in allowed_signs:
                side_lane = start - 1 if sign < 0 else end + 1
                if not (1 <= side_lane <= lat_len - 2):
                    continue
                pinks = _pink_options(start, end, sign)
                if pinks:
                    choices.append(
                        {
                            "start": start,
                            "end": end,
                            "sign": sign,
                            "side_lane": side_lane,
                            "pinks": pinks,
                        }
                    )
    return choices


def _place_bar(grid: np.ndarray, orient: str, axis: int, start: int, end: int, pink: int) -> None:
    for lane in range(start, end + 1):
        r, c = _to_rc(axis, lane, grid.shape, orient)
        grid[r, c] = PINK if lane == pink else BLUE


def _paint_canonical(
    out: np.ndarray,
    orient: str,
    axis: int,
    lane: int,
    color: int,
) -> None:
    r, c = _to_rc(axis, lane, out.shape, orient)
    assert int(out[r, c]) in (BG, RED), "generated trace must not overwrite a bar"
    out[r, c] = int(color)


def _expected_from_specs(
    grid: np.ndarray,
    orient: str,
    axis_len: int,
    start_lane: int,
    specs: list[dict],
) -> np.ndarray:
    out = grid.copy()
    axis_pos = 0
    lane = int(start_lane)
    _paint_canonical(out, orient, 0, lane, RED)

    for spec in specs:
        axis = int(spec["axis"])
        start = int(spec["start"])
        end = int(spec["end"])
        side_lane = int(spec["side_lane"])
        assert start <= lane <= end

        for aa in range(axis_pos + 1, axis):
            _paint_canonical(out, orient, aa, lane, RED)

        lo, hi = sorted((lane, side_lane))
        for ll in range(lo, hi + 1):
            _paint_canonical(out, orient, axis - 1, ll, RED)

        _paint_canonical(out, orient, axis, side_lane, RED)
        axis_pos = axis
        lane = side_lane

    for aa in range(axis_pos + 1, axis_len):
        _paint_canonical(out, orient, aa, lane, RED)
    return out


def _semantic_specs_ok(specs: list[dict], initial_lane: int, final_lane: int) -> bool:
    signs = {int(spec["sign"]) for spec in specs}
    if signs != {-1, 1}:
        return False

    lanes = [int(spec["lane_before"]) for spec in specs] + [int(final_lane)]
    if len(set(lanes)) < 3:
        return False

    misses_initial = sum(
        1 for spec in specs[1:] if not (int(spec["start"]) <= initial_lane <= int(spec["end"]))
    )
    if misses_initial < 1:
        return False

    shifts = [abs(int(spec["side_lane"]) - int(spec["lane_before"])) for spec in specs]
    if max(shifts) < 2:
        return False

    for spec in specs:
        start = int(spec["start"])
        end = int(spec["end"])
        pink = int(spec["pink"])
        left_dist = pink - start
        right_dist = end - pink
        if left_dist <= 0 or right_dist <= 0 or left_dist == right_dist:
            return False
        if int(spec["sign"]) < 0 and not left_dist < right_dist:
            return False
        if int(spec["sign"]) > 0 and not right_dist < left_dist:
            return False
    return True


def _gen_attempt() -> np.ndarray | None:
    n_hits = random.randint(*HIT_COUNT_RANGE)
    min_axis_len = 3 + MIN_GAP * (n_hits - 1) + 3
    axis_len = random.randint(max(15, min_axis_len), 30)
    lat_len = random.randint(12, 30)
    orient = random.choice(ORIENTS)
    shape = _shape(axis_len, lat_len, orient)
    levels = _gen_levels(axis_len, n_hits)
    if levels is None:
        return None

    grid = np.full(shape, BG, dtype=int)
    lane = random.randint(3, lat_len - 4)
    initial_lane = int(lane)
    sr, sc = _to_rc(0, lane, shape, orient)
    grid[sr, sc] = RED

    specs: list[dict] = []
    used_signs: set[int] = set()
    for i, axis in enumerate(levels):
        remaining = n_hits - i
        missing = tuple(sign for sign in SIDES if sign not in used_signs)
        allowed = missing if len(missing) == remaining else SIDES
        choices = _bar_choices(lane, lat_len, allowed)
        if i > 0:
            stronger = [
                ch for ch in choices if not (int(ch["start"]) <= initial_lane <= int(ch["end"]))
            ]
            if stronger and random.random() < 0.7:
                choices = stronger
        if not choices:
            return None

        choice = random.choice(choices)
        pink = random.choice(choice["pinks"])
        spec = {
            "axis": int(axis),
            "lane_before": int(lane),
            "start": int(choice["start"]),
            "end": int(choice["end"]),
            "pink": int(pink),
            "sign": int(choice["sign"]),
            "side_lane": int(choice["side_lane"]),
        }
        specs.append(spec)
        _place_bar(grid, orient, int(axis), spec["start"], spec["end"], spec["pink"])
        lane = int(choice["side_lane"])
        used_signs.add(int(choice["sign"]))

    if not _semantic_specs_ok(specs, initial_lane, lane):
        return None

    try:
        expected = _expected_from_specs(grid, orient, axis_len, initial_lane, specs)
        got = main(grid)
    except Exception:
        return None
    if not np.array_equal(got, expected):
        return None
    if not np.array_equal(got[grid == BLUE], grid[grid == BLUE]):
        return None
    if not np.array_equal(got[grid == PINK], grid[grid == PINK]):
        return None
    if int(np.count_nonzero((got == RED) & (grid == BG))) < axis_len:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid fable/4/007/4 input")
