from __future__ import annotations

import random
from typing import Iterable

import numpy as np


BG = 0
WALL = 9
PALETTE = (1, 2, 3, 4, 7)
TOTAL_RANGE = range(3, 9)
COLOR_COUNT_RANGE = (3, 5)
HEIGHT_RANGE = (14, 18)
INTERIOR_WIDTH_RANGE = (10, 13)
STEP_WIDTH_RANGE = (3, 5)
GENERATION_ATTEMPTS = 600

_SEEN_INPUTS: set[tuple[tuple[int, int], bytes]] = set()
_SEEN_OUTPUTS: set[tuple[tuple[int, int], bytes]] = set()


def _signature(grid: np.ndarray) -> tuple[tuple[int, int], bytes]:
    arr = np.asarray(grid, dtype=int)
    return tuple(map(int, arr.shape)), arr.tobytes()


def _non_special_colors(grid: np.ndarray) -> list[int]:
    return sorted(int(v) for v in np.unique(grid) if int(v) not in (BG, WALL))


def _basin_span(grid: np.ndarray) -> tuple[int, int]:
    bottom_cols = np.flatnonzero(grid[-1] == WALL)
    assert bottom_cols.size >= 3, "expected a maroon floor"
    left = int(bottom_cols[0])
    right = int(bottom_cols[-1])
    assert np.all(grid[-1, left : right + 1] == WALL), "floor must be contiguous"
    return left, right


def _settle_column(column: np.ndarray) -> np.ndarray:
    out = np.asarray(column, dtype=int).copy()
    movers_mask = (out != BG) & (out != WALL)
    out[movers_mask] = BG

    barriers = np.flatnonzero(column == WALL)
    start = 0
    for barrier in barriers:
        if barrier > start:
            colors = [int(column[r]) for r in range(start, barrier) if int(column[r]) not in (BG, WALL)]
            if colors:
                base = int(barrier) - len(colors)
                for offset, color in enumerate(colors):
                    out[base + offset] = int(color)
        start = int(barrier) + 1

    if start < len(column):
        colors = [int(column[r]) for r in range(start, len(column)) if int(column[r]) not in (BG, WALL)]
        if colors:
            base = len(column) - len(colors)
            for offset, color in enumerate(colors):
                out[base + offset] = int(color)
    return out


def _simulate_gravity(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    for c in range(g.shape[1]):
        out[:, c] = _settle_column(g[:, c])
    return out


def _bottom_chamber_counts(grid: np.ndarray) -> dict[int, int]:
    left, right = _basin_span(grid)
    counts: dict[int, int] = {}
    for c in range(left + 1, right):
        r = grid.shape[0] - 2
        while r >= 0 and int(grid[r, c]) != WALL:
            color = int(grid[r, c])
            if color not in (BG, WALL):
                counts[color] = counts.get(color, 0) + 1
            r -= 1
    return counts


def _leftmost_columns(grid: np.ndarray, colors: Iterable[int]) -> dict[int, int]:
    out: dict[int, int] = {}
    for color in colors:
        cols = np.flatnonzero(grid == int(color)) % grid.shape[1]
        assert cols.size > 0
        out[int(color)] = int(cols.min())
    return out


def _ring(colors: list[int]) -> np.ndarray:
    side = 2 * len(colors) - 1
    out = np.full((side, side), int(colors[0]), dtype=int)
    for layer, color in enumerate(colors[1:], start=1):
        out[layer : side - layer, layer : side - layer] = int(color)
    return out


def _analyze(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert max(g.shape) <= 30
    colors = _non_special_colors(g)
    assert colors, "expected colored pixels above the basin"

    settled = _simulate_gravity(g)
    bottom_counts_raw = _bottom_chamber_counts(settled)
    leftmost = _leftmost_columns(g, colors)

    totals = {color: int(np.count_nonzero(g == color)) for color in colors}
    bottom_counts = {color: int(bottom_counts_raw.get(color, 0)) for color in colors}
    ordered = sorted(colors, key=lambda color: (-bottom_counts[color], leftmost[color]))
    return {
        "colors": colors,
        "settled": settled,
        "totals": totals,
        "bottom_counts": bottom_counts,
        "leftmost": leftmost,
        "ordered": ordered,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Drop colored pixels into a stepped basin, rank colors by how many pixels reach the lowest chamber, and draw nested rings in that order.

    Input structure:
    - Black background with maroon side walls and a maroon bottom floor forming one basin.
    - Inside the basin are one or two maroon horizontal steps that block vertical motion.
    - Above or between those steps are 3-5 colors of movable pixels.

    Rule:
    - Every movable pixel falls straight down within its own column and stops on the first maroon obstacle or on previously settled pixels in that chamber; there is no horizontal motion.
    - After all pixels settle, count for each color only the pixels in the lowest chamber directly above the basin floor.
    - Sort colors by that lowest-chamber count from large to small, breaking ties by the leftmost input column of that color.
    - Output a square of side 2K-1, where K is the number of colors, and paint concentric square rings from outermost to center in that sorted order.

    Key insight:
    - The visible source sizes are not the ranking key: maroon steps intercept some pixels, so the ring order is determined by gravity-filtered bottom counts rather than by each color's original total area.
    """
    analysis = _analyze(np.asarray(grid, dtype=int))
    return _ring(analysis["ordered"])


def _build_basin(k: int) -> tuple[np.ndarray, list[int], list[int]]:
    interior_w = random.randint(max(INTERIOR_WIDTH_RANGE[0], 2 * k), INTERIOR_WIDTH_RANGE[1])
    left_margin = random.randint(1, 2)
    right_margin = random.randint(1, 2)
    width = left_margin + interior_w + 2 + right_margin
    height = random.randint(*HEIGHT_RANGE)
    grid = np.full((height, width), BG, dtype=int)

    left_wall = left_margin
    right_wall = left_wall + interior_w + 1
    basin_top = random.randint(3, 5)
    grid[basin_top:, left_wall] = WALL
    grid[basin_top:, right_wall] = WALL
    grid[-1, left_wall : right_wall + 1] = WALL

    n_steps = 1 if random.random() < 0.5 else 2
    step_rows: list[int]
    if n_steps == 1:
        step_rows = [random.randint(basin_top + 2, height - 4)]
    else:
        lower = random.randint(basin_top + 4, height - 4)
        upper = random.randint(basin_top + 2, lower - 2)
        step_rows = [upper, lower]

    starts: list[int] = []
    for idx, row in enumerate(step_rows):
        for _ in range(80):
            step_w = random.randint(*STEP_WIDTH_RANGE)
            start = random.randint(left_wall + 1, right_wall - step_w)
            cells = set(range(start, start + step_w))
            if idx == 0:
                starts.append(start)
                grid[row, start : start + step_w] = WALL
                break
            prev_w = int(np.count_nonzero(grid[step_rows[0]] == WALL))
            prev_start = starts[0]
            prev_cells = set(range(prev_start, prev_start + prev_w))
            if cells != prev_cells:
                starts.append(start)
                grid[row, start : start + step_w] = WALL
                break
        else:
            raise RuntimeError("failed to place step")
    return grid, step_rows, list(range(left_wall + 1, right_wall))


def _split_columns(columns: list[int], k: int) -> list[list[int]]:
    extra = len(columns) - 2 * k
    assert extra >= 0
    widths = [2] * k
    for _ in range(extra):
        widths[random.randrange(k)] += 1

    parts: list[list[int]] = []
    start = 0
    for width in widths:
        parts.append(columns[start : start + width])
        start += width
    assert start == len(columns)
    random.shuffle(parts)
    return parts


def _candidate_maps(base: np.ndarray, interior_cols: list[int]) -> dict:
    floor_row = base.shape[0] - 1
    step_cols = {
        c
        for c in interior_cols
        if np.any(base[:floor_row, c] == WALL)
    }

    bottom_cells: list[tuple[int, int, int]] = []
    blocked_cells: list[tuple[int, int, int]] = []
    for c in interior_cols:
        for r in range(floor_row):
            if int(base[r, c]) != BG or int(base[r + 1, c]) != BG:
                continue
            below = np.flatnonzero(base[r + 1 :, c] == WALL)
            if below.size == 0:
                continue
            barrier = int(r + 1 + below[0])
            if barrier == floor_row:
                bottom_cells.append((int(r), int(c), int(c in step_cols)))
            else:
                blocked_cells.append((int(r), int(c), barrier))

    return {
        "bottom": bottom_cells,
        "blocked": blocked_cells,
        "step_cols": step_cols,
    }


def _sample_counts(segment_infos: list[dict], step_rows: list[int]) -> tuple[list[int], list[int]] | None:
    blocked_capable = [idx for idx, info in enumerate(segment_infos) if info["blocked"]]
    if len(blocked_capable) < 2:
        return None

    for _ in range(200):
        totals = random.sample(list(TOTAL_RANGE), len(segment_infos))
        extras = [0] * len(segment_infos)
        altered = random.sample(blocked_capable, random.randint(2, len(blocked_capable)))
        valid = True
        for idx, info in enumerate(segment_infos):
            max_extra = min(3, totals[idx] - 1, len(info["blocked"]))
            min_extra = 1 if idx in altered else 0
            if max_extra < min_extra:
                valid = False
                break
            extras[idx] = random.randint(min_extra, max_extra)
        if not valid:
            continue

        bottoms = [totals[idx] - extras[idx] for idx in range(len(segment_infos))]
        if len(set(bottoms)) != len(bottoms):
            continue
        if any(bottoms[idx] > len(segment_infos[idx]["bottom"]) for idx in range(len(segment_infos))):
            continue

        total_order = sorted(range(len(segment_infos)), key=lambda idx: -totals[idx])
        bottom_order = sorted(range(len(segment_infos)), key=lambda idx: -bottoms[idx])
        if total_order == bottom_order:
            continue

        if len(step_rows) == 2:
            coverable = set()
            for idx, extra in enumerate(extras):
                if extra <= 0:
                    continue
                coverable.update(segment_infos[idx]["blocked_barriers"])
            if set(step_rows) - coverable:
                continue
        return totals, bottoms
    return None


def _segment_info(base: np.ndarray, segments: list[list[int]]) -> list[dict]:
    maps = _candidate_maps(base, [c for seg in segments for c in seg])
    infos: list[dict] = []
    for seg in segments:
        seg_set = set(seg)
        bottom = [cell for cell in maps["bottom"] if int(cell[1]) in seg_set]
        blocked = [cell for cell in maps["blocked"] if int(cell[1]) in seg_set]
        infos.append(
            {
                "columns": seg,
                "bottom": bottom,
                "bottom_under_step": [cell for cell in bottom if int(cell[2]) == 1],
                "blocked": blocked,
                "blocked_barriers": sorted({int(cell[2]) for cell in blocked}),
            }
        )
    return infos


def _pick_cells(
    cells: list[tuple[int, int, int]],
    count: int,
    *,
    require_one_from: list[tuple[int, int, int]] | None = None,
    barrier_row: int | None = None,
) -> list[tuple[int, int, int]] | None:
    if barrier_row is not None:
        cells = [cell for cell in cells if int(cell[2]) == int(barrier_row)]
    if len(cells) < count:
        return None

    chosen: list[tuple[int, int, int]] = []
    remaining = cells[:]
    random.shuffle(remaining)
    if require_one_from:
        pool = require_one_from[:]
        random.shuffle(pool)
        first = pool[0]
        if barrier_row is not None and int(first[2]) != int(barrier_row):
            return None
        chosen.append(first)
        remaining = [cell for cell in remaining if cell != first]

    need = count - len(chosen)
    if need < 0 or len(remaining) < need:
        return None
    chosen.extend(remaining[:need])
    return chosen


def _gen_attempt() -> np.ndarray | None:
    k = random.randint(*COLOR_COUNT_RANGE)
    colors = random.sample(PALETTE, k)
    base, step_rows, interior_cols = _build_basin(k)
    segments = _split_columns(interior_cols, k)
    infos = _segment_info(base, segments)

    counts = _sample_counts(infos, step_rows)
    if counts is None:
        return None
    totals, bottoms = counts
    extras = [totals[idx] - bottoms[idx] for idx in range(k)]

    row_sensitive_segments = [idx for idx, info in enumerate(infos) if info["bottom_under_step"]]
    if not row_sensitive_segments:
        return None
    row_sensitive_idx = random.choice([idx for idx in row_sensitive_segments if bottoms[idx] > 0])

    forced_barriers: dict[int, int] = {}
    if len(step_rows) == 2:
        need_rows = step_rows[:]
        random.shuffle(need_rows)
        barrier_segments: list[int] = []
        for barrier in need_rows:
            candidates = [
                idx
                for idx, info in enumerate(infos)
                if extras[idx] > 0 and barrier in info["blocked_barriers"] and idx not in barrier_segments
            ]
            if not candidates:
                return None
            picked = random.choice(candidates)
            barrier_segments.append(picked)
            forced_barriers[picked] = int(barrier)

    grid = base.copy()
    used_barriers: set[int] = set()
    for idx, color in enumerate(colors):
        info = infos[idx]
        bottom_require = info["bottom_under_step"] if idx == row_sensitive_idx else None
        chosen_bottom = _pick_cells(info["bottom"], bottoms[idx], require_one_from=bottom_require)
        if chosen_bottom is None:
            return None

        if extras[idx] > 0:
            chosen_blocked = _pick_cells(
                info["blocked"],
                extras[idx],
                barrier_row=forced_barriers.get(idx),
            )
            if chosen_blocked is None:
                return None
        else:
            chosen_blocked = []

        for r, c, _meta in chosen_bottom + chosen_blocked:
            if grid[r, c] != BG:
                return None
            grid[r, c] = int(color)
        used_barriers.update(int(meta) for _r, _c, meta in chosen_blocked)

    analysis = _analyze(grid)
    bottom_counts = analysis["bottom_counts"]
    totals_by_color = analysis["totals"]

    if len(analysis["colors"]) != k:
        return None
    if len(set(totals_by_color.values())) != k:
        return None
    if len(set(bottom_counts.values())) != k:
        return None
    if sum(1 for color in colors if bottom_counts[color] != totals_by_color[color]) < 2:
        return None

    total_order = sorted(colors, key=lambda color: -totals_by_color[color])
    if total_order == analysis["ordered"]:
        return None
    if len(step_rows) == 2 and used_barriers != set(step_rows):
        return None

    inp_sig = _signature(grid)
    out_sig = _signature(_ring(analysis["ordered"]))
    if inp_sig in _SEEN_INPUTS or out_sig in _SEEN_OUTPUTS:
        return None
    _SEEN_INPUTS.add(inp_sig)
    _SEEN_OUTPUTS.add(out_sig)
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid gravity-ring example")
