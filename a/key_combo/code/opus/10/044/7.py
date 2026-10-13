from __future__ import annotations

import random
from typing import Iterable

import numpy as np


BG = 0
START = 2
INDICATOR = 5
SEGMENT_PALETTE = (1, 3, 4, 6, 7, 8, 9)
GEN_ATTEMPTS = 400

# The grey pixel selects one of four reflected 45-degree launch headings.
INDICATOR_TO_HEADING = {
    (-1, 0): (-1, 1),   # grey above    -> north-east
    (0, 1): (1, 1),     # grey right    -> south-east
    (1, 0): (1, -1),    # grey below    -> south-west
    (0, -1): (-1, -1),  # grey left     -> north-west
}
INDICATOR_SIDES = tuple(INDICATOR_TO_HEADING)

_GEN_COUNTER = 0


def _neighbors4(r: int, c: int) -> Iterable[tuple[int, int]]:
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _color_components(grid: np.ndarray, color: int) -> list[np.ndarray]:
    mask = grid == color
    h, w = grid.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                pts.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _segment_length(coords: np.ndarray) -> int:
    rows = coords[:, 0]
    cols = coords[:, 1]
    if np.unique(rows).size == 1:
        ordered = np.sort(cols)
        expected = np.arange(int(ordered[0]), int(ordered[-1]) + 1)
        assert np.array_equal(ordered, expected), "horizontal segment must be contiguous"
        return int(coords.shape[0])
    if np.unique(cols).size == 1:
        ordered = np.sort(rows)
        expected = np.arange(int(ordered[0]), int(ordered[-1]) + 1)
        assert np.array_equal(ordered, expected), "vertical segment must be contiguous"
        return int(coords.shape[0])
    raise AssertionError("foreground control objects must be straight segments")


def _extract_cycle(grid: np.ndarray) -> list[int]:
    segments: list[tuple[int, int]] = []
    for color in np.unique(grid):
        color = int(color)
        if color in (BG, START, INDICATOR):
            continue
        comps = _color_components(grid, color)
        assert len(comps) == 1, "each segment color must form exactly one component"
        coords = comps[0]
        assert np.min(coords[:, 0]) <= 2, "segments must live in the top control band"
        segments.append((_segment_length(coords), color))
    assert 3 <= len(segments) <= 4, "expected 3-4 control segments"
    lengths = [length for length, _color in segments]
    assert len(lengths) == len(set(lengths)), "segment lengths must be distinct"
    return [color for length, color in sorted(segments, key=lambda item: item[0], reverse=True)]


def _find_start_and_heading(grid: np.ndarray) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]]:
    starts = np.argwhere(grid == START)
    indicators = np.argwhere(grid == INDICATOR)
    assert starts.shape == (1, 2), "expected a unique red start marker"
    assert indicators.shape == (1, 2), "expected a unique grey direction marker"
    sr, sc = map(int, starts[0])
    ir, ic = map(int, indicators[0])
    delta = (ir - sr, ic - sc)
    assert delta in INDICATOR_TO_HEADING, "indicator must be 4-neighbor adjacent to the start"
    return (sr, sc), (ir, ic), INDICATOR_TO_HEADING[delta]


def _simulate(
    obstacle_grid: np.ndarray,
    cycle: list[int],
    start: tuple[int, int],
    heading: tuple[int, int],
    *,
    advance_on_bounce: bool = True,
    stop_on_painted: bool = True,
) -> tuple[np.ndarray, dict[str, object]]:
    h, w = obstacle_grid.shape
    out = np.zeros((h, w), dtype=int)
    obstacle_mask = obstacle_grid != BG
    obstacle_mask[start] = False

    r, c = start
    dr, dc = heading
    idx = 0
    bounces = 0
    reason = "limit"
    visited_states: set[tuple[int, int, int, int, int]] = set()

    for _ in range(h * w * 8):
        state = (r, c, dr, dc, idx)
        if state in visited_states:
            reason = "loop"
            break
        visited_states.add(state)

        hit_horizontal_edge = (r == 0 and dr < 0) or (r == h - 1 and dr > 0)
        hit_vertical_edge = (c == 0 and dc < 0) or (c == w - 1 and dc > 0)
        if hit_horizontal_edge or hit_vertical_edge:
            if hit_horizontal_edge:
                dr = -dr
            if hit_vertical_edge:
                dc = -dc
            bounces += 1
            if advance_on_bounce:
                idx = (idx + 1) % len(cycle)
            if (r, c) != start:
                out[r, c] = cycle[idx]

        nr = r + dr
        nc = c + dc
        if not (0 <= nr < h and 0 <= nc < w):
            reason = "escape"
            break
        if obstacle_mask[nr, nc]:
            reason = "object"
            break
        if stop_on_painted and out[nr, nc] != BG:
            reason = "painted"
            break

        out[nr, nc] = cycle[idx]
        r, c = nr, nc
    else:
        reason = "limit"

    return out, {
        "bounces": bounces,
        "reason": reason,
        "painted": int(np.count_nonzero(out)),
        "used_colors": {int(v) for v in np.unique(out) if int(v) != BG},
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Trace a bouncing particle whose paint cycle is determined by segment-length order.

    Input structure:
    - Black background with 3-4 monochrome straight control segments near the top, each with a distinct length and distinct color.
    - A single red start pixel in the lower half and a single grey adjacent indicator pixel on one of its four sides.
    - The segment colors are the only reusable paint colors; every other non-black pixel is a control marker or an obstacle.

    Rule:
    - Read every straight control segment, sort them by decreasing length, and use the resulting colors as the paint cycle.
    - Launch one particle from the red start; the grey pixel selects which reflected diagonal heading is used first.
    - The particle paints every empty cell it enters with the current cycle color.
    - Whenever the particle is about to leave the grid, it reflects off the touched border, advances to the next cycle color, and recolors that border collision cell with the new color.
    - The particle stops when its next cell is already non-background or has already been painted; then erase the control segments and the two markers, leaving only the painted trajectory.

    Key insight:
    - The top segments are not obstacles to copy; their distinct lengths define the runtime color order, so changing segment lengths changes the whole bounced path coloring even when the geometry stays the same.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    cycle = _extract_cycle(g)
    start, indicator, heading = _find_start_and_heading(g)

    obstacle_grid = g.copy()
    obstacle_grid[start] = BG
    obstacle_grid[indicator] = BG
    out, _meta = _simulate(obstacle_grid, cycle, start, heading)
    return out


def _segment_cells(r0: int, c0: int, length: int, orientation: str) -> list[tuple[int, int]]:
    if orientation == "H":
        return [(r0, c0 + k) for k in range(length)]
    return [(r0 + k, c0) for k in range(length)]


def _expanded(cells: Iterable[tuple[int, int]], shape: tuple[int, int]) -> set[tuple[int, int]]:
    h, w = shape
    out: set[tuple[int, int]] = set()
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr = r + dr
                nc = c + dc
                if 0 <= nr < h and 0 <= nc < w:
                    out.add((nr, nc))
    return out


def _place_segments(
    h: int,
    w: int,
    path_mask: np.ndarray,
    start: tuple[int, int],
    indicator: tuple[int, int],
    lengths: list[int],
    colors: list[int],
    orientations: list[str],
    rng: random.Random,
) -> np.ndarray | None:
    grid = np.zeros((h, w), dtype=int)
    blocked = {(int(r), int(c)) for r, c in np.argwhere(path_mask > 0)}
    blocked |= _expanded([start, indicator], (h, w))
    reserved = set(blocked)

    order = sorted(range(len(lengths)), key=lambda i: lengths[i], reverse=True)
    for idx in order:
        length = lengths[idx]
        color = colors[idx]
        orientation = orientations[idx]
        candidates: list[list[tuple[int, int]]] = []
        if orientation == "H":
            for r0 in range(3):
                for c0 in range(w - length + 1):
                    cells = _segment_cells(r0, c0, length, orientation)
                    if all(cell not in reserved for cell in cells):
                        candidates.append(cells)
        else:
            for r0 in range(3):
                if r0 + length > h:
                    continue
                for c0 in range(w):
                    cells = _segment_cells(r0, c0, length, orientation)
                    if all(cell not in reserved for cell in cells):
                        candidates.append(cells)
        rng.shuffle(candidates)
        if not candidates:
            return None
        cells = candidates[0]
        for r, c in cells:
            grid[r, c] = color
        reserved |= _expanded(cells, (h, w))
    return grid


def _sample_orientations(count: int, branch: int, rng: random.Random) -> list[str]:
    orientations = ["H" if rng.random() < 0.5 else "V" for _ in range(count)]
    if branch % 2 == 0:
        orientations[0] = "H"
        if count > 1:
            orientations[1] = "V"
    if branch % 4 == 3:
        orientations = ["V"] * count
        orientations[0] = "H"
    return orientations


def _wrong_cycle_outputs(
    obstacle_grid: np.ndarray,
    cycle: list[int],
    start: tuple[int, int],
    heading: tuple[int, int],
) -> list[np.ndarray]:
    asc_cycle = list(reversed(cycle))
    alt1, _ = _simulate(obstacle_grid, asc_cycle, start, heading)
    alt2, _ = _simulate(
        obstacle_grid,
        cycle,
        start,
        heading,
        advance_on_bounce=False,
        stop_on_painted=True,
    )
    alt3, _ = _simulate(
        obstacle_grid,
        cycle,
        start,
        heading,
        advance_on_bounce=True,
        stop_on_painted=False,
    )
    return [alt1, alt2, alt3]


def _build_attempt(rng: random.Random, branch: int) -> np.ndarray | None:
    h = rng.randint(12, 18)
    w = rng.randint(12, 18)

    count = 3 if branch % 3 == 0 else 4
    lengths = rng.sample(list(range(2, 7)), count)
    colors = rng.sample(list(SEGMENT_PALETTE), count)
    side = INDICATOR_SIDES[branch % 4]
    heading = INDICATOR_TO_HEADING[side]

    start = (
        rng.randint(max(h // 2, 4), h - 3),
        rng.randint(2, w - 3),
    )
    indicator = (start[0] + side[0], start[1] + side[1])
    if not (0 <= indicator[0] < h and 0 <= indicator[1] < w):
        return None

    cycle = [color for _length, color in sorted(zip(lengths, colors), reverse=True)]
    blank = np.zeros((h, w), dtype=int)
    expected, meta = _simulate(blank, cycle, start, heading)
    if meta["reason"] != "painted" or int(meta["bounces"]) < 3 or int(meta["painted"]) < 12:
        return None
    if meta["used_colors"] != set(cycle):
        return None

    orientations = _sample_orientations(count, branch, rng)
    segment_grid = _place_segments(h, w, expected, start, indicator, lengths, colors, orientations, rng)
    if segment_grid is None:
        return None

    grid = segment_grid.copy()
    grid[start] = START
    grid[indicator] = INDICATOR

    actual = main(grid)
    if not np.array_equal(actual, expected):
        return None

    obstacle_grid = grid.copy()
    obstacle_grid[start] = BG
    obstacle_grid[indicator] = BG
    for wrong in _wrong_cycle_outputs(obstacle_grid, cycle, start, heading):
        if np.array_equal(wrong, expected):
            return None

    top_band = grid[:3]
    if np.count_nonzero(top_band == BG) == 0:
        return None
    return grid


def generate_input() -> np.ndarray:
    global _GEN_COUNTER

    rng = random.Random()
    for attempt in range(GEN_ATTEMPTS):
        branch = (_GEN_COUNTER + attempt) % 8
        grid = _build_attempt(rng, branch)
        if grid is not None:
            _GEN_COUNTER += 1
            return grid
    raise RuntimeError("generate_input: failed to sample a valid puzzle")
