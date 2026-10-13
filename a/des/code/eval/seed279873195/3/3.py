from __future__ import annotations

import random
from typing import Iterable

import numpy as np


BG = 0
INTERSECTION = 6
TRACK_COLORS = (1, 2, 3, 4, 5, 7, 8, 9)
TRACK_SPLITS = ((1, 1), (1, 2), (2, 1), (2, 2), (1, 3), (3, 1))
POSITION_GAP = 3
GEN_RETRIES = 400


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _read_legend(grid: np.ndarray) -> tuple[int, dict[tuple[int, int], int]]:
    h, w = grid.shape
    assert w >= 4, "grid must be wide enough to hold the legend"

    rows: list[tuple[int, int, int]] = []
    for r in range(h - 1, -1, -1):
        if np.all(grid[r, :3] != BG) and np.all(grid[r, 3:] == BG):
            rows.append((int(grid[r, 0]), int(grid[r, 1]), int(grid[r, 2])))
        elif rows:
            break

    assert rows, "expected a bottom-left 3-column legend"
    rows.reverse()
    legend_top = h - len(rows)
    if legend_top > 0:
        assert np.all(grid[legend_top - 1] == BG), "legend should be separated from the tracks"

    mapping: dict[tuple[int, int], int] = {}
    for signal_color, cross_color, result_color in rows:
        assert signal_color != BG and cross_color != BG and result_color != BG
        key = (signal_color, cross_color)
        if key in mapping:
            assert mapping[key] == result_color, "legend rows must not disagree"
        mapping[key] = result_color
    return legend_top, mapping


def _classify_source(
    grid: np.ndarray,
    top: int,
    left: int,
    endpoint: tuple[int, int],
    color: int,
) -> dict[str, object] | None:
    er, ec = endpoint

    if ec == left + 2 and top <= er <= top + 1:
        step = (0, 1)
        offset = top - er
    elif ec == left - 1 and top <= er <= top + 1:
        step = (0, -1)
        offset = top - er
    elif er == top + 2 and left <= ec <= left + 1:
        step = (1, 0)
        offset = left - ec
    elif er == top - 1 and left <= ec <= left + 1:
        step = (-1, 0)
        offset = left - ec
    else:
        return None

    nr = er + step[0]
    nc = ec + step[1]
    assert _in_bounds(grid.shape, nr, nc), "track must continue away from the source block"
    assert int(grid[nr, nc]) in {color, INTERSECTION}, "source block must attach to a straight track"
    return {
        "color": int(color),
        "start": (int(er), int(ec)),
        "step": step,
        "offset": int(offset),
    }


def _find_sources(grid: np.ndarray, legend_top: int) -> list[dict[str, object]]:
    h, w = grid.shape
    sources: list[dict[str, object]] = []
    used: set[tuple[int, int]] = set()

    for r in range(0, legend_top - 1):
        for c in range(0, w - 1):
            block_cells = {
                (r, c),
                (r + 1, c),
                (r, c + 1),
                (r + 1, c + 1),
            }
            if block_cells & used:
                continue

            color = int(grid[r, c])
            if color in {BG, INTERSECTION}:
                continue
            if not np.all(grid[r : r + 2, c : c + 2] == color):
                continue

            outside: set[tuple[int, int]] = set()
            for rr in (r, r + 1):
                for cc in (c - 1, c + 2):
                    if _in_bounds(grid.shape, rr, cc) and int(grid[rr, cc]) == color:
                        outside.add((rr, cc))
            for cc in (c, c + 1):
                for rr in (r - 1, r + 2):
                    if _in_bounds(grid.shape, rr, cc) and int(grid[rr, cc]) == color:
                        outside.add((rr, cc))

            if len(outside) != 1:
                continue

            endpoint = next(iter(outside))
            source = _classify_source(grid, r, c, endpoint, color)
            if source is None:
                continue

            used |= block_cells
            sources.append(source)

    assert sources, "expected at least one source block"
    return sources


def _trace_track(
    grid: np.ndarray,
    color: int,
    start: tuple[int, int],
    step: tuple[int, int],
) -> tuple[list[tuple[int, int]], tuple[int, int]]:
    r, c = start
    dr, dc = step
    path: list[tuple[int, int]] = []

    while True:
        assert _in_bounds(grid.shape, r, c)
        value = int(grid[r, c])
        assert value in {color, INTERSECTION}, "track cells must keep the rail color or the shared intersection color"
        path.append((int(r), int(c)))

        nr = r + dr
        nc = c + dc
        if not _in_bounds(grid.shape, nr, nc) or int(grid[nr, nc]) == BG:
            return path, (int(r), int(c))

        assert int(grid[nr, nc]) in {color, INTERSECTION}, "track must stay straight"
        r, c = nr, nc


def _cross_color(grid: np.ndarray, pos: tuple[int, int], step: tuple[int, int]) -> int:
    r, c = pos
    dr, dc = step
    neighbors: Iterable[tuple[int, int]]
    if dr == 0:
        neighbors = ((r - 1, c), (r + 1, c))
    else:
        neighbors = ((r, c - 1), (r, c + 1))

    colors = {
        int(grid[nr, nc])
        for nr, nc in neighbors
        if _in_bounds(grid.shape, nr, nc) and int(grid[nr, nc]) not in {BG, INTERSECTION}
    }
    assert len(colors) == 1, "each intersection should reveal exactly one perpendicular track color"
    return next(iter(colors))


def _terminal_block_origin(
    terminal: tuple[int, int],
    step: tuple[int, int],
    offset: int,
) -> tuple[int, int]:
    r, c = terminal
    if step == (0, 1):
        return r + offset, c + 1
    if step == (0, -1):
        return r + offset, c - 2
    if step == (1, 0):
        return r + 1, c + offset
    assert step == (-1, 0)
    return r - 2, c + offset


def _paint_block(
    out: np.ndarray,
    inp: np.ndarray,
    top: int,
    left: int,
    color: int,
) -> None:
    assert 0 <= top and 0 <= left
    assert top + 1 < out.shape[0] and left + 1 < out.shape[1]
    assert np.all(inp[top : top + 2, left : left + 2] == BG), "terminal markers should be placed on empty space"
    assert np.all(out[top : top + 2, left : left + 2] == BG), "terminal markers must not overlap"
    out[top : top + 2, left : left + 2] = int(color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Propagate one signal along each colored rail, recolor it at magenta intersections,
    and place the final 2x2 marker at the rail's empty endpoint.
    """
    g = np.asarray(grid, dtype=int)
    legend_top, mapping = _read_legend(g)
    sources = _find_sources(g, legend_top)

    out = g.copy()
    for source in sources:
        track_color = int(source["color"])
        start = source["start"]
        step = source["step"]
        offset = int(source["offset"])

        path, terminal = _trace_track(g, track_color, start, step)
        signal_color = track_color
        for pos in path:
            if int(g[pos]) != INTERSECTION:
                continue
            other_color = _cross_color(g, pos, step)
            key = (signal_color, other_color)
            assert key in mapping, "legend must cover every encountered transformation"
            signal_color = mapping[key]

        top, left = _terminal_block_origin(terminal, step, offset)
        _paint_block(out, g, top, left, signal_color)

    return out


def _sample_spaced_positions(
    count: int,
    lo: int,
    hi: int,
    gap: int,
    rng: random.Random,
) -> list[int] | None:
    span = hi - lo + 1
    need = count + (count - 1) * gap
    if span < need:
        return None
    compressed = span - (count - 1) * gap
    picks = sorted(rng.sample(range(compressed), count))
    return [lo + picks[i] + i * gap for i in range(count)]


def _build_mapping(colors: list[int], rng: random.Random) -> tuple[list[tuple[int, int, int]], dict[tuple[int, int], int]]:
    rows = [(a, b, rng.choice(colors)) for a in colors for b in colors]
    rng.shuffle(rows)
    mapping = {(a, b): out for a, b, out in rows}
    return rows, mapping


def _place_horizontal_source(
    grid: np.ndarray,
    row: int,
    color: int,
    c0: int,
    c1: int,
    side: str,
    offset: int,
) -> None:
    top = row + offset
    if side == "L":
        left = c0 - 2
    else:
        left = c1 + 1
    assert np.all(grid[top : top + 2, left : left + 2] == BG)
    grid[top : top + 2, left : left + 2] = color


def _place_vertical_source(
    grid: np.ndarray,
    col: int,
    color: int,
    r0: int,
    r1: int,
    side: str,
    offset: int,
) -> None:
    left = col + offset
    if side == "T":
        top = r0 - 2
    else:
        top = r1 + 1
    assert np.all(grid[top : top + 2, left : left + 2] == BG)
    grid[top : top + 2, left : left + 2] = color


def _expected_output(
    grid: np.ndarray,
    mapping: dict[tuple[int, int], int],
    horizontals: list[dict[str, int | str]],
    verticals: list[dict[str, int | str]],
) -> np.ndarray:
    out = grid.copy()

    for track in horizontals:
        signal = int(track["color"])
        direction = int(track["dir"])
        ordered_verticals = sorted(verticals, key=lambda item: int(item["col"]), reverse=direction < 0)
        for other in ordered_verticals:
            signal = mapping[(signal, int(other["color"]))]
        terminal = (int(track["row"]), int(track["c1"] if direction > 0 else track["c0"]))
        top = int(track["row"]) + int(track["offset"])
        left = int(track["c1"]) + 1 if direction > 0 else int(track["c0"]) - 2
        _paint_block(out, grid, top, left, signal)

    for track in verticals:
        signal = int(track["color"])
        direction = int(track["dir"])
        ordered_horizontals = sorted(horizontals, key=lambda item: int(item["row"]), reverse=direction < 0)
        for other in ordered_horizontals:
            signal = mapping[(signal, int(other["color"]))]
        terminal = (int(track["r1"] if direction > 0 else track["r0"]), int(track["col"]))
        top = int(track["r1"]) + 1 if direction > 0 else int(track["r0"]) - 2
        left = int(track["col"]) + int(track["offset"])
        _paint_block(out, grid, top, left, signal)

    return out


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    n_h, n_v = rng.choice(TRACK_SPLITS)
    n_tracks = n_h + n_v
    colors = rng.sample(TRACK_COLORS, n_tracks)
    legend_rows, mapping = _build_mapping(colors, rng)

    legend_h = len(legend_rows)
    height = rng.randint(legend_h + 13, legend_h + 18)
    width = rng.randint(22, 32)
    legend_top = height - legend_h

    row_lo, row_hi = 3, legend_top - 4
    col_lo, col_hi = 6, width - 7
    rows = _sample_spaced_positions(n_h, row_lo, row_hi, POSITION_GAP, rng)
    cols = _sample_spaced_positions(n_v, col_lo, col_hi, POSITION_GAP, rng)
    if rows is None or cols is None:
        return None

    grid = np.zeros((height, width), dtype=int)
    h_c0, h_c1 = 3, width - 4
    v_r0, v_r1 = 2, legend_top - 4

    horizontals: list[dict[str, int | str]] = []
    for idx, row in enumerate(rows):
        color = colors[idx]
        grid[row, h_c0 : h_c1 + 1] = color
        side = rng.choice(("L", "R"))
        offset = rng.choice((-1, 0))
        _place_horizontal_source(grid, row, color, h_c0, h_c1, side, offset)
        horizontals.append(
            {
                "row": row,
                "color": color,
                "c0": h_c0,
                "c1": h_c1,
                "dir": 1 if side == "L" else -1,
                "offset": offset,
            }
        )

    verticals: list[dict[str, int | str]] = []
    for idx, col in enumerate(cols):
        color = colors[n_h + idx]
        grid[v_r0 : v_r1 + 1, col] = color
        side = rng.choice(("T", "B"))
        offset = rng.choice((-1, 0))
        _place_vertical_source(grid, col, color, v_r0, v_r1, side, offset)
        verticals.append(
            {
                "col": col,
                "color": color,
                "r0": v_r0,
                "r1": v_r1,
                "dir": 1 if side == "T" else -1,
                "offset": offset,
            }
        )

    for row in rows:
        for col in cols:
            grid[row, col] = INTERSECTION

    for i, (a, b, c) in enumerate(legend_rows):
        grid[legend_top + i, 0] = a
        grid[legend_top + i, 1] = b
        grid[legend_top + i, 2] = c

    try:
        expected = _expected_output(grid, mapping, horizontals, verticals)
        actual = main(grid)
    except Exception:
        return None

    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid rail-signal puzzle")
