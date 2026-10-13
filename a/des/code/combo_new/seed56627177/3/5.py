from __future__ import annotations

import random

import numpy as np


BG = 0
YELLOW = 4
SEGMENT_COLORS = [1, 2, 3, 5, 6, 7, 8, 9]
DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
GRID_H_RANGE = (16, 29)
GRID_W_RANGE = (16, 29)
SEGMENT_COUNT_RANGE = (3, 5)
SEGMENT_LENGTH_RANGE = (2, 5)
GEN_ATTEMPTS = 600


def _most_common_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[int(np.argmax(counts))])


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= int(r) < int(shape[0]) and 0 <= int(c) < int(shape[1])


def _sign(x: int) -> int:
    return 1 if int(x) > 0 else -1 if int(x) < 0 else 0


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(int(r), int(c))]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((int(rr), int(cc)))
                for dr, dc in DIRS:
                    nr = int(rr + dr)
                    nc = int(cc + dc)
                    if _in_bounds(mask.shape, nr, nc) and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((int(nr), int(nc)))
            out.append(np.array(coords, dtype=int))
    return out


def _perpendicular_offsets(direction: tuple[int, int]) -> list[tuple[int, int]]:
    if int(direction[0]) == 0:
        return [(-1, 0), (1, 0)]
    return [(0, -1), (0, 1)]


def _line_cells(start: tuple[int, int], end: tuple[int, int]) -> list[tuple[int, int]]:
    r0, c0 = map(int, start)
    r1, c1 = map(int, end)
    if (r0, c0) == (r1, c1):
        return []
    assert r0 == r1 or c0 == c1, "line must be orthogonal"
    dr = _sign(r1 - r0)
    dc = _sign(c1 - c0)
    assert (dr == 0) ^ (dc == 0)

    out: list[tuple[int, int]] = []
    rr = int(r0 + dr)
    cc = int(c0 + dc)
    while True:
        out.append((int(rr), int(cc)))
        if (rr, cc) == (r1, c1):
            return out
        rr += int(dr)
        cc += int(dc)


def _connection_cells(
    src_exit: tuple[int, int],
    src_dir: tuple[int, int],
    dst_approach: tuple[int, int],
) -> list[tuple[int, int]]:
    sr, sc = map(int, src_exit)
    pr, pc = map(int, dst_approach)
    dr, dc = map(int, src_dir)

    if dr == 0:
        assert pr != sr, "connector must bend"
        assert (pc - sc) * dc > 0, "target approach must lie ahead of the source ray"
        turn = (int(sr), int(pc))
    else:
        assert pc != sc, "connector must bend"
        assert (pr - sr) * dr > 0, "target approach must lie ahead of the source ray"
        turn = (int(pr), int(sc))

    part1 = _line_cells((int(sr), int(sc)), turn)
    part2 = _line_cells(turn, (int(pr), int(pc)))
    assert part1 and part2, "each connector needs a straight extension and a turned leg"
    return part1 + part2


def _infer_marker_color(grid: np.ndarray, bg: int) -> int:
    for color in map(int, np.unique(grid)):
        if int(color) == int(bg):
            continue
        comps = _components(grid == int(color))
        if comps and all(int(comp.shape[0]) == 1 for comp in comps):
            return int(color)
    raise AssertionError("unable to infer marker color")


def _parse_segment(grid: np.ndarray, color: int, marker_color: int, bg: int) -> dict[str, object]:
    comps = _components(grid == int(color))
    assert len(comps) == 2, "each segment color must form a legend singleton and one segment"

    singleton = next((comp for comp in comps if int(comp.shape[0]) == 1), None)
    line = next((comp for comp in comps if int(comp.shape[0]) > 1), None)
    assert singleton is not None and line is not None

    coords = [tuple(map(int, coord)) for coord in line.tolist()]
    rows = sorted({int(r) for r, _c in coords})
    cols = sorted({int(c) for _r, c in coords})
    size = len(coords)
    assert 2 <= size <= 5

    if len(rows) == 1:
        r = int(rows[0])
        c0 = int(cols[0])
        c1 = int(cols[-1])
        assert cols == list(range(c0, c1 + 1)), "horizontal segment must be contiguous"
        ep_a = (int(r), int(c0))
        ep_b = (int(r), int(c1))
        side_offsets = [(-1, 0), (1, 0)]
    else:
        assert len(cols) == 1, "segment must be horizontal or vertical"
        c = int(cols[0])
        r0 = int(rows[0])
        r1 = int(rows[-1])
        assert rows == list(range(r0, r1 + 1)), "vertical segment must be contiguous"
        ep_a = (int(r0), int(c))
        ep_b = (int(r1), int(c))
        side_offsets = [(0, -1), (0, 1)]

    def endpoint_markers(endpoint: tuple[int, int]) -> list[tuple[int, int]]:
        er, ec = map(int, endpoint)
        out: list[tuple[int, int]] = []
        for mr, mc in side_offsets:
            rr = int(er + mr)
            cc = int(ec + mc)
            if _in_bounds(grid.shape, rr, cc) and int(grid[rr, cc]) == int(marker_color):
                out.append((int(mr), int(mc)))
        return out

    marks_a = endpoint_markers(ep_a)
    marks_b = endpoint_markers(ep_b)
    assert len(marks_a) <= 1 and len(marks_b) <= 1
    assert bool(marks_a) ^ bool(marks_b), "exactly one endpoint must carry the yellow side marker"

    if marks_a:
        entrance = ep_a
        exit_pt = ep_b
        marker_offset = marks_a[0]
    else:
        entrance = ep_b
        exit_pt = ep_a
        marker_offset = marks_b[0]

    direction = (_sign(int(exit_pt[0]) - int(entrance[0])), _sign(int(exit_pt[1]) - int(entrance[1])))
    assert direction in DIRS

    approach = (int(entrance[0] - marker_offset[0]), int(entrance[1] - marker_offset[1]))
    assert _in_bounds(grid.shape, *approach)
    assert int(grid[approach]) == int(bg), "the free side opposite the marker must stay empty in the input"

    legend = tuple(map(int, singleton[0]))
    return {
        "color": int(color),
        "legend": legend,
        "entrance": entrance,
        "exit": exit_pt,
        "dir": direction,
        "marker_offset": marker_offset,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect the ordered colored line fragments with one-bend rays that respect each next fragment's side marker.

    Input structure:
    - The background is black.
    - Each non-yellow foreground color appears twice: once as a singleton in a legend column near the upper-left corner and once as a straight 1-pixel-wide horizontal or vertical segment elsewhere.
    - Every segment has one yellow side marker beside exactly one endpoint; that marked endpoint is the segment's entrance, while the opposite endpoint is its exit.

    Rule:
    - Read the legend singletons from top to bottom to get the chaining order.
    - For each consecutive pair, extend a ray from the first segment's exit in the direction away from its entrance.
    - Make exactly one right-angle turn, then stop in the background cell adjacent to the next segment's entrance on the side opposite its yellow marker; paint the whole connector with the first segment's color.
    - Leave the legend pixels, yellow markers, and original segment pixels unchanged.

    Key insight:
    - The legend gives the global order, while each yellow side marker fixes which perpendicular side of the next endpoint the elbow must reach.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = int(_most_common_color(g))
    marker_color = int(YELLOW) if bool((g == int(YELLOW)).any()) else _infer_marker_color(g, bg)

    colors = [int(c) for c in np.unique(g) if int(c) not in (int(bg), int(marker_color))]
    assert colors, "expected at least one colored segment"

    info_by_color = {
        int(color): _parse_segment(g, int(color), int(marker_color), int(bg))
        for color in colors
    }

    legend_positions = [tuple(info_by_color[int(color)]["legend"]) + (int(color),) for color in colors]
    legend_cols = {int(c) for _r, c, _color in legend_positions}
    assert len(legend_cols) == 1, "legend pixels must align in one column"
    order = [int(color) for _r, _c, color in sorted(legend_positions)]
    assert len(order) >= 2

    out = g.copy()
    for src_color, dst_color in zip(order, order[1:]):
        src = info_by_color[int(src_color)]
        dst = info_by_color[int(dst_color)]
        approach = (
            int(dst["entrance"][0] - dst["marker_offset"][0]),
            int(dst["entrance"][1] - dst["marker_offset"][1]),
        )
        cells = _connection_cells(
            tuple(src["exit"]),
            tuple(src["dir"]),
            approach,
        )
        for r, c in cells:
            assert _in_bounds(out.shape, r, c)
            assert int(out[r, c]) == int(bg), "connectors may only occupy background cells"
            out[r, c] = int(src_color)

    return out


def _segment_cells(
    entrance: tuple[int, int],
    direction: tuple[int, int],
    length: int,
) -> list[tuple[int, int]]:
    er, ec = map(int, entrance)
    dr, dc = map(int, direction)
    return [(int(er + k * dr), int(ec + k * dc)) for k in range(int(length))]


def _build_segment(
    entrance: tuple[int, int],
    direction: tuple[int, int],
    length: int,
    marker_offset: tuple[int, int],
) -> dict[str, object]:
    cells = _segment_cells(entrance, direction, length)
    exit_pt = cells[-1]
    marker = (int(entrance[0] + marker_offset[0]), int(entrance[1] + marker_offset[1]))
    approach = (int(entrance[0] - marker_offset[0]), int(entrance[1] - marker_offset[1]))
    return {
        "entrance": tuple(map(int, entrance)),
        "exit": tuple(map(int, exit_pt)),
        "dir": tuple(map(int, direction)),
        "length": int(length),
        "marker_offset": tuple(map(int, marker_offset)),
        "marker": marker,
        "approach": approach,
        "cells": [tuple(map(int, cell)) for cell in cells],
    }


def _marker_is_clear(
    marker: tuple[int, int],
    existing_markers: set[tuple[int, int]],
) -> bool:
    mr, mc = map(int, marker)
    for rr, cc in existing_markers:
        if abs(int(mr - rr)) + abs(int(mc - cc)) <= 1:
            return False
    return True


def _can_place_segment(
    segment: dict[str, object],
    shape: tuple[int, int],
    blocked: set[tuple[int, int]],
    existing_markers: set[tuple[int, int]],
    *,
    require_approach_clear: bool,
    require_forward_clear: bool,
) -> bool:
    cells = [tuple(cell) for cell in segment["cells"]]
    marker = tuple(segment["marker"])
    approach = tuple(segment["approach"])
    exit_pt = tuple(segment["exit"])
    direction = tuple(segment["dir"])

    for r, c in cells + [marker]:
        if not _in_bounds(shape, r, c) or (int(r), int(c)) in blocked:
            return False
    if not _in_bounds(shape, *approach):
        return False
    if require_approach_clear and tuple(approach) in blocked:
        return False
    if not _marker_is_clear(marker, existing_markers):
        return False

    if require_forward_clear:
        front = (int(exit_pt[0] + direction[0]), int(exit_pt[1] + direction[1]))
        if not _in_bounds(shape, *front) or front in blocked:
            return False

    return True


def _all_segment_specs() -> list[tuple[tuple[int, int], int, tuple[int, int]]]:
    specs: list[tuple[tuple[int, int], int, tuple[int, int]]] = []
    for direction in DIRS:
        for length in range(SEGMENT_LENGTH_RANGE[0], SEGMENT_LENGTH_RANGE[1] + 1):
            for marker_offset in _perpendicular_offsets(direction):
                specs.append((tuple(direction), int(length), tuple(marker_offset)))
    random.shuffle(specs)
    return specs


def _legend_guard(n_segments: int) -> set[tuple[int, int]]:
    return {
        (int(r), int(c))
        for r in range(int(n_segments) + 1)
        for c in range(3)
    }


def _place_first_segment(
    shape: tuple[int, int],
    blocked: set[tuple[int, int]],
    existing_markers: set[tuple[int, int]],
) -> dict[str, object] | None:
    h, w = map(int, shape)
    positions = [(int(r), int(c)) for r in range(h) for c in range(w)]
    random.shuffle(positions)

    for direction, length, marker_offset in _all_segment_specs():
        for entrance in positions:
            segment = _build_segment(entrance, direction, length, marker_offset)
            if _can_place_segment(
                segment,
                shape,
                blocked,
                existing_markers,
                require_approach_clear=True,
                require_forward_clear=True,
            ):
                return segment
    return None


def _approach_candidates(
    shape: tuple[int, int],
    prev_exit: tuple[int, int],
    prev_dir: tuple[int, int],
) -> list[tuple[int, int]]:
    h, w = map(int, shape)
    pr, pc = map(int, prev_exit)
    dr, dc = map(int, prev_dir)

    out: list[tuple[int, int]] = []
    if dr == 0:
        cols = list(range(int(pc + dc), w if dc > 0 else -1, int(dc)))
        rows = [int(r) for r in range(h) if int(r) != int(pr)]
        random.shuffle(cols)
        random.shuffle(rows)
        for c in cols:
            for r in rows:
                out.append((int(r), int(c)))
    else:
        rows = list(range(int(pr + dr), h if dr > 0 else -1, int(dr)))
        cols = [int(c) for c in range(w) if int(c) != int(pc)]
        random.shuffle(rows)
        random.shuffle(cols)
        for r in rows:
            for c in cols:
                out.append((int(r), int(c)))
    return out


def _place_next_segment(
    shape: tuple[int, int],
    blocked: set[tuple[int, int]],
    existing_markers: set[tuple[int, int]],
    prev_segment: dict[str, object],
    *,
    require_forward_clear: bool,
) -> tuple[dict[str, object], list[tuple[int, int]]] | None:
    prev_exit = tuple(prev_segment["exit"])
    prev_dir = tuple(prev_segment["dir"])
    approaches = _approach_candidates(shape, prev_exit, prev_dir)

    for direction, length, marker_offset in _all_segment_specs():
        for approach in approaches:
            try:
                path_cells = _connection_cells(prev_exit, prev_dir, approach)
            except AssertionError:
                continue
            if any((int(r), int(c)) in blocked or not _in_bounds(shape, r, c) for r, c in path_cells):
                continue

            entrance = (int(approach[0] + marker_offset[0]), int(approach[1] + marker_offset[1]))
            segment = _build_segment(entrance, direction, length, marker_offset)
            if not _can_place_segment(
                segment,
                shape,
                blocked,
                existing_markers,
                require_approach_clear=False,
                require_forward_clear=require_forward_clear,
            ):
                continue

            visible = set(tuple(cell) for cell in segment["cells"])
            visible.add(tuple(segment["marker"]))
            if any(tuple(cell) in visible for cell in path_cells):
                continue
            return segment, path_cells
    return None


def _render_input(
    shape: tuple[int, int],
    colors: list[int],
    segments: list[dict[str, object]],
) -> np.ndarray:
    h, w = map(int, shape)
    grid = np.full((h, w), BG, dtype=int)

    for row, color in enumerate(colors):
        grid[int(row), 0] = int(color)

    for color, segment in zip(colors, segments):
        for r, c in segment["cells"]:
            grid[int(r), int(c)] = int(color)
        mr, mc = map(int, segment["marker"])
        grid[int(mr), int(mc)] = int(YELLOW)

    return grid


def _gen_attempt() -> np.ndarray | None:
    n_segments = random.randint(SEGMENT_COUNT_RANGE[0], SEGMENT_COUNT_RANGE[1])
    colors = random.sample(SEGMENT_COLORS, n_segments)

    min_h = max(GRID_H_RANGE[0], n_segments + 6)
    h = random.randint(min_h, GRID_H_RANGE[1])
    w = random.randint(GRID_W_RANGE[0], GRID_W_RANGE[1])
    shape = (int(h), int(w))

    blocked = set(_legend_guard(n_segments))
    marker_cells: set[tuple[int, int]] = set()
    segments: list[dict[str, object]] = []

    first = _place_first_segment(shape, blocked, marker_cells)
    if first is None:
        return None
    segments.append(first)
    blocked.update(tuple(cell) for cell in first["cells"])
    blocked.add(tuple(first["marker"]))
    marker_cells.add(tuple(first["marker"]))

    for idx in range(1, n_segments):
        placed = _place_next_segment(
            shape,
            blocked,
            marker_cells,
            segments[-1],
            require_forward_clear=idx < n_segments - 1,
        )
        if placed is None:
            return None
        segment, path_cells = placed
        blocked.update(path_cells)
        blocked.update(tuple(cell) for cell in segment["cells"])
        blocked.add(tuple(segment["marker"]))
        marker_cells.add(tuple(segment["marker"]))
        segments.append(segment)

    grid = _render_input(shape, colors, segments)

    try:
        out = main(grid)
    except Exception:
        return None

    if np.array_equal(out, grid):
        return None
    if out.shape != grid.shape:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input: failed after {GEN_ATTEMPTS} attempts")
