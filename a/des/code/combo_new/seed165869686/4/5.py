import random
from collections import deque

import numpy as np


BLACK = 0
GREY = 5
COLOR_POOL = (1, 2, 3, 4, 6, 7, 8, 9)
DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]

MAX_ATTEMPTS = 400
SHAPE_ATTEMPTS = 120
PLACEMENT_ATTEMPTS = 120


def _neighbors4(r: int, c: int, h: int, w: int) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for dr, dc in DIRS:
        nr, nc = r + dr, c + dc
        if 0 <= nr < h and 0 <= nc < w:
            out.append((nr, nc))
    return out


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = map(int, mask.shape)
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            comp: list[tuple[int, int]] = []
            q = deque([(r, c)])
            seen[r, c] = True
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc, h, w):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _is_adjacent(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1


def _ordered_path(
    component: list[tuple[int, int]],
    source: tuple[int, int],
    sink: tuple[int, int],
    h: int,
    w: int,
) -> list[tuple[int, int]]:
    comp_set = set(component)
    start_cells = [cell for cell in _neighbors4(source[0], source[1], h, w) if cell in comp_set]
    end_cells = {cell for cell in _neighbors4(sink[0], sink[1], h, w) if cell in comp_set}
    assert len(start_cells) == 1
    assert len(end_cells) == 1

    seq = [source]
    prev = source
    cur = start_cells[0]
    while True:
        seq.append(cur)
        next_cells = [cell for cell in _neighbors4(cur[0], cur[1], h, w) if cell in comp_set and cell != prev]
        if cur in end_cells:
            assert not next_cells
            break
        assert len(next_cells) == 1
        prev, cur = cur, next_cells[0]

    seq.append(sink)
    assert len(seq) == len(component) + 2
    return seq


def _count_turns(seq: list[tuple[int, int]]) -> int:
    axes: list[int] = []
    for (r0, c0), (r1, c1) in zip(seq, seq[1:]):
        dr = r1 - r0
        dc = c1 - c0
        assert abs(dr) + abs(dc) == 1
        axes.append(0 if dr != 0 else 1)
    return sum(a0 != a1 for a0, a1 in zip(axes, axes[1:]))


def _paint_sink_block(
    out: np.ndarray,
    sink: tuple[int, int],
    source_color: int,
    sink_color: int,
    turns: int,
) -> None:
    h, w = map(int, out.shape)
    sr, sc = sink
    for rr in range(sr - 1, sr + 2):
        if not 0 <= rr < h:
            continue
        for cc in range(sc - 1, sc + 2):
            if 0 <= cc < w:
                out[rr, cc] = source_color

    if turns == 1:
        out[sr, sc] = sink_color
    elif turns >= 2:
        for dr, dc in ((0, 0), (-1, -1), (-1, 1), (1, -1), (1, 1)):
            rr, cc = sr + dr, sc + dc
            if 0 <= rr < h and 0 <= cc < w:
                out[rr, cc] = sink_color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Follow each grey path from its tailed source to its sink, recolor the path with the source color, and expand the sink into a bend-dependent 3x3 motif.

    Input structure:
    - The background is black.
    - Each object contains a one-pixel-wide grey orthogonal path and two non-grey endpoint pixels of different colors.
    - Exactly one endpoint has an extra same-color neighbor attached as a one-pixel tail; that tailed endpoint is the source.
    - Paths are disjoint and do not share pixels.

    Rule:
    - Repaint every grey path pixel with its source endpoint color.
    - At the sink endpoint, draw a 3x3 block centered on the sink, clipped by the grid border, using the source color.
    - Count the number of turns in the source-to-sink path: keep the block solid for 0 turns, restore the sink-color center for 1 turn, and restore the sink-color center plus four corners for 2 or more turns.
    - Leave the source endpoint and its tail unchanged.

    Key insight:
    - The tail marks path directionality, so once source versus sink is known, the grey chain can be traced and its bend count determines which sink-centered 3x3 pattern to draw.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = map(int, g.shape)

    grey_components = _components(g == GREY)
    assert grey_components, "expected at least one grey path"

    out = g.copy()
    for component in grey_components:
        comp_set = set(component)
        endpoints: set[tuple[int, int]] = set()
        for r, c in component:
            for nr, nc in _neighbors4(r, c, h, w):
                if (nr, nc) not in comp_set and g[nr, nc] not in (BLACK, GREY):
                    endpoints.add((nr, nc))

        assert len(endpoints) == 2, "each path must touch exactly two colored endpoints"

        endpoint_info: list[tuple[tuple[int, int], int, list[tuple[int, int]]]] = []
        for endpoint in endpoints:
            color = int(g[endpoint])
            grey_neighbors = 0
            same_color_neighbors: list[tuple[int, int]] = []
            for nr, nc in _neighbors4(endpoint[0], endpoint[1], h, w):
                if (nr, nc) in comp_set:
                    grey_neighbors += 1
                elif int(g[nr, nc]) == color:
                    same_color_neighbors.append((nr, nc))
            assert grey_neighbors == 1
            endpoint_info.append((endpoint, color, same_color_neighbors))

        source_info = [info for info in endpoint_info if len(info[2]) == 1]
        sink_info = [info for info in endpoint_info if len(info[2]) == 0]
        assert len(source_info) == 1 and len(sink_info) == 1, "tail must identify a unique source"

        source, source_color, tail_neighbors = source_info[0]
        sink, sink_color, _ = sink_info[0]
        assert source_color != sink_color
        tail = tail_neighbors[0]
        assert tail not in comp_set and tail != sink

        seq = _ordered_path(component, source, sink, h, w)
        turns = _count_turns(seq)

        for r, c in component:
            out[r, c] = source_color
        _paint_sink_block(out, sink, source_color, sink_color, turns)

    return out


def _turn_schedule(count: int) -> list[int]:
    order = random.sample([0, 1, 2], 3)
    if count == 1:
        turns = [order[0]]
    elif count == 2:
        turns = order[:2]
    elif count == 3:
        turns = order
    else:
        turns = order + [random.choice((0, 1, 2))]
    random.shuffle(turns)
    return turns


def _random_lengths(turns: int, dirs: list[int]) -> list[int]:
    if turns == 0:
        return [random.randint(3, 8)]
    if turns == 1:
        return [random.randint(2, 6), random.randint(2, 6)]

    lengths = [random.randint(2, 5), random.randint(2, 5), random.randint(2, 5)]
    if dirs[2] == (dirs[0] + 2) % 4:
        lengths[1] = max(lengths[1], 2)
    return lengths


def _shape_is_simple(points: list[tuple[int, int]], tail: tuple[int, int]) -> bool:
    if len(set(points)) != len(points):
        return False
    for i, a in enumerate(points):
        for j in range(i + 2, len(points)):
            if _is_adjacent(a, points[j]):
                return False
    if not _is_adjacent(points[0], tail):
        return False
    return not any(_is_adjacent(tail, p) for p in points[1:])


def _make_relative_shape(turns: int) -> dict[str, object]:
    for _ in range(SHAPE_ATTEMPTS):
        dirs = [random.randrange(4)]
        if turns >= 1:
            dirs.append((dirs[-1] + random.choice((-1, 1))) % 4)
        if turns >= 2:
            dirs.append((dirs[-1] + random.choice((-1, 1))) % 4)

        lengths = _random_lengths(turns, dirs)
        points = [(0, 0)]
        r = c = 0
        for direction, length in zip(dirs, lengths):
            dr, dc = DIRS[direction]
            for _step in range(length):
                r += dr
                c += dc
                points.append((r, c))

        tail_dir = (dirs[0] + 2) % 4
        tr, tc = DIRS[tail_dir]
        tail = (tr, tc)
        if not _shape_is_simple(points, tail):
            continue

        input_cells = set(points)
        input_cells.add(tail)
        sink = points[-1]
        return {
            "points": points,
            "source": points[0],
            "sink": sink,
            "tail": tail,
            "grey_cells": set(points[1:-1]),
            "input_cells": input_cells,
        }
    raise RuntimeError(f"failed to generate a simple {turns}-turn path")


def _sink_block_cells(sink: tuple[int, int], h: int, w: int) -> set[tuple[int, int]]:
    sr, sc = sink
    cells: set[tuple[int, int]] = set()
    for rr in range(sr - 1, sr + 2):
        for cc in range(sc - 1, sc + 2):
            if 0 <= rr < h and 0 <= cc < w:
                cells.add((rr, cc))
    return cells


def _expand_cells(cells: set[tuple[int, int]], h: int, w: int) -> set[tuple[int, int]]:
    expanded: set[tuple[int, int]] = set()
    for r, c in cells:
        for rr in range(r - 1, r + 2):
            for cc in range(c - 1, c + 2):
                if 0 <= rr < h and 0 <= cc < w:
                    expanded.add((rr, cc))
    return expanded


def _candidate_offsets(
    shape: dict[str, object],
    h: int,
    w: int,
    prefer_edge: bool,
) -> list[tuple[int, int]]:
    input_cells = list(shape["input_cells"])
    sink_r, sink_c = shape["sink"]
    min_r = min(r for r, _ in input_cells)
    max_r = max(r for r, _ in input_cells)
    min_c = min(c for _, c in input_cells)
    max_c = max(c for _, c in input_cells)

    row_lo = -min_r
    row_hi = h - 1 - max_r
    col_lo = -min_c
    col_hi = w - 1 - max_c
    if row_lo > row_hi or col_lo > col_hi:
        return []

    candidates: list[tuple[int, int]] = []

    def add(off_r: int, off_c: int) -> None:
        if row_lo <= off_r <= row_hi and col_lo <= off_c <= col_hi:
            candidates.append((off_r, off_c))

    if prefer_edge:
        edge_rows = [-sink_r, h - 1 - sink_r]
        edge_cols = [-sink_c, w - 1 - sink_c]
        for off_r in edge_rows:
            add(off_r, random.randint(col_lo, col_hi))
        for off_c in edge_cols:
            add(random.randint(row_lo, row_hi), off_c)
        for off_r in edge_rows:
            for off_c in edge_cols:
                add(off_r, off_c)

    for _ in range(PLACEMENT_ATTEMPTS):
        add(random.randint(row_lo, row_hi), random.randint(col_lo, col_hi))

    deduped: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for offset in candidates:
        if offset not in seen:
            seen.add(offset)
            deduped.append(offset)
    return deduped


def _place_shape(
    grid: np.ndarray,
    reserved: set[tuple[int, int]],
    shape: dict[str, object],
    source_color: int,
    sink_color: int,
) -> bool:
    h, w = map(int, grid.shape)
    prefer_edge = random.random() < 0.35
    for off_r, off_c in _candidate_offsets(shape, h, w, prefer_edge):
        points = [(r + off_r, c + off_c) for r, c in shape["points"]]
        tail = (shape["tail"][0] + off_r, shape["tail"][1] + off_c)
        sink = (shape["sink"][0] + off_r, shape["sink"][1] + off_c)
        grey_cells = {(r + off_r, c + off_c) for r, c in shape["grey_cells"]}

        footprint = set(points)
        footprint.add(tail)
        footprint |= _sink_block_cells(sink, h, w)
        if not footprint.isdisjoint(reserved):
            continue

        if any(grid[r, c] != BLACK for r, c in footprint if (r, c) in set(points) | {tail}):
            continue

        source = points[0]
        grid[source] = source_color
        grid[tail] = source_color
        grid[sink] = sink_color
        for r, c in grey_cells:
            grid[r, c] = GREY

        reserved |= _expand_cells(footprint, h, w)
        return True
    return False


def _gen_attempt() -> np.ndarray | None:
    path_count = random.randint(1, 4)
    min_side = 12 + 3 * path_count
    max_side = min(30, min_side + 6)
    h = random.randint(min_side, max_side)
    w = random.randint(min_side, max_side)
    grid = np.full((h, w), BLACK, dtype=int)

    colors = random.sample(COLOR_POOL, 2 * path_count)
    color_pairs = [tuple(colors[2 * i : 2 * i + 2]) for i in range(path_count)]
    turns_by_path = _turn_schedule(path_count)

    specs = []
    for turns, (source_color, sink_color) in zip(turns_by_path, color_pairs):
        shape = _make_relative_shape(turns)
        footprint_size = len(shape["input_cells"]) + 9
        specs.append((footprint_size, turns, source_color, sink_color, shape))
    specs.sort(reverse=True)

    reserved: set[tuple[int, int]] = set()
    for _size, _turns, source_color, sink_color, shape in specs:
        if not _place_shape(grid, reserved, shape, source_color, sink_color):
            return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is None:
            continue
        try:
            out = main(grid)
        except Exception:
            continue
        if np.array_equal(out, grid):
            continue
        return grid
    raise RuntimeError("failed to generate a valid source-sink path puzzle")
