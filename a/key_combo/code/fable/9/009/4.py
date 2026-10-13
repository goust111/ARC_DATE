from __future__ import annotations

from collections import Counter
import random

import numpy as np


Coord = tuple[int, int]
Bar = dict[str, object]

ARC_COLORS = tuple(range(10))
GENERATION_RETRIES = 600
MIN_BARS = 3
MAX_BARS = 6
MIN_HEIGHT = 7
MAX_HEIGHT = 15
MIN_WIDTH = 8
MAX_WIDTH = 18


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _find_components(grid: np.ndarray, bg: int) -> list[list[Coord]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[list[Coord]] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) == bg:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[Coord] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) != bg:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            components.append(coords)
    return components


def _bar_from_coords(grid: np.ndarray, coords: list[Coord]) -> Bar:
    assert len(coords) == 2, "each bar must contain exactly two pixels"
    ordered = sorted(coords)
    (r0, c0), (r1, c1) = ordered
    assert r0 == r1 and c1 == c0 + 1, "each bar must be horizontal and adjacent"
    values = (int(grid[r0, c0]), int(grid[r1, c1]))
    assert values[0] != values[1], "bar colors must be distinct"
    return {
        "row": int(r0),
        "start": int(c0),
        "values": values,
        "colors": frozenset(values),
    }


def _parse_input(grid: np.ndarray, bg: int) -> tuple[Coord, list[Bar]]:
    marker: Coord | None = None
    bars: list[Bar] = []
    for coords in _find_components(grid, bg):
        if len(coords) == 1:
            assert marker is None, "expected exactly one marker pixel"
            marker = coords[0]
        else:
            bars.append(_bar_from_coords(grid, coords))
    assert marker is not None, "missing marker pixel"
    assert bars, "expected at least one two-color bar"
    return marker, bars


def _build_adjacency(bars: list[Bar]) -> list[list[int]]:
    adj = [[] for _ in bars]
    for i in range(len(bars)):
        for j in range(i + 1, len(bars)):
            shared = bars[i]["colors"] & bars[j]["colors"]
            assert len(shared) < 2, "distinct bars must not share both colors"
            if len(shared) == 1:
                adj[i].append(j)
                adj[j].append(i)
    return adj


def _chain_order(bars: list[Bar], marker_color: int) -> list[int]:
    head_candidates = [i for i, bar in enumerate(bars) if marker_color in bar["colors"]]
    assert len(head_candidates) == 1, "marker color must identify exactly one head bar"
    head = head_candidates[0]

    if len(bars) == 1:
        return [head]

    adj = _build_adjacency(bars)
    degrees = [len(neighbors) for neighbors in adj]
    assert degrees[head] == 1, "head must be an endpoint of the color-sharing chain"
    assert sum(deg == 1 for deg in degrees) == 2, "chain must have exactly two endpoints"
    assert all(deg in (1, 2) for deg in degrees), "color-sharing graph must be a simple path"

    order = [head]
    prev = -1
    cur = head
    while len(order) < len(bars):
        nxt = [idx for idx in adj[cur] if idx != prev]
        assert len(nxt) == 1, "path traversal must be unique at every step"
        prev, cur = cur, nxt[0]
        order.append(cur)

    assert len(set(order)) == len(bars), "chain must visit each bar exactly once"
    return order


def _paint_bar(grid: np.ndarray, row: int, start: int, values: tuple[int, int], bg: int) -> None:
    assert 0 <= row < grid.shape[0]
    assert 0 <= start and start + 1 < grid.shape[1]
    assert int(grid[row, start]) == bg and int(grid[row, start + 1]) == bg, "bars must not overlap"
    grid[row, start : start + 2] = values


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rehang the scattered two-color bars into a cascading vertical chain starting from the marker-colored head.

    Input structure:
    - A dominant background contains several disconnected horizontal 1x2 bars, each using two different colors.
    - Exactly one foreground component is a single marker pixel; its color appears on exactly one bar and identifies the chain head.
    - The bars are spatially scattered, but by color-sharing they form one unique path where consecutive bars share exactly one color.

    Rule:
    - Remove the isolated marker pixel and keep the head bar in its original position and orientation.
    - Follow the unique color-sharing chain from the head to the tail.
    - Place each later bar one row below the previous placed bar, shifting it horizontally without flipping so the shared color lands in the same column as on the previous bar.
    - Clear every original bar position to background; the output contains only the reassembled staircase chain.

    Key insight:
    - The chain order is encoded purely by which bars share a color, and each placement is a local transmission step because every new bar aligns to the previous bar's already-set shared-color column.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _get_bg(g)
    marker, bars = _parse_input(g, bg)
    marker_color = int(g[marker])
    order = _chain_order(bars, marker_color)

    out = np.full_like(g, bg)
    placements: list[tuple[int, int]] = []

    head = bars[order[0]]
    head_row = int(head["row"])
    head_start = int(head["start"])
    head_values = tuple(head["values"])
    _paint_bar(out, head_row, head_start, head_values, bg)
    placements.append((head_row, head_start))

    for step in range(1, len(order)):
        prev_bar = bars[order[step - 1]]
        cur_bar = bars[order[step]]
        shared = tuple(prev_bar["colors"] & cur_bar["colors"])
        assert len(shared) == 1, "adjacent chain bars must share exactly one color"
        shared_color = int(shared[0])

        prev_values = tuple(prev_bar["values"])
        cur_values = tuple(cur_bar["values"])
        prev_row, prev_start = placements[-1]
        cur_row = prev_row + 1
        cur_start = prev_start + prev_values.index(shared_color) - cur_values.index(shared_color)

        _paint_bar(out, cur_row, cur_start, cur_values, bg)
        placements.append((cur_row, cur_start))

    return out


def _compute_relative_starts(bars: list[tuple[int, int]], chain_colors: list[int]) -> tuple[list[int], list[int]]:
    rel = [0]
    deltas: list[int] = []
    for i in range(1, len(bars)):
        shared = chain_colors[i]
        delta = bars[i - 1].index(shared) - bars[i].index(shared)
        deltas.append(int(delta))
        rel.append(rel[-1] + int(delta))
    return rel, deltas


def _can_place(candidate: list[Coord], occupied: set[Coord], height: int, width: int) -> bool:
    for r, c in candidate:
        if not (0 <= r < height and 0 <= c < width):
            return False
        for rr in range(r - 1, r + 2):
            for cc in range(c - 1, c + 2):
                if (rr, cc) in occupied:
                    return False
    return True


def _reserve(candidate: list[Coord], occupied: set[Coord]) -> None:
    for cell in candidate:
        occupied.add(cell)


def _bar_cells(row: int, start: int) -> list[Coord]:
    return [(row, start), (row, start + 1)]


def _render(height: int, width: int, bg: int, bars: list[tuple[int, int]], positions: list[tuple[int, int]]) -> np.ndarray:
    grid = np.full((height, width), bg, dtype=int)
    for values, (row, start) in zip(bars, positions):
        grid[row, start : start + 2] = values
    return grid


def _is_monotone(seq: list[int]) -> bool:
    return all(seq[i] < seq[i + 1] for i in range(len(seq) - 1)) or all(
        seq[i] > seq[i + 1] for i in range(len(seq) - 1)
    )


def _scatter_positions(
    height: int,
    width: int,
    output_positions: list[tuple[int, int]],
) -> list[tuple[int, int]] | None:
    occupied: set[Coord] = set()
    positions = [output_positions[0]]
    _reserve(_bar_cells(*output_positions[0]), occupied)

    for index in range(1, len(output_positions)):
        candidates = []
        for row in range(height):
            for start in range(width - 1):
                pos = (row, start)
                if pos == output_positions[index]:
                    continue
                cells = _bar_cells(row, start)
                if _can_place(cells, occupied, height, width):
                    candidates.append(pos)
        if not candidates:
            return None
        pos = random.choice(candidates)
        positions.append(pos)
        _reserve(_bar_cells(*pos), occupied)

    rows = [row for row, _ in positions]
    cols = [start for _, start in positions]
    moved = sum(pos != out for pos, out in zip(positions, output_positions))
    if moved < 2:
        return None
    if _is_monotone(rows) or _is_monotone(cols):
        return None

    return positions


def _place_marker(inp: np.ndarray, marker_color: int, occupied: set[Coord]) -> bool:
    height, width = inp.shape
    candidates = []
    for row in range(height):
        for col in range(width):
            if _can_place([(row, col)], occupied, height, width):
                candidates.append((row, col))
    if not candidates:
        return False
    row, col = random.choice(candidates)
    inp[row, col] = marker_color
    occupied.add((row, col))
    return True


def _semantic_checks(
    inp: np.ndarray,
    expected: np.ndarray,
    chain_colors: list[int],
    bars: list[tuple[int, int]],
    output_positions: list[tuple[int, int]],
    deltas: list[int],
) -> bool:
    bg = _get_bg(inp)
    try:
        marker, parsed_bars = _parse_input(inp, bg)
        marker_color = int(inp[marker])
        order = _chain_order(parsed_bars, marker_color)
        actual = main(inp)
    except Exception:
        return False

    if not np.array_equal(actual, expected):
        return False
    if marker_color != chain_colors[0]:
        return False
    if int(np.count_nonzero(inp == marker_color)) != 2:
        return False
    if len(order) != len(bars):
        return False
    if len(set(deltas)) < 2 or not any(delta != 0 for delta in deltas):
        return False
    if len({start for _, start in output_positions}) < 2:
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    bar_count = random.randint(MIN_BARS, MAX_BARS)
    palette = random.sample(ARC_COLORS, bar_count + 2)
    bg = palette[0]
    chain_colors = palette[1:]

    bars: list[tuple[int, int]] = []
    for i in range(bar_count):
        pair = [chain_colors[i], chain_colors[i + 1]]
        if random.random() < 0.5:
            pair.reverse()
        bars.append((pair[0], pair[1]))

    rel_starts, deltas = _compute_relative_starts(bars, chain_colors)
    if len(set(deltas)) < 2 or not any(delta != 0 for delta in deltas):
        return None

    min_rel = min(rel_starts)
    max_rel = max(rel_starts)
    span = max_rel - min_rel + 2

    height_lo = max(MIN_HEIGHT, bar_count + 3)
    height_hi = min(MAX_HEIGHT, bar_count + 8)
    width_lo = max(MIN_WIDTH, span + 5)
    width_hi = min(MAX_WIDTH, span + 10)
    if height_lo > height_hi or width_lo > width_hi:
        return None

    height = random.randint(height_lo, height_hi)
    width = random.randint(width_lo, width_hi)

    head_row = random.randint(0, height - bar_count)
    head_col_lo = -min_rel
    head_col_hi = width - 2 - max_rel
    if head_col_lo > head_col_hi:
        return None
    head_col = random.randint(head_col_lo, head_col_hi)
    output_positions = [(head_row + i, head_col + rel_starts[i]) for i in range(bar_count)]

    input_positions = _scatter_positions(height, width, output_positions)
    if input_positions is None:
        return None

    inp = _render(height, width, bg, bars, input_positions)
    occupied: set[Coord] = set()
    for row, start in input_positions:
        _reserve(_bar_cells(row, start), occupied)
    if not _place_marker(inp, chain_colors[0], occupied):
        return None

    expected = _render(height, width, bg, bars, output_positions)
    if not _semantic_checks(inp, expected, chain_colors, bars, output_positions, deltas):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 4")
