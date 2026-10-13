from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import random

import numpy as np


BG = 0
GREY = 5
LINE_COLORS = tuple(color for color in range(10) if color not in (BG, GREY))

GRID_H_RANGE = (12, 20)
GRID_W_RANGE = (12, 20)
H_COUNT_RANGE = (2, 4)
V_COUNT_RANGE = (2, 4)
MIN_GAP = 2
ROW_COL_BORDER = 2
H_MARGIN_RANGE = (1, 3)
V_MARGIN_RANGE = (1, 3)
REVEAL_PROB_RANGE = (0.25, 0.65)
MAX_GENERATION_ATTEMPTS = 300


@dataclass(frozen=True)
class Segment:
    axis: str
    color: int
    fixed: int
    start: int
    end: int


def _mode_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _spaced_positions(length: int, count: int, border: int, min_gap: int) -> list[int] | None:
    lo = border
    hi = length - border - 1
    span = hi - lo + 1
    extra_gap = min_gap - 1
    needed = count + (count - 1) * extra_gap
    if span < needed:
        return None

    compressed = span - (count - 1) * extra_gap
    picks = sorted(random.sample(range(compressed), count))
    return [lo + pick + idx * extra_gap for idx, pick in enumerate(picks)]


def _extract_segments(grid: np.ndarray) -> tuple[list[Segment], list[Segment]]:
    g = np.asarray(grid, dtype=int)
    bg = _mode_color(g)
    assert bg == BG, "expected black background"

    colors = sorted(int(x) for x in np.unique(g) if int(x) not in {bg, GREY})
    assert colors, "expected colored line segments"

    horizontals: list[Segment] = []
    verticals: list[Segment] = []
    for color in colors:
        coords = np.argwhere(g == color)
        assert coords.shape[0] >= 2, "each segment needs at least two visible pixels"
        rows = np.unique(coords[:, 0])
        cols = np.unique(coords[:, 1])
        if rows.size == 1:
            horizontals.append(
                Segment(
                    axis="h",
                    color=color,
                    fixed=int(rows[0]),
                    start=int(cols.min()),
                    end=int(cols.max()),
                )
            )
        elif cols.size == 1:
            verticals.append(
                Segment(
                    axis="v",
                    color=color,
                    fixed=int(cols[0]),
                    start=int(rows.min()),
                    end=int(rows.max()),
                )
            )
        else:
            raise AssertionError("each color must form a straight horizontal or vertical segment")

    horizontals.sort(key=lambda seg: seg.fixed)
    verticals.sort(key=lambda seg: seg.fixed)
    assert horizontals and verticals, "expected both horizontal and vertical segments"
    return horizontals, verticals


def _crossings(
    horizontals: list[Segment],
    verticals: list[Segment],
) -> tuple[list[tuple[int, int, int, int]], dict[tuple[int, int], int], dict[tuple[int, int], int]]:
    crossings: list[tuple[int, int, int, int]] = []
    h_items: dict[int, list[tuple[int, int]]] = {idx: [] for idx in range(len(horizontals))}
    v_items: dict[int, list[tuple[int, int]]] = {idx: [] for idx in range(len(verticals))}

    for h_idx, h_seg in enumerate(horizontals):
        for v_idx, v_seg in enumerate(verticals):
            if not (h_seg.start <= v_seg.fixed <= h_seg.end):
                continue
            if not (v_seg.start <= h_seg.fixed <= v_seg.end):
                continue
            r = h_seg.fixed
            c = v_seg.fixed
            crossings.append((h_idx, v_idx, r, c))
            h_items[h_idx].append((c, v_idx))
            v_items[v_idx].append((r, h_idx))

    assert crossings, "expected at least one crossing"
    assert all(h_items[idx] for idx in h_items), "every horizontal segment must cross a vertical segment"
    assert all(v_items[idx] for idx in v_items), "every vertical segment must cross a horizontal segment"

    h_order: dict[tuple[int, int], int] = {}
    for h_idx, items in h_items.items():
        items.sort()
        for order, (_coord, v_idx) in enumerate(items):
            h_order[(h_idx, v_idx)] = order

    v_order: dict[tuple[int, int], int] = {}
    for v_idx, items in v_items.items():
        items.sort()
        for order, (_coord, h_idx) in enumerate(items):
            v_order[(h_idx, v_idx)] = order

    return crossings, h_order, v_order


def _solve_phases(
    grid: np.ndarray,
    horizontals: list[Segment],
    verticals: list[Segment],
    crossings: list[tuple[int, int, int, int]],
    h_order: dict[tuple[int, int], int],
    v_order: dict[tuple[int, int], int],
) -> dict[tuple[str, int], int]:
    g = np.asarray(grid, dtype=int)
    nodes = [("h", idx) for idx in range(len(horizontals))] + [("v", idx) for idx in range(len(verticals))]
    adjacency: dict[tuple[str, int], list[tuple[tuple[str, int], int]]] = {node: [] for node in nodes}
    seeds: dict[tuple[str, int], int] = {}

    def add_seed(node: tuple[str, int], value: int) -> None:
        if node in seeds:
            assert seeds[node] == value, "colored crossings must agree on each line phase"
        else:
            seeds[node] = value

    for h_idx, v_idx, r, c in crossings:
        h_seg = horizontals[h_idx]
        v_seg = verticals[v_idx]
        h_parity = h_order[(h_idx, v_idx)] % 2
        v_parity = v_order[(h_idx, v_idx)] % 2
        relation = 1 ^ h_parity ^ v_parity

        h_node = ("h", h_idx)
        v_node = ("v", v_idx)
        adjacency[h_node].append((v_node, relation))
        adjacency[v_node].append((h_node, relation))

        value = int(g[r, c])
        if value == GREY:
            continue
        if value == h_seg.color:
            add_seed(h_node, 1 ^ h_parity)
            add_seed(v_node, v_parity)
        elif value == v_seg.color:
            add_seed(h_node, h_parity)
            add_seed(v_node, 1 ^ v_parity)
        else:
            raise AssertionError("crossing cell must be grey or one of the two crossing colors")

    assert seeds, "at least one crossing must reveal its visible line"

    phases = dict(seeds)
    queue = deque(seeds)
    while queue:
        node = queue.popleft()
        value = phases[node]
        for other, relation in adjacency[node]:
            inferred = value ^ relation
            if other not in phases:
                phases[other] = inferred
                queue.append(other)
            else:
                assert phases[other] == inferred, "crossing constraints must be globally consistent"

    missing = [node for node in nodes if node not in phases]
    assert not missing, "each connected line family needs at least one revealed crossing"
    return phases


def main(grid: np.ndarray) -> np.ndarray:
    """
    Resolve woven line crossings by recovering each straight segment's alternating over/under phase.

    Input structure
    - The background is black.
    - Each non-black, non-grey color forms one unique straight horizontal or vertical segment.
    - Horizontal and vertical segments cross. Some crossings already show the visible line color,
      while grey crossings hide the winner and must be inferred.

    Rule
    - For each segment, crossings alternate between visible and hidden along the segment.
    - A colored crossing fixes the phase of both participating segments.
    - A grey crossing keeps only the parity relation between the two segments.
    - Recover all segment phases, then repaint every crossing with the segment that should be visible there.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    horizontals, verticals = _extract_segments(g)
    crossings, h_order, v_order = _crossings(horizontals, verticals)
    phases = _solve_phases(g, horizontals, verticals, crossings, h_order, v_order)

    crossing_positions = {(r, c) for _h_idx, _v_idx, r, c in crossings}
    grey_positions = {tuple(map(int, rc)) for rc in np.argwhere(g == GREY)}
    assert grey_positions <= crossing_positions, "grey pixels may only appear at crossings"

    out = g.copy()
    for h_idx, v_idx, r, c in crossings:
        h_visible = phases[("h", h_idx)] ^ (h_order[(h_idx, v_idx)] % 2)
        v_visible = phases[("v", v_idx)] ^ (v_order[(h_idx, v_idx)] % 2)
        assert h_visible ^ v_visible, "exactly one segment must be visible at each crossing"
        out[r, c] = horizontals[h_idx].color if h_visible else verticals[v_idx].color

    assert not np.any(out == GREY), "output should not contain grey"
    return out


def _edge_cover(h_count: int, v_count: int) -> set[tuple[int, int]]:
    horiz = list(range(h_count))
    vert = list(range(v_count))
    random.shuffle(horiz)
    random.shuffle(vert)

    revealed: set[tuple[int, int]] = set()
    shared = min(h_count, v_count)
    for idx in range(shared):
        revealed.add((horiz[idx], vert[idx]))

    if h_count > v_count:
        for h_idx in horiz[shared:]:
            revealed.add((h_idx, random.choice(vert)))
    elif v_count > h_count:
        for v_idx in vert[shared:]:
            revealed.add((random.choice(horiz), v_idx))

    return revealed


def _coverage_ok(
    revealed: set[tuple[int, int]],
    h_count: int,
    v_count: int,
    h_wins: set[tuple[int, int]],
    v_wins: set[tuple[int, int]],
) -> bool:
    if not revealed:
        return False
    if not any(edge in revealed for edge in h_wins):
        return False
    if not any(edge in revealed for edge in v_wins):
        return False
    for h_idx in range(h_count):
        if not any((h_idx, v_idx) in revealed for v_idx in range(v_count)):
            return False
    for v_idx in range(v_count):
        if not any((h_idx, v_idx) in revealed for h_idx in range(h_count)):
            return False
    return True


def _sample_revealed_edges(
    h_count: int,
    v_count: int,
    h_wins: set[tuple[int, int]],
    v_wins: set[tuple[int, int]],
) -> set[tuple[int, int]]:
    all_edges = {(h_idx, v_idx) for h_idx in range(h_count) for v_idx in range(v_count)}
    revealed = _edge_cover(h_count, v_count)
    reveal_prob = random.uniform(*REVEAL_PROB_RANGE)
    for edge in all_edges - revealed:
        if random.random() < reveal_prob:
            revealed.add(edge)

    if not any(edge in revealed for edge in h_wins):
        revealed.add(random.choice(tuple(h_wins)))
    if not any(edge in revealed for edge in v_wins):
        revealed.add(random.choice(tuple(v_wins)))

    if revealed == all_edges:
        removable = [
            edge
            for edge in revealed
            if _coverage_ok(revealed - {edge}, h_count, v_count, h_wins, v_wins)
        ]
        assert removable, "expected at least one removable revealed crossing"
        revealed.remove(random.choice(removable))

    assert revealed != all_edges, "expected at least one grey crossing"
    assert _coverage_ok(revealed, h_count, v_count, h_wins, v_wins)
    return revealed


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(*GRID_H_RANGE)
    width = random.randint(*GRID_W_RANGE)
    h_count = random.randint(*H_COUNT_RANGE)
    v_count = random.randint(*V_COUNT_RANGE)

    rows = _spaced_positions(height, h_count, ROW_COL_BORDER, MIN_GAP)
    cols = _spaced_positions(width, v_count, ROW_COL_BORDER, MIN_GAP)
    if rows is None or cols is None:
        return None

    if h_count + v_count > len(LINE_COLORS):
        return None
    palette = random.sample(LINE_COLORS, h_count + v_count)

    left_anchor = min(cols)
    right_anchor = max(cols)
    top_anchor = min(rows)
    bottom_anchor = max(rows)

    horizontals: list[Segment] = []
    verticals: list[Segment] = []
    for idx, row in enumerate(rows):
        left = left_anchor - random.randint(*H_MARGIN_RANGE)
        right = right_anchor + random.randint(*H_MARGIN_RANGE)
        if left < 0 or right >= width:
            return None
        horizontals.append(Segment(axis="h", color=palette[idx], fixed=row, start=left, end=right))

    for idx, col in enumerate(cols):
        top = top_anchor - random.randint(*V_MARGIN_RANGE)
        bottom = bottom_anchor + random.randint(*V_MARGIN_RANGE)
        if top < 0 or bottom >= height:
            return None
        verticals.append(Segment(axis="v", color=palette[h_count + idx], fixed=col, start=top, end=bottom))

    phase0 = random.randint(0, 1)
    h_phase = {idx: phase0 ^ (idx % 2) for idx in range(h_count)}
    v_phase = {idx: (1 ^ phase0) ^ (idx % 2) for idx in range(v_count)}

    h_wins: set[tuple[int, int]] = set()
    v_wins: set[tuple[int, int]] = set()
    for h_idx in range(h_count):
        for v_idx in range(v_count):
            visible = h_phase[h_idx] ^ (v_idx % 2)
            if visible:
                h_wins.add((h_idx, v_idx))
            else:
                v_wins.add((h_idx, v_idx))

    revealed = _sample_revealed_edges(h_count, v_count, h_wins, v_wins)

    grid = np.full((height, width), BG, dtype=int)
    for seg in horizontals:
        grid[seg.fixed, seg.start : seg.end + 1] = seg.color
    for seg in verticals:
        grid[seg.start : seg.end + 1, seg.fixed] = seg.color

    for h_idx, h_seg in enumerate(horizontals):
        for v_idx, v_seg in enumerate(verticals):
            r = h_seg.fixed
            c = v_seg.fixed
            if (h_idx, v_idx) not in revealed:
                grid[r, c] = GREY
            else:
                grid[r, c] = h_seg.color if (h_idx, v_idx) in h_wins else v_seg.color

    try:
        out = main(grid)
    except AssertionError:
        return None

    if np.array_equal(out, grid):
        return None
    if np.any(out == GREY):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid woven-line input")
