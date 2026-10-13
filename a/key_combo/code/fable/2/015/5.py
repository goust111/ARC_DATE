from __future__ import annotations

from dataclasses import dataclass
import random

import numpy as np


class Color:
    BLACK = 0
    BLUE = 1
    RED = 2
    GREEN = 3
    YELLOW = 4
    GREY = 5
    PINK = 6
    ORANGE = 7
    TEAL = 8
    MAROON = 9


BG = Color.BLACK
MARKER = Color.YELLOW
BAR_COLORS = (
    Color.BLUE,
    Color.RED,
    Color.GREEN,
    Color.GREY,
    Color.PINK,
    Color.ORANGE,
    Color.TEAL,
    Color.MAROON,
)

COUNT_RANGE = (2, 5)
GRID_EXTRA_RANGE = (0, 12)
MIN_BAR_LEN = 2
GEN_RETRIES = 400


@dataclass(frozen=True)
class CandidateRect:
    color: int
    r0: int
    r1: int
    c0: int
    c1: int

    @property
    def height(self) -> int:
        return int(self.r1 - self.r0 + 1)

    @property
    def width(self) -> int:
        return int(self.c1 - self.c0 + 1)

    def contains(self, r: int, c: int) -> bool:
        return self.r0 <= int(r) <= self.r1 and self.c0 <= int(c) <= self.c1


def _scan_edge_bars(values: np.ndarray) -> dict[int, tuple[int, int]]:
    bars: dict[int, tuple[int, int]] = {}
    i = 1
    n = int(values.shape[0])
    while i < n:
        color = int(values[i])
        if color == BG:
            i += 1
            continue
        assert color != MARKER, "yellow marker cannot be used as an edge-bar color"
        j = i
        while j + 1 < n and int(values[j + 1]) == color:
            j += 1
        assert color not in bars, "each edge-bar color must appear in one contiguous run"
        bars[color] = (i, j)
        i = j + 1
    return bars


def _extract_scene(grid: np.ndarray) -> tuple[list[CandidateRect], tuple[int, int]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = g.shape
    assert h >= 4 and w >= 4, "grid must have room for edge bars and an interior marker"
    assert int(g[0, 0]) == BG, "top-left corner must remain background"

    marker_coords = np.argwhere(g == MARKER)
    assert marker_coords.shape[0] == 1, "expected exactly one yellow marker pixel"
    marker_r, marker_c = map(int, marker_coords[0])
    assert marker_r > 0 and marker_c > 0, "marker must be inside the interior, not on control edges"

    top_bars = _scan_edge_bars(g[0, :])
    left_bars = _scan_edge_bars(g[:, 0])
    assert top_bars, "expected at least one top edge bar"
    assert set(top_bars) == set(left_bars), "top and left edges must contain the same control colors"

    candidates = [
        CandidateRect(color=color, r0=row_span[0], r1=row_span[1], c0=col_span[0], c1=col_span[1])
        for color, col_span in top_bars.items()
        for row_span in [left_bars[color]]
    ]

    interior = g.copy()
    interior[0, :] = BG
    interior[:, 0] = BG
    interior[marker_r, marker_c] = BG
    assert np.all(interior == BG), "expected only control bars plus one interior marker"
    return sorted(candidates, key=lambda rect: (rect.color, rect.r0, rect.c0)), (marker_r, marker_c)


def _selected_rect(candidates: list[CandidateRect], marker: tuple[int, int]) -> CandidateRect:
    marker_r, marker_c = marker
    hits = [rect for rect in candidates if rect.contains(marker_r, marker_c)]
    assert len(hits) == 1, "marker must lie in exactly one latent rectangle"
    return hits[0]


def _render_output(rect: CandidateRect) -> np.ndarray:
    out = np.full((rect.height, rect.width), BG, dtype=int)
    out[0, :] = rect.color
    out[-1, :] = rect.color
    out[:, 0] = rect.color
    out[:, -1] = rect.color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reconstruct the latent rectangle chosen by the yellow marker and output only its hollow frame.

    Input structure:
    - Row 0 contains several non-overlapping colored horizontal bars, and column 0 contains one vertical bar in the same color for each top bar.
    - The rest of the interior is black except for one yellow marker pixel; yellow is never used as a bar color.
    - Each non-yellow control color appears exactly once on the top edge and exactly once on the left edge.

    Rule:
    - Pair the top and left bars by color.
    - For each color, form a latent rectangle whose column span comes from the top bar and whose row span comes from the left bar.
    - Find the unique latent rectangle that contains the yellow marker.
    - Return a new grid with exactly that rectangle's height and width, drawing a one-cell-thick hollow frame in the rectangle's color on black background.

    Key insight:
    - No rectangle is actually drawn in the input: the edge bars are orthogonal projection traces, and the yellow marker only selects which reconstructed candidate survives into the output.
    """
    candidates, marker = _extract_scene(np.asarray(grid, dtype=int))
    return _render_output(_selected_rect(candidates, marker))


def _sample_intervals(edge_len: int, count: int) -> list[tuple[int, int]]:
    assert edge_len >= count * MIN_BAR_LEN + (count - 1)
    spare = edge_len - (count * MIN_BAR_LEN + (count - 1))

    length_extras = [0] * count
    gap_extras = [0] * (count + 1)
    weighted_slots = [("len", idx) for idx in range(count) for _ in range(3)]
    weighted_slots += [("gap", idx) for idx in range(count + 1)]
    for _ in range(spare):
        kind, idx = random.choice(weighted_slots)
        if kind == "len":
            length_extras[idx] += 1
        else:
            gap_extras[idx] += 1

    intervals: list[tuple[int, int]] = []
    pos = 1 + gap_extras[0]
    for i in range(count):
        length = MIN_BAR_LEN + length_extras[i]
        start = pos
        end = start + length - 1
        intervals.append((start, end))
        if i < count - 1:
            pos = end + 2 + gap_extras[i + 1]
    assert intervals[-1][1] <= edge_len
    return intervals


def _choose_orders(colors: list[int]) -> tuple[list[int], list[int]]:
    top_order = random.sample(colors, len(colors))
    left_order = random.sample(colors, len(colors))
    if len(colors) > 1 and top_order == left_order:
        left_order = left_order[1:] + left_order[:1]
    assert top_order != left_order or len(colors) == 1
    return top_order, left_order


def _sample_marker_cell(rect: CandidateRect) -> tuple[int, int]:
    cells = [(r, c) for r in range(rect.r0, rect.r1 + 1) for c in range(rect.c0, rect.c1 + 1)]
    border = [
        (r, c)
        for r, c in cells
        if r in (rect.r0, rect.r1) or c in (rect.c0, rect.c1)
    ]
    interior = [cell for cell in cells if cell not in border]
    if border and interior:
        if random.random() < 0.5:
            return random.choice(interior)
        return random.choice(border)
    if interior:
        return random.choice(interior)
    return random.choice(cells)


def _gen_attempt() -> np.ndarray:
    count = random.randint(*COUNT_RANGE)
    min_side = 3 * count
    rows = random.randint(min_side, min(30, min_side + GRID_EXTRA_RANGE[1]))
    cols = random.randint(min_side, min(30, min_side + GRID_EXTRA_RANGE[1]))

    row_intervals = _sample_intervals(rows - 1, count)
    col_intervals = _sample_intervals(cols - 1, count)
    colors = random.sample(BAR_COLORS, count)
    top_order, left_order = _choose_orders(colors)

    top_by_color = {color: interval for color, interval in zip(top_order, col_intervals)}
    left_by_color = {color: interval for color, interval in zip(left_order, row_intervals)}

    grid = np.full((rows, cols), BG, dtype=int)
    for color, (c0, c1) in top_by_color.items():
        grid[0, c0 : c1 + 1] = color
    for color, (r0, r1) in left_by_color.items():
        grid[r0 : r1 + 1, 0] = color

    chosen_color = random.choice(colors)
    chosen_rect = CandidateRect(
        color=chosen_color,
        r0=left_by_color[chosen_color][0],
        r1=left_by_color[chosen_color][1],
        c0=top_by_color[chosen_color][0],
        c1=top_by_color[chosen_color][1],
    )
    marker_r, marker_c = _sample_marker_cell(chosen_rect)
    grid[marker_r, marker_c] = MARKER

    candidates, marker = _extract_scene(grid)
    selected = _selected_rect(candidates, marker)
    assert selected == chosen_rect, "generator must preserve the intended latent rectangle"
    out = _render_output(selected)
    assert out.shape == (chosen_rect.height, chosen_rect.width)
    assert np.all(out[0, :] == chosen_color) and np.all(out[:, 0] == chosen_color)
    assert np.all(out[-1, :] == chosen_color) and np.all(out[:, -1] == chosen_color)
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        try:
            return _gen_attempt()
        except AssertionError:
            continue
    raise RuntimeError("failed to generate a valid projection-pairing input")
