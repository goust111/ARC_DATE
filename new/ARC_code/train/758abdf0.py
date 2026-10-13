from __future__ import annotations

import numpy as np

from arc_utils import colors_present, get_bg
import random

Coord = tuple[int, int]
CellPair = tuple[Coord, Coord]


def _edge_bar(g: np.ndarray, bg: int) -> tuple[str, int]:
    h, w = g.shape
    candidates: list[tuple[str, int, np.ndarray]] = []

    for color in colors_present(g, exclude={bg}):
        if np.all(g[0, :] == color):
            mask = np.zeros_like(g, dtype=bool)
            mask[0, :] = True
            candidates.append(("top", int(color), mask))
        if np.all(g[h - 1, :] == color):
            mask = np.zeros_like(g, dtype=bool)
            mask[h - 1, :] = True
            candidates.append(("bottom", int(color), mask))
        if np.all(g[:, 0] == color):
            mask = np.zeros_like(g, dtype=bool)
            mask[:, 0] = True
            candidates.append(("left", int(color), mask))
        if np.all(g[:, w - 1] == color):
            mask = np.zeros_like(g, dtype=bool)
            mask[:, w - 1] = True
            candidates.append(("right", int(color), mask))

    assert len(candidates) == 1, "expected exactly one full non-bg edge bar"
    edge, color, mask = candidates[0]
    assert np.array_equal(g == color, mask), "edge-bar color must occur only on that edge"
    return edge, color


def _lane_cells(edge: str, shape: tuple[int, int], lane: int) -> tuple[CellPair, CellPair]:
    h, w = shape
    if edge == "top":
        return ((1, lane), (2, lane)), ((h - 2, lane), (h - 1, lane))
    if edge == "bottom":
        return ((h - 2, lane), (h - 3, lane)), ((1, lane), (0, lane))
    if edge == "left":
        return ((lane, 1), (lane, 2)), ((lane, w - 2), (lane, w - 1))
    if edge == "right":
        return ((lane, w - 2), (lane, w - 3)), ((lane, 1), (lane, 0))
    raise AssertionError(f"unknown edge {edge!r}")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Advance short marker bars away from the full edge bar, turning completed two-pixel markers into edge-color marks at the opposite side.

    Input structure
    - The background is the dominant color.
    - One non-background color forms a complete one-cell-thick bar on exactly one outer edge.
    - The other foreground color appears only in the first two cells inward from that bar, lane by lane.

    Rule
    - In each lane perpendicular to the edge bar, a one-pixel marker adjacent to the bar grows one more cell inward.
    - A two-pixel marker in those first two inward cells is cleared back to background.
    - The cleared two-pixel marker is replaced by a two-pixel mark of the edge-bar color at the opposite side of the same lane.

    Key insight
    - The full edge bar defines both the orientation and the direction of the local two-step marker cycle.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = get_bg(g)
    edge, edge_color = _edge_bar(g, bg)
    marker_colors = colors_present(g, exclude={bg, edge_color})
    assert len(marker_colors) == 1, "expected exactly one marker color"
    marker = int(marker_colors[0])

    h, w = g.shape
    perp_len = h if edge in {"top", "bottom"} else w
    assert perp_len >= 5, "need non-overlapping near and opposite two-cell marks"
    lanes = range(w) if edge in {"top", "bottom"} else range(h)

    out = g.copy()
    near_cells: set[tuple[int, int]] = set()
    changed = False
    for lane in lanes:
        near, far = _lane_cells(edge, g.shape, int(lane))
        near_cells.update(near)
        first, second = (int(g[r, c]) for r, c in near)

        if first == marker and second == marker:
            assert all(int(g[r, c]) == bg for r, c in far), "opposite target cells must be background"
            for r, c in near:
                out[r, c] = bg
            for r, c in far:
                out[r, c] = edge_color
            changed = True
        elif first == marker:
            assert second == bg, "single marker must have background behind it"
            out[near[1]] = marker
            changed = True
        else:
            assert first == bg and second == bg, "empty lanes must be background in the first two inward cells"

    for r, c in np.argwhere(g == marker):
        assert (int(r), int(c)) in near_cells, "markers must be in the first two inward cells"
    assert changed, "expected at least one marker to transform"
    return out

# --------------- generator (att2) ---------------

# Canvas and edge choices.
EDGE_NAMES = ["top", "bottom", "left", "right"]
SIZE_MIN, SIZE_MAX = 8, 16

# Lane population constraints.
NONEMPTY_LANES_MIN, NONEMPTY_LANES_MAX = 3, 7
NONEMPTY_RATIO_MIN, NONEMPTY_RATIO_MAX = 0.25, 0.44
SINGLE_LANES_MIN, SINGLE_LANES_MAX = 1, 4
DOUBLE_LANES_MIN, DOUBLE_LANES_MAX = 1, 4

# Palette and retry limits.
COLOR_COUNT = 10
PALETTE_SIZE = 3
MAX_GENERATION_ATTEMPTS = 300


def _choose_shape(edge: str, nonempty: int) -> tuple[int, int]:
    """Choose dimensions whose lane count satisfies the active-lane ratio."""
    lane_lengths = [
        n
        for n in range(SIZE_MIN, SIZE_MAX + 1)
        if NONEMPTY_RATIO_MIN <= nonempty / n <= NONEMPTY_RATIO_MAX
    ]
    lanes = random.choice(lane_lengths)
    perp = random.randint(SIZE_MIN, SIZE_MAX)
    if edge in {"top", "bottom"}:
        return perp, lanes
    return lanes, perp


def _paint_edge_bar(g: np.ndarray, out: np.ndarray, edge: str, edge_color: int) -> None:
    """Paint the same full edge bar in the input and expected output grids."""
    h, w = g.shape
    edge_index = {
        "top": (0, slice(None)),
        "bottom": (h - 1, slice(None)),
        "left": (slice(None), 0),
        "right": (slice(None), w - 1),
    }[edge]
    g[edge_index] = edge_color
    out[edge_index] = edge_color


def _choose_lane_states(lane_count: int, nonempty: int) -> tuple[list[int], set[int]]:
    """Choose active lanes and the subset that should start as singles."""
    min_singles = max(SINGLE_LANES_MIN, nonempty - DOUBLE_LANES_MAX)
    max_singles = min(SINGLE_LANES_MAX, nonempty - DOUBLE_LANES_MIN)
    singles = random.randint(min_singles, max_singles)
    active_lanes = random.sample(range(lane_count), nonempty)
    return active_lanes, set(random.sample(active_lanes, singles))


def _apply_lane(g, out, edge, lane, marker, edge_color, is_single):
    """Render one input lane plus its expected transformed output cells."""
    near, far = _lane_cells(edge, g.shape, lane)
    first, second = near
    g[first] = marker
    if is_single:
        out[first] = marker
        out[second] = marker
        return

    g[second] = marker
    for r, c in far:
        out[r, c] = edge_color


def _gen_attempt() -> np.ndarray | None:
    edge = random.choice(EDGE_NAMES)
    nonempty = random.randint(NONEMPTY_LANES_MIN, NONEMPTY_LANES_MAX)
    h, w = _choose_shape(edge, nonempty)

    bg, edge_color, marker = random.sample(range(COLOR_COUNT), PALETTE_SIZE)
    g = np.full((h, w), bg, dtype=int)
    out = g.copy()

    _paint_edge_bar(g, out, edge, edge_color)
    lane_count = w if edge in {"top", "bottom"} else h
    active_lanes, single_lanes = _choose_lane_states(lane_count, nonempty)

    for lane in active_lanes:
        _apply_lane(g, out, edge, lane, marker, edge_color, lane in single_lanes)

    try:
        solved = main(g)
    except Exception:
        return None
    if not np.array_equal(solved, out):
        return None
    return g


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 758abdf0 input")
