from __future__ import annotations

import random
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np


# Keep the shared toolbox import compatible with SourceFileLoader.
_TOOLBOX_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_TOOLBOX_DIR) not in sys.path:
    sys.path.append(str(_TOOLBOX_DIR))

from arc_utils import Color, get_bg, in_bounds, paint_coords


Coord = tuple[int, int]

BACKGROUND = Color.BLACK
GRID_SIDE_RANGE = (21, 30)
OBJECT_COUNTS = (4, 3, 2)
TRACE_DOT_RANGE = (2, 5)
GENERATION_ATTEMPTS = 500
PLACEMENT_ATTEMPTS = 400
SIDES = ("top", "right", "bottom", "left")
SIDE_NORMAL = {
    "top": (-1, 0),
    "right": (0, 1),
    "bottom": (1, 0),
    "left": (0, -1),
}

# Cycling only the object count makes the first three generated training examples
# cover 4, 3, and 2 sources, while all geometry and colors remain random.
_generation_counter = 0


@dataclass(frozen=True)
class TraceSpec:
    side: str
    color: int
    direction: Coord
    dots: tuple[Coord, ...]
    marker: Coord
    body: tuple[Coord, ...]


def _add(a: Coord, b: Coord, scale: int = 1) -> Coord:
    return int(a[0] + scale * b[0]), int(a[1] + scale * b[1])


def _sprite_body(marker: Coord, direction: Coord) -> tuple[Coord, ...]:
    """Seven-cell 3x3 corner template whose centroid is behind marker."""
    br, bc = -int(direction[0]), -int(direction[1])
    cells = []
    for i in range(3):
        for j in range(3):
            # (0, 0) is the colored marker; the opposite corner is the
            # template's second missing cell, leaving exactly seven body cells.
            if (i, j) in {(0, 0), (2, 2)}:
                continue
            cells.append((marker[0] + i * br, marker[1] + j * bc))
    return tuple(cells)


def _side_directions(side: str) -> tuple[Coord, Coord]:
    if side == "top":
        return (-1, -1), (-1, 1)
    if side == "bottom":
        return (1, -1), (1, 1)
    if side == "left":
        return (-1, -1), (1, -1)
    assert side == "right"
    return (-1, 1), (1, 1)


def _is_side_adjacent(cell: Coord, side: str, shape: tuple[int, int]) -> bool:
    r, c = cell
    h, w = shape
    return (
        (side == "top" and r == 1)
        or (side == "bottom" and r == h - 2)
        or (side == "left" and c == 1)
        or (side == "right" and c == w - 2)
    )


def _edge_cell(cell: Coord, direction: Coord) -> Coord:
    return _add(cell, direction)


def _decode_trace(points: np.ndarray, side: str, shape: tuple[int, int]) -> tuple[Coord, Coord]:
    """Return the uniquely decoded (direction, marker) for one dotted trace."""
    pts = {tuple(map(int, rc)) for rc in np.asarray(points, dtype=int)}
    assert len(pts) >= 2, "each ray needs at least two visible dots"
    candidates: list[tuple[Coord, Coord]] = []

    for direction in _side_directions(side):
        dr, dc = direction
        ordered = sorted(pts, key=lambda p: p[0] * dr + p[1] * dc)
        far = ordered[0]
        expected = {_add(far, direction, 2 * k) for k in range(len(pts))}
        if expected != pts:
            continue
        near = _add(far, direction, 2 * (len(pts) - 1))
        if not _is_side_adjacent(near, side, shape):
            continue
        marker = _add(far, direction, -1)
        candidates.append((direction, marker))

    assert len(candidates) == 1, "painted edge and dotted slope must select one direction"
    return candidates[0]


def _painted_sides(g: np.ndarray, bg: int) -> dict[str, int]:
    h, w = g.shape
    assert h >= 7 and w >= 7
    interiors = {
        "top": g[0, 1 : w - 1],
        "right": g[1 : h - 1, w - 1],
        "bottom": g[h - 1, 1 : w - 1],
        "left": g[1 : h - 1, 0],
    }
    sides: dict[str, int] = {}
    for side, values in interiors.items():
        unique = set(map(int, values))
        if unique == {bg}:
            continue
        assert len(unique) == 1 and bg not in unique, "a painted edge interior must be solid"
        sides[side] = unique.pop()

    assert 2 <= len(sides) <= 4, "expected two to four painted edges"
    assert len(set(sides.values())) == len(sides), "every ray/edge must have its own color"

    corner_info = (
        ((0, 0), "top", "left"),
        ((0, w - 1), "top", "right"),
        ((h - 1, w - 1), "bottom", "right"),
        ((h - 1, 0), "bottom", "left"),
    )
    for (r, c), side_a, side_b in corner_info:
        active = [sides[s] for s in (side_a, side_b) if s in sides]
        expected = bg if len(active) != 1 else active[0]
        assert int(g[r, c]) == int(expected), "adjacent painted edges must share a black corner"
    return sides


def _legend_color(g: np.ndarray, bg: int, edge_colors: set[int]) -> tuple[int, Coord]:
    colors = set(map(int, np.unique(g))) - {int(bg)} - set(map(int, edge_colors))
    assert len(colors) == 1, "there must be one non-edge legend color"
    color = colors.pop()
    coords = np.argwhere(g == color)
    assert coords.shape == (1, 2), "the body-color legend must be a singleton"
    r, c = map(int, coords[0])
    assert r < max(4, g.shape[0] // 3) and c < max(4, g.shape[1] // 3), "legend must be in the upper-left"
    neighborhood = g[max(0, r - 1) : r + 2, max(0, c - 1) : c + 2]
    assert int(np.count_nonzero(neighborhood != bg)) == 1, "legend must be isolated"
    return int(color), (r, c)


def _analyze(grid: np.ndarray) -> tuple[int, int, list[TraceSpec]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert max(g.shape) <= 30, "ARC grids may not exceed 30x30"
    bg = int(get_bg(g))
    assert bg == BACKGROUND, "the trace canvas must have a black background"

    sides = _painted_sides(g, bg)
    body_color, legend = _legend_color(g, bg, set(sides.values()))
    h, w = g.shape
    specs: list[TraceSpec] = []

    for side in SIDES:
        if side not in sides:
            continue
        color = int(sides[side])
        points = np.argwhere(g[1 : h - 1, 1 : w - 1] == color) + np.array([1, 1])
        direction, marker = _decode_trace(points, side, g.shape)
        point_set = {tuple(map(int, rc)) for rc in points}
        far = _add(marker, direction)
        near = max(point_set, key=lambda p: p[0] * direction[0] + p[1] * direction[1])

        assert _edge_cell(near, direction) != near
        er, ec = _edge_cell(near, direction)
        assert in_bounds(g.shape, er, ec) and int(g[er, ec]) == color, "ray must point into its same-color edge"
        assert int(g[marker]) == bg, "the erased marker position must be blank"

        # Every odd phase is a visible dot, and every even phase is black.
        for k in range(2 * len(point_set) - 1):
            cell = _add(far, direction, k)
            if k % 2 == 0:
                assert cell in point_set
            else:
                assert int(g[cell]) == bg, "dotted rays must leave one black cell between dots"

        body = _sprite_body(marker, direction)
        assert len(body) == 7 and len(set(body)) == 7
        assert all(in_bounds(g.shape, r, c) for r, c in (marker, *body))
        centroid = np.mean(np.asarray(body, dtype=float), axis=0)
        displacement = centroid - np.asarray(marker, dtype=float)
        assert float(np.dot(displacement, np.asarray(direction, dtype=float))) < 0.0
        assert legend not in {marker, *body}, "legend cannot overlap a reconstructed sprite"
        specs.append(TraceSpec(side, color, direction, tuple(sorted(point_set)), marker, body))

    assert len(specs) == len(sides)
    output_sets = [{spec.marker, *spec.body} for spec in specs]
    for i in range(len(output_sets)):
        for j in range(i):
            assert output_sets[i].isdisjoint(output_sets[j]), "reconstructed sprites must not overlap"
    return bg, body_color, specs


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reconstruct erased directional sprites from their dotted rays and same-color painted destination edges.

    Input structure:
    - A black canvas has two to four differently colored diagonal dotted rays; each ray's destination edge is painted in its color, with a black shared corner when adjacent edges are painted.
    - A single isolated upper-left legend pixel gives the common body color, while every original sprite and marker has been erased.
    - Ray dots start one diagonal step beyond the missing marker and then occur every other cell up to the painted edge.

    Rule:
    - Match each ray color to its painted edge, use the ray slope and that edge to determine the forward diagonal direction, and extend the far endpoint one step backward to recover the marker.
    - Paint the marker in the ray color and place the same rotated/reflected seven-cell body-color template behind it, so the body's centroid lies opposite the ray direction.
    - Clear the legend, all dotted traces, and all painted edges; retain only the reconstructed sprites on black.

    Key insight:
    - The visible pixels are effects rather than sources: edge color resolves which end is forward, the odd/even dotted phase locates the missing marker, and the recovered direction orients the centroid-biased body.
    """
    g = np.asarray(grid, dtype=int)
    bg, body_color, specs = _analyze(g)
    out = np.full_like(g, bg)
    for spec in specs:
        out[spec.marker] = int(spec.color)
        paint_coords(out, spec.body, body_color)
    return out


def _chebyshev_clear(cells: set[Coord], other: set[Coord], gap: int) -> bool:
    for r, c in cells:
        for rr, cc in other:
            if max(abs(r - rr), abs(c - cc)) <= gap:
                return False
    return True


def _terminal_cells(side: str, shape: tuple[int, int]) -> list[Coord]:
    h, w = shape
    if side == "top":
        return [(1, c) for c in range(3, w - 3)]
    if side == "bottom":
        return [(h - 2, c) for c in range(3, w - 3)]
    if side == "left":
        return [(r, 1) for r in range(3, h - 3)]
    assert side == "right"
    return [(r, w - 2) for r in range(3, h - 3)]


def _candidate_spec(side: str, color: int, direction: Coord, shape: tuple[int, int]) -> TraceSpec | None:
    choices = [
        (terminal, n_dots)
        for terminal in _terminal_cells(side, shape)
        for n_dots in range(TRACE_DOT_RANGE[0], TRACE_DOT_RANGE[1] + 1)
    ]
    random.shuffle(choices)
    for terminal, n_dots in choices:
        far = _add(terminal, direction, -2 * (n_dots - 1))
        marker = _add(far, direction, -1)
        dots = tuple(_add(far, direction, 2 * k) for k in range(n_dots))
        body = _sprite_body(marker, direction)
        all_cells = (marker, *dots, *body)
        if not all(in_bounds(shape, r, c) for r, c in all_cells):
            continue
        if any(r in {0, shape[0] - 1} or c in {0, shape[1] - 1} for r, c in (marker, *body)):
            continue
        er, ec = _edge_cell(terminal, direction)
        if not in_bounds(shape, er, ec):
            continue
        if er not in {0, shape[0] - 1} and ec not in {0, shape[1] - 1}:
            continue
        # Avoid a ray aiming at a shared corner, which would be black by rule.
        if (er, ec) in {(0, 0), (0, shape[1] - 1), (shape[0] - 1, 0), (shape[0] - 1, shape[1] - 1)}:
            continue
        return TraceSpec(side, int(color), direction, dots, marker, body)
    return None


def _choose_sides(count: int) -> list[str]:
    if count == 4:
        sides = list(SIDES)
        random.shuffle(sides)
        return sides
    if count == 3:
        return random.sample(list(SIDES), 3)
    # Two perpendicular edge classes make both row- and column-referenced parsing visible.
    return [random.choice(("top", "bottom")), random.choice(("left", "right"))]


def _choose_directions(sides: list[str]) -> list[Coord]:
    if len(sides) == 4:
        # One occurrence of every diagonal direction in each four-source example.
        desired = [(-1, -1), (-1, 1), (1, 1), (1, -1)]
        for _ in range(100):
            random.shuffle(desired)
            if all(direction in _side_directions(side) for side, direction in zip(sides, desired)):
                return list(desired)
        # A valid perfect matching always exists; this is a deterministic fallback.
        return [random.choice(_side_directions(side)) for side in sides]

    for _ in range(100):
        directions = [random.choice(_side_directions(side)) for side in sides]
        slopes = {dr * dc for dr, dc in directions}
        if len(slopes) >= 2:
            return directions
    return [random.choice(_side_directions(side)) for side in sides]


def _place_specs(shape: tuple[int, int], sides: list[str], colors: list[int], directions: list[Coord]) -> list[TraceSpec] | None:
    for _ in range(PLACEMENT_ATTEMPTS):
        specs: list[TraceSpec] = []
        ok = True
        for side, color, direction in zip(sides, colors, directions):
            candidate = _candidate_spec(side, color, direction, shape)
            if candidate is None:
                ok = False
                break
            visible = set(candidate.dots)
            hidden = {candidate.marker, *candidate.body}
            for previous in specs:
                old_visible = set(previous.dots)
                old_hidden = {previous.marker, *previous.body}
                if not _chebyshev_clear(visible, old_visible, 1):
                    ok = False
                if not _chebyshev_clear(hidden, old_hidden, 1):
                    ok = False
                if not _chebyshev_clear(visible, old_hidden, 1):
                    ok = False
                if not _chebyshev_clear(hidden, old_visible, 1):
                    ok = False
            if not ok:
                break
            specs.append(candidate)
        if ok:
            return specs
    return None


def _paint_edges(grid: np.ndarray, specs: list[TraceSpec]) -> None:
    h, w = grid.shape
    side_colors = {spec.side: spec.color for spec in specs}
    for side, color in side_colors.items():
        if side == "top":
            grid[0, :] = color
        elif side == "bottom":
            grid[h - 1, :] = color
        elif side == "left":
            grid[:, 0] = color
        else:
            grid[:, w - 1] = color

    for (r, c), adjacent in (
        ((0, 0), ("top", "left")),
        ((0, w - 1), ("top", "right")),
        ((h - 1, w - 1), ("bottom", "right")),
        ((h - 1, 0), ("bottom", "left")),
    ):
        active = [side_colors[side] for side in adjacent if side in side_colors]
        grid[r, c] = BACKGROUND if len(active) != 1 else active[0]


def _independent_expected(shape: tuple[int, int], body_color: int, specs: list[TraceSpec]) -> np.ndarray:
    out = np.full(shape, BACKGROUND, dtype=int)
    for spec in specs:
        out[spec.marker] = spec.color
        for cell in spec.body:
            out[cell] = body_color
    return out


def _validate_generated(grid: np.ndarray, expected: np.ndarray, specs: list[TraceSpec], body_color: int) -> bool:
    """Check puzzle semantics independently of mere generate/main closure."""
    try:
        actual = main(grid)
        if not np.array_equal(actual, expected):
            return False
        if int(np.count_nonzero(actual)) != 8 * len(specs):
            return False
        if np.array_equal(actual, grid):
            return False

        decoded_bg, decoded_body, decoded = _analyze(grid)
        if decoded_bg != BACKGROUND or decoded_body != body_color:
            return False
        truth = {(s.side, s.color): (s.direction, s.marker) for s in specs}
        found = {(s.side, s.color): (s.direction, s.marker) for s in decoded}
        if truth != found:
            return False

        # The opposite endpoint gives a different possible source if the painted
        # edge is ignored; hence the edge direction is genuinely disambiguating.
        for spec in specs:
            near = spec.dots[-1]
            reverse_marker = _add(near, spec.direction)
            if reverse_marker == spec.marker:
                return False
            far = spec.dots[0]
            if _add(spec.marker, spec.direction) != far:
                return False
            for a, b in zip(spec.dots, spec.dots[1:]):
                if (b[0] - a[0], b[1] - a[1]) != (2 * spec.direction[0], 2 * spec.direction[1]):
                    return False
        return True
    except (AssertionError, IndexError, ValueError):
        return False


def _generation_attempt(count: int) -> np.ndarray | None:
    shape = (random.randint(*GRID_SIDE_RANGE), random.randint(*GRID_SIDE_RANGE))
    palette = random.sample(Color.NOT_BLACK, count + 1)
    body_color = int(palette[0])
    trace_colors = list(map(int, palette[1:]))
    sides = _choose_sides(count)
    directions = _choose_directions(sides)
    specs = _place_specs(shape, sides, trace_colors, directions)
    if specs is None:
        return None

    legend = (2, 2)
    visible_cells = {cell for spec in specs for cell in spec.dots}
    hidden_cells = {cell for spec in specs for cell in (spec.marker, *spec.body)}
    if not _chebyshev_clear({legend}, visible_cells, 1):
        return None
    if legend in hidden_cells:
        return None

    grid = np.full(shape, BACKGROUND, dtype=int)
    _paint_edges(grid, specs)
    for spec in specs:
        paint_coords(grid, spec.dots, spec.color)
    grid[legend] = body_color

    expected = _independent_expected(shape, body_color, specs)
    if not _validate_generated(grid, expected, specs, body_color):
        return None
    return grid


def generate_input() -> np.ndarray:
    global _generation_counter
    count = OBJECT_COUNTS[_generation_counter % len(OBJECT_COUNTS)]
    _generation_counter += 1
    for _ in range(GENERATION_ATTEMPTS):
        grid = _generation_attempt(count)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid erased-emitter reconstruction grid")
