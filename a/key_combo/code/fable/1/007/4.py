from __future__ import annotations

from collections import Counter

import numpy as np


BLACK = 0
RED = 2
PORTAL_COLORS = (1, 3, 4, 5, 6, 7, 8, 9)

TOP = 0
BOTTOM = 1
LEFT = 2
RIGHT = 3

GEN_ATTEMPTS = 600
RNG = np.random.default_rng()

Coord = tuple[int, int]


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, np.asarray(grid).ravel())).most_common(1)[0][0])


def _is_corner(shape: tuple[int, int], cell: Coord) -> bool:
    h, w = shape
    r, c = cell
    return (r, c) in ((0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1))


def _is_boundary(shape: tuple[int, int], cell: Coord) -> bool:
    h, w = shape
    r, c = cell
    return r in (0, h - 1) or c in (0, w - 1)


def _side(shape: tuple[int, int], cell: Coord) -> int:
    h, w = shape
    r, c = cell
    if r == 0:
        return TOP
    if r == h - 1:
        return BOTTOM
    if c == 0:
        return LEFT
    if c == w - 1:
        return RIGHT
    raise AssertionError("cell must be on the boundary")


def _axis(shape: tuple[int, int], cell: Coord) -> int:
    return 0 if _side(shape, cell) in (TOP, BOTTOM) else 1


def _inward_direction(shape: tuple[int, int], cell: Coord) -> Coord:
    side = _side(shape, cell)
    if side == TOP:
        return 1, 0
    if side == BOTTOM:
        return -1, 0
    if side == LEFT:
        return 0, 1
    return 0, -1


def _opposite_boundary(shape: tuple[int, int], cell: Coord) -> Coord:
    h, w = shape
    r, c = cell
    side = _side(shape, cell)
    if side == TOP:
        return h - 1, c
    if side == BOTTOM:
        return 0, c
    if side == LEFT:
        return r, w - 1
    return r, 0


def _segment_cells(src: Coord, dst: Coord, *, include_end: bool) -> list[Coord]:
    sr, sc = src
    dr = 0 if dst[0] == sr else (1 if dst[0] > sr else -1)
    dc = 0 if dst[1] == sc else (1 if dst[1] > sc else -1)
    cells: list[Coord] = []
    rr, cc = sr + dr, sc + dc
    while (rr, cc) != dst:
        cells.append((rr, cc))
        rr += dr
        cc += dc
    if include_end:
        cells.append(dst)
    return cells


def _boundary_cells(shape: tuple[int, int], side: int | None = None) -> list[Coord]:
    h, w = shape
    groups = {
        TOP: [(0, c) for c in range(1, w - 1)],
        BOTTOM: [(h - 1, c) for c in range(1, w - 1)],
        LEFT: [(r, 0) for r in range(1, h - 1)],
        RIGHT: [(r, w - 1) for r in range(1, h - 1)],
    }
    if side is None:
        out: list[Coord] = []
        for key in (TOP, BOTTOM, LEFT, RIGHT):
            out.extend(groups[key])
        return out
    return groups[side]


def _portal_lookup(grid: np.ndarray, bg: int) -> dict[Coord, Coord]:
    g = np.asarray(grid, dtype=int)
    lookup: dict[Coord, Coord] = {}
    colors = [int(color) for color in np.unique(g) if int(color) not in (bg, RED)]
    assert colors, "expected at least one portal pair"
    for color in colors:
        coords = np.argwhere(g == color)
        assert coords.shape == (2, 2), "each portal color must appear exactly twice"
        a = tuple(map(int, coords[0]))
        b = tuple(map(int, coords[1]))
        assert _is_boundary(g.shape, a) and _is_boundary(g.shape, b)
        assert not _is_corner(g.shape, a) and not _is_corner(g.shape, b)
        lookup[a] = b
        lookup[b] = a
    return lookup


def _trace_from_grid(grid: np.ndarray) -> tuple[np.ndarray, dict]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    assert 5 <= h <= 30 and 5 <= w <= 30

    bg = _get_bg(g)
    assert bg != RED
    if h > 2 and w > 2:
        assert np.all(g[1:-1, 1:-1] == bg), "only boundary markers are allowed in the input"

    starts = np.argwhere(g == RED)
    assert starts.shape == (1, 2), "expected exactly one red start marker"
    start = tuple(map(int, starts[0]))
    assert _is_boundary(g.shape, start) and not _is_corner(g.shape, start)

    lookup = _portal_lookup(g, bg)
    assert start not in lookup, "the red start cannot also be a portal"

    out = g.copy()
    current = start
    seen_sources: set[Coord] = set()
    used_pairs: list[tuple[Coord, Coord]] = []

    while True:
        assert current not in seen_sources, "portal traversal must terminate"
        seen_sources.add(current)
        hit = _opposite_boundary(g.shape, current)

        for cell in _segment_cells(current, hit, include_end=False):
            assert int(g[cell]) in (bg, RED), "interior cells must be background"
            out[cell] = RED

        twin = lookup.get(hit)
        if twin is None:
            out[hit] = RED
            return out, {"start": start, "used_pairs": used_pairs, "stop": hit}

        used_pairs.append((hit, twin))
        current = twin


def main(grid: np.ndarray) -> np.ndarray:
    """Draw the full red ray across stitched border portals until it stops at a non-portal border cell.

    Input structure:
    - A single background color fills the grid interior.
    - Exactly one red start marker lies on the border, never on a corner.
    - One or more other colors each appear exactly twice on the border as same-colored portal pairs.

    Rule:
    - The red ray starts at the red border marker and moves straight inward, perpendicular to that border side.
    - Whenever the ray reaches a portal border cell, it jumps to the other portal of the same color and resumes from there in the inward direction of the new border side; portal cells keep their original color.
    - When the ray reaches a border cell that is not a portal, that border cell is painted red and the trace stops.

    Key insight:
    - Each same-colored border pair stitches two boundary locations together, so the trace is determined by border-to-border travel plus portal pairing, not by local turning or nearest-neighbor geometry.
    """
    out, _info = _trace_from_grid(grid)
    return out


def _render_from_sources(grid: np.ndarray, sources: list[Coord]) -> tuple[np.ndarray, dict]:
    out = np.array(grid, copy=True)
    shape = out.shape
    visit_counts: dict[Coord, int] = {}
    used_pairs: list[tuple[Coord, Coord]] = []

    current = sources[0]
    for nxt in sources[1:]:
        hit = _opposite_boundary(shape, current)
        for cell in _segment_cells(current, hit, include_end=False):
            visit_counts[cell] = visit_counts.get(cell, 0) + 1
            out[cell] = RED
        used_pairs.append((hit, nxt))
        current = nxt

    stop = _opposite_boundary(shape, current)
    for cell in _segment_cells(current, stop, include_end=True):
        visit_counts[cell] = visit_counts.get(cell, 0) + 1
        out[cell] = RED

    return out, {
        "used_pairs": used_pairs,
        "stop": stop,
        "crossing": any(count > 1 for count in visit_counts.values()),
        "painted": len(visit_counts),
    }


def _render_without_teleport(grid: np.ndarray, start: Coord) -> np.ndarray:
    out = np.array(grid, copy=True)
    stop = _opposite_boundary(out.shape, start)
    for cell in _segment_cells(start, stop, include_end=True):
        out[cell] = RED
    return out


def _make_side_sequence(teleports: int, expect_crossing: bool) -> list[int]:
    start_side = int(RNG.integers(0, 4))
    seq = [start_side]
    if expect_crossing:
        for _ in range(teleports):
            seq.append(int(RNG.integers(0, 4)))
        axes = {0 if side in (TOP, BOTTOM) else 1 for side in seq}
        if len(axes) == 1:
            other_axis = 1 - next(iter(axes))
            seq[int(RNG.integers(1, len(seq)))] = int(
                RNG.choice([TOP, BOTTOM] if other_axis == 0 else [LEFT, RIGHT])
            )
    else:
        same_axis_sides = [TOP, BOTTOM] if start_side in (TOP, BOTTOM) else [LEFT, RIGHT]
        other_side = same_axis_sides[1] if start_side == same_axis_sides[0] else same_axis_sides[0]
        for _ in range(teleports):
            seq.append(int(RNG.choice(same_axis_sides)))
        if all(side == start_side for side in seq[1:]):
            seq[int(RNG.integers(1, len(seq)))] = other_side
    return seq


def _sample_sources(shape: tuple[int, int], side_seq: list[int]) -> list[Coord] | None:
    candidates_by_side = {side: _boundary_cells(shape, side) for side in (TOP, BOTTOM, LEFT, RIGHT)}
    for _ in range(150):
        used: set[Coord] = set()
        sources: list[Coord] = []

        start_choices = [
            cell for cell in candidates_by_side[side_seq[0]] if _opposite_boundary(shape, cell) not in used
        ]
        if not start_choices:
            return None
        start = start_choices[int(RNG.integers(len(start_choices)))]
        sources.append(start)
        used.add(start)

        ok = True
        for next_side in side_seq[1:]:
            hit = _opposite_boundary(shape, sources[-1])
            if hit in used:
                ok = False
                break
            used.add(hit)

            choices = [
                cell
                for cell in candidates_by_side[next_side]
                if cell not in used and _opposite_boundary(shape, cell) not in used
            ]
            if not choices:
                ok = False
                break
            nxt = choices[int(RNG.integers(len(choices)))]
            sources.append(nxt)
            used.add(nxt)

        if ok and _opposite_boundary(shape, sources[-1]) not in used:
            return sources
    return None


def _render_input(shape: tuple[int, int], sources: list[Coord]) -> np.ndarray:
    grid = np.full(shape, BLACK, dtype=int)
    grid[sources[0]] = RED
    used_pairs = [(_opposite_boundary(shape, sources[i]), sources[i + 1]) for i in range(len(sources) - 1)]
    colors = RNG.choice(np.array(PORTAL_COLORS, dtype=int), size=len(used_pairs), replace=False)
    for color, (a, b) in zip(colors.tolist(), used_pairs):
        grid[a] = int(color)
        grid[b] = int(color)
    return grid


def _validate_generated(grid: np.ndarray, sources: list[Coord], expect_crossing: bool) -> bool:
    bg = _get_bg(grid)
    expected, plan = _render_from_sources(grid, sources)
    actual, traced = _trace_from_grid(grid)
    if not np.array_equal(actual, expected):
        return False
    if traced["start"] != sources[0]:
        return False

    planned_pairs = [(_opposite_boundary(grid.shape, sources[i]), sources[i + 1]) for i in range(len(sources) - 1)]
    if traced["used_pairs"] != planned_pairs:
        return False
    if traced["stop"] != _opposite_boundary(grid.shape, sources[-1]):
        return False

    axes = {_axis(grid.shape, cell) for cell in sources}
    if expect_crossing != (len(axes) == 2):
        return False
    if plan["crossing"] != expect_crossing:
        return False

    side_changes = sum(_side(grid.shape, a) != _side(grid.shape, b) for a, b in zip(sources, sources[1:]))
    if side_changes == 0:
        return False

    if np.array_equal(_render_without_teleport(grid, sources[0]), expected):
        return False

    painted = int(plan["painted"])
    if painted < max(6, min(grid.shape) - 2):
        return False
    if painted * 3 >= grid.shape[0] * grid.shape[1] * 2:
        return False

    marks = [tuple(map(int, rc)) for rc in np.argwhere(grid != bg)]
    if len(marks) != 1 + 2 * (len(sources) - 1):
        return False
    for cell in marks:
        if not _is_boundary(grid.shape, cell) or _is_corner(grid.shape, cell):
            return False

    return True


def _gen_attempt(expect_crossing: bool) -> np.ndarray | None:
    height = int(RNG.integers(8, 20))
    width = int(RNG.integers(8, 20))
    teleports = int(RNG.choice(np.array([1, 2, 2, 3, 3, 4], dtype=int)))

    side_seq = _make_side_sequence(teleports, expect_crossing)
    sources = _sample_sources((height, width), side_seq)
    if sources is None:
        return None

    grid = _render_input((height, width), sources)
    if not _validate_generated(grid, sources, expect_crossing):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        expect_crossing = bool(RNG.integers(0, 2))
        grid = _gen_attempt(expect_crossing)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid stitched-portal ray input")
