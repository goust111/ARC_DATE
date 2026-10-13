from __future__ import annotations

from collections import Counter
import itertools

import numpy as np


Coord = tuple[int, int]


RING_SPECS = (
    {"dest_offset": (-1, -1), "step": (-1, -1), "family": "diag", "group": "nw"},
    {"dest_offset": (-1, 0), "step": (-1, 0), "family": "vertical", "group": "n"},
    {"dest_offset": (-1, 1), "step": (-1, 0), "family": "vertical", "group": "n"},
    {"dest_offset": (-1, 2), "step": (-1, 1), "family": "diag", "group": "ne"},
    {"dest_offset": (0, -1), "step": (0, -1), "family": "horizontal", "group": "w"},
    {"dest_offset": (1, -1), "step": (0, -1), "family": "horizontal", "group": "w"},
    {"dest_offset": (0, 2), "step": (0, 1), "family": "horizontal", "group": "e"},
    {"dest_offset": (1, 2), "step": (0, 1), "family": "horizontal", "group": "e"},
    {"dest_offset": (2, -1), "step": (1, -1), "family": "diag", "group": "sw"},
    {"dest_offset": (2, 0), "step": (1, 0), "family": "vertical", "group": "s"},
    {"dest_offset": (2, 1), "step": (1, 0), "family": "vertical", "group": "s"},
    {"dest_offset": (2, 2), "step": (1, 1), "family": "diag", "group": "se"},
)


GENERATOR_RETRIES = 500
GRID_SIDE_RANGE = (10, 19)
ANCHOR_MARGIN = 2
MARKER_COUNT_RANGE = (5, 8)
MARKER_DISTANCE_RANGE = (1, 6)
COLOR_POOL = tuple(range(1, 10))

_RNG = np.random.default_rng()


def _background_color(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _foreground_colors(grid: np.ndarray, bg: int) -> list[int]:
    return sorted(int(color) for color in np.unique(grid) if int(color) != bg)


def _project_marker_to_ring(r: int, c: int, r0: int, c0: int) -> Coord | None:
    r1 = r0 + 1
    c1 = c0 + 1

    if r == r0 and c <= c0 - 2:
        return r0, c0 - 1
    if r == r1 and c <= c0 - 2:
        return r1, c0 - 1
    if r == r0 and c >= c1 + 2:
        return r0, c1 + 1
    if r == r1 and c >= c1 + 2:
        return r1, c1 + 1

    if c == c0 and r <= r0 - 2:
        return r0 - 1, c0
    if c == c1 and r <= r0 - 2:
        return r0 - 1, c1
    if c == c0 and r >= r1 + 2:
        return r1 + 1, c0
    if c == c1 and r >= r1 + 2:
        return r1 + 1, c1

    if r <= r0 - 2 and c <= c0 - 2 and (r0 - r) == (c0 - c):
        return r0 - 1, c0 - 1
    if r <= r0 - 2 and c >= c1 + 2 and (r0 - r) == (c - c1):
        return r0 - 1, c1 + 1
    if r >= r1 + 2 and c <= c0 - 2 and (r - r1) == (c0 - c):
        return r1 + 1, c0 - 1
    if r >= r1 + 2 and c >= c1 + 2 and (r - r1) == (c - c1):
        return r1 + 1, c1 + 1

    return None


def _candidate_destinations(markers: list[Coord], shape: tuple[int, int], r0: int, c0: int) -> list[Coord] | None:
    height, width = map(int, shape)
    if not (1 <= r0 <= height - 3 and 1 <= c0 <= width - 3):
        return None

    destinations: list[Coord] = []
    seen: set[Coord] = set()
    for r, c in markers:
        dest = _project_marker_to_ring(int(r), int(c), int(r0), int(c0))
        if dest is None:
            return None
        if not (0 <= dest[0] < height and 0 <= dest[1] < width):
            return None
        if dest in seen:
            return None
        seen.add(dest)
        destinations.append(dest)
    return destinations


def _candidate_solutions(markers: list[Coord], shape: tuple[int, int]) -> list[tuple[Coord, list[Coord]]]:
    height, width = map(int, shape)
    solutions: list[tuple[Coord, list[Coord]]] = []
    for r0 in range(1, height - 2):
        for c0 in range(1, width - 2):
            destinations = _candidate_destinations(markers, shape, r0, c0)
            if destinations is not None:
                solutions.append(((int(r0), int(c0)), destinations))
    return solutions


def _solve_anchor(markers: list[Coord], shape: tuple[int, int]) -> tuple[Coord, list[Coord]]:
    solutions = _candidate_solutions(markers, shape)
    assert len(solutions) == 1, "expected exactly one hidden 2x2 anchor position"
    return solutions[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Infer the hidden 2x2 anchor from concurrent marker rays, draw it, and slide each marker onto the anchor ring.

    Input structure:
    - The grid has one dominant background color and exactly one foreground color.
    - Every foreground cell is a marker pixel; the solid 2x2 anchor block itself is absent from the input.
    - Each marker lies strictly outside the future block, on one of its two rows, two columns, or four 45-degree corner rays.

    Rule:
    - Find the unique 2x2 position whose row, column, and corner-ray constraints are satisfied by every marker.
    - Draw a filled 2x2 block of the marker color at that hidden anchor position.
    - Clear every original marker and move it inward along its own ray so it stops at the corresponding one-cell ring location.

    Key insight:
    - No single marker reveals the anchor by itself; the anchor only becomes visible as the unique common concurrency zone
      that satisfies all marker rays simultaneously, and only then can the forward sliding step be executed.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _background_color(g)
    colors = _foreground_colors(g, bg)
    assert len(colors) == 1, "expected exactly one foreground marker color"
    marker_color = int(colors[0])

    marker_coords = [tuple(map(int, coord)) for coord in np.argwhere(g == marker_color)]
    assert marker_coords, "expected at least one marker"

    (r0, c0), destinations = _solve_anchor(marker_coords, g.shape)

    out = np.full_like(g, bg)
    out[r0 : r0 + 2, c0 : c0 + 2] = marker_color
    for r, c in destinations:
        out[r, c] = marker_color
    return out


def _max_distance_from_destination(dest: Coord, step: Coord, shape: tuple[int, int]) -> int:
    height, width = map(int, shape)
    r, c = map(int, dest)
    dr, dc = map(int, step)

    row_room = height if dr == 0 else (r if dr < 0 else height - 1 - r)
    col_room = width if dc == 0 else (c if dc < 0 else width - 1 - c)
    return int(min(MARKER_DISTANCE_RANGE[1], row_room, col_room))


def _source_from_spec(anchor: Coord, spec: dict, dist: int) -> Coord:
    r0, c0 = anchor
    rr = r0 + int(spec["dest_offset"][0])
    cc = c0 + int(spec["dest_offset"][1])
    dr, dc = spec["step"]
    return int(rr + dr * dist), int(cc + dc * dist)


def _markers_are_isolated(markers: list[Coord]) -> bool:
    for i, (r1, c1) in enumerate(markers):
        for r2, c2 in markers[i + 1 :]:
            if max(abs(r1 - r2), abs(c1 - c2)) <= 1:
                return False
    return True


def _families_covered(specs: list[dict]) -> bool:
    families = {str(spec["family"]) for spec in specs}
    groups = {str(spec["group"]) for spec in specs}
    return families == {"horizontal", "vertical", "diag"} and len(groups) >= 4


def _choose_specs() -> list[dict] | None:
    marker_count = int(_RNG.integers(MARKER_COUNT_RANGE[0], MARKER_COUNT_RANGE[1]))
    for _ in range(80):
        idxs = _RNG.choice(len(RING_SPECS), size=marker_count, replace=False)
        specs = [RING_SPECS[int(idx)] for idx in idxs]
        if _families_covered(specs):
            return specs
    return None


def _build_input(shape: tuple[int, int], color: int, markers: list[Coord]) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    for r, c in markers:
        grid[int(r), int(c)] = int(color)
    return grid


def _pairwise_ambiguous(markers: list[Coord], shape: tuple[int, int]) -> bool:
    for pair in itertools.combinations(markers, 2):
        if len(_candidate_solutions(list(pair), shape)) <= 1:
            return False
    return True


def _single_marker_ambiguous(markers: list[Coord], shape: tuple[int, int]) -> bool:
    return all(len(_candidate_solutions([marker], shape)) > 1 for marker in markers)


def _distances_vary(distances: list[int]) -> bool:
    return max(distances) - min(distances) >= 2


def _semantic_checks(inp: np.ndarray, expected: np.ndarray, specs: list[dict], distances: list[int]) -> bool:
    color = int(_foreground_colors(inp, _background_color(inp))[0])
    markers = [tuple(map(int, coord)) for coord in np.argwhere(inp == color)]

    if not _markers_are_isolated(markers):
        return False
    if not _pairwise_ambiguous(markers, inp.shape):
        return False
    if not _single_marker_ambiguous(markers, inp.shape):
        return False
    if not _families_covered(specs):
        return False
    if not _distances_vary(distances):
        return False

    solutions = _candidate_solutions(markers, inp.shape)
    if len(solutions) != 1:
        return False

    try:
        actual = main(inp)
    except AssertionError:
        return False
    return np.array_equal(actual, expected)


def _gen_attempt() -> np.ndarray | None:
    height = int(_RNG.integers(GRID_SIDE_RANGE[0], GRID_SIDE_RANGE[1]))
    width = int(_RNG.integers(GRID_SIDE_RANGE[0], GRID_SIDE_RANGE[1]))
    color = int(_RNG.choice(COLOR_POOL))

    if height < 2 * ANCHOR_MARGIN + 2 or width < 2 * ANCHOR_MARGIN + 2:
        return None

    r0 = int(_RNG.integers(ANCHOR_MARGIN, height - ANCHOR_MARGIN - 1))
    c0 = int(_RNG.integers(ANCHOR_MARGIN, width - ANCHOR_MARGIN - 1))
    anchor = (r0, c0)

    specs = _choose_specs()
    if specs is None:
        return None

    markers: list[Coord] = []
    distances: list[int] = []
    for spec in specs:
        dest = (r0 + int(spec["dest_offset"][0]), c0 + int(spec["dest_offset"][1]))
        max_dist = _max_distance_from_destination(dest, spec["step"], (height, width))
        if max_dist < MARKER_DISTANCE_RANGE[0]:
            return None
        dist = int(_RNG.integers(MARKER_DISTANCE_RANGE[0], max_dist + 1))
        source = _source_from_spec(anchor, spec, dist)
        if source in markers:
            return None
        markers.append(source)
        distances.append(dist)

    if not _markers_are_isolated(markers):
        return None

    inp = _build_input((height, width), color, markers)
    expected = np.zeros((height, width), dtype=int)
    expected[r0 : r0 + 2, c0 : c0 + 2] = color
    for spec in specs:
        rr = r0 + int(spec["dest_offset"][0])
        cc = c0 + int(spec["dest_offset"][1])
        expected[rr, cc] = color

    if _semantic_checks(inp, expected, specs, distances):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(GENERATOR_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid hidden-anchor ARC input")
