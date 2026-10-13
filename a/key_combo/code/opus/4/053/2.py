from __future__ import annotations

import random
from itertools import combinations

import numpy as np


Coord = tuple[int, int]
Pair = tuple[Coord, Coord]

BLACK = 0
BLUE = 1
RED = 2
GREEN = 3

MIN_ARM = 1
MAX_ARM = 4
MIN_PAIRS = 2
MAX_PAIRS = 3
MIN_PAIR_GAP = 6
MAX_GENERATION_ATTEMPTS = 500
RED_SAFE_DISTANCE = 4
GRID_HEIGHT_RANGE = (14, 22)
GRID_WIDTH_RANGE = (14, 22)
LAYOUT_MODES = ("horizontal", "vertical", "mixed")


def _sorted_coord(coord: Coord) -> Coord:
    return int(coord[0]), int(coord[1])


def _canonical_pair(a: Coord, b: Coord) -> Pair:
    aa = _sorted_coord(a)
    bb = _sorted_coord(b)
    return (aa, bb) if aa <= bb else (bb, aa)


def _pair_midpoint(pair: Pair) -> Coord:
    (r1, c1), (r2, c2) = pair
    if r1 == r2:
        gap = abs(c1 - c2)
        assert gap >= MIN_PAIR_GAP and gap % 2 == 0
        return int(r1), int((c1 + c2) // 2)
    if c1 == c2:
        gap = abs(r1 - r2)
        assert gap >= MIN_PAIR_GAP and gap % 2 == 0
        return int((r1 + r2) // 2), int(c1)
    raise AssertionError("blue pair must be axis-aligned")


def _cross_cells(center: Coord, arm: int) -> set[Coord]:
    r, c = center
    cells = {(int(r), int(c))}
    for d in range(1, int(arm) + 1):
        cells.add((int(r - d), int(c)))
        cells.add((int(r + d), int(c)))
        cells.add((int(r), int(c - d)))
        cells.add((int(r), int(c + d)))
    return cells


def _draw_cross(grid: np.ndarray, center: Coord, arm: int, color: int) -> None:
    for r, c in _cross_cells(center, arm):
        assert 0 <= r < grid.shape[0] and 0 <= c < grid.shape[1]
        grid[r, c] = int(color)


def _blue_candidates(blue_coords: list[Coord]) -> list[Pair]:
    candidates: list[Pair] = []
    for a, b in combinations(sorted(blue_coords), 2):
        if a[0] == b[0]:
            gap = abs(a[1] - b[1])
            if gap >= MIN_PAIR_GAP and gap % 2 == 0:
                candidates.append(_canonical_pair(a, b))
        elif a[1] == b[1]:
            gap = abs(a[0] - b[0])
            if gap >= MIN_PAIR_GAP and gap % 2 == 0:
                candidates.append(_canonical_pair(a, b))
    return candidates


def _solve_blue_pairs(blue_coords: list[Coord]) -> list[Pair]:
    coords = sorted(_sorted_coord(coord) for coord in blue_coords)
    assert MIN_PAIRS * 2 <= len(coords) <= MAX_PAIRS * 2 and len(coords) % 2 == 0
    coord_to_idx = {coord: idx for idx, coord in enumerate(coords)}
    candidates = _blue_candidates(coords)
    assert candidates, "expected at least one valid blue pairing candidate"

    indexed: list[tuple[int, Pair]] = []
    by_index: list[list[tuple[int, Pair]]] = [[] for _ in coords]
    for pair in candidates:
        mask = 0
        for coord in pair:
            mask |= 1 << coord_to_idx[coord]
        indexed.append((mask, pair))
        for coord in pair:
            by_index[coord_to_idx[coord]].append((mask, pair))

    full_mask = (1 << len(coords)) - 1
    solutions: list[tuple[Pair, ...]] = []

    def search(remaining: int, chosen: list[Pair]) -> None:
        if len(solutions) > 1:
            return
        if remaining == 0:
            solutions.append(tuple(sorted(chosen)))
            return

        first_idx = (remaining & -remaining).bit_length() - 1
        for mask, pair in by_index[first_idx]:
            if mask & ~remaining:
                continue
            search(remaining & ~mask, chosen + [pair])

    search(full_mask, [])
    unique = {solution for solution in solutions}
    assert len(unique) == 1, "blue pairing must be unique"
    return list(next(iter(unique)))


def _normalize_pairs(pairs: list[Pair]) -> tuple[Pair, ...]:
    return tuple(sorted(_canonical_pair(a, b) for a, b in pairs))


def _render_output_from_pairs(inp: np.ndarray, pairs: list[Pair], arm: int) -> np.ndarray:
    out = inp.copy()
    for pair in pairs:
        _draw_cross(out, _pair_midpoint(pair), arm, GREEN)
    out[inp == BLUE] = BLUE
    out[inp == RED] = RED
    return out


def _wrong_center(center: Coord, arm: int, shape: tuple[int, int]) -> Coord | None:
    height, width = shape
    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        nr = center[0] + dr
        nc = center[1] + dc
        if arm <= nr < height - arm and arm <= nc < width - arm:
            return int(nr), int(nc)
    return None


def _arm_fits(center: Coord, arm: int, shape: tuple[int, int]) -> bool:
    r, c = center
    height, width = shape
    return arm <= r < height - arm and arm <= c < width - arm


def main(grid: np.ndarray) -> np.ndarray:
    """
    Count the red dots and use that count as the arm length for green crosses centered at each blue-pair midpoint.

    Input structure:
    - The grid has a black background with 2 to 3 pairs of blue pixels and a
      small number of red scatter pixels.
    - Each blue pair is aligned horizontally or vertically, has even spacing of
      at least six cells, and the overall blue layout admits a unique pairing.
    - The red dots are independent markers whose count is the only size cue for
      the inserted shapes.

    Rule:
    - Count all red pixels and call that number N.
    - Pair the blue pixels into the unique set of axis-aligned pairs, then take
      the midpoint of each pair.
    - At every midpoint, draw a green plus whose center stays fixed and whose
      four arms extend exactly N cells.
    - Keep the original blue and red pixels unchanged by restoring them after
      the green crosses are drawn.

    Key insight:
    - The red dots determine one global geometric parameter, while the blue
      pairs determine only the cross centers; the output requires combining both.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    arm = int(np.count_nonzero(g == RED))
    assert arm >= 1

    blue_coords = [tuple(map(int, coord)) for coord in np.argwhere(g == BLUE)]
    pairs = _solve_blue_pairs(blue_coords)
    return _render_output_from_pairs(g, pairs, arm)


def _sample_centers(height: int, width: int, arm: int, pair_count: int) -> list[Coord] | None:
    candidates = [
        (r, c)
        for r in range(arm, height - arm)
        for c in range(arm, width - arm)
    ]
    random.shuffle(candidates)

    centers: list[Coord] = []
    occupied: set[Coord] = set()
    for center in candidates:
        cells = _cross_cells(center, arm)
        if occupied & cells:
            continue
        centers.append(center)
        occupied |= cells
        if len(centers) == pair_count:
            return centers
    return None


def _axes_for_mode(mode: str, pair_count: int) -> list[str]:
    if mode == "horizontal":
        return ["h"] * pair_count
    if mode == "vertical":
        return ["v"] * pair_count
    axes = ["h", "v"]
    if pair_count == 3:
        axes.append(random.choice(["h", "v"]))
    random.shuffle(axes)
    return axes


def _sample_halfspan(
    center: Coord,
    axis: str,
    shape: tuple[int, int],
    arm: int,
    forbidden: set[Coord],
) -> int | None:
    r, c = center
    height, width = shape
    min_halfspan = max(3, arm + 2)
    if axis == "h":
        max_halfspan = min(c, width - 1 - c)
        endpoints = lambda span: ((r, c - span), (r, c + span))
    else:
        max_halfspan = min(r, height - 1 - r)
        endpoints = lambda span: ((r - span, c), (r + span, c))

    candidates = list(range(min_halfspan, max_halfspan + 1))
    random.shuffle(candidates)
    for span in candidates:
        a, b = endpoints(span)
        if a in forbidden or b in forbidden:
            continue
        return span
    return None


def _sample_pairs(
    centers: list[Coord],
    axes: list[str],
    shape: tuple[int, int],
    arm: int,
) -> list[Pair] | None:
    cross_occupied = set().union(*(_cross_cells(center, arm) for center in centers))
    pairs: list[Pair] = []
    used_blue: set[Coord] = set()

    order = list(range(len(centers)))
    random.shuffle(order)
    placed: dict[int, Pair] = {}
    for idx in order:
        center = centers[idx]
        axis = axes[idx]
        span = _sample_halfspan(center, axis, shape, arm, cross_occupied | used_blue)
        if span is None:
            return None
        r, c = center
        if axis == "h":
            pair = _canonical_pair((r, c - span), (r, c + span))
        else:
            pair = _canonical_pair((r - span, c), (r + span, c))
        placed[idx] = pair
        used_blue.update(pair)

    for idx in range(len(centers)):
        pairs.append(placed[idx])
    return pairs


def _sample_red_points(
    height: int,
    width: int,
    arm: int,
    forbidden: set[Coord],
    centers: list[Coord],
) -> list[Coord] | None:
    candidates = []
    for r in range(height):
        for c in range(width):
            coord = (r, c)
            if coord in forbidden:
                continue
            if any(max(abs(r - mr), abs(c - mc)) <= RED_SAFE_DISTANCE for mr, mc in centers):
                continue
            candidates.append(coord)

    random.shuffle(candidates)
    chosen: list[Coord] = []
    for coord in candidates:
        if any(max(abs(coord[0] - other[0]), abs(coord[1] - other[1])) <= 1 for other in chosen):
            continue
        chosen.append(coord)
        if len(chosen) == arm:
            return chosen
    return None


def _build_input(shape: tuple[int, int], pairs: list[Pair], red_points: list[Coord]) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    for a, b in pairs:
        grid[a] = BLUE
        grid[b] = BLUE
    for coord in red_points:
        grid[coord] = RED
    return grid


def _semantic_checks(inp: np.ndarray, pairs: list[Pair], arm: int) -> bool:
    actual_pairs = _normalize_pairs(pairs)
    try:
        solved_pairs = _normalize_pairs(
            _solve_blue_pairs([tuple(map(int, x)) for x in np.argwhere(inp == BLUE)])
        )
    except AssertionError:
        return False
    if actual_pairs != solved_pairs:
        return False

    expected = _render_output_from_pairs(inp, list(actual_pairs), arm)
    try:
        actual = main(inp)
    except Exception:
        return False
    if not np.array_equal(actual, expected):
        return False

    alt_candidates = [candidate for candidate in (arm - 1, arm + 1) if candidate >= 1]
    alt_arm = next(
        (
            candidate
            for candidate in alt_candidates
            if all(_arm_fits(_pair_midpoint(pair), candidate, inp.shape) for pair in actual_pairs)
        ),
        None,
    )
    if alt_arm is not None:
        alt = _render_output_from_pairs(inp, list(actual_pairs), alt_arm)
        if np.array_equal(alt, expected):
            return False

    wrong_centers = [_pair_midpoint(pair) for pair in actual_pairs]
    replacement = _wrong_center(wrong_centers[0], arm, inp.shape)
    if replacement is None:
        return False
    wrong_centers[0] = replacement
    moved = inp.copy()
    for center in wrong_centers:
        _draw_cross(moved, center, arm, GREEN)
    moved[inp == BLUE] = BLUE
    moved[inp == RED] = RED
    if np.array_equal(moved, expected):
        return False

    return True


def _attempt_generate() -> np.ndarray | None:
    height = random.randint(*GRID_HEIGHT_RANGE)
    width = random.randint(*GRID_WIDTH_RANGE)
    arm = random.randint(MIN_ARM, MAX_ARM)
    pair_count = random.randint(MIN_PAIRS, MAX_PAIRS)
    mode = random.choice(LAYOUT_MODES)

    centers = _sample_centers(height, width, arm, pair_count)
    if centers is None:
        return None

    axes = _axes_for_mode(mode, pair_count)
    pairs = _sample_pairs(centers, axes, (height, width), arm)
    if pairs is None:
        return None

    blue_cells = {cell for pair in pairs for cell in pair}
    red_points = _sample_red_points(height, width, arm, blue_cells, centers)
    if red_points is None:
        return None

    inp = _build_input((height, width), pairs, red_points)
    if not _semantic_checks(inp, pairs, arm):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for ARC task 053/2")
