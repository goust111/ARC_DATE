from __future__ import annotations

import random

import numpy as np


BLACK = 0
GREEN = 3
GREY = 5
TEAL = 8

ORIENTATION_RAYS = {
    "up": ((-1, -1), (-1, 1)),
    "down": ((1, -1), (1, 1)),
    "left": ((-1, -1), (1, -1)),
    "right": ((-1, 1), (1, 1)),
}

TOP_HEIGHT_RANGE = (7, 12)  # inclusive via random.randint
WIDTH_RANGE = (11, 19)  # inclusive via random.randint
WEDGE_HEIGHTS = (2, 3, 4, 5)
MAX_GENERATION_ATTEMPTS = 400
TIP_MIN_GAP = 4
REGION_MIN_MANHATTAN_GAP = 3


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _find_separator_row(grid: np.ndarray) -> int:
    rows = [r for r in range(grid.shape[0]) if np.all(grid[r] == GREY)]
    if len(rows) != 1:
        raise AssertionError("expected exactly one solid grey separator row")
    sep = int(rows[0])
    if grid.shape[0] != 2 * sep + 1:
        raise AssertionError("separator must split equal-height top and bottom halves")
    return sep


def _mirror_true_mask(grid: np.ndarray) -> tuple[int, np.ndarray]:
    sep = _find_separator_row(grid)
    top = grid[:sep]
    true_mask = np.zeros(top.shape, dtype=bool)
    for r_raw, c_raw in np.argwhere(top == GREEN):
        r = int(r_raw)
        c = int(c_raw)
        if int(grid[2 * sep - r, c]) == GREEN:
            true_mask[r, c] = True
    return sep, true_mask


def _ray_first_distance(mask: np.ndarray, r: int, c: int, dr: int, dc: int) -> int | None:
    d = 1
    while _in_bounds(mask.shape, r + dr * d, c + dc * d):
        if mask[r + dr * d, c + dc * d]:
            return d
        d += 1
    return None


def _ray_prefix_len(mask: np.ndarray, r: int, c: int, dr: int, dc: int) -> int:
    d = 1
    while _in_bounds(mask.shape, r + dr * d, c + dc * d) and mask[r + dr * d, c + dc * d]:
        d += 1
    return d - 1


def _wedge_cells(
    orientation: str, r: int, c: int, height: int, shape: tuple[int, int]
) -> list[tuple[int, int]]:
    cells: list[tuple[int, int]] = []
    if orientation == "down":
        for d in range(1, height + 1):
            rr = r + d
            for cc in range(c - d, c + d + 1):
                if _in_bounds(shape, rr, cc):
                    cells.append((rr, cc))
    elif orientation == "up":
        for d in range(1, height + 1):
            rr = r - d
            for cc in range(c - d, c + d + 1):
                if _in_bounds(shape, rr, cc):
                    cells.append((rr, cc))
    elif orientation == "left":
        for d in range(1, height + 1):
            cc = c - d
            for rr in range(r - d, r + d + 1):
                if _in_bounds(shape, rr, cc):
                    cells.append((rr, cc))
    elif orientation == "right":
        for d in range(1, height + 1):
            cc = c + d
            for rr in range(r - d, r + d + 1):
                if _in_bounds(shape, rr, cc):
                    cells.append((rr, cc))
    else:
        raise AssertionError(f"unexpected orientation: {orientation}")
    return cells


def _find_apexes(true_mask: np.ndarray) -> list[tuple[int, int, str, int]]:
    apexes: list[tuple[int, int, str, int]] = []
    for r_raw, c_raw in np.argwhere(true_mask):
        r = int(r_raw)
        c = int(c_raw)
        for orientation, (ray_a, ray_b) in ORIENTATION_RAYS.items():
            nearest_a = _ray_first_distance(true_mask, r, c, *ray_a)
            nearest_b = _ray_first_distance(true_mask, r, c, *ray_b)
            if nearest_a is None or nearest_b is None or nearest_a != nearest_b:
                continue
            height = min(
                _ray_prefix_len(true_mask, r, c, *ray_a),
                _ray_prefix_len(true_mask, r, c, *ray_b),
            )
            if height >= nearest_a and height >= 1:
                apexes.append((r, c, orientation, height))
    return apexes


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use mirrored lower-half pixels to verify true markers, then fill the wedges they define.

    Input structure:
    - Black background with a full grey horizontal separator row splitting the grid into equal top and bottom halves.
    - The top half contains green pixels, some of which are true markers and some of which are distractors.
    - The bottom half contains green pixels only at mirror positions of the upper true markers across the separator.

    Rule:
    - Keep only upper green pixels whose mirror position below the separator is also green; those are the true markers.
    - Clear all upper distractors, all lower green mirror pixels, and the grey separator row to black.
    - In the surviving top markers, any pixel that has matched diagonal marker rays in one wedge orientation is an apex; the wedge height is the shared contiguous diagonal run length from that apex.
    - Fill every non-green cell inside each apex-and-rays triangular wedge with teal, while preserving the true green markers.

    Key insight:
    - The lower half is not output content; it is a verification plate that filters out fake upper markers before wedge geometry can be recovered.
    """
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2:
        raise AssertionError("expected a 2D grid")

    sep, true_mask = _mirror_true_mask(g)
    out = np.zeros_like(g)
    out[:sep][true_mask] = GREEN

    for r, c, orientation, height in _find_apexes(true_mask):
        for rr, cc in _wedge_cells(orientation, r, c, height, true_mask.shape):
            if out[rr, cc] != GREEN:
                out[rr, cc] = TEAL

    return out


def _boundary_points(orientation: str, r: int, c: int, height: int) -> set[tuple[int, int]]:
    ray_a, ray_b = ORIENTATION_RAYS[orientation]
    pts = {(r, c)}
    for d in range(1, height + 1):
        pts.add((r + ray_a[0] * d, c + ray_a[1] * d))
        pts.add((r + ray_b[0] * d, c + ray_b[1] * d))
    return pts


def _wedge_region(
    orientation: str, r: int, c: int, height: int, shape: tuple[int, int]
) -> set[tuple[int, int]]:
    return _boundary_points(orientation, r, c, height) | set(
        _wedge_cells(orientation, r, c, height, shape)
    )


def _candidate_specs(
    shape: tuple[int, int],
) -> list[tuple[str, int, int, int, set[tuple[int, int]], set[tuple[int, int]]]]:
    specs: list[tuple[str, int, int, int, set[tuple[int, int]], set[tuple[int, int]]]] = []
    H, W = shape
    for orientation in ORIENTATION_RAYS:
        for height in WEDGE_HEIGHTS:
            for r in range(H):
                for c in range(W):
                    region = _wedge_region(orientation, r, c, height, shape)
                    if not all(_in_bounds(shape, rr, cc) for rr, cc in region):
                        continue
                    boundary = _boundary_points(orientation, r, c, height)
                    specs.append((orientation, r, c, height, boundary, region))
    random.shuffle(specs)
    return specs


def _regions_far_enough(region: set[tuple[int, int]], used: set[tuple[int, int]]) -> bool:
    for r1, c1 in region:
        for r2, c2 in used:
            if abs(r1 - r2) + abs(c1 - c2) < REGION_MIN_MANHATTAN_GAP:
                return False
    return True


def _choose_wedges(
    shape: tuple[int, int],
) -> list[tuple[str, int, int, int, set[tuple[int, int]], set[tuple[int, int]]]] | None:
    target = 1 if random.random() < 0.45 else 2
    chosen: list[tuple[str, int, int, int, set[tuple[int, int]], set[tuple[int, int]]]] = []
    used: set[tuple[int, int]] = set()

    for spec in _candidate_specs(shape):
        orientation, r, c, height, boundary, region = spec
        if any(abs(r - r2) + abs(c - c2) < TIP_MIN_GAP for _, r2, c2, _, _, _ in chosen):
            continue
        if not _regions_far_enough(region, used):
            continue
        chosen.append((orientation, r, c, height, boundary, region))
        used |= region
        if len(chosen) == target:
            return chosen
    return None


def _build_expected_output(
    full_shape: tuple[int, int],
    sep: int,
    chosen: list[tuple[str, int, int, int, set[tuple[int, int]], set[tuple[int, int]]]],
) -> tuple[np.ndarray, np.ndarray]:
    top_shape = (sep, full_shape[1])
    expected = np.zeros(full_shape, dtype=int)
    true_mask = np.zeros(top_shape, dtype=bool)

    for orientation, r, c, height, boundary, _region in chosen:
        for rr, cc in boundary:
            true_mask[rr, cc] = True
            expected[rr, cc] = GREEN
        for rr, cc in _wedge_cells(orientation, r, c, height, top_shape):
            if not true_mask[rr, cc]:
                expected[rr, cc] = TEAL

    expected[:sep][true_mask] = GREEN
    return expected, true_mask


def _place_distractors(
    grid: np.ndarray,
    sep: int,
    forbidden: set[tuple[int, int]],
    count: int,
) -> bool:
    placed = 0
    tries = 0
    while placed < count and tries < 1200:
        tries += 1
        r = random.randrange(sep)
        c = random.randrange(grid.shape[1])
        if (r, c) in forbidden or int(grid[r, c]) != BLACK:
            continue
        if int(grid[2 * sep - r, c]) == GREEN:
            continue
        grid[r, c] = GREEN
        placed += 1
    return placed == count


def _validate_instance(
    inp: np.ndarray,
    expected: np.ndarray,
    chosen: list[tuple[str, int, int, int, set[tuple[int, int]], set[tuple[int, int]]]],
) -> bool:
    try:
        sep, true_mask = _mirror_true_mask(inp)
    except AssertionError:
        return False

    intended_true = np.zeros(true_mask.shape, dtype=bool)
    apex_targets: dict[tuple[int, int], tuple[str, int]] = {}
    forbidden: set[tuple[int, int]] = set()
    for orientation, r, c, height, boundary, region in chosen:
        for rr, cc in boundary:
            intended_true[rr, cc] = True
        apex_targets[(r, c)] = (orientation, height)
        forbidden |= region

    if not np.array_equal(true_mask, intended_true):
        return False
    if np.count_nonzero(inp[:sep] == GREEN) <= np.count_nonzero(intended_true):
        return False

    apexes = _find_apexes(true_mask)
    if len(apexes) != len(chosen):
        return False

    seen_tips: set[tuple[int, int]] = set()
    for r, c, orientation, height in apexes:
        tip = (r, c)
        if tip in seen_tips:
            return False
        seen_tips.add(tip)
        if apex_targets.get(tip) != (orientation, height):
            return False

    out = main(inp)
    if not np.array_equal(out, expected):
        return False
    if np.any(out[sep:]):
        return False
    if np.count_nonzero(out == TEAL) < len(chosen):
        return False

    for r_raw, c_raw in np.argwhere(inp[:sep] == GREEN):
        r = int(r_raw)
        c = int(c_raw)
        if not intended_true[r, c] and (r, c) in forbidden:
            return False

    return True


def _attempt_generate() -> np.ndarray | None:
    top_height = random.randint(*TOP_HEIGHT_RANGE)
    width = random.randint(*WIDTH_RANGE)
    sep = top_height
    shape = (2 * top_height + 1, width)
    chosen = _choose_wedges((top_height, width))
    if chosen is None:
        return None

    expected, true_mask = _build_expected_output(shape, sep, chosen)
    grid = np.zeros(shape, dtype=int)
    grid[sep, :] = GREY

    forbidden: set[tuple[int, int]] = set()
    for orientation, r, c, height, boundary, region in chosen:
        for rr, cc in boundary:
            grid[rr, cc] = GREEN
            grid[2 * sep - rr, cc] = GREEN
        forbidden |= region

    distractor_count = random.randint(max(1, len(chosen)), len(chosen) + 4)
    if not _place_distractors(grid, sep, forbidden, distractor_count):
        return None
    if not _validate_instance(grid, expected, chosen):
        return None
    if np.array_equal(main(grid), grid):
        return None
    if np.count_nonzero(true_mask) < 5:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid mirrored-wedge ARC input")
