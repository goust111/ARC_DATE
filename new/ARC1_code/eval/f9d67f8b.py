from __future__ import annotations

import numpy as np


def _best_reflection_axis(
    grid: np.ndarray, hole_color: int, *, along_rows: bool
) -> tuple[int, int] | None:
    limit = grid.shape[0] if along_rows else grid.shape[1]
    candidates: list[tuple[int, int]] = []

    for axis_sum in range(2 * limit - 1):
        evidence = 0
        conflicts = 0
        for i in range(limit):
            j = axis_sum - i
            if not (0 <= j < limit and i < j):
                continue

            a = grid[i, :] if along_rows else grid[:, i]
            b = grid[j, :] if along_rows else grid[:, j]
            known = (a != hole_color) & (b != hole_color)
            evidence += int(np.count_nonzero(known))
            conflicts += int(np.count_nonzero(known & (a != b)))

        if conflicts == 0 and evidence > 0:
            candidates.append((evidence, axis_sum))

    if not candidates:
        return None

    best_evidence = max(evidence for evidence, _axis_sum in candidates)
    best = [axis_sum for evidence, axis_sum in candidates if evidence == best_evidence]
    assert len(best) == 1, "expected a unique strongest reflection axis"
    return best_evidence, best[0]


def _find_hole_color_and_axes(grid: np.ndarray) -> tuple[int, int, int]:
    scored: list[tuple[int, int, int, int]] = []
    for color in map(int, np.unique(grid)):
        row_axis = _best_reflection_axis(grid, color, along_rows=True)
        col_axis = _best_reflection_axis(grid, color, along_rows=False)
        if row_axis is None or col_axis is None:
            continue
        score = min(row_axis[0], col_axis[0])
        scored.append((score, color, row_axis[1], col_axis[1]))

    assert scored, "expected one color to act as symmetry holes"
    scored.sort(reverse=True)
    assert len(scored) == 1 or scored[0][0] > scored[1][0], "hole color must be structurally unique"

    _score, hole_color, row_axis, col_axis = scored[0]
    assert np.any(grid == hole_color), "expected at least one hole pixel"
    return hole_color, row_axis, col_axis


def _in_bounds(shape: tuple[int, int], row: int, col: int) -> bool:
    height, width = shape
    return 0 <= row < height and 0 <= col < width


def _visible_values(grid: np.ndarray, coords: set[tuple[int, int]], hole_color: int) -> set[int]:
    return {int(grid[row, col]) for row, col in coords if int(grid[row, col]) != hole_color}


def _fill_from_reflections(
    grid: np.ndarray, hole_color: int, row_axis: int, col_axis: int
) -> bool:
    changed = False
    shape = grid.shape
    for row, col in np.argwhere(grid == hole_color):
        row = int(row)
        col = int(col)
        orbit = {(row, col)}
        for rr, cc in (
            (row_axis - row, col),
            (row, col_axis - col),
            (row_axis - row, col_axis - col),
        ):
            if _in_bounds(shape, rr, cc):
                orbit.add((rr, cc))

        values = _visible_values(grid, orbit, hole_color)
        if not values:
            continue
        assert len(values) == 1, "reflection orbit must have one visible color"
        grid[row, col] = values.pop()
        changed = True
    return changed


def _fill_from_quarter_turns(grid: np.ndarray, hole_color: int, center_axis: int) -> bool:
    changed = False
    shape = grid.shape
    for row, col in np.argwhere(grid == hole_color):
        row = int(row)
        col = int(col)
        orbit: set[tuple[int, int]] = set()
        for rr, cc in ((center_axis - col, row), (col, center_axis - row)):
            if _in_bounds(shape, rr, cc):
                orbit.add((rr, cc))

        values = _visible_values(grid, orbit, hole_color)
        if not values:
            continue
        assert len(values) == 1, "quarter-turn orbit must have one visible color"
        grid[row, col] = values.pop()
        changed = True
    return changed


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repair the maroon-stained pattern by replacing each damaged pixel with the color forced by the pattern symmetries.

    Input structure:
    - A square grid contains one occluding color whose pixels are not part of the final design.
    - Ignoring the occluding color reveals unique shifted horizontal and vertical reflection axes with the same center.
    - Any occluding pixels left unresolved by reflection lie in edge bands that also obey the same-center quarter-turn relation.

    Rule:
    - Infer the occluding color as the only color whose removal gives conflict-free high-evidence row and column reflection axes.
    - Fill occluding pixels from the unique visible color in their horizontal, vertical, and 180-degree reflection orbit.
    - Fill any remaining occluding pixels from the unique visible color at a 90-degree quarter-turn counterpart.

    Key insight:
    - The maroon pixels are damage over a shifted kaleidoscope-like pattern; the hidden color is determined by a visible mate in the same symmetry orbit.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    height, width = map(int, g.shape)
    assert height == width, "expected a square grid"

    hole_color, row_axis, col_axis = _find_hole_color_and_axes(g)
    assert row_axis == col_axis, "row and column axes must share a center"

    out = g.copy()
    while _fill_from_reflections(out, hole_color, row_axis, col_axis):
        pass

    while _fill_from_quarter_turns(out, hole_color, row_axis):
        pass

    assert not np.any(out == hole_color), "all damaged pixels must be repairable"
    return out

# --------------- generator (o5x) ---------------

# Canvas and symmetry parameters.
_GEN_GRID_SIZE = 30
_GEN_AXIS_SUM = 31
_GEN_FREE_SIZE = 16
_GEN_EDGE_BAND = 2
_GEN_PATTERN_COLORS = np.arange(1, 9, dtype=int)
_GEN_HOLE_COLOR = 9

# Correlated base-pattern synthesis.
_GEN_COARSE_SIZE = 8
_GEN_COARSE_SCALE = 2
_GEN_NOISE_RATE_RANGE = (0.12, 0.28)
_GEN_SMOOTH_PASS_RANGE = (1, 3)
_GEN_SMOOTH_COPY_PROB = 0.28

# Hole component geometry and total damaged area.
_GEN_BLOB_COUNT_CHOICES = [2, 3]
_GEN_BLOB_HEIGHT_RANGE = (2, 9)
_GEN_BLOB_WIDTH_RANGE = (2, 12)
_GEN_BLOB_AREA_RANGE = (4, 56)
_GEN_TOTAL_HOLE_AREA_RANGE = (30, 110)

# Rejection sampling limit.
_GEN_MAX_ATTEMPTS = 300

_GEN_SHAPES_BY_AREA: dict[int, list[np.ndarray]] | None = None
_GEN_AREA_TUPLES: dict[int, dict[int, list[tuple[int, ...]]]] = {}


def _gen_correlated_base() -> np.ndarray:
    """Return the free 16x16 domain before shifted reflections are applied."""
    coarse = np.random.choice(_GEN_PATTERN_COLORS, size=(_GEN_COARSE_SIZE, _GEN_COARSE_SIZE))
    base = np.repeat(np.repeat(coarse, _GEN_COARSE_SCALE, axis=0), _GEN_COARSE_SCALE, axis=1)

    noise_field = np.random.random((_GEN_FREE_SIZE, _GEN_FREE_SIZE))
    noise = noise_field < np.random.uniform(*_GEN_NOISE_RATE_RANGE)
    base[noise] = np.random.choice(_GEN_PATTERN_COLORS, size=int(np.count_nonzero(noise)))

    for _ in range(np.random.randint(*_GEN_SMOOTH_PASS_RANGE)):
        for row in np.random.permutation(_GEN_FREE_SIZE):
            for col in np.random.permutation(_GEN_FREE_SIZE):
                if np.random.random() >= _GEN_SMOOTH_COPY_PROB:
                    continue
                nbrs: list[int] = []
                if row:
                    nbrs.append(int(base[row - 1, col]))
                if col:
                    nbrs.append(int(base[row, col - 1]))
                if row < _GEN_FREE_SIZE - 1:
                    nbrs.append(int(base[row + 1, col]))
                if col < _GEN_FREE_SIZE - 1:
                    nbrs.append(int(base[row, col + 1]))
                if nbrs:
                    base[row, col] = np.random.choice(nbrs)

    for k in range(_GEN_EDGE_BAND, _GEN_FREE_SIZE):
        for edge in range(_GEN_EDGE_BAND):
            base[edge, k] = base[k, edge]

    for color in _GEN_PATTERN_COLORS:
        if color not in base:
            row = np.random.randint(_GEN_EDGE_BAND, _GEN_FREE_SIZE)
            col = np.random.randint(_GEN_EDGE_BAND, _GEN_FREE_SIZE)
            base[row, col] = color

    return base


def _gen_reflect_base(base: np.ndarray) -> np.ndarray:
    """Expand the free domain into the full shifted-reflection output grid."""
    out = np.zeros((_GEN_GRID_SIZE, _GEN_GRID_SIZE), dtype=int)
    out[:_GEN_FREE_SIZE, :_GEN_FREE_SIZE] = base
    reflected = np.arange(_GEN_EDGE_BAND, _GEN_FREE_SIZE)
    out[_GEN_AXIS_SUM - reflected, :_GEN_FREE_SIZE] = out[reflected, :_GEN_FREE_SIZE]
    out[:, _GEN_AXIS_SUM - reflected] = out[:, reflected]
    return out


def _gen_output_pattern() -> np.ndarray:
    return _gen_reflect_base(_gen_correlated_base())


def _gen_shape_candidates() -> dict[int, list[np.ndarray]]:
    global _GEN_SHAPES_BY_AREA
    if _GEN_SHAPES_BY_AREA is not None:
        return _GEN_SHAPES_BY_AREA

    by_area: dict[int, list[np.ndarray]] = {}
    seen: set[tuple[tuple[int, int], bytes]] = set()

    def add(mask: np.ndarray) -> None:
        area = int(np.count_nonzero(mask))
        if not (_GEN_BLOB_AREA_RANGE[0] <= area <= _GEN_BLOB_AREA_RANGE[1]):
            return
        key = (tuple(map(int, mask.shape)), mask.tobytes())
        if key in seen:
            return
        seen.add(key)
        by_area.setdefault(area, []).append(mask.copy())

    for height in range(*_GEN_BLOB_HEIGHT_RANGE):
        for width in range(*_GEN_BLOB_WIDTH_RANGE):
            add(np.ones((height, width), dtype=bool))

            for leg_width in range(1, width):
                for leg_height in range(1, height):
                    mask = np.zeros((height, width), dtype=bool)
                    mask[:, :leg_width] = True
                    mask[:leg_height, :] = True
                    for flip_r in (False, True):
                        for flip_c in (False, True):
                            oriented = mask
                            if flip_r:
                                oriented = np.flipud(oriented)
                            if flip_c:
                                oriented = np.fliplr(oriented)
                            add(oriented)

    _GEN_SHAPES_BY_AREA = by_area
    return by_area


def _gen_area_tuple_map(count: int) -> dict[int, list[tuple[int, ...]]]:
    if count in _GEN_AREA_TUPLES:
        return _GEN_AREA_TUPLES[count]

    areas = sorted(_gen_shape_candidates())
    by_total: dict[int, list[tuple[int, ...]]] = {}
    if count == 2:
        for a in areas:
            for b in areas:
                total = a + b
                if _GEN_TOTAL_HOLE_AREA_RANGE[0] <= total <= _GEN_TOTAL_HOLE_AREA_RANGE[1]:
                    by_total.setdefault(total, []).append((a, b))
    else:
        for a in areas:
            for b in areas:
                for c in areas:
                    total = a + b + c
                    if _GEN_TOTAL_HOLE_AREA_RANGE[0] <= total <= _GEN_TOTAL_HOLE_AREA_RANGE[1]:
                        by_total.setdefault(total, []).append((a, b, c))

    _GEN_AREA_TUPLES[count] = by_total
    return by_total


def _gen_choose_shapes(count: int) -> list[np.ndarray]:
    by_area = _gen_shape_candidates()
    by_total = _gen_area_tuple_map(count)
    total = int(np.random.choice(sorted(by_total)))
    area_tuple = by_total[total][np.random.randint(len(by_total[total]))]
    return [by_area[area][np.random.randint(len(by_area[area]))] for area in area_tuple]


def _gen_in_bounds(row: int, col: int) -> bool:
    return 0 <= row < _GEN_GRID_SIZE and 0 <= col < _GEN_GRID_SIZE


def _gen_directly_repairable(mask: np.ndarray, row: int, col: int) -> bool:
    reflection_mates = (
        (_GEN_AXIS_SUM - row, col),
        (row, _GEN_AXIS_SUM - col),
        (_GEN_AXIS_SUM - row, _GEN_AXIS_SUM - col),
    )
    if any(_gen_in_bounds(r, c) and not mask[r, c] for r, c in reflection_mates):
        return True

    if row >= _GEN_EDGE_BAND and col >= _GEN_EDGE_BAND:
        return False

    quarter_mates = ((_GEN_AXIS_SUM - col, row), (col, _GEN_AXIS_SUM - row))
    return any(_gen_in_bounds(r, c) and not mask[r, c] for r, c in quarter_mates)


def _gen_repairable(mask: np.ndarray) -> bool:
    for row, col in np.argwhere(mask):
        if not _gen_directly_repairable(mask, int(row), int(col)):
            return False
    return True


def _gen_touch_forbidden_mask(mask: np.ndarray) -> np.ndarray:
    """Return occupied cells plus their 4-neighbor no-touch padding."""
    forbidden = mask.copy()
    forbidden[1:, :] |= mask[:-1, :]
    forbidden[:-1, :] |= mask[1:, :]
    forbidden[:, 1:] |= mask[:, :-1]
    forbidden[:, :-1] |= mask[:, 1:]
    return forbidden


def _gen_valid_placements(mask: np.ndarray, shape: np.ndarray) -> list[tuple[int, int]]:
    local = np.argwhere(shape)
    height, width = map(int, shape.shape)
    forbidden = _gen_touch_forbidden_mask(mask)
    placements: list[tuple[int, int]] = []
    for row0 in range(_GEN_GRID_SIZE - height + 1):
        for col0 in range(_GEN_GRID_SIZE - width + 1):
            coords = local + np.array([row0, col0])
            if np.any(forbidden[coords[:, 0], coords[:, 1]]):
                continue
            trial = mask.copy()
            trial[coords[:, 0], coords[:, 1]] = True
            if _gen_repairable(trial):
                placements.append((row0, col0))
    return placements


def _gen_place_holes(shapes: list[np.ndarray]) -> np.ndarray | None:
    mask = np.zeros((_GEN_GRID_SIZE, _GEN_GRID_SIZE), dtype=bool)
    for shape in shapes:
        placements = _gen_valid_placements(mask, shape)
        if not placements:
            return None
        row0, col0 = placements[np.random.randint(len(placements))]
        coords = np.argwhere(shape) + np.array([row0, col0])
        mask[coords[:, 0], coords[:, 1]] = True
    return mask


def _gen_attempt() -> np.ndarray | None:
    out = _gen_output_pattern()
    count = int(np.random.choice(_GEN_BLOB_COUNT_CHOICES))
    shapes = _gen_choose_shapes(count)
    np.random.shuffle(shapes)
    hole_mask = _gen_place_holes(shapes)
    if hole_mask is None:
        return None

    inp = out.copy()
    inp[hole_mask] = _GEN_HOLE_COLOR
    try:
        solved = main(inp)
    except Exception:
        return None
    if not np.array_equal(solved, out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(_GEN_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid f9d67f8b input")
