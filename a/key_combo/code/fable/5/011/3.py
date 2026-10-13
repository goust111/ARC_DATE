from __future__ import annotations

from collections import Counter
from itertools import combinations

import numpy as np


# This file is deliberately self-contained so SourceFileLoader can load it from
# any working directory.  Coordinates and inclusive bboxes follow arc_utils.py:
# every coordinate is (row, column), and a bbox is (r0, c0, r1, c1).

BBox = tuple[int, int, int, int]
Coord = tuple[int, int]


def _dominant_color(grid: np.ndarray) -> int:
    """Return the unique most frequent color."""
    counts = Counter(int(value) for value in grid.ravel())
    largest = max(counts.values())
    candidates = [color for color, count in counts.items() if count == largest]
    assert len(candidates) == 1, "expected a unique dominant background color"
    return int(candidates[0])


def _regularly_spaced(values: list[int]) -> bool:
    if len(values) < 3:
        return False
    gaps = np.diff(np.asarray(values, dtype=int))
    return bool(np.all(gaps == gaps[0]) and int(gaps[0]) >= 2)


def _find_lattice(grid: np.ndarray, bg: int) -> tuple[int, list[int], list[int], list[Coord]]:
    """Find the unique complete lattice, allowing holes only at intersections."""
    h, w = grid.shape
    candidates: list[tuple[int, list[int], list[int], list[Coord]]] = []

    for color_value in np.unique(grid):
        color = int(color_value)
        if color == int(bg):
            continue

        # At most four of the eight corner holes can lie on one lattice line.
        rows = [r for r in range(h) if int(np.count_nonzero(grid[r, :] == color)) >= w - 4]
        cols = [c for c in range(w) if int(np.count_nonzero(grid[:, c] == color)) >= h - 4]
        if not (_regularly_spaced(rows) and _regularly_spaced(cols)):
            continue

        line_mask = np.zeros(grid.shape, dtype=bool)
        line_mask[rows, :] = True
        line_mask[:, cols] = True
        intersections = np.zeros(grid.shape, dtype=bool)
        intersections[np.ix_(rows, cols)] = True
        hole_mask = intersections & (grid == int(bg))
        holes = [(int(r), int(c)) for r, c in np.argwhere(hole_mask)]
        if len(holes) != 8:
            continue

        # Every non-hole lattice pixel is line-colored, and a hole can occur
        # only where a horizontal and vertical line cross.
        if not np.all(grid[line_mask & ~hole_mask] == color):
            continue
        if not np.all(grid[hole_mask] == int(bg)):
            continue
        candidates.append((color, rows, cols, holes))

    assert len(candidates) == 1, "expected one regular lattice with eight intersection holes"
    return candidates[0]


def _corner_rectangle(points: set[Coord]) -> BBox | None:
    """Return the bbox when four points are exactly its four corners."""
    if len(points) != 4:
        return None
    rows = sorted({r for r, _c in points})
    cols = sorted({c for _r, c in points})
    if len(rows) != 2 or len(cols) != 2:
        return None
    expected = {(r, c) for r in rows for c in cols}
    if points != expected:
        return None
    return int(rows[0]), int(cols[0]), int(rows[1]), int(cols[1])


def _rectangle_partitions(holes: list[Coord]) -> list[tuple[BBox, BBox]]:
    """Enumerate all partitions into two equal-size corner rectangles."""
    assert len(holes) == 8 and len(set(holes)) == 8
    all_holes = set(holes)
    partitions: set[tuple[BBox, BBox]] = set()

    for chosen_tuple in combinations(sorted(all_holes), 4):
        chosen = set(chosen_tuple)
        first = _corner_rectangle(chosen)
        second = _corner_rectangle(all_holes - chosen)
        if first is None or second is None:
            continue
        first_size = (first[2] - first[0], first[3] - first[1])
        second_size = (second[2] - second[0], second[3] - second[1])
        if first_size != second_size:
            continue
        partitions.add(tuple(sorted((first, second))))

    return sorted(partitions)


def _coords_in_bbox(coords: np.ndarray, box: BBox) -> np.ndarray:
    r0, c0, r1, c1 = box
    if coords.size == 0:
        return np.empty((0, 2), dtype=int)
    keep = (
        (coords[:, 0] >= r0)
        & (coords[:, 0] <= r1)
        & (coords[:, 1] >= c0)
        & (coords[:, 1] <= c1)
    )
    return coords[keep]


def _inside(point: Coord, box: BBox) -> bool:
    r, c = point
    r0, c0, r1, c1 = box
    return r0 <= r <= r1 and c0 <= c <= c1


def _analyze(
    grid: np.ndarray,
) -> tuple[int, int, list[int], list[int], list[Coord], BBox, BBox, np.ndarray]:
    """Recover the lattice, unique rectangle pair, source, target, and payload."""
    bg = _dominant_color(grid)
    line, line_rows, line_cols, holes = _find_lattice(grid, bg)
    partitions = _rectangle_partitions(holes)
    assert len(partitions) == 1, "the eight holes must have one equal-rectangle partition"
    first, second = partitions[0]

    marker_coords = np.argwhere((grid != int(bg)) & (grid != int(line)))
    first_markers = _coords_in_bbox(marker_coords, first)
    second_markers = _coords_in_bbox(marker_coords, second)
    marked = [len(first_markers) > 0, len(second_markers) > 0]
    assert sum(marked) == 1, "exactly one hole rectangle must contain marker pixels"

    if marked[0]:
        source, target, source_markers = first, second, first_markers
    else:
        source, target, source_markers = second, first, second_markers
    return bg, line, line_rows, line_cols, holes, source, target, source_markers


def main(grid: np.ndarray) -> np.ndarray:
    """
    Copy the marked hole-corner rectangle by a 180-degree rotation into its empty mate, then repair all eight holes.

    Input structure:
    - A dominant background and one line color form a regular one-cell-thick lattice.
    - Exactly eight background-colored lattice intersections uniquely split into
      two equal-size rectangles; one contains colored cell-interior markers and
      the other contains none. Other markers may lie outside both rectangles.

    Rule:
    - Identify the unique equal-rectangle partition of the eight holes and use
      marker occupancy to select the source and empty target.
    - Rotate every source-rectangle marker 180 degrees about the midpoint of the
      two rectangle centers, preserving its color, and copy it into the target.
    - Restore all eight missing intersections to the lattice line color and keep
      every other input pixel unchanged.

    Key insight:
    - The holes are paired corner labels: only after their unique partition is
      solved do the source, target, and shared external rotation center become fixed.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0
    bg, line, _rows, _cols, holes, source, target, source_markers = _analyze(g)

    # If C is the midpoint of the two rectangle centers, this value is 2*C.
    endpoint_sum_r = source[0] + source[2] + target[0] + target[2]
    endpoint_sum_c = source[1] + source[3] + target[1] + target[3]
    assert endpoint_sum_r % 2 == 0 and endpoint_sum_c % 2 == 0
    twice_center_r = endpoint_sum_r // 2
    twice_center_c = endpoint_sum_c // 2

    out = g.copy()
    for r, c in holes:
        out[r, c] = int(line)

    for r_value, c_value in source_markers:
        r, c = int(r_value), int(c_value)
        rr = int(twice_center_r - r)
        cc = int(twice_center_c - c)
        assert _inside((rr, cc), target), "rotation must map source into target"
        assert int(g[rr, cc]) == int(bg), "target marker cells must initially be empty"
        out[rr, cc] = int(g[r, c])

    return out


# ------------------------------ generator ------------------------------

_COLOR_COUNT = 10
_LINE_COUNT_OPTIONS = (5, 6)
_SPACING_OPTIONS = (4, 5)
_MARGIN_RANGE = (1, 3)
_MARKER_COUNT_RANGE = (4, 8)
_EXTERNAL_MARKER_COUNT_RANGE = (1, 3)
_GENERATION_ATTEMPTS = 500


def _rectangle_candidates(
    line_rows: list[int], line_cols: list[int], row_span: int, col_span: int
) -> list[tuple[BBox, BBox]]:
    """Build separated, non-axis-aligned pairs whose hole partition is unique."""
    index_boxes = [
        (ri, ci, ri + row_span, ci + col_span)
        for ri in range(len(line_rows) - row_span)
        for ci in range(len(line_cols) - col_span)
    ]
    valid: list[tuple[BBox, BBox]] = []

    for a_index, b_index in combinations(index_boxes, 2):
        # A nonzero shift in both axes avoids the inherently ambiguous pairing
        # produced by two perfectly row- or column-aligned rectangles.
        if a_index[0] == b_index[0] or a_index[1] == b_index[1]:
            continue
        separated = (
            a_index[2] < b_index[0]
            or b_index[2] < a_index[0]
            or a_index[3] < b_index[1]
            or b_index[3] < a_index[1]
        )
        if not separated:
            continue

        a = (
            line_rows[a_index[0]],
            line_cols[a_index[1]],
            line_rows[a_index[2]],
            line_cols[a_index[3]],
        )
        b = (
            line_rows[b_index[0]],
            line_cols[b_index[1]],
            line_rows[b_index[2]],
            line_cols[b_index[3]],
        )
        holes = sorted(
            {(r, c) for r in (a[0], a[2]) for c in (a[1], a[3])}
            | {(r, c) for r in (b[0], b[2]) for c in (b[1], b[3])}
        )
        if len(holes) == 8 and len(_rectangle_partitions(holes)) == 1:
            valid.append(tuple(sorted((a, b))))
    return valid


def _sparse_choice(
    candidates: list[Coord], count: int, rng: np.random.Generator, forbidden: set[Coord] | None = None
) -> list[Coord] | None:
    """Choose visually separated cell-interior pixels."""
    blocked = set() if forbidden is None else set(forbidden)
    order = rng.permutation(len(candidates))
    chosen: list[Coord] = []
    for index in order:
        point = candidates[int(index)]
        if any(abs(point[0] - q[0]) + abs(point[1] - q[1]) < 2 for q in blocked):
            continue
        chosen.append(point)
        blocked.add(point)
        if len(chosen) == count:
            return chosen
    return None


def _paint_expected(
    grid: np.ndarray,
    holes: list[Coord],
    line: int,
    source: BBox,
    target: BBox,
    source_points: list[Coord],
) -> np.ndarray:
    """Independent metadata oracle used by the generator."""
    expected = grid.copy()
    for r, c in holes:
        expected[r, c] = int(line)
    twice_center_r = (source[0] + source[2] + target[0] + target[2]) // 2
    twice_center_c = (source[1] + source[3] + target[1] + target[3]) // 2
    for r, c in source_points:
        expected[twice_center_r - r, twice_center_c - c] = int(grid[r, c])
    return expected


def _semantic_validation(grid: np.ndarray, expected: np.ndarray, meta: dict) -> bool:
    """Check the stated semantics and reject shortcut-compatible samples."""
    bg = int(meta["bg"])
    line = int(meta["line"])
    rows: list[int] = meta["rows"]
    cols: list[int] = meta["cols"]
    holes: list[Coord] = meta["holes"]
    source: BBox = meta["source"]
    target: BBox = meta["target"]
    source_points: list[Coord] = meta["source_points"]
    external_points: list[Coord] = meta["external_points"]

    if grid.ndim != 2 or max(grid.shape) > 30 or _dominant_color(grid) != bg:
        return False
    if len(holes) != 8 or len(_rectangle_partitions(holes)) != 1:
        return False
    if set(_rectangle_partitions(holes)[0]) != {source, target}:
        return False

    line_mask = np.zeros(grid.shape, dtype=bool)
    line_mask[rows, :] = True
    line_mask[:, cols] = True
    hole_mask = np.zeros(grid.shape, dtype=bool)
    for r, c in holes:
        hole_mask[r, c] = True
    if not np.all(grid[line_mask & ~hole_mask] == line):
        return False
    if not np.all(grid[hole_mask] == bg):
        return False
    if any(line_mask[r, c] for r, c in source_points + external_points):
        return False

    all_markers = [(int(r), int(c)) for r, c in np.argwhere((grid != bg) & (grid != line))]
    in_source = {point for point in all_markers if _inside(point, source)}
    in_target = {point for point in all_markers if _inside(point, target)}
    if in_source != set(source_points) or in_target or not external_points:
        return False
    if len(source_points) < 4 or len({int(grid[p]) for p in source_points}) < 2:
        return False

    twice_center_r = (source[0] + source[2] + target[0] + target[2]) // 2
    twice_center_c = (source[1] + source[3] + target[1] + target[3]) // 2
    rotated = {(twice_center_r - r, twice_center_c - c) for r, c in source_points}
    if len(rotated) != len(source_points) or not all(_inside(p, target) for p in rotated):
        return False

    # Both advertised operations must alter the result.
    repaired_only = grid.copy()
    for r, c in holes:
        repaired_only[r, c] = line
    if np.array_equal(repaired_only, expected):
        return False
    copied_only = expected.copy()
    for r, c in holes:
        copied_only[r, c] = bg
    if np.array_equal(copied_only, expected):
        return False

    # A same-orientation translation into the target must give a different grid.
    translated = repaired_only.copy()
    for r, c in source_points:
        rr = target[0] + (r - source[0])
        cc = target[1] + (c - source[1])
        translated[rr, cc] = int(grid[r, c])
    if np.array_equal(translated, expected):
        return False

    # The specified shared center must not accidentally equal the canvas center.
    global_center_copy = repaired_only.copy()
    h, w = grid.shape
    for r, c in source_points:
        rr, cc = h - 1 - r, w - 1 - c
        if 0 <= rr < h and 0 <= cc < w and int(global_center_copy[rr, cc]) == bg:
            global_center_copy[rr, cc] = int(grid[r, c])
    if np.array_equal(global_center_copy, expected):
        return False

    # Copying every marker around the correct center is also wrong: the payload
    # scope is the source rectangle, not the full canvas.
    copy_all = repaired_only.copy()
    for r, c in all_markers:
        rr, cc = twice_center_r - r, twice_center_c - c
        if 0 <= rr < h and 0 <= cc < w and int(copy_all[rr, cc]) == bg:
            copy_all[rr, cc] = int(grid[r, c])
    if np.array_equal(copy_all, expected):
        return False

    # Direct checks of preservation, repair, color transfer, and solver output.
    for r, c in holes:
        if int(expected[r, c]) != line:
            return False
    for r, c in source_points + external_points:
        if int(expected[r, c]) != int(grid[r, c]):
            return False
    for r, c in source_points:
        if int(expected[twice_center_r - r, twice_center_c - c]) != int(grid[r, c]):
            return False
    try:
        actual = main(grid)
    except Exception:
        return False
    return bool(np.array_equal(actual, expected) and not np.array_equal(grid, expected))


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()
    colors = [int(value) for value in rng.choice(_COLOR_COUNT, size=5, replace=False)]
    bg, line = colors[:2]
    marker_palette = colors[2:]

    n_rows = int(rng.choice(_LINE_COUNT_OPTIONS))
    n_cols = int(rng.choice(_LINE_COUNT_OPTIONS))
    row_spacing = int(rng.choice(_SPACING_OPTIONS))
    col_spacing = int(rng.choice(_SPACING_OPTIONS))
    top = int(rng.integers(_MARGIN_RANGE[0], _MARGIN_RANGE[1] + 1))
    left = int(rng.integers(_MARGIN_RANGE[0], _MARGIN_RANGE[1] + 1))
    bottom = int(rng.integers(_MARGIN_RANGE[0], _MARGIN_RANGE[1] + 1))
    right = int(rng.integers(_MARGIN_RANGE[0], _MARGIN_RANGE[1] + 1))
    h = top + (n_rows - 1) * row_spacing + 1 + bottom
    w = left + (n_cols - 1) * col_spacing + 1 + right
    if h > 30 or w > 30:
        return None

    rows = [top + index * row_spacing for index in range(n_rows)]
    cols = [left + index * col_spacing for index in range(n_cols)]
    grid = np.full((h, w), bg, dtype=int)
    grid[rows, :] = line
    grid[:, cols] = line

    row_span = int(rng.integers(1, min(2, n_rows - 2) + 1))
    col_span = int(rng.integers(1, min(2, n_cols - 2) + 1))
    pair_options = _rectangle_candidates(rows, cols, row_span, col_span)
    if not pair_options:
        return None
    first, second = pair_options[int(rng.integers(len(pair_options)))]
    if bool(rng.integers(2)):
        source, target = first, second
    else:
        source, target = second, first

    holes = sorted(
        {(r, c) for r in (first[0], first[2]) for c in (first[1], first[3])}
        | {(r, c) for r in (second[0], second[2]) for c in (second[1], second[3])}
    )
    for r, c in holes:
        grid[r, c] = bg

    source_candidates = [
        (r, c)
        for r in range(source[0] + 1, source[2])
        for c in range(source[1] + 1, source[3])
        if r not in rows and c not in cols
    ]
    if len(source_candidates) < _MARKER_COUNT_RANGE[0]:
        return None
    marker_count = int(rng.integers(_MARKER_COUNT_RANGE[0], min(_MARKER_COUNT_RANGE[1], len(source_candidates)) + 1))
    source_points = _sparse_choice(source_candidates, marker_count, rng)
    if source_points is None:
        return None

    # Ensure that rotating the payload is visibly different from translating it.
    local = {(r - source[0], c - source[1]) for r, c in source_points}
    flipped = {(source[2] - r, source[3] - c) for r, c in source_points}
    if local == flipped:
        return None

    marker_color_count = int(rng.integers(2, min(3, len(source_points)) + 1))
    used_marker_colors = marker_palette[:marker_color_count]
    shuffled_points = list(source_points)
    rng.shuffle(shuffled_points)
    for index, point in enumerate(shuffled_points):
        color = used_marker_colors[index] if index < marker_color_count else int(rng.choice(used_marker_colors))
        grid[point] = int(color)

    twice_center_r = (source[0] + source[2] + target[0] + target[2]) // 2
    twice_center_c = (source[1] + source[3] + target[1] + target[3]) // 2
    rotated_source = {(twice_center_r - r, twice_center_c - c) for r, c in source_points}
    forbidden = set(source_points) | rotated_source | set(holes)
    external_candidates: list[Coord] = []
    for r in range(h):
        for c in range(w):
            point = (r, c)
            reflected = (twice_center_r - r, twice_center_c - c)
            if r in rows or c in cols or point in forbidden:
                continue
            if _inside(point, first) or _inside(point, second):
                continue
            if not (0 <= reflected[0] < h and 0 <= reflected[1] < w):
                continue
            if reflected[0] in rows or reflected[1] in cols or reflected in forbidden:
                continue
            if _inside(reflected, target):
                continue
            external_candidates.append(point)

    external_count = int(rng.integers(_EXTERNAL_MARKER_COUNT_RANGE[0], _EXTERNAL_MARKER_COUNT_RANGE[1] + 1))
    external_points = _sparse_choice(external_candidates, external_count, rng, forbidden=set(source_points))
    if external_points is None:
        return None
    for point in external_points:
        grid[point] = int(rng.choice(used_marker_colors))

    # Keep background inference structural rather than dependent on a tie.
    counts = Counter(int(value) for value in grid.ravel())
    if counts[bg] <= max(count for color, count in counts.items() if color != bg):
        return None

    expected = _paint_expected(grid, holes, line, source, target, source_points)
    meta = {
        "bg": bg,
        "line": line,
        "rows": rows,
        "cols": cols,
        "holes": holes,
        "source": source,
        "target": target,
        "source_points": source_points,
        "external_points": external_points,
    }
    if _semantic_validation(grid, expected, meta):
        return grid
    return None


def generate_input() -> np.ndarray:
    """Generate a varied, non-degenerate input satisfying the puzzle statement."""
    for _ in range(_GENERATION_ATTEMPTS):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("failed to generate a valid hole-pair rotation puzzle")
