from __future__ import annotations

import random
from importlib.machinery import SourceFileLoader

import numpy as np

try:
    from arc_utils import bbox, get_bg, in_bounds
except ModuleNotFoundError:
    _ARC_UTILS = SourceFileLoader(
        "_arc_utils_shared",
        "/home/lixinyu/snap/new/solvers/arc_utils.py",
    ).load_module()
    bbox = _ARC_UTILS.bbox
    get_bg = _ARC_UTILS.get_bg
    in_bounds = _ARC_UTILS.in_bounds


BG = 0
SCAFFOLD = 5
CONTENT_COLORS = (1, 2, 3, 4, 7)
CELL_COUNTS = (3, 4)
DIAGONALS = ((-1, -1), (-1, 1), (1, -1), (1, 1))
FIXED_DIRECTION_COLORS = {
    (-1, -1): 1,
    (-1, 1): 2,
    (1, -1): 3,
    (1, 1): 4,
}

ACTIVE_COUNT_RANGE = (4, 8)
PADDING_RANGE = (0, 2)
GEN_ATTEMPTS = 800


def _bbox_or_raise(mask: np.ndarray) -> tuple[int, int, int, int]:
    b = bbox(mask)
    assert b is not None, "expected a scaffold bbox"
    return tuple(map(int, b))


def _regular_bounds(grid: np.ndarray, scaffold_color: int, axis: int) -> list[int]:
    grey_bbox = _bbox_or_raise(grid == scaffold_color)
    lo = grey_bbox[0] - 1 if axis == 0 else grey_bbox[1] - 1
    hi = grey_bbox[2] + 1 if axis == 0 else grey_bbox[3] + 1
    assert 0 <= lo < (grid.shape[0] if axis == 0 else grid.shape[1])
    assert 0 <= hi < (grid.shape[0] if axis == 0 else grid.shape[1])

    bounds = []
    for idx in range(lo, hi + 1):
        has_scaffold = np.any(grid[idx, :] == scaffold_color) if axis == 0 else np.any(grid[:, idx] == scaffold_color)
        if not has_scaffold:
            bounds.append(int(idx))

    assert len(bounds) >= 4, "expected a repeated scaffold lattice"
    gaps = [b - a for a, b in zip(bounds, bounds[1:])]
    assert len(set(gaps)) == 1 and gaps[0] == 4, "expected evenly spaced 3x3 scaffold cells"
    return bounds


def _parse_grid(grid: np.ndarray) -> tuple[tuple[int, int], np.ndarray]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert max(g.shape) <= 30, "ARC grids must be at most 30x30"
    assert int(get_bg(g)) == BG, "background must be black"

    allowed = set(CONTENT_COLORS) | {BG, SCAFFOLD}
    assert set(map(int, np.unique(g))).issubset(allowed), "unexpected colors present"

    row_bounds = _regular_bounds(g, SCAFFOLD, axis=0)
    col_bounds = _regular_bounds(g, SCAFFOLD, axis=1)
    assert len(row_bounds) == len(col_bounds), "expected a square cell lattice"
    cell_count = len(row_bounds) - 1
    assert cell_count in CELL_COUNTS, "expected a 3x3 or 4x4 lattice"

    r0, c0 = row_bounds[0], col_bounds[0]
    r1, c1 = row_bounds[-1], col_bounds[-1]
    outside = g.copy()
    outside[r0 : r1 + 1, c0 : c1 + 1] = BG
    assert np.all(outside == BG), "only the scaffold region may contain non-background pixels"

    contents = np.zeros((cell_count, cell_count), dtype=int)
    for i in range(cell_count):
        for j in range(cell_count):
            top = row_bounds[i] + 1
            left = col_bounds[j] + 1
            patch = g[top : top + 3, left : left + 3]
            assert patch.shape == (3, 3)
            border_mask = np.ones((3, 3), dtype=bool)
            border_mask[1, 1] = False
            assert np.all(patch[border_mask] == SCAFFOLD), "each cell must be a hollow grey 3x3 frame"
            center = int(patch[1, 1])
            assert center in set(CONTENT_COLORS) | {BG}, "cell centers must be black or a content color"
            contents[i, j] = center

    for r in row_bounds:
        assert np.all(g[r, c0 : c1 + 1] == BG), "boundary rows between scaffold cells must stay black"
    for c in col_bounds:
        assert np.all(g[r0 : r1 + 1, c] == BG), "boundary cols between scaffold cells must stay black"

    return (r0, c0), contents


def _corner_local(cell_r: int, cell_c: int, dr: int, dc: int) -> tuple[int, int]:
    top = 4 * cell_r + 1
    left = 4 * cell_c + 1
    return top + (0 if dr == -1 else 2), left + (0 if dc == -1 else 2)


def _marker_map_from_contents(
    contents: np.ndarray,
) -> tuple[dict[tuple[int, int], int], list[tuple[int, int, int, int, int, int]]]:
    planned: dict[tuple[int, int], int] = {}
    events: list[tuple[int, int, int, int, int, int]] = []
    rows, cols = map(int, contents.shape)

    for r, c in np.argwhere(contents != BG):
        r = int(r)
        c = int(c)
        self_color = int(contents[r, c])
        for dr, dc in DIAGONALS:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and int(contents[nr, nc]) != BG:
                color = int(contents[nr, nc])
                planned[_corner_local(r, c, dr, dc)] = color
                events.append((r, c, dr, dc, self_color, color))

    return planned, events


def _paint_markers(
    base: np.ndarray,
    origin: tuple[int, int],
    markers: dict[tuple[int, int], int],
) -> np.ndarray:
    out = np.asarray(base, dtype=int).copy()
    r0, c0 = map(int, origin)
    for (rr, cc), color in markers.items():
        ar = r0 + rr
        ac = c0 + cc
        assert in_bounds(out.shape, ar, ac)
        assert int(out[ar, ac]) in (SCAFFOLD, color), "markers only overwrite the cell-frame corner pixels"
        out[ar, ac] = int(color)
    return out


def _output_from_grid(grid: np.ndarray) -> np.ndarray:
    origin, contents = _parse_grid(grid)
    markers, _events = _marker_map_from_contents(contents)
    return _paint_markers(grid, origin, markers)


def main(grid: np.ndarray) -> np.ndarray:
    """Decorate each active scaffold cell corner with the content color of its diagonal active neighbor.

    Input structure:
    - The grid has a black background and one regular grey scaffold lattice made
      from repeated hollow 3x3 frames separated by one-cell black boundary lines.
    - Each scaffold cell has a one-pixel center that is either black or one
      active content color.
    - The scaffold forms a 3x3 or 4x4 square array of cells.

    Rule:
    - Preserve the scaffold and every center content pixel.
    - For each active cell, inspect its four diagonal neighboring scaffold cells.
    - If a diagonal neighbor exists and is also active, recolor the current
      cell's corresponding grey corner pixel with that diagonal neighbor's
      content color; otherwise leave that corner grey.

    Key insight:
    - The grey scaffold defines both the cell topology and the exact corner
      pixels to recolor, while the corner colors are determined dynamically by
      diagonal neighbor contents rather than by a fixed direction-to-color map.
    """
    return _output_from_grid(np.asarray(grid, dtype=int))


def _features(contents: np.ndarray) -> dict[str, object]:
    active = contents != BG
    rows, cols = map(int, contents.shape)
    markers, events = _marker_map_from_contents(contents)
    diag_pairs: set[tuple[tuple[int, int], tuple[int, int]]] = set()
    orth_pairs = 0
    degrees: dict[tuple[int, int], int] = {}
    direction_colors = {direction: set() for direction in DIAGONALS}
    edge_active = 0
    has_empty_in_bounds_diagonal = False
    has_neighbor_with_different_color = False

    for r, c in np.argwhere(active):
        r = int(r)
        c = int(c)
        if r in (0, rows - 1) or c in (0, cols - 1):
            edge_active += 1
        deg = 0
        for dr, dc in DIAGONALS:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                if int(contents[nr, nc]) != BG:
                    deg += 1
                    diag_pairs.add(tuple(sorted(((r, c), (nr, nc)))))
                    direction_colors[(dr, dc)].add(int(contents[nr, nc]))
                    if int(contents[nr, nc]) != int(contents[r, c]):
                        has_neighbor_with_different_color = True
                else:
                    has_empty_in_bounds_diagonal = True
        degrees[(r, c)] = deg
        if r + 1 < rows and bool(active[r + 1, c]):
            orth_pairs += 1
        if c + 1 < cols and bool(active[r, c + 1]):
            orth_pairs += 1

    return {
        "active_count": int(np.count_nonzero(active)),
        "distinct_active_colors": len(set(map(int, contents[active]))),
        "marker_count": len(markers),
        "diag_pair_count": len(diag_pairs),
        "orth_pairs": orth_pairs,
        "degrees": degrees,
        "edge_active": edge_active,
        "has_empty_in_bounds_diagonal": has_empty_in_bounds_diagonal,
        "has_neighbor_with_different_color": has_neighbor_with_different_color,
        "direction_color_variation": any(len(colors) >= 2 for colors in direction_colors.values()),
        "event_count": len(events),
    }


def _marker_map_fixed_direction(contents: np.ndarray) -> dict[tuple[int, int], int]:
    planned: dict[tuple[int, int], int] = {}
    rows, cols = map(int, contents.shape)
    for r, c in np.argwhere(contents != BG):
        r = int(r)
        c = int(c)
        for dr, dc in DIAGONALS:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and int(contents[nr, nc]) != BG:
                planned[_corner_local(r, c, dr, dc)] = FIXED_DIRECTION_COLORS[(dr, dc)]
    return planned


def _marker_map_self_color(contents: np.ndarray) -> dict[tuple[int, int], int]:
    planned: dict[tuple[int, int], int] = {}
    rows, cols = map(int, contents.shape)
    for r, c in np.argwhere(contents != BG):
        r = int(r)
        c = int(c)
        self_color = int(contents[r, c])
        for dr, dc in DIAGONALS:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and int(contents[nr, nc]) != BG:
                planned[_corner_local(r, c, dr, dc)] = self_color
    return planned


def _marker_map_all_in_bounds_self(contents: np.ndarray) -> dict[tuple[int, int], int]:
    planned: dict[tuple[int, int], int] = {}
    rows, cols = map(int, contents.shape)
    for r, c in np.argwhere(contents != BG):
        r = int(r)
        c = int(c)
        self_color = int(contents[r, c])
        for dr, dc in DIAGONALS:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                planned[_corner_local(r, c, dr, dc)] = self_color
    return planned


def _render_input(contents: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
    cell_count = int(contents.shape[0])
    top = random.randint(*PADDING_RANGE)
    bottom = random.randint(*PADDING_RANGE)
    left = random.randint(*PADDING_RANGE)
    right = random.randint(*PADDING_RANGE)

    span = 4 * cell_count + 1
    grid = np.zeros((top + span + bottom, left + span + right), dtype=int)
    row_bounds = [top + 4 * i for i in range(cell_count + 1)]
    col_bounds = [left + 4 * i for i in range(cell_count + 1)]

    for i in range(cell_count):
        for j in range(cell_count):
            r = row_bounds[i] + 1
            c = col_bounds[j] + 1
            grid[r : r + 3, c : c + 3] = SCAFFOLD
            grid[r + 1, c + 1] = int(contents[i, j])

    return grid, (row_bounds[0], col_bounds[0])


def _base_layout_ok(mode: int, feats: dict[str, object]) -> bool:
    degrees = list(feats["degrees"].values())
    if not (ACTIVE_COUNT_RANGE[0] <= int(feats["active_count"]) <= ACTIVE_COUNT_RANGE[1]):
        return False
    if int(feats["diag_pair_count"]) < 1:
        return False
    if int(feats["marker_count"]) < 2:
        return False
    if int(feats["distinct_active_colors"]) < 2:
        return False
    if not bool(feats["has_neighbor_with_different_color"]):
        return False
    if mode == 0:
        return 0 in degrees and any(deg > 0 for deg in degrees) and bool(feats["has_empty_in_bounds_diagonal"])
    if mode == 1:
        return max(degrees) >= 2 and int(feats["edge_active"]) >= 1
    if mode == 2:
        return int(feats["orth_pairs"]) >= 1 and bool(feats["has_empty_in_bounds_diagonal"])
    return bool(feats["direction_color_variation"]) and int(feats["edge_active"]) >= 1


def _semantics_ok(contents: np.ndarray) -> bool:
    correct, _events = _marker_map_from_contents(contents)
    if not correct:
        return False
    if correct == _marker_map_fixed_direction(contents):
        return False
    if correct == _marker_map_self_color(contents):
        return False
    if correct == _marker_map_all_in_bounds_self(contents):
        return False
    return True


def _sample_contents(mode: int) -> np.ndarray | None:
    cell_count = 4 if mode in (1, 3) and random.random() < 0.7 else random.choice(CELL_COUNTS)
    all_cells = [(r, c) for r in range(cell_count) for c in range(cell_count)]
    active_count = random.randint(ACTIVE_COUNT_RANGE[0], min(ACTIVE_COUNT_RANGE[1], cell_count * cell_count))
    active_cells = random.sample(all_cells, active_count)

    contents = np.zeros((cell_count, cell_count), dtype=int)
    palette_size = random.randint(2, min(len(CONTENT_COLORS), active_count))
    palette = random.sample(list(CONTENT_COLORS), palette_size)
    colors = palette[:] + [random.choice(palette) for _ in range(active_count - palette_size)]
    random.shuffle(colors)
    for (r, c), color in zip(active_cells, colors):
        contents[r, c] = int(color)

    feats = _features(contents)
    if not _base_layout_ok(mode, feats):
        return None
    if not _semantics_ok(contents):
        return None
    return contents


def _validate_generated(grid: np.ndarray, contents: np.ndarray, origin: tuple[int, int]) -> bool:
    try:
        parsed_origin, parsed_contents = _parse_grid(grid)
        if parsed_origin != origin or not np.array_equal(parsed_contents, contents):
            return False

        expected = _output_from_grid(grid)
        actual = main(grid)
        if not np.array_equal(actual, expected):
            return False
        if np.array_equal(actual, grid):
            return False

        wrong_marker_sets = [
            _marker_map_fixed_direction(contents),
            _marker_map_self_color(contents),
            _marker_map_all_in_bounds_self(contents),
        ]
        for wrong_markers in wrong_marker_sets:
            wrong = _paint_markers(grid, origin, wrong_markers)
            if np.array_equal(wrong, actual):
                return False

        return True
    except AssertionError:
        return False


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        mode = random.randrange(4)
        contents = _sample_contents(mode)
        if contents is None:
            continue
        grid, origin = _render_input(contents)
        if _validate_generated(grid, contents, origin):
            return grid
    raise RuntimeError("failed to generate a valid scaffold-diagonal-decoration input")

