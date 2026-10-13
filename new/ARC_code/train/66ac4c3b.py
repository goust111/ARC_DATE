from __future__ import annotations

import numpy as np

from arc_utils import colors_present, get_bg, is_contiguous
import random


def _marker_line(coords: np.ndarray) -> tuple[int, int, set[int]]:
    rows = np.unique(coords[:, 0])
    cols = np.unique(coords[:, 1])
    same_row = len(rows) == 1
    same_col = len(cols) == 1
    assert same_row != same_col, "marker pixels must form one row or one column"

    if same_row:
        return 0, int(rows[0]), set(map(int, cols))
    return 1, int(cols[0]), set(map(int, rows))


def _find_sparse_marker(
    g: np.ndarray, colors: list[int]
) -> tuple[int, int, int, set[int]]:
    candidates: list[tuple[int, int, int, set[int]]] = []
    for color in colors:
        coords = np.argwhere(g == color)
        rows = np.unique(coords[:, 0])
        cols = np.unique(coords[:, 1])
        if (len(rows) == 1) != (len(cols) == 1):
            axis, line, positions = _marker_line(coords)
            candidates.append((color, axis, line, positions))

    assert len(candidates) == 1, "expected exactly one sparse marker color"
    return candidates[0]


def _matching_source_line(
    g: np.ndarray, source: int, axis: int, marker_positions: set[int]
) -> int:
    other_axis = 1 - axis
    coords = np.argwhere(g == source)
    candidates = []
    for line in sorted(set(map(int, coords[:, axis]))):
        positions = {
            int(coord[other_axis])
            for coord in coords
            if int(coord[axis]) == line
        }
        if positions == marker_positions:
            candidates.append(line)

    assert len(candidates) == 1, "expected one matching source marker line"
    return int(candidates[0])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete the sparse-color reflected copy of the foreground motif across its matching marker row or column.

    Input structure
    - The grid has one background color and two foreground colors.
    - One foreground color appears only as a sparse marker row or column.
    - The other foreground color contains the same sparse marker positions on a parallel row or column, plus the motif on the far side of the sparse markers.

    Rule
    - Keep the original grid unchanged.
    - Reflect every motif pixel of the larger foreground color across the two parallel marker lines.
    - Paint the reflected motif pixels with the sparse marker color.

    Key insight
    - The two equal marker rows or columns are paired landmarks: the sparse-color marker line borders the source motif, and the matching source-color marker line anchors the reflected copy.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    colors = colors_present(g, exclude={bg})
    assert len(colors) == 2, "expected exactly two foreground colors"

    target, axis, target_line, marker_positions = _find_sparse_marker(g, colors)
    source = next(color for color in colors if color != target)
    source_line = _matching_source_line(g, source, axis, marker_positions)

    line_delta = source_line - target_line
    assert line_delta != 0, "marker lines must be distinct"
    direction = 1 if line_delta > 0 else -1
    other_axis = 1 - axis

    source_coords = np.argwhere(g == source)
    target_coords = np.argwhere(g == target)
    assert len(marker_positions) == len(target_coords), "marker positions must be unique"

    body_coords: list[tuple[int, int]] = []
    source_marker_count = 0
    for coord_arr in source_coords:
        coord = (int(coord_arr[0]), int(coord_arr[1]))
        on_source_marker = (
            coord[axis] == source_line and coord[other_axis] in marker_positions
        )
        on_body_side = (coord[axis] - target_line) * direction < 0
        assert on_source_marker or on_body_side, "source pixels must be motif or marker"
        if on_source_marker:
            source_marker_count += 1
        else:
            body_coords.append(coord)

    assert source_marker_count == len(target_coords), "marker lines must have equal length"
    assert len(body_coords) + len(target_coords) == len(source_coords)

    out = g.copy()
    h, w = g.shape
    for coord in body_coords:
        dst = [coord[0], coord[1]]
        dst[axis] = source_line + target_line - coord[axis]
        r, c = dst
        assert 0 <= r < h and 0 <= c < w, "reflected motif must stay in bounds"
        assert out[r, c] == bg, "reflected motif must land on background"
        out[r, c] = target

    return out

# --------------- generator (att2) ---------------

# Canvas, palette, and retry budget.
GRID_SIZE = 22
COLOR_COUNT = 10
PALETTE_SIZE = 3
GEN_ATTEMPTS = 300

# Marker layout and connection profiles.
MARKER_LAYOUTS = [(6, 4), (7, 3), (8, 3)]
LINE_DIRECTIONS = [-1, 1]
SPINE_PROFILES_BY_SPAN = {
    3: ((0, 0, 0, 0), (0, 1, 1, 0), (0, 1, 0, 0), (0, 0, 1, 0)),
    4: (
        (0, 0, 0, 0, 0),
        (0, 1, 1, 1, 0),
        (0, 1, 2, 1, 0),
        (0, 1, 1, 0, 0),
        (0, 0, 1, 1, 0),
    ),
}

# Body geometry.
BODY_DEPTH_RANGE = (3, 6)
BODY_GAP_MIN = 4
BODY_GAP_MAX = 6
BODY_GAP_EXTRA_MAX = 2
BODY_CROSS_EXTENSION_RANGE = (0, 1)
BODY_PIXEL_MIN = 30
BODY_PIXEL_MAX = 45
BODY_CONNECTIVITY = 4


def _connect_marker_spine(mask: np.ndarray, positions: list[int]) -> None:
    for a, b in zip(positions, positions[1:]):
        profile = random.choice(SPINE_PROFILES_BY_SPAN[b - a])
        mask[0, a] = True
        prev = profile[0]
        for offset, depth in enumerate(profile[1:], start=1):
            c = a + offset
            lo, hi = sorted((prev, depth))
            mask[lo : hi + 1, c] = True
            mask[prev, c] = True
            prev = depth


def _repair_marker_like_body_lines(mask: np.ndarray, marker_positions: set[int], cross_max: int) -> None:
    for depth in range(mask.shape[0]):
        line_positions = {int(c) for c in np.flatnonzero(mask[depth])}
        if line_positions == marker_positions:
            extra = 1 if 1 <= cross_max else 0
            if extra in marker_positions:
                extra = min(cross_max, max(marker_positions) - 1)
            mask[depth, extra] = True


def _grow_body(mask: np.ndarray, target_count: int, cross_max: int) -> bool:
    while int(mask.sum()) < target_count:
        frontier: set[tuple[int, int]] = set()
        for d, c in np.argwhere(mask):
            d = int(d)
            c = int(c)
            for nd, nc in ((d - 1, c), (d + 1, c), (d, c - 1), (d, c + 1)):
                if (
                    0 <= nd < mask.shape[0]
                    and 0 <= nc <= cross_max
                    and not mask[nd, nc]
                ):
                    frontier.add((nd, nc))
        if not frontier:
            return False
        d, c = random.choice(tuple(frontier))
        mask[d, c] = True
    return True


def _make_body(depth: int, positions: list[int]) -> np.ndarray | None:
    last_pos = positions[-1]
    cross_max = min(GRID_SIZE - 1, last_pos + random.randint(*BODY_CROSS_EXTENSION_RANGE))
    marker_positions = set(positions)
    mask = np.zeros((depth, GRID_SIZE), dtype=bool)

    _connect_marker_spine(mask, positions)

    coords = np.argwhere(mask)
    root_d, root_c = map(int, random.choice(coords))
    mask[root_d:depth, root_c] = True

    _repair_marker_like_body_lines(mask, marker_positions, cross_max)
    current = int(mask.sum())
    if current > BODY_PIXEL_MAX:
        return None

    target_count = random.randint(max(BODY_PIXEL_MIN, current), BODY_PIXEL_MAX)
    if not _grow_body(mask, target_count, cross_max):
        return None

    _repair_marker_like_body_lines(mask, marker_positions, cross_max)
    if not (BODY_PIXEL_MIN <= int(mask.sum()) <= BODY_PIXEL_MAX):
        return None
    if not is_contiguous(mask.astype(int), background=0, connectivity=BODY_CONNECTIVITY):
        return None
    if not set(np.flatnonzero(mask[0])).issuperset(marker_positions):
        return None
    if max(np.argwhere(mask)[:, 0]) != depth - 1:
        return None
    return mask


def _sample_layout() -> tuple[int, int, list[int], int, int, int] | None:
    """Return axis, direction, marker positions, depth, target line, and source line."""
    axis = random.randint(0, 1)
    direction = random.choice(LINE_DIRECTIONS)
    count, spacing = random.choice(MARKER_LAYOUTS)
    positions = [i * spacing for i in range(count)]
    depth = random.randint(*BODY_DEPTH_RANGE)
    gap = random.randint(max(BODY_GAP_MIN, depth), min(BODY_GAP_MAX, depth + BODY_GAP_EXTRA_MAX))

    if direction == 1:
        lo, hi = depth, GRID_SIZE - 1 - gap - depth
    else:
        lo, hi = gap + depth, GRID_SIZE - 1 - depth
    if lo > hi:
        return None
    target_line = random.randint(lo, hi)
    source_line = target_line + direction * gap
    return axis, direction, positions, depth, target_line, source_line


def _render_candidate(bg, target, source, axis, direction, positions, target_line, source_line, body):
    """Return input/output pair after drawing the body and hiding its reflection."""
    out = np.full((GRID_SIZE, GRID_SIZE), bg, dtype=int)

    if axis == 0:
        out[target_line, positions] = target
        out[source_line, positions] = source
    else:
        out[positions, target_line] = target
        out[positions, source_line] = source

    offsets = np.argwhere(body)
    depths = offsets[:, 0] + 1
    crosses = offsets[:, 1]
    body_lines = target_line - direction * depths
    reflected_lines = source_line + direction * depths
    if axis == 0:
        body_rows, body_cols = body_lines, crosses
        reflected_rows, reflected_cols = reflected_lines, crosses
    else:
        body_rows, body_cols = crosses, body_lines
        reflected_rows, reflected_cols = crosses, reflected_lines

    body_bg = out[body_rows, body_cols] == bg
    reflected_bg = out[reflected_rows, reflected_cols] == bg
    if not (np.all(body_bg) and np.all(reflected_bg)):
        return None

    out[body_rows, body_cols] = source
    out[reflected_rows, reflected_cols] = target

    inp = out.copy()
    inp[reflected_rows, reflected_cols] = bg
    return inp, out


def _gen_attempt() -> np.ndarray | None:
    bg, target, source = random.sample(range(COLOR_COUNT), PALETTE_SIZE)

    layout = _sample_layout()
    if layout is None:
        return None
    axis, direction, positions, depth, target_line, source_line = layout

    body = _make_body(depth, positions)
    if body is None:
        return None

    rendered = _render_candidate(bg, target, source, axis, direction, positions, target_line, source_line, body)
    if rendered is None:
        return None
    inp, out = rendered

    try:
        if np.array_equal(main(inp), out):
            return inp
    except Exception:
        return None
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 66ac4c3b input")
