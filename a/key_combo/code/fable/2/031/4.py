from __future__ import annotations

import random
from collections import Counter
from itertools import combinations

import numpy as np


Coord = tuple[int, int]

BLOCKER_COLOR = 8  # teal / cyan
COLOR_POOL = [0, 1, 2, 3, 4, 5, 6, 7, 9]
HEIGHT_RANGE = (10, 23)
WIDTH_RANGE = (10, 23)
TURN_CHOICES = (2, 3, 4, 5, 6)
GEN_ATTEMPTS = 500


def get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(values[np.argmax(counts)])


def colors_present(grid: np.ndarray, *, exclude: set[int] | None = None) -> list[int]:
    colors = set(map(int, np.asarray(grid).ravel()))
    if exclude is not None:
        colors -= set(map(int, exclude))
    return sorted(colors)


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    height, width = map(int, shape)
    return 0 <= int(r) < height and 0 <= int(c) < width


def _neighbors4(r: int, c: int) -> list[Coord]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _is_boundary(shape: tuple[int, int], cell: Coord) -> bool:
    r, c = map(int, cell)
    height, width = map(int, shape)
    return r in (0, height - 1) or c in (0, width - 1)


def _segment_cells(a: Coord, b: Coord) -> list[Coord]:
    r0, c0 = map(int, a)
    r1, c1 = map(int, b)
    if r0 == r1:
        step = 1 if c1 >= c0 else -1
        return [(r0, c) for c in range(c0, c1 + step, step)]
    assert c0 == c1, "polyline segments must be axis-aligned"
    step = 1 if r1 >= r0 else -1
    return [(r, c0) for r in range(r0, r1 + step, step)]


def _polyline_cells(vertices: list[Coord]) -> list[Coord]:
    cells: list[Coord] = []
    for i in range(len(vertices) - 1):
        segment = _segment_cells(vertices[i], vertices[i + 1])
        if i:
            segment = segment[1:]
        cells.extend(segment)
    return cells


def _path_adjacency(path: set[Coord]) -> dict[Coord, list[Coord]]:
    adjacency: dict[Coord, list[Coord]] = {}
    for cell in path:
        r, c = cell
        adjacency[cell] = [nbr for nbr in _neighbors4(r, c) if nbr in path]
    return adjacency


def _ordered_path(path: set[Coord]) -> list[Coord]:
    assert path, "expected a non-empty path"
    adjacency = _path_adjacency(path)
    degrees = {cell: len(nbrs) for cell, nbrs in adjacency.items()}
    assert all(degree in (1, 2) for degree in degrees.values()), "path must be simple"
    endpoints = [cell for cell, degree in degrees.items() if degree == 1]
    assert len(endpoints) == 2, "path must have exactly two endpoints"

    order = [endpoints[0]]
    prev: Coord | None = None
    cur = endpoints[0]
    while True:
        options = [nbr for nbr in adjacency[cur] if nbr != prev]
        if not options:
            break
        assert len(options) == 1, "simple path traversal must be unique"
        nxt = options[0]
        order.append(nxt)
        prev, cur = cur, nxt

    assert len(order) == len(path), "path must be connected"
    return order


def _corner_blockers(order: list[Coord], shape: tuple[int, int], path: set[Coord]) -> list[Coord]:
    blockers: list[Coord] = []
    for i in range(1, len(order) - 1):
        prev = order[i - 1]
        cur = order[i]
        nxt = order[i + 1]
        d_in = (cur[0] - prev[0], cur[1] - prev[1])
        d_out = (nxt[0] - cur[0], nxt[1] - cur[1])
        assert abs(d_in[0]) + abs(d_in[1]) == 1
        assert abs(d_out[0]) + abs(d_out[1]) == 1
        assert d_out != (-d_in[0], -d_in[1]), "path cannot U-turn"
        if d_in == d_out:
            continue
        blocker = (cur[0] + d_in[0], cur[1] + d_in[1])
        assert _in_bounds(shape, *blocker), "blocker continuation must stay in-bounds"
        assert blocker not in path, "blocker cell cannot lie on the path"
        blockers.append(blocker)
    return blockers


def _parse_input(grid: np.ndarray) -> tuple[int, int, int, Coord, set[Coord]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    bg = get_bg(g)
    fg_colors = colors_present(g, exclude={bg})
    assert len(fg_colors) == 2, "expected exactly two foreground colors"

    counts = Counter(map(int, g.ravel()))
    blocker_candidates = [color for color in fg_colors if counts[color] == 1]
    assert len(blocker_candidates) == 1, "expected one singleton blocker color"
    blocker_color = int(blocker_candidates[0])
    path_color = next(color for color in fg_colors if color != blocker_color)

    witness_coords = np.argwhere(g == blocker_color)
    assert witness_coords.shape[0] == 1, "expected exactly one blocker pixel"
    witness = tuple(map(int, witness_coords[0]))

    path_coords = {tuple(map(int, rc)) for rc in np.argwhere(g == path_color)}
    assert path_coords, "expected a visible path"
    return bg, path_color, blocker_color, witness, path_coords


def _expected_output(grid: np.ndarray) -> tuple[np.ndarray, list[Coord], list[Coord], Coord]:
    _bg, _path_color, blocker_color, witness, path = _parse_input(grid)
    order = _ordered_path(path)
    assert _is_boundary(grid.shape, order[0]) and _is_boundary(grid.shape, order[-1])

    forward = _corner_blockers(order, grid.shape, path)
    backward = _corner_blockers(order[::-1], grid.shape, path)
    assert len(forward) >= 1 and len(backward) >= 1, "expected at least one corner"
    assert len(set(forward)) == len(forward), "forward blockers must be distinct"
    assert len(set(backward)) == len(backward), "backward blockers must be distinct"

    witness_in_forward = witness in set(forward)
    witness_in_backward = witness in set(backward)
    assert witness_in_forward ^ witness_in_backward, "witness must disambiguate direction"

    chosen = forward if witness_in_forward else backward
    out = np.asarray(grid, dtype=int).copy()
    for r, c in chosen:
        out[r, c] = blocker_color
    return out, forward, backward, witness


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover every hidden blocker cell that forced a boundary-to-boundary corridor path to turn.

    Input structure:
    - The background is dominant.
    - One non-background color forms a one-cell-wide 4-connected path that starts on one boundary, turns orthogonally several times, and leaves through a boundary.
    - Exactly one extra foreground pixel of a different color marks one blocker cell; no other foreground objects exist.

    Rule:
    - The path can be traversed from either endpoint, and each traversal direction assigns one blocker cell to every corner: the cell one step straight ahead before the turn.
    - Compare the two candidate blocker sets and keep the unique traversal whose blocker set contains the given singleton marker.
    - Paint that marker color onto every blocker cell from the chosen traversal, while preserving the original path.

    Key insight:
    - The drawn path is the visible effect, but each turn implies an unseen cause just beyond the corner; the lone blocker pixel is the only evidence that selects the correct travel direction.
    """
    out, _forward, _backward, _witness = _expected_output(grid)
    return out


def _sample_spaced_values(lo: int, hi: int, count: int, min_gap: int = 2) -> list[int] | None:
    if count == 0:
        return []
    adjusted_hi = hi - (count - 1) * (min_gap - 1)
    if adjusted_hi < lo:
        return None
    bases = sorted(random.sample(range(lo, adjusted_hi + 1), count))
    return [base + i * (min_gap - 1) for i, base in enumerate(bases)]


def _sample_vertices(height: int, width: int) -> list[Coord] | None:
    valid_turns = []
    for turns in TURN_CHOICES:
        row_need = turns // 2 + 1
        col_need = turns // 2 if turns % 2 == 0 else turns // 2 + 1
        if len(range(2, height - 2)) < row_need:
            continue
        if _sample_spaced_values(2, width - 3, col_need, min_gap=2) is None:
            continue
        valid_turns.append(turns)
    if not valid_turns:
        return None

    turns = random.choice(valid_turns)
    rows = random.sample(list(range(2, height - 2)), turns // 2 + 1)
    if turns % 2 == 0:
        m = turns // 2
        cols = _sample_spaced_values(2, width - 3, m, min_gap=2)
        assert cols is not None
        vertices: list[Coord] = [(rows[0], 0)]
        for i in range(m):
            vertices.append((rows[i], cols[i]))
            vertices.append((rows[i + 1], cols[i]))
        vertices.append((rows[m], width - 1))
        return vertices

    m = turns // 2
    cols = _sample_spaced_values(2, width - 3, m + 1, min_gap=2)
    assert cols is not None
    exit_row = random.choice([0, height - 1])
    vertices = [(rows[0], 0)]
    for i in range(m + 1):
        vertices.append((rows[i], cols[i]))
        if i < m:
            vertices.append((rows[i + 1], cols[i]))
    vertices.append((exit_row, cols[-1]))
    return vertices


def _all_far_enough(points: list[Coord]) -> bool:
    return all(abs(a[0] - b[0]) + abs(a[1] - b[1]) > 1 for a, b in combinations(points, 2))


def _validate_semantics(inp: np.ndarray, expected: np.ndarray) -> bool:
    bg, path_color, blocker_color, witness, path = _parse_input(inp)
    if blocker_color != BLOCKER_COLOR:
        return False
    if path_color == blocker_color or bg in (path_color, blocker_color):
        return False
    if int(np.count_nonzero(inp == bg)) <= len(path):
        return False

    try:
        order = _ordered_path(path)
    except AssertionError:
        return False
    if not (_is_boundary(inp.shape, order[0]) and _is_boundary(inp.shape, order[-1])):
        return False

    try:
        forward = _corner_blockers(order, inp.shape, path)
        backward = _corner_blockers(order[::-1], inp.shape, path)
    except AssertionError:
        return False

    if len(forward) < 2 or len(backward) != len(forward):
        return False
    if len(set(forward)) != len(forward) or len(set(backward)) != len(backward):
        return False
    if set(forward) & set(backward):
        return False
    if not _all_far_enough(forward) or not _all_far_enough(backward):
        return False

    witness_in_forward = witness in set(forward)
    witness_in_backward = witness in set(backward)
    if not (witness_in_forward ^ witness_in_backward):
        return False

    chosen = forward if witness_in_forward else backward
    wrong = backward if witness_in_forward else forward

    expected_from_rule = inp.copy()
    for r, c in chosen:
        expected_from_rule[r, c] = blocker_color
    if not np.array_equal(expected, expected_from_rule):
        return False
    if np.array_equal(expected, inp):
        return False

    wrong_output = inp.copy()
    for r, c in wrong:
        wrong_output[r, c] = blocker_color
    if np.array_equal(expected, wrong_output):
        return False

    try:
        solved = main(inp)
    except AssertionError:
        return False
    return np.array_equal(solved, expected)


def _apply_symmetry(grid: np.ndarray, rotation_k: int, flip_lr: bool) -> np.ndarray:
    out = np.rot90(grid, rotation_k)
    if flip_lr:
        out = np.fliplr(out)
    return out.copy()


def _generate_canonical_case() -> tuple[np.ndarray, np.ndarray] | None:
    height = random.randint(*HEIGHT_RANGE)
    width = random.randint(*WIDTH_RANGE)
    vertices = _sample_vertices(height, width)
    if vertices is None:
        return None

    ordered_cells = _polyline_cells(vertices)
    path = set(ordered_cells)
    if len(path) != len(ordered_cells):
        return None

    try:
        discovered = _ordered_path(path)
    except AssertionError:
        return None
    if set(discovered) != path:
        return None
    if discovered[0] not in {ordered_cells[0], ordered_cells[-1]}:
        return None

    try:
        forward = _corner_blockers(ordered_cells, (height, width), path)
        backward = _corner_blockers(ordered_cells[::-1], (height, width), path)
    except AssertionError:
        return None

    if len(forward) < 2 or len(forward) != len(backward):
        return None
    if len(set(forward)) != len(forward) or len(set(backward)) != len(backward):
        return None
    if set(forward) & set(backward):
        return None
    if not _all_far_enough(forward) or not _all_far_enough(backward):
        return None
    if not (_is_boundary((height, width), ordered_cells[0]) and _is_boundary((height, width), ordered_cells[-1])):
        return None

    bg, path_color = random.sample(COLOR_POOL, 2)
    witness = random.choice(forward)

    inp = np.full((height, width), bg, dtype=int)
    for r, c in ordered_cells:
        inp[r, c] = path_color
    inp[witness] = BLOCKER_COLOR

    expected = inp.copy()
    for r, c in forward:
        expected[r, c] = BLOCKER_COLOR

    if not _validate_semantics(inp, expected):
        return None
    return inp, expected


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        case = _generate_canonical_case()
        if case is None:
            continue
        inp, expected = case
        rotation_k = random.randrange(4)
        flip_lr = random.random() < 0.5
        transformed_inp = _apply_symmetry(inp, rotation_k, flip_lr)
        transformed_expected = _apply_symmetry(expected, rotation_k, flip_lr)
        if _validate_semantics(transformed_inp, transformed_expected):
            return transformed_inp
    raise RuntimeError("failed to generate a valid causal-inversion path puzzle")
