from __future__ import annotations

import random
from collections import Counter

import numpy as np


BG = 0
FG_COLORS = (1, 2, 3, 7)
MARKER = 4
SEPARATOR = 5

QUADRANT_COUNT = 4
QUADRANT_RANGE = (6, 7)
SHAPE_SIZE_RANGE = (4, 7)
MAX_GENERATION_ATTEMPTS = 600
MARK_PAIR_ATTEMPTS = 300

TL, TR, BL, BR = range(4)
QUAD_ORDER = (TL, TR, BL, BR)
CORNER_BIAS = {
    TL: (-1, -1),
    TR: (-1, 1),
    BL: (1, -1),
    BR: (1, 1),
}


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _in_bounds(q: int, cell: tuple[int, int]) -> bool:
    r, c = cell
    return 0 <= r < q and 0 <= c < q


def _chebyshev(a: tuple[int, int], b: tuple[int, int]) -> int:
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


def _is_connected(cells: set[tuple[int, int]]) -> bool:
    if not cells:
        return False
    seen = {next(iter(cells))}
    stack = [next(iter(seen))]
    while stack:
        r, c = stack.pop()
        for nbr in _neighbors4(r, c):
            if nbr in cells and nbr not in seen:
                seen.add(nbr)
                stack.append(nbr)
    return len(seen) == len(cells)


def _most_common_color(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.flat)).most_common(1)[0][0])


def _find_separator_cross(grid: np.ndarray) -> tuple[int, int]:
    h, w = map(int, grid.shape)
    rows = [r for r in range(h) if np.all(grid[r, :] == SEPARATOR)]
    cols = [c for c in range(w) if np.all(grid[:, c] == SEPARATOR)]
    assert len(rows) == 1 and len(cols) == 1, "expected one full separator row and one full separator column"
    cross_r = rows[0]
    cross_c = cols[0]
    assert cross_r == h - cross_r - 1, "separator row must split equal top/bottom quadrants"
    assert cross_c == w - cross_c - 1, "separator column must split equal left/right quadrants"
    return cross_r, cross_c


def _split_quadrants(grid: np.ndarray) -> list[np.ndarray]:
    cross_r, cross_c = _find_separator_cross(grid)
    return [
        grid[:cross_r, :cross_c].copy(),
        grid[:cross_r, cross_c + 1 :].copy(),
        grid[cross_r + 1 :, :cross_c].copy(),
        grid[cross_r + 1 :, cross_c + 1 :].copy(),
    ]


def _parse_quadrant(quadrant: np.ndarray) -> dict:
    colors = sorted(set(map(int, np.unique(quadrant))) - {BG, MARKER})
    assert colors, "each quadrant must contain one foreground shape"
    assert len(colors) == 1, "each quadrant must use exactly one foreground color"
    color = colors[0]
    shape = {
        (int(r), int(c))
        for r, c in np.argwhere(quadrant == color)
    }
    marker_cells = [(int(r), int(c)) for r, c in np.argwhere(quadrant == MARKER)]
    assert 4 <= len(shape) <= 8, "foreground shape size must be within the puzzle range"
    assert _is_connected(shape), "foreground shape must be 4-connected"
    assert len(marker_cells) <= 1, "each quadrant may contain at most one marker"
    marker = marker_cells[0] if marker_cells else None
    if marker is not None:
        assert all(_chebyshev(marker, cell) >= 2 for cell in shape), "marker must be separated from its shape"
    return {"color": color, "shape": shape, "marker": marker}


def _overlay_oracle(quadrants: list[dict], q: int) -> tuple[np.ndarray, dict[tuple[int, int], list[int]], dict[tuple[int, int], int]]:
    out = np.full((q, q), BG, dtype=int)
    contenders_by_cell: dict[tuple[int, int], list[int]] = {}
    winners: dict[tuple[int, int], int] = {}

    for r in range(q):
        for c in range(q):
            cell = (r, c)
            contenders = [idx for idx, info in enumerate(quadrants) if cell in info["shape"]]
            contenders_by_cell[cell] = contenders
            if not contenders:
                continue
            if len(contenders) == 1:
                winner = contenders[0]
            else:
                scored = []
                for idx in contenders:
                    marker = quadrants[idx]["marker"]
                    if marker is None:
                        continue
                    scored.append((_chebyshev(marker, cell), idx))
                assert scored, "every conflict must involve at least one marked quadrant"
                dists = [dist for dist, _idx in scored]
                assert len(set(dists)) == len(dists), "marked contenders must have distinct Chebyshev distances"
                winner = min(scored)[1]
            winners[cell] = winner
            out[r, c] = quadrants[winner]["color"]
    return out, contenders_by_cell, winners


def _contenders_by_cell(shapes: list[set[tuple[int, int]]], q: int) -> dict[tuple[int, int], list[int]]:
    contenders: dict[tuple[int, int], list[int]] = {}
    for r in range(q):
        for c in range(q):
            cell = (r, c)
            contenders[cell] = [idx for idx, shape in enumerate(shapes) if cell in shape]
    return contenders


def main(grid: np.ndarray) -> np.ndarray:
    """Overlay the four separator-split quadrants and resolve overlaps by nearest marker distance.

    Input structure:
    - A black background is split by one full grey row and one full grey column into four equal quadrants.
    - Each quadrant contains one 4-connected foreground shape in its own color, plus optionally one yellow single-pixel marker.
    - Markers are not part of the shapes and are separated from their quadrant's shape by at least one cell.

    Rule:
    - Remove the grey separator cross and align the four quadrants into one shared quadrant-sized canvas.
    - At each aligned position, output the unique foreground color if exactly one quadrant has shape there; output black if none do.
    - If multiple quadrants occupy the same position, ignore any unmarked quadrant, compute each marked quadrant's Chebyshev distance from its yellow marker to that position, and keep the color of the uniquely nearest marked quadrant.

    Key insight:
    - The yellow pixels are not copied; they are spatial priority anchors whose local distances decide each overlap independently, so a quadrant can win one conflict and lose another.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"
    assert _most_common_color(g) == BG, "background must be black"

    quadrant_grids = _split_quadrants(g)
    quadrants = [_parse_quadrant(qgrid) for qgrid in quadrant_grids]
    assert len({info["color"] for info in quadrants}) == QUADRANT_COUNT, "quadrant colors must be distinct"
    q = int(quadrant_grids[0].shape[0])
    return _overlay_oracle(quadrants, int(q))[0]


def _line_setup(q: int) -> tuple[list[tuple[int, int]], list[tuple[int, int]], list[tuple[int, int]]]:
    horizontal = random.choice([True, False])
    if horizontal:
        r = random.randint(2, q - 3)
        c = random.randint(1, q - 4)
        line = [(r, c), (r, c + 1), (r, c + 2)]
        if random.choice([True, False]):
            pool_a = [(r - 1, c), (r - 1, c + 1), (r, c - 1), (r + 1, c)]
            pool_b = [(r + 1, c + 2), (r + 1, c + 1), (r, c + 3), (r - 1, c + 2)]
        else:
            pool_a = [(r + 1, c), (r + 1, c + 1), (r, c - 1), (r - 1, c)]
            pool_b = [(r - 1, c + 2), (r - 1, c + 1), (r, c + 3), (r + 1, c + 2)]
    else:
        r = random.randint(1, q - 4)
        c = random.randint(2, q - 3)
        line = [(r, c), (r + 1, c), (r + 2, c)]
        if random.choice([True, False]):
            pool_a = [(r, c - 1), (r + 1, c - 1), (r - 1, c), (r, c + 1)]
            pool_b = [(r + 2, c + 1), (r + 1, c + 1), (r + 3, c), (r + 2, c - 1)]
        else:
            pool_a = [(r, c + 1), (r + 1, c + 1), (r - 1, c), (r, c - 1)]
            pool_b = [(r + 2, c - 1), (r + 1, c - 1), (r + 3, c), (r + 2, c + 1)]
    assert all(_in_bounds(q, cell) for cell in line + pool_a + pool_b)
    return line, pool_a, pool_b


def _make_line_shape(base_line: list[tuple[int, int]], extra_pool: list[tuple[int, int]]) -> set[tuple[int, int]]:
    target_size = random.randint(*SHAPE_SIZE_RANGE)
    need = target_size - len(base_line)
    extras = random.sample(extra_pool, need)
    cells = set(base_line) | set(extras)
    assert _is_connected(cells)
    return cells


def _grow_shape(
    q: int,
    base: set[tuple[int, int]],
    target_size: int,
    blocked: set[tuple[int, int]],
    bias: tuple[int, int],
) -> set[tuple[int, int]] | None:
    cells = set(base)
    if any(not _in_bounds(q, cell) for cell in cells):
        return None
    if any(cell in blocked for cell in cells):
        return None

    while len(cells) < target_size:
        frontier = []
        for r, c in cells:
            for nbr in _neighbors4(r, c):
                if not _in_bounds(q, nbr) or nbr in cells or nbr in blocked:
                    continue
                frontier.append(nbr)
        frontier = sorted(set(frontier))
        if not frontier:
            return None
        score_base_r = sum(r for r, _ in cells) / len(cells)
        score_base_c = sum(c for _, c in cells) / len(cells)

        def score(cell: tuple[int, int]) -> tuple[int, float, float]:
            dr = cell[0] - score_base_r
            dc = cell[1] - score_base_c
            outward = dr * bias[0] + dc * bias[1]
            edge = min(cell[0], cell[1], q - 1 - cell[0], q - 1 - cell[1])
            return (int(round(outward * 10)), -edge, random.random())

        best = max(frontier, key=score)
        cells.add(best)

    return cells if _is_connected(cells) else None


def _available_marker_cells(q: int, shape: set[tuple[int, int]]) -> list[tuple[int, int]]:
    cells = []
    for r in range(q):
        for c in range(q):
            cell = (r, c)
            if cell in shape:
                continue
            if all(_chebyshev(cell, s) >= 2 for s in shape):
                cells.append(cell)
    random.shuffle(cells)
    return cells


def _choose_primary_markers(
    q: int,
    shapes: list[set[tuple[int, int]]],
    primary: tuple[int, int],
    marked_set: set[int],
) -> dict[int, tuple[int, int]] | None:
    a, b = primary
    contenders_by_cell = _contenders_by_cell(shapes, q)
    conflict_cells = [
        cell
        for cell, contenders in contenders_by_cell.items()
        if len(contenders) > 1 and a in contenders and b in contenders
    ]
    if len(conflict_cells) < 2:
        return None

    union_shape = set().union(*shapes)
    preferred_a = [cell for cell in _available_marker_cells(q, shapes[a]) if cell not in union_shape]
    preferred_b = [cell for cell in _available_marker_cells(q, shapes[b]) if cell not in union_shape]
    candidates_a = preferred_a or _available_marker_cells(q, shapes[a])
    candidates_b = preferred_b or _available_marker_cells(q, shapes[b])

    pairs = [(ma, mb) for ma in candidates_a for mb in candidates_b if ma != mb]
    random.shuffle(pairs)
    for ma, mb in pairs[:MARK_PAIR_ATTEMPTS]:
        wins = {a: 0, b: 0}
        valid = True
        for cell in conflict_cells:
            da = _chebyshev(ma, cell)
            db = _chebyshev(mb, cell)
            if da == db:
                valid = False
                break
            wins[a if da < db else b] += 1
        if valid and wins[a] > 0 and wins[b] > 0:
            markers = {a: ma, b: mb}
            for idx in marked_set - {a, b}:
                extras = [cell for cell in _available_marker_cells(q, shapes[idx]) if cell not in markers.values()]
                if not extras:
                    valid = False
                    break
                extras.sort(key=lambda cell: (cell in union_shape, random.random()))
                markers[idx] = extras[0]
            if valid:
                return markers
    return None


def _shape_wins_some_conflicts(winners: dict[tuple[int, int], int], shape_idx: int) -> bool:
    return any(idx == shape_idx for idx in winners.values())


def _build_example() -> tuple[np.ndarray, np.ndarray] | None:
    q = random.randint(*QUADRANT_RANGE)
    colors = random.sample(FG_COLORS, QUADRANT_COUNT)
    primary = tuple(random.sample(list(QUAD_ORDER), 2))
    mark_count = random.choices([2, 3, 4], weights=[5, 3, 1], k=1)[0]
    marked_set = set(primary)
    while len(marked_set) < mark_count:
        marked_set.add(random.choice(QUAD_ORDER))

    shapes: list[set[tuple[int, int]] | None] = [None] * QUADRANT_COUNT
    line, pool_a, pool_b = _line_setup(q)
    a, b = primary
    shapes[a] = _make_line_shape(line, pool_a)
    shapes[b] = _make_line_shape(line, pool_b)
    occupied = set(shapes[a]) | set(shapes[b])

    unmarked_quads = [idx for idx in QUAD_ORDER if idx not in marked_set]
    shared_unmarked = None
    if unmarked_quads and random.random() < 0.75:
        shared_unmarked = random.choice(unmarked_quads)
        shared_cell = random.choice([line[0], line[-1]])
        target = random.randint(*SHAPE_SIZE_RANGE)
        shape = _grow_shape(q, {shared_cell}, target, occupied - {shared_cell}, CORNER_BIAS[shared_unmarked])
        if shape is None:
            return None
        shapes[shared_unmarked] = shape
        occupied |= shape

    remaining = [idx for idx in QUAD_ORDER if shapes[idx] is None]
    random.shuffle(remaining)
    for idx in remaining:
        free_cells = [(r, c) for r in range(q) for c in range(q) if (r, c) not in occupied]
        if not free_cells:
            return None
        free_cells.sort(
            key=lambda cell: (
                _chebyshev(cell, (0 if CORNER_BIAS[idx][0] < 0 else q - 1, 0 if CORNER_BIAS[idx][1] < 0 else q - 1)),
                random.random(),
            )
        )
        seed = free_cells[0]
        target = random.randint(*SHAPE_SIZE_RANGE)
        shape = _grow_shape(q, {seed}, target, occupied, CORNER_BIAS[idx])
        if shape is None:
            return None
        shapes[idx] = shape
        occupied |= shape

    assert all(shape is not None for shape in shapes)
    final_shapes = [set(shape) for shape in shapes]
    markers = _choose_primary_markers(q, final_shapes, primary, marked_set)
    if markers is None:
        return None

    quadrant_infos = [
        {"color": colors[idx], "shape": final_shapes[idx], "marker": markers.get(idx)}
        for idx in QUAD_ORDER
    ]
    out, contenders_by_cell, winners = _overlay_oracle(quadrant_infos, q)

    conflict_cells = [cell for cell, contenders in contenders_by_cell.items() if len(contenders) > 1]
    if len(conflict_cells) < 2:
        return None
    if not (_shape_wins_some_conflicts(winners, a) and _shape_wins_some_conflicts(winners, b)):
        return None
    if any(len([idx for idx in contenders if quadrant_infos[idx]["marker"] is not None]) == 0 for cell, contenders in contenders_by_cell.items() if len(contenders) > 1):
        return None
    if any(
        len({
            _chebyshev(quadrant_infos[idx]["marker"], cell)
            for idx in contenders
            if quadrant_infos[idx]["marker"] is not None
        }) != len([idx for idx in contenders if quadrant_infos[idx]["marker"] is not None])
        for cell, contenders in contenders_by_cell.items()
        if len(contenders) > 1
    ):
        return None
    if any(not any(winner == idx for winner in winners.values()) and not any(cell in final_shapes[idx] for cell, contenders in contenders_by_cell.items() if len(contenders) == 1 and idx in contenders) for idx in QUAD_ORDER):
        return None
    if len(set(map(int, out.flat)) - {BG}) < 3:
        return None
    if not any(
        all(marker != cell for shape in final_shapes for cell in shape)
        for marker in markers.values()
    ):
        return None

    grid = np.full((2 * q + 1, 2 * q + 1), BG, dtype=int)
    grid[q, :] = SEPARATOR
    grid[:, q] = SEPARATOR
    origins = [(0, 0), (0, q + 1), (q + 1, 0), (q + 1, q + 1)]
    for idx, (r0, c0) in enumerate(origins):
        for r, c in final_shapes[idx]:
            grid[r0 + r, c0 + c] = colors[idx]
        marker = markers.get(idx)
        if marker is not None:
            grid[r0 + marker[0], c0 + marker[1]] = MARKER

    try:
        solved = main(grid)
    except Exception:
        return None
    if not np.array_equal(solved, out):
        return None
    return grid, out


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        built = _build_example()
        if built is not None:
            return built[0]
    raise RuntimeError("failed to generate a valid quadrant-overlay puzzle")
