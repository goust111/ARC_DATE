from __future__ import annotations

import random
from typing import Iterable

import numpy as np


BLACK = 0
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5

GEN_ATTEMPTS = 400
RADIUS_ATTEMPTS = 160
BLOCKER_ATTEMPTS = 120


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _chebyshev(a: tuple[int, int], b: tuple[int, int]) -> int:
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


def _find_cross_axes(grid: np.ndarray) -> tuple[int, int, int]:
    g = np.asarray(grid, dtype=int)
    candidates: list[tuple[int, np.ndarray, np.ndarray]] = []
    for color in np.unique(g):
        rows = np.where(np.all(g == int(color), axis=1))[0]
        cols = np.where(np.all(g == int(color), axis=0))[0]
        if len(rows) and len(cols):
            candidates.append((int(color), rows, cols))
    assert len(candidates) == 1, "expected exactly one full-row/full-column cross"
    color, rows, cols = candidates[0]
    assert len(rows) == 1 and len(cols) == 1, "cross must have one row axis and one column axis"
    return color, int(rows[0]), int(cols[0])


def _reflect_all(
    marker: tuple[int, int], axis_r: int, axis_c: int, shape: tuple[int, int]
) -> list[tuple[int, int]]:
    r, c = marker
    coords: list[tuple[int, int]] = []
    for rr in (r, 2 * axis_r - r):
        for cc in (c, 2 * axis_c - c):
            if _in_bounds(shape, rr, cc):
                coords.append((int(rr), int(cc)))
    return coords


def _complete_markers(
    originals: Iterable[tuple[int, int]], axis_r: int, axis_c: int, shape: tuple[int, int]
) -> set[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    for marker in originals:
        out.update(_reflect_all(marker, axis_r, axis_c, shape))
    return out


def _apply_fill(
    base: np.ndarray, markers: Iterable[tuple[int, int]], blockers: set[tuple[int, int]]
) -> np.ndarray:
    out = np.asarray(base, dtype=int).copy()
    blocker_list = sorted(blockers)
    assert blocker_list, "expected at least one blocker"
    for r, c in sorted(set(markers)):
        dist = min(_chebyshev((r, c), blocker) for blocker in blocker_list)
        radius = dist - 1
        assert radius >= 1, "nearest blocker must be at least Chebyshev distance 2"
        for rr in range(r - radius, r + radius + 1):
            for cc in range(c - radius, c + radius + 1):
                if _in_bounds(out.shape, rr, cc) and out[rr, cc] == BLACK:
                    out[rr, cc] = RED
    return out


def _output_from_originals(grid: np.ndarray, originals: set[tuple[int, int]]) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    cross_color, axis_r, axis_c = _find_cross_axes(g)
    assert cross_color == GREEN
    blockers = {(int(r), int(c)) for r, c in np.argwhere(g == GREY)}
    completed = _complete_markers(originals, axis_r, axis_c, g.shape)
    out = g.copy()
    for r, c in completed:
        assert out[r, c] in (BLACK, YELLOW)
        out[r, c] = YELLOW
    return _apply_fill(out, completed, blockers)


def _quadrant(cell: tuple[int, int], axis_r: int, axis_c: int) -> int:
    r, c = cell
    assert r != axis_r and c != axis_c
    if r < axis_r and c < axis_c:
        return 0
    if r < axis_r and c > axis_c:
        return 1
    if r > axis_r and c < axis_c:
        return 2
    return 3


def _max_square_radius(cell: tuple[int, int], axis_r: int, axis_c: int, shape: tuple[int, int]) -> int:
    h, w = shape
    r, c = cell
    edge_gap = min(r, h - 1 - r, c, w - 1 - c)
    if r < axis_r:
        row_axis_gap = axis_r - r - 1
    else:
        row_axis_gap = r - axis_r - 1
    if c < axis_c:
        col_axis_gap = axis_c - c - 1
    else:
        col_axis_gap = c - axis_c - 1
    return int(min(edge_gap, row_axis_gap, col_axis_gap))


def _squares_overlap(a: tuple[int, int], ra: int, b: tuple[int, int], rb: int) -> bool:
    return abs(a[0] - b[0]) <= ra + rb and abs(a[1] - b[1]) <= ra + rb


def _group_reflections(
    sources: list[tuple[int, int]], axis_r: int, axis_c: int, shape: tuple[int, int]
) -> list[list[tuple[int, int]]]:
    return [_reflect_all(source, axis_r, axis_c, shape) for source in sources]


def _sample_sources(
    shape: tuple[int, int], axis_r: int, axis_c: int, count: int
) -> list[tuple[int, int]] | None:
    row_lo = 2
    row_hi = axis_r - 3
    col_lo = 2
    col_hi = axis_c - 3
    if row_lo > row_hi or col_lo > col_hi:
        return None
    candidates = [(r, c) for r in range(row_lo, row_hi + 1) for c in range(col_lo, col_hi + 1)]
    random.shuffle(candidates)
    min_sep = random.choice((3, 4))
    sources: list[tuple[int, int]] = []
    for cell in candidates:
        if all(_chebyshev(cell, other) >= min_sep for other in sources):
            sources.append(cell)
            if len(sources) == count:
                return sources
    return None


def _sample_radii_for_groups(
    groups: list[list[tuple[int, int]]], axis_r: int, axis_c: int, shape: tuple[int, int]
) -> dict[tuple[int, int], int] | None:
    for _ in range(RADIUS_ATTEMPTS):
        radii: dict[tuple[int, int], int] = {}
        varied_group = False
        all_markers = [cell for group in groups for cell in group]
        for group_idx, group in enumerate(groups):
            max_radius = min(_max_square_radius(cell, axis_r, axis_c, shape) for cell in group)
            max_radius = min(max_radius, 3)
            if max_radius < 2:
                break
            values = list(range(1, max_radius + 1))
            if group_idx == 0:
                while True:
                    pattern = [random.choice(values) for _ in range(4)]
                    if len(set(pattern)) >= 2:
                        varied_group = True
                        break
            else:
                pattern = [random.choice(values) for _ in range(4)]
                if len(set(pattern)) >= 2:
                    varied_group = True
            for cell, radius in zip(group, pattern):
                radii[cell] = int(radius)
        else:
            if len(set(radii.values())) < 2 or not varied_group:
                continue
            ok = True
            for i, cell_a in enumerate(all_markers):
                for cell_b in all_markers[i + 1 :]:
                    if _squares_overlap(cell_a, radii[cell_a], cell_b, radii[cell_b]):
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                return radii
    return None


def _blocker_candidates(
    marker: tuple[int, int],
    radius: int,
    axis_r: int,
    axis_c: int,
    shape: tuple[int, int],
    all_markers: set[tuple[int, int]],
) -> list[tuple[int, int]]:
    h, w = shape
    d = radius + 1
    candidates: list[tuple[int, int]] = []
    quadrant = _quadrant(marker, axis_r, axis_c)
    r, c = marker
    for rr in range(max(0, r - d), min(h - 1, r + d) + 1):
        for cc in range(max(0, c - d), min(w - 1, c + d) + 1):
            cell = (rr, cc)
            if _chebyshev(marker, cell) != d:
                continue
            if cell in all_markers:
                continue
            if rr == axis_r or cc == axis_c:
                continue
            if _quadrant(cell, axis_r, axis_c) != quadrant:
                continue
            if any(_chebyshev(cell, other) <= 1 for other in all_markers):
                continue
            candidates.append(cell)
    random.shuffle(candidates)
    return candidates


def _choose_blockers(
    radii: dict[tuple[int, int], int],
    axis_r: int,
    axis_c: int,
    shape: tuple[int, int],
) -> set[tuple[int, int]] | None:
    all_markers = set(radii)
    candidate_map = {
        marker: _blocker_candidates(marker, radius, axis_r, axis_c, shape, all_markers)
        for marker, radius in radii.items()
    }
    if any(not cells for cells in candidate_map.values()):
        return None
    ordered_markers = sorted(candidate_map, key=lambda marker: len(candidate_map[marker]))

    def backtrack(idx: int, placed: list[tuple[int, int]]) -> list[tuple[int, int]] | None:
        if idx == len(ordered_markers):
            return placed.copy()
        marker = ordered_markers[idx]
        for cell in candidate_map[marker]:
            if any(_chebyshev(cell, blocker) <= 1 for blocker in placed):
                continue
            placed.append(cell)
            result = backtrack(idx + 1, placed)
            if result is not None:
                return result
            placed.pop()
        return None

    result = backtrack(0, [])
    if result is None:
        return None
    blockers = set(result)
    desired = {marker: radius + 1 for marker, radius in radii.items()}
    actual = {marker: min(_chebyshev(marker, blocker) for blocker in blockers) for marker in radii}
    if actual != desired:
        return None
    return blockers


def _add_extra_blockers(
    blockers: set[tuple[int, int]],
    radii: dict[tuple[int, int], int],
    axis_r: int,
    axis_c: int,
    shape: tuple[int, int],
) -> set[tuple[int, int]]:
    h, w = shape
    desired = {marker: radius + 1 for marker, radius in radii.items()}
    out = set(blockers)
    markers = set(radii)
    per_quadrant = {
        quadrant: sum(1 for cell in out if _quadrant(cell, axis_r, axis_c) == quadrant)
        for quadrant in range(4)
    }
    for quadrant in range(4):
        if per_quadrant[quadrant] >= 5:
            continue
        if random.random() < 0.5:
            continue
        candidates: list[tuple[int, int]] = []
        for r in range(h):
            for c in range(w):
                cell = (r, c)
                if cell in markers or cell in out:
                    continue
                if r == axis_r or c == axis_c:
                    continue
                if _quadrant(cell, axis_r, axis_c) != quadrant:
                    continue
                if any(_chebyshev(cell, marker) <= 1 for marker in markers):
                    continue
                if any(_chebyshev(cell, blocker) <= 1 for blocker in out):
                    continue
                if any(_chebyshev(cell, marker) <= desired[marker] for marker in radii):
                    continue
                candidates.append(cell)
        if candidates:
            choice = random.choice(candidates)
            out.add(choice)
            per_quadrant[quadrant] += 1
    return out


def _render_input(
    shape: tuple[int, int],
    axis_r: int,
    axis_c: int,
    sources: list[tuple[int, int]],
    blockers: set[tuple[int, int]],
) -> np.ndarray:
    grid = np.full(shape, BLACK, dtype=int)
    grid[axis_r, :] = GREEN
    grid[:, axis_c] = GREEN
    for r, c in sources:
        grid[r, c] = YELLOW
    for r, c in blockers:
        assert grid[r, c] == BLACK
        grid[r, c] = GREY
    return grid


def _verify_semantics(grid: np.ndarray, sources: list[tuple[int, int]]) -> None:
    g = np.asarray(grid, dtype=int)
    cross_color, axis_r, axis_c = _find_cross_axes(g)
    assert cross_color == GREEN
    assert g.shape[0] % 2 == 1 and g.shape[1] % 2 == 1
    assert axis_r == g.shape[0] // 2 and axis_c == g.shape[1] // 2

    originals = {(int(r), int(c)) for r, c in np.argwhere(g == YELLOW)}
    blockers = {(int(r), int(c)) for r, c in np.argwhere(g == GREY)}
    assert originals == set(sources)
    assert blockers
    assert all(r < axis_r and c < axis_c for r, c in originals)
    assert all(_quadrant(blocker, axis_r, axis_c) in {0, 1, 2, 3} for blocker in blockers)

    per_quadrant = {quadrant: 0 for quadrant in range(4)}
    for blocker in blockers:
        per_quadrant[_quadrant(blocker, axis_r, axis_c)] += 1
    assert all(2 <= count <= 5 for count in per_quadrant.values())

    for a in originals:
        for b in originals:
            if a != b:
                assert _chebyshev(a, b) > 1

    completed = _complete_markers(originals, axis_r, axis_c, g.shape)
    assert len(completed) == 4 * len(originals)

    for source in originals:
        for refl in _reflect_all(source, axis_r, axis_c, g.shape):
            if refl != source:
                assert g[refl] == BLACK

    for blocker in blockers:
        assert all(_chebyshev(blocker, marker) > 1 for marker in completed)
    blocker_pairs = sorted(blockers)
    for i, a in enumerate(blocker_pairs):
        for b in blocker_pairs[i + 1 :]:
            assert _chebyshev(a, b) > 1

    distances = {marker: min(_chebyshev(marker, blocker) for blocker in blockers) for marker in completed}
    radii = {marker: dist - 1 for marker, dist in distances.items()}
    assert min(radii.values()) >= 1
    assert len(set(radii.values())) >= 2

    groups = _group_reflections(sorted(originals), axis_r, axis_c, g.shape)
    assert any(len({radii[cell] for cell in group}) >= 2 for group in groups)

    completed_list = sorted(completed)
    for i, a in enumerate(completed_list):
        for b in completed_list[i + 1 :]:
            assert not _squares_overlap(a, radii[a], b, radii[b])

    with_symmetry = _output_from_originals(g, originals)
    without_symmetry = _apply_fill(g.copy(), originals, blockers)
    assert not np.array_equal(with_symmetry, without_symmetry)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reflect single-quadrant yellow markers across a full cross, then grow red squares sized by nearby grey blockers.

    Input structure:
    - A black background contains one full green row and one full green column forming a centered cross axis.
    - Yellow marker pixels lie away from the axes, initially only in one quadrant, and grey blocker pixels are scattered across all quadrants.
    - Markers and blockers are sparse single pixels; blockers do not touch the markers.

    Rule:
    - Find the unique full-row/full-column cross and use its row and column as mirror axes.
    - Reflect every yellow marker across the row axis, the column axis, and both axes, keeping only in-bounds copies; paint those reflected positions yellow.
    - For each yellow marker in the completed set, compute the minimum Chebyshev distance to any grey blocker.
    - Let the fill radius be one less than that distance, and color the centered square red, but only on black background cells, so green axes, yellow markers, grey blockers, and earlier red cells stay unchanged.

    Key insight:
    - The fill centers are incomplete until the yellow markers are mirrored across both cross axes, and each reflected copy can receive a different square size because its nearest grey blocker is chosen in its own local quadrant geometry.
    """
    g = np.asarray(grid, dtype=int)
    cross_color, axis_r, axis_c = _find_cross_axes(g)
    assert cross_color == GREEN

    originals = {(int(r), int(c)) for r, c in np.argwhere(g == YELLOW)}
    blockers = {(int(r), int(c)) for r, c in np.argwhere(g == GREY)}
    assert originals, "expected at least one yellow marker"
    assert blockers, "expected at least one grey blocker"

    completed = _complete_markers(originals, axis_r, axis_c, g.shape)
    out = g.copy()
    for r, c in completed:
        assert out[r, c] in (BLACK, YELLOW)
        out[r, c] = YELLOW
    return _apply_fill(out, completed, blockers)


def _generate_attempt() -> np.ndarray | None:
    h = random.choice((19, 21, 23, 25))
    w = random.choice((19, 21, 23, 25))
    axis_r = h // 2
    axis_c = w // 2
    source_count = random.randint(2, 4)

    sources = _sample_sources((h, w), axis_r, axis_c, source_count)
    if sources is None:
        return None

    groups = _group_reflections(sources, axis_r, axis_c, (h, w))
    radii = _sample_radii_for_groups(groups, axis_r, axis_c, (h, w))
    if radii is None:
        return None

    blockers = None
    for _ in range(BLOCKER_ATTEMPTS):
        base_blockers = _choose_blockers(radii, axis_r, axis_c, (h, w))
        if base_blockers is None:
            continue
        blockers = _add_extra_blockers(base_blockers, radii, axis_r, axis_c, (h, w))
        actual = {
            marker: min(_chebyshev(marker, blocker) for blocker in blockers)
            for marker in radii
        }
        if actual == {marker: radius + 1 for marker, radius in radii.items()}:
            break
        blockers = None
    if blockers is None:
        return None

    grid = _render_input((h, w), axis_r, axis_c, sources, blockers)
    try:
        _verify_semantics(grid, sources)
    except AssertionError:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid symmetry-plus-blocker ARC input")
