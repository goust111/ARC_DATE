from __future__ import annotations

import random
from typing import Iterable, Sequence

import numpy as np


BACKGROUND = 0
OBSTACLE = 5
MARKER_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
DIRS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))

# --------------- solver ---------------


def _iter_markers(grid: np.ndarray) -> list[tuple[int, int, int]]:
    coords = np.argwhere((grid != BACKGROUND) & (grid != OBSTACLE))
    return [(int(r), int(c), int(grid[r, c])) for r, c in coords]


def _ray_cells(
    grid: np.ndarray,
    start: tuple[int, int],
    direction: tuple[int, int],
) -> Iterable[tuple[int, int]]:
    h, w = grid.shape
    r, c = start
    dr, dc = direction
    rr, cc = r + dr, c + dc
    while 0 <= rr < h and 0 <= cc < w and int(grid[rr, cc]) != OBSTACLE:
        yield rr, cc
        rr += dr
        cc += dc


def _collect_claims(
    grid: np.ndarray,
) -> tuple[list[tuple[int, int, int]], dict[tuple[int, int], set[int]]]:
    markers = _iter_markers(grid)
    claims: dict[tuple[int, int], set[int]] = {}

    for marker_idx, (r, c, _color) in enumerate(markers):
        parity = (r + c) % 2
        for direction in DIRS4:
            for rr, cc in _ray_cells(grid, (r, c), direction):
                if int(grid[rr, cc]) != BACKGROUND:
                    continue
                if (rr + cc) % 2 != parity:
                    continue
                claims.setdefault((rr, cc), set()).add(marker_idx)

    return markers, claims


def main(grid: np.ndarray) -> np.ndarray:
    """
    Project each colored marker along four orthogonal rays, but only paint same-parity ray cells and erase overlaps.

    Input structure:
    - The background is black.
    - Gray obstacle pixels are isolated blockers that stop rays but remain unchanged.
    - Two to four non-black, non-gray colored marker pixels are scattered apart from one another.

    Rule:
    - For each marker at (r, c), compute the checkerboard parity (r + c) % 2.
    - Extend rays upward, downward, leftward, and rightward until a gray obstacle or the grid edge.
    - Along those ray paths, only background cells whose checkerboard parity matches the source marker become paint claims.
    - After all markers are processed, any background cell claimed by multiple markers is reset to black, while a singly claimed cell takes its unique source marker color.
    - Preserve the original markers, gray obstacles, and every untouched background cell.

    Key insight:
    - The ray geometry is not enough: each marker also fixes a checkerboard phase, so every ray colors only every other reachable background cell, and only those filtered cells participate in conflict cancellation.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    markers, claims = _collect_claims(g)
    out = g.copy()
    for (r, c), source_ids in claims.items():
        if len(source_ids) == 1:
            source = next(iter(source_ids))
            out[r, c] = markers[source][2]
        else:
            out[r, c] = BACKGROUND
    return out


# --------------- generator ---------------

GEN_HEIGHT_RANGE = (13, 18)
GEN_WIDTH_RANGE = (13, 18)
GEN_MARKER_COUNTS = (2, 2, 3, 3, 4)
GEN_OBSTACLE_COUNT_RANGE = (8, 15)
GEN_LAYOUTS = ("rect", "row", "col")
GEN_LAYOUT_WEIGHTS = (0.5, 0.25, 0.25)
GEN_MAX_ATTEMPTS = 500


def _manhattan(a: tuple[int, int], b: tuple[int, int]) -> int:
    return abs(int(a[0]) - int(b[0])) + abs(int(a[1]) - int(b[1]))


def _chebyshev(a: tuple[int, int], b: tuple[int, int]) -> int:
    return max(abs(int(a[0]) - int(b[0])), abs(int(a[1]) - int(b[1])))


def _segment_cells(a: tuple[int, int], b: tuple[int, int]) -> set[tuple[int, int]]:
    r1, c1 = a
    r2, c2 = b
    cells: set[tuple[int, int]] = set()
    if r1 == r2:
        lo, hi = sorted((c1, c2))
        cells.update((r1, c) for c in range(lo, hi + 1))
    elif c1 == c2:
        lo, hi = sorted((r1, r2))
        cells.update((r, c1) for r in range(lo, hi + 1))
    else:
        raise ValueError("expected an axis-aligned segment")
    return cells


def _cells_along_direction(
    shape: tuple[int, int],
    start: tuple[int, int],
    direction: tuple[int, int],
    max_steps: int | None = None,
) -> list[tuple[int, int]]:
    h, w = shape
    r, c = start
    dr, dc = direction
    out: list[tuple[int, int]] = []
    step = 1
    rr, cc = r + dr, c + dc
    while 0 <= rr < h and 0 <= cc < w and (max_steps is None or step <= max_steps):
        out.append((rr, cc))
        rr += dr
        cc += dc
        step += 1
    return out


def _choose_layout(shape: tuple[int, int]) -> dict[str, object] | None:
    h, w = shape
    layout = random.choices(GEN_LAYOUTS, weights=GEN_LAYOUT_WEIGHTS, k=1)[0]

    if layout == "rect":
        row_pairs = [(r1, r2) for r1 in range(h) for r2 in range(r1 + 4, h) if (r1 - r2) % 2 == 0]
        col_pairs = [(c1, c2) for c1 in range(w) for c2 in range(c1 + 4, w) if (c1 - c2) % 2 == 0]
        if not row_pairs or not col_pairs:
            return None
        r1, r2 = random.choice(row_pairs)
        c1, c2 = random.choice(col_pairs)
        markers = [(r1, c1), (r2, c2)]
        reserve = (
            _segment_cells((r1, c1), (r1, c2))
            | _segment_cells((r1, c1), (r2, c1))
            | _segment_cells((r2, c2), (r2, c1))
            | _segment_cells((r2, c2), (r1, c2))
        )
        block_dirs = [
            ((r1, c1), (-1, 0)),
            ((r1, c1), (0, -1)),
            ((r2, c2), (1, 0)),
            ((r2, c2), (0, 1)),
        ]
    elif layout == "row":
        rows = list(range(h))
        if w < 9:
            return None
        r = random.choice(rows)
        choices = [(c1, c2) for c1 in range(2, w - 4) for c2 in range(c1 + 4, w - 2) if (c1 - c2) % 2 == 0]
        if not choices:
            return None
        c1, c2 = random.choice(choices)
        markers = [(r, c1), (r, c2)]
        reserve = _segment_cells((r, c1 - 2), (r, c2 + 2))
        block_dirs = [
            ((r, c1), (-1, 0)),
            ((r, c1), (1, 0)),
            ((r, c2), (-1, 0)),
            ((r, c2), (1, 0)),
        ]
    else:
        cols = list(range(w))
        if h < 9:
            return None
        c = random.choice(cols)
        choices = [(r1, r2) for r1 in range(2, h - 4) for r2 in range(r1 + 4, h - 2) if (r1 - r2) % 2 == 0]
        if not choices:
            return None
        r1, r2 = random.choice(choices)
        markers = [(r1, c), (r2, c)]
        reserve = _segment_cells((r1 - 2, c), (r2 + 2, c))
        block_dirs = [
            ((r1, c), (0, -1)),
            ((r1, c), (0, 1)),
            ((r2, c), (0, -1)),
            ((r2, c), (0, 1)),
        ]

    marker_block_reserve: set[tuple[int, int]] = set(reserve)
    for start, direction in block_dirs:
        marker_block_reserve.update(_cells_along_direction(shape, start, direction, max_steps=3))

    return {
        "layout": layout,
        "markers": markers,
        "reserve": reserve,
        "marker_block_reserve": marker_block_reserve,
        "block_dirs": block_dirs,
    }


def _marker_candidates(
    shape: tuple[int, int],
    used: set[tuple[int, int]],
    reserve_for_markers: set[tuple[int, int]],
    markers: Sequence[tuple[int, int]],
    *,
    parity: int | None = None,
    prefer_edge: bool = False,
) -> list[tuple[int, int]]:
    h, w = shape
    out: list[tuple[int, int]] = []
    for r in range(h):
        for c in range(w):
            cell = (r, c)
            if cell in used or cell in reserve_for_markers:
                continue
            if parity is not None and (r + c) % 2 != parity:
                continue
            if any(_manhattan(cell, marker) < 3 for marker in markers):
                continue
            if prefer_edge and min(r, c, h - 1 - r, w - 1 - c) > 1:
                continue
            out.append(cell)
    return out


def _place_extra_markers(
    shape: tuple[int, int],
    base_markers: list[tuple[int, int]],
    total_markers: int,
    reserve_for_markers: set[tuple[int, int]],
) -> list[tuple[int, int]] | None:
    markers = list(base_markers)
    used = set(base_markers)
    anchor_parity = (base_markers[0][0] + base_markers[0][1]) % 2

    for extra_idx in range(total_markers - len(base_markers)):
        want_opposite = extra_idx == 0 and total_markers >= 3 and random.random() < 0.7
        want_edge = extra_idx == total_markers - len(base_markers) - 1 and random.random() < 0.5
        preferred_parity = 1 - anchor_parity if want_opposite else None

        candidates = _marker_candidates(
            shape,
            used,
            reserve_for_markers,
            markers,
            parity=preferred_parity,
            prefer_edge=want_edge,
        )
        if not candidates and preferred_parity is not None:
            candidates = _marker_candidates(
                shape,
                used,
                reserve_for_markers,
                markers,
                parity=None,
                prefer_edge=want_edge,
            )
        if not candidates and want_edge:
            candidates = _marker_candidates(
                shape,
                used,
                reserve_for_markers,
                markers,
                parity=preferred_parity,
                prefer_edge=False,
            )
        if not candidates:
            return None
        cell = random.choice(candidates)
        markers.append(cell)
        used.add(cell)

    return markers


def _can_place_isolated_obstacle(
    cell: tuple[int, int],
    blocked: set[tuple[int, int]],
    obstacles: set[tuple[int, int]],
) -> bool:
    if cell in blocked or cell in obstacles:
        return False
    return all(_chebyshev(cell, other) > 1 for other in obstacles)


def _place_deliberate_blockers(
    shape: tuple[int, int],
    blocked: set[tuple[int, int]],
    block_dirs: Sequence[tuple[tuple[int, int], tuple[int, int]]],
) -> set[tuple[int, int]] | None:
    obstacles: set[tuple[int, int]] = set()
    options = list(block_dirs)
    random.shuffle(options)

    for start, direction in options:
        ray = _cells_along_direction(shape, start, direction, max_steps=3)
        random.shuffle(ray)
        for cell in ray:
            if _can_place_isolated_obstacle(cell, blocked, obstacles):
                obstacles.add(cell)
                return obstacles
    return None


def _place_random_obstacles(
    shape: tuple[int, int],
    blocked: set[tuple[int, int]],
    fixed_obstacles: set[tuple[int, int]],
    target_count: int,
) -> set[tuple[int, int]] | None:
    obstacles = set(fixed_obstacles)
    h, w = shape
    candidates = [(r, c) for r in range(h) for c in range(w)]
    random.shuffle(candidates)
    for cell in candidates:
        if len(obstacles) >= target_count:
            break
        if _can_place_isolated_obstacle(cell, blocked, obstacles):
            obstacles.add(cell)
    return obstacles if len(obstacles) == target_count else None


def _build_grid(
    shape: tuple[int, int],
    markers: Sequence[tuple[int, int]],
    colors: Sequence[int],
    obstacles: set[tuple[int, int]],
) -> np.ndarray:
    grid = np.full(shape, BACKGROUND, dtype=int)
    for r, c in obstacles:
        grid[r, c] = OBSTACLE
    for (r, c), color in zip(markers, colors):
        grid[r, c] = int(color)
    return grid


# --------------- semantic validation ---------------


def _analyze_grid(grid: np.ndarray) -> dict[str, object]:
    markers = _iter_markers(grid)
    marker_cells = [(r, c) for r, c, _color in markers]
    claims: dict[tuple[int, int], set[int]] = {}
    mixed_rays = 0
    blocked_rays = 0

    for marker_idx, (r, c, _color) in enumerate(markers):
        parity = (r + c) % 2
        for direction in DIRS4:
            saw_match = False
            saw_mismatch = False
            rr, cc = r + direction[0], c + direction[1]
            while 0 <= rr < grid.shape[0] and 0 <= cc < grid.shape[1]:
                if int(grid[rr, cc]) == OBSTACLE:
                    blocked_rays += 1
                    break
                if int(grid[rr, cc]) == BACKGROUND:
                    if (rr + cc) % 2 == parity:
                        saw_match = True
                        claims.setdefault((rr, cc), set()).add(marker_idx)
                    else:
                        saw_mismatch = True
                rr += direction[0]
                cc += direction[1]
            if saw_match and saw_mismatch:
                mixed_rays += 1

    unique_by_marker: list[set[tuple[int, int]]] = [set() for _ in markers]
    conflict_cells: set[tuple[int, int]] = set()
    for cell, owners in claims.items():
        if len(owners) == 1:
            unique_by_marker[next(iter(owners))].add(cell)
        else:
            conflict_cells.add(cell)

    return {
        "markers": markers,
        "marker_cells": marker_cells,
        "claims": claims,
        "unique_by_marker": unique_by_marker,
        "conflict_cells": conflict_cells,
        "mixed_rays": mixed_rays,
        "blocked_rays": blocked_rays,
    }


def _validate_generated_case(grid: np.ndarray) -> None:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert max(g.shape) <= 30

    markers = _iter_markers(g)
    obstacles = [(int(r), int(c)) for r, c in np.argwhere(g == OBSTACLE)]

    assert 2 <= len(markers) <= 4, "marker count must be 2..4"
    assert 8 <= len(obstacles) <= 15, "obstacle count must be 8..15"
    colors = [color for _r, _c, color in markers]
    assert len(colors) == len(set(colors)), "marker colors must be distinct"
    assert all(color in MARKER_COLORS for color in colors)

    for i, (r1, c1, _color1) in enumerate(markers):
        for r2, c2, _color2 in markers[i + 1 :]:
            assert _manhattan((r1, c1), (r2, c2)) >= 3, "markers too close"

    for i, obstacle in enumerate(obstacles):
        for other in obstacles[i + 1 :]:
            assert _chebyshev(obstacle, other) > 1, "obstacles must stay isolated"

    analysis = _analyze_grid(g)
    assert analysis["conflict_cells"], "need at least one conflict cell"
    assert int(analysis["mixed_rays"]) >= 1, "need a visible parity-filtered ray"
    assert int(analysis["blocked_rays"]) >= 1, "at least one ray should hit an obstacle"

    unique_by_marker: list[set[tuple[int, int]]] = analysis["unique_by_marker"]  # type: ignore[assignment]
    visible_markers = sum(len(cells) >= 2 for cells in unique_by_marker)
    assert visible_markers >= 2, "at least two markers should have visible unique output"

    out = main(g)
    added = (g == BACKGROUND) & (out != BACKGROUND)
    added_count = int(added.sum())
    assert 8 <= added_count <= max(24, g.size // 3), "output should stay readable"
    assert np.all(out[g == OBSTACLE] == OBSTACLE)
    assert np.all(out[(g != BACKGROUND) & (g != OBSTACLE)] == g[(g != BACKGROUND) & (g != OBSTACLE)])

    conflict_cells: set[tuple[int, int]] = analysis["conflict_cells"]  # type: ignore[assignment]
    for cell in conflict_cells:
        assert int(out[cell]) == BACKGROUND, "conflicts must reset to background"

    marker_info: list[tuple[int, int, int]] = analysis["markers"]  # type: ignore[assignment]
    for marker_idx, cells in enumerate(unique_by_marker):
        color = marker_info[marker_idx][2]
        for cell in cells:
            assert int(out[cell]) == color, "unique claims must keep the source color"


def _generate_case() -> np.ndarray:
    height = random.randint(*GEN_HEIGHT_RANGE)
    width = random.randint(*GEN_WIDTH_RANGE)
    shape = (height, width)

    layout_info = _choose_layout(shape)
    if layout_info is None:
        raise RuntimeError("failed to choose layout")

    total_markers = random.choice(GEN_MARKER_COUNTS)
    base_markers: list[tuple[int, int]] = list(layout_info["markers"])  # type: ignore[arg-type]
    reserve_for_markers: set[tuple[int, int]] = set(layout_info["marker_block_reserve"])  # type: ignore[arg-type]
    markers = _place_extra_markers(shape, base_markers, total_markers, reserve_for_markers)
    if markers is None:
        raise RuntimeError("failed to place markers")

    blocked_for_obstacles = set(markers) | set(layout_info["reserve"])  # type: ignore[arg-type]
    blockers = _place_deliberate_blockers(
        shape,
        blocked_for_obstacles,
        layout_info["block_dirs"],  # type: ignore[arg-type]
    )
    if blockers is None:
        raise RuntimeError("failed to place deliberate blocker")

    blocked_for_obstacles |= blockers
    obstacle_count = random.randint(*GEN_OBSTACLE_COUNT_RANGE)
    obstacles = _place_random_obstacles(shape, blocked_for_obstacles, blockers, obstacle_count)
    if obstacles is None:
        raise RuntimeError("failed to place random obstacles")

    colors = random.sample(MARKER_COLORS, k=len(markers))
    grid = _build_grid(shape, markers, colors, obstacles)
    _validate_generated_case(grid)
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        try:
            return _generate_case()
        except (AssertionError, RuntimeError, ValueError):
            continue
    raise RuntimeError("failed to generate a valid ray-parity puzzle input")
