from __future__ import annotations

import itertools
import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
YELLOW = 4
ORANGE = 7

PAIR_COLORS = (BLUE, RED, YELLOW)
COARSE_SIZE = 3
TILE_SIZE = 5
GRID_SIZE = COARSE_SIZE * TILE_SIZE


def _build_segments() -> list[dict[str, object]]:
    segments: list[dict[str, object]] = []
    for r in range(COARSE_SIZE):
        for c0, c1 in itertools.combinations(range(COARSE_SIZE), 2):
            path = tuple((r, c) for c in range(c0, c1 + 1))
            segments.append(
                {
                    "endpoints": ((r, c0), (r, c1)),
                    "path": path,
                    "has_gap": len(path) == 3,
                }
            )
    for c in range(COARSE_SIZE):
        for r0, r1 in itertools.combinations(range(COARSE_SIZE), 2):
            path = tuple((r, c) for r in range(r0, r1 + 1))
            segments.append(
                {
                    "endpoints": ((r0, c), (r1, c)),
                    "path": path,
                    "has_gap": len(path) == 3,
                }
            )
    return segments


SEGMENTS = _build_segments()


def _valid_segment_combos(n_colors: int) -> list[tuple[int, ...]]:
    combos: list[tuple[int, ...]] = []
    for combo in itertools.combinations(range(len(SEGMENTS)), n_colors):
        endpoints: set[tuple[int, int]] = set()
        used: set[tuple[int, int]] = set()
        has_gap = False
        ok = True
        for idx in combo:
            segment = SEGMENTS[idx]
            a, b = segment["endpoints"]
            path = segment["path"]
            if a in endpoints or b in endpoints:
                ok = False
                break
            if used & set(path):
                ok = False
                break
            endpoints.add(a)
            endpoints.add(b)
            used.update(path)
            has_gap = has_gap or bool(segment["has_gap"])
        if ok and has_gap and len(endpoints) < COARSE_SIZE * COARSE_SIZE:
            combos.append(combo)
    return combos


VALID_COMBOS = {2: _valid_segment_combos(2), 3: _valid_segment_combos(3)}


def _tile_bounds(tr: int, tc: int, tile_size: int) -> tuple[int, int, int, int]:
    r0 = tr * tile_size
    c0 = tc * tile_size
    return r0, c0, r0 + tile_size, c0 + tile_size


def _classify_tile(tile: np.ndarray) -> tuple[str, int]:
    unique = {int(v) for v in np.unique(tile)}
    if unique == {BLACK}:
        return "black", BLACK
    if len(unique) == 1:
        color = next(iter(unique))
        if color in PAIR_COLORS:
            return "solid", color
    center = tile.shape[0] // 2
    if (
        tile.shape[0] == tile.shape[1]
        and tile.shape[0] % 2 == 1
        and int(tile[center, center]) == ORANGE
    ):
        probe = tile.copy()
        probe[center, center] = BLACK
        if np.all(probe == BLACK):
            return "marker", ORANGE
    raise AssertionError("tile does not match any valid input role")


def _compress_input(grid: np.ndarray) -> tuple[np.ndarray, tuple[int, int], int]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = g.shape
    assert h == w and h % COARSE_SIZE == 0, "grid must be a square 3x3 tile mosaic"
    tile_size = h // COARSE_SIZE
    assert tile_size >= 3 and tile_size % 2 == 1, "tile size must be odd"

    abstract = np.zeros((COARSE_SIZE, COARSE_SIZE), dtype=int)
    marker_pos: tuple[int, int] | None = None
    for tr in range(COARSE_SIZE):
        for tc in range(COARSE_SIZE):
            r0, c0, r1, c1 = _tile_bounds(tr, tc, tile_size)
            kind, color = _classify_tile(g[r0:r1, c0:c1])
            if kind == "solid":
                abstract[tr, tc] = color
            elif kind == "marker":
                assert marker_pos is None, "expected exactly one marker tile"
                marker_pos = (tr, tc)

    assert marker_pos is not None, "expected exactly one orange marker"
    return abstract, marker_pos, tile_size


def _solve_abstract(abstract: np.ndarray) -> np.ndarray:
    out = abstract.copy()
    colors = [int(c) for c in np.unique(abstract) if int(c) != BLACK]
    assert len(colors) in (2, 3), "expected two or three endpoint colors"

    changed = False
    painted: dict[tuple[int, int], int] = {}
    for color in colors:
        coords = [tuple(map(int, rc)) for rc in np.argwhere(abstract == color)]
        assert len(coords) == 2, "each color must appear exactly twice"
        (r0, c0), (r1, c1) = sorted(coords)
        assert r0 == r1 or c0 == c1, "same-colored endpoints must align"

        if r0 == r1:
            path = [(r0, c) for c in range(min(c0, c1), max(c0, c1) + 1)]
        else:
            path = [(r, c0) for r in range(min(r0, r1), max(r0, r1) + 1)]

        changed = changed or len(path) == 3
        for cell in path:
            previous = painted.get(cell)
            assert previous in (None, color), "different color paths may not overlap"
            painted[cell] = color
            existing = int(out[cell])
            assert existing in (BLACK, color), "intermediate abstract cells must be black"
            out[cell] = color

    assert changed, "at least one pair must require a connection fill"
    return out


def _render_output(abstract: np.ndarray, marker_pos: tuple[int, int], tile_size: int) -> np.ndarray:
    out = np.zeros((COARSE_SIZE * tile_size, COARSE_SIZE * tile_size), dtype=int)
    for tr in range(COARSE_SIZE):
        for tc in range(COARSE_SIZE):
            color = int(abstract[tr, tc])
            if color == BLACK:
                continue
            r0, c0, r1, c1 = _tile_bounds(tr, tc, tile_size)
            out[r0:r1, c0:c1] = color

    mr, mc = marker_pos
    r0, c0, r1, c1 = _tile_bounds(mr, mc, tile_size)
    center = tile_size // 2
    out[r0 + center, c0 + center] = ORANGE
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Compress a tile mosaic into a 3x3 endpoint map, connect aligned same-color pairs, then expand it back.

    Input structure:
    - The grid is a black 3x3 mosaic of equal odd-sized square tiles; the intended task uses 5x5 tiles.
    - Some tiles are completely filled in blue, red, or yellow, and each used color appears exactly twice.
    - One additional tile is otherwise black except for a single orange pixel at that tile's center.

    Rule:
    - Compress each tile to one abstract cell: solid monochrome tiles become their fill color, and every other tile counts as black.
    - For every color pair in the abstract 3x3 grid, verify the two endpoints share a row or a column, then fill every abstract cell between them with that color.
    - Expand the connected abstract grid back to full tiles, painting each non-black abstract cell as a solid square block.
    - Restore the original orange center marker on top of the expanded result, even if its tile was filled by a connection.

    Key insight:
    - The colored endpoints only make sense after compressing the 5x5 tile lattice, because the actual connection happens in tile space rather than pixel space.
    """
    abstract, marker_pos, tile_size = _compress_input(grid)
    solved = _solve_abstract(abstract)
    return _render_output(solved, marker_pos, tile_size)


def _render_input_from_spec(
    assignments: list[tuple[int, dict[str, object]]],
    marker_pos: tuple[int, int],
) -> np.ndarray:
    grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
    for color, segment in assignments:
        for tr, tc in segment["endpoints"]:
            r0, c0, r1, c1 = _tile_bounds(tr, tc, TILE_SIZE)
            grid[r0:r1, c0:c1] = color

    mr, mc = marker_pos
    r0, c0, _, _ = _tile_bounds(mr, mc, TILE_SIZE)
    center = TILE_SIZE // 2
    grid[r0 + center, c0 + center] = ORANGE
    return grid


def _spec_to_output(
    assignments: list[tuple[int, dict[str, object]]],
    marker_pos: tuple[int, int],
) -> np.ndarray:
    abstract = np.zeros((COARSE_SIZE, COARSE_SIZE), dtype=int)
    for color, segment in assignments:
        for tr, tc in segment["endpoints"]:
            abstract[tr, tc] = color
    return _render_output(_solve_abstract(abstract), marker_pos, TILE_SIZE)


def _choose_marker(
    rng: random.Random,
    assignments: list[tuple[int, dict[str, object]]],
) -> tuple[int, int]:
    endpoints = {
        tuple(cell)
        for _color, segment in assignments
        for cell in segment["endpoints"]
    }
    gap_cells = [
        tuple(cell)
        for _color, segment in assignments
        for cell in segment["path"]
        if tuple(cell) not in endpoints
    ]
    empty_cells = [
        (r, c)
        for r in range(COARSE_SIZE)
        for c in range(COARSE_SIZE)
        if (r, c) not in endpoints and (r, c) not in gap_cells
    ]

    if gap_cells and (not empty_cells or rng.random() < 0.55):
        return rng.choice(gap_cells)
    assert empty_cells, "there must be a legal marker tile"
    return rng.choice(empty_cells)


def _generate_spec(rng: random.Random) -> tuple[list[tuple[int, dict[str, object]]], tuple[int, int]]:
    n_colors = rng.choices([2, 3], weights=[3, 2], k=1)[0]
    combo = rng.choice(VALID_COMBOS[n_colors])
    colors = rng.sample(PAIR_COLORS, n_colors)
    rng.shuffle(colors)
    assignments = [(color, SEGMENTS[idx]) for color, idx in zip(colors, combo)]
    marker_pos = _choose_marker(rng, assignments)
    return assignments, marker_pos


def _validate_generated_case(
    grid: np.ndarray,
    assignments: list[tuple[int, dict[str, object]]],
    marker_pos: tuple[int, int],
) -> bool:
    try:
        abstract, parsed_marker, tile_size = _compress_input(grid)
        if tile_size != TILE_SIZE or parsed_marker != marker_pos:
            return False

        expected_abstract = np.zeros((COARSE_SIZE, COARSE_SIZE), dtype=int)
        endpoint_cells: set[tuple[int, int]] = set()
        non_endpoint_path_cells: set[tuple[int, int]] = set()
        has_gap = False
        for color, segment in assignments:
            path = list(segment["path"])
            endpoints = {tuple(cell) for cell in segment["endpoints"]}
            if marker_pos in endpoints:
                return False
            endpoint_cells.update(endpoints)
            for cell in endpoints:
                expected_abstract[cell] = color
            if len(path) == 3:
                has_gap = True
                middle = tuple(path[1])
                if middle not in endpoints:
                    non_endpoint_path_cells.add(middle)
                if middle not in endpoints and middle != marker_pos and int(abstract[middle]) != BLACK:
                    return False

        if not np.array_equal(abstract, expected_abstract):
            return False

        if not has_gap:
            return False
        for r in range(COARSE_SIZE):
            for c in range(COARSE_SIZE):
                cell = (r, c)
                if cell not in endpoint_cells and cell != marker_pos and cell not in non_endpoint_path_cells:
                    if int(abstract[cell]) != BLACK:
                        return False

        expected_output = _spec_to_output(assignments, marker_pos)
        actual_output = main(grid)
        return np.array_equal(actual_output, expected_output)
    except Exception:
        return False


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(400):
        assignments, marker_pos = _generate_spec(rng)
        grid = _render_input_from_spec(assignments, marker_pos)
        if _validate_generated_case(grid, assignments, marker_pos):
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
