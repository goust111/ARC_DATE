import random
from itertools import combinations

import numpy as np


BG = 0
FILL_COLORS = (1, 2, 3)
MARKER = 4
ROUTE = 5
TILE = 6
COARSE = 3
GRID = TILE * COARSE
MAX_ATTEMPTS = 400


def main(grid: np.ndarray) -> np.ndarray:
    """
    Compress the tile mosaic to route between marker tiles, then restore the route as tile-wide channels.

    Input structure:
    - The grid is an 18x18 black-background canvas partitioned into a 3x3 array of 6x6 tiles.
    - Each tile is either entirely black or entirely filled with one color from blue/red/green.
    - Two or three filled tiles contain a single yellow marker pixel at tile-local position (2,2); the rest of each such tile keeps its fill color.

    Rule:
    - Compress the 3x3 tile array to a 3x3 summary: black for empty tiles, the tile fill color for ordinary filled tiles, and yellow for marker tiles.
    - Read the yellow summary pixels in reading order and, for each consecutive pair, route from the first to the second by moving horizontally to the target column and then vertically to the target row.
    - While routing, recolor only black summary cells to grey; colored cells and yellow markers are left unchanged even if the route passes through them.
    - Upscale the routed 3x3 summary back to 18x18 by replacing each summary cell with a 6x6 block of the same color, then restore each marker tile so only its center pixel stays yellow and the rest of that tile returns to its original fill color.

    Key insight:
    - The real path is defined only after collapsing each 6x6 tile to one coarse pixel, so a one-pixel route in the summary becomes a full-tile-wide corridor when expanded back.
    """
    g = np.asarray(grid, dtype=int)
    summary, base_colors = _compress_grid(g)
    routed = _route_summary(summary)
    out = np.kron(routed, np.ones((TILE, TILE), dtype=int))

    for r, c in np.argwhere(summary == MARKER):
        tr = int(r) * TILE
        tc = int(c) * TILE
        out[tr : tr + TILE, tc : tc + TILE] = int(base_colors[r, c])
        out[tr + 2, tc + 2] = MARKER
    return out


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_ATTEMPTS):
        branch = "blocked" if rng.random() < 0.5 else "clean"
        marker_count = 2 if rng.random() < 0.5 else 3
        base_colors, markers = _generate_coarse_case(rng, marker_count, branch)
        grid = _expand_input(base_colors, markers)
        if _validate_generated_case(grid, branch):
            return grid
    raise RuntimeError("failed to generate a valid input")


def _compress_grid(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    assert grid.shape == (GRID, GRID), f"expected {GRID}x{GRID} grid"
    summary = np.zeros((COARSE, COARSE), dtype=int)
    base_colors = np.zeros((COARSE, COARSE), dtype=int)
    center_mask = np.ones((TILE, TILE), dtype=bool)
    center_mask[2, 2] = False

    for tr in range(COARSE):
        for tc in range(COARSE):
            tile = grid[tr * TILE : (tr + 1) * TILE, tc * TILE : (tc + 1) * TILE]
            center = int(tile[2, 2])
            body_values = set(map(int, tile[center_mask]))

            if center == MARKER:
                assert len(body_values) == 1, "marker tile must be otherwise uniform"
                fill = next(iter(body_values))
                assert fill in FILL_COLORS, "marker tile must sit on a filled tile"
                summary[tr, tc] = MARKER
                base_colors[tr, tc] = fill
                continue

            values = set(map(int, tile.flatten()))
            assert MARKER not in values, "only marker tiles may contain yellow"
            if values == {BG}:
                continue

            assert len(values) == 1, "non-marker tiles must be uniform"
            fill = next(iter(values))
            assert fill in FILL_COLORS, "unexpected tile color"
            summary[tr, tc] = fill
            base_colors[tr, tc] = fill

    markers = np.argwhere(summary == MARKER)
    assert 2 <= len(markers) <= 3, "expected two or three marker tiles"
    return summary, base_colors


def _route_summary(summary: np.ndarray) -> np.ndarray:
    routed = summary.copy()
    markers = [tuple(map(int, rc)) for rc in np.argwhere(summary == MARKER)]
    for start, end in zip(markers, markers[1:]):
        for cell in _segment_cells((start[0], start[1]), (start[0], end[1])):
            if routed[cell] == BG:
                routed[cell] = ROUTE
        for cell in _segment_cells((start[0], end[1]), (end[0], end[1])):
            if routed[cell] == BG:
                routed[cell] = ROUTE
    return routed


def _segment_cells(start: tuple[int, int], end: tuple[int, int]) -> list[tuple[int, int]]:
    sr, sc = start
    er, ec = end
    if sr == er:
        step = 1 if ec >= sc else -1
        return [(sr, c) for c in range(sc, ec + step, step)]
    assert sc == ec
    step = 1 if er >= sr else -1
    return [(r, sc) for r in range(sr, er + step, step)]


def _generate_coarse_case(
    rng: random.Random, marker_count: int, branch: str
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    marker_sets = _valid_marker_sets(marker_count, require_blockable=(branch == "blocked"))
    for _ in range(MAX_ATTEMPTS):
        markers = list(rng.choice(marker_sets))
        base = np.zeros((COARSE, COARSE), dtype=int)
        for pos in markers:
            base[pos] = rng.choice(FILL_COLORS)

        reserved_black = _route_cells(markers)
        obstacle = None
        if branch == "blocked":
            obstacle = rng.choice(_blockable_cells(markers))
            base[obstacle] = rng.choice(FILL_COLORS)
            reserved_black.discard(obstacle)

        free_cells = [
            (r, c)
            for r in range(COARSE)
            for c in range(COARSE)
            if (r, c) not in markers and (r, c) not in reserved_black and (r, c) != obstacle
        ]
        rng.shuffle(free_cells)
        extra_count = min(len(free_cells), 1 + (rng.random() < 0.65) + (rng.random() < 0.25))
        for r, c in free_cells[:extra_count]:
            base[r, c] = rng.choice(FILL_COLORS)

        summary = base.copy()
        for r, c in markers:
            summary[r, c] = MARKER

        stats = _route_stats(summary)
        off_route_fills = [
            (r, c)
            for r in range(COARSE)
            for c in range(COARSE)
            if base[r, c] in FILL_COLORS and (r, c) not in markers
        ]
        if not off_route_fills:
            continue
        if stats["painted"] < 2:
            continue
        if branch == "clean" and stats["obstacles"] == 0:
            return base, markers
        if branch == "blocked" and stats["continuations"] >= 1:
            return base, markers
    raise RuntimeError("failed to construct a coarse case")


def _valid_marker_sets(marker_count: int, require_blockable: bool) -> list[tuple[tuple[int, int], ...]]:
    cells = [(r, c) for r in range(COARSE) for c in range(COARSE)]
    valid: list[tuple[tuple[int, int], ...]] = []
    for combo in combinations(cells, marker_count):
        markers = tuple(sorted(combo))
        if not _markers_need_both_axes(markers):
            continue
        if require_blockable and not _blockable_cells(markers):
            continue
        if len(_route_cells(list(markers))) >= 7:
            continue
        valid.append(markers)
    assert valid
    return valid


def _markers_need_both_axes(markers: tuple[tuple[int, int], ...]) -> bool:
    for (r1, c1), (r2, c2) in zip(markers, markers[1:]):
        if r1 == r2 or c1 == c2:
            return False
    return True


def _route_cells(markers: list[tuple[int, int]] | tuple[tuple[int, int], ...]) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    for start, end in zip(markers, markers[1:]):
        cells.update(_segment_cells((start[0], start[1]), (start[0], end[1])))
        cells.update(_segment_cells((start[0], end[1]), (end[0], end[1])))
    return cells - set(markers)


def _blockable_cells(markers: list[tuple[int, int]] | tuple[tuple[int, int], ...]) -> list[tuple[int, int]]:
    candidates: list[tuple[int, int]] = []
    for (sr, sc), (er, ec) in zip(markers, markers[1:]):
        if sr != er and abs(ec - sc) == 2:
            candidates.append((sr, (sc + ec) // 2))
    return candidates


def _route_stats(summary: np.ndarray) -> dict[str, int]:
    original = summary.copy()
    routed = summary.copy()
    markers = [tuple(map(int, rc)) for rc in np.argwhere(summary == MARKER)]
    painted = 0
    obstacles = 0
    continuations = 0

    for start, end in zip(markers, markers[1:]):
        horizontal = _segment_cells((start[0], start[1]), (start[0], end[1]))
        vertical = _segment_cells((start[0], end[1]), (end[0], end[1]))

        for segment in (horizontal, vertical):
            blocked_here = False
            painted_after_block = False
            for cell in segment:
                if original[cell] == BG:
                    if routed[cell] == BG:
                        routed[cell] = ROUTE
                        painted += 1
                    if blocked_here:
                        painted_after_block = True
                elif cell != start and cell != end:
                    blocked_here = True
                    obstacles += 1
            if blocked_here and painted_after_block:
                continuations += 1

    return {"painted": painted, "obstacles": obstacles, "continuations": continuations}


def _expand_input(base_colors: np.ndarray, markers: list[tuple[int, int]]) -> np.ndarray:
    grid = np.kron(base_colors, np.ones((TILE, TILE), dtype=int))
    for r, c in markers:
        grid[r * TILE + 2, c * TILE + 2] = MARKER
    return grid


def _validate_generated_case(grid: np.ndarray, branch: str) -> bool:
    try:
        summary, base_colors = _compress_grid(grid)
        out = main(grid)
    except Exception:
        return False

    markers = [tuple(map(int, rc)) for rc in np.argwhere(summary == MARKER)]
    if len(markers) not in {2, 3}:
        return False
    if any(base_colors[pos] == BG for pos in markers):
        return False

    stats = _route_stats(summary)
    if stats["painted"] < 2:
        return False
    if branch == "clean" and stats["obstacles"] != 0:
        return False
    if branch == "blocked" and stats["continuations"] < 1:
        return False

    if np.array_equal(out, grid):
        return False
    if int(np.count_nonzero(out == ROUTE)) < TILE * TILE:
        return False

    for r, c in markers:
        tr = r * TILE
        tc = c * TILE
        tile = out[tr : tr + TILE, tc : tc + TILE]
        if int(tile[2, 2]) != MARKER:
            return False
        body = tile.copy()
        body[2, 2] = int(base_colors[r, c])
        if not np.all(body == int(base_colors[r, c])):
            return False

    return True
