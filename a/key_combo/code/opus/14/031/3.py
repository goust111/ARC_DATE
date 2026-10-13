from __future__ import annotations

from collections import Counter
import math
import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
ORANGE = 7

DOMINANT_COLORS = (BLUE, RED, GREEN, YELLOW)
GEN_ATTEMPTS = 400


def _find_cross(grid: np.ndarray) -> tuple[int, int]:
    rows = [r for r in range(grid.shape[0]) if np.all(grid[r, :] == GREY)]
    cols = [c for c in range(grid.shape[1]) if np.all(grid[:, c] == GREY)]
    assert len(rows) == 1 and len(cols) == 1, "expected exactly one solid grey row and one solid grey col"
    cross_r = int(rows[0])
    cross_c = int(cols[0])
    assert 0 < cross_r < grid.shape[0] - 1
    assert 0 < cross_c < grid.shape[1] - 1
    return cross_r, cross_c


def _quadrant_slices(shape: tuple[int, int], cross_r: int, cross_c: int) -> dict[str, tuple[slice, slice]]:
    h, w = map(int, shape)
    return {
        "tl": (slice(0, cross_r), slice(0, cross_c)),
        "tr": (slice(0, cross_r), slice(cross_c + 1, w)),
        "bl": (slice(cross_r + 1, h), slice(0, cross_c)),
        "br": (slice(cross_r + 1, h), slice(cross_c + 1, w)),
    }


def _quadrant_of_coord(r: int, c: int, cross_r: int, cross_c: int) -> str | None:
    if r == cross_r or c == cross_c:
        return None
    if r < cross_r and c < cross_c:
        return "tl"
    if r < cross_r and c > cross_c:
        return "tr"
    if r > cross_r and c < cross_c:
        return "bl"
    return "br"


def _dominant_color(region: np.ndarray) -> int:
    vals = region[(region != GREY) & (region != ORANGE)]
    assert vals.size > 0, "quadrant must contain non-separator non-marker pixels"
    counts = Counter(map(int, vals.reshape(-1)))
    ranked = counts.most_common()
    assert len(ranked) >= 2 and ranked[0][1] > ranked[1][1], "quadrant dominant color must be unique"
    return int(ranked[0][0])


def _quadrant_dominants(grid: np.ndarray, cross_r: int, cross_c: int) -> dict[str, int]:
    dominants: dict[str, int] = {}
    for name, src in _quadrant_slices(grid.shape, cross_r, cross_c).items():
        dominants[name] = _dominant_color(grid[src])
    assert set(dominants.values()) == set(DOMINANT_COLORS), "expected one unique dominant among 1,2,3,4 in each quadrant"
    return dominants


def _markers_by_quadrant(grid: np.ndarray, cross_r: int, cross_c: int) -> dict[str, tuple[int, int]]:
    locs = [tuple(map(int, rc)) for rc in np.argwhere(grid == ORANGE)]
    assert len(locs) == 4, "expected exactly four orange markers"
    assert len({r for r, _c in locs}) == 4, "marker rows must be pairwise distinct"
    assert len({c for _r, c in locs}) == 4, "marker cols must be pairwise distinct"

    markers: dict[str, tuple[int, int]] = {}
    for r, c in locs:
        quad = _quadrant_of_coord(r, c, cross_r, cross_c)
        assert quad is not None, "marker cannot be on the separator"
        assert quad not in markers, "expected exactly one marker per quadrant"
        markers[quad] = (r, c)
    assert set(markers) == {"tl", "tr", "bl", "br"}
    return markers


def _route_cells(start: tuple[int, int], end: tuple[int, int]) -> list[tuple[int, int]]:
    r0, c0 = start
    r1, c1 = end
    cells: list[tuple[int, int]] = []
    dc = 1 if c1 > c0 else -1
    for c in range(c0 + dc, c1 + dc, dc):
        cells.append((r0, c))
    dr = 1 if r1 > r0 else -1
    for r in range(r0 + dr, r1 + dr, dr):
        cells.append((r, c1))
    return cells


def _draw_leg(
    out: np.ndarray,
    start: tuple[int, int],
    end: tuple[int, int],
    cross_r: int,
    cross_c: int,
    dominants: dict[str, int],
) -> None:
    for r, c in _route_cells(start, end):
        if r == cross_r or c == cross_c:
            out[r, c] = GREY
            continue
        if out[r, c] != BLACK:
            continue
        quad = _quadrant_of_coord(int(r), int(c), cross_r, cross_c)
        assert quad is not None
        out[r, c] = int(dominants[quad])


def _expected_output(grid: np.ndarray) -> np.ndarray:
    cross_r, cross_c = _find_cross(grid)
    dominants = _quadrant_dominants(grid, cross_r, cross_c)
    markers = _markers_by_quadrant(grid, cross_r, cross_c)

    out = np.asarray(grid, dtype=int).copy()
    order = ("tl", "tr", "br", "bl", "tl")
    for a, b in zip(order, order[1:]):
        _draw_leg(out, markers[a], markers[b], cross_r, cross_c, dominants)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Denoise four cross-separated quadrants and route a clockwise orthogonal loop whose color changes by quadrant.

    Input structure:
    - A black-background grid contains one solid grey row and one solid grey column forming a separator cross.
    - The cross splits the board into four nonempty quadrants; each quadrant has one dominant color among blue/red/green/yellow plus a few noise pixels.
    - Each quadrant contains exactly one orange marker pixel away from the separator.

    Rule:
    - Find the unique solid grey row and solid grey column.
    - In each quadrant, ignore grey separator cells and orange markers, then take the most frequent remaining color as that quadrant's main color.
    - Connect the markers in clockwise order TL -> TR -> BR -> BL -> TL using horizontal-first Manhattan routing.
    - Paint only black route cells; use grey on separator-line cells and otherwise color each painted cell by the main color of the quadrant containing that cell.

    Key insight:
    - The route geometry comes from the marker positions, but the visible route colors come from quadrant-wise denoising, so color changes occur exactly when the path crosses the separator cross.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    return _expected_output(g)


def _segment_visibility(
    inp: np.ndarray,
    out: np.ndarray,
    start: tuple[int, int],
    end: tuple[int, int],
    start_quad: str,
    end_quad: str,
    cross_r: int,
    cross_c: int,
    dominants: dict[str, int],
) -> bool:
    crossed = False
    start_visible = False
    end_visible = False
    for r, c in _route_cells(start, end):
        if r == cross_r or c == cross_c:
            crossed = True
            if out[r, c] != GREY:
                return False
            continue
        quad = _quadrant_of_coord(r, c, cross_r, cross_c)
        if quad == start_quad and inp[r, c] == BLACK and out[r, c] == dominants[start_quad]:
            start_visible = True
        if quad == end_quad and inp[r, c] == BLACK and out[r, c] == dominants[end_quad]:
            end_visible = True
    return crossed and start_visible and end_visible


def _validate_semantics(inp: np.ndarray) -> bool:
    try:
        cross_r, cross_c = _find_cross(inp)
        markers = _markers_by_quadrant(inp, cross_r, cross_c)
        dominants = _quadrant_dominants(inp, cross_r, cross_c)
        out = main(inp)
    except Exception:
        return False

    if inp.shape != out.shape or inp.ndim != 2 or max(inp.shape) > 30:
        return False

    order = ("tl", "tr", "br", "bl", "tl")
    for a, b in zip(order, order[1:]):
        if not _segment_visibility(inp, out, markers[a], markers[b], a, b, cross_r, cross_c, dominants):
            return False

    changed = (inp != out) & (inp == BLACK)
    if int(np.count_nonzero(changed)) < 8:
        return False

    quadrant_cells = _quadrant_slices(inp.shape, cross_r, cross_c)
    for quad, src in quadrant_cells.items():
        region = inp[src]
        flat = region.reshape(-1)
        marker_count = int(np.count_nonzero(flat == ORANGE))
        if marker_count != 1:
            return False
        non_separator_non_marker = int(flat.size - marker_count)
        dominant = dominants[quad]
        dominant_count = int(np.count_nonzero(flat == dominant))
        if dominant_count < math.ceil(0.70 * non_separator_non_marker):
            return False
        other_payload = [c for c in DOMINANT_COLORS if c != dominant]
        noise_count = int(np.count_nonzero(np.isin(flat, other_payload)))
        if noise_count not in {2, 3}:
            return False

    return True


def _path_black_cells(
    markers: dict[str, tuple[int, int]],
    cross_r: int,
    cross_c: int,
) -> dict[str, set[tuple[int, int]]]:
    reserved = {quad: set() for quad in ("tl", "tr", "bl", "br")}
    order = ("tl", "tr", "br", "bl", "tl")
    marker_cells = set(markers.values())
    for a, b in zip(order, order[1:]):
        for r, c in _route_cells(markers[a], markers[b]):
            quad = _quadrant_of_coord(r, c, cross_r, cross_c)
            if quad is None or (r, c) in marker_cells:
                continue
            reserved[quad].add((r, c))
    return reserved


def _random_marker_layout(
    top_h: int,
    bottom_h: int,
    left_w: int,
    right_w: int,
    cross_r: int,
    cross_c: int,
) -> dict[str, tuple[int, int]]:
    top_rows = random.sample(list(range(1, top_h - 1)), 2)
    bottom_rows = random.sample(list(range(cross_r + 2, cross_r + bottom_h)), 2)
    left_cols = random.sample(list(range(1, left_w - 1)), 2)
    right_cols = random.sample(list(range(cross_c + 2, cross_c + right_w)), 2)
    return {
        "tl": (top_rows[0], left_cols[0]),
        "tr": (top_rows[1], right_cols[0]),
        "bl": (bottom_rows[0], left_cols[1]),
        "br": (bottom_rows[1], right_cols[1]),
    }


def _attempt_generate() -> np.ndarray | None:
    top_h = random.randint(7, 10)
    bottom_h = random.randint(7, 10)
    left_w = random.randint(7, 10)
    right_w = random.randint(7, 10)
    h = top_h + bottom_h + 1
    w = left_w + right_w + 1
    cross_r = top_h
    cross_c = left_w

    dominants = dict(zip(("tl", "tr", "bl", "br"), random.sample(DOMINANT_COLORS, 4)))
    markers = _random_marker_layout(top_h, bottom_h, left_w, right_w, cross_r, cross_c)
    reserved_black = _path_black_cells(markers, cross_r, cross_c)

    grid = np.full((h, w), BLACK, dtype=int)
    grid[cross_r, :] = GREY
    grid[:, cross_c] = GREY

    slices = _quadrant_slices(grid.shape, cross_r, cross_c)
    for quad, src in slices.items():
        region_coords = [(r, c) for r in range(src[0].start, src[0].stop) for c in range(src[1].start, src[1].stop)]
        marker = markers[quad]
        path_black = reserved_black[quad]
        free_cells = [cell for cell in region_coords if cell != marker and cell not in path_black]
        if len(free_cells) < 5:
            return None

        total_non_marker = len(region_coords) - 1
        noise_count = random.randint(2, 3)
        max_nondominant = math.floor(0.30 * total_non_marker)
        extra_black_budget = max_nondominant - len(path_black) - noise_count
        if extra_black_budget < 0:
            return None

        extra_black_count = random.randint(0, min(2, extra_black_budget))
        sampled = random.sample(free_cells, noise_count + extra_black_count)
        noise_cells = sampled[:noise_count]
        extra_black_cells = set(sampled[noise_count:])

        other_colors = [color for color in DOMINANT_COLORS if color != dominants[quad]]
        noise_colors = [random.choice(other_colors) for _ in range(noise_count)]

        for r, c in region_coords:
            if (r, c) == marker:
                grid[r, c] = ORANGE
            elif (r, c) in path_black or (r, c) in extra_black_cells:
                grid[r, c] = BLACK
            else:
                grid[r, c] = dominants[quad]

        for (r, c), color in zip(noise_cells, noise_colors):
            grid[r, c] = color

    if not _validate_semantics(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 3")
