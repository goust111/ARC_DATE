from __future__ import annotations

import numpy as np

from arc_utils import Color, bbox_coords, components_mask


def _paint_expanded_border(out: np.ndarray, bbox: tuple[int, int, int, int]) -> None:
    h, w = out.shape
    r0, c0, r1, c1 = bbox
    er0 = max(0, r0 - 1)
    ec0 = max(0, c0 - 1)
    er1 = min(h - 1, r1 + 1)
    ec1 = min(w - 1, c1 + 1)
    assert er0 <= er1 and ec0 <= ec1

    for r in range(er0, er1 + 1):
        for c in range(ec0, ec1 + 1):
            if r in (er0, er1) or c in (ec0, ec1):
                if out[r, c] == Color.BLACK:
                    out[r, c] = Color.GREEN


def main(grid: np.ndarray) -> np.ndarray:
    """
    Draw a green one-cell border around every adjacent red pair while leaving isolated red pixels unchanged.

    Input structure
    - The grid is a black background containing only red foreground pixels.
    - Each 4-connected red object is either a single pixel or a two-pixel horizontal/vertical domino.

    Rule
    - Preserve all input pixels.
    - For each red domino, expand its bounding box by one cell in all directions, clipped to the grid.
    - Paint the perimeter of that expanded box green wherever it is currently black.

    Key insight
    - The green marks are not drawn around all red pixels, only around the 4-connected red objects of area two.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert set(np.unique(g)).issubset({Color.BLACK, Color.RED})

    red_objects = components_mask(g == Color.RED, connectivity=4)
    assert red_objects

    out = g.copy()
    domino_count = 0
    for coords in red_objects:
        area = int(coords.shape[0])
        if area == 1:
            continue
        assert area == 2, "red objects must be isolated pixels or dominoes"
        r0, c0, r1, c1 = bbox_coords(coords)
        assert (r1 - r0) + (c1 - c0) == 1, "domino must be edge-connected"
        _paint_expanded_border(out, (r0, c0, r1, c1))
        domino_count += 1

    assert domino_count > 0
    return out

# --------------- generator (o5x) ---------------

# Canvas/object bounds; rng high values are exclusive.
GEN_ROW_RANGE = (15, 18)
GEN_COL_RANGE = (16, 19)
GEN_DOMINO_COUNT_RANGE = (1, 6)
GEN_SINGLE_COUNT_RANGE = (8, 19)

# Placement colors, adjacency, and retry budget.
GEN_BACKGROUND, GEN_FOREGROUND, GEN_FRAME = Color.BLACK, Color.RED, Color.GREEN
GEN_NEIGHBOR4_DELTAS = ((-1, 0), (1, 0), (0, -1), (0, 1))
GEN_ORIENTATION_COUNT = 2
GEN_MAX_ATTEMPTS = 300


def _gen_neighbors4(r: int, c: int, h: int, w: int):
    for dr, dc in GEN_NEIGHBOR4_DELTAS:
        nr, nc = r + dr, c + dc
        if 0 <= nr < h and 0 <= nc < w:
            yield nr, nc


def _gen_object_fits(grid: np.ndarray, coords: tuple[tuple[int, int], ...]) -> bool:
    h, w = grid.shape
    coord_set = set(coords)
    for r, c in coords:
        if grid[r, c] != GEN_BACKGROUND:
            return False
        for nr, nc in _gen_neighbors4(r, c, h, w):
            if (nr, nc) not in coord_set and grid[nr, nc] == GEN_FOREGROUND:
                return False
    return True


def _gen_domino_placements(grid: np.ndarray, horizontal: bool) -> list[tuple[tuple[int, int], ...]]:
    h, w = grid.shape
    placements = []
    col_stop = w - 1 if horizontal else w
    for r in range(h):
        for c in range(col_stop):
            coords = ((r, c), (r, c + 1)) if horizontal else ((r, c), (r + 1, c))
            if coords[-1][0] < h and _gen_object_fits(grid, coords):
                placements.append(coords)
    return placements


def _gen_single_locations(grid: np.ndarray) -> list[tuple[tuple[int, int], ...]]:
    h, w = grid.shape
    placements = []
    for r in range(h):
        for c in range(w):
            coords = ((r, c),)
            if _gen_object_fits(grid, coords):
                placements.append(coords)
    return placements


def _gen_paint_expected_border(out: np.ndarray, box: tuple[int, int, int, int]) -> None:
    h, w = out.shape
    r0, c0, r1, c1 = box
    er0 = max(0, r0 - 1)
    ec0 = max(0, c0 - 1)
    er1 = min(h - 1, r1 + 1)
    ec1 = min(w - 1, c1 + 1)
    for r in range(er0, er1 + 1):
        for c in range(ec0, ec1 + 1):
            if (r in (er0, er1) or c in (ec0, ec1)) and out[r, c] == GEN_BACKGROUND:
                out[r, c] = GEN_FRAME


def _gen_place_dominoes(rng, grid: np.ndarray, domino_count: int):
    """Place requested dominoes, returning their bounding boxes or None on crowding."""
    domino_boxes: list[tuple[int, int, int, int]] = []
    for _ in range(domino_count):
        horizontal = bool(rng.integers(0, GEN_ORIENTATION_COUNT))
        placements = _gen_domino_placements(grid, horizontal)
        if not placements:
            return None
        coords = placements[int(rng.integers(0, len(placements)))]
        for r, c in coords:
            grid[r, c] = GEN_FOREGROUND
        domino_boxes.append(bbox_coords(np.array(coords, dtype=int)))
    return domino_boxes


def _gen_place_singles(rng, grid: np.ndarray, single_count: int) -> bool:
    for _ in range(single_count):
        placements = _gen_single_locations(grid)
        if not placements:
            return False
        coords = placements[int(rng.integers(0, len(placements)))]
        r, c = coords[0]
        grid[r, c] = GEN_FOREGROUND
    return True


def _gen_matches_oracle(grid: np.ndarray, boxes: list[tuple[int, int, int, int]]) -> bool:
    """Check that the independent border renderer agrees with main()."""
    expected = grid.copy()
    for box in boxes:
        _gen_paint_expected_border(expected, box)
    try:
        actual = main(grid)
    except Exception:
        return False
    return np.array_equal(actual, expected)


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()
    h = int(rng.integers(*GEN_ROW_RANGE))
    w = int(rng.integers(*GEN_COL_RANGE))
    domino_count = int(rng.integers(*GEN_DOMINO_COUNT_RANGE))
    single_count = int(rng.integers(*GEN_SINGLE_COUNT_RANGE))
    grid = np.zeros((h, w), dtype=int)

    domino_boxes = _gen_place_dominoes(rng, grid, domino_count)
    if domino_boxes is None:
        return None
    if not _gen_place_singles(rng, grid, single_count):
        return None
    if not _gen_matches_oracle(grid, domino_boxes):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid b27ca6d3 input")
