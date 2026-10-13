from __future__ import annotations

import numpy as np

from arc_utils import Color, colors_present, find_components, get_bg, paste, random_free_location
import random


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each black component according to its size while preserving the grey background.

    Input structure:
    - The grid has a dominant non-black background color.
    - The only foreground pixels are black.
    - Each black 4-connected component has area 1, 2, or 3.

    Rule:
    - Preserve all background pixels.
    - Recolor area-1 black components green, area-2 components red, and area-3 components blue.

    Key insight:
    - The black cells are not independent pixels; each connected blob is classified by its 4-connected area.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = get_bg(g)
    assert bg != Color.BLACK, "background must be non-black"
    assert colors_present(g, exclude={bg}) == [Color.BLACK], "expected only black foreground"

    area_to_color = {
        1: Color.GREEN,
        2: Color.RED,
        3: Color.BLUE,
    }
    components = find_components(g, bg=bg, colors=[Color.BLACK], connectivity=4)
    assert components, "expected at least one black component"

    out = g.copy()
    for component in components:
        area = int(component["area"])
        assert area in area_to_color, "black component area must be 1, 2, or 3"
        coords = component["coords"]
        out[coords[:, 0], coords[:, 1]] = area_to_color[area]

    return out

# --------------- generator (o5x) ---------------

# Colors and solver oracle mapping.
_GEN_BG, _GEN_FG = Color.GREY, Color.BLACK
_GEN_AREA_TO_COLOR = {1: Color.GREEN, 2: Color.RED, 3: Color.BLUE}

# Canvas and placement constraints.
_GEN_GRID_SHAPE = (10, 10)
_GEN_PLACEMENT_PADDING, _GEN_CONNECTIVITY = 1, 4

# Component count and total foreground area bounds.
_GEN_COUNT_RANGES = (range(6, 9), range(5, 7), range(2, 5))
_GEN_COMPONENT_BOUNDS = (15, 16)
_GEN_BLACK_PIXEL_BOUNDS = (25, 28)
_GEN_MAX_ATTEMPTS = 300

_GEN_COUNT_OPTIONS = [
    (n1, n2, n3)
    for n1 in _GEN_COUNT_RANGES[0]
    for n2 in _GEN_COUNT_RANGES[1]
    for n3 in _GEN_COUNT_RANGES[2]
    if (
        _GEN_COMPONENT_BOUNDS[0] <= n1 + n2 + n3 <= _GEN_COMPONENT_BOUNDS[1]
        and _GEN_BLACK_PIXEL_BOUNDS[0] <= n1 + 2 * n2 + 3 * n3 <= _GEN_BLACK_PIXEL_BOUNDS[1]
    )
]
_GEN_DOT = ((0, 0),)

# Shape templates sampled for each component area.
_GEN_DOMINOES = (((0, 0), (0, 1)), ((0, 0), (1, 0)))
_GEN_TRIOMINOES = (
    ((0, 0), (0, 1), (0, 2)),
    ((0, 0), (1, 0), (2, 0)),
    ((0, 0), (1, 0), (1, 1)),
    ((0, 1), (1, 0), (1, 1)),
    ((0, 0), (0, 1), (1, 0)),
    ((0, 0), (0, 1), (1, 1)),
)


def _gen_patch(cells: tuple[tuple[int, int], ...]) -> np.ndarray:
    h = max(r for r, _ in cells) + 1
    w = max(c for _, c in cells) + 1
    patch = np.full((h, w), _GEN_BG, dtype=int)
    for r, c in cells:
        patch[r, c] = _GEN_FG
    return patch


def _gen_components(n1: int, n2: int, n3: int) -> list[tuple[int, np.ndarray]]:
    components = []
    components.extend((3, _gen_patch(random.choice(_GEN_TRIOMINOES))) for _ in range(n3))
    components.extend((2, _gen_patch(random.choice(_GEN_DOMINOES))) for _ in range(n2))
    components.extend((1, _gen_patch(_GEN_DOT)) for _ in range(n1))
    components.sort(key=lambda item: item[0], reverse=True)
    return components


def _gen_place_components(components: list[tuple[int, np.ndarray]]) -> np.ndarray | None:
    """Place every patch with no 4-adjacent black contact, returning None on packing failure."""
    grid = np.full(_GEN_GRID_SHAPE, _GEN_BG, dtype=int)
    for _, patch in components:
        try:
            r, c = random_free_location(
                grid, patch, transparent=_GEN_BG, padding=_GEN_PLACEMENT_PADDING,
                padding_connectivity=_GEN_CONNECTIVITY,
            )
        except ValueError:
            return None
        paste(grid, patch, r, c, transparent=_GEN_BG)
    return grid


def _gen_expected_output(grid: np.ndarray) -> np.ndarray | None:
    out = grid.copy()
    for comp in find_components(grid, bg=_GEN_BG, colors=[_GEN_FG], connectivity=_GEN_CONNECTIVITY):
        area = int(comp["area"])
        if area not in _GEN_AREA_TO_COLOR:
            return None
        coords = comp["coords"]
        out[coords[:, 0], coords[:, 1]] = _GEN_AREA_TO_COLOR[area]
    return out


def _gen_is_valid(grid: np.ndarray, counts: tuple[int, int, int]) -> bool:
    """Check sampled counts, foreground area bounds, and solver oracle agreement."""
    n1, n2, n3 = counts
    comps = find_components(grid, bg=_GEN_BG, colors=[_GEN_FG], connectivity=_GEN_CONNECTIVITY)
    areas = [int(comp["area"]) for comp in comps]
    if areas.count(1) != n1 or areas.count(2) != n2 or areas.count(3) != n3:
        return False
    if tuple(sorted(set(areas))) != tuple(_GEN_AREA_TO_COLOR):
        return False
    if not _GEN_BLACK_PIXEL_BOUNDS[0] <= int(np.sum(grid == _GEN_FG)) <= _GEN_BLACK_PIXEL_BOUNDS[1]:
        return False
    expected = _gen_expected_output(grid)
    return expected is not None and np.array_equal(main(grid), expected)


def _gen_attempt() -> np.ndarray | None:
    counts = random.choice(_GEN_COUNT_OPTIONS)
    grid = _gen_place_components(_gen_components(*counts))
    return grid if grid is not None and _gen_is_valid(grid, counts) else None


def generate_input() -> np.ndarray:
    for _ in range(_GEN_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid e8593010 input")
