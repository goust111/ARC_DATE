from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if _ARC_UTILS_DIR.exists():
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import Color, components_mask


BG = Color.BLACK
DIVIDER = Color.GREY
NEAR_COLOR = Color.PINK
FAR_COLOR = Color.BLUE
FRAGMENT_COLORS = {FAR_COLOR, NEAR_COLOR}

GRID_SIDE_RANGE = (13, 23)
QUADRANT_MIN_SIDE = 5
OUTER_MARGIN = 1
GENERATION_ATTEMPTS = 400

ORIENTATION_PATTERNS = [
    {"tl": "h", "tr": "v", "bl": "v", "br": "h"},
    {"tl": "v", "tr": "h", "bl": "h", "br": "v"},
]


def _find_dividers(grid: np.ndarray) -> tuple[int, int]:
    row_hits = [r for r in range(grid.shape[0]) if np.all(grid[r, :] == DIVIDER)]
    col_hits = [c for c in range(grid.shape[1]) if np.all(grid[:, c] == DIVIDER)]
    assert len(row_hits) == 1 and len(col_hits) == 1, "expected one full grey row and one full grey column"
    mid_r, mid_c = row_hits[0], col_hits[0]
    assert mid_r >= QUADRANT_MIN_SIDE and grid.shape[0] - mid_r - 1 >= QUADRANT_MIN_SIDE
    assert mid_c >= QUADRANT_MIN_SIDE and grid.shape[1] - mid_c - 1 >= QUADRANT_MIN_SIDE
    return mid_r, mid_c


def _quadrant_bounds(shape: tuple[int, int], mid_r: int, mid_c: int) -> dict[str, tuple[int, int, int, int]]:
    h, w = shape
    return {
        "tl": (0, mid_r - 1, 0, mid_c - 1),
        "tr": (0, mid_r - 1, mid_c + 1, w - 1),
        "bl": (mid_r + 1, h - 1, 0, mid_c - 1),
        "br": (mid_r + 1, h - 1, mid_c + 1, w - 1),
    }


def _view_direction(name: str) -> tuple[int, int]:
    if name == "tl":
        return 1, 1
    if name == "tr":
        return 1, -1
    if name == "bl":
        return -1, 1
    if name == "br":
        return -1, -1
    raise ValueError(f"unknown quadrant {name}")


def _fragment_coords(grid: np.ndarray, bounds: tuple[int, int, int, int]) -> np.ndarray:
    r0, r1, c0, c1 = bounds
    sub = grid[r0 : r1 + 1, c0 : c1 + 1]
    comps = components_mask(np.isin(sub, list(FRAGMENT_COLORS)), connectivity=4)
    assert len(comps) == 1, "each quadrant must contain exactly one fragment"
    coords = comps[0]
    assert coords.shape == (2, 2), "fragment must contain exactly two cells"
    abs_coords = coords.copy()
    abs_coords[:, 0] += r0
    abs_coords[:, 1] += c0
    colors = {int(grid[r, c]) for r, c in abs_coords}
    assert colors == FRAGMENT_COLORS, "fragment must contain one blue and one pink pixel"
    return abs_coords


def _ordered_fragment(coords: np.ndarray, direction: tuple[int, int]) -> tuple[str, list[tuple[int, int]]]:
    pts = [tuple(map(int, rc)) for rc in coords.tolist()]
    rows = {r for r, _ in pts}
    cols = {c for _, c in pts}
    dr, dc = direction

    if len(rows) == 1:
        reverse = dc < 0
        ordered = sorted(pts, key=lambda rc: rc[1], reverse=reverse)
        return "h", ordered
    if len(cols) == 1:
        reverse = dr < 0
        ordered = sorted(pts, key=lambda rc: rc[0], reverse=reverse)
        return "v", ordered
    raise AssertionError("fragment pixels must be 4-neighbor horizontal or vertical")


def _target_pairs(grid: np.ndarray) -> dict[str, tuple[str, tuple[int, int], tuple[int, int]]]:
    mid_r, mid_c = _find_dividers(grid)
    pairs: dict[str, tuple[str, tuple[int, int], tuple[int, int]]] = {}
    for name, bounds in _quadrant_bounds(grid.shape, mid_r, mid_c).items():
        coords = _fragment_coords(grid, bounds)
        orient, ordered = _ordered_fragment(coords, _view_direction(name))
        pairs[name] = (orient, ordered[0], ordered[1])
    return pairs


def _matches_target(grid: np.ndarray, name: str, near: tuple[int, int], far: tuple[int, int]) -> bool:
    return int(grid[near]) == NEAR_COLOR and int(grid[far]) == FAR_COLOR


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each two-pixel quadrant fragment so the pixel nearer the quadrant-side view becomes pink.

    Input structure:
    - A black grid is split by one full grey row and one full grey column into four quadrants.
    - Each quadrant contains exactly one 2-cell fragment made of one blue pixel and one pink pixel.
    - Every fragment is either horizontal or vertical and stays at least one cell away from the divider lines.

    Rule:
    - Use the grey cross as the center reference and determine each quadrant's viewing direction toward that center.
    - Project that direction onto the fragment axis: use the horizontal component for horizontal fragments and the vertical component for vertical fragments.
    - Along that projected direction, recolor the first encountered fragment pixel pink and the second encountered pixel blue.
    - Leave the grey divider and all black background cells unchanged.

    Key insight:
    - The viewing direction is not marked explicitly; it is encoded only by which quadrant the fragment occupies relative to the divider cross.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    out = g.copy()
    for _name, (_orient, near, far) in _target_pairs(g).items():
        out[near] = NEAR_COLOR
        out[far] = FAR_COLOR
    return out


def _sample_side(min_side: int = GRID_SIDE_RANGE[0], max_side: int = GRID_SIDE_RANGE[1]) -> int:
    return random.randint(min_side, max_side)


def _sample_divider(side: int) -> int:
    return random.randint(QUADRANT_MIN_SIDE, side - QUADRANT_MIN_SIDE - 1)


def _sample_fragment_start(
    bounds: tuple[int, int, int, int], mid_r: int, mid_c: int, name: str, orient: str
) -> tuple[int, int]:
    r0, r1, c0, c1 = bounds

    row_lo = r0 + OUTER_MARGIN
    row_hi = r1 - OUTER_MARGIN
    col_lo = c0 + OUTER_MARGIN
    col_hi = c1 - OUTER_MARGIN

    if name in {"tl", "tr"}:
        row_hi = min(row_hi, mid_r - 2)
    else:
        row_lo = max(row_lo, mid_r + 2)

    if name in {"tl", "bl"}:
        col_hi = min(col_hi, mid_c - 2)
    else:
        col_lo = max(col_lo, mid_c + 2)

    if orient == "h":
        assert col_hi - col_lo + 1 >= 2
        c = random.randint(col_lo, col_hi - 1)
        r = random.randint(row_lo, row_hi)
        return r, c

    assert row_hi - row_lo + 1 >= 2
    r = random.randint(row_lo, row_hi - 1)
    c = random.randint(col_lo, col_hi)
    return r, c


def _fragment_cells(start: tuple[int, int], orient: str) -> list[tuple[int, int]]:
    r, c = start
    if orient == "h":
        return [(r, c), (r, c + 1)]
    return [(r, c), (r + 1, c)]


def _paint_fragment(grid: np.ndarray, cells: list[tuple[int, int]], target_correct: bool, name: str) -> None:
    orient, ordered = _ordered_fragment(np.array(cells, dtype=int), _view_direction(name))
    del orient
    colors = [NEAR_COLOR, FAR_COLOR] if target_correct else [FAR_COLOR, NEAR_COLOR]
    for cell, color in zip(ordered, colors, strict=True):
        grid[cell] = color


def _nondegenerate_statuses() -> dict[str, bool]:
    names = ["tl", "tr", "bl", "br"]
    while True:
        vals = [random.choice([False, True]) for _ in names]
        if any(vals) and not all(vals):
            return dict(zip(names, vals, strict=True))


def _build_input() -> np.ndarray:
    h = _sample_side()
    w = _sample_side()
    mid_r = _sample_divider(h)
    mid_c = _sample_divider(w)

    grid = np.full((h, w), BG, dtype=int)
    grid[mid_r, :] = DIVIDER
    grid[:, mid_c] = DIVIDER

    bounds_by_name = _quadrant_bounds(grid.shape, mid_r, mid_c)
    pattern = random.choice(ORIENTATION_PATTERNS)
    statuses = _nondegenerate_statuses()

    for name, orient in pattern.items():
        start = _sample_fragment_start(bounds_by_name[name], mid_r, mid_c, name, orient)
        cells = _fragment_cells(start, orient)
        _paint_fragment(grid, cells, statuses[name], name)

    return grid


def _validate_structure(grid: np.ndarray) -> None:
    assert isinstance(grid, np.ndarray)
    assert grid.ndim == 2
    assert max(grid.shape) <= 30
    mid_r, mid_c = _find_dividers(grid)
    bounds_by_name = _quadrant_bounds(grid.shape, mid_r, mid_c)

    orientations = {}
    statuses = []
    for name, bounds in bounds_by_name.items():
        coords = _fragment_coords(grid, bounds)
        orient, ordered = _ordered_fragment(coords, _view_direction(name))
        orientations[name] = orient
        near, far = ordered
        statuses.append(_matches_target(grid, name, near, far))

        if orient == "h":
            assert abs(near[1] - mid_c) > abs(far[1] - mid_c), "horizontal near end must be farther from center"
        else:
            assert abs(near[0] - mid_r) > abs(far[0] - mid_r), "vertical near end must be farther from center"

    assert orientations["tl"] != orientations["tr"]
    assert orientations["tl"] != orientations["bl"]
    assert orientations["br"] != orientations["tr"]
    assert orientations["br"] != orientations["bl"]
    assert {orientations["tl"], orientations["tr"], orientations["bl"], orientations["br"]} == {"h", "v"}
    assert any(statuses) and not all(statuses), "generator must include both already-correct and incorrect fragments"


def _validate_semantics(grid: np.ndarray) -> bool:
    try:
        _validate_structure(grid)
        out = main(grid)
        pairs = _target_pairs(grid)
        for _name, (_orient, near, far) in pairs.items():
            if int(out[near]) != NEAR_COLOR or int(out[far]) != FAR_COLOR:
                return False
        if np.array_equal(out, grid):
            return False
        return True
    except Exception:
        return False


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _build_input()
        if _validate_semantics(grid):
            return grid
    raise RuntimeError("failed to generate a valid quadrant-view fragment input")
