from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import Color, find_components, get_bg


MAX_GENERATION_ATTEMPTS = 500
HEIGHT_RANGE = (8, 19)
WIDTH_RANGE = (8, 19)
LEGEND_LENGTH_RANGE = (3, 6)
MIN_POINT_COUNT = 4
MAX_POINT_COUNT = 14
PALETTE = tuple(Color.ALL_COLORS)


def _touches_border(bbox: tuple[int, int, int, int], shape: tuple[int, int]) -> bool:
    r0, c0, r1, c1 = bbox
    height, width = shape
    return r0 == 0 or c0 == 0 or r1 == height - 1 or c1 == width - 1


def _extract_legend_colors(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> tuple[str, list[int]]:
    r0, c0, r1, c1 = bbox
    height = r1 - r0 + 1
    width = c1 - c0 + 1
    if height == 1 and width >= 3:
        return "horizontal", [int(grid[r0, c]) for c in range(c0, c1 + 1)]
    if width == 1 and height >= 3:
        return "vertical", [int(grid[r, c0]) for r in range(r0, r1 + 1)]
    raise AssertionError("legend must be a straight bar")


def _solve_details(grid: np.ndarray) -> tuple[np.ndarray, dict[str, object]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = int(get_bg(g))
    components = find_components(g, bg=bg, connectivity=4, monochromatic=False)

    legend_candidates: list[tuple[int, str, list[int]]] = []
    for idx, obj in enumerate(components):
        area = int(obj["area"])
        if area < 3 or area > 5:
            continue

        bbox = obj["bbox"]
        r0, c0, r1, c1 = bbox
        height = r1 - r0 + 1
        width = c1 - c0 + 1
        if not ((height == 1) ^ (width == 1)):
            continue
        if area != max(height, width):
            continue
        if not _touches_border(bbox, g.shape):
            continue

        orientation, cycle = _extract_legend_colors(g, bbox)
        if len(set(cycle)) != len(cycle):
            continue
        if any(color == bg for color in cycle):
            continue
        legend_candidates.append((idx, orientation, cycle))

    assert len(legend_candidates) == 1, "expected exactly one legend bar"
    legend_idx, orientation, cycle = legend_candidates[0]

    points: list[tuple[int, int, int]] = []
    for idx, obj in enumerate(components):
        if idx == legend_idx:
            continue
        assert int(obj["area"]) == 1, "all non-legend foreground objects must be isolated pixels"
        r, c = map(int, obj["coords"][0])
        points.append((r, c, int(g[r, c])))

    assert points, "expected at least one isolated pixel"
    if orientation == "horizontal":
        points.sort(key=lambda item: (item[1], item[0]))
    else:
        points.sort(key=lambda item: (item[0], item[1]))

    start_color = points[0][2]
    assert start_color in cycle, "the first pixel color must anchor the legend phase"
    phase = cycle.index(start_color)

    out = g.copy()
    for r, c in components[legend_idx]["coords"]:
        out[int(r), int(c)] = bg
    for offset, (r, c, _color) in enumerate(points):
        out[r, c] = cycle[(phase + offset) % len(cycle)]

    details: dict[str, object] = {
        "bg": bg,
        "cycle": cycle,
        "orientation": orientation,
        "points": points,
        "phase": phase,
        "legend_coords": np.asarray(components[legend_idx]["coords"], dtype=int),
    }
    return out, details


def main(input_grid: np.ndarray) -> np.ndarray:
    """
    Use the edge-attached color bar as a cyclic legend.

    Horizontal bars encode a left-to-right color cycle and trigger column-major
    scanning of isolated pixels. Vertical bars encode a top-to-bottom cycle and
    trigger row-major scanning. The first isolated pixel's original color sets
    the starting phase of the cycle. The legend bar is erased in the output.
    """
    return _solve_details(np.asarray(input_grid, dtype=int))[0]


def _reserve_neighbors(blocked: np.ndarray, cells: list[tuple[int, int]]) -> None:
    height, width = blocked.shape
    for r, c in cells:
        for rr in range(max(0, r - 1), min(height, r + 2)):
            for cc in range(max(0, c - 1), min(width, c + 2)):
                blocked[rr, cc] = True


def _place_legend(
    grid: np.ndarray,
    blocked: np.ndarray,
    rng: np.random.Generator,
    legend_colors: list[int],
    orientation: str,
) -> None:
    height, width = grid.shape
    length = len(legend_colors)
    cells: list[tuple[int, int]] = []

    if orientation == "horizontal":
        row = 0 if bool(rng.integers(0, 2)) else height - 1
        start = int(rng.integers(0, width - length + 1))
        for offset, color in enumerate(legend_colors):
            c = start + offset
            grid[row, c] = int(color)
            cells.append((row, c))
    else:
        col = 0 if bool(rng.integers(0, 2)) else width - 1
        start = int(rng.integers(0, height - length + 1))
        for offset, color in enumerate(legend_colors):
            r = start + offset
            grid[r, col] = int(color)
            cells.append((r, col))

    _reserve_neighbors(blocked, cells)


def _available_cells(grid: np.ndarray, blocked: np.ndarray, bg: int) -> list[tuple[int, int]]:
    rows, cols = np.where((grid == int(bg)) & (~blocked))
    return [(int(r), int(c)) for r, c in zip(rows.tolist(), cols.tolist())]


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    height = int(rng.integers(*HEIGHT_RANGE))
    width = int(rng.integers(*WIDTH_RANGE))
    bg = int(rng.choice(PALETTE))
    usable_colors = [color for color in PALETTE if color != bg]

    legend_length = int(rng.integers(*LEGEND_LENGTH_RANGE))
    if (legend_length > width and legend_length > height) or len(usable_colors) < legend_length:
        return None

    orientation_options: list[str] = []
    if legend_length <= width:
        orientation_options.append("horizontal")
    if legend_length <= height:
        orientation_options.append("vertical")
    orientation = str(rng.choice(orientation_options))
    legend_colors = [int(color) for color in rng.choice(usable_colors, size=legend_length, replace=False).tolist()]

    grid = np.full((height, width), bg, dtype=int)
    blocked = np.zeros((height, width), dtype=bool)
    _place_legend(grid, blocked, rng, legend_colors, orientation)

    target_points = int(rng.integers(MIN_POINT_COUNT, MAX_POINT_COUNT + 1))
    points: list[tuple[int, int]] = []
    for _ in range(target_points):
        candidates = _available_cells(grid, blocked, bg)
        if not candidates:
            break
        r, c = candidates[int(rng.integers(0, len(candidates)))]
        points.append((r, c))
        grid[r, c] = int(rng.choice(usable_colors))
        _reserve_neighbors(blocked, [(r, c)])

    if len(points) < MIN_POINT_COUNT:
        return None

    if orientation == "horizontal":
        points.sort(key=lambda item: (item[1], item[0]))
    else:
        points.sort(key=lambda item: (item[0], item[1]))

    first_r, first_c = points[0]
    grid[first_r, first_c] = int(rng.choice(legend_colors))

    outside_legend = [color for color in usable_colors if color not in legend_colors]
    if outside_legend and len(points) > 1:
        chosen_idx = int(rng.integers(1, len(points)))
        rr, cc = points[chosen_idx]
        grid[rr, cc] = int(rng.choice(outside_legend))

    return grid


def _validate_generated(grid: np.ndarray) -> bool:
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2 or g.shape[0] > 30 or g.shape[1] > 30:
        return False

    try:
        out, details = _solve_details(g)
    except AssertionError:
        return False

    bg = int(details["bg"])
    cycle = [int(color) for color in details["cycle"]]
    orientation = str(details["orientation"])
    points = [(int(r), int(c), int(color)) for r, c, color in details["points"]]
    phase = int(details["phase"])
    legend_coords = np.asarray(details["legend_coords"], dtype=int)

    if len(points) < MIN_POINT_COUNT or len(cycle) < 3 or len(cycle) > 5:
        return False
    if len(set(cycle)) != len(cycle):
        return False
    if orientation not in {"horizontal", "vertical"}:
        return False
    if points[0][2] != cycle[phase]:
        return False

    for r, c in legend_coords:
        if out[int(r), int(c)] != bg:
            return False

    for offset, (r, c, _color) in enumerate(points):
        expected = cycle[(phase + offset) % len(cycle)]
        if int(out[r, c]) != int(expected):
            return False

    if out.ndim != 2 or out.shape != g.shape:
        return False
    return True


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        candidate = _gen_attempt(rng)
        if candidate is not None and _validate_generated(candidate):
            return candidate
    raise RuntimeError("failed to generate a valid input grid")
