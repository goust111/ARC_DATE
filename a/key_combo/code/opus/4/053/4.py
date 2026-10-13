from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(_ARC_UTILS_DIR))

from arc_utils import find_components, get_bg, paste, random_free_location


BG = 7
GRID_SIZES = (11, 13, 15, 17)
CORNERS = ("top_left", "top_right", "bottom_left", "bottom_right")
CORNER_ORDER = {corner: idx for idx, corner in enumerate(CORNERS)}
CORNER_TO_LOCAL = {
    "top_left": frozenset({(0, 0), (0, 1), (1, 0)}),
    "top_right": frozenset({(0, 0), (0, 1), (1, 1)}),
    "bottom_left": frozenset({(0, 0), (1, 0), (1, 1)}),
    "bottom_right": frozenset({(0, 1), (1, 0), (1, 1)}),
}
LOCAL_TO_CORNER = {cells: corner for corner, cells in CORNER_TO_LOCAL.items()}
COLOR_POOL = (1, 2, 3, 4, 5, 6, 8, 9)
MAX_GENERATION_ATTEMPTS = 400


def _draw_corner_l(grid: np.ndarray, corner: str, color: int) -> None:
    arm = grid.shape[0] // 2
    n = grid.shape[0]
    if corner == "top_left":
        grid[0, :arm] = int(color)
        grid[:arm, 0] = int(color)
    elif corner == "top_right":
        grid[0, n - arm :] = int(color)
        grid[:arm, n - 1] = int(color)
    elif corner == "bottom_left":
        grid[n - 1, :arm] = int(color)
        grid[n - arm :, 0] = int(color)
    else:
        assert corner == "bottom_right"
        grid[n - 1, n - arm :] = int(color)
        grid[n - arm :, n - 1] = int(color)


def _marker_patch(corner: str, color: int) -> np.ndarray:
    patch = np.full((2, 2), BG, dtype=int)
    for r, c in CORNER_TO_LOCAL[corner]:
        patch[r, c] = int(color)
    return patch


def _legend_patch(colors: list[int], is_row: bool) -> np.ndarray:
    if is_row:
        return np.asarray([colors], dtype=int)
    return np.asarray(colors, dtype=int).reshape(-1, 1)


def _component_corner(component: dict) -> str:
    r0, c0, r1, c1 = component["bbox"]
    assert (int(r1 - r0), int(c1 - c0)) == (1, 1)
    local = frozenset((int(r) - int(r0), int(c) - int(c0)) for r, c in component["coords"])
    assert local in LOCAL_TO_CORNER
    return LOCAL_TO_CORNER[local]


def _extract_markers_and_legend(grid: np.ndarray) -> tuple[list[dict], list[int]]:
    bg = int(get_bg(grid))
    components = find_components(grid, bg=bg, connectivity=4, monochromatic=True)
    markers = []
    marker_cells: set[tuple[int, int]] = set()

    for comp in components:
        area = int(comp["area"])
        r0, c0, r1, c1 = comp["bbox"]
        if area == 3 and (int(r1 - r0), int(c1 - c0)) == (1, 1):
            _component_corner(comp)
            markers.append(comp)
            marker_cells.update((int(r), int(c)) for r, c in comp["coords"])

    assert 2 <= len(markers) <= 4
    corners = [_component_corner(comp) for comp in markers]
    assert len(set(corners)) == len(corners)

    legend_mask = (grid != bg).copy()
    for r, c in marker_cells:
        legend_mask[r, c] = False
    legend_coords = np.argwhere(legend_mask)
    assert legend_coords.shape[0] == len(markers)

    legend_components = find_components(
        np.where(legend_mask, grid, bg),
        bg=bg,
        connectivity=4,
        monochromatic=False,
    )
    assert len(legend_components) == 1
    legend = legend_components[0]
    r0, c0, r1, c1 = legend["bbox"]
    assert int(legend["area"]) == len(markers)
    assert (r0 == r1) or (c0 == c1)

    coords = [(int(r), int(c)) for r, c in legend["coords"]]
    if r0 == r1:
        coords.sort(key=lambda rc: rc[1])
    else:
        coords.sort(key=lambda rc: rc[0])
    legend_colors = [int(grid[r, c]) for r, c in coords]
    assert len(set(legend_colors)) == len(legend_colors)
    return markers, legend_colors


def _expected_from_spec(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    n, m = g.shape
    assert n == m and n % 2 == 1
    bg = int(get_bg(g))
    assert bg == BG

    markers, legend_colors = _extract_markers_and_legend(g)
    used_corners = sorted((_component_corner(comp) for comp in markers), key=lambda corner: CORNER_ORDER[corner])
    corner_to_color = {corner: int(legend_colors[idx]) for idx, corner in enumerate(used_corners)}

    out = np.full_like(g, bg)
    for corner in used_corners:
        _draw_corner_l(out, corner, corner_to_color[corner])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use each 3-pixel L marker's missing-corner orientation to choose a corner and paint that corner L with the legend color at the same corner-rank position.

    Input structure:
    - The grid is an odd square with orange background.
    - It contains 2 to 4 monochromatic 3-pixel L markers, each occupying a 2x2 box with one missing corner, and every marker points to a different outer corner.
    - It also contains one nearby central legend made of the same number of single pixels in one compact row or one compact column, with colors distinct from the marker colors.

    Rule:
    - Read each marker's missing corner to determine which outer corner it points to: top-left, top-right, bottom-left, or bottom-right.
    - Sort the used outer corners in reading order and sort the legend pixels in reading order along their row or column.
    - Match the first used corner to the first legend color, the second used corner to the second legend color, and so on.
    - Clear the original markers and legend, then draw an L from each matched outer corner along the bordering row and column with arm length (grid size - 1) / 2, using the matched legend color.

    Key insight:
    - A marker's orientation encodes both where to draw and which legend slot to use; marker position and marker color themselves are irrelevant.
    """
    return _expected_from_spec(grid)


def _valid_legend_positions(size: int, patch_shape: tuple[int, int]) -> list[tuple[int, int]]:
    ph, pw = patch_shape
    center = size // 2
    rows = range(max(1, center - ph), min(size - ph, center) + 1)
    cols = range(max(1, center - pw), min(size - pw, center) + 1)
    positions = []
    for r0 in rows:
        for c0 in cols:
            mid_r = r0 + (ph - 1) / 2
            mid_c = c0 + (pw - 1) / 2
            if abs(mid_r - center) <= 1.5 and abs(mid_c - center) <= 1.5:
                positions.append((int(r0), int(c0)))
    return positions


def _place_legend(canvas: np.ndarray, colors: list[int], is_row: bool) -> tuple[np.ndarray, tuple[int, int]] | None:
    patch = _legend_patch(colors, is_row)
    positions = _valid_legend_positions(canvas.shape[0], patch.shape)
    if not positions:
        return None
    random.shuffle(positions)
    for r0, c0 in positions:
        trial = canvas.copy()
        paste(trial, patch, r0, c0, transparent=BG)
        return trial, (r0, c0)
    return None


def _independent_expected(size: int, corners: list[str], legend_colors: list[int]) -> np.ndarray:
    out = np.full((size, size), BG, dtype=int)
    ordered = sorted(corners, key=lambda corner: CORNER_ORDER[corner])
    for idx, corner in enumerate(ordered):
        _draw_corner_l(out, corner, int(legend_colors[idx]))
    return out


def _counterfactual_changes_output(size: int, corners: list[str], legend_colors: list[int]) -> bool:
    expected = _independent_expected(size, corners, legend_colors)
    if len(legend_colors) >= 2:
        swapped = legend_colors[:]
        swapped[0], swapped[1] = swapped[1], swapped[0]
        if np.array_equal(expected, _independent_expected(size, corners, swapped)):
            return False

    ordered = sorted(corners, key=lambda corner: CORNER_ORDER[corner])
    available = [corner for corner in CORNERS if corner not in ordered]
    if available:
        altered = ordered[:]
        altered[-1] = random.choice(available)
        if np.array_equal(expected, _independent_expected(size, altered, legend_colors)):
            return False
    return True


def _attempt() -> np.ndarray | None:
    size = random.choice(GRID_SIZES)
    count = random.choices([2, 3, 4], weights=[4, 4, 3])[0]
    chosen_corners = random.sample(list(CORNERS), count)
    chosen_corners.sort(key=lambda corner: CORNER_ORDER[corner])

    palette = list(COLOR_POOL)
    random.shuffle(palette)
    legend_colors = palette[:count]
    marker_colors = palette[count : 2 * count]
    assert len(set(legend_colors + marker_colors)) == 2 * count

    if not _counterfactual_changes_output(size, chosen_corners, legend_colors):
        return None

    canvas = np.full((size, size), BG, dtype=int)
    is_row = random.choice([True, False])
    legend_result = _place_legend(canvas, legend_colors, is_row)
    if legend_result is None:
        return None
    canvas, _legend_pos = legend_result

    placement_order = list(range(count))
    random.shuffle(placement_order)
    for idx in placement_order:
        patch = _marker_patch(chosen_corners[idx], marker_colors[idx])
        try:
            r0, c0 = random_free_location(
                canvas,
                patch,
                transparent=BG,
                border=1,
                padding=2,
                padding_connectivity=8,
            )
        except ValueError:
            return None
        paste(canvas, patch, r0, c0, transparent=BG)

    expected = _independent_expected(size, chosen_corners, legend_colors)
    try:
        actual = main(canvas)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None

    markers, parsed_legend = _extract_markers_and_legend(canvas)
    parsed_corners = sorted((_component_corner(comp) for comp in markers), key=lambda corner: CORNER_ORDER[corner])
    if parsed_corners != chosen_corners:
        return None
    if parsed_legend != legend_colors:
        return None

    if count < 4:
        missing = [corner for corner in CORNERS if corner not in chosen_corners]
        if not missing:
            return None

    return canvas


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 4")
