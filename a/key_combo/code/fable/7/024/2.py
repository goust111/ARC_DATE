from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(ARC_UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(ARC_UTILS_DIR))

from arc_utils import Color, find_components, get_bg


BACKGROUND = Color.BLACK
FRAME_COLOR = Color.YELLOW
DOT_COLOR = Color.ORANGE
OUTER_COLORS = [c for c in Color.ALL_COLORS if c not in {BACKGROUND, FRAME_COLOR, DOT_COLOR}]
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
SECTOR_ORDER = ("N", "NE", "E", "SE", "S", "SW", "W", "NW")

OUTER_MARGIN = 1
FRAME_GAP = 2
BAND_RANGE = (4, 7)
FRAME_HW_RANGE = (6, 10)
REGION_COUNT_RANGE = (4, 8)
REGION_AREA_MAX = 6
GEN_ATTEMPTS = 300


def _component_centroid(coords: np.ndarray) -> tuple[float, float]:
    rows = coords[:, 0].astype(float)
    cols = coords[:, 1].astype(float)
    return float(rows.mean()), float(cols.mean())


def _frame_center(bbox: tuple[int, int, int, int]) -> tuple[float, float]:
    r0, c0, r1, c1 = bbox
    return (r0 + r1) / 2.0, (c0 + c1) / 2.0


def _clockwise_angle(center: tuple[float, float], point: tuple[float, float]) -> float:
    dr = point[0] - center[0]
    dc = point[1] - center[1]
    return math.atan2(dc, -dr) % (2.0 * math.pi)


def _coords_key(coords: np.ndarray) -> tuple[tuple[int, int], ...]:
    return tuple(sorted((int(r), int(c)) for r, c in coords))


def _is_hollow_rect_frame(grid: np.ndarray, comp: dict) -> bool:
    color = int(comp["color"])
    if color != FRAME_COLOR:
        return False
    r0, c0, r1, c1 = map(int, comp["bbox"])
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return False
    patch = grid[r0 : r1 + 1, c0 : c1 + 1]
    border = np.zeros((h, w), dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True
    return int(comp["area"]) == int(border.sum()) and np.array_equal(patch == color, border)


def _extract_scene(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = int(get_bg(g))

    frame_candidates = [
        comp
        for comp in find_components(
            g,
            bg=bg,
            connectivity=4,
            monochromatic=True,
            colors=[FRAME_COLOR],
            return_format="dict",
        )
        if _is_hollow_rect_frame(g, comp)
    ]
    assert len(frame_candidates) == 1, "expected a unique yellow hollow rectangle"
    frame = frame_candidates[0]
    r0, c0, r1, c1 = map(int, frame["bbox"])
    assert r1 - r0 >= 2 and c1 - c0 >= 2, "frame must enclose an interior"

    interior = g[r0 + 1 : r1, c0 + 1 : c1]
    interior_other = set(map(int, np.unique(interior))) - {bg, DOT_COLOR}
    assert not interior_other, "frame interior may contain only background and orange dots"
    dot_coords = np.argwhere(g == DOT_COLOR)
    k = int(np.count_nonzero(interior == DOT_COLOR))
    assert k >= 1, "expected at least one orange dot inside the frame"
    assert len(dot_coords) == k, "orange dots must appear only inside the frame"

    regions: list[dict] = []
    for comp in find_components(
        g,
        bg=bg,
        connectivity=4,
        monochromatic=True,
        return_format="dict",
    ):
        color = int(comp["color"])
        if color in {FRAME_COLOR, DOT_COLOR}:
            continue
        coords = comp["coords"].astype(int, copy=False)
        inside_frame_bbox = (
            (coords[:, 0] >= r0)
            & (coords[:, 0] <= r1)
            & (coords[:, 1] >= c0)
            & (coords[:, 1] <= c1)
        )
        assert not np.any(inside_frame_bbox), "outer regions must stay outside the frame"
        regions.append(comp)

    assert len(regions) >= 2, "expected multiple outer regions"
    region_colors = [int(comp["color"]) for comp in regions]
    assert len(region_colors) == len(set(region_colors)), "outer regions must all have distinct colors"

    center = _frame_center((r0, c0, r1, c1))
    ordered_regions = sorted(
        regions,
        key=lambda comp: _clockwise_angle(center, _component_centroid(comp["coords"])),
    )
    angles = [
        _clockwise_angle(center, _component_centroid(comp["coords"])) for comp in ordered_regions
    ]
    assert len({round(angle, 6) for angle in angles}) == len(angles), "angles must be unique"

    return {
        "grid": g,
        "bg": bg,
        "frame_bbox": (r0, c0, r1, c1),
        "frame_center": center,
        "k": k,
        "dot_coords": _coords_key(dot_coords),
        "ordered_regions": ordered_regions,
        "ordered_region_coords": [_coords_key(comp["coords"]) for comp in ordered_regions],
        "ordered_colors": [int(comp["color"]) for comp in ordered_regions],
    }


def main(input_grid: np.ndarray) -> np.ndarray:
    """Rotate the colors of the outer regions clockwise by a dot-controlled number of steps.

    Input structure:
    - A yellow one-pixel-thick hollow rectangle sits near the center of the grid.
    - The frame interior contains k orange dots on background.
    - Around the frame are several separated monochrome 4-connected regions, each with a distinct color.

    Rule:
    - Keep the yellow frame and all orange dots unchanged.
    - Order the outer regions by the clockwise angle of each region centroid around the frame centroid.
    - Recolor each region with the color of the region k places counterclockwise in that cyclic order, so the color ring shifts clockwise by k.

    Key insight:
    - The orange dots are the step counter: they do not move, but their count determines how far the surrounding color cycle rotates.
    """
    scene = _extract_scene(np.asarray(input_grid, dtype=int))
    g = scene["grid"]
    ordered_regions = scene["ordered_regions"]
    ordered_colors = scene["ordered_colors"]
    n_regions = len(ordered_regions)
    shift = scene["k"] % n_regions

    out = g.copy()
    if shift == 0:
        return out

    for idx, comp in enumerate(ordered_regions):
        new_color = ordered_colors[(idx - shift) % n_regions]
        coords = comp["coords"].astype(int, copy=False)
        out[coords[:, 0], coords[:, 1]] = int(new_color)
    return out


def _grow_polyomino(
    rng: np.random.Generator, height: int, width: int, area: int
) -> set[tuple[int, int]]:
    start = (int(rng.integers(height)), int(rng.integers(width)))
    cells = {start}
    while len(cells) < area:
        frontier: list[tuple[int, int]] = []
        for r, c in cells:
            for dr, dc in DIRS4:
                rr, cc = r + dr, c + dc
                if 0 <= rr < height and 0 <= cc < width and (rr, cc) not in cells:
                    frontier.append((rr, cc))
        if not frontier:
            break
        rr, cc = frontier[int(rng.integers(len(frontier)))]
        cells.add((int(rr), int(cc)))
    assert len(cells) == area
    return cells


def _shrink_window(window: tuple[int, int, int, int], pad: int = 1) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = window
    if r1 - r0 + 1 > 2 * pad:
        r0 += pad
        r1 -= pad
    if c1 - c0 + 1 > 2 * pad:
        c0 += pad
        c1 -= pad
    return r0, c0, r1, c1


def _can_place_with_gap(grid: np.ndarray, coords: list[tuple[int, int]]) -> bool:
    coord_set = set(coords)
    h, w = grid.shape
    for r, c in coords:
        if not (0 <= r < h and 0 <= c < w):
            return False
        if int(grid[r, c]) != BACKGROUND:
            return False
        for rr in range(r - 1, r + 2):
            for cc in range(c - 1, c + 2):
                if 0 <= rr < h and 0 <= cc < w and (rr, cc) not in coord_set:
                    if int(grid[rr, cc]) != BACKGROUND:
                        return False
    return True


def _place_region(
    grid: np.ndarray,
    rng: np.random.Generator,
    window: tuple[int, int, int, int],
    color: int,
) -> bool:
    ur0, uc0, ur1, uc1 = _shrink_window(window, pad=1)
    usable_h = ur1 - ur0 + 1
    usable_w = uc1 - uc0 + 1
    if usable_h < 2 or usable_w < 2:
        return False

    for _ in range(80):
        local_h = int(rng.integers(2, usable_h + 1))
        local_w = int(rng.integers(2, usable_w + 1))
        area = int(rng.integers(2, min(REGION_AREA_MAX, local_h * local_w) + 1))
        shape = _grow_polyomino(rng, local_h, local_w, area)
        top = int(rng.integers(ur0, ur1 - local_h + 2))
        left = int(rng.integers(uc0, uc1 - local_w + 2))
        coords = [(top + r, left + c) for r, c in shape]
        if not _can_place_with_gap(grid, coords):
            continue
        for r, c in coords:
            grid[r, c] = int(color)
        return True
    return False


def _sample_non_adjacent_dots(
    rng: np.random.Generator, bbox: tuple[int, int, int, int], count: int
) -> list[tuple[int, int]] | None:
    r0, c0, r1, c1 = bbox
    candidates = {(r, c) for r in range(r0 + 1, r1) for c in range(c0 + 1, c1)}
    chosen: list[tuple[int, int]] = []
    while len(chosen) < count and candidates:
        ordered = list(candidates)
        r, c = ordered[int(rng.integers(len(ordered)))]
        chosen.append((r, c))
        blocked = {(r, c)}
        for dr, dc in DIRS4:
            blocked.add((r + dr, c + dc))
        candidates -= blocked
    if len(chosen) != count:
        return None
    return chosen


def _sector_windows(frame_bbox: tuple[int, int, int, int], grid_shape: tuple[int, int]) -> dict[str, tuple[int, int, int, int]]:
    r0, c0, r1, c1 = frame_bbox
    h, w = grid_shape
    top = (OUTER_MARGIN, r0 - FRAME_GAP - 1)
    bottom = (r1 + FRAME_GAP + 1, h - OUTER_MARGIN - 1)
    left = (OUTER_MARGIN, c0 - FRAME_GAP - 1)
    right = (c1 + FRAME_GAP + 1, w - OUTER_MARGIN - 1)
    return {
        "NW": (top[0], left[0], top[1], left[1]),
        "N": (top[0], c0, top[1], c1),
        "NE": (top[0], right[0], top[1], right[1]),
        "E": (r0, right[0], r1, right[1]),
        "SE": (bottom[0], right[0], bottom[1], right[1]),
        "S": (bottom[0], c0, bottom[1], c1),
        "SW": (bottom[0], left[0], bottom[1], left[1]),
        "W": (r0, left[0], r1, left[1]),
    }


def _validate_transformation(inp: np.ndarray, out: np.ndarray) -> None:
    scene_in = _extract_scene(inp)
    scene_out = _extract_scene(out)

    assert inp.shape == out.shape
    assert np.array_equal(inp == FRAME_COLOR, out == FRAME_COLOR), "frame must stay fixed"
    assert scene_in["dot_coords"] == scene_out["dot_coords"], "dots must stay fixed"
    assert scene_in["frame_bbox"] == scene_out["frame_bbox"], "frame bbox must stay fixed"
    assert scene_in["ordered_region_coords"] == scene_out["ordered_region_coords"], "region geometry must stay fixed"

    colors_in = scene_in["ordered_colors"]
    colors_out = scene_out["ordered_colors"]
    n = len(colors_in)
    shift = scene_in["k"] % n
    expected = [colors_in[(idx - shift) % n] for idx in range(n)]
    assert colors_out == expected, "outer colors must rotate clockwise by k"
    assert set(colors_out) == set(colors_in), "output must reuse the same region colors"


def _generate_once(rng: np.random.Generator) -> np.ndarray | None:
    top_band = int(rng.integers(*BAND_RANGE))
    bottom_band = int(rng.integers(*BAND_RANGE))
    left_band = int(rng.integers(*BAND_RANGE))
    right_band = int(rng.integers(*BAND_RANGE))
    frame_h = int(rng.integers(*FRAME_HW_RANGE))
    frame_w = int(rng.integers(*FRAME_HW_RANGE))

    height = OUTER_MARGIN + top_band + FRAME_GAP + frame_h + FRAME_GAP + bottom_band + OUTER_MARGIN
    width = OUTER_MARGIN + left_band + FRAME_GAP + frame_w + FRAME_GAP + right_band + OUTER_MARGIN
    if max(height, width) > 30:
        return None

    fr0 = OUTER_MARGIN + top_band + FRAME_GAP
    fc0 = OUTER_MARGIN + left_band + FRAME_GAP
    fr1 = fr0 + frame_h - 1
    fc1 = fc0 + frame_w - 1
    frame_bbox = (fr0, fc0, fr1, fc1)

    n_regions = int(rng.integers(*REGION_COUNT_RANGE))
    n_regions = min(n_regions, len(OUTER_COLORS))
    k = int(rng.integers(1, n_regions))
    dots = _sample_non_adjacent_dots(rng, frame_bbox, k)
    if dots is None:
        return None

    grid = np.full((height, width), BACKGROUND, dtype=int)
    grid[fr0, fc0 : fc1 + 1] = FRAME_COLOR
    grid[fr1, fc0 : fc1 + 1] = FRAME_COLOR
    grid[fr0 : fr1 + 1, fc0] = FRAME_COLOR
    grid[fr0 : fr1 + 1, fc1] = FRAME_COLOR
    for r, c in dots:
        grid[r, c] = DOT_COLOR

    windows = _sector_windows(frame_bbox, grid.shape)
    chosen_sectors = list(rng.choice(SECTOR_ORDER, size=n_regions, replace=False))
    chosen_colors = list(rng.choice(OUTER_COLORS, size=n_regions, replace=False))
    rng.shuffle(chosen_sectors)
    for sector, color in zip(chosen_sectors, chosen_colors):
        if not _place_region(grid, rng, windows[sector], int(color)):
            return None

    out = main(grid)
    _validate_transformation(grid, out)
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_ATTEMPTS):
        grid = _generate_once(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input after repeated attempts")
