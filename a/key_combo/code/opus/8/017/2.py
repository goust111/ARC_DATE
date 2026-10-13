from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_UTILS_DIR) not in sys.path:
    sys.path.append(str(_UTILS_DIR))

from arc_utils import bbox_coords, colors_present, components_mask, get_bg


BLACK = 0
PALETTE = tuple(range(1, 10))

GEN_ATTEMPTS = 500
GRID_ROWS_RANGE = (18, 24)
GRID_COLS_RANGE = (18, 24)
ANCHOR_COUNT_CHOICES = (3, 4)
TOTAL_COLOR_CHOICES = (5, 6, 7)
ANCHOR_AREA_RANGE = (4, 8)
ANCHOR_TOP_MARGIN_RANGE = (3, 4)
ANCHOR_BOTTOM_MARGIN_RANGE = (1, 3)
ANCHOR_GAP_RANGE = (2, 4)
PLACEMENT_GAP = 1


ANCHOR_LIBRARY = (
    np.array([[1, 1, 0], [0, 1, 1]], dtype=bool),
    np.array([[1, 0], [1, 1], [1, 0]], dtype=bool),
    np.array([[1, 1, 1], [1, 0, 0]], dtype=bool),
    np.array([[1, 1, 0], [0, 1, 0], [0, 1, 1]], dtype=bool),
    np.array([[1, 1, 1], [0, 1, 0], [0, 1, 0]], dtype=bool),
    np.array([[1, 0, 0], [1, 1, 1], [0, 0, 1]], dtype=bool),
    np.array([[1, 1, 0], [0, 1, 1], [0, 0, 1]], dtype=bool),
    np.array([[1, 1, 1], [1, 0, 0], [1, 0, 0]], dtype=bool),
    np.array([[1, 1, 1, 0], [0, 1, 0, 1]], dtype=bool),
    np.array([[1, 0, 1], [1, 1, 1], [0, 1, 0]], dtype=bool),
)

NOISE_LIBRARY = (
    np.array([[1]], dtype=bool),
    np.array([[1, 1]], dtype=bool),
    np.array([[1], [1]], dtype=bool),
    np.array([[1, 1, 1]], dtype=bool),
    np.array([[1], [1], [1]], dtype=bool),
    np.array([[1, 1], [1, 0]], dtype=bool),
    np.array([[1, 1], [0, 1]], dtype=bool),
)


def _transform_mask(mask: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    out = np.rot90(mask, int(rng.integers(0, 4)))
    if bool(rng.integers(0, 2)):
        out = np.flipud(out)
    if bool(rng.integers(0, 2)):
        out = np.fliplr(out)
    return np.asarray(out, dtype=bool)


def _anchor_like(mask: np.ndarray) -> bool:
    area = int(mask.sum())
    return (
        ANCHOR_AREA_RANGE[0] <= area <= ANCHOR_AREA_RANGE[1]
        and components_mask(mask, connectivity=4)[0].shape[0] == area
        and not np.array_equal(mask, np.flipud(mask))
        and not np.array_equal(mask, np.fliplr(mask))
    )


def _noise_patch(rng: np.random.Generator) -> np.ndarray:
    return _transform_mask(NOISE_LIBRARY[int(rng.integers(0, len(NOISE_LIBRARY)))], rng)


def _colorize_mask(mask: np.ndarray, color: int) -> np.ndarray:
    patch = np.full(mask.shape, BLACK, dtype=int)
    patch[mask] = int(color)
    return patch


def _anchor_patch(rng: np.random.Generator, color: int) -> np.ndarray:
    for _ in range(100):
        base = ANCHOR_LIBRARY[int(rng.integers(0, len(ANCHOR_LIBRARY)))]
        mask = _transform_mask(base, rng)
        if _anchor_like(mask):
            return _colorize_mask(mask, color)
    raise RuntimeError("failed to sample an asymmetric anchor patch")


def _random_split(rng: np.random.Generator, total: int, parts: int) -> list[int]:
    out = [0] * parts
    for _ in range(total):
        out[int(rng.integers(0, parts))] += 1
    return out


def _can_place_patch(grid: np.ndarray, patch: np.ndarray, r0: int, c0: int, gap: int = PLACEMENT_GAP) -> bool:
    h, w = map(int, patch.shape)
    rows, cols = grid.shape
    if r0 < 0 or c0 < 0 or r0 + h > rows or c0 + w > cols:
        return False

    rr0 = max(0, r0 - gap)
    cc0 = max(0, c0 - gap)
    rr1 = min(rows, r0 + h + gap)
    cc1 = min(cols, c0 + w + gap)
    if np.any(grid[rr0:rr1, cc0:cc1] != BLACK):
        return False
    return True


def _place_patch(grid: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    h, w = map(int, patch.shape)
    view = grid[r0 : r0 + h, c0 : c0 + w]
    mask = patch != BLACK
    view[mask] = patch[mask]


def _band_bounds(sorted_rows: list[int], height: int) -> list[tuple[int, int]]:
    mids = [(sorted_rows[i] + sorted_rows[i + 1]) // 2 for i in range(len(sorted_rows) - 1)]
    bounds: list[tuple[int, int]] = []
    start = 0
    for mid in mids:
        bounds.append((start, int(mid)))
        start = int(mid) + 1
    bounds.append((start, int(height) - 1))
    return bounds


def _scene_info(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    infos: list[dict] = []
    for color in colors_present(g, exclude={bg}):
        coords = np.argwhere(g == color)
        comps = components_mask(g == color, connectivity=4)
        infos.append(
            {
                "color": int(color),
                "coords": coords,
                "components": comps,
                "component_count": len(comps),
                "area": int(coords.shape[0]),
                "bbox": bbox_coords(coords),
                "centroid_row": int(np.floor(coords[:, 0].mean())),
            }
        )
    anchors = [info for info in infos if info["component_count"] == 1]
    noises = [info for info in infos if info["component_count"] > 1]
    anchors.sort(key=lambda info: info["centroid_row"])
    return {"grid": g, "bg": bg, "infos": infos, "anchors": anchors, "noises": noises}


def _solve_from_scene(scene: dict) -> np.ndarray:
    g = scene["grid"]
    bg = scene["bg"]
    anchors = scene["anchors"]
    rows, _cols = g.shape
    assert anchors, "expected at least one anchor color"

    centroid_rows = [int(info["centroid_row"]) for info in anchors]
    bounds = _band_bounds(centroid_rows, rows)
    out = np.full_like(g, bg)

    for band_index, (start, end) in enumerate(bounds, start=1):
        band_anchors = [info for info in anchors if start <= int(info["centroid_row"]) <= end]
        for info in band_anchors:
            coords = info["coords"]
            r0, c0, r1, c1 = info["bbox"]
            if band_index % 2 == 1:
                rr = coords[:, 0]
                cc = c0 + c1 - coords[:, 1]
            else:
                rr = r0 + r1 - coords[:, 0]
                cc = coords[:, 1]
            out[rr, cc] = int(info["color"])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Filter to single-component anchor colors, split the canvas into centroid-row bands, and mirror anchors by band parity.

    Input structure:
    - A black background contains 5-7 non-background colors.
    - Three or four colors each form one irregular 4-connected object whose centroid rows are all different.
    - The remaining colors are clutter split into multiple small 4-connected components scattered around the canvas.

    Rule:
    - For each non-background color, keep it only if all of its pixels belong to exactly one 4-connected component; erase every multi-component color back to black.
    - Compute the floor of each surviving color's mean row, sort those centroid rows, and use floor midpoints between neighbors as horizontal band boundaries.
    - Number the bands from top to bottom; mirror every surviving object inside its own bounding box left-right in odd bands and top-bottom in even bands.

    Key insight:
    - The mirror axis is not tied to the color or to a global flip: it is determined by the object's band, and the bands themselves only become correct after fragmented noise colors are discarded.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    return _solve_from_scene(_scene_info(g))


def _anchor_band_assignments(scene: dict) -> list[tuple[dict, int]]:
    anchors = scene["anchors"]
    bounds = _band_bounds([int(info["centroid_row"]) for info in anchors], scene["grid"].shape[0])
    assigned: list[tuple[dict, int]] = []
    for info in anchors:
        matches = [idx for idx, (start, end) in enumerate(bounds, start=1) if start <= int(info["centroid_row"]) <= end]
        assert matches and len(matches) == 1
        assigned.append((info, matches[0]))
    return assigned


def _non_overlapping_bboxes(infos: list[dict]) -> bool:
    for i, info_a in enumerate(infos):
        a0, b0, a1, b1 = info_a["bbox"]
        for info_b in infos[i + 1 :]:
            c0, d0, c1, d1 = info_b["bbox"]
            if not (a1 < c0 or c1 < a0 or b1 < d0 or d1 < b0):
                return False
    return True


def _wrong_band_indices_if_unfiltered(scene: dict) -> dict[int, int]:
    infos = sorted(scene["infos"], key=lambda info: int(info["centroid_row"]))
    bounds = _band_bounds([int(info["centroid_row"]) for info in infos], scene["grid"].shape[0])
    by_color: dict[int, int] = {}
    for info in scene["anchors"]:
        matches = [idx for idx, (start, end) in enumerate(bounds, start=1) if start <= int(info["centroid_row"]) <= end]
        assert matches and len(matches) == 1
        by_color[int(info["color"])] = matches[0]
    return by_color


def _validate_sample(grid: np.ndarray) -> bool:
    scene = _scene_info(grid)
    anchors = scene["anchors"]
    noises = scene["noises"]
    rows, cols = scene["grid"].shape

    if scene["bg"] != BLACK:
        return False
    if not (18 <= rows <= 24 and 18 <= cols <= 24):
        return False
    if len(scene["infos"]) not in TOTAL_COLOR_CHOICES:
        return False
    if len(anchors) not in ANCHOR_COUNT_CHOICES:
        return False
    if len(noises) != len(scene["infos"]) - len(anchors):
        return False
    if len(noises) < 2 or len(noises) > 3:
        return False
    if len({int(info["centroid_row"]) for info in anchors}) != len(anchors):
        return False
    if not _non_overlapping_bboxes(anchors):
        return False

    horizontal_changes = 0
    vertical_changes = 0
    for info, band_index in _anchor_band_assignments(scene):
        mask = np.zeros((info["bbox"][2] - info["bbox"][0] + 1, info["bbox"][3] - info["bbox"][1] + 1), dtype=bool)
        local = info["coords"] - np.array([info["bbox"][0], info["bbox"][1]])
        mask[local[:, 0], local[:, 1]] = True
        if not _anchor_like(mask):
            return False
        if band_index % 2 == 1:
            horizontal_changes += int(not np.array_equal(mask, np.fliplr(mask)))
        else:
            vertical_changes += int(not np.array_equal(mask, np.flipud(mask)))

    for info in noises:
        if info["component_count"] < 2:
            return False
        if any(int(comp.shape[0]) > 3 for comp in info["components"]):
            return False

    noise_rows = [int(coord[0]) for info in noises for coord in info["coords"]]
    if 0 not in noise_rows:
        return False
    anchor_top = min(int(info["centroid_row"]) for info in anchors)
    if not any(int(info["centroid_row"]) < anchor_top for info in noises):
        return False

    out = main(grid)
    if np.array_equal(out, grid):
        return False
    if not np.array_equal(out, _solve_from_scene(scene)):
        return False
    if horizontal_changes < 1 or vertical_changes < 1:
        return False

    wrong_bands = _wrong_band_indices_if_unfiltered(scene)
    if not any(wrong_bands[int(info["color"])] != band_index for info, band_index in _anchor_band_assignments(scene)):
        return False

    return True


def _sample_anchor_layout(rng: np.random.Generator, heights: list[int]) -> tuple[int, list[int]] | None:
    top_margin = int(rng.integers(ANCHOR_TOP_MARGIN_RANGE[0], ANCHOR_TOP_MARGIN_RANGE[1] + 1))
    bottom_margin = int(rng.integers(ANCHOR_BOTTOM_MARGIN_RANGE[0], ANCHOR_BOTTOM_MARGIN_RANGE[1] + 1))
    base_gaps = [int(rng.integers(ANCHOR_GAP_RANGE[0], ANCHOR_GAP_RANGE[1] + 1)) for _ in range(len(heights) - 1)]
    min_rows = top_margin + sum(heights) + sum(base_gaps) + bottom_margin
    if min_rows > GRID_ROWS_RANGE[1]:
        return None

    rows = int(rng.integers(max(GRID_ROWS_RANGE[0], min_rows), GRID_ROWS_RANGE[1] + 1))
    extras = _random_split(rng, rows - min_rows, len(heights) + 1)

    starts: list[int] = []
    cursor = top_margin + extras[0]
    for idx, height in enumerate(heights):
        starts.append(cursor)
        cursor += height
        if idx < len(base_gaps):
            cursor += base_gaps[idx] + extras[idx + 1]
    return rows, starts


def _place_extreme_noise_color(
    rng: np.random.Generator,
    grid: np.ndarray,
    color: int,
    component_count: int,
) -> bool:
    for forced_row in [0] + [None] * (component_count - 1):
        placed_here = False
        for _ in range(60):
            mask = _noise_patch(rng)
            patch = _colorize_mask(mask, color)
            if forced_row is not None and not mask[0, :].any():
                continue
            h, w = map(int, patch.shape)
            if forced_row is None:
                r0 = int(rng.integers(0, max(1, min(2, grid.shape[0] - h + 1))))
            else:
                r0 = int(forced_row)
            c0 = int(rng.integers(0, grid.shape[1] - w + 1))
            if not _can_place_patch(grid, patch, r0, c0):
                continue
            _place_patch(grid, patch, r0, c0)
            placed_here = True
            break
        if not placed_here:
            return False
    return True


def _place_regular_noise_color(
    rng: np.random.Generator,
    grid: np.ndarray,
    color: int,
    component_count: int,
) -> bool:
    placed = 0
    for _ in range(component_count * 60):
        patch = _colorize_mask(_noise_patch(rng), color)
        h, w = map(int, patch.shape)
        r0 = int(rng.integers(0, grid.shape[0] - h + 1))
        c0 = int(rng.integers(0, grid.shape[1] - w + 1))
        if not _can_place_patch(grid, patch, r0, c0):
            continue
        _place_patch(grid, patch, r0, c0)
        placed += 1
        if placed == component_count:
            return True
    return False


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    anchor_count = int(rng.choice(np.array(ANCHOR_COUNT_CHOICES)))
    total_colors = int(rng.choice(np.array([n for n in TOTAL_COLOR_CHOICES if 2 <= n - anchor_count <= 3])))
    colors = list(map(int, rng.choice(np.array(PALETTE), size=total_colors, replace=False)))
    anchor_colors = colors[:anchor_count]
    noise_colors = colors[anchor_count:]

    anchor_patches = [_anchor_patch(rng, color) for color in anchor_colors]
    layout = _sample_anchor_layout(rng, [int(patch.shape[0]) for patch in anchor_patches])
    if layout is None:
        return None
    rows, row_starts = layout
    cols = int(rng.integers(GRID_COLS_RANGE[0], GRID_COLS_RANGE[1] + 1))
    grid = np.zeros((rows, cols), dtype=int)

    for patch, r0 in zip(anchor_patches, row_starts):
        placed = False
        for _ in range(80):
            c0 = int(rng.integers(0, cols - patch.shape[1] + 1))
            if not _can_place_patch(grid, patch, int(r0), c0):
                continue
            _place_patch(grid, patch, int(r0), c0)
            placed = True
            break
        if not placed:
            return None

    extreme_components = int(rng.integers(2, 4))
    if not _place_extreme_noise_color(rng, grid, noise_colors[0], extreme_components):
        return None

    for color in noise_colors[1:]:
        component_count = int(rng.integers(2, 5))
        if not _place_regular_noise_color(rng, grid, color, component_count):
            return None

    if not _validate_sample(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
