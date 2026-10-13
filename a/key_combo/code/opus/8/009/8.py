from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_UTILS_DIR) not in sys.path:
    sys.path.append(str(_UTILS_DIR))

from arc_utils import Color, bbox_h, bbox_w, crop_bbox, find_components, get_bg, paste


SHAPE_LIBRARY = [
    np.array([[1, 1], [1, 0]], dtype=bool),
    np.array([[1, 1], [0, 1]], dtype=bool),
    np.array([[1, 1, 1], [1, 0, 0]], dtype=bool),
    np.array([[1, 1, 1], [0, 1, 0]], dtype=bool),
    np.array([[1, 1, 0], [0, 1, 1]], dtype=bool),
    np.array([[1, 1, 1], [1, 1, 0]], dtype=bool),
    np.array([[1, 1, 0], [1, 1, 1]], dtype=bool),
    np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool),
    np.array([[1, 1, 0], [0, 1, 1], [0, 0, 1]], dtype=bool),
    np.array([[1, 1, 1], [1, 0, 1]], dtype=bool),
]

TILE_COLORS = [Color.RED, Color.GREEN, Color.YELLOW, Color.PINK, Color.ORANGE, Color.TEAL, Color.MAROON]
MAX_GENERATION_ATTEMPTS = 400


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _transform_shape(shape: np.ndarray) -> np.ndarray:
    out = np.array(shape, dtype=bool, copy=True)
    if random.random() < 0.5:
        out = np.fliplr(out)
    if random.random() < 0.5:
        out = np.flipud(out)
    return np.rot90(out, k=random.randrange(4)).copy()


def _marker_candidates(shape: np.ndarray) -> list[tuple[int, int]]:
    h, w = shape.shape
    filled = {(int(r), int(c)) for r, c in np.argwhere(shape)}
    candidates: list[tuple[int, int]] = []
    for r in range(-1, h + 1):
        for c in range(-1, w + 1):
            if (r, c) in filled:
                continue
            touch_count = sum((nr, nc) in filled for nr, nc in _neighbors4(r, c))
            if touch_count == 1:
                candidates.append((r, c))
    return candidates


def _make_selected_group(shape: np.ndarray) -> np.ndarray:
    marker_r, marker_c = random.choice(_marker_candidates(shape))
    filled = [(int(r), int(c)) for r, c in np.argwhere(shape)]
    rows = [r for r, _ in filled] + [marker_r]
    cols = [c for _, c in filled] + [marker_c]
    r_shift = -min(rows)
    c_shift = -min(cols)
    h = max(rows) - min(rows) + 1
    w = max(cols) - min(cols) + 1
    patch = np.zeros((h, w), dtype=int)
    for r, c in filled:
        patch[r + r_shift, c + c_shift] = Color.BLUE
    patch[marker_r + r_shift, marker_c + c_shift] = Color.GREY
    return patch


def _make_tile() -> np.ndarray:
    size = random.choice([2, 3, 4])
    colors = random.sample(TILE_COLORS, k=random.choice([2, 3]))
    tile = np.random.choice(colors, size=(size, size))
    for color in colors:
        if color not in tile:
            rr = random.randrange(size)
            cc = random.randrange(size)
            tile[rr, cc] = color
    if len(set(map(int, tile.flatten()))) < 2:
        tile[0, 0] = colors[0]
        tile[-1, -1] = colors[-1]
    return tile.astype(int)


def _fits_with_gap(
    occupied: set[tuple[int, int]],
    cells: list[tuple[int, int]],
    shape: tuple[int, int],
) -> bool:
    h, w = shape
    for r, c in cells:
        if not (0 <= r < h and 0 <= c < w):
            return False
        for rr in range(r - 1, r + 2):
            for cc in range(c - 1, c + 2):
                if (rr, cc) in occupied:
                    return False
    return True


def _place_groups(groups: list[np.ndarray], grid_shape: tuple[int, int]) -> np.ndarray | None:
    h, w = grid_shape
    order = sorted(range(len(groups)), key=lambda i: groups[i].size, reverse=True)
    grid = np.zeros((h, w), dtype=int)
    occupied: set[tuple[int, int]] = set()

    for idx in order:
        patch = groups[idx]
        ph, pw = patch.shape
        nonzero = [(int(r), int(c)) for r, c in np.argwhere(patch != Color.BLACK)]
        if not nonzero:
            return None

        positions = [(r0, c0) for r0 in range(h - ph + 1) for c0 in range(w - pw + 1)]
        random.shuffle(positions)
        placed = False
        for r0, c0 in positions:
            abs_cells = [(r0 + r, c0 + c) for r, c in nonzero]
            if not _fits_with_gap(occupied, abs_cells, grid_shape):
                continue
            paste(grid, patch, r0, c0, transparent=Color.BLACK)
            occupied.update(abs_cells)
            placed = True
            break
        if not placed:
            return None
    return grid


def _find_selected_component(grid: np.ndarray) -> dict:
    marker_coords = np.argwhere(grid == Color.GREY)
    assert marker_coords.shape == (1, 2), "expected exactly one grey marker"
    mr, mc = map(int, marker_coords[0])

    adjacent_blue = {
        (nr, nc)
        for nr, nc in _neighbors4(mr, mc)
        if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1] and int(grid[nr, nc]) == Color.BLUE
    }
    assert adjacent_blue, "marker must touch a blue object"

    blue_components = find_components(
        grid,
        bg=get_bg(grid),
        colors=[Color.BLUE],
        connectivity=4,
        monochromatic=True,
        return_format="dict",
    )
    matches = []
    for comp in blue_components:
        coords = {(int(r), int(c)) for r, c in comp["coords"]}
        if coords & adjacent_blue:
            matches.append(comp)
    assert len(matches) == 1, "marker must select exactly one blue object"
    return matches[0]


def _find_tile(grid: np.ndarray) -> np.ndarray:
    bg = get_bg(grid)
    components = find_components(grid, bg=bg, connectivity=4, monochromatic=False, return_format="dict")
    candidates: list[np.ndarray] = []
    for comp in components:
        patch = crop_bbox(grid, comp["bbox"])
        colors = set(map(int, np.unique(patch))) - {int(bg)}
        if Color.BLUE in colors or Color.GREY in colors:
            continue
        if len(colors) < 2:
            continue
        if bbox_h(comp["bbox"]) != bbox_w(comp["bbox"]):
            continue
        if int(comp["area"]) != patch.size:
            continue
        if np.any(patch == bg):
            continue
        candidates.append(patch)
    assert len(candidates) == 1, "expected one solid multicolor square tile"
    return candidates[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use the grey-selected blue shape as a mask that pixel-upscales into copies of the multicolor tile.

    Input structure:
    - A black background contains several separate 4-connected blue objects.
    - Exactly one grey marker pixel touches one blue object by a 4-neighbor edge.
    - Elsewhere there is one solid multicolor square tile that uses colors other than black, blue, and grey.

    Rule:
    - Find the unique blue object touched by the grey marker.
    - Crop that object's bounding box and convert it into a binary mask of filled blue cells versus empty background.
    - Find the solid multicolor square tile.
    - For each filled mask cell, stamp a full copy of the tile into the corresponding output block; for each empty mask cell, leave a same-sized black block.

    Key insight:
    - The marked blue object is not itself the answer: its cropped shape becomes the control mask that decides where enlarged tile copies appear.
    """
    g = np.asarray(grid, dtype=int)
    selected = _find_selected_component(g)
    control_mask = crop_bbox(g == Color.BLUE, selected["bbox"])
    tile = _find_tile(g)

    ch, cw = control_mask.shape
    th, tw = tile.shape
    out = np.zeros((ch * th, cw * tw), dtype=int)
    for r in range(ch):
        for c in range(cw):
            if bool(control_mask[r, c]):
                paste(out, tile, r * th, c * tw)
    return out


def _generate_attempt() -> np.ndarray | None:
    grid_h = random.randint(16, 22)
    grid_w = random.randint(16, 22)
    object_count = random.choice([3, 4])

    base_shapes = random.sample(SHAPE_LIBRARY, k=object_count)
    blue_shapes = [_transform_shape(shape) for shape in base_shapes]
    selected_index = random.randrange(object_count)
    selected_patch = _make_selected_group(blue_shapes[selected_index])
    tile = _make_tile()

    groups: list[np.ndarray] = []
    for idx, shape in enumerate(blue_shapes):
        if idx == selected_index:
            groups.append(selected_patch)
        else:
            patch = np.zeros(shape.shape, dtype=int)
            patch[shape] = Color.BLUE
            groups.append(patch)
    groups.append(tile)

    grid = _place_groups(groups, (grid_h, grid_w))
    if grid is None:
        return None

    selected = _find_selected_component(grid)
    control_mask = crop_bbox(grid == Color.BLUE, selected["bbox"])
    th, tw = tile.shape
    if max(control_mask.shape[0] * th, control_mask.shape[1] * tw) > 30:
        return None

    try:
        out = main(grid)
    except Exception:
        return None

    if out.ndim != 2 or max(out.shape) > 30:
        return None
    if len(np.argwhere(grid == Color.GREY)) != 1:
        return None
    if len(find_components(grid, bg=Color.BLACK, colors=[Color.BLUE], connectivity=4, monochromatic=True)) != object_count:
        return None
    if not np.any(out != Color.BLACK):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
