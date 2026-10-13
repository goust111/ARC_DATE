from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import Color, bbox_coords, components_mask, crop_bbox


PALETTE = (Color.BLUE, Color.GREEN, Color.YELLOW, Color.ORANGE)
MIN_REGIONS = 3
MAX_REGIONS = 5
MAX_ATTEMPTS = 400


def _touches_border(coords: np.ndarray, shape: tuple[int, int]) -> bool:
    height, width = map(int, shape)
    rows = coords[:, 0]
    cols = coords[:, 1]
    return bool(
        np.any(rows == 0)
        or np.any(rows == height - 1)
        or np.any(cols == 0)
        or np.any(cols == width - 1)
    )


def _region_infos(grid: np.ndarray) -> list[dict]:
    infos: list[dict] = []
    for coords in components_mask(grid != Color.RED, connectivity=4):
        if _touches_border(coords, grid.shape):
            continue
        box = bbox_coords(coords)
        patch = crop_bbox(grid, box)
        colors = sorted(set(map(int, patch[(patch != Color.BLACK) & (patch != Color.RED)].ravel())))
        assert len(colors) == 1, "each enclosed region must contain exactly one foreground color"
        color = colors[0]
        color_coords = np.argwhere(patch == color)
        assert color_coords.size > 0, "enclosed region must contain visible stripes"
        row_ids = sorted(set(map(int, color_coords[:, 0])))
        col_ids = sorted(set(map(int, color_coords[:, 1])))
        horizontal = all(r % 2 == 0 for r in row_ids)
        vertical = all(c % 2 == 0 for c in col_ids)
        assert horizontal ^ vertical, "stripe orientation must be uniquely determined"
        infos.append(
            {
                "coords": coords,
                "bbox": box,
                "color": color,
                "orientation": "horizontal" if horizontal else "vertical",
                "height": int(box[2] - box[0] + 1),
                "width": int(box[3] - box[1] + 1),
            }
        )
    return infos


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor the separator border segment above or to the right of each enclosed striped region according to that region's stripe direction.

    Input structure
    - A black background is partitioned by red separator lines into one exterior area
      and several enclosed non-red regions.
    - Each enclosed region contains black cells plus stripes of exactly one
      foreground color chosen from blue, green, yellow, or orange.
    - The stripe phase is measured from the enclosed region's own bbox top-left:
      foreground pixels appear either only on even local rows or only on even
      local columns.

    Rule
    - Find every 4-connected non-red component that does not touch the grid border;
      these are the enclosed regions surrounded by the red separator network.
    - If a region's color appears only on even local rows, treat it as horizontal
      stripes and recolor the red cells immediately above the region bbox, across
      the full bbox width, with that region color.
    - If a region's color appears only on even local columns, treat it as vertical
      stripes and recolor the red cells immediately to the right of the region bbox,
      across the full bbox height, with that region color.
    - Only overwrite cells that are red in the input. Keep every region interior,
      the exterior background, and all untouched separator cells unchanged.

    Key insight
    - The red geometry tells you which components are enclosed regions, but the
      stripe texture inside each enclosed region encodes the transformation
      parameter: horizontal stripes mean grow upward, vertical stripes mean grow
      rightward.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    out = g.copy()
    height, width = map(int, g.shape)
    for info in _region_infos(g):
        r0, c0, r1, c1 = info["bbox"]
        color = int(info["color"])
        if info["orientation"] == "horizontal":
            target_r = r0 - 1
            if target_r < 0:
                continue
            for c in range(c0, c1 + 1):
                if 0 <= c < width and g[target_r, c] == Color.RED:
                    out[target_r, c] = color
        else:
            target_c = c1 + 1
            if target_c >= width:
                continue
            for r in range(r0, r1 + 1):
                if 0 <= r < height and g[r, target_c] == Color.RED:
                    out[r, target_c] = color
    return out


def _paint_stripes(grid: np.ndarray, top: int, left: int, height: int, width: int, color: int, orientation: str) -> None:
    if orientation == "horizontal":
        for rr in range(height):
            if rr % 2 == 0:
                grid[top + rr, left : left + width] = color
    else:
        for cc in range(width):
            if cc % 2 == 0:
                grid[top : top + height, left + cc] = color


def _draw_frame(grid: np.ndarray, top: int, left: int, height: int, width: int) -> None:
    bottom = top + height + 1
    right = left + width + 1
    grid[top, left : right + 1] = Color.RED
    grid[bottom, left : right + 1] = Color.RED
    grid[top : bottom + 1, left] = Color.RED
    grid[top : bottom + 1, right] = Color.RED


def _expected_from_infos(grid: np.ndarray, infos: list[dict]) -> np.ndarray:
    out = grid.copy()
    for info in infos:
        r0, c0, r1, c1 = info["bbox"]
        color = int(info["color"])
        if info["orientation"] == "horizontal":
            rr = r0 - 1
            if rr >= 0:
                for c in range(c0, c1 + 1):
                    if grid[rr, c] == Color.RED:
                        out[rr, c] = color
        else:
            cc = c1 + 1
            if cc < grid.shape[1]:
                for r in range(r0, r1 + 1):
                    if grid[r, cc] == Color.RED:
                        out[r, cc] = color
    return out


def _connected_red_network(grid: np.ndarray) -> bool:
    red = components_mask(grid == Color.RED, connectivity=4)
    return len(red) == 1


def _layout_ok(specs: list[dict]) -> bool:
    heights = [spec["height"] for spec in specs]
    widths = [spec["width"] for spec in specs]
    orientations = [spec["orientation"] for spec in specs]
    if len(set(orientations)) < 2:
        return False
    if len(set(zip(heights, widths))) == 1:
        return False

    has_tall_horizontal = any(
        spec["orientation"] == "horizontal" and spec["height"] >= spec["width"]
        for spec in specs
    )
    has_wide_vertical = any(
        spec["orientation"] == "vertical" and spec["width"] >= spec["height"]
        for spec in specs
    )
    return has_tall_horizontal and has_wide_vertical


def _sample_specs() -> tuple[int, list[dict]] | None:
    n_regions = random.randint(MIN_REGIONS, MAX_REGIONS)
    specs: list[dict] = []

    for idx in range(n_regions):
        height = random.randint(3, 5 if n_regions <= 4 else 4)
        width = random.randint(3, 6)
        specs.append({"height": height, "width": width, "index": idx})

    total_height = sum(spec["height"] + 2 for spec in specs) + (n_regions - 1)
    if total_height > 30:
        return None

    orientations = ["horizontal"] * n_regions
    vertical_count = random.randint(1, n_regions - 1)
    for idx in random.sample(range(n_regions), vertical_count):
        orientations[idx] = "vertical"
    for spec, orientation in zip(specs, orientations):
        spec["orientation"] = orientation

    if not _layout_ok(specs):
        return None
    return n_regions, specs


def _assign_colors(specs: list[dict]) -> None:
    horiz = [idx for idx, spec in enumerate(specs) if spec["orientation"] == "horizontal"]
    vert = [idx for idx, spec in enumerate(specs) if spec["orientation"] == "vertical"]
    shared_color = random.choice(PALETTE)
    colors = [random.choice(PALETTE) for _ in specs]
    colors[random.choice(horiz)] = shared_color
    colors[random.choice(vert)] = shared_color
    for spec, color in zip(specs, colors):
        spec["color"] = int(color)


def _build_grid(specs: list[dict]) -> np.ndarray | None:
    trunk_col = 1
    row = 0
    frame_positions: list[tuple[int, int]] = []
    anchor_rows: list[int] = []
    max_right = 0

    for spec in specs:
        frame_top = row
        interior_left = random.randint(4, 11)
        frame_left = interior_left - 1
        frame_positions.append((frame_top, interior_left))
        anchor_row = frame_top + 1 + spec["height"] // 2
        anchor_rows.append(anchor_row)
        max_right = max(max_right, interior_left + spec["width"])
        row += spec["height"] + 3

    height = row - 1
    width = max_right + 2
    if height > 30 or width > 30:
        return None

    grid = np.zeros((height, width), dtype=int)

    min_anchor = min(anchor_rows)
    max_anchor = max(anchor_rows)
    grid[min_anchor : max_anchor + 1, trunk_col] = Color.RED

    for spec, (frame_top, interior_left), anchor_row in zip(specs, frame_positions, anchor_rows):
        height_i = spec["height"]
        width_i = spec["width"]
        _draw_frame(grid, frame_top, interior_left - 1, height_i, width_i)
        _paint_stripes(grid, frame_top + 1, interior_left, height_i, width_i, spec["color"], spec["orientation"])
        grid[anchor_row, trunk_col : interior_left] = Color.RED

    return grid


def _validate_generated(grid: np.ndarray, specs: list[dict]) -> bool:
    if grid is None or grid.ndim != 2 or max(grid.shape) > 30:
        return False
    if not _connected_red_network(grid):
        return False

    infos = _region_infos(grid)
    if len(infos) != len(specs):
        return False

    input_signature = sorted(
        (info["bbox"], info["color"], info["orientation"], info["height"], info["width"])
        for info in infos
    )
    spec_signature = sorted(
        (
            (frame_top + 1, interior_left, frame_top + spec["height"], interior_left + spec["width"] - 1),
            int(spec["color"]),
            spec["orientation"],
            spec["height"],
            spec["width"],
        )
        for spec, (frame_top, interior_left) in zip(specs, _frame_positions_from_specs(specs))
    )
    if input_signature != spec_signature:
        return False

    expected = _expected_from_infos(grid, infos)
    actual = main(grid)
    if not np.array_equal(actual, expected):
        return False

    changed = actual != grid
    if not np.any(changed):
        return False
    for info in infos:
        if info["orientation"] == "horizontal":
            rr = info["bbox"][0] - 1
            if not np.any(changed[rr, info["bbox"][1] : info["bbox"][3] + 1]):
                return False
        else:
            cc = info["bbox"][3] + 1
            if not np.any(changed[info["bbox"][0] : info["bbox"][2] + 1, cc]):
                return False
    return True


def _frame_positions_from_specs(specs: list[dict]) -> list[tuple[int, int]]:
    row = 0
    positions = []
    for spec in specs:
        interior_left = int(spec["interior_left"])
        positions.append((row, interior_left))
        row += spec["height"] + 3
    return positions


def _gen_attempt() -> np.ndarray | None:
    sampled = _sample_specs()
    if sampled is None:
        return None
    _n_regions, specs = sampled
    _assign_colors(specs)

    row = 0
    max_right = 0
    for spec in specs:
        spec["interior_left"] = random.randint(4, 11)
        max_right = max(max_right, spec["interior_left"] + spec["width"])
        row += spec["height"] + 3
    if row - 1 > 30 or max_right + 2 > 30:
        return None

    grid = np.zeros((row - 1, max_right + 2), dtype=int)
    positions = _frame_positions_from_specs(specs)
    anchor_rows = [frame_top + 1 + spec["height"] // 2 for spec, (frame_top, _left) in zip(specs, positions)]
    trunk_col = 1
    grid[min(anchor_rows) : max(anchor_rows) + 1, trunk_col] = Color.RED

    for spec, (frame_top, interior_left), anchor_row in zip(specs, positions, anchor_rows):
        _draw_frame(grid, frame_top, interior_left - 1, spec["height"], spec["width"])
        _paint_stripes(
            grid,
            frame_top + 1,
            interior_left,
            spec["height"],
            spec["width"],
            spec["color"],
            spec["orientation"],
        )
        grid[anchor_row, trunk_col : interior_left] = Color.RED

    if not _validate_generated(grid, specs):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle-10 input")
