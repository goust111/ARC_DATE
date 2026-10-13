from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(_UTILS_DIR))

from arc_utils import crop_bbox, find_components, paste


BG = 0
GREY = 5
PANEL_COLORS = (1, 2, 3)
H_CHOICES = (8, 9)
RULER_RANGE = (2, 5)
MAX_GEN_ATTEMPTS = 400
SHAPE_ATTEMPTS = 120
PANEL_ATTEMPTS = 120


def _ordered_panel_objects(panel: np.ndarray) -> list[dict]:
    objects = find_components(panel, bg=BG, connectivity=4, monochromatic=True)
    assert len(objects) == 3, "each panel must contain exactly three objects"

    colors = {int(obj["color"]) for obj in objects}
    assert len(colors) == 1 and next(iter(colors)) in PANEL_COLORS, "panel objects must share one panel color"

    objects = sorted(objects, key=lambda obj: (int(obj["bbox"][1]), int(obj["bbox"][0])))
    prev_right = -1
    for obj in objects:
        r0, c0, r1, c1 = map(int, obj["bbox"])
        assert c0 > prev_right, "panel object bboxes must be horizontally disjoint"
        assert 0 <= r0 <= r1 < panel.shape[0]
        assert 0 <= c0 <= c1 < panel.shape[1]
        prev_right = c1
    return objects


def _analyze(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    rows, cols = g.shape
    assert rows >= 4, "grid must be tall enough for panels plus rulers"
    panel_size = rows - 1
    assert cols == 3 * panel_size + 2, "expected three square panels plus two separator columns"

    for sep_col in (panel_size, 2 * panel_size + 1):
        assert np.all(g[:panel_size, sep_col] == GREY), "separator columns must be solid grey across panel rows"

    panels: list[dict] = []
    panel_starts = (0, panel_size + 1, 2 * panel_size + 2)
    used_colors: set[int] = set()
    for panel_index, start_col in enumerate(panel_starts):
        panel = g[:panel_size, start_col : start_col + panel_size]
        assert panel.shape == (panel_size, panel_size)
        assert not np.any(panel == GREY), "grey should only appear in separators and rulers"

        objects = _ordered_panel_objects(panel)
        panel_color = int(objects[0]["color"])
        assert panel_color not in used_colors, "panel colors must be distinct"
        used_colors.add(panel_color)

        ruler_row = g[panel_size, start_col : start_col + panel_size]
        ruler_len = 0
        while ruler_len < panel_size and int(ruler_row[ruler_len]) == GREY:
            ruler_len += 1
        assert RULER_RANGE[0] <= ruler_len <= RULER_RANGE[1], "ruler length out of range"
        assert np.all(ruler_row[ruler_len:] == BG), "ruler must be left-aligned and contiguous"

        selected = objects[panel_index]
        tile = crop_bbox(panel, tuple(map(int, selected["bbox"])))
        panels.append(
            {
                "panel_index": int(panel_index),
                "panel": panel,
                "objects": objects,
                "selected": selected,
                "tile": tile,
                "ruler": int(ruler_len),
            }
        )
    return panels


def _tile_row(tile: np.ndarray, repeats: int) -> np.ndarray:
    h, w = tile.shape
    out_w = repeats * w + (repeats - 1)
    row = np.full((h, out_w), BG, dtype=int)
    for i in range(repeats):
        paste(row, tile, 0, i * (w + 1))
    return row


def _compose_rows(rows: list[np.ndarray]) -> np.ndarray:
    out_h = sum(int(row.shape[0]) for row in rows) + (len(rows) - 1)
    out_w = max(int(row.shape[1]) for row in rows)
    out = np.full((out_h, out_w), BG, dtype=int)
    cursor = 0
    for row in rows:
        paste(out, row, cursor, 0)
        cursor += int(row.shape[0]) + 1
    return out


def _expected_output(grid: np.ndarray) -> np.ndarray:
    panels = _analyze(grid)
    rows = [_tile_row(info["tile"], info["ruler"]) for info in panels]
    return _compose_rows(rows)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Select one ordered object from each square panel, crop it, and tile it by that panel's ruler length.

    Input structure:
    - A black background grid has three HxH panels on the top H rows, separated by one-cell grey vertical bars.
    - Each panel contains exactly three same-color 4-connected foreground objects whose bounding boxes are ordered left-to-right without horizontal overlap.
    - Directly below the panels is one row of grey ruler bars, each left-aligned under its panel and padded with black to the panel width.

    Rule:
    - In the left panel choose the leftmost object, in the middle panel choose the center object, and in the right panel choose the rightmost object.
    - Crop each chosen object to its own bounding box.
    - Read the grey ruler lengths under the three panels and repeat each cropped object horizontally that many times with one black column between copies.
    - Stack the three repeated rows vertically with one black row between them, and right-pad shorter rows with black up to the maximum row width.

    Key insight:
    - The panel position determines which ordered object survives, while the separate ruler beneath that same panel determines how many times the cropped object is tiled.
    """
    return _expected_output(grid)


def _grow_connected_mask(height: int, width: int, area: int) -> np.ndarray | None:
    start = (random.randrange(height), random.randrange(width))
    cells = {start}
    frontier: set[tuple[int, int]] = set()

    def add_frontier(r: int, c: int) -> None:
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            rr = r + dr
            cc = c + dc
            if 0 <= rr < height and 0 <= cc < width and (rr, cc) not in cells:
                frontier.add((rr, cc))

    add_frontier(*start)
    while len(cells) < area:
        if not frontier:
            return None
        frontier_list = list(frontier)
        frontier_list.sort(
            key=lambda rc: sum(
                (rc[0] + dr, rc[1] + dc) in cells for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1))
            ),
            reverse=True,
        )
        pick = random.choice(frontier_list[: min(3, len(frontier_list))])
        frontier.remove(pick)
        cells.add(pick)
        add_frontier(*pick)

    mask = np.zeros((height, width), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    return mask


def _sample_object_mask() -> np.ndarray | None:
    for _ in range(SHAPE_ATTEMPTS):
        height = random.randint(2, 4)
        width = random.randint(2, 4)
        max_area = min(8, height * width - 1)
        if max_area < 3:
            continue
        min_area = max(3, height + width - 2)
        if min_area > max_area:
            continue
        area = random.randint(min_area, max_area)
        mask = _grow_connected_mask(height, width, area)
        if mask is None:
            continue
        coords = np.argwhere(mask)
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        if (int(r0), int(c0), int(r1), int(c1)) != (0, 0, height - 1, width - 1):
            continue
        if area == height * width:
            continue
        return mask
    return None


def _mask_signature(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    return tuple(map(int, mask.shape)), mask.astype(np.uint8).tobytes()


def _build_panel(height: int, color: int) -> np.ndarray | None:
    for _ in range(PANEL_ATTEMPTS):
        masks = [_sample_object_mask() for _ in range(3)]
        if any(mask is None for mask in masks):
            continue
        masks = [mask for mask in masks if mask is not None]
        widths = [int(mask.shape[1]) for mask in masks]
        total_width = sum(widths) + 2
        if total_width > height:
            continue
        signatures = {_mask_signature(mask) for mask in masks}
        if len(signatures) != 3:
            continue

        slack = height - total_width
        extras = [0, 0, 0, 0]
        for _ in range(slack):
            extras[random.randrange(4)] += 1
        starts = []
        cursor = extras[0]
        for idx, width in enumerate(widths):
            starts.append(cursor)
            cursor += width
            if idx < 2:
                cursor += 1 + extras[idx + 1]

        panel = np.full((height, height), BG, dtype=int)
        for idx, mask in enumerate(masks):
            patch = np.where(mask, color, BG)
            top = random.randint(0, height - int(mask.shape[0]))
            paste(panel, patch, top, starts[idx], transparent=BG)

        try:
            objs = _ordered_panel_objects(panel)
        except AssertionError:
            continue
        if [int(obj["bbox"][1]) for obj in objs] != starts:
            continue
        return panel
    return None


def _semantic_checks(grid: np.ndarray) -> None:
    panels = _analyze(grid)
    rulers = [int(info["ruler"]) for info in panels]
    assert len(set(rulers)) >= 2, "need distinct ruler lengths so repetition count matters"

    tiled_rows = [_tile_row(info["tile"], info["ruler"]) for info in panels]
    row_widths = [int(row.shape[1]) for row in tiled_rows]
    assert len(set(row_widths)) >= 2, "need varying row widths so right-padding is exercised"

    area_middle_panel = False
    width_middle_panel = False
    for info in panels:
        obj_areas = [int(obj["area"]) for obj in info["objects"]]
        obj_widths = [int(obj["bbox"][3] - obj["bbox"][1] + 1) for obj in info["objects"]]
        idx = int(info["panel_index"])
        sel_area = obj_areas[idx]
        sel_width = obj_widths[idx]
        if any(area < sel_area for area in obj_areas) and any(area > sel_area for area in obj_areas):
            area_middle_panel = True
        if any(width < sel_width for width in obj_widths) and any(width > sel_width for width in obj_widths):
            width_middle_panel = True

        crop_sigs = []
        for obj in info["objects"]:
            crop_sigs.append(_mask_signature(crop_bbox(info["panel"], tuple(map(int, obj["bbox"]))) != BG))
        assert len(set(crop_sigs)) == 3, "panel objects must stay visually distinguishable"

    assert area_middle_panel or width_middle_panel, "reject size-based shortcut samples"


def _generate_attempt() -> np.ndarray | None:
    height = random.choice(H_CHOICES)
    panel_colors = random.sample(list(PANEL_COLORS), 3)
    panels = []
    for color in panel_colors:
        panel = _build_panel(height, color)
        if panel is None:
            return None
        panels.append(panel)

    rulers = [random.randint(*RULER_RANGE) for _ in range(3)]
    grid = np.full((height + 1, 3 * height + 2), BG, dtype=int)
    starts = (0, height + 1, 2 * height + 2)

    for sep_col in (height, 2 * height + 1):
        grid[:height, sep_col] = GREY

    for start, panel, ruler in zip(starts, panels, rulers):
        paste(grid, panel, 0, start)
        grid[height, start : start + ruler] = GREY

    try:
        expected = _expected_output(grid)
        actual = main(grid)
        assert np.array_equal(actual, expected)
        assert max(grid.shape) <= 30 and max(actual.shape) <= 30
        _semantic_checks(grid)
    except AssertionError:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 8")
