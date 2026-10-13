from __future__ import annotations

import random
import sys
from collections import Counter

import numpy as np

try:
    from arc_utils import Color, find_components
except ModuleNotFoundError:
    sys.path.append("/home/lixinyu/snap/new/solvers")
    from arc_utils import Color, find_components


PALETTE = tuple(Color.ALL_COLORS)
MAX_GENERATION_ATTEMPTS = 500


def _mode(values: np.ndarray) -> int:
    flat = [int(v) for v in np.asarray(values).ravel()]
    return int(Counter(flat).most_common(1)[0][0])


def _bbox_of(coords: set[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _positive_partition(total: int, parts: int) -> list[int]:
    assert total >= parts >= 1
    if parts == 1:
        return [int(total)]
    cuts = sorted(random.sample(range(1, total), parts - 1))
    points = [0, *cuts, total]
    return [points[i + 1] - points[i] for i in range(parts)]


def _make_intervals(panel_w: int, count: int) -> list[tuple[int, int]]:
    usable = panel_w - (count - 1)
    widths = _positive_partition(usable, count)
    intervals: list[tuple[int, int]] = []
    cursor = 0
    for width in widths:
        intervals.append((cursor, cursor + width - 1))
        cursor += width + 1
    return intervals


def _selected_shape(row: int, interval: tuple[int, int]) -> set[tuple[int, int]]:
    c0, c1 = interval
    return {(int(row), c) for c in range(int(c0), int(c1) + 1)}


def _distractor_shape(panel_h: int, interval: tuple[int, int]) -> set[tuple[int, int]]:
    c0, c1 = interval
    width = c1 - c0 + 1
    shape = random.choice(("hbar", "rect", "vbar", "lshape"))

    if shape == "hbar":
        row = random.randrange(panel_h)
        left = random.randint(c0, c1)
        right = random.randint(left, c1)
        return {(row, c) for c in range(left, right + 1)}

    if shape == "rect":
        height = random.randint(1, min(3, panel_h))
        top = random.randint(0, panel_h - height)
        left = random.randint(c0, c1)
        right = random.randint(left, c1)
        return {(r, c) for r in range(top, top + height) for c in range(left, right + 1)}

    if shape == "vbar":
        col = random.randint(c0, c1)
        height = random.randint(1, min(4, panel_h))
        top = random.randint(0, panel_h - height)
        return {(r, col) for r in range(top, top + height)}

    if width == 1 or panel_h == 1:
        return _distractor_shape(panel_h, interval)

    box_w = random.randint(2, min(3, width))
    box_h = random.randint(2, min(3, panel_h))
    left = random.randint(c0, c1 - box_w + 1)
    top = random.randint(0, panel_h - box_h)
    cells = {(top, c) for c in range(left, left + box_w)}
    if random.random() < 0.5:
        cells |= {(r, left) for r in range(top, top + box_h)}
    else:
        cells |= {(r, left + box_w - 1) for r in range(top, top + box_h)}
    return cells


def _parse_scene(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape[0] >= 2 and g.shape[1] >= 3, "expected a 2D ARC grid"
    h, w = g.shape

    sep = _mode(g[-1, :])
    separator_cols = [c for c in range(w) if bool(np.all(g[:, c] == sep))]
    panel_count = len(separator_cols) + 1
    assert panel_count >= 2, "expected at least two panels"
    assert (w - (panel_count - 1)) % panel_count == 0, "invalid equal-panel layout"
    panel_w = (w - (panel_count - 1)) // panel_count
    panel_h = h - 1

    expected_separators = [panel_w + i * (panel_w + 1) for i in range(panel_count - 1)]
    assert separator_cols == expected_separators, "separator columns must be equally spaced"

    panels: list[np.ndarray] = []
    markers: list[int] = []
    for panel_index in range(panel_count):
        c0 = panel_index * (panel_w + 1)
        c1 = c0 + panel_w
        panels.append(g[:panel_h, c0:c1].copy())
        marker_positions = np.flatnonzero(g[-1, c0:c1] != sep)
        assert marker_positions.size == 1, "each panel span needs exactly one marker"
        markers.append(int(marker_positions[0]))

    panel_cells = np.concatenate([panel.ravel() for panel in panels])
    fg_pool = [int(v) for v in panel_cells if int(v) != sep]
    assert fg_pool, "panels cannot be empty"
    bg = int(Counter(fg_pool).most_common(1)[0][0])

    objects_by_panel: list[list[dict]] = []
    for panel in panels:
        objects = find_components(panel, bg=bg, connectivity=4, monochromatic=False)
        assert objects, "each panel must contain at least one object"
        objects_by_panel.append(objects)

    return {
        "grid": g,
        "bg": bg,
        "sep": sep,
        "panel_h": panel_h,
        "panel_w": panel_w,
        "panel_count": panel_count,
        "panels": panels,
        "markers": markers,
        "objects_by_panel": objects_by_panel,
    }


def _pointed_object(objects: list[dict], rel_col: int) -> dict:
    matches = []
    for obj in objects:
        _r0, c0, _r1, c1 = map(int, obj["bbox"])
        if c0 <= rel_col <= c1:
            matches.append(obj)
    assert len(matches) == 1, "marker must point to a unique object in the next panel"
    return matches[0]


def _compose_output(scene: dict) -> np.ndarray:
    bg = int(scene["bg"])
    out = np.full((scene["panel_h"], scene["panel_w"]), bg, dtype=int)

    for panel_index, rel_col in enumerate(scene["markers"]):
        target_panel_index = (panel_index + 1) % scene["panel_count"]
        target_panel = scene["panels"][target_panel_index]
        target_obj = _pointed_object(scene["objects_by_panel"][target_panel_index], rel_col)
        for r, c in target_obj["coords"]:
            rr, cc = int(r), int(c)
            assert out[rr, cc] == bg, "selected objects must not overlap in the output"
            out[rr, cc] = int(target_panel[rr, cc])

    return out


def main(input_grid: np.ndarray) -> np.ndarray:
    """
    Read each control-row marker as a relative column pointer into the next panel.

    The pointed object from each next panel is copied onto a single panel-sized
    canvas at its original in-panel coordinates.
    """
    scene = _parse_scene(input_grid)
    return _compose_output(scene)


def _generate_case() -> np.ndarray | None:
    panel_count = random.choice((3, 4))
    max_panel_w = (30 - (panel_count - 1)) // panel_count
    panel_w = random.randint(5, min(8, max_panel_w))
    panel_h = random.randint(max(panel_count + 2, 6), 9)

    bg, sep = random.sample(PALETTE, 2)
    obj_colors = [color for color in PALETTE if color not in {bg, sep}]
    selected_rows = random.sample(range(panel_h), panel_count)

    panels: list[np.ndarray] = []
    target_columns: list[int] = []
    local_colors: list[list[int]] = []

    for panel_index in range(panel_count):
        panel = np.full((panel_h, panel_w), bg, dtype=int)
        max_objects = min(4, (panel_w + 1) // 2)
        object_count = random.randint(2, max_objects)
        intervals = _make_intervals(panel_w, object_count)
        target_index = random.randrange(object_count)
        colors = random.choices(obj_colors, k=object_count)
        local_colors.append(colors)

        for obj_index, (color, interval) in enumerate(zip(colors, intervals)):
            if obj_index == target_index:
                coords = _selected_shape(selected_rows[panel_index], interval)
            else:
                coords = _distractor_shape(panel_h, interval)
            for r, c in coords:
                panel[r, c] = int(color)

        target_bbox = _bbox_of(_selected_shape(selected_rows[panel_index], intervals[target_index]))
        target_columns.append(random.randint(target_bbox[1], target_bbox[3]))
        panels.append(panel)

    control = np.full(panel_count * panel_w + (panel_count - 1), sep, dtype=int)
    for panel_index in range(panel_count):
        c0 = panel_index * (panel_w + 1)
        rel_col = target_columns[(panel_index + 1) % panel_count]
        marker_choices = [target_columns[(panel_index + 1) % panel_count]]
        marker_palette = [color for color in local_colors[panel_index] if color != sep]
        if marker_palette and random.random() < 0.7:
            marker_color = random.choice(marker_palette)
        else:
            marker_color = random.choice(obj_colors)
        control[c0 + rel_col] = int(marker_color)

    grid = np.full((panel_h + 1, panel_count * panel_w + (panel_count - 1)), bg, dtype=int)
    for panel_index, panel in enumerate(panels):
        c0 = panel_index * (panel_w + 1)
        grid[:panel_h, c0 : c0 + panel_w] = panel
        if panel_index < panel_count - 1:
            grid[:, c0 + panel_w] = sep
    grid[-1, :] = control

    if max(grid.shape) > 30:
        return None

    expected = _compose_output(_parse_scene(grid))
    if expected.shape != (panel_h, panel_w):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        case = _generate_case()
        if case is not None:
            return case
    raise RuntimeError("failed to generate a valid ARC case")


def _verify_semantics(input_grid: np.ndarray, output_grid: np.ndarray) -> None:
    scene = _parse_scene(input_grid)
    expected = _compose_output(scene)
    assert isinstance(output_grid, np.ndarray)
    assert output_grid.ndim == 2
    assert output_grid.shape == expected.shape
    assert np.array_equal(output_grid, expected)
    assert max(output_grid.shape) <= 30


if __name__ == "__main__":
    for _ in range(5):
        sample = generate_input()
        result = main(sample)
        _verify_semantics(sample, result)
