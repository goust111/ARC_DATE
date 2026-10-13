from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import random

import numpy as np


_ARC_UTILS_PATH = Path("/home/lixinyu/snap/new/solvers/arc_utils.py")
_ARC_UTILS_SPEC = spec_from_file_location("arc_utils_shared", _ARC_UTILS_PATH)
if _ARC_UTILS_SPEC is None or _ARC_UTILS_SPEC.loader is None:
    raise ImportError(f"cannot load arc_utils from {_ARC_UTILS_PATH}")
_ARC_UTILS = module_from_spec(_ARC_UTILS_SPEC)
_ARC_UTILS_SPEC.loader.exec_module(_ARC_UTILS)

Color = _ARC_UTILS.Color
crop_bbox = _ARC_UTILS.crop_bbox
find_components = _ARC_UTILS.find_components
paste = _ARC_UTILS.paste


SEPARATOR = Color.GREY
STATIC = Color.BLUE
MOVER = Color.RED
BOUNDARY = Color.GREEN
BACKGROUND = Color.BLACK

GEN_RETRIES = 400
PANEL_H_CHOICES = (5, 6)
PANEL_W_CHOICES = (9, 10, 11, 12)
RED_SIZE_MIN = 3
RED_SIZE_MAX = 5
BLUE_COUNT_RANGE = (3, 6)


def _regions_between(length: int, separators: list[int]) -> list[tuple[int, int]]:
    regions: list[tuple[int, int]] = []
    start = 0
    for stop in separators:
        regions.append((start, stop - 1))
        start = stop + 1
    regions.append((start, length - 1))
    return regions


def _split_panels(grid: np.ndarray) -> list[np.ndarray]:
    height, width = map(int, grid.shape)
    row_lines = [r for r in range(height) if np.all(grid[r, :] == SEPARATOR)]
    col_lines = [c for c in range(width) if np.all(grid[:, c] == SEPARATOR)]

    if col_lines and not row_lines:
        regions = _regions_between(width, col_lines)
        assert len(regions) >= 3, "expected a horizontal panel sequence"
        widths = {c1 - c0 + 1 for c0, c1 in regions}
        assert len(widths) == 1, "panels must have equal width"
        return [grid[:, c0 : c1 + 1].copy() for c0, c1 in regions]

    if row_lines and not col_lines:
        regions = _regions_between(height, row_lines)
        assert len(regions) >= 3, "expected a vertical panel sequence"
        heights = {r1 - r0 + 1 for r0, r1 in regions}
        assert len(heights) == 1, "panels must have equal height"
        return [grid[r0 : r1 + 1, :].copy() for r0, r1 in regions]

    raise AssertionError("expected separators in exactly one axis")


def _normalize(coords: np.ndarray) -> set[tuple[int, int]]:
    base = coords.min(axis=0)
    shifted = coords - base
    return {tuple(map(int, rc)) for rc in shifted.tolist()}


def _component_shift(a: np.ndarray, b: np.ndarray) -> tuple[int, int]:
    a0 = a.min(axis=0)
    b0 = b.min(axis=0)
    shift = tuple(map(int, (b0 - a0).tolist()))
    translated = {tuple(map(int, rc)) for rc in (a + np.array(shift, dtype=int)).tolist()}
    assert translated == {tuple(map(int, rc)) for rc in b.tolist()}, "component shape changed over time"
    return shift


def _single_color_component(panel: np.ndarray, color: int) -> dict:
    comps = find_components(panel, bg=BACKGROUND, colors=[color], connectivity=4, monochromatic=True)
    assert len(comps) == 1, f"expected one component of color {color}"
    return comps[0]


def _green_pair(panel: np.ndarray) -> tuple[dict, dict]:
    comps = find_components(panel, bg=BACKGROUND, colors=[BOUNDARY], connectivity=4, monochromatic=True)
    assert len(comps) == 2, "expected two green boundary segments"
    ordered = sorted(comps, key=lambda comp: (int(comp["bbox"][1]), int(comp["bbox"][0])))
    for comp in ordered:
        cols = {int(c) for c in comp["coords"][:, 1].tolist()}
        rows = sorted(int(r) for r in comp["coords"][:, 0].tolist())
        assert len(cols) == 1, "green component must be vertical"
        assert rows == list(range(rows[0], rows[-1] + 1)), "green segment must be contiguous"
    return ordered[0], ordered[1]


def _predict_shifted_component(components: list[dict]) -> np.ndarray:
    coords_list = [comp["coords"] for comp in components]
    assert all(_normalize(coords) == _normalize(coords_list[0]) for coords in coords_list[1:])
    shifts = [_component_shift(a, b) for a, b in zip(coords_list, coords_list[1:])]
    assert len(set(shifts)) == 1, "motion must be arithmetic"
    dr, dc = shifts[0]
    return coords_list[-1] + np.array([dr, dc], dtype=int)


def _paint(out: np.ndarray, coords: np.ndarray, color: int) -> None:
    out[coords[:, 0], coords[:, 1]] = int(color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Predict the next panel in a separator-framed sequence, then crop it using the predicted green boundaries.

    Input structure:
    - The grid is split by one-pixel grey separator lines into equal panels arranged in one row or one column.
    - Every non-final panel contains the same blue static pattern plus one red connected shape and two green vertical segments.
    - The final panel is a blank template that keeps only the blue static pattern.

    Rule:
    - Use the last panel as the output canvas for the next frame.
    - Extrapolate the red component by continuing its constant panel-to-panel translation.
    - Extrapolate the two green vertical segments by continuing each segment's constant translation; their predicted columns are the crop boundaries.
    - Draw the predicted red and green objects onto the template, then crop the full panel rows between the two predicted green columns, inclusive.

    Key insight:
    - The green segments are both moving content and self-referential crop markers, so the same extrapolation step determines what to draw and what window to return.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0

    panels = _split_panels(g)
    assert len(panels) >= 4, "expected at least three observed panels plus one template"
    prior = panels[:-1]
    template = panels[-1]

    red_prediction = _predict_shifted_component([_single_color_component(panel, MOVER) for panel in prior])

    left_track: list[dict] = []
    right_track: list[dict] = []
    for panel in prior:
        left, right = _green_pair(panel)
        left_track.append(left)
        right_track.append(right)

    left_prediction = _predict_shifted_component(left_track)
    right_prediction = _predict_shifted_component(right_track)

    height, width = map(int, template.shape)
    for coords in (red_prediction, left_prediction, right_prediction):
        assert np.all(coords[:, 0] >= 0) and np.all(coords[:, 0] < height)
        assert np.all(coords[:, 1] >= 0) and np.all(coords[:, 1] < width)

    out = template.copy()
    _paint(out, red_prediction, MOVER)
    _paint(out, left_prediction, BOUNDARY)
    _paint(out, right_prediction, BOUNDARY)

    left_col = int(np.min(left_prediction[:, 1]))
    right_col = int(np.min(right_prediction[:, 1]))
    if left_col > right_col:
        left_col, right_col = right_col, left_col

    return crop_bbox(out, (0, left_col, height - 1, right_col))


def _random_polyomino(n_cells: int, max_h: int, max_w: int, attempts: int = 200) -> np.ndarray | None:
    if n_cells > max_h * max_w:
        return None

    for _ in range(attempts):
        cells = {(0, 0)}
        while len(cells) < n_cells:
            base_r, base_c = random.choice(tuple(cells))
            dr, dc = random.choice(((-1, 0), (1, 0), (0, -1), (0, 1)))
            candidate = (base_r + dr, base_c + dc)
            if candidate in cells:
                continue
            trial = cells | {candidate}
            rows = [r for r, _ in trial]
            cols = [c for _, c in trial]
            if max(rows) - min(rows) + 1 > max_h or max(cols) - min(cols) + 1 > max_w:
                continue
            cells = trial

        r0 = min(r for r, _ in cells)
        c0 = min(c for _, c in cells)
        coords = np.array(sorted((r - r0, c - c0) for r, c in cells), dtype=int)
        if len(_normalize(coords)) == n_cells:
            return coords
    return None


def _shift_coords(coords: np.ndarray, dr: int, dc: int) -> np.ndarray:
    return coords + np.array([dr, dc], dtype=int)


def _line_coords(top: int, col: int, length: int) -> np.ndarray:
    return np.array([(top + i, col) for i in range(length)], dtype=int)


def _sample_blue_cells(
    panel_h: int,
    panel_w: int,
    blocked: set[tuple[int, int]],
    left_pred: int,
    right_pred: int,
) -> list[tuple[int, int]] | None:
    inside = [
        (r, c)
        for r in range(panel_h)
        for c in range(left_pred, right_pred + 1)
        if (r, c) not in blocked
    ]
    outside = [
        (r, c)
        for r in range(panel_h)
        for c in range(panel_w)
        if (r, c) not in blocked and not (left_pred <= c <= right_pred)
    ]
    if not inside:
        return None

    total = random.randint(*BLUE_COUNT_RANGE)
    inside_count = min(len(inside), random.randint(1, min(3, total)))
    chosen = random.sample(inside, inside_count)

    remaining = total - inside_count
    pool = [cell for cell in inside + outside if cell not in chosen]
    if len(pool) < remaining:
        return None
    chosen.extend(random.sample(pool, remaining))

    rows = {r for r, _ in chosen}
    cols = {c for _, c in chosen}
    if len(rows) == 1 or len(cols) == 1:
        return None
    return chosen


def _build_panel(panel_h: int, panel_w: int, blue_cells: list[tuple[int, int]]) -> np.ndarray:
    panel = np.full((panel_h, panel_w), BACKGROUND, dtype=int)
    for r, c in blue_cells:
        panel[r, c] = STATIC
    return panel


def _render_case(
    panel_h: int,
    panel_w: int,
    blue_cells: list[tuple[int, int]],
    red_base: np.ndarray,
    red_start: tuple[int, int],
    red_step: int,
    left_top: int,
    left_len: int,
    right_top: int,
    right_len: int,
    left_pred: int,
    right_pred: int,
) -> tuple[np.ndarray, np.ndarray]:
    template = _build_panel(panel_h, panel_w, blue_cells)
    red_origin = np.array(red_start, dtype=int)

    panels: list[np.ndarray] = []
    for t in range(3):
        panel = template.copy()
        red_coords = red_base + red_origin + np.array([t * red_step, 0], dtype=int)
        left_coords = _line_coords(left_top, left_pred - 3 + t, left_len)
        right_coords = _line_coords(right_top, right_pred + 3 - t, right_len)
        _paint(panel, red_coords, MOVER)
        _paint(panel, left_coords, BOUNDARY)
        _paint(panel, right_coords, BOUNDARY)
        panels.append(panel)
    panels.append(template)

    grid = np.full((panel_h * 4 + 3, panel_w), BACKGROUND, dtype=int)
    for idx, panel in enumerate(panels):
        paste(grid, panel, idx * (panel_h + 1), 0)
        if idx < 3:
            grid[idx * (panel_h + 1) + panel_h, :] = SEPARATOR

    predicted = template.copy()
    _paint(predicted, red_base + red_origin + np.array([3 * red_step, 0], dtype=int), MOVER)
    _paint(predicted, _line_coords(left_top, left_pred, left_len), BOUNDARY)
    _paint(predicted, _line_coords(right_top, right_pred, right_len), BOUNDARY)
    expected = crop_bbox(predicted, (0, left_pred, panel_h - 1, right_pred))
    return grid, expected


def _gen_attempt() -> np.ndarray | None:
    panel_h = random.choice(PANEL_H_CHOICES)
    panel_w = random.choice(PANEL_W_CHOICES)

    red_max_h = panel_h - 3
    max_interior = panel_w - 8
    if red_max_h <= 0 or max_interior <= 0 or red_max_h * max_interior < RED_SIZE_MIN:
        return None

    red_size = random.randint(RED_SIZE_MIN, min(RED_SIZE_MAX, red_max_h * max_interior))
    red_base = _random_polyomino(red_size, red_max_h, max_interior)
    if red_base is None:
        return None

    red_h = int(red_base[:, 0].max()) + 1
    red_w = int(red_base[:, 1].max()) + 1
    interior = random.randint(red_w, max_interior)

    left_pred_max = panel_w - 4 - (interior + 1)
    if left_pred_max < 3:
        return None
    left_pred = random.randint(3, left_pred_max)
    right_pred = left_pred + interior + 1

    red_step = random.choice((-1, 1))
    if red_step > 0:
        red_r0_max = panel_h - red_h - 3
        if red_r0_max < 0:
            return None
        red_r0 = random.randint(0, red_r0_max)
    else:
        red_r0_min = 3
        red_r0_max = panel_h - red_h
        if red_r0_min > red_r0_max:
            return None
        red_r0 = random.randint(red_r0_min, red_r0_max)

    red_c0_min = left_pred + 1
    red_c0_max = right_pred - red_w
    if red_c0_min > red_c0_max:
        return None
    red_c0 = random.randint(red_c0_min, red_c0_max)

    left_len = random.randint(2, min(3, panel_h))
    right_len = random.randint(2, min(3, panel_h))
    left_top = random.randint(0, panel_h - left_len)
    right_top = random.randint(0, panel_h - right_len)

    blocked: set[tuple[int, int]] = set()
    red_origin = np.array([red_r0, red_c0], dtype=int)
    for t in range(4):
        for r, c in _shift_coords(red_base + red_origin, t * red_step, 0).tolist():
            blocked.add((int(r), int(c)))
        for r, c in _line_coords(left_top, left_pred - 3 + t, left_len).tolist():
            blocked.add((int(r), int(c)))
        for r, c in _line_coords(right_top, right_pred + 3 - t, right_len).tolist():
            blocked.add((int(r), int(c)))

    blue_cells = _sample_blue_cells(panel_h, panel_w, blocked, left_pred, right_pred)
    if blue_cells is None:
        return None

    grid, expected = _render_case(
        panel_h=panel_h,
        panel_w=panel_w,
        blue_cells=blue_cells,
        red_base=red_base,
        red_start=(red_r0, red_c0),
        red_step=red_step,
        left_top=left_top,
        left_len=left_len,
        right_top=right_top,
        right_len=right_len,
        left_pred=left_pred,
        right_pred=right_pred,
    )

    try:
        actual = main(grid)
    except AssertionError:
        return None

    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 4")
