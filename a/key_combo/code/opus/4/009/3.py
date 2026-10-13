from __future__ import annotations

import itertools
import random
import sys
from pathlib import Path

import numpy as np

_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_UTILS_DIR) not in sys.path:
    sys.path.append(str(_UTILS_DIR))

from arc_utils import find_components, get_bg


BG = 0
SEP = 1
NOISE = 5
CONTROL = 7

GEN_ATTEMPTS = 400
GRID_H_RANGE = (15, 19)
GRID_W_RANGE = (15, 19)
RECT_COUNT_CHOICES = (2, 3)
RECT_H_RANGE = (2, 3)
RECT_W_RANGE = (3, 4)
TOP_GAP_RANGE = (0, 3)
COLUMN_MARGIN = 1
COLUMN_GAP = 2
EXTRA_NOISE_RANGE = (0, 3)
SOURCE_PALETTE = (2, 3, 4, 6, 8, 9)


def _find_separator_row(grid: np.ndarray) -> int:
    rows = [r for r in range(grid.shape[0]) if np.all(grid[r, :] == SEP)]
    assert len(rows) == 1, "expected one full-width blue separator row"
    return int(rows[0])


def _is_solid_rectangle(component: dict) -> bool:
    r0, c0, r1, c1 = map(int, component["bbox"])
    area = int(component["area"])
    return area == (r1 - r0 + 1) * (c1 - c0 + 1)


def _find_source_rectangles(grid: np.ndarray, axis: int) -> list[dict]:
    bg = get_bg(grid)
    colors = [
        int(color)
        for color in np.unique(grid[:axis, :])
        if int(color) not in {bg, SEP, NOISE, CONTROL}
    ]
    components = find_components(
        grid,
        bg=bg,
        connectivity=4,
        monochromatic=True,
        colors=colors,
        min_area=4,
    )

    rects = []
    for component in components:
        r0, c0, r1, c1 = map(int, component["bbox"])
        height = r1 - r0 + 1
        width = c1 - c0 + 1
        if r1 >= axis:
            continue
        if height < 2 or width < 2:
            continue
        if not _is_solid_rectangle(component):
            continue
        rects.append(component)

    rects.sort(key=lambda comp: (int(comp["bbox"][1]), int(comp["bbox"][0])))
    assert 2 <= len(rects) <= 3, "expected two or three source rectangles"
    return rects


def _find_controls(grid: np.ndarray, axis: int) -> list[tuple[int, int]]:
    controls = [
        (int(r), int(c))
        for r, c in np.argwhere(grid == CONTROL)
        if int(r) > axis
    ]
    controls.sort()
    assert controls, "expected control pixels below the separator"
    return controls


def _match_controls(
    rects: list[dict],
    controls: list[tuple[int, int]],
    axis: int,
) -> list[tuple[dict, tuple[int, int]]]:
    unused = list(controls)
    pairs: list[tuple[dict, tuple[int, int]]] = []

    for rect in rects:
        r0, c0, r1, c1 = map(int, rect["bbox"])
        rect_h = r1 - r0 + 1
        matches = [control for control in unused if c0 <= control[1] <= c1]
        assert len(matches) == 1, "each rectangle must have exactly one control pixel underneath"
        control = matches[0]
        copy_h = int(control[0] - axis)
        assert 1 <= copy_h <= rect_h, "control must truncate the reflected copy to a valid height"
        pairs.append((rect, control))
        unused.remove(control)

    assert not unused, "every control pixel must belong to one rectangle"
    return pairs


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reflect each source rectangle below the separator, but stop its far edge at the orange control pixel and fill only the upper gap with separator color.

    Input structure:
    - Black background with one full-width blue separator row across the grid.
    - Two or three solid monochrome rectangles lie strictly above that separator.
    - One orange singleton control pixel lies below the separator within each rectangle's column span, while grey pixels are just distractors.

    Rule:
    - Keep the blue separator and preserve every orange control pixel.
    - Ignore grey distractors when identifying the source rectangles.
    - For each rectangle, paint the vertical strip from its bottom edge down to the separator in blue.
    - Then place a same-color reflected copy directly under the separator in the same columns, with top edge at the row just below the separator and bottom edge at the control pixel row; if that height is shorter than the source rectangle, keep only that many rows from the separator-facing side.

    Key insight:
    - The separator still fixes the reflection side and the rectangles still define the copied footprint, but the orange pixel no longer moves the axis: it only truncates the reflected copy's far edge.
    """
    g = np.asarray(grid, dtype=int)
    axis = _find_separator_row(g)
    rects = _find_source_rectangles(g, axis)
    controls = _find_controls(g, axis)
    pairs = _match_controls(rects, controls, axis)

    out = g.copy()
    for rect, control in pairs:
        r0, c0, r1, c1 = map(int, rect["bbox"])
        color = int(rect["color"])
        control_r, _control_c = control
        out[r1 + 1 : axis + 1, c0 : c1 + 1] = SEP
        out[axis + 1 : control_r + 1, c0 : c1 + 1] = color

    for control_r, control_c in controls:
        out[control_r, control_c] = CONTROL
    return out


def _random_composition(total: int, parts: int) -> list[int]:
    buckets = [0] * parts
    for _ in range(total):
        buckets[random.randrange(parts)] += 1
    return buckets


def _pack_intervals(width: int, sizes: list[int], *, margin: int, min_gap: int) -> list[tuple[int, int]] | None:
    extra = width - 2 * margin - sum(sizes) - min_gap * (len(sizes) - 1)
    if extra < 0:
        return None

    gap_buckets = _random_composition(extra, len(sizes) + 1)
    intervals: list[tuple[int, int]] = []
    c = margin + gap_buckets[0]
    for idx, size in enumerate(sizes):
        intervals.append((c, c + size - 1))
        c += size
        if idx + 1 < len(sizes):
            c += min_gap + gap_buckets[idx + 1]
    return intervals


def _choose_control_distances(heights: list[int]) -> list[int] | None:
    feasible: list[tuple[int, ...]] = []
    n = len(heights)
    for subset in itertools.combinations((1, 2, 3), n):
        for perm in itertools.permutations(subset):
            if all(distance <= height for distance, height in zip(perm, heights)) and any(
                distance < height for distance, height in zip(perm, heights)
            ):
                feasible.append(perm)
    if not feasible:
        return None
    return list(random.choice(feasible))


def _render_expected(grid: np.ndarray, rects: list[dict], axis: int) -> np.ndarray:
    out = grid.copy()
    for rect in rects:
        out[rect["r1"] + 1 : axis + 1, rect["c0"] : rect["c1"] + 1] = SEP
        out[axis + 1 : rect["control_r"] + 1, rect["c0"] : rect["c1"] + 1] = rect["color"]
    for rect in rects:
        out[rect["control_r"], rect["control_c"]] = CONTROL
    return out


def _naive_full_height_output(grid: np.ndarray, rects: list[dict], axis: int) -> np.ndarray:
    out = grid.copy()
    for rect in rects:
        full_bottom = axis + rect["h"]
        out[rect["r1"] + 1 : axis + 1, rect["c0"] : rect["c1"] + 1] = SEP
        out[axis + 1 : full_bottom + 1, rect["c0"] : rect["c1"] + 1] = rect["color"]
    for rect in rects:
        out[rect["control_r"], rect["control_c"]] = CONTROL
    return out


def _naive_symmetric_output(grid: np.ndarray, rects: list[dict], axis: int) -> np.ndarray:
    out = grid.copy()
    for rect in rects:
        mirror_r0 = 2 * axis - rect["r1"]
        mirror_r1 = 2 * axis - rect["r0"]
        out[rect["r1"] + 1 : mirror_r0, rect["c0"] : rect["c1"] + 1] = SEP
        out[mirror_r0 : mirror_r1 + 1, rect["c0"] : rect["c1"] + 1] = rect["color"]
    for rect in rects:
        out[rect["control_r"], rect["control_c"]] = CONTROL
    return out


def _all_top_half_cells(shape: tuple[int, int], axis: int) -> list[tuple[int, int]]:
    return [(r, c) for r in range(axis) for c in range(shape[1])]


def _build_case() -> tuple[np.ndarray, np.ndarray, list[dict], int] | None:
    n_rects = random.choice(RECT_COUNT_CHOICES)
    heights = [random.randint(*RECT_H_RANGE) for _ in range(n_rects)]
    widths = [random.randint(*RECT_W_RANGE) for _ in range(n_rects)]
    distances = _choose_control_distances(heights)
    if distances is None:
        return None

    gaps = [random.randint(*TOP_GAP_RANGE) for _ in range(n_rects)]
    if max(gaps) == 0:
        gaps[random.randrange(n_rects)] = random.randint(1, TOP_GAP_RANGE[1])

    rect_specs = [
        {"h": h, "w": w, "distance": d, "gap": gap}
        for h, w, d, gap in zip(heights, widths, distances, gaps)
    ]
    random.shuffle(rect_specs)

    min_axis = max(spec["h"] + spec["gap"] for spec in rect_specs) + 1
    axis_low = max(GRID_H_RANGE[0] // 2 - 1, min_axis)
    axis_high = GRID_H_RANGE[1] // 2 + 1
    if axis_low > axis_high:
        return None
    axis = random.randint(axis_low, axis_high)

    max_distance = max(spec["distance"] for spec in rect_specs)
    height_low = max(GRID_H_RANGE[0], axis + max_distance + 2)
    height_high = GRID_H_RANGE[1]
    if height_low > height_high:
        return None
    grid_h = random.randint(height_low, height_high)

    width_low = max(
        GRID_W_RANGE[0],
        2 * COLUMN_MARGIN + sum(spec["w"] for spec in rect_specs) + COLUMN_GAP * (n_rects - 1),
    )
    width_high = GRID_W_RANGE[1]
    if width_low > width_high:
        return None
    grid_w = random.randint(width_low, width_high)

    intervals = _pack_intervals(
        grid_w,
        [spec["w"] for spec in rect_specs],
        margin=COLUMN_MARGIN,
        min_gap=COLUMN_GAP,
    )
    if intervals is None:
        return None

    colors = random.sample(list(SOURCE_PALETTE), n_rects)
    rects: list[dict] = []
    for spec, color, (c0, c1) in zip(rect_specs, colors, intervals):
        r1 = axis - spec["gap"] - 1
        r0 = r1 - spec["h"] + 1
        control_r = axis + spec["distance"]
        control_c = random.randint(c0, c1)
        rects.append(
            {
                "r0": r0,
                "c0": c0,
                "r1": r1,
                "c1": c1,
                "h": spec["h"],
                "w": spec["w"],
                "gap": spec["gap"],
                "distance": spec["distance"],
                "color": color,
                "control_r": control_r,
                "control_c": control_c,
            }
        )

    grid = np.full((grid_h, grid_w), BG, dtype=int)
    grid[axis, :] = SEP

    occupied = np.zeros_like(grid, dtype=bool)
    occupied[axis, :] = True
    corridor_cells: list[tuple[int, int]] = []
    for rect in rects:
        grid[rect["r0"] : rect["r1"] + 1, rect["c0"] : rect["c1"] + 1] = rect["color"]
        grid[rect["control_r"], rect["control_c"]] = CONTROL
        occupied[rect["r0"] : rect["r1"] + 1, rect["c0"] : rect["c1"] + 1] = True
        occupied[rect["control_r"], rect["control_c"]] = True
        for r in range(rect["r1"] + 1, axis):
            for c in range(rect["c0"], rect["c1"] + 1):
                corridor_cells.append((r, c))

    if not corridor_cells:
        return None

    overwrite_noise = random.choice(corridor_cells)
    grid[overwrite_noise] = NOISE

    corridor_set = set(corridor_cells)
    preserved_candidates = [
        cell
        for cell in _all_top_half_cells(grid.shape, axis)
        if not occupied[cell] and cell not in corridor_set
    ]
    if not preserved_candidates:
        return None
    preserved_noise = random.choice(preserved_candidates)
    grid[preserved_noise] = NOISE

    extra_candidates = [
        cell
        for cell in _all_top_half_cells(grid.shape, axis)
        if not occupied[cell] and cell not in {overwrite_noise, preserved_noise}
    ]
    random.shuffle(extra_candidates)
    for cell in extra_candidates[: random.randint(*EXTRA_NOISE_RANGE)]:
        grid[cell] = NOISE

    expected = _render_expected(grid, rects, axis)
    return grid, expected, rects, axis


def _validate_case(grid: np.ndarray, expected: np.ndarray, rects: list[dict], axis: int) -> bool:
    if not np.array_equal(main(grid), expected):
        return False
    if np.array_equal(grid, expected):
        return False
    if np.array_equal(expected, _naive_full_height_output(grid, rects, axis)):
        return False
    if np.array_equal(expected, _naive_symmetric_output(grid, rects, axis)):
        return False

    distances = [rect["distance"] for rect in rects]
    heights = [rect["h"] for rect in rects]
    if len(set(distances)) != len(distances):
        return False
    if not any(distance < height for distance, height in zip(distances, heights)):
        return False
    if not any(rect["gap"] > 0 for rect in rects):
        return False

    input_noise = {tuple(map(int, rc)) for rc in np.argwhere(grid == NOISE)}
    output_noise = {tuple(map(int, rc)) for rc in np.argwhere(expected == NOISE)}
    if not input_noise.intersection(output_noise):
        return False
    if not input_noise.difference(output_noise):
        return False

    return True


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        built = _build_case()
        if built is None:
            continue
        grid, expected, rects, axis = built
        if _validate_case(grid, expected, rects, axis):
            return grid
    raise RuntimeError("failed to generate a valid control-truncated reflection case")
