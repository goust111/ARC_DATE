from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random
import sys

import numpy as np


_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_UTILS_DIR) not in sys.path:
    sys.path.append(str(_UTILS_DIR))

from arc_utils import Color


@dataclass(frozen=True)
class SpanBar:
    axis: str
    lo: int
    hi: int
    color: int


@dataclass(frozen=True)
class Scene:
    horizontal: tuple[SpanBar, ...]
    vertical: tuple[SpanBar, ...]
    pixels: tuple[tuple[int, int], ...]


BAR_COLORS = tuple(range(1, 10))
H_COUNT_RANGE = (2, 4)
V_COUNT_RANGE = (3, 4)
HEIGHT_RANGE = (12, 22)
WIDTH_RANGE = (17, 27)
THICKNESS_CHOICES = (1, 1, 2)
TOP_MARGIN = 2
BOTTOM_MARGIN = 1
SIDE_MARGIN = 1
MIN_BAR_GAP = 1
GEN_RETRY_LIMIT = 500


def _group_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for idx, value in enumerate(mask.tolist()):
        if value and start is None:
            start = idx
        elif not value and start is not None:
            runs.append((start, idx - 1))
            start = None
    if start is not None:
        runs.append((start, int(mask.size) - 1))
    return runs


def _composition(total: int, parts: int) -> list[int]:
    if parts == 1:
        return [int(total)]
    cuts = sorted(random.sample(range(total + parts - 1), parts - 1))
    values: list[int] = []
    prev = -1
    for cut in cuts + [total + parts - 1]:
        values.append(cut - prev - 1)
        prev = cut
    return values


def _sample_intervals(
    length: int,
    count: int,
    *,
    thickness_choices: tuple[int, ...],
    start_margin: int,
    end_margin: int,
    gap: int,
) -> list[tuple[int, int]] | None:
    for _ in range(80):
        thicknesses = [int(random.choice(thickness_choices)) for _ in range(count)]
        minimum = start_margin + end_margin + gap * (count - 1) + sum(thicknesses)
        if minimum > length:
            continue
        extras = _composition(length - minimum, count + 1)
        pos = start_margin + extras[0]
        intervals: list[tuple[int, int]] = []
        for idx, thickness in enumerate(thicknesses):
            intervals.append((pos, pos + thickness - 1))
            pos += thickness
            if idx < count - 1:
                pos += gap + extras[idx + 1]
        return intervals
    return None


def _bar_cell_ranges(bars: tuple[SpanBar, ...]) -> set[int]:
    cells: set[int] = set()
    for bar in bars:
        cells.update(range(bar.lo, bar.hi + 1))
    return cells


def _analyze_scene(grid: np.ndarray) -> Scene:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = int(Color.BLACK)
    height, width = g.shape
    assert height >= 5 and width >= 5, "grid is too small"

    full_rows = np.all(g != bg, axis=1)
    full_cols = np.all(g != bg, axis=0)
    assert full_rows.any(), "expected at least one horizontal bar row"
    assert full_cols.any(), "expected at least one vertical bar column"
    assert not bool(full_rows[0]), "top row must contain falling pixels, not a full-span bar"

    h_runs = _group_runs(full_rows)
    v_runs = _group_runs(full_cols)
    non_h_rows = [int(r) for r in np.where(~full_rows)[0]]
    non_v_cols = [int(c) for c in np.where(~full_cols)[0]]
    assert non_h_rows, "expected some rows outside horizontal bars"
    assert non_v_cols, "expected some columns outside vertical bars"

    horizontal: list[SpanBar] = []
    for r0, r1 in h_runs:
        sample = g[r0 : r1 + 1][:, non_v_cols]
        colors = {int(v) for v in np.unique(sample) if int(v) != bg}
        assert len(colors) == 1, "each horizontal bar must expose one color off vertical crossings"
        horizontal.append(SpanBar("h", int(r0), int(r1), colors.pop()))

    vertical: list[SpanBar] = []
    for c0, c1 in v_runs:
        sample = g[non_h_rows][:, c0 : c1 + 1]
        colors = {int(v) for v in np.unique(sample) if int(v) != bg}
        assert len(colors) == 1, "each vertical bar must expose one color off horizontal crossings"
        vertical.append(SpanBar("v", int(c0), int(c1), colors.pop()))

    bar_colors = [bar.color for bar in horizontal] + [bar.color for bar in vertical]
    assert len(set(bar_colors)) == len(bar_colors), "bar colors must be globally unique"

    horizontal_rows = _bar_cell_ranges(tuple(horizontal))
    vertical_cols = _bar_cell_ranges(tuple(vertical))

    pixels: list[tuple[int, int]] = []
    for r in range(height):
        for c in range(width):
            value = int(g[r, c])
            on_bar = r in horizontal_rows or c in vertical_cols
            if on_bar:
                assert value != bg, "bar cells must stay filled"
            elif r == 0:
                if value != bg:
                    pixels.append((int(c), value))
            else:
                assert value == bg, "non-bar cells below the top row must be background"

    assert pixels, "expected at least one falling pixel"
    assert len({c for c, _color in pixels}) == len(pixels), "each falling pixel must start in its own column"

    vertical_colors = {bar.color for bar in vertical}
    for c, color in pixels:
        assert c not in vertical_cols, "falling pixels cannot start inside a vertical bar"
        assert color in vertical_colors, "each falling pixel color must match a vertical bar"

    return Scene(tuple(horizontal), tuple(vertical), tuple(sorted(pixels)))


def _cross_top_color(grid: np.ndarray, hbar: SpanBar, vbar: SpanBar) -> int:
    block = grid[hbar.lo : hbar.hi + 1, vbar.lo : vbar.hi + 1]
    colors = {int(v) for v in np.unique(block)}
    assert len(colors) == 1, "each bar crossing must show a single visible color"
    top = colors.pop()
    assert top in {hbar.color, vbar.color}, "crossing color must belong to one of the two bars"
    return top


def _landing_row(grid: np.ndarray, scene: Scene, pixel_color: int) -> int:
    vertical_by_color = {bar.color: bar for bar in scene.vertical}
    assert pixel_color in vertical_by_color, "falling pixel must map to a vertical bar"
    vbar = vertical_by_color[pixel_color]
    for hbar in scene.horizontal:
        if _cross_top_color(grid, hbar, vbar) == hbar.color:
            row = hbar.lo - 1
            assert row >= 0, "a stopping row must exist above the catching bar"
            return row
    return int(grid.shape[0] - 1)


def _apply_landings(
    grid: np.ndarray,
    scene: Scene,
    landing_rows: dict[tuple[int, int], int],
) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    bg = int(Color.BLACK)
    for c, _color in scene.pixels:
        out[0, c] = bg
    for key, row in landing_rows.items():
        c, color = key
        assert int(out[int(row), int(c)]) == bg, "falling pixels must land in empty non-bar cells"
        out[int(row), int(c)] = int(color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Drop each top-row pixel until a bar-crossing layer relation says it must stop.

    Input structure:
    - The background is black.
    - Several full-width horizontal bars and several full-height vertical bars cross each other.
    - Each bar has its own color, and every crossing cell shows whichever bar is higher in the layer stack.
    - Extra single pixels sit on the top row in columns without vertical bars; each such pixel matches one vertical bar color.

    Rule:
    - Keep all bars fixed and move only the top-row single pixels.
    - For a falling pixel, inspect horizontal bars from top to bottom.
    - At each horizontal bar, look at that bar's crossing with the vertical bar of the same color as the pixel.
    - If the crossing shows the horizontal color, the pixel stops in the cell just above that bar.
    - If the crossing shows the vertical color, the pixel passes through and continues downward.
    - A pixel that passes every horizontal bar lands on the bottom row; its start cell becomes black.

    Key insight:
    - The falling column itself is a decoy: each pixel uses its same-color vertical bar as a private hardness lookup table for every horizontal layer.
    """
    g = np.asarray(grid, dtype=int)
    scene = _analyze_scene(g)

    landing_rows: dict[tuple[int, int], int] = {}
    for c, color in scene.pixels:
        landing_rows[(int(c), int(color))] = _landing_row(g, scene, int(color))
    return _apply_landings(g, scene, landing_rows)


def _stop_index_for_vertical(
    horizontal_order: list[int],
    vertical_rank: int,
) -> int:
    for idx, h_rank in enumerate(horizontal_order):
        if h_rank > vertical_rank:
            return int(idx)
    return int(len(horizontal_order))


def _candidate_pixel_columns(width: int, vertical_intervals: list[tuple[int, int]]) -> list[int]:
    blocked = {c for c0, c1 in vertical_intervals for c in range(c0, c1 + 1)}
    candidates = [
        c
        for c in range(width)
        if c not in blocked
        and c - 1 not in blocked
        and c + 1 not in blocked
    ]
    random.shuffle(candidates)
    chosen: list[int] = []
    for c in candidates:
        if all(abs(c - prev) >= 2 for prev in chosen):
            chosen.append(int(c))
    return sorted(chosen)


def _build_input(spec: dict) -> np.ndarray:
    height, width = spec["shape"]
    grid = np.zeros((height, width), dtype=int)

    for axis, index in spec["paint_order"]:
        if axis == "h":
            bar = spec["horizontal"][index]
            grid[bar.lo : bar.hi + 1, :] = bar.color
        else:
            bar = spec["vertical"][index]
            grid[:, bar.lo : bar.hi + 1] = bar.color

    for col, v_idx in spec["pixels"]:
        grid[0, col] = spec["vertical"][v_idx].color
    return grid


def _expected_output_from_spec(spec: dict, input_grid: np.ndarray) -> np.ndarray:
    scene = _analyze_scene(input_grid)
    landing_rows: dict[tuple[int, int], int] = {}
    for col, v_idx in spec["pixels"]:
        color = spec["vertical"][v_idx].color
        stop_index = int(spec["stop_indices"][v_idx])
        if stop_index < len(spec["horizontal"]):
            row = spec["horizontal"][stop_index].lo - 1
        else:
            row = spec["shape"][0] - 1
        landing_rows[(int(col), int(color))] = int(row)
    return _apply_landings(input_grid, scene, landing_rows)


def _heuristic_first_bar(grid: np.ndarray, scene: Scene) -> np.ndarray:
    first_row = scene.horizontal[0].lo - 1
    landing_rows = {(c, color): int(first_row) for c, color in scene.pixels}
    return _apply_landings(grid, scene, landing_rows)


def _heuristic_all_bottom(grid: np.ndarray, scene: Scene) -> np.ndarray:
    bottom_row = int(grid.shape[0] - 1)
    landing_rows = {(c, color): bottom_row for c, color in scene.pixels}
    return _apply_landings(grid, scene, landing_rows)


def _heuristic_inverted_crossings(grid: np.ndarray, scene: Scene) -> np.ndarray:
    vertical_by_color = {bar.color: bar for bar in scene.vertical}
    landing_rows: dict[tuple[int, int], int] = {}
    for c, color in scene.pixels:
        vbar = vertical_by_color[color]
        row = int(grid.shape[0] - 1)
        for hbar in scene.horizontal:
            if _cross_top_color(grid, hbar, vbar) == vbar.color:
                row = hbar.lo - 1
                break
        landing_rows[(c, color)] = int(row)
    return _apply_landings(grid, scene, landing_rows)


def _spec_matches_scene(spec: dict, scene: Scene) -> bool:
    expected_h = [(bar.lo, bar.hi, bar.color) for bar in spec["horizontal"]]
    expected_v = [(bar.lo, bar.hi, bar.color) for bar in spec["vertical"]]
    actual_h = [(bar.lo, bar.hi, bar.color) for bar in scene.horizontal]
    actual_v = [(bar.lo, bar.hi, bar.color) for bar in scene.vertical]
    expected_pixels = sorted((int(col), spec["vertical"][v_idx].color) for col, v_idx in spec["pixels"])
    return actual_h == expected_h and actual_v == expected_v and list(scene.pixels) == expected_pixels


def _validate_generated_case(spec: dict, input_grid: np.ndarray) -> bool:
    try:
        scene = _analyze_scene(input_grid)
    except AssertionError:
        return False

    if not _spec_matches_scene(spec, scene):
        return False

    expected = _expected_output_from_spec(spec, input_grid)
    try:
        solved = main(input_grid)
    except AssertionError:
        return False
    if not np.array_equal(solved, expected):
        return False

    pixel_stop_indices = [int(spec["stop_indices"][v_idx]) for _col, v_idx in spec["pixels"]]
    has_first = any(idx == 0 for idx in pixel_stop_indices)
    has_later = any(0 < idx < len(spec["horizontal"]) for idx in pixel_stop_indices)
    has_bottom = any(idx == len(spec["horizontal"]) for idx in pixel_stop_indices)
    if not (has_first and has_later and has_bottom):
        return False

    if np.array_equal(expected, _heuristic_first_bar(input_grid, scene)):
        return False
    if np.array_equal(expected, _heuristic_all_bottom(input_grid, scene)):
        return False
    if np.array_equal(expected, _heuristic_inverted_crossings(input_grid, scene)):
        return False

    return True


def _sample_spec() -> dict | None:
    n_h = random.randint(H_COUNT_RANGE[0], H_COUNT_RANGE[1])
    n_v = random.randint(V_COUNT_RANGE[0], V_COUNT_RANGE[1])
    if n_h + n_v > len(BAR_COLORS):
        return None

    height = random.randint(HEIGHT_RANGE[0], HEIGHT_RANGE[1])
    width = random.randint(WIDTH_RANGE[0], WIDTH_RANGE[1])

    h_intervals = _sample_intervals(
        height,
        n_h,
        thickness_choices=THICKNESS_CHOICES,
        start_margin=TOP_MARGIN,
        end_margin=BOTTOM_MARGIN,
        gap=MIN_BAR_GAP,
    )
    v_intervals = _sample_intervals(
        width,
        n_v,
        thickness_choices=THICKNESS_CHOICES,
        start_margin=SIDE_MARGIN,
        end_margin=SIDE_MARGIN,
        gap=MIN_BAR_GAP,
    )
    if h_intervals is None or v_intervals is None:
        return None

    colors = random.sample(BAR_COLORS, n_h + n_v)
    horizontal = tuple(SpanBar("h", r0, r1, colors[idx]) for idx, (r0, r1) in enumerate(h_intervals))
    vertical = tuple(SpanBar("v", c0, c1, colors[n_h + idx]) for idx, (c0, c1) in enumerate(v_intervals))

    labels = [("h", idx) for idx in range(n_h)] + [("v", idx) for idx in range(n_v)]
    paint_order = random.sample(labels, len(labels))
    rank = {label: order for order, label in enumerate(paint_order)}
    horizontal_ranks = [rank[("h", idx)] for idx in range(n_h)]
    stop_indices = [int(_stop_index_for_vertical(horizontal_ranks, rank[("v", idx)])) for idx in range(n_v)]

    first_choices = [idx for idx, stop in enumerate(stop_indices) if stop == 0]
    later_choices = [idx for idx, stop in enumerate(stop_indices) if 0 < stop < n_h]
    bottom_choices = [idx for idx, stop in enumerate(stop_indices) if stop == n_h]
    if not first_choices or not later_choices or not bottom_choices:
        return None

    pixel_columns = _candidate_pixel_columns(width, v_intervals)
    base_verticals = [
        random.choice(first_choices),
        random.choice(later_choices),
        random.choice(bottom_choices),
    ]
    duplicate_choice = random.choice(base_verticals)
    extra_count = random.randint(0, 2)
    pixel_verticals = base_verticals + [duplicate_choice]
    population = list(range(n_v))
    for _ in range(extra_count):
        pixel_verticals.append(random.choice(population))

    if len(pixel_columns) < len(pixel_verticals):
        return None

    random.shuffle(pixel_verticals)
    pixels = tuple((int(col), int(v_idx)) for col, v_idx in zip(pixel_columns[: len(pixel_verticals)], pixel_verticals))

    return {
        "shape": (height, width),
        "horizontal": horizontal,
        "vertical": vertical,
        "paint_order": paint_order,
        "stop_indices": tuple(stop_indices),
        "pixels": pixels,
    }


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        spec = _sample_spec()
        if spec is None:
            continue
        input_grid = _build_input(spec)
        if _validate_generated_case(spec, input_grid):
            return input_grid
    raise RuntimeError(f"generate_input: failed after {GEN_RETRY_LIMIT} attempts")
