from __future__ import annotations

from collections import Counter
from pathlib import Path
import random
import sys

import numpy as np


_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_UTILS_DIR) not in sys.path:
    sys.path.append(str(_UTILS_DIR))

from arc_utils import bbox, colors_present, crop_bbox, find_components, get_bg, paste


def _split_legend_and_canvas(grid: np.ndarray) -> tuple[int, np.ndarray, np.ndarray]:
    bg = int(get_bg(grid))
    sep_row = None
    seen_legend_fg = False
    for r in range(grid.shape[0]):
        row_is_bg = bool(np.all(grid[r] == bg))
        if not seen_legend_fg:
            seen_legend_fg = not row_is_bg
            continue
        if row_is_bg and r + 1 < grid.shape[0] and np.any(grid[r + 1 :] != bg):
            sep_row = r
            break
    assert sep_row is not None, "expected a background separator row below the legend"
    legend = grid[:sep_row, :]
    canvas = grid[sep_row + 1 :, :]
    assert legend.size and canvas.size
    return bg, legend, canvas


def _extract_legend_colors(legend: np.ndarray, bg: int) -> set[int]:
    non_bg_box = bbox(legend != bg)
    assert non_bg_box is not None, "legend frame must exist"
    crop = crop_bbox(legend, non_bg_box)
    assert crop.shape[0] >= 3 and crop.shape[1] >= 3, "legend frame must surround an interior"

    border_color = int(crop[0, 0])
    assert border_color != bg, "legend border must be non-background"
    assert np.all(crop[0, :] == border_color)
    assert np.all(crop[-1, :] == border_color)
    assert np.all(crop[:, 0] == border_color)
    assert np.all(crop[:, -1] == border_color)

    interior = crop[1:-1, 1:-1]
    legend_colors = set(colors_present(interior, exclude={bg}))
    assert legend_colors, "legend interior must list at least one defect color"
    return legend_colors


def _orbit_cells(r: int, c: int, h: int, w: int) -> tuple[tuple[int, int], ...]:
    cells = {
        (int(r), int(c)),
        (int(r), int(w - 1 - c)),
        (int(h - 1 - r), int(c)),
        (int(h - 1 - r), int(w - 1 - c)),
    }
    return tuple(sorted(cells))


def _resolve_orbit_color(values: list[int], legend_colors: set[int]) -> int:
    counts = Counter(map(int, values))
    if len(counts) == 1:
        return int(values[0])
    assert sorted(counts.values()) == [1, len(values) - 1], "each broken orbit must have one odd defect cell"

    minority = next(color for color, count in counts.items() if count == 1)
    majority = next(color for color, count in counts.items() if count == len(values) - 1)
    return int(majority if minority in legend_colors else minority)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repair or spread each asymmetry defect according to the legend, then return only the restored canvas.

    Input structure:
    - The top region is a legend: a single-color rectangular frame encloses one or more sample colors.
    - One full background row separates that legend from the main canvas below.
    - The main canvas is based on a pattern that is symmetric across both the horizontal and vertical midlines, except for several monochrome defect patches that appear in exactly one orbit position.

    Rule:
    - Read the set of colors listed inside the legend frame.
    - For every four-cell mirror orbit on the main canvas, inspect its visible colors. A broken orbit has three equal cells and one odd cell.
    - If the odd color is listed in the legend, treat it as a stain and restore the whole orbit to the three matching base values. Otherwise treat it as a new element and copy that odd color to the full orbit. Output only the processed main canvas.

    Key insight:
    - Symmetry alone does not tell whether an odd cell should disappear or replicate; the legend colors are the only signal that distinguishes repair defects from spread defects.
    """
    g = np.asarray(grid, dtype=int)
    bg, legend, canvas = _split_legend_and_canvas(g)
    legend_colors = _extract_legend_colors(legend, bg)

    h, w = canvas.shape
    out = canvas.copy()
    seen: set[tuple[tuple[int, int], ...]] = set()
    for r in range(h):
        for c in range(w):
            orbit = _orbit_cells(r, c, h, w)
            if orbit in seen:
                continue
            seen.add(orbit)
            values = [int(canvas[rr, cc]) for rr, cc in orbit]
            fill = _resolve_orbit_color(values, legend_colors)
            for rr, cc in orbit:
                out[rr, cc] = fill
    return out


BG = 0
MAX_ATTEMPTS = 400
CANVAS_H_RANGE = (10, 18)
CANVAS_W_RANGE = (10, 18)
BASE_COLOR_COUNT_RANGE = (2, 3)
BASE_SHAPE_COUNT_BONUS = 2
STAIN_COUNT_RANGE = (1, 2)
SPREAD_COUNT_RANGE = (1, 2)

SHAPE_SEEDS = (
    ((0, 0),),
    ((0, 0), (0, 1)),
    ((0, 0), (1, 0)),
    ((0, 0), (0, 1), (1, 0)),
    ((0, 0), (0, 1), (1, 1)),
    ((0, 0), (1, 0), (1, 1)),
    ((0, 0), (0, 1), (0, 2)),
    ((0, 0), (1, 0), (2, 0)),
    ((0, 0), (0, 1), (1, 0), (1, 1)),
    ((0, 1), (1, 0), (1, 1)),
)


def _normalize_cells(cells: list[tuple[int, int]] | tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    min_r = min(rows)
    min_c = min(cols)
    return tuple(sorted((int(r - min_r), int(c - min_c)) for r, c in cells))


def _shape_variants(seed: tuple[tuple[int, int], ...]) -> list[tuple[tuple[int, int], ...]]:
    variants: set[tuple[tuple[int, int], ...]] = set()
    work = list(seed)
    for flip in (False, True):
        cur = [(r, -c) for r, c in work] if flip else list(work)
        for _ in range(4):
            variants.add(_normalize_cells(cur))
            cur = [(c, -r) for r, c in cur]
    return sorted(variants)


ALL_SHAPES = sorted({variant for seed in SHAPE_SEEDS for variant in _shape_variants(seed)})
BASE_SHAPES = [shape for shape in ALL_SHAPES if len(shape) >= 2]
DEFECT_SHAPES = [shape for shape in ALL_SHAPES if len(shape) <= 4]


def _fit_with_gap(occ: np.ndarray, cells: tuple[tuple[int, int], ...], gap: int = 1) -> bool:
    h, w = occ.shape
    for r, c in cells:
        if not (0 <= r < h and 0 <= c < w):
            return False
        r0 = max(0, r - gap)
        r1 = min(h, r + gap + 1)
        c0 = max(0, c - gap)
        c1 = min(w, c + gap + 1)
        if occ[r0:r1, c0:c1].any():
            return False
    return True


def _mark_cells(occ: np.ndarray, cells: tuple[tuple[int, int], ...]) -> None:
    for r, c in cells:
        occ[r, c] = True


def _sample_local_shape(
    rng: random.Random,
    variants: list[tuple[tuple[int, int], ...]],
    occ: np.ndarray,
    *,
    gap: int = 1,
) -> tuple[tuple[int, int], ...] | None:
    local_h, local_w = occ.shape
    choices = variants[:]
    rng.shuffle(choices)
    for shape in choices:
        shape_h = max(r for r, _ in shape) + 1
        shape_w = max(c for _, c in shape) + 1
        if shape_h > local_h or shape_w > local_w:
            continue
        anchors = [(r, c) for r in range(local_h - shape_h + 1) for c in range(local_w - shape_w + 1)]
        rng.shuffle(anchors)
        for r0, c0 in anchors:
            cells = tuple(sorted((r0 + r, c0 + c) for r, c in shape))
            if _fit_with_gap(occ, cells, gap=gap):
                return cells
    return None


def _map_local_to_quadrant(cell: tuple[int, int], h: int, w: int, quadrant: int) -> tuple[int, int]:
    r, c = cell
    if quadrant == 0:
        return r, c
    if quadrant == 1:
        return r, w - 1 - c
    if quadrant == 2:
        return h - 1 - r, c
    return h - 1 - r, w - 1 - c


def _paint_orbit(grid: np.ndarray, cell: tuple[int, int], color: int) -> None:
    h, w = grid.shape
    for rr, cc in _orbit_cells(cell[0], cell[1], h, w):
        grid[rr, cc] = int(color)


def _is_double_symmetric(grid: np.ndarray) -> bool:
    return bool(np.array_equal(grid, grid[::-1, :]) and np.array_equal(grid, grid[:, ::-1]))


def _build_base_canvas(rng: random.Random, h: int, w: int, base_colors: list[int]) -> np.ndarray | None:
    local_h = h // 2
    local_w = w // 2
    if local_h < 2 or local_w < 2:
        return None

    local_occ = np.zeros((local_h, local_w), dtype=bool)
    items: list[tuple[tuple[tuple[int, int], ...], int]] = []
    target_shapes = rng.randint(len(base_colors), len(base_colors) + BASE_SHAPE_COUNT_BONUS)
    forced_colors = rng.sample(base_colors, len(base_colors))

    for idx in range(target_shapes):
        cells = _sample_local_shape(rng, BASE_SHAPES, local_occ, gap=1)
        if cells is None:
            return None
        _mark_cells(local_occ, cells)
        color = forced_colors[idx] if idx < len(forced_colors) else rng.choice(base_colors)
        items.append((cells, color))

    canvas = np.full((h, w), BG, dtype=int)
    for cells, color in items:
        for cell in cells:
            _paint_orbit(canvas, cell, color)

    if len(colors_present(canvas, exclude={BG})) < 2:
        return None
    if np.count_nonzero(canvas != BG) >= canvas.size // 2:
        return None
    return canvas


def _build_legend(width: int, border_color: int, legend_colors: list[int], rng: random.Random) -> np.ndarray:
    inner_w = 2 * len(legend_colors) - 1
    patch = np.full((3, inner_w + 2), BG, dtype=int)
    patch[0, :] = border_color
    patch[-1, :] = border_color
    patch[:, 0] = border_color
    patch[:, -1] = border_color
    colors = legend_colors[:]
    rng.shuffle(colors)
    for idx, color in enumerate(colors):
        patch[1, 1 + 2 * idx] = int(color)

    legend = np.full((3, width), BG, dtype=int)
    start_col = rng.randint(0, width - patch.shape[1])
    paste(legend, patch, 0, start_col, transparent=BG)
    return legend


def _validate_input_components(canvas: np.ndarray, defect_colors: set[int]) -> bool:
    components = find_components(canvas, bg=BG, connectivity=4, monochromatic=True, return_format="dict")
    counts = Counter(comp["color"] for comp in components)
    return all(counts.get(int(color), 0) == 1 for color in defect_colors)


def _validate_case(inp: np.ndarray, expected: np.ndarray, stain_colors: list[int], defect_colors: set[int]) -> bool:
    if inp.ndim != 2 or expected.ndim != 2:
        return False
    if max(inp.shape) > 30 or max(expected.shape) > 30:
        return False
    if not _is_double_symmetric(expected):
        return False
    if int(get_bg(inp)) != BG:
        return False

    bg, legend, canvas = _split_legend_and_canvas(inp)
    if bg != BG:
        return False
    if _extract_legend_colors(legend, bg) != set(stain_colors):
        return False
    if not _validate_input_components(canvas, defect_colors):
        return False

    stain_orbits = 0
    spread_orbits = 0
    seen: set[tuple[tuple[int, int], ...]] = set()
    for r in range(canvas.shape[0]):
        for c in range(canvas.shape[1]):
            orbit = _orbit_cells(r, c, canvas.shape[0], canvas.shape[1])
            if orbit in seen:
                continue
            seen.add(orbit)
            values = [int(canvas[rr, cc]) for rr, cc in orbit]
            counts = Counter(values)
            if len(counts) == 1:
                continue
            if len(orbit) != 4 or sorted(counts.values()) != [1, 3]:
                return False
            minority = next(color for color, count in counts.items() if count == 1)
            majority = next(color for color, count in counts.items() if count == 3)
            out_values = {int(expected[rr, cc]) for rr, cc in orbit}
            if len(out_values) != 1:
                return False
            out_color = next(iter(out_values))
            if minority in stain_colors:
                stain_orbits += 1
                if out_color != majority:
                    return False
            else:
                spread_orbits += 1
                if out_color != minority:
                    return False

    if stain_orbits == 0 or spread_orbits == 0:
        return False
    return np.array_equal(main(inp), expected)


def _generate_case() -> tuple[np.ndarray, np.ndarray] | None:
    rng = random.Random()
    h = rng.randint(*CANVAS_H_RANGE)
    w = rng.randint(*CANVAS_W_RANGE)

    stain_count = rng.randint(*STAIN_COUNT_RANGE)
    spread_count = rng.randint(*SPREAD_COUNT_RANGE)
    base_count = rng.randint(*BASE_COLOR_COUNT_RANGE)
    needed = 1 + stain_count + spread_count + base_count
    if needed > 9:
        return None

    palette = rng.sample(list(range(1, 10)), needed)
    border_color = palette[0]
    base_colors = palette[1 : 1 + base_count]
    stain_colors = palette[1 + base_count : 1 + base_count + stain_count]
    spread_colors = palette[1 + base_count + stain_count :]

    base = _build_base_canvas(rng, h, w, base_colors)
    if base is None:
        return None

    local_h = h // 2
    local_w = w // 2
    defect_occ = np.zeros((local_h, local_w), dtype=bool)
    inp_canvas = base.copy()
    out_canvas = base.copy()

    tasks = [("stain", color) for color in stain_colors] + [("spread", color) for color in spread_colors]
    rng.shuffle(tasks)

    defect_colors: set[int] = set()
    for kind, color in tasks:
        placed = False
        for _ in range(80):
            cells = _sample_local_shape(rng, DEFECT_SHAPES, defect_occ, gap=1)
            if cells is None:
                break
            underlying = [int(base[r, c]) for r, c in cells]
            if kind == "stain" and not any(value != BG for value in underlying):
                continue

            quadrant = rng.randrange(4)
            actual_cells = tuple(sorted(_map_local_to_quadrant(cell, h, w, quadrant) for cell in cells))
            if any(inp_canvas[r, c] in defect_colors for r, c in actual_cells):
                continue

            _mark_cells(defect_occ, cells)
            defect_colors.add(int(color))
            for r, c in actual_cells:
                inp_canvas[r, c] = int(color)
            if kind == "spread":
                for cell in cells:
                    _paint_orbit(out_canvas, cell, color)
            placed = True
            break
        if not placed:
            return None

    legend = _build_legend(w, border_color, stain_colors, rng)
    separator = np.full((1, w), BG, dtype=int)
    inp = np.vstack([legend, separator, inp_canvas])
    if not _validate_case(inp, out_canvas, stain_colors, defect_colors):
        return None
    return inp, out_canvas


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        case = _generate_case()
        if case is not None:
            return case[0]
    raise RuntimeError("failed to generate a valid legend symmetry defect puzzle")
