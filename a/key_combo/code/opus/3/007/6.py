from __future__ import annotations

import itertools
from typing import Iterable

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
TEAL = 8

NEIGHBORS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
PALETTE = (BLUE, RED, GREEN, YELLOW)


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components(mask: np.ndarray) -> list[np.ndarray]:
    seen = np.zeros(mask.shape, dtype=bool)
    comps: list[np.ndarray] = []
    h, w = mask.shape
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in NEIGHBORS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.asarray(coords, dtype=int))
    return comps


def _find_color_components(grid: np.ndarray, color: int) -> list[dict]:
    comps = []
    for coords in _components(grid == int(color)):
        comps.append(
            {
                "coords": coords,
                "bbox": _bbox_coords(coords),
                "area": int(coords.shape[0]),
                "color": int(color),
            }
        )
    return comps


def _normalize_shape(coords: Iterable[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    arr = np.asarray(list(coords), dtype=int)
    arr -= arr.min(axis=0)
    return tuple(sorted((int(r), int(c)) for r, c in arr))


def _shape_dims(shape: tuple[tuple[int, int], ...]) -> tuple[int, int]:
    rows = [r for r, _ in shape]
    cols = [c for _, c in shape]
    return max(rows) + 1, max(cols) + 1


def _shape_variants(shape: tuple[tuple[int, int], ...]) -> list[tuple[tuple[int, int], ...]]:
    h, w = _shape_dims(shape)
    variants = {
        _normalize_shape((r, c) for r, c in shape),
        _normalize_shape((c, h - 1 - r) for r, c in shape),
        _normalize_shape((h - 1 - r, w - 1 - c) for r, c in shape),
        _normalize_shape((w - 1 - c, r) for r, c in shape),
        _normalize_shape((r, w - 1 - c) for r, c in shape),
        _normalize_shape((h - 1 - r, c) for r, c in shape),
        _normalize_shape((c, r) for r, c in shape),
        _normalize_shape((w - 1 - c, h - 1 - r) for r, c in shape),
    }
    return sorted(variants)


def _orbit_key(shape: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    return min(_shape_variants(shape))


def _paint_shape(grid: np.ndarray, shape: tuple[tuple[int, int], ...], r0: int, c0: int, color: int) -> None:
    for dr, dc in shape:
        grid[int(r0) + int(dr), int(c0) + int(dc)] = int(color)


def _random_composition(total: int, parts: int, rng: np.random.Generator) -> list[int]:
    if parts == 1:
        return [int(total)]
    probs = np.full(parts, 1.0 / parts)
    return [int(x) for x in rng.multinomial(int(total), probs)]


def _place_rows(
    shapes: list[tuple[tuple[int, int], ...]],
    total_h: int,
    min_gap: int,
    rng: np.random.Generator,
) -> list[int] | None:
    heights = [_shape_dims(shape)[0] for shape in shapes]
    need = sum(heights) + int(min_gap) * (len(shapes) - 1)
    extra = int(total_h) - int(need)
    if extra < 0:
        return None
    pads = _random_composition(extra, len(shapes) + 1, rng)
    rows = []
    r = pads[0]
    for i, height in enumerate(heights):
        rows.append(int(r))
        r += int(height)
        if i + 1 < len(shapes):
            r += int(min_gap) + pads[i + 1]
    return rows


def _panel_width(grid: np.ndarray) -> int:
    w = 0
    while w < grid.shape[1] and np.all(grid[:, w] != BLACK):
        w += 1
    if w < 4:
        raise AssertionError("failed to locate the left legend panel")
    return int(w)


def _render_lines(
    work_shape: tuple[int, int],
    gray_components: list[dict],
    area_to_color: dict[int, int],
) -> np.ndarray:
    out = np.full(work_shape, BLACK, dtype=int)
    _, width = work_shape
    c_center = width // 2
    for comp in gray_components:
        area = int(comp["area"])
        color = int(area_to_color[area])
        r0, _, r1, _ = comp["bbox"]
        center_row = (int(r0) + int(r1)) // 2
        start = c_center - area // 2
        end = start + area
        start = max(0, start)
        end = min(width, end)
        out[center_row, start:end] = color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Replace each gray component by a centered horizontal line whose color is looked up from a left legend by matching pixel count.

    Input structure:
    - The grid has a black background and a full-height teal legend panel on the left.
    - Inside the panel are 3-4 small single-color 4-connected shapes, one per color.
    - To the right are 3-4 gray 4-connected regions whose areas match the legend-shape areas.

    Rule:
    - Build a lookup table from each legend shape's pixel count to that shape's color.
    - Crop away the left legend panel and keep only the right work area.
    - For every gray component in the work area, count its pixels, look up the corresponding color, and draw a horizontal line of that length on the component bbox's center row, centered in the output width.

    Key insight:
    - The legend uses area, not shape, as the code: only the pixel count determines the replacement color, while the output geometry is always a centered straight line.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    panel_w = _panel_width(g)
    panel = g[:, :panel_w]
    work = g[:, panel_w:]

    area_to_color: dict[int, int] = {}
    legend_colors = sorted(set(map(int, np.unique(panel))) - {TEAL})
    assert legend_colors, "missing legend entries"
    for color in legend_colors:
        comps = _find_color_components(panel, color)
        assert len(comps) == 1, "each legend color must form one component"
        area = int(comps[0]["area"])
        assert area not in area_to_color, "legend areas must be unique"
        area_to_color[area] = int(color)

    gray_components = _find_color_components(work, GREY)
    assert gray_components, "missing gray work components"
    assert {int(comp["area"]) for comp in gray_components} == set(area_to_color), "work areas must match legend areas"

    return _render_lines(work.shape, gray_components, area_to_color)


BASE_SHAPES_BY_AREA: dict[int, list[tuple[tuple[int, int], ...]]] = {
    3: [
        ((0, 0), (1, 0), (1, 1)),
        ((0, 0), (0, 1), (0, 2)),
    ],
    4: [
        ((0, 0), (1, 0), (2, 0), (2, 1)),
        ((0, 0), (0, 1), (0, 2), (1, 1)),
        ((0, 1), (0, 2), (1, 0), (1, 1)),
        ((0, 0), (0, 1), (0, 2), (0, 3)),
    ],
    5: [
        ((0, 0), (0, 1), (1, 0), (1, 1), (2, 0)),
        ((0, 0), (0, 1), (0, 2), (1, 1), (2, 1)),
        ((0, 0), (0, 1), (1, 1), (1, 2), (2, 2)),
        ((0, 0), (0, 2), (1, 0), (1, 1), (1, 2)),
        ((0, 0), (1, 0), (2, 0), (3, 0), (3, 1)),
    ],
    6: [
        ((0, 0), (1, 0), (2, 0), (3, 0), (3, 1), (3, 2)),
        ((0, 0), (0, 1), (1, 1), (1, 2), (2, 2), (2, 3)),
        ((0, 0), (0, 1), (1, 0), (2, 0), (2, 1), (2, 2)),
        ((0, 0), (0, 1), (0, 2), (1, 1), (2, 1), (2, 2)),
    ],
}


def _catalog_ok() -> None:
    for area, shapes in BASE_SHAPES_BY_AREA.items():
        seen = set()
        for shape in shapes:
            shape = _normalize_shape(shape)
            assert len(shape) == area
            h, w = _shape_dims(shape)
            assert len(_components(_shape_to_mask(shape))) == 1
            assert not (h > 1 and w > 1 and h * w == area), "avoid fat rectangles"
            key = _orbit_key(shape)
            assert key not in seen, f"duplicate orbit in area {area}"
            seen.add(key)


def _shape_to_mask(shape: tuple[tuple[int, int], ...]) -> np.ndarray:
    h, w = _shape_dims(shape)
    mask = np.zeros((h, w), dtype=bool)
    for r, c in shape:
        mask[r, c] = True
    return mask


_catalog_ok()


def _sample_variant(
    base_shape: tuple[tuple[int, int], ...],
    rng: np.random.Generator,
    *,
    require_height: int | None = None,
    require_floor_centroid_mismatch: bool = False,
) -> tuple[tuple[int, int], ...] | None:
    candidates = []
    for variant in _shape_variants(_normalize_shape(base_shape)):
        h, _ = _shape_dims(variant)
        if require_height is not None and h != int(require_height):
            continue
        if require_floor_centroid_mismatch:
            rows = [r for r, _ in variant]
            if int(np.floor(np.mean(rows))) == (h - 1) // 2:
                continue
        candidates.append(variant)
    if not candidates:
        return None
    return candidates[int(rng.integers(len(candidates)))]


def _sample_area_set(rng: np.random.Generator) -> list[int]:
    if rng.random() < 0.5:
        return [3, 4, 5, 6]
    combos = ([3, 4, 6], [3, 5, 6], [4, 5, 6])
    return list(combos[int(rng.integers(len(combos)))])


def _sample_shapes(
    areas: list[int], rng: np.random.Generator
) -> tuple[dict[int, tuple[tuple[int, int], ...]], dict[int, tuple[tuple[int, int], ...]]] | None:
    legend_shapes: dict[int, tuple[tuple[int, int], ...]] = {}
    gray_shapes: dict[int, tuple[tuple[int, int], ...]] = {}

    tall_gray = _sample_variant(
        BASE_SHAPES_BY_AREA[6][0],
        rng,
        require_height=4,
        require_floor_centroid_mismatch=True,
    )
    if tall_gray is None:
        return None

    for area in areas:
        pool = BASE_SHAPES_BY_AREA[area]
        if area == 6:
            gray_shapes[area] = tall_gray
            legend_candidates = pool[1:]
            legend_base = legend_candidates[int(rng.integers(len(legend_candidates)))]
            legend_shape = _sample_variant(legend_base, rng)
            if legend_shape is None:
                return None
            legend_shapes[area] = legend_shape
            continue

        idx = rng.permutation(len(pool))
        first = pool[int(idx[0])]
        second = pool[int(idx[1])]
        legend_shape = _sample_variant(first, rng)
        gray_shape = _sample_variant(second, rng)
        if legend_shape is None or gray_shape is None:
            return None
        legend_shapes[area] = legend_shape
        gray_shapes[area] = gray_shape

    return legend_shapes, gray_shapes


def _build_input_and_output(
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, dict] | None:
    areas = _sample_area_set(rng)
    sampled = _sample_shapes(areas, rng)
    if sampled is None:
        return None
    legend_shapes, gray_shapes = sampled

    colors = rng.choice(np.asarray(PALETTE, dtype=int), size=len(areas), replace=False)
    area_to_color = {int(area): int(color) for area, color in zip(areas, colors)}

    legend_order = list(rng.permutation(areas))
    gray_order = list(rng.permutation(areas))
    if legend_order == gray_order:
        gray_order = gray_order[1:] + gray_order[:1]

    max_legend_w = max(_shape_dims(legend_shapes[area])[1] for area in areas)
    panel_w_min = max(5, max_legend_w + 2)
    panel_w_max = min(7, panel_w_min + 1)
    panel_w = int(rng.integers(panel_w_min, panel_w_max + 1))

    work_w = int(rng.integers(7, 13))

    legend_stack = [legend_shapes[area] for area in legend_order]
    gray_stack = [gray_shapes[area] for area in gray_order]
    legend_need = sum(_shape_dims(shape)[0] for shape in legend_stack) + (len(areas) - 1)
    gray_need = sum(_shape_dims(shape)[0] for shape in gray_stack) + 2 * (len(areas) - 1)
    h_min = max(13, legend_need, gray_need)
    h_max = min(24, h_min + 5)
    if h_min > h_max:
        return None
    height = int(rng.integers(h_min, h_max + 1))

    legend_rows = _place_rows(legend_stack, height, 1, rng)
    gray_rows = _place_rows(gray_stack, height, 2, rng)
    if legend_rows is None or gray_rows is None:
        return None

    grid = np.full((height, panel_w + work_w), BLACK, dtype=int)
    grid[:, :panel_w] = TEAL

    legend_meta = {}
    for row, area in zip(legend_rows, legend_order):
        shape = legend_shapes[area]
        sh, sw = _shape_dims(shape)
        c0 = int(rng.integers(1, panel_w - sw))
        _paint_shape(grid[:, :panel_w], shape, row, c0, area_to_color[area])
        legend_meta[area] = {"shape": shape, "row": int(row), "col": int(c0), "dims": (sh, sw)}

    gray_meta = {}
    for row, area in zip(gray_rows, gray_order):
        shape = gray_shapes[area]
        sh, sw = _shape_dims(shape)
        c0 = int(rng.integers(1, work_w - sw))
        _paint_shape(grid[:, panel_w:], shape, row, c0, GREY)
        gray_meta[area] = {"shape": shape, "row": int(row), "col": int(c0), "dims": (sh, sw)}

    if not np.all(np.any(grid[:, panel_w:] == BLACK, axis=0)):
        return None

    expected = np.full((height, work_w), BLACK, dtype=int)
    center_col = work_w // 2
    for area in areas:
        row = gray_meta[area]["row"]
        sh, _ = gray_meta[area]["dims"]
        center_row = row + (sh - 1) // 2
        start = center_col - area // 2
        end = min(work_w, start + area)
        expected[center_row, max(0, start):end] = area_to_color[area]

    meta = {
        "areas": areas,
        "area_to_color": area_to_color,
        "panel_w": panel_w,
        "legend_order": legend_order,
        "gray_order": gray_order,
        "legend_meta": legend_meta,
        "gray_meta": gray_meta,
    }
    return grid, expected, meta


def _order_shortcut_output(meta: dict, height: int, work_w: int) -> np.ndarray:
    wrong_map = {
        int(gray_area): int(meta["area_to_color"][legend_area])
        for legend_area, gray_area in zip(meta["legend_order"], meta["gray_order"])
    }
    comps = []
    for area in meta["areas"]:
        shape = meta["gray_meta"][area]["shape"]
        row = meta["gray_meta"][area]["row"]
        col = meta["gray_meta"][area]["col"]
        coords = np.asarray([(row + r, col + c) for r, c in shape], dtype=int)
        comps.append({"coords": coords, "bbox": _bbox_coords(coords), "area": int(area)})
    return _render_lines((height, work_w), comps, wrong_map)


def _centroid_shortcut_output(meta: dict, height: int, work_w: int) -> np.ndarray:
    out = np.full((height, work_w), BLACK, dtype=int)
    center_col = work_w // 2
    for area in meta["areas"]:
        shape = meta["gray_meta"][area]["shape"]
        row = meta["gray_meta"][area]["row"]
        rows = [row + r for r, _ in shape]
        wrong_row = int(np.floor(np.mean(rows)))
        start = center_col - area // 2
        end = min(work_w, start + area)
        out[wrong_row, max(0, start):end] = meta["area_to_color"][area]
    return out


def _validate_semantics(grid: np.ndarray, expected: np.ndarray, meta: dict) -> bool:
    height, total_w = grid.shape
    panel_w = _panel_width(grid)
    if panel_w != meta["panel_w"]:
        return False
    work = grid[:, panel_w:]
    if expected.shape != work.shape or total_w > 30 or height > 30:
        return False

    panel = grid[:, :panel_w]
    legend_map = {}
    for color in sorted(set(map(int, np.unique(panel))) - {TEAL}):
        comps = _find_color_components(panel, color)
        if len(comps) != 1:
            return False
        area = int(comps[0]["area"])
        if area in legend_map:
            return False
        legend_map[area] = int(color)
    if legend_map != meta["area_to_color"]:
        return False

    gray_comps = sorted(_find_color_components(work, GREY), key=lambda comp: comp["bbox"][0])
    if len(gray_comps) != len(meta["areas"]):
        return False
    if {int(comp["area"]) for comp in gray_comps} != set(meta["areas"]):
        return False

    for left, right in zip(gray_comps, gray_comps[1:]):
        if right["bbox"][0] - left["bbox"][2] < 3:
            return False

    for area in meta["areas"]:
        legend_orbit = _orbit_key(meta["legend_meta"][area]["shape"])
        gray_orbit = _orbit_key(meta["gray_meta"][area]["shape"])
        if legend_orbit == gray_orbit:
            return False

    if meta["legend_order"] == meta["gray_order"]:
        return False

    if np.array_equal(_order_shortcut_output(meta, height, work.shape[1]), expected):
        return False
    if np.array_equal(_centroid_shortcut_output(meta, height, work.shape[1]), expected):
        return False

    independent = _render_lines(work.shape, gray_comps, legend_map)
    if not np.array_equal(independent, expected):
        return False

    try:
        actual = main(grid)
    except Exception:
        return False
    return np.array_equal(actual, expected)


def generate_input() -> np.ndarray:
    for _ in range(400):
        rng = np.random.default_rng()
        built = _build_input_and_output(rng)
        if built is None:
            continue
        grid, expected, meta = built
        if _validate_semantics(grid, expected, meta):
            return grid
    raise RuntimeError("failed to generate a valid input")
