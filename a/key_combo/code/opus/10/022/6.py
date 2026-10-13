from __future__ import annotations

import itertools

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
GREY = 5
MARKER_COLORS = (BLUE, RED, GREEN)

_RNG = np.random.default_rng()
_COUNT_PATTERNS = {
    2: ((2, 3), (2, 4)),
    3: ((1, 2, 3), (1, 2, 4), (2, 3, 4)),
}
_MAX_GENERATION_ATTEMPTS = 600


def _rect_area(rect: tuple[int, int, int, int]) -> int:
    r0, c0, r1, c1 = rect
    return (r1 - r0 + 1) * (c1 - c0 + 1)


def _rect_cells(rect: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = rect
    return [(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)]


def _cell_center(cell: tuple[int, int]) -> tuple[int, int]:
    r, c = cell
    return 2 * r + 1, 2 * c + 1


def _rect_center_pixels(rect: tuple[int, int, int, int]) -> set[tuple[int, int]]:
    return {_cell_center(cell) for cell in _rect_cells(rect)}


def _rect_perimeter_pixels(rect: tuple[int, int, int, int]) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = rect
    top = 2 * r0
    bottom = 2 * (r1 + 1)
    left = 2 * c0
    right = 2 * (c1 + 1)
    pixels: set[tuple[int, int]] = set()
    for c in range(left, right + 1):
        pixels.add((top, c))
        pixels.add((bottom, c))
    for r in range(top, bottom + 1):
        pixels.add((r, left))
        pixels.add((r, right))
    return pixels


def _priority_key(spec: dict) -> tuple[int, int]:
    return int(spec["count"]), -int(spec["first_scan"])


def _infer_lattice(g: np.ndarray) -> tuple[list[int], list[int], list[int], list[int]]:
    row_bounds = [r for r in range(g.shape[0]) if np.all(g[r, :] == GREY)]
    col_bounds = [c for c in range(g.shape[1]) if np.all(g[:, c] == GREY)]
    assert len(row_bounds) >= 2 and len(col_bounds) >= 2
    assert row_bounds[0] == 0 and row_bounds[-1] == g.shape[0] - 1
    assert col_bounds[0] == 0 and col_bounds[-1] == g.shape[1] - 1
    assert all(b - a == 2 for a, b in zip(row_bounds, row_bounds[1:]))
    assert all(b - a == 2 for a, b in zip(col_bounds, col_bounds[1:]))
    center_rows = [r + 1 for r in row_bounds[:-1]]
    center_cols = [c + 1 for c in col_bounds[:-1]]
    return row_bounds, col_bounds, center_rows, center_cols


def _extract_specs(g: np.ndarray) -> tuple[list[dict], list[int], list[int]]:
    row_bounds, col_bounds, center_rows, center_cols = _infer_lattice(g)
    center_row_to_idx = {r: i for i, r in enumerate(center_rows)}
    center_col_to_idx = {c: i for i, c in enumerate(center_cols)}
    row_bound_set = set(row_bounds)
    col_bound_set = set(col_bounds)

    grouped: dict[int, list[tuple[int, int, int]]] = {}
    for r in range(g.shape[0]):
        for c in range(g.shape[1]):
            value = int(g[r, c])
            on_boundary = r in row_bound_set or c in col_bound_set
            if on_boundary:
                assert value == GREY
                continue
            assert r in center_row_to_idx and c in center_col_to_idx
            assert value in (BLACK, *MARKER_COLORS)
            if value != BLACK:
                grouped.setdefault(value, []).append(
                    (center_row_to_idx[r], center_col_to_idx[c], r * g.shape[1] + c)
                )

    marker_colors = sorted(grouped)
    assert 2 <= len(marker_colors) <= 3

    specs = []
    for color in marker_colors:
        entries = grouped[color]
        rows = [r for r, _c, _flat in entries]
        cols = [c for _r, c, _flat in entries]
        specs.append(
            {
                "color": color,
                "count": len(entries),
                "rect": (min(rows), min(cols), max(rows), max(cols)),
                "markers": tuple(sorted((r, c) for r, c, _flat in entries)),
                "first_scan": min(flat for _r, _c, flat in entries),
            }
        )
    return specs, row_bounds, col_bounds


def _render_output(input_grid: np.ndarray, specs: list[dict]) -> np.ndarray:
    out = np.asarray(input_grid, dtype=int).copy()
    for spec in sorted(specs, key=_priority_key):
        for r, c in _rect_perimeter_pixels(spec["rect"]):
            out[r, c] = int(spec["color"])
        if int(spec["count"]) >= 3:
            for r, c in _rect_center_pixels(spec["rect"]):
                out[r, c] = int(spec["color"])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Expand scaffold-center markers into count-driven rectangles with count-driven overlap priority.

    Input structure:
    - A black background is subdivided by a grey scaffold lattice made of shared-border hollow 3x3 cells.
    - The only non-grey, non-black pixels are 2-3 marker colors placed at cell-center positions.
    - Each marker color appears in one or more distinct scaffold cells.

    Rule:
    - Infer the lattice boundary rows and columns from the fully grey scaffold lines.
    - For each marker color, count its markers and take the smallest rectangle of scaffold cells covering them.
    - Always recolor that rectangle's scaffold perimeter; if the count is at least 3, also fill every enclosed cell center.
    - When colored rectangles overlap on a perimeter segment or center, the color with the larger marker count wins.

    Key insight:
    - The marker count is reused twice: it selects outline-versus-fill behavior and simultaneously determines overlap priority.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    specs, _row_bounds, _col_bounds = _extract_specs(g)
    return _render_output(g, specs)


def _build_base_grid(cell_rows: int, cell_cols: int) -> np.ndarray:
    grid = np.full((2 * cell_rows + 1, 2 * cell_cols + 1), GREY, dtype=int)
    grid[1::2, 1::2] = BLACK
    return grid


def _rect_options(count: int, cell_rows: int, cell_cols: int) -> list[tuple[int, int]]:
    dims = []
    for height in range(1, min(cell_rows, 5) + 1):
        for width in range(1, min(cell_cols, 5) + 1):
            area = height * width
            if count == 1:
                valid = area == 1
            elif count == 2:
                valid = area >= 3
            else:
                valid = count + 1 <= area <= count + 4
            if valid:
                dims.append((height, width))
    return dims


def _sample_adjacent_pair(
    count_a: int,
    count_b: int,
    cell_rows: int,
    cell_cols: int,
) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int]] | None:
    dims_a = _rect_options(count_a, cell_rows, cell_cols)
    dims_b = _rect_options(count_b, cell_rows, cell_cols)
    if not dims_a or not dims_b:
        return None

    for _ in range(240):
        ha, wa = dims_a[int(_RNG.integers(len(dims_a)))]
        hb, wb = dims_b[int(_RNG.integers(len(dims_b)))]
        if int(_RNG.integers(2)) == 0:
            if ha + hb > cell_rows:
                continue
            top_a = int(_RNG.integers(0, cell_rows - (ha + hb) + 1))
            top_b = top_a + ha
            left_a = int(_RNG.integers(0, cell_cols - wa + 1))
            low = max(0, left_a - wb + 1)
            high = min(cell_cols - wb, left_a + wa - 1)
            if low > high:
                continue
            left_b = int(_RNG.integers(low, high + 1))
            rect_a = (top_a, left_a, top_a + ha - 1, left_a + wa - 1)
            rect_b = (top_b, left_b, top_b + hb - 1, left_b + wb - 1)
        else:
            if wa + wb > cell_cols:
                continue
            left_a = int(_RNG.integers(0, cell_cols - (wa + wb) + 1))
            left_b = left_a + wa
            top_a = int(_RNG.integers(0, cell_rows - ha + 1))
            low = max(0, top_a - hb + 1)
            high = min(cell_rows - hb, top_a + ha - 1)
            if low > high:
                continue
            top_b = int(_RNG.integers(low, high + 1))
            rect_a = (top_a, left_a, top_a + ha - 1, left_a + wa - 1)
            rect_b = (top_b, left_b, top_b + hb - 1, left_b + wb - 1)

        if _rect_perimeter_pixels(rect_a) & _rect_perimeter_pixels(rect_b):
            return rect_a, rect_b
    return None


def _interiors_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _sample_free_rect(
    count: int,
    cell_rows: int,
    cell_cols: int,
    existing: list[tuple[int, int, int, int]],
) -> tuple[int, int, int, int] | None:
    dims = _rect_options(count, cell_rows, cell_cols)
    if not dims:
        return None
    for _ in range(240):
        height, width = dims[int(_RNG.integers(len(dims)))]
        top = int(_RNG.integers(0, cell_rows - height + 1))
        left = int(_RNG.integers(0, cell_cols - width + 1))
        rect = (top, left, top + height - 1, left + width - 1)
        if all(not _interiors_overlap(rect, other) for other in existing):
            return rect
    return None


def _marker_score(combo: tuple[tuple[int, int], ...], rect: tuple[int, int, int, int]) -> tuple[int, int]:
    r0, c0, r1, c1 = rect
    corners = {(r0, c0), (r0, c1), (r1, c0), (r1, c1)}
    corner_hits = sum(cell in corners for cell in combo)
    spread = 0
    for a, b in itertools.combinations(combo, 2):
        spread += abs(a[0] - b[0]) + abs(a[1] - b[1])
    return corner_hits, spread


def _choose_markers(
    rect: tuple[int, int, int, int],
    count: int,
    used_cells: set[tuple[int, int]],
) -> tuple[tuple[int, int], ...] | None:
    cells = [cell for cell in _rect_cells(rect) if cell not in used_cells]
    if len(cells) < count:
        return None

    valid: list[tuple[tuple[int, int], ...]] = []
    for combo in itertools.combinations(cells, count):
        rows = [r for r, _c in combo]
        cols = [c for _r, c in combo]
        if (min(rows), min(cols), max(rows), max(cols)) == rect:
            valid.append(tuple(sorted(combo)))
    if not valid:
        return None

    best_score = max(_marker_score(combo, rect) for combo in valid)
    best = [combo for combo in valid if _marker_score(combo, rect) == best_score]
    return best[int(_RNG.integers(len(best)))]


def _visible_outline_hole(spec: dict, specs: list[dict], output: np.ndarray) -> bool:
    if int(spec["count"]) != 2:
        return False
    stronger_fill_rects = [
        other["rect"]
        for other in specs
        if int(other["count"]) >= 3 and _priority_key(other) > _priority_key(spec)
    ]
    marker_set = set(spec["markers"])
    for cell in _rect_cells(spec["rect"]):
        if cell in marker_set:
            continue
        if any(
            rect[0] <= cell[0] <= rect[2] and rect[1] <= cell[1] <= rect[3]
            for rect in stronger_fill_rects
        ):
            continue
        if output[_cell_center(cell)] == BLACK:
            return True
    return False


def _highest_fill_gain(specs: list[dict], input_grid: np.ndarray, output: np.ndarray) -> bool:
    fill_specs = [spec for spec in specs if int(spec["count"]) >= 3]
    if not fill_specs:
        return False
    strongest = max(fill_specs, key=_priority_key)
    marker_set = set(strongest["markers"])
    for cell in _rect_cells(strongest["rect"]):
        if cell in marker_set:
            continue
        center = _cell_center(cell)
        if input_grid[center] == BLACK and output[center] == int(strongest["color"]):
            return True
    return False


def _has_priority_overlap(specs: list[dict], output: np.ndarray) -> bool:
    for a, b in itertools.combinations(specs, 2):
        overlap = _rect_perimeter_pixels(a["rect"]) & _rect_perimeter_pixels(b["rect"])
        if not overlap:
            continue
        winner = a if _priority_key(a) > _priority_key(b) else b
        if any(output[pixel] == int(winner["color"]) for pixel in overlap):
            return True
    return False


def _spec_signature(specs: list[dict]) -> list[tuple]:
    return sorted(
        (
            int(spec["color"]),
            int(spec["count"]),
            tuple(spec["rect"]),
            tuple(sorted(spec["markers"])),
        )
        for spec in specs
    )


def _validate_scene(input_grid: np.ndarray, specs: list[dict]) -> bool:
    counts = [int(spec["count"]) for spec in specs]
    if len(set(counts)) != len(counts):
        return False
    if 2 not in counts or max(counts) < 3:
        return False
    if any(max(input_grid.shape) > 30 for _ in (0,)):
        return False

    perimeter_overlaps = [
        _rect_perimeter_pixels(a["rect"]) & _rect_perimeter_pixels(b["rect"])
        for a, b in itertools.combinations(specs, 2)
    ]
    if not any(perimeter_overlaps):
        return False

    output = _render_output(input_grid, specs)
    if not _visible_outline_hole(next(spec for spec in specs if int(spec["count"]) == 2), specs, output):
        return False
    if not _highest_fill_gain(specs, input_grid, output):
        return False
    if not _has_priority_overlap(specs, output):
        return False

    try:
        parsed_specs, _row_bounds, _col_bounds = _extract_specs(input_grid)
        actual = main(input_grid)
    except Exception:
        return False
    if _spec_signature(parsed_specs) != _spec_signature(specs):
        return False
    return np.array_equal(actual, output)


def _build_scene() -> np.ndarray | None:
    color_count = int(_RNG.integers(2, 4))
    counts = list(_COUNT_PATTERNS[color_count][int(_RNG.integers(len(_COUNT_PATTERNS[color_count])) )])
    colors = list(_RNG.choice(np.array(MARKER_COLORS, dtype=int), size=color_count, replace=False))
    _RNG.shuffle(colors)

    cell_rows = int(_RNG.integers(6, 10))
    cell_cols = int(_RNG.integers(6, 10))

    outline_idx = counts.index(2)
    strongest_idx = int(np.argmax(counts))
    pair = _sample_adjacent_pair(counts[outline_idx], counts[strongest_idx], cell_rows, cell_cols)
    if pair is None:
        return None

    rects: list[tuple[int, int, int, int] | None] = [None] * color_count
    rects[outline_idx] = pair[0]
    rects[strongest_idx] = pair[1]

    existing = [pair[0], pair[1]]
    for idx in range(color_count):
        if rects[idx] is not None:
            continue
        rect = _sample_free_rect(counts[idx], cell_rows, cell_cols, existing)
        if rect is None:
            return None
        rects[idx] = rect
        existing.append(rect)

    specs = []
    used_cells: set[tuple[int, int]] = set()
    for idx in range(color_count):
        rect = rects[idx]
        assert rect is not None
        markers = _choose_markers(rect, counts[idx], used_cells)
        if markers is None:
            return None
        used_cells.update(markers)
        specs.append(
            {
                "color": int(colors[idx]),
                "count": int(counts[idx]),
                "rect": rect,
                "markers": markers,
            }
        )

    grid = _build_base_grid(cell_rows, cell_cols)
    for spec in specs:
        for cell in spec["markers"]:
            grid[_cell_center(cell)] = int(spec["color"])

    width = grid.shape[1]
    for spec in specs:
        spec["first_scan"] = min(r * width + c for r, c in (_cell_center(cell) for cell in spec["markers"]))

    return grid if _validate_scene(grid, specs) else None


def generate_input() -> np.ndarray:
    for _ in range(_MAX_GENERATION_ATTEMPTS):
        grid = _build_scene()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input for puzzle 6")
