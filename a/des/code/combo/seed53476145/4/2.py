from collections import Counter

import numpy as np


BG_COLOR = 5
SEPARATOR_COLOR = 9
RED = 2
PALETTE = (1, 3, 4, 6, 7, 8)
MAX_GENERATION_ATTEMPTS = 400


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _infer_layout(grid: np.ndarray) -> tuple[int, int, int, list[np.ndarray], list[int]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    height, width = map(int, g.shape)
    bg = _most_common_color(g)

    candidates: dict[int, list[int]] = {}
    for col in range(width):
        column = g[:, col]
        if np.all(column == column[0]) and int(column[0]) != bg:
            candidates.setdefault(int(column[0]), []).append(col)

    for color, cols in sorted(candidates.items(), key=lambda item: (-len(item[1]), item[0])):
        if len(cols) not in (3, 4):
            continue

        panel_width = int(cols[0])
        if panel_width < 2:
            continue
        step = panel_width + 1
        expected = [panel_width + i * step for i in range(len(cols))]
        if cols != expected:
            continue
        if width != (len(cols) + 1) * panel_width + len(cols):
            continue

        panels: list[np.ndarray] = []
        start = 0
        valid = True
        for index, sep_col in enumerate(cols):
            if sep_col != start + panel_width:
                valid = False
                break
            panels.append(g[:, start : start + panel_width].copy())
            if not np.all(g[:, sep_col] == color):
                valid = False
                break
            start = sep_col + 1
        if not valid:
            continue

        panels.append(g[:, start : start + panel_width].copy())
        if start + panel_width != width:
            continue
        if np.any(panels[-1] != bg):
            continue
        if any(np.all(panel == bg) for panel in panels[:-1]):
            continue
        return bg, int(color), panel_width, panels, cols

    raise AssertionError("could not infer equal-width panel layout")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Compare matching pixels across repeated source panels and reveal anomalies in the blank panel.

    Input structure:
    - Grey background.
    - 3 or 4 equal-width source panels separated by full-height maroon columns.
    - A final blank panel of the same width acts as the output canvas.

    Rule:
    - Compare each pixel location across the non-empty source panels.
    - If all source panels agree, write background in the output panel.
    - If exactly one panel differs and every other panel shares the same color,
      copy that unique color to the output panel.
    - Otherwise write red.
    - Preserve the separator columns and every source panel.
    """
    g = np.asarray(grid, dtype=int)
    bg, _sep_color, panel_width, panels, _sep_cols = _infer_layout(g)
    source_panels = panels[:-1]

    out_panel = np.full((g.shape[0], panel_width), bg, dtype=int)
    for row in range(g.shape[0]):
        for col in range(panel_width):
            colors = [int(panel[row, col]) for panel in source_panels]
            counts = Counter(colors)
            if len(counts) == 1:
                continue
            if len(counts) == 2 and sorted(counts.values()) == [1, len(source_panels) - 1]:
                unique_color = next(color for color, count in counts.items() if count == 1)
                out_panel[row, col] = unique_color
            else:
                out_panel[row, col] = RED

    out = g.copy()
    out[:, -panel_width:] = out_panel
    return out


def _sample_distinct_colors(
    rng: np.random.Generator, count: int, exclude: set[int] | None = None
) -> list[int]:
    excluded = set() if exclude is None else set(exclude)
    choices = [color for color in PALETTE if color not in excluded]
    assert len(choices) >= count
    return [int(color) for color in rng.choice(choices, size=count, replace=False)]


def _rectangle_coords(
    rng: np.random.Generator, height: int, width: int
) -> list[tuple[int, int]]:
    rect_h = int(rng.integers(1, min(4, height) + 1))
    rect_w = int(rng.integers(1, min(4, width) + 1))
    r0 = int(rng.integers(0, height - rect_h + 1))
    c0 = int(rng.integers(0, width - rect_w + 1))
    return [(r, c) for r in range(r0, r0 + rect_h) for c in range(c0, c0 + rect_w)]


def _hline_coords(rng: np.random.Generator, height: int, width: int) -> list[tuple[int, int]]:
    length = int(rng.integers(2, min(width, 6) + 1))
    row = int(rng.integers(0, height))
    c0 = int(rng.integers(0, width - length + 1))
    return [(row, c) for c in range(c0, c0 + length)]


def _vline_coords(rng: np.random.Generator, height: int, width: int) -> list[tuple[int, int]]:
    length = int(rng.integers(2, min(height, 6) + 1))
    col = int(rng.integers(0, width))
    r0 = int(rng.integers(0, height - length + 1))
    return [(r, col) for r in range(r0, r0 + length)]


def _plus_coords(rng: np.random.Generator, height: int, width: int) -> list[tuple[int, int]] | None:
    if height < 3 or width < 3:
        return None
    row = int(rng.integers(1, height - 1))
    col = int(rng.integers(1, width - 1))
    return [(row, col), (row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)]


def _l_coords(rng: np.random.Generator, height: int, width: int) -> list[tuple[int, int]]:
    arm_h = int(rng.integers(2, min(5, height) + 1))
    arm_w = int(rng.integers(2, min(5, width) + 1))
    r0 = int(rng.integers(0, height - arm_h + 1))
    c0 = int(rng.integers(0, width - arm_w + 1))
    orientation = int(rng.integers(0, 4))
    coords: set[tuple[int, int]] = set()
    if orientation == 0:
        coords.update((r0 + r, c0) for r in range(arm_h))
        coords.update((r0 + arm_h - 1, c0 + c) for c in range(arm_w))
    elif orientation == 1:
        coords.update((r0, c0 + c) for c in range(arm_w))
        coords.update((r0 + r, c0) for r in range(arm_h))
    elif orientation == 2:
        coords.update((r0 + r, c0 + arm_w - 1) for r in range(arm_h))
        coords.update((r0, c0 + c) for c in range(arm_w))
    else:
        coords.update((r0 + r, c0 + arm_w - 1) for r in range(arm_h))
        coords.update((r0 + arm_h - 1, c0 + c) for c in range(arm_w))
    return sorted(coords)


def _random_shape_coords(
    rng: np.random.Generator, height: int, width: int
) -> list[tuple[int, int]]:
    builders = (_rectangle_coords, _hline_coords, _vline_coords, _plus_coords, _l_coords)
    for _ in range(12):
        builder = builders[int(rng.integers(0, len(builders)))]
        coords = builder(rng, height, width)
        if coords:
            return coords
    return _rectangle_coords(rng, height, width)


def _build_base_panel(rng: np.random.Generator, height: int, width: int) -> np.ndarray:
    min_filled = max(5, (height * width) // 6)
    for _ in range(200):
        panel = np.full((height, width), BG_COLOR, dtype=int)
        target_shapes = int(rng.integers(3, 7))
        placed = 0

        for _ in range(target_shapes * 12):
            coords = _random_shape_coords(rng, height, width)
            rr = np.array([coord[0] for coord in coords], dtype=int)
            cc = np.array([coord[1] for coord in coords], dtype=int)
            if np.any(panel[rr, cc] != BG_COLOR):
                continue
            panel[rr, cc] = int(rng.choice(PALETTE))
            placed += 1
            if placed >= target_shapes:
                break

        filled = int(np.count_nonzero(panel != BG_COLOR))
        colors = np.unique(panel[panel != BG_COLOR])
        if placed >= 3 and filled >= min_filled and len(colors) >= 2 and filled <= (height * width * 2) // 3:
            return panel

    raise RuntimeError("failed to build a base panel")


def _apply_unique_outlier(
    panels: list[np.ndarray], row: int, col: int, rng: np.random.Generator
) -> None:
    panel_index = int(rng.integers(0, len(panels)))
    majority_color = int(panels[0][row, col])
    unique_color = _sample_distinct_colors(rng, 1, exclude={majority_color})[0]
    panels[panel_index][row, col] = unique_color


def _apply_conflict(
    panels: list[np.ndarray], row: int, col: int, rng: np.random.Generator
) -> None:
    n_panels = len(panels)
    base_color = int(panels[0][row, col])

    if n_panels == 3:
        colors = _sample_distinct_colors(rng, 3)
        for index, color in enumerate(colors):
            panels[index][row, col] = color
        return

    conflict_type = int(rng.integers(0, 3))
    if conflict_type == 0:
        colors = _sample_distinct_colors(rng, 4)
    elif conflict_type == 1:
        pair = _sample_distinct_colors(rng, 2, exclude={base_color})
        colors = [pair[0], pair[0], pair[1], pair[1]]
    else:
        trio = _sample_distinct_colors(rng, 3)
        colors = [trio[0], trio[0], trio[1], trio[2]]

    order = rng.permutation(n_panels)
    for panel_index, color in zip(order, colors):
        panels[int(panel_index)][row, col] = int(color)


def _assemble_grid(source_panels: list[np.ndarray]) -> np.ndarray:
    height, panel_width = map(int, source_panels[0].shape)
    total_panels = len(source_panels) + 1
    width = total_panels * panel_width + (total_panels - 1)

    grid = np.full((height, width), BG_COLOR, dtype=int)
    start = 0
    for index, panel in enumerate(source_panels):
        grid[:, start : start + panel_width] = panel
        start += panel_width
        grid[:, start] = SEPARATOR_COLOR
        start += 1
    grid[:, start : start + panel_width] = BG_COLOR
    return grid


def _gen_attempt(rng: np.random.Generator) -> np.ndarray:
    n_sources = int(rng.choice([3, 4]))
    height = int(rng.integers(5, 11))
    panel_width = int(rng.integers(5, 11))
    base = _build_base_panel(rng, height, panel_width)
    panels = [base.copy() for _ in range(n_sources)]

    area = height * panel_width
    max_unique = min(3, max(1, area // 12))
    max_conflict = min(3, max(1, area // 15))
    unique_count = int(rng.integers(1, max_unique + 1))
    conflict_count = int(rng.integers(1, max_conflict + 1))
    total_changes = unique_count + conflict_count

    positions = rng.choice(area, size=total_changes, replace=False)
    split = unique_count
    for flat_index in positions[:split]:
        row, col = divmod(int(flat_index), panel_width)
        _apply_unique_outlier(panels, row, col, rng)
    for flat_index in positions[split:]:
        row, col = divmod(int(flat_index), panel_width)
        _apply_conflict(panels, row, col, rng)

    return _assemble_grid(panels)


def _validate_generated(grid: np.ndarray) -> None:
    bg, sep, panel_width, panels, sep_cols = _infer_layout(grid)
    assert bg == BG_COLOR
    assert sep == SEPARATOR_COLOR
    assert len(sep_cols) in (3, 4)
    assert np.all(panels[-1] == BG_COLOR)

    out = main(grid)
    result = out[:, -panel_width:]
    assert np.any(result == bg)
    assert np.any(result == RED)
    assert np.any((result != bg) & (result != RED))
    assert np.array_equal(out[:, :-panel_width], grid[:, :-panel_width])


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        try:
            _validate_generated(grid)
            return grid
        except AssertionError:
            continue
    raise RuntimeError("failed to generate a valid input")
