from __future__ import annotations

import numpy as np


BG = 0
MARKER = 2
DIVIDER = 5
LATTICE = 7
CONTENT_PALETTE = [1, 2, 3, 4, 6, 8, 9]
GENERATION_ATTEMPTS = 500


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(values[np.argmax(counts)])


def _colors_present(grid: np.ndarray, *, exclude: set[int] | None = None) -> list[int]:
    colors = set(map(int, np.unique(np.asarray(grid, dtype=int))))
    if exclude is not None:
        colors -= set(map(int, exclude))
    return sorted(colors)


def _is_regular_spacing(positions: list[int]) -> bool:
    if len(positions) < 2:
        return False
    gaps = np.diff(np.asarray(positions, dtype=int))
    return bool(np.all(gaps == gaps[0]))


def _find_divider_col(grid: np.ndarray, _bg: int) -> tuple[int, int]:
    candidates = [c for c in range(grid.shape[1]) if np.all(grid[:, c] == DIVIDER)]
    assert len(candidates) == 1, "expected exactly one full-height gray divider"
    return int(candidates[0]), DIVIDER


def _find_lattice(left: np.ndarray, bg: int) -> tuple[int, list[int], list[int]]:
    h, w = left.shape
    candidates: list[tuple[int, list[int], list[int]]] = []
    for color in _colors_present(left, exclude={bg}):
        rows = [r for r in range(h) if np.count_nonzero(left[r, :] == color) >= w - 2]
        cols = [c for c in range(w) if np.count_nonzero(left[:, c] == color) >= h - 2]
        if len(rows) >= 2 and len(cols) >= 2 and _is_regular_spacing(rows) and _is_regular_spacing(cols):
            candidates.append((color, rows, cols))
    assert len(candidates) == 1, "expected a unique regular lattice color"
    return candidates[0]


def _analyze(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    bg = _get_bg(g)
    divider_col, divider_color = _find_divider_col(g, bg)
    assert divider_col > 0 and divider_col < g.shape[1] - 1, "divider must split left and right panels"

    left = g[:, :divider_col]
    right = g[:, divider_col + 1 :]
    lattice_color, lattice_rows, lattice_cols = _find_lattice(left, bg)

    holes = [(r, c) for r in lattice_rows for c in lattice_cols if int(left[r, c]) == bg]
    assert len(holes) == 4, "expected exactly four background holes at lattice intersections"

    hole_rows = sorted({r for r, _ in holes})
    hole_cols = sorted({c for _, c in holes})
    assert len(hole_rows) == 2 and len(hole_cols) == 2, "holes must form rectangle corners"
    r0, r1 = hole_rows
    c0, c1 = hole_cols
    assert set(holes) == {(r0, c0), (r0, c1), (r1, c0), (r1, c1)}

    template = left[r0 : r1 + 1, c0 : c1 + 1].copy()
    slot_h, slot_w = template.shape
    assert slot_h % 2 == 1 and slot_w % 2 == 1, "slot size must have a unique center"
    assert right.shape[1] == slot_w, "right panel width must match the crop width"
    assert g.shape[0] % slot_h == 0, "right panel must partition into equal-height slots"

    slot_count = int(g.shape[0] // slot_h)
    marker_colors = set(map(int, np.unique(right))) - {bg}
    assert len(marker_colors) == 1, "right panel must contain exactly one marker color"
    marker_color = marker_colors.pop()
    center = (slot_h // 2, slot_w // 2)

    marked_slots: list[int] = []
    for slot_i in range(slot_count):
        sr0 = slot_i * slot_h
        slot = right[sr0 : sr0 + slot_h, :]
        marker_cells = np.argwhere(slot != bg)
        if marker_cells.size == 0:
            continue
        assert marker_cells.shape == (1, 2), "marked slots must contain exactly one non-background cell"
        mr, mc = int(marker_cells[0, 0]), int(marker_cells[0, 1])
        assert (mr, mc) == center, "marker must be centered inside its slot"
        assert int(slot[mr, mc]) == marker_color, "all slot markers must share one color"
        marked_slots.append(slot_i)

    assert 1 <= len(marked_slots) <= min(4, slot_count), "marked slot count must be between 1 and 4"

    return {
        "grid": g,
        "bg": bg,
        "divider_col": divider_col,
        "divider_color": divider_color,
        "left": left,
        "right": right,
        "lattice_color": lattice_color,
        "lattice_rows": lattice_rows,
        "lattice_cols": lattice_cols,
        "hole_rows": hole_rows,
        "hole_cols": hole_cols,
        "template": template,
        "slot_h": slot_h,
        "slot_w": slot_w,
        "slot_count": slot_count,
        "marker_color": marker_color,
        "marked_slots": marked_slots,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Crop the lattice rectangle marked by four missing intersections and stamp it into each marked slot.

    Input structure:
    - The left panel is a regular one-cell-thick lattice whose line intersections include exactly four background-colored holes.
    - The four holes lie on lattice intersections and form the corners of one axis-aligned crop rectangle; sparse colored pixels sit inside lattice cells.
    - A full-height divider separates the lattice from a right panel made of same-size empty slots, some with one centered marker pixel.

    Rule:
    - Find the unique regular lattice and the four missing intersection pixels on it.
    - Take the inclusive crop whose corners are those four holes; this crop is the stamp template.
    - Partition the right panel into slots of the crop's size and locate the slots whose single non-background pixel is centered.
    - Replace every marked slot with the crop template, leaving the left panel, divider, and unmarked slots unchanged.

    Key insight:
    - The template is not given directly: the four missing lattice intersections uniquely define which subgrid must be cropped before any slot can be filled.
    """
    info = _analyze(grid)
    out = info["grid"].copy()
    r_panel_c0 = info["divider_col"] + 1
    for slot_i in info["marked_slots"]:
        r0 = slot_i * info["slot_h"]
        out[r0 : r0 + info["slot_h"], r_panel_c0 : r_panel_c0 + info["slot_w"]] = info["template"]
    return out


def _place_pixels_in_cell(
    left: np.ndarray,
    row_lines: list[int],
    col_lines: list[int],
    cell_r: int,
    cell_c: int,
    rng: np.random.Generator,
    colors: np.ndarray,
    count_range: tuple[int, int],
) -> None:
    r0, r1 = row_lines[cell_r] + 1, row_lines[cell_r + 1] - 1
    c0, c1 = col_lines[cell_c] + 1, col_lines[cell_c + 1] - 1
    positions = [(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)]
    if not positions:
        return
    lo, hi = count_range
    count = int(rng.integers(lo, min(hi, len(positions)) + 1))
    chosen = rng.choice(len(positions), size=count, replace=False)
    for idx in np.atleast_1d(chosen):
        r, c = positions[int(idx)]
        left[r, c] = int(colors[int(rng.integers(0, len(colors)))])


def _candidate_crop_count(rows: list[int], cols: list[int], span_r: int, span_c: int) -> int:
    return max(0, len(rows) - span_r) * max(0, len(cols) - span_c)


def _template_is_unique(left: np.ndarray, rows: list[int], cols: list[int], span_r: int, span_c: int, template: np.ndarray) -> bool:
    matches = 0
    for i in range(len(rows) - span_r):
        for j in range(len(cols) - span_c):
            r0, r1 = rows[i], rows[i + span_r]
            c0, c1 = cols[j], cols[j + span_c]
            if np.array_equal(left[r0 : r1 + 1, c0 : c1 + 1], template):
                matches += 1
                if matches > 1:
                    return False
    return matches == 1


def _build_expected_output(grid: np.ndarray, info: dict) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    c0 = info["divider_col"] + 1
    for slot_i in info["marked_slots"]:
        r0 = slot_i * info["slot_h"]
        out[r0 : r0 + info["slot_h"], c0 : c0 + info["slot_w"]] = info["template"]
    return out


def _validate_semantics(grid: np.ndarray) -> bool:
    try:
        info = _analyze(grid)
    except AssertionError:
        return False

    rows = info["lattice_rows"]
    cols = info["lattice_cols"]
    hole_rows = info["hole_rows"]
    hole_cols = info["hole_cols"]
    span_r = rows.index(hole_rows[1]) - rows.index(hole_rows[0])
    span_c = cols.index(hole_cols[1]) - cols.index(hole_cols[0])

    if span_r * span_c < 2:
        return False
    if _candidate_crop_count(rows, cols, span_r, span_c) < 2:
        return False
    if not _template_is_unique(info["left"], rows, cols, span_r, span_c, info["template"]):
        return False

    inner = (info["template"] != BG) & (info["template"] != LATTICE)
    inner_colors = set(map(int, np.unique(info["template"][inner])))
    if np.count_nonzero(inner) < 3 or len(inner_colors) < 2:
        return False

    outside = info["left"].copy()
    outside[hole_rows[0] : hole_rows[1] + 1, hole_cols[0] : hole_cols[1] + 1] = BG
    if not np.any((outside != BG) & (outside != LATTICE)):
        return False

    if info["slot_count"] > 1 and len(info["marked_slots"]) >= info["slot_count"]:
        return False

    expected = _build_expected_output(grid, info)
    try:
        if not np.array_equal(main(grid), expected):
            return False
    except AssertionError:
        return False

    return True


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()

    cell_size = int(rng.choice([2, 3]))
    if cell_size == 2:
        span_r, span_c = 2, 2
    else:
        span_r, span_c = [(1, 2), (2, 1), (2, 2)][int(rng.integers(0, 3))]

    step = cell_size + 1
    crop_h = span_r * step + 1
    crop_w = span_c * step + 1

    feasible_slot_counts = [k for k in range(1, 5) if k * crop_h <= 30]
    if not feasible_slot_counts:
        return None
    slot_count = int(rng.choice(feasible_slot_counts))
    height = slot_count * crop_h

    max_left_intervals = (28 - crop_w) // step
    min_left_intervals = span_c + 1
    if max_left_intervals < min_left_intervals:
        return None
    left_intervals = int(rng.integers(min_left_intervals, max_left_intervals + 1))
    left_w = left_intervals * step + 1

    line_rows = list(range(0, height, step))
    line_cols = list(range(0, left_w, step))
    if len(line_rows) <= span_r or len(line_cols) <= span_c:
        return None

    hole_row_i = int(rng.integers(0, len(line_rows) - span_r))
    hole_col_i = int(rng.integers(0, len(line_cols) - span_c))
    hole_rows = [line_rows[hole_row_i], line_rows[hole_row_i + span_r]]
    hole_cols = [line_cols[hole_col_i], line_cols[hole_col_i + span_c]]

    left = np.full((height, left_w), BG, dtype=int)
    left[line_rows, :] = LATTICE
    left[:, line_cols] = LATTICE

    color_count = int(rng.integers(3, min(6, len(CONTENT_PALETTE)) + 1))
    content_colors = rng.choice(CONTENT_PALETTE, size=color_count, replace=False)

    crop_cells = [(r, c) for r in range(hole_row_i, hole_row_i + span_r) for c in range(hole_col_i, hole_col_i + span_c)]
    other_cells = [
        (r, c)
        for r in range(len(line_rows) - 1)
        for c in range(len(line_cols) - 1)
        if (r, c) not in crop_cells
    ]

    inside_pick = min(len(crop_cells), int(rng.integers(2, len(crop_cells) + 1)))
    inside_cells = rng.choice(len(crop_cells), size=inside_pick, replace=False)
    for idx in np.atleast_1d(inside_cells):
        _place_pixels_in_cell(left, line_rows, line_cols, crop_cells[int(idx)][0], crop_cells[int(idx)][1], rng, content_colors, (1, 2))

    if other_cells:
        outside_pick = min(len(other_cells), int(rng.integers(1, min(5, len(other_cells)) + 1)))
        outside_cells = rng.choice(len(other_cells), size=outside_pick, replace=False)
        for idx in np.atleast_1d(outside_cells):
            _place_pixels_in_cell(left, line_rows, line_cols, other_cells[int(idx)][0], other_cells[int(idx)][1], rng, content_colors, (1, 2))

    for r in range(len(line_rows) - 1):
        for c in range(len(line_cols) - 1):
            if rng.random() < 0.28:
                _place_pixels_in_cell(left, line_rows, line_cols, r, c, rng, content_colors, (1, 2))

    for r in hole_rows:
        for c in hole_cols:
            left[r, c] = BG

    right = np.full((height, crop_w), BG, dtype=int)
    if slot_count == 1:
        marked_count = 1
    else:
        marked_count = int(rng.integers(1, min(4, slot_count - 1) + 1))
    marked_slots = sorted(map(int, rng.choice(slot_count, size=marked_count, replace=False)))
    center_r, center_c = crop_h // 2, crop_w // 2
    for slot_i in marked_slots:
        right[slot_i * crop_h + center_r, center_c] = MARKER

    grid = np.full((height, left_w + 1 + crop_w), BG, dtype=int)
    grid[:, :left_w] = left
    grid[:, left_w] = DIVIDER
    grid[:, left_w + 1 :] = right

    return grid if _validate_semantics(grid) else None


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
