import numpy as np


BLACK = 0
GREY = 5
COLOR_POOL = tuple(c for c in range(1, 10) if c != GREY)

GENERATION_ATTEMPTS = 300
MAX_GRID_SIDE = 28
PANEL_COUNT_RANGE = (3, 6)
PANEL_DIM_MIN = 3
PANEL_DIM_MAX = 8
MARGIN_RANGE = (1, 4)
MAX_ANOMALY_RATIO = 0.35
MAX_ANOMALY_COLORS = 4


def _randint(bounds: tuple[int, int]) -> int:
    lo, hi = bounds
    return int(np.random.randint(int(lo), int(hi) + 1))


def _nonblack_bbox(grid: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(grid != BLACK)
    assert coords.size > 0, "expected at least one non-black pixel"
    (r0, c0), (r1, c1) = coords.min(axis=0), coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _segments_from_separators(length: int, separators: list[int]) -> list[tuple[int, int]] | None:
    seps = sorted(set(map(int, separators)))
    if len(seps) < 2 or seps[0] == 0 or seps[-1] == length - 1:
        return None

    segments: list[tuple[int, int]] = []
    start = 0
    for sep in seps:
        if sep <= start:
            return None
        segments.append((start, sep))
        start = sep + 1
    if start >= length:
        return None
    segments.append((start, length))

    if len(segments) < 3:
        return None
    widths = {b - a for a, b in segments}
    if len(widths) != 1:
        return None
    return segments


def _find_panels(grid: np.ndarray) -> list[np.ndarray]:
    r0, c0, r1, c1 = _nonblack_bbox(grid)
    sub = grid[r0 : r1 + 1, c0 : c1 + 1]
    colors = sorted(set(map(int, sub.ravel())) - {BLACK})

    candidates: list[list[np.ndarray]] = []
    h, w = sub.shape
    for color in colors:
        full_cols = [c for c in range(w) if bool(np.all(sub[:, c] == color))]
        col_segments = _segments_from_separators(w, full_cols)
        if col_segments is not None:
            panels = [sub[:, a:b] for a, b in col_segments]
            if panels and all(not np.any(panel == color) for panel in panels):
                candidates.append(panels)

        full_rows = [r for r in range(h) if bool(np.all(sub[r, :] == color))]
        row_segments = _segments_from_separators(h, full_rows)
        if row_segments is not None:
            panels = [sub[a:b, :] for a, b in row_segments]
            if panels and all(not np.any(panel == color) for panel in panels):
                candidates.append(panels)

    assert len(candidates) == 1, "expected a unique one-dimensional panel layout"
    shapes = {panel.shape for panel in candidates[0]}
    assert len(shapes) == 1, "all panels must have the same shape"
    return [np.array(panel, dtype=int, copy=True) for panel in candidates[0]]


def _majority_template(panels: list[np.ndarray]) -> np.ndarray:
    stack = np.stack(panels, axis=0)
    _, h, w = stack.shape
    template = np.zeros((h, w), dtype=int)
    for r in range(h):
        for c in range(w):
            values, counts = np.unique(stack[:, r, c], return_counts=True)
            best = counts.max()
            assert int(np.sum(counts == best)) == 1, "majority vote must be unambiguous"
            template[r, c] = int(values[np.argmax(counts)])
    return template


def main(grid: np.ndarray) -> np.ndarray:
    """
    Detect repeated same-size panels separated by single-cell lines, infer the
    anomaly-free template by majority vote, and overlay all anomaly pixels onto
    a shared output canvas.
    """
    g = np.asarray(grid, dtype=int)
    panels = _find_panels(g)
    template = _majority_template(panels)

    out = np.zeros_like(template)
    for panel in panels:
        diff = panel != template
        conflict = diff & (out != BLACK) & (out != panel)
        assert not np.any(conflict), "anomaly projections must not overlap"
        out[diff] = panel[diff]
    return out


def _choose_panel_shape(
    orientation: str,
    panel_count: int,
    top: int,
    bottom: int,
    left: int,
    right: int,
) -> tuple[int, int] | None:
    if orientation == "horizontal":
        max_h = min(PANEL_DIM_MAX, MAX_GRID_SIDE - top - bottom)
        max_w = min(PANEL_DIM_MAX, (MAX_GRID_SIDE - left - right - (panel_count - 1)) // panel_count)
    else:
        max_h = min(PANEL_DIM_MAX, (MAX_GRID_SIDE - top - bottom - (panel_count - 1)) // panel_count)
        max_w = min(PANEL_DIM_MAX, MAX_GRID_SIDE - left - right)

    if max_h < PANEL_DIM_MIN or max_w < PANEL_DIM_MIN:
        return None
    return _randint((PANEL_DIM_MIN, max_h)), _randint((PANEL_DIM_MIN, max_w))


def _gen_attempt() -> np.ndarray | None:
    orientation = str(np.random.choice(["horizontal", "vertical"]))
    panel_count = _randint(PANEL_COUNT_RANGE)

    top = _randint(MARGIN_RANGE)
    bottom = _randint(MARGIN_RANGE)
    left = _randint(MARGIN_RANGE)
    right = _randint(MARGIN_RANGE)

    panel_shape = _choose_panel_shape(orientation, panel_count, top, bottom, left, right)
    if panel_shape is None:
        return None
    ph, pw = panel_shape

    base_color = int(np.random.choice(COLOR_POOL))
    anomaly_colors = [c for c in COLOR_POOL if c != base_color]
    palette_size = _randint((1, min(MAX_ANOMALY_COLORS, len(anomaly_colors))))
    palette = list(map(int, np.random.choice(anomaly_colors, size=palette_size, replace=False)))

    capacity = ph * pw
    max_anomalies = max(1, min(capacity, int(np.ceil(capacity * MAX_ANOMALY_RATIO))))
    anomalous_panel_count = _randint((1, min(panel_count, max_anomalies)))
    total_anomalies = _randint((anomalous_panel_count, max_anomalies))

    anomalous_panels = list(map(int, np.random.choice(panel_count, size=anomalous_panel_count, replace=False)))
    positions = list(map(int, np.random.choice(capacity, size=total_anomalies, replace=False)))

    panel_assignments = anomalous_panels.copy()
    while len(panel_assignments) < total_anomalies:
        panel_assignments.append(int(np.random.choice(anomalous_panels)))
    np.random.shuffle(panel_assignments)
    np.random.shuffle(positions)

    panels = np.full((panel_count, ph, pw), base_color, dtype=int)
    expected = np.zeros((ph, pw), dtype=int)
    for flat_pos, panel_index in zip(positions, panel_assignments):
        r, c = divmod(flat_pos, pw)
        color = int(np.random.choice(palette))
        panels[panel_index, r, c] = color
        expected[r, c] = color

    if orientation == "horizontal":
        height = top + ph + bottom
        width = left + panel_count * pw + (panel_count - 1) + right
        grid = np.zeros((height, width), dtype=int)
        c = left
        for idx in range(panel_count):
            grid[top : top + ph, c : c + pw] = panels[idx]
            c += pw
            if idx != panel_count - 1:
                grid[top : top + ph, c] = GREY
                c += 1
    else:
        height = top + panel_count * ph + (panel_count - 1) + bottom
        width = left + pw + right
        grid = np.zeros((height, width), dtype=int)
        r = top
        for idx in range(panel_count):
            grid[r : r + ph, left : left + pw] = panels[idx]
            r += ph
            if idx != panel_count - 1:
                grid[r, left : left + pw] = GREY
                r += 1

    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
