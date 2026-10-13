import random

import numpy as np


BACKGROUND = 0
SEPARATOR = 5
MARKER_TO_OP = {
    1: "diag",        # blue
    2: "horizontal",  # red
    3: "vertical",    # green
    4: "both",        # yellow
}
MARKER_COLORS = tuple(MARKER_TO_OP)
SEED_COLORS = (6, 7, 8, 9)
QUADRANTS = ((0, 0), (0, 1), (1, 0), (1, 1))
NEIGHBORS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
QUADRANT_RADIUS_RANGE = (2, 4)
PANEL_COUNT_CHOICES = (2, 3)
MAX_GENERATION_ATTEMPTS = 300


def _mode(values: np.ndarray) -> int:
    vals = np.asarray(values, dtype=int).ravel()
    assert vals.size > 0
    return int(np.bincount(vals).argmax())


def _separator_lines(grid: np.ndarray) -> tuple[int, list[int], list[int]]:
    h, w = grid.shape

    row_by_color: dict[int, list[int]] = {}
    for r in range(h):
        color = int(grid[r, 0])
        if np.all(grid[r, :] == color):
            row_by_color.setdefault(color, []).append(r)

    col_by_color: dict[int, list[int]] = {}
    for c in range(w):
        color = int(grid[0, c])
        if np.all(grid[:, c] == color):
            col_by_color.setdefault(color, []).append(c)

    common = set(row_by_color) & set(col_by_color)
    assert len(common) == 1, "expected one separator color shared by full rows and columns"
    separator = common.pop()
    return int(separator), sorted(row_by_color[separator]), sorted(col_by_color[separator])


def _spans(length: int, cuts: list[int]) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    start = 0
    for cut in sorted(cuts) + [length]:
        if cut > start:
            spans.append((start, cut))
        start = cut + 1
    return spans


def _quadrant_slices(radius: int) -> dict[tuple[int, int], tuple[slice, slice]]:
    return {
        (0, 0): (slice(0, radius), slice(0, radius)),
        (0, 1): (slice(0, radius), slice(radius + 1, 2 * radius + 1)),
        (1, 0): (slice(radius + 1, 2 * radius + 1), slice(0, radius)),
        (1, 1): (slice(radius + 1, 2 * radius + 1), slice(radius + 1, 2 * radius + 1)),
    }


def _mirror_targets(r: int, c: int, center: int, op: str) -> set[tuple[int, int]]:
    if op == "horizontal":
        return {(r, c), (2 * center - r, c)}
    if op == "vertical":
        return {(r, c), (r, 2 * center - c)}
    if op == "both":
        return {
            (r, c),
            (2 * center - r, c),
            (r, 2 * center - c),
            (2 * center - r, 2 * center - c),
        }
    if op == "diag":
        return {(r, c), (c, r)}
    raise AssertionError(f"unknown op: {op}")


def _expand_panel(panel: np.ndarray, bg: int) -> np.ndarray:
    panel = np.asarray(panel, dtype=int)
    h, w = panel.shape
    assert h == w and h % 2 == 1, "panels must be odd-sized squares"
    center = h // 2

    marker = int(panel[center, center])
    assert marker in MARKER_TO_OP, "expected a center control pixel"
    op = MARKER_TO_OP[marker]

    quadrants = []
    payload = panel.copy()
    payload[center, center] = bg
    for quadrant, (rs, cs) in _quadrant_slices(center).items():
        if np.any(payload[rs, cs] != bg):
            quadrants.append(quadrant)
    assert len(quadrants) == 1, "expected exactly one seeded quadrant per panel"

    out = payload.copy()
    for r, c in np.argwhere(payload != bg):
        color = int(payload[r, c])
        for tr, tc in _mirror_targets(int(r), int(c), center, op):
            out[tr, tc] = color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand the seeded motif inside each separator-defined panel using the center marker's symmetry code.

    Input structure:
    - Full separator-colored rows and columns partition the grid into equal-sized square panels.
    - Each panel has a dominant background color, exactly one non-background seed motif in a single quadrant, and one control pixel at the panel center.
    - The center control pixel is one of blue, red, green, or yellow, while the other three quadrants start empty.

    Rule:
    - Detect the separator lattice, then process each panel independently.
    - Remove the center control pixel and keep the original seed motif in place.
    - Red mirrors the seed across the horizontal midline, green mirrors it across the vertical midline, yellow mirrors across both midlines, and blue mirrors it across the panel's main diagonal by transposing coordinates.

    Key insight:
    - The lone center pixel is the instruction channel: its color selects which symmetry to apply to the single seeded quadrant.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    separator, sep_rows, sep_cols = _separator_lines(g)
    row_spans = _spans(g.shape[0], sep_rows)
    col_spans = _spans(g.shape[1], sep_cols)
    assert row_spans and col_spans, "expected at least one panel row and column"

    heights = {r1 - r0 for r0, r1 in row_spans}
    widths = {c1 - c0 for c0, c1 in col_spans}
    assert len(heights) == 1 and len(widths) == 1, "expected equal panel sizes"
    panel_h = heights.pop()
    panel_w = widths.pop()
    assert panel_h == panel_w and panel_h % 2 == 1, "expected odd square panels"

    bg = _mode(g[g != separator])
    out = g.copy()
    for r0, r1 in row_spans:
        for c0, c1 in col_spans:
            panel = g[r0:r1, c0:c1]
            out[r0:r1, c0:c1] = _expand_panel(panel, bg)
    return out


def _neighbors4(r: int, c: int, size: int):
    for dr, dc in NEIGHBORS4:
        nr, nc = r + dr, c + dc
        if 0 <= nr < size and 0 <= nc < size:
            yield nr, nc


def _grow_shape(size: int, filled: int, rng: random.Random) -> set[tuple[int, int]] | None:
    start = (rng.randrange(size), rng.randrange(size))
    cells = {start}
    while len(cells) < filled:
        frontier = []
        for r, c in cells:
            for nr, nc in _neighbors4(r, c, size):
                if (nr, nc) not in cells:
                    frontier.append((nr, nc))
        if not frontier:
            return None
        cells.add(rng.choice(frontier))
    return cells


def _random_seed(radius: int, rng: random.Random) -> np.ndarray | None:
    total = radius * radius
    filled = rng.randint(1, total - 1)
    cells = _grow_shape(radius, filled, rng)
    if cells is None:
        return None

    palette_size = rng.randint(1, min(3, len(SEED_COLORS), filled))
    palette = rng.sample(SEED_COLORS, palette_size)
    seed = np.full((radius, radius), BACKGROUND, dtype=int)
    ordered = list(cells)
    rng.shuffle(ordered)
    for idx, (r, c) in enumerate(ordered):
        if idx < palette_size:
            color = palette[idx]
        else:
            color = rng.choice(palette)
        seed[r, c] = color
    return seed


def _assemble_panel(seed: np.ndarray, marker: int, quadrant: tuple[int, int]) -> np.ndarray:
    radius = seed.shape[0]
    panel = np.full((2 * radius + 1, 2 * radius + 1), BACKGROUND, dtype=int)
    r0 = 0 if quadrant[0] == 0 else radius + 1
    c0 = 0 if quadrant[1] == 0 else radius + 1
    panel[r0 : r0 + radius, c0 : c0 + radius] = seed
    panel[radius, radius] = marker
    return panel


def _panel_is_interesting(panel: np.ndarray) -> bool:
    center = panel.shape[0] // 2
    without_marker = panel.copy()
    without_marker[center, center] = BACKGROUND
    expanded = _expand_panel(panel, BACKGROUND)
    return not np.array_equal(expanded, without_marker)


def _marker_schedule(count: int, rng: random.Random) -> list[int]:
    markers = [MARKER_COLORS[i % len(MARKER_COLORS)] for i in range(count)]
    rng.shuffle(markers)
    return markers


def _grid_shape(panel_size: int, panel_rows: int, panel_cols: int) -> tuple[int, int]:
    return panel_rows * panel_size + (panel_rows - 1), panel_cols * panel_size + (panel_cols - 1)


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    radius = rng.randint(*QUADRANT_RADIUS_RANGE)
    panel_size = 2 * radius + 1

    row_choices = [n for n in PANEL_COUNT_CHOICES if _grid_shape(panel_size, n, 2)[0] <= 30]
    col_choices = [n for n in PANEL_COUNT_CHOICES if _grid_shape(panel_size, 2, n)[1] <= 30]
    if not row_choices or not col_choices:
        return None

    panel_rows = rng.choice(row_choices)
    panel_cols = rng.choice(col_choices)
    grid_h, grid_w = _grid_shape(panel_size, panel_rows, panel_cols)
    if grid_h > 30 or grid_w > 30:
        return None

    grid = np.full((grid_h, grid_w), BACKGROUND, dtype=int)
    for pr in range(1, panel_rows):
        grid[pr * panel_size + (pr - 1), :] = SEPARATOR
    for pc in range(1, panel_cols):
        grid[:, pc * panel_size + (pc - 1)] = SEPARATOR

    markers = _marker_schedule(panel_rows * panel_cols, rng)
    assert set(MARKER_COLORS).issubset(markers), "every sample should expose all symmetry branches"

    idx = 0
    for pr in range(panel_rows):
        for pc in range(panel_cols):
            marker = markers[idx]
            quadrant = rng.choice(QUADRANTS)
            panel = None
            for _ in range(60):
                seed = _random_seed(radius, rng)
                if seed is None:
                    continue
                candidate = _assemble_panel(seed, marker, quadrant)
                if _panel_is_interesting(candidate):
                    panel = candidate
                    break
            if panel is None:
                return None
            r0 = pr * (panel_size + 1)
            c0 = pc * (panel_size + 1)
            grid[r0 : r0 + panel_size, c0 : c0 + panel_size] = panel
            idx += 1

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for the symmetry-seed task")
