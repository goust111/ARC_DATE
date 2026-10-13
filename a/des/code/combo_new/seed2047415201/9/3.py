from __future__ import annotations

import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
MAGENTA = 6
ORANGE = 7
AZURE = 8

# Human-visible hue progression. Green is reserved for the common motif, so the
# marker sequence skips it.
MARKER_GRADIENT = (RED, ORANGE, YELLOW, AZURE, BLUE, MAGENTA)
MARKER_RANK = {color: idx for idx, color in enumerate(MARKER_GRADIENT)}


def _bands_from_separator_columns(g: np.ndarray) -> list[tuple[int, int]]:
    h, w = g.shape
    separators = [c for c in range(w) if np.all(g[:, c] == GREY)]
    assert separators, "expected grey columns between panels"

    bands: list[tuple[int, int]] = []
    start = 0
    for col in separators:
        if start < col:
            bands.append((start, col - 1))
        start = col + 1
    if start < w:
        bands.append((start, w - 1))

    assert len(bands) >= 2, "expected at least two panels"
    widths = [c1 - c0 + 1 for c0, c1 in bands]
    assert len(set(widths)) == 1, "all panels must have the same width"

    for c0, c1 in bands:
        panel = g[:, c0 : c1 + 1]
        assert not np.any(panel == GREY), "grey may only appear in separator columns"

    return bands


def _marker_in_panel(panel: np.ndarray) -> tuple[int, tuple[int, int]]:
    marker_mask = (panel != BLACK) & (panel != GREY) & (panel != GREEN)
    coords = np.argwhere(marker_mask)
    assert len(coords) == 1, "each panel must contain exactly one marker pixel"

    r, c = map(int, coords[0])
    color = int(panel[r, c])
    assert color in MARKER_RANK, "marker color must belong to the visible gradient"
    return color, (r, c)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reorder equal rectangular panels by the hue-gradient order of their embedded marker pixels.

    Input structure:
    - The background is black, and full-height grey separator columns divide the grid into equal-width panels in one row.
    - Each panel contains the same green foreground motif; one motif cell is recolored as a single marker pixel.
    - Marker colors are distinct members of a visible hue-gradient sequence such as red, orange, yellow, cyan, blue, and magenta.

    Rule:
    - Split the grid into panels at the grey separator columns.
    - Extract the unique non-black, non-grey, non-green marker color from each panel.
    - Sort the whole panels by the marker color's position in the hue-gradient sequence.
    - Paste the sorted panels back into the original panel slots, preserving grey separators, and erase each marker pixel to black.

    Key insight:
    - The marker colors are ordering keys rather than labels; after sorting, those keys are removed, leaving only the ordered missing cells in the common green motif.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert max(g.shape) <= 30, "ARC grids must not exceed 30 in either dimension"

    bands = _bands_from_separator_columns(g)
    keyed_panels: list[tuple[int, np.ndarray, tuple[int, int]]] = []
    seen_marker_colors: set[int] = set()

    for c0, c1 in bands:
        panel = g[:, c0 : c1 + 1].copy()
        color, marker_pos = _marker_in_panel(panel)
        assert color not in seen_marker_colors, "marker colors must be unique"
        seen_marker_colors.add(color)
        keyed_panels.append((MARKER_RANK[color], panel, marker_pos))

    keyed_panels.sort(key=lambda item: item[0])

    out = g.copy()
    for (c0, c1), (_rank, panel, (mr, mc)) in zip(bands, keyed_panels, strict=False):
        cleaned = panel.copy()
        cleaned[mr, mc] = BLACK
        out[:, c0 : c1 + 1] = cleaned

    return out


# --------------- generator ---------------

PANEL_COUNT_RANGE = (3, 6)
PANEL_HEIGHT_RANGE = (5, 10)
PANEL_WIDTH_RANGE = (4, 8)
MOTIF_SIZE_EXTRA_RANGE = (0, 7)
GEN_RETRIES = 500


def _grow_connected_cells(h: int, w: int, size: int, rng: random.Random) -> set[tuple[int, int]]:
    cells = {(rng.randrange(h), rng.randrange(w))}
    while len(cells) < size:
        r, c = rng.choice(tuple(cells))
        neighbors = []
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            rr, cc = r + dr, c + dc
            if 0 <= rr < h and 0 <= cc < w and (rr, cc) not in cells:
                neighbors.append((rr, cc))
        if neighbors:
            cells.add(rng.choice(neighbors))
        elif len(cells) < h * w:
            frontier = [
                (rr, cc)
                for rr in range(h)
                for cc in range(w)
                if (rr, cc) not in cells
                and any(abs(rr - r0) + abs(cc - c0) == 1 for r0, c0 in cells)
            ]
            if not frontier:
                break
            cells.add(rng.choice(frontier))
    return cells


def _make_motif_mask(panel_h: int, panel_w: int, n_markers: int, rng: random.Random) -> np.ndarray | None:
    for _ in range(80):
        box_h = rng.randint(2, panel_h)
        box_w = rng.randint(2, panel_w)
        area = box_h * box_w
        if area < n_markers:
            continue

        max_size = min(area, n_markers + rng.randint(*MOTIF_SIZE_EXTRA_RANGE))
        size = rng.randint(n_markers, max_size)
        cells = _grow_connected_cells(box_h, box_w, size, rng)
        if len(cells) < n_markers:
            continue

        r0 = rng.randint(0, panel_h - box_h)
        c0 = rng.randint(0, panel_w - box_w)
        mask = np.zeros((panel_h, panel_w), dtype=bool)
        for r, c in cells:
            mask[r0 + r, c0 + c] = True
        return mask
    return None


def _render_panel(mask: np.ndarray, marker_pos: tuple[int, int], marker_color: int) -> np.ndarray:
    panel = np.full(mask.shape, BLACK, dtype=int)
    panel[mask] = GREEN
    panel[marker_pos] = int(marker_color)
    return panel


def _assemble_panels(panels: list[np.ndarray]) -> np.ndarray:
    panel_h, panel_w = panels[0].shape
    width = len(panels) * panel_w + len(panels) - 1
    grid = np.full((panel_h, width), GREY, dtype=int)

    col = 0
    for panel in panels:
        grid[:, col : col + panel_w] = panel
        col += panel_w + 1
    return grid


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    n_panels = rng.randint(*PANEL_COUNT_RANGE)
    max_panel_w = min(PANEL_WIDTH_RANGE[1], (30 - (n_panels - 1)) // n_panels)
    if max_panel_w < PANEL_WIDTH_RANGE[0]:
        return None

    panel_h = rng.randint(*PANEL_HEIGHT_RANGE)
    panel_w = rng.randint(PANEL_WIDTH_RANGE[0], max_panel_w)
    if panel_h > 30:
        return None

    start = rng.randint(0, len(MARKER_GRADIENT) - n_panels)
    marker_colors = list(MARKER_GRADIENT[start : start + n_panels])
    mask = _make_motif_mask(panel_h, panel_w, n_panels, rng)
    if mask is None:
        return None

    motif_cells = [tuple(map(int, pos)) for pos in np.argwhere(mask)]
    marker_positions = rng.sample(motif_cells, n_panels)

    sorted_panels = [
        _render_panel(mask, marker_pos, marker_color)
        for marker_color, marker_pos in zip(marker_colors, marker_positions, strict=False)
    ]

    permutation = list(range(n_panels))
    for _ in range(20):
        rng.shuffle(permutation)
        if permutation != list(range(n_panels)):
            break
    if permutation == list(range(n_panels)):
        return None

    input_panels = [sorted_panels[i] for i in permutation]
    grid = _assemble_panels(input_panels)
    if max(grid.shape) > 30:
        return None

    expected_panels = []
    for panel, marker_pos in zip(sorted_panels, marker_positions, strict=False):
        cleaned = panel.copy()
        cleaned[marker_pos] = BLACK
        expected_panels.append(cleaned)
    expected = _assemble_panels(expected_panels)

    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {GEN_RETRIES} attempts")
