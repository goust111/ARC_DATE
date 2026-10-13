from collections import Counter

import numpy as np

from arc_utils import find_components, get_bg, paste, random_free_location
import itertools
import random


def main(grid: np.ndarray) -> np.ndarray:
    """Overlay the colored contents of same-sized blank panels after removing the dominant separator color.

    Input structure
    - The most common color is a separator surrounding several rectangular
      panels made from a common blank color plus sparse colored marks.
    - The panels are 4-connected regions of non-separator cells, and all panel
      bounding boxes have the same shape.
    Rule
    - Create one blank panel of that common shape.
    - For each input panel, copy every non-blank mark to the same relative
      position in the output panel.
    Key insight
    - The large separator color is not part of the answer; the answer is the
      transparent overlay of the small drawings inside its panels.
    """
    g = np.asarray(grid, dtype=int)

    separator = get_bg(g)
    panel_values = g[g != separator]
    assert panel_values.size > 0

    counts = Counter(map(int, panel_values))
    blank, blank_count = counts.most_common(1)[0]
    assert all(blank_count > count for color, count in counts.items() if color != blank)

    panels = find_components(g, bg=separator, connectivity=4, monochromatic=False)
    assert panels

    boxes = [panel["bbox"] for panel in panels]
    panel_shapes = {(r1 - r0 + 1, c1 - c0 + 1) for r0, c0, r1, c1 in boxes}
    assert len(panel_shapes) == 1
    height, width = next(iter(panel_shapes))

    out = np.full((height, width), blank, dtype=int)
    for r0, c0, r1, c1 in boxes:
        patch = g[r0 : r1 + 1, c0 : c1 + 1]
        assert patch.shape == (height, width)
        assert np.all(patch != separator)

        mark_mask = patch != blank
        assert np.any(mark_mask)
        assert not np.any((out != blank) & mark_mask)
        out[mark_mask] = patch[mark_mask]

    return out

# --------------- generator (o5x) ---------------

# Canvas/panel geometry.
CANVAS_SHAPE, PANEL_HEIGHT_BOUNDS, PANEL_WIDTH_BOUNDS, PANEL_COUNT_BOUNDS = (16, 16), (3, 5), (3, 5), (2, 4)
# Mark blob sizing and preferred output coverage.
BLANK_COLOR, MARK_SIZE_VALUES, COVERAGE_MIN, COVERAGE_MAX = 0, range(2, 6), 0.45, 0.80
# Color/placement constraints.
ARC_COLORS, PANEL_BORDER, PANEL_GAP, PADDING_CONNECTIVITY = range(1, 10), 1, 2, 8
# Retry limit and 4-neighbor growth.
MAX_GENERATION_ATTEMPTS, NEIGHBORS4 = 300, ((-1, 0), (1, 0), (0, -1), (0, 1))

def _gen_neighbors4(r, c, h, w):
    for dr, dc in NEIGHBORS4:
        rr, cc = r + dr, c + dc
        if 0 <= rr < h and 0 <= cc < w:
            yield rr, cc


def _gen_size_tuple(h, w, n):
    """Choose per-panel mark sizes near the observed output coverage range."""
    area = h * w
    tuples = [sizes for sizes in itertools.product(MARK_SIZE_VALUES, repeat=n) if sum(sizes) <= area]
    preferred = [sizes for sizes in tuples if COVERAGE_MIN <= sum(sizes) / area <= COVERAGE_MAX]
    if preferred:
        return random.choice(preferred)
    def coverage_distance(sizes):
        coverage = sum(sizes) / area
        return min(abs(coverage - COVERAGE_MIN), abs(coverage - COVERAGE_MAX))

    best_distance = min(coverage_distance(sizes) for sizes in tuples)
    closest = [sizes for sizes in tuples if coverage_distance(sizes) == best_distance]
    return random.choice(closest)


def _gen_blob_cells(h, w, size, occupied):
    """Grow one connected mark inside the panel, avoiding already-used cells."""
    free = [(r, c) for r in range(h) for c in range(w) if (r, c) not in occupied]
    if len(free) < size:
        return None

    blob = {random.choice(free)}
    frontier = {
        nbr
        for cell in blob
        for nbr in _gen_neighbors4(*cell, h, w)
        if nbr not in occupied and nbr not in blob
    }
    while len(blob) < size:
        if not frontier:
            return None
        cell = random.choice(tuple(frontier))
        frontier.remove(cell)
        blob.add(cell)
        blocked = occupied | blob
        for nbr in _gen_neighbors4(cell[0], cell[1], h, w):
            if nbr not in blocked:
                frontier.add(nbr)
    return blob


def _gen_output_panel(h, w, mark_colors, sizes):
    """Render non-overlapping connected marks, or return None if packing fails."""
    out = np.full((h, w), BLANK_COLOR, dtype=int)
    occupied = set()
    for color, size in zip(mark_colors, sizes):
        cells = _gen_blob_cells(h, w, size, occupied)
        if cells is None:
            return None
        for r, c in cells:
            out[r, c] = color
        occupied.update(cells)
    return out


def _place_panel_patches(out, separator, mark_colors):
    """Scatter one same-sized single-color patch per mark color on the canvas."""
    grid = np.full(CANVAS_SHAPE, separator, dtype=int)
    for color in mark_colors:
        patch = np.full_like(out, BLANK_COLOR)
        patch[out == color] = color
        try:
            r, c = random_free_location(
                grid,
                patch,
                transparent=separator,
                border=PANEL_BORDER,
                padding=PANEL_GAP,
                padding_connectivity=PADDING_CONNECTIVITY,
            )
        except ValueError:
            return None
        paste(grid, patch, r, c)
    return grid


def _passes_oracle(grid, expected):
    """Accept only examples whose solver output is exactly the composed panel."""
    try:
        return np.array_equal(main(grid), expected)
    except AssertionError:
        return False


def _gen_attempt():
    h = random.randint(*PANEL_HEIGHT_BOUNDS)
    w = random.randint(*PANEL_WIDTH_BOUNDS)
    n = random.randint(*PANEL_COUNT_BOUNDS)

    separator = random.choice(ARC_COLORS)
    mark_colors = random.sample([c for c in ARC_COLORS if c != separator], n)
    sizes = _gen_size_tuple(h, w, n)

    out = _gen_output_panel(h, w, mark_colors, sizes)
    if out is None:
        return None

    grid = _place_panel_patches(out, separator, mark_colors)
    if grid is None:
        return None

    return grid if _passes_oracle(grid, out) else None


def generate_input():
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input for 7c9b52a0")
