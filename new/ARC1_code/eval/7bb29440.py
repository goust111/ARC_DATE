from __future__ import annotations

import numpy as np

from arc_utils import crop_bbox, find_components, get_bg, paste, random_free_location
import random


def _unique_dominant_color(crop: np.ndarray) -> tuple[int, int]:
    colors, counts = np.unique(crop, return_counts=True)
    max_count = int(counts.max())
    assert int(np.count_nonzero(counts == max_count)) == 1, "expected a unique object fill color"
    fill = int(colors[int(np.argmax(counts))])
    return fill, max_count


def main(grid: np.ndarray) -> np.ndarray:
    """
    Copy the solid foreground rectangle that has the fewest internal marker pixels.

    Input structure
    - The input has a background and several separated 4-connected foreground rectangles.
    - Each rectangle is mostly one fill color, with marker pixels of other foreground colors.

    Rule
    - Crop every solid foreground rectangle.
    - Count a rectangle's markers as the pixels that differ from its unique dominant fill color.
    - Return the crop of the single rectangle with the smallest marker count.

    Key insight
    - The wanted object is selected by marker sparsity, not by size, position, or marker color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = get_bg(g)
    objects = find_components(g, bg=bg, connectivity=4, monochromatic=False)
    assert objects, "expected at least one foreground rectangle"

    scored: list[tuple[int, np.ndarray]] = []
    for obj in objects:
        crop = crop_bbox(g, obj["bbox"])
        assert int(obj["area"]) == int(crop.size), "expected each foreground object to fill its bbox"

        fill, fill_count = _unique_dominant_color(crop)
        assert fill != bg, "object fill must be foreground"
        marker_count = int(crop.size) - fill_count
        scored.append((marker_count, crop))

    min_markers = min(marker_count for marker_count, _ in scored)
    winners = [crop for marker_count, crop in scored if marker_count == min_markers]
    assert len(winners) == 1, "expected a unique rectangle with the fewest markers"
    return winners[0]

# --------------- generator (o5x) ---------------
# Canvas bounds and packing density.
GRID_ROW_MIN, GRID_ROW_MAX = 12, 22
GRID_COL_MIN, GRID_COL_MAX = 11, 25
MIN_CANVAS_AREA_FACTOR, BACKGROUND_COLOR = 2, 0
# Rectangle and marker ranges.
RECT_COUNT_MIN, RECT_COUNT_MAX = 2, 7
RECT_SIDE_MIN, RECT_SIDE_MAX = 3, 10
RECT_FILL_COLOR, MARKER_COLORS = 1, (4, 6)
WINNER_MARKER_MIN, WINNER_MARKER_MAX = 1, 3
MAX_MARKERS_PER_RECT, MIN_CELLS_PER_MARKER = 6, 2
# Layout compaction, marker colors, placement, retries.
DENSE_RECT_COUNT_MIN, DENSE_RECT_SIDE_MAX, DENSE_RECT_PROB = 6, 7, 0.75
MID_RECT_COUNT, MID_RECT_SIDE_MAX, MID_RECT_PROB = 5, 8, 0.55
TWO_COLOR_MIN_MARKERS, TWO_COLOR_MARKER_PROB = 2, 0.55
RECT_PADDING, PADDING_CONNECTIVITY = 1, 8
MAX_GENERATION_ATTEMPTS = 300


def _gen_dims_for_markers(marker_count: int, max_side: int) -> tuple[int, int]:
    pairs = [
        (h, w)
        for h in range(RECT_SIDE_MIN, int(max_side) + 1)
        for w in range(RECT_SIDE_MIN, int(max_side) + 1)
        if h * w >= MIN_CELLS_PER_MARKER * int(marker_count) + 1
    ]
    return random.choice(pairs)


def _gen_rect_patch(h: int, w: int, marker_count: int) -> np.ndarray:
    patch = np.full((int(h), int(w)), RECT_FILL_COLOR, dtype=int)
    coords = random.sample(
        [(r, c) for r in range(int(h)) for c in range(int(w))],
        int(marker_count),
    )

    if int(marker_count) >= TWO_COLOR_MIN_MARKERS and random.random() < TWO_COLOR_MARKER_PROB:
        marker_colors = MARKER_COLORS
    else:
        marker_colors = [random.choice(MARKER_COLORS)]

    for i, (r, c) in enumerate(coords):
        if len(marker_colors) == 2 and i < 2:
            patch[r, c] = marker_colors[i]
        else:
            patch[r, c] = random.choice(marker_colors)
    return patch


def _gen_rect_specs() -> tuple[list[tuple[int, int, int]], int]:
    """Build rectangle specs and return the index of the unique sparse winner."""
    n_rects = random.randint(RECT_COUNT_MIN, RECT_COUNT_MAX)
    winner_idx = random.randrange(n_rects)
    winner_markers = random.randint(WINNER_MARKER_MIN, WINNER_MARKER_MAX)

    rect_specs: list[tuple[int, int, int]] = []
    for i in range(n_rects):
        marker_count = winner_markers
        if i != winner_idx:
            marker_count = random.randint(winner_markers + 1, MAX_MARKERS_PER_RECT)
        side_limit = RECT_SIDE_MAX
        if n_rects >= DENSE_RECT_COUNT_MIN and random.random() < DENSE_RECT_PROB:
            side_limit = DENSE_RECT_SIDE_MAX
        elif n_rects == MID_RECT_COUNT and random.random() < MID_RECT_PROB:
            side_limit = MID_RECT_SIDE_MAX
        h, w = _gen_dims_for_markers(marker_count, side_limit)
        rect_specs.append((h, w, marker_count))
    return rect_specs, winner_idx


def _gen_attempt() -> np.ndarray | None:
    rect_specs, winner_idx = _gen_rect_specs()
    total_area = sum(h * w for h, w, _ in rect_specs)
    max_h = max(h for h, _, _ in rect_specs)
    max_w = max(w for _, w, _ in rect_specs)
    grid_choices = [
        (h, w)
        for h in range(max(GRID_ROW_MIN, max_h), GRID_ROW_MAX + 1)
        for w in range(max(GRID_COL_MIN, max_w), GRID_COL_MAX + 1)
        if h * w > MIN_CANVAS_AREA_FACTOR * total_area
    ]
    if not grid_choices:
        return None
    grid = np.zeros(random.choice(grid_choices), dtype=int)

    patches = [_gen_rect_patch(h, w, m) for h, w, m in rect_specs]
    expected = patches[winner_idx].copy()
    order = sorted(range(len(patches)), key=lambda i: patches[i].size + random.random(), reverse=True)

    for i in order:
        try:
            r, c = random_free_location(
                grid,
                patches[i],
                transparent=BACKGROUND_COLOR,
                padding=RECT_PADDING,
                padding_connectivity=PADDING_CONNECTIVITY,
            )
        except ValueError:
            return None
        paste(grid, patches[i], r, c)

    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 7bb29440 input")
