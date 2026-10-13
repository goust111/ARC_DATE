from __future__ import annotations

import random

import numpy as np


BG = 0
COLORS = tuple(range(1, 10))
NEIGHBORS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))

# Generator bounds.
PATTERN_COUNT_RANGE = (2, 4)
LAYER_COUNT_RANGE = (3, 5)
CENTER_SIZE_CHOICES = (1, 2)
MARKER_ROW_STEP = 2
PATTERN_PADDING = 1
MARKER_GAP_COL_RANGE = (2, 4)
MIN_LEFT_WIDTH = 12
MAX_ATTEMPTS = 400


def _components(mask: np.ndarray) -> list[dict[str, object]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[dict[str, object]] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue

            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in NEIGHBORS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))

            arr = np.array(coords, dtype=int)
            r0, c0 = arr.min(axis=0)
            r1, c1 = arr.max(axis=0)
            comps.append(
                {
                    "coords": arr,
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    "area": int(len(coords)),
                }
            )
    return comps


def _extract_layer_colors(patch: np.ndarray) -> list[int]:
    work = np.asarray(patch, dtype=int)
    colors: list[int] = []

    while True:
        h, w = work.shape
        assert h == w and h >= 1, "pattern patch must stay square while peeling rings"
        assert np.all(work != BG), "pattern interior must be fully filled"

        border = np.zeros_like(work, dtype=bool)
        border[0, :] = True
        border[-1, :] = True
        border[:, 0] = True
        border[:, -1] = True
        border_colors = {int(color) for color in np.unique(work[border])}
        assert len(border_colors) == 1, "each ring border must be monochrome"
        color = border_colors.pop()
        assert color != BG, "ring color must be non-background"
        colors.append(color)

        if h <= 2:
            assert np.all(work == color), "final 1x1 or 2x2 core must be solid"
            break

        work = work[1:-1, 1:-1]

    assert 3 <= len(colors) <= 5, "expected 3-5 nested layers"
    assert len(set(colors)) == len(colors), "layers inside one pattern must use distinct colors"
    return colors


def _pattern_sequences(grid: np.ndarray) -> tuple[dict[int, list[int]], list[tuple[int, int, int, int]]]:
    patterns: dict[int, list[int]] = {}
    boxes: list[tuple[int, int, int, int]] = []

    for comp in _components(grid != BG):
        area = int(comp["area"])
        r0, c0, r1, c1 = comp["bbox"]
        if area == 1:
            continue

        h = r1 - r0 + 1
        w = c1 - c0 + 1
        assert h == w, "nested target must have a square bbox"
        patch = grid[r0 : r1 + 1, c0 : c1 + 1]
        assert int(np.count_nonzero(patch != BG)) == h * w == area, "pattern must fill its bbox"
        sequence = _extract_layer_colors(patch)
        outer = sequence[0]
        assert outer not in patterns, "outer ring colors must uniquely identify patterns"
        patterns[outer] = sequence
        boxes.append((r0, c0, r1, c1))

    assert patterns, "expected at least one nested pattern"
    return patterns, boxes


def _markers(grid: np.ndarray) -> list[tuple[int, int, int]]:
    markers: list[tuple[int, int, int]] = []
    for comp in _components(grid != BG):
        if int(comp["area"]) != 1:
            continue
        r, c = map(int, comp["coords"][0])
        markers.append((r, c, int(grid[r, c])))

    assert markers, "expected a right-side marker column"
    cols = {c for _r, c, _color in markers}
    assert len(cols) == 1, "all marker pixels must share one column"
    rows = [r for r, _c, _color in sorted(markers)]
    assert len(rows) == len(set(rows)), "marker rows must be distinct"
    assert all(b - a == MARKER_ROW_STEP for a, b in zip(rows, rows[1:])), "markers must be separated by one blank row"
    colors = [color for _r, _c, color in markers]
    assert len(colors) == len(set(colors)), "marker colors must be unique"
    return sorted(markers)


def main(grid: np.ndarray) -> np.ndarray:
    """Flatten each nested square-ring pattern into a leftward color row aligned to its marker.

    Input structure:
    - The background is black.
    - The left side contains several separated concentric square patterns made of 3-5 colored layers.
    - The right side contains a single marker column of singleton pixels, with one background row between adjacent markers.

    Rule:
    - Match every marker pixel to the unique nested pattern whose outermost ring has the same color.
    - Read that pattern's colors from the outer ring to the innermost core.
    - Clear the original patterns, keep the marker column, and draw the matched color sequence leftward on the marker row so the outer color sits immediately left of the marker and the inner color ends farthest left.

    Key insight:
    - The task first requires parsing spatial ring depth, then re-encoding that depth as a one-dimensional row ordered from outer layer on the right to inner layer on the left.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    markers = _markers(g)
    sequences, boxes = _pattern_sequences(g)
    marker_col = markers[0][1]

    assert len(markers) == len(sequences), "markers and nested patterns must match one-to-one"
    assert set(color for _r, _c, color in markers) == set(sequences), "marker colors must list all outer ring colors"
    assert all(c1 < marker_col for _r0, _c0, _r1, c1 in boxes), "patterns must lie left of the marker column"

    out = np.full_like(g, BG)
    for row, col, color in markers:
        sequence = sequences[color]
        assert col - len(sequence) >= 0, "marker must have enough space for the flattened row"
        out[row, col] = color
        for offset, layer_color in enumerate(sequence, start=1):
            out[row, col - offset] = layer_color

    return out


def _render_pattern(sequence: list[int], center_size: int) -> np.ndarray:
    side = center_size + 2 * (len(sequence) - 1)
    patch = np.full((side, side), sequence[0], dtype=int)
    for depth, color in enumerate(sequence[1:], start=1):
        patch[depth : side - depth, depth : side - depth] = int(color)
    return patch


def _place_patterns(height: int, left_width: int, sides: list[int]) -> list[tuple[int, int]] | None:
    occupied = np.zeros((height, left_width), dtype=bool)
    placements: list[tuple[int, int] | None] = [None] * len(sides)

    order = sorted(range(len(sides)), key=lambda idx: sides[idx], reverse=True)
    for idx in order:
        side = sides[idx]
        candidates = []
        for r in range(PATTERN_PADDING, height - side - PATTERN_PADDING + 1):
            for c in range(PATTERN_PADDING, left_width - side - PATTERN_PADDING + 1):
                region = occupied[
                    r - PATTERN_PADDING : r + side + PATTERN_PADDING,
                    c - PATTERN_PADDING : c + side + PATTERN_PADDING,
                ]
                if not region.any():
                    candidates.append((r, c))
        if not candidates:
            return None

        r, c = random.choice(candidates)
        occupied[
            r - PATTERN_PADDING : r + side + PATTERN_PADDING,
            c - PATTERN_PADDING : c + side + PATTERN_PADDING,
        ] = True
        placements[idx] = (r, c)

    return [(r, c) for r, c in placements if r is not None and c is not None]


def _expected_output(
    shape: tuple[int, int],
    marker_rows: list[int],
    marker_col: int,
    marker_colors: list[int],
    sequences: dict[int, list[int]],
) -> np.ndarray:
    out = np.full(shape, BG, dtype=int)
    for row, color in zip(marker_rows, marker_colors):
        out[row, marker_col] = color
        for offset, layer_color in enumerate(sequences[color], start=1):
            out[row, marker_col - offset] = layer_color
    return out


def _sample_center_sizes(count: int) -> list[int]:
    centers = [random.choice(CENTER_SIZE_CHOICES) for _ in range(count)]
    if count >= 2 and len(set(centers)) == 1:
        centers[random.randrange(count)] = 3 - centers[0]
    return centers


def _gen_attempt() -> np.ndarray | None:
    count = random.randint(*PATTERN_COUNT_RANGE)
    layer_counts = [random.randint(*LAYER_COUNT_RANGE) for _ in range(count)]
    center_sizes = _sample_center_sizes(count)
    sides = [center + 2 * (layers - 1) for center, layers in zip(center_sizes, layer_counts)]
    max_side = max(sides)

    min_height = max(max_side + 2 * PATTERN_PADDING + 1, MARKER_ROW_STEP * (count - 1) + 3)
    if min_height > 30:
        return None
    height = random.randint(min_height, 30)

    gap_cols = random.randint(*MARKER_GAP_COL_RANGE)
    min_left_width = max(MIN_LEFT_WIDTH, max_side + 2 * PATTERN_PADDING)
    max_left_width = 29 - gap_cols
    if min_left_width > max_left_width:
        return None
    left_width = random.randint(min_left_width, max_left_width)
    placements = _place_patterns(height, left_width, sides)
    if placements is None:
        return None

    marker_col = left_width + gap_cols
    width = random.randint(marker_col + 1, 30)

    max_marker_start = height - (MARKER_ROW_STEP * (count - 1) + 2)
    if max_marker_start < 1:
        return None
    marker_start = random.randint(1, max_marker_start)
    marker_rows = [marker_start + MARKER_ROW_STEP * i for i in range(count)]

    outer_colors = random.sample(COLORS, k=count)
    sequences_by_outer: dict[int, list[int]] = {}
    pattern_specs = []
    for outer, layers, center, side, pos in zip(outer_colors, layer_counts, center_sizes, sides, placements):
        inner_choices = random.sample([color for color in COLORS if color != outer], k=layers - 1)
        sequence = [outer] + inner_choices
        sequences_by_outer[outer] = sequence
        pattern_specs.append((sequence, center, side, pos))

    marker_colors = outer_colors[:]
    random.shuffle(marker_colors)

    grid = np.full((height, width), BG, dtype=int)
    for sequence, center, _side, (r, c) in pattern_specs:
        patch = _render_pattern(sequence, center)
        h, w = patch.shape
        grid[r : r + h, c : c + w] = patch

    for row, color in zip(marker_rows, marker_colors):
        grid[row, marker_col] = color

    expected = _expected_output(grid.shape, marker_rows, marker_col, marker_colors, sequences_by_outer)
    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid nested-ring input")
