from __future__ import annotations

import random
from collections import deque

import numpy as np


BLACK = 0
GREY = 5
PATTERN_SIZE = 3
LABEL_WIDTH = 1
ENTRY_HEIGHT = PATTERN_SIZE
ENTRY_WIDTH = LABEL_WIDTH + PATTERN_SIZE
LEGEND_GAP = 1
AVAILABLE_COLORS = tuple(c for c in range(10) if c not in {BLACK, GREY})

ORIENTATIONS = ("top", "left", "right")
GEN_MAX_RETRIES = 400
PLACEMENT_RETRIES = 200


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            pts: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                pts.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(pts)
    return comps


def _frame_bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int] | None:
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    if r1 - r0 < 2 or c1 - c0 < 2:
        return None

    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in points:
        mask[r - r0, c - c0] = True

    expected = np.zeros_like(mask, dtype=bool)
    expected[0, :] = True
    expected[-1, :] = True
    expected[:, 0] = True
    expected[:, -1] = True
    if not np.array_equal(mask, expected):
        return None
    return r0, c0, r1, c1


def _parse_horizontal_strip(strip: np.ndarray) -> dict[int, np.ndarray] | None:
    if strip.shape[0] != ENTRY_HEIGHT:
        return None

    mapping: dict[int, np.ndarray] = {}
    occupied = np.zeros_like(strip, dtype=bool)
    c = 0
    w = strip.shape[1]
    while c < w:
        if np.all(strip[:, c] == BLACK):
            c += 1
            continue
        if c + ENTRY_WIDTH > w:
            return None

        block = strip[:, c : c + ENTRY_WIDTH]
        if np.any((block == BLACK) | (block == GREY)):
            return None

        label_col = block[:, 0]
        if not np.all(label_col == label_col[0]):
            return None

        label = int(label_col[0])
        if label in mapping:
            return None

        pattern = block[:, 1:].copy()
        if np.unique(pattern).size < 2:
            return None

        mapping[label] = pattern
        occupied[:, c : c + ENTRY_WIDTH] = True
        c += ENTRY_WIDTH

        if c < w and not np.all(strip[:, c] == BLACK):
            return None

    if len(mapping) < 2:
        return None
    if np.any(strip[(~occupied) & (strip != BLACK)]):
        return None
    return mapping


def _parse_vertical_strip(strip: np.ndarray) -> dict[int, np.ndarray] | None:
    if strip.shape[1] != ENTRY_WIDTH:
        return None

    mapping: dict[int, np.ndarray] = {}
    occupied = np.zeros_like(strip, dtype=bool)
    r = 0
    h = strip.shape[0]
    while r < h:
        if np.all(strip[r, :] == BLACK):
            r += 1
            continue
        if r + ENTRY_HEIGHT > h:
            return None

        block = strip[r : r + ENTRY_HEIGHT, :]
        if np.any((block == BLACK) | (block == GREY)):
            return None

        label_col = block[:, 0]
        if not np.all(label_col == label_col[0]):
            return None

        label = int(label_col[0])
        if label in mapping:
            return None

        pattern = block[:, 1:].copy()
        if np.unique(pattern).size < 2:
            return None

        mapping[label] = pattern
        occupied[r : r + ENTRY_HEIGHT, :] = True
        r += ENTRY_HEIGHT

        if r < h and not np.all(strip[r, :] == BLACK):
            return None

    if len(mapping) < 2:
        return None
    if np.any(strip[(~occupied) & (strip != BLACK)]):
        return None
    return mapping


def _parse_legend(grid: np.ndarray) -> dict[int, np.ndarray]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    candidates: list[dict[int, np.ndarray]] = []

    if h >= ENTRY_HEIGHT + 1 and np.all(g[ENTRY_HEIGHT, :] == BLACK):
        mapping = _parse_horizontal_strip(g[:ENTRY_HEIGHT, :])
        if mapping is not None:
            candidates.append(mapping)

    if w >= ENTRY_WIDTH + 1 and np.all(g[:, ENTRY_WIDTH] == BLACK):
        mapping = _parse_vertical_strip(g[:, :ENTRY_WIDTH])
        if mapping is not None:
            candidates.append(mapping)

    if w >= ENTRY_WIDTH + 1 and np.all(g[:, -(ENTRY_WIDTH + 1)] == BLACK):
        mapping = _parse_vertical_strip(g[:, -ENTRY_WIDTH:])
        if mapping is not None:
            candidates.append(mapping)

    assert candidates, "expected a legend on the top, left, or right edge"
    return max(candidates, key=len)


def _find_regions(grid: np.ndarray) -> list[tuple[int, int, int, int]]:
    g = np.asarray(grid, dtype=int)
    regions: list[tuple[int, int, int, int]] = []
    for comp in _components(g == GREY):
        bbox = _frame_bbox(comp)
        if bbox is not None:
            regions.append(bbox)
    assert regions, "expected at least one grey rectangular frame"
    return regions


def _tile_pattern(height: int, width: int, pattern: np.ndarray) -> np.ndarray:
    tiled = np.empty((height, width), dtype=int)
    for r in range(height):
        for c in range(width):
            tiled[r, c] = int(pattern[r % pattern.shape[0], c % pattern.shape[1]])
    return tiled


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each grey-bordered region with the repeated pattern selected by its seed color from the edge legend.

    Input structure:
    - The background is black.
    - A legend sits on the top edge or on one side edge, made of 3x4 blocks whose left column is a solid label color and whose right 3x3 area is a fill pattern.
    - Elsewhere, one-pixel-thick grey rectangular frames enclose black interiors, and each interior contains exactly one non-black seed pixel.

    Rule:
    - Read every legend block as a mapping from its solid label color to its adjacent 3x3 pattern.
    - For each grey rectangular frame, inspect the unique seed color inside the frame.
    - Look up the matching legend pattern and tile it across the entire interior from the interior's top-left corner, overwriting the seed pixel.

    Key insight:
    - The seed pixel is only a lookup key: its color selects a legend pattern, and that pattern is the true content that must fill the whole enclosed region.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    legend = _parse_legend(g)
    out = g.copy()

    for r0, c0, r1, c1 in _find_regions(g):
        inner = g[r0 + 1 : r1, c0 + 1 : c1]
        assert inner.size > 0, "region interior must be non-empty"

        seed_positions = np.argwhere((inner != BLACK) & (inner != GREY))
        assert len(seed_positions) == 1, "each region must contain exactly one seed pixel"
        sr, sc = map(int, seed_positions[0])
        seed_color = int(inner[sr, sc])
        assert seed_color in legend, "every seed color must appear in the legend"

        pattern = legend[seed_color]
        out[r0 + 1 : r1, c0 + 1 : c1] = _tile_pattern(inner.shape[0], inner.shape[1], pattern)

    return out


def _random_pattern(rng: random.Random, colors: list[int], seen: set[bytes]) -> np.ndarray:
    while True:
        palette = rng.sample(colors, rng.randint(2, min(4, len(colors))))
        flat = [rng.choice(palette) for _ in range(PATTERN_SIZE * PATTERN_SIZE)]
        pattern = np.array(flat, dtype=int).reshape(PATTERN_SIZE, PATTERN_SIZE)
        if np.unique(pattern).size < 2:
            continue
        key = pattern.tobytes()
        if key in seen:
            continue
        seen.add(key)
        return pattern


def _canvas_shape(rng: random.Random, orientation: str, n_entries: int, n_regions: int) -> tuple[int, int]:
    if orientation == "top":
        legend_w = n_entries * ENTRY_WIDTH + (n_entries - 1) * LEGEND_GAP
        width = rng.randint(max(16, legend_w + 2), 30)
        height = rng.randint(max(14, 8 + n_regions * 3), 28)
    else:
        legend_h = n_entries * ENTRY_HEIGHT + (n_entries - 1) * LEGEND_GAP
        height = rng.randint(max(16, legend_h + 2), 30)
        width = rng.randint(max(14, 8 + n_regions * 3), 28)
    return height, width


def _place_legend(
    grid: np.ndarray,
    orientation: str,
    labels: list[int],
    patterns: dict[int, np.ndarray],
    rng: random.Random,
) -> None:
    n = len(labels)
    if orientation == "top":
        total_w = n * ENTRY_WIDTH + (n - 1) * LEGEND_GAP
        start_c = rng.randint(0, grid.shape[1] - total_w)
        for idx, label in enumerate(labels):
            c0 = start_c + idx * (ENTRY_WIDTH + LEGEND_GAP)
            grid[:ENTRY_HEIGHT, c0] = label
            grid[:ENTRY_HEIGHT, c0 + 1 : c0 + ENTRY_WIDTH] = patterns[label]
    elif orientation == "left":
        total_h = n * ENTRY_HEIGHT + (n - 1) * LEGEND_GAP
        start_r = rng.randint(0, grid.shape[0] - total_h)
        for idx, label in enumerate(labels):
            r0 = start_r + idx * (ENTRY_HEIGHT + LEGEND_GAP)
            grid[r0 : r0 + ENTRY_HEIGHT, 0] = label
            grid[r0 : r0 + ENTRY_HEIGHT, 1:ENTRY_WIDTH] = patterns[label]
    else:
        total_h = n * ENTRY_HEIGHT + (n - 1) * LEGEND_GAP
        start_r = rng.randint(0, grid.shape[0] - total_h)
        c0 = grid.shape[1] - ENTRY_WIDTH
        for idx, label in enumerate(labels):
            r0 = start_r + idx * (ENTRY_HEIGHT + LEGEND_GAP)
            grid[r0 : r0 + ENTRY_HEIGHT, c0] = label
            grid[r0 : r0 + ENTRY_HEIGHT, c0 + 1 : c0 + ENTRY_WIDTH] = patterns[label]


def _main_bounds(shape: tuple[int, int], orientation: str) -> tuple[int, int, int, int]:
    h, w = shape
    if orientation == "top":
        return 4, 0, h - 1, w - 1
    if orientation == "left":
        return 0, 5, h - 1, w - 1
    return 0, 0, h - 1, w - 6


def _far_enough(
    bbox: tuple[int, int, int, int],
    placed: list[tuple[int, int, int, int]],
    padding: int = 1,
) -> bool:
    r0, c0, r1, c1 = bbox
    for pr0, pc0, pr1, pc1 in placed:
        if not (r1 + padding < pr0 or pr1 + padding < r0 or c1 + padding < pc0 or pc1 + padding < c0):
            return False
    return True


def _place_regions(
    grid: np.ndarray,
    orientation: str,
    seed_labels: list[int],
    rng: random.Random,
) -> list[tuple[int, int, int, int]]:
    h, w = grid.shape
    main_r0, main_c0, main_r1, main_c1 = _main_bounds(grid.shape, orientation)
    usable_r0, usable_c0 = main_r0 + 1, main_c0 + 1
    usable_r1, usable_c1 = main_r1 - 1, main_c1 - 1
    placed: list[tuple[int, int, int, int]] = []

    for idx, label in enumerate(seed_labels):
        placed_one = False
        for _ in range(PLACEMENT_RETRIES):
            if idx == 0:
                inner_h = rng.randint(4, 7)
                inner_w = rng.randint(4, 8)
            else:
                inner_h = rng.randint(2, 8)
                inner_w = rng.randint(2, 8)
            box_h = inner_h + 2
            box_w = inner_w + 2

            if usable_r0 > usable_r1 - box_h + 1 or usable_c0 > usable_c1 - box_w + 1:
                return []

            r0 = rng.randint(usable_r0, usable_r1 - box_h + 1)
            c0 = rng.randint(usable_c0, usable_c1 - box_w + 1)
            bbox = (r0, c0, r0 + box_h - 1, c0 + box_w - 1)
            if not _far_enough(bbox, placed):
                continue

            grid[r0, c0 : c0 + box_w] = GREY
            grid[r0 + box_h - 1, c0 : c0 + box_w] = GREY
            grid[r0 : r0 + box_h, c0] = GREY
            grid[r0 : r0 + box_h, c0 + box_w - 1] = GREY

            sr = rng.randint(r0 + 1, r0 + box_h - 2)
            sc = rng.randint(c0 + 1, c0 + box_w - 2)
            grid[sr, sc] = label

            placed.append(bbox)
            placed_one = True
            break
        if not placed_one:
            return []

    assert h <= 30 and w <= 30
    return placed


def _build_candidate(rng: random.Random) -> np.ndarray | None:
    orientation = rng.choice(ORIENTATIONS)
    n_entries = rng.randint(2, 4)
    n_regions = n_entries + rng.randint(0, 1)
    height, width = _canvas_shape(rng, orientation, n_entries, n_regions)
    grid = np.zeros((height, width), dtype=int)

    labels = rng.sample(list(AVAILABLE_COLORS), n_entries)
    pattern_colors = [c for c in AVAILABLE_COLORS if c not in labels]
    seen_patterns: set[bytes] = set()
    patterns = {label: _random_pattern(rng, pattern_colors, seen_patterns) for label in labels}

    _place_legend(grid, orientation, labels, patterns, rng)

    seed_labels = labels.copy()
    while len(seed_labels) < n_regions:
        seed_labels.append(rng.choice(labels))
    rng.shuffle(seed_labels)

    placed = _place_regions(grid, orientation, seed_labels, rng)
    if not placed:
        return None

    if not any((r1 - r0 - 1) > PATTERN_SIZE or (c1 - c0 - 1) > PATTERN_SIZE for r0, c0, r1, c1 in placed):
        return None
    if not any(((r1 - r0 - 1) % PATTERN_SIZE != 0) or ((c1 - c0 - 1) % PATTERN_SIZE != 0) for r0, c0, r1, c1 in placed):
        return None

    try:
        out = main(grid)
    except AssertionError:
        return None

    if np.array_equal(out, grid):
        return None
    if set(seed_labels) != set(labels):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_MAX_RETRIES):
        grid = _build_candidate(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
