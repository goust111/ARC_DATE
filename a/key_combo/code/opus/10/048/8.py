from __future__ import annotations

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4

MARKER_SIZE = 2
REGION_HEIGHT = 2


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _blue_count_to_color(counts: np.ndarray) -> np.ndarray:
    out = np.zeros_like(counts, dtype=int)
    out[counts == 1] = BLUE
    out[counts == 2] = RED
    out[counts >= 3] = YELLOW
    return out


def _find_green_blocks(grid: np.ndarray) -> list[tuple[int, int, int, int]]:
    mask = np.asarray(grid == GREEN, dtype=bool)
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    blocks: list[tuple[int, int, int, int]] = []

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
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))

            rows = [rr for rr, _ in coords]
            cols = [cc for _, cc in coords]
            r0, r1 = min(rows), max(rows)
            c0, c1 = min(cols), max(cols)
            assert len(coords) == 4 and r1 - r0 == 1 and c1 - c0 == 1
            assert np.all(mask[r0 : r1 + 1, c0 : c1 + 1])
            blocks.append((r0, c0, r1, c1))

    return blocks


def _extract_regions(grid: np.ndarray) -> list[np.ndarray]:
    blocks = _find_green_blocks(np.asarray(grid, dtype=int))
    assert len(blocks) in {6, 8}

    by_rows: dict[tuple[int, int], list[tuple[int, int, int, int]]] = {}
    for block in blocks:
        by_rows.setdefault((block[0], block[2]), []).append(block)

    assert len(by_rows) in {3, 4}
    regions: list[np.ndarray] = []
    widths = set()

    for row_span in sorted(by_rows):
        pair = sorted(by_rows[row_span], key=lambda b: b[1])
        assert len(pair) == 2
        left, right = pair
        gap_width = right[1] - left[3] - 1
        assert gap_width > 0
        region = np.asarray(grid[left[0] : left[2] + 1, left[3] + 1 : right[1]], dtype=int)
        assert region.shape == (REGION_HEIGHT, gap_width)
        assert set(np.unique(region)) <= {BLACK, BLUE}
        regions.append(region)
        widths.add(gap_width)

    assert len(widths) == 1
    return regions


def main(grid: np.ndarray) -> np.ndarray:
    """Extract the blue regions between green marker pairs, overlay them, and recolor by overlap depth.

    Input structure:
    - The grid has a black background and contains 3 or 4 row-aligned pairs of solid 2x2 green marker blocks.
    - Each marker pair occupies one two-row band, and the strict rectangle between the left/right markers has the same width for every pair.
    - Inside each such rectangle, blue foreground cells form a small 4-connected pattern on black.

    Rule:
    - Identify every green 2x2 block and pair the two blocks that share the same two-row span.
    - Sort the pairs from top to bottom, extract the strict interior rectangle between each pair, and align all extracted 2xW regions on one common canvas.
    - For each output cell, count how many extracted regions contain a blue cell there, then recolor by layer count: 0->black, 1->blue, 2->red, 3 or more->yellow.

    Key insight:
    - The output colors are not copied from any single source region; they encode a counted property created only after all marker-delimited regions are overlaid in register.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    regions = _extract_regions(g)
    counts = np.zeros_like(regions[0], dtype=int)
    for region in regions:
        counts += (region == BLUE).astype(int)
    return _blue_count_to_color(counts)


GEN_ATTEMPTS = 400
PAIR_COUNT_CHOICES = (3, 4)
WIDTH_RANGE = (5, 8)
AREA_RANGE = (3, 8)
ROW_GAP_RANGE = (1, 3)
TOP_BOTTOM_PAD_RANGE = (0, 3)
SIDE_PAD_EXTRA = (3, 7)
PEAK_CHOICES = (2, 3)


def _manhattan(a: tuple[int, int], b: tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _all_region_cells(width: int) -> list[tuple[int, int]]:
    return [(r, c) for r in range(REGION_HEIGHT) for c in range(width)]


def _path_between(
    start: tuple[int, int],
    end: tuple[int, int],
) -> list[list[tuple[int, int]]]:
    if start == end:
        return [[start]]

    sr, sc = start
    er, ec = end
    paths: list[list[tuple[int, int]]] = []

    path_hv: list[tuple[int, int]] = []
    step = 1 if ec >= sc else -1
    for cc in range(sc, ec + step, step):
        path_hv.append((sr, cc))
    if sr != er:
        path_hv.append((er, ec))
    paths.append(path_hv)

    path_vh = [(sr, sc)]
    if sr != er:
        path_vh.append((er, sc))
    step = 1 if ec >= sc else -1
    for cc in range(sc + step, ec + step, step):
        path_vh.append((er, cc))
    if path_vh not in paths:
        paths.append(path_vh)

    return paths


def _sample_connected_mask(
    width: int,
    required: list[tuple[int, int]],
    forbidden: set[tuple[int, int]],
    rng: np.random.Generator,
) -> np.ndarray | None:
    required = list(dict.fromkeys(required))
    all_cells = _all_region_cells(width)

    if not required:
        choices = [cell for cell in all_cells if cell not in forbidden]
        if not choices:
            return None
        required = [choices[int(rng.integers(len(choices)))]]

    path_options = [required[:1]]
    if len(required) == 2:
        path_options = _path_between(required[0], required[1])
    elif len(required) > 2:
        return None

    valid_bases = []
    for path in path_options:
        base = list(dict.fromkeys(path))
        if any(cell in forbidden and cell not in required for cell in base):
            continue
        if len(base) <= AREA_RANGE[1]:
            valid_bases.append(base)
    if not valid_bases:
        return None

    base = valid_bases[int(rng.integers(len(valid_bases)))]
    target_area = int(rng.integers(max(AREA_RANGE[0], len(base)), AREA_RANGE[1] + 1))
    cells = set(base)

    while len(cells) < target_area:
        frontier = []
        for cell in cells:
            for nbr in _neighbors4(*cell):
                if 0 <= nbr[0] < REGION_HEIGHT and 0 <= nbr[1] < width:
                    if nbr not in cells and nbr not in forbidden:
                        frontier.append(nbr)
        if not frontier:
            return None
        chosen = frontier[int(rng.integers(len(frontier)))]
        cells.add(chosen)

    mask = np.zeros((REGION_HEIGHT, width), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    return mask


def _nearby_choices(
    width: int,
    anchor: tuple[int, int],
    *,
    max_distance: int,
    exclude: set[tuple[int, int]],
) -> list[tuple[int, int]]:
    return [
        cell
        for cell in _all_region_cells(width)
        if cell not in exclude and 1 <= _manhattan(cell, anchor) <= max_distance
    ]


def _sample_regions(
    num_pairs: int,
    width: int,
    rng: np.random.Generator,
) -> tuple[list[np.ndarray], np.ndarray] | None:
    peak_target = int(rng.choice(np.array(PEAK_CHOICES, dtype=int)))
    peak_indices = set(int(x) for x in rng.choice(num_pairs, size=peak_target, replace=False))

    focus_cell = (int(rng.integers(REGION_HEIGHT)), int(rng.integers(width)))
    all_indices = list(range(num_pairs))
    non_peak = [idx for idx in all_indices if idx not in peak_indices]

    if non_peak:
        solo_idx = int(rng.choice(np.array(non_peak, dtype=int)))
        solo_choices = [cell for cell in _all_region_cells(width) if cell != focus_cell]
    else:
        remaining = [idx for idx in all_indices if idx not in set(list(peak_indices)[:2])]
        solo_idx = remaining[0] if remaining else all_indices[-1]
        solo_choices = _nearby_choices(width, focus_cell, max_distance=4, exclude={focus_cell})
    if not solo_choices:
        return None
    solo_cell = solo_choices[int(rng.integers(len(solo_choices)))]

    double_indices: set[int] = set()
    double_cell: tuple[int, int] | None = None
    if peak_target >= 3:
        double_indices = set(int(x) for x in rng.choice(sorted(peak_indices), size=2, replace=False))
        double_choices = _nearby_choices(width, focus_cell, max_distance=4, exclude={focus_cell, solo_cell})
        if not double_choices:
            return None
        double_cell = double_choices[int(rng.integers(len(double_choices)))]

    masks: list[np.ndarray] = []
    for idx in all_indices:
        required: list[tuple[int, int]] = []
        forbidden: set[tuple[int, int]] = set()

        if idx in peak_indices:
            required.append(focus_cell)
        else:
            forbidden.add(focus_cell)

        if idx == solo_idx:
            required.append(solo_cell)
        else:
            forbidden.add(solo_cell)

        if double_cell is not None:
            if idx in double_indices:
                required.append(double_cell)
            else:
                forbidden.add(double_cell)

        mask = _sample_connected_mask(width, required, forbidden, rng)
        if mask is None:
            return None
        masks.append(mask)

    if len({mask.tobytes() for mask in masks}) != len(masks):
        return None

    counts = np.zeros((REGION_HEIGHT, width), dtype=int)
    for mask in masks:
        counts += mask.astype(int)

    if counts[focus_cell] != peak_target:
        return None
    if counts[solo_cell] != 1:
        return None
    if double_cell is not None and counts[double_cell] != 2:
        return None
    if peak_target == 2 and np.max(counts) != 2:
        return None
    if peak_target >= 3 and (np.max(counts) < 3 or not np.any(counts == 2)):
        return None
    if not np.any(counts == 0):
        return None
    if not np.any(counts == 1):
        return None
    if not np.any(counts >= 2):
        return None

    regions = [np.where(mask, BLUE, BLACK).astype(int) for mask in masks]
    return regions, counts


def _choose_layout(
    num_pairs: int,
    width: int,
    rng: np.random.Generator,
) -> tuple[int, int, list[int], list[int]]:
    row_gaps = [int(rng.integers(ROW_GAP_RANGE[0], ROW_GAP_RANGE[1] + 1)) for _ in range(num_pairs - 1)]
    top_pad = int(rng.integers(TOP_BOTTOM_PAD_RANGE[0], TOP_BOTTOM_PAD_RANGE[1] + 1))
    bottom_pad = int(rng.integers(TOP_BOTTOM_PAD_RANGE[0], TOP_BOTTOM_PAD_RANGE[1] + 1))
    height = top_pad + bottom_pad + REGION_HEIGHT * num_pairs + sum(row_gaps)

    row_starts = [top_pad]
    for gap in row_gaps:
        row_starts.append(row_starts[-1] + REGION_HEIGHT + gap)

    min_width = width + MARKER_SIZE * 2 + SIDE_PAD_EXTRA[0]
    max_width = min(30, width + MARKER_SIZE * 2 + SIDE_PAD_EXTRA[1])
    canvas_width = int(rng.integers(min_width, max_width + 1))

    left_max = canvas_width - width - MARKER_SIZE * 2
    left_choices = np.arange(left_max + 1, dtype=int)
    if len(left_choices) >= num_pairs:
        left_cols = [int(x) for x in rng.choice(left_choices, size=num_pairs, replace=False)]
    else:
        left_cols = [int(x) for x in rng.choice(left_choices, size=num_pairs, replace=True)]

    return height, canvas_width, row_starts, left_cols


def _render_input(
    regions: list[np.ndarray],
    height: int,
    width: int,
    row_starts: list[int],
    left_cols: list[int],
) -> np.ndarray:
    grid = np.zeros((height, width), dtype=int)
    for region, top, left in zip(regions, row_starts, left_cols):
        right_marker_left = left + MARKER_SIZE + region.shape[1]
        grid[top : top + MARKER_SIZE, left : left + MARKER_SIZE] = GREEN
        grid[top : top + REGION_HEIGHT, left + MARKER_SIZE : right_marker_left] = region
        grid[top : top + MARKER_SIZE, right_marker_left : right_marker_left + MARKER_SIZE] = GREEN
    return grid


def _semantic_checks(grid: np.ndarray, regions: list[np.ndarray], counts: np.ndarray) -> bool:
    try:
        predicted = main(grid)
    except Exception:
        return False

    expected = _blue_count_to_color(counts)
    if not np.array_equal(predicted, expected):
        return False
    if any(np.array_equal(expected, region) for region in regions):
        return False
    union = np.where(counts > 0, BLUE, BLACK)
    if np.array_equal(expected, union):
        return False
    return True


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    num_pairs = int(rng.choice(np.array(PAIR_COUNT_CHOICES, dtype=int)))
    width = int(rng.integers(WIDTH_RANGE[0], WIDTH_RANGE[1] + 1))

    sampled = _sample_regions(num_pairs, width, rng)
    if sampled is None:
        return None
    regions, counts = sampled

    height, canvas_width, row_starts, left_cols = _choose_layout(num_pairs, width, rng)
    grid = _render_input(regions, height, canvas_width, row_starts, left_cols)
    return grid if _semantic_checks(grid, regions, counts) else None


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_ATTEMPTS} attempts")
