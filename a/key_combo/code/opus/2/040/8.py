from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


DIVIDER_COLOR = 0
SHAPE_COLOR = 5
REGION_BG_COLOR = 9

UNIQUE_OUT = 2
RARE_OUT = 4
COMMON_OUT = 1

GEN_RETRIES = 400
LAYOUTS = ((3, 3), (4, 3))
COUNT_PATTERNS = {
    9: (
        (1, 2, 3, 3),
        (1, 2, 2, 4),
        (1, 1, 2, 2, 3),
        (1, 2, 6),
    ),
    12: (
        (1, 2, 3, 3, 3),
        (1, 2, 2, 3, 4),
        (1, 1, 2, 2, 3, 3),
        (1, 2, 4, 5),
        (1, 2, 3, 6),
    ),
}


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(values[int(np.argmax(counts))])


def _crop_bbox(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    if coords.size == 0:
        raise ValueError("cannot crop an empty mask")
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return mask[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1].copy()


def _shape_key(mask: np.ndarray) -> tuple[tuple[int, ...], ...]:
    cropped = _crop_bbox(np.asarray(mask, dtype=bool))
    return tuple(tuple(int(cell) for cell in row) for row in cropped.astype(int))


def _components4(mask: np.ndarray) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((int(rr), int(cc)))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            components.append(np.array(coords, dtype=int))

    return components


def _non_divider_spans(divider_mask: np.ndarray) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    start: int | None = None
    for idx, is_divider in enumerate(np.asarray(divider_mask, dtype=bool)):
        if not is_divider and start is None:
            start = int(idx)
        elif is_divider and start is not None:
            spans.append((int(start), int(idx - 1)))
            start = None
    if start is not None:
        spans.append((int(start), int(len(divider_mask) - 1)))
    return spans


def _count_to_color(count: int) -> int:
    if count == 1:
        return UNIQUE_OUT
    if count == 2:
        return RARE_OUT
    return COMMON_OUT


def _extract_region_key(region: np.ndarray) -> tuple[tuple[int, ...], ...]:
    region = np.asarray(region, dtype=int)
    region_bg = REGION_BG_COLOR if np.any(region == REGION_BG_COLOR) else _most_common_color(region)
    fg_mask = region != region_bg
    assert fg_mask.any(), "each region must contain one shape"
    components = _components4(fg_mask)
    assert len(components) == 1, "each region must contain exactly one 4-connected shape"
    return _shape_key(fg_mask)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Compress a divider-partitioned grid into one pixel per region colored by global shape frequency.

    Input structure
    - Black full-row and full-column divider lines partition the canvas into a regular grid of rectangular regions.
    - Each region contains a maroon background and exactly one 4-connected gray shape.

    Rule
    - Recover the region layout from the black divider lines.
    - Normalize each region's shape to its bounding-box mask, count identical masks across all regions, and map frequency 1 to red, 2 to yellow, and 3 or more to blue.
    - Output the compressed region-layout grid with one pixel per region using that frequency color.

    Key insight
    - The output color of a region is a global statistic of repeated shapes, so the correct label cannot be inferred from that region alone.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0

    row_spans = _non_divider_spans(np.all(g == DIVIDER_COLOR, axis=1))
    col_spans = _non_divider_spans(np.all(g == DIVIDER_COLOR, axis=0))
    assert row_spans and col_spans, "expected black divider rows and columns"

    keys: list[tuple[tuple[int, ...], ...]] = []
    for r0, r1 in row_spans:
        for c0, c1 in col_spans:
            region = g[r0 : r1 + 1, c0 : c1 + 1]
            keys.append(_extract_region_key(region))

    counts = Counter(keys)
    out = np.zeros((len(row_spans), len(col_spans)), dtype=int)
    idx = 0
    for rr in range(len(row_spans)):
        for cc in range(len(col_spans)):
            out[rr, cc] = _count_to_color(int(counts[keys[idx]]))
            idx += 1
    return out


def _random_connected_shape(max_h: int, max_w: int) -> np.ndarray | None:
    if max_h < 2 or max_w < 2:
        return None

    for _ in range(80):
        h = random.randint(2, max_h)
        w = random.randint(2, max_w)
        max_area = min(8, h * w)
        if max_area < 3:
            continue
        area = random.randint(3, max_area)

        cells = {(random.randrange(h), random.randrange(w))}
        frontier = set()
        while len(cells) < area:
            for r, c in tuple(cells):
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr = r + dr
                    nc = c + dc
                    if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                        frontier.add((nr, nc))
            if not frontier:
                break
            choice = random.choice(tuple(frontier))
            frontier.discard(choice)
            cells.add(choice)

        if len(cells) != area:
            continue

        mask = np.zeros((h, w), dtype=bool)
        for r, c in cells:
            mask[int(r), int(c)] = True
        mask = _crop_bbox(mask)

        if random.random() < 0.5:
            mask = np.flipud(mask)
        if random.random() < 0.5:
            mask = np.fliplr(mask)

        return np.asarray(mask, dtype=bool)

    return None


def _sample_distinct_shapes(count: int, cell_h: int, cell_w: int) -> list[np.ndarray] | None:
    shapes: dict[tuple[tuple[int, ...], ...], np.ndarray] = {}
    for _ in range(500):
        shape = _random_connected_shape(cell_h - 1, cell_w - 1)
        if shape is None:
            continue
        if shape.shape[0] > cell_h or shape.shape[1] > cell_w:
            continue
        key = _shape_key(shape)
        shapes.setdefault(key, shape.astype(bool))
        if len(shapes) == count:
            return list(shapes.values())
    return None


def _render_region(cell_h: int, cell_w: int, shape: np.ndarray) -> np.ndarray:
    patch = np.full((cell_h, cell_w), REGION_BG_COLOR, dtype=int)
    h, w = shape.shape
    top = random.randint(0, cell_h - h)
    left = random.randint(0, cell_w - w)
    view = patch[top : top + h, left : left + w]
    view[shape.astype(bool)] = SHAPE_COLOR
    return patch


def _gen_attempt() -> np.ndarray | None:
    n_rows, n_cols = random.choice(LAYOUTS)
    cell_h = random.randint(4, 6)
    cell_w = random.randint(4, 6)
    region_count = n_rows * n_cols

    counts = random.choice(COUNT_PATTERNS[region_count])
    shapes = _sample_distinct_shapes(len(counts), cell_h, cell_w)
    if shapes is None:
        return None

    assignments: list[tuple[np.ndarray, int]] = []
    for shape, freq in zip(shapes, counts):
        assignments.extend((shape, int(freq)) for _ in range(freq))
    random.shuffle(assignments)

    height = n_rows * cell_h + (n_rows - 1)
    width = n_cols * cell_w + (n_cols - 1)
    if height > 30 or width > 30:
        return None

    inp = np.full((height, width), DIVIDER_COLOR, dtype=int)
    expected = np.zeros((n_rows, n_cols), dtype=int)

    idx = 0
    for rr in range(n_rows):
        for cc in range(n_cols):
            top = rr * (cell_h + 1)
            left = cc * (cell_w + 1)
            shape, freq = assignments[idx]
            inp[top : top + cell_h, left : left + cell_w] = _render_region(cell_h, cell_w, shape)
            expected[rr, cc] = _count_to_color(freq)
            idx += 1

    if len(set(map(int, expected.ravel()))) < 3:
        return None
    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
