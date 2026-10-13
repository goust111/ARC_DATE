import math
import random
from collections import Counter

import numpy as np


MAX_GENERATION_ATTEMPTS = 400
ARC_GRID_LIMIT = 30
COLOR_POOL = tuple(range(10))

SEGMENT_COUNT_RANGE = (2, 5)
GRID_HEIGHT_RANGE = (10, 24)
GRID_WIDTH_UPPER = 28
SPACER_COLS = 1
MIN_OBJECTS_EXTRA = 0
MAX_OBJECTS_EXTRA = 4
OBJECT_MASK_ATTEMPTS = 80
MIN_OBJECT_AREA = 2
MAX_OBJECT_AREA = 10


def _most_common(values) -> int:
    counts = Counter(int(v) for v in values)
    best_color = None
    best_count = -1
    for color, count in counts.items():
        if count > best_count:
            best_color = color
            best_count = count
    return int(best_color)


def _read_color_runs(column: np.ndarray, skip_color: int | None = None) -> list[int]:
    runs: list[int] = []
    prev = None
    for value in column.tolist():
        color = int(value)
        if color == prev:
            continue
        prev = color
        if skip_color is not None and color == skip_color:
            continue
        runs.append(color)
    return runs


def _infer_background(interior: np.ndarray, cycle_colors: list[int], fallback: int) -> int:
    if interior.size == 0:
        return int(fallback)

    cycle_set = set(int(color) for color in cycle_colors)
    counts = Counter(int(v) for v in interior.flat)
    best_color = None
    best_count = -1
    for color, count in counts.items():
        if color in cycle_set:
            continue
        if count > best_count:
            best_color = color
            best_count = count
    if best_color is not None:
        return int(best_color)
    return int(fallback)


def _apply_cycle(grid: np.ndarray, bg: int, cycle_colors: list[int]) -> np.ndarray:
    out = np.array(grid, dtype=int, copy=True)
    out[:, 0] = int(bg)
    if out.shape[1] == 1 or not cycle_colors:
        return out

    source = np.array(grid[:, 1:], dtype=int, copy=False)
    mapped = source.copy()
    for idx, color in enumerate(cycle_colors):
        mapped[source == int(color)] = int(cycle_colors[(idx + 1) % len(cycle_colors)])
    out[:, 1:] = mapped
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor every interior object by the next color in the left-edge cyclic key strip.

    Input structure:
    - The background is the dominant color in the interior region.
    - The leftmost column is a solid vertical key strip made of distinct color runs,
      each at least two cells tall and with no background gaps.
    - To the right of the key strip sit several monochromatic 4-connected objects
      drawn only with colors that appear in the strip.

    Rule:
    - Read the key-strip colors from top to bottom to form a cyclic mapping:
      color_1 -> color_2 -> ... -> color_n -> color_1.
    - Recolor every non-background interior object pixel with the next color in
      that cycle, preserving each object's shape and position.
    - Clear the key strip itself to the background color.

    Key insight:
    - The edge strip is an ordered replacement cycle, so the top-to-bottom run
      order determines directional color substitution and the last color wraps
      back to the first.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    interior = g[:, 1:] if g.shape[1] > 1 else g
    rough_bg = _most_common(interior.flat)
    rough_cycle = _read_color_runs(g[:, 0], skip_color=rough_bg)
    bg = _infer_background(interior, rough_cycle, rough_bg)
    cycle_colors = _read_color_runs(g[:, 0], skip_color=bg)

    return _apply_cycle(g, bg, cycle_colors)


def _partition_height(total: int, parts: int) -> list[int]:
    lengths = [2] * parts
    extra = total - 2 * parts
    for _ in range(extra):
        lengths[random.randrange(parts)] += 1
    return lengths


def _trim_mask(mask: np.ndarray) -> np.ndarray:
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    return mask[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1]


def _random_transform(mask: np.ndarray) -> np.ndarray:
    transformed = np.rot90(mask, random.randrange(4))
    if random.random() < 0.5:
        transformed = np.fliplr(transformed)
    if random.random() < 0.5:
        transformed = np.flipud(transformed)
    return _trim_mask(np.array(transformed, dtype=bool, copy=True))


def _rectangle_mask() -> np.ndarray:
    h = random.randint(1, 4)
    w = random.randint(1, 4)
    if h * w == 1:
        h, w = random.choice([(1, 2), (2, 1)])
    return np.ones((h, w), dtype=bool)


def _l_mask() -> np.ndarray:
    h = random.randint(2, 5)
    w = random.randint(2, 5)
    thickness = 1 if min(h, w) <= 3 else random.randint(1, 2)
    mask = np.zeros((h, w), dtype=bool)
    mask[:, :thickness] = True
    mask[-thickness:, :] = True
    return mask


def _t_mask() -> np.ndarray:
    h = random.randint(2, 5)
    w = random.randint(3, 5)
    mask = np.zeros((h, w), dtype=bool)
    stem_w = 1 if w < 5 else random.choice((1, 2))
    c0 = (w - stem_w) // 2
    mask[0, :] = True
    mask[:, c0 : c0 + stem_w] = True
    return mask


def _plus_mask() -> np.ndarray:
    h = random.choice((3, 5))
    w = random.choice((3, 5))
    mask = np.zeros((h, w), dtype=bool)
    mask[h // 2, :] = True
    mask[:, w // 2] = True
    return mask


def _stair_mask() -> np.ndarray:
    h = random.randint(2, 5)
    w = random.randint(2, 5)
    mask = np.zeros((h, w), dtype=bool)
    for r in range(h):
        end = min(w, r + 1)
        mask[r, :end] = True
    return mask


def _random_connected_mask() -> np.ndarray:
    h = random.randint(2, 5)
    w = random.randint(2, 5)
    target = random.randint(3, min(MAX_OBJECT_AREA, h * w))
    mask = np.zeros((h, w), dtype=bool)
    start = (random.randrange(h), random.randrange(w))
    mask[start] = True

    while int(mask.sum()) < target:
        frontier: list[tuple[int, int]] = []
        for r, c in np.argwhere(mask):
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr = int(r + dr)
                nc = int(c + dc)
                if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc]:
                    frontier.append((nr, nc))
        if not frontier:
            break
        nr, nc = random.choice(frontier)
        mask[nr, nc] = True
    return mask


def _sample_object_mask() -> np.ndarray:
    builders = (
        _rectangle_mask,
        _l_mask,
        _t_mask,
        _plus_mask,
        _stair_mask,
        _random_connected_mask,
    )
    for _ in range(OBJECT_MASK_ATTEMPTS):
        mask = random.choice(builders)()
        mask = _random_transform(mask)
        area = int(mask.sum())
        if MIN_OBJECT_AREA <= area <= MAX_OBJECT_AREA:
            return mask
    raise RuntimeError("failed to sample an object mask")


def _fits_with_padding(
    occupied: np.ndarray,
    mask: np.ndarray,
    top: int,
    left: int,
) -> bool:
    h, w = occupied.shape
    for r, c in np.argwhere(mask):
        gr = int(top + r)
        gc = int(left + c)
        for nr in range(max(0, gr - 1), min(h, gr + 2)):
            for nc in range(max(0, gc - 1), min(w, gc + 2)):
                if occupied[nr, nc]:
                    return False
    return True


def _candidate_positions(occupied: np.ndarray, mask: np.ndarray, min_col: int) -> list[tuple[int, int]]:
    h, w = occupied.shape
    mh, mw = mask.shape
    positions: list[tuple[int, int]] = []
    for top in range(h - mh + 1):
        for left in range(min_col, w - mw + 1):
            if _fits_with_padding(occupied, mask, top, left):
                positions.append((top, left))
    random.shuffle(positions)
    return positions


def _place_object(grid: np.ndarray, occupied: np.ndarray, mask: np.ndarray, color: int, min_col: int) -> bool:
    positions = _candidate_positions(occupied, mask, min_col)
    if not positions:
        return False
    top, left = positions[0]
    coords = np.argwhere(mask)
    for r, c in coords:
        gr = int(top + r)
        gc = int(left + c)
        grid[gr, gc] = int(color)
        occupied[gr, gc] = True
    return True


def _build_strip(height: int, colors: list[int], lengths: list[int], bg: int) -> np.ndarray:
    grid = np.full((height, 1), int(bg), dtype=int)
    start = 0
    for color, length in zip(colors, lengths):
        grid[start : start + length, 0] = int(color)
        start += length
    return grid


def _build_expected(grid: np.ndarray, bg: int, cycle_colors: list[int]) -> np.ndarray:
    return _apply_cycle(grid, bg, cycle_colors)


def _background_is_dominant(interior: np.ndarray, bg: int) -> bool:
    bg_count = int(np.count_nonzero(interior == int(bg)))
    if bg_count == 0:
        return False
    for color in np.unique(interior):
        color = int(color)
        if color == int(bg):
            continue
        if int(np.count_nonzero(interior == color)) >= bg_count:
            return False
    return True


def _gen_attempt() -> np.ndarray | None:
    color_count = random.randint(*SEGMENT_COUNT_RANGE)
    height = random.randint(max(GRID_HEIGHT_RANGE[0], 2 * color_count), GRID_HEIGHT_RANGE[1])
    segment_lengths = _partition_height(height, color_count)

    bg = random.choice(COLOR_POOL)
    cycle_colors = random.sample([c for c in COLOR_POOL if c != bg], color_count)

    counts = [1] * color_count
    for _ in range(random.randint(MIN_OBJECTS_EXTRA, MAX_OBJECTS_EXTRA)):
        counts[random.randrange(color_count)] += 1

    objects: list[tuple[int, np.ndarray]] = []
    for color, count in zip(cycle_colors, counts):
        for _ in range(count):
            objects.append((int(color), _sample_object_mask()))
    random.shuffle(objects)
    objects.sort(key=lambda item: int(item[1].sum()), reverse=True)

    total_area = sum(int(mask.sum()) for _color, mask in objects)
    min_inner_cols = max(6, math.ceil(total_area / max(1.0, 0.28 * height)))
    min_width = 1 + SPACER_COLS + min_inner_cols
    if min_width > ARC_GRID_LIMIT:
        return None
    max_width = min(ARC_GRID_LIMIT, max(min_width, GRID_WIDTH_UPPER))
    width = random.randint(min_width, max_width)

    grid = np.full((height, width), int(bg), dtype=int)
    grid[:, :1] = _build_strip(height, cycle_colors, segment_lengths, bg)
    occupied = np.zeros((height, width), dtype=bool)

    min_col = 1 + SPACER_COLS
    for color, mask in objects:
        if not _place_object(grid, occupied, mask, color, min_col):
            return None

    interior = grid[:, 1:]
    if not _background_is_dominant(interior, bg):
        return None

    expected = _build_expected(grid, bg, cycle_colors)
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
            assert grid.ndim == 2 and max(grid.shape) <= ARC_GRID_LIMIT
            return grid
    raise RuntimeError("failed to generate a valid cyclic color-key input")
