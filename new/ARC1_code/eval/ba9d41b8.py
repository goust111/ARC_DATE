import numpy as np

from arc_utils import find_components, get_bg, paste
import random
from functools import lru_cache


def main(grid: np.ndarray) -> np.ndarray:
    """
    Carve an alternating background checkerboard through the interior of each solid colored rectangle.

    Input structure:
    - The background is the most common color.
    - Every non-background object is a monochrome 4-connected filled rectangle.
    - Each rectangle is at least 3x3, so it has a border and a nonempty interior.

    Rule:
    - Preserve each rectangle border unchanged.
    - Within each rectangle interior, replace cells whose row-plus-column parity is
      even relative to the interior's top-left cell with the background color.
    - Leave the other interior cells and all background cells unchanged.

    Key insight:
    - The output is not moving the rectangles; it applies the same top-left-anchored
      checkerboard mask independently inside every solid rectangular object.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = get_bg(g)
    colors, counts = np.unique(g, return_counts=True)
    bg_count = int(counts[colors == bg][0])
    other_counts = counts[colors != bg]
    assert other_counts.size == 0 or bg_count > int(other_counts.max()), "background color must be uniquely most common"
    out = g.copy()

    components = find_components(g, bg=bg, connectivity=4, monochromatic=True)
    assert components, "expected at least one colored rectangle"

    for component in components:
        color = int(component["color"])
        r0, c0, r1, c1 = component["bbox"]
        height = r1 - r0 + 1
        width = c1 - c0 + 1
        assert height >= 3 and width >= 3, "each rectangle must have an interior"
        assert int(component["area"]) == height * width, "expected a filled rectangular object"
        assert np.all(g[r0 : r1 + 1, c0 : c1 + 1] == color), "rectangle must be monochrome"

        interior = out[r0 + 1 : r1, c0 + 1 : c1]
        rows, cols = np.indices(interior.shape)
        interior[(rows + cols) % 2 == 0] = bg

    return out

# --------------- generator (o5x) ---------------

# Canvas and rectangle count bounds.
_GEN_MIN_HEIGHT, _GEN_MAX_HEIGHT = 13, 19
_GEN_MIN_WIDTH, _GEN_MAX_WIDTH = 15, 19
_GEN_MIN_RECT_COUNT, _GEN_MAX_RECT_COUNT = 1, 4
# Rectangle geometry and partition gap bounds.
_GEN_MIN_RECT_SIDE, _GEN_MAX_RECT_SIDE = 5, 10
_GEN_MIN_GAP, _GEN_MAX_GAP = 1, 3
# Palette and generation retry limits.
_GEN_BACKGROUND_COLOR = 0
_GEN_MIN_COLOR, _GEN_MAX_COLOR = 1, 9
_GEN_RETRY_LIMIT = 300


def _gen_split_candidates(height: int, width: int, count: int):
    """Yield recursive partition choices in the exact order used for sampling."""
    for first_count in range(1, count):
        second_count = count - first_count
        for gap in range(_GEN_MIN_GAP, _GEN_MAX_GAP + 1):
            for first_height in range(_GEN_MIN_RECT_SIDE, height - gap - _GEN_MIN_RECT_SIDE + 1):
                second_height = height - gap - first_height
                first_req = (first_height, width, first_count)
                second_req = (second_height, width, second_count)
                yield "horizontal", first_count, first_height, gap, first_req, second_req

            for first_width in range(_GEN_MIN_RECT_SIDE, width - gap - _GEN_MIN_RECT_SIDE + 1):
                second_width = width - gap - first_width
                first_req = (height, first_width, first_count)
                second_req = (height, second_width, second_count)
                yield "vertical", first_count, first_width, gap, first_req, second_req


def _gen_valid_splits(height: int, width: int, count: int):
    for orient, first_count, span, gap, first_req, second_req in _gen_split_candidates(height, width, count):
        if _gen_can_fit(*first_req) and _gen_can_fit(*second_req):
            yield orient, first_count, span, gap


@lru_cache(None)
def _gen_can_fit(height: int, width: int, count: int) -> bool:
    if count == 1:
        return height >= _GEN_MIN_RECT_SIDE and width >= _GEN_MIN_RECT_SIDE

    return any(True for _ in _gen_valid_splits(height, width, count))


def _gen_split_region(
    region: tuple[int, int, int, int], orient: str, span: int, gap: int
) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int]]:
    r0, c0, r1, c1 = region
    if orient == "horizontal":
        return (r0, c0, r0 + span - 1, c1), (r0 + span + gap, c0, r1, c1)
    return (r0, c0, r1, c0 + span - 1), (r0, c0 + span + gap, r1, c1)


def _gen_partition_region(region: tuple[int, int, int, int], count: int) -> list[tuple[int, int, int, int]] | None:
    if count == 1:
        return [region]

    r0, c0, r1, c1 = region
    height = r1 - r0 + 1
    width = c1 - c0 + 1
    options = list(_gen_valid_splits(height, width, count))

    if not options:
        return None

    orient, first_count, span, gap = random.choice(options)
    second_count = count - first_count
    first, second = _gen_split_region(region, orient, span, gap)

    first_regions = _gen_partition_region(first, first_count)
    second_regions = _gen_partition_region(second, second_count)
    if first_regions is None or second_regions is None:
        return None

    regions = first_regions + second_regions
    random.shuffle(regions)
    return regions


def _gen_size_candidates(region: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = region
    max_height = min(_GEN_MAX_RECT_SIDE, r1 - r0 + 1)
    max_width = min(_GEN_MAX_RECT_SIDE, c1 - c0 + 1)
    return [
        (height, width)
        for height in range(_GEN_MIN_RECT_SIDE, max_height + 1)
        for width in range(_GEN_MIN_RECT_SIDE, max_width + 1)
    ]


def _gen_choose_sizes(regions: list[tuple[int, int, int, int]], total_area: int) -> list[tuple[int, int]] | None:
    order = list(range(len(regions)))
    random.shuffle(order)
    choices = [_gen_size_candidates(region) for region in regions]
    if any(not candidates for candidates in choices):
        return None

    selected: list[tuple[int, int] | None] = [None] * len(regions)

    def choose_at(pos: int, used_area: int, largest_area: int) -> bool:
        if pos == len(order):
            return total_area - used_area > largest_area

        idx = order[pos]
        candidates = choices[idx][:]
        random.shuffle(candidates)
        remaining_min_area = (_GEN_MIN_RECT_SIDE * _GEN_MIN_RECT_SIDE) * (len(order) - pos - 1)

        for height, width in candidates:
            area = height * width
            next_used = used_area + area
            next_largest = max(largest_area, area)
            if total_area - next_used - remaining_min_area <= next_largest:
                continue
            selected[idx] = (height, width)
            if choose_at(pos + 1, next_used, next_largest):
                return True
            selected[idx] = None

        return False

    if not choose_at(0, 0, 0):
        return None
    return [size for size in selected if size is not None]


def _gen_checkerboard_output(grid: np.ndarray) -> np.ndarray:
    out = grid.copy()
    for color in np.unique(grid):
        if color == _GEN_BACKGROUND_COLOR:
            continue
        coords = np.argwhere(grid == color)
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        interior = out[r0 + 1 : r1, c0 + 1 : c1]
        rows, cols = np.indices(interior.shape)
        interior[(rows + cols) % 2 == 0] = 0
    return out


def _gen_sample_layout(
    height: int, width: int, rect_count: int
) -> tuple[list[tuple[int, int, int, int]], list[tuple[int, int]]] | None:
    """Return partition regions and rectangle sizes for one sampled canvas."""
    if not _gen_can_fit(height, width, rect_count):
        return None

    regions = _gen_partition_region((0, 0, height - 1, width - 1), rect_count)
    if regions is None:
        return None

    sizes = _gen_choose_sizes(regions, height * width)
    if sizes is None:
        return None

    return regions, sizes


def _gen_render_rectangles(
    height: int,
    width: int,
    regions: list[tuple[int, int, int, int]],
    sizes: list[tuple[int, int]],
    rect_count: int,
) -> np.ndarray:
    grid = np.zeros((height, width), dtype=int)
    colors = random.sample(range(_GEN_MIN_COLOR, _GEN_MAX_COLOR + 1), rect_count)
    for region, (rect_height, rect_width), color in zip(regions, sizes, colors):
        r0, c0, r1, c1 = region
        top = random.randint(r0, r1 - rect_height + 1)
        left = random.randint(c0, c1 - rect_width + 1)
        patch = np.full((rect_height, rect_width), color, dtype=int)
        paste(grid, patch, top, left)
    return grid


def _gen_matches_solver(grid: np.ndarray) -> bool:
    expected = _gen_checkerboard_output(grid)
    try:
        actual = main(grid)
    except AssertionError:
        return False
    return np.array_equal(actual, expected)


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(_GEN_MIN_HEIGHT, _GEN_MAX_HEIGHT)
    width = random.randint(_GEN_MIN_WIDTH, _GEN_MAX_WIDTH)
    rect_count = random.randint(_GEN_MIN_RECT_COUNT, _GEN_MAX_RECT_COUNT)

    layout = _gen_sample_layout(height, width, rect_count)
    if layout is None:
        return None

    regions, sizes = layout
    grid = _gen_render_rectangles(height, width, regions, sizes, rect_count)
    if not _gen_matches_solver(grid):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input for ba9d41b8")
