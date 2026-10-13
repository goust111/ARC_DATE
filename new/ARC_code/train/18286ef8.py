import numpy as np

from arc_utils import Color, bbox
import random


def _relative_index(coord: int, lo: int, hi: int) -> int:
    if coord < lo:
        return lo
    if coord > hi:
        return hi
    assert (hi - lo + 1) % 2 == 1, "same-band target needs a middle cell"
    return (lo + hi) // 2


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use the pink control pixel to choose where the maroon pixel belongs inside the grey 3x3 region, and turn the control pixel maroon.

    Input structure:
    - The grid contains one 3x3 region made from eight grey pixels and one maroon pixel.
    - There is exactly one pink control pixel outside that region.

    Rule:
    - Restore the old maroon cell in the 3x3 region to grey.
    - Compare the pink pixel to the 3x3 region: above/same band/below selects top/middle/bottom, and left/same band/right selects left/middle/right.
    - Paint the selected cell maroon and recolor the pink control pixel maroon.

    Key insight:
    - The pink pixel acts as a relative-position marker for a 3x3 target inside the grey region.
    """
    g = np.asarray(grid, dtype=int)
    marker = np.argwhere(g == Color.PINK)
    assert marker.shape == (1, 2), f"expected one pink control pixel, got {len(marker)}"
    pr, pc = map(int, marker[0])

    guide_mask = (g == Color.GREY) | (g == Color.MAROON)
    guide_bbox = bbox(guide_mask)
    assert guide_bbox is not None, "missing grey/maroon guide region"
    r0, c0, r1, c1 = guide_bbox
    assert (r1 - r0 + 1, c1 - c0 + 1) == (3, 3), "guide region must be 3x3"

    block = g[r0 : r1 + 1, c0 : c1 + 1]
    assert np.all((block == Color.GREY) | (block == Color.MAROON)), "guide bbox contains non-guide colors"
    old = np.argwhere(block == Color.MAROON)
    assert old.shape == (1, 2), f"expected one maroon guide pixel, got {len(old)}"
    assert np.count_nonzero(block == Color.GREY) == 8, "expected eight grey guide pixels"
    assert not (r0 <= pr <= r1 and c0 <= pc <= c1), "pink control pixel must be outside guide region"

    tr = _relative_index(pr, r0, r1)
    tc = _relative_index(pc, c0, c1)
    old_r = r0 + int(old[0, 0])
    old_c = c0 + int(old[0, 1])

    out = g.copy()
    out[old_r, old_c] = Color.GREY
    out[tr, tc] = Color.MAROON
    out[pr, pc] = Color.MAROON
    return out

# --------------- generator (o5x) ---------------

# --- CFG: tunable generation parameters ---
_BAND_SIZE_MIN = 1          # min height/width of non-center bands
_BAND_SIZE_MAX = 9          # max height/width of non-center bands
_CENTER_BAND = 3            # fixed size of center band (always 3x3)
_TOTAL_COLORED_MIN = 4      # min colored pixels (pink + distractors) across grid
_TOTAL_COLORED_MAX = 10     # max colored pixels
_OCCUPIED_CELLS_MIN = 4     # min non-center sub-cells containing >=1 colored pixel
_OCCUPIED_CELLS_MAX = 7     # max non-center sub-cells containing >=1 colored pixel
_DISTINCT_COLORS_MIN = 3    # min distinct distractor colors
_DISTINCT_COLORS_MAX = 4    # max distinct distractor colors
_MAX_DISTRACTORS_PER_CELL = 3  # cap on distractor pixels per sub-cell
_RETRY_LIMIT = 300          # max generation attempts before raising

_DISTRACTOR_COLORS = [Color.BLUE, Color.RED, Color.GREEN, Color.YELLOW, Color.TEAL]
_TARGETS = [(r, c) for r in range(3) for c in range(3) if (r, c) != (1, 1)]


def _band_spans(before: int, after: int) -> list[tuple[int, int]]:
    """Return (start, end) spans for the three bands along one axis.

    Layout: [0..before) | separator | [center 3 cells] | separator | [last..last+after).
    Spans use half-open indexing suitable for range()/slicing.
    """
    mid0 = before + 1
    last0 = mid0 + _CENTER_BAND + 1
    return [(0, before), (mid0, mid0 + _CENTER_BAND), (last0, last0 + after)]


def _cell_pixels(span_r: tuple[int, int], span_c: tuple[int, int]) -> list[tuple[int, int]]:
    """All (r, c) coordinates in the sub-cell defined by two spans."""
    return [(r, c) for r in range(*span_r) for c in range(*span_c)]


def _choose_occupied(
    cell_keys: list[tuple[int, int]],
    pink_key: tuple[int, int],
    caps: dict[tuple[int, int], int],
    occupied_count: int,
    distractor_count: int,
) -> list[tuple[int, int]] | None:
    """Select which non-center sub-cells (besides pink's) will hold distractors."""
    other_count = occupied_count - 1
    candidates = [key for key in cell_keys if key != pink_key and caps[key] > 0]
    if len(candidates) < other_count or distractor_count < other_count:
        return None

    selected = random.sample(candidates, other_count)
    while caps[pink_key] + sum(caps[key] for key in selected) < distractor_count:
        outside = [key for key in candidates if key not in selected]
        if not outside:
            return None
        low = min(selected, key=lambda key: caps[key])
        high = max(outside, key=lambda key: caps[key])
        if caps[high] <= caps[low]:
            return None
        selected.remove(low)
        selected.append(high)
    return selected


def _allocate_counts(
    occupied: list[tuple[int, int]],
    pink_key: tuple[int, int],
    caps: dict[tuple[int, int], int],
    distractor_count: int,
) -> dict[tuple[int, int], int] | None:
    """Distribute distractor_count pixels across occupied cells, each non-pink cell gets >=1."""
    counts = {key: (0 if key == pink_key else 1) for key in occupied}
    remaining = distractor_count - sum(counts.values())
    while remaining > 0:
        available = [key for key in occupied if counts[key] < caps[key]]
        if not available:
            return None
        counts[random.choice(available)] += 1
        remaining -= 1
    return counts


def _make_color_bag(distractor_count: int) -> list[int]:
    """Build a shuffled bag of distractor_count colors using 3-4 distinct hues."""
    distinct = random.randint(_DISTINCT_COLORS_MIN, min(_DISTINCT_COLORS_MAX, distractor_count))
    palette = random.sample(_DISTRACTOR_COLORS, distinct)
    bag = palette + [random.choice(palette) for _ in range(distractor_count - distinct)]
    random.shuffle(bag)
    return bag


def _gen_attempt() -> np.ndarray | None:
    # --- Grid skeleton: orange canvas + black separators + grey center ---
    top_h = random.randint(_BAND_SIZE_MIN, _BAND_SIZE_MAX)
    bot_h = random.randint(_BAND_SIZE_MIN, _BAND_SIZE_MAX)
    left_w = random.randint(_BAND_SIZE_MIN, _BAND_SIZE_MAX)
    right_w = random.randint(_BAND_SIZE_MIN, _BAND_SIZE_MAX)

    row_spans = _band_spans(top_h, bot_h)
    col_spans = _band_spans(left_w, right_w)
    height = top_h + bot_h + _CENTER_BAND + 2   # +2 for separators
    width = left_w + right_w + _CENTER_BAND + 2

    grid = np.full((height, width), Color.ORANGE, dtype=int)
    grid[top_h, :] = Color.BLACK
    grid[top_h + _CENTER_BAND + 1, :] = Color.BLACK
    grid[:, left_w] = Color.BLACK
    grid[:, left_w + _CENTER_BAND + 1] = Color.BLACK

    cr0, _ = row_spans[1]
    cc0, _ = col_spans[1]
    grid[cr0:cr0 + _CENTER_BAND, cc0:cc0 + _CENTER_BAND] = Color.GREY
    grid[cr0 + 1, cc0 + 1] = Color.MAROON  # initial maroon at center(1,1)

    # --- Pink control pixel: placed in the sub-cell matching target direction ---
    target = random.choice(_TARGETS)
    pr0, pr1 = row_spans[target[0]]
    pc0, pc1 = col_spans[target[1]]
    pink_r = random.randrange(pr0, pr1)
    pink_c = random.randrange(pc0, pc1)
    grid[pink_r, pink_c] = Color.PINK

    # --- Enumerate non-center sub-cells and their pixel coordinates ---
    cell_coords: dict[tuple[int, int], list[tuple[int, int]]] = {}
    cell_keys: list[tuple[int, int]] = []
    for br in range(3):
        for bc in range(3):
            if (br, bc) == (1, 1):
                continue
            cell_keys.append((br, bc))
            cell_coords[(br, bc)] = _cell_pixels(row_spans[br], col_spans[bc])

    # --- Distractor allocation ---
    total_colored = random.randint(_TOTAL_COLORED_MIN, _TOTAL_COLORED_MAX)
    distractor_count = total_colored - 1  # 1 is the pink pixel
    occupied_count = random.randint(_OCCUPIED_CELLS_MIN, min(_OCCUPIED_CELLS_MAX, total_colored))

    caps = {
        key: min(_MAX_DISTRACTORS_PER_CELL, len(coords) - (1 if key == target else 0))
        for key, coords in cell_coords.items()
    }

    other_occupied = _choose_occupied(cell_keys, target, caps, occupied_count, distractor_count)
    if other_occupied is None:
        return None

    occupied = [target] + other_occupied
    counts = _allocate_counts(occupied, target, caps, distractor_count)
    if counts is None:
        return None

    # --- Place distractor pixels ---
    color_bag = _make_color_bag(distractor_count)
    for key in occupied:
        n = counts[key]
        if n == 0:
            continue
        coords = [rc for rc in cell_coords[key] if rc != (pink_r, pink_c)]
        for r, c in random.sample(coords, n):
            grid[r, c] = color_bag.pop()

    # --- Oracle check ---
    expected = grid.copy()
    expected[cr0 + 1, cc0 + 1] = Color.GREY
    expected[cr0 + target[0], cc0 + target[1]] = Color.MAROON
    expected[pink_r, pink_c] = Color.MAROON

    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(_RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 18286ef8 input")
