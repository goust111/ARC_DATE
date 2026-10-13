import numpy as np

from arc_utils import find_components, get_bg


def _is_straight_bar(coords: np.ndarray) -> bool:
    rows = coords[:, 0]
    cols = coords[:, 1]

    if np.unique(rows).size == 1:
        ordered = np.sort(cols)
        return bool(np.array_equal(ordered, np.arange(ordered[0], ordered[-1] + 1)))

    if np.unique(cols).size == 1:
        ordered = np.sort(rows)
        return bool(np.array_equal(ordered, np.arange(ordered[0], ordered[-1] + 1)))

    return False


def main(grid: np.ndarray) -> np.ndarray:
    """
    Build a compact square of concentric layers, coloring layers by foreground segment length from longest outside to shortest inside.

    Input structure
    - The background is the dominant color.
    - Each foreground object is a monochrome 4-connected straight horizontal or vertical segment.
    - Segment lengths are distinct, and their descending order gives the outside-to-inside layer order.

    Rule
    - Let N be the length of the longest segment, and create an N x N output grid.
    - Sort the segments by decreasing length.
    - Fill the outermost square layer with the longest segment's color, then each next inner layer with the next segment's color.

    Key insight
    - The input positions and orientations of the segments are irrelevant; their lengths are an ordered legend for the concentric output layers.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    components = find_components(g, bg=bg, connectivity=4, monochromatic=True)
    assert components

    bars: list[tuple[int, int]] = []
    for comp in components:
        coords = comp["coords"]
        assert _is_straight_bar(coords)
        bars.append((int(comp["area"]), int(comp["color"])))

    lengths = [length for length, _ in bars]
    assert len(lengths) == len(set(lengths))

    side = max(lengths)
    assert len(bars) == (side + 1) // 2

    layers = sorted(bars, key=lambda bar: bar[0], reverse=True)
    out = np.empty((side, side), dtype=int)
    for layer, (_, color) in enumerate(layers):
        out[layer : side - layer, layer : side - layer] = color

    return out

# --------------- generator (o5x) ---------------
from arc_utils import paste, random_free_location

# Output, segment, canvas, and placement sampling.
SIDE_RANGE, COLOR_RANGE, SEGMENT_LENGTH_MIN = (5, 9), (1, 10), 1
GRID_HEIGHT_RANGE, GRID_WIDTH_RANGE = (20, 30), (20, 30)
BACKGROUND_COLOR, PLACEMENT_PADDING, PLACEMENT_PADDING_CONNECTIVITY = 0, 2, 8
MAX_GENERATION_ATTEMPTS = 300  # Retry budget for rejected placements or oracle mismatches.

def _gen_layered_output(side: int, bars: list[tuple[int, int]]) -> np.ndarray:
    out = np.empty((side, side), dtype=int)
    for layer, (_, color) in enumerate(sorted(bars, key=lambda bar: bar[0], reverse=True)):
        out[layer : side - layer, layer : side - layer] = int(color)
    return out


def _gen_bars(side: int) -> list[tuple[int, int]]:
    """Sample permuted (length, color) bars with the original distinctness rules."""
    num_segments = (side + 1) // 2
    colors = np.random.choice(np.arange(*COLOR_RANGE), size=num_segments, replace=False).astype(int).tolist()
    shorter_lengths = np.random.choice(
        np.arange(SEGMENT_LENGTH_MIN, side), size=num_segments - 1, replace=False
    ).astype(int).tolist()
    lengths = [side, *shorter_lengths]
    perm = np.random.permutation(num_segments)
    return [(int(lengths[i]), int(colors[i])) for i in perm]


def _gen_orientations(lengths: list[int]) -> list[str]:
    orientations = ["H" if np.random.randint(2) == 0 else "V" for _ in lengths]
    visible = [i for i, length in enumerate(lengths) if length > 1]
    if len(visible) >= 2:
        i, j = np.random.choice(visible, size=2, replace=False)
        orientations[int(i)] = "H"
        orientations[int(j)] = "V"
    return orientations


def _gen_bar_patch(length: int, color: int, orientation: str) -> np.ndarray:
    if orientation == "H":
        return np.full((1, int(length)), int(color), dtype=int)
    return np.full((int(length), 1), int(color), dtype=int)


def _gen_attempt() -> np.ndarray | None:
    side = int(np.random.randint(*SIDE_RANGE))
    bars = _gen_bars(side)
    expected = _gen_layered_output(side, bars)

    height = int(np.random.randint(*GRID_HEIGHT_RANGE))
    width = int(np.random.randint(*GRID_WIDTH_RANGE))
    grid = np.full((height, width), BACKGROUND_COLOR, dtype=int)
    orientations = _gen_orientations([length for length, _ in bars])

    place_order = sorted(range(len(bars)), key=lambda i: bars[i][0], reverse=True)
    for i in place_order:
        length, color = bars[i]
        patch = _gen_bar_patch(length, color, orientations[i])
        try:
            r0, c0 = random_free_location(
                grid, patch, transparent=BACKGROUND_COLOR, padding=PLACEMENT_PADDING,
                padding_connectivity=PLACEMENT_PADDING_CONNECTIVITY,
            )
        except ValueError:
            return None
        paste(grid, patch, r0, c0, transparent=BACKGROUND_COLOR)

    try:
        actual = main(grid)
    except Exception:
        return None
    if np.array_equal(actual, expected):
        return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input failed for 3ee1011a after {MAX_GENERATION_ATTEMPTS} attempts")
