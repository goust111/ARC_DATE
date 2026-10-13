import numpy as np

from arc_utils import Color, colors_present, get_bg
import random


def _paint(assignments: dict[tuple[int, int], int], coord: tuple[int, int], color: int) -> None:
    if coord in assignments:
        assert assignments[coord] == color
    assignments[coord] = color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand each green control pixel into a fixed gantry shape with a full-width baseline beneath it.

    Input structure
    - The grid has a black background and one or more green control pixels.
    - Each control pixel has at least two cells of margin above, below, left, and right.

    Rule
    - Around each green pixel, draw a 5-cell grey cap two rows above it, red side posts at offsets +/-2 for rows -1, 0, +1, and a grey pixel directly above the marker.
    - On the row two cells below the marker, fill the entire row red, then recolor the 5 cells centered below the marker teal.
    - Keep the original green control pixels.

    Key insight
    - The marker is the center of a fixed 5-column scaffold; only its baseline extends horizontally across the whole grid.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    assert bg == Color.BLACK
    assert set(colors_present(g)) <= {Color.BLACK, Color.GREEN}

    markers = np.argwhere(g == Color.GREEN)
    assert len(markers) > 0

    height, width = g.shape
    assignments: dict[tuple[int, int], int] = {}
    base_centers_by_row: dict[int, list[int]] = {}

    for r_raw, c_raw in markers:
        r = int(r_raw)
        c = int(c_raw)
        assert 2 <= r and r + 2 < height
        assert 2 <= c and c + 2 < width

        for cc in range(c - 2, c + 3):
            _paint(assignments, (r - 2, cc), Color.GREY)

        _paint(assignments, (r - 1, c - 2), Color.RED)
        _paint(assignments, (r - 1, c), Color.GREY)
        _paint(assignments, (r - 1, c + 2), Color.RED)
        _paint(assignments, (r, c - 2), Color.RED)
        _paint(assignments, (r, c + 2), Color.RED)
        _paint(assignments, (r + 1, c - 2), Color.RED)
        _paint(assignments, (r + 1, c + 2), Color.RED)
        base_centers_by_row.setdefault(r + 2, []).append(c)

    for br, centers in base_centers_by_row.items():
        for cc in range(width):
            color = Color.TEAL if any(c - 2 <= cc <= c + 2 for c in centers) else Color.RED
            _paint(assignments, (br, cc), color)

    marker_cells = {tuple(map(int, coord)) for coord in markers}
    assert not (marker_cells & assignments.keys())

    out = g.copy()
    for (r, c), color in assignments.items():
        out[r, c] = color
    return out

# --------------- generator (o5x) ---------------

# Canvas and marker-count sampling.
GEN_MARKER_COUNT_RANGE = (1, 3)
GEN_HEIGHT_BY_MARKER_COUNT = {1: (10, 17), 2: (12, 17), 3: 17}
GEN_WIDTH_RANGE = (10, 19)
GEN_MARKER_MARGIN = 3
GEN_MIN_ROW_SEPARATION = 5

# Scaffold geometry used by the oracle output check.
GEN_SCAFFOLD_RADIUS = 2
GEN_CAP_ROW_OFFSET = -2
GEN_STEM_ROW_OFFSET = -1
GEN_POST_OFFSETS = ((-1, -2), (-1, 2), (0, -2), (0, 2), (1, -2), (1, 2))
GEN_BASELINE_ROW_OFFSET = 2
GEN_MAX_ATTEMPTS = 300

def _gen_sample_height(marker_count: int) -> int:
    """Sample height, preserving the fixed-height branch for 3 markers."""
    height_range = GEN_HEIGHT_BY_MARKER_COUNT[marker_count]
    return height_range if isinstance(height_range, int) else random.randint(*height_range)

def _gen_markers_compatible(a: tuple[int, int], b: tuple[int, int]) -> bool:
    """Return whether two marker centers satisfy the generator spacing rule."""
    return abs(a[0] - b[0]) >= GEN_MIN_ROW_SEPARATION

def _gen_choose_marker_set(rows: list[int], cols: list[int], k: int) -> tuple[tuple[int, int], ...] | None:
    """Uniformly sample one compatible k-marker set from the candidate axes."""
    positions = [(r, c) for r in rows for c in cols]
    if k == 1:
        return (random.choice(positions),)

    chosen = None
    seen = 0
    n = len(positions)
    for i in range(n):
        a = positions[i]
        for j in range(i + 1, n):
            b = positions[j]
            if not _gen_markers_compatible(a, b):
                continue
            if k == 2:
                seen += 1
                if random.randrange(seen) == 0:
                    chosen = (a, b)
                continue
            for m in range(j + 1, n):
                c = positions[m]
                if _gen_markers_compatible(a, c) and _gen_markers_compatible(b, c):
                    seen += 1
                    if random.randrange(seen) == 0:
                        chosen = (a, b, c)
    return chosen

def _gen_render_input(height: int, width: int, markers: tuple[tuple[int, int], ...]) -> np.ndarray:
    """Render the black input grid with the sampled green marker centers."""
    grid = np.zeros((height, width), dtype=int)
    for r, c in markers:
        grid[r, c] = Color.GREEN
    return grid

def _gen_expected_output(height: int, width: int, markers: tuple[tuple[int, int], ...]) -> np.ndarray:
    """Render the exact gantry output expected from the solver."""
    out = _gen_render_input(height, width, markers)
    by_baseline: dict[int, list[int]] = {}
    for r, c in markers:
        out[r + GEN_CAP_ROW_OFFSET, c - GEN_SCAFFOLD_RADIUS:c + GEN_SCAFFOLD_RADIUS + 1] = Color.GREY
        out[r + GEN_STEM_ROW_OFFSET, c] = Color.GREY
        for dr, dc in GEN_POST_OFFSETS:
            out[r + dr, c + dc] = Color.RED
        by_baseline.setdefault(r + GEN_BASELINE_ROW_OFFSET, []).append(c)

    for br, centers in by_baseline.items():
        out[br, :] = Color.RED
        for c in centers:
            out[br, c - GEN_SCAFFOLD_RADIUS:c + GEN_SCAFFOLD_RADIUS + 1] = Color.TEAL

    for r, c in markers:
        out[r, c] = Color.GREEN
    return out

def _gen_oracle_accepts(grid: np.ndarray, expected: np.ndarray) -> bool:
    """Return whether the solver maps this input to the expected output."""
    try:
        return np.array_equal(main(grid), expected)
    except Exception:
        return False

def _gen_attempt() -> np.ndarray | None:
    """Sample one candidate input and reject it if the oracle check fails."""
    marker_count = random.randint(*GEN_MARKER_COUNT_RANGE)
    height = _gen_sample_height(marker_count)
    width = random.randint(*GEN_WIDTH_RANGE)

    rows = list(range(GEN_MARKER_MARGIN, height - GEN_MARKER_MARGIN))
    cols = list(range(GEN_MARKER_MARGIN, width - GEN_MARKER_MARGIN))
    markers = _gen_choose_marker_set(rows, cols, marker_count)
    if markers is None:
        return None

    grid = _gen_render_input(height, width, markers)
    expected = _gen_expected_output(height, width, markers)
    if not _gen_oracle_accepts(grid, expected):
        return None
    return grid

def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError('failed to generate a valid 3f23242b input')
