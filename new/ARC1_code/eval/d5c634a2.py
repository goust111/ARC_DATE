import numpy as np

from arc_utils import Color, find_components, get_bg, paste


UP_T = frozenset({(0, 1), (1, 0), (1, 1), (1, 2)})
DOWN_T = frozenset({(0, 0), (0, 1), (0, 2), (1, 1)})
TALLY_SLOTS = ((0, 0), (2, 0), (0, 2), (2, 2))


def _normalized_shape(coords: np.ndarray) -> frozenset[tuple[int, int]]:
    r0, c0 = coords.min(axis=0)
    return frozenset((int(r - r0), int(c - c0)) for r, c in coords)


def main(grid: np.ndarray) -> np.ndarray:
    """Count the upward and downward red T tetrominoes into fixed green and blue tally panels.

    Input structure
    - A black background containing only red 4-connected T tetromino objects.
    - Each object is a 2x3 T, either with its stem above the horizontal bar or below it.

    Rule
    - Make a 3x6 black output split into two 3x3 panels.
    - For each upward T, fill the next corner slot in the left panel with green.
    - For each downward T, fill the next corner slot in the right panel with blue.
    - The slot order inside each panel is top-left, bottom-left, top-right, bottom-right.

    Key insight
    - The red objects' positions are irrelevant; only the count of each T orientation is summarized.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    assert bg == Color.BLACK
    assert set(map(int, np.unique(g))) <= {Color.BLACK, Color.RED}

    up_count = 0
    down_count = 0
    objects = find_components(g, bg=bg, connectivity=4, monochromatic=True, colors=[Color.RED])
    assert objects

    for obj in objects:
        assert obj["area"] == 4
        shape = _normalized_shape(obj["coords"])
        if shape == UP_T:
            up_count += 1
        elif shape == DOWN_T:
            down_count += 1
        else:
            raise AssertionError(f"unexpected red object shape: {sorted(shape)}")

    assert up_count <= len(TALLY_SLOTS)
    assert down_count <= len(TALLY_SLOTS)

    out = np.full((3, 6), Color.BLACK, dtype=int)
    for r, c in TALLY_SLOTS[:up_count]:
        out[r, c] = Color.GREEN
    for r, c in TALLY_SLOTS[:down_count]:
        out[r, c + 3] = Color.BLUE
    return out

# --------------- generator (o5x) ---------------

_gen_rng = np.random.default_rng()

# Orientation count bounds; NumPy uses an exclusive high endpoint.
COUNT_RANGE = (1, 5)

# Canvas size bounds and area heuristic.
ROW_CHOICES = range(5, 16)
COL_CHOICES = range(4, 16)
MIN_CANVAS_AREA = 20
AREA_PER_T = 8
AREA_EXTRA = 1

# Tetromino rendering.
UP_KIND = "up"
DOWN_KIND = "down"
PATCH_SHAPE = (2, 3)
T_COORDS = {UP_KIND: UP_T, DOWN_KIND: DOWN_T}

# Output tally panel geometry and retry limit.
OUTPUT_SHAPE = (3, 6)
RIGHT_PANEL_COL_OFFSET = 3
MAX_GENERATION_ATTEMPTS = 300


def _gen_make_t_patch(kind: str) -> np.ndarray:
    patch = np.full(PATCH_SHAPE, Color.BLACK, dtype=int)
    for r, c in T_COORDS[kind]:
        patch[r, c] = Color.RED
    return patch


def _gen_4_connected_forbidden(canvas: np.ndarray) -> np.ndarray:
    """Cells where a new red pixel would overlap or share an edge."""
    occupied = canvas != Color.BLACK
    forbidden = occupied.copy()
    forbidden[1:, :] |= occupied[:-1, :]
    forbidden[:-1, :] |= occupied[1:, :]
    forbidden[:, 1:] |= occupied[:, :-1]
    forbidden[:, :-1] |= occupied[:, 1:]
    return forbidden


def _gen_valid_locations(canvas: np.ndarray, patch: np.ndarray) -> list[tuple[int, int]]:
    """Return all top-left placements that keep red T objects 4-separated."""
    rows, cols = canvas.shape
    pr, pc = np.nonzero(patch != Color.BLACK)
    ph, pw = patch.shape
    forbidden = _gen_4_connected_forbidden(canvas)
    locations = []

    for r0 in range(rows - ph + 1):
        for c0 in range(cols - pw + 1):
            rr = pr + r0
            cc = pc + c0
            if not np.any(forbidden[rr, cc]):
                locations.append((r0, c0))
    return locations


def _gen_choose_size(total_count: int) -> tuple[int, int]:
    min_area = max(MIN_CANVAS_AREA, AREA_PER_T * total_count + AREA_EXTRA)
    choices = [
        (r, c)
        for r in ROW_CHOICES
        for c in COL_CHOICES
        if r * c >= min_area
    ]
    return choices[int(_gen_rng.integers(len(choices)))]


def _gen_expected_output(up_count: int, down_count: int) -> np.ndarray:
    out = np.full(OUTPUT_SHAPE, Color.BLACK, dtype=int)
    for r, c in TALLY_SLOTS[:up_count]:
        out[r, c] = Color.GREEN
    for r, c in TALLY_SLOTS[:down_count]:
        out[r, c + RIGHT_PANEL_COL_OFFSET] = Color.BLUE
    return out


def _gen_solver_accepts(grid: np.ndarray, up_count: int, down_count: int) -> bool:
    """Run the solver oracle and convert assertion failures into rejection."""
    expected = _gen_expected_output(up_count, down_count)
    try:
        actual = main(grid)
    except Exception:
        return False
    return np.array_equal(actual, expected)


def _gen_attempt() -> np.ndarray | None:
    up_count = int(_gen_rng.integers(*COUNT_RANGE))
    down_count = int(_gen_rng.integers(*COUNT_RANGE))
    rows, cols = _gen_choose_size(up_count + down_count)
    grid = np.full((rows, cols), Color.BLACK, dtype=int)

    kinds = [UP_KIND] * up_count + [DOWN_KIND] * down_count
    _gen_rng.shuffle(kinds)
    for kind in kinds:
        patch = _gen_make_t_patch(kind)
        locations = _gen_valid_locations(grid, patch)
        if not locations:
            return None
        r0, c0 = locations[int(_gen_rng.integers(len(locations)))]
        paste(grid, patch, r0, c0, transparent=Color.BLACK)

    if not _gen_solver_accepts(grid, up_count, down_count):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid d5c634a2 input")
