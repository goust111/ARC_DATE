import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
MAGENTA = 6
ORANGE = 7
AZURE = 8
MAROON = 9

WARM_START_COLORS = {RED, ORANGE, YELLOW, MAROON}
COLD_START_COLORS = {BLUE, GREEN, AZURE}
NEUTRAL_START_COLORS = {GREY, MAGENTA}
NON_BLACK_COLORS = tuple(range(1, 10))

GENERATION_RETRIES = 300
PLACEMENT_RETRIES = 160
MIN_STRIPE_LENGTH = 4


def _components4(mask: np.ndarray) -> list[np.ndarray]:
    visited = np.zeros_like(mask, dtype=bool)
    components: list[np.ndarray] = []
    height, width = mask.shape

    for r in range(height):
        for c in range(width):
            if not mask[r, c] or visited[r, c]:
                continue
            stack = [(r, c)]
            visited[r, c] = True
            cells: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < height and 0 <= nc < width and mask[nr, nc] and not visited[nr, nc]:
                        visited[nr, nc] = True
                        stack.append((nr, nc))
            components.append(np.array(cells, dtype=int))
    return components


def _decode_line(line: np.ndarray) -> np.ndarray:
    values = np.asarray(line, dtype=int)
    assert values.ndim == 1 and values.size >= 2, "stripe must be a 1D alternating run of length at least 2"

    start_color = int(values[0])
    second_color = int(values[1])
    assert start_color != BLACK and second_color != BLACK, "stripe colors must be non-black"
    assert start_color != second_color, "stripe must alternate between two distinct colors"
    for idx, value in enumerate(values):
        expected = start_color if idx % 2 == 0 else second_color
        assert int(value) == expected, "stripe pixels must alternate strictly"

    if start_color in WARM_START_COLORS:
        return np.full(values.shape, start_color, dtype=int)
    if start_color in COLD_START_COLORS:
        return np.full(values.shape, second_color, dtype=int)
    assert start_color in NEUTRAL_START_COLORS, "start color must be warm, cold, or neutral"
    return values.copy()


def _component_line(grid: np.ndarray, component: np.ndarray) -> tuple[str, int, int, np.ndarray]:
    rows = component[:, 0]
    cols = component[:, 1]

    unique_rows = np.unique(rows)
    unique_cols = np.unique(cols)
    if unique_rows.size == 1:
        row = int(unique_rows[0])
        left = int(cols.min())
        right = int(cols.max())
        assert component.shape[0] == right - left + 1, "horizontal stripe must be gap-free"
        line = np.asarray(grid[row, left : right + 1], dtype=int)
        assert np.all(line != BLACK), "horizontal stripe cannot include background gaps"
        return "h", row, left, line

    if unique_cols.size == 1:
        col = int(unique_cols[0])
        top = int(rows.min())
        bottom = int(rows.max())
        assert component.shape[0] == bottom - top + 1, "vertical stripe must be gap-free"
        line = np.asarray(grid[top : bottom + 1, col], dtype=int)
        assert np.all(line != BLACK), "vertical stripe cannot include background gaps"
        return "v", top, col, line

    raise AssertionError("each connected component must be a straight 1-pixel stripe")


def main(grid: np.ndarray) -> np.ndarray:
    """Decode alternating stripe bands into monochrome or preserved bands based on the start color."""
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    mask = g != BLACK
    components = _components4(mask)
    assert components, "expected at least one stripe band"

    out = np.full(g.shape, BLACK, dtype=int)
    for component in components:
        orientation, anchor_r, anchor_c, line = _component_line(g, component)
        decoded = _decode_line(line)
        if orientation == "h":
            out[anchor_r, anchor_c : anchor_c + line.size] = decoded
        else:
            out[anchor_r : anchor_r + line.size, anchor_c] = decoded
    return out


def _alternating_line(length: int, start_color: int, second_color: int) -> np.ndarray:
    line = np.empty(length, dtype=int)
    line[0::2] = start_color
    line[1::2] = second_color
    return line


def _sample_start_color(branch: str, rng: np.random.Generator) -> int:
    if branch == "warm":
        return int(rng.choice(tuple(WARM_START_COLORS)))
    if branch == "cold":
        return int(rng.choice(tuple(COLD_START_COLORS)))
    assert branch == "neutral"
    return int(rng.choice(tuple(NEUTRAL_START_COLORS)))


def _sample_specs(rng: np.random.Generator) -> list[tuple[str, str, int, int, int]]:
    copies = int(rng.integers(1, 3))
    branches = ["warm", "cold", "neutral"] * copies
    rng.shuffle(branches)

    if len(branches) == 3:
        orientations = ["h", "v", str(rng.choice(np.array(["h", "v"])))]
    else:
        orientations = ["h", "h", "h", "v", "v", "v"]
    rng.shuffle(orientations)

    specs: list[tuple[str, str, int, int, int]] = []
    for branch, orientation in zip(branches, orientations):
        start_color = _sample_start_color(branch, rng)
        second_choices = [color for color in NON_BLACK_COLORS if color != start_color]
        second_color = int(rng.choice(second_choices))
        specs.append((branch, orientation, start_color, second_color, copies))
    return specs


def _mark_blocked(blocked: np.ndarray, cells: np.ndarray) -> None:
    height, width = blocked.shape
    for r_raw, c_raw in cells:
        r = int(r_raw)
        c = int(c_raw)
        for rr in range(max(0, r - 1), min(height, r + 2)):
            for cc in range(max(0, c - 1), min(width, c + 2)):
                blocked[rr, cc] = True


def _candidate_cells(orientation: str, anchor_a: int, anchor_b: int, length: int) -> np.ndarray:
    if orientation == "h":
        return np.column_stack((np.full(length, anchor_a, dtype=int), np.arange(anchor_b, anchor_b + length)))
    return np.column_stack((np.arange(anchor_a, anchor_a + length), np.full(length, anchor_b, dtype=int)))


def _place_stripe(
    grid: np.ndarray,
    expected: np.ndarray,
    blocked: np.ndarray,
    orientation: str,
    start_color: int,
    second_color: int,
    length: int,
    rng: np.random.Generator,
) -> bool:
    height, width = grid.shape
    if orientation == "h":
        if length > width:
            return False
        for _ in range(PLACEMENT_RETRIES):
            row = int(rng.integers(0, height))
            left = int(rng.integers(0, width - length + 1))
            cells = _candidate_cells("h", row, left, length)
            if np.any(blocked[cells[:, 0], cells[:, 1]]):
                continue
            line = _alternating_line(length, start_color, second_color)
            grid[row, left : left + length] = line
            expected[row, left : left + length] = _decode_line(line)
            _mark_blocked(blocked, cells)
            return True
        return False

    if length > height:
        return False
    for _ in range(PLACEMENT_RETRIES):
        top = int(rng.integers(0, height - length + 1))
        col = int(rng.integers(0, width))
        cells = _candidate_cells("v", top, col, length)
        if np.any(blocked[cells[:, 0], cells[:, 1]]):
            continue
        line = _alternating_line(length, start_color, second_color)
        grid[top : top + length, col] = line
        expected[top : top + length, col] = _decode_line(line)
        _mark_blocked(blocked, cells)
        return True
    return False


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()
    specs = _sample_specs(rng)
    copies = specs[0][4]

    if copies == 1:
        height = int(rng.integers(10, 17))
        width = int(rng.integers(10, 17))
        max_length_cap = 8
    else:
        height = int(rng.integers(14, 19))
        width = int(rng.integers(14, 19))
        max_length_cap = 7

    grid = np.full((height, width), BLACK, dtype=int)
    expected = np.full_like(grid, BLACK)
    blocked = np.zeros_like(grid, dtype=bool)

    for _branch, orientation, start_color, second_color, _ in specs:
        max_length = min(max_length_cap, width if orientation == "h" else height)
        if max_length < MIN_STRIPE_LENGTH:
            return None
        length = int(rng.integers(MIN_STRIPE_LENGTH, max_length + 1))
        if not _place_stripe(grid, expected, blocked, orientation, start_color, second_color, length, rng):
            return None

    if not np.any(grid != BLACK):
        return None
    if np.array_equal(grid, expected):
        return None

    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid stripe-decoding input")
