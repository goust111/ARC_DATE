import numpy as np

from arc_utils import colors_present, components_mask, get_bg
import random


def _role_colors(g: np.ndarray, bg: int) -> tuple[int, int]:
    colors = colors_present(g, exclude={bg})
    assert len(colors) == 2, "expected exactly two non-background colors"

    boundary = set(map(int, np.concatenate((g[0, :], g[-1, :], g[:, 0], g[:, -1]))))
    guide_colors = [color for color in colors if color in boundary]
    object_colors = [color for color in colors if color not in boundary]
    assert len(guide_colors) == 1, "expected one non-background guide color touching the boundary"
    assert len(object_colors) == 1, "expected one interior object color"
    return int(guide_colors[0]), int(object_colors[0])


def _midpoint(pair: list[int], axis_name: str) -> int:
    assert len(pair) == 2, f"expected exactly two full {axis_name} guide bars"
    a, b = pair
    assert (a + b) % 2 == 0, f"expected full {axis_name} guide bars to have a middle cell"
    return (a + b) // 2


def _unique_max_index(counts: np.ndarray, axis_name: str) -> int:
    max_count = int(counts.max())
    indices = np.flatnonzero(counts == max_count)
    assert len(indices) == 1, f"expected a unique perpendicular {axis_name} guide bar"
    return int(indices[0])


def _centered_span(center: int, length: int, limit: int) -> tuple[int, int]:
    assert length % 2 == 1, "expected the interior object area to be odd"
    start = center - length // 2
    stop = center + length // 2
    assert 0 <= start <= stop < limit, "flattened object line must fit inside the grid"
    return start, stop


def main(grid: np.ndarray) -> np.ndarray:
    """
    Flatten the interior object into a centered line and move the guide bars inward to bracket it.

    Input structure:
    - The dominant color is the background.
    - One non-background guide color touches the boundary and forms exactly two full parallel bars plus a split perpendicular center bar.
    - The other non-background color is a single 8-connected interior object whose area determines the flattened line length.

    Rule:
    - If the guide pair is vertical, make a vertical object-colored line centered between those bars; otherwise use the analogous horizontal line.
    - The line length equals the number of interior object pixels and is centered on the split perpendicular guide bar.
    - Rebuild the guide color as the perpendicular center bar plus two full bars immediately beside the flattened object line.

    Key insight:
    - The guide bars supply only the orientation and center; the interior object's pixel count supplies the length of the final one-cell-thick line.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    height, width = map(int, g.shape)

    bg = get_bg(g)
    guide_color, object_color = _role_colors(g, bg)
    object_mask = g == object_color
    object_components = components_mask(object_mask, connectivity=8)
    assert len(object_components) == 1, "expected one 8-connected interior object"
    object_area = int(object_components[0].shape[0])
    assert object_area > 0, "expected a nonempty interior object"

    guide_mask = g == guide_color
    row_counts = np.count_nonzero(guide_mask, axis=1)
    col_counts = np.count_nonzero(guide_mask, axis=0)
    full_rows = np.flatnonzero(row_counts == width).tolist()
    full_cols = np.flatnonzero(col_counts == height).tolist()
    has_horizontal_pair = len(full_rows) == 2
    has_vertical_pair = len(full_cols) == 2
    assert has_horizontal_pair ^ has_vertical_pair, "expected exactly one pair of full parallel guide bars"

    out = np.full_like(g, int(bg))
    if has_vertical_pair:
        center_col = _midpoint(full_cols, "vertical")
        center_row = _unique_max_index(row_counts, "row")
        start_row, stop_row = _centered_span(center_row, object_area, height)
        assert 0 < center_col < width - 1, "vertical flattened line must have guide bars on both sides"

        out[:, center_col - 1] = guide_color
        out[:, center_col + 1] = guide_color
        out[center_row, :] = guide_color
        out[start_row : stop_row + 1, center_col] = object_color
    else:
        center_row = _midpoint(full_rows, "horizontal")
        center_col = _unique_max_index(col_counts, "column")
        start_col, stop_col = _centered_span(center_col, object_area, width)
        assert 0 < center_row < height - 1, "horizontal flattened line must have guide bars above and below"

        out[center_row - 1, :] = guide_color
        out[center_row + 1, :] = guide_color
        out[:, center_col] = guide_color
        out[center_row, start_col : stop_col + 1] = object_color

    return out

# --------------- generator (att2) ---------------

GEN_BG = 7
GEN_GUIDE = 0
GEN_OBJECT = 5

# Canvas, object, guide gap, and retry sampling bounds.
GEN_GRID_SIZES = [11, 13, 15, 17, 19]
GEN_ORIENTATIONS = ["V", "H"]
GEN_AREA_MIN = 5
GEN_AREA_MAX = 13
GEN_AREA_STEP = 2
GEN_GAPS = (4, 6, 8)
GEN_MIN_GAP_EXTRA = 2
GEN_BOUNDARY_MARGIN = 1
GEN_MAX_ATTEMPTS = 300

GEN_NEIGHBOR_STEPS = [
    (dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if dr or dc
]


def _gen_expected_output(n: int, orientation: str, area: int) -> np.ndarray:
    center = n // 2
    out = np.full((n, n), GEN_BG, dtype=int)
    start, stop = center - area // 2, center + area // 2

    if orientation == "V":
        out[:, center - 1] = GEN_GUIDE
        out[:, center + 1] = GEN_GUIDE
        out[center, :] = GEN_GUIDE
        out[start : stop + 1, center] = GEN_OBJECT
    else:
        out[center - 1, :] = GEN_GUIDE
        out[center + 1, :] = GEN_GUIDE
        out[:, center] = GEN_GUIDE
        out[center, start : stop + 1] = GEN_OBJECT

    return out


def _gen_sample_layout() -> tuple[int, int, str, int]:
    n = random.choice(GEN_GRID_SIZES)
    center = n // 2
    orientation = random.choice(GEN_ORIENTATIONS)
    area = random.choice(range(GEN_AREA_MIN, min(GEN_AREA_MAX, n) + 1, GEN_AREA_STEP))
    return n, center, orientation, area


def _gen_gap_options(n: int, area: int) -> list[int]:
    min_gap = int(np.ceil(np.sqrt(area))) + GEN_MIN_GAP_EXTRA
    return [g for g in GEN_GAPS if g <= n - 4 and g >= min_gap]


def _gen_draw_input_guides(
    grid: np.ndarray, orientation: str, center: int, lo: int, hi: int
) -> tuple[int, int, int, int]:
    if orientation == "V":
        grid[:, lo] = GEN_GUIDE
        grid[:, hi] = GEN_GUIDE
        grid[center, : lo + 1] = GEN_GUIDE
        grid[center, hi:] = GEN_GUIDE
        return GEN_BOUNDARY_MARGIN, grid.shape[0] - 1 - GEN_BOUNDARY_MARGIN, lo + 1, hi - 1
    else:
        grid[lo, :] = GEN_GUIDE
        grid[hi, :] = GEN_GUIDE
        grid[: lo + 1, center] = GEN_GUIDE
        grid[hi:, center] = GEN_GUIDE
        return lo + 1, hi - 1, GEN_BOUNDARY_MARGIN, grid.shape[1] - 1 - GEN_BOUNDARY_MARGIN


def _gen_blob_coords(
    area: int,
    row_min: int,
    row_max: int,
    col_min: int,
    col_max: int,
    seed_r: int,
    seed_c: int,
) -> set[tuple[int, int]]:
    start = (
        min(max(seed_r, row_min), row_max),
        min(max(seed_c, col_min), col_max),
    )
    coords = {start}
    frontier = {
        (start[0] + dr, start[1] + dc)
        for dr, dc in GEN_NEIGHBOR_STEPS
        if row_min <= start[0] + dr <= row_max
        and col_min <= start[1] + dc <= col_max
    }

    while len(coords) < area:
        options = tuple(frontier)
        weights = []
        for r, c in options:
            neighbors = sum(
                (r + dr, c + dc) in coords
                for dr, dc in GEN_NEIGHBOR_STEPS
            )
            dist = abs(r - seed_r) + abs(c - seed_c)
            weights.append((neighbors + 1) ** 2 / (dist + 1))
        cell = random.choices(options, weights=weights, k=1)[0]
        frontier.remove(cell)
        coords.add(cell)
        r, c = cell
        for dr, dc in GEN_NEIGHBOR_STEPS:
            nxt = (r + dr, c + dc)
            if (
                row_min <= nxt[0] <= row_max
                and col_min <= nxt[1] <= col_max
                and nxt not in coords
            ):
                frontier.add(nxt)

    return coords


def _gen_attempt() -> np.ndarray | None:
    n, center, orientation, area = _gen_sample_layout()
    expected = _gen_expected_output(n, orientation, area)

    gaps = _gen_gap_options(n, area)
    if not gaps:
        return None
    gap = random.choice(gaps)
    lo = center - gap // 2
    hi = center + gap // 2

    grid = np.full((n, n), GEN_BG, dtype=int)
    row_min, row_max, col_min, col_max = _gen_draw_input_guides(
        grid, orientation, center, lo, hi
    )
    coords = _gen_blob_coords(area, row_min, row_max, col_min, col_max, center, center)
    rows, cols = zip(*coords)
    grid[rows, cols] = GEN_OBJECT

    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid c61be7dc input")
