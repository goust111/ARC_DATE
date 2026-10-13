import numpy as np

from arc_utils import components_mask, find_components, get_bg, paste, random_free_location
import random


def _has_enclosed_bg_hole(region: np.ndarray, color: int, bg: int) -> bool:
    assert np.all((region == color) | (region == bg)), "object bbox must contain only object and background colors"

    border = np.zeros(region.shape, dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True
    assert np.all(region[border] == color), "object bbox border must be complete"

    for coords in components_mask(region == bg, connectivity=4):
        rows = coords[:, 0]
        cols = coords[:, 1]
        touches_border = (
            np.any(rows == 0)
            or np.any(rows == region.shape[0] - 1)
            or np.any(cols == 0)
            or np.any(cols == region.shape[1] - 1)
        )
        if not touches_border:
            return True
    return False


def main(grid: np.ndarray) -> np.ndarray:
    """Return the color of the unique rectangular foreground object that encloses a background hole.

    Input structure
    - The input has a dominant background and several separated single-color rectangular foreground objects.
    - Exactly one foreground object is a complete rectangular frame around one or more background pixels.

    Rule
    - Find all monochrome 4-connected foreground objects.
    - For each object's bbox, require a complete border in that object's color.
    - Select the unique object whose bbox contains a 4-connected background component that does not touch the bbox border.
    - Output a 1x1 grid containing the selected object's color.

    Key insight
    - The answer is not determined by object size, position, or color identity; it is the only object with an enclosed background hole.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = get_bg(g)
    components = find_components(g, bg=bg, connectivity=4, monochromatic=True)
    assert components, "expected foreground objects"

    selected_colors = []
    for component in components:
        color = int(component["color"])
        r0, c0, r1, c1 = component["bbox"]
        region = g[r0 : r1 + 1, c0 : c1 + 1]
        if _has_enclosed_bg_hole(region, color, bg):
            selected_colors.append(color)

    assert len(selected_colors) == 1, "expected exactly one hole-containing object"
    return np.array([[selected_colors[0]]], dtype=int)

# --------------- generator (o5x) ---------------

# Canvas, colors, and object counts.
BACKGROUND = 0
CANVAS_ROWS, CANVAS_COLS = (14, 16), (13, 17)
OBJECT_COUNT_RANGE = (4, 7)
COLOR_POOL = range(1, 10)

# Frame and solid patch dimensions.
FRAME_HOLE_ROWS, FRAME_HOLE_COLS = (1, 2), (1, 3)
FRAME_THICKNESS_RANGE, MAX_PATCH_DIM = (1, 3), 8
SOLID_DIM_BY_COUNT = {4: (8, 8), 5: (8, 7), 6: (7, 7), 7: (6, 6)}
SOLID_DIM_MIN, ELONGATED_PROB, ELONGATED_SHORT_MAX, ELONGATED_LONG_MIN = 2, 0.25, 4, 5

# Placement filters and retry limits.
PLACEMENT_PADDING, PLACEMENT_PADDING_CONNECTIVITY = 1, 8
MAX_BORDER_TOUCHES, COVERAGE_RANGE, MAX_ATTEMPTS = 4, (0.33, 0.57), 300


def _gen_frame_patch(color: int) -> np.ndarray:
    """Sample the one rectangular frame patch with a background hole."""
    hole_h = random.randint(*FRAME_HOLE_ROWS)
    hole_w = random.randint(*FRAME_HOLE_COLS)
    lo, hi = FRAME_THICKNESS_RANGE
    vertical = [(t, b) for t in range(lo, hi + 1) for b in range(lo, hi + 1) if t + hole_h + b <= MAX_PATCH_DIM]
    horizontal = [(l, r) for l in range(lo, hi + 1) for r in range(lo, hi + 1) if l + hole_w + r <= MAX_PATCH_DIM]
    top, bottom = random.choice(vertical)
    left, right = random.choice(horizontal)

    h = top + hole_h + bottom
    w = left + hole_w + right
    patch = np.full((h, w), int(color), dtype=int)
    patch[top : top + hole_h, left : left + hole_w] = BACKGROUND
    return patch


def _gen_solid_patch(color: int, n_objects: int) -> np.ndarray:
    """Sample a solid rectangular distractor patch sized for the object count."""
    h_hi, w_hi = SOLID_DIM_BY_COUNT[n_objects]

    if random.random() < ELONGATED_PROB:
        if random.random() < 0.5:
            h = random.randint(SOLID_DIM_MIN, min(ELONGATED_SHORT_MAX, h_hi))
            w = random.randint(ELONGATED_LONG_MIN, w_hi) if w_hi >= ELONGATED_LONG_MIN else random.randint(SOLID_DIM_MIN, w_hi)
        else:
            h = random.randint(ELONGATED_LONG_MIN, h_hi) if h_hi >= ELONGATED_LONG_MIN else random.randint(SOLID_DIM_MIN, h_hi)
            w = random.randint(SOLID_DIM_MIN, min(ELONGATED_SHORT_MAX, w_hi))
    else:
        h = random.randint(SOLID_DIM_MIN, h_hi)
        w = random.randint(SOLID_DIM_MIN, w_hi)
    return np.full((h, w), int(color), dtype=int)


def _build_patches(n_objects: int) -> tuple[list[np.ndarray], int]:
    """Return sampled object patches and the color the solver should select."""
    colors = random.sample(COLOR_POOL, n_objects)
    frame_color = random.choice(colors)
    patches = []
    for color in colors:
        if color == frame_color:
            patch = _gen_frame_patch(color)
        else:
            patch = _gen_solid_patch(color, n_objects)
        patches.append(patch)
    return patches, frame_color


def _place_patches(rows: int, cols: int, patches: list[np.ndarray]) -> np.ndarray | None:
    """Place separated patches; return None when greedy placement fails filters."""
    grid = np.zeros((rows, cols), dtype=int)
    occupied = np.zeros((rows, cols), dtype=int)
    ordered = sorted(
        patches,
        key=lambda patch: (patch.shape[0] * patch.shape[1], random.random()),
        reverse=True,
    )
    touching_border = 0
    for patch in ordered:
        footprint = np.ones(patch.shape, dtype=int)
        try:
            r0, c0 = random_free_location(
                occupied,
                footprint,
                transparent=BACKGROUND,
                padding=PLACEMENT_PADDING,
                padding_connectivity=PLACEMENT_PADDING_CONNECTIVITY,
            )
        except ValueError:
            return None
        if r0 == 0 or c0 == 0 or r0 + patch.shape[0] == rows or c0 + patch.shape[1] == cols:
            touching_border += 1
        paste(occupied, footprint, r0, c0)
        paste(grid, patch, r0, c0, transparent=BACKGROUND)

    if touching_border > MAX_BORDER_TOUCHES:
        return None
    return grid


def _passes_filters(grid: np.ndarray, frame_color: int) -> bool:
    """Apply distribution guards and solver-oracle consistency checks."""
    coverage = float(np.mean(grid != 0))
    if coverage < COVERAGE_RANGE[0] or coverage > COVERAGE_RANGE[1]:
        return False
    counts = np.bincount(grid.ravel(), minlength=10)
    if int(counts[0]) <= int(max(counts[1:])):
        return False
    expected = np.array([[frame_color]], dtype=int)
    try:
        actual = main(grid)
    except AssertionError:
        return False
    return np.array_equal(actual, expected)


def _gen_attempt() -> np.ndarray | None:
    rows = random.randint(*CANVAS_ROWS)
    cols = random.randint(*CANVAS_COLS)
    n_objects = random.randint(*OBJECT_COUNT_RANGE)
    patches, frame_color = _build_patches(n_objects)
    grid = _place_patches(rows, cols, patches)
    return grid if grid is not None and _passes_filters(grid, frame_color) else None


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid b9b7f026 input")
