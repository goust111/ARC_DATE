import numpy as np

from arc_utils import Color, bbox_coords, colors_present, components_mask, get_bg, paste, random_free_location
import random


def main(grid: np.ndarray) -> np.ndarray:
    """
    Erase the red rectangular outlines and recolor each enclosed black interior green.

    Input structure
    - The grid has a black background and one foreground color, red.
    - Red foreground components are 4-connected axis-aligned rectangles.
    - A component is either a one-cell-thick hollow rectangular frame or a solid rectangle with no black interior.

    Rule
    - Start from an all-background output grid of the same shape.
    - For each hollow red rectangular frame, fill its strict bbox interior with green.
    - Solid red rectangles and all red frame borders are erased to background.

    Key insight
    - The red pixels only mark rectangular boundaries; the retained output is the background region enclosed by each boundary, recolored green.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = get_bg(g)
    assert bg == Color.BLACK, "expected black background"
    foreground_colors = colors_present(g, exclude={bg})
    assert foreground_colors == [Color.RED], "expected red to be the only foreground color"

    out = np.full_like(g, bg)
    frame_count = 0

    for coords in components_mask(g == Color.RED, connectivity=4):
        r0, c0, r1, c1 = bbox_coords(coords)
        sub = g[r0 : r1 + 1, c0 : c1 + 1]
        mask = sub == Color.RED
        height, width = mask.shape

        if height >= 3 and width >= 3:
            border = np.zeros_like(mask, dtype=bool)
            border[0, :] = True
            border[-1, :] = True
            border[:, 0] = True
            border[:, -1] = True
            if np.array_equal(mask, border):
                out[r0 + 1 : r1, c0 + 1 : c1] = Color.GREEN
                frame_count += 1
                continue

        assert np.all(mask), "non-frame red components must be solid rectangles"

    assert frame_count > 0, "expected at least one hollow rectangular frame"
    return out

# --------------- generator (o5x) ---------------
# Canvas, component count, and component size bounds.
GRID_SIZE_RANGE = (10, 25)
FRAME_COUNT_RANGE, SOLID_COUNT_RANGE = (1, 3), (0, 1)
FRAME_HEIGHT_RANGE, FRAME_WIDTH_RANGE = (3, 6), (3, 9)
SOLID_SIZE = (2, 2)

# Placement policy and retry limit.
FOOTPRINT_COLOR = 1
PLACEMENT_OPTIONS = {"border": 0, "padding": 1, "padding_connectivity": 8}
GENERATION_ATTEMPTS = 300


def _gen_frame_patch(height: int, width: int) -> np.ndarray:
    patch = np.full((height, width), Color.BLACK, dtype=int)
    patch[0, :] = Color.RED
    patch[-1, :] = Color.RED
    patch[:, 0] = Color.RED
    patch[:, -1] = Color.RED
    return patch


def _gen_expected_output(size: int, frame_boxes: list[tuple[int, int, int, int]]) -> np.ndarray:
    out = np.full((size, size), Color.BLACK, dtype=int)
    for r0, c0, height, width in frame_boxes:
        out[r0 + 1 : r0 + height - 1, c0 + 1 : c0 + width - 1] = Color.GREEN
    return out


def _gen_component_specs() -> list[tuple[str, int, int]]:
    """Return shuffled component specs while preserving the original draw order."""
    frame_count = random.randint(*FRAME_COUNT_RANGE)
    solid_count = random.randint(*SOLID_COUNT_RANGE)
    components = [
        ("frame", random.randint(*FRAME_HEIGHT_RANGE), random.randint(*FRAME_WIDTH_RANGE))
        for _ in range(frame_count)
    ]
    components.extend(("solid", SOLID_SIZE[0], SOLID_SIZE[1]) for _ in range(solid_count))
    random.shuffle(components)
    return components


def _gen_attempt() -> np.ndarray | None:
    size = random.randint(*GRID_SIZE_RANGE)
    components = _gen_component_specs()
    grid = np.full((size, size), Color.BLACK, dtype=int)
    footprint_canvas = np.full((size, size), Color.BLACK, dtype=int)
    frame_boxes: list[tuple[int, int, int, int]] = []

    for kind, height, width in components:
        footprint = np.full((height, width), FOOTPRINT_COLOR, dtype=int)
        try:
            r0, c0 = random_free_location(
                footprint_canvas,
                footprint,
                transparent=Color.BLACK,
                **PLACEMENT_OPTIONS,
            )
        except ValueError:
            return None

        paste(footprint_canvas, footprint, r0, c0, transparent=Color.BLACK)
        if kind == "frame":
            paste(grid, _gen_frame_patch(height, width), r0, c0)
            frame_boxes.append((r0, c0, height, width))
        else:
            paste(grid, np.full(SOLID_SIZE, Color.RED, dtype=int), r0, c0)

    expected = _gen_expected_output(size, frame_boxes)
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid d5d6de2d input")
