import numpy as np

from arc_utils import Color, bbox_h, bbox_w, find_components, get_bg, paste, random_free_location
import random


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill the empty interior of each colored rectangular frame, using grey for a one-pixel interior and orange for a larger interior.

    Input structure:
    - The background is the dominant color.
    - Every foreground object is a monochrome one-cell-thick rectangular frame.
    - Each frame has a background-colored rectangular interior.

    Rule:
    - Preserve the frame pixels.
    - Fill a frame's interior with grey when the interior has exactly one pixel.
    - Fill a frame's interior with orange when the interior has more than one pixel.

    Key insight:
    - The output color is determined by the size of the enclosed background region, not by the frame color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = get_bg(g)
    objects = find_components(g, bg=bg, connectivity=4, monochromatic=True)
    assert objects, "expected at least one foreground frame"

    out = g.copy()
    for obj in objects:
        color = int(obj["color"])
        r0, c0, r1, c1 = obj["bbox"]
        height = bbox_h(obj["bbox"])
        width = bbox_w(obj["bbox"])
        assert height >= 3 and width >= 3, "frames must enclose a nonempty interior"

        patch = g[r0 : r1 + 1, c0 : c1 + 1]
        border = np.zeros((height, width), dtype=bool)
        border[0, :] = True
        border[-1, :] = True
        border[:, 0] = True
        border[:, -1] = True

        assert np.all(patch[border] == color), "expected a solid monochrome frame border"
        assert np.all(patch[~border] == bg), "expected an empty frame interior"
        interior_area = int((~border).sum())
        assert interior_area >= 1, "expected a nonempty interior"

        fill = Color.GREY if interior_area == 1 else Color.ORANGE
        out[r0 + 1 : r1, c0 + 1 : c1] = fill

    return out

# --------------- generator (o5x) ---------------

# Canvas and frame sampling.
_GEN_BACKGROUND = Color.BLACK
_GEN_ROWS_RANGE, _GEN_COLS_RANGE = (6, 16), (6, 13)
_GEN_FRAME_COUNT_RANGE, _GEN_FRAME_SIZES = (1, 5), ((3, 3), (3, 4), (4, 3))
_GEN_FRAME_COLORS = (1, 2, 3, 4, 6, 8, 9)

# Placement, fill, and retry controls.
_GEN_FRAME_BORDER = 1
_GEN_FRAME_PADDING, _GEN_PADDING_CONNECTIVITY = 1, 8
_GEN_MAX_ATTEMPTS, _GEN_SINGLE_FILL, _GEN_MULTI_FILL = 300, Color.GREY, Color.ORANGE

def _gen_frame_patch(height: int, width: int, color: int) -> np.ndarray:
    """Render a black-interior rectangular frame patch."""
    patch = np.full((height, width), _GEN_BACKGROUND, dtype=int)
    patch[0, :] = color
    patch[-1, :] = color
    patch[:, 0] = color
    patch[:, -1] = color
    return patch

def _sample_frames() -> list[tuple[int, int, int]]:
    """Sample frame dimensions/colors and shuffle placement order."""
    frame_count = random.randint(*_GEN_FRAME_COUNT_RANGE)
    frames = [
        (*random.choice(_GEN_FRAME_SIZES), random.choice(_GEN_FRAME_COLORS))
        for _ in range(frame_count)
    ]
    random.shuffle(frames)
    return frames

def _paste_frame(grid: np.ndarray, expected: np.ndarray, frame: tuple[int, int, int]) -> bool:
    """Paste one frame into both grids; return False if no legal slot exists."""
    height, width, color = frame
    patch = _gen_frame_patch(height, width, color)
    try:
        r0, c0 = random_free_location(
            grid,
            patch,
            transparent=_GEN_BACKGROUND,
            padding=_GEN_FRAME_PADDING,
            padding_connectivity=_GEN_PADDING_CONNECTIVITY,
        )
    except ValueError:
        return False

    paste(grid, patch, r0, c0, transparent=_GEN_BACKGROUND)
    paste(expected, patch, r0, c0, transparent=_GEN_BACKGROUND)
    interior_area = (height - 2 * _GEN_FRAME_BORDER) * (width - 2 * _GEN_FRAME_BORDER)
    fill = _GEN_SINGLE_FILL if interior_area == 1 else _GEN_MULTI_FILL
    expected[
        r0 + _GEN_FRAME_BORDER : r0 + height - _GEN_FRAME_BORDER,
        c0 + _GEN_FRAME_BORDER : c0 + width - _GEN_FRAME_BORDER,
    ] = fill
    return True

def _has_strict_background_mode(grid: np.ndarray) -> bool:
    """Require black to outnumber every possible frame color."""
    return int(np.count_nonzero(grid == _GEN_BACKGROUND)) > max(
        int(np.count_nonzero(grid == color)) for color in _GEN_FRAME_COLORS
    )

def _gen_attempt() -> np.ndarray | None:
    rows, cols = random.randint(*_GEN_ROWS_RANGE), random.randint(*_GEN_COLS_RANGE)
    grid = np.full((rows, cols), _GEN_BACKGROUND, dtype=int)
    expected = grid.copy()

    for frame in _sample_frames():
        if not _paste_frame(grid, expected, frame):
            return None
    if not _has_strict_background_mode(grid):
        return None
    try:
        solved = main(grid)
    except Exception:
        return None
    if not np.array_equal(solved, expected):
        return None
    return grid

def generate_input() -> np.ndarray:
    for _ in range(_GEN_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input: failed after {_GEN_MAX_ATTEMPTS} attempts")
