import numpy as np

from arc_utils import get_bg
import random


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill the rectangle from the lone foreground pixel to its nearest grid corner.

    Input structure
    - The grid is a uniform background field with exactly one foreground pixel.
    - The foreground pixel's color is the color used for all added pixels.

    Rule
    - Find the grid corner with the smallest Manhattan distance to the foreground pixel.
    - Recolor every pixel in the inclusive axis-aligned rectangle bounded by that
      corner and the foreground pixel to the foreground color.

    Key insight
    - The single pixel is one corner of the rectangle; the opposite corner is the
      nearest corner of the whole canvas.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    foreground = np.argwhere(g != bg)
    assert foreground.shape[0] == 1, "expected exactly one foreground pixel"

    r, c = map(int, foreground[0])
    color = int(g[r, c])
    H, W = g.shape
    corners = [(0, 0), (0, W - 1), (H - 1, 0), (H - 1, W - 1)]
    distances = [abs(r - corner_r) + abs(c - corner_c) for corner_r, corner_c in corners]
    nearest = min(distances)
    assert distances.count(nearest) == 1, "expected a unique nearest corner"

    corner_r, corner_c = corners[distances.index(nearest)]
    r0, r1 = sorted((r, corner_r))
    c0, c1 = sorted((c, corner_c))

    out = g.copy()
    out[r0 : r1 + 1, c0 : c1 + 1] = color
    return out

# --------------- generator (att2, o5x) ---------------

# Canvas and palette.
BACKGROUND_COLOR = 8
FOREGROUND_COLORS = [1, 2, 3, 4, 5, 6, 7, 9]
GRID_SIZE_MIN = 4
GRID_SIZE_MAX = 10

# Corner-biased placement.
CORNER_LABELS = ("TL", "TR", "BL", "BR")
CORNER_SIDES = dict(
    TL=("near", "near"), TR=("near", "far"),
    BL=("far", "near"), BR=("far", "far"),
)

# Retry budget for oracle-checked generation.
MAX_GENERATION_ATTEMPTS = 300


def _axis_ranges(length: int) -> dict[str, tuple[int, int]]:
    """Return inclusive near/far bands that avoid corner-distance ties."""
    return {
        "near": (0, (length - 2) // 2),
        "far": ((length + 1) // 2, length - 1),
    }


def _pick_cell(r0: int, r1: int, c0: int, c1: int, exclude: tuple[int, int]) -> tuple[int, int]:
    cells = [(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1) if (r, c) != exclude]
    return random.choice(cells)


def _coord_for_corner(h: int, w: int, corner: str) -> tuple[int, int, int, int]:
    row_side, col_side = CORNER_SIDES[corner]
    r0, r1 = _axis_ranges(h)[row_side]
    c0, c1 = _axis_ranges(w)[col_side]
    corner_r = 0 if row_side == "near" else h - 1
    corner_c = 0 if col_side == "near" else w - 1
    r, c = _pick_cell(r0, r1, c0, c1, (corner_r, corner_c))
    return r, c, corner_r, corner_c


def _expected_output(grid: np.ndarray, r: int, c: int, corner_r: int, corner_c: int, color: int) -> np.ndarray:
    out = grid.copy()
    r0, r1 = sorted((r, corner_r))
    c0, c1 = sorted((c, corner_c))
    out[r0 : r1 + 1, c0 : c1 + 1] = color
    return out


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(GRID_SIZE_MIN, GRID_SIZE_MAX)
    w = random.randint(GRID_SIZE_MIN, GRID_SIZE_MAX)
    color = random.choice(FOREGROUND_COLORS)
    corner = random.choice(CORNER_LABELS)
    r, c, corner_r, corner_c = _coord_for_corner(h, w, corner)

    grid = np.full((h, w), BACKGROUND_COLOR, dtype=int)
    grid[r, c] = color
    expected = _expected_output(grid, r, c, corner_r, corner_c, color)
    return grid if np.array_equal(main(grid), expected) else None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
