import numpy as np

from arc_utils import Color, get_bg
import random


def _outline_from_missing_corner(
    r0: int, c0: int, r1: int, c1: int, missing_corner: tuple[int, int]
) -> set[tuple[int, int]]:
    n = r1 - r0
    outline: set[tuple[int, int]] = set()

    if missing_corner == (r0, c0):
        outline.update((r0, c) for c in range(c0, c1 + 1))
        outline.update((r, c0) for r in range(r0, r1 + 1))
        outline.update((r0 + i, c1 - i) for i in range(n + 1))
    elif missing_corner == (r0, c1):
        outline.update((r0, c) for c in range(c0, c1 + 1))
        outline.update((r, c1) for r in range(r0, r1 + 1))
        outline.update((r0 + i, c0 + i) for i in range(n + 1))
    elif missing_corner == (r1, c0):
        outline.update((r, c0) for r in range(r0, r1 + 1))
        outline.update((r1, c) for c in range(c0, c1 + 1))
        outline.update((r0 + i, c0 + i) for i in range(n + 1))
    else:
        assert missing_corner == (r1, c1)
        outline.update((r, c1) for r in range(r0, r1 + 1))
        outline.update((r1, c) for c in range(c0, c1 + 1))
        outline.update((r0 + i, c1 - i) for i in range(n + 1))

    return outline


def main(grid: np.ndarray) -> np.ndarray:
    """Paint the teal outline of the right triangle determined by the missing corner of the grey-marker square.

    Input structure
    The grid is an orange background with exactly four foreground marker pixels of one color. The markers have a square bounding box: three markers are bbox corners, and the fourth marker is inside that square.
    Rule
    Find the one corner of the marker bounding square that has no marker. That missing corner is the right-angle vertex of a right isosceles triangle; paint its two axis-aligned legs and the opposite diagonal in teal, leaving existing marker pixels unchanged.
    Key insight
    The present corner marker opposite the missing corner is a decoy; the three triangle vertices are the missing corner and the two marker corners adjacent to it.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)

    marker_colors = [int(c) for c in np.unique(g) if int(c) != bg]
    assert len(marker_colors) == 1
    marker = marker_colors[0]
    assert marker != Color.TEAL and bg != Color.TEAL

    marker_coords = np.argwhere(g == marker)
    assert marker_coords.shape == (4, 2)
    r0, c0 = marker_coords.min(axis=0)
    r1, c1 = marker_coords.max(axis=0)
    r0, c0, r1, c1 = map(int, (r0, c0, r1, c1))
    assert r1 - r0 == c1 - c0

    corners = {(r0, c0), (r0, c1), (r1, c0), (r1, c1)}
    marker_set = {tuple(map(int, coord)) for coord in marker_coords}
    missing_corners = corners - marker_set
    assert len(missing_corners) == 1
    assert len(marker_set & corners) == 3

    out = g.copy()
    outline = _outline_from_missing_corner(r0, c0, r1, c1, missing_corners.pop())
    for r, c in outline:
        if out[r, c] == bg:
            out[r, c] = Color.TEAL

    return out

# --------------- generator (c) ---------------

# --- CFG ---
GRID_SIZE_MIN, GRID_SIZE_MAX = 7, 12  # square grid side range
SIDE_MIN = 3                          # minimum marker-bbox side length
MAX_RETRIES = 300                     # generate_input retry budget
_NON_TEAL = [c for c in range(10) if c != Color.TEAL]


def _gen_attempt() -> np.ndarray | None:
    """Build a single candidate input grid, or None on layout conflict."""
    H = W = random.randint(GRID_SIZE_MIN, GRID_SIZE_MAX)

    # Pick bg and marker colors (neither may be teal)
    bg = random.choice(_NON_TEAL)
    marker_color = random.choice([c for c in _NON_TEAL if c != bg])

    # Square bbox: side length S in [SIDE_MIN, H-1], placed randomly
    S = random.randint(SIDE_MIN, H - 1)
    r0 = random.randint(0, H - 1 - S)
    c0 = random.randint(0, W - 1 - S)
    r1, c1 = r0 + S, c0 + S

    # Choose which bbox corner to omit; the 4th marker sits 1 diagonal
    # step inward from the missing corner toward the opposite corner
    corners = [(r0, c0), (r0, c1), (r1, c0), (r1, c1)]
    idx = random.randint(0, 3)
    missing = corners[idx]
    opposite = corners[3 - idx]
    dr = 1 if opposite[0] > missing[0] else -1
    dc = 1 if opposite[1] > missing[1] else -1
    inner = (missing[0] + dr, missing[1] + dc)

    # Place markers
    grid = np.full((H, W), bg, dtype=int)
    for r, c in corners:
        if (r, c) != missing:
            grid[r, c] = marker_color
    grid[inner] = marker_color

    # Guard: exactly 4 foreground pixels (collision would reduce count)
    if np.count_nonzero(grid != bg) != 4:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is None:
            continue
        try:
            out = main(grid)
            if not np.array_equal(out, grid):
                return grid
        except Exception:
            continue
    raise RuntimeError(f"Failed to generate valid input after {MAX_RETRIES} attempts")
