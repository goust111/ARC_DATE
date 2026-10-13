import numpy as np

from arc_utils import get_bg
import random


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend each non-background pixel into an L reaching the nearest vertical and horizontal grid edges.

    Input structure
    A mostly background grid contains one or more isolated foreground marker pixels.
    Each marker is closer to one vertical edge and one horizontal edge, with no midpoint ties.

    Rule
    For every marker at (r, c), draw its color along column c from the marker to the nearer
    top or bottom edge, and along row r from the marker to the nearer left or right edge.

    Key insight
    Each single pixel is the corner of an edge-facing L; its directions are chosen by nearest
    grid-boundary distance rather than by color identity.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    bg = get_bg(g)
    markers = np.argwhere(g != bg)
    assert len(markers) > 0

    for r, c in markers:
        r = int(r)
        c = int(c)
        for rr in range(max(0, r - 1), min(h, r + 2)):
            for cc in range(max(0, c - 1), min(w, c + 2)):
                if (rr, cc) != (r, c):
                    assert g[rr, cc] == bg

    out = g.copy()
    claimed = np.full(g.shape, -1, dtype=int)

    for r, c in markers:
        r = int(r)
        c = int(c)
        color = int(g[r, c])

        up = r
        down = h - 1 - r
        left = c
        right = w - 1 - c
        assert up != down
        assert left != right

        r_edge = 0 if up < down else h - 1
        c_edge = 0 if left < right else w - 1

        mask = np.zeros(g.shape, dtype=bool)
        mask[min(r, r_edge) : max(r, r_edge) + 1, c] = True
        mask[r, min(c, c_edge) : max(c, c_edge) + 1] = True

        assert np.all((g[mask] == bg) | (g[mask] == color))
        assert np.all((claimed[mask] == -1) | (claimed[mask] == color))
        claimed[mask] = color
        out[mask] = color

    return out

# --------------- generator (o5x) ---------------

# Canvas bounds and background color.
_GEN_SIZE_RANGE = (8, 16)
_GEN_BACKGROUND = 0

# Marker counts, colors, and per-marker quadrant assignment.
_GEN_MARKER_COUNT_RANGE = (1, 4)
_GEN_COLOR_RANGE = range(1, 10)
_GEN_QUADRANTS = ("TL", "TR", "BL", "BR")
_GEN_MARKER_MARGIN = 1

# Retry policy for rejection sampling.
_GEN_RETRY_LIMIT = 300


def _gen_l_mask(h: int, w: int, r: int, c: int) -> np.ndarray:
    r_edge = 0 if r < h - 1 - r else h - 1
    c_edge = 0 if c < w - 1 - c else w - 1
    mask = np.zeros((h, w), dtype=bool)
    mask[min(r, r_edge) : max(r, r_edge) + 1, c] = True
    mask[r, min(c, c_edge) : max(c, c_edge) + 1] = True
    return mask


def _gen_in_quadrant(q: str, h: int, w: int, r: int, c: int) -> bool:
    row_side = "T" if 2 * r < h - 1 else "B" if 2 * r > h - 1 else ""
    col_side = "L" if 2 * c < w - 1 else "R" if 2 * c > w - 1 else ""
    return q == row_side + col_side


def _gen_is_isolated(markers: list[tuple[int, int]], r: int, c: int) -> bool:
    return all(max(abs(r - pr), abs(c - pc)) > 1 for pr, pc in markers)


def _gen_valid_cells(
    q: str,
    h: int,
    w: int,
    markers: list[tuple[int, int]],
    occupied: np.ndarray,
) -> list[tuple[int, int, np.ndarray]]:
    cells = []
    for r in range(_GEN_MARKER_MARGIN, h - _GEN_MARKER_MARGIN):
        if 2 * r == h - 1:
            continue
        for c in range(_GEN_MARKER_MARGIN, w - _GEN_MARKER_MARGIN):
            if 2 * c == w - 1:
                continue
            if not _gen_in_quadrant(q, h, w, r, c):
                continue
            if not _gen_is_isolated(markers, r, c):
                continue
            mask = _gen_l_mask(h, w, r, c)
            if np.any(mask & occupied):
                continue
            cells.append((r, c, mask))
    return cells


def _gen_place_marker(
    inp: np.ndarray,
    out: np.ndarray,
    occupied: np.ndarray,
    markers: list[tuple[int, int]],
    color: int,
    quadrant: str,
) -> bool:
    h, w = inp.shape
    cells = _gen_valid_cells(quadrant, h, w, markers, occupied)
    if not cells:
        return False
    r, c, mask = random.choice(cells)
    markers.append((r, c))
    inp[r, c] = color
    out[mask] = color
    occupied[mask] = True
    return True


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*_GEN_SIZE_RANGE)
    w = random.randint(*_GEN_SIZE_RANGE)
    n_markers = random.randint(*_GEN_MARKER_COUNT_RANGE)
    colors = random.sample(_GEN_COLOR_RANGE, n_markers)
    quadrants = random.sample(_GEN_QUADRANTS, n_markers)
    random.shuffle(quadrants)

    inp = np.full((h, w), _GEN_BACKGROUND, dtype=int)
    out = np.full((h, w), _GEN_BACKGROUND, dtype=int)
    occupied = np.zeros((h, w), dtype=bool)
    markers: list[tuple[int, int]] = []

    for color, quadrant in zip(colors, quadrants):
        if not _gen_place_marker(inp, out, occupied, markers, color, quadrant):
            return None

    if np.array_equal(main(inp), out):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 705a3229 input")
