import numpy as np


def _corner_l_spec(g: np.ndarray) -> tuple[int, int, bool, bool]:
    h, w = g.shape
    colors = [int(c) for c in np.unique(g)]
    assert len(colors) == 2

    candidates: list[tuple[int, int, bool, bool]] = []
    for border_color in colors:
        fill_color = next(c for c in colors if c != border_color)
        for flip_rows, edge_r in ((False, 0), (True, h - 1)):
            for flip_cols, edge_c in ((False, 0), (True, w - 1)):
                l_mask = np.zeros((h, w), dtype=bool)
                l_mask[edge_r, :] = True
                l_mask[:, edge_c] = True
                if np.all(g[l_mask] == border_color) and np.all(g[~l_mask] == fill_color):
                    candidates.append((border_color, fill_color, flip_rows, flip_cols))

    assert len(candidates) == 1
    return candidates[0]


def main(grid: np.ndarray) -> np.ndarray:
    """Rewrite a corner L block into two nested L borders and a diagonal trace.

    Input structure
    A rectangular grid has exactly two colors: one color forms a one-cell-thick
    L on two adjacent outer edges, and the other color fills the remaining
    rectangle.

    Rule
    Normalize the L to the top-left corner. Paint the outer top-left L with the
    fill color, the next inner L with the original border color, and the main
    diagonal of the remaining inner rectangle with the fill color. All other
    cells become color 8, then the original orientation is restored.

    Key insight
    The corner of the input L determines the orientation; after flipping to a
    canonical top-left L, the output is the same nested-layer pattern for every
    example.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    assert h >= 3 and w >= 3
    assert 8 not in np.unique(g)

    border_color, fill_color, flip_rows, flip_cols = _corner_l_spec(g)

    canon = g
    if flip_rows:
        canon = np.flipud(canon)
    if flip_cols:
        canon = np.fliplr(canon)

    assert np.all(canon[0, :] == border_color)
    assert np.all(canon[:, 0] == border_color)
    assert np.all(canon[1:, 1:] == fill_color)

    out = np.full((h, w), 8, dtype=int)
    out[0, 1:] = fill_color
    out[1:, 0] = fill_color
    out[1, 2:] = border_color
    out[2:, 1] = border_color
    for k in range(min(h - 2, w - 2)):
        out[2 + k, 2 + k] = fill_color

    if flip_cols:
        out = np.fliplr(out)
    if flip_rows:
        out = np.flipud(out)
    return out

# --------------- generator (att2) ---------------

# Canvas, palette, and orientation sampling knobs.
HEIGHT_RANGE = (5, 11)
WIDTH_RANGE = (6, 11)
AVAILABLE_COLORS = [0, 1, 2, 3, 4, 5, 6, 7, 9]
CORNER_EDGES = ((0, 0), (0, -1), (-1, 0), (-1, -1))
CORNER_FLIPS = ((False, False), (False, True), (True, False), (True, True))
# Rejection budget: oracle check should pass on the first attempt.
MAX_ATTEMPTS = 300

def _sample_params() -> tuple[int, int, int, int, int]:
    """Sample dimensions, ordered colors, and one of the four L corners."""
    h = int(np.random.randint(*HEIGHT_RANGE))
    w = int(np.random.randint(*WIDTH_RANGE))
    border_color, fill_color = np.random.choice(AVAILABLE_COLORS, size=2, replace=False)
    corner = int(np.random.randint(len(CORNER_EDGES)))
    return h, w, int(border_color), int(fill_color), corner


def _paint_corner_l(inp: np.ndarray, border_color: int, corner: int) -> None:
    """Paint the full row and column for the requested corner in-place."""
    edge_r, edge_c = CORNER_EDGES[corner]
    inp[edge_r, :] = border_color
    inp[:, edge_c] = border_color


def _canonical_output(h: int, w: int, border_color: int, fill_color: int) -> np.ndarray:
    """Render the top-left canonical answer before restoring orientation."""
    out = np.full((h, w), 8, dtype=int)
    out[0, 1:] = fill_color
    out[1:, 0] = fill_color
    out[1, 2:] = border_color
    out[2:, 1] = border_color
    diag = np.arange(min(h - 2, w - 2))
    out[2 + diag, 2 + diag] = fill_color
    return out


def _restore_orientation(grid: np.ndarray, corner: int) -> np.ndarray:
    """Flip a canonical top-left grid back to the sampled corner."""
    flip_rows, flip_cols = CORNER_FLIPS[corner]
    if flip_cols:
        grid = np.fliplr(grid)
    if flip_rows:
        grid = np.flipud(grid)
    return grid


def _gen_attempt() -> np.ndarray | None:
    h, w, border_color, fill_color, corner = _sample_params()
    inp = np.full((h, w), fill_color, dtype=int)
    _paint_corner_l(inp, border_color, corner)
    out = _restore_orientation(_canonical_output(h, w, border_color, fill_color), corner)

    if not np.array_equal(main(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid bd5af378 input")
