import numpy as np

from arc_utils import Color, get_bg


def main(grid: np.ndarray) -> np.ndarray:
    """Replace each grey center marker with a solid blue 3x3 block on the 3x3 macro-grid.

    Input structure
    - The grid is a black-background canvas whose dimensions are multiples of 3.
    - Foreground pixels are grey markers, each centered inside a 3x3 macro block.

    Rule
    - Start with a blank output of the same shape.
    - For every grey marker, fill its containing 3x3 macro block with blue.

    Key insight
    - The grey pixels are control markers for block activation; they are not preserved as output objects.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = get_bg(g)
    assert bg == Color.BLACK, "expected black background"
    height, width = g.shape
    assert height % 3 == 0 and width % 3 == 0, "grid must partition into 3x3 blocks"

    non_bg = np.argwhere(g != bg)
    assert non_bg.shape[0] > 0, "expected at least one marker"
    assert np.all(g[g != bg] == Color.GREY), "all foreground pixels must be grey markers"

    out = np.full_like(g, bg)
    for r, c in non_bg:
        r = int(r)
        c = int(c)
        assert r % 3 == 1 and c % 3 == 1, "markers must be centered in 3x3 blocks"
        block = (r // 3, c // 3)
        r0 = block[0] * 3
        c0 = block[1] * 3
        out[r0 : r0 + 3, c0 : c0 + 3] = Color.BLUE

    return out

# --------------- generator (o5x) ---------------
# Canvas and macro-grid configuration.
SIDE_CHOICES = (9, 12, 15)
BLOCK_SIZE = 3
# Active macro-block sampling limits.
MIN_MARKS = 2
MAX_MARKS = 5
# Oracle retry limit.
GEN_RETRIES = 300


def _sample_blocks() -> tuple[int, int, np.ndarray]:
    side = int(np.random.choice(SIDE_CHOICES))
    macro_side = side // BLOCK_SIZE
    macro_count = macro_side * macro_side
    mark_count = int(np.random.randint(MIN_MARKS, min(MAX_MARKS, macro_count - 1) + 1))
    return side, macro_side, np.random.choice(macro_count, size=mark_count, replace=False)


def _gen_attempt() -> np.ndarray | None:
    side, macro_side, chosen = _sample_blocks()
    out = np.full((side, side), Color.BLACK, dtype=int)
    inp = np.full_like(out, Color.BLACK)
    for block_i in chosen:
        br, bc = divmod(int(block_i), macro_side)
        r0, c0 = br * BLOCK_SIZE, bc * BLOCK_SIZE
        out[r0 : r0 + BLOCK_SIZE, c0 : c0 + BLOCK_SIZE] = Color.BLUE
        inp[r0 + BLOCK_SIZE // 2, c0 + BLOCK_SIZE // 2] = Color.GREY

    return inp if np.array_equal(main(inp), out) else None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid ce22a75a input")
