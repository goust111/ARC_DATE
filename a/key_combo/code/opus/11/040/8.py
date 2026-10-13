import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
ORANGE = 7

PROMPT_BLOCK = np.array([[YELLOW, BLACK], [BLACK, YELLOW]], dtype=int)

PANEL_H_RANGE = (4, 8)   # inclusive-exclusive
PANEL_W_RANGE = (10, 16)
FG_COUNT_RANGE = (8, 16)
OVERLAP_RANGE = (3, 7)
GEN_RETRY_LIMIT = 500


def _prompt_mask(h: int, w: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    mask[h - 2 :, w - 2 :] = True
    return mask


def _split_input(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    full_dividers = np.where(np.all(g == GREY, axis=1))[0]
    assert len(full_dividers) == 1, "expected exactly one full-width grey divider row"
    sep = int(full_dividers[0])
    assert sep * 2 + 1 == g.shape[0], "divider must be centered"

    top = g[:sep].copy()
    bottom = g[sep + 1 :].copy()
    assert top.shape == bottom.shape, "divider must split two equal panels"
    assert top.shape[0] >= 2 and top.shape[1] >= 2, "panels must allow a 2x2 prompt"

    assert np.array_equal(bottom[-2:, -2:], PROMPT_BLOCK), "missing lower-right checkerboard prompt"
    bottom[-2:, -2:] = BLACK
    return top, bottom


def _expected_output(top: np.ndarray, bottom: np.ndarray) -> np.ndarray:
    h, w = top.shape
    top_fg = top == BLUE
    bottom_fg = bottom == RED

    assert np.all((top == BLACK) | (top == BLUE)), "top panel must use only black and blue"
    assert np.all((bottom == BLACK) | (bottom == RED)), "bottom panel must use only black and red outside prompt"

    rows = np.arange(h)[:, None]
    cols = np.arange(w)[None, :]
    keep = ((rows + cols) % 2) == 0

    out = np.zeros((h, w), dtype=int)
    both = top_fg & bottom_fg & keep
    top_only = top_fg & ~bottom_fg & keep
    bottom_only = ~top_fg & bottom_fg & keep

    out[both] = GREEN
    out[top_only] = YELLOW
    out[bottom_only] = ORANGE
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Overlay aligned dual panels, keep only even checkerboard cells, and recolor by source.

    Input structure:
    - The grid has a black background and one centered full-width grey divider row.
    - Above and below the divider are two equal-sized panels aligned by shared (r, c) coordinates.
    - The top panel uses blue foreground pixels, the bottom panel uses red foreground pixels, and the input's lower-right corner contains a 2x2 yellow-black checkerboard hint block.

    Rule:
    - Remove the divider and ignore the 2x2 hint block from the lower-right of the bottom panel.
    - Compare the aligned top and bottom panel cells position by position.
    - If (r + c) is odd in the output panel coordinates, output black.
    - If (r + c) is even, output green where both panels have foreground, yellow where only the top panel has foreground, orange where only the bottom panel has foreground, and black otherwise.

    Key insight:
    - The final image is not the raw overlay of the two masks: the overlay categories are first classified by source and then half of the positions are erased by an even-parity checkerboard filter signaled by the corner hint.
    """
    top, bottom = _split_input(grid)
    return _expected_output(top, bottom)


def _sample_branch_counts(rng: np.random.Generator) -> dict[str, int]:
    for _ in range(200):
        both_even = int(rng.integers(1, 4))
        both_odd = int(rng.integers(1, 4))
        top_even = int(rng.integers(1, 5))
        top_odd = int(rng.integers(1, 6))
        bottom_even = int(rng.integers(1, 5))
        bottom_odd = int(rng.integers(1, 6))

        overlap = both_even + both_odd
        top_total = overlap + top_even + top_odd
        bottom_total = overlap + bottom_even + bottom_odd
        kept_total = both_even + top_even + bottom_even
        odd_overlay = both_odd + top_odd + bottom_odd

        if not (OVERLAP_RANGE[0] <= overlap < OVERLAP_RANGE[1]):
            continue
        if not (FG_COUNT_RANGE[0] <= top_total < FG_COUNT_RANGE[1]):
            continue
        if not (FG_COUNT_RANGE[0] <= bottom_total < FG_COUNT_RANGE[1]):
            continue
        if kept_total < 5 or odd_overlay < 4:
            continue

        return {
            "both_even": both_even,
            "both_odd": both_odd,
            "top_even": top_even,
            "top_odd": top_odd,
            "bottom_even": bottom_even,
            "bottom_odd": bottom_odd,
        }
    raise RuntimeError("failed to sample non-degenerate branch counts")


def _coord_pools(h: int, w: int) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    prompt = _prompt_mask(h, w)
    even, odd = [], []
    for r in range(h):
        for c in range(w):
            if prompt[r, c]:
                continue
            if (r + c) % 2 == 0:
                even.append((r, c))
            else:
                odd.append((r, c))
    return even, odd


def _take_coords(
    rng: np.random.Generator, coords: list[tuple[int, int]], count: int
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    assert count <= len(coords), "not enough coordinates in pool"
    order = rng.permutation(len(coords))
    chosen_idx = set(int(i) for i in order[:count])
    chosen = [coords[i] for i in order[:count]]
    remaining = [coords[i] for i in range(len(coords)) if i not in chosen_idx]
    return chosen, remaining


def _build_masks(h: int, w: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    counts = _sample_branch_counts(rng)
    even_pool, odd_pool = _coord_pools(h, w)

    both_even, even_pool = _take_coords(rng, even_pool, counts["both_even"])
    top_even, even_pool = _take_coords(rng, even_pool, counts["top_even"])
    bottom_even, even_pool = _take_coords(rng, even_pool, counts["bottom_even"])

    both_odd, odd_pool = _take_coords(rng, odd_pool, counts["both_odd"])
    top_odd, odd_pool = _take_coords(rng, odd_pool, counts["top_odd"])
    bottom_odd, odd_pool = _take_coords(rng, odd_pool, counts["bottom_odd"])

    top = np.zeros((h, w), dtype=int)
    bottom = np.zeros((h, w), dtype=int)

    for r, c in both_even + both_odd:
        top[r, c] = BLUE
        bottom[r, c] = RED
    for r, c in top_even + top_odd:
        top[r, c] = BLUE
    for r, c in bottom_even + bottom_odd:
        bottom[r, c] = RED

    return top, bottom


def _assemble_grid(top: np.ndarray, bottom: np.ndarray) -> np.ndarray:
    h, w = top.shape
    grid = np.zeros((2 * h + 1, w), dtype=int)
    grid[:h] = top
    grid[h] = GREY
    grid[h + 1 :] = bottom
    grid[-2:, -2:] = PROMPT_BLOCK
    return grid


def _validate_generated(grid: np.ndarray) -> None:
    top, bottom = _split_input(grid)
    out = main(grid)
    expected = _expected_output(top, bottom)
    assert np.array_equal(out, expected), "main does not match the specified rule"

    top_fg = top == BLUE
    bottom_fg = bottom == RED
    overlap = top_fg & bottom_fg
    assert FG_COUNT_RANGE[0] <= int(top_fg.sum()) < FG_COUNT_RANGE[1], "top count out of range"
    assert FG_COUNT_RANGE[0] <= int(bottom_fg.sum()) < FG_COUNT_RANGE[1], "bottom count out of range"
    assert OVERLAP_RANGE[0] <= int(overlap.sum()) < OVERLAP_RANGE[1], "overlap count out of range"

    rows = np.arange(top.shape[0])[:, None]
    cols = np.arange(top.shape[1])[None, :]
    even = ((rows + cols) % 2) == 0
    odd = ~even

    both_even = overlap & even
    top_only_even = top_fg & ~bottom_fg & even
    bottom_only_even = ~top_fg & bottom_fg & even
    overlay_odd = (top_fg | bottom_fg) & odd

    assert both_even.any(), "need kept overlap cells to expose green branch"
    assert top_only_even.any(), "need kept top-only cells to expose yellow branch"
    assert bottom_only_even.any(), "need kept bottom-only cells to expose orange branch"
    assert overlay_odd.any(), "need odd-parity occupied cells so erasure matters"
    assert int((out != BLACK).sum()) >= 5, "too few kept cells after parity filtering"

    no_parity = np.zeros_like(out)
    no_parity[overlap] = GREEN
    no_parity[top_fg & ~bottom_fg] = YELLOW
    no_parity[~top_fg & bottom_fg] = ORANGE
    assert not np.array_equal(no_parity, out), "parity filter became unnecessary"

    union_even = np.where((top_fg | bottom_fg) & even, GREEN, BLACK)
    assert not np.array_equal(union_even, out), "source classification became unnecessary"


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        rng = np.random.default_rng()
        h = int(rng.integers(PANEL_H_RANGE[0], PANEL_H_RANGE[1]))
        w = int(rng.integers(PANEL_W_RANGE[0], PANEL_W_RANGE[1]))
        top, bottom = _build_masks(h, w, rng)
        grid = _assemble_grid(top, bottom)
        try:
            _validate_generated(grid)
        except AssertionError:
            continue
        return grid
    raise RuntimeError("failed to generate a valid checkerboard-filter overlay input")
