import numpy as np


BG = 0
MOVE_COLOR = 1
DEFAULT_COLLISION_COLOR = 2
GEN_RETRY_LIMIT = 400


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(values[int(np.argmax(counts))])


def _non_bg_colors(grid: np.ndarray, bg: int) -> list[int]:
    return [int(v) for v in np.unique(np.asarray(grid, dtype=int)) if int(v) != int(bg)]


def _parse_segment(grid: np.ndarray) -> tuple[np.ndarray, int, int, int, int, int, int, int]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = map(int, g.shape)
    bg = _most_common_color(g)
    fg_colors = _non_bg_colors(g, bg)
    assert len(fg_colors) == 1, "expected exactly one foreground color"
    fg = int(fg_colors[0])

    coords = np.argwhere(g == fg)
    assert len(coords) > 0, "expected at least one foreground pixel"
    diag_sums = {int(r + c) for r, c in coords}
    assert len(diag_sums) == 1, "foreground must lie on one anti-diagonal"
    diag_sum = next(iter(diag_sums))

    cols = sorted(int(c) for _r, c in coords)
    assert cols == list(range(cols[0], cols[-1] + 1)), "segment columns must be consecutive"
    expected = {(int(diag_sum - c), int(c)) for c in cols}
    actual = {tuple(map(int, rc)) for rc in coords}
    assert expected == actual, "foreground must be a contiguous anti-diagonal segment"

    c0 = int(cols[0])
    c1 = int(cols[-1])
    length = int(len(cols))
    return g, h, w, bg, fg, int(diag_sum), c0, c1, length


def _collision_color(bg: int, move_color: int) -> int:
    for candidate in (DEFAULT_COLLISION_COLOR, 3, 4, 5, 6, 7, 8, 9, 1):
        if candidate not in {int(bg), int(move_color)}:
            return int(candidate)
    raise AssertionError("no spare collision color available")


def _advanced_half_masks(h: int, w: int, diag_sum: int, c1: int, length: int) -> tuple[np.ndarray, np.ndarray]:
    cols = np.arange(c1 + 1, c1 + length + 2, dtype=int)
    assert cols[-1] < w, "advanced segment exceeds grid width"

    top_rows = diag_sum - cols
    bottom_rows = (h - 1 - diag_sum) + cols
    assert np.all((0 <= top_rows) & (top_rows < h)), "top advanced segment exceeds grid height"
    assert np.all((0 <= bottom_rows) & (bottom_rows < h)), "bottom advanced segment exceeds grid height"

    top = np.zeros((h, w), dtype=bool)
    bottom = np.zeros((h, w), dtype=bool)
    top[top_rows, cols] = True
    bottom[bottom_rows, cols] = True
    return top, bottom


def main(grid: np.ndarray) -> np.ndarray:
    """Mirror the anti-diagonal segment downward, advance both halves, then mark aligned collisions.

    Input structure:
    - The grid has a dominant background and exactly one foreground segment.
    - The foreground is a contiguous one-pixel-wide run on a single anti-diagonal.
    - The source segment sits away from the top and bottom borders so the mirrored-and-advanced
      construction fits cleanly.

    Rule:
    - Double the canvas height by placing a top-bottom mirror copy of the input underneath the
      original half.
    - Erase the source segment in each half and draw the next longer continuation immediately past
      the segment's active end: in the top half this is the usual up-right anti-diagonal advance,
      while in the mirrored lower half the same local move appears as a down-right diagonal.
    - Compare the two advanced halves cell-by-cell in aligned half coordinates. Any aligned pair of
      occupied cells is a collision and is recolored red in both halves; other moved pixels keep the
      original segment color.

    Key insight:
    - The puzzle is a three-step composition: vertical mirror stacking creates a paired segment,
      diagonal advancement changes both positions, and the final red pixels come from transparent
      overlay of the advanced top and bottom halves rather than from the source segment itself.
    """
    g, h, w, bg, fg, diag_sum, _c0, c1, length = _parse_segment(grid)
    move_color = int(fg)
    hit_color = _collision_color(bg, move_color)
    top_mask, bottom_rel_mask = _advanced_half_masks(h, w, diag_sum, c1, length)
    collision_mask = top_mask & bottom_rel_mask

    out = np.full((2 * h, w), int(bg), dtype=int)
    out[:h][top_mask] = move_color
    out[h:][bottom_rel_mask] = move_color
    out[:h][collision_mask] = hit_color
    out[h:][collision_mask] = hit_color
    return out


def _valid_input_rows(h: int, diag_sum: int, c0: int, c1: int) -> bool:
    return 1 <= diag_sum - c1 and diag_sum - c0 <= h - 2


def _feasible_parameter_sets() -> dict[int, list[tuple[int, int, int, int, int]]]:
    by_length: dict[int, list[tuple[int, int, int, int, int]]] = {2: [], 3: [], 4: []}
    for h in range(7, 16, 2):
        if 2 * h > 30:
            continue
        for w in range(8, 15):
            for length in range(2, 5):
                max_c0 = w - 1 - 2 * length
                if max_c0 < 0:
                    continue
                for c0 in range(max_c0 + 1):
                    c1 = c0 + length - 1
                    for diag_sum in range(h + w - 1):
                        if not _valid_input_rows(h, diag_sum, c0, c1):
                            continue
                        try:
                            top_mask, bottom_mask = _advanced_half_masks(h, w, diag_sum, c1, length)
                        except AssertionError:
                            continue
                        collision = top_mask & bottom_mask
                        if int(collision.sum()) < 1:
                            continue
                        by_length[length].append((h, w, length, c0, diag_sum))
    return by_length


_FEASIBLE_BY_LENGTH = _feasible_parameter_sets()


def _build_input(h: int, w: int, length: int, c0: int, diag_sum: int) -> np.ndarray:
    grid = np.full((h, w), BG, dtype=int)
    cols = np.arange(c0, c0 + length, dtype=int)
    rows = diag_sum - cols
    grid[rows, cols] = MOVE_COLOR
    return grid


def _validate_semantics(inp: np.ndarray, out: np.ndarray) -> None:
    g, h, w, bg, fg, diag_sum, c0, c1, length = _parse_segment(inp)
    assert 2 <= length <= 4, "generated length must stay in the requested range"
    assert h <= 15 and w <= 30 and 2 * h <= 30, "generated grids must respect ARC size limits"
    assert _valid_input_rows(h, diag_sum, c0, c1), "input segment must stay off the top/bottom borders"
    top_mask, bottom_mask = _advanced_half_masks(h, w, diag_sum, c1, length)
    collision_mask = top_mask & bottom_mask
    assert int(collision_mask.sum()) >= 1, "generator must include at least one aligned collision"

    hit_color = _collision_color(bg, fg)
    assert out.shape == (2 * h, w), "output height must double"
    assert np.array_equal(out, main(g)), "output must match the solver"

    top = out[:h]
    bottom = out[h:]
    top_hits = top == hit_color
    bottom_hits = bottom == hit_color
    top_moves = top == fg
    bottom_moves = bottom == fg

    assert np.array_equal(top_hits, collision_mask), "top-half collisions must match the overlay mask"
    assert np.array_equal(bottom_hits, collision_mask), "bottom-half collisions must match the overlay mask"
    assert np.array_equal(top_moves, top_mask & ~collision_mask), "top-half blue pixels must be non-colliding moves"
    assert np.array_equal(bottom_moves, bottom_mask & ~collision_mask), "bottom-half blue pixels must be non-colliding moves"

    input_signature = (h, w, length, c0, diag_sum)
    output_signature = (tuple(map(int, np.argwhere(top_hits)[0])) if int(top_hits.sum()) == 1 else int(top_hits.sum()))
    assert input_signature != output_signature, "the output should not collapse to a trivial fixed signature"


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    lengths = np.array([2, 3, 4], dtype=int)
    for _ in range(GEN_RETRY_LIMIT):
        length = int(rng.choice(lengths))
        params = _FEASIBLE_BY_LENGTH[length]
        h, w, length, c0, diag_sum = params[int(rng.integers(len(params)))]
        grid = _build_input(h, w, length, c0, diag_sum)
        out = main(grid)
        _validate_semantics(grid, out)
        return grid
    raise RuntimeError("failed to generate a valid input")
