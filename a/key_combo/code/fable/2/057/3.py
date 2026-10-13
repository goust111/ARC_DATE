from __future__ import annotations

import random

import numpy as np


BLACK = 0
GREY = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
DIRS4 = ((-1, 0), (0, 1), (1, 0), (0, -1))  # up, right, down, left
GEN_ATTEMPTS = 400


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = map(int, shape)
    return 0 <= int(r) < h and 0 <= int(c) < w


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _rows_constant(block: np.ndarray) -> bool:
    return bool(block.size) and all(np.all(block[r, :] == block[r, 0]) for r in range(block.shape[0]))


def _find_divider(grid: np.ndarray) -> tuple[int, int]:
    h, w = grid.shape
    candidates: list[tuple[int, int]] = []
    for c in range(1, w - 1):
        col_color = int(grid[0, c])
        if col_color == BLACK or not np.all(grid[:, c] == col_color):
            continue
        left = grid[:, :c]
        right = grid[:, c + 1 :]
        if left.shape[1] == 0 or right.shape[1] == 0:
            continue
        if not _rows_constant(left):
            continue
        if int(np.sum(right == col_color)) != 1:
            continue
        if not np.all((right == BLACK) | (right == col_color)):
            continue
        candidates.append((c, col_color))
    assert len(candidates) == 1, "expected exactly one full-height divider column"
    return candidates[0]


def _compress_stripes(left_block: np.ndarray) -> list[tuple[int, int]]:
    assert _rows_constant(left_block), "left block must be full-width horizontal stripes"
    row_colors = [int(left_block[r, 0]) for r in range(left_block.shape[0])]
    program: list[tuple[int, int]] = []
    start = 0
    for r in range(1, len(row_colors) + 1):
        if r == len(row_colors) or row_colors[r] != row_colors[start]:
            program.append((row_colors[start], r - start))
            start = r
    colors = [color for color, _length in program]
    assert len(program) >= 2, "expected at least two stripes"
    assert len(colors) == len(set(colors)), "stripe colors must be distinct"
    return program


def _find_seed(canvas: np.ndarray, seed_color: int) -> tuple[int, int]:
    seeds = np.argwhere(canvas == int(seed_color))
    assert len(seeds) == 1, "expected exactly one seed pixel"
    return tuple(map(int, seeds[0]))


def _execute_program(
    shape: tuple[int, int],
    seed: tuple[int, int],
    program: list[tuple[int, int]],
    *,
    seed_color: int,
    start_dir_index: int = 0,
) -> np.ndarray | None:
    out = np.zeros(shape, dtype=int)
    sr, sc = map(int, seed)
    if not _in_bounds(shape, sr, sc):
        return None
    out[sr, sc] = int(seed_color)
    r, c = sr, sc
    for i, (color, length) in enumerate(program):
        dr, dc = DIRS4[(start_dir_index + i) % 4]
        for _ in range(int(length)):
            r += dr
            c += dc
            if not _in_bounds(shape, r, c):
                return None
            out[r, c] = int(color)
    out[sr, sc] = int(seed_color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Read the left stripe stack as a drawing program and execute it on the right canvas.

    Input structure:
    - One full-height grey divider column splits the grid into a left stripe block and a right canvas.
    - The left block is a stack of full-width horizontal color stripes with distinct colors and varying thicknesses.
    - The right block is a black canvas containing exactly one grey seed pixel.

    Rule:
    - Compress the left block top-to-bottom into ordered (stripe color, stripe thickness) instructions.
    - Starting at the seed, draw consecutive segments whose directions cycle up, right, down, left.
    - The i-th segment advances exactly the thickness of the i-th stripe and paints those visited cells with that stripe color.
    - Keep the seed grey and output only the right canvas.

    Key insight:
    - Stripe thickness is the program data: each horizontal band encodes both the segment color and the exact step count for one polyline move.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    divider_col, divider_color = _find_divider(g)
    left = g[:, :divider_col]
    canvas = g[:, divider_col + 1 :].copy()
    seed = _find_seed(canvas, divider_color)
    program = _compress_stripes(left)

    out = _execute_program(canvas.shape, seed, program, seed_color=divider_color)
    assert out is not None, "program must stay inside the canvas"
    return out


def _relative_segments(lengths: list[int]) -> list[list[tuple[int, int]]]:
    r = 0
    c = 0
    segments: list[list[tuple[int, int]]] = []
    for i, length in enumerate(lengths):
        dr, dc = DIRS4[i % 4]
        seg: list[tuple[int, int]] = []
        for _ in range(int(length)):
            r += dr
            c += dc
            seg.append((r, c))
        segments.append(seg)
    return segments


def _path_is_clean(segments: list[list[tuple[int, int]]]) -> bool:
    chain = [(0, 0)] + [cell for seg in segments for cell in seg]
    if len(chain) != len(set(chain)):
        return False
    index = {coord: i for i, coord in enumerate(chain)}
    for i, (r, c) in enumerate(chain):
        for nbr in _neighbors4(r, c):
            j = index.get(nbr)
            if j is None:
                continue
            if abs(i - j) != 1:
                return False
    return True


def _path_bbox(segments: list[list[tuple[int, int]]]) -> tuple[int, int, int, int]:
    coords = [(0, 0)] + [cell for seg in segments for cell in seg]
    rows = [r for r, _c in coords]
    cols = [c for _r, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _decode_program(canvas: np.ndarray, seed_color: int = GREY) -> tuple[list[tuple[int, int]], list[tuple[int, int]], tuple[int, int]]:
    seed = _find_seed(canvas, seed_color)
    colored_mask = (canvas != BLACK) & (canvas != int(seed_color))
    total = int(np.sum(colored_mask))
    assert total > 0

    first_steps = [nbr for nbr in _neighbors4(*seed) if _in_bounds(canvas.shape, *nbr) and colored_mask[nbr]]
    assert len(first_steps) == 1, "seed must touch exactly one first segment cell"

    program: list[tuple[int, int]] = []
    directions: list[tuple[int, int]] = []
    visited: set[tuple[int, int]] = set()
    prev = seed
    curr = first_steps[0]

    while True:
        color = int(canvas[curr])
        direction = (curr[0] - prev[0], curr[1] - prev[1])
        directions.append(direction)
        length = 0

        while True:
            visited.add(curr)
            length += 1

            nxts = [
                nbr
                for nbr in _neighbors4(*curr)
                if _in_bounds(canvas.shape, *nbr) and colored_mask[nbr] and nbr != prev and nbr not in visited
            ]
            assert len(nxts) <= 1, "polyline must not branch"
            if nxts and int(canvas[nxts[0]]) == color:
                nxt = nxts[0]
                assert (nxt[0] - curr[0], nxt[1] - curr[1]) == direction, "segment must stay straight"
                prev, curr = curr, nxt
                continue

            program.append((color, length))
            if not nxts:
                assert len(visited) == total, "decoded path must cover every colored pixel"
                return program, directions, seed

            nxt = nxts[0]
            prev, curr = curr, nxt
            break


def _make_lengths(n_stripes: int) -> list[int] | None:
    for _ in range(80):
        vertical_start = random.randint(1, 2)
        horizontal_start = random.randint(1, 2)
        vertical_step = random.choice((2, 3))
        horizontal_step = random.choice((2, 3))
        lengths: list[int] = []
        next_vertical = vertical_start
        next_horizontal = horizontal_start
        for i in range(n_stripes):
            if i % 2 == 0:
                lengths.append(next_vertical)
                next_vertical += vertical_step
            else:
                lengths.append(next_horizontal)
                next_horizontal += horizontal_step
        if sum(lengths) <= 28 and len(set(lengths)) >= 2:
            return lengths
    return None


def _build_case() -> tuple[np.ndarray, np.ndarray] | None:
    n_stripes = random.randint(5, 7)
    lengths = _make_lengths(n_stripes)
    if lengths is None:
        return None

    segments = _relative_segments(lengths)
    if not _path_is_clean(segments):
        return None

    colors = random.sample(PALETTE, n_stripes)
    program = list(zip(colors, lengths))

    height = sum(lengths)
    left_width = n_stripes + 1
    min_r, min_c, max_r, max_c = _path_bbox(segments)
    span_h = max_r - min_r + 1
    span_w = max_c - min_c + 1

    max_canvas_width = 30 - left_width - 1
    if max_canvas_width < span_w:
        return None
    width = random.randint(span_w, min(max_canvas_width, span_w + 6))

    top_margin = random.randint(0, height - span_h)
    left_margin = random.randint(0, width - span_w)
    seed = (top_margin - min_r, left_margin - min_c)

    out = _execute_program((height, width), seed, program, seed_color=GREY)
    if out is None:
        return None

    decoded_program, decoded_dirs, decoded_seed = _decode_program(out, GREY)
    if decoded_program != program:
        return None
    if decoded_dirs != [DIRS4[i % 4] for i in range(n_stripes)]:
        return None
    if decoded_seed != seed:
        return None

    shortcut = _execute_program((height, width), seed, [(color, 1) for color, _ in program], seed_color=GREY)
    if shortcut is not None and np.array_equal(shortcut, out):
        return None

    shifted = _execute_program((height, width), seed, program, seed_color=GREY, start_dir_index=1)
    if shifted is not None and np.array_equal(shifted, out):
        return None

    reversed_program = list(reversed(program))
    reversed_out = _execute_program((height, width), seed, reversed_program, seed_color=GREY)
    if reversed_out is not None and np.array_equal(reversed_out, out):
        return None

    inp = np.zeros((height, left_width + 1 + width), dtype=int)
    inp[:, left_width] = GREY
    row = 0
    for color, length in program:
        inp[row : row + length, :left_width] = int(color)
        row += length
    inp[seed[0], left_width + 1 + seed[1]] = GREY

    if not np.array_equal(main(inp), out):
        return None
    return inp, out


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        built = _build_case()
        if built is not None:
            inp, _out = built
            return inp
    raise RuntimeError("failed to generate a valid instruction-stripe polyline input")
