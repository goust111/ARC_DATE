import itertools
import random

import numpy as np


BACKGROUND = 0
BLOCK_COLOR = 1
FILL_COLOR = 2

GRID_HEIGHT_RANGE = (13, 18)
GRID_WIDTH_RANGE = (13, 18)
BLOCK_COUNT_OPTIONS = (2, 3)
BLOCK_MARGIN = 2
MIN_CENTER_GAP = 4
GEN_ATTEMPTS = 400


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _colors_present(grid: np.ndarray, exclude: set[int] | None = None) -> list[int]:
    colors = set(map(int, np.unique(grid)))
    if exclude is not None:
        colors -= set(exclude)
    return sorted(colors)


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    height, width = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(height):
        for c in range(width):
            if seen[r, c] or not bool(mask[r, c]):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < height and 0 <= nc < width and bool(mask[nr, nc]) and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(coords)
    return comps


def _find_blocks(grid: np.ndarray) -> tuple[list[tuple[int, int]], int, int]:
    bg = _get_bg(grid)
    fg_colors = _colors_present(grid, exclude={bg})
    assert len(fg_colors) == 1, "expected exactly one foreground color"
    color = fg_colors[0]

    blocks: list[tuple[int, int]] = []
    for coords in _components(grid == color):
        arr = np.array(coords, dtype=int)
        r0, c0 = arr.min(axis=0)
        r1, c1 = arr.max(axis=0)
        assert len(coords) == 9, "expected each component to be a solid 3x3 block"
        assert r1 - r0 == 2 and c1 - c0 == 2, "expected each component bbox to be 3x3"
        assert np.all(grid[r0 : r1 + 1, c0 : c1 + 1] == color), "expected solid 3x3 blocks"
        blocks.append((int(r0 + 1), int(c0 + 1)))

    blocks.sort()
    assert 2 <= len(blocks) <= 3, "expected two or three blocks"
    return blocks, color, bg


def _build_cross_grid(
    shape: tuple[int, int],
    blocks: list[tuple[int, int]],
    color: int,
    bg: int,
    *,
    reset_crossings: bool,
) -> np.ndarray:
    out = np.full(shape, bg, dtype=int)
    for r, c in blocks:
        out[r - 1 : r + 2, c - 1 : c + 2] = color
    for r, c in blocks:
        out[r, :] = color
        out[:, c] = color
    if reset_crossings:
        for i, (r, _) in enumerate(blocks):
            for j, (_, c) in enumerate(blocks):
                if i != j:
                    out[r, c] = bg
    return out


def _scan_fill(cross_grid: np.ndarray, color: int, bg: int) -> np.ndarray:
    out = cross_grid.copy()
    height, width = cross_grid.shape
    for r in range(height):
        inside = False
        for c in range(width):
            if int(cross_grid[r, c]) == color and r + 1 < height and int(cross_grid[r + 1, c]) == color:
                inside = not inside
            elif int(cross_grid[r, c]) == bg and inside:
                out[r, c] = FILL_COLOR
    return out


def _expected_output_from_blocks(
    shape: tuple[int, int],
    blocks: list[tuple[int, int]],
    *,
    reset_crossings: bool,
) -> np.ndarray:
    cross = _build_cross_grid(shape, blocks, BLOCK_COLOR, BACKGROUND, reset_crossings=reset_crossings)
    return _scan_fill(cross, BLOCK_COLOR, BACKGROUND)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend each 3x3 blue block into a gapped cross network and fill scanline interiors created by the gaps.

    Input structure:
    - The grid has a black background and contains only two or three blue solid 3x3 blocks.
    - Each block lies away from the border, and block centers have distinct rows and distinct columns.
    - Center rows and center columns are separated enough that cross intersections occur outside the source blocks.

    Rule:
    - Detect the blue 3x3 blocks and copy them to the output.
    - For each block, paint its center row across the full width and its center column across the full height in blue.
    - At every crossing between the center row of one block and the center column of a different block, reset that pixel to black.
    - Then scan each row left to right with an inside/outside toggle; flip the state only when the current blue pixel also has a blue pixel directly below it, and recolor background pixels seen while inside to red.

    Key insight:
    - The reset crossings punch single-cell gaps into vertical arm segments, so the scanline toggle parity changes relative to an uninterrupted set of crosses, producing emergent red spans.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    blocks, color, bg = _find_blocks(g)
    rows = [r for r, _ in blocks]
    cols = [c for _, c in blocks]
    assert len(rows) == len(set(rows)), "expected unique center rows"
    assert len(cols) == len(set(cols)), "expected unique center columns"
    assert all(abs(a - b) >= MIN_CENTER_GAP for a, b in itertools.combinations(rows, 2))
    assert all(abs(a - b) >= MIN_CENTER_GAP for a, b in itertools.combinations(cols, 2))

    cross = _build_cross_grid(g.shape, blocks, color, bg, reset_crossings=True)
    return _scan_fill(cross, color, bg)


def _choose_centers(length: int, count: int) -> list[int] | None:
    candidates = list(range(BLOCK_MARGIN, length - BLOCK_MARGIN))
    valid = [
        list(combo)
        for combo in itertools.combinations(candidates, count)
        if all(abs(a - b) >= MIN_CENTER_GAP for a, b in itertools.combinations(combo, 2))
    ]
    if not valid:
        return None
    chosen = list(random.choice(valid))
    random.shuffle(chosen)
    return chosen


def _render_input(shape: tuple[int, int], blocks: list[tuple[int, int]]) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    for r, c in blocks:
        grid[r - 1 : r + 2, c - 1 : c + 2] = BLOCK_COLOR
    return grid


def _is_interesting(inp: np.ndarray, out: np.ndarray, blocks: list[tuple[int, int]]) -> bool:
    red_count = int(np.count_nonzero(out == FILL_COLOR))
    if red_count == 0:
        return False

    fill_fraction = red_count / float(inp.size)
    if fill_fraction > 0.4:
        return False

    no_reset = _expected_output_from_blocks(inp.shape, blocks, reset_crossings=False)
    if np.array_equal(out, no_reset):
        return False

    changed_rows = sum(bool(np.any(out[r] != no_reset[r])) for r in range(out.shape[0]))
    return changed_rows >= 1


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(*GRID_HEIGHT_RANGE)
    width = random.randint(*GRID_WIDTH_RANGE)
    count = random.choice(BLOCK_COUNT_OPTIONS)

    rows = _choose_centers(height, count)
    cols = _choose_centers(width, count)
    if rows is None or cols is None:
        return None

    random.shuffle(cols)
    blocks = [(int(r), int(c)) for r, c in zip(rows, cols)]
    grid = _render_input((height, width), blocks)

    expected = _expected_output_from_blocks(grid.shape, blocks, reset_crossings=True)
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    if not _is_interesting(grid, actual, blocks):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
