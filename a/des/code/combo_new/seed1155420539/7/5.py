import random

import numpy as np


BG = 0
GRID = 5
PATTERN_COLORS = (1, 4, 6, 7)
SIGNAL_COLORS = (2, 3, 8, 9)
INTERIOR_SIZES = (3, 4, 5)
MAX_GENERATION_ATTEMPTS = 400
PATTERN_ATTEMPTS = 300


def _divisors(n: int) -> list[int]:
    return [d for d in range(2, n + 1) if n % d == 0]


def _neighbors4(r: int, c: int, size: int) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        nr, nc = r + dr, c + dc
        if 0 <= nr < size and 0 <= nc < size:
            out.append((nr, nc))
    return out


def _is_rotation_asymmetric(mask: np.ndarray) -> bool:
    return all(not np.array_equal(mask, np.rot90(mask, k)) for k in (1, 2, 3))


def _random_pattern(size: int) -> np.ndarray:
    for _ in range(PATTERN_ATTEMPTS):
        target = random.randint(max(3, size - 1), min(size * size - 2, size + 4))
        cells = {(random.randrange(size), random.randrange(size))}
        while len(cells) < target:
            frontier = {
                nbr
                for r, c in cells
                for nbr in _neighbors4(r, c, size)
                if nbr not in cells
            }
            if not frontier:
                break
            cells.add(random.choice(tuple(frontier)))

        if len(cells) != target:
            continue

        mask = np.zeros((size, size), dtype=bool)
        for r, c in cells:
            mask[r, c] = True

        if _is_rotation_asymmetric(mask):
            return mask

    raise AssertionError("failed to build a non-symmetric pattern")


def _find_lattice(grid: np.ndarray) -> tuple[int, int, list[int], list[int]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    common = int(np.gcd(h - 1, w - 1))

    for step in _divisors(common):
        if step < 3:
            continue

        n_rows = (h - 1) // step
        n_cols = (w - 1) // step
        if n_rows < 2 or n_cols < 2:
            continue

        line_rows = list(range(0, h, step))
        line_cols = list(range(0, w, step))
        if line_rows[-1] != h - 1 or line_cols[-1] != w - 1:
            continue

        row_mask = np.zeros(h, dtype=bool)
        col_mask = np.zeros(w, dtype=bool)
        row_mask[line_rows] = True
        col_mask[line_cols] = True

        line_mask = row_mask[:, None] | col_mask[None, :]
        grey = int(np.bincount(g[line_mask]).argmax())

        strict_line_mask = row_mask[:, None] ^ col_mask[None, :]
        if np.any(g[strict_line_mask] != grey):
            continue

        valid = True
        for r0, r1 in zip(line_rows, line_rows[1:]):
            for c0, c1 in zip(line_cols, line_cols[1:]):
                interior = g[r0 + 1 : r1, c0 + 1 : c1]
                if interior.shape != (step - 1, step - 1):
                    valid = False
                    break
                if np.any(interior == grey):
                    valid = False
                    break
                corners = [
                    int(g[r0, c0]),
                    int(g[r0, c1]),
                    int(g[r1, c1]),
                    int(g[r1, c0]),
                ]
                if sum(value != grey for value in corners) != 1:
                    valid = False
                    break
            if not valid:
                break

        if valid:
            return step, grey, line_rows, line_cols

    raise AssertionError("failed to detect the grey lattice")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate each cell's interior pattern according to which border corner contains the signal pixel.

    Input structure:
    - The background is black.
    - Grey one-pixel lines form a full rectangular lattice that splits the grid into equal square subregions.
    - Each subregion interior contains a non-symmetric monochrome pattern on black background.
    - Among the four border-corner pixels of each subregion, exactly one corner is replaced by a non-grey colored signal pixel.

    Rule:
    - Detect the shared grey lattice and treat every square between consecutive lattice rows and columns as one cell.
    - Read the unique non-grey corner of each cell: top-left means keep the interior unchanged, top-right means rotate it 90 degrees clockwise, bottom-right means rotate it 180 degrees, and bottom-left means rotate it 270 degrees clockwise.
    - Leave the grey borders and signal pixels untouched, and replace only each cell interior with its rotated version.

    Key insight:
    - The signal color itself is irrelevant; only the signal pixel's corner position inside the grey frame encodes the rotation.
    """
    g = np.asarray(grid, dtype=int)
    step, grey, line_rows, line_cols = _find_lattice(g)
    _ = step  # keeps the lattice size explicit for readability

    rotation_by_corner = {
        0: 0,  # top-left
        1: 3,  # top-right -> 90 clockwise
        2: 2,  # bottom-right -> 180
        3: 1,  # bottom-left -> 270 clockwise
    }

    out = g.copy()
    for r0, r1 in zip(line_rows, line_rows[1:]):
        for c0, c1 in zip(line_cols, line_cols[1:]):
            interior = g[r0 + 1 : r1, c0 + 1 : c1]
            corners = [
                int(g[r0, c0]),
                int(g[r0, c1]),
                int(g[r1, c1]),
                int(g[r1, c0]),
            ]
            signal_index = next(idx for idx, value in enumerate(corners) if value != grey)
            out[r0 + 1 : r1, c0 + 1 : c1] = np.rot90(interior, rotation_by_corner[signal_index])
    return out


def _build_grid() -> np.ndarray:
    inner = random.choice(INTERIOR_SIZES)
    step = inner + 1
    max_cells = min(5, (30 - 1) // step)
    if max_cells < 2:
        raise AssertionError("grid step leaves no room for multiple cells")

    n_rows = random.randint(2, max_cells)
    n_cols = random.randint(2, max_cells)
    height = n_rows * step + 1
    width = n_cols * step + 1

    grid = np.full((height, width), BG, dtype=int)
    grid[::step, :] = GRID
    grid[:, ::step] = GRID

    row_phase = random.randint(0, 1)
    col_phase = random.randint(0, 1)
    for ir in range(n_rows + 1):
        for ic in range(n_cols + 1):
            if ir % 2 == row_phase and ic % 2 == col_phase:
                grid[ir * step, ic * step] = random.choice(SIGNAL_COLORS)

    for ir in range(n_rows):
        for ic in range(n_cols):
            mask = _random_pattern(inner)
            color = random.choice(PATTERN_COLORS)
            patch = np.full((inner, inner), BG, dtype=int)
            patch[mask] = color
            r0 = ir * step + 1
            c0 = ic * step + 1
            grid[r0 : r0 + inner, c0 : c0 + inner] = patch

    return grid


def _validate_signal_layout(grid: np.ndarray) -> bool:
    step, grey, line_rows, line_cols = _find_lattice(grid)
    if step < 3:
        return False

    seen = set()
    for r0, r1 in zip(line_rows, line_rows[1:]):
        for c0, c1 in zip(line_cols, line_cols[1:]):
            corners = [
                int(grid[r0, c0]),
                int(grid[r0, c1]),
                int(grid[r1, c1]),
                int(grid[r1, c0]),
            ]
            idx = next(k for k, value in enumerate(corners) if value != grey)
            seen.add(idx)
    return seen == {0, 1, 2, 3}


def _gen_attempt() -> np.ndarray | None:
    grid = _build_grid()
    if not _validate_signal_layout(grid):
        return None

    out = main(grid)
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise AssertionError("failed to generate a valid puzzle instance")
