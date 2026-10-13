from __future__ import annotations

import random

import numpy as np


BACKGROUND = 0
SEPARATOR = 7
PAYLOAD_COLORS = (1, 2, 3, 4, 6, 8, 9)
GEN_ATTEMPTS = 500


def _rot90_orbit(side: int, r: int, c: int) -> tuple[tuple[int, int], ...]:
    last = side - 1
    return (
        (r, c),
        (c, last - r),
        (last - r, last - c),
        (last - c, r),
    )


def _find_separator_layout(grid: np.ndarray) -> tuple[int, int, int, np.ndarray, np.ndarray]:
    h, w = grid.shape
    candidates: list[tuple[int, int, int, np.ndarray, np.ndarray]] = []

    for color in sorted(set(map(int, grid.flatten()))):
        row_lines = np.flatnonzero(np.all(grid == color, axis=1))
        col_lines = np.flatnonzero(np.all(grid == color, axis=0))
        if row_lines.size == 0 or col_lines.size == 0:
            continue

        row_steps = np.diff(np.concatenate(([-1], row_lines, [h])))
        col_steps = np.diff(np.concatenate(([-1], col_lines, [w])))
        if len(set(map(int, row_steps))) != 1 or len(set(map(int, col_steps))) != 1:
            continue

        row_step = int(row_steps[0])
        col_step = int(col_steps[0])
        cell_h = row_step - 1
        cell_w = col_step - 1
        if cell_h < 1 or cell_w < 1:
            continue

        n_rows = len(row_lines) + 1
        n_cols = len(col_lines) + 1
        compact = np.zeros((n_rows, n_cols), dtype=int)
        ok = True
        for rr in range(n_rows):
            r0 = rr * row_step
            for cc in range(n_cols):
                c0 = cc * col_step
                block = grid[r0 : r0 + cell_h, c0 : c0 + cell_w]
                if block.shape != (cell_h, cell_w):
                    ok = False
                    break
                if not np.all(block == block[0, 0]):
                    ok = False
                    break
                compact[rr, cc] = int(block[0, 0])
            if not ok:
                break
        if ok:
            candidates.append((int(color), cell_h, cell_w, row_lines, col_lines))

    assert len(candidates) == 1, "expected a unique separator color/layout"
    return candidates[0]


def _compress_regions(grid: np.ndarray) -> np.ndarray:
    separator, cell_h, cell_w, row_lines, col_lines = _find_separator_layout(grid)
    _ = separator, row_lines, col_lines

    h, w = grid.shape
    row_step = cell_h + 1
    col_step = cell_w + 1
    n_rows = (h + 1) // row_step
    n_cols = (w + 1) // col_step
    compact = np.zeros((n_rows, n_cols), dtype=int)

    for rr in range(n_rows):
        r0 = rr * row_step
        for cc in range(n_cols):
            c0 = cc * col_step
            compact[rr, cc] = int(grid[r0, c0])
    return compact


def _mirror_to_square(compact: np.ndarray) -> np.ndarray:
    rows, cols = compact.shape
    if rows < cols:
        assert cols == 2 * rows, "expected width to be twice height before vertical mirroring"
        return np.concatenate((compact, np.flipud(compact)), axis=0)
    if rows > cols:
        assert rows == 2 * cols, "expected height to be twice width before horizontal mirroring"
        return np.concatenate((compact, np.fliplr(compact)), axis=1)
    return compact.copy()


def _rotation_complete(square: np.ndarray) -> np.ndarray:
    side = square.shape[0]
    out = np.zeros_like(square)
    owner: dict[tuple[int, int], tuple[int, int]] = {}
    color_of: dict[tuple[int, int], int] = {}

    for src in sorted(map(tuple, np.argwhere(square != BACKGROUND))):
        color = int(square[src])
        for pos in _rot90_orbit(side, src[0], src[1]):
            prev = owner.get(pos)
            if prev is None or src < prev:
                owner[pos] = src
                color_of[pos] = color

    for pos, color in color_of.items():
        out[pos] = int(color)
    return out


def _orbit_infos(square: np.ndarray) -> list[dict]:
    side = square.shape[0]
    seen: set[tuple[tuple[int, int], ...]] = set()
    infos: list[dict] = []
    for r in range(side):
        for c in range(side):
            orbit = tuple(sorted(set(_rot90_orbit(side, r, c))))
            if orbit in seen:
                continue
            seen.add(orbit)
            fg_positions = [pos for pos in orbit if int(square[pos]) != BACKGROUND]
            colors = {int(square[pos]) for pos in fg_positions}
            infos.append(
                {
                    "orbit": orbit,
                    "size": len(orbit),
                    "fg_positions": fg_positions,
                    "colors": colors,
                }
            )
    return infos


def main(grid: np.ndarray) -> np.ndarray:
    """
    Compress a separator-partitioned grid, mirror it to a square if needed, then complete 90-degree rotational symmetry.

    Input structure:
    - Orange separator lines run across full rows and full columns, partitioning the canvas into equal rectangular regions.
    - Every separator-delimited region is uniformly colored; black is background and other colors are foreground.
    - The compressed region map is rectangular or square, with the rectangular cases sized so one side is exactly twice the other.

    Rule:
    - Identify the separator color from the full monochrome rows and columns, then compress each uniform region to one pixel.
    - If the compressed map is wider than tall, append its upside-down copy below; if it is taller than wide, append its left-right mirror on the right; if it is already square, keep it unchanged.
    - On the resulting square map, rotate every non-background pixel through 90, 180, and 270 degrees about the square center and paint those positions too.
    - When two source pixels would assign different non-background colors to the same rotated position, keep the color from the lexicographically smallest source coordinate `(r, c)`.

    Key insight:
    - The symmetry operation does not act on the raw pixels of the large grid: the separator grid must first be abstracted into a coarse region map, and rectangular maps must be mirrored into a square before quarter-turn closure makes sense.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    compact = _compress_regions(g)
    square = _mirror_to_square(compact)
    assert square.shape[0] == square.shape[1], "expected a square compact map before rotation completion"
    return _rotation_complete(square)


def _sample_branch() -> tuple[str, tuple[int, int]]:
    branch = random.choice(("wide", "tall", "square"))
    if branch == "wide":
        side = random.choice((4, 6))
        return branch, (side // 2, side)
    if branch == "tall":
        side = random.choice((4, 6))
        return branch, (side, side // 2)
    side = random.choice((3, 4, 5, 6))
    return branch, (side, side)


def _sample_compact(shape: tuple[int, int], conflict: bool) -> np.ndarray:
    rows, cols = shape
    palette_size = random.randint(2, 4) if conflict else 1
    colors = random.sample(list(PAYLOAD_COLORS), palette_size)

    density = random.uniform(0.28, 0.58)
    compact = np.zeros((rows, cols), dtype=int)
    for r in range(rows):
        for c in range(cols):
            if random.random() < density:
                compact[r, c] = random.choice(colors)

    if np.count_nonzero(compact) < 2:
        picks = random.sample([(r, c) for r in range(rows) for c in range(cols)], 2)
        for pos in picks:
            compact[pos] = random.choice(colors)

    if conflict:
        # Encourage true color conflicts after rotation closure.
        square = _mirror_to_square(compact)
        infos = [info for info in _orbit_infos(square) if len(info["orbit"]) > 1]
        if infos:
            orbit = random.choice(infos)["orbit"]
            orbit_colors = random.sample(list(PAYLOAD_COLORS), 2)
            for idx, pos in enumerate(orbit[:2]):
                rr, cc = pos
                if rows < cols:
                    src = (rr if rr < rows else square.shape[0] - 1 - rr, cc)
                elif rows > cols:
                    src = (rr, cc if cc < cols else square.shape[1] - 1 - cc)
                else:
                    src = (rr, cc)
                compact[src] = orbit_colors[idx]
    return compact


def _expand_compact(compact: np.ndarray, cell_h: int, cell_w: int) -> np.ndarray:
    rows, cols = compact.shape
    h = rows * cell_h + (rows - 1)
    w = cols * cell_w + (cols - 1)
    grid = np.full((h, w), SEPARATOR, dtype=int)
    row_step = cell_h + 1
    col_step = cell_w + 1
    for rr in range(rows):
        r0 = rr * row_step
        for cc in range(cols):
            c0 = cc * col_step
            grid[r0 : r0 + cell_h, c0 : c0 + cell_w] = int(compact[rr, cc])
    return grid


def _count_conflict_orbits(square: np.ndarray) -> int:
    return sum(1 for info in _orbit_infos(square) if len(info["colors"]) > 1)


def _semantic_checks(grid: np.ndarray, compact: np.ndarray, want_conflict: bool) -> bool:
    try:
        separator, cell_h, cell_w, row_lines, col_lines = _find_separator_layout(grid)
    except AssertionError:
        return False
    if separator != SEPARATOR:
        return False
    if cell_h not in (3, 4) or cell_w not in (3, 4):
        return False
    if row_lines.size == 0 or col_lines.size == 0:
        return False

    recovered = _compress_regions(grid)
    if not np.array_equal(recovered, compact):
        return False

    square = _mirror_to_square(compact)
    if square.shape[0] != square.shape[1]:
        return False

    completed = _rotation_complete(square)
    if np.array_equal(square, completed):
        return False

    infos = _orbit_infos(square)
    nonempty = [info for info in infos if info["fg_positions"]]
    if len(nonempty) < 2:
        return False
    if sum(1 for info in nonempty if len(info["fg_positions"]) < len(info["orbit"])) == 0 and _count_conflict_orbits(square) == 0:
        return False

    conflict_count = _count_conflict_orbits(square)
    if want_conflict and conflict_count == 0:
        return False
    if not want_conflict and conflict_count != 0:
        return False

    solved = main(grid)
    if not np.array_equal(solved, completed):
        return False

    if max(grid.shape) > 30 or max(solved.shape) > 30:
        return False
    return True


def _attempt_generate() -> np.ndarray | None:
    _, shape = _sample_branch()
    want_conflict = random.random() < 0.5
    compact = _sample_compact(shape, want_conflict)

    cell_h = random.choice((3, 4))
    cell_w = random.choice((3, 4))
    grid = _expand_compact(compact, cell_h, cell_w)

    if not _semantic_checks(grid, compact, want_conflict):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
