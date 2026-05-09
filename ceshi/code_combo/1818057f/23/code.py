# Combo variant of 1818057f (index 23)
# combo_spec: D5/2+D13/2
# Name: 窄条带内共线十字

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
SOLVER_DIR = ROOT / "solvers"
if str(SOLVER_DIR) not in sys.path:
    sys.path.append(str(SOLVER_DIR))

from arc_utils import get_bg


# Design analysis for the fused variant:
# - D5/2 changes placement, not the output rule: all true '+' motifs must be collinear on
#   one shared row or one shared column.
# - D13/2 changes the canvas to an extreme strip such as 5x30 or 30x5, making that
#   collinearity visually obvious.
# - Therefore `main()` should still recolor every valid local plus; the fusion lives in
#   the generator's geometry, not in a new selection rule.
# - To keep the task ARC-solvable, samples use 3-5 embedded pluses on the same line and
#   only non-plus distractors elsewhere, so there is no hidden "choose one subgroup"
#   behavior for a human to reverse-engineer.

TEAL = 8
ORTHO_DIRS = [(1, 0), (-1, 0), (0, 1), (0, -1)]
_GEN_COUNTER = 0


def _plus_cells(r: int, c: int) -> list[tuple[int, int]]:
    return [(r, c), (r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _is_plus_center(grid: np.ndarray, r: int, c: int, fg: int) -> bool:
    H, W = grid.shape
    if not (1 <= r < H - 1 and 1 <= c < W - 1):
        return False
    if int(grid[r, c]) != fg:
        return False
    return all(int(grid[r + dr, c + dc]) == fg for dr, dc in ORTHO_DIRS)


def _detect_plus_centers(grid: np.ndarray, fg: int) -> list[tuple[int, int]]:
    H, W = grid.shape
    centers: list[tuple[int, int]] = []
    for r in range(1, H - 1):
        for c in range(1, W - 1):
            if _is_plus_center(grid, r, c, fg):
                centers.append((r, c))
    return centers


def main(grid: np.ndarray) -> np.ndarray:
    """
    Detect embedded '+' motifs inside a narrow strip and recolor each one to teal.

    Input structure:
    - bg is the most common color.
    - A single foreground color occupies an extreme-aspect-ratio grid such as 5x30 or 30x5.
    - The true '+' motifs are arranged on one shared row or one shared column, so they read as a narrow 1-D chain, while other foreground cells are only distractors.

    Rule:
    - Find every foreground pixel whose 4 orthogonal neighbors are also foreground.
    - Recolor that center and its 4 orthogonal neighbors to teal (8).
    - Leave every other cell unchanged.

    Key insight: the fusion changes where pluses are allowed to appear, not how they are transformed. The strip geometry makes the collinear chain obvious, but every valid local '+' is still recolored.
    """
    g = np.array(grid, dtype=int)
    out = g.copy()
    bg = int(get_bg(g))

    colors = sorted(set(map(int, g.flatten())) - {bg})
    if not colors:
        return out
    fg = int(colors[0])

    for r, c in _detect_plus_centers(g, fg):
        for nr, nc in _plus_cells(r, c):
            out[nr, nc] = TEAL
    return out


def _stamp(grid: np.ndarray, cells: list[tuple[int, int]], color: int) -> None:
    for r, c in cells:
        grid[r, c] = color


def _all_bg(grid: np.ndarray, cells: list[tuple[int, int]], bg: int) -> bool:
    H, W = grid.shape
    for r, c in cells:
        if not (0 <= r < H and 0 <= c < W):
            return False
        if int(grid[r, c]) != bg:
            return False
    return True


def _would_create_plus(grid: np.ndarray, r: int, c: int, fg: int) -> bool:
    H, W = grid.shape
    candidates = [(r, c), (r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]
    for cr, cc in candidates:
        if cr < 1 or cr >= H - 1 or cc < 1 or cc >= W - 1:
            continue
        all_fg = True
        for dr, dc in [(0, 0), *ORTHO_DIRS]:
            nr, nc = cr + dr, cc + dc
            val = fg if (nr, nc) == (r, c) else int(grid[nr, nc])
            if val != fg:
                all_fg = False
                break
        if all_fg:
            return True
    return False


def _center_from_line(orientation: str, fixed: int, long_pos: int) -> tuple[int, int]:
    if orientation == "row":
        return fixed, long_pos
    return long_pos, fixed


def _choose_chain(
    orientation: str, shape: tuple[int, int], rng: np.random.Generator
) -> list[tuple[int, int]]:
    H, W = shape
    if orientation == "row":
        fixed = H // 2
        usable = list(range(2, W - 2))
    else:
        fixed = W // 2
        usable = list(range(2, H - 2))

    count = int(rng.integers(3, 6))
    positions: list[int] = []
    for pos in usable:
        if not positions or pos - positions[-1] >= 5:
            positions.append(pos)
        if len(positions) >= count:
            break
    if len(positions) < 3:
        raise RuntimeError("failed to place enough collinear pluses")
    return [_center_from_line(orientation, fixed, pos) for pos in positions]


def _extend_along_direction(
    grid: np.ndarray, center: tuple[int, int], direction: tuple[int, int], fg: int, length: int
) -> None:
    r, c = center
    dr, dc = direction
    cells = [(r + dr * step, c + dc * step) for step in range(2, 2 + length)]
    _stamp(grid, cells, fg)


def _available_extension(center: tuple[int, int], direction: tuple[int, int], shape: tuple[int, int]) -> int:
    H, W = shape
    r, c = center
    dr, dc = direction
    length = 0
    nr, nc = r + 2 * dr, c + 2 * dc
    while 0 <= nr < H and 0 <= nc < W:
        length += 1
        nr += dr
        nc += dc
    return length


def _add_chain_extensions(
    grid: np.ndarray, centers: list[tuple[int, int]], orientation: str, fg: int, rng: np.random.Generator
) -> None:
    for center in centers:
        primary_dirs = [(-1, 0), (1, 0)] if orientation == "row" else [(0, -1), (0, 1)]
        rng.shuffle(primary_dirs)
        used_extension = False
        for dir_idx, direction in enumerate(primary_dirs):
            max_len = _available_extension(center, direction, grid.shape)
            if max_len <= 0:
                continue
            use_this_dir = not used_extension or (dir_idx == 1 and bool(rng.integers(0, 2)))
            if not use_this_dir:
                continue
            length = int(rng.integers(1, max_len + 1))
            _extend_along_direction(grid, center, direction, fg, length)
            used_extension = True
        if not used_extension:
            raise RuntimeError(f"plus at {center} could not be embedded")


def _add_border_noise(
    grid: np.ndarray, orientation: str, fg: int, bg: int, rng: np.random.Generator
) -> None:
    H, W = grid.shape
    segments = int(rng.integers(1, 4))
    for _ in range(segments):
        if orientation == "row":
            r = 0 if bool(rng.integers(0, 2)) else H - 1
            length = int(rng.integers(1, min(4, W)))
            start = int(rng.integers(0, W - length + 1))
            cells = [(r, c) for c in range(start, start + length)]
        else:
            c = 0 if bool(rng.integers(0, 2)) else W - 1
            length = int(rng.integers(1, min(4, H)))
            start = int(rng.integers(0, H - length + 1))
            cells = [(r, c) for r in range(start, start + length)]

        if not _all_bg(grid, cells, bg):
            continue
        if any(_would_create_plus(grid, r, c, fg) for r, c in cells):
            continue
        _stamp(grid, cells, fg)


def _expected_output(inp: np.ndarray, chain_centers: list[tuple[int, int]]) -> np.ndarray:
    out = inp.copy()
    for r, c in chain_centers:
        for nr, nc in _plus_cells(r, c):
            out[nr, nc] = TEAL
    return out


def _gen_attempt(orientation: str) -> np.ndarray | None:
    rng = np.random.default_rng()
    if orientation == "row":
        H = int(rng.integers(5, 8))
        W = int(rng.integers(24, 33))
    else:
        H = int(rng.integers(24, 33))
        W = int(rng.integers(5, 8))

    bg, fg = 2, 4
    inp = np.full((H, W), bg, dtype=int)

    chain_centers = _choose_chain(orientation, inp.shape, rng)
    for center in chain_centers:
        _stamp(inp, _plus_cells(*center), fg)

    _add_chain_extensions(inp, chain_centers, orientation, fg, rng)
    _add_border_noise(inp, orientation, fg, bg, rng)

    centers = _detect_plus_centers(inp, fg)
    if set(centers) != set(chain_centers):
        return None
    if orientation == "row" and len({r for r, _ in centers}) != 1:
        return None
    if orientation == "col" and len({c for _, c in centers}) != 1:
        return None

    expected = _expected_output(inp, chain_centers)
    actual = main(inp)
    if not np.array_equal(actual, expected):
        return None
    if not np.any(actual == TEAL):
        return None
    return inp


def generate_input() -> np.ndarray:
    global _GEN_COUNTER
    orientation = "row" if _GEN_COUNTER % 2 == 0 else "col"
    _GEN_COUNTER += 1
    for _ in range(300):
        result = _gen_attempt(orientation)
        if result is not None:
            return result
    raise RuntimeError("generate_input failed after 300 attempts")
