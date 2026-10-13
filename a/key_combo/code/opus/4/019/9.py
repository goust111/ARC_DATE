from __future__ import annotations

import importlib.util
import random
from pathlib import Path

import numpy as np


def _load_arc_utils():
    path = Path("/home/lixinyu/snap/new/solvers/arc_utils.py")
    if not path.exists():
        return None
    try:
        spec = importlib.util.spec_from_file_location("_shared_arc_utils", path)
        if spec is None or spec.loader is None:
            return None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    except Exception:
        return None


_ARC_UTILS = _load_arc_utils()


if _ARC_UTILS is not None:
    Color = _ARC_UTILS.Color
else:
    class Color:
        BLACK = 0
        BLUE = 1
        RED = 2
        GREEN = 3
        YELLOW = 4
        ORANGE = 7
        TEAL = 8


BG = int(Color.BLACK)
BLUE = int(Color.BLUE)
RED = int(Color.RED)
GREEN = int(Color.GREEN)
YELLOW = int(Color.YELLOW)
ORANGE = int(Color.ORANGE)
TEAL = int(Color.TEAL)

OUT_SIZE = 15
GEN_MAX_ATTEMPTS = 500


def _top_left_square(grid: np.ndarray) -> tuple[int, int]:
    g = np.asarray(grid, dtype=int)
    color = int(g[0, 0])

    side_row = 0
    while side_row < g.shape[1] and int(g[0, side_row]) == color:
        side_row += 1

    side_col = 0
    while side_col < g.shape[0] and int(g[side_col, 0]) == color:
        side_col += 1

    side = min(side_row, side_col)
    assert side >= 1
    assert np.all(g[:side, :side] == color)
    if side < g.shape[1]:
        assert int(g[0, side]) != color
    if side < g.shape[0]:
        assert int(g[side, 0]) != color
    return side, color


def _map_index(idx: int, side: int) -> int:
    return int(idx) * (int(side) + 1) + int(side) // 2


def _grid_line_positions(side: int) -> list[int]:
    return list(range(int(side), OUT_SIZE, int(side) + 1))


def _collect_markers(grid: np.ndarray, side: int) -> list[tuple[int, int, int]]:
    g = np.asarray(grid, dtype=int)
    markers: list[tuple[int, int, int]] = []
    for r in range(g.shape[0]):
        for c in range(g.shape[1]):
            if r < side and c < side:
                continue
            color = int(g[r, c])
            if color in (RED, BLUE):
                markers.append((r, c, color))
            else:
                assert color == BG, "expected only background outside the square and markers"
    return markers


def _build_base_output(markers: list[tuple[int, int, int]], side: int, line_color: int) -> np.ndarray:
    out = np.full((OUT_SIZE, OUT_SIZE), BG, dtype=int)
    for pos in _grid_line_positions(side):
        out[pos, :] = int(line_color)
        out[:, pos] = int(line_color)

    for r, c, color in markers:
        rr = _map_index(r, side)
        cc = _map_index(c, side)
        assert 0 <= rr < OUT_SIZE and 0 <= cc < OUT_SIZE
        out[rr, cc] = int(color)
    return out


def _path_masks(base: np.ndarray, markers: list[tuple[int, int, int]], side: int) -> tuple[np.ndarray, np.ndarray]:
    hmask = np.zeros_like(base, dtype=bool)
    vmask = np.zeros_like(base, dtype=bool)

    red_by_row: dict[int, list[int]] = {}
    blue_by_col: dict[int, list[int]] = {}
    for r, c, color in markers:
        if color == RED:
            red_by_row.setdefault(int(r), []).append(int(c))
        elif color == BLUE:
            blue_by_col.setdefault(int(c), []).append(int(r))

    for r, cols in red_by_row.items():
        assert len(cols) <= 2, "each input row may contain at most two red markers"
        if len(cols) != 2:
            continue
        c0, c1 = sorted(cols)
        rr = _map_index(r, side)
        cc0 = _map_index(c0, side)
        cc1 = _map_index(c1, side)
        segment = base[rr, cc0 + 1 : cc1] == BG
        hmask[rr, cc0 + 1 : cc1] = segment

    for c, rows in blue_by_col.items():
        assert len(rows) <= 2, "each input column may contain at most two blue markers"
        if len(rows) != 2:
            continue
        r0, r1 = sorted(rows)
        cc = _map_index(c, side)
        rr0 = _map_index(r0, side)
        rr1 = _map_index(r1, side)
        segment = base[rr0 + 1 : rr1, cc] == BG
        vmask[rr0 + 1 : rr1, cc] = segment

    return hmask, vmask


def _compose_output(grid: np.ndarray) -> np.ndarray:
    side, line_color = _top_left_square(grid)
    markers = _collect_markers(grid, side)
    base = _build_base_output(markers, side, line_color)
    hmask, vmask = _path_masks(base, markers, side)

    out = base.copy()
    out[hmask & ~vmask] = ORANGE
    out[vmask & ~hmask] = GREEN
    out[hmask & vmask] = YELLOW
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Project marker pixels onto a square-parameterized lattice, connect aligned pairs, and recolor their crossing.

    Input structure:
    - The top-left corner contains one solid teal square whose side length s is the only scale cue in the input.
    - Every other non-background cell is an isolated red or blue marker pixel outside that square.
    - Red markers can share rows, blue markers can share columns, and valid examples include at least one red-row / blue-column crossing opportunity.

    Rule:
    - Create a 15x15 black output grid and draw teal lattice lines starting at row and column s with period s+1.
    - Map each input marker at (r, c) to the center of its corresponding lattice cell: (r*(s+1)+s//2, c*(s+1)+s//2), preserving the marker color.
    - Fill black cells strictly between same-row red markers with orange and strictly between same-column blue markers with green; any cell claimed by both paths becomes yellow, while lattice lines and marker endpoints stay unchanged.

    Key insight:
    - The top-left square is not copied directly; its side length parameterizes the output spacing, so marker coordinates must first be reinterpreted in the enlarged lattice before the horizontal and vertical paths can be drawn.
    """
    return _compose_output(np.asarray(grid, dtype=int))


def _max_mappable_extent(side: int) -> int:
    return ((OUT_SIZE - 1 - side // 2) // (side + 1)) + 1


def _free_cells(shape: tuple[int, int], side: int) -> set[tuple[int, int]]:
    h, w = map(int, shape)
    return {
        (r, c)
        for r in range(h)
        for c in range(w)
        if not (r < side and c < side)
    }


def _cross_candidates(shape: tuple[int, int], side: int) -> list[tuple[int, int, list[int], list[int], list[int], list[int]]]:
    free = _free_cells(shape, side)
    h, w = map(int, shape)
    candidates: list[tuple[int, int, list[int], list[int], list[int], list[int]]] = []
    for r in range(h):
        for c in range(w):
            if (r, c) not in free:
                continue
            left = [cc for cc in range(c) if (r, cc) in free]
            right = [cc for cc in range(c + 1, w) if (r, cc) in free]
            up = [rr for rr in range(r) if (rr, c) in free]
            down = [rr for rr in range(r + 1, h) if (rr, c) in free]
            if left and right and up and down:
                candidates.append((r, c, left, right, up, down))
    return candidates


def _enumerate_generator_layouts() -> list[tuple[int, int, int]]:
    layouts: list[tuple[int, int, int]] = []
    for side in (2, 3):
        limit = _max_mappable_extent(side)
        for h in range(4, limit + 1):
            for w in range(4, limit + 1):
                if _cross_candidates((h, w), side):
                    layouts.append((side, h, w))
    return layouts


# With a fixed 15x15 output and the mandatory crossing, s=3 yields no valid layout:
# the 4x4 mappable input leaves only an L-shaped free region around the 3x3 square.
GEN_LAYOUTS = _enumerate_generator_layouts()
GEN_EXTRA_COUNTS = [(2, 2), (3, 2), (2, 3), (3, 3)]


def _random_marker_layout(side: int, shape: tuple[int, int]) -> tuple[list[tuple[int, int]], list[tuple[int, int]]] | None:
    candidates = _cross_candidates(shape, side)
    if not candidates:
        return None

    cross_r, cross_c, left, right, up, down = random.choice(candidates)
    red_pair = [
        (cross_r, random.choice(left)),
        (cross_r, random.choice(right)),
    ]
    blue_pair = [
        (random.choice(up), cross_c),
        (random.choice(down), cross_c),
    ]

    used = set(red_pair) | set(blue_pair)
    blocked = used | {(cross_r, cross_c)}
    blocked |= {(cross_r, c) for c in range(min(c for _r, c in red_pair) + 1, max(c for _r, c in red_pair))}
    blocked |= {(r, cross_c) for r in range(min(r for r, _c in blue_pair) + 1, max(r for r, _c in blue_pair))}

    free = _free_cells(shape, side)
    red_total, blue_total = random.choice(GEN_EXTRA_COUNTS)

    red_markers = list(red_pair)
    if red_total == 3:
        red_candidates = [
            cell
            for cell in free
            if cell not in blocked
            and cell not in used
            and cell[0] != cross_r
            and cell[1] != cross_c
        ]
        if not red_candidates:
            return None
        extra_red = random.choice(red_candidates)
        red_markers.append(extra_red)
        used.add(extra_red)

    blue_markers = list(blue_pair)
    if blue_total == 3:
        blue_candidates = [
            cell
            for cell in free
            if cell not in blocked
            and cell not in used
            and cell[0] != cross_r
            and cell[1] != cross_c
        ]
        if not blue_candidates:
            return None
        blue_markers.append(random.choice(blue_candidates))

    return red_markers, blue_markers


def _semantic_check(inp: np.ndarray, out: np.ndarray) -> bool:
    try:
        side, square_color = _top_left_square(inp)
        markers = _collect_markers(inp, side)
        red_markers = [(r, c) for r, c, color in markers if color == RED]
        blue_markers = [(r, c) for r, c, color in markers if color == BLUE]

        assert square_color == TEAL
        assert out.shape == (OUT_SIZE, OUT_SIZE)
        assert 2 <= len(red_markers) <= 3
        assert 2 <= len(blue_markers) <= 3
        assert any(sum(1 for r, _c in red_markers if r == row) == 2 for row in {r for r, _c in red_markers})
        assert any(sum(1 for _r, c in blue_markers if c == col) == 2 for col in {c for _r, c in blue_markers})

        line_rows = _grid_line_positions(side)
        line_mask = np.zeros_like(out, dtype=bool)
        line_mask[line_rows, :] = True
        line_mask[:, line_rows] = True
        assert np.all(out[line_mask] == square_color)

        base = _build_base_output(markers, side, square_color)
        hmask, vmask = _path_masks(base, markers, side)
        assert np.any(hmask)
        assert np.any(vmask)
        assert np.any(hmask & vmask)

        for r, c in red_markers:
            rr = _map_index(r, side)
            cc = _map_index(c, side)
            assert int(out[rr, cc]) == RED
        for r, c in blue_markers:
            rr = _map_index(r, side)
            cc = _map_index(c, side)
            assert int(out[rr, cc]) == BLUE

        expected = base.copy()
        expected[hmask & ~vmask] = ORANGE
        expected[vmask & ~hmask] = GREEN
        expected[hmask & vmask] = YELLOW
        assert np.array_equal(out, expected)
        return True
    except AssertionError:
        return False


def _gen_attempt() -> np.ndarray | None:
    side, h, w = random.choice(GEN_LAYOUTS)
    layout = _random_marker_layout(side, (h, w))
    if layout is None:
        return None

    red_markers, blue_markers = layout
    grid = np.full((h, w), BG, dtype=int)
    grid[:side, :side] = TEAL
    for r, c in red_markers:
        grid[r, c] = RED
    for r, c in blue_markers:
        if grid[r, c] != BG:
            return None
        grid[r, c] = BLUE

    out = main(grid)
    if _semantic_check(grid, out):
        return grid
    return None


def generate_input() -> np.ndarray:
    assert GEN_LAYOUTS, "expected at least one feasible generator layout"
    for _ in range(GEN_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 9")
