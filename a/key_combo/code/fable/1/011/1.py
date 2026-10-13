from __future__ import annotations

import importlib.util
import random
from pathlib import Path

import numpy as np

_ARC_UTILS_PATH = Path("/home/lixinyu/snap/new/solvers/arc_utils.py")
_ARC_UTILS_SPEC = importlib.util.spec_from_file_location("_shared_arc_utils", _ARC_UTILS_PATH)
_ARC_UTILS = importlib.util.module_from_spec(_ARC_UTILS_SPEC)
assert _ARC_UTILS_SPEC is not None and _ARC_UTILS_SPEC.loader is not None
_ARC_UTILS_SPEC.loader.exec_module(_ARC_UTILS)

Color = _ARC_UTILS.Color
colors_present = _ARC_UTILS.colors_present
get_bg = _ARC_UTILS.get_bg
in_bounds = _ARC_UTILS.in_bounds


CELL_SIZE = 5
CELL_STRIDE = CELL_SIZE + 1
CENTER = (2, 2)
UNIT_DIRECTIONS = [
    (-1, 0),
    (1, 0),
    (0, -1),
    (0, 1),
    (-1, -1),
    (-1, 1),
    (1, -1),
    (1, 1),
]
ALL_JUMPS = [(dr * step, dc * step) for dr, dc in UNIT_DIRECTIONS for step in (1, 2)]
GEN_RETRIES = 400


def _is_valid_jump(dr: int, dc: int) -> bool:
    dr = int(dr)
    dc = int(dc)
    if dr == 0 and dc == 0:
        return False
    if dr == 0 or dc == 0:
        return min(abs(dr), abs(dc)) == 0 and max(abs(dr), abs(dc)) in (1, 2)
    return abs(dr) == abs(dc) and abs(dr) in (1, 2)


def _extract_cell_spans(size: int, separators: list[int]) -> list[tuple[int, int]] | None:
    sep_set = set(map(int, separators))
    spans: list[tuple[int, int]] = []
    i = 0
    while i < int(size):
        if i in sep_set:
            i += 1
            continue
        j = i
        while j < int(size) and j not in sep_set:
            j += 1
        spans.append((i, j))
        i = j
    if len(spans) < 2 or any(int(b - a) != CELL_SIZE for a, b in spans):
        return None
    return spans


def _find_lattice(grid: np.ndarray) -> tuple[int, list[tuple[int, int]], list[tuple[int, int]]]:
    g = np.asarray(grid, dtype=int)
    H, W = g.shape
    best = None
    for color in colors_present(g):
        row_separators = [r for r in range(H) if bool(np.all(g[r, :] == int(color)))]
        col_separators = [c for c in range(W) if bool(np.all(g[:, c] == int(color)))]
        if not row_separators or not col_separators:
            continue
        row_spans = _extract_cell_spans(H, row_separators)
        col_spans = _extract_cell_spans(W, col_separators)
        if row_spans is None or col_spans is None:
            continue
        score = (len(row_separators) + len(col_separators), len(row_spans) * len(col_spans))
        if best is None or score > best[0]:
            best = (score, int(color), row_spans, col_spans)
    assert best is not None, "failed to detect the 5x5 lattice"
    return int(best[1]), best[2], best[3]


def _parse_cell(block: np.ndarray, bg: int) -> dict | None:
    pts = np.argwhere(block != int(bg))
    if pts.size == 0:
        return None

    colors = {int(v) for v in block[block != int(bg)].tolist()}
    assert len(colors) == 1, "each non-empty cell must be monochrome"
    color = int(next(iter(colors)))
    coords = [tuple(map(int, pt)) for pt in pts]

    if len(coords) == 1:
        pos = coords[0]
        if pos == CENTER:
            return {"kind": "end", "color": color, "jump": None}
        dr, dc = int(pos[0] - CENTER[0]), int(pos[1] - CENTER[1])
        assert _is_valid_jump(dr, dc), "pointer must be cardinal/diagonal at distance 1 or 2"
        return {"kind": "pointer", "color": color, "jump": (dr, dc)}

    assert len(coords) == 2 and CENTER in coords, "two-pixel cells must contain the center"
    other = coords[0] if coords[1] == CENTER else coords[1]
    dr, dc = int(other[0] - CENTER[0]), int(other[1] - CENTER[1])
    assert _is_valid_jump(dr, dc), "start pointer must be cardinal/diagonal at distance 1 or 2"
    return {"kind": "start", "color": color, "jump": (dr, dc)}


def main(grid: np.ndarray) -> np.ndarray:
    """
    Read the visited cell colors from a 5x5 lattice by jumping between cells according to each cell's offset marker.

    Input structure:
    - The grid is partitioned by full separator rows and columns into equal 5x5 cells.
    - Every non-empty cell is monochrome and contains either one offset pixel, one center pixel, or a center pixel plus one offset pixel.
    - Exactly one cell is the start (center + offset), exactly one cell is the end (center only), and all other non-empty cells are pointer cells.

    Rule:
    - Interpret each offset pixel relative to the 5x5 cell center: its direction gives the jump direction and its distance 1 or 2 gives the jump length in cell units.
    - Start from the unique start cell and repeatedly jump directly to the landing cell; skipped intermediate cells are not visited and their pointers do not matter.
    - Stop when the landing cell is the unique center-only end cell, then output a 1xk row of the visited cell colors in order, including the start and end colors.

    Key insight:
    - The payload pixels encode cell-to-cell jump vectors, so a distance-2 pointer must leap over the middle cell instead of walking through it step by step.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    _separator_color, row_spans, col_spans = _find_lattice(g)
    bg = int(get_bg(g))

    cells: dict[tuple[int, int], dict] = {}
    start = None
    end = None
    for rr, (r0, r1) in enumerate(row_spans):
        for cc, (c0, c1) in enumerate(col_spans):
            cell = g[int(r0) : int(r1), int(c0) : int(c1)]
            meta = _parse_cell(cell, bg)
            if meta is None:
                continue
            cells[(int(rr), int(cc))] = meta
            if meta["kind"] == "start":
                assert start is None, "expected a unique start cell"
                start = (int(rr), int(cc))
            elif meta["kind"] == "end":
                assert end is None, "expected a unique end cell"
                end = (int(rr), int(cc))

    assert start is not None and end is not None, "missing start or end cell"

    path_colors: list[int] = []
    seen: set[tuple[int, int]] = set()
    cur = start
    while True:
        assert cur not in seen, "path must terminate without loops"
        seen.add(cur)
        assert cur in cells, "jump must land on a non-empty cell"
        meta = cells[cur]
        path_colors.append(int(meta["color"]))
        if meta["kind"] == "end":
            break
        dr, dc = meta["jump"]
        nxt = (int(cur[0] + dr), int(cur[1] + dc))
        assert in_bounds((len(row_spans), len(col_spans)), *nxt), "jump left the lattice"
        cur = nxt

    return np.array([path_colors], dtype=int)


def _move_flags(jump: tuple[int, int]) -> set[str]:
    dr, dc = map(int, jump)
    step2 = max(abs(dr), abs(dc)) == 2
    diagonal = dr != 0 and dc != 0
    flags = {"step2" if step2 else "step1"}
    flags.add("diagonal" if diagonal else "cardinal")
    return flags


def _search_path(
    rows: int, cols: int, length: int, rng: random.Random
) -> tuple[list[tuple[int, int]], list[tuple[int, int]], set[tuple[int, int]]] | None:
    required = {"step1", "step2", "diagonal", "cardinal"}
    starts = [(r, c) for r in range(rows) for c in range(cols)]
    rng.shuffle(starts)

    def dfs(
        path: list[tuple[int, int]],
        path_set: set[tuple[int, int]],
        reserved: set[tuple[int, int]],
        jumps: list[tuple[int, int]],
        covered: set[str],
    ) -> bool:
        if len(path) == length:
            return covered >= required

        cur = path[-1]
        candidates = []
        jump_order = ALL_JUMPS[:]
        rng.shuffle(jump_order)
        for dr, dc in jump_order:
            dest = (int(cur[0] + dr), int(cur[1] + dc))
            if not in_bounds((rows, cols), *dest):
                continue
            if dest in path_set or dest in reserved:
                continue

            midpoint = None
            if max(abs(dr), abs(dc)) == 2:
                midpoint = (int(cur[0] + dr // 2), int(cur[1] + dc // 2))
                if midpoint in path_set or midpoint in reserved:
                    continue

            new_flags = covered | _move_flags((dr, dc))
            gain = len(new_flags - covered)
            candidates.append((gain, rng.random(), (dr, dc), dest, midpoint, new_flags))

        candidates.sort(reverse=True)
        for _gain, _noise, jump, dest, midpoint, new_flags in candidates:
            path.append(dest)
            path_set.add(dest)
            jumps.append(jump)
            if midpoint is not None:
                reserved.add(midpoint)
            if dfs(path, path_set, reserved, jumps, new_flags):
                return True
            if midpoint is not None:
                reserved.remove(midpoint)
            jumps.pop()
            path_set.remove(dest)
            path.pop()
        return False

    for start in starts:
        path = [start]
        path_set = {start}
        reserved: set[tuple[int, int]] = set()
        jumps: list[tuple[int, int]] = []
        if dfs(path, path_set, reserved, jumps, set()):
            return path, jumps, reserved
    return None


def _cell_origin(cell: tuple[int, int]) -> tuple[int, int]:
    r, c = map(int, cell)
    return int(r * CELL_STRIDE), int(c * CELL_STRIDE)


def _paint_cell(
    grid: np.ndarray,
    cell: tuple[int, int],
    color: int,
    *,
    jump: tuple[int, int] | None = None,
    center: bool = False,
) -> None:
    r0, c0 = _cell_origin(cell)
    if center:
        grid[int(r0 + CENTER[0]), int(c0 + CENTER[1])] = int(color)
    if jump is not None:
        dr, dc = map(int, jump)
        grid[int(r0 + CENTER[0] + dr), int(c0 + CENTER[1] + dc)] = int(color)


def _decoy_jump(
    cell: tuple[int, int],
    rows: int,
    cols: int,
    occupied: set[tuple[int, int]],
    rng: random.Random,
) -> tuple[int, int]:
    preferred = []
    fallback = []
    for jump in ALL_JUMPS:
        dest = (int(cell[0] + jump[0]), int(cell[1] + jump[1]))
        if not in_bounds((rows, cols), *dest):
            continue
        fallback.append(jump)
        if dest not in occupied:
            preferred.append(jump)
    assert fallback, "decoy cell must have at least one valid pointer"
    choices = preferred if preferred else fallback
    return tuple(map(int, rng.choice(choices)))


def _render_grid(
    rows: int,
    cols: int,
    path: list[tuple[int, int]],
    jumps: list[tuple[int, int]],
    reserved: set[tuple[int, int]],
    rng: random.Random,
) -> np.ndarray:
    bg, separator = rng.sample(list(range(10)), 2)
    payload = [c for c in range(10) if c not in (bg, separator)]
    path_colors = rng.sample(payload, len(path))

    H = rows * CELL_SIZE + (rows - 1)
    W = cols * CELL_SIZE + (cols - 1)
    grid = np.full((H, W), int(separator), dtype=int)
    for r in range(rows):
        for c in range(cols):
            r0, c0 = _cell_origin((r, c))
            grid[int(r0) : int(r0 + CELL_SIZE), int(c0) : int(c0 + CELL_SIZE)] = int(bg)

    for idx, cell in enumerate(path):
        color = int(path_colors[idx])
        if idx == 0:
            _paint_cell(grid, cell, color, jump=jumps[0], center=True)
        elif idx == len(path) - 1:
            _paint_cell(grid, cell, color, center=True)
        else:
            _paint_cell(grid, cell, color, jump=jumps[idx])

    remaining = [c for c in payload if c not in path_colors]
    occupied = set(path) | set(reserved)
    for cell in sorted(reserved):
        decoy_color = int(remaining.pop()) if remaining else int(rng.choice(payload))
        _paint_cell(grid, cell, decoy_color, jump=_decoy_jump(cell, rows, cols, occupied, rng))

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        rows = rng.randint(3, 5)
        cols = rng.randint(3, 5)
        max_length = min(6, rows * cols - 1)
        length = rng.randint(4, max_length)
        result = _search_path(rows, cols, length, rng)
        if result is None:
            continue
        path, jumps, reserved = result
        if not reserved:
            continue
        grid = _render_grid(rows, cols, path, jumps, reserved, rng)
        out = main(grid)
        if out.shape == (1, len(path)):
            return grid
    raise RuntimeError(f"generate_input failed after {GEN_RETRIES} attempts")
