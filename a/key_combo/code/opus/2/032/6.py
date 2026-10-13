from __future__ import annotations

import random
from itertools import permutations

import numpy as np


BLACK = 0
GREY = 5
COLOR_POOL = [1, 2, 3, 4, 6, 7, 8, 9]


def _control_row(grid: np.ndarray) -> tuple[int, int, list[int]]:
    g = np.asarray(grid, dtype=int)
    candidates: list[tuple[int, int, list[int]]] = []
    for r in range(g.shape[0]):
        cols = [int(c) for c in np.flatnonzero((g[r] != BLACK) & (g[r] != GREY))]
        if len(cols) not in (4, 5):
            continue
        if cols != list(range(cols[0], cols[-1] + 1)):
            continue
        seq = [int(g[r, c]) for c in cols]
        if len(seq) == len(set(seq)):
            candidates.append((r, cols[0], seq))
    assert candidates, "expected a bottom control row"
    return max(candidates, key=lambda item: item[0])


def _cell_origins(n: int, inner: int, r0: int, c0: int) -> list[list[tuple[int, int]]]:
    tile = inner + 2
    step = tile + 1
    return [[(r0 + rr * step, c0 + cc * step) for cc in range(n)] for rr in range(n)]


def _read_layout(grid: np.ndarray, n: int, control_r: int, palette: set[int]) -> dict:
    g = np.asarray(grid, dtype=int)
    H, W = g.shape
    for inner in (2, 3, 4):
        tile = inner + 2
        main_h = n * tile + (n - 1)
        main_w = n * tile + (n - 1)
        for r0 in range(0, max(0, control_r - main_h + 1)):
            for c0 in range(0, W - main_w + 1):
                origins = _cell_origins(n, inner, r0, c0)
                matrix: list[list[int | None]] = []
                ok = True
                for rr in range(n):
                    row: list[int | None] = []
                    for cc in range(n):
                        sr, sc = origins[rr][cc]
                        patch = g[sr : sr + tile, sc : sc + tile]
                        if patch.shape != (tile, tile):
                            ok = False
                            break
                        border_ok = (
                            np.all(patch[0, :] == GREY)
                            and np.all(patch[-1, :] == GREY)
                            and np.all(patch[:, 0] == GREY)
                            and np.all(patch[:, -1] == GREY)
                        )
                        if not border_ok:
                            ok = False
                            break
                        inside = patch[1:-1, 1:-1]
                        vals = set(map(int, inside.ravel()))
                        if vals == {BLACK}:
                            row.append(None)
                        elif len(vals) == 1 and next(iter(vals)) in palette:
                            row.append(next(iter(vals)))
                        else:
                            ok = False
                            break
                    if not ok:
                        break
                    matrix.append(row)
                if ok:
                    return {
                        "n": n,
                        "inner": inner,
                        "tile": tile,
                        "r0": r0,
                        "c0": c0,
                        "origins": origins,
                        "matrix": matrix,
                    }
    raise AssertionError("could not parse a regular framed Latin-square board")


def _latin_completions(partial: list[list[int | None]], colors: list[int], limit: int = 200) -> list[tuple[tuple[int, ...], ...]]:
    n = len(partial)
    palette = set(colors)
    work = [row[:] for row in partial]
    rows = [set(v for v in row if v is not None) for row in work]
    cols = [set(work[r][c] for r in range(n) if work[r][c] is not None) for c in range(n)]
    blanks = [(r, c) for r in range(n) for c in range(n) if work[r][c] is None]
    out: list[tuple[tuple[int, ...], ...]] = []

    def rec() -> None:
        if len(out) >= limit:
            return
        best: tuple[int, int] | None = None
        best_domain: list[int] | None = None
        for r, c in blanks:
            if work[r][c] is not None:
                continue
            domain = sorted(palette - rows[r] - cols[c])
            if not domain:
                return
            if best_domain is None or len(domain) < len(best_domain):
                best = (r, c)
                best_domain = domain
                if len(domain) == 1:
                    break
        if best is None:
            out.append(tuple(tuple(int(v) for v in row) for row in work))
            return
        r, c = best
        assert best_domain is not None
        for value in best_domain:
            work[r][c] = value
            rows[r].add(value)
            cols[c].add(value)
            rec()
            cols[c].remove(value)
            rows[r].remove(value)
            work[r][c] = None

    rec()
    return out


def _control_paths(solution: tuple[tuple[int, ...], ...], control: list[int], limit: int = 3) -> list[tuple[tuple[int, int], ...]]:
    n = len(solution)
    by_color: dict[int, list[tuple[int, int]]] = {color: [] for color in control}
    for r in range(n):
        for c in range(n):
            value = int(solution[r][c])
            if value in by_color:
                by_color[value].append((r, c))

    paths: list[tuple[tuple[int, int], ...]] = []

    def rec(index: int, path: list[tuple[int, int]]) -> None:
        if len(paths) >= limit:
            return
        if index == len(control):
            paths.append(tuple(path))
            return
        color = control[index]
        if index == 0:
            candidates = by_color[color]
        else:
            r, c = path[-1]
            candidates = [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]
        for r, c in candidates:
            if 0 <= r < n and 0 <= c < n and solution[r][c] == color and (r, c) not in path:
                rec(index + 1, path + [(r, c)])

    rec(0, [])
    return paths


def _solve_matrix(partial: list[list[int | None]], control: list[int]) -> tuple[tuple[tuple[int, ...], ...], tuple[tuple[int, int], ...]]:
    latin = _latin_completions(partial, control, limit=200)
    assert len(latin) >= 2, "Latin-square constraint alone should be ambiguous"
    candidates: list[tuple[tuple[tuple[int, ...], ...], tuple[tuple[int, int], ...]]] = []
    for sol in latin:
        paths = _control_paths(sol, control, limit=2)
        if len(paths) == 1:
            candidates.append((sol, paths[0]))
    assert len(candidates) == 1, "control path must select a unique Latin completion"
    return candidates[0]


def _paint_solution(out: np.ndarray, layout: dict, solution: tuple[tuple[int, ...], ...]) -> None:
    n = int(layout["n"])
    inner = int(layout["inner"])
    origins = layout["origins"]
    for r in range(n):
        for c in range(n):
            sr, sc = origins[r][c]
            out[sr + 1 : sr + 1 + inner, sc + 1 : sc + 1 + inner] = int(solution[r][c])


def _draw_bridges(out: np.ndarray, layout: dict, path: tuple[tuple[int, int], ...], control: list[int]) -> None:
    inner = int(layout["inner"])
    tile = int(layout["tile"])
    origins = layout["origins"]
    for index, ((r0, c0), (r1, c1)) in enumerate(zip(path, path[1:])):
        color = int(control[index])
        sr0, sc0 = origins[r0][c0]
        sr1, sc1 = origins[r1][c1]
        if r0 == r1 and abs(c0 - c1) == 1:
            left_c = min(sc0, sc1)
            gap_c = left_c + tile
            row0 = sr0 + 1
            out[row0 : row0 + inner, gap_c] = color
        elif c0 == c1 and abs(r0 - r1) == 1:
            top_r = min(sr0, sr1)
            gap_r = top_r + tile
            col0 = sc0 + 1
            out[gap_r, col0 : col0 + inner] = color
        else:
            raise AssertionError("control path cells must be orthogonally adjacent")


def main(grid: np.ndarray) -> np.ndarray:
    """Complete the framed Latin square using the control-row adjacency path, then draw bridges along that path."""
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    control_r, _control_c, control = _control_row(g)
    layout = _read_layout(g, len(control), control_r, set(control))
    solution, path = _solve_matrix(layout["matrix"], control)
    out = g.copy()
    _paint_solution(out, layout, solution)
    _draw_bridges(out, layout, path, control)
    return out


BASE_MATRIX = (
    (0, 1, 2, 3),
    (3, 0, 1, 2),
    (1, 2, 3, 0),
    (2, 3, 0, 1),
)
BASE_BLANKS = ((1, 1), (1, 3), (2, 1), (2, 3))
BASE_CONTROL = (0, 2, 3, 1)


def _transform_coord(r: int, c: int, n: int, kind: int) -> tuple[int, int]:
    if kind == 0:
        return r, c
    if kind == 1:
        return c, n - 1 - r
    if kind == 2:
        return n - 1 - r, n - 1 - c
    if kind == 3:
        return n - 1 - c, r
    if kind == 4:
        return r, n - 1 - c
    if kind == 5:
        return n - 1 - r, c
    if kind == 6:
        return c, r
    return n - 1 - c, n - 1 - r


def _transformed_template() -> tuple[list[list[int]], set[tuple[int, int]], list[int]]:
    n = 4
    kind = random.randrange(8)
    matrix = [[0 for _ in range(n)] for _ in range(n)]
    blanks: set[tuple[int, int]] = set()
    for r in range(n):
        for c in range(n):
            rr, cc = _transform_coord(r, c, n, kind)
            matrix[rr][cc] = BASE_MATRIX[r][c]
    for r, c in BASE_BLANKS:
        blanks.add(_transform_coord(r, c, n, kind))
    return matrix, blanks, list(BASE_CONTROL)


def _render_input(matrix: list[list[int]], blanks: set[tuple[int, int]], control_symbols: list[int]) -> np.ndarray:
    n = len(matrix)
    inner = random.choice([3, 4])
    tile = inner + 2
    step = tile + 1
    main_h = n * tile + (n - 1)
    main_w = n * tile + (n - 1)
    top = 0 if inner == 4 else random.randint(0, 2)
    left = random.randint(0, max(0, 30 - main_w))
    control_r = top + main_h + 2
    height = control_r + 1
    width = max(main_w + left, left + n + 1)
    assert height <= 30 and width <= 30

    colors = random.sample(COLOR_POOL, n)
    symbol_to_color = {symbol: colors[i] for i, symbol in enumerate(random.sample(range(n), n))}
    grid = np.zeros((height, width), dtype=int)

    for rr in range(n):
        for cc in range(n):
            sr = top + rr * step
            sc = left + cc * step
            grid[sr : sr + tile, sc : sc + tile] = GREY
            fill = BLACK if (rr, cc) in blanks else symbol_to_color[matrix[rr][cc]]
            grid[sr + 1 : sr + 1 + inner, sc + 1 : sc + 1 + inner] = fill

    control = [symbol_to_color[s] for s in control_symbols]
    control_c = left + max(0, (main_w - n) // 2) + random.randint(0, min(2, max(0, main_w - n)))
    control_c = min(control_c, width - n)
    for offset, color in enumerate(control):
        grid[control_r, control_c + offset] = color
    return grid


def generate_input() -> np.ndarray:
    for _ in range(100):
        matrix, blanks, control_symbols = _transformed_template()
        grid = _render_input(matrix, blanks, control_symbols)
        try:
            _ = main(grid)
        except AssertionError:
            continue
        return grid
    raise RuntimeError("failed to generate a valid Latin-bridge input")
