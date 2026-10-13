import random

import numpy as np


RED = 2
CYAN = 8
GRAY = 5
PAINT_CYCLE = (RED, CYAN, GRAY)


def _all_positions(n: int) -> list[tuple[int, int]]:
    return [(r, c) for r in range(n) for c in range(n)]


def _tile_origin(n: int, tr: int, tc: int) -> tuple[int, int]:
    step = n + 1
    return tr * step, tc * step


def _infer_n(side: int) -> int:
    for n in range(2, 31):
        if n * n + n - 1 == side:
            return n
    raise AssertionError("grid side must be n*n+n-1 for some n >= 2")


def _infer_background(g: np.ndarray, n: int) -> int:
    values = []
    for k in range(1, n):
        idx = k * (n + 1) - 1
        row = g[idx, :]
        col = g[:, idx]
        assert np.all(row == row[0]), "separator rows must be solid"
        assert np.all(col == col[0]), "separator columns must be solid"
        values.append(int(row[0]))
        values.append(int(col[0]))
    assert len(set(values)) == 1, "all separators must share one background color"
    return values[0]


def _read_pointers(g: np.ndarray, n: int, bg: int) -> dict[tuple[int, int], tuple[int, int]]:
    pointers: dict[tuple[int, int], tuple[int, int]] = {}
    foreground_colors: set[int] = set()

    for tr in range(n):
        for tc in range(n):
            r0, c0 = _tile_origin(n, tr, tc)
            tile = g[r0 : r0 + n, c0 : c0 + n]
            holes = np.argwhere(tile == bg)
            assert holes.shape[0] == 1, "each tile must contain exactly one background hole"
            hr, hc = map(int, holes[0])

            fill = tile[tile != bg]
            assert fill.size == n * n - 1, "tile foreground must fill every non-hole cell"
            unique_fill = set(map(int, np.unique(fill)))
            assert len(unique_fill) == 1, "each tile foreground must be monochrome"
            foreground_colors.update(unique_fill)

            pointers[(tr, tc)] = (hr, hc)

    assert len(foreground_colors) == 1, "all tiles must share the same foreground color"
    return pointers


def _trace_chain(pointers: dict[tuple[int, int], tuple[int, int]], n: int) -> list[tuple[int, int]]:
    positions = _all_positions(n)
    position_set = set(positions)
    assert set(pointers) == position_set, "expected one pointer for every tile"

    indegree = {pos: 0 for pos in positions}
    for dst in pointers.values():
        assert dst in position_set, "hole coordinates must point inside the tile array"
        indegree[dst] += 1

    heads = [pos for pos, count in indegree.items() if count == 0]
    tails = [pos for pos, dst in pointers.items() if pos == dst]
    assert len(heads) == 1, "expected a unique chain head"
    assert len(tails) == 1, "expected a unique self-pointing chain tail"
    tail = tails[0]

    chain: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    cur = heads[0]
    while True:
        assert cur not in seen, "pointer chain must not cycle before the tail"
        chain.append(cur)
        seen.add(cur)
        nxt = pointers[cur]
        if nxt == cur:
            assert cur == tail, "only the self-pointer tile can terminate the chain"
            break
        cur = nxt

    assert len(chain) == n * n, "the pointer chain must cover every tile exactly once"
    assert set(chain) == position_set, "the pointer chain must cover exactly the tile array"
    return chain


def _paint_chain(g: np.ndarray, n: int, bg: int, chain: list[tuple[int, int]]) -> np.ndarray:
    out = g.copy()
    for i, (tr, tc) in enumerate(chain):
        r0, c0 = _tile_origin(n, tr, tc)
        source_tile = g[r0 : r0 + n, c0 : c0 + n]
        target_tile = out[r0 : r0 + n, c0 : c0 + n]
        target_tile[source_tile != bg] = PAINT_CYCLE[i % len(PAINT_CYCLE)]
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Trace the tile-hole pointer chain and recolor tiles red, cyan, and gray cyclically.

    Input structure:
    A square grid has side length n*n+n-1 and is divided by one-pixel
    background separator rows and columns into an n by n array of n by n tiles.
    Every tile has the same monochrome foreground fill except for one
    background hole pixel.

    Rule:
    In a tile at array position (tr, tc), read the hole's local coordinate
    (r, c) as a pointer to the tile at array position (r, c). The pointers form
    one chain that covers every tile once; the unique self-pointing tile is the
    tail, and the unique tile with indegree zero is the head. Starting at the
    head, recolor each tile's foreground pixels red, cyan, gray, red, ... while
    leaving the holes and separator lines as background.

    Key insight:
    The hole location is not a label for the tile that contains it; it is a
    coordinate pointer to the next tile, so the hidden chain order rather than
    scan order determines the cyclic recoloring.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape[0] == g.shape[1], "expected a square 2D grid"

    n = _infer_n(int(g.shape[0]))
    bg = _infer_background(g, n)
    pointers = _read_pointers(g, n, bg)
    chain = _trace_chain(pointers, n)
    return _paint_chain(g, n, bg, chain)


# --------------- generator ---------------

GEN_N_VALUES = (3, 4, 5)
GEN_BG = 0
GEN_FOREGROUND_COLORS = (1, 3, 4, 6, 7, 9)
GEN_RETRY_LIMIT = 300


def _row_major(n: int) -> list[tuple[int, int]]:
    return [(r, c) for r in range(n) for c in range(n)]


def _col_major(n: int) -> list[tuple[int, int]]:
    return [(r, c) for c in range(n) for r in range(n)]


def _snake_rows(n: int) -> list[tuple[int, int]]:
    out = []
    for r in range(n):
        cols = range(n) if r % 2 == 0 else range(n - 1, -1, -1)
        out.extend((r, c) for c in cols)
    return out


def _snake_cols(n: int) -> list[tuple[int, int]]:
    out = []
    for c in range(n):
        rows = range(n) if c % 2 == 0 else range(n - 1, -1, -1)
        out.extend((r, c) for r in rows)
    return out


def _path_is_nontrivial(path: list[tuple[int, int]], n: int) -> bool:
    canonical = (
        _row_major(n),
        list(reversed(_row_major(n))),
        _col_major(n),
        list(reversed(_col_major(n))),
        _snake_rows(n),
        list(reversed(_snake_rows(n))),
        _snake_cols(n),
        list(reversed(_snake_cols(n))),
    )
    if any(path == order for order in canonical):
        return False
    if path[0] == (0, 0) or path[-1] == (n - 1, n - 1):
        return False

    order_index = {pos: i for i, pos in enumerate(_row_major(n))}
    deltas = [order_index[path[i + 1]] - order_index[path[i]] for i in range(len(path) - 1)]
    if not (any(delta < 0 for delta in deltas) and any(delta > 0 for delta in deltas)):
        return False

    jumps = [
        abs(path[i + 1][0] - path[i][0]) + abs(path[i + 1][1] - path[i][1])
        for i in range(len(path) - 1)
    ]
    return sum(jump > 1 for jump in jumps) >= max(2, n)


def _gen_path(n: int) -> list[tuple[int, int]] | None:
    positions = _all_positions(n)
    for _ in range(200):
        path = positions[:]
        random.shuffle(path)
        if _path_is_nontrivial(path, n):
            return path
    return None


def _pointers_from_path(path: list[tuple[int, int]]) -> dict[tuple[int, int], tuple[int, int]]:
    pointers = {}
    for src, dst in zip(path, path[1:]):
        pointers[src] = dst
    pointers[path[-1]] = path[-1]
    return pointers


def _build_input(n: int, bg: int, fg: int, path: list[tuple[int, int]]) -> np.ndarray:
    side = n * n + n - 1
    grid = np.full((side, side), bg, dtype=int)
    pointers = _pointers_from_path(path)

    for tr in range(n):
        for tc in range(n):
            r0, c0 = _tile_origin(n, tr, tc)
            grid[r0 : r0 + n, c0 : c0 + n] = fg
            hr, hc = pointers[(tr, tc)]
            grid[r0 + hr, c0 + hc] = bg
    return grid


def _expected_from_sample(grid: np.ndarray, n: int, bg: int, path: list[tuple[int, int]]) -> np.ndarray:
    return _paint_chain(grid, n, bg, path)


def _gen_semantically_valid(grid: np.ndarray, n: int, bg: int, path: list[tuple[int, int]], expected: np.ndarray) -> bool:
    if max(grid.shape) > 30 or max(expected.shape) > 30:
        return False
    if not all(np.any(expected == color) for color in PAINT_CYCLE):
        return False
    if not np.all(expected[grid == bg] == bg):
        return False

    try:
        parsed_n = _infer_n(int(grid.shape[0]))
        parsed_bg = _infer_background(grid, parsed_n)
        pointers = _read_pointers(grid, parsed_n, parsed_bg)
        traced = _trace_chain(pointers, parsed_n)
        actual = main(grid)
    except Exception:
        return False

    return (
        parsed_n == n
        and parsed_bg == bg
        and traced == path
        and not np.array_equal(grid, expected)
        and np.array_equal(actual, expected)
    )


def _gen_attempt() -> np.ndarray | None:
    n = random.choice(GEN_N_VALUES)
    path = _gen_path(n)
    if path is None:
        return None

    fg = random.choice(GEN_FOREGROUND_COLORS)
    grid = _build_input(n, GEN_BG, fg, path)
    expected = _expected_from_sample(grid, n, GEN_BG, path)

    if _gen_semantically_valid(grid, n, GEN_BG, path, expected):
        return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid pointer-chain tile grid")
