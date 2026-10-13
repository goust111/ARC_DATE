from __future__ import annotations

import random

import numpy as np


BG = 0
GEN_GREY = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
SIDES = ("top", "bottom", "left", "right")
MIN_N = 4
MAX_N = 7
MAX_RETRIES = 500


def _board_shape(n: int) -> int:
    return 3 * n + 3


def _channel_coords(n: int, side: str, index: int) -> list[tuple[int, int]]:
    assert 0 <= index < n
    if side == "top":
        return [(r, index) for r in range(n)]
    if side == "bottom":
        return [(r, index) for r in range(n - 1, -1, -1)]
    if side == "left":
        return [(index, c) for c in range(n)]
    assert side == "right"
    return [(index, c) for c in range(n - 1, -1, -1)]


def _marker_coord(n: int, side: str, index: int) -> tuple[int, int]:
    board0 = 1
    line = board0 + 3 * index
    size = _board_shape(n)
    if side == "top":
        return 0, line
    if side == "bottom":
        return size - 1, line
    if side == "left":
        return line, 0
    assert side == "right"
    return line, size - 1


def _cell_patch(board0: int, r: int, c: int) -> tuple[slice, slice]:
    rr = board0 + 1 + 3 * r
    cc = board0 + 1 + 3 * c
    return slice(rr, rr + 2), slice(cc, cc + 2)


def _extract_cells_and_markers(grid: np.ndarray) -> tuple[np.ndarray, list[dict[str, int | str]], int, int]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    assert h == w
    assert h >= _board_shape(MIN_N)
    assert (h - 3) % 3 == 0

    n = (h - 3) // 3
    board0 = 1
    board1 = board0 + 3 * n
    grey = int(g[board0, board0])

    for k in range(n + 1):
        line = board0 + 3 * k
        assert np.all(g[line, board0 : board1 + 1] == grey)
        assert np.all(g[board0 : board1 + 1, line] == grey)

    cells = np.empty((n, n), dtype=int)
    for r in range(n):
        for c in range(n):
            rs, cs = _cell_patch(board0, r, c)
            patch = g[rs, cs]
            color = int(patch[0, 0])
            assert patch.shape == (2, 2)
            assert color != grey
            assert np.all(patch == color)
            cells[r, c] = color

    bg = int(g[0, 0])
    assert int(g[0, -1]) == bg
    assert int(g[-1, 0]) == bg
    assert int(g[-1, -1]) == bg

    markers: list[dict[str, int | str]] = []
    for side in SIDES:
        if side == "top":
            coords = [(0, c) for c in range(1, w - 1)]
        elif side == "bottom":
            coords = [(h - 1, c) for c in range(1, w - 1)]
        elif side == "left":
            coords = [(r, 0) for r in range(1, h - 1)]
        else:
            coords = [(r, w - 1) for r in range(1, h - 1)]

        hits = [(r, c) for r, c in coords if int(g[r, c]) != bg]
        assert len(hits) == 1
        r, c = hits[0]
        color = int(g[r, c])
        assert color != grey

        if side in ("top", "bottom"):
            assert c >= board0 and (c - board0) % 3 == 0
            index = (c - board0) // 3
        else:
            assert r >= board0 and (r - board0) % 3 == 0
            index = (r - board0) // 3
        assert 0 <= index < n

        markers.append({"side": side, "index": index, "color": color})

    return cells, markers, grey, bg


def _apply_channels(cells: np.ndarray, markers: list[dict[str, int | str]]) -> np.ndarray:
    n = int(cells.shape[0])
    out = cells.copy()

    for marker in markers:
        side = str(marker["side"])
        index = int(marker["index"])
        color = int(marker["color"])
        coords = _channel_coords(n, side, index)
        seq = [int(cells[r, c]) for r, c in coords]

        try:
            start = next(i for i, value in enumerate(seq) if value == color)
        except StopIteration:
            continue

        stop = n
        for i in range(start + 1, n):
            if seq[i] == color:
                stop = i
                break

        for i in range(start + 1, stop):
            r, c = coords[i]
            out[r, c] = color

    return out


def _paint_cells(base: np.ndarray, cells: np.ndarray) -> np.ndarray:
    out = np.array(base, dtype=int, copy=True)
    board0 = 1
    n = int(cells.shape[0])
    for r in range(n):
        for c in range(n):
            rs, cs = _cell_patch(board0, r, c)
            out[rs, cs] = int(cells[r, c])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each marked row/column channel after the first natural cell whose
    color matches that channel's border marker.
    """
    cells, markers, _grey, _bg = _extract_cells_and_markers(grid)
    out_cells = _apply_channels(cells, markers)
    return _paint_cells(np.asarray(grid, dtype=int), out_cells)


def _descriptor_targets(n: int, desc: dict[str, int | str]) -> list[tuple[int, int]]:
    coords = _channel_coords(n, str(desc["side"]), int(desc["index"]))
    mode = str(desc["mode"])
    if mode == "none":
        return []
    start = int(desc["source"]) + 1
    stop = n if mode == "edge" else int(desc["barrier"])
    return coords[start:stop]


def _random_descriptor(n: int, side: str, index: int, color: int) -> dict[str, int | str]:
    mode = random.choice(("none", "edge", "barrier"))
    if mode == "none":
        return {"side": side, "index": index, "color": color, "mode": mode}
    if mode == "edge":
        source = random.randint(0, n - 2)
        return {
            "side": side,
            "index": index,
            "color": color,
            "mode": mode,
            "source": source,
        }
    source = random.randint(0, n - 3)
    barrier = random.randint(source + 2, n - 1)
    return {
        "side": side,
        "index": index,
        "color": color,
        "mode": mode,
        "source": source,
        "barrier": barrier,
    }


def _render_input(cells: np.ndarray, markers: list[dict[str, int | str]], grey: int) -> np.ndarray:
    n = int(cells.shape[0])
    size = _board_shape(n)
    board0 = 1
    board1 = board0 + 3 * n

    grid = np.full((size, size), BG, dtype=int)
    for k in range(n + 1):
        line = board0 + 3 * k
        grid[line, board0 : board1 + 1] = grey
        grid[board0 : board1 + 1, line] = grey

    grid = _paint_cells(grid, cells)
    for marker in markers:
        r, c = _marker_coord(n, str(marker["side"]), int(marker["index"]))
        grid[r, c] = int(marker["color"])
    return grid


def _build_case() -> tuple[np.ndarray, np.ndarray] | None:
    n = random.randint(MIN_N, MAX_N)
    palette_size = random.randint(4, min(6, len(PALETTE)))
    active_palette = random.sample(list(PALETTE), palette_size)
    marker_colors = random.sample(active_palette, 4)

    top_col, bottom_col = random.sample(range(n), 2)
    left_row, right_row = random.sample(range(n), 2)
    descriptors = [
        _random_descriptor(n, "top", top_col, marker_colors[0]),
        _random_descriptor(n, "bottom", bottom_col, marker_colors[1]),
        _random_descriptor(n, "left", left_row, marker_colors[2]),
        _random_descriptor(n, "right", right_row, marker_colors[3]),
    ]

    exact: dict[tuple[int, int], int] = {}
    forbidden: dict[tuple[int, int], set[int]] = {}
    claims: dict[tuple[int, int], int] = {}

    for desc in descriptors:
        color = int(desc["color"])
        mode = str(desc["mode"])
        coords = _channel_coords(n, str(desc["side"]), int(desc["index"]))
        exact_indices: set[int]
        if mode == "none":
            exact_indices = set()
        elif mode == "edge":
            exact_indices = {int(desc["source"])}
        else:
            exact_indices = {int(desc["source"]), int(desc["barrier"])}

        for i, coord in enumerate(coords):
            if i in exact_indices:
                prev = exact.get(coord)
                if prev is not None and prev != color:
                    return None
                exact[coord] = color
            else:
                forbidden.setdefault(coord, set()).add(color)

        for coord in _descriptor_targets(n, desc):
            prev = claims.get(coord)
            if prev is not None and prev != color:
                return None
            claims[coord] = color

    if not claims:
        return None

    cells = np.empty((n, n), dtype=int)
    for r in range(n):
        for c in range(n):
            coord = (r, c)
            if coord in exact:
                color = exact[coord]
                if color in forbidden.get(coord, set()):
                    return None
                cells[r, c] = color
                continue

            options = [color for color in active_palette if color not in forbidden.get(coord, set())]
            if not options:
                return None
            cells[r, c] = random.choice(options)

    grid = _render_input(cells, descriptors, GEN_GREY)
    expected = _paint_cells(grid, _apply_channels(cells, descriptors))
    return grid, expected


def generate_input() -> np.ndarray:
    """
    Build a random checkerboard with four border markers. The generator varies
    board size, channel positions, marker colors, and whether each channel has
    no source, an open-ended infection, or a bounded infection.
    """
    for _ in range(MAX_RETRIES):
        built = _build_case()
        if built is None:
            continue
        grid, expected = built
        try:
            actual = main(grid)
        except Exception:
            continue
        if np.array_equal(actual, expected) and not np.array_equal(actual, grid):
            return grid
    raise RuntimeError(f"failed to generate a valid puzzle after {MAX_RETRIES} attempts")
