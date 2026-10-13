from __future__ import annotations

from collections import Counter
import random

import numpy as np


BLUE = 1
RED = 2
GREEN = 3
GREY = 5
BLACK = 0

QUADRANT_TO_ROT_K = {
    "tl": 0,
    "tr": 3,  # 90 deg clockwise
    "br": 2,  # 180 deg
    "bl": 1,  # 270 deg clockwise
}

ALL_QUADRANTS = ("tl", "tr", "br", "bl")
NEIGHBORS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
GEN_RETRIES = 400


def _dominant_color(values: np.ndarray) -> int:
    counts = Counter(map(int, values.reshape(-1)))
    return int(counts.most_common(1)[0][0])


def _separator_lines(grid: np.ndarray) -> tuple[int, list[int], list[int]]:
    h, w = grid.shape
    candidates = []
    for color in np.unique(grid):
        color = int(color)
        rows = [r for r in range(h) if np.all(grid[r, :] == color)]
        cols = [c for c in range(w) if np.all(grid[:, c] == color)]
        if rows and cols:
            candidates.append((color, rows, cols))
    assert len(candidates) == 1, "expected one separator color"
    return candidates[0]


def _spans(length: int, seps: list[int]) -> list[tuple[int, int]]:
    spans = []
    start = 0
    for sep in seps:
        assert start < sep, "separator must leave non-empty cells"
        spans.append((start, sep - 1))
        start = sep + 1
    assert start < length, "separator must leave final non-empty cell"
    spans.append((start, length - 1))
    return spans


def _quadrant_bounds(size: int) -> dict[str, tuple[range, range]]:
    split = size // 2 if size % 2 == 0 else size // 2 + 1
    return {
        "tl": (range(0, split), range(0, split)),
        "tr": (range(0, split), range(split, size)),
        "br": (range(split, size), range(split, size)),
        "bl": (range(split, size), range(0, split)),
    }


def _quadrant_of(size: int, r: int, c: int) -> str:
    for name, (rows, cols) in _quadrant_bounds(size).items():
        if r in rows and c in cols:
            return name
    raise AssertionError("marker is not inside any quadrant")


def _nonempty_quadrant_positions(size: int, quadrant: str) -> list[tuple[int, int]]:
    rows, cols = _quadrant_bounds(size)[quadrant]
    return [(r, c) for r in rows for c in cols]


def _rotations_distinct(tile: np.ndarray) -> bool:
    rots = {
        np.rot90(tile, k).tobytes()
        for k in range(4)
    }
    return len(rots) == 4


def _sorted_cells(row_spans: list[tuple[int, int]], col_spans: list[tuple[int, int]]) -> list[tuple[int, int, int, int]]:
    cells = []
    for r0, r1 in row_spans:
        for c0, c1 in col_spans:
            cells.append((r0, r1, c0, c1))
    return cells


def _parse(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape[0] >= 2 and g.shape[1] >= 3

    control = g[0]
    n = 0
    while n < control.shape[0] and int(control[n]) == BLUE:
        n += 1
    assert n >= 1, "control bar must start at column 0"
    assert np.all(control[n:] != BLUE), "control bar must be a single prefix run"

    board = g[1:]
    separator, sep_rows, sep_cols = _separator_lines(board)
    assert separator == GREY, "separator color must be grey"

    row_spans = _spans(board.shape[0], sep_rows)
    col_spans = _spans(board.shape[1], sep_cols)
    row_sizes = {r1 - r0 + 1 for r0, r1 in row_spans}
    col_sizes = {c1 - c0 + 1 for c0, c1 in col_spans}
    assert len(row_sizes) == 1 and len(col_sizes) == 1, "cells must be equal-sized"
    cell_h = row_sizes.pop()
    cell_w = col_sizes.pop()
    assert cell_h == cell_w, "cells must be square to support rotation"
    cell_size = cell_h
    assert cell_size in (3, 4), "cell size must be 3 or 4"
    assert len(row_spans) in (3, 4) and len(col_spans) in (3, 4)

    non_sep = board[board != separator]
    bg = _dominant_color(non_sep)
    assert bg == BLACK, "background must be black"

    source_tile = None
    source_pos = None
    markers = []
    for cell_idx, (r0, r1, c0, c1) in enumerate(_sorted_cells(row_spans, col_spans)):
        tile = board[r0 : r1 + 1, c0 : c1 + 1]
        has_red = np.any(tile == RED)
        has_green = np.any(tile == GREEN)
        if has_red:
            assert not has_green, "source cell cannot also contain a marker"
            assert set(map(int, np.unique(tile))).issubset({bg, RED})
            assert source_tile is None, "expected exactly one source tile"
            source_tile = tile.copy()
            source_pos = (r0, c0)
        elif has_green:
            coords = np.argwhere(tile == GREEN)
            assert coords.shape[0] == 1, "marker cell must contain exactly one green pixel"
            assert set(map(int, np.unique(tile))).issubset({bg, GREEN})
            mr, mc = map(int, coords[0])
            quadrant = _quadrant_of(cell_size, mr, mc)
            markers.append(
                {
                    "index": cell_idx,
                    "r0": r0,
                    "c0": c0,
                    "marker_rc": (mr, mc),
                    "quadrant": quadrant,
                }
            )
        else:
            assert np.all(tile == bg), "non-source, non-marker cells must be empty"

    assert source_tile is not None and source_pos is not None
    assert 1 <= n < len(markers), "control bar length must be smaller than marker count"
    return {
        "n": n,
        "board": board,
        "separator": separator,
        "row_spans": row_spans,
        "col_spans": col_spans,
        "cell_size": cell_size,
        "bg": bg,
        "source_tile": source_tile,
        "source_pos": source_pos,
        "markers": markers,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stamp the source sprite into marked separator-grid cells, rotating it by the marker's in-cell quadrant and limiting how many stamps execute by the top control bar.

    Input structure:
    - Row 0 contains a blue prefix bar on a black background.
    - Below it, full grey separator rows and columns partition the board into a 3x3 or 4x4 grid of equal 3x3 or 4x4 square cells.
    - Exactly one cell contains a red non-symmetric sprite; several other cells contain exactly one green marker pixel on black.

    Rule:
    - Read the green marker cells in row-major cell order.
    - The marker's quadrant inside its cell encodes the sprite rotation: top-left = 0 degrees, top-right = 90 degrees clockwise, bottom-right = 180 degrees, bottom-left = 270 degrees clockwise.
    - Stamp the rotated red sprite into only the first n marker cells, where n is the length of the blue control bar; clear later marker cells to black.
    - Keep the source cell, separator grid, and control bar unchanged.

    Key insight:
    - Three independent cues must be decoded together: the red cell provides the template, the green marker's local quadrant provides the rotation, and the blue bar provides the global cutoff on how many marked cells are actually stamped.
    """
    parsed = _parse(grid)
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    row_offset = 1

    source_tile = parsed["source_tile"]
    cell_size = parsed["cell_size"]
    bg = parsed["bg"]
    assert _rotations_distinct(source_tile), "source sprite must have four distinct rotations"

    for i, marker in enumerate(parsed["markers"], start=1):
        rr = row_offset + marker["r0"]
        cc = marker["c0"]
        if i <= parsed["n"]:
            rot_k = QUADRANT_TO_ROT_K[marker["quadrant"]]
            stamped = np.rot90(source_tile, rot_k)
            out[rr : rr + cell_size, cc : cc + cell_size] = stamped
        else:
            out[rr : rr + cell_size, cc : cc + cell_size] = bg

    return out


def _random_connected_shape(size: int, cells: int) -> np.ndarray:
    mask = np.zeros((size, size), dtype=bool)
    start = (random.randrange(size), random.randrange(size))
    mask[start] = True
    frontier = {start}
    while int(mask.sum()) < cells:
        growth = []
        for r, c in frontier:
            for dr, dc in NEIGHBORS4:
                rr, cc = r + dr, c + dc
                if 0 <= rr < size and 0 <= cc < size and not mask[rr, cc]:
                    growth.append((rr, cc))
        if not growth:
            return mask
        cell = random.choice(growth)
        mask[cell] = True
        frontier.add(cell)
    return mask


def _random_source_tile(size: int) -> np.ndarray | None:
    for _ in range(200):
        cells = random.randint(2, min(5, size * size - 1))
        mask = _random_connected_shape(size, cells)
        if int(mask.sum()) != cells:
            continue
        tile = np.full((size, size), BLACK, dtype=int)
        tile[mask] = RED
        if _rotations_distinct(tile):
            return tile
    return None


def _board_shape(cell_size: int, rows: int, cols: int) -> tuple[int, int]:
    return rows * cell_size + (rows - 1), cols * cell_size + (cols - 1)


def _cell_origin(cell_size: int, cell_r: int, cell_c: int) -> tuple[int, int]:
    return cell_r * (cell_size + 1), cell_c * (cell_size + 1)


def _make_empty_input(cell_size: int, rows: int, cols: int, n: int) -> np.ndarray:
    board_h, board_w = _board_shape(cell_size, rows, cols)
    grid = np.full((board_h + 1, board_w), BLACK, dtype=int)
    grid[0, :n] = BLUE
    for r in range(1, rows):
        grid[1 + r * cell_size + (r - 1), :] = GREY
    for c in range(1, cols):
        grid[1:, c * cell_size + (c - 1)] = GREY
    return grid


def _cell_order_positions(rows: int, cols: int) -> list[tuple[int, int]]:
    return [(r, c) for r in range(rows) for c in range(cols)]


def _stamp_into_cell(grid: np.ndarray, cell_size: int, cell_r: int, cell_c: int, patch: np.ndarray) -> None:
    r0, c0 = _cell_origin(cell_size, cell_r, cell_c)
    grid[1 + r0 : 1 + r0 + cell_size, c0 : c0 + cell_size] = patch


def _place_marker(tile: np.ndarray, quadrant: str) -> tuple[np.ndarray, tuple[int, int]]:
    coords = _nonempty_quadrant_positions(tile.shape[0], quadrant)
    rc = random.choice(coords)
    out = tile.copy()
    out[rc] = GREEN
    return out, rc


def _oracle_variants(inp: np.ndarray, parsed: dict) -> tuple[np.ndarray, np.ndarray]:
    out = main(inp)

    all_stamped = inp.copy()
    for marker in parsed["markers"]:
        rr = 1 + marker["r0"]
        cc = marker["c0"]
        rot_k = QUADRANT_TO_ROT_K[marker["quadrant"]]
        all_stamped[rr : rr + parsed["cell_size"], cc : cc + parsed["cell_size"]] = np.rot90(parsed["source_tile"], rot_k)

    unrotated = inp.copy()
    for i, marker in enumerate(parsed["markers"], start=1):
        rr = 1 + marker["r0"]
        cc = marker["c0"]
        if i <= parsed["n"]:
            unrotated[rr : rr + parsed["cell_size"], cc : cc + parsed["cell_size"]] = parsed["source_tile"]
        else:
            unrotated[rr : rr + parsed["cell_size"], cc : cc + parsed["cell_size"]] = parsed["bg"]
    return out, all_stamped, unrotated


def _validate_generated(inp: np.ndarray) -> bool:
    try:
        parsed = _parse(inp)
        out, all_stamped, unrotated = _oracle_variants(inp, parsed)
    except Exception:
        return False

    if inp.ndim != 2 or out.ndim != 2:
        return False
    if max(inp.shape) > 30 or max(out.shape) > 30:
        return False
    if not _rotations_distinct(parsed["source_tile"]):
        return False

    quadrants = [m["quadrant"] for m in parsed["markers"]]
    active_quadrants = quadrants[: parsed["n"]]
    if len(set(quadrants)) < 4:
        return False
    if len(set(active_quadrants)) < 2:
        return False
    if active_quadrants.count("tl") == len(active_quadrants):
        return False
    if parsed["n"] >= len(parsed["markers"]):
        return False
    if np.array_equal(out, all_stamped):
        return False
    if np.array_equal(out, unrotated):
        return False

    for i, marker in enumerate(parsed["markers"], start=1):
        rr = 1 + marker["r0"]
        cc = marker["c0"]
        tile = out[rr : rr + parsed["cell_size"], cc : cc + parsed["cell_size"]]
        if i <= parsed["n"]:
            want = np.rot90(parsed["source_tile"], QUADRANT_TO_ROT_K[marker["quadrant"]])
            if not np.array_equal(tile, want):
                return False
        else:
            if not np.all(tile == BLACK):
                return False

    return True


def _attempt_generate() -> np.ndarray | None:
    cell_size = random.choice([3, 4])
    rows = random.choice([3, 4])
    cols = random.choice([3, 4])
    total_cells = rows * cols
    k = random.randint(4, min(6, total_cells - 1))
    n = random.randint(2, k - 1)

    source_tile = _random_source_tile(cell_size)
    if source_tile is None:
        return None

    grid = _make_empty_input(cell_size, rows, cols, n)
    cells = _cell_order_positions(rows, cols)
    source_idx = random.randrange(total_cells)
    source_cell = cells[source_idx]

    remaining = [cell for i, cell in enumerate(cells) if i != source_idx]
    marker_cells = random.sample(remaining, k)
    marker_cells.sort(key=lambda rc: (rc[0], rc[1]))

    quadrants = list(ALL_QUADRANTS)
    while len(quadrants) < k:
        quadrants.append(random.choice(ALL_QUADRANTS))
    random.shuffle(quadrants)
    quadrants = quadrants[:k]
    if len(set(quadrants[:n])) < 2 or all(q == "tl" for q in quadrants[:n]):
        swap_candidates = [i for i in range(k) if quadrants[i] != "tl"]
        if not swap_candidates:
            return None
        swap_idx = random.choice(swap_candidates)
        quadrants[0], quadrants[swap_idx] = quadrants[swap_idx], quadrants[0]
        if len(set(quadrants[:n])) < 2:
            quadrants[1] = "tl" if quadrants[0] != "tl" else "tr"

    _stamp_into_cell(grid, cell_size, source_cell[0], source_cell[1], source_tile)
    for (cell_r, cell_c), quadrant in zip(marker_cells, quadrants):
        tile = np.full((cell_size, cell_size), BLACK, dtype=int)
        tile, _marker_rc = _place_marker(tile, quadrant)
        _stamp_into_cell(grid, cell_size, cell_r, cell_c, tile)

    if not _validate_generated(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
