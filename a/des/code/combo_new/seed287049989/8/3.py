from __future__ import annotations

import random
from typing import Any

import numpy as np


BLACK = 0
GREY = 5
CELL_SIZE = 3
SEPARATOR = 1
COLOR_POOL = (1, 2, 3, 4, 6, 7, 8, 9)
GEN_RETRIES = 500

DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]
MARKER_TO_DIR = {
    (0, 1): (-1, 0),
    (1, 2): (0, 1),
    (2, 1): (1, 0),
    (1, 0): (0, -1),
}
DIR_TO_MARKER = {value: key for key, value in MARKER_TO_DIR.items()}
NEIGHBORS4 = [(-1, 0), (0, -1), (0, 1), (1, 0)]


def _build_layout(n: int, border: bool = True, sep_color: int = GREY) -> dict[str, Any]:
    step = CELL_SIZE + SEPARATOR
    if border:
        size = n * CELL_SIZE + (n + 1) * SEPARATOR
        sep_rows = [i * step for i in range(n + 1)]
        sep_cols = [i * step for i in range(n + 1)]
        row_spans = [(1 + i * step, 1 + i * step + CELL_SIZE) for i in range(n)]
        col_spans = [(1 + i * step, 1 + i * step + CELL_SIZE) for i in range(n)]
    else:
        size = n * CELL_SIZE + (n - 1) * SEPARATOR
        sep_rows = [CELL_SIZE + i * step for i in range(n - 1)]
        sep_cols = [CELL_SIZE + i * step for i in range(n - 1)]
        row_spans = [(i * step, i * step + CELL_SIZE) for i in range(n)]
        col_spans = [(i * step, i * step + CELL_SIZE) for i in range(n)]
    return {
        "n": n,
        "shape": (size, size),
        "sep_color": int(sep_color),
        "sep_rows": sep_rows,
        "sep_cols": sep_cols,
        "row_spans": row_spans,
        "col_spans": col_spans,
    }


def _cell_spans(size: int, separators: list[int]) -> list[tuple[int, int]] | None:
    separators = sorted(int(x) for x in separators)
    bounds = [-1, *separators, size]
    spans: list[tuple[int, int]] = []
    for left, right in zip(bounds, bounds[1:], strict=False):
        gap = right - left - 1
        if gap == 0:
            continue
        if gap != CELL_SIZE:
            return None
        spans.append((left + 1, right))
    return spans or None


def _detect_layout(grid: np.ndarray) -> dict[str, Any]:
    h, w = grid.shape
    assert h == w, "expected a square board grid"
    best: tuple[int, int, list[int], list[int], list[tuple[int, int]], list[tuple[int, int]]] | None = None
    for color in np.unique(grid):
        color = int(color)
        sep_rows = [r for r in range(h) if np.all(grid[r, :] == color)]
        sep_cols = [c for c in range(w) if np.all(grid[:, c] == color)]
        if not sep_rows or not sep_cols:
            continue
        row_spans = _cell_spans(h, sep_rows)
        col_spans = _cell_spans(w, sep_cols)
        if row_spans is None or col_spans is None:
            continue
        if len(row_spans) != len(col_spans) or len(row_spans) not in (4, 5):
            continue
        score = len(sep_rows) + len(sep_cols)
        candidate = (score, color, sep_rows, sep_cols, row_spans, col_spans)
        if best is None or candidate[0] > best[0]:
            best = candidate
    assert best is not None, "failed to detect the grey board separators"
    _, sep_color, sep_rows, sep_cols, row_spans, col_spans = best
    return {
        "n": len(row_spans),
        "shape": grid.shape,
        "sep_color": int(sep_color),
        "sep_rows": list(sep_rows),
        "sep_cols": list(sep_cols),
        "row_spans": list(row_spans),
        "col_spans": list(col_spans),
    }


def _plain_patch(color: int) -> np.ndarray:
    return np.full((CELL_SIZE, CELL_SIZE), int(color), dtype=int)


def _make_tile(color: int, marker: tuple[int, int] | None, initial: tuple[int, int]) -> dict[str, Any]:
    patch = _plain_patch(color)
    if marker is not None:
        mr, mc = DIR_TO_MARKER[marker]
        patch[mr, mc] = BLACK
    return {
        "color": int(color),
        "marker": marker,
        "patch": patch,
        "initial": tuple(int(x) for x in initial),
    }


def _parse_input(grid: np.ndarray) -> tuple[dict[str, Any], dict[int, dict[str, Any]], list[list[int | None]], list[int]]:
    layout = _detect_layout(grid)
    sep_color = int(layout["sep_color"])
    n = int(layout["n"])
    tiles: dict[int, dict[str, Any]] = {}
    board: list[list[int | None]] = [[None for _ in range(n)] for _ in range(n)]
    marked_ids: list[int] = []
    tid = 0
    for r, (r0, r1) in enumerate(layout["row_spans"]):
        for c, (c0, c1) in enumerate(layout["col_spans"]):
            patch = np.asarray(grid[r0:r1, c0:c1], dtype=int)
            assert patch.shape == (CELL_SIZE, CELL_SIZE)
            assert not np.any(patch == sep_color), "separator color may not appear inside a tile"
            colors = [int(v) for v in np.unique(patch) if int(v) != BLACK]
            assert len(colors) == 1, "each input tile must be monochrome apart from the black marker"
            marker_cells = np.argwhere(patch == BLACK)
            marker: tuple[int, int] | None = None
            if len(marker_cells) == 1:
                pos = tuple(int(x) for x in marker_cells[0])
                assert pos in MARKER_TO_DIR, "marker must sit at one of the four edge centers"
                marker = MARKER_TO_DIR[pos]
            else:
                assert len(marker_cells) == 0, "only marked tiles may contain black pixels"
            tiles[tid] = {
                "color": colors[0],
                "marker": marker,
                "patch": patch.copy(),
                "initial": (r, c),
            }
            board[r][c] = tid
            if marker is not None:
                marked_ids.append(tid)
            tid += 1
    assert len(marked_ids) == 2, "expected exactly two marked tiles"
    marked_ids.sort(key=lambda current: tiles[current]["initial"])
    return layout, tiles, board, marked_ids


def _copy_board(board: list[list[int | None]]) -> list[list[int | None]]:
    return [row[:] for row in board]


def _find_tile(board: list[list[int | None]], tile_id: int) -> tuple[int, int] | None:
    for r, row in enumerate(board):
        for c, value in enumerate(row):
            if value == tile_id:
                return (r, c)
    return None


def _apply_pop(board: list[list[int | None]], position: tuple[int, int], direction: tuple[int, int]) -> None:
    n = len(board)
    r, c = position
    dr, dc = direction
    if (dr, dc) == (0, -1):
        for cc in range(c):
            board[r][cc] = board[r][cc + 1]
        board[r][c] = None
    elif (dr, dc) == (0, 1):
        for cc in range(n - 1, c, -1):
            board[r][cc] = board[r][cc - 1]
        board[r][c] = None
    elif (dr, dc) == (-1, 0):
        for rr in range(r):
            board[rr][c] = board[rr + 1][c]
        board[r][c] = None
    elif (dr, dc) == (1, 0):
        for rr in range(n - 1, r, -1):
            board[rr][c] = board[rr - 1][c]
        board[r][c] = None
    else:
        raise ValueError(f"unsupported direction: {direction}")


def _fill_votes(
    board: list[list[int | None]], tiles: dict[int, dict[str, Any]]
) -> tuple[dict[tuple[int, int], int], list[dict[str, Any]], bool]:
    n = len(board)
    fills: dict[tuple[int, int], int] = {}
    details: list[dict[str, Any]] = []
    saw_tie = False
    for r in range(n):
        for c in range(n):
            if board[r][c] is not None:
                continue
            counts: dict[int, int] = {}
            best_pos: dict[int, tuple[int, int]] = {}
            for dr, dc in NEIGHBORS4:
                nr, nc = r + dr, c + dc
                if not (0 <= nr < n and 0 <= nc < n):
                    continue
                neighbor = board[nr][nc]
                if neighbor is None:
                    continue
                color = int(tiles[neighbor]["color"])
                counts[color] = counts.get(color, 0) + 1
                pos = (nr, nc)
                if color not in best_pos or pos < best_pos[color]:
                    best_pos[color] = pos
            if not counts:
                continue
            top = max(counts.values())
            tied = [color for color, count in counts.items() if count == top]
            if len(tied) > 1:
                saw_tie = True
            winner = min(tied, key=lambda color: best_pos[color])
            fills[(r, c)] = winner
            details.append(
                {
                    "pos": (r, c),
                    "counts": counts,
                    "winner": winner,
                    "tie": len(tied) > 1,
                }
            )
    return fills, details, saw_tie


def _simulate(
    board: list[list[int | None]],
    tiles: dict[int, dict[str, Any]],
    marked_ids: list[int],
) -> dict[str, Any]:
    current = _copy_board(board)
    op_positions: list[tuple[int, int] | None] = []
    for tile_id in marked_ids:
        pos = _find_tile(current, tile_id)
        op_positions.append(pos)
        if pos is None:
            continue
        _apply_pop(current, pos, tiles[tile_id]["marker"])
    fills, fill_details, saw_tie = _fill_votes(current, tiles)
    return {
        "board_after_pops": current,
        "fills": fills,
        "fill_details": fill_details,
        "saw_tie": saw_tie,
        "op_positions": op_positions,
    }


def _render(
    board: list[list[int | None]],
    tiles: dict[int, dict[str, Any]],
    layout: dict[str, Any],
    fills: dict[tuple[int, int], int] | None = None,
) -> np.ndarray:
    fills = fills or {}
    h, w = layout["shape"]
    out = np.full((h, w), BLACK, dtype=int)
    sep_color = int(layout["sep_color"])
    out[layout["sep_rows"], :] = sep_color
    out[:, layout["sep_cols"]] = sep_color
    for r, (r0, r1) in enumerate(layout["row_spans"]):
        for c, (c0, c1) in enumerate(layout["col_spans"]):
            tile_id = board[r][c]
            if tile_id is not None:
                out[r0:r1, c0:c1] = np.asarray(tiles[tile_id]["patch"], dtype=int)
            elif (r, c) in fills:
                out[r0:r1, c0:c1] = int(fills[(r, c)])
            else:
                out[r0:r1, c0:c1] = BLACK
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pop marked board tiles toward their indicated edge, then refill the vacated slots by neighbor-majority color.

    Input structure:
    - The background is black, and a square N×N board (N is 4 or 5) is divided by 1-pixel grey separator lines.
    - Each board cell is a 3×3 monochrome tile using a non-black, non-grey color.
    - Exactly two tiles contain a single black marker pixel at one edge-center position inside the 3×3 tile.

    Rule:
    - The marker position encodes a pop direction: top, right, bottom, or left edge-center means up, right, down, or left.
    - Process the two marked tiles in top-to-bottom then left-to-right order; each marked tile pops using its current board position, shifting its row or column segment one tile toward the indicated board edge, dropping the edge tile and leaving the marked tile's old slot empty.
    - After both pops, fill every empty slot with a solid 3×3 tile whose color is the most common among its orthogonal neighboring tiles; ties are broken by the highest neighbor, then the leftmost.

    Key insight:
    - The tiny in-tile black dot is only a direction cue: the actual state change happens on the coarse tile grid, and the final black gaps are resolved by a local majority vote over neighboring tile colors.
    """
    g = np.asarray(grid, dtype=int)
    layout, tiles, board, marked_ids = _parse_input(g)
    result = _simulate(board, tiles, marked_ids)
    return _render(result["board_after_pops"], tiles, layout, result["fills"])


def _sample_palette() -> list[int]:
    size = random.randint(3, 6)
    return random.sample(list(COLOR_POOL), size)


def _color_grid_cluster(n: int, palette: list[int]) -> list[list[int]]:
    colors: list[list[int]] = [[palette[0] for _ in range(n)] for _ in range(n)]
    for r in range(n):
        for c in range(n):
            options = [random.choice(palette)]
            if r > 0:
                options.extend([colors[r - 1][c]] * 2)
            if c > 0:
                options.extend([colors[r][c - 1]] * 2)
            colors[r][c] = random.choice(options)
    return colors


def _color_grid_stripes(n: int, palette: list[int]) -> list[list[int]]:
    row_colors = [random.choice(palette) for _ in range(n)]
    col_colors = [random.choice(palette) for _ in range(n)]
    colors: list[list[int]] = []
    for r in range(n):
        row: list[int] = []
        for c in range(n):
            if random.random() < 0.55:
                row.append(row_colors[r])
            else:
                row.append(col_colors[c])
        colors.append(row)
    return colors


def _color_grid_blocks(n: int, palette: list[int]) -> list[list[int]]:
    split_r = random.randint(1, n - 1)
    split_c = random.randint(1, n - 1)
    quad_colors = [random.choice(palette) for _ in range(4)]
    colors: list[list[int]] = []
    for r in range(n):
        row: list[int] = []
        for c in range(n):
            idx = (r >= split_r) * 2 + (c >= split_c)
            value = quad_colors[idx]
            if random.random() < 0.2:
                value = random.choice(palette)
            row.append(value)
        colors.append(row)
    return colors


def _sample_colors(n: int) -> list[list[int]]:
    palette = _sample_palette()
    mode = random.choice(("cluster", "stripes", "blocks"))
    if mode == "cluster":
        colors = _color_grid_cluster(n, palette)
    elif mode == "stripes":
        colors = _color_grid_stripes(n, palette)
    else:
        colors = _color_grid_blocks(n, palette)
    flat = [value for row in colors for value in row]
    if len(set(flat)) < 2:
        return _sample_colors(n)
    return colors


def _sample_marker_positions(n: int) -> list[tuple[int, int]]:
    cells = [(r, c) for r in range(n) for c in range(n)]
    first = random.choice(cells)
    if random.random() < 0.5:
        if random.random() < 0.5:
            choices = [(first[0], c) for c in range(n) if c != first[1]]
        else:
            choices = [(r, first[1]) for r in range(n) if r != first[0]]
        second = random.choice(choices)
    else:
        second = random.choice([cell for cell in cells if cell != first])
    positions = [first, second]
    positions.sort()
    return positions


def _make_input_state(
    colors: list[list[int]],
    markers: dict[tuple[int, int], tuple[int, int]],
) -> tuple[dict[int, dict[str, Any]], list[list[int | None]], list[int]]:
    n = len(colors)
    tiles: dict[int, dict[str, Any]] = {}
    board: list[list[int | None]] = [[None for _ in range(n)] for _ in range(n)]
    marked_ids: list[int] = []
    tid = 0
    for r in range(n):
        for c in range(n):
            marker = markers.get((r, c))
            tiles[tid] = _make_tile(colors[r][c], marker, (r, c))
            board[r][c] = tid
            if marker is not None:
                marked_ids.append(tid)
            tid += 1
    marked_ids.sort(key=lambda current: tiles[current]["initial"])
    return tiles, board, marked_ids


def _validate_candidate(
    colors: list[list[int]],
    markers: dict[tuple[int, int], tuple[int, int]],
    require_tie: bool,
) -> np.ndarray | None:
    n = len(colors)
    layout = _build_layout(n, border=True, sep_color=GREY)
    tiles, board, marked_ids = _make_input_state(colors, markers)
    outcome = _simulate(board, tiles, marked_ids)

    if outcome["op_positions"][1] is None:
        return None
    if require_tie and not outcome["saw_tie"]:
        return None
    if not require_tie and outcome["saw_tie"]:
        return None
    if not outcome["fills"]:
        return None
    if any(not detail["counts"] for detail in outcome["fill_details"]):
        return None

    inp = _render(board, tiles, layout)
    expected = _render(outcome["board_after_pops"], tiles, layout, outcome["fills"])
    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        n = random.choice((4, 5))
        colors = _sample_colors(n)
        positions = _sample_marker_positions(n)
        directions = [random.choice(DIRS), random.choice(DIRS)]
        markers = {pos: direction for pos, direction in zip(positions, directions, strict=False)}
        require_tie = random.random() < 0.5
        grid = _validate_candidate(colors, markers, require_tie=require_tie)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input grid")
