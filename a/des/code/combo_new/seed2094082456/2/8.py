from __future__ import annotations

import numpy as np


RNG = np.random.default_rng()
GEN_ATTEMPTS = 400
PALETTE = tuple(range(10))
LINE_LAYOUTS = ((1, 2), (2, 1), (2, 2))


def _dominant_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _solid_block_color(block: np.ndarray) -> int:
    color = int(block[0, 0])
    assert block.shape == (2, 2)
    assert np.all(block == color), "corner reference blocks must be solid 2x2 patches"
    return color


def _mode_color(values: np.ndarray) -> int:
    colors, counts = np.unique(values, return_counts=True)
    return int(colors[np.argmax(counts)])


def _corner_priority(grid: np.ndarray) -> list[int]:
    h, w = grid.shape
    assert h >= 6 and w >= 6
    tl = _solid_block_color(grid[0:2, 0:2])
    tr = _solid_block_color(grid[0:2, w - 2 : w])
    br = _solid_block_color(grid[h - 2 : h, w - 2 : w])
    bl = _solid_block_color(grid[h - 2 : h, 0:2])
    order = [tl, tr, br, bl]
    assert len(set(order)) == 4, "corner reference colors must be distinct"
    return order


def _find_lines(grid: np.ndarray, bg: int) -> tuple[list[int], list[int], dict[int, int], dict[int, int]]:
    h, w = grid.shape
    rows = [r for r in range(h) if np.all(grid[r, :] != bg)]
    cols = [c for c in range(w) if np.all(grid[:, c] != bg)]
    assert rows and cols, "expected at least one horizontal and one vertical path"

    row_colors = {r: _mode_color(grid[r, :]) for r in rows}
    col_colors = {c: _mode_color(grid[:, c]) for c in cols}

    for r in rows:
        assert row_colors[r] != bg
    for c in cols:
        assert col_colors[c] != bg

    return rows, cols, row_colors, col_colors


def main(grid: np.ndarray) -> np.ndarray:
    """Recolor each straight path crossing using the corner-indicated path priority.

    Input structure:
    - The background is the dominant color.
    - Several one-cell-wide straight paths span the grid from edge to edge; each
      horizontal or vertical path has its own color.
    - Horizontal and vertical paths may cross, and the visible input color at a
      crossing may belong to only one of the two underlying paths.
    - Each corner contains a solid 2x2 reference block, whose color order around
      the corners encodes the priority ranking.

    Rule:
    - Read the four corner block colors in clockwise priority order:
      top-left, top-right, bottom-right, bottom-left.
    - Detect every full-width horizontal path row and every full-height vertical
      path column, then recover each path's true color from the dominant color on
      that row or column.
    - At every row/column crossing, repaint the single shared pixel with the
      higher-priority of the two path colors.
    - Leave all non-crossing path pixels and the corner reference blocks unchanged.

    Key insight:
    - The row/column lines reveal the hidden path identities, while the corner
      blocks are not decoration: they define which path color wins at crossings.
    """
    g = np.asarray(grid, dtype=int)
    bg = _dominant_color(g)
    order = _corner_priority(g)
    rank = {color: idx for idx, color in enumerate(order)}

    rows, cols, row_colors, col_colors = _find_lines(g, bg)
    for color in set(row_colors.values()) | set(col_colors.values()):
        assert color in rank, "all path colors must appear in the corner reference order"

    out = g.copy()
    for r in rows:
        h_color = row_colors[r]
        for c in cols:
            v_color = col_colors[c]
            out[r, c] = h_color if rank[h_color] < rank[v_color] else v_color
    return out


def _sample_positions(length: int, count: int) -> list[int] | None:
    if count == 0:
        return []
    candidates = list(range(3, length - 3))
    if len(candidates) < count:
        return None
    RNG.shuffle(candidates)
    chosen: list[int] = []
    for pos in candidates:
        if all(abs(pos - other) >= 2 for other in chosen):
            chosen.append(int(pos))
            if len(chosen) == count:
                return sorted(chosen)
    return None


def _assign_colors(order: list[int], h_count: int, v_count: int) -> tuple[list[int], list[int]] | None:
    ranks = list(range(4))
    if (h_count, v_count) == (1, 2):
        mid = int(RNG.choice([1, 2]))
        low = int(RNG.choice([rank for rank in ranks if rank < mid]))
        high = int(RNG.choice([rank for rank in ranks if rank > mid]))
        v_ranks = [low, high]
        RNG.shuffle(v_ranks)
        return [order[mid]], [order[v_ranks[0]], order[v_ranks[1]]]

    if (h_count, v_count) == (2, 1):
        mid = int(RNG.choice([1, 2]))
        low = int(RNG.choice([rank for rank in ranks if rank < mid]))
        high = int(RNG.choice([rank for rank in ranks if rank > mid]))
        h_ranks = [low, high]
        RNG.shuffle(h_ranks)
        return [order[h_ranks[0]], order[h_ranks[1]]], [order[mid]]

    if (h_count, v_count) == (2, 2):
        for _ in range(20):
            perm = list(RNG.permutation(4))
            h_ranks = perm[:2]
            v_ranks = perm[2:]
            if min(h_ranks) < max(v_ranks) and min(v_ranks) < max(h_ranks):
                return [order[h_ranks[0]], order[h_ranks[1]]], [order[v_ranks[0]], order[v_ranks[1]]]
        return None

    return None


def _line_ids(rows: list[int], cols: list[int], width: int) -> tuple[dict[int, int], dict[int, int]]:
    row_ids = {r: r * width for r in rows}
    col_ids = {c: c for c in cols}
    return row_ids, col_ids


def _render_case(
    shape: tuple[int, int],
    bg: int,
    order: list[int],
    rows: list[int],
    cols: list[int],
    h_colors: list[int],
    v_colors: list[int],
) -> tuple[np.ndarray, np.ndarray]:
    h, w = shape
    inp = np.full((h, w), bg, dtype=int)
    out = np.full((h, w), bg, dtype=int)

    tl, tr, br, bl = order
    inp[0:2, 0:2] = tl
    inp[0:2, w - 2 : w] = tr
    inp[h - 2 : h, w - 2 : w] = br
    inp[h - 2 : h, 0:2] = bl
    out[:] = inp

    row_colors = dict(zip(rows, h_colors))
    col_colors = dict(zip(cols, v_colors))
    row_ids, col_ids = _line_ids(rows, cols, w)
    rank = {color: idx for idx, color in enumerate(order)}

    for r, color in row_colors.items():
        inp[r, :] = color
        out[r, :] = color
    for c, color in col_colors.items():
        inp[:, c] = color
        out[:, c] = color

    for r in rows:
        for c in cols:
            h_color = row_colors[r]
            v_color = col_colors[c]
            inp[r, c] = h_color if row_ids[r] < col_ids[c] else v_color
            out[r, c] = h_color if rank[h_color] < rank[v_color] else v_color

    inp[0:2, 0:2] = tl
    inp[0:2, w - 2 : w] = tr
    inp[h - 2 : h, w - 2 : w] = br
    inp[h - 2 : h, 0:2] = bl
    out[0:2, 0:2] = tl
    out[0:2, w - 2 : w] = tr
    out[h - 2 : h, w - 2 : w] = br
    out[h - 2 : h, 0:2] = bl
    return inp, out


def _has_both_crossing_branches(out: np.ndarray, rows: list[int], cols: list[int], row_colors: list[int], col_colors: list[int]) -> bool:
    changed = 0
    unchanged = 0
    row_map = dict(zip(rows, row_colors))
    col_map = dict(zip(cols, col_colors))
    for r in rows:
        for c in cols:
            if int(out[r, c]) == int(col_map[c]):
                unchanged += 1
            elif int(out[r, c]) == int(row_map[r]):
                changed += 1
    return changed > 0 and unchanged > 0


def _gen_attempt() -> np.ndarray | None:
    h = int(RNG.integers(10, 27))
    w = int(RNG.integers(10, 27))
    h_count, v_count = LINE_LAYOUTS[int(RNG.integers(len(LINE_LAYOUTS)))]

    rows = _sample_positions(h, h_count)
    cols = _sample_positions(w, v_count)
    if rows is None or cols is None:
        return None

    colors = list(RNG.choice(PALETTE, size=5, replace=False))
    bg = int(colors[0])
    order = [int(color) for color in colors[1:5]]
    assigned = _assign_colors(order, h_count, v_count)
    if assigned is None:
        return None
    h_colors, v_colors = assigned

    inp, expected = _render_case((h, w), bg, order, rows, cols, h_colors, v_colors)
    if not _has_both_crossing_branches(expected, rows, cols, h_colors, v_colors):
        return None
    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid crossing-priority puzzle")
