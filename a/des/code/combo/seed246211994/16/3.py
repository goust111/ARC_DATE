from __future__ import annotations

from collections import Counter
import random

import numpy as np


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for idx, value in enumerate(mask.tolist()):
        if value:
            if start is None:
                start = idx
        elif start is not None:
            runs.append((start, idx - 1))
            start = None
    if start is not None:
        runs.append((start, len(mask) - 1))
    return runs


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(mask)
    assert coords.size > 0, "expected at least one foreground cell"
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _unique_mode(values: np.ndarray) -> int:
    colors, counts = np.unique(np.asarray(values, dtype=int), return_counts=True)
    best = int(np.argmax(counts))
    assert int(np.sum(counts == counts[best])) == 1, "expected a unique dominant color"
    return int(colors[best])


def _dominant_foreground_color(values: np.ndarray) -> int:
    counts = Counter(map(int, np.asarray(values, dtype=int).reshape(-1)))
    assert counts, "expected at least one foreground pixel"
    best_count = max(counts.values())
    best = [color for color, count in counts.items() if count == best_count]
    assert len(best) == 1, "expected a unique repeated motif color"
    return int(best[0])


def _crop_non_background(grid: np.ndarray, bg: int) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(grid != bg)
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _find_separator_color(grid: np.ndarray, bg: int) -> tuple[int, list[tuple[int, int]], list[tuple[int, int]]]:
    candidates: list[tuple[int, list[tuple[int, int]], list[tuple[int, int]]]] = []
    for color in np.unique(grid):
        color = int(color)
        if color == bg:
            continue
        row_runs = _runs(np.all(grid == color, axis=1))
        col_runs = _runs(np.all(grid == color, axis=0))
        if row_runs and col_runs:
            candidates.append((color, row_runs, col_runs))
    assert len(candidates) == 1, "expected exactly one separator color"
    return candidates[0]


def _tile_spans(length: int, separator_runs: list[tuple[int, int]]) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    start = 0
    for sep0, sep1 in separator_runs:
        assert start < sep0, "separator bands must leave a nonempty tile"
        spans.append((start, sep0 - 1))
        start = sep1 + 1
    assert start < length, "separator bands must leave a final tile"
    spans.append((start, length - 1))
    return spans


def _slot_positions(rows: int, cols: int) -> list[tuple[int, int]]:
    return [(1 + 2 * r, 1 + 2 * c) for r in range(rows) for c in range(cols)]


def _render_lattice(
    rows: int,
    cols: int,
    tile_h: int,
    tile_w: int,
    bg: int,
    separator: int,
    motif_mask: np.ndarray,
    cell_colors: np.ndarray,
    separator_width: int,
) -> np.ndarray:
    out_h = rows * tile_h + (rows - 1) * separator_width
    out_w = cols * tile_w + (cols - 1) * separator_width
    out = np.full((out_h, out_w), bg, dtype=int)

    for boundary in range(1, rows):
        r0 = boundary * tile_h + (boundary - 1) * separator_width
        out[r0 : r0 + separator_width, :] = separator
    for boundary in range(1, cols):
        c0 = boundary * tile_w + (boundary - 1) * separator_width
        out[:, c0 : c0 + separator_width] = separator

    for r in range(rows):
        for c in range(cols):
            r0 = r * (tile_h + separator_width)
            c0 = c * (tile_w + separator_width)
            tile = out[r0 : r0 + tile_h, c0 : c0 + tile_w]
            tile[motif_mask] = int(cell_colors[r, c])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode the uniquely marked tile of a separator grid and rebuild the compact recolored lattice.

    Input structure:
    - A dominant background surrounds a rectangular lattice of equal tiles.
    - Full separator-color rows and columns split the lattice into cells.
    - Every cell contains the same motif drawn in one repeated motif color.
    - Exactly one cell has some odd-row/odd-column motif pixels recolored as markers.

    Rule:
    - If the lattice has R rows and C columns, every tile is (2R+1) x (2C+1).
    - Read the marked tile's odd-row/odd-column intersections as an R x C color table.
    - Positions still showing the motif color mean "use the motif color".
    - Rebuild the lattice with 1-pixel separator lines, drawing the same motif in each cell
      using the color from the table entry at that cell's lattice position.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a nonempty 2D grid"

    bg = _unique_mode(g.reshape(-1))
    board = _crop_non_background(g, bg)
    separator, row_sep_runs, col_sep_runs = _find_separator_color(board, bg)
    row_spans = _tile_spans(board.shape[0], row_sep_runs)
    col_spans = _tile_spans(board.shape[1], col_sep_runs)

    tile_heights = {r1 - r0 + 1 for r0, r1 in row_spans}
    tile_widths = {c1 - c0 + 1 for c0, c1 in col_spans}
    assert len(tile_heights) == 1 and len(tile_widths) == 1, "all tiles must share one size"
    tile_h = tile_heights.pop()
    tile_w = tile_widths.pop()

    rows = len(row_spans)
    cols = len(col_spans)
    assert tile_h == 2 * rows + 1, "tile height must encode the lattice row count"
    assert tile_w == 2 * cols + 1, "tile width must encode the lattice column count"

    foreground = board[(board != bg) & (board != separator)]
    motif_color = _dominant_foreground_color(foreground)

    motif_mask: np.ndarray | None = None
    marked_tiles: list[np.ndarray] = []
    slot_mask = np.zeros((tile_h, tile_w), dtype=bool)
    for sr, sc in _slot_positions(rows, cols):
        slot_mask[sr, sc] = True

    for r0, r1 in row_spans:
        for c0, c1 in col_spans:
            tile = board[r0 : r1 + 1, c0 : c1 + 1]
            assert tile.shape == (tile_h, tile_w)
            assert not np.any(tile == separator), "separator pixels must stay between tiles"

            support = (tile != bg) & (tile != separator)
            assert np.any(support), "each tile must contain the shared motif"
            if motif_mask is None:
                motif_mask = support
            else:
                assert np.array_equal(support, motif_mask), "all tiles must share the same motif mask"

            extra_colors = {int(color) for color in np.unique(tile) if int(color) not in {bg, separator, motif_color}}
            if extra_colors:
                marked_tiles.append(tile.copy())
            else:
                assert np.all(tile[support] == motif_color), "unmarked tiles must use only the motif color"
                assert np.all(tile[~support] == bg), "non-motif cells must be background"

    assert motif_mask is not None
    assert len(marked_tiles) == 1, "expected exactly one marked tile"
    marked = marked_tiles[0]

    marker_mask = (marked != bg) & (marked != separator) & (marked != motif_color)
    assert np.all(marker_mask <= slot_mask), "marker pixels must stay on odd-row/odd-column slots"
    assert np.all(slot_mask <= motif_mask), "every legend slot must belong to the motif"
    assert np.all(marked[motif_mask & ~slot_mask] == motif_color), "non-slot motif cells must keep the motif color"
    assert np.all(marked[~motif_mask] == bg), "off-motif cells must stay background"

    color_map = np.empty((rows, cols), dtype=int)
    for r in range(rows):
        for c in range(cols):
            value = int(marked[1 + 2 * r, 1 + 2 * c])
            assert value not in {bg, separator}, "legend slots must name visible motif colors"
            color_map[r, c] = value

    return _render_lattice(
        rows=rows,
        cols=cols,
        tile_h=tile_h,
        tile_w=tile_w,
        bg=bg,
        separator=separator,
        motif_mask=motif_mask,
        cell_colors=color_map,
        separator_width=1,
    )


# --------------- generator ---------------

COLOR_POOL = tuple(range(10))
LATTICE_ROWS_RANGE = (2, 4)
LATTICE_COLS_RANGE = (2, 4)
INPUT_SEPARATOR_WIDTH_CHOICES = (1, 2)
OUTER_PADDING_RANGE = (0, 3)
MAX_EXTRA_COLORS = 4
GENERATION_ATTEMPTS = 400


def _coarse_neighbors(r: int, c: int, rows: int, cols: int):
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        rr = r + dr
        cc = c + dc
        if 0 <= rr < rows and 0 <= cc < cols:
            yield rr, cc


def _adjacency_edges(rows: int, cols: int) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    edges: list[tuple[tuple[int, int], tuple[int, int]]] = []
    for r in range(rows):
        for c in range(cols):
            if r + 1 < rows:
                edges.append(((r, c), (r + 1, c)))
            if c + 1 < cols:
                edges.append(((r, c), (r, c + 1)))
    return edges


def _edge_key(a: tuple[int, int], b: tuple[int, int]) -> tuple[tuple[int, int], tuple[int, int]]:
    return (a, b) if a <= b else (b, a)


def _tree_edges(rows: int, cols: int) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    start = (random.randrange(rows), random.randrange(cols))
    visited = {start}
    frontier = [(start, neigh) for neigh in _coarse_neighbors(start[0], start[1], rows, cols)]
    edges: list[tuple[tuple[int, int], tuple[int, int]]] = []

    while frontier:
        idx = random.randrange(len(frontier))
        a, b = frontier.pop(idx)
        if b in visited:
            continue
        edges.append((a, b))
        visited.add(b)
        for neigh in _coarse_neighbors(b[0], b[1], rows, cols):
            if neigh not in visited:
                frontier.append((b, neigh))

    assert len(visited) == rows * cols, "coarse slot graph must be connected"
    return edges


def _draw_connector(mask: np.ndarray, a: tuple[int, int], b: tuple[int, int]) -> None:
    ar, ac = a
    br, bc = b
    mask[1 + ar + br, 1 + ac + bc] = True


def _build_motif(rows: int, cols: int) -> np.ndarray:
    tile_h = 2 * rows + 1
    tile_w = 2 * cols + 1
    mask = np.zeros((tile_h, tile_w), dtype=bool)

    for sr, sc in _slot_positions(rows, cols):
        mask[sr, sc] = True

    all_edges = _adjacency_edges(rows, cols)
    tree = _tree_edges(rows, cols)
    used = {_edge_key(a, b) for a, b in tree}
    for a, b in tree:
        _draw_connector(mask, a, b)

    for a, b in all_edges:
        if _edge_key(a, b) in used:
            continue
        if random.random() < 0.35:
            _draw_connector(mask, a, b)

    top_c = random.randrange(cols)
    bottom_c = random.randrange(cols)
    left_r = random.randrange(rows)
    right_r = random.randrange(rows)
    mask[0, 1 + 2 * top_c] = True
    mask[tile_h - 1, 1 + 2 * bottom_c] = True
    mask[1 + 2 * left_r, 0] = True
    mask[1 + 2 * right_r, tile_w - 1] = True

    return mask


def _sample_color_map(rows: int, cols: int, base: int, forbidden: set[int]) -> np.ndarray:
    available = [color for color in COLOR_POOL if color not in forbidden]
    n_cells = rows * cols
    n_marked = random.randint(1, n_cells - 1)
    extra_count = random.randint(1, min(MAX_EXTRA_COLORS, n_marked, len(available)))
    extras = random.sample(available, extra_count)

    marked_positions = random.sample(range(n_cells), n_marked)
    values = extras + [random.choice(extras) for _ in range(n_marked - extra_count)]
    random.shuffle(values)

    color_map = np.full((rows, cols), base, dtype=int)
    for flat_index, color in zip(marked_positions, values):
        r, c = divmod(flat_index, cols)
        color_map[r, c] = int(color)
    return color_map


def _pad_with_background(grid: np.ndarray, bg: int) -> np.ndarray:
    top = random.randint(*OUTER_PADDING_RANGE)
    bottom = random.randint(*OUTER_PADDING_RANGE)
    left = random.randint(*OUTER_PADDING_RANGE)
    right = random.randint(*OUTER_PADDING_RANGE)
    return np.pad(grid, ((top, bottom), (left, right)), constant_values=bg)


def _render_input(
    rows: int,
    cols: int,
    bg: int,
    separator: int,
    motif_color: int,
    motif_mask: np.ndarray,
    color_map: np.ndarray,
    separator_width: int,
) -> np.ndarray:
    tile_h, tile_w = motif_mask.shape
    base_colors = np.full((rows, cols), motif_color, dtype=int)
    lattice = _render_lattice(
        rows=rows,
        cols=cols,
        tile_h=tile_h,
        tile_w=tile_w,
        bg=bg,
        separator=separator,
        motif_mask=motif_mask,
        cell_colors=base_colors,
        separator_width=separator_width,
    )

    legend_r = random.randrange(rows)
    legend_c = random.randrange(cols)
    r0 = legend_r * (tile_h + separator_width)
    c0 = legend_c * (tile_w + separator_width)
    tile = lattice[r0 : r0 + tile_h, c0 : c0 + tile_w]
    for lr in range(rows):
        for lc in range(cols):
            tile[1 + 2 * lr, 1 + 2 * lc] = int(color_map[lr, lc])
    return lattice


def _gen_attempt() -> np.ndarray | None:
    rows = random.randint(*LATTICE_ROWS_RANGE)
    cols = random.randint(*LATTICE_COLS_RANGE)
    separator_width = random.choice(INPUT_SEPARATOR_WIDTH_CHOICES)

    bg, separator, motif_color = random.sample(COLOR_POOL, 3)
    motif_mask = _build_motif(rows, cols)
    color_map = _sample_color_map(rows, cols, motif_color, {bg, separator, motif_color})

    inp = _pad_with_background(
        _render_input(
            rows=rows,
            cols=cols,
            bg=bg,
            separator=separator,
            motif_color=motif_color,
            motif_mask=motif_mask,
            color_map=color_map,
            separator_width=separator_width,
        ),
        bg=bg,
    )
    expected = _render_lattice(
        rows=rows,
        cols=cols,
        tile_h=motif_mask.shape[0],
        tile_w=motif_mask.shape[1],
        bg=bg,
        separator=separator,
        motif_mask=motif_mask,
        cell_colors=color_map,
        separator_width=1,
    )

    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    if np.array_equal(inp, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
