from __future__ import annotations

import random

import numpy as np


BLACK = 0
GRAY = 5
ARC_LIMIT = 30
COLOR_POOL = (1, 2, 3, 4, 6, 7, 8, 9)
GEN_ATTEMPTS = 400

DIRS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}

ARROW_MASKS = {
    "up": np.array(
        [
            [1, 1, 1],
            [0, 1, 0],
            [0, 1, 0],
        ],
        dtype=bool,
    ),
    "down": np.array(
        [
            [0, 1, 0],
            [0, 1, 0],
            [1, 1, 1],
        ],
        dtype=bool,
    ),
    "left": np.array(
        [
            [1, 0, 0],
            [1, 1, 1],
            [1, 0, 0],
        ],
        dtype=bool,
    ),
    "right": np.array(
        [
            [0, 0, 1],
            [1, 1, 1],
            [0, 0, 1],
        ],
        dtype=bool,
    ),
}


def _dominant_color(values: np.ndarray) -> int:
    flat = list(map(int, np.asarray(values).reshape(-1)))
    assert flat, "expected at least one value"
    counts: dict[int, int] = {}
    for value in flat:
        counts[value] = counts.get(value, 0) + 1
    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    assert len(ranked) == 1 or ranked[0][1] > ranked[1][1], "expected a unique dominant color"
    return int(ranked[0][0])


def _spans(length: int, separators: list[int]) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    start = 0
    for sep in separators:
        assert start < sep, "separator rows and columns must leave nonempty cells"
        spans.append((start, sep - 1))
        start = sep + 1
    assert start < length, "separator rows and columns must leave a final nonempty cell"
    spans.append((start, length - 1))
    return spans


def _partition_grid(g: np.ndarray) -> tuple[int, list[tuple[int, int]], list[tuple[int, int]]]:
    height, width = map(int, g.shape)
    candidates: list[tuple[int, list[tuple[int, int]], list[tuple[int, int]], int]] = []

    for color in map(int, np.unique(g)):
        rows = [r for r in range(height) if np.all(g[r, :] == color)]
        cols = [c for c in range(width) if np.all(g[:, c] == color)]
        if not rows or not cols:
            continue
        try:
            row_spans = _spans(height, rows)
            col_spans = _spans(width, cols)
        except AssertionError:
            continue
        row_sizes = {r1 - r0 + 1 for r0, r1 in row_spans}
        col_sizes = {c1 - c0 + 1 for c0, c1 in col_spans}
        if len(row_sizes) != 1 or len(col_sizes) != 1:
            continue
        score = len(rows) + len(cols)
        candidates.append((color, row_spans, col_spans, score))

    assert candidates, "expected separator rows and columns"
    best_score = max(score for _color, _rs, _cs, score in candidates)
    winners = [(color, rs, cs) for color, rs, cs, score in candidates if score == best_score]
    assert len(winners) == 1, "separator color must be uniquely identifiable"
    return winners[0]


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _arrow_direction(mask: np.ndarray) -> str | None:
    box = _bbox(mask)
    if box is None:
        return None
    r0, c0, r1, c1 = box
    patch = mask[r0 : r1 + 1, c0 : c1 + 1]
    if patch.shape != (3, 3) or int(patch.sum()) != 5:
        return None
    for direction, template in ARROW_MASKS.items():
        if np.array_equal(patch, template):
            return direction
    return None


def _cell_kind(cell: np.ndarray, bg: int) -> dict[str, object]:
    colors = [int(color) for color in np.unique(cell) if int(color) != bg]
    if not colors:
        return {"kind": "empty"}
    assert len(colors) == 1, "each nonempty cell must contain a single monochrome object"
    color = colors[0]
    mask = cell == color
    direction = _arrow_direction(mask)
    if direction is not None:
        return {"kind": "arrow", "color": color, "direction": direction}
    return {"kind": "pattern", "color": color, "mask": mask.copy()}


def _flip_mask(mask: np.ndarray, direction: str) -> np.ndarray:
    if direction in {"left", "right"}:
        return np.fliplr(mask)
    return np.flipud(mask)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Replace each arrow tile with a mirrored copy of the neighboring motif tile and recolor it with the arrow color.

    Input structure:
    - Black background cells are partitioned by full grey separator rows and columns into equal-sized rectangular tiles.
    - Some tiles contain one monochrome motif on the black background.
    - Some tiles contain one monochrome 3x3 arrow made from a length-3 shaft plus two tip-side pixels, pointing up, down, left, or right.
    - Every arrow tile has exactly one adjacent motif tile on the arrow's tail side across a shared separator line.

    Rule:
    - Detect the tile lattice from the full separator rows and columns.
    - For each arrow tile, use the adjacent motif tile opposite the arrow direction as the source tile.
    - Mirror the source motif within the full tile window: left/right arrows trigger a left-right reflection, while up/down arrows trigger an up-down reflection.
    - Clear the arrow tile and redraw the mirrored motif there using the arrow's own color. Keep the original motif tiles unchanged.

    Key insight:
    - The arrow encodes three independent pieces of information at once: which neighboring tile is the source, which reflection axis to use, and which color to repaint into the destination tile.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a nonempty 2D grid"

    separator, row_spans, col_spans = _partition_grid(g)
    bg = _dominant_color(g[g != separator])
    assert bg != separator, "background and separator colors must differ"

    cell_infos: list[list[dict[str, object]]] = []
    for r0, r1 in row_spans:
        row_info: list[dict[str, object]] = []
        for c0, c1 in col_spans:
            cell = g[r0 : r1 + 1, c0 : c1 + 1]
            row_info.append(_cell_kind(cell, bg))
        cell_infos.append(row_info)

    out = g.copy()
    rows = len(row_spans)
    cols = len(col_spans)
    for rr in range(rows):
        for cc in range(cols):
            info = cell_infos[rr][cc]
            if info["kind"] != "arrow":
                continue

            direction = str(info["direction"])
            dr, dc = DIRS[direction]
            src_r, src_c = rr - dr, cc - dc
            assert 0 <= src_r < rows and 0 <= src_c < cols, "source tile must stay in bounds"

            adjacent_patterns = []
            for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                if 0 <= nr < rows and 0 <= nc < cols and cell_infos[nr][nc]["kind"] == "pattern":
                    adjacent_patterns.append((nr, nc))
            assert adjacent_patterns == [(src_r, src_c)], "each arrow must have exactly one adjacent source motif"

            sr0, sr1 = row_spans[src_r]
            sc0, sc1 = col_spans[src_c]
            source_tile = g[sr0 : sr1 + 1, sc0 : sc1 + 1]
            source_mask = source_tile != bg
            mirrored = _flip_mask(source_mask, direction)

            ar0, ar1 = row_spans[rr]
            ac0, ac1 = col_spans[cc]
            dest = out[ar0 : ar1 + 1, ac0 : ac1 + 1]
            dest[:] = bg
            dest[mirrored] = int(info["color"])

    return out


def _neighbors4(r: int, c: int, height: int, width: int):
    if r > 0:
        yield r - 1, c
    if r + 1 < height:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < width:
        yield r, c + 1


def _random_connected_mask(height: int, width: int, area: int) -> np.ndarray:
    start = (random.randrange(height), random.randrange(width))
    cells = {start}
    frontier = [start]
    while len(cells) < area:
        r, c = random.choice(frontier)
        choices = [
            (nr, nc)
            for nr, nc in _neighbors4(r, c, height, width)
            if (nr, nc) not in cells
        ]
        if not choices:
            frontier = list(cells)
            continue
        cell = random.choice(choices)
        cells.add(cell)
        frontier.append(cell)

    mask = np.zeros((height, width), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    return mask


def _make_pattern_mask(cell_h: int, cell_w: int, direction: str) -> np.ndarray | None:
    max_h = min(4, cell_h - 2)
    max_w = min(4, cell_w - 2)
    if max_h < 2 or max_w < 2:
        return None

    for _ in range(100):
        box_h = random.randint(2, max_h)
        box_w = random.randint(2, max_w)
        area = random.randint(3, min(6, box_h * box_w))
        top = random.randint(1, cell_h - box_h - 1)
        left = random.randint(1, cell_w - box_w - 1)

        inner = _random_connected_mask(box_h, box_w, area)
        full = np.zeros((cell_h, cell_w), dtype=bool)
        full[top : top + box_h, left : left + box_w] = inner

        if _arrow_direction(full) is not None:
            continue
        flipped = _flip_mask(full, direction)
        if np.array_equal(full, flipped):
            continue
        return full
    return None


def _place_arrow(tile: np.ndarray, direction: str, color: int) -> bool:
    arrow = ARROW_MASKS[direction]
    cell_h, cell_w = tile.shape
    if cell_h < 5 or cell_w < 5:
        return False
    top = random.randint(1, cell_h - 4)
    left = random.randint(1, cell_w - 4)
    tile[top : top + 3, left : left + 3][arrow] = color
    return True


def _layout_options() -> list[tuple[int, int, int, int]]:
    options = []
    for cell_h in (5, 6, 7):
        for cell_w in (5, 6, 7):
            for rows in range(2, 6):
                for cols in range(2, 6):
                    height = rows * cell_h + (rows - 1)
                    width = cols * cell_w + (cols - 1)
                    if height <= ARC_LIMIT and width <= ARC_LIMIT:
                        options.append((rows, cols, cell_h, cell_w))
    return options


LAYOUT_OPTIONS = _layout_options()


def _build_grid(rows: int, cols: int, cell_h: int, cell_w: int) -> tuple[np.ndarray, list[tuple[int, int]], list[tuple[int, int]]]:
    height = rows * cell_h + (rows - 1)
    width = cols * cell_w + (cols - 1)
    g = np.full((height, width), BLACK, dtype=int)
    for r in range(cell_h, height, cell_h + 1):
        g[r, :] = GRAY
    for c in range(cell_w, width, cell_w + 1):
        g[:, c] = GRAY

    row_spans = [(r * (cell_h + 1), r * (cell_h + 1) + cell_h - 1) for r in range(rows)]
    col_spans = [(c * (cell_w + 1), c * (cell_w + 1) + cell_w - 1) for c in range(cols)]
    return g, row_spans, col_spans


def _can_place_pair(
    occupied: set[tuple[int, int]],
    source: tuple[int, int],
    arrow: tuple[int, int],
    rows: int,
    cols: int,
) -> bool:
    pair = {source, arrow}
    if occupied & pair:
        return False
    for r, c in pair:
        for nr, nc in _neighbors4(r, c, rows, cols):
            if (nr, nc) in occupied and (nr, nc) not in pair:
                return False
    return True


def _select_pairs(rows: int, cols: int) -> list[tuple[tuple[int, int], tuple[int, int], str]] | None:
    candidates = []
    for sr in range(rows):
        for sc in range(cols):
            for direction, (dr, dc) in DIRS.items():
                ar, ac = sr + dr, sc + dc
                if 0 <= ar < rows and 0 <= ac < cols:
                    candidates.append(((sr, sc), (ar, ac), direction))
    random.shuffle(candidates)

    max_pairs = min(4, rows * cols // 2)
    target_pairs = random.randint(1, max_pairs)
    occupied: set[tuple[int, int]] = set()
    pairs: list[tuple[tuple[int, int], tuple[int, int], str]] = []
    for source, arrow, direction in candidates:
        if not _can_place_pair(occupied, source, arrow, rows, cols):
            continue
        pairs.append((source, arrow, direction))
        occupied.add(source)
        occupied.add(arrow)
        if len(pairs) == target_pairs:
            return pairs
    return None


def _safe_decoy_cells(
    rows: int,
    cols: int,
    occupied: set[tuple[int, int]],
) -> list[tuple[int, int]]:
    candidates = []
    for r in range(rows):
        for c in range(cols):
            if (r, c) in occupied:
                continue
            if any((nr, nc) in occupied for nr, nc in _neighbors4(r, c, rows, cols)):
                continue
            candidates.append((r, c))
    return candidates


def _render_expected(
    inp: np.ndarray,
    row_spans: list[tuple[int, int]],
    col_spans: list[tuple[int, int]],
    pairs: list[tuple[tuple[int, int], tuple[int, int], str, int]],
) -> np.ndarray:
    out = inp.copy()
    for (sr, sc), (ar, ac), direction, arrow_color in pairs:
        sr0, sr1 = row_spans[sr]
        sc0, sc1 = col_spans[sc]
        ar0, ar1 = row_spans[ar]
        ac0, ac1 = col_spans[ac]
        source_mask = inp[sr0 : sr1 + 1, sc0 : sc1 + 1] != BLACK
        flipped = _flip_mask(source_mask, direction)
        dest = out[ar0 : ar1 + 1, ac0 : ac1 + 1]
        dest[:] = BLACK
        dest[flipped] = arrow_color
    return out


def _generate_attempt() -> np.ndarray | None:
    rows, cols, cell_h, cell_w = random.choice(LAYOUT_OPTIONS)
    pairs = _select_pairs(rows, cols)
    if pairs is None:
        return None

    inp, row_spans, col_spans = _build_grid(rows, cols, cell_h, cell_w)
    occupied = {cell for pair in pairs for cell in pair[:2]}
    rendered_pairs: list[tuple[tuple[int, int], tuple[int, int], str, int]] = []

    for source, arrow, direction in pairs:
        source_color = random.choice(COLOR_POOL)
        arrow_choices = [color for color in COLOR_POOL if color != source_color]
        arrow_color = random.choice(arrow_choices)

        mask = _make_pattern_mask(cell_h, cell_w, direction)
        if mask is None:
            return None

        sr0, sr1 = row_spans[source[0]]
        sc0, sc1 = col_spans[source[1]]
        source_tile = inp[sr0 : sr1 + 1, sc0 : sc1 + 1]
        source_tile[mask] = source_color

        ar0, ar1 = row_spans[arrow[0]]
        ac0, ac1 = col_spans[arrow[1]]
        arrow_tile = inp[ar0 : ar1 + 1, ac0 : ac1 + 1]
        if not _place_arrow(arrow_tile, direction, arrow_color):
            return None

        rendered_pairs.append((source, arrow, direction, arrow_color))

    decoy_cells = _safe_decoy_cells(rows, cols, occupied)
    random.shuffle(decoy_cells)
    decoy_count = random.randint(0, min(2, len(decoy_cells)))
    for cell in decoy_cells[:decoy_count]:
        direction = random.choice(tuple(DIRS))
        mask = _make_pattern_mask(cell_h, cell_w, direction)
        if mask is None:
            return None
        color = random.choice(COLOR_POOL)
        r0, r1 = row_spans[cell[0]]
        c0, c1 = col_spans[cell[1]]
        tile = inp[r0 : r1 + 1, c0 : c1 + 1]
        tile[mask] = color

    expected = _render_expected(inp, row_spans, col_spans, rendered_pairs)
    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid arrow-mirror input")
