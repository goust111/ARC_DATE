import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5

MARKER_TO_ROT_K = {
    RED: 0,
    GREEN: 3,   # 90 degrees clockwise
    BLUE: 2,
    YELLOW: 1,  # 90 degrees counterclockwise
}
MARKER_COLORS = tuple(MARKER_TO_ROT_K)
PATTERN_COLORS = (6, 7, 8, 9)
CORNER_NAMES = ("tl", "tr", "bl", "br")

PANEL_SIZE_CHOICES = (6, 7, 8)
PANEL_ROW_CHOICES = (2, 3)
PANEL_COL_CHOICES = (2, 3)
SEPARATOR_WIDTH_CHOICES = (1, 2)
BORDER_WIDTH_CHOICES = (0, 1)
MAX_GEN_ATTEMPTS = 300

_RNG = random.Random()


def _corner_positions(size: int) -> dict[str, tuple[int, int]]:
    return {
        "tl": (0, 0),
        "tr": (0, size - 2),
        "bl": (size - 2, 0),
        "br": (size - 2, size - 2),
    }


def _false_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start = None
    for idx, flag in enumerate(mask.tolist()):
        if not flag and start is None:
            start = idx
        elif flag and start is not None:
            runs.append((start, idx))
            start = None
    if start is not None:
        runs.append((start, len(mask)))
    return runs


def _marker_candidates(tile: np.ndarray) -> list[tuple[str, int, int, int]]:
    size = int(tile.shape[0])
    candidates: list[tuple[str, int, int, int]] = []
    for name, (r0, c0) in _corner_positions(size).items():
        block = tile[r0 : r0 + 2, c0 : c0 + 2]
        if block.shape != (2, 2):
            continue
        color = int(block[0, 0])
        if color not in MARKER_TO_ROT_K:
            continue
        if np.all(block == color):
            candidates.append((name, r0, c0, color))
    return candidates


def _find_marker(tile: np.ndarray) -> tuple[str, int, int, int]:
    markers = _marker_candidates(tile)
    assert len(markers) == 1, f"expected exactly one 2x2 corner marker, found {len(markers)}"
    return markers[0]


def _infer_layout(grid: np.ndarray) -> tuple[int, list[tuple[int, int]], list[tuple[int, int]], int]:
    best = None
    for color in map(int, np.unique(grid)):
        row_sep = np.all(grid == color, axis=1)
        col_sep = np.all(grid == color, axis=0)
        if not row_sep.any() or not col_sep.any():
            continue

        row_spans = _false_runs(row_sep)
        col_spans = _false_runs(col_sep)
        if not row_spans or not col_spans:
            continue

        row_sizes = {r1 - r0 for r0, r1 in row_spans}
        col_sizes = {c1 - c0 for c0, c1 in col_spans}
        if len(row_sizes) != 1 or len(col_sizes) != 1:
            continue

        size = row_sizes.pop()
        if size != col_sizes.pop() or size < 4:
            continue
        if len(row_spans) * len(col_spans) < 2:
            continue

        try:
            for r0, r1 in row_spans:
                for c0, c1 in col_spans:
                    tile = grid[r0:r1, c0:c1]
                    assert tile.shape == (size, size)
                    _find_marker(tile)
        except AssertionError:
            continue

        score = (
            len(row_spans) * len(col_spans),
            int(row_sep.sum() + col_sep.sum()),
            size,
        )
        if best is None or score > best[0]:
            best = (score, color, row_spans, col_spans, size)

    assert best is not None, "failed to infer panel layout"
    _, color, row_spans, col_spans, size = best
    return color, row_spans, col_spans, size


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    _sep_color, row_spans, col_spans, _panel_size = _infer_layout(g)
    out = g.copy()
    for r0, r1 in row_spans:
        for c0, c1 in col_spans:
            tile = g[r0:r1, c0:c1]
            _corner, mr, mc, marker_color = _find_marker(tile)
            work = tile.copy()
            work[mr : mr + 2, mc : mc + 2] = BLACK
            out[r0:r1, c0:c1] = np.rot90(work, k=MARKER_TO_ROT_K[marker_color])
    return out


def _panel_spans(count: int, size: int, sep_w: int, border_w: int) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    pos = border_w
    for idx in range(count):
        spans.append((pos, pos + size))
        pos += size
        if idx < count - 1:
            pos += sep_w
    return spans


def _build_canvas(rows: int, cols: int, size: int, sep_w: int, border_w: int) -> tuple[np.ndarray, list[tuple[int, int]], list[tuple[int, int]]]:
    height = rows * size + (rows - 1) * sep_w + 2 * border_w
    width = cols * size + (cols - 1) * sep_w + 2 * border_w
    grid = np.full((height, width), BLACK, dtype=int)

    if border_w > 0:
        grid[:border_w, :] = GREY
        grid[-border_w:, :] = GREY
        grid[:, :border_w] = GREY
        grid[:, -border_w:] = GREY

    for ridx in range(1, rows):
        start = border_w + ridx * size + (ridx - 1) * sep_w
        grid[start : start + sep_w, :] = GREY
    for cidx in range(1, cols):
        start = border_w + cidx * size + (cidx - 1) * sep_w
        grid[:, start : start + sep_w] = GREY

    return grid, _panel_spans(rows, size, sep_w, border_w), _panel_spans(cols, size, sep_w, border_w)


def _neighbors(r: int, c: int, size: int) -> list[tuple[int, int]]:
    result: list[tuple[int, int]] = []
    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        nr, nc = r + dr, c + dc
        if 0 <= nr < size and 0 <= nc < size:
            result.append((nr, nc))
    return result


def _grow_component(safe: np.ndarray, occupied: np.ndarray, target_size: int) -> list[tuple[int, int]] | None:
    free = [tuple(map(int, pos)) for pos in np.argwhere(safe & (~occupied))]
    if not free:
        return None

    start = free[_RNG.randrange(len(free))]
    cells = [start]
    cell_set = {start}
    while len(cells) < target_size:
        frontier: list[tuple[int, int]] = []
        for r, c in cells:
            for nr, nc in _neighbors(r, c, int(safe.shape[0])):
                if not safe[nr, nc] or occupied[nr, nc] or (nr, nc) in cell_set:
                    continue
                frontier.append((nr, nc))
        if not frontier:
            break
        nxt = frontier[_RNG.randrange(len(frontier))]
        cell_set.add(nxt)
        cells.append(nxt)

    if len(cells) < 2:
        return None
    return cells


def _is_rotationally_distinct(tile: np.ndarray) -> bool:
    return all(not np.array_equal(tile, np.rot90(tile, k)) for k in (1, 2, 3))


def _make_pattern(size: int) -> np.ndarray | None:
    safe = np.ones((size, size), dtype=bool)
    for r0, c0 in _corner_positions(size).values():
        safe[r0 : r0 + 2, c0 : c0 + 2] = False

    for _ in range(80):
        tile = np.full((size, size), BLACK, dtype=int)
        occupied = np.zeros((size, size), dtype=bool)

        component_count = _RNG.randint(2, 4)
        colors = list(PATTERN_COLORS)
        _RNG.shuffle(colors)
        colors = colors[:component_count]

        ok = True
        for color in colors:
            comp_size = _RNG.randint(2, min(5, size))
            cells = _grow_component(safe, occupied, comp_size)
            if cells is None:
                ok = False
                break
            for r, c in cells:
                tile[r, c] = color
                occupied[r, c] = True

        if not ok:
            continue

        extras = [tuple(map(int, pos)) for pos in np.argwhere(safe & (~occupied))]
        _RNG.shuffle(extras)
        for idx in range(_RNG.randint(0, 2)):
            if idx >= len(extras):
                break
            r, c = extras[idx]
            tile[r, c] = colors[_RNG.randrange(len(colors))]
            occupied[r, c] = True

        if int(np.count_nonzero(tile)) < 6:
            continue
        if not _is_rotationally_distinct(tile):
            continue
        return tile

    return None


def _make_panel(size: int, marker_color: int, marker_corner: str) -> np.ndarray | None:
    for _ in range(80):
        tile = _make_pattern(size)
        if tile is None:
            continue
        r0, c0 = _corner_positions(size)[marker_corner]
        tile[r0 : r0 + 2, c0 : c0 + 2] = marker_color
        if len(_marker_candidates(tile)) == 1:
            return tile
    return None


def _gen_attempt() -> np.ndarray | None:
    rows = _RNG.choice(PANEL_ROW_CHOICES)
    cols = _RNG.choice(PANEL_COL_CHOICES)
    size = _RNG.choice(PANEL_SIZE_CHOICES)
    sep_w = _RNG.choice(SEPARATOR_WIDTH_CHOICES)
    border_w = _RNG.choice(BORDER_WIDTH_CHOICES)

    grid, row_spans, col_spans = _build_canvas(rows, cols, size, sep_w, border_w)
    panel_slots = [(r0, r1, c0, c1) for r0, r1 in row_spans for c0, c1 in col_spans]
    if len(panel_slots) < 4:
        return None

    marker_colors = [MARKER_COLORS[idx % len(MARKER_COLORS)] for idx in range(len(panel_slots))]
    _RNG.shuffle(marker_colors)

    for idx, (r0, r1, c0, c1) in enumerate(panel_slots):
        marker_corner = _RNG.choice(CORNER_NAMES)
        tile = _make_panel(size, marker_colors[idx], marker_corner)
        if tile is None:
            return None
        grid[r0:r1, c0:c1] = tile

    try:
        out = main(grid)
    except AssertionError:
        return None

    if out.shape != grid.shape:
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {MAX_GEN_ATTEMPTS} attempts")
