from __future__ import annotations

import itertools
import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
PINK = 6

MAX_GENERATION_ATTEMPTS = 400
_TILE_CACHE: dict[tuple[int, int], list[np.ndarray]] = {}


def _has_adjacent_pair(values: np.ndarray, color: int) -> bool:
    return bool(np.any((values[:-1] == color) & (values[1:] == color)))


def _detect_line_rows(grid: np.ndarray) -> list[int]:
    return [r for r in range(grid.shape[0]) if _has_adjacent_pair(grid[r], BLUE)]


def _detect_line_cols(grid: np.ndarray) -> list[int]:
    return [c for c in range(grid.shape[1]) if _has_adjacent_pair(grid[:, c], BLUE)]


def _complete_grid_lines(grid: np.ndarray, line_rows: list[int], line_cols: list[int]) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    for r in line_rows:
        row = out[r]
        row[row == BLACK] = RED
    for c in line_cols:
        col = out[:, c]
        col[col == BLACK] = RED
    return out


def _cell_boxes(line_rows: list[int], line_cols: list[int]) -> list[tuple[int, int, int, int]]:
    boxes: list[tuple[int, int, int, int]] = []
    for r0, r1 in zip(line_rows, line_rows[1:]):
        for c0, c1 in zip(line_cols, line_cols[1:]):
            inner_r0 = r0 + 1
            inner_r1 = r1 - 1
            inner_c0 = c0 + 1
            inner_c1 = c1 - 1
            if inner_r0 <= inner_r1 and inner_c0 <= inner_c1:
                boxes.append((inner_r0, inner_c0, inner_r1, inner_c1))
    return boxes


def _candidate_tile(cell: np.ndarray, row_period: int, col_period: int) -> np.ndarray | None:
    tile = np.full((row_period, col_period), -1, dtype=int)
    for r, c in np.argwhere(cell != PINK):
        rr = int(r) % row_period
        cc = int(c) % col_period
        value = int(cell[r, c])
        if tile[rr, cc] == -1:
            tile[rr, cc] = value
        elif tile[rr, cc] != value:
            return None
    if np.any(tile == -1):
        return None
    return tile


def _infer_minimal_tile(cell: np.ndarray) -> tuple[int, int, np.ndarray]:
    h, w = cell.shape
    candidates: list[tuple[int, int, int, np.ndarray]] = []
    for row_period in range(1, h + 1):
        for col_period in range(1, w + 1):
            tile = _candidate_tile(cell, row_period, col_period)
            if tile is not None:
                candidates.append((row_period * col_period, row_period, col_period, tile))
    assert candidates, "pattern cell must admit a periodic explanation"
    min_area = min(area for area, _rp, _cp, _tile in candidates)
    minimal = [(rp, cp, tile) for area, rp, cp, tile in candidates if area == min_area]
    assert len(minimal) == 1, "pattern cell must have a unique minimum-area period"
    return minimal[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete the broken blue grid lines, then repair each marked cell by tiling its own periodic pattern.

    Input structure:
    - A black background contains several incomplete blue horizontal and vertical grid lines.
    - A row is a true grid line iff it contains a horizontally adjacent blue pair; a column is a true grid line iff it contains a vertically adjacent blue pair.
    - The completed grid partitions the interior into rectangular cells; some cells are empty, while some contain red/green periodic content with pink missing-value markers.

    Rule:
    - Detect every valid blue grid row and column, and recolor each black gap on those rows or columns to red.
    - Use the completed line set as the cell boundary system; grid-line pixels themselves are not part of any cell.
    - Inside each cell that contains pink markers, infer the unique minimum-area row/column period from the visible non-pink pixels, then replace each pink pixel with the tile value at the same cell-relative residue.

    Key insight:
    - The task is a cascade of two different completions: first recover the structural grid, then use those recovered cell boundaries to infer a separate local periodic tile for each cell.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "input must be a 2D grid"

    line_rows = _detect_line_rows(g)
    line_cols = _detect_line_cols(g)
    assert len(line_rows) >= 2 and len(line_cols) >= 2, "completed grid must define at least one cell"

    out = _complete_grid_lines(g, line_rows, line_cols)
    for r0, c0, r1, c1 in _cell_boxes(line_rows, line_cols):
        cell = out[r0 : r1 + 1, c0 : c1 + 1]
        if not np.any(cell == PINK):
            continue
        visible_colors = set(map(int, np.unique(cell))) - {PINK}
        assert visible_colors - {BLACK, RED, GREEN} == set()
        assert visible_colors & {RED, GREEN}, "pink markers must appear only inside non-empty pattern cells"

        row_period, col_period, tile = _infer_minimal_tile(cell)
        for rr, cc in np.argwhere(cell == PINK):
            cell[int(rr), int(cc)] = tile[int(rr) % row_period, int(cc) % col_period]
    return out


def _primitive_period_area(tile: np.ndarray) -> int:
    h, w = tile.shape
    best = h * w
    for row_period in range(1, h + 1):
        for col_period in range(1, w + 1):
            ok = True
            for r in range(h):
                for c in range(w):
                    if int(tile[r, c]) != int(tile[r % row_period, c % col_period]):
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                best = min(best, row_period * col_period)
    return best


def _primitive_tiles(tile_h: int, tile_w: int) -> list[np.ndarray]:
    key = (tile_h, tile_w)
    if key in _TILE_CACHE:
        return _TILE_CACHE[key]

    tiles: list[np.ndarray] = []
    area = tile_h * tile_w
    for bits in itertools.product((RED, GREEN), repeat=area):
        if len(set(bits)) != 2:
            continue
        tile = np.array(bits, dtype=int).reshape(tile_h, tile_w)
        if _primitive_period_area(tile) == area:
            tiles.append(tile)
    _TILE_CACHE[key] = tiles
    return tiles


def _tile_signature(tile: np.ndarray) -> tuple[tuple[int, int], bytes]:
    return tile.shape, tile.tobytes()


def _tile_cell(tile: np.ndarray, height: int, width: int) -> np.ndarray:
    rr = np.arange(height)[:, None] % tile.shape[0]
    cc = np.arange(width)[None, :] % tile.shape[1]
    return tile[rr, cc].astype(int)


def _line_positions(cell_sizes: list[int]) -> list[int]:
    pos = 0
    lines = [0]
    for size in cell_sizes:
        pos += size + 1
        lines.append(pos)
    return lines


def _build_gap_mask(height: int, width: int, line_rows: list[int], line_cols: list[int]) -> np.ndarray:
    line_mask = np.zeros((height, width), dtype=bool)
    line_mask[line_rows, :] = True
    line_mask[:, line_cols] = True

    protected = np.zeros_like(line_mask)
    for r in line_rows:
        start = random.randint(0, width - 2)
        protected[r, start : start + 2] = True
    for c in line_cols:
        start = random.randint(0, height - 2)
        protected[start : start + 2, c] = True

    gap_prob = random.uniform(0.22, 0.42)
    gap_mask = line_mask & ~protected & (np.random.random((height, width)) < gap_prob)

    for r in line_rows:
        if not np.any(gap_mask[r]):
            candidates = [c for c in range(width) if not protected[r, c]]
            if candidates:
                gap_mask[r, random.choice(candidates)] = True
    for c in line_cols:
        if not np.any(gap_mask[:, c]):
            candidates = [r for r in range(height) if not protected[r, c]]
            if candidates:
                gap_mask[random.choice(candidates), c] = True

    assert all(_has_adjacent_pair((line_mask[r] & ~gap_mask[r]).astype(int), 1) for r in line_rows)
    assert all(_has_adjacent_pair((line_mask[:, c] & ~gap_mask[:, c]).astype(int), 1) for c in line_cols)
    assert np.any(gap_mask[line_rows, :]) and np.any(gap_mask[:, line_cols])
    return gap_mask


def _choose_tile_for_cell(
    cell_h: int,
    cell_w: int,
    require_repeat: bool,
    banned: set[tuple[tuple[int, int], bytes]] | None = None,
) -> np.ndarray | None:
    choices: list[np.ndarray] = []
    for tile_h in range(1, min(3, cell_h) + 1):
        for tile_w in range(1, min(3, cell_w) + 1):
            if tile_h == 1 and tile_w == 1:
                continue
            if require_repeat and tile_h == cell_h and tile_w == cell_w:
                continue
            for tile in _primitive_tiles(tile_h, tile_w):
                if banned is not None and _tile_signature(tile) in banned:
                    continue
                choices.append(tile)
    if not choices:
        return None
    return random.choice(choices).copy()


def _choose_missing_positions(cell_h: int, cell_w: int, tile_h: int, tile_w: int) -> list[tuple[int, int]]:
    residues: dict[tuple[int, int], list[tuple[int, int]]] = {}
    all_coords = [(r, c) for r in range(cell_h) for c in range(cell_w)]
    for r, c in all_coords:
        residues.setdefault((r % tile_h, c % tile_w), []).append((r, c))

    preserved = {random.choice(coords) for coords in residues.values()}
    candidates = [coord for coord in all_coords if coord not in preserved]
    if not candidates:
        return []

    max_missing = min(len(candidates), max(1, (cell_h * cell_w) // 4))
    count = random.randint(1, max_missing)
    return random.sample(candidates, count)


def _validate_instance(
    inp: np.ndarray,
    out: np.ndarray,
    line_rows: list[int],
    line_cols: list[int],
    cell_specs: list[dict],
    gap_mask: np.ndarray,
) -> bool:
    if _detect_line_rows(inp) != line_rows or _detect_line_cols(inp) != line_cols:
        return False
    try:
        actual = main(inp)
    except AssertionError:
        return False
    if not np.array_equal(actual, out):
        return False

    total_gaps = int(np.count_nonzero(gap_mask))
    if total_gaps < max(4, len(line_rows) + len(line_cols) - 2):
        return False

    empty_cells = 0
    pattern_signatures: set[tuple[tuple[int, int], bytes]] = set()
    seen_marker_cell = False
    for spec in cell_specs:
        r0, c0, r1, c1 = spec["box"]
        cell_in = inp[r0 : r1 + 1, c0 : c1 + 1]
        cell_out = out[r0 : r1 + 1, c0 : c1 + 1]
        if spec["tile"] is None:
            empty_cells += 1
            if np.any(cell_in != BLACK) or np.any(cell_out != BLACK):
                return False
            continue

        tile = spec["tile"]
        tiled = _tile_cell(tile, cell_out.shape[0], cell_out.shape[1])
        if not np.array_equal(cell_out, tiled):
            return False
        pattern_signatures.add(_tile_signature(tile))

        missing = spec["missing"]
        if missing:
            seen_marker_cell = True
            if not np.any(cell_in == PINK):
                return False
            try:
                row_period, col_period, inferred = _infer_minimal_tile(cell_in)
            except AssertionError:
                return False
            if (row_period, col_period) != tile.shape:
                return False
            if not np.array_equal(inferred, tile):
                return False
        else:
            if np.any(cell_in == PINK):
                return False

    return empty_cells >= 1 and seen_marker_cell and len(pattern_signatures) >= 2


def _gen_attempt() -> np.ndarray | None:
    row_cell_count = random.choice((2, 3))
    col_cell_count = random.choice((2, 3))
    cell_heights = [random.randint(3, 5) for _ in range(row_cell_count)]
    cell_widths = [random.randint(3, 5) for _ in range(col_cell_count)]

    line_rows = _line_positions(cell_heights)
    line_cols = _line_positions(cell_widths)
    height = line_rows[-1] + 1
    width = line_cols[-1] + 1
    if max(height, width) > 30:
        return None

    gap_mask = _build_gap_mask(height, width, line_rows, line_cols)
    line_mask = np.zeros((height, width), dtype=bool)
    line_mask[line_rows, :] = True
    line_mask[:, line_cols] = True

    out = np.zeros((height, width), dtype=int)
    out[line_mask] = BLUE
    out[gap_mask] = RED
    inp = out.copy()
    inp[gap_mask] = BLACK

    boxes = _cell_boxes(line_rows, line_cols)
    total_cells = len(boxes)
    if total_cells < 4:
        return None

    pattern_count = random.randint(2, min(5, total_cells - 1))
    pattern_ids = set(random.sample(range(total_cells), pattern_count))
    marker_ids = set(random.sample(sorted(pattern_ids), random.randint(1, pattern_count)))

    cell_specs: list[dict] = []
    signatures: set[tuple[tuple[int, int], bytes]] = set()
    distinct_ensured = 0
    for idx, box in enumerate(boxes):
        r0, c0, r1, c1 = box
        cell_h = r1 - r0 + 1
        cell_w = c1 - c0 + 1
        spec = {"box": box, "tile": None, "missing": []}
        if idx not in pattern_ids:
            cell_specs.append(spec)
            continue

        banned = None
        if distinct_ensured < 2:
            banned = signatures
        tile = _choose_tile_for_cell(
            cell_h,
            cell_w,
            require_repeat=idx in marker_ids,
            banned=banned,
        )
        if tile is None:
            return None

        tiled = _tile_cell(tile, cell_h, cell_w)
        out[r0 : r1 + 1, c0 : c1 + 1] = tiled
        inp[r0 : r1 + 1, c0 : c1 + 1] = tiled

        missing = []
        if idx in marker_ids:
            missing = _choose_missing_positions(cell_h, cell_w, tile.shape[0], tile.shape[1])
            if not missing:
                return None
            for rr, cc in missing:
                inp[r0 + rr, c0 + cc] = PINK

        sig = _tile_signature(tile)
        if sig not in signatures:
            signatures.add(sig)
            distinct_ensured += 1

        spec["tile"] = tile
        spec["missing"] = missing
        cell_specs.append(spec)

    if not _validate_instance(inp, out, line_rows, line_cols, cell_specs, gap_mask):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 6")
