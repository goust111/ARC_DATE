from __future__ import annotations

import numpy as np


GEN_MAX_ATTEMPTS = 600
GEN_BG = 0
GEN_MOTIFS = (
    "diag",
    "anti",
    "border",
    "vstripe",
    "hstripe",
    "plus",
    "elbow",
    "xshape",
    "corners",
)
GEN_MIN_CELL = 3
GEN_MAX_CELL = 5
GEN_MIN_ROWS = 2
GEN_MAX_ROWS = 4
GEN_MIN_COLS = 4
GEN_MAX_COLS = 6
GEN_MIN_PERIOD = 3
GEN_MAX_PERIOD = 7
GEN_MIN_PROTOTYPES = 2
GEN_MAX_PROTOTYPES = 4
GEN_PALETTE = (1, 2, 3, 4, 5, 6, 7, 8, 9)


def _true_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for idx, value in enumerate(mask.tolist()):
        if value and start is None:
            start = idx
        elif not value and start is not None:
            runs.append((start, idx))
            start = None
    if start is not None:
        runs.append((start, len(mask)))
    return runs


def _most_common_color(grid: np.ndarray, exclude: tuple[int, ...] = ()) -> int:
    values = grid.ravel()
    if exclude:
        values = values[~np.isin(values, np.array(exclude, dtype=int))]
    counts = np.bincount(values.astype(int))
    return int(np.argmax(counts))


def _find_separator_layout(grid: np.ndarray) -> tuple[int, list[tuple[int, int]], list[tuple[int, int]]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    candidates: list[tuple[int, list[tuple[int, int]], list[tuple[int, int]]]] = []

    for color in np.unique(g):
        row_runs = _true_runs(np.all(g == int(color), axis=1))
        col_runs = _true_runs(np.all(g == int(color), axis=0))
        if len(row_runs) < 3 or len(col_runs) < 3:
            continue
        if row_runs[0][0] != 0 or row_runs[-1][1] != h:
            continue
        if col_runs[0][0] != 0 or col_runs[-1][1] != w:
            continue

        row_heights = {end - start for start, end in row_runs}
        col_widths = {end - start for start, end in col_runs}
        if len(row_heights) != 1 or len(col_widths) != 1:
            continue

        cell_heights = [row_runs[i + 1][0] - row_runs[i][1] for i in range(len(row_runs) - 1)]
        cell_widths = [col_runs[i + 1][0] - col_runs[i][1] for i in range(len(col_runs) - 1)]
        if min(cell_heights) <= 0 or min(cell_widths) <= 0:
            continue
        if len(set(cell_heights)) != 1 or len(set(cell_widths)) != 1:
            continue
        candidates.append((int(color), row_runs, col_runs))

    assert len(candidates) == 1, "expected a unique separator grid"
    return candidates[0]


def _extract_cells(
    grid: np.ndarray, row_runs: list[tuple[int, int]], col_runs: list[tuple[int, int]]
) -> tuple[list[np.ndarray], int, int]:
    cells: list[np.ndarray] = []
    n_rows = len(row_runs) - 1
    n_cols = len(col_runs) - 1
    for rr in range(n_rows):
        r0 = row_runs[rr][1]
        r1 = row_runs[rr + 1][0]
        for cc in range(n_cols):
            c0 = col_runs[cc][1]
            c1 = col_runs[cc + 1][0]
            cells.append(grid[r0:r1, c0:c1].copy())
    return cells, n_rows, n_cols


def _tile_equal(a: np.ndarray, b: np.ndarray) -> bool:
    return bool(np.array_equal(a, b))


def _infer_period_from_visible_tiles(visible_tiles: list[np.ndarray | None]) -> tuple[int, list[np.ndarray]]:
    n = len(visible_tiles)
    for period in range(1, n + 1):
        residues: list[np.ndarray] = []
        ok = True
        for residue in range(period):
            observed = [visible_tiles[idx] for idx in range(residue, n, period) if visible_tiles[idx] is not None]
            if not observed:
                ok = False
                break
            base = observed[0]
            assert base is not None
            if any(not _tile_equal(base, other) for other in observed[1:]):
                ok = False
                break
            residues.append(base.copy())
        if ok:
            return period, residues
    raise AssertionError("no valid period found")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill the empty separator-grid cells with the repeating tile that belongs at each row-major position.

    Input structure:
    - A single-color separator lattice divides the canvas into equal rectangular cells with an outer border.
    - Some cells contain non-empty tile patterns on a shared background, while other cells are completely blank.
    - Reading cells in row-major order produces a partially visible cyclic sequence of tile patterns.

    Rule:
    - Identify the separator color and partition the grid into equal cells.
    - Treat each non-blank cell as a known tile in the row-major cell sequence, and find the unique minimal period whose residues are all observed and whose repeated tiles match every visible cell.
    - Copy that period through the whole cell sequence and fill each blank cell with the tile that belongs at its position, leaving the separator lines unchanged.

    Key insight:
    - The repetition lives at the whole-cell level, not along pixel rows or columns, and the true period is intentionally misaligned with the number of cells per row.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    separator, row_runs, col_runs = _find_separator_layout(g)
    bg = _most_common_color(g, exclude=(separator,))
    cells, _cell_rows, cell_cols = _extract_cells(g, row_runs, col_runs)
    visible_tiles = [None if np.all(cell == bg) else cell for cell in cells]
    assert any(tile is None for tile in visible_tiles), "expected at least one blank cell"

    period, residues = _infer_period_from_visible_tiles(visible_tiles)
    assert cell_cols % period != 0, "period should be misaligned with the row length"

    out = g.copy()
    for idx, tile in enumerate(visible_tiles):
        if tile is not None:
            continue
        rr = idx // cell_cols
        cc = idx % cell_cols
        r0 = row_runs[rr][1]
        r1 = row_runs[rr + 1][0]
        c0 = col_runs[cc][1]
        c1 = col_runs[cc + 1][0]
        out[r0:r1, c0:c1] = residues[idx % period]
    return out


def _render_motif(motif: str, shape: tuple[int, int], color: int, bg: int) -> np.ndarray:
    h, w = shape
    tile = np.full((h, w), bg, dtype=int)
    mid_r = h // 2
    mid_c = w // 2

    if motif == "diag":
        for r in range(h):
            c = int(round(r * (w - 1) / max(1, h - 1)))
            tile[r, c] = color
    elif motif == "anti":
        for r in range(h):
            c = int(round((h - 1 - r) * (w - 1) / max(1, h - 1)))
            tile[r, c] = color
    elif motif == "border":
        tile[0, :] = color
        tile[-1, :] = color
        tile[:, 0] = color
        tile[:, -1] = color
    elif motif == "vstripe":
        tile[:, mid_c] = color
    elif motif == "hstripe":
        tile[mid_r, :] = color
    elif motif == "plus":
        tile[mid_r, :] = color
        tile[:, mid_c] = color
    elif motif == "elbow":
        tile[0, :] = color
        tile[:, 0] = color
    elif motif == "xshape":
        for r in range(h):
            c0 = int(round(r * (w - 1) / max(1, h - 1)))
            c1 = int(round((h - 1 - r) * (w - 1) / max(1, h - 1)))
            tile[r, c0] = color
            tile[r, c1] = color
    elif motif == "corners":
        tile[0, 0] = color
        tile[0, -1] = color
        tile[-1, 0] = color
        tile[mid_r, mid_c] = color
    else:
        raise AssertionError("unknown motif")
    return tile


def _prototype_pool(cell_shape: tuple[int, int], bg: int, colors: list[int]) -> list[np.ndarray]:
    seen: set[bytes] = set()
    pool: list[np.ndarray] = []
    for motif in GEN_MOTIFS:
        for color in colors:
            tile = _render_motif(motif, cell_shape, color, bg)
            key = tile.tobytes()
            if np.all(tile == bg) or key in seen:
                continue
            seen.add(key)
            pool.append(tile)
    return pool


def _minimal_period_ids(seq: list[int]) -> int:
    n = len(seq)
    for period in range(1, n + 1):
        if all(seq[idx] == seq[idx % period] for idx in range(n)):
            return period
    return n


def _candidate_period_from_visible_ids(visible_ids: list[int | None]) -> tuple[int, list[int]]:
    n = len(visible_ids)
    for period in range(1, n + 1):
        residues: list[int] = []
        ok = True
        for residue in range(period):
            observed = [visible_ids[idx] for idx in range(residue, n, period) if visible_ids[idx] is not None]
            if not observed:
                ok = False
                break
            base = int(observed[0])
            if any(int(value) != base for value in observed[1:]):
                ok = False
                break
            residues.append(base)
        if ok:
            return period, residues
    raise AssertionError("no consistent visible period")


def _build_full_ids(period_ids: list[int], total_cells: int) -> list[int]:
    return [int(period_ids[idx % len(period_ids)]) for idx in range(total_cells)]


def _column_varies(full_ids: list[int], cell_rows: int, cell_cols: int, col: int) -> bool:
    column_values = {full_ids[row * cell_cols + col] for row in range(cell_rows)}
    return len(column_values) > 1


def _visibility_mask(
    rng: np.random.Generator, full_ids: list[int], period: int, cell_rows: int, cell_cols: int
) -> np.ndarray | None:
    total = len(full_ids)
    residue_positions = [list(range(residue, total, period)) for residue in range(period)]
    anchors = [positions[int(rng.integers(len(positions)))] for positions in residue_positions]
    visible = np.zeros(total, dtype=bool)
    visible[anchors] = True

    target_min = max(period, int(np.ceil(total * 0.55)))
    target_max = max(target_min, total - 2)
    target = int(rng.integers(target_min, target_max + 1))
    extras = [idx for idx in range(total) if not visible[idx]]
    rng.shuffle(extras)
    for idx in extras[: max(0, target - int(np.count_nonzero(visible)))]:
        visible[idx] = True

    if np.count_nonzero(~visible) < 2:
        return None
    if not np.any(~visible & (np.arange(total) >= cell_cols)):
        return None

    hidden_ids = {full_ids[idx] for idx in range(total) if not visible[idx]}
    if len(hidden_ids) < 2:
        return None

    if not any(_column_varies(full_ids, cell_rows, cell_cols, col) for col in range(cell_cols)):
        return None
    if not any(
        (not visible[idx]) and _column_varies(full_ids, cell_rows, cell_cols, idx % cell_cols)
        for idx in range(total)
    ):
        return None

    visible_ids = [full_ids[idx] if visible[idx] else None for idx in range(total)]
    chosen_period, residues = _candidate_period_from_visible_ids(visible_ids)
    if chosen_period != period:
        return None
    if residues != full_ids[:period]:
        return None
    return visible


def _render_grid(
    full_ids: list[int],
    prototypes: list[np.ndarray],
    cell_rows: int,
    cell_cols: int,
    sep: int,
    bg: int,
) -> np.ndarray:
    cell_h, cell_w = prototypes[0].shape
    height = cell_rows * cell_h + cell_rows + 1
    width = cell_cols * cell_w + cell_cols + 1
    grid = np.full((height, width), sep, dtype=int)
    for idx, proto_id in enumerate(full_ids):
        rr = idx // cell_cols
        cc = idx % cell_cols
        r0 = 1 + rr * (cell_h + 1)
        c0 = 1 + cc * (cell_w + 1)
        grid[r0 : r0 + cell_h, c0 : c0 + cell_w] = prototypes[proto_id]
    return grid


def _erase_hidden_cells(
    full_grid: np.ndarray,
    visible: np.ndarray,
    cell_rows: int,
    cell_cols: int,
    cell_shape: tuple[int, int],
    bg: int,
) -> np.ndarray:
    cell_h, cell_w = cell_shape
    grid = full_grid.copy()
    for idx, keep in enumerate(visible.tolist()):
        if keep:
            continue
        rr = idx // cell_cols
        cc = idx % cell_cols
        r0 = 1 + rr * (cell_h + 1)
        c0 = 1 + cc * (cell_w + 1)
        grid[r0 : r0 + cell_h, c0 : c0 + cell_w] = bg
    return grid


def _choose_dimensions(rng: np.random.Generator) -> tuple[int, int, int, int] | None:
    cell_h = int(rng.integers(GEN_MIN_CELL, GEN_MAX_CELL + 1))
    cell_w = int(rng.integers(GEN_MIN_CELL, GEN_MAX_CELL + 1))
    max_rows = min(GEN_MAX_ROWS, (30 - 1) // (cell_h + 1))
    max_cols = min(GEN_MAX_COLS, (30 - 1) // (cell_w + 1))
    if max_rows < GEN_MIN_ROWS or max_cols < GEN_MIN_COLS:
        return None
    cell_rows = int(rng.integers(GEN_MIN_ROWS, max_rows + 1))
    cell_cols = int(rng.integers(GEN_MIN_COLS, max_cols + 1))
    return cell_h, cell_w, cell_rows, cell_cols


def _choose_period(rng: np.random.Generator, total_cells: int, cell_cols: int) -> int | None:
    options = [
        period
        for period in range(GEN_MIN_PERIOD, min(GEN_MAX_PERIOD, total_cells - 2) + 1)
        if cell_cols % period != 0 and total_cells >= period + 3
    ]
    if not options:
        return None
    return int(options[int(rng.integers(len(options)))])


def _choose_period_ids(rng: np.random.Generator, prototype_count: int, period: int) -> list[int] | None:
    for _ in range(100):
        period_ids = rng.integers(0, prototype_count, size=period).astype(int).tolist()
        if len(set(period_ids)) < 2:
            continue
        if _minimal_period_ids(period_ids) != period:
            continue
        return [int(value) for value in period_ids]
    return None


def _generate_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()
    dims = _choose_dimensions(rng)
    if dims is None:
        return None
    cell_h, cell_w, cell_rows, cell_cols = dims
    total_cells = cell_rows * cell_cols

    period = _choose_period(rng, total_cells, cell_cols)
    if period is None:
        return None

    sep = int(rng.choice(np.array(GEN_PALETTE, dtype=int)))
    colors = [color for color in GEN_PALETTE if color != sep]
    rng.shuffle(colors)
    pool = _prototype_pool((cell_h, cell_w), GEN_BG, colors)
    if len(pool) < GEN_MIN_PROTOTYPES:
        return None

    prototype_count = int(rng.integers(GEN_MIN_PROTOTYPES, min(GEN_MAX_PROTOTYPES, len(pool)) + 1))
    selected = rng.choice(len(pool), size=prototype_count, replace=False)
    prototypes = [pool[int(idx)] for idx in selected]

    period_ids = _choose_period_ids(rng, prototype_count, period)
    if period_ids is None:
        return None
    full_ids = _build_full_ids(period_ids, total_cells)

    visible = _visibility_mask(rng, full_ids, period, cell_rows, cell_cols)
    if visible is None:
        return None

    full_grid = _render_grid(full_ids, prototypes, cell_rows, cell_cols, sep, GEN_BG)
    inp = _erase_hidden_cells(full_grid, visible, cell_rows, cell_cols, (cell_h, cell_w), GEN_BG)

    if _most_common_color(inp, exclude=(sep,)) != GEN_BG:
        return None
    try:
        out = main(inp)
    except Exception:
        return None
    if not np.array_equal(out, full_grid):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid periodic tile grid")
