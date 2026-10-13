import numpy as np


SEPARATOR = 1
BLACK = 0
RED = 2
NOISE_COLORS = (3, 4, 5, 6, 7, 8, 9)
DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]
MAX_GENERATION_ATTEMPTS = 400
CELL_COUNT = 5
CELL_SIZE = 3
GRID_SIZE = CELL_COUNT * CELL_SIZE + CELL_COUNT + 1


def _bands(length: int, separators: list[int]) -> list[tuple[int, int]]:
    bands: list[tuple[int, int]] = []
    prev = -1
    for pos in separators + [length]:
        if pos - prev > 1:
            bands.append((prev + 1, pos - 1))
        prev = pos
    return bands


def _parse_lattice(grid: np.ndarray) -> dict[str, object]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape

    row_colors = {int(g[r, 0]) for r in range(h) if np.all(g[r, :] == g[r, 0])}
    col_colors = {int(g[0, c]) for c in range(w) if np.all(g[:, c] == g[0, c])}
    sep_colors = row_colors & col_colors
    assert len(sep_colors) == 1, "expected exactly one separator color"
    sep = next(iter(sep_colors))

    sep_rows = [r for r in range(h) if np.all(g[r, :] == sep)]
    sep_cols = [c for c in range(w) if np.all(g[:, c] == sep)]
    assert sep_rows and sep_cols, "expected full separator rows and columns"
    assert sep_rows[0] == 0 and sep_rows[-1] == h - 1
    assert sep_cols[0] == 0 and sep_cols[-1] == w - 1

    row_bands = _bands(h, sep_rows)
    col_bands = _bands(w, sep_cols)
    assert row_bands and col_bands

    row_sizes = {r1 - r0 + 1 for r0, r1 in row_bands}
    col_sizes = {c1 - c0 + 1 for c0, c1 in col_bands}
    assert len(row_sizes) == 1 and len(col_sizes) == 1
    cell_h = row_sizes.pop()
    cell_w = col_sizes.pop()
    assert cell_h == cell_w, "cells must be square"

    row_to_band = np.full(h, -1, dtype=int)
    col_to_band = np.full(w, -1, dtype=int)
    for i, (r0, r1) in enumerate(row_bands):
        row_to_band[r0 : r1 + 1] = i
    for i, (c0, c1) in enumerate(col_bands):
        col_to_band[c0 : c1 + 1] = i

    return {
        "sep": sep,
        "sep_rows": sep_rows,
        "sep_cols": sep_cols,
        "row_bands": row_bands,
        "col_bands": col_bands,
        "row_to_band": row_to_band,
        "col_to_band": col_to_band,
        "cell_size": cell_h,
    }


def _passable_cells(grid: np.ndarray, parsed: dict[str, object]) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    row_bands = parsed["row_bands"]
    col_bands = parsed["col_bands"]
    allowed = np.array([BLACK, RED], dtype=int)
    passable = np.zeros((len(row_bands), len(col_bands)), dtype=bool)
    for ri, (r0, r1) in enumerate(row_bands):
        for ci, (c0, c1) in enumerate(col_bands):
            cell = g[r0 : r1 + 1, c0 : c1 + 1]
            passable[ri, ci] = bool(np.isin(cell, allowed).all())
    return passable


def _find_seed(grid: np.ndarray) -> tuple[int, int]:
    seeds = np.argwhere(np.asarray(grid, dtype=int) == RED)
    assert len(seeds) == 1, "expected exactly one red seed pixel"
    return int(seeds[0, 0]), int(seeds[0, 1])


def _spiral_trace(shape: tuple[int, int], start: tuple[int, int]) -> list[tuple[int, int]]:
    h, w = shape
    sr, sc = start
    r, c = sr, sc
    max_extent = max(sr, h - 1 - sr, sc, w - 1 - sc)
    trace: list[tuple[int, int]] = []
    segment = 0
    while True:
        dr, dc = DIRS[segment % 4]
        length = 2 * (segment // 2 + 1)
        for _ in range(length):
            r += dr
            c += dc
            trace.append((r, c))
        segment += 1
        if segment % 4 == 0 and max(abs(r - sr), abs(c - sc)) > max_extent:
            break
        assert segment < 4 * (max_extent + 6), "spiral trace failed to terminate"
    return trace


def _simulate(grid: np.ndarray) -> tuple[np.ndarray, dict[str, object]]:
    g = np.asarray(grid, dtype=int)
    parsed = _parse_lattice(g)
    seed = _find_seed(g)
    row_to_band = parsed["row_to_band"]
    col_to_band = parsed["col_to_band"]
    passable = _passable_cells(g, parsed)

    sr, sc = seed
    seed_cell = (int(row_to_band[sr]), int(col_to_band[sc]))
    assert seed_cell[0] >= 0 and seed_cell[1] >= 0, "seed must be inside a cell"
    assert passable[seed_cell], "seed cell must be passable"

    out = g.copy()
    trace = _spiral_trace(g.shape, seed)
    painted = 0
    stop_step = None
    stop_cell = None
    oob_steps = 0

    for step, (r, c) in enumerate(trace):
        if not (0 <= r < g.shape[0] and 0 <= c < g.shape[1]):
            oob_steps += 1
            continue
        ri = int(row_to_band[r])
        ci = int(col_to_band[c])
        if ri == -1 or ci == -1:
            out[r, c] = RED
            painted += 1
            continue
        if not passable[ri, ci]:
            stop_step = step
            stop_cell = (ri, ci)
            break
        out[r, c] = RED
        painted += 1

    return out, {
        "parsed": parsed,
        "passable": passable,
        "seed": seed,
        "seed_cell": seed_cell,
        "trace": trace,
        "painted": painted,
        "stop_step": stop_step,
        "stop_cell": stop_cell,
        "oob_steps": oob_steps,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Draw a red spiral across a separator lattice until it would enter an impure cell.

    Input structure:
    - A blue separator lattice partitions the grid into square cells.
    - Exactly one cell contains a single red seed pixel at its center; the remaining passable cells are pure black.
    - Several other cells contain multicolor noise patterns; those entire cells act as macro obstacles.

    Rule:
    - Split the grid into lattice cells using the full separator rows and columns.
    - Mark a cell as passable iff every interior pixel is black or the single red seed; any other color makes the whole cell blocked.
    - Starting from the red seed, follow the pixel spiral with directions up, right, down, left and step lengths 2,2,4,4,6,6,...
    - Paint each in-bounds separator pixel or passable-cell pixel red, but stop immediately before the first step that would enter a blocked cell interior; out-of-bounds steps are skipped while the spiral counting continues.

    Key insight:
    - The path itself is pixel-level, but whether it may continue is decided one scale higher: a single non-black, non-red pixel anywhere inside a cell blocks the entire cell.
    """
    out, _info = _simulate(grid)
    return out


def _cell_bounds(cell: tuple[int, int]) -> tuple[int, int, int, int]:
    ri, ci = cell
    r0 = 1 + ri * (CELL_SIZE + 1)
    c0 = 1 + ci * (CELL_SIZE + 1)
    return r0, r0 + CELL_SIZE - 1, c0, c0 + CELL_SIZE - 1


def _cell_center(cell: tuple[int, int]) -> tuple[int, int]:
    r0, r1, c0, c1 = _cell_bounds(cell)
    return (r0 + r1) // 2, (c0 + c1) // 2


def _fill_blocked_patch(
    grid: np.ndarray,
    cell: tuple[int, int],
    rng: np.random.Generator,
    near_pure: bool,
) -> dict[str, int]:
    r0, r1, c0, c1 = _cell_bounds(cell)
    patch = np.zeros((CELL_SIZE, CELL_SIZE), dtype=int)
    coords = [(r, c) for r in range(CELL_SIZE) for c in range(CELL_SIZE)]

    if near_pure:
        picks = int(rng.integers(1, 3))
        chosen = rng.choice(len(coords), size=picks, replace=False)
        palette = rng.choice(NOISE_COLORS, size=picks, replace=True)
        for idx, color in zip(chosen, palette):
            rr, cc = coords[int(idx)]
            patch[rr, cc] = int(color)
    else:
        fill_count = int(rng.integers(3, 7))
        chosen = rng.choice(len(coords), size=fill_count, replace=False)
        color_count = int(rng.integers(2, min(4, fill_count) + 1))
        palette = rng.choice(NOISE_COLORS, size=color_count, replace=False)
        repeated = rng.choice(palette, size=fill_count, replace=True)
        repeated[:color_count] = palette
        rng.shuffle(repeated)
        for idx, color in zip(chosen, repeated):
            rr, cc = coords[int(idx)]
            patch[rr, cc] = int(color)

    assert np.any((patch != BLACK) & (patch != RED))
    grid[r0 : r1 + 1, c0 : c1 + 1] = patch
    return {
        "nonblack": int(np.count_nonzero(patch)),
        "ncolors": int(len(set(map(int, patch[patch != BLACK])))),
    }


def _empty_grid() -> np.ndarray:
    g = np.full((GRID_SIZE, GRID_SIZE), SEPARATOR, dtype=int)
    for ri in range(CELL_COUNT):
        for ci in range(CELL_COUNT):
            r0, r1, c0, c1 = _cell_bounds((ri, ci))
            g[r0 : r1 + 1, c0 : c1 + 1] = BLACK
    return g


def _first_hit_data(seed: tuple[int, int]) -> dict[str, object]:
    trace = _spiral_trace((GRID_SIZE, GRID_SIZE), seed)
    row_to_band = np.full(GRID_SIZE, -1, dtype=int)
    col_to_band = np.full(GRID_SIZE, -1, dtype=int)
    for i in range(CELL_COUNT):
        r0, r1, c0, c1 = _cell_bounds((i, i))
        row_to_band[r0 : r1 + 1] = i
        col_to_band[c0 : c1 + 1] = i

    first_hit: dict[tuple[int, int], int] = {}
    first_oob = None
    first_line = None
    for step, (r, c) in enumerate(trace):
        if not (0 <= r < GRID_SIZE and 0 <= c < GRID_SIZE):
            if first_oob is None:
                first_oob = step
            continue
        ri = int(row_to_band[r])
        ci = int(col_to_band[c])
        if ri == -1 or ci == -1:
            if first_line is None:
                first_line = step
            continue
        first_hit.setdefault((ri, ci), step)
    return {"trace": trace, "first_hit": first_hit, "first_oob": first_oob, "first_line": first_line}


def _pick_stop_cell(
    seed_cell: tuple[int, int],
    seed_pos: tuple[int, int],
    scenario: str,
    rng: np.random.Generator,
) -> tuple[tuple[int, int], int, dict[str, object]] | None:
    data = _first_hit_data(seed_pos)
    first_hit = data["first_hit"]
    first_oob = data["first_oob"]
    threshold = int(rng.integers(20, 37))

    candidates = []
    for cell, step in first_hit.items():
        if cell == seed_cell or step < threshold:
            continue
        if scenario == "boundary_then_stop":
            if first_oob is None or step <= first_oob + 2:
                continue
        else:
            if first_oob is not None and step >= first_oob:
                continue
        candidates.append((cell, step))

    if not candidates:
        return None
    stop_cell, stop_step = candidates[int(rng.integers(len(candidates)))]
    return stop_cell, stop_step, data


def _semantic_checks(
    grid: np.ndarray,
    scenario: str,
    stop_cell: tuple[int, int],
    stop_step: int,
    blocked_cells: set[tuple[int, int]],
    blocked_stats: list[dict[str, int]],
) -> bool:
    out, info = _simulate(grid)
    assert info["stop_cell"] == stop_cell
    assert info["stop_step"] == stop_step
    assert info["painted"] >= 18
    assert info["stop_step"] is not None
    assert np.count_nonzero(out == RED) > 1

    if scenario == "boundary_then_stop":
        assert info["oob_steps"] > 0
        prefix_oob = sum(
            1
            for r, c in info["trace"][:stop_step]
            if not (0 <= r < GRID_SIZE and 0 <= c < GRID_SIZE)
        )
        assert prefix_oob > 0
    else:
        prefix_oob = sum(
            1
            for r, c in info["trace"][:stop_step]
            if not (0 <= r < GRID_SIZE and 0 <= c < GRID_SIZE)
        )
        assert prefix_oob == 0

    assert any(stat["nonblack"] <= 2 for stat in blocked_stats)
    assert any(stat["ncolors"] >= 2 for stat in blocked_stats)

    passable = info["passable"]
    for ri in range(CELL_COUNT):
        for ci in range(CELL_COUNT):
            expected = (ri, ci) not in blocked_cells
            if (ri, ci) == info["seed_cell"]:
                expected = True
            assert bool(passable[ri, ci]) == expected

    line_hits = 0
    for r, c in info["trace"][:stop_step]:
        if 0 <= r < GRID_SIZE and 0 <= c < GRID_SIZE:
            if r % (CELL_SIZE + 1) == 0 or c % (CELL_SIZE + 1) == 0:
                line_hits += 1
    assert line_hits > 0
    return True


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    grid = _empty_grid()

    scenario = str(rng.choice(["inner_stop", "inner_stop", "boundary_then_stop"]))
    if scenario == "boundary_then_stop":
        seed_choices = [(1, 1), (1, 2), (1, 3), (2, 1), (2, 3), (3, 1), (3, 2), (3, 3)]
    else:
        seed_choices = [(1, 1), (1, 2), (1, 3), (2, 1), (2, 2), (2, 3), (3, 1), (3, 2), (3, 3)]

    seed_cell = seed_choices[int(rng.integers(len(seed_choices)))]
    seed_pos = _cell_center(seed_cell)
    grid[seed_pos] = RED

    picked = _pick_stop_cell(seed_cell, seed_pos, scenario, rng)
    if picked is None:
        return None
    stop_cell, stop_step, hit_data = picked

    all_cells = {(ri, ci) for ri in range(CELL_COUNT) for ci in range(CELL_COUNT)}
    late_cells = []
    for cell in all_cells:
        if cell == seed_cell:
            continue
        first = hit_data["first_hit"].get(cell)
        if first is None or first > stop_step:
            late_cells.append(cell)
    if len(late_cells) < 7:
        return None

    max_blocked = min(12, len(late_cells) + 1)
    if max_blocked < 8:
        return None
    blocked_count = int(rng.integers(8, max_blocked + 1))
    extras = rng.choice(len(late_cells), size=blocked_count - 1, replace=False)
    blocked_cells = {stop_cell}
    for idx in extras:
        blocked_cells.add(late_cells[int(idx)])
    if len(blocked_cells) != blocked_count:
        return None

    blocked_stats: list[dict[str, int]] = []
    near_pure_cells = list(blocked_cells)
    rng.shuffle(near_pure_cells)
    guaranteed_near_pure = set(near_pure_cells[:2])
    guaranteed_busy = set(near_pure_cells[2:4])
    if stop_cell not in guaranteed_near_pure and stop_cell not in guaranteed_busy:
        guaranteed_busy.add(stop_cell)

    for cell in blocked_cells:
        if cell in guaranteed_near_pure:
            near_pure = True
        elif cell in guaranteed_busy:
            near_pure = False
        else:
            near_pure = bool(rng.integers(0, 2))
        blocked_stats.append(_fill_blocked_patch(grid, cell, rng, near_pure=near_pure))

    if not _semantic_checks(grid, scenario, stop_cell, stop_step, blocked_cells, blocked_stats):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid spiral-blocker input")
