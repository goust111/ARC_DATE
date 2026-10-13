from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4

TARGET_HEIGHT = 9
PERIOD_OPTIONS = (2, 3)
WIDTH_RANGE = (5, 7)
GENERATION_RETRIES = 400


def _neighbors4(r: int, c: int, h: int, w: int):
    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        rr = r + dr
        cc = c + dc
        if 0 <= rr < h and 0 <= cc < w:
            yield rr, cc


def _components4(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords = []
            while q:
                cr, cc = q.popleft()
                coords.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc, h, w):
                    if bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _has_period(grid: np.ndarray, period: int) -> bool:
    h = grid.shape[0]
    if h <= period:
        return False
    for r in range(period, h):
        if not np.array_equal(grid[r], grid[r - period]):
            return False
    return True


def _infer_period(grid: np.ndarray) -> int:
    candidates = [period for period in PERIOD_OPTIONS if _has_period(grid, period)]
    assert len(candidates) == 1, "expected a unique vertical period in {2, 3}"
    return int(candidates[0])


def _tile_to_height(grid: np.ndarray, period: int, target_height: int = TARGET_HEIGHT) -> np.ndarray:
    h, w = grid.shape
    assert h < target_height
    base = grid[:period]
    tiled = np.zeros((target_height, w), dtype=int)
    for r in range(target_height):
        tiled[r] = base[r % period]
    return tiled


def _classify_tiled_components(tiled: np.ndarray, boundary_row: int) -> np.ndarray:
    out = np.zeros_like(tiled)
    for comp in _components4(tiled == BLUE):
        if len(comp) == 1:
            continue
        rows = comp[:, 0]
        if int(rows.max()) <= boundary_row:
            color = RED
        elif int(rows.min()) > boundary_row:
            color = GREEN
        else:
            color = YELLOW
        out[comp[:, 0], comp[:, 1]] = color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Tile the vertically periodic blue pattern to nine rows, then recolor each non-singleton component by its relation to the hidden tiling boundary.

    Input structure:
    - The grid has a black background and blue foreground pixels only.
    - The blue pattern repeats vertically with an unknown period of 2 or 3 rows.
    - The visible input height is below 9 rows, so the final repetition is truncated.

    Rule:
    - Infer the unique vertical period in {2, 3}.
    - Repeat the first period downward until the grid reaches exactly 9 rows.
    - Treat the input's last row as the hidden boundary between the original region and the tiled extension.
    - Find all 4-connected blue components in the 9-row tiled grid.
    - Erase singleton components.
    - Recolor every larger component red if it stays in the original region, green if it stays in the extension, and yellow if it spans both.

    Key insight:
    - The decisive structure is not the input alone but the 9-row tiled completion: tiling changes connectivity, while the invisible original/extension boundary determines the recolor class.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = map(int, g.shape)
    assert 4 <= h < TARGET_HEIGHT, "expected input height in the ARC task range"
    assert w <= 30, "expected ARC-sized width"
    assert set(map(int, np.unique(g))) <= {BG, BLUE}, "expected only black and blue in the input"

    period = _infer_period(g)
    tiled = _tile_to_height(g, period)
    return _classify_tiled_components(tiled, boundary_row=h - 1)


def _layout_options(width: int) -> list[tuple[int, tuple[int, ...], int]]:
    options: list[tuple[int, tuple[int, ...], int]] = []
    for cross_col in range(width):
        for run_len in (2, 3):
            if run_len > width:
                continue
            for start in range(width - run_len + 1):
                run_cols = tuple(range(start, start + run_len))
                if cross_col in run_cols:
                    continue
                if any(abs(cross_col - col) == 1 for col in run_cols):
                    continue
                for single_col in range(width):
                    if single_col == cross_col or single_col in run_cols:
                        continue
                    if abs(single_col - cross_col) <= 1:
                        continue
                    options.append((cross_col, run_cols, single_col))
    return options


def _build_period_rows(
    period: int,
    width: int,
    cross_col: int,
    run_cols: tuple[int, ...],
    single_col: int,
    multi_phase: int,
    single_phase: int,
) -> np.ndarray:
    rows = np.zeros((period, width), dtype=int)
    rows[:, cross_col] = BLUE
    rows[multi_phase, list(run_cols)] = BLUE
    rows[single_phase, single_col] = BLUE
    return rows


def _semantic_summary(grid: np.ndarray) -> dict[str, object]:
    period = _infer_period(grid)
    tiled = _tile_to_height(grid, period)
    boundary_row = int(grid.shape[0] - 1)
    tiled_components = _components4(tiled == BLUE)
    input_components = _components4(grid == BLUE)

    original_only = 0
    extension_only = 0
    spanning = 0
    singles = 0
    for comp in tiled_components:
        if len(comp) == 1:
            singles += 1
            continue
        rows = comp[:, 0]
        if int(rows.max()) <= boundary_row:
            original_only += 1
        elif int(rows.min()) > boundary_row:
            extension_only += 1
        else:
            spanning += 1

    return {
        "period": period,
        "tiled": tiled,
        "output": _classify_tiled_components(tiled, boundary_row),
        "input_component_sizes": sorted(len(comp) for comp in input_components),
        "original_only": original_only,
        "extension_only": extension_only,
        "spanning": spanning,
        "singles": singles,
    }


def _valid_generated_grid(grid: np.ndarray) -> bool:
    h, w = map(int, grid.shape)
    if not (WIDTH_RANGE[0] <= w <= WIDTH_RANGE[1]):
        return False
    if h >= TARGET_HEIGHT or h in {0, 1, 2, 3}:
        return False

    try:
        summary = _semantic_summary(grid)
    except AssertionError:
        return False

    period = int(summary["period"])
    if h % period == 0:
        return False

    sizes = list(summary["input_component_sizes"])
    if not any(size == 1 for size in sizes):
        return False
    if not any(size > 1 for size in sizes):
        return False

    if int(summary["original_only"]) < 1:
        return False
    if int(summary["extension_only"]) < 1:
        return False
    if int(summary["spanning"]) < 1:
        return False
    if int(summary["singles"]) < 1:
        return False

    out = np.asarray(summary["output"], dtype=int)
    colors = set(map(int, np.unique(out)))
    if not {RED, GREEN, YELLOW}.issubset(colors):
        return False
    if BLUE in colors:
        return False

    return np.array_equal(main(grid), out)


def _choose_height(period: int) -> int:
    choices = [h for h in range(4, 7) if h < TARGET_HEIGHT and h % period != 0]
    return int(random.choice(choices))


def _gen_attempt() -> np.ndarray | None:
    period = int(random.choice(PERIOD_OPTIONS))
    height = _choose_height(period)
    width = int(random.randint(*WIDTH_RANGE))
    layouts = _layout_options(width)
    if not layouts:
        return None

    cross_col, run_cols, single_col = random.choice(layouts)
    multi_phase = int(random.randrange(period))
    single_candidates = [phase for phase in range(period) if phase != multi_phase]
    single_phase = int(random.choice(single_candidates))

    base_rows = _build_period_rows(
        period=period,
        width=width,
        cross_col=cross_col,
        run_cols=run_cols,
        single_col=single_col,
        multi_phase=multi_phase,
        single_phase=single_phase,
    )

    grid = np.zeros((height, width), dtype=int)
    for r in range(height):
        grid[r] = base_rows[r % period]

    if _valid_generated_grid(grid):
        return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 8")
