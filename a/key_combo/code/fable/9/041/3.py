from __future__ import annotations

import random
from typing import Any

import numpy as np


COLOR_POOL = tuple(range(10))
GRID_LIMIT = 30
GENERATION_RETRIES = 1200
TILE_RETRIES = 200
PERIOD_RANGE = (2, 5)
SEGMENT_LENGTH_RANGE = (2, 6)


def _mode_color(values: np.ndarray) -> int | None:
    flat = values.astype(int).ravel()
    counts = np.bincount(flat, minlength=10)
    best = counts.max()
    winners = np.flatnonzero(counts == best)
    if winners.size != 1:
        return None
    return int(winners[0])


def _tile_grid(tile: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    rr = np.arange(h)[:, None] % tile.shape[0]
    cc = np.arange(w)[None, :] % tile.shape[1]
    return tile[rr, cc]


def _minimal_exact_periods(tile: np.ndarray) -> tuple[int, int]:
    h, w = tile.shape

    row_period = h
    for p in range(1, h + 1):
        if all(np.array_equal(tile[r], tile[r % p]) for r in range(h)):
            row_period = p
            break

    col_period = w
    for p in range(1, w + 1):
        if all(np.array_equal(tile[:, c], tile[:, c % p]) for c in range(w)):
            col_period = p
            break

    return row_period, col_period


def _segment_info_from_mask(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None

    sums = coords[:, 0] + coords[:, 1]
    if np.unique(sums).size != 1:
        return None
    diag_sum = int(sums[0])

    order = np.argsort(coords[:, 1], kind="stable")
    coords = coords[order]
    cols = coords[:, 1].astype(int)
    rows = coords[:, 0].astype(int)

    if not np.array_equal(cols, np.arange(cols[0], cols[0] + len(cols))):
        return None
    if not np.array_equal(rows, diag_sum - cols):
        return None

    return diag_sum, int(cols[0]), int(cols[-1]), int(len(cols))


def _extension_cells(
    shape: tuple[int, int], diag_sum: int, c1: int, length: int
) -> list[tuple[int, int]] | None:
    h, w = shape
    cells: list[tuple[int, int]] = []
    for c in range(c1 + 1, c1 + length + 2):
        r = diag_sum - c
        if not (0 <= r < h and 0 <= c < w):
            return None
        cells.append((r, c))
    return cells


def _candidate_for_periods(grid: np.ndarray, pr: int, pc: int) -> dict[str, Any] | None:
    h, w = grid.shape
    tile = np.empty((pr, pc), dtype=int)
    for rr in range(pr):
        for cc in range(pc):
            color = _mode_color(grid[rr::pr, cc::pc])
            if color is None:
                return None
            tile[rr, cc] = color

    ideal = _tile_grid(tile, (h, w))
    mismatch = grid != ideal
    segment = _segment_info_from_mask(mismatch)
    if segment is None:
        return None

    diag_sum, c0, c1, length = segment
    defect_colors = np.unique(grid[mismatch])
    if defect_colors.size != 1:
        return None
    defect_color = int(defect_colors[0])

    extension = _extension_cells((h, w), diag_sum, c1, length)
    if extension is None:
        return None

    return {
        "periods": (pr, pc),
        "tile": tile,
        "ideal": ideal,
        "diag_sum": diag_sum,
        "c0": c0,
        "c1": c1,
        "length": length,
        "defect_color": defect_color,
        "extension": extension,
    }


def _best_candidates(grid: np.ndarray) -> list[dict[str, Any]]:
    h, w = grid.shape
    best_area: int | None = None
    best: list[dict[str, Any]] = []
    for pr in range(1, h + 1):
        for pc in range(1, w + 1):
            info = _candidate_for_periods(grid, pr, pc)
            if info is None:
                continue
            area = pr * pc
            if best_area is None or area < best_area:
                best_area = area
                best = [info]
            elif area == best_area:
                best.append(info)
    return best


def main(grid: np.ndarray) -> np.ndarray:
    """Locate the anti-diagonal defect by wallpaper periodicity, repair it, then paint a longer copy one step ahead.

    Input structure:
    - The whole grid is a rectangular wallpaper made by periodically tiling a smaller color tile.
    - One contiguous segment lying on a single anti-diagonal has been overwritten to one uniform defect color.
    - That defect color also belongs to the wallpaper palette, so color alone does not identify the defect.

    Rule:
    - Infer the smallest repeating tile from the uncorrupted periodic evidence and compare the implied wallpaper to the input.
    - The unique disagreement set is a continuous anti-diagonal defect segment; first restore those cells to the wallpaper colors.
    - Then continue on the same anti-diagonal immediately after the segment and paint the defect color over the next segment of length L+1.

    Key insight:
    - The altered segment is detectable only as a periodicity violation relative to the wallpaper model, and that same recovered model is needed both to repair the old segment and to place the longer advanced one.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    candidates = _best_candidates(g)
    assert len(candidates) == 1, "expected a unique smallest periodic explanation"
    info = candidates[0]

    out = info["ideal"].copy()
    for r, c in info["extension"]:
        out[r, c] = info["defect_color"]
    return out


def _random_dimension(period: int) -> int:
    max_reps = min(6, GRID_LIMIT // period)
    min_reps = min(max_reps, max(4, (10 + period - 1) // period))
    reps = random.randint(min_reps, max_reps)
    base = reps * period
    extra = random.randint(0, min(period - 1, GRID_LIMIT - base))
    return base + extra


def _random_tile(pr: int, pc: int) -> np.ndarray | None:
    palette_cap = min(6, pr * pc, len(COLOR_POOL))
    for _ in range(TILE_RETRIES):
        palette_size = random.randint(3, palette_cap)
        palette = random.sample(COLOR_POOL, palette_size)
        tile = np.array(
            [[random.choice(palette) for _ in range(pc)] for _ in range(pr)],
            dtype=int,
        )
        if len(set(map(int, tile.flat))) < 3:
            continue
        if _minimal_exact_periods(tile) != (pr, pc):
            continue
        return tile
    return None


def _segment_cells(diag_sum: int, c0: int, length: int) -> list[tuple[int, int]]:
    return [(diag_sum - c, c) for c in range(c0, c0 + length)]


def _enumerate_segment_options(base: np.ndarray, defect_color: int) -> list[dict[str, Any]]:
    h, w = base.shape
    opts: list[dict[str, Any]] = []
    total_d_count = int(np.count_nonzero(base == defect_color))
    if total_d_count == 0:
        return opts

    for diag_sum in range(h + w - 1):
        min_c = max(0, diag_sum - (h - 1))
        max_c = min(w - 1, diag_sum)
        max_len = min(SEGMENT_LENGTH_RANGE[1], (max_c - min_c) // 2)
        for length in range(SEGMENT_LENGTH_RANGE[0], max_len + 1):
            for c0 in range(min_c, max_c - 2 * length + 2):
                c1 = c0 + length - 1
                segment = _segment_cells(diag_sum, c0, length)
                extension = _extension_cells((h, w), diag_sum, c1, length)
                if extension is None:
                    continue

                segment_colors = [int(base[r, c]) for r, c in segment]
                extension_colors = [int(base[r, c]) for r, c in extension]
                if defect_color in segment_colors or defect_color in extension_colors:
                    continue
                if len(set(segment_colors)) < 2:
                    continue

                diag_d_elsewhere = 0
                for c in range(min_c, max_c + 1):
                    if c0 <= c <= c1:
                        continue
                    if int(base[diag_sum - c, c]) == defect_color:
                        diag_d_elsewhere += 1

                opts.append(
                    {
                        "diag_sum": diag_sum,
                        "c0": c0,
                        "c1": c1,
                        "length": length,
                        "segment": segment,
                        "extension": extension,
                        "same_diag_d_count": diag_d_elsewhere,
                    }
                )
    return opts


def _semantic_checks(
    inp: np.ndarray,
    base: np.ndarray,
    periods: tuple[int, int],
    option: dict[str, Any],
    defect_color: int,
) -> bool:
    best = _best_candidates(inp)
    if len(best) != 1:
        return False

    info = best[0]
    if tuple(info["periods"]) != periods:
        return False
    if not np.array_equal(info["ideal"], base):
        return False
    if info["diag_sum"] != option["diag_sum"]:
        return False
    if info["c0"] != option["c0"] or info["c1"] != option["c1"]:
        return False
    if info["length"] != option["length"]:
        return False
    if info["defect_color"] != defect_color:
        return False

    out = main(inp)
    expected = base.copy()
    for r, c in option["extension"]:
        expected[r, c] = defect_color
    if not np.array_equal(out, expected):
        return False

    if np.unique(inp[tuple(np.array(option["segment"]).T)]).size != 1:
        return False
    if int(inp[option["segment"][0]]) != defect_color:
        return False

    natural_d = np.count_nonzero(base == defect_color)
    if natural_d < len(option["segment"]) + 1:
        return False

    return True


def _gen_attempt() -> np.ndarray | None:
    pr = random.randint(*PERIOD_RANGE)
    pc = random.randint(*PERIOD_RANGE)
    tile = _random_tile(pr, pc)
    if tile is None:
        return None

    h = _random_dimension(pr)
    w = _random_dimension(pc)
    base = _tile_grid(tile, (h, w))
    periods = _minimal_exact_periods(tile)
    if periods != (pr, pc):
        return None

    palette = list(set(map(int, tile.flat)))
    random.shuffle(palette)
    for defect_color in palette:
        options = _enumerate_segment_options(base, defect_color)
        if not options:
            continue

        random.shuffle(options)
        options.sort(key=lambda item: item["same_diag_d_count"], reverse=True)
        for option in options[:40]:
            inp = base.copy()
            for r, c in option["segment"]:
                inp[r, c] = defect_color

            if _semantic_checks(inp, base, periods, option, defect_color):
                return inp

    return None


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid wallpaper-defect instance")
