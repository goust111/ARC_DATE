from __future__ import annotations

from collections import Counter
import random

import numpy as np


ARC_COLORS = tuple(range(10))
PATTERN_COLORS = tuple(range(1, 10))
MAX_GRID_SIZE = 30
GEN_ATTEMPTS = 400


def _separator_color(grid: np.ndarray) -> int:
    row_colors = {
        int(grid[r, 0])
        for r in range(grid.shape[0])
        if np.all(grid[r, :] == grid[r, 0])
    }
    col_colors = {
        int(grid[0, c])
        for c in range(grid.shape[1])
        if np.all(grid[:, c] == grid[0, c])
    }
    sep_colors = row_colors & col_colors
    assert len(sep_colors) == 1, "expected one separator color spanning full rows and columns"
    return next(iter(sep_colors))


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    bands: list[tuple[int, int]] = []
    start: int | None = None
    for i, keep in enumerate(mask.tolist()):
        if keep and start is None:
            start = i
        if not keep and start is not None:
            bands.append((start, i))
            start = None
    if start is not None:
        bands.append((start, len(mask)))
    return bands


def _compress_separator_grid(grid: np.ndarray) -> np.ndarray:
    sep = _separator_color(grid)
    row_bands = _runs(~np.all(grid == sep, axis=1))
    col_bands = _runs(~np.all(grid == sep, axis=0))
    assert len(row_bands) >= 2 and len(col_bands) >= 2, "expected multiple separator-defined regions"

    compact = np.empty((len(row_bands), len(col_bands)), dtype=int)
    for rr, (r0, r1) in enumerate(row_bands):
        for cc, (c0, c1) in enumerate(col_bands):
            patch = grid[r0:r1, c0:c1]
            assert patch.size > 0
            colors = np.unique(patch)
            assert len(colors) == 1 and int(colors[0]) != sep, "each region must be uniformly colored"
            compact[rr, cc] = int(colors[0])
    return compact


def _strict_majority(values: np.ndarray) -> int | None:
    flat = [int(v) for v in values.ravel()]
    counts = Counter(flat)
    color, count = counts.most_common(1)[0]
    if count * 2 <= len(flat):
        return None
    if sum(1 for v in counts.values() if v == count) != 1:
        return None
    return color


def _is_exact_period(grid: np.ndarray, pr: int, pc: int) -> bool:
    tile = grid[:pr, :pc]
    for r in range(grid.shape[0]):
        for c in range(grid.shape[1]):
            if int(grid[r, c]) != int(tile[r % pr, c % pc]):
                return False
    return True


def _minimal_exact_periods(grid: np.ndarray) -> list[tuple[int, int]]:
    h, w = grid.shape
    best: list[tuple[int, int]] = []
    best_area = h * w + 1
    for pr in range(1, h + 1):
        for pc in range(1, w + 1):
            area = pr * pc
            if area > best_area:
                continue
            if _is_exact_period(grid, pr, pc):
                if area < best_area:
                    best_area = area
                    best = [(pr, pc)]
                else:
                    best.append((pr, pc))
    return best


def _repair_for_period(grid: np.ndarray, pr: int, pc: int) -> np.ndarray | None:
    h, w = grid.shape
    tile = np.empty((pr, pc), dtype=int)
    for rr in range(pr):
        for cc in range(pc):
            residue = grid[rr:h:pr, cc:w:pc]
            if residue.size < 2:
                return None
            majority = _strict_majority(residue)
            if majority is None:
                return None
            tile[rr, cc] = majority

    repaired = np.empty_like(grid)
    for r in range(h):
        for c in range(w):
            repaired[r, c] = tile[r % pr, c % pc]

    if _minimal_exact_periods(repaired) != [(pr, pc)]:
        return None
    return repaired


def _solve_compact(grid: np.ndarray) -> tuple[tuple[int, int], np.ndarray]:
    h, w = grid.shape
    candidates: list[tuple[int, int, int, np.ndarray]] = []
    for pr in range(1, h + 1):
        for pc in range(1, w + 1):
            repaired = _repair_for_period(grid, pr, pc)
            if repaired is not None:
                candidates.append((pr * pc, pr, pc, repaired))

    assert candidates, "no valid periodic repair found"
    min_area = min(area for area, _pr, _pc, _rep in candidates)
    best = [(pr, pc, rep) for area, pr, pc, rep in candidates if area == min_area]
    assert len(best) == 1, "minimal periodic repair must be unique"
    pr, pc, repaired = best[0]
    return (pr, pc), repaired


def main(grid: np.ndarray) -> np.ndarray:
    """
    Compress a separator grid, infer its smallest periodic domain, and majority-repair the corrupted regions.

    Input structure:
    - Full-width and full-height separator lines partition the canvas into rectangular regions with uneven heights and widths.
    - Every separator-delimited region is monochromatic, so the region colors form a smaller hidden grid.
    - That compressed region-color grid is almost a tiling of one smallest periodic rectangle, except for a few corrupted regions.

    Rule:
    - Identify the separator color and collapse each rectangular region to one pixel, preserving region row/column order.
    - On the compressed grid, search for the unique smallest-area period rectangle whose row/column residue classes each have a strict majority color.
    - Replace every compressed cell by the majority color of its residue class modulo that period rectangle.
    - Output the repaired compressed grid.

    Key insight:
    - The periodic structure exists only after compressing uneven separator-defined regions; the pixel-scale image itself is not the periodic object to repair.
    """
    compact = _compress_separator_grid(np.asarray(grid, dtype=int))
    _period, repaired = _solve_compact(compact)
    return repaired


def _repeat_tile(tile: np.ndarray, rows: int, cols: int) -> np.ndarray:
    out = np.empty((rows, cols), dtype=int)
    for r in range(rows):
        for c in range(cols):
            out[r, c] = int(tile[r % tile.shape[0], c % tile.shape[1]])
    return out


def _color_pool(exclude: set[int]) -> list[int]:
    return [c for c in ARC_COLORS if c not in exclude]


def _seed_palette_into_tile(tile: np.ndarray, palette: list[int]) -> None:
    coords = [(r, c) for r in range(tile.shape[0]) for c in range(tile.shape[1])]
    random.shuffle(coords)
    for color, (r, c) in zip(palette, coords):
        tile[r, c] = color


def _sample_true_pattern() -> tuple[tuple[int, int], np.ndarray]:
    for _ in range(200):
        pr = random.randint(2, 4)
        pc = random.randint(2, 4)
        rep_r = random.randint(2, 3)
        rep_c = random.randint(2, 3)
        extra_r = 0 if random.random() < 0.5 else random.randint(1, pr - 1)
        extra_c = 0 if random.random() < 0.5 else random.randint(1, pc - 1)
        rows = pr * rep_r + extra_r
        cols = pc * rep_c + extra_c
        palette_size = random.randint(3, min(6, pr * pc))
        palette = random.sample(list(PATTERN_COLORS), palette_size)

        tile = np.array(
            [[random.choice(palette) for _ in range(pc)] for _ in range(pr)],
            dtype=int,
        )
        _seed_palette_into_tile(tile, palette)
        clean = _repeat_tile(tile, rows, cols)

        if _minimal_exact_periods(clean) != [(pr, pc)]:
            continue
        try:
            found_period, repaired = _solve_compact(clean)
        except AssertionError:
            continue
        if found_period != (pr, pc) or not np.array_equal(repaired, clean):
            continue
        return (pr, pc), clean
    raise RuntimeError("failed to sample a non-degenerate periodic compact grid")


def _residue_positions(shape: tuple[int, int], pr: int, pc: int) -> dict[tuple[int, int], list[tuple[int, int]]]:
    rows, cols = shape
    out: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for r in range(rows):
        for c in range(cols):
            out.setdefault((r % pr, c % pc), []).append((r, c))
    return out


def _corrupt_compact(clean: np.ndarray, period: tuple[int, int]) -> np.ndarray:
    pr, pc = period
    residues = _residue_positions(clean.shape, pr, pc)
    all_colors = set(int(v) for v in np.unique(clean))
    for _ in range(250):
        noisy = clean.copy()
        changed: set[tuple[int, int]] = set()
        budgets = {
            residue: (len(coords) - 1) // 2
            for residue, coords in residues.items()
        }
        usable = [residue for residue, budget in budgets.items() if budget > 0]
        if not usable:
            break

        max_changes = min(sum(budgets.values()), max(2, clean.size // 4))
        target_changes = random.randint(1, max_changes)
        while len(changed) < target_changes:
            residue = random.choice(usable)
            if budgets[residue] <= 0:
                usable = [res for res in usable if budgets[res] > 0]
                if not usable:
                    break
                continue

            candidates = [pos for pos in residues[residue] if pos not in changed]
            if not candidates:
                budgets[residue] = 0
                usable = [res for res in usable if budgets[res] > 0]
                if not usable:
                    break
                continue

            r, c = random.choice(candidates)
            replacement_choices = [color for color in PATTERN_COLORS if color != int(clean[r, c])]
            if all_colors - {int(clean[r, c])}:
                replacement_choices = list(all_colors - {int(clean[r, c])}) + replacement_choices
            noisy[r, c] = random.choice(replacement_choices)
            changed.add((r, c))
            budgets[residue] -= 1

        if not changed:
            continue
        try:
            found_period, repaired = _solve_compact(noisy)
        except AssertionError:
            continue
        if found_period != period:
            continue
        if not np.array_equal(repaired, clean):
            continue
        if np.array_equal(noisy, clean):
            continue
        return noisy
    raise RuntimeError("failed to corrupt compact grid without introducing ambiguity")


def _sample_band_sizes(count: int, limit: int) -> list[int] | None:
    for _ in range(120):
        sizes = [random.randint(1, 3) for _ in range(count)]
        total = sum(sizes) + count - 1
        if total <= limit and len(set(sizes)) >= 2:
            return sizes
    return None


def _build_separator_grid(compact: np.ndarray, sep: int, row_sizes: list[int], col_sizes: list[int]) -> np.ndarray:
    rows = sum(row_sizes) + len(row_sizes) - 1
    cols = sum(col_sizes) + len(col_sizes) - 1
    grid = np.full((rows, cols), sep, dtype=int)

    r = 0
    for rr, height in enumerate(row_sizes):
        c = 0
        for cc, width in enumerate(col_sizes):
            grid[r : r + height, c : c + width] = int(compact[rr, cc])
            c += width + 1
        r += height + 1
    return grid


def _validate_instance(inp: np.ndarray, clean_compact: np.ndarray, noisy_compact: np.ndarray, period: tuple[int, int]) -> bool:
    if inp.ndim != 2 or max(inp.shape) > MAX_GRID_SIZE:
        return False
    if np.array_equal(clean_compact, noisy_compact):
        return False
    if np.array_equal(clean_compact, np.full_like(clean_compact, clean_compact[0, 0])):
        return False
    try:
        compressed = _compress_separator_grid(inp)
        found_period, repaired = _solve_compact(compressed)
        out = main(inp)
    except AssertionError:
        return False
    if found_period != period:
        return False
    if not np.array_equal(compressed, noisy_compact):
        return False
    if not np.array_equal(repaired, clean_compact):
        return False
    if not np.array_equal(out, clean_compact):
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        period, clean_compact = _sample_true_pattern()
        noisy_compact = _corrupt_compact(clean_compact, period)
        sep = 0
        row_sizes = _sample_band_sizes(noisy_compact.shape[0], MAX_GRID_SIZE)
        col_sizes = _sample_band_sizes(noisy_compact.shape[1], MAX_GRID_SIZE)
        if row_sizes is None or col_sizes is None:
            continue

        inp = _build_separator_grid(noisy_compact, sep, row_sizes, col_sizes)
        if _validate_instance(inp, clean_compact, noisy_compact, period):
            return inp
    raise RuntimeError("failed to generate a valid compressed-domain periodic-repair puzzle")
