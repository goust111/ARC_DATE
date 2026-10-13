from __future__ import annotations

from collections import Counter
import random

import numpy as np


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, np.asarray(grid).reshape(-1))).most_common(1)[0][0])


def _foreground_tile_for_period(
    grid: np.ndarray, bg: int, row_period: int, col_period: int
) -> dict[tuple[int, int], int] | None:
    tile: dict[tuple[int, int], int] = {}
    height, width = grid.shape
    for r in range(height):
        rr = r % row_period
        for c in range(width):
            color = int(grid[r, c])
            if color == bg:
                continue
            key = (rr, c % col_period)
            prev = tile.get(key)
            if prev is not None and prev != color:
                return None
            tile[key] = color
    return tile


def _restore_periodic_wallpaper(
    grid: np.ndarray, bg: int
) -> tuple[np.ndarray, int, int, dict[tuple[int, int], int]] | None:
    g = np.asarray(grid, dtype=int)
    height, width = g.shape
    candidates: list[tuple[int, int, dict[tuple[int, int], int]]] = []
    for row_period in range(1, height + 1):
        for col_period in range(1, width + 1):
            tile = _foreground_tile_for_period(g, bg, row_period, col_period)
            if tile is not None and tile:
                candidates.append((row_period, col_period, tile))

    if not candidates:
        return None

    min_area = min(row_period * col_period for row_period, col_period, _ in candidates)
    best = [
        candidate
        for candidate in candidates
        if candidate[0] * candidate[1] == min_area
    ]
    if len(best) != 1:
        return None

    row_period, col_period, tile = best[0]
    out = np.full_like(g, bg)
    for (tile_r, tile_c), color in tile.items():
        out[tile_r::row_period, tile_c::col_period] = color
    return out, row_period, col_period, tile


def _row_pair_stats(a: np.ndarray, b: np.ndarray, bg: int) -> tuple[int, int]:
    overlap = (a != bg) & (b != bg)
    if not bool(overlap.any()):
        return 0, 0
    agree = int(np.count_nonzero(overlap & (a == b)))
    conflict = int(np.count_nonzero(overlap & (a != b)))
    return agree, conflict


def _visible_row_matches_template(row: np.ndarray, template: np.ndarray, bg: int) -> bool:
    fg = row != bg
    return bool(fg.any()) and bool(np.all(template[fg] == row[fg]))


def _infer_class_template(
    rows: list[np.ndarray], bg: int
) -> tuple[np.ndarray, list[int]] | None:
    if len(rows) < 2:
        return None

    best_template = None
    best_support = None
    best_score = None

    for seed_idx, seed in enumerate(rows):
        subset = []
        support_score = 0
        for idx, row in enumerate(rows):
            agree, conflict = _row_pair_stats(seed, row, bg)
            if conflict == 0 and (agree > 0 or idx == seed_idx):
                subset.append(idx)
                support_score += agree
        if not subset:
            continue

        template = np.full(rows[0].shape, bg, dtype=int)
        valid = True
        for idx in subset:
            row = rows[idx]
            fg = row != bg
            if bool(np.any(fg & (template != bg) & (template != row))):
                valid = False
                break
            template[fg] = row[fg]
        if not valid:
            continue

        support = [
            idx for idx, row in enumerate(rows)
            if _visible_row_matches_template(row, template, bg)
        ]
        score = (len(support), support_score, int(np.count_nonzero(template != bg)))
        if best_score is None or score > best_score:
            best_score = score
            best_template = template
            best_support = support

    if best_template is None or best_support is None:
        return None
    if 2 * len(best_support) <= len(rows):
        return None
    return best_template, best_support


def _best_shift_against_template(
    row: np.ndarray, template: np.ndarray, bg: int
) -> tuple[int, np.ndarray] | None:
    width = row.shape[0]
    options: list[tuple[int, int, np.ndarray]] = []
    for shift in range(width):
        rolled = np.roll(row, -shift)
        fg = rolled != bg
        if not bool(fg.any()):
            continue
        if not bool(np.all(template[fg] == rolled[fg])):
            continue
        score = int(np.count_nonzero(rolled == template))
        options.append((score, shift, rolled))

    if not options:
        return None

    best_score = max(score for score, _shift, _rolled in options)
    best = [(shift, rolled) for score, shift, rolled in options if score == best_score]
    if len(best) != 1:
        return None
    return best[0]


def _align_with_row_period(
    grid: np.ndarray, row_period: int, bg: int
) -> tuple[np.ndarray, dict[int, int], list[np.ndarray]] | None:
    g = np.asarray(grid, dtype=int)
    height, _width = g.shape
    groups = [list(range(rr, height, row_period)) for rr in range(row_period)]
    if not groups or min(len(group) for group in groups) < 2:
        return None

    templates: list[np.ndarray] = []
    for group in groups:
        rows = [g[r].copy() for r in group]
        inferred = _infer_class_template(rows, bg)
        if inferred is None:
            return None
        template, support = inferred
        if 2 * len(support) <= len(group):
            return None
        templates.append(template)

    aligned = np.empty_like(g)
    shifts: dict[int, int] = {}
    for rr, group in enumerate(groups):
        template = templates[rr]
        for r in group:
            best = _best_shift_against_template(g[r], template, bg)
            if best is None:
                return None
            shift, rolled = best
            aligned[r] = rolled
            shifts[r] = shift

    if sum(shift == 0 for shift in shifts.values()) <= height // 2:
        return None
    if all(shift == 0 for shift in shifts.values()):
        return None
    return aligned, shifts, templates


def main(grid: np.ndarray) -> np.ndarray:
    """Restore a doubly periodic wallpaper by undoing whole-row circular slips before filling erased cells.

    Input structure:
    - The grid is a multicolor wallpaper with both row and column periodicity, and the background
      color is also a legitimate tile color.
    - A minority of rows have been circularly shifted left or right by row-specific amounts.
    - Some foreground wallpaper pixels have been erased to the background color.

    Rule:
    - Use the unshifted majority phase to infer, for each row, the unique circular rollback that
      makes rows sharing the same row-period residue agree again.
    - After all rows are realigned, find the unique smallest row/column period consistent with the
      visible non-background pixels.
    - Repaint every non-background residue class across the whole grid, leaving true background
      residue classes untouched.

    Key insight:
    - The holes are not directly solvable on the misaligned image: the hidden structure is a clean
      periodic wallpaper, so row alignment must happen before periodic completion.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and 0 < g.shape[0] <= 30 and 0 < g.shape[1] <= 30
    bg = get_bg(g)

    outputs: dict[bytes, tuple[int, np.ndarray]] = {}
    for row_period in range(1, g.shape[0] // 2 + 1):
        aligned_info = _align_with_row_period(g, row_period, bg)
        if aligned_info is None:
            continue
        aligned, _shifts, _templates = aligned_info
        restored = _restore_periodic_wallpaper(aligned, bg)
        if restored is None:
            continue
        out, final_row_period, final_col_period, _tile = restored
        key = out.tobytes()
        area = final_row_period * final_col_period
        prev = outputs.get(key)
        if prev is None or area < prev[0]:
            outputs[key] = (area, out)

    assert outputs, "expected a uniquely alignable periodic wallpaper"
    min_area = min(area for area, _out in outputs.values())
    best = [out for area, out in outputs.values() if area == min_area]
    assert len(best) == 1, "expected a unique smallest restored wallpaper"
    return best[0]


GEN_BG = 0
GEN_HEIGHT_RANGE = (15, 26)
GEN_WIDTH_RANGE = (17, 28)
GEN_ROW_PERIOD_RANGE = (2, 5)
GEN_COL_PERIOD_RANGE = (3, 7)
GEN_FG_COLOR_COUNT_RANGE = (3, 5)
GEN_BG_DENSITY_RANGE = (0.28, 0.42)
GEN_SHIFTED_CLASS_COUNT_RANGE = (1, 3)
GEN_BLOB_COUNT_RANGE = (2, 5)
GEN_BLOB_H_RANGE = (1, 3)
GEN_BLOB_W_RANGE = (2, 5)
GEN_ERASE_FRAC_RANGE = (0.05, 0.12)
GEN_ATTEMPTS = 400


def _wallpaper_from_tile(tile: np.ndarray, height: int, width: int) -> np.ndarray:
    rows = np.arange(height)[:, None] % tile.shape[0]
    cols = np.arange(width)[None, :] % tile.shape[1]
    return tile[rows, cols]


def _has_smaller_row_period(tile: np.ndarray) -> bool:
    row_period, _col_period = tile.shape
    for p in range(1, row_period):
        if bool(np.array_equal(tile, tile[np.arange(row_period) % p])):
            return True
    return False


def _has_smaller_col_period(tile: np.ndarray) -> bool:
    _row_period, col_period = tile.shape
    for p in range(1, col_period):
        if bool(np.array_equal(tile, tile[:, np.arange(col_period) % p])):
            return True
    return False


def _row_strings_are_distinct(wallpaper: np.ndarray, row_period: int) -> bool:
    width = wallpaper.shape[1]
    rows = [wallpaper[r].copy() for r in range(row_period)]
    for i, row_i in enumerate(rows):
        for shift in range(1, width):
            if bool(np.array_equal(row_i, np.roll(row_i, shift))):
                return False
        for j, row_j in enumerate(rows):
            if i == j:
                continue
            for shift in range(width):
                if bool(np.array_equal(row_i, np.roll(row_j, shift))):
                    return False
    return True


def _sample_tile(height: int, width: int, row_period: int, col_period: int) -> np.ndarray | None:
    fg_count = random.randint(*GEN_FG_COLOR_COUNT_RANGE)
    palette = [GEN_BG] + random.sample(list(range(1, 10)), fg_count)
    bg_prob = random.uniform(*GEN_BG_DENSITY_RANGE)

    for _ in range(120):
        tile = np.full((row_period, col_period), GEN_BG, dtype=int)
        for r in range(row_period):
            while True:
                row = np.array(
                    [
                        GEN_BG if random.random() < bg_prob else random.choice(palette[1:])
                        for _ in range(col_period)
                    ],
                    dtype=int,
                )
                if np.count_nonzero(row != GEN_BG) >= 2 and np.count_nonzero(row == GEN_BG) >= 1:
                    break
            tile[r] = row

        if _has_smaller_row_period(tile) or _has_smaller_col_period(tile):
            continue

        wallpaper = _wallpaper_from_tile(tile, height, width)
        if not _row_strings_are_distinct(wallpaper, row_period):
            continue

        values, counts = np.unique(wallpaper, return_counts=True)
        max_count = int(counts.max())
        bg_count = int(counts[values == GEN_BG][0]) if GEN_BG in values else 0
        if bg_count != max_count or int(np.count_nonzero(counts == max_count)) != 1:
            continue

        return tile
    return None


def _choose_row_period(height: int) -> int | None:
    options = [
        period
        for period in range(GEN_ROW_PERIOD_RANGE[0], GEN_ROW_PERIOD_RANGE[1] + 1)
        if height >= 3 * period
    ]
    if not options:
        return None
    return random.choice(options)


def _choose_col_period(width: int) -> int | None:
    options = [
        period
        for period in range(GEN_COL_PERIOD_RANGE[0], min(GEN_COL_PERIOD_RANGE[1], width - 1) + 1)
        if width % period != 0
    ]
    if not options:
        return None
    return random.choice(options)


def _erase_with_rectangles(
    wallpaper: np.ndarray,
    row_period: int,
    col_period: int,
    shifted_rows: dict[int, int],
) -> tuple[np.ndarray, int] | None:
    height, width = wallpaper.shape
    fg_positions = int(np.count_nonzero(wallpaper != GEN_BG))
    min_erased = max(1, int(np.ceil(fg_positions * GEN_ERASE_FRAC_RANGE[0])))
    max_erased = max(min_erased, int(np.floor(fg_positions * GEN_ERASE_FRAC_RANGE[1])))

    for _ in range(120):
        erased = wallpaper.copy()
        before = erased.copy()
        for _blob in range(random.randint(*GEN_BLOB_COUNT_RANGE)):
            blob_h = random.randint(*GEN_BLOB_H_RANGE)
            blob_w = random.randint(*GEN_BLOB_W_RANGE)
            r0 = random.randint(0, height - blob_h)
            c0 = random.randint(0, width - blob_w)
            patch = erased[r0 : r0 + blob_h, c0 : c0 + blob_w]
            patch[patch != GEN_BG] = GEN_BG

        erased_count = int(np.count_nonzero((before != GEN_BG) & (erased == GEN_BG)))
        if not (min_erased <= erased_count <= max_erased):
            continue

        valid = True
        for rr in range(row_period):
            class_rows = [r for r in range(rr, height, row_period) if r not in shifted_rows]
            template = wallpaper[rr]
            for c in range(width):
                if template[c] == GEN_BG:
                    continue
                if not any(int(erased[r, c]) == int(template[c]) for r in class_rows):
                    valid = False
                    break
            if not valid:
                break

        if not valid:
            continue

        for rr in range(row_period):
            for cc in range(col_period):
                sample_color = int(wallpaper[rr, cc])
                if sample_color == GEN_BG:
                    continue
                coords = [
                    (r, c)
                    for r in range(rr, height, row_period)
                    for c in range(cc, width, col_period)
                ]
                if not any(int(erased[r, c]) == sample_color for r, c in coords):
                    valid = False
                    break
            if not valid:
                break

        if valid:
            return erased, erased_count

    return None


def _apply_nontrivial_row_shifts(
    grid: np.ndarray, row_period: int, col_period: int
) -> tuple[np.ndarray, dict[int, int]] | None:
    height, width = grid.shape
    classes = list(range(row_period))
    random.shuffle(classes)
    max_classes = min(len(classes), GEN_SHIFTED_CLASS_COUNT_RANGE[1])
    min_classes = min(len(classes), GEN_SHIFTED_CLASS_COUNT_RANGE[0])
    shifted_count = random.randint(min_classes, max_classes)
    chosen_classes = classes[:shifted_count]

    shifted_rows: dict[int, int] = {}
    out = grid.copy()
    for rr in chosen_classes:
        candidates = list(range(rr, height, row_period))
        if len(candidates) < 3:
            return None
        row = random.choice(candidates)
        shifts = [shift for shift in range(1, width) if shift % col_period != 0]
        if not shifts:
            return None
        chosen_shift = random.choice(shifts)
        out[row] = np.roll(out[row], chosen_shift)
        shifted_rows[row] = chosen_shift

    return out, shifted_rows


def _fill_only_output(grid: np.ndarray) -> np.ndarray | None:
    restored = _restore_periodic_wallpaper(grid, GEN_BG)
    if restored is None:
        return None
    return restored[0]


def _rows_have_unique_true_shifts(
    grid: np.ndarray, wallpaper: np.ndarray, row_period: int
) -> bool:
    height = grid.shape[0]
    for rr in range(row_period):
        template = wallpaper[rr]
        for r in range(rr, height, row_period):
            best = _best_shift_against_template(grid[r], template, GEN_BG)
            if best is None:
                return False
    return True


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(*GEN_HEIGHT_RANGE)
    width = random.randint(*GEN_WIDTH_RANGE)
    row_period = _choose_row_period(height)
    col_period = _choose_col_period(width)
    if row_period is None or col_period is None:
        return None

    tile = _sample_tile(height, width, row_period, col_period)
    if tile is None:
        return None

    wallpaper = _wallpaper_from_tile(tile, height, width)
    shifted = _apply_nontrivial_row_shifts(wallpaper, row_period, col_period)
    if shifted is None:
        return None
    shifted_grid, shifted_rows = shifted

    erased = _erase_with_rectangles(wallpaper, row_period, col_period, shifted_rows)
    if erased is None:
        return None
    erased_grid, erased_count = erased
    if erased_count <= 0:
        return None

    inp = erased_grid.copy()
    for row, shift in shifted_rows.items():
        inp[row] = np.roll(inp[row], shift)

    if np.array_equal(inp, wallpaper):
        return None
    if np.array_equal(inp, erased_grid):
        return None
    if not _rows_have_unique_true_shifts(inp, wallpaper, row_period):
        return None

    filled_only = _fill_only_output(inp)
    if filled_only is not None and bool(np.array_equal(filled_only, wallpaper)):
        return None

    aligned = inp.copy()
    for row, shift in shifted_rows.items():
        aligned[row] = np.roll(aligned[row], -shift)
    if np.array_equal(aligned, wallpaper):
        return None

    try:
        solved = main(inp)
    except AssertionError:
        return None
    if not bool(np.array_equal(solved, wallpaper)):
        return None

    values, counts = np.unique(inp, return_counts=True)
    max_count = int(counts.max())
    bg_count = int(counts[values == GEN_BG][0]) if GEN_BG in values else 0
    if bg_count != max_count or int(np.count_nonzero(counts == max_count)) != 1:
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid misaligned-wallpaper input")
