import math
import random
from itertools import combinations, product

import numpy as np


BLACK = 0
GREY = 5
COLOR_ORDER = (1, 2, 3, 4)
PERIOD_CHOICES = {
    3: ((6, 4, 2), (6, 4, 3), (4, 3, 2)),
    4: ((6, 4, 3, 2),),
}
MAX_ATTEMPTS = 400


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    pts = np.argwhere(mask)
    if pts.size == 0:
        raise ValueError("empty mask")
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _lcm(values: list[int]) -> int:
    out = 1
    for v in values:
        out = out * v // math.gcd(out, v)
    return out


def _all_subsets(colors: list[int]) -> list[tuple[int, ...]]:
    out: list[tuple[int, ...]] = []
    for size in range(1, len(colors) + 1):
        out.extend(combinations(colors, size))
    return out


def _phase_score(color: int, row: int, periods: dict[int, int], offsets: dict[int, int]) -> tuple[int, int, int]:
    return ((row - offsets[color]) % periods[color], periods[color], color)


def _cell_color(
    subset: tuple[int, ...] | list[int],
    row: int,
    periods: dict[int, int],
    offsets: dict[int, int],
) -> int:
    return min(subset, key=lambda color: _phase_score(color, row, periods, offsets))


def _simultaneous_hits(
    subset: tuple[int, ...] | list[int],
    row: int,
    periods: dict[int, int],
    offsets: dict[int, int],
) -> list[int]:
    return [
        color
        for color in subset
        if (row - offsets[color]) % periods[color] == 0
    ]


def _fill_from_structure(
    top: np.ndarray,
    periods: dict[int, int],
    offsets: dict[int, int],
    p_col: int,
    residue_subsets: list[tuple[int, ...]],
) -> np.ndarray:
    h, w = top.shape
    out = top.copy()
    for r in range(h):
        for c in range(w):
            color = _cell_color(residue_subsets[c % p_col], r, periods, offsets)
            if out[r, c] == BLACK:
                out[r, c] = color
            elif int(out[r, c]) != color:
                raise AssertionError("visible cell disagrees with inferred structure")
    return out


def _separator_row(grid: np.ndarray) -> int:
    rows = [r for r in range(grid.shape[0]) if np.all(grid[r] == GREY)]
    if len(rows) != 1:
        raise AssertionError("expected exactly one full-width grey separator row")
    return rows[0]


def _table_bbox(grid: np.ndarray, sep_r: int) -> tuple[int, int, int, int]:
    mask = (grid[:sep_r] != BLACK) & (grid[:sep_r] != GREY)
    return _bbox(mask)


def _periods_from_bars(grid: np.ndarray, sep_r: int) -> dict[int, int]:
    lower = grid[sep_r + 1 :]
    comps = _components(lower == GREY)
    bars = []
    for comp in comps:
        rows = [r for r, _ in comp]
        cols = {c for _, c in comp}
        if len(cols) != 1:
            raise AssertionError("lower grey metadata must be vertical 1-pixel bars")
        height = max(rows) - min(rows) + 1
        bars.append(height)
    bars.sort(reverse=True)
    if len(bars) not in (3, 4) or len(set(bars)) != len(bars):
        raise AssertionError("expected 3 or 4 distinct bar heights")
    colors = list(COLOR_ORDER[: len(bars)])
    return {color: bars[i] for i, color in enumerate(colors)}


def _offset_product(periods: dict[int, int]):
    colors = sorted(periods)
    spaces = [range(periods[color]) for color in colors]
    for values in product(*spaces):
        yield {color: values[i] for i, color in enumerate(colors)}


def _valid_tables_for_period(
    top: np.ndarray,
    periods: dict[int, int],
    p_col: int,
) -> dict[bytes, tuple[np.ndarray, dict[int, int], list[tuple[int, ...]]]]:
    colors = sorted(periods)
    subset_pool = _all_subsets(colors)
    h, w = top.shape
    tables: dict[bytes, tuple[np.ndarray, dict[int, int], list[tuple[int, ...]]]] = {}
    for offsets in _offset_product(periods):
        residue_subsets: list[tuple[int, ...]] = []
        ok = True
        for residue in range(p_col):
            observations = [
                (r, int(top[r, c]))
                for r in range(h)
                for c in range(w)
                if c % p_col == residue and int(top[r, c]) in periods
            ]
            if not observations:
                ok = False
                break
            valid_subsets = [
                subset
                for subset in subset_pool
                if all(_cell_color(subset, r, periods, offsets) == color for r, color in observations)
            ]
            if len(valid_subsets) != 1:
                ok = False
                break
            residue_subsets.append(valid_subsets[0])
        if not ok:
            continue
        filled = _fill_from_structure(top, periods, offsets, p_col, residue_subsets)
        tables.setdefault(filled.tobytes(), (filled, offsets, residue_subsets))
    return tables


def _solve_top(
    top: np.ndarray,
    periods: dict[int, int],
) -> tuple[int, dict[int, int], list[tuple[int, ...]], np.ndarray]:
    for p_col in range(1, top.shape[1] + 1):
        tables = _valid_tables_for_period(top, periods, p_col)
        if len(tables) == 1:
            filled, offsets, residue_subsets = next(iter(tables.values()))
            return p_col, offsets, residue_subsets, filled
    raise AssertionError("no unique periodic repair found")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repair the damaged upper color table using row-period metadata encoded by ranked grey bars.

    Input structure:
    - A full-width grey separator row divides the grid into an upper table region and a lower metadata region.
    - The upper region contains one rectangular periodic color table on black background, using 3 or 4 colors among blue(1), red(2), green(3), yellow(4); some interior cells are replaced by black(0) holes.
    - The lower region contains the same number of separate 1-pixel-wide grey vertical bars, each with a distinct height.

    Rule:
    - Sort the lower grey bars by height descending and map them to blue, red, green, yellow in that order; each mapped height is that color's row period.
    - Infer the upper table rectangle and the smallest column period whose residues each admit a unique candidate-color subset and unique row-phase offsets consistent with all visible cells.
    - Within one column residue, each candidate color repeats a hit every `period` rows; the visible color at row `r` is the candidate whose most recent hit is closest above `r`, with simultaneous hits broken by shorter period.
    - Fill every black hole in the table from that recovered periodic structure while leaving all visible cells, the separator row, and the grey bars unchanged.

    Key insight:
    - The lower bars are not recolored output objects; they are metadata whose height ranking binds colors to row periods, and those periods are the missing parameters needed to reconstruct the damaged periodic table above.
    """
    g = np.asarray(grid, dtype=int)
    sep_r = _separator_row(g)
    r0, c0, r1, c1 = _table_bbox(g, sep_r)
    top = g[r0 : r1 + 1, c0 : c1 + 1]
    periods = _periods_from_bars(g, sep_r)
    _p_col, _offsets, _subsets, repaired = _solve_top(top, periods)
    out = g.copy()
    out[r0 : r1 + 1, c0 : c1 + 1] = repaired
    return out


def _strict_period(seq: list[tuple[int, ...]]) -> int:
    for period in range(1, len(seq) + 1):
        if len(seq) % period != 0:
            continue
        if all(seq[i] == seq[i % period] for i in range(len(seq))):
            return period
    return len(seq)


def _subset_candidates(
    colors: list[int],
    periods: dict[int, int],
    offsets: dict[int, int],
    height: int,
) -> list[tuple[int, ...]]:
    candidates = []
    for subset in _all_subsets(colors):
        seq = [_cell_color(subset, r, periods, offsets) for r in range(height)]
        if len(set(seq)) < 2:
            continue
        candidates.append(subset)
    return candidates


def _build_table(
    height: int,
    width: int,
    p_col: int,
    periods: dict[int, int],
    offsets: dict[int, int],
    residue_subsets: list[tuple[int, ...]],
) -> np.ndarray:
    table = np.zeros((height, width), dtype=int)
    for r in range(height):
        for c in range(width):
            table[r, c] = _cell_color(residue_subsets[c % p_col], r, periods, offsets)
    return table


def _same_row_copy_shortcut_fails(mask: np.ndarray, p_col: int) -> bool:
    h, w = mask.shape
    for r in range(h):
        for residue in range(p_col):
            cols = [c for c in range(w) if c % p_col == residue]
            if cols and all(mask[r, c] for c in cols):
                return True
    return False


def _pick_spaced_columns(width: int, n: int) -> list[int]:
    for _ in range(300):
        cols = sorted(random.sample(list(range(1, width - 1)), n))
        if all(cols[i + 1] - cols[i] >= 3 for i in range(n - 1)):
            return cols
    raise AssertionError("failed to place spaced bar columns")


def _forced_candidates(
    height: int,
    p_col: int,
    periods: dict[int, int],
    offsets: dict[int, int],
    residue_subsets: list[tuple[int, ...]],
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    generic: list[tuple[int, int]] = []
    tie_cells: list[tuple[int, int]] = []
    for r in range(1, height - 1):
        for residue in range(p_col):
            generic.append((r, residue))
            if len(_simultaneous_hits(residue_subsets[residue], r, periods, offsets)) >= 2:
                tie_cells.append((r, residue))
    return generic, tie_cells


def _make_holes(
    table: np.ndarray,
    colors: list[int],
    periods: dict[int, int],
    offsets: dict[int, int],
    p_col: int,
    residue_subsets: list[tuple[int, ...]],
) -> tuple[np.ndarray, np.ndarray, list[tuple[int, int]], list[tuple[int, int]]]:
    h, w = table.shape
    hole_mask = np.zeros_like(table, dtype=bool)
    generic, tie_cells = _forced_candidates(h, p_col, periods, offsets, residue_subsets)
    if tie_cells:
        r, residue = random.choice(tie_cells)
        for c in range(residue, w, p_col):
            hole_mask[r, c] = True
        forced = [(r, residue)]
        forced_ties = [(r, residue)]
    else:
        forced = []
        forced_ties = []
    random.shuffle(generic)
    target_extra = random.randint(1, 3)
    for r, residue in generic:
        if len(forced) >= len(forced_ties) + target_extra + (1 if forced_ties else 1):
            break
        if (r, residue) in forced:
            continue
        for c in range(residue, w, p_col):
            hole_mask[r, c] = True
        forced.append((r, residue))
    for _ in range(random.randint(2, 4)):
        rh = random.randint(1, 3)
        cw = random.randint(1, 2)
        r0 = random.randint(1, h - rh - 2)
        c0 = random.randint(1, w - cw - 2)
        hole_mask[r0 : r0 + rh, c0 : c0 + cw] = True
    hole_mask[0, :] = False
    hole_mask[-1, :] = False
    hole_mask[:, 0] = False
    hole_mask[:, -1] = False
    if hole_mask.sum() < max(6, table.size // 8):
        return table.copy(), hole_mask, [], []
    if not all(np.any((table == color) & (~hole_mask)) for color in colors):
        return table.copy(), hole_mask, [], []
    if not _same_row_copy_shortcut_fails(hole_mask, p_col):
        return table.copy(), hole_mask, [], []
    damaged = table.copy()
    damaged[hole_mask] = BLACK
    return damaged, hole_mask, forced, forced_ties


def _swap_mapping_changes_result(meta: dict) -> bool:
    colors = meta["colors"]
    periods = meta["periods"]
    offsets = meta["offsets"]
    residue_subsets = meta["residue_subsets"]
    forced = meta["forced_missing"]
    for i in range(len(colors)):
        for j in range(i + 1, len(colors)):
            alt_periods = dict(periods)
            c1, c2 = colors[i], colors[j]
            alt_periods[c1], alt_periods[c2] = alt_periods[c2], alt_periods[c1]
            for r, residue in forced:
                actual = _cell_color(residue_subsets[residue], r, periods, offsets)
                alt = _cell_color(residue_subsets[residue], r, alt_periods, offsets)
                if actual != alt:
                    return True
    return False


def _tie_priority_changes_result(meta: dict) -> bool:
    periods = meta["periods"]
    offsets = meta["offsets"]
    residue_subsets = meta["residue_subsets"]
    for r, residue in meta["tie_missing"]:
        active = _simultaneous_hits(residue_subsets[residue], r, periods, offsets)
        if len(active) < 2:
            continue
        smaller = min(active, key=lambda color: periods[color])
        larger = max(active, key=lambda color: periods[color])
        if smaller != larger:
            return True
    return False


def _build_grid(periods_desc: tuple[int, ...]) -> tuple[np.ndarray, dict]:
    k = len(periods_desc)
    colors = list(COLOR_ORDER[:k])
    periods = {color: periods_desc[i] for i, color in enumerate(colors)}
    height = _lcm(list(periods.values()))
    if height != 12:
        raise AssertionError("generator expects a 12-row tile height")
    offsets = {color: random.randrange(periods[color]) for color in colors}
    p_col = random.choice((3, 4, 5))
    width_choices = {3: (9, 12), 4: (8, 12), 5: (10,)}
    width = random.choice(width_choices[p_col])
    subset_pool = _subset_candidates(colors, periods, offsets, height)
    if not subset_pool:
        raise AssertionError("no valid subset pool")
    residue_subsets = [random.choice(subset_pool) for _ in range(p_col)]
    if len(set(residue_subsets)) < 2:
        raise AssertionError("need distinct column residues")
    if _strict_period(residue_subsets) != p_col:
        raise AssertionError("column pattern has a smaller period")
    table = _build_table(height, width, p_col, periods, offsets, residue_subsets)
    if len(set(map(int, np.unique(table)))) != k:
        raise AssertionError("every color must appear somewhere in the full table")
    if not any(
        len(_simultaneous_hits(residue_subsets[residue], r, periods, offsets)) >= 2
        for residue in range(p_col)
        for r in range(height)
    ):
        raise AssertionError("need at least one simultaneous hit for tie priority")
    if not any(len(set(map(int, table[r]))) >= 2 for r in range(height)):
        raise AssertionError("column period would be visually irrelevant")
    damaged, hole_mask, forced, tie_missing = _make_holes(
        table, colors, periods, offsets, p_col, residue_subsets
    )
    if not forced or not tie_missing:
        raise AssertionError("failed to create informative holes")

    top_r = random.choice((0, 1))
    grid_h = 21 + top_r
    grid_w = random.randint(max(18, width + 6), 22)
    left = random.randint(1, grid_w - width - 1)
    sep_r = top_r + height
    grid = np.zeros((grid_h, grid_w), dtype=int)
    grid[top_r : top_r + height, left : left + width] = damaged
    grid[sep_r, :] = GREY

    bar_cols = _pick_spaced_columns(grid_w, k)
    left_to_right = random.sample(list(periods_desc), k)
    if tuple(left_to_right) == periods_desc:
        raise AssertionError("left-to-right order should not equal rank order")
    baseline = grid_h - 1 - random.randint(0, 1)
    for col, bar_h in zip(bar_cols, left_to_right):
        top_bar = baseline - bar_h + 1
        if top_bar <= sep_r + 1:
            raise AssertionError("bar would touch the separator")
        grid[top_bar : baseline + 1, col] = GREY

    expected = grid.copy()
    expected[top_r : top_r + height, left : left + width] = table
    meta = {
        "colors": colors,
        "periods": periods,
        "offsets": offsets,
        "p_col": p_col,
        "residue_subsets": residue_subsets,
        "forced_missing": forced,
        "tie_missing": tie_missing,
        "table": table,
        "damaged_top": damaged,
        "hole_mask": hole_mask,
        "expected": expected,
    }
    return grid, meta


def _validate_generated(grid: np.ndarray, meta: dict) -> None:
    if not np.array_equal(main(grid), meta["expected"]):
        raise AssertionError("solver does not reconstruct the intended output")
    if not _swap_mapping_changes_result(meta):
        raise AssertionError("bar-to-color ranking is not actually needed")
    if not _tie_priority_changes_result(meta):
        raise AssertionError("shorter-period tie priority is not actually needed")
    if meta["p_col"] > 1:
        tables = _valid_tables_for_period(meta["damaged_top"], meta["periods"], 1)
        expected_key = meta["table"].tobytes()
        if len(tables) == 1 and expected_key in tables:
            raise AssertionError("column period would collapse to 1")


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        try:
            k = random.choice((3, 4))
            periods_desc = random.choice(PERIOD_CHOICES[k])
            grid, meta = _build_grid(periods_desc)
            _validate_generated(grid, meta)
            return grid
        except AssertionError:
            continue
    raise RuntimeError("failed to generate a valid puzzle instance")
