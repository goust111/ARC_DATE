from __future__ import annotations

import itertools
from typing import Iterable

import numpy as np


BG = 0
GRAY = 5
NEUTRAL = 8
PALETTE_POOL = (1, 2, 3, 4, 6, 7, 9)
GEN_RETRIES = 300
RNG = np.random.default_rng()
_PERM_CACHE: dict[int, list[tuple[int, ...]]] = {}


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each cyan table cell by the marker colors filtered through a unique Latin-square completion.

    Input structure:
    - Black is the background, and full gray rows/columns form a framed K x K table of rectangular cells.
    - Every cell interior is monochrome: either a fixed foreground color or cyan for a neutral undecided cell.
    - Colored single-pixel markers lie outside the table, one pixel beyond the outer gray frame, aligned with border-side neutral cells.

    Rule:
    - Detect the full gray separator rows and columns, then recover the K x K cell grid from the gaps between them.
    - For each cyan cell, collect the marker colors found one pixel beyond any adjacent outer separator, aligned with that cell, as its candidate set.
    - Keep fixed cells unchanged and solve the remaining cells so every row and every column contains each foreground table color exactly once.
    - Recolor each cyan cell to its uniquely determined color, erase all outside markers to black, and keep the gray separators unchanged.

    Key insight:
    - Marker adjacency only narrows each neutral cell to a local candidate set; the final color is forced only after combining those local candidates with global Latin-square row/column uniqueness.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    parsed = _parse_puzzle(g)
    solution = _solve_latin_unique(parsed["board"], parsed["candidate_map"], parsed["palette"])

    out = g.copy()
    out[parsed["marker_mask"]] = BG
    row_spans = parsed["row_spans"]
    col_spans = parsed["col_spans"]
    for r, c in parsed["blanks"]:
        r0, r1 = row_spans[r]
        c0, c1 = col_spans[c]
        out[r0 : r1 + 1, c0 : c1 + 1] = solution[r, c]
    return out


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid, meta = _gen_attempt()
        if grid is not None and _validate_generated_instance(grid, meta):
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")


def _parse_puzzle(grid: np.ndarray) -> dict[str, object]:
    sep_rows = np.flatnonzero(np.all(grid == GRAY, axis=1))
    sep_cols = np.flatnonzero(np.all(grid == GRAY, axis=0))
    assert sep_rows.size >= 5 and sep_cols.size >= 5, "expected framed separator rows and columns"

    row_spans = _cell_spans(sep_rows.tolist())
    col_spans = _cell_spans(sep_cols.tolist())
    k = len(row_spans)
    assert k == len(col_spans), "table must be square"
    assert k in (4, 5), "expected a 4x4 or 5x5 table"

    top, bottom = int(sep_rows[0]), int(sep_rows[-1])
    left, right = int(sep_cols[0]), int(sep_cols[-1])

    outside_mask = np.ones_like(grid, dtype=bool)
    outside_mask[top : bottom + 1, left : right + 1] = False
    marker_mask = outside_mask & (grid != BG) & (grid != GRAY)

    board = np.full((k, k), -1, dtype=int)
    blanks: list[tuple[int, int]] = []
    fixed_colors: list[int] = []

    for r, (r0, r1) in enumerate(row_spans):
        for c, (c0, c1) in enumerate(col_spans):
            patch = grid[r0 : r1 + 1, c0 : c1 + 1]
            colors = np.unique(patch)
            assert colors.size == 1, "each cell interior must be monochrome"
            color = int(colors[0])
            assert color not in (BG, GRAY), "cell interiors may not contain background or separators"
            if color == NEUTRAL:
                blanks.append((r, c))
            else:
                board[r, c] = color
                fixed_colors.append(color)

    assert blanks, "expected at least one neutral cell"
    palette = _appearance_order(fixed_colors)
    assert len(palette) == k, "fixed cells must reveal the full Latin palette"

    candidate_map: dict[tuple[int, int], set[int]] = {}
    for r, c in blanks:
        candidates = _collect_candidates(grid, row_spans, col_spans, r, c, top, bottom, left, right)
        assert candidates, "every neutral cell must touch at least one marker color"
        assert candidates.issubset(set(palette)), "marker colors must come from the table palette"
        candidate_map[(r, c)] = candidates

    return {
        "board": board,
        "candidate_map": candidate_map,
        "palette": palette,
        "row_spans": row_spans,
        "col_spans": col_spans,
        "marker_mask": marker_mask,
        "blanks": blanks,
    }


def _cell_spans(lines: list[int]) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    for a, b in zip(lines, lines[1:]):
        assert b - a >= 2, "separator lines must have at least one cell row/column between them"
        spans.append((a + 1, b - 1))
    return spans


def _collect_candidates(
    grid: np.ndarray,
    row_spans: list[tuple[int, int]],
    col_spans: list[tuple[int, int]],
    row_idx: int,
    col_idx: int,
    top: int,
    bottom: int,
    left: int,
    right: int,
) -> set[int]:
    r0, r1 = row_spans[row_idx]
    c0, c1 = col_spans[col_idx]
    h, w = grid.shape
    candidates: set[int] = set()

    def add_if_marker(rr: int, cc: int) -> None:
        if not (0 <= rr < h and 0 <= cc < w):
            return
        if top <= rr <= bottom and left <= cc <= right:
            return
        value = int(grid[rr, cc])
        if value not in (BG, GRAY, NEUTRAL):
            candidates.add(value)

    for c in range(c0, c1 + 1):
        add_if_marker(r0 - 2, c)
        add_if_marker(r1 + 2, c)
    for r in range(r0, r1 + 1):
        add_if_marker(r, c0 - 2)
        add_if_marker(r, c1 + 2)
    return candidates


def _appearance_order(values: Iterable[int]) -> list[int]:
    seen: set[int] = set()
    ordered: list[int] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            ordered.append(int(value))
    return ordered


def _solve_latin_unique(
    board: np.ndarray,
    candidate_map: dict[tuple[int, int], set[int]],
    palette: list[int],
    max_solutions: int = 2,
) -> np.ndarray:
    work = np.array(board, dtype=int, copy=True)
    k = work.shape[0]
    palette_set = set(palette)
    color_order = {color: idx for idx, color in enumerate(palette)}

    row_missing = [set(palette) for _ in range(k)]
    col_missing = [set(palette) for _ in range(k)]
    for r in range(k):
        for c in range(k):
            value = int(work[r, c])
            if value == -1:
                continue
            assert value in palette_set, "fixed cell color must belong to the palette"
            assert value in row_missing[r], "row already contains a duplicate color"
            assert value in col_missing[c], "column already contains a duplicate color"
            row_missing[r].remove(value)
            col_missing[c].remove(value)

    blanks = [(r, c) for r in range(k) for c in range(k) if work[r, c] == -1]
    solution: np.ndarray | None = None
    count = 0

    def search() -> None:
        nonlocal solution, count
        if count >= max_solutions:
            return

        best_cell: tuple[int, int] | None = None
        best_domain: list[int] | None = None
        for r, c in blanks:
            if work[r, c] != -1:
                continue
            domain = row_missing[r] & col_missing[c] & candidate_map.get((r, c), palette_set)
            if not domain:
                return
            ordered = sorted(domain, key=color_order.__getitem__)
            if best_domain is None or len(ordered) < len(best_domain):
                best_cell = (r, c)
                best_domain = ordered
                if len(best_domain) == 1:
                    break

        if best_cell is None:
            count += 1
            if solution is None:
                solution = work.copy()
            return

        r, c = best_cell
        assert best_domain is not None
        for color in best_domain:
            work[r, c] = color
            row_missing[r].remove(color)
            col_missing[c].remove(color)
            search()
            row_missing[r].add(color)
            col_missing[c].add(color)
            work[r, c] = -1
            if count >= max_solutions:
                return

    search()
    assert count == 1 and solution is not None, "candidate-constrained Latin completion must be unique"
    return solution


def _gen_attempt() -> tuple[np.ndarray | None, dict[str, object] | None]:
    k = int(RNG.choice([4, 5]))
    cell_h, cell_w = _choose_cell_size(k)
    margins = _choose_margins()

    height = margins[0] + (k + 1) + k * cell_h + margins[1]
    width = margins[2] + (k + 1) + k * cell_w + margins[3]
    if max(height, width) > 30:
        return None, None

    orientation = str(RNG.choice(["rows", "cols"]))
    if orientation == "rows":
        line_a, line_b = 0, k - 1
        other = sorted(RNG.choice(k, size=2, replace=False).tolist())
        c1, c2 = other
        blank_positions = [(line_a, c1), (line_a, c2), (line_b, c1), (line_b, c2)]
    else:
        line_a, line_b = 0, k - 1
        other = sorted(RNG.choice(k, size=2, replace=False).tolist())
        r1, r2 = other
        blank_positions = [(r1, line_a), (r2, line_a), (r1, line_b), (r2, line_b)]

    palette = list(RNG.choice(PALETTE_POOL, size=k, replace=False))
    symbol_map = {idx: color for idx, color in enumerate(palette)}

    symbol_a, symbol_b = sorted(RNG.choice(k, size=2, replace=False).tolist())
    if orientation == "rows":
        intercalate = {
            (0, 0): symbol_a,
            (0, 1): symbol_b,
            (1, 0): symbol_b,
            (1, 1): symbol_a,
        }
        forced = {
            blank_positions[0]: symbol_a,
            blank_positions[1]: symbol_b,
            blank_positions[2]: symbol_b,
            blank_positions[3]: symbol_a,
        }
    else:
        forced = {
            blank_positions[0]: symbol_a,
            blank_positions[1]: symbol_b,
            blank_positions[2]: symbol_b,
            blank_positions[3]: symbol_a,
        }

    latin_symbols = _random_latin_with_forced_cells(k, forced)
    solution = np.vectorize(symbol_map.__getitem__, otypes=[int])(latin_symbols)

    candidate_map = _build_candidate_sets(solution, blank_positions)
    if not any(len(cands) == 1 for cands in candidate_map.values()):
        return None, None
    if not any(len(cands) > 1 for cands in candidate_map.values()):
        return None, None

    grid, row_spans, col_spans = _render_grid(solution, blank_positions, candidate_map, cell_h, cell_w, margins)
    meta = {
        "solution": solution,
        "blank_positions": blank_positions,
        "candidate_map": candidate_map,
        "palette": _appearance_order(solution.ravel().tolist()),
        "row_spans": row_spans,
        "col_spans": col_spans,
    }
    return grid, meta


def _choose_cell_size(k: int) -> tuple[int, int]:
    if k == 4:
        return int(RNG.choice([3, 4])), int(RNG.choice([3, 4]))
    return int(RNG.choice([3, 4])), int(RNG.choice([3, 4]))


def _choose_margins() -> tuple[int, int, int, int]:
    return (
        int(RNG.integers(1, 3)),
        int(RNG.integers(1, 3)),
        int(RNG.integers(1, 3)),
        int(RNG.integers(1, 3)),
    )


def _random_latin_with_forced_cells(k: int, forced: dict[tuple[int, int], int]) -> np.ndarray:
    perms = list(_permutations(k))
    order = list(range(k))
    constrained_rows = sorted({r for r, _ in forced})
    unconstrained_rows = [r for r in order if r not in constrained_rows]
    RNG.shuffle(unconstrained_rows)
    order = constrained_rows + unconstrained_rows

    col_used = [set() for _ in range(k)]
    rows = [None] * k

    def row_ok(r: int, perm: tuple[int, ...]) -> bool:
        for (rr, cc), value in forced.items():
            if rr == r and perm[cc] != value:
                return False
        return True

    def search(pos: int) -> bool:
        if pos == k:
            return True
        r = order[pos]
        perm_indices = RNG.permutation(len(perms))
        for idx in perm_indices:
            perm = perms[int(idx)]
            if not row_ok(r, perm):
                continue
            if any(perm[c] in col_used[c] for c in range(k)):
                continue
            rows[r] = perm
            for c in range(k):
                col_used[c].add(perm[c])
            if search(pos + 1):
                return True
            for c in range(k):
                col_used[c].remove(perm[c])
            rows[r] = None
        return False

    ok = search(0)
    assert ok, "failed to construct a Latin square with the requested intercalate"
    return np.array(rows, dtype=int)


def _permutations(k: int) -> list[tuple[int, ...]]:
    perms = _PERM_CACHE.get(k)
    if perms is None:
        perms = list(itertools.permutations(range(k)))
        _PERM_CACHE[k] = perms
    return perms


def _build_candidate_sets(
    solution: np.ndarray,
    blank_positions: list[tuple[int, int]],
) -> dict[tuple[int, int], set[int]]:
    values = [int(solution[r, c]) for r, c in blank_positions]
    unique_values = sorted(set(values))
    assert len(unique_values) == 2, "intercalate blanks must use exactly two colors"
    a, b = unique_values

    num_singletons = int(RNG.integers(1, 3))
    singleton_positions = set(RNG.choice(len(blank_positions), size=num_singletons, replace=False).tolist())

    candidate_map: dict[tuple[int, int], set[int]] = {}
    for idx, (r, c) in enumerate(blank_positions):
        true_color = int(solution[r, c])
        other_color = b if true_color == a else a
        if idx in singleton_positions:
            candidate_map[(r, c)] = {true_color}
        else:
            candidate_map[(r, c)] = {true_color, other_color}
    return candidate_map


def _render_grid(
    solution: np.ndarray,
    blank_positions: list[tuple[int, int]],
    candidate_map: dict[tuple[int, int], set[int]],
    cell_h: int,
    cell_w: int,
    margins: tuple[int, int, int, int],
) -> tuple[np.ndarray, list[tuple[int, int]], list[tuple[int, int]]]:
    k = solution.shape[0]
    top_margin, bottom_margin, left_margin, right_margin = margins
    height = top_margin + (k + 1) + k * cell_h + bottom_margin
    width = left_margin + (k + 1) + k * cell_w + right_margin

    grid = np.full((height, width), BG, dtype=int)
    sep_rows = [top_margin + i * (cell_h + 1) for i in range(k + 1)]
    sep_cols = [left_margin + i * (cell_w + 1) for i in range(k + 1)]
    grid[sep_rows, :] = GRAY
    grid[:, sep_cols] = GRAY

    row_spans = _cell_spans(sep_rows)
    col_spans = _cell_spans(sep_cols)
    blank_set = set(blank_positions)

    for r, (r0, r1) in enumerate(row_spans):
        for c, (c0, c1) in enumerate(col_spans):
            fill = NEUTRAL if (r, c) in blank_set else int(solution[r, c])
            grid[r0 : r1 + 1, c0 : c1 + 1] = fill

    for r, c in blank_positions:
        _place_markers_for_cell(grid, row_spans, col_spans, r, c, sorted(candidate_map[(r, c)]))

    return grid, row_spans, col_spans


def _place_markers_for_cell(
    grid: np.ndarray,
    row_spans: list[tuple[int, int]],
    col_spans: list[tuple[int, int]],
    r: int,
    c: int,
    colors: list[int],
) -> None:
    r0, r1 = row_spans[r]
    c0, c1 = col_spans[c]
    cell_h = r1 - r0 + 1
    cell_w = c1 - c0 + 1

    side_coords: dict[str, list[tuple[int, int]]] = {}
    if r == 0:
        side_coords["top"] = [(r0 - 2, cc) for cc in range(c0, c1 + 1)]
    if r == len(row_spans) - 1:
        side_coords["bottom"] = [(r1 + 2, cc) for cc in range(c0, c1 + 1)]
    if c == 0:
        side_coords["left"] = [(rr, c0 - 2) for rr in range(r0, r1 + 1)]
    if c == len(col_spans) - 1:
        side_coords["right"] = [(rr, c1 + 2) for rr in range(r0, r1 + 1)]
    assert side_coords, "blank cells must touch the outer table border"

    if len(colors) == 1:
        side = str(RNG.choice(sorted(side_coords)))
        coord = _choose_spaced_slots(len(side_coords[side]), 1)[0]
        rr, cc = side_coords[side][coord]
        grid[rr, cc] = colors[0]
        return

    split_allowed = len(side_coords) >= 2 and any(name in side_coords for name in ("top", "bottom")) and any(
        name in side_coords for name in ("left", "right")
    )
    if split_allowed and RNG.random() < 0.5:
        side_names = sorted(side_coords)
        chosen_sides = [side_names[0], side_names[1]]
        for color, side in zip(colors, chosen_sides):
            coord = _choose_spaced_slots(len(side_coords[side]), 1)[0]
            rr, cc = side_coords[side][coord]
            grid[rr, cc] = color
        return

    feasible_sides = [name for name, coords in side_coords.items() if _choose_spaced_slots(len(coords), len(colors), allow_none=True) is not None]
    assert feasible_sides, "need enough room to place all marker colors on an exposed side"
    side = str(RNG.choice(sorted(feasible_sides)))
    slots = _choose_spaced_slots(len(side_coords[side]), len(colors))
    for color, slot in zip(colors, slots):
        rr, cc = side_coords[side][slot]
        grid[rr, cc] = color


def _choose_spaced_slots(length: int, count: int, allow_none: bool = False) -> list[int] | None:
    options: list[tuple[int, ...]] = []
    for combo in itertools.combinations(range(length), count):
        if all(abs(a - b) >= 2 for a, b in zip(combo, combo[1:])):
            options.append(combo)
    if not options:
        if allow_none:
            return None
        raise AssertionError("no non-touching marker placement is available")
    return list(options[int(RNG.integers(len(options)))])


def _validate_generated_instance(grid: np.ndarray, meta: dict[str, object] | None) -> bool:
    if meta is None:
        return False

    parsed = _parse_puzzle(grid)
    solution = np.asarray(meta["solution"], dtype=int)
    blank_positions = set(tuple(pos) for pos in meta["blank_positions"])
    candidate_map = {tuple(k): set(v) for k, v in dict(meta["candidate_map"]).items()}

    if set(parsed["blanks"]) != blank_positions:
        return False
    if any(parsed["candidate_map"][pos] != candidate_map[pos] for pos in blank_positions):
        return False
    if not any(len(cands) == 1 for cands in candidate_map.values()):
        return False
    if not any(len(cands) > 1 for cands in candidate_map.values()):
        return False

    solved = _solve_latin_unique(parsed["board"], parsed["candidate_map"], parsed["palette"])
    if not np.array_equal(solved, solution):
        return False

    unrestricted = {pos: set(parsed["palette"]) for pos in blank_positions}
    if _solution_count(parsed["board"], unrestricted, parsed["palette"], limit=3) < 2:
        return False
    if _solution_count(parsed["board"], parsed["candidate_map"], parsed["palette"], limit=3) != 1:
        return False
    return True


def _solution_count(
    board: np.ndarray,
    candidate_map: dict[tuple[int, int], set[int]],
    palette: list[int],
    limit: int,
) -> int:
    work = np.array(board, dtype=int, copy=True)
    k = work.shape[0]
    palette_set = set(palette)

    row_missing = [set(palette) for _ in range(k)]
    col_missing = [set(palette) for _ in range(k)]
    for r in range(k):
        for c in range(k):
            value = int(work[r, c])
            if value == -1:
                continue
            if value not in row_missing[r] or value not in col_missing[c]:
                return 0
            row_missing[r].remove(value)
            col_missing[c].remove(value)

    blanks = [(r, c) for r in range(k) for c in range(k) if work[r, c] == -1]
    count = 0

    def search() -> None:
        nonlocal count
        if count >= limit:
            return
        best: tuple[int, int] | None = None
        best_domain: set[int] | None = None
        for r, c in blanks:
            if work[r, c] != -1:
                continue
            domain = row_missing[r] & col_missing[c] & candidate_map.get((r, c), palette_set)
            if not domain:
                return
            if best_domain is None or len(domain) < len(best_domain):
                best = (r, c)
                best_domain = domain
        if best is None:
            count += 1
            return
        r, c = best
        assert best_domain is not None
        for color in list(best_domain):
            work[r, c] = color
            row_missing[r].remove(color)
            col_missing[c].remove(color)
            search()
            row_missing[r].add(color)
            col_missing[c].add(color)
            work[r, c] = -1
            if count >= limit:
                return

    search()
    return count
