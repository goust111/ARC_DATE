from __future__ import annotations

from collections import Counter
import random

import numpy as np


PALETTE = tuple(range(1, 10))
BG = 0
GEN_RETRIES = 300


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _foreground_components(grid: np.ndarray, *, bg: int) -> list[list[tuple[int, int]]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) == bg:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) != bg:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(coords)
    return comps


def _is_chain_run(vals: np.ndarray) -> bool:
    n = int(len(vals))
    if n < 2 or n % 2 != 0:
        return False
    for i in range(0, n, 2):
        if int(vals[i]) == int(vals[i + 1]):
            return False
    for i in range(1, n - 2, 2):
        if int(vals[i]) != int(vals[i + 1]):
            return False
    return True


def _find_strip(grid: np.ndarray, *, bg: int) -> tuple[int, int, int, list[dict[str, int]]]:
    candidates: list[tuple[int, int, int, list[dict[str, int]]]] = []
    h, _w = grid.shape
    for r in range(h):
        cols = np.flatnonzero(grid[r] != bg)
        if len(cols) < 4:
            continue
        left = int(cols[0])
        right = int(cols[-1])
        vals = grid[r, left : right + 1]
        if int(vals[0]) == bg or int(vals[-1]) == bg:
            continue

        runs: list[tuple[bool, int, int]] = []
        i = 0
        while i < len(vals):
            j = i + 1
            is_fg = int(vals[i]) != bg
            while j < len(vals) and (int(vals[j]) != bg) == is_fg:
                j += 1
            runs.append((is_fg, i, j))
            i = j

        if len(runs) < 3 or not runs[0][0] or not runs[-1][0]:
            continue

        gaps: list[dict[str, int]] = []
        ok = True
        saw_long_segment = False
        for idx, (is_fg, start, end) in enumerate(runs):
            length = int(end - start)
            if is_fg:
                segment = vals[start:end]
                if not _is_chain_run(segment):
                    ok = False
                    break
                if length >= 4:
                    saw_long_segment = True
            else:
                if length not in (2, 4):
                    ok = False
                    break
                gaps.append(
                    {
                        "start": left + start,
                        "end": left + end - 1,
                        "width": length,
                        "left_color": int(vals[start - 1]),
                        "right_color": int(vals[end]),
                    }
                )
        if ok and gaps and saw_long_segment:
            candidates.append((r, left, right, gaps))

    assert len(candidates) == 1
    return candidates[0]


def _piece_type(colors: tuple[int, int] | list[int]) -> tuple[int, int]:
    a, b = map(int, colors)
    assert a != b
    return (a, b) if a < b else (b, a)


def _collect_scattered_pieces(
    grid: np.ndarray, *, bg: int, strip_row: int
) -> Counter[tuple[int, int]]:
    g = np.array(grid, copy=True)
    g[strip_row, :] = bg
    pieces: Counter[tuple[int, int]] = Counter()
    for comp in _foreground_components(g, bg=bg):
        assert len(comp) == 2
        rows = {r for r, _c in comp}
        cols = {c for _r, c in comp}
        assert (len(rows) == 1 and len(cols) == 2) or (len(rows) == 2 and len(cols) == 1)
        colors = [int(g[r, c]) for r, c in comp]
        pieces[_piece_type(colors)] += 1
    return pieces


def _choices_for_gap(gap: dict[str, int], counts: Counter[tuple[int, int]]) -> list[tuple[tuple[int, ...], Counter[tuple[int, int]]]]:
    left_color = int(gap["left_color"])
    right_color = int(gap["right_color"])
    width = int(gap["width"])
    out: list[tuple[tuple[int, ...], Counter[tuple[int, int]]]] = []

    if width == 2:
        needed = _piece_type((left_color, right_color))
        if counts[needed] > 0:
            out.append(((left_color, right_color), Counter({needed: 1})))
        return out

    seen_cells: set[tuple[int, ...]] = set()
    for t1, n1 in counts.items():
        if n1 <= 0 or left_color not in t1:
            continue
        mid = int(t1[0] if t1[1] == left_color else t1[1])
        if mid == left_color or mid == right_color:
            continue
        t2 = _piece_type((mid, right_color))
        if counts[t2] <= 0:
            continue
        need_second = 2 if t1 == t2 else 1
        if counts[t2] < need_second:
            continue
        cells = (left_color, mid, mid, right_color)
        if cells in seen_cells:
            continue
        seen_cells.add(cells)
        use = Counter({t1: 1, t2: 1})
        out.append((cells, use))
    return out


def _solve_gaps(
    gaps: list[dict[str, int]], counts: Counter[tuple[int, int]]
) -> list[tuple[int, ...]]:
    order = sorted(range(len(gaps)), key=lambda idx: int(gaps[idx]["width"]))
    fills: list[tuple[int, ...] | None] = [None] * len(gaps)
    solutions: set[tuple[tuple[int, ...], ...]] = set()

    def rec(pos: int) -> None:
        if len(solutions) > 1:
            return
        if pos == len(order):
            solutions.add(tuple(fill for fill in fills if fill is not None))
            return

        gap_idx = order[pos]
        choices = _choices_for_gap(gaps[gap_idx], counts)
        assert choices
        for cells, use in choices:
            for piece, n in use.items():
                counts[piece] -= n
            fills[gap_idx] = cells
            rec(pos + 1)
            fills[gap_idx] = None
            for piece, n in use.items():
                counts[piece] += n

    rec(0)
    assert len(solutions) == 1
    return [fill for fill in next(iter(solutions))]


def main(grid: np.ndarray) -> np.ndarray:
    """Fill the missing slots of a horizontal domino strip using scattered two-color dominos.

    Input structure:
    - A dominant background surrounds one horizontal strip made of 1x2 two-color dominos.
    - The strip is split by one or more internal background gaps of width 2 or 4 cells,
      and every visible strip segment still follows the seam pattern where adjacent dominos
      touch with the same color.
    - Away from the strip lie isolated 1x2 scattered dominos, including distractors.

    Rule:
    - Detect the strip row and its internal gaps.
    - Read each gap's boundary colors from the cells immediately to its left and right.
    - Insert one scattered domino into a width-2 gap, or two scattered dominos into a
      width-4 gap, allowing left-right flips but requiring every new seam to match colors.
    - Ignore unused distractors and output only the completed strip cropped to its bbox.

    Key insight:
    - The visible strip segments reveal the hidden domino chaining rule: equal colors occur
      only at seams between consecutive dominos, so each gap becomes a small exact cover
      problem over the scattered domino multiset.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _get_bg(g)

    strip_row, left, right, gaps = _find_strip(g, bg=bg)
    counts = _collect_scattered_pieces(g, bg=bg, strip_row=strip_row)
    fills = _solve_gaps(gaps, counts)

    out_row = g[strip_row, left : right + 1].copy()
    for gap, cells in zip(gaps, fills):
        start = int(gap["start"]) - left
        end = int(gap["end"]) - left + 1
        out_row[start:end] = np.array(cells, dtype=int)
    return out_row[np.newaxis, :]


def _full_strip_from_chain(chain: list[int]) -> np.ndarray:
    cells: list[int] = []
    for a, b in zip(chain, chain[1:]):
        cells.extend((int(a), int(b)))
    return np.array(cells, dtype=int)


def _enumerate_gap_plans(n_dominos: int) -> list[list[tuple[int, int]]]:
    plans: list[list[tuple[int, int]]] = []
    for width1 in (1, 2):
        for start1 in range(1, n_dominos - width1):
            if start1 + width1 > n_dominos - 1:
                continue
            segs = [start1, n_dominos - (start1 + width1)]
            if min(segs) >= 1 and max(segs) >= 2:
                plans.append([(start1, width1)])
    for width1 in (1, 2):
        for start1 in range(1, n_dominos - width1):
            end1 = start1 + width1
            if end1 > n_dominos - 1:
                continue
            for width2 in (1, 2):
                for start2 in range(end1 + 1, n_dominos - width2):
                    end2 = start2 + width2
                    if end2 > n_dominos - 1:
                        continue
                    segs = [start1, start2 - end1, n_dominos - end2]
                    if min(segs) >= 1 and max(segs) >= 2:
                        plans.append([(start1, width1), (start2, width2)])
    assert plans
    return plans


def _can_place(grid: np.ndarray, cells: list[tuple[int, int]]) -> bool:
    h, w = grid.shape
    cell_set = set(cells)
    for r, c in cells:
        if not (0 <= r < h and 0 <= c < w) or int(grid[r, c]) != BG:
            return False
    for r, c in cells:
        for rr in range(r - 1, r + 2):
            for cc in range(c - 1, c + 2):
                if 0 <= rr < h and 0 <= cc < w and (rr, cc) not in cell_set and int(grid[rr, cc]) != BG:
                    return False
    return True


def _place_dominos(
    grid: np.ndarray, dominos: list[tuple[int, int]], strip_row: int
) -> bool:
    h, w = grid.shape
    touched_rows: set[int] = {strip_row}
    for colors in dominos:
        options: list[tuple[list[tuple[int, int]], tuple[int, int]]] = []
        for orientation in ("h", "v"):
            if orientation == "h":
                for r in range(h):
                    if r in touched_rows:
                        continue
                    for c in range(w - 1):
                        cells = [(r, c), (r, c + 1)]
                        if _can_place(grid, cells):
                            options.append((cells, (r,)))
            else:
                for r in range(h - 1):
                    rows_used = (r, r + 1)
                    if any(row in touched_rows for row in rows_used):
                        continue
                    for c in range(w):
                        cells = [(r, c), (r + 1, c)]
                        if _can_place(grid, cells):
                            options.append((cells, rows_used))
        if not options:
            return False
        cells, rows_used = random.choice(options)
        a, b = colors if random.random() < 0.5 else (colors[1], colors[0])
        grid[cells[0]] = int(a)
        grid[cells[1]] = int(b)
        touched_rows.update(rows_used)
    return True


def _gen_attempt() -> np.ndarray | None:
    n_dominos = random.randint(4, 6)
    gap_plan = random.choice(_enumerate_gap_plans(n_dominos))
    chain = random.sample(PALETTE, n_dominos + 1)
    full_row = _full_strip_from_chain(chain)
    out = full_row[np.newaxis, :].copy()

    missing_dominos: list[tuple[int, int]] = []
    visible = full_row.copy()
    for start, width in gap_plan:
        for idx in range(start, start + width):
            missing_dominos.append((int(chain[idx]), int(chain[idx + 1])))
        cell_start = 2 * start
        cell_end = 2 * (start + width)
        visible[cell_start:cell_end] = BG

    spare_colors = [c for c in PALETTE if c not in chain]
    if len(spare_colors) < 2:
        return None
    distractor_type = tuple(random.sample(spare_colors, 2))
    distractor_count = random.randint(1, 3)
    scattered = missing_dominos + [distractor_type] * distractor_count
    random.shuffle(scattered)

    width = int(len(full_row))
    min_h = max(10, len(scattered) + 4)
    max_h = min(20, len(scattered) + 9)
    h = random.randint(min_h, max_h)
    min_w = max(width + 4, 14)
    max_w = min(26, width + 10)
    w = random.randint(min_w, max_w)

    grid = np.full((h, w), BG, dtype=int)
    strip_row = random.randint(2, h - 3)
    strip_col = random.randint(1, w - width - 1)
    grid[strip_row, strip_col : strip_col + width] = visible

    if not _place_dominos(grid, scattered, strip_row):
        return None

    try:
        result = main(grid)
    except Exception:
        return None
    if not np.array_equal(result, out):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
