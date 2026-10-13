from __future__ import annotations

import random
from typing import Iterable

import numpy as np


BG = 0
PIVOT_COLOR = 6
RANK_COLORS = [1, 2, 3, 4, 5]
BAR_COLOR_POOL = [1, 2, 3, 4, 5, 7, 8, 9]

MIN_BARS = 3
MAX_BARS = 5
GEN_RETRY_LIMIT = 400
PAIR_RETRY_LIMIT = 250


def _neighbors8(shape: tuple[int, int], r: int, c: int) -> Iterable[tuple[int, int]]:
    height, width = shape
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            rr = r + dr
            cc = c + dc
            if 0 <= rr < height and 0 <= cc < width:
                yield rr, cc


def _rotate_ccw(pivot: tuple[int, int], source: tuple[int, int]) -> tuple[int, int]:
    pr, pc = pivot
    sr, sc = source
    dr = sr - pr
    dc = sc - pc
    return pr - dc, pc + dr


def _find_bars(grid: np.ndarray) -> dict[int, dict[str, int]]:
    height, width = grid.shape
    bars: dict[int, dict[str, int]] = {}
    for r in range(height):
        row_hits: list[tuple[int, int]] = []
        for c in range(1, width - 1):
            if int(grid[r, c]) != BG:
                continue
            left = int(grid[r, c - 1])
            right = int(grid[r, c + 1])
            if left == right and left not in (BG, PIVOT_COLOR):
                row_hits.append((left, c))
        assert len(row_hits) <= 1, "expected at most one one-pixel bar gap per row"
        for color, gap_col in row_hits:
            assert color not in bars, "expected one horizontal bar per color"
            bars[color] = {"row": int(r), "gap_col": int(gap_col)}
    assert MIN_BARS <= len(bars) <= MAX_BARS, "expected 3 to 5 bars"
    return bars


def _find_isolated_source(grid: np.ndarray, color: int) -> tuple[int, int]:
    coords = np.argwhere(grid == int(color))
    isolated: list[tuple[int, int]] = []
    for rr, cc in coords:
        r = int(rr)
        c = int(cc)
        if all(int(grid[nr, nc]) != int(color) for nr, nc in _neighbors8(grid.shape, r, c)):
            isolated.append((r, c))
    assert len(isolated) == 1, f"expected one isolated source pixel for color {color}"
    return isolated[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each broken horizontal bar's gap with the rank color induced by a rotated source-pivot pair.

    Input structure:
    - The background is black, with 3 to 5 single-color horizontal bars on distinct rows.
    - Each bar has exactly one one-pixel black gap, and each bar color also appears once as an isolated source pixel.
    - Magenta pivot pixels sit in the right-side empty area, with each source sharing exactly one row or column with its matching pivot.

    Rule:
    - For each isolated source pixel, find the unique magenta pivot in the same row or column and rotate the source 90 degrees counterclockwise around that pivot.
    - Use the rotated target row as that bar's hidden sorting key, matching the source to its bar by color.
    - Sort bars by those target rows from top to bottom, then fill their gap pixels with rank colors 1, 2, 3, 4, 5 in that order while keeping every other pixel unchanged.

    Key insight:
    - The rotation is never drawn; it only produces an ordering key that decides which gap gets which rank color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0

    bars = _find_bars(g)
    pivots = [tuple(map(int, coord)) for coord in np.argwhere(g == PIVOT_COLOR)]
    assert len(pivots) == len(bars), "expected one pivot per bar"

    keyed_bars: list[tuple[int, int, int, int]] = []
    for color, bar in bars.items():
        source = _find_isolated_source(g, color)
        aligned = [pivot for pivot in pivots if pivot[0] == source[0] or pivot[1] == source[1]]
        assert len(aligned) == 1, f"expected a unique pivot for source color {color}"
        target_row, _target_col = _rotate_ccw(aligned[0], source)
        keyed_bars.append((int(target_row), int(color), int(bar["row"]), int(bar["gap_col"])))

    keyed_bars.sort(key=lambda item: item[0])
    assert len(keyed_bars) <= len(RANK_COLORS)
    assert len({item[0] for item in keyed_bars}) == len(keyed_bars), "rotated target rows must be distinct"

    out = g.copy()
    for rank, (_target_row, _color, row, gap_col) in enumerate(keyed_bars, start=1):
        out[row, gap_col] = RANK_COLORS[rank - 1]
    return out


def _sample_spaced_rows(height: int, count: int) -> list[int]:
    rows = list(range(1, height - 1))
    for _ in range(200):
        chosen = sorted(random.sample(rows, count))
        if all(chosen[i + 1] - chosen[i] >= 2 for i in range(count - 1)):
            return chosen
    raise RuntimeError("failed to sample non-adjacent bar rows")


def _pair_cells_clear(pairs: list[dict[str, tuple[int, int] | str]]) -> bool:
    cells: list[tuple[int, int, int]] = []
    for idx, pair in enumerate(pairs):
        pr, pc = pair["pivot"]
        sr, sc = pair["source"]
        cells.append((int(pr), int(pc), idx))
        cells.append((int(sr), int(sc), idx))
    for i, (r1, c1, idx1) in enumerate(cells):
        for r2, c2, idx2 in cells[i + 1 :]:
            if idx1 == idx2:
                continue
            if max(abs(r1 - r2), abs(c1 - c2)) <= 1:
                return False
    return len({(r, c) for r, c, _ in cells}) == len(cells)


def _build_pairs(height: int, width: int, bar_rows: list[int], orientations: list[str]) -> list[dict[str, tuple[int, int] | str]] | None:
    pair_cols = list(range(width - 6, width))
    pair_rows = [r for r in range(1, height - 1) if r not in set(bar_rows)]
    if len(pair_rows) < len(orientations) + 1:
        return None

    for _ in range(PAIR_RETRY_LIMIT):
        pivot_cols = random.sample(pair_cols, len(orientations))
        pivot_rows = random.sample(pair_rows, len(orientations))
        safe_source_cols = [c for c in pair_cols if c not in pivot_cols]
        if not safe_source_cols:
            continue
        horizontal_source_col = random.choice(safe_source_cols)
        safe_rows = [r for r in pair_rows if r not in pivot_rows]
        if not safe_rows:
            continue

        pairs: list[dict[str, tuple[int, int] | str]] = []
        failed = False
        for orientation, pivot_row, pivot_col in zip(orientations, pivot_rows, pivot_cols):
            pr = int(pivot_row)
            pc = int(pivot_col)
            if orientation == "H":
                source = (pr, int(horizontal_source_col))
            else:
                valid_source_rows = []
                for sr in safe_rows:
                    row_delta = int(sr - pr)
                    if row_delta == 0 or abs(row_delta) > 3:
                        continue
                    target = _rotate_ccw((pr, pc), (int(sr), pc))
                    if 0 <= target[0] < height and 0 <= target[1] < width:
                        valid_source_rows.append(int(sr))
                if not valid_source_rows:
                    failed = True
                    break
                source_row = random.choice(valid_source_rows)
                source = (int(source_row), pc)

            target = _rotate_ccw((pr, pc), source)
            if not (0 <= target[0] < height and 0 <= target[1] < width):
                failed = True
                break
            pairs.append(
                {
                    "orientation": orientation,
                    "pivot": (pr, pc),
                    "source": source,
                    "target": (int(target[0]), int(target[1])),
                }
            )

        if failed:
            continue
        if len({pair["target"][0] for pair in pairs}) != len(pairs):
            continue
        pivots = [pair["pivot"] for pair in pairs]
        sources = [pair["source"] for pair in pairs]
        if any(sum(1 for pivot in pivots if pivot[0] == sr or pivot[1] == sc) != 1 for sr, sc in sources):
            continue
        if any(sum(1 for source in sources if source[0] == pr or source[1] == pc) != 1 for pr, pc in pivots):
            continue
        if not _pair_cells_clear(pairs):
            continue
        return pairs
    return None


def _assign_colors(bar_ranks: list[int]) -> list[int] | None:
    order = list(range(len(bar_ranks)))
    random.shuffle(order)
    assigned = [0] * len(bar_ranks)

    def backtrack(pos: int, available: list[int]) -> bool:
        if pos == len(order):
            return True
        idx = order[pos]
        forbidden = RANK_COLORS[bar_ranks[idx] - 1]
        choices = [color for color in available if color != forbidden]
        random.shuffle(choices)
        for color in choices:
            assigned[idx] = int(color)
            remaining = [value for value in available if value != color]
            if backtrack(pos + 1, remaining):
                return True
        assigned[idx] = 0
        return False

    if backtrack(0, BAR_COLOR_POOL.copy()):
        return assigned
    return None


def _build_bars(width: int, bar_rows: list[int]) -> list[dict[str, int]]:
    bars: list[dict[str, int]] = []
    right_limit = width - 8
    for row in bar_rows:
        left = random.randint(0, 1)
        right = random.randint(max(left + 6, right_limit - 1), right_limit)
        gap_col = random.randint(left + 2, right - 2)
        bars.append({"row": int(row), "left": int(left), "right": int(right), "gap_col": int(gap_col)})
    return bars


def _gen_attempt() -> np.ndarray | None:
    bar_count = random.choices([3, 4, 5], weights=[1, 2, 3], k=1)[0]
    height = random.randint(max(14, 2 * bar_count + 7), 22)
    width = random.randint(16, 22)

    try:
        bar_rows = _sample_spaced_rows(height, bar_count)
    except RuntimeError:
        return None

    bars = _build_bars(width, bar_rows)
    orientations = ["H", "V"] + [random.choice(["H", "V"]) for _ in range(bar_count - 2)]
    random.shuffle(orientations)
    pairs = _build_pairs(height, width, bar_rows, orientations)
    if pairs is None:
        return None

    pair_rank: dict[int, int] = {}
    for rank, pair_idx in enumerate(sorted(range(bar_count), key=lambda idx: pairs[idx]["target"][0]), start=1):
        pair_rank[pair_idx] = rank

    pair_for_bar = list(range(bar_count))
    random.shuffle(pair_for_bar)
    bar_ranks = [pair_rank[pair_idx] for pair_idx in pair_for_bar]
    bar_colors = _assign_colors(bar_ranks)
    if bar_colors is None:
        return None

    pair_color: dict[int, int] = {}
    for bar_idx, pair_idx in enumerate(pair_for_bar):
        pair_color[pair_idx] = bar_colors[bar_idx]

    inp = np.zeros((height, width), dtype=int)
    expected = np.zeros((height, width), dtype=int)

    for bar_idx, bar in enumerate(bars):
        color = bar_colors[bar_idx]
        row = bar["row"]
        left = bar["left"]
        right = bar["right"]
        gap_col = bar["gap_col"]
        inp[row, left : right + 1] = color
        inp[row, gap_col] = BG
        expected[row, left : right + 1] = color
        expected[row, gap_col] = RANK_COLORS[bar_ranks[bar_idx] - 1]

    for pair_idx, pair in enumerate(pairs):
        source_r, source_c = pair["source"]
        pivot_r, pivot_c = pair["pivot"]
        inp[pivot_r, pivot_c] = PIVOT_COLOR
        inp[source_r, source_c] = pair_color[pair_idx]
        expected[pivot_r, pivot_c] = PIVOT_COLOR
        expected[source_r, source_c] = pair_color[pair_idx]

    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        generated = _gen_attempt()
        if generated is not None:
            return generated
    raise RuntimeError("failed to generate a valid input for puzzle 6")
