from __future__ import annotations

import random
from collections import Counter

import numpy as np


BG = 0
BASELINE = 5
MARKER = 7
BAR_COLORS = (1, 2, 3, 4)
BAR_COUNT = 4
COUNT_SETS = (
    (1, 2, 3, 4),
    (1, 2, 3, 5),
    (1, 2, 4, 5),
    (1, 3, 4, 5),
    (2, 3, 4, 5),
)
MAX_ATTEMPTS = 400


def _get_bg(grid: np.ndarray) -> int:
    vals, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(vals[int(counts.argmax())])


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components_of_color(grid: np.ndarray, color: int) -> list[list[tuple[int, int]]]:
    h, w = grid.shape
    seen: set[tuple[int, int]] = set()
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if int(grid[r, c]) != int(color) or (r, c) in seen:
                continue
            stack = [(r, c)]
            seen.add((r, c))
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in seen and int(grid[nr, nc]) == int(color):
                        seen.add((nr, nc))
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _find_baseline(grid: np.ndarray, bg: int) -> tuple[int, int]:
    rows = [r for r in range(grid.shape[0]) if int(grid[r, 0]) != int(bg) and np.all(grid[r, :] == grid[r, 0])]
    assert len(rows) == 1, "expected exactly one full-width baseline row"
    r = int(rows[0])
    return r, int(grid[r, 0])


def _extract_bars(grid: np.ndarray, base_r: int, base_color: int, bg: int) -> list[dict[str, int]]:
    h, w = grid.shape
    bars: list[dict[str, int]] = []
    covered: set[tuple[int, int]] = set()
    for c in range(w):
        rows = [r for r in range(base_r) if int(grid[r, c]) not in {int(bg), int(base_color), MARKER}]
        if not rows:
            continue
        colors = {int(grid[r, c]) for r in rows}
        assert len(colors) == 1, "each bar must be monochrome"
        top = min(rows)
        bottom = max(rows)
        assert bottom == base_r - 1, "bars must stand directly above the baseline"
        assert rows == list(range(top, bottom + 1)), "bar cells must be vertically contiguous"
        color = int(next(iter(colors)))
        for r in rows:
            covered.add((r, c))
        bars.append({"x": int(c), "color": color, "height": int(len(rows)), "top": int(top)})
    bars.sort(key=lambda bar: bar["x"])
    assert len(bars) == BAR_COUNT, f"expected {BAR_COUNT} bars"
    fg_above = {(r, c) for r in range(base_r) for c in range(w) if int(grid[r, c]) != int(bg)}
    marker_cells = {(r, c) for r in range(base_r) for c in range(w) if int(grid[r, c]) == MARKER}
    assert fg_above == covered | marker_cells, "unexpected foreground above baseline"
    return bars


def _extract_dominoes(grid: np.ndarray, base_r: int) -> list[dict[str, int]]:
    dominoes: list[dict[str, int]] = []
    covered: set[tuple[int, int]] = set()
    for comp in _components_of_color(grid[:base_r, :], MARKER):
        assert len(comp) == 2, "each marker must be a two-pixel domino"
        rows = sorted({r for r, _ in comp})
        cols = sorted(c for _, c in comp)
        assert len(rows) == 1 and cols[1] == cols[0] + 1, "each marker must be horizontal"
        covered.update(comp)
        dominoes.append({"row": int(rows[0]), "left": int(cols[0]), "right": int(cols[1])})
    assert covered == {(r, c) for r in range(base_r) for c in range(grid.shape[1]) if int(grid[r, c]) == MARKER}
    dominoes.sort(key=lambda d: (d["row"], d["left"]))
    assert dominoes, "expected at least one marker"
    return dominoes


def _absorption_counts(grid: np.ndarray, bars: list[dict[str, int]], dominoes: list[dict[str, int]], bg: int) -> list[int]:
    h, w = grid.shape
    bar_at = {(r, bar["x"]): i for i, bar in enumerate(bars) for r in range(bar["top"], bar["top"] + bar["height"])}
    counts = [0] * len(bars)
    for dom in dominoes:
        r = int(dom["row"])
        c = int(dom["right"]) + 1
        while c < w and int(grid[r, c]) == int(bg):
            c += 1
        assert 0 <= r < h and c < w, "each ray must terminate at a bar"
        assert (r, c) in bar_at, "ray must be absorbed by a bar pixel"
        counts[bar_at[(r, c)]] += 1
    return counts


def _render_output(shape: tuple[int, int], base_r: int, base_color: int, slots: list[dict[str, int]], sorted_heights: list[int], bg: int) -> np.ndarray:
    h, w = shape
    out = np.full((h, w), int(bg), dtype=int)
    out[base_r, :] = int(base_color)
    for slot, height in zip(slots, sorted_heights):
        x = int(slot["x"])
        color = int(slot["color"])
        out[base_r - int(height):base_r, x] = color
    return out


def _analyze(grid: np.ndarray) -> tuple[int, int, int, list[dict[str, int]], list[dict[str, int]], list[int]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0
    bg = _get_bg(g)
    base_r, base_color = _find_baseline(g, bg)
    bars = _extract_bars(g, base_r, base_color, bg)
    dominoes = _extract_dominoes(g, base_r)
    counts = _absorption_counts(g, bars, dominoes, bg)
    return bg, base_r, base_color, bars, dominoes, counts


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reorder baseline-mounted vertical bars by how many horizontal marker rays they absorb.

    Input structure:
    - The background is mostly black, with one full-width grey baseline near the bottom.
    - Four single-column colored bars stand directly above the baseline in fixed left-to-right slots.
    - Several orange horizontal two-pixel dominoes lie above the baseline to the left of the bar region.

    Rule:
    - Each orange domino emits one horizontal ray to the right, starting immediately after its right end and continuing through black cells until the first non-black cell.
    - If that first non-black cell is a bar pixel, that bar absorbs the ray; count absorbed rays per bar.
    - Sort the bar heights by absorbed-ray count from small to large, then place the sorted heights back into the original left-to-right slots while keeping each slot's original bar color; erase the rays and preserve the baseline.

    Key insight:
    - The bars are both obstacles and score targets: which rows reach which bar determines the sort key that later redistributes the bar silhouettes.
    """
    bg, base_r, base_color, bars, _dominoes, counts = _analyze(grid)
    slots = sorted(bars, key=lambda bar: bar["x"])
    shapes = sorted(zip(bars, counts), key=lambda item: (int(item[1]), int(item[0]["x"])))
    sorted_heights = [int(bar["height"]) for bar, _count in shapes]
    return _render_output(np.asarray(grid).shape, base_r, base_color, slots, sorted_heights, bg)


def _pick_counts() -> list[int]:
    counts = list(random.choice(COUNT_SETS))
    random.shuffle(counts)
    if counts == sorted(counts):
        counts[0], counts[1] = counts[1], counts[0]
    return counts


def _pick_slot_columns(width: int) -> list[int] | None:
    for _ in range(80):
        cols = sorted(random.sample(range(8, width - 2), BAR_COUNT))
        if min(b - a for a, b in zip(cols, cols[1:])) >= 2:
            return cols
    return None


def _marker_starts(first_bar_x: int, row_count: int) -> list[int] | None:
    max_start = first_bar_x - 3
    starts = list(range(1, max_start + 1))
    pairs = [(a, b) for a in starts for b in starts if abs(a - b) >= 2]
    if not pairs:
        return None
    a, b = random.choice(pairs)
    phase = random.randint(0, 1)
    return [a if (i + phase) % 2 == 0 else b for i in range(row_count)]


def _build_case() -> tuple[np.ndarray, np.ndarray, list[int], list[int]] | None:
    counts = _pick_counts()
    heights = list(np.cumsum(counts))
    top_margin = random.randint(1, 4)
    bottom_margin = random.randint(1, 3)
    base_r = top_margin + heights[-1]
    height = base_r + 1 + bottom_margin
    width = random.randint(18, 26)
    cols = _pick_slot_columns(width)
    if cols is None:
        return None
    starts = _marker_starts(cols[0], heights[-1])
    if starts is None:
        return None

    colors = random.sample(list(BAR_COLORS), BAR_COUNT)
    inp = np.full((height, width), BG, dtype=int)
    inp[base_r, :] = BASELINE
    for x, color, bar_h in zip(cols, colors, heights):
        inp[base_r - bar_h:base_r, x] = int(color)

    row_cursor = base_r
    marker_rows: list[int] = []
    for count in counts:
        rows = list(range(row_cursor - count, row_cursor))
        marker_rows.extend(rows)
        row_cursor -= count
    assert len(marker_rows) == heights[-1]
    marker_rows.sort()
    for r, c0 in zip(marker_rows, starts):
        inp[r, c0:c0 + 2] = MARKER

    slots = [{"x": int(x), "color": int(color)} for x, color in zip(cols, colors)]
    shapes = sorted(zip(heights, counts), key=lambda item: (int(item[1]), int(item[0])))
    out_heights = [int(h) for h, _cnt in shapes]
    out = _render_output(inp.shape, base_r, BASELINE, slots, out_heights, BG)
    return inp, out, heights, counts


def _validate_generated(inp: np.ndarray, expected: np.ndarray, heights: list[int], counts: list[int]) -> bool:
    try:
        bg, base_r, base_color, bars, dominoes, found_counts = _analyze(inp)
    except AssertionError:
        return False
    if bg != BG or base_color != BASELINE:
        return False
    if [bar["height"] for bar in bars] != heights:
        return False
    if found_counts != counts:
        return False
    if sorted(found_counts) != sorted(set(found_counts)):
        return False
    if min(found_counts) <= 0:
        return False
    if len(dominoes) != sum(counts):
        return False
    if np.array_equal(inp, expected):
        return False
    produced = main(inp)
    if not np.array_equal(produced, expected):
        return False
    # Later bars must genuinely depend on earlier shorter bars being skipped on upper rows.
    if not all(bars[i]["height"] > bars[i - 1]["height"] for i in range(1, len(bars))):
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        built = _build_case()
        if built is None:
            continue
        inp, expected, heights, counts = built
        if _validate_generated(inp, expected, heights, counts):
            return inp
    raise RuntimeError("failed to generate a valid input for puzzle 3")
