from __future__ import annotations

import random

import numpy as np


BACKGROUND = 0
BLUE = 1
RED = 2
SOURCE_COLORS = (3, 4, 5, 6, 7, 8, 9)

BAR_COUNT_CHOICES = (5, 6, 7, 8, 9)
BAR_COUNT_WEIGHTS = (3, 1, 4, 1, 4)
MAX_GENERATION_ATTEMPTS = 500


def _find_components(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    height, width = g.shape
    seen = np.zeros_like(g, dtype=bool)
    objects = []

    for r in range(height):
        for c in range(width):
            if seen[r, c] or int(g[r, c]) == BACKGROUND:
                continue
            color = int(g[r, c])
            stack = [(r, c)]
            seen[r, c] = True
            cells = []
            while stack:
                cr, cc = stack.pop()
                cells.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < height and 0 <= nc < width and not seen[nr, nc] and int(g[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))

            coords = np.array(cells, dtype=int)
            r0, c0 = coords.min(axis=0)
            r1, c1 = coords.max(axis=0)
            objects.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                }
            )
    return objects


def _bar_objects(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert np.any(g == BACKGROUND), "expected black background"

    objects = _find_components(g)
    assert objects, "expected at least one foreground bar"

    ordered = sorted(objects, key=lambda obj: (obj["bbox"][0], obj["bbox"][1]))
    used_widths = set()
    prev_bottom = -1
    for obj in ordered:
        r0, c0, r1, c1 = obj["bbox"]
        color = int(obj["color"])
        assert color not in (BACKGROUND,), "bars must be foreground"
        assert r0 > prev_bottom, "bars must have an unambiguous top-to-bottom order"
        prev_bottom = r1

        patch = g[r0 : r1 + 1, c0 : c1 + 1]
        assert np.all(patch == color), "each bar must be a solid monochromatic rectangle"
        width = c1 - c0 + 1
        height = r1 - r0 + 1
        assert width > height, "each object must be a horizontal bar"
        assert width not in used_widths, "bar lengths must be distinct"
        used_widths.add(width)
        obj["width"] = int(width)
        obj["height"] = int(height)
    return ordered


def _paint_by_rule(
    grid: np.ndarray,
    bars: list[dict],
    *,
    use_top: bool,
    use_bottom: bool,
    blue_on_double: bool,
) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    n_bars = len(bars)

    for idx, bar in enumerate(bars, start=1):
        length = int(bar["width"])
        top_rank = idx
        bottom_rank = n_bars + 1 - idx
        match_top = use_top and length == top_rank
        match_bottom = use_bottom and length == bottom_rank

        if blue_on_double and match_top and match_bottom:
            new_color = BLUE
        elif match_top or match_bottom:
            new_color = RED
        else:
            continue

        coords = bar["coords"]
        out[coords[:, 0], coords[:, 1]] = new_color
    return out


def _bar_match_kinds(bars: list[dict]) -> list[str]:
    n_bars = len(bars)
    kinds = []
    for idx, bar in enumerate(bars, start=1):
        length = int(bar["width"])
        top_rank = idx
        bottom_rank = n_bars + 1 - idx
        if length == top_rank == bottom_rank:
            kinds.append("both")
        elif length == top_rank:
            kinds.append("top")
        elif length == bottom_rank:
            kinds.append("bottom")
        else:
            kinds.append("none")
    return kinds


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each horizontal bar whose length matches its top or bottom rank, with double matches turning blue.

    Input structure:
    - Black cells are the background.
    - The foreground consists of separated solid monochromatic horizontal bars with pairwise distinct lengths.
    - The bars form a unique top-to-bottom order.

    Rule:
    - Number the bars from top to bottom to get rank t, and independently from bottom to top to get rank b.
    - If a bar's length equals exactly one of t or b, recolor the whole bar red.
    - If a bar's length equals both t and b, recolor the whole bar blue; otherwise leave it unchanged.

    Key insight:
    - Selection depends on a coincidence between visible bar length and two invisible ordinal counts; using only one counting direction fails on valid examples.
    """
    g = np.asarray(grid, dtype=int)
    bars = _bar_objects(g)
    return _paint_by_rule(g, bars, use_top=True, use_bottom=True, blue_on_double=True)


def _sample_bar_count() -> int:
    return int(random.choices(BAR_COUNT_CHOICES, weights=BAR_COUNT_WEIGHTS, k=1)[0])


def _sample_widths(n_bars: int) -> tuple[list[int], bool] | None:
    middle_rank = (n_bars + 1) // 2 if n_bars % 2 == 1 else None
    include_both = middle_rank is not None and random.random() < 0.7
    rank_choices = [rank for rank in range(2, n_bars) if rank != middle_rank]
    if len(rank_choices) < 2:
        return None

    for _ in range(100):
        top_idx = random.choice(rank_choices)
        bottom_idx = random.choice(rank_choices)
        if bottom_idx == top_idx:
            continue
        bottom_length = n_bars + 1 - bottom_idx
        if bottom_length == top_idx:
            continue
        if include_both and (top_idx == middle_rank or bottom_length == middle_rank):
            continue

        widths: list[int | None] = [None] * n_bars
        widths[top_idx - 1] = top_idx
        widths[bottom_idx - 1] = bottom_length
        if include_both:
            widths[middle_rank - 1] = middle_rank

        filler_slots = [i for i, width in enumerate(widths) if width is None]
        filler_pool = list(range(n_bars + 1, n_bars + 13))
        filler_lengths = random.sample(filler_pool, len(filler_slots))
        random.shuffle(filler_lengths)
        for slot, filler in zip(filler_slots, filler_lengths):
            widths[slot] = filler

        result = [int(width) for width in widths]
        if len(set(result)) == n_bars:
            return result, include_both
    return None


def _sample_layout(widths: list[int]) -> tuple[int, int, list[int], list[int]] | None:
    n_bars = len(widths)
    for _ in range(100):
        top_margin = random.randint(1, 3)
        bottom_margin = random.randint(1, 3)
        gaps = [random.randint(1, 2) for _ in range(n_bars - 1)]
        height = top_margin + n_bars + sum(gaps) + bottom_margin
        if height > 30:
            continue

        left_margin = random.randint(0, 5)
        right_margin = random.randint(0, 5)
        width = max(widths) + left_margin + right_margin
        if width > 30:
            continue

        rows = []
        row = top_margin
        for i in range(n_bars):
            rows.append(row)
            row += 1
            if i < n_bars - 1:
                row += gaps[i]
        offsets = [random.randint(0, width - bar_width) for bar_width in widths]
        return height, width, rows, offsets
    return None


def _sample_colors(n_bars: int) -> list[int]:
    palette_size = random.randint(3, min(5, len(SOURCE_COLORS)))
    palette = random.sample(SOURCE_COLORS, palette_size)
    return [int(random.choice(palette)) for _ in range(n_bars)]


def _render_input(widths: list[int], colors: list[int], layout: tuple[int, int, list[int], list[int]]) -> np.ndarray:
    height, width, rows, offsets = layout
    grid = np.full((height, width), BACKGROUND, dtype=int)
    for row, length, color, c0 in zip(rows, widths, colors, offsets):
        grid[row, c0 : c0 + length] = int(color)
    return grid


def _validate_instance(grid: np.ndarray, include_both: bool) -> None:
    bars = _bar_objects(grid)
    kinds = _bar_match_kinds(bars)
    actual = main(grid)
    expected = _paint_by_rule(grid, bars, use_top=True, use_bottom=True, blue_on_double=True)

    assert np.array_equal(actual, expected)
    assert kinds.count("top") == 1, "must include a top-only coincidence"
    assert kinds.count("bottom") == 1, "must include a bottom-only coincidence"
    assert kinds.count("none") >= 2, "must include unchanged distractor bars"
    assert kinds.count("both") == int(include_both), "branch coverage does not match the sampled plan"
    assert np.array_equal(actual[grid == BACKGROUND], grid[grid == BACKGROUND]), "background must stay black"
    assert not np.array_equal(actual, grid), "the transform must change at least one bar"

    top_only_guess = _paint_by_rule(grid, bars, use_top=True, use_bottom=False, blue_on_double=False)
    bottom_only_guess = _paint_by_rule(grid, bars, use_top=False, use_bottom=True, blue_on_double=False)
    assert not np.array_equal(top_only_guess, actual), "top-only shortcut must fail"
    assert not np.array_equal(bottom_only_guess, actual), "bottom-only shortcut must fail"

    if include_both:
        no_blue_guess = _paint_by_rule(grid, bars, use_top=True, use_bottom=True, blue_on_double=False)
        assert not np.array_equal(no_blue_guess, actual), "double-match branch must matter"
        assert np.count_nonzero(actual == BLUE) > 0, "blue branch should visibly appear"
    else:
        assert np.count_nonzero(actual == BLUE) == 0, "blue must be absent when no double match is planned"


def _gen_attempt() -> np.ndarray | None:
    n_bars = _sample_bar_count()
    width_plan = _sample_widths(n_bars)
    if width_plan is None:
        return None
    widths, include_both = width_plan
    layout = _sample_layout(widths)
    if layout is None:
        return None
    colors = _sample_colors(n_bars)
    grid = _render_input(widths, colors, layout)
    _validate_instance(grid, include_both)
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid bidirectional-rank bar puzzle")
