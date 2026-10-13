import numpy as np

from arc_utils import Color, find_components, get_bg
import random


def _operand_bars(g: np.ndarray, bg: int) -> list[dict[str, int]]:
    height = g.shape[0]
    components = find_components(g, bg=bg, connectivity=4, monochromatic=True)
    assert components

    bars: list[dict[str, int]] = []
    for comp in components:
        color = int(comp["color"])
        assert color in (Color.TEAL, Color.RED)

        r0, c0, r1, c1 = map(int, comp["bbox"])
        area = int(comp["area"])
        assert c0 == c1
        assert r1 == height - 1
        assert area == r1 - r0 + 1
        assert bool(np.all(g[r0 : r1 + 1, c0] == color))

        bars.append({"color": color, "col": c0, "height": area})

    bars.sort(key=lambda bar: bar["col"])
    assert len({bar["col"] for bar in bars}) == len(bars)
    assert any(bar["color"] == Color.TEAL for bar in bars)
    return bars


def main(grid: np.ndarray) -> np.ndarray:
    """Compute a grey result bar by adding teal bar heights and subtracting red bar heights.

    Input structure
    - The grid has a dominant background and several bottom-aligned one-cell-wide vertical bars.
    - Teal bars are positive operands; red bars are negative operands.
    - The penultimate column is an empty result slot.

    Rule
    - Keep every input bar unchanged.
    - Compute result height = total teal bar height - total red bar height.
    - Paint that many bottom cells of the penultimate column grey.

    Key insight
    - The bars encode arithmetic by height, with color providing the sign.
    """
    g = np.asarray(grid, dtype=int)
    height, width = g.shape
    bg = get_bg(g)
    bars = _operand_bars(g, bg)

    result_height = 0
    for bar in bars:
        if bar["color"] == Color.TEAL:
            result_height += bar["height"]
        else:
            assert bar["color"] == Color.RED
            result_height -= bar["height"]

    assert 0 < result_height <= height
    assert width >= 2
    result_col = width - 2
    assert bool(np.all(g[:, result_col] == bg))

    out = g.copy()
    out[height - result_height : height, result_col] = Color.GREY
    return out

# --------------- generator (att2, o5x) ---------------

# Canvas, object, and result-size bounds.
BAR_COUNT_RANGE = (2, 6)
GRID_HEIGHT_RANGE = (6, 12)
TOP_MARGIN_ROWS, RESULT_HEIGHT_MIN = 1, 1
# Column layout.
BAR_COL_START, BAR_COL_STRIDE, WIDTH_EXTRA_COLS, RESULT_COL_FROM_RIGHT = 1, 2, 3, 2
# Color palette and retry budget.
BG_COLOR, RESULT_COLOR, POSITIVE_COLOR, NEGATIVE_COLOR = Color.ORANGE, Color.GREY, Color.TEAL, Color.RED
GEN_RETRY_LIMIT = 300


def _bounded_composition(total: int, parts: int, *, upper: int) -> list[int] | None:
    """Sample positive bounded parts summing to total, or None if impossible."""
    if parts == 0:
        return [] if total == 0 else None
    if not (parts <= total <= parts * upper):
        return None
    values = []
    remaining = total
    for i in range(parts):
        slots = parts - i - 1
        lo = max(1, remaining - slots * upper)
        hi = min(upper, remaining - slots)
        value = random.randint(lo, hi)
        values.append(value)
        remaining -= value
    random.shuffle(values)
    return values


def _sample_bar_specs(n: int, max_bar_height: int) -> tuple[list[tuple[int, int]], int] | None:
    """Sample shuffled operand specs and the matching grey result height."""
    result_height = random.randint(RESULT_HEIGHT_MIN, max_bar_height)
    feasible_teal_counts = []
    for teal_count in range(1, n + 1):
        red_count = n - teal_count
        red_sum_lo = max(red_count, teal_count - result_height)
        red_sum_hi = min(red_count * max_bar_height, teal_count * max_bar_height - result_height)
        if red_sum_lo <= red_sum_hi:
            feasible_teal_counts.append((teal_count, red_sum_lo, red_sum_hi))
    if not feasible_teal_counts:
        return None

    teal_count, red_sum_lo, red_sum_hi = random.choice(feasible_teal_counts)
    red_count = n - teal_count
    red_sum = random.randint(red_sum_lo, red_sum_hi)
    teal_sum = red_sum + result_height
    teal_heights = _bounded_composition(teal_sum, teal_count, upper=max_bar_height)
    red_heights = _bounded_composition(red_sum, red_count, upper=max_bar_height)
    if teal_heights is None or red_heights is None:
        return None

    bar_specs = [(POSITIVE_COLOR, h) for h in teal_heights] + [(NEGATIVE_COLOR, h) for h in red_heights]
    random.shuffle(bar_specs)
    return bar_specs, result_height


def _gen_attempt() -> np.ndarray | None:
    n = random.randint(*BAR_COUNT_RANGE)
    height = random.randint(*GRID_HEIGHT_RANGE)
    max_bar_height = height - TOP_MARGIN_ROWS
    width = BAR_COL_STRIDE * n + WIDTH_EXTRA_COLS
    if (sampled := _sample_bar_specs(n, max_bar_height)) is None:
        return None
    bar_specs, result_height = sampled
    out = np.full((height, width), BG_COLOR, dtype=int)
    for i, (color, bar_height) in enumerate(bar_specs):
        col = BAR_COL_START + BAR_COL_STRIDE * i
        out[height - bar_height : height, col] = color
    out[height - result_height : height, width - RESULT_COL_FROM_RIGHT] = RESULT_COLOR

    inp = out.copy()
    inp[:, width - RESULT_COL_FROM_RIGHT] = BG_COLOR
    if not np.array_equal(main(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
