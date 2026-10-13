import numpy as np

from arc_utils import find_components, get_bg
import itertools
import random


MARKER_COLOR = 1
BODY_COLOR = 2


def _vertical_bars(g: np.ndarray, bg: int) -> list[dict]:
    comps = find_components(g, bg=bg, connectivity=4, monochromatic=True)
    bars = []
    baseline = None
    for comp in comps:
        r0, c0, r1, c1 = comp["bbox"]
        assert c0 == c1
        coords = comp["coords"]
        height = r1 - r0 + 1
        assert comp["area"] == height
        assert {(int(r), int(c)) for r, c in coords} == {(r, c0) for r in range(r0, r1 + 1)}
        if baseline is None:
            baseline = r1
        else:
            assert r1 == baseline
        bars.append(
            {
                "color": int(comp["color"]),
                "col": int(c0),
                "bottom": int(r1),
                "height": int(height),
                "coords": coords,
            }
        )

    assert bars
    return sorted(bars, key=lambda b: b["col"])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate each blue-started chain of bottom-aligned vertical bars into horizontal bars resting on the same baseline.
    Input structure
    - The grid has a background and bottom-aligned, one-cell-wide vertical bars.
    - Blue bars are markers; red bars are body bars that may be chained after a marker.
    Rule
    - Starting at each blue bar, repeatedly include the next red bar whose column is exactly the current bar's column plus the current bar's height.
    - Clear every included vertical bar, then redraw each included bar horizontally to the right from its original bottom cell, with length equal to its vertical height.
    - Red bars that are not reached from a blue marker remain unchanged.
    Key insight
    - A vertical bar's height is also the width it occupies after rotation, so the chain follows immediate horizontal contact after each rotated bar.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    colors = set(map(int, np.unique(g))) - {bg}
    assert colors <= {MARKER_COLOR, BODY_COLOR}
    assert MARKER_COLOR in colors

    bars = _vertical_bars(g, bg)
    by_col = {bar["col"]: bar for bar in bars}
    assert len(by_col) == len(bars)

    baselines = {bar["bottom"] for bar in bars}
    assert len(baselines) == 1
    baseline = baselines.pop()
    out = g.copy()
    for bar in bars:
        if bar["color"] != MARKER_COLOR:
            continue

        group = [bar]
        current = bar
        while True:
            next_col = current["col"] + current["height"]
            next_bar = by_col.get(next_col)
            if next_bar is None:
                break
            if next_bar["color"] != BODY_COLOR:
                break
            group.append(next_bar)
            current = next_bar

        for item in group:
            rows = item["coords"][:, 0]
            cols = item["coords"][:, 1]
            out[rows, cols] = bg

        for item in group:
            col = item["col"]
            height = item["height"]
            assert col + height <= g.shape[1]
            out[baseline, col : col + height] = item["color"]

    return out

# --------------- generator (att2) ---------------

# Canvas bounds.
MIN_ROWS, MAX_ROWS, MIN_WIDTH, MAX_WIDTH = 3, 9, 7, 30

# Bar and chain structure.
MIN_BLUE_BARS, MAX_BLUE_BARS = 1, 3
MIN_RED_BARS, MAX_RED_BARS, MAX_TOTAL_BARS = 1, 6, 9
MIN_BAR_HEIGHT, MAX_BAR_HEIGHT, MAX_CHAIN_RED_PER_BLUE, MAX_UNCHAINED_RED = 2, 4, 4, 2

# Layout padding and retry budget.
GAP_VALUES = (1, 2, 3)
MAX_RIGHT_MARGIN, MAX_RANDOM_LEFT_MARGIN, MAX_GENERATION_ATTEMPTS = 2, 2, 300


def _chain_red_distributions(n_blue, n_red):
    """Return all red-count allocations across blue-started chains."""
    return [c for c in itertools.product(range(MAX_CHAIN_RED_PER_BLUE + 1), repeat=n_blue) if sum(c) == n_red]


def _bounded_tuples(values, n, max_sum):
    """Return ordered n-tuples from values whose sum fits the width budget."""
    return [t for t in itertools.product(values, repeat=n) if sum(t) <= max_sum]


def _sample_groups():
    """Sample row count and bar groups before physical layout."""
    rows = random.randint(MIN_ROWS, MAX_ROWS)
    max_height = min(MAX_BAR_HEIGHT, rows)
    n_blue = random.randint(MIN_BLUE_BARS, MAX_BLUE_BARS)
    max_red = min(MAX_RED_BARS, MAX_TOTAL_BARS - n_blue, MAX_CHAIN_RED_PER_BLUE * n_blue + MAX_UNCHAINED_RED)
    red_total = random.randint(MIN_RED_BARS, max_red)
    min_unchained = max(0, red_total - MAX_CHAIN_RED_PER_BLUE * n_blue)
    max_unchained = min(MAX_UNCHAINED_RED, red_total)
    n_unchained = random.randint(min_unchained, max_unchained)

    distributions = _chain_red_distributions(n_blue, red_total - n_unchained)
    if not distributions:
        return None

    groups = [
        {"kind": "chain", "colors": [MARKER_COLOR] + [BODY_COLOR] * n_red}
        for n_red in random.choice(distributions)
    ]
    groups.extend({"kind": "unchained", "colors": [BODY_COLOR]} for _ in range(n_unchained))
    random.shuffle(groups)
    if groups[0]["kind"] == "unchained":
        for i, group in enumerate(groups):
            if group["kind"] == "chain":
                groups[0], groups[i] = groups[i], groups[0]
                break
    return rows, max_height, groups


def _sample_layout(max_height, groups):
    """Sample gaps, heights, margins, and positioned bar columns."""
    n_bars = sum(len(group["colors"]) for group in groups)
    right_margin = random.randint(0, MAX_RIGHT_MARGIN)

    gap_count = len(groups) - 1
    if gap_count:
        max_gap_sum = MAX_WIDTH - right_margin - MIN_BAR_HEIGHT * n_bars
        gap_options = _bounded_tuples(GAP_VALUES, gap_count, max_gap_sum)
        if not gap_options:
            return None
        gaps = list(random.choice(gap_options))
    else:
        gaps = []

    height_options = _bounded_tuples(
        tuple(range(MIN_BAR_HEIGHT, max_height + 1)),
        n_bars,
        MAX_WIDTH - right_margin - sum(gaps),
    )
    if not height_options:
        return None
    heights = list(random.choice(height_options))

    idx = 0
    for group in groups:
        count = len(group["colors"])
        group["heights"] = heights[idx : idx + count]
        idx += count

    span = sum(heights) + sum(gaps)
    min_left = max(0, MIN_WIDTH - right_margin - span)
    max_left = MAX_WIDTH - right_margin - span
    if min_left > max_left:
        return None
    if min_left <= min(MAX_RANDOM_LEFT_MARGIN, max_left):
        left_margin = random.randint(min_left, min(MAX_RANDOM_LEFT_MARGIN, max_left))
    else:
        left_margin = min_left

    width = left_margin + span + right_margin
    if not (MIN_WIDTH <= width <= MAX_WIDTH):
        return None

    bars = []
    col = left_margin
    gap_i = 0
    for group_i, group in enumerate(groups):
        chained = group["kind"] == "chain"
        for color, height in zip(group["colors"], group["heights"]):
            bars.append(
                {
                    "color": color,
                    "height": height,
                    "col": col,
                    "chained": chained,
                }
            )
            col += height
        if group_i + 1 < len(groups):
            col += gaps[gap_i]
            gap_i += 1

    if col + right_margin != width:
        return None
    return bars, width


def _render_pair(rows, width, bars):
    """Render the sampled input grid and its expected transformed output."""
    baseline = rows - 1
    inp = np.zeros((rows, width), dtype=int)
    out = np.zeros_like(inp)
    for bar in bars:
        color, height, col = bar["color"], bar["height"], bar["col"]
        vertical = slice(baseline - height + 1, baseline + 1)
        inp[vertical, col] = color
        if bar["chained"]:
            out[baseline, col : col + height] = color
        else:
            out[vertical, col] = color
    return inp, out


def _passes_oracle(inp, expected):
    """Return whether the solver accepts and matches the rendered output."""
    try:
        actual = main(inp)
    except Exception:
        return False
    return np.array_equal(actual, expected)


def _gen_attempt():
    sampled = _sample_groups()
    if sampled is None:
        return None
    rows, max_height, groups = sampled

    layout = _sample_layout(max_height, groups)
    if layout is None:
        return None
    bars, width = layout

    inp, expected = _render_pair(rows, width, bars)
    if not _passes_oracle(inp, expected):
        return None
    return inp


def generate_input():
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 84551f4c input")
