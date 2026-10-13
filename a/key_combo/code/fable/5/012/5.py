from __future__ import annotations

import random

import numpy as np


BLACK = 0
GRAY = 5
PALETTE = tuple(color for color in range(1, 10) if color != GRAY)
MAX_GENERATION_ATTEMPTS = 300


def _panel_layout(g: np.ndarray) -> tuple[int, tuple[int, int, int]]:
    """Return panel width and starts after validating the two separators."""
    h, w = g.shape
    assert (w - 2) % 3 == 0, "expected three equal panels and two separator columns"
    panel_w = (w - 2) // 3
    assert panel_w >= 2
    separator_cols = (panel_w, 2 * panel_w + 1)
    separator_color = int(g[0, separator_cols[0]])
    assert separator_color != BLACK
    assert np.all(g[:, separator_cols[0]] == separator_color)
    assert np.all(g[:, separator_cols[1]] == separator_color)

    starts = (0, panel_w + 1, 2 * panel_w + 2)
    for start in starts:
        panel = g[:, start : start + panel_w]
        assert not np.any(panel == separator_color), "separator color must occur only in separators"
    return panel_w, starts


def _extract_frames(g: np.ndarray) -> dict[str, object]:
    """Parse bars and prophecy dots using only panel-local (row, column) coordinates."""
    panel_w, starts = _panel_layout(g)
    panels = [g[:, start : start + panel_w] for start in starts]
    color_sets = [set(map(int, np.unique(panel))) - {BLACK} for panel in panels]
    assert all(len(colors) == 2 for colors in color_sets)
    assert color_sets[0] == color_sets[1] == color_sets[2]
    colors = color_sets[0]

    dot_candidates = [
        color
        for color in colors
        if all(int(np.count_nonzero(panel == color)) == 1 for panel in panels)
    ]
    bar_candidates: list[int] = []
    for color in colors:
        valid = True
        for panel in panels:
            coords = np.argwhere(panel == color)
            if len(coords) < 2 or len(set(map(int, coords[:, 1]))) != 1:
                valid = False
                break
            rows = np.sort(coords[:, 0])
            if not np.array_equal(rows, np.arange(int(rows[0]), int(rows[-1]) + 1)):
                valid = False
                break
        if valid:
            bar_candidates.append(color)

    assert len(dot_candidates) == 1 and len(bar_candidates) == 1
    dot_color = int(dot_candidates[0])
    bar_color = int(bar_candidates[0])
    assert dot_color != bar_color

    frames: list[dict[str, int]] = []
    for panel in panels:
        bar = np.argwhere(panel == bar_color)
        dot = np.argwhere(panel == dot_color)
        top = int(bar[:, 0].min())
        bottom = int(bar[:, 0].max())
        column = int(bar[0, 1])
        dot_r, dot_c = map(int, dot[0])
        assert dot_c == column, "each prophecy dot must share its bar's panel-local column"
        assert dot_r < top, "each prophecy dot must hang strictly above its current bar"
        frames.append(
            {
                "top": top,
                "bottom": bottom,
                "length": bottom - top + 1,
                "column": column,
                "dot_row": dot_r,
            }
        )

    return {
        "panel_width": panel_w,
        "frames": frames,
        "bar_color": bar_color,
        "dot_color": dot_color,
    }


def _analyze(g: np.ndarray) -> dict[str, int]:
    """Validate the complete time-line and return the uniquely determined future facts."""
    parsed = _extract_frames(g)
    frames = parsed["frames"]
    assert isinstance(frames, list)

    bottoms = [int(frame["bottom"]) for frame in frames]
    assert bottoms[0] == bottoms[1] == bottoms[2], "bar bottoms must share one row"
    bottom = bottoms[0]

    lengths = [int(frame["length"]) for frame in frames]
    delta = lengths[1] - lengths[0]
    assert delta > 0
    assert lengths[2] - lengths[1] == delta + 1, "bar growth increments must rise by one"

    columns = [int(frame["column"]) for frame in frames]
    step = columns[1] - columns[0]
    assert step > 0
    assert columns[2] - columns[1] == step, "bars must translate by one fixed panel-local step"
    column4 = columns[2] + step
    assert column4 < int(parsed["panel_width"]), "fourth bar must remain inside its panel"

    length4 = lengths[2] + delta + 2
    length5 = length4 + delta + 3
    length6 = length5 + delta + 4
    predicted_tops = [
        bottom - lengths[2] + 1,
        bottom - length4 + 1,
        bottom - length5 + 1,
    ]
    dot_rows = [int(frame["dot_row"]) for frame in frames]
    assert dot_rows == predicted_tops, "dots must name the bar tops exactly two frames ahead"

    # Use the second prophecy as the fourth-frame top.  Continue from the third
    # prophecy by the next growth increment to obtain the sixth-frame top.
    top4 = dot_rows[1]
    top6 = dot_rows[2] - (delta + 4)
    assert top6 == bottom - length6 + 1

    return {
        "panel_width": int(parsed["panel_width"]),
        "bottom": bottom,
        "delta": delta,
        "step": step,
        "column4": column4,
        "top4": top4,
        "top6": top6,
        "bar_color": int(parsed["bar_color"]),
        "dot_color": int(parsed["dot_color"]),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extrapolate three separated bar-and-prophecy panels into the fourth frame.

    Input structure:
    - Black is the panel background, and two full-height uniform columns divide the
      grid into three equal panels.
    - Each panel contains one same-colored vertical bar with a common bottom row and
      one differently colored singleton dot above it in the same panel-local column.

    Rule:
    - Read the panels as consecutive frames.  Bar lengths grow first by delta and
      then by delta+1, while their panel-local columns move right by a fixed step.
    - A dot in frame i is exactly the top row of the bar in frame i+2.  Draw frame 4
      using growth delta+2 and one more translation, then place its dot at frame 6's
      top after the further growth increments delta+3 and delta+4.  Clip at the top.

    Key insight:
    - The isolated dots are cross-frame prophecies, not fixed-distance ornaments:
      together with second-order bar growth they extend the same timeline to frame 6.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and 1 <= g.shape[0] <= 30 and 1 <= g.shape[1] <= 30
    facts = _analyze(g)

    out = np.full((g.shape[0], facts["panel_width"]), BLACK, dtype=int)
    top4 = max(0, facts["top4"])
    out[top4 : facts["bottom"] + 1, facts["column4"]] = facts["bar_color"]
    if 0 <= facts["top6"] < out.shape[0]:
        out[facts["top6"], facts["column4"]] = facts["dot_color"]
    return out


def _render_input(
    height: int,
    panel_w: int,
    bottom: int,
    length1: int,
    delta: int,
    step: int,
    column1: int,
    bar_color: int,
    dot_color: int,
) -> np.ndarray:
    """Construct the first three frames directly from the stated semantics."""
    grid = np.full((height, 3 * panel_w + 2), BLACK, dtype=int)
    grid[:, panel_w] = GRAY
    grid[:, 2 * panel_w + 1] = GRAY
    starts = (0, panel_w + 1, 2 * panel_w + 2)

    lengths = (length1, length1 + delta, length1 + 2 * delta + 1)
    length4 = lengths[2] + delta + 2
    length5 = length4 + delta + 3
    future_tops = (
        bottom - lengths[2] + 1,
        bottom - length4 + 1,
        bottom - length5 + 1,
    )
    for i, start in enumerate(starts):
        column = column1 + i * step
        top = bottom - lengths[i] + 1
        grid[top : bottom + 1, start + column] = bar_color
        grid[future_tops[i], start + column] = dot_color
    return grid


def _expected_output(
    height: int,
    panel_w: int,
    bottom: int,
    length1: int,
    delta: int,
    step: int,
    column1: int,
    bar_color: int,
    dot_color: int,
) -> np.ndarray:
    """Independent generator oracle for the fourth bar and its sixth-frame dot."""
    out = np.full((height, panel_w), BLACK, dtype=int)
    fourth_length = length1 + 3 * delta + 3
    sixth_length = length1 + 5 * delta + 10
    fourth_top = bottom - fourth_length + 1
    sixth_top = bottom - sixth_length + 1
    fourth_column = column1 + 3 * step
    out[max(0, fourth_top) : bottom + 1, fourth_column] = bar_color
    if 0 <= sixth_top < height:
        out[sixth_top, fourth_column] = dot_color
    return out


def _solver_rejects(grid: np.ndarray) -> bool:
    try:
        main(grid)
    except (AssertionError, IndexError, ValueError):
        return True
    return False


def _semantic_checks(
    inp: np.ndarray,
    expected: np.ndarray,
    *,
    panel_w: int,
    bottom: int,
    length1: int,
    delta: int,
    step: int,
    column1: int,
    clipped_dot: bool,
) -> bool:
    """Check semantic dependencies, edge behavior, and tempting shortcut counterexamples."""
    try:
        facts = _analyze(inp)
    except (AssertionError, IndexError, ValueError):
        return False

    fourth_length = length1 + 3 * delta + 3
    sixth_length = length1 + 5 * delta + 10
    if facts["top4"] != bottom - fourth_length + 1:
        return False
    if facts["top6"] != bottom - sixth_length + 1:
        return False
    if facts["column4"] != column1 + 3 * step:
        return False
    if clipped_dot != (facts["top6"] < 0):
        return False

    parsed = _extract_frames(inp)
    frames = parsed["frames"]
    gaps = [int(frame["top"]) - int(frame["dot_row"]) for frame in frames]
    if not (gaps[1] - gaps[0] == 2 and gaps[2] - gaps[1] == 2):
        return False

    # Repeating the last observed growth increment gives a different fourth bar.
    wrong = expected.copy()
    wrong[:, facts["column4"]] = BLACK
    wrong_top = bottom - ((length1 + 2 * delta + 1) + (delta + 1)) + 1
    wrong[max(0, wrong_top) : bottom + 1, facts["column4"]] = facts["bar_color"]
    if np.array_equal(wrong, expected):
        return False

    # A constant dot-to-dot step either misplaces the new dot or fails to clip it.
    dot_rows = [int(frame["dot_row"]) for frame in frames]
    wrong_dot_row = dot_rows[2] + (dot_rows[2] - dot_rows[1])
    if wrong_dot_row == facts["top6"]:
        return False
    if clipped_dot and not (facts["top6"] == -1 and wrong_dot_row == 0):
        return False

    # Each visible clue family is enforced: corrupting a prophecy, a growth term,
    # or the frame-3 local translation must make the solver reject the input.
    starts = (0, panel_w + 1, 2 * panel_w + 2)
    dot_color = facts["dot_color"]
    bar_color = facts["bar_color"]

    prophecy_mutant = inp.copy()
    p1 = int(frames[0]["dot_row"])
    c1 = int(frames[0]["column"])
    if p1 <= 0:
        return False
    prophecy_mutant[p1, starts[0] + c1] = BLACK
    prophecy_mutant[p1 - 1, starts[0] + c1] = dot_color

    growth_mutant = inp.copy()
    top2 = int(frames[1]["top"])
    c2 = int(frames[1]["column"])
    growth_mutant[top2 - 1, starts[1] + c2] = bar_color

    translation_mutant = inp.copy()
    old_c3 = int(frames[2]["column"])
    new_c3 = old_c3 + 1
    if new_c3 >= panel_w:
        return False
    panel3 = translation_mutant[:, starts[2] : starts[2] + panel_w]
    rows3 = np.flatnonzero((panel3[:, old_c3] == bar_color) | (panel3[:, old_c3] == dot_color))
    panel3[rows3, old_c3] = BLACK
    source = inp[:, starts[2] : starts[2] + panel_w]
    panel3[rows3, new_c3] = source[rows3, old_c3]

    return all(_solver_rejects(mutant) for mutant in (prophecy_mutant, growth_mutant, translation_mutant))


def _gen_attempt(clipped_dot: bool) -> tuple[np.ndarray, np.ndarray] | None:
    panel_w = random.randint(7, 9)
    step = random.randint(1, (panel_w - 1) // 3)
    column1 = random.randint(0, panel_w - 1 - 3 * step)
    delta = random.randint(1, 3)
    length1 = random.randint(2, 5)

    length5 = length1 + 4 * delta + 6
    length6 = length1 + 5 * delta + 10
    if length6 > 30:
        return None

    # In the clipped branch the sixth top is exactly -1.  This makes clipping
    # visible and defeats a constant-step shortcut, which would draw at row 0.
    capacity = length6 - 1 if clipped_dot else random.randint(length6, 30)
    if capacity < length5:
        return None
    bottom = capacity - 1
    height = random.randint(capacity, min(30, capacity + 3))
    bar_color, dot_color = random.sample(PALETTE, 2)

    inp = _render_input(
        height, panel_w, bottom, length1, delta, step, column1, bar_color, dot_color
    )
    expected = _expected_output(
        height, panel_w, bottom, length1, delta, step, column1, bar_color, dot_color
    )
    if max(inp.shape) > 30 or max(expected.shape) > 30:
        return None
    if not _semantic_checks(
        inp,
        expected,
        panel_w=panel_w,
        bottom=bottom,
        length1=length1,
        delta=delta,
        step=step,
        column1=column1,
        clipped_dot=clipped_dot,
    ):
        return None
    try:
        actual = main(inp)
    except (AssertionError, IndexError, ValueError):
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp, expected


_next_clipped_branch = bool(random.getrandbits(1))
_seen_output_signatures: set[tuple[tuple[int, int], bytes]] = set()


def generate_input() -> np.ndarray:
    """Generate varied valid inputs while alternating visible and clipped future dots."""
    global _next_clipped_branch
    for _ in range(MAX_GENERATION_ATTEMPTS):
        candidate = _gen_attempt(_next_clipped_branch)
        if candidate is None:
            continue
        inp, expected = candidate
        signature = (expected.shape, expected.tobytes())
        if signature in _seen_output_signatures:
            continue
        _seen_output_signatures.add(signature)
        _next_clipped_branch = not _next_clipped_branch
        return inp
    raise RuntimeError("failed to generate a valid prophecy-dot input")
