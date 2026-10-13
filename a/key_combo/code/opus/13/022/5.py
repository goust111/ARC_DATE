import random

import numpy as np


SEPARATOR_COLOR = 5
PANEL_SIZE = 9
SUMMARY_SIZE = 3
BLOCK_SIZE = 3
VALID_SHIFTS = [
    (dr, dc)
    for dr in (-1, 0, 1)
    for dc in (-1, 0, 1)
]
GEN_SHIFTS = [shift for shift in VALID_SHIFTS if shift != (0, 0)]
GEN_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
GEN_WIDTHS = [9, 10]
GEN_MAX_ATTEMPTS = 400


def _split_panels(grid: np.ndarray) -> list[np.ndarray]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert g.shape[0] == PANEL_SIZE * 3 + 2, "expected three 9-row panels and two separator rows"

    sep_rows = [r for r in range(g.shape[0]) if np.all(g[r] == SEPARATOR_COLOR)]
    assert len(sep_rows) == 2, "expected exactly two full-width separator rows"
    assert sep_rows[0] + 1 < sep_rows[1], "separator rows must be distinct"

    starts = [0, sep_rows[0] + 1, sep_rows[1] + 1]
    ends = [sep_rows[0], sep_rows[1], g.shape[0]]
    panels = [g[r0:r1].copy() for r0, r1 in zip(starts, ends)]
    assert all(panel.shape[0] == PANEL_SIZE for panel in panels), "every panel must be 9 rows tall"
    return panels


def _is_valid_panel_content(panel: np.ndarray) -> bool:
    if panel.shape != (PANEL_SIZE, PANEL_SIZE):
        return False
    for block_r in range(SUMMARY_SIZE):
        for block_c in range(SUMMARY_SIZE):
            block = panel[
                block_r * BLOCK_SIZE : (block_r + 1) * BLOCK_SIZE,
                block_c * BLOCK_SIZE : (block_c + 1) * BLOCK_SIZE,
            ]
            if np.count_nonzero(block) != 1:
                return False
    return True


def _extract_contents(panels: list[np.ndarray]) -> list[np.ndarray]:
    width = panels[0].shape[1]
    assert all(panel.shape[1] == width for panel in panels), "all panels must have the same width"
    assert PANEL_SIZE <= width <= PANEL_SIZE + 1, "expected panel width 9 or 10"

    candidates = []
    for c0 in range(width - PANEL_SIZE + 1):
        extracted = []
        ok = True
        for panel in panels:
            content = panel[:, c0 : c0 + PANEL_SIZE]
            outside = np.hstack([panel[:, :c0], panel[:, c0 + PANEL_SIZE :]]) if width > PANEL_SIZE else np.zeros((PANEL_SIZE, 0), dtype=int)
            if outside.size and np.any(outside != 0):
                ok = False
                break
            if not _is_valid_panel_content(content):
                ok = False
                break
            extracted.append(content)
        if ok:
            candidates.append(extracted)

    assert len(candidates) == 1, "expected a unique 9x9 panel content area"
    return candidates[0]


def _compress_panel(panel: np.ndarray) -> np.ndarray:
    summary = np.zeros((SUMMARY_SIZE, SUMMARY_SIZE), dtype=int)
    for block_r in range(SUMMARY_SIZE):
        for block_c in range(SUMMARY_SIZE):
            block = panel[
                block_r * BLOCK_SIZE : (block_r + 1) * BLOCK_SIZE,
                block_c * BLOCK_SIZE : (block_c + 1) * BLOCK_SIZE,
            ]
            coords = np.argwhere(block != 0)
            assert coords.shape[0] == 1, "each 3x3 block must contain exactly one foreground pixel"
            r, c = map(int, coords[0])
            summary[block_r, block_c] = int(block[r, c])
    return summary


def _roll_summary(summary: np.ndarray, shift: tuple[int, int]) -> np.ndarray:
    dr, dc = shift
    return np.roll(summary, shift=(int(dr), int(dc)), axis=(0, 1))


def _infer_shift(summaries: list[np.ndarray]) -> tuple[int, int]:
    matches = []
    for shift in VALID_SHIFTS:
        if np.array_equal(_roll_summary(summaries[0], shift), summaries[1]) and np.array_equal(
            _roll_summary(summaries[1], shift), summaries[2]
        ):
            matches.append(shift)
    assert len(matches) == 1, "expected a unique constant wrapped translation on the compressed summaries"
    return matches[0]


def _expand_summary(summary: np.ndarray) -> np.ndarray:
    out = np.zeros((PANEL_SIZE, PANEL_SIZE), dtype=int)
    for r in range(SUMMARY_SIZE):
        for c in range(SUMMARY_SIZE):
            out[r * BLOCK_SIZE + 1, c * BLOCK_SIZE + 1] = int(summary[r, c])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extrapolate the next panel by compressing each observed panel to a 3x3 summary, continuing the summary motion, and expanding it back.

    Input structure:
    The input is a vertical stack of three 9-row panels separated by two full-width grey rows. Each panel contains a 9x9 content area, possibly with one extra black margin column, and every 3x3 block inside that content area has exactly one non-black pixel.

    Rule:
    Split the strip into three panels, locate the unique 9x9 content area, and compress each panel by replacing every 3x3 block with the color of its lone foreground pixel. The resulting three 3x3 summaries follow one constant wrapped translation `(dr, dc)` from frame to frame; apply the same translation once more to the third summary, then expand the predicted 3x3 summary back to 9x9 by placing each color at the center of its corresponding 3x3 block and leaving all other cells black.

    Key insight:
    The frame-to-frame motion is hidden by sparse per-block pixel positions in the 9x9 panels and becomes obvious only after compressing each 3x3 block to a single summary cell.
    """
    panels = _split_panels(grid)
    contents = _extract_contents(panels)
    summaries = [_compress_panel(panel) for panel in contents]
    shift = _infer_shift(summaries)
    next_summary = _roll_summary(summaries[-1], shift)
    return _expand_summary(next_summary)


def _sample_base_summary() -> np.ndarray:
    while True:
        distinct_count = random.randint(4, len(GEN_COLORS))
        palette = random.sample(GEN_COLORS, distinct_count)
        values = [random.choice(palette) for _ in range(SUMMARY_SIZE * SUMMARY_SIZE)]
        if len(set(values)) < 4:
            continue
        summary = np.array(values, dtype=int).reshape(SUMMARY_SIZE, SUMMARY_SIZE)
        return summary


def _unique_shift_for_sequence(base: np.ndarray, shift: tuple[int, int]) -> bool:
    summaries = [base]
    for _ in range(2):
        summaries.append(_roll_summary(summaries[-1], shift))
    matches = [
        candidate
        for candidate in VALID_SHIFTS
        if np.array_equal(_roll_summary(summaries[0], candidate), summaries[1])
        and np.array_equal(_roll_summary(summaries[1], candidate), summaries[2])
    ]
    return matches == [shift]


def _render_panel(summary: np.ndarray) -> np.ndarray:
    panel = np.zeros((PANEL_SIZE, PANEL_SIZE), dtype=int)
    for block_r in range(SUMMARY_SIZE):
        for block_c in range(SUMMARY_SIZE):
            inner_r = random.randrange(BLOCK_SIZE)
            inner_c = random.randrange(BLOCK_SIZE)
            panel[block_r * BLOCK_SIZE + inner_r, block_c * BLOCK_SIZE + inner_c] = int(summary[block_r, block_c])
    return panel


def _embed_panel(content: np.ndarray, width: int, offset: int) -> np.ndarray:
    panel = np.zeros((PANEL_SIZE, width), dtype=int)
    panel[:, offset : offset + PANEL_SIZE] = content
    return panel


def _generate_case() -> tuple[np.ndarray, np.ndarray] | None:
    width = random.choice(GEN_WIDTHS)
    offset = 0 if width == PANEL_SIZE else random.choice([0, 1])
    shift = random.choice(GEN_SHIFTS)

    base = _sample_base_summary()
    if not _unique_shift_for_sequence(base, shift):
        return None

    summaries = [base]
    for _ in range(2):
        summaries.append(_roll_summary(summaries[-1], shift))
    predicted = _roll_summary(summaries[-1], shift)

    panels = [_embed_panel(_render_panel(summary), width, offset) for summary in summaries]
    separator = np.full((1, width), SEPARATOR_COLOR, dtype=int)
    inp = np.vstack([panels[0], separator, panels[1], separator, panels[2]])
    expected = _expand_summary(predicted)

    try:
        actual = main(inp)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp, expected


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        case = _generate_case()
        if case is not None:
            inp, _expected = case
            return inp
    raise RuntimeError("failed to generate a valid input for puzzle 5")
