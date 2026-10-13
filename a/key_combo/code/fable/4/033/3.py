"""ARC task: continue motif rows in the order encoded by ranked side bars."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


BACKGROUND = 0
GREY = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
_GENERATION_INDEX = 0


@dataclass(frozen=True)
class _Layout:
    separator_col: int
    background: int
    motif_count: int
    continuation_count: int
    suffix_start: int
    motif_colors: tuple[int, ...]
    ranked_colors: tuple[int, ...]
    templates: tuple[np.ndarray, ...]
    heights: tuple[int, ...]


def _analyze(grid: np.ndarray) -> _Layout:
    """Read all visible controls without assigning meaning to color ids."""
    g = np.asarray(grid)
    assert g.ndim == 2, "expected a 2D grid"
    height, width = g.shape
    assert 1 <= height <= 30 and 1 <= width <= 30

    separator_cols = np.flatnonzero(np.all(g == GREY, axis=0))
    assert len(separator_cols) == 1, "expected one full-height grey separator"
    separator = int(separator_cols[0])
    assert 4 <= separator < width - 1, "separator must leave two panels"

    bottom_main_colors = np.unique(g[-1, :separator])
    assert len(bottom_main_colors) == 1, "the main-panel suffix must be blank"
    background = int(bottom_main_colors[0])
    assert background != GREY

    bracket_mask = g[:, 0] == GREY
    motif_count = 0
    while motif_count < height and bool(bracket_mask[motif_count]):
        motif_count += 1
    assert motif_count >= 3, "the bracket must mark at least three motif rows"
    assert not np.any(bracket_mask[motif_count:]), "the bracket must be one top run"

    templates: list[np.ndarray] = []
    motif_colors: list[int] = []
    for row_index in range(motif_count):
        row = g[row_index, :separator]
        assert int(row[0]) == GREY
        assert not np.any(row[1:] == GREY), "grey is reserved for structural markers"
        visible = set(map(int, np.unique(row))) - {background, GREY}
        assert len(visible) == 1, "each motif row must use one independent color"
        color = visible.pop()
        template = row.copy()
        template[template == GREY] = background
        assert np.count_nonzero(template != background) >= 2
        motif_colors.append(color)
        templates.append(template)
    assert len(set(motif_colors)) == motif_count, "motif colors must be unique"

    suffix_start = motif_count
    while suffix_start < height and np.any(g[suffix_start, :separator] != background):
        suffix_start += 1
    continuation_count = suffix_start - motif_count
    assert 1 <= continuation_count <= 2
    assert continuation_count < motif_count
    assert suffix_start < height, "expected a nonempty blank suffix"
    assert np.all(g[suffix_start:, :separator] == background)

    side = g[:, separator + 1 :]
    allowed = {background, *motif_colors}
    assert set(map(int, np.unique(side))) <= allowed

    heights_by_color: dict[int, int] = {}
    occupied_columns: set[int] = set()
    for color in motif_colors:
        color_columns = np.flatnonzero(np.any(side == color, axis=0))
        assert len(color_columns) == 1, "each color must form one one-cell-wide bar"
        column = int(color_columns[0])
        assert column not in occupied_columns
        occupied_columns.add(column)
        bar_mask = side[:, column] == color
        bar_height = int(np.count_nonzero(bar_mask))
        assert bar_height > 0
        assert np.all(bar_mask[height - bar_height :])
        assert not np.any(bar_mask[: height - bar_height])
        assert np.all(side[:, column][~bar_mask] == background)
        heights_by_color[color] = bar_height
    assert len(occupied_columns) == motif_count
    assert np.count_nonzero(side != background) == sum(heights_by_color.values())
    assert len(set(heights_by_color.values())) == motif_count, "bar heights must be unique"

    ranked_colors = tuple(
        sorted(motif_colors, key=lambda color: heights_by_color[color], reverse=True)
    )
    by_color = {color: template for color, template in zip(motif_colors, templates)}
    for phase in range(continuation_count):
        actual = g[motif_count + phase, :separator]
        expected = by_color[ranked_colors[phase]]
        assert np.array_equal(actual, expected), "continuation rows must open the ranked cycle"

    return _Layout(
        separator_col=separator,
        background=background,
        motif_count=motif_count,
        continuation_count=continuation_count,
        suffix_start=suffix_start,
        motif_colors=tuple(motif_colors),
        ranked_colors=ranked_colors,
        templates=tuple(templates),
        heights=tuple(heights_by_color[color] for color in motif_colors),
    )


def main(grid: np.ndarray) -> np.ndarray:
    """Continue the bracketed motif rows cyclically using the side bars' height ranking.

    Input structure:
    A full-height grey separator divides a wide main panel from a narrow side
    panel.  A continuous grey run at the upper-left brackets k one-color motif
    rows.  Beneath them are one or two already drawn continuation rows and then
    a blank suffix.  The side panel contains k unique-color, bottom-aligned,
    one-cell-wide bars whose colors match the motif rows and whose heights differ.

    Rule:
    Rank motif colors by their matching side-bar heights from tallest to
    shortest.  Treat each grey bracket cell as background, and use the complete
    corresponding motif row as an item in that ranked cycle.  The drawn rows are
    the first one or two cycle items; resume after their phase and copy whole
    motif rows until the blank main-panel suffix is full.  Preserve the prefix,
    separator, and side panel.

    Key insight:
    Motif placement order is only a legend: bar height supplies playback order,
    while the short drawn prefix supplies the phase at which filling must resume.
    """
    g = np.asarray(grid)
    layout = _analyze(g)
    out = g.copy()
    template_by_color = {
        color: template
        for color, template in zip(layout.motif_colors, layout.templates)
    }
    for offset, row in enumerate(range(layout.suffix_start, g.shape[0])):
        phase = (layout.continuation_count + offset) % layout.motif_count
        out[row, : layout.separator_col] = template_by_color[layout.ranked_colors[phase]]
    return out


def _random_pattern(
    rng: np.random.Generator,
    main_width: int,
    color: int,
    used_masks: set[tuple[bool, ...]],
) -> np.ndarray:
    """Create a sparse but legible horizontal motif with an adjacent pair."""
    candidates = np.arange(2, main_width - 1)
    for _ in range(100):
        density = float(rng.uniform(0.32, 0.68))
        mask = rng.random(len(candidates)) < density
        count = int(mask.sum())
        key = tuple(map(bool, mask))
        if not (3 <= count <= len(candidates) - 2):
            continue
        if not np.any(mask[:-1] & mask[1:]):
            continue
        if key in used_masks:
            continue
        used_masks.add(key)
        row = np.full(main_width, BACKGROUND, dtype=int)
        row[candidates[mask]] = int(color)
        return row
    raise RuntimeError("failed to sample a distinct motif pattern")


def _independent_expected(
    grid: np.ndarray,
    separator: int,
    motif_count: int,
    continuation_count: int,
    motif_colors: list[int],
    ranked_colors: list[int],
    templates: list[np.ndarray],
) -> np.ndarray:
    """Construct the intended result directly from generator-known semantics."""
    expected = grid.copy()
    by_color = {color: row for color, row in zip(motif_colors, templates)}
    start = motif_count + continuation_count
    for offset, row_index in enumerate(range(start, grid.shape[0])):
        color = ranked_colors[(continuation_count + offset) % motif_count]
        expected[row_index, :separator] = by_color[color]
    return expected


def _valid_generated(
    grid: np.ndarray,
    motif_count: int,
    continuation_count: int,
    separator: int,
    motif_colors: list[int],
    ranked_colors: list[int],
    heights_by_color: dict[int, int],
    templates: list[np.ndarray],
) -> bool:
    """Check uniqueness, all semantic dependencies, and non-degeneracy."""
    try:
        layout = _analyze(grid)
        if layout.motif_count != motif_count:
            return False
        if layout.continuation_count != continuation_count:
            return False
        if list(layout.motif_colors) != motif_colors:
            return False
        if list(layout.ranked_colors) != ranked_colors:
            return False
        if list(layout.heights) != [heights_by_color[c] for c in motif_colors]:
            return False

        expected = _independent_expected(
            grid,
            separator,
            motif_count,
            continuation_count,
            motif_colors,
            ranked_colors,
            templates,
        )
        actual = main(grid)
        if not np.array_equal(actual, expected) or np.array_equal(actual, grid):
            return False

        # Both hidden controls are indispensable: replacing height order with
        # display order, or restarting at phase zero, must change the answer.
        display_order_answer = _independent_expected(
            grid,
            separator,
            motif_count,
            continuation_count,
            motif_colors,
            motif_colors,
            templates,
        )
        phase_zero_answer = grid.copy()
        by_color = {c: row for c, row in zip(motif_colors, templates)}
        start = motif_count + continuation_count
        for offset, row_index in enumerate(range(start, grid.shape[0])):
            phase_zero_answer[row_index, :separator] = by_color[
                ranked_colors[offset % motif_count]
            ]
        if np.array_equal(expected, display_order_answer):
            return False
        if np.array_equal(expected, phase_zero_answer):
            return False

        # A complete suffix cycle exposes every color and every row pattern.
        suffix = actual[start:, :separator]
        if any(not np.any(suffix == color) for color in motif_colors):
            return False
        if np.any(actual[:start] != grid[:start]):
            return False
        if np.any(actual[:, separator:] != grid[:, separator:]):
            return False
        return True
    except (AssertionError, RuntimeError, ValueError, IndexError):
        return False


def _generate_attempt(
    rng: np.random.Generator,
    motif_count: int,
    continuation_count: int,
) -> np.ndarray | None:
    suffix_length = int(rng.integers(motif_count + 2, 2 * motif_count + 7))
    height = motif_count + continuation_count + suffix_length
    main_width = int(rng.integers(13, 19))
    side_width = 2 * motif_count + 1
    width = main_width + 1 + side_width
    if height > 30 or width > 30:
        return None

    motif_colors = [
        int(color)
        for color in rng.choice(np.array(PALETTE), size=motif_count, replace=False)
    ]
    used_masks: set[tuple[bool, ...]] = set()
    templates = [
        _random_pattern(rng, main_width, color, used_masks) for color in motif_colors
    ]

    ranked_indices = list(map(int, rng.permutation(motif_count)))
    if ranked_indices == list(range(motif_count)):
        ranked_indices = ranked_indices[1:] + ranked_indices[:1]
    ranked_colors = [motif_colors[index] for index in ranked_indices]

    possible_heights = np.arange(2, height)
    sampled_heights = sorted(
        map(int, rng.choice(possible_heights, size=motif_count, replace=False)),
        reverse=True,
    )
    heights_by_color = {
        color: bar_height for color, bar_height in zip(ranked_colors, sampled_heights)
    }

    grid = np.full((height, width), BACKGROUND, dtype=int)
    grid[:, main_width] = GREY
    for row_index, template in enumerate(templates):
        grid[row_index, :main_width] = template
        grid[row_index, 0] = GREY

    template_by_color = {color: row for color, row in zip(motif_colors, templates)}
    for phase in range(continuation_count):
        grid[motif_count + phase, :main_width] = template_by_color[
            ranked_colors[phase]
        ]

    shuffled_columns = list(map(int, rng.permutation(motif_count)))
    for color, slot in zip(motif_colors, shuffled_columns):
        side_col = main_width + 1 + (2 * slot + 1)
        bar_height = heights_by_color[color]
        grid[height - bar_height :, side_col] = color

    if _valid_generated(
        grid,
        motif_count,
        continuation_count,
        main_width,
        motif_colors,
        ranked_colors,
        heights_by_color,
        templates,
    ):
        return grid
    return None


def generate_input() -> np.ndarray:
    """Generate a varied, oracle-checked instance of the ranked motif cycle."""
    global _GENERATION_INDEX
    rng = np.random.default_rng()
    # A three-example training batch always covers k=3,4,5 and both phase
    # lengths.  All geometry and visible encodings inside each branch stay random.
    motif_count = 3 + (_GENERATION_INDEX % 3)
    continuation_count = 1 + (_GENERATION_INDEX % 2)
    for _ in range(200):
        grid = _generate_attempt(rng, motif_count, continuation_count)
        if grid is not None:
            _GENERATION_INDEX += 1
            return grid
    raise RuntimeError("failed to generate a valid ranked-motif input")
