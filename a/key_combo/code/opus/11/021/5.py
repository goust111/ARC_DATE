from __future__ import annotations

import random

import numpy as np


BG = 0
PANEL_HEIGHT = 5
PANEL_COLORS = (1, 2)
SEPARATOR_COLORS = (3, 4, 5, 6, 7, 8, 9)
GEN_ATTEMPTS = 1000


def _nonzero_colors(grid: np.ndarray) -> list[int]:
    return [int(color) for color in np.unique(grid) if int(color) != BG]


def _separator_row_index(grid: np.ndarray) -> int:
    candidates = []
    for r in range(grid.shape[0]):
        row_colors = {int(v) for v in grid[r] if int(v) != BG}
        if any(color not in PANEL_COLORS for color in row_colors):
            candidates.append(r)
    assert len(candidates) == 1, "expected a unique separator row"
    return int(candidates[0])


def _separator_sequence(row: np.ndarray) -> list[int]:
    seq = [int(v) for v in row if int(v) != BG]
    assert len(seq) in (3, 4), "separator row should encode a 3-4 color sequence"
    assert len(seq) == len(set(seq)), "separator sequence colors should be distinct"
    return seq


def _overlay_top_priority(top: np.ndarray, bottom: np.ndarray) -> np.ndarray:
    out = bottom.copy()
    mask = top != BG
    out[mask] = top[mask]
    return out


def _overlay_bottom_priority(top: np.ndarray, bottom: np.ndarray) -> np.ndarray:
    out = top.copy()
    mask = bottom != BG
    out[mask] = bottom[mask]
    return out


def _panel_color_mapping(top: np.ndarray, bottom: np.ndarray, seq: list[int]) -> dict[int, int]:
    present = [color for color in PANEL_COLORS if np.any(top == color) or np.any(bottom == color)]
    assert present, "expected at least one panel color"
    assert len(seq) >= len(present), "separator sequence is too short for the visible panel colors"
    return {src: seq[i] for i, src in enumerate(present)}


def _recolor(grid: np.ndarray, mapping: dict[int, int]) -> np.ndarray:
    out = grid.copy()
    for src, dst in mapping.items():
        out[grid == src] = dst
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reverse each panel row, overlay the two panels, then recolor the merged pattern from the separator code.

    Input structure:
    - The grid has black background and an odd height with one separator row between two equal-height panels.
    - The top and bottom panels are 5-row bands whose foreground uses only blue and red pixels.
    - The separator row contains 3 or 4 distinct non-background colors laid out left-to-right on black.

    Rule:
    - Identify the unique separator row and split the grid into the top and bottom panels around it.
    - Reverse every row inside each panel independently by mirroring it left-to-right.
    - Overlay the reversed panels positionally: keep background if both are blank, keep the lone visible pixel if only one panel is non-background, and keep the top-panel pixel on conflicts.
    - Read the separator row's non-background colors from left to right. Because the panel palette is the fixed blue/red pair, map blue to the first separator color and red to the second whenever both are present, then recolor the overlaid result with that mapping.

    Key insight:
    - The middle row is both a physical divider and an instruction row: it first tells you where the two panels stop, and then its color order controls the final recoloring after the flip-and-overlay step.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    sep = _separator_row_index(g)
    top = g[:sep]
    bottom = g[sep + 1 :]
    assert top.shape == bottom.shape, "the two panels must have the same shape"
    assert top.shape[0] == PANEL_HEIGHT, "expected 5 rows per panel"

    top_rev = np.fliplr(top)
    bottom_rev = np.fliplr(bottom)
    merged = _overlay_top_priority(top_rev, bottom_rev)

    seq = _separator_sequence(g[sep])
    mapping = _panel_color_mapping(top, bottom, seq)
    return _recolor(merged, mapping)


def _random_panel(width: int) -> np.ndarray:
    panel = np.zeros((PANEL_HEIGHT, width), dtype=int)
    for r in range(PANEL_HEIGHT):
        count = random.randint(1, 3)
        cols = random.sample(range(width), count)
        for c in cols:
            panel[r, c] = random.choice(PANEL_COLORS)
    return panel


def _post_panels_valid(top_post: np.ndarray, bottom_post: np.ndarray) -> bool:
    if any(not (1 <= int(np.count_nonzero(top_post[r])) <= 3) for r in range(PANEL_HEIGHT)):
        return False
    if any(not (1 <= int(np.count_nonzero(bottom_post[r])) <= 3) for r in range(PANEL_HEIGHT)):
        return False

    used = set(_nonzero_colors(np.vstack([top_post, bottom_post])))
    if used != set(PANEL_COLORS):
        return False

    overlap = (top_post != BG) & (bottom_post != BG)
    if int(overlap.sum()) < 2:
        return False
    if not np.any(overlap & (top_post != bottom_post)):
        return False

    top_only = (top_post != BG) & (bottom_post == BG)
    bottom_only = (top_post == BG) & (bottom_post != BG)
    if not np.any(top_only) or not np.any(bottom_only):
        return False

    merged = _overlay_top_priority(top_post, bottom_post)
    if set(_nonzero_colors(merged)) != set(PANEL_COLORS):
        return False

    top_in = np.fliplr(top_post)
    bottom_in = np.fliplr(bottom_post)
    if np.array_equal(_overlay_top_priority(top_in, bottom_in), merged):
        return False

    return True


def _random_post_panels(width: int) -> tuple[np.ndarray, np.ndarray] | None:
    for _ in range(500):
        top_post = _random_panel(width)
        bottom_post = _random_panel(width)
        if _post_panels_valid(top_post, bottom_post):
            return top_post, bottom_post
    return None


def _random_separator_row(width: int) -> np.ndarray | None:
    seq_len = random.choice((3, 4))
    for _ in range(200):
        cols = sorted(random.sample(range(width), seq_len))
        if all(cols[i + 1] - cols[i] >= 2 for i in range(seq_len - 1)):
            break
    else:
        return None

    seq = random.sample(SEPARATOR_COLORS, seq_len)
    row = np.zeros(width, dtype=int)
    for c, color in zip(cols, seq):
        row[c] = color
    return row


def _validate_generated(grid: np.ndarray) -> None:
    out = main(grid)

    assert grid.shape[0] == 2 * PANEL_HEIGHT + 1
    assert out.shape == (PANEL_HEIGHT, grid.shape[1])
    assert max(grid.shape) <= 30 and max(out.shape) <= 30

    sep = _separator_row_index(grid)
    assert sep == PANEL_HEIGHT, "separator row should sit in the middle"
    top_in = grid[:sep]
    bottom_in = grid[sep + 1 :]
    sep_seq = _separator_sequence(grid[sep])
    assert all(color in SEPARATOR_COLORS for color in sep_seq)

    for panel in (top_in, bottom_in):
        assert set(_nonzero_colors(panel)) <= set(PANEL_COLORS)
        for r in range(PANEL_HEIGHT):
            count = int(np.count_nonzero(panel[r]))
            assert 1 <= count <= 3, "each panel row must contain 1-3 foreground pixels"

    top_post = np.fliplr(top_in)
    bottom_post = np.fliplr(bottom_in)
    assert _post_panels_valid(top_post, bottom_post)

    merged = _overlay_top_priority(top_post, bottom_post)
    mapping = _panel_color_mapping(top_in, bottom_in, sep_seq)
    assert mapping[1] == sep_seq[0] and mapping[2] == sep_seq[1]

    overlap = (top_post != BG) & (bottom_post != BG)
    assert np.any(overlap & (top_post != bottom_post)), "top-priority conflicts must matter"
    assert np.any((top_post != BG) & (bottom_post == BG)), "top-only pixels must survive"
    assert np.any((top_post == BG) & (bottom_post != BG)), "bottom-only pixels must survive"

    expected = _recolor(merged, mapping)
    assert np.array_equal(out, expected)
    assert set(_nonzero_colors(out)) == {sep_seq[0], sep_seq[1]}

    no_flip = _recolor(_overlay_top_priority(top_in, bottom_in), mapping)
    bottom_priority = _recolor(_overlay_bottom_priority(top_post, bottom_post), mapping)
    top_only = _recolor(top_post, mapping)
    bottom_only = _recolor(bottom_post, mapping)
    swapped = _recolor(merged, {1: sep_seq[1], 2: sep_seq[0]})

    assert not np.array_equal(out, merged), "recoloring must change the merged panel"
    assert not np.array_equal(out, no_flip), "local reversal must affect the result"
    assert not np.array_equal(out, bottom_priority), "top-priority overlay must affect the result"
    assert not np.array_equal(out, top_only), "the bottom panel must contribute"
    assert not np.array_equal(out, bottom_only), "the top panel must contribute"
    assert not np.array_equal(out, swapped), "separator order must matter"


def _gen_attempt() -> np.ndarray | None:
    width = random.randint(9, 15)
    panels = _random_post_panels(width)
    if panels is None:
        return None
    top_post, bottom_post = panels

    separator = _random_separator_row(width)
    if separator is None:
        return None

    grid = np.zeros((2 * PANEL_HEIGHT + 1, width), dtype=int)
    grid[:PANEL_HEIGHT] = np.fliplr(top_post)
    grid[PANEL_HEIGHT] = separator
    grid[PANEL_HEIGHT + 1 :] = np.fliplr(bottom_post)

    try:
        _validate_generated(grid)
    except Exception:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid separator-flip-overlay sample")
