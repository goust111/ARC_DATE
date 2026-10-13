import numpy as np

from arc_utils import colors_present, get_bg
import random


def _boundary_values(g: np.ndarray) -> np.ndarray:
    return np.concatenate([g[0, :], g[-1, :], g[1:-1, 0], g[1:-1, -1]])


def _full_inner_rows(g: np.ndarray, color: int) -> list[int]:
    inner = g[1:-1, 1:-1]
    return [r + 1 for r in range(inner.shape[0]) if np.all(inner[r, :] == color)]


def _full_inner_cols(g: np.ndarray, color: int) -> list[int]:
    inner = g[1:-1, 1:-1]
    return [c + 1 for c in range(inner.shape[1]) if np.all(inner[:, c] == color)]


def _periodic_axis_mask(
    length: int,
    observed_indices: np.ndarray,
    observed_values: np.ndarray,
) -> np.ndarray:
    obs = list(zip(map(int, observed_indices), map(bool, observed_values)))
    assert any(value for _, value in obs), "source stripes must appear on the observed axis"
    assert any(not value for _, value in obs), "source stripes must have gaps"

    for period in range(1, length + 1):
        residues: dict[int, bool] = {}
        ok = True
        for index, value in obs:
            residue = index % period
            if residue in residues and residues[residue] != value:
                ok = False
                break
            residues[residue] = value
        if ok and len(residues) == period:
            return np.array([residues[i % period] for i in range(length)], dtype=bool)

    assert False, "source stripe mask must be periodic"


def main(grid: np.ndarray) -> np.ndarray:
    """
    Project the framed periodic stripe pattern across the canvas and erase the unused frame cells.

    Input structure
    - A bg canvas with a one-cell-thick rectangular frame of one color.
    - Inside the frame are two parallel full-interior marker bars of another color.
    - A third color fills the region between the marker bars on periodically spaced rows or columns.

    Rule
    - Infer the periodic row or column mask from the third-color stripes inside the frame.
    - Paint the frame color along those full rows or columns, extended to the grid boundary.
    - Extend the marker bars across the full canvas and repaint the source-color stripe cells between them.

    Key insight
    - The outer frame is not preserved as a border; it supplies the color for the periodic stripes selected by the inner source pattern.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = g.shape
    assert h > 2 and w > 2, "grid must have an interior"

    bg = get_bg(g)
    boundary = _boundary_values(g)
    frame_color = int(boundary[0])
    assert frame_color != bg, "frame color must differ from background"
    assert np.all(boundary == frame_color), "input must have a uniform one-color boundary frame"

    fg_colors = colors_present(g, exclude={bg})
    assert len(fg_colors) == 3, "expected frame, marker-bar, and source-stripe colors"
    assert frame_color in fg_colors, "boundary frame color must be foreground"
    inner_colors = [color for color in fg_colors if color != frame_color]

    bar_candidates: list[tuple[int, list[int], list[int]]] = []
    for color in inner_colors:
        rows = _full_inner_rows(g, color)
        cols = _full_inner_cols(g, color)
        if bool(rows) ^ bool(cols):
            bar_candidates.append((color, rows, cols))
    assert len(bar_candidates) == 1, "marker-bar color must be structurally unique"

    bar_color, bar_rows, bar_cols = bar_candidates[0]
    source_colors = [color for color in inner_colors if color != bar_color]
    assert len(source_colors) == 1, "source-stripe color must be unique"
    source_color = source_colors[0]
    source_coords = np.argwhere(g == source_color)
    assert source_coords.size > 0, "source stripes must be present"

    out = np.full_like(g, bg)
    if bar_cols:
        assert not bar_rows, "marker bars must have one orientation"
        assert len(bar_cols) == 2, "expected two vertical marker bars"
        source_cols = sorted({int(c) for _, c in source_coords})
        left, right = sorted(bar_cols)
        assert source_cols == list(range(left + 1, right)), "source stripes must fill between marker bars"

        observed_rows = np.arange(1, h - 1)
        row_has_source = np.any(g[1 : h - 1, 1 : w - 1] == source_color, axis=1)
        selected_rows = _periodic_axis_mask(h, observed_rows, row_has_source)

        out[selected_rows, :] = frame_color
        out[:, bar_cols] = bar_color
        out[np.ix_(selected_rows, source_cols)] = source_color
    else:
        assert bar_rows, "marker bars must be present"
        assert len(bar_rows) == 2, "expected two horizontal marker bars"
        source_rows = sorted({int(r) for r, _ in source_coords})
        top, bottom = sorted(bar_rows)
        assert source_rows == list(range(top + 1, bottom)), "source stripes must fill between marker bars"

        observed_cols = np.arange(1, w - 1)
        col_has_source = np.any(g[1 : h - 1, 1 : w - 1] == source_color, axis=0)
        selected_cols = _periodic_axis_mask(w, observed_cols, col_has_source)

        out[:, selected_cols] = frame_color
        out[bar_rows, :] = bar_color
        out[np.ix_(source_rows, selected_cols)] = source_color

    return out

# --------------- generator (att2) ---------------

# Canvas dimensions, palette, and retry limit.
CANVAS_ROW_RANGE, CANVAS_COL_RANGE = (14, 19), (15, 19)
BACKGROUND_COLOR, BAR_COLOR, SOURCE_COLOR = 0, 3, 2
FRAME_COLORS = (1, 4, 5, 6, 7, 8, 9)
MAX_GENERATION_ATTEMPTS = 300

# Pattern and bar placement constraints.
ORIENTATION_CHOICES = (False, True)
ON_COUNT_RANGE, OFF_COUNT_RANGE = (1, 4), (2, 3)
MAX_PERIOD = 6
PATTERN_START_INDEX, FRAME_WIDTH, BAR_THICKNESS = 1, 1, 1


def _periodic_mask(length: int, on_count: int, off_count: int) -> np.ndarray:
    period = on_count + off_count
    return ((np.arange(length) - PATTERN_START_INDEX) % period) < on_count


def _bar_positions(axis_length: int, gap: int) -> tuple[int, int]:
    """Choose two one-cell bars with exactly gap cells between them."""
    first_max = axis_length - gap - FRAME_WIDTH - 2 * BAR_THICKNESS
    first = random.randint(FRAME_WIDTH, first_max)
    return first, first + gap + BAR_THICKNESS


def _derive_input(out: np.ndarray, frame_color: int) -> np.ndarray:
    inp = np.zeros_like(out)
    inp[[0, -1], :] = frame_color
    inp[:, [0, -1]] = frame_color
    inner = out[1:-1, 1:-1]
    active = (inner == BAR_COLOR) | (inner == SOURCE_COLOR)
    inp[1:-1, 1:-1] = np.where(active, inner, BACKGROUND_COLOR)
    return inp


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*CANVAS_ROW_RANGE)
    w = random.randint(*CANVAS_COL_RANGE)
    horizontal = random.choice(ORIENTATION_CHOICES)
    frame_color = random.choice(FRAME_COLORS)
    on_count = random.randint(*ON_COUNT_RANGE)
    max_off = min(OFF_COUNT_RANGE[1], MAX_PERIOD - on_count)
    off_count = random.randint(OFF_COUNT_RANGE[0], max_off)
    gap = on_count

    out = np.zeros((h, w), dtype=int)
    if horizontal:
        selected = _periodic_mask(w, on_count, off_count)
        top, bottom = _bar_positions(h, gap)
        source_rows = list(range(top + BAR_THICKNESS, bottom))

        out[:, selected] = frame_color
        out[[top, bottom], :] = BAR_COLOR
        out[np.ix_(source_rows, np.flatnonzero(selected))] = SOURCE_COLOR
    else:
        selected = _periodic_mask(h, on_count, off_count)
        left, right = _bar_positions(w, gap)
        source_cols = list(range(left + BAR_THICKNESS, right))

        out[selected, :] = frame_color
        out[:, [left, right]] = BAR_COLOR
        out[np.ix_(np.flatnonzero(selected), source_cols)] = SOURCE_COLOR

    inp = _derive_input(out, frame_color)
    try:
        solved = main(inp)
    except AssertionError:
        return None
    return inp if np.array_equal(solved, out) else None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 6a980be1 input")
