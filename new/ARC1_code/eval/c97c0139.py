from __future__ import annotations

import numpy as np

from arc_utils import Color, colors_present, find_components, get_bg


def main(grid: np.ndarray) -> np.ndarray:
    """
    Paint a teal diamond envelope around each straight foreground bar.

    Input structure
    - The grid has a single background color and one non-background foreground color.
    - Every foreground object is a solid horizontal or vertical one-cell-thick bar.
    - Teal is absent from the input and is reserved for the added envelope pixels.

    Rule
    - Keep every original foreground bar unchanged.
    - For a horizontal bar, paint teal rows above and below it; each step away from the bar shortens the interval by one pixel at both ends.
    - For a vertical bar, apply the same shrinking construction rotated ninety degrees.

    Key insight
    - A bar of length L defines a centered diamond of radius floor((L - 1) / 2), with the original bar forming the middle line and teal filling only the surrounding background cells.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = get_bg(g)
    colors = colors_present(g, exclude={bg})
    assert len(colors) == 1, "expected exactly one foreground color"
    bar_color = colors[0]
    fill_color = Color.TEAL
    assert fill_color != bg and fill_color != bar_color, "teal must be a new output color"
    assert not np.any(g == fill_color), "teal must not appear in the input"

    out = g.copy()
    height, width = map(int, g.shape)
    bars = find_components(g, bg=bg, connectivity=4, monochromatic=True, colors=[bar_color])
    assert bars, "expected at least one foreground bar"

    for bar in bars:
        r0, c0, r1, c1 = bar["bbox"]
        h = int(r1 - r0 + 1)
        w = int(c1 - c0 + 1)
        area = int(bar["area"])
        assert area == h * w, "foreground bar must be solid"
        assert h == 1 or w == 1, "foreground object must be a straight bar"

        length = max(h, w)
        radius = (length - 1) // 2
        assert radius >= 1, "bar must be long enough to form an envelope"

        if h == 1:
            row = int(r0)
            for offset in range(1, radius + 1):
                left = int(c0 + offset)
                right = int(c1 - offset)
                assert 0 <= left <= right < width, "horizontal envelope interval must stay in bounds"
                for rr in (row - offset, row + offset):
                    assert 0 <= rr < height, "horizontal envelope row must stay in bounds"
                    view = out[rr, left : right + 1]
                    view[view == bg] = fill_color
        else:
            col = int(c0)
            for offset in range(1, radius + 1):
                top = int(r0 + offset)
                bottom = int(r1 - offset)
                assert 0 <= top <= bottom < height, "vertical envelope interval must stay in bounds"
                for cc in (col - offset, col + offset):
                    assert 0 <= cc < width, "vertical envelope column must stay in bounds"
                    view = out[top : bottom + 1, cc]
                    view[view == bg] = fill_color

    return out

# --------------- generator (o5x) ---------------

# Canvas and colors.
GRID_HEIGHT_BOUNDS, GRID_WIDTH_BOUNDS = (17, 23), (19, 23)
BACKGROUND_COLOR, BAR_COLOR, ENVELOPE_COLOR = Color.BLACK, Color.RED, Color.TEAL

# Bar sampling.
BAR_COUNT_BOUNDS, BAR_LENGTH_BOUNDS = (1, 4), (5, 15)
ORIENTATIONS = ("H", "V")
SORT_TIE_BOUNDS = (0, 1_000_000)

# Generation retry control.
MAX_GENERATION_ATTEMPTS = 300


def _gen_bbox_for_bar(r: int, c: int, length: int, orientation: str) -> tuple[int, int, int, int]:
    radius = (length - 1) // 2
    if orientation == "H":
        return r - radius, c, r + radius, c + length - 1
    return r, c - radius, r + length - 1, c + radius


def _gen_anchor_ranges(height: int, width: int, length: int, orientation: str) -> tuple[range, range]:
    radius = (length - 1) // 2
    if orientation == "H":
        return range(radius, height - radius), range(0, width - length + 1)
    return range(0, height - length + 1), range(radius, width - radius)


def _gen_bar_specs(count: int) -> list[tuple[str, int]]:
    specs = [
        (ORIENTATIONS[int(np.random.randint(0, len(ORIENTATIONS)))], int(np.random.randint(*BAR_LENGTH_BOUNDS)))
        for _ in range(count)
    ]
    return sorted(specs, key=lambda item: (-item[1], int(np.random.randint(*SORT_TIE_BOUNDS))))


def _gen_legal_positions(
    blocked: np.ndarray, length: int, orientation: str
) -> list[tuple[int, int, tuple[int, int, int, int]]]:
    height, width = map(int, blocked.shape)
    positions: list[tuple[int, int, tuple[int, int, int, int]]] = []
    row_range, col_range = _gen_anchor_ranges(height, width, length, orientation)

    for r in row_range:
        for c in col_range:
            r0, c0, r1, c1 = _gen_bbox_for_bar(r, c, length, orientation)
            if not blocked[r0 : r1 + 1, c0 : c1 + 1].any():
                positions.append((r, c, (r0, c0, r1, c1)))

    return positions


def _gen_paint_bar_and_envelope(
    inp: np.ndarray, out: np.ndarray, r: int, c: int, length: int, orientation: str
) -> None:
    radius = (length - 1) // 2
    if orientation == "H":
        inp[r, c : c + length] = BAR_COLOR
        out[r, c : c + length] = BAR_COLOR
        for offset in range(1, radius + 1):
            left = c + offset
            right = c + length - 1 - offset
            out[r - offset, left : right + 1] = ENVELOPE_COLOR
            out[r + offset, left : right + 1] = ENVELOPE_COLOR
    else:
        inp[r : r + length, c] = BAR_COLOR
        out[r : r + length, c] = BAR_COLOR
        for offset in range(1, radius + 1):
            top = r + offset
            bottom = r + length - 1 - offset
            out[top : bottom + 1, c - offset] = ENVELOPE_COLOR
            out[top : bottom + 1, c + offset] = ENVELOPE_COLOR


def _gen_place_bar(inp: np.ndarray, out: np.ndarray, blocked: np.ndarray, orientation: str, length: int) -> bool:
    """Place one sampled bar, marking its envelope bbox as unavailable."""
    positions = _gen_legal_positions(blocked, length, orientation)
    if not positions:
        return False

    r, c, envelope_bbox = positions[int(np.random.randint(0, len(positions)))]
    r0, c0, r1, c1 = envelope_bbox
    blocked[r0 : r1 + 1, c0 : c1 + 1] = True
    _gen_paint_bar_and_envelope(inp, out, r, c, length, orientation)
    return True


def _gen_attempt() -> np.ndarray | None:
    height = int(np.random.randint(*GRID_HEIGHT_BOUNDS))
    width = int(np.random.randint(*GRID_WIDTH_BOUNDS))
    inp = np.full((height, width), BACKGROUND_COLOR, dtype=int)
    expected = np.full((height, width), BACKGROUND_COLOR, dtype=int)
    blocked = np.zeros((height, width), dtype=bool)

    count = int(np.random.randint(*BAR_COUNT_BOUNDS))
    for orientation, length in _gen_bar_specs(count):
        if not _gen_place_bar(inp, expected, blocked, orientation, length):
            return None
    try:
        return inp if np.array_equal(main(inp), expected) else None
    except Exception:
        return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("failed to generate valid c97c0139 input")
