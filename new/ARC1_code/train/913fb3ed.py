from __future__ import annotations

import numpy as np

from arc_utils import Color, get_bg


OUTLINE_COLOR = {
    Color.GREEN: Color.PINK,
    Color.TEAL: Color.YELLOW,
    Color.RED: Color.BLUE,
}


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand each isolated colored marker into a centered 3x3 hollow outline whose color is paired with the marker color.

    Input structure:
    - A black background contains one or more isolated single-pixel markers.
    - Marker colors are red, green, and/or teal.
    - Each marker has an empty one-cell neighborhood, so a centered 3x3 outline
      fits without hitting other foreground.

    Rule:
    - Keep each marker unchanged.
    - Around each green marker, paint the surrounding 3x3 outline pink.
    - Around each teal marker, paint the surrounding 3x3 outline yellow.
    - Around each red marker, paint the surrounding 3x3 outline blue.

    Key insight:
    - The marker pixel is the center; its color selects the outline color, while
      the geometry is always the same 3x3 hollow square.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    assert bg == Color.BLACK, "expected black background"

    marker_coords = np.argwhere(g != bg)
    assert marker_coords.size > 0, "expected at least one marker"

    h, w = g.shape
    out = g.copy()
    for r_raw, c_raw in marker_coords:
        r, c = int(r_raw), int(c_raw)
        marker_color = int(g[r, c])
        assert marker_color in OUTLINE_COLOR, "unexpected marker color"
        assert 1 <= r < h - 1 and 1 <= c < w - 1, "3x3 outline must fit"

        outline_color = OUTLINE_COLOR[marker_color]
        for rr in range(r - 1, r + 2):
            for cc in range(c - 1, c + 2):
                if rr == r and cc == c:
                    continue
                assert g[rr, cc] == bg, "marker neighborhoods must be empty"
                assert out[rr, cc] == bg, "marker outlines must not overlap"
                out[rr, cc] = outline_color

    return out

# --------------- generator (o5x) ---------------
# Canvas bounds, inclusive.
GEN_GRID_MIN_SIZE = 6
GEN_GRID_MAX_SIZE = 16

# Marker count and palette.
GEN_MARKER_MIN_COUNT = 1
GEN_MARKER_MAX_COUNT = 3
GEN_MARKER_COLORS = np.array([Color.RED, Color.GREEN, Color.TEAL], dtype=int)

# Marker geometry and placement.
GEN_BACKGROUND_COLOR = Color.BLACK
GEN_OUTLINE_RADIUS = 1
GEN_MIN_CENTER_SEPARATION = 3

# Retry budget for oracle-checked generation.
GEN_MAX_ATTEMPTS = 300


def _gen_centers_are_separated(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return max(abs(a[0] - b[0]), abs(a[1] - b[1])) >= GEN_MIN_CENTER_SEPARATION


def _gen_sample_centers(h: int, w: int, count: int) -> list[tuple[int, int]] | None:
    """Sample non-overlapping marker centers, or return None if no packing is found."""
    centers = [
        (r, c)
        for r in range(GEN_OUTLINE_RADIUS, h - GEN_OUTLINE_RADIUS)
        for c in range(GEN_OUTLINE_RADIUS, w - GEN_OUTLINE_RADIUS)
    ]

    def rec(
        chosen: list[tuple[int, int]], available: list[tuple[int, int]], remaining: int
    ) -> list[tuple[int, int]] | None:
        if remaining == 0:
            return chosen
        if len(available) < remaining:
            return None

        for idx in np.random.permutation(len(available)):
            selected_idx = int(idx)
            center = available[selected_idx]
            next_available = [
                other
                for j, other in enumerate(available)
                if j != selected_idx and _gen_centers_are_separated(center, other)
            ]
            result = rec(chosen + [center], next_available, remaining - 1)
            if result is not None:
                return result
        return None

    return rec([], centers, count)


def _gen_expected_output(inp: np.ndarray) -> np.ndarray:
    out = inp.copy()
    for r_raw, c_raw in np.argwhere(inp != GEN_BACKGROUND_COLOR):
        r, c = int(r_raw), int(c_raw)
        outline_color = OUTLINE_COLOR[int(inp[r, c])]
        for rr in range(r - GEN_OUTLINE_RADIUS, r + GEN_OUTLINE_RADIUS + 1):
            for cc in range(c - GEN_OUTLINE_RADIUS, c + GEN_OUTLINE_RADIUS + 1):
                if rr != r or cc != c:
                    out[rr, cc] = outline_color
    return out


def _gen_attempt() -> np.ndarray | None:
    h = int(np.random.randint(GEN_GRID_MIN_SIZE, GEN_GRID_MAX_SIZE + 1))
    w = int(np.random.randint(GEN_GRID_MIN_SIZE, GEN_GRID_MAX_SIZE + 1))
    count = int(np.random.randint(GEN_MARKER_MIN_COUNT, GEN_MARKER_MAX_COUNT + 1))

    centers = _gen_sample_centers(h, w, count)
    if centers is None:
        return None

    colors = GEN_MARKER_COLORS[np.random.permutation(len(GEN_MARKER_COLORS))[:count]]
    grid = np.full((h, w), GEN_BACKGROUND_COLOR, dtype=int)
    for (r, c), color in zip(centers, colors):
        grid[r, c] = int(color)
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        inp = _gen_attempt()
        if inp is None:
            continue

        expected = _gen_expected_output(inp)
        try:
            actual = main(inp)
        except Exception:
            continue
        if np.array_equal(actual, expected):
            return inp

    raise RuntimeError(f"generate_input: failed after {GEN_MAX_ATTEMPTS} attempts")
