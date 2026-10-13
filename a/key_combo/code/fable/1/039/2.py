from __future__ import annotations

from collections import Counter
import random

import numpy as np


GEN_MAX_ATTEMPTS = 500
GEN_PANEL_H_RANGE = (4, 6)
GEN_PANEL_W_RANGE = (5, 14)
GEN_GAP_MIN = 2
GEN_BAND_MIN = 4
GEN_STEP_WRAP_PROB = 0.5
GEN_COLOR_COUNT_RANGE = (3, 4)


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, np.asarray(grid).reshape(-1))).most_common(1)[0][0])


def _ring_coords(shape: tuple[int, int]) -> list[tuple[int, int]]:
    h, w = map(int, shape)
    assert h >= 2 and w >= 2, "panel border ring needs at least 2 rows and 2 columns"

    coords = [(0, c) for c in range(w)]
    coords += [(r, w - 1) for r in range(1, h)]
    coords += [(h - 1, c) for c in range(w - 2, -1, -1)]
    coords += [(r, 0) for r in range(h - 2, 0, -1)]
    return coords


def _corner_indices(shape: tuple[int, int]) -> tuple[int, int, int, int]:
    h, w = map(int, shape)
    return (0, w - 1, w + h - 2, w + h + w - 3)


def _panel_to_ring(panel: np.ndarray) -> np.ndarray:
    coords = _ring_coords(tuple(panel.shape))
    return np.array([int(panel[r, c]) for r, c in coords], dtype=int)


def _ring_to_panel(ring: np.ndarray, shape: tuple[int, int], bg: int) -> np.ndarray:
    out = np.full(shape, int(bg), dtype=int)
    coords = _ring_coords(shape)
    assert len(coords) == int(ring.shape[0]), "ring length must match panel perimeter"
    for (r, c), color in zip(coords, ring):
        out[r, c] = int(color)
    return out


def _is_single_cyclic_band(ring: np.ndarray, bg: int) -> bool:
    fg = ring != int(bg)
    if not np.any(fg):
        return False
    transitions = int(np.count_nonzero(fg != np.roll(fg, 1)))
    return transitions == 2 or bool(np.all(fg))


def _split_panels(grid: np.ndarray) -> tuple[list[np.ndarray], int]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _get_bg(g)
    height, width = map(int, g.shape)
    assert (height - 3) % 4 == 0, "expected four panels separated by three rows"
    panel_h = (height - 3) // 4
    assert panel_h >= 2 and width >= 2, "panels must be large enough to have a border ring"

    separator_rows = [panel_h, 2 * panel_h + 1, 3 * panel_h + 2]
    for row in separator_rows:
        assert np.all(g[row, :] == int(g[row, 0])), "separator rows must be solid"
        assert int(g[row, 0]) != bg, "separator rows must differ from the background"

    panels = [
        g[0:panel_h, :],
        g[panel_h + 1 : 2 * panel_h + 1, :],
        g[2 * panel_h + 2 : 3 * panel_h + 2, :],
        g[3 * panel_h + 3 : 4 * panel_h + 3, :],
    ]
    assert len({panel.shape for panel in panels}) == 1, "all panels must share one shape"
    return [panel.copy() for panel in panels], int(bg)


def _infer_step(rings: list[np.ndarray]) -> int:
    ring_len = int(rings[0].shape[0])
    candidates = [
        shift
        for shift in range(1, ring_len)
        if all(np.array_equal(np.roll(rings[i], shift), rings[i + 1]) for i in range(3))
    ]
    assert len(candidates) == 1, "expected a unique constant clockwise perimeter step"
    return int(candidates[0])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Continue the border patrol around the panel perimeter and return the next frame.

    Input structure:
    - The input is a vertical stack of four equal-size panels separated by full-width solid rows.
    - Each panel has one shared background color, an empty interior, and all non-background cells lie on the outer border ring.
    - In every panel those non-background border cells form one contiguous cyclic band whose colors keep the same order along the ring.

    Rule:
    - Split the filmstrip into its four panels and enumerate one panel's outer border cells in clockwise order.
    - Convert each panel border into a cyclic 1D ring and find the unique constant shift that maps frame 1 to 2, 2 to 3, and 3 to 4.
    - Apply that same clockwise shift once more to frame 4 and redraw the shifted ring on an otherwise empty panel.

    Key insight:
    - The motion is not a 2D translation inside the panel; it is a circular shift along the perimeter path, so corner turns only become simple after unwrapping the border into ring coordinates.
    """
    panels, bg = _split_panels(grid)
    rings = []
    for panel in panels:
        h, w = map(int, panel.shape)
        if h > 2 and w > 2:
            assert np.all(panel[1:-1, 1:-1] == bg), "panel interior must stay empty"
        ring = _panel_to_ring(panel)
        assert _is_single_cyclic_band(ring, bg), "panel border must contain one contiguous band"
        rebuilt = _ring_to_panel(ring, tuple(panel.shape), bg)
        assert np.array_equal(panel, rebuilt), "all non-background cells must lie on the border ring"
        rings.append(ring)

    step = _infer_step(rings)
    next_ring = np.roll(rings[-1], step)
    return _ring_to_panel(next_ring, tuple(panels[0].shape), bg)


def _stack_panels(panels: list[np.ndarray], separator_color: int) -> np.ndarray:
    height, width = map(int, panels[0].shape)
    out = np.full((4 * height + 3, width), int(separator_color), dtype=int)
    row = 0
    for i, panel in enumerate(panels):
        out[row : row + height, :] = panel
        row += height
        if i < 3:
            out[row, :] = int(separator_color)
            row += 1
    return out


def _has_proper_period(seq: list[int]) -> bool:
    n = len(seq)
    for period in range(1, n):
        if n % period != 0:
            continue
        if all(seq[i] == seq[i % period] for i in range(n)):
            return True
    return False


def _random_sequence(length: int, palette: list[int]) -> np.ndarray:
    while True:
        seq = [random.choice(palette)]
        while len(seq) < length:
            choices = [color for color in palette if color != seq[-1]]
            seq.append(random.choice(choices))
        if len(set(seq)) != len(palette):
            continue
        if _has_proper_period(seq):
            continue
        if seq == list(reversed(seq)):
            continue
        return np.array(seq, dtype=int)


def _band_ring(
    ring_len: int,
    start: int,
    colors: np.ndarray,
    bg: int,
) -> np.ndarray:
    ring = np.full(ring_len, int(bg), dtype=int)
    for offset, color in enumerate(colors):
        ring[(start + offset) % ring_len] = int(color)
    return ring


def _crossed_corners(
    start: int,
    band_len: int,
    step: int,
    corners: tuple[int, int, int, int],
    ring_len: int,
) -> set[int]:
    occupied = {(start + offset) % ring_len for offset in range(band_len)}
    crossed: set[int] = set()
    for corner in corners:
        for delta in range(1, step + 1):
            if (corner - delta) % ring_len in occupied:
                crossed.add(int(corner))
                break
    return crossed


def _valid_separator_rows(grid: np.ndarray, bg: int) -> bool:
    height = int(grid.shape[0])
    if (height - 3) % 4 != 0:
        return False
    panel_h = (height - 3) // 4
    return all(
        np.all(grid[row, :] == int(grid[row, 0])) and int(grid[row, 0]) != bg
        for row in (panel_h, 2 * panel_h + 1, 3 * panel_h + 2)
    )


def _generate_case() -> tuple[np.ndarray, np.ndarray]:
    for _ in range(GEN_MAX_ATTEMPTS):
        panel_h = random.randint(*GEN_PANEL_H_RANGE)
        panel_w = random.randint(*GEN_PANEL_W_RANGE)
        shape = (panel_h, panel_w)
        ring_len = len(_ring_coords(shape))
        max_band = ring_len - GEN_GAP_MIN
        if max_band < GEN_BAND_MIN:
            continue

        bg = random.randrange(10)
        remaining = [color for color in range(10) if color != bg]
        separator = random.choice(remaining)
        remaining = [color for color in remaining if color != separator]

        color_count = random.randint(
            GEN_COLOR_COUNT_RANGE[0],
            min(GEN_COLOR_COUNT_RANGE[1], len(remaining)),
        )
        palette = random.sample(remaining, color_count)

        band_len = random.randint(GEN_BAND_MIN, max_band)
        colors = _random_sequence(band_len, palette)

        max_step = min(ring_len - 1, max(panel_h, panel_w) + 2)
        if max_step < 1:
            continue
        step = random.randint(1, max_step)
        start0 = random.randrange(ring_len)
        corners = _corner_indices(shape)

        start3 = (start0 + 3 * step) % ring_len
        crossed = _crossed_corners(start3, band_len, step, corners, ring_len)
        if not crossed:
            continue

        want_wrap = random.random() < GEN_STEP_WRAP_PROB
        if want_wrap and 0 not in crossed:
            continue
        if not want_wrap and 0 in crossed:
            continue

        rings = [
            _band_ring(ring_len, (start0 + frame * step) % ring_len, colors, bg)
            for frame in range(5)
        ]
        ring_keys = {(tuple(ring.tolist())) for ring in rings}
        if len(ring_keys) != 5:
            continue

        panels = [_ring_to_panel(ring, shape, bg) for ring in rings]
        inp = _stack_panels(panels[:4], separator)
        expected = panels[4]

        if _get_bg(inp) != bg:
            continue
        if not _valid_separator_rows(inp, bg):
            continue
        if max(inp.shape) > 30 or max(expected.shape) > 30:
            continue

        try:
            out = main(inp)
        except Exception:
            continue
        if not np.array_equal(out, expected):
            continue
        return inp, expected

    raise RuntimeError("failed to generate a valid border-patrol animation")


def generate_input() -> np.ndarray:
    return _generate_case()[0]
