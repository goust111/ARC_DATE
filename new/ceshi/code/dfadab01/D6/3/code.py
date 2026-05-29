# Variant of dfadab01
# Dimension: D6 — 网格边缘裁剪规则 (grid_shape)
# Change: 字形放置时如果超出网格边界，超出部分被裁剪（只显示网格内的部分）... → 禁止裁剪——仅处理完全在网格内的标记

from __future__ import annotations

import random

import numpy as np


# Design analysis for the D6 variant
# a) Concretization and behavior trace
#    - The scene still contains single-cell markers in colors {2, 3, 5, 8} and an
#      optional legend made of a 4x4 sample glyph plus a label marker at (+4, +4).
#    - Each real marker still points to the same fixed 4x4 glyph/color pair.
#    - The changed behavior is positional: a marker is active only when its full 4x4
#      glyph box lies inside the grid. Markers in the last three rows and/or last
#      three columns are now ignored instead of being partially drawn.
#    - Example on a 10x10 grid: marker 2 at (7, 1).
#      Old rule: paint rows 7..9 of the yellow frame and clip row 10 away.
#      New rule: 7 + 4 > 10, so the marker is discarded and contributes nothing.
# b) Complexity calibration
#    - The glyph dictionary, colors, legend mechanics, and marker lattice stay fixed.
#      Only the boundary policy changes, so the task remains close to the source.
#    - To make the new branch visible, generated inputs always contain both:
#      at least one fully fitting marker and at least one overflowing marker whose
#      clipped rendering would have produced visible pixels under the old rule.
# c) Implicit-assumption audit for main()
#    - Original _paint_clipped() assumed every marker should be rendered somehow.
#      That assumption is no longer valid; rendering must now be gated by a fit test.
#    - Legend skipping via _has_source_glyph() still works because legend glyphs are
#      always fully inside the grid; otherwise a label marker could be misread.
#    - The output is still built on a blank canvas, so ignoring a marker is safe:
#      there is no need to erase anything because no partial glyph is ever written.
# d) Placement legality
#    - Real markers are placed on a regular grid with spacing 4..6, so fully fitting
#      glyphs never overlap each other.
#    - The generator chooses row/column sequences so that some coordinates cross the
#      fit threshold H-4 or W-4 while remaining valid input cells inside the grid.
#    - The optional legend is placed in a marker-free 5x5 box, guaranteeing that the
#      sample glyph is recognizable and its label marker is skipped correctly.
# e) Visual observability
#    - Each sample contains surviving markers and skipped edge markers together.
#      A human can compare similar marker placements and notice that edge-near ones
#      disappear completely instead of leaving cropped fragments.
#    - The generator rejects cases where old clipped output and new output coincide.
# f) Visual cleanliness
#    - Marker spacing is at least 4, so surviving 4x4 glyphs do not overlap.
#    - The legend is isolated from the marker lattice, avoiding ambiguous overlays.
# g) Color inference
#    - Color mapping is unchanged from the original task. No new color-dependent rule
#      is introduced, so the only extra inference burden is the in-bounds test.

GLYPH_SIZE = 4
MARKER_COLORS = (2, 3, 5, 8)
GLYPH_COLORS = (1, 4, 6, 7)
ALLOWED_COLORS = {0, *MARKER_COLORS, *GLYPH_COLORS}
MAX_RETRIES = 400

GLYPHS: dict[int, tuple[int, np.ndarray]] = {
    2: (
        4,
        np.array(
            [
                [1, 1, 1, 1],
                [1, 0, 0, 1],
                [1, 0, 0, 1],
                [1, 1, 1, 1],
            ],
            dtype=bool,
        ),
    ),
    3: (
        1,
        np.array(
            [
                [0, 1, 1, 0],
                [1, 0, 0, 1],
                [1, 0, 0, 1],
                [0, 1, 1, 0],
            ],
            dtype=bool,
        ),
    ),
    5: (
        6,
        np.array(
            [
                [1, 1, 0, 0],
                [1, 1, 0, 0],
                [0, 0, 1, 1],
                [0, 0, 1, 1],
            ],
            dtype=bool,
        ),
    ),
    8: (
        7,
        np.array(
            [
                [1, 0, 0, 1],
                [0, 1, 1, 0],
                [0, 1, 1, 0],
                [1, 0, 0, 1],
            ],
            dtype=bool,
        ),
    ),
}


def _has_source_glyph(g: np.ndarray, r0: int, c0: int, color: int, mask: np.ndarray) -> bool:
    h, w = g.shape
    if r0 < 0 or c0 < 0 or r0 + GLYPH_SIZE > h or c0 + GLYPH_SIZE > w:
        return False
    patch = g[r0 : r0 + GLYPH_SIZE, c0 : c0 + GLYPH_SIZE]
    return bool(np.all(patch[mask] == color) and np.all(patch[~mask] != color))


def _fits_glyph(shape: tuple[int, int], r0: int, c0: int) -> bool:
    h, w = shape
    return 0 <= r0 and 0 <= c0 and r0 + GLYPH_SIZE <= h and c0 + GLYPH_SIZE <= w


def _paint_full(out: np.ndarray, r0: int, c0: int, color: int, mask: np.ndarray) -> None:
    out[r0 : r0 + GLYPH_SIZE, c0 : c0 + GLYPH_SIZE][mask] = color


def _paint_clipped(out: np.ndarray, r0: int, c0: int, color: int, mask: np.ndarray) -> None:
    h, w = out.shape
    for dr in range(GLYPH_SIZE):
        rr = r0 + dr
        if rr < 0 or rr >= h:
            continue
        for dc in range(GLYPH_SIZE):
            cc = c0 + dc
            if 0 <= cc < w and bool(mask[dr, dc]):
                out[rr, cc] = color


def _render(grid: np.ndarray, *, allow_clipping: bool) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert set(map(int, np.unique(g))).issubset(ALLOWED_COLORS)

    out = np.zeros_like(g)
    seen_marker = False
    for marker_color, (glyph_color, mask) in GLYPHS.items():
        coords = np.argwhere(g == marker_color)
        if coords.size:
            seen_marker = True
        for r, c in coords:
            rr = int(r)
            cc = int(c)
            if _has_source_glyph(g, rr - GLYPH_SIZE, cc - GLYPH_SIZE, glyph_color, mask):
                continue
            if allow_clipping:
                _paint_clipped(out, rr, cc, glyph_color, mask)
            elif _fits_glyph(g.shape, rr, cc):
                _paint_full(out, rr, cc, glyph_color, mask)

    assert seen_marker
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand each eligible control pixel into its fixed 4x4 glyph, but only when the full glyph fits.

    Input structure:
    - Black background with single-cell markers in colors 2, 3, 5, and/or 8.
    - Some inputs also include one legend entry: a 4x4 sample glyph in the output color plus a same-type label marker placed one row-block down-right.

    Rule:
    - Map each marker color to the same fixed glyph alphabet as the original puzzle.
    - Ignore legend label markers, and erase the legend sample itself from the output.
    - For real markers, draw the glyph only if its entire 4x4 footprint stays inside the grid; edge-overflowing markers are skipped completely.

    Key insight:
    - Marker color still selects the glyph, but position now matters too: a boundary-near marker is inactive if its glyph would need clipping.
    """
    return _render(grid, allow_clipping=False)


def _choose_axis_positions(size: int, rng: random.Random, require_overflow: bool) -> list[int]:
    fit_limit = size - GLYPH_SIZE
    for _ in range(200):
        count = rng.randint(2, 5)
        spacing = rng.randint(4, 6)
        max_start = size - 1 - spacing * (count - 1)
        if max_start < 0:
            continue

        if require_overflow:
            low = max(0, fit_limit - spacing * (count - 1) + 1)
            high = min(max_start, fit_limit)
            if low > high:
                continue
            start = rng.randint(low, high)
        else:
            high = min(max_start, fit_limit - spacing * (count - 1))
            if high < 0:
                continue
            start = rng.randint(0, high)

        positions = [start + spacing * i for i in range(count)]
        has_fit = any(p <= fit_limit for p in positions)
        has_overflow = any(p > fit_limit for p in positions)
        if require_overflow:
            if has_fit and has_overflow:
                return positions
        elif has_fit and not has_overflow:
            return positions

    raise ValueError("could not choose axis positions")


def _place_legend(
    grid: np.ndarray,
    marker_positions: set[tuple[int, int]],
    used_colors: tuple[int, ...],
    rng: random.Random,
) -> bool:
    h, w = grid.shape
    candidates: list[tuple[int, int, int]] = []
    for marker_color in used_colors:
        for r0 in range(h - GLYPH_SIZE):
            for c0 in range(w - GLYPH_SIZE):
                blocked = False
                for rr in range(r0, r0 + GLYPH_SIZE + 1):
                    for cc in range(c0, c0 + GLYPH_SIZE + 1):
                        if (rr, cc) in marker_positions:
                            blocked = True
                            break
                    if blocked:
                        break
                if blocked:
                    continue
                if np.any(grid[r0 : r0 + GLYPH_SIZE + 1, c0 : c0 + GLYPH_SIZE + 1] != 0):
                    continue
                candidates.append((marker_color, r0, c0))

    if not candidates:
        return False

    marker_color, r0, c0 = rng.choice(candidates)
    glyph_color, mask = GLYPHS[marker_color]
    patch = grid[r0 : r0 + GLYPH_SIZE, c0 : c0 + GLYPH_SIZE]
    patch[mask] = glyph_color
    grid[r0 + GLYPH_SIZE, c0 + GLYPH_SIZE] = marker_color
    return True


def generate_input() -> np.ndarray:
    rng = random.Random()

    for _ in range(MAX_RETRIES):
        h = rng.choice((10, 20))
        w = rng.choice((10, 20))

        overflow_rows = rng.choice((True, False))
        overflow_cols = True if not overflow_rows else rng.choice((True, False))

        try:
            row_positions = _choose_axis_positions(h, rng, overflow_rows)
            col_positions = _choose_axis_positions(w, rng, overflow_cols)
        except ValueError:
            continue

        cells = [(r, c) for r in row_positions for c in col_positions]
        fit_cells = [(r, c) for r, c in cells if _fits_glyph((h, w), r, c)]
        overflow_cells = [(r, c) for r, c in cells if not _fits_glyph((h, w), r, c)]
        if not fit_cells or not overflow_cells:
            continue

        fill_min = max(2, (3 * len(cells) + 4) // 5)
        fill_count = rng.randint(fill_min, len(cells))

        chosen = {
            rng.choice(fit_cells),
            rng.choice(overflow_cells),
        }
        remaining = [cell for cell in cells if cell not in chosen]
        rng.shuffle(remaining)
        for cell in remaining[: fill_count - len(chosen)]:
            chosen.add(cell)

        used_count = rng.randint(1, len(MARKER_COLORS))
        used_colors = tuple(sorted(rng.sample(MARKER_COLORS, used_count)))

        grid = np.zeros((h, w), dtype=int)
        marker_positions: set[tuple[int, int]] = set()
        for r, c in chosen:
            marker_positions.add((r, c))
            grid[r, c] = rng.choice(used_colors)

        if rng.random() < 0.6:
            _place_legend(grid, marker_positions, used_colors, rng)

        new_out = main(grid)
        old_out = _render(grid, allow_clipping=True)
        if np.array_equal(new_out, old_out):
            continue
        if not np.any(new_out != 0):
            continue

        assert np.array_equal(new_out, _render(grid, allow_clipping=False))
        return grid

    raise RuntimeError("failed to generate a non-degenerate D6 sample")
