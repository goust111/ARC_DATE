"""ARC task: decode colored glyph keys and adjacent grey orientation keys."""

from __future__ import annotations

import random

import numpy as np


BACKGROUND = 0
GREY = 5
DIRECTIONS = ((-1, 0), (0, 1), (1, 0), (0, -1))  # up, right, down, left
COLORS = (1, 2, 3, 4, 6, 7, 8, 9)

# Each palette color has a fixed, asymmetric, 4-connected glyph.  The arrays
# show the dictionary's visible "up" orientation; color numbers are merely keys.
_GLYPH_ROWS = (
    ((0, 0, 0), (0, 1, 1), (1, 1, 1)),
    ((0, 0, 0), (1, 1, 0), (1, 1, 1)),
    ((0, 0, 0), (1, 1, 1), (0, 1, 1)),
    ((0, 0, 0), (1, 1, 1), (1, 0, 1)),
    ((0, 0, 0), (1, 1, 1), (1, 1, 0)),
    ((0, 0, 1), (0, 1, 1), (1, 1, 0)),
    ((0, 0, 1), (1, 1, 1), (0, 0, 1)),
    ((0, 0, 1), (1, 1, 1), (0, 1, 0)),
)
GLYPHS = {
    color: np.asarray(rows, dtype=bool)
    for color, rows in zip(COLORS, _GLYPH_ROWS)
}


def _demo_slots(width: int) -> tuple[tuple[int, int, int, int], ...]:
    """Return (patch_row, patch_col, label_row, label_col) for the two corners."""
    assert width >= 10
    return ((0, 0, 3, 3), (0, width - 4, 3, width - 1))


def _extract_dictionary(grid: np.ndarray) -> tuple[dict[int, np.ndarray], set[tuple[int, int]]]:
    glyphs: dict[int, np.ndarray] = {}
    demo_cells: set[tuple[int, int]] = set()
    for r0, c0, lr, lc in _demo_slots(grid.shape[1]):
        color = int(grid[lr, lc])
        assert color not in (BACKGROUND, GREY), "each corner sample needs a colored label"
        assert color not in glyphs, "sample labels must name distinct colors"
        patch = grid[r0 : r0 + 3, c0 : c0 + 3]
        assert set(map(int, np.unique(patch))) <= {BACKGROUND, color}
        mask = patch == color
        assert mask[1, 1] and 3 <= int(mask.sum()) <= 7
        rotations = {np.rot90(mask, k).tobytes() for k in range(4)}
        assert len(rotations) == 4, "glyph must have four distinguishable orientations"
        glyphs[color] = mask
        demo_cells.update((r, c) for r in range(r0, r0 + 3) for c in range(c0, c0 + 3))
        demo_cells.add((lr, lc))
    return glyphs, demo_cells


def _oriented(mask: np.ndarray, direction_index: int) -> np.ndarray:
    # np.rot90 uses counter-clockwise quarter turns; directions are clockwise.
    return np.rot90(mask, -int(direction_index))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Replace each colored control pixel by its corner-sampled 3x3 glyph, rotated toward its adjacent grey pixel.

    Input structure:
    - A black grid contains two asymmetric 3x3 glyph samples in the upper
      corners, each with a same-color label immediately down-right.
    - Elsewhere, isolated colored control pixels use those two label colors;
      every control has exactly one grey pixel in its four-neighborhood.
    - Control neighborhoods do not overlap, though a centered 3x3 footprint may
      cross a canvas edge and therefore require clipping.

    Rule:
    - Read each sample as that color's glyph in its baseline upward orientation.
    - For every control, rotate its color's glyph so upward points toward the
      adjacent grey pixel, clear the in-bounds 3x3 footprint, and stamp the
      colored glyph centered on the control, clipping at the grid boundary.
    - Erase both corner samples and their labels from the output.

    Key insight:
    - Color and the grey neighbor are independent channels: color selects what
      glyph to stamp, while the grey neighbor selects which rotation to use.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and min(g.shape) >= 10
    assert int(np.sum(g == BACKGROUND)) > g.size // 2, "black must be the background"

    glyphs, demo_cells = _extract_dictionary(g)
    controls: list[tuple[int, int, int, int]] = []
    used_controls: set[tuple[int, int]] = set()
    grey_coords = np.argwhere(g == GREY)
    assert len(grey_coords) >= 4

    for gr, gc in grey_coords:
        gr, gc = int(gr), int(gc)
        adjacent: list[tuple[int, int, int]] = []
        for direction_index, (dr, dc) in enumerate(DIRECTIONS):
            cr, cc = gr - dr, gc - dc
            if 0 <= cr < g.shape[0] and 0 <= cc < g.shape[1]:
                color = int(g[cr, cc])
                if color in glyphs and (cr, cc) not in demo_cells:
                    adjacent.append((cr, cc, color))
        assert len(adjacent) == 1, "each grey pixel must orient exactly one control"
        cr, cc, color = adjacent[0]
        assert (cr, cc) not in used_controls, "each control needs one grey neighbor"
        used_controls.add((cr, cc))
        direction_index = DIRECTIONS.index((gr - cr, gc - cc))
        controls.append((cr, cc, color, direction_index))

    # There must be no unexplained colored work pixels.
    work_colors = set(glyphs)
    unexplained = {
        (int(r), int(c))
        for r, c in np.argwhere(np.isin(g, list(work_colors)))
        if (int(r), int(c)) not in demo_cells and (int(r), int(c)) not in used_controls
    }
    assert not unexplained

    out = g.copy()
    for r, c in demo_cells:
        out[r, c] = BACKGROUND

    # Clear all footprints before painting, making simultaneous stamping explicit.
    h, w = g.shape
    for r, c, _, _ in controls:
        out[max(0, r - 1) : min(h, r + 2), max(0, c - 1) : min(w, c + 2)] = BACKGROUND
    for r, c, color, direction_index in controls:
        mask = _oriented(glyphs[color], direction_index)
        for mr, mc in np.argwhere(mask):
            rr, cc = r + int(mr) - 1, c + int(mc) - 1
            if 0 <= rr < h and 0 <= cc < w:
                out[rr, cc] = color
    return out


def _paint_demo(grid: np.ndarray, slot: tuple[int, int, int, int], color: int) -> None:
    r0, c0, lr, lc = slot
    patch = np.where(GLYPHS[color], color, BACKGROUND)
    grid[r0 : r0 + 3, c0 : c0 + 3] = patch
    grid[lr, lc] = color


def _candidate_centers(
    height: int,
    width: int,
    direction_index: int,
    placed: list[tuple[int, int, int, int]],
    *,
    require_clipping: bool,
    mask: np.ndarray,
) -> list[tuple[int, int]]:
    dr, dc = DIRECTIONS[direction_index]
    candidates: list[tuple[int, int]] = []
    for r in range(5, height):
        for c in range(width):
            gr, gc = r + dr, c + dc
            if not (0 <= gr < height and 0 <= gc < width):
                continue
            if any(max(abs(r - pr), abs(c - pc)) < 4 for pr, pc, _, _ in placed):
                continue
            rotated = _oriented(mask, direction_index)
            clipped_ink = any(
                rotated[mr, mc] and not (0 <= r + mr - 1 < height and 0 <= c + mc - 1 < width)
                for mr in range(3) for mc in range(3)
            )
            if require_clipping != bool(clipped_ink):
                continue
            candidates.append((r, c))
    random.shuffle(candidates)
    return candidates


def _render_known(
    grid: np.ndarray,
    colors: tuple[int, int],
    specs: list[tuple[int, int, int, int]],
    *,
    rotate: bool = True,
    single_shape: bool = False,
) -> np.ndarray:
    out = grid.copy()
    for r0, c0, lr, lc in _demo_slots(grid.shape[1]):
        out[r0 : r0 + 3, c0 : c0 + 3] = BACKGROUND
        out[lr, lc] = BACKGROUND
    h, w = out.shape
    for r, c, _, _ in specs:
        out[max(0, r - 1) : min(h, r + 2), max(0, c - 1) : min(w, c + 2)] = BACKGROUND
    for r, c, color, direction_index in specs:
        shape_color = colors[0] if single_shape else color
        mask = GLYPHS[shape_color]
        if rotate:
            mask = _oriented(mask, direction_index)
        for mr, mc in np.argwhere(mask):
            rr, cc = r + int(mr) - 1, c + int(mc) - 1
            if 0 <= rr < h and 0 <= cc < w:
                out[rr, cc] = color
    return out


def _valid_generated(
    grid: np.ndarray,
    colors: tuple[int, int],
    specs: list[tuple[int, int, int, int]],
) -> bool:
    expected = _render_known(grid, colors, specs)
    try:
        actual = main(grid)
    except (AssertionError, ValueError):
        return False
    if not np.array_equal(actual, expected) or np.array_equal(actual, grid):
        return False

    # Semantic anti-shortcut checks: both channels and rotation change the result.
    if np.array_equal(expected, _render_known(grid, colors, specs, rotate=False)):
        return False
    if np.array_equal(expected, _render_known(grid, colors, specs, single_shape=True)):
        return False
    directions_by_color = {
        color: {d for _, _, col, d in specs if col == color}
        for color in colors
    }
    if set().union(*directions_by_color.values()) != set(range(4)):
        return False
    if any(len(ds) < 2 for ds in directions_by_color.values()):
        return False
    if not (directions_by_color[colors[0]] & directions_by_color[colors[1]]):
        return False
    if np.any(actual == GREY):
        return False
    return True


def generate_input() -> np.ndarray:
    """Generate a diverse, oracle-checked dual-key stamping instance."""
    for _ in range(400):
        height = random.randint(15, 24)
        width = random.randint(14, 24)
        colors = tuple(random.sample(COLORS, 2))
        grid = np.zeros((height, width), dtype=int)
        for slot, color in zip(_demo_slots(width), colors):
            _paint_demo(grid, slot, color)

        # Both colors vary in direction, both share up, and all four directions occur.
        requests = [
            (colors[0], 0), (colors[0], 1),
            (colors[1], 0), (colors[1], 3),
            (random.choice(colors), 2),
        ]
        for _extra in range(random.randint(1, 4)):
            requests.append((random.choice(colors), random.randrange(4)))
        random.shuffle(requests)

        specs: list[tuple[int, int, int, int]] = []
        failed = False
        # Force one edge case whose colored ink, not merely its bbox, is clipped.
        edge_index = random.randrange(len(requests))
        for i, (color, direction_index) in enumerate(requests):
            candidates = _candidate_centers(
                height, width, direction_index, specs,
                require_clipping=(i == edge_index), mask=GLYPHS[color],
            )
            if not candidates:
                failed = True
                break
            r, c = candidates[0]
            specs.append((r, c, color, direction_index))
        if failed:
            continue

        for r, c, color, direction_index in specs:
            dr, dc = DIRECTIONS[direction_index]
            grid[r, c] = color
            grid[r + dr, c + dc] = GREY

        if _valid_generated(grid, colors, specs):
            return grid
    raise RuntimeError("failed to generate a valid dual-key stamp puzzle")
