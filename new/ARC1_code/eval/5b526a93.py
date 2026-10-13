from __future__ import annotations

import numpy as np

from arc_utils import Color, find_components, get_bg, paste


def _instances(g: np.ndarray, bg: int) -> list[tuple[int, int, np.ndarray]]:
    comps = find_components(g, bg=bg, connectivity=4, monochromatic=True)
    assert comps, "expected foreground sprites"

    colors = {int(comp["color"]) for comp in comps}
    assert len(colors) == 1, "expected a single sprite color"
    color = colors.pop()

    out: list[tuple[int, int, np.ndarray]] = []
    for comp in comps:
        r0, c0, r1, c1 = map(int, comp["bbox"])
        patch = g[r0 : r1 + 1, c0 : c1 + 1]
        assert np.all((patch == bg) | (patch == color)), "sprite bbox must contain only bg and sprite color"
        mask = patch == color
        assert int(mask.sum()) == int(comp["area"]), "component area must match its local mask"
        out.append((r0, c0, mask))

    ref = out[0][2]
    assert ref.size > 0, "sprite mask must be non-empty"
    assert all(mask.shape == ref.shape and np.array_equal(mask, ref) for _r, _c, mask in out), (
        "all sprites must share one shape"
    )
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete a rectangular table of identical sprites by stamping teal copies at the empty row-column intersections.

    Input structure
    - A black background contains one foreground color forming identical connected sprites.
    - The sprites sit on a rectangular row/column lattice.
    - The occupied lattice slots include at least one complete row and at least one complete column.

    Rule
    - Find the common sprite mask and the top-left position of every existing sprite.
    - Use the existing sprite row starts and column starts as the table lattice.
    - Preserve existing sprites and paste a teal copy of the same mask into every unoccupied lattice slot.

    Key insight
    The present sprites are the row and column headers of a table; the missing Cartesian-product slots are the cells to fill.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    bg = int(get_bg(g))
    sprites = _instances(g, bg)
    rows = sorted({int(r) for r, _c, _mask in sprites})
    cols = sorted({int(c) for _r, c, _mask in sprites})
    assert len(rows) >= 2 and len(cols) >= 2, "expected a row-column sprite lattice"

    occupied = {(int(r), int(c)) for r, c, _mask in sprites}
    assert len(occupied) == len(sprites), "sprite top-left positions must be unique"
    assert any(all((r, c) in occupied for c in cols) for r in rows), "expected a complete sprite row"
    assert any(all((r, c) in occupied for r in rows) for c in cols), "expected a complete sprite column"

    sprite_mask = sprites[0][2]
    h, w = map(int, sprite_mask.shape)
    fill_patch = np.full((h, w), bg, dtype=int)
    fill_patch[sprite_mask] = Color.TEAL

    missing = [(r, c) for r in rows for c in cols if (r, c) not in occupied]
    assert missing, "expected at least one empty lattice slot"

    out = g.copy()
    for r, c in missing:
        target = out[r : r + h, c : c + w]
        assert target.shape == (h, w), "empty slot must fit in the grid"
        assert np.all(target == bg), "empty lattice slot must be background"
        paste(out, fill_patch, r, c, transparent=bg)

    return out

# --------------- generator (o5x) ---------------

# Sprite shape, colors, and reusable patch templates.
SPRITE_SIZE, SPRITE_CENTER = 3, 1
BACKGROUND_COLOR, ORIGINAL_COLOR, FILL_COLOR = Color.BLACK, Color.BLUE, Color.TEAL
SPRITE_MASK = np.ones((SPRITE_SIZE, SPRITE_SIZE), dtype=bool)
SPRITE_MASK[SPRITE_CENTER, SPRITE_CENTER] = False
BLUE_PATCH = np.where(SPRITE_MASK, ORIGINAL_COLOR, BACKGROUND_COLOR)
TEAL_PATCH = np.where(SPRITE_MASK, FILL_COLOR, BACKGROUND_COLOR)
ERASE_PATCH = np.zeros((SPRITE_SIZE, SPRITE_SIZE), dtype=int)

# Counts use randint upper-exclusive bounds; spacing and canvas bounds are inclusive.
ROW_COUNT_BOUNDS, COL_COUNT_BOUNDS = (3, 5), (3, 6)
ROW_GAP_BOUNDS, COL_GAP_BOUNDS = (4, 7), (4, 6)
MARGIN_BOUNDS, HEIGHT_BOUNDS, WIDTH_BOUNDS = (1, 3), (19, 30), (16, 30)
MAX_GENERATION_ATTEMPTS = 300


def _gen_axis(count, gap_bounds, size_bounds):
    """Sample one valid axis size and its top-left sprite positions uniformly."""
    gap_values = list(range(gap_bounds[0], gap_bounds[1] + 1))
    options = []
    for gap_idxs in np.ndindex(*([len(gap_values)] * (count - 1))):
        gaps = [gap_values[i] for i in gap_idxs]
        if len(set(gaps)) == 1:
            continue
        span = sum(gaps) + SPRITE_SIZE
        for margin0 in range(MARGIN_BOUNDS[0], MARGIN_BOUNDS[1] + 1):
            for margin1 in range(MARGIN_BOUNDS[0], MARGIN_BOUNDS[1] + 1):
                size = margin0 + span + margin1
                if size_bounds[0] <= size <= size_bounds[1]:
                    positions = [margin0]
                    for gap in gaps:
                        positions.append(positions[-1] + gap)
                    options.append((size, positions))
    return None if not options else options[int(np.random.randint(len(options)))]


def _sample_lattice():
    """Sample canvas dimensions and row/column lattice positions."""
    n_rows = int(np.random.randint(*ROW_COUNT_BOUNDS))
    n_cols = int(np.random.randint(*COL_COUNT_BOUNDS))
    row_axis = _gen_axis(n_rows, ROW_GAP_BOUNDS, HEIGHT_BOUNDS)
    col_axis = _gen_axis(n_cols, COL_GAP_BOUNDS, WIDTH_BOUNDS)
    if row_axis is None or col_axis is None:
        return None
    height, rows = row_axis
    width, cols = col_axis
    return height, width, rows, cols


def _build_solved_grid(height, width, rows, cols):
    """Return the full output grid and the L-complement slots erased for input."""
    out = np.zeros((height, width), dtype=int)
    for r in rows:
        for c in cols:
            paste(out, BLUE_PATCH, r, c, transparent=BACKGROUND_COLOR)

    complete_row = rows[0] if int(np.random.randint(2)) == 0 else rows[-1]
    complete_col = cols[0]
    missing = []
    for r in rows:
        for c in cols:
            if r == complete_row or c == complete_col:
                continue
            missing.append((r, c))
            paste(out, TEAL_PATCH, r, c, transparent=BACKGROUND_COLOR)
    return out, missing


def _gen_attempt() -> "np.ndarray | None":
    lattice = _sample_lattice()
    if lattice is None:
        return None
    out, missing = _build_solved_grid(*lattice)
    if not missing:
        return None
    inp = out.copy()
    for r, c in missing:
        paste(inp, ERASE_PATCH, r, c)
    if np.array_equal(main(inp), out):
        return inp
    return None


def generate_input() -> "np.ndarray":
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 5b526a93 input")
