"""ARC task: extend a periodic wallpaper through a one-gap rectangular frame."""

from __future__ import annotations

import numpy as np


_SIDES = ("top", "bottom", "left", "right")
_RNG = np.random.default_rng()
_side_cycle: list[str] = []


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    """Return the inclusive (r0, c0, r1, c1) bbox of a nonempty mask."""
    coords = np.argwhere(mask)
    assert coords.size, "expected a nonempty foreground"
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _infer_period(pattern: np.ndarray, axis: int) -> int:
    """Find the unique fundamental visible period, requiring two repetitions."""
    length = int(pattern.shape[axis])
    for period in range(1, length // 2 + 1):
        if axis == 0:
            agrees = np.array_equal(pattern[:-period, :], pattern[period:, :])
        else:
            agrees = np.array_equal(pattern[:, :-period], pattern[:, period:])
        if agrees:
            return period
    raise AssertionError("wallpaper must visibly repeat at least twice on each axis")


def _analyze_frame(
    grid: np.ndarray,
) -> tuple[int, tuple[int, int, int, int], tuple[int, int], str, np.ndarray, int, int]:
    """Locate the frame, its sole gap, and the wallpaper's fundamental tile."""
    height, width = map(int, grid.shape)
    corners = [int(grid[0, 0]), int(grid[0, -1]), int(grid[-1, 0]), int(grid[-1, -1])]
    assert len(set(corners)) == 1, "the four grid corners must be background"
    background = corners[0]

    r0, c0, r1, c1 = _bbox(grid != background)
    assert 0 < r0 < r1 < height - 1 and 0 < c0 < c1 < width - 1
    assert r1 - r0 >= 5 and c1 - c0 >= 5, "frame interior is too small"

    border = {
        (r, c)
        for r in range(r0, r1 + 1)
        for c in range(c0, c1 + 1)
        if r in (r0, r1) or c in (c0, c1)
    }
    gaps = [(r, c) for r, c in border if int(grid[r, c]) == background]
    assert len(gaps) == 1, "expected exactly one missing frame pixel"
    gap_r, gap_c = gaps[0]

    frame_values = {int(grid[r, c]) for r, c in border if (r, c) != (gap_r, gap_c)}
    assert len(frame_values) == 1, "the remaining frame must be monochrome"
    assert all(int(grid[r, c]) != background for r, c in border if (r, c) != (gap_r, gap_c))

    if gap_r == r0 and c0 < gap_c < c1:
        side = "top"
    elif gap_r == r1 and c0 < gap_c < c1:
        side = "bottom"
    elif gap_c == c0 and r0 < gap_r < r1:
        side = "left"
    elif gap_c == c1 and r0 < gap_r < r1:
        side = "right"
    else:
        raise AssertionError("the gap must be a non-corner pixel on exactly one side")

    interior = grid[r0 + 1 : r1, c0 + 1 : c1]
    assert interior.size and np.all(interior != background), "wallpaper must completely fill the interior"
    assert np.unique(interior).size >= 2, "wallpaper must be multicolored"
    row_period = _infer_period(interior, axis=0)
    col_period = _infer_period(interior, axis=1)
    assert interior.shape[0] >= 2 * row_period and interior.shape[1] >= 2 * col_period

    tile = interior[:row_period, :col_period].copy()
    rr = np.arange(interior.shape[0]) % row_period
    cc = np.arange(interior.shape[1]) % col_period
    assert np.array_equal(interior, tile[rr[:, None], cc[None, :]])
    return background, (r0, c0, r1, c1), (gap_r, gap_c), side, tile, row_period, col_period


def _tile_value(
    tile: np.ndarray,
    row_period: int,
    col_period: int,
    base_r: int,
    base_c: int,
    r: int,
    c: int,
) -> int:
    return int(tile[(r - base_r) % row_period, (c - base_c) % col_period])


def main(grid: np.ndarray) -> np.ndarray:
    """Extend the interior wallpaper as a full band through the frame's unique gap.

    Input structure:
    - A uniform background surrounds one one-cell-thick monochrome rectangular frame.
    - Exactly one non-corner frame pixel is missing, and the frame interior is fully
      occupied by a multicolor wallpaper with two visible repeats on each axis.

    Rule:
    - Use the gap side's outward normal as the extension direction and fill the
      entire exterior band aligned with the interior, plus the gap pixel itself.
    - Continue the wallpaper's fundamental row and column periods in frame-relative
      coordinates, counting the one-cell frame thickness as a hidden pattern line.
      Preserve every other frame pixel and all background outside the selected band.

    Key insight:
    - The gap points a whole-band spill rather than a ray, while the frame overlays a
      continuous wallpaper, so the first exterior line is two phase steps beyond the
      last interior line rather than its immediate successor.
    """
    g = np.asarray(grid)
    assert g.ndim == 2, "expected a 2D grid"
    assert g.shape[0] <= 30 and g.shape[1] <= 30, "ARC grids are at most 30x30"

    _, (r0, c0, r1, c1), gap, side, tile, row_period, col_period = _analyze_frame(g)
    base_r, base_c = r0 + 1, c0 + 1
    out = g.copy()

    if side == "top":
        targets = [(r, c) for r in range(0, r0) for c in range(c0 + 1, c1)]
    elif side == "bottom":
        targets = [(r, c) for r in range(r1 + 1, g.shape[0]) for c in range(c0 + 1, c1)]
    elif side == "left":
        targets = [(r, c) for r in range(r0 + 1, r1) for c in range(0, c0)]
    else:
        targets = [(r, c) for r in range(r0 + 1, r1) for c in range(c1 + 1, g.shape[1])]

    targets.append(gap)
    for r, c in targets:
        out[r, c] = _tile_value(tile, row_period, col_period, base_r, base_c, r, c)
    return out


def _next_side() -> str:
    """Cycle through all directions in random order before repeating a direction."""
    global _side_cycle
    if not _side_cycle:
        _side_cycle = [str(x) for x in _RNG.permutation(_SIDES)]
    return _side_cycle.pop()


def _fundamental_tile(row_period: int, col_period: int, colors: np.ndarray) -> np.ndarray:
    """Sample a multicolor tile whose two requested periods are both exact."""
    for _ in range(500):
        tile = _RNG.choice(colors, size=(row_period, col_period), replace=True).astype(int)
        if np.unique(tile).size < 2:
            continue
        repeated = np.tile(tile, (2, 2))
        if _infer_period(repeated, 0) == row_period and _infer_period(repeated, 1) == col_period:
            return tile
    raise RuntimeError("failed to sample a nondegenerate wallpaper tile")


def _expected_from_spec(
    inp: np.ndarray,
    box: tuple[int, int, int, int],
    gap: tuple[int, int],
    side: str,
    tile: np.ndarray,
) -> tuple[np.ndarray, set[tuple[int, int]]]:
    """Render the rule directly from generator metadata, independently of inference."""
    r0, c0, r1, c1 = box
    row_period, col_period = map(int, tile.shape)
    base_r, base_c = r0 + 1, c0 + 1
    if side == "top":
        cells = {(r, c) for r in range(r0) for c in range(c0 + 1, c1)}
    elif side == "bottom":
        cells = {(r, c) for r in range(r1 + 1, inp.shape[0]) for c in range(c0 + 1, c1)}
    elif side == "left":
        cells = {(r, c) for r in range(r0 + 1, r1) for c in range(c0)}
    else:
        cells = {(r, c) for r in range(r0 + 1, r1) for c in range(c1 + 1, inp.shape[1])}
    cells.add(gap)

    expected = inp.copy()
    for r, c in cells:
        expected[r, c] = _tile_value(tile, row_period, col_period, base_r, base_c, r, c)
    return expected, cells


def _semantic_checks(
    inp: np.ndarray,
    expected: np.ndarray,
    cells: set[tuple[int, int]],
    box: tuple[int, int, int, int],
    gap: tuple[int, int],
    side: str,
    tile: np.ndarray,
    background: int,
) -> None:
    """Reject ray, wrong-phase, ambiguous-period, and degenerate generated cases."""
    r0, c0, r1, c1 = box
    rp, cp = map(int, tile.shape)
    interior = inp[r0 + 1 : r1, c0 + 1 : c1]
    assert _infer_period(interior, 0) == rp and _infer_period(interior, 1) == cp
    assert np.unique(interior).size >= 2 and np.all(interior != background)

    changed = {tuple(map(int, rc)) for rc in np.argwhere(inp != expected)}
    assert changed == cells, "every and only the full-band cells must visibly change"
    assert gap in changed
    if side in ("top", "bottom"):
        assert len({c for _, c in changed if c0 < c < c1}) == c1 - c0 - 1
        exterior = [(r, c) for r, c in cells if r != gap[0]]
    else:
        assert len({r for r, _ in changed if r0 < r < r1}) == r1 - r0 - 1
        exterior = [(r, c) for r, c in cells if c != gap[1]]

    # A shortcut that joins the first exterior line directly to the adjacent
    # interior line ignores the hidden frame line and shifts this whole region.
    base_r, base_c = r0 + 1, c0 + 1
    phase_shift = {"top": (1, 0), "bottom": (-1, 0), "left": (0, 1), "right": (0, -1)}[side]
    dr, dc = phase_shift
    wrong_phase_differs = any(
        int(expected[r, c])
        != _tile_value(tile, rp, cp, base_r, base_c, r + dr, c + dc)
        for r, c in exterior
    )
    assert wrong_phase_differs, "frame-thickness phase must affect visible output"

    solved = main(inp)
    assert np.array_equal(solved, expected), "solver disagrees with the independent specification"


def generate_input() -> np.ndarray:
    """Generate a diverse, validated one-gap-frame wallpaper input."""
    for _ in range(300):
        side = _next_side()
        row_period = int(_RNG.integers(2, 5))
        col_period = int(_RNG.integers(2, 5))
        interior_h = 2 * row_period + int(_RNG.integers(0, row_period + 1))
        interior_w = 2 * col_period + int(_RNG.integers(0, col_period + 1))

        margins = {name: int(_RNG.integers(3, 6)) for name in _SIDES}
        required = row_period + 1 if side in ("top", "bottom") else col_period + 1
        margins[side] = int(_RNG.integers(required, 6)) if required < 6 else required

        frame_h, frame_w = interior_h + 2, interior_w + 2
        height = margins["top"] + frame_h + margins["bottom"]
        width = margins["left"] + frame_w + margins["right"]
        if height > 30 or width > 30:
            continue

        color_count = int(_RNG.integers(2, 4))
        chosen = _RNG.choice(np.arange(10), size=color_count + 2, replace=False).astype(int)
        background, frame_color = map(int, chosen[:2])
        wallpaper_colors = chosen[2:]
        tile = _fundamental_tile(row_period, col_period, wallpaper_colors)

        r0, c0 = margins["top"], margins["left"]
        r1, c1 = r0 + frame_h - 1, c0 + frame_w - 1
        box = (r0, c0, r1, c1)
        inp = np.full((height, width), background, dtype=int)
        inp[r0, c0 : c1 + 1] = frame_color
        inp[r1, c0 : c1 + 1] = frame_color
        inp[r0 : r1 + 1, c0] = frame_color
        inp[r0 : r1 + 1, c1] = frame_color

        rr = np.arange(interior_h) % row_period
        cc = np.arange(interior_w) % col_period
        inp[r0 + 1 : r1, c0 + 1 : c1] = tile[rr[:, None], cc[None, :]]

        if side in ("top", "bottom"):
            gap = (r0 if side == "top" else r1, int(_RNG.integers(c0 + 1, c1)))
        else:
            gap = (int(_RNG.integers(r0 + 1, r1)), c0 if side == "left" else c1)
        inp[gap] = background

        expected, cells = _expected_from_spec(inp, box, gap, side, tile)
        try:
            _semantic_checks(inp, expected, cells, box, gap, side, tile, background)
        except (AssertionError, ValueError):
            continue
        return inp
    raise RuntimeError("failed to generate a valid wallpaper spill input")
