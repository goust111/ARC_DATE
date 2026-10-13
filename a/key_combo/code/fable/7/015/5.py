"""ARC task: holes set a column rhythm and object position sets its phase."""

from __future__ import annotations

from collections import deque

import numpy as np


BLACK = 0
YELLOW = 4
PINK = 6
TEAL = 8
_DIRS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _components(mask: np.ndarray) -> list[np.ndarray]:
    """Return 4-connected True components as arrays of global (row, col)."""
    mask = np.asarray(mask, dtype=bool)
    seen = np.zeros(mask.shape, dtype=bool)
    result: list[np.ndarray] = []
    height, width = mask.shape

    for start_r, start_c in np.argwhere(mask):
        sr, sc = int(start_r), int(start_c)
        if seen[sr, sc]:
            continue
        seen[sr, sc] = True
        queue = deque([(sr, sc)])
        coords: list[tuple[int, int]] = []
        while queue:
            r, c = queue.popleft()
            coords.append((r, c))
            for dr, dc in _DIRS4:
                nr, nc = r + dr, c + dc
                if 0 <= nr < height and 0 <= nc < width and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    queue.append((nr, nc))
        result.append(np.asarray(coords, dtype=int))
    return result


def _hole_count(object_coords: np.ndarray) -> int:
    """Count 4-connected background regions enclosed by one object."""
    assert object_coords.ndim == 2 and object_coords.shape[1] == 2 and len(object_coords) > 0
    r0, c0 = object_coords.min(axis=0)
    r1, c1 = object_coords.max(axis=0)
    local = np.zeros((int(r1 - r0 + 1), int(c1 - c0 + 1)), dtype=bool)
    local[object_coords[:, 0] - r0, object_coords[:, 1] - c0] = True

    # Padding makes the unique exterior component explicit even when the object
    # touches its own bbox on every side.
    background = ~np.pad(local, 1, constant_values=False)
    height, width = background.shape
    holes = 0
    for coords in _components(background):
        rows, cols = coords[:, 0], coords[:, 1]
        touches_edge = np.any((rows == 0) | (rows == height - 1) | (cols == 0) | (cols == width - 1))
        if not touches_edge:
            holes += 1
    return holes


def main(grid: np.ndarray) -> np.ndarray:
    """Use a teal object's holes and left edge to rhythmically recolor yellow columns pink.

    Input structure:
    - A black background separates one 4-connected teal object from a horizontal
      yellow strip made of alternating black/yellow rows and solid yellow rows.
    - The teal object encloses one or more 4-connected black holes and may have
      an irregular outline; all coordinates use the full input grid.

    Rule:
    - Let n be the number of black holes enclosed by the teal object and c be
      the global column of its leftmost pixel.
    - Preserve every teal and black pixel. Recolor exactly those yellow pixels
      in global columns j satisfying j modulo n equals c modulo n to pink.

    Key insight:
    - The remote object's topology supplies the column period, while its global
      horizontal position independently supplies the phase of the yellow highlights.
    """
    g = np.asarray(grid)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"
    assert set(map(int, np.unique(g))) <= {BLACK, YELLOW, TEAL}, "unexpected input color"

    teal_components = _components(g == TEAL)
    assert len(teal_components) == 1, "expected exactly one 4-connected teal object"
    teal = teal_components[0]
    period = _hole_count(teal)
    assert period > 0, "the teal object must enclose at least one hole"
    phase_column = int(teal[:, 1].min())
    assert np.any(g == YELLOW), "expected a separate yellow strip"

    out = g.copy()
    columns = np.arange(g.shape[1])
    selected_columns = columns % period == phase_column % period
    recolor = (g == YELLOW) & selected_columns[np.newaxis, :]
    out[recolor] = PINK
    return out


def _make_teal_object(rng: np.random.Generator, holes: int) -> np.ndarray:
    """Construct a varied connected silhouette with exactly `holes` cavities."""
    height = int(rng.integers(5, 9))
    hole_widths = [int(rng.integers(1, 3)) for _ in range(holes)]
    wall_widths = [int(rng.integers(1, 3)) for _ in range(max(0, holes - 1))]
    width = 2 + sum(hole_widths) + sum(wall_widths)
    sprite = np.full((height, width), TEAL, dtype=int)

    col = 1
    for i, hole_width in enumerate(hole_widths):
        hole_height = int(rng.integers(1, height - 1))
        hole_row = int(rng.integers(1, height - hole_height))
        sprite[hole_row : hole_row + hole_height, col : col + hole_width] = BLACK
        if i < len(wall_widths):
            col += hole_width + wall_widths[i]

    # Independent one-cell corner cuts vary the outer contour without opening a
    # 4-connected route from an interior cavity to the exterior.
    corners = ((0, 0), (0, width - 1), (height - 1, 0), (height - 1, width - 1))
    cut = rng.random(4) < 0.45
    if np.all(cut):
        cut[int(rng.integers(0, 4))] = False
    for do_cut, (r, c) in zip(cut, corners):
        if do_cut:
            sprite[r, c] = BLACK
    return sprite


def _reference_holes(sprite: np.ndarray) -> int:
    """Generator-side flood fill, deliberately independent of `_hole_count`."""
    height, width = sprite.shape
    exterior = np.zeros((height, width), dtype=bool)
    queue: deque[tuple[int, int]] = deque()
    for r in range(height):
        for c in (0, width - 1):
            if sprite[r, c] == BLACK and not exterior[r, c]:
                exterior[r, c] = True
                queue.append((r, c))
    for c in range(width):
        for r in (0, height - 1):
            if sprite[r, c] == BLACK and not exterior[r, c]:
                exterior[r, c] = True
                queue.append((r, c))
    while queue:
        r, c = queue.popleft()
        for dr, dc in _DIRS4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < height and 0 <= nc < width and sprite[nr, nc] == BLACK and not exterior[nr, nc]:
                exterior[nr, nc] = True
                queue.append((nr, nc))
    return len(_components((sprite == BLACK) & ~exterior))


def _semantic_check(
    grid: np.ndarray,
    expected_holes: int,
    object_left: int,
    band_rows: slice,
) -> bool:
    """Reject degenerate scenes and shortcuts before exposing an example."""
    if grid.ndim != 2 or max(grid.shape) > 30 or set(map(int, np.unique(grid))) != {BLACK, YELLOW, TEAL}:
        return False
    teal_components = _components(grid == TEAL)
    if len(teal_components) != 1:
        return False
    teal = teal_components[0]
    if int(teal[:, 1].min()) != object_left or _hole_count(teal) != expected_holes:
        return False

    band = grid[band_rows]
    full_rows = np.all(band == YELLOW, axis=1)
    mixed_rows = np.any(band == YELLOW, axis=1) & np.any(band == BLACK, axis=1)
    if not np.any(full_rows) or not np.any(mixed_rows):
        return False
    if np.any(grid[: band_rows.start] == YELLOW) or np.any(grid[band_rows.stop :] == YELLOW):
        return False

    expected = grid.copy()
    cols = np.arange(grid.shape[1])
    selected = cols % expected_holes == object_left % expected_holes
    expected[(grid == YELLOW) & selected[np.newaxis, :]] = PINK
    try:
        actual = main(grid)
    except (AssertionError, ValueError):
        return False
    if not np.array_equal(actual, expected) or np.array_equal(actual, grid):
        return False
    if not np.array_equal(actual[grid == TEAL], grid[grid == TEAL]):
        return False
    if np.any(actual[grid == BLACK] != BLACK):
        return False

    # A neighboring phase and every other supported hole count must predict a
    # visibly different result. Thus neither source clue is decorative.
    shifted = grid.copy()
    shifted_sel = cols % expected_holes == (object_left + 1) % expected_holes
    shifted[(grid == YELLOW) & shifted_sel[np.newaxis, :]] = PINK
    if np.array_equal(shifted, expected):
        return False
    for other_period in (2, 3, 4):
        if other_period == expected_holes:
            continue
        alternative = grid.copy()
        other_sel = cols % other_period == object_left % other_period
        alternative[(grid == YELLOW) & other_sel[np.newaxis, :]] = PINK
        if np.array_equal(alternative, expected):
            return False
    return True


def _generate_attempt(rng: np.random.Generator) -> np.ndarray | None:
    holes = int(rng.integers(2, 5))
    sprite = _make_teal_object(rng, holes)
    if _reference_holes(sprite) != holes:
        return None

    object_h, object_w = sprite.shape
    band_h = int(rng.integers(3, 7))
    gap = int(rng.integers(2, 5))
    margin_a = int(rng.integers(1, 4))
    margin_b = int(rng.integers(1, 4))
    height = margin_a + object_h + gap + band_h + margin_b
    min_width = max(14, object_w + 4)
    width = int(rng.integers(min_width, min(27, min_width + 8) + 1))
    grid = np.zeros((height, width), dtype=int)

    object_first = bool(rng.integers(0, 2))
    if object_first:
        object_row = margin_a
        band_start = object_row + object_h + gap
    else:
        band_start = margin_a
        object_row = band_start + band_h + gap
    object_col = int(rng.integers(1, width - object_w))
    target_residue = int(rng.integers(0, holes))
    candidates = [c for c in range(1, width - object_w) if c % holes == target_residue]
    if candidates:
        object_col = int(candidates[int(rng.integers(0, len(candidates)))])
    grid[object_row : object_row + object_h, object_col : object_col + object_w] = sprite

    row_kinds = ["full", "alt0"]
    row_kinds.extend("full" if rng.random() < 0.45 else ("alt0" if rng.random() < 0.5 else "alt1") for _ in range(band_h - 2))
    rng.shuffle(row_kinds)
    for offset, kind in enumerate(row_kinds):
        row = band_start + offset
        if kind == "full":
            grid[row, :] = YELLOW
        else:
            phase = 0 if kind == "alt0" else 1
            grid[row, phase::2] = YELLOW

    band_rows = slice(band_start, band_start + band_h)
    if not _semantic_check(grid, holes, object_col, band_rows):
        return None
    return grid


def generate_input() -> np.ndarray:
    """Generate a diverse, non-degenerate puzzle input of at most 30 by 30."""
    rng = np.random.default_rng()
    for _ in range(200):
        result = _generate_attempt(rng)
        if result is not None:
            return result
    raise RuntimeError("failed to generate a valid hole-rhythm input")
