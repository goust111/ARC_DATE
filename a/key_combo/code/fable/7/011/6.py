"""ARC task 6: reveal the shared blind spots of two diagonal-pattern fragments."""

from __future__ import annotations

import random

import numpy as np


BLACK = 0
SEPARATOR_COLOR = 5
PATTERN_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)


def _split_panels(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    """Return the equally sized panels and the structurally marked separator."""
    h, w = grid.shape
    assert w % 2 == 1, "expected two equal panels separated by one column"
    sep_col = w // 2
    separator = grid[:, sep_col]
    assert np.all(separator == separator[0]), "separator must be a solid bar"
    sep_color = int(separator[0])
    assert sep_color != BLACK, "separator must be visibly distinct from holes"
    left = grid[:, :sep_col]
    right = grid[:, sep_col + 1 :]
    assert left.shape == right.shape and left.shape[0] == h
    return left, right, sep_color


def _infer_cycle(
    left: np.ndarray, right: np.ndarray, sep_color: int
) -> tuple[int, dict[int, int]]:
    """Infer direction and residue colors from all visible cells in both copies."""
    visible_colors = set(map(int, left[left != BLACK])) | set(
        map(int, right[right != BLACK])
    )
    assert sep_color not in visible_colors, "separator color must not occur in the tile"
    period = len(visible_colors)
    assert period >= 2, "a diagonal cycle needs at least two colors"

    candidates: list[tuple[int, dict[int, int]]] = []
    for direction in (1, -1):
        residue_to_color: dict[int, int] = {}
        color_to_residue: dict[int, int] = {}
        valid = True
        for panel in (left, right):
            for r, c in np.argwhere(panel != BLACK):
                color = int(panel[r, c])
                residue = (int(r) + direction * int(c)) % period
                if residue in residue_to_color and residue_to_color[residue] != color:
                    valid = False
                    break
                if color in color_to_residue and color_to_residue[color] != residue:
                    valid = False
                    break
                residue_to_color[residue] = color
                color_to_residue[color] = residue
            if not valid:
                break
        if valid and len(residue_to_color) == period and set(residue_to_color.values()) == visible_colors:
            candidates.append((direction, residue_to_color))

    assert candidates, "the fragments do not determine a complete diagonal cycle"
    if len(candidates) > 1:
        # An ambiguity is harmless only when every required blind cell gets the
        # same color under all consistent interpretations (for example period 2).
        blind = np.argwhere((left == BLACK) & (right == BLACK))
        predictions = []
        for direction, mapping in candidates:
            predictions.append(
                tuple(mapping[(int(r) + direction * int(c)) % period] for r, c in blind)
            )
        assert all(p == predictions[0] for p in predictions[1:]), (
            "ambiguous diagonal interpretations predict different outputs"
        )
    return candidates[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reveal only the jointly missing cells of two damaged diagonal-cycle copies.

    Input structure:
    - A solid vertical separator divides the grid into equal left and right panels.
    - The panels share one coordinate frame and are damaged copies of the same
      non-black diagonal periodic tile; black cells are holes.
    - The separator has a color not used by the tile, and the two panels jointly
      expose every residue color of the cycle.

    Rule:
    - Combine all visible cells in both panels to infer whether equal-color
      diagonals follow (row + column) or (row - column), and recover the color
      assigned to every residue modulo the number of tile colors.
    - Produce one panel-sized black grid.  At a coordinate that is black in both
      panels, paint its inferred cycle color; leave every coordinate visible in
      at least one panel black.

    Key insight:
    - The output is the intersection of the two blind-spot masks, colored from
      the jointly reconstructed diagonal cycle rather than copied from either side.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    left, right, sep_color = _split_panels(g)
    direction, residue_to_color = _infer_cycle(left, right, sep_color)
    period = len(residue_to_color)

    both_visible = (left != BLACK) & (right != BLACK)
    assert np.all(left[both_visible] == right[both_visible]), (
        "visible copies must agree at corresponding coordinates"
    )

    out = np.zeros(left.shape, dtype=int)
    for r, c in np.argwhere((left == BLACK) & (right == BLACK)):
        residue = (int(r) + direction * int(c)) % period
        out[r, c] = residue_to_color[residue]
    return out


def _complete_tile(
    height: int, width: int, direction: int, palette: list[int]
) -> np.ndarray:
    period = len(palette)
    rows = np.arange(height)[:, None]
    cols = np.arange(width)[None, :]
    residues = (rows + direction * cols) % period
    return np.asarray(palette, dtype=int)[residues]


def _opposite_direction_is_rejected(
    left: np.ndarray, right: np.ndarray, period: int, direction: int
) -> bool:
    """Check that visible evidence distinguishes the two diagonal orientations."""
    mapping: dict[int, int] = {}
    opposite = -direction
    for panel in (left, right):
        for r, c in np.argwhere(panel != BLACK):
            residue = (int(r) + opposite * int(c)) % period
            color = int(panel[r, c])
            if residue in mapping and mapping[residue] != color:
                return True
            mapping[residue] = color
    return False


def _semantic_checks(
    grid: np.ndarray,
    complete: np.ndarray,
    left_missing_residue: int,
    right_missing_residue: int,
    direction: int,
) -> bool:
    """Independent oracle plus anti-shortcut and non-degeneracy checks."""
    left, right, _ = _split_panels(grid)
    period = len(set(map(int, complete.flat)))
    rows = np.arange(left.shape[0])[:, None]
    cols = np.arange(left.shape[1])[None, :]
    residues = (rows + direction * cols) % period
    blind = (left == BLACK) & (right == BLACK)
    expected = np.zeros_like(left)
    expected[blind] = complete[blind]

    if not np.array_equal(main(grid), expected):
        return False
    # Each fragment hides a whole, different cycle color, while the other side
    # reveals it.  Thus neither side alone contains the complete mapping.
    if left_missing_residue == right_missing_residue:
        return False
    if np.any(left[residues == left_missing_residue] != BLACK):
        return False
    if np.any(right[residues == right_missing_residue] != BLACK):
        return False
    if not np.any(right[residues == left_missing_residue] != BLACK):
        return False
    if not np.any(left[residues == right_missing_residue] != BLACK):
        return False
    # Cover common holes and both asymmetric cases, ruling out copying either
    # panel, union/intersection recoloring, or filling every hole.
    if np.count_nonzero(blind) < max(3, period // 2):
        return False
    if len(set(map(int, complete[blind]))) < 2:
        return False
    if not np.any((left == BLACK) & (right != BLACK)):
        return False
    if not np.any((left != BLACK) & (right == BLACK)):
        return False
    if not np.any((left != BLACK) & (right != BLACK)):
        return False
    if not _opposite_direction_is_rejected(left, right, period, direction):
        return False
    return True


def generate_input() -> np.ndarray:
    """Generate a varied, semantically checked pair of diagonal fragments."""
    for _ in range(300):
        height = random.randint(8, 15)
        panel_width = random.randint(8, 14)
        period = random.randint(3, min(6, height, panel_width))
        direction = random.choice((-1, 1))
        palette = random.sample(PATTERN_COLORS, period)
        complete = _complete_tile(height, panel_width, direction, palette)

        left_missing, right_missing = random.sample(range(period), 2)
        rows = np.arange(height)[:, None]
        cols = np.arange(panel_width)[None, :]
        residues = (rows + direction * cols) % period
        left_holes = residues == left_missing
        right_holes = residues == right_missing

        # Moderate independent erosion supplies left-only, right-only and shared
        # blind spots without obscuring the global diagonal bands.
        erosion_rate = random.uniform(0.10, 0.22)
        left_holes |= np.random.random((height, panel_width)) < erosion_rate
        right_holes |= np.random.random((height, panel_width)) < erosion_rate

        # Force shared blind cells in two residues, including one residue whose
        # color is unavailable from the left fragment alone.
        forced_residues = {left_missing, random.choice([x for x in range(period) if x != left_missing])}
        for residue in forced_residues:
            cells = np.argwhere(residues == residue)
            r, c = cells[random.randrange(len(cells))]
            left_holes[r, c] = True
            right_holes[r, c] = True

        left = complete.copy()
        right = complete.copy()
        left[left_holes] = BLACK
        right[right_holes] = BLACK
        grid = np.concatenate(
            (left, np.full((height, 1), SEPARATOR_COLOR, dtype=int), right), axis=1
        )

        if max(grid.shape) <= 30 and _semantic_checks(
            grid, complete, left_missing, right_missing, direction
        ):
            return grid
    raise RuntimeError("failed to generate a non-degenerate double-fragment input")
