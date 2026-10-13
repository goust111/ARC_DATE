from __future__ import annotations

import numpy as np


BACKGROUND = 0
MARKER = 6
_MAX_GENERATION_ATTEMPTS = 500
_LAST_META: dict[str, object] = {}


def _visible_radial_model(
    grid: np.ndarray, center: tuple[int, int]
) -> tuple[int, dict[int, int]] | None:
    """Return the shortest fully observed radial period for one center."""
    cr, cc = center
    radial_colors: dict[int, int] = {}
    visible = np.argwhere((grid != BACKGROUND) & (grid != MARKER))
    for r, c in visible:
        radius = max(abs(int(r) - cr), abs(int(c) - cc))
        if radius == 0:
            return None
        color = int(grid[r, c])
        previous = radial_colors.get(radius)
        if previous is not None and previous != color:
            return None
        radial_colors[radius] = color

    if len(set(radial_colors.values())) < 2:
        return None

    max_radius = max(radial_colors, default=0)
    for period in range(1, max_radius + 1):
        phase_colors: dict[int, int] = {}
        valid = True
        for radius, color in radial_colors.items():
            phase = radius % period
            previous = phase_colors.get(phase)
            if previous is not None and previous != color:
                valid = False
                break
            phase_colors[phase] = color
        # Every phase must have a visible witness, so marker colors are unique.
        if valid and len(phase_colors) == period:
            return period, phase_colors
    return None


def _infer_unique_model(
    grid: np.ndarray,
) -> tuple[tuple[int, int], int, dict[int, int]]:
    """Infer the sole black center and its shortest radial color cycle."""
    models: list[tuple[tuple[int, int], int, dict[int, int]]] = []
    for cr, cc in np.argwhere(grid == BACKGROUND):
        center = (int(cr), int(cc))
        model = _visible_radial_model(grid, center)
        if model is not None:
            period, phase_colors = model
            models.append((center, period, phase_colors))
    assert len(models) == 1, "visible arcs must uniquely determine the hidden center"
    return models[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete all magenta gaps from the radial color cycle of hidden concentric square rings.

    Input structure:
    - A black grid contains visible, monochromatic arc segments of concentric
      Chebyshev-distance square rings and magenta pixels marking covered cells.
    - The common center is black and never drawn; the visible arcs uniquely locate it.
    - Ring colors follow a short repeating radial sequence, with every cycle phase
      witnessed by visible arcs even though one entire ring has no visible color.

    Rule:
    - Find the unique black center for which each visible radius has one color, then
      infer the shortest radial period consistent with all visible ring colors.
    - Replace each magenta pixel by the color indexed by its Chebyshev radius modulo
      that period. Leave every non-magenta cell, including other black ring cells,
      unchanged.

    Key insight:
    - The repetition is one-dimensional in Chebyshev radius around an invisible
      center, so the wholly covered ring must be recovered from another cycle turn.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert np.any(g == MARKER), "expected at least one magenta missing-value marker"

    (cr, cc), period, phase_colors = _infer_unique_model(g)
    out = g.copy()
    for r, c in np.argwhere(g == MARKER):
        radius = max(abs(int(r) - cr), abs(int(c) - cc))
        phase = radius % period
        assert phase in phase_colors
        out[r, c] = phase_colors[phase]
    return out


def _ring_coords(center: tuple[int, int], radius: int) -> list[tuple[int, int]]:
    """Clockwise coordinates of a complete Chebyshev ring."""
    cr, cc = center
    d = int(radius)
    top, bottom = cr - d, cr + d
    left, right = cc - d, cc + d
    coords = [(top, c) for c in range(left, right + 1)]
    coords += [(r, right) for r in range(top + 1, bottom + 1)]
    coords += [(bottom, c) for c in range(right - 1, left - 1, -1)]
    coords += [(r, left) for r in range(bottom - 1, top, -1)]
    assert len(coords) == 8 * d and len(set(coords)) == 8 * d
    return coords


def _sample_arc_cells(
    ring: list[tuple[int, int]], radius: int, rng: np.random.Generator
) -> set[tuple[int, int]]:
    """Choose two or three separated cyclic arc segments on a square ring."""
    n = len(ring)
    selected: set[int] = set()
    arc_count = int(rng.integers(2, 4))
    for _ in range(arc_count):
        start = int(rng.integers(0, n))
        length = int(rng.integers(max(2, radius), 2 * radius + 2))
        selected.update((start + k) % n for k in range(length))

    # Preserve visible gaps: omitted black cells and magenta gaps remain legible.
    limit = n - max(2, radius // 2)
    if len(selected) > limit:
        selected = set(rng.choice(sorted(selected), size=limit, replace=False).tolist())
    return {ring[i] for i in selected}


def _oracle_output(
    grid: np.ndarray,
    center: tuple[int, int],
    period: int,
    phase_colors: dict[int, int],
) -> np.ndarray:
    out = grid.copy()
    cr, cc = center
    for r, c in np.argwhere(grid == MARKER):
        radius = max(abs(int(r) - cr), abs(int(c) - cc))
        out[r, c] = phase_colors[radius % period]
    return out


def _semantic_checks(
    grid: np.ndarray,
    expected: np.ndarray,
    center: tuple[int, int],
    period: int,
    phase_colors: dict[int, int],
    max_radius: int,
    hidden_radius: int,
) -> bool:
    """Check the described rule, unique clues, and anti-shortcut dependencies."""
    try:
        inferred_center, inferred_period, inferred_colors = _infer_unique_model(grid)
        actual = main(grid)
    except (AssertionError, ValueError):
        return False

    if inferred_center != center or inferred_period != period:
        return False
    if inferred_colors != phase_colors or not np.array_equal(actual, expected):
        return False

    marker_mask = grid == MARKER
    if not marker_mask.any() or not np.all(actual[~marker_mask] == grid[~marker_mask]):
        return False
    if np.any(actual[marker_mask] == MARKER):
        return False

    # The fully hidden ring defeats copying from another cell on the same ring.
    hidden = _ring_coords(center, hidden_radius)
    if any(int(grid[r, c]) != MARKER for r, c in hidden):
        return False
    same_phase_radii = [
        d for d in range(1, max_radius + 1)
        if d != hidden_radius and d % period == hidden_radius % period
    ]
    if not any(
        any(int(grid[r, c]) == phase_colors[d % period] for r, c in _ring_coords(center, d))
        for d in same_phase_radii
    ):
        return False

    # Markers exercise every phase, so the whole inferred cycle affects the output.
    corrected_colors = set(map(int, actual[marker_mask]))
    if corrected_colors != set(phase_colors.values()):
        return False
    marker_radii = {
        max(abs(int(r) - center[0]), abs(int(c) - center[1]))
        for r, c in np.argwhere(marker_mask)
    }
    if len(marker_radii) < period or len(corrected_colors) < 2:
        return False
    return True


def _generation_attempt(rng: np.random.Generator) -> np.ndarray | None:
    period = int(rng.integers(2, 5))
    max_radius = int(rng.integers(max(6, period + 2), 10))

    top_pad, bottom_pad, left_pad, right_pad = map(
        int, rng.integers(0, 4, size=4)
    )
    height = 2 * max_radius + 1 + top_pad + bottom_pad
    width = 2 * max_radius + 1 + left_pad + right_pad
    if max(height, width) > 30:
        return None
    center = (max_radius + top_pad, max_radius + left_pad)

    palette_pool = [c for c in range(1, 10) if c != MARKER]
    colors = [int(c) for c in rng.choice(palette_pool, size=period, replace=False)]
    phase_colors = {phase: colors[phase] for phase in range(period)}

    # Keep the fully covered ring moderate in size while ensuring a later cycle
    # witnesses its color.
    hidden_candidates = list(range(1, min(4, max_radius - period) + 1))
    if not hidden_candidates:
        return None
    hidden_radius = int(rng.choice(hidden_candidates))

    grid = np.zeros((height, width), dtype=int)
    for radius in range(1, max_radius + 1):
        ring = _ring_coords(center, radius)
        if radius == hidden_radius:
            for r, c in ring:
                grid[r, c] = MARKER
            continue
        color = phase_colors[radius % period]
        for r, c in _sample_arc_cells(ring, radius, rng):
            grid[r, c] = color

    # Put at least one missing-value marker in every radial phase. For the hidden
    # phase the whole ring already supplies it; other markers puncture visible arcs
    # or occupy omitted cells, covering both common visual cases.
    for phase in range(period):
        candidate_radii = [
            d for d in range(1, max_radius + 1)
            if d % period == phase and d != hidden_radius
        ]
        if not candidate_radii:
            return None
        radius = int(rng.choice(candidate_radii))
        ring = _ring_coords(center, radius)
        if rng.random() < 0.65:
            colored = [(r, c) for r, c in ring if grid[r, c] not in (BACKGROUND, MARKER)]
            pos = colored[int(rng.integers(len(colored)))] if colored else ring[int(rng.integers(len(ring)))]
        else:
            pos = ring[int(rng.integers(len(ring)))]
        grid[pos] = MARKER

    for _ in range(int(rng.integers(3, 8))):
        radius = int(rng.integers(1, max_radius + 1))
        if radius == hidden_radius:
            continue
        ring = _ring_coords(center, radius)
        grid[ring[int(rng.integers(len(ring)))]] = MARKER

    expected = _oracle_output(grid, center, period, phase_colors)
    if not _semantic_checks(
        grid, expected, center, period, phase_colors, max_radius, hidden_radius
    ):
        return None

    global _LAST_META
    _LAST_META = {
        "center": center,
        "period": period,
        "max_radius": max_radius,
        "hidden_radius": hidden_radius,
        "shape": grid.shape,
    }
    return grid


def generate_input() -> np.ndarray:
    """Generate a varied, uniquely solvable radial-ring completion instance."""
    rng = np.random.default_rng()
    for _ in range(_MAX_GENERATION_ATTEMPTS):
        grid = _generation_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a unique radial-ring completion input")
