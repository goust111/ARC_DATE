"""ARC task: complete a half-turn template and stamp it on yellow centers."""

from __future__ import annotations

import random

import numpy as np


BACKGROUND = 0
MARKER = 4
RADIUS = 2
MAX_GENERATION_ATTEMPTS = 400


def _inside(shape: tuple[int, int], r: int, c: int) -> bool:
    """Return whether the row-column coordinate lies in the grid."""
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _window_has_pattern(grid: np.ndarray, center: tuple[int, int]) -> bool:
    """Test for a non-background, non-marker cell in the clipped 5x5 window."""
    r, c = center
    r0, r1 = max(0, r - RADIUS), min(grid.shape[0], r + RADIUS + 1)
    c0, c1 = max(0, c - RADIUS), min(grid.shape[1], c + RADIUS + 1)
    window = grid[r0:r1, c0:c1]
    return bool(np.any((window != BACKGROUND) & (window != MARKER)))


def main(grid: np.ndarray) -> np.ndarray:
    """Complete the half-turn-symmetric example and stamp it at every isolated yellow marker.

    Input structure:
    - A black background contains several isolated yellow center pixels.
    - Exactly one center has colored cells in its clipped 5x5 neighborhood; every
      other center has no non-yellow colored cell there.
    - The example neighborhood contains exactly one member of each used pair of
      offsets related by 180-degree point symmetry. Same-colored distractor pixels
      may occur only outside every center's 5x5 neighborhood.

    Rule:
    - Reflect every colored example cell through its yellow center, preserving its
      color, to form a complete point-symmetric 5x5 template.
    - Overlay that completed template, center aligned, on the example and every
      isolated yellow marker. Clip cells beyond the grid and preserve distractors.

    Key insight:
    - The visible example is only one half of the stamp, so point-symmetry
      completion must happen before the template is copied to the other centers.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a two-dimensional grid"
    assert g.size and int(np.count_nonzero(g == BACKGROUND)) > g.size // 2

    centers = [tuple(map(int, p)) for p in np.argwhere(g == MARKER)]
    assert len(centers) >= 2, "expected an example and at least one target marker"
    assert all(
        max(abs(r1 - r2), abs(c1 - c2)) > RADIUS
        for i, (r1, c1) in enumerate(centers)
        for r2, c2 in centers[i + 1 :]
    ), "yellow markers must be isolated from one another"

    decorated = [center for center in centers if _window_has_pattern(g, center)]
    assert len(decorated) == 1, "the decorated example center must be unique"
    er, ec = decorated[0]

    shown: dict[tuple[int, int], int] = {}
    for dr in range(-RADIUS, RADIUS + 1):
        for dc in range(-RADIUS, RADIUS + 1):
            if dr == dc == 0:
                continue
            r, c = er + dr, ec + dc
            if not _inside(g.shape, r, c):
                continue
            color = int(g[r, c])
            if color not in (BACKGROUND, MARKER):
                shown[(dr, dc)] = color

    assert shown, "the example must contain a visible half-template"
    for (dr, dc), color in shown.items():
        opposite = (-dr, -dc)
        assert opposite not in shown, "exactly one cell from each symmetric pair is shown"
        rr, cc = er - dr, ec - dc
        assert _inside(g.shape, rr, cc) and int(g[rr, cc]) == BACKGROUND
        assert color not in (BACKGROUND, MARKER)

    template = dict(shown)
    template.update({(-dr, -dc): color for (dr, dc), color in shown.items()})

    # Collect first so the result cannot depend on marker traversal order.
    assignments: dict[tuple[int, int], int] = {}
    for cr, cc in centers:
        for (dr, dc), color in template.items():
            r, c = cr + dr, cc + dc
            if not _inside(g.shape, r, c):
                continue
            assert int(g[r, c]) != MARKER, "a stamp may not cover another center"
            old = assignments.get((r, c))
            assert old is None or old == color, "overlapping stamps must agree in color"
            assignments[(r, c)] = color

    out = g.copy()
    for (r, c), color in assignments.items():
        out[r, c] = color
    return out


_PAIR_REPRESENTATIVES = tuple(
    (dr, dc)
    for dr in range(-RADIUS, RADIUS + 1)
    for dc in range(-RADIUS, RADIUS + 1)
    if dr > 0 or (dr == 0 and dc > 0)
)


def _far_enough(point: tuple[int, int], points: list[tuple[int, int]]) -> bool:
    """Keep 5x5 semantic windows disjoint and visually separated."""
    return all(max(abs(point[0] - r), abs(point[1] - c)) >= 6 for r, c in points)


def _sample_centers(h: int, w: int, count: int) -> list[tuple[int, int]] | None:
    """Choose one full-window example, one clipped target, and further targets."""
    example = (random.randint(2, h - 3), random.randint(2, w - 3))
    side = random.choice(("top", "bottom", "left", "right", "corner"))
    if side == "top":
        edge = (0, random.randrange(w))
    elif side == "bottom":
        edge = (h - 1, random.randrange(w))
    elif side == "left":
        edge = (random.randrange(h), 0)
    elif side == "right":
        edge = (random.randrange(h), w - 1)
    else:
        edge = random.choice(((0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)))
    if not _far_enough(edge, [example]):
        return None

    centers = [example, edge]
    candidates = [(r, c) for r in range(h) for c in range(w)]
    random.shuffle(candidates)
    for point in candidates:
        if _far_enough(point, centers):
            centers.append(point)
            if len(centers) == count:
                return centers
    return None


def _render_expected(
    inp: np.ndarray,
    centers: list[tuple[int, int]],
    complete: dict[tuple[int, int], int],
) -> np.ndarray:
    """Render the independently known full template with boundary clipping."""
    expected = inp.copy()
    for cr, cc in centers:
        for (dr, dc), color in complete.items():
            r, c = cr + dr, cc + dc
            if _inside(inp.shape, r, c):
                expected[r, c] = color
    return expected


def _semantic_checks(
    inp: np.ndarray,
    expected: np.ndarray,
    centers: list[tuple[int, int]],
    shown: dict[tuple[int, int], int],
    distractors: set[tuple[int, int]],
) -> bool:
    """Check uniqueness, true half-completion, clipping, and distractor preservation."""
    decorated = [p for p in centers if _window_has_pattern(inp, p)]
    if decorated != [centers[0]]:
        return False

    er, ec = centers[0]
    for dr, dc in shown:
        if int(inp[er - dr, ec - dc]) != BACKGROUND:
            return False

    complete = dict(shown)
    complete.update({(-dr, -dc): col for (dr, dc), col in shown.items()})
    if expected.shape != inp.shape or np.array_equal(inp, expected):
        return False

    # A solver that merely copies the visible half must be observably wrong at
    # every target; this makes the completion step indispensable.
    direct_copy = _render_expected(inp, centers, shown)
    if np.array_equal(direct_copy, expected):
        return False
    for cr, cc in centers[1:]:
        if not any(
            _inside(inp.shape, cr - dr, cc - dc)
            and int(expected[cr - dr, cc - dc]) == color
            and int(direct_copy[cr - dr, cc - dc]) != color
            for (dr, dc), color in shown.items()
        ):
            return False

    # At least one full-template cell is clipped at the deliberately edge-bound target.
    cr, cc = centers[1]
    if not any(not _inside(inp.shape, cr + dr, cc + dc) for dr, dc in complete):
        return False
    if any(int(expected[r, c]) != int(inp[r, c]) for r, c in distractors):
        return False
    return True


def _generation_attempt() -> np.ndarray | None:
    h, w = random.randint(13, 24), random.randint(13, 24)
    marker_count = random.randint(3, 5)
    centers = _sample_centers(h, w, marker_count)
    if centers is None:
        return None

    pair_count = random.randint(4, 8)
    pairs = random.sample(_PAIR_REPRESENTATIVES, pair_count)
    if not any(max(abs(dr), abs(dc)) == RADIUS for dr, dc in pairs):
        pairs[0] = random.choice(
            [p for p in _PAIR_REPRESENTATIVES if max(abs(p[0]), abs(p[1])) == RADIUS]
        )

    color_count = random.randint(2, min(3, pair_count))
    palette = random.sample([c for c in range(1, 10) if c != MARKER], color_count)
    colors = [palette[i % color_count] for i in range(pair_count)]
    random.shuffle(colors)

    shown: dict[tuple[int, int], int] = {}
    for pair, color in zip(pairs, colors):
        dr, dc = pair
        offset = (dr, dc) if random.random() < 0.5 else (-dr, -dc)
        shown[offset] = color
    complete = dict(shown)
    complete.update({(-dr, -dc): color for (dr, dc), color in shown.items()})

    inp = np.full((h, w), BACKGROUND, dtype=int)
    for r, c in centers:
        inp[r, c] = MARKER
    er, ec = centers[0]
    for (dr, dc), color in shown.items():
        inp[er + dr, ec + dc] = color

    distractors: set[tuple[int, int]] = set()
    candidates = [
        (r, c)
        for r in range(h)
        for c in range(w)
        if inp[r, c] == BACKGROUND
        and all(max(abs(r - cr), abs(c - cc)) > RADIUS for cr, cc in centers)
    ]
    random.shuffle(candidates)
    for r, c in candidates:
        if len(distractors) >= random.randint(2, 5):
            break
        if all(max(abs(r - rr), abs(c - cc)) >= 2 for rr, cc in distractors):
            inp[r, c] = random.choice(palette)
            distractors.add((r, c))
    if len(distractors) < 2:
        return None

    expected = _render_expected(inp, centers, complete)
    if not _semantic_checks(inp, expected, centers, shown, distractors):
        return None
    try:
        solved = main(inp)
    except (AssertionError, IndexError):
        return None
    if not np.array_equal(solved, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    """Generate a varied, non-degenerate instance satisfying all puzzle invariants."""
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generation_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid half-shadow template input")
