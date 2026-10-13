from __future__ import annotations

from itertools import product
import random

import numpy as np


ANCHOR_COLOR = 5


def _background(grid: np.ndarray) -> int:
    """Return the unique most frequent color."""
    values, counts = np.unique(grid, return_counts=True)
    order = np.argsort(counts)[::-1]
    assert len(order) > 0
    if len(order) > 1:
        assert counts[order[0]] > counts[order[1]], "background must be uniquely dominant"
    return int(values[order[0]])


def _crop_foreground(grid: np.ndarray, bg: int) -> np.ndarray:
    """Crop to the inclusive bbox of non-background cells (arc_utils convention)."""
    coords = np.argwhere(grid != int(bg))
    assert coords.size, "expected a nonempty hidden figure"
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return grid[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1].copy()


def _is_connected4(mask: np.ndarray) -> bool:
    """Whether all True cells form one 4-connected component."""
    points = np.argwhere(mask)
    if points.size == 0:
        return False
    start = tuple(map(int, points[0]))
    seen = {start}
    stack = [start]
    h, w = mask.shape
    while stack:
        r, c = stack.pop()
        for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= rr < h and 0 <= cc < w and bool(mask[rr, cc]) and (rr, cc) not in seen:
                seen.add((rr, cc))
                stack.append((rr, cc))
    return len(seen) == int(mask.sum())


def _valid_revelation(canvas: np.ndarray, bg: int) -> bool:
    """Check color-preserving left/right symmetry and geometric connectivity."""
    if not bool(np.any(canvas != int(bg))):
        return False
    crop = _crop_foreground(canvas, bg)
    return bool(
        np.array_equal(crop, crop[:, ::-1])
        and _is_connected4(crop != int(bg))
    )


def _place_shifted_row(
    canvas: np.ndarray,
    source: np.ndarray,
    row: int,
    shift: int,
    bg: int,
) -> None:
    """Copy only colored cells of one source row after a nonnegative shift."""
    cols = np.flatnonzero(source[row] != int(bg))
    if cols.size == 0:
        return
    dst = cols + int(shift)
    assert int(dst[-1]) < canvas.shape[1], "row shift must remain inside the grid"
    canvas[int(row), dst] = source[int(row), cols]


def _solve(grid: np.ndarray) -> tuple[np.ndarray, tuple[int, ...], tuple[int, ...]]:
    """Solve and also return unanchored row ids and their unique shifts."""
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and 1 <= g.shape[0] <= 30 and 1 <= g.shape[1] <= 30
    bg = _background(g)
    assert bg != ANCHOR_COLOR, "gray is reserved for row anchors"

    anchor_mask = g == ANCHOR_COLOR
    colored_mask = (g != int(bg)) & ~anchor_mask
    active_rows = [
        r for r in range(g.shape[0])
        if bool(np.any(anchor_mask[r])) or bool(np.any(colored_mask[r]))
    ]
    assert active_rows, "expected active rows"

    anchored_rows: list[tuple[int, int]] = []
    unanchored_rows: list[int] = []
    for r in active_rows:
        anchors = np.flatnonzero(anchor_mask[r])
        assert len(anchors) <= 1, "an active row has at most one gray anchor"
        assert bool(np.any(colored_mask[r])), "every active row must carry colored pixels"
        if len(anchors) == 1:
            anchored_rows.append((int(r), int(anchors[0])))
        else:
            unanchored_rows.append(int(r))

    assert unanchored_rows, "expected at least one row without an anchor"
    assert len(anchored_rows) > len(unanchored_rows), "most active rows must be anchored"

    # A common destination must be at or right of every anchor.  Therefore the
    # rightmost observed anchor is the unique destination minimizing total shift.
    target_col = max(c for _r, c in anchored_rows)
    colored = np.where(colored_mask, g, int(bg))
    fixed = np.full_like(g, int(bg))
    for r, anchor_col in anchored_rows:
        _place_shifted_row(fixed, colored, r, target_col - anchor_col, bg)

    ranges: list[range] = []
    for r in unanchored_rows:
        cols = np.flatnonzero(colored_mask[r])
        assert cols.size
        ranges.append(range(g.shape[1] - int(cols[-1])))

    combinations = 1
    for choices in ranges:
        combinations *= len(choices)
    assert combinations <= 250_000, "too many unanchored-row shift combinations"

    matches: list[tuple[tuple[int, ...], np.ndarray]] = []
    for shifts in product(*ranges):
        candidate = fixed.copy()
        for r, shift in zip(unanchored_rows, shifts):
            _place_shifted_row(candidate, colored, r, int(shift), bg)
        if _valid_revelation(candidate, bg):
            matches.append((tuple(map(int, shifts)), candidate))
            if len(matches) > 1:
                break

    assert len(matches) == 1, "symmetry and connectivity must determine one shift tuple"
    shifts, revealed = matches[0]
    return _crop_foreground(revealed, bg), tuple(unanchored_rows), shifts


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reveal a scrambled multicolor figure by anchor-aligning rows and symmetry-solving the unanchored rows.

    Input structure:
    - A rectangular grid has one uniquely dominant background color, colored pixels
      on every active row, and gray row anchors that never represent figure pixels.
    - Most active rows contain exactly one gray anchor; a smaller nonempty set has
      no anchor.  The valid row shifts are nonnegative and stay within the canvas.

    Rule:
    - Shift every anchored row rightward until its gray pixel reaches the common
      column giving minimum total nonnegative shift, namely the rightmost anchor.
    - Erase the anchors.  Exhaustively choose nonnegative shifts for unanchored rows
      and retain the unique arrangement whose colored figure is 4-connected and
      color-preserving left/right mirror symmetric.  Return its minimum bbox crop.

    Key insight:
    - Anchors determine the first layer of row offsets, but absent anchors carry no
      positional clue; their offsets are recovered only from the revealed figure's
      common mirror axis, with connectivity ruling out invalid assemblies.
    """
    result, _rows, _shifts = _solve(grid)
    return result


# ------------------------------ generator ------------------------------

_GEN_ATTEMPTS = 400
_BACKGROUNDS = (0, 1, 4, 7, 8, 9)
_FIGURE_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)


def _symmetric_shape(rng: random.Random, h: int, w: int) -> np.ndarray:
    """Make a non-rectangular symmetric 4-connected mask with a full-width row."""
    radius = w // 2
    if w % 2:
        levels = [rng.randint(0, radius - 1) for _ in range(h)]
    else:
        levels = [rng.randint(1, radius - 1) for _ in range(h)]
    full_row = rng.randrange(h)
    levels[full_row] = radius

    # Force silhouette diversity rather than occasionally returning a rectangle
    # or a two-level hourglass repeatedly.
    if len(set(levels)) < 3:
        levels[(full_row + 1) % h] = 0 if w % 2 else 1
        levels[(full_row + 2) % h] = max(1, radius - 1)

    mask = np.zeros((h, w), dtype=bool)
    for r, level in enumerate(levels):
        if w % 2:
            mask[r, radius - level : radius + level + 1] = True
        else:
            mask[r, radius - level : radius + level] = True
    assert np.array_equal(mask, mask[:, ::-1])
    assert _is_connected4(mask)
    assert bool(np.all(mask[full_row]))
    return mask


def _color_shape(
    rng: random.Random,
    mask: np.ndarray,
    bg: int,
) -> np.ndarray:
    """Color mirror orbits so the complete multicolor image mirrors exactly."""
    available = [c for c in _FIGURE_COLORS if c not in (int(bg), ANCHOR_COLOR)]
    color_count = rng.choice((2, 3))
    palette = rng.sample(available, color_count)
    h, w = mask.shape
    orbits: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range((w + 1) // 2):
            if not bool(mask[r, c]):
                continue
            mirror = w - 1 - c
            orbit = [(r, c)] if mirror == c else [(r, c), (r, mirror)]
            orbits.append(orbit)
    rng.shuffle(orbits)

    figure = np.full(mask.shape, int(bg), dtype=int)
    for i, orbit in enumerate(orbits):
        color = palette[i % len(palette)] if i < len(palette) else rng.choice(palette)
        for r, c in orbit:
            figure[r, c] = int(color)
    assert set(map(int, np.unique(figure[mask]))) == set(palette)
    assert np.array_equal(figure, figure[:, ::-1])
    return figure


def _build_attempt(rng: random.Random) -> np.ndarray | None:
    bg = rng.choice(_BACKGROUNDS)
    h = rng.randint(7, 12)
    w = rng.randint(7, 13)
    mask = _symmetric_shape(rng, h, w)
    figure = _color_shape(rng, mask, bg)

    full_rows = [r for r in range(h) if bool(np.all(mask[r]))]
    protected_full_row = rng.choice(full_rows)
    missing_count = rng.choice((1, 1, 2, 2, 2))
    missing_pool = [r for r in range(h) if r != protected_full_row]
    missing_rows = sorted(rng.sample(missing_pool, missing_count))
    anchored_rows = [r for r in range(h) if r not in missing_rows]
    if len(anchored_rows) <= len(missing_rows):
        return None

    anchor_target = rng.randint(4, 7)
    gap = rng.randint(1, 2)
    figure_col0 = anchor_target + gap + 1
    grid_w = figure_col0 + w + rng.randint(2, 5)
    if grid_w > 30:
        return None

    shifts = [0] * h
    # At least two anchored rows have different offsets, so the direct-anchor
    # alignment stage visibly changes the assembly.
    zero_row = rng.choice(anchored_rows)
    positive_row = rng.choice([r for r in anchored_rows if r != zero_row])
    for r in anchored_rows:
        shifts[r] = rng.randint(0, anchor_target)
    shifts[zero_row] = 0
    shifts[positive_row] = rng.randint(1, anchor_target)
    for r in missing_rows:
        shifts[r] = rng.randint(1, anchor_target)

    inp = np.full((h, grid_w), int(bg), dtype=int)
    for r in range(h):
        s = shifts[r]
        source_cols = np.flatnonzero(mask[r])
        inp[r, figure_col0 + source_cols - s] = figure[r, source_cols]
        if r not in missing_rows:
            inp[r, anchor_target - s] = ANCHOR_COLOR

    values, counts = np.unique(inp, return_counts=True)
    bg_count = int(counts[np.flatnonzero(values == int(bg))[0]])
    if any(bg_count <= int(count) for value, count in zip(values, counts) if int(value) != int(bg)):
        return None

    expected = figure.copy()
    try:
        actual, solved_rows, solved_shifts = _solve(inp)
    except (AssertionError, ValueError):
        return None
    if not np.array_equal(actual, expected):
        return None
    if solved_rows != tuple(missing_rows):
        return None
    if solved_shifts != tuple(shifts[r] for r in missing_rows):
        return None

    # Semantic counterexamples: neither leaving all rows untouched nor doing only
    # the anchor-marked rows may accidentally satisfy the target constraints.
    colored = np.where((inp != int(bg)) & (inp != ANCHOR_COLOR), inp, int(bg))
    if _valid_revelation(colored, bg):
        return None
    anchored_only = np.full_like(inp, int(bg))
    anchor_cols = np.argwhere(inp == ANCHOR_COLOR)
    target = int(anchor_cols[:, 1].max())
    for r in anchored_rows:
        c = int(np.flatnonzero(inp[r] == ANCHOR_COLOR)[0])
        _place_shifted_row(anchored_only, colored, r, target - c, bg)
    for r in missing_rows:
        _place_shifted_row(anchored_only, colored, r, 0, bg)
    if _valid_revelation(anchored_only, bg):
        return None
    return inp


def generate_input() -> np.ndarray:
    """Generate a diverse valid scrambled-row instance with a unique solution."""
    rng = random.Random()
    for _ in range(_GEN_ATTEMPTS):
        result = _build_attempt(rng)
        if result is not None:
            return result
    raise RuntimeError("failed to generate a unique nondegenerate row-alignment puzzle")
