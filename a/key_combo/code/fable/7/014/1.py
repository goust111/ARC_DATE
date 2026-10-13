"""ARC task: cascaded reflections through sparse, colored mirror lines."""

from __future__ import annotations

import random
from collections import deque

import numpy as np


BACKGROUND = 0
PALETTE = tuple(range(1, 10))
GENERATION_ATTEMPTS = 400


def _roles(grid: np.ndarray):
    """Return (orientation, motif_color, motif_coords, [(axis, color), ...])."""
    colors = [int(v) for v in np.unique(grid) if int(v) != BACKGROUND]
    assert len(colors) >= 4, "expected one motif and at least three colored marker lines"

    horizontal = []
    vertical = []
    for color in colors:
        coords = np.argwhere(grid == color)
        if len(coords) >= 3 and len(np.unique(coords[:, 0])) == 1:
            horizontal.append((int(coords[0, 0]), color))
        if len(coords) >= 3 and len(np.unique(coords[:, 1])) == 1:
            vertical.append((int(coords[0, 1]), color))

    if len(horizontal) == len(colors) - 1 and len(vertical) != len(colors) - 1:
        orientation, raw_lines = "horizontal", horizontal
    elif len(vertical) == len(colors) - 1 and len(horizontal) != len(colors) - 1:
        orientation, raw_lines = "vertical", vertical
    else:
        raise AssertionError("marker lines must have one unambiguous common orientation")

    line_colors = {color for _, color in raw_lines}
    motif_colors = set(colors) - line_colors
    assert len(motif_colors) == 1, "expected one uniquely identifiable motif color"
    motif_color = motif_colors.pop()
    motif = np.argwhere(grid == motif_color)
    assert len(motif) >= 4, "expected a multi-cell motif"

    # The motif is one 4-connected object, and is not itself line-like.
    cells = {tuple(map(int, p)) for p in motif}
    seen = {next(iter(cells))}
    queue = deque(seen)
    while queue:
        r, c = queue.popleft()
        for q in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if q in cells and q not in seen:
                seen.add(q)
                queue.append(q)
    assert seen == cells, "motif must be 4-connected"
    assert len(np.unique(motif[:, 0])) > 1 and len(np.unique(motif[:, 1])) > 1

    coord_index = 0 if orientation == "horizontal" else 1
    lo = int(motif[:, coord_index].min())
    hi = int(motif[:, coord_index].max())
    axes = [axis for axis, _ in raw_lines]
    assert len(set(axes)) == len(axes), "marker axes must be distinct"
    if all(axis > hi for axis in axes):
        lines = sorted(raw_lines)
    elif all(axis < lo for axis in axes):
        lines = sorted(raw_lines, reverse=True)
    else:
        raise AssertionError("all marker lines must lie on the same side of the motif")

    gaps = [abs(lines[i + 1][0] - lines[i][0]) for i in range(len(lines) - 1)]
    assert len(set(gaps)) == len(gaps), "successive marker-line gaps must be unequal"
    return orientation, motif_color, motif, lines


def _reflect(coords: np.ndarray, orientation: str, axis: int) -> np.ndarray:
    reflected = coords.copy()
    index = 0 if orientation == "horizontal" else 1
    reflected[:, index] = 2 * int(axis) - reflected[:, index]
    return reflected


def main(grid: np.ndarray) -> np.ndarray:
    """
    Cascade a foreground motif through parallel sparse mirrors, recoloring each reflected copy by its mirror.

    Input structure:
    - A black background contains one multi-cell, single-color, 4-connected and non-line-like foreground motif.
    - On one side of the motif are at least three parallel sparse marker lines; each line has its own color.
    - The successive distances between marker axes are unequal, and every marker is geometrically separate from the motif.

    Rule:
    - Preserve the motif and all marker pixels. Reflect the motif across the nearest marker axis and color that copy with the marker color.
    - Reflect that newly obtained copy (not the original motif) across the next outward marker and recolor it with that marker's color; continue outward.
    - After every reflection discard coordinates outside the original canvas, so later stages cascade from the visible previous copy.

    Key insight:
    - Unequal mirror spacing makes each stage depend on the preceding reflected coordinates; reflecting the original independently or translating by a fixed step gives different positions.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and max(g.shape) <= 30, "expected an ARC-sized 2D grid"
    assert int(np.sum(g == BACKGROUND)) > g.size // 2, "black must be the dominant background"
    orientation, _motif_color, current, lines = _roles(g)

    out = g.copy()
    height, width = g.shape
    for axis, color in lines:
        current = _reflect(current, orientation, axis)
        inside = ((current[:, 0] >= 0) & (current[:, 0] < height) &
                  (current[:, 1] >= 0) & (current[:, 1] < width))
        current = current[inside]
        if len(current):
            out[current[:, 0], current[:, 1]] = color

    # Input objects always remain visible, even if a reflected cell meets one.
    foreground = g != BACKGROUND
    out[foreground] = g[foreground]
    return out


def _polyomino(perp_extent: int, along_extent: int) -> set[tuple[int, int]]:
    """Make an asymmetric connected motif in local (perpendicular, along) coordinates."""
    for _ in range(200):
        target = random.randint(max(5, perp_extent + along_extent - 1),
                                min(10, perp_extent * along_extent - 1))
        cells = {(random.randrange(perp_extent), random.randrange(along_extent))}
        while len(cells) < target:
            p, x = random.choice(tuple(cells))
            dp, dx = random.choice(((-1, 0), (1, 0), (0, -1), (0, 1)))
            q = (p + dp, x + dx)
            if 0 <= q[0] < perp_extent and 0 <= q[1] < along_extent:
                cells.add(q)
        ps = [p for p, _ in cells]
        xs = [x for _, x in cells]
        if min(ps) or max(ps) != perp_extent - 1 or min(xs) or max(xs) != along_extent - 1:
            continue
        flipped = {(p, along_extent - 1 - x) for p, x in cells}
        if flipped != cells:
            return cells
    raise RuntimeError("could not sample an asymmetric connected motif")


def _render_expected(inp: np.ndarray, motif: np.ndarray, orientation: str,
                     lines: list[tuple[int, int]]) -> tuple[np.ndarray, list[np.ndarray]]:
    """Independent generator oracle retaining every intermediate visible stage."""
    expected = inp.copy()
    stages = []
    current = motif.copy()
    h, w = inp.shape
    index = 0 if orientation == "horizontal" else 1
    for axis, color in lines:
        current[:, index] = 2 * axis - current[:, index]
        current = current[(current[:, 0] >= 0) & (current[:, 0] < h) &
                          (current[:, 1] >= 0) & (current[:, 1] < w)]
        stages.append(current.copy())
        if len(current):
            expected[current[:, 0], current[:, 1]] = color
    expected[inp != BACKGROUND] = inp[inp != BACKGROUND]
    return expected, stages


def _attempt() -> np.ndarray | None:
    line_count = random.choice((3, 3, 4))
    along_extent = random.randint(2, 3 if line_count == 4 else 4)
    perp_extent = random.randint(3, 5)
    motif_local = _polyomino(perp_extent, along_extent)

    # Canonical construction propagates toward increasing "along" coordinates.
    motif_start = random.randint(1, 3)
    motif_along = np.array([motif_start + x for _, x in motif_local], dtype=int)
    axis1 = motif_start + along_extent - 1 + random.randint(2, 3)
    base_gap = along_extent + 3
    gaps = random.sample(range(base_gap, base_gap + line_count + 1), line_count - 1)
    axes = [axis1]
    for gap in gaps:
        axes.append(axes[-1] + gap)

    # Track full canonical copies to choose a canvas that sometimes clips only the last copy.
    full_stages = []
    cur = motif_along.copy()
    for axis in axes:
        cur = 2 * axis - cur
        full_stages.append(cur.copy())
    if any(int(full_stages[i].max()) >= axes[i + 1] for i in range(line_count - 1)):
        return None

    clip_last = random.random() < 0.35
    last_min, last_max = int(full_stages[-1].min()), int(full_stages[-1].max())
    if clip_last and last_min < last_max:
        along_size = random.randint(last_min + 1, last_max)
    else:
        along_size = last_max + random.randint(2, 4)
    if along_size > 30 or axes[-1] >= along_size:
        return None
    perp_size = random.randint(max(14, perp_extent + 6), 22)

    perp_start = random.randint(2, perp_size - perp_extent - 2)
    canonical = np.array([(perp_start + p, motif_start + x) for p, x in motif_local], dtype=int)
    colors = random.sample(PALETTE, line_count + 1)
    motif_color, line_colors = colors[0], colors[1:]
    canonical_lines = list(zip(axes, line_colors))

    # Random transpose and side reversal cover horizontal/vertical and all four directions.
    orientation = random.choice(("horizontal", "vertical"))
    reverse = random.choice((False, True))
    if orientation == "vertical":
        shape = (perp_size, along_size)
        motif = canonical.copy()
        if reverse:
            motif[:, 1] = along_size - 1 - motif[:, 1]
            line_axes = [along_size - 1 - a for a in axes]
        else:
            line_axes = axes[:]
    else:
        shape = (along_size, perp_size)
        motif = canonical[:, ::-1].copy()
        if reverse:
            motif[:, 0] = along_size - 1 - motif[:, 0]
            line_axes = [along_size - 1 - a for a in axes]
        else:
            line_axes = axes[:]

    inp = np.zeros(shape, dtype=int)
    inp[motif[:, 0], motif[:, 1]] = motif_color
    dot_step = random.choice((2, 3))
    dot_phase = random.randrange(dot_step)
    dots = list(range(dot_phase, perp_size, dot_step))
    if len(dots) < 4:
        return None
    for axis, color in zip(line_axes, line_colors):
        if orientation == "vertical":
            inp[dots, axis] = color
        else:
            inp[axis, dots] = color

    ordered_lines = list(zip(line_axes, line_colors))
    expected, stages = _render_expected(inp, motif, orientation, ordered_lines)
    if any(len(stage) == 0 for stage in stages):
        return None
    if any(np.any(inp[stage[:, 0], stage[:, 1]] != BACKGROUND) for stage in stages):
        return None

    # Semantic counterexample: independent reflection of the original must be wrong.
    independent = inp.copy()
    for (axis, color) in ordered_lines:
        wrong = _reflect(motif, orientation, axis)
        valid = ((wrong[:, 0] >= 0) & (wrong[:, 0] < shape[0]) &
                 (wrong[:, 1] >= 0) & (wrong[:, 1] < shape[1]))
        wrong = wrong[valid]
        independent[wrong[:, 0], wrong[:, 1]] = color
    independent[inp != BACKGROUND] = inp[inp != BACKGROUND]
    if np.array_equal(independent, expected):
        return None

    try:
        detected_orientation, detected_color, detected_motif, detected_lines = _roles(inp)
        actual = main(inp)
    except (AssertionError, ValueError):
        return None
    if detected_orientation != orientation or detected_color != motif_color:
        return None
    if detected_lines != ordered_lines or {tuple(x) for x in detected_motif} != {tuple(x) for x in motif}:
        return None
    if not np.array_equal(actual, expected) or np.array_equal(actual, inp):
        return None
    return inp


def generate_input() -> np.ndarray:
    """Generate a diverse, oracle-checked mirror-hall instance of at most 30x30."""
    for _ in range(GENERATION_ATTEMPTS):
        grid = _attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid cascaded-reflection input")
