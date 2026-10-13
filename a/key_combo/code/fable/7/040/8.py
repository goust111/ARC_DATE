"""ARC puzzle 8: gate control bar."""

from __future__ import annotations

import numpy as np


def _background(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _components(mask: np.ndarray) -> list[np.ndarray]:
    """Return 4-connected components as arrays of (row, column) coordinates."""
    height, width = mask.shape
    unseen = mask.copy()
    result: list[np.ndarray] = []
    while bool(unseen.any()):
        start = tuple(map(int, np.argwhere(unseen)[0]))
        unseen[start] = False
        stack = [start]
        cells: list[tuple[int, int]] = []
        while stack:
            row, col = stack.pop()
            cells.append((row, col))
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr, nc = row + dr, col + dc
                if 0 <= nr < height and 0 <= nc < width and unseen[nr, nc]:
                    unseen[nr, nc] = False
                    stack.append((nr, nc))
        result.append(np.asarray(cells, dtype=int))
    return result


def _line_candidate(line: np.ndarray, bg: int):
    """Recognize a boundary-to-boundary monochrome line with one inner gap."""
    if line.size < 3 or int(line[0]) == bg or int(line[-1]) == bg:
        return None
    color = int(line[0])
    if int(line[-1]) != color:
        return None
    gaps = np.flatnonzero(line == bg)
    if gaps.size == 0:
        return None
    gap0, gap1 = int(gaps[0]), int(gaps[-1])
    if gap0 == 0 or gap1 == line.size - 1:
        return None
    if not np.all(line[gap0 : gap1 + 1] == bg):
        return None
    outside = np.concatenate((line[:gap0], line[gap1 + 1 :]))
    if not np.all(outside == color):
        return None
    return color, gap0, gap1


def _find_bar(grid: np.ndarray, bg: int):
    candidates = []
    for row in range(grid.shape[0]):
        found = _line_candidate(grid[row, :], bg)
        if found is not None:
            candidates.append(("horizontal", row, *found))
    for col in range(grid.shape[1]):
        found = _line_candidate(grid[:, col], bg)
        if found is not None:
            candidates.append(("vertical", col, *found))
    assert len(candidates) == 1, "expected one uniquely recognizable control bar"
    return candidates[0]


def main(grid: np.ndarray) -> np.ndarray:
    """Move sprites toward a gapped control bar, passing and recoloring only exact-span fits.

    Input structure:
    - A dominant background contains one boundary-to-boundary, one-cell-thick
      monochrome row or column with exactly one continuous background gap.
    - Several separated monochrome sprites lie on the same side of that bar;
      their projections parallel to the bar do not overlap, and their colors
      differ from the bar color.

    Rule:
    - Translate every sprite perpendicularly toward the bar without changing
      its parallel coordinate.  A sprite whose complete parallel-axis bounding
      span lies inside the gap continues to the opposite canvas edge and is
      recolored to the bar color.  Every other sprite stops flush against the
      near side of the bar and retains its original color.  Preserve the bar
      and its gap.

    Key insight:
    - The gap is a gate rather than a destination: full span containment alone
      selects both the farther translation and the linked recoloring operation.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a nonempty 2D grid"
    bg = _background(g)
    orientation, bar_index, bar_color, gap0, gap1 = _find_bar(g, bg)

    # Remove the complete bar axis before component finding.  This prevents its
    # two visible segments from being mistaken for sprites.
    without_bar = g.copy()
    if orientation == "horizontal":
        without_bar[bar_index, :] = bg
    else:
        without_bar[:, bar_index] = bg

    sprites = []
    for color in np.unique(without_bar):
        color = int(color)
        if color in (bg, bar_color):
            continue
        for coords in _components(without_bar == color):
            r0, c0 = coords.min(axis=0)
            r1, c1 = coords.max(axis=0)
            sprites.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                }
            )
    assert sprites, "expected at least one sprite"

    if orientation == "horizontal":
        sides = [
            "before" if sprite["bbox"][2] < bar_index else
            "after" if sprite["bbox"][0] > bar_index else "crossing"
            for sprite in sprites
        ]
        spans = [(sprite["bbox"][1], sprite["bbox"][3]) for sprite in sprites]
    else:
        sides = [
            "before" if sprite["bbox"][3] < bar_index else
            "after" if sprite["bbox"][1] > bar_index else "crossing"
            for sprite in sprites
        ]
        spans = [(sprite["bbox"][0], sprite["bbox"][2]) for sprite in sprites]
    assert "crossing" not in sides and len(set(sides)) == 1, "sprites must share one side"
    ordered_spans = sorted(spans)
    assert all(a1 < b0 for (_a0, a1), (b0, _b1) in zip(ordered_spans, ordered_spans[1:])), \
        "parallel sprite projections must not overlap"

    # Erase all sources first so destinations are independent of processing order.
    out = g.copy()
    for sprite in sprites:
        coords = sprite["coords"]
        out[coords[:, 0], coords[:, 1]] = bg

    for sprite, span in zip(sprites, spans):
        r0, c0, r1, c1 = sprite["bbox"]
        fits = gap0 <= span[0] and span[1] <= gap1
        if orientation == "horizontal":
            if sides[0] == "before":
                dr = (g.shape[0] - 1 - r1) if fits else (bar_index - 1 - r1)
            else:
                dr = -r0 if fits else (bar_index + 1 - r0)
            dc = 0
        else:
            if sides[0] == "before":
                dc = (g.shape[1] - 1 - c1) if fits else (bar_index - 1 - c1)
            else:
                dc = -c0 if fits else (bar_index + 1 - c0)
            dr = 0

        new_color = bar_color if fits else int(sprite["color"])
        for row, col in sprite["coords"]:
            nr, nc = int(row) + dr, int(col) + dc
            assert 0 <= nr < out.shape[0] and 0 <= nc < out.shape[1]
            assert int(out[nr, nc]) == bg, "sprite destinations must remain separate"
            out[nr, nc] = new_color
    return out


_RNG = np.random.default_rng()
_MAX_GENERATION_ATTEMPTS = 300


def _connected(mask: np.ndarray) -> bool:
    return bool(mask.any()) and len(_components(mask)) == 1


def _random_sprite(height: int, width: int) -> np.ndarray:
    """Make a tight, connected and usually non-rectangular polyomino mask."""
    mask = np.ones((height, width), dtype=bool)
    cells = [(r, c) for r in range(height) for c in range(width)]
    _RNG.shuffle(cells)
    target_removals = int(_RNG.integers(0, max(1, (height * width) // 3 + 1)))
    removed = 0
    for row, col in cells:
        if removed >= target_removals or int(mask.sum()) <= 2:
            break
        trial = mask.copy()
        trial[row, col] = False
        if not trial[0, :].any() or not trial[-1, :].any():
            continue
        if not trial[:, 0].any() or not trial[:, -1].any():
            continue
        if _connected(trial):
            mask = trial
            removed += 1
    return mask


def _separated(interval: tuple[int, int], intervals: list[tuple[int, int]]) -> bool:
    a0, a1 = interval
    return all(a1 < b0 - 1 or b1 < a0 - 1 for b0, b1 in intervals)


def _choose_intervals(length: int, gap0: int, gap1: int, count: int):
    """Seed fit, partial-overlap, and outside counterexamples, then diversify."""
    gap_width = gap1 - gap0 + 1
    partial_on_left = bool(_RNG.integers(2))
    partial_width = int(_RNG.integers(2, min(4, gap_width) + 1))
    overlap = int(_RNG.integers(1, min(2, partial_width - 1) + 1))

    if partial_on_left:
        partial = (gap0 - (partial_width - overlap), gap0 + overlap - 1)
        pass_low = partial[1] + 2
        pass_width = int(_RNG.integers(1, min(3, gap1 - pass_low + 1) + 1))
        fitting = (int(_RNG.integers(pass_low, gap1 - pass_width + 2)), 0)
        fitting = (fitting[0], fitting[0] + pass_width - 1)
        outside_width = int(_RNG.integers(1, 4))
        outside_start = int(_RNG.integers(gap1 + 2, length - outside_width))
        outside = (outside_start, outside_start + outside_width - 1)
    else:
        partial = (gap1 - overlap + 1, gap1 + (partial_width - overlap))
        pass_width = int(_RNG.integers(1, min(3, partial[0] - gap0 - 1) + 1))
        pass_high = partial[0] - 2
        pass_start = int(_RNG.integers(gap0, pass_high - pass_width + 2))
        fitting = (pass_start, pass_start + pass_width - 1)
        outside_width = int(_RNG.integers(1, 4))
        # end <= gap0 - 2, hence start <= gap0 - outside_width - 1;
        # numpy's upper endpoint is exclusive.
        outside_start = int(_RNG.integers(1, gap0 - outside_width))
        outside = (outside_start, outside_start + outside_width - 1)

    intervals = [fitting, partial, outside]
    candidates = []
    for width in (1, 2, 3):
        for start in range(1, length - width):
            interval = (start, start + width - 1)
            if _separated(interval, intervals):
                candidates.append(interval)
    while len(intervals) < count:
        feasible = [item for item in candidates if _separated(item, intervals)]
        if not feasible:
            return None
        intervals.append(feasible[int(_RNG.integers(len(feasible)))])
    _RNG.shuffle(intervals)
    return intervals


def _expected_from_plan(
    inp: np.ndarray,
    orientation: str,
    bar_index: int,
    bar_color: int,
    gap: tuple[int, int],
    side: str,
    sprites: list[dict],
) -> np.ndarray:
    """Plan-based oracle kept independent of main's scene inference."""
    out = inp.copy()
    for sprite in sprites:
        coords = sprite["coords"]
        out[coords[:, 0], coords[:, 1]] = sprite["bg"]
    for sprite in sprites:
        coords = sprite["coords"]
        span = sprite["span"]
        fits = gap[0] <= span[0] and span[1] <= gap[1]
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        if orientation == "horizontal":
            if side == "before":
                shift = inp.shape[0] - 1 - r1 if fits else bar_index - 1 - r1
            else:
                shift = -r0 if fits else bar_index + 1 - r0
            delta = (int(shift), 0)
        else:
            if side == "before":
                shift = inp.shape[1] - 1 - c1 if fits else bar_index - 1 - c1
            else:
                shift = -c0 if fits else bar_index + 1 - c0
            delta = (0, int(shift))
        color = bar_color if fits else sprite["color"]
        out[coords[:, 0] + delta[0], coords[:, 1] + delta[1]] = color
    return out


def _generate_attempt() -> np.ndarray | None:
    # Seventeen cells are needed for a 7-cell gap plus five-cell margins.
    height = int(_RNG.integers(17, 26))
    width = int(_RNG.integers(17, 26))
    orientation = "horizontal" if bool(_RNG.integers(2)) else "vertical"
    parallel = width if orientation == "horizontal" else height
    perpendicular = height if orientation == "horizontal" else width

    gap_width = int(_RNG.integers(7, min(11, parallel - 10) + 1))
    gap0 = int(_RNG.integers(5, parallel - gap_width - 4))
    gap1 = gap0 + gap_width - 1
    bar_index = int(_RNG.integers(7, perpendicular - 7))
    side = "before" if bool(_RNG.integers(2)) else "after"
    sprite_count = int(_RNG.integers(3, 6))
    intervals = _choose_intervals(parallel, gap0, gap1, sprite_count)
    if intervals is None:
        return None

    palette = list(map(int, _RNG.permutation(10)))
    bg = palette.pop()
    bar_color = palette.pop()
    colors = [palette.pop() for _ in range(sprite_count)]
    inp = np.full((height, width), bg, dtype=int)
    if orientation == "horizontal":
        inp[bar_index, :] = bar_color
        inp[bar_index, gap0 : gap1 + 1] = bg
    else:
        inp[:, bar_index] = bar_color
        inp[gap0 : gap1 + 1, bar_index] = bg

    sprites: list[dict] = []
    for interval, color in zip(intervals, colors):
        span0, span1 = interval
        sprite_parallel = span1 - span0 + 1
        sprite_perp = int(_RNG.integers(2, 5))
        mask = _random_sprite(sprite_perp, sprite_parallel)
        if side == "before":
            origin = int(_RNG.integers(1, bar_index - sprite_perp))
        else:
            origin = int(_RNG.integers(bar_index + 2, perpendicular - sprite_perp))
        local = np.argwhere(mask)
        if orientation == "horizontal":
            coords = local + np.array([origin, span0])
        else:
            coords = local[:, ::-1] + np.array([span0, origin])
        inp[coords[:, 0], coords[:, 1]] = color
        sprites.append(
            {"coords": coords, "span": interval, "color": color, "bg": bg}
        )

    # Structural and semantic oracle checks: both branches, partial-overlap
    # rejection, actual movement, recoloring, and exact bar preservation.
    fits = [gap0 <= s["span"][0] and s["span"][1] <= gap1 for s in sprites]
    partials = [
        s for s in sprites
        if not (s["span"][1] < gap0 or gap1 < s["span"][0])
        and not (gap0 <= s["span"][0] and s["span"][1] <= gap1)
    ]
    if not any(fits) or all(fits) or not partials:
        return None
    if not any(s["span"][1] - s["span"][0] + 1 <= gap_width for s in partials):
        return None

    expected = _expected_from_plan(
        inp, orientation, bar_index, bar_color, (gap0, gap1), side, sprites
    )
    try:
        actual = main(inp)
    except (AssertionError, IndexError, ValueError):
        return None
    if not np.array_equal(actual, expected) or np.array_equal(actual, inp):
        return None
    if orientation == "horizontal":
        if not np.array_equal(actual[bar_index, :], inp[bar_index, :]):
            return None
    elif not np.array_equal(actual[:, bar_index], inp[:, bar_index]):
        return None
    return inp


def generate_input() -> np.ndarray:
    """Generate a varied, nondegenerate gate scene satisfying main's assumptions."""
    for _ in range(_MAX_GENERATION_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid gate-control input")
