"""ARC task: momentum relay between separated lane blocks."""

from __future__ import annotations

import random

import numpy as np


def _background(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _wall_side(grid: np.ndarray, bg: int) -> tuple[str, int]:
    """Return the unique full non-background edge whose color occurs nowhere else."""
    h, w = grid.shape
    edges = (
        ("top", grid[0, :], w),
        ("bottom", grid[-1, :], w),
        ("left", grid[:, 0], h),
        ("right", grid[:, -1], h),
    )
    found: list[tuple[str, int]] = []
    for side, edge, required_count in edges:
        color = int(edge[0])
        if color != bg and bool(np.all(edge == color)):
            if int(np.count_nonzero(grid == color)) == required_count:
                found.append((side, color))
    assert len(found) == 1, "expected one complete, uniquely colored edge wall"
    return found[0]


def _to_top(grid: np.ndarray, side: str) -> np.ndarray:
    if side == "top":
        return grid.copy()
    if side == "bottom":
        return np.flipud(grid).copy()
    if side == "left":
        return grid.T.copy()
    if side == "right":
        return np.flipud(grid.T).copy()
    raise ValueError(side)


def _from_top(grid: np.ndarray, side: str) -> np.ndarray:
    if side == "top":
        return grid.copy()
    if side == "bottom":
        return np.flipud(grid).copy()
    if side == "left":
        return grid.T.copy()
    if side == "right":
        return np.flipud(grid).T.copy()
    raise ValueError(side)


def _marker_color(top: np.ndarray, bg: int, wall_color: int) -> int:
    """Identify the common singleton immediately outside each farthest block."""
    h, w = top.shape
    candidates: list[int] = []
    for color in np.unique(top):
        color = int(color)
        if color in (bg, wall_color):
            continue
        coords = np.argwhere(top == color)
        if len(coords) == 0:
            continue
        valid = True
        for r, c in coords:
            r, c = int(r), int(c)
            # A marker is one cell thick along its lane, has a block immediately
            # inward, and is the outermost foreground cell of that lane.
            if r <= 1 or int(top[r - 1, c]) in (bg, wall_color, color):
                valid = False
                break
            if r + 1 < h and int(top[r + 1, c]) != bg:
                valid = False
                break
            if np.any((top[r + 1 :, c] != bg)):
                valid = False
                break
        if valid:
            candidates.append(color)
    assert len(candidates) == 1, "impulse marker color must be structurally unique"
    return candidates[0]


def _runs_in_lane(top: np.ndarray, col: int, bg: int, marker: int) -> list[tuple[int, int, int]]:
    """Return (near, far, color) runs in increasing distance from the top wall."""
    h = top.shape[0]
    runs: list[tuple[int, int, int]] = []
    r = 1
    while r < h:
        color = int(top[r, col])
        if color in (bg, marker):
            r += 1
            continue
        start = r
        while r + 1 < h and int(top[r + 1, col]) == color:
            r += 1
        runs.append((start, r, color))
        r += 1
    return runs


def main(grid: np.ndarray) -> np.ndarray:
    """Relay an impulse through each marked lane, sliding its blocks one at a time toward the edge wall.

    Input structure:
    - The dominant background surrounds a complete one-pixel-thick edge wall whose
      color occurs nowhere else; the wall side defines the lane direction.
    - Perpendicular lanes contain separated solid short blocks, with distinct block
      colors within a lane.  A common singleton marker touches the outer side of
      the farthest block in each triggered lane.

    Rule:
    - Erase every impulse marker and leave unmarked lanes and the wall unchanged.
    - In a marked lane, process blocks from farthest to nearest: each slides toward
      the wall until it hits the next block, passing the impulse onward; the nearest
      block finally slides against the wall.

    Key insight:
    - The blocks do not move as a packed group: each block consumes its own original
      near-side gap, so the old gap sequence is relayed one position away from the wall.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and min(g.shape) >= 3
    bg = _background(g)
    side, wall_color = _wall_side(g, bg)
    top = _to_top(g, side)
    marker = _marker_color(top, bg, wall_color)

    out = top.copy()
    out[out == marker] = bg
    marked_cols = sorted(set(map(int, np.argwhere(top == marker)[:, 1])))

    for c in marked_cols:
        marker_rows = np.flatnonzero(top[:, c] == marker)
        assert len(marker_rows) == 1
        marker_r = int(marker_rows[0])
        runs = _runs_in_lane(top, c, bg, marker)
        assert len(runs) >= 2, "a relay lane needs at least two blocks"
        assert marker_r == runs[-1][1] + 1, "marker must touch the farthest block"
        assert len({color for _a, _b, color in runs}) == len(runs), (
            "block colors must be distinct within a lane"
        )

        for start, end, _color in runs:
            out[start : end + 1, c] = bg

        previous_original_end = 0  # wall occupies distance zero
        for start, end, color in runs:
            gap = start - previous_original_end - 1
            assert gap >= 0
            new_start = start - gap
            length = end - start + 1
            out[new_start : new_start + length, c] = color
            previous_original_end = end

    return _from_top(out, side)


_COLORS = tuple(range(10))
_SIDES = ("top", "bottom", "left", "right")


def _render_canonical() -> tuple[np.ndarray, np.ndarray, dict]:
    """Build a top-wall scene and an independent descriptor-based expected result."""
    # Seventeen rows are the minimum for the four-block fallback layout:
    # 4 two-cell blocks + gaps 1,2,1,2 + marker + outer background.
    h = random.randint(17, 25)
    w = random.randint(9, 17)
    bg, wall, marker = random.sample(_COLORS, 3)
    available = [c for c in _COLORS if c not in (bg, wall, marker)]

    inp = np.full((h, w), bg, dtype=int)
    expected = inp.copy()
    inp[0, :] = wall
    expected[0, :] = wall

    lane_pool = list(range(1, w - 1, 2))
    random.shuffle(lane_pool)
    lane_count = min(len(lane_pool), random.randint(3, 6))
    lanes = sorted(lane_pool[:lane_count])
    marked_count = random.randint(2, lane_count - 1)
    marked = set(random.sample(lanes, marked_count))

    descriptors: dict[int, list[tuple[int, int, int]]] = {}
    varied_shift = False
    retained_gap = False
    for c in lanes:
        nblocks = random.randint(2, min(4, len(available)))
        lengths = [random.randint(2, 3) for _ in range(nblocks)]
        # Every gap is visible and active.  Marked lanes additionally receive a
        # deliberately nonconstant sequence to refute equal translation.
        gaps = [random.randint(1, 3) for _ in range(nblocks)]
        if c in marked and len(set(gaps)) == 1:
            gaps[-1] = 1 + (gaps[-1] % 3)
        needed = sum(lengths) + sum(gaps) + 2  # marker plus outer background
        if needed > h - 1:
            lengths = [2] * nblocks
            gaps = [1 + (i % 2) for i in range(nblocks)]
        assert sum(lengths) + sum(gaps) + 2 <= h - 1

        colors = random.sample(available, nblocks)
        runs: list[tuple[int, int, int]] = []
        previous_end = 0
        for gap, length, color in zip(gaps, lengths, colors):
            start = previous_end + gap + 1
            end = start + length - 1
            inp[start : end + 1, c] = color
            runs.append((start, end, color))
            previous_end = end
        descriptors[c] = runs

        if c in marked:
            inp[previous_end + 1, c] = marker
            expected[:, c] = bg
            expected[0, c] = wall
            previous_original_end = 0
            shifts = []
            for start, end, color in runs:
                gap = start - previous_original_end - 1
                shifts.append(gap)
                length = end - start + 1
                expected[start - gap : start - gap + length, c] = color
                previous_original_end = end
            varied_shift |= len(set(shifts)) > 1
            # The first old wall gap becomes the first final inter-block gap.
            retained_gap |= gaps[0] > 0
        else:
            expected[:, c] = inp[:, c]

    meta = {
        "bg": bg,
        "wall": wall,
        "marker": marker,
        "lanes": lanes,
        "marked": marked,
        "runs": descriptors,
        "varied_shift": varied_shift,
        "retained_gap": retained_gap,
    }
    return inp, expected, meta


def _semantic_check(inp: np.ndarray, out: np.ndarray, expected: np.ndarray, meta: dict) -> bool:
    """Check task semantics and anti-shortcut properties, not merely solver closure."""
    if not np.array_equal(out, expected):
        return False
    if np.any(out == int(meta["marker"])):
        return False
    if not meta["varied_shift"] or not meta["retained_gap"]:
        return False
    bg = int(meta["bg"])
    for c in meta["lanes"]:
        if c not in meta["marked"] and not np.array_equal(inp[:, c], out[:, c]):
            return False
        if c in meta["marked"]:
            # A compact-to-wall shortcut would contain no background between blocks.
            nonbg = np.flatnonzero((out[:, c] != bg) & (out[:, c] != meta["wall"]))
            if len(nonbg) == 0 or not np.any(out[1 : int(nonbg[-1]), c] == bg):
                return False
    return not np.array_equal(inp, out)


def generate_input() -> np.ndarray:
    """Generate varied, unambiguous scenes covering all wall directions."""
    for _ in range(300):
        top_inp, top_expected, meta = _render_canonical()
        side = random.choice(_SIDES)
        inp = _from_top(top_inp, side)
        expected = _from_top(top_expected, side)
        try:
            actual = main(inp)
        except (AssertionError, ValueError):
            continue
        # Check detailed semantics in the wall-relative coordinate system.
        actual_top = _to_top(actual, side)
        if _semantic_check(top_inp, actual_top, top_expected, meta) and np.array_equal(actual, expected):
            assert max(inp.shape) <= 30 and max(actual.shape) <= 30
            return inp
    raise RuntimeError("failed to generate a valid momentum-relay input")
