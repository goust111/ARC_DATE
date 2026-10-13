"""ARC task: live-gate gravity sieve."""

from __future__ import annotations

import random
from collections import Counter, deque

import numpy as np


def _components_by_color(grid: np.ndarray, excluded: set[int]) -> list[dict]:
    """Return monochromatic 4-components using canonical (row, column) coords."""
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    result: list[dict] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if seen[r, c] or color in excluded:
                continue
            todo = deque([(r, c)])
            seen[r, c] = True
            cells = []
            while todo:
                rr, cc = todo.popleft()
                cells.append((rr, cc))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = rr + dr, cc + dc
                    if (0 <= nr < h and 0 <= nc < w and not seen[nr, nc]
                            and int(grid[nr, nc]) == color):
                        seen[nr, nc] = True
                        todo.append((nr, nc))
            coords = np.asarray(cells, dtype=int)
            r0, c0 = coords.min(axis=0)
            r1, c1 = coords.max(axis=0)
            result.append({
                "color": color,
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
                "area": len(cells),
            })
    return result


def _wall_rotation(grid: np.ndarray) -> int:
    """Number of counter-clockwise quarter turns that puts the wall at bottom."""
    black = 0
    edges = [
        bool(np.all(grid[-1, :] == black)),  # bottom
        bool(np.all(grid[:, 0] == black)),   # left
        bool(np.all(grid[0, :] == black)),   # top
        bool(np.all(grid[:, -1] == black)),  # right
    ]
    assert sum(edges) == 1, "exactly one edge must be a solid black wall"
    return edges.index(True)


def _gap_runs(row: np.ndarray, bg: int) -> list[tuple[int, int]]:
    runs = []
    start = None
    for c, value in enumerate(row):
        if int(value) == bg and start is None:
            start = c
        if int(value) != bg and start is not None:
            runs.append((start, c - 1))
            start = None
    if start is not None:
        runs.append((start, len(row) - 1))
    return runs


def _solve_down(grid: np.ndarray) -> np.ndarray:
    """Solve a scene already normalized to downward gravity."""
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = Counter(map(int, g.ravel())).most_common(1)[0][0]
    assert bg != 0, "the black wall and bars must differ from the background"
    assert np.all(g[-1, :] == 0)

    objects = _components_by_color(g, {0, bg})
    assert len(objects) >= 2, "expected several colored rectangles"
    for obj in objects:
        r0, c0, r1, c1 = obj["bbox"]
        assert obj["area"] == (r1 - r0 + 1) * (c1 - c0 + 1)
        assert np.all(g[r0:r1 + 1, c0:c1 + 1] == obj["color"])

    # Near-to-far is measured along the stated gravity axis, from the wall.
    bottoms = [obj["bbox"][2] for obj in objects]
    assert len(bottoms) == len(set(bottoms)), "fall order must be uniquely visible"
    objects.sort(key=lambda obj: -obj["bbox"][2])

    out = g.copy()
    for obj in objects:
        r0, c0, r1, c1 = obj["bbox"]
        out[r0:r1 + 1, c0:c1 + 1] = bg

    fixed = out == 0
    occupied = np.zeros((h, w), dtype=bool)
    sealed = np.zeros((h, w), dtype=bool)
    bar_rows = [r for r in range(h - 1) if np.any(fixed[r])]
    assert 1 <= len(bar_rows) <= 2, "expected one or two horizontal bars"
    gaps = {r: _gap_runs(out[r], bg) for r in bar_rows}
    assert all(gaps[r] for r in bar_rows), "every bar must have a visible gap"

    for obj in objects:
        r0, c0, r1, c1 = obj["bbox"]
        oh, ow = r1 - r0 + 1, c1 - c0 + 1
        top = r0
        while top + oh < h:
            nr0, nr1 = top + 1, top + oh
            footprint_blocked = (
                fixed[nr0:nr1 + 1, c0:c0 + ow].any()
                or sealed[nr0:nr1 + 1, c0:c0 + ow].any()
                or occupied[nr0:nr1 + 1, c0:c0 + ow].any()
            )
            if footprint_blocked:
                break
            top += 1

        out[top:top + oh, c0:c0 + ow] = obj["color"]
        occupied[top:top + oh, c0:c0 + ow] = True

        # A rectangle arrested directly on a bar closes every gap mouth it covers.
        bottom = top + oh - 1
        if bottom + 1 in gaps:
            br = bottom + 1
            for a, b in gaps[br]:
                if max(a, c0) <= min(b, c0 + ow - 1):
                    sealed[br, a:b + 1] = True
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Drop colored rectangles through live gaps toward the solid-wall floor.

    Input structure:
    A non-black background contains one completely black boundary wall, one or
    two black bars parallel to that wall with visible gaps, and several separate
    solid colored rectangles.  The wall uniquely specifies the gravity axis.

    Rule:
    Rotate the wall mentally to the floor and process rectangles from nearest
    to farthest from it, keeping each cross-gravity span fixed.  A rectangle can
    cross a bar only when its whole span lies in an open gap; otherwise it rests
    immediately before the bar.  Settled rectangles block later ones, and one
    resting across a gap seals that gap.  Preserve the wall and bars exactly.

    Key insight:
    Gap admission is interval containment in the wall-defined coordinate frame,
    but it is stateful: an earlier landing can turn a formerly adequate gap into
    an obstacle for a later, narrower rectangle.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and max(g.shape) <= 30
    k = _wall_rotation(g)
    normalized = np.rot90(g, k)
    solved = _solve_down(normalized)
    return np.rot90(solved, -k).copy()


def _bbox_for_color(grid: np.ndarray, color: int) -> tuple[int, int, int, int]:
    pts = np.argwhere(grid == color)
    assert len(pts) > 0
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _attempt() -> np.ndarray | None:
    """Construct a downward scene with a guaranteed live-gate counterexample."""
    h = random.randint(23, 30)
    w = random.randint(14, 22)
    bg = random.choice(list(range(1, 10)))
    palette = [c for c in range(1, 10) if c != bg]
    random.shuffle(palette)
    passer_color, sealer_color, follower_color = palette[:3]
    two_bars = random.choice((False, True))

    top_bar = random.randint(13, min(17, h - 8))
    lower_bar = random.randint(top_bar + 4, h - 4) if two_bars else None
    gate_w = random.randint(4, min(6, w - 6))
    gate0 = random.randint(2, w - gate_w - 2)
    gate1 = gate0 + gate_w - 1

    g = np.full((h, w), bg, dtype=int)
    g[-1, :] = 0
    g[top_bar, :] = 0
    g[top_bar, gate0:gate1 + 1] = bg

    passer_w = 2 if two_bars else random.choice((1, 2))
    passer_c = gate0
    follower_w = random.choice((1, 2))
    follower_c = gate1 - follower_w + 1
    if two_bars:
        assert lower_bar is not None
        g[lower_bar, :] = 0
        # Narrow, offset lower gap rejects the first passer after the top gate.
        g[lower_bar, passer_c:passer_c + 1] = bg

    heights = [random.choice((1, 2)) for _ in range(3)]
    # Allocate distinct leading edges from near to far: passer, sealer, follower.
    passer_bottom = top_bar - random.choice((2, 3))
    passer_top = passer_bottom - heights[0] + 1
    sealer_bottom = passer_top - 2
    sealer_top = sealer_bottom - heights[1] + 1
    follower_bottom = sealer_top - 2
    follower_top = follower_bottom - heights[2] + 1
    if follower_top < 1:
        return None

    g[passer_top:passer_bottom + 1, passer_c:passer_c + passer_w] = passer_color
    g[sealer_top:sealer_bottom + 1, gate0 - 1:gate1 + 2] = sealer_color
    g[follower_top:follower_bottom + 1, follower_c:follower_c + follower_w] = follower_color

    # Optional independent rectangle broadens object count and pass/fail branches.
    extra_colors = palette[3:]
    if random.random() < 0.55 and extra_colors:
        ec = extra_colors[0]
        ew = random.choice((1, 2))
        candidates = [c for c in range(1, w - ew - 1)
                      if c + ew - 1 < gate0 - 1 or c > gate1 + 1]
        if candidates:
            er = random.randint(1, max(1, follower_top - 2))
            if er not in {passer_bottom, sealer_bottom, follower_bottom}:
                ec0 = random.choice(candidates)
                g[er, ec0:ec0 + ew] = ec

    try:
        out = _solve_down(g)
        sb = _bbox_for_color(out, sealer_color)
        fb = _bbox_for_color(out, follower_color)
        pb = _bbox_for_color(out, passer_color)
    except (AssertionError, ValueError):
        return None

    # Semantic oracle: pass, seal, then stack; each key step changes the answer.
    if sb[2] != top_bar - 1 or fb[2] != sb[0] - 1 or pb[0] <= top_bar:
        return None
    counterfactual = g.copy()
    counterfactual[counterfactual == sealer_color] = bg
    try:
        cf = _solve_down(counterfactual)
        if _bbox_for_color(cf, follower_color)[0] <= top_bar:
            return None
    except AssertionError:
        return None
    if np.array_equal(g, out):
        return None

    # Random orientation is applied last, preserving the exact wall-based frame.
    return np.rot90(g, random.randrange(4)).copy()


def generate_input() -> np.ndarray:
    """Generate varied, non-degenerate live-gate gravity scenes."""
    for _ in range(400):
        grid = _attempt()
        if grid is not None:
            # Final independent loading-path behavior and ARC-size checks.
            solved = main(grid)
            if grid.ndim == solved.ndim == 2 and max(grid.shape) <= 30:
                return grid
    raise RuntimeError("failed to generate a valid live-gate scene")
