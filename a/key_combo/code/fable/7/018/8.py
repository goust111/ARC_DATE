"""ARC task 8: ballistic color transfer between outlined shells."""

from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


DIRS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _components_of_color(grid: np.ndarray, color: int) -> list[set[tuple[int, int]]]:
    """Return monochromatic 4-connected components in (row, column) coordinates."""
    unseen = {tuple(map(int, p)) for p in np.argwhere(grid == color)}
    result: list[set[tuple[int, int]]] = []
    while unseen:
        seed = unseen.pop()
        comp = {seed}
        queue = deque([seed])
        while queue:
            r, c = queue.popleft()
            for dr, dc in DIRS4:
                q = (r + dr, c + dc)
                if q in unseen:
                    unseen.remove(q)
                    comp.add(q)
                    queue.append(q)
        result.append(comp)
    return result


def _border(r0: int, c0: int, r1: int, c1: int) -> set[tuple[int, int]]:
    return ({(r0, c) for c in range(c0, c1 + 1)} |
            {(r1, c) for c in range(c0, c1 + 1)} |
            {(r, c0) for r in range(r0, r1 + 1)} |
            {(r, c1) for r in range(r0, r1 + 1)})


def _shells(grid: np.ndarray, bg: int) -> list[dict]:
    """Recognize complete or exactly-one-gap rectangular outline components."""
    shells: list[dict] = []
    for color in map(int, np.unique(grid)):
        if color == bg:
            continue
        for cells in _components_of_color(grid, color):
            if len(cells) < 7:
                continue
            rs = [p[0] for p in cells]
            cs = [p[1] for p in cells]
            r0, r1, c0, c1 = min(rs), max(rs), min(cs), max(cs)
            if r1 - r0 < 2 or c1 - c0 < 2:
                continue
            border = _border(r0, c0, r1, c1)
            missing = border - cells
            if cells <= border and len(missing) <= 1 and len(cells) == len(border) - len(missing):
                shells.append({
                    "color": color, "cells": cells, "bbox": (r0, c0, r1, c1),
                    "closed": not missing, "gap": next(iter(missing), None),
                })
    return shells


def _shots(grid: np.ndarray, bg: int, shells: list[dict]) -> list[dict]:
    """Extract the unique internal marker and its gap-defined direction."""
    shots: list[dict] = []
    shell_cells = set().union(*(s["cells"] for s in shells)) if shells else set()
    for shell in shells:
        if shell["closed"]:
            continue
        r0, c0, r1, c1 = shell["bbox"]
        markers = [(r, c) for r in range(r0 + 1, r1)
                   for c in range(c0 + 1, c1)
                   if int(grid[r, c]) != bg and (r, c) not in shell_cells]
        if len(markers) != 1:
            continue
        marker = markers[0]
        gap = shell["gap"]
        if marker[0] == gap[0]:
            direction = (0, -1 if gap[1] == c0 else 1)
        elif marker[1] == gap[1]:
            direction = (-1 if gap[0] == r0 else 1, 0)
        else:
            continue
        shots.append({"shell": shell, "marker": marker,
                      "color": int(grid[marker]), "direction": direction})
    return shots


def _solve(grid: np.ndarray, return_details: bool = False):
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size, "expected a nonempty 2D grid"
    bg = int(Counter(map(int, g.flat)).most_common(1)[0][0])
    shells = _shells(g, bg)
    shots = _shots(g, bg, shells)
    assert shots, "expected at least one one-gap launcher shell with an internal marker"

    occupied: dict[tuple[int, int], int] = {}
    for i, shell in enumerate(shells):
        for p in shell["cells"]:
            occupied[p] = i

    h, w = g.shape
    outcomes = []
    recolor: dict[int, int] = {}
    for shot in shots:
        r, c = shot["marker"]
        dr, dc = shot["direction"]
        hit = None
        while True:
            r, c = r + dr, c + dc
            if not (0 <= r < h and 0 <= c < w):
                break
            if (r, c) in occupied:
                hit = occupied[(r, c)]
                break
        if hit is not None and shells[hit]["closed"]:
            previous = recolor.get(hit)
            assert previous is None or previous == shot["color"], "conflicting bullets hit one shell"
            recolor[hit] = shot["color"]
            outcomes.append((shot, hit, None))
        elif hit is None:
            # The last in-bounds position is the grid-edge resting place.
            edge = (r - dr, c - dc)
            outcomes.append((shot, None, edge))
        else:
            # Any shell is an obstacle, but open launcher shells are never recolored.
            outcomes.append((shot, hit, None))

    out = g.copy()
    for shot, _hit, _edge in outcomes:
        out[shot["marker"]] = bg
    for idx, color in recolor.items():
        for p in shells[idx]["cells"]:
            out[p] = color
    for shot, hit, edge in outcomes:
        if hit is None:
            out[edge] = shot["color"]
    if return_details:
        return out, shells, shots, outcomes, recolor, bg
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fire each marker through its shell gap and transfer its color to the first closed shell hit.

    Input structure:
    - A black background contains several monochromatic rectangular outline shells.
    - Receiver shells are complete; launcher shells have exactly one non-corner gap and
      contain one differently colored marker aligned with that gap by row or column.

    Rule:
    - From every launcher marker, cast an orthogonal ray from the marker through its gap.
    - On meeting the first shell, remove the marker and recolor that whole shell with the
      marker color when it is closed. If no shell is met, move the marker to the grid edge.
    - Launcher outlines retain their original colors and rays leave no painted trail.

    Key insight:
    - The missing border cell supplies both the topological launcher role and the firing
      direction; a receiver's new color is determined by first ray collision, not proximity.
    """
    return _solve(grid)


def _draw_outline(grid: np.ndarray, box: tuple[int, int, int, int], color: int,
                  gap: tuple[int, int] | None = None) -> set[tuple[int, int]]:
    cells = _border(*box)
    if gap is not None:
        cells.remove(gap)
    for p in cells:
        grid[p] = color
    return cells


def _generate_attempt() -> np.ndarray | None:
    """Build three separated lanes: two collisions and one unobstructed flight."""
    height = random.randint(23, 29)
    width = random.randint(25, 30)
    grid = np.zeros((height, width), dtype=int)
    outline_colors = random.sample([1, 5, 8], 3)
    bullet_colors = random.sample([2, 3, 4, 6, 7, 9], 3)
    lane_tops = [1, 8, 15]
    random.shuffle(lane_tops)

    for lane_index, top in enumerate(lane_tops):
        shoot_right = bool(random.getrandbits(1))
        shell_w = random.choice((5, 6))
        shell_h = 5
        marker_row = top + random.randint(1, 3)
        if shoot_right:
            left = random.randint(7, 9) if lane_index == 2 else random.randint(2, 4)
            gap = (marker_row, left + shell_w - 1)
            direction = 1
        else:
            margin = random.randint(7, 9) if lane_index == 2 else random.randint(2, 4)
            left = width - shell_w - margin
            gap = (marker_row, left)
            direction = -1
        launcher_box = (top, left, top + shell_h - 1, left + shell_w - 1)
        _draw_outline(grid, launcher_box, outline_colors[lane_index], gap)
        marker_col = random.randint(left + 1, left + shell_w - 2)
        grid[marker_row, marker_col] = bullet_colors[lane_index]

        if lane_index < 2:  # a closed receiver lies downstream, with random flight length
            target_w = random.choice((3, 4, 5))
            target_top = marker_row - 1
            if direction == 1:
                target_left = left + shell_w + random.randint(2, 5)
            else:
                target_left = left - target_w - random.randint(2, 5)
            if target_left < 1 or target_left + target_w >= width:
                return None
            _draw_outline(grid, (target_top, target_left, target_top + 2,
                                 target_left + target_w - 1),
                          outline_colors[(lane_index + 1) % 3])
        else:
            # A nearby closed distractor is behind the shot, ruling out nearest-shell shortcuts.
            target_w = 3
            target_top = marker_row - 1
            if direction == 1:
                target_left = left - target_w - 2
            else:
                target_left = left + shell_w + 2
            if target_left < 1 or target_left + target_w >= width:
                return None
            _draw_outline(grid, (target_top, target_left, target_top + 2,
                                 target_left + target_w - 1),
                          outline_colors[0])

    # Transposition makes all four orthogonal firing directions occur across samples.
    if random.random() < 0.5:
        grid = grid.T.copy()
    if random.random() < 0.5:
        grid = np.fliplr(grid).copy()
    if random.random() < 0.5:
        grid = np.flipud(grid).copy()

    try:
        out, shells, shots, outcomes, recolor, bg = _solve(grid, True)
    except (AssertionError, IndexError):
        return None
    hits = [(shot, hit) for shot, hit, edge in outcomes
            if hit is not None and shells[hit]["closed"]]
    misses = [(shot, edge) for shot, hit, edge in outcomes if hit is None]
    if len(shots) != 3 or len(hits) != 2 or len(misses) != 1:
        return None
    if len({hit for _shot, hit in hits}) != 2 or len(recolor) != 2:
        return None
    if np.array_equal(out, grid) or bg != 0:
        return None
    # Semantic checks: topology, transfer color, unchanged launchers, and edge stop.
    for shot in shots:
        if any(int(out[p]) != shot["shell"]["color"] for p in shot["shell"]["cells"]):
            return None
    for shot, hit in hits:
        if any(int(out[p]) != shot["color"] for p in shells[hit]["cells"]):
            return None
    for shot, edge in misses:
        if int(out[edge]) != shot["color"] or edge[0] not in (0, out.shape[0] - 1) and edge[1] not in (0, out.shape[1] - 1):
            return None
    return grid.astype(int)


def generate_input() -> np.ndarray:
    """Generate varied, nondegenerate examples covering hit and edge-stop branches."""
    for _ in range(300):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid ballistic-shell grid")
