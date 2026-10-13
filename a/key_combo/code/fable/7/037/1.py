"""ARC task: color only intersections of invisible nearest-edge L rays."""

from __future__ import annotations

import random

import numpy as np


Coord = tuple[int, int]
Marker = tuple[int, int, int]


def _edge_directions(r: int, c: int, h: int, w: int) -> tuple[int, int]:
    """Return the row and column step toward each strictly nearer edge."""
    up, down = r, h - 1 - r
    left, right = c, w - 1 - c
    assert up != down, "a marker must not tie between top and bottom"
    assert left != right, "a marker must not tie between left and right"
    return (-1 if up < down else 1), (-1 if left < right else 1)


def _on_vertical_ray(source: Marker, row: int, h: int, w: int) -> bool:
    r, c, _color = source
    dr, _dc = _edge_directions(r, c, h, w)
    return row <= r if dr < 0 else row >= r


def _on_horizontal_ray(source: Marker, col: int, h: int, w: int) -> bool:
    r, c, _color = source
    _dr, dc = _edge_directions(r, c, h, w)
    return col <= c if dc < 0 else col >= c


def _intersection_colors(markers: list[Marker], h: int, w: int) -> dict[Coord, int]:
    """Infer all cross-source intersections and their unambiguous nearer colors."""
    proposals: dict[Coord, list[int]] = {}
    for vi, vertical in enumerate(markers):
        vr, vc, vcolor = vertical
        for hi, horizontal in enumerate(markers):
            if vi == hi:
                continue
            hr, hc, hcolor = horizontal
            point = (hr, vc)
            if not _on_vertical_ray(vertical, hr, h, w):
                continue
            if not _on_horizontal_ray(horizontal, vc, h, w):
                continue

            vertical_distance = abs(hr - vr)
            horizontal_distance = abs(vc - hc)
            assert vertical_distance != horizontal_distance, (
                "an intersection must not be equidistant from its two markers"
            )
            winner = vcolor if vertical_distance < horizontal_distance else hcolor
            proposals.setdefault(point, []).append(winner)

    result: dict[Coord, int] = {}
    for point, colors in proposals.items():
        assert len(set(colors)) == 1, "overlapping pairwise intersections have conflicting colors"
        result[point] = colors[0]
    return result


def main(grid: np.ndarray) -> np.ndarray:
    """Color only cross-source intersections of invisible nearest-edge L rays.

    Input structure:
    A black background contains isolated colored single-pixel markers.  Each marker
    is strictly nearer one horizontal boundary and one vertical boundary, and every
    relevant pair has unequal Manhattan distances to its intersection.

    Rule:
    Imagine, but do not draw, a vertical ray from every marker to its nearer top or
    bottom edge and a horizontal ray to its nearer left or right edge.  Whenever a
    marker's vertical ray crosses a different marker's horizontal ray, color that
    cell with the color of the closer of those two markers; copy everything else.

    Key insight:
    The visible pixels are generators of hidden edge-facing L shapes: only crossings
    between rays from different generators appear, never the rays themselves.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a two-dimensional grid"
    h, w = map(int, g.shape)
    assert h <= 30 and w <= 30
    assert np.count_nonzero(g == 0) > g.size // 2, "black must be the background"

    coords = np.argwhere(g != 0)
    assert len(coords) >= 2, "expected at least two markers"
    markers = [(int(r), int(c), int(g[r, c])) for r, c in coords]
    for r, c, _color in markers:
        _edge_directions(r, c, h, w)

    out = g.copy()
    for (r, c), color in _intersection_colors(markers, h, w).items():
        out[r, c] = color
    return out


def _isolated(points: list[Coord]) -> bool:
    return all(
        max(abs(r1 - r2), abs(c1 - c2)) > 1
        for i, (r1, c1) in enumerate(points)
        for r2, c2 in points[i + 1 :]
    )


def _far_edge_intersections(markers: list[Marker], h: int, w: int) -> set[Coord]:
    """Counterfactual intersections if every ray incorrectly chose the far edge."""
    result: set[Coord] = set()
    for vi, (vr, vc, _vcolor) in enumerate(markers):
        vdr, _ = _edge_directions(vr, vc, h, w)
        for hi, (hr, hc, _hcolor) in enumerate(markers):
            if vi == hi:
                continue
            _, hdc = _edge_directions(hr, hc, h, w)
            vertical_ok = hr >= vr if vdr < 0 else hr <= vr
            horizontal_ok = vc >= hc if hdc < 0 else vc <= hc
            if vertical_ok and horizontal_ok:
                result.add((hr, vc))
    return result


def _valid_generated(grid: np.ndarray) -> bool:
    """Check semantic coverage, uniqueness, clarity, and anti-shortcut properties."""
    h, w = grid.shape
    coords = [tuple(map(int, rc)) for rc in np.argwhere(grid != 0)]
    if not (4 <= len(coords) <= 7) or not _isolated(coords):
        return False
    if len({r for r, _ in coords}) != len(coords) or len({c for _, c in coords}) != len(coords):
        return False

    markers = [(r, c, int(grid[r, c])) for r, c in coords]
    try:
        intersections = _intersection_colors(markers, h, w)
    except AssertionError:
        return False
    marker_cells = set(coords)
    if marker_cells & set(intersections):
        return False
    if not (3 <= len(intersections) <= 10):
        return False
    if len(set(intersections.values())) < 2:
        return False

    # Both possible distance winners must occur, so coloring cannot be replaced
    # by always choosing the vertical or always choosing the horizontal source.
    vertical_wins = horizontal_wins = False
    for vi, (vr, vc, vcolor) in enumerate(markers):
        for hi, (hr, hc, hcolor) in enumerate(markers):
            if vi == hi or (hr, vc) not in intersections:
                continue
            if not (_on_vertical_ray(markers[vi], hr, h, w) and
                    _on_horizontal_ray(markers[hi], vc, h, w)):
                continue
            dv, dh = abs(hr - vr), abs(vc - hc)
            vertical_wins |= dv < dh and intersections[(hr, vc)] == vcolor
            horizontal_wins |= dh < dv and intersections[(hr, vc)] == hcolor
    if not (vertical_wins and horizontal_wins):
        return False

    actual_cells = set(intersections)
    if actual_cells == _far_edge_intersections(markers, h, w):
        return False

    out = main(grid)
    changed = set(map(tuple, np.argwhere(out != grid)))
    marker_rows = np.array([r for r, _c in marker_cells], dtype=int)
    marker_cols = np.array([c for _r, c in marker_cells], dtype=int)
    markers_preserved = np.all(
        out[marker_rows, marker_cols] == grid[marker_rows, marker_cols]
    )
    return changed == actual_cells and bool(markers_preserved)


def generate_input() -> np.ndarray:
    """Generate varied, semantically checked marker configurations."""
    for _ in range(3000):
        h = random.randint(12, 22)
        w = random.randint(12, 22)
        count = random.randint(4, 7)
        colors = random.sample(range(1, 10), count)

        # Seed all four nearest-edge direction combinations, then add random ones.
        quadrants = [(0, 0), (0, 1), (1, 0), (1, 1)]
        quadrants += [random.choice(quadrants) for _ in range(count - 4)]
        random.shuffle(quadrants)

        points: list[Coord] = []
        ok = True
        for row_side, col_side in quadrants:
            row_candidates = [r for r in range(2, h - 2) if (2 * r < h - 1) == (row_side == 0) and 2 * r != h - 1]
            col_candidates = [c for c in range(2, w - 2) if (2 * c < w - 1) == (col_side == 0) and 2 * c != w - 1]
            candidates = [
                (r, c) for r in row_candidates for c in col_candidates
                if r not in {pr for pr, _ in points}
                and c not in {pc for _, pc in points}
                and _isolated(points + [(r, c)])
            ]
            if not candidates:
                ok = False
                break
            points.append(random.choice(candidates))
        if not ok:
            continue

        grid = np.zeros((h, w), dtype=int)
        for (r, c), color in zip(points, colors):
            grid[r, c] = color
        if _valid_generated(grid):
            return grid
    raise RuntimeError("failed to generate a non-degenerate invisible-intersection input")
