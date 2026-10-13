from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np


_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import Color, find_components, get_bg  # noqa: E402


DIRS4 = [(-1, 0), (0, 1), (1, 0), (0, -1)]  # up, right, down, left
MARKER_COLORS = (Color.BLUE, Color.RED, Color.GREEN, Color.YELLOW, Color.PINK, Color.ORANGE)


_ARROW_BASES_UP: dict[str, list[tuple[int, int]]] = {
    "filled4": [(0, 0), (1, -1), (1, 0), (1, 1)],
    "pike5": [(0, 0), (1, -1), (1, 0), (1, 1), (2, 0)],
    "dart5": [(0, 0), (1, 0), (2, -1), (2, 0), (2, 1)],
}


def _rot90(dr: int, dc: int) -> tuple[int, int]:
    return dc, -dr


def _rotate_offsets(offsets: list[tuple[int, int]], turns: int) -> list[tuple[int, int]]:
    out = [(int(dr), int(dc)) for dr, dc in offsets]
    for _ in range(int(turns) % 4):
        out = [_rot90(dr, dc) for dr, dc in out]
    return out


ARROW_TEMPLATES: dict[str, dict[int, list[tuple[int, int]]]] = {
    name: {d: _rotate_offsets(offsets, d) for d in range(4)} for name, offsets in _ARROW_BASES_UP.items()
}


def _find_arrow(grid: np.ndarray) -> tuple[tuple[int, int], int, str]:
    gray = find_components(grid, bg=Color.BLACK, colors=[Color.GRAY], connectivity=4, monochromatic=True)
    assert len(gray) == 1, "expected exactly one gray arrow component"
    pts = {tuple(map(int, coord)) for coord in gray[0]["coords"]}

    best: tuple[int, int, tuple[int, int], int, str] | None = None
    for name, by_dir in ARROW_TEMPLATES.items():
        for direction, offsets in by_dir.items():
            for hr, hc in pts:
                placed = {(hr + dr, hc + dc) for dr, dc in offsets}
                if placed == pts:
                    return (int(hr), int(hc)), int(direction), name
                overlap = len(placed & pts)
                score = (overlap, -len(pts ^ placed))
                if best is None or score > best[:2]:
                    best = (score[0], score[1], (int(hr), int(hc)), int(direction), name)

    assert best is not None and best[0] >= 3, "could not identify arrow tip and facing"
    return best[2], best[3], best[4]


def _extract_markers(grid: np.ndarray) -> dict[int, tuple[int, int]]:
    markers: dict[int, tuple[int, int]] = {}
    comps = find_components(grid, bg=Color.BLACK, connectivity=4, monochromatic=True)
    for comp in comps:
        color = int(comp["color"])
        if color in (Color.BLACK, Color.GRAY):
            continue
        assert comp["area"] == 1, "markers must be single pixels"
        coord = tuple(map(int, comp["coords"][0]))
        assert color not in markers, "marker colors must be unique"
        markers[color] = coord
    assert 3 <= len(markers) <= 6, "expected 3..6 markers"
    return markers


def _direction_hits(
    pos: tuple[int, int],
    markers: dict[int, tuple[int, int]],
) -> dict[int, list[tuple[int, int, tuple[int, int]]]]:
    r, c = map(int, pos)
    hits: dict[int, list[tuple[int, int, tuple[int, int]]]] = {d: [] for d in range(4)}
    for color, (rr, cc) in markers.items():
        rr = int(rr)
        cc = int(cc)
        if cc == c and rr < r:
            hits[0].append((r - rr, int(color), (rr, cc)))
        elif rr == r and cc > c:
            hits[1].append((cc - c, int(color), (rr, cc)))
        elif cc == c and rr > r:
            hits[2].append((rr - r, int(color), (rr, cc)))
        elif rr == r and cc < c:
            hits[3].append((c - cc, int(color), (rr, cc)))
    for d in range(4):
        hits[d].sort()
    return hits


def _simulate_consumption(
    start: tuple[int, int],
    facing: int,
    markers: dict[int, tuple[int, int]],
) -> tuple[list[int], list[dict[str, object]]]:
    remaining = {int(color): tuple(map(int, coord)) for color, coord in markers.items()}
    pos = tuple(map(int, start))
    direction = int(facing)
    order: list[int] = []
    steps: list[dict[str, object]] = []

    while remaining:
        hits = _direction_hits(pos, remaining)
        active_dirs = [d for d in range(4) if hits[d]]
        assert active_dirs, "simulation got stuck before consuming all markers"

        if hits[direction]:
            chosen_dir = direction
            used_forward = True
        else:
            assert len(active_dirs) == 1, "turning must remain unambiguous when forward is empty"
            chosen_dir = active_dirs[0]
            used_forward = False

        dist, color, target = hits[chosen_dir][0]
        steps.append(
            {
                "pos": pos,
                "facing": direction,
                "active_dirs": tuple(active_dirs),
                "chosen_dir": chosen_dir,
                "used_forward": used_forward,
                "distance": int(dist),
                "target": target,
                "color": int(color),
            }
        )
        order.append(int(color))
        del remaining[int(color)]
        pos = target
        direction = int(chosen_dir)

    return order, steps


def _rings_from_order(order: list[int]) -> np.ndarray:
    n = len(order)
    size = 2 * n + 1
    center = n
    rr, cc = np.indices((size, size))
    cheb = np.maximum(np.abs(rr - center), np.abs(cc - center))
    out = np.zeros((size, size), dtype=int)
    for layer, color in enumerate(order):
        out[cheb == layer] = int(color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Consume colored markers in the arrow's deterministic search order, then map that order to concentric square rings.

    Input structure:
    - Black background with one gray 4-connected triangular arrow sprite whose tip marks the starting cell and facing.
    - 3 to 6 single-pixel colored markers, all in different colors chosen from the standard ARC palette.
    - Markers are spaced so the arrow can repeatedly find one row/column-aligned target at a time.

    Rule:
    - Read the arrow tip and its facing direction.
    - From the current tip cell, inspect row/column half-lines containing remaining markers.
    - If the current facing half-line contains markers, move to and consume the nearest one there.
    - Otherwise the forward line is empty, and exactly one other half-line contains markers; move to its nearest marker and turn to that direction.
    - Record the consumed colors as c1..cN, then output a (2N+1)x(2N+1) square where Chebyshev layer 0 is c1, layer 1 is c2, ..., layer N-1 is cN, and the outer layer N stays black.

    Key insight:
    The marker coordinates matter only through the dynamic consumption order induced by the arrow's facing-aware line search; that 1D order is then remapped into 2D square rings.
    """
    g = np.asarray(grid, dtype=int)
    _bg = int(get_bg(g))
    assert _bg == Color.BLACK, "expected black background"

    start, facing, _shape = _find_arrow(g)
    markers = _extract_markers(g)
    order, _steps = _simulate_consumption(start, facing, markers)
    return _rings_from_order(order)


def _canonical_points(kind: str, n: int) -> list[tuple[int, int]]:
    assert 3 <= n <= 6
    if kind == "return":
        a1 = random.randint(3, 7)
        b1 = random.randint(3, 7)
        points = [(0, a1), (b1, a1), (b1, 0)]
        if n >= 4:
            b2 = b1 + random.randint(2, 6)
            points.append((b2, 0))
        if n >= 5:
            a2 = random.randint(3, 7)
            while a2 == a1:
                a2 = random.randint(3, 7)
            points.append((b2, a2))
        if n >= 6:
            b3 = b2 + random.randint(2, 6)
            points.append((b3, a2))
        return points[:n]

    if kind == "stair":
        a1 = random.randint(3, 6)
        a2 = a1 + random.randint(2, 5)
        a3 = a2 + random.randint(2, 5)
        b1 = random.randint(3, 6)
        b2 = b1 + random.randint(2, 5)
        b3 = b2 + random.randint(2, 5)
        points = [(0, a1), (b1, a1), (b1, a2), (b2, a2), (b2, a3), (b3, a3)]
        return points[:n]

    raise ValueError(f"unknown path kind: {kind}")


def _rotate_points(points: list[tuple[int, int]], facing: int) -> list[tuple[int, int]]:
    turns = (int(facing) - 1) % 4
    return [_rotate_offsets([pt], turns)[0] for pt in points]


def _paint_scene(
    points: list[tuple[int, int]],
    arrow_name: str,
    facing: int,
    colors: list[int],
) -> np.ndarray:
    arrow_offsets = ARROW_TEMPLATES[arrow_name][facing]
    all_coords = [(0, 0), *points, *arrow_offsets]
    min_r = min(r for r, _c in all_coords)
    max_r = max(r for r, _c in all_coords)
    min_c = min(c for _r, c in all_coords)
    max_c = max(c for _r, c in all_coords)

    top = random.randint(1, 4)
    left = random.randint(1, 4)
    bottom = random.randint(1, 4)
    right = random.randint(1, 4)

    height = (max_r - min_r + 1) + top + bottom
    width = (max_c - min_c + 1) + left + right
    assert height <= 30 and width <= 30

    shift_r = top - min_r
    shift_c = left - min_c
    grid = np.zeros((height, width), dtype=int)

    for dr, dc in arrow_offsets:
        grid[shift_r + dr, shift_c + dc] = Color.GRAY
    for color, (pr, pc) in zip(colors, points):
        grid[shift_r + pr, shift_c + pc] = int(color)
    return grid


def _validate_scene(
    grid: np.ndarray,
    expected_order: list[int],
    kind: str,
) -> None:
    start, facing, _shape = _find_arrow(grid)
    markers = _extract_markers(grid)
    order, steps = _simulate_consumption(start, facing, markers)

    assert order == [int(color) for color in expected_order], "simulation order mismatch"
    assert len(set(order)) == len(order)
    assert len({step["chosen_dir"] for step in steps}) >= 2, "path must contain turns"
    assert all(int(step["distance"]) >= 2 for step in steps), "markers must stay visually separated"

    if kind == "return":
        assert len(steps[0]["active_dirs"]) >= 2, "return-path samples must make initial facing necessary"
        assert bool(steps[0]["used_forward"]), "first move must follow the arrow's facing"
    else:
        assert steps[0]["used_forward"], "stair-path samples must still start by facing forward"

    out = main(grid)
    expected = _rings_from_order(order)
    assert np.array_equal(out, expected)

    rr, cc = np.indices(out.shape)
    center = len(order)
    cheb = np.maximum(np.abs(rr - center), np.abs(cc - center))
    for layer, color in enumerate(order):
        assert set(map(int, np.unique(out[cheb == layer]))) == {int(color)}
    assert set(map(int, np.unique(out[cheb == len(order)]))) == {Color.BLACK}


def generate_input() -> np.ndarray:
    for _ in range(300):
        n = random.randint(3, 6)
        kind = random.choices(["return", "stair"], weights=[3, 2])[0]
        facing = random.randint(0, 3)
        arrow_name = random.choice(["filled4", "pike5", "dart5"])

        points = _rotate_points(_canonical_points(kind, n), facing)
        colors = random.sample(list(MARKER_COLORS), n)
        grid = _paint_scene(points, arrow_name, facing, colors)

        try:
            _validate_scene(grid, colors, kind)
        except AssertionError:
            continue
        return grid

    raise RuntimeError("failed to generate a valid puzzle instance")
