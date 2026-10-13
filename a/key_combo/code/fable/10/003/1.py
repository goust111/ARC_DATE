from __future__ import annotations

import random
from collections import Counter
from typing import Iterable

import numpy as np


BLACK = 0
RED = 2
GREEN = 3
MARKER_COLORS = (1, 4, 5, 6, 7, 8, 9)
DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]

Coord = tuple[int, int]


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _left(d: Coord) -> Coord:
    dr, dc = d
    return -dc, dr


def _right(d: Coord) -> Coord:
    dr, dc = d
    return dc, -dr


def _sign(x: int) -> int:
    return (x > 0) - (x < 0)


def _segment_cells(a: Coord, b: Coord) -> list[Coord]:
    r0, c0 = a
    r1, c1 = b
    if r0 != r1 and c0 != c1:
        raise ValueError("segments must be axis-aligned")
    dr = _sign(r1 - r0)
    dc = _sign(c1 - c0)
    out: list[Coord] = []
    r, c = r0, c0
    while True:
        out.append((r, c))
        if (r, c) == (r1, c1):
            return out
        r += dr
        c += dc


def _marker_mask(grid: np.ndarray) -> np.ndarray:
    return ~np.isin(grid, [BLACK, RED, GREEN])


def _wall_mask_from_pairs(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    walls = np.zeros((h, w), dtype=bool)
    marker_coords = np.argwhere(_marker_mask(g))
    by_color: dict[int, list[Coord]] = {}
    for r, c in marker_coords:
        by_color.setdefault(int(g[r, c]), []).append((int(r), int(c)))

    for coords in by_color.values():
        for i in range(len(coords)):
            r0, c0 = coords[i]
            for j in range(i + 1, len(coords)):
                r1, c1 = coords[j]
                if r0 == r1:
                    a, b = sorted((c0, c1))
                    if b - a > 1:
                        walls[r0, a + 1 : b] = True
                elif c0 == c1:
                    a, b = sorted((r0, r1))
                    if b - a > 1:
                        walls[a + 1 : b, c0] = True
    return walls


def _find_boundary_line(grid: np.ndarray, color: int) -> tuple[list[Coord], Coord, Coord]:
    pts = [tuple(map(int, rc)) for rc in np.argwhere(grid == color)]
    assert len(pts) == 2, f"expected exactly one length-2 line of color {color}"
    (r0, c0), (r1, c1) = pts
    assert abs(r0 - r1) + abs(c0 - c1) == 1, "boundary line must be contiguous"
    h, w = grid.shape
    border_pts = [(r, c) for r, c in pts if r in (0, h - 1) or c in (0, w - 1)]
    assert len(border_pts) == 1, "boundary line must have exactly one border endpoint"
    border = border_pts[0]
    inner = pts[0] if pts[1] == border else pts[1]
    direction = (inner[0] - border[0], inner[1] - border[1])
    return pts, inner, direction


def _trace_path(grid: np.ndarray) -> tuple[list[Coord], list[dict], np.ndarray]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    _, current, direction = _find_boundary_line(g, GREEN)
    _find_boundary_line(g, RED)
    markers = _marker_mask(g)
    walls = _wall_mask_from_pairs(g)

    path: list[Coord] = []
    turns: list[dict] = []
    seen: set[tuple[Coord, Coord]] = {((current[0], current[1]), direction)}

    for _ in range(h * w * 4):
        nr, nc = current[0] + direction[0], current[1] + direction[1]
        if _in_bounds(g.shape, nr, nc) and g[nr, nc] == RED:
            return path, turns, walls

        if not _in_bounds(g.shape, nr, nc):
            cause = "oob"
            blocked = True
        elif walls[nr, nc]:
            cause = "wall"
            blocked = True
        elif markers[nr, nc]:
            cause = "marker"
            blocked = True
        elif g[nr, nc] == GREEN:
            cause = "start"
            blocked = True
        elif g[nr, nc] == RED:
            cause = "red"
            blocked = False
        else:
            cause = "open"
            blocked = False

        if not blocked:
            current = (nr, nc)
            path.append(current)
            state = (current, direction)
            assert state not in seen, "path looped unexpectedly"
            seen.add(state)
            continue

        options: list[tuple[Coord, Coord]] = []
        for nd in (_left(direction), _right(direction)):
            rr, cc = current[0] + nd[0], current[1] + nd[1]
            if not _in_bounds(g.shape, rr, cc):
                continue
            if walls[rr, cc] or markers[rr, cc]:
                continue
            if g[rr, cc] in (GREEN, RED):
                continue
            options.append((nd, (rr, cc)))

        assert len(options) == 1, "turn must have exactly one valid side"
        nd, nxt = options[0]
        turns.append(
            {
                "at": current,
                "incoming": direction,
                "outgoing": nd,
                "blocked_ahead": (nr, nc),
                "cause": cause,
            }
        )
        current = nxt
        direction = nd
        path.append(current)
        state = (current, direction)
        assert state not in seen, "path looped after turning"
        seen.add(state)

    raise AssertionError("path did not terminate")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Route the green path from the boundary start line to the red end line through corridors implied by same-color invisible walls.

    Input structure:
    - A black grid contains one green border-attached start line and one red border-attached end line, each exactly two cells long.
    - Interior non-black pixels are isolated marker pixels; some same-color pairs lie exactly on one row or one column, while aligned mismatched pairs and lone markers are distractors.
    - The true walls are not drawn: only the pair endpoints are visible.

    Rule:
    - For every pair of same-color markers that share a row or a column, treat the cells strictly between them as an invisible wall, while leaving the marker endpoints themselves visible.
    - Start from the inner tip of the green line and move straight in the line's inward direction.
    - Whenever the next cell ahead is outside the grid, is a marker pixel, or belongs to an invisible wall, turn to the unique open side among left and right; when the next cell ahead is red, stop; paint every traversed cell green and leave markers unchanged.

    Key insight:
    - The path is controlled by obstacles that must be mentally reconstructed from same-color aligned endpoints; the many visible aligned decoys do nothing because only equal-color endpoint pairs create walls.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    path, _turns, _walls = _trace_path(g)
    out = g.copy()
    for r, c in path:
        if out[r, c] == BLACK:
            out[r, c] = GREEN
    return out


def _paint_wall(grid: np.ndarray, walls: np.ndarray, color: int, cells: list[Coord]) -> list[Coord]:
    assert cells
    for r, c in cells:
        assert grid[r, c] == BLACK and not walls[r, c]
    for r, c in cells:
        walls[r, c] = True
    if cells[0][0] == cells[-1][0]:
        row = cells[0][0]
        c0 = min(c for _r, c in cells)
        c1 = max(c for _r, c in cells)
        markers = [(row, c0 - 1), (row, c1 + 1)]
    else:
        col = cells[0][1]
        r0 = min(r for r, _c in cells)
        r1 = max(r for r, _c in cells)
        markers = [(r0 - 1, col), (r1 + 1, col)]
    for r, c in markers:
        assert grid[r, c] == BLACK and not walls[r, c]
        grid[r, c] = color
    return markers


def _safe_noise_candidates(shape: tuple[int, int], reserved: set[Coord], path_cells: set[Coord]) -> list[Coord]:
    h, w = shape
    candidates: list[Coord] = []
    for r in range(h):
        for c in range(w):
            if (r, c) in reserved:
                continue
            ok = True
            for pr, pc in path_cells:
                if abs(pr - r) + abs(pc - c) <= 1:
                    ok = False
                    break
            if ok:
                candidates.append((r, c))
    random.shuffle(candidates)
    return candidates


def _add_decoy_pair(
    grid: np.ndarray,
    reserved: set[Coord],
    path_cells: set[Coord],
    palette: Iterable[int],
) -> bool:
    colors = list(palette)
    h, w = grid.shape
    for _ in range(120):
        orientation = random.choice(["h", "v"])
        if orientation == "h":
            row = random.randrange(h)
            gap = random.randint(3, min(7, w - 2))
            c0 = random.randint(0, w - gap - 1)
            c1 = c0 + gap
            a, b = (row, c0), (row, c1)
        else:
            col = random.randrange(w)
            gap = random.randint(3, min(7, h - 2))
            r0 = random.randint(0, h - gap - 1)
            r1 = r0 + gap
            a, b = (r0, col), (r1, col)
        if a in reserved or b in reserved:
            continue
        if any(abs(a[0] - pr) + abs(a[1] - pc) <= 1 for pr, pc in path_cells):
            continue
        if any(abs(b[0] - pr) + abs(b[1] - pc) <= 1 for pr, pc in path_cells):
            continue
        ca, cb = random.sample(colors, 2)
        grid[a] = ca
        grid[b] = cb
        reserved.add(a)
        reserved.add(b)
        return True
    return False


def _add_singletons(
    grid: np.ndarray,
    reserved: set[Coord],
    path_cells: set[Coord],
    count: int,
    palette: Iterable[int],
) -> int:
    candidates = _safe_noise_candidates(grid.shape, reserved, path_cells)
    colors = list(palette)
    placed = 0
    for cell in candidates:
        if placed >= count:
            break
        grid[cell] = random.choice(colors)
        reserved.add(cell)
        placed += 1
    return placed


def _build_canonical_instance() -> tuple[np.ndarray, np.ndarray, dict]:
    h = random.randint(16, 24)
    w = random.randint(16, 24)
    grid = np.zeros((h, w), dtype=int)
    wall_mask = np.zeros((h, w), dtype=bool)

    c1 = random.randint(3, 5)
    c2 = random.randint(w - 6, w - 4)
    c4 = random.randint(2, min(5, c2 - 5))
    r1 = random.randint(4, max(4, h // 2 - 2))
    r3 = random.randint(r1 + 4, h - 6)

    start_inner = (1, c1)
    p1 = (r1, c1)
    p2 = (r1, c2)
    p3 = (r3, c2)
    p4 = (r3, c4)
    end_pre = (h - 3, c4)

    points = [start_inner, p1, p2, p3, p4, end_pre]
    path_order: list[Coord] = []
    for a, b in zip(points, points[1:]):
        seg = _segment_cells(a, b)
        path_order.extend(seg if not path_order else seg[1:])
    path_cells = set(path_order)

    grid[0, c1] = GREEN
    grid[1, c1] = GREEN
    grid[h - 2, c4] = RED
    grid[h - 1, c4] = RED

    reserved: set[Coord] = set(path_cells)
    reserved.update({(0, c1), (h - 2, c4), (h - 1, c4)})

    wall_colors = random.sample(MARKER_COLORS, 4)
    wall_markers: list[Coord] = []

    l1 = random.randint(2, min(4, c1))
    cells1 = [(r1 + 1, c) for c in range(c1 - l1 + 1, c1 + 1)]
    wall_markers += _paint_wall(grid, wall_mask, wall_colors[0], cells1)

    l2 = random.randint(2, min(4, r1))
    cells2 = [(r, c2 + 1) for r in range(r1 - l2 + 1, r1 + 1)]
    wall_markers += _paint_wall(grid, wall_mask, wall_colors[1], cells2)

    l3 = random.randint(2, min(4, w - 2 - c2))
    cells3 = [(r3 + 1, c) for c in range(c2, c2 + l3)]
    wall_markers += _paint_wall(grid, wall_mask, wall_colors[2], cells3)

    l4 = random.randint(2, min(4, r3 - r1 - 2))
    cells4 = [(r, c4 - 1) for r in range(r3 - l4 + 1, r3 + 1)]
    wall_markers += _paint_wall(grid, wall_mask, wall_colors[3], cells4)

    for cell in wall_markers:
        reserved.add(cell)

    blockers = [
        (r1, c1 - 1),
        (r1 - 1, c2),
        (r3, c2 + 1),
        (r3 - 1, c4),
    ]
    for cell in blockers:
        assert cell not in reserved and not wall_mask[cell]
        grid[cell] = random.choice(MARKER_COLORS)
        reserved.add(cell)

    decoy_pairs = 1 + (random.random() < 0.6)
    placed_pairs = 0
    for _ in range(decoy_pairs):
        if _add_decoy_pair(grid, reserved, path_cells, MARKER_COLORS):
            placed_pairs += 1

    extra_singletons_target = random.randint(1, 3)
    placed_singletons = _add_singletons(grid, reserved, path_cells, extra_singletons_target, MARKER_COLORS)

    meta = {
        "wall_mask": wall_mask.copy(),
        "path_len": len(path_order) - 1,
        "decoy_pairs": placed_pairs,
        "decoy_singletons": placed_singletons,
    }
    return grid, wall_mask, meta


def _validate_instance(grid: np.ndarray, expected_walls: np.ndarray, meta: dict) -> bool:
    try:
        reconstructed = _wall_mask_from_pairs(grid)
        if not np.array_equal(reconstructed, expected_walls):
            return False
        path, turns, walls = _trace_path(grid)
        if not np.array_equal(walls, expected_walls):
            return False
        if len(path) < 12:
            return False
        if len(turns) != 4:
            return False
        if any(t["cause"] != "wall" for t in turns):
            return False
        turn_dirs = [t["outgoing"] for t in turns]
        incoming = [t["incoming"] for t in turns]
        saw_left = any(out == _left(inc) for inc, out in zip(incoming, turn_dirs))
        saw_right = any(out == _right(inc) for inc, out in zip(incoming, turn_dirs))
        if not (saw_left and saw_right):
            return False
        if meta["decoy_pairs"] < 1 or meta["decoy_singletons"] < 1:
            return False
        out = main(grid)
        if np.array_equal(out, grid):
            return False
        if Counter(map(int, out.ravel()))[GREEN] <= Counter(map(int, grid.ravel()))[GREEN]:
            return False
    except AssertionError:
        return False
    return True


def _random_transform(grid: np.ndarray, walls: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    k = random.randint(0, 3)
    g = np.rot90(grid, k)
    w = np.rot90(walls, k)
    if random.random() < 0.5:
        g = np.fliplr(g)
        w = np.fliplr(w)
    return np.asarray(g, dtype=int), np.asarray(w, dtype=bool)


def _gen_attempt() -> np.ndarray | None:
    grid, walls, meta = _build_canonical_instance()
    grid, walls = _random_transform(grid, walls)
    if not _validate_instance(grid, walls, meta):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(500):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
