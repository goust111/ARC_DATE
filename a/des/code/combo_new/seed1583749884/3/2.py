from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 8      # azure
WALL = 9    # maroon
MARKER_POOL = [1, 2, 3, 4, 5, 6, 7]
GEN_RETRIES = 500


def _mode_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _non_bg_wall_color(grid: np.ndarray, bg: int) -> int:
    if np.any(grid == WALL):
        return WALL
    values, counts = np.unique(grid[grid != bg], return_counts=True)
    return int(values[int(np.argmax(counts))])


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _wall_components(grid: np.ndarray, wall: int) -> list[list[tuple[int, int]]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != wall:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == wall:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            components.append(comp)
    components.sort(key=lambda comp: (min(r for r, _ in comp), min(c for _, c in comp)))
    return components


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _detect_open_side(grid: np.ndarray, wall: int, box: tuple[int, int, int, int]) -> str:
    r0, c0, r1, c1 = box
    closed = {
        "up": bool(np.all(grid[r0, c0 : c1 + 1] == wall)),
        "down": bool(np.all(grid[r1, c0 : c1 + 1] == wall)),
        "left": bool(np.all(grid[r0 : r1 + 1, c0] == wall)),
        "right": bool(np.all(grid[r0 : r1 + 1, c1] == wall)),
    }
    open_sides = [side for side, present in closed.items() if not present]
    assert len(open_sides) == 1, f"expected one opening, got {closed}"
    return open_sides[0]


def _marker_sort_key(
    marker: tuple[int, int, int],
    side: str,
    box: tuple[int, int, int, int],
) -> tuple[int, int, int, int]:
    r, c, _color = marker
    r0, c0, r1, c1 = box
    inner_r0, inner_c0 = r0 + 1, c0 + 1
    inner_r1, inner_c1 = r1 - 1, c1 - 1
    if side == "up":
        return (r - inner_r0, abs(c - ((inner_c0 + inner_c1) // 2)), r, c)
    if side == "down":
        return (inner_r1 - r, abs(c - ((inner_c0 + inner_c1) // 2)), r, c)
    if side == "left":
        return (c - inner_c0, abs(r - ((inner_r0 + inner_r1) // 2)), r, c)
    return (inner_c1 - c, abs(r - ((inner_r0 + inner_r1) // 2)), r, c)


def _spill_targets(
    box: tuple[int, int, int, int],
    side: str,
    count: int,
) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = box
    mid_r = (r0 + r1) // 2
    mid_c = (c0 + c1) // 2
    if side == "up":
        return [(r0 - count + i, mid_c) for i in range(count)]
    if side == "down":
        return [(r1 + count - i, mid_c) for i in range(count)]
    if side == "left":
        return [(mid_r, c0 - count + i) for i in range(count)]
    return [(mid_r, c1 + count - i) for i in range(count)]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pour the loose marker pixels out of each three-sided container along its opening.

    Input structure:
    - The background is Azure.
    - Maroon pixels form several one-pixel-thick U- or C-shaped containers: a rectangular frame missing exactly one side.
    - Each container's interior holds scattered single-pixel markers of other colors.

    Rule:
    - Detect the open side of each container from the missing edge of its Maroon bounding frame.
    - Remove the interior markers and spill them out through the opening onto the central exterior row or column aligned with that opening.
    - Markers are ordered by distance to the opening: the nearest marker is poured first and ends up farthest outside, while the farthest marker remains closest to the container.
    - The container walls stay fixed and cleared marker cells become background.

    Key insight:
    - The missing side of the frame gives both the spill direction and the single exterior line where the sorted markers must be placed.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    bg = _mode_color(g)
    wall = _non_bg_wall_color(g, bg)

    out = g.copy()
    clear_cells: list[tuple[int, int]] = []
    paint_ops: list[tuple[int, int, int]] = []

    for comp in _wall_components(g, wall):
        box = _bbox(comp)
        r0, c0, r1, c1 = box
        if r1 - r0 < 2 or c1 - c0 < 2:
            continue
        side = _detect_open_side(g, wall, box)
        markers: list[tuple[int, int, int]] = []
        for r in range(r0 + 1, r1):
            for c in range(c0 + 1, c1):
                color = int(g[r, c])
                if color != bg and color != wall:
                    markers.append((r, c, color))
        if not markers:
            continue
        ordered = sorted(markers, key=lambda marker: _marker_sort_key(marker, side, box))
        targets = _spill_targets(box, side, len(ordered))
        assert len(targets) == len(ordered)
        for r, c, color in ordered:
            clear_cells.append((r, c))
        for (tr, tc), (_r, _c, color) in zip(targets, ordered):
            assert 0 <= tr < h and 0 <= tc < w
            paint_ops.append((tr, tc, color))

    for r, c in clear_cells:
        out[r, c] = bg
    for r, c, color in paint_ops:
        out[r, c] = color
    return out


def _make_spec(rng: random.Random, side: str) -> dict[str, int | str]:
    if side in {"up", "down"}:
        inner_h = rng.randint(2, 6)
        inner_w = rng.choice([1, 3, 5])
        count = rng.randint(1, min(4, inner_h, len(MARKER_POOL)))
    else:
        inner_h = rng.choice([1, 3, 5])
        inner_w = rng.randint(2, 6)
        count = rng.randint(1, min(4, inner_w, len(MARKER_POOL)))
    return {
        "side": side,
        "inner_h": inner_h,
        "inner_w": inner_w,
        "count": count,
        "outer_h": inner_h + 2,
        "outer_w": inner_w + 2,
    }


def _reserve_rect(
    h: int,
    w: int,
    r0: int,
    c0: int,
    outer_h: int,
    outer_w: int,
    side: str,
    count: int,
) -> tuple[int, int, int, int] | None:
    top = r0
    left = c0
    bottom = r0 + outer_h - 1
    right = c0 + outer_w - 1
    if side == "up":
        top -= count
    elif side == "down":
        bottom += count
    elif side == "left":
        left -= count
    else:
        right += count
    top = max(0, top - 1)
    left = max(0, left - 1)
    bottom = min(h - 1, bottom + 1)
    right = min(w - 1, right + 1)
    if top > bottom or left > right:
        return None
    return top, left, bottom, right


def _candidate_positions(
    occ: np.ndarray,
    spec: dict[str, int | str],
    h: int,
    w: int,
) -> list[tuple[int, int]]:
    side = str(spec["side"])
    outer_h = int(spec["outer_h"])
    outer_w = int(spec["outer_w"])
    count = int(spec["count"])

    r0_min = 1 + (count if side == "up" else 0)
    r0_max = h - outer_h - 2 - (count if side == "down" else 0)
    c0_min = 1 + (count if side == "left" else 0)
    c0_max = w - outer_w - 2 - (count if side == "right" else 0)
    if r0_min > r0_max or c0_min > c0_max:
        return []

    positions: list[tuple[int, int]] = []
    for r0 in range(r0_min, r0_max + 1):
        for c0 in range(c0_min, c0_max + 1):
            rect = _reserve_rect(h, w, r0, c0, outer_h, outer_w, side, count)
            if rect is None:
                continue
            top, left, bottom, right = rect
            if not np.any(occ[top : bottom + 1, left : right + 1]):
                positions.append((r0, c0))
    return positions


def _paint_container(grid: np.ndarray, r0: int, c0: int, inner_h: int, inner_w: int, side: str) -> None:
    r1 = r0 + inner_h + 1
    c1 = c0 + inner_w + 1
    grid[r0 : r1 + 1, c0] = WALL
    grid[r0 : r1 + 1, c1] = WALL
    grid[r0, c0 : c1 + 1] = WALL
    grid[r1, c0 : c1 + 1] = WALL
    if side == "up":
        grid[r0, c0 + 1 : c1] = BG
    elif side == "down":
        grid[r1, c0 + 1 : c1] = BG
    elif side == "left":
        grid[r0 + 1 : r1, c0] = BG
    else:
        grid[r0 + 1 : r1, c1] = BG


def _place_markers(
    rng: random.Random,
    grid: np.ndarray,
    r0: int,
    c0: int,
    inner_h: int,
    inner_w: int,
    side: str,
    count: int,
) -> None:
    colors = rng.sample(MARKER_POOL, count)
    if side == "up":
        depths = rng.sample(range(inner_h), count)
        for color, depth in zip(colors, depths):
            r = r0 + 1 + depth
            c = c0 + 1 + rng.randrange(inner_w)
            grid[r, c] = color
    elif side == "down":
        depths = rng.sample(range(inner_h), count)
        for color, depth in zip(colors, depths):
            r = r0 + inner_h - depth
            c = c0 + 1 + rng.randrange(inner_w)
            grid[r, c] = color
    elif side == "left":
        depths = rng.sample(range(inner_w), count)
        for color, depth in zip(colors, depths):
            r = r0 + 1 + rng.randrange(inner_h)
            c = c0 + 1 + depth
            grid[r, c] = color
    else:
        depths = rng.sample(range(inner_w), count)
        for color, depth in zip(colors, depths):
            r = r0 + 1 + rng.randrange(inner_h)
            c = c0 + inner_w - depth
            grid[r, c] = color


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(22, 30)
    w = rng.randint(22, 30)
    grid = np.full((h, w), BG, dtype=int)
    occ = np.zeros((h, w), dtype=bool)

    num_containers = rng.randint(2, 4)
    directions = rng.sample(["up", "down", "left", "right"], k=num_containers)
    specs = [_make_spec(rng, side) for side in directions]
    specs.sort(
        key=lambda spec: (
            int(spec["outer_h"]) * int(spec["outer_w"]) + int(spec["count"]),
            int(spec["count"]),
        ),
        reverse=True,
    )

    for spec in specs:
        positions = _candidate_positions(occ, spec, h, w)
        if not positions:
            return None
        r0, c0 = rng.choice(positions)
        side = str(spec["side"])
        inner_h = int(spec["inner_h"])
        inner_w = int(spec["inner_w"])
        count = int(spec["count"])
        outer_h = int(spec["outer_h"])
        outer_w = int(spec["outer_w"])

        _paint_container(grid, r0, c0, inner_h, inner_w, side)
        _place_markers(rng, grid, r0, c0, inner_h, inner_w, side, count)

        rect = _reserve_rect(h, w, r0, c0, outer_h, outer_w, side, count)
        assert rect is not None
        top, left, bottom, right = rect
        occ[top : bottom + 1, left : right + 1] = True

    if _mode_color(grid) != BG:
        return None
    out = main(grid)
    if np.array_equal(out, grid):
        return None
    if out.shape != grid.shape or max(out.shape) > 30:
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid ARC sample")
