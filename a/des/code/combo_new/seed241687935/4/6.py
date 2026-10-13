from __future__ import annotations

import random
from typing import Iterable

import numpy as np


DIRS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}

SIDE_ORDER = ("up", "down", "left", "right")
NON_BG_COLORS = tuple(range(1, 10))
GEN_RETRY_LIMIT = 500
PLACEMENT_TRIES = 140


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(points: Iterable[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows, cols = zip(*points, strict=False)
    return min(rows), min(cols), max(rows), max(cols)


def _inside_bbox(r: int, c: int, box: tuple[int, int, int, int]) -> bool:
    r0, c0, r1, c1 = box
    return r0 <= r <= r1 and c0 <= c <= c1


def _expanded_bbox_clear(reserved: np.ndarray, box: tuple[int, int, int, int], pad: int = 1) -> bool:
    h, w = reserved.shape
    r0, c0, r1, c1 = box
    er0 = max(0, r0 - pad)
    ec0 = max(0, c0 - pad)
    er1 = min(h - 1, r1 + pad)
    ec1 = min(w - 1, c1 + pad)
    return not bool(reserved[er0 : er1 + 1, ec0 : ec1 + 1].any())


def _side_cells(box: tuple[int, int, int, int], side: str) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = box
    if side == "up":
        return [(r0, c) for c in range(c0, c1 + 1)]
    if side == "down":
        return [(r1, c) for c in range(c0, c1 + 1)]
    if side == "left":
        return [(r, c0) for r in range(r0, r1 + 1)]
    if side == "right":
        return [(r, c1) for r in range(r0, r1 + 1)]
    raise ValueError(side)


def _ring_pixels_from_spec(
    box: tuple[int, int, int, int], side: str, gap_start: int, gap_len: int
) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = box
    ring: set[tuple[int, int]] = set()
    for c in range(c0, c1 + 1):
        ring.add((r0, c))
        ring.add((r1, c))
    for r in range(r0, r1 + 1):
        ring.add((r, c0))
        ring.add((r, c1))

    if side in ("up", "down"):
        fixed_r = r0 if side == "up" else r1
        for offset in range(gap_start, gap_start + gap_len):
            ring.remove((fixed_r, c0 + 1 + offset))
    else:
        fixed_c = c0 if side == "left" else c1
        for offset in range(gap_start, gap_start + gap_len):
            ring.remove((r0 + 1 + offset, fixed_c))
    return ring


def _detect_ring(comp: list[tuple[int, int]], color: int) -> dict | None:
    box = _bbox(comp)
    r0, c0, r1, c1 = box
    if r1 - r0 < 4 or c1 - c0 < 4:
        return None

    points = set(comp)
    boundary = set()
    for side in SIDE_ORDER:
        boundary.update(_side_cells(box, side))
    if any(pt not in boundary for pt in points):
        return None

    corners = {(r0, c0), (r0, c1), (r1, c0), (r1, c1)}
    if any(corner not in points for corner in corners):
        return None

    missing_by_side: dict[str, list[tuple[int, int]]] = {}
    for side in SIDE_ORDER:
        missing = [pt for pt in _side_cells(box, side) if pt not in points]
        missing_by_side[side] = missing

    gap_sides = [side for side, missing in missing_by_side.items() if missing]
    if len(gap_sides) != 1:
        return None

    gap_side = gap_sides[0]
    gap_cells = missing_by_side[gap_side]
    if any(pt in corners for pt in gap_cells):
        return None

    coords = [c if gap_side in ("up", "down") else r for r, c in gap_cells]
    coords = sorted(coords)
    if not coords or coords[-1] - coords[0] + 1 != len(coords):
        return None

    inner_box = (r0 + 1, c0 + 1, r1 - 1, c1 - 1)
    if inner_box[0] > inner_box[2] or inner_box[1] > inner_box[3]:
        return None

    return {
        "color": color,
        "box": box,
        "points": points,
        "gap_side": gap_side,
        "gap_cells": set(gap_cells),
        "inner_box": inner_box,
    }


def _find_rings(grid: np.ndarray, bg: int) -> list[dict]:
    rings: list[dict] = []
    for color in sorted(int(v) for v in np.unique(grid) if int(v) != bg):
        for comp in _components(grid == color):
            ring = _detect_ring(comp, color)
            if ring is not None:
                rings.append(ring)
    return rings


def _trace_ray(
    start: tuple[int, int], direction: tuple[int, int], ring_points: set[tuple[int, int]], shape: tuple[int, int]
) -> list[tuple[int, int]] | None:
    h, w = shape
    r, c = start
    path: list[tuple[int, int]] = []
    while 0 <= r < h and 0 <= c < w:
        if (r, c) in ring_points:
            return None
        path.append((r, c))
        r += direction[0]
        c += direction[1]
    return path


def main(grid: np.ndarray) -> np.ndarray:
    """
    Project each ring's inner dots straight through its opening and out to the grid edge.

    Input structure:
    - The background is black.
    - Several colored rectangular ring frames appear; each is a 1-pixel-thick rectangle with one contiguous gap on exactly one side.
    - Each ring contains 1-4 interior dot pixels of one other color, and the dot color differs from the frame color.
    - Different rings may use different frame colors, dot colors, sizes, and opening directions.

    Rule:
    - Detect each ring's missing side to determine the leak direction.
    - For every non-background interior dot, cast a straight cardinal ray from the dot in that direction.
    - If that ray would hit the ring's own frame, cancel it entirely; otherwise color every cell from the dot, through the gap, and onward to the grid boundary with the dot color.
    - Leave the ring frames unchanged and keep the original interior dots.

    Key insight: The missing side is both the direction cue and the only escape corridor, so a dot leaks only when its straight path lines up with the exact gap span.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = 0 if int((g == 0).sum()) > 0 else int(np.bincount(g.ravel()).argmax())

    out = g.copy()
    for ring in _find_rings(g, bg):
        r0, c0, r1, c1 = ring["inner_box"]
        dr, dc = DIRS[ring["gap_side"]]
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                color = int(g[r, c])
                if color == bg or color == ring["color"]:
                    continue
                path = _trace_ray((r, c), (dr, dc), ring["points"], g.shape)
                if path is None:
                    continue
                for rr, cc in path:
                    out[rr, cc] = color
    return out


def _sample_dot_positions(
    box: tuple[int, int, int, int], side: str, gap_start: int, gap_len: int, dot_count: int
) -> tuple[list[tuple[int, int]], set[tuple[int, int]]]:
    r0, c0, r1, c1 = box
    rows = list(range(r0 + 1, r1))
    cols = list(range(c0 + 1, c1))

    blocked_positions: list[tuple[int, int]] = []
    leaking_dots: list[tuple[int, int]] = []

    if side in ("up", "down"):
        leak_cols = cols[gap_start : gap_start + gap_len]
        leak_col_set = set(leak_cols)
        for r in rows:
            for c in cols:
                if c not in leak_col_set:
                    blocked_positions.append((r, c))
        max_leakers = min(dot_count, len(leak_cols))
        leaker_count = random.randint(0, max_leakers)
        if leaker_count:
            for c in random.sample(leak_cols, leaker_count):
                leaking_dots.append((random.choice(rows), c))
    else:
        leak_rows = rows[gap_start : gap_start + gap_len]
        leak_row_set = set(leak_rows)
        for r in rows:
            for c in cols:
                if r not in leak_row_set:
                    blocked_positions.append((r, c))
        max_leakers = min(dot_count, len(leak_rows))
        leaker_count = random.randint(0, max_leakers)
        if leaker_count:
            for r in random.sample(leak_rows, leaker_count):
                leaking_dots.append((r, random.choice(cols)))

    chosen = list(leaking_dots)
    remaining = dot_count - len(chosen)
    blocked_pool = [pt for pt in blocked_positions if pt not in leaking_dots]
    if remaining > len(blocked_pool):
        return [], set()
    chosen.extend(random.sample(blocked_pool, remaining))
    random.shuffle(chosen)
    return chosen, set(leaking_dots)


def _paint_ring(
    grid: np.ndarray,
    ring_points: set[tuple[int, int]],
    ring_color: int,
    dot_positions: list[tuple[int, int]],
    dot_color: int,
) -> None:
    for r, c in ring_points:
        grid[r, c] = ring_color
    for r, c in dot_positions:
        grid[r, c] = dot_color


def _choose_grid_shape() -> tuple[int, int]:
    return random.randint(14, 30), random.randint(14, 30)


def _choose_ring_count(h: int, w: int) -> int:
    m = min(h, w)
    if m < 18:
        return random.randint(1, 2)
    if m < 24:
        return random.randint(1, 3)
    return random.randint(2, 4)


def _candidate_directions(count: int) -> list[str]:
    dirs = random.sample(list(SIDE_ORDER), min(count, len(SIDE_ORDER)))
    while len(dirs) < count:
        dirs.append(random.choice(SIDE_ORDER))
    random.shuffle(dirs)
    return dirs


def _build_ring_candidate(
    h: int,
    w: int,
    reserved: np.ndarray,
    ring_color: int,
    dot_color: int,
    side: str,
) -> dict | None:
    for _ in range(PLACEMENT_TRIES):
        inner_h = random.randint(3, min(6, h - 4))
        inner_w = random.randint(3, min(6, w - 4))
        outer_h = inner_h + 2
        outer_w = inner_w + 2
        if outer_h + 2 > h or outer_w + 2 > w:
            continue

        r0 = random.randint(1, h - outer_h - 1)
        c0 = random.randint(1, w - outer_w - 1)
        box = (r0, c0, r0 + outer_h - 1, c0 + outer_w - 1)
        if not _expanded_bbox_clear(reserved, box, pad=1):
            continue

        axis_len = inner_w if side in ("up", "down") else inner_h
        gap_len = random.randint(1, axis_len - 1)
        gap_start = random.randint(0, axis_len - gap_len)
        ring_points = _ring_pixels_from_spec(box, side, gap_start, gap_len)

        dot_count = random.randint(1, 4)
        dot_positions, leaking_dots = _sample_dot_positions(box, side, gap_start, gap_len, dot_count)
        if not dot_positions:
            continue

        leak_paths: list[list[tuple[int, int]]] = []
        for dot in leaking_dots:
            path = _trace_ray(dot, DIRS[side], ring_points, (h, w))
            if path is None:
                leak_paths = []
                break
            leak_paths.append(path)
        if len(leak_paths) != len(leaking_dots):
            continue

        blocked_dots = [pt for pt in dot_positions if pt not in leaking_dots]
        if any(_trace_ray(dot, DIRS[side], ring_points, (h, w)) is not None for dot in blocked_dots):
            continue

        flat_path = {pt for path in leak_paths for pt in path}
        if any(reserved[r, c] for r, c in flat_path):
            continue

        return {
            "box": box,
            "ring_points": ring_points,
            "ring_color": ring_color,
            "dot_positions": dot_positions,
            "dot_color": dot_color,
            "gap_side": side,
            "gap_start": gap_start,
            "gap_len": gap_len,
            "leaker_count": len(leaking_dots),
            "blocked_count": dot_count - len(leaking_dots),
            "ray_cells": flat_path,
        }
    return None


def _mark_reserved(reserved: np.ndarray, box: tuple[int, int, int, int], ray_cells: set[tuple[int, int]]) -> None:
    r0, c0, r1, c1 = box
    reserved[r0 - 1 : r1 + 2, c0 - 1 : c1 + 2] = True
    for r, c in ray_cells:
        reserved[r, c] = True


def _generate_scene() -> np.ndarray | None:
    h, w = _choose_grid_shape()
    ring_count = _choose_ring_count(h, w)
    colors = random.sample(list(NON_BG_COLORS), ring_count * 2)
    ring_colors = colors[:ring_count]
    dot_colors = colors[ring_count:]
    directions = _candidate_directions(ring_count)

    grid = np.zeros((h, w), dtype=int)
    reserved = np.zeros((h, w), dtype=bool)

    total_leakers = 0
    total_blockers = 0
    for idx in range(ring_count):
        candidate = _build_ring_candidate(
            h=h,
            w=w,
            reserved=reserved,
            ring_color=ring_colors[idx],
            dot_color=dot_colors[idx],
            side=directions[idx],
        )
        if candidate is None:
            return None
        _paint_ring(
            grid,
            candidate["ring_points"],
            candidate["ring_color"],
            candidate["dot_positions"],
            candidate["dot_color"],
        )
        _mark_reserved(reserved, candidate["box"], candidate["ray_cells"])
        total_leakers += candidate["leaker_count"]
        total_blockers += candidate["blocked_count"]

    if total_leakers == 0 or total_blockers == 0:
        return None

    rings = _find_rings(grid, 0)
    if len(rings) != ring_count:
        return None

    out = main(grid)
    if np.array_equal(out, grid):
        return None
    if out.shape != grid.shape:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        scene = _generate_scene()
        if scene is not None:
            return scene
    raise RuntimeError("failed to generate a valid ARC input")
