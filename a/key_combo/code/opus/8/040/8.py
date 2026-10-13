import random
from collections import Counter

import numpy as np


BACKGROUND = 0
FILL_COLOR = 2


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.flatten())).most_common(1)[0][0])


def in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def components_mask(mask: np.ndarray, connectivity: int = 4) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    if connectivity == 4:
        nbrs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    else:
        nbrs = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0)]

    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts = []
            while stack:
                cr, cc = stack.pop()
                pts.append((cr, cc))
                for dr, dc in nbrs:
                    nr, nc = cr + dr, cc + dc
                    if in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _is_arrow_points(points: list[tuple[int, int]]) -> bool:
    if len(points) != 3:
        return False
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    if max(rows) - min(rows) != 1 or max(cols) - min(cols) != 1:
        return False
    pts = set(points)
    deg2 = 0
    for r, c in points:
        deg = 0
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if (nr, nc) in pts:
                deg += 1
        if deg == 2:
            deg2 += 1
        elif deg != 1:
            return False
    return deg2 == 1


def _arrow_info(color: int, points: list[tuple[int, int]]) -> dict:
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    corners = {(r0, c0), (r0, c1), (r1, c0), (r1, c1)}
    pts = set(points)
    missing = next(iter(corners - pts))
    dr = 1 if missing[0] == r0 else -1
    dc = 1 if missing[1] == c0 else -1

    start = None
    for r, c in points:
        deg = sum((nr, nc) in pts for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)))
        if deg == 2:
            start = (r, c)
            break
    if start is None:
        raise AssertionError("arrow must have one right-angle vertex")

    return {
        "color": int(color),
        "points": tuple(sorted(points)),
        "anchor": min(points),
        "start": start,
        "direction": (int(dr), int(dc)),
    }


def _detect_arrows_and_walls(grid: np.ndarray) -> tuple[list[dict], dict[tuple[int, int], int]]:
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    arrows: list[dict] = []
    walls: dict[tuple[int, int], int] = {}

    for color in sorted(int(v) for v in np.unique(g) if int(v) != bg):
        for comp in components_mask(g == color, connectivity=4):
            pts = [tuple(map(int, rc)) for rc in comp]
            if _is_arrow_points(pts):
                arrows.append(_arrow_info(color, pts))
            else:
                for cell in pts:
                    walls[cell] = int(color)

    arrows.sort(key=lambda arrow: arrow["anchor"])
    return arrows, walls


def _trace_ray(
    out: np.ndarray,
    arrow_cells: set[tuple[int, int]],
    wall_colors: dict[tuple[int, int], int],
    start: tuple[int, int],
    direction: tuple[int, int],
    start_color: int,
    blocker_cells: set[tuple[int, int]],
) -> dict:
    r, c = start
    dr, dc = direction
    current_color = int(start_color)
    cells: list[tuple[int, int]] = []
    cell_set: set[tuple[int, int]] = set()
    seen: set[tuple[int, int, int, int, int]] = set()
    stop_reason = "boundary"
    stop_target = None

    while True:
        state = (r, c, dr, dc, current_color)
        if state in seen:
            stop_reason = "loop"
            break
        seen.add(state)

        nr, nc = r + dr, c + dc
        if not in_bounds(out.shape, nr, nc):
            stop_reason = "boundary"
            break
        if (nr, nc) in blocker_cells:
            stop_reason = "collision"
            stop_target = (nr, nc)
            break
        if (nr, nc) in wall_colors:
            current_color = int(wall_colors[(nr, nc)])
            out[nr, nc] = current_color
            if (nr, nc) not in cell_set:
                cell_set.add((nr, nc))
                cells.append((nr, nc))
            r, c = nr, nc
            dr, dc = -dr, -dc
            continue
        if (nr, nc) in cell_set:
            r, c = nr, nc
            continue
        if (nr, nc) in arrow_cells:
            stop_reason = "arrow"
            stop_target = (nr, nc)
            break
        if int(out[nr, nc]) == BACKGROUND:
            out[nr, nc] = current_color
            cell_set.add((nr, nc))
            cells.append((nr, nc))
            r, c = nr, nc
            continue
        stop_reason = "blocked"
        stop_target = (nr, nc)
        break

    return {
        "cells": cells,
        "cell_set": cell_set,
        "stop_reason": stop_reason,
        "stop_target": stop_target,
    }


def _solve(grid: np.ndarray) -> tuple[np.ndarray, dict]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    arrows, wall_colors = _detect_arrows_and_walls(g)
    assert len(arrows) == 2, "expected exactly two arrows"

    out = g.copy()
    arrow_cells = {cell for arrow in arrows for cell in arrow["points"]}

    rays = []
    blockers: set[tuple[int, int]] = set()
    for arrow in arrows:
        ray = _trace_ray(
            out,
            arrow_cells,
            wall_colors,
            arrow["start"],
            arrow["direction"],
            arrow["color"],
            blockers,
        )
        rays.append(ray)
        blockers |= ray["cell_set"]

    fill_rows: list[int] = []
    fill_spans: dict[int, tuple[int, int]] = {}
    all_ray_cells = rays[0]["cell_set"] | rays[1]["cell_set"]
    for r in range(g.shape[0]):
        cols = sorted(c for rr, c in all_ray_cells if rr == r)
        if len(cols) != 2:
            continue
        c0, c1 = cols
        if c1 - c0 <= 1:
            continue
        fill_rows.append(r)
        fill_spans[r] = (c0, c1)
        for c in range(c0 + 1, c1):
            if out[r, c] == BACKGROUND:
                out[r, c] = FILL_COLOR

    meta = {
        "arrows": arrows,
        "rays": rays,
        "fill_rows": fill_rows,
        "fill_spans": fill_spans,
        "ray_cells": all_ray_cells,
    }
    return out, meta


def main(grid: np.ndarray) -> np.ndarray:
    """
    Trace two ordered diagonal rays whose combined painted cells become row-wise fill boundaries.

    Input structure:
    - A mostly black grid contains exactly two monochrome 3-cell L-shaped arrows and 2-4 scattered wall pixels.
    - Each arrow occupies a 2x2 box with one missing corner; the missing corner determines the diagonal firing direction.
    - Wall pixels use non-arrow colors and are separate from the arrow bodies.

    Rule:
    - Order the two arrows by reading order of their topmost-leftmost pixel; that arrow fires first.
    - Each ray starts from the arrow's right-angle vertex and steps diagonally away from the missing corner.
    - On a wall hit, the ray paints that wall cell, adopts the wall color, reverses both axes, and continues; on an out-of-bounds step it stops.
    - The second ray additionally stops before entering any cell already painted by the first ray.
    - After both rays stop, scan each row: if exactly two ray-painted cells occur on that row, fill the background cells strictly between them with red.

    Key insight:
    - The first ray fixes one side of a sparse contour, the second ray is causally clipped by that contour, and those two painted boundary cells per row determine the horizontal fill spans.
    """
    out, _meta = _solve(grid)
    return out


def _arrow_from_start(start: tuple[int, int], direction: tuple[int, int], color: int) -> dict:
    r, c = start
    dr, dc = direction
    if (dr, dc) == (1, 1):
        points = [(r - 1, c), (r, c - 1), (r, c)]
    elif (dr, dc) == (1, -1):
        points = [(r - 1, c), (r, c), (r, c + 1)]
    elif (dr, dc) == (-1, 1):
        points = [(r, c - 1), (r, c), (r + 1, c)]
    elif (dr, dc) == (-1, -1):
        points = [(r, c), (r, c + 1), (r + 1, c)]
    else:
        raise ValueError("direction must be diagonal")
    return {"color": int(color), "points": tuple(points), "start": start, "direction": direction}


def _manhattan_gap(points_a: tuple[tuple[int, int], ...], points_b: tuple[tuple[int, int], ...]) -> int:
    return min(abs(ra - rb) + abs(ca - cb) for ra, ca in points_a for rb, cb in points_b)


def _longest_run(rows: list[int]) -> int:
    if not rows:
        return 0
    best = cur = 1
    for i in range(1, len(rows)):
        if rows[i] == rows[i - 1] + 1:
            cur += 1
            best = max(best, cur)
        else:
            cur = 1
    return best


def _extra_wall_candidates(
    shape: tuple[int, int],
    arrow_cells: set[tuple[int, int]],
    ray_cells: set[tuple[int, int]],
    fill_spans: dict[int, tuple[int, int]],
) -> list[tuple[int, int]]:
    h, w = shape
    forbidden = set(arrow_cells) | set(ray_cells)
    for r, (c0, c1) in fill_spans.items():
        for c in range(c0, c1 + 1):
            forbidden.add((r, c))

    candidates = []
    for r in range(h):
        for c in range(w):
            if (r, c) in forbidden:
                continue
            if any(max(abs(r - ar), abs(c - ac)) <= 1 for ar, ac in arrow_cells):
                continue
            candidates.append((r, c))
    random.shuffle(candidates)
    return candidates


def _build_grid(shape: tuple[int, int], arrows: list[dict], walls: dict[tuple[int, int], int]) -> np.ndarray:
    g = np.zeros(shape, dtype=int)
    for arrow in arrows:
        for r, c in arrow["points"]:
            g[r, c] = int(arrow["color"])
    for (r, c), color in walls.items():
        g[r, c] = int(color)
    return g


def _attempt_generate() -> np.ndarray | None:
    h = random.randint(13, 18)
    w = random.randint(13, 18)
    mode = random.choice(["down", "up"])

    arrow_palette = [1, 3, 8, 9]
    wall_palette = [4, 5, 6, 7, 8, 9, 1, 3]
    arrow_colors = random.sample(arrow_palette, 2)
    wall_palette = [c for c in wall_palette if c not in arrow_colors and c != FILL_COLOR]

    if mode == "down":
        left_dir, right_dir = (1, 1), (1, -1)
        row_choices = list(range(1, min(4, h - 2) + 1))
    else:
        left_dir, right_dir = (-1, 1), (-1, -1)
        row_choices = list(range(max(1, h - 5), h - 1))

    left_cols = list(range(1, max(2, w // 3) + 1))
    right_cols = list(range(max(w // 2 + 1, (2 * w) // 3), w - 1))
    if not row_choices or not left_cols or not right_cols:
        return None

    for _ in range(120):
        left_start = (random.choice(row_choices), random.choice(left_cols))
        right_start = (random.choice(row_choices), random.choice(right_cols))

        left_arrow = _arrow_from_start(left_start, left_dir, arrow_colors[0])
        right_arrow = _arrow_from_start(right_start, right_dir, arrow_colors[1])
        if _manhattan_gap(left_arrow["points"], right_arrow["points"]) < 5:
            continue

        arrows = [left_arrow, right_arrow]
        base = _build_grid((h, w), arrows, {})
        try:
            _out0, meta0 = _solve(base)
        except AssertionError:
            continue

        if meta0["rays"][1]["stop_reason"] != "collision":
            continue
        if len(meta0["rays"][0]["cells"]) < 4 or len(meta0["rays"][1]["cells"]) < 3:
            continue
        if _longest_run(meta0["fill_rows"]) < 3:
            continue
        if not meta0["fill_spans"]:
            continue

        collision_cell = meta0["rays"][1]["stop_target"]
        first_cells = meta0["rays"][0]["cells"]
        collision_idx = first_cells.index(collision_cell)

        wall_count = random.randint(2, 4)
        walls: dict[tuple[int, int], int] = {}

        use_bounce = random.random() < 0.7 and collision_idx >= 2
        if use_bounce:
            candidates = list(range(collision_idx, len(first_cells)))
            if candidates:
                wall_cell = first_cells[random.choice(candidates)]
                walls[wall_cell] = 0  # color assigned later

        extra_needed = wall_count - len(walls)
        arrow_cells = {cell for arrow in arrows for cell in arrow["points"]}
        candidates = _extra_wall_candidates((h, w), arrow_cells, meta0["ray_cells"], meta0["fill_spans"])
        if len(candidates) < extra_needed:
            continue
        for cell in candidates[:extra_needed]:
            walls[cell] = 0

        wall_colors = random.sample(wall_palette, len(walls))
        colored_walls = {cell: color for cell, color in zip(walls, wall_colors)}
        grid = _build_grid((h, w), arrows, colored_walls)

        try:
            out, meta = _solve(grid)
        except AssertionError:
            continue

        if meta["rays"][1]["stop_reason"] != "collision":
            continue
        if _longest_run(meta["fill_rows"]) < 3:
            continue
        if np.count_nonzero(out == FILL_COLOR) < 4:
            continue
        if len(colored_walls) < 2 or len(colored_walls) > 4:
            continue
        if len(set(colored_walls.values())) != len(colored_walls):
            continue
        if use_bounce:
            bounce_cell = next(iter(walls))
            if bounce_cell not in meta["rays"][0]["cell_set"]:
                continue
        if np.array_equal(out, grid):
            continue
        return grid

    return None


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 8")
