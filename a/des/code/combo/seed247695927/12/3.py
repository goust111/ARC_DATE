from __future__ import annotations

import numpy as np


def _background_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _diamond_candidates(grid: np.ndarray, bg: int) -> list[tuple[int, int, int, int, int, int]]:
    candidates: list[tuple[int, int, int, int, int, int]] = []
    for color in (int(v) for v in np.unique(grid) if int(v) != bg):
        coords = np.argwhere(grid == color)
        if coords.size == 0:
            continue
        r0 = int(coords[:, 0].min())
        r1 = int(coords[:, 0].max())
        c0 = int(coords[:, 1].min())
        c1 = int(coords[:, 1].max())
        height = r1 - r0 + 1
        width = c1 - c0 + 1
        if height != width or height % 2 == 0:
            continue
        radius = height // 2
        center_r = r0 + radius
        center_c = c0 + radius
        dists = np.abs(coords[:, 0] - center_r) + np.abs(coords[:, 1] - center_c)
        if len(coords) == 4 * radius and np.all(dists == radius):
            candidates.append((color, r0, c0, center_r, center_c, radius))
    return candidates


def _frame_info(grid: np.ndarray) -> tuple[int, int, int, int, int, int, int]:
    bg = _background_color(grid)
    candidates = _diamond_candidates(grid, bg)
    assert len(candidates) == 1, "expected exactly one complete diamond frame"
    color, r0, c0, center_r, center_c, radius = candidates[0]
    return bg, color, r0, c0, center_r, center_c, radius


def _marker_directions(
    r: int,
    c: int,
    r0: int,
    c0: int,
    radius: int,
) -> list[tuple[int, int]]:
    r1 = r0 + 2 * radius
    c1 = c0 + 2 * radius
    dirs: list[tuple[int, int]] = []
    if r == r0:
        dirs.append((1, 0))
    if r == r1:
        dirs.append((-1, 0))
    if c == c0:
        dirs.append((0, 1))
    if c == c1:
        dirs.append((0, -1))
    return dirs


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fire straight rays from colored side markers into the inside of a solid diamond frame.

    Input structure:
    - The background contains one complete one-pixel-thick diamond frame.
    - Colored marker pixels sit on the top, bottom, left, or right side of the
      diamond's bounding box.
    - The diamond interior may already contain non-background blocker pixels.

    Rule:
    - Each marker emits inward, perpendicular to the bounding-box side that contains it.
    - Only background pixels strictly inside the diamond are painted.
    - A ray stops when it reaches an existing interior foreground pixel.
    - The frame and all original colored pixels stay unchanged.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = map(int, g.shape)
    bg, frame_color, r0, c0, center_r, center_c, radius = _frame_info(g)
    out = g.copy()

    planned: dict[tuple[int, int], int] = {}
    marker_count = 0

    for r_raw, c_raw in np.argwhere((g != bg) & (g != frame_color)):
        r = int(r_raw)
        c = int(c_raw)
        dist = abs(r - center_r) + abs(c - center_c)
        if dist < radius:
            continue

        dirs = _marker_directions(r, c, r0, c0, radius)
        assert dirs, "foreground outside the diamond interior must be a side marker"
        assert dist > radius, "markers cannot lie on the frame"
        marker_count += 1
        color = int(g[r, c])

        for dr, dc in dirs:
            rr = r + dr
            cc = c + dc
            entered_interior = False
            while 0 <= rr < h and 0 <= cc < w:
                inside = abs(rr - center_r) + abs(cc - center_c) < radius
                if inside:
                    entered_interior = True
                    if int(g[rr, cc]) == bg:
                        previous = planned.get((rr, cc))
                        assert previous is None or previous == color, "conflicting ray colors"
                        planned[(rr, cc)] = color
                    else:
                        break
                elif entered_interior:
                    break
                rr += dr
                cc += dc

    assert marker_count > 0, "expected at least one side marker"
    for (r, c), color in planned.items():
        out[r, c] = color
    return out


PALETTE = tuple(range(10))
MIN_SIZE = 11
MAX_SIZE = 21
MIN_RADIUS = 3
MAX_ATTEMPTS = 400


def _draw_diamond(grid: np.ndarray, center_r: int, center_c: int, radius: int, color: int) -> None:
    for dr in range(-radius, radius + 1):
        dc = radius - abs(dr)
        grid[center_r + dr, center_c - dc] = color
        grid[center_r + dr, center_c + dc] = color


def _interior_cells(center_r: int, center_c: int, radius: int) -> list[tuple[int, int]]:
    cells: list[tuple[int, int]] = []
    for r in range(center_r - radius + 1, center_r + radius):
        span = radius - abs(r - center_r) - 1
        for c in range(center_c - span, center_c + span + 1):
            cells.append((r, c))
    return cells


def _ray_cells(
    side: str,
    pos: int,
    center_r: int,
    center_c: int,
    radius: int,
) -> list[tuple[int, int]]:
    if side in ("top", "bottom"):
        return [
            (r, pos)
            for r in range(center_r - radius + 1, center_r + radius)
            if abs(r - center_r) + abs(pos - center_c) < radius
        ]
    return [
        (pos, c)
        for c in range(center_c - radius + 1, center_c + radius)
        if abs(pos - center_r) + abs(c - center_c) < radius
    ]


def _choose_side_positions(radius: int, upper_first: bool) -> dict[str, list[int]]:
    low = list(range(radius))
    high = list(range(radius + 1, 2 * radius + 1))
    if upper_first:
        top_pool, bottom_pool = low, high
        left_pool, right_pool = low, high
    else:
        top_pool, bottom_pool = high, low
        left_pool, right_pool = high, low

    side_positions: dict[str, list[int]] = {}
    for side, pool in (
        ("top", top_pool),
        ("bottom", bottom_pool),
        ("left", left_pool),
        ("right", right_pool),
    ):
        count = int(np.random.randint(1, len(pool) + 1))
        picked = sorted(int(v) for v in np.random.choice(pool, size=count, replace=False))
        side_positions[side] = picked
    return side_positions


def _place_markers(
    grid: np.ndarray,
    r0: int,
    c0: int,
    center_r: int,
    center_c: int,
    radius: int,
    colors: list[int],
) -> tuple[list[tuple[str, int, int]], dict[str, list[int]], set[tuple[int, int]]]:
    side_positions = _choose_side_positions(radius, upper_first=bool(np.random.randint(2)))
    side_positions = {
        side: positions
        for side, positions in side_positions.items()
        if len(positions) == 1 or bool(np.random.randint(2)) or side in ("top", "left")
    }

    markers: list[tuple[str, int, int]] = []
    covered_rays: set[tuple[int, int]] = set()
    for side, positions in side_positions.items():
        if not positions:
            continue
        for pos in positions:
            color = colors[int(np.random.randint(len(colors)))]
            if side == "top":
                coord = (r0, c0 + pos)
            elif side == "bottom":
                coord = (r0 + 2 * radius, c0 + pos)
            elif side == "left":
                coord = (r0 + pos, c0)
            else:
                coord = (r0 + pos, c0 + 2 * radius)
            grid[coord] = color
            markers.append((side, pos, color))
            covered_rays.update(_ray_cells(side, (c0 + pos) if side in ("top", "bottom") else (r0 + pos), center_r, center_c, radius))
    return markers, side_positions, covered_rays


def _place_blockers(
    grid: np.ndarray,
    center_r: int,
    center_c: int,
    radius: int,
    frame_color: int,
    bg: int,
    markers: list[tuple[str, int, int]],
) -> None:
    interior = _interior_cells(center_r, center_c, radius)
    if not interior:
        return

    blocker_pool = [color for color in PALETTE if color not in {bg, frame_color}]
    chosen_lines = markers[:]
    np.random.shuffle(chosen_lines)
    desired = int(np.random.randint(0, min(5, len(interior)) + 1))

    placed = 0
    used: set[tuple[int, int]] = set()

    for side, pos, _marker_color in chosen_lines:
        if placed >= desired:
            break
        line = _ray_cells(
            side,
            (center_c - radius + pos) if side in ("top", "bottom") else (center_r - radius + pos),
            center_r,
            center_c,
            radius,
        )
        if len(line) < 2:
            continue
        blocker_index = int(np.random.randint(1, len(line)))
        cell = line[blocker_index]
        if cell in used:
            continue
        grid[cell] = blocker_pool[int(np.random.randint(len(blocker_pool)))]
        used.add(cell)
        placed += 1

    extra_budget = int(np.random.randint(0, 3))
    np.random.shuffle(interior)
    for cell in interior:
        if extra_budget == 0:
            break
        if grid[cell] != bg:
            continue
        if bool(np.random.randint(3)):
            continue
        grid[cell] = blocker_pool[int(np.random.randint(len(blocker_pool)))]
        extra_budget -= 1


def _valid_sample(grid: np.ndarray) -> bool:
    try:
        out = main(grid)
    except Exception:
        return False
    return out.shape == grid.shape and not np.array_equal(out, grid)


def _gen_attempt() -> np.ndarray | None:
    h = int(np.random.randint(MIN_SIZE, MAX_SIZE + 1))
    w = int(np.random.randint(MIN_SIZE, MAX_SIZE + 1))
    max_radius = min((h - 3) // 2, (w - 3) // 2)
    if max_radius < MIN_RADIUS:
        return None
    radius = int(np.random.randint(MIN_RADIUS, max_radius + 1))

    center_r = int(np.random.randint(radius + 1, h - radius - 1))
    center_c = int(np.random.randint(radius + 1, w - radius - 1))

    colors = list(PALETTE)
    bg = colors.pop(int(np.random.randint(len(colors))))
    frame_color = colors.pop(int(np.random.randint(len(colors))))
    np.random.shuffle(colors)

    grid = np.full((h, w), bg, dtype=int)
    _draw_diamond(grid, center_r, center_c, radius, frame_color)
    r0 = center_r - radius
    c0 = center_c - radius

    markers, _side_positions, _covered_rays = _place_markers(
        grid,
        r0,
        c0,
        center_r,
        center_c,
        radius,
        colors,
    )
    if len(markers) < 4:
        return None

    _place_blockers(grid, center_r, center_c, radius, frame_color, bg, markers)
    if not _valid_sample(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
