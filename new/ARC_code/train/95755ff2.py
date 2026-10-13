from __future__ import annotations

import numpy as np

from arc_utils import bbox_coords, get_bg


def _diamond_frame(g: np.ndarray, bg: int) -> tuple[int, tuple[int, int, int, int], int, int, int]:
    candidates = []
    for color in set(map(int, np.unique(g))) - {int(bg)}:
        coords = np.argwhere(g == color)
        r0, c0, r1, c1 = bbox_coords(coords)
        h = int(r1 - r0 + 1)
        w = int(c1 - c0 + 1)
        if h != w or h % 2 == 0:
            continue

        radius = h // 2
        center_r = int(r0 + radius)
        center_c = int(c0 + radius)
        dists = np.abs(coords[:, 0] - center_r) + np.abs(coords[:, 1] - center_c)
        if coords.shape[0] == 4 * radius and np.all(dists == radius):
            candidates.append((int(color), (int(r0), int(c0), int(r1), int(c1)), center_r, center_c, radius))

    assert len(candidates) == 1, "expected exactly one complete diamond frame"
    return candidates[0]


def _side_directions(r: int, c: int, frame_bbox: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = frame_bbox
    dirs = []
    if r == r0:
        dirs.append((1, 0))
    if r == r1:
        dirs.append((-1, 0))
    if c == c0:
        dirs.append((0, 1))
    if c == c1:
        dirs.append((0, -1))
    assert dirs, "outside marker must lie on a side of the diamond bbox"
    return dirs


def main(grid: np.ndarray) -> np.ndarray:
    """
    Paint straight rays from colored side markers into the hollow diamond frame until an interior foreground pixel blocks each ray.

    Input structure
    - The background contains one complete one-pixel-thick diamond frame.
    - Colored marker pixels outside the diamond sit on the top, bottom, left, or right side of the frame's bounding box.
    - Any non-background, non-frame pixels inside the diamond are blockers that remain unchanged.

    Rule
    - Each outside marker emits inward perpendicular to its side of the diamond bounding box.
    - Along that row or column, paint only background pixels strictly inside the diamond with the marker color.
    - Stop a ray when it reaches an existing interior foreground pixel; keep the frame and all original colored pixels unchanged.

    Key insight
    - The diamond frame defines a strict interior, while the side markers are column or row emitters whose color fills the visible interior interval until blocked.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    frame_color, frame_bbox, center_r, center_c, radius = _diamond_frame(g, bg)
    out = g.copy()

    H, W = g.shape
    planned: dict[tuple[int, int], int] = {}
    marker_count = 0

    marker_coords = np.argwhere((g != bg) & (g != frame_color))
    for r_raw, c_raw in marker_coords:
        r = int(r_raw)
        c = int(c_raw)
        dist = abs(r - center_r) + abs(c - center_c)
        if dist < radius:
            continue
        assert dist > radius, "non-frame foreground cannot lie on the complete diamond frame"

        marker_count += 1
        color = int(g[r, c])
        for dr, dc in _side_directions(r, c, frame_bbox):
            rr = r + dr
            cc = c + dc
            entered_interior = False
            while 0 <= rr < H and 0 <= cc < W:
                inside = abs(rr - center_r) + abs(cc - center_c) < radius
                if inside:
                    entered_interior = True
                    if int(g[rr, cc]) == bg:
                        previous = planned.get((rr, cc))
                        assert previous is None or previous == color, "conflicting marker rays"
                        planned[(rr, cc)] = color
                    else:
                        break
                elif entered_interior:
                    break
                rr += dr
                cc += dc

    assert marker_count > 0, "expected at least one outside marker"
    for (r, c), color in planned.items():
        out[r, c] = color
    return out

# --------------- generator (att2) ---------------
# Canvas and frame.
MIN_RADIUS = 3
MAX_RADIUS_EXCLUSIVE = 6
BG_COLOR = 0
FRAME_COLOR = 2
MARKER_COLORS = [1, 3, 4, 5, 6, 7, 8, 9]

# Marker, blocker, and rejection sampling.
MAX_BLOCKERS_EXCLUSIVE = 5
MIN_MARKERS = 4
MAX_MARKERS = 10
GAP_PROB_DENOMINATOR = 2
MAX_GENERATION_ATTEMPTS = 300


def _gen_draw_diamond(grid: np.ndarray, radius: int) -> None:
    center = radius
    for rr in range(2 * radius + 1):
        dc = radius - abs(rr - center)
        grid[rr, center - dc] = FRAME_COLOR
        grid[rr, center + dc] = FRAME_COLOR


def _gen_active_edges(radius: int) -> list[tuple[str, list[int]]]:
    n = 2 * radius + 1
    diag = int(np.random.randint(2))
    axis = int(np.random.randint(2))

    if diag == 0 and axis == 0:
        return [('top', list(range(radius + 1, n))), ('bottom', list(range(radius)))]
    if diag == 0:
        return [('right', list(range(radius))), ('left', list(range(radius + 1, n)))]
    if axis == 0:
        return [('top', list(range(radius))), ('bottom', list(range(radius + 1, n)))]
    return [('left', list(range(radius))), ('right', list(range(radius + 1, n)))]


def _gen_with_optional_gap(positions: list[int]) -> list[int]:
    kept = positions[:]
    if len(kept) > 1 and int(np.random.randint(GAP_PROB_DENOMINATOR)) == 1:
        del kept[int(np.random.randint(len(kept)))]
    return kept


def _gen_ray_cells(side: str, pos: int, radius: int) -> list[tuple[int, int]]:
    n = 2 * radius + 1
    if side in ('top', 'bottom'):
        return [(rr, pos) for rr in range(n) if abs(rr - radius) + abs(pos - radius) < radius]
    return [(pos, cc) for cc in range(n) if abs(pos - radius) + abs(cc - radius) < radius]


def _gen_marker_coord(side: str, pos: int, radius: int) -> tuple[int, int]:
    n = 2 * radius + 1
    if side == 'top':
        return 0, pos
    if side == 'bottom':
        return n - 1, pos
    if side == 'left':
        return pos, 0
    return pos, n - 1


def _gen_place_markers(grid: np.ndarray, radius: int, marker_palette: list[int]) -> tuple[int, set[tuple[int, int]]]:
    ray_cells: set[tuple[int, int]] = set()
    marker_count = 0
    for side, positions in _gen_active_edges(radius):
        for pos in _gen_with_optional_gap(positions):
            rr, cc = _gen_marker_coord(side, pos, radius)
            grid[rr, cc] = int(marker_palette[int(np.random.randint(len(marker_palette)))])
            marker_count += 1
            ray_cells.update(_gen_ray_cells(side, pos, radius))
    return marker_count, ray_cells


def _gen_place_blockers(grid: np.ndarray, blocker_count: int, blocker_color: int | None, ray_cells: set[tuple[int, int]]) -> bool:
    if not blocker_count:
        return True

    candidates = sorted(ray_cells)
    if len(candidates) < blocker_count:
        return False

    chosen = np.random.choice(len(candidates), size=blocker_count, replace=False)
    for idx in chosen:
        rr, cc = candidates[int(idx)]
        grid[rr, cc] = blocker_color
    return True


def _gen_solver_accepts(grid: np.ndarray) -> bool:
    try:
        out = main(grid)
    except Exception:
        return False
    return out.shape == grid.shape and not np.array_equal(out, grid)


def _gen_attempt() -> np.ndarray | None:
    radius = int(np.random.randint(MIN_RADIUS, MAX_RADIUS_EXCLUSIVE))
    n = 2 * radius + 1
    grid = np.zeros((n, n), dtype=int) + BG_COLOR
    _gen_draw_diamond(grid, radius)

    blocker_count = int(np.random.randint(MAX_BLOCKERS_EXCLUSIVE))
    blocker_color = None
    marker_palette = MARKER_COLORS[:]
    if blocker_count:
        blocker_color = int(marker_palette[int(np.random.randint(len(marker_palette)))])
        marker_palette = [color for color in marker_palette if color != blocker_color]

    marker_count, ray_cells = _gen_place_markers(grid, radius, marker_palette)
    if not MIN_MARKERS <= marker_count <= MAX_MARKERS:
        return None

    if not _gen_place_blockers(grid, blocker_count, blocker_color, ray_cells):
        return None
    if not _gen_solver_accepts(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError('failed to generate valid 95755ff2 input')
