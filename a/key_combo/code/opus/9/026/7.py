from __future__ import annotations

import math
import random
import sys
from pathlib import Path

import numpy as np

_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_UTILS_DIR) not in sys.path:
    sys.path.append(str(_UTILS_DIR))

from arc_utils import Color, in_bounds


BG = int(Color.BLACK)
RED = int(Color.RED)
ORANGE = int(Color.ORANGE)
MAX_GEN_ATTEMPTS = 400


def _overlap_slices(shape: tuple[int, int], dr: int, dc: int) -> tuple[int, int, int, int]:
    h, w = shape
    r0 = max(0, -dr)
    r1 = min(h, h - dr)
    c0 = max(0, -dc)
    c1 = min(w, w - dc)
    return r0, r1, c0, c1


def _best_supported_translations(grid: np.ndarray) -> list[tuple[int, int]]:
    mask = np.asarray(grid == RED, dtype=bool)
    h, w = mask.shape
    best_support = 0
    best: list[tuple[int, int]] = []
    for dr in range(-(h - 1), h):
        for dc in range(-(w - 1), w):
            if dr == 0 and dc == 0:
                continue
            r0, r1, c0, c1 = _overlap_slices(mask.shape, dr, dc)
            if r0 >= r1 or c0 >= c1:
                continue
            support = int(np.count_nonzero(mask[r0:r1, c0:c1] & mask[r0 + dr : r1 + dr, c0 + dc : c1 + dc]))
            if support == 0:
                continue
            if support > best_support:
                best_support = support
                best = [(dr, dc)]
            elif support == best_support:
                best.append((dr, dc))
    assert best_support > 0, "expected at least one repeated red-point translation"
    assert len(best) == 2 and best[0] == (-best[1][0], -best[1][1]), "translation must be uniquely determined up to sign"
    return best


def _complete_periodic_points(grid: np.ndarray, shift: tuple[int, int]) -> np.ndarray:
    dr, dc = map(int, shift)
    out = np.asarray(grid, dtype=int).copy()
    for r, c in np.argwhere(grid == RED):
        r0 = int(r)
        c0 = int(c)
        for sgn in (-1, 1):
            rr = r0
            cc = c0
            while True:
                rr += sgn * dr
                cc += sgn * dc
                if not in_bounds(out.shape, rr, cc):
                    break
                out[rr, cc] = RED
    return out


def _chebyshev(a: tuple[int, int], b: tuple[int, int]) -> int:
    return max(abs(int(a[0]) - int(b[0])), abs(int(a[1]) - int(b[1])))


def _cluster_by_proximity(
    points: list[tuple[int, int]],
    *,
    min_size: int = 4,
    max_size: int = 6,
) -> list[list[tuple[int, int]]]:
    assert len(points) >= min_size * 2
    distances = sorted({_chebyshev(points[i], points[j]) for i in range(len(points)) for j in range(i + 1, len(points)) if points[i] != points[j]})
    assert distances, "need at least two distinct red points"
    for threshold in distances:
        seen = [False] * len(points)
        comps: list[list[tuple[int, int]]] = []
        for i in range(len(points)):
            if seen[i]:
                continue
            stack = [i]
            seen[i] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cur = stack.pop()
                comp.append(points[cur])
                for j in range(len(points)):
                    if seen[j]:
                        continue
                    if _chebyshev(points[cur], points[j]) <= threshold:
                        seen[j] = True
                        stack.append(j)
            comps.append(sorted(comp))
        if len(comps) >= 2 and all(min_size <= len(comp) <= max_size for comp in comps):
            return comps
    raise AssertionError("could not recover 4-6 point clusters from spatial separation")


def _angle_cycle(points: list[tuple[int, int]]) -> list[tuple[int, int]]:
    cr = sum(r for r, _ in points) / len(points)
    cc = sum(c for _, c in points) / len(points)

    def key(p: tuple[int, int]) -> tuple[float, float, int, int]:
        r, c = p
        angle = math.atan2(-(r - cr), c - cc) % (2.0 * math.pi)
        radius = (r - cr) ** 2 + (c - cc) ** 2
        return (angle, radius, r, c)

    return sorted(points, key=key)


def _aligned_8(a: tuple[int, int], b: tuple[int, int]) -> bool:
    dr = int(b[0]) - int(a[0])
    dc = int(b[1]) - int(a[1])
    return (dr == 0 and dc != 0) or (dc == 0 and dr != 0) or (dr != 0 and abs(dr) == abs(dc))


def _segment_cells(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    dr = int(b[0]) - int(a[0])
    dc = int(b[1]) - int(a[1])
    sr = 0 if dr == 0 else (1 if dr > 0 else -1)
    sc = 0 if dc == 0 else (1 if dc > 0 else -1)
    steps = max(abs(dr), abs(dc))
    return [(int(a[0]) + sr * k, int(a[1]) + sc * k) for k in range(steps + 1)]


def _render_contours(grid: np.ndarray) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    points = [tuple(map(int, p)) for p in np.argwhere(out == RED)]
    clusters = _cluster_by_proximity(points)
    for cluster in clusters:
        cycle = _angle_cycle(cluster)
        for i in range(len(cycle)):
            a = cycle[i]
            b = cycle[(i + 1) % len(cycle)]
            if not _aligned_8(a, b):
                continue
            for rr, cc in _segment_cells(a, b):
                if out[rr, cc] == BG:
                    out[rr, cc] = ORANGE
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Complete periodic missing vertices, then connect each recovered vertex cluster into a closed outline.

    Input structure:
    - The grid is mostly black background.
    - Red pixels are sparse polygon vertices arranged as several translated copies of one cluster shape.
    - At least two clusters are complete, while another translated copy is missing one or two vertices.

    Rule:
    - Find the unique non-zero translation whose red/red overlap support is maximal.
    - Use that translation and its inverse to fill every black position that belongs to a red periodic equivalence class.
    - Partition the completed red pixels into spatially separated 4-6 point clusters.
    - For each cluster, sort vertices by polar angle around the cluster centroid starting at 0 degrees and going counterclockwise.
    - Draw orange segments between consecutive sorted vertices, including the closing segment, only when the pair is horizontal, vertical, or 45-degree diagonal; keep every red vertex unchanged.

    Key insight:
    The strongest self-translation identifies which missing red vertices must be restored, and only after that completion does each local vertex cloud reveal the full polygon cycle to trace.
    """
    g = np.asarray(grid, dtype=int)
    shifts = _best_supported_translations(g)
    completed = _complete_periodic_points(g, shifts[0])
    return _render_contours(completed)


def _transform_coords(coords: list[tuple[int, int]], rotate_k: int, flip_lr: bool) -> list[tuple[int, int]]:
    arr = np.array(coords, dtype=int)
    if flip_lr:
        arr[:, 1] *= -1
    for _ in range(int(rotate_k) % 4):
        arr = np.stack([-arr[:, 1], arr[:, 0]], axis=1)
    arr[:, 0] -= int(arr[:, 0].min())
    arr[:, 1] -= int(arr[:, 1].min())
    return [tuple(map(int, p)) for p in arr]


def _template_library() -> list[list[tuple[int, int]]]:
    return [
        [(0, 2), (2, 4), (4, 2), (2, 0)],
        [(2, 4), (0, 2), (1, 0), (3, 0), (4, 2)],
        [(2, 4), (1, 3), (1, 1), (2, 0), (4, 2)],
        [(2, 4), (1, 3), (1, 1), (2, 0), (3, 1), (3, 3)],
    ]


def _paint_points(shape: tuple[int, int], coords: list[tuple[int, int]]) -> np.ndarray:
    grid = np.full(shape, BG, dtype=int)
    for r, c in coords:
        grid[int(r), int(c)] = RED
    return grid


def _render_visible_only(grid: np.ndarray) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    points = [tuple(map(int, p)) for p in np.argwhere(out == RED)]
    try:
        clusters = _cluster_by_proximity(points, min_size=3, max_size=6)
    except AssertionError:
        return out
    for cluster in clusters:
        cycle = _angle_cycle(cluster)
        for i in range(len(cycle)):
            a = cycle[i]
            b = cycle[(i + 1) % len(cycle)]
            if not _aligned_8(a, b):
                continue
            for rr, cc in _segment_cells(a, b):
                if out[rr, cc] == BG:
                    out[rr, cc] = ORANGE
    return out


def _expected_output(full_red_grid: np.ndarray) -> np.ndarray:
    return _render_contours(np.asarray(full_red_grid, dtype=int))


def _build_instance() -> tuple[np.ndarray, np.ndarray] | None:
    template = random.choice(_template_library())
    template = _transform_coords(template, rotate_k=random.randrange(4), flip_lr=bool(random.randrange(2)))
    ordered = _angle_cycle(template)
    if not all(_aligned_8(ordered[i], ordered[(i + 1) % len(ordered)]) for i in range(len(ordered))):
        return None

    t_rows = max(r for r, _ in template) + 1
    t_cols = max(c for _, c in template) + 1
    cluster_count = 3
    if t_rows <= 3 and random.random() < 0.45:
        cluster_count = 4

    dr = random.randint(max(7, t_rows + 4), 10 if cluster_count == 3 else 8)
    dc_mag = random.randint(3, 8)
    dc = dc_mag if random.random() < 0.5 else -dc_mag

    max_r0 = dr - t_rows
    if max_r0 < 0:
        return None
    r0 = random.randint(0, max_r0)

    h_min = r0 + (cluster_count - 1) * dr + t_rows
    h_max = min(30, r0 + cluster_count * dr)
    if h_min > h_max:
        return None
    height = random.randint(h_min, h_max)

    min_col = min(k * dc for k in range(cluster_count))
    max_col = max(k * dc for k in range(cluster_count)) + t_cols - 1
    c0_low = -min_col
    c0_high = 29 - max_col
    if c0_low > c0_high:
        return None
    c0 = random.randint(c0_low, c0_high)
    w_min = max(c0 + max_col - c0 + 1, max_col + 1)
    width = random.randint(max(max_col + 1, 18), min(30, max(max_col + 1, 30)))
    if width > 30:
        return None

    full_coords: list[tuple[int, int]] = []
    cluster_points: list[list[tuple[int, int]]] = []
    for k in range(cluster_count):
        pts = []
        for tr, tc in template:
            rr = r0 + k * dr + tr
            cc = c0 + k * dc + tc
            if not (0 <= rr < height and 0 <= cc < width):
                return None
            pts.append((rr, cc))
            full_coords.append((rr, cc))
        cluster_points.append(pts)

    if len(set(full_coords)) != len(full_coords):
        return None

    incomplete_idx = random.randrange(cluster_count)
    n_missing = 1 if len(template) == 4 and incomplete_idx == 1 else random.randint(1, min(2, len(template) - 1))
    missing_indices = sorted(random.sample(range(len(template)), n_missing))
    visible_coords = [p for idx, pts in enumerate(cluster_points) for j, p in enumerate(pts) if idx != incomplete_idx or j not in missing_indices]

    full_grid = _paint_points((height, width), full_coords)
    inp = _paint_points((height, width), visible_coords)
    expected = _expected_output(full_grid)

    shifts = _best_supported_translations(inp)
    if set(shifts) != {(dr, dc), (-dr, -dc)}:
        return None

    completed = _complete_periodic_points(inp, (dr, dc))
    if not np.array_equal(completed, full_grid):
        return None

    full_clusters = _cluster_by_proximity([tuple(map(int, p)) for p in np.argwhere(full_grid == RED)])
    if len(full_clusters) != cluster_count or any(len(comp) != len(template) for comp in full_clusters):
        return None

    if np.array_equal(_render_visible_only(inp), expected):
        return None

    if np.array_equal(inp, full_grid):
        return None

    if np.array_equal(expected, full_grid):
        return None

    if not np.array_equal(main(inp), expected):
        return None

    return inp, expected


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        built = _build_instance()
        if built is not None:
            return built[0]
    raise RuntimeError("failed to generate a valid ARC input for puzzle 7")
