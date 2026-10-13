from __future__ import annotations

from collections import Counter
import random

import numpy as np


def get_bg(grid: np.ndarray) -> int:
    """Return the most common grid value."""
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    """BBox of an (N,2) coordinate array using inclusive r/c bounds."""
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _diamond_frame(
    grid: np.ndarray, bg: int
) -> tuple[int, tuple[int, int, int, int], int, int, int]:
    candidates = []
    for color in sorted(set(map(int, np.unique(grid))) - {int(bg)}):
        coords = np.argwhere(grid == color)
        r0, c0, r1, c1 = bbox_coords(coords)
        height = int(r1 - r0 + 1)
        width = int(c1 - c0 + 1)
        if height != width or height % 2 == 0:
            continue

        radius = height // 2
        center_r = int(r0 + radius)
        center_c = int(c0 + radius)
        dists = np.abs(coords[:, 0] - center_r) + np.abs(coords[:, 1] - center_c)
        if coords.shape[0] == 4 * radius and np.all(dists == radius):
            candidates.append((int(color), (r0, c0, r1, c1), center_r, center_c, radius))

    assert len(candidates) == 1, "expected exactly one complete diamond frame"
    return candidates[0]


def _inside_diamond(r: int, c: int, center_r: int, center_c: int, radius: int) -> bool:
    return abs(int(r) - int(center_r)) + abs(int(c) - int(center_c)) < int(radius)


def _marker_specs(
    bbox: tuple[int, int, int, int], center_r: int, center_c: int
) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    r0, c0, r1, c1 = bbox
    return [
        ((r0 - 1, center_c), (1, 0)),
        ((r1 + 1, center_c), (-1, 0)),
        ((center_r, c0 - 1), (0, 1)),
        ((center_r, c1 + 1), (0, -1)),
    ]


def _majority_color(
    grid: np.ndarray,
    bg: int,
    frame_color: int,
    center_r: int,
    center_c: int,
    radius: int,
) -> int:
    counts: Counter[int] = Counter()
    for r, c in np.argwhere(grid != bg):
        rr = int(r)
        cc = int(c)
        value = int(grid[rr, cc])
        if value == frame_color:
            continue
        if _inside_diamond(rr, cc, center_r, center_c, radius):
            counts[value] += 1

    assert 2 <= len(counts) <= 3, "expected 2-3 interior colors"
    ranking = counts.most_common()
    assert ranking[0][1] > ranking[1][1], "expected a unique highest-frequency color"
    return int(ranking[0][0])


def _filter_interior(
    grid: np.ndarray,
    bg: int,
    frame_color: int,
    center_r: int,
    center_c: int,
    radius: int,
    keep_color: int,
) -> tuple[np.ndarray, set[tuple[int, int]]]:
    out = grid.copy()
    erased: set[tuple[int, int]] = set()
    for r, c in np.argwhere(grid != bg):
        rr = int(r)
        cc = int(c)
        value = int(grid[rr, cc])
        if value == frame_color or value == keep_color:
            continue
        if _inside_diamond(rr, cc, center_r, center_c, radius):
            out[rr, cc] = bg
            erased.add((rr, cc))
    return out, erased


def _plan_rays(
    base: np.ndarray,
    bg: int,
    bbox: tuple[int, int, int, int],
    center_r: int,
    center_c: int,
    radius: int,
) -> dict[tuple[int, int], int]:
    height, width = map(int, base.shape)
    planned: dict[tuple[int, int], int] = {}

    for (mr, mc), (dr, dc) in _marker_specs(bbox, center_r, center_c):
        assert 0 <= mr < height and 0 <= mc < width, "marker must be in bounds"
        color = int(base[mr, mc])
        assert color in MARKER_COLORS, "expected a red or green outside marker"

        rr = mr + dr
        cc = mc + dc
        entered = False
        while 0 <= rr < height and 0 <= cc < width:
            if _inside_diamond(rr, cc, center_r, center_c, radius):
                entered = True
                value = int(base[rr, cc])
                if value == bg:
                    previous = planned.get((rr, cc))
                    assert previous is None or previous == color, "conflicting ray colors"
                    planned[(rr, cc)] = color
                else:
                    break
            elif entered:
                break
            rr += dr
            cc += dc

    return planned


def _apply_planned(base: np.ndarray, planned: dict[tuple[int, int], int]) -> np.ndarray:
    out = base.copy()
    for (r, c), color in planned.items():
        out[r, c] = color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Erase all non-majority colors inside the diamond, then let the side markers shoot farther rays through the cleared gaps.

    Input structure
    - The background is black and contains one complete one-pixel-thick cyan diamond frame.
    - Four red or green marker pixels sit just outside the midpoint of the frame bbox's top, bottom, left, and right sides.
    - Inside the diamond are 6-10 single-pixel foreground points drawn with 2-3 colors from blue, yellow, and orange.

    Rule
    - Count the colors of the strict interior pixels inside the diamond, ignoring background and frame cells, and keep only the unique highest-frequency color.
    - Erase every other interior color to background.
    - From each outside marker, shoot a straight inward ray along the center row or center column, painting only strict-interior background cells with the marker color.
    - Stop each ray at the first non-background cell left after the erase step, without overwriting that blocker; keep the frame, surviving majority pixels, and outside markers unchanged.

    Key insight
    - The frequency filter changes which interior cells still block the rays, so erased minority pixels open longer tunnels and the final fill pattern reveals which color survived the count comparison.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = get_bg(g)
    frame_color, bbox, center_r, center_c, radius = _diamond_frame(g, bg)
    keep_color = _majority_color(g, bg, frame_color, center_r, center_c, radius)
    filtered, _erased = _filter_interior(g, bg, frame_color, center_r, center_c, radius, keep_color)
    planned = _plan_rays(filtered, bg, bbox, center_r, center_c, radius)
    return _apply_planned(filtered, planned)


BG_COLOR = 0
FRAME_COLOR = 8
MARKER_COLORS = (2, 3)
INNER_COLORS = (1, 4, 7)
RADIUS_CHOICES = (5, 6)
GRID_MARGIN_MIN = 1
GRID_MARGIN_MAX = 3
TOTAL_POINT_RANGE = (6, 10)
MAX_GENERATION_ATTEMPTS = 400
SIDES = ("top", "bottom", "left", "right")


def _positive_parts(total: int, count: int) -> list[tuple[int, ...]]:
    if count == 1:
        return [(total,)]
    out = []
    for head in range(1, total - count + 2):
        for tail in _positive_parts(total - head, count - 1):
            out.append((head,) + tail)
    return out


COUNT_CHOICES: dict[int, list[tuple[int, ...]]] = {2: [], 3: []}
for color_count in (2, 3):
    for total in range(TOTAL_POINT_RANGE[0], TOTAL_POINT_RANGE[1] + 1):
        for majority_count in range(3, total - (color_count - 1) + 1):
            minority_total = total - majority_count
            if minority_total < 2:
                continue
            for rest in _positive_parts(minority_total, color_count - 1):
                if all(value < majority_count for value in rest):
                    COUNT_CHOICES[color_count].append((majority_count,) + rest)


def _draw_diamond(grid: np.ndarray, r0: int, c0: int, radius: int) -> None:
    center_r = r0 + radius
    center_c = c0 + radius
    for rr in range(r0, r0 + 2 * radius + 1):
        spread = radius - abs(rr - center_r)
        grid[rr, center_c - spread] = FRAME_COLOR
        grid[rr, center_c + spread] = FRAME_COLOR


def _interior_coords(center_r: int, center_c: int, radius: int) -> list[tuple[int, int]]:
    coords = []
    for dr in range(-radius + 1, radius):
        for dc in range(-radius + 1, radius):
            if abs(dr) + abs(dc) < radius:
                coords.append((center_r + dr, center_c + dc))
    return coords


def _arm_coords(center_r: int, center_c: int, radius: int, side: str) -> list[tuple[int, int]]:
    if side == "top":
        return [(center_r - delta, center_c) for delta in range(radius - 1, 0, -1)]
    if side == "bottom":
        return [(center_r + delta, center_c) for delta in range(radius - 1, 0, -1)]
    if side == "left":
        return [(center_r, center_c - delta) for delta in range(radius - 1, 0, -1)]
    return [(center_r, center_c + delta) for delta in range(radius - 1, 0, -1)]


def _sample_counts() -> tuple[list[int], dict[int, int]]:
    color_count = 2 if random.random() < 0.5 else 3
    colors = random.sample(list(INNER_COLORS), color_count)
    majority_color = colors[0]
    rest_colors = colors[1:]
    counts = random.choice(COUNT_CHOICES[color_count])
    color_counts = {majority_color: counts[0]}
    for color, count in zip(rest_colors, counts[1:]):
        color_counts[color] = count
    return colors, color_counts


def _choose_ray_blockers(
    center_r: int,
    center_c: int,
    radius: int,
    minority_counts: dict[int, int],
) -> dict[tuple[int, int], int]:
    minority_total = sum(minority_counts.values())
    max_blockers = min(4, minority_total - 1)
    blocker_count = random.randint(1, max_blockers)
    chosen_sides = random.sample(list(SIDES), blocker_count)

    pool = []
    for color, count in minority_counts.items():
        pool.extend([color] * count)
    ray_colors = random.sample(pool, blocker_count)

    blockers: dict[tuple[int, int], int] = {}
    for side, color in zip(chosen_sides, ray_colors):
        coord = random.choice(_arm_coords(center_r, center_c, radius, side))
        blockers[coord] = int(color)
        minority_counts[int(color)] -= 1
    return blockers


def _semantic_ok(grid: np.ndarray) -> bool:
    try:
        bg = get_bg(grid)
        frame_color, bbox, center_r, center_c, radius = _diamond_frame(grid, bg)
        if frame_color != FRAME_COLOR:
            return False

        expected_markers = {pos for pos, _ in _marker_specs(bbox, center_r, center_c)}
        if int(np.count_nonzero((grid == MARKER_COLORS[0]) | (grid == MARKER_COLORS[1]))) != 4:
            return False
        actual_markers = {
            (int(r), int(c))
            for r, c in np.argwhere((grid == MARKER_COLORS[0]) | (grid == MARKER_COLORS[1]))
            if (int(r), int(c)) in expected_markers
        }
        if actual_markers != expected_markers:
            return False

        keep_color = _majority_color(grid, bg, frame_color, center_r, center_c, radius)
        filtered, erased = _filter_interior(
            grid, bg, frame_color, center_r, center_c, radius, keep_color
        )
        if not erased:
            return False

        ray_cross = {
            (r, c)
            for r, c in _interior_coords(center_r, center_c, radius)
            if r == center_r or c == center_c
        }
        if not (erased & ray_cross):
            return False

        planned_filtered = _plan_rays(filtered, bg, bbox, center_r, center_c, radius)
        planned_unfiltered = _plan_rays(grid, bg, bbox, center_r, center_c, radius)
        if not any(
            planned_filtered.get(cell) != planned_unfiltered.get(cell)
            for cell in set(planned_filtered) | set(planned_unfiltered)
        ):
            return False

        out = main(grid)
        if np.array_equal(out, grid):
            return False
        if not np.array_equal(out, _apply_planned(filtered, planned_filtered)):
            return False
        if not any(grid[cell] == bg for cell in planned_filtered):
            return False
        return True
    except Exception:
        return False


def _gen_attempt() -> np.ndarray | None:
    radius = random.choice(RADIUS_CHOICES)
    span = 2 * radius + 1

    height = span + random.randint(2 * GRID_MARGIN_MIN, 2 * GRID_MARGIN_MAX)
    width = span + random.randint(2 * GRID_MARGIN_MIN, 2 * GRID_MARGIN_MAX)
    top = random.randint(GRID_MARGIN_MIN, height - span - GRID_MARGIN_MIN)
    left = random.randint(GRID_MARGIN_MIN, width - span - GRID_MARGIN_MIN)
    center_r = top + radius
    center_c = left + radius

    grid = np.full((height, width), BG_COLOR, dtype=int)
    _draw_diamond(grid, top, left, radius)

    for (mr, mc), _dir in _marker_specs((top, left, top + span - 1, left + span - 1), center_r, center_c):
        grid[mr, mc] = random.choice(MARKER_COLORS)

    colors, color_counts = _sample_counts()
    majority_color = colors[0]
    minority_counts = {color: count for color, count in color_counts.items() if color != majority_color}

    placements: dict[tuple[int, int], int] = {(center_r, center_c): majority_color}
    color_counts[majority_color] -= 1
    placements.update(_choose_ray_blockers(center_r, center_c, radius, minority_counts))
    for color, count in minority_counts.items():
        color_counts[color] = count

    off_ray_candidates = [
        coord
        for coord in _interior_coords(center_r, center_c, radius)
        if coord not in placements and coord[0] != center_r and coord[1] != center_c
    ]
    remaining_total = sum(color_counts.values())
    if remaining_total > len(off_ray_candidates):
        return None

    chosen_coords = random.sample(off_ray_candidates, remaining_total)
    index = 0
    for color in colors:
        for _ in range(color_counts[color]):
            placements[chosen_coords[index]] = color
            index += 1

    for (r, c), color in placements.items():
        grid[r, c] = color

    if not _semantic_ok(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
