from __future__ import annotations

import itertools
import math
import random

import numpy as np


BG = 0
AREA_RANGE = range(2, 7)
MAX_ATTEMPTS = 600


def _ordered_points(coords: list[tuple[int, int]] | tuple[tuple[int, int], ...]) -> list[tuple[int, int]]:
    pts = [(int(r), int(c)) for r, c in coords]
    cr = sum(r for r, _c in pts) / len(pts)
    cc = sum(c for _r, c in pts) / len(pts)
    return sorted(pts, key=lambda p: math.atan2(p[0] - cr, p[1] - cc))


def _double_area(coords: list[tuple[int, int]] | tuple[tuple[int, int], ...]) -> int:
    pts = _ordered_points(coords)
    total = 0
    for (r0, c0), (r1, c1) in zip(pts, pts[1:] + pts[:1]):
        total += c0 * r1 - c1 * r0
    return abs(int(total))


def _area_value(coords: list[tuple[int, int]] | tuple[tuple[int, int], ...]) -> int:
    return int(_double_area(coords) // 2)


def _render_squares(items: list[dict]) -> np.ndarray:
    ordered = sorted(
        items,
        key=lambda d: (
            -int(d["area"]),
            float(d["centroid"][1]),
            float(d["centroid"][0]),
            tuple(d["coords"]),
        ),
    )
    out_h = max(int(d["area"]) for d in ordered)
    out_w = sum(int(d["area"]) for d in ordered) + len(ordered) - 1
    out = np.full((out_h, out_w), BG, dtype=int)

    c0 = 0
    for d in ordered:
        side = int(d["area"])
        out[:side, c0 : c0 + side] = int(d["color"])
        c0 += side + 1
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Turn same-colored scatter points into area-sized filled squares.

    Input structure:
    - Black background.
    - Three or four foreground colors are present.
    - Each foreground color appears as three or four isolated pixels, with no overlapping pixels.

    Rule:
    - For each foreground color, collect all pixels of that color.
    - Sort those pixels clockwise by polar angle around their color group's centroid.
    - Compute the polygon area using the shoelace formula, then floor it to an integer side length.
    - Draw filled squares from left to right in descending area order, using one black separator column between squares.

    Key insight:
    - The scattered pixels are not objects to move; each color's point cloud defines an implicit polygon whose area becomes the output square size.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    items: list[dict] = []
    for color in [int(c) for c in np.unique(g) if int(c) != BG]:
        coords_arr = np.argwhere(g == int(color))
        coords = [(int(r), int(c)) for r, c in coords_arr]
        assert len(coords) >= 3, "each color must provide at least three polygon points"
        area = _area_value(coords)
        assert area > 0, "polygon points must enclose positive area"
        centroid = (
            sum(r for r, _c in coords) / len(coords),
            sum(c for _r, c in coords) / len(coords),
        )
        items.append(
            {
                "color": int(color),
                "coords": tuple(sorted(coords)),
                "centroid": centroid,
                "area": int(area),
            }
        )

    assert items, "expected foreground colors"
    return _render_squares(items)


def _normalize(coords: list[tuple[int, int]] | tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    min_r = min(r for r, _c in coords)
    min_c = min(c for _r, c in coords)
    return tuple(sorted((int(r - min_r), int(c - min_c)) for r, c in coords))


def _cross(o: tuple[int, int], a: tuple[int, int], b: tuple[int, int]) -> int:
    ox, oy = o[1], o[0]
    ax, ay = a[1], a[0]
    bx, by = b[1], b[0]
    return (ax - ox) * (by - oy) - (ay - oy) * (bx - ox)


def _convex_hull(coords: tuple[tuple[int, int], ...]) -> list[tuple[int, int]]:
    pts = sorted(set(coords), key=lambda p: (p[1], p[0]))
    if len(pts) <= 1:
        return pts

    lower: list[tuple[int, int]] = []
    for p in pts:
        while len(lower) >= 2 and _cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)

    upper: list[tuple[int, int]] = []
    for p in reversed(pts):
        while len(upper) >= 2 and _cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)

    return lower[:-1] + upper[:-1]


def _make_templates() -> dict[tuple[int, int], list[tuple[tuple[int, int], ...]]]:
    cells = [(r, c) for r in range(6) for c in range(6)]
    templates: dict[tuple[int, int], list[tuple[tuple[int, int], ...]]] = {
        (n, area): [] for n in (3, 4) for area in AREA_RANGE
    }
    seen: set[tuple[tuple[int, int], ...]] = set()

    for n in (3, 4):
        for combo in itertools.combinations(cells, n):
            norm = _normalize(combo)
            if norm in seen:
                continue
            seen.add(norm)
            if len(_convex_hull(norm)) != n:
                continue
            if any(
                abs(r0 - r1) + abs(c0 - c1) < 2
                for (r0, c0), (r1, c1) in itertools.combinations(norm, 2)
            ):
                continue

            double_area = _double_area(norm)
            if double_area % 2:
                continue
            area = double_area // 2
            if area in AREA_RANGE:
                templates[(n, int(area))].append(norm)
    return templates


TEMPLATES = _make_templates()


def _transform(coords: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    pts = list(coords)
    if random.random() < 0.5:
        pts = [(r, -c) for r, c in pts]
    for _ in range(random.randrange(4)):
        pts = [(c, -r) for r, c in pts]
    return _normalize(pts)


def _too_close_to_existing(
    points: list[tuple[int, int]], occupied: np.ndarray, gap: int = 1
) -> bool:
    h, w = occupied.shape
    for r, c in points:
        r0 = max(0, int(r) - gap)
        r1 = min(h, int(r) + gap + 1)
        c0 = max(0, int(c) - gap)
        c1 = min(w, int(c) + gap + 1)
        if bool(occupied[r0:r1, c0:c1].any()):
            return True
    return False


def _place_points(
    grid: np.ndarray,
    occupied: np.ndarray,
    local_coords: tuple[tuple[int, int], ...],
    color: int,
) -> tuple[tuple[int, int], ...] | None:
    h, w = grid.shape
    ph = max(r for r, _c in local_coords) + 1
    pw = max(c for _r, c in local_coords) + 1
    if ph + 2 > h or pw + 2 > w:
        return None

    candidates = [
        (r0, c0)
        for r0 in range(1, h - ph)
        for c0 in range(1, w - pw)
    ]
    random.shuffle(candidates)

    for r0, c0 in candidates[:240]:
        points = [(int(r0 + r), int(c0 + c)) for r, c in local_coords]
        if _too_close_to_existing(points, occupied):
            continue
        for r, c in points:
            grid[r, c] = int(color)
            occupied[r, c] = True
        return tuple(sorted(points))
    return None


def _expected_items(grid: np.ndarray) -> list[dict]:
    items: list[dict] = []
    for color in [int(c) for c in np.unique(grid) if int(c) != BG]:
        coords = [(int(r), int(c)) for r, c in np.argwhere(grid == int(color))]
        items.append(
            {
                "color": int(color),
                "coords": tuple(sorted(coords)),
                "centroid": (
                    sum(r for r, _c in coords) / len(coords),
                    sum(c for _r, c in coords) / len(coords),
                ),
                "area": _area_value(coords),
            }
        )
    return items


def _gen_attempt() -> np.ndarray | None:
    n_groups = random.choice([3, 4])
    h = random.randint(13, 30)
    w = random.randint(13, 30)
    grid = np.full((h, w), BG, dtype=int)
    occupied = np.zeros((h, w), dtype=bool)

    colors = random.sample(range(1, 10), n_groups)
    areas = random.sample(list(AREA_RANGE), n_groups)
    point_counts = [random.choice([3, 4]) for _ in range(n_groups)]

    groups = list(zip(colors, areas, point_counts))
    random.shuffle(groups)

    placed_areas: list[int] = []
    for color, area, n_points in groups:
        choices = TEMPLATES.get((int(n_points), int(area)), [])
        if not choices:
            return None
        local = _transform(random.choice(choices))
        coords = _place_points(grid, occupied, local, int(color))
        if coords is None:
            return None
        placed_areas.append(_area_value(coords))

    if len(set(placed_areas)) != n_groups:
        return None
    if min(placed_areas) < min(AREA_RANGE) or max(placed_areas) > max(AREA_RANGE):
        return None

    items = _expected_items(grid)
    if any(len(d["coords"]) not in (3, 4) for d in items):
        return None

    expected = _render_squares(items)
    if max(expected.shape) > 30:
        return None

    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid polygon-area sample after {MAX_ATTEMPTS} attempts")
