from __future__ import annotations

from collections import Counter
import random

import numpy as np


BG = 0
GREY = 5
TARGET_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
ORIENTATIONS = ("tl", "tr", "bl", "br")
MAX_ATTEMPTS = 400


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _distance_map(shape: tuple[int, int], center: tuple[int, int]) -> np.ndarray:
    height, width = map(int, shape)
    cr, cc = map(int, center)
    rows = np.abs(np.arange(height)[:, None] - cr)
    cols = np.abs(np.arange(width)[None, :] - cc)
    return np.maximum(rows, cols)


def _compress_profile(profile: list[int]) -> list[tuple[int, int]]:
    bands: list[tuple[int, int]] = []
    i = 0
    while i < len(profile):
        color = int(profile[i])
        j = i + 1
        while j < len(profile) and int(profile[j]) == color:
            j += 1
        bands.append((color, j - i))
        i = j
    return bands


def _band_starts(bands: list[tuple[int, int]]) -> list[int]:
    starts: list[int] = []
    cursor = 0
    for _color, thickness in bands:
        starts.append(cursor)
        cursor += int(thickness)
    return starts


def _color_at_radius(bands: list[tuple[int, int]], radius: int) -> int:
    period = sum(int(thickness) for _color, thickness in bands)
    offset = int(radius) % period
    cursor = 0
    for color, thickness in bands:
        thickness = int(thickness)
        if offset < cursor + thickness:
            return int(color)
        cursor += thickness
    raise AssertionError("radius must fall into one band")


def _all_candidates(grid: np.ndarray) -> list[tuple[tuple[int, int], list[tuple[int, int]], np.ndarray]]:
    g = np.asarray(grid, dtype=int)
    height, width = map(int, g.shape)
    target_mask = (g != BG) & (g != GREY)
    target_coords = np.argwhere(target_mask)
    assert target_coords.size > 0, "expected visible target pixels"

    candidates: list[tuple[tuple[int, int], list[tuple[int, int]], np.ndarray]] = []
    for cr in range(height):
        for cc in range(width):
            dist = _distance_map((height, width), (cr, cc))
            outer_radius = int(dist[target_mask].max())
            if not np.array_equal(dist <= outer_radius, target_mask):
                continue

            radial_colors: list[int] = []
            valid = True
            for radius in range(outer_radius + 1):
                vals = np.unique(g[(dist == radius) & target_mask])
                if len(vals) != 1:
                    valid = False
                    break
                radial_colors.append(int(vals[0]))
            if not valid:
                continue

            bands = _compress_profile(radial_colors)
            colors = [color for color, _thickness in bands]
            thicknesses = [thickness for _color, thickness in bands]
            if len(bands) < 3:
                continue
            if len(colors) != len(set(colors)):
                continue
            if len(set(thicknesses)) < 2:
                continue
            candidates.append(((cr, cc), bands, dist))
    return candidates


def _infer_scene(grid: np.ndarray) -> tuple[tuple[int, int], list[tuple[int, int]], np.ndarray]:
    candidates = _all_candidates(grid)
    assert len(candidates) == 1, "expected a unique center and band sequence"
    return candidates[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor every gray marker by extending the truncated square-target bands outward in Chebyshev distance.

    Input structure
    - A black background contains a clipped concentric square target whose center sits near one corner.
    - The visible target is a complete radius prefix: every in-grid cell up to one outer Chebyshev radius is colored,
      but the larger rings are truncated by the grid edges.
    - Separate gray singleton pixels lie on the surrounding black field.

    Rule
    - Infer the target center and read the visible rings as consecutive Chebyshev-distance bands from radius 0 outward.
    - Keep each band's color and thickness, and repeat that thickness-color sequence periodically for all larger radii.
    - Recolor every gray pixel with the color of its repeated-radius band; leave the visible target unchanged.

    Key insight
    - The colored corner arcs are not independent shapes: they are the clipped prefix of a full-plane Chebyshev radial field,
      so solving requires recovering the hidden center and the unequal band thicknesses before extrapolating.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert _get_bg(g) == BG, "expected black to remain the dominant background"

    center, bands, dist = _infer_scene(g)
    _ = center

    out = g.copy()
    for r, c in np.argwhere(g == GREY):
        out[int(r), int(c)] = _color_at_radius(bands, int(dist[int(r), int(c)]))
    return out


def _sample_bands() -> list[tuple[int, int]]:
    for _ in range(200):
        band_count = random.randint(3, 5)
        thicknesses = [random.randint(1, 3) for _ in range(band_count)]
        period = sum(thicknesses)
        if not (5 <= period <= 9):
            continue
        if len(set(thicknesses)) < 2:
            continue
        colors = random.sample(TARGET_COLORS, band_count)
        return [(int(color), int(thickness)) for color, thickness in zip(colors, thicknesses)]
    raise RuntimeError("failed to sample a valid band sequence")


def _paint_target(grid: np.ndarray, center: tuple[int, int], bands: list[tuple[int, int]]) -> np.ndarray:
    dist = _distance_map(grid.shape, center)
    cursor = 0
    for color, thickness in bands:
        band_mask = (cursor <= dist) & (dist < cursor + int(thickness))
        grid[band_mask] = int(color)
        cursor += int(thickness)
    grid[dist >= cursor] = BG
    return dist


def _marker_templates(center: tuple[int, int], radius: int) -> list[tuple[str, tuple[int, int]]]:
    cr, cc = map(int, center)
    radius = int(radius)
    offsets = [0, max(1, radius // 3), max(1, radius // 2), max(1, radius - 2)]
    candidates: list[tuple[str, tuple[int, int]]] = [
        ("corner", (cr + radius, cc + radius)),
        ("axis", (cr + radius, cc)),
        ("axis", (cr, cc + radius)),
    ]
    for offset in offsets:
        if 0 < offset < radius:
            candidates.append(("edge", (cr + radius, cc + offset)))
            candidates.append(("edge", (cr + offset, cc + radius)))
    seen: set[tuple[int, int]] = set()
    unique: list[tuple[str, tuple[int, int]]] = []
    for cls, coord in candidates:
        if coord in seen:
            continue
        seen.add(coord)
        unique.append((cls, coord))
    return unique


def _cheb_gap(a: tuple[int, int], b: tuple[int, int]) -> int:
    return max(abs(int(a[0]) - int(b[0])), abs(int(a[1]) - int(b[1])))


def _choose_gray_cells(
    grid: np.ndarray,
    center: tuple[int, int],
    bands: list[tuple[int, int]],
) -> list[tuple[int, int]]:
    height, width = map(int, grid.shape)
    starts = _band_starts(bands)
    period = sum(thickness for _color, thickness in bands)
    chosen: list[tuple[int, int]] = []
    chosen_classes: list[str] = []
    class_preferences = [
        ("corner", "edge", "axis"),
        ("axis", "edge", "corner"),
        ("edge", "axis", "corner"),
        ("corner", "axis", "edge"),
        ("edge", "corner", "axis"),
    ]

    for band_index, (_color, thickness) in enumerate(bands):
        options: list[tuple[str, tuple[int, int], int]] = []
        for radius in range(period + starts[band_index], period + starts[band_index] + int(thickness)):
            for cls, coord in _marker_templates(center, radius):
                r, c = coord
                if not (0 <= r < height and 0 <= c < width):
                    continue
                if int(grid[r, c]) != BG:
                    continue
                if any(_cheb_gap(coord, other) <= 1 for other in chosen):
                    continue
                options.append((cls, coord, radius))
        if not options:
            return []

        preference = class_preferences[band_index % len(class_preferences)]
        random.shuffle(options)
        options.sort(key=lambda item: (preference.index(item[0]), item[2]))
        cls, coord, _radius = options[0]
        chosen.append(coord)
        chosen_classes.append(cls)

    extra_pool: list[tuple[str, tuple[int, int], int]] = []
    for extra_radius in range(2 * period, 3 * period):
        for cls, coord in _marker_templates(center, extra_radius):
            r, c = coord
            if not (0 <= r < height and 0 <= c < width):
                continue
            if int(grid[r, c]) != BG:
                continue
            if any(_cheb_gap(coord, other) <= 1 for other in chosen):
                continue
            extra_pool.append((cls, coord, extra_radius))
    if extra_pool and random.random() < 0.7:
        random.shuffle(extra_pool)
        chosen.append(extra_pool[0][1])
        chosen_classes.append(extra_pool[0][0])

    if "corner" not in chosen_classes or "edge" not in chosen_classes:
        return []
    return chosen


def _apply_orientation(grid: np.ndarray, orientation: str) -> np.ndarray:
    if orientation == "tl":
        return grid.copy()
    if orientation == "tr":
        return np.fliplr(grid).copy()
    if orientation == "bl":
        return np.flipud(grid).copy()
    if orientation == "br":
        return np.flipud(np.fliplr(grid)).copy()
    raise ValueError("invalid orientation")


def _transform_coord(coord: tuple[int, int], shape: tuple[int, int], orientation: str) -> tuple[int, int]:
    r, c = map(int, coord)
    height, width = map(int, shape)
    if orientation in {"tr", "br"}:
        c = width - 1 - c
    if orientation in {"bl", "br"}:
        r = height - 1 - r
    return int(r), int(c)


def _target_touches_expected_corner(mask: np.ndarray, orientation: str) -> bool:
    if orientation == "tl":
        return bool(mask[0].any() and mask[:, 0].any() and not mask[-1].any() and not mask[:, -1].any())
    if orientation == "tr":
        return bool(mask[0].any() and mask[:, -1].any() and not mask[-1].any() and not mask[:, 0].any())
    if orientation == "bl":
        return bool(mask[-1].any() and mask[:, 0].any() and not mask[0].any() and not mask[:, -1].any())
    if orientation == "br":
        return bool(mask[-1].any() and mask[:, -1].any() and not mask[0].any() and not mask[:, 0].any())
    raise ValueError("invalid orientation")


def _build_expected(grid: np.ndarray, center: tuple[int, int], bands: list[tuple[int, int]]) -> np.ndarray:
    out = grid.copy()
    dist = _distance_map(grid.shape, center)
    for r, c in np.argwhere(out == GREY):
        out[int(r), int(c)] = _color_at_radius(bands, int(dist[int(r), int(c)]))
    return out


def _gen_attempt() -> np.ndarray | None:
    bands = _sample_bands()
    period = sum(thickness for _color, thickness in bands)

    max_center_coord = min(4, period - 2)
    if max_center_coord < 1:
        return None

    center_row = random.randint(1, max_center_coord)
    center_col_choices = [value for value in range(1, max_center_coord + 1) if value != center_row]
    center_col = random.choice(center_col_choices) if center_col_choices else center_row
    center = (center_row, center_col)

    height_min = center_row + 2 * period + 1
    width_min = center_col + 2 * period + 1
    if height_min > 30 or width_min > 30:
        return None
    height = random.randint(height_min, min(30, height_min + 4))
    width = random.randint(width_min, min(30, width_min + 4))
    if height == width:
        if width < 30:
            width += 1
        elif height > height_min:
            height -= 1

    grid = np.full((height, width), BG, dtype=int)
    dist = _paint_target(grid, center, bands)

    outer_radius = period - 1
    target_mask = dist <= outer_radius
    if not (target_mask[0].any() and target_mask[:, 0].any()):
        return None
    if target_mask[-1].any() or target_mask[:, -1].any():
        return None

    gray_cells = _choose_gray_cells(grid, center, bands)
    if len(gray_cells) < len(bands):
        return None
    for r, c in gray_cells:
        grid[int(r), int(c)] = GREY

    orientation = random.choice(ORIENTATIONS)
    oriented = _apply_orientation(grid, orientation)
    oriented_center = _transform_coord(center, grid.shape, orientation)
    expected = _build_expected(oriented, oriented_center, bands)

    if not _target_touches_expected_corner((oriented != BG) & (oriented != GREY), orientation):
        return None

    inferred = _all_candidates(oriented)
    if len(inferred) != 1:
        return None
    inferred_center, inferred_bands, _dist = inferred[0]
    if inferred_center != oriented_center or inferred_bands != bands:
        return None

    gray_out_colors = {int(expected[r, c]) for r, c in np.argwhere(oriented == GREY)}
    if gray_out_colors != {color for color, _thickness in bands}:
        return None

    if not np.array_equal(main(oriented), expected):
        return None
    return oriented


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid truncated-target extrapolation input")
