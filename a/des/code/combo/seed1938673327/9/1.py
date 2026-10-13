from __future__ import annotations

from collections import Counter, defaultdict
import random

import numpy as np


DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _dominant_color(region: np.ndarray) -> int:
    counts = Counter(map(int, region.reshape(-1)))
    ranked = counts.most_common()
    assert ranked, "expected a nonempty region"
    assert len(ranked) == 1 or ranked[0][1] > ranked[1][1], "expected a unique dominant background color"
    return int(ranked[0][0])


def _find_separator_cross(grid: np.ndarray) -> tuple[int, int, int]:
    height, width = map(int, grid.shape)
    solid_rows = [
        (row, int(grid[row, 0]))
        for row in range(height)
        if bool(np.all(grid[row, :] == int(grid[row, 0])))
    ]
    solid_cols = [
        (col, int(grid[0, col]))
        for col in range(width)
        if bool(np.all(grid[:, col] == int(grid[0, col])))
    ]

    candidates = []
    for row, row_color in solid_rows:
        for col, col_color in solid_cols:
            if row_color != col_color:
                continue
            if row == 0 or row == height - 1 or col == 0 or col == width - 1:
                continue
            if row != height - row - 1 or col != width - col - 1:
                continue
            candidates.append((int(row), int(col), int(row_color)))

    assert len(candidates) == 1, f"expected one separator cross, got {candidates}"
    return candidates[0]


def _quadrant_slices(cross_row: int, cross_col: int, height: int, width: int) -> list[tuple[slice, slice]]:
    top_h = int(cross_row)
    left_w = int(cross_col)
    bottom_h = int(height) - top_h - 1
    right_w = int(width) - left_w - 1
    assert top_h > 0 and left_w > 0 and bottom_h > 0 and right_w > 0, "quadrants must be nonempty"
    assert top_h == bottom_h and left_w == right_w, "cross must split the grid into four equal quadrants"
    return [
        (slice(0, cross_row), slice(0, cross_col)),
        (slice(0, cross_row), slice(cross_col + 1, width)),
        (slice(cross_row + 1, height), slice(0, cross_col)),
        (slice(cross_row + 1, height), slice(cross_col + 1, width)),
    ]


def _components(region: np.ndarray, bg: int) -> list[dict]:
    height, width = map(int, region.shape)
    visited = np.zeros((height, width), dtype=bool)
    out: list[dict] = []

    for row in range(height):
        for col in range(width):
            color = int(region[row, col])
            if visited[row, col] or color == int(bg):
                continue

            stack = [(int(row), int(col))]
            visited[row, col] = True
            coords: list[tuple[int, int]] = []
            while stack:
                r, c = stack.pop()
                coords.append((int(r), int(c)))
                for dr, dc in DIR4:
                    nr, nc = int(r + dr), int(c + dc)
                    if not (0 <= nr < height and 0 <= nc < width):
                        continue
                    if visited[nr, nc] or int(region[nr, nc]) != color:
                        continue
                    visited[nr, nc] = True
                    stack.append((nr, nc))

            out.append(
                {
                    "color": color,
                    "area": len(coords),
                    "coords": np.array(coords, dtype=int),
                }
            )

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Split the board by the solid cross, find the unique color missing from each quadrant,
    and recolor that quadrant's smallest foreground object with the missing color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    cross_row, cross_col, _separator = _find_separator_cross(g)
    quads = _quadrant_slices(cross_row, cross_col, *g.shape)

    infos = []
    for row_slice, col_slice in quads:
        region = g[row_slice, col_slice]
        bg = _dominant_color(region)
        comps = _components(region, bg)
        assert comps, "each quadrant must contain foreground objects"
        infos.append(
            {
                "slice": (row_slice, col_slice),
                "components": comps,
                "colors": {int(comp["color"]) for comp in comps},
            }
        )

    out = g.copy()
    color_sets = [info["colors"] for info in infos]
    for idx, info in enumerate(infos):
        other_union = set().union(*(color_sets[j] for j in range(len(infos)) if j != idx))
        missing = other_union - info["colors"]
        assert len(missing) == 1, f"quadrant {idx} should miss exactly one color, got {sorted(missing)}"
        fill_color = next(iter(missing))

        areas = [int(comp["area"]) for comp in info["components"]]
        min_area = min(areas)
        smallest = [comp for comp in info["components"] if int(comp["area"]) == min_area]
        assert len(smallest) == 1, f"quadrant {idx} should have a unique smallest object"

        row_slice, col_slice = info["slice"]
        coords = smallest[0]["coords"]
        out[row_slice.start + coords[:, 0], col_slice.start + coords[:, 1]] = int(fill_color)

    return out


# --------------- generator ---------------

COLOR_COUNT = 10
QUAD_HEIGHT_RANGE = (8, 13)
QUAD_WIDTH_RANGE = (8, 13)
FULL_COLOR_COUNT_OPTIONS = (4, 5, 6)
SMALL_AREA_RANGE = (1, 3)
LARGE_AREA_MAX = 7
MAX_MASK_ATTEMPTS = 120
MAX_PLACEMENT_ATTEMPTS = 80
MAX_QUADRANT_ATTEMPTS = 180
MAX_GENERATION_ATTEMPTS = 400


def _normalize_points(points: set[tuple[int, int]]) -> np.ndarray:
    min_row = min(row for row, _col in points)
    min_col = min(col for _row, col in points)
    shifted = sorted((row - min_row, col - min_col) for row, col in points)
    return np.array(shifted, dtype=int)


def _coords_to_mask(coords: np.ndarray) -> np.ndarray:
    height = int(coords[:, 0].max()) + 1
    width = int(coords[:, 1].max()) + 1
    mask = np.zeros((height, width), dtype=bool)
    mask[coords[:, 0], coords[:, 1]] = True
    return mask


def _random_connected_mask(area: int) -> np.ndarray | None:
    if area <= 0:
        return None

    max_dim = 3 if area <= 3 else 4 if area <= 5 else 5
    for _ in range(MAX_MASK_ATTEMPTS):
        points = {(0, 0)}
        while len(points) < int(area):
            frontier = []
            for row, col in points:
                for dr, dc in DIR4:
                    nxt = (int(row + dr), int(col + dc))
                    if nxt not in points:
                        frontier.append(nxt)
            assert frontier, "connected growth frontier should be nonempty"
            points.add(random.choice(frontier))

        coords = _normalize_points(points)
        mask = _coords_to_mask(coords)
        if mask.shape[0] > max_dim or mask.shape[1] > max_dim:
            continue

        turns = random.randrange(4)
        for _ in range(turns):
            mask = np.rot90(mask)
        if random.random() < 0.5:
            mask = np.flipud(mask)
        if random.random() < 0.5:
            mask = np.fliplr(mask)
        return np.array(mask, copy=True)

    return None


def _place_mask(occupied: np.ndarray, mask: np.ndarray) -> tuple[int, int] | None:
    height, width = map(int, occupied.shape)
    mask_h, mask_w = map(int, mask.shape)
    candidates = []
    for row in range(height - mask_h + 1):
        for col in range(width - mask_w + 1):
            if not bool(np.any(occupied[row : row + mask_h, col : col + mask_w] & mask)):
                candidates.append((int(row), int(col)))
    if not candidates:
        return None
    return random.choice(candidates)


def _build_quadrant(height: int, width: int, bg: int, full_colors: list[int], missing_color: int) -> tuple[np.ndarray, np.ndarray] | None:
    region_area = int(height * width)
    present_colors = [color for color in full_colors if color != int(missing_color)]
    donor_color = random.choice(present_colors)

    for _ in range(MAX_QUADRANT_ATTEMPTS):
        small_area = random.randint(*SMALL_AREA_RANGE)
        min_large_area = small_area + 2
        if min_large_area > LARGE_AREA_MAX:
            continue

        specs = [{"color": donor_color, "area": small_area, "is_small": True}]
        total_area = small_area
        for color in present_colors:
            area = random.randint(min_large_area, LARGE_AREA_MAX)
            specs.append({"color": int(color), "area": int(area), "is_small": False})
            total_area += int(area)
        if total_area > int(0.45 * region_area):
            continue

        masks = []
        for spec in specs:
            mask = _random_connected_mask(int(spec["area"]))
            if mask is None:
                masks = []
                break
            masks.append(mask)
        if not masks:
            continue

        order = sorted(range(len(specs)), key=lambda idx: int(specs[idx]["area"]), reverse=True)
        occupied = np.zeros((height, width), dtype=bool)
        placements: dict[int, np.ndarray] = {}
        ok = True
        for idx in order:
            placed = None
            for _ in range(MAX_PLACEMENT_ATTEMPTS):
                top_left = _place_mask(occupied, masks[idx])
                if top_left is None:
                    break
                row0, col0 = top_left
                coords = np.argwhere(masks[idx]) + np.array([row0, col0], dtype=int)
                placements[idx] = coords
                occupied[coords[:, 0], coords[:, 1]] = True
                placed = coords
                break
            if placed is None:
                ok = False
                break
        if not ok:
            continue

        region = np.full((height, width), int(bg), dtype=int)
        small_coords = None
        for idx, spec in enumerate(specs):
            coords = placements[idx]
            region[coords[:, 0], coords[:, 1]] = int(spec["color"])
            if spec["is_small"]:
                small_coords = coords
        assert small_coords is not None, "small object placement should exist"

        if _dominant_color(region) != int(bg):
            continue

        comps = _components(region, bg)
        color_to_count = defaultdict(int)
        for comp in comps:
            color_to_count[int(comp["color"])] += 1
        if set(color_to_count) != set(present_colors):
            continue
        if any(color_to_count[color] != (2 if color == donor_color else 1) for color in present_colors):
            continue

        min_area = min(int(comp["area"]) for comp in comps)
        smallest = [comp for comp in comps if int(comp["area"]) == min_area]
        if len(smallest) != 1 or int(smallest[0]["color"]) != int(donor_color) or min_area != int(small_area):
            continue

        out_region = region.copy()
        out_region[small_coords[:, 0], small_coords[:, 1]] = int(missing_color)
        out_colors = {int(color) for color in np.unique(out_region) if int(color) != int(bg)}
        if out_colors != set(full_colors):
            continue
        return region, out_region

    return None


def _attempt_instance() -> np.ndarray | None:
    quad_h = random.randint(*QUAD_HEIGHT_RANGE)
    quad_w = random.randint(*QUAD_WIDTH_RANGE)
    height = 2 * quad_h + 1
    width = 2 * quad_w + 1

    bg, separator = random.sample(range(COLOR_COUNT), 2)
    available = [color for color in range(COLOR_COUNT) if color not in (bg, separator)]
    full_color_count = random.choice(FULL_COLOR_COUNT_OPTIONS)
    full_colors = random.sample(available, full_color_count)
    missing_by_quadrant = random.sample(full_colors, 4)

    inp = np.full((height, width), int(bg), dtype=int)
    out = inp.copy()
    inp[quad_h, :] = int(separator)
    inp[:, quad_w] = int(separator)
    out[quad_h, :] = int(separator)
    out[:, quad_w] = int(separator)

    quadrants = _quadrant_slices(quad_h, quad_w, height, width)
    for (row_slice, col_slice), missing_color in zip(quadrants, missing_by_quadrant):
        built = _build_quadrant(quad_h, quad_w, bg, full_colors, int(missing_color))
        if built is None:
            return None
        region_in, region_out = built
        inp[row_slice, col_slice] = region_in
        out[row_slice, col_slice] = region_out

    try:
        if np.array_equal(main(inp), out):
            return inp
    except AssertionError:
        return None
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _attempt_instance()
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input: failed after {MAX_GENERATION_ATTEMPTS} attempts")
