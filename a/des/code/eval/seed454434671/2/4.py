from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np


BG = 0
WALL = 5
LIQUID_COLORS = (1, 2, 3, 4)
WIDTH_CHOICES = (3, 5)
HEIGHT_RANGE = (6, 11)
GAP_CHOICES = (1, 2)
SIDE_MARGIN_RANGE = (1, 5)
TOP_MARGIN_RANGE = (1, 7)
MIN_CONTAINERS = 4
MAX_CONTAINERS = 8
MAX_GROUP_SIZE = 3
MAX_GENERATION_ATTEMPTS = 300
DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))

_RNG = np.random.default_rng()


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(int(r), int(c))]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((int(rr), int(cc)))
                for dr, dc in DIR4:
                    nr = int(rr + dr)
                    nc = int(cc + dc)
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(coords)
    return comps


def _extract_container(
    grid: np.ndarray, coords: list[tuple[int, int]], wall_color: int, bg: int
) -> dict[str, int] | None:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    top = min(rows)
    bottom = max(rows)
    left = min(cols)
    right = max(cols)
    height = int(bottom - top + 1)
    width = int(right - left + 1)
    if width < 3 or height < 3:
        return None

    patch = grid[top : bottom + 1, left : right + 1]
    expected = np.zeros((height, width), dtype=bool)
    expected[:, 0] = True
    expected[:, -1] = True
    expected[-1, :] = True
    if not np.array_equal(patch == wall_color, expected):
        return None

    interior = patch[:-1, 1:-1]
    if interior.size == 0:
        return None

    marker_row = interior[-1]
    marker_cols = np.flatnonzero(marker_row != bg)
    if marker_cols.size != 1:
        return None
    marker_col = int(marker_cols[0])
    marker_color = int(marker_row[marker_col])
    if marker_color == wall_color:
        return None
    if np.any(marker_row[np.arange(marker_row.size) != marker_col] != bg):
        return None

    liquid_height = 0
    seen_gap = False
    for rr in range(interior.shape[0] - 2, -1, -1):
        row = interior[rr]
        if not seen_gap and bool(np.all(row == marker_color)):
            liquid_height += 1
            continue
        seen_gap = True
        if np.any(row != bg):
            return None

    return {
        "top": int(top),
        "bottom": int(bottom),
        "left": int(left),
        "right": int(right),
        "width": int(width),
        "height": int(height),
        "marker_color": int(marker_color),
        "liquid_height": int(liquid_height),
    }


def _find_wall_color_and_containers(grid: np.ndarray, bg: int) -> tuple[int, list[dict[str, int]]]:
    colors = [int(color) for color in np.unique(grid) if int(color) != bg]
    best_color = None
    best_containers: list[dict[str, int]] = []
    best_pixels = -1
    for color in colors:
        containers: list[dict[str, int]] = []
        for comp in _components(grid == color):
            container = _extract_container(grid, comp, color, bg)
            if container is not None:
                containers.append(container)
        pixel_count = int(np.count_nonzero(grid == color))
        if len(containers) > len(best_containers) or (
            len(containers) == len(best_containers) and len(containers) > 0 and pixel_count > best_pixels
        ):
            best_color = int(color)
            best_containers = containers
            best_pixels = pixel_count

    assert best_color is not None and best_containers, "expected at least one valid container"
    return best_color, best_containers


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    corner_colors = [int(g[0, 0]), int(g[0, -1]), int(g[-1, 0]), int(g[-1, -1])]
    bg = int(Counter(corner_colors).most_common(1)[0][0])
    _wall_color, containers = _find_wall_color_and_containers(g, bg)

    target_heights: dict[int, int] = defaultdict(int)
    for container in containers:
        color = int(container["marker_color"])
        target_heights[color] = max(target_heights[color], int(container["liquid_height"]))

    out = g.copy()
    for container in containers:
        target = int(target_heights[int(container["marker_color"])])
        if target <= int(container["liquid_height"]):
            continue
        fill_top = int(container["bottom"]) - 1 - target
        fill_bottom = int(container["bottom"]) - 1
        out[fill_top:fill_bottom, int(container["left"]) + 1 : int(container["right"])] = int(
            container["marker_color"]
        )

    return out


def _sample_group_sizes(rng: np.random.Generator, color_count: int, container_count: int) -> tuple[list[int], int]:
    sizes = [1] * color_count
    primary = int(rng.integers(color_count))
    sizes[primary] += 1
    remaining = int(container_count - color_count - 1)
    while remaining > 0:
        choices = [idx for idx, size in enumerate(sizes) if size < MAX_GROUP_SIZE]
        if not choices:
            return _sample_group_sizes(rng, color_count, container_count)
        idx = int(rng.choice(choices))
        sizes[idx] += 1
        remaining -= 1
    return sizes, primary


def _assign_group_liquids(
    rng: np.random.Generator, capacities: list[int], require_change: bool
) -> list[int]:
    size = len(capacities)
    if size == 1:
        cap = int(capacities[0])
        return [int(rng.integers(cap + 1))]

    target = int(rng.integers(1, min(capacities) + 1))
    varied = require_change or bool(rng.integers(2))
    if not varied:
        return [target] * size

    max_count = int(rng.integers(1, size))
    at_target = set(int(v) for v in rng.choice(size, size=max_count, replace=False).tolist())
    liquids: list[int] = []
    for idx in range(size):
        if idx in at_target:
            liquids.append(int(target))
        else:
            liquids.append(int(rng.integers(target)))
    return liquids


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    color_count = int(rng.integers(2, len(LIQUID_COLORS) + 1))
    max_containers = min(MAX_CONTAINERS, color_count * MAX_GROUP_SIZE)
    min_containers = max(MIN_CONTAINERS, color_count + 1)
    if min_containers > max_containers:
        return None
    container_count = int(rng.integers(min_containers, max_containers + 1))

    colors = [int(v) for v in rng.choice(LIQUID_COLORS, size=color_count, replace=False).tolist()]
    group_sizes, primary_group = _sample_group_sizes(rng, color_count, container_count)

    containers: list[dict[str, int]] = []
    for group_idx, (color, group_size) in enumerate(zip(colors, group_sizes)):
        widths = [int(rng.choice(WIDTH_CHOICES)) for _ in range(group_size)]
        heights = [int(rng.integers(*HEIGHT_RANGE)) for _ in range(group_size)]
        capacities = [height - 2 for height in heights]
        liquids = _assign_group_liquids(rng, capacities, require_change=group_idx == primary_group)
        if group_idx == primary_group and len(set(liquids)) == 1:
            lower_choices = [idx for idx in range(group_size) if liquids[idx] > 0]
            if not lower_choices:
                return None
            idx = int(rng.choice(lower_choices))
            liquids[idx] -= 1
        for width, height, liquid_height in zip(widths, heights, liquids):
            containers.append(
                {
                    "color": int(color),
                    "width": int(width),
                    "height": int(height),
                    "liquid_height": int(liquid_height),
                }
            )

    rng.shuffle(containers)

    gaps = [int(rng.choice(GAP_CHOICES)) for _ in range(len(containers) - 1)]
    left_margin = int(rng.integers(*SIDE_MARGIN_RANGE))
    right_margin = int(rng.integers(*SIDE_MARGIN_RANGE))
    top_margin = int(rng.integers(*TOP_MARGIN_RANGE))
    grid_width = (
        left_margin
        + sum(int(container["width"]) for container in containers)
        + sum(gaps)
        + right_margin
    )
    grid_height = top_margin + max(int(container["height"]) for container in containers)

    grid = np.full((grid_height, grid_width), BG, dtype=int)
    cursor = left_margin
    for idx, container in enumerate(containers):
        width = int(container["width"])
        height = int(container["height"])
        color = int(container["color"])
        liquid_height = int(container["liquid_height"])

        left = int(cursor)
        right = int(left + width - 1)
        top = int(grid_height - height)
        bottom = int(grid_height - 1)

        grid[top : bottom + 1, left] = WALL
        grid[top : bottom + 1, right] = WALL
        grid[bottom, left : right + 1] = WALL

        marker_row = int(bottom - 1)
        marker_col = int(left + width // 2)
        grid[marker_row, marker_col] = color
        if liquid_height > 0:
            fill_top = int(marker_row - liquid_height)
            grid[fill_top:marker_row, left + 1 : right] = color

        cursor = right + 1
        if idx < len(gaps):
            cursor += int(gaps[idx])

    try:
        out = main(grid)
    except Exception:
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(_RNG)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
