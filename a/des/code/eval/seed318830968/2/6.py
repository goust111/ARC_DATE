from __future__ import annotations

import random

import numpy as np


DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))

BG_COLOR = 0
CONTAINER_COLOR = 5
SEED_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)

GRID_H_RANGE = (16, 24)
GRID_W_RANGE = (18, 30)
CONTAINER_COUNT_CHOICES = (2, 3, 3, 4)
INTERIOR_WIDTH_RANGE = (2, 5)
CONTAINER_HEIGHT_RANGE = (3, 6)
PLACEMENT_TRIES = 300
GEN_RETRIES = 500


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    height, width = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(height):
        for c in range(width):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((int(rr), int(cc)))
                for dr, dc in DIR4:
                    nr = int(rr + dr)
                    nc = int(cc + dc)
                    if 0 <= nr < height and 0 <= nc < width and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _parse_container(
    grid: np.ndarray,
    coords: list[tuple[int, int]],
    container_color: int,
    bg: int,
) -> dict[str, int] | None:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    r0 = min(rows)
    r1 = max(rows)
    c0 = min(cols)
    c1 = max(cols)

    interior_width = c1 - c0 - 1
    interior_height = r1 - r0
    if not (INTERIOR_WIDTH_RANGE[0] <= interior_width <= INTERIOR_WIDTH_RANGE[1]):
        return None
    if interior_height < 1:
        return None

    expected = {(r, c0) for r in range(r0, r1 + 1)}
    expected.update((r, c1) for r in range(r0, r1 + 1))
    expected.update((r1, c) for c in range(c0, c1 + 1))
    if set(coords) != expected:
        return None

    interior = grid[r0:r1, c0 + 1 : c1]
    if interior.size == 0:
        return None

    seed_mask = (interior != bg) & (interior != container_color)
    seed_positions = np.argwhere(seed_mask)
    if len(seed_positions) == 0:
        return None

    bottom_local_row = interior.shape[0] - 1
    if not np.all(seed_positions[:, 0] == bottom_local_row):
        return None

    seed_colors = np.unique(interior[seed_mask])
    if seed_colors.size != 1:
        return None

    seed_cols = sorted(int(c) for c in seed_positions[:, 1])
    if seed_cols != list(range(seed_cols[0], seed_cols[0] + len(seed_cols))):
        return None

    seed_count = len(seed_cols)
    if not (1 <= seed_count <= 5 and seed_count <= interior_width):
        return None

    return {
        "top": int(r0),
        "left": int(c0),
        "right": int(c1),
        "seed_color": int(seed_colors[0]),
        "seed_count": int(seed_count),
    }


def _find_containers(grid: np.ndarray) -> list[dict[str, int]]:
    bg = _most_common_color(grid)
    containers: list[dict[str, int]] = []
    for color in sorted(int(v) for v in np.unique(grid) if int(v) != bg):
        for comp in _components(grid == color):
            parsed = _parse_container(grid, comp, color, bg)
            if parsed is not None:
                containers.append(parsed)
    containers.sort(key=lambda item: (item["top"], item["left"]))
    return containers


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow a plant above each U-shaped container.

    The plant width equals the container opening width, the plant height equals
    the number of seed pixels on the inner bottom row, and the plant color is
    the seed color.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    for container in _find_containers(g):
        top = int(container["top"])
        left = int(container["left"])
        right = int(container["right"])
        seed_color = int(container["seed_color"])
        seed_count = int(container["seed_count"])
        plant_top = max(0, top - seed_count)
        out[plant_top:top, left + 1 : right] = seed_color

    return out


def _build_expected(grid: np.ndarray, placed: list[dict[str, int]]) -> np.ndarray:
    out = grid.copy()
    for spec in placed:
        top = int(spec["container_top"])
        left = int(spec["left"])
        width = int(spec["width"])
        seed_count = int(spec["seed_count"])
        seed_color = int(spec["seed_color"])
        out[top - seed_count : top, left + 1 : left + 1 + width] = seed_color
    return out


def _make_specs() -> list[dict[str, int]]:
    count = random.choice(CONTAINER_COUNT_CHOICES)
    colors = random.sample(SEED_COLORS, count)
    specs: list[dict[str, int]] = []

    if random.random() < 0.5:
        shared_seed_count = random.randint(1, 4)
        width_choices = [w for w in range(shared_seed_count, INTERIOR_WIDTH_RANGE[1] + 1)]
        width_a, width_b = random.sample(width_choices, 2)
        initial = [(width_a, shared_seed_count), (width_b, shared_seed_count)]
    else:
        shared_width = random.randint(*INTERIOR_WIDTH_RANGE)
        seed_a, seed_b = random.sample(range(1, shared_width + 1), 2)
        initial = [(shared_width, seed_a), (shared_width, seed_b)]

    for index, (width, seed_count) in enumerate(initial):
        specs.append(
            {
                "index": index,
                "width": int(width),
                "seed_count": int(seed_count),
                "seed_color": int(colors[index]),
                "box_height": random.randint(*CONTAINER_HEIGHT_RANGE),
                "seed_offset": random.randint(0, int(width - seed_count)),
            }
        )

    for index in range(2, count):
        width = random.randint(*INTERIOR_WIDTH_RANGE)
        seed_count = random.randint(1, width)
        specs.append(
            {
                "index": index,
                "width": int(width),
                "seed_count": int(seed_count),
                "seed_color": int(colors[index]),
                "box_height": random.randint(*CONTAINER_HEIGHT_RANGE),
                "seed_offset": random.randint(0, int(width - seed_count)),
            }
        )

    return specs


def _place_specs(height: int, width: int, specs: list[dict[str, int]]) -> list[dict[str, int]] | None:
    occupied = np.zeros((height, width), dtype=bool)
    placed: list[dict[str, int]] = []

    for spec in sorted(specs, key=lambda item: (item["seed_count"] + item["box_height"]) * (item["width"] + 2), reverse=True):
        box_width = int(spec["width"] + 2)
        reserved_height = int(spec["seed_count"] + spec["box_height"])
        if height - reserved_height < 3 or width - box_width < 3:
            return None

        found = None
        for _ in range(PLACEMENT_TRIES):
            plant_top = random.randint(1, height - reserved_height - 1)
            left = random.randint(1, width - box_width - 1)
            r0 = max(0, plant_top - 1)
            r1 = min(height - 1, plant_top + reserved_height)
            c0 = max(0, left - 1)
            c1 = min(width - 1, left + box_width)
            if occupied[r0 : r1 + 1, c0 : c1 + 1].any():
                continue
            found = (plant_top, left, r0, r1, c0, c1)
            break

        if found is None:
            return None

        plant_top, left, r0, r1, c0, c1 = found
        occupied[r0 : r1 + 1, c0 : c1 + 1] = True
        placed.append(
            {
                **spec,
                "plant_top": int(plant_top),
                "container_top": int(plant_top + spec["seed_count"]),
                "left": int(left),
            }
        )

    placed.sort(key=lambda item: item["index"])
    return placed


def _render_input(height: int, width: int, placed: list[dict[str, int]]) -> np.ndarray:
    grid = np.full((height, width), BG_COLOR, dtype=int)
    for spec in placed:
        top = int(spec["container_top"])
        left = int(spec["left"])
        inner_width = int(spec["width"])
        box_height = int(spec["box_height"])
        seed_count = int(spec["seed_count"])
        seed_color = int(spec["seed_color"])
        seed_offset = int(spec["seed_offset"])
        right = left + inner_width + 1
        bottom = top + box_height - 1

        grid[top : bottom + 1, left] = CONTAINER_COLOR
        grid[top : bottom + 1, right] = CONTAINER_COLOR
        grid[bottom, left : right + 1] = CONTAINER_COLOR

        seed_row = bottom - 1
        seed_start = left + 1 + seed_offset
        grid[seed_row, seed_start : seed_start + seed_count] = seed_color

    return grid


def _is_informative(placed: list[dict[str, int]]) -> bool:
    widths = [int(spec["width"]) for spec in placed]
    counts = [int(spec["seed_count"]) for spec in placed]
    if not any(count < width for width, count in zip(widths, counts)):
        return False

    same_width_diff_count = any(
        len({int(spec["seed_count"]) for spec in placed if int(spec["width"]) == width}) >= 2
        for width in set(widths)
    )
    same_count_diff_width = any(
        len({int(spec["width"]) for spec in placed if int(spec["seed_count"]) == count}) >= 2
        for count in set(counts)
    )
    return same_width_diff_count or same_count_diff_width


def _gen_attempt() -> np.ndarray | None:
    specs = _make_specs()
    height = random.randint(*GRID_H_RANGE)
    width = random.randint(*GRID_W_RANGE)
    placed = _place_specs(height, width, specs)
    if placed is None or not _is_informative(placed):
        return None

    inp = _render_input(height, width, placed)
    expected = _build_expected(inp, placed)

    try:
        actual = main(inp)
    except Exception:
        return None

    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
