from __future__ import annotations

import random

import numpy as np


BACKGROUND = 0
GRAY = 5
WALL = 7
SEED_COLORS = (1, 2, 3, 4, 6, 8, 9)

TEMPLATE_SHAPES: tuple[tuple[tuple[int, int], ...], ...] = (
    ((0, -1), (0, 0), (0, 1)),
    ((-1, 0), (0, 0), (1, 0)),
    ((-1, 0), (0, -1), (0, 0), (0, 1), (1, 0)),
    ((-1, 0), (0, -1), (0, 0), (1, 0), (1, 1)),
    ((-1, 0), (0, 1), (0, 0), (1, 0), (1, -1)),
    ((1, 0), (0, -1), (0, 0), (-1, 0), (-1, 1)),
    ((1, 0), (0, 1), (0, 0), (-1, 0), (-1, -1)),
    ((-1, -1), (-1, 0), (0, 0), (0, 1), (1, 0)),
    ((-1, 1), (-1, 0), (0, 0), (0, -1), (1, 0)),
)


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _find_components(g: np.ndarray, color: int, start_row: int = 0) -> list[list[tuple[int, int]]]:
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[list[tuple[int, int]]] = []

    for r in range(start_row, h):
        for c in range(w):
            if seen[r, c] or int(g[r, c]) != color:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(g[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            components.append(coords)

    return components


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _template_mapping(g: np.ndarray) -> tuple[dict[int, tuple[tuple[int, int], ...]], int]:
    marker_cells = [(c, int(g[0, c])) for c in range(g.shape[1]) if int(g[0, c]) not in {BACKGROUND, GRAY, WALL}]
    marker_colors = [color for _, color in marker_cells]
    assert marker_cells, "expected palette markers in the top row"
    assert len(marker_colors) == len(set(marker_colors)), "palette colors must be unique"

    components = _find_components(g, GRAY, start_row=1)
    assert components, "expected gray template components"

    mapping: dict[int, tuple[tuple[int, int], ...]] = {}
    palette_bottom = 0
    for coords in components:
        r0, c0, r1, c1 = _bbox(coords)
        palette_bottom = max(palette_bottom, r1)
        matches = [(col, color) for col, color in marker_cells if c0 <= col <= c1]
        assert len(matches) == 1, "each gray template must align to exactly one palette color"
        marker_col, color = matches[0]
        center_r = (r0 + r1) // 2
        center_c = (c0 + c1) // 2
        assert (r1 - r0) % 2 == 0 and (c1 - c0) % 2 == 0, "template bbox must have a center pixel"
        assert (center_r, center_c) in set(coords), "template center pixel must belong to the template"
        assert c0 <= marker_col <= c1, "palette marker must sit above its template"
        mapping[color] = tuple(sorted((r - center_r, c - center_c) for r, c in coords))

    assert set(mapping) == set(marker_colors), "every palette color must have one template"
    return mapping, palette_bottom


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow each colored seed into the gray template assigned to that color while skipping template cells blocked by orange walls.

    Input structure:
    - The background is black.
    - The top row contains distinct single-pixel palette markers, and gray 4-connected template components sit directly below them.
    - The remaining rows contain orange horizontal or vertical wall segments and scattered single-pixel colored seeds that reuse palette colors.

    Rule:
    - Build a color-to-template mapping by pairing each top-row marker with the gray template component underneath it.
    - For every seed, align the template bbox center to the seed location.
    - For each template offset, paint the corresponding output cell with the seed color only when the original input cell there is black.
    - Skip any offset that lands on an orange wall, another non-black cell, outside the grid, or inside the protected palette area; keep the palette unchanged.

    Key insight:
    - The top legend is not decorative: color selects a template, then the template is spatially re-centered on each seed, and walls block only the individual covered pixels instead of stopping the whole growth.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    mapping, palette_bottom = _template_mapping(g)
    out = g.copy()
    h, w = g.shape

    seed_found = False
    for r in range(palette_bottom + 1, h):
        for c in range(w):
            color = int(g[r, c])
            if color not in mapping:
                continue
            seed_found = True
            for dr, dc in mapping[color]:
                rr = r + dr
                cc = c + dc
                if not (0 <= rr < h and 0 <= cc < w):
                    continue
                if rr <= palette_bottom:
                    continue
                if int(g[rr, cc]) == BACKGROUND:
                    out[rr, cc] = color

    assert seed_found, "expected at least one seed below the palette"
    return out


# --------------- generator ---------------

GRID_HEIGHT_MIN = 14
GRID_HEIGHT_MAX = 24
GRID_WIDTH_MIN = 18
GRID_WIDTH_MAX = 30

PALETTE_CENTER_ROW = 2
PALETTE_BOTTOM = 3
FIELD_START_ROW = PALETTE_BOTTOM + 2

PALETTE_COLOR_COUNT_MIN = 2
PALETTE_COLOR_COUNT_MAX = 4

MANDATORY_SEEDS = 2
EXTRA_SEEDS_MAX = 3
MAX_WALLS_EXTRA = 2

WALL_LENGTH_MIN = 2
WALL_LENGTH_MAX = 7

GENERATION_RETRIES = 400


def _footprint(
    offsets: tuple[tuple[int, int], ...],
    center_r: int,
    center_c: int,
    height: int,
    width: int,
) -> tuple[set[tuple[int, int]], bool]:
    cells: set[tuple[int, int]] = set()
    clipped = False
    for dr, dc in offsets:
        r = center_r + dr
        c = center_c + dc
        if 0 <= r < height and 0 <= c < width:
            cells.add((r, c))
        else:
            clipped = True
    return cells, clipped


def _candidate_seed_positions(
    offsets: tuple[tuple[int, int], ...],
    height: int,
    width: int,
    reserved: set[tuple[int, int]],
    clip: bool | None,
) -> list[tuple[int, int, set[tuple[int, int]]]]:
    options: list[tuple[int, int, set[tuple[int, int]]]] = []
    for r in range(FIELD_START_ROW, height):
        for c in range(width):
            footprint, clipped = _footprint(offsets, r, c, height, width)
            if not footprint:
                continue
            if clip is True and not clipped:
                continue
            if clip is False and clipped:
                continue
            if any(rr <= PALETTE_BOTTOM for rr, _ in footprint):
                continue
            if len(footprint) < 2:
                continue
            if footprint & reserved:
                continue
            options.append((r, c, footprint))
    return options


def _marker_columns(width: int, count: int) -> list[int] | None:
    candidates = list(range(2, width - 2, 4))
    if len(candidates) < count:
        return None
    return sorted(random.sample(candidates, count))


def _all_footprints(seeds: list[dict]) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    for seed in seeds:
        cells |= seed["footprint"]
    return cells


def _wall_options_for_seed(
    height: int,
    width: int,
    seed: dict,
    occupied_by_other_seeds: set[tuple[int, int]],
    existing_walls: set[tuple[int, int]],
) -> list[set[tuple[int, int]]]:
    sr, sc = seed["pos"]
    allowed = seed["footprint"]
    target_cells = list(allowed - {(sr, sc)})
    random.shuffle(target_cells)
    options: list[set[tuple[int, int]]] = []

    for tr, tc in target_cells:
        for orientation in random.sample(["h", "v"], 2):
            limit = width if orientation == "h" else height
            for length in range(WALL_LENGTH_MIN, min(WALL_LENGTH_MAX, limit) + 1):
                for index in range(length):
                    if orientation == "h":
                        c0 = tc - index
                        c1 = c0 + length - 1
                        if c0 < 0 or c1 >= width:
                            continue
                        cells = {(tr, cc) for cc in range(c0, c1 + 1)}
                    else:
                        r0 = tr - index
                        r1 = r0 + length - 1
                        if r0 <= PALETTE_BOTTOM or r1 >= height:
                            continue
                        cells = {(rr, tc) for rr in range(r0, r1 + 1)}

                    if any(rr <= PALETTE_BOTTOM for rr, _ in cells):
                        continue
                    if (sr, sc) in cells:
                        continue
                    if cells & existing_walls:
                        continue
                    if any(cell in occupied_by_other_seeds for cell in cells):
                        continue
                    if not (cells & allowed):
                        continue
                    if not (cells - allowed):
                        continue
                    if not ((allowed - {(sr, sc)}) - cells):
                        continue
                    options.append(cells)

    return options


def _random_wall(
    height: int,
    width: int,
    forbidden: set[tuple[int, int]],
    existing_walls: set[tuple[int, int]],
) -> set[tuple[int, int]] | None:
    for _ in range(100):
        orientation = random.choice(["h", "v"])
        length = random.randint(WALL_LENGTH_MIN, min(WALL_LENGTH_MAX, width if orientation == "h" else height))
        if orientation == "h":
            r = random.randint(FIELD_START_ROW, height - 1)
            c0 = random.randint(0, width - length)
            cells = {(r, c0 + i) for i in range(length)}
        else:
            c = random.randint(0, width - 1)
            r0 = random.randint(FIELD_START_ROW, height - length)
            cells = {(r0 + i, c) for i in range(length)}

        if cells & forbidden or cells & existing_walls:
            continue
        return cells
    return None


def _build_specs(palette_colors: list[int]) -> list[dict]:
    if len(palette_colors) >= 2:
        blocked_color, free_color = random.sample(palette_colors, 2)
    else:
        blocked_color = free_color = palette_colors[0]

    specs = [
        {"color": blocked_color, "blocked": True, "clip": False},
        {"color": free_color, "blocked": False, "clip": False},
    ]

    if random.random() < 0.65:
        specs.append(
            {
                "color": random.choice(palette_colors),
                "blocked": random.choice([True, False]),
                "clip": True,
            }
        )

    for _ in range(random.randint(0, EXTRA_SEEDS_MAX)):
        specs.append(
            {
                "color": random.choice(palette_colors),
                "blocked": random.choice([True, False]),
                "clip": random.choice([False, True, None]),
            }
        )

    return specs


def _place_seeds(
    color_to_shape: dict[int, tuple[tuple[int, int], ...]],
    height: int,
    width: int,
) -> list[dict] | None:
    specs = _build_specs(list(color_to_shape))
    specs.sort(key=lambda spec: (spec["clip"] is not False, len(color_to_shape[spec["color"]]), spec["blocked"]), reverse=True)

    seeds: list[dict] = []
    reserved: set[tuple[int, int]] = set()

    for spec in specs:
        offsets = color_to_shape[spec["color"]]
        options = _candidate_seed_positions(offsets, height, width, reserved, spec["clip"])
        if not options:
            return None
        r, c, footprint = random.choice(options)
        seed = {
            "color": spec["color"],
            "blocked": spec["blocked"],
            "clip": spec["clip"],
            "pos": (r, c),
            "footprint": footprint,
        }
        seeds.append(seed)
        reserved |= footprint

    blocked_count = sum(seed["blocked"] for seed in seeds)
    free_count = len(seeds) - blocked_count
    if blocked_count == 0 or free_count == 0 or len(seeds) < MANDATORY_SEEDS:
        return None
    return seeds


def _place_walls(height: int, width: int, seeds: list[dict]) -> set[tuple[int, int]] | None:
    walls: set[tuple[int, int]] = set()
    all_footprints = _all_footprints(seeds)

    for seed in seeds:
        if not seed["blocked"]:
            continue
        others = all_footprints - seed["footprint"]
        options = _wall_options_for_seed(height, width, seed, others, walls)
        if not options:
            return None
        walls |= random.choice(options)

    forbidden = all_footprints
    for _ in range(random.randint(0, MAX_WALLS_EXTRA)):
        wall = _random_wall(height, width, forbidden, walls)
        if wall is None:
            break
        walls |= wall

    for seed in seeds:
        blocked_here = bool(seed["footprint"] & walls)
        if blocked_here != bool(seed["blocked"]):
            return None
        if len(seed["footprint"] - walls) < 2:
            return None

    return walls


def _render_input(
    height: int,
    width: int,
    palette_colors: list[int],
    palette_shapes: list[tuple[tuple[int, int], ...]],
    marker_cols: list[int],
    seeds: list[dict],
    walls: set[tuple[int, int]],
) -> np.ndarray:
    grid = np.zeros((height, width), dtype=int)

    for color, shape, col in zip(palette_colors, palette_shapes, marker_cols, strict=True):
        grid[0, col] = color
        for dr, dc in shape:
            grid[PALETTE_CENTER_ROW + dr, col + dc] = GRAY

    for r, c in walls:
        grid[r, c] = WALL

    for seed in seeds:
        r, c = seed["pos"]
        grid[r, c] = seed["color"]

    return grid


def _valid_case(inp: np.ndarray, seeds: list[dict]) -> bool:
    try:
        out = main(inp)
    except Exception:
        return False

    if not np.any(out != inp):
        return False

    for seed in seeds:
        sr, sc = seed["pos"]
        color = seed["color"]
        grown = 0
        for r, c in seed["footprint"]:
            if (r, c) == (sr, sc):
                continue
            if int(inp[r, c]) == WALL:
                continue
            if int(out[r, c]) == color:
                grown += 1
        if grown == 0:
            return False

    return True


def _gen_attempt() -> np.ndarray | None:
    palette_count = random.randint(PALETTE_COLOR_COUNT_MIN, PALETTE_COLOR_COUNT_MAX)
    height = random.randint(GRID_HEIGHT_MIN, GRID_HEIGHT_MAX)
    width = random.randint(max(GRID_WIDTH_MIN, 4 * palette_count + 2), GRID_WIDTH_MAX)

    palette_colors = random.sample(SEED_COLORS, palette_count)
    palette_shapes = random.sample(list(TEMPLATE_SHAPES), palette_count)
    marker_cols = _marker_columns(width, palette_count)
    if marker_cols is None:
        return None

    color_to_shape = {color: shape for color, shape in zip(palette_colors, palette_shapes, strict=True)}
    seeds = _place_seeds(color_to_shape, height, width)
    if seeds is None:
        return None

    walls = _place_walls(height, width, seeds)
    if walls is None:
        return None

    inp = _render_input(height, width, palette_colors, palette_shapes, marker_cols, seeds, walls)
    if _valid_case(inp, seeds):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid growth-template ARC input")
