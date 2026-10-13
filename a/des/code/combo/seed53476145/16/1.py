from __future__ import annotations

import random

import numpy as np


def _mode_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _find_separator_column(grid: np.ndarray, bg: int) -> tuple[int, int]:
    h, w = grid.shape
    candidates = []
    for c in range(1, w - 1):
        col = grid[:, c]
        color = int(col[0])
        if color == bg or not np.all(col == color):
            continue
        if np.any(grid[:, :c] != bg) and np.any(grid[:, c + 1 :] != bg):
            candidates.append((c, color))
    assert candidates, "expected a full-height separator column"
    return candidates[0]


def _parse_legend(grid: np.ndarray, sep_col: int, sep_color: int, bg: int) -> dict[int, np.ndarray]:
    legend = grid[:, :sep_col]
    h, w = legend.shape
    entries: dict[int, np.ndarray] = {}

    for r in range(h - 4):
        for c in range(w - 2):
            block = legend[r : r + 2, c : c + 2]
            color = int(block[0, 0])
            if color in {bg, sep_color} or not np.all(block == color):
                continue
            template_patch = legend[r + 2 : r + 5, c : c + 3]
            template = template_patch == sep_color
            if not template.any():
                continue
            patch = legend[r : r + 5, c : c + 3]
            if not set(map(int, np.unique(patch))).issubset({bg, sep_color, color}):
                continue
            if r > 0 and np.any(legend[r - 1, c : c + 2] == color):
                continue
            assert color not in entries, "legend colors must be unique"
            entries[color] = template.copy()

    assert entries, "expected at least one legend entry"
    return entries


def _rectangle_border_cells(r0: int, c0: int, r1: int, c1: int) -> set[tuple[int, int]]:
    border = set()
    for c in range(c0, c1 + 1):
        border.add((r0, c))
        border.add((r1, c))
    for r in range(r0, r1 + 1):
        border.add((r, c0))
        border.add((r, c1))
    return border


def _find_regions(
    grid: np.ndarray, sep_col: int, bg: int, sep_color: int
) -> list[tuple[tuple[int, int, int, int], int]]:
    scene = grid[:, sep_col + 1 :]
    offset = sep_col + 1
    regions: list[tuple[tuple[int, int, int, int], int]] = []

    for color in np.unique(scene):
        color = int(color)
        if color in {bg, sep_color}:
            continue
        for comp in _components(scene == color):
            r0, c0, r1, c1 = _bbox(comp)
            if r1 - r0 + 1 < 5 or c1 - c0 + 1 < 5:
                continue
            if set(comp) != _rectangle_border_cells(r0, c0, r1, c1):
                continue
            interior = scene[r0 + 1 : r1, c0 + 1 : c1]
            markers = np.argwhere(interior != bg)
            assert len(markers) == 1, "each region interior must contain exactly one marker"
            mr, mc = map(int, markers[0])
            marker_color = int(interior[mr, mc])
            assert marker_color not in {bg, color}, "marker must differ from background and wall"
            abs_bbox = (r0, c0 + offset, r1, c1 + offset)
            regions.append((abs_bbox, marker_color))

    assert regions, "expected at least one walled region"
    regions.sort()
    return regions


def main(grid: np.ndarray) -> np.ndarray:
    """
    Apply a color-to-template legend to marked rectangular regions.

    Input structure
    - Background is black.
    - The left legend column is separated from the scene by a full-height grey column.
    - Each legend entry is a 2x2 color block with a 3x3 grey template directly below it.
    - The scene contains blue rectangular walls; each enclosed interior has one colored marker pixel.

    Rule
    - Match each marker color to the legend entry with the same color block.
    - Center that entry's 3x3 template inside the region interior.
    - Repaint template foreground pixels with the marker color and erase the original marker.
    - Keep walls, separator, and legend unchanged.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a nonempty 2D grid"

    bg = _mode_color(g)
    sep_col, sep_color = _find_separator_column(g, bg)
    legend = _parse_legend(g, sep_col, sep_color, bg)
    regions = _find_regions(g, sep_col, bg, sep_color)

    out = g.copy()
    for (r0, c0, r1, c1), marker_color in regions:
        assert marker_color in legend, "every marker color must appear in the legend"
        template = legend[marker_color]
        inner_h = r1 - r0 - 1
        inner_w = c1 - c0 - 1
        assert inner_h >= 3 and inner_w >= 3, "region interior must fit a 3x3 template"
        place_r = r0 + 1 + (inner_h - 3) // 2
        place_c = c0 + 1 + (inner_w - 3) // 2
        out[r0 + 1 : r1, c0 + 1 : c1] = bg
        patch = out[place_r : place_r + 3, place_c : place_c + 3]
        patch[template] = marker_color

    return out


# --------------- generator ---------------

BACKGROUND = 0
WALL = 1
SEPARATOR = 5
COLOR_POOL = (2, 3, 4, 6, 7, 8, 9)

LEGEND_WIDTH = 3
ENTRY_HEIGHT = 5
LEGEND_GAP_CHOICES = (1, 2)

ENTRY_COUNT_RANGE = (2, 4)
REGION_COUNT_RANGE = (1, 4)
INTERIOR_LARGE_CHOICES = (
    (3, 5),
    (5, 3),
    (5, 5),
    (3, 7),
    (7, 3),
    (5, 7),
    (7, 5),
    (7, 7),
)
GRID_HEIGHT_MAX = 32
GRID_WIDTH_MAX = 34
PLACEMENT_TRIES = 200
GENERATION_RETRIES = 400

_BASE_TEMPLATES = (
    np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool),
    np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]], dtype=bool),
    np.array([[1, 0, 0], [1, 0, 0], [1, 1, 1]], dtype=bool),
    np.array([[1, 1, 1], [0, 1, 0], [0, 1, 0]], dtype=bool),
    np.array([[1, 1, 0], [0, 1, 0], [0, 1, 1]], dtype=bool),
    np.array([[1, 1, 0], [1, 0, 0], [1, 1, 0]], dtype=bool),
    np.array([[1, 0, 1], [1, 1, 1], [0, 0, 0]], dtype=bool),
)


def _template_variants() -> list[np.ndarray]:
    variants: dict[tuple[int, ...], np.ndarray] = {}
    for base in _BASE_TEMPLATES:
        for rot in range(4):
            rotated = np.rot90(base, rot)
            for flipped in (rotated, np.fliplr(rotated)):
                key = tuple(int(v) for v in flipped.ravel())
                variants.setdefault(key, flipped.copy())
    return list(variants.values())


TEMPLATE_CHOICES = _template_variants()


def _sample_entry_rows(rng: random.Random, count: int) -> tuple[list[int], int]:
    top_margin = rng.randint(0, 2)
    bottom_margin = rng.randint(0, 2)
    rows = []
    cursor = top_margin
    for idx in range(count):
        rows.append(cursor)
        cursor += ENTRY_HEIGHT
        if idx < count - 1:
            cursor += rng.choice(LEGEND_GAP_CHOICES)
    return rows, cursor + bottom_margin


def _sample_region_sizes(rng: random.Random, count: int) -> list[tuple[int, int]]:
    kinds = [rng.choice(("exact", "large")) for _ in range(count)]
    if count >= 2:
        kinds[0] = "exact"
        kinds[1] = "large"
        rng.shuffle(kinds)

    sizes = []
    for kind in kinds:
        if kind == "exact":
            sizes.append((3, 3))
        else:
            sizes.append(rng.choice(INTERIOR_LARGE_CHOICES))
    return sizes


def _place_regions(
    rng: random.Random, grid_h: int, grid_w: int, sizes: list[tuple[int, int]]
) -> list[tuple[int, int, int, int]] | None:
    occupied = np.zeros((grid_h, grid_w), dtype=bool)
    placements: list[tuple[int, int, int, int]] = []

    for inner_h, inner_w in sorted(sizes, key=lambda hw: hw[0] * hw[1], reverse=True):
        outer_h = inner_h + 2
        outer_w = inner_w + 2
        if grid_h < outer_h + 2 or grid_w < LEGEND_WIDTH + 3 + outer_w:
            return None

        options = []
        for _ in range(PLACEMENT_TRIES):
            r0 = rng.randint(1, grid_h - outer_h - 1)
            c0 = rng.randint(LEGEND_WIDTH + 2, grid_w - outer_w - 1)
            rr0 = max(0, r0 - 1)
            rr1 = min(grid_h, r0 + outer_h + 1)
            cc0 = max(0, c0 - 1)
            cc1 = min(grid_w, c0 + outer_w + 1)
            if not occupied[rr0:rr1, cc0:cc1].any():
                options.append((r0, c0, inner_h, inner_w))
                if len(options) >= 12:
                    break
        if not options:
            return None

        choice = rng.choice(options)
        r0, c0, inner_h, inner_w = choice
        outer_h = inner_h + 2
        outer_w = inner_w + 2
        occupied[max(0, r0 - 1) : min(grid_h, r0 + outer_h + 1), max(0, c0 - 1) : min(grid_w, c0 + outer_w + 1)] = True
        placements.append(choice)

    rng.shuffle(placements)
    return placements


def _draw_frame(grid: np.ndarray, r0: int, c0: int, inner_h: int, inner_w: int, marker_color: int, rng: random.Random) -> None:
    outer_h = inner_h + 2
    outer_w = inner_w + 2
    grid[r0, c0 : c0 + outer_w] = WALL
    grid[r0 + outer_h - 1, c0 : c0 + outer_w] = WALL
    grid[r0 : r0 + outer_h, c0] = WALL
    grid[r0 : r0 + outer_h, c0 + outer_w - 1] = WALL

    mr = rng.randint(r0 + 1, r0 + inner_h)
    mc = rng.randint(c0 + 1, c0 + inner_w)
    grid[mr, mc] = marker_color


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    entry_count = rng.randint(*ENTRY_COUNT_RANGE)
    region_count = rng.randint(*REGION_COUNT_RANGE)

    colors = rng.sample(COLOR_POOL, entry_count)
    templates = rng.sample(TEMPLATE_CHOICES, entry_count)
    entry_rows, legend_height = _sample_entry_rows(rng, entry_count)
    sizes = _sample_region_sizes(rng, region_count)

    max_outer_h = max(inner_h + 2 for inner_h, _ in sizes)
    max_outer_w = max(inner_w + 2 for _, inner_w in sizes)
    min_h = max(legend_height, max_outer_h + 2, 12)
    min_w = max(LEGEND_WIDTH + 2 + max_outer_w + 1, 18)
    if min_h > GRID_HEIGHT_MAX or min_w > GRID_WIDTH_MAX:
        return None

    grid_h = rng.randint(min_h, GRID_HEIGHT_MAX)
    grid_w = rng.randint(min_w, GRID_WIDTH_MAX)
    placements = _place_regions(rng, grid_h, grid_w, sizes)
    if placements is None:
        return None

    grid = np.full((grid_h, grid_w), BACKGROUND, dtype=int)
    grid[:, LEGEND_WIDTH] = SEPARATOR

    for row, color, template in zip(entry_rows, colors, templates):
        grid[row : row + 2, 0:2] = color
        patch = grid[row + 2 : row + 5, 0:3]
        patch[template] = SEPARATOR

    marker_colors = list(colors[: min(entry_count, region_count)])
    while len(marker_colors) < region_count:
        marker_colors.append(rng.choice(colors))
    rng.shuffle(marker_colors)

    for (r0, c0, inner_h, inner_w), marker_color in zip(placements, marker_colors):
        _draw_frame(grid, r0, c0, inner_h, inner_w, marker_color, rng)

    try:
        out = main(grid)
    except AssertionError:
        return None
    if out.shape != grid.shape or np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
