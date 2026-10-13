from __future__ import annotations

import math

import numpy as np


Coord = tuple[int, int]
BBox = tuple[int, int, int, int]


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _bbox_coords(coords: np.ndarray) -> BBox:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _bbox_mask(mask: np.ndarray) -> BBox | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    return _bbox_coords(coords)


def _neighbors4(r: int, c: int) -> tuple[Coord, Coord, Coord, Coord]:
    return (r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)


def _components_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[Coord] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            components.append(np.array(coords, dtype=int))
    return components


def _find_legend_component(grid: np.ndarray, components: list[np.ndarray]) -> tuple[np.ndarray, list[int]]:
    candidates: list[tuple[np.ndarray, list[int]]] = []
    for coords in components:
        r0, c0, r1, c1 = _bbox_coords(coords)
        height = r1 - r0 + 1
        width = c1 - c0 + 1
        colors = [int(grid[r, c]) for r, c in coords]
        unique_colors = list(dict.fromkeys(colors))
        if height == 1 and len(coords) == width and width >= 3 and len(unique_colors) == width:
            legend = [int(x) for x in grid[r0, c0 : c1 + 1]]
            candidates.append((coords, legend))
    assert len(candidates) == 1, "expected exactly one legend bar"
    return candidates[0]


def _solve_cavity(
    region: np.ndarray,
    legend: list[int],
) -> tuple[np.ndarray, int, int, np.ndarray, int, int]:
    colors, counts = np.unique(region, return_counts=True)
    assert len(colors) == 2, "each cavity must contain exactly empty + seed colors"
    empty = int(colors[np.argmax(counts)])
    seed_color = int(colors[np.argmin(counts)])
    assert seed_color in legend, "seed color must appear in the legend"

    seed_mask = region == seed_color
    seed_bbox = _bbox_mask(seed_mask)
    assert seed_bbox is not None, "seed must be non-empty"
    sr0, sc0, sr1, sc1 = seed_bbox
    tile_mask = seed_mask[sr0 : sr1 + 1, sc0 : sc1 + 1]
    assert tile_mask.shape[0] >= 1 and tile_mask.shape[1] >= 1
    assert _bbox_mask(tile_mask) == (0, 0, tile_mask.shape[0] - 1, tile_mask.shape[1] - 1)

    tile_h, tile_w = tile_mask.shape
    cavity_h, cavity_w = region.shape
    assert cavity_h % tile_h == 0 and cavity_w % tile_w == 0
    assert sr0 % tile_h == 0 and sc0 % tile_w == 0

    seed_block_r = sr0 // tile_h
    seed_block_c = sc0 // tile_w
    p = legend.index(seed_color)
    out = np.full_like(region, empty)

    for block_r in range(cavity_h // tile_h):
        for block_c in range(cavity_w // tile_w):
            color = legend[(p + block_r - seed_block_r + block_c - seed_block_c) % len(legend)]
            r0 = block_r * tile_h
            c0 = block_c * tile_w
            patch = out[r0 : r0 + tile_h, c0 : c0 + tile_w]
            patch[tile_mask] = color

    return out, empty, seed_color, tile_mask, seed_block_r, seed_block_c


def main(grid: np.ndarray) -> np.ndarray:
    """Tile each cavity with the seed mask, recoloring each tile copy by walking around the legend cycle.

    Input structure:
    - A dominant wall color fills most of the grid and encloses one or more rectangular cavities.
    - Each cavity is a solid rectangle of one empty color containing exactly one monochrome seed pattern.
    - Elsewhere there is exactly one horizontal 1×k legend bar with k>=3 distinct colors, and every seed color appears in that legend.

    Rule:
    - For each cavity, take the seed pattern's bounding box as the tile unit and keep the seed's current tile position as the phase origin.
    - Repeat that mask over the whole cavity. If the seed color is legend index p, then the tile j steps right and i steps down from the seed tile uses legend[(p+i+j) mod k].
    - Cells outside the mask stay the cavity's empty color; walls and the legend bar remain unchanged.

    Key insight:
    - The seed locks two phases at once: its bounding box fixes the tiling lattice, and its current color fixes the starting point on the legend cycle.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    wall = _get_bg(g)
    components = _components_mask(g != wall)
    legend_coords, legend = _find_legend_component(g, components)
    legend_set = {tuple(map(int, rc)) for rc in legend_coords}

    out = g.copy()
    for coords in components:
        coord_set = {tuple(map(int, rc)) for rc in coords}
        if coord_set == legend_set:
            continue
        r0, c0, r1, c1 = _bbox_coords(coords)
        region = g[r0 : r1 + 1, c0 : c1 + 1]
        assert len(coords) == region.size, "cavity component must fill its rectangle"
        filled, _empty, _seed, _mask, _sbr, _sbc = _solve_cavity(region, legend)
        out[r0 : r1 + 1, c0 : c1 + 1] = filled

    return out


MAX_ATTEMPTS = 500
GRID_MIN = 18
GRID_MAX = 30
GAP = 2
BORDER = 1
ALL_COLORS = tuple(range(10))
WALL_COLORS = tuple(range(1, 10))


def _rect_separated(a: BBox, b: BBox, gap: int) -> bool:
    return (
        a[2] + gap < b[0]
        or b[2] + gap < a[0]
        or a[3] + gap < b[1]
        or b[3] + gap < a[1]
    )


def _proper_divisors(n: int) -> list[int]:
    return [d for d in range(1, n) if n % d == 0]


def _has_smaller_period(mask: np.ndarray) -> bool:
    h, w = mask.shape
    row_divs = _proper_divisors(h) + [h]
    col_divs = _proper_divisors(w) + [w]
    for hh in row_divs:
        for ww in col_divs:
            if hh == h and ww == w:
                continue
            if h % hh != 0 or w % ww != 0:
                continue
            base = mask[:hh, :ww]
            tiled = np.tile(base, (h // hh, w // ww))
            if np.array_equal(mask, tiled):
                return True
    return False


def _random_tile_mask(rng: np.random.Generator, tile_h: int, tile_w: int) -> np.ndarray | None:
    area = tile_h * tile_w
    min_fill = max(3, math.ceil(area * 0.35))
    max_fill = min(area - 1, math.floor(area * 0.72))
    if min_fill > max_fill:
        return None

    for _ in range(200):
        fill = int(rng.integers(min_fill, max_fill + 1))
        choice = rng.choice(area, size=fill, replace=False)
        mask = np.zeros((tile_h, tile_w), dtype=bool)
        mask.flat[choice] = True
        if _bbox_mask(mask) != (0, 0, tile_h - 1, tile_w - 1):
            continue
        if not np.all(mask.any(axis=1)) or not np.all(mask.any(axis=0)):
            continue
        if _has_smaller_period(mask):
            continue
        return mask
    return None


def _sample_cavity(
    rng: np.random.Generator,
    *,
    require_interior_seed: bool,
) -> dict | None:
    for _ in range(200):
        tile_h = int(rng.choice([2, 3]))
        tile_w = int(rng.choice([2, 3, 4]))

        rep_h_options = [r for r in range(2, 5) if 6 <= tile_h * r <= 12]
        rep_w_options = [r for r in range(2, 5) if 6 <= tile_w * r <= 12]
        if require_interior_seed:
            rep_h_options = [r for r in rep_h_options if r >= 3]
            rep_w_options = [r for r in rep_w_options if r >= 3]
        if not rep_h_options or not rep_w_options:
            continue

        reps_h = int(rng.choice(rep_h_options))
        reps_w = int(rng.choice(rep_w_options))
        mask = _random_tile_mask(rng, tile_h, tile_w)
        if mask is None:
            continue

        if require_interior_seed:
            seed_block_r = int(rng.integers(1, reps_h - 1))
            seed_block_c = int(rng.integers(1, reps_w - 1))
        else:
            seed_block_r = int(rng.integers(0, reps_h))
            seed_block_c = int(rng.integers(0, reps_w))

        return {
            "tile_h": tile_h,
            "tile_w": tile_w,
            "reps_h": reps_h,
            "reps_w": reps_w,
            "height": tile_h * reps_h,
            "width": tile_w * reps_w,
            "mask": mask,
            "seed_block": (seed_block_r, seed_block_c),
        }
    return None


def _choose_canvas_sizes(non_wall_area: int, rect_sizes: list[tuple[int, int]]) -> list[tuple[int, int]]:
    min_h = max(GRID_MIN, max(h for h, _ in rect_sizes) + 2 * BORDER)
    min_w = max(GRID_MIN, max(w for _, w in rect_sizes) + 2 * BORDER)
    sizes: list[tuple[int, int]] = []
    for rows in range(min_h, GRID_MAX + 1):
        for cols in range(min_w, GRID_MAX + 1):
            if rows * cols <= 2 * non_wall_area:
                continue
            sizes.append((rows, cols))
    return sizes


def _place_rectangles(
    rng: np.random.Generator,
    rows: int,
    cols: int,
    sizes: list[tuple[int, int]],
) -> list[BBox] | None:
    order = sorted(range(len(sizes)), key=lambda idx: sizes[idx][0] * sizes[idx][1], reverse=True)
    placed: dict[int, BBox] = {}
    for idx in order:
        h, w = sizes[idx]
        options: list[BBox] = []
        for r0 in range(BORDER, rows - h - BORDER + 1):
            for c0 in range(BORDER, cols - w - BORDER + 1):
                rect = (r0, c0, r0 + h - 1, c0 + w - 1)
                if all(_rect_separated(rect, other, GAP) for other in placed.values()):
                    options.append(rect)
        if not options:
            return None
        placed[idx] = options[int(rng.integers(0, len(options)))]
    return [placed[idx] for idx in range(len(sizes))]


def _build_input_and_output(design: dict) -> tuple[np.ndarray, np.ndarray]:
    rows = design["rows"]
    cols = design["cols"]
    wall = design["wall"]
    empty = design["empty"]
    legend = design["legend"]
    legend_rect = design["legend_rect"]

    inp = np.full((rows, cols), wall, dtype=int)
    out = inp.copy()

    lr0, lc0, _lr1, lc1 = legend_rect
    inp[lr0, lc0 : lc1 + 1] = legend
    out[lr0, lc0 : lc1 + 1] = legend

    for cavity in design["cavities"]:
        r0, c0, r1, c1 = cavity["rect"]
        tile_h = cavity["tile_h"]
        tile_w = cavity["tile_w"]
        reps_h = cavity["reps_h"]
        reps_w = cavity["reps_w"]
        mask = cavity["mask"]
        seed_color = cavity["seed_color"]
        seed_block_r, seed_block_c = cavity["seed_block"]
        seed_top = r0 + seed_block_r * tile_h
        seed_left = c0 + seed_block_c * tile_w

        inp[r0 : r1 + 1, c0 : c1 + 1] = empty
        out[r0 : r1 + 1, c0 : c1 + 1] = empty
        seed_patch = inp[seed_top : seed_top + tile_h, seed_left : seed_left + tile_w]
        seed_patch[mask] = seed_color

        p = legend.index(seed_color)
        for block_r in range(reps_h):
            for block_c in range(reps_w):
                color = legend[(p + block_r - seed_block_r + block_c - seed_block_c) % len(legend)]
                rr = r0 + block_r * tile_h
                cc = c0 + block_c * tile_w
                out_patch = out[rr : rr + tile_h, cc : cc + tile_w]
                out_patch[mask] = color

    return inp, out


def _validate_design(design: dict, inp: np.ndarray, out: np.ndarray) -> bool:
    if np.array_equal(inp, out):
        return False

    wall = design["wall"]
    if _get_bg(inp) != wall:
        return False

    try:
        solved = main(inp)
    except Exception:
        return False
    if not np.array_equal(solved, out):
        return False

    cavities = design["cavities"]
    legend = design["legend"]
    empty = design["empty"]

    if not any(
        cavity["seed_block"][0] > 0
        and cavity["seed_block"][0] < cavity["reps_h"] - 1
        and cavity["seed_block"][1] > 0
        and cavity["seed_block"][1] < cavity["reps_w"] - 1
        for cavity in cavities
    ):
        return False

    if not any(legend.index(cavity["seed_color"]) != 0 for cavity in cavities):
        return False

    used_output_colors: set[int] = set()
    cavity_with_three_colors = False
    for cavity in cavities:
        r0, c0, r1, c1 = cavity["rect"]
        region_in = inp[r0 : r1 + 1, c0 : c1 + 1]
        region_out = out[r0 : r1 + 1, c0 : c1 + 1]
        colors_in = set(map(int, np.unique(region_in)))
        if colors_in != {empty, cavity["seed_color"]}:
            return False
        if set(map(int, np.unique(region_out))) <= {empty, cavity["seed_color"]}:
            return False

        seed_mask = region_in == cavity["seed_color"]
        seed_bbox = _bbox_mask(seed_mask)
        if seed_bbox is None:
            return False
        sr0, sc0, sr1, sc1 = seed_bbox
        if (sr1 - sr0 + 1, sc1 - sc0 + 1) != (cavity["tile_h"], cavity["tile_w"]):
            return False
        if not np.array_equal(seed_mask[sr0 : sr1 + 1, sc0 : sc1 + 1], cavity["mask"]):
            return False

        mask_colors: set[int] = set()
        for block_r in range(cavity["reps_h"]):
            for block_c in range(cavity["reps_w"]):
                rr = block_r * cavity["tile_h"]
                cc = block_c * cavity["tile_w"]
                tile = region_out[rr : rr + cavity["tile_h"], cc : cc + cavity["tile_w"]]
                mask_colors.update(map(int, tile[cavity["mask"]]))
        used_output_colors.update(mask_colors)
        cavity_with_three_colors |= len(mask_colors) >= 3

    return len(used_output_colors) >= 3 and cavity_with_three_colors


def _generate_design(rng: np.random.Generator) -> dict | None:
    cavity_count = int(rng.choice([1, 2, 2, 3]))
    cavities: list[dict] = []
    for idx in range(cavity_count):
        cavity = _sample_cavity(rng, require_interior_seed=(idx == 0))
        if cavity is None:
            return None
        cavities.append(cavity)

    wall = int(rng.choice(WALL_COLORS))
    remaining = [c for c in ALL_COLORS if c != wall]
    empty = int(rng.choice(remaining))
    remaining = [c for c in remaining if c != empty]

    min_k = max(3, cavity_count + 1)
    max_k = min(6, len(remaining))
    if min_k > max_k:
        return None
    k = int(rng.integers(min_k, max_k + 1))
    legend = [int(x) for x in rng.choice(remaining, size=k, replace=False)]

    anchor_index = int(rng.integers(1, k))
    seed_indices = [anchor_index]
    other_indices = [i for i in range(k) if i != anchor_index]
    rng.shuffle(other_indices)
    seed_indices.extend(other_indices[: cavity_count - 1])
    if len(seed_indices) > 2:
        shuffled_tail = seed_indices[1:]
        rng.shuffle(shuffled_tail)
        seed_indices[1:] = shuffled_tail
    for cavity, idx in zip(cavities, seed_indices):
        cavity["seed_color"] = legend[idx]

    sizes = [(cavity["height"], cavity["width"]) for cavity in cavities]
    legend_size = (1, k)
    rect_sizes = sizes + [legend_size]
    non_wall_area = sum(h * w for h, w in sizes) + k

    canvas_sizes = _choose_canvas_sizes(non_wall_area, rect_sizes)
    if not canvas_sizes:
        return None

    for _ in range(120):
        rows, cols = canvas_sizes[int(rng.integers(0, len(canvas_sizes)))]
        placed = _place_rectangles(rng, rows, cols, rect_sizes)
        if placed is None:
            continue
        for cavity, rect in zip(cavities, placed[:-1]):
            cavity["rect"] = rect
        design = {
            "rows": rows,
            "cols": cols,
            "wall": wall,
            "empty": empty,
            "legend": legend,
            "legend_rect": placed[-1],
            "cavities": cavities,
        }
        inp, out = _build_input_and_output(design)
        if _validate_design(design, inp, out):
            return design
    return None


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_ATTEMPTS):
        design = _generate_design(rng)
        if design is None:
            continue
        inp, _out = _build_input_and_output(design)
        return inp
    raise RuntimeError("failed to generate a valid input")
