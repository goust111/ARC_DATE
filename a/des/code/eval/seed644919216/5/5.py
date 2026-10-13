import random
from typing import Iterable

import numpy as np


GRID_H_RANGE = (18, 28)
GRID_W_RANGE = (22, 32)
REGION_COUNT_CHOICES = (1, 2, 3)
LEGEND_LEN_BY_REGION_COUNT = {
    1: (3, 4),
    2: (3, 6),
    3: (3, 6),
}
REGION_COLOR_COUNT_RANGE = (2, 4)
REGION_H_RANGE = (6, 11)
REGION_W_RANGE = (6, 12)
COMMON_CORE_SIZE_RANGE = (1, 3)
LEGEND_GAP_CHOICES = (1, 2)
PLACEMENT_MARGIN = 1
PLACEMENT_RETRIES = 200
REGION_RETRIES = 250
GEN_RETRIES = 300
MIN_VISIBLE_PIXELS = 3


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                if cr > 0 and mask[cr - 1, cc] and not seen[cr - 1, cc]:
                    seen[cr - 1, cc] = True
                    stack.append((cr - 1, cc))
                if cr + 1 < h and mask[cr + 1, cc] and not seen[cr + 1, cc]:
                    seen[cr + 1, cc] = True
                    stack.append((cr + 1, cc))
                if cc > 0 and mask[cr, cc - 1] and not seen[cr, cc - 1]:
                    seen[cr, cc - 1] = True
                    stack.append((cr, cc - 1))
                if cc + 1 < w and mask[cr, cc + 1] and not seen[cr, cc + 1]:
                    seen[cr, cc + 1] = True
                    stack.append((cr, cc + 1))
            comps.append(np.asarray(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.asarray(coords, dtype=int)
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _render_rectangles(
    shape: tuple[int, int],
    rects: dict[int, tuple[int, int, int, int]],
    paint_order: Iterable[int],
    bg: int,
) -> np.ndarray:
    grid = np.full(shape, bg, dtype=int)
    for color in paint_order:
        r0, c0, r1, c1 = rects[color]
        grid[r0 : r1 + 1, c0 : c1 + 1] = color
    return grid


def _find_legend(grid: np.ndarray, bg: int) -> tuple[list[int], np.ndarray]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    candidates: list[tuple[int, int, int, np.ndarray]] = []
    for comp in _components(g != bg):
        r0, c0, r1, c1 = _bbox(comp)
        if c1 != w - 1:
            continue
        if r1 - r0 + 1 != 2 or c1 - c0 + 1 != 2:
            continue
        if comp.shape[0] != 4:
            continue
        color = int(g[comp[0, 0], comp[0, 1]])
        if np.all(g[comp[:, 0], comp[:, 1]] == color):
            candidates.append((r0, c0, color, comp))

    assert candidates, "legend blocks not found on the right edge"
    c0_values, c0_counts = np.unique([c0 for _r0, c0, _color, _comp in candidates], return_counts=True)
    legend_c0 = int(c0_values[np.argmax(c0_counts)])
    blocks = [(r0, color, comp) for r0, c0, color, comp in candidates if c0 == legend_c0]
    blocks.sort(key=lambda item: item[0])

    legend_mask = np.zeros((h, w), dtype=bool)
    legend: list[int] = []
    prev_r1 = -10
    for r0, color, comp in blocks:
        assert color not in legend, "legend colors must be distinct"
        assert r0 > prev_r1, "legend blocks must be vertically ordered without overlap"
        legend.append(color)
        legend_mask[comp[:, 0], comp[:, 1]] = True
        prev_r1 = r0 + 1

    assert len(legend) >= 2, "legend must contain at least two depth colors"
    return legend, legend_mask


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = _most_common_color(g)
    legend, legend_mask = _find_legend(g, bg)
    depth_rank = {color: idx for idx, color in enumerate(legend)}

    out = g.copy()
    region_mask = (g != bg) & ~legend_mask
    for comp in _components(region_mask):
        comp_mask = np.zeros_like(region_mask, dtype=bool)
        comp_mask[comp[:, 0], comp[:, 1]] = True
        colors = [int(color) for color in np.unique(g[comp_mask])]
        assert colors, "each region must contain at least one color"
        assert set(colors).issubset(depth_rank), "region colors must be present in the legend"

        out[comp_mask] = bg
        for color in sorted(colors, key=lambda value: depth_rank[value], reverse=True):
            color_coords = np.argwhere(comp_mask & (g == color))
            assert color_coords.size > 0, "each region color must have visible pixels"
            r0, c0, r1, c1 = _bbox(color_coords)
            out[r0 : r1 + 1, c0 : c1 + 1] = color

    return out


def _random_rectangles(
    colors: list[int],
    height: int,
    width: int,
    rng: random.Random,
) -> dict[int, tuple[int, int, int, int]]:
    core_h = rng.randint(COMMON_CORE_SIZE_RANGE[0], min(COMMON_CORE_SIZE_RANGE[1], height - 2))
    core_w = rng.randint(COMMON_CORE_SIZE_RANGE[0], min(COMMON_CORE_SIZE_RANGE[1], width - 2))
    core_r0 = rng.randint(1, height - core_h - 1)
    core_c0 = rng.randint(1, width - core_w - 1)
    core_r1 = core_r0 + core_h - 1
    core_c1 = core_c0 + core_w - 1

    rects: dict[int, tuple[int, int, int, int]] = {}
    used: set[tuple[int, int, int, int]] = set()
    for color in colors:
        for _ in range(60):
            r0 = rng.randint(0, core_r0)
            c0 = rng.randint(0, core_c0)
            r1 = rng.randint(core_r1, height - 1)
            c1 = rng.randint(core_c1, width - 1)
            if r1 - r0 < 1 or c1 - c0 < 1:
                continue
            rect = (r0, c0, r1, c1)
            if rect not in used:
                rects[color] = rect
                used.add(rect)
                break
        else:
            raise RuntimeError("failed to sample unique rectangles")
    return rects


def _build_region_patch(
    colors: list[int],
    depth_rank: dict[int, int],
    bg: int,
    rng: random.Random,
    require_change: bool,
) -> tuple[np.ndarray, bool] | None:
    for _ in range(REGION_RETRIES):
        height = rng.randint(*REGION_H_RANGE)
        width = rng.randint(*REGION_W_RANGE)
        rects = _random_rectangles(colors, height, width, rng)

        input_paint_order = colors[:]
        rng.shuffle(input_paint_order)
        patch = _render_rectangles((height, width), rects, input_paint_order, bg)
        if len(_components(patch != bg)) != 1:
            continue

        ok = True
        for color in colors:
            visible = np.argwhere(patch == color)
            if visible.shape[0] < MIN_VISIBLE_PIXELS:
                ok = False
                break
            if _bbox(visible) != rects[color]:
                ok = False
                break
        if not ok:
            continue

        output_paint_order = sorted(colors, key=lambda value: depth_rank[value], reverse=True)
        out_patch = _render_rectangles((height, width), rects, output_paint_order, bg)
        changed = not np.array_equal(patch, out_patch)
        if require_change and not changed:
            continue
        return patch, changed

    return None


def _choose_region_color_sets(
    legend: list[int], region_count: int, rng: random.Random
) -> list[list[int]] | None:
    max_region_colors = min(REGION_COLOR_COUNT_RANGE[1], len(legend))
    for _ in range(80):
        target_sizes = [rng.randint(REGION_COLOR_COUNT_RANGE[0], max_region_colors) for _ in range(region_count)]
        if sum(target_sizes) < len(legend):
            continue

        region_sets = [set() for _ in range(region_count)]
        shuffled = legend[:]
        rng.shuffle(shuffled)
        for color in shuffled:
            candidates = [idx for idx in range(region_count) if len(region_sets[idx]) < target_sizes[idx]]
            if not candidates:
                break
            region_sets[rng.choice(candidates)].add(color)
        else:
            for idx in range(region_count):
                while len(region_sets[idx]) < target_sizes[idx]:
                    region_sets[idx].add(rng.choice(legend))
            if all(REGION_COLOR_COUNT_RANGE[0] <= len(s) <= REGION_COLOR_COUNT_RANGE[1] for s in region_sets):
                result = [list(s) for s in region_sets]
                for colors in result:
                    rng.shuffle(colors)
                return result
    return None


def _legend_total_height(block_count: int, gaps: list[int]) -> int:
    return 2 * block_count + sum(gaps)


def _place_patch(
    grid: np.ndarray,
    occupancy: np.ndarray,
    patch: np.ndarray,
    max_col: int,
    rng: random.Random,
) -> bool:
    h, w = grid.shape
    ph, pw = patch.shape
    if ph + 2 * PLACEMENT_MARGIN > h or pw + 2 * PLACEMENT_MARGIN > max_col + 1:
        return False

    row_lo = PLACEMENT_MARGIN
    row_hi = h - ph - PLACEMENT_MARGIN
    col_lo = PLACEMENT_MARGIN
    col_hi = max_col - pw - PLACEMENT_MARGIN + 1
    if row_hi < row_lo or col_hi < col_lo:
        return False

    for _ in range(PLACEMENT_RETRIES):
        r0 = rng.randint(row_lo, row_hi)
        c0 = rng.randint(col_lo, col_hi)
        rs = slice(max(0, r0 - 1), min(h, r0 + ph + 1))
        cs = slice(max(0, c0 - 1), min(grid.shape[1], c0 + pw + 1))
        if occupancy[rs, cs].any():
            continue
        target = grid[r0 : r0 + ph, c0 : c0 + pw]
        target[:] = patch
        occupancy[r0 : r0 + ph, c0 : c0 + pw] = True
        return True
    return False


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    region_count = rng.choice(REGION_COUNT_CHOICES)
    legend_min, legend_max = LEGEND_LEN_BY_REGION_COUNT[region_count]
    legend_len = rng.randint(legend_min, legend_max)

    bg = rng.randint(0, 9)
    palette = [color for color in range(10) if color != bg]
    rng.shuffle(palette)
    legend = palette[:legend_len]
    depth_rank = {color: idx for idx, color in enumerate(legend)}

    region_color_sets = _choose_region_color_sets(legend, region_count, rng)
    if region_color_sets is None:
        return None

    max_gap = max(LEGEND_GAP_CHOICES)
    legend_height_cap = _legend_total_height(legend_len, [max_gap] * (legend_len - 1))
    h_min = max(GRID_H_RANGE[0], legend_height_cap + 2)
    h_max = GRID_H_RANGE[1]
    if h_min > h_max:
        return None

    grid_h = rng.randint(h_min, h_max)
    grid_w = rng.randint(*GRID_W_RANGE)
    if grid_w < 8:
        return None

    grid = np.full((grid_h, grid_w), bg, dtype=int)

    gaps = [rng.choice(LEGEND_GAP_CHOICES) for _ in range(legend_len - 1)]
    total_legend_height = _legend_total_height(legend_len, gaps)
    legend_r0 = rng.randint(0, grid_h - total_legend_height)
    cursor = legend_r0
    for idx, color in enumerate(legend):
        grid[cursor : cursor + 2, grid_w - 2 : grid_w] = color
        cursor += 2
        if idx < len(gaps):
            cursor += gaps[idx]

    usable_max_col = grid_w - 4
    if usable_max_col < 5:
        return None

    patches: list[np.ndarray] = []
    changed_regions = 0
    must_change_index = rng.randrange(region_count)
    for idx, colors in enumerate(region_color_sets):
        built = _build_region_patch(
            colors=colors,
            depth_rank=depth_rank,
            bg=bg,
            rng=rng,
            require_change=idx == must_change_index,
        )
        if built is None:
            return None
        patch, changed = built
        if changed:
            changed_regions += 1
        patches.append(patch)

    if changed_regions == 0:
        return None

    occupancy = np.zeros_like(grid, dtype=bool)
    occupancy[:, grid_w - 3 :] = True

    for patch in sorted(patches, key=lambda item: item.shape[0] * item.shape[1], reverse=True):
        if not _place_patch(grid, occupancy, patch, usable_max_col, rng):
            return None

    try:
        out = main(grid)
    except AssertionError:
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
