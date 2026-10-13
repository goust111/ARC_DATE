import random

import numpy as np


GEN_RETRY_LIMIT = 400
ARC_GRID_MAX = 30
BLACK = 0


def _neighbors4(r: int, c: int, h: int, w: int):
    if r > 0:
        yield r - 1, c
    if r + 1 < h:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < w:
        yield r, c + 1


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
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
                for nr, nc in _neighbors4(cr, cc, h, w):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _find_palette_component(g: np.ndarray, comps: list[list[tuple[int, int]]]) -> list[tuple[int, int]]:
    best = None
    best_key = None
    for comp in comps:
        r0, c0, r1, c1 = _bbox(comp)
        height = r1 - r0 + 1
        width = c1 - c0 + 1
        if width != 1 or len(comp) != height:
            continue
        rows = sorted(r for r, _ in comp)
        colors = [int(g[r, c0]) for r in rows]
        if len(set(colors)) != len(colors):
            continue
        key = (c0, len(comp), r0)
        if best_key is None or key < best_key:
            best = comp
            best_key = key
    assert best is not None, "expected a one-column spectrum palette"
    return best


def _find_main_region_component(
    g: np.ndarray,
    comps: list[list[tuple[int, int]]],
    palette_comp: list[tuple[int, int]],
) -> list[tuple[int, int]]:
    palette_set = set(palette_comp)
    palette_bbox = _bbox(palette_comp)
    best = None
    best_area = -1
    for comp in comps:
        if set(comp) == palette_set:
            continue
        r0, c0, r1, c1 = _bbox(comp)
        area = len(comp)
        height = r1 - r0 + 1
        width = c1 - c0 + 1
        if area != height * width:
            continue
        if height < 2 or width < 2:
            continue
        if c0 <= palette_bbox[3]:
            continue
        if area > best_area:
            best = comp
            best_area = area
    assert best is not None, "expected a filled rectangular main region"
    return best


def _priority_map(g: np.ndarray, palette_comp: list[tuple[int, int]]) -> dict[int, int]:
    _, c0, _, _ = _bbox(palette_comp)
    order = [int(g[r, c0]) for r, _ in sorted(palette_comp)]
    return {color: idx for idx, color in enumerate(order)}


def _transform_region(region: np.ndarray, priority: dict[int, int]) -> np.ndarray:
    h, w = region.shape
    out = region.copy()
    for r in range(h):
        for c in range(w):
            color = int(region[r, c])
            assert color in priority, "region colors must come from the palette"
            best_color = color
            best_rank = priority[color]
            for nr, nc in _neighbors4(r, c, h, w):
                ncolor = int(region[nr, nc])
                assert ncolor in priority, "region colors must come from the palette"
                nrank = priority[ncolor]
                if nrank < best_rank:
                    best_color = ncolor
                    best_rank = nrank
            out[r, c] = best_color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use the left-hand spectrum order so higher-priority colors erode one pixel into lower-priority neighbors.

    Input structure:
    - Background is black.
    - One narrow non-black component on the left is a one-column spectrum bar whose colors, read from top to bottom, define the priority order.
    - Another non-black component on the right is a solid rectangular region fully tiled by touching color blocks that use colors from the spectrum.

    Rule:
    - Read the spectrum bar from top to bottom: earlier colors have higher priority.
    - In the rectangular main region, inspect each pixel together with its four orthogonal neighbors in the original input.
    - If one or more neighboring pixels have higher-priority colors, recolor the pixel with the highest-priority such neighbor; otherwise keep its original color.
    - Apply only this single-layer update to the main region and leave the spectrum bar unchanged.

    Key insight:
    - The color ranking is encoded spatially by the vertical spectrum bar, and the output is a one-step priority max-filter over the main region under that learned order.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    comps = _components(g != BLACK)
    assert len(comps) >= 2, "expected a palette component and a main region"

    palette_comp = _find_palette_component(g, comps)
    region_comp = _find_main_region_component(g, comps, palette_comp)
    priority = _priority_map(g, palette_comp)

    r0, c0, r1, c1 = _bbox(region_comp)
    out = g.copy()
    out[r0 : r1 + 1, c0 : c1 + 1] = _transform_region(g[r0 : r1 + 1, c0 : c1 + 1], priority)
    return out


def _sample_partition(total: int, parts: int, rng: random.Random, force_exact_one: int | None = None) -> list[int]:
    assert parts >= 1 and total >= parts
    sizes = [1] * parts
    remaining = total - parts
    choices = [idx for idx in range(parts) if idx != force_exact_one]
    if not choices:
        choices = [force_exact_one]
    for _ in range(remaining):
        sizes[rng.choice(choices)] += 1
    rng.shuffle(sizes)
    if force_exact_one is not None:
        one_index = sizes.index(1)
        sizes[one_index], sizes[force_exact_one] = sizes[force_exact_one], sizes[one_index]
    return sizes


def _expand_cells(cell_colors: np.ndarray, row_heights: list[int], col_widths: list[int]) -> np.ndarray:
    rect_h = sum(row_heights)
    rect_w = sum(col_widths)
    region = np.zeros((rect_h, rect_w), dtype=int)
    rr = 0
    for br, rh in enumerate(row_heights):
        cc = 0
        for bc, cw in enumerate(col_widths):
            region[rr : rr + rh, cc : cc + cw] = int(cell_colors[br, bc])
            cc += cw
        rr += rh
    return region


def _random_cell_colors(
    block_rows: int,
    block_cols: int,
    palette_colors: list[int],
    rng: random.Random,
) -> np.ndarray | None:
    min_unique = min(3, len(palette_colors))
    for _ in range(200):
        cells = np.array(
            [[rng.choice(palette_colors) for _ in range(block_cols)] for _ in range(block_rows)],
            dtype=int,
        )
        if len(set(cells.flatten().tolist())) < min_unique:
            continue
        if not (
            np.any(cells[1:, :] != cells[:-1, :])
            or np.any(cells[:, 1:] != cells[:, :-1])
        ):
            continue
        return cells
    return None


def _pick_conflict_colors(palette_colors: list[int], rng: random.Random) -> tuple[int, int, int]:
    low_idx = rng.randrange(2, len(palette_colors))
    hi_a_idx, hi_b_idx = rng.sample(range(low_idx), 2)
    return palette_colors[hi_a_idx], palette_colors[hi_b_idx], palette_colors[low_idx]


def _build_cell_layout(
    palette_colors: list[int],
    rect_h: int,
    rect_w: int,
    mode: int,
    rng: random.Random,
) -> tuple[np.ndarray, list[int], list[int]] | None:
    block_rows = rng.randint(3, min(6, rect_h))
    block_cols = rng.randint(3, min(6, rect_w))
    thin_row = None
    thin_col = None
    if mode == 1:
        thin_row = rng.randrange(1, block_rows - 1)
    if mode == 2:
        thin_col = rng.randrange(1, block_cols - 1)

    row_heights = _sample_partition(rect_h, block_rows, rng, force_exact_one=thin_row)
    col_widths = _sample_partition(rect_w, block_cols, rng, force_exact_one=thin_col)

    cells = _random_cell_colors(block_rows, block_cols, palette_colors, rng)
    if cells is None:
        return None

    if mode == 1:
        top_color, bottom_color, low_color = _pick_conflict_colors(palette_colors, rng)
        cells[thin_row - 1, :] = top_color
        cells[thin_row, :] = low_color
        cells[thin_row + 1, :] = bottom_color
    elif mode == 2:
        left_color, right_color, low_color = _pick_conflict_colors(palette_colors, rng)
        cells[:, thin_col - 1] = left_color
        cells[:, thin_col] = low_color
        cells[:, thin_col + 1] = right_color

    return cells, row_heights, col_widths


def _branch_stats(region: np.ndarray, priority: dict[int, int]) -> tuple[bool, bool]:
    h, w = region.shape
    any_change = False
    any_multi = False
    for r in range(h):
        for c in range(w):
            color = int(region[r, c])
            higher_neighbors = set()
            for nr, nc in _neighbors4(r, c, h, w):
                ncolor = int(region[nr, nc])
                if priority[ncolor] < priority[color]:
                    higher_neighbors.add(ncolor)
            if higher_neighbors:
                any_change = True
                if len(higher_neighbors) >= 2:
                    any_multi = True
    return any_change, any_multi


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    palette_len = rng.randint(3, 6)
    palette_colors = rng.sample(list(range(1, 10)), palette_len)
    priority = {color: idx for idx, color in enumerate(palette_colors)}
    mode = rng.randrange(3)

    rect_h = rng.randint(8, 18)
    rect_w = rng.randint(8, 18)
    layout = _build_cell_layout(palette_colors, rect_h, rect_w, mode, rng)
    if layout is None:
        return None
    cells, row_heights, col_widths = layout
    region = _expand_cells(cells, row_heights, col_widths)

    any_change, any_multi = _branch_stats(region, priority)
    if not any_change:
        return None
    if mode in (1, 2) and not any_multi:
        return None

    transformed = _transform_region(region, priority)
    if np.array_equal(region, transformed):
        return None

    left_margin = rng.randint(0, 2)
    palette_top = rng.randint(0, 3)
    main_top = rng.randint(0, 3)
    gap = rng.randint(2, 4)
    right_margin = rng.randint(0, 3)
    bottom_margin = rng.randint(0, 3)

    palette_col = left_margin
    main_left = palette_col + 1 + gap
    grid_h = max(palette_top + palette_len, main_top + rect_h) + bottom_margin
    grid_w = main_left + rect_w + right_margin
    if grid_h > ARC_GRID_MAX or grid_w > ARC_GRID_MAX:
        return None

    grid = np.zeros((grid_h, grid_w), dtype=int)
    for idx, color in enumerate(palette_colors):
        grid[palette_top + idx, palette_col] = color
    grid[main_top : main_top + rect_h, main_left : main_left + rect_w] = region

    comps = _components(grid != BLACK)
    if len(comps) != 2:
        return None
    palette_comp = _find_palette_component(grid, comps)
    region_comp = _find_main_region_component(grid, comps, palette_comp)
    rr0, rc0, rr1, rc1 = _bbox(region_comp)
    if (rr1 - rr0 + 1, rc1 - rc0 + 1) != region.shape:
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRY_LIMIT):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input grid")
