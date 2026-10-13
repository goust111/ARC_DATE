from __future__ import annotations

import random
from collections import Counter

import numpy as np


SEPARATOR_COLOR = 6
ARC_MAX_SIDE = 30
GENERATION_ATTEMPTS = 400
COMMON_COUNT_RANGE = (2, 3)
LAYOUT_OPTIONS = ((2, 2), (2, 3), (3, 2))
BASE_SHAPES = (
    ((0, 0), (0, 1), (0, 2)),
    ((0, 0), (1, 0), (1, 1)),
    ((0, 0), (0, 1), (1, 0), (1, 1)),
    ((0, 0), (0, 1), (0, 2), (0, 3)),
    ((0, 0), (1, 0), (2, 0), (2, 1)),
    ((0, 0), (0, 1), (0, 2), (1, 1)),
    ((0, 1), (0, 2), (1, 0), (1, 1)),
    ((0, 0), (0, 1), (1, 0), (1, 1), (2, 0)),
    ((0, 1), (1, 0), (1, 1), (1, 2), (2, 1)),
    ((0, 0), (0, 2), (1, 0), (1, 1), (1, 2)),
    ((0, 0), (1, 0), (2, 0), (2, 1), (2, 2)),
    ((0, 1), (0, 2), (1, 0), (1, 1), (2, 0)),
    ((0, 0), (0, 1), (0, 2), (1, 0), (1, 2)),
    ((0, 0), (0, 1), (0, 2), (1, 2), (2, 2)),
    ((0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)),
    ((0, 0), (0, 1), (0, 2), (1, 1), (2, 1), (2, 2)),
    ((0, 0), (1, 0), (2, 0), (2, 1), (2, 2), (1, 2)),
    ((0, 0), (1, 0), (1, 1), (1, 2), (2, 2), (0, 2)),
)


def _bands(size: int, separators: list[int]) -> list[tuple[int, int]]:
    bands: list[tuple[int, int]] = []
    start = 0
    for cut in separators:
        if start < cut:
            bands.append((start, cut - 1))
        start = cut + 1
    if start < size:
        bands.append((start, size - 1))
    return bands


def _components(mask: np.ndarray) -> list[np.ndarray]:
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
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.asarray(coords, dtype=int))
    return comps


def _normalize_coords(coords: np.ndarray) -> tuple[tuple[int, int], ...]:
    r0 = int(coords[:, 0].min())
    c0 = int(coords[:, 1].min())
    rel = coords - np.array([r0, c0], dtype=int)
    return tuple(sorted((int(r), int(c)) for r, c in rel))


def _component_patch(cell: np.ndarray, coords: np.ndarray) -> np.ndarray:
    r0 = int(coords[:, 0].min())
    c0 = int(coords[:, 1].min())
    r1 = int(coords[:, 0].max())
    c1 = int(coords[:, 1].max())
    return cell[r0 : r1 + 1, c0 : c1 + 1].copy()


def _separator_color(grid: np.ndarray) -> int:
    h, w = grid.shape
    row_colors = {int(grid[r, 0]) for r in range(h) if np.all(grid[r, :] == grid[r, 0])}
    col_colors = {int(grid[0, c]) for c in range(w) if np.all(grid[:, c] == grid[0, c])}
    candidates = row_colors & col_colors
    assert len(candidates) == 1, "expected exactly one full-row/full-column separator color"
    return next(iter(candidates))


def _odd_patch_from_region(region: np.ndarray) -> tuple[np.ndarray, int]:
    colors = [int(c) for c in np.unique(region)]
    assert len(colors) == 2, "each region must contain exactly one foreground color on one background"
    counts = Counter(map(int, region.ravel()))
    bg = int(max(colors, key=lambda color: counts[color]))
    fg = next(color for color in colors if color != bg)

    comps = _components(region == fg)
    assert len(comps) >= 3, "each region must contain several sprites"

    by_shape = Counter(_normalize_coords(comp) for comp in comps)
    odd_shapes = [shape for shape, n in by_shape.items() if n == 1]
    assert len(odd_shapes) == 1, "each region must have exactly one odd sprite shape"
    odd_shape = odd_shapes[0]

    odd_comp = next(comp for comp in comps if _normalize_coords(comp) == odd_shape)
    return _component_patch(region, odd_comp), bg


def _pack_patches(patches: list[list[np.ndarray]], bg: int) -> np.ndarray:
    n_rows = len(patches)
    n_cols = len(patches[0])
    row_heights = [max(int(patches[r][c].shape[0]) for c in range(n_cols)) for r in range(n_rows)]
    col_widths = [max(int(patches[r][c].shape[1]) for r in range(n_rows)) for c in range(n_cols)]

    out_h = sum(row_heights) + (n_rows - 1)
    out_w = sum(col_widths) + (n_cols - 1)
    out = np.full((out_h, out_w), int(bg), dtype=int)

    r0 = 0
    for r in range(n_rows):
        c0 = 0
        for c in range(n_cols):
            patch = patches[r][c]
            ph, pw = patch.shape
            out[r0 : r0 + ph, c0 : c0 + pw] = patch
            c0 += col_widths[c] + 1
        r0 += row_heights[r] + 1
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Find the unique odd sprite in each separator cell and repack those crops in cell order.

    Input structure:
    - Full horizontal and vertical separator lines of one color divide the grid into equal-sized rectangular regions.
    - Every region has one shared background color and several 4-connected monochrome sprites of one other color.
    - Inside each region, one sprite shape occurs exactly once while the remaining sprites all share the same normalized shape.

    Rule:
    - Detect the separator rows and columns, split the grid into regions, and examine each region independently.
    - Compare region sprites by their translation-normalized pixel pattern to find the unique odd shape, then crop it to its tight bounding box.
    - Preserve reading order of the source regions and place one odd-sprite crop per output slot, separating neighboring slots by one background row or column.
    - When slot crops have different sizes, each output row uses the tallest crop in that source row and each output column uses the widest crop in that source column, with every crop anchored at the upper-left of its slot.

    Key insight:
    - The puzzle requires two linked steps: first solve a shape-based odd-one-out task inside each partition, then treat the partitions themselves as an ordered grid that defines how the extracted crops are repacked.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    sep = _separator_color(g)
    sep_rows = [r for r in range(g.shape[0]) if np.all(g[r, :] == sep)]
    sep_cols = [c for c in range(g.shape[1]) if np.all(g[:, c] == sep)]
    assert sep_rows and sep_cols, "expected both horizontal and vertical separators"

    row_bands = _bands(g.shape[0], sep_rows)
    col_bands = _bands(g.shape[1], sep_cols)
    assert len(row_bands) >= 2 and len(col_bands) >= 2, "expected a multi-region lattice"

    band_heights = {r1 - r0 + 1 for r0, r1 in row_bands}
    band_widths = {c1 - c0 + 1 for c0, c1 in col_bands}
    assert len(band_heights) == 1 and len(band_widths) == 1, "regions must be equal-sized"

    patches: list[list[np.ndarray]] = []
    bg = None
    for r0, r1 in row_bands:
        row: list[np.ndarray] = []
        for c0, c1 in col_bands:
            region = g[r0 : r1 + 1, c0 : c1 + 1]
            patch, region_bg = _odd_patch_from_region(region)
            if bg is None:
                bg = int(region_bg)
            else:
                assert int(bg) == int(region_bg), "all regions must share the same background"
            row.append(patch)
        patches.append(row)

    assert bg is not None
    return _pack_patches(patches, int(bg))


# --------------- generator ---------------


def _normalize_cells(cells: tuple[tuple[int, int], ...] | set[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    pts = list(cells)
    r0 = min(r for r, _c in pts)
    c0 = min(c for _r, c in pts)
    return tuple(sorted((int(r - r0), int(c - c0)) for r, c in pts))


def _rotate_cells(cells: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    max_r = max(r for r, _c in cells)
    return _normalize_cells({(c, max_r - r) for r, c in cells})


def _mirror_cells(cells: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    max_c = max(c for _r, c in cells)
    return _normalize_cells({(r, max_c - c) for r, c in cells})


def _shape_bank() -> list[tuple[tuple[int, int], ...]]:
    bank: set[tuple[tuple[int, int], ...]] = set()
    for shape in BASE_SHAPES:
        current = _normalize_cells(shape)
        variants = {current}
        for _ in range(4):
            current = _rotate_cells(current)
            variants.add(current)
            variants.add(_mirror_cells(current))
        bank.update(variants)
    return sorted(bank, key=lambda shape: (len(shape), shape))


SHAPES = _shape_bank()


def _shape_dims(shape: tuple[tuple[int, int], ...]) -> tuple[int, int]:
    return max(r for r, _c in shape) + 1, max(c for _r, c in shape) + 1


def _render_shape_patch(shape: tuple[tuple[int, int], ...], color: int, bg: int) -> np.ndarray:
    h, w = _shape_dims(shape)
    patch = np.full((h, w), int(bg), dtype=int)
    for r, c in shape:
        patch[int(r), int(c)] = int(color)
    return patch


def _fits(shape: tuple[tuple[int, int], ...], region_h: int, region_w: int) -> bool:
    h, w = _shape_dims(shape)
    return h <= region_h and w <= region_w


def _can_place_shape(
    occupied: np.ndarray,
    shape: tuple[tuple[int, int], ...],
    top: int,
    left: int,
) -> bool:
    h, w = occupied.shape
    for dr, dc in shape:
        rr = top + int(dr)
        cc = left + int(dc)
        r0 = max(0, rr - 1)
        r1 = min(h, rr + 2)
        c0 = max(0, cc - 1)
        c1 = min(w, cc + 2)
        if occupied[r0:r1, c0:c1].any():
            return False
    return True


def _place_region_shapes(
    region_h: int,
    region_w: int,
    shapes: list[tuple[tuple[int, int], ...]],
) -> list[tuple[int, int]] | None:
    if not 3 <= len(shapes) <= 4:
        return None

    occupied = np.zeros((region_h, region_w), dtype=bool)
    slots = list(range(4))
    random.shuffle(slots)
    slots = slots[: len(shapes)]
    placements: list[tuple[int, int]] = []

    for shape, slot in zip(shapes, slots):
        sh, sw = _shape_dims(shape)
        if sh > region_h // 2 or sw > region_w // 2:
            return None
        top = 0 if slot in (0, 1) else region_h - sh
        left = 0 if slot in (0, 2) else region_w - sw
        if not _can_place_shape(occupied, shape, top, left):
            return None
        for dr, dc in shape:
            occupied[top + int(dr), left + int(dc)] = True
        placements.append((top, left))
    return placements


def _build_region(
    region_h: int,
    region_w: int,
    bg: int,
    color: int,
    used_odd_shapes: set[tuple[tuple[int, int], ...]],
) -> tuple[np.ndarray, np.ndarray, tuple[tuple[int, int], ...]] | None:
    fitting = [
        shape
        for shape in SHAPES
        if _fits(shape, region_h, region_w) and max(_shape_dims(shape)) <= 3
    ]
    if len(fitting) < 2:
        return None

    same_area_first = random.choice([False, True])
    for same_area in (same_area_first, not same_area_first):
        choices: list[tuple[tuple[tuple[int, int], ...], tuple[tuple[int, int], ...], int]] = []
        for common in fitting:
            for odd in fitting:
                if odd == common or odd in used_odd_shapes:
                    continue
                if same_area and len(common) != len(odd):
                    continue
                for common_count in range(COMMON_COUNT_RANGE[0], COMMON_COUNT_RANGE[1] + 1):
                    total_area = common_count * len(common) + len(odd)
                    if total_area >= (region_h * region_w + 1) // 2:
                        continue
                    choices.append((common, odd, common_count))
        random.shuffle(choices)

        for common, odd, common_count in choices[:160]:
            shapes = [common] * common_count + [odd]
            random.shuffle(shapes)
            placements = _place_region_shapes(region_h, region_w, shapes)
            if placements is None:
                continue

            region = np.full((region_h, region_w), int(bg), dtype=int)
            for shape, (top, left) in zip(shapes, placements):
                for dr, dc in shape:
                    region[top + int(dr), left + int(dc)] = int(color)

            try:
                odd_patch, region_bg = _odd_patch_from_region(region)
            except AssertionError:
                continue
            if int(region_bg) != int(bg):
                continue
            if _normalize_coords(np.argwhere(odd_patch == color)) != odd:
                continue
            return region, odd_patch, odd
    return None


def _assemble_input(
    regions: list[list[np.ndarray]],
    sep: int,
) -> np.ndarray:
    n_rows = len(regions)
    n_cols = len(regions[0])
    region_h = int(regions[0][0].shape[0])
    region_w = int(regions[0][0].shape[1])
    grid_h = n_rows * region_h + (n_rows - 1)
    grid_w = n_cols * region_w + (n_cols - 1)
    grid = np.full((grid_h, grid_w), int(sep), dtype=int)

    for r in range(n_rows):
        for c in range(n_cols):
            r0 = r * (region_h + 1)
            c0 = c * (region_w + 1)
            grid[r0 : r0 + region_h, c0 : c0 + region_w] = regions[r][c]
    return grid


def _gen_attempt() -> tuple[np.ndarray, np.ndarray] | None:
    n_rows, n_cols = random.choice(LAYOUT_OPTIONS)
    region_h_max = min(9, (ARC_MAX_SIDE - (n_rows - 1)) // n_rows)
    region_w_max = min(9, (ARC_MAX_SIDE - (n_cols - 1)) // n_cols)
    if region_h_max < 7 or region_w_max < 7:
        return None

    region_h = random.randint(7, region_h_max)
    region_w = random.randint(7, region_w_max)

    bg_choices = [color for color in range(10) if color != SEPARATOR_COLOR]
    bg = random.choice(bg_choices)
    region_colors = random.sample([color for color in range(10) if color not in {bg, SEPARATOR_COLOR}], n_rows * n_cols)

    regions: list[list[np.ndarray]] = []
    odd_patches: list[list[np.ndarray]] = []
    used_odd_shapes: set[tuple[tuple[int, int], ...]] = set()
    color_iter = iter(region_colors)
    for _r in range(n_rows):
        region_row: list[np.ndarray] = []
        patch_row: list[np.ndarray] = []
        for _c in range(n_cols):
            color = next(color_iter)
            built = _build_region(region_h, region_w, bg, color, used_odd_shapes)
            if built is None:
                return None
            region, odd_patch, odd_shape = built
            used_odd_shapes.add(odd_shape)
            region_row.append(region)
            patch_row.append(odd_patch)
        regions.append(region_row)
        odd_patches.append(patch_row)

    inp = _assemble_input(regions, SEPARATOR_COLOR)
    out = _pack_patches(odd_patches, bg)

    if max(inp.shape) > ARC_MAX_SIDE or max(out.shape) > ARC_MAX_SIDE:
        return None
    if not np.array_equal(main(inp), out):
        return None
    return inp, out


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        built = _gen_attempt()
        if built is not None:
            return built[0]
    raise RuntimeError("failed to generate a valid odd-sprite partition puzzle")
