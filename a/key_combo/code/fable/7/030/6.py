from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import random

import numpy as np


def _load_arc_utils():
    path = Path("/home/lixinyu/snap/new/solvers/arc_utils.py")
    spec = spec_from_file_location("arc_utils_shared_for_fable_7_030_6", path)
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_ARC = _load_arc_utils()
Color = _ARC.Color
components_mask = _ARC.components_mask
find_components = _ARC.find_components
get_bg = _ARC.get_bg


BG = int(Color.BLACK)
GREY = int(Color.GREY)
MASK_SIZE = 3
PALETTE_LEN = 4
LEFT_MASK_COL = 0
GREY_MIN_COL = 5
MAX_GENERATION_ATTEMPTS = 400
COLOR_POOL = [1, 2, 3, 4, 6, 7, 8, 9]


def _corner_count(mask: np.ndarray) -> int:
    return int(mask[0, 0]) + int(mask[0, -1]) + int(mask[-1, 0]) + int(mask[-1, -1])


def _build_mask_catalog() -> dict[int, list[np.ndarray]]:
    catalog = {1: [], 2: [], 3: [], 4: []}
    seen = {k: set() for k in catalog}
    for bits in range(1 << (MASK_SIZE * MASK_SIZE)):
        mask = np.array([(bits >> i) & 1 for i in range(9)], dtype=bool).reshape(MASK_SIZE, MASK_SIZE)
        if not mask[1, 1]:
            continue
        area = int(mask.sum())
        if not 5 <= area <= 7:
            continue
        k = _corner_count(mask)
        if k not in catalog:
            continue
        if not (mask[0].any() and mask[-1].any() and mask[:, 0].any() and mask[:, -1].any()):
            continue
        if len(components_mask(mask, connectivity=8)) != 1:
            continue
        key = mask.astype(np.uint8).tobytes()
        if key not in seen[k]:
            seen[k].add(key)
            catalog[k].append(mask.copy())
    for k, masks in catalog.items():
        assert masks, f"missing mask patterns for corner count {k}"
    return catalog


MASK_CATALOG = _build_mask_catalog()


def _palette_from_top_row(grid: np.ndarray, bg: int) -> list[int]:
    palette = []
    seen = set()
    for value in grid[0]:
        color = int(value)
        if color == bg:
            break
        assert color not in seen
        seen.add(color)
        palette.append(color)
    assert len(palette) >= PALETTE_LEN
    return palette


def _extract_masks(main_area: np.ndarray, bg: int) -> list[dict]:
    mask_layer = main_area.copy()
    mask_layer[mask_layer == GREY] = bg
    components = find_components(mask_layer, bg=bg, connectivity=8, monochromatic=False)
    masks = []
    for comp in components:
        r0, c0, r1, c1 = comp["bbox"]
        if (r1 - r0, c1 - c0) != (2, 2):
            continue
        if c0 > 2:
            continue
        patch = mask_layer[r0 : r1 + 1, c0 : c1 + 1]
        colors = set(map(int, np.unique(patch))) - {bg}
        if len(colors) != 2:
            continue
        occupancy = patch != bg
        masks.append(
            {
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
                "center_r": int(r0 + 1),
                "patch": patch.copy(),
                "occupancy": occupancy,
            }
        )
    masks.sort(key=lambda item: item["center_r"])
    assert masks
    return masks


def _band_ranges(height: int, centers: list[int]) -> list[tuple[int, int]]:
    bands = []
    start = 0
    for i, center in enumerate(centers):
        if i + 1 < len(centers):
            end = (center + centers[i + 1]) // 2
        else:
            end = height - 1
        assert start <= end
        bands.append((int(start), int(end)))
        start = end + 1
    assert bands[0][0] == 0 and bands[-1][1] == height - 1
    return bands


def main(grid: np.ndarray) -> np.ndarray:
    """Recolor each horizontal band of grey placeholders by indexing the top palette with its left mask's occupied-corner count.

    Input structure:
    - The top row starts with a left-aligned palette of distinct colors, followed by background cells; the next row is a solid separator.
    - Below the separator is a main area whose left edge contains several separated 3x3 two-color mask blocks.
    - The rest of the main area contains grey placeholder shapes on the background, while mask colors may overlap the palette as decoys.

    Rule:
    - Sort the left-edge masks by row and split the main area into horizontal bands at the midpoints between adjacent mask center rows.
    - For each mask, count how many of the four corners of its 3x3 footprint are occupied, call that count k, and pick the k-th palette color from the top row.
    - Recolor every grey placeholder cell in that mask's band with the chosen palette color, leaving the palette, separator, masks, and background unchanged.

    Key insight:
    - The mask colors and detailed interiors are distractions; only the number of occupied corners matters, and grey ownership is by vertical band around the nearest mask rather than by row alignment or connected shape.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape[0] >= 5 and g.shape[1] >= 8

    bg = int(get_bg(g))
    palette = _palette_from_top_row(g, bg)
    assert len(set(map(int, g[1]))) == 1

    main_area = g[2:]
    masks = _extract_masks(main_area, bg)
    centers = [item["center_r"] for item in masks]
    bands = _band_ranges(main_area.shape[0], centers)

    out = g.copy()
    for band, mask in zip(bands, masks):
        k = _corner_count(mask["occupancy"])
        assert 1 <= k <= len(palette)
        color = int(palette[k - 1])
        r0, r1 = band
        band_view = out[2 + r0 : 2 + r1 + 1, :]
        band_view[band_view == GREY] = color
    return out


def _choose_band_count() -> int:
    return random.choice([3, 4, 5])


def _choose_corner_counts(count: int) -> list[int]:
    if count == 3:
        ks = random.sample([1, 2, 3, 4], 3)
    elif count == 4:
        ks = [1, 2, 3, 4]
    else:
        ks = [1, 2, 3, 4, random.choice([1, 2, 3, 4])]
    random.shuffle(ks)
    return ks


def _choose_layout(count: int) -> tuple[int, list[int]]:
    diff_pool = [3, 5] if count == 5 else [3, 5, 7]
    while True:
        diffs = [random.choice(diff_pool) for _ in range(count - 1)]
        if count > 2 and len(set(diffs)) == 1:
            continue
        top_slack = random.randint(0, 3)
        bottom_slack = random.randint(0, 3)
        main_h = top_slack + MASK_SIZE + sum(diffs) + bottom_slack
        if main_h > 28:
            continue
        if top_slack == 0 and bottom_slack == 0 and not any(diff > 3 for diff in diffs):
            continue
        tops = [top_slack]
        for diff in diffs:
            tops.append(tops[-1] + diff)
        if tops[-1] + MASK_SIZE > main_h:
            continue
        return main_h, tops


def _choose_palette_and_mask_colors(count: int) -> tuple[list[int], int, list[int], list[int]]:
    palette = random.sample(COLOR_POOL, PALETTE_LEN)
    separator = random.choice([color for color in COLOR_POOL if color not in palette])
    non_palette = [color for color in COLOR_POOL if color not in palette and color != separator]
    overlap_min = max(1, count - len(non_palette))
    overlap_max = min(3, count - 1, len(palette))
    overlap_count = random.randint(overlap_min, overlap_max)
    overlap_colors = random.sample(palette, overlap_count)
    primary_rest = random.sample(non_palette, count - overlap_count)
    primaries = overlap_colors + primary_rest
    random.shuffle(primaries)

    secondaries = []
    for primary in primaries:
        choices = [color for color in COLOR_POOL if color != primary and color != separator]
        secondaries.append(random.choice(choices))
    return palette, separator, primaries, secondaries


def _render_mask(mask: np.ndarray, primary: int, secondary: int) -> np.ndarray:
    patch = np.full((MASK_SIZE, MASK_SIZE), BG, dtype=int)
    patch[mask] = int(primary)
    alt_cells = [(1, 1)]
    others = [(int(r), int(c)) for r, c in np.argwhere(mask) if (int(r), int(c)) != (1, 1)]
    random.shuffle(others)
    alt_cells.extend(others[: random.randint(0, min(2, len(others)))])
    for r, c in alt_cells:
        patch[r, c] = int(secondary)
    assert len(set(map(int, np.unique(patch))) - {BG}) == 2
    return patch


def _random_placeholder_patch(max_h: int, max_w: int) -> np.ndarray:
    h = random.randint(1, max_h)
    w = random.randint(1, max_w)
    min_area = 1 if h * w == 1 else 2
    target = random.randint(min_area, min(5, h * w))
    cells = {(random.randrange(h), random.randrange(w))}
    frontier = set()
    while len(cells) < target:
        for r, c in list(cells):
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                rr, cc = r + dr, c + dc
                if 0 <= rr < h and 0 <= cc < w and (rr, cc) not in cells:
                    frontier.add((rr, cc))
        if not frontier:
            break
        cells.add(random.choice(tuple(frontier)))
        frontier.clear()

    patch = np.full((h, w), BG, dtype=int)
    for r, c in cells:
        patch[r, c] = GREY

    rows = np.any(patch == GREY, axis=1)
    cols = np.any(patch == GREY, axis=0)
    patch = patch[np.ix_(rows, cols)]
    return patch


def _place_patch(area: np.ndarray, patch: np.ndarray, top: int, left: int, color: int) -> None:
    mask = patch == GREY
    block = area[top : top + patch.shape[0], left : left + patch.shape[1]]
    assert np.all(block[mask] == BG)
    block[mask] = color


def _add_band_placeholders(
    main_area: np.ndarray,
    band: tuple[int, int],
    mask_top: int,
    width: int,
) -> bool:
    band_top, band_bottom = band
    band_h = band_bottom - band_top + 1
    shapes_to_place = random.randint(1, 3 if width >= 17 else 2)
    placed = 0
    rows_outside_mask = [r for r in range(band_top, band_bottom + 1) if r < mask_top or r > mask_top + 2]
    forced_off_mask = bool(rows_outside_mask)

    for shape_index in range(shapes_to_place):
        success = False
        for _ in range(80):
            patch = _random_placeholder_patch(min(3, band_h), min(5, width - GREY_MIN_COL))
            ph, pw = patch.shape
            if ph > band_h or GREY_MIN_COL + pw > width:
                continue
            if forced_off_mask and shape_index == 0:
                valid_tops = [
                    top
                    for top in range(band_top, band_bottom - ph + 2)
                    if any((top + rr) in rows_outside_mask for rr in range(ph) if np.any(patch[rr] == GREY))
                ]
                if not valid_tops:
                    continue
                top = random.choice(valid_tops)
            else:
                top = random.randint(band_top, band_bottom - ph + 1)
            left = random.randint(GREY_MIN_COL, width - pw)
            block = main_area[top : top + ph, left : left + pw]
            if np.any((patch == GREY) & (block != BG)):
                continue
            _place_patch(main_area, patch, top, left, GREY)
            placed += 1
            success = True
            break
        if not success and placed == 0:
            return False
    return placed > 0


def _oracle_output(grid: np.ndarray, bands: list[tuple[int, int]], ks: list[int], palette: list[int]) -> np.ndarray:
    out = grid.copy()
    for (r0, r1), k in zip(bands, ks):
        band_view = out[2 + r0 : 2 + r1 + 1, :]
        band_view[band_view == GREY] = int(palette[k - 1])
    return out


def _generate_one() -> np.ndarray | None:
    band_count = _choose_band_count()
    ks = _choose_corner_counts(band_count)
    main_h, mask_tops = _choose_layout(band_count)
    width = random.randint(12, 26)
    if width - GREY_MIN_COL < 2:
        return None

    palette, separator, primaries, secondaries = _choose_palette_and_mask_colors(band_count)
    main_area = np.full((main_h, width), BG, dtype=int)

    mask_patterns = [random.choice(MASK_CATALOG[k]).copy() for k in ks]
    for top, pattern, primary, secondary in zip(mask_tops, mask_patterns, primaries, secondaries):
        patch = _render_mask(pattern, primary, secondary)
        main_area[top : top + MASK_SIZE, LEFT_MASK_COL : LEFT_MASK_COL + MASK_SIZE] = patch

    centers = [top + 1 for top in mask_tops]
    bands = _band_ranges(main_h, centers)
    for band, mask_top in zip(bands, mask_tops):
        if not _add_band_placeholders(main_area, band, mask_top, width):
            return None

    if not np.any(main_area == GREY):
        return None

    grid = np.full((main_h + 2, width), BG, dtype=int)
    grid[0, :PALETTE_LEN] = np.array(palette, dtype=int)
    grid[1, :] = int(separator)
    grid[2:, :] = main_area

    expected = _oracle_output(grid, bands, ks, palette)
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None

    parsed_masks = _extract_masks(grid[2:], BG)
    if len(parsed_masks) != band_count:
        return None
    parsed_ks = [_corner_count(item["occupancy"]) for item in parsed_masks]
    if parsed_ks != ks:
        return None
    if any(not np.any(grid[2 + r0 : 2 + r1 + 1, :] == GREY) for r0, r1 in bands):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generate_one()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 6")
