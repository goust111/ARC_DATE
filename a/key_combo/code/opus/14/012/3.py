from __future__ import annotations

import random
from collections import defaultdict

import numpy as np


BG = 5
OUTLINE_PALETTE = (1, 2, 3, 4)
BAND_PALETTE = (6, 7, 8, 9)
NEIGHBORS8 = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
)


def _find_monochrome_components(grid: np.ndarray, bg: int) -> list[dict]:
    height, width = map(int, grid.shape)
    visited = np.zeros((height, width), dtype=bool)
    components: list[dict] = []

    for row in range(height):
        for col in range(width):
            color = int(grid[row, col])
            if color == bg or visited[row, col]:
                continue

            stack = [(row, col)]
            visited[row, col] = True
            coords: list[tuple[int, int]] = []
            while stack:
                r, c = stack.pop()
                coords.append((r, c))
                for dr, dc in NEIGHBORS8:
                    rr = r + dr
                    cc = c + dc
                    if 0 <= rr < height and 0 <= cc < width and not visited[rr, cc] and int(grid[rr, cc]) == color:
                        visited[rr, cc] = True
                        stack.append((rr, cc))

            coord_array = np.array(coords, dtype=int)
            r0, c0 = coord_array.min(axis=0)
            r1, c1 = coord_array.max(axis=0)
            components.append(
                {
                    "color": color,
                    "coords": coord_array,
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                }
            )

    return components


def _fill_outline_component(shape: tuple[int, int], coords: np.ndarray) -> tuple[np.ndarray, int]:
    filled = np.zeros(shape, dtype=bool)
    row_to_cols: dict[int, list[int]] = defaultdict(list)
    for row, col in coords:
        row_to_cols[int(row)].append(int(col))

    for row, cols in row_to_cols.items():
        assert len(cols) == 2, "expected exactly two contour pixels on each occupied row"
        c0 = min(cols)
        c1 = max(cols)
        assert c1 - c0 >= 1, "expected each contour row to span at least two columns"
        filled[row, c0 : c1 + 1] = True

    area = int(filled.sum())
    return filled, area


def _analyze(grid: np.ndarray) -> tuple[list[int], list[dict]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    height, width = map(int, g.shape)
    assert height >= 6 and width >= 6, "expected room for outlines and a bottom band"

    band_positions = [col for col in range(width) if int(g[height - 1, col]) != BG]
    assert 3 <= len(band_positions) <= 4, "expected 3-4 bottom-band pixels"
    band_colors = [int(g[height - 1, col]) for col in band_positions]
    assert len(set(band_colors)) == len(band_colors), "expected distinct bottom-band colors"

    work = g.copy()
    work[height - 1, :] = BG
    components = _find_monochrome_components(work, BG)
    assert len(components) == len(band_colors), "expected one contour per bottom-band color"

    objects: list[dict] = []
    for comp in components:
        filled, area = _fill_outline_component(work.shape, comp["coords"])
        rows, cols = np.where(filled)
        r0 = int(rows.min())
        c0 = int(cols.min())
        r1 = int(rows.max())
        c1 = int(cols.max())
        objects.append(
            {
                "color": int(comp["color"]),
                "outline_coords": comp["coords"],
                "filled_mask": filled,
                "area": area,
                "bbox": (r0, c0, r1, c1),
                "height": int(r1 - r0 + 1),
                "left": c0,
                "top": r0,
            }
        )

    areas = [obj["area"] for obj in objects]
    assert len(set(areas)) == len(areas), "expected unique filled areas"
    return band_colors, objects


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each hollow row-convex contour, rank the filled regions by area, and recolor them from the bottom band.

    Input structure:
    - The grid uses grey as background, with 3-4 monochrome 8-connected contour objects above the last row.
    - Each contour occupies several rows and has exactly two contour pixels on every occupied row, which are that row's left and right boundary pixels.
    - The bottom row contains a left-to-right band of distinct non-grey colors, one pixel per color, and those band colors do not belong to the contours.

    Rule:
    - Read the bottom-band colors from left to right.
    - For each contour object, fill every occupied row between its two boundary pixels, inclusive, to recover one solid region and count its area.
    - Sort the solid regions from largest area to smallest area and assign them the bottom-band colors in that order.
    - Recolor each whole filled region with its assigned band color, remove the bottom band by turning the entire last row grey, and keep all other background cells grey.

    Key insight:
    - The contour color is only a temporary label: the decisive information is the filled area obtained after row-span filling, because the final recoloring is driven solely by area rank.
    """
    band_colors, objects = _analyze(grid)

    out = np.full_like(np.asarray(grid, dtype=int), BG)
    ranked = sorted(objects, key=lambda obj: -int(obj["area"]))
    for band_color, obj in zip(band_colors, ranked):
        out[obj["filled_mask"]] = int(band_color)
    out[-1, :] = BG
    return out


def _sample_monotone_path(start: int, end: int, steps: int, deltas: tuple[int, ...], rng: np.random.Generator) -> list[int] | None:
    path = [int(start)]
    cur = int(start)
    for step in range(steps):
        remaining = steps - step - 1
        candidates = []
        for delta in deltas:
            nxt = cur + int(delta)
            min_reachable = nxt + remaining * min(deltas)
            max_reachable = nxt + remaining * max(deltas)
            if min_reachable <= end <= max_reachable:
                candidates.append(nxt)
        if not candidates:
            return None
        cur = int(candidates[int(rng.integers(len(candidates)))])
        path.append(cur)
    return path if cur == end else None


def _mask_from_widths(widths: list[int], rng: np.random.Generator) -> np.ndarray | None:
    lefts = [0]
    rights = [int(widths[0] - 1)]
    for idx in range(1, len(widths)):
        delta_w = int(widths[idx] - widths[idx - 1])
        valid_dls = []
        for dl in (-1, 0, 1):
            dr = dl + delta_w
            if -1 <= dr <= 1:
                valid_dls.append(dl)
        if not valid_dls:
            return None
        dl = int(valid_dls[int(rng.integers(len(valid_dls)))])
        lefts.append(lefts[-1] + dl)
        rights.append(lefts[-1] + widths[idx] - 1)

    min_left = min(lefts)
    lefts = [left - min_left for left in lefts]
    rights = [right - min_left for right in rights]
    width = max(rights) + 1
    mask = np.zeros((len(widths), width), dtype=bool)
    for row, (c0, c1) in enumerate(zip(lefts, rights)):
        mask[row, c0 : c1 + 1] = True

    if rng.random() < 0.5:
        mask = mask[:, ::-1]
    if rng.random() < 0.5:
        mask = mask[::-1, :]
    return mask


def _generate_shape_mask(rng: np.random.Generator) -> np.ndarray | None:
    for _ in range(300):
        height = int(rng.integers(5, 9))
        peak_row = int(rng.integers(1, height - 1))
        max_peak = min(8, 2 + 2 * min(peak_row, height - 1 - peak_row))
        if max_peak < 4:
            continue
        peak_width = int(rng.integers(4, max_peak + 1))

        upper = _sample_monotone_path(2, peak_width, peak_row, (0, 1, 2), rng)
        lower = _sample_monotone_path(peak_width, 2, height - 1 - peak_row, (-2, -1, 0), rng)
        if upper is None or lower is None:
            continue

        widths = upper + lower[1:]
        if widths[0] != 2 or widths[-1] != 2:
            continue
        if max(widths) < 4:
            continue
        if len(set(widths)) < 3:
            continue

        mask = _mask_from_widths(widths, rng)
        if mask is None:
            continue
        if mask.shape[1] > 9:
            continue

        interior_rows = widths[1:-1]
        if not interior_rows or max(interior_rows) < 5:
            continue
        return mask
    return None


def _outline_from_filled(mask: np.ndarray) -> np.ndarray:
    outline = np.zeros_like(mask, dtype=bool)
    for row in range(mask.shape[0]):
        cols = np.flatnonzero(mask[row])
        assert cols.size >= 2, "expected one contiguous span per row"
        c0 = int(cols[0])
        c1 = int(cols[-1])
        assert np.all(mask[row, c0 : c1 + 1]), "expected contiguous row spans"
        outline[row, c0] = True
        outline[row, c1] = True
    return outline


def _dilate_cells(cells: list[tuple[int, int]], shape: tuple[int, int]) -> set[tuple[int, int]]:
    height, width = shape
    blocked: set[tuple[int, int]] = set()
    for row, col in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr = row + dr
                cc = col + dc
                if 0 <= rr < height and 0 <= cc < width:
                    blocked.add((rr, cc))
    return blocked


def _place_shapes(shape_masks: list[np.ndarray], grid_shape: tuple[int, int], rng: np.random.Generator) -> list[dict] | None:
    height, width = grid_shape
    blocked: set[tuple[int, int]] = set()
    placed: list[dict] = []

    order = list(range(len(shape_masks)))
    rng.shuffle(order)
    for idx in order:
        mask = shape_masks[idx]
        obj_h, obj_w = map(int, mask.shape)
        candidates = []
        for top in range(1, height - obj_h - 1):
            for left in range(1, width - obj_w - 1):
                coords = [(top + r, left + c) for r, c in np.argwhere(mask)]
                if any(cell in blocked for cell in coords):
                    continue
                candidates.append((top, left, coords))
        if not candidates:
            return None

        top, left, coords = candidates[int(rng.integers(len(candidates)))]
        placed.append(
            {
                "index": idx,
                "top": int(top),
                "left": int(left),
                "filled_coords": coords,
                "filled_mask_local": mask,
                "outline_local": _outline_from_filled(mask),
                "area": int(mask.sum()),
                "height": int(mask.shape[0]),
                "width": int(mask.shape[1]),
            }
        )
        blocked.update(_dilate_cells(coords, grid_shape))

    placed.sort(key=lambda item: int(item["index"]))
    return placed


def _build_expected(grid_shape: tuple[int, int], placed: list[dict], band_colors: list[int]) -> np.ndarray:
    out = np.full(grid_shape, BG, dtype=int)
    ranked = sorted(placed, key=lambda item: -int(item["area"]))
    for band_color, obj in zip(band_colors, ranked):
        for row, col in obj["filled_coords"]:
            out[row, col] = int(band_color)
    out[-1, :] = BG
    return out


def _render_input(grid_shape: tuple[int, int], placed: list[dict], outline_colors: list[int], band_colors: list[int], band_cols: list[int]) -> np.ndarray:
    grid = np.full(grid_shape, BG, dtype=int)
    for obj, color in zip(placed, outline_colors):
        outline_local = obj["outline_local"]
        top = int(obj["top"])
        left = int(obj["left"])
        for row, col in np.argwhere(outline_local):
            grid[top + int(row), left + int(col)] = int(color)
    for col, color in zip(band_cols, band_colors):
        grid[-1, int(col)] = int(color)
    return grid


def _has_ranking_inversion(areas: list[int], heights: list[int]) -> bool:
    area_order = sorted(range(len(areas)), key=lambda idx: -int(areas[idx]))
    height_order = sorted(range(len(heights)), key=lambda idx: -int(heights[idx]))
    return area_order != height_order


def _passes_semantic_checks(inp: np.ndarray, expected: np.ndarray) -> bool:
    try:
        band_colors, objects = _analyze(inp)
    except AssertionError:
        return False

    if not np.array_equal(main(inp), expected):
        return False

    areas = [int(obj["area"]) for obj in objects]
    heights = [int(obj["height"]) for obj in objects]
    left_order = sorted(range(len(objects)), key=lambda idx: int(objects[idx]["left"]))
    top_order = sorted(range(len(objects)), key=lambda idx: int(objects[idx]["top"]))
    area_order = sorted(range(len(objects)), key=lambda idx: -int(objects[idx]["area"]))

    if len(set(areas)) != len(areas):
        return False
    if not _has_ranking_inversion(areas, heights):
        return False
    if area_order == left_order:
        return False
    if area_order == top_order:
        return False
    if len(band_colors) != len(objects):
        return False
    if not np.all(expected[-1] == BG):
        return False
    return True


def _generate_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()
    obj_count = 3 + int(rng.integers(2))
    height = int(rng.integers(17, 23))
    width = int(rng.integers(17, 23))
    grid_shape = (height, width)

    shape_masks: list[np.ndarray] = []
    for _ in range(obj_count):
        mask = _generate_shape_mask(rng)
        if mask is None:
            return None
        shape_masks.append(mask)

    areas = [int(mask.sum()) for mask in shape_masks]
    if len(set(areas)) != len(areas):
        return None
    heights = [int(mask.shape[0]) for mask in shape_masks]
    if not _has_ranking_inversion(areas, heights):
        return None

    placed = _place_shapes(shape_masks, grid_shape, rng)
    if placed is None:
        return None

    area_order = sorted(range(obj_count), key=lambda idx: -int(placed[idx]["area"]))
    left_order = sorted(range(obj_count), key=lambda idx: int(placed[idx]["left"]))
    top_order = sorted(range(obj_count), key=lambda idx: int(placed[idx]["top"]))
    if area_order == left_order or area_order == top_order:
        return None

    band_colors = random.sample(BAND_PALETTE, obj_count)
    outline_colors = random.sample(OUTLINE_PALETTE, obj_count)
    band_cols = sorted(random.sample(range(1, width - 1), obj_count))

    inp = _render_input(grid_shape, placed, outline_colors, band_colors, band_cols)
    expected = _build_expected(grid_shape, placed, band_colors)
    return inp if _passes_semantic_checks(inp, expected) else None


def generate_input() -> np.ndarray:
    for _ in range(500):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 3")
