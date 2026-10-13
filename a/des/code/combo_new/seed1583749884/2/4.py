from __future__ import annotations

import random

import numpy as np


BG = 8
MARKER = 4
DIR4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
OBJECT_COLORS = (0, 1, 2, 3, 5, 6, 7, 9)
GEN_ATTEMPTS = 400


def _mask_components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    mask = np.asarray(mask, dtype=bool)
    height, width = map(int, mask.shape)
    seen = np.zeros((height, width), dtype=bool)
    comps: list[list[tuple[int, int]]] = []

    for r in range(height):
        for c in range(width):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(int(r), int(c))]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((int(rr), int(cc)))
                for dr, dc in DIR4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < height and 0 <= nc < width and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((int(nr), int(nc)))
            comps.append(coords)
    return comps


def _extract_regions(grid: np.ndarray) -> list[dict[str, object]]:
    g = np.asarray(grid, dtype=int)
    height, width = map(int, g.shape)
    seen = np.zeros((height, width), dtype=bool)
    regions: list[dict[str, object]] = []

    for r in range(height):
        for c in range(width):
            if seen[r, c] or int(g[r, c]) == BG:
                continue

            stack = [(int(r), int(c))]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            colors: set[int] = set()
            marker_count = 0

            while stack:
                rr, cc = stack.pop()
                value = int(g[rr, cc])
                cells.append((int(rr), int(cc)))
                if value == MARKER:
                    marker_count += 1
                else:
                    colors.add(value)

                for dr, dc in DIR4:
                    nr, nc = rr + dr, cc + dc
                    if nr < 0 or nr >= height or nc < 0 or nc >= width or seen[nr, nc]:
                        continue
                    next_value = int(g[nr, nc])
                    if next_value == BG:
                        continue
                    if value == MARKER or next_value == MARKER or value == next_value:
                        seen[nr, nc] = True
                        stack.append((int(nr), int(nc)))

            assert len(colors) == 1, "each region must have exactly one non-marker color"
            regions.append(
                {
                    "color": int(next(iter(colors))),
                    "cells": cells,
                    "marker_count": int(marker_count),
                }
            )

    return regions


def _expand_regions(
    shape: tuple[int, int],
    regions: list[dict[str, object]],
    original_fg: np.ndarray,
) -> np.ndarray:
    height, width = map(int, shape)
    out = np.full((height, width), BG, dtype=int)
    proposals: dict[tuple[int, int], set[int]] = {}

    for region in regions:
        color = int(region["color"])
        for r, c in region["cells"]:
            out[int(r), int(c)] = color

    for region in regions:
        color = int(region["color"])
        distance = int(region["marker_count"])
        if distance <= 0:
            continue
        cells = {tuple(cell) for cell in region["cells"]}

        for r, c in cells:
            for dr, dc in DIR4:
                nr, nc = r + dr, c + dc
                if 0 <= nr < height and 0 <= nc < width and (nr, nc) in cells:
                    continue

                for step in range(1, distance + 1):
                    rr = r + dr * step
                    cc = c + dc * step
                    if rr < 0 or rr >= height or cc < 0 or cc >= width:
                        break
                    if bool(original_fg[rr, cc]):
                        break
                    proposals.setdefault((int(rr), int(cc)), set()).add(color)

    for (r, c), colors in proposals.items():
        if len(colors) == 1:
            out[int(r), int(c)] = int(next(iter(colors)))

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand each colored region outward from its exposed boundary by the number of yellow markers it contains.

    Input structure:
    - The background is Azure.
    - The grid contains several compact 4-connected foreground regions in colors such as Red, Green, Blue, or Maroon.
    - Each region contains one to four Yellow marker pixels embedded inside the same object.

    Rule:
    - For each region, count its Yellow marker pixels and call that distance N.
    - Recolor the Yellow marker pixels to the region's own foreground color.
    - For every exposed 4-neighbor side of every region pixel, extend a straight segment of that region color outward by N cells along the local normal direction.
    - Stop an outward segment at the grid edge or just before an originally occupied foreground cell, so expansions never overwrite existing regions.

    Key insight:
    - The Yellow pixels are parameters rather than final content: their count determines how far the region's boundary rays are pushed outward.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    if not bool(np.any(g != BG)):
        return g.copy()

    regions = _extract_regions(g)
    return _expand_regions(g.shape, regions, g != BG)


def _interior_cells(mask: np.ndarray) -> list[tuple[int, int]]:
    height, width = map(int, mask.shape)
    cells: list[tuple[int, int]] = []
    for r, c in np.argwhere(mask):
        rr, cc = int(r), int(c)
        if all(
            0 <= rr + dr < height
            and 0 <= cc + dc < width
            and bool(mask[rr + dr, cc + dc])
            for dr, dc in DIR4
        ):
            cells.append((rr, cc))
    return cells


def _pick_marker_cells(candidates: list[tuple[int, int]], count: int) -> list[tuple[int, int]] | None:
    if count > len(candidates):
        return None

    pool = list(candidates)
    for _ in range(80):
        random.shuffle(pool)
        picked: list[tuple[int, int]] = []
        for cell in pool:
            if all(abs(cell[0] - other[0]) + abs(cell[1] - other[1]) > 1 for other in picked):
                picked.append(cell)
                if len(picked) == count:
                    return picked
    return None


def _corner_coords(height: int, width: int) -> dict[str, tuple[int, int]]:
    return {
        "TL": (0, 0),
        "TR": (0, width - 1),
        "BL": (height - 1, 0),
        "BR": (height - 1, width - 1),
    }


def _apply_shape_variant(mask: np.ndarray, variant: str) -> bool:
    height, width = map(int, mask.shape)
    corners = _corner_coords(height, width)

    if variant == "rect":
        return True

    if variant == "single_corner":
        r, c = random.choice(list(corners.values()))
        mask[r, c] = False
        return True

    if variant == "opposite_corners":
        pair = random.choice([("TL", "BR"), ("TR", "BL")])
        for name in pair:
            r, c = corners[name]
            mask[r, c] = False
        return True

    if variant == "corner_domino":
        name = random.choice(list(corners))
        orientation = random.choice(["h", "v"])
        if name == "TL":
            coords = [(0, 0), (0, 1)] if orientation == "h" else [(0, 0), (1, 0)]
        elif name == "TR":
            coords = [(0, width - 1), (0, width - 2)] if orientation == "h" else [(0, width - 1), (1, width - 1)]
        elif name == "BL":
            coords = [(height - 1, 0), (height - 1, 1)] if orientation == "h" else [(height - 1, 0), (height - 2, 0)]
        else:
            coords = (
                [(height - 1, width - 1), (height - 1, width - 2)]
                if orientation == "h"
                else [(height - 1, width - 1), (height - 2, width - 1)]
            )
        if any(r < 0 or r >= height or c < 0 or c >= width for r, c in coords):
            return False
        for r, c in coords:
            mask[r, c] = False
        return True

    if variant == "side_pair":
        side = random.choice(["top", "bottom", "left", "right"])
        if side == "top":
            coords = [(0, 0), (0, width - 1)]
        elif side == "bottom":
            coords = [(height - 1, 0), (height - 1, width - 1)]
        elif side == "left":
            coords = [(0, 0), (height - 1, 0)]
        else:
            coords = [(0, width - 1), (height - 1, width - 1)]
        for r, c in coords:
            mask[r, c] = False
        return True

    return False


def _make_region_mask(marker_count: int) -> tuple[np.ndarray, list[tuple[int, int]]] | None:
    min_side = 4 if marker_count <= 2 else 5
    variants = ("rect", "single_corner", "opposite_corners", "corner_domino", "side_pair")

    for _ in range(200):
        height = random.randint(min_side, 7)
        width = random.randint(min_side, 7)
        mask = np.ones((height, width), dtype=bool)
        if not _apply_shape_variant(mask, random.choice(variants)):
            continue
        if np.count_nonzero(mask) == 0 or len(_mask_components(mask)) != 1:
            continue

        interior = _interior_cells(mask)
        if len(interior) < marker_count:
            continue
        markers = _pick_marker_cells(interior, marker_count)
        if markers is None:
            continue
        return mask, markers

    return None


def _boxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _choose_canvas(specs: list[dict[str, object]]) -> tuple[int, int] | None:
    max_halo_h = 0
    max_halo_w = 0
    total_halo_area = 0

    for spec in specs:
        halo_h = int(spec["height"]) + 2 * int(spec["marker_count"])
        halo_w = int(spec["width"]) + 2 * int(spec["marker_count"])
        max_halo_h = max(max_halo_h, halo_h)
        max_halo_w = max(max_halo_w, halo_w)
        total_halo_area += halo_h * halo_w

    base = max(max_halo_h + 1, max_halo_w + 1, int(np.ceil(np.sqrt(total_halo_area * 1.4))))
    if base > 30:
        return None

    min_h = max(12, base)
    min_w = max(12, base)
    if min_h > 30 or min_w > 30:
        return None

    height = random.randint(min_h, 30)
    width = random.randint(min_w, 30)
    return int(height), int(width)


def _place_regions(specs: list[dict[str, object]], shape: tuple[int, int]) -> list[dict[str, object]] | None:
    height, width = map(int, shape)
    placed: list[dict[str, object]] = []
    halo_boxes: list[tuple[int, int, int, int]] = []

    order = list(range(len(specs)))
    random.shuffle(order)

    for idx in order:
        spec = specs[idx]
        region_h = int(spec["height"])
        region_w = int(spec["width"])
        distance = int(spec["marker_count"])

        r_lo = distance
        c_lo = distance
        r_hi = height - region_h - distance
        c_hi = width - region_w - distance
        if r_lo > r_hi or c_lo > c_hi:
            return None

        placed_here = False
        for _ in range(250):
            r0 = random.randint(r_lo, r_hi)
            c0 = random.randint(c_lo, c_hi)
            halo = (r0 - distance, c0 - distance, r0 + region_h - 1 + distance, c0 + region_w - 1 + distance)
            if any(_boxes_overlap(halo, other) for other in halo_boxes):
                continue

            cells = [(r0 + int(r), c0 + int(c)) for r, c in spec["local_cells"]]
            marker_cells = [(r0 + int(r), c0 + int(c)) for r, c in spec["local_markers"]]
            placed.append(
                {
                    "color": int(spec["color"]),
                    "cells": cells,
                    "marker_cells": marker_cells,
                    "marker_count": distance,
                }
            )
            halo_boxes.append(halo)
            placed_here = True
            break

        if not placed_here:
            return None

    return placed


def _render_input(shape: tuple[int, int], regions: list[dict[str, object]]) -> np.ndarray:
    grid = np.full(shape, BG, dtype=int)
    for region in regions:
        color = int(region["color"])
        for r, c in region["cells"]:
            grid[int(r), int(c)] = color
        for r, c in region["marker_cells"]:
            grid[int(r), int(c)] = MARKER
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        n_regions = random.randint(2, 4)
        colors = random.sample(OBJECT_COLORS, n_regions)

        specs: list[dict[str, object]] = []
        ok = True
        for color in colors:
            marker_count = random.randint(1, 4)
            made = _make_region_mask(marker_count)
            if made is None:
                ok = False
                break
            mask, local_markers = made
            specs.append(
                {
                    "color": int(color),
                    "marker_count": int(marker_count),
                    "height": int(mask.shape[0]),
                    "width": int(mask.shape[1]),
                    "local_cells": [(int(r), int(c)) for r, c in np.argwhere(mask)],
                    "local_markers": [(int(r), int(c)) for r, c in local_markers],
                }
            )
        if not ok:
            continue

        canvas = _choose_canvas(specs)
        if canvas is None:
            continue

        regions = _place_regions(specs, canvas)
        if regions is None:
            continue

        grid = _render_input(canvas, regions)
        expected_regions = [
            {
                "color": int(region["color"]),
                "cells": list(region["cells"]),
                "marker_count": int(region["marker_count"]),
            }
            for region in regions
        ]
        expected = _expand_regions(grid.shape, expected_regions, grid != BG)

        try:
            extracted = _extract_regions(grid)
            actual = main(grid)
        except AssertionError:
            continue

        extracted_sig = sorted((int(region["color"]), int(region["marker_count"])) for region in extracted)
        expected_sig = sorted((int(region["color"]), int(region["marker_count"])) for region in expected_regions)
        if extracted_sig != expected_sig:
            continue
        if not np.array_equal(actual, expected):
            continue

        return grid

    raise RuntimeError("failed to generate a valid input")
