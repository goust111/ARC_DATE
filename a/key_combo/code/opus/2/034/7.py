from __future__ import annotations

import math
import random
import sys
from pathlib import Path

import numpy as np

# Make the shared ARC toolbox importable even when this file is loaded via SourceFileLoader.
_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(_ARC_UTILS_DIR))

from arc_utils import Color, find_components, get_bg, paste, random_free_location


PALETTE = [Color.BLUE, Color.RED, Color.GREEN, Color.YELLOW, Color.GREY, Color.ORANGE]
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
MAX_GENERATION_ATTEMPTS = 400


def _kept_colors(grid: np.ndarray, bg: int) -> list[dict]:
    components = find_components(
        grid,
        bg=bg,
        connectivity=4,
        monochromatic=True,
        return_format="dict",
    )
    by_color: dict[int, list[dict]] = {}
    for comp in components:
        area = int(comp["area"])
        if area < 2:
            continue
        color = int(comp["color"])
        by_color.setdefault(color, []).append(comp)

    ranked = []
    for color, comps in by_color.items():
        effective_area = sum(int(comp["area"]) for comp in comps)
        largest_area = max(int(comp["area"]) for comp in comps)
        anchor = min(
            (int(comp["bbox"][0]), int(comp["bbox"][1]), int(comp["bbox"][2]), int(comp["bbox"][3]))
            for comp in comps
            if int(comp["area"]) == largest_area
        )
        ranked.append(
            {
                "color": color,
                "effective_area": effective_area,
                "anchor": anchor,
            }
        )

    ranked.sort(key=lambda item: (-item["effective_area"], item["anchor"]))
    return ranked


def _draw_rings(colors: list[int]) -> np.ndarray:
    side = 2 * len(colors) - 1
    out = np.full((side, side), int(colors[0]), dtype=int)
    for layer, color in enumerate(colors[1:], start=1):
        out[layer : side - layer, layer : side - layer] = int(color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Filter out colors represented only by isolated pixels, then redraw the survivors as area-ordered nested square rings.

    Input structure:
    - The grid has a black background and several non-black monochromatic foreground objects.
    - Foreground objects are 4-connected, and a color may appear in multiple separate objects.
    - Some colors appear only as isolated single pixels, while other colors have at least one multi-pixel component.

    Rule:
    - Find every 4-connected monochromatic non-black component.
    - For each color, compute its effective area as the sum of areas of that color's components with area at least 2, ignoring isolated single pixels.
    - Discard colors whose effective area is 0; sort the remaining colors by decreasing effective area, using the upper-left corner of the color's largest component bbox as a geometric tie-breaker.
    - Create a square output of side 2 * kept_color_count - 1 and paint those colors as concentric 1-pixel square rings from outside to inside.

    Key insight:
    - Isolated pixels are decoys: they reveal a color in the input, but they neither contribute to the ranking nor guarantee that the color appears in the output at all.
    """
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    ranked = _kept_colors(g, bg)
    assert ranked, "expected at least one color with a multi-pixel component"
    return _draw_rings([int(item["color"]) for item in ranked])


def _partition_total(total: int, parts: int) -> list[int]:
    assert parts >= 1
    assert total >= 2 * parts
    values = [2] * parts
    extras = total - 2 * parts
    for _ in range(extras):
        values[random.randrange(parts)] += 1
    random.shuffle(values)
    return values


def _random_connected_patch(area: int, color: int) -> np.ndarray:
    area = int(area)
    for _ in range(120):
        cells = {(0, 0)}
        frontier = set(DIR4)
        while len(cells) < area:
            r, c = random.choice(tuple(frontier))
            frontier.remove((r, c))
            cells.add((r, c))
            for dr, dc in DIR4:
                nr, nc = r + dr, c + dc
                if (nr, nc) in cells:
                    continue
                if max(abs(nr), abs(nc)) > area:
                    continue
                frontier.add((nr, nc))

        rows = [r for r, _ in cells]
        cols = [c for _, c in cells]
        r0, c0 = min(rows), min(cols)
        h = max(rows) - r0 + 1
        w = max(cols) - c0 + 1
        if h > 5 or w > 5:
            continue

        patch = np.zeros((h, w), dtype=int)
        for r, c in cells:
            patch[r - r0, c - c0] = int(color)

        patch = np.rot90(patch, random.randrange(4)).copy()
        if random.random() < 0.5:
            patch = np.flipud(patch).copy()
        if random.random() < 0.5:
            patch = np.fliplr(patch).copy()
        return patch

    raise RuntimeError("failed to sample a connected patch")


def _choose_effective_totals(count: int) -> list[int]:
    totals = random.sample(list(range(3, 10)), count)
    random.shuffle(totals)
    return totals


def _choose_multi_parts(total: int, *, force_split: bool) -> list[int]:
    max_parts = min(3, total // 2)
    if max_parts == 1:
        return [total]
    if force_split:
        parts = random.randint(2, max_parts)
    else:
        parts = random.choices(
            population=list(range(1, max_parts + 1)),
            weights=[4] + [3] * (max_parts - 1),
            k=1,
        )[0]
    return _partition_total(total, parts) if parts > 1 else [total]


def _single_pixel_patch(color: int) -> np.ndarray:
    return np.array([[int(color)]], dtype=int)


def _place_all(shape: tuple[int, int], patches: list[np.ndarray]) -> np.ndarray | None:
    grid = np.zeros(shape, dtype=int)
    ordered = sorted(patches, key=lambda patch: int(np.count_nonzero(patch)), reverse=True)
    for patch in ordered:
        try:
            r0, c0 = random_free_location(
                grid,
                patch,
                transparent=Color.BLACK,
                border=1,
                padding=1,
                padding_connectivity=8,
            )
        except ValueError:
            return None
        paste(grid, patch, r0, c0, transparent=Color.BLACK)
    return grid


def _valid_generated_grid(grid: np.ndarray) -> bool:
    if int(get_bg(grid)) != Color.BLACK:
        return False

    ranked = _kept_colors(grid, Color.BLACK)
    if len(ranked) < 3:
        return False

    areas = [int(item["effective_area"]) for item in ranked]
    if len(areas) != len(set(areas)):
        return False

    present_colors = {int(v) for v in np.unique(grid) if int(v) != Color.BLACK}
    kept_colors = {int(item["color"]) for item in ranked}
    excluded = present_colors - kept_colors
    if not excluded:
        return False

    try:
        out = main(grid)
    except AssertionError:
        return False
    if out.shape != (2 * len(ranked) - 1, 2 * len(ranked) - 1):
        return False
    return True


def _generate_attempt() -> np.ndarray | None:
    kept_count = random.randint(3, 5)
    excluded_count = random.randint(1, min(2, len(PALETTE) - kept_count))
    chosen_colors = random.sample(PALETTE, kept_count + excluded_count)
    random.shuffle(chosen_colors)
    kept_colors = chosen_colors[:kept_count]
    excluded_colors = chosen_colors[kept_count:]

    effective_totals = _choose_effective_totals(kept_count)

    split_candidates = [idx for idx, total in enumerate(effective_totals) if total >= 5]
    forced_split_index = random.choice(split_candidates) if split_candidates else None
    forced_extra_single_index = random.randrange(kept_count) if random.random() < 0.75 else None

    patches: list[np.ndarray] = []
    occupied_cells = 0
    for idx, (color, total) in enumerate(zip(kept_colors, effective_totals)):
        part_areas = _choose_multi_parts(total, force_split=idx == forced_split_index)
        for area in part_areas:
            patch = _random_connected_patch(area, color)
            patches.append(patch)
            occupied_cells += int(np.count_nonzero(patch))

        extra_single_count = 0
        if idx == forced_extra_single_index:
            extra_single_count = 1
        elif random.random() < 0.35:
            extra_single_count = 1
        for _ in range(extra_single_count):
            patches.append(_single_pixel_patch(color))
            occupied_cells += 1

    for color in excluded_colors:
        single_count = random.randint(1, 2)
        for _ in range(single_count):
            patches.append(_single_pixel_patch(color))
            occupied_cells += 1

    min_side = max(12, math.ceil(math.sqrt(max(occupied_cells * 4, 1))))
    if min_side > 16:
        return None
    height = random.randint(min_side, 16)
    width = random.randint(min_side, 16)

    random.shuffle(patches)
    grid = _place_all((height, width), patches)
    if grid is None:
        return None
    if not _valid_generated_grid(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
