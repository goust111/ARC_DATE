import random
from typing import Iterable

import numpy as np


BG = 0
PALETTE = tuple(range(1, 10))
MAX_GEN_ATTEMPTS = 400
_GEN_CALL_COUNT = 0


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(values[np.argmax(counts)])


def _iter_border_controls(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    controls = []
    border_mask = np.zeros_like(grid, dtype=bool)
    border_mask[0, :] = True
    border_mask[-1, :] = True
    border_mask[:, 0] = True
    border_mask[:, -1] = True

    for r, c in np.argwhere(border_mask & (grid != bg)):
        r = int(r)
        c = int(c)
        if r == 0 and 0 < c < w - 1:
            side = "top"
        elif r == h - 1 and 0 < c < w - 1:
            side = "bottom"
        elif c == 0 and 0 < r < h - 1:
            side = "left"
        elif c == w - 1 and 0 < r < h - 1:
            side = "right"
        else:
            raise AssertionError("control pixels must lie on a non-corner border cell")
        controls.append({"r": r, "c": c, "side": side, "color": int(grid[r, c])})
    assert controls, "expected border control pixels"
    return controls


def _extract_layout(grid: np.ndarray) -> tuple[str, list[dict]]:
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    controls = _iter_border_controls(g, bg)
    colors = [ctrl["color"] for ctrl in controls]
    assert len(colors) == len(set(colors)), "control colors must be distinct"

    sides = {ctrl["side"] for ctrl in controls}
    if sides <= {"top", "bottom"}:
        orientation = "vertical"
        axis_key = "c"
    elif sides <= {"left", "right"}:
        orientation = "horizontal"
        axis_key = "r"
    else:
        raise AssertionError("all control pixels must launch parallel rays")

    rays = []
    used_axes = set()
    for ctrl in controls:
        axis = int(ctrl[axis_key])
        assert axis not in used_axes, "rays may not share a row or column"
        used_axes.add(axis)
        rays.append({"axis": axis, "color": int(ctrl["color"]), "side": ctrl["side"]})
    rays.sort(key=lambda ray: ray["axis"])
    return orientation, rays


def _region_geometry(shape: tuple[int, int], orientation: str, rays: list[dict]) -> list[dict]:
    h, w = shape
    limit = w if orientation == "vertical" else h
    axes = [ray["axis"] for ray in rays]
    bounds = [-1, *axes, limit]
    regions = []
    scale = h if orientation == "vertical" else w

    for idx in range(len(bounds) - 1):
        start = int(bounds[idx] + 1)
        end = int(bounds[idx + 1] - 1)
        width = int(end - start + 1)
        assert width > 0, "every partition region must have positive area"
        area = int(width * scale)
        regions.append(
            {
                "index": idx,
                "span": (start, end),
                "area": area,
            }
        )

    areas = [region["area"] for region in regions]
    assert len(areas) == len(set(areas)), "partition areas must be pairwise distinct"
    return regions


def _ordered_candidate_colors(
    region: dict,
    rays: list[dict],
    prefer_negative: bool,
) -> list[int]:
    start, end = region["span"]
    center2 = start + end
    negative_side = []
    positive_side = []

    for ray in rays:
        axis = int(ray["axis"])
        distance = abs(center2 - 2 * axis)
        item = (distance, axis, int(ray["color"]))
        if axis < start:
            negative_side.append(item)
        elif axis > end:
            positive_side.append(item)
        else:
            raise AssertionError("regions exclude the ray cells themselves")

    negative_side.sort(key=lambda item: (item[0], -item[1]))
    positive_side.sort(key=lambda item: (item[0], item[1]))
    ordered = negative_side + positive_side if prefer_negative else positive_side + negative_side
    return [color for _, _, color in ordered]


def _layer_colors(shape: tuple[int, int], orientation: str, rays: list[dict]) -> list[int]:
    regions = sorted(
        _region_geometry(shape, orientation, rays),
        key=lambda region: region["area"],
        reverse=True,
    )
    used_colors = set()
    layer_colors = []

    for rank, region in enumerate(regions):
        prefer_negative = (rank % 2 == 0)
        candidates = _ordered_candidate_colors(region, rays, prefer_negative)
        chosen = None
        for color in candidates:
            if color not in used_colors:
                chosen = color
                break
        if chosen is None:
            chosen = candidates[0]
        used_colors.add(chosen)
        layer_colors.append(int(chosen))
    return layer_colors


def _fill_concentric(layer_colors: Iterable[int]) -> np.ndarray:
    colors = list(layer_colors)
    side = 2 * len(colors) - 1
    out = np.full((side, side), BG, dtype=int)
    for layer, color in enumerate(colors):
        out[layer : side - layer, layer : side - layer] = int(color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Convert border-launched partition rays into an area-ranked concentric square.

    Input structure:
    - The background is black, and the only foreground pixels are distinct-colored control pixels on the grid border.
    - Every control pixel lies on a non-corner edge cell and launches a full straight ray perpendicular to that edge.
    - All valid control pixels launch parallel rays, so the implicit partition consists of non-overlapping rectangular strips.

    Rule:
    - Extend every border control pixel across the whole grid along its launch direction, treating those ray cells as cut lines.
    - Remove the ray cells, measure the area of every rectangular region between consecutive rays and the outer border, and sort the regions by decreasing area.
    - For the largest, third-largest, fifth-largest, ... region choose the nearest unused ray color on its upper/left side when possible; for the second-largest, fourth-largest, sixth-largest, ... region do the same on its lower/right side, falling back to the opposite side and finally to the nearest ray if every color has already been used once.
    - Build a square with one visible concentric layer per region, so the ranked region colors fill the output from the outside inward.

    Key insight:
    - The border pixels are only ray launchers; their positions determine strip areas, and those area ranks in turn determine the outside-to-inside color order of the final concentric square.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    orientation, rays = _extract_layout(g)
    layer_colors = _layer_colors(g.shape, orientation, rays)
    return _fill_concentric(layer_colors)


def _widths_for_total(num_regions: int, total: int, rng: random.Random) -> list[int]:
    widths = list(range(1, num_regions + 1))
    leftover = total - sum(widths)
    assert leftover >= 0
    base_add, remainder = divmod(leftover, num_regions)
    widths = [width + base_add for width in widths]
    for i in range(remainder):
        widths[num_regions - 1 - i] += 1

    for _ in range(80):
        shuffled = widths[:]
        rng.shuffle(shuffled)
        if shuffled != sorted(shuffled) and shuffled != sorted(shuffled, reverse=True):
            return shuffled
    return widths


def _ray_axes_from_widths(widths: list[int]) -> list[int]:
    cursor = 0
    axes = []
    for width in widths[:-1]:
        cursor += int(width)
        axes.append(cursor)
        cursor += 1
    return axes


def _independent_expected(meta: dict) -> np.ndarray:
    orientation = meta["orientation"]
    rays = [{"axis": axis, "color": color} for axis, color in zip(meta["axes"], meta["colors"])]
    scale = meta["shape"][0] if orientation == "vertical" else meta["shape"][1]
    areas = [int(width * scale) for width in meta["widths"]]
    region_order = sorted(range(len(areas)), key=lambda idx: areas[idx], reverse=True)

    layer_colors = []
    used = set()
    for rank, region_idx in enumerate(region_order):
        start = sum(meta["widths"][:region_idx]) + region_idx
        end = start + meta["widths"][region_idx] - 1
        center2 = start + end

        negative = []
        positive = []
        for axis, color in zip(meta["axes"], meta["colors"]):
            item = (abs(center2 - 2 * axis), axis, color)
            if axis < start:
                negative.append(item)
            elif axis > end:
                positive.append(item)
        negative.sort(key=lambda item: (item[0], -item[1]))
        positive.sort(key=lambda item: (item[0], item[1]))
        ordered = negative + positive if rank % 2 == 0 else positive + negative
        ordered_colors = [item[2] for item in ordered]

        chosen = None
        for color in ordered_colors:
            if color not in used:
                chosen = color
                break
        if chosen is None:
            chosen = ordered_colors[0]
        used.add(chosen)
        layer_colors.append(int(chosen))

    return _fill_concentric(layer_colors)


def _build_attempt(rng: random.Random, call_index: int) -> tuple[np.ndarray, dict]:
    orientation = "vertical" if call_index % 2 == 0 else "horizontal"
    num_rays = 3 + (call_index % 3)
    num_regions = num_rays + 1

    min_axis = num_rays + num_regions * (num_regions + 1) // 2
    max_axis = min(30, min_axis + 7)
    axis_len = rng.randint(min_axis, max_axis)
    total_region_width = axis_len - num_rays
    widths = _widths_for_total(num_regions, total_region_width, rng)
    axes = _ray_axes_from_widths(widths)
    colors = rng.sample(PALETTE, num_rays)

    other_len = rng.randint(12, 20)
    if orientation == "vertical":
        h, w = other_len, axis_len
    else:
        h, w = axis_len, other_len

    grid = np.full((h, w), BG, dtype=int)
    forced_mixed_sides = (call_index % 4) < 2
    for idx, (axis, color) in enumerate(zip(axes, colors)):
        if orientation == "vertical":
            if forced_mixed_sides and idx == 0:
                row = 0
            elif forced_mixed_sides and idx == 1:
                row = h - 1
            else:
                row = 0 if rng.random() < 0.5 else h - 1
            grid[row, axis] = int(color)
        else:
            if forced_mixed_sides and idx == 0:
                col = 0
            elif forced_mixed_sides and idx == 1:
                col = w - 1
            else:
                col = 0 if rng.random() < 0.5 else w - 1
            grid[axis, col] = int(color)

    meta = {
        "orientation": orientation,
        "shape": (h, w),
        "widths": widths,
        "axes": axes,
        "colors": colors,
    }
    return grid, meta


def _validate_generated(grid: np.ndarray, meta: dict) -> bool:
    bg = _get_bg(grid)
    assert bg == BG
    controls = _iter_border_controls(grid, bg)
    assert len(controls) == len(meta["colors"])
    assert np.count_nonzero(grid[1:-1, 1:-1]) == 0

    orientation, rays = _extract_layout(grid)
    assert orientation == meta["orientation"]
    assert [ray["axis"] for ray in rays] == meta["axes"]
    assert [ray["color"] for ray in rays] == meta["colors"]

    regions = _region_geometry(grid.shape, orientation, rays)
    widths = [region["span"][1] - region["span"][0] + 1 for region in regions]
    assert widths == meta["widths"]
    assert widths != sorted(widths)
    assert widths != sorted(widths, reverse=True)

    out = main(grid)
    expected = _independent_expected(meta)
    assert np.array_equal(out, expected)
    assert out.shape == (2 * len(widths) - 1, 2 * len(widths) - 1)
    assert max(out.shape) <= 30
    assert len(np.unique(out)) >= 3
    return True


def generate_input() -> np.ndarray:
    global _GEN_CALL_COUNT
    for _ in range(MAX_GEN_ATTEMPTS):
        rng = random.Random(random.randrange(1 << 30) ^ (_GEN_CALL_COUNT << 8))
        grid, meta = _build_attempt(rng, _GEN_CALL_COUNT)
        try:
            if _validate_generated(grid, meta):
                _GEN_CALL_COUNT += 1
                return grid
        except AssertionError:
            pass
    raise RuntimeError("failed to generate a valid ray-partition input")
