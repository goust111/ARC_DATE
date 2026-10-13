from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


BG = 0
TERRAIN = 5
PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]
MAX_GENERATION_ATTEMPTS = 400


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _most_common_color(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _component_coords(mask: np.ndarray) -> list[np.ndarray]:
    h, w = map(int, mask.shape)
    seen = np.zeros((h, w), dtype=bool)
    parts: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                cells.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            parts.append(np.array(cells, dtype=int))
    return parts


def _normalize_coords(coords: np.ndarray) -> np.ndarray:
    return coords - coords.min(axis=0, keepdims=True)


def _coords_to_mask(coords: np.ndarray) -> np.ndarray:
    norm = _normalize_coords(coords)
    h = int(norm[:, 0].max()) + 1
    w = int(norm[:, 1].max()) + 1
    mask = np.zeros((h, w), dtype=bool)
    mask[norm[:, 0], norm[:, 1]] = True
    return mask


def _infer_terrain_color(grid: np.ndarray, bg: int) -> int:
    candidates = sorted(set(map(int, grid[-1])) - {bg})
    assert len(candidates) == 1, "expected exactly one bottom-touching terrain color"
    return candidates[0]


def _terrain_heights(grid: np.ndarray, terrain: int) -> list[int]:
    h, w = map(int, grid.shape)
    heights: list[int] = []
    for c in range(w):
        rows = np.flatnonzero(grid[:, c] == terrain)
        if len(rows) == 0:
            heights.append(0)
            continue
        top = int(rows[0])
        assert np.array_equal(rows, np.arange(top, h)), "terrain must be bottom-supported per column"
        heights.append(h - top)
    return heights


def _water_mask_from_heights(heights: list[int], h: int) -> np.ndarray:
    w = len(heights)
    left_max = [0] * w
    right_max = [0] * w

    best = 0
    for c in range(w):
        best = max(best, int(heights[c]))
        left_max[c] = best

    best = 0
    for c in range(w - 1, -1, -1):
        best = max(best, int(heights[c]))
        right_max[c] = best

    water = np.zeros((h, w), dtype=bool)
    for c, ground in enumerate(heights):
        level = min(left_max[c], right_max[c])
        if level <= ground:
            continue
        r0 = h - level
        r1 = h - ground
        water[r0:r1, c] = True
    return water


def _basins(grid: np.ndarray, terrain: int) -> list[dict]:
    h = int(grid.shape[0])
    heights = _terrain_heights(grid, terrain)
    water = _water_mask_from_heights(heights, h)
    basins = []
    for coords in _component_coords(water):
        basins.append(
            {
                "coords": coords,
                "area": int(coords.shape[0]),
                "mask": _coords_to_mask(coords),
            }
        )
    return basins


def _blobs(grid: np.ndarray, bg: int, terrain: int) -> list[dict]:
    blobs = []
    for color in sorted(set(map(int, grid.ravel())) - {bg, terrain}):
        for coords in _component_coords(grid == color):
            blobs.append(
                {
                    "coords": coords,
                    "area": int(coords.shape[0]),
                    "color": int(color),
                    "mask": _coords_to_mask(coords),
                }
            )
    return blobs


def main(grid: np.ndarray) -> np.ndarray:
    """Treat each floating color blob as liquid, erase it, and pour it into the unique basin with matching capacity.

    Input structure:
    - A single bottom-touching terrain color forms a solid lower profile whose top contour creates multiple upward-open basins.
    - Several disconnected monochrome blobs float above the terrain on the background, with distinct colors and arbitrary shapes.
    - Basin capacities are the background cells that would be occupied by trapped liquid above the terrain profile.

    Rule:
    - Infer the terrain profile from the bottom-supported terrain columns and compute every trapped-water basin as a 4-connected water region.
    - Measure each source blob by area and each basin by capacity, then match them uniquely by area equality.
    - Erase every source blob from its original location and fill the matched basin completely with that blob's color.
    - Leave the terrain and every unmatched basin unchanged.

    Key insight:
    - The apparent shape match is a decoy: only conserved area matters, because each blob behaves like liquid and is rearranged by gravity to exactly occupy its matching basin.
    """
    g = np.asarray(grid, dtype=int)
    bg = _most_common_color(g)
    terrain = _infer_terrain_color(g, bg)

    basins = _basins(g, terrain)
    blobs = _blobs(g, bg, terrain)
    assert basins, "expected at least one basin"
    assert blobs, "expected at least one floating blob"

    area_to_basin = {basin["area"]: basin for basin in basins}
    assert len(area_to_basin) == len(basins), "basin capacities must be unique"

    blob_areas = [blob["area"] for blob in blobs]
    assert len(set(blob_areas)) == len(blob_areas), "blob areas must be unique"
    assert set(blob_areas) <= set(area_to_basin), "every blob must match a basin capacity"

    out = g.copy()
    for blob in blobs:
        out[blob["coords"][:, 0], blob["coords"][:, 1]] = bg

    for blob in blobs:
        basin = area_to_basin[blob["area"]]
        out[basin["coords"][:, 0], basin["coords"][:, 1]] = blob["color"]

    return out


def _profile_from_specs(rim_height: int, basin_specs: list[tuple[int, int]]) -> list[int]:
    heights = [rim_height]
    for width, depth in basin_specs:
        floor_height = rim_height - depth
        heights.extend([floor_height] * width)
        heights.append(rim_height)
    return heights


def _random_connected_mask(area: int, max_h: int, max_w: int) -> np.ndarray | None:
    if area > max_h * max_w:
        return None
    for _ in range(80):
        start_r = random.randrange(max_h)
        start_c = random.randrange(max_w)
        cells = {(start_r, start_c)}
        frontier = {
            (nr, nc)
            for nr, nc in _neighbors4(start_r, start_c)
            if 0 <= nr < max_h and 0 <= nc < max_w
        }
        while frontier and len(cells) < area:
            nr, nc = random.choice(tuple(frontier))
            frontier.remove((nr, nc))
            cells.add((nr, nc))
            for rr, cc in _neighbors4(nr, nc):
                if 0 <= rr < max_h and 0 <= cc < max_w and (rr, cc) not in cells:
                    frontier.add((rr, cc))
        if len(cells) != area:
            continue
        coords = np.array(sorted(cells), dtype=int)
        return _coords_to_mask(coords)
    return None


def _make_blob_mask(area: int, *, fit_h: int | None = None, fit_w: int | None = None) -> np.ndarray | None:
    if fit_h is not None and fit_w is not None:
        return _random_connected_mask(area, fit_h, fit_w)

    limits = []
    for h in range(1, min(6, area) + 1):
        for w in range(1, min(6, area) + 1):
            if h * w >= area:
                limits.append((h, w))
    random.shuffle(limits)
    for h, w in limits:
        mask = _random_connected_mask(area, h, w)
        if mask is not None:
            return mask
    return None


def _paint_terrain(grid: np.ndarray, heights: list[int], terrain: int) -> None:
    h = int(grid.shape[0])
    for c, height in enumerate(heights):
        if height > 0:
            grid[h - height :, c] = terrain


def _place_masks(
    grid_shape: tuple[int, int], masks: list[np.ndarray], top_limit: int
) -> list[tuple[int, int]] | None:
    _, w = grid_shape
    placed_boxes: list[tuple[int, int, int, int]] = []
    placements: list[tuple[int, int]] = []
    for mask in masks:
        h0, w0 = map(int, mask.shape)
        if h0 > top_limit or w0 > w:
            return None
        choices = [
            (r, c)
            for r in range(0, top_limit - h0 + 1)
            for c in range(0, w - w0 + 1)
        ]
        random.shuffle(choices)
        placed = None
        for r, c in choices:
            box = (r, c, r + h0 - 1, c + w0 - 1)
            ok = True
            for or0, oc0, or1, oc1 in placed_boxes:
                if not (box[2] + 1 < or0 or or1 + 1 < box[0] or box[3] + 1 < oc0 or oc1 + 1 < box[1]):
                    ok = False
                    break
            if ok:
                placed = (r, c)
                placed_boxes.append(box)
                placements.append(placed)
                break
        if placed is None:
            return None
    return placements


def _render_blob(grid: np.ndarray, mask: np.ndarray, origin: tuple[int, int], color: int) -> None:
    r0, c0 = origin
    h, w = map(int, mask.shape)
    patch = grid[r0 : r0 + h, c0 : c0 + w]
    patch[mask] = color


def _choose_basin_specs() -> tuple[int, list[tuple[int, int]], list[int]] | None:
    basin_count = random.randint(3, 5)
    rim_height = random.randint(5, 8)

    candidates = []
    for width in range(2, 5):
        for depth in range(1, min(5, rim_height)):
            area = width * depth
            candidates.append((width, depth, area))
    random.shuffle(candidates)

    specs: list[tuple[int, int]] = []
    seen_areas: set[int] = set()
    for width, depth, area in candidates:
        if area in seen_areas:
            continue
        specs.append((width, depth))
        seen_areas.add(area)
        if len(specs) == basin_count:
            break
    if len(specs) != basin_count:
        return None

    heights = _profile_from_specs(rim_height, specs)
    if len(heights) > 28:
        return None

    capacities = [width * depth for width, depth in specs]
    smallest = int(np.argmin(capacities))
    largest = int(np.argmax(capacities))
    if capacities[largest] <= capacities[smallest]:
        return None
    return rim_height, specs, [smallest, largest]


def _build_input() -> np.ndarray | None:
    picked = _choose_basin_specs()
    if picked is None:
        return None
    rim_height, basin_specs, key_indices = picked
    heights = _profile_from_specs(rim_height, basin_specs)
    width = len(heights)
    top_margin = random.randint(7, 11)
    height = rim_height + top_margin
    if height > 30:
        return None

    basins_meta = [
        {"width": width_i, "depth": depth_i, "area": width_i * depth_i}
        for width_i, depth_i in basin_specs
    ]
    smallest_idx, largest_idx = key_indices

    basin_count = len(basins_meta)
    fill_count = random.randint(2, basin_count - 1)
    remaining = [i for i in range(basin_count) if i not in {smallest_idx, largest_idx}]
    random.shuffle(remaining)
    filled = {smallest_idx}
    for idx in remaining[: fill_count - 1]:
        filled.add(idx)
    if largest_idx in filled:
        filled.remove(largest_idx)
    while len(filled) < fill_count:
        candidate = random.choice([i for i in range(basin_count) if i != largest_idx and i not in filled])
        filled.add(candidate)
    filled_indices = sorted(filled)

    colors = random.sample(PALETTE, len(filled_indices))
    blob_masks: list[np.ndarray] = []
    for basin_idx in filled_indices:
        area = basins_meta[basin_idx]["area"]
        if basin_idx == smallest_idx:
            wrong = basins_meta[largest_idx]
            mask = _make_blob_mask(area, fit_h=wrong["depth"], fit_w=wrong["width"])
        else:
            mask = _make_blob_mask(area)
        if mask is None:
            return None
        blob_masks.append(mask)

    top_limit = height - rim_height - 2
    placements = _place_masks((height, width), blob_masks, top_limit)
    if placements is None:
        return None

    grid = np.full((height, width), BG, dtype=int)
    _paint_terrain(grid, heights, TERRAIN)
    for mask, origin, color in zip(blob_masks, placements, colors):
        _render_blob(grid, mask, origin, color)

    try:
        solved = main(grid)
    except Exception:
        return None

    basin_areas = [meta["area"] for meta in basins_meta]
    if len(set(basin_areas)) != len(basin_areas):
        return None
    if len(_basins(grid, TERRAIN)) != basin_count:
        return None
    if not np.array_equal(main(grid), solved):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _build_input()
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input failed after {MAX_GENERATION_ATTEMPTS} attempts")
