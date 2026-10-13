from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
DIVIDER = 5
PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]
DIRS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
GEN_ATTEMPTS = 400
PANEL_H_RANGE = (9, 15)
PANEL_W_CHOICES = (7, 8, 9)
AREA_CHOICES = tuple(range(2, 9))
MAX_MASK_DIM = 5
MASK_ATTEMPTS = 80


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    return mask[np.ix_(rows, cols)]


def _mask_coords(mask: np.ndarray, r0: int = 0, c0: int = 0) -> np.ndarray:
    coords = np.argwhere(mask)
    return coords + np.array([int(r0), int(c0)])


def _find_components(grid: np.ndarray, bg: int) -> list[dict[str, object]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict[str, object]] = []

    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == bg or seen[r, c]:
                continue

            queue = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []

            while queue:
                rr, cc = queue.popleft()
                cells.append((rr, cc))
                for dr, dc in DIRS4:
                    nr = rr + dr
                    nc = cc + dc
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if seen[nr, nc] or int(grid[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    queue.append((nr, nc))

            coords = np.array(cells, dtype=int)
            comps.append({"color": color, "coords": coords, "area": int(len(cells))})

    return comps


def _panel_slices(grid: np.ndarray, bg: int) -> list[np.ndarray]:
    h, w = grid.shape
    candidates: list[tuple[int, int]] = []
    uniform_cols = [
        c for c in range(w) if np.all(grid[:, c] == grid[0, c]) and int(grid[0, c]) != bg
    ]

    for i, c1 in enumerate(uniform_cols):
        for c2 in uniform_cols[i + 1 :]:
            if int(grid[0, c1]) != int(grid[0, c2]):
                continue
            left_w = int(c1)
            mid_w = int(c2 - c1 - 1)
            right_w = int(w - c2 - 1)
            if left_w > 0 and left_w == mid_w == right_w:
                candidates.append((c1, c2))

    assert len(candidates) == 1, "expected exactly two equal-panel divider columns"
    c1, c2 = candidates[0]
    panel_w = c1
    panels = [grid[:, :panel_w], grid[:, c1 + 1 : c2], grid[:, c2 + 1 :]]
    assert all(panel.shape == (h, panel_w) for panel in panels)
    return panels


def main(grid: np.ndarray) -> np.ndarray:
    """
    Merge one size-ranked object from each of three equal panels into a single panel.

    Input structure:
    - Two full-height vertical divider columns split the grid into three equal-width panels.
    - Each panel has one shared background color and exactly three monochrome 4-connected foreground objects.
    - Within every panel, the three object areas are all different.

    Rule:
    - In each panel, sort the three objects by area from smallest to largest.
    - Select the rank-1 object from the left panel, the rank-2 object from the middle panel, and the rank-3 object from the right panel.
    - Create an output canvas with the size of one panel and place each selected object at the same panel-relative coordinates it had in its own panel.
    - Paste in left-to-right panel order, so later selected objects overwrite earlier ones on overlaps.

    Key insight:
    - The panel order itself gives the ranking rule: panel 1 contributes the smallest object, panel 2 the middle object, and panel 3 the largest object.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and min(g.shape) >= 3, "expected a non-empty 2D grid"

    bg = _most_common_color(g)
    panels = _panel_slices(g, bg)
    out = np.full_like(panels[0], bg)

    for panel_index, panel in enumerate(panels):
        objects = _find_components(panel, bg)
        assert len(objects) == 3, "each panel must contain exactly three objects"
        areas = [int(obj["area"]) for obj in objects]
        assert len(set(areas)) == 3, "object areas must be distinct within each panel"

        ranked = sorted(objects, key=lambda obj: int(obj["area"]))
        chosen = ranked[panel_index]
        coords = chosen["coords"]
        out[coords[:, 0], coords[:, 1]] = int(chosen["color"])

    return out


def _random_connected_mask(area: int, max_h: int, max_w: int) -> np.ndarray:
    dims = [
        (h, w)
        for h in range(1, max_h + 1)
        for w in range(1, max_w + 1)
        if area <= h * w <= min(max_h * max_w, area + 4)
    ]
    assert dims, "no feasible mask dimensions"

    for _ in range(MASK_ATTEMPTS):
        h, w = random.choice(dims)
        cells = {(random.randrange(h), random.randrange(w))}

        while len(cells) < area:
            frontier = []
            for r, c in cells:
                for dr, dc in DIRS4:
                    nr = r + dr
                    nc = c + dc
                    if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                        frontier.append((nr, nc))
            if not frontier:
                break
            cells.add(random.choice(frontier))

        if len(cells) != area:
            continue

        mask = np.zeros((h, w), dtype=bool)
        for r, c in cells:
            mask[r, c] = True

        mask = _crop_mask(mask)
        mask = np.rot90(mask, random.randrange(4))
        if random.random() < 0.5:
            mask = np.fliplr(mask)
        mask = _crop_mask(mask)
        if mask.shape[0] <= max_h and mask.shape[1] <= max_w:
            return mask

    raise RuntimeError("failed to generate a connected mask")


def _paint_mask(grid: np.ndarray, mask: np.ndarray, r0: int, c0: int, color: int) -> np.ndarray:
    coords = _mask_coords(mask, r0, c0)
    grid[coords[:, 0], coords[:, 1]] = int(color)
    return coords


def _occupied_from_coords(shape: tuple[int, int], coords_list: list[np.ndarray]) -> np.ndarray:
    occupied = np.zeros(shape, dtype=bool)
    for coords in coords_list:
        occupied[coords[:, 0], coords[:, 1]] = True
    return occupied


def _dilate8(mask: np.ndarray) -> np.ndarray:
    out = mask.copy()
    out[1:, :] |= mask[:-1, :]
    out[:-1, :] |= mask[1:, :]
    out[:, 1:] |= mask[:, :-1]
    out[:, :-1] |= mask[:, 1:]
    out[1:, 1:] |= mask[:-1, :-1]
    out[1:, :-1] |= mask[:-1, 1:]
    out[:-1, 1:] |= mask[1:, :-1]
    out[:-1, :-1] |= mask[1:, 1:]
    return out


def _feasible_origins(mask: np.ndarray, shape: tuple[int, int], blocked: np.ndarray) -> list[tuple[int, int]]:
    h, w = shape
    mh, mw = mask.shape
    positions: list[tuple[int, int]] = []
    for r0 in range(h - mh + 1):
        for c0 in range(w - mw + 1):
            if not np.any(blocked[r0 : r0 + mh, c0 : c0 + mw] & mask):
                positions.append((r0, c0))
    return positions


def _mask_pivot(mask: np.ndarray) -> tuple[int, int]:
    coords = np.argwhere(mask)
    order = np.lexsort((coords[:, 1], coords[:, 0]))
    r, c = coords[int(order[0])]
    return int(r), int(c)


def _selected_origins_overlap(masks: list[np.ndarray], shape: tuple[int, int]) -> list[tuple[int, int]] | None:
    h, w = shape
    r_min = 0
    r_max = h - 1
    c_min = 0
    c_max = w - 1
    pivots = [_mask_pivot(mask) for mask in masks]

    for mask, (pr, pc) in zip(masks, pivots):
        mh, mw = mask.shape
        r_min = max(r_min, pr)
        r_max = min(r_max, h - mh + pr)
        c_min = max(c_min, pc)
        c_max = min(c_max, w - mw + pc)

    if r_min > r_max or c_min > c_max:
        return None

    anchor_r = random.randint(r_min, r_max)
    anchor_c = random.randint(c_min, c_max)
    return [(anchor_r - pr, anchor_c - pc) for pr, pc in pivots]


def _selected_origins_disjoint(masks: list[np.ndarray], shape: tuple[int, int]) -> list[tuple[int, int]] | None:
    occupied = np.zeros(shape, dtype=bool)
    origins: list[tuple[int, int] | None] = [None, None, None]
    order = sorted(range(len(masks)), key=lambda i: int(masks[i].sum()), reverse=True)

    for idx in order:
        positions = _feasible_origins(masks[idx], shape, occupied)
        if not positions:
            return None
        r0, c0 = random.choice(positions)
        origins[idx] = (r0, c0)
        coords = _mask_coords(masks[idx], r0, c0)
        occupied[coords[:, 0], coords[:, 1]] = True

    assert all(origin is not None for origin in origins)
    return [(int(r0), int(c0)) for r0, c0 in origins if r0 is not None and c0 is not None]


def _place_panel_objects(
    panel_shape: tuple[int, int],
    masks: list[np.ndarray],
    areas: list[int],
    selected_index: int,
    colors: list[int],
    selected_origin: tuple[int, int],
) -> tuple[np.ndarray, dict[str, object]] | None:
    panel_h, panel_w = panel_shape
    selected_color = int(colors[selected_index])
    selected_mask = masks[selected_index]
    sr0, sc0 = selected_origin
    sh, sw = selected_mask.shape
    if not (0 <= sr0 and 0 <= sc0 and sr0 + sh <= panel_h and sc0 + sw <= panel_w):
        return None

    panel = np.full(panel_shape, BG, dtype=int)
    placed_coords: list[np.ndarray] = []
    selected_coords = _paint_mask(panel, selected_mask, sr0, sc0, selected_color)
    placed_coords.append(selected_coords)

    occupied = _occupied_from_coords(panel_shape, placed_coords)
    blocked = _dilate8(occupied)
    place_order = sorted(
        [i for i in range(3) if i != selected_index],
        key=lambda i: int(masks[i].sum()),
        reverse=True,
    )

    for idx in place_order:
        positions = _feasible_origins(masks[idx], panel_shape, blocked)
        if not positions:
            return None
        r0, c0 = random.choice(positions)
        coords = _paint_mask(panel, masks[idx], r0, c0, colors[idx])
        placed_coords.append(coords)
        occupied = _occupied_from_coords(panel_shape, placed_coords)
        blocked = _dilate8(occupied)

    comps = _find_components(panel, BG)
    if len(comps) != 3:
        return None
    found_areas = sorted(int(comp["area"]) for comp in comps)
    if found_areas != sorted(areas):
        return None

    objects = []
    for idx in range(3):
        coords = selected_coords if idx == selected_index else None
        objects.append(
            {
                "mask": masks[idx],
                "color": int(colors[idx]),
                "coords": coords,
                "area": int(areas[idx]),
            }
        )

    selected_area = int(areas[selected_index])
    selected_matches = [
        comp
        for comp in comps
        if int(comp["area"]) == selected_area and int(comp["color"]) == int(selected_color)
    ]
    if len(selected_matches) != 1:
        return None
    objects[selected_index]["coords"] = selected_matches[0]["coords"]
    return panel, {"objects": objects, "selected": objects[selected_index]}


def _render_expected(shape: tuple[int, int], selected: list[dict[str, object]]) -> np.ndarray:
    out = np.full(shape, BG, dtype=int)
    for obj in selected:
        coords = obj["coords"]
        out[coords[:, 0], coords[:, 1]] = int(obj["color"])
    return out


def _generate_case() -> tuple[np.ndarray, np.ndarray] | None:
    panel_h = random.randint(*PANEL_H_RANGE)
    panel_w = random.choice(PANEL_W_CHOICES)
    panel_shape = (panel_h, panel_w)
    overlap_mode = random.random() < 0.5

    selected_colors = random.sample(PALETTE, 3)
    area_sets = [sorted(random.sample(AREA_CHOICES, 3)) for _ in range(3)]
    selected_masks = [
        _random_connected_mask(
            area_sets[i][i],
            min(MAX_MASK_DIM, panel_h),
            min(MAX_MASK_DIM, panel_w),
        )
        for i in range(3)
    ]

    if overlap_mode:
        selected_origins = _selected_origins_overlap(selected_masks, panel_shape)
    else:
        selected_origins = _selected_origins_disjoint(selected_masks, panel_shape)
    if selected_origins is None:
        return None

    panels = []
    selected_objects = []
    for panel_index in range(3):
        areas = area_sets[panel_index]
        selected_area = areas[panel_index]
        panel_masks = [
            selected_masks[panel_index]
            if area_index == panel_index
            else _random_connected_mask(
                areas[area_index],
                min(MAX_MASK_DIM, panel_h),
                min(MAX_MASK_DIM, panel_w),
            )
            for area_index in range(3)
        ]
        other_colors = random.sample([c for c in PALETTE if c != selected_colors[panel_index]], 2)
        panel_colors = other_colors[:]
        panel_colors.insert(panel_index, selected_colors[panel_index])

        panel_result = _place_panel_objects(
            panel_shape=panel_shape,
            masks=panel_masks,
            areas=areas,
            selected_index=panel_index,
            colors=panel_colors,
            selected_origin=selected_origins[panel_index],
        )
        if panel_result is None:
            return None
        panel, meta = panel_result

        objects = sorted(_find_components(panel, BG), key=lambda comp: int(comp["area"]))
        chosen = objects[panel_index]
        if int(chosen["area"]) != selected_area or int(chosen["color"]) != selected_colors[panel_index]:
            return None

        panels.append(panel)
        selected_objects.append(
            {"coords": chosen["coords"].copy(), "color": int(chosen["color"])}
        )

    expected = _render_expected(panel_shape, selected_objects)

    grid = np.full((panel_h, 3 * panel_w + 2), BG, dtype=int)
    grid[:, panel_w] = DIVIDER
    grid[:, 2 * panel_w + 1] = DIVIDER
    grid[:, :panel_w] = panels[0]
    grid[:, panel_w + 1 : 2 * panel_w + 1] = panels[1]
    grid[:, 2 * panel_w + 2 :] = panels[2]

    if not np.array_equal(main(grid), expected):
        return None
    return grid, expected


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        case = _generate_case()
        if case is not None:
            return case[0]
    raise RuntimeError("failed to generate a valid input")
