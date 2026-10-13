from __future__ import annotations

from collections import Counter
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import random

import numpy as np


_ARC_UTILS_PATH = Path("/home/lixinyu/snap/new/solvers/arc_utils.py")
_ARC_UTILS_SPEC = spec_from_file_location("_arc_utils_shared", _ARC_UTILS_PATH)
assert _ARC_UTILS_SPEC is not None and _ARC_UTILS_SPEC.loader is not None
_ARC_UTILS = module_from_spec(_ARC_UTILS_SPEC)
_ARC_UTILS_SPEC.loader.exec_module(_ARC_UTILS)

bbox_coords = _ARC_UTILS.bbox_coords
components_mask = _ARC_UTILS.components_mask
paste = _ARC_UTILS.paste


BG = 0
SEPARATOR = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
SHIFT_MODES = ("horizontal", "vertical", "diagonal")
GEN_RETRIES = 400


def _neighbors4(shape: tuple[int, int], row: int, col: int) -> list[tuple[int, int]]:
    height, width = map(int, shape)
    out: list[tuple[int, int]] = []
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        rr = int(row) + dr
        cc = int(col) + dc
        if 0 <= rr < height and 0 <= cc < width:
            out.append((rr, cc))
    return out


def _separator_row(grid: np.ndarray) -> int:
    height, width = map(int, grid.shape)
    assert height % 2 == 1 and width > 0, "expected two equal regions separated by one row"
    sep = height // 2
    assert np.all(grid[sep, :] == grid[sep, 0]), "expected a full separator row"
    return int(sep)


def _region_background(top: np.ndarray, bottom: np.ndarray) -> int:
    values = np.concatenate([top.reshape(-1), bottom.reshape(-1)])
    return int(Counter(map(int, values)).most_common(1)[0][0])


def _foreground_data(region: np.ndarray, bg: int) -> tuple[np.ndarray, np.ndarray]:
    mask = region != int(bg)
    coords = np.argwhere(mask)
    values = region[mask]
    return coords, values


def _candidate_offsets(top: np.ndarray, bottom: np.ndarray, bg: int) -> list[tuple[int, int, int]]:
    height, width = map(int, top.shape)
    top = np.asarray(top, dtype=int)
    bottom = np.asarray(bottom, dtype=int)
    bottom_coords, bottom_values = _foreground_data(bottom, bg)
    assert len(bottom_coords) > 0, "lower region must contain foreground"

    br0, bc0, br1, bc1 = bbox_coords(bottom_coords)
    dr_min = -int(br0)
    dr_max = height - 1 - int(br1)
    dc_min = -int(bc0)
    dc_max = width - 1 - int(bc1)

    candidates: list[tuple[int, int, int]] = []
    for dr in range(dr_min, dr_max + 1):
        shifted_rows = bottom_coords[:, 0] + int(dr)
        for dc in range(dc_min, dc_max + 1):
            shifted_cols = bottom_coords[:, 1] + int(dc)
            top_values = top[shifted_rows, shifted_cols]
            overlap = top_values != int(bg)
            if not bool(np.any(overlap)):
                continue
            if np.any(top_values[overlap] != bottom_values[overlap]):
                continue
            candidates.append((int(dr), int(dc), int(overlap.sum())))
    return candidates


def _find_unique_offset(top: np.ndarray, bottom: np.ndarray, bg: int) -> tuple[int, int]:
    candidates = _candidate_offsets(top, bottom, bg)
    assert len(candidates) == 1, "expected one conflict-free registration offset"
    dr, dc, _overlap = candidates[0]
    return int(dr), int(dc)


def _compose_regions(top: np.ndarray, bottom: np.ndarray, bg: int, offset: tuple[int, int]) -> np.ndarray:
    dr, dc = map(int, offset)
    out = np.asarray(top, dtype=int).copy()
    paste(out, np.asarray(bottom, dtype=int), dr, dc, transparent=int(bg), clip=True)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Align the misregistered lower partial sprite onto the upper partial sprite and merge them into one completed view.

    Input structure:
    - A full grey separator row splits the grid into two equal-sized regions.
    - Each region contains one partial copy of the same multicolor sprite on a common background.
    - The two partial copies sit at different positions inside their regions and hide different cells.

    Rule:
    - Search for the unique translation that moves the lower-region copy into the upper-region coordinates without any color conflicts on overlapping colored cells.
    - Overlay the shifted lower copy on top of the upper copy, keeping every upper-region colored cell fixed and filling the cells that are missing there.
    - Return only the completed upper-region-sized view.

    Key insight:
    - The separator is a distraction: the two halves are not pixel-aligned, so the task is to recover the one conflict-free registration offset before taking the union.
    """
    g = np.asarray(grid, dtype=int)
    sep = _separator_row(g)
    top = g[:sep, :]
    bottom = g[sep + 1 :, :]
    assert top.shape == bottom.shape, "regions must have the same size"

    bg = _region_background(top, bottom)
    offset = _find_unique_offset(top, bottom, bg)
    return _compose_regions(top, bottom, bg, offset)


def _random_connected_mask(height: int, width: int, target_area: int) -> np.ndarray:
    mask = np.zeros((int(height), int(width)), dtype=bool)
    start = (random.randrange(height), random.randrange(width))
    mask[start] = True
    frontier = set(_neighbors4(mask.shape, start[0], start[1]))

    while int(mask.sum()) < int(target_area):
        if not frontier:
            return np.zeros((0, 0), dtype=bool)
        cell = random.choice(tuple(frontier))
        frontier.remove(cell)
        mask[cell] = True
        for nxt in _neighbors4(mask.shape, cell[0], cell[1]):
            if not mask[nxt]:
                frontier.add(nxt)

    coords = np.argwhere(mask)
    r0, c0, r1, c1 = bbox_coords(coords)
    return mask[r0 : r1 + 1, c0 : c1 + 1]


def _colorize_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    n_cells = len(coords)
    n_colors = random.randint(2, min(4, n_cells))
    colors = random.sample(list(PALETTE), n_colors)
    seed_ids = random.sample(range(n_cells), n_colors)
    seeds = coords[seed_ids]

    sprite = np.full(mask.shape, BG, dtype=int)
    for r, c in coords:
        distances = np.abs(seeds[:, 0] - int(r)) + np.abs(seeds[:, 1] - int(c))
        sprite[int(r), int(c)] = int(colors[int(np.argmin(distances))])

    for _ in range(random.randint(1, min(3, n_cells))):
        rr, cc = map(int, random.choice(coords.tolist()))
        sprite[rr, cc] = int(random.choice(colors))

    assert len(set(map(int, sprite[mask].tolist()))) >= 2
    return sprite


def _visibility_split(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    coords = np.argwhere(mask)
    if len(coords) < 6:
        return None

    focus_top = coords[random.randrange(len(coords))]
    focus_bottom = coords[random.randrange(len(coords))]
    if len(coords) > 1 and np.array_equal(focus_top, focus_bottom):
        focus_bottom = coords[(random.randrange(len(coords) - 1) + 1) % len(coords)]

    show_top = np.zeros_like(mask, dtype=bool)
    show_bottom = np.zeros_like(mask, dtype=bool)
    top_only = 0
    bottom_only = 0
    both = 0

    for r, c in coords:
        da = abs(int(r) - int(focus_top[0])) + abs(int(c) - int(focus_top[1]))
        db = abs(int(r) - int(focus_bottom[0])) + abs(int(c) - int(focus_bottom[1]))
        delta = da - db + random.randint(-1, 1)
        if delta <= -2:
            show_top[int(r), int(c)] = True
            top_only += 1
        elif delta >= 2:
            show_bottom[int(r), int(c)] = True
            bottom_only += 1
        else:
            show_top[int(r), int(c)] = True
            show_bottom[int(r), int(c)] = True
            both += 1

    top_count = int(show_top.sum())
    bottom_count = int(show_bottom.sum())
    if min(top_only, bottom_only, both) == 0:
        return None
    if min(top_count, bottom_count) * 10 < len(coords) * 4:
        return None
    return show_top, show_bottom


def _partial_from_visibility(sprite: np.ndarray, visible: np.ndarray) -> np.ndarray:
    part = np.full(sprite.shape, BG, dtype=int)
    part[visible] = sprite[visible]
    return part


def _positions_for_mode(
    region_h: int,
    region_w: int,
    sprite_h: int,
    sprite_w: int,
    top_anchor: tuple[int, int],
    mode: str,
) -> list[tuple[int, int]]:
    top_r, top_c = map(int, top_anchor)
    out: list[tuple[int, int]] = []
    for row in range(region_h - sprite_h + 1):
        for col in range(region_w - sprite_w + 1):
            if (row, col) == (top_r, top_c):
                continue
            if mode == "horizontal" and row == top_r and col != top_c:
                out.append((row, col))
            elif mode == "vertical" and col == top_c and row != top_r:
                out.append((row, col))
            elif mode == "diagonal" and row != top_r and col != top_c:
                out.append((row, col))
    random.shuffle(out)
    return out


def _expected_region(region_h: int, region_w: int, sprite: np.ndarray, anchor: tuple[int, int]) -> np.ndarray:
    out = np.full((int(region_h), int(region_w)), BG, dtype=int)
    paste(out, sprite, int(anchor[0]), int(anchor[1]), transparent=BG)
    return out


def _build_case() -> np.ndarray | None:
    region_h = random.randint(8, 14)
    region_w = random.randint(10, 18)

    mask_h = random.randint(4, min(9, region_h - 2))
    mask_w = random.randint(4, min(10, region_w - 2))
    max_area = min(mask_h * mask_w - 1, int(mask_h * mask_w * 0.72))
    min_area = max(8, int(mask_h * mask_w * 0.42))
    if min_area > max_area:
        return None

    mask = _random_connected_mask(mask_h, mask_w, random.randint(min_area, max_area))
    if mask.size == 0:
        return None
    sprite_h, sprite_w = map(int, mask.shape)
    if sprite_h < 3 or sprite_w < 3:
        return None
    if region_h - sprite_h < 1 or region_w - sprite_w < 1:
        return None
    if np.all(mask):
        return None

    sprite = _colorize_mask(mask)
    visible = _visibility_split(mask)
    if visible is None:
        return None
    top_visible, bottom_visible = visible

    top_part = _partial_from_visibility(sprite, top_visible)
    bottom_part = _partial_from_visibility(sprite, bottom_visible)
    if len(components_mask(top_part != BG, connectivity=4)) > 3:
        return None
    if len(components_mask(bottom_part != BG, connectivity=4)) > 3:
        return None

    top_anchor = (
        random.randint(0, region_h - sprite_h),
        random.randint(0, region_w - sprite_w),
    )
    mode_order = list(SHIFT_MODES)
    random.shuffle(mode_order)
    bottom_anchor: tuple[int, int] | None = None
    for mode in mode_order:
        choices = _positions_for_mode(region_h, region_w, sprite_h, sprite_w, top_anchor, mode)
        if choices:
            bottom_anchor = choices[0]
            break
    if bottom_anchor is None:
        return None

    top_region = np.full((region_h, region_w), BG, dtype=int)
    bottom_region = np.full((region_h, region_w), BG, dtype=int)
    paste(top_region, top_part, int(top_anchor[0]), int(top_anchor[1]), transparent=BG)
    paste(bottom_region, bottom_part, int(bottom_anchor[0]), int(bottom_anchor[1]), transparent=BG)

    input_grid = np.full((2 * region_h + 1, region_w), BG, dtype=int)
    input_grid[:region_h, :] = top_region
    input_grid[region_h, :] = SEPARATOR
    input_grid[region_h + 1 :, :] = bottom_region

    intended = (int(top_anchor[0] - bottom_anchor[0]), int(top_anchor[1] - bottom_anchor[1]))
    candidates = _candidate_offsets(top_region, bottom_region, BG)
    if len(candidates) != 1 or candidates[0][:2] != intended:
        return None

    expected = _expected_region(region_h, region_w, sprite, top_anchor)
    solved = main(input_grid)
    if not np.array_equal(solved, expected):
        return None
    if np.array_equal(solved, top_region):
        return None
    shifted_bottom_only = _compose_regions(np.full_like(top_region, BG), bottom_region, BG, intended)
    if np.array_equal(solved, shifted_bottom_only):
        return None
    return input_grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        case = _build_case()
        if case is not None:
            return case
    raise RuntimeError("failed to generate a valid registration puzzle")
