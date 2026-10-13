from __future__ import annotations

from collections import deque
import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREY = 5
ORANGE = 7

GRID_MIN = 15
GRID_MAX = 18
WALL_ANCHOR_MIN = 5
MIN_REGION_AREA = 45
MAX_GENERATION_ATTEMPTS = 500


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _neighbors8(r: int, c: int):
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            yield r + dr, c + dc


def _label_mask(mask: np.ndarray) -> tuple[np.ndarray, int]:
    h, w = mask.shape
    labels = np.zeros((h, w), dtype=int)
    label = 0
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or labels[r, c] != 0:
                continue
            label += 1
            q = deque([(r, c)])
            labels[r, c] = label
            while q:
                rr, cc = q.popleft()
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and labels[nr, nc] == 0:
                        labels[nr, nc] = label
                        q.append((nr, nc))
    return labels, label


def _flood_from(passable: np.ndarray, start: tuple[int, int]) -> np.ndarray:
    h, w = passable.shape
    sr, sc = start
    seen = np.zeros((h, w), dtype=bool)
    q = deque([(sr, sc)])
    seen[sr, sc] = True
    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and passable[nr, nc] and not seen[nr, nc]:
                seen[nr, nc] = True
                q.append((nr, nc))
    return seen


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(mask)
    assert coords.size > 0
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _minimal_axis_period(tile: np.ndarray, axis: int) -> int:
    size = tile.shape[axis]
    for period in range(1, size):
        ok = True
        for idx in range(size):
            sl_a = np.take(tile, idx, axis=axis)
            sl_b = np.take(tile, idx % period, axis=axis)
            if not np.array_equal(sl_a, sl_b):
                ok = False
                break
        if ok:
            return period
    return size


def _tile_signature(tile: np.ndarray) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(int(v) for v in row) for row in tile.astype(int))


def _tile_is_valid(tile: np.ndarray) -> bool:
    h, w = tile.shape
    filled = int(tile.sum())
    if filled < 2 or filled == h * w:
        return False
    if not tile[0].any() or not tile[-1].any() or not tile[:, 0].any() or not tile[:, -1].any():
        return False
    if _minimal_axis_period(tile, axis=0) != h:
        return False
    if _minimal_axis_period(tile, axis=1) != w:
        return False
    return True


def _sample_tile(rng: random.Random, shape: tuple[int, int], forbidden: tuple[tuple[int, ...], ...] | None) -> np.ndarray:
    h, w = shape
    cells = [(r, c) for r in range(h) for c in range(w)]
    for _ in range(200):
        count = rng.randint(2, h * w - 1)
        chosen = set(rng.sample(cells, count))
        tile = np.zeros((h, w), dtype=bool)
        for r, c in chosen:
            tile[r, c] = True
        sig = _tile_signature(tile)
        if forbidden is not None and sig == forbidden:
            continue
        if _tile_is_valid(tile):
            return tile
    raise RuntimeError("failed to sample a non-degenerate seed tile")


def _make_vertical_wall(height: int, width: int, rng: random.Random) -> np.ndarray:
    for _ in range(300):
        anchor_count = rng.randint(3, 5)
        mid_rows = sorted(rng.sample(range(2, height - 2), anchor_count - 2))
        rows = [0, *mid_rows, height - 1]

        direction = rng.choice([-1, 1])
        start_col = rng.randint(WALL_ANCHOR_MIN, width - WALL_ANCHOR_MIN - 1)
        cols = [start_col]
        for _seg in range(anchor_count - 1):
            cols.append(cols[-1] + direction * rng.choice([0, 1, 1, 2]))

        if len(set(cols)) < 3:
            continue
        if min(cols) < WALL_ANCHOR_MIN or max(cols) > width - WALL_ANCHOR_MIN - 1:
            continue

        wall = np.zeros((height, width), dtype=bool)
        cur_r, cur_c = rows[0], cols[0]
        wall[cur_r, cur_c] = True
        for next_r, next_c in zip(rows[1:], cols[1:]):
            for r in range(cur_r + 1, next_r + 1):
                wall[r, cur_c] = True
            if next_c != cur_c:
                step = 1 if next_c > cur_c else -1
                for c in range(cur_c, next_c + step, step):
                    wall[next_r, c] = True
            cur_r, cur_c = next_r, next_c

        labels, n_regions = _label_mask(~wall)
        if n_regions != 2:
            continue
        areas = [int(np.sum(labels == idx)) for idx in range(1, n_regions + 1)]
        if min(areas) < MIN_REGION_AREA:
            continue
        return wall
    raise RuntimeError("failed to build a two-region wall")


def _make_wall(height: int, width: int, rng: random.Random) -> np.ndarray:
    if rng.random() < 0.5:
        return _make_vertical_wall(height, width, rng)
    return _make_vertical_wall(width, height, rng).T


def _candidate_seed_positions(region_mask: np.ndarray, tile_shape: tuple[int, int]) -> list[tuple[int, int]]:
    h, w = region_mask.shape
    tile_h, tile_w = tile_shape
    r0_box, c0_box, _r1_box, _c1_box = _bbox(region_mask)
    positions: list[tuple[int, int]] = []
    for r0 in range(1, h - tile_h):
        for c0 in range(1, w - tile_w):
            if not region_mask[r0 : r0 + tile_h, c0 : c0 + tile_w].all():
                continue
            if not region_mask[r0 - 1 : r0 + tile_h + 1, c0 - 1 : c0 + tile_w + 1].all():
                continue
            global_misaligned = (r0 % tile_h != 0) or (c0 % tile_w != 0)
            bbox_misaligned = ((r0 - r0_box) % tile_h != 0) or ((c0 - c0_box) % tile_w != 0)
            if global_misaligned and bbox_misaligned:
                positions.append((r0, c0))
    return positions


def _indicator_candidates(region_mask: np.ndarray, occupied: np.ndarray) -> list[tuple[int, int]]:
    h, w = region_mask.shape
    out: list[tuple[int, int]] = []
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            if not region_mask[r, c] or occupied[r, c]:
                continue
            ok = True
            for nr, nc in _neighbors8(r, c):
                if not region_mask[nr, nc] or occupied[nr, nc]:
                    ok = False
                    break
            if ok:
                out.append((r, c))
    return out


def _paint_seed(grid: np.ndarray, tile: np.ndarray, top_left: tuple[int, int], color: int) -> None:
    r0, c0 = top_left
    h, w = tile.shape
    for r in range(h):
        for c in range(w):
            if tile[r, c]:
                grid[r0 + r, c0 + c] = color


def _fill_region(out: np.ndarray, region_mask: np.ndarray, tile: np.ndarray, top_left: tuple[int, int], color: int) -> None:
    tile_h, tile_w = tile.shape
    anchor_r, anchor_c = top_left
    coords = np.argwhere(region_mask)
    for r, c in coords:
        rr = (int(r) - anchor_r) % tile_h
        cc = (int(c) - anchor_c) % tile_w
        out[int(r), int(c)] = color if tile[rr, cc] else BLACK


def _solve_from_layout(
    wall_mask: np.ndarray,
    indicator: tuple[int, int],
    blue_tile: np.ndarray,
    blue_top_left: tuple[int, int],
    red_tile: np.ndarray,
    red_top_left: tuple[int, int],
) -> np.ndarray:
    passable = ~wall_mask
    inside = _flood_from(passable, indicator)
    outside = passable & ~inside

    out = np.zeros(wall_mask.shape, dtype=int)
    out[wall_mask] = GREY
    _fill_region(out, inside, blue_tile, blue_top_left, BLUE)
    _fill_region(out, outside, red_tile, red_top_left, RED)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Flood-partition the grid by the grey wall and tile the two resulting regions from their seed masks.

    Input structure:
    - The grid has a black background with one connected grey wall running from one side of the canvas to the opposite side.
    - A single orange indicator pixel lies in one non-wall 4-connected region, while a blue seed mask lies in that same region and a red seed mask lies in the other region.
    - Each seed is a sparse 2x2 or 2x3 rectangular mask whose bounding box is fully contained in its region.

    Rule:
    - Flood-fill 4-connectedly from the orange indicator, treating grey as blocking, to obtain the indicator-side region; the other non-wall cells form the second region.
    - Take the blue pixels inside the indicator region, crop their bounding box, and use that boolean mask as a periodic tile for the whole indicator region; do the analogous operation with the red pixels in the other region.
    - Preserve each seed's absolute phase: cell (r, c) uses the tile position ((r - seed_r0) mod tile_h, (c - seed_c0) mod tile_w), grey wall cells stay grey, and the orange indicator is overwritten by the tiling.

    Key insight:
    - The bent wall is both the region separator and the irregular clipping boundary for two different periodic fills, so the output depends on combining flood-based region assignment with phase-preserving seed tiling.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    indicator_cells = np.argwhere(g == ORANGE)
    assert indicator_cells.shape[0] == 1
    indicator = tuple(int(v) for v in indicator_cells[0])

    wall_mask = g == GREY
    labels, n_regions = _label_mask(~wall_mask)
    assert n_regions == 2

    inside = _flood_from(~wall_mask, indicator)
    outside = (~wall_mask) & ~inside
    assert inside[indicator]
    assert outside.any()

    blue_mask = (g == BLUE) & inside
    red_mask = (g == RED) & outside
    assert blue_mask.any() and red_mask.any()
    assert not ((g == BLUE) & outside).any()
    assert not ((g == RED) & inside).any()

    br0, bc0, br1, bc1 = _bbox(blue_mask)
    rr0, rc0, rr1, rc1 = _bbox(red_mask)
    blue_tile = blue_mask[br0 : br1 + 1, bc0 : bc1 + 1]
    red_tile = red_mask[rr0 : rr1 + 1, rc0 : rc1 + 1]
    assert _tile_is_valid(blue_tile)
    assert _tile_is_valid(red_tile)

    return _solve_from_layout(
        wall_mask,
        indicator,
        blue_tile,
        (br0, bc0),
        red_tile,
        (rr0, rc0),
    )


def _generate_attempt(rng: random.Random) -> np.ndarray | None:
    height = rng.randint(GRID_MIN, GRID_MAX)
    width = rng.randint(GRID_MIN, GRID_MAX)
    wall_mask = _make_wall(height, width, rng)

    region_labels, _ = _label_mask(~wall_mask)
    region_masks = [region_labels == 1, region_labels == 2]
    if min(int(mask.sum()) for mask in region_masks) < MIN_REGION_AREA:
        return None

    inner_idx = rng.randrange(2)
    inside_mask = region_masks[inner_idx]
    outside_mask = region_masks[1 - inner_idx]

    blue_shape = rng.choice([(2, 2), (2, 3)])
    red_shape = rng.choice([(2, 2), (2, 3)])
    blue_tile = _sample_tile(rng, blue_shape, forbidden=None)
    red_tile = _sample_tile(rng, red_shape, forbidden=_tile_signature(blue_tile))

    blue_positions = _candidate_seed_positions(inside_mask, blue_tile.shape)
    red_positions = _candidate_seed_positions(outside_mask, red_tile.shape)
    if not blue_positions or not red_positions:
        return None

    blue_top_left = rng.choice(blue_positions)
    red_top_left = rng.choice(red_positions)

    grid = np.zeros((height, width), dtype=int)
    grid[wall_mask] = GREY
    _paint_seed(grid, blue_tile, blue_top_left, BLUE)
    _paint_seed(grid, red_tile, red_top_left, RED)

    occupied = grid != BLACK
    indicator_choices = _indicator_candidates(inside_mask, occupied)
    if not indicator_choices:
        return None
    indicator = rng.choice(indicator_choices)
    grid[indicator] = ORANGE

    if int(np.sum(grid == ORANGE)) != 1:
        return None

    inside = _flood_from(~wall_mask, indicator)
    outside = (~wall_mask) & ~inside
    if not np.array_equal(inside, inside_mask):
        return None
    if not np.array_equal(outside, outside_mask):
        return None

    blue_mask = (grid == BLUE) & inside
    red_mask = (grid == RED) & outside
    if not blue_mask.any() or not red_mask.any():
        return None

    br0, bc0, br1, bc1 = _bbox(blue_mask)
    rr0, rc0, rr1, rc1 = _bbox(red_mask)
    if _tile_signature(blue_mask[br0 : br1 + 1, bc0 : bc1 + 1]) != _tile_signature(blue_tile):
        return None
    if _tile_signature(red_mask[rr0 : rr1 + 1, rc0 : rc1 + 1]) != _tile_signature(red_tile):
        return None

    # Reject layouts where a wrong "anchor at region bbox" shortcut would accidentally work.
    inside_bbox = _bbox(inside)
    outside_bbox = _bbox(outside)
    if ((blue_top_left[0] - inside_bbox[0]) % blue_tile.shape[0] == 0 and (blue_top_left[1] - inside_bbox[1]) % blue_tile.shape[1] == 0):
        return None
    if ((red_top_left[0] - outside_bbox[0]) % red_tile.shape[0] == 0 and (red_top_left[1] - outside_bbox[1]) % red_tile.shape[1] == 0):
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generate_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input grid")
