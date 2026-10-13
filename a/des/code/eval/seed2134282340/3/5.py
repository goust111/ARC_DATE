from __future__ import annotations

import random

import numpy as np


BLACK = 0
GREY = 5
FILL_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
DIR8 = DIR4 + ((1, 1), (1, -1), (-1, 1), (-1, -1))

GRID_HEIGHT_RANGE = (22, 30)
GRID_WIDTH_RANGE = (22, 30)
OBJECT_COUNT_RANGE = (3, 5)
BASE_HEIGHT_RANGE = (5, 8)
BASE_WIDTH_RANGE = (5, 8)
BULGE_DEPTH_RANGE = (1, 3)
MAX_BULGES = 3
PLACEMENT_TRIES = 120
SHAPE_TRIES = 120
GENERATION_TRIES = 400


def _components(mask: np.ndarray) -> list[tuple[list[tuple[int, int]], bool]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[tuple[list[tuple[int, int]], bool]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            touches_border = False
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                if cr == 0 or cr == h - 1 or cc == 0 or cc == w - 1:
                    touches_border = True
                for dr, dc in DIR4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append((coords, touches_border))
    return comps


def _crop(mask: np.ndarray) -> np.ndarray:
    rows = np.flatnonzero(mask.any(axis=1))
    cols = np.flatnonzero(mask.any(axis=0))
    assert rows.size and cols.size, "mask must be non-empty"
    return mask[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1]


def _erode4(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    out = np.zeros_like(mask, dtype=bool)
    for r in range(h):
        for c in range(w):
            if not mask[r, c]:
                continue
            ok = True
            for dr, dc in DIR4:
                nr, nc = r + dr, c + dc
                if not (0 <= nr < h and 0 <= nc < w and mask[nr, nc]):
                    ok = False
                    break
            if ok:
                out[r, c] = True
    return out


def _dilate8(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    out = mask.copy()
    for r, c in np.argwhere(mask):
        rr = int(r)
        cc = int(c)
        for dr, dc in DIR8:
            nr, nc = rr + dr, cc + dc
            if 0 <= nr < h and 0 <= nc < w:
                out[nr, nc] = True
    return out


def _bbox_dims(coords: list[tuple[int, int]]) -> tuple[int, int]:
    rows = [r for r, _c in coords]
    cols = [c for _r, c in coords]
    return max(rows) - min(rows) + 1, max(cols) - min(cols) + 1


def _find_enclosed_regions(grid: np.ndarray) -> list[list[tuple[int, int]]]:
    comps = _components(grid != GREY)
    return [coords for coords, touches_border in comps if not touches_border]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each grey-bounded region with a seed-colored pattern determined by the
    number of same-colored seed pixels inside that region.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    out = g.copy()
    regions = _find_enclosed_regions(g)
    assert regions, "expected at least one enclosed region"

    for region in regions:
        seeds = [(r, c) for r, c in region if int(g[r, c]) not in (BLACK, GREY)]
        assert 1 <= len(seeds) <= 3, "expected 1 to 3 seeds per region"
        seed_colors = {int(g[r, c]) for r, c in seeds}
        assert len(seed_colors) == 1, "all seeds in a region must share one color"
        fill_color = next(iter(seed_colors))

        top_row = min(r for r, _c in region)
        anchor_r, anchor_c = min(region)

        for r, c in region:
            if len(seeds) == 1:
                out[r, c] = fill_color
            elif len(seeds) == 2:
                out[r, c] = fill_color if (r - top_row) % 2 == 0 else BLACK
            else:
                parity = (r - anchor_r) + (c - anchor_c)
                out[r, c] = fill_color if parity % 2 == 0 else BLACK

    return out


def _shape_from_outer(outer: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    outer = _crop(outer)
    interior = _erode4(outer)
    boundary = outer & ~interior

    if not boundary.any() or not interior.any():
        return None
    if len(_components(boundary)) != 1:
        return None
    if len(_components(interior)) != 1:
        return None

    interior_coords = [(int(r), int(c)) for r, c in np.argwhere(interior)]
    height, width = _bbox_dims(interior_coords)
    if height < 3 or width < 3 or len(interior_coords) < 9:
        return None

    test = np.zeros((outer.shape[0] + 4, outer.shape[1] + 4), dtype=int)
    patch = test[2 : 2 + boundary.shape[0], 2 : 2 + boundary.shape[1]]
    patch[boundary] = GREY
    enclosed = _find_enclosed_regions(test)
    if len(enclosed) != 1:
        return None

    shifted_interior = {(r + 2, c + 2) for r, c in interior_coords}
    if set(enclosed[0]) != shifted_interior:
        return None

    return outer, boundary, interior


def _random_shape() -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    for _ in range(SHAPE_TRIES):
        base_h = random.randint(*BASE_HEIGHT_RANGE)
        base_w = random.randint(*BASE_WIDTH_RANGE)
        pad = max(BULGE_DEPTH_RANGE[1], 3)
        outer = np.zeros((base_h + 2 * pad, base_w + 2 * pad), dtype=bool)
        outer[pad : pad + base_h, pad : pad + base_w] = True

        bulge_count = random.choices((0, 1, 2, 3), weights=(1, 3, 4, 2), k=1)[0]
        sides = random.sample(("top", "bottom", "left", "right"), k=bulge_count)
        for side in sides:
            depth = random.randint(*BULGE_DEPTH_RANGE)
            if side in ("top", "bottom"):
                span = random.randint(2, base_w)
                start = random.randint(0, base_w - span)
                c0 = pad + start
                c1 = c0 + span
                if side == "top":
                    outer[pad - depth : pad, c0:c1] = True
                else:
                    outer[pad + base_h : pad + base_h + depth, c0:c1] = True
            else:
                span = random.randint(2, base_h)
                start = random.randint(0, base_h - span)
                r0 = pad + start
                r1 = r0 + span
                if side == "left":
                    outer[r0:r1, pad - depth : pad] = True
                else:
                    outer[r0:r1, pad + base_w : pad + base_w + depth] = True

        shaped = _shape_from_outer(outer)
        if shaped is not None:
            return shaped
    return None


def _pick_seed_color(used_colors: list[int]) -> int:
    if used_colors and random.random() < 0.35:
        return int(random.choice(used_colors))
    return int(random.choice(FILL_COLORS))


def _place_shape(
    grid: np.ndarray,
    blocked: np.ndarray,
    boundary: np.ndarray,
    interior: np.ndarray,
    seed_count: int,
    seed_color: int,
) -> bool:
    h, w = boundary.shape
    gh, gw = grid.shape
    interior_coords = [(int(r), int(c)) for r, c in np.argwhere(interior)]

    row_positions = list(range(1, gh - h))
    col_positions = list(range(1, gw - w))
    random.shuffle(row_positions)
    random.shuffle(col_positions)

    for r0 in row_positions[:PLACEMENT_TRIES]:
        for c0 in col_positions[:PLACEMENT_TRIES]:
            if blocked[r0 - 1 : r0 + h + 1, c0 - 1 : c0 + w + 1].any():
                continue

            chosen = random.sample(interior_coords, seed_count)
            patch = grid[r0 : r0 + h, c0 : c0 + w]
            patch[boundary] = GREY
            for sr, sc in chosen:
                grid[r0 + sr, c0 + sc] = seed_color
            blocked[r0 - 1 : r0 + h + 1, c0 - 1 : c0 + w + 1] = True
            return True
    return False


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_TRIES):
        h = random.randint(*GRID_HEIGHT_RANGE)
        w = random.randint(*GRID_WIDTH_RANGE)
        grid = np.full((h, w), BLACK, dtype=int)
        blocked = np.zeros((h, w), dtype=bool)

        object_count = random.randint(*OBJECT_COUNT_RANGE)
        seed_counts = [1, 2, 3]
        while len(seed_counts) < object_count:
            seed_counts.append(random.randint(1, 3))
        random.shuffle(seed_counts)

        used_colors: list[int] = []
        ok = True
        for seed_count in seed_counts:
            shaped = _random_shape()
            if shaped is None:
                ok = False
                break
            _outer, boundary, interior = shaped
            seed_color = _pick_seed_color(used_colors)
            if not _place_shape(grid, blocked, boundary, interior, seed_count, seed_color):
                ok = False
                break
            used_colors.append(seed_color)

        if not ok:
            continue

        regions = _find_enclosed_regions(grid)
        if len(regions) != object_count:
            continue

        counts_seen = {
            len([(r, c) for r, c in region if grid[r, c] not in (BLACK, GREY)])
            for region in regions
        }
        if not {1, 2, 3}.issubset(counts_seen):
            continue

        try:
            out = main(grid)
        except AssertionError:
            continue
        if np.array_equal(out, grid):
            continue
        return grid

    raise RuntimeError("failed to generate a valid input")
