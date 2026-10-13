from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
WALL = 5
COLOR_POOL = [1, 2, 3, 4, 6, 7, 8, 9]

GRID_MIN = 22
GRID_MAX = 30
TEMPLATE_COUNT_RANGE = (2, 4)
REGION_EXTRA_RANGE = (0, 2)
GEN_ATTEMPTS = 300

DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _components(mask: np.ndarray) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
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
                for dr, dc in DIR4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _touches_border(coords: np.ndarray, h: int, w: int) -> bool:
    return bool(
        np.any(coords[:, 0] == 0)
        or np.any(coords[:, 0] == h - 1)
        or np.any(coords[:, 1] == 0)
        or np.any(coords[:, 1] == w - 1)
    )


def _extract_regions(grid: np.ndarray) -> list[dict]:
    h, w = grid.shape
    regions: list[dict] = []
    for coords in _components(grid != WALL):
        if _touches_border(coords, h, w):
            continue
        values = grid[coords[:, 0], coords[:, 1]]
        non_bg = np.flatnonzero(values != BG)
        if non_bg.size != 1:
            continue
        seed_idx = int(non_bg[0])
        seed_r, seed_c = map(int, coords[seed_idx])
        color = int(values[seed_idx])
        region_mask = np.zeros_like(grid, dtype=bool)
        region_mask[coords[:, 0], coords[:, 1]] = True
        regions.append(
            {
                "coords": coords,
                "mask": region_mask,
                "seed": (seed_r, seed_c),
                "color": color,
            }
        )
    assert regions, "expected at least one enclosed grey region with a singleton seed"
    return regions


def _extract_templates(grid: np.ndarray, regions: list[dict]) -> dict[int, np.ndarray]:
    seed_positions = {tuple(region["seed"]) for region in regions}
    templates: dict[int, np.ndarray] = {}
    for color in sorted({int(region["color"]) for region in regions}):
        all_coords = np.argwhere(grid == color)
        legend_cells = [tuple(map(int, coord)) for coord in all_coords if tuple(map(int, coord)) not in seed_positions]
        assert legend_cells, f"missing legend pixels for color {color}"
        coords = np.array(legend_cells, dtype=int)
        r0, c0, r1, c1 = _bbox(coords)
        assert r1 - r0 + 1 >= 3 and c1 - c0 + 1 >= 3, "legend frame must be at least 3x3"
        patch = grid[r0 : r1 + 1, c0 : c1 + 1]
        border = np.zeros_like(patch, dtype=bool)
        border[0, :] = True
        border[-1, :] = True
        border[:, 0] = True
        border[:, -1] = True
        assert np.all(patch[border] == color), "legend border must be monochrome"
        inner = patch[1:-1, 1:-1]
        assert inner.size > 0, "legend interior must be non-empty"
        assert np.all((inner == BG) | (inner == color)), "legend interior may only contain bg and border color"
        pattern = inner == color
        assert np.any(pattern), "legend interior must contain a template"
        templates[color] = pattern
    return templates


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match each seed color to its legend template, anchor the template at the seed,
    and keep only the part that lands inside the seed's grey-enclosed region.
    """
    g = np.asarray(grid, dtype=int)
    regions = _extract_regions(g)
    templates = _extract_templates(g, regions)

    out = g.copy()
    for region in regions:
        seed_r, seed_c = map(int, region["seed"])
        color = int(region["color"])
        mask = region["mask"]
        pattern = templates[color]
        for dr, dc in np.argwhere(pattern):
            rr = seed_r + int(dr)
            cc = seed_c + int(dc)
            if 0 <= rr < g.shape[0] and 0 <= cc < g.shape[1] and mask[rr, cc]:
                out[rr, cc] = color
    return out


def _grow_connected_mask(height: int, width: int, area: int, rng: random.Random) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    start = (rng.randrange(height), rng.randrange(width))
    frontier = [start]
    while frontier and int(mask.sum()) < area:
        idx = rng.randrange(len(frontier))
        r, c = frontier.pop(idx)
        if mask[r, c]:
            continue
        mask[r, c] = True
        for dr, dc in DIR4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < height and 0 <= nc < width and not mask[nr, nc]:
                frontier.append((nr, nc))
    return mask


def _make_template(rng: random.Random) -> np.ndarray | None:
    for _ in range(100):
        height = rng.randint(3, 5)
        width = rng.randint(3, 5)
        total = height * width
        low = max(3, (total + 3) // 4)
        high = max(low, min(total - 1, (2 * total) // 3))
        area = rng.randint(low, high)
        mask = _grow_connected_mask(height, width, area, rng)
        if int(mask.sum()) != area:
            continue
        if len(_components(mask)) != 1:
            continue
        if mask.all():
            continue
        if not mask[0].any():
            continue
        if not mask[:, 0].any():
            continue
        return mask
    return None


def _seed_candidates(interior: np.ndarray, pattern: np.ndarray, clipped: bool) -> list[tuple[int, int]]:
    offsets = np.argwhere(pattern)
    total = int(offsets.shape[0])
    candidates: list[tuple[int, int]] = []
    for r, c in np.argwhere(interior):
        visible = 0
        for dr, dc in offsets:
            rr = int(r) + int(dr)
            cc = int(c) + int(dc)
            if 0 <= rr < interior.shape[0] and 0 <= cc < interior.shape[1] and interior[rr, cc]:
                visible += 1
        output_cells = visible + (0 if pattern[0, 0] else 1)
        if clipped:
            if 0 < visible < total and output_cells >= 2:
                candidates.append((int(r), int(c)))
        else:
            if visible == total and output_cells >= 2:
                candidates.append((int(r), int(c)))
    return candidates


def _carve_notch(mask: np.ndarray, edge: str, depth: int, span_start: int, span_end: int) -> np.ndarray:
    carved = mask.copy()
    if edge == "top":
        carved[:depth, span_start:span_end] = False
    elif edge == "bottom":
        carved[-depth:, span_start:span_end] = False
    elif edge == "left":
        carved[span_start:span_end, :depth] = False
    else:
        carved[span_start:span_end, -depth:] = False
    return carved


def _make_region(pattern: np.ndarray, clipped: bool, rng: random.Random) -> tuple[np.ndarray, tuple[int, int]] | None:
    ph, pw = pattern.shape
    for _ in range(150):
        if clipped:
            height = rng.randint(max(4, ph - 1), min(9, ph + 3))
            width = rng.randint(max(4, pw - 1), min(9, pw + 3))
        else:
            height = rng.randint(max(5, ph), min(10, ph + 4))
            width = rng.randint(max(5, pw), min(10, pw + 4))
        interior = np.ones((height, width), dtype=bool)
        changed = False
        for _ in range(rng.randint(1, 3)):
            edge = rng.choice(["top", "bottom", "left", "right"])
            if edge in ("top", "bottom"):
                if width <= 3:
                    continue
                depth = rng.randint(1, max(1, height - 2))
                span = rng.randint(1, max(1, width - 2))
                start = rng.randint(1, width - span - 1)
                trial = _carve_notch(interior, edge, depth, start, start + span)
            else:
                if height <= 3:
                    continue
                depth = rng.randint(1, max(1, width - 2))
                span = rng.randint(1, max(1, height - 2))
                start = rng.randint(1, height - span - 1)
                trial = _carve_notch(interior, edge, depth, start, start + span)
            if trial.any() and len(_components(trial)) == 1:
                if trial[0].any() and trial[-1].any() and trial[:, 0].any() and trial[:, -1].any():
                    interior = trial
                    changed = True
        if not changed:
            continue
        fill = int(interior.sum())
        total = height * width
        if fill <= 5 or fill >= total:
            continue
        candidates = _seed_candidates(interior, pattern, clipped)
        if not candidates:
            continue
        return interior, rng.choice(candidates)
    return None


def _build_region_patch(interior: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    padded = np.zeros((interior.shape[0] + 2, interior.shape[1] + 2), dtype=bool)
    padded[1:-1, 1:-1] = interior
    dilated = padded.copy()
    for r, c in np.argwhere(padded):
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr, nc = int(r) + dr, int(c) + dc
                if 0 <= nr < dilated.shape[0] and 0 <= nc < dilated.shape[1]:
                    dilated[nr, nc] = True
    boundary = dilated & ~padded
    if len(_components(boundary)) != 1:
        return None, None
    patch = np.full(boundary.shape, BG, dtype=int)
    patch[boundary] = WALL
    return patch, padded


def _stamp_expected(grid: np.ndarray, placements: list[dict], templates: dict[int, np.ndarray]) -> np.ndarray:
    out = grid.copy()
    for region in placements:
        color = int(region["color"])
        seed_r, seed_c = map(int, region["seed"])
        mask = region["mask"]
        for dr, dc in np.argwhere(templates[color]):
            rr = seed_r + int(dr)
            cc = seed_c + int(dc)
            if 0 <= rr < out.shape[0] and 0 <= cc < out.shape[1] and mask[rr, cc]:
                out[rr, cc] = color
    return out


def _place_patch(
    grid: np.ndarray,
    occupied: np.ndarray,
    patch: np.ndarray,
    row_range: tuple[int, int],
    col_range: tuple[int, int],
    rng: random.Random,
) -> tuple[int, int] | None:
    ph, pw = patch.shape
    r_lo, r_hi = row_range
    c_lo, c_hi = col_range
    if r_hi - r_lo < ph or c_hi - c_lo < pw:
        return None
    for _ in range(120):
        top = rng.randint(r_lo, r_hi - ph)
        left = rng.randint(c_lo, c_hi - pw)
        rr0 = max(0, top - 1)
        cc0 = max(0, left - 1)
        rr1 = min(grid.shape[0], top + ph + 1)
        cc1 = min(grid.shape[1], left + pw + 1)
        if occupied[rr0:rr1, cc0:cc1].any():
            continue
        sub = grid[top : top + ph, left : left + pw]
        if np.any((patch != BG) & (sub != BG)):
            continue
        sub[patch != BG] = patch[patch != BG]
        occupied[rr0:rr1, cc0:cc1] = True
        return top, left
    return None


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    height = rng.randint(GRID_MIN, GRID_MAX)
    width = rng.randint(GRID_MIN, GRID_MAX)
    grid = np.full((height, width), BG, dtype=int)
    occupied = np.zeros_like(grid, dtype=bool)

    n_templates = rng.randint(*TEMPLATE_COUNT_RANGE)
    colors = rng.sample(COLOR_POOL, n_templates)
    templates: dict[int, np.ndarray] = {}
    legend_patches: list[tuple[int, np.ndarray]] = []
    for color in colors:
        pattern = _make_template(rng)
        if pattern is None:
            return None
        patch = np.full((pattern.shape[0] + 2, pattern.shape[1] + 2), BG, dtype=int)
        patch[0, :] = color
        patch[-1, :] = color
        patch[:, 0] = color
        patch[:, -1] = color
        patch[1:-1, 1:-1][pattern] = color
        templates[color] = pattern
        legend_patches.append((color, patch))

    orientation = rng.choice(["top", "left"])
    gap = rng.randint(1, 2)
    if orientation == "top":
        total_width = sum(patch.shape[1] for _color, patch in legend_patches) + gap * (n_templates - 1)
        band_height = max(patch.shape[0] for _color, patch in legend_patches)
        if total_width > width - 2 or band_height > height - 10:
            return None
        cur_c = rng.randint(1, width - total_width - 1)
        top_row = 0
        for _color, patch in legend_patches:
            top = top_row
            left = cur_c
            ph, pw = patch.shape
            grid[top : top + ph, left : left + pw] = patch
            occupied[max(0, top - 1) : min(height, top + ph + 1), max(0, left - 1) : min(width, left + pw + 1)] = True
            cur_c += pw + gap
        row_range = (band_height + 2, height - 1)
        col_range = (1, width - 1)
    else:
        total_height = sum(patch.shape[0] for _color, patch in legend_patches) + gap * (n_templates - 1)
        band_width = max(patch.shape[1] for _color, patch in legend_patches)
        if total_height > height - 2 or band_width > width - 10:
            return None
        cur_r = rng.randint(1, height - total_height - 1)
        left_col = 0
        for _color, patch in legend_patches:
            top = cur_r
            left = left_col
            ph, pw = patch.shape
            grid[top : top + ph, left : left + pw] = patch
            occupied[max(0, top - 1) : min(height, top + ph + 1), max(0, left - 1) : min(width, left + pw + 1)] = True
            cur_r += ph + gap
        row_range = (1, height - 1)
        col_range = (band_width + 2, width - 1)

    n_regions = n_templates + rng.randint(*REGION_EXTRA_RANGE)
    region_colors = colors.copy()
    while len(region_colors) < n_regions:
        region_colors.append(rng.choice(colors))
    rng.shuffle(region_colors)

    modes = []
    if n_regions >= 2:
        modes = [False, True] + [rng.choice([False, True]) for _ in range(n_regions - 2)]
    else:
        modes = [rng.choice([False, True])]
    rng.shuffle(modes)

    placed_regions: list[dict] = []
    for color, clipped in zip(region_colors, modes):
        made = _make_region(templates[color], clipped, rng)
        if made is None:
            return None
        interior, local_seed = made
        patch, padded_mask = _build_region_patch(interior)
        if patch is None:
            return None
        placed = _place_patch(grid, occupied, patch, row_range, col_range, rng)
        if placed is None:
            return None
        top, left = placed
        global_mask = np.zeros_like(grid, dtype=bool)
        global_mask[top : top + padded_mask.shape[0], left : left + padded_mask.shape[1]] = padded_mask
        seed = (top + 1 + int(local_seed[0]), left + 1 + int(local_seed[1]))
        if not global_mask[seed]:
            return None
        if grid[seed] != BG:
            return None
        grid[seed] = color
        placed_regions.append({"color": color, "seed": seed, "mask": global_mask, "clipped": clipped})

    expected = _stamp_expected(grid, placed_regions, templates)
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    if np.array_equal(actual, grid):
        return None
    if len(_extract_regions(grid)) != n_regions:
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid sample")
