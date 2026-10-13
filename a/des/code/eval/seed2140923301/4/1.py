import random
from collections import deque

import numpy as np


BG = 0
BLUE = 1
GREY = 5
COLOR_POOL = (2, 3, 4, 6, 7, 8, 9)

LEFT_AREA_WIDTH = 4
MAX_PATTERN_SIZE = 3
LEGEND_WIDTH = 2 + LEFT_AREA_WIDTH + MAX_PATTERN_SIZE
GEN_RETRIES = 400


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    parts: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            q = deque([(r, c)])
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            parts.append(comp)
    return parts


def _bbox(cells: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _parse_legend(grid: np.ndarray) -> tuple[dict[int, np.ndarray], np.ndarray]:
    grey_comps = _components(grid == GREY)
    candidates: list[tuple[int, int, int, int]] = []
    for comp in grey_comps:
        r0, c0, r1, c1 = _bbox(comp)
        if r1 - r0 < 3 or c1 - c0 < 4:
            continue
        if not np.all(grid[r0, c0 : c1 + 1] == GREY):
            continue
        if not np.all(grid[r1, c0 : c1 + 1] == GREY):
            continue
        if not np.all(grid[r0 : r1 + 1, c0] == GREY):
            continue
        if not np.all(grid[r0 : r1 + 1, c1] == GREY):
            continue
        full_rows = [r for r in range(r0, r1 + 1) if np.all(grid[r, c0 : c1 + 1] == GREY)]
        if len(full_rows) < 2:
            continue
        ok = True
        for a, b in zip(full_rows, full_rows[1:]):
            gap = b - a - 1
            if gap == 0:
                continue
            if gap not in (2, 3):
                ok = False
                break
        if ok:
            candidates.append((r0, c0, r1, c1))

    assert candidates, "legend not found"
    r0, c0, r1, c1 = min(candidates, key=lambda box: (box[0], box[1], box[2] - box[0], box[3] - box[1]))

    legend_mask = np.zeros_like(grid, dtype=bool)
    legend_mask[r0 : r1 + 1, c0 : c1 + 1] = True

    mapping: dict[int, np.ndarray] = {}
    full_rows = [r for r in range(r0, r1 + 1) if np.all(grid[r, c0 : c1 + 1] == GREY)]
    for a, b in zip(full_rows, full_rows[1:]):
        if b == a + 1:
            continue
        rs = a + 1
        re = b - 1
        size = re - rs + 1
        assert size in (2, 3)
        band = grid[rs : re + 1, c0 + 1 : c1]
        pattern = band[:, -size:].copy()
        left = band[:, :-size]
        seeds = np.argwhere((left != BG) & (left != GREY))
        assert seeds.shape[0] == 1, "each legend row must contain one seed-color key"
        seed_color = int(left[tuple(seeds[0])])
        assert seed_color not in mapping, "legend keys must be unique"
        mapping[seed_color] = pattern

    assert mapping, "legend mapping is empty"
    return mapping, legend_mask


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    mapping, legend_mask = _parse_legend(g)

    passable = (g != BLUE) & ~legend_mask
    out = g.copy()
    seen = np.zeros((h, w), dtype=bool)

    for r in range(h):
        for c in range(w):
            if not passable[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            touches_border = False
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                if cr == 0 or cc == 0 or cr == h - 1 or cc == w - 1:
                    touches_border = True
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and passable[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))

            if touches_border:
                continue

            seeds = [(rr, cc) for rr, cc in comp if int(g[rr, cc]) not in (BG, GREY)]
            if len(seeds) != 1:
                continue

            sr, sc = seeds[0]
            seed_color = int(g[sr, sc])
            assert seed_color in mapping, "region seed must appear in legend"
            pattern = mapping[seed_color]
            ph, pw = pattern.shape
            for rr, cc in comp:
                out[rr, cc] = int(pattern[(rr - sr) % ph, (cc - sc) % pw])

    return out


def _is_connected(mask: np.ndarray) -> bool:
    cells = np.argwhere(mask)
    if cells.size == 0:
        return False
    start = tuple(map(int, cells[0]))
    seen = {start}
    q = deque([start])
    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1] and mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                q.append((nr, nc))
    return len(seen) == int(mask.sum())


def _random_pattern(rng: random.Random, size: int, used: set[tuple[int, ...]]) -> np.ndarray:
    for _ in range(200):
        palette_size = rng.randint(2, min(4, len(COLOR_POOL)))
        palette = rng.sample(COLOR_POOL, palette_size)
        pattern = np.array([[rng.choice(palette) for _ in range(size)] for _ in range(size)], dtype=int)
        sig = tuple(int(v) for v in pattern.ravel())
        if len(set(sig)) < 2:
            continue
        if sig in used:
            continue
        used.add(sig)
        return pattern
    raise RuntimeError("failed to sample a distinct pattern")


def _draw_legend(
    grid: np.ndarray,
    row_specs: list[tuple[int, np.ndarray]],
    top: int,
    left: int,
) -> tuple[int, int, int, int]:
    total_h = 2 + sum(pattern.shape[0] for _, pattern in row_specs) + (len(row_specs) - 1)
    total_w = LEGEND_WIDTH
    r0, c0 = top, left
    r1, c1 = r0 + total_h - 1, c0 + total_w - 1
    assert r1 < grid.shape[0] and c1 < grid.shape[1]

    grid[r0, c0 : c1 + 1] = GREY
    grid[r1, c0 : c1 + 1] = GREY
    grid[r0 : r1 + 1, c0] = GREY
    grid[r0 : r1 + 1, c1] = GREY

    cursor = r0 + 1
    for idx, (seed_color, pattern) in enumerate(row_specs):
        size = pattern.shape[0]
        rs = cursor
        re = cursor + size - 1
        seed_r = rs + size // 2
        seed_c = c0 + 2
        grid[seed_r, seed_c] = seed_color
        pc0 = c1 - size
        grid[rs : re + 1, pc0:c1] = pattern
        cursor = re + 1
        if idx != len(row_specs) - 1:
            grid[cursor, c0 : c1 + 1] = GREY
            cursor += 1

    return r0, c0, r1, c1


def _random_interior_mask(rng: random.Random, height: int, width: int) -> np.ndarray:
    mask = np.ones((height, width), dtype=bool)
    corners = ["tl", "tr", "bl", "br"]
    rng.shuffle(corners)
    notch_count = rng.choice((0, 1, 1, 2))
    min_area = max(8, (height * width) // 2)
    for corner in corners[:notch_count]:
        nh = rng.randint(1, max(1, height - 2))
        nw = rng.randint(1, max(1, width - 2))
        nh = min(nh, max(1, height // 2))
        nw = min(nw, max(1, width // 2))
        proposal = mask.copy()
        if corner == "tl":
            proposal[:nh, :nw] = False
        elif corner == "tr":
            proposal[:nh, width - nw :] = False
        elif corner == "bl":
            proposal[height - nh :, :nw] = False
        else:
            proposal[height - nh :, width - nw :] = False
        if proposal.sum() >= min_area and _is_connected(proposal):
            mask = proposal
    return mask


def _outer_boundary(interior: np.ndarray) -> np.ndarray:
    h, w = interior.shape
    padded = np.zeros((h + 2, w + 2), dtype=bool)
    padded[1:-1, 1:-1] = interior
    boundary = np.zeros_like(padded)
    cells = np.argwhere(padded)
    for r, c in cells:
        r = int(r)
        c = int(c)
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                nr = r + dr
                nc = c + dc
                if not padded[nr, nc]:
                    boundary[nr, nc] = True
    boundary &= ~padded
    return boundary


def _make_region_blueprint(rng: random.Random, seed_color: int, pattern: np.ndarray) -> dict[str, object]:
    size = pattern.shape[0]
    inner_h = rng.randint(size + 2, size + 6)
    inner_w = rng.randint(size + 2, size + 6)
    interior = _random_interior_mask(rng, inner_h, inner_w)
    boundary = _outer_boundary(interior)
    patch = np.full(boundary.shape, BG, dtype=int)
    patch[boundary] = BLUE
    interior_cells = [(int(r) + 1, int(c) + 1) for r, c in np.argwhere(interior)]
    seed = rng.choice(interior_cells)
    patch[seed] = seed_color
    return {
        "patch": patch,
        "interior_cells": interior_cells,
        "seed": seed,
        "pattern": pattern,
    }


def _candidate_positions(occupied: np.ndarray, shape: tuple[int, int]) -> list[tuple[int, int]]:
    h, w = occupied.shape
    ph, pw = shape
    positions: list[tuple[int, int]] = []
    for r0 in range(1, h - ph):
        for c0 in range(1, w - pw):
            rr0 = max(0, r0 - 1)
            rr1 = min(h, r0 + ph + 1)
            cc0 = max(0, c0 - 1)
            cc1 = min(w, c0 + pw + 1)
            if not occupied[rr0:rr1, cc0:cc1].any():
                positions.append((r0, c0))
    return positions


def _apply_region(
    grid: np.ndarray,
    occupied: np.ndarray,
    blueprint: dict[str, object],
    origin: tuple[int, int],
) -> dict[str, object]:
    patch = blueprint["patch"]
    ph, pw = patch.shape
    r0, c0 = origin
    patch_mask = patch != BG
    sub = grid[r0 : r0 + ph, c0 : c0 + pw]
    assert not np.any(sub[patch_mask] != BG)
    sub[patch_mask] = patch[patch_mask]

    rr0 = max(0, r0 - 1)
    rr1 = min(grid.shape[0], r0 + ph + 1)
    cc0 = max(0, c0 - 1)
    cc1 = min(grid.shape[1], c0 + pw + 1)
    occupied[rr0:rr1, cc0:cc1] = True

    interior_cells = [(r0 + lr, c0 + lc) for lr, lc in blueprint["interior_cells"]]
    seed_lr, seed_lc = blueprint["seed"]
    return {
        "cells": interior_cells,
        "seed": (r0 + seed_lr, c0 + seed_lc),
        "pattern": blueprint["pattern"],
    }


def _expected_output(inp: np.ndarray, regions: list[dict[str, object]]) -> np.ndarray:
    out = inp.copy()
    for region in regions:
        sr, sc = region["seed"]
        pattern = region["pattern"]
        ph, pw = pattern.shape
        for r, c in region["cells"]:
            out[r, c] = int(pattern[(r - sr) % ph, (c - sc) % pw])
    return out


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    n_rows = rng.randint(2, 4)
    row_sizes = [rng.choice((2, 3)) for _ in range(n_rows)]
    legend_top = rng.randint(0, 2)
    legend_left = rng.randint(0, 2)
    legend_h = 2 + sum(row_sizes) + (n_rows - 1)
    min_h = max(24, legend_top + legend_h + 8)
    min_w = max(26, legend_left + LEGEND_WIDTH + 12)
    max_h = 32
    max_w = 34
    if min_h > max_h or min_w > max_w:
        return None

    h = rng.randint(min_h, max_h)
    w = rng.randint(min_w, max_w)
    grid = np.full((h, w), BG, dtype=int)

    seed_colors = rng.sample(COLOR_POOL, n_rows)
    used_patterns: set[tuple[int, ...]] = set()
    row_specs: list[tuple[int, np.ndarray]] = []
    for seed_color, size in zip(seed_colors, row_sizes):
        row_specs.append((seed_color, _random_pattern(rng, size, used_patterns)))

    legend_box = _draw_legend(grid, row_specs, legend_top, legend_left)
    occupied = np.zeros((h, w), dtype=bool)
    lr0, lc0, lr1, lc1 = legend_box
    occupied[max(0, lr0 - 1) : min(h, lr1 + 2), max(0, lc0 - 1) : min(w, lc1 + 2)] = True

    extras = 1 if rng.random() < 0.5 else 0
    assignments = list(range(n_rows)) + [rng.randrange(n_rows) for _ in range(extras)]
    rng.shuffle(assignments)

    blueprints = [_make_region_blueprint(rng, row_specs[idx][0], row_specs[idx][1]) for idx in assignments]
    blueprints.sort(key=lambda item: item["patch"].shape[0] * item["patch"].shape[1], reverse=True)

    placed_regions: list[dict[str, object]] = []
    for blueprint in blueprints:
        positions = _candidate_positions(occupied, blueprint["patch"].shape)
        if not positions:
            return None
        origin = rng.choice(positions)
        placed_regions.append(_apply_region(grid, occupied, blueprint, origin))

    comp_count = 0
    for comp in _components((grid != BLUE) & ~_parse_legend(grid)[1]):
        if any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in comp):
            continue
        seeds = [(r, c) for r, c in comp if int(grid[r, c]) not in (BG, GREY)]
        if len(seeds) == 1:
            comp_count += 1
    if comp_count != len(placed_regions):
        return None

    expected = _expected_output(grid, placed_regions)
    try:
        if np.array_equal(main(grid), expected):
            return grid
    except Exception:
        return None
    return None


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: exceeded retries")
