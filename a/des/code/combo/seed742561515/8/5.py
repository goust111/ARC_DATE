import math
from typing import Dict, List, Tuple

import numpy as np


BG = 0
SEP = 5
PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]
GEN_ATTEMPTS = 400


def _spans(length: int, separators: List[int]) -> List[Tuple[int, int]]:
    spans: List[Tuple[int, int]] = []
    start = 0
    for idx in sorted(separators):
        if start < idx:
            spans.append((start, idx))
        start = idx + 1
    if start < length:
        spans.append((start, length))
    if not spans:
        spans.append((0, length))
    return spans


def _is_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    h, w = mask.shape
    stack = [tuple(map(int, coords[0]))]
    seen = {stack[0]}
    while stack:
        r, c = stack.pop()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and (nr, nc) not in seen:
                seen.add((nr, nc))
                stack.append((nr, nc))
    return len(seen) == int(mask.sum())


def _classify_pattern(pattern: np.ndarray) -> str:
    # "Horizontal symmetry" is interpreted as symmetry across a horizontal axis.
    horizontal_sym = np.array_equal(pattern, np.flipud(pattern))
    vertical_sym = np.array_equal(pattern, np.fliplr(pattern))
    rotational_sym = np.array_equal(pattern, np.rot90(pattern, 2))

    if horizontal_sym:
        return "horizontal"
    if vertical_sym:
        return "vertical"
    if rotational_sym:
        return "rot180"
    return "plain"


def _extract_seed(region: np.ndarray) -> Tuple[np.ndarray, str, int, int]:
    mask = (region != BG) & (region != SEP)
    coords = np.argwhere(mask)
    assert coords.size > 0, "each region must contain one seed"
    assert _is_connected(mask), "seed must be a single 4-connected component"

    r0, c0 = map(int, coords.min(axis=0))
    r1, c1 = map(int, coords.max(axis=0))
    seed = region[r0 : r1 + 1, c0 : c1 + 1].copy()

    top = r0 == 0
    bottom = r1 == region.shape[0] - 1
    left = c0 == 0
    right = c1 == region.shape[1] - 1
    assert top ^ bottom, "seed must touch exactly one horizontal border"
    assert left ^ right, "seed must touch exactly one vertical border"

    tile_h, tile_w = map(int, seed.shape)
    region_h, region_w = map(int, region.shape)
    assert region_h % tile_h == 0 and region_w % tile_w == 0, "seed size must divide the region"

    expected_r0 = 0 if top else region_h - tile_h
    expected_c0 = 0 if left else region_w - tile_w
    assert r0 == expected_r0 and c0 == expected_c0, "seed bbox must occupy a region corner tile"

    tiles_down = region_h // tile_h
    tiles_across = region_w // tile_w
    seed_tile_r = 0 if top else tiles_down - 1
    seed_tile_c = 0 if left else tiles_across - 1
    return seed, _classify_pattern(seed), seed_tile_r, seed_tile_c


def _fill_region(shape: Tuple[int, int], seed: np.ndarray, kind: str, seed_tile_r: int, seed_tile_c: int) -> np.ndarray:
    region_h, region_w = map(int, shape)
    tile_h, tile_w = map(int, seed.shape)
    tiles_down = region_h // tile_h
    tiles_across = region_w // tile_w

    hflip = np.fliplr(seed)
    vflip = np.flipud(seed)
    rot180 = np.rot90(seed, 2)

    out = np.full((region_h, region_w), BG, dtype=int)
    for tr in range(tiles_down):
        for tc in range(tiles_across):
            dr = tr - seed_tile_r
            dc = tc - seed_tile_c
            if kind == "horizontal":
                tile = seed if dc % 2 == 0 else hflip
            elif kind == "vertical":
                tile = seed if dr % 2 == 0 else vflip
            elif kind == "rot180":
                tile = seed if (dr + dc) % 2 == 0 else rot180
            else:
                tile = seed
            rr = tr * tile_h
            cc = tc * tile_w
            out[rr : rr + tile_h, cc : cc + tile_w] = tile
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Split the grid by full grey separator lines, detect the unique corner seed
    inside every rectangular region, classify its symmetry, and tile that seed
    across the region with the branch-specific orientation rule.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0

    h, w = map(int, g.shape)
    sep_rows = [r for r in range(h) if np.all(g[r, :] == SEP)]
    sep_cols = [c for c in range(w) if np.all(g[:, c] == SEP)]
    row_spans = _spans(h, sep_rows)
    col_spans = _spans(w, sep_cols)

    out = g.copy()
    for r0, r1 in row_spans:
        for c0, c1 in col_spans:
            region = g[r0:r1, c0:c1]
            seed, kind, seed_tile_r, seed_tile_c = _extract_seed(region)
            out[r0:r1, c0:c1] = _fill_region(region.shape, seed, kind, seed_tile_r, seed_tile_c)
    return out


def _matches_exact_class(pattern: np.ndarray, kind: str) -> bool:
    horizontal_sym = np.array_equal(pattern, np.flipud(pattern))
    vertical_sym = np.array_equal(pattern, np.fliplr(pattern))
    rotational_sym = np.array_equal(pattern, np.rot90(pattern, 2))

    if kind == "horizontal":
        return horizontal_sym and not vertical_sym
    if kind == "vertical":
        return vertical_sym and not horizontal_sym
    if kind == "rot180":
        return rotational_sym and not horizontal_sym and not vertical_sym
    return not horizontal_sym and not vertical_sym and not rotational_sym


def _touches_all_edges(pattern: np.ndarray) -> bool:
    return (
        np.any(pattern[0, :] != BG)
        and np.any(pattern[-1, :] != BG)
        and np.any(pattern[:, 0] != BG)
        and np.any(pattern[:, -1] != BG)
    )


def _symmetry_orbits(h: int, w: int, kind: str) -> List[List[Tuple[int, int]]]:
    seen = set()
    orbits: List[List[Tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if (r, c) in seen:
                continue
            if kind == "horizontal":
                cells = {(r, c), (h - 1 - r, c)}
            elif kind == "vertical":
                cells = {(r, c), (r, w - 1 - c)}
            elif kind == "rot180":
                cells = {(r, c), (h - 1 - r, w - 1 - c)}
            else:
                cells = {(r, c)}
            orbit = sorted(cells)
            seen.update(orbit)
            orbits.append(orbit)
    return orbits


def _random_pattern(kind: str, rng: np.random.Generator) -> np.ndarray:
    for _ in range(500):
        h = int(rng.integers(2, 5))
        w = int(rng.integers(2, 5))
        color_count = int(rng.integers(1, 3))
        colors = [int(x) for x in rng.choice(PALETTE, size=color_count, replace=False)]
        pattern = np.zeros((h, w), dtype=int)

        if kind == "plain":
            p = float(rng.uniform(0.4, 0.75))
            for r in range(h):
                for c in range(w):
                    if float(rng.random()) < p:
                        pattern[r, c] = int(rng.choice(colors))
        else:
            p = float(rng.uniform(0.45, 0.8))
            for orbit in _symmetry_orbits(h, w, kind):
                if float(rng.random()) < p:
                    color = int(rng.choice(colors))
                    for r, c in orbit:
                        pattern[r, c] = color

        mask = pattern != BG
        if mask.sum() < 3 or mask.sum() == pattern.size:
            continue
        if not _touches_all_edges(pattern):
            continue
        if not _is_connected(mask):
            continue
        if not _matches_exact_class(pattern, kind):
            continue
        return pattern
    raise RuntimeError(f"failed to build {kind} seed")


def _place_seed(region: np.ndarray, seed: np.ndarray, corner: str) -> None:
    h, w = map(int, region.shape)
    tile_h, tile_w = map(int, seed.shape)
    if corner == "tl":
        r0, c0 = 0, 0
    elif corner == "tr":
        r0, c0 = 0, w - tile_w
    elif corner == "bl":
        r0, c0 = h - tile_h, 0
    else:
        r0, c0 = h - tile_h, w - tile_w
    region[r0 : r0 + tile_h, c0 : c0 + tile_w] = seed


def _build_attempt(rng: np.random.Generator) -> np.ndarray | None:
    layouts = [(2, 2), (2, 3), (3, 2)]
    row_groups, col_groups = layouts[int(rng.integers(len(layouts)))]

    positions = [(r, c) for r in range(row_groups) for c in range(col_groups)]
    rng.shuffle(positions)
    kinds = ["horizontal", "vertical", "rot180", "plain"]

    specs: Dict[Tuple[int, int], Dict[str, object]] = {}
    assigned = list(kinds)
    extra = len(positions) - len(assigned)
    if extra > 0:
        assigned.extend([kinds[int(rng.integers(len(kinds)))] for _ in range(extra)])
    rng.shuffle(assigned)

    for pos, kind in zip(positions, assigned):
        specs[pos] = {
            "kind": kind,
            "seed": _random_pattern(kind, rng),
            "corner": ["tl", "tr", "bl", "br"][int(rng.integers(4))],
        }

    row_heights: List[int] = []
    for r in range(row_groups):
        factors = [int(specs[(r, c)]["seed"].shape[0]) for c in range(col_groups)]
        base = math.lcm(*factors)
        mult = int(rng.integers(2, 4))
        row_heights.append(base * mult)

    col_widths: List[int] = []
    for c in range(col_groups):
        factors = [int(specs[(r, c)]["seed"].shape[1]) for r in range(row_groups)]
        base = math.lcm(*factors)
        mult = int(rng.integers(2, 4))
        col_widths.append(base * mult)

    total_h = sum(row_heights) + (row_groups - 1)
    total_w = sum(col_widths) + (col_groups - 1)
    if not (12 <= total_h <= 32 and 12 <= total_w <= 32):
        return None

    grid = np.full((total_h, total_w), BG, dtype=int)
    row_offsets: List[Tuple[int, int]] = []
    cursor = 0
    for height in row_heights:
        row_offsets.append((cursor, cursor + height))
        cursor += height + 1
    col_offsets: List[Tuple[int, int]] = []
    cursor = 0
    for width in col_widths:
        col_offsets.append((cursor, cursor + width))
        cursor += width + 1

    for r in range(row_groups - 1):
        sep_r = row_offsets[r][1]
        grid[sep_r, :] = SEP
    for c in range(col_groups - 1):
        sep_c = col_offsets[c][1]
        grid[:, sep_c] = SEP

    for (r, c), spec in specs.items():
        r0, r1 = row_offsets[r]
        c0, c1 = col_offsets[c]
        region = grid[r0:r1, c0:c1]
        _place_seed(region, spec["seed"], spec["corner"])

    try:
        _ = main(grid)
    except AssertionError:
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_ATTEMPTS):
        grid = _build_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
