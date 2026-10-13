from __future__ import annotations

import random
from collections import Counter

import numpy as np


_NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
_BORDER_COLOR = 5
_GEN_RETRIES = 400


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                pts.append((rr, cc))
                for dr, dc in _NEIGHBORS4:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(pts)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _touches_edge(coords: list[tuple[int, int]], h: int, w: int) -> bool:
    return any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in coords)


def _parse_legend_component(
    grid: np.ndarray, bg: int, border: int, coords: list[tuple[int, int]]
) -> tuple[dict[int, np.ndarray], tuple[int, int, int, int]] | None:
    h, w = grid.shape
    if not _touches_edge(coords, h, w):
        return None

    r0, c0, r1, c1 = _bbox(coords)
    sub = grid[r0 : r1 + 1, c0 : c1 + 1]

    sep_rows = [i for i in range(sub.shape[0]) if np.all(sub[i, :] == border)]
    sep_cols = [j for j in range(sub.shape[1]) if np.all(sub[:, j] == border)]
    if len(sep_rows) < 2 or len(sep_cols) < 2:
        return None
    if any(b - a != 5 for a, b in zip(sep_rows, sep_rows[1:])):
        return None
    if any(b - a != 5 for a, b in zip(sep_cols, sep_cols[1:])):
        return None

    sep_row_set = set(sep_rows)
    sep_col_set = set(sep_cols)
    border_pts = np.argwhere(sub == border)
    for rr, cc in border_pts:
        if int(rr) not in sep_row_set and int(cc) not in sep_col_set:
            return None

    mapping: dict[int, np.ndarray] = {}
    cell_rows = len(sep_rows) - 1
    cell_cols = len(sep_cols) - 1
    for ri in range(cell_rows):
        for ci in range(cell_cols):
            rs = sep_rows[ri] + 1
            re = sep_rows[ri + 1]
            cs = sep_cols[ci] + 1
            ce = sep_cols[ci + 1]
            if re - rs != 4 or ce - cs != 4:
                return None
            cell = sub[rs:re, cs:ce]
            marker = int(cell[0, 0])
            if marker == bg or marker == border:
                return None
            if np.any(cell[0, 1:] != bg) or np.any(cell[1:, 0] != bg):
                return None
            pattern = cell[1:4, 1:4]
            values = set(map(int, pattern.flatten()))
            if values - {bg, marker}:
                return None
            if marker not in values or bg not in values:
                return None
            if marker in mapping and not np.array_equal(mapping[marker], pattern):
                return None
            mapping[marker] = pattern.copy()

    if not mapping:
        return None
    return mapping, (r0, c0, r1, c1)


def _find_legend(grid: np.ndarray, bg: int) -> tuple[int, dict[int, np.ndarray], tuple[int, int, int, int]]:
    colors = [int(c) for c in np.unique(grid) if int(c) != bg]
    colors.sort(key=lambda color: int(np.sum(grid == color)), reverse=True)

    best: tuple[int, dict[int, np.ndarray], tuple[int, int, int, int], int] | None = None
    for border in colors:
        for coords in _components(grid == border):
            parsed = _parse_legend_component(grid, bg, border, coords)
            if parsed is None:
                continue
            mapping, bbox = parsed
            score = len(coords)
            if best is None or score > best[3]:
                best = (border, mapping, bbox, score)
    if best is None:
        raise AssertionError("legend table not found")
    return best[0], best[1], best[2]


def _find_regions(grid: np.ndarray, bg: int, border: int) -> list[tuple[int, int, int, int, int]]:
    h, w = grid.shape
    regions: list[tuple[int, int, int, int, int]] = []
    for coords in _components(grid == border):
        if _touches_edge(coords, h, w):
            continue
        r0, c0, r1, c1 = _bbox(coords)
        if r1 - r0 < 2 or c1 - c0 < 2:
            continue
        expected = set()
        for cc in range(c0, c1 + 1):
            expected.add((r0, cc))
            expected.add((r1, cc))
        for rr in range(r0 + 1, r1):
            expected.add((rr, c0))
            expected.add((rr, c1))
        if set(coords) != expected:
            continue

        interior = grid[r0 + 1 : r1, c0 + 1 : c1]
        seed_mask = (interior != bg) & (interior != border)
        seeds = np.argwhere(seed_mask)
        if seeds.shape[0] != 1:
            continue
        if np.any(interior[~seed_mask] != bg):
            continue
        seed = int(interior[seed_mask][0])
        regions.append((r0, c0, r1, c1, seed))

    regions.sort()
    return regions


def _tile_pattern(pattern: np.ndarray, height: int, width: int) -> np.ndarray:
    reps_r = (height + pattern.shape[0] - 1) // pattern.shape[0]
    reps_c = (width + pattern.shape[1] - 1) // pattern.shape[1]
    tiled = np.tile(pattern, (reps_r, reps_c))
    return tiled[:height, :width]


def _solve(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = _most_common_color(g)
    border, mapping, _legend_bbox = _find_legend(g, bg)
    regions = _find_regions(g, bg, border)
    assert regions, "no enclosed regions found"

    out = g.copy()
    for r0, c0, r1, c1, seed in regions:
        assert seed in mapping, "seed color missing from legend"
        patch = _tile_pattern(mapping[seed], r1 - r0 - 1, c1 - c0 - 1)
        out[r0 + 1 : r1, c0 + 1 : c1] = patch
    return out


def main(grid: np.ndarray) -> np.ndarray:
    return _solve(grid)


def _random_pattern_masks(rng: random.Random, count: int) -> list[np.ndarray]:
    masks: list[np.ndarray] = []
    seen: set[tuple[int, ...]] = set()
    while len(masks) < count:
        mask = np.array(
            [[1 if rng.random() < 0.5 else 0 for _ in range(3)] for _ in range(3)],
            dtype=int,
        )
        colored = int(mask.sum())
        if not 2 <= colored <= 7:
            continue
        key = tuple(int(v) for v in mask.flatten())
        if key in seen:
            continue
        seen.add(key)
        masks.append(mask.astype(bool))
    return masks


def _build_legend(
    bg: int, border: int, side: str, seed_colors: list[int], masks: list[np.ndarray]
) -> np.ndarray:
    if side in ("top", "bottom"):
        rows, cols = 1, len(seed_colors)
    else:
        rows, cols = len(seed_colors), 1

    legend = np.full((rows * 5 + 1, cols * 5 + 1), bg, dtype=int)
    for rr in range(0, legend.shape[0], 5):
        legend[rr, :] = border
    for cc in range(0, legend.shape[1], 5):
        legend[:, cc] = border
    legend[-1, :] = border
    legend[:, -1] = border

    for idx, color in enumerate(seed_colors):
        cr = idx if cols == 1 else 0
        cc = idx if rows == 1 else 0
        r0 = 1 + cr * 5
        c0 = 1 + cc * 5
        legend[r0, c0] = color
        legend[r0 + 1 : r0 + 4, c0 + 1 : c0 + 4] = np.where(masks[idx], color, bg)
    return legend


def _place_patch(
    grid: np.ndarray, patch: np.ndarray, side: str, rng: random.Random
) -> tuple[int, int]:
    h, w = grid.shape
    ph, pw = patch.shape
    if side == "top":
        r0 = 0
        c0 = rng.randint(0, w - pw)
    elif side == "bottom":
        r0 = h - ph
        c0 = rng.randint(0, w - pw)
    elif side == "left":
        r0 = rng.randint(0, h - ph)
        c0 = 0
    else:
        r0 = rng.randint(0, h - ph)
        c0 = w - pw
    grid[r0 : r0 + ph, c0 : c0 + pw] = patch
    return r0, c0


def _sample_region_specs(
    rng: random.Random, seed_colors: list[int], side: str
) -> list[tuple[int, int, int]]:
    specs: list[tuple[int, int, int]] = []
    max_h = 6 if side in ("left", "right") and len(seed_colors) >= 4 else 7
    max_w = 6 if side in ("top", "bottom") and len(seed_colors) >= 4 else 8
    for color in seed_colors:
        inner_h = rng.randint(2, max_h)
        inner_w = rng.randint(2, max_w)
        specs.append((inner_h, inner_w, color))
    if not any(inner_h % 3 != 0 or inner_w % 3 != 0 for inner_h, inner_w, _ in specs):
        inner_h, inner_w, color = specs[0]
        if inner_w < max_w:
            specs[0] = (inner_h, inner_w + 1, color)
        else:
            specs[0] = (inner_h, inner_w - 1, color)
    specs.sort(key=lambda item: item[0] * item[1], reverse=True)
    return specs


def _place_regions(
    grid: np.ndarray,
    occupied: np.ndarray,
    bg: int,
    border: int,
    specs: list[tuple[int, int, int]],
    rng: random.Random,
) -> bool:
    h, w = grid.shape
    for inner_h, inner_w, seed in specs:
        total_h = inner_h + 2
        total_w = inner_w + 2
        if h - total_h - 1 < 1 or w - total_w - 1 < 1:
            return False
        placed = False
        for _ in range(300):
            r0 = rng.randint(1, h - total_h - 1)
            c0 = rng.randint(1, w - total_w - 1)
            pr0 = max(0, r0 - 1)
            pc0 = max(0, c0 - 1)
            pr1 = min(h, r0 + total_h + 1)
            pc1 = min(w, c0 + total_w + 1)
            if np.any(occupied[pr0:pr1, pc0:pc1]):
                continue

            grid[r0 : r0 + total_h, c0 : c0 + total_w] = bg
            grid[r0, c0 : c0 + total_w] = border
            grid[r0 + total_h - 1, c0 : c0 + total_w] = border
            grid[r0 : r0 + total_h, c0] = border
            grid[r0 : r0 + total_h, c0 + total_w - 1] = border

            sr = rng.randint(r0 + 1, r0 + total_h - 2)
            sc = rng.randint(c0 + 1, c0 + total_w - 2)
            grid[sr, sc] = seed
            occupied[pr0:pr1, pc0:pc1] = True
            placed = True
            break
        if not placed:
            return False
    return True


def _validate_candidate(
    grid: np.ndarray, bg: int, seed_colors: list[int]
) -> bool:
    if _most_common_color(grid) != bg:
        return False
    try:
        border, mapping, _ = _find_legend(grid, bg)
        if border != _BORDER_COLOR:
            return False
        regions = _find_regions(grid, bg, border)
        if len(regions) != len(seed_colors):
            return False
        if {seed for *_, seed in regions} != set(seed_colors):
            return False
        out = _solve(grid)
        for r0, c0, r1, c1, seed in regions:
            expected = _tile_pattern(mapping[seed], r1 - r0 - 1, c1 - c0 - 1)
            if not np.array_equal(out[r0 + 1 : r1, c0 + 1 : c1], expected):
                return False
    except AssertionError:
        return False
    return True


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    side = rng.choice(["top", "bottom", "left", "right"])
    bg = rng.choice([color for color in range(10) if color != _BORDER_COLOR])
    seed_count = rng.randint(2, 4)
    seed_pool = [color for color in range(10) if color not in (bg, _BORDER_COLOR)]
    rng.shuffle(seed_pool)
    seed_colors = seed_pool[:seed_count]
    masks = _random_pattern_masks(rng, seed_count)
    legend = _build_legend(bg, _BORDER_COLOR, side, seed_colors, masks)

    if side in ("top", "bottom"):
        min_h = 20
        max_h = 30
        min_w = max(legend.shape[1] + 2, 20)
        max_w = max(legend.shape[1] + 10, 30)
    else:
        min_h = max(legend.shape[0] + 2, 20)
        max_h = max(legend.shape[0] + 10, 30)
        min_w = 20
        max_w = 30

    h = rng.randint(min_h, max_h)
    w = rng.randint(min_w, max_w)
    grid = np.full((h, w), bg, dtype=int)

    occupied = np.zeros((h, w), dtype=bool)
    lr0, lc0 = _place_patch(grid, legend, side, rng)
    lph, lpw = legend.shape
    occupied[
        max(0, lr0 - 1) : min(h, lr0 + lph + 1),
        max(0, lc0 - 1) : min(w, lc0 + lpw + 1),
    ] = True

    specs = _sample_region_specs(rng, seed_colors, side)
    if not _place_regions(grid, occupied, bg, _BORDER_COLOR, specs, rng):
        return None
    if not _validate_candidate(grid, bg, seed_colors):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(_GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid sample after {_GEN_RETRIES} attempts")
