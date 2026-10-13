from __future__ import annotations

from collections import Counter
import random

import numpy as np


FRAME = 5
PATTERN_COLORS = (1, 2, 3, 4)
CORRUPT_COLORS = (0, 1, 2, 3, 4)
ROW_PARTS = 2
COL_PART_OPTIONS = (2, 3)
ROW_SIZE_OPTIONS = (6, 8)
COL_SIZE_OPTIONS = (6, 8)
GEN_RETRIES = 400


def _mode_color(values: np.ndarray) -> int | None:
    counts = Counter(int(v) for v in values)
    best_count = max(counts.values())
    winners = [color for color, count in counts.items() if count == best_count]
    if len(winners) != 1:
        return None
    return winners[0]


def _frame_lines(grid: np.ndarray) -> tuple[int, list[int], list[int]]:
    h, w = grid.shape
    row_lines = [(r, int(grid[r, 0])) for r in range(h) if np.all(grid[r, :] == grid[r, 0])]
    col_lines = [(c, int(grid[0, c])) for c in range(w) if np.all(grid[:, c] == grid[0, c])]
    colors = sorted({color for _idx, color in row_lines} & {color for _idx, color in col_lines})
    assert colors, "expected full-row/full-col frame separators"

    frame = max(
        colors,
        key=lambda color: sum(int(c == color) for _r, c in row_lines) + sum(int(c == color) for _c, c in col_lines),
    )
    rows = [-1] + sorted(r for r, color in row_lines if color == frame) + [h]
    cols = [-1] + sorted(c for c, color in col_lines if color == frame) + [w]
    return int(frame), rows, cols


def _candidate_keys(region: np.ndarray) -> list[tuple[tuple[int, int, int, int], int, int, np.ndarray]]:
    h, w = region.shape
    candidates = []
    for ph in range(1, h + 1):
        if h % ph != 0 or h // ph < 2:
            continue
        for pw in range(1, w + 1):
            if w % pw != 0 or w // pw < 2:
                continue
            tile = np.zeros((ph, pw), dtype=int)
            valid = True
            for r in range(ph):
                for c in range(pw):
                    color = _mode_color(region[r::ph, c::pw].ravel())
                    if color is None:
                        valid = False
                        break
                    tile[r, c] = color
                if not valid:
                    break
            if not valid:
                continue
            rebuilt = np.tile(tile, (h // ph, w // pw))
            key = (int(np.sum(rebuilt != region)), ph * pw, ph, pw)
            candidates.append((key, ph, pw, rebuilt))
    assert candidates, "expected at least one periodic candidate"
    return candidates


def _repair_region(region: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
    candidates = _candidate_keys(region)
    best_key, best_ph, best_pw, best_grid = min(candidates, key=lambda item: item[0])
    _ = best_key
    return best_grid, (best_ph, best_pw)


def _sources_from_grid(grid: np.ndarray) -> tuple[np.ndarray, list[tuple[int, int, int, tuple[int, int, int, int]]]]:
    g = np.asarray(grid, dtype=int)
    frame, rows, cols = _frame_lines(g)
    repaired = g.copy()
    sources: list[tuple[int, int, int, tuple[int, int, int, int]]] = []

    for i in range(len(rows) - 1):
        rr0 = rows[i] + 1
        rr1 = rows[i + 1] - 1
        if rr0 > rr1:
            continue
        for j in range(len(cols) - 1):
            cc0 = cols[j] + 1
            cc1 = cols[j + 1] - 1
            if cc0 > cc1:
                continue
            region = g[rr0 : rr1 + 1, cc0 : cc1 + 1]
            clean, _period = _repair_region(region)
            repaired[rr0 : rr1 + 1, cc0 : cc1 + 1] = clean
            diff = np.argwhere(clean != region)
            for dr, dc in diff:
                r = rr0 + int(dr)
                c = cc0 + int(dc)
                sources.append((r, c, int(clean[int(dr), int(dc)]), (rr0, cc0, rr1, cc1)))

    repaired[g == frame] = frame
    return repaired, sources


def _apply_rays(
    repaired: np.ndarray,
    sources: list[tuple[int, int, int, tuple[int, int, int, int]]],
) -> np.ndarray:
    out = repaired.copy()
    claimed = np.zeros(out.shape, dtype=bool)
    for r, c, color, (r0, c0, r1, c1) in sorted(sources, key=lambda item: (item[0], item[1])):
        for cc in range(c0, c1 + 1):
            if not claimed[r, cc]:
                out[r, cc] = color
                claimed[r, cc] = True
        for rr in range(r0, r1 + 1):
            if not claimed[rr, c]:
                out[rr, c] = color
                claimed[rr, c] = True
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repair each periodic frame partition, then radiate the repaired difference pixels as orthogonal rays.

    Input structure:
    - Grey frame lines form complete rows and complete columns, partitioning the grid into 2x2 or 2x3 rectangular regions.
    - Inside each region, a small colored tile repeats periodically and 1-2 interior pixels are corrupted away from that periodic pattern.
    - Frame pixels stay grey and the non-frame colors inside regions carry the periodic information.

    Rule:
    - For each region, infer the smallest repeating tile by majority vote over each residue class and repair the region to that clean periodic fill.
    - Compare the repaired region with the input and treat every changed pixel as a ray source colored by its repaired value.
    - From each source, paint straight rays upward, downward, leftward, and rightward until the region boundary frame line; if multiple rays compete for one cell, the source with smaller row wins, then smaller column.

    Key insight:
    - The repaired colors determine both where the difference sources are and what color each source's four-way radiation should use, so the ray step depends on the repair step.
    """
    repaired, sources = _sources_from_grid(np.asarray(grid, dtype=int))
    return _apply_rays(repaired, sources)


def _tile_is_primitive(tile: np.ndarray) -> bool:
    ph, pw = tile.shape
    for cand_h in range(1, ph + 1):
        if ph % cand_h != 0:
            continue
        for cand_w in range(1, pw + 1):
            if pw % cand_w != 0:
                continue
            if cand_h == ph and cand_w == pw:
                continue
            small = tile[:cand_h, :cand_w]
            rebuilt = np.tile(small, (ph // cand_h, pw // cand_w))
            if np.array_equal(rebuilt, tile):
                return False
    return len(set(map(int, tile.ravel()))) >= 2


def _sample_tile(ph: int, pw: int) -> np.ndarray:
    for _ in range(200):
        n_colors = random.randint(2, len(PATTERN_COLORS))
        palette = random.sample(list(PATTERN_COLORS), n_colors)
        tile = np.array([[random.choice(palette) for _c in range(pw)] for _r in range(ph)], dtype=int)
        if _tile_is_primitive(tile):
            return tile
    raise RuntimeError("failed to sample a primitive tile")


def _region_shape_options(width: int) -> list[tuple[int, int]]:
    options = [(2, 2)]
    if width == 6:
        options.append((2, 3))
    return options


def _render_framed_clean(
    row_sizes: list[int],
    col_sizes: list[int],
    specs: list[list[dict]],
) -> tuple[np.ndarray, list[dict]]:
    h = sum(row_sizes) + len(row_sizes) + 1
    w = sum(col_sizes) + len(col_sizes) + 1
    grid = np.full((h, w), FRAME, dtype=int)
    regions: list[dict] = []

    r0 = 1
    for i, rh in enumerate(row_sizes):
        c0 = 1
        for j, cw in enumerate(col_sizes):
            spec = specs[i][j]
            tiled = np.tile(spec["tile"], (rh // spec["tile"].shape[0], cw // spec["tile"].shape[1]))
            grid[r0 : r0 + rh, c0 : c0 + cw] = tiled
            regions.append(
                {
                    "bounds": (r0, c0, r0 + rh - 1, c0 + cw - 1),
                    "clean": tiled.copy(),
                    "tile_shape": spec["tile"].shape,
                }
            )
            c0 += cw + 1
        r0 += rh + 1
    return grid, regions


def _non_corner_cells(shape: tuple[int, int]) -> list[tuple[int, int]]:
    h, w = shape
    cells = [(r, c) for r in range(h) for c in range(w)]
    corners = {(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)}
    return [cell for cell in cells if cell not in corners]


def _pick_damage_cells(clean: np.ndarray, count: int, require_cross: bool = False) -> list[tuple[int, int]]:
    h, w = clean.shape
    candidates = _non_corner_cells((h, w))
    if count == 1:
        return [random.choice(candidates)]

    pairs = []
    for idx, a in enumerate(candidates):
        for b in candidates[idx + 1 :]:
            if (a[0] % 2, a[1] % 3) == (b[0] % 2, b[1] % 3):
                continue
            if require_cross and (a[0] == b[0] or a[1] == b[1]):
                continue
            if require_cross and int(clean[a]) == int(clean[b]):
                continue
            pairs.append((a, b))
    if not pairs:
        for idx, a in enumerate(candidates):
            for b in candidates[idx + 1 :]:
                if a[0] != b[0] and a[1] != b[1]:
                    pairs.append((a, b))
    assert pairs
    return list(random.choice(pairs))


def _corrupt_color(correct: int) -> int:
    choices = [color for color in CORRUPT_COLORS if color != correct]
    if 0 in choices and random.random() < 0.5:
        return 0
    return random.choice(choices)


def _expected_from_regions(clean_grid: np.ndarray, regions: list[dict]) -> tuple[np.ndarray, list[tuple[int, int, int, tuple[int, int, int, int]]]]:
    repaired = clean_grid.copy()
    sources: list[tuple[int, int, int, tuple[int, int, int, int]]] = []
    for region in regions:
        r0, c0, r1, c1 = region["bounds"]
        for dr, dc in region["damages"]:
            rr = r0 + dr
            cc = c0 + dc
            repaired[rr, cc] = int(region["clean"][dr, dc])
            sources.append((rr, cc, int(region["clean"][dr, dc]), (r0, c0, r1, c1)))
    return _apply_rays(repaired, sources), sources


def _has_overlap(sources: list[tuple[int, int, int, tuple[int, int, int, int]]]) -> bool:
    cover: dict[tuple[int, int], int] = {}
    for r, c, _color, (r0, c0, r1, c1) in sources:
        for cc in range(c0, c1 + 1):
            cover[(r, cc)] = cover.get((r, cc), 0) + 1
        for rr in range(r0, r1 + 1):
            cover[(rr, c)] = cover.get((rr, c), 0) + 1
    return any(count >= 2 for count in cover.values())


def _semantic_checks(inp: np.ndarray, expected: np.ndarray, clean: np.ndarray, regions: list[dict]) -> bool:
    if np.array_equal(inp, expected):
        return False
    if np.sum(expected != clean) <= sum(len(region["damages"]) for region in regions):
        return False

    source_colors = {int(region["clean"][dr, dc]) for region in regions for dr, dc in region["damages"]}
    if len(source_colors) < 2:
        return False

    cross_regions = [
        region
        for region in regions
        if len(region["damages"]) == 2
        and region["damages"][0][0] != region["damages"][1][0]
        and region["damages"][0][1] != region["damages"][1][1]
    ]
    if not cross_regions:
        return False

    for region in regions:
        r0, c0, r1, c1 = region["bounds"]
        noisy = inp[r0 : r1 + 1, c0 : c1 + 1]
        candidates = _candidate_keys(noisy)
        best_key = min(candidate[0] for candidate in candidates)
        winners = [(ph, pw) for key, ph, pw, _grid in candidates if key == best_key]
        if winners != [region["tile_shape"]]:
            return False

    repaired, sources = _sources_from_grid(inp)
    if not _has_overlap(sources):
        return False
    if not np.array_equal(repaired[clean == FRAME], clean[clean == FRAME]):
        return False
    return True


def _attempt_generate() -> np.ndarray | None:
    n_cols = random.choice(COL_PART_OPTIONS)
    row_sizes = [random.choice(ROW_SIZE_OPTIONS) for _ in range(ROW_PARTS)]
    col_sizes = [random.choice(COL_SIZE_OPTIONS) for _ in range(n_cols)]
    forced_col = random.randrange(n_cols)
    col_sizes[forced_col] = 6

    specs: list[list[dict]] = [[{} for _ in range(n_cols)] for _ in range(ROW_PARTS)]
    forced_23 = (random.randrange(ROW_PARTS), forced_col)
    forced_22 = (random.randrange(ROW_PARTS), random.randrange(n_cols))
    while forced_22 == forced_23:
        forced_22 = (random.randrange(ROW_PARTS), random.randrange(n_cols))

    for i in range(ROW_PARTS):
        for j in range(n_cols):
            options = _region_shape_options(col_sizes[j])
            if (i, j) == forced_23:
                ph, pw = (2, 3)
            elif (i, j) == forced_22:
                ph, pw = (2, 2)
            else:
                ph, pw = random.choice(options)
            specs[i][j] = {"tile": _sample_tile(ph, pw)}

    clean_grid, regions = _render_framed_clean(row_sizes, col_sizes, specs)
    inp = clean_grid.copy()

    overlap_region_idx = random.randrange(len(regions))
    single_region_idx = random.randrange(len(regions))
    while single_region_idx == overlap_region_idx:
        single_region_idx = random.randrange(len(regions))

    for idx, region in enumerate(regions):
        clean = region["clean"]
        if idx == overlap_region_idx:
            damages = _pick_damage_cells(clean, 2, require_cross=True)
        elif idx == single_region_idx:
            damages = _pick_damage_cells(clean, 1)
        else:
            damages = _pick_damage_cells(clean, random.choice((1, 2)))

        region["damages"] = damages
        for dr, dc in damages:
            r0, c0, _r1, _c1 = region["bounds"]
            rr = r0 + dr
            cc = c0 + dc
            inp[rr, cc] = _corrupt_color(int(clean[dr, dc]))

    expected, sources = _expected_from_regions(clean_grid, regions)
    if not _has_overlap(sources):
        return None
    if not _semantic_checks(inp, expected, clean_grid, regions):
        return None

    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid periodic-repair radiation puzzle")
