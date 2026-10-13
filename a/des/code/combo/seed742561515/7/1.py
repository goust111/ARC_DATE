import random
from collections import Counter, deque

import numpy as np


BLACK = 0
GREY = 5
SEED_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)

GEN_RETRIES = 400
PARTITION_TRIES = 200
SUBSET_TRIES = 600

ROW_COUNT_OPTIONS = (2, 3, 4)
COL_COUNT_OPTIONS = (2, 3, 4)
ROW_COUNT_WEIGHTS = (2, 4, 3)
COL_COUNT_WEIGHTS = (2, 4, 3)
CELL_SIZE_VALUES = (2, 3, 4, 5, 6, 7)
MARGIN_OPTIONS = (0, 1, 1, 2)
MODE_WEIGHTS = {"chain": 3, "mixed": 4, "separated": 2}
SEED_COUNT_WEIGHTS = {2: 1, 3: 4, 4: 4, 5: 2}


def _components(mask: np.ndarray) -> tuple[np.ndarray, list[list[tuple[int, int]]]]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    labels = np.zeros((h, w), dtype=int)
    comps: list[list[tuple[int, int]]] = []
    comp_id = 0

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or labels[r, c] != 0:
                continue
            comp_id += 1
            labels[r, c] = comp_id
            q = deque([(r, c)])
            pts: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                pts.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and labels[nr, nc] == 0:
                        labels[nr, nc] = comp_id
                        q.append((nr, nc))
            comps.append(pts)
    return labels, comps


def _bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    return min(rows), min(cols), max(rows), max(cols)


def _infer_grey(grid: np.ndarray) -> int:
    counts = Counter(int(v) for v in np.ravel(grid) if int(v) != BLACK)
    if not counts:
        return GREY
    return int(counts.most_common(1)[0][0])


def _extract_regions(
    grid: np.ndarray,
    grey: int | None = None,
) -> tuple[np.ndarray, list[dict], int]:
    g = np.asarray(grid, dtype=int)
    if grey is None:
        grey = _infer_grey(g)
    h, w = g.shape
    labels, comps = _components(g != grey)
    regions: list[dict] = []

    for comp_id, pts in enumerate(comps, start=1):
        if any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in pts):
            continue
        r0, c0, r1, c1 = _bbox(pts)
        seed_pts = [(r, c) for r, c in pts if int(g[r, c]) not in (BLACK, grey)]
        seed_color = int(g[seed_pts[0]]) if len(seed_pts) == 1 else None
        regions.append(
            {
                "id": comp_id,
                "cells": pts,
                "bbox": (r0, c0, r1, c1),
                "area": len(pts),
                "seed_pts": seed_pts,
                "seed_color": seed_color,
            }
        )
    return labels, regions, int(grey)


def _shared_wall_cells(
    labels: np.ndarray,
    grey_mask: np.ndarray,
    region_a: dict,
    region_b: dict,
) -> list[tuple[int, int]]:
    h, w = labels.shape
    a_id = int(region_a["id"])
    b_id = int(region_b["id"])
    candidate = np.zeros((h, w), dtype=bool)

    for r in range(h):
        for c in range(w):
            if not grey_mask[r, c]:
                continue
            matched = False
            if 0 < c < w - 1:
                left = int(labels[r, c - 1])
                right = int(labels[r, c + 1])
                if left and right and {left, right} == {a_id, b_id}:
                    matched = True
            if not matched and 0 < r < h - 1:
                up = int(labels[r - 1, c])
                down = int(labels[r + 1, c])
                if up and down and {up, down} == {a_id, b_id}:
                    matched = True
            if matched:
                candidate[r, c] = True

    if not candidate.any():
        return []

    _, comps = _components(candidate)
    if len(comps) == 1:
        return comps[0]

    ar0, ac0, ar1, ac1 = region_a["bbox"]
    br0, bc0, br1, bc1 = region_b["bbox"]
    target_r = (ar0 + ar1 + br0 + br1) / 4.0
    target_c = (ac0 + ac1 + bc0 + bc1) / 4.0

    def score(pts: list[tuple[int, int]]) -> tuple[float, int]:
        rr0, cc0, rr1, cc1 = _bbox(pts)
        center_r = (rr0 + rr1) / 2.0
        center_c = (cc0 + cc1) / 2.0
        return (abs(center_r - target_r) + abs(center_c - target_c), -len(pts))

    return min(comps, key=score)


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    labels, regions, grey = _extract_regions(g)
    grey_mask = g == grey
    out = g.copy()

    seeded_regions: list[dict] = []
    for region in regions:
        if len(region["seed_pts"]) != 1:
            continue
        color = int(region["seed_color"])
        for r, c in region["cells"]:
            out[r, c] = color
        seeded_regions.append(region)

    seeded_regions.sort(
        key=lambda region: (
            int(region["area"]),
            int(region["bbox"][0]),
            int(region["bbox"][1]),
            int(region["bbox"][2]),
            int(region["bbox"][3]),
        )
    )

    for start_region, next_region in zip(seeded_regions, seeded_regions[1:]):
        for r, c in _shared_wall_cells(labels, grey_mask, start_region, next_region):
            out[r, c] = int(start_region["seed_color"])

    return out


def _line_positions(sizes: list[int], margin: int) -> list[int]:
    positions = [margin]
    cur = margin
    for size in sizes:
        cur += size + 1
        positions.append(cur)
    return positions


def _draw_partition(
    grid: np.ndarray,
    line_rows: list[int],
    line_cols: list[int],
) -> None:
    top, bottom = line_rows[0], line_rows[-1]
    left, right = line_cols[0], line_cols[-1]
    grid[top : bottom + 1, line_cols] = GREY
    grid[line_rows, left : right + 1] = GREY


def _build_cells(
    line_rows: list[int],
    line_cols: list[int],
) -> list[dict]:
    cells: list[dict] = []
    for ri in range(len(line_rows) - 1):
        for ci in range(len(line_cols) - 1):
            r0 = line_rows[ri] + 1
            r1 = line_rows[ri + 1] - 1
            c0 = line_cols[ci] + 1
            c1 = line_cols[ci + 1] - 1
            cells.append(
                {
                    "ri": ri,
                    "ci": ci,
                    "bbox": (r0, c0, r1, c1),
                    "area": (r1 - r0 + 1) * (c1 - c0 + 1),
                }
            )
    return cells


def _adjacent(cell_a: dict, cell_b: dict) -> bool:
    return abs(int(cell_a["ri"]) - int(cell_b["ri"])) + abs(int(cell_a["ci"]) - int(cell_b["ci"])) == 1


def _mode_for_order(cells: list[dict]) -> str:
    flags = [_adjacent(cells[i], cells[i + 1]) for i in range(len(cells) - 1)]
    if all(flags):
        return "chain"
    if any(flags):
        return "mixed"
    return "separated"


def _pick_seed_count(rng: random.Random, max_k: int) -> int:
    options = [k for k in range(2, max_k + 1)]
    weights = [SEED_COUNT_WEIGHTS.get(k, 1) for k in options]
    return int(rng.choices(options, weights=weights, k=1)[0])


def _pick_seeded_cells(rng: random.Random, cells: list[dict]) -> tuple[list[dict], str] | None:
    max_k = min(5, len(cells) - 1)
    if max_k < 2:
        return None

    found: dict[str, list[dict]] = {}
    for _ in range(SUBSET_TRIES):
        k = _pick_seed_count(rng, max_k)
        subset = rng.sample(cells, k)
        areas = [int(cell["area"]) for cell in subset]
        if len(set(areas)) != k:
            continue
        ordered = sorted(subset, key=lambda cell: (int(cell["area"]), int(cell["ri"]), int(cell["ci"])))
        mode = _mode_for_order(ordered)
        if mode not in found or rng.random() < 0.25:
            found[mode] = ordered
        if len(found) == 3:
            break

    if not found:
        return None

    modes = list(found)
    weights = [MODE_WEIGHTS[mode] for mode in modes]
    chosen_mode = str(rng.choices(modes, weights=weights, k=1)[0])
    return found[chosen_mode], chosen_mode


def _paint_seeds(rng: random.Random, grid: np.ndarray, seeded_cells: list[dict]) -> None:
    colors = list(rng.sample(SEED_COLORS, len(seeded_cells)))
    rng.shuffle(colors)
    for cell, color in zip(seeded_cells, colors, strict=False):
        r0, c0, r1, c1 = cell["bbox"]
        rr = rng.randint(r0, r1)
        cc = rng.randint(c0, c1)
        grid[rr, cc] = int(color)


def _validate_generated_input(grid: np.ndarray) -> bool:
    if not isinstance(grid, np.ndarray) or grid.ndim != 2:
        return False

    labels, regions, grey = _extract_regions(grid, GREY)
    if grey != GREY:
        return False
    if labels.shape != grid.shape:
        return False

    seeded = [region for region in regions if len(region["seed_pts"]) == 1]
    if len(seeded) < 2:
        return False
    if len(seeded) >= len(regions):
        return False

    areas = [int(region["area"]) for region in seeded]
    colors = [int(region["seed_color"]) for region in seeded]
    if len(set(areas)) != len(areas):
        return False
    if len(set(colors)) != len(colors):
        return False

    for region in regions:
        r0, c0, r1, c1 = region["bbox"]
        if int(region["area"]) != (r1 - r0 + 1) * (c1 - c0 + 1):
            return False

    try:
        out = main(grid)
    except Exception:
        return False
    return isinstance(out, np.ndarray) and out.shape == grid.shape


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    for _ in range(PARTITION_TRIES):
        n_rows = int(rng.choices(ROW_COUNT_OPTIONS, weights=ROW_COUNT_WEIGHTS, k=1)[0])
        n_cols = int(rng.choices(COL_COUNT_OPTIONS, weights=COL_COUNT_WEIGHTS, k=1)[0])
        row_sizes = list(rng.sample(CELL_SIZE_VALUES, n_rows))
        col_sizes = list(rng.sample(CELL_SIZE_VALUES, n_cols))
        rng.shuffle(row_sizes)
        rng.shuffle(col_sizes)
        margin = int(rng.choice(MARGIN_OPTIONS))

        line_rows = _line_positions(row_sizes, margin)
        line_cols = _line_positions(col_sizes, margin)
        height = line_rows[-1] + margin + 1
        width = line_cols[-1] + margin + 1

        cells = _build_cells(line_rows, line_cols)
        picked = _pick_seeded_cells(rng, cells)
        if picked is None:
            continue
        seeded_cells, _mode = picked

        grid = np.full((height, width), BLACK, dtype=int)
        _draw_partition(grid, line_rows, line_cols)
        _paint_seeds(rng, grid, seeded_cells)

        if _validate_generated_input(grid):
            return grid
    return None


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: failed to create a valid puzzle instance")
