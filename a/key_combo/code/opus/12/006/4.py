from __future__ import annotations

import random
from collections import deque

import numpy as np


GREY = 5
WALL = 2
SEED_COLORS = (1, 3, 4)
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
GEN_RETRY_LIMIT = 500

_GEN_CALL_COUNT = 0


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= int(r) < int(shape[0]) and 0 <= int(c) < int(shape[1])


def _neighbors4(r: int, c: int):
    for dr, dc in DIR4:
        yield int(r) + int(dr), int(c) + int(dc)


def _label_mask(mask: np.ndarray) -> tuple[np.ndarray, list[list[tuple[int, int]]]]:
    h, w = mask.shape
    labels = np.zeros((h, w), dtype=int)
    regions: list[list[tuple[int, int]]] = []
    next_id = 1
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or int(labels[r, c]) != 0:
                continue
            q = deque([(r, c)])
            labels[r, c] = next_id
            cells: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                cells.append((int(rr), int(cc)))
                for nr, nc in _neighbors4(rr, cc):
                    if _in_bounds(mask.shape, nr, nc) and bool(mask[nr, nc]) and int(labels[nr, nc]) == 0:
                        labels[nr, nc] = next_id
                        q.append((nr, nc))
            regions.append(cells)
            next_id += 1
    return labels, regions


def _find_color_components(grid: np.ndarray, allowed_mask: np.ndarray | None = None) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color in (GREY, WALL) or bool(seen[r, c]):
                continue
            if allowed_mask is not None and not bool(allowed_mask[r, c]):
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((int(rr), int(cc)))
                for nr, nc in _neighbors4(rr, cc):
                    if not _in_bounds(grid.shape, nr, nc):
                        continue
                    if bool(seen[nr, nc]) or int(grid[nr, nc]) != color:
                        continue
                    if allowed_mask is not None and not bool(allowed_mask[nr, nc]):
                        continue
                    seen[nr, nc] = True
                    q.append((nr, nc))
            comps.append({"color": color, "coords": coords})
    return comps


def _denoise(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    h, w = g.shape
    for r in range(h):
        for c in range(w):
            color = int(g[r, c])
            if color in (GREY, WALL):
                continue
            same_color_neighbor = False
            wall_neighbor = False
            for nr, nc in _neighbors4(r, c):
                if not _in_bounds(g.shape, nr, nc):
                    continue
                if int(g[nr, nc]) == color:
                    same_color_neighbor = True
                if int(g[nr, nc]) == WALL:
                    wall_neighbor = True
            if not same_color_neighbor and not wall_neighbor:
                out[r, c] = GREY
    return out


def _region_infos(grid: np.ndarray) -> tuple[np.ndarray, list[dict]]:
    labels, regions = _label_mask(np.asarray(grid) != WALL)
    infos: list[dict] = []
    h, w = grid.shape
    for rid, cells in enumerate(regions, start=1):
        cell_set = set(cells)
        wall_adj = set()
        interior = set()
        for r, c in cells:
            near_wall = False
            for nr, nc in _neighbors4(r, c):
                if _in_bounds(grid.shape, nr, nc) and int(grid[nr, nc]) == WALL:
                    near_wall = True
                    break
            if near_wall:
                wall_adj.add((r, c))
            if not near_wall and 0 < r < h - 1 and 0 < c < w - 1:
                interior.add((r, c))
        infos.append(
            {
                "id": rid,
                "cells": cells,
                "set": cell_set,
                "wall_adjacent": wall_adj,
                "interior": interior,
                "area": len(cells),
            }
        )
    return labels, infos


def _seed_components_for_region(cleaned: np.ndarray, region_info: dict) -> list[dict]:
    mask = np.zeros_like(cleaned, dtype=bool)
    for r, c in region_info["cells"]:
        mask[r, c] = True
    comps = _find_color_components(cleaned, allowed_mask=mask)
    seeds: list[dict] = []
    for comp in comps:
        coords = sorted(comp["coords"])
        anchor = coords[0]
        ar, ac = anchor
        same_color_count = int(np.sum(cleaned[max(0, ar - 1) : ar + 2, max(0, ac - 1) : ac + 2] == int(comp["color"])))
        seeds.append(
            {
                "color": int(comp["color"]),
                "coords": coords,
                "anchor": (int(ar), int(ac)),
                "score": same_color_count,
            }
        )
    return seeds


def _solve_details(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    cleaned = _denoise(g)
    _labels, regions = _region_infos(cleaned)

    out = cleaned.copy()
    region_details: list[dict] = []
    for region in regions:
        seeds = _seed_components_for_region(cleaned, region)
        colors = {int(seed["color"]) for seed in seeds}
        if not seeds:
            fill_color = GREY
            winner = None
        elif len(colors) == 1:
            fill_color = int(next(iter(colors)))
            winner = None
        else:
            winner = max(seeds, key=lambda seed: (int(seed["score"]), -int(seed["anchor"][0]), -int(seed["anchor"][1])))
            fill_color = int(winner["color"])

        for r, c in region["cells"]:
            out[r, c] = fill_color

        region_details.append(
            {
                "id": int(region["id"]),
                "area": int(region["area"]),
                "seed_count": len(seeds),
                "seed_colors": sorted(colors),
                "fill_color": fill_color,
                "winner": winner,
                "seeds": seeds,
            }
        )

    out[g == WALL] = WALL
    return {"cleaned": cleaned, "output": out, "regions": region_details}


def main(grid: np.ndarray) -> np.ndarray:
    """
    Denoise intra-wall seeds, then flood each wall-bounded region with its surviving winner color.

    Input structure:
    - A grey background grid contains a one-cell-thick red wall network that partitions the non-wall area into 4-connected regions.
    - Inside regions, blue, green, and yellow foreground pixels appear as small seed blobs of size one or two.
    - Some extra foreground pixels are isolated noise: they have no same-color 4-neighbor and are not 4-adjacent to any red wall.

    Rule:
    - First denoise the grid: any non-grey, non-red pixel with neither a same-color 4-neighbor nor a red-wall 4-neighbor is restored to grey.
    - Treat every surviving monochrome 4-connected blob as a seed, and for each non-wall region gather the seeds lying inside it.
    - A region with no surviving seeds stays grey; a region whose surviving seeds all share one color is fully filled with that color.
    - If a region contains multiple seed colors, score each seed by counting same-color pixels in the seed anchor's 3x3 neighborhood, choose the highest score, and break ties by the smallest anchor row then smallest anchor column; fill the whole region with that winning color.

    Key insight:
    - The denoising step is part of the logic, not optional cleanup: it determines which seeds survive to activate a region and which colors are even eligible to win the region-wide flood fill.
    """
    return _solve_details(np.asarray(grid, dtype=int))["output"]


def _transform_grid(grid: np.ndarray, op: int) -> np.ndarray:
    arr = np.asarray(grid, dtype=int)
    op = int(op) % 8
    if op == 0:
        return arr.copy()
    if op == 1:
        return np.rot90(arr, 1)
    if op == 2:
        return np.rot90(arr, 2)
    if op == 3:
        return np.rot90(arr, 3)
    if op == 4:
        return np.fliplr(arr)
    if op == 5:
        return np.flipud(arr)
    if op == 6:
        return np.transpose(arr)
    return np.fliplr(np.transpose(arr))


def _build_wall_layout(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(13, 18)
    w = rng.randint(13, 18)
    grid = np.full((h, w), GREY, dtype=int)

    trunk_c = rng.randint(4, w - 5)
    left_row = rng.randint(3, h - 4)
    right_row = rng.randint(3, h - 4)

    grid[:, trunk_c] = WALL
    grid[left_row, : trunk_c + 1] = WALL
    grid[right_row, trunk_c:] = WALL

    extra_side = rng.choice(["left", "right", None])
    if extra_side is not None:
        candidates = [r for r in range(2, h - 2) if abs(r - (left_row if extra_side == "left" else right_row)) >= 2]
        if candidates:
            extra_row = rng.choice(candidates)
            if extra_side == "left":
                grid[extra_row, : trunk_c + 1] = WALL
            else:
                grid[extra_row, trunk_c:] = WALL

    infos = _region_infos(grid)[1]
    if len(infos) < 4:
        return None
    if min(info["area"] for info in infos) < 10:
        return None
    return grid


def _is_clear(candidate: list[tuple[int, int]], occupied: set[tuple[int, int]]) -> bool:
    for r, c in candidate:
        if (r, c) in occupied:
            return False
        for rr, cc in occupied:
            if max(abs(int(r) - int(rr)), abs(int(c) - int(cc))) <= 1:
                return False
    return True


def _sample_single(region: dict, occupied: set[tuple[int, int]], rng: random.Random, *, require_wall_adj: bool, forbid_wall_adj: bool) -> tuple[int, int] | None:
    cells = list(region["cells"])
    rng.shuffle(cells)
    for r, c in cells:
        if require_wall_adj and (r, c) not in region["wall_adjacent"]:
            continue
        if forbid_wall_adj and (r, c) in region["wall_adjacent"]:
            continue
        if forbid_wall_adj and (r, c) not in region["interior"]:
            continue
        if _is_clear([(r, c)], occupied):
            return (int(r), int(c))
    return None


def _sample_pair(region: dict, occupied: set[tuple[int, int]], rng: random.Random, *, require_wall_adj: bool, forbid_wall_adj: bool) -> list[tuple[int, int]] | None:
    candidates: list[list[tuple[int, int]]] = []
    cell_set = region["set"]
    for r, c in region["cells"]:
        for dr, dc in ((1, 0), (0, 1)):
            rr = int(r) + int(dr)
            cc = int(c) + int(dc)
            pair = [(int(r), int(c)), (int(rr), int(cc))]
            if (rr, cc) not in cell_set:
                continue
            if require_wall_adj and not any(cell in region["wall_adjacent"] for cell in pair):
                continue
            if forbid_wall_adj and any(cell in region["wall_adjacent"] for cell in pair):
                continue
            if forbid_wall_adj and not all(cell in region["interior"] for cell in pair):
                continue
            if _is_clear(pair, occupied):
                candidates.append(pair)
    if not candidates:
        return None
    rng.shuffle(candidates)
    return candidates[0]


def _paint_cells(grid: np.ndarray, cells: list[tuple[int, int]], color: int, occupied: set[tuple[int, int]]) -> None:
    for r, c in cells:
        grid[int(r), int(c)] = int(color)
        occupied.add((int(r), int(c)))


def _place_blob(grid: np.ndarray, region: dict, color: int, size: int, occupied: set[tuple[int, int]], rng: random.Random, *, require_wall_adj: bool = False, forbid_wall_adj: bool = False) -> bool:
    if int(size) == 1:
        cell = _sample_single(region, occupied, rng, require_wall_adj=require_wall_adj, forbid_wall_adj=forbid_wall_adj)
        if cell is None:
            return False
        _paint_cells(grid, [cell], color, occupied)
        return True
    pair = _sample_pair(region, occupied, rng, require_wall_adj=require_wall_adj, forbid_wall_adj=forbid_wall_adj)
    if pair is None:
        return False
    _paint_cells(grid, pair, color, occupied)
    return True


def _conflict_mode_for_call(rng: random.Random) -> str:
    global _GEN_CALL_COUNT
    mode = "score" if _GEN_CALL_COUNT % 2 == 0 else "tie"
    _GEN_CALL_COUNT += 1
    if _GEN_CALL_COUNT > 4 and rng.random() < 0.5:
        mode = rng.choice(["score", "tie"])
    return mode


def _place_conflict_region(grid: np.ndarray, region: dict, occupied: set[tuple[int, int]], rng: random.Random, mode: str) -> bool:
    colors = list(SEED_COLORS)
    rng.shuffle(colors)
    a, b, noise_color = colors

    if mode == "score":
        if not _place_blob(grid, region, a, 2, occupied, rng, forbid_wall_adj=False):
            return False
        if not _place_blob(grid, region, b, 1, occupied, rng, require_wall_adj=True):
            return False
    else:
        if not _place_blob(grid, region, a, 1, occupied, rng, require_wall_adj=True):
            return False
        if not _place_blob(grid, region, b, 1, occupied, rng, require_wall_adj=True):
            return False

    if rng.random() < 0.8:
        if not _place_blob(grid, region, noise_color, 1, occupied, rng, forbid_wall_adj=True):
            return False
    return True


def _place_same_color_region(grid: np.ndarray, region: dict, occupied: set[tuple[int, int]], rng: random.Random) -> bool:
    color = rng.choice(SEED_COLORS)
    if not _place_blob(grid, region, color, 2, occupied, rng):
        return False
    if not _place_blob(grid, region, color, 1, occupied, rng, require_wall_adj=True):
        return False
    other = rng.choice([c for c in SEED_COLORS if c != color])
    if rng.random() < 0.6:
        if not _place_blob(grid, region, other, 1, occupied, rng, forbid_wall_adj=True):
            return False
    return True


def _place_simple_region(grid: np.ndarray, region: dict, occupied: set[tuple[int, int]], rng: random.Random) -> bool:
    color = rng.choice(SEED_COLORS)
    if rng.random() < 0.5:
        return _place_blob(grid, region, color, 2, occupied, rng)
    return _place_blob(grid, region, color, 1, occupied, rng, require_wall_adj=True)


def _place_empty_region_noise(grid: np.ndarray, region: dict, occupied: set[tuple[int, int]], rng: random.Random) -> bool:
    noise_count = 1 if rng.random() < 0.5 else 2
    noise_colors = list(SEED_COLORS)
    rng.shuffle(noise_colors)
    for i in range(noise_count):
        color = noise_colors[i % len(noise_colors)]
        if not _place_blob(grid, region, color, 1, occupied, rng, forbid_wall_adj=True):
            return False
    return True


def _independent_semantic_check(grid: np.ndarray) -> bool:
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2 or max(g.shape) > 30:
        return False
    if not set(np.unique(g)).issubset({GREY, WALL, *SEED_COLORS}):
        return False
    details = _solve_details(g)
    cleaned = details["cleaned"]
    out = details["output"]
    regions = details["regions"]

    removed = int(np.sum((g != cleaned) & (g != WALL)))
    if removed < 1:
        return False
    if len(regions) < 4:
        return False
    if not any(int(region["seed_count"]) == 0 for region in regions):
        return False
    if not any(len(region["seed_colors"]) >= 2 for region in regions):
        return False
    if not any(len(region["seed_colors"]) == 1 and int(region["seed_count"]) >= 2 for region in regions):
        return False
    if np.array_equal(g, out):
        return False
    if not np.any(g == WALL):
        return False
    return True


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    base = _build_wall_layout(rng)
    if base is None:
        return None

    regions = sorted(_region_infos(base)[1], key=lambda info: (-int(info["area"]), int(info["id"])))
    if len(regions) < 4:
        return None

    grid = base.copy()
    occupied_by_region = {int(region["id"]): set() for region in regions}

    conflict_region = regions[0]
    same_color_region = regions[1]
    empty_region = regions[-1]
    remaining = [region for region in regions if region["id"] not in {conflict_region["id"], same_color_region["id"], empty_region["id"]}]
    if not remaining:
        return None
    simple_region = remaining[0]
    extra_region = remaining[1] if len(remaining) > 1 else None

    main_conflict_mode = _conflict_mode_for_call(rng)
    if not _place_conflict_region(grid, conflict_region, occupied_by_region[conflict_region["id"]], rng, main_conflict_mode):
        return None
    if not _place_same_color_region(grid, same_color_region, occupied_by_region[same_color_region["id"]], rng):
        return None
    if not _place_empty_region_noise(grid, empty_region, occupied_by_region[empty_region["id"]], rng):
        return None
    if not _place_simple_region(grid, simple_region, occupied_by_region[simple_region["id"]], rng):
        return None

    if extra_region is not None:
        use_secondary_conflict = rng.random() < 0.6
        if use_secondary_conflict:
            other_mode = "tie" if main_conflict_mode == "score" else "score"
            if not _place_conflict_region(grid, extra_region, occupied_by_region[extra_region["id"]], rng, other_mode):
                return None
        else:
            if not _place_simple_region(grid, extra_region, occupied_by_region[extra_region["id"]], rng):
                return None

    transformed = _transform_grid(grid, rng.randrange(8))
    if not _independent_semantic_check(transformed):
        return None
    return transformed


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRY_LIMIT):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 4")
