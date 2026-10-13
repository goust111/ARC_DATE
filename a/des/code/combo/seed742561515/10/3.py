import random
from collections import deque

import numpy as np


BLACK = 0
RED = 2
GREEN = 3
ORANGE = 7
SEED_COLORS = (1, 4, 5, 6, 8, 9)

DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIRS8 = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
]
DIR_TO_MODE = {
    (-1, 0): "up",
    (0, 1): "right",
    (1, 0): "down",
    (0, -1): "left",
}

GRID_H_RANGE = (26, 34)
GRID_W_RANGE = (26, 34)
PAIR_COUNT_RANGE = (2, 4)
REGION_INNER_H_RANGE = (4, 7)
REGION_INNER_W_RANGE = (4, 7)
REGION_MIN_AREA = 8
REGION_MAX_CARVES = 3
SEED_SIZE_RANGE = (2, 3)
SEED_MIN_AREA = 3
SEED_MAX_COLORS = 3
BORDER_MARGIN = 1
PLACEMENT_PAD = 1
GEN_RETRIES = 400


def _neighbors(r: int, c: int, h: int, w: int, dirs) -> list[tuple[int, int]]:
    out = []
    for dr, dc in dirs:
        nr, nc = r + dr, c + dc
        if 0 <= nr < h and 0 <= nc < w:
            out.append((nr, nc))
    return out


def _components_mask(mask: np.ndarray, connectivity: int = 4) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    dirs = DIRS4 if connectivity == 4 else DIRS8
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            queue = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while queue:
                cr, cc = queue.popleft()
                comp.append((cr, cc))
                for nr, nc in _neighbors(cr, cc, h, w, dirs):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(cells: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _touches_border(cells: list[tuple[int, int]], h: int, w: int) -> bool:
    return any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in cells)


def _dilate8(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    out = mask.copy()
    for r, c in np.argwhere(mask):
        rr = int(r)
        cc = int(c)
        for nr, nc in _neighbors(rr, cc, h, w, DIRS8):
            out[nr, nc] = True
    return out


def _connected(mask: np.ndarray) -> bool:
    cells = int(mask.sum())
    return cells > 0 and len(_components_mask(mask, connectivity=4)) == 1


def _extract_regions(grid: np.ndarray) -> tuple[list[dict], set[tuple[int, int]]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    regions: list[dict] = []
    region_cells: set[tuple[int, int]] = set()
    for comp in _components_mask(g != GREEN, connectivity=4):
        if _touches_border(comp, h, w):
            continue
        vals = {int(g[r, c]) for r, c in comp}
        if not vals <= {BLACK, RED, ORANGE}:
            continue
        reds = [(r, c) for r, c in comp if int(g[r, c]) == RED]
        oranges = [(r, c) for r, c in comp if int(g[r, c]) == ORANGE]
        if len(reds) != 1 or len(oranges) != 1:
            continue
        rr, rc = reds[0]
        orr, orc = oranges[0]
        direction = (orr - rr, orc - rc)
        if direction not in DIR_TO_MODE:
            continue
        bbox = _bbox(comp)
        region = {
            "cells": comp,
            "cell_set": set(comp),
            "bbox": bbox,
            "red": (rr, rc),
            "orange": (orr, orc),
            "direction": direction,
        }
        regions.append(region)
        region_cells.update(comp)
    assert regions, "expected at least one enclosed region"
    return regions, region_cells


def _extract_seeds(grid: np.ndarray, region_cells: set[tuple[int, int]]) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seed_mask = (g != BLACK) & (g != GREEN) & (g != RED) & (g != ORANGE)
    seed_components = _components_mask(seed_mask, connectivity=4)
    external_reds = {
        (r, c)
        for r, c in np.argwhere(g == RED)
        if (int(r), int(c)) not in region_cells
    }
    seeds: list[dict] = []
    used_reds: set[tuple[int, int]] = set()
    for comp in seed_components:
        adj_reds: set[tuple[int, int]] = set()
        for r, c in comp:
            for nr, nc in _neighbors(r, c, h, w, DIRS4):
                if (nr, nc) in external_reds:
                    adj_reds.add((nr, nc))
        assert len(adj_reds) == 1, "every seed must touch exactly one external red marker"
        red = next(iter(adj_reds))
        used_reds.add(red)
        r0, c0, r1, c1 = _bbox(comp)
        tile = np.full((r1 - r0 + 1, c1 - c0 + 1), BLACK, dtype=int)
        for r, c in comp:
            tile[r - r0, c - c0] = int(g[r, c])
        colors = set(int(tile[r, c]) for r in range(tile.shape[0]) for c in range(tile.shape[1]) if int(tile[r, c]) != BLACK)
        assert len(colors) >= 2, "seed must be multi-color"
        seeds.append(
            {
                "cells": comp,
                "cell_set": set(comp),
                "red": red,
                "tile": tile,
            }
        )
    assert seeds, "expected at least one seed"
    assert used_reds == external_reds, "every external red must belong to a seed"
    return seeds


def _match_regions_to_seeds(regions: list[dict], seeds: list[dict]) -> list[int]:
    assert len(regions) == len(seeds), "region and seed counts must match"
    chosen: list[int] = []
    for region in regions:
        rr, rc = region["red"]
        scores = [
            min(abs(rr - sr), abs(rc - sc))
            for sr, sc in (seed["red"] for seed in seeds)
        ]
        best = min(scores)
        assert scores.count(best) == 1, "closest row-or-column match must be unique"
        chosen.append(scores.index(best))
    assert len(set(chosen)) == len(chosen), "each seed must match at most one region"
    return chosen


def _tile_color(tile: np.ndarray, region_bbox: tuple[int, int, int, int], direction: tuple[int, int], r: int, c: int) -> int:
    r0, c0, r1, c1 = region_bbox
    if direction == (-1, 0):
        tr = r - r0
        tc = c - c0
    elif direction == (0, 1):
        tr = r - r0
        tc = c1 - c
    elif direction == (1, 0):
        tr = r1 - r
        tc = c1 - c
    else:
        tr = r1 - r
        tc = c - c0
    return int(tile[tr % tile.shape[0], tc % tile.shape[1]])


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    regions, region_cells = _extract_regions(g)
    seeds = _extract_seeds(g, region_cells)
    matches = _match_regions_to_seeds(regions, seeds)

    out = g.copy()
    for seed in seeds:
        for r, c in seed["cells"]:
            out[r, c] = BLACK
        sr, sc = seed["red"]
        out[sr, sc] = BLACK

    for region, seed_idx in zip(regions, matches):
        tile = seeds[seed_idx]["tile"]
        for r, c in region["cells"]:
            if int(out[r, c]) != BLACK:
                continue
            color = _tile_color(tile, region["bbox"], region["direction"], r, c)
            if color != BLACK:
                out[r, c] = color

    return out


def _randint_range(rng: random.Random, bounds: tuple[int, int]) -> int:
    lo, hi = bounds
    return rng.randint(lo, hi)


def _paint_patch(grid: np.ndarray, patch: np.ndarray, top: int, left: int) -> None:
    for r in range(patch.shape[0]):
        for c in range(patch.shape[1]):
            color = int(patch[r, c])
            if color != BLACK:
                grid[top + r, left + c] = color


def _rect_is_free(blocked: np.ndarray, top: int, left: int, h: int, w: int, pad: int) -> bool:
    gh, gw = blocked.shape
    if top < BORDER_MARGIN or left < BORDER_MARGIN:
        return False
    if top + h > gh - BORDER_MARGIN or left + w > gw - BORDER_MARGIN:
        return False
    r0 = max(0, top - pad)
    c0 = max(0, left - pad)
    r1 = min(gh, top + h + pad)
    c1 = min(gw, left + w + pad)
    return not bool(blocked[r0:r1, c0:c1].any())


def _mark_rect(blocked: np.ndarray, top: int, left: int, h: int, w: int) -> None:
    blocked[top:top + h, left:left + w] = True


def _grow_connected_cells(h: int, w: int, target: int, rng: random.Random) -> set[tuple[int, int]]:
    start = (rng.randrange(h), rng.randrange(w))
    cells = {start}
    while len(cells) < target:
        frontier = set()
        for r, c in cells:
            for nr, nc in _neighbors(r, c, h, w, DIRS4):
                if (nr, nc) not in cells:
                    frontier.add((nr, nc))
        if not frontier:
            break
        cells.add(rng.choice(tuple(frontier)))
    return cells


def _make_seed_patch(rng: random.Random) -> dict | None:
    sh = _randint_range(rng, SEED_SIZE_RANGE)
    sw = _randint_range(rng, SEED_SIZE_RANGE)
    max_area = sh * sw
    min_area = max(SEED_MIN_AREA, (max_area + 1) // 2)
    target = rng.randint(min_area, max_area)
    cells = _grow_connected_cells(sh, sw, target, rng)
    if len(cells) < SEED_MIN_AREA:
        return None

    color_count = rng.randint(2, min(SEED_MAX_COLORS, len(SEED_COLORS)))
    palette = rng.sample(SEED_COLORS, color_count)
    color_map: dict[tuple[int, int], int] = {}
    ordered_cells = sorted(cells)
    color_map[ordered_cells[0]] = palette[0]
    color_map[ordered_cells[-1]] = palette[1]
    for cell in ordered_cells[1:-1]:
        color_map[cell] = rng.choice(palette)
    used = {color_map[cell] for cell in ordered_cells}
    if len(used) < 2:
        return None

    red_candidates: list[tuple[int, int]] = []
    for r, c in cells:
        for dr, dc in DIRS4:
            pos = (r + dr, c + dc)
            if pos in cells:
                continue
            adjacent = 0
            for nr, nc in _neighbors(pos[0], pos[1], sh, sw, DIRS4):
                if (nr, nc) in cells:
                    adjacent += 1
            if adjacent == 1:
                red_candidates.append(pos)
    if not red_candidates:
        return None
    red = rng.choice(red_candidates)

    all_cells = list(cells) + [red]
    min_r = min(r for r, _ in all_cells)
    min_c = min(c for _, c in all_cells)
    max_r = max(r for r, _ in all_cells)
    max_c = max(c for _, c in all_cells)
    patch = np.full((max_r - min_r + 1, max_c - min_c + 1), BLACK, dtype=int)
    for (r, c), color in color_map.items():
        patch[r - min_r, c - min_c] = color
    red_local = (red[0] - min_r, red[1] - min_c)
    patch[red_local] = RED
    return {
        "patch": patch,
        "red_local": red_local,
    }


def _carve_notch(mask: np.ndarray, side: str, rng: random.Random) -> np.ndarray:
    h, w = mask.shape
    trial = mask.copy()
    if side in ("top", "bottom"):
        depth = rng.randint(1, min(2, h - 2))
        width = rng.randint(1, max(1, w - 2))
        start = rng.randint(0, w - width)
        rows = slice(0, depth) if side == "top" else slice(h - depth, h)
        trial[rows, start:start + width] = False
    else:
        depth = rng.randint(1, min(2, w - 2))
        height = rng.randint(1, max(1, h - 2))
        start = rng.randint(0, h - height)
        cols = slice(0, depth) if side == "left" else slice(w - depth, w)
        trial[start:start + height, cols] = False
    return trial


def _make_region_patch(rng: random.Random) -> dict | None:
    inner_h = _randint_range(rng, REGION_INNER_H_RANGE)
    inner_w = _randint_range(rng, REGION_INNER_W_RANGE)
    mask = np.ones((inner_h, inner_w), dtype=bool)
    for _ in range(rng.randint(0, REGION_MAX_CARVES)):
        side = rng.choice(("top", "bottom", "left", "right"))
        trial = _carve_notch(mask, side, rng)
        if int(trial.sum()) < REGION_MIN_AREA:
            continue
        if not _connected(trial):
            continue
        mask = trial

    interior = np.zeros((inner_h + 2, inner_w + 2), dtype=bool)
    interior[1:1 + inner_h, 1:1 + inner_w] = mask
    boundary = _dilate8(interior) & ~interior
    if not boundary.any():
        return None

    direction = rng.choice(list(DIR_TO_MODE))
    dr, dc = direction
    candidates: list[tuple[int, int]] = []
    for r, c in np.argwhere(interior):
        rr = int(r)
        cc = int(c)
        nr, nc = rr + dr, cc + dc
        if 0 <= nr < interior.shape[0] and 0 <= nc < interior.shape[1] and interior[nr, nc]:
            candidates.append((rr, cc))
    if not candidates:
        return None
    red_local = rng.choice(candidates)
    orange_local = (red_local[0] + dr, red_local[1] + dc)

    patch = np.full(interior.shape, BLACK, dtype=int)
    patch[boundary] = GREEN
    patch[red_local] = RED
    patch[orange_local] = ORANGE
    return {
        "patch": patch,
        "red_local": red_local,
    }


def _place_regions(grid: np.ndarray, blocked: np.ndarray, rng: random.Random, pair_count: int) -> list[dict] | None:
    h, w = grid.shape
    used_rows: set[int] = set()
    used_cols: set[int] = set()
    placed: list[dict] = []
    for _ in range(pair_count):
        success = False
        for _ in range(120):
            region = _make_region_patch(rng)
            if region is None:
                continue
            patch = region["patch"]
            ph, pw = patch.shape
            max_top = h - BORDER_MARGIN - ph
            max_left = w - BORDER_MARGIN - pw
            if max_top < BORDER_MARGIN or max_left < BORDER_MARGIN:
                return None
            for _ in range(120):
                top = rng.randint(BORDER_MARGIN, max_top)
                left = rng.randint(BORDER_MARGIN, max_left)
                if not _rect_is_free(blocked, top, left, ph, pw, PLACEMENT_PAD):
                    continue
                red_r = top + region["red_local"][0]
                red_c = left + region["red_local"][1]
                if red_r in used_rows or red_c in used_cols:
                    continue
                _paint_patch(grid, patch, top, left)
                _mark_rect(blocked, top, left, ph, pw)
                used_rows.add(red_r)
                used_cols.add(red_c)
                placed.append({"red": (red_r, red_c)})
                success = True
                break
            if success:
                break
        if not success:
            return None
    return placed


def _place_one_seed(
    grid: np.ndarray,
    blocked: np.ndarray,
    rng: random.Random,
    region_red: tuple[int, int],
    other_region_reds: list[tuple[int, int]],
    mode: str,
) -> bool:
    h, w = grid.shape
    forbidden_rows = {r for r, _ in other_region_reds}
    forbidden_cols = {c for _, c in other_region_reds}
    for _ in range(160):
        seed = _make_seed_patch(rng)
        if seed is None:
            continue
        patch = seed["patch"]
        ph, pw = patch.shape
        rr, rc = seed["red_local"]

        if mode == "row":
            top = region_red[0] - rr
            if top < BORDER_MARGIN or top + ph > h - BORDER_MARGIN:
                continue
            left_candidates = list(range(BORDER_MARGIN, w - BORDER_MARGIN - pw + 1))
            rng.shuffle(left_candidates)
            for left in left_candidates:
                seed_red_c = left + rc
                if seed_red_c in forbidden_cols:
                    continue
                if not _rect_is_free(blocked, top, left, ph, pw, PLACEMENT_PAD):
                    continue
                _paint_patch(grid, patch, top, left)
                _mark_rect(blocked, top, left, ph, pw)
                return True
        else:
            left = region_red[1] - rc
            if left < BORDER_MARGIN or left + pw > w - BORDER_MARGIN:
                continue
            top_candidates = list(range(BORDER_MARGIN, h - BORDER_MARGIN - ph + 1))
            rng.shuffle(top_candidates)
            for top in top_candidates:
                seed_red_r = top + rr
                if seed_red_r in forbidden_rows:
                    continue
                if not _rect_is_free(blocked, top, left, ph, pw, PLACEMENT_PAD):
                    continue
                _paint_patch(grid, patch, top, left)
                _mark_rect(blocked, top, left, ph, pw)
                return True
    return False


def _validate_generated(grid: np.ndarray) -> bool:
    try:
        regions, region_cells = _extract_regions(grid)
        seeds = _extract_seeds(grid, region_cells)
        if len(regions) != len(seeds):
            return False
        matches = _match_regions_to_seeds(regions, seeds)
        out = main(grid)
    except Exception:
        return False

    if not np.any(out != grid):
        return False

    for region, seed_idx in zip(regions, matches):
        tile = seeds[seed_idx]["tile"]
        if tile.size == 0:
            return False
        painted = 0
        for r, c in region["cells"]:
            if int(grid[r, c]) == BLACK and int(out[r, c]) != BLACK:
                painted += 1
        if painted == 0:
            return False
    return True


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = _randint_range(rng, GRID_H_RANGE)
    w = _randint_range(rng, GRID_W_RANGE)
    grid = np.full((h, w), BLACK, dtype=int)
    blocked = np.zeros((h, w), dtype=bool)

    pair_count = rng.randint(PAIR_COUNT_RANGE[0], PAIR_COUNT_RANGE[1])
    regions = _place_regions(grid, blocked, rng, pair_count)
    if regions is None:
        return None

    modes = [rng.choice(("row", "col")) for _ in range(pair_count)]
    order = list(range(pair_count))
    rng.shuffle(order)
    red_positions = [region["red"] for region in regions]
    for idx in order:
        others = [red_positions[j] for j in range(pair_count) if j != idx]
        if not _place_one_seed(grid, blocked, rng, red_positions[idx], others, modes[idx]):
            return None

    if not _validate_generated(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
