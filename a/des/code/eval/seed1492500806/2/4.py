import random
from collections import Counter, deque

import numpy as np


BG_COLOR = 0
WALL_COLOR = 5
DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
DIR8 = tuple(
    (dr, dc)
    for dr in (-1, 0, 1)
    for dc in (-1, 0, 1)
    if not (dr == 0 and dc == 0)
)

# Generator configuration.
GEN_RETRIES = 400
GRID_H_RANGE = (18, 24)
GRID_W_RANGE = (18, 24)
SHAPE_H_RANGE = (3, 6)
SHAPE_W_RANGE = (3, 7)
MIN_REGION_AREA = 6
MAX_PLACEMENT_TRIES = 100
COLOR_POOL = (1, 2, 3, 4, 6, 7, 8, 9)


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            pts: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                pts.append((int(rr), int(cc)))
                for dr, dc in DIR4:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        q.append((int(nr), int(nc)))
            comps.append(pts)
    return comps


def _trim_mask(mask: np.ndarray) -> np.ndarray:
    rows, cols = np.where(mask)
    return mask[rows.min() : rows.max() + 1, cols.min() : cols.max() + 1]


def _is_connected(mask: np.ndarray) -> bool:
    return len(_components(mask)) == 1


def _detect_bg(grid: np.ndarray) -> int:
    if bool(np.any(grid == BG_COLOR)):
        return BG_COLOR
    counts = Counter(map(int, grid.ravel()))
    return int(counts.most_common(1)[0][0])


def _detect_wall(grid: np.ndarray, bg: int) -> int:
    colors = [int(c) for c in np.unique(grid) if int(c) != int(bg)]
    assert colors, "expected a wall color"
    if WALL_COLOR in colors:
        return WALL_COLOR
    counts = Counter(map(int, grid[grid != bg].ravel()))
    return max(colors, key=lambda color: (int(counts[color]), int(color)))


def _closed_regions(grid: np.ndarray, bg: int, wall: int) -> list[list[tuple[int, int]]]:
    mask = grid != wall
    h, w = grid.shape
    regions: list[list[tuple[int, int]]] = []
    for comp in _components(mask):
        if any(r == 0 or r == h - 1 or c == 0 or c == w - 1 for r, c in comp):
            continue
        regions.append(comp)
    return regions


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each enclosed region according to the majority vote color inside it.

    Grey pixels act as walls. For every non-wall region that does not touch the
    grid edge, count the non-black pixels inside that region. If one color has
    a strict majority, fill the region's black cells with that color and keep
    the vote pixels. If several colors tie for the maximum count, keep the
    black cells unchanged and erase the tied top-vote pixels to black.
    """
    g = np.asarray(grid, dtype=int)
    bg = _detect_bg(g)
    wall = _detect_wall(g, bg)

    out = g.copy()
    for region in _closed_regions(g, bg, wall):
        votes = Counter(
            int(g[r, c])
            for r, c in region
            if int(g[r, c]) not in (int(bg), int(wall))
        )
        if not votes:
            continue

        top = max(votes.values())
        winners = [int(color) for color, count in votes.items() if int(count) == int(top)]
        if len(winners) == 1:
            winner = int(winners[0])
            for r, c in region:
                if int(out[r, c]) == int(bg):
                    out[r, c] = winner
        else:
            tied = set(winners)
            for r, c in region:
                if int(out[r, c]) in tied:
                    out[r, c] = bg

    return out


def _make_shape(rng: random.Random) -> np.ndarray:
    for _ in range(80):
        h = rng.randint(*SHAPE_H_RANGE)
        w = rng.randint(*SHAPE_W_RANGE)
        mask = np.ones((h, w), dtype=bool)
        notch_count = rng.randint(0, 3)

        for _ in range(notch_count):
            prev = mask.copy()
            side = rng.choice(("top", "bottom", "left", "right"))
            if side in ("top", "bottom"):
                max_depth = min(2, mask.shape[0] - 2)
                max_width = mask.shape[1] - 1
                if max_depth < 1 or max_width < 1:
                    continue
                depth = rng.randint(1, max_depth)
                width = rng.randint(1, max_width)
                start = rng.randint(0, mask.shape[1] - width)
                rows = slice(0, depth) if side == "top" else slice(mask.shape[0] - depth, mask.shape[0])
                mask[rows, start : start + width] = False
            else:
                max_depth = min(2, mask.shape[1] - 2)
                max_height = mask.shape[0] - 1
                if max_depth < 1 or max_height < 1:
                    continue
                depth = rng.randint(1, max_depth)
                height = rng.randint(1, max_height)
                start = rng.randint(0, mask.shape[0] - height)
                cols = slice(0, depth) if side == "left" else slice(mask.shape[1] - depth, mask.shape[1])
                mask[start : start + height, cols] = False

            if mask.sum() < MIN_REGION_AREA or not _is_connected(mask):
                mask = prev

        if mask.sum() < MIN_REGION_AREA:
            continue
        mask = _trim_mask(mask)
        if mask.shape[0] < 3 or mask.shape[1] < 3:
            continue
        if _is_connected(mask):
            return mask

    raise RuntimeError("failed to build a valid region shape")


def _build_patch(shape: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    h, w = shape.shape
    interior = np.zeros((h + 2, w + 2), dtype=bool)
    interior[1 : h + 1, 1 : w + 1] = shape

    boundary = np.zeros_like(interior)
    for r, c in np.argwhere(interior):
        rr = int(r)
        cc = int(c)
        for dr, dc in DIR8:
            nr = rr + dr
            nc = cc + dc
            if 0 <= nr < interior.shape[0] and 0 <= nc < interior.shape[1] and not bool(interior[nr, nc]):
                boundary[nr, nc] = True
    return interior, boundary


def _sample_majority_votes(
    rng: random.Random, capacity: int
) -> list[tuple[int, int]]:
    max_voters = min(6, capacity - 2)
    assert max_voters >= 1

    if capacity >= 9 and rng.random() < 0.45:
        color_count = 3
    elif capacity >= 7 and rng.random() < 0.7:
        color_count = 2
    else:
        color_count = 1

    colors = rng.sample(COLOR_POOL, color_count)
    if color_count == 1:
        return [(int(colors[0]), rng.randint(1, min(3, max_voters)))]

    majority = rng.randint(2, min(4, max_voters - (color_count - 1)))
    counts = [majority]
    remaining = max_voters - majority
    for idx in range(color_count - 1):
        reserve = color_count - 2 - idx
        upper = min(majority - 1, remaining - reserve)
        count = rng.randint(1, max(1, upper))
        counts.append(count)
        remaining -= count

    pairs = list(zip(colors, counts))
    rng.shuffle(pairs)
    while True:
        tally = Counter(count for _, count in pairs)
        if tally[max(counts)] == 1:
            return [(int(color), int(count)) for color, count in pairs]
        rng.shuffle(pairs)


def _sample_tie_votes(
    rng: random.Random, capacity: int
) -> list[tuple[int, int]]:
    colors = rng.sample(COLOR_POOL, 2)
    tie_count = rng.randint(1, min(3, (capacity - 1) // 2))
    return [(int(colors[0]), int(tie_count)), (int(colors[1]), int(tie_count))]


def _sample_votes(
    rng: random.Random, capacity: int, mode: str
) -> list[tuple[int, int]]:
    if mode == "tie":
        return _sample_tie_votes(rng, capacity)
    return _sample_majority_votes(rng, capacity)


def _place_specs(
    rng: random.Random, height: int, width: int, specs: list[dict]
) -> tuple[np.ndarray, list[dict]] | None:
    grid = np.full((height, width), BG_COLOR, dtype=int)
    reserved = np.zeros((height, width), dtype=bool)
    placed: list[dict] = []

    for spec in sorted(specs, key=lambda item: item["patch_area"], reverse=True):
        interior = spec["interior"]
        boundary = spec["boundary"]
        patch_h, patch_w = interior.shape
        if height - patch_h - 1 < 1 or width - patch_w - 1 < 1:
            return None

        top_left: tuple[int, int] | None = None
        for _ in range(MAX_PLACEMENT_TRIES):
            top = rng.randint(1, height - patch_h - 1)
            left = rng.randint(1, width - patch_w - 1)
            r0 = max(0, top - 1)
            c0 = max(0, left - 1)
            r1 = min(height, top + patch_h + 1)
            c1 = min(width, left + patch_w + 1)
            if bool(reserved[r0:r1, c0:c1].any()):
                continue
            top_left = (top, left)
            reserved[r0:r1, c0:c1] = True
            break

        if top_left is None:
            return None

        top, left = top_left
        for r, c in np.argwhere(boundary):
            grid[top + int(r), left + int(c)] = WALL_COLOR

        interior_cells = [
            (top + int(r), left + int(c))
            for r, c in np.argwhere(interior)
        ]
        placed.append(
            {
                "mode": spec["mode"],
                "votes": spec["votes"],
                "cells": interior_cells,
            }
        )

    return grid, placed


def _make_specs(rng: random.Random, region_count: int) -> list[dict]:
    tie_count = max(1, region_count // 2)
    majority_count = region_count - tie_count
    modes = ["tie"] * tie_count + ["majority"] * majority_count
    rng.shuffle(modes)

    specs: list[dict] = []
    for mode in modes:
        shape = _make_shape(rng)
        interior, boundary = _build_patch(shape)
        votes = _sample_votes(rng, int(shape.sum()), mode)
        specs.append(
            {
                "mode": mode,
                "interior": interior,
                "boundary": boundary,
                "votes": votes,
                "patch_area": int(interior.size),
            }
        )
    return specs


def _paint_votes(
    rng: random.Random, grid: np.ndarray, placed_regions: list[dict]
) -> None:
    for region in placed_regions:
        cells = list(region["cells"])
        vote_plan = list(region["votes"])
        vote_total = sum(count for _, count in vote_plan)
        chosen = rng.sample(cells, vote_total)
        offset = 0
        for color, count in vote_plan:
            for r, c in chosen[offset : offset + count]:
                grid[r, c] = int(color)
            offset += count


def _validate_grid(grid: np.ndarray, expected_regions: int) -> bool:
    bg = _detect_bg(grid)
    wall = _detect_wall(grid, bg)
    if bg != BG_COLOR or wall != WALL_COLOR:
        return False

    wall_count = int(np.count_nonzero(grid == wall))
    other_counts = Counter(int(v) for v in grid.ravel() if int(v) not in (bg, wall))
    if other_counts and wall_count <= max(other_counts.values()):
        return False

    regions = _closed_regions(grid, bg, wall)
    if len(regions) != expected_regions:
        return False

    saw_majority = False
    saw_tie = False
    for region in regions:
        votes = Counter(
            int(grid[r, c])
            for r, c in region
            if int(grid[r, c]) not in (bg, wall)
        )
        if not votes:
            return False
        top = max(votes.values())
        winner_count = sum(1 for count in votes.values() if int(count) == int(top))
        if winner_count == 1:
            saw_majority = True
            if len(region) - sum(votes.values()) < 2:
                return False
        else:
            saw_tie = True

    if not (saw_majority and saw_tie):
        return False

    out = main(grid)
    if int(np.count_nonzero(out != grid)) < expected_regions:
        return False
    return True


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    height = rng.randint(*GRID_H_RANGE)
    width = rng.randint(*GRID_W_RANGE)
    max_regions = 5 if min(height, width) >= 21 else 4
    region_count = rng.randint(3, max_regions)

    specs = _make_specs(rng, region_count)
    placed = _place_specs(rng, height, width, specs)
    if placed is None:
        return None

    grid, regions = placed
    _paint_votes(rng, grid, regions)

    if not _validate_grid(grid, region_count):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: exhausted retries")
