from __future__ import annotations

from collections import deque
import random

import numpy as np


BLACK = 0
GREY = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
DIRS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
GEN_ATTEMPTS = 500


def _components4(mask: np.ndarray) -> list[np.ndarray]:
    height, width = map(int, mask.shape)
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(height):
        for c in range(width):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            queue = deque([(r, c)])
            coords: list[tuple[int, int]] = []
            while queue:
                cr, cc = queue.popleft()
                coords.append((int(cr), int(cc)))
                for dr, dc in DIRS4:
                    nr = int(cr + dr)
                    nc = int(cc + dc)
                    if 0 <= nr < height and 0 <= nc < width and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            comps.append(np.asarray(coords, dtype=int))
    return comps


def _ring_cells(fence: dict[str, int]) -> list[tuple[int, int]]:
    r0 = int(fence["r0"])
    c0 = int(fence["c0"])
    r1 = int(fence["r1"])
    c1 = int(fence["c1"])
    cells: list[tuple[int, int]] = []
    for c in range(c0, c1 + 1):
        cells.append((r0, c))
        if r1 != r0:
            cells.append((r1, c))
    for r in range(r0 + 1, r1):
        cells.append((r, c0))
        if c1 != c0:
            cells.append((r, c1))
    return cells


def _interior_mask(shape: tuple[int, int], fences: list[dict[str, int]]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    for fence in fences:
        mask[
            int(fence["r0"]) + 1 : int(fence["r1"]),
            int(fence["c0"]) + 1 : int(fence["c1"]),
        ] = True
    return mask


def _extract_fences(grid: np.ndarray) -> list[dict[str, int]]:
    fences: list[dict[str, int]] = []
    for coords in _components4(grid == GREY):
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        r0, c0, r1, c1 = map(int, (r0, c0, r1, c1))
        height = int(r1 - r0 + 1)
        width = int(c1 - c0 + 1)
        assert height >= 3 and width >= 3, "fence must enclose a nonempty interior"
        assert int(len(coords)) == 2 * (height + width) - 4, "grey component must be a 1px ring"
        assert np.all(grid[r0, c0 : c1 + 1] == GREY)
        assert np.all(grid[r1, c0 : c1 + 1] == GREY)
        assert np.all(grid[r0 : r1 + 1, c0] == GREY)
        assert np.all(grid[r0 : r1 + 1, c1] == GREY)
        assert not np.any(grid[r0 + 1 : r1, c0 + 1 : c1] == GREY), "ring interior must not contain grey"

        inner = grid[r0 + 1 : r1, c0 + 1 : c1]
        seeds = inner[(inner != BLACK) & (inner != GREY)]
        assert seeds.size >= 1, "each fence must contain at least one seed"
        colors = np.unique(seeds)
        assert colors.size == 1, "all seeds inside a fence must share one color"
        fences.append(
            {
                "r0": r0,
                "c0": c0,
                "r1": r1,
                "c1": c1,
                "seed_color": int(colors[0]),
                "seed_count": int(seeds.size),
            }
        )
    assert fences, "expected at least one fence"
    return fences


def _scan_ray(grid: np.ndarray, start: tuple[int, int], step: tuple[int, int]) -> tuple[list[int], list[tuple[int, int]]]:
    height, width = map(int, grid.shape)
    r, c = map(int, start)
    dr, dc = map(int, step)
    values: list[int] = []
    cells: list[tuple[int, int]] = []
    while 0 <= r < height and 0 <= c < width:
        value = int(grid[r, c])
        if value in (BLACK, GREY):
            break
        values.append(value)
        cells.append((r, c))
        r += dr
        c += dc
    return values, cells


def _find_priority_strip(grid: np.ndarray, seed_colors: set[int]) -> tuple[list[int], np.ndarray]:
    height, width = map(int, grid.shape)
    candidates: list[tuple[list[int], list[tuple[int, int]]]] = []
    specs = [
        ((0, 0), (0, 1)),
        ((0, 0), (1, 0)),
        ((0, width - 1), (0, -1)),
        ((0, width - 1), (1, 0)),
        ((height - 1, 0), (0, 1)),
        ((height - 1, 0), (-1, 0)),
        ((height - 1, width - 1), (0, -1)),
        ((height - 1, width - 1), (-1, 0)),
    ]
    for start, step in specs:
        values, cells = _scan_ray(grid, start, step)
        if not values:
            continue
        if len(values) != len(seed_colors):
            continue
        if len(set(values)) != len(values):
            continue
        if set(values) != seed_colors:
            continue
        candidates.append((values, cells))
    assert candidates, "failed to detect the corner priority strip"
    values, cells = max(candidates, key=lambda item: len(item[1]))
    mask = np.zeros_like(grid, dtype=bool)
    for r, c in cells:
        mask[int(r), int(c)] = True
    return list(map(int, values)), mask


def _growth_mask(
    shape: tuple[int, int],
    fence: dict[str, int],
    blocked: np.ndarray,
) -> np.ndarray:
    height, width = map(int, shape)
    limit = int(fence["seed_count"])
    dist = np.full(shape, -1, dtype=int)
    queue: deque[tuple[int, int]] = deque()
    for r, c in _ring_cells(fence):
        dist[int(r), int(c)] = 0
        queue.append((int(r), int(c)))

    while queue:
        r, c = queue.popleft()
        if dist[r, c] >= limit:
            continue
        for dr, dc in DIRS4:
            nr = int(r + dr)
            nc = int(c + dc)
            if 0 <= nr < height and 0 <= nc < width and dist[nr, nc] < 0 and not blocked[nr, nc]:
                dist[nr, nc] = int(dist[r, c] + 1)
                queue.append((nr, nc))
    return dist >= 0


def _apply_growth(
    grid: np.ndarray,
    fences: list[dict[str, int]],
    priority_order: list[int],
    strip_mask: np.ndarray,
) -> tuple[np.ndarray, list[tuple[int, np.ndarray]]]:
    out = np.array(grid, dtype=int, copy=True)
    ranks = {int(color): idx for idx, color in enumerate(priority_order)}
    blocked = _interior_mask(grid.shape, fences) | strip_mask
    best_rank = np.full(grid.shape, len(priority_order) + 1, dtype=int)
    claims: list[tuple[int, np.ndarray]] = []

    for fence in fences:
        color = int(fence["seed_color"])
        mask = _growth_mask(grid.shape, fence, blocked)
        claims.append((color, mask))
        rank = int(ranks[color])
        paint = mask & (rank <= best_rank)
        best_rank[paint] = rank
        out[paint] = color

    out[strip_mask] = grid[strip_mask]
    return out, claims


def _overlap_count(claims: list[tuple[int, np.ndarray]]) -> int:
    if len(claims) < 2:
        return 0
    merged: dict[int, np.ndarray] = {}
    for color, mask in claims:
        if color not in merged:
            merged[color] = mask.copy()
        else:
            merged[color] |= mask
    total = np.zeros_like(next(iter(merged.values())), dtype=int)
    for mask in merged.values():
        total += mask.astype(int)
    return int(np.count_nonzero(total >= 2))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each grey fence and grow it outward by the number of same-color seeds it encloses.

    Growth only happens outside every fenced interior. When different colors claim the same
    exterior cell, the corner priority strip decides which color remains.
    """
    g = np.asarray(grid, dtype=int)
    fences = _extract_fences(g)
    seed_colors = {int(fence["seed_color"]) for fence in fences}
    priority_order, strip_mask = _find_priority_strip(g, seed_colors)
    out, _claims = _apply_growth(g, fences, priority_order, strip_mask)
    return out


def _priority_cells(height: int, width: int, corner: str, orientation: str, length: int) -> list[tuple[int, int]]:
    if corner == "tl":
        start = (0, 0)
        step = (0, 1) if orientation == "h" else (1, 0)
    elif corner == "tr":
        start = (0, width - 1)
        step = (0, -1) if orientation == "h" else (1, 0)
    elif corner == "bl":
        start = (height - 1, 0)
        step = (0, 1) if orientation == "h" else (-1, 0)
    else:
        start = (height - 1, width - 1)
        step = (0, -1) if orientation == "h" else (-1, 0)
    return [
        (int(start[0] + i * step[0]), int(start[1] + i * step[1]))
        for i in range(length)
    ]


def _sample_fence_spec(overlap_mode: bool) -> dict[str, int]:
    for _ in range(50):
        seed_count = random.randint(2, 4) if overlap_mode else random.randint(1, 4)
        inner_h = random.randint(2, 6)
        inner_w = random.randint(2, 6)
        if inner_h * inner_w < seed_count + 1:
            continue
        return {
            "inner_h": inner_h,
            "inner_w": inner_w,
            "height": inner_h + 2,
            "width": inner_w + 2,
            "seed_count": seed_count,
        }
    raise RuntimeError("failed to sample a fence spec")


def _place_fences(
    height: int,
    width: int,
    strip_mask: np.ndarray,
    fence_specs: list[dict[str, int]],
) -> list[dict[str, int]] | None:
    used = np.array(strip_mask, dtype=bool, copy=True)
    placed: list[dict[str, int]] = []
    for spec in fence_specs:
        fh = int(spec["height"])
        fw = int(spec["width"])
        row_choices = list(range(2, height - fh - 1))
        col_choices = list(range(2, width - fw - 1))
        if not row_choices or not col_choices:
            return None
        random.shuffle(row_choices)
        random.shuffle(col_choices)
        found = None
        for r0 in row_choices:
            for c0 in col_choices:
                r1 = int(r0 + fh - 1)
                c1 = int(c0 + fw - 1)
                rr0 = max(0, r0 - 1)
                rr1 = min(height, r1 + 2)
                cc0 = max(0, c0 - 1)
                cc1 = min(width, c1 + 2)
                if used[rr0:rr1, cc0:cc1].any():
                    continue
                found = (r0, c0, r1, c1)
                used[rr0:rr1, cc0:cc1] = True
                break
            if found is not None:
                break
        if found is None:
            return None
        r0, c0, r1, c1 = map(int, found)
        placed.append(
            {
                "r0": r0,
                "c0": c0,
                "r1": r1,
                "c1": c1,
                "seed_count": int(spec["seed_count"]),
            }
        )
    return placed


def _seed_cells(fence: dict[str, int]) -> list[tuple[int, int]]:
    cells = [
        (r, c)
        for r in range(int(fence["r0"]) + 1, int(fence["r1"]))
        for c in range(int(fence["c0"]) + 1, int(fence["c1"]))
    ]
    return cells


def _build_input(
    height: int,
    width: int,
    priority_order: list[int],
    strip_cells: list[tuple[int, int]],
    fences: list[dict[str, int]],
) -> np.ndarray:
    grid = np.full((height, width), BLACK, dtype=int)
    for color, (r, c) in zip(priority_order, strip_cells):
        grid[int(r), int(c)] = int(color)

    for fence in fences:
        for r, c in _ring_cells(fence):
            grid[int(r), int(c)] = GREY
        seed_positions = random.sample(_seed_cells(fence), int(fence["seed_count"]))
        for r, c in seed_positions:
            grid[int(r), int(c)] = int(fence["seed_color"])
    return grid


def _attempt_generate() -> np.ndarray | None:
    overlap_mode = random.choice([False, True])
    if overlap_mode:
        height = random.randint(16, 22)
        width = random.randint(16, 22)
        fence_count = random.randint(2, 4)
        distinct_colors = random.randint(2, min(4, fence_count))
    else:
        height = random.randint(18, 28)
        width = random.randint(18, 28)
        fence_count = random.randint(1, 4)
        distinct_colors = 1 if fence_count == 1 else random.randint(1, min(4, fence_count))

    priority_order = random.sample(list(PALETTE), distinct_colors)
    corner = random.choice(["tl", "tr", "bl", "br"])
    orientation = random.choice(["h", "v"])
    strip_cells = _priority_cells(height, width, corner, orientation, distinct_colors)
    strip_mask = np.zeros((height, width), dtype=bool)
    for r, c in strip_cells:
        strip_mask[int(r), int(c)] = True

    fence_specs = [_sample_fence_spec(overlap_mode) for _ in range(fence_count)]
    placed = _place_fences(height, width, strip_mask, fence_specs)
    if placed is None:
        return None

    color_assignments = priority_order[:]
    while len(color_assignments) < fence_count:
        color_assignments.append(random.choice(priority_order))
    random.shuffle(color_assignments)
    for fence, color in zip(placed, color_assignments):
        fence["seed_color"] = int(color)

    grid = _build_input(height, width, priority_order, strip_cells, placed)
    expected, claims = _apply_growth(grid, placed, priority_order, strip_mask)
    overlap_cells = _overlap_count(claims)
    if overlap_mode and overlap_cells == 0:
        return None
    if not overlap_mode and overlap_cells != 0:
        return None

    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input failed after {GEN_ATTEMPTS} attempts")
