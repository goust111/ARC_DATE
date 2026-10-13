from __future__ import annotations

from collections import deque
import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
SOURCE_COLORS = (BLUE, RED, GREEN, YELLOW)
GEN_ATTEMPTS = 800


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _side_step(col: int, width: int) -> int:
    return -1 if col <= (width - 1 - col) else 1


def _trace_rays(grid: np.ndarray, *, return_info: bool = False):
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    h, w = out.shape

    sources = [(c, int(out[0, c])) for c in range(w) if int(out[0, c]) in SOURCE_COLORS]
    sources.sort()

    infos: list[dict] = []
    for c0, color in sources:
        r = 0
        c = c0
        path = [(0, c0)]
        deflections = 0
        terminated = False
        while r < h - 1:
            if int(out[r + 1, c]) == BLACK:
                r += 1
                out[r, c] = color
                path.append((r, c))
                continue

            dc = _side_step(c, w)
            nc = c + dc
            if 0 <= nc < w and int(out[r, nc]) == BLACK:
                c = nc
                out[r, c] = color
                path.append((r, c))
                deflections += 1
            else:
                terminated = True
                break

        infos.append(
            {
                "color": color,
                "source_col": c0,
                "path": path,
                "deflections": deflections,
                "terminated": terminated,
            }
        )

    if return_info:
        return out, infos
    return out


def _ray_mask(grid: np.ndarray) -> np.ndarray:
    return (grid != BLACK) & (grid != GREY)


def _nearest_ray_left(ray_grid: np.ndarray, r: int, c: int):
    for cc in range(c - 1, -1, -1):
        color = int(ray_grid[r, cc])
        if color != BLACK and color != GREY:
            return cc, color
    return None


def _nearest_ray_right(ray_grid: np.ndarray, r: int, c: int):
    w = ray_grid.shape[1]
    for cc in range(c + 1, w):
        color = int(ray_grid[r, cc])
        if color != BLACK and color != GREY:
            return cc, color
    return None


def _region_owner(comp: np.ndarray, ray_grid: np.ndarray) -> int:
    left_col = int(comp[:, 1].min())
    rows = sorted(int(r) for r in comp[comp[:, 1] == left_col][:, 0])
    owner_colors = set()
    for r in rows:
        hit = _nearest_ray_left(ray_grid, r, left_col)
        if hit is None:
            owner_colors.add(BLACK)
        else:
            owner_colors.add(int(hit[1]))

    if len(owner_colors) == 1:
        return owner_colors.pop()

    anchor_row = rows[0]
    hit = _nearest_ray_left(ray_grid, anchor_row, left_col)
    return BLACK if hit is None else int(hit[1])


def _analyze_regions(ray_grid: np.ndarray) -> list[dict]:
    regions = []
    for comp in _components(ray_grid == BLACK):
        regions.append({"coords": comp, "owner": _region_owner(comp, ray_grid)})
    return regions


def _fill_regions(ray_grid: np.ndarray) -> np.ndarray:
    out = ray_grid.copy()
    for region in _analyze_regions(ray_grid):
        color = int(region["owner"])
        if color == BLACK:
            continue
        coords = region["coords"]
        out[coords[:, 0], coords[:, 1]] = color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend top sources into deflecting rays and flood-fill each black compartment by its left owning ray.

    Input structure:
    - Black is the background and the top row contains 3 or 4 distinct colored source pixels.
    - Grey single-cell obstacles are scattered in the interior, while every other cell starts black.
    - Sources and later-drawn ray cells use the same visible colors, so the final ray network is built directly from the source colors.

    Rule:
    - Process the top-row sources from left to right. From each source, grow a downward ray through black cells.
    - Whenever the cell below the current ray head is blocked by a grey obstacle or an already drawn ray, move one cell sideways on the current row toward the nearer side border (ties go left); if that sideways cell is also blocked or out of bounds, terminate that ray.
    - After all rays are drawn, treat all non-black cells as barriers and split the remaining black cells into 4-connected regions.
    - Fill each region with the color of the nearest ray found by scanning left from the region's leftmost column; if no ray exists to its left, leave that region black.

    Key insight:
    - The fill color is owned by the region's left boundary ray, not by the spatially nearest ray to each pixel, and the compartments are created dynamically by the deflected ray network plus the obstacles.
    """
    g = np.asarray(grid, dtype=int)
    ray_grid = _trace_rays(g)
    return _fill_regions(ray_grid)


def _sample_source_columns(rng: random.Random, width: int, count: int) -> list[int] | None:
    cols = None
    for _ in range(80):
        centers = np.linspace(1, width - 2, count)
        picked = []
        for idx, center in enumerate(centers):
            if count == 3 and idx == 1 and width % 2 == 1 and rng.random() < 0.3:
                col = width // 2
            else:
                lo = max(1, int(round(center)) - 1)
                hi = min(width - 2, int(round(center)) + 1)
                col = rng.randint(lo, hi)
            picked.append(col)
        picked = sorted(picked)
        if min(b - a for a, b in zip(picked, picked[1:])) >= 3:
            cols = picked
            break
    return cols


def _max_deflections(col: int, width: int) -> int:
    cur = col
    steps = 0
    while steps < 2:
        nxt = cur + _side_step(cur, width)
        if not (0 <= nxt < width):
            break
        steps += 1
        cur = nxt
    return steps


def _plan_active_obstacles(
    rng: random.Random, height: int, width: int, source_cols: list[int]
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]] | None:
    # The prose requires real deflections, which need at least one blocker on the
    # ray's current column; passive extra obstacles are kept off source columns.
    n = len(source_cols)
    counts = [0] * n
    candidates = [i for i, c in enumerate(source_cols) if _max_deflections(c, width) > 0]
    rng.shuffle(candidates)
    for i in candidates:
        max_k = _max_deflections(source_cols[i], width)
        counts[i] = rng.choices([0, 1, 2], weights=[1, 4, 2])[0]
        counts[i] = min(counts[i], max_k)

    if sum(counts) < 2:
        boosted = [i for i in candidates if counts[i] < _max_deflections(source_cols[i], width)]
        rng.shuffle(boosted)
        for i in boosted:
            counts[i] += 1
            if sum(counts) >= 2:
                break
    if sum(counts) < 2:
        return None

    used_rows: set[int] = set()
    obstacles: list[tuple[int, int]] = []
    terminal_hints: list[tuple[int, int]] = []

    for idx, source_col in enumerate(source_cols):
        cur_col = source_col
        prev_row = 0
        rows: list[int] = []
        for _ in range(counts[idx]):
            choices = [
                row
                for row in range(prev_row + 2, height - 1)
                if row not in used_rows
            ]
            if not choices:
                return None
            row = rng.choice(choices)
            nxt = cur_col + _side_step(cur_col, width)
            if not (0 <= nxt < width):
                return None
            rows.append(row)
            used_rows.add(row)
            obstacles.append((row, cur_col))
            cur_col = nxt
            prev_row = row

        if cur_col in (0, width - 1) and prev_row + 2 < height and rng.random() < 0.35:
            choices = [row for row in range(prev_row + 2, height) if row not in used_rows]
            if choices:
                row = rng.choice(choices)
                used_rows.add(row)
                terminal_hints.append((row, cur_col))

    obstacles.extend(terminal_hints)
    return obstacles, terminal_hints


def _place_extra_obstacles(
    rng: random.Random,
    grid: np.ndarray,
    target_count: int,
    used_rows: set[int],
    protected: set[tuple[int, int]],
    source_cols: set[int],
) -> bool:
    h, w = grid.shape
    while int(np.count_nonzero(grid == GREY)) < target_count:
        cells = []
        for r in range(1, h):
            if r in used_rows:
                continue
            for c in range(w):
                if (r, c) in protected or c in source_cols:
                    continue
                if int(grid[r, c]) == BLACK:
                    cells.append((r, c))
        if not cells:
            return False
        r, c = rng.choice(cells)
        grid[r, c] = GREY
        used_rows.add(r)
    return True


def _has_counterexample(regions: list[dict], ray_grid: np.ndarray) -> bool:
    for region in regions:
        owner = int(region["owner"])
        if owner == BLACK:
            continue
        coords = region["coords"]
        for r, c in coords:
            left = _nearest_ray_left(ray_grid, int(r), int(c))
            right = _nearest_ray_right(ray_grid, int(r), int(c))
            if left is None or right is None:
                continue
            left_c, left_color = left
            right_c, right_color = right
            if owner == left_color and right_color != owner:
                if (int(c) - int(left_c)) > (int(right_c) - int(c)):
                    return True
    return False


def _validate_case(inp: np.ndarray) -> bool:
    h, w = inp.shape
    if inp.ndim != 2 or h > 30 or w > 30:
        return False

    source_cols = [c for c in range(w) if int(inp[0, c]) in SOURCE_COLORS]
    if len(source_cols) not in (3, 4):
        return False
    if any(b - a < 2 for a, b in zip(source_cols, source_cols[1:])):
        return False
    if len({int(inp[0, c]) for c in source_cols}) != len(source_cols):
        return False

    obstacle_coords = [tuple(map(int, rc)) for rc in np.argwhere(inp == GREY)]
    if not (5 <= len(obstacle_coords) <= 8):
        return False
    if len({r for r, _ in obstacle_coords}) != len(obstacle_coords):
        return False

    ray_grid, infos = _trace_rays(inp, return_info=True)
    if sum(info["deflections"] for info in infos) < 2:
        return False
    if not any(info["deflections"] == 0 for info in infos):
        return False
    if any(info["deflections"] > 2 for info in infos):
        return False

    ray_cells = set()
    for info in infos:
        for cell in info["path"]:
            if cell in ray_cells:
                return False
            ray_cells.add(cell)

    regions = _analyze_regions(ray_grid)
    owners = [int(region["owner"]) for region in regions]
    if not any(owner == BLACK for owner in owners):
        return False
    if len([owner for owner in owners if owner != BLACK]) < 2:
        return False

    for region in regions:
        coords = region["coords"]
        left_col = int(coords[:, 1].min())
        colors = set()
        for r in coords[coords[:, 1] == left_col][:, 0]:
            hit = _nearest_ray_left(ray_grid, int(r), left_col)
            colors.add(BLACK if hit is None else int(hit[1]))
        if len(colors) != 1:
            return False

    if not _has_counterexample(regions, ray_grid):
        return False

    out = main(inp)
    if np.array_equal(out, inp):
        return False
    if len(set(map(int, out.ravel()))) < 4:
        return False
    return True


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    height = rng.randint(13, 18)
    width = rng.randint(13, 18)
    source_count = rng.choice([3, 4])

    source_cols = _sample_source_columns(rng, width, source_count)
    if source_cols is None:
        return None

    colors = rng.sample(list(SOURCE_COLORS), source_count)
    grid = np.zeros((height, width), dtype=int)
    for c, color in zip(source_cols, colors):
        grid[0, c] = color

    planned = _plan_active_obstacles(rng, height, width, source_cols)
    if planned is None:
        return None
    active_obstacles, _terminal = planned

    used_rows: set[int] = set()
    for r, c in active_obstacles:
        if not (1 <= r < height) or int(grid[r, c]) != BLACK:
            return None
        grid[r, c] = GREY
        used_rows.add(r)

    ray_only = _trace_rays(grid)
    protected = set(tuple(map(int, rc)) for rc in np.argwhere(ray_only != BLACK))
    protected.update((0, c) for c in source_cols)

    target_obstacles = rng.randint(5, 8)
    if not _place_extra_obstacles(
        rng,
        grid,
        target_obstacles,
        used_rows,
        protected,
        set(source_cols),
    ):
        return None

    if _validate_case(grid):
        return grid
    return None


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid ray-deflection puzzle input")
