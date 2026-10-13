from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


BACKGROUND = 0
CONTAINER = 5
SEED_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
DIRS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))

GRID_MIN = 16
GRID_MAX = 28
CONTAINER_COUNT_MIN = 3
CONTAINER_COUNT_MAX = 5
CANDIDATE_LIMIT = 180
NOHIT_LIMIT = 120
EXTRA_NOHIT_MAX = 2
GENERATE_RETRIES = 500


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.flat)).most_common(1)[0][0])


def _components_of_color(grid: np.ndarray, color: int) -> list[np.ndarray]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != int(color):
                continue
            queue = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while queue:
                rr, cc = queue.popleft()
                coords.append((int(rr), int(cc)))
                for dr, dc in DIRS4:
                    nr = int(rr + dr)
                    nc = int(cc + dc)
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == int(color):
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            comps.append(np.asarray(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _is_solid_container(coords: np.ndarray) -> bool:
    if len(coords) != 9:
        return False
    r0, c0, r1, c1 = _bbox(coords)
    if r1 - r0 != 2 or c1 - c0 != 2:
        return False
    cells = {tuple(map(int, coord)) for coord in coords}
    return all((r, c) in cells for r in range(r0, r1 + 1) for c in range(c0, c1 + 1))


def _parse_scene(grid: np.ndarray) -> tuple[int, int, list[dict], list[dict]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = _get_bg(g)

    containers: list[dict] = []
    seeds: list[dict] = []
    container_color: int | None = None

    for color in map(int, np.unique(g)):
        if color == bg:
            continue
        comps = _components_of_color(g, color)
        for coords in comps:
            if _is_solid_container(coords):
                if container_color is None:
                    container_color = int(color)
                assert int(color) == int(container_color), "all containers must share one color"
                containers.append(
                    {
                        "color": int(color),
                        "coords": coords,
                        "cells": [tuple(map(int, cell)) for cell in coords],
                        "bbox": _bbox(coords),
                    }
                )
            else:
                assert len(coords) == 1, "all non-container foreground objects must be singleton seeds"
                seeds.append({"pos": tuple(map(int, coords[0])), "color": int(color)})

    assert containers, "expected at least one 3x3 container"
    assert container_color is not None
    assert len({seed["color"] for seed in seeds}) == len(seeds), "seed colors must be distinct"
    assert all(int(seed["color"]) != int(container_color) for seed in seeds), "seed color cannot equal container color"

    for seed in seeds:
        sr, sc = seed["pos"]
        for container in containers:
            r0, c0, r1, c1 = container["bbox"]
            near_row = r0 - 1 <= sr <= r1 + 1
            near_col = c0 - 1 <= sc <= c1 + 1
            assert not (near_row and near_col), "seeds must not touch containers, even diagonally"

    return bg, int(container_color), containers, seeds


def _build_container_index(shape: tuple[int, int], containers: list[dict]) -> np.ndarray:
    index = np.full(shape, -1, dtype=int)
    for idx, container in enumerate(containers):
        for r, c in container["cells"]:
            assert index[r, c] == -1
            index[r, c] = int(idx)
    return index


def _trace_seed(
    grid: np.ndarray,
    seed_pos: tuple[int, int],
    container_index: np.ndarray,
    bg: int,
) -> tuple[set[int], set[tuple[int, int]], set[tuple[int, int]]]:
    h, w = grid.shape
    sr, sc = seed_pos
    hits: set[int] = set()
    paint_cells: set[tuple[int, int]] = set()
    traversed: set[tuple[int, int]] = set()

    for dr, dc in DIRS4:
        r = int(sr + dr)
        c = int(sc + dc)
        while 0 <= r < h and 0 <= c < w:
            traversed.add((r, c))
            container_id = int(container_index[r, c])
            if container_id != -1:
                hits.add(container_id)
                break
            if int(grid[r, c]) == int(bg):
                paint_cells.add((r, c))
            r += dr
            c += dc

    return hits, paint_cells, traversed


def _scene_effects(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    bg, container_color, containers, seeds = _parse_scene(g)
    container_index = _build_container_index(g.shape, containers)

    paint_owner: dict[tuple[int, int], int] = {}
    hit_colors = [set() for _ in containers]
    traces: list[dict] = []

    for seed in seeds:
        hits, paint_cells, traversed = _trace_seed(g, seed["pos"], container_index, bg)
        for container_id in hits:
            hit_colors[container_id].add(int(seed["color"]))
        for cell in paint_cells:
            prev = paint_owner.get(cell)
            assert prev is None or prev == int(seed["color"]), "different-color rays cannot overlap"
            paint_owner[cell] = int(seed["color"])
        traces.append(
            {
                "pos": seed["pos"],
                "color": int(seed["color"]),
                "hits": frozenset(hits),
                "paint": frozenset(paint_cells),
                "path": frozenset(traversed),
            }
        )

    return {
        "grid": g,
        "bg": int(bg),
        "container_color": int(container_color),
        "containers": containers,
        "seeds": seeds,
        "container_index": container_index,
        "paint_owner": paint_owner,
        "hit_colors": hit_colors,
        "traces": traces,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend cross-shaped rays from colored seeds and recolor each hit container only when exactly one color reaches it.

    Input structure:
    - The background is a uniform empty color.
    - Several non-touching 3x3 solid containers share one color.
    - Additional foreground cells are singleton colored seeds placed away from the containers.

    Rule:
    - Each seed emits one-cell-wide rays upward, downward, leftward, and rightward.
    - A ray paints every background cell on its path until it reaches the grid edge or the first container cell in that direction.
    - Hitting a container records the seed color for that container, but rays never overwrite seeds or container cells.
    - After all rays are drawn, a container is recolored iff the set of colors that hit it has size exactly one; containers hit by zero or multiple colors keep their original color.

    Key insight:
    - The output depends on two separate effects: visible ray extension through empty space, and a container-level vote where only a unique hitting color is allowed to take over the whole 3x3 block.
    """
    effects = _scene_effects(np.asarray(grid, dtype=int))
    out = effects["grid"].copy()

    for (r, c), color in effects["paint_owner"].items():
        out[r, c] = int(color)

    for container, colors in zip(effects["containers"], effects["hit_colors"]):
        if len(colors) == 1:
            fill_color = int(next(iter(colors)))
            coords = container["coords"]
            out[coords[:, 0], coords[:, 1]] = fill_color

    return out


def _random_transform(grid: np.ndarray) -> np.ndarray:
    out = np.asarray(grid, dtype=int)
    if random.random() < 0.5:
        out = np.fliplr(out)
    turns = random.randrange(4)
    if turns:
        out = np.rot90(out, turns)
    return np.array(out, dtype=int, copy=True)


def _place_containers(height: int, width: int, count: int) -> list[dict] | None:
    assert height >= 9 and width >= 9
    candidates = [(r, c) for r in range(2, height - 4) for c in range(2, width - 4)]
    random.shuffle(candidates)

    forbidden = np.zeros((height, width), dtype=bool)
    containers: list[dict] = []

    for r0, c0 in candidates:
        if forbidden[r0 - 1 : r0 + 4, c0 - 1 : c0 + 4].any():
            continue
        coords = np.asarray([(r, c) for r in range(r0, r0 + 3) for c in range(c0, c0 + 3)], dtype=int)
        containers.append(
            {
                "bbox": (int(r0), int(c0), int(r0 + 2), int(c0 + 2)),
                "coords": coords,
                "cells": [tuple(map(int, cell)) for cell in coords],
            }
        )
        forbidden[r0 - 1 : r0 + 4, c0 - 1 : c0 + 4] = True
        if len(containers) == count:
            return containers
    return None


def _seed_forbidden_mask(shape: tuple[int, int], containers: list[dict]) -> np.ndarray:
    h, w = shape
    forbidden = np.zeros((h, w), dtype=bool)
    for container in containers:
        r0, c0, r1, c1 = container["bbox"]
        rr0 = max(0, r0 - 1)
        cc0 = max(0, c0 - 1)
        rr1 = min(h - 1, r1 + 1)
        cc1 = min(w - 1, c1 + 1)
        forbidden[rr0 : rr1 + 1, cc0 : cc1 + 1] = True
    return forbidden


def _enumerate_candidates(base_grid: np.ndarray, containers: list[dict]) -> list[dict]:
    container_index = _build_container_index(base_grid.shape, containers)
    forbidden = _seed_forbidden_mask(base_grid.shape, containers)
    candidates: list[dict] = []

    for r in range(base_grid.shape[0]):
        for c in range(base_grid.shape[1]):
            if forbidden[r, c] or int(base_grid[r, c]) != int(BACKGROUND):
                continue
            hits, paint_cells, traversed = _trace_seed(base_grid, (r, c), container_index, BACKGROUND)
            candidates.append(
                {
                "pos": (int(r), int(c)),
                "hits": frozenset(hits),
                "paint": frozenset(paint_cells),
                "path": frozenset(traversed),
                }
            )
    return candidates


def _compatible(a: dict, b: dict) -> bool:
    if a["pos"] == b["pos"]:
        return False
    ar, ac = a["pos"]
    br, bc = b["pos"]
    if max(abs(ar - br), abs(ac - bc)) <= 1:
        return False
    if a["paint"] & b["paint"]:
        return False
    if a["pos"] in b["path"] or b["pos"] in a["path"]:
        return False
    return True


def _select_seed_candidates(containers: list[dict], candidates: list[dict]) -> list[dict] | None:
    active = [cand for cand in candidates if cand["hits"]]
    nohit = [cand for cand in candidates if not cand["hits"]]
    random.shuffle(active)
    random.shuffle(nohit)

    active = active[:CANDIDATE_LIMIT]
    nohit = nohit[:NOHIT_LIMIT]
    selected: list[dict] | None = None

    for i, first in enumerate(active):
        for j in range(i + 1, len(active)):
            second = active[j]
            if not _compatible(first, second):
                continue
            hit_colors = [set() for _ in containers]
            for container_id in first["hits"]:
                hit_colors[container_id].add(1)
            for container_id in second["hits"]:
                hit_colors[container_id].add(2)
            counts = _status_counts_from_effects(hit_colors)
            if counts["single"] >= 1 and counts["multi"] >= 1 and counts["none"] >= 1:
                selected = [first, second]
                break
        if selected is not None:
            break

    if selected is None:
        return None

    extra_goal = random.randint(0, min(EXTRA_NOHIT_MAX, len(SEED_COLORS) - len(selected)))
    for cand in nohit:
        if extra_goal == 0:
            break
        if all(_compatible(cand, existing) for existing in selected):
            selected.append(cand)
            extra_goal -= 1

    return selected


def _status_counts_from_effects(hit_colors: list[set[int]]) -> Counter:
    labels = []
    for colors in hit_colors:
        if len(colors) == 0:
            labels.append("none")
        elif len(colors) == 1:
            labels.append("single")
        else:
            labels.append("multi")
    return Counter(labels)


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(GRID_MIN, GRID_MAX)
    width = random.randint(GRID_MIN, GRID_MAX)
    container_count = random.randint(CONTAINER_COUNT_MIN, CONTAINER_COUNT_MAX)
    containers = _place_containers(height, width, container_count)
    if containers is None:
        return None

    base_grid = np.full((height, width), BACKGROUND, dtype=int)
    for container in containers:
        coords = container["coords"]
        base_grid[coords[:, 0], coords[:, 1]] = CONTAINER

    candidates = _enumerate_candidates(base_grid, containers)
    if sum(bool(cand["hits"]) for cand in candidates) < 2:
        return None

    selected = _select_seed_candidates(containers, candidates)
    if selected is None:
        return None

    if len(selected) < 2 or len(selected) > len(SEED_COLORS):
        return None

    colors = random.sample(SEED_COLORS, len(selected))
    random.shuffle(selected)

    grid = base_grid.copy()
    for cand, color in zip(selected, colors):
        r, c = cand["pos"]
        grid[r, c] = int(color)

    try:
        effects = _scene_effects(grid)
    except AssertionError:
        return None

    counts = _status_counts_from_effects(effects["hit_colors"])
    if counts["single"] < 1 or counts["multi"] < 1 or counts["none"] < 1:
        return None

    transformed = _random_transform(grid)
    try:
        transformed_effects = _scene_effects(transformed)
    except AssertionError:
        return None

    transformed_counts = _status_counts_from_effects(transformed_effects["hit_colors"])
    if transformed_counts != counts:
        return None

    return transformed


def generate_input() -> np.ndarray:
    for _ in range(GENERATE_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for seed-container ray puzzle")
