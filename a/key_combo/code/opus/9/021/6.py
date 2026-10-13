from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import Color, find_components, get_bg, neighbors4


BG = Color.BLACK
BLUE = Color.BLUE
RED = Color.RED

GRID_MIN = 13
GRID_MAX = 16
OBJECT_COUNT_OPTIONS = (5, 7)
SIZE_CHOICES = tuple(range(2, 11))
HIT_COUNT_OPTIONS = (2, 3)
GEN_ATTEMPTS = 500
SHAPE_ATTEMPTS = 120
MIN_RAY_LENGTH = 6

DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIR8 = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0)]


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _neighbors8(r: int, c: int) -> list[tuple[int, int]]:
    return [(r + dr, c + dc) for dr, dc in DIR8]


def _parse_marker(grid: np.ndarray) -> tuple[tuple[int, int], tuple[int, int]]:
    red_coords = [tuple(map(int, rc)) for rc in np.argwhere(grid == RED)]
    assert len(red_coords) == 4, "expected exactly four red marker cells"
    red_set = set(red_coords)

    centers = []
    for r, c in red_coords:
        red_neighbors = [(nr, nc) for nr, nc in neighbors4(r, c) if (nr, nc) in red_set]
        if len(red_neighbors) == 3:
            centers.append((r, c, red_neighbors))
    assert len(centers) == 1, "expected a unique T-marker center"

    center_r, center_c, red_neighbors = centers[0]
    same_row = [p for p in red_neighbors if p[0] == center_r]
    same_col = [p for p in red_neighbors if p[1] == center_c]
    assert (len(same_row), len(same_col)) in {(2, 1), (1, 2)}, "marker must form a T"

    if len(same_row) == 2:
        bar_cells = same_row
    else:
        bar_cells = same_col
    control = next(p for p in red_neighbors if p not in bar_cells)
    step = (center_r - control[0], center_c - control[1])
    assert abs(step[0]) + abs(step[1]) == 1, "control must be 4-adjacent to center"
    return (center_r, center_c), step


def _ray_cells(shape: tuple[int, int], center: tuple[int, int], step: tuple[int, int]) -> list[tuple[int, int]]:
    cells = []
    r = center[0] + step[0]
    c = center[1] + step[1]
    while _in_bounds(shape, r, c):
        cells.append((r, c))
        r += step[0]
        c += step[1]
    return cells


def _blue_objects(grid: np.ndarray) -> list[dict]:
    bg = get_bg(grid)
    objs = find_components(
        grid,
        bg=bg,
        connectivity=4,
        monochromatic=True,
        colors=[BLUE],
    )
    objs.sort(key=lambda obj: (obj["bbox"][0], obj["bbox"][1]))
    return objs


def _analyze(grid: np.ndarray) -> dict:
    center, step = _parse_marker(grid)
    ray = set(_ray_cells(grid.shape, center, step))
    blue_objects = _blue_objects(grid)
    assert len(blue_objects) % 2 == 1, "blue object count must be odd"
    areas = [int(obj["area"]) for obj in blue_objects]
    assert len(set(areas)) == len(areas), "blue object areas must be unique"
    median = sorted(areas)[len(areas) // 2]

    hit = []
    unhit = []
    for obj in blue_objects:
        coords = {tuple(map(int, rc)) for rc in obj["coords"]}
        record = {
            "coords": coords,
            "area": int(obj["area"]),
            "hit": any(cell in ray for cell in coords),
        }
        if record["hit"]:
            hit.append(record)
        else:
            unhit.append(record)

    return {
        "center": center,
        "step": step,
        "ray": ray,
        "objects": blue_objects,
        "areas": areas,
        "median": median,
        "hit": hit,
        "unhit": unhit,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Erase only the ray-hit blue objects whose area falls below the global median blue-object area.

    Input structure:
    - A black background contains exactly one red T marker made of a 3-pixel bar plus one 4-adjacent control pixel.
    - The remaining foreground cells are blue 4-connected objects with pairwise distinct areas.
    - The red marker does not touch the blue objects.

    Rule:
    - Find the bar center and cast a straight ray from that center away from the control pixel.
    - Mark every blue 4-connected object that has at least one pixel on that ray as hit.
    - Compute the median area over all blue objects in the grid, including the unhit ones.
    - Erase every hit object whose area is strictly smaller than that global median; keep all other cells unchanged.

    Key insight:
    - Neither criterion alone is sufficient: an object disappears only if it is both ray-hit and smaller than the median area computed from the full blue-object population.
    """
    g = np.asarray(grid, dtype=int)
    info = _analyze(g)
    out = g.copy()
    for obj in info["hit"]:
        if obj["area"] < info["median"]:
            for r, c in obj["coords"]:
                out[r, c] = BG
    return out


def _marker_cells(center: tuple[int, int], axis: str, control: tuple[int, int]) -> set[tuple[int, int]]:
    r, c = center
    if axis == "horizontal":
        return {(r, c - 1), (r, c), (r, c + 1), control}
    return {(r - 1, c), (r, c), (r + 1, c), control}


def _marker_candidates(h: int, w: int) -> list[tuple[tuple[int, int], str, tuple[int, int], tuple[int, int]]]:
    candidates = []

    for r in range(1, h - 1):
        for c in range(1, w - 1):
            if 1 <= c - 1 and c + 1 < w - 1:
                for control in ((r - 1, c), (r + 1, c)):
                    if _in_bounds((h, w), *control):
                        step = (r - control[0], c - control[1])
                        if len(_ray_cells((h, w), (r, c), step)) >= MIN_RAY_LENGTH:
                            candidates.append(((r, c), "horizontal", control, step))
            if 1 <= r - 1 and r + 1 < h - 1:
                for control in ((r, c - 1), (r, c + 1)):
                    if _in_bounds((h, w), *control):
                        step = (r - control[0], c - control[1])
                        if len(_ray_cells((h, w), (r, c), step)) >= MIN_RAY_LENGTH:
                            candidates.append(((r, c), "vertical", control, step))

    return candidates


def _touches_occupied(cell: tuple[int, int], occupied: set[tuple[int, int]]) -> bool:
    r, c = cell
    if cell in occupied:
        return True
    return any((nr, nc) in occupied for nr, nc in _neighbors8(r, c))


def _grow_object(
    grid: np.ndarray,
    size: int,
    occupied: set[tuple[int, int]],
    ray: set[tuple[int, int]],
    *,
    seed: tuple[int, int] | None,
    must_hit: bool,
) -> set[tuple[int, int]] | None:
    h, w = grid.shape
    ray_limit = random.choice((1, 2)) if must_hit and size > 2 else 1

    for _ in range(SHAPE_ATTEMPTS):
        if seed is None:
            seed_choices = [
                (r, c)
                for r in range(h)
                for c in range(w)
                if grid[r, c] == BG and (r, c) not in ray and not _touches_occupied((r, c), occupied)
            ]
            if not seed_choices:
                return None
            start = random.choice(seed_choices)
        else:
            start = seed
            if grid[start] != BG or _touches_occupied(start, occupied):
                continue

        current = {start}
        ray_count = 1 if start in ray else 0

        while len(current) < size:
            frontier = set()
            for r, c in current:
                for nr, nc in neighbors4(r, c):
                    if _in_bounds((h, w), nr, nc):
                        frontier.add((nr, nc))

            options = []
            for cell in frontier:
                if cell in current:
                    continue
                if grid[cell] != BG:
                    continue
                if _touches_occupied(cell, occupied):
                    continue
                if not must_hit and cell in ray:
                    continue
                if must_hit and cell in ray and ray_count >= ray_limit:
                    continue
                options.append(cell)

            if not options:
                break

            off_ray = [cell for cell in options if cell not in ray]
            chosen = random.choice(off_ray if off_ray else options)
            current.add(chosen)
            if chosen in ray:
                ray_count += 1

        if len(current) == size and (not must_hit or any(cell in ray for cell in current)):
            return current
    return None


def _place_object(
    grid: np.ndarray,
    size: int,
    occupied: set[tuple[int, int]],
    ray: set[tuple[int, int]],
    *,
    seed: tuple[int, int] | None,
    must_hit: bool,
) -> set[tuple[int, int]] | None:
    shape = _grow_object(grid, size, occupied, ray, seed=seed, must_hit=must_hit)
    if shape is None:
        return None
    for r, c in shape:
        grid[r, c] = BLUE
    occupied.update(shape)
    return shape


def _choose_sizes(n_objects: int) -> tuple[list[int], list[int], int] | None:
    sizes = sorted(random.sample(SIZE_CHOICES, n_objects))
    median = sizes[n_objects // 2]
    low = sizes[: n_objects // 2]
    high = sizes[n_objects // 2 + 1 :]
    hit_count = random.choice(HIT_COUNT_OPTIONS)
    if hit_count >= n_objects:
        return None

    hit = []
    pick_low = random.choice(low)
    hit.append(pick_low)
    keep_pool = [median] + high
    hit.append(random.choice(keep_pool))

    remaining = [s for s in sizes if s not in hit]
    if hit_count == 3:
        extra_choices = []
        for extra in remaining:
            future = [s for s in remaining if s != extra]
            if any(s < median for s in future) and any(s > median for s in future):
                extra_choices.append(extra)
        if not extra_choices:
            return None
        hit.append(random.choice(extra_choices))

    hit = sorted(hit)
    unhit = sorted([s for s in sizes if s not in hit])
    if not any(s < median for s in hit):
        return None
    if not any(s >= median for s in hit):
        return None
    if not any(s < median for s in unhit):
        return None
    if not any(s > median for s in unhit):
        return None
    return hit, unhit, median


def _choose_ray_anchors(ray_cells: list[tuple[int, int]], count: int) -> list[tuple[int, int]] | None:
    usable = ray_cells[1:-1] if len(ray_cells) > 3 else ray_cells[:]
    if len(usable) < count:
        return None
    indexed = list(enumerate(usable))
    random.shuffle(indexed)

    chosen: list[tuple[int, int]] = []
    used_idx: list[int] = []
    for idx, cell in indexed:
        if all(abs(idx - other) >= 2 for other in used_idx):
            chosen.append(cell)
            used_idx.append(idx)
            if len(chosen) == count:
                return chosen
    return None


def _validate_spacing(grid: np.ndarray, marker_cells: set[tuple[int, int]]) -> bool:
    blue_cells = {tuple(map(int, rc)) for rc in np.argwhere(grid == BLUE)}
    for cell in blue_cells:
        if cell in marker_cells:
            return False
        if any(nei in marker_cells for nei in _neighbors8(*cell)):
            return False

    objects = _blue_objects(grid)
    sets = [{tuple(map(int, rc)) for rc in obj["coords"]} for obj in objects]
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            for r, c in sets[i]:
                if any((nr, nc) in sets[j] for nr, nc in _neighbors8(r, c)):
                    return False
    return True


def _semantic_check(inp: np.ndarray) -> bool:
    if inp.ndim != 2 or max(inp.shape) > 30:
        return False

    marker_info = _analyze(inp)
    hit = marker_info["hit"]
    unhit = marker_info["unhit"]
    median = marker_info["median"]

    if len(marker_info["objects"]) not in OBJECT_COUNT_OPTIONS:
        return False
    if any(obj["area"] < 2 or obj["area"] > 10 for obj in hit + unhit):
        return False
    if len(hit) < 2:
        return False
    if not any(obj["area"] < median for obj in hit):
        return False
    if not any(obj["area"] >= median for obj in hit):
        return False
    if not unhit:
        return False
    if not any(obj["area"] < median for obj in unhit):
        return False
    if not any(obj["area"] > median for obj in unhit):
        return False

    out = main(inp)
    removed_areas = []
    kept_hit_areas = []
    for obj in hit:
        coords = tuple(obj["coords"])
        erased = all(out[r, c] == BG for r, c in coords)
        preserved = all(out[r, c] == BLUE for r, c in coords)
        if obj["area"] < median:
            if not erased:
                return False
            removed_areas.append(obj["area"])
        else:
            if not preserved:
                return False
            kept_hit_areas.append(obj["area"])
    if not removed_areas or not kept_hit_areas:
        return False

    for obj in unhit:
        if any(out[r, c] != BLUE for r, c in obj["coords"]):
            return False

    center, step = marker_info["center"], marker_info["step"]
    control = (center[0] - step[0], center[1] - step[1])
    axis = "horizontal" if step[0] != 0 else "vertical"
    if not _validate_spacing(inp, _marker_cells(center, axis, control)):
        return False

    if np.array_equal(inp, out):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(GRID_MIN, GRID_MAX)
    w = random.randint(GRID_MIN, GRID_MAX)
    grid = np.full((h, w), BG, dtype=int)

    marker = random.choice(_marker_candidates(h, w))
    center, axis, control, step = marker
    marker_cells = _marker_cells(center, axis, control)
    for r, c in marker_cells:
        grid[r, c] = RED

    ray_cells = _ray_cells((h, w), center, step)
    ray = set(ray_cells)
    n_objects = random.choice(OBJECT_COUNT_OPTIONS)
    size_plan = _choose_sizes(n_objects)
    if size_plan is None:
        return None
    hit_sizes, unhit_sizes, _median = size_plan

    hit_anchors = _choose_ray_anchors(ray_cells, len(hit_sizes))
    if hit_anchors is None:
        return None

    occupied = set(marker_cells)
    paired_hits = list(zip(sorted(hit_sizes, reverse=True), hit_anchors))
    random.shuffle(paired_hits)
    for size, anchor in paired_hits:
        if _place_object(grid, size, occupied, ray, seed=anchor, must_hit=True) is None:
            return None

    for size in sorted(unhit_sizes, reverse=True):
        if _place_object(grid, size, occupied, ray, seed=None, must_hit=False) is None:
            return None

    return grid if _semantic_check(grid) else None


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        candidate = _gen_attempt()
        if candidate is not None:
            return candidate
    raise RuntimeError("failed to generate a valid new ARC sample")
