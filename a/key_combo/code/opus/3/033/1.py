from __future__ import annotations

from collections import Counter
from importlib.machinery import SourceFileLoader
from itertools import combinations
from pathlib import Path
import random

import numpy as np


_ARC_UTILS_PATH = Path("/home/lixinyu/snap/new/solvers/arc_utils.py")


def _fallback_get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, np.asarray(grid).reshape(-1))).most_common(1)[0][0])


def _fallback_find_components(
    grid: np.ndarray,
    *,
    bg: int | None = None,
    connectivity: int = 4,
    monochromatic: bool = True,
    return_format: str = "dict",
) -> list[dict]:
    if connectivity != 4 or not monochromatic or return_format != "dict":
        raise ValueError("fallback find_components only supports 4-connected monochromatic dict output")

    g = np.asarray(grid, dtype=int)
    bg = _fallback_get_bg(g) if bg is None else int(bg)
    visited = np.zeros(g.shape, dtype=bool)
    components: list[dict] = []

    for r in range(g.shape[0]):
        for c in range(g.shape[1]):
            color = int(g[r, c])
            if color == bg or visited[r, c]:
                continue
            stack = [(r, c)]
            visited[r, c] = True
            coords = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < g.shape[0] and 0 <= nc < g.shape[1] and not visited[nr, nc] and int(g[nr, nc]) == color:
                        visited[nr, nc] = True
                        stack.append((nr, nc))
            arr = np.array(coords, dtype=int)
            r0, c0 = arr.min(axis=0)
            r1, c1 = arr.max(axis=0)
            components.append(
                {
                    "color": color,
                    "coords": arr,
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    "area": int(arr.shape[0]),
                }
            )
    return components


try:
    _shared = SourceFileLoader("_shared_arc_utils_for_033_1", str(_ARC_UTILS_PATH)).load_module()
    get_bg = _shared.get_bg
    find_components = _shared.find_components
except Exception:
    get_bg = _fallback_get_bg
    find_components = _fallback_find_components


BACKGROUND = 0
COLOR_POOL = [1, 2, 3, 4, 5, 6, 7, 8, 9]
AREA_CHOICES = tuple(range(3, 9))
GRID_H_RANGE = (12, 16)
GRID_W_RANGE = (12, 16)
OBJECT_COUNT_CHOICES = [3, 3, 4, 4, 5]
MAX_SHAPE_SIDE = 4
MAX_PAIR_OVERLAP = 3
MAX_TOTAL_OVERLAP = 6
CLIP_MODE_PROB = 0.55
MAX_SHAPE_ATTEMPTS = 80
MAX_GEN_ATTEMPTS = 500


def _normalize_coords(coords: np.ndarray) -> np.ndarray:
    arr = np.asarray(coords, dtype=int)
    arr = arr - arr.min(axis=0, keepdims=True)
    order = np.lexsort((arr[:, 1], arr[:, 0]))
    return arr[order]


def _coords_to_mask(coords: np.ndarray) -> np.ndarray:
    pts = _normalize_coords(coords)
    h = int(pts[:, 0].max()) + 1
    w = int(pts[:, 1].max()) + 1
    mask = np.zeros((h, w), dtype=bool)
    mask[pts[:, 0], pts[:, 1]] = True
    return mask


def _mask_to_coords(mask: np.ndarray) -> np.ndarray:
    return _normalize_coords(np.argwhere(mask))


def _random_polyomino(area: int) -> np.ndarray:
    for _ in range(MAX_SHAPE_ATTEMPTS):
        cells = {(0, 0)}
        while len(cells) < area:
            frontier = []
            for r, c in cells:
                for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                    if (nr, nc) not in cells:
                        frontier.append((nr, nc))
            frontier = sorted(set(frontier))
            weights = [1 + sum((nr + dr, nc + dc) in cells for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1))) for nr, nc in frontier]
            cells.add(random.choices(frontier, weights=weights, k=1)[0])

        coords = _normalize_coords(np.array(sorted(cells), dtype=int))
        mask = _coords_to_mask(coords)
        if random.random() < 0.5:
            mask = np.fliplr(mask)
        mask = np.rot90(mask, random.randrange(4))
        coords = _mask_to_coords(mask)
        h = int(coords[:, 0].max()) + 1
        w = int(coords[:, 1].max()) + 1
        if h <= MAX_SHAPE_SIDE and w <= MAX_SHAPE_SIDE:
            return coords
    raise RuntimeError(f"failed to build a compact polyomino of area {area}")


def _placement_candidates(shape: np.ndarray, area: int, height: int, width: int) -> list[dict]:
    shape = _normalize_coords(shape)
    shape_h = int(shape[:, 0].max()) + 1
    shape_w = int(shape[:, 1].max()) + 1
    placements: list[dict] = []

    for r0 in range(height - shape_h + 1):
        for c0 in range(width - shape_w + 1):
            coords = shape + np.array([r0, c0], dtype=int)
            if float(coords[:, 0].max()) >= height / 2.0:
                continue
            landing = coords.copy()
            landing[:, 0] += int(area)
            keep = landing[:, 0] < height
            if not bool(keep.any()):
                continue
            visible = landing[keep]
            placements.append(
                {
                    "coords": coords,
                    "landing": visible,
                    "landing_set": frozenset(map(tuple, visible.tolist())),
                    "clip": bool((~keep).any()),
                }
            )
    return placements


def _coords_separated(a: np.ndarray, b: np.ndarray) -> bool:
    for r, c in a:
        if np.any(np.max(np.abs(b - np.array([r, c], dtype=int)), axis=1) <= 1):
            return False
    return True


def _landing_overlap_count(p1: dict, p2: dict) -> int:
    return len(p1["landing_set"] & p2["landing_set"])


def _overlap_position_count(selected: dict[int, dict]) -> int:
    hits: Counter[tuple[int, int]] = Counter()
    for placement in selected.values():
        hits.update(placement["landing_set"])
    return sum(count > 1 for count in hits.values())


def _visible_counts(selected: dict[int, dict], areas: list[int]) -> dict[int, int]:
    ownership: dict[tuple[int, int], int] = {}
    for idx in sorted(selected, key=lambda k: (int(areas[k]), int(k))):
        for cell in selected[idx]["landing_set"]:
            ownership[cell] = idx

    counts = {idx: 0 for idx in selected}
    for idx in ownership.values():
        counts[idx] += 1
    return counts


def _pick_collision_pair(i: int, j: int, allowed: list[list[dict]], areas: list[int]) -> tuple[dict, dict] | None:
    left = allowed[i][:]
    right = allowed[j][:]
    random.shuffle(left)
    random.shuffle(right)
    max_pair_overlap = min(MAX_PAIR_OVERLAP, min(int(areas[i]), int(areas[j])) - 1)

    for p1 in left:
        for p2 in right:
            if not _coords_separated(p1["coords"], p2["coords"]):
                continue
            overlap = _landing_overlap_count(p1, p2)
            if not (1 <= overlap <= max_pair_overlap):
                continue
            visible = _visible_counts({i: p1, j: p2}, areas)
            if min(visible.values()) >= 1:
                return p1, p2
    return None


def _place_remaining(
    order: list[int],
    allowed: list[list[dict]],
    selected: dict[int, dict],
    areas: list[int],
    pos: int = 0,
) -> bool:
    if pos == len(order):
        return True

    idx = order[pos]
    candidates = allowed[idx][:]
    random.shuffle(candidates)

    for placement in candidates:
        if any(not _coords_separated(placement["coords"], selected[other]["coords"]) for other in selected):
            continue

        selected[idx] = placement
        if _overlap_position_count(selected) <= MAX_TOTAL_OVERLAP and min(_visible_counts(selected, areas).values()) >= 1:
            if _place_remaining(order, allowed, selected, areas, pos + 1):
                return True
        del selected[idx]

    return False


def _build_grid(height: int, width: int, placements: dict[int, dict], colors: list[int]) -> np.ndarray:
    grid = np.full((height, width), BACKGROUND, dtype=int)
    for idx, placement in placements.items():
        rr = placement["coords"][:, 0]
        cc = placement["coords"][:, 1]
        grid[rr, cc] = int(colors[idx])
    return grid


def _validate_scene(grid: np.ndarray, placements: dict[int, dict], areas: list[int], colors: list[int]) -> bool:
    height, width = grid.shape
    if max(height, width) > 30:
        return False

    components = find_components(grid, bg=BACKGROUND, connectivity=4, monochromatic=True, return_format="dict")
    if len(components) != len(areas):
        return False

    found_areas = sorted(int(comp["area"]) for comp in components)
    if found_areas != sorted(map(int, areas)):
        return False

    found_colors = sorted(int(comp["color"]) for comp in components)
    if found_colors != sorted(map(int, colors)):
        return False

    comp_coords = [np.asarray(comp["coords"], dtype=int) for comp in components]
    for coords in comp_coords:
        if float(coords[:, 0].max()) >= height / 2.0:
            return False
    for a, b in combinations(comp_coords, 2):
        if not _coords_separated(a, b):
            return False

    if _overlap_position_count(placements) < 1:
        return False
    if _overlap_position_count(placements) > MAX_TOTAL_OVERLAP:
        return False
    if min(_visible_counts(placements, areas).values()) < 1:
        return False

    out = main(grid)
    if out.shape != grid.shape or np.array_equal(out, grid):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(*GRID_H_RANGE)
    width = random.randint(*GRID_W_RANGE)
    object_count = random.choice(OBJECT_COUNT_CHOICES)
    areas = random.sample(list(AREA_CHOICES), object_count)
    colors = random.sample(COLOR_POOL, object_count)

    try:
        shapes = [_random_polyomino(area) for area in areas]
    except RuntimeError:
        return None

    all_candidates = [_placement_candidates(shape, area, height, width) for shape, area in zip(shapes, areas)]
    if any(not candidates for candidates in all_candidates):
        return None

    clip_mode = random.random() < CLIP_MODE_PROB
    clip_sources = [idx for idx, candidates in enumerate(all_candidates) if any(p["clip"] for p in candidates)]
    if clip_mode and not clip_sources:
        return None
    clip_index = random.choice(clip_sources) if clip_mode else None

    allowed: list[list[dict]] = []
    for idx, candidates in enumerate(all_candidates):
        if clip_mode and idx == clip_index:
            filtered = [p for p in candidates if p["clip"]]
        elif not clip_mode:
            filtered = [p for p in candidates if not p["clip"]]
        else:
            filtered = candidates[:]
        if not filtered:
            return None
        allowed.append(filtered)

    pair_indices = list(combinations(range(object_count), 2))
    random.shuffle(pair_indices)
    for i, j in pair_indices:
        picked = _pick_collision_pair(i, j, allowed, areas)
        if picked is None:
            continue

        selected = {i: picked[0], j: picked[1]}
        remaining = [idx for idx in range(object_count) if idx not in selected]
        remaining.sort(key=lambda idx: len(allowed[idx]))

        if _place_remaining(remaining, allowed, selected, areas):
            if clip_mode and not any(selected[idx]["clip"] for idx in selected):
                continue
            grid = _build_grid(height, width, selected, colors)
            if _validate_scene(grid, selected, areas, colors):
                return grid

    return None


def main(grid: np.ndarray) -> np.ndarray:
    """
    Drop each monochromatic object downward by its own area, letting larger objects win overlaps.

    Input structure:
    - A black background contains several separated 4-connected monochromatic objects in the upper half.
    - Each object has its own color and a distinct pixel area.
    - Some landed objects may overlap after falling, and some pixels may fall past the bottom edge.

    Rule:
    - Clear the canvas to background.
    - Find every 4-connected monochromatic object and count its pixels.
    - Translate each whole object straight downward by exactly that area, clipping any pixels below the grid.
    - Paint landed objects from smaller area to larger area so larger objects overwrite smaller ones at collisions.

    Key insight:
    - Area plays two roles at once: it sets how far an object falls and also which object's color survives wherever landed shapes collide.
    """
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    objects = find_components(g, bg=bg, connectivity=4, monochromatic=True, return_format="dict")

    out = np.full_like(g, bg)
    height = g.shape[0]
    ordered = sorted(objects, key=lambda obj: (int(obj["area"]), int(obj["bbox"][0]), int(obj["bbox"][1]), int(obj["color"])))

    for obj in ordered:
        coords = np.asarray(obj["coords"], dtype=int)
        landing = coords.copy()
        landing[:, 0] += int(obj["area"])
        keep = landing[:, 0] < height
        if bool(keep.any()):
            visible = landing[keep]
            out[visible[:, 0], visible[:, 1]] = int(obj["color"])

    return out


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid collision-priority slide input")
