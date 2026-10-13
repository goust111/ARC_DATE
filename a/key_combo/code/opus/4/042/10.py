from __future__ import annotations

from collections import Counter
from pathlib import Path
import random
import sys

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(_ARC_UTILS_DIR))

from arc_utils import Color, find_components, get_bg, in_bounds, label_mask


BG = Color.BLACK
BLOCK = Color.BLUE
MARKER_COLORS = (Color.RED, Color.GREEN, Color.YELLOW)
BLOCK_SIZE = 2

CORNER_OFFSETS = {
    "NW": (-1, -1),
    "NE": (-1, 2),
    "SE": (2, 2),
    "SW": (2, -1),
}
CORNER_DIRS = {
    "NW": (-1, -1),
    "NE": (-1, 1),
    "SE": (1, 1),
    "SW": (1, -1),
}
ORIENTATIONS = tuple(CORNER_OFFSETS)


def _diag_neighbors8(r: int, c: int):
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            yield r + dr, c + dc


def _block_bbox(top_left: tuple[int, int]) -> tuple[int, int, int, int]:
    r0, c0 = top_left
    return r0, c0, r0 + 1, c0 + 1


def _extract_blocks(grid: np.ndarray) -> list[tuple[int, int]]:
    comps = find_components(
        grid,
        bg=BG,
        connectivity=4,
        monochromatic=True,
        colors=[BLOCK],
        return_format="dict",
    )
    blocks: list[tuple[int, int]] = []
    for comp in comps:
        r0, c0, r1, c1 = comp["bbox"]
        assert int(comp["area"]) == 4 and r1 == r0 + 1 and c1 == c0 + 1, "blue objects must be solid 2x2 blocks"
        blocks.append((int(r0), int(c0)))
    assert 3 <= len(blocks) <= 4, "expected 3-4 blue blocks"
    return sorted(blocks)


def _emitters(grid: np.ndarray) -> list[dict]:
    h, w = grid.shape
    emitters: list[dict] = []
    for r0, c0 in _extract_blocks(grid):
        colored = []
        for corner, (dr, dc) in CORNER_OFFSETS.items():
            rr = r0 + dr
            cc = c0 + dc
            assert 0 <= rr < h and 0 <= cc < w, "block corner ports must be in bounds"
            val = int(grid[rr, cc])
            if val != BG:
                assert val != BLOCK, "corner port cannot be blue"
                colored.append((corner, rr, cc, val))
        assert len(colored) == 1, "each block must have exactly one colored diagonal corner"
        corner, rr, cc, color = colored[0]
        emitters.append(
            {
                "block": (r0, c0),
                "corner": corner,
                "start": (rr, cc),
                "dir": CORNER_DIRS[corner],
                "color": color,
            }
        )
    return emitters


def _trace_rays(grid: np.ndarray, emitters: list[dict]) -> list[dict]:
    blue_mask = grid == BLOCK
    used: set[tuple[int, int]] = set()
    rays: list[dict] = []
    for emitter in emitters:
        start = emitter["start"]
        dr, dc = emitter["dir"]
        r, c = start
        path: list[tuple[int, int]] = []
        while True:
            assert in_bounds(grid.shape, r, c)
            assert not bool(blue_mask[r, c])
            if (r, c) != start:
                assert int(grid[r, c]) == BG, "ray path may not cover markers or other corners"
            assert (r, c) not in used, "rays must not overlap"
            used.add((r, c))
            path.append((r, c))

            nr = r + dr
            nc = c + dc
            if not in_bounds(grid.shape, nr, nc):
                stop = "boundary"
                break
            if bool(blue_mask[nr, nc]):
                stop = "block"
                break
            r, c = nr, nc

        rays.append({**emitter, "path": tuple(path), "stop": stop})
    return rays


def _obstacle_mask(shape: tuple[int, int], blocks: list[tuple[int, int]], rays: list[dict]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    for r0, c0 in blocks:
        mask[r0 : r0 + 2, c0 : c0 + 2] = True
    for ray in rays:
        for r, c in ray["path"]:
            mask[r, c] = True
    return mask


def _region_infos(grid: np.ndarray, obstacles: np.ndarray) -> tuple[list[dict], np.ndarray]:
    labels, n = label_mask(~obstacles, connectivity=4)
    infos: list[dict] = []
    for rid in range(1, n + 1):
        coords = np.argwhere(labels == rid)
        counts = Counter(
            int(grid[r, c])
            for r, c in coords
            if int(grid[r, c]) in MARKER_COLORS
        )
        assert counts, "every region must contain at least one marker"
        top = counts.most_common()
        assert len(top) == 1 or top[0][1] > top[1][1], "region marker vote must have a strict majority"
        infos.append(
            {
                "id": int(rid),
                "coords": coords,
                "counts": counts,
                "majority": int(top[0][0]),
            }
        )
    return infos, labels


def _render_output(grid: np.ndarray, rays: list[dict], obstacles: np.ndarray) -> np.ndarray:
    infos, _labels = _region_infos(grid, obstacles)
    out = np.asarray(grid, dtype=int).copy()
    for ray in rays:
        for r, c in ray["path"]:
            out[r, c] = int(ray["color"])
    for info in infos:
        fill = info["majority"]
        for r, c in info["coords"]:
            if int(out[r, c]) == BG:
                out[r, c] = fill
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend one diagonal ray from each blue 2x2 block corner, then fill every ray-bounded region by marker majority.

    Input structure:
    - The background is black.
    - The foreground contains 3-4 solid blue 2x2 blocks.
    - Each block has exactly one colored diagonal corner pixel, and the remaining diagonal corners are black.
    - Additional isolated red/green/yellow marker pixels lie away from the blocks and corner pixels.

    Rule:
    - For each block, read which diagonal corner is colored and extend a same-colored ray outward along that diagonal.
    - Stop each ray when the next step would leave the grid or enter another blue block; include the last in-bounds pre-block cell.
    - Treat blue block cells and all ray cells as obstacles, and split the remaining cells into 4-connected regions.
    - In each region, count only the marker pixels and take the strict majority marker color.
    - Recolor every black cell in that region with the majority color, while preserving the markers, blocks, and rays.

    Key insight:
    - The single active corner on each block determines an asymmetric diagonal cut, and only the resulting ray-defined regions tell you which local marker vote controls each background area.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert get_bg(g) == BG, "background must be black"

    blocks = _extract_blocks(g)
    rays = _trace_rays(g, _emitters(g))
    obstacles = _obstacle_mask(g.shape, blocks, rays)
    return _render_output(g, rays, obstacles)


GEN_ATTEMPTS = 600
GRID_SIDE_RANGE = (22, 28)
BLOCK_COUNT_CHOICES = (3, 4)
BLOCK_TOPLEFT_MARGIN = 2
MIN_BLOCK_CHEBYSHEV = 8
BLOCKED_RAY_DISTANCE_RANGE = (6, 10)


def _corner_coord(top_left: tuple[int, int], corner: str) -> tuple[int, int]:
    r0, c0 = top_left
    dr, dc = CORNER_OFFSETS[corner]
    return r0 + dr, c0 + dc


def _block_fits(shape: tuple[int, int], top_left: tuple[int, int]) -> bool:
    h, w = shape
    r0, c0 = top_left
    return 0 <= r0 - 1 and 0 <= c0 - 1 and r0 + 2 < h and c0 + 2 < w


def _chebyshev_far(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return max(abs(a[0] - b[0]), abs(a[1] - b[1])) >= MIN_BLOCK_CHEBYSHEV


def _stop_target_top_left(start: tuple[int, int], corner: str, distance: int) -> tuple[int, int]:
    sr, sc = start
    if corner == "SE":
        return sr + distance, sc + distance
    if corner == "NW":
        return sr - distance - 1, sc - distance - 1
    if corner == "NE":
        return sr - distance - 1, sc + distance
    return sr + distance, sc - distance - 1


def _place_blocks(shape: tuple[int, int], orientations: list[str], require_block_stop: bool) -> list[tuple[int, int]] | None:
    h, w = shape
    placed: list[tuple[int, int] | None] = [None] * len(orientations)

    if require_block_stop:
        target_idx = random.randrange(len(orientations))
        target_corner = orientations[target_idx]
        blocker_idx = (target_idx + 1) % len(orientations)
        for _ in range(80):
            distance = random.randint(*BLOCKED_RAY_DISTANCE_RANGE)
            r0 = random.randint(BLOCK_TOPLEFT_MARGIN, h - BLOCK_TOPLEFT_MARGIN - 4)
            c0 = random.randint(BLOCK_TOPLEFT_MARGIN, w - BLOCK_TOPLEFT_MARGIN - 4)
            first = (r0, c0)
            if not _block_fits(shape, first):
                continue
            second = _stop_target_top_left(_corner_coord(first, target_corner), target_corner, distance)
            if not _block_fits(shape, second):
                continue
            if not _chebyshev_far(first, second):
                continue
            placed[target_idx] = first
            placed[blocker_idx] = second
            break
        if any(item is None for item in (placed[target_idx], placed[blocker_idx])):
            return None

    for _ in range(400):
        if all(item is not None for item in placed):
            break
        r0 = random.randint(BLOCK_TOPLEFT_MARGIN, h - BLOCK_TOPLEFT_MARGIN - 4)
        c0 = random.randint(BLOCK_TOPLEFT_MARGIN, w - BLOCK_TOPLEFT_MARGIN - 4)
        cand = (r0, c0)
        if not _block_fits(shape, cand):
            continue
        current = [item for item in placed if item is not None]
        if any(not _chebyshev_far(cand, other) for other in current):
            continue
        if cand in current:
            continue
        next_idx = next(idx for idx, item in enumerate(placed) if item is None)
        placed[next_idx] = cand
    if any(item is None for item in placed):
        return None
    return [item for item in placed if item is not None]


def _paint_base_grid(shape: tuple[int, int], blocks: list[tuple[int, int]], orientations: list[str]) -> np.ndarray:
    grid = np.full(shape, BG, dtype=int)
    corner_colors = random.choices(MARKER_COLORS, k=len(blocks))
    for (r0, c0), corner, color in zip(blocks, orientations, corner_colors):
        grid[r0 : r0 + 2, c0 : c0 + 2] = BLOCK
        rr, cc = _corner_coord((r0, c0), corner)
        grid[rr, cc] = int(color)
    return grid


def _marker_candidate_mask(base: np.ndarray) -> np.ndarray:
    mask = np.ones(base.shape, dtype=bool)
    for r, c in np.argwhere(base != BG):
        mask[r, c] = False
        for nr, nc in _diag_neighbors8(int(r), int(c)):
            if in_bounds(base.shape, nr, nc):
                mask[nr, nc] = False
    return mask


def _pick_non_adjacent(cells: list[tuple[int, int]], taken: set[tuple[int, int]], count: int) -> list[tuple[int, int]] | None:
    chosen: list[tuple[int, int]] = []
    pool = cells[:]
    random.shuffle(pool)
    for cell in pool:
        if any(max(abs(cell[0] - rr), abs(cell[1] - cc)) <= 1 for rr, cc in taken):
            continue
        if any(max(abs(cell[0] - rr), abs(cell[1] - cc)) <= 1 for rr, cc in chosen):
            continue
        chosen.append(cell)
        if len(chosen) == count:
            return chosen
    return None


def _assign_markers(base: np.ndarray, obstacles: np.ndarray) -> np.ndarray | None:
    labels, n = label_mask(~obstacles, connectivity=4)
    if n < 2:
        return None

    region_ids = list(range(1, n + 1))
    candidate_mask = _marker_candidate_mask(base)
    regions = []
    for rid in region_ids:
        coords = np.argwhere(labels == rid)
        candidates = [
            (int(r), int(c))
            for r, c in coords
            if bool(candidate_mask[r, c])
        ]
        if not candidates:
            return None
        regions.append({"id": rid, "coords": coords, "candidates": candidates, "area": int(coords.shape[0])})

    mixable = [r for r in regions if len(r["candidates"]) >= 3]
    if not mixable:
        return None

    majorities = [random.choice(MARKER_COLORS) for _ in regions]
    if len(set(majorities)) == 1:
        majorities[-1] = next(color for color in MARKER_COLORS if color != majorities[0])
    for region, color in zip(regions, majorities):
        region["majority"] = int(color)

    mix_count = 1 if len(mixable) == 1 else random.choice((1, 2))
    mix_ids = {region["id"] for region in sorted(mixable, key=lambda item: item["area"], reverse=True)[:mix_count]}

    plans: dict[int, list[int]] = {}
    for region in regions:
        majority = region["majority"]
        if region["id"] in mix_ids:
            minority = random.choice([color for color in MARKER_COLORS if color != majority])
            plans[region["id"]] = [majority, majority, minority]
        else:
            plans[region["id"]] = [majority] if len(region["candidates"]) == 1 or random.random() < 0.6 else [majority, majority]

    grid = base.copy()
    taken: set[tuple[int, int]] = set()
    for region in sorted(regions, key=lambda item: len(plans[item["id"]]), reverse=True):
        plan = plans[region["id"]]
        if len(region["candidates"]) < len(plan):
            return None
        chosen = _pick_non_adjacent(region["candidates"], taken, len(plan))
        if chosen is None:
            return None
        for (r, c), color in zip(chosen, plan):
            grid[r, c] = int(color)
            taken.add((r, c))
        region["chosen"] = chosen
    return grid


def _rays_are_semantically_needed(full_labels: np.ndarray, full_infos: list[dict], block_mask: np.ndarray) -> bool:
    no_ray_labels, no_ray_count = label_mask(~block_mask, connectivity=4)
    majority_by_region = {info["id"]: int(info["majority"]) for info in full_infos}
    for rid in range(1, no_ray_count + 1):
        child_regions = {
            int(full_labels[r, c])
            for r, c in np.argwhere(no_ray_labels == rid)
            if int(full_labels[r, c]) != 0
        }
        if len(child_regions) < 2:
            continue
        if len({majority_by_region[child] for child in child_regions}) >= 2:
            return True
    return False


def _generator_checks(grid: np.ndarray, require_block_stop: bool) -> bool:
    try:
        blocks = _extract_blocks(grid)
        emitters = _emitters(grid)
        rays = _trace_rays(grid, emitters)
        block_stops = sum(ray["stop"] == "block" for ray in rays)
        boundary_stops = sum(ray["stop"] == "boundary" for ray in rays)
        if require_block_stop and block_stops < 1:
            return False
        if not require_block_stop and block_stops != 0:
            return False
        if boundary_stops < 1:
            return False

        obstacles = _obstacle_mask(grid.shape, blocks, rays)
        infos, labels = _region_infos(grid, obstacles)
        if len(set(info["majority"] for info in infos)) < 2:
            return False
        if not any(len(info["counts"]) >= 2 for info in infos):
            return False

        block_mask = np.zeros(grid.shape, dtype=bool)
        for r0, c0 in blocks:
            block_mask[r0 : r0 + 2, c0 : c0 + 2] = True
        if not _rays_are_semantically_needed(labels, infos, block_mask):
            return False

        for r, c in np.argwhere(np.isin(grid, MARKER_COLORS)):
            val = int(grid[r, c])
            if val not in MARKER_COLORS:
                continue
            is_corner = any((int(r), int(c)) == emitter["start"] for emitter in emitters)
            if is_corner:
                continue
            for nr, nc in _diag_neighbors8(int(r), int(c)):
                if not in_bounds(grid.shape, nr, nc):
                    continue
                if int(grid[nr, nc]) != BG:
                    return False

        expected = _render_output(grid, rays, obstacles)
        actual = main(grid)
        return bool(np.array_equal(actual, expected))
    except Exception:
        return False


def _gen_attempt() -> np.ndarray | None:
    shape = (
        random.randint(*GRID_SIDE_RANGE),
        random.randint(*GRID_SIDE_RANGE),
    )
    block_count = random.choice(BLOCK_COUNT_CHOICES)
    orientations = random.sample(ORIENTATIONS, block_count)
    require_block_stop = random.random() < 0.6

    blocks = _place_blocks(shape, orientations, require_block_stop)
    if blocks is None:
        return None

    base = _paint_base_grid(shape, blocks, orientations)
    try:
        rays = _trace_rays(base, _emitters(base))
    except Exception:
        return None

    obstacles = _obstacle_mask(shape, blocks, rays)
    grid = _assign_markers(base, obstacles)
    if grid is None:
        return None
    if not _generator_checks(grid, require_block_stop):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 10")
