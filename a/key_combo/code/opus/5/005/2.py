from __future__ import annotations

import random
import sys
from collections import deque
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(_ARC_UTILS_DIR))

from arc_utils import components_mask, get_bg, in_bounds, neighbors4


BG = 7
WALL = 6
SOURCE = 4
PROTO_COLOR = {"A": 1, "B": 2}
TOGGLE = {"A": "B", "B": "A"}
OFFSETS = {
    "A": [(-1, 0), (0, -1), (0, 0), (0, 1), (1, 0)],
    "B": [(-1, -1), (-1, 1), (0, 0), (1, -1), (1, 1)],
}
GEN_RETRIES = 600


def _sprite_cells(center: tuple[int, int], kind: str) -> list[tuple[int, int]]:
    r, c = center
    return [(r + dr, c + dc) for dr, dc in OFFSETS[kind]]


def _detect_sprite_at(grid: np.ndarray, r: int, c: int, bg: int) -> dict | None:
    h, w = grid.shape
    if not (1 <= r < h - 1 and 1 <= c < w - 1):
        return None

    for kind, color in PROTO_COLOR.items():
        good = True
        source_here = int(grid[r, c]) == SOURCE
        if int(grid[r, c]) not in {color, SOURCE}:
            continue
        expected = {(r + dr, c + dc) for dr, dc in OFFSETS[kind]}
        for rr in range(r - 1, r + 2):
            for cc in range(c - 1, c + 2):
                val = int(grid[rr, cc])
                if (rr, cc) == (r, c):
                    if val not in {color, SOURCE}:
                        good = False
                elif (rr, cc) in expected:
                    if val != color:
                        good = False
                elif val != bg:
                    good = False
        if good:
            return {
                "center": (r, c),
                "kind": kind,
                "source": source_here,
                "cells": _sprite_cells((r, c), kind),
            }
    return None


def _find_sprites(grid: np.ndarray) -> list[dict]:
    bg = int(get_bg(grid))
    sprites: list[dict] = []
    used_centers: set[tuple[int, int]] = set()
    used_cells: set[tuple[int, int]] = set()
    h, w = grid.shape
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            sprite = _detect_sprite_at(grid, r, c, bg)
            if sprite is None:
                continue
            center = sprite["center"]
            if center in used_centers:
                continue
            cells = set(sprite["cells"])
            assert not (cells & used_cells), "sprite footprints must be disjoint"
            used_centers.add(center)
            used_cells |= cells
            sprites.append(sprite)

    colored = {
        (int(r), int(c))
        for r, c in np.argwhere(np.isin(grid, [PROTO_COLOR["A"], PROTO_COLOR["B"], SOURCE]))
    }
    assert used_cells == colored, "all non-wall non-background colored cells must belong to sprites"
    sources = [sprite for sprite in sprites if sprite["source"]]
    assert len(sources) == 1, "expected exactly one marked source sprite"
    assert 6 <= len(sprites) <= 8, "expected 6-8 sprites"
    return sprites


def _reachable_background(grid: np.ndarray, source_cells: list[tuple[int, int]], bg: int) -> np.ndarray:
    h, w = grid.shape
    reach = np.zeros((h, w), dtype=bool)
    q: deque[tuple[int, int]] = deque()
    for r, c in source_cells:
        for nr, nc in neighbors4(r, c):
            if in_bounds(grid.shape, nr, nc) and int(grid[nr, nc]) == bg and not reach[nr, nc]:
                reach[nr, nc] = True
                q.append((nr, nc))
    while q:
        r, c = q.popleft()
        for nr, nc in neighbors4(r, c):
            if in_bounds(grid.shape, nr, nc) and not reach[nr, nc] and int(grid[nr, nc]) == bg:
                reach[nr, nc] = True
                q.append((nr, nc))
    return reach


def _sprite_is_reachable(sprite: dict, reach: np.ndarray) -> bool:
    if sprite["source"]:
        return True
    for r, c in sprite["cells"]:
        for nr, nc in neighbors4(r, c):
            if in_bounds(reach.shape, nr, nc) and bool(reach[nr, nc]):
                return True
    return False


def _paint_sprite(grid: np.ndarray, center: tuple[int, int], kind: str) -> None:
    color = PROTO_COLOR[kind]
    for r, c in _sprite_cells(center, kind):
        grid[r, c] = color


def analyze(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    sprites = _find_sprites(g)
    source = next(sprite for sprite in sprites if sprite["source"])
    reach = _reachable_background(g, source["cells"], bg)
    reachable = []
    unreachable = []
    for sprite in sprites:
        (reachable if _sprite_is_reachable(sprite, reach) else unreachable).append(sprite)
    return {
        "grid": g,
        "bg": bg,
        "sprites": sprites,
        "source": source,
        "reach": reach,
        "reachable": reachable,
        "unreachable": unreachable,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Toggle only the sprites that are background-reachable from the marked source through the wall-divided canvas.

    Input structure:
    - Orange (7) is the background. Magenta (6) pixels form one connected wall barrier that partitions the background.
    - The foreground sprites are exact 3x3 prototypes centered on one pixel: prototype A is a blue plus, prototype B is a red diagonal diamond.
    - Exactly one sprite center is replaced by a yellow marker, identifying the source sprite without changing its surrounding arms.

    Rule:
    - Identify every sprite and determine whether it is prototype A or prototype B from its 3x3 local pattern.
    - Starting from the source sprite, flood through 4-connected background cells that can be reached without crossing walls or sprite pixels.
    - Any sprite that touches that reachable background by a 4-neighbor relation is reachable and toggles to the opposite prototype at the same center.
    - Unreachable sprites stay unchanged, and the source marker is overwritten by the center color of its toggled prototype.

    Key insight:
    - Prototype identity answers "what does this sprite become", while wall-gated background reachability answers "which sprites change at all"; both signals are required.
    """
    info = analyze(grid)
    g = info["grid"]
    bg = info["bg"]
    out = g.copy()

    for sprite in info["sprites"]:
        for r, c in sprite["cells"]:
            out[r, c] = bg

    for sprite in info["sprites"]:
        kind = sprite["kind"]
        if sprite in info["reachable"]:
            kind = TOGGLE[kind]
        _paint_sprite(out, sprite["center"], kind)
    return out


def _draw_path(points: list[tuple[int, int]], shape: tuple[int, int]) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    h, w = shape
    for (r0, c0), (r1, c1) in zip(points, points[1:], strict=False):
        if r0 == r1:
            step = 1 if c1 >= c0 else -1
            for c in range(c0, c1 + step, step):
                assert 0 <= r0 < h and 0 <= c < w
                cells.add((r0, c))
        else:
            assert c0 == c1
            step = 1 if r1 >= r0 else -1
            for r in range(r0, r1 + step, step):
                assert 0 <= r < h and 0 <= c0 < w
                cells.add((r, c0))
    return cells


def _wall_regions(mask: np.ndarray) -> list[np.ndarray]:
    return components_mask(~mask, connectivity=4)


def _build_wall(h: int, w: int) -> np.ndarray | None:
    want_three = random.random() < 0.45
    if want_three:
        side = random.choice(["left", "right"])
        if side == "left":
            c0 = random.randint(2, 4)
            c1 = c0 + random.randint(-1, 2)
        else:
            c0 = random.randint(w - 5, w - 3)
            c1 = c0 + random.randint(-2, 1)
        c0 = max(1, min(w - 2, c0))
        c1 = max(1, min(w - 2, c1))
        r_turn = random.randint(4, h - 5)
        points = [(0, c0), (r_turn, c0), (r_turn, c1), (h - 1, c1)]
        wall = _draw_path(points, (h, w))
        branch_col = c0 if side == "left" else c1
        branch_row = random.randint(2, h - 3)
        branch_end = 0 if side == "left" else w - 1
        wall |= _draw_path([(branch_row, branch_col), (branch_row, branch_end)], (h, w))
    else:
        c0 = random.randint(4, w - 5)
        c1 = c0 + random.randint(-2, 2)
        c1 = max(2, min(w - 3, c1))
        r_turn = random.randint(4, h - 5)
        wall = _draw_path([(0, c0), (r_turn, c0), (r_turn, c1), (h - 1, c1)], (h, w))

    wall_mask = np.zeros((h, w), dtype=bool)
    for r, c in wall:
        wall_mask[r, c] = True

    if not 15 <= int(wall_mask.sum()) <= 25:
        return None
    wall_comps = components_mask(wall_mask, connectivity=4)
    if len(wall_comps) != 1:
        return None
    regions = _wall_regions(wall_mask)
    if len(regions) not in {2, 3}:
        return None
    if want_three != (len(regions) == 3):
        return None
    return wall_mask


def _region_label_grid(wall_mask: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
    regions = _wall_regions(wall_mask)
    labels = np.full(wall_mask.shape, -1, dtype=int)
    for idx, coords in enumerate(regions):
        labels[coords[:, 0], coords[:, 1]] = idx
    return labels, regions


def _candidate_centers(wall_mask: np.ndarray, labels: np.ndarray, region_id: int) -> list[tuple[int, int]]:
    h, w = wall_mask.shape
    out: list[tuple[int, int]] = []
    for r in range(2, h - 2):
        for c in range(2, w - 2):
            if labels[r, c] != region_id:
                continue
            box = labels[r - 2 : r + 3, c - 2 : c + 3]
            if box.shape != (5, 5):
                continue
            if np.all(box == region_id):
                out.append((r, c))
    random.shuffle(out)
    return out


def _sample_centers(candidates: list[tuple[int, int]], count: int) -> list[tuple[int, int]] | None:
    for _ in range(120):
        order = candidates[:]
        random.shuffle(order)
        chosen: list[tuple[int, int]] = []
        for r, c in order:
            if all(abs(r - rr) > 4 or abs(c - cc) > 4 for rr, cc in chosen):
                chosen.append((r, c))
                if len(chosen) == count:
                    return chosen
    return None


def _pick_allocations(capacities: list[int], total_unreachable: int) -> list[int] | None:
    n = len(capacities)
    if n == 1:
        return [total_unreachable] if capacities[0] >= total_unreachable else None

    options: list[list[int]] = []
    for first in range(1, total_unreachable):
        second = total_unreachable - first
        pair = [first, second]
        if all(pair[i] <= capacities[i] for i in range(2)):
            options.append(pair)
    if capacities[0] >= total_unreachable:
        options.append([total_unreachable, 0])
    if capacities[1] >= total_unreachable:
        options.append([0, total_unreachable])
    return random.choice(options) if options else None


def _assign_kinds(n: int, source_kind: str | None = None) -> list[str]:
    assert n >= 2
    while True:
        kinds = [random.choice(["A", "B"]) for _ in range(n)]
        if source_kind is not None:
            kinds[0] = source_kind
        if len(set(kinds)) == 2:
            return kinds


def _render_input(
    shape: tuple[int, int],
    wall_mask: np.ndarray,
    placements: list[tuple[tuple[int, int], str, bool]],
) -> np.ndarray:
    grid = np.full(shape, BG, dtype=int)
    grid[wall_mask] = WALL
    for center, kind, is_source in placements:
        color = SOURCE if is_source else PROTO_COLOR[kind]
        for r, c in _sprite_cells(center, kind):
            grid[r, c] = PROTO_COLOR[kind]
        if is_source:
            cr, cc = center
            grid[cr, cc] = color
    return grid


def _wall_region_of_sprite(labels: np.ndarray, sprite: dict) -> int:
    for r, c in sprite["cells"]:
        for nr, nc in neighbors4(r, c):
            if in_bounds(labels.shape, nr, nc) and labels[nr, nc] >= 0:
                return int(labels[nr, nc])
    raise AssertionError("every sprite should touch some wall-defined background region")


def _validate_sample(grid: np.ndarray) -> bool:
    if grid.ndim != 2 or max(grid.shape) > 30:
        return False
    if int(np.sum(grid == WALL)) not in range(15, 26):
        return False

    wall_mask = grid == WALL
    if len(components_mask(wall_mask, connectivity=4)) != 1:
        return False
    labels, wall_regions = _region_label_grid(wall_mask)
    if len(wall_regions) not in {2, 3}:
        return False

    sprites = _find_sprites(grid)
    if len(sprites) < 6 or len(sprites) > 8:
        return False

    if len(components_mask(grid == BG, connectivity=4)) != len(wall_regions):
        return False

    info = analyze(grid)
    reach_kinds = {sprite["kind"] for sprite in info["reachable"]}
    unreach_kinds = {sprite["kind"] for sprite in info["unreachable"]}
    if len(info["reachable"]) < 3 or len(info["unreachable"]) < 2:
        return False
    if reach_kinds != {"A", "B"} or unreach_kinds != {"A", "B"}:
        return False

    source = info["source"]
    src_region = _wall_region_of_sprite(labels, source)
    reach_regions = {_wall_region_of_sprite(labels, sprite) for sprite in info["reachable"]}
    unreach_regions = {_wall_region_of_sprite(labels, sprite) for sprite in info["unreachable"]}
    if reach_regions != {src_region}:
        return False
    if src_region in unreach_regions:
        return False

    out = main(grid)
    if np.array_equal(grid, out):
        return False
    global_toggle = grid.copy()
    for sprite in sprites:
        for r, c in sprite["cells"]:
            global_toggle[r, c] = BG
    for sprite in sprites:
        _paint_sprite(global_toggle, sprite["center"], TOGGLE[sprite["kind"]])
    if np.array_equal(global_toggle, out):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(17, 20)
    w = random.randint(17, 20)
    wall_mask = _build_wall(h, w)
    if wall_mask is None:
        return None

    labels, regions = _region_label_grid(wall_mask)
    region_ids = list(range(len(regions)))
    candidates = {rid: _candidate_centers(wall_mask, labels, rid) for rid in region_ids}
    source_options = [rid for rid in region_ids if len(candidates[rid]) >= 3]
    if not source_options:
        return None
    source_region = random.choice(source_options)
    other_regions = [rid for rid in region_ids if rid != source_region]
    if not other_regions or sum(len(candidates[rid]) for rid in other_regions) < 2:
        return None

    total = random.randint(6, 8)
    reachable_count = random.randint(3, min(4, total - 2))
    unreachable_count = total - reachable_count
    allocations = _pick_allocations([len(candidates[rid]) for rid in other_regions], unreachable_count)
    if allocations is None:
        return None

    chosen: dict[int, list[tuple[int, int]]] = {}
    source_centers = _sample_centers(candidates[source_region], reachable_count)
    if source_centers is None:
        return None
    chosen[source_region] = source_centers
    for rid, count in zip(other_regions, allocations, strict=False):
        if count == 0:
            chosen[rid] = []
            continue
        sample = _sample_centers(candidates[rid], count)
        if sample is None:
            return None
        chosen[rid] = sample

    source_kind = random.choice(["A", "B"])
    reach_kinds = _assign_kinds(reachable_count, source_kind=source_kind)
    unreach_kinds = _assign_kinds(unreachable_count)

    placements: list[tuple[tuple[int, int], str, bool]] = []
    for idx, center in enumerate(chosen[source_region]):
        placements.append((center, reach_kinds[idx], idx == 0))

    k = 0
    for rid in other_regions:
        for center in chosen[rid]:
            placements.append((center, unreach_kinds[k], False))
            k += 1

    grid = _render_input((h, w), wall_mask, placements)
    if not _validate_sample(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid reachability-gated prototype-swap input")
