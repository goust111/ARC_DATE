from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import Color, components_mask, get_bg, in_bounds


BACKGROUND = Color.BLACK
SEED_PALETTE = (Color.BLUE, Color.GREEN, Color.YELLOW)
MARKER_PALETTE = (Color.PINK, Color.ORANGE, Color.TEAL)
GREY = Color.GREY
RED = Color.RED
DIRS_BY_CORNER = {
    (0, 0): (-1, -1),
    (0, 1): (-1, 1),
    (1, 0): (1, -1),
    (1, 1): (1, 1),
}
CORNER_BY_DIR = {v: k for k, v in DIRS_BY_CORNER.items()}
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIRS8 = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0)]

GRID_MIN = 21
GRID_MAX = 28
COMPONENT_COUNT_RANGE = (3, 5)
SEED_COUNT_RANGE = (2, 3)
COMPONENT_GAP = 2
COMPONENT_RETRIES = 150
GEN_RETRIES = 400


def _component_mask(shape: tuple[int, int], coords: np.ndarray) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    mask[coords[:, 0], coords[:, 1]] = True
    return mask


def _find_seeds(grid: np.ndarray) -> list[dict]:
    h, w = grid.shape
    seeds: list[dict] = []
    used: set[tuple[int, int]] = set()
    for r0 in range(h - 1):
        for c0 in range(w - 1):
            patch = grid[r0 : r0 + 2, c0 : c0 + 2]
            vals, counts = np.unique(patch, return_counts=True)
            if BACKGROUND in vals or len(vals) != 2:
                continue
            if RED not in vals:
                continue
            if int(counts[vals.tolist().index(RED)]) != 1:
                continue
            paint_colors = [int(v) for v in vals if int(v) != RED]
            if len(paint_colors) != 1:
                continue
            paint = paint_colors[0]
            if paint == BACKGROUND:
                continue
            paint_count = int(counts[vals.tolist().index(paint)])
            if paint_count != 3:
                continue
            red_pos = np.argwhere(patch == RED)
            if red_pos.shape[0] != 1:
                continue
            cells = {(r0 + dr, c0 + dc) for dr in range(2) for dc in range(2)}
            if used & cells:
                raise AssertionError("overlapping 2x2 seed windows are ambiguous")
            used |= cells
            lr, lc = map(int, red_pos[0])
            seeds.append({"top_left": (r0, c0), "paint": paint, "red_local": (lr, lc)})
    return seeds


def _find_grey_components(grid: np.ndarray) -> list[dict]:
    comps: list[dict] = []
    for coords in components_mask(grid != BACKGROUND, connectivity=4):
        colors = {int(v) for v in grid[coords[:, 0], coords[:, 1]]}
        if GREY not in colors:
            continue
        markers = [c for c in colors if c not in {GREY}]
        if len(markers) != 1 or markers[0] not in MARKER_PALETTE:
            continue
        marker_color = markers[0]
        marker_coords = np.argwhere(grid == marker_color)
        marker_hits = [tuple(map(int, rc)) for rc in marker_coords if any(np.all(rc == p) for p in coords)]
        if len(marker_hits) != 1:
            raise AssertionError("each grey component must contain exactly one marker pixel")
        comp_mask = _component_mask(grid.shape, coords)
        interior: list[tuple[int, int]] = []
        for r_raw, c_raw in coords:
            r = int(r_raw)
            c = int(c_raw)
            if r == 0 or c == 0 or r == grid.shape[0] - 1 or c == grid.shape[1] - 1:
                continue
            if comp_mask[r - 1 : r + 2, c - 1 : c + 2].all():
                interior.append((r, c))
        comps.append(
            {
                "coords": coords,
                "mask": comp_mask,
                "marker_pos": marker_hits[0],
                "marker_color": marker_color,
                "interior": interior,
            }
        )
    return comps


def _grey_owner_map(components: list[dict]) -> dict[tuple[int, int], int]:
    owner: dict[tuple[int, int], int] = {}
    for idx, comp in enumerate(components):
        coords = comp["coords"]
        for r_raw, c_raw in coords:
            r = int(r_raw)
            c = int(c_raw)
            if owner.get((r, c)) is not None:
                raise AssertionError("components must not overlap")
            if comp["marker_pos"] != (r, c):
                owner[(r, c)] = idx
    return owner


def _seed_hit_component(seed: dict, grey_owner: dict[tuple[int, int], int], shape: tuple[int, int]) -> int | None:
    (r0, c0) = seed["top_left"]
    (lr, lc) = seed["red_local"]
    dr, dc = DIRS_BY_CORNER[(lr, lc)]
    rr = r0 + lr
    cc = c0 + lc
    while True:
        rr += dr
        cc += dc
        if not in_bounds(shape, rr, cc):
            return None
        comp_id = grey_owner.get((rr, cc))
        if comp_id is not None:
            return comp_id


def _solve(grid: np.ndarray) -> np.ndarray:
    seeds = _find_seeds(grid)
    assert 2 <= len(seeds) <= 3, "expected 2-3 seeds"
    components = _find_grey_components(grid)
    assert 3 <= len(components) <= 5, "expected 3-5 grey components"
    grey_owner = _grey_owner_map(components)

    out = np.array(grid, dtype=int, copy=True)
    hit_by_component: dict[int, int] = {}
    for seed_idx, seed in enumerate(seeds):
        r0, c0 = seed["top_left"]
        out[r0 : r0 + 2, c0 : c0 + 2] = int(seed["paint"])
        comp_id = _seed_hit_component(seed, grey_owner, grid.shape)
        if comp_id is None:
            raise AssertionError("each seed must hit a grey component")
        if comp_id in hit_by_component:
            raise AssertionError("different seeds must hit different components")
        hit_by_component[comp_id] = seed_idx

    for comp_idx, comp in enumerate(components):
        if comp_idx in hit_by_component:
            fill = int(seeds[hit_by_component[comp_idx]]["paint"])
        else:
            fill = int(comp["marker_color"])
        marker_pos = tuple(comp["marker_pos"])
        for r, c in comp["interior"]:
            if (r, c) == marker_pos:
                continue
            out[r, c] = fill
        out[marker_pos] = int(comp["marker_color"])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each hit component's thick interior from a remote seed ray, while untouched components use their own marker color.

    Input structure:
    - The background is black.
    - There are 2-3 disjoint 2x2 seed blocks; each seed has three cells of one paint color and one red corner cell.
    - There are 3-5 larger grey 4-connected components, each containing exactly one non-grey singleton marker pixel.

    Rule:
    - For every seed, read the red corner position to get one diagonal ray direction, then replace that red cell with the seed's paint color.
    - March from that red corner one diagonal step at a time; the first grey pixel touched determines the hit component, and the ray itself is not drawn.
    - In each grey component, pixels whose full 3x3 neighborhood stays inside the same connected component are its thick interior pixels.
    - A hit component recolors those interior pixels with the hitting seed's paint color; an unhit component recolors those interior pixels with its own marker color; marker pixels themselves stay unchanged.

    Key insight:
    - The marker color is only the default fill source: a distant seed can steal a component's interior color by aiming its diagonal ray at that component first.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    return _solve(g)


def _neighbors4(cell: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = cell
    return [(r + dr, c + dc) for dr, dc in DIRS4]


def _chebyshev_clear(occupied: set[tuple[int, int]], cells: set[tuple[int, int]], gap: int) -> bool:
    for r, c in cells:
        for rr in range(r - gap, r + gap + 1):
            for cc in range(c - gap, c + gap + 1):
                if (rr, cc) in occupied:
                    return False
    return True


def _make_component_shape(rng: np.random.Generator) -> tuple[set[tuple[int, int]], tuple[int, int]]:
    base_h = int(rng.integers(4, 6))
    base_w = int(rng.integers(4, 6))
    cells = {(r, c) for r in range(base_h) for c in range(base_w)}

    sides = ["N", "S", "E", "W"]
    rng.shuffle(sides)
    arm_count = int(rng.integers(1, 4))
    total_cap = int(rng.integers(18, 31))
    for side in sides[:arm_count]:
        if len(cells) >= total_cap:
            break
        length = int(rng.integers(1, min(5, total_cap - len(cells) + 1)))
        if side in {"N", "S"}:
            anchor = int(rng.integers(1, base_w - 1))
            for k in range(1, length + 1):
                rr = -k if side == "N" else base_h - 1 + k
                cells.add((rr, anchor))
        else:
            anchor = int(rng.integers(1, base_h - 1))
            for k in range(1, length + 1):
                cc = -k if side == "W" else base_w - 1 + k
                cells.add((anchor, cc))

    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    normalized = {(r - min_r, c - min_c) for r, c in cells}

    norm_h = max(r for r, _ in normalized) + 1
    norm_w = max(c for _, c in normalized) + 1
    interior = []
    for r in range(1, norm_h - 1):
        for c in range(1, norm_w - 1):
            if all((r + dr, c + dc) in normalized for dr, dc in DIRS8):
                interior.append((r, c))
    if not interior:
        raise AssertionError("component shape lost all thick interior")
    return normalized, tuple(interior[int(rng.integers(0, len(interior)))])


def _place_components(
    rng: np.random.Generator, h: int, w: int, count: int
) -> list[dict] | None:
    occupied: set[tuple[int, int]] = set()
    components: list[dict] = []
    marker_colors = [MARKER_PALETTE[i % len(MARKER_PALETTE)] for i in range(count)]
    rng.shuffle(marker_colors)

    for idx in range(count):
        placed = None
        for _ in range(COMPONENT_RETRIES):
            shape_cells, marker_local = _make_component_shape(rng)
            sh = max(r for r, _ in shape_cells) + 1
            sw = max(c for _, c in shape_cells) + 1
            if sh >= h - 2 or sw >= w - 2:
                continue
            top = int(rng.integers(1, h - sh - 1))
            left = int(rng.integers(1, w - sw - 1))
            global_cells = {(top + r, left + c) for r, c in shape_cells}
            if not _chebyshev_clear(occupied, global_cells, COMPONENT_GAP):
                continue
            marker_pos = (top + marker_local[0], left + marker_local[1])
            placed = {
                "id": idx,
                "cells": global_cells,
                "grey_cells": set(global_cells) - {marker_pos},
                "marker_pos": marker_pos,
                "marker_color": int(marker_colors[idx]),
            }
            occupied |= global_cells
            break
        if placed is None:
            return None
        components.append(placed)
    return components


def _seed_bbox_from_red(red_pos: tuple[int, int], red_local: tuple[int, int]) -> set[tuple[int, int]]:
    rr, cc = red_pos
    lr, lc = red_local
    top = rr - lr
    left = cc - lc
    return {(top + dr, left + dc) for dr in range(2) for dc in range(2)}


def _ray_component_ids(
    red_pos: tuple[int, int], step: tuple[int, int], shape: tuple[int, int], grey_owner: dict[tuple[int, int], int]
) -> list[int]:
    rr, cc = red_pos
    ids: list[int] = []
    seen: set[int] = set()
    while True:
        rr += step[0]
        cc += step[1]
        if not in_bounds(shape, rr, cc):
            return ids
        comp_id = grey_owner.get((rr, cc))
        if comp_id is not None and comp_id not in seen:
            ids.append(comp_id)
            seen.add(comp_id)


def _candidate_seeds_for_component(
    comp: dict,
    shape: tuple[int, int],
    grey_owner: dict[tuple[int, int], int],
    blocked: set[tuple[int, int]],
) -> list[dict]:
    h, w = shape
    candidates: list[dict] = []
    for step, red_local in CORNER_BY_DIR.items():
        dr, dc = step
        for hit in sorted(comp["grey_cells"]):
            if grey_owner.get((hit[0] - dr, hit[1] - dc)) == comp["id"]:
                continue
            max_k = min(
                10,
                hit[0] if dr == 1 else h - 1 - hit[0] if dr == -1 else 10,
                hit[1] if dc == 1 else w - 1 - hit[1] if dc == -1 else 10,
            )
            for k in range(3, max_k + 1):
                red_pos = (hit[0] - k * dr, hit[1] - k * dc)
                seed_cells = _seed_bbox_from_red(red_pos, red_local)
                if any(not in_bounds(shape, r, c) for r, c in seed_cells):
                    continue
                if any((r, c) in blocked for r, c in seed_cells):
                    continue
                clear = True
                rr, cc = red_pos
                for _ in range(1, k):
                    rr += dr
                    cc += dc
                    if (rr, cc) in grey_owner or (rr, cc) in blocked:
                        clear = False
                        break
                if not clear:
                    continue
                ray_ids = _ray_component_ids(red_pos, step, shape, grey_owner)
                if ray_ids != [comp["id"]]:
                    continue
                top = red_pos[0] - red_local[0]
                left = red_pos[1] - red_local[1]
                margin_ok = True
                for r, c in seed_cells:
                    for rr in range(r - 1, r + 2):
                        for cc in range(c - 1, c + 2):
                            if (rr, cc) in blocked and (rr, cc) not in seed_cells:
                                margin_ok = False
                                break
                        if not margin_ok:
                            break
                    if not margin_ok:
                        break
                if not margin_ok:
                    continue
                candidates.append(
                    {
                        "top_left": (top, left),
                        "red_local": red_local,
                        "step": step,
                        "target_id": comp["id"],
                        "hit": hit,
                    }
                )
    return candidates


def _render_input(
    h: int,
    w: int,
    components: list[dict],
    seeds: list[dict],
) -> np.ndarray:
    grid = np.full((h, w), BACKGROUND, dtype=int)
    for comp in components:
        for r, c in comp["grey_cells"]:
            grid[r, c] = GREY
        mr, mc = comp["marker_pos"]
        grid[mr, mc] = int(comp["marker_color"])
    for seed in seeds:
        r0, c0 = seed["top_left"]
        grid[r0 : r0 + 2, c0 : c0 + 2] = int(seed["paint"])
        lr, lc = seed["red_local"]
        grid[r0 + lr, c0 + lc] = RED
    return grid


def _oracle_output(
    grid: np.ndarray,
    components: list[dict],
    seeds: list[dict],
) -> np.ndarray:
    out = np.array(grid, copy=True)
    for seed in seeds:
        r0, c0 = seed["top_left"]
        out[r0 : r0 + 2, c0 : c0 + 2] = int(seed["paint"])

    comp_masks = []
    for comp in components:
        mask = np.zeros(grid.shape, dtype=bool)
        for r, c in comp["cells"]:
            mask[r, c] = True
        comp_masks.append(mask)

    hit_map = {seed["target_id"]: int(seed["paint"]) for seed in seeds}
    for comp, mask in zip(components, comp_masks, strict=False):
        fill = hit_map.get(comp["id"], int(comp["marker_color"]))
        for r, c in comp["cells"]:
            if (r, c) == comp["marker_pos"]:
                continue
            if r == 0 or c == 0 or r == grid.shape[0] - 1 or c == grid.shape[1] - 1:
                continue
            if mask[r - 1 : r + 2, c - 1 : c + 2].all():
                out[r, c] = fill
        mr, mc = comp["marker_pos"]
        out[mr, mc] = int(comp["marker_color"])
    return out


def _validate_semantics(
    inp: np.ndarray,
    components: list[dict],
    seeds: list[dict],
    expected: np.ndarray,
) -> bool:
    parsed_seeds = _find_seeds(inp)
    if len(parsed_seeds) != len(seeds):
        return False
    parsed_components = _find_grey_components(inp)
    if len(parsed_components) != len(components):
        return False
    grey_owner = _grey_owner_map(parsed_components)
    hits = {}
    for seed in parsed_seeds:
        comp_id = _seed_hit_component(seed, grey_owner, inp.shape)
        if comp_id is None or comp_id in hits:
            return False
        hits[comp_id] = int(seed["paint"])
        step = DIRS_BY_CORNER[seed["red_local"]]
        red = (seed["top_left"][0] + seed["red_local"][0], seed["top_left"][1] + seed["red_local"][1])
        if _ray_component_ids(red, step, inp.shape, grey_owner) != [comp_id]:
            return False
    if len(parsed_components) - len(hits) < 1:
        return False
    for comp in parsed_components:
        if not comp["interior"]:
            return False
        marker = tuple(comp["marker_pos"])
        solved = expected.copy()
        if solved[marker] != inp[marker]:
            return False
    return True


def _attempt_generate(rng: np.random.Generator) -> np.ndarray | None:
    h = int(rng.integers(GRID_MIN, GRID_MAX + 1))
    w = int(rng.integers(GRID_MIN, GRID_MAX + 1))
    component_count = int(rng.integers(COMPONENT_COUNT_RANGE[0], COMPONENT_COUNT_RANGE[1] + 1))
    max_seeds = min(SEED_COUNT_RANGE[1], component_count - 1)
    seed_count = int(rng.integers(SEED_COUNT_RANGE[0], max_seeds + 1))

    components = _place_components(rng, h, w, component_count)
    if components is None:
        return None

    grey_owner = {}
    blocked = set()
    for comp in components:
        blocked |= set(comp["cells"])
        for cell in comp["grey_cells"]:
            grey_owner[cell] = comp["id"]

    target_ids = rng.choice(component_count, size=seed_count, replace=False).tolist()
    paint_colors = list(rng.choice(np.array(SEED_PALETTE, dtype=int), size=seed_count, replace=False))
    seeds: list[dict] = []
    used_seed_cells: set[tuple[int, int]] = set()
    for target_id, paint in zip(target_ids, paint_colors, strict=False):
        candidates = _candidate_seeds_for_component(components[target_id], (h, w), grey_owner, blocked | used_seed_cells)
        if not candidates:
            return None
        choice = candidates[int(rng.integers(0, len(candidates)))]
        seed_cells = {(choice["top_left"][0] + dr, choice["top_left"][1] + dc) for dr in range(2) for dc in range(2)}
        used_seed_cells |= seed_cells
        seeds.append(
            {
                "top_left": choice["top_left"],
                "red_local": choice["red_local"],
                "paint": int(paint),
                "target_id": int(target_id),
            }
        )

    inp = _render_input(h, w, components, seeds)
    expected = _oracle_output(inp, components, seeds)
    if not _validate_semantics(inp, components, seeds, expected):
        return None
    solved = main(inp)
    if not np.array_equal(solved, expected):
        return None
    if np.array_equal(inp, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_RETRIES):
        inp = _attempt_generate(rng)
        if inp is not None:
            return inp
    raise RuntimeError("failed to generate a non-degenerate input")
