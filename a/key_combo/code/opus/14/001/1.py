import random
from collections import deque

import numpy as np


BG = 0
NOISE = 7
TEAL = 8
OBJECT_COLORS = (1, 2, 3, 4)
N4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
MAX_RETRIES = 400


def _neighbors4(r: int, c: int):
    for dr, dc in N4:
        yield r + dr, c + dc


def _color_components(grid: np.ndarray, ignored: set[int]) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    components = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if seen[r, c] or color in ignored:
                continue
            queue = deque([(r, c)])
            seen[r, c] = True
            cells = []
            while queue:
                cr, cc = queue.popleft()
                cells.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            coords = np.array(cells, dtype=int)
            r0, c0 = coords.min(axis=0)
            r1, c1 = coords.max(axis=0)
            components.append(
                {
                    "color": color,
                    "coords": coords,
                    "cells": set(cells),
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                }
            )
    return components


def _component_has_hole(grid: np.ndarray, component: dict, bg: int = BG) -> bool:
    r0, c0, r1, c1 = component["bbox"]
    patch = grid[r0 : r1 + 1, c0 : c1 + 1]
    obj = np.zeros_like(patch, dtype=bool)
    coords = component["coords"]
    obj[coords[:, 0] - r0, coords[:, 1] - c0] = True
    bg_mask = patch == bg
    if not np.any(bg_mask):
        return False

    h, w = patch.shape
    seen = np.zeros_like(bg_mask, dtype=bool)
    queue = deque()
    for rr in range(h):
        for cc in (0, w - 1):
            if bg_mask[rr, cc] and not seen[rr, cc]:
                seen[rr, cc] = True
                queue.append((rr, cc))
    for cc in range(w):
        for rr in (0, h - 1):
            if bg_mask[rr, cc] and not seen[rr, cc]:
                seen[rr, cc] = True
                queue.append((rr, cc))

    while queue:
        cr, cc = queue.popleft()
        for nr, nc in _neighbors4(cr, cc):
            if 0 <= nr < h and 0 <= nc < w and bg_mask[nr, nc] and not seen[nr, nc]:
                seen[nr, nc] = True
                queue.append((nr, nc))
    return bool(np.any(bg_mask & ~seen))


def _mask_has_hole(component: dict) -> bool:
    coords = component["coords"]
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    bg_mask = ~mask
    h, w = bg_mask.shape
    seen = np.zeros_like(bg_mask, dtype=bool)
    queue = deque()
    for rr in range(h):
        for cc in (0, w - 1):
            if bg_mask[rr, cc] and not seen[rr, cc]:
                seen[rr, cc] = True
                queue.append((rr, cc))
    for cc in range(w):
        for rr in (0, h - 1):
            if bg_mask[rr, cc] and not seen[rr, cc]:
                seen[rr, cc] = True
                queue.append((rr, cc))
    while queue:
        cr, cc = queue.popleft()
        for nr, nc in _neighbors4(cr, cc):
            if 0 <= nr < h and 0 <= nc < w and bg_mask[nr, nc] and not seen[nr, nc]:
                seen[nr, nc] = True
                queue.append((nr, nc))
    return bool(np.any(bg_mask & ~seen))


def _resolve_noise(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    objects = _color_components(g, ignored={BG, NOISE})
    for r, c in np.argwhere(g == NOISE):
        chosen = None
        for component in objects:
            touches = sum((nr, nc) in component["cells"] for nr, nc in _neighbors4(int(r), int(c)))
            if touches >= 2:
                chosen = component
                break
        out[r, c] = chosen["color"] if chosen is not None else BG
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Merge supported orange noise into nearby objects, erase unsupported noise, then recolor only the objects whose holes survive.

    Input structure:
    - Black background contains several separated monochrome foreground objects in blue/red/green/yellow.
    - Each foreground object is 4-connected and its same-colored mask surrounds exactly one small enclosed cavity.
    - Orange singleton noise pixels appear either in a cavity cell, in a mergeable exterior nook, or isolated in open background.

    Rule:
    - For each orange pixel, count its 4-neighbor contacts to each foreground object from the original input.
    - If one object touches that orange pixel in at least two 4-neighbor directions, recolor the noise to that object's color; otherwise erase it to black.
    - After all orange pixels are processed, re-check each resulting foreground object for enclosed black holes.
    - Recolor every object that still encloses at least one black hole to teal, and leave hole-filled objects in their original color.

    Key insight:
    - The orange pixels are topological switches: merging them can plug a hole, and only the post-merge topology decides whether an object is recolored.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    merged = _resolve_noise(g)
    out = merged.copy()
    for component in _color_components(merged, ignored={BG}):
        if _component_has_hole(merged, component, bg=BG):
            rr, cc = component["coords"].T
            out[rr, cc] = TEAL
    return out


def _plain_ring(hole_len: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    width = hole_len + 2
    mask = np.zeros((3, width), dtype=bool)
    mask[0, :] = True
    mask[2, :] = True
    mask[1, 0] = True
    mask[1, width - 1] = True
    hole = np.zeros_like(mask)
    hole[1, 1 : 1 + hole_len] = True
    slot = np.zeros_like(mask)
    return mask, hole, slot


def _tailed_ring(hole_len: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    width = hole_len + 3
    mask = np.zeros((3, width), dtype=bool)
    mask[0, :] = True
    mask[1, 0] = True
    mask[1, hole_len + 1] = True
    mask[1, hole_len + 2] = True
    mask[2, : hole_len + 2] = True
    hole = np.zeros_like(mask)
    hole[1, 1 : 1 + hole_len] = True
    slot = np.zeros_like(mask)
    slot[2, hole_len + 2] = True
    return mask, hole, slot


def _transform_masks(mask: np.ndarray, hole: np.ndarray, slot: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if random.random() < 0.5:
        mask = np.fliplr(mask)
        hole = np.fliplr(hole)
        slot = np.fliplr(slot)
    rot = random.randrange(4)
    if rot:
        mask = np.rot90(mask, rot)
        hole = np.rot90(hole, rot)
        slot = np.rot90(slot, rot)
    return mask, hole, slot


def _shape_for_branch(branch: str) -> tuple[np.ndarray, list[tuple[int, int]]]:
    if branch == "fill":
        builder = random.choice([_plain_ring, _tailed_ring])
        mask, hole, slot = builder(1)
        mask, hole, slot = _transform_masks(mask, hole, slot)
        noise_cells = [tuple(x) for x in np.argwhere(hole)]
    else:
        mask, hole, slot = _tailed_ring(random.randint(1, 3))
        mask, hole, slot = _transform_masks(mask, hole, slot)
        noise_cells = [tuple(x) for x in np.argwhere(slot)]
    return mask, noise_cells


def _quadrants(h: int, w: int) -> list[tuple[int, int, int, int]]:
    mid_r = h // 2
    mid_c = w // 2
    return [
        (1, mid_r - 2, 1, mid_c - 2),
        (1, mid_r - 2, mid_c + 1, w - 2),
        (mid_r + 1, h - 2, 1, mid_c - 2),
        (mid_r + 1, h - 2, mid_c + 1, w - 2),
    ]


def _place_shape(
    grid: np.ndarray,
    mask: np.ndarray,
    color: int,
    noise_cells: list[tuple[int, int]],
    box: tuple[int, int, int, int],
    object_infos: list[dict],
) -> bool:
    r0, r1, c0, c1 = box
    mh, mw = mask.shape
    if r1 - r0 + 1 < mh or c1 - c0 + 1 < mw:
        return False

    anchors = [
        (r, c)
        for r in range(r0, r1 - mh + 2)
        for c in range(c0, c1 - mw + 2)
    ]
    random.shuffle(anchors)
    for ar, ac in anchors:
        coords = [(ar + r, ac + c) for r, c in np.argwhere(mask)]
        if any(grid[r, c] != BG for r, c in coords):
            continue
        for r, c in coords:
            grid[r, c] = color
        orange = []
        ok = True
        for nr, nc in noise_cells:
            rr, cc = ar + nr, ac + nc
            if not (0 <= rr < grid.shape[0] and 0 <= cc < grid.shape[1]) or grid[rr, cc] != BG:
                ok = False
                break
            orange.append((rr, cc))
        if not ok:
            for r, c in coords:
                grid[r, c] = BG
            continue
        for rr, cc in orange:
            grid[rr, cc] = NOISE
        object_infos.append({"color": color, "cells": set(coords), "noise": set(orange)})
        return True
    return False


def _add_decoy_noise(grid: np.ndarray, object_infos: list[dict], count: int) -> bool:
    h, w = grid.shape
    existing_noise = {tuple(x) for x in np.argwhere(grid == NOISE)}
    candidates = []
    for r in range(h):
        for c in range(w):
            if grid[r, c] != BG:
                continue
            if any((nr, nc) in existing_noise for nr, nc in _neighbors4(r, c)):
                continue
            supported = False
            for info in object_infos:
                touches = sum((nr, nc) in info["cells"] for nr, nc in _neighbors4(r, c))
                if touches >= 2:
                    supported = True
                    break
            if not supported:
                candidates.append((r, c))
    if len(candidates) < count:
        return False
    random.shuffle(candidates)
    placed = []
    for r, c in candidates:
        if any(abs(r - pr) + abs(c - pc) == 1 for pr, pc in placed):
            continue
        grid[r, c] = NOISE
        placed.append((r, c))
        if len(placed) == count:
            return True
    for r, c in placed:
        grid[r, c] = BG
    return False


def _valid_training_signal(inp: np.ndarray) -> bool:
    foreground = _color_components(inp, ignored={BG, NOISE})
    if len(foreground) != 4:
        return False
    if not all(_mask_has_hole(comp) for comp in foreground):
        return False

    merged = _resolve_noise(inp)
    merged_components = _color_components(merged, ignored={BG})
    if len(merged_components) != 4:
        return False
    holes_after = [_component_has_hole(merged, comp, bg=BG) for comp in merged_components]
    if not any(holes_after) or all(holes_after):
        return False

    out = main(inp)
    if np.any(out == NOISE):
        return False
    if not np.any(out == TEAL):
        return False
    if not any(np.any(out == color) for color in OBJECT_COLORS):
        return False
    return bool(np.any(out != inp))


def _attempt_generate() -> np.ndarray | None:
    h = random.randint(16, 20)
    w = random.randint(16, 20)
    grid = np.full((h, w), BG, dtype=int)

    branches = ["fill"] * random.randint(1, 3) + ["keep"] * random.randint(1, 3)
    while len(branches) < 4:
        branches.append(random.choice(["fill", "keep"]))
    random.shuffle(branches)
    branches = branches[:4]
    if "fill" not in branches or "keep" not in branches:
        return None

    colors = list(OBJECT_COLORS)
    random.shuffle(colors)
    regions = _quadrants(h, w)
    random.shuffle(regions)
    object_infos = []

    for color, branch, region in zip(colors, branches, regions):
        mask, noise_cells = _shape_for_branch(branch)
        if not _place_shape(grid, mask, color, noise_cells, region, object_infos):
            return None

    if not _add_decoy_noise(grid, object_infos, random.randint(3, 6)):
        return None
    if not _valid_training_signal(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
