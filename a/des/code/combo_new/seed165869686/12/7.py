import random
from collections import Counter

import numpy as np


BG = 0
MARKER = 5
PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
GEN_RETRIES = 400
OBJECT_RETRIES = 160
PLACEMENT_RETRIES = 250
WORKSPACE = 30


def _neighbors4(r: int, c: int):
    for dr, dc in DIRS4:
        yield r + dr, c + dc


def _find_components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _crop_to_bbox(grid: np.ndarray) -> np.ndarray:
    coords = np.argwhere(grid != BG)
    if coords.size == 0:
        return grid[:1, :1].copy()
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _layer_data(local_grid: np.ndarray):
    layers = []
    colors = [int(v) for v in np.unique(local_grid) if int(v) != BG]
    for color in colors:
        coords = [(int(r), int(c)) for r, c in np.argwhere(local_grid == color)]
        r0, c0, r1, c1 = _bbox(coords)
        sprite = np.full((r1 - r0 + 1, c1 - c0 + 1), BG, dtype=int)
        for r, c in coords:
            sprite[r - r0, c - c0] = color
        layers.append(
            {
                "color": color,
                "area": len(coords),
                "sprite": sprite,
            }
        )
    return layers


def _marker_side(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> str:
    r0, c0, r1, c1 = bbox
    mid_r = (r0 + r1) // 2
    mid_c = (c0 + c1) // 2
    sides = []
    if r0 - 1 >= 0 and int(grid[r0 - 1, mid_c]) == MARKER:
        sides.append("top")
    if r1 + 1 < grid.shape[0] and int(grid[r1 + 1, mid_c]) == MARKER:
        sides.append("bottom")
    if c0 - 1 >= 0 and int(grid[mid_r, c0 - 1]) == MARKER:
        sides.append("left")
    if c1 + 1 < grid.shape[1] and int(grid[mid_r, c1 + 1]) == MARKER:
        sides.append("right")
    if len(sides) != 1:
        raise AssertionError(f"expected exactly one grey marker for bbox {bbox}, got {sides}")
    return sides[0]


def _arranged_cells_from_local(local_grid: np.ndarray, side: str) -> set[tuple[int, int, int]]:
    local_grid = _crop_to_bbox(np.asarray(local_grid, dtype=int))
    h, w = local_grid.shape
    layers = _layer_data(local_grid)
    areas = [layer["area"] for layer in layers]
    if len(areas) != len(set(areas)):
        raise AssertionError("layer areas must be distinct")
    layers.sort(key=lambda item: -item["area"])

    cells: set[tuple[int, int, int]] = set()
    if side == "top":
        cursor = 0
        for layer in layers:
            sprite = layer["sprite"]
            lh, lw = sprite.shape
            left = (w - lw) // 2
            for rr, cc in np.argwhere(sprite != BG):
                cells.add((cursor + int(rr), left + int(cc), layer["color"]))
            cursor += lh
    elif side == "bottom":
        cursor = h - 1
        for layer in layers:
            sprite = layer["sprite"]
            lh, lw = sprite.shape
            top = cursor - lh + 1
            left = (w - lw) // 2
            for rr, cc in np.argwhere(sprite != BG):
                cells.add((top + int(rr), left + int(cc), layer["color"]))
            cursor -= lh
    elif side == "left":
        cursor = 0
        for layer in layers:
            sprite = layer["sprite"]
            lh, lw = sprite.shape
            top = (h - lh) // 2
            for rr, cc in np.argwhere(sprite != BG):
                cells.add((top + int(rr), cursor + int(cc), layer["color"]))
            cursor += lw
    elif side == "right":
        cursor = w - 1
        for layer in layers:
            sprite = layer["sprite"]
            lh, lw = sprite.shape
            top = (h - lh) // 2
            left = cursor - lw + 1
            for rr, cc in np.argwhere(sprite != BG):
                cells.add((top + int(rr), left + int(cc), layer["color"]))
            cursor -= lw
    else:
        raise AssertionError(f"unknown side {side}")
    return cells


def _object_cells_and_output(local_grid: np.ndarray, side: str):
    local_grid = _crop_to_bbox(np.asarray(local_grid, dtype=int))
    h, w = local_grid.shape
    input_cells = {
        (int(r), int(c), int(local_grid[r, c]))
        for r, c in np.argwhere(local_grid != BG)
    }
    output_cells = _arranged_cells_from_local(local_grid, side)
    mid_r = h // 2
    mid_c = w // 2
    marker_by_side = {
        "top": (-1, mid_c),
        "bottom": (h, mid_c),
        "left": (mid_r, -1),
        "right": (mid_r, w),
    }
    marker = marker_by_side[side]
    return local_grid, input_cells, output_cells, marker


def main(grid: np.ndarray) -> np.ndarray:
    """Disassemble each multicolor object into color layers and restack them from its grey-marked side.

    Input structure:
    - Black is the background. Every foreground object is a 4-connected multicolor component made of 2-4 colors, and each color forms its own 4-connected region inside that object.
    - Exactly one Grey marker sits immediately outside each object, centered on one side of that object's bounding box.

    Rule:
    - For each multicolor object, split it into monochrome layers by color and count the pixels in every layer.
    - Sort the layers by decreasing pixel area. The Grey marker chooses the stacking direction: top marker means top-to-bottom, bottom means bottom-to-top, left means left-to-right, right means right-to-left.
    - Replace the original object with its sorted color layers. Each layer is cropped to its own minimal bounding box, consecutive layers touch with no gap, and all layers are centered on the perpendicular axis while the marked side of the original bounding box stays fixed.

    Key insight:
    - The colors inside one connected object are independent structural layers. The Grey pixel is not something to move toward; it selects which side becomes the starting edge of the new area-sorted stack.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    out[out == MARKER] = BG

    mask = (g != BG) & (g != MARKER)
    components = _find_components(mask)

    for comp in components:
        r0, c0, r1, c1 = _bbox(comp)
        side = _marker_side(g, (r0, c0, r1, c1))
        local = g[r0 : r1 + 1, c0 : c1 + 1].copy()

        for r, c in comp:
            out[r, c] = BG

        for rr, cc, color in _arranged_cells_from_local(local, side):
            ar = r0 + rr
            ac = c0 + cc
            if not (0 <= ar < out.shape[0] and 0 <= ac < out.shape[1]):
                raise AssertionError("rearranged object exceeded the grid")
            out[ar, ac] = color

    return out


def _grow_blob(occupied: np.ndarray, seed: tuple[int, int], area: int, rng: random.Random):
    h, w = occupied.shape
    region = {seed}
    while len(region) < area:
        frontier = set()
        for r, c in region:
            for nr, nc in _neighbors4(r, c):
                if 0 <= nr < h and 0 <= nc < w and not occupied[nr, nc] and (nr, nc) not in region:
                    frontier.add((nr, nc))
        if not frontier:
            return None
        options = list(frontier)
        rng.shuffle(options)
        anchor_r = sum(r for r, _ in region) / len(region)
        anchor_c = sum(c for _, c in region) / len(region)
        options.sort(key=lambda p: (abs(p[0] - anchor_r) + abs(p[1] - anchor_c), rng.random()))
        region.add(options[0])
    return region


def _make_local_object(rng: random.Random, side: str):
    for _ in range(OBJECT_RETRIES):
        k = rng.randint(2, 4)
        colors = rng.sample(PALETTE, k)
        areas = rng.sample(range(3, 9), k)
        canvas = np.zeros((9, 9), dtype=int)
        occupied = np.zeros((9, 9), dtype=bool)
        order = list(range(k))
        rng.shuffle(order)

        ok = True
        for idx in order:
            area = areas[idx]
            color = colors[idx]
            if not occupied.any():
                candidates = [(r, c) for r in range(2, 7) for c in range(2, 7)]
            else:
                candidates = []
                for r in range(1, 8):
                    for c in range(1, 8):
                        if occupied[r, c]:
                            continue
                        if any(0 <= nr < 9 and 0 <= nc < 9 and occupied[nr, nc] for nr, nc in _neighbors4(r, c)):
                            candidates.append((r, c))
            rng.shuffle(candidates)
            placed = None
            for seed in candidates[:30]:
                region = _grow_blob(occupied, seed, area, rng)
                if region is not None:
                    placed = region
                    break
            if placed is None:
                ok = False
                break
            for r, c in placed:
                canvas[r, c] = color
                occupied[r, c] = True
        if not ok:
            continue

        local = _crop_to_bbox(canvas)
        h, w = local.shape
        if h < 3 or w < 3 or h > 9 or w > 9:
            continue
        if h % 2 == 0 or w % 2 == 0:
            continue

        colors_present = [int(v) for v in np.unique(local) if int(v) != BG]
        if len(colors_present) != k:
            continue
        if len(_find_components(local != BG)) != 1:
            continue

        layers = _layer_data(local)
        if len({layer["area"] for layer in layers}) != len(layers):
            continue
        if not any(layer["area"] < int(np.prod(layer["sprite"].shape)) for layer in layers):
            continue

        _, input_cells, output_cells, marker = _object_cells_and_output(local, side)
        footprint_rows = [r for r, _, _ in input_cells] + [r for r, _, _ in output_cells] + [marker[0]]
        footprint_cols = [c for _, c, _ in input_cells] + [c for _, c, _ in output_cells] + [marker[1]]
        if max(footprint_rows) - min(footprint_rows) + 1 > 13:
            continue
        if max(footprint_cols) - min(footprint_cols) + 1 > 13:
            continue
        if input_cells == output_cells:
            continue

        return local, input_cells, output_cells, marker
    return None


def _place_objects(specs, rng: random.Random):
    board = np.zeros((WORKSPACE, WORKSPACE), dtype=int)
    blocked = np.zeros((WORKSPACE, WORKSPACE), dtype=bool)
    placed = []

    for local, input_cells, output_cells, marker, side in specs:
        reserved = {(r, c) for r, c, _ in input_cells} | {(r, c) for r, c, _ in output_cells} | {marker}
        min_r = min(r for r, _ in reserved)
        max_r = max(r for r, _ in reserved)
        min_c = min(c for _, c in reserved)
        max_c = max(c for _, c in reserved)

        placed_one = False
        for _ in range(PLACEMENT_RETRIES):
            base_r = rng.randint(-min_r, WORKSPACE - 1 - max_r)
            base_c = rng.randint(-min_c, WORKSPACE - 1 - max_c)

            pr0 = max(0, base_r + min_r - 1)
            pr1 = min(WORKSPACE - 1, base_r + max_r + 1)
            pc0 = max(0, base_c + min_c - 1)
            pc1 = min(WORKSPACE - 1, base_c + max_c + 1)
            if blocked[pr0 : pr1 + 1, pc0 : pc1 + 1].any():
                continue

            for r, c, color in input_cells:
                board[base_r + r, base_c + c] = color
            board[base_r + marker[0], base_c + marker[1]] = MARKER
            blocked[pr0 : pr1 + 1, pc0 : pc1 + 1] = True
            placed.append(
                {
                    "base_r": base_r,
                    "base_c": base_c,
                    "input_cells": input_cells,
                    "output_cells": output_cells,
                    "marker": marker,
                    "side": side,
                }
            )
            placed_one = True
            break
        if not placed_one:
            return None

    support = set()
    for item in placed:
        br = item["base_r"]
        bc = item["base_c"]
        for r, c, _ in item["input_cells"]:
            support.add((br + r, bc + c))
        for r, c, _ in item["output_cells"]:
            support.add((br + r, bc + c))
        mr, mc = item["marker"]
        support.add((br + mr, bc + mc))

    min_r = min(r for r, _ in support)
    max_r = max(r for r, _ in support)
    min_c = min(c for _, c in support)
    max_c = max(c for _, c in support)

    top = min_r - rng.randint(0, min(3, min_r))
    left = min_c - rng.randint(0, min(3, min_c))
    bottom = max_r + rng.randint(0, min(3, WORKSPACE - 1 - max_r))
    right = max_c + rng.randint(0, min(3, WORKSPACE - 1 - max_c))
    return board[top : bottom + 1, left : right + 1].copy()


def _gen_attempt(rng: random.Random):
    count = rng.choice([2, 2, 3, 3, 4])
    sides = [rng.choice(["top", "bottom", "left", "right"]) for _ in range(count)]

    specs = []
    side_counts = Counter()
    for side in sides:
        made = _make_local_object(rng, side)
        if made is None:
            return None
        local, input_cells, output_cells, marker = made
        specs.append((local, input_cells, output_cells, marker, side))
        side_counts[side] += 1

    grid = _place_objects(specs, rng)
    if grid is None:
        return None
    if grid.ndim != 2 or max(grid.shape) > 30:
        return None

    fg_mask = (grid != BG) & (grid != MARKER)
    comps = _find_components(fg_mask)
    if len(comps) != count:
        return None
    for comp in comps:
        bbox = _bbox(comp)
        try:
            side = _marker_side(grid, bbox)
        except AssertionError:
            return None
        local = grid[bbox[0] : bbox[2] + 1, bbox[1] : bbox[3] + 1]
        if len([int(v) for v in np.unique(local) if int(v) not in (BG, MARKER)]) < 2:
            return None
        transformed = _arranged_cells_from_local(local, side)
        for rr, cc, _ in transformed:
            ar = bbox[0] + rr
            ac = bbox[1] + cc
            if not (0 <= ar < grid.shape[0] and 0 <= ac < grid.shape[1]):
                return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid stacking-layer ARC grid")
