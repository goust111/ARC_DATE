from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
ARROW = 2
WORK_SIZE = 30
GEN_RETRIES = 600

DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIRS8 = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
]
CARDINALS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
PALETTE = [1, 3, 4, 5, 6, 7, 8, 9]


def _bbox(cells: list[tuple[int, int]] | set[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _components_from_mask(mask: np.ndarray, connectivity: int) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    dirs = DIRS8 if connectivity == 8 else DIRS4
    comps: list[list[tuple[int, int]]] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            queue = deque([(r, c)])
            while queue:
                cr, cc = queue.popleft()
                comp.append((cr, cc))
                for dr, dc in dirs:
                    nr = cr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            comps.append(comp)
    return comps


def _monochrome_components(grid: np.ndarray) -> tuple[list[dict[str, object]], dict[tuple[int, int], int]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[dict[str, object]] = []
    cell_to_component: dict[tuple[int, int], int] = {}

    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color in (BG, ARROW) or seen[r, c]:
                continue
            seen[r, c] = True
            queue = deque([(r, c)])
            cells: list[tuple[int, int]] = []
            while queue:
                cr, cc = queue.popleft()
                cells.append((cr, cc))
                for dr, dc in DIRS4:
                    nr = cr + dr
                    nc = cc + dc
                    if (
                        0 <= nr < h
                        and 0 <= nc < w
                        and not seen[nr, nc]
                        and int(grid[nr, nc]) == color
                    ):
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            idx = len(components)
            for cell in cells:
                cell_to_component[cell] = idx
            components.append({"color": color, "cells": cells, "bbox": _bbox(cells)})
    return components, cell_to_component


def _count_holes(cells: list[tuple[int, int]]) -> int:
    r0, c0, r1, c1 = _bbox(cells)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in cells:
        mask[r - r0, c - c0] = True

    h, w = mask.shape
    outside = np.zeros((h, w), dtype=bool)
    queue: deque[tuple[int, int]] = deque()

    def push(rr: int, cc: int) -> None:
        if 0 <= rr < h and 0 <= cc < w and not mask[rr, cc] and not outside[rr, cc]:
            outside[rr, cc] = True
            queue.append((rr, cc))

    for rr in range(h):
        push(rr, 0)
        push(rr, w - 1)
    for cc in range(w):
        push(0, cc)
        push(h - 1, cc)

    while queue:
        rr, cc = queue.popleft()
        for dr, dc in DIRS4:
            nr = rr + dr
            nc = cc + dc
            push(nr, nc)

    holes = 0
    seen = outside.copy()
    for rr in range(h):
        for cc in range(w):
            if mask[rr, cc] or seen[rr, cc]:
                continue
            holes += 1
            seen[rr, cc] = True
            queue = deque([(rr, cc)])
            while queue:
                cr, cc2 = queue.popleft()
                for dr, dc in DIRS4:
                    nr = cr + dr
                    nc = cc2 + dc
                    if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
    return holes


def _find_arrows(
    grid: np.ndarray,
    cell_to_component: dict[tuple[int, int], int],
) -> dict[int, tuple[int, int]]:
    arrows: dict[int, tuple[int, int]] = {}
    for comp in _components_from_mask(grid == ARROW, connectivity=8):
        if len(comp) != 3:
            continue
        candidates: list[tuple[tuple[int, int], list[tuple[int, int]]]] = []
        for cell in comp:
            neighbors: list[tuple[int, int]] = []
            for dr, dc in DIRS4:
                nr = cell[0] + dr
                nc = cell[1] + dc
                if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1]:
                    if int(grid[nr, nc]) not in (BG, ARROW):
                        neighbors.append((nr, nc))
            if neighbors:
                candidates.append((cell, neighbors))
        if len(candidates) != 1:
            continue

        tip, neighbors = candidates[0]
        wings = [cell for cell in comp if cell != tip]
        vec_r = 2 * tip[0] - wings[0][0] - wings[1][0]
        vec_c = 2 * tip[1] - wings[0][1] - wings[1][1]
        if (vec_r == 0) == (vec_c == 0):
            continue

        dr = 0 if vec_r == 0 else (1 if vec_r > 0 else -1)
        dc = 0 if vec_c == 0 else (1 if vec_c > 0 else -1)
        comp_idx = cell_to_component.get(neighbors[0])
        if comp_idx is not None:
            arrows[comp_idx] = (dr, dc)
    return arrows


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stamp each colored block forward in its arrow direction as many times as its hole count.

    Input structure:
    - Black background with several separated monochrome 4-connected blocks.
    - Each block may contain zero or more enclosed black holes inside its own shape.
    - Exactly one red 3-pixel arrowhead sits outside each block, with its tip 4-adjacent to that block.

    Rule:
    - Count the number of enclosed black holes inside each block.
    - Read the block's copy direction from its adjacent red arrowhead.
    - Keep the original block, erase all arrow pixels, and stamp the whole block shape forward that many times.
      The step size is the block bbox width plus 1 for left/right arrows, or the block bbox height plus 1 for up/down arrows.

    Key insight:
    - The arrow only gives direction; the copy count is hidden in the block topology, namely the number of enclosed holes.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    out[out == ARROW] = BG

    components, cell_to_component = _monochrome_components(g)
    arrows = _find_arrows(g, cell_to_component)

    for idx, comp in enumerate(components):
        cells = comp["cells"]
        assert isinstance(cells, list)
        r0, c0, r1, c1 = comp["bbox"]
        color = int(comp["color"])
        copies = _count_holes(cells)
        direction = arrows.get(idx)
        if copies == 0 or direction is None:
            continue

        dr, dc = direction
        step = (c1 - c0 + 2) if dc else (r1 - r0 + 2)
        for k in range(1, copies + 1):
            off_r = dr * step * k
            off_c = dc * step * k
            for r, c in cells:
                rr = r + off_r
                cc = c + off_c
                if 0 <= rr < out.shape[0] and 0 <= cc < out.shape[1]:
                    out[rr, cc] = color

    return out


def _normalize_cells(cells: set[tuple[int, int]]) -> tuple[set[tuple[int, int]], tuple[int, int]]:
    r0, c0, _, _ = _bbox(cells)
    return {(r - r0, c - c0) for r, c in cells}, (r0, c0)


def _pick_holes(base_h: int, base_w: int, hole_count: int) -> list[tuple[int, int]] | None:
    if hole_count == 0:
        return []
    interior = [(r, c) for r in range(1, base_h - 1) for c in range(1, base_w - 1)]
    for _ in range(80):
        random.shuffle(interior)
        holes: list[tuple[int, int]] = []
        for r, c in interior:
            if all(abs(r - rr) + abs(c - cc) > 1 for rr, cc in holes):
                holes.append((r, c))
                if len(holes) == hole_count:
                    return holes
    return None


def _add_protrusion(cells: set[tuple[int, int]], side: str, base_h: int, base_w: int) -> None:
    if side in ("top", "bottom"):
        span = random.randint(1, base_w - 2)
        start = random.randint(1, base_w - span - 1)
        thickness = random.randint(1, 2)
        row_start = -thickness if side == "top" else base_h
        for dc in range(span):
            for dt in range(thickness):
                cells.add((row_start + dt, start + dc))
    else:
        span = random.randint(1, base_h - 2)
        start = random.randint(1, base_h - span - 1)
        thickness = random.randint(1, 2)
        col_start = -thickness if side == "left" else base_w
        for dr in range(span):
            for dt in range(thickness):
                cells.add((start + dr, col_start + dt))


def _build_arrow_cells(arm_end: tuple[int, int], direction: str) -> set[tuple[int, int]]:
    r, c = arm_end
    if direction == "right":
        return {(r, c + 1), (r - 1, c), (r + 1, c)}
    if direction == "left":
        return {(r, c - 1), (r - 1, c), (r + 1, c)}
    if direction == "up":
        return {(r - 1, c), (r, c - 1), (r, c + 1)}
    return {(r + 1, c), (r, c - 1), (r, c + 1)}


def _make_object(direction: str, hole_count: int) -> dict[str, object] | None:
    for _ in range(200):
        base_min = 4 if hole_count <= 1 else 5
        base_h = random.randint(base_min, 6)
        base_w = random.randint(base_min, 6)
        arm_len = random.randint(1, 2)

        max_dir_span = 6 if hole_count >= 3 else 7
        if direction in ("left", "right") and base_w + arm_len > max_dir_span:
            continue
        if direction in ("up", "down") and base_h + arm_len > max_dir_span:
            continue

        holes = _pick_holes(base_h, base_w, hole_count)
        if holes is None:
            continue

        cells = {(r, c) for r in range(base_h) for c in range(base_w)}
        for hole in holes:
            cells.remove(hole)

        extra_sides = [side for side in ("top", "bottom", "left", "right") if side != direction]
        for _extra in range(random.randint(0, 2)):
            _add_protrusion(cells, random.choice(extra_sides), base_h, base_w)

        if direction == "right":
            attach = random.randint(1, base_h - 2)
            for dx in range(arm_len):
                cells.add((attach, base_w + dx))
            arm_end = (attach, base_w + arm_len - 1)
        elif direction == "left":
            attach = random.randint(1, base_h - 2)
            for dx in range(arm_len):
                cells.add((attach, -1 - dx))
            arm_end = (attach, -arm_len)
        elif direction == "up":
            attach = random.randint(1, base_w - 2)
            for dy in range(arm_len):
                cells.add((-1 - dy, attach))
            arm_end = (-arm_len, attach)
        else:
            attach = random.randint(1, base_w - 2)
            for dy in range(arm_len):
                cells.add((base_h + dy, attach))
            arm_end = (base_h + arm_len - 1, attach)

        shape_cells, shape_shift = _normalize_cells(cells)
        arm_end = (arm_end[0] - shape_shift[0], arm_end[1] - shape_shift[1])
        shape_list = sorted(shape_cells)
        if _count_holes(shape_list) != hole_count:
            continue

        arrow_cells = _build_arrow_cells(arm_end, direction)
        if any(cell in shape_cells for cell in arrow_cells):
            continue

        sr0, sc0, sr1, sc1 = _bbox(shape_cells)
        shape_h = sr1 - sr0 + 1
        shape_w = sc1 - sc0 + 1
        dr, dc = CARDINALS[direction]
        step = shape_w + 1 if dc else shape_h + 1

        copy_cells: set[tuple[int, int]] = set()
        for k in range(1, hole_count + 1):
            copy_cells.update((r + dr * step * k, c + dc * step * k) for r, c in shape_cells)

        full_footprint = set(shape_cells) | set(arrow_cells) | copy_cells
        normalized_footprint, foot_shift = _normalize_cells(full_footprint)
        sh_r, sh_c = foot_shift

        shifted_shape = {(r - sh_r, c - sh_c) for r, c in shape_cells}
        shifted_arrows = {(r - sh_r, c - sh_c) for r, c in arrow_cells}
        shifted_copies = {(r - sh_r, c - sh_c) for r, c in copy_cells}
        fr0, fc0, fr1, fc1 = _bbox(normalized_footprint)
        height = fr1 - fr0 + 1
        width = fc1 - fc0 + 1
        if height > WORK_SIZE or width > WORK_SIZE:
            continue

        return {
            "shape": shifted_shape,
            "arrows": shifted_arrows,
            "copies": shifted_copies,
            "footprint": normalized_footprint,
            "height": height,
            "width": width,
        }
    return None


def _inflate(cells: set[tuple[int, int]], radius: int = 1) -> set[tuple[int, int]]:
    inflated: set[tuple[int, int]] = set()
    for r, c in cells:
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                inflated.add((r + dr, c + dc))
    return inflated


def _place_objects(specs: list[dict[str, object]]) -> list[dict[str, object]] | None:
    occupancy = np.zeros((WORK_SIZE, WORK_SIZE), dtype=bool)
    placed: list[dict[str, object]] = []

    for spec in sorted(specs, key=lambda item: len(item["footprint"]), reverse=True):
        height = int(spec["height"])
        width = int(spec["width"])
        footprint = spec["footprint"]
        buffer_cells = _inflate(footprint)
        success = False

        for _ in range(400):
            top = random.randint(0, WORK_SIZE - height)
            left = random.randint(0, WORK_SIZE - width)
            ok = True
            for rr, cc in buffer_cells:
                gr = top + rr
                gc = left + cc
                if 0 <= gr < WORK_SIZE and 0 <= gc < WORK_SIZE and occupancy[gr, gc]:
                    ok = False
                    break
            if not ok:
                continue

            for rr, cc in buffer_cells:
                gr = top + rr
                gc = left + cc
                if 0 <= gr < WORK_SIZE and 0 <= gc < WORK_SIZE:
                    occupancy[gr, gc] = True
            placed.append({"spec": spec, "top": top, "left": left})
            success = True
            break

        if not success:
            return None

    return placed


def _render_scene(
    placed: list[dict[str, object]],
    colors: list[int],
) -> tuple[np.ndarray, np.ndarray, list[tuple[int, int]]]:
    inp = np.zeros((WORK_SIZE, WORK_SIZE), dtype=int)
    out = np.zeros((WORK_SIZE, WORK_SIZE), dtype=int)
    footprint_cells: list[tuple[int, int]] = []

    for placed_item, color in zip(placed, colors, strict=False):
        spec = placed_item["spec"]
        top = int(placed_item["top"])
        left = int(placed_item["left"])

        for r, c in spec["shape"]:
            inp[top + r, left + c] = color
            out[top + r, left + c] = color
        for r, c in spec["arrows"]:
            inp[top + r, left + c] = ARROW
        for r, c in spec["copies"]:
            out[top + r, left + c] = color
        for r, c in spec["footprint"]:
            footprint_cells.append((top + r, left + c))

    return inp, out, footprint_cells


def _crop_scene(
    inp: np.ndarray,
    out: np.ndarray,
    footprint_cells: list[tuple[int, int]],
) -> tuple[np.ndarray, np.ndarray]:
    r0, c0, r1, c1 = _bbox(footprint_cells)
    top_pad = random.randint(0, min(2, r0))
    left_pad = random.randint(0, min(2, c0))
    bottom_pad = random.randint(0, min(2, WORK_SIZE - 1 - r1))
    right_pad = random.randint(0, min(2, WORK_SIZE - 1 - c1))

    rr0 = r0 - top_pad
    cc0 = c0 - left_pad
    rr1 = r1 + bottom_pad
    cc1 = c1 + right_pad
    return inp[rr0 : rr1 + 1, cc0 : cc1 + 1].copy(), out[rr0 : rr1 + 1, cc0 : cc1 + 1].copy()


def _generate_once() -> np.ndarray | None:
    num_objects = random.choice([1, 2, 2, 3])
    directions = [random.choice(tuple(CARDINALS)) for _ in range(num_objects)]
    hole_counts = [random.randint(0, 3) for _ in range(num_objects)]
    colors = random.sample(PALETTE, num_objects)

    specs: list[dict[str, object]] = []
    for direction, hole_count in zip(directions, hole_counts, strict=False):
        spec = _make_object(direction, hole_count)
        if spec is None:
            return None
        specs.append(spec)

    placed = _place_objects(specs)
    if placed is None:
        return None

    inp, expected, footprint_cells = _render_scene(placed, colors)
    cropped_inp, cropped_expected = _crop_scene(inp, expected, footprint_cells)
    actual = main(cropped_inp)
    if not np.array_equal(actual, cropped_expected):
        return None
    if cropped_inp.shape[0] > 30 or cropped_inp.shape[1] > 30:
        return None
    return cropped_inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _generate_once()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid ARC instance")
