from __future__ import annotations

import random

import numpy as np


BG = 0
PALETTE = tuple(range(1, 10))
SIDE_SLOTS = {
    "top": [(-1, 0), (-1, 1)],
    "bottom": [(2, 0), (2, 1)],
    "left": [(0, -1), (1, -1)],
    "right": [(0, 2), (1, 2)],
}
GEN_RETRIES = 400


def _neighbors8(r: int, c: int) -> list[tuple[int, int]]:
    return [
        (r + dr, c + dc)
        for dr in (-1, 0, 1)
        for dc in (-1, 0, 1)
        if not (dr == 0 and dc == 0)
    ]


def _components_by_color(grid: np.ndarray, bg: int = BG) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []

    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) == bg:
                continue

            color = int(grid[r, c])
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []

            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors8(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))

            rows = [pt[0] for pt in coords]
            cols = [pt[1] for pt in coords]
            comps.append(
                {
                    "color": color,
                    "coords": np.array(coords, dtype=int),
                    "bbox": (min(rows), min(cols), max(rows), max(cols)),
                    "area": len(coords),
                }
            )

    return comps


def _is_square_shape(comp: dict, grid: np.ndarray) -> bool:
    if int(comp["area"]) != 4:
        return False
    r0, c0, r1, c1 = map(int, comp["bbox"])
    if (r1 - r0 + 1, c1 - c0 + 1) != (2, 2):
        return False
    return bool(np.all(grid[r0 : r1 + 1, c0 : c1 + 1] == int(comp["color"])))


def _side_of(shape_bbox: tuple[int, int, int, int], pos: tuple[int, int]) -> str | None:
    r0, c0, r1, c1 = map(int, shape_bbox)
    r, c = map(int, pos)
    if r == r0 - 1 and c0 <= c <= c1:
        return "top"
    if r == r1 + 1 and c0 <= c <= c1:
        return "bottom"
    if c == c0 - 1 and r0 <= r <= r1:
        return "left"
    if c == c1 + 1 and r0 <= r <= r1:
        return "right"
    return None


def _select_exit(signs: list[tuple[str, tuple[int, int], int]]) -> tuple[str, tuple[int, int], int]:
    top_signs = [item for item in signs if item[0] == "top"]
    if top_signs:
        return min(top_signs, key=lambda item: (item[1][0], item[1][1]))

    left_signs = [item for item in signs if item[0] == "left"]
    assert left_signs, "each shape must have at least one top/left exit sign"
    return min(left_signs, key=lambda item: (item[1][1], item[1][0]))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode paired single-pixel wormholes and teleport each 2x2 block to its exit endpoint.

    Input structure:
    - Background is black.
    - Foreground contains several solid monochrome 2x2 blocks.
    - There are also same-color singleton pixel pairs; each color appears on exactly two isolated signpost pixels.
    - Every 2x2 block has exactly two adjacent signposts on its outer top/bottom/left/right side cells.

    Rule:
    - Find all monochrome components and separate 2x2 blocks from singleton signposts.
    - Pair singleton signposts by color to build a bidirectional teleport mapping.
    - For each block, the exit sign is the adjacent signpost on the top side if one exists, otherwise the adjacent signpost on the left side.
    - Teleport the block to the other endpoint of that sign's color pair, place its top-left corner on that endpoint, recolor it to the exit sign color, and erase all original blocks and signposts.
    - Process blocks in reading order by their original top-left position; later teleports overwrite earlier ones on overlap.

    Key insight:
    - The puzzle hides a global color-to-endpoint portal map in the singleton pairs, while each block locally reveals which portal to use by putting its actionable sign on the top or left side.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = g.shape

    comps = _components_by_color(g, bg=BG)
    shapes = sorted((comp for comp in comps if _is_square_shape(comp, g)), key=lambda comp: comp["bbox"][:2])
    singletons = [comp for comp in comps if int(comp["area"]) == 1]
    assert shapes, "expected at least one 2x2 shape"

    portal_cells: dict[int, list[tuple[int, int]]] = {}
    for comp in singletons:
        pos = tuple(map(int, comp["coords"][0]))
        portal_cells.setdefault(int(comp["color"]), []).append(pos)

    portal_pairs: dict[int, tuple[tuple[int, int], tuple[int, int]]] = {}
    for color, cells in portal_cells.items():
        assert len(cells) == 2, "each portal color must appear on exactly two singleton pixels"
        a, b = sorted(cells)
        portal_pairs[int(color)] = (a, b)

    out = np.full((h, w), BG, dtype=int)
    singleton_entries = [
        (int(comp["color"]), tuple(map(int, comp["coords"][0])))
        for comp in singletons
    ]

    for shape in shapes:
        bbox = tuple(map(int, shape["bbox"]))
        local_signs: list[tuple[str, tuple[int, int], int]] = []
        for color, pos in singleton_entries:
            side = _side_of(bbox, pos)
            if side is not None:
                local_signs.append((side, pos, color))

        assert len(local_signs) == 2, "each shape must have exactly two adjacent signposts"
        _side, exit_pos, exit_color = _select_exit(local_signs)
        p0, p1 = portal_pairs[int(exit_color)]
        dst = p1 if exit_pos == p0 else p0
        dr, dc = dst
        assert 0 <= dr <= h - 2 and 0 <= dc <= w - 2, "teleport endpoint must fit a 2x2 block"
        out[dr : dr + 2, dc : dc + 2] = int(exit_color)

    return out


def _can_place(cells: set[tuple[int, int]], occupied: set[tuple[int, int]], h: int, w: int) -> bool:
    if len(cells) != len(set(cells)):
        return False

    for r, c in cells:
        if not (0 <= r < h and 0 <= c < w):
            return False

    for r, c in cells:
        if (r, c) in occupied:
            return False
        for nr, nc in _neighbors8(r, c):
            if (nr, nc) in occupied:
                return False
    return True


def _shape_cells(top_left: tuple[int, int]) -> set[tuple[int, int]]:
    r, c = top_left
    return {(r, c), (r, c + 1), (r + 1, c), (r + 1, c + 1)}


def _slot_pos(top_left: tuple[int, int], side: str, slot: int) -> tuple[int, int]:
    r, c = top_left
    dr, dc = SIDE_SLOTS[side][slot]
    return r + dr, c + dc


def _shape_candidate_positions(
    h: int,
    w: int,
    exit_side: str,
    decoy_side: str,
) -> list[dict]:
    candidates: list[dict] = []
    for r in range(h - 1):
        for c in range(w - 1):
            top_left = (r, c)
            shape = _shape_cells(top_left)
            for exit_slot in range(2):
                for decoy_slot in range(2):
                    exit_local = _slot_pos(top_left, exit_side, exit_slot)
                    decoy_local = _slot_pos(top_left, decoy_side, decoy_slot)
                    cells = set(shape)
                    cells.add(exit_local)
                    cells.add(decoy_local)
                    if len(cells) != 6:
                        continue
                    if any(not (0 <= rr < h and 0 <= cc < w) for rr, cc in cells):
                        continue
                    candidates.append(
                        {
                            "shape_top_left": top_left,
                            "shape_cells": shape,
                            "exit_side": exit_side,
                            "decoy_side": decoy_side,
                            "exit_local": exit_local,
                            "decoy_local": decoy_local,
                            "cells": cells,
                        }
                    )
    random.shuffle(candidates)
    return candidates


def _place_shape_spec(
    h: int,
    w: int,
    exit_side: str,
    decoy_side: str,
    occupied: set[tuple[int, int]],
) -> dict | None:
    for candidate in _shape_candidate_positions(h, w, exit_side, decoy_side):
        if _can_place(candidate["cells"], occupied, h, w):
            return candidate
    return None


def _place_endpoint(h: int, w: int, occupied: set[tuple[int, int]]) -> tuple[int, int] | None:
    candidates = [(r, c) for r in range(h - 1) for c in range(w - 1)]
    random.shuffle(candidates)
    for pos in candidates:
        if _can_place({pos}, occupied, h, w):
            return pos
    return None


def _stamp_square(grid: np.ndarray, top_left: tuple[int, int], color: int) -> None:
    r, c = top_left
    grid[r : r + 2, c : c + 2] = int(color)


def _generate_attempt() -> np.ndarray | None:
    n_shapes = random.choice([2, 3])
    low = 11 if n_shapes == 2 else 14
    h = random.randint(low, 24)
    w = random.randint(low, 24)

    colors = random.sample(list(PALETTE), 3 * n_shapes)
    shape_colors = colors[:n_shapes]
    portal_colors = colors[n_shapes:]

    if n_shapes == 2:
        exit_sides = ["top", "left"]
        decoy_sides = ["bottom", "right"]
    else:
        exit_sides = ["top", "left", random.choice(["top", "left"])]
        decoy_sides = ["bottom", "right", random.choice(["bottom", "right"])]
    random.shuffle(exit_sides)
    random.shuffle(decoy_sides)

    occupied: set[tuple[int, int]] = set()
    specs: list[dict] = []
    for exit_side, decoy_side in zip(exit_sides, decoy_sides, strict=True):
        spec = _place_shape_spec(h, w, exit_side, decoy_side, occupied)
        if spec is None:
            return None
        occupied.update(spec["cells"])
        specs.append(spec)

    random.shuffle(shape_colors)
    random.shuffle(portal_colors)
    for i, spec in enumerate(specs):
        spec["shape_color"] = int(shape_colors[i])
        spec["exit_color"] = int(portal_colors[2 * i])
        spec["decoy_color"] = int(portal_colors[2 * i + 1])

    endpoint_jobs: list[tuple[dict, str, int]] = []
    for spec in specs:
        endpoint_jobs.append((spec, "exit_remote", int(spec["exit_color"])))
        endpoint_jobs.append((spec, "decoy_remote", int(spec["decoy_color"])))
    random.shuffle(endpoint_jobs)

    for spec, key, _color in endpoint_jobs:
        pos = _place_endpoint(h, w, occupied)
        if pos is None:
            return None
        spec[key] = pos
        occupied.add(pos)

    inp = np.full((h, w), BG, dtype=int)
    for spec in specs:
        _stamp_square(inp, spec["shape_top_left"], spec["shape_color"])
        inp[spec["exit_local"]] = int(spec["exit_color"])
        inp[spec["decoy_local"]] = int(spec["decoy_color"])
        inp[spec["exit_remote"]] = int(spec["exit_color"])
        inp[spec["decoy_remote"]] = int(spec["decoy_color"])

    expected = np.full((h, w), BG, dtype=int)
    for spec in sorted(specs, key=lambda item: item["shape_top_left"]):
        _stamp_square(expected, spec["exit_remote"], spec["exit_color"])

    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
