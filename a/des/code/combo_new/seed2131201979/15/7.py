import random
from collections import defaultdict

import numpy as np


BG = 0
GEN_CONTAINER_COLOR = 1
GEN_BLOCK_COLORS = [2, 3, 4, 5, 6, 7, 8, 9]
ORIENTATIONS = ("up", "down", "left", "right")
GENERATE_ATTEMPTS = 400
PAIR_PLACE_ATTEMPTS = 250
WORK_SHAPE = (30, 30)


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    next_id = 0

    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) == bg:
                continue
            color = int(grid[r, c])
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((int(cr), int(cc)))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))

            rows = [r0 for r0, _ in coords]
            cols = [c0 for _, c0 in coords]
            comps.append(
                {
                    "id": next_id,
                    "color": color,
                    "coords": coords,
                    "bbox": (min(rows), min(cols), max(rows), max(cols)),
                }
            )
            next_id += 1
    return comps


def _match_container(coords: list[tuple[int, int]]) -> dict | None:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    r0, c0, r1, c1 = min(rows), min(cols), max(rows), max(cols)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    mask = np.zeros((h, w), dtype=bool)
    for r, c in coords:
        mask[r - r0, c - c0] = True

    if 3 <= h <= 5 and 4 <= w <= 6:
        expected = np.zeros((h, w), dtype=bool)
        expected[:, 0] = True
        expected[:, -1] = True
        expected[-1, :] = True
        if np.array_equal(mask, expected):
            return {
                "orientation": "up",
                "bbox": (r0, c0, r1, c1),
                "interior": (r0, c0 + 1, r1 - 1, c1 - 1),
            }

        expected = np.zeros((h, w), dtype=bool)
        expected[:, 0] = True
        expected[:, -1] = True
        expected[0, :] = True
        if np.array_equal(mask, expected):
            return {
                "orientation": "down",
                "bbox": (r0, c0, r1, c1),
                "interior": (r0 + 1, c0 + 1, r1, c1 - 1),
            }

    if 4 <= h <= 6 and 3 <= w <= 5:
        expected = np.zeros((h, w), dtype=bool)
        expected[0, :] = True
        expected[-1, :] = True
        expected[:, -1] = True
        if np.array_equal(mask, expected):
            return {
                "orientation": "left",
                "bbox": (r0, c0, r1, c1),
                "interior": (r0 + 1, c0, r1 - 1, c1 - 1),
            }

        expected = np.zeros((h, w), dtype=bool)
        expected[0, :] = True
        expected[-1, :] = True
        expected[:, 0] = True
        if np.array_equal(mask, expected):
            return {
                "orientation": "right",
                "bbox": (r0, c0, r1, c1),
                "interior": (r0 + 1, c0 + 1, r1 - 1, c1),
            }

    return None


def _extract_scene(grid: np.ndarray) -> tuple[int, list[dict], list[dict]]:
    bg = _most_common_color(grid)
    comps = _components(grid, bg)
    by_color: dict[int, list[dict]] = defaultdict(list)
    matched_by_color: dict[int, list[dict]] = defaultdict(list)

    for comp in comps:
        by_color[comp["color"]].append(comp)
        match = _match_container(comp["coords"])
        if match is not None:
            matched_by_color[comp["color"]].append({**comp, **match})

    candidates = [
        color
        for color, color_comps in by_color.items()
        if matched_by_color[color] and len(matched_by_color[color]) == len(color_comps)
    ]
    assert candidates, "expected at least one container color"
    container_color = max(candidates, key=lambda color: len(matched_by_color[color]))

    containers = sorted(matched_by_color[container_color], key=lambda comp: comp["bbox"])
    blocks = [comp for comp in comps if comp["color"] != container_color]
    assert containers and blocks, "expected containers and free blocks"
    return bg, containers, blocks


def _block_lookup(shape: tuple[int, int], blocks: list[dict]) -> np.ndarray:
    lookup = np.full(shape, -1, dtype=int)
    for block_index, block in enumerate(blocks):
        for r, c in block["coords"]:
            lookup[r, c] = block_index
    return lookup


def _scan_first_block(container: dict, lookup: np.ndarray) -> int:
    h, w = lookup.shape
    r0, c0, r1, c1 = container["bbox"]
    orientation = container["orientation"]

    if orientation == "up":
        for r in range(r0 - 1, -1, -1):
            ids = {int(lookup[r, c]) for c in range(c0 + 1, c1) if lookup[r, c] >= 0}
            if ids:
                assert len(ids) == 1, "opening must face a unique nearest block"
                return next(iter(ids))
    elif orientation == "down":
        for r in range(r1 + 1, h):
            ids = {int(lookup[r, c]) for c in range(c0 + 1, c1) if lookup[r, c] >= 0}
            if ids:
                assert len(ids) == 1, "opening must face a unique nearest block"
                return next(iter(ids))
    elif orientation == "left":
        for c in range(c0 - 1, -1, -1):
            ids = {int(lookup[r, c]) for r in range(r0 + 1, r1) if lookup[r, c] >= 0}
            if ids:
                assert len(ids) == 1, "opening must face a unique nearest block"
                return next(iter(ids))
    else:
        assert orientation == "right"
        for c in range(c1 + 1, w):
            ids = {int(lookup[r, c]) for r in range(r0 + 1, r1) if lookup[r, c] >= 0}
            if ids:
                assert len(ids) == 1, "opening must face a unique nearest block"
                return next(iter(ids))

    raise AssertionError("no aligned block found for container")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pull the first aligned free color block into each open blue container and fill the cavity.

    Input structure:
    - The background is black.
    - Several blue container components are present; each is a one-pixel-thick three-sided rectangular trough with a rectangular empty cavity and an opening on exactly one side.
    - Several detached solid monochrome square blocks are scattered elsewhere; they do not touch containers.
    - For every container, exactly one free block lies in the container's opening direction and overlaps the opening span in the same row range or column range.

    Rule:
    - Detect each container's opening direction by identifying which side of its rectangular trough is missing.
    - From the opening, scan outward along that direction and find the first detached block that overlaps the opening span.
    - Erase that block from its original position and fill the entire cavity of the container with the block's color.
    - Keep the blue container walls unchanged.

    Key insight:
    - The missing side of the blue outline is both the directional cue and the matching filter: it tells you where to search and which detached block is eligible to be absorbed.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg, containers, blocks = _extract_scene(g)
    lookup = _block_lookup(g.shape, blocks)
    out = g.copy()
    used_blocks: set[int] = set()

    for container in containers:
        block_index = _scan_first_block(container, lookup)
        assert block_index not in used_blocks, "each free block should feed at most one container"
        used_blocks.add(block_index)
        block = blocks[block_index]

        for r, c in block["coords"]:
            out[r, c] = bg

        ir0, ic0, ir1, ic1 = container["interior"]
        out[ir0 : ir1 + 1, ic0 : ic1 + 1] = int(block["color"])

    return out


def _container_layout(orientation: str, inner_h: int, inner_w: int) -> tuple[int, int, list[tuple[int, int]], list[tuple[int, int]]]:
    if orientation in {"up", "down"}:
        h = inner_h + 1
        w = inner_w + 2
    else:
        h = inner_h + 2
        w = inner_w + 1

    mask = np.zeros((h, w), dtype=bool)
    if orientation == "up":
        mask[:, 0] = True
        mask[:, -1] = True
        mask[-1, :] = True
        interior = [(r, c) for r in range(0, inner_h) for c in range(1, inner_w + 1)]
    elif orientation == "down":
        mask[:, 0] = True
        mask[:, -1] = True
        mask[0, :] = True
        interior = [(r, c) for r in range(1, inner_h + 1) for c in range(1, inner_w + 1)]
    elif orientation == "left":
        mask[0, :] = True
        mask[-1, :] = True
        mask[:, -1] = True
        interior = [(r, c) for r in range(1, inner_h + 1) for c in range(0, inner_w)]
    else:
        assert orientation == "right"
        mask[0, :] = True
        mask[-1, :] = True
        mask[:, 0] = True
        interior = [(r, c) for r in range(1, inner_h + 1) for c in range(1, inner_w + 1)]

    border = [(int(r), int(c)) for r, c in np.argwhere(mask)]
    return h, w, border, interior


def _can_place(reserved: np.ndarray, coords: list[tuple[int, int]]) -> bool:
    h, w = reserved.shape
    for r, c in coords:
        if not (0 <= r < h and 0 <= c < w):
            return False
        if reserved[r, c]:
            return False
    return True


def _reserve(reserved: np.ndarray, coords: list[tuple[int, int]]) -> None:
    h, w = reserved.shape
    for r, c in coords:
        for rr in range(max(0, r - 1), min(h, r + 2)):
            for cc in range(max(0, c - 1), min(w, c + 2)):
                reserved[rr, cc] = True


def _render_pair(grid: np.ndarray, pair: dict) -> None:
    for r, c in pair["container_cells"]:
        grid[r, c] = GEN_CONTAINER_COLOR
    for r, c in pair["block_cells"]:
        grid[r, c] = pair["color"]


def _try_place_pair(rng: random.Random, reserved: np.ndarray, orientation: str, color: int) -> dict | None:
    h, w = reserved.shape
    for _ in range(PAIR_PLACE_ATTEMPTS):
        inner_h = rng.randint(2, 4)
        inner_w = rng.randint(2, 4)
        block_size = rng.randint(2, 3)
        gap = rng.randint(1, 4)
        box_h, box_w, border_rel, interior_rel = _container_layout(orientation, inner_h, inner_w)

        if orientation == "up":
            if block_size + gap + 1 > h - box_h:
                continue
            cr0 = rng.randint(block_size + gap + 1, h - box_h)
            cc0 = rng.randint(0, w - box_w)
            open_c0 = cc0 + 1
            open_c1 = cc0 + inner_w
            bc_lo = max(0, open_c0 - block_size + 1)
            bc_hi = min(w - block_size, open_c1)
            if bc_lo > bc_hi:
                continue
            bc0 = rng.randint(bc_lo, bc_hi)
            br1 = cr0 - gap - 1
            br0 = br1 - block_size + 1
        elif orientation == "down":
            if box_h + gap + block_size > h:
                continue
            cr0 = rng.randint(0, h - (box_h + gap + block_size))
            cc0 = rng.randint(0, w - box_w)
            open_c0 = cc0 + 1
            open_c1 = cc0 + inner_w
            bc_lo = max(0, open_c0 - block_size + 1)
            bc_hi = min(w - block_size, open_c1)
            if bc_lo > bc_hi:
                continue
            bc0 = rng.randint(bc_lo, bc_hi)
            br0 = cr0 + box_h + gap
        elif orientation == "left":
            if block_size + gap + 1 > w - box_w:
                continue
            cr0 = rng.randint(0, h - box_h)
            cc0 = rng.randint(block_size + gap + 1, w - box_w)
            open_r0 = cr0 + 1
            open_r1 = cr0 + inner_h
            br_lo = max(0, open_r0 - block_size + 1)
            br_hi = min(h - block_size, open_r1)
            if br_lo > br_hi:
                continue
            br0 = rng.randint(br_lo, br_hi)
            bc1 = cc0 - gap - 1
            bc0 = bc1 - block_size + 1
        else:
            assert orientation == "right"
            if box_w + gap + block_size > w:
                continue
            cr0 = rng.randint(0, h - box_h)
            cc0 = rng.randint(0, w - (box_w + gap + block_size))
            open_r0 = cr0 + 1
            open_r1 = cr0 + inner_h
            br_lo = max(0, open_r0 - block_size + 1)
            br_hi = min(h - block_size, open_r1)
            if br_lo > br_hi:
                continue
            br0 = rng.randint(br_lo, br_hi)
            bc0 = cc0 + box_w + gap

        container_cells = [(cr0 + rr, cc0 + cc) for rr, cc in border_rel]
        interior_cells = [(cr0 + rr, cc0 + cc) for rr, cc in interior_rel]
        block_cells = [
            (br0 + dr, bc0 + dc)
            for dr in range(block_size)
            for dc in range(block_size)
        ]
        actual_cells = container_cells + block_cells
        if not _can_place(reserved, actual_cells):
            continue

        pair = {
            "orientation": orientation,
            "color": color,
            "container_cells": container_cells,
            "interior_cells": interior_cells,
            "block_cells": block_cells,
        }
        return pair
    return None


def _expected_output(grid: np.ndarray, pairs: list[dict]) -> np.ndarray:
    out = grid.copy()
    for pair in pairs:
        for r, c in pair["block_cells"]:
            out[r, c] = BG
        for r, c in pair["interior_cells"]:
            out[r, c] = pair["color"]
    return out


def _sample_orientations(rng: random.Random, count: int) -> list[str]:
    order = list(ORIENTATIONS)
    rng.shuffle(order)
    return order[:count]


def _crop_bounds(rng: random.Random, occupied: np.ndarray) -> tuple[int, int, int, int]:
    points = np.argwhere(occupied)
    r0 = int(points[:, 0].min())
    c0 = int(points[:, 1].min())
    r1 = int(points[:, 0].max())
    c1 = int(points[:, 1].max())
    h, w = occupied.shape

    top_extra = rng.randint(0, min(3, r0))
    left_extra = rng.randint(0, min(3, c0))
    bottom_extra = rng.randint(0, min(3, h - 1 - r1))
    right_extra = rng.randint(0, min(3, w - 1 - c1))
    return r0 - top_extra, c0 - left_extra, r1 + bottom_extra, c1 + right_extra


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    grid = np.zeros(WORK_SHAPE, dtype=int)
    reserved = np.zeros(WORK_SHAPE, dtype=bool)
    pair_count = rng.choice([2, 3, 4])
    colors = rng.sample(GEN_BLOCK_COLORS, pair_count)
    orientations = _sample_orientations(rng, pair_count)

    pairs: list[dict] = []
    for orientation, color in zip(orientations, colors, strict=False):
        pair = _try_place_pair(rng, reserved, orientation, color)
        if pair is None:
            return None
        _render_pair(grid, pair)
        _reserve(reserved, pair["container_cells"] + pair["block_cells"])
        pairs.append(pair)

    expected = _expected_output(grid, pairs)
    crop = _crop_bounds(rng, grid != BG)
    r0, c0, r1, c1 = crop
    inp = grid[r0 : r1 + 1, c0 : c1 + 1].copy()
    out = expected[r0 : r1 + 1, c0 : c1 + 1].copy()

    if inp.shape[0] > 30 or inp.shape[1] > 30:
        return None
    try:
        solved = main(inp)
    except Exception:
        return None
    if not np.array_equal(solved, out):
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GENERATE_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
