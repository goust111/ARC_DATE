from __future__ import annotations

import random
from collections import deque

import numpy as np


BLACK = 0
GREY = 5
BLOCK_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)

DIRS = {
    "right": (0, 1),
    "left": (0, -1),
    "down": (1, 0),
    "up": (-1, 0),
}
OPPOSITE = {"right": "left", "left": "right", "down": "up", "up": "down"}
TIP_HEAD_OFFSETS = {
    "right": ((0, -1), (-1, -1), (1, -1)),
    "left": ((0, 1), (-1, 1), (1, 1)),
    "down": ((-1, 0), (-1, -1), (-1, 1)),
    "up": ((1, 0), (1, -1), (1, 1)),
}

CANVAS = 30
MIN_BLOCKS = 4
MAX_BLOCKS = 7
LINE_LENGTHS = (1, 2, 3, 4, 5, 6)
GENERATION_ATTEMPTS = 400
PLACEMENT_ATTEMPTS = 200


def _neighbors4(r: int, c: int, h: int, w: int) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        nr, nc = r + dr, c + dc
        if 0 <= nr < h and 0 <= nc < w:
            out.append((nr, nc))
    return out


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc, h, w):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _block_cells(r: int, c: int) -> set[tuple[int, int]]:
    return {(r, c), (r, c + 1), (r + 1, c), (r + 1, c + 1)}


def _extract_blocks(grid: np.ndarray) -> tuple[list[dict[str, object]], dict[tuple[int, int], int]]:
    h, w = grid.shape
    mask = (grid != BLACK) & (grid != GREY)
    blocks: list[dict[str, object]] = []
    cell_to_block: dict[tuple[int, int], int] = {}
    for comp in _components(mask):
        color = int(grid[comp[0]])
        assert all(int(grid[r, c]) == color for r, c in comp), "mixed-color object"
        rows = sorted({r for r, _ in comp})
        cols = sorted({c for _, c in comp})
        assert len(comp) == 4 and len(rows) == 2 and len(cols) == 2, "expected solid 2x2 block"
        r0, c0 = rows[0], cols[0]
        cells = _block_cells(r0, c0)
        assert set(comp) == cells, "foreground object must be a solid 2x2 block"
        block_id = len(blocks)
        blocks.append({"r": r0, "c": c0, "color": color, "cells": cells})
        for cell in cells:
            assert cell not in cell_to_block, "blocks must be disjoint"
            cell_to_block[cell] = block_id
    return blocks, cell_to_block


def _parse_arrow_component(
    comp: list[tuple[int, int]], cell_to_block: dict[tuple[int, int], int], shape: tuple[int, int]
) -> tuple[int, int]:
    h, w = shape
    comp_set = set(comp)
    candidates: list[tuple[tuple[int, int], str]] = []
    for tip in comp:
        tr, tc = tip
        for name, offsets in TIP_HEAD_OFFSETS.items():
            req = {(tr + dr, tc + dc) for dr, dc in offsets}
            if not all(0 <= r < h and 0 <= c < w for r, c in req):
                continue
            if req <= comp_set:
                dr, dc = DIRS[name]
                front = (tr + dr, tc + dc)
                if front not in comp_set:
                    candidates.append((tip, name))
    assert len(candidates) == 1, "arrow must have exactly one valid tip"
    tip, name = candidates[0]
    dr, dc = DIRS[name]
    line_cells: list[tuple[int, int]] = []
    cur = (tip[0] - dr, tip[1] - dc)
    while cur in comp_set:
        line_cells.append(cur)
        cur = (cur[0] - dr, cur[1] - dc)
    assert line_cells, "arrow must contain a line segment"

    expected = set(line_cells)
    expected.add(tip)
    for off in TIP_HEAD_OFFSETS[name][1:]:
        expected.add((tip[0] + off[0], tip[1] + off[1]))
    assert expected == comp_set, "grey component must be one straight arrow"

    start = line_cells[-1]
    source_cell = (start[0] - dr, start[1] - dc)
    target_cell = (tip[0] + dr, tip[1] + dc)
    assert source_cell in cell_to_block and target_cell in cell_to_block, "arrow endpoints must touch blocks"
    source_id = cell_to_block[source_cell]
    target_id = cell_to_block[target_cell]
    assert source_id != target_id, "arrow must connect two different blocks"

    touched = set()
    for r, c in comp:
        for nr, nc in _neighbors4(r, c, h, w):
            if (nr, nc) in cell_to_block:
                touched.add(cell_to_block[(nr, nc)])
    assert touched == {source_id, target_id}, "arrow must touch only its source and target blocks"
    return source_id, target_id


def main(grid: np.ndarray) -> np.ndarray:
    """
    Copy each arrow target block's color onto the arrow's source block, then erase the arrows.

    Input structure:
    - The background is black.
    - Every non-grey foreground object is a solid 2x2 colored block.
    - Grey pixels form separate arrows made of a one-cell-wide horizontal or vertical line and
      a 3-pixel V head whose tip faces a target block.
    - The line starts adjacent to a source block and the tip ends adjacent to a target block.

    Rule:
    - Parse every 2x2 colored block and every grey arrow.
    - For each arrow, read its direction from the V head; the block beside the line start is
      the source and the block beside the tip is the target.
    - Simultaneously recolor each source block to the original color of its target block.
    - Remove all grey arrow pixels to black and leave unconnected blocks unchanged.

    Key insight:
    - The arrow identifies where to pull a color from, so chain effects must use the input colors
      before any source block is recolored.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    blocks, cell_to_block = _extract_blocks(g)
    original_colors = {idx: int(block["color"]) for idx, block in enumerate(blocks)}

    grey_components = _components(g == GREY)
    source_to_target: dict[int, int] = {}
    for comp in grey_components:
        source_id, target_id = _parse_arrow_component(comp, cell_to_block, g.shape)
        assert source_id not in source_to_target, "each block may have at most one outgoing arrow"
        source_to_target[source_id] = target_id

    out = g.copy()
    out[out == GREY] = BLACK
    for source_id, target_id in source_to_target.items():
        color = original_colors[target_id]
        for r, c in blocks[source_id]["cells"]:
            out[r, c] = color
    return out


def _arrow_cells(source: dict[str, object], target: dict[str, object], direction: str, attach: int) -> set[tuple[int, int]]:
    dr, dc = DIRS[direction]
    sr, sc = int(source["r"]), int(source["c"])
    tr, tc = int(target["r"]), int(target["c"])

    if direction == "right":
        source_edge = (attach, sc + 1)
        target_edge = (attach, tc)
        arm_offsets = [(-1, 0), (1, 0)]
    elif direction == "left":
        source_edge = (attach, sc)
        target_edge = (attach, tc + 1)
        arm_offsets = [(-1, 0), (1, 0)]
    elif direction == "down":
        source_edge = (sr + 1, attach)
        target_edge = (tr, attach)
        arm_offsets = [(0, -1), (0, 1)]
    else:
        source_edge = (sr, attach)
        target_edge = (tr + 1, attach)
        arm_offsets = [(0, -1), (0, 1)]

    line_start = (source_edge[0] + dr, source_edge[1] + dc)
    tip = (target_edge[0] - dr, target_edge[1] - dc)
    back = (tip[0] - dr, tip[1] - dc)

    cells: set[tuple[int, int]] = set()
    cur = line_start
    while True:
        cells.add(cur)
        if cur == back:
            break
        cur = (cur[0] + dr, cur[1] + dc)
    cells.add(tip)
    for ar, ac in arm_offsets:
        cells.add((back[0] + ar, back[1] + ac))
    return cells


def _candidate_positions(
    anchor: dict[str, object], anchor_is_source: bool, direction: str, line_len: int
) -> list[tuple[int, int, int]]:
    r0, c0 = int(anchor["r"]), int(anchor["c"])
    candidates: list[tuple[int, int, int]] = []

    if anchor_is_source:
        if direction == "right":
            for attach in (r0, r0 + 1):
                for nr in (attach, attach - 1):
                    candidates.append((nr, c0 + line_len + 3, attach))
        elif direction == "left":
            for attach in (r0, r0 + 1):
                for nr in (attach, attach - 1):
                    candidates.append((nr, c0 - line_len - 3, attach))
        elif direction == "down":
            for attach in (c0, c0 + 1):
                for nc in (attach, attach - 1):
                    candidates.append((r0 + line_len + 3, nc, attach))
        else:
            for attach in (c0, c0 + 1):
                for nc in (attach, attach - 1):
                    candidates.append((r0 - line_len - 3, nc, attach))
    else:
        if direction == "right":
            for attach in (r0, r0 + 1):
                for nr in (attach, attach - 1):
                    candidates.append((nr, c0 - line_len - 3, attach))
        elif direction == "left":
            for attach in (r0, r0 + 1):
                for nr in (attach, attach - 1):
                    candidates.append((nr, c0 + line_len + 3, attach))
        elif direction == "down":
            for attach in (c0, c0 + 1):
                for nc in (attach, attach - 1):
                    candidates.append((r0 - line_len - 3, nc, attach))
        else:
            for attach in (c0, c0 + 1):
                for nc in (attach, attach - 1):
                    candidates.append((r0 + line_len + 3, nc, attach))
    return candidates


def _in_bounds(cells: set[tuple[int, int]], size: int = CANVAS) -> bool:
    return all(0 <= r < size and 0 <= c < size for r, c in cells)


def _can_place_block(grid: np.ndarray, cells: set[tuple[int, int]]) -> bool:
    for r, c in cells:
        if grid[r, c] != BLACK:
            return False
        for nr, nc in _neighbors4(r, c, *grid.shape):
            if (nr, nc) not in cells and grid[nr, nc] != BLACK:
                return False
    return True


def _touches_block(arrow_cells: set[tuple[int, int]], block_cells: set[tuple[int, int]], shape: tuple[int, int]) -> bool:
    h, w = shape
    for r, c in arrow_cells:
        for nr, nc in _neighbors4(r, c, h, w):
            if (nr, nc) in block_cells:
                return True
    return False


def _can_place_link(
    grid: np.ndarray,
    new_block_cells: set[tuple[int, int]],
    arrow_cells: set[tuple[int, int]],
    source_cells: set[tuple[int, int]],
    target_cells: set[tuple[int, int]],
) -> bool:
    if not _in_bounds(new_block_cells | arrow_cells, grid.shape[0]):
        return False
    if not _can_place_block(grid, new_block_cells):
        return False

    h, w = grid.shape
    for r, c in arrow_cells:
        if grid[r, c] != BLACK or (r, c) in new_block_cells:
            return False
        for nr, nc in _neighbors4(r, c, h, w):
            if (nr, nc) in arrow_cells or (nr, nc) in source_cells or (nr, nc) in target_cells:
                continue
            if grid[nr, nc] != BLACK:
                return False

    return _touches_block(arrow_cells, source_cells, grid.shape) and _touches_block(arrow_cells, target_cells, grid.shape)


def _paint_block(grid: np.ndarray, block: dict[str, object]) -> None:
    r, c, color = int(block["r"]), int(block["c"]), int(block["color"])
    grid[r : r + 2, c : c + 2] = color


def _paint_arrow(grid: np.ndarray, cells: set[tuple[int, int]]) -> None:
    for r, c in cells:
        grid[r, c] = GREY


def _crop(grid: np.ndarray) -> np.ndarray:
    coords = np.argwhere(grid != BLACK)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    top_extra = random.randint(0, min(2, r0))
    left_extra = random.randint(0, min(2, c0))
    bottom_extra = random.randint(0, min(2, grid.shape[0] - 1 - r1))
    right_extra = random.randint(0, min(2, grid.shape[1] - 1 - c1))
    return grid[r0 - top_extra : r1 + bottom_extra + 1, c0 - left_extra : c1 + right_extra + 1]


def _choose_anchor(
    blocks: list[dict[str, object]], mode: str, last_idx: int | None
) -> tuple[int, bool]:
    if mode == "chain":
        assert last_idx is not None
        return last_idx, True
    if mode == "fan_in":
        return 0, False

    candidates: list[tuple[int, bool]] = []
    if last_idx is not None and not blocks[last_idx]["has_outgoing"] and random.random() < 0.5:
        candidates.append((last_idx, True))
    for idx, block in enumerate(blocks):
        if not block["has_outgoing"]:
            candidates.append((idx, True))
        candidates.append((idx, False))
    return random.choice(candidates)


def _gen_attempt() -> np.ndarray | None:
    n_blocks = random.randint(MIN_BLOCKS, MAX_BLOCKS)
    n_connected = random.randint(2, n_blocks)
    colors = random.sample(BLOCK_COLORS, n_blocks)
    mode = random.choice(("chain", "fan_in", "mixed"))

    grid = np.full((CANVAS, CANVAS), BLACK, dtype=int)
    start = {
        "r": random.randint(9, 18),
        "c": random.randint(9, 18),
        "color": colors[0],
        "used_sides": set(),
        "has_outgoing": False,
    }
    if not _can_place_block(grid, _block_cells(int(start["r"]), int(start["c"]))):
        return None
    _paint_block(grid, start)
    blocks: list[dict[str, object]] = [start]
    last_idx: int | None = 0

    for color in colors[1:n_connected]:
        placed = False
        for _ in range(PLACEMENT_ATTEMPTS):
            anchor_idx, anchor_is_source = _choose_anchor(blocks, mode, last_idx)
            anchor = blocks[anchor_idx]
            possible_dirs = list(DIRS)
            random.shuffle(possible_dirs)
            for direction in possible_dirs:
                source_side = direction if anchor_is_source else None
                target_side = OPPOSITE[direction] if not anchor_is_source else None
                if anchor_is_source:
                    if anchor["has_outgoing"] or direction in anchor["used_sides"]:
                        continue
                else:
                    if OPPOSITE[direction] in anchor["used_sides"]:
                        continue

                line_len = random.choice(LINE_LENGTHS)
                candidates = _candidate_positions(anchor, anchor_is_source, direction, line_len)
                random.shuffle(candidates)
                for nr, nc, attach in candidates:
                    new_cells = _block_cells(nr, nc)
                    if not _in_bounds(new_cells):
                        continue
                    new_block = {
                        "r": nr,
                        "c": nc,
                        "color": color,
                        "used_sides": set(),
                        "has_outgoing": False,
                    }
                    if anchor_is_source:
                        source_block = anchor
                        target_block = new_block
                    else:
                        source_block = new_block
                        target_block = anchor
                    arrow_cells = _arrow_cells(source_block, target_block, direction, attach)
                    if not _can_place_link(
                        grid,
                        new_cells,
                        arrow_cells,
                        _block_cells(int(source_block["r"]), int(source_block["c"])),
                        _block_cells(int(target_block["r"]), int(target_block["c"])),
                    ):
                        continue

                    _paint_block(grid, new_block)
                    _paint_arrow(grid, arrow_cells)
                    new_block["used_sides"].add(OPPOSITE[direction] if anchor_is_source else direction)
                    new_block["has_outgoing"] = not anchor_is_source
                    anchor["used_sides"].add(direction if anchor_is_source else OPPOSITE[direction])
                    if anchor_is_source:
                        anchor["has_outgoing"] = True
                    blocks.append(new_block)
                    last_idx = len(blocks) - 1
                    placed = True
                    break
                if placed:
                    break
            if placed:
                break
        if not placed:
            return None

    for color in colors[n_connected:]:
        placed = False
        for _ in range(PLACEMENT_ATTEMPTS):
            nr = random.randint(0, CANVAS - 2)
            nc = random.randint(0, CANVAS - 2)
            cells = _block_cells(nr, nc)
            if _in_bounds(cells) and _can_place_block(grid, cells):
                block = {"r": nr, "c": nc, "color": color, "used_sides": set(), "has_outgoing": False}
                _paint_block(grid, block)
                blocks.append(block)
                placed = True
                break
        if not placed:
            return None

    cropped = _crop(grid)
    try:
        out = main(cropped)
    except AssertionError:
        return None

    if not np.any(cropped == GREY):
        return None
    if np.array_equal(cropped, out):
        return None
    return cropped


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid arrow-transfer input")
