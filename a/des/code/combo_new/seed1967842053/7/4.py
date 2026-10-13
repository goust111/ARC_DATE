import random
from collections import defaultdict, deque

import numpy as np


ARROW_SHAPES = {
    "up": ((-1, 0), (0, 0), (1, 0), (-1, -1), (-1, 1)),
    "down": ((-1, 0), (0, 0), (1, 0), (1, -1), (1, 1)),
    "left": ((0, -1), (0, 0), (0, 1), (-1, -1), (1, -1)),
    "right": ((0, -1), (0, 0), (0, 1), (-1, 1), (1, 1)),
}
DIR_ORDER = {name: i for i, name in enumerate(("up", "down", "left", "right"))}


def _background(grid):
    values, counts = np.unique(grid, return_counts=True)
    return values[int(np.argmax(counts))]


def _components(grid, bg):
    rows, cols = grid.shape
    seen = np.zeros((rows, cols), dtype=bool)
    found = []
    for r in range(rows):
        for c in range(cols):
            if seen[r, c] or grid[r, c] == bg:
                continue
            color = grid[r, c]
            q = deque([(r, c)])
            seen[r, c] = True
            coords = []
            while q:
                cr, cc = q.popleft()
                coords.append((cr, cc))
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        if dr == 0 and dc == 0:
                            continue
                        nr, nc = cr + dr, cc + dc
                        if (
                            0 <= nr < rows
                            and 0 <= nc < cols
                            and not seen[nr, nc]
                            and grid[nr, nc] == color
                        ):
                            seen[nr, nc] = True
                            q.append((nr, nc))
            found.append((color, coords))
    return found


def _classify_block(coords):
    if len(coords) not in (4, 9):
        return None
    rs = [r for r, _ in coords]
    cs = [c for _, c in coords]
    top, bottom = min(rs), max(rs)
    left, right = min(cs), max(cs)
    height = bottom - top + 1
    width = right - left + 1
    if height != width or height not in (2, 3) or len(coords) != height * width:
        return None
    if set(coords) != {
        (r, c) for r in range(top, bottom + 1) for c in range(left, right + 1)
    }:
        return None
    return {"top": top, "left": left, "size": height}


def _classify_arrow(coords):
    if len(coords) != 5:
        return None
    coord_set = set(coords)
    for center in coord_set:
        cr, cc = center
        for direction, rels in ARROW_SHAPES.items():
            if {(cr + dr, cc + dc) for dr, dc in rels} == coord_set:
                return {"center": center, "direction": direction, "cells": coord_set}
    return None


def _overlap(a0, a1, b0, b1):
    return max(a0, b0) <= min(a1, b1)


def _slide(blocks, index, direction, rows, cols):
    block = blocks[index]
    top = block["top"]
    left = block["left"]
    size = block["size"]
    bottom = top + size - 1
    right = left + size - 1

    if direction == "up":
        steps = top
        for i, other in enumerate(blocks):
            if i == index:
                continue
            ot, ol, os = other["top"], other["left"], other["size"]
            ob, oright = ot + os - 1, ol + os - 1
            if ob < top and _overlap(left, right, ol, oright):
                steps = min(steps, top - (ot + os))
        block["top"] -= steps
    elif direction == "down":
        steps = rows - (top + size)
        for i, other in enumerate(blocks):
            if i == index:
                continue
            ot, ol, os = other["top"], other["left"], other["size"]
            ob, oright = ot + os - 1, ol + os - 1
            if ot > bottom and _overlap(left, right, ol, oright):
                steps = min(steps, ot - (top + size))
        block["top"] += steps
    elif direction == "left":
        steps = left
        for i, other in enumerate(blocks):
            if i == index:
                continue
            ot, ol, os = other["top"], other["left"], other["size"]
            ob, oright = ot + os - 1, ol + os - 1
            if oright < left and _overlap(top, bottom, ot, ob):
                steps = min(steps, left - (ol + os))
        block["left"] -= steps
    elif direction == "right":
        steps = cols - (left + size)
        for i, other in enumerate(blocks):
            if i == index:
                continue
            ot, ol, os = other["top"], other["left"], other["size"]
            ob, oright = ot + os - 1, ol + os - 1
            if ol > right and _overlap(top, bottom, ot, ob):
                steps = min(steps, ol - (left + size))
        block["left"] += steps


def main(grid: np.ndarray) -> np.ndarray:
    """Move same-colored square blocks according to nearby arrow instructions.

    Input structure:
    A mostly uniform background contains separated solid 2x2 or 3x3 square
    blocks and separated five-pixel arrows. Each arrow has the same color as
    exactly one block; multiple arrows may control the same block.

    Rule:
    Detect every arrow direction from its V-shaped tip. For each arrow, match it
    to the block of the same color, order the instructions by the orthogonal
    distance from the arrow center to that block's original center, and slide
    the block in the arrow direction until its leading edge is next to the grid
    boundary or another current block. Erase all arrows in the output.

    Key insight:
    The arrow shape encodes the direction, while the arrow-to-block distance
    encodes instruction order; applying nearer arrows first changes later slide
    stops.
    """
    bg = _background(grid)
    rows, cols = grid.shape
    blocks = []
    arrows = []

    for color, coords in _components(grid, bg):
        block = _classify_block(coords)
        if block is not None:
            block["color"] = color
            block["id"] = len(blocks)
            blocks.append(block)
            continue
        arrow = _classify_arrow(coords)
        if arrow is not None:
            arrow["color"] = color
            arrows.append(arrow)

    by_color = {}
    for block in blocks:
        by_color[block["color"]] = block

    instructions = []
    for arrow in arrows:
        block = by_color.get(arrow["color"])
        if block is None:
            continue
        direction = arrow["direction"]
        center_r2 = 2 * arrow["center"][0]
        center_c2 = 2 * arrow["center"][1]
        block_r2 = 2 * block["top"] + block["size"] - 1
        block_c2 = 2 * block["left"] + block["size"] - 1
        if direction in ("up", "down"):
            dist2 = abs(center_r2 - block_r2)
        else:
            dist2 = abs(center_c2 - block_c2)
        instructions.append(
            (
                dist2,
                arrow["center"][0],
                arrow["center"][1],
                DIR_ORDER[direction],
                block["id"],
                direction,
            )
        )

    for _, _, _, _, block_id, direction in sorted(instructions):
        _slide(blocks, block_id, direction, rows, cols)

    out = np.full_like(grid, bg)
    for block in blocks:
        top, left, size = block["top"], block["left"], block["size"]
        out[top : top + size, left : left + size] = block["color"]
    return out


def _arrow_cells(center_r, center_c, direction):
    return [(center_r + dr, center_c + dc) for dr, dc in ARROW_SHAPES[direction]]


def _can_place(cells, reserved, rows, cols):
    for r, c in cells:
        if not (0 <= r < rows and 0 <= c < cols) or reserved[r, c]:
            return False
    return True


def _reserve(grid, reserved, cells, color):
    rows, cols = grid.shape
    for r, c in cells:
        grid[r, c] = color
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    reserved[nr, nc] = True


def _square_cells(top, left, size):
    return [(r, c) for r in range(top, top + size) for c in range(left, left + size)]


def _place_block(grid, reserved, color, size, top=None, left=None):
    rows, cols = grid.shape
    candidates = []
    if top is not None and left is not None:
        candidates.append((top, left))
    else:
        for r in range(rows - size + 1):
            for c in range(cols - size + 1):
                candidates.append((r, c))
        random.shuffle(candidates)
    for r, c in candidates:
        cells = _square_cells(r, c, size)
        if _can_place(cells, reserved, rows, cols):
            _reserve(grid, reserved, cells, color)
            return {"top": r, "left": c, "size": size, "color": color}
    return None


def _distance2(block, center, direction):
    br2 = 2 * block["top"] + block["size"] - 1
    bc2 = 2 * block["left"] + block["size"] - 1
    if direction in ("up", "down"):
        return abs(2 * center[0] - br2)
    return abs(2 * center[1] - bc2)


def _place_arrow(grid, reserved, block, direction, used_distances, prefer_near=False):
    rows, cols = grid.shape
    candidates = []
    for r in range(1, rows - 1):
        for c in range(1, cols - 1):
            cells = _arrow_cells(r, c, direction)
            dist2 = _distance2(block, (r, c), direction)
            if dist2 in used_distances:
                continue
            if _can_place(cells, reserved, rows, cols):
                candidates.append((dist2, random.random(), r, c, cells))
    if not candidates:
        return False
    if prefer_near:
        candidates.sort()
        _, _, r, c, cells = candidates[0]
    else:
        _, _, r, c, cells = random.choice(candidates)
    used_distances.add(_distance2(block, (r, c), direction))
    _reserve(grid, reserved, cells, block["color"])
    return True


def _try_generate():
    bg = 0
    n_blocks = random.randint(2, 4)
    rows = random.randint(16 + 2 * n_blocks, 30)
    cols = random.randint(16 + 2 * n_blocks, 30)
    grid = np.full((rows, cols), bg, dtype=int)
    reserved = np.zeros((rows, cols), dtype=bool)
    colors = random.sample(range(1, 10), n_blocks)
    blocks = []
    forced = None

    orientation = random.choice(("horizontal", "vertical"))
    if orientation == "horizontal":
        s0, s1 = random.choice((2, 3)), random.choice((2, 3))
        top = random.randint(1, rows - max(s0, s1) - 2)
        left0 = random.randint(0, max(0, cols // 3 - s0))
        left1 = random.randint(max(left0 + s0 + 3, (2 * cols) // 3), cols - s1)
        b0 = _place_block(grid, reserved, colors[0], s0, top, left0)
        b1 = _place_block(grid, reserved, colors[1], s1, top, left1)
        if b0 is None or b1 is None:
            return None
        blocks.extend([b0, b1])
        forced = (0, "right") if random.random() < 0.5 else (1, "left")
    else:
        s0, s1 = random.choice((2, 3)), random.choice((2, 3))
        left = random.randint(1, cols - max(s0, s1) - 2)
        top0 = random.randint(0, max(0, rows // 3 - s0))
        top1 = random.randint(max(top0 + s0 + 3, (2 * rows) // 3), rows - s1)
        b0 = _place_block(grid, reserved, colors[0], s0, top0, left)
        b1 = _place_block(grid, reserved, colors[1], s1, top1, left)
        if b0 is None or b1 is None:
            return None
        blocks.extend([b0, b1])
        forced = (0, "down") if random.random() < 0.5 else (1, "up")

    for i in range(2, n_blocks):
        size = random.choice((2, 3))
        block = _place_block(grid, reserved, colors[i], size)
        if block is None:
            return None
        blocks.append(block)

    counts = [random.choice((1, 2, 3)) for _ in range(n_blocks)]
    counts[random.randrange(n_blocks)] = random.choice((2, 3))
    directions = ("up", "down", "left", "right")

    for i, block in enumerate(blocks):
        used = set()
        todo = []
        if forced is not None and i == forced[0]:
            todo.append((forced[1], True))
        choices = list(directions)
        random.shuffle(choices)
        while len(todo) < counts[i]:
            direction = choices.pop() if choices else random.choice(directions)
            if forced is not None and i == forced[0] and direction == forced[1]:
                continue
            todo.append((direction, False))
        for direction, prefer_near in todo:
            if not _place_arrow(grid, reserved, block, direction, used, prefer_near):
                return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(200):
        grid = _try_generate()
        if grid is not None:
            return grid

    grid = np.zeros((22, 22), dtype=int)
    reserved = np.zeros_like(grid, dtype=bool)
    b0 = _place_block(grid, reserved, 1, 2, 4, 3)
    b1 = _place_block(grid, reserved, 2, 3, 4, 15)
    b2 = _place_block(grid, reserved, 3, 2, 16, 10)
    for block, dirs in (
        (b0, ("right", "down")),
        (b1, ("left",)),
        (b2, ("up", "right")),
    ):
        used = set()
        for direction in dirs:
            _place_arrow(grid, reserved, block, direction, used)
    return grid
