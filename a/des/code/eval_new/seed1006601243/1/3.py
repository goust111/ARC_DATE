import random
from collections import Counter, deque

import numpy as np


BG = 0
WALL = 5
BLOCK_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
ARROW_BAND_H = 7
SEPARATOR_ROW = ARROW_BAND_H
BOARD_TOP = SEPARATOR_ROW + 1
MAX_ARROW_LEN = 6
MAX_GEN_ATTEMPTS = 500

DIRS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}


def _get_bg(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _neighbors(r: int, c: int, conn8: bool) -> list[tuple[int, int]]:
    out = [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]
    if conn8:
        out.extend([(r - 1, c - 1), (r - 1, c + 1), (r + 1, c - 1), (r + 1, c + 1)])
    return out


def _components(mask: np.ndarray, conn8: bool) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
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
                for nr, nc in _neighbors(cr, cc, conn8):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rs = [r for r, _ in coords]
    cs = [c for _, c in coords]
    return min(rs), min(cs), max(rs), max(cs)


def _is_exact_2x2(coords: list[tuple[int, int]]) -> bool:
    if len(coords) != 4:
        return False
    r0, c0, r1, c1 = _bbox(coords)
    return r1 == r0 + 1 and c1 == c0 + 1 and set(coords) == {
        (r0, c0),
        (r0, c1),
        (r1, c0),
        (r1, c1),
    }


def _parse_arrow(coords: list[tuple[int, int]]) -> dict | None:
    if len(coords) < 4:
        return None
    r0, c0, r1, c1 = _bbox(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    local = {(r - r0, c - c0) for r, c in coords}

    if h == 3 and w >= 2:
        mid = {(1, c) for c in range(w)}
        if mid.issubset(local):
            extra = local - mid
            if extra == {(0, w - 2), (2, w - 2)}:
                return {"direction": "right", "length": w - 1, "bbox": (r0, c0, r1, c1), "cells": set(coords)}
            if extra == {(0, 1), (2, 1)}:
                return {"direction": "left", "length": w - 1, "bbox": (r0, c0, r1, c1), "cells": set(coords)}

    if w == 3 and h >= 2:
        mid = {(r, 1) for r in range(h)}
        if mid.issubset(local):
            extra = local - mid
            if extra == {(1, 0), (1, 2)}:
                return {"direction": "up", "length": h - 1, "bbox": (r0, c0, r1, c1), "cells": set(coords)}
            if extra == {(h - 2, 0), (h - 2, 2)}:
                return {"direction": "down", "length": h - 1, "bbox": (r0, c0, r1, c1), "cells": set(coords)}

    return None


def _arrow_local(direction: str, length: int) -> list[tuple[int, int]]:
    if direction == "right":
        return [(1, c) for c in range(length)] + [(1, length), (0, length - 1), (2, length - 1)]
    if direction == "left":
        return [(1, c) for c in range(1, length + 1)] + [(1, 0), (0, 1), (2, 1)]
    if direction == "up":
        return [(r, 1) for r in range(1, length + 1)] + [(0, 1), (1, 0), (1, 2)]
    return [(r, 1) for r in range(length)] + [(length, 1), (length - 1, 0), (length - 1, 2)]


def _shift(coords: list[tuple[int, int]], dr: int, dc: int) -> list[tuple[int, int]]:
    return [(r + dr, c + dc) for r, c in coords]


def _square_cells(top: int, left: int) -> list[tuple[int, int]]:
    return [(top, left), (top, left + 1), (top + 1, left), (top + 1, left + 1)]


def _can_place_square(
    top: int,
    left: int,
    moving_color: int,
    positions: dict[int, tuple[int, int]],
    wall_cells: set[tuple[int, int]],
    h: int,
    w: int,
) -> bool:
    if top < 0 or left < 0 or top + 1 >= h or left + 1 >= w:
        return False
    cells = set(_square_cells(top, left))
    if cells & wall_cells:
        return False
    for color, (r0, c0) in positions.items():
        if color == moving_color:
            continue
        if cells & set(_square_cells(r0, c0)):
            return False
    return True


def _free_steps(
    color: int,
    direction: str,
    positions: dict[int, tuple[int, int]],
    wall_cells: set[tuple[int, int]],
    h: int,
    w: int,
) -> int:
    dr, dc = DIRS[direction]
    top, left = positions[color]
    steps = 0
    while _can_place_square(top + dr * (steps + 1), left + dc * (steps + 1), color, positions, wall_cells, h, w):
        steps += 1
    return steps


def _simulate_arrows(
    arrows: list[dict],
    block_positions: dict[int, tuple[int, int]],
    wall_cells: set[tuple[int, int]],
    h: int,
    w: int,
) -> dict[int, tuple[int, int]]:
    positions = dict(block_positions)
    for arrow in arrows:
        color = int(arrow["color"])
        direction = str(arrow["direction"])
        length = int(arrow["length"])
        dr, dc = DIRS[direction]
        move = min(_free_steps(color, direction, positions, wall_cells, h, w), length)
        top, left = positions[color]
        positions[color] = (top + dr * move, left + dc * move)
    return positions


def main(grid: np.ndarray) -> np.ndarray:
    """
    Execute the colored arrow instructions by sliding the matching 2x2 blocks, then erase the arrows.

    Input structure:
    - The background is black, with one or more fixed gray wall shapes.
    - Each movable object is a monochrome 2x2 square, and every square color also appears on one or more detached arrow glyphs.
    - Each arrow is a straight shaft plus a 3-pixel triangular head, pointing up, down, left, or right.

    Rule:
    - Read arrows in top-to-bottom, then left-to-right order.
    - For each arrow, find the same-colored 2x2 square and slide it in the arrow direction.
    - The square may move at most as many cells as the arrow shaft length, but it stops earlier if the next step would hit a wall, the grid edge, or another square.
    - After all instructions, remove every arrow and keep only the walls and the final square positions.

    Key insight:
    - The arrow color selects the block, the arrow orientation selects the push direction, and the shaft length caps the slide distance, so the whole task is a sequential physics simulation on 2x2 movers.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = _get_bg(g)

    non_bg_colors = [int(v) for v in np.unique(g) if int(v) != bg]
    square_comps: dict[int, list[list[tuple[int, int]]]] = {color: [] for color in non_bg_colors}
    for color in non_bg_colors:
        for comp in _components(g == color, conn8=False):
            if _is_exact_2x2(comp):
                square_comps[color].append(comp)

    active_colors: set[int] = set()
    block_positions: dict[int, tuple[int, int]] = {}
    arrows: list[dict] = []
    active_square_cells: set[tuple[int, int]] = set()
    arrow_cells: set[tuple[int, int]] = set()

    for color in non_bg_colors:
        if len(square_comps[color]) != 1:
            continue
        mask = g == color
        for r, c in square_comps[color][0]:
            mask[r, c] = False
        parsed: list[dict] = []
        for comp in _components(mask, conn8=True):
            arrow = _parse_arrow(comp)
            if arrow is not None:
                parsed.append(arrow)
        if not parsed:
            continue

        active_colors.add(color)
        r0, c0, _r1, _c1 = _bbox(square_comps[color][0])
        block_positions[color] = (r0, c0)
        active_square_cells.update(square_comps[color][0])
        for arrow in parsed:
            arrow["color"] = color
            arrows.append(arrow)
            arrow_cells.update(arrow["cells"])

    wall_mask = g != bg
    for r, c in active_square_cells | arrow_cells:
        wall_mask[r, c] = False
    wall_cells = {(int(r), int(c)) for r, c in np.argwhere(wall_mask)}

    arrows.sort(key=lambda a: (int(a["bbox"][0]), int(a["bbox"][1])))
    final_positions = _simulate_arrows(arrows, block_positions, wall_cells, h, w)

    out = np.full_like(g, bg)
    out[wall_mask] = g[wall_mask]
    for color, (top, left) in final_positions.items():
        for r, c in _square_cells(top, left):
            out[r, c] = int(color)
    return out


def _normalize(cells: set[tuple[int, int]]) -> list[tuple[int, int]]:
    r0 = min(r for r, _ in cells)
    c0 = min(c for _, c in cells)
    return sorted((r - r0, c - c0) for r, c in cells)


def _random_wall_shape() -> list[tuple[int, int]]:
    if random.random() < 0.45:
        hh = random.randint(1, 3)
        ww = random.randint(2, 6)
        return [(r, c) for r in range(hh) for c in range(ww)]

    area = random.randint(4, 8)
    hh = random.randint(2, 4)
    ww = random.randint(2, 6)
    cells = {(random.randrange(hh), random.randrange(ww))}
    while len(cells) < area:
        frontier = []
        for r, c in cells:
            for nr, nc in _neighbors(r, c, conn8=False):
                if 0 <= nr < hh and 0 <= nc < ww and (nr, nc) not in cells:
                    frontier.append((nr, nc))
        if not frontier:
            return _random_wall_shape()
        cells.add(random.choice(frontier))
    return _normalize(cells)


def _try_place_wall(grid: np.ndarray, shape: list[tuple[int, int]]) -> bool:
    h, w = grid.shape
    sh = max(r for r, _ in shape) + 1
    sw = max(c for _, c in shape) + 1
    rows = list(range(BOARD_TOP, h - sh + 1))
    cols = list(range(0, w - sw + 1))
    if not rows or not cols:
        return False
    tries = [(r, c) for r in rows for c in cols]
    random.shuffle(tries)
    for top, left in tries[:80]:
        cells = [(top + r, left + c) for r, c in shape]
        if all(grid[r, c] == BG for r, c in cells):
            for r, c in cells:
                grid[r, c] = WALL
            return True
    return False


def _block_candidates(grid: np.ndarray, used_blocks: np.ndarray) -> list[tuple[int, int]]:
    h, w = grid.shape
    out: list[tuple[int, int]] = []
    for r in range(BOARD_TOP, h - 1):
        for c in range(0, w - 1):
            if not np.all(grid[r : r + 2, c : c + 2] == BG):
                continue
            r0 = max(0, r - 1)
            r1 = min(h, r + 3)
            c0 = max(0, c - 1)
            c1 = min(w, c + 3)
            if used_blocks[r0:r1, c0:c1].any():
                continue
            out.append((r, c))
    return out


def _obstacle_score(grid: np.ndarray, top: int, left: int) -> tuple[int, int]:
    h, w = grid.shape
    margin = min(top - BOARD_TOP, h - (top + 2), left, w - (left + 2))
    ring = [
        (top - 1, left),
        (top - 1, left + 1),
        (top + 2, left),
        (top + 2, left + 1),
        (top, left - 1),
        (top + 1, left - 1),
        (top, left + 2),
        (top + 1, left + 2),
    ]
    touch = 0
    for r, c in ring:
        if r < BOARD_TOP or r >= h or c < 0 or c >= w or grid[r, c] != BG:
            touch += 1
    return (margin, -touch)


def _place_blocks(grid: np.ndarray, colors: list[int]) -> dict[int, tuple[int, int]] | None:
    used_blocks = np.zeros_like(grid, dtype=bool)
    positions: dict[int, tuple[int, int]] = {}
    for color in colors:
        candidates = _block_candidates(grid, used_blocks)
        if not candidates:
            return None
        if random.random() < 0.6:
            candidates.sort(key=lambda pos: _obstacle_score(grid, pos[0], pos[1]))
            pool = candidates[: max(1, len(candidates) // 3)]
        else:
            pool = candidates
        top, left = random.choice(pool)
        grid[top : top + 2, left : left + 2] = color
        positions[color] = (top, left)
        r0 = max(0, top - 1)
        r1 = min(grid.shape[0], top + 3)
        c0 = max(0, left - 1)
        c1 = min(grid.shape[1], left + 3)
        used_blocks[r0:r1, c0:c1] = True
    return positions


def _plan_instruction_sequence(
    positions: dict[int, tuple[int, int]],
    wall_cells: set[tuple[int, int]],
    h: int,
    w: int,
) -> list[dict] | None:
    colors = list(positions)
    if w < 18:
        total = random.randint(len(colors), min(4, len(colors) + 1))
    elif w < 22:
        total = random.randint(len(colors), min(5, len(colors) + 2))
    else:
        total = random.randint(len(colors) + 1, min(6, len(colors) + 2))

    seq_colors = colors[:]
    while len(seq_colors) < total:
        seq_colors.append(random.choice(colors))
    random.shuffle(seq_colors)

    cur_positions = dict(positions)
    branch_counts = Counter()
    instructions: list[dict] = []
    move_count = 0

    for color in seq_colors:
        free_by_dir = {d: _free_steps(color, d, cur_positions, wall_cells, h, w) for d in DIRS}
        choices = {
            "blocked": [d for d, free in free_by_dir.items() if free == 0],
            "full": [d for d, free in free_by_dir.items() if free > 0],
            "partial": [d for d, free in free_by_dir.items() if 0 < free < MAX_ARROW_LEN],
        }
        available = [name for name, dirs in choices.items() if dirs]
        if not available:
            return None

        best_count = min(branch_counts[name] for name in available)
        branch = random.choice([name for name in available if branch_counts[name] == best_count])
        direction = random.choice(choices[branch])
        free = free_by_dir[direction]
        if branch == "blocked":
            length = random.randint(1, MAX_ARROW_LEN)
        elif branch == "full":
            length = random.randint(1, min(free, MAX_ARROW_LEN))
        else:
            length = random.randint(free + 1, MAX_ARROW_LEN)

        move = min(free, length)
        top, left = cur_positions[color]
        dr, dc = DIRS[direction]
        cur_positions[color] = (top + dr * move, left + dc * move)
        instructions.append(
            {
                "color": color,
                "direction": direction,
                "length": length,
                "branch": branch,
            }
        )
        branch_counts[branch] += 1
        if move > 0:
            move_count += 1

    if move_count == 0:
        return None
    if len(set(inst["branch"] for inst in instructions)) < 2 and len(instructions) >= 3:
        return None
    return instructions


def _place_arrows(grid: np.ndarray, instructions: list[dict]) -> bool:
    band_reserved = np.zeros((ARROW_BAND_H, grid.shape[1]), dtype=bool)
    prev = (-1, -1)
    for inst in instructions:
        local = _arrow_local(str(inst["direction"]), int(inst["length"]))
        sh = max(r for r, _ in local) + 1
        sw = max(c for _, c in local) + 1
        placed = None
        for top in range(0, ARROW_BAND_H - sh + 1):
            for left in range(0, grid.shape[1] - sw + 1):
                if (top, left) <= prev:
                    continue
                cells = _shift(local, top, left)
                r0 = max(0, top - 1)
                r1 = min(ARROW_BAND_H - 1, top + sh)
                c0 = max(0, left - 1)
                c1 = min(grid.shape[1] - 1, left + sw)
                if band_reserved[r0 : r1 + 1, c0 : c1 + 1].any():
                    continue
                if any(grid[r, c] != BG for r, c in cells):
                    continue
                placed = (top, left, cells, r0, r1, c0, c1)
                break
            if placed is not None:
                break
        if placed is None:
            return False
        top, left, cells, r0, r1, c0, c1 = placed
        for r, c in cells:
            grid[r, c] = int(inst["color"])
        band_reserved[r0 : r1 + 1, c0 : c1 + 1] = True
        prev = (top, left)
    return True


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(16, 28)
    w = random.randint(16, 28)
    if h <= BOARD_TOP + 2:
        return None

    grid = np.full((h, w), BG, dtype=int)
    grid[SEPARATOR_ROW, :] = WALL

    extra_walls = random.randint(0, 3)
    for _ in range(extra_walls):
        _try_place_wall(grid, _random_wall_shape())

    block_count = random.randint(2, min(4, len(BLOCK_COLORS)))
    colors = random.sample(BLOCK_COLORS, block_count)
    start_positions = _place_blocks(grid, colors)
    if start_positions is None:
        return None

    wall_cells = {(int(r), int(c)) for r, c in np.argwhere(grid == WALL)}
    instructions = _plan_instruction_sequence(start_positions, wall_cells, h, w)
    if instructions is None:
        return None
    if not _place_arrows(grid, instructions):
        return None

    out = main(grid)
    if out.shape != grid.shape:
        return None
    if not np.all(out[SEPARATOR_ROW, :] == WALL):
        return None
    if not np.array_equal(main(grid), out):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_GEN_ATTEMPTS} attempts")
