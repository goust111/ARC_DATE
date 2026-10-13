from __future__ import annotations

from collections import Counter, defaultdict, deque
import random

import numpy as np


BG = 0
BLOCK = 5
RAY = 7

START_OFFSETS = {
    "NW": (-1, -1),
    "NE": (-1, 2),
    "SE": (2, 2),
    "SW": (2, -1),
}

DIAGONAL_DIRS = {
    "NW": (-1, -1),
    "NE": (-1, 1),
    "SE": (1, 1),
    "SW": (1, -1),
}

COUNT_TO_COLOR = {
    0: 1,
    1: 3,
    2: 4,
}

DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

GEN_RETRIES = 600
GEN_SLOT_RANGE = (4, 5)
GEN_BLOCK_COUNT_RANGE = (4, 7)


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _components_of_color(grid: np.ndarray, color: int) -> list[np.ndarray]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != color:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for dr, dc in DIRS4:
                    nr, nc = rr + dr, cc + dc
                    if _in_bounds((h, w), nr, nc) and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            components.append(np.array(cells, dtype=int))
    return components


def _find_blocks(grid: np.ndarray) -> list[dict]:
    blocks: list[dict] = []
    for coords in _components_of_color(grid, BLOCK):
        r0 = int(coords[:, 0].min())
        c0 = int(coords[:, 1].min())
        r1 = int(coords[:, 0].max())
        c1 = int(coords[:, 1].max())
        if coords.shape[0] != 4 or (r1 - r0 + 1, c1 - c0 + 1) != (2, 2):
            raise AssertionError("expected every grey component to be a solid 2x2 block")
        expected = {(r0, c0), (r0, c0 + 1), (r0 + 1, c0), (r0 + 1, c0 + 1)}
        actual = {tuple(map(int, cell)) for cell in coords}
        if actual != expected:
            raise AssertionError("expected every grey component to be a filled 2x2 block")
        blocks.append(
            {
                "top_left": (r0, c0),
                "cells": expected,
            }
        )
    if not blocks:
        raise AssertionError("expected at least one grey 2x2 block")
    return sorted(blocks, key=lambda block: block["top_left"])


def _emit_rays(grid: np.ndarray, blocks: list[dict]) -> tuple[list[dict], set[tuple[int, int]]]:
    h, w = grid.shape
    rays: list[dict] = []
    for block_index, block in enumerate(blocks):
        r0, c0 = block["top_left"]
        for corner in ("NW", "NE", "SE", "SW"):
            sr, sc = START_OFFSETS[corner]
            start = (r0 + sr, c0 + sc)
            if not _in_bounds((h, w), *start):
                raise AssertionError("expected every block corner-adjacent ray start to be in bounds")
            if int(grid[start]) != BG:
                raise AssertionError("ray starts must be black in the input")
            rays.append(
                {
                    "block_index": block_index,
                    "corner": corner,
                    "dir": DIAGONAL_DIRS[corner],
                    "pos": start,
                    "cells": [start],
                    "termination": None,
                }
            )

    orange: set[tuple[int, int]] = set()
    active = set(range(len(rays)))
    while active:
        groups: dict[tuple[int, int], list[int]] = defaultdict(list)
        for ray_id in active:
            groups[rays[ray_id]["pos"]].append(ray_id)

        for cell in groups:
            orange.add(cell)

        candidates: dict[tuple[int, int], list[int]] = defaultdict(list)
        next_active: set[int] = set()

        for cell, ray_ids in groups.items():
            if len(ray_ids) > 1:
                for ray_id in ray_ids:
                    rays[ray_id]["termination"] = "ray_meet"
                continue

            ray_id = ray_ids[0]
            ray = rays[ray_id]
            r, c = cell
            dr, dc = ray["dir"]

            if r == 0 or r == h - 1 or c == 0 or c == w - 1:
                ray["termination"] = "boundary"
                continue

            nr, nc = r + dr, c + dc
            if int(grid[nr, nc]) == BLOCK:
                ray["termination"] = "block"
                continue
            if (nr, nc) in orange:
                ray["termination"] = "ray_path"
                continue
            candidates[(nr, nc)].append(ray_id)

        for cell, ray_ids in candidates.items():
            if len(ray_ids) > 1:
                orange.add(cell)
                for ray_id in ray_ids:
                    rays[ray_id]["pos"] = cell
                    rays[ray_id]["cells"].append(cell)
                    rays[ray_id]["termination"] = "ray_meet"
            else:
                ray_id = ray_ids[0]
                rays[ray_id]["pos"] = cell
                rays[ray_id]["cells"].append(cell)
                next_active.add(ray_id)

        active = next_active

    return rays, orange


def _validate_semantics(grid: np.ndarray, blocks: list[dict], rays: list[dict], orange: set[tuple[int, int]]) -> None:
    h, w = grid.shape
    union = set()
    meet_end_counts = Counter(
        tuple(ray["cells"][-1]) for ray in rays if ray["termination"] == "ray_meet"
    )

    for ray in rays:
        block = blocks[ray["block_index"]]
        r0, c0 = block["top_left"]
        corner = ray["corner"]
        start = (r0 + START_OFFSETS[corner][0], c0 + START_OFFSETS[corner][1])
        if ray["cells"][0] != start:
            raise AssertionError("ray path must begin at the correct corner-adjacent cell")
        dr, dc = ray["dir"]
        for (r1, c1), (r2, c2) in zip(ray["cells"], ray["cells"][1:]):
            if (r2 - r1, c2 - c1) != (dr, dc):
                raise AssertionError("ray cells must follow the declared diagonal direction")
        for r, c in ray["cells"]:
            if not _in_bounds((h, w), r, c):
                raise AssertionError("ray cells must stay in bounds")
            if int(grid[r, c]) != BG:
                raise AssertionError("rays may only occupy original black cells")
            union.add((r, c))

        last_r, last_c = ray["cells"][-1]
        term = ray["termination"]
        if term == "boundary":
            if last_r not in {0, h - 1} and last_c not in {0, w - 1}:
                raise AssertionError("boundary-terminated rays must end on the boundary")
        elif term == "block":
            nr, nc = last_r + dr, last_c + dc
            if not _in_bounds((h, w), nr, nc) or int(grid[nr, nc]) != BLOCK:
                raise AssertionError("block-terminated rays must face a grey block next")
        elif term == "ray_path":
            nr, nc = last_r + dr, last_c + dc
            if not _in_bounds((h, w), nr, nc) or (nr, nc) not in orange:
                raise AssertionError("ray-path termination must face an already orange cell")
        elif term == "ray_meet":
            if meet_end_counts[(last_r, last_c)] < 2:
                raise AssertionError("ray-meet termination must end at a shared orange cell")
        else:
            raise AssertionError("every ray must terminate")

    if union != orange:
        raise AssertionError("orange cells must match the union of all ray paths")


def _solve(grid: np.ndarray) -> tuple[np.ndarray, dict]:
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2 or g.size == 0:
        raise AssertionError("expected a non-empty 2D grid")
    if max(g.shape) > 30:
        raise AssertionError("ARC grids must stay within 30x30")
    if not np.isin(g, [BG, BLOCK]).all():
        raise AssertionError("input must contain only black background and grey blocks")

    blocks = _find_blocks(g)
    rays, orange = _emit_rays(g, blocks)
    _validate_semantics(g, blocks, rays, orange)

    block_counts = [0] * len(blocks)
    for ray in rays:
        if ray["termination"] == "block":
            block_counts[ray["block_index"]] += 1

    out = g.copy()
    for r, c in orange:
        out[r, c] = RAY

    for block, count in zip(blocks, block_counts):
        color = COUNT_TO_COLOR.get(count, 2)
        for r, c in block["cells"]:
            out[r, c] = color

    meta = {
        "blocks": blocks,
        "rays": rays,
        "orange": orange,
        "block_counts": block_counts,
    }
    return out, meta


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each grey 2x2 block by how many of its four diagonal probe rays are stopped by other blocks.

    Input structure:
    - The grid has a black background and several grey 2x2 solid blocks.
    - Every block has one black diagonal neighbor outside each of its four corners, which act as ray starts.
    - No other colors are present in the input.

    Rule:
    - From the four corner-adjacent cells of each block, extend orange rays along the matching diagonals.
    - Process all rays simultaneously: a ray keeps painting black cells until it reaches the boundary, faces a grey block, or would step into an already orange cell; if multiple rays enter the same cell on the same step, that shared cell becomes orange and all of them stop there.
    - Count, for each source block, how many of its four rays stop because the next diagonal cell is a grey block, then recolor the block by that count: 0->blue, 1->green, 2->yellow, 3 or 4->red.

    Key insight:
    - The block color depends on global diagonal line-of-sight interactions after all rays are advanced together, because other rays can intercept paths before a block collision happens.
    """
    return _solve(np.asarray(grid, dtype=int))[0]


def _make_axis_positions(count: int) -> list[int] | None:
    for _ in range(100):
        pos = [random.randint(1, 2)]
        while len(pos) < count:
            pos.append(pos[-1] + random.choice([5, 6]))
        if pos[-1] <= 26:
            return pos
    return None


def _diagonal_neighbors(cell: tuple[int, int], row_slots: int, col_slots: int) -> list[tuple[int, int]]:
    r, c = cell
    out = []
    for dr, dc in DIAGONAL_DIRS.values():
        nr, nc = r + dr, c + dc
        if 0 <= nr < row_slots and 0 <= nc < col_slots:
            out.append((nr, nc))
    return out


def _shares_diagonal(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return abs(a[0] - b[0]) == abs(a[1] - b[1])


def _render_slots(
    occupied: set[tuple[int, int]],
    row_positions: list[int],
    col_positions: list[int],
) -> np.ndarray:
    h = random.randint(row_positions[-1] + 4, min(30, row_positions[-1] + 6))
    w = random.randint(col_positions[-1] + 4, min(30, col_positions[-1] + 6))
    grid = np.zeros((h, w), dtype=int)
    for sr, sc in occupied:
        r0 = row_positions[sr]
        c0 = col_positions[sc]
        grid[r0 : r0 + 2, c0 : c0 + 2] = BLOCK
    return grid


def _pick_hub(row_slots: int, col_slots: int) -> tuple[int, int] | None:
    candidates = []
    for r in range(row_slots):
        for c in range(col_slots):
            if len(_diagonal_neighbors((r, c), row_slots, col_slots)) >= 2:
                candidates.append((r, c))
    return random.choice(candidates) if candidates else None


def _build_slot_pattern() -> tuple[int, int, set[tuple[int, int]]] | None:
    row_slots = random.randint(*GEN_SLOT_RANGE)
    col_slots = random.randint(*GEN_SLOT_RANGE)
    total_slots = row_slots * col_slots
    block_count = random.randint(GEN_BLOCK_COUNT_RANGE[0], min(GEN_BLOCK_COUNT_RANGE[1], total_slots))

    occupied: set[tuple[int, int]] = set()
    hub = _pick_hub(row_slots, col_slots)
    if hub is None:
        return None
    occupied.add(hub)

    hub_neighbors = _diagonal_neighbors(hub, row_slots, col_slots)
    random.shuffle(hub_neighbors)
    wanted = min(len(hub_neighbors), random.choice([2, 2, 3, 3, 4]))
    occupied.update(hub_neighbors[:wanted])

    square_anchors = [(r, c) for r in range(row_slots - 1) for c in range(col_slots - 1)]
    random.shuffle(square_anchors)
    for r, c in square_anchors:
        if random.random() < 0.5:
            pair = {(r, c + 1), (r + 1, c)}
        else:
            pair = {(r, c), (r + 1, c + 1)}
        occupied.update(pair)
        break

    candidates = [(r, c) for r in range(row_slots) for c in range(col_slots) if (r, c) not in occupied]
    random.shuffle(candidates)
    diagonal_free = [cell for cell in candidates if all(not _shares_diagonal(cell, other) for other in occupied)]
    if diagonal_free:
        occupied.add(diagonal_free[0])
        candidates.remove(diagonal_free[0])
    elif candidates:
        occupied.add(candidates.pop(0))
    else:
        return None

    remaining = [cell for cell in candidates if cell not in occupied]
    random.shuffle(remaining)
    while len(occupied) < block_count and remaining:
        occupied.add(remaining.pop())

    if len(occupied) != block_count:
        return None
    return row_slots, col_slots, occupied


def _generator_accepts(grid: np.ndarray) -> bool:
    try:
        out, meta = _solve(grid)
    except Exception:
        return False

    blocks = meta["blocks"]
    rays = meta["rays"]
    counts = meta["block_counts"]
    terminations = Counter(ray["termination"] for ray in rays)

    if not (4 <= len(blocks) <= 7):
        return False
    if 0 not in counts or max(counts) < 2:
        return False
    if len(set(counts)) < 3:
        return False
    if terminations["block"] == 0 or terminations["boundary"] == 0:
        return False
    if terminations["ray_meet"] + terminations["ray_path"] == 0:
        return False
    if max(len(ray["cells"]) for ray in rays) < 2:
        return False

    # Semantic checks beyond solver self-consistency.
    blue = sum(count == 0 for count in counts)
    hot = sum(count >= 2 for count in counts)
    if blue == 0 or hot == 0:
        return False
    if np.array_equal(grid, out):
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        pattern = _build_slot_pattern()
        if pattern is None:
            continue
        row_slots, col_slots, occupied = pattern
        row_positions = _make_axis_positions(row_slots)
        col_positions = _make_axis_positions(col_slots)
        if row_positions is None or col_positions is None:
            continue
        grid = _render_slots(occupied, row_positions, col_positions)
        if _generator_accepts(grid):
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 5")
