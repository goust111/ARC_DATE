from __future__ import annotations

import random
from collections import defaultdict, deque

import numpy as np

BG = 0
GREY = 5
ORANGE = 7
DOOR_COLORS = (1, 2, 3, 4, 6, 8, 9)
DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
GEN_ATTEMPTS = 300


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _neighbors4(shape: tuple[int, int], r: int, c: int):
    for dr, dc in DIR4:
        nr = r + dr
        nc = c + dc
        if _in_bounds(shape, nr, nc):
            yield nr, nc


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []

    for r0, c0 in np.argwhere(mask):
        r = int(r0)
        c = int(c0)
        if seen[r, c]:
            continue
        seen[r, c] = True
        q = deque([(r, c)])
        comp: list[tuple[int, int]] = []
        while q:
            cr, cc = q.popleft()
            comp.append((cr, cc))
            for nr, nc in _neighbors4((h, w), cr, cc):
                if mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    q.append((nr, nc))
        comps.append(comp)
    return comps


def _line_cells(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    if a[0] == b[0]:
        row = a[0]
        lo = min(a[1], b[1])
        hi = max(a[1], b[1])
        return [(row, c) for c in range(lo, hi + 1)]
    if a[1] == b[1]:
        col = a[1]
        lo = min(a[0], b[0])
        hi = max(a[0], b[0])
        return [(r, col) for r in range(lo, hi + 1)]
    raise ValueError("line_cells expects an orthogonal segment")


def _degrees(mask: np.ndarray) -> dict[tuple[int, int], int]:
    deg: dict[tuple[int, int], int] = {}
    for r0, c0 in np.argwhere(mask):
        r = int(r0)
        c = int(c0)
        deg[(r, c)] = sum(1 for nr, nc in _neighbors4(mask.shape, r, c) if mask[nr, nc])
    return deg


def _leaf_paths(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    deg = _degrees(mask)
    leaves: list[list[tuple[int, int]]] = []
    for cell, degree in deg.items():
        if degree != 1:
            continue
        path = [cell]
        prev: tuple[int, int] | None = None
        cur = cell
        while True:
            nxts = [
                (nr, nc)
                for nr, nc in _neighbors4(mask.shape, cur[0], cur[1])
                if mask[nr, nc] and (nr, nc) != prev
            ]
            if not nxts:
                break
            nxt = nxts[0]
            path.append(nxt)
            prev, cur = cur, nxt
            if deg[cur] != 2:
                break
        leaves.append(path)
    return leaves


def _door_candidates(path: list[tuple[int, int]]) -> list[tuple[int, int]]:
    inner = path[1:-1]
    if len(inner) >= 4:
        return inner[1:-1]
    return inner


def _key_shapes(size: int) -> list[list[tuple[int, int]]]:
    if size == 1:
        return [[(0, 0)]]
    if size == 2:
        return [[(0, 0), (0, 1)], [(0, 0), (1, 0)]]
    if size == 3:
        return [
            [(0, 0), (0, 1), (0, 2)],
            [(0, 0), (1, 0), (2, 0)],
            [(0, 0), (0, 1), (1, 0)],
            [(0, 0), (0, 1), (1, 1)],
            [(0, 0), (1, 0), (1, 1)],
            [(0, 1), (1, 0), (1, 1)],
        ]
    raise ValueError("unsupported key size")


def _place_key(grid: np.ndarray, color: int, size: int, rng: random.Random) -> bool:
    shape = rng.choice(_key_shapes(size))
    max_r = max(dr for dr, _ in shape)
    max_c = max(dc for _, dc in shape)

    def placements(margin_only: bool):
        row_start = 1 if margin_only else 0
        col_start = 1 if margin_only else 0
        row_stop = grid.shape[0] - max_r - (1 if margin_only else 0)
        col_stop = grid.shape[1] - max_c - (1 if margin_only else 0)
        if row_stop <= row_start or col_stop <= col_start:
            return []
        return [(r, c) for r in range(row_start, row_stop) for c in range(col_start, col_stop)]

    for margin_only in (True, False):
        starts = placements(margin_only)
        rng.shuffle(starts)
        for r0, c0 in starts:
            cells = {(r0 + dr, c0 + dc) for dr, dc in shape}
            if any(int(grid[r, c]) != BG for r, c in cells):
                continue
            ok = True
            for r, c in cells:
                for nr in range(r - 1, r + 2):
                    for nc in range(c - 1, c + 2):
                        if not _in_bounds(grid.shape, nr, nc) or (nr, nc) in cells:
                            continue
                        if int(grid[nr, nc]) != BG:
                            ok = False
                            break
                    if not ok:
                        break
                if not ok:
                    break
            if not ok:
                continue
            for r, c in cells:
                grid[r, c] = color
            return True
    return False


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use color-matched key blobs to unlock matching door pixels, then flood reachable pipes from orange sources.

    Input structure:
    - The background is black.
    - Grey one-pixel-wide pipe cells form orthogonal paths with T or cross junctions.
    - Some pipe positions are replaced by colored door pixels, while separate small colored blobs act as keys.
    - Orange source pixels sit next to pipe endpoints and are not part of the grey pipe itself.

    Rule:
    - Group each key blob with all door pixels of the same color.
    - A color unlocks iff its key-blob area is at least the total number of same-colored door pixels; unlocked doors turn grey.
    - Starting from every orange source, flood through 4-connected grey pipe cells, including newly unlocked door cells, and recolor all reachable pipe cells orange.
    - Keys stay unchanged, source pixels stay orange, and still-locked door pixels keep blocking the pipe.

    Key insight:
    The crucial hidden rule is a count comparison: same-color matching is necessary but insufficient, because the key must have enough pixels to pay for all doors of that color before the pipe can connect and be flood-filled.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    key_sizes: dict[int, int] = defaultdict(int)
    door_cells: dict[int, list[tuple[int, int]]] = defaultdict(list)

    colors = [int(c) for c in np.unique(g) if int(c) not in (BG, GREY, ORANGE)]
    for color in colors:
        for comp in _components(g == color):
            touches_pipe = any(
                any(int(g[nr, nc]) == GREY for nr, nc in _neighbors4(g.shape, r, c))
                for r, c in comp
            )
            if touches_pipe:
                door_cells[color].extend(comp)
            else:
                key_sizes[color] += len(comp)

    open_pipe = g == GREY
    for color, cells in door_cells.items():
        if int(key_sizes.get(color, 0)) >= len(cells):
            for r, c in cells:
                open_pipe[r, c] = True
                out[r, c] = GREY

    visited = np.zeros_like(open_pipe, dtype=bool)
    q: deque[tuple[int, int]] = deque()
    for r0, c0 in np.argwhere(g == ORANGE):
        r = int(r0)
        c = int(c0)
        for nr, nc in _neighbors4(g.shape, r, c):
            if open_pipe[nr, nc] and not visited[nr, nc]:
                visited[nr, nc] = True
                q.append((nr, nc))

    while q:
        r, c = q.popleft()
        out[r, c] = ORANGE
        for nr, nc in _neighbors4(g.shape, r, c):
            if open_pipe[nr, nc] and not visited[nr, nc]:
                visited[nr, nc] = True
                q.append((nr, nc))

    return out


def _pick_endpoint(
    rng: random.Random, hub: tuple[int, int], direction: str, height: int, width: int
) -> tuple[int, int] | None:
    r, c = hub
    if direction == "up":
        choices = [rr for rr in (1, 2, 3) if r - rr >= 5]
        return (rng.choice(choices), c) if choices else None
    if direction == "down":
        choices = [rr for rr in (height - 2, height - 3, height - 4) if rr - r >= 5]
        return (rng.choice(choices), c) if choices else None
    if direction == "left":
        choices = [cc for cc in (1, 2, 3) if c - cc >= 5]
        return (r, rng.choice(choices)) if choices else None
    if direction == "right":
        choices = [cc for cc in (width - 2, width - 3, width - 4) if cc - c >= 5]
        return (r, rng.choice(choices)) if choices else None
    raise ValueError("bad direction")


def _build_network(rng: random.Random, height: int, width: int):
    hub = (rng.randint(6, height - 7), rng.randint(6, width - 7))
    directions = rng.sample(["up", "down", "left", "right"], rng.choice([3, 4]))

    mask = np.zeros((height, width), dtype=bool)
    arm_cells: list[dict] = []
    for direction in directions:
        endpoint = _pick_endpoint(rng, hub, direction, height, width)
        if endpoint is None:
            return None
        cells = _line_cells(hub, endpoint)
        for r, c in cells:
            mask[r, c] = True
        arm_cells.append({"direction": direction, "cells": cells, "endpoint": endpoint})

    if arm_cells and rng.random() < 0.55:
        long_arms = [arm for arm in arm_cells if len(arm["cells"]) >= 9]
        if long_arms:
            arm = rng.choice(long_arms)
            cells = arm["cells"]
            split_idx = rng.randint(3, len(cells) - 6)
            split = cells[split_idx]
            if arm["direction"] in ("up", "down"):
                sides = ["left", "right"]
            else:
                sides = ["up", "down"]
            rng.shuffle(sides)
            for side in sides:
                endpoint = _pick_endpoint(rng, split, side, height, width)
                if endpoint is None:
                    continue
                spur = _line_cells(split, endpoint)
                for r, c in spur:
                    mask[r, c] = True
                break

    leaves = _leaf_paths(mask)
    if len(leaves) < 3:
        return None
    if any(len(_door_candidates(path)) == 0 for path in leaves):
        return None
    return mask, leaves


def _door_plan(rng: random.Random, leaf_count: int) -> list[tuple[str, int]]:
    closed_count = 3 if leaf_count >= 5 and rng.random() < 0.35 else 2
    if closed_count >= leaf_count:
        return []

    plan: list[tuple[str, int]] = [("closed", closed_count)]
    remaining = leaf_count - closed_count

    if remaining >= 2 and leaf_count >= 4 and rng.random() < 0.5:
        open_counts = [1, 1]
        if remaining >= 3 and rng.random() < 0.35:
            open_counts[0] = 2
        if sum(open_counts) <= remaining:
            plan.extend(("open", count) for count in open_counts)
            return plan

    open_count = 2 if remaining >= 2 and rng.random() < 0.4 else 1
    plan.append(("open", open_count))
    return plan


def _assemble_grid(rng: random.Random, height: int, width: int) -> np.ndarray | None:
    built = _build_network(rng, height, width)
    if built is None:
        return None
    pipe_mask, leaves = built

    plan = _door_plan(rng, len(leaves))
    if not plan or sum(count for _, count in plan) > len(leaves):
        return None

    colors = rng.sample(DOOR_COLORS, len(plan))
    leaf_indices = list(range(len(leaves)))
    rng.shuffle(leaf_indices)

    leaf_state = {idx: "none" for idx in leaf_indices}
    door_cells_by_color: dict[int, list[tuple[int, int]]] = {}
    plan_meta: list[tuple[int, str, int]] = []

    cursor = 0
    for color, (state, count) in zip(colors, plan):
        chosen = leaf_indices[cursor : cursor + count]
        cursor += count
        if len(chosen) != count:
            return None
        door_cells_by_color[color] = []
        plan_meta.append((color, state, count))
        for idx in chosen:
            leaf_state[idx] = state
            cell = rng.choice(_door_candidates(leaves[idx]))
            door_cells_by_color[color].append(cell)

    accessible = [idx for idx in leaf_indices if leaf_state[idx] != "closed"]
    quiet = [idx for idx in accessible if leaf_state[idx] == "none"]
    if not accessible:
        return None

    source_leaves: list[int] = [rng.choice(quiet if quiet else accessible)]
    extras = [idx for idx in accessible if idx not in source_leaves]
    if extras:
        extra_cap = min(2, len(extras))
        extra_count = rng.randint(0, extra_cap)
        source_leaves.extend(rng.sample(extras, extra_count))

    grid = np.full((height, width), BG, dtype=int)
    grid[pipe_mask] = GREY
    for color, cells in door_cells_by_color.items():
        for r, c in cells:
            grid[r, c] = color

    for idx in source_leaves:
        endpoint = leaves[idx][0]
        nxt = leaves[idx][1]
        dr = endpoint[0] - nxt[0]
        dc = endpoint[1] - nxt[1]
        sr = endpoint[0] + dr
        sc = endpoint[1] + dc
        if not _in_bounds(grid.shape, sr, sc) or int(grid[sr, sc]) != BG:
            return None
        grid[sr, sc] = ORANGE

    for color, state, door_count in plan_meta:
        if state == "open":
            key_size = rng.randint(door_count, 3)
        else:
            key_size = rng.randint(1, door_count - 1)
        if not _place_key(grid, color, key_size, rng):
            return None

    out = main(grid)
    if not np.any((grid == GREY) & (out == ORANGE)):
        return None
    if np.count_nonzero(out == ORANGE) <= np.count_nonzero(grid == ORANGE):
        return None
    if not np.any((grid == GREY) & (out == GREY)):
        return None
    return grid


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    height = rng.randint(18, 26)
    width = rng.randint(18, 26)
    return _assemble_grid(rng, height, width)


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid pipe-door-key input")
