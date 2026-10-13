import random
from collections import deque

import numpy as np


AZURE = 8
WHITE = 0
BOARD_COLORS = (AZURE, WHITE)
FG_COLORS = (1, 2, 3, 4, 5, 6, 7, 9)
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _checkerboard(shape: tuple[int, int], even_color: int, odd_color: int) -> np.ndarray:
    rows, cols = np.indices(shape)
    return np.where(((rows + cols) % 2) == 0, even_color, odd_color).astype(int)


def _infer_board_colors(grid: np.ndarray) -> tuple[int, int]:
    g = np.asarray(grid, dtype=int)
    rows, cols = np.indices(g.shape)
    parity = (rows + cols) % 2
    colors = [int(c) for c in np.unique(g)]

    best: tuple[tuple[int, int], int, int] | None = None
    for even_color in colors:
        even_mask = g == even_color
        if np.any(parity[even_mask] != 0):
            continue
        even_count = int(even_mask.sum())
        for odd_color in colors:
            if odd_color == even_color:
                continue
            odd_mask = g == odd_color
            if np.any(parity[odd_mask] != 1):
                continue
            odd_count = int(odd_mask.sum())
            score = (even_count + odd_count, min(even_count, odd_count))
            if score[0] == 0:
                continue
            if best is None or score > best[0]:
                best = (score, even_color, odd_color)

    if best is not None:
        return best[1], best[2]
    return AZURE, WHITE


def _foreground_components(grid: np.ndarray, board_colors: set[int]) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = map(int, g.shape)
    visited = np.zeros((h, w), dtype=bool)
    components: list[dict] = []

    for r0 in range(h):
        for c0 in range(w):
            color = int(g[r0, c0])
            if color in board_colors or visited[r0, c0]:
                continue

            q = deque([(r0, c0)])
            visited[r0, c0] = True
            coords: list[tuple[int, int]] = []

            while q:
                r, c = q.popleft()
                coords.append((r, c))
                for dr, dc in DIR4:
                    nr, nc = r + dr, c + dc
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if visited[nr, nc] or int(g[nr, nc]) != color:
                        continue
                    visited[nr, nc] = True
                    q.append((nr, nc))

            rows = [r for r, _ in coords]
            cols = [c for _, c in coords]
            components.append(
                {
                    "color": color,
                    "coords": coords,
                    "area": len(coords),
                    "top": min(rows),
                    "left": min(cols),
                }
            )

    return components


def main(grid: np.ndarray) -> np.ndarray:
    """
    Restore the azure/white checkerboard, then replace each foreground object by a
    horizontal segment whose length equals the object's area and whose anchor is
    the object's bounding-box top-left coordinate.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = map(int, g.shape)

    even_color, odd_color = _infer_board_colors(g)
    out = _checkerboard((h, w), even_color, odd_color)

    objects = _foreground_components(g, {even_color, odd_color})
    objects.sort(key=lambda obj: (int(obj["top"]), int(obj["left"]), int(obj["color"])))

    for obj in objects:
        row = int(obj["top"])
        col = int(obj["left"])
        length = min(int(obj["area"]), w - col)
        out[row, col : col + length] = int(obj["color"])

    return out


GEN_RETRIES = 300
GRID_H_RANGE = (8, 16)
GRID_W_RANGE = (9, 18)
MIN_AREA = 2
MAX_AREA = 14
TRUNCATE_PROB = 0.45
CORNER_GAP_PROB = 0.35
PLACEMENT_RETRIES = 180
SHAPE_RETRIES = 90


def _sample_object_count(h: int, w: int) -> int:
    options = [1, 2, 3]
    if h * w >= 96:
        options.append(4)
    if h * w >= 150:
        options.append(5)
    return random.choice(options)


def _line_is_clear(
    reserved: dict[int, list[tuple[int, int]]], row: int, start: int, end: int
) -> bool:
    for lo, hi in reserved.get(row, []):
        if not (end < lo - 1 or start > hi + 1):
            return False
    return True


def _mark_buffer(blocked: np.ndarray, cells: list[tuple[int, int]]) -> None:
    h, w = blocked.shape
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w:
                    blocked[nr, nc] = True


def _shape_neighbors(r: int, c: int, box_h: int, box_w: int) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for dr, dc in DIR4:
        nr, nc = r + dr, c + dc
        if 0 <= nr < box_h and 0 <= nc < box_w:
            out.append((nr, nc))
    return out


def _grow_shape(area: int, box_h: int, box_w: int, mode: str) -> list[tuple[int, int]] | None:
    seed = (random.randrange(box_h), random.randrange(box_w))
    cells = {seed}
    frontier = set(_shape_neighbors(seed[0], seed[1], box_h, box_w))

    while len(cells) < area:
        if not frontier:
            return None

        used_rows = {r for r, _ in cells}
        used_cols = {c for _, c in cells}
        candidates = list(frontier)
        weights: list[float] = []

        for r, c in candidates:
            touch = sum((r + dr, c + dc) in cells for dr, dc in DIR4)
            new_axes = int(r not in used_rows) + int(c not in used_cols)
            edge_bias = min(r, c, box_h - 1 - r, box_w - 1 - c)
            diag = r + c

            if mode == "blob":
                weight = 8 * touch + 2 * (touch >= 2) + (3 - min(edge_bias, 2))
            elif mode == "snake":
                weight = 10 * (touch == 1) + 3 * new_axes + diag + 1
            else:
                weight = 6 * new_axes + 3 * touch + diag + 1
            weights.append(float(max(1, weight)))

        choice = random.choices(candidates, weights=weights, k=1)[0]
        cells.add(choice)
        frontier.discard(choice)
        for nbr in _shape_neighbors(choice[0], choice[1], box_h, box_w):
            if nbr not in cells:
                frontier.add(nbr)

    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    norm = sorted((r - min_r, c - min_c) for r, c in cells)
    return norm


def _sample_shape(
    area: int,
    max_h: int,
    max_w: int,
    require_corner_gap: bool,
) -> list[tuple[int, int]] | None:
    if max_h <= 0 or max_w <= 0:
        return None

    for _ in range(SHAPE_RETRIES):
        mode = random.choice(("blob", "snake", "stair"))
        box_choices: list[tuple[int, int, float]] = []
        for box_h in range(1, min(max_h, area) + 1):
            for box_w in range(1, min(max_w, area) + 1):
                if box_h * box_w < area:
                    continue
                if require_corner_gap and (box_h == 1 or box_w == 1):
                    continue

                slack = box_h * box_w - area
                weight = 1.0 / (1 + slack)
                if area >= 4 and box_h > 1 and box_w > 1:
                    weight *= 3.5
                if mode == "blob":
                    weight /= 1 + abs(box_h - box_w)
                elif mode == "snake":
                    weight *= 1.5 + 0.4 * abs(box_h - box_w)
                else:
                    weight *= 1.4 + 0.2 * (box_h + box_w)
                box_choices.append((box_h, box_w, weight))

        if not box_choices:
            return None

        dims = [(h, w) for h, w, _ in box_choices]
        weights = [wt for _, _, wt in box_choices]
        box_h, box_w = random.choices(dims, weights=weights, k=1)[0]

        shape = _grow_shape(area, box_h, box_w, mode)
        if shape is None:
            continue

        rows = {r for r, _ in shape}
        cols = {c for _, c in shape}
        bbox_h = 1 + max(rows)
        bbox_w = 1 + max(cols)
        is_line = bbox_h == 1 or bbox_w == 1
        has_corner = (0, 0) in shape

        if require_corner_gap:
            if has_corner or bbox_h == 1 or bbox_w == 1:
                continue
        elif area >= 4 and is_line and random.random() < 0.85:
            continue

        if area >= 5 and len(rows) == 1:
            continue
        if area >= 5 and len(cols) == 1:
            continue

        return shape

    return None


def _render_input(shape: tuple[int, int], objects: list[dict]) -> np.ndarray:
    grid = _checkerboard(shape, AZURE, WHITE)
    for obj in objects:
        for r, c in obj["cells"]:
            grid[r, c] = int(obj["color"])
    return grid


def _render_output(shape: tuple[int, int], objects: list[dict]) -> np.ndarray:
    out = _checkerboard(shape, AZURE, WHITE)
    for obj in sorted(objects, key=lambda x: (x["anchor"][0], x["anchor"][1], x["color"])):
        r0, c0 = obj["anchor"]
        area = int(obj["area"])
        out[r0, c0 : min(shape[1], c0 + area)] = int(obj["color"])
    return out


def _try_place_object(
    h: int,
    w: int,
    blocked: np.ndarray,
    reserved: dict[int, list[tuple[int, int]]],
    must_truncate: bool,
    color: int,
) -> dict | None:
    for _ in range(PLACEMENT_RETRIES):
        anchor_r = random.randrange(h)
        anchor_c = random.randrange(w)
        avail_h = h - anchor_r
        avail_w = w - anchor_c
        avail_area = avail_h * avail_w
        if avail_area < MIN_AREA:
            continue

        area_hi = min(MAX_AREA, avail_area)
        if must_truncate:
            if avail_h < 2 or area_hi <= avail_w:
                continue
            area_lo = max(MIN_AREA, avail_w + 1)
        else:
            area_hi = min(area_hi, avail_w)
            area_lo = MIN_AREA
        if area_hi < area_lo:
            continue

        area = random.randint(area_lo, area_hi)
        line_end = min(w - 1, anchor_c + area - 1)
        if not _line_is_clear(reserved, anchor_r, anchor_c, line_end):
            continue

        require_corner_gap = area >= 4 and random.random() < CORNER_GAP_PROB
        shape = _sample_shape(area, avail_h, avail_w, require_corner_gap)
        if shape is None:
            continue

        cells = [(anchor_r + dr, anchor_c + dc) for dr, dc in shape]
        if any(blocked[r, c] for r, c in cells):
            continue

        return {
            "color": color,
            "cells": cells,
            "area": area,
            "anchor": (anchor_r, anchor_c),
        }

    return None


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    n_objects = _sample_object_count(h, w)
    colors = random.sample(FG_COLORS, n_objects)

    truncate_targets: set[int] = set()
    if random.random() < TRUNCATE_PROB:
        truncate_targets.add(random.randrange(n_objects))
        if n_objects >= 4 and random.random() < 0.2:
            truncate_targets.add(random.randrange(n_objects))

    objects: list[dict] = []
    blocked = np.zeros((h, w), dtype=bool)
    reserved: dict[int, list[tuple[int, int]]] = {}

    for idx in range(n_objects):
        obj = _try_place_object(
            h,
            w,
            blocked,
            reserved,
            must_truncate=idx in truncate_targets,
            color=colors[idx],
        )
        if obj is None:
            return None

        objects.append(obj)
        _mark_buffer(blocked, obj["cells"])
        row, col = obj["anchor"]
        end = min(w - 1, col + int(obj["area"]) - 1)
        reserved.setdefault(row, []).append((col, end))

    total_area = sum(int(obj["area"]) for obj in objects)
    if total_area > int(h * w * 0.36):
        return None

    grid = _render_input((h, w), objects)
    expected = _render_output((h, w), objects)

    if np.count_nonzero((grid != AZURE) & (grid != WHITE)) == 0:
        return None
    if np.array_equal(grid, expected):
        return None
    if not np.array_equal(main(grid), expected):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid checkerboard-compression input")
