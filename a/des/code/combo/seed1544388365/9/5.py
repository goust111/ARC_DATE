import random
from collections import deque

import numpy as np


BG = 0
GREY = 5
PATH_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))

GRID_H_RANGE = (18, 32)
GRID_W_RANGE = (18, 32)
OBJECT_COUNT_RANGE = (1, 4)
COARSE_ROW_RANGE = (3, 6)
COARSE_COL_RANGE = (3, 6)
PATCH_GAP = 1
PATCH_BUILD_RETRIES = 120
PATCH_PLACE_RETRIES = 80
GEN_RETRIES = 300


def _neighbors4(r: int, c: int):
    for dr, dc in DIR4:
        yield r + dr, c + dc


def _connected_components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    components: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            cells: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                cells.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            components.append(cells)
    return components


def _extract_paths(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    colors = sorted({int(v) for v in g.flatten() if int(v) not in (BG, GREY)})
    paths: list[dict] = []
    for color in colors:
        comps = _connected_components(g == color)
        for comp in comps:
            cell_set = set(comp)
            endpoints: list[tuple[int, int]] = []
            for r, c in comp:
                degree = sum((nr, nc) in cell_set for nr, nc in _neighbors4(r, c))
                if degree == 1:
                    endpoints.append((r, c))
                else:
                    assert degree == 2, "path components must be unbranched"
            assert len(endpoints) == 2, "each path must have exactly two endpoints"
            paths.append(
                {
                    "color": color,
                    "cells": sorted(comp),
                    "cell_set": cell_set,
                    "endpoints": sorted(endpoints),
                    "length": len(comp),
                }
            )
    paths.sort(key=lambda item: (item["color"], item["endpoints"]))
    return paths


def _rectangle_perimeter(height: int, width: int) -> int:
    if height == 1 and width == 1:
        return 1
    if height == 1:
        return width
    if width == 1:
        return height
    return 2 * (height + width) - 4


def _draw_rectangle(out: np.ndarray, r0: int, c0: int, r1: int, c1: int, color: int, filled: bool) -> None:
    if filled:
        out[r0 : r1 + 1, c0 : c1 + 1] = color
        return
    out[r0, c0 : c1 + 1] = color
    out[r1, c0 : c1 + 1] = color
    out[r0 : r1 + 1, c0] = color
    out[r0 : r1 + 1, c1] = color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Shrink each colored path to its endpoints, then draw an outline or filled
    rectangle whose diagonal is defined by those endpoints.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    paths = _extract_paths(g)

    for info in paths:
        for r, c in info["cells"]:
            out[r, c] = BG

    for info in paths:
        color = int(info["color"])
        (r_a, c_a), (r_b, c_b) = info["endpoints"]
        r0, r1 = sorted((r_a, r_b))
        c0, c1 = sorted((c_a, c_b))
        height = r1 - r0 + 1
        width = c1 - c0 + 1
        perimeter = _rectangle_perimeter(height, width)
        filled = int(info["length"]) > perimeter
        _draw_rectangle(out, r0, c0, r1, c1, color, filled)

    out[g == GREY] = GREY
    return out


def _snake_vertices(rows: int, cols: int) -> list[tuple[int, int]]:
    order: list[tuple[int, int]] = []
    for r in range(rows):
        if r % 2 == 0:
            iterable = range(cols)
        else:
            iterable = range(cols - 1, -1, -1)
        for c in iterable:
            order.append((r, c))
    return order


def _coarse_to_local(r: int, c: int) -> tuple[int, int]:
    return 2 * r + 2, 2 * c + 2


def _expand_segment(segment: list[tuple[int, int]]) -> list[tuple[int, int]]:
    path: list[tuple[int, int]] = []
    pr, pc = _coarse_to_local(*segment[0])
    path.append((pr, pc))
    for vr, vc in segment[1:]:
        nr, nc = _coarse_to_local(vr, vc)
        path.append(((pr + nr) // 2, (pc + nc) // 2))
        path.append((nr, nc))
        pr, pc = nr, nc
    return path


def _touches(set_a: set[tuple[int, int]], set_b: set[tuple[int, int]]) -> bool:
    return any(abs(r1 - r2) + abs(c1 - c2) == 1 for r1, c1 in set_a for r2, c2 in set_b)


def _anchor_candidates(
    endpoint: tuple[int, int],
    neighbor: tuple[int, int],
    path_set: set[tuple[int, int]],
    height: int,
    width: int,
) -> list[list[tuple[int, int]]]:
    r, c = endpoint
    dr, dc = neighbor[0] - r, neighbor[1] - c
    blocked_side = {
        (-1, 0): "up",
        (1, 0): "down",
        (0, -1): "left",
        (0, 1): "right",
    }[(dr, dc)]
    shapes = {
        "up": [
            [(r - 2, c - 1), (r - 2, c), (r - 1, c - 1), (r - 1, c)],
            [(r - 2, c), (r - 2, c + 1), (r - 1, c), (r - 1, c + 1)],
        ],
        "down": [
            [(r + 1, c - 1), (r + 1, c), (r + 2, c - 1), (r + 2, c)],
            [(r + 1, c), (r + 1, c + 1), (r + 2, c), (r + 2, c + 1)],
        ],
        "left": [
            [(r - 1, c - 2), (r - 1, c - 1), (r, c - 2), (r, c - 1)],
            [(r, c - 2), (r, c - 1), (r + 1, c - 2), (r + 1, c - 1)],
        ],
        "right": [
            [(r - 1, c + 1), (r - 1, c + 2), (r, c + 1), (r, c + 2)],
            [(r, c + 1), (r, c + 2), (r + 1, c + 1), (r + 1, c + 2)],
        ],
    }

    options: list[list[tuple[int, int]]] = []
    for side, variants in shapes.items():
        if side == blocked_side:
            continue
        for block in variants:
            if any(not (0 <= br < height and 0 <= bc < width) for br, bc in block):
                continue
            block_set = set(block)
            if block_set & path_set:
                continue
            touching = {
                cell
                for cell in path_set
                if any(abs(cell[0] - br) + abs(cell[1] - bc) == 1 for br, bc in block_set)
            }
            if touching != {endpoint}:
                continue
            options.append(block)
    return options


def _segment_candidates(rows: int, cols: int, want_filled: bool) -> list[tuple[int, int]]:
    snake = _snake_vertices(rows, cols)
    candidates: list[tuple[int, int]] = []
    for i in range(len(snake) - 2):
        for j in range(i + 2, len(snake)):
            a = snake[i]
            b = snake[j]
            dr = abs(a[0] - b[0])
            dc = abs(a[1] - b[1])
            if dr == 0 or dc == 0:
                continue
            path_length = 2 * (j - i + 1) - 1
            rect_height = 2 * dr + 1
            rect_width = 2 * dc + 1
            perimeter = _rectangle_perimeter(rect_height, rect_width)
            if (path_length > perimeter) == want_filled:
                candidates.append((i, j))
    return candidates


def _make_object_patch(rng: random.Random, color: int, want_filled: bool) -> np.ndarray | None:
    for _ in range(PATCH_BUILD_RETRIES):
        rows = rng.randint(*COARSE_ROW_RANGE)
        cols = rng.randint(*COARSE_COL_RANGE)
        snake = _snake_vertices(rows, cols)
        candidates = _segment_candidates(rows, cols, want_filled)
        if not candidates:
            continue
        rng.shuffle(candidates)
        patch_h = 2 * rows + 3
        patch_w = 2 * cols + 3
        for i, j in candidates:
            segment = snake[i : j + 1]
            path = _expand_segment(segment)
            path_set = set(path)
            a_options = _anchor_candidates(path[0], path[1], path_set, patch_h, patch_w)
            b_options = _anchor_candidates(path[-1], path[-2], path_set, patch_h, patch_w)
            if not a_options or not b_options:
                continue
            rng.shuffle(a_options)
            rng.shuffle(b_options)
            for a_block in a_options:
                set_a = set(a_block)
                for b_block in b_options:
                    set_b = set(b_block)
                    if set_a & set_b:
                        continue
                    if _touches(set_a, set_b):
                        continue
                    patch = np.full((patch_h, patch_w), BG, dtype=int)
                    for r, c in path:
                        patch[r, c] = color
                    for r, c in a_block + b_block:
                        patch[r, c] = GREY
                    if rng.random() < 0.5:
                        patch = np.fliplr(patch)
                    if rng.random() < 0.5:
                        patch = np.flipud(patch)
                    rotations = rng.randrange(4)
                    if rotations:
                        patch = np.rot90(patch, rotations)
                    return np.array(patch, dtype=int, copy=True)
    return None


def _place_patch(
    rng: random.Random,
    grid: np.ndarray,
    reserved: np.ndarray,
    patch: np.ndarray,
) -> bool:
    h, w = grid.shape
    ph, pw = patch.shape
    if ph > h or pw > w:
        return False
    for _ in range(PATCH_PLACE_RETRIES):
        top = rng.randint(0, h - ph)
        left = rng.randint(0, w - pw)
        r0 = max(0, top - PATCH_GAP)
        c0 = max(0, left - PATCH_GAP)
        r1 = min(h, top + ph + PATCH_GAP)
        c1 = min(w, left + pw + PATCH_GAP)
        if reserved[r0:r1, c0:c1].any():
            continue
        mask = patch != BG
        grid[top : top + ph, left : left + pw][mask] = patch[mask]
        reserved[top : top + ph, left : left + pw] = True
        return True
    return False


def _validate_input(grid: np.ndarray) -> None:
    g = np.asarray(grid, dtype=int)
    paths = _extract_paths(g)
    assert paths, "generator must place at least one path"

    grey_components = _connected_components(g == GREY)
    assert grey_components, "generator must place anchor blocks"
    grey_sets = [set(comp) for comp in grey_components]
    for comp in grey_components:
        assert len(comp) == 4
        rows = sorted({r for r, _ in comp})
        cols = sorted({c for _, c in comp})
        assert len(rows) == 2 and len(cols) == 2
        expected = {(rows[0], cols[0]), (rows[0], cols[1]), (rows[1], cols[0]), (rows[1], cols[1])}
        assert set(comp) == expected

    for info in paths:
        assert info["length"] >= 4
        endpoint_set = set(info["endpoints"])
        for r, c in info["cells"]:
            if (r, c) in endpoint_set:
                adjacent_anchors = 0
                for grey_set in grey_sets:
                    if any(abs(r - gr) + abs(c - gc) == 1 for gr, gc in grey_set):
                        adjacent_anchors += 1
                assert adjacent_anchors == 1
            else:
                for grey_set in grey_sets:
                    assert not any(abs(r - gr) + abs(c - gc) == 1 for gr, gc in grey_set)

    out = main(g)
    assert out.shape == g.shape
    assert np.all(out[g == GREY] == GREY)


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    height = rng.randint(*GRID_H_RANGE)
    width = rng.randint(*GRID_W_RANGE)
    grid = np.full((height, width), BG, dtype=int)
    reserved = np.zeros((height, width), dtype=bool)

    object_count = rng.randint(*OBJECT_COUNT_RANGE)
    colors = rng.sample(PATH_COLORS, object_count)
    if object_count == 1:
        branch_flags = [rng.random() < 0.5]
    else:
        branch_flags = [False, True]
        while len(branch_flags) < object_count:
            branch_flags.append(rng.random() < 0.5)
        rng.shuffle(branch_flags)

    patches: list[np.ndarray] = []
    for color, want_filled in zip(colors, branch_flags):
        patch = _make_object_patch(rng, color, want_filled)
        if patch is None:
            return None
        patches.append(patch)

    order = list(range(object_count))
    rng.shuffle(order)
    for idx in order:
        if not _place_patch(rng, grid, reserved, patches[idx]):
            return None

    try:
        _validate_input(grid)
    except AssertionError:
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input grid")
