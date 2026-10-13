import math
import random
from collections import Counter

import numpy as np


START_COLOR = 2
TURN_COLOR = 4
CODE_COLOR = 5
RESERVED_COLORS = {START_COLOR, TURN_COLOR, CODE_COLOR}
GEN_RETRIES = 400


def _neighbors4(r: int, c: int, h: int, w: int) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    if r > 0:
        out.append((r - 1, c))
    if r + 1 < h:
        out.append((r + 1, c))
    if c > 0:
        out.append((r, c - 1))
    if c + 1 < w:
        out.append((r, c + 1))
    return out


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc, h, w):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _trace_path(mask: np.ndarray, start: tuple[int, int]) -> list[tuple[int, int]]:
    h, w = mask.shape
    path = [start]
    seen = {start}
    prev: tuple[int, int] | None = None
    cur = start
    while True:
        nxts = []
        for nr, nc in _neighbors4(cur[0], cur[1], h, w):
            if mask[nr, nc] and (nr, nc) != prev:
                nxts.append((nr, nc))
        if not nxts:
            break
        assert len(nxts) == 1, "spiral track must be a simple path"
        nxt = nxts[0]
        assert nxt not in seen, "spiral track must not branch or loop"
        seen.add(nxt)
        path.append(nxt)
        prev, cur = cur, nxt
    assert len(seen) == int(mask.sum()), "path trace must cover the whole spiral"
    return path


def _corner_indices(path: list[tuple[int, int]]) -> set[int]:
    corners: set[int] = set()
    for i in range(1, len(path) - 1):
        dr1 = path[i][0] - path[i - 1][0]
        dc1 = path[i][1] - path[i - 1][1]
        dr2 = path[i + 1][0] - path[i][0]
        dc2 = path[i + 1][1] - path[i][1]
        if (dr1, dc1) != (dr2, dc2):
            corners.add(i)
    return corners


def _extract_clues(grid: np.ndarray, bg: int, track_color: int) -> list[tuple[int, int]]:
    h, w = grid.shape
    gray_components = _components(grid == CODE_COLOR)
    gray_by_cell: dict[tuple[int, int], int] = {}
    gray_sizes: list[int] = []
    for idx, comp in enumerate(gray_components):
        gray_sizes.append(len(comp))
        for cell in comp:
            gray_by_cell[cell] = idx

    clues: list[tuple[int, int]] = []
    used_gray: set[int] = set()
    colors = sorted(set(map(int, grid.reshape(-1))))
    block_colors = [c for c in colors if c not in {bg, track_color, START_COLOR, CODE_COLOR}]
    for color in block_colors:
        for comp in _components(grid == color):
            rs = [r for r, _ in comp]
            cs = [c for _, c in comp]
            area = len(comp)
            bh = max(rs) - min(rs) + 1
            bw = max(cs) - min(cs) + 1
            assert area in (1, 4), "colored clue blocks must be 1x1 or 2x2"
            assert max(bh, bw) <= 2 and bh == bw, "colored clue blocks must be square"

            adjacent_gray = set()
            for r, c in comp:
                for nr, nc in _neighbors4(r, c, h, w):
                    idx = gray_by_cell.get((nr, nc))
                    if idx is not None:
                        adjacent_gray.add(idx)
            assert len(adjacent_gray) == 1, "each colored block must match one gray code"
            gray_idx = next(iter(adjacent_gray))
            assert gray_idx not in used_gray, "gray codes must be one-to-one with blocks"
            used_gray.add(gray_idx)
            clues.append((gray_sizes[gray_idx], color))

    assert len(clues) == len(gray_components), "all gray codes must be paired"
    return clues


def main(grid: np.ndarray) -> np.ndarray:
    """
    Paint clue colors onto a spiral track and mark every turning corner.

    Input structure:
    - A dominant background surrounds a single-color one-cell-wide rectangular spiral track.
    - The spiral has a unique start marker pixel at its outer endpoint.
    - Outside the spiral lie several monochrome colored square blocks (1x1 or 2x2), each touching one gray code component whose area gives a step count.

    Rule:
    - Restore the start-marker cell back to the spiral track color, then follow the spiral from the marked endpoint.
    - For each colored block, count the pixels in its adjacent gray component and place that block's color onto the spiral cell at that 1-based step index.
    - Every spiral turn is highlighted with the bright corner color.
    - All other spiral cells remain the original track color, and the background stays unchanged.

    Key insight:
    - The gray components encode discrete positions by area; solving the task requires tracing the spiral's true geometric order and then indexing into that path.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = g.shape

    bg = _most_common_color(g)
    starts = np.argwhere(g == START_COLOR)
    assert len(starts) == 1, "expected one start marker"
    sr, sc = map(int, starts[0])

    neighbor_colors = {
        int(g[nr, nc])
        for nr, nc in _neighbors4(sr, sc, h, w)
        if int(g[nr, nc]) not in {bg, START_COLOR, CODE_COLOR}
    }
    assert len(neighbor_colors) == 1, "start marker must touch exactly one track color"
    track_color = next(iter(neighbor_colors))

    path_mask = (g == track_color) | (g == START_COLOR)
    path = _trace_path(path_mask, (sr, sc))
    corners = _corner_indices(path)
    clues = _extract_clues(g, bg, track_color)

    out = np.full_like(g, bg)
    for r, c in path:
        out[r, c] = track_color
    for step, color in clues:
        assert 1 <= step < len(path), "step code must land on the spiral"
        out[path[step]] = color
    for idx in corners:
        out[path[idx]] = TURN_COLOR
    return out


def _build_base_spiral(height: int, width: int) -> list[tuple[int, int]]:
    dirs = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    occupied = np.zeros((height, width), dtype=bool)
    path = [(0, 0)]
    occupied[0, 0] = True
    r = c = 0
    dir_idx = 0

    def can_extend(nr: int, nc: int) -> bool:
        if not (0 <= nr < height and 0 <= nc < width):
            return False
        if occupied[nr, nc]:
            return False
        count = 0
        for ar, ac in _neighbors4(nr, nc, height, width):
            if occupied[ar, ac]:
                count += 1
        return count == 1

    while True:
        moved = False
        for _ in range(4):
            dr, dc = dirs[dir_idx]
            nr, nc = r + dr, c + dc
            if can_extend(nr, nc):
                r, c = nr, nc
                occupied[r, c] = True
                path.append((r, c))
                moved = True
                break
            dir_idx = (dir_idx + 1) % 4
        if not moved:
            break
    return path


def _rotate_coords(
    coords: list[tuple[int, int]], height: int, width: int, k: int
) -> tuple[list[tuple[int, int]], int, int]:
    out = coords
    h, w = height, width
    for _ in range(k % 4):
        out = [(c, h - 1 - r) for r, c in out]
        h, w = w, h
    return out, h, w


def _make_code_shape(area: int) -> tuple[list[tuple[int, int]], int, int]:
    options = []
    for rows in range(1, min(area, 5) + 1):
        cols = math.ceil(area / rows)
        if cols <= 5:
            options.append((rows, cols))
    if not options:
        rows = min(area, 5)
        options.append((rows, math.ceil(area / rows)))
    rows, cols = random.choice(options)

    order_name = random.choice(["row", "row_rev", "col", "col_rev", "snake"])
    order: list[tuple[int, int]] = []
    if order_name == "row":
        for r in range(rows):
            for c in range(cols):
                order.append((r, c))
    elif order_name == "row_rev":
        for r in range(rows - 1, -1, -1):
            for c in range(cols - 1, -1, -1):
                order.append((r, c))
    elif order_name == "col":
        for c in range(cols):
            for r in range(rows):
                order.append((r, c))
    elif order_name == "col_rev":
        for c in range(cols - 1, -1, -1):
            for r in range(rows - 1, -1, -1):
                order.append((r, c))
    else:
        for r in range(rows):
            cols_iter = range(cols) if r % 2 == 0 else range(cols - 1, -1, -1)
            for c in cols_iter:
                order.append((r, c))
    coords = order[:area]
    rot = random.randrange(4)
    return _rotate_coords(coords, rows, cols, rot)


def _make_clue_pattern(step: int, block_size: int, color: int) -> dict:
    code_cells, ch, cw = _make_code_shape(step)
    orient = random.choice(["right", "left", "down", "up"])

    if orient == "right":
        block_offset = (0, 0)
        code_offset = (0, block_size)
        height, width = max(block_size, ch), block_size + cw
    elif orient == "left":
        block_offset = (0, cw)
        code_offset = (0, 0)
        height, width = max(block_size, ch), cw + block_size
    elif orient == "down":
        block_offset = (0, 0)
        code_offset = (block_size, 0)
        height, width = block_size + ch, max(block_size, cw)
    else:
        block_offset = (ch, 0)
        code_offset = (0, 0)
        height, width = ch + block_size, max(block_size, cw)

    block_cells = [
        (block_offset[0] + r, block_offset[1] + c)
        for r in range(block_size)
        for c in range(block_size)
    ]
    shifted_code = [(code_offset[0] + r, code_offset[1] + c) for r, c in code_cells]
    return {
        "step": step,
        "color": color,
        "height": height,
        "width": width,
        "block_cells": block_cells,
        "code_cells": shifted_code,
    }


def _paint_pattern(grid: np.ndarray, pattern: dict, top: int, left: int) -> None:
    for r, c in pattern["block_cells"]:
        grid[top + r, left + c] = pattern["color"]
    for r, c in pattern["code_cells"]:
        grid[top + r, left + c] = CODE_COLOR


def _place_patterns(
    grid: np.ndarray, forbidden: np.ndarray, patterns: list[dict]
) -> bool:
    h, w = grid.shape
    occupied = forbidden.copy()
    placements: list[tuple[dict, int, int]] = []

    for pattern in sorted(patterns, key=lambda p: (p["height"] * p["width"], p["step"]), reverse=True):
        ph = int(pattern["height"])
        pw = int(pattern["width"])
        candidates = []
        for r0 in range(h - ph + 1):
            for c0 in range(w - pw + 1):
                if occupied[r0 : r0 + ph, c0 : c0 + pw].any():
                    continue
                candidates.append((r0, c0))
        if not candidates:
            return False
        top, left = random.choice(candidates)
        placements.append((pattern, top, left))
        r1 = max(0, top - 1)
        c1 = max(0, left - 1)
        r2 = min(h, top + ph + 1)
        c2 = min(w, left + pw + 1)
        occupied[r1:r2, c1:c2] = True

    for pattern, top, left in placements:
        _paint_pattern(grid, pattern, top, left)
    return True


def _expected_output(
    height: int,
    width: int,
    bg: int,
    track_color: int,
    path: list[tuple[int, int]],
    corners: set[int],
    clues: list[tuple[int, int]],
) -> np.ndarray:
    out = np.full((height, width), bg, dtype=int)
    for r, c in path:
        out[r, c] = track_color
    for step, color in clues:
        out[path[step]] = color
    for idx in corners:
        out[path[idx]] = TURN_COLOR
    return out


def _gen_attempt() -> np.ndarray | None:
    base_h = random.randint(7, 11)
    base_w = random.randint(7, 11)
    base_path = _build_base_spiral(base_h, base_w)
    if len(base_path) < 18:
        return None

    rotation = random.randrange(4)
    local_path, spiral_h, spiral_w = _rotate_coords(base_path, base_h, base_w, rotation)
    corners = _corner_indices(local_path)

    n_clues = random.randint(3, 5)
    max_step = min(len(local_path) - 2, random.randint(8, 14))
    eligible_steps = [i for i in range(1, max_step + 1) if i not in corners]
    if len(eligible_steps) < n_clues:
        return None
    step_values = sorted(random.sample(eligible_steps, n_clues))

    min_h = spiral_h + 7
    min_w = spiral_w + 7
    if min_h > 30 or min_w > 30:
        return None
    height = random.randint(min_h, min(30, min_h + 7))
    width = random.randint(min_w, min(30, min_w + 7))

    if height - spiral_h < 4 or width - spiral_w < 4:
        return None
    top = random.randint(2, height - spiral_h - 2)
    left = random.randint(2, width - spiral_w - 2)

    palette = [c for c in range(10) if c not in RESERVED_COLORS]
    bg = random.choice(palette)
    remaining = [c for c in palette if c != bg]
    track_color = random.choice(remaining)
    clue_colors = random.sample([c for c in remaining if c != track_color], n_clues)

    grid = np.full((height, width), bg, dtype=int)
    global_path = [(top + r, left + c) for r, c in local_path]
    for r, c in global_path[1:]:
        grid[r, c] = track_color
    grid[global_path[0]] = START_COLOR

    patterns = []
    clue_pairs = []
    for step, color in zip(step_values, clue_colors):
        block_size = 1 if random.random() < 0.5 else 2
        patterns.append(_make_clue_pattern(step, block_size, color))
        clue_pairs.append((step, color))

    forbidden = np.zeros((height, width), dtype=bool)
    r1 = max(0, top - 1)
    c1 = max(0, left - 1)
    r2 = min(height, top + spiral_h + 1)
    c2 = min(width, left + spiral_w + 1)
    forbidden[r1:r2, c1:c2] = True

    if not _place_patterns(grid, forbidden, patterns):
        return None

    if _most_common_color(grid) != bg:
        return None

    expected = _expected_output(height, width, bg, track_color, global_path, corners, clue_pairs)
    try:
        if np.array_equal(main(grid), expected):
            return grid
    except Exception:
        return None
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid spiral-step input")
