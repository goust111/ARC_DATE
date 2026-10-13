import random
from collections import deque

import numpy as np


BG = 0
ARC_COLORS = tuple(range(1, 10))
GEN_RETRIES = 400
FRAME_RETRIES = 200


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
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
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _parse_top_legend(grid: np.ndarray) -> tuple[dict[int, int] | None, set[tuple[int, int]]]:
    row = grid[0]
    w = grid.shape[1]
    if not np.any(row != BG):
        return None, set()

    mapping: dict[int, int] = {}
    cells: set[tuple[int, int]] = set()
    c = 0
    while c < w:
        if int(row[c]) == BG:
            c += 1
            continue
        if c + 1 >= w or int(row[c + 1]) == BG:
            return None, set()
        src = int(row[c])
        dst = int(row[c + 1])
        mapping[src] = dst
        cells.add((0, c))
        cells.add((0, c + 1))
        c += 2
    return mapping or None, cells


def _parse_left_legend(grid: np.ndarray) -> tuple[dict[int, int] | None, set[tuple[int, int]]]:
    h, w = grid.shape
    if w < 2 or not np.any(grid[:, :2] != BG):
        return None, set()

    mapping: dict[int, int] = {}
    cells: set[tuple[int, int]] = set()
    r = 0
    while r < h:
        left = int(grid[r, 0])
        right = int(grid[r, 1])
        if left == BG and right == BG:
            r += 1
            continue
        if left == BG or right == BG:
            return None, set()
        mapping[left] = right
        cells.add((r, 0))
        cells.add((r, 1))
        r += 1
    return mapping or None, cells


def _legend_mapping(grid: np.ndarray) -> tuple[dict[int, int], set[tuple[int, int]]]:
    top_mapping, top_cells = _parse_top_legend(grid)
    left_mapping, left_cells = _parse_left_legend(grid)
    candidates: list[tuple[int, dict[int, int], set[tuple[int, int]]]] = []
    if top_mapping:
        candidates.append((len(top_mapping), top_mapping, top_cells))
    if left_mapping:
        candidates.append((len(left_mapping), left_mapping, left_cells))
    if not candidates:
        raise AssertionError("legend mapping not found")
    candidates.sort(key=lambda item: item[0], reverse=True)
    _, mapping, cells = candidates[0]
    return mapping, cells


def _rectangle_border_cells(r0: int, c0: int, r1: int, c1: int) -> set[tuple[int, int]]:
    cells = {(r0, c) for c in range(c0, c1 + 1)}
    cells |= {(r1, c) for c in range(c0, c1 + 1)}
    cells |= {(r, c0) for r in range(r0, r1 + 1)}
    cells |= {(r, c1) for r in range(r0, r1 + 1)}
    return cells


def _find_frame_bbox(grid: np.ndarray, color: int) -> tuple[int, int, int, int] | None:
    candidates: list[tuple[int, int, int, int]] = []
    for comp in _components(grid == color):
        rows = [r for r, _ in comp]
        cols = [c for _, c in comp]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        if r1 - r0 < 2 or c1 - c0 < 2:
            continue
        border = _rectangle_border_cells(r0, c0, r1, c1)
        if set(comp) == border:
            candidates.append((r0, c0, r1, c1))
    if not candidates:
        return None
    candidates.sort(key=lambda box: (box[2] - box[0]) * (box[3] - box[1]), reverse=True)
    return candidates[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode each rectangular frame's seed color through the edge legend, then spread the decoded color in a cross inside that frame.

    Input structure:
    - The background is black.
    - The grid contains several single-color hollow rectangular frames, each with a distinct border color.
    - A legend sits on the top edge or left edge and pairs each frame color with a decoded fill color.
    - Inside every frame, one or more seed pixels use that frame's own border color on an otherwise black interior.

    Rule:
    - Read the legend to build a border-color to decoded-color lookup table.
    - Find each closed rectangular frame and locate all seed pixels strictly inside it.
    - Replace every seed pixel with the decoded color for that frame.
    - From each seed position, extend straight up, down, left, and right through the frame interior, coloring cells until the frame border is reached.
    - Keep the legend, the frame borders, and all other black cells unchanged.

    Key insight:
    - The border color is not the final paint color; it must first be translated by the legend, and the translated color is then propagated from the seed positions as axis-aligned cross arms inside each closed frame.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    mapping, _legend_cells = _legend_mapping(g)
    for border_color, decoded_color in mapping.items():
        bbox = _find_frame_bbox(g, border_color)
        if bbox is None:
            continue
        r0, c0, r1, c1 = bbox
        seeds = [
            (r, c)
            for r in range(r0 + 1, r1)
            for c in range(c0 + 1, c1)
            if int(g[r, c]) == border_color
        ]
        if not seeds:
            continue

        for r, c in seeds:
            out[r, c] = decoded_color

        for r, c in seeds:
            for rr in range(r0 + 1, r1):
                out[rr, c] = decoded_color
            for cc in range(c0 + 1, c1):
                out[r, cc] = decoded_color

    return out


def _place_legend(grid: np.ndarray, mapping: list[tuple[int, int]], side: str, rng: random.Random) -> None:
    h, w = grid.shape
    if side == "top":
        span = 3 * len(mapping) - 1
        start = rng.randint(1, w - span)
        c = start
        for i, (src, dst) in enumerate(mapping):
            grid[0, c] = src
            grid[0, c + 1] = dst
            c += 2
            if i != len(mapping) - 1:
                c += 1
        return

    span = 2 * len(mapping) - 1
    start = rng.randint(1, h - span)
    r = start
    for i, (src, dst) in enumerate(mapping):
        grid[r, 0] = src
        grid[r, 1] = dst
        r += 1
        if i != len(mapping) - 1:
            r += 1


def _sample_seeds(
    rng: random.Random,
    r0: int,
    c0: int,
    r1: int,
    c1: int,
) -> list[tuple[int, int]]:
    inner_rows = list(range(r0 + 1, r1))
    inner_cols = list(range(c0 + 1, c1))
    cells = [(r, c) for r in inner_rows for c in inner_cols]
    max_count = min(3, len(cells))
    if max_count == 1:
        return [cells[0]]

    if rng.random() < 0.5:
        seed_count = rng.randint(1, max_count)
    else:
        seed_count = rng.randint(2, max_count)

    if seed_count > 1 and rng.random() < 0.5:
        if rng.random() < 0.5 and len(inner_cols) >= seed_count:
            row = rng.choice(inner_rows)
            cols = rng.sample(inner_cols, seed_count)
            return [(row, c) for c in cols]
        if len(inner_rows) >= seed_count:
            col = rng.choice(inner_cols)
            rows = rng.sample(inner_rows, seed_count)
            return [(r, col) for r in rows]

    return rng.sample(cells, seed_count)


def _draw_frame(grid: np.ndarray, r0: int, c0: int, r1: int, c1: int, color: int) -> None:
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    n_frames = rng.randint(2, 4)
    side = rng.choice(["top", "left"])
    colors = list(ARC_COLORS)
    rng.shuffle(colors)
    border_colors = colors[:n_frames]
    decoded_colors = colors[n_frames : 2 * n_frames]
    rng.shuffle(decoded_colors)
    mapping = list(zip(border_colors, decoded_colors))

    legend_span = 3 * n_frames - 1
    min_h = max(12, 2 * n_frames + 8)
    min_w = 12
    if side == "top":
        min_w = max(min_w, legend_span + 2)
    else:
        min_h = max(min_h, 2 * n_frames + 2)

    h = rng.randint(min_h, 30)
    w = rng.randint(min_w, 30)
    if side == "left" and w < 12:
        return None

    grid = np.zeros((h, w), dtype=int)
    _place_legend(grid, mapping, side, rng)

    blocked = np.zeros((h, w), dtype=bool)
    top_margin = 2 if side == "top" else 1
    left_margin = 3 if side == "left" else 1
    max_dim = 10 if n_frames == 2 else 9 if n_frames == 3 else 8

    for border_color, _decoded_color in mapping:
        placed = False
        max_h = min(max_dim, h - top_margin - 1)
        max_w = min(max_dim, w - left_margin - 1)
        if max_h < 5 or max_w < 5:
            return None
        for _ in range(FRAME_RETRIES):
            fh = rng.randint(5, max_h)
            fw = rng.randint(5, max_w)
            r0 = rng.randint(top_margin, h - fh - 1)
            c0 = rng.randint(left_margin, w - fw - 1)
            r1 = r0 + fh - 1
            c1 = c0 + fw - 1
            er0 = max(0, r0 - 1)
            ec0 = max(0, c0 - 1)
            er1 = min(h - 1, r1 + 1)
            ec1 = min(w - 1, c1 + 1)
            if blocked[er0 : er1 + 1, ec0 : ec1 + 1].any():
                continue

            blocked[er0 : er1 + 1, ec0 : ec1 + 1] = True
            _draw_frame(grid, r0, c0, r1, c1, border_color)
            for sr, sc in _sample_seeds(rng, r0, c0, r1, c1):
                grid[sr, sc] = border_color
            placed = True
            break
        if not placed:
            return None

    try:
        out = main(grid)
    except Exception:
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
