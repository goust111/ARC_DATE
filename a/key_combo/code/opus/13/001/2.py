from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
ENDPOINT = 2
WALL = 9
BLOCK_SIZE = 5
OUTPUT_HEIGHT = 7
BLOCK_COLORS = [1, 3, 4, 5, 6, 7, 8]


def neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def bbox_coords(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _c in coords]
    cols = [c for _r, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def component_coords(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(coords)
    return comps


def shortest_path_info(
    passable: np.ndarray,
    starts: list[tuple[int, int]],
    ends: list[tuple[int, int]],
) -> tuple[list[tuple[int, int]], int, bool]:
    h, w = passable.shape
    dist = np.full((h, w), -1, dtype=int)
    prev_r = np.full((h, w), -1, dtype=int)
    prev_c = np.full((h, w), -1, dtype=int)
    q: deque[tuple[int, int]] = deque()

    for r, c in sorted(starts):
        dist[r, c] = 0
        q.append((r, c))

    end_set = set(ends)
    while q:
        r, c = q.popleft()
        for nr, nc in neighbors4(r, c):
            if not (0 <= nr < h and 0 <= nc < w):
                continue
            if not passable[nr, nc] or dist[nr, nc] >= 0:
                continue
            dist[nr, nc] = dist[r, c] + 1
            prev_r[nr, nc] = r
            prev_c[nr, nc] = c
            q.append((nr, nc))

    end_distances = [dist[r, c] for r, c in ends if dist[r, c] >= 0]
    assert end_distances, "endpoints must be connected"
    best = int(min(end_distances))
    best_ends = sorted((r, c) for r, c in ends if dist[r, c] == best)
    goal = best_ends[0]

    counts = np.zeros((h, w), dtype=int)
    for r, c in starts:
        counts[r, c] = 1
    for d in range(best):
        cells = np.argwhere(dist == d)
        for r, c in cells:
            ways = int(counts[r, c])
            if ways == 0:
                continue
            for nr, nc in neighbors4(int(r), int(c)):
                if 0 <= nr < h and 0 <= nc < w and dist[nr, nc] == d + 1:
                    counts[nr, nc] = min(2, counts[nr, nc] + ways)
    n_shortest = min(2, sum(int(counts[r, c]) for r, c in ends if dist[r, c] == best))

    path: list[tuple[int, int]] = []
    cur = goal
    while True:
        path.append(cur)
        r, c = cur
        if prev_r[r, c] < 0:
            break
        cur = (int(prev_r[r, c]), int(prev_c[r, c]))
    path.reverse()
    return path, best, n_shortest == 1


def carve_line(grid: np.ndarray, r0: int, c0: int, r1: int, c1: int) -> None:
    if r0 == r1:
        step = 1 if c1 >= c0 else -1
        for c in range(c0, c1 + step, step):
            grid[r0, c] = BG
        return
    assert c0 == c1
    step = 1 if r1 >= r0 else -1
    for r in range(r0, r1 + step, step):
        grid[r, c0] = BG


def carve_room(grid: np.ndarray, top: int, left: int) -> None:
    grid[top : top + BLOCK_SIZE, left : left + BLOCK_SIZE] = BG


def paint_room(grid: np.ndarray, top: int, left: int, color: int) -> None:
    grid[top : top + BLOCK_SIZE, left : left + BLOCK_SIZE] = color


def transformed_mask(mask: np.ndarray) -> np.ndarray:
    out = mask.copy()
    if random.random() < 0.5:
        out = np.fliplr(out)
    k = random.randint(0, 3)
    out = np.rot90(out, k)
    coords = np.argwhere(out)
    r0 = int(coords[:, 0].min())
    c0 = int(coords[:, 1].min())
    r1 = int(coords[:, 0].max())
    c1 = int(coords[:, 1].max())
    return out[r0 : r1 + 1, c0 : c1 + 1]


MOTIF_LIBRARY = [
    np.array([[1, 1, 0], [0, 1, 0], [0, 1, 0]], dtype=bool),
    np.array([[1, 1, 1], [0, 1, 0], [0, 0, 0]], dtype=bool),
    np.array([[1, 0, 0], [1, 1, 1], [0, 0, 0]], dtype=bool),
    np.array([[1, 1, 0], [1, 0, 0], [1, 0, 0]], dtype=bool),
    np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool),
    np.array([[1, 1, 0], [0, 1, 1], [0, 0, 0]], dtype=bool),
    np.array([[1, 1, 1], [1, 0, 0], [0, 0, 0]], dtype=bool),
]


def identify_blocks_and_motifs(grid: np.ndarray):
    colors = [int(c) for c in np.unique(grid) if int(c) not in (BG, ENDPOINT, WALL)]
    blocks: list[dict] = []
    motifs: dict[int, np.ndarray] = {}
    for color in colors:
        comps = component_coords(grid == color)
        for coords in comps:
            r0, c0, r1, c1 = bbox_coords(coords)
            h = r1 - r0 + 1
            w = c1 - c0 + 1
            area = len(coords)
            if h == BLOCK_SIZE and w == BLOCK_SIZE and area == BLOCK_SIZE * BLOCK_SIZE:
                blocks.append({"color": color, "bbox": (r0, c0, r1, c1)})
            else:
                rel = np.zeros((h, w), dtype=bool)
                for r, c in coords:
                    rel[r - r0, c - c0] = True
                motifs[color] = rel
    blocks.sort(key=lambda item: (item["bbox"][0], item["bbox"][1]))
    return blocks, motifs


def carved_tile(color: int, motif: np.ndarray | None) -> np.ndarray:
    tile = np.full((BLOCK_SIZE, BLOCK_SIZE), color, dtype=int)
    if motif is None:
        return tile
    mh, mw = motif.shape
    off_r = (BLOCK_SIZE - mh) // 2
    off_c = (BLOCK_SIZE - mw) // 2
    tile[off_r : off_r + mh, off_c : off_c + mw][motif] = BG
    return tile


def main(grid: np.ndarray) -> np.ndarray:
    """
    Keep only the block rooms touched by the unique shortest maze path, carve them by their matching motifs, and pack them in encounter order.

    Input structure:
    - Black background and maroon wall pixels form a corridor maze in the upper part of the grid, with a wall-separated motif strip below.
    - Exactly two red 2x2 endpoint components mark the start and end of the route.
    - The maze contains 4 or 5 solid 5x5 monochrome blocks of distinct colors, plus one small same-colored motif component for each block in the lower strip.

    Rule:
    - Treat maroon wall pixels as blocked and every other color as passable, then find the unique shortest 4-connected path between the two red endpoint components.
    - A 5x5 block survives iff the path visits at least one cell inside that block's bounding box.
    - For each surviving block, center its same-colored motif over the block and erase the covered cells to black.
    - Pack the carved surviving blocks into a single row ordered by when the path first enters them, with a 1-cell black border and 1-cell gaps.

    Key insight:
    - The shortest path is only an implicit control signal: it decides both which color blocks remain and the left-to-right order of the final packed carved tiles.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    endpoint_comps = component_coords(g == ENDPOINT)
    assert len(endpoint_comps) == 2, "expected exactly two endpoint components"
    starts = endpoint_comps[0]
    ends = endpoint_comps[1]

    path, _distance, unique = shortest_path_info(g != WALL, starts, ends)
    assert unique, "expected a unique shortest path"

    blocks, motifs = identify_blocks_and_motifs(g)
    assert 4 <= len(blocks) <= 5, "expected 4 or 5 colored blocks"

    first_hit: dict[int, int] = {}
    for step, (r, c) in enumerate(path):
        for idx, block in enumerate(blocks):
            r0, c0, r1, c1 = block["bbox"]
            if idx in first_hit:
                continue
            if r0 <= r <= r1 and c0 <= c <= c1:
                first_hit[idx] = step

    ordered_indices = [idx for idx, _step in sorted(first_hit.items(), key=lambda item: item[1])]
    assert ordered_indices and len(ordered_indices) < len(blocks), "path must touch some but not all blocks"

    tiles = []
    for idx in ordered_indices:
        color = int(blocks[idx]["color"])
        tiles.append(carved_tile(color, motifs.get(color)))

    out_w = 2 + len(tiles) * BLOCK_SIZE + (len(tiles) - 1)
    out = np.zeros((OUTPUT_HEIGHT, out_w), dtype=int)
    for i, tile in enumerate(tiles):
        c0 = 1 + i * (BLOCK_SIZE + 1)
        out[1:6, c0 : c0 + BLOCK_SIZE] = tile
    return out


def build_maze(include_center: bool, short_top_1: bool, short_top_2: bool):
    maze_h = random.randint(15, 17)
    rm = random.choice([7, 8])
    left_margin = random.choice([1, 2])
    right_margin = random.choice([1, 2])
    width = left_margin + 25 + right_margin

    grid = np.full((maze_h, width), WALL, dtype=int)

    start_left = left_margin
    end_left = left_margin + 23
    j0 = left_margin + 3
    j1 = left_margin + 9
    j2 = left_margin + 15
    j3 = left_margin + 21

    def branch_rows(short_top: bool) -> tuple[int, int]:
        if short_top:
            return rm - 2, rm + 4
        return rm - 4, rm + 2

    top1_center, bot1_center = branch_rows(short_top_1)
    top2_center, bot2_center = branch_rows(short_top_2)

    room_specs = {
        "d1_top": (top1_center - 2, j0 + 1),
        "d1_bottom": (bot1_center - 2, j0 + 1),
        "d2_top": (top2_center - 2, j2 + 1),
        "d2_bottom": (bot2_center - 2, j2 + 1),
    }
    if include_center:
        room_specs["center"] = (rm - 2, j1 + 1)

    for top, left in room_specs.values():
        assert 0 <= top and top + BLOCK_SIZE <= maze_h
        carve_room(grid, top, left)

    start_exit_row = random.choice([rm - 1, rm])
    end_entry_row = random.choice([rm - 1, rm])

    grid[rm - 1 : rm + 1, start_left : start_left + 2] = ENDPOINT
    grid[rm - 1 : rm + 1, end_left : end_left + 2] = ENDPOINT

    carve_line(grid, start_exit_row, start_left + 2, start_exit_row, j0)
    carve_line(grid, end_entry_row, j3 + 1, end_entry_row, end_left - 1)

    def carve_branch(c_left: int, c_right: int, center_row: int) -> None:
        carve_line(grid, rm, c_left, center_row, c_left)
        carve_line(grid, center_row, c_left, center_row, c_right)
        carve_line(grid, center_row, c_right, rm, c_right)

    carve_branch(j0, j1, top1_center)
    carve_branch(j0, j1, bot1_center)

    if include_center:
        carve_line(grid, rm, j1, rm, j2)
        carve_room(grid, rm - 2, j1 + 1)
    else:
        carve_line(grid, rm, j1, rm, j2)

    carve_branch(j2, j3, top2_center)
    carve_branch(j2, j3, bot2_center)

    return grid, room_specs, (rm, start_left, end_left, start_exit_row, end_entry_row, width)


def place_motifs(grid: np.ndarray, sep_row: int, color_to_mask: dict[int, np.ndarray]) -> None:
    items = list(color_to_mask.items())
    random.shuffle(items)
    strip_top = sep_row + 1
    strip_h = grid.shape[0] - strip_top

    motif_y = strip_top + max(0, (strip_h - 3) // 2)
    start_x = random.choice([1, 2])
    cursor = start_x
    for color, mask in items:
        h, w = mask.shape
        top = motif_y + random.choice([0, max(0, strip_h - h)])
        if top + h > grid.shape[0]:
            top = grid.shape[0] - h
        left = cursor
        assert left + w <= grid.shape[1] - 1
        patch = np.full((h, w), BG, dtype=int)
        patch[mask] = color
        grid[top : top + h, left : left + w][mask] = color
        cursor += w + random.choice([1, 2])


def generate_attempt() -> np.ndarray | None:
    include_center = random.random() < 0.5
    short_top_1 = random.random() < 0.5
    short_top_2 = random.random() < 0.5
    maze, room_specs, _meta = build_maze(include_center, short_top_1, short_top_2)

    block_ids = ["d1_top", "d1_bottom"]
    if include_center:
        block_ids.append("center")
    block_ids += ["d2_top", "d2_bottom"]

    colors = random.sample(BLOCK_COLORS, len(block_ids))
    id_to_color = dict(zip(block_ids, colors))
    color_to_mask = {
        color: transformed_mask(random.choice(MOTIF_LIBRARY))
        for color in colors
    }

    for block_id in block_ids:
        top, left = room_specs[block_id]
        paint_room(maze, top, left, id_to_color[block_id])

    sep_row = maze.shape[0]
    motif_h = random.randint(4, 5)
    full = np.zeros((maze.shape[0] + 1 + motif_h, maze.shape[1]), dtype=int)
    full[: maze.shape[0], :] = maze
    full[sep_row, :] = WALL

    try:
        place_motifs(full, sep_row, color_to_mask)
    except AssertionError:
        return None

    endpoint_comps = component_coords(full == ENDPOINT)
    if len(endpoint_comps) != 2:
        return None
    path, _dist, unique = shortest_path_info(full != WALL, endpoint_comps[0], endpoint_comps[1])
    if not unique:
        return None

    blocks, motifs = identify_blocks_and_motifs(full)
    if len(blocks) != len(block_ids) or len(motifs) != len(block_ids):
        return None

    touched = 0
    for block in blocks:
        r0, c0, r1, c1 = block["bbox"]
        if any(r0 <= r <= r1 and c0 <= c <= c1 for r, c in path):
            touched += 1
    if not (1 <= touched < len(blocks)):
        return None

    try:
        out = main(full)
    except Exception:
        return None
    if out.ndim != 2 or max(out.shape) > 30:
        return None
    return full


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
