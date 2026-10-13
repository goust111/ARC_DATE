"""ARC task: count-limited rays trigger a chain reaction through a divider."""

from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
RED = 2
YELLOW = 4
GREY = 5
MARKERS = {RED, YELLOW}
BODY_PALETTE = (1, 3, 6, 7, 8, 9)
DIR4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _components8(mask: np.ndarray) -> list[np.ndarray]:
    """Return boolean-mask components as (N, 2) arrays of (row, column)."""
    h, w = mask.shape
    unseen = mask.copy()
    result: list[np.ndarray] = []
    for start_r, start_c in np.argwhere(mask):
        sr, sc = int(start_r), int(start_c)
        if not unseen[sr, sc]:
            continue
        unseen[sr, sc] = False
        todo = [(sr, sc)]
        cells = []
        while todo:
            r, c = todo.pop()
            cells.append((r, c))
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    nr, nc = r + dr, c + dc
                    if (dr or dc) and 0 <= nr < h and 0 <= nc < w and unseen[nr, nc]:
                        unseen[nr, nc] = False
                        todo.append((nr, nc))
        result.append(np.asarray(cells, dtype=int))
    return result


def _parse_objects(grid: np.ndarray, bg: int) -> tuple[list[dict], int]:
    """Identify body components and attach each red/yellow nozzle unambiguously."""
    body_colors = set(map(int, np.unique(grid))) - {bg, GREY, RED, YELLOW}
    objects: list[dict] = []
    owner = np.full(grid.shape, -1, dtype=int)
    for color in body_colors:
        for coords in _components8(grid == color):
            idx = len(objects)
            owner[coords[:, 0], coords[:, 1]] = idx
            objects.append({"color": color, "coords": coords, "nozzle": None, "direction": None})
    assert objects, "expected colored body blocks"

    marker_owner: dict[tuple[int, int], int] = {}
    for rr, cc in np.argwhere(np.isin(grid, (RED, YELLOW))):
        r, c = int(rr), int(cc)
        adjacent = []
        for dr, dc in DIR4:
            nr, nc = r - dr, c - dc
            if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1] and owner[nr, nc] >= 0:
                adjacent.append((int(owner[nr, nc]), (dr, dc)))
        adjacent = list(dict.fromkeys(adjacent))
        assert len(adjacent) == 1, "each nozzle must touch exactly one body in one direction"
        idx, direction = adjacent[0]
        assert objects[idx]["nozzle"] is None, "each body must have exactly one nozzle"
        objects[idx]["nozzle"] = (r, c)
        objects[idx]["direction"] = direction
        marker_owner[(r, c)] = idx

    assert all(obj["nozzle"] is not None for obj in objects), "each body must have a nozzle"
    red = [marker_owner[tuple(map(int, p))] for p in np.argwhere(grid == RED)]
    assert len(red) == 1, "expected one red ignition nozzle"
    return objects, red[0]


def main(grid: np.ndarray) -> np.ndarray:
    """Propagate count-limited block-colored rays through a divider as a firing chain.

    Input structure:
    - A dominant background contains one complete grey divider row or column.
    - On both sides are monochrome 8-connected body blocks, each with exactly one
      orthogonally attached nozzle: one red ignition nozzle and otherwise yellow nozzles.

    Rule:
    - The red-nozzle block fires first, away from its body, in its body color for at
      most as many cells as its body contains (the nozzle is not counted).
    - A ray recolors background and grey-divider cells, consuming one unit per cell.
      On first touching another block or its nozzle it stops without overwriting it
      and queues that block to fire; blocks never reached by this chain stay unchanged.

    Key insight:
    - A nozzle gives direction but not permission to fire: reachability through the
      chain gives permission, while body pixel count alone fixes each ray's range.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    values, counts = np.unique(g, return_counts=True)
    bg = int(values[int(np.argmax(counts))])
    assert bg not in {GREY, RED, YELLOW}, "background must differ from structural colors"

    full_rows = [r for r in range(g.shape[0]) if np.all(g[r, :] == GREY)]
    full_cols = [c for c in range(g.shape[1]) if np.all(g[:, c] == GREY)]
    assert len(full_rows) + len(full_cols) == 1, "expected exactly one complete grey divider"

    objects, ignition = _parse_objects(g, bg)
    cell_owner: dict[tuple[int, int], int] = {}
    for idx, obj in enumerate(objects):
        for r, c in obj["coords"]:
            cell_owner[(int(r), int(c))] = idx
        cell_owner[obj["nozzle"]] = idx

    out = g.copy()
    fired: set[int] = set()
    queued = {ignition}
    queue = deque([ignition])
    while queue:
        idx = queue.popleft()
        queued.discard(idx)
        if idx in fired:
            continue
        fired.add(idx)
        obj = objects[idx]
        r, c = obj["nozzle"]
        dr, dc = obj["direction"]
        length = int(len(obj["coords"]))
        for step in range(1, length + 1):
            nr, nc = r + step * dr, c + step * dc
            if not (0 <= nr < g.shape[0] and 0 <= nc < g.shape[1]):
                break
            hit = cell_owner.get((nr, nc))
            if hit is not None:
                if hit != idx and hit not in fired and hit not in queued:
                    queue.append(hit)
                    queued.add(hit)
                break
            assert int(g[nr, nc]) in {bg, GREY}, "a ray may only cross background or the divider"
            out[nr, nc] = int(obj["color"])
    return out


def _paint_rect(grid: np.ndarray, r0: int, c0: int, h: int, w: int, color: int) -> list[tuple[int, int]]:
    grid[r0 : r0 + h, c0 : c0 + w] = color
    return [(r, c) for r in range(r0, r0 + h) for c in range(c0, c0 + w)]


def _candidate() -> tuple[np.ndarray, np.ndarray]:
    """Build one sample plus an independent construction-time expected output."""
    wall = random.randint(9, 12)
    h = random.randint(wall + 14, 30)
    w = random.randint(24, 30)
    grid = np.full((h, w), BG, dtype=int)
    grid[wall, :] = GREY
    colors = random.sample(BODY_PALETTE, 4)

    # A fires downward across the divider. Its area is exactly (or one beyond)
    # the distance to B, so counting affects a successful breach as well.
    ah, aw = 2, random.choice((2, 3))
    a_len = ah * aw
    b_top = wall + 2
    a_distance = a_len - random.choice((0, 1))
    a_bottom = b_top - 1 - a_distance

    bh, bw = 2, random.choice((3, 4))
    b_len = bh * bw
    direction = random.choice((-1, 1))
    c_h, c_w = 2, random.choice((2, 3))
    c_len = c_h * c_w
    horizontal_distance = b_len - random.choice((0, 1))

    if direction == 1:
        max_b0 = w - (bw + 1 + horizontal_distance + c_w + 2)
        b0 = random.randint(3, max_b0)
        c0 = b0 + bw + horizontal_distance
        b_nozzle_c = b0 + bw
        b_dir = (0, 1)
    else:
        min_b0 = 2 + c_w + horizontal_distance
        b0 = random.randint(min_b0, w - bw - 3)
        c0 = b0 - 1 - horizontal_distance - (c_w - 1)
        b_nozzle_c = b0 - 1
        b_dir = (0, -1)

    a_col = random.randint(b0, b0 + bw - 1)
    a0 = max(1, min(a_col - aw // 2, w - aw - 2))
    # Keep the A nozzle aligned to B even when its rectangle is shifted at an edge.
    a_nozzle_col = a_col
    a_cells = _paint_rect(grid, a_bottom - ah + 1, a0, ah, aw, colors[0])
    assert any(c == a_nozzle_col for _r, c in a_cells)
    a_nozzle = (a_bottom + 1, a_nozzle_col)
    grid[a_nozzle] = RED

    b_cells = _paint_rect(grid, b_top, b0, bh, bw, colors[1])
    b_nozzle = (b_top + 1, b_nozzle_c)
    grid[b_nozzle] = YELLOW

    c_cells = _paint_rect(grid, b_top, c0, c_h, c_w, colors[2])
    c_nozzle = (b_top + c_h, c0 + c_w // 2)
    grid[c_nozzle] = YELLOW

    # D is exactly one cell beyond C's count-limited ray and therefore stays inert.
    d_top = c_nozzle[0] + c_len + 1
    d0 = max(1, min(c_nozzle[1] - 1, w - 4))
    d_cells = _paint_rect(grid, d_top, d0, 2, 2, colors[3])
    d_nozzle = (d_top + 2, d0)
    grid[d_nozzle] = YELLOW

    expected = grid.copy()
    # Independent oracle: paint only known open cells; target body cells stop rays.
    for step in range(1, a_distance):
        expected[a_nozzle[0] + step, a_nozzle[1]] = colors[0]
    for step in range(1, horizontal_distance):
        expected[b_nozzle[0] + step * b_dir[0], b_nozzle[1] + step * b_dir[1]] = colors[1]
    for step in range(1, c_len + 1):
        expected[c_nozzle[0] + step, c_nozzle[1]] = colors[2]

    # Semantic checks independent of main: wall breach, two triggered turns, and
    # the count-limit counterexample immediately before the untouched fourth body.
    assert expected[wall, a_nozzle_col] == colors[0]
    assert expected[b_nozzle] == YELLOW and expected[c_nozzle] == YELLOW
    assert all(expected[p] == grid[p] for p in d_cells + [d_nozzle])
    assert (c_nozzle[0] + c_len + 1, c_nozzle[1]) in set(d_cells)

    # Randomly exchange the row/column reference frame, covering both divider forms.
    if random.random() < 0.5:
        grid = grid.T.copy()
        expected = expected.T.copy()
    return grid, expected


def generate_input() -> np.ndarray:
    """Generate diverse, solver-checked instances with all essential branches visible."""
    for _ in range(300):
        try:
            grid, expected = _candidate()
            actual = main(grid)
        except (AssertionError, ValueError):
            continue
        if np.array_equal(actual, expected):
            return grid
    raise RuntimeError("failed to generate a valid chain-ignition input")
