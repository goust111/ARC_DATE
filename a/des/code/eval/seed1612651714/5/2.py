import random
from collections import deque

import numpy as np


GREY = 5
MAROON = 9

RED = 2
BLUE = 1
GREEN = 3
YELLOW = 4
ORANGE = 7

SOURCE_COLORS = [RED, BLUE, GREEN, YELLOW, ORANGE]
PRIORITY = {
    ORANGE: 0,
    YELLOW: 1,
    GREEN: 2,
    BLUE: 3,
    RED: 4,
}

BG_CHOICES = [0, 6, 8]
GEN_RETRIES = 300


def _neighbors4(r: int, c: int, h: int, w: int):
    if r > 0:
        yield r - 1, c
    if r + 1 < h:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < w:
        yield r, c + 1


def _most_common_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _mix_colors(a: int, b: int) -> int:
    return int(a) if PRIORITY[int(a)] >= PRIORITY[int(b)] else int(b)


def _same_color_component(grid: np.ndarray, start: tuple[int, int], visited: np.ndarray) -> list[tuple[int, int]]:
    h, w = grid.shape
    color = int(grid[start])
    q = deque([start])
    visited[start] = True
    comp: list[tuple[int, int]] = []
    while q:
        r, c = q.popleft()
        comp.append((r, c))
        for nr, nc in _neighbors4(r, c, h, w):
            if visited[nr, nc] or int(grid[nr, nc]) != color:
                continue
            visited[nr, nc] = True
            q.append((nr, nc))
    return comp


def _find_sources(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    visited = np.zeros((h, w), dtype=bool)
    sources: list[dict] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if visited[r, c] or color in (bg, GREY, MAROON):
                continue
            comp = _same_color_component(grid, (r, c), visited)
            if len(comp) != 4:
                continue
            rs = [rr for rr, _ in comp]
            cs = [cc for _, cc in comp]
            if max(rs) - min(rs) != 1 or max(cs) - min(cs) != 1:
                continue
            cells = set(comp)
            entries = set()
            for rr, cc in comp:
                for nr, nc in _neighbors4(rr, cc, h, w):
                    if (nr, nc) in cells:
                        continue
                    if int(grid[nr, nc]) in (GREY, MAROON):
                        entries.add((nr, nc))
            if len(entries) != 1:
                continue
            sources.append({"color": color, "cells": cells, "entry": next(iter(entries))})
    return sources


def _find_containers(grid: np.ndarray) -> tuple[list[dict], np.ndarray]:
    h, w = grid.shape
    used = np.zeros((h, w), dtype=bool)
    containers: list[dict] = []
    mask = np.zeros((h, w), dtype=bool)

    for r in range(h - 1):
        for c in range(w - 1):
            cells = [(r, c), (r + 1, c), (r, c + 1), (r + 1, c + 1)]
            if any(int(grid[rr, cc]) != GREY for rr, cc in cells):
                continue
            if any(used[rr, cc] for rr, cc in cells):
                continue
            block = set(cells)
            entries = set()
            for rr, cc in cells:
                for nr, nc in _neighbors4(rr, cc, h, w):
                    if (nr, nc) in block:
                        continue
                    if int(grid[nr, nc]) in (GREY, MAROON):
                        entries.add((nr, nc))
            if len(entries) != 1:
                continue
            entry = next(iter(entries))
            containers.append({"cells": block, "entry": entry})
            for rr, cc in cells:
                used[rr, cc] = True
                mask[rr, cc] = True
    return containers, mask


def _network_neighbors(mask: np.ndarray) -> dict[tuple[int, int], list[tuple[int, int]]]:
    h, w = mask.shape
    neighbors: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for r in range(h):
        for c in range(w):
            if not mask[r, c]:
                continue
            pos = (r, c)
            neighbors[pos] = [(nr, nc) for nr, nc in _neighbors4(r, c, h, w) if mask[nr, nc]]
    return neighbors


def main(grid: np.ndarray) -> np.ndarray:
    """
    Propagate source colors through grey pipes and resolve maroon mixing nodes.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = _most_common_color(g)

    sources = _find_sources(g, bg)
    containers, container_mask = _find_containers(g)
    network_mask = ((g == GREY) & (~container_mask)) | (g == MAROON)
    neighbors = _network_neighbors(network_mask)

    node_positions = {tuple(p) for p in np.argwhere(g == MAROON)}
    container_by_entry = {tuple(cont["entry"]): cont for cont in containers}

    out = g.copy()
    arrivals: dict[tuple[int, int], dict[tuple[int, int], int]] = {}
    fired: set[tuple[int, int]] = set()
    q = deque()

    for src in sources:
        entry = tuple(src["entry"])
        if entry in neighbors:
            q.append((entry, None, int(src["color"])))

    while q:
        curr, prev, color = q.popleft()

        while True:
            if curr in node_positions:
                if curr in fired:
                    break
                incoming = arrivals.setdefault(curr, {})
                if prev is None or prev not in neighbors.get(curr, []):
                    break
                if prev in incoming:
                    break
                incoming[prev] = int(color)
                if len(incoming) < 2:
                    break
                mixed = max(incoming.values(), key=lambda x: PRIORITY[int(x)])
                fired.add(curr)
                for nxt in neighbors[curr]:
                    if nxt not in incoming:
                        q.append((nxt, curr, mixed))
                break

            if curr in container_by_entry:
                for rr, cc in container_by_entry[curr]["cells"]:
                    out[rr, cc] = int(color)
                break

            next_steps = [nxt for nxt in neighbors.get(curr, []) if nxt != prev]
            if not next_steps:
                break
            if len(next_steps) != 1:
                break
            prev, curr = curr, next_steps[0]

    return out


def _paint_rect(grid: np.ndarray, r0: int, c0: int, color: int) -> list[tuple[int, int]]:
    cells = []
    for dr in range(2):
        for dc in range(2):
            rr = r0 + dr
            cc = c0 + dc
            grid[rr, cc] = int(color)
            cells.append((rr, cc))
    return cells


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    bg = int(rng.choice(BG_CHOICES))
    node_count = rng.randint(1, 4)

    branch_flags = [bool(rng.getrandbits(1)) for _ in range(node_count)]
    if node_count >= 2 and (all(branch_flags) or not any(branch_flags)):
        return None

    side_dirs = [rng.choice([-1, 1]) for _ in range(node_count)]
    side_source_lengths = [rng.randint(2, 5) for _ in range(node_count)]
    side_container_lengths = [rng.randint(2, 5) if branch_flags[i] else 0 for i in range(node_count)]

    top_stem = rng.randint(2, 4)
    bottom_stem = rng.randint(2, 4)
    gaps = [rng.randint(4, 6) for _ in range(node_count - 1)]

    top_color = int(rng.choice(SOURCE_COLORS))
    side_colors = [int(rng.choice(SOURCE_COLORS)) for _ in range(node_count)]
    all_source_colors = [top_color] + side_colors
    if len(set(all_source_colors)) < 2:
        return None

    if node_count >= 2:
        upgrades = 0
        no_upgrades = 0
        running = top_color
        for color in side_colors:
            if PRIORITY[int(color)] > PRIORITY[int(running)]:
                upgrades += 1
            else:
                no_upgrades += 1
            running = _mix_colors(running, color)
        if upgrades == 0 or no_upgrades == 0:
            return None

    rows = [top_stem + 2]
    for gap in gaps:
        rows.append(rows[-1] + gap)

    left_need = 0
    right_need = 0
    for i in range(node_count):
        src_span = side_source_lengths[i] + 2
        if side_dirs[i] < 0:
            left_need = max(left_need, src_span)
        else:
            right_need = max(right_need, src_span)
        if branch_flags[i]:
            out_span = side_container_lengths[i] + 2
            if side_dirs[i] < 0:
                right_need = max(right_need, out_span)
            else:
                left_need = max(left_need, out_span)

    spine_c = left_need
    height = rows[-1] + bottom_stem + 3
    width = left_need + right_need + 2
    if height <= 0 or width <= 0:
        return None

    grid = np.full((height, width), bg, dtype=int)
    expected = grid.copy()

    for r in rows:
        grid[r, spine_c] = MAROON
        expected[r, spine_c] = MAROON

    top_entry_r = rows[0] - top_stem
    grid[top_entry_r:rows[0], spine_c] = GREY
    expected[top_entry_r:rows[0], spine_c] = GREY
    _paint_rect(grid, top_entry_r - 2, spine_c, top_color)
    _paint_rect(expected, top_entry_r - 2, spine_c, top_color)

    for i in range(node_count - 1):
        r0 = rows[i]
        r1 = rows[i + 1]
        grid[r0 + 1 : r1, spine_c] = GREY
        expected[r0 + 1 : r1, spine_c] = GREY

    bottom_entry_r = rows[-1] + bottom_stem
    grid[rows[-1] + 1 : bottom_entry_r + 1, spine_c] = GREY
    expected[rows[-1] + 1 : bottom_entry_r + 1, spine_c] = GREY
    bottom_container = _paint_rect(grid, bottom_entry_r + 1, spine_c, GREY)

    running = top_color
    branch_container_colors: list[tuple[list[tuple[int, int]], int]] = []

    for i, r in enumerate(rows):
        src_side = side_dirs[i]
        src_len = side_source_lengths[i]
        src_color = side_colors[i]

        if src_side < 0:
            entry_c = spine_c - src_len
            grid[r, entry_c:spine_c] = GREY
            expected[r, entry_c:spine_c] = GREY
            _paint_rect(grid, r, entry_c - 2, src_color)
            _paint_rect(expected, r, entry_c - 2, src_color)
        else:
            entry_c = spine_c + src_len
            grid[r, spine_c + 1 : entry_c + 1] = GREY
            expected[r, spine_c + 1 : entry_c + 1] = GREY
            _paint_rect(grid, r, entry_c + 1, src_color)
            _paint_rect(expected, r, entry_c + 1, src_color)

        running = _mix_colors(running, src_color)

        if branch_flags[i]:
            out_len = side_container_lengths[i]
            out_side = -src_side
            if out_side < 0:
                entry_c = spine_c - out_len
                grid[r, entry_c:spine_c] = GREY
                expected[r, entry_c:spine_c] = GREY
                cells = _paint_rect(grid, r, entry_c - 2, GREY)
            else:
                entry_c = spine_c + out_len
                grid[r, spine_c + 1 : entry_c + 1] = GREY
                expected[r, spine_c + 1 : entry_c + 1] = GREY
                cells = _paint_rect(grid, r, entry_c + 1, GREY)
            branch_container_colors.append((cells, running))

    for cells, color in branch_container_colors:
        for rr, cc in cells:
            expected[rr, cc] = int(color)
    for rr, cc in bottom_container:
        expected[rr, cc] = int(running)

    rot_k = rng.randrange(4)
    grid = np.rot90(grid, rot_k)
    expected = np.rot90(expected, rot_k)

    pad_top = rng.randint(2, 5)
    pad_bottom = rng.randint(2, 5)
    pad_left = rng.randint(2, 5)
    pad_right = rng.randint(2, 5)
    grid = np.pad(grid, ((pad_top, pad_bottom), (pad_left, pad_right)), constant_values=bg)
    expected = np.pad(expected, ((pad_top, pad_bottom), (pad_left, pad_right)), constant_values=bg)

    if _most_common_color(grid) != bg:
        return None

    try:
        produced = main(grid)
    except Exception:
        return None
    if produced.shape != grid.shape or not np.array_equal(produced, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: failed to build a valid pipe network")
