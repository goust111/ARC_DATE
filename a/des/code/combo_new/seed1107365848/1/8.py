from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
ARROW_DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
PAIR_COUNT_CHOICES = (2, 3, 4)
GENERATE_RETRIES = 300
PAIR_RETRIES = 150
PLACEMENT_RETRIES = 250


def neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _find_components(grid: np.ndarray, bg: int = BG) -> list[dict[str, object]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict[str, object]] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == bg or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                coords.append((cr, cc))
                for nr, nc in neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append({"color": color, "coords": coords})
    return comps


def _component_graph(coords: list[tuple[int, int]]) -> dict[tuple[int, int], list[tuple[int, int]]]:
    coord_set = set(coords)
    graph: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for cell in coords:
        graph[cell] = [nbr for nbr in neighbors4(*cell) if nbr in coord_set]
    return graph


def _parse_arrow(coords: list[tuple[int, int]]) -> dict[str, object] | None:
    graph = _component_graph(coords)
    degree_hist = sorted(len(nbrs) for nbrs in graph.values())
    if degree_hist.count(3) != 1 or degree_hist.count(1) != 3:
        return None
    if any(d not in (1, 2, 3) for d in degree_hist):
        return None

    junction = next(cell for cell, nbrs in graph.items() if len(nbrs) == 3)
    jnbrs = graph[junction]
    rels = [(nr - junction[0], nc - junction[1]) for nr, nc in jnbrs]

    cap_dirs: tuple[tuple[int, int], tuple[int, int]] | None = None
    stem_step: tuple[int, int] | None = None
    for i in range(3):
        for j in range(i + 1, 3):
            if rels[i][0] == -rels[j][0] and rels[i][1] == -rels[j][1]:
                cap_dirs = (rels[i], rels[j])
                stem_step = next(rels[k] for k in range(3) if k not in (i, j))
                break
        if cap_dirs is not None:
            break
    if cap_dirs is None or stem_step is None:
        return None

    cap_cells = {(junction[0] + dr, junction[1] + dc) for dr, dc in cap_dirs}
    if any(len(graph[cell]) != 1 for cell in cap_cells):
        return None

    stem = [junction]
    prev = junction
    cur = (junction[0] + stem_step[0], junction[1] + stem_step[1])
    if cur not in graph:
        return None
    while True:
        stem.append(cur)
        next_cells = [nbr for nbr in graph[cur] if nbr != prev]
        if not next_cells:
            break
        if len(next_cells) != 1:
            return None
        nxt = next_cells[0]
        if (nxt[0] - cur[0], nxt[1] - cur[1]) != stem_step:
            return None
        prev, cur = cur, nxt

    tail = stem[-1]
    stem_len = len(stem)
    if len(graph[tail]) != 1 or not (2 <= stem_len <= 5):
        return None
    if len(coords) != stem_len + 2:
        return None

    direction = (-stem_step[0], -stem_step[1])
    return {
        "junction": junction,
        "tail": tail,
        "direction": direction,
        "length": stem_len,
        "coords": set(coords),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Project each small sprite along its adjacent T-headed arrow and recolor the copy to the arrow color.

    Input structure:
    - The background is black.
    - Foreground contains several small monochrome 4-connected source sprites.
    - Each source sprite 4-touches one arrow of a different color.
    - Each arrow is a straight 1-pixel-thick stem of length 2-5 with a 1-pixel-thick perpendicular cap at the far end, forming a T-shaped head.

    Rule:
    - For each arrow, read the pointing direction from the stem toward the T-head and read the translation distance from the stem length.
    - Match the arrow to the unique adjacent sprite touching the arrow tail.
    - Translate the whole sprite by that distance in the arrow direction.
    - Paint the translated copy in the arrow's color while keeping the original sprite and arrow unchanged.

    Key insight:
    - The arrow simultaneously encodes direction and distance, and the projected copy uses the arrow color rather than the source sprite color.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    comps = _find_components(g, bg=BG)
    coord_sets = [set(comp["coords"]) for comp in comps]

    coord_to_comp: dict[tuple[int, int], int] = {}
    for idx, comp in enumerate(comps):
        for cell in coord_sets[idx]:
            coord_to_comp[cell] = idx

    adjacent: list[set[int]] = [set() for _ in comps]
    for idx, comp in enumerate(comps):
        for r, c in comp["coords"]:
            for nbr in neighbors4(r, c):
                j = coord_to_comp.get(nbr)
                if j is not None and j != idx:
                    adjacent[idx].add(j)

    out = g.copy()
    for arrow_idx, comp in enumerate(comps):
        arrow = _parse_arrow(comp["coords"])
        if arrow is None:
            continue

        tail = arrow["tail"]
        partners = sorted(
            j
            for j in adjacent[arrow_idx]
            if any(nbr in coord_sets[j] for nbr in neighbors4(*tail))
        )
        if len(partners) != 1:
            continue

        sprite = comps[partners[0]]
        dr, dc = arrow["direction"]
        dist = int(arrow["length"])
        arrow_color = int(comp["color"])
        for r, c in sprite["coords"]:
            rr = r + dr * dist
            cc = c + dc * dist
            if 0 <= rr < h and 0 <= cc < w:
                out[rr, cc] = arrow_color

    return out


def _normalize(coords: set[tuple[int, int]]) -> set[tuple[int, int]]:
    min_r = min(r for r, _ in coords)
    min_c = min(c for _, c in coords)
    return {(r - min_r, c - min_c) for r, c in coords}


def _random_polyomino(area: int) -> set[tuple[int, int]]:
    cells = {(0, 0)}
    while len(cells) < area:
        frontier = []
        for r, c in cells:
            for nr, nc in neighbors4(r, c):
                if (nr, nc) not in cells:
                    frontier.append((nr, nc))
        cells.add(random.choice(frontier))
    return _normalize(cells)


def _attachment_options(shape: set[tuple[int, int]], direction: tuple[int, int]) -> list[tuple[int, int]]:
    dr, dc = direction
    options: list[tuple[int, int]] = []
    for r, c in shape:
        tail = (r + dr, c + dc)
        if tail in shape:
            continue
        contact_count = sum(1 for nbr in neighbors4(*tail) if nbr in shape)
        if contact_count == 1:
            options.append((r, c))
    return options


def _contact_count(a: set[tuple[int, int]], b: set[tuple[int, int]]) -> int:
    count = 0
    for r, c in a:
        for nbr in neighbors4(r, c):
            if nbr in b:
                count += 1
    return count


def _make_pair(direction: tuple[int, int], length: int) -> dict[str, object] | None:
    dr, dc = direction
    perp = (-dc, dr)
    cap_choices = [perp, (-perp[0], -perp[1])]

    for _ in range(PAIR_RETRIES):
        area = random.randint(3, 6)
        shape = _random_polyomino(area)
        if _parse_arrow(list(shape)) is not None:
            continue
        options = _attachment_options(shape, direction)
        if not options:
            continue
        random.shuffle(options)
        for attach in options:
            for cap_dir in random.sample(cap_choices, k=2):
                junction = (attach[0] + dr * length, attach[1] + dc * length)
                stem = {(attach[0] + dr * step, attach[1] + dc * step) for step in range(1, length + 1)}
                arrow = set(stem)
                arrow.add((junction[0] + cap_dir[0], junction[1] + cap_dir[1]))
                arrow.add((junction[0] - cap_dir[0], junction[1] - cap_dir[1]))
                if shape & arrow:
                    continue
                if _contact_count(shape, arrow) != 1:
                    continue

                shifted = {(r + dr * length, c + dc * length) for r, c in shape}
                if shape & shifted:
                    continue
                if not junction in shifted:
                    continue
                if len(shifted - arrow) < 1:
                    continue

                union = shape | arrow | shifted
                min_r = min(r for r, _ in union)
                min_c = min(c for _, c in union)
                offset_r, offset_c = -min_r, -min_c
                norm_shape = {(r + offset_r, c + offset_c) for r, c in shape}
                norm_arrow = {(r + offset_r, c + offset_c) for r, c in arrow}
                norm_copy = {(r + offset_r, c + offset_c) for r, c in shifted}
                norm_union = norm_shape | norm_arrow | norm_copy
                bbox_h = max(r for r, _ in norm_union) + 1
                bbox_w = max(c for _, c in norm_union) + 1
                return {
                    "shape": norm_shape,
                    "arrow": norm_arrow,
                    "copy": norm_copy,
                    "bbox": (bbox_h, bbox_w),
                }
    return None


def _padded_bbox(coords: set[tuple[int, int]], h: int, w: int) -> tuple[int, int, int, int]:
    rs = [r for r, _ in coords]
    cs = [c for _, c in coords]
    return (
        max(0, min(rs) - 1),
        max(0, min(cs) - 1),
        min(h - 1, max(rs) + 1),
        min(w - 1, max(cs) + 1),
    )


def _place_pair(
    grid: np.ndarray,
    expected: np.ndarray,
    reserved: np.ndarray,
    pair: dict[str, object],
    pattern_color: int,
    arrow_color: int,
) -> bool:
    h, w = grid.shape
    pair_h, pair_w = pair["bbox"]
    if pair_h > h or pair_w > w:
        return False

    for _ in range(PLACEMENT_RETRIES):
        top = random.randint(0, h - pair_h)
        left = random.randint(0, w - pair_w)
        union = {(top + r, left + c) for r, c in (pair["shape"] | pair["arrow"] | pair["copy"])}
        r0, c0, r1, c1 = _padded_bbox(union, h, w)
        if reserved[r0 : r1 + 1, c0 : c1 + 1].any():
            continue

        for r, c in pair["shape"]:
            grid[top + r, left + c] = pattern_color
            expected[top + r, left + c] = pattern_color
        for r, c in pair["arrow"]:
            grid[top + r, left + c] = arrow_color
            expected[top + r, left + c] = arrow_color
        for r, c in pair["copy"]:
            expected[top + r, left + c] = arrow_color

        reserved[r0 : r1 + 1, c0 : c1 + 1] = True
        return True
    return False


def _gen_attempt() -> np.ndarray | None:
    pair_count = random.choice(PAIR_COUNT_CHOICES)
    min_side = 14 + 3 * pair_count
    h = random.randint(min_side, 30)
    w = random.randint(min_side, 30)

    directions = random.sample(ARROW_DIRS, k=pair_count)
    lengths = random.sample([2, 3, 4, 5], k=pair_count)
    palette = random.sample(range(1, 10), k=pair_count * 2)
    pattern_colors = palette[:pair_count]
    arrow_colors = palette[pair_count:]

    pairs: list[dict[str, object]] = []
    for direction, length in zip(directions, lengths):
        pair = _make_pair(direction, length)
        if pair is None:
            return None
        pairs.append(pair)

    order = sorted(range(pair_count), key=lambda i: -(pairs[i]["bbox"][0] * pairs[i]["bbox"][1]))

    grid = np.full((h, w), BG, dtype=int)
    expected = np.full((h, w), BG, dtype=int)
    reserved = np.zeros((h, w), dtype=bool)

    for idx in order:
        if not _place_pair(grid, expected, reserved, pairs[idx], pattern_colors[idx], arrow_colors[idx]):
            return None

    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATE_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GENERATE_RETRIES} attempts")
