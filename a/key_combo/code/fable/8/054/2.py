from __future__ import annotations

import random
from collections import Counter, defaultdict, deque

import numpy as np


BG = 0
BODY = 5
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
CONNECTOR_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
CELL = 5
GEN_ATTEMPTS = 400


def _bbox_of_cells(cells: set[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _crop_non_bg(grid: np.ndarray) -> np.ndarray:
    coords = np.argwhere(grid != BG)
    if coords.size == 0:
        return np.zeros((0, 0), dtype=int)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return grid[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]


def _components8(mask: np.ndarray) -> list[set[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[set[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            queue = deque([(r, c)])
            comp: set[tuple[int, int]] = {(r, c)}
            while queue:
                cr, cc = queue.popleft()
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        if dr == 0 and dc == 0:
                            continue
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                            seen[nr, nc] = True
                            queue.append((nr, nc))
                            comp.add((nr, nc))
            comps.append(comp)
    return comps


def _is_connected4(cells: set[tuple[int, int]]) -> bool:
    if not cells:
        return False
    start = next(iter(cells))
    seen = {start}
    queue = deque([start])
    while queue:
        r, c = queue.popleft()
        for dr, dc in DIRS4:
            nxt = (r + dr, c + dc)
            if nxt in cells and nxt not in seen:
                seen.add(nxt)
                queue.append(nxt)
    return len(seen) == len(cells)


def _analyze_component(sprite: np.ndarray) -> dict:
    coords = {(int(r), int(c)) for r, c in np.argwhere(sprite != BG)}
    body = {(r, c) for r, c in coords if int(sprite[r, c]) == BODY}
    connectors = []
    for r, c in sorted(coords):
        color = int(sprite[r, c])
        if color in (BG, BODY):
            continue
        comp_neighbors = []
        body_neighbors = []
        for dr, dc in DIRS4:
            nr, nc = r + dr, c + dc
            if (nr, nc) in coords:
                comp_neighbors.append((nr, nc))
                if int(sprite[nr, nc]) == BODY:
                    body_neighbors.append((nr, nc))
        assert len(comp_neighbors) == 1, "connector must attach by exactly one 4-neighbor"
        assert len(body_neighbors) == 1, "connector must have exactly one gray 4-neighbor"
        br, bc = body_neighbors[0]
        connectors.append(
            {
                "color": color,
                "pos": (r, c),
                "dir": (r - br, c - bc),
            }
        )
    assert len(connectors) in (1, 2), "each component must have one or two connectors"
    assert _is_connected4(body), "gray body must be 4-connected"
    return {
        "grid": sprite.copy(),
        "cells": [(int(r), int(c), int(sprite[r, c])) for r, c in sorted(coords)],
        "connectors": connectors,
        "body_area": len(body),
    }


def _extract_components(grid: np.ndarray) -> list[dict]:
    comps = []
    for cells in _components8(grid != BG):
        r0, c0, r1, c1 = _bbox_of_cells(cells)
        sprite = grid[r0 : r1 + 1, c0 : c1 + 1].copy()
        comps.append(_analyze_component(sprite))
    return comps


def _build_chain_graph(components: list[dict]) -> tuple[list[list[dict]], list[int]]:
    by_color: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for comp_idx, comp in enumerate(components):
        for conn_idx, conn in enumerate(comp["connectors"]):
            by_color[int(conn["color"])].append((comp_idx, conn_idx))

    assert by_color, "expected connector colors"
    adjacency: list[list[dict]] = [[] for _ in components]
    for color, hits in by_color.items():
        assert len(hits) == 2, "each connector color must appear exactly twice"
        (a, ia), (b, ib) = hits
        assert a != b, "paired connectors must belong to different components"
        adjacency[a].append({"to": b, "self_idx": ia, "other_idx": ib, "color": color})
        adjacency[b].append({"to": a, "self_idx": ib, "other_idx": ia, "color": color})

    degrees = [len(edges) for edges in adjacency]
    assert degrees.count(1) == 2, "exactly two endpoint components are required"
    assert all(deg in (1, 2) for deg in degrees), "connector graph must be a simple path"

    start = degrees.index(1)
    order = [start]
    used = {start}
    prev = -1
    while True:
        cur = order[-1]
        nxt_edges = [edge for edge in adjacency[cur] if edge["to"] != prev]
        if not nxt_edges:
            break
        assert len(nxt_edges) == 1, "path order must be unique"
        edge = nxt_edges[0]
        nxt = int(edge["to"])
        assert nxt not in used, "connector graph must not cycle"
        used.add(nxt)
        order.append(nxt)
        prev = cur

    assert len(order) == len(components), "connector graph must be connected"
    return adjacency, order


def _solve_layout(components: list[dict]) -> dict[int, tuple[int, int]]:
    adjacency, order = _build_chain_graph(components)
    placements: dict[int, tuple[int, int]] = {order[0]: (0, 0)}
    for idx in range(len(order) - 1):
        cur = order[idx]
        nxt = order[idx + 1]
        edge = next(edge for edge in adjacency[cur] if edge["to"] == nxt)
        cur_conn = components[cur]["connectors"][edge["self_idx"]]
        nxt_conn = components[nxt]["connectors"][edge["other_idx"]]
        dr0, dc0 = cur_conn["dir"]
        dr1, dc1 = nxt_conn["dir"]
        assert (dr0, dc0) == (-dr1, -dc1), "paired connectors must face each other"
        base_r, base_c = placements[cur]
        target_r = base_r + int(cur_conn["pos"][0]) + int(dr0)
        target_c = base_c + int(cur_conn["pos"][1]) + int(dc0)
        placements[nxt] = (target_r - int(nxt_conn["pos"][0]), target_c - int(nxt_conn["pos"][1]))
    return placements


def _render_output(components: list[dict], placements: dict[int, tuple[int, int]]) -> np.ndarray:
    occupied: set[tuple[int, int]] = set()
    for comp_idx, comp in enumerate(components):
        tr, tc = placements[comp_idx]
        for r, c, _color in comp["cells"]:
            pos = (r + tr, c + tc)
            assert pos not in occupied, "assembled components must not overlap"
            occupied.add(pos)
    r0, c0, r1, c1 = _bbox_of_cells(occupied)
    out = np.full((r1 - r0 + 1, c1 - c0 + 1), BG, dtype=int)
    for r, c in occupied:
        out[r - r0, c - c0] = BODY
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Dock scattered gray components into the only color-matched chain and tight-crop the assembly.

    Input structure:
    - Black background contains several disconnected gray components.
    - Each component carries one or two non-gray connector pixels that stick out by one cell from the gray body.
    - Every connector color appears exactly twice in the whole grid, on two different components, and exactly two components have only one connector.

    Rule:
    - Treat same-colored connector pixels as a pair.
    - For each matched pair, translate one component until the two connector pixels become edge-adjacent and their outward directions are opposite.
    - Starting from either one-connector endpoint, the connector pairings form a unique chain through all components; assemble that chain, repaint every connector pixel gray, and tight-crop the non-background result.

    Key insight:
    - The colors tell you which interfaces mate, while the single gray neighbor of each connector tells you the only valid docking direction, so the full translation-only chain is uniquely determined.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    components = _extract_components(g)
    placements = _solve_layout(components)
    out = _render_output(components, placements)
    assert out.ndim == 2 and max(out.shape) <= 30
    return out


def _build_turning_walk(length: int) -> list[tuple[int, int]] | None:
    for _ in range(120):
        walk = [(0, 0)]
        used = {(0, 0)}
        last_dir = None
        while len(walk) < length:
            r, c = walk[-1]
            options = []
            for dr, dc in DIRS4:
                nxt = (r + dr, c + dc)
                if nxt not in used:
                    options.append(((dr, dc), nxt))
            random.shuffle(options)
            if last_dir is not None:
                options.sort(key=lambda item: item[0] == last_dir)
            if not options:
                break
            chosen_dir, chosen_pos = options[0]
            walk.append(chosen_pos)
            used.add(chosen_pos)
            last_dir = chosen_dir
        if len(walk) == length:
            dirs = [(walk[i + 1][0] - walk[i][0], walk[i + 1][1] - walk[i][1]) for i in range(length - 1)]
            if any(dirs[i] != dirs[i - 1] for i in range(1, len(dirs))):
                return walk
    return None


def _generate_walk(length: int) -> list[tuple[int, int]]:
    require_turn = length >= 4 and random.random() < 0.7
    if not require_turn and random.random() < 0.55:
        dr, dc = random.choice(DIRS4)
        return [(i * dr, i * dc) for i in range(length)]
    walk = _build_turning_walk(length)
    if walk is not None:
        return walk
    dr, dc = random.choice(DIRS4)
    return [(i * dr, i * dc) for i in range(length)]


def _port_support(direction: tuple[int, int]) -> tuple[int, int]:
    dr, dc = direction
    return 2 + dr, 2 + dc


def _port_connector(direction: tuple[int, int]) -> tuple[int, int]:
    dr, dc = direction
    return 2 + 2 * dr, 2 + 2 * dc


def _make_component(port_dirs: list[tuple[int, int]], port_colors: list[int]) -> np.ndarray:
    sprite = np.zeros((CELL, CELL), dtype=int)
    body = {(2, 2)}
    for direction in port_dirs:
        body.add(_port_support(direction))

    optional = [(r, c) for r in range(1, 4) for c in range(1, 4) if (r, c) not in body]
    random.shuffle(optional)
    extra_count = random.randint(0, min(5, len(optional)))
    for cell in optional[:extra_count]:
        body.add(cell)

    for r, c in body:
        sprite[r, c] = BODY
    for color, direction in zip(port_colors, port_dirs):
        pr, pc = _port_connector(direction)
        sprite[pr, pc] = int(color)
    return sprite


def _assembled_components() -> tuple[list[np.ndarray], np.ndarray]:
    count = random.randint(3, 5)
    walk = _generate_walk(count)
    edge_dirs = [(walk[i + 1][0] - walk[i][0], walk[i + 1][1] - walk[i][1]) for i in range(count - 1)]
    edge_colors = random.sample(CONNECTOR_COLORS, count - 1)

    components = []
    for idx in range(count):
        dirs = []
        colors = []
        if idx > 0:
            prev_dir = edge_dirs[idx - 1]
            dirs.append((-prev_dir[0], -prev_dir[1]))
            colors.append(edge_colors[idx - 1])
        if idx < count - 1:
            dirs.append(edge_dirs[idx])
            colors.append(edge_colors[idx])
        components.append(_make_component(dirs, colors))

    min_r = min(r for r, _ in walk)
    min_c = min(c for _, c in walk)
    max_r = max(r for r, _ in walk)
    max_c = max(c for _, c in walk)
    assembled = np.zeros(((max_r - min_r + 1) * CELL, (max_c - min_c + 1) * CELL), dtype=int)
    for sprite, (gr, gc) in zip(components, walk):
        top = (gr - min_r) * CELL
        left = (gc - min_c) * CELL
        mask = sprite != BG
        assembled[top : top + CELL, left : left + CELL][mask] = sprite[mask]

    expected = np.where(assembled != BG, BODY, BG)
    expected = _crop_non_bg(expected)
    return components, expected


def _sprite_cells(sprite: np.ndarray) -> list[tuple[int, int, int]]:
    return [(int(r), int(c), int(sprite[r, c])) for r, c in np.argwhere(sprite != BG)]


def _can_place(occupied: np.ndarray, cells: list[tuple[int, int, int]], top: int, left: int) -> bool:
    h, w = occupied.shape
    for r, c, _color in cells:
        ar, ac = top + r, left + c
        if not (0 <= ar < h and 0 <= ac < w):
            return False
        r0, r1 = max(0, ar - 1), min(h, ar + 2)
        c0, c1 = max(0, ac - 1), min(w, ac + 2)
        if occupied[r0:r1, c0:c1].any():
            return False
    return True


def _place_sprites(sprites: list[np.ndarray]) -> np.ndarray | None:
    max_h = max(sprite.shape[0] for sprite in sprites)
    max_w = max(sprite.shape[1] for sprite in sprites)
    min_dim = max(18, max(max_h, max_w) + 4)
    for _ in range(80):
        h = random.randint(min_dim, 30)
        w = random.randint(min_dim, 30)
        grid = np.zeros((h, w), dtype=int)
        occupied = np.zeros((h, w), dtype=bool)
        order = list(sprites)
        random.shuffle(order)
        ok = True
        for sprite in order:
            cells = _sprite_cells(sprite)
            sh, sw = sprite.shape
            positions = [(r, c) for r in range(h - sh + 1) for c in range(w - sw + 1)]
            random.shuffle(positions)
            placed = False
            for top, left in positions:
                if not _can_place(occupied, cells, top, left):
                    continue
                for r, c, color in cells:
                    grid[top + r, left + c] = color
                    occupied[top + r, left + c] = True
                placed = True
                break
            if not placed:
                ok = False
                break
        if ok:
            return grid
    return None


def _validate_instance(inp: np.ndarray, expected: np.ndarray) -> bool:
    if inp.ndim != 2 or expected.ndim != 2:
        return False
    if max(inp.shape) > 30 or max(expected.shape) > 30:
        return False
    try:
        components = _extract_components(inp)
        placements = _solve_layout(components)
        actual = _render_output(components, placements)
    except AssertionError:
        return False
    if not np.array_equal(actual, expected):
        return False
    try:
        if not np.array_equal(main(inp), expected):
            return False
    except AssertionError:
        return False
    if len(components) < 3:
        return False
    connector_counts = Counter()
    for comp in components:
        for conn in comp["connectors"]:
            connector_counts[int(conn["color"])] += 1
    if not connector_counts or any(v != 2 for v in connector_counts.values()):
        return False
    if expected.size == 0 or np.all(expected == BG):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    assembled, expected = _assembled_components()
    sprites = [_crop_non_bg(sprite) for sprite in assembled]
    random.shuffle(sprites)
    inp = _place_sprites(sprites)
    if inp is None:
        return None
    if not _validate_instance(inp, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid chain-docking sample")
