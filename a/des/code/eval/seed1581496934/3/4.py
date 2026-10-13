from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q: deque[tuple[int, int]] = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            out.append(comp)
    return out


def _majority_color(values: list[int]) -> int:
    return Counter(values).most_common(1)[0][0]


def _draw_axis_line(grid: np.ndarray, a: tuple[int, int], b: tuple[int, int], color: int) -> None:
    ar, ac = a
    br, bc = b
    if ar == br:
        step = 1 if ac <= bc else -1
        for c in range(ac, bc + step, step):
            grid[ar, c] = color
        return
    if ac == bc:
        step = 1 if ar <= br else -1
        for r in range(ar, br + step, step):
            grid[r, ac] = color
        return
    raise AssertionError("paired ports must share a row or a column")


def _exterior_background(grid: np.ndarray, bg: int) -> np.ndarray:
    h, w = grid.shape
    outside = np.zeros((h, w), dtype=bool)
    q: deque[tuple[int, int]] = deque()

    def enqueue(r: int, c: int) -> None:
        if grid[r, c] == bg and not outside[r, c]:
            outside[r, c] = True
            q.append((r, c))

    for r in range(h):
        enqueue(r, 0)
        enqueue(r, w - 1)
    for c in range(w):
        enqueue(0, c)
        enqueue(h - 1, c)

    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and grid[nr, nc] == bg and not outside[nr, nc]:
                outside[nr, nc] = True
                q.append((nr, nc))
    return outside


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pair same-colored ports, draw straight connectors, then fill each enclosed grey region
    with the most frequent main color among the pipe fragments bordering that loop.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape

    bg = _majority_color([int(v) for v in g.ravel().tolist()])
    piece_mask = g != bg
    pieces_raw = _components(piece_mask)
    assert pieces_raw, "expected at least one pipe fragment"

    piece_id = np.full((h, w), -1, dtype=int)
    pieces: list[dict[str, object]] = []
    port_pairs: dict[int, list[tuple[int, tuple[int, int]]]] = {}

    for pid, comp in enumerate(pieces_raw):
        colors = [int(g[r, c]) for r, c in comp]
        main_color = _majority_color(colors)
        markers = [(r, c, int(g[r, c])) for r, c in comp if int(g[r, c]) != main_color]
        marker_counts = Counter(color for _r, _c, color in markers)
        assert len(markers) == 2, "each fragment must expose exactly two ports"
        assert len(marker_counts) == 2 and all(count == 1 for count in marker_counts.values())

        ports: dict[int, tuple[int, int]] = {}
        for r, c, color in markers:
            ports[color] = (r, c)
        for r, c in comp:
            piece_id[r, c] = pid

        pieces.append({"main_color": main_color, "ports": ports})
        for color, coord in ports.items():
            port_pairs.setdefault(color, []).append((pid, coord))

    out = g.copy()
    for color, endpoints in port_pairs.items():
        assert len(endpoints) == 2, "each port color must appear exactly twice"
        assert endpoints[0][0] != endpoints[1][0], "a port pair must connect two different fragments"
        _draw_axis_line(out, endpoints[0][1], endpoints[1][1], int(color))

    inside_mask = (out == bg) & ~_exterior_background(out, bg)
    for region in _components(inside_mask):
        border_piece_ids: set[int] = set()
        for r, c in region:
            for nr, nc in _neighbors4(r, c):
                if 0 <= nr < h and 0 <= nc < w and piece_id[nr, nc] != -1:
                    border_piece_ids.add(int(piece_id[nr, nc]))
        if not border_piece_ids:
            continue
        fill_color = Counter(int(pieces[pid]["main_color"]) for pid in border_piece_ids).most_common(1)[0][0]
        for r, c in region:
            out[r, c] = fill_color

    return out


BG = 5
PALETTE = [0, 1, 2, 3, 4, 6, 7, 8, 9]
SIDES = ("top", "right", "bottom", "left")
GEN_ATTEMPTS = 400
LOOP_ATTEMPTS = 300
PLACE_ATTEMPTS = 200


def _piece_from_cells(
    cells: set[tuple[int, int]],
    ports: dict[str, tuple[int, int]],
) -> dict[str, object]:
    return {
        "cells": sorted(cells),
        "ports": dict(ports),
        "port_colors": {},
        "main_color": None,
    }


def _elbow_tl(h_len: int, v_len: int) -> dict[str, object]:
    cells = {(r, c) for r in range(3) for c in range(h_len)}
    cells |= {(r, c) for r in range(v_len) for c in range(3)}
    ports = {"right": (1, h_len - 1), "down": (v_len - 1, 1)}
    return _piece_from_cells(cells, ports)


def _elbow_tr(width: int, h_len: int, v_len: int) -> dict[str, object]:
    left = width - h_len
    cells = {(r, c) for r in range(3) for c in range(left, width)}
    cells |= {(r, c) for r in range(v_len) for c in range(width - 3, width)}
    ports = {"left": (1, left), "down": (v_len - 1, width - 2)}
    return _piece_from_cells(cells, ports)


def _elbow_br(height: int, width: int, h_len: int, v_len: int) -> dict[str, object]:
    top = height - v_len
    left = width - h_len
    cells = {(r, c) for r in range(height - 3, height) for c in range(left, width)}
    cells |= {(r, c) for r in range(top, height) for c in range(width - 3, width)}
    ports = {"left": (height - 2, left), "up": (top, width - 2)}
    return _piece_from_cells(cells, ports)


def _elbow_bl(height: int, h_len: int, v_len: int) -> dict[str, object]:
    top = height - v_len
    cells = {(r, c) for r in range(height - 3, height) for c in range(h_len)}
    cells |= {(r, c) for r in range(top, height) for c in range(3)}
    ports = {"right": (height - 2, h_len - 1), "up": (top, 1)}
    return _piece_from_cells(cells, ports)


def _straight_h(row0: int, col0: int, length: int) -> dict[str, object]:
    cells = {(r, c) for r in range(row0, row0 + 3) for c in range(col0, col0 + length)}
    ports = {"left": (row0 + 1, col0), "right": (row0 + 1, col0 + length - 1)}
    return _piece_from_cells(cells, ports)


def _straight_v(row0: int, col0: int, length: int) -> dict[str, object]:
    cells = {(r, c) for r in range(row0, row0 + length) for c in range(col0, col0 + 3)}
    ports = {"up": (row0, col0 + 1), "down": (row0 + length - 1, col0 + 1)}
    return _piece_from_cells(cells, ports)


def _pieces_are_separated(pieces: dict[str, dict[str, object]]) -> bool:
    owners: dict[tuple[int, int], str] = {}
    for name, piece in pieces.items():
        for cell in piece["cells"]:
            if cell in owners:
                return False
            owners[cell] = name
    for (r, c), name in owners.items():
        for nr, nc in _neighbors4(r, c):
            other = owners.get((nr, nc))
            if other is not None and other != name:
                return False
    return True


def _build_loop(split_count: int, size_min: int, size_max: int) -> dict[str, object] | None:
    split_sides = set(random.sample(SIDES, split_count))
    for _ in range(LOOP_ATTEMPTS):
        height = random.randint(size_min, size_max)
        width = random.randint(size_min, size_max)

        max_h_arm = min(6, width - 6)
        max_v_arm = min(6, height - 6)
        if max_h_arm < 4 or max_v_arm < 4:
            continue

        tl_h = random.randint(4, max_h_arm)
        tr_h = random.randint(4, max_h_arm)
        br_h = random.randint(4, max_h_arm)
        bl_h = random.randint(4, max_h_arm)
        tl_v = random.randint(4, max_v_arm)
        tr_v = random.randint(4, max_v_arm)
        br_v = random.randint(4, max_v_arm)
        bl_v = random.randint(4, max_v_arm)

        free = {
            "top": width - tl_h - tr_h,
            "bottom": width - bl_h - br_h,
            "left": height - tl_v - bl_v,
            "right": height - tr_v - br_v,
        }
        need = {side: 6 if side in split_sides else 2 for side in SIDES}
        if any(free[side] < need[side] for side in SIDES):
            continue

        pieces: dict[str, dict[str, object]] = {
            "tl": _elbow_tl(tl_h, tl_v),
            "tr": _elbow_tr(width, tr_h, tr_v),
            "br": _elbow_br(height, width, br_h, br_v),
            "bl": _elbow_bl(height, bl_h, bl_v),
        }
        connections: list[dict[str, str]] = []

        def connect(a_piece: str, a_port: str, b_piece: str, b_port: str) -> None:
            connections.append(
                {
                    "a_piece": a_piece,
                    "a_port": a_port,
                    "b_piece": b_piece,
                    "b_port": b_port,
                }
            )

        if "top" in split_sides:
            max_len = min(8, free["top"] - 2)
            if max_len < 4:
                continue
            length = random.randint(4, max_len)
            left_port = int(pieces["tl"]["ports"]["right"][1])
            right_port = int(pieces["tr"]["ports"]["left"][1])
            start_min = left_port + 2
            start_max = right_port - length - 1
            if start_min > start_max:
                continue
            start = random.randint(start_min, start_max)
            pieces["top"] = _straight_h(0, start, length)
            connect("tl", "right", "top", "left")
            connect("top", "right", "tr", "left")
        else:
            connect("tl", "right", "tr", "left")

        if "right" in split_sides:
            max_len = min(8, free["right"] - 2)
            if max_len < 4:
                continue
            length = random.randint(4, max_len)
            top_port = int(pieces["tr"]["ports"]["down"][0])
            bottom_port = int(pieces["br"]["ports"]["up"][0])
            start_min = top_port + 2
            start_max = bottom_port - length - 1
            if start_min > start_max:
                continue
            start = random.randint(start_min, start_max)
            pieces["right"] = _straight_v(start, width - 3, length)
            connect("tr", "down", "right", "up")
            connect("right", "down", "br", "up")
        else:
            connect("tr", "down", "br", "up")

        if "bottom" in split_sides:
            max_len = min(8, free["bottom"] - 2)
            if max_len < 4:
                continue
            length = random.randint(4, max_len)
            left_port = int(pieces["bl"]["ports"]["right"][1])
            right_port = int(pieces["br"]["ports"]["left"][1])
            start_min = left_port + 2
            start_max = right_port - length - 1
            if start_min > start_max:
                continue
            start = random.randint(start_min, start_max)
            pieces["bottom"] = _straight_h(height - 3, start, length)
            connect("bl", "right", "bottom", "left")
            connect("bottom", "right", "br", "left")
        else:
            connect("bl", "right", "br", "left")

        if "left" in split_sides:
            max_len = min(8, free["left"] - 2)
            if max_len < 4:
                continue
            length = random.randint(4, max_len)
            top_port = int(pieces["tl"]["ports"]["down"][0])
            bottom_port = int(pieces["bl"]["ports"]["up"][0])
            start_min = top_port + 2
            start_max = bottom_port - length - 1
            if start_min > start_max:
                continue
            start = random.randint(start_min, start_max)
            pieces["left"] = _straight_v(start, 0, length)
            connect("tl", "down", "left", "up")
            connect("left", "down", "bl", "up")
        else:
            connect("tl", "down", "bl", "up")

        if not _pieces_are_separated(pieces):
            continue

        return {
            "height": height,
            "width": width,
            "pieces": pieces,
            "connections": connections,
        }
    return None


def _assign_single_loop_colors(loop: dict[str, object]) -> bool:
    piece_names = list(loop["pieces"].keys())
    dominant, accent = random.sample(PALETTE, 2)
    max_accent = min(2, (len(piece_names) - 1) // 2)
    if max_accent < 1:
        return False
    accent_count = random.randint(1, max_accent)
    accent_pieces = set(random.sample(piece_names, accent_count))
    for name, piece in loop["pieces"].items():
        piece["main_color"] = accent if name in accent_pieces else dominant
        piece["port_colors"] = {}

    connector_colors = [color for color in PALETTE if color not in {dominant, accent}]
    if len(connector_colors) < len(loop["connections"]):
        return False
    random.shuffle(connector_colors)
    for color, connection in zip(connector_colors, loop["connections"], strict=False):
        a_piece = loop["pieces"][connection["a_piece"]]
        b_piece = loop["pieces"][connection["b_piece"]]
        a_piece["port_colors"][connection["a_port"]] = color
        b_piece["port_colors"][connection["b_port"]] = color
    return True


def _assign_double_loop_colors(loops: list[dict[str, object]]) -> bool:
    all_connections = sum(len(loop["connections"]) for loop in loops)
    if all_connections != 8:
        return False
    main_color = random.choice(PALETTE)
    connector_colors = [color for color in PALETTE if color != main_color]
    random.shuffle(connector_colors)
    idx = 0
    for loop in loops:
        for piece in loop["pieces"].values():
            piece["main_color"] = main_color
            piece["port_colors"] = {}
        for connection in loop["connections"]:
            color = connector_colors[idx]
            idx += 1
            a_piece = loop["pieces"][connection["a_piece"]]
            b_piece = loop["pieces"][connection["b_piece"]]
            a_piece["port_colors"][connection["a_port"]] = color
            b_piece["port_colors"][connection["b_port"]] = color
    return True


def _boxes_too_close(
    a: tuple[int, int, int, int],
    b: tuple[int, int, int, int],
    margin: int,
) -> bool:
    return not (
        a[2] + margin < b[0]
        or b[2] + margin < a[0]
        or a[3] + margin < b[1]
        or b[3] + margin < a[1]
    )


def _choose_canvas(loops: list[dict[str, object]]) -> tuple[int, int]:
    heights = [int(loop["height"]) for loop in loops]
    widths = [int(loop["width"]) for loop in loops]
    if len(loops) == 1:
        min_h = max(18, heights[0] + 4)
        min_w = max(18, widths[0] + 4)
        max_h = min(28, min_h + 6)
        max_w = min(30, min_w + 8)
    else:
        if random.random() < 0.5:
            min_h = max(20, max(heights) + 4)
            min_w = max(24, sum(widths) + 6)
        else:
            min_h = max(24, sum(heights) + 6)
            min_w = max(20, max(widths) + 4)
        max_h = min(30, min_h + 4)
        max_w = min(30, min_w + 4)
    return random.randint(min_h, max_h), random.randint(min_w, max_w)


def _place_loops(loops: list[dict[str, object]], grid_h: int, grid_w: int) -> list[tuple[int, int]] | None:
    placements: list[tuple[int, int]] = []
    boxes: list[tuple[int, int, int, int]] = []
    order = list(range(len(loops)))
    random.shuffle(order)
    by_index: dict[int, tuple[int, int]] = {}

    for idx in order:
        loop = loops[idx]
        height = int(loop["height"])
        width = int(loop["width"])
        candidates: list[tuple[int, int]] = []
        for r0 in range(1, grid_h - height):
            for c0 in range(1, grid_w - width):
                box = (r0, c0, r0 + height - 1, c0 + width - 1)
                if any(_boxes_too_close(box, other, margin=2) for other in boxes):
                    continue
                candidates.append((r0, c0))
        if not candidates:
            return None
        pos = random.choice(candidates)
        by_index[idx] = pos
        boxes.append((pos[0], pos[1], pos[0] + height - 1, pos[1] + width - 1))

    for idx in range(len(loops)):
        placements.append(by_index[idx])
    return placements


def _render_input(loops: list[dict[str, object]], placements: list[tuple[int, int]], grid_h: int, grid_w: int) -> np.ndarray:
    grid = np.full((grid_h, grid_w), BG, dtype=int)
    for loop, (r0, c0) in zip(loops, placements, strict=False):
        for piece in loop["pieces"].values():
            main_color = int(piece["main_color"])
            for r, c in piece["cells"]:
                grid[r0 + r, c0 + c] = main_color
        for piece in loop["pieces"].values():
            for port_name, color in piece["port_colors"].items():
                pr, pc = piece["ports"][port_name]
                grid[r0 + pr, c0 + pc] = int(color)
    return grid


def _build_candidate() -> np.ndarray | None:
    if random.random() < 0.5:
        split_count = random.choice([0, 1, 2])
        loop = _build_loop(split_count=split_count, size_min=12, size_max=18)
        if loop is None or not _assign_single_loop_colors(loop):
            return None
        loops = [loop]
    else:
        loop_a = _build_loop(split_count=0, size_min=10, size_max=12)
        loop_b = _build_loop(split_count=0, size_min=10, size_max=12)
        if loop_a is None or loop_b is None:
            return None
        loops = [loop_a, loop_b]
        if not _assign_double_loop_colors(loops):
            return None

    grid_h, grid_w = _choose_canvas(loops)
    for _ in range(PLACE_ATTEMPTS):
        placements = _place_loops(loops, grid_h, grid_w)
        if placements is None:
            return None
        grid = _render_input(loops, placements, grid_h, grid_w)
        try:
            solved = main(grid)
        except AssertionError:
            continue
        if np.array_equal(solved, grid):
            continue
        return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _build_candidate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
