from __future__ import annotations

import random

import numpy as np


BLACK = 0
RED = 2
GREY = 5
AZURE = 8
NODE_COLORS = (1, 3, 4, 6, 7, 9)

GEN_RETRIES = 300
TOP_H_RANGE = (8, 13)
BOTTOM_H_RANGE = (10, 16)
GRID_W_MIN = 18
GRID_W_MAX = 30
GROUP_COUNT_RANGE = (2, 3)
EXTRA_REGION_RANGE = (0, 2)
TOP_GROUP_GAP_RANGE = (2, 4)
FRAME_INNER_H_RANGE = (2, 4)
FRAME_INNER_W_RANGE = (2, 5)
FRAME_PLACE_RETRIES = 120


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                cells.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(cells)
    return comps


def _bbox(cells: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _separator_row(grid: np.ndarray) -> int:
    candidates = [r for r in range(grid.shape[0]) if np.all(grid[r] == AZURE)]
    assert candidates, "expected a full-width azure separator row"
    return int(candidates[0])


def _top_color_to_dominant(top: np.ndarray) -> dict[int, int]:
    group_mask = top != BLACK
    group_cells = _components(group_mask)
    group_id = -np.ones(top.shape, dtype=int)
    for gid, cells in enumerate(group_cells):
        for r, c in cells:
            group_id[r, c] = gid

    node_mask = (top != BLACK) & (top != RED)
    node_cells = _components(node_mask)

    groups: dict[int, list[tuple[int, tuple[int, int], int]]] = {}
    color_to_group: dict[int, int] = {}
    for cells in node_cells:
        color = int(top[cells[0]])
        assert all(int(top[r, c]) == color for r, c in cells), "each node must be monochrome"
        gid = int(group_id[cells[0]])
        top_left = min(cells)
        groups.setdefault(gid, []).append((len(cells), top_left, color))
        assert color not in color_to_group, "node colors must be unique"
        color_to_group[color] = gid

    group_dominant: dict[int, int] = {}
    for gid, nodes in groups.items():
        area, top_left, color = sorted(nodes, key=lambda item: (-item[0], item[1][0], item[1][1]))[0]
        _ = area, top_left
        group_dominant[gid] = color

    return {color: group_dominant[gid] for color, gid in color_to_group.items()}


def _fill_bottom(bottom: np.ndarray, color_to_dominant: dict[int, int]) -> np.ndarray:
    out = bottom.copy()
    for cells in _components(bottom == GREY):
        r0, c0, r1, c1 = _bbox(cells)
        if r1 - r0 < 2 or c1 - c0 < 2:
            continue

        expected = 2 * ((r1 - r0 + 1) + (c1 - c0 + 1)) - 4
        if len(cells) != expected:
            continue
        if not np.all(bottom[r0, c0 : c1 + 1] == GREY):
            continue
        if not np.all(bottom[r1, c0 : c1 + 1] == GREY):
            continue
        if not np.all(bottom[r0 : r1 + 1, c0] == GREY):
            continue
        if not np.all(bottom[r0 : r1 + 1, c1] == GREY):
            continue

        inner = bottom[r0 + 1 : r1, c0 + 1 : c1]
        markers = np.argwhere((inner != BLACK) & (inner != GREY))
        assert len(markers) == 1, "each framed region must contain exactly one marker"
        mr, mc = map(int, markers[0])
        assert (mr, mc) in {
            (0, 0),
            (0, inner.shape[1] - 1),
            (inner.shape[0] - 1, 0),
            (inner.shape[0] - 1, inner.shape[1] - 1),
        }, "marker must sit in an interior corner"
        marker_color = int(inner[mr, mc])
        fill_color = int(color_to_dominant[marker_color])
        inner_out = out[r0 + 1 : r1, c0 + 1 : c1]
        inner_out[inner == BLACK] = fill_color
    return out


def _solve(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    sep = _separator_row(g)
    top = g[:sep]
    bottom = g[sep + 1 :]
    color_to_dominant = _top_color_to_dominant(top)
    out = g.copy()
    out[sep + 1 :] = _fill_bottom(bottom, color_to_dominant)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each lower framed region with the dominant color of the connected upper graph group named by its marker.

    Input structure:
    - A full-width azure row splits the grid into an upper graph panel and a lower target panel.
    - In the upper panel, black background contains small monochrome colored blocks plus red one-cell orthogonal connectors.
    - In the lower panel, grey rectangular frames enclose black interiors, and each interior has one colored corner marker matching an upper block color.

    Rule:
    - In the upper panel, treat colored blocks as graph nodes and red cells as links; blocks connected through red paths belong to the same group.
    - For each group, choose the largest colored block; if several tie, pick the one whose top-left cell is earliest in reading order. Its color is that group's dominant color.
    - In the lower panel, use each region's corner-marker color to identify the corresponding upper block and therefore its group's dominant color.
    - Fill the black cells of that region's interior with the dominant color while leaving the grey frame and the marker pixel unchanged.

    Key insight:
    - The top panel is a connectivity legend, so the task is solved by reading graph components first and only then transferring the resulting color choice into the separate lower layout.
    """
    return _solve(grid)


def _random_partition(total: int, parts: int, cap: int, rng: random.Random) -> list[int] | None:
    sizes = [1] * parts
    remain = total - parts
    while remain > 0:
        choices = [i for i, value in enumerate(sizes) if value < cap]
        if not choices:
            return None
        idx = rng.choice(choices)
        sizes[idx] += 1
        remain -= 1
    rng.shuffle(sizes)
    return sizes


def _flip_group(group: dict[str, object], flip_v: bool, flip_h: bool) -> dict[str, object]:
    h = int(group["h"])
    w = int(group["w"])
    nodes = []
    for r, c, rh, rw, color in group["nodes"]:
        nr = h - r - rh if flip_v else r
        nc = w - c - rw if flip_h else c
        nodes.append((nr, nc, rh, rw, color))
    red = []
    for r, c in group["red"]:
        nr = h - 1 - r if flip_v else r
        nc = w - 1 - c if flip_h else c
        red.append((nr, nc))
    return {"h": h, "w": w, "nodes": nodes, "red": red}


def _make_chain_group(colors: list[int], orientation: str, rng: random.Random) -> dict[str, object]:
    k = len(colors)
    tie_mode = rng.random() < 0.5
    if orientation == "h":
        height = rng.choice((2, 3))
        widths = [2] * k
        if not tie_mode:
            widths[rng.randrange(k)] = 3
        gaps = [rng.randint(2, 4) for _ in range(k - 1)]
        nodes = []
        x = 0
        for idx, color in enumerate(colors):
            nodes.append((0, x, height, widths[idx], color))
            x += widths[idx]
            if idx < k - 1:
                x += gaps[idx]
        red = []
        row = height // 2
        for idx in range(k - 1):
            _, x0, _, w0, _ = nodes[idx]
            _, x1, _, _, _ = nodes[idx + 1]
            for c in range(x0 + w0, x1):
                red.append((row, c))
        return _flip_group(
            {"h": height, "w": x, "nodes": nodes, "red": red},
            flip_v=bool(rng.getrandbits(1)),
            flip_h=bool(rng.getrandbits(1)),
        )

    width = rng.choice((2, 3))
    heights = [2] * k
    if not tie_mode:
        heights[rng.randrange(k)] = 3
    gaps = [rng.randint(2, 4) for _ in range(k - 1)]
    nodes = []
    y = 0
    for idx, color in enumerate(colors):
        nodes.append((y, 0, heights[idx], width, color))
        y += heights[idx]
        if idx < k - 1:
            y += gaps[idx]
    red = []
    col = width // 2
    for idx in range(k - 1):
        y0, _, h0, _, _ = nodes[idx]
        y1, _, _, _, _ = nodes[idx + 1]
        for r in range(y0 + h0, y1):
            red.append((r, col))
    return _flip_group(
        {"h": y, "w": width, "nodes": nodes, "red": red},
        flip_v=bool(rng.getrandbits(1)),
        flip_h=bool(rng.getrandbits(1)),
    )


def _make_elbow_group(colors: list[int], rng: random.Random) -> dict[str, object]:
    size = rng.choice((2, 3))
    gap_h = rng.randint(2, 4)
    gap_v = rng.randint(2, 4)
    nodes = [
        (0, 0, size, size, colors[0]),
        (0, size + gap_h, size, size, colors[1]),
        (size + gap_v, 0, size, size, colors[2]),
    ]
    red = []
    mid = size // 2
    for c in range(size, size + gap_h):
        red.append((mid, c))
    for r in range(size, size + gap_v):
        red.append((r, mid))
    return _flip_group(
        {"h": size * 2 + gap_v, "w": size * 2 + gap_h, "nodes": nodes, "red": red},
        flip_v=bool(rng.getrandbits(1)),
        flip_h=bool(rng.getrandbits(1)),
    )


def _make_singleton_group(colors: list[int], rng: random.Random) -> dict[str, object]:
    size = rng.choice((2, 3))
    node = (0, 0, size, size, colors[0])
    return _flip_group({"h": size, "w": size, "nodes": [node], "red": []}, False, False)


def _make_group(colors: list[int], rng: random.Random) -> dict[str, object]:
    if len(colors) == 1:
        return _make_singleton_group(colors, rng)
    if len(colors) == 3 and rng.random() < 0.34:
        return _make_elbow_group(colors, rng)
    return _make_chain_group(colors, rng.choice(("h", "v")), rng)


def _paint_group(panel: np.ndarray, group: dict[str, object], top: int, left: int) -> None:
    for r, c, h, w, color in group["nodes"]:
        panel[top + r : top + r + h, left + c : left + c + w] = int(color)
    for r, c in group["red"]:
        panel[top + r, left + c] = RED


def _build_top_panel(rng: random.Random) -> tuple[np.ndarray, dict[int, int], list[list[int]]] | None:
    group_count = rng.randint(*GROUP_COUNT_RANGE)
    total_nodes = rng.randint(group_count + 1, min(6, group_count * 3))
    sizes = _random_partition(total_nodes, group_count, 3, rng)
    if sizes is None or max(sizes) == 1:
        return None

    colors = list(rng.sample(NODE_COLORS, total_nodes))
    groups_local: list[dict[str, object]] = []
    color_groups: list[list[int]] = []
    idx = 0
    for size in sizes:
        group_colors = colors[idx : idx + size]
        idx += size
        color_groups.append(group_colors)
        groups_local.append(_make_group(group_colors, rng))

    base_gaps = [rng.randint(*TOP_GROUP_GAP_RANGE) for _ in range(group_count - 1)]
    min_w = 2 + sum(int(group["w"]) for group in groups_local) + sum(base_gaps)
    if min_w > GRID_W_MAX:
        return None
    width = rng.randint(max(GRID_W_MIN, min_w), GRID_W_MAX)

    top_h_min = max(TOP_H_RANGE[0], max(int(group["h"]) for group in groups_local) + 1)
    top_h_max = TOP_H_RANGE[1]
    if top_h_min > top_h_max:
        return None
    top_h = rng.randint(top_h_min, top_h_max)

    top = np.full((top_h, width), BLACK, dtype=int)
    slack = width - min_w
    extra_spaces = [0] * (group_count + 1)
    for _ in range(slack):
        extra_spaces[rng.randrange(len(extra_spaces))] += 1

    x = 1 + extra_spaces[0]
    for i, group in enumerate(groups_local):
        gh = int(group["h"])
        y = rng.randint(0, top_h - gh)
        _paint_group(top, group, y, x)
        x += int(group["w"])
        if i < len(base_gaps):
            x += base_gaps[i] + extra_spaces[i + 1]

    color_to_dominant = _top_color_to_dominant(top)
    return top, color_to_dominant, color_groups


def _marker_colors(
    color_to_dominant: dict[int, int], color_groups: list[list[int]], rng: random.Random
) -> list[int]:
    markers: list[int] = []
    non_dominant = [color for color, dominant in color_to_dominant.items() if color != dominant]
    used_non_dominant = False

    for colors in color_groups:
        preferred = [color for color in colors if color_to_dominant[color] != color]
        if preferred and not used_non_dominant:
            choice = rng.choice(preferred)
            used_non_dominant = True
        else:
            choice = rng.choice(colors)
            used_non_dominant = used_non_dominant or color_to_dominant[choice] != choice
        markers.append(choice)

    if non_dominant and not used_non_dominant:
        markers[0] = rng.choice(non_dominant)

    extra = rng.randint(*EXTRA_REGION_RANGE)
    palette = list(color_to_dominant.keys())
    for _ in range(extra):
        markers.append(rng.choice(palette))
    rng.shuffle(markers)
    return markers


def _place_regions(bottom_h: int, width: int, markers: list[int], rng: random.Random) -> np.ndarray | None:
    panel = np.full((bottom_h, width), BLACK, dtype=int)
    occupied = np.zeros((bottom_h, width), dtype=bool)
    corners = lambda r0, c0, h, w: (
        (r0 + 1, c0 + 1),
        (r0 + 1, c0 + w - 2),
        (r0 + h - 2, c0 + 1),
        (r0 + h - 2, c0 + w - 2),
    )

    for color in markers:
        placed = False
        for _ in range(FRAME_PLACE_RETRIES):
            inner_h = rng.randint(*FRAME_INNER_H_RANGE)
            inner_w = rng.randint(*FRAME_INNER_W_RANGE)
            frame_h = inner_h + 2
            frame_w = inner_w + 2
            if frame_h > bottom_h or frame_w > width:
                continue
            r0 = rng.randint(0, bottom_h - frame_h)
            c0 = rng.randint(0, width - frame_w)
            rr0 = max(0, r0 - 1)
            cc0 = max(0, c0 - 1)
            rr1 = min(bottom_h, r0 + frame_h + 1)
            cc1 = min(width, c0 + frame_w + 1)
            if occupied[rr0:rr1, cc0:cc1].any():
                continue

            panel[r0, c0 : c0 + frame_w] = GREY
            panel[r0 + frame_h - 1, c0 : c0 + frame_w] = GREY
            panel[r0 : r0 + frame_h, c0] = GREY
            panel[r0 : r0 + frame_h, c0 + frame_w - 1] = GREY
            mr, mc = rng.choice(corners(r0, c0, frame_h, frame_w))
            panel[mr, mc] = int(color)
            occupied[r0 : r0 + frame_h, c0 : c0 + frame_w] = True
            placed = True
            break
        if not placed:
            return None

    return panel


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    built = _build_top_panel(rng)
    if built is None:
        return None
    top, color_to_dominant, color_groups = built

    top_h, width = top.shape
    bottom_h_max = min(BOTTOM_H_RANGE[1], 30 - 1 - top_h)
    bottom_h_min = BOTTOM_H_RANGE[0]
    if bottom_h_min > bottom_h_max:
        return None
    bottom_h = rng.randint(bottom_h_min, bottom_h_max)

    markers = _marker_colors(color_to_dominant, color_groups, rng)
    bottom = _place_regions(bottom_h, width, markers, rng)
    if bottom is None:
        return None

    grid = np.vstack([top, np.full((1, width), AZURE, dtype=int), bottom])
    out = _solve(grid)
    if not np.any(out != grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(random.Random())
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
