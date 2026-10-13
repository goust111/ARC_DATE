from __future__ import annotations

import random

import numpy as np


NEIGHBORS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
GEN_RETRIES = 400
MODES = ("isolated", "chain", "star", "cycle")


def _dominant_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _label_components(grid: np.ndarray, bg: int) -> tuple[np.ndarray, list[np.ndarray]]:
    h, w = grid.shape
    labels = np.zeros((h, w), dtype=int)
    components: list[np.ndarray] = []
    comp_id = 0
    for r in range(h):
        for c in range(w):
            if labels[r, c] != 0 or int(grid[r, c]) == bg:
                continue
            color = int(grid[r, c])
            comp_id += 1
            stack = [(r, c)]
            labels[r, c] = comp_id
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in NEIGHBORS4:
                    nr, nc = rr + dr, cc + dc
                    if (
                        0 <= nr < h
                        and 0 <= nc < w
                        and labels[nr, nc] == 0
                        and int(grid[nr, nc]) == color
                    ):
                        labels[nr, nc] = comp_id
                        stack.append((nr, nc))
            components.append(np.array(coords, dtype=int))
    return labels, components


def _legend_info(grid: np.ndarray, bg: int) -> tuple[int, list[int]]:
    rows = np.flatnonzero(np.any(grid != bg, axis=1))
    assert rows.size > 0, "expected at least one non-background row"
    legend_row = int(rows[-1])
    legend_cols = np.flatnonzero(grid[legend_row] != bg)
    assert legend_cols.size > 0, "legend row must contain colored pixels"
    legend = [int(grid[legend_row, c]) for c in legend_cols]
    return legend_row, legend


def _analyze_objects(grid: np.ndarray) -> tuple[int, int, list[int], list[np.ndarray], list[set[int]]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _dominant_color(g)
    legend_row, legend = _legend_info(g, bg)
    play = g[:legend_row, :]
    labels, components = _label_components(play, bg)

    neighbors = [set() for _ in range(len(components) + 1)]
    h, w = play.shape
    for r in range(h):
        for c in range(w):
            a = int(labels[r, c])
            if a == 0:
                continue
            for dr, dc in ((1, 0), (0, 1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w:
                    b = int(labels[nr, nc])
                    if b != 0 and b != a:
                        neighbors[a].add(b)
                        neighbors[b].add(a)
    return bg, legend_row, legend, components, neighbors


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each object by how many distinct 4-neighbor contacts it has.

    The bottom legend row lists colors for contact-count 0, 1, 2, ...
    Every foreground object above the legend is recolored by the legend color
    indexed by its number of neighboring objects, and the legend row is erased.
    """
    g = np.asarray(grid, dtype=int)
    bg, legend_row, legend, components, neighbors = _analyze_objects(g)

    out = g.copy()
    out[legend_row, :] = bg
    for comp_id, coords in enumerate(components, start=1):
        contact_count = len(neighbors[comp_id])
        assert contact_count < len(legend), "legend must cover every contact count"
        color = int(legend[contact_count])
        out[coords[:, 0], coords[:, 1]] = color
    return out


def _transform_points(points: list[tuple[int, int]]) -> list[tuple[int, int]]:
    transformed = list(points)
    if random.random() < 0.5:
        transformed = [(r, -c) for r, c in transformed]
    turns = random.randrange(4)
    for _ in range(turns):
        transformed = [(c, -r) for r, c in transformed]
    min_r = min(r for r, _ in transformed)
    min_c = min(c for _, c in transformed)
    return [(r - min_r, c - min_c) for r, c in transformed]


def _base_pattern(mode: str) -> tuple[list[tuple[int, int]], int]:
    if mode == "isolated":
        points = [(0, 0)]
        count = random.randint(3, 5)
        return points, count
    if mode == "chain":
        length = random.randint(3, 5)
        points = [(0, i) for i in range(length)]
        extra = random.randint(0, 2)
        return points, length + extra
    if mode == "star":
        leaf_count = random.choice((3, 4))
        leaves = [(-1, 0), (0, 1), (1, 0), (0, -1)]
        random.shuffle(leaves)
        points = [(0, 0), *leaves[:leaf_count]]
        extra = random.randint(0, 1)
        return points, len(points) + extra
    if mode == "cycle":
        points = [(0, 0), (0, 1), (1, 0), (1, 1)]
        extra = random.randint(0, 2)
        return points, len(points) + extra
    raise AssertionError(f"unknown mode {mode}")


def _place_base_points(board_h: int, board_w: int, mode: str) -> list[set[tuple[int, int]]] | None:
    raw_points, total_objects = _base_pattern(mode)
    points = _transform_points(raw_points)
    max_r = max(r for r, _ in points)
    max_c = max(c for _, c in points)
    if max_r >= board_h or max_c >= board_w:
        return None

    top = random.randint(0, board_h - max_r - 1)
    left = random.randint(0, board_w - max_c - 1)
    placed = {(r + top, c + left) for r, c in points}
    objects = [{cell} for cell in placed]

    while len(objects) < total_objects:
        candidates = []
        for r in range(board_h):
            for c in range(board_w):
                cell = (r, c)
                if cell in placed:
                    continue
                if any(abs(r - rr) + abs(c - cc) == 1 for rr, cc in placed):
                    continue
                candidates.append(cell)
        if not candidates:
            return None
        cell = random.choice(candidates)
        placed.add(cell)
        objects.append({cell})

    return objects


def _grow_objects(objects: list[set[tuple[int, int]]], board_h: int, board_w: int) -> None:
    labels = -np.ones((board_h, board_w), dtype=int)
    for idx, cells in enumerate(objects):
        for r, c in cells:
            labels[r, c] = idx

    order = list(range(len(objects)))
    random.shuffle(order)
    for idx in order:
        for _ in range(random.randint(0, 2)):
            candidates = set()
            for r, c in objects[idx]:
                for dr, dc in NEIGHBORS4:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < board_h and 0 <= nc < board_w and labels[nr, nc] == -1:
                        ok = True
                        for ddr, ddc in NEIGHBORS4:
                            ar, ac = nr + ddr, nc + ddc
                            if 0 <= ar < board_h and 0 <= ac < board_w:
                                owner = int(labels[ar, ac])
                                if owner not in (-1, idx):
                                    ok = False
                                    break
                        if ok:
                            candidates.add((nr, nc))
            if not candidates:
                break
            nr, nc = random.choice(tuple(candidates))
            labels[nr, nc] = idx
            objects[idx].add((nr, nc))


def _neighbor_counts(objects: list[set[tuple[int, int]]], board_h: int, board_w: int) -> list[int]:
    labels = -np.ones((board_h, board_w), dtype=int)
    for idx, cells in enumerate(objects):
        for r, c in cells:
            labels[r, c] = idx

    touches = [set() for _ in objects]
    for r in range(board_h):
        for c in range(board_w):
            a = int(labels[r, c])
            if a < 0:
                continue
            for dr, dc in ((1, 0), (0, 1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < board_h and 0 <= nc < board_w:
                    b = int(labels[nr, nc])
                    if b >= 0 and b != a:
                        touches[a].add(b)
                        touches[b].add(a)
    return [len(s) for s in touches]


def _render_grid(
    objects: list[set[tuple[int, int]]],
    board_h: int,
    board_w: int,
    object_colors: list[int],
    legend: list[int],
    bg: int,
) -> np.ndarray:
    cell_size = random.choice((2, 3))
    top_margin = random.randint(1, 3)
    left_margin = random.randint(1, 3)
    right_margin = random.randint(1, 3)
    gap_rows = random.randint(1, 2)

    content_h = board_h * cell_size
    content_w = board_w * cell_size
    legend_gaps = [random.randint(1, 3) for _ in range(len(legend) - 1)]
    legend_w = len(legend) + sum(legend_gaps)
    width = max(left_margin + content_w + right_margin, legend_w + 4)
    height = top_margin + content_h + gap_rows + 1

    grid = np.full((height, width), bg, dtype=int)
    top = top_margin
    left = random.randint(left_margin, width - content_w - right_margin)
    for color, cells in zip(object_colors, objects):
        for cr, cc in cells:
            r0 = top + cr * cell_size
            c0 = left + cc * cell_size
            grid[r0 : r0 + cell_size, c0 : c0 + cell_size] = color

    legend_row = height - 1
    cursor = random.randint(1, width - legend_w - 1)
    for idx, color in enumerate(legend):
        grid[legend_row, cursor] = color
        if idx < len(legend) - 1:
            cursor += 1 + legend_gaps[idx]
    return grid


def _gen_attempt() -> np.ndarray | None:
    bg = random.randrange(10)
    board_h = random.randint(5, 8)
    board_w = random.randint(5, 8)
    mode = random.choice(MODES)

    objects = _place_base_points(board_h, board_w, mode)
    if objects is None:
        return None
    _grow_objects(objects, board_h, board_w)

    counts = _neighbor_counts(objects, board_h, board_w)
    max_count = max(counts)
    legend_len = max(max_count + 1, random.randint(2, 5))
    if legend_len > 8:
        return None

    palette = [color for color in range(10) if color != bg]
    if len(objects) > len(palette):
        return None
    object_colors = random.sample(palette, len(objects))
    legend = random.sample(palette, legend_len)

    grid = _render_grid(objects, board_h, board_w, object_colors, legend, bg)

    area_counts = np.unique(grid, return_counts=True)[1]
    if area_counts.max() != int((grid == bg).sum()):
        return None

    try:
        _bg, legend_row, parsed_legend, components, neighbors = _analyze_objects(grid)
    except AssertionError:
        return None

    if _bg != bg or legend_row != grid.shape[0] - 1:
        return None
    if parsed_legend != legend:
        return None
    if len(components) != len(objects):
        return None
    parsed_counts = sorted(len(neighbors[i]) for i in range(1, len(components) + 1))
    if parsed_counts != sorted(counts):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
