from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _detect_lattice(grid: np.ndarray) -> tuple[int, list[int], list[int]]:
    h, w = grid.shape
    best: tuple[int, list[int], list[int], tuple[int, int, int, int]] | None = None

    for color in map(int, np.unique(grid)):
        rows = [r for r in range(h) if bool(np.all(grid[r, :] == color))]
        cols = [c for c in range(w) if bool(np.all(grid[:, c] == color))]
        if len(rows) < 2 or len(cols) < 2:
            continue
        if rows[0] != 0 or rows[-1] != h - 1 or cols[0] != 0 or cols[-1] != w - 1:
            continue

        row_steps = [rows[i + 1] - rows[i] for i in range(len(rows) - 1)]
        col_steps = [cols[i + 1] - cols[i] for i in range(len(cols) - 1)]
        if not row_steps or not col_steps:
            continue
        if len(set(row_steps)) != 1 or len(set(col_steps)) != 1:
            continue
        if row_steps[0] <= 1 or col_steps[0] <= 1:
            continue

        row_cells = len(rows) - 1
        col_cells = len(cols) - 1
        score = (row_cells * col_cells, row_cells + col_cells, row_steps[0], col_steps[0])
        if best is None or score > best[3]:
            best = (color, rows, cols, score)

    if best is None:
        raise ValueError("failed to detect separator lattice")
    return best[0], best[1], best[2]


def _shape_mask(cell: np.ndarray, bg: int) -> tuple[np.ndarray, int]:
    dot_color = int(cell[0, 0]) if int(cell[0, 0]) != bg else bg
    mask = cell != bg
    if dot_color != bg:
        mask = mask.copy()
        mask[0, 0] = False
    return mask, dot_color


def _bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(mask)
    if coords.size == 0:
        raise ValueError("empty shape")
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _cavity_mask(shape_mask: np.ndarray) -> np.ndarray:
    h, w = shape_mask.shape
    outside = np.zeros((h, w), dtype=bool)
    q: deque[tuple[int, int]] = deque()

    for r in range(h):
        for c in (0, w - 1):
            if not shape_mask[r, c] and not outside[r, c]:
                outside[r, c] = True
                q.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if not shape_mask[r, c] and not outside[r, c]:
                outside[r, c] = True
                q.append((r, c))

    while q:
        r, c = q.popleft()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            rr, cc = r + dr, c + dc
            if 0 <= rr < h and 0 <= cc < w and not shape_mask[rr, cc] and not outside[rr, cc]:
                outside[rr, cc] = True
                q.append((rr, cc))

    return (~shape_mask) & (~outside)


def _can_fit_rect(cavity: np.ndarray, target_h: int, target_w: int) -> bool:
    h, w = cavity.shape
    if target_h > h or target_w > w:
        return False
    arr = cavity.astype(int)
    prefix = arr.cumsum(axis=0).cumsum(axis=1)

    def rect_sum(r0: int, c0: int, r1: int, c1: int) -> int:
        total = int(prefix[r1, c1])
        if r0 > 0:
            total -= int(prefix[r0 - 1, c1])
        if c0 > 0:
            total -= int(prefix[r1, c0 - 1])
        if r0 > 0 and c0 > 0:
            total += int(prefix[r0 - 1, c0 - 1])
        return total

    need = target_h * target_w
    for r0 in range(h - target_h + 1):
        r1 = r0 + target_h - 1
        for c0 in range(w - target_w + 1):
            c1 = c0 + target_w - 1
            if rect_sum(r0, c0, r1, c1) == need:
                return True
    return False


def _extract_cells(grid: np.ndarray) -> tuple[int, int, list[int], list[int], list[dict]]:
    bg = _most_common_color(grid)
    sep, rows, cols = _detect_lattice(grid)
    cells: list[dict] = []

    for rr in range(len(rows) - 1):
        for cc in range(len(cols) - 1):
            r0, r1 = rows[rr], rows[rr + 1]
            c0, c1 = cols[cc], cols[cc + 1]
            cell = np.array(grid[r0 + 1 : r1, c0 + 1 : c1], copy=True)
            mask, dot_color = _shape_mask(cell, bg)
            bbox_r0, bbox_c0, bbox_r1, bbox_c1 = _bbox_from_mask(mask)
            cavity = _cavity_mask(mask)

            colors = [int(v) for v in cell[mask]]
            shape_color = int(Counter(colors).most_common(1)[0][0])
            cells.append(
                {
                    "grid_row": rr,
                    "grid_col": cc,
                    "cell": cell,
                    "dot_color": dot_color,
                    "shape_color": shape_color,
                    "bbox_h": bbox_r1 - bbox_r0 + 1,
                    "bbox_w": bbox_c1 - bbox_c0 + 1,
                    "bbox_area": (bbox_r1 - bbox_r0 + 1) * (bbox_c1 - bbox_c0 + 1),
                    "cavity": cavity,
                }
            )

    return bg, sep, rows, cols, cells


def _parent_map(cells: list[dict]) -> list[int]:
    n = len(cells)
    parents = [-1] * n
    for child in range(n):
        candidates = []
        for parent in range(n):
            if child == parent:
                continue
            if cells[parent]["bbox_area"] <= cells[child]["bbox_area"]:
                continue
            if _can_fit_rect(cells[parent]["cavity"], cells[child]["bbox_h"], cells[child]["bbox_w"]):
                candidates.append(parent)
        if candidates:
            parents[child] = min(
                candidates,
                key=lambda idx: (
                    int(cells[idx]["bbox_area"]),
                    int(cells[idx]["bbox_h"] + cells[idx]["bbox_w"]),
                    int(idx),
                ),
            )
    return parents


def main(grid: np.ndarray) -> np.ndarray:
    """
    Follow the color-indicated containment chain and pack its cells into the leftmost column.
    """
    g = np.asarray(grid, dtype=int)
    bg, sep, rows, cols, cells = _extract_cells(g)
    parents = _parent_map(cells)

    children: dict[int, list[int]] = {idx: [] for idx in range(len(cells))}
    for child, parent in enumerate(parents):
        if parent != -1:
            children[parent].append(child)

    roots = [idx for idx, parent in enumerate(parents) if parent == -1]
    if len(roots) != 1:
        raise ValueError("expected exactly one root")
    root = roots[0]

    path = [root]
    current = root
    seen = {root}
    while children[current]:
        choice = cells[current]["dot_color"]
        matches = [child for child in children[current] if int(cells[child]["shape_color"]) == int(choice)]
        if len(matches) != 1:
            break
        nxt = matches[0]
        if nxt in seen:
            break
        path.append(nxt)
        seen.add(nxt)
        current = nxt

    out = np.full_like(g, bg)
    out[rows, :] = sep
    out[:, cols] = sep

    for depth, idx in enumerate(path):
        if depth >= len(rows) - 1:
            break
        r0, r1 = rows[depth], rows[depth + 1]
        c0, c1 = cols[0], cols[1]
        out[r0 + 1 : r1, c0 + 1 : c1] = cells[idx]["cell"]

    return out


# --------------- generator ---------------

MAX_GENERATION_ATTEMPTS = 500
STAR_LEAF_SPECS = [(5, 5, 1), (5, 4, 1), (5, 3, 1), (4, 4, 1), (4, 3, 1)]
CASE_NAMES = ("small_star", "wide_star", "deep_split", "deep_fan")


def _spec_parent_map(specs: list[tuple[int, int, int]]) -> list[int]:
    areas = [h * w for h, w, _t in specs]
    parents = [-1] * len(specs)
    for child, (ch, cw, _ct) in enumerate(specs):
        candidates = []
        for parent, (ph, pw, pt) in enumerate(specs):
            if child == parent:
                continue
            if areas[parent] <= areas[child]:
                continue
            if ch <= ph - 2 * pt and cw <= pw - 2 * pt:
                candidates.append(parent)
        if candidates:
            parents[child] = min(candidates, key=lambda idx: (areas[idx], specs[idx][0] + specs[idx][1], idx))
    return parents


def _transpose_specs(specs: list[tuple[int, int, int]]) -> list[tuple[int, int, int]]:
    return [(w, h, t) for h, w, t in specs]


def _build_case() -> tuple[int, int, int, list[int], list[tuple[int, int, int]], int]:
    case = random.choice(CASE_NAMES)

    if case == "small_star":
        rows, cols, cell_inner = 2, 2, 8
        parents = [-1, 0, 0, 0]
        specs = [(7, 7, 1)] + random.sample(STAR_LEAF_SPECS, 3)
        path_len = 2
    elif case == "wide_star":
        rows, cols, cell_inner = 2, 3, 8
        parents = [-1, 0, 0, 0, 0, 0]
        specs = [(7, 7, 1)] + list(STAR_LEAF_SPECS)
        path_len = 2
    elif case == "deep_split":
        rows, cols, cell_inner = 3, 2, 9
        parents = [-1, 0, 0, 1, 1, 2]
        specs = [(8, 8, 1), (6, 6, 1), (6, 5, 1), (4, 4, 1), (3, 4, 1), (3, 3, 1)]
        path_len = 3
    else:
        rows, cols, cell_inner = 3, 2, 9
        parents = [-1, 0, 0, 0, 1, 1]
        specs = [(8, 8, 1), (6, 6, 1), (5, 4, 1), (5, 3, 1), (4, 4, 1), (3, 4, 1)]
        path_len = 3

    if random.random() < 0.5:
        specs = _transpose_specs(specs)

    if _spec_parent_map(specs) != parents:
        raise ValueError("invalid case specification")
    return rows, cols, cell_inner, parents, specs, path_len


def _children_from_parents(parents: list[int]) -> dict[int, list[int]]:
    children: dict[int, list[int]] = {idx: [] for idx in range(len(parents))}
    for child, parent in enumerate(parents):
        if parent != -1:
            children[parent].append(child)
    return children


def _root_to_leaf_paths(children: dict[int, list[int]], node: int) -> list[list[int]]:
    if not children[node]:
        return [[node]]
    out: list[list[int]] = []
    for child in children[node]:
        for tail in _root_to_leaf_paths(children, child):
            out.append([node] + tail)
    return out


def _selected_edges(
    children: dict[int, list[int]],
    bg: int,
    shape_colors: list[int],
    target_path_len: int,
) -> tuple[list[int], list[int]]:
    chosen_child = [-1] * len(shape_colors)
    all_paths = _root_to_leaf_paths(children, 0)
    candidates = [path for path in all_paths if len(path) == target_path_len]
    if not candidates:
        raise ValueError("missing target-depth path")
    path = random.choice(candidates)
    for src, dst in zip(path, path[1:], strict=False):
        chosen_child[src] = dst

    for node, kids in children.items():
        if not kids or chosen_child[node] != -1:
            continue
        chosen_child[node] = random.choice(kids)

    dot_colors = [bg] * len(shape_colors)
    for node, child in enumerate(chosen_child):
        if child != -1:
            dot_colors[node] = shape_colors[child]
    return dot_colors, path


def _draw_frame(cell: np.ndarray, color: int, bg: int, spec: tuple[int, int, int]) -> None:
    h, w, t = spec
    r0 = random.randint(1, cell.shape[0] - h)
    c0 = random.randint(1, cell.shape[1] - w)
    cell[r0 : r0 + h, c0 : c0 + w] = color
    cell[r0 + t : r0 + h - t, c0 + t : c0 + w - t] = bg


def _render_input(
    rows: int,
    cols: int,
    cell_inner: int,
    sep: int,
    bg: int,
    specs: list[tuple[int, int, int]],
    shape_colors: list[int],
    dot_colors: list[int],
    slots: list[tuple[int, int]],
) -> np.ndarray:
    h = rows * (cell_inner + 1) + 1
    w = cols * (cell_inner + 1) + 1
    grid = np.full((h, w), bg, dtype=int)

    row_lines = [k * (cell_inner + 1) for k in range(rows + 1)]
    col_lines = [k * (cell_inner + 1) for k in range(cols + 1)]
    grid[row_lines, :] = sep
    grid[:, col_lines] = sep

    for node, (rr, cc) in enumerate(slots):
        r0 = row_lines[rr] + 1
        c0 = col_lines[cc] + 1
        cell = grid[r0 : r0 + cell_inner, c0 : c0 + cell_inner]
        if dot_colors[node] != bg:
            cell[0, 0] = dot_colors[node]
        _draw_frame(cell, shape_colors[node], bg, specs[node])

    return grid


def _follow_path(cells: list[dict], parents: list[int]) -> list[int]:
    children: dict[int, list[int]] = {idx: [] for idx in range(len(cells))}
    for child, parent in enumerate(parents):
        if parent != -1:
            children[parent].append(child)

    roots = [idx for idx, parent in enumerate(parents) if parent == -1]
    if len(roots) != 1:
        raise ValueError("expected exactly one root")

    path = [roots[0]]
    current = roots[0]
    seen = {current}
    while children[current]:
        choice = cells[current]["dot_color"]
        matches = [child for child in children[current] if int(cells[child]["shape_color"]) == int(choice)]
        if len(matches) != 1:
            break
        current = matches[0]
        if current in seen:
            break
        seen.add(current)
        path.append(current)
    return path


def _permute_parent_target(
    parents: list[int],
    rows: int,
    cols: int,
    slots: list[tuple[int, int]],
) -> list[int]:
    pos_to_node = {pos: node for node, pos in enumerate(slots)}
    ordered_nodes = [pos_to_node[(r, c)] for r in range(rows) for c in range(cols)]
    old_to_new = {old: new for new, old in enumerate(ordered_nodes)}
    out = [-1] * len(parents)
    for new_idx, old_idx in enumerate(ordered_nodes):
        old_parent = parents[old_idx]
        out[new_idx] = -1 if old_parent == -1 else old_to_new[old_parent]
    return out


def _valid_instance(grid: np.ndarray, parent_target: list[int], path_len: int) -> bool:
    try:
        out = main(grid)
        bg, sep, rows, cols, cells = _extract_cells(grid)
    except Exception:
        return False
    if out.shape != grid.shape:
        return False
    if len(rows) - 1 < path_len:
        return False
    if sep == _most_common_color(grid):
        return False
    if len(cells) != len(parent_target):
        return False
    parents = _parent_map(cells)
    if parents != parent_target:
        return False
    return len(_follow_path(cells, parents)) == path_len


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        rows, cols, cell_inner, parent_target, specs, path_len = _build_case()

        palette = random.sample(list(range(10)), len(parent_target) + 2)
        bg, sep = palette[:2]
        shape_colors = palette[2:]

        children = _children_from_parents(parent_target)
        dot_colors, path = _selected_edges(children, bg, shape_colors, path_len)
        slots = [(r, c) for r in range(rows) for c in range(cols)]
        random.shuffle(slots)
        grid = _render_input(rows, cols, cell_inner, sep, bg, specs, shape_colors, dot_colors, slots)
        expected_parents = _permute_parent_target(parent_target, rows, cols, slots)

        if _valid_instance(grid, expected_parents, len(path)):
            return grid

    raise RuntimeError("failed to generate a valid input")
