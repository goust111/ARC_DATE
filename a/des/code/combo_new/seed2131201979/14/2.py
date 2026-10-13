from __future__ import annotations

from collections import defaultdict, deque
import random

import numpy as np


Coord = tuple[int, int]
State = dict[str, object]


def _coords_from_mask(mask: np.ndarray) -> tuple[Coord, ...]:
    return tuple(sorted((int(r), int(c)) for r, c in np.argwhere(mask)))


def _normalize_coords(coords: set[Coord] | frozenset[Coord] | tuple[Coord, ...]) -> tuple[tuple[Coord, ...], tuple[int, int], tuple[int, int]]:
    ordered = tuple(sorted((int(r), int(c)) for r, c in coords))
    assert ordered
    r0 = min(r for r, _ in ordered)
    c0 = min(c for _, c in ordered)
    norm = tuple(sorted((r - r0, c - c0) for r, c in ordered))
    h = max(r for r, _ in norm) + 1
    w = max(c for _, c in norm) + 1
    return norm, (r0, c0), (h, w)


def _mask_from_coords(coords: tuple[Coord, ...], shape: tuple[int, int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    for r, c in coords:
        mask[int(r), int(c)] = True
    return mask


def _shift_cells(cells: tuple[Coord, ...] | set[Coord] | frozenset[Coord], dr: int, dc: int) -> frozenset[Coord]:
    return frozenset((int(r) + dr, int(c) + dc) for r, c in cells)


def _is_connected(cells: set[Coord] | frozenset[Coord] | tuple[Coord, ...]) -> bool:
    pts = {tuple(map(int, cell)) for cell in cells}
    if not pts:
        return False
    queue = deque([next(iter(pts))])
    seen = {queue[0]}
    while queue:
        r, c = queue.popleft()
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if (nr, nc) in pts and (nr, nc) not in seen:
                seen.add((nr, nc))
                queue.append((nr, nc))
    return len(seen) == len(pts)


def _has_four_distinct_rotations(mask: np.ndarray) -> bool:
    keys = {_coords_from_mask(np.rot90(mask, k)) for k in range(4)}
    return len(keys) == 4


def _visible_cells_by_color(grid: np.ndarray) -> dict[int, frozenset[Coord]]:
    colors = [int(color) for color in np.unique(grid) if int(color) != 0]
    visible: dict[int, frozenset[Coord]] = {}
    for color in colors:
        cells = frozenset((int(r), int(c)) for r, c in np.argwhere(grid == color))
        assert cells
        visible[color] = cells
    assert visible
    return visible


def _enumerate_states(template_mask: np.ndarray, visible: frozenset[Coord], support: np.ndarray) -> list[State]:
    H, W = support.shape
    visible_list = sorted(visible)
    first_r, first_c = visible_list[0]
    states: dict[frozenset[Coord], State] = {}

    for rotation in range(4):
        rotated = np.rot90(template_mask, rotation)
        rel_coords = _coords_from_mask(rotated)
        for dr0, dc0 in rel_coords:
            anchor_r = first_r - dr0
            anchor_c = first_c - dc0
            cells: list[Coord] = []
            valid = True
            for dr, dc in rel_coords:
                rr = anchor_r + dr
                cc = anchor_c + dc
                if not (0 <= rr < H and 0 <= cc < W and support[rr, cc]):
                    valid = False
                    break
                cells.append((rr, cc))
            if not valid:
                continue

            cell_set = frozenset(cells)
            if not visible.issubset(cell_set):
                continue

            states[cell_set] = {
                "cells": cell_set,
                "rotation": int(rotation),
                "anchor": (int(anchor_r), int(anchor_c)),
            }

    return list(states.values())


def _is_acyclic(colors: list[int], edges: set[tuple[int, int]]) -> bool:
    indeg = {color: 0 for color in colors}
    adj = {color: set() for color in colors}
    for low, high in edges:
        if high not in adj[low]:
            adj[low].add(high)
            indeg[high] += 1

    queue = deque([color for color in colors if indeg[color] == 0])
    seen = 0
    while queue:
        color = queue.popleft()
        seen += 1
        for nxt in adj[color]:
            indeg[nxt] -= 1
            if indeg[nxt] == 0:
                queue.append(nxt)
    return seen == len(colors)


def _partial_assignment_valid(assignment: dict[int, State], visible_by_color: dict[int, frozenset[Coord]]) -> bool:
    colors = list(assignment)
    edges: set[tuple[int, int]] = set()
    for high in colors:
        for low, state in assignment.items():
            if low == high:
                continue
            if visible_by_color[high] & state["cells"]:
                edges.add((low, high))
    return _is_acyclic(colors, edges)


def _unique_topological_order(colors: list[int], edges: set[tuple[int, int]]) -> tuple[int, ...] | None:
    indeg = {color: 0 for color in colors}
    adj = {color: set() for color in colors}
    for low, high in edges:
        if high not in adj[low]:
            adj[low].add(high)
            indeg[high] += 1

    frontier = sorted(color for color in colors if indeg[color] == 0)
    order: list[int] = []
    while frontier:
        if len(frontier) != 1:
            return None
        color = frontier.pop()
        order.append(color)
        added: list[int] = []
        for nxt in sorted(adj[color]):
            indeg[nxt] -= 1
            if indeg[nxt] == 0:
                added.append(nxt)
        frontier = sorted(frontier + added)

    if len(order) != len(colors):
        return None
    return tuple(order)


def _top_to_bottom_order(grid: np.ndarray, colors: list[int], assignment: dict[int, State], visible_by_color: dict[int, frozenset[Coord]]) -> tuple[int, ...] | None:
    occupants: defaultdict[Coord, set[int]] = defaultdict(set)
    for color, state in assignment.items():
        for cell in state["cells"]:
            occupants[cell].add(color)

    edges: set[tuple[int, int]] = set()
    for color, cells in visible_by_color.items():
        for cell in cells:
            present = occupants.get(cell)
            if not present or color not in present:
                return None
            for low in present:
                if low != color:
                    edges.add((low, color))

    bottom_to_top = _unique_topological_order(colors, edges)
    if bottom_to_top is None:
        return None

    rendered = np.zeros_like(grid)
    for color in bottom_to_top:
        for r, c in assignment[color]["cells"]:
            rendered[int(r), int(c)] = int(color)
    if not np.array_equal(rendered, grid):
        return None

    return tuple(reversed(bottom_to_top))


def _build_output(template_mask: np.ndarray, order_top_to_bottom: tuple[int, ...]) -> np.ndarray:
    h, w = template_mask.shape
    out = np.zeros((h, len(order_top_to_bottom) * w + len(order_top_to_bottom) - 1), dtype=int)
    for i, color in enumerate(order_top_to_bottom):
        c0 = i * (w + 1)
        patch = out[:, c0 : c0 + w]
        patch[template_mask] = int(color)
    return out


def _solve_outputs(grid: np.ndarray) -> list[np.ndarray]:
    visible_by_color = _visible_cells_by_color(grid)
    colors = sorted(visible_by_color)
    support = grid != 0
    max_area = max(len(cells) for cells in visible_by_color.values())
    template_colors = [
        color
        for color in colors
        if len(visible_by_color[color]) == max_area and _is_connected(visible_by_color[color])
    ]

    outputs: list[np.ndarray] = []
    seen_keys: set[tuple[bytes, tuple[int, int]]] = set()

    for template_color in template_colors:
        norm_coords, anchor, shape = _normalize_coords(visible_by_color[template_color])
        template_mask = _mask_from_coords(norm_coords, shape)
        if not _has_four_distinct_rotations(template_mask):
            continue

        candidates_by_color: dict[int, list[State]] = {}
        valid_template = True
        for color in colors:
            if color == template_color:
                candidates_by_color[color] = [
                    {
                        "cells": frozenset(visible_by_color[color]),
                        "rotation": 0,
                        "anchor": anchor,
                    }
                ]
            else:
                states = _enumerate_states(template_mask, visible_by_color[color], support)
                if not states:
                    valid_template = False
                    break
                candidates_by_color[color] = states
        if not valid_template:
            continue

        search_order = sorted(colors, key=lambda color: (len(candidates_by_color[color]), color))
        assignment: dict[int, State] = {}

        def dfs(index: int) -> None:
            if len(seen_keys) > 1:
                return
            if index == len(search_order):
                order = _top_to_bottom_order(grid, colors, assignment, visible_by_color)
                if order is None:
                    return
                out = _build_output(template_mask, order)
                key = (out.tobytes(), tuple(map(int, out.shape)))
                if key not in seen_keys:
                    seen_keys.add(key)
                    outputs.append(out)
                return

            color = search_order[index]
            for state in candidates_by_color[color]:
                assignment[color] = state
                if _partial_assignment_valid(assignment, visible_by_color):
                    dfs(index + 1)
                assignment.pop(color, None)

        dfs(0)
        if len(seen_keys) > 1:
            break

    return outputs


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover the shared rotated template, undo each color's rotation, and order the copies by occlusion depth.

    Input structure:
    - The background is black.
    - Every foreground color belongs to one copy of the same connected non-symmetric template.
    - Some copies are partially hidden because higher layers overwrite overlapping cells with another color.
    - At least one color still shows a complete unoccluded copy of the template.

    Rule:
    - Use a fully visible copy to extract the template in its standard orientation.
    - For every other color, fit one of the template's four rotations so that all visible cells are explained.
    - Infer the layer order from overwrite evidence: whenever color A is visible on a cell that another fitted copy also needs, A must be above that copy.
    - Output all copies restored to the standard orientation and place them left-to-right from top layer to bottom layer, with one black column between neighbors.

    Key insight:
    - The visible pixels are only fragments of rigid rotated copies, so the puzzle is solved by jointly reconstructing hidden template cells and the unique occlusion order that makes the observed colors render correctly.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert np.any(g != 0)

    outputs = _solve_outputs(g)
    assert len(outputs) == 1, "expected exactly one valid reconstruction"
    return outputs[0]


# --------------- generator ---------------

GEN_ATTEMPTS = 500
GEN_OBJECT_COUNT = (3, 5)
GEN_ARM_RANGE = (3, 5)
GEN_COLOR_POOL = [1, 2, 3, 4, 5, 6, 7, 8, 9]
GEN_MIN_VISIBLE = 2
GEN_CLUSTER_LIMIT = 18
GEN_EXTRA_MARGIN = 8
GEN_GRID_LIMIT = 30


def _random_template_mask() -> np.ndarray:
    while True:
        h = random.randint(*GEN_ARM_RANGE)
        w = random.randint(*GEN_ARM_RANGE)
        if h != w:
            break
    mask = np.zeros((h, w), dtype=bool)
    mask[:, 0] = True
    mask[h - 1, :] = True
    return np.rot90(mask, random.randrange(4))


def _combined_bbox(cell_sets: list[frozenset[Coord]]) -> tuple[int, int]:
    all_cells = set().union(*cell_sets)
    rows = [r for r, _ in all_cells]
    cols = [c for _, c in all_cells]
    return max(rows) - min(rows) + 1, max(cols) - min(cols) + 1


def _normalize_cell_sets(cell_sets: list[frozenset[Coord]]) -> tuple[list[frozenset[Coord]], int, int]:
    all_cells = set().union(*cell_sets)
    r0 = min(r for r, _ in all_cells)
    c0 = min(c for _, c in all_cells)
    shifted = [_shift_cells(cells, -r0, -c0) for cells in cell_sets]
    h, w = _combined_bbox(shifted)
    return shifted, int(h), int(w)


def _place_chain(shapes: list[tuple[Coord, ...]]) -> list[frozenset[Coord]] | None:
    placed: list[frozenset[Coord]] = [frozenset(shapes[0])]
    for i in range(1, len(shapes)):
        prev = placed[i - 1]
        earlier = set().union(*placed[: i - 1]) if i > 1 else set()
        above = set().union(*placed[:i])
        candidates: dict[frozenset[Coord], None] = {}

        for pr, pc in prev:
            for nr, nc in shapes[i]:
                shifted = _shift_cells(shapes[i], pr - nr, pc - nc)
                if not (shifted & above):
                    continue
                if len(shifted - above) < GEN_MIN_VISIBLE:
                    continue
                if not ((shifted & prev) - earlier):
                    continue

                trial = placed + [shifted]
                bbox_h, bbox_w = _combined_bbox(trial)
                if bbox_h > GEN_CLUSTER_LIMIT or bbox_w > GEN_CLUSTER_LIMIT:
                    continue
                candidates[shifted] = None

        if not candidates:
            return None
        placed.append(random.choice(list(candidates)))

    return placed


def _render_input(shape_sets: list[frozenset[Coord]], colors_top_to_bottom: list[int], grid_shape: tuple[int, int]) -> np.ndarray:
    grid = np.zeros(grid_shape, dtype=int)
    for color, cells in zip(reversed(colors_top_to_bottom), reversed(shape_sets)):
        for r, c in cells:
            grid[int(r), int(c)] = int(color)
    return grid


def _embed_shapes(shape_sets: list[frozenset[Coord]], cluster_h: int, cluster_w: int) -> tuple[list[frozenset[Coord]], tuple[int, int]] | None:
    extra_h = random.randint(0, min(GEN_EXTRA_MARGIN, GEN_GRID_LIMIT - cluster_h))
    extra_w = random.randint(0, min(GEN_EXTRA_MARGIN, GEN_GRID_LIMIT - cluster_w))
    H = cluster_h + extra_h
    W = cluster_w + extra_w
    if H > GEN_GRID_LIMIT or W > GEN_GRID_LIMIT:
        return None

    top = random.randint(0, H - cluster_h)
    left = random.randint(0, W - cluster_w)
    embedded = [_shift_cells(cells, top, left) for cells in shape_sets]
    return embedded, (H, W)


def _gen_attempt() -> np.ndarray | None:
    template_mask = _random_template_mask()
    template_coords = _coords_from_mask(template_mask)
    count = random.randint(*GEN_OBJECT_COUNT)
    colors_top_to_bottom = random.sample(GEN_COLOR_POOL, count)
    rotations = [0] + [random.randrange(4) for _ in range(count - 1)]
    rotated_shapes = [_coords_from_mask(np.rot90(template_mask, rotation)) for rotation in rotations]

    local_shapes = _place_chain(rotated_shapes)
    if local_shapes is None:
        return None

    local_shapes, cluster_h, cluster_w = _normalize_cell_sets(local_shapes)
    embedded = _embed_shapes(local_shapes, cluster_h, cluster_w)
    if embedded is None:
        return None

    shape_sets, grid_shape = embedded
    inp = _render_input(shape_sets, colors_top_to_bottom, grid_shape)
    expected = _build_output(template_mask, tuple(colors_top_to_bottom))

    if expected.shape[0] > GEN_GRID_LIMIT or expected.shape[1] > GEN_GRID_LIMIT:
        return None
    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
