import random
from typing import Iterable

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
GREY = 5
ORANGE = 7
LINE_COLORS = (BLUE, RED, GREEN, ORANGE)
NEIGHBORS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
CANVAS_LIMIT = 30
GEN_RETRIES = 400


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
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for dr, dc in NEIGHBORS4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: Iterable[tuple[int, int]]) -> tuple[int, int, int, int]:
    pts = list(coords)
    rows = [r for r, _ in pts]
    cols = [c for _, c in pts]
    return min(rows), min(cols), max(rows), max(cols)


def _find_objects(grid: np.ndarray) -> tuple[list[dict], np.ndarray]:
    objects: list[dict] = []
    obj_index = np.full(grid.shape, -1, dtype=int)
    for comp in _components(grid == GREY):
        r0, c0, r1, c1 = _bbox(comp)
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        if h != w or h not in (2, 3) or len(comp) != h * w:
            raise ValueError("grey components must be solid 2x2 or 3x3 squares")
        idx = len(objects)
        coords = tuple(comp)
        for r, c in coords:
            obj_index[r, c] = idx
        objects.append({"coords": coords, "bbox": (r0, c0, r1, c1)})
    return objects, obj_index


def _parse_scene(grid: np.ndarray) -> tuple[list[dict], list[dict], list[int]]:
    objects, obj_index = _find_objects(grid)
    h, w = grid.shape
    edges_by_color: dict[int, list[dict]] = {}
    object_colors = [set() for _ in objects]

    for color in sorted(int(v) for v in np.unique(grid) if int(v) not in (BLACK, GREY)):
        edges: list[dict] = []
        for comp in _components(grid == color):
            r0, c0, r1, c1 = _bbox(comp)
            if r0 != r1 and c0 != c1:
                raise ValueError("line components must be straight")

            touched: set[int] = set()
            for r, c in comp:
                for dr, dc in NEIGHBORS4:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and obj_index[nr, nc] >= 0:
                        touched.add(int(obj_index[nr, nc]))
            if len(touched) != 2:
                raise ValueError("each line must connect exactly two objects")

            a, b = sorted(touched)
            edges.append({"a": a, "b": b, "pixels": tuple(comp)})
            object_colors[a].add(color)
            object_colors[b].add(color)
        if edges:
            edges_by_color[color] = edges

    for colors in object_colors:
        if len(colors) > 1:
            raise ValueError("an object cannot belong to multiple line colors")

    groups: list[dict] = []
    for color, edges in edges_by_color.items():
        adjacency: dict[int, set[int]] = {}
        for edge in edges:
            adjacency.setdefault(edge["a"], set()).add(edge["b"])
            adjacency.setdefault(edge["b"], set()).add(edge["a"])

        seen: set[int] = set()
        for start in adjacency:
            if start in seen:
                continue
            stack = [start]
            comp_nodes: list[int] = []
            while stack:
                node = stack.pop()
                if node in seen:
                    continue
                seen.add(node)
                comp_nodes.append(node)
                for nxt in adjacency[node]:
                    if nxt not in seen:
                        stack.append(nxt)
            comp_set = set(comp_nodes)
            comp_edges = [edge for edge in edges if edge["a"] in comp_set and edge["b"] in comp_set]
            groups.append(
                {
                    "color": color,
                    "objects": tuple(sorted(comp_nodes)),
                    "edges": tuple(comp_edges),
                }
            )

    isolated = [idx for idx, colors in enumerate(object_colors) if not colors]
    return objects, groups, isolated


def main(grid: np.ndarray) -> np.ndarray:
    """
    Group grey rectangles by the color of the lines linking them, then recolor each group by its size.

    Input structure:
    - Background is black.
    - Foreground objects are solid grey 2x2 or 3x3 square components.
    - Red, blue, green, or orange one-pixel straight lines connect pairs of grey objects horizontally or vertically.

    Rule:
    - Treat same-color line connections as a graph and split the objects into connected groups by line color.
    - Erase every input line.
    - For a group of size 2, recolor both objects to the line color and redraw that group's original line paths.
    - For a group of size 3, recolor the objects to the line color and draw a one-pixel hollow bounding rectangle around the three objects.
    - For a group of size 4 or more, recolor every object in the group and fill the group's full bounding rectangle with that color.
    - Any grey object touched by no line stays grey.

    Key insight:
    The line color is not decoration: it defines a connectivity class on the grey objects, and the number of objects in each same-color connected component selects one of three geometric actions.
    """
    g = np.asarray(grid, dtype=int)
    objects, groups, isolated = _parse_scene(g)

    out = np.full_like(g, BLACK)
    for idx in isolated:
        for r, c in objects[idx]["coords"]:
            out[r, c] = GREY

    for group in groups:
        color = int(group["color"])
        members = list(group["objects"])
        coords = [pt for idx in members for pt in objects[idx]["coords"]]
        r0, c0, r1, c1 = _bbox(coords)
        n = len(members)

        if n >= 4:
            out[r0 : r1 + 1, c0 : c1 + 1] = color
            continue

        for r, c in coords:
            out[r, c] = color

        if n == 2:
            for edge in group["edges"]:
                for r, c in edge["pixels"]:
                    out[r, c] = color
        elif n == 3:
            out[r0, c0 : c1 + 1] = color
            out[r1, c0 : c1 + 1] = color
            out[r0 : r1 + 1, c0] = color
            out[r0 : r1 + 1, c1] = color

    return out


def _transform_cells(
    cells: list[tuple[int, int]],
    edges: list[tuple[tuple[int, int], tuple[int, int]]],
    mode: int,
) -> tuple[list[tuple[int, int]], list[tuple[tuple[int, int], tuple[int, int]]]]:
    def xf(cell: tuple[int, int]) -> tuple[int, int]:
        r, c = cell
        if mode == 0:
            return r, c
        if mode == 1:
            return r, -c
        if mode == 2:
            return -r, c
        if mode == 3:
            return -r, -c
        if mode == 4:
            return c, r
        if mode == 5:
            return c, -r
        if mode == 6:
            return -c, r
        return -c, -r

    new_cells = [xf(cell) for cell in cells]
    row_shift = -min(r for r, _ in new_cells)
    col_shift = -min(c for _, c in new_cells)

    def norm(cell: tuple[int, int]) -> tuple[int, int]:
        return cell[0] + row_shift, cell[1] + col_shift

    cell_map = {cell: norm(xf(cell)) for cell in cells}
    out_cells = [cell_map[cell] for cell in cells]
    out_edges = [(cell_map[a], cell_map[b]) for a, b in edges]
    return out_cells, out_edges


def _random_tree_cells(n: int, rng: random.Random) -> tuple[list[tuple[int, int]], list[tuple[tuple[int, int], tuple[int, int]]]]:
    for _ in range(200):
        cells = [(0, 0)]
        used = {(0, 0)}
        edges: list[tuple[tuple[int, int], tuple[int, int]]] = []
        while len(cells) < n:
            frontier: list[tuple[tuple[int, int], tuple[int, int]]] = []
            for cell in cells:
                for dr, dc in NEIGHBORS4:
                    nxt = (cell[0] + dr, cell[1] + dc)
                    if nxt in used:
                        continue
                    rows = [r for r, _ in used] + [nxt[0]]
                    cols = [c for _, c in used] + [nxt[1]]
                    if max(rows) - min(rows) <= 2 and max(cols) - min(cols) <= 2:
                        frontier.append((cell, nxt))
            if not frontier:
                break
            parent, nxt = rng.choice(frontier)
            used.add(nxt)
            cells.append(nxt)
            edges.append((parent, nxt))
        if len(cells) == n:
            return _transform_cells(cells, edges, rng.randrange(8))
    raise RuntimeError("failed to sample a compact connected layout")


def _make_group_patch(n: int, color: int, rng: random.Random) -> np.ndarray:
    cells, edges = _random_tree_cells(n, rng)
    step = rng.choice((5, 6))
    sizes = {cell: rng.choice((2, 3)) for cell in cells}
    positions = {cell: (1 + cell[0] * step, 1 + cell[1] * step) for cell in cells}

    max_r = max(positions[cell][0] + sizes[cell] - 1 for cell in cells)
    max_c = max(positions[cell][1] + sizes[cell] - 1 for cell in cells)
    patch = np.full((max_r + 2, max_c + 2), BLACK, dtype=int)

    for cell in cells:
        r0, c0 = positions[cell]
        size = sizes[cell]
        patch[r0 : r0 + size, c0 : c0 + size] = GREY

    for a, b in edges:
        ra, ca = positions[a]
        rb, cb = positions[b]
        sa = sizes[a]
        sb = sizes[b]
        if a[0] == b[0]:
            if ca > cb:
                ra, ca, sa, rb, cb, sb = rb, cb, sb, ra, ca, sa
            row_lo = max(ra, rb)
            row_hi = min(ra + sa - 1, rb + sb - 1)
            row = rng.randint(row_lo, row_hi)
            patch[row, ca + sa : cb] = color
        else:
            if ra > rb:
                ra, ca, sa, rb, cb, sb = rb, cb, sb, ra, ca, sa
            col_lo = max(ca, cb)
            col_hi = min(ca + sa - 1, cb + sb - 1)
            col = rng.randint(col_lo, col_hi)
            patch[ra + sa : rb, col] = color
    return patch


def _make_isolated_patch(rng: random.Random) -> np.ndarray:
    size = rng.choice((2, 3))
    patch = np.full((size + 2, size + 2), BLACK, dtype=int)
    patch[1 : 1 + size, 1 : 1 + size] = GREY
    return patch


def _place_patches(patches: list[np.ndarray], rng: random.Random) -> np.ndarray | None:
    canvas = np.full((CANVAS_LIMIT, CANVAS_LIMIT), BLACK, dtype=int)
    blocked = np.zeros_like(canvas, dtype=bool)
    gap = 2

    order = list(range(len(patches)))
    rng.shuffle(order)
    order.sort(key=lambda idx: patches[idx].shape[0] * patches[idx].shape[1], reverse=True)

    for idx in order:
        patch = patches[idx]
        ph, pw = patch.shape
        placed = False
        for _ in range(500):
            r = rng.randint(0, CANVAS_LIMIT - ph)
            c = rng.randint(0, CANVAS_LIMIT - pw)
            if blocked[r : r + ph, c : c + pw].any():
                continue
            mask = patch != BLACK
            view = canvas[r : r + ph, c : c + pw]
            view[mask] = patch[mask]
            r0 = max(0, r - gap)
            c0 = max(0, c - gap)
            r1 = min(CANVAS_LIMIT, r + ph + gap)
            c1 = min(CANVAS_LIMIT, c + pw + gap)
            blocked[r0:r1, c0:c1] = True
            placed = True
            break
        if not placed:
            return None

    pts = np.argwhere(canvas != BLACK)
    if pts.size == 0:
        return None
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    top = rng.randint(0, min(3, int(r0)))
    left = rng.randint(0, min(3, int(c0)))
    bottom = rng.randint(0, min(3, int(CANVAS_LIMIT - 1 - r1)))
    right = rng.randint(0, min(3, int(CANVAS_LIMIT - 1 - c1)))
    return canvas[r0 - top : r1 + bottom + 1, c0 - left : c1 + right + 1]


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    group_sizes = [2, 3, rng.choice((4, 5))]
    colors = list(LINE_COLORS)
    rng.shuffle(colors)
    colors = colors[:3]

    patches = [_make_group_patch(size, color, rng) for size, color in zip(group_sizes, colors)]
    isolated_count = rng.randint(0, 2)
    for _ in range(isolated_count):
        patches.append(_make_isolated_patch(rng))

    grid = _place_patches(patches, rng)
    if grid is None:
        return None

    try:
        objects, groups, isolated = _parse_scene(grid)
    except ValueError:
        return None

    if len(groups) != 3:
        return None
    parsed_sizes = sorted(len(group["objects"]) for group in groups)
    if parsed_sizes != sorted(group_sizes):
        return None
    parsed_colors = {int(group["color"]) for group in groups}
    if parsed_colors != set(colors):
        return None
    if len(isolated) != isolated_count:
        return None
    if max(grid.shape) > CANVAS_LIMIT:
        return None
    if not objects:
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {GEN_RETRIES} attempts")
