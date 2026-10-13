from __future__ import annotations

from collections import deque
import random

import numpy as np


_BG = 0
_COLORS = tuple(range(1, 10))
_NEIGHBORS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
_GEN_RETRIES = 500


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _components(grid: np.ndarray) -> tuple[list[dict], np.ndarray]:
    h, w = grid.shape
    labels = np.full((h, w), -1, dtype=int)
    comps: list[dict] = []

    for sr in range(h):
        for sc in range(w):
            if int(grid[sr, sc]) == _BG or labels[sr, sc] != -1:
                continue

            color = int(grid[sr, sc])
            cid = len(comps)
            q: deque[tuple[int, int]] = deque([(sr, sc)])
            labels[sr, sc] = cid
            coords: list[tuple[int, int]] = []

            while q:
                r, c = q.popleft()
                coords.append((r, c))
                for dr, dc in _NEIGHBORS4:
                    nr, nc = r + dr, c + dc
                    if not _in_bounds((h, w), nr, nc):
                        continue
                    if labels[nr, nc] != -1 or int(grid[nr, nc]) != color:
                        continue
                    labels[nr, nc] = cid
                    q.append((nr, nc))

            arr = np.asarray(coords, dtype=int)
            r0, c0 = arr.min(axis=0)
            r1, c1 = arr.max(axis=0)
            comps.append(
                {
                    "id": cid,
                    "color": color,
                    "coords": arr,
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    "area": int(arr.shape[0]),
                }
            )

    return comps, labels


def _adjacencies(labels: np.ndarray, n: int) -> list[set[int]]:
    h, w = labels.shape
    adj = [set() for _ in range(n)]
    for r in range(h):
        for c in range(w):
            a = int(labels[r, c])
            if a < 0:
                continue
            for dr, dc in ((1, 0), (0, 1)):
                nr, nc = r + dr, c + dc
                if not _in_bounds((h, w), nr, nc):
                    continue
                b = int(labels[nr, nc])
                if b >= 0 and b != a:
                    adj[a].add(b)
                    adj[b].add(a)
    return adj


def _line_component(comp: dict) -> bool:
    r0, c0, r1, c1 = comp["bbox"]
    height = r1 - r0 + 1
    width = c1 - c0 + 1
    return int(comp["area"]) == height * width and (height == 1 or width == 1)


def _label_at(labels: np.ndarray, r: int, c: int) -> int | None:
    if not _in_bounds(labels.shape, r, c):
        return None
    cid = int(labels[r, c])
    return cid if cid >= 0 else None


def _bridge_connections(
    comps: list[dict], labels: np.ndarray, adj: list[set[int]]
) -> list[tuple[int, int, int, int]]:
    connections: list[tuple[int, int, int, int]] = []

    for comp in comps:
        if not _line_component(comp):
            continue

        cid = int(comp["id"])
        color = int(comp["color"])
        r0, c0, r1, c1 = comp["bbox"]
        candidates: list[tuple[int, int, int, int]] = []

        if r0 == r1:
            left = _label_at(labels, r0, c0 - 1)
            right = _label_at(labels, r0, c1 + 1)
            if left is not None and right is not None and left != right:
                if adj[cid] == {left, right}:
                    candidates.append((cid, left, right, color))

        if c0 == c1:
            up = _label_at(labels, r0 - 1, c0)
            down = _label_at(labels, r1 + 1, c0)
            if up is not None and down is not None and up != down:
                if adj[cid] == {up, down}:
                    candidates.append((cid, up, down, color))

        if len(candidates) == 1:
            connections.append(candidates[0])

    return connections


def _object_key(comp: dict) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = comp["bbox"]
    return int(r0), int(c0), int(r1), int(c1)


def _patch(comp: dict) -> np.ndarray:
    r0, c0, r1, c1 = comp["bbox"]
    patch = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=int)
    coords = np.asarray(comp["coords"], dtype=int)
    patch[coords[:, 0] - r0, coords[:, 1] - c0] = int(comp["color"])
    return patch


def _pack_objects(
    ordered: list[int], separators: list[int], comps: list[dict]
) -> np.ndarray:
    patches = [_patch(comps[cid]) for cid in ordered]
    out_h = max(int(p.shape[0]) for p in patches)
    out_w = sum(int(p.shape[1]) for p in patches) + len(separators)
    assert out_h <= 30 and out_w <= 30

    out = np.zeros((out_h, out_w), dtype=int)
    col = 0
    for i, patch in enumerate(patches):
        ph, pw = patch.shape
        row = out_h - ph
        mask = patch != _BG
        out[row : row + ph, col : col + pw][mask] = patch[mask]
        col += pw
        if i < len(separators):
            out[:, col] = int(separators[i])
            col += 1
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rebuild the bridge-defined object chain as a compact bottom-aligned row.

    Input structure:
    - The background is black.
    - Non-black monochrome small objects are scattered as separate 4-connected blobs.
    - Straight one-cell-wide horizontal or vertical bridge components use colors
      different from the two objects they connect; every object has one or two
      bridge connections, forming a single chain.

    Rule:
    - Find monochrome connected components, classify straight components that
      touch exactly two other components at opposite ends as bridges, and use
      those bridge colors as the edge labels between the remaining object blobs.
    - Start from the endpoint object whose bounding box is uppermost, then
      leftmost, and traverse the chain to get the left-to-right object order.
    - Output height is the maximum object height; output width is the sum of
      object widths plus one separator column between neighboring objects.
    - Copy each object's original cropped shape into the output in traversal
      order, bottom-aligned, and fill each separator column with the color of
      the bridge that connected that object pair.

    Key insight:
    - Bridge pixels are structural edges and colored separators; the visible
      chain topology, not numeric color order, determines the packing order.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert max(g.shape) <= 30
    assert int(np.min(g)) >= 0 and int(np.max(g)) <= 9

    comps, labels = _components(g)
    assert len(comps) >= 3
    adj = _adjacencies(labels, len(comps))
    bridge_edges = _bridge_connections(comps, labels, adj)
    bridge_ids = {bid for bid, _a, _b, _color in bridge_edges}
    object_ids = {comp["id"] for comp in comps if int(comp["id"]) not in bridge_ids}
    assert len(object_ids) >= 2

    graph: dict[int, list[tuple[int, int]]] = {oid: [] for oid in object_ids}
    used_bridges: set[int] = set()
    for bid, a, b, color in bridge_edges:
        if a not in object_ids or b not in object_ids:
            continue
        assert int(comps[a]["color"]) != int(color)
        assert int(comps[b]["color"]) != int(color)
        graph[a].append((b, int(color)))
        graph[b].append((a, int(color)))
        used_bridges.add(bid)

    assert used_bridges == bridge_ids
    degrees = {oid: len(edges) for oid, edges in graph.items()}
    assert all(deg in (1, 2) for deg in degrees.values())
    endpoints = [oid for oid, deg in degrees.items() if deg == 1]
    assert len(endpoints) == 2

    start = min(endpoints, key=lambda oid: _object_key(comps[oid]))
    ordered = [start]
    separators: list[int] = []
    seen = {start}
    prev: int | None = None
    cur = start

    while True:
        choices = [(nxt, color) for nxt, color in graph[cur] if nxt != prev]
        if not choices:
            break
        assert len(choices) == 1
        nxt, color = choices[0]
        assert nxt not in seen
        separators.append(int(color))
        ordered.append(nxt)
        seen.add(nxt)
        prev, cur = cur, nxt

    assert seen == object_ids
    assert len(separators) == len(ordered) - 1
    return _pack_objects(ordered, separators, comps)


_BASE_SHAPES = (
    np.array([[1, 0, 0], [1, 1, 0], [0, 0, 0]], dtype=bool),
    np.array([[0, 1, 0], [1, 1, 1], [0, 0, 0]], dtype=bool),
    np.array([[0, 0, 0], [0, 1, 1], [0, 1, 1]], dtype=bool),
    np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool),
    np.array([[1, 1, 0], [0, 1, 1], [0, 0, 0]], dtype=bool),
    np.array([[1, 1, 0], [1, 1, 0], [1, 0, 0]], dtype=bool),
    np.array([[1, 0, 1], [1, 1, 1], [0, 0, 0]], dtype=bool),
)


def _random_shape() -> np.ndarray:
    shape = random.choice(_BASE_SHAPES).copy()
    shape = np.rot90(shape, random.randrange(4))
    if random.random() < 0.5:
        shape = np.fliplr(shape)
    assert bool(shape[1, 1])
    assert 3 <= int(shape.sum()) <= 6
    return shape


def _straight_path(n: int) -> list[tuple[int, int]]:
    if random.random() < 0.5:
        path = [(0, c) for c in range(n)]
    else:
        path = [(r, 0) for r in range(n)]
    if random.random() < 0.5:
        path.reverse()
    return path


def _turning_path(n: int) -> list[tuple[int, int]] | None:
    rows = random.randint(3, 5)
    cols = random.randint(3, 5)
    for _ in range(200):
        start = (random.randrange(rows), random.randrange(cols))
        path = [start]
        seen = {start}
        for _step in range(n - 1):
            r, c = path[-1]
            choices = [
                (r + dr, c + dc)
                for dr, dc in _NEIGHBORS4
                if 0 <= r + dr < rows
                and 0 <= c + dc < cols
                and (r + dr, c + dc) not in seen
            ]
            if not choices:
                break
            nxt = random.choice(choices)
            path.append(nxt)
            seen.add(nxt)
        if len(path) != n:
            continue
        orientations = {
            "H" if path[i][0] == path[i + 1][0] else "V"
            for i in range(n - 1)
        }
        if len(orientations) != 2:
            continue
        min_r = min(r for r, _c in path)
        min_c = min(c for _r, c in path)
        return [(r - min_r, c - min_c) for r, c in path]
    return None


def _shape_cells(center: tuple[int, int], shape: np.ndarray) -> set[tuple[int, int]]:
    cr, cc = center
    return {
        (int(cr + r - 1), int(cc + c - 1))
        for r, c in np.argwhere(shape)
    }


def _bridge_cells_between(
    cells_a: set[tuple[int, int]],
    cells_b: set[tuple[int, int]],
    center_a: tuple[int, int],
    center_b: tuple[int, int],
) -> set[tuple[int, int]] | None:
    ra, ca = center_a
    rb, cb = center_b
    if ra == rb:
        row = int(ra)
        if ca < cb:
            left, right = cells_a, cells_b
        else:
            left, right = cells_b, cells_a
        left_cols = [c for r, c in left if r == row]
        right_cols = [c for r, c in right if r == row]
        if not left_cols or not right_cols:
            return None
        start = max(left_cols) + 1
        end = min(right_cols) - 1
        if start > end:
            return None
        return {(row, c) for c in range(start, end + 1)}

    if ca == cb:
        col = int(ca)
        if ra < rb:
            top, bottom = cells_a, cells_b
        else:
            top, bottom = cells_b, cells_a
        top_rows = [r for r, c in top if c == col]
        bottom_rows = [r for r, c in bottom if c == col]
        if not top_rows or not bottom_rows:
            return None
        start = max(top_rows) + 1
        end = min(bottom_rows) - 1
        if start > end:
            return None
        return {(r, col) for r in range(start, end + 1)}

    return None


def _choose_padding(needed: int) -> tuple[int, int] | None:
    spare = 30 - needed
    if spare < 2:
        return None
    before = random.randint(1, min(4, spare - 1))
    after = random.randint(1, min(4, spare - before))
    return before, after


def _shift_cells(
    objects: list[set[tuple[int, int]]],
    bridges: list[set[tuple[int, int]]],
    dr: int,
    dc: int,
) -> tuple[list[set[tuple[int, int]]], list[set[tuple[int, int]]]]:
    shifted_objects = [
        {(int(r + dr), int(c + dc)) for r, c in cells} for cells in objects
    ]
    shifted_bridges = [
        {(int(r + dr), int(c + dc)) for r, c in cells} for cells in bridges
    ]
    return shifted_objects, shifted_bridges


def _patch_from_cells(cells: set[tuple[int, int]], color: int) -> np.ndarray:
    rows = [r for r, _c in cells]
    cols = [c for _r, c in cells]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    patch = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=int)
    for r, c in cells:
        patch[r - r0, c - c0] = int(color)
    return patch


def _expected_output(
    objects: list[set[tuple[int, int]]],
    object_colors: list[int],
    bridge_colors: list[int],
) -> np.ndarray:
    first_key = (
        min(r for r, _c in objects[0]),
        min(c for _r, c in objects[0]),
        max(r for r, _c in objects[0]),
        max(c for _r, c in objects[0]),
    )
    last_key = (
        min(r for r, _c in objects[-1]),
        min(c for _r, c in objects[-1]),
        max(r for r, _c in objects[-1]),
        max(c for _r, c in objects[-1]),
    )
    if first_key <= last_key:
        order = list(range(len(objects)))
        separators = list(bridge_colors)
    else:
        order = list(reversed(range(len(objects))))
        separators = list(reversed(bridge_colors))

    patches = [_patch_from_cells(objects[i], object_colors[i]) for i in order]
    out_h = max(int(p.shape[0]) for p in patches)
    out_w = sum(int(p.shape[1]) for p in patches) + len(separators)
    out = np.zeros((out_h, out_w), dtype=int)
    col = 0
    for i, patch in enumerate(patches):
        ph, pw = patch.shape
        row = out_h - ph
        mask = patch != _BG
        out[row : row + ph, col : col + pw][mask] = patch[mask]
        col += pw
        if i < len(separators):
            out[:, col] = int(separators[i])
            col += 1
    return out


def _gen_attempt() -> np.ndarray | None:
    n_objects = random.randint(3, 6)
    if random.random() < 0.45:
        path = _straight_path(n_objects)
    else:
        path = _turning_path(n_objects)
        if path is None:
            return None

    row_spacing = random.randint(4, 6)
    col_spacing = random.randint(4, 6)
    centers = [(r * row_spacing, c * col_spacing) for r, c in path]

    shapes = [_random_shape() for _ in range(n_objects)]
    object_cells = [_shape_cells(center, shape) for center, shape in zip(centers, shapes)]
    object_colors = [random.choice(_COLORS) for _ in range(n_objects)]
    bridge_colors = [
        random.choice(
            [c for c in _COLORS if c not in {object_colors[i], object_colors[i + 1]}]
        )
        for i in range(n_objects - 1)
    ]

    occupied = set().union(*object_cells)
    if sum(len(cells) for cells in object_cells) != len(occupied):
        return None

    bridge_cells: list[set[tuple[int, int]]] = []
    for i in range(n_objects - 1):
        cells = _bridge_cells_between(
            object_cells[i], object_cells[i + 1], centers[i], centers[i + 1]
        )
        if cells is None or cells & occupied:
            return None
        bridge_cells.append(cells)
        occupied |= cells

    all_cells = set(occupied)
    min_r = min(r for r, _c in all_cells)
    max_r = max(r for r, _c in all_cells)
    min_c = min(c for _r, c in all_cells)
    max_c = max(c for _r, c in all_cells)
    needed_h = max_r - min_r + 1
    needed_w = max_c - min_c + 1

    row_pad = _choose_padding(needed_h)
    col_pad = _choose_padding(needed_w)
    if row_pad is None or col_pad is None:
        return None
    top_pad, bottom_pad = row_pad
    left_pad, right_pad = col_pad
    height = needed_h + top_pad + bottom_pad
    width = needed_w + left_pad + right_pad
    if max(height, width) > 30:
        return None

    dr = top_pad - min_r
    dc = left_pad - min_c
    object_cells, bridge_cells = _shift_cells(object_cells, bridge_cells, dr, dc)

    grid = np.zeros((height, width), dtype=int)
    for cells, color in zip(object_cells, object_colors):
        for r, c in cells:
            grid[r, c] = int(color)
    for cells, color in zip(bridge_cells, bridge_colors):
        for r, c in cells:
            if int(grid[r, c]) != _BG:
                return None
            grid[r, c] = int(color)

    expected = _expected_output(object_cells, object_colors, bridge_colors)
    if max(expected.shape) > 30:
        return None

    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"Failed to generate valid input after {_GEN_RETRIES} attempts")
