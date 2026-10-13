from __future__ import annotations

from importlib.machinery import SourceFileLoader
from pathlib import Path

import numpy as np

try:
    from arc_utils import components_mask, get_bg, neighbors4, paint_coords
except Exception:
    _ARC_UTILS = SourceFileLoader(
        "_arc_utils_fable_2_048_1",
        "/home/lixinyu/snap/new/solvers/arc_utils.py",
    ).load_module()
    components_mask = _ARC_UTILS.components_mask
    get_bg = _ARC_UTILS.get_bg
    neighbors4 = _ARC_UTILS.neighbors4
    paint_coords = _ARC_UTILS.paint_coords


BG = 0
RED = 2
GREEN = 3
GREY = 5
MARKER_POOL = (1, 4, 6, 7, 8, 9)
CANVAS = 30
RECT_SHAPES = ((2, 2), (2, 3), (3, 2), (3, 3), (2, 4), (4, 2), (3, 4), (4, 3))
MAX_ATTEMPTS = 400

UP = (-1, 0)
DOWN = (1, 0)
LEFT = (0, -1)
RIGHT = (0, 1)
DIRS = (UP, DOWN, LEFT, RIGHT)
OPPOSITE = {UP: DOWN, DOWN: UP, LEFT: RIGHT, RIGHT: LEFT}


def _in_bounds(r: int, c: int, shape: tuple[int, int]) -> bool:
    return 0 <= int(r) < int(shape[0]) and 0 <= int(c) < int(shape[1])


def _coord_set(coords: np.ndarray) -> set[tuple[int, int]]:
    return {tuple(map(int, rc)) for rc in coords}


def _marker_palette(grid: np.ndarray, bg: int) -> list[tuple[int, int, int]]:
    markers: list[tuple[int, int, int]] = []
    special = {int(bg), RED, GREEN, GREY}
    for color in np.unique(grid):
        color = int(color)
        if color in special:
            continue
        coords = np.argwhere(grid == color)
        assert coords.shape == (1, 2), "each palette color must be a singleton marker"
        r, c = map(int, coords[0])
        markers.append((r, c, color))
    assert markers, "expected a diagonal palette"
    markers.sort(key=lambda item: (item[0], item[1]))
    for (r0, c0, _), (r1, c1, _) in zip(markers, markers[1:]):
        assert (r1, c1) == (r0 + 1, c0 + 1), "palette markers must form a contiguous down-right diagonal"
    return markers


def _analyze(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = int(get_bg(g))
    markers = _marker_palette(g, bg)

    blob_coords = components_mask(g == GREY, connectivity=4)
    n_blobs = len(blob_coords)
    assert n_blobs == len(markers), "palette length must equal the blob count"

    blob_of = np.full(g.shape, -1, dtype=int)
    blobs = []
    for blob_id, coords in enumerate(blob_coords):
        coords = coords.astype(int)
        blob_of[coords[:, 0], coords[:, 1]] = int(blob_id)
        rows = coords[:, 0]
        cols = coords[:, 1]
        blobs.append(
            {
                "coords": coords,
                "coord_set": _coord_set(coords),
                "bbox": (
                    int(rows.min()),
                    int(cols.min()),
                    int(rows.max()),
                    int(cols.max()),
                ),
                "ports": [],
            }
        )

    edges: list[tuple[int, int]] = []
    used_red: set[tuple[int, int]] = set()
    green_coords = np.argwhere(g == GREEN)
    assert green_coords.size > 0, "expected at least one green port"

    for r, c in green_coords:
        r = int(r)
        c = int(c)
        touching = []
        for nr, nc in neighbors4(r, c):
            nr = int(nr)
            nc = int(nc)
            if _in_bounds(nr, nc, g.shape) and int(blob_of[nr, nc]) >= 0:
                touching.append((int(blob_of[nr, nc]), nr, nc))
        assert len(touching) == 1, "each green port must touch exactly one grey blob"
        src, br, bc = touching[0]
        direction = (r - br, c - bc)
        assert direction in DIRS, "port direction must be axis-aligned"

        rr = r + direction[0]
        cc = c + direction[1]
        assert _in_bounds(rr, cc, g.shape) and int(g[rr, cc]) == RED, "the cell beyond each port must start a red ray"

        ray_cells = []
        while _in_bounds(rr, cc, g.shape) and int(g[rr, cc]) == RED:
            ray_cells.append((rr, cc))
            rr += direction[0]
            cc += direction[1]

        assert ray_cells, "each port must emit a non-empty ray"
        assert _in_bounds(rr, cc, g.shape) and int(blob_of[rr, cc]) >= 0, "each ray must end on another grey blob"
        dst = int(blob_of[rr, cc])
        assert dst != src, "self-loops are not allowed"

        edges.append((src, dst))
        blobs[src]["ports"].append({"coord": (r, c), "direction": direction, "dst": dst})
        for cell in ray_cells:
            assert cell not in used_red, "red rays must be disjoint"
            used_red.add(cell)

    assert len(used_red) == int(np.count_nonzero(g == RED)), "all red pixels must belong to exactly one port ray"

    adjacency = [[] for _ in range(n_blobs)]
    indegree = [0] * n_blobs
    for src, dst in edges:
        adjacency[src].append(dst)
        indegree[dst] += 1

    roots = [i for i, deg in enumerate(indegree) if deg == 0]
    assert len(roots) == 1, "there must be exactly one sourceless origin"
    assert all(deg <= 1 for deg in indegree), "each non-root blob must have exactly one parent"
    assert len(edges) == n_blobs - 1, "the directed ray network must be a connected tree"

    root = roots[0]
    depths = {root: 0}
    queue = [root]
    for src in queue:
        for dst in adjacency[src]:
            assert dst not in depths, "directed cycles are not allowed"
            depths[dst] = depths[src] + 1
            queue.append(dst)

    assert len(depths) == n_blobs, "all blobs must be reachable from the origin"
    assert max(depths.values()) < len(markers), "the palette must be long enough for every depth"

    return {
        "bg": bg,
        "markers": markers,
        "palette": [color for _r, _c, color in markers],
        "blobs": blobs,
        "edges": edges,
        "depths": depths,
        "root": root,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover the directed ray cascade and recolor each grey blob by its causal depth.

    Input structure:
    - The background is black. Several separated grey connected blobs appear in the scene.
    - Some blobs have a single green port pixel attached to one side, and a straight red ray continues outward from that port until it reaches another blob.
    - A diagonal run of singleton marker pixels forms a depth palette from upper-left to lower-right.

    Rule:
    - Read every green port as the emitting end of its aligned red ray; the first grey blob touched after that red run is the ray's receiver, which defines a directed edge.
    - Exactly one blob has no incoming edge; it is depth 0, and every other blob's depth is its directed distance from that origin.
    - Read the diagonal marker colors in order as the colors for depth 0, 1, 2, ... and recolor each grey blob with the color of its depth.
    - Keep the red rays, green ports, palette markers, and background unchanged.

    Key insight:
    - The red network only shows who is connected; the green port side of each ray reveals causality, which is what determines the unique start blob and the final recoloring palette index.
    """
    g = np.asarray(grid, dtype=int)
    scene = _analyze(g)
    out = g.copy()
    for blob_id, blob in enumerate(scene["blobs"]):
        color = int(scene["palette"][scene["depths"][blob_id]])
        paint_coords(out, blob["coords"], color)
    return out


def _draw_blob(grid: np.ndarray, top: int, left: int, h: int, w: int) -> None:
    grid[top : top + h, left : left + w] = GREY


def _can_place_blob(grid: np.ndarray, top: int, left: int, h: int, w: int, reserve: int) -> bool:
    if top < 0 or left < 0 or top + h > CANVAS or left + w > CANVAS:
        return False
    for r in range(top - 1, top + h + 1):
        for c in range(left - 1, left + w + 1):
            if not _in_bounds(r, c, grid.shape):
                continue
            if r < reserve and c < reserve:
                return False
            if int(grid[r, c]) != BG:
                return False
    return True


def _edge_geometry(
    src: dict,
    child_shape: tuple[int, int],
    direction: tuple[int, int],
    src_offset: int,
    dst_offset: int,
    ray_len: int,
) -> tuple[tuple[int, int], tuple[int, int], list[tuple[int, int]]]:
    pr = int(src["r"])
    pc = int(src["c"])
    ph = int(src["h"])
    pw = int(src["w"])
    ch, cw = map(int, child_shape)

    if direction == RIGHT:
        src_r = pr + src_offset
        child_top = src_r - dst_offset
        child_left = pc + pw + ray_len + 1
        port = (src_r, pc + pw)
        ray = [(src_r, cc) for cc in range(port[1] + 1, child_left)]
    elif direction == LEFT:
        src_r = pr + src_offset
        child_top = src_r - dst_offset
        child_left = pc - cw - ray_len - 1
        port = (src_r, pc - 1)
        ray = [(src_r, cc) for cc in range(port[1] - 1, child_left + cw - 1, -1)]
    elif direction == DOWN:
        src_c = pc + src_offset
        child_top = pr + ph + ray_len + 1
        child_left = src_c - dst_offset
        port = (pr + ph, src_c)
        ray = [(rr, src_c) for rr in range(port[0] + 1, child_top)]
    else:
        src_c = pc + src_offset
        child_top = pr - ch - ray_len - 1
        child_left = src_c - dst_offset
        port = (pr - 1, src_c)
        ray = [(rr, src_c) for rr in range(port[0] - 1, child_top + ch - 1, -1)]

    return (int(child_top), int(child_left)), port, ray


def _can_draw_port_and_ray(
    grid: np.ndarray,
    port: tuple[int, int],
    source_cell: tuple[int, int],
    ray: list[tuple[int, int]],
    reserve: int,
) -> bool:
    if not ray:
        return False
    allowed_for_port = {source_cell, ray[0]}
    allowed_for_ray = set(ray)
    for idx, (r, c) in enumerate(ray):
        if not _in_bounds(r, c, grid.shape):
            return False
        if r < reserve and c < reserve:
            return False
        if int(grid[r, c]) != BG:
            return False
        allowed_neighbors = {ray[idx - 1]} if idx > 0 else {port}
        if idx + 1 < len(ray):
            allowed_neighbors.add(ray[idx + 1])
        for nr, nc in neighbors4(r, c):
            nr = int(nr)
            nc = int(nc)
            if not _in_bounds(nr, nc, grid.shape):
                continue
            if (nr, nc) in allowed_neighbors:
                continue
            if int(grid[nr, nc]) != BG:
                return False

    pr, pc = map(int, port)
    if not _in_bounds(pr, pc, grid.shape):
        return False
    if pr < reserve and pc < reserve:
        return False
    if int(grid[pr, pc]) != BG:
        return False
    for nr, nc in neighbors4(pr, pc):
        nr = int(nr)
        nc = int(nc)
        if not _in_bounds(nr, nc, grid.shape):
            continue
        if (nr, nc) in allowed_for_port:
            continue
        if int(grid[nr, nc]) != BG:
            return False
    return True


def _center(blob: dict) -> tuple[float, float]:
    coords = blob["coords"]
    return (float(coords[:, 0].mean()), float(coords[:, 1].mean()))


def _passes_counterexample_checks(scene: dict) -> bool:
    depth_order = sorted(scene["depths"], key=scene["depths"].get)
    blobs = scene["blobs"]
    centers = {i: _center(blob) for i, blob in enumerate(blobs)}
    if set(scene["depths"].values()) != set(range(len(blobs))):
        return False

    def order_by(key):
        return [i for i in sorted(range(len(blobs)), key=lambda idx: key(idx))]

    simple_orders = (
        order_by(lambda idx: (centers[idx][0], centers[idx][1])),
        order_by(lambda idx: (centers[idx][1], centers[idx][0])),
        order_by(lambda idx: (centers[idx][0] + centers[idx][1], centers[idx][0])),
        order_by(lambda idx: (centers[idx][0] - centers[idx][1], centers[idx][0])),
    )
    if any(order == depth_order for order in simple_orders):
        return False

    direction_seq = []
    child_by_src = {src: dst for src, dst in scene["edges"]}
    for src in depth_order[:-1]:
        port = scene["blobs"][src]["ports"][0]
        direction_seq.append(port["direction"])
        assert child_by_src[src] == depth_order[scene["depths"][src] + 1]
    if len(set(direction_seq)) < 2:
        return False
    return True


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    n_blobs = int(rng.integers(3, min(len(MARKER_POOL), 6) + 1))
    reserve = n_blobs + 3
    palette = list(map(int, rng.choice(np.array(MARKER_POOL, dtype=int), size=n_blobs, replace=False)))

    grid = np.full((CANVAS, CANVAS), BG, dtype=int)
    nodes = []

    root_h, root_w = RECT_SHAPES[int(rng.integers(len(RECT_SHAPES)))]
    row_low = max(reserve + 1, 7)
    row_high = min(CANVAS - root_h - 8, 19)
    col_low = max(reserve + 1, 7)
    col_high = min(CANVAS - root_w - 8, 19)
    if row_low > row_high or col_low > col_high:
        return None
    root_r = int(rng.integers(row_low, row_high + 1))
    root_c = int(rng.integers(col_low, col_high + 1))
    if not _can_place_blob(grid, root_r, root_c, root_h, root_w, reserve):
        return None
    _draw_blob(grid, root_r, root_c, root_h, root_w)
    nodes.append({"r": root_r, "c": root_c, "h": root_h, "w": root_w})

    prev_direction = None
    for _idx in range(1, n_blobs):
        parent = nodes[-1]
        placed = False

        direction_pool = list(DIRS)
        rng.shuffle(direction_pool)
        if prev_direction is not None:
            direction_pool.sort(key=lambda d: d == prev_direction or d == OPPOSITE[prev_direction])

        for direction in direction_pool:
            child_shape = RECT_SHAPES[int(rng.integers(len(RECT_SHAPES)))]
            edge_len_parent = int(parent["h"] if direction in (LEFT, RIGHT) else parent["w"])
            edge_len_child = int(child_shape[0] if direction in (LEFT, RIGHT) else child_shape[1])
            src_offsets = list(range(edge_len_parent))
            dst_offsets = list(range(edge_len_child))
            ray_lengths = list(range(2, 7))
            rng.shuffle(src_offsets)
            rng.shuffle(dst_offsets)
            rng.shuffle(ray_lengths)

            for src_offset in src_offsets:
                for dst_offset in dst_offsets:
                    for ray_len in ray_lengths:
                        child_pos, port, ray = _edge_geometry(
                            parent, child_shape, direction, src_offset, dst_offset, ray_len
                        )
                        child_r, child_c = child_pos
                        child_h, child_w = child_shape
                        if not _can_place_blob(grid, child_r, child_c, child_h, child_w, reserve):
                            continue

                        if direction == RIGHT:
                            source_cell = (parent["r"] + src_offset, parent["c"] + parent["w"] - 1)
                        elif direction == LEFT:
                            source_cell = (parent["r"] + src_offset, parent["c"])
                        elif direction == DOWN:
                            source_cell = (parent["r"] + parent["h"] - 1, parent["c"] + src_offset)
                        else:
                            source_cell = (parent["r"], parent["c"] + src_offset)

                        if not _can_draw_port_and_ray(grid, port, source_cell, ray, reserve):
                            continue

                        grid[port] = GREEN
                        for rr, cc in ray:
                            grid[rr, cc] = RED
                        _draw_blob(grid, child_r, child_c, child_h, child_w)
                        nodes.append({"r": child_r, "c": child_c, "h": child_h, "w": child_w})
                        prev_direction = direction
                        placed = True
                        break
                    if placed:
                        break
                if placed:
                    break
            if placed:
                break
        if not placed:
            return None

    for i, color in enumerate(palette):
        grid[i, i] = int(color)

    used = np.argwhere(grid != BG)
    max_r = int(used[:, 0].max())
    max_c = int(used[:, 1].max())
    extra_bottom = int(rng.integers(0, 3))
    extra_right = int(rng.integers(0, 3))
    cropped = grid[: min(CANVAS, max_r + 1 + extra_bottom), : min(CANVAS, max_c + 1 + extra_right)].copy()

    try:
        scene = _analyze(cropped)
    except AssertionError:
        return None
    if not _passes_counterexample_checks(scene):
        return None
    if not np.array_equal(main(cropped), _expected_from_scene(cropped, scene)):
        return None
    return cropped


def _expected_from_scene(grid: np.ndarray, scene: dict | None = None) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    if scene is None:
        scene = _analyze(g)
    out = g.copy()
    for blob_id, blob in enumerate(scene["blobs"]):
        paint_coords(out, blob["coords"], int(scene["palette"][scene["depths"][blob_id]]))
    return out


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_ATTEMPTS):
        candidate = _gen_attempt(rng)
        if candidate is not None:
            return candidate
    raise RuntimeError("failed to generate a valid causal-depth palette input")
