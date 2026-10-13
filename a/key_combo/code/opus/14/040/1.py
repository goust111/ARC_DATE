from __future__ import annotations

from collections import deque
from itertools import combinations
import random

import numpy as np


WALL = 8
CORRIDOR = 0
START = 9
SPRITE_COLORS = (1, 2, 3, 6, 7)

GEN_CELL_ROWS = (7, 9)
GEN_CELL_COLS = (7, 9)
GEN_SPRITE_COUNT = (3, 5)
GEN_SHAPE_SIZE = (3, 6)
GEN_ATTEMPTS = 600
GEN_LOCAL_FOOTPRINT = (
    (1, 0),
    (2, 0),
    (3, 0),
    (1, -1),
    (1, 1),
    (2, -1),
    (2, 1),
    (3, -1),
    (3, 1),
)


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _neighbors8(r: int, c: int):
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            yield r + dr, c + dc


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _extract_sprite(grid: np.ndarray, coords: list[tuple[int, int]]) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(coords)
    color = int(grid[coords[0]])
    patch = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=int)
    for r, c in coords:
        patch[r - r0, c - c0] = color
    return patch


def _find_components_by_colors(grid: np.ndarray, colors: set[int]) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[dict] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color not in colors or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if (
                        _in_bounds(grid.shape, nr, nc)
                        and not seen[nr, nc]
                        and int(grid[nr, nc]) == color
                    ):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            r0, c0, r1, c1 = _bbox(coords)
            components.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": (r0, c0, r1, c1),
                    "area": len(coords),
                }
            )
    return components


def _corridor_distances(grid: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
    starts = np.argwhere(grid == START)
    assert len(starts) == 1, "expected exactly one maroon start pixel"
    start = tuple(int(v) for v in starts[0])
    h, w = grid.shape
    assert start[0] in (0, h - 1) or start[1] in (0, w - 1), "start must be on the edge"

    dist = np.full((h, w), -1, dtype=int)
    q: deque[tuple[int, int]] = deque()
    for nr, nc in _neighbors4(*start):
        if _in_bounds(grid.shape, nr, nc) and int(grid[nr, nc]) == CORRIDOR and dist[nr, nc] < 0:
            dist[nr, nc] = 1
            q.append((nr, nc))
    assert q, "start must touch the corridor"

    while q:
        r, c = q.popleft()
        nd = int(dist[r, c]) + 1
        for nr, nc in _neighbors4(r, c):
            if _in_bounds(grid.shape, nr, nc) and int(grid[nr, nc]) == CORRIDOR and dist[nr, nc] < 0:
                dist[nr, nc] = nd
                q.append((nr, nc))
    return dist, start


def _sprite_distance_key(grid: np.ndarray, dist: np.ndarray, coords: list[tuple[int, int]]) -> int:
    options: list[int] = []
    for r, c in coords:
        for nr, nc in _neighbors4(r, c):
            if _in_bounds(grid.shape, nr, nc) and int(grid[nr, nc]) == CORRIDOR and dist[nr, nc] >= 0:
                options.append(int(dist[nr, nc]))
    assert options, "every sprite must touch the reachable corridor"
    return min(options)


def main(grid: np.ndarray) -> np.ndarray:
    """Arrange the corridor-end sprites by increasing maze distance from the entrance.

    Input structure:
    - Black cells are maze corridors, teal cells are walls, and one maroon edge pixel marks the maze entrance.
    - Three to five monochrome 4-connected sprites of distinct non-wall colors sit at different dead ends and touch the corridor by side adjacency.
    - The corridor network is connected; walls and sprites are not traversable.

    Rule:
    - Start from the maroon entrance and run 4-connected BFS through black corridor cells only.
    - For each sprite, inspect the corridor cells side-adjacent to it and use the smallest BFS distance among those cells as that sprite's ordering key.
    - Sort the sprites by that distance, crop each sprite to its bounding box with black treated as transparent, and place the cropped sprites left to right with one black spacer column between neighbors and all bottoms aligned.
    - Omit the maze walls, corridors, and entrance from the output.

    Key insight:
    - The visible placement order is controlled by topological path distance through the maze, not by a sprite's color, area, or geometric position in the input.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    dist, _start = _corridor_distances(g)

    sprite_colors = {int(color) for color in np.unique(g)} - {CORRIDOR, WALL, START}
    sprites = _find_components_by_colors(g, sprite_colors)
    assert 3 <= len(sprites) <= 5, "expected three to five sprites"
    assert len({sprite["color"] for sprite in sprites}) == len(sprites), "sprite colors must be unique"

    ordered: list[tuple[int, np.ndarray]] = []
    keys: list[int] = []
    for sprite in sprites:
        key = _sprite_distance_key(g, dist, sprite["coords"])
        keys.append(key)
        ordered.append((key, _extract_sprite(g, sprite["coords"])))
    assert len(set(keys)) == len(keys), "sprite distance keys must be unique"

    ordered.sort(key=lambda item: item[0])
    out_h = max(sprite.shape[0] for _key, sprite in ordered)
    out_w = sum(sprite.shape[1] for _key, sprite in ordered) + len(ordered) - 1
    out = np.zeros((out_h, out_w), dtype=int)
    col = 0
    for _key, sprite in ordered:
        sh, sw = sprite.shape
        window = out[out_h - sh : out_h, col : col + sw]
        mask = sprite != CORRIDOR
        window[mask] = sprite[mask]
        col += sw + 1
    return out


def _generate_maze_grid(cell_rows: int, cell_cols: int) -> np.ndarray:
    h = 2 * cell_rows + 1
    w = 2 * cell_cols + 1
    grid = np.full((h, w), WALL, dtype=int)

    visited = np.zeros((cell_rows, cell_cols), dtype=bool)

    def carve(cr: int, cc: int) -> None:
        visited[cr, cc] = True
        grid[2 * cr + 1, 2 * cc + 1] = CORRIDOR
        dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        random.shuffle(dirs)
        for dr, dc in dirs:
            nr, nc = cr + dr, cc + dc
            if 0 <= nr < cell_rows and 0 <= nc < cell_cols and not visited[nr, nc]:
                grid[2 * cr + 1 + dr, 2 * cc + 1 + dc] = CORRIDOR
                carve(nr, nc)

    carve(random.randrange(cell_rows), random.randrange(cell_cols))
    return grid


def _entrance_candidates(cell_rows: int, cell_cols: int) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    h = 2 * cell_rows + 1
    w = 2 * cell_cols + 1
    options: list[tuple[tuple[int, int], tuple[int, int]]] = []
    for cc in range(cell_cols):
        col = 2 * cc + 1
        options.append(((0, col), (1, col)))
        options.append(((h - 1, col), (h - 2, col)))
    for rr in range(cell_rows):
        row = 2 * rr + 1
        options.append(((row, 0), (row, 1)))
        options.append(((row, w - 1), (row, w - 2)))
    return options


def _corridor_degree(grid: np.ndarray, pos: tuple[int, int]) -> int:
    r, c = pos
    return sum(
        1
        for nr, nc in _neighbors4(r, c)
        if _in_bounds(grid.shape, nr, nc) and int(grid[nr, nc]) == CORRIDOR
    )


def _leaf_infos(grid: np.ndarray, dist: np.ndarray, start: tuple[int, int]) -> list[dict]:
    leaves: list[dict] = []
    for r, c in np.argwhere(grid == CORRIDOR):
        rr, cc = int(r), int(c)
        if any((nr, nc) == start for nr, nc in _neighbors4(rr, cc)):
            continue
        if _corridor_degree(grid, (rr, cc)) != 1:
            continue
        parent = next(
            (nr, nc)
            for nr, nc in _neighbors4(rr, cc)
            if _in_bounds(grid.shape, nr, nc) and int(grid[nr, nc]) == CORRIDOR
        )
        outward = (rr - parent[0], cc - parent[1])
        locals_to_abs: dict[tuple[int, int], tuple[int, int]] = {}
        valid_locals: set[tuple[int, int]] = set()
        pr, pc = outward[1], -outward[0]
        for forward, lateral in GEN_LOCAL_FOOTPRINT:
            ar = rr + forward * outward[0] + lateral * pr
            ac = cc + forward * outward[1] + lateral * pc
            loc = (forward, lateral)
            locals_to_abs[loc] = (ar, ac)
            if not _in_bounds(grid.shape, ar, ac) or int(grid[ar, ac]) != WALL:
                continue
            corridor_neighbors = {
                (nr, nc)
                for nr, nc in _neighbors4(ar, ac)
                if _in_bounds(grid.shape, nr, nc) and int(grid[nr, nc]) == CORRIDOR
            }
            if corridor_neighbors <= {(rr, cc)}:
                valid_locals.add(loc)

        root = (1, 0)
        if root not in valid_locals:
            continue

        graph = {loc: set() for loc in valid_locals}
        for loc, abs_pos in locals_to_abs.items():
            if loc not in valid_locals:
                continue
            for other, other_abs in locals_to_abs.items():
                if other not in valid_locals or other == loc:
                    continue
                if abs(abs_pos[0] - other_abs[0]) + abs(abs_pos[1] - other_abs[1]) == 1:
                    graph[loc].add(other)

        reachable = set()
        q = deque([root])
        while q:
            loc = q.popleft()
            if loc in reachable:
                continue
            reachable.add(loc)
            for nxt in graph[loc]:
                if nxt not in reachable:
                    q.append(nxt)
        max_size = min(GEN_SHAPE_SIZE[1], len(reachable))
        if max_size < GEN_SHAPE_SIZE[0]:
            continue

        leaves.append(
            {
                "leaf": (rr, cc),
                "dist": int(dist[rr, cc]),
                "outward": outward,
                "root": root,
                "graph": graph,
                "locals_to_abs": locals_to_abs,
                "reachable": reachable,
                "max_size": max_size,
            }
        )
    return leaves


def _touches_existing(cell: tuple[int, int], occupied: set[tuple[int, int]]) -> bool:
    if cell in occupied:
        return True
    return any(nei in occupied for nei in _neighbors8(*cell))


def _grow_shape(info: dict, target_size: int, occupied: set[tuple[int, int]]) -> list[tuple[int, int]] | None:
    root = info["root"]
    locals_to_abs = info["locals_to_abs"]
    graph = info["graph"]
    if _touches_existing(locals_to_abs[root], occupied):
        return None

    chosen = {root}
    while len(chosen) < target_size:
        frontier = [
            loc
            for loc in info["reachable"]
            if loc not in chosen
            and any(nei in chosen for nei in graph[loc])
            and not _touches_existing(locals_to_abs[loc], occupied)
        ]
        if not frontier:
            return None
        chosen.add(random.choice(frontier))
    coords = [locals_to_abs[loc] for loc in chosen]
    return coords


def _order_signature(items: list[dict], key) -> tuple[int, ...]:
    return tuple(item["id"] for item in sorted(items, key=key))


def _select_leaves(leaves: list[dict], start: tuple[int, int]) -> list[dict] | None:
    if len(leaves) < GEN_SPRITE_COUNT[0]:
        return None

    indexed = []
    for idx, info in enumerate(leaves):
        row, col = info["leaf"]
        indexed.append(
            {
                **info,
                "id": idx,
                "rowcol": (row, col),
                "colrow": (col, row),
                "manhattan": abs(row - start[0]) + abs(col - start[1]),
            }
        )

    max_count = min(GEN_SPRITE_COUNT[1], len(indexed))
    counts = list(range(GEN_SPRITE_COUNT[0], max_count + 1))
    random.shuffle(counts)
    for count in counts:
        combos = list(combinations(indexed, count))
        random.shuffle(combos)
        for chosen in combos[:120]:
            picked = list(chosen)
            dists = [item["dist"] for item in picked]
            if len(set(dists)) != count:
                continue
            bfs_order = _order_signature(picked, key=lambda item: (item["dist"], item["id"]))
            if bfs_order == _order_signature(picked, key=lambda item: (item["manhattan"], item["id"])):
                continue
            if bfs_order == _order_signature(picked, key=lambda item: (item["rowcol"], item["id"])):
                continue
            if bfs_order == _order_signature(picked, key=lambda item: (item["colrow"], item["id"])):
                continue
            return sorted(picked, key=lambda item: (item["dist"], item["id"]))
    return None


def _assign_sizes(leaves: list[dict]) -> list[int] | None:
    for _ in range(80):
        sizes = [random.randint(GEN_SHAPE_SIZE[0], info["max_size"]) for info in leaves]
        if len(set(sizes)) == len(sizes):
            area_order = tuple(
                idx for idx, _size in sorted(enumerate(sizes), key=lambda item: (item[1], item[0]))
            )
            if area_order == tuple(range(len(leaves))):
                continue
        return sizes
    return None


def _paint_sprites(
    base_grid: np.ndarray, leaves: list[dict], sizes: list[int]
) -> list[dict] | None:
    grid = base_grid.copy()
    colors = random.sample(list(SPRITE_COLORS), len(leaves))
    occupied: set[tuple[int, int]] = set()
    sprites: list[dict | None] = [None] * len(leaves)

    order = sorted(range(len(leaves)), key=lambda idx: (-sizes[idx], leaves[idx]["dist"]))
    for idx in order:
        coords = None
        for _ in range(50):
            coords = _grow_shape(leaves[idx], sizes[idx], occupied)
            if coords is not None:
                break
        if coords is None:
            return None
        color = colors[idx]
        for r, c in coords:
            grid[r, c] = color
            occupied.add((r, c))
        sprites[idx] = {
            "coords": coords,
            "color": color,
            "dist": leaves[idx]["dist"],
            "leaf": leaves[idx]["leaf"],
        }
    return [sprite for sprite in sprites if sprite is not None], grid


def _expected_output_from_sprites(sprites: list[dict]) -> np.ndarray:
    ordered = []
    for sprite in sprites:
        r0, c0, r1, c1 = _bbox(sprite["coords"])
        patch = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=int)
        for r, c in sprite["coords"]:
            patch[r - r0, c - c0] = int(sprite["color"])
        ordered.append((int(sprite["dist"]), patch))
    ordered.sort(key=lambda item: item[0])

    out_h = max(sprite.shape[0] for _dist, sprite in ordered)
    out_w = sum(sprite.shape[1] for _dist, sprite in ordered) + len(ordered) - 1
    out = np.zeros((out_h, out_w), dtype=int)
    col = 0
    for _dist, sprite in ordered:
        sh, sw = sprite.shape
        mask = sprite != 0
        out[out_h - sh : out_h, col : col + sw][mask] = sprite[mask]
        col += sw + 1
    return out


def _tree_and_sprite_checks(grid: np.ndarray, expected: np.ndarray) -> bool:
    if grid.ndim != 2 or max(grid.shape) > 30 or max(expected.shape) > 30:
        return False

    dist, start = _corridor_distances(grid)

    corridor = grid == CORRIDOR
    corridor_cells = [tuple(int(v) for v in rc) for rc in np.argwhere(corridor)]
    if not corridor_cells:
        return False
    seen = set()
    q = deque([corridor_cells[0]])
    while q:
        cur = q.popleft()
        if cur in seen:
            continue
        seen.add(cur)
        for nxt in _neighbors4(*cur):
            if _in_bounds(grid.shape, *nxt) and corridor[nxt] and nxt not in seen:
                q.append(nxt)
    if len(seen) != len(corridor_cells):
        return False

    edge_count = 0
    for r, c in corridor_cells:
        for nr, nc in ((r + 1, c), (r, c + 1)):
            if _in_bounds(grid.shape, nr, nc) and corridor[nr, nc]:
                edge_count += 1
    if edge_count != len(corridor_cells) - 1:
        return False

    if sum(1 for nr, nc in _neighbors4(*start) if _in_bounds(grid.shape, nr, nc) and grid[nr, nc] == CORRIDOR) != 1:
        return False

    sprite_colors = set(int(color) for color in np.unique(grid)) - {CORRIDOR, WALL, START}
    sprites = _find_components_by_colors(grid, sprite_colors)
    if not (3 <= len(sprites) <= 5):
        return False
    if len({sprite["color"] for sprite in sprites}) != len(sprites):
        return False

    keys = []
    areas = []
    leaves = []
    for sprite in sprites:
        if not (GEN_SHAPE_SIZE[0] <= sprite["area"] <= GEN_SHAPE_SIZE[1]):
            return False
        adjacent_corridor = {
            (nr, nc)
            for r, c in sprite["coords"]
            for nr, nc in _neighbors4(r, c)
            if _in_bounds(grid.shape, nr, nc) and int(grid[nr, nc]) == CORRIDOR
        }
        if not adjacent_corridor:
            return False
        key = _sprite_distance_key(grid, dist, sprite["coords"])
        keys.append(key)
        areas.append(sprite["area"])
        if len(adjacent_corridor) != 1:
            return False
        leaf = next(iter(adjacent_corridor))
        if _corridor_degree(grid, leaf) != 1:
            return False
        leaves.append(leaf)
    if len(set(keys)) != len(keys):
        return False
    if len(set(leaves)) != len(leaves):
        return False

    bfs_order = tuple(idx for idx, _key in sorted(enumerate(keys), key=lambda item: (item[1], item[0])))
    manhattan_order = tuple(
        idx
        for idx, _leaf in sorted(
            enumerate(leaves),
            key=lambda item: (abs(item[1][0] - start[0]) + abs(item[1][1] - start[1]), item[0]),
        )
    )
    if bfs_order == manhattan_order:
        return False
    if len(set(areas)) == len(areas):
        area_order = tuple(idx for idx, _area in sorted(enumerate(areas), key=lambda item: (item[1], item[0])))
        if bfs_order == area_order:
            return False

    solver_out = main(grid)
    if not np.array_equal(solver_out, expected):
        return False
    return True


def _generate_attempt() -> np.ndarray | None:
    cell_rows = random.randint(*GEN_CELL_ROWS)
    cell_cols = random.randint(*GEN_CELL_COLS)
    base = _generate_maze_grid(cell_rows, cell_cols)

    entrance, first_corridor = random.choice(_entrance_candidates(cell_rows, cell_cols))
    if base[first_corridor] != CORRIDOR:
        return None
    base[entrance] = START

    dist, start = _corridor_distances(base)
    leaves = _leaf_infos(base, dist, start)
    chosen_leaves = _select_leaves(leaves, start)
    if chosen_leaves is None:
        return None

    sizes = _assign_sizes(chosen_leaves)
    if sizes is None:
        return None

    painted = _paint_sprites(base, chosen_leaves, sizes)
    if painted is None:
        return None
    sprites, grid = painted
    expected = _expected_output_from_sprites(sprites)
    if not _tree_and_sprite_checks(grid, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid maze-distance sprite-ordering input")
