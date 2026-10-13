import random
from collections import Counter, defaultdict

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
PINK = 6
ORANGE = 7
TEAL = 8
MAROON = 9

CONNECTOR_COLORS = (RED, BLUE, GREEN, YELLOW)
CONNECTOR_PRIORITY = {RED: 0, BLUE: 1, GREEN: 2, YELLOW: 3}
BODY_COLOR_POOL = (BLUE, RED, GREEN, YELLOW, GREY, ORANGE, TEAL, MAROON)
DIR_ORDER = ("U", "R", "D", "L")
DIR_DELTA = {"U": (-1, 0), "R": (0, 1), "D": (1, 0), "L": (0, -1)}
OPPOSITE = {"U": "D", "D": "U", "L": "R", "R": "L"}
MAX_ATTEMPTS = 400


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_bbox(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                pts.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _paste(dst: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    h, w = patch.shape
    dst[r0 : r0 + h, c0 : c0 + w] = patch


def _dominant_body_color(tile: np.ndarray) -> int:
    vals = [int(v) for v in tile.flat if int(v) not in (BLACK, PINK)]
    if not vals:
        raise AssertionError("tile must contain body pixels")
    counts = Counter(vals)
    top = counts.most_common(2)
    if len(top) > 1:
        assert top[0][1] > top[1][1], "body color must be uniquely dominant"
    return int(top[0][0])


def _has_anchor_marker(tile: np.ndarray) -> bool:
    h, w = tile.shape
    for r in range(h - 1):
        for c in range(w - 1):
            if np.all(tile[r : r + 2, c : c + 2] == PINK):
                return True
    return False


def _connector_side_and_offset(tile: np.ndarray, r: int, c: int) -> tuple[str, int]:
    h, w = tile.shape
    if r == 0 and 0 < c < w - 1:
        return "U", int(c)
    if r == h - 1 and 0 < c < w - 1:
        return "D", int(c)
    if c == 0 and 0 < r < h - 1:
        return "L", int(r)
    if c == w - 1 and 0 < r < h - 1:
        return "R", int(r)
    raise AssertionError("connector must lie on a non-corner border cell")


def _extract_tile_info(tile: np.ndarray) -> dict:
    body_color = _dominant_body_color(tile)
    connectors: dict[int, dict] = {}
    h, w = tile.shape
    for r in range(h):
        for c in range(w):
            color = int(tile[r, c])
            if color not in CONNECTOR_COLORS or color == body_color:
                continue
            if r not in (0, h - 1) and c not in (0, w - 1):
                raise AssertionError("connector colors may only appear on tile borders")
            side, offset = _connector_side_and_offset(tile, r, c)
            assert color not in connectors, "connector color must be unique within a tile"
            connectors[color] = {"side": side, "offset": offset, "coord": (int(r), int(c))}

    return {
        "tile": tile.copy(),
        "h": int(h),
        "w": int(w),
        "body_color": body_color,
        "connectors": connectors,
        "is_anchor": _has_anchor_marker(tile),
    }


def _parse_grid(grid: np.ndarray) -> tuple[list[dict], dict[int, list[int]]]:
    g = np.asarray(grid, dtype=int)
    comps = _components_mask(g != BLACK)
    tiles: list[dict] = []
    for coords in comps:
        bbox = _bbox_coords(coords)
        tile = _crop_bbox(g, bbox)
        info = _extract_tile_info(tile)
        info["bbox"] = bbox
        tiles.append(info)

    anchors = [i for i, info in enumerate(tiles) if info["is_anchor"]]
    assert len(anchors) == 1, "expected exactly one anchor tile"

    color_to_tiles: dict[int, list[int]] = defaultdict(list)
    for i, info in enumerate(tiles):
        assert 1 <= len(info["connectors"]) <= 3, "every tile must expose 1-3 connectors"
        for color in info["connectors"]:
            color_to_tiles[color].append(i)

    for color, owners in color_to_tiles.items():
        assert len(owners) == 2, f"connector color {color} must appear on exactly two tiles"
        a = tiles[owners[0]]["connectors"][color]["side"]
        b = tiles[owners[1]]["connectors"][color]["side"]
        assert OPPOSITE[a] == b, "paired connectors must lie on opposite edges"

    assert len(color_to_tiles) == len(tiles) - 1, "connector pairs must form a tree edge set"
    return tiles, color_to_tiles


def _assemble_tiles(tiles: list[dict], color_to_tiles: dict[int, list[int]]) -> np.ndarray:
    n = len(tiles)
    anchor = next(i for i, info in enumerate(tiles) if info["is_anchor"])
    placed = {anchor}
    placements = {anchor: (0, 0)}
    used_colors: set[int] = set()
    occupied: dict[tuple[int, int], int] = {}

    def add_tile(tile_id: int) -> None:
        r0, c0 = placements[tile_id]
        tile = tiles[tile_id]["tile"]
        for rr in range(tile.shape[0]):
            for cc in range(tile.shape[1]):
                val = int(tile[rr, cc])
                if val == BLACK:
                    continue
                pos = (r0 + rr, c0 + cc)
                prev = occupied.get(pos)
                if prev is None:
                    occupied[pos] = val
                else:
                    assert prev == val, "overlapping tiles may only agree on the same connector color"

    add_tile(anchor)

    while len(placed) < n:
        chosen_color = None
        chosen_parent = None
        chosen_child = None
        for color in sorted(color_to_tiles, key=lambda x: CONNECTOR_PRIORITY[x]):
            if color in used_colors:
                continue
            owners = color_to_tiles[color]
            placed_owners = [tile_id for tile_id in owners if tile_id in placed]
            pending_owners = [tile_id for tile_id in owners if tile_id not in placed]
            if len(placed_owners) == 1 and len(pending_owners) == 1:
                chosen_color = color
                chosen_parent = placed_owners[0]
                chosen_child = pending_owners[0]
                break
        assert chosen_color is not None, "no legal frontier connector found"

        parent_conn = tiles[chosen_parent]["connectors"][chosen_color]["coord"]
        child_conn = tiles[chosen_child]["connectors"][chosen_color]["coord"]
        pr0, pc0 = placements[chosen_parent]
        placements[chosen_child] = (pr0 + parent_conn[0] - child_conn[0], pc0 + parent_conn[1] - child_conn[1])
        add_tile(chosen_child)
        used_colors.add(chosen_color)
        placed.add(chosen_child)

    assert len(used_colors) == n - 1, "tree assembly must use every connector pair"
    rs = [r for r, _ in occupied]
    cs = [c for _, c in occupied]
    out = np.zeros((max(rs) - min(rs) + 1, max(cs) - min(cs) + 1), dtype=int)
    r_shift, c_shift = -min(rs), -min(cs)
    for (r, c), val in occupied.items():
        out[r + r_shift, c + c_shift] = int(val)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Crop scattered sprite tiles and reassemble them by matching border connector colors.

    Input structure:
    - Black background surrounds 4-5 separated multicolor 4-connected sprite tiles.
    - Each tile has one dominant body color, 1-3 border connector pixels in red/blue/green/yellow, and one unique anchor tile containing a 2x2 pink marker block.
    - Every connector color appears on exactly two different tiles, and the two occurrences lie on opposite bbox edges.

    Rule:
    - Crop each sprite to its minimal bounding box and treat it as an independent tile.
    - Start from the anchor tile and repeatedly inspect the current assembly frontier.
    - Among frontier connector colors, choose the earliest in the fixed order red -> blue -> green -> yellow, then translate the matching unplaced tile so the two same-colored connector pixels coincide.
    - After all tiles are attached, output the assembled structure cropped to its minimal bounding box.

    Key insight:
    - The input positions are irrelevant noise; only the cropped tile shapes plus connector color, side, and offset determine the final topology.
    """
    tiles, color_to_tiles = _parse_grid(np.asarray(grid, dtype=int))
    return _assemble_tiles(tiles, color_to_tiles)


def _tree_template(tile_count: int) -> tuple[list[tuple[int, int, int]], list[int]]:
    colors = random.sample(CONNECTOR_COLORS, tile_count - 1)
    sorted_colors = sorted(colors, key=lambda x: CONNECTOR_PRIORITY[x])
    root_color = random.choice(sorted_colors[1:])
    hidden_color = random.choice([c for c in sorted_colors if CONNECTOR_PRIORITY[c] < CONNECTOR_PRIORITY[root_color]])
    remaining = [c for c in colors if c not in {root_color, hidden_color}]
    random.shuffle(remaining)

    edges = [(0, 1, root_color), (1, 2, hidden_color)]
    if tile_count == 4:
        edges.append((0, 3, remaining[0]))
    else:
        edges.append((0, 3, remaining[0]))
        edges.append((random.choice((1, 3)), 4, remaining[1]))
    return edges, colors


def _children_from_edges(tile_count: int, edges: list[tuple[int, int, int]]) -> dict[int, list[int]]:
    children: dict[int, list[int]] = {i: [] for i in range(tile_count)}
    for parent, child, _color in edges:
        children[parent].append(child)
    return children


def _assign_directions(tile_count: int, edges: list[tuple[int, int, int]]) -> tuple[dict[tuple[int, int], str], dict[int, tuple[int, int]]] | None:
    children = _children_from_edges(tile_count, edges)
    parent_of = {child: parent for parent, child, _ in edges}
    edge_index = {(parent, child): color for parent, child, color in edges}
    order = [child for _parent, child, _color in edges]
    used_sides = [set() for _ in range(tile_count)]
    positions = {0: (0, 0)}
    side_by_edge: dict[tuple[int, int], str] = {}
    occupied = {(0, 0)}

    def rec(k: int) -> bool:
        if k == len(order):
            return True
        child = order[k]
        parent = parent_of[child]
        pr, pc = positions[parent]
        dirs = list(DIR_ORDER)
        random.shuffle(dirs)
        for side in dirs:
            if side in used_sides[parent]:
                continue
            if OPPOSITE[side] in used_sides[child]:
                continue
            nr, nc = pr + DIR_DELTA[side][0], pc + DIR_DELTA[side][1]
            if (nr, nc) in occupied:
                continue
            used_sides[parent].add(side)
            used_sides[child].add(OPPOSITE[side])
            side_by_edge[(parent, child)] = side
            positions[child] = (nr, nc)
            occupied.add((nr, nc))
            if len(DIR_ORDER) - len(used_sides[child]) >= len(children[child]) and rec(k + 1):
                return True
            occupied.remove((nr, nc))
            positions.pop(child)
            side_by_edge.pop((parent, child))
            used_sides[parent].remove(side)
            used_sides[child].remove(OPPOSITE[side])
        return False

    if rec(0):
        assert set(edge_index) == set(side_by_edge)
        return side_by_edge, positions
    return None


def _assign_body_colors(tile_count: int, incident_colors: list[set[int]]) -> list[int] | None:
    remaining = list(BODY_COLOR_POOL)
    random.shuffle(remaining)
    chosen: list[int] = []
    for i in range(tile_count):
        candidates = [color for color in remaining if color not in incident_colors[i]]
        if not candidates:
            return None
        color = random.choice(candidates)
        remaining.remove(color)
        chosen.append(color)
    return chosen


def _is_connected(mask: np.ndarray) -> bool:
    comps = _components_mask(mask)
    return len(comps) == 1


def _trim_optional_cells(mask: np.ndarray, protected: set[tuple[int, int]], max_remove: int) -> np.ndarray:
    h, w = mask.shape
    candidates = [(r, c) for r in range(1, h - 1) for c in range(1, w - 1) if mask[r, c] and (r, c) not in protected]
    random.shuffle(candidates)
    removed = 0
    for r, c in candidates:
        if removed >= max_remove:
            break
        mask[r, c] = False
        if not _is_connected(mask):
            mask[r, c] = True
            continue
        removed += 1
    return mask


def _make_tile(h: int, w: int, body_color: int, connector_offsets: dict[str, int], is_anchor: bool) -> np.ndarray | None:
    tile = np.zeros((h, w), dtype=int)
    body = np.zeros((h, w), dtype=bool)
    body[1 : h - 1, 1 : w - 1] = True

    protected: set[tuple[int, int]] = set()
    for side in DIR_ORDER:
        if side in connector_offsets:
            continue
        if side in ("U", "D"):
            offset = random.randint(1, w - 2)
            if side == "U":
                body[0, offset] = True
                protected.add((1, offset))
            else:
                body[h - 1, offset] = True
                protected.add((h - 2, offset))
        else:
            offset = random.randint(1, h - 2)
            if side == "L":
                body[offset, 0] = True
                protected.add((offset, 1))
            else:
                body[offset, w - 1] = True
                protected.add((offset, w - 2))

    for side, offset in connector_offsets.items():
        if side == "U":
            protected.add((1, offset))
        elif side == "D":
            protected.add((h - 2, offset))
        elif side == "L":
            protected.add((offset, 1))
        else:
            protected.add((offset, w - 2))

    tile[body] = int(body_color)
    if is_anchor:
        marker_r = random.randint(1, h - 3)
        marker_c = random.randint(1, w - 3)
        tile[marker_r : marker_r + 2, marker_c : marker_c + 2] = PINK
        protected.update(
            {
                (marker_r, marker_c),
                (marker_r + 1, marker_c),
                (marker_r, marker_c + 1),
                (marker_r + 1, marker_c + 1),
            }
        )

    body = tile != BLACK
    body = _trim_optional_cells(body.copy(), protected, max_remove=2 if h == 5 and w == 5 else 1)
    tile = np.where(body, tile, BLACK)

    for side, offset in connector_offsets.items():
        if side == "U":
            tile[0, offset] = body_color
            if tile[1, offset] == BLACK:
                return None
        elif side == "D":
            tile[h - 1, offset] = body_color
            if tile[h - 2, offset] == BLACK:
                return None
        elif side == "L":
            tile[offset, 0] = body_color
            if tile[offset, 1] == BLACK:
                return None
        else:
            tile[offset, w - 1] = body_color
            if tile[offset, w - 2] == BLACK:
                return None

    if np.any(tile[:, 0] != BLACK) and np.any(tile[:, w - 1] != BLACK) and np.any(tile[0] != BLACK) and np.any(tile[h - 1] != BLACK):
        if _is_connected(tile != BLACK):
            counts = Counter(int(v) for v in tile.flat if int(v) != BLACK)
            body_count = counts[body_color]
            pink_count = counts[PINK]
            if body_count > pink_count and body_count > len(connector_offsets):
                return tile
    return None


def _build_tiles(tile_count: int, edges: list[tuple[int, int, int]], side_by_edge: dict[tuple[int, int], str]) -> tuple[list[np.ndarray], list[dict]] | None:
    connectors_by_tile: list[dict[str, int]] = [dict() for _ in range(tile_count)]
    incident_colors = [set() for _ in range(tile_count)]
    for parent, child, color in edges:
        parent_side = side_by_edge[(parent, child)]
        child_side = OPPOSITE[parent_side]
        assert parent_side not in connectors_by_tile[parent]
        assert child_side not in connectors_by_tile[child]
        connectors_by_tile[parent][parent_side] = color
        connectors_by_tile[child][child_side] = color
        incident_colors[parent].add(color)
        incident_colors[child].add(color)

    body_colors = _assign_body_colors(tile_count, incident_colors)
    if body_colors is None:
        return None

    tiles: list[np.ndarray] = []
    infos: list[dict] = []
    for i in range(tile_count):
        h = w = 5 if i == 0 else random.choice((4, 5))
        offset_map: dict[str, int] = {}
        for side in connectors_by_tile[i]:
            offset_map[side] = random.randint(1, (w - 2) if side in ("U", "D") else (h - 2))
        tile = _make_tile(h, w, body_colors[i], offset_map, i == 0)
        if tile is None:
            return None
        for side, color in connectors_by_tile[i].items():
            offset = offset_map[side]
            if side == "U":
                tile[0, offset] = int(color)
            elif side == "D":
                tile[h - 1, offset] = int(color)
            elif side == "L":
                tile[offset, 0] = int(color)
            else:
                tile[offset, w - 1] = int(color)
        info = _extract_tile_info(tile)
        expected_sides = {color: side for side, color in connectors_by_tile[i].items()}
        if {color: data["side"] for color, data in info["connectors"].items()} != expected_sides:
            return None
        tiles.append(tile)
        infos.append(info)
    return tiles, infos


def _placement_from_tree(tiles: list[np.ndarray], edges: list[tuple[int, int, int]]) -> dict[int, tuple[int, int]] | None:
    infos = [_extract_tile_info(tile) for tile in tiles]
    placement = {0: (0, 0)}
    for parent, child, color in edges:
        parent_coord = infos[parent]["connectors"][color]["coord"]
        child_coord = infos[child]["connectors"][color]["coord"]
        pr0, pc0 = placement[parent]
        placement[child] = (pr0 + parent_coord[0] - child_coord[0], pc0 + parent_coord[1] - child_coord[1])

    cell_sets: list[set[tuple[int, int]]] = []
    color_maps: list[dict[tuple[int, int], int]] = []
    for tile_id, tile in enumerate(tiles):
        r0, c0 = placement[tile_id]
        cells = set()
        cmap = {}
        for r in range(tile.shape[0]):
            for c in range(tile.shape[1]):
                val = int(tile[r, c])
                if val == BLACK:
                    continue
                pos = (r0 + r, c0 + c)
                cells.add(pos)
                cmap[pos] = val
        cell_sets.append(cells)
        color_maps.append(cmap)

    edge_positions = {}
    for parent, child, color in edges:
        pr0, pc0 = placement[parent]
        cr0, cc0 = placement[child]
        pcoord = infos[parent]["connectors"][color]["coord"]
        ccoord = infos[child]["connectors"][color]["coord"]
        ppos = (pr0 + pcoord[0], pc0 + pcoord[1])
        cpos = (cr0 + ccoord[0], cc0 + ccoord[1])
        if ppos != cpos:
            return None
        edge_positions[(parent, child)] = ppos

    for i in range(len(tiles)):
        for j in range(i + 1, len(tiles)):
            overlap = cell_sets[i] & cell_sets[j]
            if not overlap:
                continue
            allowed = {pos for (a, b), pos in edge_positions.items() if {a, b} == {i, j}}
            if overlap != allowed:
                return None
            for pos in overlap:
                if color_maps[i][pos] != color_maps[j][pos]:
                    return None

    all_rs = [pos[0] for cells in cell_sets for pos in cells]
    all_cs = [pos[1] for cells in cell_sets for pos in cells]
    if max(all_rs) - min(all_rs) + 1 > 30 or max(all_cs) - min(all_cs) + 1 > 30:
        return None
    return placement


def _render_tiles(tiles: list[np.ndarray], placement: dict[int, tuple[int, int]]) -> np.ndarray:
    cells: dict[tuple[int, int], int] = {}
    for tile_id, tile in enumerate(tiles):
        r0, c0 = placement[tile_id]
        for r in range(tile.shape[0]):
            for c in range(tile.shape[1]):
                val = int(tile[r, c])
                if val == BLACK:
                    continue
                cells[(r0 + r, c0 + c)] = val
    rs = [r for r, _ in cells]
    cs = [c for _, c in cells]
    out = np.zeros((max(rs) - min(rs) + 1, max(cs) - min(cs) + 1), dtype=int)
    for (r, c), val in cells.items():
        out[r - min(rs), c - min(cs)] = val
    return out


def _normalized_positions(placement: dict[int, tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    pts = list(placement.values())
    r0 = min(r for r, _ in pts)
    c0 = min(c for _, c in pts)
    return tuple(sorted((r - r0, c - c0) for r, c in pts))


def _scatter_tiles(tiles: list[np.ndarray], output_positions: dict[int, tuple[int, int]]) -> np.ndarray | None:
    min_h = max(22, max(tile.shape[0] for tile in tiles) * 2 + 4)
    min_w = max(22, max(tile.shape[1] for tile in tiles) * 2 + 4)
    h = random.randint(min_h, 30)
    w = random.randint(min_w, 30)
    grid = np.zeros((h, w), dtype=int)
    rects: list[tuple[int, int, int, int]] = []
    placement: dict[int, tuple[int, int]] = {}
    order = list(range(len(tiles)))
    random.shuffle(order)

    def ok(r0: int, c0: int, hh: int, ww: int) -> bool:
        r1, c1 = r0 + hh - 1, c0 + ww - 1
        for or0, oc0, or1, oc1 in rects:
            if not (r1 + 1 < or0 or or1 + 1 < r0 or c1 + 1 < oc0 or oc1 + 1 < c0):
                return False
        return True

    for tile_id in order:
        tile = tiles[tile_id]
        hh, ww = tile.shape
        candidates = [(r0, c0) for r0 in range(h - hh + 1) for c0 in range(w - ww + 1)]
        random.shuffle(candidates)
        placed = False
        for r0, c0 in candidates[:300]:
            if not ok(r0, c0, hh, ww):
                continue
            _paste(grid, tile, r0, c0)
            rects.append((r0, c0, r0 + hh - 1, c0 + ww - 1))
            placement[tile_id] = (r0, c0)
            placed = True
            break
        if not placed:
            return None

    if _normalized_positions(placement) == _normalized_positions(output_positions):
        return None
    return grid


def _validate_instance(inp: np.ndarray, expected: np.ndarray, edges: list[tuple[int, int, int]]) -> bool:
    tiles, color_to_tiles = _parse_grid(inp)
    if len(tiles) not in (4, 5):
        return False
    if set(color_to_tiles) != {color for _p, _c, color in edges}:
        return False
    if not np.array_equal(main(inp), expected):
        return False

    root_frontier = {color for parent, _child, color in edges if parent == 0}
    hidden = {color for parent, _child, color in edges if parent != 0}
    if not root_frontier or not hidden:
        return False
    if min(CONNECTOR_PRIORITY[c] for c in hidden) >= max(CONNECTOR_PRIORITY[c] for c in root_frontier):
        return False
    if len(root_frontier) < 2:
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    tile_count = random.choice((4, 5))
    edges, _colors = _tree_template(tile_count)
    direction_data = _assign_directions(tile_count, edges)
    if direction_data is None:
        return None
    side_by_edge, _coarse_positions = direction_data

    built = _build_tiles(tile_count, edges, side_by_edge)
    if built is None:
        return None
    tiles, _infos = built

    output_positions = _placement_from_tree(tiles, edges)
    if output_positions is None:
        return None
    expected = _render_tiles(tiles, output_positions)
    if max(expected.shape) > 30:
        return None

    inp = _scatter_tiles(tiles, output_positions)
    if inp is None:
        return None
    if not _validate_instance(inp, expected, edges):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid connector-assembly instance")
