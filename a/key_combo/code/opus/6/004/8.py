import random
from collections import Counter, deque

import numpy as np


BODY = 5
MARKERS = (1, 2, 3, 4)
SIDES = ("T", "B", "L", "R")
DIRS = {"T": (-1, 0), "B": (1, 0), "L": (0, -1), "R": (0, 1)}
OPPOSITE = {"T": "B", "B": "T", "L": "R", "R": "L"}

MAX_GENERATION_ATTEMPTS = 500
SPRITE_BUILD_ATTEMPTS = 180
SCATTER_ATTEMPTS = 240
PORT_ATTEMPTS = 80
CANVAS_RANGE = (20, 26)
AREA_VALUES = tuple(range(9, 19))
DIM_VALUES = tuple(range(4, 7))
SCATTER_PAD = 2


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.flatten())).most_common(1)[0][0])


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def crop_bbox(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def paste(dst: np.ndarray, patch: np.ndarray, r0: int, c0: int, *, transparent: int | None = None) -> None:
    h, w = patch.shape
    view = dst[r0 : r0 + h, c0 : c0 + w]
    if transparent is None:
        view[:] = patch
    else:
        mask = patch != int(transparent)
        view[mask] = patch[mask]


def neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def find_components(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    out = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) == int(bg):
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) != int(bg):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            arr = np.asarray(coords, dtype=int)
            out.append({"coords": arr, "bbox": bbox_coords(arr), "area": int(arr.shape[0])})
    return out


def tight_crop(grid: np.ndarray, bg: int) -> np.ndarray:
    coords = np.argwhere(grid != int(bg))
    assert coords.size > 0
    return crop_bbox(grid, bbox_coords(coords))


def connector_dir(sprite: np.ndarray, r: int, c: int, bg: int) -> tuple[int, int]:
    dirs = []
    h, w = sprite.shape
    for dr, dc in DIRS.values():
        rr, cc = r + dr, c + dc
        if not (0 <= rr < h and 0 <= cc < w) or int(sprite[rr, cc]) == int(bg):
            dirs.append((dr, dc))
    assert len(dirs) == 1
    return dirs[0]


def occupied_abs(sprite: np.ndarray, base: tuple[int, int], bg: int) -> set[tuple[int, int]]:
    br, bc = base
    return {
        (br + int(r), bc + int(c))
        for r, c in np.argwhere(sprite != int(bg))
    }


def parse_objects(grid: np.ndarray) -> tuple[int, int, set[int], list[dict]]:
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    fg = g[g != int(bg)]
    assert fg.size > 0
    values, counts = np.unique(fg, return_counts=True)
    max_count = int(counts.max())
    body_candidates = [int(v) for v, n in zip(values, counts) if int(n) == max_count]
    assert len(body_candidates) == 1
    body = body_candidates[0]
    marker_colors = {int(v) for v in values if int(v) != int(body)}
    components = find_components(g, bg)
    assert 3 <= len(components) <= 4

    objects = []
    occurrences: dict[int, list[tuple[int, tuple[int, int], tuple[int, int]]]] = {color: [] for color in marker_colors}
    for idx, comp in enumerate(components):
        sprite = crop_bbox(g, comp["bbox"])
        marker_locs = np.argwhere(np.isin(sprite, list(marker_colors)))
        assert 1 <= marker_locs.shape[0] <= 2
        markers = {}
        for rr, cc in marker_locs:
            r, c = int(rr), int(cc)
            color = int(sprite[r, c])
            assert color not in markers
            direction = connector_dir(sprite, r, c, bg)
            markers[color] = {"pos": (r, c), "dir": direction}
            occurrences[color].append((idx, (r, c), direction))
        objects.append({"sprite": sprite, "area": int(comp["area"]), "markers": markers, "bbox": comp["bbox"]})

    areas = [obj["area"] for obj in objects]
    assert len(set(areas)) == len(areas)
    for color, recs in occurrences.items():
        assert len(recs) == 2
        assert recs[0][0] != recs[1][0]
    return bg, body, marker_colors, objects


def render_assembled(objects: list[dict], placements: dict[int, tuple[int, int]], bg: int) -> np.ndarray:
    r0 = min(int(placements[idx][0]) for idx in range(len(objects)))
    c0 = min(int(placements[idx][1]) for idx in range(len(objects)))
    r1 = max(int(placements[idx][0] + objects[idx]["sprite"].shape[0] - 1) for idx in range(len(objects)))
    c1 = max(int(placements[idx][1] + objects[idx]["sprite"].shape[1] - 1) for idx in range(len(objects)))
    out = np.full((r1 - r0 + 1, c1 - c0 + 1), int(bg), dtype=int)
    for idx, obj in enumerate(objects):
        br, bc = placements[idx]
        rr, cc = br - r0, bc - c0
        sprite = obj["sprite"]
        target = out[rr : rr + sprite.shape[0], cc : cc + sprite.shape[1]]
        mask = sprite != int(bg)
        assert target.shape == sprite.shape
        assert not np.any((target != int(bg)) & mask)
        paste(out, sprite, rr, cc, transparent=bg)
    return out


def simulate_sorted_assembly(objects: list[dict], bg: int) -> tuple[dict[int, tuple[int, int]], list[list[int]]]:
    ordered = sorted(range(len(objects)), key=lambda idx: int(objects[idx]["area"]), reverse=True)
    placements = {ordered[0]: (0, 0)}
    placed_order = [ordered[0]]
    open_markers = {
        color: {"abs_pos": rec["pos"], "dir": rec["dir"]}
        for color, rec in objects[ordered[0]]["markers"].items()
    }
    candidate_history = []
    placed_cells = occupied_abs(objects[ordered[0]]["sprite"], (0, 0), bg)

    for idx in ordered[1:]:
        remaining = [j for j in ordered if j not in placements]
        candidates = [j for j in remaining if any(color in open_markers for color in objects[j]["markers"])]
        candidate_history.append(candidates)

        shared = [color for color in objects[idx]["markers"] if color in open_markers]
        assert len(shared) == 1
        color = shared[0]
        anchor = open_markers[color]
        pos = objects[idx]["markers"][color]["pos"]
        direction = objects[idx]["markers"][color]["dir"]
        assert direction == (-anchor["dir"][0], -anchor["dir"][1])
        base = (
            int(anchor["abs_pos"][0] + anchor["dir"][0] - pos[0]),
            int(anchor["abs_pos"][1] + anchor["dir"][1] - pos[1]),
        )
        new_cells = occupied_abs(objects[idx]["sprite"], base, bg)
        assert placed_cells.isdisjoint(new_cells)
        placements[idx] = base
        placed_order.append(idx)

        del open_markers[color]
        for other_color, rec in objects[idx]["markers"].items():
            if other_color == color:
                continue
            open_markers[other_color] = {
                "abs_pos": (base[0] + rec["pos"][0], base[1] + rec["pos"][1]),
                "dir": rec["dir"],
            }
        placed_cells |= new_cells
    return placements, candidate_history


def main(grid: np.ndarray) -> np.ndarray:
    """Assemble the separated gray objects by docking boundary markers in descending pixel-count order.

    Input structure:
    - Black background with 3-4 disconnected 4-connected foreground objects.
    - Each object has a gray body and 1-2 single-pixel colored connector markers on exposed boundary sides.
    - Every connector color appears exactly twice across two different objects, and object sizes are all distinct.

    Rule:
    - Count total non-background pixels in each object and sort the objects from largest to smallest.
    - Place the largest object first, then for each later object use its unique connector color shared with the already placed assembly.
    - Translate that object so the two same-colored markers become 4-neighbors on opposite exposed sides, preserving orientation.
    - After all objects are attached, replace every connector marker with gray and return the tight crop of the assembled result.

    Key insight:
    - The connector colors say how objects can dock, but the size ranking decides which dockable object must be attached next.
    """
    g = np.asarray(grid, dtype=int)
    bg, body, marker_colors, objects = parse_objects(g)
    placements, _candidate_history = simulate_sorted_assembly(objects, bg)
    out = render_assembled(objects, placements, bg)
    out[np.isin(out, list(marker_colors))] = int(body)
    return tight_crop(out, bg)


def path_edges_for_size_order(n: int) -> list[tuple[int, int]]:
    if n == 3:
        positions = random.choice([[0, -1, 1], [0, 1, -1]])
    else:
        positions = random.choice(
            [
                [0, -1, 1, -2],
                [0, 1, -1, 2],
                [0, -1, 1, 2],
                [0, 1, -1, -2],
            ]
        )
    pos_to_obj = {pos: idx for idx, pos in enumerate(positions)}
    sorted_pos = sorted(pos_to_obj)
    return [(pos_to_obj[a], pos_to_obj[b]) for a, b in zip(sorted_pos, sorted_pos[1:])]


def edge_port_plan(n: int, edges: list[tuple[int, int]]) -> tuple[list[list[tuple[int, str]]], list[tuple[int, int, int]]]:
    colors = random.sample(MARKERS, len(edges))
    used_sides = [set() for _ in range(n)]
    ports = [[] for _ in range(n)]
    edge_records = []
    for color, (a, b) in zip(colors, edges):
        choices = [
            side
            for side in SIDES
            if side not in used_sides[a] and OPPOSITE[side] not in used_sides[b]
        ]
        assert choices
        side = random.choice(choices)
        ports[a].append((color, side))
        ports[b].append((color, OPPOSITE[side]))
        used_sides[a].add(side)
        used_sides[b].add(OPPOSITE[side])
        edge_records.append((a, b, color))
    assert all(1 <= len(p) <= 2 for p in ports)
    assert len(ports[0]) == 2
    return ports, edge_records


def support_cells_for_port(side: str, r: int, c: int) -> set[tuple[int, int]]:
    if side == "T":
        return {(r, c), (r, c - 1), (r, c + 1), (r + 1, c)}
    if side == "B":
        return {(r, c), (r, c - 1), (r, c + 1), (r - 1, c)}
    if side == "L":
        return {(r, c), (r - 1, c), (r + 1, c), (r, c + 1)}
    return {(r, c), (r - 1, c), (r + 1, c), (r, c - 1)}


def port_candidates(side: str, h: int, w: int) -> list[tuple[int, int]]:
    if side == "T":
        return [(0, c) for c in range(1, w - 1)]
    if side == "B":
        return [(h - 1, c) for c in range(1, w - 1)]
    if side == "L":
        return [(r, 0) for r in range(1, h - 1)]
    return [(r, w - 1) for r in range(1, h - 1)]


def add_path(cells: set[tuple[int, int]], start: tuple[int, int], end: tuple[int, int]) -> None:
    r, c = start
    er, ec = end
    cells.add((r, c))
    axes = [0, 1]
    random.shuffle(axes)
    for axis in axes:
        while (axis == 0 and r != er) or (axis == 1 and c != ec):
            if axis == 0:
                r += 1 if er > r else -1
            else:
                c += 1 if ec > c else -1
            cells.add((r, c))


def build_sprite(target_area: int, ports: list[tuple[int, str]]) -> tuple[np.ndarray, dict[int, tuple[int, int]]] | None:
    for _ in range(SPRITE_BUILD_ATTEMPTS):
        dims = [(h, w) for h in DIM_VALUES for w in DIM_VALUES if h * w >= target_area]
        h, w = random.choice(dims)

        for _ in range(PORT_ATTEMPTS):
            port_locs = {}
            blocked = set()
            valid = True
            for color, side in random.sample(ports, len(ports)):
                choices = port_candidates(side, h, w)
                random.shuffle(choices)
                picked = None
                for r, c in choices:
                    support = support_cells_for_port(side, r, c)
                    if blocked & support:
                        continue
                    picked = (r, c)
                    blocked |= support
                    break
                if picked is None:
                    valid = False
                    break
                port_locs[color] = picked
            if valid and len(port_locs) == len(ports):
                break
        else:
            continue

        cells = set()
        for color, side in ports:
            r, c = port_locs[color]
            cells |= support_cells_for_port(side, r, c)
        center = (random.randint(1, h - 2), random.randint(1, w - 2))
        cells.add(center)
        seeds = list(cells)
        for cell in seeds:
            add_path(cells, cell, center)
        if len(cells) > target_area:
            continue

        while len(cells) < target_area:
            frontier = []
            for r, c in list(cells):
                for nr, nc in neighbors4(r, c):
                    if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                        frontier.append((nr, nc))
            if not frontier:
                break
            cells.add(random.choice(frontier))
        if len(cells) != target_area:
            continue

        sprite = np.zeros((h, w), dtype=int)
        for r, c in cells:
            sprite[r, c] = BODY
        for color, _side in ports:
            r, c = port_locs[color]
            sprite[r, c] = color

        if len(find_components(sprite, 0)) != 1:
            continue
        try:
            for color, side in ports:
                r, c = port_locs[color]
                if connector_dir(sprite, r, c, 0) != DIRS[side]:
                    raise AssertionError
        except AssertionError:
            continue
        if int(np.count_nonzero(sprite)) != int(target_area):
            continue
        return sprite, port_locs
    return None


def placements_from_edges(
    sprites: list[np.ndarray],
    ports: list[list[tuple[int, str]]],
    port_locs: list[dict[int, tuple[int, int]]],
    edge_records: list[tuple[int, int, int]],
) -> dict[int, tuple[int, int]] | None:
    side_map = [{color: side for color, side in sprite_ports} for sprite_ports in ports]
    adj = [[] for _ in sprites]
    for a, b, color in edge_records:
        side = side_map[a][color]
        dr, dc = DIRS[side]
        ar, ac = port_locs[a][color]
        br, bc = port_locs[b][color]
        delta = (ar + dr - br, ac + dc - bc)
        adj[a].append((b, delta))
        adj[b].append((a, (-delta[0], -delta[1])))

    placements = {0: (0, 0)}
    q = deque([0])
    while q:
        idx = q.popleft()
        br, bc = placements[idx]
        for nbr, (dr, dc) in adj[idx]:
            pos = (br + dr, bc + dc)
            if nbr in placements:
                if placements[nbr] != pos:
                    return None
            else:
                placements[nbr] = pos
                q.append(nbr)
    return placements if len(placements) == len(sprites) else None


def contact_pairs_between(
    sprite_a: np.ndarray,
    base_a: tuple[int, int],
    sprite_b: np.ndarray,
    base_b: tuple[int, int],
    bg: int,
) -> set[tuple[tuple[int, int], tuple[int, int]]]:
    cells_a = occupied_abs(sprite_a, base_a, bg)
    cells_b = occupied_abs(sprite_b, base_b, bg)
    contacts = set()
    for ra, ca in cells_a:
        for nb in neighbors4(ra, ca):
            if nb in cells_b:
                contacts.add(((ra, ca), nb))
    return contacts


def validate_assembled_layout(
    objects: list[dict],
    placements: dict[int, tuple[int, int]],
    order: list[int],
    edge_records: list[tuple[int, int, int]],
    bg: int,
) -> bool:
    edge_to_color = {
        frozenset((a, b)): color
        for a, b, color in edge_records
    }
    placed = {order[0]}
    placed_cells = occupied_abs(objects[order[0]]["sprite"], placements[order[0]], bg)
    for idx in order[1:]:
        new_cells = occupied_abs(objects[idx]["sprite"], placements[idx], bg)
        if placed_cells & new_cells:
            return False
        touching = []
        for prev in placed:
            pairs = contact_pairs_between(objects[idx]["sprite"], placements[idx], objects[prev]["sprite"], placements[prev], bg)
            if pairs:
                touching.append(prev)
        if len(touching) != 1:
            return False
        prev = touching[0]
        color = edge_to_color.get(frozenset((idx, prev)))
        if color is None:
            return False
        local0 = objects[idx]["markers"][color]["pos"]
        local1 = objects[prev]["markers"][color]["pos"]
        abs0 = (placements[idx][0] + local0[0], placements[idx][1] + local0[1])
        abs1 = (placements[prev][0] + local1[0], placements[prev][1] + local1[1])
        if abs(abs0[0] - abs1[0]) + abs(abs0[1] - abs1[1]) != 1:
            return False
        if int(objects[idx]["sprite"][local0]) != int(color):
            return False
        if int(objects[prev]["sprite"][local1]) != int(color):
            return False
        placed.add(idx)
        placed_cells |= new_cells
    return True


def scatter_input(sprites: list[np.ndarray]) -> np.ndarray | None:
    h = random.randint(*CANVAS_RANGE)
    w = random.randint(*CANVAS_RANGE)
    canvas = np.zeros((h, w), dtype=int)
    shadow = np.zeros((h, w), dtype=bool)
    order = list(range(len(sprites)))
    random.shuffle(order)
    for idx in order:
        sprite = sprites[idx]
        sh, sw = sprite.shape
        placed = False
        for _ in range(SCATTER_ATTEMPTS):
            r0 = random.randint(0, h - sh)
            c0 = random.randint(0, w - sw)
            rr0 = max(0, r0 - SCATTER_PAD)
            cc0 = max(0, c0 - SCATTER_PAD)
            rr1 = min(h, r0 + sh + SCATTER_PAD)
            cc1 = min(w, c0 + sw + SCATTER_PAD)
            if shadow[rr0:rr1, cc0:cc1].any():
                continue
            shadow[r0 : r0 + sh, c0 : c0 + sw] = True
            paste(canvas, sprite, r0, c0, transparent=0)
            placed = True
            break
        if not placed:
            return None
    return canvas


def validate_generated(grid: np.ndarray) -> bool:
    try:
        bg, body, marker_colors, objects = parse_objects(grid)
        if bg != 0 or body != BODY or not marker_colors.issubset(set(MARKERS)):
            return False
        order = sorted(range(len(objects)), key=lambda idx: int(objects[idx]["area"]), reverse=True)
        placements, candidate_history = simulate_sorted_assembly(objects, bg)
        if not any(len(cands) >= 2 for cands in candidate_history):
            return False
        if order == sorted(order, key=lambda idx: (objects[idx]["bbox"][0], objects[idx]["bbox"][1])):
            return False
        out = main(grid)
        expected = render_assembled(objects, placements, bg)
        expected[np.isin(expected, list(marker_colors))] = body
        expected = tight_crop(expected, bg)
        if not np.array_equal(out, expected):
            return False
        if not validate_assembled_layout(objects, placements, order, infer_edge_records(objects), bg):
            return False
        return max(grid.shape) <= 30 and max(out.shape) <= 30
    except Exception:
        return False


def infer_edge_records(objects: list[dict]) -> list[tuple[int, int, int]]:
    color_owner = {}
    for idx, obj in enumerate(objects):
        for color in obj["markers"]:
            color_owner.setdefault(color, []).append(idx)
    return [(owners[0], owners[1], color) for color, owners in color_owner.items()]


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        n = random.choice((3, 4))
        edges = path_edges_for_size_order(n)
        ports, edge_records = edge_port_plan(n, edges)
        areas = sorted(random.sample(AREA_VALUES, n), reverse=True)

        built = []
        for area, sprite_ports in zip(areas, ports):
            item = build_sprite(area, sprite_ports)
            if item is None:
                built = []
                break
            built.append(item)
        if len(built) != n:
            continue

        sprites = [item[0] for item in built]
        port_locs = [item[1] for item in built]
        placements = placements_from_edges(sprites, ports, port_locs, edge_records)
        if placements is None:
            continue

        objects = [{"sprite": sprite, "area": area, "markers": {color: {"pos": locs[color], "dir": DIRS[side]} for color, side in sprite_ports}} for sprite, area, locs, sprite_ports in zip(sprites, areas, port_locs, ports)]
        order = list(range(n))
        if not validate_assembled_layout(objects, placements, order, edge_records, 0):
            continue

        out = render_assembled(objects, placements, 0)
        out[np.isin(out, list(MARKERS))] = BODY
        if max(out.shape) > 30:
            continue

        inp = scatter_input(sprites)
        if inp is None:
            continue
        if not validate_generated(inp):
            continue
        return inp
    raise RuntimeError("failed to generate valid input")
