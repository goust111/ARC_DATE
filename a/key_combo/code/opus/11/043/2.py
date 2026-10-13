from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
RED = 2
CORRIDOR_COLORS = (1, 3, 4)
BODY_COLORS = (5, 6, 7, 8, 9)
DIRS = {
    "N": (-1, 0),
    "E": (0, 1),
    "S": (1, 0),
    "W": (0, -1),
}
OPPOSITE = {"N": "S", "E": "W", "S": "N", "W": "E"}
DIR_ORDER = ("N", "E", "S", "W")


TEMPLATES = {
    "end_a": {
        "size": 3,
        "cells": [
            (0, 1, "conn"),
            (1, 1, "a"),
            (1, 0, "b"),
            (2, 1, "c"),
        ],
        "sides": {"N": (0, 1)},
    },
    "end_b": {
        "size": 3,
        "cells": [
            (0, 1, "conn"),
            (1, 1, "a"),
            (1, 0, "b"),
            (2, 0, "c"),
            (2, 1, "d"),
        ],
        "sides": {"N": (0, 1)},
    },
    "straight": {
        "size": 3,
        "cells": [
            (0, 1, "conn0"),
            (1, 1, "a"),
            (1, 0, "b"),
            (1, 2, "c"),
            (2, 1, "conn1"),
        ],
        "sides": {"N": (0, 1), "S": (2, 1)},
    },
    "elbow": {
        "size": 3,
        "cells": [
            (0, 1, "conn0"),
            (1, 1, "a"),
            (1, 0, "b"),
            (1, 2, "conn1"),
            (2, 1, "c"),
        ],
        "sides": {"N": (0, 1), "E": (1, 2)},
    },
}


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _neighbors4(r: int, c: int):
    for dr, dc in DIRS.values():
        yield r + dr, c + dc


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    out: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            out.append(comp)
    return out


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rs = [r for r, _ in coords]
    cs = [c for _, c in coords]
    return min(rs), min(cs), max(rs), max(cs)


def _crop_pixels(pixels: dict[tuple[int, int], int]) -> np.ndarray:
    coords = list(pixels)
    r0, c0, r1, c1 = _bbox(coords)
    out = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=int)
    for (r, c), color in pixels.items():
        out[r - r0, c - c0] = color
    return out


def _rotate_side(side: str, k: int) -> str:
    idx = DIR_ORDER.index(side)
    return DIR_ORDER[(idx + k) % 4]


def _rotate_cells(
    cells: list[tuple[int, int, str]],
    side_map: dict[str, tuple[int, int]],
    size: int,
    k: int,
) -> tuple[list[tuple[int, int, str]], dict[str, tuple[int, int]]]:
    cur_cells = list(cells)
    cur_sides = dict(side_map)
    cur_size = size
    for _ in range(k % 4):
        cur_cells = [(c, cur_size - 1 - r, token) for r, c, token in cur_cells]
        cur_sides = {
            _rotate_side(side, 1): (c, cur_size - 1 - r)
            for side, (r, c) in cur_sides.items()
        }
    return cur_cells, cur_sides


def _make_sprite(required_sides: tuple[str, ...], rng: random.Random) -> dict:
    req = tuple(sorted(required_sides))
    if len(req) == 1:
        name = rng.choice(["end_a", "end_b"])
    elif {req[0], req[1]} in ({"N", "S"}, {"E", "W"}):
        name = "straight"
    else:
        name = "elbow"

    template = TEMPLATES[name]
    for k in range(4):
        rot_cells, rot_sides = _rotate_cells(
            template["cells"],
            template["sides"],
            template["size"],
            k,
        )
        if tuple(sorted(rot_sides)) != req:
            continue

        body_tokens = sorted({token for _, _, token in rot_cells if "conn" not in token})
        body_palette = list(BODY_COLORS)
        rng.shuffle(body_palette)
        color_for = {token: body_palette[i] for i, token in enumerate(body_tokens)}
        cells: dict[tuple[int, int], int] = {}
        for r, c, token in rot_cells:
            cells[(r, c)] = RED if "conn" in token else color_for[token]
        return {
            "cells": cells,
            "connectors": dict(rot_sides),
            "shape": (template["size"], template["size"]),
        }
    raise ValueError(f"cannot orient sprite for sides {required_sides}")


def _shift_cells(
    cells: dict[tuple[int, int], int], delta: tuple[int, int]
) -> dict[tuple[int, int], int]:
    dr, dc = delta
    return {(r + dr, c + dc): color for (r, c), color in cells.items()}


def _shift_points(
    pts: dict[str, tuple[int, int]], delta: tuple[int, int]
) -> dict[str, tuple[int, int]]:
    dr, dc = delta
    return {key: (r + dr, c + dc) for key, (r, c) in pts.items()}


def _connectors_for_walk(dirs: list[str]) -> list[tuple[str, ...]]:
    n = len(dirs) + 1
    sides: list[list[str]] = [[] for _ in range(n)]
    for i, d in enumerate(dirs):
        sides[i].append(d)
        sides[i + 1].append(OPPOSITE[d])
    return [tuple(v) for v in sides]


def _sample_walk(n: int, rng: random.Random) -> list[str] | None:
    for _ in range(80):
        dirs: list[str] = []
        pos = (0, 0)
        seen = {pos}
        ok = True
        for i in range(n - 1):
            choices = list(DIR_ORDER)
            rng.shuffle(choices)
            picked = None
            for d in choices:
                if i > 0 and d == OPPOSITE[dirs[-1]]:
                    continue
                dr, dc = DIRS[d]
                nxt = (pos[0] + dr, pos[1] + dc)
                if nxt in seen:
                    continue
                picked = d
                pos = nxt
                seen.add(nxt)
                dirs.append(d)
                break
            if picked is None:
                ok = False
                break
        if ok:
            return dirs
    return None


def _place_final_chain(
    sprites: list[dict], dirs: list[str]
) -> dict[tuple[int, int], int] | None:
    placed = []
    anchor_cells = dict(sprites[0]["cells"])
    anchor_connectors = dict(sprites[0]["connectors"])
    placed.append({"cells": anchor_cells, "connectors": anchor_connectors})

    union = dict(anchor_cells)
    for i, d in enumerate(dirs, start=1):
        prev = placed[i - 1]
        cur = sprites[i]
        target = prev["connectors"][d]
        source = cur["connectors"][OPPOSITE[d]]
        delta = (target[0] - source[0], target[1] - source[1])
        cur_cells = _shift_cells(cur["cells"], delta)
        cur_conns = _shift_points(cur["connectors"], delta)
        for coord, color in cur_cells.items():
            if coord not in union:
                continue
            if union[coord] != color or color != RED:
                return None
        union.update(cur_cells)
        placed.append({"cells": cur_cells, "connectors": cur_conns})
    return union


def _scatter_sprites(
    sprites: list[dict], shape: tuple[int, int], rng: random.Random
) -> list[dict] | None:
    h, w = shape
    placed: list[dict] = []
    inflated_boxes: list[tuple[int, int, int, int]] = []
    for sprite in sprites:
        sh, sw = sprite["shape"]
        for _ in range(120):
            r0 = rng.randint(2, h - sh - 3)
            c0 = rng.randint(2, w - sw - 3)
            box = (r0 - 2, c0 - 2, r0 + sh + 1, c0 + sw + 1)
            clash = False
            for br0, bc0, br1, bc1 in inflated_boxes:
                if not (box[2] < br0 or br1 < box[0] or box[3] < bc0 or bc1 < box[1]):
                    clash = True
                    break
            if clash:
                continue
            cells = _shift_cells(sprite["cells"], (r0, c0))
            conns = _shift_points(sprite["connectors"], (r0, c0))
            ok = True
            for side, (cr, cc) in conns.items():
                dr, dc = DIRS[side]
                er, ec = cr + dr, cc + dc
                if not _in_bounds(shape, er, ec):
                    ok = False
                    break
                if (er, ec) in cells:
                    ok = False
                    break
            if not ok:
                continue
            placed.append({"cells": cells, "connectors": conns, "shape": sprite["shape"]})
            inflated_boxes.append(box)
            break
        else:
            return None
    return placed


def _shortest_path(
    start: tuple[int, int],
    end: tuple[int, int],
    blocked: set[tuple[int, int]],
    shape: tuple[int, int],
    forbidden: set[tuple[int, int]],
) -> list[tuple[int, int]] | None:
    q = deque([start])
    prev = {start: None}
    while q:
        cur = q.popleft()
        if cur == end:
            break
        for nr, nc in _neighbors4(*cur):
            if not _in_bounds(shape, nr, nc):
                continue
            nxt = (nr, nc)
            if nxt in prev:
                continue
            if nxt in blocked and nxt != end:
                continue
            if nxt in forbidden and nxt != end:
                continue
            prev[nxt] = cur
            q.append(nxt)
    if end not in prev:
        return None
    path = []
    cur = end
    while cur is not None:
        path.append(cur)
        cur = prev[cur]
    path.reverse()
    return path


def _nonconsecutive_adjacent(path: list[tuple[int, int]]) -> bool:
    idx = {cell: i for i, cell in enumerate(path)}
    for i, (r, c) in enumerate(path):
        for nr, nc in _neighbors4(r, c):
            j = idx.get((nr, nc))
            if j is None:
                continue
            if abs(i - j) != 1:
                return True
    return False


def _path_open_sides(
    path: list[tuple[int, int]],
    start_side: str,
    end_side: str,
) -> dict[tuple[int, int], set[str]]:
    open_sides: dict[tuple[int, int], set[str]] = {}
    for i, cell in enumerate(path):
        sides: set[str] = set()
        if i == 0:
            sides.add(OPPOSITE[start_side])
        else:
            pr, pc = path[i - 1]
            dr = pr - cell[0]
            dc = pc - cell[1]
            for side, vec in DIRS.items():
                if vec == (dr, dc):
                    sides.add(side)
                    break
        if i == len(path) - 1:
            sides.add(OPPOSITE[end_side])
        else:
            nr, nc = path[i + 1]
            dr = nr - cell[0]
            dc = nc - cell[1]
            for side, vec in DIRS.items():
                if vec == (dr, dc):
                    sides.add(side)
                    break
        open_sides[cell] = sides
    return open_sides


def _corridor_wall_cells(
    path: list[tuple[int, int]],
    start_side: str,
    end_side: str,
) -> tuple[dict[tuple[int, int], set[str]], set[tuple[int, int]]]:
    open_sides = _path_open_sides(path, start_side, end_side)
    path_set = set(path)
    walls: set[tuple[int, int]] = set()
    for cell, sides in open_sides.items():
        if len(sides) != 2:
            raise ValueError("corridor cell must have exactly two open sides")
        for side, (dr, dc) in DIRS.items():
            nr, nc = cell[0] + dr, cell[1] + dc
            if side in sides:
                if (nr, nc) in path_set:
                    continue
            else:
                walls.add((nr, nc))
    return open_sides, walls


def _route_corridor(
    start: tuple[int, int],
    end: tuple[int, int],
    start_side: str,
    end_side: str,
    blocked: set[tuple[int, int]],
    shape: tuple[int, int],
    waypoint_count: int,
    rng: random.Random,
) -> list[tuple[int, int]] | None:
    def valid_path(path: list[tuple[int, int]]) -> bool:
        if len(path) < 2:
            return False
        if len(set(path)) != len(path):
            return False
        if any(not _in_bounds(shape, r, c) for r, c in path):
            return False
        if any(cell in blocked and cell not in (start, end) for cell in path):
            return False
        if _nonconsecutive_adjacent(path):
            return False
        try:
            _, walls = _corridor_wall_cells(path, start_side, end_side)
        except ValueError:
            return False
        if any(not _in_bounds(shape, r, c) for r, c in walls):
            return False
        if any(w in blocked for w in walls):
            return False
        if any(w in path for w in walls):
            return False
        return True

    if waypoint_count == 0:
        r0, c0 = start
        r1, c1 = end
        paths: list[list[tuple[int, int]]] = []
        step_c = 1 if c1 >= c0 else -1
        p1 = [(r0, c) for c in range(c0, c1 + step_c, step_c)]
        step_r = 1 if r1 >= r0 else -1
        p1.extend((r, c1) for r in range(r0 + step_r, r1 + step_r, step_r))
        p2 = [(r, c0) for r in range(r0, r1 + step_r, step_r)]
        p2.extend((r1, c) for c in range(c0 + step_c, c1 + step_c, step_c))
        paths.extend([p1, p2])
        rng.shuffle(paths)
        for path in paths:
            if valid_path(path):
                return path
        return None

    candidates = [
        (r, c)
        for r in range(1, shape[0] - 1)
        for c in range(1, shape[1] - 1)
        if (r, c) not in blocked
    ]
    direct = abs(start[0] - end[0]) + abs(start[1] - end[1])
    for _ in range(120):
        points = [start]
        if waypoint_count:
            waypoints = []
            rng.shuffle(candidates)
            for pt in candidates:
                if pt == start or pt == end:
                    continue
                if abs(pt[0] - start[0]) + abs(pt[1] - start[1]) < 3:
                    continue
                if abs(pt[0] - end[0]) + abs(pt[1] - end[1]) < 3:
                    continue
                if any(abs(pt[0] - wr) + abs(pt[1] - wc) < 4 for wr, wc in waypoints):
                    continue
                waypoints.append(pt)
                if len(waypoints) == waypoint_count:
                    break
            if len(waypoints) != waypoint_count:
                continue
            points.extend(waypoints)
        points.append(end)

        cur_blocked = set(blocked)
        path: list[tuple[int, int]] = []
        ok = True
        for i in range(len(points) - 1):
            segment = _shortest_path(
                points[i],
                points[i + 1],
                cur_blocked - {points[i + 1]},
                shape,
                forbidden=set(path[1:]),
            )
            if segment is None:
                ok = False
                break
            if path:
                segment = segment[1:]
            path.extend(segment)
            cur_blocked.update(segment[:-1])
        if not ok:
            continue
        if len(path) <= direct:
            continue
        if not valid_path(path):
            continue
        return path
    return None


def _paint_grid(shape: tuple[int, int], sprites: list[dict]) -> np.ndarray:
    g = np.zeros(shape, dtype=int)
    for sprite in sprites:
        for (r, c), color in sprite["cells"].items():
            g[r, c] = color
    return g


def _build_input_from_scatter(
    scattered: list[dict],
    dirs: list[str],
    shape: tuple[int, int],
    rng: random.Random,
) -> np.ndarray | None:
    grid = _paint_grid(shape, scattered)
    occupied = {
        cell
        for sprite in scattered
        for cell in sprite["cells"]
    }
    edge_order = list(range(len(dirs)))
    rng.shuffle(edge_order)
    # Direct corridor routing is enough to encode both connector pairing and
    # length ordering; random waypoints made generation disproportionately slow.
    waypoint_counts = [0] * len(dirs)

    colors = list(CORRIDOR_COLORS[: len(dirs)])
    rng.shuffle(colors)

    for edge_idx in edge_order:
        a = scattered[edge_idx]
        b = scattered[edge_idx + 1]
        side_a = dirs[edge_idx]
        side_b = OPPOSITE[side_a]
        ca = a["connectors"][side_a]
        cb = b["connectors"][side_b]
        sa = (ca[0] + DIRS[side_a][0], ca[1] + DIRS[side_a][1])
        sb = (cb[0] + DIRS[side_b][0], cb[1] + DIRS[side_b][1])
        if sa in occupied or sb in occupied:
            return None
        path = _route_corridor(
            sa,
            sb,
            side_a,
            side_b,
            occupied,
            shape,
            waypoint_counts[edge_idx],
            rng,
        )
        if path is None:
            return None
        _, walls = _corridor_wall_cells(path, side_a, side_b)
        color = colors[edge_idx]
        for wr, wc in walls:
            if grid[wr, wc] != BG:
                return None
            grid[wr, wc] = color
        occupied.update(walls)
        occupied.update(path)
    return grid


def _corridor_cells_for_color(
    g: np.ndarray, color: int
) -> dict[tuple[int, int], set[str]]:
    h, w = g.shape
    cells: dict[tuple[int, int], set[str]] = {}
    for r in range(h):
        for c in range(w):
            if g[r, c] != BG:
                continue
            wall_sides = []
            for side, (dr, dc) in DIRS.items():
                nr, nc = r + dr, c + dc
                if _in_bounds(g.shape, nr, nc) and g[nr, nc] == color:
                    wall_sides.append(side)
            if len(wall_sides) != 2:
                continue
            open_sides = set(DIR_ORDER) - set(wall_sides)
            cells[(r, c)] = open_sides
    return cells


def _corridor_edges(cells: dict[tuple[int, int], set[str]]) -> dict[tuple[int, int], list[tuple[int, int]]]:
    edges = {cell: [] for cell in cells}
    for (r, c), open_sides in cells.items():
        for side in open_sides:
            dr, dc = DIRS[side]
            nxt = (r + dr, c + dc)
            if nxt not in cells:
                continue
            if OPPOSITE[side] in cells[nxt]:
                edges[(r, c)].append(nxt)
    return edges


def _extract_sprites(g: np.ndarray) -> list[dict]:
    sprite_mask = (g != BG) & ~np.isin(g, CORRIDOR_COLORS)
    comps = _components(sprite_mask)
    sprites = []
    for idx, comp in enumerate(comps):
        pixels = {(r, c): int(g[r, c]) for r, c in comp}
        connectors = [coord for coord in comp if g[coord] == RED]
        sprites.append(
            {
                "id": idx,
                "pixels": pixels,
                "connectors": connectors,
            }
        )
    return sprites


def _pairings(g: np.ndarray) -> list[dict]:
    connector_coords = {
        (r, c)
        for r, c in zip(*np.where(g == RED))
    }
    pairings = []
    for color in CORRIDOR_COLORS:
        if not np.any(g == color):
            continue
        cells = _corridor_cells_for_color(g, color)
        if not cells:
            continue
        edges = _corridor_edges(cells)
        endpoints: dict[tuple[int, int], tuple[int, int]] = {}
        for cr, cc in connector_coords:
            matches = []
            for side, (dr, dc) in DIRS.items():
                entry = (cr + dr, cc + dc)
                if entry in cells and OPPOSITE[side] in cells[entry]:
                    matches.append(entry)
            if len(matches) == 1:
                endpoints[(cr, cc)] = matches[0]
        touched = [(conn, entry) for conn, entry in endpoints.items() if entry in cells]
        touched = [
            item
            for item in touched
            if any(
                _in_bounds(g.shape, item[0][0] + dr, item[0][1] + dc)
                and g[item[0][0] + dr, item[0][1] + dc] == color
                for dr, dc in DIRS.values()
            )
            or item[1] in cells
        ]

        relevant = []
        for conn, entry in endpoints.items():
            if entry not in cells:
                continue
            relevant.append((conn, entry))
        if len(relevant) != 2:
            continue

        start_conn, start_entry = relevant[0]
        end_conn, end_entry = relevant[1]
        q = deque([start_entry])
        dist = {start_entry: 1}
        while q:
            cur = q.popleft()
            if cur == end_entry:
                break
            for nxt in edges[cur]:
                if nxt not in dist:
                    dist[nxt] = dist[cur] + 1
                    q.append(nxt)
        if end_entry not in dist:
            continue
        pairings.append(
            {
                "color": color,
                "a": start_conn,
                "b": end_conn,
                "length": dist[end_entry],
            }
        )
    pairings.sort(key=lambda item: item["length"])
    return pairings


def _assemble(g: np.ndarray, pairings: list[dict]) -> np.ndarray:
    sprites = _extract_sprites(g)
    connector_to_sprite = {
        coord: sprite["id"]
        for sprite in sprites
        for coord in sprite["connectors"]
    }
    groups = {}
    connector_group = {}
    for sprite in sprites:
        groups[sprite["id"]] = {
            "pixels": dict(sprite["pixels"]),
            "connectors": {coord: coord for coord in sprite["connectors"]},
            "members": {sprite["id"]},
        }
        for coord in sprite["connectors"]:
            connector_group[coord] = sprite["id"]

    for pair in pairings:
        a = pair["a"]
        b = pair["b"]
        ga = connector_group[a]
        gb = connector_group[b]
        if ga == gb:
            continue
        group_a = groups[ga]
        group_b = groups[gb]
        if len(group_a["members"]) > len(group_b["members"]):
            fixed_id, moving_id = ga, gb
            fixed_conn, moving_conn = a, b
        elif len(group_b["members"]) > len(group_a["members"]):
            fixed_id, moving_id = gb, ga
            fixed_conn, moving_conn = b, a
        else:
            if min(group_a["members"]) <= min(group_b["members"]):
                fixed_id, moving_id = ga, gb
                fixed_conn, moving_conn = a, b
            else:
                fixed_id, moving_id = gb, ga
                fixed_conn, moving_conn = b, a
        fixed = groups[fixed_id]
        moving = groups[moving_id]
        tr = fixed["connectors"][fixed_conn][0] - moving["connectors"][moving_conn][0]
        tc = fixed["connectors"][fixed_conn][1] - moving["connectors"][moving_conn][1]
        moved_pixels = _shift_cells(moving["pixels"], (tr, tc))
        moved_connectors = _shift_points(moving["connectors"], (tr, tc))
        merged_pixels = dict(fixed["pixels"])
        for coord, color in moved_pixels.items():
            if coord in merged_pixels and not (merged_pixels[coord] == color == RED):
                raise ValueError("non-connector overlap during assembly")
            merged_pixels[coord] = color
        merged_connectors = dict(fixed["connectors"])
        merged_connectors.update(moved_connectors)
        del merged_connectors[fixed_conn]
        del merged_connectors[moving_conn]
        merged = {
            "pixels": merged_pixels,
            "connectors": merged_connectors,
            "members": fixed["members"] | moving["members"],
        }
        groups[fixed_id] = merged
        del groups[moving_id]
        for coord in merged_connectors:
            connector_group[coord] = fixed_id
    assert len(groups) == 1
    return _crop_pixels(next(iter(groups.values()))["pixels"])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Assemble the sprites by pairing red connectors through colored wall-corridors and applying the corridor-length order.

    Input structure:
    - The grid has a black background, 3-4 small multicolor sprite bodies, and red connector pixels on sprite edges.
    - Separate colored wall systems form one-cell-wide background corridors; each corridor color links exactly one connector pair.
    - Corridor walls use non-red colors while sprite body pixels use non-corridor colors.

    Rule:
    - Detect every sprite as a 4-connected non-corridor component and every red pixel on it as a connector.
    - For each corridor color, follow the background cells that are open between that color's wall pixels to find the two connected red endpoints, and measure the corridor path length by counting those background cells.
    - Sort connector pairs from shorter corridor to longer corridor. Merge them in that order by translating one current sprite/group so the paired red pixels coincide.
    - Remove the corridors and crop the final merged sprite assembly to its bounding box.

    Key insight:
    - The corridor geometry carries two signals at once: which connectors belong together and which merge must happen first, because different corridor lengths impose the assembly order.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    pairings = _pairings(g)
    assert pairings
    lengths = [item["length"] for item in pairings]
    assert len(lengths) == len(set(lengths))
    return _assemble(g, pairings)


def _generate_case(rng: random.Random) -> tuple[np.ndarray, np.ndarray] | None:
    n = 4 if rng.random() < 0.6 else 3
    dirs = _sample_walk(n, rng)
    if dirs is None:
        return None
    side_specs = _connectors_for_walk(dirs)
    sprites = [_make_sprite(sides, rng) for sides in side_specs]
    final_pixels = _place_final_chain(sprites, dirs)
    if final_pixels is None:
        return None
    target = _crop_pixels(final_pixels)
    if max(target.shape) > 12:
        return None

    for _ in range(80):
        shape = (rng.randint(18, 24), rng.randint(18, 24))
        scattered = _scatter_sprites(sprites, shape, rng)
        if scattered is None:
            continue
        inp = _build_input_from_scatter(scattered, dirs, shape, rng)
        if inp is None:
            continue
        pairings = _pairings(inp)
        if len(pairings) != len(dirs):
            continue
        lengths = [item["length"] for item in pairings]
        if len(lengths) != len(set(lengths)):
            continue
        sprites_found = _extract_sprites(inp)
        if not (3 <= len(sprites_found) <= 4):
            continue
        if any(not (4 <= len(sprite["pixels"]) <= 8) for sprite in sprites_found):
            continue
        if any(not (1 <= len(sprite["connectors"]) <= 2) for sprite in sprites_found):
            continue
        if not any(len(sprite["connectors"]) == 2 for sprite in sprites_found):
            continue
        paired_connectors = {pair["a"] for pair in pairings} | {pair["b"] for pair in pairings}
        all_connectors = {
            coord
            for sprite in sprites_found
            for coord in sprite["connectors"]
        }
        if paired_connectors != all_connectors:
            continue
        if min(lengths) < 3:
            continue
        if not any(item["length"] > min(lengths) + 3 for item in pairings):
            continue
        out = main(inp)
        if not np.array_equal(out, target):
            continue
        return inp, target
    return None


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(400):
        case = _generate_case(rng)
        if case is None:
            continue
        inp, _ = case
        return inp
    raise RuntimeError("failed to generate valid corridor-assembly puzzle")
