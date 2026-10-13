import random

import numpy as np


GEN_RETRIES = 400
GRID_H_RANGE = (24, 33)
GRID_W_RANGE = (24, 33)
GRAY = 5
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _bg(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp = []
            while stack:
                cr, cc = stack.pop()
                comp.append((int(cr), int(cc)))
                for dr, dc in DIRS4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(comp)
    return out


def _tower_from_component(comp: list[tuple[int, int]]) -> dict | None:
    pts = set(comp)
    if len(pts) not in (11, 12):
        return None
    rows = [r for r, _ in pts]
    cols = [c for _, c in pts]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    arm = len(pts) - 9

    if c1 - c0 + 1 == 3 and r1 - r0 + 1 == 3 + arm:
        body = {(r, c) for r in range(r0, r0 + 3) for c in range(c0, c1 + 1)}
        down = body | {(r, c0 + 1) for r in range(r0 + 3, r1 + 1)}
        if pts == down:
            return {"dir": (1, 0), "tip": (r1, c0 + 1)}
        body = {(r, c) for r in range(r1 - 2, r1 + 1) for c in range(c0, c1 + 1)}
        up = body | {(r, c0 + 1) for r in range(r0, r1 - 2)}
        if pts == up:
            return {"dir": (-1, 0), "tip": (r0, c0 + 1)}

    if r1 - r0 + 1 == 3 and c1 - c0 + 1 == 3 + arm:
        body = {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c0 + 3)}
        right = body | {(r0 + 1, c) for c in range(c0 + 3, c1 + 1)}
        if pts == right:
            return {"dir": (0, 1), "tip": (r0 + 1, c1)}
        body = {(r, c) for r in range(r0, r1 + 1) for c in range(c1 - 2, c1 + 1)}
        left = body | {(r0 + 1, c) for c in range(c0, c1 - 2)}
        if pts == left:
            return {"dir": (0, -1), "tip": (r0 + 1, c0)}
    return None


def _detect_objects(grid: np.ndarray) -> tuple[int, list[dict], dict[tuple[int, int], dict], dict[tuple[int, int], int], list[list[tuple[int, int]]]]:
    g = np.asarray(grid, dtype=int)
    bg = _bg(g)
    colors = [int(x) for x in np.unique(g) if int(x) != bg]

    receiver_color = None
    receivers: list[list[tuple[int, int]]] = []
    for color in colors:
        comps = _components(g == color)
        if 2 <= len(comps) <= 3 and all(
            len(comp) == 4
            and max(r for r, _ in comp) - min(r for r, _ in comp) == 1
            and max(c for _, c in comp) - min(c for _, c in comp) == 1
            for comp in comps
        ):
            receiver_color = color
            receivers = comps
            break
    assert receiver_color is not None

    towers: list[dict] = []
    relays: dict[tuple[int, int], dict] = {}
    receiver_map: dict[tuple[int, int], int] = {}
    for rid, comp in enumerate(receivers):
        for cell in comp:
            receiver_map[cell] = rid

    for color in colors:
        if color == receiver_color:
            continue
        comps = _components(g == color)
        tower_matches = []
        dominoes = []
        for comp in comps:
            tower = _tower_from_component(comp)
            if tower is not None:
                tower["color"] = color
                tower_matches.append(tower)
            elif len(comp) == 2:
                rs = sorted(r for r, _ in comp)
                cs = sorted(c for _, c in comp)
                if rs[0] == rs[1] and cs[1] == cs[0] + 1:
                    a = (rs[0], cs[0])
                    b = (rs[0], cs[1])
                    dominoes.append({"color": color, "cells": (a, b), "axis": (0, 1)})
                elif cs[0] == cs[1] and rs[1] == rs[0] + 1:
                    a = (rs[0], cs[0])
                    b = (rs[1], cs[0])
                    dominoes.append({"color": color, "cells": (a, b), "axis": (1, 0)})
                else:
                    raise AssertionError("invalid relay shape")
            else:
                raise AssertionError("unknown object shape")
        if tower_matches:
            assert len(tower_matches) == 1 and not dominoes
            towers.extend(tower_matches)
        else:
            for relay in dominoes:
                for cell in relay["cells"]:
                    relays[cell] = relay

    assert 2 <= len(towers) <= 3
    assert 2 <= len(receivers) <= 3
    return bg, towers, relays, receiver_map, receivers


def _trace(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg, towers, relays, receiver_map, receivers = _detect_objects(g)
    out = g.copy()

    for tower in towers:
        color = int(tower["color"])
        dr, dc = tower["dir"]
        pos = tower["tip"]
        seen: set[tuple[tuple[int, int], tuple[int, int], int]] = set()
        while True:
            state = (pos, (dr, dc), color)
            if state in seen:
                break
            seen.add(state)
            nr, nc = pos[0] + dr, pos[1] + dc
            if not (0 <= nr < h and 0 <= nc < w):
                break
            cell = int(g[nr, nc])
            if cell == bg:
                out[nr, nc] = color
                pos = (nr, nc)
                continue
            receiver_id = receiver_map.get((nr, nc))
            if receiver_id is not None:
                for rr, cc in receivers[receiver_id]:
                    out[rr, cc] = color
                break
            relay = relays.get((nr, nc))
            if relay is not None:
                a, b = relay["cells"]
                if (nr, nc) == a:
                    pos = b
                    dr = b[0] - a[0]
                    dc = b[1] - a[1]
                else:
                    pos = a
                    dr = a[0] - b[0]
                    dc = a[1] - b[1]
                color = int(relay["color"])
                continue
            break
    return out


def main(grid: np.ndarray) -> np.ndarray:
    return _trace(grid)


def _add(p: tuple[int, int], d: tuple[int, int], k: int = 1) -> tuple[int, int]:
    return p[0] + d[0] * k, p[1] + d[1] * k


def _perpendiculars(d: tuple[int, int]) -> list[tuple[int, int]]:
    dr, dc = d
    return [(-dc, dr), (dc, -dr)]


def _receiver_cells(hit: tuple[int, int], d: tuple[int, int], side: int) -> list[tuple[int, int]]:
    r, c = hit
    if d == (1, 0):
        c0 = c - side
        return [(r, c0), (r, c0 + 1), (r + 1, c0), (r + 1, c0 + 1)]
    if d == (-1, 0):
        c0 = c - side
        return [(r - 1, c0), (r - 1, c0 + 1), (r, c0), (r, c0 + 1)]
    if d == (0, 1):
        r0 = r - side
        return [(r0, c), (r0, c + 1), (r0 + 1, c), (r0 + 1, c + 1)]
    r0 = r - side
    return [(r0, c - 1), (r0, c), (r0 + 1, c - 1), (r0 + 1, c)]


def _tower_cells(side: str, anchor: int, antenna_len: int, h: int, w: int) -> tuple[list[tuple[int, int]], tuple[int, int], tuple[int, int]]:
    if side == "top":
        cells = [(r, c) for r in range(3) for c in range(anchor - 1, anchor + 2)]
        cells += [(r, anchor) for r in range(3, 3 + antenna_len)]
        return cells, (2 + antenna_len, anchor), (1, 0)
    if side == "bottom":
        cells = [(r, c) for r in range(h - 3, h) for c in range(anchor - 1, anchor + 2)]
        cells += [(r, anchor) for r in range(h - 3 - antenna_len, h - 3)]
        return cells, (h - 3 - antenna_len, anchor), (-1, 0)
    if side == "left":
        cells = [(r, c) for r in range(anchor - 1, anchor + 2) for c in range(3)]
        cells += [(anchor, c) for c in range(3, 3 + antenna_len)]
        return cells, (anchor, 2 + antenna_len), (0, 1)
    cells = [(r, c) for r in range(anchor - 1, anchor + 2) for c in range(w - 3, w)]
    cells += [(anchor, c) for c in range(w - 3 - antenna_len, w - 3)]
    return cells, (anchor, w - 3 - antenna_len), (0, -1)


def _in_bounds(cells: list[tuple[int, int]], h: int, w: int) -> bool:
    return all(0 <= r < h and 0 <= c < w for r, c in cells)


def _neighbors8(cells: set[tuple[int, int]], h: int, w: int) -> set[tuple[int, int]]:
    out = set()
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w:
                    out.add((nr, nc))
    return out


def _build_route(
    rng: random.Random,
    side: str,
    branch_kind: int,
    reaches_receiver: bool,
    h: int,
    w: int,
    used_cells: set[tuple[int, int]],
    object_buffer: set[tuple[int, int]],
    tower_color: int,
    relay_palette: list[int],
) -> dict | None:
    for _ in range(80):
        antenna_len = rng.randint(2, 3)
        if side in ("top", "bottom"):
            anchor = rng.randint(2, w - 3)
        else:
            anchor = rng.randint(2, h - 3)
        tower_cells, tip, d0 = _tower_cells(side, anchor, antenna_len, h, w)
        tower_set = set(tower_cells)
        if tower_set & object_buffer or tower_set & used_cells:
            continue

        path_cells: list[tuple[int, int]] = []
        object_cells = list(tower_cells)
        relay_specs = []
        cur = tip
        cur_dir = d0

        segment_lengths = [rng.randint(2, 6) for _ in range(branch_kind + 1)]
        turn_dirs = []
        prev = cur_dir
        for _j in range(branch_kind):
            turn = rng.choice(_perpendiculars(prev))
            turn_dirs.append(turn)
            prev = turn

        ok = True
        for seg_len in segment_lengths[:-1]:
            hit = _add(cur, cur_dir, seg_len + 1)
            for k in range(1, seg_len + 1):
                path_cells.append(_add(cur, cur_dir, k))
            next_dir = turn_dirs[len(relay_specs)]
            exit_cell = _add(hit, next_dir)
            relay_cells = [hit, exit_cell]
            relay_specs.append({"color": rng.choice(relay_palette), "cells": relay_cells})
            object_cells.extend(relay_cells)
            cur = exit_cell
            cur_dir = next_dir

        last_len = segment_lengths[-1]
        if reaches_receiver:
            hit = _add(cur, cur_dir, last_len + 1)
            for k in range(1, last_len + 1):
                path_cells.append(_add(cur, cur_dir, k))
            receiver = _receiver_cells(hit, cur_dir, rng.randint(0, 1))
            object_cells.extend(receiver)
        else:
            step = 1
            while True:
                nxt = _add(cur, cur_dir, step)
                if not (0 <= nxt[0] < h and 0 <= nxt[1] < w):
                    break
                path_cells.append(nxt)
                step += 1
            if len(path_cells) == 0:
                ok = False
            receiver = []

        all_cells = object_cells + path_cells
        if not ok or not _in_bounds(all_cells, h, w):
            continue
        if len(set(all_cells)) != len(all_cells):
            continue
        object_set = set(object_cells)
        path_set = set(path_cells)
        if object_set & object_buffer or object_set & used_cells:
            continue
        if path_set & used_cells:
            continue
        if receiver and any(cell in object_buffer - tower_set for cell in receiver):
            continue
        return {
            "tower_color": tower_color,
            "tower_cells": tower_cells,
            "relay_specs": relay_specs,
            "receiver_cells": receiver,
            "path_cells": path_cells,
            "object_set": object_set,
            "object_buffer": _neighbors8(object_set, h, w),
            "used_set": set(all_cells),
        }
    return None


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(*GRID_H_RANGE)
    w = rng.randint(*GRID_W_RANGE)
    palette = list(range(10))
    rng.shuffle(palette)
    bg = palette.pop()
    gray = GRAY if GRAY != bg else palette.pop()
    if gray in palette:
        palette.remove(gray)

    tower_count = rng.randint(2, 3)
    receiver_count = rng.randint(2, 3)
    tower_colors = [palette.pop() for _ in range(tower_count)]
    relay_palette = palette[: max(2, min(3, len(palette)))]
    if not relay_palette:
        return None

    grid = np.full((h, w), bg, dtype=int)
    used_cells: set[tuple[int, int]] = set()
    object_buffer: set[tuple[int, int]] = set()

    branch_options = [[0, 1], [0, 2], [1, 2]]
    branch_kinds = rng.choice(branch_options if tower_count == 2 else branch_options + [[0, 1, 2]])[:tower_count]
    if len(branch_kinds) < tower_count:
        branch_kinds = branch_kinds + [rng.choice([0, 1, 2]) for _ in range(tower_count - len(branch_kinds))]
    rng.shuffle(branch_kinds)

    hit_count = rng.randint(1, min(receiver_count, tower_count))
    reaches = [False] * tower_count
    for idx in rng.sample(range(tower_count), hit_count):
        reaches[idx] = True

    routes = []
    sides = ["top", "bottom", "left", "right"]
    rng.shuffle(sides)
    for idx in range(tower_count):
        side = sides[idx]
        route = _build_route(
            rng,
            side,
            branch_kinds[idx],
            reaches[idx],
            h,
            w,
            used_cells,
            object_buffer,
            tower_colors[idx],
            relay_palette,
        )
        if route is None:
            return None
        routes.append(route)
        used_cells |= route["used_set"]
        object_buffer |= route["object_buffer"]

    receivers = [route["receiver_cells"] for route in routes if route["receiver_cells"]]
    while len(receivers) < receiver_count:
        placed = False
        for _ in range(80):
            r0 = rng.randint(1, h - 3)
            c0 = rng.randint(1, w - 3)
            cells = [(r0, c0), (r0 + 1, c0), (r0, c0 + 1), (r0 + 1, c0 + 1)]
            cell_set = set(cells)
            if cell_set & used_cells or cell_set & object_buffer:
                continue
            receivers.append(cells)
            used_cells |= cell_set
            object_buffer |= _neighbors8(cell_set, h, w)
            placed = True
            break
        if not placed:
            return None

    for route in routes:
        for r, c in route["tower_cells"]:
            grid[r, c] = route["tower_color"]
        for relay in route["relay_specs"]:
            for r, c in relay["cells"]:
                grid[r, c] = relay["color"]
    for receiver in receivers:
        for r, c in receiver:
            grid[r, c] = gray

    out = main(grid)
    detected_bg, towers, _relays, receiver_map, receiver_comps = _detect_objects(grid)
    assert detected_bg == bg
    assert len(towers) == tower_count
    assert len(receiver_comps) == receiver_count

    colored_receivers = 0
    for rid, comp in enumerate(receiver_comps):
        vals = {int(out[r, c]) for r, c in comp}
        if vals != {gray}:
            colored_receivers += 1
    if colored_receivers < hit_count:
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid signal-tower input")
