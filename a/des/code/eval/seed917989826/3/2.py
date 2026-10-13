from __future__ import annotations

import random
from collections import Counter

import numpy as np


BG = 0
START = 2
CORRIDOR = 5
ORANGE = 7

DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _in_bounds(h: int, w: int, r: int, c: int) -> bool:
    return 0 <= r < h and 0 <= c < w


def _add(a: tuple[int, int], b: tuple[int, int]) -> tuple[int, int]:
    return a[0] + b[0], a[1] + b[1]


def _sub(a: tuple[int, int], b: tuple[int, int]) -> tuple[int, int]:
    return a[0] - b[0], a[1] - b[1]


def _perp(d: tuple[int, int]) -> list[tuple[int, int]]:
    dr, dc = d
    return [(-dc, dr), (dc, -dr)]


def _component_cells(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for dr, dc in DIRS4:
                    nr, nc = cr + dr, cc + dc
                    if _in_bounds(h, w, nr, nc) and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _find_start_room(grid: np.ndarray, bg: int) -> tuple[int, tuple[int, int]]:
    h, w = grid.shape
    candidates: list[tuple[int, tuple[int, int]]] = []
    for color in sorted(int(v) for v in np.unique(grid) if int(v) != bg):
        for comp in _component_cells(grid == color):
            if len(comp) != 4:
                continue
            rs = [r for r, _ in comp]
            cs = [c for _, c in comp]
            r0, r1 = min(rs), max(rs)
            c0, c1 = min(cs), max(cs)
            if r1 - r0 != 1 or c1 - c0 != 1:
                continue
            if bool((grid[r0 : r1 + 1, c0 : c1 + 1] == color).all()):
                candidates.append((color, (r0, c0)))
    assert len(candidates) == 1, "expected a unique 2x2 start room"
    return candidates[0]


def _find_endpoint_rooms(
    grid: np.ndarray, bg: int, corridor_color: int
) -> tuple[list[tuple[int, int]], dict[tuple[int, int], tuple[int, int]]]:
    h, w = grid.shape
    rooms: list[tuple[int, int]] = []
    touch_map: dict[tuple[int, int], tuple[int, int]] = {}
    for r in range(1, h - 2):
        for c in range(1, w - 2):
            if not bool((grid[r : r + 2, c : c + 2] == bg).all()):
                continue
            patch = grid[r - 1 : r + 3, c - 1 : c + 3]
            ring = np.ones((4, 4), dtype=bool)
            ring[1:3, 1:3] = False
            if not bool((patch[ring] == corridor_color).all()):
                continue
            room = (r, c)
            rooms.append(room)
            for rr in range(r, r + 2):
                for cc in range(c, c + 2):
                    for dr, dc in DIRS4:
                        tr, tc = rr + dr, cc + dc
                        if _in_bounds(h, w, tr, tc) and int(grid[tr, tc]) == corridor_color:
                            touch_map[(tr, tc)] = room
    assert rooms, "expected at least one endpoint room"
    return rooms, touch_map


def _next_dirs(mask: np.ndarray, pos: tuple[int, int], heading: tuple[int, int]) -> list[tuple[int, int]]:
    h, w = mask.shape
    out: list[tuple[int, int]] = []
    fr, fc = pos[0] + heading[0], pos[1] + heading[1]
    if _in_bounds(h, w, fr, fc) and bool(mask[fr, fc]):
        out.append(heading)
    for side in _perp(heading):
        r1, c1 = pos[0] + side[0], pos[1] + side[1]
        r2, c2 = pos[0] + 2 * side[0], pos[1] + 2 * side[1]
        if _in_bounds(h, w, r2, c2) and bool(mask[r1, c1]) and bool(mask[r2, c2]):
            out.append(side)
    return out


def _branch_is_wide(mask: np.ndarray, pos: tuple[int, int], direction: tuple[int, int]) -> bool:
    h, w = mask.shape
    for side in _perp(direction):
        run = 0
        best = 0
        for k in range(1, 6):
            mr = pos[0] + direction[0] * k
            mc = pos[1] + direction[1] * k
            sr = mr + side[0]
            sc = mc + side[1]
            if (
                _in_bounds(h, w, mr, mc)
                and _in_bounds(h, w, sr, sc)
                and bool(mask[mr, mc])
                and bool(mask[sr, sc])
            ):
                run += 1
                best = max(best, run)
            else:
                run = 0
        if best >= 2:
            return True
    return False


def _paint_from_centerline(
    grid: np.ndarray, centerline: list[tuple[int, int]], corridor_color: int
) -> set[tuple[int, int]]:
    h, w = grid.shape
    painted = set(centerline)
    for i in range(len(centerline) - 1):
        cur = centerline[i]
        nxt = centerline[i + 1]
        direction = _sub(nxt, cur)
        if i == len(centerline) - 2:
            continue
        for side in _perp(direction):
            sr, sc = cur[0] + side[0], cur[1] + side[1]
            rr, rc = cur[0] + 2 * side[0], cur[1] + 2 * side[1]
            if not _in_bounds(h, w, sr, sc):
                continue
            if int(grid[sr, sc]) != corridor_color:
                continue
            if _in_bounds(h, w, rr, rc) and int(grid[rr, rc]) == corridor_color:
                continue
            painted.add((sr, sc))
    return painted


def main(grid: np.ndarray) -> np.ndarray:
    """
    Follow the grey corridor network from the red room.

    At each junction the chosen branch is the only one that locally widens to
    two cells for a short run. Recolor the used corridor orange and fill the
    reached 2x2 endpoint cavity with the start-room color.
    """
    g = np.array(grid, dtype=int)
    h, w = g.shape
    bg = int(Counter([int(g[0, 0]), int(g[0, -1]), int(g[-1, 0]), int(g[-1, -1])]).most_common(1)[0][0])

    start_color, start_top_left = _find_start_room(g, bg)
    counts = Counter(int(v) for v in g.ravel())
    corridor_color = None
    for color, _ in counts.most_common():
        if color not in (bg, start_color):
            corridor_color = int(color)
            break
    assert corridor_color is not None, "expected a corridor color"

    _, touch_map = _find_endpoint_rooms(g, bg, corridor_color)
    start_cells = {(start_top_left[0] + dr, start_top_left[1] + dc) for dr in range(2) for dc in range(2)}
    corridor_mask = g == corridor_color

    entrances: list[tuple[int, int]] = []
    room_touch: dict[tuple[int, int], tuple[int, int]] = {}
    for rr, cc in start_cells:
        for dr, dc in DIRS4:
            nr, nc = rr + dr, cc + dc
            if _in_bounds(h, w, nr, nc) and bool(corridor_mask[nr, nc]):
                if (nr, nc) not in room_touch:
                    entrances.append((nr, nc))
                    room_touch[(nr, nc)] = (rr, cc)
    assert len(entrances) == 1, "expected a unique corridor entrance"

    cur = entrances[0]
    room_cell = room_touch[cur]
    heading = _sub(cur, room_cell)
    centerline = [cur]
    seen_states = {(cur, heading)}

    for _ in range(h * w * 4):
        if cur in touch_map:
            target_room = touch_map[cur]
            break
        candidates = _next_dirs(corridor_mask, cur, heading)
        assert candidates, "corridor terminated before reaching a room"
        if len(candidates) == 1:
            ndir = candidates[0]
        else:
            wide = [d for d in candidates if _branch_is_wide(corridor_mask, cur, d)]
            assert len(wide) == 1, "junction must have a unique widened branch"
            ndir = wide[0]
        cur = (cur[0] + ndir[0], cur[1] + ndir[1])
        heading = ndir
        state = (cur, heading)
        assert state not in seen_states, "unexpected traversal loop"
        seen_states.add(state)
        centerline.append(cur)
    else:
        raise AssertionError("path tracing exceeded safety limit")

    out = g.copy()
    for r, c in _paint_from_centerline(g, centerline, corridor_color):
        out[r, c] = ORANGE
    tr, tc = target_room
    out[tr : tr + 2, tc : tc + 2] = start_color
    return out


STEP = 6
CANVAS_MARGIN = 6


def _inside_logical(hc: int, wc: int, cell: tuple[int, int]) -> bool:
    return 1 <= cell[0] < hc - 1 and 1 <= cell[1] < wc - 1


def _logical_path(rng: random.Random, hc: int, wc: int, nodes: int) -> list[tuple[int, int]] | None:
    start = (rng.randrange(1, hc - 1), rng.randrange(1, wc - 1))
    path = [start]
    used = {start}

    def dfs() -> bool:
        if len(path) == nodes:
            return True
        cur = path[-1]
        choices = []
        for d in DIRS4:
            nxt = (cur[0] + d[0], cur[1] + d[1])
            if not _inside_logical(hc, wc, nxt) or nxt in used:
                continue
            future = 0
            for d2 in DIRS4:
                nn = (nxt[0] + d2[0], nxt[1] + d2[1])
                if _inside_logical(hc, wc, nn) and nn not in used and nn != cur:
                    future += 1
            choices.append((future, rng.random(), nxt))
        choices.sort(reverse=True)
        for _, _, nxt in choices:
            used.add(nxt)
            path.append(nxt)
            if dfs():
                return True
            path.pop()
            used.remove(nxt)
        return False

    return path if dfs() else None


def _build_decoys(
    rng: random.Random, hc: int, wc: int, trunk: list[tuple[int, int]]
) -> dict[int, list[list[tuple[int, int]]]] | None:
    used = set(trunk)
    branches: dict[int, list[list[tuple[int, int]]]] = {}
    for i in range(1, len(trunk) - 1):
        cur = trunk[i]
        back = _sub(trunk[i - 1], cur)
        forward = _sub(trunk[i + 1], cur)
        available: list[tuple[tuple[int, int], int]] = []
        for d in DIRS4:
            if d == back or d == forward:
                continue
            max_len = 0
            probe = cur
            for _ in range(3):
                probe = (probe[0] + d[0], probe[1] + d[1])
                if not _inside_logical(hc, wc, probe) or probe in used:
                    break
                max_len += 1
            if max_len > 0:
                available.append((d, max_len))
        if not available:
            return None
        desired = rng.choice([1, 2])
        rng.shuffle(available)
        chosen = available[: min(desired, len(available))]
        if not chosen:
            return None
        branches[i] = []
        for d, max_len in chosen:
            length = rng.randint(1, max_len)
            branch: list[tuple[int, int]] = []
            cell = cur
            ok = True
            for _ in range(length):
                cell = (cell[0] + d[0], cell[1] + d[1])
                if cell in used:
                    ok = False
                    break
                branch.append(cell)
            if not ok:
                return None
            for cell in branch:
                used.add(cell)
            branches[i].append(branch)
    return branches


def _anchor(cell: tuple[int, int]) -> tuple[int, int]:
    return CANVAS_MARGIN + cell[0] * STEP, CANVAS_MARGIN + cell[1] * STEP


def _draw_segment(grid: np.ndarray, a: tuple[int, int], b: tuple[int, int], color: int) -> None:
    if a[0] == b[0]:
        r = a[0]
        c0, c1 = sorted((a[1], b[1]))
        grid[r, c0 : c1 + 1] = color
    else:
        c = a[1]
        r0, r1 = sorted((a[0], b[0]))
        grid[r0 : r1 + 1, c] = color


def _start_room_top_left(anchor: tuple[int, int], room_dir: tuple[int, int]) -> tuple[int, int]:
    r, c = anchor
    if room_dir == (0, -1):
        return r - 1, c - 2
    if room_dir == (0, 1):
        return r - 1, c + 1
    if room_dir == (-1, 0):
        return r - 2, c - 1
    return r + 1, c - 1


def _room_outer_top_left(anchor: tuple[int, int], incoming: tuple[int, int]) -> tuple[int, int]:
    r, c = anchor
    if incoming == (0, 1):
        return r - 1, c
    if incoming == (0, -1):
        return r - 1, c - 3
    if incoming == (1, 0):
        return r, c - 1
    return r - 3, c - 1


def _draw_room(grid: np.ndarray, anchor: tuple[int, int], incoming: tuple[int, int]) -> tuple[int, int]:
    r0, c0 = _room_outer_top_left(anchor, incoming)
    grid[r0 : r0 + 4, c0 : c0 + 4] = CORRIDOR
    grid[r0 + 1 : r0 + 3, c0 + 1 : c0 + 3] = BG
    return r0 + 1, c0 + 1


def _add_widening(
    grid: np.ndarray, rng: random.Random, anchor: tuple[int, int], direction: tuple[int, int]
) -> bool:
    length = rng.choice([2, 3])
    offset = rng.choice([2, 3])
    for side in rng.sample(_perp(direction), k=2):
        cells = []
        ok = True
        for k in range(offset, offset + length):
            rr = anchor[0] + direction[0] * k + side[0]
            cc = anchor[1] + direction[1] * k + side[1]
            mr = anchor[0] + direction[0] * k
            mc = anchor[1] + direction[1] * k
            if not _in_bounds(*grid.shape, rr, cc) or int(grid[mr, mc]) != CORRIDOR:
                ok = False
                break
            if int(grid[rr, cc]) != BG:
                ok = False
                break
            cells.append((rr, cc))
        if ok:
            for rr, cc in cells:
                grid[rr, cc] = CORRIDOR
            return True
    return False


def _expand_polyline(points: list[tuple[int, int]]) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for i in range(len(points) - 1):
        a = points[i]
        b = points[i + 1]
        if a[0] == b[0]:
            step = 1 if b[1] > a[1] else -1
            cells = [(a[0], c) for c in range(a[1], b[1] + step, step)]
        else:
            step = 1 if b[0] > a[0] else -1
            cells = [(r, a[1]) for r in range(a[0], b[0] + step, step)]
        if out:
            cells = cells[1:]
        out.extend(cells)
    return out


def _crop_pad(
    grid: np.ndarray,
    centerline: list[tuple[int, int]],
    room_top_left: tuple[int, int],
    rng: random.Random,
) -> tuple[np.ndarray, list[tuple[int, int]], tuple[int, int]]:
    pts = np.argwhere(grid != BG)
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    cropped = grid[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]
    top = rng.randint(1, 4)
    bottom = rng.randint(1, 4)
    left = rng.randint(1, 4)
    right = rng.randint(1, 4)
    out = np.full((cropped.shape[0] + top + bottom, cropped.shape[1] + left + right), BG, dtype=int)
    out[top : top + cropped.shape[0], left : left + cropped.shape[1]] = cropped
    shifted_line = [(r - int(r0) + top, c - int(c0) + left) for r, c in centerline]
    shifted_room = (room_top_left[0] - int(r0) + top, room_top_left[1] - int(c0) + left)
    return out, shifted_line, shifted_room


def _expected_output(grid: np.ndarray, centerline: list[tuple[int, int]], room_top_left: tuple[int, int]) -> np.ndarray:
    out = grid.copy()
    for r, c in _paint_from_centerline(grid, centerline, CORRIDOR):
        out[r, c] = ORANGE
    rr, cc = room_top_left
    out[rr : rr + 2, cc : cc + 2] = START
    return out


def _make_instance(rng: random.Random) -> tuple[np.ndarray, np.ndarray]:
    hc = rng.randint(6, 9)
    wc = rng.randint(6, 9)
    trunk_nodes = rng.randint(4, 6)
    trunk = None
    decoys = None
    for _ in range(120):
        trunk = _logical_path(rng, hc, wc, trunk_nodes)
        if trunk is None:
            continue
        decoys = _build_decoys(rng, hc, wc, trunk)
        if decoys is not None:
            break
    assert trunk is not None and decoys is not None, "failed to build a logical network"

    h = 2 * CANVAS_MARGIN + (hc - 1) * STEP
    w = 2 * CANVAS_MARGIN + (wc - 1) * STEP
    grid = np.full((h, w), BG, dtype=int)

    trunk_anchors = [_anchor(cell) for cell in trunk]
    for i in range(len(trunk_anchors) - 1):
        _draw_segment(grid, trunk_anchors[i], trunk_anchors[i + 1], CORRIDOR)
    for i, branches in decoys.items():
        start = _anchor(trunk[i])
        for branch in branches:
            prev = start
            for cell in branch:
                cur = _anchor(cell)
                _draw_segment(grid, prev, cur, CORRIDOR)
                prev = cur

    start_dir = _sub(trunk[1], trunk[0])
    sr, sc = _start_room_top_left(trunk_anchors[0], (-start_dir[0], -start_dir[1]))
    grid[sr : sr + 2, sc : sc + 2] = START

    target_room = _draw_room(grid, trunk_anchors[-1], _sub(trunk[-1], trunk[-2]))
    for i, branches in decoys.items():
        for branch in branches:
            leaf_anchor = _anchor(branch[-1])
            parent = branch[-2] if len(branch) > 1 else trunk[i]
            incoming = _sub(branch[-1], parent)
            _draw_room(grid, leaf_anchor, incoming)

    for i in range(1, len(trunk) - 1):
        ok = _add_widening(grid, rng, trunk_anchors[i], _sub(trunk[i + 1], trunk[i]))
        assert ok, "failed to place a widening marker"

    centerline = _expand_polyline(trunk_anchors)
    grid, centerline, target_room = _crop_pad(grid, centerline, target_room, rng)
    expected = _expected_output(grid, centerline, target_room)
    return grid, expected


def generate_input() -> np.ndarray:
    for _ in range(300):
        rng = random.Random(random.randrange(1 << 30))
        try:
            grid, expected = _make_instance(rng)
            if len(_find_endpoint_rooms(grid, BG, CORRIDOR)[0]) < 2:
                continue
            if not np.array_equal(main(grid), expected):
                continue
            return grid
        except AssertionError:
            continue
    raise RuntimeError("failed to generate a valid instance")
