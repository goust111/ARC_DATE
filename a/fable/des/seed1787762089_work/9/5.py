import random
from collections import Counter, deque

import numpy as np


BLACK = 0
RED = 2
GREY = 5
AZURE = 8
BODY_COLORS = [1, 3, 4, 6, 7, 9]
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
GEN_ATTEMPTS = 300


def _bg_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[int(np.argmax(counts))])


def _find_components(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) == bg:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for dr, dc in DIR4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) != bg:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            arr = np.array(coords, dtype=int)
            comps.append(
                {
                    "coords": arr,
                    "area": len(coords),
                    "colors": {int(grid[rr, cc]) for rr, cc in coords},
                }
            )
    return comps


def _degree(cell: tuple[int, int], cells: set[tuple[int, int]]) -> int:
    r, c = cell
    return sum((r + dr, c + dc) in cells for dr, dc in DIR4)


def _normalize(cells: list[tuple[int, int]]) -> list[tuple[int, int]]:
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    return [(r - min_r, c - min_c) for r, c in cells]


def _shift_cells(cells: list[tuple[int, int]], dr: int, dc: int) -> list[tuple[int, int]]:
    return [(r + dr, c + dc) for r, c in cells]


def _in_bounds(cells: list[tuple[int, int]], h: int, w: int) -> bool:
    return all(0 <= r < h and 0 <= c < w for r, c in cells)


def _touches(a: set[tuple[int, int]], b: set[tuple[int, int]], include_diagonal: bool = True) -> bool:
    if include_diagonal:
        for r, c in a:
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if (r + dr, c + dc) in b:
                        return True
        return False
    return any((r + dr, c + dc) in b for r, c in a for dr, dc in DIR4)


def _ordered_path_from_component(comp: dict) -> list[tuple[int, int]]:
    cells = {tuple(map(int, xy)) for xy in comp["coords"]}
    endpoints = [cell for cell in cells if _degree(cell, cells) == 1]
    assert len(endpoints) == 2
    ordered = [endpoints[0]]
    prev = None
    cur = endpoints[0]
    while cur != endpoints[1]:
        nxts = []
        r, c = cur
        for dr, dc in DIR4:
            nxt = (r + dr, c + dc)
            if nxt in cells and nxt != prev:
                nxts.append(nxt)
        assert len(nxts) == 1
        prev, cur = cur, nxts[0]
        ordered.append(cur)
    return ordered


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = _bg_color(g)

    comps = _find_components(g, bg)
    assert len(comps) == 3

    path_comp = None
    sprites = []
    for comp in comps:
        colors = comp["colors"]
        coords = {tuple(map(int, xy)) for xy in comp["coords"]}
        if colors == {GREY, RED, AZURE}:
            red_pts = [cell for cell in coords if int(g[cell]) == RED]
            azure_pts = [cell for cell in coords if int(g[cell]) == AZURE]
            degrees = [_degree(cell, coords) for cell in coords]
            if (
                len(red_pts) == 1
                and len(azure_pts) == 1
                and degrees.count(1) == 2
                and degrees.count(2) == len(coords) - 2
                and _degree(red_pts[0], coords) == 1
                and _degree(azure_pts[0], coords) == 1
            ):
                path_comp = comp
                continue
        sprites.append(comp)

    assert path_comp is not None
    assert len(sprites) == 2

    ordered_path = _ordered_path_from_component(path_comp)
    first, last = ordered_path[0], ordered_path[-1]
    if int(g[first]) == RED and int(g[last]) == AZURE:
        red_end, azure_end = first, last
    elif int(g[first]) == AZURE and int(g[last]) == RED:
        red_end, azure_end = last, first
    else:
        raise AssertionError("path endpoints must be red and azure")

    v = (azure_end[0] - red_end[0], azure_end[1] - red_end[1])
    assert v != (0, 0)

    out = np.full((h, w), bg, dtype=int)
    occupied = np.zeros((h, w), dtype=bool)

    for comp in sprites:
        coords = comp["coords"]
        colors = comp["colors"]
        assert 4 <= int(comp["area"]) <= 6
        assert len(colors) == 2
        marker_colors = colors & {RED, AZURE}
        body_colors = colors - {RED, AZURE}
        assert len(marker_colors) == 1 and len(body_colors) == 1
        marker_color = next(iter(marker_colors))

        dr, dc = v if marker_color == RED else (-v[0], -v[1])
        shifted = coords + np.array([dr, dc], dtype=int)
        assert bool(np.all((0 <= shifted[:, 0]) & (shifted[:, 0] < h)))
        assert bool(np.all((0 <= shifted[:, 1]) & (shifted[:, 1] < w)))
        assert not bool(np.any(occupied[shifted[:, 0], shifted[:, 1]]))

        values = g[coords[:, 0], coords[:, 1]]
        out[shifted[:, 0], shifted[:, 1]] = values
        occupied[shifted[:, 0], shifted[:, 1]] = True

    return out


def _random_path(length: int) -> list[tuple[int, int]] | None:
    def dfs(path: list[tuple[int, int]], used: set[tuple[int, int]]) -> bool:
        if len(path) == length:
            return True
        cur = path[-1]
        dirs = DIR4[:]
        random.shuffle(dirs)
        for dr, dc in dirs:
            nxt = (cur[0] + dr, cur[1] + dc)
            if nxt in used:
                continue
            if any((nxt[0] + rr, nxt[1] + cc) in used and (nxt[0] + rr, nxt[1] + cc) != cur for rr, cc in DIR4):
                continue
            used.add(nxt)
            path.append(nxt)
            if dfs(path, used):
                return True
            path.pop()
            used.remove(nxt)
        return False

    for _ in range(60):
        path = [(0, 0)]
        if dfs(path, {(0, 0)}):
            norm = _normalize(path)
            dr = norm[-1][0] - norm[0][0]
            dc = norm[-1][1] - norm[0][1]
            if abs(dr) + abs(dc) >= 3:
                return norm
    return None


def _random_polyomino(area: int) -> tuple[list[tuple[int, int]], tuple[int, int]] | None:
    for _ in range(80):
        cells = {(0, 0)}
        while len(cells) < area:
            frontier = []
            for r, c in cells:
                for dr, dc in DIR4:
                    nxt = (r + dr, c + dc)
                    if nxt not in cells:
                        frontier.append(nxt)
            cells.add(random.choice(frontier))

        norm = _normalize(sorted(cells))
        cell_set = set(norm)
        marker_candidates = [cell for cell in norm if _degree(cell, cell_set) >= 2]
        if not marker_candidates:
            continue
        if len(set(r for r, _ in norm)) == 1 and len(set(c for _, c in norm)) == 1:
            continue
        return norm, random.choice(marker_candidates)
    return None


def _placements_for_sprite(
    shape: list[tuple[int, int]],
    delta: tuple[int, int],
    h: int,
    w: int,
    blocked_original: set[tuple[int, int]],
) -> list[dict]:
    max_r = max(r for r, _ in shape)
    max_c = max(c for _, c in shape)
    placements = []
    for r0 in range(h - max_r):
        for c0 in range(w - max_c):
            original = _shift_cells(shape, r0, c0)
            target = _shift_cells(original, delta[0], delta[1])
            orig_set = set(original)
            targ_set = set(target)
            if not _in_bounds(target, h, w):
                continue
            if orig_set & targ_set:
                continue
            if _touches(orig_set, blocked_original, include_diagonal=True):
                continue
            placements.append(
                {
                    "anchor": (r0, c0),
                    "original": original,
                    "target": target,
                    "orig_set": orig_set,
                    "targ_set": targ_set,
                }
            )
    random.shuffle(placements)
    return placements


def _paint_sprite(
    grid: np.ndarray,
    shape: list[tuple[int, int]],
    anchor: tuple[int, int],
    body_color: int,
    marker_color: int,
    marker_cell: tuple[int, int],
) -> None:
    ar, ac = anchor
    for r, c in shape:
        grid[ar + r, ac + c] = marker_color if (r, c) == marker_cell else body_color


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(18, 24)
    w = random.randint(18, 24)

    path = _random_path(random.randint(6, 12))
    if path is None:
        return None
    path_h = max(r for r, _ in path) + 1
    path_w = max(c for _, c in path) + 1
    if path_h > h - 2 or path_w > w - 2:
        return None

    path_r0 = random.randint(1, h - path_h - 1)
    path_c0 = random.randint(1, w - path_w - 1)
    path_abs = _shift_cells(path, path_r0, path_c0)

    if random.random() < 0.5:
        red_end = path_abs[0]
        azure_end = path_abs[-1]
    else:
        red_end = path_abs[-1]
        azure_end = path_abs[0]
    v = (azure_end[0] - red_end[0], azure_end[1] - red_end[1])
    if abs(v[0]) + abs(v[1]) < 3:
        return None

    sprite_specs = []
    body_colors = random.sample(BODY_COLORS, 2)
    marker_order = [RED, AZURE]
    random.shuffle(marker_order)
    for body_color, marker_color in zip(body_colors, marker_order, strict=True):
        made = _random_polyomino(random.randint(4, 6))
        if made is None:
            return None
        shape, marker_cell = made
        sprite_specs.append(
            {
                "shape": shape,
                "marker_cell": marker_cell,
                "body_color": body_color,
                "marker_color": marker_color,
                "delta": v if marker_color == RED else (-v[0], -v[1]),
            }
        )

    blocked_original = set(path_abs)
    placement_lists = []
    for spec in sprite_specs:
        placements = _placements_for_sprite(spec["shape"], spec["delta"], h, w, blocked_original)
        if not placements:
            return None
        placement_lists.append(placements)

    pair = None
    for place_a in placement_lists[0][:120]:
        for place_b in placement_lists[1][:120]:
            if _touches(place_a["orig_set"], place_b["orig_set"], include_diagonal=True):
                continue
            if _touches(place_a["targ_set"], place_b["targ_set"], include_diagonal=True):
                continue
            pair = (place_a, place_b)
            break
        if pair is not None:
            break
    if pair is None:
        return None

    grid = np.full((h, w), BLACK, dtype=int)
    for idx, cell in enumerate(path_abs):
        if cell == red_end:
            grid[cell] = RED
        elif cell == azure_end:
            grid[cell] = AZURE
        else:
            grid[cell] = GREY

    for spec, place in zip(sprite_specs, pair, strict=True):
        _paint_sprite(
            grid,
            spec["shape"],
            place["anchor"],
            spec["body_color"],
            spec["marker_color"],
            spec["marker_cell"],
        )

    try:
        out = main(grid)
    except AssertionError:
        return None

    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid instance")
