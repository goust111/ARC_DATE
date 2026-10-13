from __future__ import annotations

import random

import numpy as np


_BG = 0
_NON_BG_COLORS = tuple(range(1, 10))
_DIRS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
_GEN_ATTEMPTS = 400


def _components_mask(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    """Return 4-connected components of a boolean mask."""
    H, W = map(int, mask.shape)
    seen = np.zeros((H, W), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(H):
        for c in range(W):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < H and 0 <= nc < W and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _components_by_color(grid: np.ndarray) -> list[tuple[int, list[tuple[int, int]]]]:
    """Return monochrome 4-connected components of all non-background pixels."""
    g = np.asarray(grid, dtype=int)
    H, W = map(int, g.shape)
    seen = np.zeros((H, W), dtype=bool)
    comps: list[tuple[int, list[tuple[int, int]]]] = []
    for r in range(H):
        for c in range(W):
            color = int(g[r, c])
            if color == _BG or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < H and 0 <= nc < W and not seen[nr, nc] and int(g[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append((color, comp))
    return comps


def _parse_tower(color: int, coords: list[tuple[int, int]]) -> dict | None:
    """Recognize a 2x2 body plus one same-color antenna cell."""
    if len(coords) != 5:
        return None
    cells = set(coords)
    rows = [r for r, _c in coords]
    cols = [c for _r, c in coords]
    min_r, max_r = min(rows), max(rows)
    min_c, max_c = min(cols), max(cols)
    candidates: list[dict] = []
    for r0 in range(min_r - 1, max_r + 1):
        for c0 in range(min_c - 1, max_c + 1):
            body = {(r0, c0), (r0, c0 + 1), (r0 + 1, c0), (r0 + 1, c0 + 1)}
            if not body.issubset(cells):
                continue
            extra = cells - body
            if len(extra) != 1:
                continue
            antenna = next(iter(extra))
            ar, ac = antenna
            direction = None
            if ar == r0 - 1 and ac in (c0, c0 + 1):
                direction = "up"
            elif ar == r0 + 2 and ac in (c0, c0 + 1):
                direction = "down"
            elif ac == c0 - 1 and ar in (r0, r0 + 1):
                direction = "left"
            elif ac == c0 + 2 and ar in (r0, r0 + 1):
                direction = "right"
            if direction is None:
                continue
            candidates.append(
                {
                    "color": int(color),
                    "body_cells": body,
                    "tower_cells": set(cells),
                    "antenna": antenna,
                    "direction": direction,
                    "anchor": (r0, c0),
                }
            )
    if len(candidates) != 1:
        return None
    return candidates[0]


def _detect_scene(grid: np.ndarray) -> tuple[list[dict], set[tuple[int, int]], set[tuple[int, int]], dict[tuple[int, int], tuple[int, int]]]:
    g = np.asarray(grid, dtype=int)
    towers: list[dict] = []
    tower_cells: set[tuple[int, int]] = set()
    body_cells: set[tuple[int, int]] = set()
    for color, coords in _components_by_color(g):
        tower = _parse_tower(color, coords)
        if tower is None:
            continue
        towers.append(tower)
        tower_cells.update(tower["tower_cells"])
        body_cells.update(tower["body_cells"])

    assert towers, "expected at least one signal tower"

    mask = g != _BG
    for r, c in tower_cells:
        mask[r, c] = False

    prism_map: dict[tuple[int, int], tuple[int, int]] = {}
    for comp in _components_mask(mask):
        assert len(comp) == 2, "every non-tower foreground object must be a two-cell prism"
        a, b = comp
        assert abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1, "prisms must be 1x2 or 2x1"
        assert int(g[a]) != int(g[b]), "prism halves must have different colors"
        prism_map[a] = b
        prism_map[b] = a

    towers.sort(key=lambda tower: (tower["anchor"][0], tower["anchor"][1], tower["antenna"][0], tower["antenna"][1]))
    return towers, tower_cells, body_cells, prism_map


def _ray_trace(
    tower: dict,
    body_cells: set[tuple[int, int]],
    shape: tuple[int, int],
) -> tuple[list[tuple[int, int]], tuple[int, int] | None]:
    H, W = map(int, shape)
    dr, dc = _DIRS[tower["direction"]]
    r = tower["antenna"][0] + dr
    c = tower["antenna"][1] + dc
    path: list[tuple[int, int]] = []
    stop_cell: tuple[int, int] | None = None
    while 0 <= r < H and 0 <= c < W:
        if (r, c) in body_cells:
            stop_cell = (r, c)
            break
        path.append((r, c))
        r += dr
        c += dc
    return path, stop_cell


def main(grid: np.ndarray) -> np.ndarray:
    """
    Cast rays from each tower and let two-color prisms rewrite the later ray color.

    Input structure:
    - The background is black.
    - Each signal tower is a monochrome 2x2 body with one same-color antenna pixel attached to one side.
    - Every prism is a detached two-cell domino, perpendicular to the ray it serves, with two different colors; the ray hits one cell and the other cell supplies the outgoing color.

    Rule:
    - Identify every tower, read the antenna side as its emission direction, and process towers in top-to-bottom, left-to-right order by body position.
    - From the cell immediately beyond the antenna tip, extend a one-pixel-wide ray until it would leave the grid or enter another tower body.
    - Paint only background cells along that path; when the ray passes through a prism cell, keep the prism unchanged and switch the ray's later color to the prism's partner cell color.
    - If multiple rays reach the same background cell, the later tower in scan order overwrites the earlier ray color.

    Key insight:
    - The 2x2-plus-one geometry simultaneously identifies the source and its direction, while each two-color prism is a color transformer rather than a blocker.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    towers, tower_cells, body_cells, prism_map = _detect_scene(g)

    out = g.copy()
    H, W = map(int, g.shape)
    for tower in towers:
        current_color = int(tower["color"])
        dr, dc = _DIRS[tower["direction"]]
        r = tower["antenna"][0] + dr
        c = tower["antenna"][1] + dc
        while 0 <= r < H and 0 <= c < W:
            pos = (r, c)
            if pos in body_cells:
                break
            if pos in prism_map:
                current_color = int(g[prism_map[pos]])
            elif pos not in tower_cells:
                out[r, c] = current_color
            r += dr
            c += dc
    return out


def _can_place_object(
    cells: set[tuple[int, int]],
    occupied: set[tuple[int, int]],
    H: int,
    W: int,
) -> bool:
    for r, c in cells:
        if not (0 <= r < H and 0 <= c < W):
            return False
        if (r, c) in occupied:
            return False
    for r, c in cells:
        for rr in range(r - 1, r + 2):
            for cc in range(c - 1, c + 2):
                if (rr, cc) in occupied and (rr, cc) not in cells:
                    return False
    return True


def _make_tower(r0: int, c0: int, direction: str, offset: int, color: int) -> dict:
    body = {(r0, c0), (r0, c0 + 1), (r0 + 1, c0), (r0 + 1, c0 + 1)}
    if direction == "up":
        antenna = (r0 - 1, c0 + offset)
    elif direction == "down":
        antenna = (r0 + 2, c0 + offset)
    elif direction == "left":
        antenna = (r0 + offset, c0 - 1)
    else:
        antenna = (r0 + offset, c0 + 2)
    return {
        "color": int(color),
        "body_cells": body,
        "tower_cells": body | {antenna},
        "antenna": antenna,
        "direction": direction,
        "anchor": (r0, c0),
    }


def _paint_cells(grid: np.ndarray, cells: set[tuple[int, int]], color: int) -> None:
    for r, c in cells:
        grid[r, c] = int(color)


def _random_tower_candidate(H: int, W: int, color: int) -> dict:
    direction = random.choice(tuple(_DIRS))
    offset = random.randint(0, 1)
    if direction == "up":
        r0 = random.randint(4, H - 3)
        c0 = random.randint(1, W - 3)
    elif direction == "down":
        r0 = random.randint(1, H - 6)
        c0 = random.randint(1, W - 3)
    elif direction == "left":
        r0 = random.randint(1, H - 3)
        c0 = random.randint(4, W - 3)
    else:
        r0 = random.randint(1, H - 3)
        c0 = random.randint(1, W - 6)
    return _make_tower(r0, c0, direction, offset, color)


def _place_cross_pair(H: int, W: int, colors: list[int]) -> list[dict] | None:
    for _ in range(80):
        cross_r = random.randint(6, H - 7)
        cross_c = random.randint(6, W - 7)

        hdir = random.choice(("left", "right"))
        vdir = random.choice(("up", "down"))
        hoff = random.randint(0, 1)
        voff = random.randint(0, 1)

        if hdir == "right":
            max_dist = min(6, cross_c - 4)
            if max_dist < 2:
                continue
            dist = random.randint(2, max_dist)
            h_c0 = cross_c - dist - 3
        else:
            max_dist = min(6, W - cross_c - 4)
            if max_dist < 2:
                continue
            dist = random.randint(2, max_dist)
            h_c0 = cross_c + dist + 2
        h_r0 = cross_r - hoff

        if vdir == "down":
            max_dist = min(6, cross_r - 4)
            if max_dist < 2:
                continue
            dist = random.randint(2, max_dist)
            v_r0 = cross_r - dist - 3
        else:
            max_dist = min(6, H - cross_r - 4)
            if max_dist < 2:
                continue
            dist = random.randint(2, max_dist)
            v_r0 = cross_r + dist + 2
        v_c0 = cross_c - voff

        towers = [
            _make_tower(h_r0, h_c0, hdir, hoff, colors[0]),
            _make_tower(v_r0, v_c0, vdir, voff, colors[1]),
        ]
        occupied: set[tuple[int, int]] = set()
        ok = True
        for tower in towers:
            if not _can_place_object(tower["tower_cells"], occupied, H, W):
                ok = False
                break
            occupied.update(tower["tower_cells"])
        if ok:
            return towers
    return None


def _place_block_pair(H: int, W: int, colors: list[int]) -> list[dict] | None:
    for _ in range(80):
        axis = random.choice(("h", "v"))
        if axis == "h":
            ray_row = random.randint(5, H - 6)
            edir = random.choice(("left", "right"))
            eoff = random.randint(0, 1)
            toff = random.randint(0, 1)
            tdir = random.choice(("up", "down"))
            t_r0 = ray_row - toff
            if tdir == "up" and t_r0 < 1:
                continue
            if tdir == "down" and t_r0 > H - 3:
                continue
            if edir == "right":
                t_c0 = random.randint(8, W - 4)
                max_gap = min(6, t_c0 - 4)
                if max_gap < 2:
                    continue
                gap = random.randint(2, max_gap)
                e_c0 = t_c0 - gap - 3
            else:
                t_c0 = random.randint(2, W - 10)
                max_gap = min(6, W - t_c0 - 6)
                if max_gap < 2:
                    continue
                gap = random.randint(2, max_gap)
                e_c0 = t_c0 + gap + 2
            e_r0 = ray_row - eoff
            towers = [
                _make_tower(e_r0, e_c0, edir, eoff, colors[0]),
                _make_tower(t_r0, t_c0, tdir, toff, colors[1]),
            ]
        else:
            ray_col = random.randint(5, W - 6)
            edir = random.choice(("up", "down"))
            eoff = random.randint(0, 1)
            toff = random.randint(0, 1)
            tdir = random.choice(("left", "right"))
            t_c0 = ray_col - toff
            if tdir == "left" and t_c0 < 1:
                continue
            if tdir == "right" and t_c0 > W - 3:
                continue
            if edir == "down":
                t_r0 = random.randint(8, H - 4)
                max_gap = min(6, t_r0 - 4)
                if max_gap < 2:
                    continue
                gap = random.randint(2, max_gap)
                e_r0 = t_r0 - gap - 3
            else:
                t_r0 = random.randint(2, H - 10)
                max_gap = min(6, H - t_r0 - 6)
                if max_gap < 2:
                    continue
                gap = random.randint(2, max_gap)
                e_r0 = t_r0 + gap + 2
            e_c0 = ray_col - eoff
            towers = [
                _make_tower(e_r0, e_c0, edir, eoff, colors[0]),
                _make_tower(t_r0, t_c0, tdir, toff, colors[1]),
            ]

        occupied: set[tuple[int, int]] = set()
        ok = True
        for tower in towers:
            if not _can_place_object(tower["tower_cells"], occupied, H, W):
                ok = False
                break
            occupied.update(tower["tower_cells"])
        if ok:
            return towers
    return None


def _all_tower_cells(towers: list[dict]) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    for tower in towers:
        cells.update(tower["tower_cells"])
    return cells


def _all_body_cells(towers: list[dict]) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    for tower in towers:
        cells.update(tower["body_cells"])
    return cells


def _path_infos(towers: list[dict], shape: tuple[int, int]) -> list[dict]:
    body_cells = _all_body_cells(towers)
    infos: list[dict] = []
    for tower in towers:
        path, stop_cell = _ray_trace(tower, body_cells, shape)
        infos.append({"tower": tower, "path": path, "stop_cell": stop_cell})
    return infos


def _sample_extra_towers(
    towers: list[dict],
    H: int,
    W: int,
    colors: list[int],
    reserved: set[tuple[int, int]],
) -> list[dict] | None:
    occupied = _all_tower_cells(towers)
    for color in colors:
        placed = False
        for _ in range(120):
            tower = _random_tower_candidate(H, W, color)
            if tower["tower_cells"] & reserved:
                continue
            if not _can_place_object(tower["tower_cells"], occupied, H, W):
                continue
            towers.append(tower)
            occupied.update(tower["tower_cells"])
            placed = True
            break
        if not placed:
            return None
    return towers


def _validate_tower_layout(towers: list[dict], shape: tuple[int, int], mode: str) -> tuple[bool, list[dict]]:
    infos = _path_infos(towers, shape)
    tower_cells = _all_tower_cells(towers)
    for info in infos:
        own_cells = info["tower"]["tower_cells"]
        foreign = tower_cells - own_cells
        if any(cell in foreign for cell in info["path"]):
            return False, infos
        if len(info["path"]) < 3:
            return False, infos

    if mode == "cross":
        crossed = set(infos[0]["path"]) & set(infos[1]["path"])
        if not crossed:
            return False, infos
    elif mode == "block":
        stop = infos[0]["stop_cell"]
        if stop is None or stop not in infos[1]["tower"]["body_cells"]:
            return False, infos

    return True, infos


def _place_prisms(
    grid: np.ndarray,
    towers: list[dict],
    infos: list[dict],
) -> bool:
    H, W = map(int, grid.shape)
    occupied = _all_tower_cells(towers)
    all_path_cells = [cell for info in infos for cell in info["path"]]
    path_count: dict[tuple[int, int], int] = {}
    for cell in all_path_cells:
        path_count[cell] = path_count.get(cell, 0) + 1
    all_path_set = set(all_path_cells)

    prism_counts = [random.choice((0, 1, 2)) for _ in towers]
    if sum(prism_counts) == 0:
        prism_counts[random.randrange(len(prism_counts))] = 1

    for info, target_count in zip(infos, prism_counts):
        path = info["path"]
        direction = info["tower"]["direction"]
        used_idx: list[int] = []
        for _ in range(target_count):
            candidates: list[tuple[int, tuple[int, int], tuple[int, int]]] = []
            for idx, cell in enumerate(path):
                if idx == 0 or idx == len(path) - 1:
                    continue
                if path_count.get(cell, 0) != 1:
                    continue
                if any(abs(idx - prev) <= 1 for prev in used_idx):
                    continue
                r, c = cell
                if direction in ("left", "right"):
                    mates = [(r - 1, c), (r + 1, c)]
                else:
                    mates = [(r, c - 1), (r, c + 1)]
                for mate in mates:
                    if mate in all_path_set:
                        continue
                    cells = {cell, mate}
                    if _can_place_object(cells, occupied, H, W):
                        candidates.append((idx, cell, mate))
            if not candidates:
                return False
            idx, near_cell, far_cell = random.choice(candidates)
            near_color, far_color = random.sample(_NON_BG_COLORS, 2)
            grid[near_cell] = near_color
            grid[far_cell] = far_color
            occupied.update({near_cell, far_cell})
            used_idx.append(idx)
    return True


def _gen_attempt() -> np.ndarray | None:
    H = random.randint(16, 29)
    W = random.randint(16, 29)
    tower_count = random.randint(2, 4)
    mode = random.choice(("cross", "block", "free"))
    tower_colors = random.sample(list(_NON_BG_COLORS), tower_count)

    if mode == "cross":
        towers = _place_cross_pair(H, W, tower_colors[:2])
    elif mode == "block":
        towers = _place_block_pair(H, W, tower_colors[:2])
    else:
        towers = []
        occupied: set[tuple[int, int]] = set()
        for color in tower_colors[:2]:
            placed = False
            for _ in range(120):
                tower = _random_tower_candidate(H, W, color)
                if not _can_place_object(tower["tower_cells"], occupied, H, W):
                    continue
                towers.append(tower)
                occupied.update(tower["tower_cells"])
                placed = True
                break
            if not placed:
                return None

    if towers is None:
        return None

    base_infos = _path_infos(towers, (H, W))
    reserved = {cell for info in base_infos for cell in info["path"]}
    towers = _sample_extra_towers(towers, H, W, tower_colors[len(towers) :], reserved)
    if towers is None:
        return None

    ok, infos = _validate_tower_layout(towers, (H, W), mode)
    if not ok:
        return None

    grid = np.zeros((H, W), dtype=int)
    for tower in towers:
        _paint_cells(grid, tower["tower_cells"], tower["color"])

    if not _place_prisms(grid, towers, infos):
        return None

    try:
        out = main(grid)
    except AssertionError:
        return None

    if not np.any(out != grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(_GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid signal-tower prism input")
