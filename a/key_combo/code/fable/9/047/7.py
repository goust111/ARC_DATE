from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


BACKGROUND = 0
PALETTE = [1, 2, 3, 4, 5, 6, 7, 8, 9]

GRID_H_RANGE = (16, 24)
GRID_W_RANGE = (16, 24)
COLOR_COUNT_RANGE = (2, 4)
OBJECT_SIZE_RANGE = (4, 8)
WINDOW_RADIUS_RANGE = (3, 5)
GEN_ATTEMPTS = 500
LOCAL_ATTEMPTS = 160

PARITY_MODES = [(0, 0), (1, 0), (0, 1), (1, 1)]
PAIR_ORIENTATIONS = ("horizontal", "vertical", "diag", "oblique")


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _rotate_point(sum_r: int, sum_c: int, r: int, c: int) -> tuple[int, int]:
    return int(sum_r - r), int(sum_c - c)


def _rotate_coords(coords: np.ndarray, sum_r: int, sum_c: int) -> np.ndarray:
    return np.array([_rotate_point(sum_r, sum_c, int(r), int(c)) for r, c in coords], dtype=int)


def _parse_color_roles(grid: np.ndarray) -> dict[int, dict[str, np.ndarray | tuple[int, int]]]:
    bg = _get_bg(grid)
    roles: dict[int, dict[str, np.ndarray | tuple[int, int]]] = {}
    for color in sorted(int(v) for v in np.unique(grid) if int(v) != bg):
        comps = _components(grid == color)
        singles = [comp[0] for comp in comps if len(comp) == 1]
        objects = [comp for comp in comps if len(comp) > 1]
        assert len(singles) == 2, f"color {color} must have exactly two singleton markers"
        assert len(objects) == 1, f"color {color} must have exactly one multi-cell object"
        p = tuple(map(int, singles[0]))
        q = tuple(map(int, singles[1]))
        roles[color] = {"object": objects[0], "p": p, "q": q}
    return roles


def main(grid: np.ndarray) -> np.ndarray:
    """Complete each colored object by 180-degree rotation around the midpoint of its singleton pair.

    Input structure:
    - A dominant background contains several colors, and each non-background color appears in exactly three 4-connected components.
    - For each color, one component is a multi-cell foreground object and the other two are isolated single pixels.
    - The singleton pair may define either a cell-center midpoint or a half-cell midpoint, but both markers stay inside the grid and away from the object.

    Rule:
    - For each color independently, take the midpoint of its two singleton pixels as the center of a 180-degree rotation.
    - Rotate the multi-cell object around that center, paint the rotated copy in the same color, and union it with the original object.
    - Erase all singleton pixels in the final output while leaving the background unchanged.

    Key insight:
    - The two same-color singletons are not line endpoints; together they encode the invisible rotation center that completes the object.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    bg = _get_bg(g)
    roles = _parse_color_roles(g)
    h, w = g.shape

    for color, role in roles.items():
        obj = role["object"]
        p = role["p"]
        q = role["q"]
        sum_r = p[0] + q[0]
        sum_c = p[1] + q[1]

        out[p] = bg
        out[q] = bg
        for r, c in obj:
            rr, cc = _rotate_point(sum_r, sum_c, int(r), int(c))
            assert 0 <= rr < h and 0 <= cc < w, "rotated copy must stay in bounds"
            out[rr, cc] = color

    return out


def _bresenham(p: tuple[int, int], q: tuple[int, int]) -> list[tuple[int, int]]:
    r0, c0 = p
    r1, c1 = q
    dr = abs(r1 - r0)
    dc = abs(c1 - c0)
    sr = 1 if r0 < r1 else -1
    sc = 1 if c0 < c1 else -1
    err = dr - dc
    pts = []
    while True:
        pts.append((r0, c0))
        if r0 == r1 and c0 == c1:
            break
        e2 = 2 * err
        if e2 > -dc:
            err -= dc
            r0 += sr
        if e2 < dr:
            err += dr
            c0 += sc
    return pts


def _expanded_mask(shape: tuple[int, int], cells: set[tuple[int, int]]) -> np.ndarray:
    h, w = shape
    mask = np.zeros((h, w), dtype=bool)
    for r, c in cells:
        for rr in range(r - 1, r + 2):
            for cc in range(c - 1, c + 2):
                if 0 <= rr < h and 0 <= cc < w:
                    mask[rr, cc] = True
    return mask


def _orientation_matches(dr: int, dc: int, orientation: str) -> bool:
    adr = abs(dr)
    adc = abs(dc)
    if adr <= 1 and adc <= 1:
        return False
    if orientation == "horizontal":
        return adc >= 5 and adr <= 1
    if orientation == "vertical":
        return adr >= 5 and adc <= 1
    if orientation == "diag":
        return adr >= 4 and adc >= 4 and abs(adr - adc) <= 1
    assert orientation == "oblique"
    return adr >= 3 and adc >= 4 and abs(adr - adc) >= 2


def _sample_pair(
    h: int,
    w: int,
    sum_r: int,
    sum_c: int,
    blocked: np.ndarray,
) -> tuple[tuple[int, int], tuple[int, int]] | None:
    orientation = random.choice(PAIR_ORIENTATIONS)
    candidates = []
    for r in range(h):
        for c in range(w):
            rr, cc = _rotate_point(sum_r, sum_c, r, c)
            if not _in_bounds((h, w), rr, cc):
                continue
            p = (r, c)
            q = (rr, cc)
            if p >= q:
                continue
            if blocked[p] or blocked[q]:
                continue
            if max(abs(rr - r), abs(cc - c)) < 3:
                continue
            if not _orientation_matches(rr - r, cc - c, orientation):
                continue
            candidates.append((p, q))
    if not candidates:
        return None
    return random.choice(candidates)


def _candidate_cells(
    h: int,
    w: int,
    sum_r: int,
    sum_c: int,
    p: tuple[int, int],
    q: tuple[int, int],
    blocked: np.ndarray,
) -> list[tuple[int, int]]:
    radius_r = random.randint(*WINDOW_RADIUS_RANGE)
    radius_c = random.randint(*WINDOW_RADIUS_RANGE)
    center_r = sum_r / 2.0
    center_c = sum_c / 2.0
    r0 = max(0, int(np.floor(center_r)) - radius_r)
    r1 = min(h - 1, int(np.ceil(center_r)) + radius_r)
    c0 = max(0, int(np.floor(center_c)) - radius_c)
    c1 = min(w - 1, int(np.ceil(center_c)) + radius_c)

    cells = []
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            rr, cc = _rotate_point(sum_r, sum_c, r, c)
            if not _in_bounds((h, w), rr, cc):
                continue
            if (r, c) >= (rr, cc):
                continue
            if (r, c) in {p, q} or (rr, cc) in {p, q}:
                continue
            if blocked[r, c] or blocked[rr, cc]:
                continue
            if max(abs(r - p[0]), abs(c - p[1])) <= 1:
                continue
            if max(abs(r - q[0]), abs(c - q[1])) <= 1:
                continue
            if max(abs(rr - p[0]), abs(cc - p[1])) <= 1:
                continue
            if max(abs(rr - q[0]), abs(cc - q[1])) <= 1:
                continue
            cells.append((r, c))
    random.shuffle(cells)
    return cells


def _grow_connected_shape(candidates: list[tuple[int, int]], target_size: int) -> np.ndarray | None:
    cand_set = set(candidates)
    if len(cand_set) < target_size:
        return None

    for _ in range(LOCAL_ATTEMPTS):
        seed = random.choice(candidates)
        chosen = {seed}
        while len(chosen) < target_size:
            frontier = []
            for r, c in chosen:
                for nr, nc in _neighbors4(r, c):
                    if (nr, nc) in cand_set and (nr, nc) not in chosen:
                        frontier.append((nr, nc))
            if not frontier:
                break
            frontier = list(dict.fromkeys(frontier))
            weights = []
            for cell in frontier:
                degree = sum((nbr in cand_set) for nbr in _neighbors4(*cell))
                weights.append(max(1, degree))
            chosen.add(random.choices(frontier, weights=weights, k=1)[0])

        if len(chosen) != target_size:
            continue

        coords = np.array(sorted(chosen), dtype=int)
        r0, c0, r1, c1 = _bbox(coords)
        height = r1 - r0 + 1
        width = c1 - c0 + 1
        if height == 1 and width == 1:
            continue
        if target_size >= 5 and (height == 1 or width == 1):
            continue
        if target_size == height * width:
            continue
        return coords
    return None


def _place_color(
    h: int,
    w: int,
    color: int,
    parity: tuple[int, int],
    blocked: np.ndarray,
) -> dict[str, object] | None:
    par_r, par_c = parity
    row_options = [v for v in range(4, 2 * h - 4) if v % 2 == par_r]
    col_options = [v for v in range(4, 2 * w - 4) if v % 2 == par_c]
    if not row_options or not col_options:
        return None

    for _ in range(LOCAL_ATTEMPTS):
        sum_r = random.choice(row_options)
        sum_c = random.choice(col_options)
        pair = _sample_pair(h, w, sum_r, sum_c, blocked)
        if pair is None:
            continue
        p, q = pair
        candidates = _candidate_cells(h, w, sum_r, sum_c, p, q, blocked)
        if not candidates:
            continue
        target_size = random.randint(*OBJECT_SIZE_RANGE)
        obj = _grow_connected_shape(candidates, target_size)
        if obj is None:
            continue
        rot = _rotate_coords(obj, sum_r, sum_c)
        obj_set = {tuple(map(int, cell)) for cell in obj.tolist()}
        rot_set = {tuple(map(int, cell)) for cell in rot.tolist()}
        if obj_set & rot_set:
            continue

        full_cells = obj_set | rot_set | {p, q}
        if any(blocked[r, c] for r, c in full_cells):
            continue

        if any(max(abs(r - p[0]), abs(c - p[1])) <= 1 for r, c in obj_set | rot_set):
            continue
        if any(max(abs(r - q[0]), abs(c - q[1])) <= 1 for r, c in obj_set | rot_set):
            continue
        if max(abs(p[0] - q[0]), abs(p[1] - q[1])) <= 1:
            continue

        return {
            "color": color,
            "sum_r": sum_r,
            "sum_c": sum_c,
            "object": obj,
            "rotated": rot,
            "p": p,
            "q": q,
        }
    return None


def _erase_singletons(grid: np.ndarray) -> np.ndarray:
    g = grid.copy()
    bg = _get_bg(g)
    for role in _parse_color_roles(g).values():
        g[role["p"]] = bg
        g[role["q"]] = bg
    return g


def _line_shortcut(grid: np.ndarray) -> np.ndarray:
    g = _erase_singletons(grid)
    for color, role in _parse_color_roles(grid).items():
        for r, c in _bresenham(role["p"], role["q"]):
            g[r, c] = color
    return g


def _bbox_center_shortcut(grid: np.ndarray) -> np.ndarray:
    g = _erase_singletons(grid)
    h, w = g.shape
    for color, role in _parse_color_roles(grid).items():
        obj = role["object"]
        r0, c0, r1, c1 = _bbox(obj)
        sum_r = r0 + r1
        sum_c = c0 + c1
        for r, c in obj:
            rr, cc = _rotate_point(sum_r, sum_c, int(r), int(c))
            if 0 <= rr < h and 0 <= cc < w:
                g[rr, cc] = color
    return g


def _render_input(shape: tuple[int, int], placements: list[dict[str, object]]) -> np.ndarray:
    h, w = shape
    grid = np.full((h, w), BACKGROUND, dtype=int)
    for item in placements:
        color = int(item["color"])
        obj = np.asarray(item["object"], dtype=int)
        grid[obj[:, 0], obj[:, 1]] = color
        grid[item["p"]] = color
        grid[item["q"]] = color
    return grid


def _render_output(shape: tuple[int, int], placements: list[dict[str, object]]) -> np.ndarray:
    h, w = shape
    grid = np.full((h, w), BACKGROUND, dtype=int)
    for item in placements:
        color = int(item["color"])
        obj = np.asarray(item["object"], dtype=int)
        rot = np.asarray(item["rotated"], dtype=int)
        grid[obj[:, 0], obj[:, 1]] = color
        grid[rot[:, 0], rot[:, 1]] = color
    return grid


def _validate_instance(inp: np.ndarray, expected: np.ndarray, placements: list[dict[str, object]]) -> bool:
    if not np.array_equal(main(inp), expected):
        return False
    if np.array_equal(_erase_singletons(inp), expected):
        return False
    if np.array_equal(_line_shortcut(inp), expected):
        return False
    if np.array_equal(_bbox_center_shortcut(inp), expected):
        return False

    roles = _parse_color_roles(inp)
    has_half_center = False
    has_integer_center = False
    for item in placements:
        color = int(item["color"])
        role = roles[color]
        p = role["p"]
        q = role["q"]
        if (p[0] + q[0]) % 2 or (p[1] + q[1]) % 2:
            has_half_center = True
        else:
            has_integer_center = True

        obj = role["object"]
        rot = _rotate_coords(obj, p[0] + q[0], p[1] + q[1])
        rot_set = {tuple(map(int, cell)) for cell in rot.tolist()}
        obj_set = {tuple(map(int, cell)) for cell in obj.tolist()}
        if rot_set & obj_set:
            return False
        if len(rot_set) != len(obj_set):
            return False

    if len(placements) >= 2 and not (has_half_center and has_integer_center):
        return False
    return True


def _choose_parities(k: int) -> list[tuple[int, int]]:
    modes = []
    if k >= 2:
        modes.append((0, 0))
        modes.append(random.choice(PARITY_MODES[1:]))
    while len(modes) < k:
        modes.append(random.choice(PARITY_MODES))
    random.shuffle(modes)
    return modes


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    count = random.randint(*COLOR_COUNT_RANGE)
    colors = random.sample(PALETTE, count)
    parities = _choose_parities(count)

    placements: list[dict[str, object]] = []
    occupied_cells: set[tuple[int, int]] = set()
    blocked = np.zeros((h, w), dtype=bool)

    for color, parity in zip(colors, parities):
        placed = _place_color(h, w, color, parity, blocked)
        if placed is None:
            return None
        placements.append(placed)

        obj = {tuple(map(int, cell)) for cell in np.asarray(placed["object"], dtype=int).tolist()}
        rot = {tuple(map(int, cell)) for cell in np.asarray(placed["rotated"], dtype=int).tolist()}
        full_cells = obj | rot | {placed["p"], placed["q"]}
        occupied_cells |= full_cells
        blocked |= _expanded_mask((h, w), full_cells)

    inp = _render_input((h, w), placements)
    expected = _render_output((h, w), placements)
    if not _validate_instance(inp, expected, placements):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
