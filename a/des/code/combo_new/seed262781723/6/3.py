from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


BG = 0
WALL = 5
SEED_PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)

GEN_RETRIES = 400
ROOM_RETRIES = 200
RECT_RETRIES = 60
PLACEMENT_RETRIES = 160


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r, c in np.argwhere(mask):
        r = int(r)
        c = int(c)
        if seen[r, c]:
            continue
        seen[r, c] = True
        q = deque([(r, c)])
        comp: list[tuple[int, int]] = []
        while q:
            cr, cc = q.popleft()
            comp.append((cr, cc))
            for nr, nc in _neighbors4(cr, cc):
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    q.append((nr, nc))
        comps.append(comp)
    return comps


def _touches_border(comp: list[tuple[int, int]], h: int, w: int) -> bool:
    return any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in comp)


def _border_mode(grid: np.ndarray) -> int:
    h, w = grid.shape
    if h == 1:
        border = grid[0, :]
    elif w == 1:
        border = grid[:, 0]
    else:
        border = np.concatenate([grid[0, :], grid[h - 1, :], grid[1 : h - 1, 0], grid[1 : h - 1, w - 1]])
    vals, counts = np.unique(border, return_counts=True)
    return int(vals[int(np.argmax(counts))])


def _wall_score(grid: np.ndarray, bg: int, wall_color: int) -> tuple[int, int, int, int]:
    room_with_seeds = 0
    room_count = 0
    enclosed_area = 0
    for comp in _components(grid != wall_color):
        if _touches_border(comp, *grid.shape):
            continue
        room_count += 1
        enclosed_area += len(comp)
        if any(int(grid[r, c]) not in (bg, wall_color) for r, c in comp):
            room_with_seeds += 1
    wall_pixels = int(np.count_nonzero(grid == wall_color))
    return room_with_seeds, room_count, enclosed_area, wall_pixels


def _detect_wall_color(grid: np.ndarray, bg: int) -> int:
    colors = [int(color) for color in np.unique(grid) if int(color) != bg]
    assert colors, "expected at least one non-background color"
    scored = [(color, _wall_score(grid, bg, color)) for color in colors]
    best_color, best_score = max(scored, key=lambda item: item[1])
    if best_score[0] == 0 and best_score[1] == 0:
        counts = Counter(map(int, grid.reshape(-1)))
        return int(max(colors, key=lambda color: counts[int(color)]))
    return int(best_color)


def _fill_positions_from_walls(
    grid: np.ndarray,
    room_comp: list[tuple[int, int]],
    bg: int,
    wall_color: int,
    layer_count: int,
) -> set[tuple[int, int]]:
    black_cells = {(r, c) for r, c in room_comp if int(grid[r, c]) == bg}
    dist: dict[tuple[int, int], int] = {}
    q: deque[tuple[int, int]] = deque()
    for r, c in black_cells:
        if any(
            0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1] and int(grid[nr, nc]) == wall_color
            for nr, nc in _neighbors4(r, c)
        ):
            dist[(r, c)] = 1
            q.append((r, c))
    while q:
        r, c = q.popleft()
        if dist[(r, c)] >= layer_count:
            continue
        for nr, nc in _neighbors4(r, c):
            if (nr, nc) in black_cells and (nr, nc) not in dist:
                dist[(nr, nc)] = dist[(r, c)] + 1
                q.append((nr, nc))
    return {pos for pos, d in dist.items() if d <= layer_count}


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow color-matched wall-hugging layers inside each sealed grey room.

    Input structure:
    - Black background surrounds several closed grey wall loops.
    - Each loop encloses one room of black floor cells.
    - Every room contains 1 to 4 seed pixels of one non-grey color, and the seeds are not 4-adjacent to the wall.

    Rule:
    - For each enclosed room, count its seed pixels N and read their shared color.
    - Starting from the room's black cells that are 4-adjacent to the grey wall, grow inward through black cells only for N layers.
    - Recolor all black cells in those first N wall-hugging layers with the room's seed color.
    - Keep the grey walls and the original seed pixels unchanged; any deeper cells stay black.

    Key insight: The seed color chooses the paint, but the number of seeds controls how many layers are filled from the wall inward rather than from the seeds outward.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _border_mode(g)
    wall_color = _detect_wall_color(g, bg)

    out = g.copy()
    for comp in _components(g != wall_color):
        if _touches_border(comp, *g.shape):
            continue
        seeds = [(r, c) for r, c in comp if int(g[r, c]) not in (bg, wall_color)]
        if not seeds:
            continue
        seed_colors = {int(g[r, c]) for r, c in seeds}
        assert len(seed_colors) == 1, "each room must contain one seed color"
        seed_color = next(iter(seed_colors))
        for r, c in _fill_positions_from_walls(g, comp, bg, wall_color, len(seeds)):
            out[r, c] = seed_color
    return out


def _distance_inside(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    dist = np.zeros((h, w), dtype=int)
    q: deque[tuple[int, int]] = deque()
    for r, c in np.argwhere(mask):
        r = int(r)
        c = int(c)
        if any(not (0 <= nr < h and 0 <= nc < w) or not mask[nr, nc] for nr, nc in _neighbors4(r, c)):
            dist[r, c] = 1
            q.append((r, c))
    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and dist[nr, nc] == 0:
                dist[nr, nc] = dist[r, c] + 1
                q.append((nr, nc))
    return dist


def _has_hole(mask: np.ndarray) -> bool:
    inv = ~mask
    h, w = mask.shape
    return any(not _touches_border(comp, h, w) for comp in _components(inv))


def _pick_seed_positions(dist: np.ndarray, seed_count: int) -> list[tuple[int, int]] | None:
    candidates = [(int(r), int(c)) for r, c in np.argwhere(dist >= 2)]
    if len(candidates) < seed_count:
        return None
    for _ in range(80):
        ordered = sorted(candidates, key=lambda pos: (-int(dist[pos]), random.random()))
        chosen: list[tuple[int, int]] = []
        for r, c in ordered:
            if all(abs(r - rr) + abs(c - cc) > 1 for rr, cc in chosen):
                chosen.append((r, c))
                if len(chosen) == seed_count:
                    return chosen
    return None


def _make_core_shape(core_h: int, core_w: int, seed_count: int) -> tuple[np.ndarray, list[tuple[int, int]]]:
    min_area = max(seed_count + 6, (core_h * core_w) // 3)
    for _ in range(ROOM_RETRIES):
        mask = np.zeros((core_h, core_w), dtype=bool)
        rect_count = random.randint(1, 4)
        ok = True
        for rect_i in range(rect_count):
            placed = False
            for _ in range(RECT_RETRIES):
                rh = random.randint(2, core_h)
                rw = random.randint(2, core_w)
                r0 = random.randint(0, core_h - rh)
                c0 = random.randint(0, core_w - rw)
                rect = np.zeros_like(mask)
                rect[r0 : r0 + rh, c0 : c0 + rw] = True
                if rect_i > 0 and not np.any(mask & rect):
                    continue
                if np.array_equal(mask | rect, mask):
                    continue
                mask |= rect
                placed = True
                break
            if not placed:
                ok = False
                break
        if not ok:
            continue
        area = int(mask.sum())
        if area < min_area or area >= core_h * core_w:
            continue
        if len(_components(mask)) != 1 or _has_hole(mask):
            continue
        dist = _distance_inside(mask)
        if int(dist.max()) < 2:
            continue
        seeds = _pick_seed_positions(dist, seed_count)
        if seeds is None:
            continue
        return mask, seeds
    raise RuntimeError("failed to build a valid room core")


def _local_fill_mask(black_mask: np.ndarray, wall_mask: np.ndarray, layer_count: int) -> np.ndarray:
    h, w = black_mask.shape
    dist = np.zeros((h, w), dtype=int)
    q: deque[tuple[int, int]] = deque()
    for r, c in np.argwhere(black_mask):
        r = int(r)
        c = int(c)
        if any(0 <= nr < h and 0 <= nc < w and wall_mask[nr, nc] for nr, nc in _neighbors4(r, c)):
            dist[r, c] = 1
            q.append((r, c))
    while q:
        r, c = q.popleft()
        if dist[r, c] >= layer_count:
            continue
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and black_mask[nr, nc] and dist[nr, nc] == 0:
                dist[nr, nc] = dist[r, c] + 1
                q.append((nr, nc))
    return (dist > 0) & (dist <= layer_count)


def _make_room(seed_count: int, seed_color: int, max_h: int, max_w: int) -> dict:
    min_h = max(7, seed_count + 5)
    min_w = max(7, seed_count + 5)
    if max_h < min_h or max_w < min_w:
        raise RuntimeError("room bounds too small")
    for _ in range(ROOM_RETRIES):
        room_h = random.randint(min_h, max_h)
        room_w = random.randint(min_w, max_w)
        core_mask, seed_core = _make_core_shape(room_h - 2, room_w - 2, seed_count)

        interior_mask = np.zeros((room_h, room_w), dtype=bool)
        interior_mask[1:-1, 1:-1] = core_mask
        wall_mask = np.zeros((room_h, room_w), dtype=bool)
        for r, c in np.argwhere(interior_mask):
            r = int(r)
            c = int(c)
            for nr, nc in _neighbors4(r, c):
                if 0 <= nr < room_h and 0 <= nc < room_w and not interior_mask[nr, nc]:
                    wall_mask[nr, nc] = True

        seed_positions = [(int(r + 1), int(c + 1)) for r, c in seed_core]
        if any(wall_mask[r, c] for r, c in seed_positions):
            continue

        local_grid = np.zeros((room_h, room_w), dtype=int)
        local_grid[wall_mask] = WALL
        for r, c in seed_positions:
            local_grid[r, c] = seed_color

        black_mask = interior_mask.copy()
        for r, c in seed_positions:
            black_mask[r, c] = False
        fill_mask = _local_fill_mask(black_mask, wall_mask, seed_count)

        assert not np.any(fill_mask & wall_mask)
        for r, c in seed_positions:
            assert not fill_mask[r, c]

        return {
            "grid": local_grid,
            "fill_mask": fill_mask,
            "color": seed_color,
            "h": room_h,
            "w": room_w,
        }
    raise RuntimeError("failed to build a valid room")


def _room_cap(height: int, width: int) -> int:
    cap = 1
    if min(height, width) >= 18:
        cap = 2
    if min(height, width) >= 22 and height * width >= 360:
        cap = 3
    if min(height, width) >= 26 and height * width >= 560:
        cap = 4
    return cap


def _non_overlapping(box: tuple[int, int, int, int], others: list[tuple[int, int, int, int]]) -> bool:
    r0, c0, r1, c1 = box
    for or0, oc0, or1, oc1 in others:
        if not (r1 + 1 < or0 or or1 + 1 < r0 or c1 + 1 < oc0 or oc1 + 1 < c0):
            return False
    return True


def _attempt_generate() -> np.ndarray | None:
    height = random.randint(14, 30)
    width = random.randint(14, 30)
    room_count = random.randint(1, _room_cap(height, width))
    room_colors = random.sample(SEED_PALETTE, room_count)
    room_specs = [random.randint(1, 4) for _ in range(room_count)]
    plans: list[dict] = []
    boxes: list[tuple[int, int, int, int]] = []

    max_room_h = min(13, height - 3)
    max_room_w = min(13, width - 3)
    if max_room_h < 7 or max_room_w < 7:
        return None

    for seed_count, color in zip(room_specs, room_colors, strict=False):
        try:
            room = _make_room(seed_count, color, max_room_h, max_room_w)
        except RuntimeError:
            return None
        placed = False
        for _ in range(PLACEMENT_RETRIES):
            r0 = random.randint(1, height - room["h"] - 1)
            c0 = random.randint(1, width - room["w"] - 1)
            box = (r0, c0, r0 + room["h"] - 1, c0 + room["w"] - 1)
            if _non_overlapping(box, boxes):
                room["r0"] = r0
                room["c0"] = c0
                plans.append(room)
                boxes.append(box)
                placed = True
                break
        if not placed:
            return None

    grid = np.zeros((height, width), dtype=int)
    for room in plans:
        r0 = int(room["r0"])
        c0 = int(room["c0"])
        patch = room["grid"]
        mask = patch != BG
        grid[r0 : r0 + room["h"], c0 : c0 + room["w"]][mask] = patch[mask]

    expected = grid.copy()
    for room in plans:
        r0 = int(room["r0"])
        c0 = int(room["c0"])
        expected[r0 : r0 + room["h"], c0 : c0 + room["w"]][room["fill_mask"]] = int(room["color"])

    solved = main(grid)
    if not np.array_equal(solved, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
