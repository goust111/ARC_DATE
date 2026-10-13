from __future__ import annotations

import random

import numpy as np


BG = 0
MARKER = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
DIRS = ("up", "down", "left", "right")


def _side_cells(r: int, c: int, direction: str) -> list[tuple[int, int]]:
    if direction == "up":
        return [(r - 1, c), (r - 1, c + 1)]
    if direction == "down":
        return [(r + 2, c), (r + 2, c + 1)]
    if direction == "left":
        return [(r, c - 1), (r + 1, c - 1)]
    if direction == "right":
        return [(r, c + 2), (r + 1, c + 2)]
    raise ValueError(direction)


def _in_bounds(h: int, w: int, r: int, c: int) -> bool:
    return 0 <= r < h and 0 <= c < w


def _is_exact_tower(grid: np.ndarray, r: int, c: int) -> bool:
    h, w = grid.shape
    if r + 1 >= h or c + 1 >= w:
        return False
    color = int(grid[r, c])
    if color in (BG, MARKER):
        return False
    patch = grid[r : r + 2, c : c + 2]
    if not bool((patch == color).all()):
        return False
    for rr in range(r - 1, r + 3):
        for cc in range(c - 1, c + 3):
            if not _in_bounds(h, w, rr, cc):
                continue
            if r <= rr < r + 2 and c <= cc < c + 2:
                continue
            if int(grid[rr, cc]) == color:
                return False
    return True


def _read_dirs(grid: np.ndarray, r: int, c: int) -> tuple[str, ...]:
    h, w = grid.shape
    active: list[str] = []
    for direction in DIRS:
        cells = _side_cells(r, c, direction)
        if any(_in_bounds(h, w, rr, cc) and int(grid[rr, cc]) == MARKER for rr, cc in cells):
            active.append(direction)
    return tuple(active)


def _detect_towers(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    used = np.zeros_like(g, dtype=bool)
    towers: list[dict] = []

    for r in range(h - 1):
        for c in range(w - 1):
            if not _is_exact_tower(g, r, c):
                continue
            if bool(used[r : r + 2, c : c + 2].any()):
                continue
            towers.append(
                {
                    "r": int(r),
                    "c": int(c),
                    "color": int(g[r, c]),
                    "dirs": _read_dirs(g, r, c),
                }
            )
            used[r : r + 2, c : c + 2] = True

    assert towers, "expected at least one 2x2 tower"

    colored = (g != BG) & (g != MARKER)
    assert int(colored.sum()) == 4 * len(towers), "non-marker foreground must be only tower bodies"

    for rr, cc in np.argwhere(g == MARKER):
        hits = 0
        for tower in towers:
            for direction in DIRS:
                if (int(rr), int(cc)) in _side_cells(int(tower["r"]), int(tower["c"]), direction):
                    hits += 1
        assert hits == 1, "each grey marker must indicate exactly one tower side"

    return towers


def _tower_mask(shape: tuple[int, int], towers: list[dict]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    for tower in towers:
        r = int(tower["r"])
        c = int(tower["c"])
        mask[r : r + 2, c : c + 2] = True
    return mask


def _cast_ray(
    h: int,
    w: int,
    tower: dict,
    direction: str,
    mask: np.ndarray,
) -> tuple[list[tuple[int, int]], bool]:
    r = int(tower["r"])
    c = int(tower["c"])
    cells: list[tuple[int, int]] = []

    if direction == "up":
        for rr in range(r - 1, -1, -1):
            lane = [(rr, c), (rr, c + 1)]
            if any(bool(mask[lr, lc]) for lr, lc in lane):
                return cells, True
            cells.extend(lane)
        return cells, False

    if direction == "down":
        for rr in range(r + 2, h):
            lane = [(rr, c), (rr, c + 1)]
            if any(bool(mask[lr, lc]) for lr, lc in lane):
                return cells, True
            cells.extend(lane)
        return cells, False

    if direction == "left":
        for cc in range(c - 1, -1, -1):
            lane = [(r, cc), (r + 1, cc)]
            if any(bool(mask[lr, lc]) for lr, lc in lane):
                return cells, True
            cells.extend(lane)
        return cells, False

    if direction == "right":
        for cc in range(c + 2, w):
            lane = [(r, cc), (r + 1, cc)]
            if any(bool(mask[lr, lc]) for lr, lc in lane):
                return cells, True
            cells.extend(lane)
        return cells, False

    raise ValueError(direction)


def _render_output_from_towers(grid: np.ndarray, towers: list[dict]) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    out = g.copy()
    out[out == MARKER] = BG
    mask = _tower_mask((h, w), towers)

    for tower in sorted(towers, key=lambda t: (int(t["r"]), int(t["c"]))):
        for direction in tower["dirs"]:
            ray_cells, _blocked = _cast_ray(h, w, tower, direction, mask)
            for rr, cc in ray_cells:
                out[rr, cc] = int(tower["color"])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Emit 2-pixel-wide rays from marked 2x2 towers and resolve overlaps by tower scan order.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    towers = _detect_towers(g)
    return _render_output_from_towers(g, towers)


def _fits_tower(r: int, c: int, h: int, w: int, towers: list[dict]) -> bool:
    if not (1 <= r <= h - 3 and 1 <= c <= w - 3):
        return False
    for tower in towers:
        tr = int(tower["r"])
        tc = int(tower["c"])
        if not (r + 2 < tr - 1 or tr + 2 < r - 1 or c + 2 < tc - 1 or tc + 2 < c - 1):
            return False
    return True


def _corridor_clear_for_extra(r: int, c: int, base_a: dict, base_b: dict) -> bool:
    ar = int(base_a["r"])
    ac = int(base_a["c"])
    br = int(base_b["r"])
    bc = int(base_b["c"])
    min_c = min(ac, bc)
    max_c = max(ac, bc)
    min_r = min(ar, br)
    max_r = max(ar, br)

    if r == ar and max(c, min_c) <= min(c + 1, max_c + 1):
        return False
    if c == bc and max(r, min_r) <= min(r + 1, max_r + 1):
        return False
    return True


def _build_grid(shape: tuple[int, int], towers: list[dict]) -> np.ndarray | None:
    h, w = shape
    grid = np.full((h, w), BG, dtype=int)
    occupied: set[tuple[int, int]] = set()

    for tower in towers:
        r = int(tower["r"])
        c = int(tower["c"])
        color = int(tower["color"])
        for rr in range(r, r + 2):
            for cc in range(c, c + 2):
                if (rr, cc) in occupied:
                    return None
                grid[rr, cc] = color
                occupied.add((rr, cc))

    for tower in towers:
        tower["marker_cells"] = {}
        r = int(tower["r"])
        c = int(tower["c"])
        for direction in tower["dirs"]:
            choices = [cell for cell in _side_cells(r, c, direction) if _in_bounds(h, w, cell[0], cell[1])]
            random.shuffle(choices)
            placed = None
            for cell in choices:
                if cell not in occupied:
                    placed = cell
                    break
            if placed is None:
                return None
            tower["marker_cells"][direction] = placed
            grid[placed] = MARKER
            occupied.add(placed)

    return grid


def _ray_sets(towers: list[dict], shape: tuple[int, int]) -> tuple[list[set[tuple[int, int]]], bool]:
    h, w = shape
    mask = _tower_mask(shape, towers)
    sets: list[set[tuple[int, int]]] = []
    blocked_any = False
    for tower in towers:
        acc: set[tuple[int, int]] = set()
        for direction in tower["dirs"]:
            cells, blocked = _cast_ray(h, w, tower, direction, mask)
            acc.update(cells)
            blocked_any = blocked_any or blocked
        sets.append(acc)
    return sets, blocked_any


def _has_intersection(towers: list[dict], shape: tuple[int, int]) -> bool:
    ray_sets, _blocked = _ray_sets(towers, shape)
    for i in range(len(towers)):
        for j in range(i + 1, len(towers)):
            if int(towers[i]["color"]) == int(towers[j]["color"]):
                continue
            if ray_sets[i] & ray_sets[j]:
                return True
    return False


def _augment_dirs(tower: dict, preferred_forbidden: set[str] | None = None) -> None:
    forbidden = preferred_forbidden or set()
    current = set(tower["dirs"])
    extras = [d for d in DIRS if d not in current and d not in forbidden]
    random.shuffle(extras)
    for direction in extras:
        if random.random() < 0.3:
            current.add(direction)
    if not current and extras:
        current.add(random.choice(extras))
    tower["dirs"] = tuple(d for d in DIRS if d in current)


def _make_base_pair(h: int, w: int) -> tuple[dict, dict] | None:
    for _ in range(200):
        ar = random.randint(1, h - 3)
        br = random.randint(1, h - 3)
        ac = random.randint(1, w - 3)
        bc = random.randint(1, w - 3)
        if abs(ar - br) < 4 or abs(ac - bc) < 4:
            continue
        horiz = "right" if ac < bc else "left"
        vert = "down" if br < ar else "up"
        a = {"r": ar, "c": ac, "dirs": (horiz,)}
        b = {"r": br, "c": bc, "dirs": (vert,)}
        return a, b
    return None


def _add_blocker(towers: list[dict], h: int, w: int) -> bool:
    a, b = towers[0], towers[1]
    options: list[tuple[int, int, set[str]]] = []

    if "right" in a["dirs"]:
        for c in range(int(b["c"]) + 4, w - 2):
            if _fits_tower(int(a["r"]), c, h, w, towers):
                options.append((int(a["r"]), c, {"left"}))
    if "left" in a["dirs"]:
        for c in range(1, int(b["c"]) - 3):
            if _fits_tower(int(a["r"]), c, h, w, towers):
                options.append((int(a["r"]), c, {"right"}))
    if "down" in b["dirs"]:
        for r in range(int(a["r"]) + 4, h - 2):
            if _fits_tower(r, int(b["c"]), h, w, towers):
                options.append((r, int(b["c"]), {"up"}))
    if "up" in b["dirs"]:
        for r in range(1, int(a["r"]) - 3):
            if _fits_tower(r, int(b["c"]), h, w, towers):
                options.append((r, int(b["c"]), {"down"}))

    if not options:
        return False

    r, c, forbidden = random.choice(options)
    tower = {"r": int(r), "c": int(c), "dirs": tuple()}
    if random.random() < 0.7:
        _augment_dirs(tower, preferred_forbidden=forbidden)
    towers.append(tower)
    return True


def _add_random_extras(towers: list[dict], h: int, w: int, count: int) -> None:
    base_a, base_b = towers[0], towers[1]
    for _ in range(count):
        for _attempt in range(120):
            r = random.randint(1, h - 3)
            c = random.randint(1, w - 3)
            if not _fits_tower(r, c, h, w, towers):
                continue
            if not _corridor_clear_for_extra(r, c, base_a, base_b):
                continue
            dirs = list(DIRS)
            random.shuffle(dirs)
            k = random.choices([1, 2, 3], weights=[4, 3, 1])[0]
            towers.append({"r": int(r), "c": int(c), "dirs": tuple(sorted(dirs[:k], key=DIRS.index))})
            break


def _generate_candidate() -> np.ndarray | None:
    h = random.randint(14, 23)
    w = random.randint(14, 23)
    pair = _make_base_pair(h, w)
    if pair is None:
        return None

    towers = [pair[0], pair[1]]
    want_blocker = random.random() < 0.55
    extra_count = random.randint(0, 2)

    if want_blocker and not _add_blocker(towers, h, w):
        return None

    _add_random_extras(towers, h, w, extra_count)

    if len(towers) < 3:
        return None
    if len(towers) > len(PALETTE):
        return None

    colors = random.sample(PALETTE, len(towers))
    for tower, color in zip(towers, colors):
        tower["color"] = int(color)

    _augment_dirs(towers[0])
    _augment_dirs(towers[1])

    grid = _build_grid((h, w), towers)
    if grid is None:
        return None

    if not _has_intersection(towers, (h, w)):
        return None

    ray_sets, blocked_any = _ray_sets(towers, (h, w))
    if want_blocker and not blocked_any:
        return None

    active_dirs = sum(len(tower["dirs"]) for tower in towers)
    if active_dirs < 3:
        return None

    out = main(grid)
    if np.array_equal(grid, out):
        return None
    if float(np.count_nonzero(out != BG)) / float(out.size) > 0.55:
        return None

    marker_count = int(np.count_nonzero(grid == MARKER))
    if marker_count != active_dirs:
        return None

    overlap_cells = 0
    for i in range(len(ray_sets)):
        for j in range(i + 1, len(ray_sets)):
            overlap_cells += len(ray_sets[i] & ray_sets[j])
    if overlap_cells == 0:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(500):
        grid = _generate_candidate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid signal-tower grid")
