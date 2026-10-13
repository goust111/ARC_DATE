import random
from collections import deque

import numpy as np


GREY = 5
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
MAX_RETRIES = 400

SEED_TEMPLATES = [
    ((0, 0),),
    ((0, 0), (0, 1)),
    ((0, 0), (1, 0)),
    ((0, 0), (0, 1), (1, 0)),
    ((0, 0), (1, 0), (1, 1)),
    ((0, 1), (1, 0), (1, 1)),
    ((0, 0), (0, 1), (0, 2)),
    ((0, 1), (1, 0), (1, 1), (1, 2)),
    ((0, 0), (1, 0), (1, 1), (2, 1)),
]

KEY_SIZES = [(3, 3), (3, 4), (4, 4), (4, 5), (5, 5)]


def _mode(arr: np.ndarray) -> int:
    vals, counts = np.unique(arr, return_counts=True)
    return int(vals[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r, c in np.argwhere(mask):
        r = int(r)
        c = int(c)
        if seen[r, c]:
            continue
        q = deque([(r, c)])
        seen[r, c] = True
        pts: list[tuple[int, int]] = []
        while q:
            cr, cc = q.popleft()
            pts.append((cr, cc))
            for dr, dc in DIR4:
                nr = cr + dr
                nc = cc + dc
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    q.append((nr, nc))
        comps.append(pts)
    return comps


def _normalize(coords: list[tuple[int, int]]) -> tuple[list[tuple[int, int]], int, int]:
    min_r = min(r for r, _ in coords)
    min_c = min(c for _, c in coords)
    norm = sorted((r - min_r, c - min_c) for r, c in coords)
    h = max(r for r, _ in norm) + 1
    w = max(c for _, c in norm) + 1
    return norm, h, w


def _rotate_once(
    coords: list[tuple[int, int]], h: int, w: int
) -> tuple[list[tuple[int, int]], int, int]:
    rot = [(c, h - 1 - r) for r, c in coords]
    return _normalize(rot)


def _random_seed_shape(rng: random.Random) -> tuple[list[tuple[int, int]], int, int]:
    coords = [tuple(cell) for cell in rng.choice(SEED_TEMPLATES)]
    coords, h, w = _normalize(coords)
    if rng.random() < 0.5:
        coords, h, w = _normalize([(r, w - 1 - c) for r, c in coords])
    for _ in range(rng.randrange(4)):
        coords, h, w = _rotate_once(coords, h, w)
    return coords, h, w


def _perimeter_coords(r0: int, c0: int, r1: int, c1: int) -> list[tuple[int, int]]:
    pts: list[tuple[int, int]] = []
    for c in range(c0, c1 + 1):
        pts.append((r0, c))
    for r in range(r0 + 1, r1):
        pts.append((r, c1))
    for c in range(c1, c0 - 1, -1):
        pts.append((r1, c))
    for r in range(r1 - 1, r0, -1):
        pts.append((r, c0))
    return pts


def _find_locks(grid: np.ndarray, bg: int) -> list[dict]:
    locks: list[dict] = []
    for pts in _components(grid == GREY):
        rows = [r for r, _ in pts]
        cols = [c for _, c in pts]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        if r1 - r0 < 2 or c1 - c0 < 2:
            continue
        perimeter = _perimeter_coords(r0, c0, r1, c1)
        grey_cells = set(pts)
        if len(grey_cells) != len(perimeter) - 1:
            continue
        perimeter_set = set(perimeter)
        if not grey_cells.issubset(perimeter_set):
            continue
        marker_cells = list(perimeter_set - grey_cells)
        if len(marker_cells) != 1:
            continue
        mr, mc = marker_cells[0]
        marker_color = int(grid[mr, mc])
        if marker_color in (bg, GREY):
            continue
        inner = grid[r0 + 1 : r1, c0 + 1 : c1]
        seed_mask = inner != bg
        if not seed_mask.any():
            continue
        seed_colors = {int(v) for v in inner[seed_mask]}
        if len(seed_colors) != 1:
            continue
        seed_color = next(iter(seed_colors))
        if seed_color == GREY:
            continue
        if len(_components(seed_mask)) != 1:
            continue
        seed_coords = [
            (r0 + 1 + int(r), c0 + 1 + int(c)) for r, c in np.argwhere(seed_mask)
        ]
        locks.append(
            {
                "bbox": (r0, c0, r1, c1),
                "perimeter": perimeter,
                "marker_color": marker_color,
                "seed_color": seed_color,
                "seed_coords": seed_coords,
            }
        )
    locks.sort(key=lambda item: item["bbox"])
    return locks


def _find_keys(grid: np.ndarray, bg: int, blocked: np.ndarray) -> dict[int, dict]:
    key_mask = (grid != bg) & (~blocked)
    found: dict[int, dict] = {}
    bad: set[int] = set()
    h, w = grid.shape
    for pts in _components(key_mask):
        colors = {int(grid[r, c]) for r, c in pts}
        if len(colors) != 2:
            continue
        pset = set(pts)
        boundary_colors = {
            int(grid[r, c])
            for r, c in pts
            if any(
                not (0 <= r + dr < h and 0 <= c + dc < w) or (r + dr, c + dc) not in pset
                for dr, dc in DIR4
            )
        }
        if len(boundary_colors) != 1:
            continue
        outline = next(iter(boundary_colors))
        fill_colors = colors - {outline}
        if len(fill_colors) != 1:
            continue
        fill = next(iter(fill_colors))
        if outline in found:
            bad.add(outline)
        else:
            found[outline] = {"outline": outline, "fill": fill, "coords": pts}
    for outline in bad:
        found.pop(outline, None)
    return found


def _flood_fill_dist(
    base: np.ndarray, bg: int, seed_coords: list[tuple[int, int]]
) -> np.ndarray:
    h, w = base.shape
    dist = np.full((h, w), -1, dtype=int)
    q = deque()
    for r, c in seed_coords:
        dist[r, c] = 0
        q.append((r, c))
    while q:
        r, c = q.popleft()
        for dr, dc in DIR4:
            nr = r + dr
            nc = c + dc
            if 0 <= nr < h and 0 <= nc < w and dist[nr, nc] == -1 and int(base[nr, nc]) == bg:
                dist[nr, nc] = dist[r, c] + 1
                q.append((nr, nc))
    return dist


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = _mode(g)

    locks = _find_locks(g, bg)
    if not locks:
        return g.copy()

    blocked = np.zeros_like(g, dtype=bool)
    for lock in locks:
        r0, c0, r1, c1 = lock["bbox"]
        blocked[r0 : r1 + 1, c0 : c1 + 1] = True
    keys = _find_keys(g, bg, blocked)

    unlocked = [
        lock
        for lock in locks
        if lock["marker_color"] in keys and keys[lock["marker_color"]]["fill"] == lock["seed_color"]
    ]

    base = g.copy()
    for lock in unlocked:
        for r, c in lock["perimeter"]:
            base[r, c] = bg

    out = base.copy()
    for lock in unlocked:
        dist = _flood_fill_dist(base, bg, lock["seed_coords"])
        seed_color = lock["seed_color"]
        outline_color = lock["marker_color"]
        for r, c in np.argwhere(dist >= 1):
            d = int(dist[r, c])
            out[r, c] = outline_color if d % 2 == 1 else seed_color
        for r, c in lock["seed_coords"]:
            out[r, c] = seed_color

    return out


def _can_place(occupied: np.ndarray, top: int, left: int, h: int, w: int, pad: int = 1) -> bool:
    hh, ww = occupied.shape
    if top < 1 or left < 1 or top + h > hh - 1 or left + w > ww - 1:
        return False
    r0 = max(0, top - pad)
    r1 = min(hh, top + h + pad)
    c0 = max(0, left - pad)
    c1 = min(ww, left + w + pad)
    return not occupied[r0:r1, c0:c1].any()


def _reserve(occupied: np.ndarray, top: int, left: int, h: int, w: int) -> None:
    occupied[top : top + h, left : left + w] = True


def _pick_spot(
    rng: random.Random, occupied: np.ndarray, h: int, w: int, attempts: int = 100
) -> tuple[int, int] | None:
    hh, ww = occupied.shape
    if h > hh - 2 or w > ww - 2:
        return None
    for _ in range(attempts):
        top = rng.randint(1, hh - h - 1)
        left = rng.randint(1, ww - w - 1)
        if _can_place(occupied, top, left, h, w, pad=1):
            return top, left
    return None


def _draw_lock(
    grid: np.ndarray,
    top: int,
    left: int,
    inner_h: int,
    inner_w: int,
    seed_shape: list[tuple[int, int]],
    seed_h: int,
    seed_w: int,
    seed_color: int,
    marker_color: int,
    rng: random.Random,
) -> None:
    r0, c0 = top, left
    r1, c1 = top + inner_h + 1, left + inner_w + 1
    for r, c in _perimeter_coords(r0, c0, r1, c1):
        grid[r, c] = GREY
    mr, mc = rng.choice(_perimeter_coords(r0, c0, r1, c1))
    grid[mr, mc] = marker_color
    off_r = rng.randint(0, inner_h - seed_h)
    off_c = rng.randint(0, inner_w - seed_w)
    for sr, sc in seed_shape:
        grid[r0 + 1 + off_r + sr, c0 + 1 + off_c + sc] = seed_color


def _random_key_size(rng: random.Random) -> tuple[int, int]:
    h, w = rng.choice(KEY_SIZES)
    if rng.random() < 0.5:
        return w, h
    return h, w


def _draw_key(
    grid: np.ndarray, top: int, left: int, h: int, w: int, outline: int, fill: int
) -> None:
    grid[top : top + h, left : left + w] = outline
    grid[top + 1 : top + h - 1, left + 1 : left + w - 1] = fill


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    n_locks = rng.choice([2, 2, 3])
    if n_locks == 2:
        h = rng.randint(18, 26)
        w = rng.randint(20, 30)
    else:
        h = rng.randint(22, 30)
        w = rng.randint(24, 32)

    bg = rng.choice([c for c in range(10) if c != GREY])
    palette = [c for c in range(10) if c not in (bg, GREY)]
    rng.shuffle(palette)
    seed_colors = palette[:n_locks]
    outline_colors = palette[n_locks : 2 * n_locks]
    spare_colors = palette[2 * n_locks :]
    unlocked_index = rng.randrange(n_locks)

    grid = np.full((h, w), bg, dtype=int)
    occupied = np.zeros((h, w), dtype=bool)

    lock_specs = []
    for i in range(n_locks):
        seed_shape, seed_h, seed_w = _random_seed_shape(rng)
        inner_h = rng.randint(max(3, seed_h + 1), max(5, seed_h + 3))
        inner_w = rng.randint(max(3, seed_w + 1), max(5, seed_w + 3))
        lock_specs.append(
            {
                "index": i,
                "seed_shape": seed_shape,
                "seed_h": seed_h,
                "seed_w": seed_w,
                "inner_h": inner_h,
                "inner_w": inner_w,
                "area": (inner_h + 2) * (inner_w + 2),
            }
        )

    for spec in sorted(lock_specs, key=lambda item: -item["area"]):
        full_h = spec["inner_h"] + 2
        full_w = spec["inner_w"] + 2
        pos = _pick_spot(rng, occupied, full_h, full_w)
        if pos is None:
            return None
        top, left = pos
        _reserve(occupied, top, left, full_h, full_w)
        idx = spec["index"]
        _draw_lock(
            grid,
            top,
            left,
            spec["inner_h"],
            spec["inner_w"],
            spec["seed_shape"],
            spec["seed_h"],
            spec["seed_w"],
            seed_colors[idx],
            outline_colors[idx],
            rng,
        )

    for i in range(n_locks):
        key_h, key_w = _random_key_size(rng)
        pos = _pick_spot(rng, occupied, key_h, key_w)
        if pos is None:
            return None
        top, left = pos
        _reserve(occupied, top, left, key_h, key_w)
        if i == unlocked_index:
            fill_color = seed_colors[i]
        else:
            mismatch_choices = [
                c
                for c in list(seed_colors) + spare_colors
                if c not in (seed_colors[i], outline_colors[i])
            ]
            if not mismatch_choices:
                return None
            fill_color = rng.choice(mismatch_choices)
        _draw_key(grid, top, left, key_h, key_w, outline_colors[i], fill_color)

    if _mode(grid) != bg:
        return None

    try:
        locks = _find_locks(grid, bg)
        if len(locks) != n_locks:
            return None
        blocked = np.zeros_like(grid, dtype=bool)
        for lock in locks:
            r0, c0, r1, c1 = lock["bbox"]
            blocked[r0 : r1 + 1, c0 : c1 + 1] = True
        keys = _find_keys(grid, bg, blocked)
        if set(keys.keys()) != set(outline_colors):
            return None
        out = main(grid)
    except Exception:
        return None

    if not np.any(out != grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input failed after {MAX_RETRIES} attempts")
