from __future__ import annotations

import random

import numpy as np


BLACK = 0
BLUE = 1
PALETTE = (2, 3, 4, 8)
MAX_ATTEMPTS = 400


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = map(int, shape)
    return 0 <= r < h and 0 <= c < w


def _known_mask(grid: np.ndarray) -> np.ndarray:
    return (grid != BLACK) & (grid != BLUE)


def _infer_tile(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    known = _known_mask(g)
    assert np.any(known), "expected visible wallpaper pixels"

    h, w = g.shape
    candidates: list[tuple[int, int, int, np.ndarray]] = []
    coords = np.argwhere(known)
    for pr in range(1, h + 1):
        for pc in range(1, w + 1):
            tile = np.full((pr, pc), -1, dtype=int)
            conflict = False
            for r0, c0 in coords:
                r, c = int(r0), int(c0)
                tr, tc = r % pr, c % pc
                color = int(g[r, c])
                if tile[tr, tc] == -1:
                    tile[tr, tc] = color
                elif int(tile[tr, tc]) != color:
                    conflict = True
                    break
            if not conflict and np.all(tile != -1):
                candidates.append((pr * pc, pr, pc, tile))

    assert candidates, "expected a fully observed periodic tile"
    best_area = min(area for area, _pr, _pc, _tile in candidates)
    best = [item for item in candidates if item[0] == best_area]
    assert len(best) == 1, "expected a unique smallest periodic tile"
    _area, pr, pc, tile = best[0]
    assert pr < h or pc < w, "expected the tile to repeat"
    return tile


def _seed_cells(grid: np.ndarray) -> set[tuple[int, int]]:
    coords = [tuple(map(int, rc)) for rc in np.argwhere(np.asarray(grid) == BLUE)]
    assert 2 <= len(coords) <= 3, "expected a 2-3 pixel blue seed"
    seed = set(coords)

    same_main_diag = len({r - c for r, c in seed}) == 1
    same_anti_diag = len({r + c for r, c in seed}) == 1
    assert same_main_diag or same_anti_diag, "seed must lie on one diagonal"

    if same_main_diag:
        dirs = ((1, 1), (-1, -1))
    else:
        dirs = ((1, -1), (-1, 1))

    for dr, dc in dirs:
        found = 0
        for r, c in seed:
            seq = [(r + k * dr, c + k * dc) for k in range(len(seed))]
            if set(seq) == seed:
                found += 1
        assert found == 1, "seed must be a contiguous diagonal segment"

    return seed


def _ray_path(grid: np.ndarray) -> list[tuple[int, int]]:
    g = np.asarray(grid, dtype=int)
    seed = _seed_cells(g)
    h, w = g.shape

    if len({r - c for r, c in seed}) == 1:
        dirs = ((1, 1), (-1, -1))
    else:
        dirs = ((1, -1), (-1, 1))

    candidates: list[tuple[int, int, int, int]] = []
    n = len(seed)
    for dr, dc in dirs:
        for r, c in seed:
            seq = [(r + k * dr, c + k * dc) for k in range(n)]
            if set(seq) != seed:
                continue
            prev_r, prev_c = r - dr, c - dc
            next_r, next_c = r + n * dr, c + n * dc
            if not _in_bounds((h, w), prev_r, prev_c) and _in_bounds((h, w), next_r, next_c):
                candidates.append((r, c, dr, dc))

    assert len(candidates) == 1, "seed direction must be uniquely edge-anchored"
    r, c, dr, dc = candidates[0]
    path: list[tuple[int, int]] = []
    while _in_bounds((h, w), r, c):
        path.append((r, c))
        r += dr
        c += dc
    return path


def main(grid: np.ndarray) -> np.ndarray:
    """
    Restore only the ray-traced damage in a periodic wallpaper.

    Input structure
    - The grid is a repeated square wallpaper tile with black holes marking corrupted pixels.
    - A short blue diagonal seed of length 2-3 lies near one border and overwrites the wallpaper below it.
    - Blue does not belong to the wallpaper; all other non-black non-blue pixels are valid wallpaper samples.

    Rule
    - Read the blue segment as the start of a straight diagonal ray and extend it in the same diagonal direction until it leaves the grid.
    - Infer the unique smallest repeating tile from all non-black non-blue pixels.
    - Repaint every cell on the ray path with its wallpaper phase color, while black holes outside the path stay black.

    Key insight
    - The blue seed determines where repair is allowed, and the wallpaper period determines what color each repaired path cell must become.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    tile = _infer_tile(g)
    path = _ray_path(g)
    out = g.copy()
    for r, c in path:
        color = int(tile[r % tile.shape[0], c % tile.shape[1]])
        if int(g[r, c]) not in (BLACK, BLUE):
            assert int(g[r, c]) == color, "visible wallpaper must match the inferred tile"
        out[r, c] = color
    return out


def _full_wallpaper(tile: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    h, w = map(int, shape)
    pr, pc = tile.shape
    out = np.empty((h, w), dtype=int)
    for r in range(h):
        for c in range(w):
            out[r, c] = int(tile[r % pr, c % pc])
    return out


def _ray_start_candidates(shape: tuple[int, int], dr: int, dc: int, min_len: int) -> list[tuple[int, int]]:
    h, w = map(int, shape)
    out: list[tuple[int, int]] = []
    for r in range(h):
        for c in range(w):
            if _in_bounds(shape, r - dr, c - dc):
                continue
            rr, cc = r, c
            length = 0
            while _in_bounds(shape, rr, cc):
                length += 1
                rr += dr
                cc += dc
            if length >= min_len:
                out.append((r, c))
    return out


def _random_tile(period: int, shape: tuple[int, int]) -> np.ndarray | None:
    color_count = random.randint(3, 4)
    colors = random.sample(PALETTE, color_count)
    for _ in range(120):
        tile = np.array(
            [[random.choice(colors) for _ in range(period)] for _ in range(period)],
            dtype=int,
        )
        if len(set(map(int, tile.flatten()))) < color_count:
            continue
        if any(len(set(map(int, tile[r, :]))) < 2 for r in range(period)):
            continue
        if any(len(set(map(int, tile[:, c]))) < 2 for c in range(period)):
            continue
        wallpaper = _full_wallpaper(tile, shape)
        inferred = _infer_tile(wallpaper)
        if inferred.shape == (period, period) and np.array_equal(inferred, tile):
            return tile
    return None


def _path_from_start(shape: tuple[int, int], start: tuple[int, int], dr: int, dc: int) -> list[tuple[int, int]]:
    r, c = start
    path: list[tuple[int, int]] = []
    while _in_bounds(shape, r, c):
        path.append((r, c))
        r += dr
        c += dc
    return path


def _choose_holes(
    shape: tuple[int, int],
    period: int,
    path: list[tuple[int, int]],
    seed_len: int,
) -> set[tuple[int, int]] | None:
    h, w = map(int, shape)
    total = h * w
    min_holes = max(1, int(np.ceil(total * 0.20)))
    max_holes = max(min_holes, int(np.floor(total * 0.30)))
    path_set = set(path)
    seed_set = set(path[:seed_len])
    all_cells = [(r, c) for r in range(h) for c in range(w)]
    off_path = [rc for rc in all_cells if rc not in path_set]

    for _ in range(120):
        keep_visible: set[tuple[int, int]] = set()
        for rr in range(period):
            for cc in range(period):
                candidates = [
                    rc
                    for rc in off_path
                    if rc[0] % period == rr and rc[1] % period == cc
                ]
                if not candidates:
                    keep_visible = set()
                    break
                keep_visible.add(random.choice(candidates))
            if not keep_visible:
                break
        if not keep_visible:
            continue

        path_known_pool = [rc for rc in path[seed_len:] if rc not in keep_visible]
        if len(path_known_pool) < 1:
            continue
        keep_visible.add(random.choice(path_known_pool))

        target = random.randint(min_holes, max_holes)
        path_hole_pool = [rc for rc in path[seed_len:] if rc not in keep_visible]
        off_hole_pool = [rc for rc in off_path if rc not in keep_visible]
        if len(path_hole_pool) < 1 or len(off_hole_pool) < 1:
            continue

        holes: set[tuple[int, int]] = {
            random.choice(path_hole_pool),
            random.choice(off_hole_pool),
        }
        pool = [
            rc
            for rc in all_cells
            if rc not in holes and rc not in keep_visible and rc not in seed_set
        ]
        if len(holes) > target or len(pool) < target - len(holes):
            continue
        holes.update(random.sample(pool, target - len(holes)))

        if any(rc in seed_set for rc in holes):
            continue
        if not any(rc in path_set for rc in holes):
            continue
        if not any(rc not in path_set for rc in holes):
            continue
        return holes
    return None


def _make_example() -> tuple[np.ndarray, np.ndarray] | None:
    h = random.randint(13, 18)
    w = random.randint(13, 18)
    period = random.choice((3, 4))
    tile = _random_tile(period, (h, w))
    if tile is None:
        return None
    wallpaper = _full_wallpaper(tile, (h, w))

    directions = [(1, 1), (1, -1), (-1, 1), (-1, -1)]
    random.shuffle(directions)
    for dr, dc in directions:
        starts = _ray_start_candidates((h, w), dr, dc, min_len=5)
        if not starts:
            continue
        start = random.choice(starts)
        path = _path_from_start((h, w), start, dr, dc)
        seed_len = random.randint(2, min(3, len(path) - 2))
        holes = _choose_holes((h, w), period, path, seed_len)
        if holes is None:
            continue

        inp = wallpaper.copy()
        for r, c in holes:
            inp[r, c] = BLACK
        for r, c in path[:seed_len]:
            inp[r, c] = BLUE

        out = inp.copy()
        for r, c in path:
            out[r, c] = int(wallpaper[r, c])

        if not np.array_equal(main(inp), out):
            continue
        inferred = _infer_tile(inp)
        if inferred.shape != (period, period) or not np.array_equal(inferred, tile):
            continue

        path_set = set(path)
        if not np.any((inp == BLUE) & (out != BLUE)):
            continue
        if not any(inp[r, c] == BLACK and out[r, c] != BLACK for r, c in path):
            continue
        if not any(inp[r, c] == BLACK and out[r, c] == BLACK for r in range(h) for c in range(w) if (r, c) not in path_set):
            continue
        if not any(inp[r, c] not in (BLACK, BLUE) for r, c in path[seed_len:]):
            continue

        return inp, out
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        example = _make_example()
        if example is not None:
            return example[0]
    raise RuntimeError("failed to generate a valid ray-wallpaper repair input")
