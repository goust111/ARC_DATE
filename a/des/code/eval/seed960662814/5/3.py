from __future__ import annotations

import numpy as np


GREY = 5
YELLOW = 4
DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
MAX_GENERATE_ATTEMPTS = 400


def _get_bg(grid: np.ndarray) -> int:
    vals, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(vals[int(np.argmax(counts))])


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _find_mirrors(grid: np.ndarray) -> tuple[list[dict], np.ndarray]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    mirror_id = np.full((h, w), -1, dtype=int)
    mirrors: list[dict] = []

    for r in range(h):
        for c in range(w):
            if int(g[r, c]) != GREY or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((int(cr), int(cc)))
                for dr, dc in DIRS:
                    nr, nc = int(cr + dr), int(cc + dc)
                    if _in_bounds(g.shape, nr, nc) and not seen[nr, nc] and int(g[nr, nc]) == GREY:
                        seen[nr, nc] = True
                        stack.append((nr, nc))

            rows = {rr for rr, _cc in coords}
            cols = {cc for _rr, cc in coords}
            if len(rows) == 1:
                orientation = "horizontal"
                endpoints = [min(coords, key=lambda p: p[1]), max(coords, key=lambda p: p[1])]
            elif len(cols) == 1:
                orientation = "vertical"
                endpoints = [min(coords, key=lambda p: p[0]), max(coords, key=lambda p: p[0])]
            else:
                continue

            idx = len(mirrors)
            for rr, cc in coords:
                mirror_id[rr, cc] = idx
            mirrors.append(
                {
                    "coords": coords,
                    "coord_set": set(coords),
                    "orientation": orientation,
                    "endpoints": [(int(endpoints[0][0]), int(endpoints[0][1])), (int(endpoints[1][0]), int(endpoints[1][1]))],
                }
            )
    return mirrors, mirror_id


def _find_towers(grid: np.ndarray, bg: int) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    towers = []
    for r, c in np.argwhere((g != int(bg)) & (g != GREY)):
        towers.append({"pos": (int(r), int(c)), "color": int(g[int(r), int(c)])})
    towers.sort(key=lambda t: (t["pos"][0], t["pos"][1], t["color"]))
    return towers


def _trace_until_mirror(grid: np.ndarray, start: tuple[int, int], step: tuple[int, int]) -> tuple[list[tuple[int, int]], tuple[int, int] | None]:
    g = np.asarray(grid, dtype=int)
    dr, dc = step
    r, c = start
    path: list[tuple[int, int]] = []

    while True:
        r += dr
        c += dc
        if not _in_bounds(g.shape, r, c):
            return path, None
        if int(g[r, c]) == GREY:
            return path, (int(r), int(c))
        path.append((int(r), int(c)))


def _nearest_other_tower_color(
    towers: list[dict],
    tower_pos: tuple[int, int],
    anchor: tuple[int, int],
) -> int:
    ar, ac = anchor
    best: tuple[int, int, int] | None = None
    best_color = YELLOW
    for tower in towers:
        pos = tower["pos"]
        if pos == tower_pos:
            continue
        tr, tc = pos
        if tr != ar and tc != ac:
            continue
        dist = abs(tr - ar) + abs(tc - ac)
        cand = (int(dist), int(tr), int(tc))
        if best is None or cand < best:
            best = cand
            best_color = int(tower["color"])
    return int(best_color)


def _reflection_from_hit(
    hit: tuple[int, int],
    step: tuple[int, int],
    mirror: dict,
) -> tuple[tuple[int, int], tuple[int, int]] | None:
    hr, hc = hit
    dr, dc = step
    orientation = mirror["orientation"]
    if dr != 0 and orientation != "horizontal":
        return None
    if dc != 0 and orientation != "vertical":
        return None

    endpoints = mirror["endpoints"]
    d0 = abs(endpoints[0][0] - hr) + abs(endpoints[0][1] - hc)
    d1 = abs(endpoints[1][0] - hr) + abs(endpoints[1][1] - hc)
    if d0 == d1:
        return None

    exit_pt = endpoints[0] if d0 > d1 else endpoints[1]
    if orientation == "horizontal":
        out_step = (0, int(np.sign(exit_pt[1] - hc)))
    else:
        out_step = (int(np.sign(exit_pt[0] - hr)), 0)
    if out_step == (0, 0):
        return None
    return (int(exit_pt[0]), int(exit_pt[1])), out_step


def _reflect_ray_cells(shape: tuple[int, int], start: tuple[int, int], step: tuple[int, int]) -> list[tuple[int, int]]:
    dr, dc = step
    r, c = start
    cells: list[tuple[int, int]] = []
    while True:
        r += dr
        c += dc
        if not _in_bounds(shape, r, c):
            return cells
        cells.append((int(r), int(c)))


def _solve(grid: np.ndarray, *, collect: bool = False) -> tuple[np.ndarray, list[tuple[tuple[int, int], int]]]:
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    towers = _find_towers(g, bg)
    mirrors, mirror_id = _find_mirrors(g)

    out = g.copy()
    paint_ops: list[tuple[tuple[int, int], int]] = []
    protected = {(int(r), int(c)) for r, c in np.argwhere((g != int(bg)))}

    for tower in towers:
        pos = tower["pos"]
        color = int(tower["color"])
        for step in DIRS:
            path, hit = _trace_until_mirror(g, pos, step)
            for cell in path:
                if cell not in protected:
                    out[cell] = color
                    if collect:
                        paint_ops.append((cell, color))
            if hit is None:
                continue
            idx = int(mirror_id[int(hit[0]), int(hit[1])])
            if idx < 0:
                continue
            refl = _reflection_from_hit(hit, step, mirrors[idx])
            if refl is None:
                continue
            exit_pt, out_step = refl
            refl_color = _nearest_other_tower_color(towers, pos, exit_pt)
            for cell in _reflect_ray_cells(g.shape, exit_pt, out_step):
                if cell not in protected:
                    out[cell] = refl_color
                    if collect:
                        paint_ops.append((cell, refl_color))

    return out, paint_ops


def main(grid: np.ndarray) -> np.ndarray:
    """
    Towers emit orthogonal rays. A ray stops before the first grey mirror it hits.
    From the far endpoint of that mirror, a reflected ray continues along the mirror's
    orientation using the color of the nearest other tower aligned with the exit
    endpoint's row or column, or yellow if no such tower exists.
    """
    out, _paint_ops = _solve(grid, collect=False)
    return out


def _direction_hits(grid: np.ndarray, tower_pos: tuple[int, int]) -> list[tuple[int, int]]:
    hits = []
    for step in DIRS:
        _path, hit = _trace_until_mirror(grid, tower_pos, step)
        if hit is not None:
            hits.append(step)
    return hits


def _validate_generated(grid: np.ndarray) -> bool:
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    towers = _find_towers(g, bg)
    mirrors, _mirror_id = _find_mirrors(g)

    if len(towers) != 2:
        return False
    if len(mirrors) < 2:
        return False

    for mirror in mirrors:
        if len(mirror["coords"]) not in (2, 3):
            return False

    for tower in towers:
        if len(_direction_hits(g, tower["pos"])) != 1:
            return False

    _out, ops = _solve(g, collect=True)
    seen: dict[tuple[int, int], int] = {}
    for cell, color in ops:
        if int(g[cell]) != int(bg):
            return False
        prev = seen.get(cell)
        if prev is not None and prev != color:
            return False
        if prev is not None:
            return False
        seen[cell] = color
    return True


def _pick_distinct_colors(rng: np.random.Generator, k: int, exclude: set[int]) -> list[int]:
    pool = [c for c in range(10) if c not in exclude]
    picked = rng.choice(np.array(pool, dtype=int), size=k, replace=False)
    return [int(v) for v in picked]


def _place_decoy_mirrors(
    rng: np.random.Generator,
    grid: np.ndarray,
    reserved_rows: set[int],
    reserved_cols: set[int],
    occupied: set[tuple[int, int]],
) -> None:
    h, w = grid.shape
    decoy_count = int(rng.integers(0, 3))
    for _ in range(decoy_count):
        placed = False
        for _attempt in range(80):
            length = int(rng.choice(np.array([2, 3], dtype=int)))
            horizontal = bool(int(rng.integers(0, 2)) == 0)
            if horizontal:
                r = int(rng.integers(1, h - 1))
                if r in reserved_rows:
                    continue
                c0 = int(rng.integers(1, w - length))
                cells = [(r, c0 + dc) for dc in range(length)]
            else:
                c = int(rng.integers(1, w - 1))
                if c in reserved_cols:
                    continue
                r0 = int(rng.integers(1, h - length))
                cells = [(r0 + dr, c) for dr in range(length)]

            rows = {r for r, _c in cells}
            cols = {c for _r, c in cells}
            if rows & reserved_rows or cols & reserved_cols:
                continue
            if any(cell in occupied for cell in cells):
                continue
            for cell in cells:
                grid[cell] = GREY
                occupied.add(cell)
            placed = True
            break
        if not placed:
            continue


def _build_vertical_pair(
    rng: np.random.Generator,
    bg: int,
    tower_colors: list[int],
    colored_flags: tuple[bool, bool],
) -> np.ndarray | None:
    h = int(rng.integers(12, 21))
    w = int(rng.integers(11, 18))
    grid = np.full((h, w), bg, dtype=int)

    delta = int(rng.choice(np.array([1, 2], dtype=int)))
    sign = int(rng.choice(np.array([-1, 1], dtype=int)))

    c0_lo = 3 if sign > 0 else 3 + delta
    c0_hi = (w - 4 - delta) if sign > 0 else (w - 4)
    if c0_lo > c0_hi:
        return None
    c0 = int(rng.integers(c0_lo, c0_hi + 1))
    c1 = int(c0 + sign * delta)

    r0 = int(rng.integers(2, h - 8))
    r1 = int(rng.integers(r0 + 6, h - 2))

    m0_lo, m0_hi = r0 + 2, r1 - 4
    m1_lo, m1_hi = r0 + 4, r1 - 2
    if m0_lo > m0_hi or m1_lo > m1_hi:
        return None

    for _attempt in range(80):
        m0 = int(rng.integers(m0_lo, m0_hi + 1))
        m1 = int(rng.integers(max(m0 + 1, m1_lo), m1_hi + 1))
        if m0 < m1:
            break
    else:
        return None

    specs = []
    for idx, (tower_r, tower_c, other_c, mirror_r, use_color) in enumerate(
        [
            (r0, c0, c1, m0, colored_flags[0]),
            (r1, c1, c0, m1, colored_flags[1]),
        ]
    ):
        if use_color:
            length = abs(other_c - tower_c) + 1
            exit_sign = int(np.sign(other_c - tower_c))
        else:
            options = []
            for length in (2, 3):
                for exit_sign in (-1, 1):
                    exit_c = int(tower_c + exit_sign * (length - 1))
                    if exit_c == other_c:
                        continue
                    if 1 <= min(tower_c, exit_c) and max(tower_c, exit_c) <= w - 2:
                        options.append((length, exit_sign))
            if not options:
                return None
            length, exit_sign = options[int(rng.integers(0, len(options)))]

        exit_c = int(tower_c + exit_sign * (length - 1))
        if not (1 <= min(tower_c, exit_c) and max(tower_c, exit_c) <= w - 2):
            return None
        span = range(tower_c, exit_c + exit_sign, exit_sign)
        cells = [(mirror_r, int(cc)) for cc in span]
        specs.append(
            {
                "tower": (tower_r, tower_c),
                "mirror": cells,
                "color": tower_colors[idx],
            }
        )

    occupied: set[tuple[int, int]] = set()
    for spec in specs:
        tr, tc = spec["tower"]
        if (tr, tc) in occupied:
            return None
        grid[tr, tc] = int(spec["color"])
        occupied.add((tr, tc))
        for cell in spec["mirror"]:
            if cell in occupied:
                return None
            grid[cell] = GREY
            occupied.add(cell)

    reserved_rows = {r0, r1, m0, m1}
    reserved_cols = {c0, c1}
    _place_decoy_mirrors(rng, grid, reserved_rows, reserved_cols, occupied)
    return grid


def _build_horizontal_pair(
    rng: np.random.Generator,
    bg: int,
    tower_colors: list[int],
    colored_flags: tuple[bool, bool],
) -> np.ndarray | None:
    h = int(rng.integers(11, 18))
    w = int(rng.integers(12, 21))
    grid = np.full((h, w), bg, dtype=int)

    delta = int(rng.choice(np.array([1, 2], dtype=int)))
    sign = int(rng.choice(np.array([-1, 1], dtype=int)))

    r0_lo = 3 if sign > 0 else 3 + delta
    r0_hi = (h - 4 - delta) if sign > 0 else (h - 4)
    if r0_lo > r0_hi:
        return None
    r0 = int(rng.integers(r0_lo, r0_hi + 1))
    r1 = int(r0 + sign * delta)

    c0 = int(rng.integers(2, w - 8))
    c1 = int(rng.integers(c0 + 6, w - 2))

    m0_lo, m0_hi = c0 + 2, c1 - 4
    m1_lo, m1_hi = c0 + 4, c1 - 2
    if m0_lo > m0_hi or m1_lo > m1_hi:
        return None

    for _attempt in range(80):
        m0 = int(rng.integers(m0_lo, m0_hi + 1))
        m1 = int(rng.integers(max(m0 + 1, m1_lo), m1_hi + 1))
        if m0 < m1:
            break
    else:
        return None

    specs = []
    for idx, (tower_r, tower_c, other_r, mirror_c, use_color) in enumerate(
        [
            (r0, c0, r1, m0, colored_flags[0]),
            (r1, c1, r0, m1, colored_flags[1]),
        ]
    ):
        if use_color:
            length = abs(other_r - tower_r) + 1
            exit_sign = int(np.sign(other_r - tower_r))
        else:
            options = []
            for length in (2, 3):
                for exit_sign in (-1, 1):
                    exit_r = int(tower_r + exit_sign * (length - 1))
                    if exit_r == other_r:
                        continue
                    if 1 <= min(tower_r, exit_r) and max(tower_r, exit_r) <= h - 2:
                        options.append((length, exit_sign))
            if not options:
                return None
            length, exit_sign = options[int(rng.integers(0, len(options)))]

        exit_r = int(tower_r + exit_sign * (length - 1))
        if not (1 <= min(tower_r, exit_r) and max(tower_r, exit_r) <= h - 2):
            return None
        span = range(tower_r, exit_r + exit_sign, exit_sign)
        cells = [(int(rr), mirror_c) for rr in span]
        specs.append(
            {
                "tower": (tower_r, tower_c),
                "mirror": cells,
                "color": tower_colors[idx],
            }
        )

    occupied: set[tuple[int, int]] = set()
    for spec in specs:
        tr, tc = spec["tower"]
        if (tr, tc) in occupied:
            return None
        grid[tr, tc] = int(spec["color"])
        occupied.add((tr, tc))
        for cell in spec["mirror"]:
            if cell in occupied:
                return None
            grid[cell] = GREY
            occupied.add(cell)

    reserved_rows = {r0, r1}
    reserved_cols = {c0, c1, m0, m1}
    _place_decoy_mirrors(rng, grid, reserved_rows, reserved_cols, occupied)
    return grid


def generate_input() -> np.ndarray:
    patterns = [(False, False), (False, True), (True, False), (True, True)]
    rng = np.random.default_rng()

    for _attempt in range(MAX_GENERATE_ATTEMPTS):
        bg = _pick_distinct_colors(rng, 1, {GREY})[0]
        tower_colors = _pick_distinct_colors(rng, 2, {GREY, YELLOW, bg})
        colored_flags = patterns[int(rng.integers(0, len(patterns)))]

        if bool(int(rng.integers(0, 2)) == 0):
            grid = _build_vertical_pair(rng, bg, tower_colors, colored_flags)
        else:
            grid = _build_horizontal_pair(rng, bg, tower_colors, colored_flags)

        if grid is None:
            continue
        if _validate_generated(grid):
            return grid

    raise RuntimeError("failed to generate a valid signal-tower mirror puzzle")
