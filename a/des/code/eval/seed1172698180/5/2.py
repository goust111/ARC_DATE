from collections import Counter, defaultdict, deque

import numpy as np


DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
ALL_COLORS = tuple(range(10))
GEN_RETRIES = 400


def _most_common_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _neighbors(r: int, c: int, h: int, w: int):
    for dr, dc in DIR4:
        nr = r + dr
        nc = c + dc
        if 0 <= nr < h and 0 <= nc < w:
            yield nr, nc


def _bfs_dist(mask: np.ndarray, start: tuple[int, int]) -> np.ndarray:
    h, w = mask.shape
    dist = -np.ones((h, w), dtype=int)
    q = deque([start])
    dist[start] = 0
    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors(r, c, h, w):
            if mask[nr, nc] and dist[nr, nc] == -1:
                dist[nr, nc] = dist[r, c] + 1
                q.append((nr, nc))
    return dist


def _ray_cells(mask: np.ndarray, start: tuple[int, int], dr: int, dc: int) -> list[tuple[int, int]]:
    h, w = mask.shape
    r = start[0] + dr
    c = start[1] + dc
    cells = []
    while 0 <= r < h and 0 <= c < w and mask[r, c]:
        cells.append((r, c))
        r += dr
        c += dc
    return cells


def _pick_spaced(low: int, high: int, count: int, rng: np.random.Generator) -> list[int] | None:
    values = list(range(low, high + 1))
    for _ in range(120):
        pick = sorted(int(v) for v in rng.choice(values, size=count, replace=False))
        if all(b - a >= 2 for a, b in zip(pick, pick[1:])):
            return pick
    return None


def _channel_degree(mask: np.ndarray, cell: tuple[int, int]) -> int:
    r, c = cell
    return sum(1 for nr, nc in _neighbors(r, c, *mask.shape) if mask[nr, nc])


def _infer_structure(grid: np.ndarray):
    g = np.asarray(grid, dtype=int)
    bg = _most_common_color(g)
    non_bg = [int(c) for c in np.unique(g) if int(c) != bg]
    if not non_bg:
        return bg, None, [], {}
    counts = Counter(int(v) for v in g.ravel() if int(v) != bg)
    grey = max(non_bg, key=lambda c: (counts[c], -c))
    seed_colors = sorted(c for c in non_bg if c != grey)
    seed_positions = {
        color: [tuple(map(int, rc)) for rc in np.argwhere(g == color)]
        for color in seed_colors
    }
    return bg, grey, seed_colors, seed_positions


def _competitive_spread(grid: np.ndarray, with_meta: bool = False):
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg, grey, seed_colors, seed_positions = _infer_structure(g)
    out = g.copy()
    if grey is None or not seed_colors:
        meta = {
            "bg": bg,
            "grey": grey,
            "seed_counts": {},
            "tie_cells": set(),
            "winner_cells": set(),
        }
        return (out, meta) if with_meta else out

    channel = g != bg
    seed_mask = np.isin(g, seed_colors)
    seed_counts = {color: len(seed_positions[color]) for color in seed_colors}

    owner = np.full((h, w), -1, dtype=int)
    strength = np.full((h, w), -1, dtype=int)
    tied = np.zeros((h, w), dtype=bool)

    active = defaultdict(list)
    max_power = 0
    for color, coords in seed_positions.items():
        power = seed_counts[color]
        max_power = max(max_power, power)
        for r, c in coords:
            owner[r, c] = color
            strength[r, c] = power
            active[power].append((r, c, color))

    tie_cells: set[tuple[int, int]] = set()
    winner_cells: set[tuple[int, int]] = set()

    for power in range(max_power, 0, -1):
        states = active.get(power, [])
        if not states:
            continue

        proposals: dict[tuple[int, int], set[int]] = defaultdict(set)
        next_power = power - 1
        for r, c, color in states:
            for nr, nc in _neighbors(r, c, h, w):
                if not channel[nr, nc]:
                    continue
                cell_color = int(g[nr, nc])
                if cell_color != grey and cell_color != color:
                    continue
                if seed_mask[nr, nc] and cell_color == color:
                    continue
                if tied[nr, nc]:
                    continue
                if owner[nr, nc] == color:
                    continue
                if owner[nr, nc] not in (-1, color):
                    winner_cells.add((nr, nc))
                    continue
                proposals[(nr, nc)].add(color)

        for (r, c), colors in proposals.items():
            if tied[r, c] or owner[r, c] != -1:
                continue
            if len(colors) == 1:
                color = next(iter(colors))
                owner[r, c] = color
                strength[r, c] = next_power
                if next_power > 0:
                    active[next_power].append((r, c, color))
            else:
                tied[r, c] = True
                strength[r, c] = next_power
                tie_cells.add((r, c))

    grey_cells = np.argwhere(g == grey)
    for r, c in grey_cells:
        rr = int(r)
        cc = int(c)
        if owner[rr, cc] >= 0 and not tied[rr, cc]:
            out[rr, cc] = owner[rr, cc]

    meta = {
        "bg": bg,
        "grey": grey,
        "seed_counts": seed_counts,
        "tie_cells": tie_cells,
        "winner_cells": winner_cells,
    }
    return (out, meta) if with_meta else out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Spread colored seed pixels along a thin orthogonal channel network.

    The channel color is the dominant non-background color. Every other
    non-background color is treated as a seed color. A color with N seeds
    expands from each seed up to N steps through the channel, using a
    competitive wavefront:
    - seeds are fixed;
    - a wave may only continue through cells it actually captures;
    - stronger remaining range wins a conflict;
    - equal remaining range leaves the cell grey.
    """
    return _competitive_spread(grid, with_meta=False)


def _build_channel_mask(rng: np.random.Generator):
    for _ in range(160):
        h = int(rng.integers(10, 19))
        w = int(rng.integers(10, 19))
        row_count = int(rng.integers(2, 5))
        col_count = int(rng.integers(2, 5))

        min_box_h = max(5, 2 * row_count)
        min_box_w = max(5, 2 * col_count)
        max_box_h = h - 2
        max_box_w = w - 2
        if min_box_h > max_box_h or min_box_w > max_box_w:
            continue

        box_h = int(rng.integers(min_box_h, max_box_h + 1))
        box_w = int(rng.integers(min_box_w, max_box_w + 1))
        r0_max = h - box_h - 1
        c0_max = w - box_w - 1
        if r0_max < 1 or c0_max < 1:
            continue

        r0 = int(rng.integers(1, r0_max + 1))
        c0 = int(rng.integers(1, c0_max + 1))
        r1 = r0 + box_h - 1
        c1 = c0 + box_w - 1

        rows = _pick_spaced(r0, r1, row_count, rng)
        cols = _pick_spaced(c0, c1, col_count, rng)
        if rows is None or cols is None:
            continue

        mask = np.zeros((h, w), dtype=bool)
        for r in rows:
            mask[r, c0 : c1 + 1] = True
        for c in cols:
            mask[r0 : r1 + 1, c] = True

        for _ in range(int(rng.integers(0, 4))):
            if rng.random() < 0.5:
                r = int(rng.choice(rows))
                if rng.random() < 0.5 and c0 > 1:
                    max_len = c0 - 1
                    length = int(rng.integers(1, max_len + 1))
                    mask[r, c0 - length : c0] = True
                elif c1 < w - 2:
                    max_len = w - c1 - 2
                    if max_len >= 1:
                        length = int(rng.integers(1, max_len + 1))
                        mask[r, c1 + 1 : c1 + 1 + length] = True
            else:
                c = int(rng.choice(cols))
                if rng.random() < 0.5 and r0 > 1:
                    max_len = r0 - 1
                    length = int(rng.integers(1, max_len + 1))
                    mask[r0 - length : r0, c] = True
                elif r1 < h - 2:
                    max_len = h - r1 - 2
                    if max_len >= 1:
                        length = int(rng.integers(1, max_len + 1))
                        mask[r1 + 1 : r1 + 1 + length, c] = True

        channel_cells = [tuple(map(int, rc)) for rc in np.argwhere(mask)]
        if len(channel_cells) < 18:
            continue

        intersections = [(r, c) for r in rows for c in cols]
        return mask, intersections
    return None


def _select_cells(
    candidates: list[tuple[int, int]],
    count: int,
    used: set[tuple[int, int]],
    rng: np.random.Generator,
    min_gap: int = 2,
) -> list[tuple[int, int]] | None:
    order = list(candidates)
    rng.shuffle(order)
    chosen: list[tuple[int, int]] = []
    for cell in order:
        if cell in used:
            continue
        if any(abs(cell[0] - rr) + abs(cell[1] - cc) < min_gap for rr, cc in used):
            continue
        if any(abs(cell[0] - rr) + abs(cell[1] - cc) < min_gap for rr, cc in chosen):
            continue
        chosen.append(cell)
        if len(chosen) == count:
            return chosen
    return None


def _conflict_counts(
    scenario: str,
    ray1_len: int,
    ray2_len: int,
    rng: np.random.Generator,
) -> tuple[int, int, int, int] | None:
    for _ in range(120):
        n1 = int(rng.integers(1, 4))
        n2 = int(rng.integers(1, 4))
        if scenario == "tie":
            rem = int(rng.integers(0, min(n1, n2)))
            d1 = n1 - rem
            d2 = n2 - rem
            if d1 <= ray1_len and d2 <= ray2_len and max(n1, n2) > 1:
                return n1, n2, d1, d2
        else:
            d1 = int(rng.integers(1, min(n1, ray1_len) + 1))
            d2 = int(rng.integers(1, min(n2, ray2_len) + 1))
            if n1 - d1 != n2 - d2 and max(n1, n2) > 1:
                return n1, n2, d1, d2
    return None


def _paint_grid(
    mask: np.ndarray,
    bg: int,
    grey: int,
    seeds_by_color: dict[int, list[tuple[int, int]]],
) -> np.ndarray:
    grid = np.full(mask.shape, bg, dtype=int)
    grid[mask] = grey
    for color, coords in seeds_by_color.items():
        for r, c in coords:
            grid[r, c] = color
    return grid


def _place_conflict_scenario(mask: np.ndarray, intersections: list[tuple[int, int]], scenario: str, rng: np.random.Generator):
    if not intersections:
        return None

    colors_pool = list(ALL_COLORS)
    for _ in range(160):
        bg = int(rng.choice(colors_pool))
        grey_choices = [c for c in colors_pool if c != bg]
        grey = int(rng.choice(grey_choices))
        seed_palette = [c for c in colors_pool if c not in (bg, grey)]
        rng.shuffle(seed_palette)
        c1, c2 = seed_palette[:2]

        target = tuple(intersections[int(rng.integers(0, len(intersections)))])
        rays = {
            "up": _ray_cells(mask, target, -1, 0),
            "down": _ray_cells(mask, target, 1, 0),
            "left": _ray_cells(mask, target, 0, -1),
            "right": _ray_cells(mask, target, 0, 1),
        }
        usable = [name for name, cells in rays.items() if len(cells) >= 1]
        if len(usable) < 2:
            continue
        dname1, dname2 = rng.choice(usable, size=2, replace=False)
        ray1 = rays[str(dname1)]
        ray2 = rays[str(dname2)]

        params = _conflict_counts(scenario, len(ray1), len(ray2), rng)
        if params is None:
            continue
        n1, n2, d1, d2 = params

        primary1 = ray1[d1 - 1]
        primary2 = ray2[d2 - 1]
        if primary1 == primary2:
            continue

        counts = {c1: n1, c2: n2}
        if rng.random() < 0.6:
            extra_colors = int(rng.integers(0, 3))
        else:
            extra_colors = 0
        extra_palette = seed_palette[2 : 2 + extra_colors]
        for color in extra_palette:
            counts[color] = int(rng.integers(1, 3))
        if max(counts.values()) == 1:
            counts[c1] = 2

        target_dist = _bfs_dist(mask, target)
        channel_cells = [tuple(map(int, rc)) for rc in np.argwhere(mask)]
        branch_cells = [cell for cell in channel_cells if _channel_degree(mask, cell) <= 2]
        pool_preferred = branch_cells if len(branch_cells) >= len(channel_cells) // 2 else channel_cells
        used = {primary1, primary2}
        seeds_by_color = {c1: [primary1], c2: [primary2]}

        ok = True
        for color, primary in ((c1, primary1), (c2, primary2)):
            need = counts[color] - 1
            if need <= 0:
                continue
            safe = [
                cell
                for cell in pool_preferred
                if cell not in used and target_dist[cell] > counts[color]
            ]
            extra = _select_cells(safe, need, used, rng)
            if extra is None:
                ok = False
                break
            seeds_by_color[color].extend(extra)
            used.update(extra)
        if not ok:
            continue

        other_colors = [color for color in counts if color not in (c1, c2)]
        for color in other_colors:
            need = counts[color]
            safe = [
                cell
                for cell in pool_preferred
                if cell not in used and target_dist[cell] > max(counts[color], n1, n2)
            ]
            placed = _select_cells(safe, need, used, rng)
            if placed is None:
                ok = False
                break
            seeds_by_color[color] = placed
            used.update(placed)
        if not ok:
            continue

        grid = _paint_grid(mask, bg, grey, seeds_by_color)
        out, meta = _competitive_spread(grid, with_meta=True)
        if np.array_equal(out, grid):
            continue
        if scenario == "winner":
            if target not in meta["winner_cells"] or meta["tie_cells"]:
                continue
            if out[target] not in (c1, c2):
                continue
        else:
            if target not in meta["tie_cells"] or meta["winner_cells"]:
                continue
            if int(out[target]) != grey:
                continue
        return grid
    return None


def _place_basic_scenario(mask: np.ndarray, rng: np.random.Generator):
    channel_cells = [tuple(map(int, rc)) for rc in np.argwhere(mask)]
    branch_cells = [cell for cell in channel_cells if _channel_degree(mask, cell) <= 2]
    candidates = branch_cells + [cell for cell in channel_cells if cell not in branch_cells]
    dist_cache: dict[tuple[int, int], np.ndarray] = {}

    def dist_from(cell: tuple[int, int]) -> np.ndarray:
        if cell not in dist_cache:
            dist_cache[cell] = _bfs_dist(mask, cell)
        return dist_cache[cell]

    for _ in range(160):
        bg = int(rng.choice(ALL_COLORS))
        grey = int(rng.choice([c for c in ALL_COLORS if c != bg]))
        palette = [c for c in ALL_COLORS if c not in (bg, grey)]
        rng.shuffle(palette)
        color_count = int(rng.integers(2, min(5, len(palette)) + 1))
        active_colors = palette[:color_count]

        counts = {color: int(rng.integers(1, 4)) for color in active_colors}
        if max(counts.values()) == 1:
            counts[active_colors[0]] = 2

        seeds_by_color: dict[int, list[tuple[int, int]]] = {color: [] for color in active_colors}
        used: set[tuple[int, int]] = set()

        colors_order = sorted(active_colors, key=lambda color: (-counts[color], color))
        ok = True
        for color in colors_order:
            need = counts[color]
            for _ in range(need):
                order = list(candidates)
                rng.shuffle(order)
                placed = None
                for cell in order:
                    if cell in used:
                        continue
                    good = True
                    for other_color, coords in seeds_by_color.items():
                        for prev in coords:
                            dist = int(dist_from(cell)[prev])
                            if other_color == color:
                                if dist != -1 and dist < 2:
                                    good = False
                                    break
                            else:
                                if dist != -1 and dist <= counts[color] + counts[other_color]:
                                    good = False
                                    break
                        if not good:
                            break
                    if good:
                        placed = cell
                        break
                if placed is None:
                    ok = False
                    break
                seeds_by_color[color].append(placed)
                used.add(placed)
            if not ok:
                break
        if not ok:
            continue

        grid = _paint_grid(mask, bg, grey, seeds_by_color)
        out, meta = _competitive_spread(grid, with_meta=True)
        if np.array_equal(out, grid):
            continue
        if meta["winner_cells"] or meta["tie_cells"]:
            continue
        return grid
    return None


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    built = _build_channel_mask(rng)
    if built is None:
        return None
    mask, intersections = built
    scenario = str(rng.choice(["basic", "winner", "tie"]))
    if scenario == "basic":
        return _place_basic_scenario(mask, rng)
    return _place_conflict_scenario(mask, intersections, scenario, rng)


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {GEN_RETRIES} attempts")
