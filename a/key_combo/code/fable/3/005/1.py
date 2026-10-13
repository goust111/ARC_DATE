import random
from collections import deque

import numpy as np


BLACK = 0
COLORS = tuple(range(1, 10))


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _neighbors8(r: int, c: int):
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr != 0 or dc != 0:
                yield r + dr, c + dc


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or not mask[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _divisors(n: int) -> list[int]:
    return [k for k in range(1, n + 1) if n % k == 0]


def _pattern_bbox(g: np.ndarray) -> tuple[int, int, int, int]:
    comps = _components(g != BLACK)
    assert comps, "expected visible pattern cells and spare pixels"
    largest = max(comps, key=lambda x: x.shape[0])
    assert largest.shape[0] > 1, "expected the periodic pattern to be the largest component"
    return _bbox(largest)


def _period_demands(pattern: np.ndarray) -> tuple[int, int, dict[tuple[int, int], int]]:
    h, w = pattern.shape
    holes = [(int(r), int(c)) for r, c in np.argwhere(pattern == BLACK)]
    assert holes, "expected at least one black hole inside the pattern"
    visible = [(int(r), int(c), int(pattern[r, c])) for r, c in np.argwhere(pattern != BLACK)]

    def check(pr: int, pc: int):
        residue_color: dict[tuple[int, int], int] = {}
        for r, c, color in visible:
            key = (r % pr, c % pc)
            old = residue_color.get(key)
            if old is None:
                residue_color[key] = color
            elif old != color:
                return None

        demands: dict[tuple[int, int], int] = {}
        for r, c in holes:
            color = residue_color.get((r % pr, c % pc))
            if color is None:
                return None
            demands[(r, c)] = color
        return demands

    candidate_periods = [(pr, pc) for pr in _divisors(h) for pc in _divisors(w)]
    candidate_periods += [
        (pr, pc)
        for pr in range(1, h + 1)
        for pc in range(1, w + 1)
        if h % pr != 0 or w % pc != 0
    ]

    candidates = []
    for pr, pc in candidate_periods:
        demands = check(pr, pc)
        if demands is not None:
            candidates.append((pr * pc, pr + pc, pr, pc, demands))

    assert candidates, "visible cells do not determine a periodic color for every hole"
    _area, _span, pr, pc, demands = min(candidates, key=lambda x: x[:4])
    return pr, pc, demands


def _outside_spares(g: np.ndarray, bbox: tuple[int, int, int, int]) -> list[dict]:
    r0, c0, r1, c1 = bbox
    spares = []
    for r, c in np.argwhere(g != BLACK):
        rr, cc = int(r), int(c)
        if r0 <= rr <= r1 and c0 <= cc <= c1:
            continue
        color = int(g[rr, cc])
        spares.append({"r": rr, "c": cc, "color": color})
    return spares


def _path_to_pattern(
    spare: dict, bbox: tuple[int, int, int, int]
) -> tuple[int, list[tuple[int, int]]]:
    r0, c0, r1, c1 = bbox
    r, c = int(spare["r"]), int(spare["c"])
    if r0 <= r <= r1 and c < c0:
        return c0 - c, [(r - r0, cc - c0) for cc in range(c0, c1 + 1)]
    if r0 <= r <= r1 and c > c1:
        return c - c1, [(r - r0, cc - c0) for cc in range(c1, c0 - 1, -1)]
    if c0 <= c <= c1 and r < r0:
        return r0 - r, [(rr - r0, c - c0) for rr in range(r0, r1 + 1)]
    if c0 <= c <= c1 and r > r1:
        return r - r1, [(rr - r0, c - c0) for rr in range(r1, r0 - 1, -1)]
    return 10**9, []


def _apply_repair(
    g: np.ndarray,
    bbox: tuple[int, int, int, int],
    demands: dict[tuple[int, int], int],
    spares: list[dict],
    ordered_spares: list[dict] | None = None,
):
    r0, c0, _r1, _c1 = bbox
    out = g.copy()
    for spare in spares:
        out[int(spare["r"]), int(spare["c"])] = BLACK

    available = set(demands)
    filled: list[tuple[dict, tuple[int, int]]] = []
    deleted: list[dict] = []
    if ordered_spares is None:
        ordered_spares = sorted(
            spares,
            key=lambda s: (_path_to_pattern(s, bbox)[0], int(s["r"]), int(s["c"])),
        )

    for spare in ordered_spares:
        _dist, path = _path_to_pattern(spare, bbox)
        color = int(spare["color"])
        target = None
        for loc in path:
            if loc in available and demands[loc] == color:
                target = loc
                break
        if target is None:
            deleted.append(spare)
            continue
        tr, tc = target
        out[r0 + tr, c0 + tc] = color
        available.remove(target)
        filled.append((spare, target))

    return out, {"filled": filled, "deleted": deleted, "unfilled": available}


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repair only the black holes supplied by matching spare pixels sliding in from outside.

    Input structure:
    - A black canvas contains one rectangular, fully colored periodic pattern region.
    - Some cells inside that region are black holes hiding periodic colors.
    - Outside the region, isolated colored single pixels lie in rows or columns that cross the pattern.

    Rule:
    - Infer the smallest visible translational period of the pattern region, ignoring black holes.
    - Give every internal black hole its periodic demand color from that period.
    - Process outside spare pixels from nearest to farthest from the pattern; each slides straight along
      its row or column and fills the first still-empty hole on that path whose demand color equals
      the spare color.
    - Delete every spare pixel; holes without a matching supplied spare remain black.

    Key insight:
    - Periodicity determines what color each invisible hole needs, but the visible output is gated by
      the outside supply pixels and their straight sliding paths.
    """
    g = np.asarray(grid, dtype=int)
    bbox = _pattern_bbox(g)
    r0, c0, r1, c1 = bbox
    pattern = g[r0 : r1 + 1, c0 : c1 + 1]
    _pr, _pc, demands = _period_demands(pattern)
    spares = _outside_spares(g, bbox)
    out, _stats = _apply_repair(g, bbox, demands, spares)
    return out


def _is_period(tile: np.ndarray, pr: int, pc: int) -> bool:
    h, w = tile.shape
    for r in range(h):
        for c in range(w):
            if int(tile[r, c]) != int(tile[r % pr, c % pc]):
                return False
    return True


def _primitive_period(tile: np.ndarray) -> tuple[int, int]:
    h, w = tile.shape
    for pr in _divisors(h):
        for pc in _divisors(w):
            if _is_period(tile, pr, pc):
                return pr, pc
    return h, w


def _random_primitive_tile(pr: int, pc: int, palette: list[int]) -> np.ndarray:
    for _ in range(300):
        tile = np.array(
            [[random.choice(palette) for _ in range(pc)] for _ in range(pr)],
            dtype=int,
        )
        if len(set(map(int, tile.flatten()))) < min(3, len(palette)):
            continue
        if _primitive_period(tile) == (pr, pc):
            return tile
    raise RuntimeError("failed to build a primitive tile")


def _far_from(points: set[tuple[int, int]], r: int, c: int) -> bool:
    return all(max(abs(r - rr), abs(c - cc)) > 1 for rr, cc in points)


def _can_place_spare(g: np.ndarray, r: int, c: int, bbox: tuple[int, int, int, int]) -> bool:
    h, w = g.shape
    r0, c0, r1, c1 = bbox
    if not (0 <= r < h and 0 <= c < w):
        return False
    if r0 <= r <= r1 and c0 <= c <= c1:
        return False
    if int(g[r, c]) != BLACK:
        return False
    for nr, nc in _neighbors8(r, c):
        if 0 <= nr < h and 0 <= nc < w and int(g[nr, nc]) != BLACK:
            return False
    return True


def _place_extra_spare(
    g: np.ndarray,
    bbox: tuple[int, int, int, int],
    local_cell: tuple[int, int],
    color: int,
    used_distances: set[int],
):
    r0, c0, r1, c1 = bbox
    lr, lc = local_cell
    options = []
    for dist in range(2, c0 + 1):
        options.append((r0 + lr, c0 - dist, dist))
    for dist in range(2, g.shape[1] - c1):
        options.append((r0 + lr, c1 + dist, dist))
    for dist in range(2, r0 + 1):
        options.append((r0 - dist, c0 + lc, dist))
    for dist in range(2, g.shape[0] - r1):
        options.append((r1 + dist, c0 + lc, dist))
    random.shuffle(options)
    for rr, cc, dist in options:
        if dist in used_distances:
            continue
        if _can_place_spare(g, rr, cc, bbox):
            g[rr, cc] = color
            used_distances.add(dist)
            return {"r": rr, "c": cc, "color": color, "name": "extra"}
    return None


def _find_unfilled_cell(
    full: np.ndarray, holes: set[tuple[int, int]], avoid_color: int
) -> tuple[int, int] | None:
    h, w = full.shape
    candidates = [
        (r, c)
        for r in range(h)
        for c in range(w)
        if int(full[r, c]) != avoid_color and _far_from(holes, r, c)
    ]
    random.shuffle(candidates)
    return candidates[0] if candidates else None


def _find_extra_hole(
    full: np.ndarray,
    holes: set[tuple[int, int]],
    avoid_color: int,
    avoid_col: int,
) -> tuple[int, int] | None:
    h, w = full.shape
    candidates = []
    for r in range(h):
        for c in range(w):
            if (r, c) in holes:
                continue
            if c == avoid_col and int(full[r, c]) == avoid_color:
                continue
            if _far_from(holes, r, c):
                candidates.append((r, c))
    random.shuffle(candidates)
    return candidates[0] if candidates else None


def _attempt_generate() -> np.ndarray | None:
    pr = random.choice((2, 3, 4))
    pc = random.choice((2, 3, 4))
    rep_r = random.choice((3, 4, 5))
    rep_c = random.choice((3, 4, 5))
    ph, pw = pr * rep_r, pc * rep_c

    top = random.randint(2, 5)
    bottom = random.randint(2, 5)
    left = random.randint(4, 6)
    right = random.randint(2, 5)
    h, w = top + ph + bottom, left + pw + right
    if h > 30 or w > 30:
        return None

    palette = random.sample(list(COLORS), random.randint(4, 5))
    tile = _random_primitive_tile(pr, pc, palette)
    full = np.tile(tile, (rep_r, rep_c))

    forced_row = random.randrange(ph)
    base_col = random.randrange(pc)
    first_repeat = random.randrange(max(1, rep_c - 2))
    h1 = (forced_row, base_col + first_repeat * pc)
    h2 = (forced_row, base_col + (first_repeat + 1) * pc)
    forced_color = int(full[h1])
    if int(full[h2]) != forced_color:
        return None

    holes = {h1, h2}
    unfilled = _find_unfilled_cell(full, holes, forced_color)
    if unfilled is None:
        return None
    holes.add(unfilled)

    for _ in range(random.randint(0, 4)):
        extra = _find_extra_hole(full, holes, forced_color, h1[1])
        if extra is not None:
            holes.add(extra)

    grid = np.zeros((h, w), dtype=int)
    r0, c0 = top, left
    r1, c1 = r0 + ph - 1, c0 + pw - 1
    bbox = (r0, c0, r1, c1)
    grid[r0 : r1 + 1, c0 : c1 + 1] = full
    for hr, hc in holes:
        grid[r0 + hr, c0 + hc] = BLACK

    used_distances = {2, 4}
    spare_a = {
        "r": r0 - 2,
        "c": c0 + h1[1],
        "color": forced_color,
        "name": "near_column",
    }
    spare_b = {
        "r": r0 + h1[0],
        "c": c0 - 4,
        "color": forced_color,
        "name": "far_row",
    }
    for spare in (spare_a, spare_b):
        if not _can_place_spare(grid, spare["r"], spare["c"], bbox):
            return None
        grid[spare["r"], spare["c"]] = spare["color"]
    spares = [spare_a, spare_b]

    unused_colors = [c for c in COLORS if c not in set(map(int, full.flatten()))]
    if not unused_colors:
        return None
    deleted_color = random.choice(unused_colors)
    deleted = None
    deleted_options = []
    for rr in range(r0, r1 + 1):
        for dist in range(2, w - c1):
            deleted_options.append((rr, c1 + dist, dist))
    for cc in range(c0, c1 + 1):
        for dist in range(2, top + 1):
            deleted_options.append((r0 - dist, cc, dist))
    random.shuffle(deleted_options)
    for rr, cc, dist in deleted_options:
        if dist in used_distances:
            continue
        if _can_place_spare(grid, rr, cc, bbox):
            grid[rr, cc] = deleted_color
            deleted = {"r": rr, "c": cc, "color": deleted_color, "name": "deleted"}
            used_distances.add(dist)
            break
    if deleted is None:
        return None
    spares.append(deleted)

    fillable_extras = [
        cell
        for cell in holes
        if cell not in {h1, h2, unfilled} and int(full[cell]) != int(full[unfilled])
    ]
    random.shuffle(fillable_extras)
    for cell in fillable_extras[: random.randint(0, min(2, len(fillable_extras)))]:
        spare = _place_extra_spare(grid, bbox, cell, int(full[cell]), used_distances)
        if spare is not None:
            spares.append(spare)

    known_demands = {cell: int(full[cell]) for cell in holes}
    expected, stats = _apply_repair(grid, bbox, known_demands, spares)
    if len(stats["filled"]) < 2 or len(stats["deleted"]) < 1 or len(stats["unfilled"]) < 1:
        return None
    if (h1 not in [target for _spare, target in stats["filled"]]) or (
        h2 not in [target for _spare, target in stats["filled"]]
    ):
        return None
    if int(expected[r0 + unfilled[0], c0 + unfilled[1]]) != BLACK:
        return None
    if int(expected[deleted["r"], deleted["c"]]) != BLACK:
        return None

    alt_order = [spare_b, spare_a] + [
        s for s in sorted(spares, key=lambda x: (_path_to_pattern(x, bbox)[0], x["r"], x["c"]))
        if s not in (spare_a, spare_b)
    ]
    alt, _alt_stats = _apply_repair(grid, bbox, known_demands, spares, alt_order)
    if np.array_equal(alt, expected):
        return None

    try:
        _ipr, _ipc, inferred_demands = _period_demands(grid[r0 : r1 + 1, c0 : c1 + 1])
        for cell, color in known_demands.items():
            if inferred_demands.get(cell) != color:
                return None
        solved = main(grid)
    except Exception:
        return None
    if not np.array_equal(solved, expected):
        return None
    if np.array_equal(grid, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(600):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid spare-repair input")
