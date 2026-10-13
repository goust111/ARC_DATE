import random
from collections import Counter, deque

import numpy as np


BG = 0
FRAME = 5
COLOR_POOL = [1, 2, 3, 4, 6, 7, 8, 9]
BRANCH_TYPES = ("straight_h", "straight_v", "hv", "vh", "both")
GRID_H_RANGE = (24, 30)
GRID_W_RANGE = (24, 30)
PAIR_COUNT_RANGE = (2, 4)
ZONE_MIN = 8
ZONE_MAX = 11
MAX_GENERATION_ATTEMPTS = 500


def _neighbors4(r: int, c: int, h: int, w: int):
    if r > 0:
        yield r - 1, c
    if r + 1 < h:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < w:
        yield r, c + 1


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc, h, w):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rs = [r for r, _ in coords]
    cs = [c for _, c in coords]
    return min(rs), min(cs), max(rs), max(cs)


def _segment_cells(
    start: tuple[int, int],
    end: tuple[int, int],
    *,
    include_start: bool,
    include_end: bool,
) -> list[tuple[int, int]]:
    r1, c1 = start
    r2, c2 = end
    assert r1 == r2 or c1 == c2
    dr = 0 if r1 == r2 else (1 if r2 > r1 else -1)
    dc = 0 if c1 == c2 else (1 if c2 > c1 else -1)
    pts = [(r1, c1)]
    while pts[-1] != (r2, c2):
        rr, cc = pts[-1]
        pts.append((rr + dr, cc + dc))
    if not include_start:
        pts = pts[1:]
    if not include_end:
        pts = pts[:-1]
    return pts


def _straight_route(start: tuple[int, int], end: tuple[int, int]) -> list[tuple[int, int]] | None:
    if start[0] != end[0] and start[1] != end[1]:
        return None
    return _segment_cells(start, end, include_start=False, include_end=False)


def _hv_route(start: tuple[int, int], end: tuple[int, int]) -> list[tuple[int, int]] | None:
    if start[0] == end[0] or start[1] == end[1]:
        return None
    corner = (start[0], end[1])
    first = _segment_cells(start, corner, include_start=False, include_end=True)
    second = _segment_cells(corner, end, include_start=False, include_end=False)
    return first + second


def _vh_route(start: tuple[int, int], end: tuple[int, int]) -> list[tuple[int, int]] | None:
    if start[0] == end[0] or start[1] == end[1]:
        return None
    corner = (end[0], start[1])
    first = _segment_cells(start, corner, include_start=False, include_end=True)
    second = _segment_cells(corner, end, include_start=False, include_end=False)
    return first + second


def _route_is_open(route: list[tuple[int, int]] | None, is_bg) -> bool:
    return route is not None and len(route) > 0 and all(is_bg(r, c) for r, c in route)


def _choose_route(
    start: tuple[int, int],
    end: tuple[int, int],
    is_bg,
) -> list[tuple[int, int]] | None:
    straight = _straight_route(start, end)
    if _route_is_open(straight, is_bg):
        return straight
    hv = _hv_route(start, end)
    if _route_is_open(hv, is_bg):
        return hv
    vh = _vh_route(start, end)
    if _route_is_open(vh, is_bg):
        return vh
    return None


def _find_mapping_table(grid: np.ndarray, bg: int) -> tuple[dict[int, int], np.ndarray]:
    h, w = grid.shape
    colors = [int(v) for v in np.unique(grid) if int(v) != bg]
    for color in colors:
        for comp in _components(grid == color):
            r0, c0, r1, c1 = _bbox(comp)
            if c1 - c0 + 1 != 4 or r1 - r0 + 1 < 3:
                continue
            touches_corner = (
                (r0 == 0 and c0 == 0)
                or (r0 == 0 and c1 == w - 1)
                or (r1 == h - 1 and c0 == 0)
                or (r1 == h - 1 and c1 == w - 1)
            )
            if not touches_corner:
                continue
            frame = {
                (rr, cc)
                for rr in range(r0, r1 + 1)
                for cc in range(c0, c1 + 1)
                if rr in (r0, r1) or cc in (c0, c1)
            }
            if set(comp) != frame:
                continue
            interior = grid[r0 + 1 : r1, c0 + 1 : c1]
            if interior.shape[1] != 2 or interior.shape[0] == 0:
                continue
            if np.any(interior == bg) or np.any(interior == color):
                continue
            mapping: dict[int, int] = {}
            ok = True
            for rr in range(interior.shape[0]):
                src = int(interior[rr, 0])
                dst = int(interior[rr, 1])
                if src == dst or src in mapping:
                    ok = False
                    break
                mapping[src] = dst
            if not ok or not mapping:
                continue
            mask = np.zeros_like(grid, dtype=bool)
            mask[r0 : r1 + 1, c0 : c1 + 1] = True
            return mapping, mask
    raise AssertionError("mapping table not found")


def _degree1_cells(comp: list[tuple[int, int]], shape: tuple[int, int]) -> list[tuple[int, int]]:
    h, w = shape
    pts = set(comp)
    out = []
    for r, c in comp:
        deg = sum((nr, nc) in pts for nr, nc in _neighbors4(r, c, h, w))
        if deg == 1:
            out.append((r, c))
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pair same-color path fragments, infer the missing orthogonal connection, and
    draw that connector in the shadow color given by the corner lookup table.
    """
    g = np.asarray(grid, dtype=int)
    bg = int(Counter(map(int, g.ravel())).most_common(1)[0][0])
    mapping, table_mask = _find_mapping_table(g, bg)
    h, w = g.shape
    out = g.copy()

    for color, shadow in mapping.items():
        mask = (g == int(color)) & ~table_mask
        comps = _components(mask)
        assert len(comps) == 2, f"expected two fragments for color {color}"

        cand1 = _degree1_cells(comps[0], (h, w))
        cand2 = _degree1_cells(comps[1], (h, w))
        assert cand1 and cand2, f"missing endpoint candidates for color {color}"

        chosen_route = None
        if len(cand1) == 1 and len(cand2) == 1:
            chosen_route = _choose_route(
                cand1[0],
                cand2[0],
                lambda r, c: int(g[r, c]) == bg,
            )
        if chosen_route is None:
            valid_routes = []
            for p1 in cand1:
                for p2 in cand2:
                    route = _choose_route(
                        p1,
                        p2,
                        lambda r, c: int(g[r, c]) == bg,
                    )
                    if route is not None:
                        valid_routes.append(route)
            assert len(valid_routes) == 1, f"ambiguous endpoint pairing for color {color}"
            chosen_route = valid_routes[0]

        for r, c in chosen_route:
            out[r, c] = int(shadow)
    return out


def _rect_perimeter(top: int, left: int, height: int, width: int) -> list[tuple[int, int]]:
    return [
        (r, c)
        for r in range(top, top + height)
        for c in range(left, left + width)
        if r in (top, top + height - 1) or c in (left, left + width - 1)
    ]


def _build_fragment_variant(
    endpoint: tuple[int, int],
    side: str,
    ring_h: int,
    ring_w: int,
    attach: int,
) -> list[tuple[int, int]]:
    er, ec = endpoint
    if side == "W":
        top = er - attach
        left = ec - ring_w - 1
        cells = _rect_perimeter(top, left, ring_h, ring_w)
        cells.extend([(er, ec - 1), (er, ec)])
        return cells
    if side == "E":
        top = er - attach
        left = ec + 2
        cells = _rect_perimeter(top, left, ring_h, ring_w)
        cells.extend([(er, ec + 1), (er, ec)])
        return cells
    if side == "N":
        top = er - ring_h - 1
        left = ec - attach
        cells = _rect_perimeter(top, left, ring_h, ring_w)
        cells.extend([(er - 1, ec), (er, ec)])
        return cells
    top = er + 2
    left = ec - attach
    cells = _rect_perimeter(top, left, ring_h, ring_w)
    cells.extend([(er + 1, ec), (er, ec)])
    return cells


def _fragment_variants(
    endpoint: tuple[int, int],
    zone_h: int,
    zone_w: int,
) -> list[list[tuple[int, int]]]:
    variants: list[list[tuple[int, int]]] = []
    seen: set[tuple[tuple[int, int], ...]] = set()
    for side in ("W", "E", "N", "S"):
        for ring_h in (3, 4):
            for ring_w in (3, 4):
                attach_values = range(1, ring_h - 1) if side in ("W", "E") else range(1, ring_w - 1)
                for attach in attach_values:
                    coords = _build_fragment_variant(endpoint, side, ring_h, ring_w, attach)
                    if not all(0 <= r < zone_h and 0 <= c < zone_w for r, c in coords):
                        continue
                    key = tuple(sorted(set(coords)))
                    if len(key) != len(coords) or key in seen:
                        continue
                    seen.add(key)
                    variants.append(list(key))
    return variants


def _classify_branch(
    start: tuple[int, int],
    end: tuple[int, int],
    occupied: set[tuple[int, int]],
) -> str | None:
    free = lambda r, c: (r, c) not in occupied
    if start[0] == end[0]:
        return "straight_h" if _route_is_open(_straight_route(start, end), free) else None
    if start[1] == end[1]:
        return "straight_v" if _route_is_open(_straight_route(start, end), free) else None
    hv_ok = _route_is_open(_hv_route(start, end), free)
    vh_ok = _route_is_open(_vh_route(start, end), free)
    if hv_ok and vh_ok:
        return "both"
    if hv_ok:
        return "hv"
    if vh_ok:
        return "vh"
    return None


def _route_has_buffer(
    route: list[tuple[int, int]],
    occupied: set[tuple[int, int]],
    endpoints: set[tuple[int, int]],
    zone_h: int,
    zone_w: int,
) -> bool:
    for r, c in route:
        for nr, nc in _neighbors4(r, c, zone_h, zone_w):
            if (nr, nc) in occupied and (nr, nc) not in endpoints:
                return False
    return True


def _sample_endpoint_pairs(zone_h: int, zone_w: int, branch: str, rng: random.Random):
    inner_rows = list(range(1, zone_h - 1))
    inner_cols = list(range(1, zone_w - 1))
    pairs = []
    seen = set()
    attempts = 0
    while len(pairs) < 120 and attempts < 2000:
        attempts += 1
        if branch == "straight_h":
            r = rng.choice(inner_rows)
            c1, c2 = sorted(rng.sample(inner_cols, 2))
            if c2 - c1 < 4:
                continue
            p1, p2 = (r, c1), (r, c2)
        elif branch == "straight_v":
            c = rng.choice(inner_cols)
            r1, r2 = sorted(rng.sample(inner_rows, 2))
            if r2 - r1 < 4:
                continue
            p1, p2 = (r1, c), (r2, c)
        else:
            r1, r2 = rng.sample(inner_rows, 2)
            c1, c2 = rng.sample(inner_cols, 2)
            if abs(r1 - r2) < 2 or abs(c1 - c2) < 2:
                continue
            p1, p2 = (r1, c1), (r2, c2)
        key = (p1, p2)
        if key in seen:
            continue
        seen.add(key)
        pairs.append((p1, p2))
    rng.shuffle(pairs)
    return pairs


def _generate_pair_in_zone(zone_h: int, zone_w: int, branch: str, rng: random.Random):
    endpoint_pairs = _sample_endpoint_pairs(zone_h, zone_w, branch, rng)
    for p1, p2 in endpoint_pairs:
        variants1 = _fragment_variants(p1, zone_h, zone_w)
        variants2 = _fragment_variants(p2, zone_h, zone_w)
        rng.shuffle(variants1)
        rng.shuffle(variants2)
        for frag1 in variants1:
            set1 = set(frag1)
            for frag2 in variants2:
                set2 = set(frag2)
                if set1 & set2:
                    continue
                occupied = set1 | set2
                actual_branch = _classify_branch(p1, p2, occupied)
                if actual_branch != branch:
                    continue
                route = _choose_route(p1, p2, lambda r, c: (r, c) not in occupied)
                assert route is not None
                if not _route_has_buffer(route, occupied, {p1, p2}, zone_h, zone_w):
                    continue
                return frag1, frag2, route
    return None


def _mark_rect(mask: np.ndarray, top: int, left: int, bottom: int, right: int, pad: int) -> None:
    h, w = mask.shape
    r0 = max(0, top - pad)
    c0 = max(0, left - pad)
    r1 = min(h - 1, bottom + pad)
    c1 = min(w - 1, right + pad)
    mask[r0 : r1 + 1, c0 : c1 + 1] = True


def _pick_zone(reserved: np.ndarray, rng: random.Random):
    h, w = reserved.shape
    for _ in range(200):
        zh = rng.randint(ZONE_MIN, min(ZONE_MAX, h - 2))
        zw = rng.randint(ZONE_MIN, min(ZONE_MAX, w - 2))
        top = rng.randint(0, h - zh)
        left = rng.randint(0, w - zw)
        if reserved[top : top + zh, left : left + zw].any():
            continue
        if reserved[max(0, top - 1) : min(h, top + zh + 1), max(0, left - 1) : min(w, left + zw + 1)].any():
            continue
        return top, left, zh, zw
    return None


def _table_bbox(corner: str, table_h: int, table_w: int, h: int, w: int) -> tuple[int, int, int, int]:
    if corner == "tl":
        return 0, 0, table_h - 1, table_w - 1
    if corner == "tr":
        return 0, w - table_w, table_h - 1, w - 1
    if corner == "bl":
        return h - table_h, 0, h - 1, table_w - 1
    return h - table_h, w - table_w, h - 1, w - 1


def _paint_table(
    grid: np.ndarray,
    rows: list[tuple[int, int]],
    corner: str,
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    h, w = grid.shape
    table_h = len(rows) + 2
    table_w = 4
    r0, c0, r1, c1 = _table_bbox(corner, table_h, table_w, h, w)
    grid[r0 : r1 + 1, c0 : c1 + 1] = BG
    grid[r0, c0 : c1 + 1] = FRAME
    grid[r1, c0 : c1 + 1] = FRAME
    grid[r0 : r1 + 1, c0] = FRAME
    grid[r0 : r1 + 1, c1] = FRAME
    for i, (src, dst) in enumerate(rows):
        grid[r0 + 1 + i, c0 + 1] = int(src)
        grid[r0 + 1 + i, c0 + 2] = int(dst)
    mask = np.zeros_like(grid, dtype=bool)
    mask[r0 : r1 + 1, c0 : c1 + 1] = True
    return mask, (r0, c0, r1, c1)


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    pair_count = rng.randint(*PAIR_COUNT_RANGE)
    h = rng.randint(*GRID_H_RANGE)
    w = rng.randint(*GRID_W_RANGE)
    grid = np.full((h, w), BG, dtype=int)
    expected = np.full((h, w), BG, dtype=int)

    chosen = rng.sample(COLOR_POOL, 2 * pair_count)
    originals = chosen[:pair_count]
    shadows = chosen[pair_count:]
    rows = list(zip(originals, shadows))
    rng.shuffle(rows)
    branches = [rng.choice(BRANCH_TYPES) for _ in range(pair_count)]

    corner = rng.choice(["tl", "tr", "bl", "br"])
    _, table_box = _paint_table(grid, rows, corner)
    _paint_table(expected, rows, corner)

    reserved = np.zeros((h, w), dtype=bool)
    _mark_rect(reserved, *table_box, pad=1)

    for (src, dst), branch in zip(rows, branches):
        zone = _pick_zone(reserved, rng)
        if zone is None:
            return None
        top, left, zh, zw = zone
        pair = _generate_pair_in_zone(zh, zw, branch, rng)
        if pair is None:
            return None
        frag1, frag2, route = pair

        abs_frag1 = [(top + r, left + c) for r, c in frag1]
        abs_frag2 = [(top + r, left + c) for r, c in frag2]
        abs_route = [(top + r, left + c) for r, c in route]

        for r, c in abs_frag1 + abs_frag2:
            grid[r, c] = int(src)
            expected[r, c] = int(src)
        for r, c in abs_route:
            expected[r, c] = int(dst)

        _mark_rect(reserved, top, left, top + zh - 1, left + zw - 1, pad=1)

    if np.array_equal(grid, expected):
        return None
    produced = main(grid)
    if not np.array_equal(produced, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        try:
            grid = _gen_attempt(rng)
        except AssertionError:
            grid = None
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
