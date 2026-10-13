import random
from collections import deque

import numpy as np


BG = 0
YELLOW = 4
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIR8 = [
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
]
GEN_RETRIES = 400


def neighbors4(r: int, c: int, h: int, w: int):
    for dr, dc in DIR4:
        rr, cc = r + dr, c + dc
        if 0 <= rr < h and 0 <= cc < w:
            yield rr, cc


def neighbors8(r: int, c: int, h: int, w: int):
    for dr, dc in DIR8:
        rr, cc = r + dr, c + dc
        if 0 <= rr < h and 0 <= cc < w:
            yield rr, cc


def right_of(direction: tuple[int, int]) -> tuple[int, int]:
    dr, dc = direction
    return dc, -dr


def left_of(direction: tuple[int, int]) -> tuple[int, int]:
    dr, dc = direction
    return -dc, dr


def mask_components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            dq = deque([(r, c)])
            while dq:
                rr, cc = dq.popleft()
                comp.append((rr, cc))
                for nr, nc in neighbors4(rr, cc, h, w):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        dq.append((nr, nc))
            comps.append(comp)
    return comps


def color_component(grid: np.ndarray, color: int, start: tuple[int, int]) -> set[tuple[int, int]]:
    h, w = grid.shape
    comp = {start}
    dq = deque([start])
    while dq:
        r, c = dq.popleft()
        for nr, nc in neighbors4(r, c, h, w):
            if int(grid[nr, nc]) == color and (nr, nc) not in comp:
                comp.add((nr, nc))
                dq.append((nr, nc))
    return comp


def flood_from_border(passable: np.ndarray) -> np.ndarray:
    h, w = passable.shape
    seen = np.zeros_like(passable, dtype=bool)
    dq: deque[tuple[int, int]] = deque()
    for r in range(h):
        for c in (0, w - 1):
            if passable[r, c] and not seen[r, c]:
                seen[r, c] = True
                dq.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if passable[r, c] and not seen[r, c]:
                seen[r, c] = True
                dq.append((r, c))
    while dq:
        r, c = dq.popleft()
        for nr, nc in neighbors4(r, c, h, w):
            if passable[nr, nc] and not seen[nr, nc]:
                seen[nr, nc] = True
                dq.append((nr, nc))
    return seen


def boundary_from_region(region: np.ndarray) -> np.ndarray:
    h, w = region.shape
    boundary = np.zeros_like(region, dtype=bool)
    for r in range(h):
        for c in range(w):
            if not region[r, c]:
                continue
            for nr, nc in neighbors4(r, c, h, w):
                if not region[nr, nc]:
                    boundary[r, c] = True
                    break
            else:
                if r == 0 or c == 0 or r == h - 1 or c == w - 1:
                    boundary[r, c] = True
    return boundary


def is_simple_cycle(path_mask: np.ndarray) -> bool:
    comps = mask_components(path_mask)
    if len(comps) != 1:
        return False
    h, w = path_mask.shape
    for r, c in comps[0]:
        deg = sum(1 for nr, nc in neighbors4(r, c, h, w) if path_mask[nr, nc])
        if deg != 2:
            return False
    return True


def order_cycle(path_mask: np.ndarray) -> list[tuple[int, int]] | None:
    if not is_simple_cycle(path_mask):
        return None
    h, w = path_mask.shape
    points = [(int(r), int(c)) for r, c in np.argwhere(path_mask)]
    start = min(points)
    total = len(points)
    cycle = [start]
    prev = None
    curr = start
    while True:
        nexts = [
            (nr, nc)
            for nr, nc in neighbors4(curr[0], curr[1], h, w)
            if path_mask[nr, nc] and (nr, nc) != prev
        ]
        if not nexts:
            return None
        nxt = min(nexts)
        if nxt == start:
            return cycle if len(cycle) == total else None
        cycle.append(nxt)
        prev, curr = curr, nxt
        if len(cycle) > total:
            return None


def reverse_cycle(cycle: list[tuple[int, int]]) -> list[tuple[int, int]]:
    return [cycle[0]] + list(reversed(cycle[1:]))


def orientation_score(cycle: list[tuple[int, int]], interior_mask: np.ndarray) -> int:
    score = 0
    h, w = interior_mask.shape
    n = len(cycle)
    for i, (r, c) in enumerate(cycle):
        nr, nc = cycle[(i + 1) % n]
        direction = (nr - r, nc - c)
        rr, rc = right_of(direction)
        lr, lc = left_of(direction)
        if 0 <= r + rr < h and 0 <= c + rc < w and interior_mask[r + rr, c + rc]:
            score += 1
        if 0 <= r + lr < h and 0 <= c + lc < w and interior_mask[r + lr, c + lc]:
            score -= 1
    return score


def clockwise_cycle(cycle: list[tuple[int, int]], interior_mask: np.ndarray) -> list[tuple[int, int]]:
    rev = reverse_cycle(cycle)
    return cycle if orientation_score(cycle, interior_mask) >= orientation_score(rev, interior_mask) else rev


def follow_open_chain(component: set[tuple[int, int]], start: tuple[int, int], end: tuple[int, int]) -> list[tuple[int, int]]:
    if start == end:
        return [start]
    h = max(r for r, _ in component) + 1
    w = max(c for _, c in component) + 1
    chain = [start]
    prev = None
    curr = start
    while curr != end:
        nexts = [
            (nr, nc)
            for nr, nc in neighbors4(curr[0], curr[1], h, w)
            if (nr, nc) in component and (nr, nc) != prev
        ]
        if len(nexts) != 1:
            raise AssertionError("path component should be an open chain")
        nxt = nexts[0]
        chain.append(nxt)
        prev, curr = curr, nxt
        if len(chain) > len(component):
            raise AssertionError("chain traversal overflow")
    return chain


def enclosed_region(path_mask: np.ndarray) -> np.ndarray:
    passable = ~path_mask
    outside = flood_from_border(passable)
    return passable & (~outside)


def inward_anchor(cycle: list[tuple[int, int]], idx: int, interior_mask: np.ndarray) -> tuple[int, int] | None:
    h, w = interior_mask.shape
    n = len(cycle)
    prev = cycle[(idx - 1) % n]
    curr = cycle[idx]
    nxt = cycle[(idx + 1) % n]
    d_in = (curr[0] - prev[0], curr[1] - prev[1])
    d_out = (nxt[0] - curr[0], nxt[1] - curr[1])
    r_in = right_of(d_in)
    r_out = right_of(d_out)

    candidates: list[tuple[int, int]] = []
    if d_in == d_out:
        candidates.append((curr[0] + r_out[0], curr[1] + r_out[1]))
    else:
        candidates.append((curr[0] + r_in[0] + r_out[0], curr[1] + r_in[1] + r_out[1]))
        candidates.append((curr[0] + r_out[0], curr[1] + r_out[1]))
        candidates.append((curr[0] + r_in[0], curr[1] + r_in[1]))

    for rr, cc in candidates:
        if 0 <= rr < h and 0 <= cc < w and interior_mask[rr, cc]:
            return rr, cc

    vec = (r_in[0] + r_out[0], r_in[1] + r_out[1])
    best: tuple[tuple[int, int, int, int], tuple[int, int]] | None = None
    for rr, cc in neighbors8(curr[0], curr[1], h, w):
        if not interior_mask[rr, cc]:
            continue
        off = (rr - curr[0], cc - curr[1])
        dot = off[0] * vec[0] + off[1] * vec[1]
        dist = max(abs(off[0]), abs(off[1]))
        orth = abs(off[0] * vec[1] - off[1] * vec[0])
        key = (dot, -orth, -dist, -abs(off[0]) - abs(off[1]))
        if best is None or key > best[0]:
            best = (key, (rr, cc))
    return None if best is None else best[1]


def anchor_of(points: list[tuple[int, int]]) -> tuple[int, int]:
    return min(points)


def nearest_path_index(cycle: list[tuple[int, int]], anchor: tuple[int, int]) -> tuple[int, bool]:
    distances = [abs(r - anchor[0]) + abs(c - anchor[1]) for r, c in cycle]
    best_dist = min(distances)
    idxs = [i for i, d in enumerate(distances) if d == best_dist]
    best_idx = min(idxs, key=lambda i: (cycle[i][0], cycle[i][1], i))
    return best_idx, len(idxs) == 1


def extract_loops(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    loops: list[dict] = []
    for yr, yc in sorted(map(tuple, np.argwhere(g == YELLOW))):
        color_to_neighbors: dict[int, list[tuple[int, int]]] = {}
        for nr, nc in neighbors4(yr, yc, h, w):
            v = int(g[nr, nc])
            if v in (BG, YELLOW):
                continue
            color_to_neighbors.setdefault(v, []).append((nr, nc))
        path_color = None
        endpoints: list[tuple[int, int]] = []
        for color, pts in color_to_neighbors.items():
            if len(pts) >= 2:
                path_color = color
                endpoints = pts[:2]
                break
        if path_color is None:
            raise AssertionError("yellow marker must touch its path twice")

        comp = color_component(g, path_color, endpoints[0])
        if endpoints[1] not in comp:
            raise AssertionError("broken path component")

        chain = follow_open_chain(comp, endpoints[0], endpoints[1])
        path_mask = np.zeros((h, w), dtype=bool)
        for r, c in chain:
            path_mask[r, c] = True
        path_mask[yr, yc] = True
        interior_mask = enclosed_region(path_mask)
        cycle = clockwise_cycle([(yr, yc)] + chain, interior_mask)

        cargo_mask = interior_mask & (g != BG) & (~path_mask)
        cargo_comps = mask_components(cargo_mask)
        if not cargo_comps:
            raise AssertionError("each loop must contain a cargo sprite")
        cargo = max(cargo_comps, key=len)
        cargo_colors = {int(g[r, c]) for r, c in cargo}
        if len(cargo_colors) != 1:
            raise AssertionError("cargo must be monochrome")

        loops.append(
            {
                "path_color": path_color,
                "yellow": (yr, yc),
                "cycle": cycle,
                "path_mask": path_mask,
                "interior_mask": interior_mask,
                "cargo": cargo,
                "cargo_color": next(iter(cargo_colors)),
            }
        )
    return loops


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move each interior cargo along its enclosing loop by a number of steps equal to its area.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    loops = extract_loops(g)
    for info in loops:
        path_color = int(info["path_color"])
        yr, yc = info["yellow"]
        cycle = info["cycle"]
        interior_mask = info["interior_mask"]
        cargo = [(int(r), int(c)) for r, c in info["cargo"]]
        cargo_color = int(info["cargo_color"])

        out[yr, yc] = path_color

        anchor = anchor_of(cargo)
        start_idx, _unique = nearest_path_index(cycle, anchor)
        steps = len(cargo)
        target_idx = (start_idx + steps) % len(cycle)
        target_anchor = inward_anchor(cycle, target_idx, interior_mask)
        if target_anchor is None:
            raise AssertionError("target path pixel must have an interior anchor location")

        dr = target_anchor[0] - anchor[0]
        dc = target_anchor[1] - anchor[1]
        for r, c in cargo:
            out[r, c] = BG
        for r, c in cargo:
            rr, cc = r + dr, c + dc
            if not (0 <= rr < out.shape[0] and 0 <= cc < out.shape[1]):
                raise AssertionError("cargo translation left grid")
            out[rr, cc] = cargo_color

    return out


def normalize_cells(cells: set[tuple[int, int]]) -> list[tuple[int, int]]:
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    return sorted((r - min_r, c - min_c) for r, c in cells)


def random_polyomino(rng: random.Random, area: int) -> list[tuple[int, int]]:
    cells = {(0, 0)}
    while len(cells) < area:
        frontier = []
        for r, c in cells:
            for dr, dc in DIR4:
                cand = (r + dr, c + dc)
                if cand not in cells:
                    frontier.append(cand)
        cells.add(rng.choice(frontier))

    def transform(pt: tuple[int, int], rot: int, flip: bool) -> tuple[int, int]:
        r, c = pt
        if flip:
            c = -c
        for _ in range(rot):
            r, c = c, -r
        return r, c

    rot = rng.randrange(4)
    flip = bool(rng.randrange(2))
    transformed = {transform(pt, rot, flip) for pt in cells}
    return normalize_cells(transformed)


def crop_mask(mask: np.ndarray) -> np.ndarray:
    pts = np.argwhere(mask)
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    return mask[r0 : r1 + 1, c0 : c1 + 1]


def make_region(rng: random.Random) -> np.ndarray | None:
    canvas = np.zeros((24, 24), dtype=bool)
    base_h = rng.randint(5, 8)
    base_w = rng.randint(5, 8)
    r0 = rng.randint(7, 24 - base_h - 7)
    c0 = rng.randint(7, 24 - base_w - 7)
    canvas[r0 : r0 + base_h, c0 : c0 + base_w] = True

    sides = ["top", "bottom", "left", "right"]
    rng.shuffle(sides)
    bump_count = rng.choice([0, 1, 1, 2])
    for side in sides[:bump_count]:
        if side in ("top", "bottom"):
            max_span = base_w - 2
            if max_span < 2:
                continue
            span = rng.randint(2, max_span)
            start = rng.randint(c0 + 1, c0 + base_w - span - 1)
            depth = rng.randint(2, 4)
            if side == "top":
                rr0 = r0 - depth
                if rr0 < 1:
                    continue
                canvas[rr0 : r0 + 1, start : start + span] = True
            else:
                rr1 = r0 + base_h - 1 + depth
                if rr1 >= canvas.shape[0] - 1:
                    continue
                canvas[r0 + base_h - 1 : rr1 + 1, start : start + span] = True
        else:
            max_span = base_h - 2
            if max_span < 2:
                continue
            span = rng.randint(2, max_span)
            start = rng.randint(r0 + 1, r0 + base_h - span - 1)
            depth = rng.randint(2, 4)
            if side == "left":
                cc0 = c0 - depth
                if cc0 < 1:
                    continue
                canvas[start : start + span, cc0 : c0 + 1] = True
            else:
                cc1 = c0 + base_w - 1 + depth
                if cc1 >= canvas.shape[1] - 1:
                    continue
                canvas[start : start + span, c0 + base_w - 1 : cc1 + 1] = True

    region = crop_mask(canvas)
    boundary = boundary_from_region(region)
    interior = region & (~boundary)
    if interior.sum() < 6:
        return None
    if len(mask_components(interior)) != 1:
        return None
    if not is_simple_cycle(boundary):
        return None
    cycle = order_cycle(boundary)
    if cycle is None:
        return None
    cycle = clockwise_cycle(cycle, interior)
    direct_count = sum(inward_anchor(cycle, i, interior) is not None for i in range(len(cycle)))
    if direct_count < 10 or len(cycle) < 14:
        return None
    return region


def placed_cells(anchor: tuple[int, int], offsets: list[tuple[int, int]]) -> set[tuple[int, int]]:
    return {(anchor[0] + dr, anchor[1] + dc) for dr, dc in offsets}


def offsets_fit(anchor: tuple[int, int], offsets: list[tuple[int, int]], interior_mask: np.ndarray) -> bool:
    h, w = interior_mask.shape
    for dr, dc in offsets:
        rr, cc = anchor[0] + dr, anchor[1] + dc
        if not (0 <= rr < h and 0 <= cc < w and interior_mask[rr, cc]):
            return False
    return True


def make_loop_patch(rng: random.Random, path_color: int, cargo_color: int) -> np.ndarray | None:
    region = make_region(rng)
    if region is None:
        return None
    boundary = boundary_from_region(region)
    interior = region & (~boundary)
    cycle = clockwise_cycle(order_cycle(boundary), interior)  # type: ignore[arg-type]

    anchors = [(int(r), int(c)) for r, c in np.argwhere(interior)]
    rng.shuffle(anchors)
    area_choices = [1, 2, 2, 3, 3, 4, 4, 5]
    for _ in range(120):
        area = rng.choice(area_choices)
        offsets = random_polyomino(rng, area)
        candidates = []
        for anchor in anchors:
            if not offsets_fit(anchor, offsets, interior):
                continue
            start_idx, unique = nearest_path_index(cycle, anchor)
            if not unique:
                continue
            target_idx = (start_idx + area) % len(cycle)
            target_anchor = inward_anchor(cycle, target_idx, interior)
            if target_anchor is None:
                continue
            if not offsets_fit(target_anchor, offsets, interior):
                continue
            old_cells = placed_cells(anchor, offsets)
            new_cells = placed_cells(target_anchor, offsets)
            if old_cells == new_cells:
                continue
            overlap = len(old_cells & new_cells)
            if overlap == area:
                continue
            candidates.append((overlap, anchor, target_anchor, old_cells, new_cells, offsets))
        if candidates:
            candidates.sort(key=lambda item: (item[0], item[1], item[2]))
            _, anchor, target_anchor, old_cells, new_cells, offsets = candidates[0]
            patch = np.full(region.shape, BG, dtype=int)
            patch[boundary] = path_color
            for r, c in old_cells:
                patch[r, c] = cargo_color
            protected = old_cells | new_cells
            yellow_choices = [
                cycle[i]
                for i in range(len(cycle))
                if all(abs(cycle[i][0] - r) + abs(cycle[i][1] - c) > 1 for r, c in protected)
            ]
            yellow = rng.choice(yellow_choices if yellow_choices else cycle)
            patch[yellow] = YELLOW
            test_grid = np.pad(patch, 1, constant_values=BG)
            test_out = main(test_grid)
            if np.array_equal(test_out, test_grid):
                continue
            return patch
    return None


def can_place(occupied: np.ndarray, patch: np.ndarray, top: int, left: int, pad: int = 2) -> bool:
    h, w = occupied.shape
    ph, pw = patch.shape
    if top < 1 or left < 1 or top + ph >= h - 1 or left + pw >= w - 1:
        return False
    r0 = max(0, top - pad)
    r1 = min(h, top + ph + pad)
    c0 = max(0, left - pad)
    c1 = min(w, left + pw + pad)
    return not occupied[r0:r1, c0:c1].any()


def paste_patch(grid: np.ndarray, occupied: np.ndarray, patch: np.ndarray, top: int, left: int):
    ph, pw = patch.shape
    mask = patch != BG
    view = grid[top : top + ph, left : left + pw]
    view[mask] = patch[mask]
    occupied[top : top + ph, left : left + pw] |= mask


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(24, 38)
    w = rng.randint(24, 38)
    grid = np.full((h, w), BG, dtype=int)
    occupied = np.zeros((h, w), dtype=bool)

    object_count = rng.choice([1, 2, 3])
    color_pool = [1, 2, 3, 5, 6, 7, 8, 9]
    rng.shuffle(color_pool)
    if 2 * object_count > len(color_pool):
        return None

    patches = []
    for i in range(object_count):
        path_color = color_pool[2 * i]
        cargo_color = color_pool[2 * i + 1]
        patch = None
        for _ in range(80):
            patch = make_loop_patch(rng, path_color, cargo_color)
            if patch is not None:
                break
        if patch is None:
            return None
        patches.append(patch)

    rng.shuffle(patches)
    for patch in patches:
        ph, pw = patch.shape
        positions = [
            (top, left)
            for top in range(1, h - ph - 1)
            for left in range(1, w - pw - 1)
            if can_place(occupied, patch, top, left)
        ]
        if not positions:
            return None
        top, left = rng.choice(positions)
        paste_patch(grid, occupied, patch, top, left)

    out = main(grid)
    if out.shape != grid.shape:
        return None
    if np.any(out == YELLOW):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
