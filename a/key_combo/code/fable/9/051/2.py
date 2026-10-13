import random
from collections import defaultdict

import numpy as np


BLACK = 0
BLUE = 1
GRAY = 5
ORANGE = 7

SIDES = ("top", "bottom", "left", "right")
OPPOSITE = {"top": "bottom", "bottom": "top", "left": "right", "right": "left"}
DIR_TO_SIDE = {(-1, 0): "top", (1, 0): "bottom", (0, -1): "left", (0, 1): "right"}
LATTICE = (4, 11, 18, 25)
GEN_RETRIES = 400


def neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def components_mask(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp = []
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for nr, nc in neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rs = [r for r, _ in coords]
    cs = [c for _, c in coords]
    return min(rs), min(cs), max(rs), max(cs)


def _line_side(container_bbox: tuple[int, int, int, int], cell: tuple[int, int], other: tuple[int, int]) -> str:
    r0, c0, r1, c1 = container_bbox
    rr, cc = cell
    orr, occ = other
    if rr == r0 and orr == r0 - 1 and cc == occ:
        return "top"
    if rr == r1 and orr == r1 + 1 and cc == occ:
        return "bottom"
    if cc == c0 and occ == c0 - 1 and rr == orr:
        return "left"
    if cc == c1 and occ == c1 + 1 and rr == orr:
        return "right"
    raise AssertionError("invalid contact geometry")


def _stripe_len(container: dict, entry_side: str) -> int:
    return int(container["w"] if entry_side in ("top", "bottom") else container["h"])


def _stripe_count_capacity(container: dict, entry_side: str) -> int:
    return int(container["h"] if entry_side in ("top", "bottom") else container["w"])


def _fill_container(out: np.ndarray, container: dict, entry_side: str, amount: int) -> None:
    r0, c0, r1, c1 = container["bbox"]
    stripe_len = _stripe_len(container, entry_side)
    assert amount % stripe_len == 0
    stripes = amount // stripe_len
    if entry_side == "top":
        out[r0 : r0 + stripes, c0 : c1 + 1] = BLUE
    elif entry_side == "bottom":
        out[r1 - stripes + 1 : r1 + 1, c0 : c1 + 1] = BLUE
    elif entry_side == "left":
        out[r0 : r1 + 1, c0 : c0 + stripes] = BLUE
    else:
        assert entry_side == "right"
        out[r0 : r1 + 1, c1 - stripes + 1 : c1 + 1] = BLUE


def _parse_grid(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert max(g.shape) <= 30
    assert set(map(int, np.unique(g))) <= {BLACK, GRAY, ORANGE}

    orange = g == ORANGE
    gray = g == GRAY
    assert orange.any() and gray.any()
    assert len(components_mask(orange)) == 1, "pipes and containers must form one chain"

    core = np.zeros_like(orange, dtype=bool)
    h, w = g.shape
    for r in range(h):
        for c in range(w):
            if not orange[r, c]:
                continue
            has_h = (c > 0 and orange[r, c - 1]) or (c + 1 < w and orange[r, c + 1])
            has_v = (r > 0 and orange[r - 1, c]) or (r + 1 < h and orange[r + 1, c])
            core[r, c] = has_h and has_v

    container_comps = components_mask(core)
    assert len(container_comps) >= 2
    containers = []
    container_id = -np.ones_like(g, dtype=int)
    container_mask = np.zeros_like(orange, dtype=bool)
    for idx, comp in enumerate(container_comps):
        r0, c0, r1, c1 = bbox(comp)
        assert r1 > r0 and c1 > c0
        block = g[r0 : r1 + 1, c0 : c1 + 1]
        assert np.all(block == ORANGE), "containers must be solid rectangles"
        assert not container_mask[r0 : r1 + 1, c0 : c1 + 1].any()
        container_mask[r0 : r1 + 1, c0 : c1 + 1] = True
        container_id[r0 : r1 + 1, c0 : c1 + 1] = idx
        containers.append(
            {
                "id": idx,
                "bbox": (r0, c0, r1, c1),
                "h": r1 - r0 + 1,
                "w": c1 - c0 + 1,
                "area": (r1 - r0 + 1) * (c1 - c0 + 1),
            }
        )

    pipe_mask = orange & (~container_mask)
    pipe_comps = components_mask(pipe_mask)
    assert len(pipe_comps) == len(containers) - 1
    pipes = []
    adjacency: dict[int, list[dict]] = defaultdict(list)
    for pid, comp in enumerate(pipe_comps):
        rows = {r for r, _ in comp}
        cols = {c for _, c in comp}
        assert len(rows) == 1 or len(cols) == 1, "pipe must be straight"
        if len(rows) == 1:
            ordered = sorted(comp, key=lambda x: x[1])
            assert [c for _, c in ordered] == list(range(ordered[0][1], ordered[-1][1] + 1))
        else:
            ordered = sorted(comp, key=lambda x: x[0])
            assert [r for r, _ in ordered] == list(range(ordered[0][0], ordered[-1][0] + 1))

        touched: dict[int, set[str]] = defaultdict(set)
        for pr, pc in comp:
            for nr, nc in neighbors4(pr, pc):
                if 0 <= nr < h and 0 <= nc < w and container_id[nr, nc] >= 0:
                    cid = int(container_id[nr, nc])
                    side = _line_side(containers[cid]["bbox"], (nr, nc), (pr, pc))
                    touched[cid].add(side)
        assert len(touched) == 2
        items = []
        for cid, sides in touched.items():
            assert len(sides) == 1
            items.append((cid, next(iter(sides))))
        (a, side_a), (b, side_b) = sorted(items)
        pipe = {"id": pid, "coords": comp, "a": a, "b": b, "side_a": side_a, "side_b": side_b}
        pipes.append(pipe)
        adjacency[a].append({"neighbor": b, "pipe_id": pid, "side_self": side_a, "side_neighbor": side_b})
        adjacency[b].append({"neighbor": a, "pipe_id": pid, "side_self": side_b, "side_neighbor": side_a})

    gray_comps = components_mask(gray)
    assert len(gray_comps) == 1
    source = gray_comps[0]
    rows = {r for r, _ in source}
    cols = {c for _, c in source}
    assert len(rows) == 1 or len(cols) == 1
    if len(rows) == 1:
        ordered = sorted(source, key=lambda x: x[1])
        assert [c for _, c in ordered] == list(range(ordered[0][1], ordered[-1][1] + 1))
    else:
        ordered = sorted(source, key=lambda x: x[0])
        assert [r for r, _ in ordered] == list(range(ordered[0][0], ordered[-1][0] + 1))

    source_touch: dict[int, set[str]] = defaultdict(set)
    for sr, sc in source:
        for nr, nc in neighbors4(sr, sc):
            if 0 <= nr < h and 0 <= nc < w and container_id[nr, nc] >= 0:
                cid = int(container_id[nr, nc])
                side = _line_side(containers[cid]["bbox"], (nr, nc), (sr, sc))
                source_touch[cid].add(side)
    assert len(source_touch) == 1
    source_container, source_sides = next(iter(source_touch.items()))
    assert len(source_sides) == 1
    source_side = next(iter(source_sides))

    degrees = {c["id"]: len(adjacency[c["id"]]) for c in containers}
    endpoints = [cid for cid, deg in degrees.items() if deg == 1]
    assert len(endpoints) == 2 and all(deg in (1, 2) for deg in degrees.values())
    assert source_container in endpoints

    order = [source_container]
    pipe_order = []
    prev = None
    cur = source_container
    while True:
        options = [edge for edge in adjacency[cur] if edge["neighbor"] != prev]
        if not options:
            break
        assert len(options) == 1
        edge = options[0]
        pipe_order.append(edge["pipe_id"])
        prev, cur = cur, edge["neighbor"]
        order.append(cur)
    assert len(order) == len(containers)

    entry_sides = {order[0]: source_side}
    for i in range(1, len(order)):
        cur = order[i]
        prev_c = order[i - 1]
        match = [edge for edge in adjacency[cur] if edge["neighbor"] == prev_c]
        assert len(match) == 1
        entry_sides[cur] = match[0]["side_self"]

    return {
        "grid": g,
        "containers": containers,
        "pipes": pipes,
        "adjacency": adjacency,
        "order": order,
        "pipe_order": pipe_order,
        "entry_sides": entry_sides,
        "source": source,
        "source_container": source_container,
        "source_side": source_side,
    }


def _simulate(parsed: dict, amount: int) -> np.ndarray:
    g = parsed["grid"]
    out = g.copy()
    for r, c in parsed["source"]:
        out[r, c] = BLACK

    remaining = int(amount)
    for i, cid in enumerate(parsed["order"]):
        container = parsed["containers"][cid]
        entry_side = parsed["entry_sides"][cid]
        stripe_len = _stripe_len(container, entry_side)
        area = int(container["area"])
        fill_amount = min(remaining, area)
        assert fill_amount % stripe_len == 0
        _fill_container(out, container, entry_side, fill_amount)
        remaining -= fill_amount
        if remaining > 0:
            assert fill_amount == area and i < len(parsed["pipe_order"])
            for r, c in parsed["pipes"][parsed["pipe_order"][i]]["coords"]:
                out[r, c] = BLUE
    assert remaining == 0
    return out


def _feasible_amounts(order: list[int], containers: list[dict], entry_sides: dict[int, str]) -> list[int]:
    totals = []
    prefix = 0
    for cid in order:
        container = containers[cid]
        full_stripes = _stripe_count_capacity(container, entry_sides[cid])
        stripe_len = _stripe_len(container, entry_sides[cid])
        for stripes in range(1, full_stripes + 1):
            totals.append(prefix + stripes * stripe_len)
        prefix += int(container["area"])
    return sorted(set(totals))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Convert the gray source-bar count into blue liquid stripes that overflow through the orange pipe chain.

    Input structure:
    - Black background surrounds one connected orange chain made of several solid rectangular containers and one-cell-thick straight pipes.
    - The pipes connect containers edge-to-edge into a single linear chain, and a single one-cell-thick gray source bar 4-touches exactly one endpoint container.
    - Container capacity is its rectangle area; the source bar length is the total liquid amount.

    Rule:
    - Count the gray source pixels and erase the gray bar to background.
    - Starting from the touched side of the endpoint container, fill complete blue rows or columns from that entry side toward the opposite side.
    - When a container is completely full and liquid remains, repaint the full connecting pipe to the next container blue and continue filling the next container from the side where that pipe enters.

    Key insight:
    - The gray bar is only a conserved quantity; the visible result is determined by edge-oriented stripe filling inside rectangles plus cascade overflow along the chain.
    """
    parsed = _parse_grid(grid)
    return _simulate(parsed, len(parsed["source"]))


def _sample_walk(rng: random.Random, n: int, require_turn: bool) -> list[tuple[int, int]] | None:
    cells = [(r, c) for r in range(4) for c in range(4)]
    rng.shuffle(cells)
    for start in cells:
        walk = [start]
        used = {start}
        while len(walk) < n:
            r, c = walk[-1]
            nbrs = [(r + dr, c + dc) for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1))]
            nbrs = [(rr, cc) for rr, cc in nbrs if 0 <= rr < 4 and 0 <= cc < 4 and (rr, cc) not in used]
            rng.shuffle(nbrs)
            if not nbrs:
                break
            nxt = nbrs[0]
            walk.append(nxt)
            used.add(nxt)
        if len(walk) != n:
            continue
        dirs = [(walk[i + 1][0] - walk[i][0], walk[i + 1][1] - walk[i][1]) for i in range(n - 1)]
        has_turn = len({d for d in dirs}) > 1
        if require_turn and n >= 3 and not has_turn:
            continue
        return walk
    return None


def _container_from_lattice(lr: int, lc: int, h: int, w: int) -> dict:
    center_r = LATTICE[lr]
    center_c = LATTICE[lc]
    r0 = center_r - h // 2
    c0 = center_c - w // 2
    r1 = r0 + h - 1
    c1 = c0 + w - 1
    return {
        "bbox": (r0, c0, r1, c1),
        "h": h,
        "w": w,
        "area": h * w,
        "port_r": center_r,
        "port_c": center_c,
    }


def _paint_chain(base: np.ndarray, containers: list[dict], order: list[int]) -> list[list[tuple[int, int]]]:
    for container in containers:
        r0, c0, r1, c1 = container["bbox"]
        base[r0 : r1 + 1, c0 : c1 + 1] = ORANGE
    pipes = []
    for i in range(len(order) - 1):
        a = containers[order[i]]
        b = containers[order[i + 1]]
        ar0, ac0, ar1, ac1 = a["bbox"]
        br0, bc0, br1, bc1 = b["bbox"]
        coords = []
        if a["port_r"] == b["port_r"]:
            row = a["port_r"]
            if a["port_c"] < b["port_c"]:
                for c in range(ac1 + 1, bc0):
                    coords.append((row, c))
            else:
                for c in range(bc1 + 1, ac0):
                    coords.append((row, c))
        else:
            col = a["port_c"]
            if a["port_r"] < b["port_r"]:
                for r in range(ar1 + 1, br0):
                    coords.append((r, col))
            else:
                for r in range(br1 + 1, ar0):
                    coords.append((r, col))
        assert coords
        for r, c in coords:
            assert base[r, c] == BLACK
            base[r, c] = ORANGE
        pipes.append(coords)
    return pipes


def _direction(a: tuple[int, int], b: tuple[int, int]) -> tuple[int, int]:
    dr = b[0] - a[0]
    dc = b[1] - a[1]
    return (0 if dr == 0 else dr // abs(dr), 0 if dc == 0 else dc // abs(dc))


def _find_source_placement(grid: np.ndarray, target_bbox: tuple[int, int, int, int], side: str, length: int):
    h, w = grid.shape
    r0, c0, r1, c1 = target_bbox
    if side == "top":
        row = r0 - 1
        if row < 0:
            return None
        starts = range(0, w - length + 1)
        fixed = lambda s: [(row, c) for c in range(s, s + length)]
    elif side == "bottom":
        row = r1 + 1
        if row >= h:
            return None
        starts = range(0, w - length + 1)
        fixed = lambda s: [(row, c) for c in range(s, s + length)]
    elif side == "left":
        col = c0 - 1
        if col < 0:
            return None
        starts = range(0, h - length + 1)
        fixed = lambda s: [(r, col) for r in range(s, s + length)]
    else:
        assert side == "right"
        col = c1 + 1
        if col >= w:
            return None
        starts = range(0, h - length + 1)
        fixed = lambda s: [(r, col) for r in range(s, s + length)]

    for start in starts:
        coords = fixed(start)
        if any(grid[r, c] != BLACK for r, c in coords):
            continue
        overlaps = []
        other_touches = False
        for sr, sc in coords:
            for nr, nc in neighbors4(sr, sc):
                if 0 <= nr < h and 0 <= nc < w and grid[nr, nc] == ORANGE:
                    if r0 <= nr <= r1 and c0 <= nc <= c1:
                        overlaps.append((nr, nc))
                    else:
                        other_touches = True
        if other_touches or not overlaps:
            continue
        return coords
    return None


def _crop_with_margin(grid: np.ndarray, rng: random.Random) -> np.ndarray:
    coords = np.argwhere(grid != BLACK)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    top = rng.randint(0, min(2, r0))
    left = rng.randint(0, min(2, c0))
    bottom = rng.randint(0, min(2, grid.shape[0] - 1 - r1))
    right = rng.randint(0, min(2, grid.shape[1] - 1 - c1))
    return grid[r0 - top : r1 + bottom + 1, c0 - left : c1 + right + 1].copy()


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    n = rng.choices([2, 3, 4], weights=[2, 3, 2])[0]
    require_turn = n >= 3 and rng.random() < 0.65
    walk = _sample_walk(rng, n, require_turn)
    if walk is None:
        return None
    if rng.random() < 0.5:
        walk = list(reversed(walk))

    containers = []
    for lr, lc in walk:
        h = rng.randint(2, 4)
        w = rng.randint(2, 4)
        containers.append(_container_from_lattice(lr, lc, h, w))

    base = np.full((30, 30), BLACK, dtype=int)
    try:
        _paint_chain(base, containers, list(range(n)))
    except AssertionError:
        return None

    dirs = [_direction(walk[i], walk[i + 1]) for i in range(n - 1)]
    pipe_sides = [DIR_TO_SIDE[d] for d in dirs]
    source_candidates = [s for s in SIDES if s != pipe_sides[0]]
    rng.shuffle(source_candidates)

    side_choice = None
    amount_choice = None
    source_coords = None
    order = list(range(n))
    for source_side in source_candidates:
        entry_sides = {0: source_side}
        for i in range(1, n):
            entry_sides[i] = OPPOSITE[pipe_sides[i - 1]]
        feasible = _feasible_amounts(order, containers, entry_sides)
        assert len(feasible) >= n

        full_prefixes = [sum(containers[k]["area"] for k in range(i + 1)) for i in range(n)]
        candidates = []
        for amount in feasible:
            if amount in full_prefixes[:-1]:
                score = 3
            elif amount > containers[0]["area"]:
                score = 2
            else:
                score = 1
            candidates.append((score, amount))
        rng.shuffle(candidates)
        candidates.sort(reverse=True)

        for _score, amount in candidates:
            coords = _find_source_placement(base, containers[0]["bbox"], source_side, amount)
            if coords is not None:
                side_choice = source_side
                amount_choice = amount
                source_coords = coords
                break
        if side_choice is not None:
            break
    if side_choice is None:
        return None

    grid = base.copy()
    for r, c in source_coords:
        grid[r, c] = GRAY

    cropped = _crop_with_margin(grid, rng)
    parsed = _parse_grid(cropped)
    if len(parsed["source"]) != amount_choice:
        return None

    feasible = _feasible_amounts(parsed["order"], parsed["containers"], parsed["entry_sides"])
    signatures = set()
    for amount in feasible:
        out = _simulate(parsed, amount)
        sig = (out.shape, out.tobytes())
        if sig in signatures:
            return None
        signatures.add(sig)

    out = _simulate(parsed, amount_choice)
    if np.array_equal(out, cropped):
        return None
    if BLUE not in out:
        return None

    overflow_cells = np.count_nonzero(out == BLUE) - amount_choice
    if amount_choice > parsed["containers"][parsed["order"][0]]["area"] and overflow_cells <= 0:
        return None
    if amount_choice <= parsed["containers"][parsed["order"][0]]["area"] and overflow_cells != 0:
        return None

    if not np.array_equal(main(cropped), out):
        return None
    return cropped


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_RETRIES} attempts")
