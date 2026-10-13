from __future__ import annotations

from collections import deque

import numpy as np


BG = 0
GREY = 5
TEAL = 8
PALETTE = (1, 2, 3, 4, 6, 7, 9)
DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))
MACRO = 5
MAX_ATTEMPTS = 400


def _get_bg(grid: np.ndarray) -> int:
    vals = np.asarray(grid, dtype=int).ravel()
    return int(np.bincount(vals).argmax())


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= int(r) < int(shape[0]) and 0 <= int(c) < int(shape[1])


def _neighbors4(r: int, c: int):
    yield int(r) - 1, int(c)
    yield int(r) + 1, int(c)
    yield int(r), int(c) - 1
    yield int(r), int(c) + 1


def _components(mask: np.ndarray) -> tuple[np.ndarray, list[list[tuple[int, int]]]]:
    mask = np.asarray(mask, dtype=bool)
    labels = np.full(mask.shape, -1, dtype=int)
    comps: list[list[tuple[int, int]]] = []
    next_id = 0
    for r, c in np.argwhere(mask):
        rr = int(r)
        cc = int(c)
        if labels[rr, cc] >= 0:
            continue
        labels[rr, cc] = next_id
        q = deque([(rr, cc)])
        coords: list[tuple[int, int]] = []
        while q:
            cr, cc2 = q.popleft()
            coords.append((cr, cc2))
            for nr, nc in _neighbors4(cr, cc2):
                if not _in_bounds(mask.shape, nr, nc):
                    continue
                if not mask[nr, nc] or labels[nr, nc] >= 0:
                    continue
                labels[nr, nc] = next_id
                q.append((nr, nc))
        comps.append(coords)
        next_id += 1
    return labels, comps


def _port_entry(direction: tuple[int, int], lane: int) -> tuple[int, int]:
    dr, dc = direction
    if (dr, dc) == (0, 1):
        return int(lane), 1
    if (dr, dc) == (0, -1):
        return int(lane), 3
    if (dr, dc) == (1, 0):
        return 1, int(lane)
    if (dr, dc) == (-1, 0):
        return 3, int(lane)
    raise ValueError("bad direction")


def _port_exit(direction: tuple[int, int], lane: int) -> tuple[int, int]:
    dr, dc = direction
    if (dr, dc) == (0, 1):
        return int(lane), 3
    if (dr, dc) == (0, -1):
        return int(lane), 1
    if (dr, dc) == (1, 0):
        return 3, int(lane)
    if (dr, dc) == (-1, 0):
        return 1, int(lane)
    raise ValueError("bad direction")


def _teal_local(direction: tuple[int, int], lane: int) -> tuple[int, int]:
    dr, dc = direction
    if (dr, dc) == (0, 1):
        return int(lane), 4
    if (dr, dc) == (0, -1):
        return int(lane), 0
    if (dr, dc) == (1, 0):
        return 4, int(lane)
    if (dr, dc) == (-1, 0):
        return 0, int(lane)
    raise ValueError("bad direction")


def _source_local(direction: tuple[int, int], lane: int) -> tuple[tuple[int, int], tuple[int, int]]:
    dr, dc = direction
    if (dr, dc) == (0, 1):
        return (int(lane), 1), (int(lane), 2)
    if (dr, dc) == (0, -1):
        return (int(lane), 3), (int(lane), 2)
    if (dr, dc) == (1, 0):
        return (1, int(lane)), (2, int(lane))
    if (dr, dc) == (-1, 0):
        return (3, int(lane)), (2, int(lane))
    raise ValueError("bad direction")


def _macro_base(cell: tuple[int, int]) -> tuple[int, int]:
    return int(cell[0]) * MACRO, int(cell[1]) * MACRO


def _abs_cell(cell: tuple[int, int], local: tuple[int, int]) -> tuple[int, int]:
    br, bc = _macro_base(cell)
    return br + int(local[0]), bc + int(local[1])


def _paint_region(grid: np.ndarray, cell: tuple[int, int]) -> list[tuple[int, int]]:
    br, bc = _macro_base(cell)
    coords = []
    for lr in range(1, 4):
        for lc in range(1, 4):
            rr = br + lr
            cc = bc + lc
            grid[rr, cc] = GREY
            coords.append((rr, cc))
    return coords


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor grey regions by following color rays fired through teal muzzles, then erase the firing devices.

    Input structure
    - The grid has a black background, several grey 4-connected mask regions, several teal single pixels, and some non-grey colored single pixels.
    - A colored single pixel orthogonally adjacent to a teal pixel is an initial cannon; some grey regions also have exactly one adjacent teal pixel ready to act as a relay muzzle after the region is colored.
    - Additional isolated colored single pixels may appear as distractors and are not adjacent to teal.
    Rule
    - For every colored object that has an adjacent teal pixel, take the direction from the object toward that teal pixel and shoot a ray forward from the teal cell through black cells.
    - The first grey region hit by that ray is recolored completely to the shooter's color; if that newly colored region has its own teal muzzle, it immediately fires again in its own encoded direction with the same color.
    - After all relays finish, erase every teal pixel and every initial colored source pixel; untouched grey regions and isolated colored distractors stay as they are.
    Key insight
    - Teal pixels are reusable direction markers rather than colored objects: color travels remotely along a chain of object-to-muzzle directions, and only the grey targets persist in the final scene.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _get_bg(g)
    assert bg == BG, "expected black background"

    grey_labels, grey_components = _components(g == GREY)
    out = g.copy()
    out[g == TEAL] = bg

    relay_by_region: dict[int, tuple[tuple[int, int], tuple[int, int]]] = {}
    initial_queue: list[tuple[int, tuple[int, int], tuple[int, int]]] = []
    source_cells: set[tuple[int, int]] = set()

    for tr, tc in np.argwhere(g == TEAL):
        teal = (int(tr), int(tc))
        touching = [
            (nr, nc)
            for nr, nc in _neighbors4(*teal)
            if _in_bounds(g.shape, nr, nc) and int(g[nr, nc]) not in (bg, TEAL)
        ]
        assert len(touching) == 1, "each teal muzzle must touch exactly one non-black object cell"
        obj_r, obj_c = touching[0]
        direction = (teal[0] - obj_r, teal[1] - obj_c)
        value = int(g[obj_r, obj_c])
        if value == GREY:
            region_id = int(grey_labels[obj_r, obj_c])
            assert region_id >= 0 and region_id not in relay_by_region, "grey regions may have at most one teal muzzle"
            relay_by_region[region_id] = (teal, direction)
        else:
            same_color_neighbors = sum(
                1
                for nr, nc in _neighbors4(obj_r, obj_c)
                if _in_bounds(g.shape, nr, nc) and int(g[nr, nc]) == value
            )
            assert same_color_neighbors == 0, "initial emitters must be single colored pixels"
            initial_queue.append((value, teal, direction))
            source_cells.add((obj_r, obj_c))

    for sr, sc in source_cells:
        out[sr, sc] = bg

    recolored: set[int] = set()
    queue = deque(sorted(initial_queue, key=lambda item: (item[1][0], item[1][1], item[0])))
    while queue:
        color, teal, direction = queue.popleft()
        dr, dc = direction
        r = teal[0] + dr
        c = teal[1] + dc
        while _in_bounds(g.shape, r, c):
            cell = int(g[r, c])
            if cell == bg:
                r += dr
                c += dc
                continue
            if cell == GREY:
                region_id = int(grey_labels[r, c])
                if region_id not in recolored:
                    recolored.add(region_id)
                    for rr, cc in grey_components[region_id]:
                        out[rr, cc] = int(color)
                    if region_id in relay_by_region:
                        next_teal, next_dir = relay_by_region[region_id]
                        queue.append((int(color), next_teal, next_dir))
                break
            break

    return out


def _shuffle(seq: list, rng: np.random.Generator) -> list:
    seq = list(seq)
    rng.shuffle(seq)
    return seq


def _free_macro_cells(mh: int, mw: int, blocked: set[tuple[int, int]]) -> list[tuple[int, int]]:
    return [(r, c) for r in range(mh) for c in range(mw) if (r, c) not in blocked]


def _build_path(
    mh: int,
    mw: int,
    region_count: int,
    blocked: set[tuple[int, int]],
    rng: np.random.Generator,
    require_turn: bool,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]], list[int]] | None:
    starts = _shuffle(_free_macro_cells(mh, mw, blocked), rng)
    max_starts = min(30, len(starts))

    def dfs(
        path: list[tuple[int, int]],
        dirs: list[tuple[int, int]],
        spans: list[int],
        used: set[tuple[int, int]],
    ):
        if len(path) == region_count + 1:
            if not require_turn or len({tuple(d) for d in dirs}) > 1:
                return path, dirs, spans
            return None

        cur = path[-1]
        options = []
        for direction in DIRS:
            for span in (1, 2):
                cells = []
                ok = True
                for step in range(1, span + 1):
                    nr = cur[0] + direction[0] * step
                    nc = cur[1] + direction[1] * step
                    cell = (int(nr), int(nc))
                    if not (0 <= cell[0] < mh and 0 <= cell[1] < mw):
                        ok = False
                        break
                    if cell in used:
                        ok = False
                        break
                    cells.append(cell)
                if ok:
                    options.append((direction, span, cells))
        options = _shuffle(options, rng)
        for direction, span, cells in options:
            result = dfs(
                path + [cells[-1]],
                dirs + [direction],
                spans + [int(span)],
                used | set(cells),
            )
            if result is not None:
                return result
        return None

    for start in starts[:max_starts]:
        result = dfs([start], [], [], set(blocked) | {start})
        if result is not None:
            return result
    return None


def _reserve_path(
    nodes: list[tuple[int, int]],
    dirs: list[tuple[int, int]],
    spans: list[int],
    used_nodes: set[tuple[int, int]],
    reserved: set[tuple[int, int]],
) -> None:
    used_nodes.update(nodes)
    for node, direction, span in zip(nodes[:-1], dirs, spans, strict=True):
        for step in range(1, int(span)):
            reserved.add((node[0] + direction[0] * step, node[1] + direction[1] * step))


def _try_shadow(
    chain: dict,
    mh: int,
    mw: int,
    used_nodes: set[tuple[int, int]],
    reserved: set[tuple[int, int]],
    rng: np.random.Generator,
) -> tuple[int, int] | None:
    terminal = chain["nodes"][-1]
    direction = chain["dirs"][-1]
    for span in _shuffle([1, 2], rng):
        cells = []
        ok = True
        for step in range(1, span + 1):
            nr = terminal[0] + direction[0] * step
            nc = terminal[1] + direction[1] * step
            cell = (int(nr), int(nc))
            if not (0 <= cell[0] < mh and 0 <= cell[1] < mw):
                ok = False
                break
            if cell in used_nodes or cell in reserved:
                ok = False
                break
            cells.append(cell)
        if not ok:
            continue
        used_nodes.add(cells[-1])
        for mid in cells[:-1]:
            reserved.add(mid)
        return cells[-1]
    return None


def _pick_lanes(dirs: list[tuple[int, int]], rng: np.random.Generator) -> list[int] | None:
    lanes = [int(rng.integers(1, 4))]
    for idx in range(1, len(dirs)):
        prev_dir = dirs[idx - 1]
        cur_dir = dirs[idx]
        prev_lane = lanes[idx - 1]
        choices = [
            lane
            for lane in (1, 2, 3)
            if _port_entry(prev_dir, prev_lane) != _port_exit(cur_dir, lane)
        ]
        if not choices:
            return None
        lanes.append(int(rng.choice(choices)))
    return lanes


def _render_scene(mh: int, mw: int, plan: dict) -> tuple[np.ndarray, dict]:
    grid = np.zeros((mh * MACRO, mw * MACRO), dtype=int)
    meta = {"chains": [], "dormant_regions": [], "distractors": [], "teals": [], "sources": [], "shadows": []}

    for chain in plan["chains"]:
        color = int(chain["color"])
        nodes = chain["nodes"]
        dirs = chain["dirs"]
        lanes = chain["lanes"]

        source_local, teal_local = _source_local(dirs[0], lanes[0])
        source = _abs_cell(nodes[0], source_local)
        source_teal = _abs_cell(nodes[0], teal_local)
        grid[source] = color
        grid[source_teal] = TEAL

        chain_meta = {
            "color": color,
            "source": source,
            "source_teal": source_teal,
            "regions": [],
            "edges": [],
            "dirs": list(dirs),
            "relay_teals": [],
        }
        meta["sources"].append(source)
        meta["teals"].append(source_teal)

        for idx, region_cell in enumerate(nodes[1:], start=1):
            region_coords = _paint_region(grid, region_cell)
            chain_meta["regions"].append(region_coords)

            origin_teal = source_teal if idx == 1 else chain_meta["relay_teals"][idx - 2]
            chain_meta["edges"].append(
                {
                    "teal": origin_teal,
                    "direction": dirs[idx - 1],
                    "target": region_coords,
                }
            )

            if idx < len(nodes) - 1:
                teal = _abs_cell(region_cell, _teal_local(dirs[idx], lanes[idx]))
                grid[teal] = TEAL
                meta["teals"].append(teal)
                chain_meta["relay_teals"].append(teal)

        if chain.get("shadow") is not None:
            shadow = _paint_region(grid, chain["shadow"])
            meta["shadows"].append(shadow)
        meta["chains"].append(chain_meta)

    for dormant in plan["dormant_regions"]:
        region = _paint_region(grid, dormant["cell"])
        teal = _abs_cell(dormant["cell"], _teal_local(dormant["direction"], dormant["lane"]))
        grid[teal] = TEAL
        meta["dormant_regions"].append({"coords": region, "teal": teal})
        meta["teals"].append(teal)

    for distractor in plan["distractors"]:
        cell = _abs_cell(distractor["cell"], (2, 2))
        grid[cell] = int(distractor["color"])
        meta["distractors"].append(cell)

    return grid, meta


def _expected_output(grid: np.ndarray, meta: dict) -> np.ndarray:
    out = np.array(grid, copy=True)
    for r, c in meta["teals"]:
        out[r, c] = BG
    for r, c in meta["sources"]:
        out[r, c] = BG
    for chain in meta["chains"]:
        color = int(chain["color"])
        for region in chain["regions"]:
            for r, c in region:
                out[r, c] = color
    return out


def _first_non_black(grid: np.ndarray, teal: tuple[int, int], direction: tuple[int, int]):
    dr, dc = direction
    r = teal[0] + dr
    c = teal[1] + dc
    traversed = []
    while _in_bounds(grid.shape, r, c):
        traversed.append((r, c))
        if int(grid[r, c]) != BG:
            return (r, c), traversed
        r += dr
        c += dc
    return None, traversed


def _semantic_ok(grid: np.ndarray, meta: dict) -> bool:
    if not meta["chains"] or not meta["dormant_regions"] or not meta["distractors"]:
        return False
    if max(len(chain["regions"]) for chain in meta["chains"]) < 2:
        return False
    if not any(len({tuple(d) for d in chain["dirs"]}) > 1 for chain in meta["chains"] if len(chain["dirs"]) > 1):
        return False

    for teal in meta["teals"]:
        touching = [
            (nr, nc)
            for nr, nc in _neighbors4(*teal)
            if _in_bounds(grid.shape, nr, nc) and int(grid[nr, nc]) not in (BG, TEAL)
        ]
        if len(touching) != 1:
            return False

    for dr, dc in meta["distractors"]:
        if any(_in_bounds(grid.shape, nr, nc) and int(grid[nr, nc]) == TEAL for nr, nc in _neighbors4(dr, dc)):
            return False

    incoming_targets = set()
    for chain in meta["chains"]:
        for edge in chain["edges"]:
            target = set(edge["target"])
            first_hit, traversed = _first_non_black(grid, edge["teal"], edge["direction"])
            if first_hit is None or first_hit not in target:
                return False
            if any(cell != first_hit and int(grid[cell]) != BG for cell in traversed[: traversed.index(first_hit)]):
                return False
            frozen = tuple(sorted(target))
            if frozen in incoming_targets:
                return False
            incoming_targets.add(frozen)

    expected = _expected_output(grid, meta)
    try:
        actual = main(grid)
    except AssertionError:
        return False
    return bool(np.array_equal(actual, expected) and not np.array_equal(actual, grid))


def _build_plan(rng: np.random.Generator) -> tuple[int, int, dict] | None:
    mh = int(rng.integers(5, 7))
    mw = int(rng.integers(5, 7))

    n_chains = int(rng.integers(2, 4))
    lengths = [int(rng.integers(2, 4))]
    while len(lengths) < n_chains:
        lengths.append(int(rng.integers(1, 3)))

    chain_colors = [int(c) for c in rng.choice(PALETTE, size=n_chains, replace=False)]
    used_nodes: set[tuple[int, int]] = set()
    reserved: set[tuple[int, int]] = set()
    chains = []

    for idx, length in enumerate(lengths):
        path = _build_path(mh, mw, length, used_nodes | reserved, rng, require_turn=(idx == 0))
        if path is None:
            return None
        nodes, dirs, spans = path
        lanes = _pick_lanes(dirs, rng)
        if lanes is None:
            return None
        _reserve_path(nodes, dirs, spans, used_nodes, reserved)
        chains.append({"color": chain_colors[idx], "nodes": nodes, "dirs": dirs, "spans": spans, "lanes": lanes})

    shadow_hosts = [chain for chain in chains if len(chain["dirs"]) >= 1]
    rng.shuffle(shadow_hosts)
    placed_shadow = False
    for chain in shadow_hosts:
        if rng.random() < 0.7:
            shadow = _try_shadow(chain, mh, mw, used_nodes, reserved, rng)
            if shadow is not None:
                chain["shadow"] = shadow
                placed_shadow = True
                break
    if not placed_shadow:
        chains[0]["shadow"] = None

    dormant_regions = []
    free = _free_macro_cells(mh, mw, used_nodes | reserved)
    if not free:
        return None
    cell = free[int(rng.integers(len(free)))]
    used_nodes.add(cell)
    dormant_regions.append(
        {
            "cell": cell,
            "direction": DIRS[int(rng.integers(len(DIRS)))],
            "lane": int(rng.integers(1, 4)),
        }
    )

    distractors = []
    n_distractors = int(rng.integers(1, 3))
    for _ in range(n_distractors):
        free = _free_macro_cells(mh, mw, used_nodes | reserved)
        if not free:
            return None
        cell = free[int(rng.integers(len(free)))]
        used_nodes.add(cell)
        distractors.append({"cell": cell, "color": int(rng.choice(PALETTE))})

    return mh, mw, {"chains": chains, "dormant_regions": dormant_regions, "distractors": distractors}


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        rng = np.random.default_rng()
        built = _build_plan(rng)
        if built is None:
            continue
        mh, mw, plan = built
        grid, meta = _render_scene(mh, mw, plan)
        if _semantic_ok(grid, meta):
            return grid
    raise RuntimeError("failed to generate a valid chain-relay cannon input")
