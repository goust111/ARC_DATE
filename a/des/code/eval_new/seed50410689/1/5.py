from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
WALL = 5
VALVE = 9
SOURCE_COLORS = (1, 2, 3, 4, 6, 7, 8)

MASTER_SIZE = 30
ROW_TOPS = (1, 12)
COL_LEFTS = (1, 11, 21)
BOX_SIZE_RANGE = (5, 8)
GEN_RETRY_LIMIT = 500


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    out: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                coords.append((int(cr), int(cc)))
                for nr, nc in _neighbors4(int(cr), int(cc)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            out.append(np.array(coords, dtype=int))
    return out


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _touches_boundary(coords: np.ndarray, shape: tuple[int, int]) -> bool:
    h, w = shape
    return bool(
        np.any(coords[:, 0] == 0)
        or np.any(coords[:, 0] == h - 1)
        or np.any(coords[:, 1] == 0)
        or np.any(coords[:, 1] == w - 1)
    )


def _perimeter_cells(box: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = box
    cells: list[tuple[int, int]] = []
    for c in range(c0, c1 + 1):
        cells.append((r0, c))
        if r1 != r0:
            cells.append((r1, c))
    for r in range(r0 + 1, r1):
        cells.append((r, c0))
        if c1 != c0:
            cells.append((r, c1))
    return cells


def _container_source_color(
    grid: np.ndarray, interior_box: tuple[int, int, int, int]
) -> int | None:
    r0, c0, r1, c1 = interior_box
    patch = grid[r0 : r1 + 1, c0 : c1 + 1]
    colors = {int(v) for v in np.unique(patch) if int(v) != BG}
    if not colors:
        return None
    assert len(colors) == 1, "source container interior must use one color"
    color = next(iter(colors))
    assert color not in {WALL, VALVE}, "source color must differ from wall and valve"
    assert np.all(patch[:-1, :] == BG), "only the bottom interior row may be prefilled"
    assert np.all(patch[-1, :] == color), "source fill must cover the full bottom row"
    return color


def _detect_containers(grid: np.ndarray) -> list[dict]:
    free_mask = (grid != WALL) & (grid != VALVE)
    containers: list[dict] = []
    for coords in _components(free_mask):
        if _touches_boundary(coords, grid.shape):
            continue
        ir0, ic0, ir1, ic1 = _bbox(coords)
        area = int((ir1 - ir0 + 1) * (ic1 - ic0 + 1))
        if int(coords.shape[0]) != area:
            continue
        if ir0 == 0 or ic0 == 0 or ir1 == grid.shape[0] - 1 or ic1 == grid.shape[1] - 1:
            continue
        wall_box = (ir0 - 1, ic0 - 1, ir1 + 1, ic1 + 1)
        wr0, wc0, wr1, wc1 = wall_box
        if not np.all(grid[wr0, wc0 : wc1 + 1] == WALL):
            continue
        if not np.all(grid[wr1, wc0 : wc1 + 1] == WALL):
            continue
        if not np.all(grid[wr0 : wr1 + 1, wc0] == WALL):
            continue
        if not np.all(grid[wr0 : wr1 + 1, wc1] == WALL):
            continue
        containers.append(
            {
                "interior_box": (ir0, ic0, ir1, ic1),
                "wall_box": wall_box,
                "source_color": _container_source_color(grid, (ir0, ic0, ir1, ic1)),
            }
        )

    containers.sort(key=lambda item: item["wall_box"])
    for idx, container in enumerate(containers):
        container["id"] = idx
    assert containers, "expected at least one container"
    return containers


def _border_owner(containers: list[dict]) -> dict[tuple[int, int], int]:
    owners: dict[tuple[int, int], int] = {}
    for container in containers:
        cid = int(container["id"])
        for cell in _perimeter_cells(container["wall_box"]):
            assert cell not in owners, "container walls must not overlap"
            owners[cell] = cid
    return owners


def _pipe_starts(container: dict) -> list[tuple[int, int, int, int]]:
    r0, c0, r1, c1 = container["wall_box"]
    starts: list[tuple[int, int, int, int]] = []
    for c in range(c0 + 1, c1):
        starts.append((r0, c, -1, 0))
        starts.append((r1, c, 1, 0))
    for r in range(r0 + 1, r1):
        starts.append((r, c0, 0, -1))
        starts.append((r, c1, 0, 1))
    return starts


def _trace_pipe(
    grid: np.ndarray,
    source_id: int,
    start: tuple[int, int, int, int],
    owners: dict[tuple[int, int], int],
) -> tuple[int, bool] | None:
    r, c, dr, dc = start
    cr, cc = r + dr, c + dc
    if not (0 <= cr < grid.shape[0] and 0 <= cc < grid.shape[1]):
        return None
    if (cr, cc) in owners:
        return None
    if int(grid[cr, cc]) not in {WALL, VALVE}:
        return None

    blocked = False
    while True:
        if (cr, cc) in owners:
            target = int(owners[(cr, cc)])
            if target == source_id:
                return None
            return target, blocked
        value = int(grid[cr, cc])
        if value not in {WALL, VALVE}:
            return None
        if value == VALVE:
            blocked = True
        cr += dr
        cc += dc
        if not (0 <= cr < grid.shape[0] and 0 <= cc < grid.shape[1]):
            return None


def _find_edges(grid: np.ndarray, containers: list[dict]) -> dict[tuple[int, int], bool]:
    owners = _border_owner(containers)
    edges: dict[tuple[int, int], bool] = {}
    for container in containers:
        cid = int(container["id"])
        for start in _pipe_starts(container):
            traced = _trace_pipe(grid, cid, start, owners)
            if traced is None:
                continue
            target, blocked = traced
            key = tuple(sorted((cid, int(target))))
            if key in edges:
                edges[key] = edges[key] and blocked
            else:
                edges[key] = blocked
    return edges


def _distances(
    node_count: int, edges: dict[tuple[int, int], bool], source_id: int
) -> list[int | None]:
    adj = [[] for _ in range(node_count)]
    for (a, b), blocked in edges.items():
        if blocked:
            continue
        adj[int(a)].append(int(b))
        adj[int(b)].append(int(a))

    dist: list[int | None] = [None] * node_count
    dist[source_id] = 0
    q: deque[int] = deque([source_id])
    while q:
        cur = q.popleft()
        for nxt in adj[cur]:
            if dist[nxt] is None:
                dist[nxt] = int(dist[cur]) + 1
                q.append(nxt)
    return dist


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each container according to which colored liquid sources can reach it through open pipes.

    Input structure:
    - The background is black.
    - Grey hollow rectangles form containers whose interiors are initially black,
      except that some source containers have their full interior bottom row
      filled with one non-grey color.
    - Grey one-pixel horizontal or vertical pipe segments connect container walls.
    - A maroon pixel on a pipe marks a closed valve that blocks flow through that pipe.

    Rule:
    - Treat each container as a graph node and each pipe as one graph edge; pipes
      containing a maroon valve are blocked, while plain grey pipes are traversable.
    - From every source container, measure open-pipe distances in pipe segments to
      every reachable container.
    - A non-source container reachable from exactly one source is filled across its
      whole interior with that source color. A non-source container reachable from
      multiple sources gets only its bottom interior row filled with the unique
      nearest source color, and stays empty if the nearest distance is tied.
      Source containers, grey walls, pipes, and valves remain unchanged.

    Key insight:
    - The puzzle is a shortest-path competition on a valve-pruned container graph:
      first infer connectivity through open pipes, then distinguish single-source,
      nearest-source, and tied-source containers by how much of the interior is filled.
    """
    g = np.asarray(grid, dtype=int)
    containers = _detect_containers(g)
    edges = _find_edges(g, containers)

    sources = [container for container in containers if container["source_color"] is not None]
    assert sources, "expected at least one colored source container"

    source_dists = {
        int(source["id"]): _distances(len(containers), edges, int(source["id"]))
        for source in sources
    }

    out = g.copy()
    for container in containers:
        if container["source_color"] is None:
            r0, c0, r1, c1 = container["interior_box"]
            out[r0 : r1 + 1, c0 : c1 + 1] = BG

    for container in containers:
        cid = int(container["id"])
        if container["source_color"] is not None:
            continue
        interior = container["interior_box"]
        candidates = []
        for source in sources:
            sid = int(source["id"])
            dist = source_dists[sid][cid]
            if dist is not None:
                candidates.append((sid, int(dist)))
        if len(candidates) == 1:
            source_color = int(containers[candidates[0][0]]["source_color"])
            r0, c0, r1, c1 = interior
            out[r0 : r1 + 1, c0 : c1 + 1] = source_color
            continue
        if len(candidates) <= 1:
            continue
        best = min(dist for _sid, dist in candidates)
        winners = [sid for sid, dist in candidates if dist == best]
        if len(winners) != 1:
            continue
        source_color = int(containers[winners[0]]["source_color"])
        _r0, c0, r1, c1 = interior
        out[r1, c0 : c1 + 1] = source_color

    return out


def _draw_container(
    grid: np.ndarray,
    top: int,
    left: int,
    height: int,
    width: int,
    source_color: int | None,
) -> dict:
    bottom = top + height - 1
    right = left + width - 1
    grid[top : bottom + 1, left] = WALL
    grid[top : bottom + 1, right] = WALL
    grid[top, left : right + 1] = WALL
    grid[bottom, left : right + 1] = WALL
    if source_color is not None:
        grid[bottom - 1, left + 1 : right] = int(source_color)
    return {
        "wall_box": (top, left, bottom, right),
        "interior_box": (top + 1, left + 1, bottom - 1, right - 1),
        "source_color": source_color,
    }


def _draw_pipe(grid: np.ndarray, a: dict, b: dict, blocked: bool) -> None:
    ar0, ac0, ar1, ac1 = a["wall_box"]
    br0, bc0, br1, bc1 = b["wall_box"]
    if ar0 == br0:
        row = ar0 + 2
        if ac0 < bc0:
            start = ac1 + 1
            end = bc0 - 1
        else:
            start = bc1 + 1
            end = ac0 - 1
        assert start <= end, "horizontal pipe needs at least one cell"
        grid[row, start : end + 1] = WALL
        if blocked:
            grid[row, start + (end - start) // 2] = VALVE
    else:
        col = ac0 + 2
        if ar0 < br0:
            start = ar1 + 1
            end = br0 - 1
        else:
            start = br1 + 1
            end = ar0 - 1
        assert start <= end, "vertical pipe needs at least one cell"
        grid[start : end + 1, col] = WALL
        if blocked:
            grid[start + (end - start) // 2, col] = VALVE


def _crop_nonblack(inp: np.ndarray, out: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mask = (inp != BG) | (out != BG)
    coords = np.argwhere(mask)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    top_extra = random.randint(0, min(2, int(r0)))
    left_extra = random.randint(0, min(2, int(c0)))
    bottom_extra = random.randint(0, min(2, inp.shape[0] - 1 - int(r1)))
    right_extra = random.randint(0, min(2, inp.shape[1] - 1 - int(c1)))
    rr0 = int(r0) - top_extra
    cc0 = int(c0) - left_extra
    rr1 = int(r1) + bottom_extra
    cc1 = int(c1) + right_extra
    return inp[rr0 : rr1 + 1, cc0 : cc1 + 1].copy(), out[rr0 : rr1 + 1, cc0 : cc1 + 1].copy()


def _transform_pair(inp: np.ndarray, out: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    k = random.randrange(4)
    inp2 = np.rot90(inp, k).copy()
    out2 = np.rot90(out, k).copy()
    if random.random() < 0.5:
        inp2 = np.fliplr(inp2).copy()
        out2 = np.fliplr(out2).copy()
    return inp2, out2


def _scenario_single() -> tuple[dict[tuple[int, int], int | None], list[tuple[tuple[int, int], tuple[int, int], bool]]]:
    nodes: dict[tuple[int, int], int | None] = {
        (0, 0): None,
        (0, 1): None,
        (1, 1): None,
        (0, 2): None,
    }
    if random.random() < 0.5:
        nodes[(1, 2)] = None
    edges = [
        ((0, 0), (0, 1), False),
        ((0, 1), (1, 1), False),
        ((0, 1), (0, 2), True),
    ]
    if (1, 2) in nodes:
        edges.append(((1, 1), (1, 2), False))
    return nodes, edges


def _scenario_nearest() -> tuple[dict[tuple[int, int], int | None], list[tuple[tuple[int, int], tuple[int, int], bool]]]:
    nodes = {
        (0, 0): None,
        (0, 1): None,
        (1, 1): None,
        (1, 2): None,
        (0, 2): None,
    }
    edges = [
        ((0, 0), (0, 1), False),
        ((0, 1), (1, 1), False),
        ((1, 1), (1, 2), False),
        ((0, 1), (0, 2), True),
    ]
    return nodes, edges


def _scenario_tie() -> tuple[dict[tuple[int, int], int | None], list[tuple[tuple[int, int], tuple[int, int], bool]]]:
    nodes = {
        (0, 0): None,
        (0, 1): None,
        (1, 1): None,
        (1, 2): None,
        (0, 2): None,
        (1, 0): None,
    }
    edges = [
        ((0, 0), (0, 1), False),
        ((0, 1), (1, 1), False),
        ((1, 1), (1, 2), False),
        ((1, 2), (0, 2), False),
        ((1, 1), (1, 0), True),
    ]
    return nodes, edges


def _paint_expected(
    inp: np.ndarray,
    node_map: dict[tuple[int, int], dict],
    edges: list[tuple[tuple[int, int], tuple[int, int], bool]],
) -> np.ndarray:
    out = inp.copy()
    positions = list(node_map.keys())
    index = {pos: idx for idx, pos in enumerate(positions)}
    open_adj = [[] for _ in positions]
    for a, b, blocked in edges:
        if blocked:
            continue
        ai = index[a]
        bi = index[b]
        open_adj[ai].append(bi)
        open_adj[bi].append(ai)

    source_ids = [index[pos] for pos, data in node_map.items() if data["source_color"] is not None]
    dists: dict[int, list[int | None]] = {}
    for sid in source_ids:
        dist: list[int | None] = [None] * len(positions)
        dist[sid] = 0
        q: deque[int] = deque([sid])
        while q:
            cur = q.popleft()
            for nxt in open_adj[cur]:
                if dist[nxt] is None:
                    dist[nxt] = int(dist[cur]) + 1
                    q.append(nxt)
        dists[sid] = dist

    for pos, data in node_map.items():
        if data["source_color"] is not None:
            continue
        r0, c0, r1, c1 = data["interior_box"]
        out[r0 : r1 + 1, c0 : c1 + 1] = BG
        cid = index[pos]
        candidates = [(sid, dists[sid][cid]) for sid in source_ids if dists[sid][cid] is not None]
        if len(candidates) == 1:
            color = int(node_map[positions[candidates[0][0]]]["source_color"])
            out[r0 : r1 + 1, c0 : c1 + 1] = color
        elif len(candidates) > 1:
            best = min(int(dist) for _sid, dist in candidates if dist is not None)
            winners = [sid for sid, dist in candidates if int(dist) == best]
            if len(winners) == 1:
                color = int(node_map[positions[winners[0]]]["source_color"])
                out[r1, c0 : c1 + 1] = color
    return out


def _build_case(
    scenario: str,
) -> tuple[np.ndarray, np.ndarray]:
    if scenario == "single":
        nodes, edges = _scenario_single()
        source_count = 1
    elif scenario == "nearest":
        nodes, edges = _scenario_nearest()
        source_count = 2
    else:
        nodes, edges = _scenario_tie()
        source_count = 2

    colors = random.sample(SOURCE_COLORS, source_count)
    if scenario == "single":
        source_positions = [(0, 0)]
    elif scenario == "nearest":
        source_positions = [(0, 0), (1, 2)]
    else:
        source_positions = [(0, 0), (0, 2)]
    source_color_map = {pos: color for pos, color in zip(source_positions, colors, strict=False)}

    grid = np.full((MASTER_SIZE, MASTER_SIZE), BG, dtype=int)
    node_data: dict[tuple[int, int], dict] = {}
    for row, col in nodes:
        height = random.randint(*BOX_SIZE_RANGE)
        width = random.randint(*BOX_SIZE_RANGE)
        top = ROW_TOPS[row]
        left = COL_LEFTS[col]
        node_data[(row, col)] = _draw_container(
            grid,
            top,
            left,
            height,
            width,
            source_color_map.get((row, col)),
        )

    for a, b, blocked in edges:
        _draw_pipe(grid, node_data[a], node_data[b], blocked)

    expected = _paint_expected(grid, node_data, edges)
    transformed_inp, transformed_out = _transform_pair(grid, expected)
    return _crop_nonblack(transformed_inp, transformed_out)


def _gen_attempt() -> np.ndarray | None:
    scenario = random.choice(("single", "nearest", "tie"))
    inp, expected = _build_case(scenario)
    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        candidate = _gen_attempt()
        if candidate is not None:
            return candidate
    raise RuntimeError("failed to generate a valid container-network input")
