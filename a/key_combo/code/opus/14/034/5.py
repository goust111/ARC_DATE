import itertools
import random
from collections import deque

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
ORANGE = 7

UP = (-1, 0)
RIGHT = (0, 1)
DOWN = (1, 0)
LEFT = (0, -1)
DIRS = (UP, RIGHT, DOWN, LEFT)
CW = {UP: RIGHT, RIGHT: DOWN, DOWN: LEFT, LEFT: UP}
OPPOSITE = {UP: DOWN, RIGHT: LEFT, DOWN: UP, LEFT: RIGHT}

GEN_RETRIES = 1200
MIN_SIZE = 22
MAX_SIZE = 30


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _neighbors8(r: int, c: int):
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            yield r + dr, c + dc


def _components_mask(mask: np.ndarray) -> list[list[tuple[int, int]]]:
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
        comp: list[tuple[int, int]] = []
        while q:
            rr, cc = q.popleft()
            comp.append((rr, cc))
            for nr, nc in _neighbors4(rr, cc):
                if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    q.append((nr, nc))
        comps.append(comp)
    return comps


def _centroid(cells: list[tuple[int, int]]) -> tuple[float, float]:
    rs = [r for r, _ in cells]
    cs = [c for _, c in cells]
    return float(sum(rs) / len(rs)), float(sum(cs) / len(cs))


def _infer_direction_from_centroid(body_cells: list[tuple[int, int]], port: tuple[int, int]) -> tuple[int, int] | None:
    cr, cc = _centroid(body_cells)
    pr, pc = port
    dr = pr - cr
    dc = pc - cc
    if abs(dr) >= abs(dc) and pr < cr:
        return UP
    if abs(dc) >= abs(dr) and pc > cc:
        return RIGHT
    if abs(dr) >= abs(dc) and pr > cr:
        return DOWN
    if abs(dc) >= abs(dr) and pc < cc:
        return LEFT
    return None


def _parse_grid(grid: np.ndarray, *, enforce_input_constraints: bool = True) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape

    yellow_pts = np.argwhere(g == YELLOW)
    assert len(yellow_pts) == 1, "expected exactly one yellow start marker"
    yr, yc = map(int, yellow_pts[0])
    assert yr == h - 1, "yellow marker must be on the bottom row"
    assert yr > 0 and int(g[yr - 1, yc]) == RED, "red seed must be directly above the yellow marker"
    red_pts = np.argwhere(g == RED)
    assert len(red_pts) == 1, "input must contain exactly one red seed pixel"
    assert tuple(map(int, red_pts[0])) == (yr - 1, yc)

    blue_comps = _components_mask(g == BLUE)
    if enforce_input_constraints:
        assert 3 <= len(blue_comps) <= 5, "expected 3-5 blue blobs"
    else:
        assert blue_comps, "expected at least one blue blob"

    blob_at = np.full((h, w), -1, dtype=int)
    blobs: list[dict] = []
    for bid, comp in enumerate(blue_comps):
        if enforce_input_constraints:
            assert 5 <= len(comp) <= 10, "each blue blob must have 5-10 cells"
        for r, c in comp:
            blob_at[r, c] = bid
        blobs.append({"cells": sorted(comp), "ports": []})

    green_pts = [tuple(map(int, pt)) for pt in np.argwhere(g == GREEN)]
    port_counts = [0 for _ in blobs]
    for pr, pc in green_pts:
        adj = sorted({int(blob_at[nr, nc]) for nr, nc in _neighbors4(pr, pc) if _in_bounds(g.shape, nr, nc) and blob_at[nr, nc] != -1})
        assert len(adj) == 1, "each green port must touch exactly one blue blob"
        bid = adj[0]
        direction = _infer_direction_from_centroid(blobs[bid]["cells"], (pr, pc))
        assert direction is not None, "port direction must be inferable from blob centroid"
        blobs[bid]["ports"].append({"cell": (pr, pc), "dir": direction})
        port_counts[bid] += 1

    for bid, blob in enumerate(blobs):
        if enforce_input_constraints:
            assert 1 <= port_counts[bid] <= 2, "each blob must have 1-2 ports"
        else:
            assert port_counts[bid] >= 1, "each blob must have at least one port"
        blob["ports"].sort(key=lambda port: port["cell"])

    orange_pts = [tuple(map(int, pt)) for pt in np.argwhere(g == ORANGE)]
    if enforce_input_constraints:
        assert 4 <= len(orange_pts) <= 8, "expected 4-8 orange obstacles"

    return {
        "grid": g,
        "shape": g.shape,
        "seed": (yr - 1, yc),
        "yellow": (yr, yc),
        "blobs": blobs,
        "blob_at": blob_at,
        "orange": orange_pts,
    }


def _simulate(
    grid: np.ndarray,
    *,
    treat_obstacles_as_black: bool = False,
    initial_only: bool = False,
    strict_input: bool = False,
) -> dict:
    parsed = _parse_grid(grid, enforce_input_constraints=strict_input)
    g = parsed["grid"]
    out = g.copy()
    blobs = parsed["blobs"]
    blob_at = parsed["blob_at"]

    activated: set[int] = set()
    queue: deque[int] = deque()
    activation_order: list[int] = []
    traces: list[dict] = []

    def activate(bid: int) -> None:
        if bid in activated:
            return
        activated.add(bid)
        activation_order.append(bid)
        for r, c in blobs[bid]["cells"]:
            out[r, c] = RED
        if not initial_only:
            queue.append(bid)

    def cast_ray(start: tuple[int, int], direction: tuple[int, int], label: str) -> None:
        current = start
        dir_now = direction
        painted: list[tuple[int, int]] = []
        turns = 0
        hit_blob = None
        stop_reason = "boundary"
        seen_states: set[tuple[int, int, tuple[int, int]]] = set()
        while True:
            state = (current[0], current[1], dir_now)
            if state in seen_states:
                stop_reason = "cycle"
                break
            seen_states.add(state)

            nr = current[0] + dir_now[0]
            nc = current[1] + dir_now[1]
            if not _in_bounds(g.shape, nr, nc):
                stop_reason = "boundary"
                break

            if out[nr, nc] == RED or g[nr, nc] in (GREEN, YELLOW):
                stop_reason = "red_or_green"
                break

            if g[nr, nc] == ORANGE and not treat_obstacles_as_black:
                turns += 1
                dir_now = CW[dir_now]
                stop_reason = "turned"
                continue

            bid = int(blob_at[nr, nc])
            if bid != -1:
                activate(bid)
                hit_blob = bid
                stop_reason = "blob"
                break

            if g[nr, nc] in (BLACK, ORANGE):
                out[nr, nc] = RED
                painted.append((nr, nc))
                current = (nr, nc)
                stop_reason = "advance"
                continue

            stop_reason = "blocked"
            break

        traces.append(
            {
                "label": label,
                "start": start,
                "initial_dir": direction,
                "final_dir": dir_now,
                "painted": painted,
                "turns": turns,
                "hit_blob": hit_blob,
                "stop_reason": stop_reason,
            }
        )

    cast_ray(parsed["seed"], UP, "seed")
    while queue:
        bid = queue.popleft()
        for idx, port in enumerate(blobs[bid]["ports"]):
            cast_ray(port["cell"], port["dir"], f"blob{bid}_port{idx}")

    return {
        "out": out,
        "activated": activated,
        "activation_order": activation_order,
        "traces": traces,
        "parsed": parsed,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor the blue obstacle-routed cascade network reached by clockwise-turning rays.

    Input structure:
    - A black background contains 3-5 irregular blue 4-connected blobs, each with 1-2 adjacent green port pixels.
    - Orange obstacle pixels are isolated from blobs and ports, and a yellow marker sits on the bottom row with a single red seed directly above it.
    - A port's emission direction is encoded by which side of its blob centroid the green pixel lies on.

    Rule:
    - Fire a red ray upward from the seed pixel. Each step examines the next cell in the current direction.
    - Black cells are recolored red and traversed; orange obstacle cells are not recolored but rotate the ray 90 degrees clockwise from the current position; blue blob cells stop the ray and recolor the whole 4-connected blob red.
    - Green port cells and already-red cells stop a ray. Whenever a blob is activated, its green ports emit new rays in row-major order. Repeat until no new rays remain.

    Key insight:
    - Orange pixels are turn triggers rather than walls, so the obstacle layout rewires straight port rays into a 2D activation graph whose reachable red blobs depend on both turning and cascade emission.
    """
    return _simulate(grid, strict_input=True)["out"]


def _rotations_and_flips(mask: np.ndarray) -> list[np.ndarray]:
    variants: list[np.ndarray] = []
    seen: set[bytes] = set()
    for flip in (False, True):
        base = np.fliplr(mask) if flip else mask
        for k in range(4):
            var = np.rot90(base, k)
            key = var.astype(np.uint8).tobytes() + bytes(var.shape)
            if key not in seen:
                seen.add(key)
                variants.append(var.copy())
    return variants


BASE_TEMPLATES = [
    np.array([[1, 1, 1], [1, 0, 0], [1, 0, 0]], dtype=bool),
    np.array([[1, 1, 1], [0, 1, 0], [0, 1, 1]], dtype=bool),
    np.array([[1, 1, 1], [1, 1, 0], [0, 1, 0]], dtype=bool),
    np.array([[1, 1, 1], [1, 0, 1], [1, 1, 0]], dtype=bool),
    np.array([[1, 1, 0], [0, 1, 1], [1, 1, 0]], dtype=bool),
    np.array([[1, 1, 1, 0], [0, 1, 1, 1], [0, 0, 1, 0]], dtype=bool),
    np.array([[1, 1, 1, 0], [1, 0, 1, 1], [0, 0, 1, 0]], dtype=bool),
    np.array([[1, 1, 1, 0], [1, 1, 1, 1], [1, 0, 1, 0]], dtype=bool),
]

TEMPLATE_VARIANTS = [variant for base in BASE_TEMPLATES for variant in _rotations_and_flips(base)]


def _mask_cells(mask: np.ndarray) -> list[tuple[int, int]]:
    return [tuple(map(int, pt)) for pt in np.argwhere(mask)]


def _boundary_body_locals(mask: np.ndarray, side: tuple[int, int]) -> list[tuple[int, int]]:
    h, w = mask.shape
    out: list[tuple[int, int]] = []
    for r, c in _mask_cells(mask):
        nr = r + side[0]
        nc = c + side[1]
        if not _in_bounds((h, w), nr, nc) or not mask[nr, nc]:
            out.append((r, c))
    return out


def _sorted_side_candidates(mask: np.ndarray, side: tuple[int, int], *, for_ports: bool) -> list[tuple[int, int]]:
    body = _mask_cells(mask)
    cr, cc = _centroid(body)
    scored: list[tuple[float, tuple[int, int]]] = []
    for r, c in _boundary_body_locals(mask, side):
        if for_ports:
            port = (r + side[0], c + side[1])
            if _infer_direction_from_centroid(body, port) != side:
                continue
            rr, cc0 = port
        else:
            rr, cc0 = r, c
        orth = abs(cc0 - cc) if side in (UP, DOWN) else abs(rr - cr)
        main = abs(rr - cr) if side in (UP, DOWN) else abs(cc0 - cc)
        scored.append((orth + 0.01 * main, (r, c)))
    scored.sort(key=lambda item: item[0])
    return [cell for _, cell in scored]


def _new_blocked(blocked: np.ndarray, cells: list[tuple[int, int]]) -> np.ndarray:
    new_blocked = blocked.copy()
    for r, c in cells:
        for nr, nc in itertools.chain([(r, c)], _neighbors4(r, c)):
            if _in_bounds(blocked.shape, nr, nc):
                new_blocked[nr, nc] = True
    return new_blocked


def _place_blob(
    rng: random.Random,
    grid: np.ndarray,
    blocked: np.ndarray,
    hit: tuple[int, int] | None = None,
    ray_dir: tuple[int, int] | None = None,
    required_port_sides: list[tuple[int, int]] | None = None,
) -> dict | None:
    required_port_sides = list(required_port_sides or [])
    variants = TEMPLATE_VARIANTS[:]
    rng.shuffle(variants)
    for mask in variants:
        body_locals = _mask_cells(mask)
        hit_locals = [None]
        if hit is not None and ray_dir is not None:
            hit_side = OPPOSITE[ray_dir]
            hit_locals = _sorted_side_candidates(mask, hit_side, for_ports=False)
            if not hit_locals:
                continue
            rng.shuffle(hit_locals)

        for hit_local in hit_locals:
            if hit is None:
                h, w = mask.shape
                top = rng.randint(1, grid.shape[0] - h - 2)
                left = rng.randint(1, grid.shape[1] - w - 2)
            else:
                top = hit[0] - hit_local[0]
                left = hit[1] - hit_local[1]

            abs_body = [(top + r, left + c) for r, c in body_locals]
            if not all(_in_bounds(grid.shape, r, c) for r, c in abs_body):
                continue
            if any(blocked[r, c] or grid[r, c] != BLACK for r, c in abs_body):
                continue

            port_options: list[list[tuple[int, int]]] = []
            failed = False
            for side in required_port_sides:
                candidates: list[tuple[int, int]] = []
                for br, bc in _sorted_side_candidates(mask, side, for_ports=True):
                    pr = top + br + side[0]
                    pc = left + bc + side[1]
                    if not _in_bounds(grid.shape, pr, pc):
                        continue
                    if blocked[pr, pc] or grid[pr, pc] != BLACK:
                        continue
                    candidates.append((pr, pc))
                if not candidates:
                    failed = True
                    break
                port_options.append(candidates)
            if failed:
                continue

            for combo in itertools.product(*port_options) if port_options else [()]:
                if len(set(combo)) != len(combo):
                    continue
                abs_ports = list(combo)
                occupied = set(abs_body)
                if any(port in occupied for port in abs_ports):
                    continue
                return {
                    "mask": mask,
                    "top": top,
                    "left": left,
                    "body": abs_body,
                    "ports": list(zip(required_port_sides, abs_ports)),
                    "blocked": _new_blocked(blocked, abs_body + abs_ports),
                }
    return None


def _apply_blob(grid: np.ndarray, blob: dict) -> None:
    for r, c in blob["body"]:
        grid[r, c] = BLUE
    for _side, (r, c) in blob["ports"]:
        grid[r, c] = GREEN


def _port_candidates_for_blob(grid: np.ndarray, blocked: np.ndarray, blob: dict, side: tuple[int, int]) -> list[tuple[int, int]]:
    mask = blob["mask"]
    top = blob["top"]
    left = blob["left"]
    out: list[tuple[int, int]] = []
    for br, bc in _sorted_side_candidates(mask, side, for_ports=True):
        pr = top + br + side[0]
        pc = left + bc + side[1]
        if not _in_bounds(grid.shape, pr, pc):
            continue
        if blocked[pr, pc] or grid[pr, pc] != BLACK:
            continue
        out.append((pr, pc))
    return out


def _add_port(
    grid: np.ndarray,
    blocked: np.ndarray,
    blob: dict,
    side: tuple[int, int],
    port: tuple[int, int],
) -> np.ndarray:
    grid[port[0], port[1]] = GREEN
    blob["ports"].append((side, port))
    blob["blocked"] = _new_blocked(blocked, [port])
    return blob["blocked"]


def _straight_clear_to_boundary(grid: np.ndarray, start: tuple[int, int], direction: tuple[int, int]) -> bool:
    r, c = start
    while True:
        r += direction[0]
        c += direction[1]
        if not _in_bounds(grid.shape, r, c):
            return True
        if grid[r, c] != BLACK:
            return False


def _painted_mask(out: np.ndarray) -> np.ndarray:
    return out == RED


def _first_step_sources(grid: np.ndarray) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    parsed = _parse_grid(grid, enforce_input_constraints=False)
    sources = [(parsed["seed"], UP)]
    for blob in parsed["blobs"]:
        for port in blob["ports"]:
            sources.append((port["cell"], port["dir"]))
    return sources


def _validate_semantics(grid: np.ndarray) -> bool:
    parsed = _parse_grid(grid)
    g = parsed["grid"]
    h, w = g.shape

    # Obstacles must be isolated from blobs, ports, start markers, and each other.
    orange_set = set(parsed["orange"])
    for r, c in parsed["orange"]:
        for nr, nc in _neighbors4(r, c):
            if not _in_bounds(g.shape, nr, nc):
                continue
            if (nr, nc) in orange_set:
                return False
            if g[nr, nc] in (BLUE, GREEN, RED, YELLOW):
                return False

    # No blue blob adjacency.
    blob_at = parsed["blob_at"]
    for r, c in np.argwhere(g == BLUE):
        r = int(r)
        c = int(c)
        bid = int(blob_at[r, c])
        for nr, nc in _neighbors4(r, c):
            if _in_bounds(g.shape, nr, nc) and g[nr, nc] == BLUE and int(blob_at[nr, nc]) != bid:
                return False

    # No obstacle immediately in front of any source.
    for src, direction in _first_step_sources(grid):
        nr = src[0] + direction[0]
        nc = src[1] + direction[1]
        if _in_bounds(g.shape, nr, nc) and g[nr, nc] == ORANGE:
            return False

    full = _simulate(grid)
    activated = full["activated"]
    if len(activated) < 2:
        return False
    if len(parsed["blobs"]) - len(activated) < 1:
        return False
    if any(trace["turns"] > 2 for trace in full["traces"]):
        return False

    initial_only = _simulate(grid, initial_only=True)
    if len(initial_only["activated"]) != 1:
        return False

    no_obstacles = grid.copy()
    no_obstacles[no_obstacles == ORANGE] = BLACK
    straight = _simulate(no_obstacles)
    if len(straight["activated"]) >= len(activated):
        return False
    if not (activated - straight["activated"]):
        return False

    essential_found = False
    for r, c in parsed["orange"]:
        single = grid.copy()
        single[r, c] = BLACK
        if _simulate(single)["activated"] != activated:
            essential_found = True
            break
    if not essential_found:
        return False

    # Ensure outputs are visibly non-degenerate.
    out = full["out"]
    if np.array_equal(out, grid):
        return False
    if np.count_nonzero(_painted_mask(out)) < 12:
        return False

    # At least one activated ray beyond the seed must exist.
    if len(full["traces"]) < 2:
        return False

    # All activated blobs must have had at least one port in the input.
    if any(len(blob["ports"]) < 1 for blob in parsed["blobs"]):
        return False

    return h <= 30 and w <= 30


def _clear_path(grid: np.ndarray, cells: list[tuple[int, int]]) -> bool:
    return all(_in_bounds(grid.shape, r, c) and grid[r, c] == BLACK for r, c in cells)


def _line_cells_exclusive(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    if a[0] == b[0]:
        step = 1 if b[1] > a[1] else -1
        return [(a[0], c) for c in range(a[1] + step, b[1], step)]
    if a[1] == b[1]:
        step = 1 if b[0] > a[0] else -1
        return [(r, a[1]) for r in range(a[0] + step, b[0], step)]
    raise ValueError("line must be axis-aligned")


def _random_dead_port_side(rng: random.Random, blob: dict) -> list[tuple[int, int]]:
    used = {side for side, _port in blob["ports"]}
    choices = [side for side in DIRS if side not in used]
    rng.shuffle(choices)
    return choices


def _try_add_dead_port(rng: random.Random, grid: np.ndarray, blocked: np.ndarray, blob: dict) -> np.ndarray:
    for side in _random_dead_port_side(rng, blob):
        for port in _port_candidates_for_blob(grid, blocked, blob, side):
            if _straight_clear_to_boundary(grid, port, side):
                return _add_port(grid, blocked, blob, side, port)
    return blocked


def _place_blob_with_dead_port(
    rng: random.Random,
    grid: np.ndarray,
    blocked: np.ndarray,
    *,
    hit: tuple[int, int] | None = None,
    ray_dir: tuple[int, int] | None = None,
) -> dict | None:
    sides = list(DIRS)
    rng.shuffle(sides)
    for side in sides:
        blob = _place_blob(rng, grid, blocked, hit=hit, ray_dir=ray_dir, required_port_sides=[side])
        if blob is None:
            continue
        port = blob["ports"][0][1]
        if _straight_clear_to_boundary(grid, port, side):
            return blob
    return None


def _place_free_blob(rng: random.Random, grid: np.ndarray, blocked: np.ndarray) -> tuple[dict, np.ndarray] | None:
    for _ in range(120):
        original_grid = grid.copy()
        blob = _place_blob_with_dead_port(rng, grid, blocked)
        if blob is None:
            continue
        _apply_blob(grid, blob)
        local_blocked = blob["blocked"]
        if len(blob["ports"]) == 1 and rng.random() < 0.55:
            local_blocked = _try_add_dead_port(rng, grid, local_blocked, blob)
        return blob, local_blocked
    return None


def _gen_attempt(rng: random.Random, debug: bool = False):
    def fail(reason: str):
        return (None, reason) if debug else None

    h = rng.randint(MIN_SIZE, MAX_SIZE)
    w = rng.randint(MIN_SIZE + 1, MAX_SIZE + 2)
    w = min(w, 30)
    grid = np.zeros((h, w), dtype=int)
    blocked = np.zeros((h, w), dtype=bool)

    seed_c = rng.randint(2, max(3, w // 3))
    grid[h - 1, seed_c] = YELLOW
    grid[h - 2, seed_c] = RED
    blocked = _new_blocked(blocked, [(h - 1, seed_c), (h - 2, seed_c)])

    chain_len = 2 if rng.random() < 0.8 else 3
    turned_final = chain_len == 3 and rng.random() < 0.5

    o1_r = rng.randint(4, h - 11)
    if not _clear_path(grid, [(r, seed_c) for r in range(o1_r + 1, h - 2)]):
        return None
    grid[o1_r, seed_c] = ORANGE
    blocked = _new_blocked(blocked, [(o1_r, seed_c)])

    a_lo = seed_c + 4
    a_hi = min(w - 12, seed_c + 7)
    if a_lo > a_hi:
        return fail("a_range")
    a_hit = (o1_r + 1, rng.randint(a_lo, a_hi))
    if not _clear_path(grid, _line_cells_exclusive((o1_r + 1, seed_c), a_hit)):
        return None
    a_blob = _place_blob(rng, grid, blocked, hit=a_hit, ray_dir=RIGHT, required_port_sides=[RIGHT])
    if a_blob is None:
        return fail("a_blob")
    _apply_blob(grid, a_blob)
    blocked = a_blob["blocked"]
    if rng.random() < 0.55:
        blocked = _try_add_dead_port(rng, grid, blocked, a_blob)
    a_port = next(port for side, port in a_blob["ports"] if side == RIGHT)

    o2_lo = a_port[1] + 3
    o2_hi = w - 6
    if o2_lo > o2_hi:
        return fail("o2_range")
    o2_c = rng.randint(o2_lo, o2_hi)
    if not _clear_path(grid, _line_cells_exclusive(a_port, (a_port[0], o2_c))):
        return fail("o2_path")
    grid[a_port[0], o2_c] = ORANGE
    blocked = _new_blocked(blocked, [(a_port[0], o2_c)])

    b_row_lo = a_port[0] + 4
    b_row_hi = h - 6
    if b_row_lo > b_row_hi:
        return fail("b_range")
    b_hit = (rng.randint(b_row_lo, b_row_hi), o2_c - 1)
    if not _clear_path(grid, _line_cells_exclusive((a_port[0], o2_c - 1), b_hit)):
        return fail("b_path")

    b_required = [LEFT] if turned_final else [UP]
    b_blob = _place_blob(rng, grid, blocked, hit=b_hit, ray_dir=DOWN, required_port_sides=b_required)
    if b_blob is None:
        return fail("b_blob")
    _apply_blob(grid, b_blob)
    blocked = b_blob["blocked"]

    active_blobs = [a_blob, b_blob]
    if chain_len == 3:
        if turned_final:
            b_port = next(port for side, port in b_blob["ports"] if side == LEFT)
            lo = max(2, seed_c + 3)
            hi = b_port[1] - 3
            if lo > hi:
                return fail("o3_range")
            o3_c = rng.randint(lo, hi)
            if not _clear_path(grid, _line_cells_exclusive((b_port[0], b_port[1]), (b_port[0], o3_c))):
                return fail("o3_path")
            grid[b_port[0], o3_c] = ORANGE
            blocked = _new_blocked(blocked, [(b_port[0], o3_c)])
            c_row_hi = b_port[0] - 5
            if 2 > c_row_hi:
                return fail("c_range_turn")
            c_hit = (rng.randint(2, c_row_hi), o3_c + 1)
            if not _clear_path(grid, _line_cells_exclusive((b_port[0], o3_c + 1), c_hit)):
                return fail("c_path_turn")
        else:
            b_port = next(port for side, port in b_blob["ports"] if side == UP)
            c_row_hi = b_port[0] - 5
            if 2 > c_row_hi:
                return fail("c_range_straight")
            c_hit = (rng.randint(2, c_row_hi), b_port[1])
            if not _clear_path(grid, _line_cells_exclusive(b_port, c_hit)):
                return fail("c_path_straight")
        c_blob = _place_blob_with_dead_port(rng, grid, blocked, hit=c_hit, ray_dir=UP)
        if c_blob is None:
            return fail("c_blob")
        _apply_blob(grid, c_blob)
        blocked = c_blob["blocked"]
        if len(c_blob["ports"]) == 1 and rng.random() < 0.45:
            blocked = _try_add_dead_port(rng, grid, blocked, c_blob)
        active_blobs.append(c_blob)
    else:
        if len(b_blob["ports"]) == 1:
            blocked = _try_add_dead_port(rng, grid, blocked, b_blob)
        if len(b_blob["ports"]) == 1 and rng.random() < 0.45:
            blocked = _try_add_dead_port(rng, grid, blocked, b_blob)

    partial = _simulate(grid)
    expected_active = 3 if chain_len == 3 else 2
    if len(partial["activated"]) != expected_active:
        return fail("partial_active")
    red_mask = _painted_mask(partial["out"])

    total_blobs = rng.choices([4, 5], weights=[3, 1])[0] if chain_len == 3 else rng.choices([3, 4], weights=[2, 3])[0]

    while len(active_blobs) < total_blobs:
        result = _place_free_blob(rng, grid, blocked)
        if result is None:
            return fail("free_blob")
        blob, blocked = result
        if any(red_mask[r, c] for r, c in blob["body"]):
            return fail("free_body_on_red")
        if any(red_mask[r, c] for _side, (r, c) in blob["ports"]):
            return fail("free_port_on_red")
        active_blobs.append(blob)

    essential_obstacles = len(np.argwhere(grid == ORANGE))
    if essential_obstacles > 8:
        return fail("too_many_essential_obstacles")
    target_obstacles = rng.randint(max(4, essential_obstacles), 8)

    blocked_for_obstacles = blocked.copy()
    for src, direction in _first_step_sources(grid):
        nr = src[0] + direction[0]
        nc = src[1] + direction[1]
        if _in_bounds(grid.shape, nr, nc):
            blocked_for_obstacles[nr, nc] = True

    free_obstacle_cells = [
        (r, c)
        for r in range(h)
        for c in range(w)
        if grid[r, c] == BLACK and not red_mask[r, c] and not blocked_for_obstacles[r, c]
    ]
    rng.shuffle(free_obstacle_cells)
    while len(np.argwhere(grid == ORANGE)) < target_obstacles and free_obstacle_cells:
        r, c = free_obstacle_cells.pop()
        ok = True
        for nr, nc in itertools.chain([(r, c)], _neighbors4(r, c)):
            if not _in_bounds(grid.shape, nr, nc):
                continue
            if grid[nr, nc] in (BLUE, GREEN, RED, YELLOW, ORANGE):
                ok = False
                break
        if not ok:
            continue
        grid[r, c] = ORANGE
        blocked_for_obstacles = _new_blocked(blocked_for_obstacles, [(r, c)])

    if not _validate_semantics(grid):
        return fail("final_validate")
    return (grid, "ok") if debug else grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for obstacle-turn cascade network")
