from __future__ import annotations

from collections import Counter, deque
from itertools import combinations
import random

import numpy as np


BG = 0
SEED = 1
ACTIVE_COLORS = (2, 3, 4, 5)
SMALL_MAX_AREA = 2
GEN_ATTEMPTS = 400


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _is_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    start = tuple(map(int, coords[0]))
    q = deque([start])
    seen = {start}
    h, w = mask.shape
    while q:
        r, c = q.popleft()
        for rr, cc in _neighbors4(r, c):
            if 0 <= rr < h and 0 <= cc < w and mask[rr, cc] and (rr, cc) not in seen:
                seen.add((rr, cc))
                q.append((rr, cc))
    return len(seen) == int(mask.sum())


def _build_shape_catalog() -> dict[int, list[np.ndarray]]:
    catalog: dict[int, list[np.ndarray]] = {area: [] for area in range(2, 9)}
    for h in range(1, 5):
        for w in range(1, 5):
            total = h * w
            for area in range(2, min(8, total) + 1):
                seen = set()
                for cells in combinations(range(total), area):
                    mask = np.zeros((h, w), dtype=bool)
                    mask.ravel()[list(cells)] = True
                    coords = np.argwhere(mask)
                    if coords.size == 0:
                        continue
                    if _bbox(coords) != (0, 0, h - 1, w - 1):
                        continue
                    if not _is_connected(mask):
                        continue
                    key = tuple(int(x) for x in mask.ravel())
                    if key in seen:
                        continue
                    seen.add(key)
                    catalog[area].append(mask.copy())
    return catalog


SHAPE_CATALOG = _build_shape_catalog()
SMALL_SHAPES = [mask for mask in SHAPE_CATALOG[2] if mask.shape in {(1, 2), (2, 1)}]
MOVER_SHAPES = [
    np.array([[1, 1, 1]], dtype=bool),
    np.array([[1], [1], [1]], dtype=bool),
    np.array([[1, 1], [1, 0]], dtype=bool),
    np.array([[1, 1], [0, 1]], dtype=bool),
    np.array([[1, 0], [1, 1]], dtype=bool),
    np.array([[0, 1], [1, 1]], dtype=bool),
    np.array([[1, 1], [1, 1]], dtype=bool),
    np.array([[1, 1, 1], [0, 1, 0]], dtype=bool),
]


def _find_seed(grid: np.ndarray) -> tuple[int, int]:
    coords = np.argwhere(grid == SEED)
    assert coords.shape[0] == 1, "expected exactly one blue seed"
    return int(coords[0, 0]), int(coords[0, 1])


def _ray_path(shape: tuple[int, int], seed: tuple[int, int]) -> list[tuple[int, int]]:
    h, w = map(int, shape)
    r, c = map(int, seed)
    dc = 1
    path: list[tuple[int, int]] = []
    while 0 <= r < h and 0 <= c < w:
        path.append((r, c))
        nr = r - 1
        if nr < 0:
            break
        nc = c + dc
        if nc < 0 or nc >= w:
            dc = -dc
            nc = c + dc
        r, c = nr, nc
    return path


def _build_lane_map(shape: tuple[int, int], path: list[tuple[int, int]]) -> np.ndarray:
    h, w = map(int, shape)
    fine = np.zeros((3 * h, 3 * w), dtype=bool)

    for r, c in path:
        fine[3 * r : 3 * r + 3, 3 * c : 3 * c + 3] = True

    for (r1, c1), (r2, c2) in zip(path, path[1:]):
        if c2 == c1 + 1:
            bridge = (
                (3 * r1, 3 * c1 + 2),
                (3 * r1 + 1, 3 * c1 + 2),
                (3 * r1, 3 * c1 + 3),
                (3 * r1 + 1, 3 * c1 + 3),
                (3 * r1 + 2, 3 * c1 + 3),
            )
        else:
            bridge = (
                (3 * r1, 3 * c1),
                (3 * r1 + 1, 3 * c1),
                (3 * r1, 3 * c1 - 1),
                (3 * r1 + 1, 3 * c1 - 1),
                (3 * r1 + 2, 3 * c1 - 1),
            )
        for rr, cc in bridge:
            if 0 <= rr < 3 * h and 0 <= cc < 3 * w:
                fine[rr, cc] = True

    lane_map = np.full((h, w), -1, dtype=int)
    seen = np.zeros_like(fine, dtype=bool)
    lane_id = 0

    for r in range(h):
        for c in range(w):
            cr = 3 * r + 1
            cc = 3 * c + 1
            if fine[cr, cc] or seen[cr, cc]:
                continue
            q = deque([(cr, cc)])
            seen[cr, cc] = True
            cells: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                if rr % 3 == 1 and cc % 3 == 1:
                    cells.append((rr // 3, cc // 3))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < 3 * h and 0 <= nc < 3 * w and not seen[nr, nc] and not fine[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            if cells:
                for rr, cc in cells:
                    lane_map[rr, cc] = lane_id
                lane_id += 1

    unique_ids = sorted(int(x) for x in np.unique(lane_map) if int(x) >= 0)
    order = sorted(unique_ids, key=lambda lid: float(np.argwhere(lane_map == lid)[:, 1].mean()))
    remap = {old: new for new, old in enumerate(order)}
    for old, new in remap.items():
        lane_map[lane_map == old] = new
    return lane_map


def _find_components(grid: np.ndarray) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color in (BG, SEED) or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            coords = np.array(cells, dtype=int)
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": _bbox(coords),
                    "area": int(coords.shape[0]),
                }
            )
    return comps


def _component_lane(comp: dict, lane_map: np.ndarray) -> int:
    r0, c0, r1, c1 = map(int, comp["bbox"])
    center_c = (c0 + c1) // 2
    column_ids = [int(v) for v in lane_map[r0 : r1 + 1, center_c] if int(v) >= 0]
    if column_ids and len(set(column_ids)) == 1:
        return int(column_ids[0])

    lane_counts = Counter(int(lane_map[r, c]) for r, c in comp["coords"] if int(lane_map[r, c]) >= 0)
    assert lane_counts, "component must belong to at least one lane cell"
    return int(lane_counts.most_common(1)[0][0])


def _drop_coords(coords: np.ndarray, occupied: np.ndarray, h: int) -> np.ndarray:
    dr = 0
    while True:
        shifted = coords + np.array([dr + 1, 0], dtype=int)
        if np.any(shifted[:, 0] >= h) or bool(occupied[shifted[:, 0], shifted[:, 1]].any()):
            break
        dr += 1
    return coords + np.array([dr, 0], dtype=int)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Drop each large object inside the hidden ray-defined lane that contains it while keeping small platforms fixed.

    Input structure:
    - The grid has black background, one blue seed pixel on the bottom edge, and several separated monochrome 4-connected objects.
    - Foreground object colors come from red, green, yellow, and gray.
    - Objects of area at most two are fixed platforms, while larger objects are movable sprites.

    Rule:
    - Fire an implicit 45-degree ray from the blue seed toward the upper right, reflecting horizontally on the side walls and stopping at the top edge.
    - Treat that hidden bounced path as a separating wall that partitions the board into vertical lanes, and assign each foreground object to the lane indicated by its bounding-box center column.
    - In each lane independently, erase the large objects from their starts and let them fall straight down, processing them from lower to higher original position, until they would hit the bottom, a fixed small platform in that lane, or a previously dropped large object from the same lane.
    - Erase the blue seed; keep every small platform unchanged; never draw the ray itself.

    Key insight:
    - The diagonal bounced ray is invisible in the output, but it still defines independent gravity channels, so similar objects fall differently depending on which hidden lane contains them.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape

    seed = _find_seed(g)
    path = _ray_path((h, w), seed)
    lane_map = _build_lane_map((h, w), path)
    comps = _find_components(g)
    assert comps, "expected foreground objects"

    stationary_by_lane: dict[int, list[dict]] = {}
    movers_by_lane: dict[int, list[dict]] = {}
    for comp in comps:
        lane = _component_lane(comp, lane_map)
        comp["lane"] = lane
        if int(comp["area"]) <= SMALL_MAX_AREA:
            stationary_by_lane.setdefault(lane, []).append(comp)
        else:
            movers_by_lane.setdefault(lane, []).append(comp)

    out = g.copy()
    out[seed] = BG
    for comp in comps:
        if int(comp["area"]) > SMALL_MAX_AREA:
            coords = comp["coords"]
            out[coords[:, 0], coords[:, 1]] = BG

    for lane in sorted(set(stationary_by_lane) | set(movers_by_lane)):
        occupied = np.zeros((h, w), dtype=bool)
        for comp in stationary_by_lane.get(lane, []):
            coords = comp["coords"]
            occupied[coords[:, 0], coords[:, 1]] = True

        movers = movers_by_lane.get(lane, [])
        movers.sort(key=lambda comp: (int(comp["bbox"][2]), int(comp["bbox"][0])), reverse=True)
        for comp in movers:
            dropped = _drop_coords(comp["coords"], occupied, h)
            occupied[dropped[:, 0], dropped[:, 1]] = True
            out[dropped[:, 0], dropped[:, 1]] = int(comp["color"])

    return out


def _coords_from_mask(mask: np.ndarray, r0: int, c0: int) -> np.ndarray:
    return np.argwhere(mask) + np.array([r0, c0], dtype=int)


def _touches_occupied(coords: np.ndarray, occupied: np.ndarray) -> bool:
    h, w = occupied.shape
    for r, c in coords:
        for rr, cc in _neighbors4(int(r), int(c)):
            if 0 <= rr < h and 0 <= cc < w and occupied[rr, cc]:
                return True
    return False


def _placement_candidates(
    mask: np.ndarray,
    lane_map: np.ndarray,
    lane_id: int,
    blocked: np.ndarray,
    *,
    row_lo: int = 0,
    row_hi: int | None = None,
    support_occ: np.ndarray | None = None,
    support_mode: str | None = None,
    no_touch: bool = False,
) -> list[np.ndarray]:
    h, w = lane_map.shape
    mh, mw = mask.shape
    if row_hi is None:
        row_hi = h - mh
    row_lo = max(0, int(row_lo))
    row_hi = min(int(row_hi), h - mh)
    candidates: list[np.ndarray] = []
    if row_lo > row_hi:
        return candidates

    for r0 in range(row_lo, row_hi + 1):
        for c0 in range(0, w - mw + 1):
            coords = _coords_from_mask(mask, r0, c0)
            rows = coords[:, 0]
            cols = coords[:, 1]
            if np.any(lane_map[rows, cols] != lane_id):
                continue
            if bool(blocked[rows, cols].any()):
                continue
            if no_touch and _touches_occupied(coords, blocked):
                continue

            if support_mode is not None:
                assert support_occ is not None
                below_rows = rows + 1
                bottom_hits = below_rows >= h
                occ_hits = np.zeros(len(coords), dtype=bool)
                inside = ~bottom_hits
                occ_hits[inside] = support_occ[below_rows[inside], cols[inside]]
                if support_mode == "bottom":
                    if not bool(bottom_hits.any()) or bool(occ_hits.any()):
                        continue
                elif support_mode == "platform":
                    if bool(bottom_hits.any()) or not bool(occ_hits.any()):
                        continue
                elif support_mode == "any":
                    if not (bool(bottom_hits.any()) or bool(occ_hits.any())):
                        continue
                else:
                    raise ValueError(f"unknown support_mode: {support_mode}")

            candidates.append(coords)
    return candidates


def _paint(grid: np.ndarray, coords: np.ndarray, color: int) -> None:
    grid[coords[:, 0], coords[:, 1]] = int(color)


def _sample_color(exclude: set[int] | None = None) -> int:
    choices = [c for c in ACTIVE_COLORS if exclude is None or c not in exclude]
    return int(random.choice(choices))


def _component_from_coords(coords: np.ndarray, color: int) -> dict:
    return {"coords": coords.copy(), "bbox": _bbox(coords), "area": int(coords.shape[0]), "color": int(color)}


def _sample_canvas() -> tuple[int, int, tuple[int, int], list[tuple[int, int]], np.ndarray] | None:
    w = random.choice([15, 17])
    h = random.randint(18, 21)
    seed = (h - 1, w // 2)
    path = _ray_path((h, w), seed)
    lane_map = _build_lane_map((h, w), path)
    lanes = sorted(int(x) for x in np.unique(lane_map) if int(x) >= 0)
    if len(lanes) < 3:
        return None
    return h, w, seed, path, lane_map


def _place_stationaries(
    lane_map: np.ndarray, lanes: list[int], lane_modes: dict[int, str]
) -> tuple[np.ndarray, dict[int, list[dict]]] | None:
    h, w = lane_map.shape
    occupied = np.zeros((h, w), dtype=bool)
    stationary_by_lane: dict[int, list[dict]] = {lane: [] for lane in lanes}

    for lane in lanes:
        shapes = SMALL_SHAPES[:]
        random.shuffle(shapes)
        placed = False
        for mask in shapes:
            if lane_modes[lane] == "platform":
                row_lo, row_hi = h // 2, h - 3
            else:
                row_lo, row_hi = 1, h - 3
            candidates = _placement_candidates(
                mask,
                lane_map,
                lane,
                occupied,
                row_lo=row_lo,
                row_hi=row_hi,
                no_touch=True,
            )
            if not candidates:
                continue
            coords = random.choice(candidates)
            color = _sample_color()
            stationary_by_lane[lane].append(_component_from_coords(coords, color))
            occupied[coords[:, 0], coords[:, 1]] = True
            placed = True
            break
        if not placed:
            return None
    return occupied, stationary_by_lane


def _place_final_movers(
    lane_map: np.ndarray,
    lanes: list[int],
    lane_modes: dict[int, str],
    stacked_lane: int,
    stationary_by_lane: dict[int, list[dict]],
) -> tuple[np.ndarray, dict[int, list[dict]]] | None:
    h, w = lane_map.shape
    global_occ = np.zeros((h, w), dtype=bool)
    movers_by_lane: dict[int, list[dict]] = {lane: [] for lane in lanes}
    local_occ: dict[int, np.ndarray] = {lane: np.zeros((h, w), dtype=bool) for lane in lanes}

    for lane in lanes:
        for comp in stationary_by_lane[lane]:
            coords = comp["coords"]
            global_occ[coords[:, 0], coords[:, 1]] = True
            local_occ[lane][coords[:, 0], coords[:, 1]] = True

    for lane in lanes:
        base_support = lane_modes[lane]
        mover_count = 2 if lane == stacked_lane else 1
        used_colors = {int(comp["color"]) for comp in stationary_by_lane[lane]}

        for idx in range(mover_count):
            shapes = MOVER_SHAPES[:]
            random.shuffle(shapes)
            placed = False
            for mask in shapes:
                if idx == 0:
                    support_mode = "bottom" if base_support == "bottom" else "platform"
                else:
                    support_mode = "platform"
                if idx == 0 and support_mode == "bottom":
                    row_lo = row_hi = h - mask.shape[0]
                elif idx == 0:
                    row_lo = max(1, h // 2 - 1)
                    row_hi = h - mask.shape[0]
                else:
                    row_lo = 1
                    row_hi = h - mask.shape[0]
                candidates = _placement_candidates(
                    mask,
                    lane_map,
                    lane,
                    global_occ,
                    row_lo=row_lo,
                    row_hi=row_hi,
                    support_occ=local_occ[lane],
                    support_mode=support_mode,
                )
                if not candidates:
                    continue
                coords = random.choice(candidates)
                color = _sample_color(used_colors if len(used_colors) < len(ACTIVE_COLORS) else None)
                comp = _component_from_coords(coords, color)
                movers_by_lane[lane].append(comp)
                global_occ[coords[:, 0], coords[:, 1]] = True
                local_occ[lane][coords[:, 0], coords[:, 1]] = True
                used_colors.add(color)
                placed = True
                break
            if not placed:
                return None
    return global_occ, movers_by_lane


def _lift_options(
    comp: dict,
    lane_map: np.ndarray,
    lane_id: int,
    blocked: np.ndarray,
    min_lift: int,
) -> list[tuple[int, np.ndarray]]:
    coords = comp["coords"]
    options: list[tuple[int, np.ndarray]] = []
    max_lift = int(coords[:, 0].min())
    for lift in range(int(min_lift), max_lift + 1):
        moved = coords - np.array([lift, 0], dtype=int)
        rows = moved[:, 0]
        cols = moved[:, 1]
        if np.any(lane_map[rows, cols] != lane_id):
            continue
        if bool(blocked[rows, cols].any()):
            continue
        if _touches_occupied(moved, blocked):
            continue
        options.append((lift, moved))
    return options


def _build_input_from_output(
    h: int,
    w: int,
    seed: tuple[int, int],
    lane_map: np.ndarray,
    lanes: list[int],
    stationary_by_lane: dict[int, list[dict]],
    movers_by_lane: dict[int, list[dict]],
) -> tuple[np.ndarray, np.ndarray] | None:
    inp = np.full((h, w), BG, dtype=int)
    out = np.full((h, w), BG, dtype=int)
    inp[seed] = SEED

    blocked_in = np.zeros((h, w), dtype=bool)
    for lane in lanes:
        for comp in stationary_by_lane[lane]:
            coords = comp["coords"]
            _paint(inp, coords, comp["color"])
            _paint(out, coords, comp["color"])
            blocked_in[coords[:, 0], coords[:, 1]] = True

    for lane in lanes:
        movers = movers_by_lane[lane]
        movers.sort(key=lambda comp: int(comp["bbox"][2]), reverse=True)
        prev_lift = 0
        for comp in movers:
            min_lift = max(1, prev_lift + 1)
            options = _lift_options(comp, lane_map, lane, blocked_in, min_lift)
            if not options:
                return None
            lift, moved = random.choice(options[: min(4, len(options))])
            prev_lift = lift
            _paint(inp, moved, comp["color"])
            blocked_in[moved[:, 0], moved[:, 1]] = True

        for comp in movers:
            _paint(out, comp["coords"], comp["color"])
    return inp, out


def _lane_summary(grid: np.ndarray) -> tuple[dict[int, int], dict[int, int]] | None:
    seed = _find_seed(grid)
    path = _ray_path(grid.shape, seed)
    lane_map = _build_lane_map(grid.shape, path)
    comps = _find_components(grid)
    if not comps:
        return None
    smalls = Counter()
    movers = Counter()
    for comp in comps:
        lane = _component_lane(comp, lane_map)
        if int(comp["area"]) <= SMALL_MAX_AREA:
            smalls[lane] += 1
        else:
            movers[lane] += 1
    return dict(smalls), dict(movers)


def _gen_attempt() -> np.ndarray | None:
    sampled = _sample_canvas()
    if sampled is None:
        return None
    h, w, seed, _path, lane_map = sampled
    lanes = sorted(int(x) for x in np.unique(lane_map) if int(x) >= 0)
    lane_sizes = {lane: int(np.sum(lane_map == lane)) for lane in lanes}
    ordered = sorted(lanes, key=lambda lane: lane_sizes[lane])
    bottom_lane = ordered[0]
    platform_lane = ordered[-1]
    lane_modes = {}
    for lane in lanes:
        if lane == bottom_lane:
            lane_modes[lane] = "bottom"
        elif lane == platform_lane:
            lane_modes[lane] = "platform"
        else:
            lane_modes[lane] = "platform" if lane_sizes[lane] >= 45 else "bottom"
    stacked_candidates = [lane for lane in lanes if lane != bottom_lane and lane_sizes[lane] >= 45]
    stacked_lane = random.choice(stacked_candidates or [platform_lane])

    stationary_result = _place_stationaries(lane_map, lanes, lane_modes)
    if stationary_result is None:
        return None
    _stationary_occ, stationary_by_lane = stationary_result

    movers_result = _place_final_movers(lane_map, lanes, lane_modes, stacked_lane, stationary_by_lane)
    if movers_result is None:
        return None
    _final_occ, movers_by_lane = movers_result

    built = _build_input_from_output(h, w, seed, lane_map, lanes, stationary_by_lane, movers_by_lane)
    if built is None:
        return None
    inp, expected = built

    summary = _lane_summary(inp)
    if summary is None:
        return None
    smalls, movers = summary
    if any(smalls.get(lane, 0) < 1 or movers.get(lane, 0) < 1 for lane in lanes):
        return None
    if sum(v >= 2 for v in movers.values()) == 0:
        return None

    actual = main(inp)
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid lane-gravity ARC input")
