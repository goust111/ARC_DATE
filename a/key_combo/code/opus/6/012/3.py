from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
PATH = 5
SEED_COLORS = (1, 2, 3, 4, 6)
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
SIDES = ("N", "S", "W", "E")


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _label_components(mask: np.ndarray) -> list[np.ndarray]:
    """Return 4-connected components as arrays of (r, c) coordinates."""
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                pts.append((rr, cc))
                for dr, dc in DIR4:
                    nr, nc = rr + dr, cc + dc
                    if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _rect_sum(ps: np.ndarray, r0: int, c0: int, r1: int, c1: int) -> int:
    total = int(ps[r1, c1])
    if r0 > 0:
        total -= int(ps[r0 - 1, c1])
    if c0 > 0:
        total -= int(ps[r1, c0 - 1])
    if r0 > 0 and c0 > 0:
        total += int(ps[r0 - 1, c0 - 1])
    return total


def _find_hub_square(network_mask: np.ndarray) -> tuple[int, int, int]:
    """Find the largest solid square of PATH pixels inside the network mask."""
    h, w = network_mask.shape
    ps = network_mask.astype(int).cumsum(axis=0).cumsum(axis=1)
    best_size = 0
    best_pos: tuple[int, int] | None = None
    for size in range(min(h, w), 1, -1):
        found: list[tuple[int, int]] = []
        area = size * size
        for r0 in range(h - size + 1):
            r1 = r0 + size - 1
            for c0 in range(w - size + 1):
                c1 = c0 + size - 1
                if _rect_sum(ps, r0, c0, r1, c1) == area:
                    found.append((r0, c0))
        if found:
            best_size = size
            best_pos = found[0]
            break
    if best_pos is None:
        raise ValueError("no solid hub square found")
    return best_size, best_pos[0], best_pos[1]


def _seed_port(seed: dict, code: str) -> tuple[int, int]:
    r0, c0, r1, c1 = seed["bbox"]
    rm = (r0 + r1) // 2
    cm = (c0 + c1) // 2
    if code == "N":
        return int(r0 - 1), int(cm)
    if code == "S":
        return int(r1 + 1), int(cm)
    if code == "W":
        return int(rm), int(c0 - 1)
    if code == "E":
        return int(rm), int(c1 + 1)
    raise ValueError(code)


def _hub_side_for(seed_side: str, port_code: str) -> str:
    mapping = {
        "N": {"S": "N", "W": "W", "E": "E"},
        "S": {"N": "S", "W": "W", "E": "E"},
        "W": {"E": "W", "N": "N", "S": "S"},
        "E": {"W": "E", "N": "N", "S": "S"},
    }
    return mapping[seed_side][port_code]


def _port_options(seed_side: str, count: int) -> list[tuple[str, ...]]:
    table = {
        "N": {1: [("S",), ("W",), ("E",)], 2: [("S", "W"), ("S", "E"), ("W", "E")], 3: [("S", "W", "E")]},
        "S": {1: [("N",), ("W",), ("E",)], 2: [("N", "W"), ("N", "E"), ("W", "E")], 3: [("N", "W", "E")]},
        "W": {1: [("E",), ("N",), ("S",)], 2: [("E", "N"), ("E", "S"), ("N", "S")], 3: [("E", "N", "S")]},
        "E": {1: [("W",), ("N",), ("S",)], 2: [("W", "N"), ("W", "S"), ("N", "S")], 3: [("W", "N", "S")]},
    }
    return table[seed_side][count]


def _extract_structure(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape

    path_components = _label_components(g == PATH)
    if not path_components:
        raise ValueError("missing path network")
    network_coords = max(path_components, key=lambda pts: int(pts.shape[0]))
    network_mask = np.zeros((h, w), dtype=bool)
    network_mask[network_coords[:, 0], network_coords[:, 1]] = True

    hub_size, hub_r0, hub_c0 = _find_hub_square(network_mask)
    hub_r1 = hub_r0 + hub_size - 1
    hub_c1 = hub_c0 + hub_size - 1
    hub_mask = np.zeros((h, w), dtype=bool)
    hub_mask[hub_r0 : hub_r1 + 1, hub_c0 : hub_c1 + 1] = True

    branch_mask = network_mask & ~hub_mask
    branch_components = _label_components(branch_mask)

    seed_id_grid = -np.ones((h, w), dtype=int)
    seeds: list[dict] = []
    for color in SEED_COLORS:
        for coords in _label_components(g == color):
            r0, c0, r1, c1 = _bbox(coords)
            hh = r1 - r0 + 1
            ww = c1 - c0 + 1
            if hh not in (2, 3) or ww not in (2, 3) or int(coords.shape[0]) != hh * ww:
                raise ValueError("seed region must be a solid 2x2..3x3 rectangle")
            sid = len(seeds)
            seed_id_grid[coords[:, 0], coords[:, 1]] = sid
            seeds.append(
                {
                    "id": sid,
                    "color": int(color),
                    "coords": coords,
                    "bbox": (r0, c0, r1, c1),
                    "centroid_row": (r0 + r1) / 2.0,
                    "centroid_col": (c0 + c1) / 2.0,
                }
            )
    if len(seeds) < 2:
        raise ValueError("need at least two seed regions")

    for seed in seeds:
        for r, c in seed["coords"]:
            for dr, dc in DIR4:
                nr, nc = int(r + dr), int(c + dc)
                if _in_bounds(g.shape, nr, nc) and hub_mask[nr, nc]:
                    raise ValueError("hub may not touch seeds directly")

    for i, seed_a in enumerate(seeds):
        for seed_b in seeds[i + 1 :]:
            for r, c in seed_a["coords"]:
                for dr, dc in DIR4:
                    nr, nc = int(r + dr), int(c + dc)
                    if _in_bounds(g.shape, nr, nc) and seed_id_grid[nr, nc] == seed_b["id"]:
                        raise ValueError("seed regions may not touch")

    branch_to_seed: dict[int, int] = {}
    seed_to_branch_ids = {seed["id"]: [] for seed in seeds}
    for bid, coords in enumerate(branch_components):
        touches: set[int] = set()
        hub_contacts = 0
        seed_contact_cells: set[tuple[int, int]] = set()
        branch_local = np.zeros((h, w), dtype=bool)
        branch_local[coords[:, 0], coords[:, 1]] = True
        for r, c in coords:
            deg = 0
            for dr, dc in DIR4:
                nr, nc = int(r + dr), int(c + dc)
                if not _in_bounds(g.shape, nr, nc):
                    continue
                if branch_local[nr, nc]:
                    deg += 1
                sid = int(seed_id_grid[nr, nc])
                if sid >= 0:
                    touches.add(sid)
                    seed_contact_cells.add((int(r), int(c)))
                if hub_mask[nr, nc]:
                    hub_contacts += 1
            if deg > 2:
                raise ValueError("branch component is not a simple segment")
        if hub_contacts == 0:
            raise ValueError("every branch must connect to the hub")
        if len(touches) != 1:
            raise ValueError("every branch must touch exactly one seed")
        sid = next(iter(touches))
        branch_to_seed[bid] = sid
        seed_to_branch_ids[sid].append(bid)
        if not seed_contact_cells:
            raise ValueError("branch must touch seed")

    for seed in seeds:
        seed["count"] = len(seed_to_branch_ids[seed["id"]])
        if seed["count"] == 0:
            raise ValueError("every seed must be touched by a branch")

    seeds_sorted = sorted(
        seeds,
        key=lambda s: (int(s["count"]), float(s["centroid_row"]), float(s["centroid_col"])),
    )
    palette = [int(seed["color"]) for seed in seeds_sorted]

    return {
        "grid": g,
        "hub_bbox": (hub_r0, hub_c0, hub_r1, hub_c1),
        "hub_size": hub_size,
        "hub_mask": hub_mask,
        "branch_components": branch_components,
        "branch_to_seed": branch_to_seed,
        "seeds": seeds,
        "seeds_sorted": seeds_sorted,
        "palette": palette,
    }


def _transform_from_structure(meta: dict) -> np.ndarray:
    g = meta["grid"]
    out = g.copy()
    palette = meta["palette"]
    r0, c0, r1, c1 = meta["hub_bbox"]

    for seed in meta["seeds"]:
        pass

    for bid, coords in enumerate(meta["branch_components"]):
        seed_id = meta["branch_to_seed"][bid]
        color = int(meta["seeds"][seed_id]["color"])
        out[coords[:, 0], coords[:, 1]] = color

    plen = len(palette)
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            d = min(r - r0, r1 - r, c - c0, c1 - c)
            out[r, c] = int(palette[d % plen])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor a hub-and-branches network using seed-touch counts to drive concentric hub filling.

    Input structure:
    - Black background surrounds a thin grey orthogonal path network.
    - The network contains one solid grey square hub and several grey branch segments obtained by cutting at the hub boundary.
    - Each branch touches exactly one colored rectangular seed region, while a seed may be touched by multiple branches.

    Rule:
    - Find the largest grey connected component and its largest solid grey square hub.
    - Remove the hub, split the remaining grey pixels into independent 4-connected branch segments, and count how many segments touch each seed.
    - Sort seed colors by increasing touched-branch count, breaking ties by seed centroid row, to form a palette.
    - Recolor every branch with the color of the unique seed it touches, then fill the hub by concentric square layers using palette[distance_to_hub_boundary mod palette_length].

    Key insight:
    - The hidden topological count of how many branch segments reach each seed simultaneously determines the branch recoloring identities and the cyclic palette order for the hub.
    """
    meta = _extract_structure(np.asarray(grid, dtype=int))
    return _transform_from_structure(meta)


def _choose_profile() -> tuple[int, int, tuple[int, ...]]:
    hub_size = random.choice([4, 5])
    if hub_size == 4:
        seed_count = 2
        branch_count = random.choice([3, 4, 5])
        count_options = {3: [(1, 2)], 4: [(1, 3), (2, 2)], 5: [(2, 3)]}[branch_count]
    else:
        seed_count = random.choices([2, 3], weights=[2, 3])[0]
        if seed_count == 2:
            branch_count = random.choice([3, 4, 5])
            count_options = {3: [(1, 2)], 4: [(1, 3), (2, 2)], 5: [(2, 3)]}[branch_count]
        else:
            branch_count = random.choices([4, 5, 3], weights=[4, 4, 1])[0]
            count_options = {3: [(1, 1, 1)], 4: [(2, 1, 1)], 5: [(3, 1, 1), (2, 2, 1)]}[branch_count]
    counts = list(random.choice(count_options))
    random.shuffle(counts)
    return hub_size, seed_count, tuple(counts)


def _place_seed_bbox(
    side: str,
    count: int,
    hub_bbox: tuple[int, int, int, int],
    shape: tuple[int, int],
) -> tuple[int, int, int, int] | None:
    h, w = shape
    hr0, hc0, hr1, hc1 = hub_bbox
    gap = random.randint(3, 5)
    if count > 1:
        sh = sw = 3
    else:
        sh = random.choice([2, 3])
        sw = random.choice([2, 3])

    if side == "N":
        max_r0 = hr0 - gap - sh
        lo_c = max(1, hc0 - 5)
        hi_c = min(w - sw - 2, hc1 + 2)
        if max_r0 < 1 or lo_c > hi_c:
            return None
        r0 = random.randint(1, max_r0)
        c0 = random.randint(lo_c, hi_c)
    elif side == "S":
        min_r0 = hr1 + gap + 1
        lo_c = max(1, hc0 - 5)
        hi_c = min(w - sw - 2, hc1 + 2)
        if min_r0 > h - sh - 2 or lo_c > hi_c:
            return None
        r0 = random.randint(min_r0, h - sh - 2)
        c0 = random.randint(lo_c, hi_c)
    elif side == "W":
        max_c0 = hc0 - gap - sw
        lo_r = max(1, hr0 - 5)
        hi_r = min(h - sh - 2, hr1 + 2)
        if max_c0 < 1 or lo_r > hi_r:
            return None
        r0 = random.randint(lo_r, hi_r)
        c0 = random.randint(1, max_c0)
    else:
        min_c0 = hc1 + gap + 1
        lo_r = max(1, hr0 - 5)
        hi_r = min(h - sh - 2, hr1 + 2)
        if min_c0 > w - sw - 2 or lo_r > hi_r:
            return None
        r0 = random.randint(lo_r, hi_r)
        c0 = random.randint(min_c0, w - sw - 2)
    return r0, c0, r0 + sh - 1, c0 + sw - 1


def _expand_bbox(bbox: tuple[int, int, int, int], margin: int = 1) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = bbox
    return r0 - margin, c0 - margin, r1 + margin, c1 + margin


def _bboxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _choose_port_assignment(seed_specs: list[dict], hub_size: int) -> dict[int, tuple[str, ...]] | None:
    capacity = 2 if hub_size == 4 else 3
    order = sorted(range(len(seed_specs)), key=lambda i: (-seed_specs[i]["count"], seed_specs[i]["side"]))
    assignments: dict[int, tuple[str, ...]] = {}
    demand = {side: 0 for side in SIDES}

    def backtrack(pos: int) -> bool:
        if pos == len(order):
            return True
        idx = order[pos]
        spec = seed_specs[idx]
        options = _port_options(spec["side"], spec["count"])
        random.shuffle(options)
        for ports in options:
            added = {side: 0 for side in SIDES}
            for port_code in ports:
                added[_hub_side_for(spec["side"], port_code)] += 1
            if any(demand[side] + added[side] > capacity for side in SIDES):
                continue
            for side in SIDES:
                demand[side] += added[side]
            assignments[idx] = ports
            if backtrack(pos + 1):
                return True
            del assignments[idx]
            for side in SIDES:
                demand[side] -= added[side]
        return False

    if backtrack(0):
        return assignments
    return None


def _side_start_offsets(hub_size: int, needed: int) -> list[int]:
    if hub_size == 4:
        table = {1: [1], 2: [0, 3]}
    else:
        table = {1: [2], 2: [0, 4], 3: [0, 2, 4]}
    return table[needed]


def _start_cell(hub_bbox: tuple[int, int, int, int], hub_side: str, offset: int) -> tuple[int, int]:
    r0, c0, r1, c1 = hub_bbox
    if hub_side == "N":
        return r0 - 1, c0 + offset
    if hub_side == "S":
        return r1 + 1, c0 + offset
    if hub_side == "W":
        return r0 + offset, c0 - 1
    return r0 + offset, c1 + 1


def _sort_key_for_side(side: str, target: tuple[int, int]) -> tuple[int, int]:
    if side in ("N", "S"):
        return int(target[1]), int(target[0])
    return int(target[0]), int(target[1])


def _route_branch(
    blocked: np.ndarray,
    path_mask: np.ndarray,
    hub_mask: np.ndarray,
    seed_id_grid: np.ndarray,
    start: tuple[int, int],
    target: tuple[int, int],
    target_seed: int,
) -> list[tuple[int, int]] | None:
    h, w = blocked.shape
    q = deque([start])
    prev = {start: None}
    while q:
        cur = q.popleft()
        if cur == target:
            path: list[tuple[int, int]] = []
            node: tuple[int, int] | None = cur
            while node is not None:
                path.append(node)
                node = prev[node]
            path.reverse()
            local = set(path)
            for r, c in path:
                deg = 0
                for dr, dc in DIR4:
                    nr, nc = r + dr, c + dc
                    if (nr, nc) in local:
                        deg += 1
                if deg > 2:
                    return None
            return path

        dirs = DIR4[:]
        random.shuffle(dirs)
        dirs.sort(key=lambda d: abs((cur[0] + d[0]) - target[0]) + abs((cur[1] + d[1]) - target[1]))
        for dr, dc in dirs:
            nr, nc = cur[0] + dr, cur[1] + dc
            nxt = (nr, nc)
            if not _in_bounds((h, w), nr, nc):
                continue
            if nxt in prev:
                continue
            if blocked[nr, nc]:
                continue

            safe = True
            for ar, ac in ((nr - 1, nc), (nr + 1, nc), (nr, nc - 1), (nr, nc + 1)):
                if not _in_bounds((h, w), ar, ac):
                    continue
                if (ar, ac) == cur:
                    continue
                if path_mask[ar, ac]:
                    safe = False
                    break
                if hub_mask[ar, ac] and nxt != start:
                    safe = False
                    break
                if seed_id_grid[ar, ac] >= 0:
                    if nxt != target or int(seed_id_grid[ar, ac]) != target_seed:
                        safe = False
                        break
            if not safe:
                continue
            prev[nxt] = cur
            q.append(nxt)
    return None


def _build_input() -> tuple[np.ndarray, dict] | None:
    hub_size, seed_count, counts = _choose_profile()
    h = random.randint(20, 28)
    w = random.randint(20, 28)

    row_lo = max(5, h // 2 - 2)
    row_hi = min(h - hub_size - 6, h // 2 + 1)
    col_lo = max(5, w // 2 - 2)
    col_hi = min(w - hub_size - 6, w // 2 + 1)
    if row_lo > row_hi or col_lo > col_hi:
        return None
    hub_r0 = random.randint(row_lo, row_hi)
    hub_c0 = random.randint(col_lo, col_hi)
    hub_bbox = (hub_r0, hub_c0, hub_r0 + hub_size - 1, hub_c0 + hub_size - 1)

    sides = random.sample(list(SIDES), seed_count)
    colors = random.sample(list(SEED_COLORS), seed_count)
    seed_specs: list[dict] = []
    placed_boxes: list[tuple[int, int, int, int]] = []
    for side, count, color in zip(sides, counts, colors):
        bbox = _place_seed_bbox(side, count, hub_bbox, (h, w))
        if bbox is None:
            return None
        expanded = _expand_bbox(bbox, margin=1)
        for other in placed_boxes:
            if _bboxes_overlap(expanded, _expand_bbox(other, margin=1)):
                return None
        placed_boxes.append(bbox)
        seed_specs.append({"side": side, "count": int(count), "color": int(color), "bbox": bbox})

    for i, spec_a in enumerate(seed_specs):
        for spec_b in seed_specs[i + 1 :]:
            if spec_a["count"] == spec_b["count"]:
                row_a = (spec_a["bbox"][0] + spec_a["bbox"][2]) / 2.0
                row_b = (spec_b["bbox"][0] + spec_b["bbox"][2]) / 2.0
                if abs(row_a - row_b) < 0.5:
                    return None

    port_assignment = _choose_port_assignment(seed_specs, hub_size)
    if port_assignment is None:
        return None

    hub_mask = np.zeros((h, w), dtype=bool)
    hr0, hc0, hr1, hc1 = hub_bbox
    hub_mask[hr0 : hr1 + 1, hc0 : hc1 + 1] = True

    g = np.zeros((h, w), dtype=int)
    g[hub_mask] = PATH
    seed_id_grid = -np.ones((h, w), dtype=int)
    for sid, spec in enumerate(seed_specs):
        r0, c0, r1, c1 = spec["bbox"]
        g[r0 : r1 + 1, c0 : c1 + 1] = int(spec["color"])
        seed_id_grid[r0 : r1 + 1, c0 : c1 + 1] = sid

    requests: list[dict] = []
    for sid, spec in enumerate(seed_specs):
        seed = {"bbox": spec["bbox"]}
        for port_code in port_assignment[sid]:
            requests.append(
                {
                    "seed_id": sid,
                    "color": int(spec["color"]),
                    "seed_side": spec["side"],
                    "hub_side": _hub_side_for(spec["side"], port_code),
                    "target": _seed_port(seed, port_code),
                }
            )

    for req in requests:
        tr, tc = req["target"]
        if not _in_bounds((h, w), tr, tc) or g[tr, tc] != BG:
            return None

    grouped: dict[str, list[dict]] = {side: [] for side in SIDES}
    for req in requests:
        grouped[req["hub_side"]].append(req)

    for side in SIDES:
        if not grouped[side]:
            continue
        grouped[side].sort(key=lambda req: _sort_key_for_side(side, req["target"]))
        offsets = _side_start_offsets(hub_size, len(grouped[side]))
        for req, offset in zip(grouped[side], offsets):
            req["start"] = _start_cell(hub_bbox, side, offset)

    blocked = np.zeros((h, w), dtype=bool)
    blocked[hub_mask] = True
    blocked[seed_id_grid >= 0] = True
    path_mask = np.zeros((h, w), dtype=bool)

    requests.sort(
        key=lambda req: (
            -abs(req["start"][0] - req["target"][0]) - abs(req["start"][1] - req["target"][1]),
            req["hub_side"],
        )
    )

    for req in requests:
        path = _route_branch(blocked, path_mask, hub_mask, seed_id_grid, req["start"], req["target"], req["seed_id"])
        if path is None:
            return None
        for r, c in path:
            g[r, c] = PATH
            blocked[r, c] = True
            path_mask[r, c] = True

    parsed = _extract_structure(g)
    parsed_counts = sorted(int(seed["count"]) for seed in parsed["seeds"])
    if parsed["hub_size"] != hub_size or parsed_counts != sorted(counts):
        return None

    out = _transform_from_structure(parsed)
    if np.array_equal(g, out):
        return None

    hub_colors = set(int(out[r, c]) for r in range(parsed["hub_bbox"][0], parsed["hub_bbox"][2] + 1) for c in range(parsed["hub_bbox"][1], parsed["hub_bbox"][3] + 1))
    if len(parsed["palette"]) == 2 and len(hub_colors) != 2:
        return None
    if len(parsed["palette"]) == 3 and parsed["hub_size"] == 5 and len(hub_colors) != 3:
        return None

    count_hist = sorted(seed["count"] for seed in parsed["seeds"])
    if hub_size == 4 and len(parsed["seeds"]) != 2:
        return None
    if hub_size == 5 and len(parsed["seeds"]) not in (2, 3):
        return None
    if count_hist == [1, 1, 1]:
        return None

    return g, {"hub_size": hub_size, "counts": counts}


def generate_input() -> np.ndarray:
    for _ in range(600):
        built = _build_input()
        if built is not None:
            return built[0]
    raise RuntimeError("failed to generate a valid puzzle instance")
