from __future__ import annotations

import random
from collections import deque

import numpy as np


BLACK = 0
GREY = 5
SEED_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)

CARDINALS = ((-1, 0), (1, 0), (0, -1), (0, 1))
DIAGONALS = ((-1, -1), (-1, 1), (1, -1), (1, 1))

GRID_MIN = 18
GRID_MAX = 26
PURE_LENGTHS = (2, 3, 4, 5)
PURE_COUNT_CHOICES = (1, 2)
DISTRACTOR_COUNT_RANGE = (2, 4)
GEN_RETRIES = 400


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _neighbors(r: int, c: int, offsets: tuple[tuple[int, int], ...]) -> list[tuple[int, int]]:
    return [(r + dr, c + dc) for dr, dc in offsets]


def _connected_components(mask: np.ndarray, connectivity: int) -> list[list[tuple[int, int]]]:
    offsets = CARDINALS if connectivity == 4 else CARDINALS + DIAGONALS
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    components: list[list[tuple[int, int]]] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            queue = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while queue:
                rr, cc = queue.popleft()
                comp.append((rr, cc))
                for nr, nc in _neighbors(rr, cc, offsets):
                    if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            components.append(comp)
    return components


def _has_cardinal_contact(points: set[tuple[int, int]]) -> bool:
    for r, c in points:
        for nr, nc in _neighbors(r, c, CARDINALS):
            if (nr, nc) in points:
                return True
    return False


def _diag_degree_map(points: set[tuple[int, int]]) -> dict[tuple[int, int], int]:
    degrees: dict[tuple[int, int], int] = {}
    for r, c in points:
        degrees[(r, c)] = sum((nr, nc) in points for nr, nc in _neighbors(r, c, DIAGONALS))
    return degrees


def _walk_diagonal_path(points: set[tuple[int, int]], start: tuple[int, int]) -> list[tuple[int, int]]:
    ordered = [start]
    prev: tuple[int, int] | None = None
    current = start
    while True:
        nxts = [
            (nr, nc)
            for nr, nc in _neighbors(current[0], current[1], DIAGONALS)
            if (nr, nc) in points and (nr, nc) != prev
        ]
        if not nxts:
            break
        if len(nxts) != 1:
            return []
        nxt = nxts[0]
        ordered.append(nxt)
        prev, current = current, nxt
        if len(ordered) > len(points):
            return []
    return ordered


def _endpoint_seed_candidates(grid: np.ndarray, endpoint: tuple[int, int]) -> list[tuple[int, int, int]]:
    h, w = grid.shape
    out: list[tuple[int, int, int]] = []
    for nr, nc in _neighbors(endpoint[0], endpoint[1], DIAGONALS):
        if _in_bounds((h, w), nr, nc):
            color = int(grid[nr, nc])
            if color not in (BLACK, GREY):
                out.append((nr, nc, color))
    return out


def _analyze_grey_region(grid: np.ndarray, coords: list[tuple[int, int]]) -> dict[str, object]:
    points = set(coords)
    info: dict[str, object] = {
        "coords": sorted(points),
        "length": len(points),
        "pure_topology": False,
        "active": False,
        "reason": "unknown",
    }

    if len(points) < 2:
        info["reason"] = "too_short"
        return info

    if _has_cardinal_contact(points):
        info["reason"] = "edge_touch"
        return info

    degrees = _diag_degree_map(points)
    if any(degree == 0 or degree > 2 for degree in degrees.values()):
        info["reason"] = "bad_degree"
        return info

    endpoints = [cell for cell, degree in degrees.items() if degree == 1]
    if len(endpoints) != 2:
        info["reason"] = "not_single_chain"
        return info

    ordered = _walk_diagonal_path(points, endpoints[0])
    if len(ordered) != len(points) or set(ordered) != points:
        info["reason"] = "walk_failed"
        return info

    seed_map = {endpoint: _endpoint_seed_candidates(grid, endpoint) for endpoint in endpoints}
    if any(len(cands) > 1 for cands in seed_map.values()):
        info["pure_topology"] = True
        info["reason"] = "ambiguous_seed"
        return info

    seeded_endpoints = [endpoint for endpoint, cands in seed_map.items() if len(cands) == 1]
    if len(seeded_endpoints) != 1:
        info["pure_topology"] = True
        info["reason"] = "missing_or_double_seed"
        return info

    seed_endpoint = seeded_endpoints[0]
    far_endpoint = endpoints[0] if seed_endpoint == endpoints[1] else endpoints[1]
    ordered = _walk_diagonal_path(points, seed_endpoint)
    if len(ordered) != len(points):
        info["pure_topology"] = True
        info["reason"] = "orientation_failed"
        return info

    seed_r, seed_c, seed_color = seed_map[seed_endpoint][0]
    info.update(
        {
            "pure_topology": True,
            "active": True,
            "reason": "active_chain",
            "ordered": ordered,
            "endpoints": tuple(endpoints),
            "seed_endpoint": seed_endpoint,
            "far_endpoint": far_endpoint,
            "seed": (seed_r, seed_c),
            "seed_color": seed_color,
        }
    )
    return info


def _square_border_cells(
    center: tuple[int, int],
    radius: int,
    shape: tuple[int, int],
) -> set[tuple[int, int]]:
    cr, cc = center
    h, w = shape
    cells: set[tuple[int, int]] = set()
    for r in range(cr - radius, cr + radius + 1):
        for c in (cc - radius, cc + radius):
            if 0 <= r < h and 0 <= c < w:
                cells.add((r, c))
    for c in range(cc - radius, cc + radius + 1):
        for r in (cr - radius, cr + radius):
            if 0 <= r < h and 0 <= c < w:
                cells.add((r, c))
    return cells


def _transform(grid: np.ndarray) -> tuple[np.ndarray, list[dict[str, object]]]:
    grey_components = _connected_components(grid == GREY, connectivity=8)
    analyses = [_analyze_grey_region(grid, comp) for comp in grey_components]

    out = np.array(grid, copy=True)
    for info in analyses:
        if not bool(info["active"]):
            continue
        color = int(info["seed_color"])
        for r, c in info["ordered"]:
            out[r, c] = color
        for r, c in _square_border_cells(
            center=info["far_endpoint"],
            radius=int(info["length"]),
            shape=grid.shape,
        ):
            out[r, c] = color
    return out, analyses


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each seeded pure grey diagonal chain and explode a square outline from its far end.

    Input structure:
    - The grid has a black background, several grey 8-connected regions, and some non-grey single-pixel colored seeds.
    - A valid target region is a pure diagonal chain: its grey pixels touch only by diagonal corners and form one non-branching path.
    - Each valid chain has exactly one endpoint diagonally adjacent to one seed pixel; other grey regions may also have nearby seeds as distractors.

    Rule:
    - Inspect every grey 8-connected region and keep only the ones whose internal topology is a pure diagonal path with no side-touch contacts or branches.
    - If exactly one endpoint of such a chain is diagonally adjacent to a seed, repaint the whole chain with the seed color.
    - From the opposite endpoint, draw a hollow square border in that same color whose Chebyshev radius equals the chain length, clipping anything outside the grid; leave all non-pure grey regions and their seeds unchanged.

    Key insight:
    - The trigger is determined by topology, not by shape size or seed proximity alone: only a branch-free, side-touch-free diagonal path can conduct the color.
    - The seed chooses the near end, so the border center must be the far endpoint, and the chain length alone sets the blast radius.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    out, _ = _transform(g)
    return out


def _normalize(points: list[tuple[int, int]]) -> list[tuple[int, int]]:
    min_r = min(r for r, _ in points)
    min_c = min(c for _, c in points)
    return sorted((r - min_r, c - min_c) for r, c in points)


def _transform_points(
    points: list[tuple[int, int]],
    *,
    rotate_k: int,
    flip: bool,
) -> list[tuple[int, int]]:
    pts = _normalize(points)
    height = max(r for r, _ in pts) + 1
    width = max(c for _, c in pts) + 1
    for _ in range(rotate_k % 4):
        pts = [(c, height - 1 - r) for r, c in pts]
        height, width = width, height
        pts = _normalize(pts)
    if flip:
        pts = [(r, width - 1 - c) for r, c in pts]
    return _normalize(pts)


def _transform_groups(
    groups: list[list[tuple[int, int]]],
    *,
    rotate_k: int,
    flip: bool,
) -> list[list[tuple[int, int]]]:
    all_points = [pt for group in groups for pt in group]
    min_r = min(r for r, _ in all_points)
    min_c = min(c for _, c in all_points)
    max_r = max(r for r, _ in all_points)
    max_c = max(c for _, c in all_points)
    height = max_r - min_r + 1
    width = max_c - min_c + 1
    work = [[(r - min_r, c - min_c) for r, c in group] for group in groups]

    for _ in range(rotate_k % 4):
        work = [[(c, height - 1 - r) for r, c in group] for group in work]
        height, width = width, height
    if flip:
        work = [[(r, width - 1 - c) for r, c in group] for group in work]

    return work


def _random_chain_path(length: int) -> list[tuple[int, int]]:
    assert length >= 2
    for _ in range(200):
        path = [(0, 0)]
        used = {(0, 0)}
        while len(path) < length:
            current = path[-1]
            candidates = []
            dirs = list(DIAGONALS)
            random.shuffle(dirs)
            for dr, dc in dirs:
                nxt = (current[0] + dr, current[1] + dc)
                if nxt in used:
                    continue
                if any(abs(nxt[0] - r) + abs(nxt[1] - c) == 1 for r, c in used):
                    continue
                diag_hits = [(r, c) for r, c in used if abs(nxt[0] - r) == 1 and abs(nxt[1] - c) == 1]
                if diag_hits != [current]:
                    continue
                candidates.append(nxt)
            if not candidates:
                break
            nxt = random.choice(candidates)
            used.add(nxt)
            path.append(nxt)
        if len(path) == length:
            return _normalize(path)
    raise RuntimeError("failed to build pure diagonal chain")


def _choose_seed_site(points: set[tuple[int, int]]) -> tuple[int, int] | None:
    preferred: list[tuple[int, int]] = []
    fallback: list[tuple[int, int]] = []
    for r, c in points:
        for sr, sc in _neighbors(r, c, DIAGONALS):
            if (sr, sc) in points:
                continue
            diag_hits = sum((nr, nc) in points for nr, nc in _neighbors(sr, sc, DIAGONALS))
            card_hits = sum((nr, nc) in points for nr, nc in _neighbors(sr, sc, CARDINALS))
            if diag_hits == 1 and card_hits == 0:
                preferred.append((sr, sc))
            elif diag_hits >= 1:
                fallback.append((sr, sc))
    if preferred:
        return random.choice(preferred)
    if fallback:
        return random.choice(fallback)
    return None


def _make_pure_chain_object(color: int) -> dict[str, object]:
    length = random.choice(PURE_LENGTHS)
    path = _random_chain_path(length)
    if random.random() < 0.5:
        path = list(reversed(path))
    seed_endpoint = path[0]
    next_point = path[1]
    seed = (
        2 * seed_endpoint[0] - next_point[0],
        2 * seed_endpoint[1] - next_point[1],
    )
    path, seed_group = _transform_groups(
        [path, [seed]],
        rotate_k=random.randrange(4),
        flip=bool(random.getrandbits(1)),
    )
    seed = seed_group[0]
    far_endpoint = path[-1]
    return {
        "kind": "pure",
        "grey": path,
        "seeds": [(seed, color)],
        "far": far_endpoint,
        "length": len(path),
    }


def _make_branch_distractor() -> list[tuple[int, int]]:
    options = [
        [(1, 1), (0, 0), (0, 2), (2, 0)],
        [(1, 1), (0, 0), (0, 2), (2, 0), (2, 2)],
    ]
    return _transform_points(
        random.choice(options),
        rotate_k=random.randrange(4),
        flip=bool(random.getrandbits(1)),
    )


def _make_side_touch_distractor() -> list[tuple[int, int]]:
    options = [
        [(0, 0), (1, 0), (2, 0), (2, 1)],
        [(0, 0), (1, 1), (2, 2), (2, 1)],
        [(0, 0), (0, 1), (1, 1), (2, 2)],
        [(0, 0), (0, 1), (1, 0), (1, 1)],
    ]
    return _transform_points(
        random.choice(options),
        rotate_k=random.randrange(4),
        flip=bool(random.getrandbits(1)),
    )


def _make_distractor_object(kind: str, color: int | None) -> dict[str, object]:
    if kind == "branch":
        grey = _make_branch_distractor()
    elif kind == "side_touch":
        grey = _make_side_touch_distractor()
    else:
        raise ValueError(f"unknown distractor kind: {kind}")

    seeds: list[tuple[tuple[int, int], int]] = []
    if color is not None:
        site = _choose_seed_site(set(grey))
        if site is None:
            raise RuntimeError("failed to attach distractor seed")
        grey, seed_group = _transform_groups([grey, [site]], rotate_k=0, flip=False)
        seeds = [(seed_group[0], color)]
    return {"kind": kind, "grey": grey, "seeds": seeds}


def _translate_cells(cells: list[tuple[int, int]], dr: int, dc: int) -> list[tuple[int, int]]:
    return [(r + dr, c + dc) for r, c in cells]


def _translated_object(
    obj: dict[str, object],
    dr: int,
    dc: int,
) -> dict[str, object]:
    grey = _translate_cells(list(obj["grey"]), dr, dc)
    seeds = [((r + dr, c + dc), color) for (r, c), color in obj["seeds"]]
    out = {"kind": obj["kind"], "grey": grey, "seeds": seeds}
    if "far" in obj:
        far_r, far_c = obj["far"]
        out["far"] = (far_r + dr, far_c + dc)
        out["length"] = obj["length"]
    return out


def _occupied_cells(obj: dict[str, object]) -> set[tuple[int, int]]:
    cells = set(obj["grey"])
    for cell, _color in obj["seeds"]:
        cells.add(cell)
    return cells


def _separated_from_existing(
    cells: set[tuple[int, int]],
    occupied: set[tuple[int, int]],
    forbidden: set[tuple[int, int]],
) -> bool:
    if cells & forbidden:
        return False
    for r, c in cells:
        for rr in range(r - 1, r + 2):
            for cc in range(c - 1, c + 2):
                if (rr, cc) in occupied:
                    return False
    return True


def _clip_status(center: tuple[int, int], radius: int, shape: tuple[int, int]) -> bool:
    r, c = center
    h, w = shape
    return r - radius < 0 or c - radius < 0 or r + radius >= h or c + radius >= w


def _candidate_translations(
    obj: dict[str, object],
    shape: tuple[int, int],
) -> list[tuple[int, int]]:
    all_local = list(_occupied_cells(obj))
    min_r = min(r for r, _ in all_local)
    max_r = max(r for r, _ in all_local)
    min_c = min(c for _, c in all_local)
    max_c = max(c for _, c in all_local)
    h, w = shape
    offsets = [
        (dr, dc)
        for dr in range(-min_r, h - max_r)
        for dc in range(-min_c, w - max_c)
    ]
    random.shuffle(offsets)
    return offsets


def _place_object(
    grid: np.ndarray,
    obj: dict[str, object],
    occupied: set[tuple[int, int]],
    forbidden: set[tuple[int, int]],
    *,
    want_clip: bool | None,
) -> dict[str, object] | None:
    for dr, dc in _candidate_translations(obj, grid.shape):
        placed = _translated_object(obj, dr, dc)
        cells = _occupied_cells(placed)
        if not _separated_from_existing(cells, occupied, forbidden):
            continue
        border_cells: set[tuple[int, int]] = set()
        if placed["kind"] == "pure":
            border_cells = _square_border_cells(placed["far"], int(placed["length"]), grid.shape)
            if want_clip is not None and _clip_status(placed["far"], int(placed["length"]), grid.shape) != want_clip:
                continue
            if len(border_cells) < 4:
                continue
            if border_cells & occupied or border_cells & forbidden:
                continue
        return placed
    return None


def _paint_object(grid: np.ndarray, obj: dict[str, object]) -> None:
    for r, c in obj["grey"]:
        grid[r, c] = GREY
    for (r, c), color in obj["seeds"]:
        grid[r, c] = color


def _validate_case(grid: np.ndarray) -> tuple[bool, str]:
    if not isinstance(grid, np.ndarray) or grid.ndim != 2 or max(grid.shape) > 30:
        return False, "bad_shape"

    colored_mask = (grid != BLACK) & (grid != GREY)
    colored_components = _connected_components(colored_mask, connectivity=8)
    if any(len(comp) != 1 for comp in colored_components):
        return False, "non_singleton_seed"

    out, analyses = _transform(grid)
    if np.array_equal(out, grid):
        return False, "no_effect"

    pure_infos = [info for info in analyses if bool(info["active"])]
    non_pure_infos = [info for info in analyses if not bool(info["active"])]
    if not pure_infos or not non_pure_infos:
        return False, "missing_branch_coverage"

    non_pure_seeded = 0
    for info in pure_infos:
        color = int(info["seed_color"])
        for r, c in info["coords"]:
            if out[r, c] != color:
                return False, "pure_chain_not_recolored"
        border = _square_border_cells(info["far_endpoint"], int(info["length"]), grid.shape)
        if len(border) < 4:
            return False, "border_too_small"
        if any(out[r, c] != color for r, c in border):
            return False, "border_not_drawn"
        if not border - set(info["coords"]):
            return False, "degenerate_border"

    for info in non_pure_infos:
        for r, c in info["coords"]:
            if out[r, c] != GREY:
                return False, "non_pure_changed"
        endpoints = []
        if info.get("endpoints") is not None:
            endpoints = list(info["endpoints"])
        else:
            endpoints = list(info["coords"])
        seeded_here = False
        for cell in endpoints:
            for sr, sc, color in _endpoint_seed_candidates(grid, cell):
                seeded_here = True
                if out[sr, sc] != color:
                    return False, "distractor_seed_changed"
        non_pure_seeded += int(seeded_here)

    if non_pure_seeded == 0:
        return False, "no_seeded_distractor"

    reasons = {str(info["reason"]) for info in non_pure_infos}
    if "edge_touch" not in reasons:
        return False, "missing_edge_touch"
    if not any(reason in {"bad_degree", "not_single_chain"} for reason in reasons):
        return False, "missing_branch_distractor"

    return True, "ok"


def _build_attempt() -> np.ndarray | None:
    height = random.randint(GRID_MIN, GRID_MAX)
    width = random.randint(GRID_MIN, GRID_MAX)
    grid = np.full((height, width), BLACK, dtype=int)

    occupied: set[tuple[int, int]] = set()
    forbidden: set[tuple[int, int]] = set()

    pure_count = random.choice(PURE_COUNT_CHOICES)
    distractor_count = random.randint(*DISTRACTOR_COUNT_RANGE)

    pure_clip_flags = [random.random() < 0.4 for _ in range(pure_count)]
    if pure_count == 2 and pure_clip_flags[0] == pure_clip_flags[1]:
        pure_clip_flags[random.randrange(2)] = not pure_clip_flags[0]

    available_colors = list(SEED_COLORS)
    random.shuffle(available_colors)

    placed_objects: list[dict[str, object]] = []

    for want_clip in pure_clip_flags:
        color = available_colors.pop() if available_colors else random.choice(SEED_COLORS)
        raw = _make_pure_chain_object(color)
        placed = _place_object(grid, raw, occupied, forbidden, want_clip=want_clip)
        if placed is None:
            placed = _place_object(grid, raw, occupied, forbidden, want_clip=None)
        if placed is None:
            return None
        _paint_object(grid, placed)
        occupied |= _occupied_cells(placed)
        forbidden |= _square_border_cells(placed["far"], int(placed["length"]), grid.shape)
        placed_objects.append(placed)

    distractor_kinds = ["branch", "side_touch"]
    while len(distractor_kinds) < distractor_count:
        distractor_kinds.append(random.choice(("branch", "side_touch")))
    random.shuffle(distractor_kinds)

    seeded_slots = {0}
    if distractor_count > 2 and random.random() < 0.7:
        seeded_slots.add(random.randrange(distractor_count))

    for idx, kind in enumerate(distractor_kinds):
        color = None
        if idx in seeded_slots:
            color = available_colors.pop() if available_colors else random.choice(SEED_COLORS)
        raw = _make_distractor_object(kind, color)
        placed = _place_object(grid, raw, occupied, forbidden, want_clip=None)
        if placed is None:
            return None
        _paint_object(grid, placed)
        occupied |= _occupied_cells(placed)
        placed_objects.append(placed)

    ok, _reason = _validate_case(grid)
    if not ok:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _build_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid diagonal-fuse input")
