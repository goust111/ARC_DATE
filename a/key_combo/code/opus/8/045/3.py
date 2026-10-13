from __future__ import annotations

import random
from collections import deque

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREY = 5

N4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
N8 = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0)]
DIAGONALS = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
RING_OFFSETS = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
]
PLUS_OFFSETS = [(-1, 0), (0, -1), (0, 0), (0, 1), (1, 0)]


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _components_from_mask(mask: np.ndarray, connectivity: int) -> list[np.ndarray]:
    nbrs = N4 if connectivity == 4 else N8
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    parts: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for dr, dc in nbrs:
                    nr, nc = rr + dr, cc + dc
                    if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            parts.append(np.array(coords, dtype=int))
    return parts


def _coords_to_mask(coords: np.ndarray) -> tuple[np.ndarray, int, int]:
    r0 = int(coords[:, 0].min())
    c0 = int(coords[:, 1].min())
    r1 = int(coords[:, 0].max())
    c1 = int(coords[:, 1].max())
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return mask, r0, c0


def _is_diagonal_chain_region(coords: np.ndarray) -> bool:
    mask, r0, c0 = _coords_to_mask(coords)
    lobes = _components_from_mask(mask, connectivity=4)
    if len(lobes) < 2:
        return False

    point_to_lobe: dict[tuple[int, int], int] = {}
    for idx, lobe in enumerate(lobes):
        for rr, cc in lobe:
            point_to_lobe[(int(rr), int(cc))] = idx

    graph = {i: set() for i in range(len(lobes))}
    pair_dirs: dict[tuple[int, int], set[int]] = {}
    for rr, cc in np.argwhere(mask):
        src = point_to_lobe[(int(rr), int(cc))]
        for dr, dc in DIAGONALS:
            nr, nc = int(rr + dr), int(cc + dc)
            if not _in_bounds(mask.shape, nr, nc) or not mask[nr, nc]:
                continue
            dst = point_to_lobe[(nr, nc)]
            if src == dst:
                continue
            a, b = sorted((src, dst))
            graph[a].add(b)
            graph[b].add(a)
            pair_dirs.setdefault((a, b), set()).add(1 if dr == dc else -1)

    if any(len(dirs) > 1 for dirs in pair_dirs.values()):
        return False

    if any(len(graph[i]) == 0 for i in range(len(lobes))):
        return False

    seen = set()
    stack = [0]
    while stack:
        cur = stack.pop()
        if cur in seen:
            continue
        seen.add(cur)
        stack.extend(graph[cur] - seen)
    if len(seen) != len(lobes):
        return False

    degrees = [len(graph[i]) for i in range(len(lobes))]
    if max(degrees) > 2:
        return False

    edge_count = sum(degrees) // 2
    if edge_count == len(lobes) - 1 and degrees.count(1) == 2:
        return True
    if len(lobes) >= 4 and edge_count == len(lobes) and all(deg == 2 for deg in degrees):
        return True
    return False


def _find_blue_boxes(grid: np.ndarray) -> list[tuple[int, int]]:
    h, w = grid.shape
    centers: list[tuple[int, int]] = []
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            if int(grid[r, c]) != BLACK:
                continue
            ok = True
            for dr, dc in RING_OFFSETS:
                if int(grid[r + dr, c + dc]) != BLUE:
                    ok = False
                    break
            if ok:
                centers.append((r, c))
    return centers


def _box_band(shape: tuple[int, int], center: tuple[int, int]) -> set[tuple[int, int]]:
    r, c = center
    out: set[tuple[int, int]] = set()
    for cc in range(c - 1, c + 2):
        if _in_bounds(shape, r - 2, cc):
            out.add((r - 2, cc))
        if _in_bounds(shape, r + 2, cc):
            out.add((r + 2, cc))
    for rr in range(r - 1, r + 2):
        if _in_bounds(shape, rr, c - 2):
            out.add((rr, c - 2))
        if _in_bounds(shape, rr, c + 2):
            out.add((rr, c + 2))
    return out


def _analyze(grid: np.ndarray) -> dict:
    grey_regions = _components_from_mask(grid == GREY, connectivity=8)
    chain_mask = np.zeros_like(grid, dtype=bool)
    normal_mask = np.zeros_like(grid, dtype=bool)
    chain_regions = 0
    for comp in grey_regions:
        if _is_diagonal_chain_region(comp):
            chain_regions += 1
            chain_mask[comp[:, 0], comp[:, 1]] = True
        else:
            normal_mask[comp[:, 0], comp[:, 1]] = True
    boxes = _find_blue_boxes(grid)
    box_info = []
    for center in boxes:
        band = _box_band(grid.shape, center)
        has_chain = any(chain_mask[r, c] for r, c in band)
        has_any_grey = any(int(grid[r, c]) == GREY for r, c in band)
        box_info.append(
            {
                "center": center,
                "band": band,
                "trigger": has_chain,
                "has_any_grey": has_any_grey,
            }
        )
    return {
        "grey_regions": grey_regions,
        "chain_regions": chain_regions,
        "boxes": box_info,
        "chain_mask": chain_mask,
        "normal_mask": normal_mask,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """Classify grey diagonal-chain regions and gate each blue hollow square to cross-or-erase.

    Input structure:
    - Black background with grey 8-connected regions and blue 3x3 hollow squares.
    - Each blue target is an 8-pixel square outline with a black center.
    - Grey regions may be diagonal-only lobe chains/rings or ordinary non-chain shapes.

    Rule:
    - Find every grey 8-connected region and decompose it into 4-connected lobes.
    - Mark a region as a diagonal chain only when its lobes touch only by diagonal corners and their lobe-contact graph is a non-branching path or simple cycle.
    - For each blue hollow square, inspect the one-cell-thick outer 4-neighbor band around its 3x3 window: if any marked diagonal-chain grey pixel is there, rewrite the square as a red plus; otherwise erase the whole blue square to black.

    Key insight:
    The blue squares are visually identical, so their fate is determined only by nearby grey topology: adjacency to a grey region is insufficient unless that region is specifically a diagonal-chain component.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    analysis = _analyze(g)
    chain_mask = analysis["chain_mask"]

    for center in _find_blue_boxes(g):
        r, c = center
        trigger = any(chain_mask[rr, cc] for rr, cc in _box_band(g.shape, center))
        out[r - 1 : r + 2, c - 1 : c + 2] = BLACK
        if trigger:
            for dr, dc in PLUS_OFFSETS:
                out[r + dr, c + dc] = RED
    return out


def _box_cells(center: tuple[int, int]) -> set[tuple[int, int]]:
    r, c = center
    return {(r + dr, c + dc) for dr, dc in RING_OFFSETS}


def _box_reserved(center: tuple[int, int]) -> set[tuple[int, int]]:
    r, c = center
    return {(rr, cc) for rr in range(r - 2, r + 3) for cc in range(c - 2, c + 3)}


def _transform_local(anchor: tuple[int, int], side: str, local_pts: list[tuple[int, int]]) -> set[tuple[int, int]]:
    ar, ac = anchor
    out = set()
    for x, y in local_pts:
        if side == "top":
            out.add((ar - y, ac + x))
        elif side == "bottom":
            out.add((ar + y, ac + x))
        elif side == "left":
            out.add((ar + x, ac - y))
        else:
            out.add((ar + x, ac + y))
    return out


def _chain_local(rng: random.Random) -> list[tuple[int, int]]:
    if rng.random() < 0.25:
        return [(0, 0), (-1, 1), (0, 2), (1, 1)]
    length = rng.randint(4, 7)
    x = 0
    step = rng.choice([-1, 1])
    pts = [(0, 0)]
    for y in range(1, length):
        if y > 1 and rng.random() < 0.45:
            step *= -1
        x += step
        pts.append((x, y))
    return pts


def _branch_local(rng: random.Random) -> list[tuple[int, int]]:
    pts = [(0, 0), (-1, 1), (1, 1), (0, 2), (2, 2)]
    if rng.random() < 0.5:
        pts = [(-x, y) for x, y in pts]
    return pts


def _polyomino_local(rng: random.Random) -> list[tuple[int, int]]:
    target = rng.randint(4, 8)
    cells = {(0, 0)}
    frontier = [(0, 0)]
    while len(cells) < target:
        base = rng.choice(frontier)
        candidates = []
        for dx, dy in [(-1, 0), (1, 0), (0, 1)]:
            nxt = (base[0] + dx, base[1] + dy)
            if nxt not in cells and nxt[1] >= 0:
                candidates.append(nxt)
        if not candidates:
            frontier = list(cells)
            continue
        cell = rng.choice(candidates)
        cells.add(cell)
        frontier.append(cell)
    min_x = min(x for x, _ in cells)
    min_y = min(y for _, y in cells)
    return sorted((x - min_x, y - min_y) for x, y in cells)


def _all_in_bounds(shape: tuple[int, int], coords: set[tuple[int, int]]) -> bool:
    return all(_in_bounds(shape, r, c) for r, c in coords)


def _touches_existing_grey(coords: set[tuple[int, int]], grey_occupied: set[tuple[int, int]]) -> bool:
    for r, c in coords:
        if (r, c) in grey_occupied:
            return True
        for dr, dc in N8:
            if (r + dr, c + dc) in grey_occupied:
                return True
    return False


def _place_region(
    shape: tuple[int, int],
    occupied: set[tuple[int, int]],
    grey_occupied: set[tuple[int, int]],
    reserved_forbidden: set[tuple[int, int]],
    candidates: list[set[tuple[int, int]]],
) -> set[tuple[int, int]] | None:
    for coords in candidates:
        if not _all_in_bounds(shape, coords):
            continue
        if coords & occupied:
            continue
        if coords & reserved_forbidden:
            continue
        if _touches_existing_grey(coords, grey_occupied):
            continue
        return coords
    return None


def _attached_candidates(center: tuple[int, int], kind: str, shape: tuple[int, int], rng: random.Random) -> list[set[tuple[int, int]]]:
    r, c = center
    by_side = {
        "top": [(r - 2, c - 1), (r - 2, c), (r - 2, c + 1)],
        "bottom": [(r + 2, c - 1), (r + 2, c), (r + 2, c + 1)],
        "left": [(r - 1, c - 2), (r, c - 2), (r + 1, c - 2)],
        "right": [(r - 1, c + 2), (r, c + 2), (r + 1, c + 2)],
    }
    sides = list(by_side)
    rng.shuffle(sides)
    out: list[set[tuple[int, int]]] = []
    for side in sides:
        local_pts = _chain_local(rng) if kind == "chain" else _branch_local(rng)
        for anchor in rng.sample(by_side[side], k=len(by_side[side])):
            out.append(_transform_local(anchor, side, local_pts))
    return out


def _free_candidates(
    kind: str,
    shape: tuple[int, int],
    rng: random.Random,
) -> list[set[tuple[int, int]]]:
    h, w = shape
    out: list[set[tuple[int, int]]] = []
    for _ in range(80):
        if kind == "chain":
            local = _chain_local(rng)
            side = rng.choice(["top", "bottom", "left", "right"])
        elif kind == "branch":
            local = _branch_local(rng)
            side = rng.choice(["top", "bottom", "left", "right"])
        else:
            local = _polyomino_local(rng)
            side = rng.choice(["top", "bottom", "left", "right"])
        anchor = (rng.randint(0, h - 1), rng.randint(0, w - 1))
        out.append(_transform_local(anchor, side, local))
    return out


def _semantic_ok(grid: np.ndarray, expect_boxes: int, want_isolated_negative: bool) -> bool:
    analysis = _analyze(grid)
    grey_regions = analysis["grey_regions"]
    boxes = analysis["boxes"]
    if len(boxes) != expect_boxes:
        return False
    if not (3 <= len(grey_regions) <= 5):
        return False
    if any(not (4 <= len(comp) <= 8) for comp in grey_regions):
        return False
    if analysis["chain_regions"] < 1 or analysis["chain_regions"] >= len(grey_regions):
        return False
    triggers = [info["trigger"] for info in boxes]
    if not any(triggers) or all(triggers):
        return False
    if not any(info["has_any_grey"] and not info["trigger"] for info in boxes):
        return False
    if want_isolated_negative and not any((not info["has_any_grey"]) and (not info["trigger"]) for info in boxes):
        return False

    out = main(grid)
    cross_count = 0
    blank_count = 0
    for info in boxes:
        r, c = info["center"]
        patch = out[r - 1 : r + 2, c - 1 : c + 2]
        cross = np.array(
            [
                [BLACK, RED, BLACK],
                [RED, RED, RED],
                [BLACK, RED, BLACK],
            ],
            dtype=int,
        )
        if info["trigger"]:
            if not np.array_equal(patch, cross):
                return False
            cross_count += 1
        else:
            if not np.array_equal(patch, np.zeros((3, 3), dtype=int)):
                return False
            blank_count += 1
    return cross_count >= 1 and blank_count >= 1


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(15, 19)
    w = rng.randint(15, 19)
    grid = np.zeros((h, w), dtype=int)

    preferred_boxes = [3, 2]
    if h >= 17 and w >= 17 and rng.random() < 0.35:
        preferred_boxes = [4, 3, 2]
    box_count = rng.choice(preferred_boxes)

    centers: list[tuple[int, int]] = []
    reserved_union: set[tuple[int, int]] = set()
    candidates = [(r, c) for r in range(2, h - 2) for c in range(2, w - 2)]
    rng.shuffle(candidates)
    for center in candidates:
        zone = _box_reserved(center)
        if zone & reserved_union:
            continue
        centers.append(center)
        reserved_union |= zone
        if len(centers) == box_count:
            break
    if len(centers) != box_count:
        return None

    occupied: set[tuple[int, int]] = set()
    grey_occupied: set[tuple[int, int]] = set()
    regions: list[set[tuple[int, int]]] = []

    for center in centers:
        for r, c in _box_cells(center):
            grid[r, c] = BLUE
            occupied.add((r, c))

    roles = ["trigger", "branch"]
    if box_count >= 3:
        roles.append("isolated")
    while len(roles) < box_count:
        roles.append(rng.choice(["trigger", "isolated"]))
    rng.shuffle(roles)

    box_reserved = {center: _box_reserved(center) for center in centers}

    for center, role in zip(centers, roles):
        if role == "isolated":
            continue
        forbidden = set()
        for other in centers:
            if other != center:
                forbidden |= box_reserved[other]
        kind = "chain" if role == "trigger" else "branch"
        region = _place_region(
            grid.shape,
            occupied,
            grey_occupied,
            forbidden,
            _attached_candidates(center, kind, grid.shape, rng),
        )
        if region is None:
            return None
        regions.append(region)
        grey_occupied |= region
        occupied |= region

    target_regions = rng.randint(3, 5)
    while len(regions) < target_regions:
        kind = rng.choices(["chain", "branch", "poly"], weights=[3, 2, 4])[0]
        region = _place_region(
            grid.shape,
            occupied,
            grey_occupied,
            reserved_union,
            _free_candidates(kind, grid.shape, rng),
        )
        if region is None:
            return None
        regions.append(region)
        grey_occupied |= region
        occupied |= region

    for region in regions:
        for r, c in region:
            grid[r, c] = GREY

    want_isolated_negative = box_count >= 3
    if not _semantic_ok(grid, box_count, want_isolated_negative):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(400):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid diagonal-chain gated square input")
