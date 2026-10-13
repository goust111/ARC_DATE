from __future__ import annotations

import random
from collections import defaultdict
from functools import lru_cache

import numpy as np


BG = 0
MAX_GRID_SIDE = 30
CONNECTIVITY = 8
PAIR_COUNT_RANGE = (2, 4)
BBOX_SIDE_RANGE = (4, 7)
LOCAL_MASK_ATTEMPTS = 300
PLACEMENT_ATTEMPTS = 400
MAX_GENERATION_ATTEMPTS = 300
PADDING = 1
MARGIN_LIMIT = 4
PALETTE = tuple(range(1, 10))
DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
DIR8 = tuple(
    (dr, dc)
    for dr in (-1, 0, 1)
    for dc in (-1, 0, 1)
    if not (dr == 0 and dc == 0)
)


def _is_connected(mask: np.ndarray, connectivity: int = CONNECTIVITY) -> bool:
    coords = np.argwhere(mask)
    if len(coords) == 0:
        return False
    dirs = DIR4 if connectivity == 4 else DIR8
    h, w = mask.shape
    start = (int(coords[0, 0]), int(coords[0, 1]))
    seen = {start}
    stack = [start]
    while stack:
        r, c = stack.pop()
        for dr, dc in dirs:
            nr = r + dr
            nc = c + dc
            if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and (nr, nc) not in seen:
                seen.add((nr, nc))
                stack.append((nr, nc))
    return len(seen) == len(coords)


def _touches_full_bbox(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if len(coords) == 0:
        return False
    rows = coords[:, 0]
    cols = coords[:, 1]
    return (
        int(rows.min()) == 0
        and int(rows.max()) == mask.shape[0] - 1
        and int(cols.min()) == 0
        and int(cols.max()) == mask.shape[1] - 1
    )


def _extract_components(grid: np.ndarray, bg: int = BG) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    visited = np.zeros((h, w), dtype=bool)
    components: list[dict] = []

    for r in range(h):
        for c in range(w):
            if visited[r, c] or int(g[r, c]) == bg:
                continue
            color = int(g[r, c])
            stack = [(r, c)]
            visited[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in DIR8:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc] and int(g[nr, nc]) == color:
                        visited[nr, nc] = True
                        stack.append((nr, nc))

            rows = [rr for rr, _cc in coords]
            cols = [cc for _rr, cc in coords]
            r0 = min(rows)
            r1 = max(rows)
            c0 = min(cols)
            c1 = max(cols)
            mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
            for rr, cc in coords:
                mask[rr - r0, cc - c0] = True
            components.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": (r0, c0, r1, c1),
                    "mask": mask,
                    "area": len(coords),
                }
            )
    return components


def _solve_group_matching(group: list[dict]) -> list[tuple[int, int]] | None:
    n = len(group)
    if n % 2 != 0:
        return None

    compat: list[list[int]] = [[] for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if np.array_equal(group[i]["mask"], np.logical_not(group[j]["mask"])):
                compat[i].append(j)
                compat[j].append(i)

    @lru_cache(maxsize=None)
    def rec(state: tuple[int, ...]) -> tuple[tuple[int, int], ...] | None:
        if not state:
            return ()
        i = state[0]
        rest = set(state[1:])
        for j in compat[i]:
            if j not in rest:
                continue
            next_state = tuple(x for x in state[1:] if x != j)
            suffix = rec(next_state)
            if suffix is not None:
                return ((i, j),) + suffix
        return None

    return list(rec(tuple(range(n))) or []) or None


def _matching_count(group: list[dict], limit: int = 2) -> int:
    n = len(group)
    compat: list[list[int]] = [[] for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if np.array_equal(group[i]["mask"], np.logical_not(group[j]["mask"])):
                compat[i].append(j)
                compat[j].append(i)

    @lru_cache(maxsize=None)
    def rec(state: tuple[int, ...]) -> int:
        if not state:
            return 1
        i = state[0]
        total = 0
        rest = set(state[1:])
        for j in compat[i]:
            if j not in rest:
                continue
            next_state = tuple(x for x in state[1:] if x != j)
            total += rec(next_state)
            if total >= limit:
                return limit
        return total

    return rec(tuple(range(n)))


def _pair_components(components: list[dict]) -> list[tuple[dict, dict]]:
    groups: dict[tuple[int, int], list[dict]] = defaultdict(list)
    for comp in components:
        groups[tuple(map(int, comp["mask"].shape))].append(comp)

    pairs: list[tuple[dict, dict]] = []
    for group in groups.values():
        solution = _solve_group_matching(group)
        assert solution is not None, "equal-sized fragments must form complementary pairs"
        for i, j in solution:
            pairs.append((group[i], group[j]))
    return pairs


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pair complementary fragments into full rectangles and repaint each kept rectangle with the smaller fragment's color.

    Input structure:
    - Black is the background.
    - Scattered non-black components are connected colored fragments that come in disjoint pairs.
    - Within each pair, the two tight bounding boxes have the same height and width, and overlaying the two local masks fills that shared box exactly.

    Rule:
    - Extract every foreground fragment and compare only fragments with equal bbox size.
    - Pair fragments whose local masks are exact complements inside the shared bbox.
    - For each pair, identify the smaller-area fragment and the larger-area fragment.
    - Clear the smaller fragment's location, then fill the larger fragment's bbox as a solid rectangle in the smaller fragment's color.

    Key insight:
    - The hidden relation is mask complementarity inside equal-sized bounding boxes; after finding that pairing, the surviving rectangle is not colored by its own fragment but by the smaller partner's color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert max(g.shape) <= MAX_GRID_SIDE, "ARC grids must stay within 30x30"

    components = _extract_components(g, bg=BG)
    assert components and len(components) % 2 == 0, "expected an even number of foreground fragments"

    out = np.full(g.shape, BG, dtype=int)
    for left, right in _pair_components(components):
        assert int(left["area"]) != int(right["area"]), "pair areas must be different"
        small, big = (left, right) if int(left["area"]) < int(right["area"]) else (right, left)
        r0, c0, r1, c1 = map(int, big["bbox"])
        out[r0 : r1 + 1, c0 : c1 + 1] = int(small["color"])
    return out


def _random_local_mask(h: int, w: int) -> np.ndarray | None:
    total = h * w
    for _ in range(LOCAL_MASK_ATTEMPTS):
        density = random.uniform(0.28, 0.72)
        mask = np.random.random((h, w)) < density
        area = int(mask.sum())
        if area == 0 or area == total or area * 2 == total:
            continue
        if min(area, total - area) < max(3, min(h, w) - 1):
            continue
        comp = np.logical_not(mask)
        if not _touches_full_bbox(mask) or not _touches_full_bbox(comp):
            continue
        if not _is_connected(mask) or not _is_connected(comp):
            continue
        return mask
    return None


def _sample_pair_dims(pair_count: int) -> list[tuple[int, int]]:
    dims: list[tuple[int, int]] = []
    pool: list[tuple[int, int]] = []
    for _ in range(pair_count):
        if pool and random.random() < 0.45:
            dims.append(random.choice(pool))
            continue
        dim = (
            random.randint(BBOX_SIDE_RANGE[0], BBOX_SIDE_RANGE[1]),
            random.randint(BBOX_SIDE_RANGE[0], BBOX_SIDE_RANGE[1]),
        )
        pool.append(dim)
        dims.append(dim)
    return dims


def _place_objects(objects: list[dict]) -> tuple[np.ndarray, list[tuple[int, int]]] | None:
    canvas = np.full((MAX_GRID_SIDE, MAX_GRID_SIDE), BG, dtype=int)
    padded = np.zeros((MAX_GRID_SIDE, MAX_GRID_SIDE), dtype=bool)
    positions: list[tuple[int, int] | None] = [None] * len(objects)
    order = list(range(len(objects)))
    random.shuffle(order)

    for obj_idx in order:
        obj = objects[obj_idx]
        h, w = map(int, obj["mask"].shape)
        for _ in range(PLACEMENT_ATTEMPTS):
            r0 = random.randint(0, MAX_GRID_SIDE - h)
            c0 = random.randint(0, MAX_GRID_SIDE - w)
            rr0 = max(0, r0 - PADDING)
            cc0 = max(0, c0 - PADDING)
            rr1 = min(MAX_GRID_SIDE, r0 + h + PADDING)
            cc1 = min(MAX_GRID_SIDE, c0 + w + PADDING)
            if np.any(padded[rr0:rr1, cc0:cc1]):
                continue
            positions[obj_idx] = (r0, c0)
            padded[rr0:rr1, cc0:cc1] = True
            block = canvas[r0 : r0 + h, c0 : c0 + w]
            block[obj["mask"]] = int(obj["color"])
            break
        if positions[obj_idx] is None:
            return None
    return canvas, [(int(r), int(c)) for r, c in positions]


def _gen_attempt() -> np.ndarray | None:
    pair_count = random.randint(*PAIR_COUNT_RANGE)
    dims_list = _sample_pair_dims(pair_count)
    colors = random.sample(PALETTE, 2 * pair_count)

    objects: list[dict] = []
    pair_to_obj_indices: list[tuple[int, int]] = []
    used_signatures: dict[tuple[int, int], set[bytes]] = defaultdict(set)

    for pair_idx, dims in enumerate(dims_list):
        h, w = dims
        mask = None
        for _ in range(LOCAL_MASK_ATTEMPTS):
            candidate = _random_local_mask(h, w)
            if candidate is None:
                continue
            sig = candidate.tobytes()
            comp_sig = np.logical_not(candidate).tobytes()
            if sig in used_signatures[dims] or comp_sig in used_signatures[dims]:
                continue
            used_signatures[dims].add(sig)
            used_signatures[dims].add(comp_sig)
            mask = candidate
            break
        if mask is None:
            return None

        comp = np.logical_not(mask)
        local_masks = [mask, comp]
        random.shuffle(local_masks)
        start = len(objects)
        for local_mask in local_masks:
            objects.append(
                {
                    "pair_idx": pair_idx,
                    "mask": local_mask,
                    "area": int(local_mask.sum()),
                    "color": colors[len(objects)],
                }
            )
        pair_to_obj_indices.append((start, start + 1))

    placed = _place_objects(objects)
    if placed is None:
        return None
    canvas, positions = placed

    expected = np.full_like(canvas, BG)
    for left_idx, right_idx in pair_to_obj_indices:
        left = objects[left_idx]
        right = objects[right_idx]
        if int(left["area"]) == int(right["area"]):
            return None
        small_idx, big_idx = (
            (left_idx, right_idx) if int(left["area"]) < int(right["area"]) else (right_idx, left_idx)
        )
        br0, bc0 = positions[big_idx]
        bh, bw = map(int, objects[big_idx]["mask"].shape)
        expected[br0 : br0 + bh, bc0 : bc0 + bw] = int(objects[small_idx]["color"])

    row_mins = [positions[i][0] for i in range(len(objects))]
    col_mins = [positions[i][1] for i in range(len(objects))]
    row_maxs = [positions[i][0] + objects[i]["mask"].shape[0] - 1 for i in range(len(objects))]
    col_maxs = [positions[i][1] + objects[i]["mask"].shape[1] - 1 for i in range(len(objects))]
    top = max(0, min(row_mins) - random.randint(0, min(MARGIN_LIMIT, min(row_mins))))
    left = max(0, min(col_mins) - random.randint(0, min(MARGIN_LIMIT, min(col_mins))))
    bottom_margin = random.randint(0, min(MARGIN_LIMIT, MAX_GRID_SIDE - 1 - max(row_maxs)))
    right_margin = random.randint(0, min(MARGIN_LIMIT, MAX_GRID_SIDE - 1 - max(col_maxs)))
    bottom = min(MAX_GRID_SIDE - 1, max(row_maxs) + bottom_margin)
    right = min(MAX_GRID_SIDE - 1, max(col_maxs) + right_margin)

    grid = canvas[top : bottom + 1, left : right + 1]
    exp = expected[top : bottom + 1, left : right + 1]

    comps = _extract_components(grid, bg=BG)
    groups: dict[tuple[int, int], list[dict]] = defaultdict(list)
    for comp in comps:
        groups[tuple(map(int, comp["mask"].shape))].append(comp)
    for group in groups.values():
        if _matching_count(group, limit=2) != 1:
            return None

    if not np.array_equal(main(grid), exp):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate input after {MAX_GENERATION_ATTEMPTS} attempts")
