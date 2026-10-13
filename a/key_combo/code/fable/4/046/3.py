from __future__ import annotations

import random
from collections import Counter, deque

import numpy as np


BLACK = 0
PALETTE = [1, 2, 3, 4, 5, 6, 7, 8, 9]
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.flatten())).most_common(1)[0][0])


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _count_bends(mask: np.ndarray) -> int:
    h, w = mask.shape
    bends = 0
    for r in range(h - 1):
        for c in range(w - 1):
            if int(mask[r : r + 2, c : c + 2].sum()) == 3:
                bends += 1
    return bends


def _mask_signature(mask: np.ndarray) -> bytes:
    return mask.astype(np.uint8).tobytes() + str(mask.shape).encode()


def _normalize_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy()


def _dihedral_masks(mask: np.ndarray) -> list[np.ndarray]:
    variants = []
    seen = set()
    for flip in (False, True):
        base = np.fliplr(mask) if flip else mask
        for k in range(4):
            variant = _normalize_mask(np.rot90(base, k))
            sig = _mask_signature(variant)
            if sig not in seen:
                seen.add(sig)
                variants.append(variant)
    return variants


def _extract_objects(grid: np.ndarray, bg: int) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    objects = []

    for sr, sc in np.argwhere(g != bg):
        sr = int(sr)
        sc = int(sc)
        if seen[sr, sc]:
            continue
        q = deque([(sr, sc)])
        seen[sr, sc] = True
        coords = []
        colors = set()
        while q:
            r, c = q.popleft()
            coords.append((r, c))
            colors.add(int(g[r, c]))
            for dr, dc in DIRS4:
                nr = r + dr
                nc = c + dc
                if _in_bounds(g.shape, nr, nc) and not seen[nr, nc] and int(g[nr, nc]) != bg:
                    seen[nr, nc] = True
                    q.append((nr, nc))

        assert len(colors) == 1, "each falling object must be monochromatic"
        color = colors.pop()
        coords_arr = np.array(coords, dtype=int)
        r0, c0 = coords_arr.min(axis=0)
        r1, c1 = coords_arr.max(axis=0)
        mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        rel = coords_arr - np.array([r0, c0])
        mask[rel[:, 0], rel[:, 1]] = True
        objects.append(
            {
                "color": int(color),
                "r0": int(r0),
                "c0": int(c0),
                "mask": mask,
                "bend": _count_bends(mask),
            }
        )

    return objects


def _drop_one(board: np.ndarray, obj: dict, bg: int) -> tuple[int, bool]:
    mask = obj["mask"]
    rel = np.argwhere(mask)
    shift = 0
    blocked_by_object = False

    while True:
        for dr, dc in rel:
            nr = int(obj["r0"] + dr + shift + 1)
            nc = int(obj["c0"] + dc)
            if nr >= board.shape[0]:
                return shift, blocked_by_object
            if int(board[nr, nc]) != bg:
                blocked_by_object = True
                return shift, blocked_by_object
        shift += 1


def _paint(board: np.ndarray, obj: dict, shift: int) -> None:
    rel = np.argwhere(obj["mask"])
    for dr, dc in rel:
        board[int(obj["r0"] + dr + shift), int(obj["c0"] + dc)] = int(obj["color"])


def _drop_objects(shape: tuple[int, int], objects: list[dict], bg: int, order: list[int]) -> tuple[np.ndarray, int]:
    board = np.full(shape, bg, dtype=int)
    supports = 0
    for idx in order:
        shift, blocked_by_object = _drop_one(board, objects[idx], bg)
        supports += int(blocked_by_object)
        _paint(board, objects[idx], shift)
    return board, supports


def _correct_order(objects: list[dict]) -> list[int]:
    return sorted(
        range(len(objects)),
        key=lambda i: (objects[i]["bend"], objects[i]["r0"], objects[i]["c0"], objects[i]["color"]),
    )


def main(grid: np.ndarray) -> np.ndarray:
    """
    Drop each floating monochrome shape in increasing bend-count order until all shapes are stacked on the ground.

    Input structure:
    - The grid has a black background with several 4-connected monochrome foreground shapes floating above the bottom edge.
    - Each shape has a distinct number of local 2x2 L-corner bends.
    - Shapes overlap in column span so earlier landings can change the terrain for later ones.

    Rule:
    - Find every connected foreground shape and crop its boolean mask.
    - Count its bends by scanning the crop for 2x2 windows containing exactly three foreground cells.
    - Sort the shapes by bend count from fewest to most, then let them fall vertically one at a time onto an empty black board until blocked by the floor or already settled shapes.

    Key insight:
    - The final stack comes from serial gravity, not simultaneous gravity, and the execution order is encoded by each shape's topology.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = get_bg(g)
    objects = _extract_objects(g, bg)
    order = _correct_order(objects)
    out, _supports = _drop_objects(g.shape, objects, bg, order)
    return out


def _random_path_mask(target_bends: int, rng: random.Random) -> np.ndarray | None:
    for _ in range(300):
        directions = [rng.choice(DIRS4)]
        for _turn in range(target_bends):
            prev = directions[-1]
            options = [d for d in DIRS4 if d[0] != prev[0] and d[1] != prev[1]]
            directions.append(rng.choice(options))

        lengths = [rng.randint(1, 3) for _ in directions]
        lengths[0] = rng.randint(2, 4)
        lengths[-1] = rng.randint(2, 4)
        if target_bends >= 3:
            lengths[rng.randrange(len(lengths))] += rng.randint(0, 1)

        cells = {(0, 0)}
        r = c = 0
        ok = True
        for (dr, dc), seg_len in zip(directions, lengths):
            for _step in range(seg_len):
                r += dr
                c += dc
                if (r, c) in cells:
                    ok = False
                    break
                cells.add((r, c))
            if not ok:
                break
        if not ok:
            continue

        pts = np.array(sorted(cells), dtype=int)
        pts -= pts.min(axis=0)
        h, w = pts.max(axis=0) + 1
        if h > 8 or w > 8:
            continue
        mask = np.zeros((int(h), int(w)), dtype=bool)
        mask[pts[:, 0], pts[:, 1]] = True
        mask = _normalize_mask(mask)
        if _count_bends(mask) != target_bends:
            continue
        if mask.sum() < target_bends + 3 or mask.sum() > 14:
            continue
        return mask
    return None


def _build_catalog() -> dict[int, list[np.ndarray]]:
    rng = random.Random(0)
    catalog = {1: [], 2: [], 3: [], 4: []}
    seen = {k: set() for k in catalog}
    target_per_bend = 16

    for bends in catalog:
        attempts = 0
        while len(catalog[bends]) < target_per_bend and attempts < 6000:
            attempts += 1
            mask = _random_path_mask(bends, rng)
            if mask is None:
                continue
            for variant in _dihedral_masks(mask):
                if _count_bends(variant) != bends:
                    continue
                sig = _mask_signature(variant)
                if sig in seen[bends]:
                    continue
                seen[bends].add(sig)
                catalog[bends].append(variant.astype(bool))
                if len(catalog[bends]) >= target_per_bend:
                    break
    return catalog


SHAPE_CATALOG = _build_catalog()


def _sample_masks(rng: random.Random) -> list[tuple[int, np.ndarray]]:
    bend_counts = [1, 2, 3]
    if SHAPE_CATALOG[4] and rng.random() < 0.35:
        bend_counts.append(4)

    sampled = []
    for bends in bend_counts:
        mask = rng.choice(SHAPE_CATALOG[bends]).copy()
        sampled.append((bends, mask))
    return sampled


def _mask_cells(mask: np.ndarray, r0: int, c0: int) -> list[tuple[int, int]]:
    rel = np.argwhere(mask)
    return [(int(r0 + dr), int(c0 + dc)) for dr, dc in rel]


def _can_place(mask: np.ndarray, r0: int, c0: int, blocked: set[tuple[int, int]], shape: tuple[int, int]) -> bool:
    for rr, cc in _mask_cells(mask, r0, c0):
        if not _in_bounds(shape, rr, cc) or (rr, cc) in blocked:
            return False
    return True


def _add_blocked(mask: np.ndarray, r0: int, c0: int, blocked: set[tuple[int, int]]) -> None:
    for rr, cc in _mask_cells(mask, r0, c0):
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                blocked.add((rr + dr, cc + dc))


def _column_intervals(objects: list[dict]) -> list[tuple[int, int]]:
    intervals = []
    for obj in objects:
        h, w = obj["mask"].shape
        intervals.append((obj["c0"], obj["c0"] + w - 1))
    return intervals


def _overlap_graph_connected(objects: list[dict]) -> bool:
    intervals = _column_intervals(objects)
    n = len(intervals)
    seen = {0}
    q = deque([0])
    while q:
        i = q.popleft()
        a0, a1 = intervals[i]
        for j in range(n):
            if j in seen:
                continue
            b0, b1 = intervals[j]
            if max(a0, b0) <= min(a1, b1):
                seen.add(j)
                q.append(j)
    return len(seen) == n


def _height_order(objects: list[dict]) -> list[int]:
    return sorted(
        range(len(objects)),
        key=lambda i: (
            -(objects[i]["r0"] + objects[i]["mask"].shape[0] - 1),
            objects[i]["c0"],
            objects[i]["bend"],
        ),
    )


def _reverse_bend_order(objects: list[dict]) -> list[int]:
    return sorted(
        range(len(objects)),
        key=lambda i: (-objects[i]["bend"], objects[i]["r0"], objects[i]["c0"], objects[i]["color"]),
    )


def _make_input_grid(shape: tuple[int, int], objects: list[dict], bg: int) -> np.ndarray:
    grid = np.full(shape, bg, dtype=int)
    for obj in objects:
        for rr, cc in _mask_cells(obj["mask"], obj["r0"], obj["c0"]):
            grid[rr, cc] = obj["color"]
    return grid


def _place_objects(sampled: list[tuple[int, np.ndarray]], rng: random.Random, shape: tuple[int, int]) -> list[dict] | None:
    h, w = shape
    overlap_anchor = rng.randint(2, w - 3)
    blocked: set[tuple[int, int]] = set()
    objects: list[dict | None] = [None] * len(sampled)
    placement_order = list(range(len(sampled)))
    rng.shuffle(placement_order)

    for idx in placement_order:
        bends, mask = sampled[idx]
        mh, mw = mask.shape
        max_r = max(1, h // 2 - mh - 1)
        if max_r < 0:
            return None
        col_lo = max(0, overlap_anchor - mw + 1)
        col_hi = min(w - mw, overlap_anchor)
        if col_lo > col_hi:
            return None

        candidates = [(r0, c0) for r0 in range(0, max_r + 1) for c0 in range(col_lo, col_hi + 1)]
        rng.shuffle(candidates)
        chosen = None
        for r0, c0 in candidates:
            if _can_place(mask, r0, c0, blocked, shape):
                chosen = (r0, c0)
                break
        if chosen is None:
            return None

        r0, c0 = chosen
        objects[idx] = {
            "color": None,
            "r0": int(r0),
            "c0": int(c0),
            "mask": mask.copy(),
            "bend": int(bends),
        }
        _add_blocked(mask, r0, c0, blocked)

    placed = [obj for obj in objects if obj is not None]
    assert len(placed) == len(sampled)
    if not _overlap_graph_connected(placed):
        return None

    rows_by_bend = [obj["r0"] for obj in sorted(placed, key=lambda item: item["bend"])]
    if rows_by_bend == sorted(rows_by_bend) or rows_by_bend == sorted(rows_by_bend, reverse=True):
        return None
    return placed


def _nondegenerate(objects: list[dict], shape: tuple[int, int], bg: int) -> bool:
    correct, supports = _drop_objects(shape, objects, bg, _correct_order(objects))
    reverse, _ = _drop_objects(shape, objects, bg, _reverse_bend_order(objects))
    by_height, _ = _drop_objects(shape, objects, bg, _height_order(objects))
    return supports >= 1 and not np.array_equal(correct, reverse) and not np.array_equal(correct, by_height)


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    shape = (rng.randint(16, 26), rng.randint(11, 18))
    sampled = _sample_masks(rng)
    objects = _place_objects(sampled, rng, shape)
    if objects is None:
        return None

    colors = rng.sample(PALETTE, len(objects))
    for obj, color in zip(objects, colors):
        obj["color"] = int(color)

    if not _nondegenerate(objects, shape, BLACK):
        return None

    grid = _make_input_grid(shape, objects, BLACK)
    try:
        solved = main(grid)
    except Exception:
        return None

    correct, _ = _drop_objects(shape, objects, BLACK, _correct_order(objects))
    if not np.array_equal(solved, correct):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(400):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid chained-gravity ARC input")
