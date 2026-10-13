from __future__ import annotations

from functools import lru_cache
import random

import numpy as np


BLACK = 0
NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))

# --------------- solver ---------------


def _tight_bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    bbox = _tight_bbox(mask)
    assert bbox is not None
    r0, c0, r1, c1 = bbox
    return np.asarray(mask[r0 : r1 + 1, c0 : c1 + 1], dtype=bool)


def _mask_from_cells(shape: tuple[int, int], cells: set[tuple[int, int]]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    for r, c in cells:
        mask[int(r), int(c)] = True
    return mask


def _is_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    h, w = mask.shape
    start = tuple(int(v) for v in coords[0])
    seen = {start}
    stack = [start]
    while stack:
        r, c = stack.pop()
        for dr, dc in NEIGHBORS4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                stack.append((nr, nc))
    return len(seen) == int(mask.sum())


def _unique_rotations(mask: np.ndarray) -> tuple[np.ndarray, ...]:
    variants: list[np.ndarray] = []
    seen: set[tuple[tuple[int, int], bytes]] = set()
    cur = np.asarray(mask, dtype=bool)
    for _ in range(4):
        cur = np.rot90(cur)
        key = (tuple(int(v) for v in cur.shape), cur.tobytes())
        if key not in seen:
            seen.add(key)
            variants.append(cur.copy())
    return tuple(variants)


@lru_cache(maxsize=None)
def _ordered_factor_pairs(area: int) -> tuple[tuple[int, int], ...]:
    pairs = []
    for h in range(1, area + 1):
        if area % h == 0:
            pairs.append((int(h), int(area // h)))
    return tuple(pairs)


def _extract_components(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []

    for r in range(h):
        for c in range(w):
            color = int(g[r, c])
            if color == BLACK or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells = []
            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                for dr, dc in NEIGHBORS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(g[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))

            rows = [rr for rr, _ in cells]
            cols = [cc for _, cc in cells]
            r0, r1 = min(rows), max(rows)
            c0, c1 = min(cols), max(cols)
            mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
            for rr, cc in cells:
                mask[rr - r0, cc - c0] = True
            comps.append(
                {
                    "color": color,
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    "mask": _crop_mask(mask),
                    "area": int(mask.sum()),
                }
            )

    comps.sort(key=lambda comp: (comp["bbox"][0], comp["bbox"][1], comp["color"]))
    return comps


def _pair_candidates(mask_a: np.ndarray, color_a: int, mask_b: np.ndarray, color_b: int) -> list[dict]:
    area_a = int(mask_a.sum())
    area_b = int(mask_b.sum())
    total_area = area_a + area_b
    candidates: dict[tuple[int, int, bytes], dict] = {}

    for rot_a in _unique_rotations(mask_a):
        ah, aw = rot_a.shape
        for rot_b in _unique_rotations(mask_b):
            bh, bw = rot_b.shape
            for out_h, out_w in _ordered_factor_pairs(total_area):
                if out_h < ah or out_w < aw or out_h < bh or out_w < bw:
                    continue
                for r0 in range(out_h - ah + 1):
                    for c0 in range(out_w - aw + 1):
                        fill_a = np.zeros((out_h, out_w), dtype=bool)
                        fill_a[r0 : r0 + ah, c0 : c0 + aw] = rot_a
                        fill_b = ~fill_a
                        bbox_b = _tight_bbox(fill_b)
                        if bbox_b is None:
                            continue
                        br0, bc0, br1, bc1 = bbox_b
                        if (br1 - br0 + 1, bc1 - bc0 + 1) != rot_b.shape:
                            continue
                        if not np.array_equal(fill_b[br0 : br1 + 1, bc0 : bc1 + 1], rot_b):
                            continue

                        label_patch = np.full((out_h, out_w), 2, dtype=np.uint8)
                        label_patch[fill_a] = 1
                        key = (int(out_h), int(out_w), label_patch.tobytes())
                        if key in candidates:
                            continue

                        patch = np.full((out_h, out_w), int(color_b), dtype=int)
                        patch[fill_a] = int(color_a)
                        candidates[key] = {
                            "patch": patch,
                            "candidate_key": key,
                            "area": total_area,
                        }

    return [candidates[key] for key in sorted(candidates)]


def _best_pair_candidate(comp_a: dict, comp_b: dict) -> dict | None:
    candidates = _pair_candidates(comp_a["mask"], comp_a["color"], comp_b["mask"], comp_b["color"])
    if not candidates:
        return None
    return candidates[0]


def _pack_output(patches: list[np.ndarray]) -> np.ndarray:
    if not patches:
        return np.zeros((1, 1), dtype=int)
    height = max(int(patch.shape[0]) for patch in patches)
    width = sum(int(patch.shape[1]) for patch in patches) + len(patches) - 1
    out = np.zeros((height, width), dtype=int)
    col = 0
    for patch in patches:
        ph, pw = patch.shape
        out[:ph, col : col + pw] = patch
        col += pw + 1
    return out


def _matching_signature(keys: tuple[tuple[int, int], ...], pair_info: dict[tuple[int, int], dict]) -> tuple:
    ordered = tuple(sorted(keys, key=lambda key: pair_info[key]["sort_key"]))
    return tuple(pair_info[key]["match_key"] for key in ordered)


def _best_matching(pair_info: dict[tuple[int, int], dict], n_comp: int) -> tuple[tuple[int, int], ...] | None:
    full_mask = (1 << n_comp) - 1

    @lru_cache(maxsize=None)
    def solve(mask: int) -> tuple[tuple[int, int], ...] | None:
        if mask == full_mask:
            return ()
        first = 0
        while (mask >> first) & 1:
            first += 1

        best_keys = None
        best_sig = None
        for second in range(first + 1, n_comp):
            if (mask >> second) & 1:
                continue
            key = (first, second)
            if key not in pair_info:
                continue
            rest = solve(mask | (1 << first) | (1 << second))
            if rest is None:
                continue
            candidate = (key,) + rest
            sig = _matching_signature(candidate, pair_info)
            if best_sig is None or sig < best_sig:
                best_sig = sig
                best_keys = candidate
        return best_keys

    return solve(0)


def _count_matchings(pair_info: dict[tuple[int, int], dict], n_comp: int, limit: int = 2) -> int:
    full_mask = (1 << n_comp) - 1

    @lru_cache(maxsize=None)
    def count(mask: int) -> int:
        if mask == full_mask:
            return 1
        first = 0
        while (mask >> first) & 1:
            first += 1

        total = 0
        for second in range(first + 1, n_comp):
            if (mask >> second) & 1:
                continue
            key = (first, second)
            if key not in pair_info:
                continue
            total += count(mask | (1 << first) | (1 << second))
            if total >= limit:
                return limit
        return total

    return int(count(0))


def _pair_info_for_components(comps: list[dict]) -> dict[tuple[int, int], dict]:
    pair_info: dict[tuple[int, int], dict] = {}
    for i in range(len(comps)):
        for j in range(i + 1, len(comps)):
            cand = _best_pair_candidate(comps[i], comps[j])
            if cand is None:
                continue
            min_r = min(int(comps[i]["bbox"][0]), int(comps[j]["bbox"][0]))
            min_c = min(int(comps[i]["bbox"][1]), int(comps[j]["bbox"][1]))
            pair_info[(i, j)] = {
                "patch": cand["patch"],
                "sort_key": (int(cand["area"]), int(min_c), int(min_r), cand["candidate_key"]),
                "match_key": (int(cand["area"]), cand["candidate_key"]),
            }
    return pair_info


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pair scattered monochrome pieces by complementary shape and assemble sorted rectangles.

    Input structure:
    - Black background with several colored 4-connected components.
    - Components come in pairs of different colors.
    - After rotating components by multiples of 90 degrees, each pair can be translated to
      tile a solid axis-aligned rectangle without overlap.

    Rule:
    - Find the perfect pairing of components.
    - For each pair, keep both original colors inside the assembled rectangle.
    - Sort assembled rectangles by area and place them left-to-right with one black column gap,
      top-aligned.
    """
    g = np.asarray(grid, dtype=int)
    comps = _extract_components(g)
    assert len(comps) >= 4 and len(comps) % 2 == 0

    pair_info = _pair_info_for_components(comps)
    matching = _best_matching(pair_info, len(comps))
    assert matching is not None

    ordered = sorted(matching, key=lambda key: pair_info[key]["sort_key"])
    patches = [pair_info[key]["patch"] for key in ordered]
    return _pack_output(patches)


# --------------- generator ---------------

# Pair count and output rectangle sizes.
PAIR_COUNT_RANGE = (2, 4)
RECT_DIMS = tuple(
    (h, w)
    for h in range(2, 6)
    for w in range(2, 7)
    if 6 <= h * w <= 20
)

# Canvas and placement.
GRID_SIDE_RANGE = (18, 30)
PLACEMENT_PADDING = 1

# Partition and retry budgets.
PARTITION_ATTEMPTS = 200
PLACEMENT_ATTEMPTS = 80
GENERATION_ATTEMPTS = 400


def _is_rectangle(mask: np.ndarray) -> bool:
    return bool(mask.all())


def _grow_connected_subset(h: int, w: int, target: int) -> set[tuple[int, int]]:
    start = (random.randrange(h), random.randrange(w))
    cells = {start}
    frontier = set()
    frontier.update(
        (start[0] + dr, start[1] + dc)
        for dr, dc in NEIGHBORS4
        if 0 <= start[0] + dr < h and 0 <= start[1] + dc < w
    )

    while len(cells) < target and frontier:
        pick = random.choice(tuple(frontier))
        frontier.remove(pick)
        cells.add(pick)
        pr, pc = pick
        for dr, dc in NEIGHBORS4:
            nr, nc = pr + dr, pc + dc
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                frontier.add((nr, nc))
    return cells


def _random_partition(h: int, w: int) -> tuple[np.ndarray, np.ndarray] | None:
    total = h * w
    low = max(2, total // 3)
    high = min(total - 2, total - total // 3)
    if low > high:
        low, high = 2, total - 2
    targets = list(range(low, high + 1))
    random.shuffle(targets)

    for target in targets:
        for _ in range(PARTITION_ATTEMPTS):
            cells_a = _grow_connected_subset(h, w, target)
            if len(cells_a) != target:
                continue
            mask_a = _mask_from_cells((h, w), cells_a)
            mask_b = ~mask_a
            if not _is_connected(mask_a) or not _is_connected(mask_b):
                continue
            crop_a = _crop_mask(mask_a)
            crop_b = _crop_mask(mask_b)
            if _is_rectangle(crop_a) and _is_rectangle(crop_b):
                continue
            if min(int(crop_a.sum()), int(crop_b.sum())) < 2:
                continue
            return crop_a, crop_b
    return None


def _colorize(mask: np.ndarray, color: int) -> np.ndarray:
    patch = np.zeros(mask.shape, dtype=int)
    patch[mask] = int(color)
    return patch


def _place_patches(patches: list[np.ndarray], height: int, width: int) -> np.ndarray | None:
    grid = np.zeros((height, width), dtype=int)
    reserved = np.zeros((height, width), dtype=bool)
    order = list(range(len(patches)))
    random.shuffle(order)
    order.sort(key=lambda idx: patches[idx].shape[0] * patches[idx].shape[1], reverse=True)

    for idx in order:
        patch = patches[idx]
        ph, pw = patch.shape
        slots = []
        for r0 in range(height - ph + 1):
            for c0 in range(width - pw + 1):
                rr0 = max(0, r0 - PLACEMENT_PADDING)
                cc0 = max(0, c0 - PLACEMENT_PADDING)
                rr1 = min(height, r0 + ph + PLACEMENT_PADDING)
                cc1 = min(width, c0 + pw + PLACEMENT_PADDING)
                if reserved[rr0:rr1, cc0:cc1].any():
                    continue
                slots.append((r0, c0))
        if not slots:
            return None
        r0, c0 = random.choice(slots)
        grid[r0 : r0 + ph, c0 : c0 + pw][patch != 0] = patch[patch != 0]
        reserved[r0 : r0 + ph, c0 : c0 + pw] = True
    return grid


def _gen_attempt() -> np.ndarray | None:
    pair_count = random.randint(*PAIR_COUNT_RANGE)
    color_pool = list(range(1, 10))
    random.shuffle(color_pool)
    if len(color_pool) < 2 * pair_count:
        return None

    dims_by_area: dict[int, list[tuple[int, int]]] = {}
    for dims in RECT_DIMS:
        dims_by_area.setdefault(dims[0] * dims[1], []).append(dims)
    if len(dims_by_area) < pair_count:
        return None

    areas = random.sample(sorted(dims_by_area), pair_count)
    input_patches: list[np.ndarray] = []

    for idx, area in enumerate(areas):
        h, w = random.choice(dims_by_area[area])
        partition = _random_partition(h, w)
        if partition is None:
            return None
        mask_a, mask_b = partition
        color_a = color_pool[2 * idx]
        color_b = color_pool[2 * idx + 1]

        rot_a = np.rot90(mask_a, random.randrange(4))
        rot_b = np.rot90(mask_b, random.randrange(4))
        input_patches.append(_colorize(rot_a, color_a))
        input_patches.append(_colorize(rot_b, color_b))

    max_h = max(patch.shape[0] for patch in input_patches)
    max_w = max(patch.shape[1] for patch in input_patches)
    min_side = max(18, max_h + 2, max_w + 2)

    for _ in range(PLACEMENT_ATTEMPTS):
        height = random.randint(min_side, GRID_SIDE_RANGE[1])
        width = random.randint(min_side, GRID_SIDE_RANGE[1])
        placed = _place_patches(input_patches, height, width)
        if placed is None:
            continue

        comps = _extract_components(placed)
        if len(comps) != 2 * pair_count:
            continue
        pair_info = _pair_info_for_components(comps)
        if _count_matchings(pair_info, len(comps), limit=2) != 1:
            continue

        try:
            main(placed)
        except AssertionError:
            continue
        else:
            return placed
    return None


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid complementary-rectangle puzzle")
