import random
from collections import Counter, deque

import numpy as np


BLACK = 0
RED = 2
YELLOW = 4
TEAL = 8

GRID_MIN = 14
GRID_MAX = 19
RECT_COUNT_RANGE = (3, 5)
EDGE_MARGIN = 1
MIN_GAP = 1
GEN_ATTEMPTS = 400

AREA_TO_DIMS = {
    4: [(2, 2)],
    6: [(2, 3), (3, 2)],
    8: [(2, 4), (4, 2)],
    9: [(3, 3)],
    12: [(3, 4), (4, 3)],
    15: [(3, 5), (5, 3)],
    16: [(4, 4)],
    20: [(4, 5), (5, 4)],
    25: [(5, 5)],
}
SMALL_AREAS = [4, 6, 8]
BIG_AREAS = [9, 12, 15, 16, 20, 25]
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def components_mask(mask: np.ndarray) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    components = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            queue = deque([(r, c)])
            seen[r, c] = True
            coords = []
            while queue:
                cr, cc = queue.popleft()
                coords.append((cr, cc))
                for dr, dc in DIRS4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            components.append(np.array(coords, dtype=int))
    return components


def find_teal_fragments(grid: np.ndarray) -> list[dict]:
    out = []
    for coords in components_mask(grid == TEAL):
        out.append(
            {
                "coords": coords,
                "bbox": bbox_coords(coords),
                "area": int(coords.shape[0]),
                "fragments": 1,
            }
        )
    return out


def _spans_overlap(a0: int, a1: int, b0: int, b1: int) -> bool:
    return max(int(a0), int(b0)) <= min(int(a1), int(b1))


def _bboxes_share_row_or_col(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return _spans_overlap(ar0, ar1, br0, br1) or _spans_overlap(ac0, ac1, bc0, bc1)


def merged_fragment_groups(fragments: list[dict]) -> list[dict]:
    groups = [
        {
            "coords": fragment["coords"].copy(),
            "bbox": fragment["bbox"],
            "fragments": int(fragment["fragments"]),
        }
        for fragment in fragments
    ]
    changed = True
    while changed:
        changed = False
        for i in range(len(groups)):
            if changed:
                break
            for j in range(i + 1, len(groups)):
                if _bboxes_share_row_or_col(groups[i]["bbox"], groups[j]["bbox"]):
                    coords = np.vstack([groups[i]["coords"], groups[j]["coords"]])
                    groups[i] = {
                        "coords": coords,
                        "bbox": bbox_coords(coords),
                        "fragments": int(groups[i]["fragments"] + groups[j]["fragments"]),
                    }
                    del groups[j]
                    changed = True
                    break
    for group in groups:
        r0, c0, r1, c1 = group["bbox"]
        group["height"] = int(r1 - r0 + 1)
        group["width"] = int(c1 - c0 + 1)
        group["area"] = int(group["height"] * group["width"])
    return groups


def main(grid: np.ndarray) -> np.ndarray:
    """Merge teal fragments into rectangles, then hollow large ones and solid-fill small ones.

    Input structure:
    - Black is the background.
    - Foreground pixels are teal fragments; each fragment is a 4-connected teal component.
    - Fragments belonging to one hidden rectangle may be disconnected, but their bounding boxes overlap in row span or in column span.

    Rule:
    - Find all 4-connected teal fragments.
    - Merge fragments transitively when their bounding boxes overlap in row span or in column span.
    - For each merged bounding box, compute its area.
    - If the area is at least 9, keep the 1-pixel teal border and fill the strict interior yellow.
    - If the area is below 9, fill the whole bounding box red.

    Key insight:
    - The same fragment-merging step recovers every hidden rectangle, and only after that recovered area determines which color transformation branch applies.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = get_bg(g)
    colors = set(map(int, np.unique(g)))
    assert colors <= {bg, TEAL}
    fragments = find_teal_fragments(g)
    assert fragments

    out = g.copy()
    for group in merged_fragment_groups(fragments):
        r0, c0, r1, c1 = group["bbox"]
        h = group["height"]
        w = group["width"]
        if group["area"] >= 9:
            out[r0 : r1 + 1, c0 : c1 + 1] = TEAL
            if h > 2 and w > 2:
                out[r0 + 1 : r1, c0 + 1 : c1] = YELLOW
        else:
            out[r0 : r1 + 1, c0 : c1 + 1] = RED
    return out


def _randint(rng: random.Random, lo: int, hi: int) -> int:
    return int(rng.randint(lo, hi))


def _weighted_choice(rng: random.Random, items, weights):
    total = float(sum(weights))
    x = rng.random() * total
    acc = 0.0
    for item, weight in zip(items, weights):
        acc += float(weight)
        if x <= acc:
            return item
    return items[-1]


def _scatter_extra(rng: random.Random, total: int, slots: int) -> list[int]:
    values = [0] * slots
    for _ in range(total):
        values[rng.randrange(slots)] += 1
    return values


def _choose_areas(rng: random.Random, count: int) -> list[int]:
    small_count = 1 if count == 3 else _weighted_choice(rng, [1, 2], [3, 2])
    big_count = count - small_count

    chosen = []
    first_small = _weighted_choice(rng, SMALL_AREAS, [1, 2, 3])
    first_big = _weighted_choice(rng, BIG_AREAS, [4, 3, 2, 2, 1, 1])
    chosen.extend([first_small, first_big])

    remaining_small = [area for area in SMALL_AREAS if area != first_small]
    remaining_big = [area for area in BIG_AREAS if area != first_big]

    while sum(area < 9 for area in chosen) < small_count:
        pick = rng.choice(remaining_small)
        remaining_small.remove(pick)
        chosen.append(pick)
    while len(chosen) < count:
        if not remaining_big:
            raise RuntimeError("ran out of distinct big areas")
        pick = rng.choice(remaining_big)
        remaining_big.remove(pick)
        chosen.append(pick)

    rng.shuffle(chosen)
    return chosen


def _sample_layout(rng: random.Random):
    count = _randint(rng, *RECT_COUNT_RANGE)
    areas = _choose_areas(rng, count)
    dims = [rng.choice(AREA_TO_DIMS[area]) for area in areas]

    min_h = 2 * EDGE_MARGIN + sum(h for h, _ in dims) + MIN_GAP * (count - 1)
    min_w = 2 * EDGE_MARGIN + sum(w for _, w in dims) + MIN_GAP * (count - 1)
    if min_h > GRID_MAX or min_w > GRID_MAX:
        return None

    grid_h = _randint(rng, max(GRID_MIN, min_h), GRID_MAX)
    grid_w = _randint(rng, max(GRID_MIN, min_w), GRID_MAX)
    extra_rows = _scatter_extra(rng, grid_h - min_h, count + 1)
    extra_cols = _scatter_extra(rng, grid_w - min_w, count + 1)

    specs = []
    r = EDGE_MARGIN + extra_rows[0]
    c = EDGE_MARGIN + extra_cols[0]
    for idx, (area, (h, w)) in enumerate(zip(areas, dims)):
        specs.append({"bbox": (r, c, r + h - 1, c + w - 1), "area": area, "dims": (h, w)})
        r += h
        c += w
        if idx + 1 < count:
            r += MIN_GAP + extra_rows[idx + 1]
            c += MIN_GAP + extra_cols[idx + 1]

    return grid_h, grid_w, specs


def _small_patch(h: int, w: int, rng: random.Random) -> np.ndarray:
    patch = np.zeros((h, w), dtype=int)
    if (h, w) == (2, 4):
        patterns = [
            np.array([[1, 1, 0, 1], [1, 0, 1, 1]], dtype=bool),
            np.array([[1, 0, 1, 1], [1, 1, 0, 1]], dtype=bool),
        ]
        patch[rng.choice(patterns)] = TEAL
        return patch
    if (h, w) == (4, 2):
        return _small_patch(2, 4, rng).T
    patch[:, :] = TEAL
    return patch


def _large_two_fragment_patch(h: int, w: int, rng: random.Random) -> np.ndarray | None:
    patch = np.zeros((h, w), dtype=int)
    max_top = min(w - 2, 4)
    if max_top < 1 or h < 3 or w < 3:
        return None
    top_end = _randint(rng, 1, max_top)
    bottom_lo = max(1, w - 4)
    bottom_hi = min(w - 2, top_end)
    if bottom_lo > bottom_hi:
        return None
    bottom_start = _randint(rng, bottom_lo, bottom_hi)
    left_tail = _randint(rng, 1, min(2, h - 2))
    right_tail = _randint(rng, 1, min(2, h - 2))

    patch[0, 0 : top_end + 1] = TEAL
    patch[1 : left_tail + 1, 0] = TEAL
    patch[h - 1, bottom_start:w] = TEAL
    patch[h - 1 - right_tail : h - 1, w - 1] = TEAL
    return patch


def _large_three_fragment_patch(h: int, w: int, rng: random.Random) -> np.ndarray | None:
    if h < 5 or w < 4:
        return None
    patch = np.zeros((h, w), dtype=int)

    mid_len = _randint(rng, 3, min(6, w - 1))
    mid_start = _randint(rng, 1, w - mid_len)
    mid_end = mid_start + mid_len - 1
    top_end = _randint(rng, max(1, mid_start), min(4, w - 2))
    bottom_lo = max(1, w - 4)
    bottom_hi = min(w - 2, mid_end)
    if bottom_lo > bottom_hi:
        return None
    bottom_start = _randint(rng, bottom_lo, bottom_hi)
    mid_row = _randint(rng, 2, h - 3)
    left_tail = _randint(rng, 1, min(2, h - 2))
    right_tail = _randint(rng, 1, min(2, h - 2))

    patch[0, 0 : top_end + 1] = TEAL
    patch[1 : left_tail + 1, 0] = TEAL
    patch[mid_row, mid_start : mid_end + 1] = TEAL
    patch[h - 1, bottom_start:w] = TEAL
    patch[h - 1 - right_tail : h - 1, w - 1] = TEAL
    return patch


def _make_patch(h: int, w: int, area: int, rng: random.Random) -> np.ndarray | None:
    if area < 9:
        patch = _small_patch(h, w, rng)
    else:
        if h >= 5 and w >= 4 and rng.random() < 0.5:
            patch = _large_three_fragment_patch(h, w, rng)
            if patch is None:
                patch = _large_two_fragment_patch(h, w, rng)
        else:
            patch = _large_two_fragment_patch(h, w, rng)
        if patch is None:
            return None

    fragments = find_teal_fragments(patch)
    if not fragments:
        return None
    if any(not (3 <= fragment["area"] <= 8) for fragment in fragments):
        return None
    groups = merged_fragment_groups(fragments)
    if len(groups) != 1:
        return None
    group = groups[0]
    if group["bbox"] != (0, 0, h - 1, w - 1):
        return None
    if group["area"] != area:
        return None
    if area >= 9 and (h < 3 or w < 3):
        return None
    return patch


def _validate(grid: np.ndarray, specs: list[dict]) -> bool:
    if not isinstance(grid, np.ndarray) or grid.ndim != 2 or max(grid.shape) > 30:
        return False
    colors = set(map(int, np.unique(grid)))
    if colors - {BLACK, TEAL}:
        return False
    fragments = find_teal_fragments(grid)
    if not fragments or any(not (3 <= fragment["area"] <= 8) for fragment in fragments):
        return False

    groups = merged_fragment_groups(fragments)
    if len(groups) != len(specs):
        return False
    groups.sort(key=lambda group: group["bbox"])
    specs_sorted = sorted(specs, key=lambda spec: spec["bbox"])
    if [group["bbox"] for group in groups] != [spec["bbox"] for spec in specs_sorted]:
        return False
    if [group["area"] for group in groups] != [spec["area"] for spec in specs_sorted]:
        return False
    if len({group["area"] for group in groups}) != len(groups):
        return False
    if not any(group["area"] < 9 for group in groups):
        return False
    if not any(group["area"] >= 9 for group in groups):
        return False
    if sum(group["fragments"] >= 2 for group in groups) < 2:
        return False
    if not any(group["area"] >= 9 and group["fragments"] >= 2 for group in groups):
        return False

    for i, left in enumerate(groups):
        for right in groups[i + 1 :]:
            if _spans_overlap(left["bbox"][0], left["bbox"][2], right["bbox"][0], right["bbox"][2]):
                return False
            if _spans_overlap(left["bbox"][1], left["bbox"][3], right["bbox"][1], right["bbox"][3]):
                return False

    out = main(grid)
    if out.shape != grid.shape:
        return False
    if not np.any(out == RED) or not np.any(out == YELLOW):
        return False

    for group in groups:
        r0, c0, r1, c1 = group["bbox"]
        region = out[r0 : r1 + 1, c0 : c1 + 1]
        if group["area"] < 9:
            if not np.all(region == RED):
                return False
        else:
            if not np.all(region[0, :] == TEAL):
                return False
            if not np.all(region[-1, :] == TEAL):
                return False
            if not np.all(region[:, 0] == TEAL):
                return False
            if not np.all(region[:, -1] == TEAL):
                return False
            if group["height"] <= 2 or group["width"] <= 2:
                return False
            if not np.all(region[1:-1, 1:-1] == YELLOW):
                return False

    return True


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    layout = _sample_layout(rng)
    if layout is None:
        return None
    grid_h, grid_w, specs = layout
    grid = np.zeros((grid_h, grid_w), dtype=int)

    for spec in specs:
        r0, c0, r1, c1 = spec["bbox"]
        h, w = spec["dims"]
        patch = _make_patch(h, w, spec["area"], rng)
        if patch is None:
            return None
        grid[r0 : r1 + 1, c0 : c1 + 1] = patch

    if not _validate(grid, specs):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid fragmented-rectangle input")
