import random
from collections import Counter, defaultdict, deque

import numpy as np


MAX_GENERATION_ATTEMPTS = 500
GRID_MIN = 16
GRID_MAX = 30
PAIR_COUNT_CHOICES = (2, 3, 4)
BBOX_MIN = 3
BBOX_MAX = 7
PLACEMENT_PADDING = 1
MIN_OVERLAP_CELLS = 2
DIM_POOL = [(h, w) for h in range(BBOX_MIN, BBOX_MAX + 1) for w in range(BBOX_MIN, BBOX_MAX + 1)]


def get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _neighbors4(r: int, c: int, h: int, w: int):
    if r > 0:
        yield r - 1, c
    if r + 1 < h:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < w:
        yield r, c + 1


def _mask_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if len(coords) == 0:
        return False
    h, w = mask.shape
    start = tuple(coords[0])
    queue = deque([start])
    seen = {start}
    while queue:
        r, c = queue.popleft()
        for nr, nc in _neighbors4(r, c, h, w):
            if mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                queue.append((nr, nc))
    return len(seen) == int(mask.sum())


def _touches_all_sides(mask: np.ndarray) -> bool:
    return bool(mask[0].any() and mask[-1].any() and mask[:, 0].any() and mask[:, -1].any())


def _mask_from_coords(coords: list[tuple[int, int]], shape: tuple[int, int], origin: tuple[int, int]) -> np.ndarray:
    r0, c0 = origin
    mask = np.zeros(shape, dtype=bool)
    for r, c in coords:
        mask[r - r0, c - c0] = True
    return mask


def find_components(grid: np.ndarray, bg: int) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            color = int(g[r, c])
            if color == bg or seen[r, c]:
                continue
            queue = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while queue:
                cr, cc = queue.popleft()
                coords.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc, h, w):
                    if not seen[nr, nc] and int(g[nr, nc]) == color:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            rows = [rr for rr, _ in coords]
            cols = [cc for _, cc in coords]
            r0, r1 = min(rows), max(rows)
            c0, c1 = min(cols), max(cols)
            mask = _mask_from_coords(coords, (r1 - r0 + 1, c1 - c0 + 1), (r0, c0))
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": (r0, c0, r1, c1),
                    "mask": mask,
                    "area": int(mask.sum()),
                }
            )
    return comps


def _larger_object(a: dict, b: dict) -> dict:
    if a["area"] != b["area"]:
        return a if a["area"] > b["area"] else b
    return a if (a["bbox"][0], a["bbox"][1]) <= (b["bbox"][0], b["bbox"][1]) else b


def _pair_patch(a: dict, b: dict, bg: int) -> np.ndarray:
    # After the prompt's explicit logical-AND step, only the common cells remain.
    overlap = a["mask"] & b["mask"]
    patch = np.full(a["mask"].shape, bg, dtype=int)
    patch[overlap] = _larger_object(a, b)["color"]
    return patch


def _solve_from_components(components: list[dict], bg: int) -> np.ndarray:
    assert components, "expected at least one foreground object"
    groups: dict[tuple[int, int], list[dict]] = defaultdict(list)
    for comp in components:
        groups[tuple(map(int, comp["mask"].shape))].append(comp)

    records: list[tuple[int, int, np.ndarray]] = []
    for pair in groups.values():
        assert len(pair) == 2, "each bbox size must appear exactly twice"
        upper = min(pair, key=lambda obj: (obj["bbox"][0], obj["bbox"][1]))
        records.append((int(upper["bbox"][0]), int(upper["bbox"][1]), _pair_patch(pair[0], pair[1], bg)))

    records.sort(key=lambda item: (item[0], item[1]))
    out_h = sum(patch.shape[0] for _, _, patch in records) + max(0, len(records) - 1)
    out_w = max(patch.shape[1] for _, _, patch in records)
    out = np.full((out_h, out_w), bg, dtype=int)

    r = 0
    for idx, (_, _, patch) in enumerate(records):
        h, w = patch.shape
        out[r : r + h, :w] = patch
        r += h
        if idx + 1 < len(records):
            r += 1
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pair monochrome objects by equal bbox size, intersect each aligned mask pair,
    and stack the overlap patches from topmost pair to bottommost pair.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    components = find_components(g, bg)
    return _solve_from_components(components, bg)


def _random_shape_with_area(
    h: int,
    w: int,
    target_area: int,
    rng: random.Random,
    base_mask: np.ndarray | None = None,
) -> np.ndarray | None:
    if base_mask is None:
        base = np.ones((h, w), dtype=bool)
    else:
        base = np.array(base_mask, dtype=bool, copy=True)
    if int(base.sum()) < target_area:
        return None

    for _ in range(80):
        mask = base.copy()
        if int(mask.sum()) == target_area and _touches_all_sides(mask) and _mask_connected(mask):
            return mask
        while int(mask.sum()) > target_area:
            removable: list[tuple[int, int]] = []
            cells = [(int(r), int(c)) for r, c in np.argwhere(mask)]
            rng.shuffle(cells)
            for r, c in cells:
                trial = mask.copy()
                trial[r, c] = False
                if not _touches_all_sides(trial):
                    continue
                if not _mask_connected(trial):
                    continue
                removable.append((r, c))
            if not removable:
                break
            rr, cc = rng.choice(removable)
            mask[rr, cc] = False
        if int(mask.sum()) == target_area and _touches_all_sides(mask) and _mask_connected(mask):
            return mask
    return None


def _sample_nested_pair(h: int, w: int, rng: random.Random) -> tuple[np.ndarray, np.ndarray] | None:
    max_area = h * w
    min_area = max(h + w - 1, 5)
    if min_area >= max_area:
        return None
    for _ in range(80):
        big_lo = max(min_area + 1, (2 * max_area) // 3)
        if big_lo > max_area:
            big_lo = min_area + 1
        big_area = rng.randint(big_lo, max_area)
        small_area = rng.randint(min_area, big_area - 1)
        big_mask = _random_shape_with_area(h, w, big_area, rng)
        if big_mask is None:
            continue
        small_mask = _random_shape_with_area(h, w, small_area, rng, base_mask=big_mask)
        if small_mask is None:
            continue
        if np.array_equal(big_mask, small_mask):
            continue
        return big_mask, small_mask
    return None


def _sample_cross_pair(h: int, w: int, rng: random.Random) -> tuple[np.ndarray, np.ndarray] | None:
    max_area = h * w
    min_area = max(h + w - 1, (max_area + 1) // 2)
    if min_area >= max_area:
        return None
    for _ in range(120):
        area_a = rng.randint(min_area, max_area - 1)
        area_b = rng.randint(min_area, max_area)
        if area_a == area_b:
            continue
        mask_a = _random_shape_with_area(h, w, area_a, rng)
        mask_b = _random_shape_with_area(h, w, area_b, rng)
        if mask_a is None or mask_b is None:
            continue
        if np.array_equal(mask_a, mask_b):
            continue
        overlap = mask_a & mask_b
        if int(overlap.sum()) < MIN_OVERLAP_CELLS:
            continue
        if not np.any(mask_a & ~mask_b):
            continue
        if not np.any(mask_b & ~mask_a):
            continue
        return mask_a, mask_b
    return None


def _sample_pairs(bg: int, rng: random.Random) -> list[dict] | None:
    pair_count = rng.choice(PAIR_COUNT_CHOICES)
    dims = rng.sample(DIM_POOL, pair_count)
    if sum(h for h, _ in dims) + pair_count - 1 > GRID_MAX:
        return None
    fg_pool = [c for c in range(10) if c != bg]
    colors = rng.sample(fg_pool, 2 * pair_count)

    pairs: list[dict] = []
    for idx, (h, w) in enumerate(dims):
        if rng.random() < 0.5:
            masks = _sample_nested_pair(h, w, rng)
            mode = "nested"
        else:
            masks = _sample_cross_pair(h, w, rng)
            mode = "cross"
        if masks is None:
            return None
        mask_a, mask_b = masks
        pair_colors = colors[2 * idx : 2 * idx + 2]
        rng.shuffle(pair_colors)
        if int(mask_a.sum()) == int(mask_b.sum()):
            return None
        pairs.append(
            {
                "mode": mode,
                "size": (h, w),
                "objects": [
                    {"mask": mask_a, "color": int(pair_colors[0]), "area": int(mask_a.sum())},
                    {"mask": mask_b, "color": int(pair_colors[1]), "area": int(mask_b.sum())},
                ],
            }
        )
    return pairs


def _flatten_objects(pairs: list[dict]) -> list[dict]:
    objects: list[dict] = []
    for pair_id, pair in enumerate(pairs):
        for obj_id, obj in enumerate(pair["objects"]):
            record = dict(obj)
            record["pair_id"] = pair_id
            record["obj_id"] = obj_id
            objects.append(record)
    return objects


def _sample_grid_shape(objects: list[dict], rng: random.Random) -> tuple[int, int] | None:
    max_h = max(obj["mask"].shape[0] for obj in objects)
    max_w = max(obj["mask"].shape[1] for obj in objects)
    min_h = max(GRID_MIN, max_h + 2 * PLACEMENT_PADDING + 2)
    min_w = max(GRID_MIN, max_w + 2 * PLACEMENT_PADDING + 2)
    footprint = sum((obj["mask"].shape[0] + 2) * (obj["mask"].shape[1] + 2) for obj in objects)
    for _ in range(40):
        height = rng.randint(min_h, GRID_MAX)
        width = rng.randint(min_w, GRID_MAX)
        if height * width < footprint + 30:
            continue
        return height, width
    return None


def _place_objects(objects: list[dict], height: int, width: int, rng: random.Random) -> list[tuple[int, int]] | None:
    reserve = np.zeros((height, width), dtype=bool)
    placements: list[tuple[int, int] | None] = [None] * len(objects)
    order = sorted(
        range(len(objects)),
        key=lambda idx: objects[idx]["mask"].shape[0] * objects[idx]["mask"].shape[1],
        reverse=True,
    )

    for idx in order:
        h, w = objects[idx]["mask"].shape
        candidates: list[tuple[int, int]] = []
        for r in range(height - h + 1):
            for c in range(width - w + 1):
                r0 = max(0, r - PLACEMENT_PADDING)
                r1 = min(height, r + h + PLACEMENT_PADDING)
                c0 = max(0, c - PLACEMENT_PADDING)
                c1 = min(width, c + w + PLACEMENT_PADDING)
                if reserve[r0:r1, c0:c1].any():
                    continue
                candidates.append((r, c))
        if not candidates:
            return None
        r, c = rng.choice(candidates)
        placements[idx] = (r, c)
        r0 = max(0, r - PLACEMENT_PADDING)
        r1 = min(height, r + h + PLACEMENT_PADDING)
        c0 = max(0, c - PLACEMENT_PADDING)
        c1 = min(width, c + w + PLACEMENT_PADDING)
        reserve[r0:r1, c0:c1] = True
    return [pos for pos in placements if pos is not None]


def _build_grid(height: int, width: int, bg: int, objects: list[dict], placements: list[tuple[int, int]]) -> np.ndarray:
    grid = np.full((height, width), bg, dtype=int)
    for obj, (r0, c0) in zip(objects, placements, strict=True):
        h, w = obj["mask"].shape
        patch = grid[r0 : r0 + h, c0 : c0 + w]
        patch[obj["mask"]] = obj["color"]
    return grid


def _synthetic_components(objects: list[dict], placements: list[tuple[int, int]]) -> list[dict]:
    comps: list[dict] = []
    for obj, (r0, c0) in zip(objects, placements, strict=True):
        h, w = obj["mask"].shape
        comps.append(
            {
                "color": int(obj["color"]),
                "mask": np.array(obj["mask"], copy=True),
                "area": int(obj["area"]),
                "bbox": (int(r0), int(c0), int(r0 + h - 1), int(c0 + w - 1)),
                "pair_id": int(obj["pair_id"]),
            }
        )
    return comps


def _valid_component_structure(components: list[dict]) -> bool:
    groups: dict[tuple[int, int], list[dict]] = defaultdict(list)
    for comp in components:
        groups[tuple(map(int, comp["mask"].shape))].append(comp)
    if not groups:
        return False
    for pair in groups.values():
        if len(pair) != 2:
            return False
        if len({int(obj["color"]) for obj in pair}) != 2:
            return False
        if int(pair[0]["area"]) == int(pair[1]["area"]):
            return False
    return True


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    bg = rng.randint(0, 9)
    pairs = _sample_pairs(bg, rng)
    if pairs is None:
        return None

    objects = _flatten_objects(pairs)
    shape = _sample_grid_shape(objects, rng)
    if shape is None:
        return None
    height, width = shape

    placements = _place_objects(objects, height, width, rng)
    if placements is None:
        return None

    grid = _build_grid(height, width, bg, objects, placements)
    counts = Counter(grid.ravel().tolist())
    if counts[bg] <= max(count for color, count in counts.items() if color != bg):
        return None

    expected = _solve_from_components(_synthetic_components(objects, placements), bg)
    components = find_components(grid, bg)
    if len(components) != len(objects):
        return None
    if not _valid_component_structure(components):
        return None

    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_GENERATION_ATTEMPTS} attempts")
