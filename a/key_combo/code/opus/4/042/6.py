import random
from collections import Counter, deque

import numpy as np


class Color:
    BLACK = 0
    GREEN = 3
    YELLOW = 4
    GREY = 5


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.flatten())).most_common(1)[0][0])


def in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = map(int, shape)
    return 0 <= int(r) < h and 0 <= int(c) < w


def neighbors4(r: int, c: int):
    r = int(r)
    c = int(c)
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def neighbors8(r: int, c: int):
    r = int(r)
    c = int(c)
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            yield r + dr, c + dc


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _find_components_of_color(grid: np.ndarray, color: int, connectivity: int = 8) -> list[np.ndarray]:
    mask = grid == int(color)
    seen = np.zeros(mask.shape, dtype=bool)
    components: list[np.ndarray] = []
    nbrs = neighbors8 if int(connectivity) == 8 else neighbors4
    for r, c in np.argwhere(mask):
        r = int(r)
        c = int(c)
        if seen[r, c]:
            continue
        q = deque([(r, c)])
        seen[r, c] = True
        pts: list[tuple[int, int]] = []
        while q:
            rr, cc = q.popleft()
            pts.append((rr, cc))
            for nr, nc in nbrs(rr, cc):
                if in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    q.append((int(nr), int(nc)))
        components.append(np.array(pts, dtype=int))
    return components


def _component_mask(coords: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    r0, c0, r1, c1 = bbox_coords(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return mask, (r0, c0, r1, c1)


def _shape_key(mask: np.ndarray) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(int(v) for v in row) for row in mask.astype(int))


def _exterior_background_mask(grid: np.ndarray, bg: int) -> np.ndarray:
    h, w = grid.shape
    exterior = np.zeros((h, w), dtype=bool)
    q: deque[tuple[int, int]] = deque()
    for r in range(h):
        for c in (0, w - 1):
            if int(grid[r, c]) == int(bg) and not exterior[r, c]:
                exterior[r, c] = True
                q.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if int(grid[r, c]) == int(bg) and not exterior[r, c]:
                exterior[r, c] = True
                q.append((r, c))
    while q:
        r, c = q.popleft()
        for nr, nc in neighbors4(r, c):
            if in_bounds(grid.shape, nr, nc) and not exterior[nr, nc] and int(grid[nr, nc]) == int(bg):
                exterior[nr, nc] = True
                q.append((int(nr), int(nc)))
    return exterior


def _hole_mask(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    exterior = np.zeros((h, w), dtype=bool)
    q: deque[tuple[int, int]] = deque()
    for r in range(h):
        for c in (0, w - 1):
            if not mask[r, c] and not exterior[r, c]:
                exterior[r, c] = True
                q.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if not mask[r, c] and not exterior[r, c]:
                exterior[r, c] = True
                q.append((r, c))
    while q:
        r, c = q.popleft()
        for nr, nc in neighbors4(r, c):
            if in_bounds(mask.shape, nr, nc) and not mask[nr, nc] and not exterior[nr, nc]:
                exterior[nr, nc] = True
                q.append((int(nr), int(nc)))
    return (~mask) & (~exterior)


def _analyze_scene(grid: np.ndarray) -> tuple[int, int, list[dict], Counter]:
    bg = get_bg(grid)
    fg_colors = sorted(set(map(int, grid.flatten())) - {bg})
    assert len(fg_colors) == 1, "expected exactly one foreground color"
    fg = int(fg_colors[0])
    components = _find_components_of_color(grid, fg, connectivity=8)
    info = []
    for coords in components:
        mask, box = _component_mask(coords)
        holes = _hole_mask(mask)
        info.append(
            {
                "coords": coords,
                "bbox": box,
                "mask": mask,
                "shape_key": _shape_key(mask),
                "area": int(coords.shape[0]),
                "holes": holes,
                "hole_count": int(holes.sum()),
            }
        )
    counts = Counter(item["shape_key"] for item in info)
    return bg, fg, info, counts


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill the holes of singleton grey shapes and add green exterior outlines around repeated grey shapes.

    Input structure:
    - The background is black, and every foreground object is a grey 8-connected component.
    - Objects are compared in their own minimal bounding boxes using the local (r, c) mask pattern.
    - Green and yellow are absent from the input and are reserved for the output markings.

    Rule:
    - Group all grey objects by their normalized bbox mask.
    - For every shape that appears exactly once, recolor its enclosed black holes to yellow.
    - For every shape that appears multiple times, recolor to green every exterior black cell that 8-touches that object.
    - Keep all original grey pixels unchanged.

    Key insight:
    - The visible operation on each object is not determined locally by size or position; it is gated by the global frequency of that exact normalized shape.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert Color.GREEN not in g and Color.YELLOW not in g, "green and yellow are output-only colors"

    bg, fg, components, counts = _analyze_scene(g)
    assert fg == Color.GREY, "expected grey foreground objects"

    exterior = _exterior_background_mask(g, bg)
    out = g.copy()
    for item in components:
        r0, c0, _r1, _c1 = item["bbox"]
        if counts[item["shape_key"]] == 1:
            holes = item["holes"]
            if holes.any():
                rr, cc = np.argwhere(holes).T
                out[r0 + rr, c0 + cc] = Color.YELLOW
        else:
            for r, c in item["coords"]:
                for nr, nc in neighbors8(int(r), int(c)):
                    if in_bounds(g.shape, nr, nc) and int(g[nr, nc]) == int(bg) and exterior[nr, nc]:
                        out[nr, nc] = Color.GREEN
    return out


def _mask(rows: list[str]) -> np.ndarray:
    return np.array([[ch == "1" for ch in row] for row in rows], dtype=bool)


SHAPES = [
    {"name": "hole_3x3_ring", "mask": _mask(["111", "101", "111"]), "holey": True},
    {"name": "hole_4x3_tube", "mask": _mask(["111", "101", "101", "111"]), "holey": True},
    {"name": "hole_3x4_ring", "mask": _mask(["1111", "1001", "1111"]), "holey": True},
    {"name": "hole_4x4_ring", "mask": _mask(["1111", "1001", "1001", "1111"]), "holey": True},
    {"name": "solid_3x3_notch_a", "mask": _mask(["111", "111", "011"]), "holey": False},
    {"name": "solid_3x3_notch_b", "mask": _mask(["111", "110", "111"]), "holey": False},
    {"name": "solid_4x3_stair_a", "mask": _mask(["111", "111", "011", "011"]), "holey": False},
    {"name": "solid_4x3_stair_b", "mask": _mask(["111", "110", "111", "011"]), "holey": False},
    {"name": "solid_3x4_stair", "mask": _mask(["1111", "1111", "0011"]), "holey": False},
    {"name": "solid_3x4_barbell", "mask": _mask(["1111", "1100", "1111"]), "holey": False},
    {"name": "solid_4x4_block", "mask": _mask(["1111", "1111", "0011", "0011"]), "holey": False},
    {"name": "solid_4x4_sweep", "mask": _mask(["1111", "1110", "0111", "0011"]), "holey": False},
]

for shape in SHAPES:
    shape["area"] = int(shape["mask"].sum())
    shape["size"] = tuple(map(int, shape["mask"].shape))


def _transforms(mask: np.ndarray) -> list[np.ndarray]:
    out = []
    cur = mask
    for _ in range(4):
        out.append(cur.copy())
        out.append(np.fliplr(cur).copy())
        cur = np.rot90(cur)
    dedup: list[np.ndarray] = []
    seen = set()
    for item in out:
        key = _shape_key(item)
        if key not in seen:
            seen.add(key)
            dedup.append(item)
    return dedup


def _coords_from_mask(mask: np.ndarray, top: tuple[int, int]) -> np.ndarray:
    r0, c0 = map(int, top)
    pts = np.argwhere(mask)
    pts[:, 0] += r0
    pts[:, 1] += c0
    return pts.astype(int)


def _chebyshev_distance(a: np.ndarray, b: np.ndarray) -> int:
    best = 10**9
    for r1, c1 in a:
        for r2, c2 in b:
            best = min(best, max(abs(int(r1) - int(r2)), abs(int(c1) - int(c2))))
    return int(best)


def _choose_groups() -> list[dict] | None:
    holey = [shape for shape in SHAPES if shape["holey"]]
    unique = random.choice(holey)

    close = [
        shape
        for shape in SHAPES
        if shape["name"] != unique["name"] and int(shape["area"]) == int(unique["area"])
    ]
    if not close:
        close = [
            shape
            for shape in SHAPES
            if shape["name"] != unique["name"] and tuple(shape["size"]) == tuple(unique["size"])
        ]
    if not close:
        return None
    repeat_a = random.choice(close)

    if repeat_a["holey"]:
        second_pool = [shape for shape in SHAPES if shape["name"] not in {unique["name"], repeat_a["name"]}]
    else:
        second_pool = [
            shape
            for shape in SHAPES
            if shape["name"] not in {unique["name"], repeat_a["name"]} and shape["holey"]
        ]
    if not second_pool:
        return None
    repeat_b = random.choice(second_pool)

    groups = []
    for role, base in zip(("unique", "repeat_a", "repeat_b"), (unique, repeat_a, repeat_b)):
        groups.append(
            {
                "role": role,
                "name": base["name"],
                "mask": random.choice(_transforms(base["mask"])),
                "holey": bool(base["holey"]),
            }
        )
    keys = [_shape_key(group["mask"]) for group in groups]
    if len(set(keys)) != 3:
        return None
    return groups


def _place_instances(instances: list[dict], height: int, width: int) -> list[dict] | None:
    placements: list[dict] = []
    border = 1
    ordered = sorted(instances, key=lambda item: int(item["mask"].sum()), reverse=True)
    for item in ordered:
        mask = item["mask"]
        h, w = mask.shape
        if height - h - border < border or width - w - border < border:
            return None
        candidates = [
            (r, c)
            for r in range(border, height - h - border + 1)
            for c in range(border, width - w - border + 1)
        ]
        random.shuffle(candidates)
        placed = False
        for top in candidates[:250]:
            coords = _coords_from_mask(mask, top)
            if any(_chebyshev_distance(coords, other["coords"]) < 2 for other in placements):
                continue
            placements.append({**item, "top": top, "coords": coords})
            placed = True
            break
        if not placed:
            return None
    return placements


def _shared_outline_exists(grid: np.ndarray, repeated: list[dict], bg: int) -> bool:
    exterior = _exterior_background_mask(grid, bg)
    seen: Counter = Counter()
    for item in repeated:
        touched = set()
        for r, c in item["coords"]:
            for nr, nc in neighbors8(int(r), int(c)):
                if in_bounds(grid.shape, nr, nc) and int(grid[nr, nc]) == int(bg) and exterior[nr, nc]:
                    touched.add((int(nr), int(nc)))
        for cell in touched:
            seen[cell] += 1
    return any(v >= 2 for v in seen.values())


def _validate_generated(grid: np.ndarray, want_shared_outline: bool) -> bool:
    if grid.ndim != 2 or max(grid.shape) > 30:
        return False
    if Color.GREEN in grid or Color.YELLOW in grid:
        return False

    bg, fg, components, counts = _analyze_scene(grid)
    if bg != Color.BLACK or fg != Color.GREY:
        return False
    if not (5 <= len(components) <= 7):
        return False
    if any(not (6 <= item["area"] <= 14) for item in components):
        return False

    count_hist = sorted(counts.values())
    if count_hist[0] != 1 or any(v not in (2, 3) for v in count_hist[1:]):
        return False
    if list(counts.values()).count(1) != 1:
        return False
    if len(counts) != 3:
        return False

    singleton = [item for item in components if counts[item["shape_key"]] == 1]
    repeated = [item for item in components if counts[item["shape_key"]] > 1]
    if len(singleton) != 1 or not repeated:
        return False
    if singleton[0]["hole_count"] == 0:
        return False
    if not any(item["hole_count"] > 0 for item in repeated):
        return False

    if not any(
        item["area"] == singleton[0]["area"] or item["mask"].shape == singleton[0]["mask"].shape
        for item in repeated
    ):
        return False

    for i in range(len(components)):
        for j in range(i + 1, len(components)):
            if _chebyshev_distance(components[i]["coords"], components[j]["coords"]) < 2:
                return False

    if want_shared_outline and not _shared_outline_exists(grid, repeated, bg):
        return False

    out = main(grid)
    if not np.any(out == Color.YELLOW):
        return False
    if not np.any(out == Color.GREEN):
        return False
    if np.any(grid[grid == Color.GREY] != out[grid == Color.GREY]):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    groups = _choose_groups()
    if groups is None:
        return None

    count_a, count_b = random.choice([(2, 2), (3, 2), (2, 3), (3, 3)])
    instances = []
    for group, count in zip(groups, (1, count_a, count_b)):
        for idx in range(count):
            instances.append(
                {
                    "role": group["role"],
                    "name": f'{group["name"]}_{idx}',
                    "mask": group["mask"],
                    "holey": group["holey"],
                }
            )

    height = random.randint(16, 22)
    width = random.randint(17, 24)
    placements = _place_instances(instances, height, width)
    if placements is None:
        return None

    grid = np.full((height, width), Color.BLACK, dtype=int)
    for item in placements:
        grid[item["coords"][:, 0], item["coords"][:, 1]] = Color.GREY

    want_shared_outline = random.random() < 0.35
    if not _validate_generated(grid, want_shared_outline=want_shared_outline):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
