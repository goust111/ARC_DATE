from __future__ import annotations

from collections import Counter, deque
import itertools

import numpy as np


BG = 0
OUT_BLUE = 1
OUT_RED = 2
LINE = 7
INPUT_COLORS = (1, 2, 3, 4, 5, 6, 8, 9)
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
RNG = np.random.default_rng()


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _mask_from_coords(coords: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    rel = coords - np.array([[r0, c0]])
    mask[rel[:, 0], rel[:, 1]] = True
    return mask


def _tight(mask: np.ndarray) -> np.ndarray:
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    return mask[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1]


def _unique_variants(mask: np.ndarray) -> list[np.ndarray]:
    variants: list[np.ndarray] = []
    seen = set()
    for flip in (False, True):
        base = np.fliplr(mask) if flip else mask
        for k in range(4):
            var = _tight(np.rot90(base, k))
            key = (var.shape, tuple(int(x) for x in var.ravel()))
            if key not in seen:
                seen.add(key)
                variants.append(var.copy())
    return variants


def _canonical_shape_key(mask: np.ndarray) -> tuple[tuple[int, int], tuple[int, ...]]:
    keys = []
    for var in _unique_variants(mask):
        keys.append((var.shape, tuple(int(x) for x in var.ravel())))
    return min(keys)


def _find_objects(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    H, W = g.shape
    visited = np.zeros((H, W), dtype=bool)
    out = []
    for r in range(H):
        for c in range(W):
            color = int(g[r, c])
            if visited[r, c] or color in (BG, LINE):
                continue
            q = deque([(r, c)])
            visited[r, c] = True
            coords = []
            while q:
                cr, cc = q.popleft()
                coords.append((cr, cc))
                for dr, dc in DIRS4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < H and 0 <= nc < W and not visited[nr, nc] and int(g[nr, nc]) == color:
                        visited[nr, nc] = True
                        q.append((nr, nc))
            arr = np.array(coords, dtype=int)
            bbox = _bbox_from_coords(arr)
            out.append(
                {
                    "color": color,
                    "coords": arr,
                    "bbox": bbox,
                    "mask": _mask_from_coords(arr, bbox),
                }
            )
    return out


def _paste_mask(grid: np.ndarray, mask: np.ndarray, r0: int, c0: int, color: int) -> None:
    h, w = mask.shape
    sl = grid[r0 : r0 + h, c0 : c0 + w]
    if np.any(sl[mask] != BG):
        raise AssertionError("object overlap")
    sl[mask] = color


def _can_fit(occ: np.ndarray, mask: np.ndarray, r0: int, c0: int) -> bool:
    h, w = mask.shape
    H, W = occ.shape
    if r0 < 0 or c0 < 0 or r0 + h > H or c0 + w > W:
        return False
    return not np.any(occ[r0 : r0 + h, c0 : c0 + w] & mask)


def _place_group(objects: list[dict], line_row: int, width: int, toward_top: bool) -> dict[int, int]:
    occ = np.zeros((line_row, width), dtype=bool)
    dest_rows: dict[int, int] = {}
    for obj in sorted(objects, key=lambda item: (item["bbox"][1], item["bbox"][0])):
        mask = obj["mask"]
        c0 = obj["bbox"][1]
        h = mask.shape[0]
        if toward_top:
            candidates = range(0, line_row - h + 1)
        else:
            candidates = range(line_row - h, -1, -1)
        for r0 in candidates:
            if _can_fit(occ, mask, r0, c0):
                occ[r0 : r0 + h, c0 : c0 + mask.shape[1]] |= mask
                dest_rows[id(obj)] = r0
                break
        else:
            raise AssertionError("could not compact object")
    return dest_rows


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move each shape upward or downward according to whether its canonical shape is unique, then recolor it by displacement parity.

    Input structure:
    - A black background contains 6-8 small monochrome 4-connected objects, all located above one full-width orange horizontal line.
    - Objects may share a binary shape up to rotation and reflection, even when their colors and orientations differ.
    - The orange line stays fixed and separates the upper active region from the empty lower region.

    Rule:
    - Compute each object's canonical binary shape by taking the lexicographically smallest tight mask over 4 rotations and optional horizontal flip.
    - Objects whose canonical shape appears at least twice are compacted downward within the upper region; unique-shape objects are compacted upward within the upper region.
    - Within each target region, process objects by increasing bbox left column, breaking equal columns by smaller original top row; keep columns fixed and move only vertically until each object is blocked by the boundary or an earlier object in the same region.
    - Recolor each moved object blue if its absolute row displacement is even, otherwise red; keep the orange line unchanged.

    Key insight:
    - Shape frequency under dihedral equivalence decides the gravity direction, while the final color is determined only after the actual compacted travel distance is known.
    """
    g = np.asarray(grid, dtype=int)
    H, W = g.shape
    line_rows = [r for r in range(H) if np.all(g[r] == LINE)]
    assert len(line_rows) == 1, "expected exactly one full-width orange line"
    line_row = line_rows[0]

    objects = _find_objects(g)
    assert 1 <= len(objects)
    for obj in objects:
        assert obj["bbox"][2] < line_row, "all objects must start above the orange line"
        obj["shape_key"] = _canonical_shape_key(obj["mask"])

    freq = Counter(obj["shape_key"] for obj in objects)
    upward = [obj for obj in objects if freq[obj["shape_key"]] == 1]
    downward = [obj for obj in objects if freq[obj["shape_key"]] >= 2]
    assert upward and downward, "need both unique and shared shape groups"

    dest_rows = {}
    dest_rows.update(_place_group(upward, line_row, W, toward_top=True))
    dest_rows.update(_place_group(downward, line_row, W, toward_top=False))

    out = np.zeros_like(g)
    out[line_row, :] = LINE
    for obj in objects:
        r0 = obj["bbox"][0]
        c0 = obj["bbox"][1]
        dest_r0 = dest_rows[id(obj)]
        color = OUT_RED if abs(dest_r0 - r0) % 2 == 1 else OUT_BLUE
        _paste_mask(out, obj["mask"], dest_r0, c0, color)
    return out


def _random_polyomino(area: int, *, require_asymmetric: bool, forbid: set) -> tuple[np.ndarray, tuple]:
    for _ in range(2000):
        bbox_h = int(RNG.integers(2, min(6, area + 2)))
        bbox_w = int(RNG.integers(2, min(6, area + 2)))
        start = (int(RNG.integers(bbox_h)), int(RNG.integers(bbox_w)))
        cells = {start}
        while len(cells) < area:
            frontier = []
            for r, c in cells:
                for dr, dc in DIRS4:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < bbox_h and 0 <= nc < bbox_w and (nr, nc) not in cells:
                        frontier.append((nr, nc))
            if not frontier:
                break
            cells.add(frontier[int(RNG.integers(len(frontier)))])
        if len(cells) != area:
            continue
        mask = np.zeros((bbox_h, bbox_w), dtype=bool)
        for r, c in cells:
            mask[r, c] = True
        mask = _tight(mask)
        if min(mask.shape) < 2:
            continue
        key = _canonical_shape_key(mask)
        if key in forbid:
            continue
        if require_asymmetric and len(_unique_variants(mask)) < 2:
            continue
        return mask, key
    raise RuntimeError("failed to create shape")


def _bottom_aligned_overlap(mask_a: np.ndarray, mask_b: np.ndarray) -> bool:
    h = max(mask_a.shape[0], mask_b.shape[0])
    w = max(mask_a.shape[1], mask_b.shape[1])
    canvas_a = np.zeros((h, w), dtype=bool)
    canvas_b = np.zeros((h, w), dtype=bool)
    canvas_a[h - mask_a.shape[0] :, : mask_a.shape[1]] = mask_a
    canvas_b[h - mask_b.shape[0] :, : mask_b.shape[1]] = mask_b
    return bool(np.any(canvas_a & canvas_b))


def _top_aligned_overlap(mask_a: np.ndarray, c0_a: int, mask_b: np.ndarray, c0_b: int) -> bool:
    left = min(c0_a, c0_b)
    right = max(c0_a + mask_a.shape[1], c0_b + mask_b.shape[1])
    h = max(mask_a.shape[0], mask_b.shape[0])
    w = right - left
    canvas_a = np.zeros((h, w), dtype=bool)
    canvas_b = np.zeros((h, w), dtype=bool)
    canvas_a[: mask_a.shape[0], c0_a - left : c0_a - left + mask_a.shape[1]] = mask_a
    canvas_b[: mask_b.shape[0], c0_b - left : c0_b - left + mask_b.shape[1]] = mask_b
    return bool(np.any(canvas_a & canvas_b))


def _can_place_input(occ: np.ndarray, mask: np.ndarray, r0: int, c0: int) -> bool:
    h, w = mask.shape
    H, W = occ.shape
    if r0 < 0 or c0 < 0 or r0 + h > H or c0 + w > W:
        return False
    coords = np.argwhere(mask)
    for rr, cc in coords:
        gr = r0 + int(rr)
        gc = c0 + int(cc)
        if occ[gr, gc]:
            return False
        for dr, dc in DIRS4:
            nr, nc = gr + dr, gc + dc
            if 0 <= nr < H and 0 <= nc < W and occ[nr, nc]:
                return False
    return True


def _occupy(occ: np.ndarray, mask: np.ndarray, r0: int, c0: int) -> None:
    coords = np.argwhere(mask)
    for rr, cc in coords:
        occ[r0 + int(rr), c0 + int(cc)] = True


def _sample_group_sizes(n_objects: int) -> list[int]:
    options = {
        6: ([2], [3], [2, 2]),
        7: ([2], [3], [2, 2], [3, 2]),
        8: ([2], [3], [2, 2], [3, 2]),
    }
    candidates = [list(x) for x in options[n_objects] if n_objects - sum(x) >= 2]
    return candidates[int(RNG.integers(len(candidates)))]


def _build_object_specs(H: int, W: int) -> list[dict] | None:
    n_objects = int(RNG.choice([6, 7, 8], p=[0.6, 0.35, 0.05]))
    shared_sizes = _sample_group_sizes(n_objects)
    unique_count = n_objects - sum(shared_sizes)
    line_row = H // 2
    if line_row < 7:
        return None

    used_keys = set()
    area = int(RNG.integers(3, 7))
    try:
        shared_template, shared_key = _random_polyomino(area, require_asymmetric=True, forbid=used_keys)
    except RuntimeError:
        return None
    shared_vars = _unique_variants(shared_template)
    overlap_pair = None
    for va, vb in itertools.permutations(shared_vars, 2):
        if _bottom_aligned_overlap(va, vb):
            overlap_pair = (va.copy(), vb.copy())
            break
    if overlap_pair is None:
        return None
    used_keys.add(shared_key)

    unique_same_area = None
    for _ in range(200):
        try:
            mask, key = _random_polyomino(area, require_asymmetric=False, forbid=used_keys)
        except RuntimeError:
            return None
        if key != shared_key:
            unique_same_area = (mask, key)
            break
    if unique_same_area is None:
        return None
    used_keys.add(unique_same_area[1])

    specs = []
    first_group_size = shared_sizes[0]
    variants = [overlap_pair[0], overlap_pair[1]]
    extra_vars = shared_vars.copy()
    RNG.shuffle(extra_vars)
    while len(variants) < first_group_size:
        variants.append(extra_vars[len(variants) % len(extra_vars)].copy())
    for idx in range(first_group_size):
        specs.append(
            {
                "kind": "shared",
                "group": 0,
                "mask": variants[idx].copy(),
                "shape_key": shared_key,
            }
        )

    for group_id, group_size in enumerate(shared_sizes[1:], start=1):
        try:
            template, key = _random_polyomino(int(RNG.integers(3, 7)), require_asymmetric=False, forbid=used_keys)
        except RuntimeError:
            return None
        used_keys.add(key)
        vars_for_group = _unique_variants(template)
        for _ in range(group_size):
            specs.append(
                {
                    "kind": "shared",
                    "group": group_id,
                    "mask": vars_for_group[int(RNG.integers(len(vars_for_group)))].copy(),
                    "shape_key": key,
                }
            )

    specs.append(
        {
            "kind": "unique",
            "group": 0,
            "mask": unique_same_area[0].copy(),
            "shape_key": unique_same_area[1],
        }
    )
    for group_id in range(1, unique_count):
        try:
            template, key = _random_polyomino(int(RNG.integers(3, 7)), require_asymmetric=False, forbid=used_keys)
        except RuntimeError:
            return None
        used_keys.add(key)
        variants = _unique_variants(template)
        specs.append(
            {
                "kind": "unique",
                "group": group_id,
                "mask": variants[int(RNG.integers(len(variants)))].copy(),
                "shape_key": key,
            }
        )

    colors = list(INPUT_COLORS)
    RNG.shuffle(colors)
    for spec, color in zip(specs, colors[: len(specs)]):
        spec["color"] = int(color)
    return specs


def _try_place_shared_stack(
    occ: np.ndarray, objs: list[dict], col_lo: int, col_hi: int, line_row: int
) -> bool:
    a, b = objs[0], objs[1]
    max_w = max(a["mask"].shape[1], b["mask"].shape[1])
    if col_hi - col_lo < max_w:
        return False
    for _ in range(200):
        c0 = int(RNG.integers(col_lo, col_hi - max_w + 1))
        ra_max = line_row - a["mask"].shape[0]
        rb_max = line_row - b["mask"].shape[0]
        if ra_max < 0 or rb_max < 0:
            return False
        ra = int(RNG.integers(0, max(1, ra_max + 1)))
        rb_lo = ra + a["mask"].shape[0] + 1
        if rb_lo > rb_max:
            continue
        rb = int(RNG.integers(rb_lo, rb_max + 1))
        if not _can_place_input(occ, a["mask"], ra, c0):
            continue
        occ2 = occ.copy()
        _occupy(occ2, a["mask"], ra, c0)
        if not _can_place_input(occ2, b["mask"], rb, c0):
            continue
        a["r0"], a["c0"] = ra, c0
        b["r0"], b["c0"] = rb, c0
        _occupy(occ, a["mask"], ra, c0)
        _occupy(occ, b["mask"], rb, c0)
        return True
    return False


def _try_place_unique_overlap(
    occ: np.ndarray, objs: list[dict], col_lo: int, col_hi: int, line_row: int
) -> bool:
    a, b = objs[0], objs[1]
    wa, wb = a["mask"].shape[1], b["mask"].shape[1]
    for _ in range(400):
        shift = int(RNG.integers(1, min(wa, wb)))
        usable = col_hi - col_lo
        base_max = usable - max(wa, shift + wb)
        if base_max < 0:
            continue
        c0a = int(RNG.integers(col_lo, col_lo + base_max + 1))
        c0b = c0a + shift
        if not _top_aligned_overlap(a["mask"], c0a, b["mask"], c0b):
            continue
        ra_max = line_row - a["mask"].shape[0]
        rb_max = line_row - b["mask"].shape[0]
        if ra_max < 0 or rb_max < 0:
            return False
        ra = int(RNG.integers(0, ra_max + 1))
        rb = int(RNG.integers(0, rb_max + 1))
        if not _can_place_input(occ, a["mask"], ra, c0a):
            continue
        occ2 = occ.copy()
        _occupy(occ2, a["mask"], ra, c0a)
        if not _can_place_input(occ2, b["mask"], rb, c0b):
            continue
        a["r0"], a["c0"] = ra, c0a
        b["r0"], b["c0"] = rb, c0b
        _occupy(occ, a["mask"], ra, c0a)
        _occupy(occ, b["mask"], rb, c0b)
        return True
    return False


def _try_place_random(
    occ: np.ndarray, obj: dict, col_lo: int, col_hi: int, line_row: int
) -> bool:
    h, w = obj["mask"].shape
    if col_hi - col_lo < w or line_row < h:
        return False
    for _ in range(300):
        c0 = int(RNG.integers(col_lo, col_hi - w + 1))
        r0 = int(RNG.integers(line_row - h + 1))
        if _can_place_input(occ, obj["mask"], r0, c0):
            obj["r0"], obj["c0"] = r0, c0
            _occupy(occ, obj["mask"], r0, c0)
            return True
    return False


def _render_input(H: int, W: int, objs: list[dict]) -> np.ndarray:
    grid = np.zeros((H, W), dtype=int)
    grid[H // 2, :] = LINE
    for obj in objs:
        _paste_mask(grid, obj["mask"], obj["r0"], obj["c0"], obj["color"])
    return grid


def _intervals_overlap(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return max(a[0], b[0]) <= min(a[1], b[1])


def _validate_instance(grid: np.ndarray) -> bool:
    H, W = grid.shape
    if H > 30 or W > 30:
        return False
    line_row = H // 2
    if not np.all(grid[line_row] == LINE):
        return False
    if np.any(grid[line_row + 1 :] != BG):
        return False

    objects = _find_objects(grid)
    if not (6 <= len(objects) <= 8):
        return False
    if any(obj["bbox"][2] >= line_row for obj in objects):
        return False
    if any(not (3 <= int(obj["mask"].sum()) <= 6) for obj in objects):
        return False

    freq = Counter(_canonical_shape_key(obj["mask"]) for obj in objects)
    if not any(v >= 2 for v in freq.values()):
        return False
    if not any(v == 1 for v in freq.values()):
        return False

    keys_by_area = {}
    for obj in objects:
        area = int(obj["mask"].sum())
        key = _canonical_shape_key(obj["mask"])
        keys_by_area.setdefault(area, set()).add(key)
    if not any(len(keys) >= 2 for keys in keys_by_area.values()):
        return False

    for obj in objects:
        obj["shape_key"] = _canonical_shape_key(obj["mask"])
    upward = [obj for obj in objects if freq[obj["shape_key"]] == 1]
    downward = [obj for obj in objects if freq[obj["shape_key"]] >= 2]
    dest_rows = {}
    try:
        dest_rows.update(_place_group(upward, line_row, W, toward_top=True))
        dest_rows.update(_place_group(downward, line_row, W, toward_top=False))
        out = main(grid)
    except AssertionError:
        return False
    if not np.all(out[line_row] == LINE):
        return False
    if np.any(out[line_row + 1 :] != BG):
        return False
    if not set(np.unique(out)) <= {BG, OUT_BLUE, OUT_RED, LINE}:
        return False

    odd = even = False
    collision_needed = False
    moved_up = moved_down = False
    for obj in objects:
        group_shared = freq[obj["shape_key"]] >= 2
        dest_r0 = dest_rows[id(obj)]
        c0 = obj["bbox"][1]
        mask = obj["mask"]
        h, w = mask.shape
        patch = out[dest_r0 : dest_r0 + h, c0 : c0 + w]
        expected_color = OUT_RED if abs(dest_r0 - obj["bbox"][0]) % 2 == 1 else OUT_BLUE
        if not np.all(patch[mask] == expected_color):
            return False
        if group_shared:
            moved_down |= dest_r0 > obj["bbox"][0]
        else:
            moved_up |= dest_r0 < obj["bbox"][0]
        odd |= expected_color == OUT_RED
        even |= expected_color == OUT_BLUE
        naive_row = 0 if not group_shared else line_row - h
        if dest_r0 != naive_row:
            collision_needed = True

    if not (odd and even and moved_up and moved_down and collision_needed):
        return False

    # Enforce an order-sensitive pair inside one target region.
    order_sensitive = False
    for a, b in itertools.combinations(objects, 2):
        key_a = _canonical_shape_key(a["mask"])
        key_b = _canonical_shape_key(b["mask"])
        if (freq[key_a] >= 2) != (freq[key_b] >= 2):
            continue
        ia = (a["bbox"][1], a["bbox"][3])
        ib = (b["bbox"][1], b["bbox"][3])
        if _intervals_overlap(ia, ib):
            order_sensitive = True
            break
    return order_sensitive


def _generate_attempt() -> np.ndarray | None:
    H = int(RNG.choice([19, 21, 23]))
    W = int(RNG.integers(18, 24))
    line_row = H // 2
    specs = _build_object_specs(H, W)
    if specs is None:
        return None

    shared_objs = [obj for obj in specs if obj["kind"] == "shared"]
    unique_objs = [obj for obj in specs if obj["kind"] == "unique"]
    if len(shared_objs) < 2 or len(unique_objs) < 2:
        return None

    occ = np.zeros((line_row, W), dtype=bool)
    split = W // 2
    shared_hi = max(split - 1, max(obj["mask"].shape[1] for obj in shared_objs))
    unique_lo = min(split + 1, W - max(obj["mask"].shape[1] for obj in unique_objs))
    if shared_hi >= unique_lo:
        return None
    stack_pair = shared_objs[:2]
    overlap_pair = unique_objs[:2]
    if not _try_place_shared_stack(occ, stack_pair, 0, shared_hi, line_row):
        return None
    if not _try_place_unique_overlap(occ, overlap_pair, unique_lo, W, line_row):
        return None

    remaining = [obj for obj in specs if "r0" not in obj]
    RNG.shuffle(remaining)
    for obj in remaining:
        if obj["kind"] == "shared":
            ok = _try_place_random(occ, obj, 0, shared_hi, line_row)
        else:
            ok = _try_place_random(occ, obj, unique_lo, W, line_row)
        if not ok:
            return None

    grid = _render_input(H, W, specs)
    if not _validate_instance(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(500):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid sample for puzzle 1")
