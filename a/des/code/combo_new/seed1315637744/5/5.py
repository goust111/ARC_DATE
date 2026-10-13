import random
from collections import Counter, defaultdict

import numpy as np


BG = 0
BRIDGE = 5
OBJECT_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]

CELL = 6
GAP = 2
SLOT_ROWS = 2
SLOT_COLS = 4
MAX_GEN_RETRIES = 300


def _neighbors4(r: int, c: int, h: int, w: int):
    if r > 0:
        yield r - 1, c
    if r + 1 < h:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < w:
        yield r, c + 1


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc, h, w):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _trim(mask: np.ndarray) -> np.ndarray:
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    assert rows.size and cols.size
    return mask[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1].copy()


def _mask_order(mask: np.ndarray) -> tuple[int, int, bytes]:
    arr = np.asarray(mask, dtype=np.uint8)
    return int(arr.shape[0]), int(arr.shape[1]), arr.tobytes()


def _unique_transforms(mask: np.ndarray) -> list[np.ndarray]:
    seen: dict[tuple[int, int, bytes], np.ndarray] = {}
    base = np.asarray(mask, dtype=bool)
    for flipped in (False, True):
        cur = np.fliplr(base) if flipped else base
        for k in range(4):
            arr = _trim(np.rot90(cur, k))
            key = _mask_order(arr)
            if key not in seen:
                seen[key] = arr
    return list(seen.values())


def _canonical_mask(mask: np.ndarray) -> np.ndarray:
    return min(_unique_transforms(mask), key=_mask_order)


def _canonical_key(mask: np.ndarray) -> tuple[int, int, bytes]:
    return _mask_order(_canonical_mask(mask))


def _shape_ports_horizontal(mask: np.ndarray) -> list[int]:
    h, w = mask.shape
    return [r for r in range(h) if mask[r, 0] and mask[r, w - 1]]


def _slot_origin(slot_r: int, slot_c: int) -> tuple[int, int]:
    return slot_r * (CELL + GAP), slot_c * (CELL + GAP)


def _build_shape_catalog():
    base_shapes = [
        ("I3", np.array([[1, 1, 1]], dtype=bool)),
        ("L3", np.array([[1, 0], [1, 1]], dtype=bool)),
        ("O4", np.array([[1, 1], [1, 1]], dtype=bool)),
        ("T4", np.array([[1, 1, 1], [0, 1, 0]], dtype=bool)),
        ("L4", np.array([[1, 0], [1, 0], [1, 1]], dtype=bool)),
        ("Z4", np.array([[1, 1, 0], [0, 1, 1]], dtype=bool)),
        ("P5", np.array([[1, 1], [1, 1], [1, 0]], dtype=bool)),
        ("PLUS5", np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)),
        ("U5", np.array([[1, 0, 1], [1, 1, 1]], dtype=bool)),
    ]

    catalog = []
    seen_keys = set()
    for name, mask in base_shapes:
        variants = _unique_transforms(mask)
        h_variants = []
        for arr in variants:
            rows = _shape_ports_horizontal(arr)
            if rows:
                h_variants.append({"mask": arr, "port_rows": rows})
        canonical = min(variants, key=_mask_order)
        key = _mask_order(canonical)
        assert key not in seen_keys
        seen_keys.add(key)
        catalog.append(
            {
                "name": name,
                "area": int(mask.sum()),
                "canonical": canonical,
                "key": key,
                "variants": variants,
                "h_variants": h_variants,
            }
        )
    return catalog


SHAPES = _build_shape_catalog()
SHAPES_BY_AREA: dict[int, list[int]] = defaultdict(list)
for idx, shape in enumerate(SHAPES):
    SHAPES_BY_AREA[int(shape["area"])].append(idx)


def _find_objects(grid: np.ndarray) -> tuple[list[dict], np.ndarray]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    mask = (g != BG) & (g != BRIDGE)
    comp_list = _components(mask)
    obj_index = np.full((h, w), -1, dtype=int)
    objects: list[dict] = []
    for oid, comp in enumerate(comp_list):
        rows = [r for r, _ in comp]
        cols = [c for _, c in comp]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        colors = {int(g[r, c]) for r, c in comp}
        assert len(colors) == 1
        color = int(next(iter(colors)))
        mask_crop = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        for r, c in comp:
            obj_index[r, c] = oid
            mask_crop[r - r0, c - c0] = True
        objects.append(
            {
                "id": oid,
                "color": color,
                "coords": comp,
                "bbox": (r0, c0, r1, c1),
                "mask": mask_crop,
            }
        )
    return objects, obj_index


def main(grid: np.ndarray) -> np.ndarray:
    """
    Group colored objects by grey-bridge connectivity, vote a shared shape per group, and redraw every member in that voted shape.

    Input structure:
    - Background is black.
    - Non-black, non-grey cells form several small monochrome 4-connected objects, each in its own color.
    - Some object pairs are linked by one-cell-wide grey horizontal or vertical bridge segments; transitive bridge chains define object groups.

    Rule:
    - Remove every grey bridge from the output.
    - Find object groups from direct or indirect grey connections.
    - For each group with at least two objects, crop each object to its tight binary mask, normalize that mask up to rotation and reflection, and count shape-class frequencies.
    - Choose the most frequent normalized shape class; if counts tie, choose the tied class with larger area.
    - Redraw every object in that group using one fixed canonical bitmap of the winning class, anchored at the original tight-bbox top-left and recolored with the object's original color.
    - Groups of size one stay unchanged apart from bridge removal.

    Key insight:
    - The task combines graph connectivity through grey bridges with rotation/flip-invariant shape voting: the bridges say which objects must agree, and the normalized majority shape says what every member should become.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    out = g.copy()
    out[out == BRIDGE] = BG

    objects, obj_index = _find_objects(g)
    if not objects:
        return out

    parent = list(range(len(objects)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for comp in _components(g == BRIDGE):
        touched = set()
        for r, c in comp:
            for nr, nc in _neighbors4(r, c, h, w):
                oid = int(obj_index[nr, nc])
                if oid >= 0:
                    touched.add(oid)
        touched = sorted(touched)
        for i in range(1, len(touched)):
            union(touched[0], touched[i])

    groups: dict[int, list[int]] = defaultdict(list)
    for oid in range(len(objects)):
        groups[find(oid)].append(oid)

    for members in groups.values():
        if len(members) <= 1:
            continue

        freq: Counter = Counter()
        winners: dict[tuple[int, int, bytes], np.ndarray] = {}
        areas: dict[tuple[int, int, bytes], int] = {}
        for oid in members:
            mask = np.asarray(objects[oid]["mask"], dtype=bool)
            canon = _canonical_mask(mask)
            key = _mask_order(canon)
            freq[key] += 1
            winners[key] = canon
            areas[key] = int(canon.sum())

        winner_key = max(
            freq,
            key=lambda key: (
                int(freq[key]),
                int(areas[key]),
                int(winners[key].shape[0]),
                int(winners[key].shape[1]),
                winners[key].astype(np.uint8).tobytes(),
            ),
        )
        winner_mask = np.asarray(winners[winner_key], dtype=bool)
        win_h, win_w = winner_mask.shape

        for oid in members:
            r0, c0, r1, c1 = objects[oid]["bbox"]
            color = int(objects[oid]["color"])
            out[r0 : r1 + 1, c0 : c1 + 1] = BG
            rr1 = min(h, r0 + win_h)
            cc1 = min(w, c0 + win_w)
            submask = winner_mask[: rr1 - r0, : cc1 - c0]
            patch = out[r0:rr1, c0:cc1]
            patch[submask] = color
            out[r0:rr1, c0:cc1] = patch

    return out


def _pick_distinct_shape(rng: random.Random, forbidden: set[int] | None = None, area: int | None = None) -> int:
    forbidden = forbidden or set()
    candidates = [idx for idx, shape in enumerate(SHAPES) if idx not in forbidden and (area is None or int(shape["area"]) == area)]
    assert candidates
    return int(rng.choice(candidates))


def _sample_group_shape_ids(mode: str, size: int, rng: random.Random) -> list[int]:
    if mode == "majority":
        winner = _pick_distinct_shape(rng)
        if size == 2:
            ids = [winner, winner]
        elif size == 3:
            other = _pick_distinct_shape(rng, {winner})
            ids = [winner, winner, other]
        else:
            pattern = rng.choice(["2-1-1", "3-1", "4"])
            if pattern == "2-1-1":
                other1 = _pick_distinct_shape(rng, {winner})
                other2 = _pick_distinct_shape(rng, {winner, other1})
                ids = [winner, winner, other1, other2]
            elif pattern == "3-1":
                other = _pick_distinct_shape(rng, {winner})
                ids = [winner, winner, winner, other]
            else:
                ids = [winner, winner, winner, winner]
    else:
        if size == 2:
            areas = rng.sample(sorted(SHAPES_BY_AREA.keys()), 2)
            a = _pick_distinct_shape(rng, area=int(areas[0]))
            b = _pick_distinct_shape(rng, {a}, area=int(areas[1]))
            ids = [a, b]
        elif size == 3:
            areas = rng.sample(sorted(SHAPES_BY_AREA.keys()), 3)
            ids = [_pick_distinct_shape(rng, area=int(area)) for area in areas]
        else:
            areas = rng.sample(sorted(SHAPES_BY_AREA.keys()), 2)
            a = _pick_distinct_shape(rng, area=int(areas[0]))
            b = _pick_distinct_shape(rng, {a}, area=int(areas[1]))
            ids = [a, a, b, b]
    rng.shuffle(ids)
    return [int(x) for x in ids]


def _pick_horizontal_variants(shape_ids: list[int], rng: random.Random):
    for _ in range(200):
        picks = []
        lo, hi = 0, CELL - 1
        for shape_id in shape_ids:
            shape = SHAPES[int(shape_id)]
            var = rng.choice(shape["h_variants"])
            mask = np.asarray(var["mask"], dtype=bool)
            port_row = int(rng.choice(var["port_rows"]))
            h = int(mask.shape[0])
            lo = max(lo, port_row)
            hi = min(hi, port_row + CELL - h)
            picks.append((mask, port_row))
        if lo > hi:
            continue
        track = rng.randint(lo, hi)
        placed = []
        for mask, port_row in picks:
            anchor_r = track - port_row
            anchor_c = rng.randint(0, CELL - int(mask.shape[1]))
            placed.append({"mask": mask, "anchor_r": int(anchor_r), "anchor_c": int(anchor_c)})
        return int(track), placed
    return None


def _place_singleton(shape_id: int, rng: random.Random) -> dict:
    shape = SHAPES[int(shape_id)]
    mask = np.asarray(rng.choice(shape["variants"]), dtype=bool)
    anchor_r = rng.randint(0, CELL - int(mask.shape[0]))
    anchor_c = rng.randint(0, CELL - int(mask.shape[1]))
    return {"mask": mask, "anchor_r": int(anchor_r), "anchor_c": int(anchor_c)}


def _draw_mask(canvas: np.ndarray, origin_r: int, origin_c: int, mask: np.ndarray, color: int) -> None:
    h, w = mask.shape
    patch = canvas[origin_r : origin_r + h, origin_c : origin_c + w]
    assert np.all(patch[mask] == BG)
    patch[mask] = int(color)
    canvas[origin_r : origin_r + h, origin_c : origin_c + w] = patch


def _draw_horizontal_bridge(canvas: np.ndarray, left_obj: dict, right_obj: dict) -> None:
    r = int(left_obj["bridge_row"])
    left_end = int(left_obj["abs_c"] + left_obj["mask"].shape[1] - 1)
    right_end = int(right_obj["abs_c"])
    assert right_end - left_end >= 2
    span = canvas[r, left_end + 1 : right_end]
    assert np.all(span == BG)
    canvas[r, left_end + 1 : right_end] = BRIDGE


def _tight_crop_with_padding(canvas: np.ndarray, rng: random.Random) -> np.ndarray:
    pts = np.argwhere(canvas != BG)
    assert pts.size
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    core = canvas[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1].copy()
    h, w = core.shape

    extra_h = rng.randint(0, min(4, 30 - h))
    extra_w = rng.randint(0, min(4, 30 - w))
    top = rng.randint(0, extra_h)
    left = rng.randint(0, extra_w)
    out = np.zeros((h + extra_h, w + extra_w), dtype=int)
    out[top : top + h, left : left + w] = core
    return out


def _validate_generated_grid(grid: np.ndarray) -> None:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and max(g.shape) <= 30
    objects, obj_index = _find_objects(g)
    assert objects
    assert len(objects) <= len(OBJECT_COLORS)
    colors = [int(obj["color"]) for obj in objects]
    assert len(colors) == len(set(colors))
    for obj in objects:
        area = int(np.asarray(obj["mask"], dtype=bool).sum())
        assert 3 <= area <= 8
    for comp in _components(g == BRIDGE):
        rows = {r for r, _ in comp}
        cols = {c for _, c in comp}
        assert len(rows) == 1 or len(cols) == 1
        touched = set()
        for r, c in comp:
            for nr, nc in _neighbors4(r, c, g.shape[0], g.shape[1]):
                oid = int(obj_index[nr, nc])
                if oid >= 0:
                    touched.add(oid)
        assert len(touched) == 2

    out = main(g)
    assert out.shape == g.shape
    assert not np.any(out == BRIDGE)


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    canvas = np.zeros((SLOT_ROWS * CELL + (SLOT_ROWS - 1) * GAP, SLOT_COLS * CELL + (SLOT_COLS - 1) * GAP), dtype=int)
    all_slots = [(r, c) for r in range(SLOT_ROWS) for c in range(SLOT_COLS)]

    row_group_lengths: dict[int, int] = {}
    row_group_starts: dict[int, int] = {}
    group_rows = [0, 1]
    rng.shuffle(group_rows)

    if rng.random() < 0.5:
        chosen_rows = [group_rows[0]]
    else:
        chosen_rows = group_rows[:]

    for row in chosen_rows:
        size = rng.choice([2, 3, 4])
        start = rng.randint(0, SLOT_COLS - size)
        row_group_lengths[row] = int(size)
        row_group_starts[row] = int(start)

    used_slots = set()
    groups = []
    for row in sorted(row_group_lengths):
        size = int(row_group_lengths[row])
        start = int(row_group_starts[row])
        slots = [(row, col) for col in range(start, start + size)]
        used_slots.update(slots)
        mode = rng.choice(["majority", "tie_area"])
        groups.append({"mode": mode, "slots": slots})

    remaining_slots = [slot for slot in all_slots if slot not in used_slots]
    rng.shuffle(remaining_slots)
    singleton_slots: list[tuple[int, int]] = []
    want_singleton = rng.random() < 0.5
    max_singletons = len(OBJECT_COLORS) - sum(len(group["slots"]) for group in groups)
    for slot in remaining_slots:
        if len(singleton_slots) >= max_singletons:
            break
        if rng.random() < 0.45:
            singleton_slots.append(slot)
    if want_singleton and max_singletons > 0 and remaining_slots and not singleton_slots:
        singleton_slots.append(remaining_slots[0])

    total_objects = sum(len(group["slots"]) for group in groups) + len(singleton_slots)
    if total_objects == 0 or total_objects > len(OBJECT_COLORS):
        return None

    colors = rng.sample(OBJECT_COLORS, total_objects)
    color_idx = 0

    for group in groups:
        slots = list(group["slots"])
        size = len(slots)
        shape_ids = _sample_group_shape_ids(str(group["mode"]), size, rng)
        picked = _pick_horizontal_variants(shape_ids, rng)
        if picked is None:
            return None
        track_row, placed = picked
        objects = []
        for (slot_r, slot_c), shape_id, placement in zip(slots, shape_ids, placed):
            base_r, base_c = _slot_origin(slot_r, slot_c)
            abs_r = base_r + int(placement["anchor_r"])
            abs_c = base_c + int(placement["anchor_c"])
            color = int(colors[color_idx])
            color_idx += 1
            obj = {
                "shape_id": int(shape_id),
                "color": color,
                "mask": np.asarray(placement["mask"], dtype=bool),
                "abs_r": abs_r,
                "abs_c": abs_c,
                "bridge_row": base_r + int(track_row),
            }
            _draw_mask(canvas, abs_r, abs_c, obj["mask"], color)
            objects.append(obj)
        for left, right in zip(objects, objects[1:]):
            _draw_horizontal_bridge(canvas, left, right)

    forbidden = set()
    for group in groups:
        for slot in group["slots"]:
            forbidden.add(slot)

    for slot in singleton_slots:
        if slot in forbidden:
            return None
        shape_id = _pick_distinct_shape(rng)
        placement = _place_singleton(shape_id, rng)
        base_r, base_c = _slot_origin(slot[0], slot[1])
        abs_r = base_r + int(placement["anchor_r"])
        abs_c = base_c + int(placement["anchor_c"])
        color = int(colors[color_idx])
        color_idx += 1
        _draw_mask(canvas, abs_r, abs_c, np.asarray(placement["mask"], dtype=bool), color)

    cropped = _tight_crop_with_padding(canvas, rng)
    _validate_generated_grid(cropped)
    return cropped


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid ARC grid")
