from __future__ import annotations

from collections import Counter, deque
import itertools
import random

import numpy as np


_NEIGHBORS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _dominant_color(grid: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(values[np.argmax(counts)])


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    assert coords.size > 0, "expected a nonempty mask"
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return mask[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1].copy()


def _shape_key(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    cropped = _crop_mask(np.asarray(mask, dtype=bool))
    return tuple(map(int, cropped.shape)), cropped.tobytes()


def _mask_connected(mask: np.ndarray) -> bool:
    mask = np.asarray(mask, dtype=bool)
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False

    start = tuple(map(int, coords[0]))
    seen = {start}
    q: deque[tuple[int, int]] = deque([start])
    h, w = map(int, mask.shape)

    while q:
        r, c = q.popleft()
        for dr, dc in _NEIGHBORS4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                q.append((nr, nc))

    return len(seen) == int(mask.sum())


def _mask_has_hole(mask: np.ndarray) -> bool:
    cropped = _crop_mask(np.asarray(mask, dtype=bool))
    h, w = map(int, cropped.shape)
    empty = ~cropped
    if not empty.any():
        return False

    padded = np.pad(empty, 1, constant_values=True)
    seen = np.zeros_like(padded, dtype=bool)
    q: deque[tuple[int, int]] = deque([(0, 0)])
    seen[0, 0] = True
    ph, pw = map(int, padded.shape)

    while q:
        r, c = q.popleft()
        for dr, dc in _NEIGHBORS4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < ph and 0 <= nc < pw and padded[nr, nc] and not seen[nr, nc]:
                seen[nr, nc] = True
                q.append((nr, nc))

    return bool((empty & ~seen[1 : h + 1, 1 : w + 1]).any())


def _find_components(grid: np.ndarray, bg: int) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = map(int, g.shape)
    seen = np.zeros((h, w), dtype=bool)
    components: list[dict] = []

    for r in range(h):
        for c in range(w):
            color = int(g[r, c])
            if color == bg or seen[r, c]:
                continue

            q: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []

            while q:
                cr, cc = q.popleft()
                coords.append((cr, cc))
                for dr, dc in _NEIGHBORS4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(g[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))

            arr = np.asarray(coords, dtype=int)
            r0, c0 = arr.min(axis=0)
            r1, c1 = arr.max(axis=0)
            mask = np.zeros((int(r1 - r0 + 1), int(c1 - c0 + 1)), dtype=bool)
            mask[arr[:, 0] - r0, arr[:, 1] - c0] = True
            components.append(
                {
                    "color": color,
                    "coords": arr,
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    "mask": mask,
                    "shape_key": _shape_key(mask),
                }
            )

    return components


def _candidate_lock_shape(component: dict) -> tuple[tuple[int, int], bytes] | None:
    mask = np.asarray(component["mask"], dtype=bool)
    if mask.all():
        return None
    cavity = ~mask
    if not _mask_connected(cavity):
        return None
    return _shape_key(cavity)


def _resolve_pairs(components: list[dict]) -> dict[int, int]:
    n = len(components)
    assert n > 0 and n % 2 == 0, "expected an even number of foreground objects"

    key_shapes = [comp["shape_key"] for comp in components]
    lock_shapes = [_candidate_lock_shape(comp) for comp in components]
    solutions: list[dict[int, int]] = []

    for lock_ids_tuple in itertools.combinations(range(n), n // 2):
        lock_ids = set(lock_ids_tuple)
        lock_counter = Counter()
        locks_by_shape: dict[tuple[tuple[int, int], bytes], list[int]] = {}
        valid = True

        for idx in lock_ids:
            shape = lock_shapes[idx]
            if shape is None:
                valid = False
                break
            lock_counter[shape] += 1
            locks_by_shape.setdefault(shape, []).append(idx)
        if not valid:
            continue

        key_counter = Counter(key_shapes[idx] for idx in range(n) if idx not in lock_ids)
        if lock_counter != key_counter:
            continue

        mapping: dict[int, int] = {}
        for shape, count in lock_counter.items():
            if count != 1:
                valid = False
                break
            key_matches = [idx for idx in range(n) if idx not in lock_ids and key_shapes[idx] == shape]
            if len(key_matches) != 1:
                valid = False
                break
            lock_idx = locks_by_shape[shape][0]
            key_idx = key_matches[0]
            if int(components[lock_idx]["color"]) == int(components[key_idx]["color"]):
                valid = False
                break
            mapping[lock_idx] = key_idx

        if valid:
            solutions.append(mapping)

    assert len(solutions) == 1, "expected a unique lock/key decomposition"
    return solutions[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match each lock's missing bg cavity to the key with the same shape, move the
    key into the cavity, and repaint the restored rectangle in the key color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a nonempty 2D grid"

    bg = _dominant_color(g)
    components = _find_components(g, bg)
    pairing = _resolve_pairs(components)

    out = g.copy()

    for key_idx in pairing.values():
        kr0, kc0, kr1, kc1 = components[key_idx]["bbox"]
        out[kr0 : kr1 + 1, kc0 : kc1 + 1] = bg

    for lock_idx, key_idx in pairing.items():
        lr0, lc0, lr1, lc1 = components[lock_idx]["bbox"]
        out[lr0 : lr1 + 1, lc0 : lc1 + 1] = int(components[key_idx]["color"])

    return out


# --------------- generator ---------------

_GRID_H_RANGE = (14, 30)
_GRID_W_RANGE = (14, 30)
_PAIR_COUNT_RANGE = (1, 3)
_RECT_H_RANGE = (4, 8)
_RECT_W_RANGE = (4, 8)
_PLACEMENT_PADDING = 1
_SHAPE_ATTEMPTS = 80
_PAIR_ATTEMPTS = 80
_GRID_ATTEMPTS = 40
_GEN_RETRIES = 400


def _touches_border(mask: np.ndarray) -> bool:
    return bool(mask[0].any() or mask[-1].any() or mask[:, 0].any() or mask[:, -1].any())


def _bbox_preserved(mask: np.ndarray) -> bool:
    return bool(mask[0].any() and mask[-1].any() and mask[:, 0].any() and mask[:, -1].any())


def _grow_polyomino(allowed: np.ndarray, target_area: int, seed_cells: list[tuple[int, int]]) -> np.ndarray | None:
    seeds = seed_cells[:]
    random.shuffle(seeds)
    h, w = map(int, allowed.shape)

    for sr, sc in seeds:
        mask = np.zeros((h, w), dtype=bool)
        mask[sr, sc] = True
        frontier = {
            (sr + dr, sc + dc)
            for dr, dc in _NEIGHBORS4
            if 0 <= sr + dr < h and 0 <= sc + dc < w and allowed[sr + dr, sc + dc]
        }

        while int(mask.sum()) < target_area:
            frontier = {cell for cell in frontier if not mask[cell] and allowed[cell]}
            if not frontier:
                break
            nr, nc = random.choice(tuple(frontier))
            mask[nr, nc] = True
            for dr, dc in _NEIGHBORS4:
                rr, cc = nr + dr, nc + dc
                if 0 <= rr < h and 0 <= cc < w and allowed[rr, cc] and not mask[rr, cc]:
                    frontier.add((rr, cc))

        if int(mask.sum()) == target_area:
            return mask

    return None


def _random_cavity(rect_h: int, rect_w: int, mode: str) -> np.ndarray | None:
    allowed = np.ones((rect_h, rect_w), dtype=bool)
    if mode == "interior":
        allowed[0, :] = False
        allowed[-1, :] = False
        allowed[:, 0] = False
        allowed[:, -1] = False
        if not allowed.any():
            return None
        seed_cells = [tuple(map(int, cell)) for cell in np.argwhere(allowed)]
    else:
        seed_cells = [
            (r, c)
            for r in range(rect_h)
            for c in range(rect_w)
            if r in (0, rect_h - 1) or c in (0, rect_w - 1)
        ]

    allowed_area = int(allowed.sum())
    max_area = min(allowed_area, max(1, rect_h * rect_w // 3 + 2))
    min_area = 1
    if mode == "edge":
        max_area = max(min_area, min(max_area, rect_h * rect_w - 2))
    if max_area < min_area:
        return None

    for _ in range(_SHAPE_ATTEMPTS):
        area = random.randint(min_area, max_area)
        cavity = _grow_polyomino(allowed, area, seed_cells)
        if cavity is None:
            continue
        if mode == "interior" and _touches_border(cavity):
            continue
        if mode == "edge" and not _touches_border(cavity):
            continue
        if _mask_has_hole(cavity):
            continue
        lock_mask = ~cavity
        if not _mask_connected(lock_mask):
            continue
        if not _bbox_preserved(lock_mask):
            continue
        return cavity

    return None


def _build_pair(lock_color: int, key_color: int, used_shapes: set[tuple[tuple[int, int], bytes]]) -> dict | None:
    for _ in range(_PAIR_ATTEMPTS):
        rect_h = random.randint(*_RECT_H_RANGE)
        rect_w = random.randint(*_RECT_W_RANGE)
        mode = random.choice(("interior", "edge"))
        cavity = _random_cavity(rect_h, rect_w, mode)
        if cavity is None:
            continue

        key_mask = _crop_mask(cavity)
        shape = _shape_key(key_mask)
        if shape in used_shapes:
            continue

        lock_patch = np.full((rect_h, rect_w), lock_color, dtype=int)
        lock_patch[cavity] = -1

        key_patch = np.full(key_mask.shape, -1, dtype=int)
        key_patch[key_mask] = key_color

        used_shapes.add(shape)
        return {
            "mode": mode,
            "lock_color": lock_color,
            "key_color": key_color,
            "cavity": cavity,
            "key_mask": key_mask,
            "shape_key": shape,
            "lock_patch": lock_patch,
            "key_patch": key_patch,
        }

    return None


def _sample_grid_shape(objects: list[dict], rect_area_sum: int) -> tuple[int, int]:
    bbox_need = sum((obj["h"] + 2 * _PLACEMENT_PADDING) * (obj["w"] + 2 * _PLACEMENT_PADDING) for obj in objects)
    target_area = max(3 * rect_area_sum, bbox_need)
    max_h = max(obj["h"] for obj in objects)
    max_w = max(obj["w"] for obj in objects)
    base = int(np.ceil(np.sqrt(target_area)))

    h_lo = max(_GRID_H_RANGE[0], max_h + 1, base - 3)
    h_hi = min(_GRID_H_RANGE[1], base + 4)
    if h_lo > h_hi:
        h_lo = h_hi = min(_GRID_H_RANGE[1], max(h_lo, _GRID_H_RANGE[0]))

    height = random.randint(h_lo, h_hi)
    w_lo = max(_GRID_W_RANGE[0], max_w + 1, int(np.ceil(target_area / height)))
    w_hi = min(_GRID_W_RANGE[1], max(w_lo, base + 6))
    if w_lo > w_hi:
        w_lo = w_hi = min(_GRID_W_RANGE[1], max(w_lo, _GRID_W_RANGE[0]))
    width = random.randint(w_lo, w_hi)

    return height, width


def _place_objects(height: int, width: int, objects: list[dict]) -> dict[int, tuple[int, int]] | None:
    blocked = np.zeros((height, width), dtype=bool)
    placements: dict[int, tuple[int, int]] = {}
    order = sorted(range(len(objects)), key=lambda i: objects[i]["h"] * objects[i]["w"], reverse=True)

    for idx in order:
        obj = objects[idx]
        h, w = int(obj["h"]), int(obj["w"])
        candidates = [
            (r, c)
            for r in range(height - h + 1)
            for c in range(width - w + 1)
            if not blocked[r : r + h, c : c + w].any()
        ]
        if not candidates:
            return None
        r0, c0 = random.choice(candidates)
        placements[idx] = (r0, c0)
        rr0 = max(0, r0 - _PLACEMENT_PADDING)
        cc0 = max(0, c0 - _PLACEMENT_PADDING)
        rr1 = min(height, r0 + h + _PLACEMENT_PADDING)
        cc1 = min(width, c0 + w + _PLACEMENT_PADDING)
        blocked[rr0:rr1, cc0:cc1] = True

    return placements


def _render_input(bg: int, pairs: list[dict], placements: dict[int, tuple[int, int]]) -> tuple[np.ndarray, list[dict]]:
    objects = []
    for pair_idx, pair in enumerate(pairs):
        objects.append({"kind": "lock", "pair_idx": pair_idx, "patch": pair["lock_patch"]})
        objects.append({"kind": "key", "pair_idx": pair_idx, "patch": pair["key_patch"]})

    height = max(r + obj["patch"].shape[0] for idx, (r, _c) in placements.items() for obj in [objects[idx]])
    width = max(c + obj["patch"].shape[1] for idx, (_r, c) in placements.items() for obj in [objects[idx]])
    grid = np.full((height, width), bg, dtype=int)
    placement_info: list[dict] = []

    for idx, obj in enumerate(objects):
        r0, c0 = placements[idx]
        patch = obj["patch"]
        h, w = patch.shape
        mask = patch != -1
        grid[r0 : r0 + h, c0 : c0 + w][mask] = patch[mask]
        placement_info.append({"kind": obj["kind"], "pair_idx": obj["pair_idx"], "r0": r0, "c0": c0, "h": h, "w": w})

    return grid, placement_info


def _expected_output(inp: np.ndarray, bg: int, pairs: list[dict], placement_info: list[dict]) -> np.ndarray:
    out = inp.copy()
    locks = {info["pair_idx"]: info for info in placement_info if info["kind"] == "lock"}
    keys = {info["pair_idx"]: info for info in placement_info if info["kind"] == "key"}

    for pair_idx, pair in enumerate(pairs):
        key_info = keys[pair_idx]
        out[
            int(key_info["r0"]) : int(key_info["r0"]) + int(key_info["h"]),
            int(key_info["c0"]) : int(key_info["c0"]) + int(key_info["w"]),
        ] = bg

        lock_info = locks[pair_idx]
        out[
            int(lock_info["r0"]) : int(lock_info["r0"]) + int(lock_info["h"]),
            int(lock_info["c0"]) : int(lock_info["c0"]) + int(lock_info["w"]),
        ] = int(pair["key_color"])

    return out


def _gen_attempt() -> np.ndarray | None:
    pair_count = random.randint(*_PAIR_COUNT_RANGE)
    palette = random.sample(range(10), 1 + 2 * pair_count)
    bg = palette[0]
    pair_colors = palette[1:]
    used_shapes: set[tuple[tuple[int, int], bytes]] = set()

    pairs = []
    for i in range(pair_count):
        pair = _build_pair(pair_colors[2 * i], pair_colors[2 * i + 1], used_shapes)
        if pair is None:
            return None
        pairs.append(pair)

    objects = []
    rect_area_sum = 0
    for pair in pairs:
        lock_h, lock_w = pair["lock_patch"].shape
        key_h, key_w = pair["key_patch"].shape
        objects.append({"h": int(lock_h), "w": int(lock_w)})
        objects.append({"h": int(key_h), "w": int(key_w)})
        rect_area_sum += int(lock_h) * int(lock_w)

    for _ in range(_GRID_ATTEMPTS):
        height, width = _sample_grid_shape(objects, rect_area_sum)
        placements = _place_objects(height, width, objects)
        if placements is None:
            continue

        inp, placement_info = _render_input(bg, pairs, placements)
        if inp.shape != (height, width):
            padded = np.full((height, width), bg, dtype=int)
            padded[: inp.shape[0], : inp.shape[1]] = inp
            inp = padded

        if _dominant_color(inp) != bg:
            continue

        expected = _expected_output(inp, bg, pairs, placement_info)
        try:
            actual = main(inp)
        except AssertionError:
            continue
        if np.array_equal(actual, expected):
            return inp

    return None


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid lock-and-key ARC input")
