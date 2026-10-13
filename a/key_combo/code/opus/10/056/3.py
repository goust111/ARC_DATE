from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


BG = 0
COLOR_POOL = (1, 2, 3, 4, 6, 7, 8, 9)
GRID_SIZE_RANGE = (15, 18)
MAX_GENERATION_ATTEMPTS = 400


HOLE_TEMPLATE_MASKS = (
    np.array(
        [
            [1, 1, 1],
            [1, 0, 1],
            [1, 1, 1],
            [0, 1, 0],
        ],
        dtype=bool,
    ),
    np.array(
        [
            [1, 1, 1],
            [1, 0, 1],
            [1, 0, 1],
            [1, 1, 1],
            [0, 1, 0],
        ],
        dtype=bool,
    ),
    np.array(
        [
            [1, 1, 1, 1],
            [1, 0, 0, 1],
            [1, 1, 1, 1],
            [0, 1, 0, 0],
        ],
        dtype=bool,
    ),
)

OPEN_TEMPLATE_MASKS = (
    np.array(
        [
            [1, 0, 1],
            [1, 0, 1],
            [1, 1, 1],
        ],
        dtype=bool,
    ),
    np.array(
        [
            [1, 1, 1, 1],
            [1, 0, 0, 0],
            [1, 1, 1, 1],
        ],
        dtype=bool,
    ),
    np.array(
        [
            [1, 1, 0],
            [0, 1, 0],
            [0, 1, 1],
            [0, 0, 1],
        ],
        dtype=bool,
    ),
    np.array(
        [
            [1, 1, 1, 0],
            [1, 0, 0, 0],
            [1, 1, 1, 1],
            [0, 0, 0, 1],
        ],
        dtype=bool,
    ),
    np.array(
        [
            [1, 1, 1, 1, 1],
            [0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0],
        ],
        dtype=bool,
    ),
)


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _component_coords(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            components.append(np.array(cells, dtype=int))
    return components


def _has_enclosed_hole(mask: np.ndarray) -> bool:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    for r in range(h):
        for c in range(w):
            if mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            touches_bbox_edge = r in (0, h - 1) or c in (0, w - 1)
            while q:
                rr, cc = q.popleft()
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        if nr in (0, h - 1) or nc in (0, w - 1):
                            touches_bbox_edge = True
                        q.append((nr, nc))
            if not touches_bbox_edge:
                return True
    return False


def _mirror_coords(coords: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, _c0, r1, _c1 = bbox
    mirrored = coords.copy()
    mirrored[:, 0] = r0 + r1 - mirrored[:, 0]
    return mirrored


def _extract_objects(grid: np.ndarray, bg: int) -> list[dict]:
    objects = []
    for color in sorted(int(c) for c in np.unique(grid) if int(c) != bg):
        for coords in _component_coords(grid == color):
            r0, c0, r1, c1 = _bbox_coords(coords)
            local = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
            local[coords[:, 0] - r0, coords[:, 1] - c0] = True
            objects.append(
                {
                    "color": int(color),
                    "coords": coords,
                    "bbox": (r0, c0, r1, c1),
                    "bbox_area": int((r1 - r0 + 1) * (c1 - c0 + 1)),
                    "mask": local,
                    "has_hole": _has_enclosed_hole(local),
                }
            )
    return objects


def _render_specs(shape: tuple[int, int], specs: list[dict]) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    for spec in specs:
        rows, cols = np.nonzero(spec["mask"])
        grid[rows + spec["top"], cols + spec["left"]] = int(spec["color"])
    return grid


def _unique_variants(mask: np.ndarray) -> list[np.ndarray]:
    variants = []
    seen = set()
    for rot in range(4):
        rotated = np.rot90(mask, rot)
        for flip in (False, True):
            cand = np.fliplr(rotated) if flip else rotated
            key = (cand.shape, cand.tobytes())
            if key not in seen:
                seen.add(key)
                variants.append(cand.copy())
    return variants


def _mask_changes_under_vertical_mirror(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    bbox = (0, 0, mask.shape[0] - 1, mask.shape[1] - 1)
    mirrored = np.zeros_like(mask)
    mirrored_coords = _mirror_coords(coords, bbox)
    mirrored[mirrored_coords[:, 0], mirrored_coords[:, 1]] = True
    return not np.array_equal(mask, mirrored)


def _open_variants(mask: np.ndarray) -> list[np.ndarray]:
    return [cand for cand in _unique_variants(mask) if not _has_enclosed_hole(cand)]


def _has_interior_background_but_no_hole(mask: np.ndarray) -> bool:
    h, w = mask.shape
    if h < 3 or w < 3:
        return False
    return bool((~mask[1:-1, 1:-1]).any() and not _has_enclosed_hole(mask))


def _collision_candidates(mask_a: np.ndarray, mask_b: np.ndarray) -> list[tuple[int, int, int]]:
    a_coords = {tuple(x) for x in np.argwhere(mask_a)}
    b_coords = np.argwhere(mask_b)

    a_mirror = np.zeros_like(mask_a, dtype=bool)
    am = _mirror_coords(np.argwhere(mask_a), (0, 0, mask_a.shape[0] - 1, mask_a.shape[1] - 1))
    a_mirror[am[:, 0], am[:, 1]] = True
    am_coords = {tuple(x) for x in np.argwhere(a_mirror)}

    b_mirror = np.zeros_like(mask_b, dtype=bool)
    bm = _mirror_coords(np.argwhere(mask_b), (0, 0, mask_b.shape[0] - 1, mask_b.shape[1] - 1))
    b_mirror[bm[:, 0], bm[:, 1]] = True
    bm_coords = np.argwhere(b_mirror)

    candidates = []
    max_r = mask_a.shape[0] + mask_b.shape[0]
    max_c = mask_a.shape[1] + mask_b.shape[1]
    for dr in range(-max_r, max_r + 1):
        for dc in range(-max_c, max_c + 1):
            shifted_b = {(int(r + dr), int(c + dc)) for r, c in b_coords}
            if a_coords & shifted_b:
                continue
            collision = am_coords & {(int(r + dr), int(c + dc)) for r, c in bm_coords}
            if not collision:
                continue
            ar0, ac0, ar1, ac1 = 0, 0, mask_a.shape[0] - 1, mask_a.shape[1] - 1
            br0, bc0 = dr, dc
            br1, bc1 = dr + mask_b.shape[0] - 1, dc + mask_b.shape[1] - 1
            if ar1 < br0 or br1 < ar0 or ac1 < bc0 or bc1 < ac0:
                continue
            candidates.append((dr, dc, len(collision)))
    random.shuffle(candidates)
    candidates.sort(key=lambda item: (abs(item[2] - 2), abs(item[0]) + abs(item[1])))
    return candidates


def _can_place(mask: np.ndarray, top: int, left: int, shape: tuple[int, int], occupied: np.ndarray, gap: int) -> bool:
    h, w = mask.shape
    gh, gw = shape
    if top < 0 or left < 0 or top + h > gh or left + w > gw:
        return False
    rows, cols = np.nonzero(mask)
    for rr, cc in zip(rows + top, cols + left):
        if occupied[rr, cc]:
            return False
        if gap:
            for nr in range(rr - gap, rr + gap + 1):
                for nc in range(cc - gap, cc + gap + 1):
                    if 0 <= nr < gh and 0 <= nc < gw and occupied[nr, nc]:
                        return False
    return True


def _mark(mask: np.ndarray, top: int, left: int, occupied: np.ndarray) -> None:
    rows, cols = np.nonzero(mask)
    occupied[rows + top, cols + left] = True


def _choose_counts() -> tuple[int, int]:
    return random.choice(((2, 2), (2, 3), (3, 2), (3, 3)))


def _choose_hole_specs(n_holes: int) -> tuple[list[dict], tuple[int, int], tuple[int, int, int]] | None:
    template_indices = random.sample(range(len(HOLE_TEMPLATE_MASKS)), n_holes)
    chosen = []
    for index in template_indices:
        variants = [cand for cand in _unique_variants(HOLE_TEMPLATE_MASKS[index]) if _has_enclosed_hole(cand) and _mask_changes_under_vertical_mirror(cand)]
        chosen.append({"mask": random.choice(variants).copy(), "kind": "hole"})

    pairs = [(i, j) for i in range(n_holes) for j in range(i + 1, n_holes)]
    random.shuffle(pairs)
    for ia, ib in pairs:
        candidates = _collision_candidates(chosen[ia]["mask"], chosen[ib]["mask"])
        if candidates:
            return chosen, (ia, ib), random.choice(candidates)
    return None


def _place_collision_pair(specs: list[dict], pair: tuple[int, int], offset: tuple[int, int, int], shape: tuple[int, int]) -> bool:
    ia, ib = pair
    dr, dc, _count = offset
    mask_a = specs[ia]["mask"]
    mask_b = specs[ib]["mask"]
    gh, gw = shape
    low_r = max(0, -dr)
    high_r = min(gh - mask_a.shape[0], gh - dr - mask_b.shape[0])
    low_c = max(0, -dc)
    high_c = min(gw - mask_a.shape[1], gw - dc - mask_b.shape[1])
    if low_r > high_r or low_c > high_c:
        return False

    top_a = random.randint(low_r, high_r)
    left_a = random.randint(low_c, high_c)
    specs[ia]["top"] = top_a
    specs[ia]["left"] = left_a
    specs[ib]["top"] = top_a + dr
    specs[ib]["left"] = left_a + dc
    return True


def _place_remaining(specs: list[dict], shape: tuple[int, int], occupied: np.ndarray) -> bool:
    order = [idx for idx, spec in enumerate(specs) if "top" not in spec]
    random.shuffle(order)
    order.sort(key=lambda idx: specs[idx]["mask"].size, reverse=True)
    gh, gw = shape
    for idx in order:
        mask = specs[idx]["mask"]
        positions = [(r, c) for r in range(gh - mask.shape[0] + 1) for c in range(gw - mask.shape[1] + 1)]
        random.shuffle(positions)
        placed = False
        for top, left in positions:
            if _can_place(mask, top, left, shape, occupied, gap=1):
                specs[idx]["top"] = top
                specs[idx]["left"] = left
                _mark(mask, top, left, occupied)
                placed = True
                break
        if not placed:
            return False
    return True


def _spec_coords(spec: dict) -> np.ndarray:
    rows, cols = np.nonzero(spec["mask"])
    return np.column_stack([rows + spec["top"], cols + spec["left"]]).astype(int)


def _spec_bbox(spec: dict) -> tuple[int, int, int, int]:
    coords = _spec_coords(spec)
    return _bbox_coords(coords)


def _spec_has_collision(spec_a: dict, spec_b: dict) -> bool:
    bbox_a = _spec_bbox(spec_a)
    bbox_b = _spec_bbox(spec_b)
    ma = _mirror_coords(_spec_coords(spec_a), bbox_a)
    mb = _mirror_coords(_spec_coords(spec_b), bbox_b)
    return bool({tuple(x) for x in ma} & {tuple(x) for x in mb})


def _semantic_checks(specs: list[dict], grid: np.ndarray) -> bool:
    detected = _extract_objects(grid, BG)
    if len(detected) != len(specs):
        return False
    if not (4 <= len(specs) <= 6):
        return False
    if len({int(spec["color"]) for spec in specs}) != len(specs):
        return False
    if any(not (5 <= int(spec["mask"].sum()) <= 12) for spec in specs):
        return False

    holes = [spec for spec in specs if spec["kind"] == "hole"]
    opens = [spec for spec in specs if spec["kind"] == "open"]
    if not (2 <= len(holes) <= 3 and 2 <= len(opens) <= 3):
        return False

    hole_areas = [int(spec["mask"].shape[0] * spec["mask"].shape[1]) for spec in holes]
    if len(set(hole_areas)) != len(hole_areas):
        return False
    if not all(_mask_changes_under_vertical_mirror(spec["mask"]) for spec in holes):
        return False
    if not any(_has_interior_background_but_no_hole(spec["mask"]) for spec in opens):
        return False

    collision_count = 0
    for i in range(len(holes)):
        for j in range(i + 1, len(holes)):
            collision_count += int(_spec_has_collision(holes[i], holes[j]))
    if collision_count == 0:
        return False

    analyzed_holes = [obj for obj in detected if obj["has_hole"]]
    analyzed_opens = [obj for obj in detected if not obj["has_hole"]]
    if len(analyzed_holes) != len(holes) or len(analyzed_opens) != len(opens):
        return False
    if len({obj["bbox_area"] for obj in analyzed_holes}) != len(analyzed_holes):
        return False

    out = main(grid)
    if np.array_equal(out, grid):
        return False
    if np.count_nonzero(out) >= np.count_nonzero(grid):
        return False

    contested = {}
    for obj in analyzed_holes:
        mirrored = _mirror_coords(obj["coords"], obj["bbox"])
        for cell in map(tuple, mirrored):
            contested.setdefault(cell, []).append(obj["bbox_area"])
    if not any(len(areas) >= 2 and min(areas) != max(areas) for areas in contested.values()):
        return False

    return True


def main(grid: np.ndarray) -> np.ndarray:
    """Classify foreground objects by enclosed holes, mirror the hole-objects vertically, and erase the rest.

    Input structure:
    - The background is black and the foreground consists of 4-connected monochrome objects.
    - Some objects enclose at least one black 4-connected hole inside their own tight bounding box.
    - Other objects are open shapes whose bbox contains no enclosed hole.

    Rule:
    - Detect each monochrome 4-connected object and inspect black components inside its tight bbox.
    - If an internal black component does not touch the bbox edge, the object is a hole-object; mirror it vertically inside its own bbox.
    - If no enclosed hole exists, erase that object.
    - When mirrored hole-objects land on the same cell, the object with the smaller bbox area overwrites the larger one.

    Key insight:
    - The enclosed hole is only a topology signal: it decides whether an object survives as a bbox-local vertical mirror or disappears entirely.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = _get_bg(g)
    out = np.full_like(g, bg)
    objects = _extract_objects(g, bg)
    hole_objects = [obj for obj in objects if obj["has_hole"]]

    for obj in sorted(
        hole_objects,
        key=lambda item: (-item["bbox_area"], item["bbox"][0], item["bbox"][1], item["bbox"][2], item["bbox"][3]),
    ):
        mirrored = _mirror_coords(obj["coords"], obj["bbox"])
        out[mirrored[:, 0], mirrored[:, 1]] = int(obj["color"])

    return out


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        height = random.randint(*GRID_SIZE_RANGE)
        width = random.randint(*GRID_SIZE_RANGE)
        n_holes, n_open = _choose_counts()

        hole_pick = _choose_hole_specs(n_holes)
        if hole_pick is None:
            continue
        hole_specs, pair, offset = hole_pick

        open_specs = []
        counterexample_pool = [mask for base in OPEN_TEMPLATE_MASKS for mask in _open_variants(base) if _has_interior_background_but_no_hole(mask)]
        general_pool = [mask for base in OPEN_TEMPLATE_MASKS for mask in _open_variants(base)]
        open_specs.append({"mask": random.choice(counterexample_pool).copy(), "kind": "open"})
        for _open_idx in range(n_open - 1):
            open_specs.append({"mask": random.choice(general_pool).copy(), "kind": "open"})

        specs = hole_specs + open_specs
        colors = random.sample(COLOR_POOL, len(specs))
        for spec, color in zip(specs, colors):
            spec["color"] = int(color)

        if not _place_collision_pair(specs, pair, offset, (height, width)):
            continue

        occupied = np.zeros((height, width), dtype=bool)
        for spec in specs:
            if "top" in spec:
                _mark(spec["mask"], spec["top"], spec["left"], occupied)

        if not _place_remaining(specs, (height, width), occupied):
            continue

        grid = _render_specs((height, width), specs)
        if _semantic_checks(specs, grid):
            return grid

    raise RuntimeError("failed to generate a valid input for puzzle 3")
