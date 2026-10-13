from __future__ import annotations

import random
from collections import Counter

import numpy as np


ORTHO_DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
ALL_COLORS = list(range(10))
GEN_RETRIES = 400
PLACE_RETRIES = 300
MASK_RETRIES = 250


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(values[np.argmax(counts)])


def _is_connected(mask: np.ndarray) -> bool:
    mask = np.asarray(mask, dtype=bool)
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    seen = {tuple(map(int, coords[0]))}
    stack = [tuple(map(int, coords[0]))]
    h, w = mask.shape
    while stack:
        r, c = stack.pop()
        for dr, dc in ORTHO_DIRS:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                stack.append((nr, nc))
    return len(seen) == int(mask.sum())


def _touches_top_left(mask: np.ndarray) -> bool:
    mask = np.asarray(mask, dtype=bool)
    return bool(mask[0, :].any() and mask[:, 0].any())


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    assert coords.size > 0
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return np.asarray(mask[r0 : r1 + 1, c0 : c1 + 1], dtype=bool)


def _pad_mask(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    h, w = map(int, shape)
    out = np.zeros((h, w), dtype=bool)
    mh, mw = mask.shape
    out[:mh, :mw] = mask
    return out


def _find_objects(grid: np.ndarray, bg: int) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    objects: list[dict] = []

    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(g[r, c]) == int(bg):
                continue
            color = int(g[r, c])
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((int(cr), int(cc)))
                for dr, dc in ORTHO_DIRS:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(g[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))

            rows = [p[0] for p in coords]
            cols = [p[1] for p in coords]
            r0, r1 = min(rows), max(rows)
            c0, c1 = min(cols), max(cols)
            mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
            for rr, cc in coords:
                mask[rr - r0, cc - c0] = True
            objects.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    "mask": mask,
                }
            )

    objects.sort(key=lambda obj: (obj["bbox"][0], obj["bbox"][1], obj["color"]))
    return objects


def _mask_key(mask: np.ndarray) -> tuple[int, int, bytes]:
    m = np.asarray(mask, dtype=bool)
    return int(m.shape[0]), int(m.shape[1]), m.tobytes()


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pair foreground sprites by complementary normalized shapes and replace each pair
    by their merged rectangle at the lexicographically earlier sprite location.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    objects = _find_objects(g, bg)
    assert len(objects) % 2 == 0 and objects, "expected an even number of foreground objects"

    out = g.copy()
    paired: set[int] = set()

    for i, obj in enumerate(objects):
        if i in paired:
            continue
        partners: list[int] = []
        for j, other in enumerate(objects):
            if j == i or j in paired:
                continue
            common_h = max(int(obj["mask"].shape[0]), int(other["mask"].shape[0]))
            common_w = max(int(obj["mask"].shape[1]), int(other["mask"].shape[1]))
            padded_obj = _pad_mask(obj["mask"], (common_h, common_w))
            padded_other = _pad_mask(other["mask"], (common_h, common_w))
            if np.array_equal(padded_obj, ~padded_other):
                partners.append(j)
        assert len(partners) == 1, "each object must have exactly one complementary partner"
        j = int(partners[0])
        other = objects[j]
        paired.add(i)
        paired.add(j)

        first, second = (obj, other) if obj["bbox"][:2] <= other["bbox"][:2] else (other, obj)

        for rr, cc in first["coords"]:
            out[int(rr), int(cc)] = int(bg)
        for rr, cc in second["coords"]:
            out[int(rr), int(cc)] = int(bg)

        common_h = max(int(first["mask"].shape[0]), int(second["mask"].shape[0]))
        common_w = max(int(first["mask"].shape[1]), int(second["mask"].shape[1]))
        padded_first = _pad_mask(first["mask"], (common_h, common_w))
        patch = np.where(padded_first, int(first["color"]), int(second["color"]))
        r0, c0 = first["bbox"][0], first["bbox"][1]
        out[r0 : r0 + common_h, c0 : c0 + common_w] = patch

    assert len(paired) == len(objects)
    return out


def _grow_connected_mask(h: int, w: int, area: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    seed = (random.randrange(h), random.randrange(w))
    mask[seed] = True
    frontier = set()
    for dr, dc in ORTHO_DIRS:
        nr, nc = seed[0] + dr, seed[1] + dc
        if 0 <= nr < h and 0 <= nc < w:
            frontier.add((nr, nc))

    while int(mask.sum()) < area:
        if not frontier:
            break
        nr, nc = random.choice(tuple(frontier))
        frontier.remove((nr, nc))
        if mask[nr, nc]:
            continue
        mask[nr, nc] = True
        for dr, dc in ORTHO_DIRS:
            rr, cc = nr + dr, nc + dc
            if 0 <= rr < h and 0 <= cc < w and not mask[rr, cc]:
                frontier.add((rr, cc))
    return mask


def _sample_pair_mask(h: int, w: int) -> np.ndarray | None:
    area_lo = 2 if h * w >= 4 else 1
    area_hi = h * w - area_lo
    if area_lo > area_hi:
        return None
    for _ in range(MASK_RETRIES):
        area = random.randint(area_lo, area_hi)
        mask = _grow_connected_mask(h, w, area)
        inv = ~mask
        if mask.sum() != area:
            continue
        if not _is_connected(mask):
            continue
        if not _is_connected(inv):
            continue
        if not _touches_top_left(mask):
            continue
        if not _touches_top_left(inv):
            continue
        return mask
    return None


def _can_place(occupied: np.ndarray, r0: int, c0: int, h: int, w: int) -> bool:
    H, W = occupied.shape
    r1 = r0 + h - 1
    c1 = c0 + w - 1
    pr0 = max(0, r0 - 1)
    pc0 = max(0, c0 - 1)
    pr1 = min(H - 1, r1 + 1)
    pc1 = min(W - 1, c1 + 1)
    return not bool(occupied[pr0 : pr1 + 1, pc0 : pc1 + 1].any())


def _place_bbox(occupied: np.ndarray, h: int, w: int) -> tuple[int, int] | None:
    H, W = occupied.shape
    if h > H or w > W:
        return None
    for _ in range(PLACE_RETRIES):
        r0 = random.randint(0, H - h)
        c0 = random.randint(0, W - w)
        if _can_place(occupied, r0, c0, h, w):
            occupied[r0 : r0 + h, c0 : c0 + w] = True
            return int(r0), int(c0)
    return None


def _render_object(grid: np.ndarray, top: int, left: int, mask: np.ndarray, color: int) -> None:
    h, w = mask.shape
    view = grid[top : top + h, left : left + w]
    view[mask] = int(color)


def _generate_attempt() -> np.ndarray | None:
    pair_count = random.randint(2, 4)
    bg = random.choice(ALL_COLORS)
    palette = [c for c in ALL_COLORS if c != bg]
    if 2 * pair_count > len(palette):
        return None
    colors = random.sample(palette, 2 * pair_count)

    pair_specs: list[dict] = []
    used_keys: set[tuple[int, int, bytes]] = set()
    for pair_idx in range(pair_count):
        for _ in range(MASK_RETRIES):
            h = random.randint(2, 6)
            w = random.randint(2, 6)
            mask = _sample_pair_mask(h, w)
            if mask is None:
                continue
            key = _mask_key(mask)
            comp_key = _mask_key(~mask)
            if key in used_keys or comp_key in used_keys:
                continue
            used_keys.add(key)
            used_keys.add(comp_key)
            c1 = colors[2 * pair_idx]
            c2 = colors[2 * pair_idx + 1]
            if random.random() < 0.5:
                full_first, full_second = mask, ~mask
            else:
                full_first, full_second = ~mask, mask
            first_mask = _crop_mask(full_first)
            second_mask = _crop_mask(full_second)
            pair_specs.append(
                {
                    "full_h": h,
                    "full_w": w,
                    "first_mask": first_mask.copy(),
                    "second_mask": second_mask.copy(),
                    "first_color": int(c1),
                    "second_color": int(c2),
                }
            )
            break
        else:
            return None

    total_h = sum(spec["full_h"] + 2 for spec in pair_specs)
    total_w = sum(spec["full_w"] + 2 for spec in pair_specs)
    min_h = max(18, total_h // 2)
    min_w = max(18, total_w // 2)
    max_h = max(min_h, min(30, total_h + 6))
    max_w = max(min_w, min(30, total_w + 6))
    H = random.randint(min_h, max_h)
    W = random.randint(min_w, max_w)
    grid = np.full((H, W), int(bg), dtype=int)
    occupied = np.zeros((H, W), dtype=bool)

    placements: list[dict] = []
    order = list(range(pair_count))
    random.shuffle(order)
    order.sort(key=lambda idx: -(pair_specs[idx]["full_h"] * pair_specs[idx]["full_w"]))
    for idx in order:
        spec = pair_specs[idx]
        pos1 = _place_bbox(occupied, spec["full_h"], spec["full_w"])
        if pos1 is None:
            return None
        pos2 = _place_bbox(occupied, spec["full_h"], spec["full_w"])
        if pos2 is None:
            return None
        placements.append({"spec": spec, "pos1": pos1, "pos2": pos2})

    for item in placements:
        spec = item["spec"]
        r1, c1 = item["pos1"]
        r2, c2 = item["pos2"]
        _render_object(grid, r1, c1, spec["first_mask"], spec["first_color"])
        _render_object(grid, r2, c2, spec["second_mask"], spec["second_color"])

    try:
        out = main(grid)
    except AssertionError:
        return None

    if np.array_equal(out, grid):
        return None

    objs = _find_objects(grid, bg)
    if len(objs) != 2 * pair_count:
        return None
    if len({obj["color"] for obj in objs}) != len(objs):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
