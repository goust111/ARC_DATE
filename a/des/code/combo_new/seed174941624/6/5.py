from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
COLOR_POOL = tuple(range(1, 10))
MAX_RETRIES = 300
NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
NEIGHBORS8 = tuple(
    (dr, dc)
    for dr in (-1, 0, 1)
    for dc in (-1, 0, 1)
    if (dr, dc) != (0, 0)
)


def _is_connected(mask: np.ndarray, connectivity: int = 8) -> bool:
    mask = np.asarray(mask, dtype=bool)
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False

    neighbors = NEIGHBORS8 if connectivity == 8 else NEIGHBORS4
    start = tuple(map(int, coords[0]))
    seen = {start}
    q = deque([start])
    h, w = mask.shape

    while q:
        r, c = q.popleft()
        for dr, dc in neighbors:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                q.append((nr, nc))

    return len(seen) == int(mask.sum())


def _touches_all_sides(mask: np.ndarray) -> bool:
    mask = np.asarray(mask, dtype=bool)
    return bool(mask[0].any() and mask[-1].any() and mask[:, 0].any() and mask[:, -1].any())


def _orthogonal_edge_count(mask: np.ndarray) -> int:
    mask = np.asarray(mask, dtype=bool)
    return int(np.sum(mask[1:, :] & mask[:-1, :]) + np.sum(mask[:, 1:] & mask[:, :-1]))


def _extract_components(grid: np.ndarray, bg: int = BG, connectivity: int = 8) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    neighbors = NEIGHBORS8 if connectivity == 8 else NEIGHBORS4
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []

    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(g[r, c]) == bg:
                continue

            color = int(g[r, c])
            q = deque([(int(r), int(c))])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []

            while q:
                cr, cc = q.popleft()
                coords.append((cr, cc))
                for dr, dc in neighbors:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(g[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))

            rs = [rr for rr, _ in coords]
            cs = [cc for _, cc in coords]
            r0, r1 = min(rs), max(rs)
            c0, c1 = min(cs), max(cs)
            mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
            for rr, cc in coords:
                mask[rr - r0, cc - c0] = True

            comps.append(
                {
                    "color": color,
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    "anchor": (int(r0), int(c0)),
                    "mask": mask,
                }
            )

    return comps


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pair scattered monochrome objects by exact shape complement and assemble each pair into a full rectangle.

    Input structure:
    - The background is black.
    - The grid contains an even number of non-black monochrome connected objects, each with a distinct color.
    - Every object's bounding box has the same height and width, and every object is a subset of that shared rectangle.
    - For each object there is exactly one partner whose occupied cells are the cellwise complement inside that rectangle.

    Rule:
    - Extract every object and normalize it to its own bounding box.
    - Pair two objects when one normalized mask is exactly the inverse of the other, with translation allowed but no rotation or reflection.
    - For each pair, rebuild the full shared rectangle and keep each cell in the color of the source object that contributes that cell.
    - Order the pairs by the row-major order of their source objects and place the rebuilt rectangles left-to-right with a one-cell black gap.

    Key insight:
    - The hidden pairing signal is negative space: an object's mate is not nearby or same-sized by chance, but precisely fills the cells that object leaves empty inside the common bounding box.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    comps = _extract_components(g, bg=BG, connectivity=8)
    assert comps and len(comps) % 2 == 0, "expected an even number of foreground objects"

    dims = {tuple(map(int, comp["mask"].shape)) for comp in comps}
    assert len(dims) == 1, "all objects must share one bounding-box shape"
    rect_h, rect_w = next(iter(dims))

    mask_to_idx: dict[bytes, int] = {}
    for idx, comp in enumerate(comps):
        key = comp["mask"].tobytes()
        assert key not in mask_to_idx, "duplicate object silhouettes make pairing ambiguous"
        mask_to_idx[key] = idx

    partner_of: dict[int, int] = {}
    for idx, comp in enumerate(comps):
        partner_key = (~comp["mask"]).tobytes()
        assert partner_key in mask_to_idx, "every object must have an exact complement partner"
        partner = mask_to_idx[partner_key]
        assert partner != idx, "an object cannot complement itself"
        partner_of[idx] = partner

    used: set[int] = set()
    ordered_pairs: list[tuple[tuple[tuple[int, int], tuple[int, int]], int, int]] = []
    for idx in range(len(comps)):
        if idx in used:
            continue
        partner = partner_of[idx]
        assert partner_of.get(partner) == idx, "complement pairing must be symmetric"
        used.add(idx)
        used.add(partner)
        pair_key = tuple(sorted((comps[idx]["anchor"], comps[partner]["anchor"])))
        ordered_pairs.append((pair_key, idx, partner))

    assert len(used) == len(comps), "all objects must be paired exactly once"
    ordered_pairs.sort(key=lambda item: item[0])

    out_w = len(ordered_pairs) * rect_w + max(0, len(ordered_pairs) - 1)
    out = np.zeros((rect_h, out_w), dtype=int)

    for pair_idx, (_, left_idx, right_idx) in enumerate(ordered_pairs):
        c0 = pair_idx * (rect_w + 1)
        patch = out[:, c0 : c0 + rect_w]
        patch[comps[left_idx]["mask"]] = comps[left_idx]["color"]
        patch[comps[right_idx]["mask"]] = comps[right_idx]["color"]
        out[:, c0 : c0 + rect_w] = patch

    return out


def _random_valid_mask(rect_h: int, rect_w: int) -> np.ndarray:
    area = rect_h * rect_w
    lower = max(4, area // 4)
    upper = area - lower

    for _ in range(4000):
        fill_prob = random.uniform(0.35, 0.65)
        mask = np.random.random((rect_h, rect_w)) < fill_prob
        filled = int(mask.sum())
        if filled < lower or filled > upper:
            continue

        comp = ~mask
        if not _touches_all_sides(mask) or not _touches_all_sides(comp):
            continue
        if not _is_connected(mask, connectivity=8) or not _is_connected(comp, connectivity=8):
            continue
        if _orthogonal_edge_count(mask) < 2 or _orthogonal_edge_count(comp) < 2:
            continue
        return mask

    raise RuntimeError("failed to sample a valid complementary mask")


def _layout_options(obj_count: int, rect_h: int, rect_w: int) -> list[tuple[int, int, int, int, int, int]]:
    slot_h = rect_h + 2
    slot_w = rect_w + 2
    layouts: list[tuple[int, int, int, int, int, int]] = []

    for rows in range(1, 5):
        for cols in range(1, 5):
            grid_h = rows * slot_h + 1
            grid_w = cols * slot_w + 1
            if rows * cols >= obj_count and grid_h <= 30 and grid_w <= 30:
                layouts.append((rows, cols, slot_h, slot_w, grid_h, grid_w))

    return layouts


def _expected_output(
    rect_h: int,
    rect_w: int,
    objects: list[dict],
    pair_ids: list[tuple[int, int]],
) -> np.ndarray:
    ordered = sorted(
        pair_ids,
        key=lambda pair: tuple(sorted((objects[pair[0]]["anchor"], objects[pair[1]]["anchor"]))),
    )
    out_w = len(ordered) * rect_w + max(0, len(ordered) - 1)
    out = np.zeros((rect_h, out_w), dtype=int)

    for pair_idx, (left_idx, right_idx) in enumerate(ordered):
        c0 = pair_idx * (rect_w + 1)
        patch = out[:, c0 : c0 + rect_w]
        patch[objects[left_idx]["mask"]] = objects[left_idx]["color"]
        patch[objects[right_idx]["mask"]] = objects[right_idx]["color"]
        out[:, c0 : c0 + rect_w] = patch

    return out


def _gen_attempt() -> np.ndarray | None:
    rect_h = random.randint(4, 6)
    rect_w = random.randint(4, 6)
    num_pairs = random.randint(2, 4)
    obj_count = 2 * num_pairs
    layouts = _layout_options(obj_count, rect_h, rect_w)
    if not layouts:
        return None

    rows, cols, slot_h, slot_w, grid_h, grid_w = random.choice(layouts)
    colors = random.sample(COLOR_POOL, obj_count)
    used_masks: set[bytes] = set()
    objects: list[dict] = []
    pair_ids: list[tuple[int, int]] = []

    for _ in range(num_pairs):
        for _mask_try in range(500):
            mask = _random_valid_mask(rect_h, rect_w)
            key = mask.tobytes()
            comp_key = (~mask).tobytes()
            if key in used_masks or comp_key in used_masks:
                continue
            used_masks.add(key)
            used_masks.add(comp_key)
            break
        else:
            return None

        first_idx = len(objects)
        objects.append({"mask": mask, "color": colors[first_idx]})
        second_idx = len(objects)
        objects.append({"mask": ~mask, "color": colors[second_idx]})
        pair_ids.append((first_idx, second_idx))

    slot_choices = [(sr, sc) for sr in range(rows) for sc in range(cols)]
    chosen_slots = random.sample(slot_choices, obj_count)
    random.shuffle(chosen_slots)
    grid = np.zeros((grid_h, grid_w), dtype=int)

    for obj, (slot_r, slot_c) in zip(objects, chosen_slots, strict=False):
        r0 = 1 + slot_r * slot_h + random.randint(0, 1)
        c0 = 1 + slot_c * slot_w + random.randint(0, 1)
        patch = grid[r0 : r0 + rect_h, c0 : c0 + rect_w]
        patch[obj["mask"]] = obj["color"]
        grid[r0 : r0 + rect_h, c0 : c0 + rect_w] = patch
        obj["anchor"] = (int(r0), int(c0))

    if len(_extract_components(grid, bg=BG, connectivity=8)) != obj_count:
        return None

    expected = _expected_output(rect_h, rect_w, objects, pair_ids)
    try:
        actual = main(grid)
    except AssertionError:
        return None

    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
