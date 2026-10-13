from __future__ import annotations

import math
import random
from collections import Counter

import numpy as np


NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
GEN_RETRIES = 400
OBJECT_PAD = 1
OBJECT_MAX_DIM = 7


def get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components_of_color(grid: np.ndarray, color: int) -> list[np.ndarray]:
    mask = grid == color
    visited = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    height, width = mask.shape

    for start_r, start_c in np.argwhere(mask):
        r0 = int(start_r)
        c0 = int(start_c)
        if visited[r0, c0]:
            continue
        stack = [(r0, c0)]
        visited[r0, c0] = True
        coords: list[tuple[int, int]] = []
        while stack:
            r, c = stack.pop()
            coords.append((r, c))
            for dr, dc in NEIGHBORS4:
                nr = r + dr
                nc = c + dc
                if 0 <= nr < height and 0 <= nc < width and mask[nr, nc] and not visited[nr, nc]:
                    visited[nr, nc] = True
                    stack.append((nr, nc))
        comps.append(np.asarray(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _mask_from_coords(coords: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    r0, c0, r1, c1 = _bbox(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return mask, (r0, c0, r1, c1)


def _extract_objects(grid: np.ndarray, bg: int) -> list[dict]:
    objects: list[dict] = []
    for color in map(int, np.unique(grid)):
        if color == bg:
            continue
        for coords in _components_of_color(grid, color):
            mask, bbox = _mask_from_coords(coords)
            objects.append({"color": color, "coords": coords, "bbox": bbox, "mask": mask})
    return objects


def _contains_template(container: np.ndarray, template: np.ndarray) -> bool:
    ch, cw = container.shape
    th, tw = template.shape
    if ch < th or cw < tw:
        return False
    template_pts = np.argwhere(template)
    if len(template_pts) == 0:
        return True
    tr = template_pts[:, 0]
    tc = template_pts[:, 1]
    for dr in range(ch - th + 1):
        for dc in range(cw - tw + 1):
            if np.all(container[dr + tr, dc + tc]):
                return True
    return False


def main(grid: np.ndarray) -> np.ndarray:
    """
    Keep only the objects that can contain the uniquely colored stamp template, and recolor them to the stamp color.

    Input structure:
    - The grid has a dominant background color and several scattered monochrome 4-connected objects.
    - Exactly one object uses a color that appears in no other object; that uniquely colored object is the stamp.
    - The stamp object's tight bounding box is a small 3x3 to 5x5 template, while the other objects may have larger or different shapes.

    Rule:
    - Find the unique-color stamp object and extract its binary shape inside its tight bounding box.
    - For every other object, test whether the stamp shape can be translated somewhere inside that object's shape without rotation or reflection.
    - If an object contains the stamp template, recolor the whole object to the stamp color and keep its shape unchanged.
    - If an object does not contain the stamp template, erase it to the background color.
    - Leave the stamp object itself unchanged.

    Key insight:
    - The special color only identifies the template source; the real task is a shape-within-shape translation test on the other objects.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    objects = _extract_objects(g, bg)
    assert objects, "grid must contain at least one foreground object"

    color_freq = Counter(int(obj["color"]) for obj in objects)
    unique_colors = [color for color, count in color_freq.items() if count == 1]
    assert len(unique_colors) == 1, "there must be exactly one unique-color stamp"
    stamp_color = int(unique_colors[0])
    stamp_obj = next(obj for obj in objects if int(obj["color"]) == stamp_color)
    stamp_mask = np.asarray(stamp_obj["mask"], dtype=bool)

    out = g.copy()
    for obj in objects:
        if obj is stamp_obj:
            continue
        rr, cc = obj["coords"][:, 0], obj["coords"][:, 1]
        if _contains_template(np.asarray(obj["mask"], dtype=bool), stamp_mask):
            out[rr, cc] = stamp_color
        else:
            out[rr, cc] = bg
    return out


def _mask_key(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    return (tuple(int(x) for x in mask.shape), mask.astype(bool, copy=False).tobytes())


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    pts = np.argwhere(mask)
    assert len(pts) > 0
    r0, c0, r1, c1 = _bbox(pts)
    return np.asarray(mask[r0 : r1 + 1, c0 : c1 + 1], dtype=bool)


def _touches_all_sides(mask: np.ndarray) -> bool:
    return bool(mask[0].any() and mask[-1].any() and mask[:, 0].any() and mask[:, -1].any())


def _frontier_from_mask(mask: np.ndarray) -> set[tuple[int, int]]:
    height, width = mask.shape
    frontier: set[tuple[int, int]] = set()
    for r, c in np.argwhere(mask):
        rr = int(r)
        cc = int(c)
        for dr, dc in NEIGHBORS4:
            nr = rr + dr
            nc = cc + dc
            if 0 <= nr < height and 0 <= nc < width and not mask[nr, nc]:
                frontier.add((nr, nc))
    return frontier


def _grow_connected_mask(
    height: int,
    width: int,
    target_area: int,
    seed: np.ndarray | None = None,
    require_full_bbox: bool = False,
    max_tries: int = 120,
) -> np.ndarray | None:
    for _ in range(max_tries):
        if seed is None:
            mask = np.zeros((height, width), dtype=bool)
            mask[random.randrange(height), random.randrange(width)] = True
        else:
            mask = np.array(seed, dtype=bool, copy=True)
        if int(mask.sum()) > target_area:
            continue
        frontier = _frontier_from_mask(mask)
        while int(mask.sum()) < target_area and frontier:
            r, c = random.choice(tuple(frontier))
            frontier.remove((r, c))
            mask[r, c] = True
            for dr, dc in NEIGHBORS4:
                nr = r + dr
                nc = c + dc
                if 0 <= nr < height and 0 <= nc < width and not mask[nr, nc]:
                    frontier.add((nr, nc))
        if int(mask.sum()) != target_area:
            continue
        if require_full_bbox and not _touches_all_sides(mask):
            continue
        return mask
    return None


def _orientation_variants(mask: np.ndarray) -> list[np.ndarray]:
    variants: list[np.ndarray] = []
    seen: set[tuple[tuple[int, int], bytes]] = set()
    base = np.asarray(mask, dtype=bool)
    for flipped in (False, True):
        cur = np.fliplr(base) if flipped else base
        for turns in range(4):
            variant = np.rot90(cur, turns).copy()
            key = _mask_key(variant)
            if key not in seen:
                seen.add(key)
                variants.append(variant)
    return variants


def _generate_stamp() -> np.ndarray | None:
    for _ in range(200):
        height = random.randint(3, 5)
        width = random.randint(3, 5)
        total = height * width
        min_area = max(4, int(math.ceil(total * 0.45)))
        max_area = total - 1
        if min_area > max_area:
            continue
        area = random.randint(min_area, max_area)
        mask = _grow_connected_mask(height, width, area, require_full_bbox=True, max_tries=100)
        if mask is None or bool(mask.all()):
            continue
        if len(_orientation_variants(mask)) < 2:
            continue
        return mask
    return None


def _make_container(stamp: np.ndarray) -> np.ndarray | None:
    stamp = np.asarray(stamp, dtype=bool)
    sh, sw = stamp.shape
    stamp_area = int(stamp.sum())
    for _ in range(250):
        if random.random() < 0.45 and stamp_area < sh * sw:
            height, width = sh, sw
        else:
            height = random.randint(sh, min(OBJECT_MAX_DIM, sh + 2))
            width = random.randint(sw, min(OBJECT_MAX_DIM, sw + 2))
        seed = np.zeros((height, width), dtype=bool)
        top = random.randint(0, height - sh)
        left = random.randint(0, width - sw)
        seed[top : top + sh, left : left + sw] |= stamp
        extra_cap = height * width - stamp_area
        extra = 0 if extra_cap == 0 else random.randint(0, min(extra_cap, stamp_area // 2 + 3))
        area = stamp_area + extra
        mask = _grow_connected_mask(
            height,
            width,
            area,
            seed=seed,
            require_full_bbox=(random.random() < 0.35),
            max_tries=90,
        )
        if mask is None:
            continue
        mask = _crop_mask(mask)
        if _contains_template(mask, stamp):
            return mask
    return None


def _make_noncontainer(stamp: np.ndarray, force_large: bool = False) -> np.ndarray | None:
    stamp = np.asarray(stamp, dtype=bool)
    sh, sw = stamp.shape
    stamp_area = int(stamp.sum())
    for _ in range(300):
        if force_large or random.random() < 0.6:
            height = random.randint(sh, min(OBJECT_MAX_DIM, sh + 2))
            width = random.randint(sw, min(OBJECT_MAX_DIM, sw + 2))
        else:
            height = random.randint(2, min(OBJECT_MAX_DIM, max(2, sh)))
            width = random.randint(2, min(OBJECT_MAX_DIM, max(2, sw)))
        max_cells = height * width
        lo = max(3, min(max_cells, max(3, stamp_area - 3)))
        hi = min(max_cells, stamp_area + 4)
        if lo > hi:
            continue
        area = random.randint(lo, hi)
        mask = _grow_connected_mask(
            height,
            width,
            area,
            require_full_bbox=(random.random() < 0.35),
            max_tries=100,
        )
        if mask is None:
            continue
        mask = _crop_mask(mask)
        if force_large and (mask.shape[0] < sh or mask.shape[1] < sw):
            continue
        if not _contains_template(mask, stamp):
            return mask
    return None


def _other_colors(total: int, bg: int, stamp_color: int) -> list[int]:
    palette = [color for color in range(10) if color not in (bg, stamp_color)]
    color_count = random.randint(2, min(3, total // 2))
    chosen = random.sample(palette, color_count)
    counts = [2] * color_count
    remaining = total - 2 * color_count
    while remaining > 0:
        counts[random.randrange(color_count)] += 1
        remaining -= 1
    colors: list[int] = []
    for color, count in zip(chosen, counts):
        colors.extend([color] * count)
    random.shuffle(colors)
    return colors


def _select_canvas(masks: list[np.ndarray]) -> tuple[int, int] | None:
    total_padded_area = sum((mask.shape[0] + 2 * OBJECT_PAD) * (mask.shape[1] + 2 * OBJECT_PAD) for mask in masks)
    side_hint = int(math.ceil(math.sqrt(total_padded_area * 1.25)))
    min_height = max(18, max(int(mask.shape[0]) for mask in masks) + 2, side_hint - random.randint(0, 2))
    min_width = max(18, max(int(mask.shape[1]) for mask in masks) + 2, side_hint - random.randint(0, 2))
    if min_height > 30 or min_width > 30:
        return None
    height = random.randint(min_height, min(30, min_height + 6))
    width = random.randint(min_width, min(30, min_width + 6))
    return height, width


def _pack_scene(masks: list[np.ndarray], colors: list[int], bg: int) -> np.ndarray | None:
    for _ in range(30):
        dims = _select_canvas(masks)
        if dims is None:
            return None
        height, width = dims
        grid = np.full((height, width), bg, dtype=int)
        occupied = np.zeros((height, width), dtype=bool)
        order = list(range(len(masks)))
        random.shuffle(order)
        order.sort(key=lambda idx: masks[idx].shape[0] * masks[idx].shape[1], reverse=True)

        ok = True
        for idx in order:
            mask = masks[idx]
            mh, mw = mask.shape
            placed = False
            for _ in range(300):
                top = random.randint(0, height - mh)
                left = random.randint(0, width - mw)
                r0 = max(0, top - OBJECT_PAD)
                c0 = max(0, left - OBJECT_PAD)
                r1 = min(height, top + mh + OBJECT_PAD)
                c1 = min(width, left + mw + OBJECT_PAD)
                if occupied[r0:r1, c0:c1].any():
                    continue
                patch = grid[top : top + mh, left : left + mw]
                if np.any(patch[mask] != bg):
                    continue
                patch[mask] = colors[idx]
                occupied[r0:r1, c0:c1] = True
                placed = True
                break
            if not placed:
                ok = False
                break
        if ok:
            return grid
    return None


def _gen_attempt() -> np.ndarray | None:
    bg = random.randrange(10)
    stamp_color = random.choice([color for color in range(10) if color != bg])
    stamp = _generate_stamp()
    if stamp is None:
        return None

    total_other = random.randint(4, 6)
    positive_count = random.choice([total_other // 2, (total_other + 1) // 2])
    negative_count = total_other - positive_count
    if positive_count < 1 or negative_count < 1:
        return None

    positive_masks: list[np.ndarray] = []
    for _ in range(positive_count):
        mask = _make_container(stamp)
        if mask is None:
            return None
        positive_masks.append(mask)

    negative_masks: list[np.ndarray] = []
    transformed_options = [variant for variant in _orientation_variants(stamp) if _mask_key(variant) != _mask_key(stamp)]
    if transformed_options and negative_count > 0 and random.random() < 0.8:
        negative_masks.append(random.choice(transformed_options).copy())
    while len(negative_masks) < negative_count:
        mask = _make_noncontainer(stamp, force_large=(len(negative_masks) == 0 and not transformed_options))
        if mask is None:
            return None
        negative_masks.append(mask)

    if not any(mask.shape != stamp.shape or int(mask.sum()) > int(stamp.sum()) for mask in positive_masks):
        return None
    if not any(mask.shape[0] >= stamp.shape[0] and mask.shape[1] >= stamp.shape[1] for mask in negative_masks):
        return None

    other_colors = _other_colors(total_other, bg, stamp_color)
    object_masks = [stamp] + positive_masks + negative_masks
    object_colors = [stamp_color] + other_colors
    grid = _pack_scene(object_masks, object_colors, bg)
    if grid is None:
        return None

    if get_bg(grid) != bg:
        return None
    objects = _extract_objects(grid, bg)
    if len(objects) != total_other + 1:
        return None
    color_freq = Counter(int(obj["color"]) for obj in objects)
    if color_freq.get(stamp_color) != 1:
        return None
    if sum(1 for count in color_freq.values() if count == 1) != 1:
        return None

    out = main(grid)
    if np.array_equal(out, grid):
        return None
    if int(np.count_nonzero(out == stamp_color)) <= int(np.count_nonzero(grid == stamp_color)):
        return None
    if int(np.count_nonzero(out == bg)) <= int(np.count_nonzero(grid == bg)):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
