from __future__ import annotations

from collections import deque
import random

import numpy as np


DIR4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
PALETTE = tuple(range(10))
MAX_GENERATION_ATTEMPTS = 400


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _count_holes(mask: np.ndarray) -> int:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    holes = 0

    for r in range(h):
        for c in range(w):
            if mask[r, c] or seen[r, c]:
                continue
            queue = deque([(r, c)])
            seen[r, c] = True
            touches_border = r in (0, h - 1) or c in (0, w - 1)

            while queue:
                cr, cc = queue.popleft()
                for dr, dc in DIR4:
                    nr, nc = cr + dr, cc + dc
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if mask[nr, nc] or seen[nr, nc]:
                        continue
                    seen[nr, nc] = True
                    if nr in (0, h - 1) or nc in (0, w - 1):
                        touches_border = True
                    queue.append((nr, nc))

            if not touches_border:
                holes += 1
    return holes


def _extract_objects(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    objects: list[dict] = []

    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(g[r, c]) == bg:
                continue

            color = int(g[r, c])
            queue = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []

            while queue:
                cr, cc = queue.popleft()
                coords.append((cr, cc))
                for dr, dc in DIR4:
                    nr, nc = cr + dr, cc + dc
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if seen[nr, nc] or int(g[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    queue.append((nr, nc))

            arr = np.asarray(coords, dtype=int)
            r0 = int(arr[:, 0].min())
            r1 = int(arr[:, 0].max())
            c0 = int(arr[:, 1].min())
            c1 = int(arr[:, 1].max())
            mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
            rel = arr - np.array([r0, c0], dtype=int)
            mask[rel[:, 0], rel[:, 1]] = True

            objects.append(
                {
                    "color": color,
                    "coords": arr,
                    "bbox": (r0, c0, r1, c1),
                    "rel_coords": rel,
                    "mask": mask,
                    "height": int(mask.shape[0]),
                    "width": int(mask.shape[1]),
                    "area": int(mask.sum()),
                    "holes": _count_holes(mask),
                }
            )

    return objects


def _sort_objects(objects: list[dict]) -> list[dict]:
    return sorted(
        objects,
        key=lambda obj: (-int(obj["area"]), int(obj["bbox"][0]), int(obj["bbox"][1])),
    )


def _paint_object(out: np.ndarray, obj: dict, top: int, left: int, bg: int) -> None:
    h, w = out.shape
    for rr, cc in obj["rel_coords"]:
        r = int(top + rr)
        c = int(left + cc)
        assert 0 <= r < h and 0 <= c < w
        assert int(out[r, c]) == bg, "rearranged objects must not overlap"
        out[r, c] = int(obj["color"])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Group foreground objects by hole-count parity, move even-hole objects to the
    left edge and odd-hole objects to the right edge, and stack each group from
    largest area to smallest with one blank row between consecutive objects.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    h, w = g.shape
    objects = _extract_objects(g)
    assert objects, "expected at least one foreground object"

    left_group = _sort_objects([obj for obj in objects if int(obj["holes"]) % 2 == 0])
    right_group = _sort_objects([obj for obj in objects if int(obj["holes"]) % 2 == 1])

    out = np.full((h, w), bg, dtype=int)

    top = 0
    for obj in left_group:
        assert top + int(obj["height"]) <= h
        _paint_object(out, obj, top, 0, bg)
        top += int(obj["height"]) + 1

    top = 0
    for obj in right_group:
        assert top + int(obj["height"]) <= h
        left = w - int(obj["width"])
        assert left >= 0
        _paint_object(out, obj, top, left, bg)
        top += int(obj["height"]) + 1

    return out


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    rows = np.flatnonzero(mask.any(axis=1))
    cols = np.flatnonzero(mask.any(axis=0))
    assert rows.size and cols.size
    return mask[rows.min() : rows.max() + 1, cols.min() : cols.max() + 1].copy()


def _random_transform(mask: np.ndarray) -> np.ndarray:
    out = mask.copy()
    for _ in range(random.randrange(4)):
        out = np.rot90(out)
    if random.random() < 0.5:
        out = np.fliplr(out)
    return out.copy()


def _boxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (ar1 < br0 or br1 < ar0 or ac1 < bc0 or bc1 < ac0)


def _choose_hole_rects(base_h: int, base_w: int, hole_count: int) -> list[tuple[int, int, int, int]] | None:
    if hole_count == 0:
        return []

    for _ in range(80):
        chosen: list[tuple[int, int, int, int]] = []
        blocked: list[tuple[int, int, int, int]] = []
        ok = True

        for _idx in range(hole_count):
            options: list[tuple[int, int, int, int]] = []
            hole_sizes = [(1, 1), (1, 2), (2, 1), (2, 2)]
            random.shuffle(hole_sizes)

            for hh, ww in hole_sizes:
                if base_h <= hh + 2 or base_w <= ww + 2:
                    continue
                for r in range(1, base_h - hh):
                    for c in range(1, base_w - ww):
                        rect = (r, c, r + hh - 1, c + ww - 1)
                        halo = (r - 1, c - 1, r + hh, c + ww)
                        if any(_boxes_overlap(halo, prev) for prev in blocked):
                            continue
                        options.append(rect)

            if not options:
                ok = False
                break

            rect = random.choice(options)
            rr0, cc0, rr1, cc1 = rect
            chosen.append(rect)
            blocked.append((rr0 - 1, cc0 - 1, rr1 + 1, cc1 + 1))

        if ok:
            return chosen
    return None


def _add_tabs(mask: np.ndarray, base_r0: int, base_c0: int, base_h: int, base_w: int) -> None:
    tab_count = random.randint(0, 3)
    for _ in range(tab_count):
        side = random.choice(("top", "bottom", "left", "right"))
        if side in ("top", "bottom"):
            depth = random.randint(1, 2)
            width = random.randint(1, min(base_w, max(2, base_w // 2)))
            start_c = random.randint(base_c0, base_c0 + base_w - width)
            r0 = base_r0 - depth if side == "top" else base_r0 + base_h
            r1 = r0 + depth - 1
            c0 = start_c
            c1 = start_c + width - 1
        else:
            depth = random.randint(1, 2)
            height = random.randint(1, min(base_h, max(2, base_h // 2)))
            start_r = random.randint(base_r0, base_r0 + base_h - height)
            r0 = start_r
            r1 = start_r + height - 1
            c0 = base_c0 - depth if side == "left" else base_c0 + base_w
            c1 = c0 + depth - 1

        if r0 < 0 or c0 < 0 or r1 >= mask.shape[0] or c1 >= mask.shape[1]:
            continue
        if bool(mask[r0 : r1 + 1, c0 : c1 + 1].any()):
            continue
        mask[r0 : r1 + 1, c0 : c1 + 1] = True


def _trim_corners(mask: np.ndarray, base_r0: int, base_c0: int, base_h: int, base_w: int) -> None:
    corners = [
        (base_r0, base_c0),
        (base_r0, base_c0 + base_w - 1),
        (base_r0 + base_h - 1, base_c0),
        (base_r0 + base_h - 1, base_c0 + base_w - 1),
    ]
    for r, c in corners:
        if random.random() < 0.35:
            mask[r, c] = False


def _make_object_mask(hole_count: int) -> np.ndarray | None:
    pad = 2
    for _ in range(120):
        base_h_min = 3 if hole_count == 0 else 4 + (hole_count >= 2)
        base_w_min = 3 if hole_count == 0 else 4 + hole_count
        base_h = random.randint(base_h_min, min(8, base_h_min + 3))
        base_w = random.randint(base_w_min, min(10, base_w_min + 4))

        hole_rects = _choose_hole_rects(base_h, base_w, hole_count)
        if hole_rects is None:
            continue

        canvas = np.zeros((base_h + 2 * pad, base_w + 2 * pad), dtype=bool)
        r0 = c0 = pad
        canvas[r0 : r0 + base_h, c0 : c0 + base_w] = True

        for hr0, hc0, hr1, hc1 in hole_rects:
            canvas[r0 + hr0 : r0 + hr1 + 1, c0 + hc0 : c0 + hc1 + 1] = False

        _trim_corners(canvas, r0, c0, base_h, base_w)
        _add_tabs(canvas, r0, c0, base_h, base_w)

        mask = _crop_mask(canvas)
        mask = _random_transform(mask)

        if int(mask.sum()) < 4:
            continue
        if mask.shape[0] > 10 or mask.shape[1] > 12:
            continue
        if _count_holes(mask) != hole_count:
            continue
        return mask
    return None


def _sample_hole_count(parity: int, must_be_zero: bool = False) -> int:
    if must_be_zero:
        return 0
    if parity == 0:
        return random.choices([0, 2], weights=[3, 1])[0]
    return random.choices([1, 3], weights=[3, 1])[0]


def _build_group_objects(parity: int, count: int, color_iter: list[int], force_zero: bool) -> list[dict] | None:
    objects: list[dict] = []
    used_areas: set[int] = set()

    for idx in range(count):
        hole_count = _sample_hole_count(parity, must_be_zero=force_zero and idx == 0)
        built = None
        for _ in range(120):
            mask = _make_object_mask(hole_count)
            if mask is None:
                continue
            area = int(mask.sum())
            if area in used_areas:
                continue
            coords = np.argwhere(mask).astype(int)
            built = {
                "color": int(color_iter[len(objects)]),
                "mask": mask,
                "rel_coords": coords,
                "height": int(mask.shape[0]),
                "width": int(mask.shape[1]),
                "area": area,
                "holes": hole_count,
            }
            break
        if built is None:
            return None
        used_areas.add(int(built["area"]))
        objects.append(built)

    return objects


def _stack_height(objects: list[dict]) -> int:
    if not objects:
        return 0
    return sum(int(obj["height"]) for obj in objects) + len(objects) - 1


def _boxes_too_close(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (ar1 + 1 < br0 or br1 + 1 < ar0 or ac1 + 1 < bc0 or bc1 + 1 < ac0)


def _place_objects_randomly(objects: list[dict], height: int, width: int, bg: int) -> np.ndarray | None:
    grid = np.full((height, width), bg, dtype=int)
    placed_boxes: list[tuple[int, int, int, int]] = []

    order = sorted(range(len(objects)), key=lambda i: objects[i]["height"] * objects[i]["width"], reverse=True)
    for idx in order:
        obj = objects[idx]
        oh = int(obj["height"])
        ow = int(obj["width"])
        if height - oh - 1 < 1 or width - ow - 1 < 1:
            return None

        candidates = [
            (r, c)
            for r in range(1, height - oh)
            for c in range(1, width - ow)
        ]
        random.shuffle(candidates)

        placed = False
        for top, left in candidates[:200]:
            box = (top, left, top + oh - 1, left + ow - 1)
            if any(_boxes_too_close(box, prev) for prev in placed_boxes):
                continue
            coords = obj["rel_coords"] + np.array([top, left], dtype=int)
            if np.any(grid[coords[:, 0], coords[:, 1]] != bg):
                continue
            grid[coords[:, 0], coords[:, 1]] = int(obj["color"])
            placed_boxes.append(box)
            placed = True
            break

        if not placed:
            return None

    return grid


def _choose_grid_shape(objects: list[dict], left_group: list[dict], right_group: list[dict]) -> tuple[int, int] | None:
    min_h = max(_stack_height(left_group), _stack_height(right_group), max(int(obj["height"]) for obj in objects) + 2)
    min_w = max(
        max(int(obj["width"]) for obj in left_group) + max(int(obj["width"]) for obj in right_group) + 2,
        max(int(obj["width"]) for obj in objects) + 2,
    )
    total_fg = sum(int(obj["area"]) for obj in objects)

    if min_h > 30 or min_w > 30:
        return None

    for _ in range(80):
        height = random.randint(max(18, min_h), min(30, max(min_h, 24)))
        width = random.randint(max(18, min_w), min(30, max(min_w, 26)))
        if height * width <= 2 * total_fg:
            continue
        return height, width
    return None


def _generate_attempt() -> np.ndarray | None:
    obj_count = random.randint(4, 6)
    even_count = obj_count // 2
    if obj_count % 2 == 1 and random.random() < 0.5:
        even_count += 1
    odd_count = obj_count - even_count
    assert even_count >= 1 and odd_count >= 1

    bg = random.choice(PALETTE)
    colors = random.sample([color for color in PALETTE if color != bg], obj_count)

    left_colors = colors[:even_count]
    right_colors = colors[even_count:]

    left_group = _build_group_objects(0, even_count, left_colors, force_zero=True)
    if left_group is None:
        return None
    right_group = _build_group_objects(1, odd_count, right_colors, force_zero=False)
    if right_group is None:
        return None

    all_objects = left_group + right_group
    dims = _choose_grid_shape(all_objects, left_group, right_group)
    if dims is None:
        return None
    height, width = dims

    grid = _place_objects_randomly(all_objects, height, width, bg)
    if grid is None:
        return None

    try:
        objs = _extract_objects(grid)
        if len(objs) != obj_count:
            return None
        if not any(int(obj["holes"]) == 0 for obj in objs):
            return None
        if not any(int(obj["holes"]) > 0 for obj in objs):
            return None
        if not any(int(obj["holes"]) % 2 == 0 for obj in objs):
            return None
        if not any(int(obj["holes"]) % 2 == 1 for obj in objs):
            return None
        out = main(grid)
    except Exception:
        return None

    if np.array_equal(grid, out):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
