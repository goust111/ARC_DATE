from __future__ import annotations

import random
from typing import Any

import numpy as np


BG = 0
GREY = 5
NEIGHBORS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
CATEGORIES = ("rectangle", "L", "cross")
COUNT_PATTERNS = (
    (1, 1, 1),
    (2, 1, 1),
    (1, 2, 1),
    (1, 1, 2),
    (2, 2, 1),
    (2, 1, 2),
    (1, 2, 2),
)


def _get_bg(grid: np.ndarray) -> int:
    vals, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(vals[int(np.argmax(counts))])


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)
    return int(mins[0]), int(mins[1]), int(maxs[0]), int(maxs[1])


def _mask_from_coords(coords: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    r0, c0, r1, c1 = _bbox(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return mask, (r0, c0, r1, c1)


def _components_by_color(grid: np.ndarray, bg: int) -> list[dict[str, Any]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict[str, Any]] = []

    for r in range(h):
        for c in range(w):
            color = int(g[r, c])
            if color == int(bg) or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                cells.append((int(rr), int(cc)))
                for dr, dc in NEIGHBORS4:
                    nr = int(rr + dr)
                    nc = int(cc + dc)
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(g[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))

            coords = np.array(cells, dtype=int)
            mask, box = _mask_from_coords(coords)
            comps.append(
                {
                    "color": int(color),
                    "coords": coords,
                    "bbox": box,
                    "mask": mask,
                    "area": int(coords.shape[0]),
                }
            )

    return comps


def _is_frame(mask: np.ndarray) -> bool:
    h, w = mask.shape
    if h < 3 or w < 3:
        return False
    expected = np.zeros_like(mask, dtype=bool)
    expected[0, :] = True
    expected[-1, :] = True
    expected[:, 0] = True
    expected[:, -1] = True
    return bool(np.array_equal(mask, expected))


def _cross_mask(corner_size: int, inner_h: int, inner_w: int) -> np.ndarray:
    h = inner_h + 2 * corner_size
    w = inner_w + 2 * corner_size
    mask = np.zeros((h, w), dtype=bool)
    mask[corner_size : h - corner_size, :] = True
    mask[:, corner_size : w - corner_size] = True
    return mask


def _classify_mask(mask: np.ndarray) -> str | None:
    shape = np.asarray(mask, dtype=bool)
    h, w = shape.shape

    if bool(shape.all()):
        return "rectangle"

    max_corner = min(h, w) // 2
    for corner in range(1, max_corner + 1):
        if h <= 2 * corner or w <= 2 * corner:
            continue
        if np.array_equal(shape, _cross_mask(corner, h - 2 * corner, w - 2 * corner)):
            return "cross"

    empties = ~shape
    if bool(empties.any()):
        er0, ec0, er1, ec1 = _bbox(np.argwhere(empties))
        removed = empties[er0 : er1 + 1, ec0 : ec1 + 1]
        if bool(removed.all()):
            anchored = (
                (er0 == 0 and ec0 == 0)
                or (er0 == 0 and ec1 == w - 1)
                or (er1 == h - 1 and ec0 == 0)
                or (er1 == h - 1 and ec1 == w - 1)
            )
            if anchored:
                return "L"

    return None


def _paste_mask(grid: np.ndarray, mask: np.ndarray, top: int, left: int, color: int) -> None:
    h, w = mask.shape
    grid[top : top + h, left : left + w][mask] = int(color)


def _container_specs(grid: np.ndarray, bg: int) -> list[dict[str, Any]]:
    comps = _components_by_color(grid, bg)
    containers: list[dict[str, Any]] = []
    for comp in comps:
        if not _is_frame(comp["mask"]):
            continue
        r0, c0, r1, c1 = map(int, comp["bbox"])
        if r1 - r0 < 4 or c1 - c0 < 4:
            continue
        inner = grid[r0 + 1 : r1, c0 + 1 : c1]
        icon_coords = np.argwhere(inner != int(bg))
        if icon_coords.size == 0:
            continue
        icon_mask, _icon_bbox = _mask_from_coords(icon_coords)
        category = _classify_mask(icon_mask)
        if category is None:
            continue
        containers.append(
            {
                "bbox": (r0, c0, r1, c1),
                "category": category,
                "color": int(comp["color"]),
            }
        )

    containers.sort(key=lambda item: (int(item["bbox"][0]), int(item["bbox"][1])))
    return containers


def _objects_outside_containers(grid: np.ndarray, bg: int, containers: list[dict[str, Any]]) -> list[dict[str, Any]]:
    comps = _components_by_color(grid, bg)
    frame_boxes = {tuple(comp["bbox"]) for comp in comps if _is_frame(comp["mask"])}

    def inside_any_container(r: int, c: int) -> bool:
        for container in containers:
            r0, c0, r1, c1 = map(int, container["bbox"])
            if r0 <= r <= r1 and c0 <= c <= c1:
                return True
        return False

    objects: list[dict[str, Any]] = []
    for comp in comps:
        if tuple(comp["bbox"]) in frame_boxes and _is_frame(comp["mask"]):
            continue
        rr, cc = map(int, comp["coords"][0])
        if inside_any_container(rr, cc):
            continue
        category = _classify_mask(comp["mask"])
        assert category is not None, "all scattered objects must match one of the three shape families"
        item = dict(comp)
        item["category"] = category
        objects.append(item)

    return objects


def _apply_rule(grid: np.ndarray, containers: list[dict[str, Any]], objects: list[dict[str, Any]], bg: int) -> np.ndarray:
    out = np.array(grid, dtype=int, copy=True)
    by_category: dict[str, list[dict[str, Any]]] = {name: [] for name in CATEGORIES}
    for obj in objects:
        by_category[str(obj["category"])].append(obj)
        out[obj["coords"][:, 0], obj["coords"][:, 1]] = int(bg)

    container_by_category = {str(container["category"]): container for container in containers}
    assert set(container_by_category) == set(CATEGORIES), "expected one container icon for each shape family"

    for category, items in by_category.items():
        if not items:
            continue
        container = container_by_category[category]
        r0, c0, r1, c1 = map(int, container["bbox"])
        out[r0 + 1 : r1, c0 + 1 : c1] = int(bg)

        ordered = sorted(
            items,
            key=lambda item: (
                int(item["area"]),
                int(item["bbox"][0]),
                int(item["bbox"][1]),
                int(item["mask"].shape[0]),
                int(item["mask"].shape[1]),
            ),
        )

        cursor = int(c0 + 1)
        inner_bottom = int(r1 - 1)
        max_right = int(c1 - 1)
        for item in ordered:
            mask = np.asarray(item["mask"], dtype=bool)
            h, w = mask.shape
            top = int(inner_bottom - h + 1)
            assert top >= int(r0 + 1)
            assert cursor + w - 1 <= max_right
            _paste_mask(out, mask, top, cursor, int(item["color"]))
            cursor += int(w + 1)

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move each scattered monochrome block into the bottom container whose grey icon shows the same shape family.

    Input structure:
    - Black background with several colored 4-connected monochrome components scattered above three bottom containers.
    - Each bottom container is a 1-pixel rectangular frame of its own color, and each interior has a small grey exemplar shape near the top-left.
    - The scattered components and the grey exemplars belong to exactly three geometry families: filled rectangles, L-shapes made by removing one corner rectangle, and crosses made by removing equal corner squares.

    Rule:
    - Detect the three container frames and read each grey exemplar by its occupied-cell pattern inside its tight bounding box.
    - Classify every scattered component by the same bounding-box test: fully filled bbox -> rectangle, full bbox minus one corner rectangle -> L, full bbox minus equal corner squares -> cross.
    - For each family, erase the original scattered components and place them into the matching container from left to right in ascending pixel area, with a 1-cell horizontal gap and with all bottoms aligned to the inner bottom edge.
    - When a container receives blocks, clear its grey exemplar before pasting the sorted shapes.

    Key insight:
    - The grey miniatures are shape exemplars rather than color labels, so the task requires geometric classification first and area-based ordering second.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    containers = _container_specs(g, bg)
    assert len(containers) == 3, "expected exactly three bottom containers"
    objects = _objects_outside_containers(g, bg, containers)
    return _apply_rule(g, containers, objects, bg)


def _make_rectangle_mask() -> np.ndarray:
    h = random.randint(2, 4)
    w = random.randint(2, 4)
    return np.ones((h, w), dtype=bool)


def _make_l_mask(icon: bool = False) -> np.ndarray:
    if icon:
        h, w = random.choice(((3, 3), (3, 4), (4, 3)))
    else:
        h = random.randint(3, 4)
        w = random.randint(3, 4)
    for _ in range(100):
        cut_h = random.randint(1, h - 1)
        cut_w = random.randint(1, w - 1)
        if cut_h * cut_w < 2:
            continue
        corner = random.choice(("tl", "tr", "bl", "br"))
        mask = np.ones((h, w), dtype=bool)
        if corner == "tl":
            mask[:cut_h, :cut_w] = False
        elif corner == "tr":
            mask[:cut_h, w - cut_w :] = False
        elif corner == "bl":
            mask[h - cut_h :, :cut_w] = False
        else:
            mask[h - cut_h :, w - cut_w :] = False
        if _classify_mask(mask) == "L":
            return mask
    raise RuntimeError("failed to build L-shape mask")


def _make_cross_mask(icon: bool = False) -> np.ndarray:
    corner = 1
    if icon:
        inner_h, inner_w = random.choice(((1, 1), (1, 2), (2, 1)))
    else:
        inner_h = random.randint(1, 2)
        inner_w = random.randint(1, 2)
    return _cross_mask(corner, inner_h, inner_w)


def _sample_mask(category: str, icon: bool = False) -> np.ndarray:
    if category == "rectangle":
        if icon:
            h, w = random.choice(((2, 3), (3, 2), (3, 3)))
            return np.ones((h, w), dtype=bool)
        return _make_rectangle_mask()
    if category == "L":
        return _make_l_mask(icon=icon)
    if category == "cross":
        return _make_cross_mask(icon=icon)
    raise ValueError(category)


def _shape_sprite(mask: np.ndarray, color: int) -> np.ndarray:
    sprite = np.full(mask.shape, BG, dtype=int)
    sprite[mask] = int(color)
    return sprite


def _try_place_sprite(grid: np.ndarray, sprite: np.ndarray, blocked: np.ndarray, row_limit: int) -> tuple[int, int] | None:
    h, w = sprite.shape
    H, W = grid.shape
    if h > row_limit or w > W:
        return None

    choices = [(r, c) for r in range(0, row_limit - h + 1) for c in range(0, W - w + 1)]
    random.shuffle(choices)
    for r, c in choices:
        r0 = max(0, r - 1)
        c0 = max(0, c - 1)
        r1 = min(H, r + h + 1)
        c1 = min(W, c + w + 1)
        if bool(blocked[r0:r1, c0:c1].any()):
            continue
        _paste_mask(grid, sprite != BG, r, c, int(sprite[sprite != BG][0]))
        blocked[r0:r1, c0:c1] = True
        return int(r), int(c)
    return None


def _build_expected(grid: np.ndarray, containers: list[dict[str, Any]], placed_objects: list[dict[str, Any]]) -> np.ndarray:
    objs: list[dict[str, Any]] = []
    for item in placed_objects:
        coords = item["coords"]
        mask, box = _mask_from_coords(coords)
        objs.append(
            {
                "color": int(item["color"]),
                "coords": coords,
                "bbox": box,
                "mask": mask,
                "area": int(coords.shape[0]),
                "category": str(item["category"]),
            }
        )
    return _apply_rule(grid, containers, objs, BG)


def _gen_attempt() -> np.ndarray | None:
    border_palette = [1, 2, 3, 4, 6, 7, 8, 9]
    border_colors = random.sample(border_palette, 3)
    object_palette = [color for color in border_palette if color not in border_colors]

    counts = dict(zip(CATEGORIES, random.choice(COUNT_PATTERNS), strict=False))
    total_objects = sum(counts.values())
    if total_objects > len(object_palette):
        return None

    object_colors = random.sample(object_palette, total_objects)
    icons = {category: _sample_mask(category, icon=True) for category in CATEGORIES}

    shapes_by_category: dict[str, list[dict[str, Any]]] = {name: [] for name in CATEGORIES}
    color_index = 0
    for category in CATEGORIES:
        used_areas: set[int] = set()
        for _ in range(counts[category]):
            mask = None
            for _attempt in range(200):
                candidate = _sample_mask(category, icon=False)
                area = int(candidate.sum())
                if area in used_areas:
                    continue
                mask = candidate
                used_areas.add(area)
                break
            if mask is None:
                return None
            color = int(object_colors[color_index])
            color_index += 1
            shapes_by_category[category].append(
                {
                    "category": category,
                    "color": color,
                    "mask": mask,
                    "sprite": _shape_sprite(mask, color),
                    "area": int(mask.sum()),
                }
            )
        shapes_by_category[category].sort(key=lambda item: int(item["area"]))

    inner_h = max(
        max(int(item["mask"].shape[0]) for items in shapes_by_category.values() for item in items),
        max(int(icon.shape[0]) for icon in icons.values()),
    )
    inner_widths = {}
    for category, items in shapes_by_category.items():
        shape_width = sum(int(item["mask"].shape[1]) for item in items) + max(0, len(items) - 1)
        inner_widths[category] = max(shape_width, int(icons[category].shape[1]))

    outer_widths = {category: int(inner_widths[category] + 2) for category in CATEGORIES}
    gap = random.choice((1, 2))
    total_container_width = sum(outer_widths.values()) + 2 * gap
    if total_container_width > 30:
        if gap == 2:
            gap = 1
            total_container_width = sum(outer_widths.values()) + 2 * gap
        if total_container_width > 30:
            return None

    container_h = int(inner_h + 2)
    top_h = random.randint(8, min(18, 30 - container_h - 1))
    H = int(top_h + 1 + container_h)
    if H > 30:
        return None

    extra_w = random.randint(0, max(0, 30 - total_container_width))
    W = int(total_container_width + extra_w)
    if W > 30:
        return None

    grid = np.full((H, W), BG, dtype=int)
    container_order = random.sample(list(CATEGORIES), len(CATEGORIES))
    x = random.randint(0, W - total_container_width)
    top = H - container_h

    containers: list[dict[str, Any]] = []
    container_by_category: dict[str, dict[str, Any]] = {}
    for color, category in zip(border_colors, container_order, strict=False):
        outer_w = int(outer_widths[category])
        r0 = int(top)
        c0 = int(x)
        r1 = int(r0 + container_h - 1)
        c1 = int(c0 + outer_w - 1)
        grid[r0 : r1 + 1, c0] = int(color)
        grid[r0 : r1 + 1, c1] = int(color)
        grid[r0, c0 : c1 + 1] = int(color)
        grid[r1, c0 : c1 + 1] = int(color)
        _paste_mask(grid, icons[category], r0 + 1, c0 + 1, GREY)
        container = {"bbox": (r0, c0, r1, c1), "category": category, "color": int(color)}
        containers.append(container)
        container_by_category[category] = container
        x += int(outer_w + gap)

    blocked = np.zeros_like(grid, dtype=bool)
    blocked[top:H, :] = True
    row_limit = int(top - 1)
    if row_limit < 4:
        return None

    all_items = [item for items in shapes_by_category.values() for item in items]
    random.shuffle(all_items)
    placed_objects: list[dict[str, Any]] = []
    for item in all_items:
        sprite = np.asarray(item["sprite"], dtype=int)
        placed = _try_place_sprite(grid, sprite, blocked, row_limit)
        if placed is None:
            return None
        r, c = placed
        mask = np.asarray(item["mask"], dtype=bool)
        coords_local = np.argwhere(mask)
        coords = coords_local + np.array([[r, c]], dtype=int)
        placed_objects.append(
            {
                "category": item["category"],
                "color": int(item["color"]),
                "coords": coords,
            }
        )

    if _get_bg(grid) != BG:
        return None

    expected = _build_expected(grid, containers, placed_objects)
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None

    detected = _container_specs(grid, BG)
    if len(detected) != 3 or {item["category"] for item in detected} != set(CATEGORIES):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(300):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("generate_input failed after 300 attempts")
