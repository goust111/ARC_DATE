import random

import numpy as np


BLACK = 0
GREY = 5
PALETTE = tuple(color for color in range(10) if color not in {BLACK, GREY})

LEGEND_ORDERS = (1, 2, 4)
LEGEND_WIDTH = 7

SHAPE_SIDES = (5, 7, 9)
GEN_ATTEMPTS = 400
SHAPE_ATTEMPTS = 200
PLACEMENT_PADDING = 1


def _neighbors(r: int, c: int, connectivity: int):
    if connectivity == 4:
        deltas = ((1, 0), (-1, 0), (0, 1), (0, -1))
    else:
        deltas = (
            (1, 0),
            (-1, 0),
            (0, 1),
            (0, -1),
            (1, 1),
            (1, -1),
            (-1, 1),
            (-1, -1),
        )
    for dr, dc in deltas:
        yield r + dr, c + dc


def _components(mask: np.ndarray, connectivity: int = 8) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    parts: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors(rr, cc, connectivity):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            parts.append(np.asarray(coords, dtype=int))
    return parts


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _most_common_color(grid: np.ndarray) -> int:
    colors, counts = np.unique(grid, return_counts=True)
    return int(colors[np.argmax(counts)])


def _rotation_order(coords: np.ndarray, center: tuple[int, int]) -> int:
    cr, cc = center
    points = {(int(r) - cr, int(c) - cc) for r, c in coords}
    rot90 = {(-dc, dr) for dr, dc in points}
    rot180 = {(-dr, -dc) for dr, dc in points}
    rot270 = {(dc, -dr) for dr, dc in points}
    if rot90 == points and rot180 == points and rot270 == points:
        return 4
    if rot180 == points:
        return 2
    return 1


def _legend_colors(legend_row: np.ndarray, bg: int) -> dict[int, int]:
    cols = np.flatnonzero(legend_row != bg)
    assert len(cols) == 3
    start = int(cols[0])
    assert start + 6 < legend_row.size
    assert int(cols[1]) == start + 2
    assert int(cols[2]) == start + 6
    assert int(legend_row[start + 1]) == bg
    assert int(legend_row[start + 3]) == bg
    assert int(legend_row[start + 4]) == bg
    assert int(legend_row[start + 5]) == bg
    return {
        1: int(legend_row[start]),
        2: int(legend_row[start + 2]),
        4: int(legend_row[start + 6]),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Replace each ring's grey center with the legend color for that ring's rotation-symmetry order.

    Input structure:
    - The background is black.
    - Above the bottom row are several one-pixel-wide closed colored rings.
    - Each ring encloses exactly one grey pixel placed at the ring's geometric center.
    - The bottom row is a spaced legend whose 1st, 2nd, and 4th slots encode the colors for rotation orders 1, 2, and 4.

    Rule:
    - Keep the rings, legend, and background unchanged.
    - For each ring, use its enclosed grey pixel as the rotation center and compare the ring pixel set with its 90-degree, 180-degree, and 270-degree rotations.
    - Let k be the highest matching rotation order from {1, 2, 4}.
    - Recolor that ring's grey center with the legend color stored in slot k of the bottom-row legend.

    Key insight:
    - The ring color does not matter; the task first converts geometry into a symmetry order and then converts that order into a color through the legend.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = _most_common_color(g)
    legend = _legend_colors(g[-1], bg)

    object_mask = (g[:-1] != bg) & (g[:-1] != GREY)
    parts = _components(object_mask, connectivity=8)
    assert parts

    out = g.copy()
    centers_seen: set[tuple[int, int]] = set()
    for coords in parts:
        r0, c0, r1, c1 = _bbox(coords)
        grey_coords = np.argwhere(g[r0 : r1 + 1, c0 : c1 + 1] == GREY)
        assert len(grey_coords) == 1
        cr, cc = map(int, grey_coords[0])
        center = (r0 + cr, c0 + cc)
        centers_seen.add(center)
        order = _rotation_order(coords, center)
        out[center] = legend[order]

    assert len(centers_seen) == len(parts)
    return out


def _centered_interval(width: int, total_width: int) -> tuple[int, int]:
    left = (total_width - width) // 2
    return left, left + width - 1


def _step_interval(interval: tuple[int, int], center_col: int, total_width: int) -> tuple[int, int]:
    left, right = interval
    left = random.randint(max(0, left - 1), min(center_col - 1, left + 1))
    right = random.randint(max(center_col + 1, right - 1), min(total_width - 1, right + 1))
    return left, right


def _solid_from_intervals(rows: list[tuple[int, int]], total_width: int) -> np.ndarray:
    solid = np.zeros((len(rows), total_width), dtype=bool)
    for r, (left, right) in enumerate(rows):
        solid[r, left : right + 1] = True
    return solid


def _solid_from_widths(widths: list[int], total_width: int | None = None) -> np.ndarray:
    width = max(widths) if total_width is None else total_width
    rows = [_centered_interval(span, width) for span in widths]
    return _solid_from_intervals(rows, width)


def _solid_has_center_cross(solid: np.ndarray) -> bool:
    h, w = solid.shape
    cr, cc = h // 2, w // 2
    needed = ((cr, cc), (cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1))
    return all(solid[r, c] for r, c in needed)


def _ring_from_solid(solid: np.ndarray) -> np.ndarray:
    h, w = solid.shape
    ring = np.zeros_like(solid, dtype=bool)
    for r in range(h):
        for c in range(w):
            if not solid[r, c]:
                continue
            for nr, nc in _neighbors(r, c, 4):
                if not (0 <= nr < h and 0 <= nc < w and solid[nr, nc]):
                    ring[r, c] = True
                    break
    return ring


def _valid_ring_mask(mask: np.ndarray, order: int) -> bool:
    h, w = mask.shape
    if h % 2 == 0 or w % 2 == 0:
        return False
    cr, cc = h // 2, w // 2
    if mask[cr, cc]:
        return False
    if len(_components(mask, connectivity=8)) != 1:
        return False
    return _rotation_order(np.argwhere(mask), (cr, cc)) == order


def _make_order4_solid() -> np.ndarray:
    size = random.choice(SHAPE_SIDES)
    family = random.choice(("square", "rounded", "octagon"))
    if family == "square":
        return np.ones((size, size), dtype=bool)
    if family == "rounded":
        widths = [size - 2] + [size] * (size - 2) + [size - 2]
        return _solid_from_widths(widths, size)
    shoulders = list(range(3, size, 2))
    widths = shoulders + [size, size, size] + shoulders[::-1]
    return _solid_from_widths(widths, size)


def _make_rectangle_like_solid() -> np.ndarray:
    h = random.choice(SHAPE_SIDES)
    w = random.choice([side for side in SHAPE_SIDES if side != h])
    if random.random() < 0.5:
        return np.ones((h, w), dtype=bool)
    widths = [w - 2] + [w] * (h - 2) + [w - 2]
    return _solid_from_widths(widths, w)


def _make_half_turn_solid() -> np.ndarray:
    h = random.choice(SHAPE_SIDES)
    w = random.choice([side for side in SHAPE_SIDES if side != h])
    cr, cc = h // 2, w // 2

    center_width = random.choice(range(3, w + 1, 2))
    rows: list[tuple[int, int]] = [(-1, -1)] * h
    rows[cr] = _centered_interval(center_width, w)
    prev = rows[cr]
    for r in range(cr - 1, -1, -1):
        prev = _step_interval(prev, cc, w)
        rows[r] = prev
    for r in range(cr + 1, h):
        left, right = rows[h - 1 - r]
        rows[r] = (w - 1 - right, w - 1 - left)
    return _solid_from_intervals(rows, w)


def _make_asymmetric_solid() -> np.ndarray:
    h = random.choice(SHAPE_SIDES)
    w = random.choice(SHAPE_SIDES)
    cr, cc = h // 2, w // 2

    center_width = random.choice(range(3, w + 1, 2))
    center = _centered_interval(center_width, w)
    rows: list[tuple[int, int]] = [(-1, -1)] * h
    rows[cr] = center

    prev = center
    for r in range(cr - 1, -1, -1):
        prev = _step_interval(prev, cc, w)
        rows[r] = prev

    prev = center
    for r in range(cr + 1, h):
        prev = _step_interval(prev, cc, w)
        rows[r] = prev
    return _solid_from_intervals(rows, w)


def _generate_ring_mask(order: int) -> np.ndarray:
    for _ in range(SHAPE_ATTEMPTS):
        if order == 4:
            solid = _make_order4_solid()
        elif order == 2:
            solid = _make_half_turn_solid() if random.random() < 0.5 else _make_rectangle_like_solid()
        else:
            solid = _make_asymmetric_solid()

        if not _solid_has_center_cross(solid):
            continue
        ring = _ring_from_solid(solid)
        if _valid_ring_mask(ring, order):
            return ring
    raise RuntimeError(f"failed to build a rotation-order-{order} ring")


def _render_patch(ring_mask: np.ndarray, color: int) -> np.ndarray:
    patch = np.full(ring_mask.shape, BLACK, dtype=int)
    patch[ring_mask] = color
    patch[ring_mask.shape[0] // 2, ring_mask.shape[1] // 2] = GREY
    return patch


def _sample_orders(count: int) -> list[int]:
    if count == 2:
        return random.sample(list(LEGEND_ORDERS), 2)
    if count == 3:
        out = list(LEGEND_ORDERS)
        random.shuffle(out)
        return out
    out = [1, 2, 4, random.choice(LEGEND_ORDERS)]
    random.shuffle(out)
    return out


def _place_patch(
    grid: np.ndarray,
    blocked: np.ndarray,
    patch: np.ndarray,
) -> tuple[int, int] | None:
    h, w = grid.shape
    ph, pw = patch.shape
    rows = list(range(1, h - 1 - ph))
    cols = list(range(1, w - pw))
    if not rows or not cols:
        return None
    candidates = [(r, c) for r in rows for c in cols]
    random.shuffle(candidates)

    for r, c in candidates:
        r0 = max(0, r - PLACEMENT_PADDING)
        c0 = max(0, c - PLACEMENT_PADDING)
        r1 = min(h, r + ph + PLACEMENT_PADDING)
        c1 = min(w, c + pw + PLACEMENT_PADDING)
        if blocked[r0:r1, c0:c1].any():
            continue

        blocked[r0:r1, c0:c1] = True
        mask = patch != BLACK
        grid[r : r + ph, c : c + pw][mask] = patch[mask]
        return r, c
    return None


def _gen_attempt() -> np.ndarray | None:
    ring_count = random.randint(2, 4)
    orders = _sample_orders(ring_count)

    legend_colors = random.sample(PALETTE, len(LEGEND_ORDERS))
    order_to_color = {order: color for order, color in zip(LEGEND_ORDERS, legend_colors)}
    ring_colors = random.sample([color for color in PALETTE if color not in legend_colors], ring_count)

    items = []
    for order, ring_color in zip(orders, ring_colors):
        ring_mask = _generate_ring_mask(order)
        patch = _render_patch(ring_mask, ring_color)
        items.append({"order": order, "patch": patch, "area": int(ring_mask.sum())})

    bbox_load = sum((item["patch"].shape[0] + 2) * (item["patch"].shape[1] + 2) for item in items)
    max_h = max(item["patch"].shape[0] for item in items)
    max_w = max(item["patch"].shape[1] for item in items)
    min_side = max(max_h + 4, max_w + 4, int(np.ceil(np.sqrt(bbox_load))) + 4, LEGEND_WIDTH + 3)
    if min_side > 30:
        return None

    side_hi = min(30, min_side + 6)
    h = random.randint(min_side, side_hi)
    w = random.randint(min_side, 30)
    if w < LEGEND_WIDTH + 2:
        return None

    grid = np.full((h, w), BLACK, dtype=int)
    expected = grid.copy()
    blocked = np.zeros_like(grid, dtype=bool)
    blocked[-2:, :] = True

    centers: list[tuple[int, int] | None] = [None] * len(items)
    place_order = sorted(range(len(items)), key=lambda idx: items[idx]["area"], reverse=True)
    for idx in place_order:
        patch = items[idx]["patch"]
        top_left = _place_patch(grid, blocked, patch)
        if top_left is None:
            return None
        r, c = top_left
        cr = r + patch.shape[0] // 2
        cc = c + patch.shape[1] // 2
        centers[idx] = (cr, cc)

    legend_start_choices = list(range(1, w - LEGEND_WIDTH + 1))
    if not legend_start_choices:
        return None
    legend_start = random.choice(legend_start_choices)
    grid[-1, legend_start] = order_to_color[1]
    grid[-1, legend_start + 2] = order_to_color[2]
    grid[-1, legend_start + 6] = order_to_color[4]

    expected[:] = grid
    for item, center in zip(items, centers):
        assert center is not None
        expected[center] = order_to_color[item["order"]]

    try:
        solved = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(solved, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
