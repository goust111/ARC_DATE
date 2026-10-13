from __future__ import annotations

import random
import numpy as np


BG = 0
FRAME = 5
DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
PALETTE_HEIGHTS = (3, 4)
COLOR_POOL = tuple(color for color in range(10) if color not in (BG, FRAME))


def _components(mask: np.ndarray) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            stack = [(r, c)]
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((int(rr), int(cc)))
                for dr, dc in DIR4:
                    nr = int(rr + dr)
                    nc = int(cc + dc)
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(np.asarray(coords, dtype=int))
    return out


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    assert coords.size > 0, "pattern must contain at least one cell"
    r0, c0, r1, c1 = _bbox(coords)
    return np.asarray(mask[r0 : r1 + 1, c0 : c1 + 1], dtype=bool)


def _rotation_order(mask: np.ndarray) -> int:
    sprite = _crop_mask(mask)
    if np.array_equal(sprite, np.rot90(sprite, 1)):
        return 4
    if np.array_equal(sprite, np.rot90(sprite, 2)):
        return 2
    return 1


def _scan_palette_candidate(
    g: np.ndarray,
    *,
    top: bool,
    left: bool,
    height: int,
) -> dict | None:
    h, w = g.shape
    if height >= h:
        return None

    row_slice = slice(0, height) if top else slice(h - height, h)
    cols_iter = range(w) if left else range(w - 1, -1, -1)

    stripe_cols: list[int] = []
    stripe_colors: list[int] = []
    prev_color: int | None = None

    for c in cols_iter:
        column = g[row_slice, c]
        if not np.all(column == column[0]):
            break
        color = int(column[0])
        if color in (BG, FRAME):
            break
        if prev_color is not None and color == prev_color:
            return None
        stripe_cols.append(int(c))
        stripe_colors.append(color)
        prev_color = color

    if len(stripe_cols) < 4:
        return None

    c0 = min(stripe_cols)
    c1 = max(stripe_cols)
    r0 = 0 if top else h - height
    r1 = height - 1 if top else h - 1

    boundary_r = height if top else h - height - 1
    if 0 <= boundary_r < h and np.any(g[boundary_r, c0 : c1 + 1] != BG):
        return None

    boundary_c = len(stripe_cols) if left else w - len(stripe_cols) - 1
    if 0 <= boundary_c < w and np.any(g[r0 : r1 + 1, boundary_c] != BG):
        return None

    ordered = [
        color
        for _col, color in sorted(zip(stripe_cols, stripe_colors, strict=True), key=lambda item: item[0])
    ]
    return {
        "colors": ordered,
        "bbox": (int(r0), int(c0), int(r1), int(c1)),
    }


def _find_palette(g: np.ndarray) -> dict:
    candidates: list[dict] = []
    for height in PALETTE_HEIGHTS:
        for top in (True, False):
            for left in (True, False):
                candidate = _scan_palette_candidate(g, top=top, left=left, height=height)
                if candidate is not None:
                    candidates.append(candidate)
    unique: dict[tuple[int, int, int, int], dict] = {tuple(c["bbox"]): c for c in candidates}
    assert len(unique) == 1, "expected exactly one corner palette"
    palette = next(iter(unique.values()))
    assert len(palette["colors"]) >= 4, "palette must expose at least four stripes"
    return palette


def _extract_containers(g: np.ndarray) -> list[dict]:
    containers: list[dict] = []
    for coords in _components(g == FRAME):
        r0, c0, r1, c1 = _bbox(coords)
        height = int(r1 - r0 + 1)
        width = int(c1 - c0 + 1)
        if height < 3 or width < 3:
            continue

        border = np.zeros((height, width), dtype=bool)
        border[0, :] = True
        border[-1, :] = True
        border[:, 0] = True
        border[:, -1] = True
        rel = coords - np.asarray([r0, c0], dtype=int)
        if coords.shape[0] != int(border.sum()):
            continue
        if not np.all(border[rel[:, 0], rel[:, 1]]):
            continue

        interior = np.asarray(g[r0 + 1 : r1, c0 + 1 : c1], dtype=int)
        non_bg = {int(v) for v in np.unique(interior) if int(v) != BG}
        if len(non_bg) != 1 or FRAME in non_bg:
            continue
        color = next(iter(non_bg))
        pattern_mask = interior == color
        containers.append(
            {
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
                "pattern_mask": pattern_mask,
            }
        )

    assert containers, "expected at least one grey rectangular container"
    return containers


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each container's interior pattern according to its rotational symmetry order.

    The palette is a corner block of vertical stripes. Index 1 picks the first stripe,
    index 2 the second, and index 4 the fourth. Each grey rectangular frame keeps its
    border, while the monochrome pattern inside is recolored without changing shape.
    """
    g = np.asarray(grid, dtype=int)
    palette = _find_palette(g)["colors"]
    order_to_color = {1: int(palette[0]), 2: int(palette[1]), 4: int(palette[3])}

    out = g.copy()
    for container in _extract_containers(g):
        r0, c0, r1, c1 = container["bbox"]
        pattern_mask = container["pattern_mask"]
        order = _rotation_order(pattern_mask)
        interior = out[r0 + 1 : r1, c0 + 1 : c1]
        interior[pattern_mask] = int(order_to_color[order])
    return out


ORDER_PATTERNS = {
    1: (
        np.array([[1, 0], [1, 0], [1, 1]], dtype=bool),
        np.array([[1, 1, 1], [0, 1, 0]], dtype=bool),
        np.array([[1, 0, 0], [1, 1, 1]], dtype=bool),
    ),
    2: (
        np.array([[1, 1, 1]], dtype=bool),
        np.array([[1, 1, 0], [0, 1, 1]], dtype=bool),
        np.array([[1, 0], [0, 1]], dtype=bool),
    ),
    4: (
        np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool),
        np.ones((2, 2), dtype=bool),
        np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]], dtype=bool),
    ),
}

GEN_ATTEMPTS = 400
GRID_HEIGHT_RANGE = (18, 28)
GRID_WIDTH_RANGE = (18, 28)
PALETTE_WIDTHS = (4, 5, 6)
CONTAINER_COUNT_RANGE = (3, 6)
CONTAINER_GAP = 2


def _transform_mask(mask: np.ndarray) -> np.ndarray:
    out = np.asarray(mask, dtype=bool)
    if random.random() < 0.5:
        out = np.fliplr(out)
    out = np.rot90(out, random.randrange(4))
    return np.asarray(out, dtype=bool)


def _pattern_for_order(order: int) -> np.ndarray:
    for _ in range(50):
        mask = _transform_mask(random.choice(ORDER_PATTERNS[order]))
        if _rotation_order(mask) == order:
            return mask
    raise RuntimeError(f"failed to sample a pattern of order {order}")


def _separated(box: tuple[int, int, int, int], others: list[tuple[int, int, int, int]], gap: int) -> bool:
    r0, c0, r1, c1 = box
    for or0, oc0, or1, oc1 in others:
        if not (r1 + gap < or0 or or1 + gap < r0 or c1 + gap < oc0 or oc1 + gap < c0):
            return False
    return True


def _sample_box(
    grid_h: int,
    grid_w: int,
    box_h: int,
    box_w: int,
    occupied: list[tuple[int, int, int, int]],
) -> tuple[int, int] | None:
    positions = [
        (r, c)
        for r in range(1, grid_h - box_h)
        for c in range(1, grid_w - box_w)
    ]
    random.shuffle(positions)
    for r, c in positions:
        box = (int(r), int(c), int(r + box_h - 1), int(c + box_w - 1))
        if _separated(box, occupied, CONTAINER_GAP):
            return int(r), int(c)
    return None


def _paint_palette(
    grid: np.ndarray,
    corner: str,
    palette_colors: list[int],
    palette_height: int,
) -> tuple[int, int, int, int]:
    grid_h, grid_w = grid.shape
    palette_width = len(palette_colors)
    top = corner.startswith("top")
    left = corner.endswith("left")
    r0 = 0 if top else grid_h - palette_height
    c0 = 0 if left else grid_w - palette_width
    for idx, color in enumerate(palette_colors):
        grid[r0 : r0 + palette_height, c0 + idx] = int(color)
    return (int(r0), int(c0), int(r0 + palette_height - 1), int(c0 + palette_width - 1))


def _build_container_patch(order: int, color: int) -> tuple[np.ndarray, np.ndarray]:
    pattern = _pattern_for_order(order)
    ph, pw = pattern.shape
    inner_h = int(ph + random.randint(2, 5))
    inner_w = int(pw + random.randint(2, 5))
    patch = np.full((inner_h + 2, inner_w + 2), BG, dtype=int)
    patch[0, :] = FRAME
    patch[-1, :] = FRAME
    patch[:, 0] = FRAME
    patch[:, -1] = FRAME

    top_margin = random.randint(1, inner_h - ph - 1)
    left_margin = random.randint(1, inner_w - pw - 1)
    inner = patch[1:-1, 1:-1]
    window = inner[top_margin : top_margin + ph, left_margin : left_margin + pw]
    window[pattern] = int(color)
    return patch, np.asarray(pattern, dtype=bool)


def _place_patch(grid: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    h, w = patch.shape
    grid[r0 : r0 + h, c0 : c0 + w] = patch


def _render_expected(
    grid: np.ndarray,
    placements: list[dict],
    palette_colors: list[int],
) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    order_to_color = {1: int(palette_colors[0]), 2: int(palette_colors[1]), 4: int(palette_colors[3])}
    for item in placements:
        r0, c0, r1, c1 = item["bbox"]
        interior = out[r0 + 1 : r1, c0 + 1 : c1]
        interior[item["pattern_mask"]] = int(order_to_color[item["order"]])
    return out


def _sample_orders(count: int) -> list[int]:
    orders = [1, 2, 4]
    while len(orders) < count:
        orders.append(random.choice((1, 2, 4)))
    random.shuffle(orders)
    return orders


def _generate_attempt() -> np.ndarray | None:
    grid_h = random.randint(*GRID_HEIGHT_RANGE)
    grid_w = random.randint(*GRID_WIDTH_RANGE)
    grid = np.full((grid_h, grid_w), BG, dtype=int)

    palette_height = random.choice(PALETTE_HEIGHTS)
    palette_width = random.choice(PALETTE_WIDTHS)
    if palette_height >= grid_h or palette_width >= grid_w:
        return None

    palette_colors = random.sample(COLOR_POOL, palette_width)
    corner = random.choice(("top-left", "top-right", "bottom-left", "bottom-right"))
    occupied = [_paint_palette(grid, corner, palette_colors, palette_height)]

    count = random.randint(*CONTAINER_COUNT_RANGE)
    orders = _sample_orders(count)
    pattern_colors = [random.choice(tuple(c for c in COLOR_POOL if c not in palette_colors) or COLOR_POOL) for _ in orders]

    placements: list[dict] = []
    for order, color in zip(orders, pattern_colors, strict=True):
        patch, pattern = _build_container_patch(order, color)
        loc = _sample_box(grid_h, grid_w, patch.shape[0], patch.shape[1], occupied)
        if loc is None:
            return None
        r0, c0 = loc
        _place_patch(grid, patch, r0, c0)
        bbox = (int(r0), int(c0), int(r0 + patch.shape[0] - 1), int(c0 + patch.shape[1] - 1))
        placements.append({"bbox": bbox, "pattern_mask": patch[1:-1, 1:-1] == color, "order": int(order)})
        occupied.append(bbox)

    expected = _render_expected(grid, placements, palette_colors)
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    found_orders = {_rotation_order(item["pattern_mask"]) for item in placements}
    if found_orders != {1, 2, 4}:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid sample")
