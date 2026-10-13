from __future__ import annotations

import numpy as np


BG = 0
FRAME = 5
SPRITE_COLORS = (1, 2, 3)
RNG = np.random.default_rng()

SMALL_RECT_CHOICES = ((2, 2), (2, 3), (3, 2))
MID_RECT_CHOICES = ((2, 4), (4, 2), (3, 3), (2, 5), (5, 2), (3, 4), (4, 3))
LARGE_RECT_CHOICES = ((4, 4), (3, 5), (5, 3), (4, 5), (5, 4))

SHAPE_MASKS = (
    np.array([[1, 1]], dtype=int),
    np.array([[1], [1]], dtype=int),
    np.array([[1, 1, 1]], dtype=int),
    np.array([[1], [1], [1]], dtype=int),
    np.array([[1, 0], [1, 1]], dtype=int),
    np.array([[0, 1], [1, 1]], dtype=int),
    np.array([[1, 1], [1, 0]], dtype=int),
    np.array([[1, 1], [0, 1]], dtype=int),
    np.array([[1, 1], [1, 1]], dtype=int),
    np.array([[1, 1, 1], [0, 1, 0]], dtype=int),
    np.array([[0, 1, 0], [1, 1, 1]], dtype=int),
    np.array([[1, 0], [1, 1], [1, 0]], dtype=int),
    np.array([[0, 1], [1, 1], [0, 1]], dtype=int),
    np.array([[1, 1, 0], [0, 1, 1]], dtype=int),
    np.array([[0, 1, 1], [1, 1, 0]], dtype=int),
)


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_bbox(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _find_color_components(grid: np.ndarray, color: int) -> list[dict]:
    mask = grid == int(color)
    height, width = grid.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[dict] = []

    for row, col in np.argwhere(mask):
        r0 = int(row)
        c0 = int(col)
        if seen[r0, c0]:
            continue
        stack = [(r0, c0)]
        seen[r0, c0] = True
        coords: list[tuple[int, int]] = []
        while stack:
            r, c = stack.pop()
            coords.append((r, c))
            for rr, cc in _neighbors4(r, c):
                if 0 <= rr < height and 0 <= cc < width and mask[rr, cc] and not seen[rr, cc]:
                    seen[rr, cc] = True
                    stack.append((rr, cc))
        arr = np.array(coords, dtype=int)
        comps.append(
            {
                "color": int(color),
                "coords": arr,
                "bbox": _bbox_coords(arr),
                "area": int(arr.shape[0]),
            }
        )
    return comps


def _is_hollow_rect(grid: np.ndarray, comp: dict) -> bool:
    r0, c0, r1, c1 = comp["bbox"]
    height = r1 - r0 + 1
    width = c1 - c0 + 1
    if height < 3 or width < 3:
        return False
    if comp["area"] != 2 * height + 2 * width - 4:
        return False

    rel = comp["coords"] - np.array([r0, c0], dtype=int)
    mask = np.zeros((height, width), dtype=bool)
    mask[rel[:, 0], rel[:, 1]] = True

    frame = np.zeros((height, width), dtype=bool)
    frame[0, :] = True
    frame[-1, :] = True
    frame[:, 0] = True
    frame[:, -1] = True
    if not np.array_equal(mask, frame):
        return False

    interior = grid[r0 + 1 : r1, c0 + 1 : c1]
    return bool(np.all(interior == BG))


def _detect_hollow_rects(grid: np.ndarray) -> list[dict]:
    rects: list[dict] = []
    for comp in _find_color_components(grid, FRAME):
        if not _is_hollow_rect(grid, comp):
            continue
        r0, c0, r1, c1 = comp["bbox"]
        inner_h = r1 - r0 - 1
        inner_w = c1 - c0 - 1
        rects.append(
            {
                **comp,
                "id": len(rects),
                "inner_shape": (int(inner_h), int(inner_w)),
                "interior_area": int(inner_h * inner_w),
            }
        )
    return rects


def _detect_sprites(grid: np.ndarray) -> list[dict]:
    sprites: list[dict] = []
    for color in sorted(int(v) for v in np.unique(grid) if int(v) not in (BG, FRAME)):
        for comp in _find_color_components(grid, color):
            sprites.append(
                {
                    **comp,
                    "id": len(sprites),
                    "patch": _crop_bbox(grid, comp["bbox"]),
                }
            )
    return sprites


def _render_output_with_orders(
    grid: np.ndarray, rect_order: list[dict], sprite_order: list[dict], *, strict: bool = True
) -> np.ndarray | None:
    out = np.array(grid, dtype=int, copy=True)
    assert len(rect_order) == len(sprite_order)

    for sprite in sprite_order:
        rows = sprite["coords"][:, 0]
        cols = sprite["coords"][:, 1]
        out[rows, cols] = BG

    for sprite, rect in zip(sprite_order, rect_order):
        patch = sprite["patch"]
        patch_h, patch_w = patch.shape
        inner_h, inner_w = rect["inner_shape"]
        if patch_h > inner_h or patch_w > inner_w:
            if strict:
                raise AssertionError("sprite bbox does not fit target rectangle interior")
            return None
        r0, c0, _r1, _c1 = rect["bbox"]
        rr = r0 + 1
        cc = c0 + 1
        window = out[rr : rr + patch_h, cc : cc + patch_w]
        mask = patch != BG
        window[mask] = patch[mask]

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Sort hollow rectangles by interior area, sort sprites by top row, and stamp them rank-to-rank.

    Input structure:
    - Black background contains several grey 1-pixel-thick hollow rectangles whose interiors stay black.
    - Separate non-black, non-grey monochrome sprites lie outside the rectangles on the background.
    - Each sprite is a 4-connected component, and each rectangle is identified by its grey border.

    Rule:
    - Detect every grey hollow rectangle and compute its interior area.
    - Sort rectangles by interior area from small to large, and sort sprites by bbox top row from top to bottom.
    - Erase each sprite from its original location, then place its tight bbox pattern at the top-left of the matched rectangle interior.
    - Keep grey borders unchanged and keep zero-valued cells in the sprite bbox transparent.

    Key insight:
    - The correct assignment is determined only by crossing two independent rankings: rectangle interior area and sprite vertical order.
    """
    g = np.asarray(grid, dtype=int)
    rects = _detect_hollow_rects(g)
    sprites = _detect_sprites(g)

    assert rects and len(rects) == len(sprites)
    assert len({rect["interior_area"] for rect in rects}) == len(rects)
    assert len({sprite["bbox"][0] for sprite in sprites}) == len(sprites)

    rect_order = sorted(rects, key=lambda rect: (rect["interior_area"], rect["bbox"][0], rect["bbox"][1]))
    sprite_order = sorted(
        sprites,
        key=lambda sprite: (sprite["bbox"][0], sprite["bbox"][1], sprite["bbox"][2], sprite["color"]),
    )
    return _render_output_with_orders(g, rect_order, sprite_order)


def _shape_patch(mask: np.ndarray, color: int) -> np.ndarray:
    return np.where(mask != 0, int(color), BG).astype(int)


def _sample_rect_dims() -> list[tuple[int, int]]:
    return [
        tuple(int(v) for v in SMALL_RECT_CHOICES[int(RNG.integers(len(SMALL_RECT_CHOICES)))]),
        tuple(int(v) for v in MID_RECT_CHOICES[int(RNG.integers(len(MID_RECT_CHOICES)))]),
        tuple(int(v) for v in LARGE_RECT_CHOICES[int(RNG.integers(len(LARGE_RECT_CHOICES)))]),
    ]


def _sample_shape_for_dims(inner_h: int, inner_w: int) -> np.ndarray | None:
    choices = [mask for mask in SHAPE_MASKS if mask.shape[0] <= inner_h and mask.shape[1] <= inner_w]
    if not choices:
        return None
    return choices[int(RNG.integers(len(choices)))]


def _free_positions(
    occupied: np.ndarray,
    box_h: int,
    box_w: int,
    *,
    pad: int,
    row_min: int,
    row_max: int,
) -> list[tuple[int, int]]:
    height, width = occupied.shape
    low = max(1, int(row_min))
    high = min(int(row_max), height - box_h - 1)
    positions: list[tuple[int, int]] = []
    if low > high:
        return positions

    for r in range(low, high + 1):
        for c in range(1, width - box_w):
            rr0 = max(0, r - pad)
            cc0 = max(0, c - pad)
            rr1 = min(height, r + box_h + pad)
            cc1 = min(width, c + box_w + pad)
            if not occupied[rr0:rr1, cc0:cc1].any():
                positions.append((r, c))
    return positions


def _occupy(occupied: np.ndarray, top: int, left: int, box_h: int, box_w: int, *, pad: int) -> None:
    height, width = occupied.shape
    rr0 = max(0, top - pad)
    cc0 = max(0, left - pad)
    rr1 = min(height, top + box_h + pad)
    cc1 = min(width, left + box_w + pad)
    occupied[rr0:rr1, cc0:cc1] = True


def _draw_rect(grid: np.ndarray, top: int, left: int, inner_h: int, inner_w: int) -> None:
    outer_h = inner_h + 2
    outer_w = inner_w + 2
    grid[top, left : left + outer_w] = FRAME
    grid[top + outer_h - 1, left : left + outer_w] = FRAME
    grid[top : top + outer_h, left] = FRAME
    grid[top : top + outer_h, left + outer_w - 1] = FRAME


def _build_input(
    shape: tuple[int, int], rects: list[dict], sprites: list[dict]
) -> np.ndarray:
    grid = np.full(shape, BG, dtype=int)
    for rect in rects:
        _draw_rect(grid, rect["top"], rect["left"], rect["inner_shape"][0], rect["inner_shape"][1])
    for sprite in sprites:
        patch = sprite["patch"]
        h, w = patch.shape
        window = grid[sprite["top"] : sprite["top"] + h, sprite["left"] : sprite["left"] + w]
        mask = patch != BG
        window[mask] = patch[mask]
    return grid


def _expected_from_layout(shape: tuple[int, int], rects: list[dict], sprites: list[dict]) -> np.ndarray:
    inp = _build_input(shape, rects, sprites)
    out = np.array(inp, copy=True)
    sprite_order = sorted(sprites, key=lambda sprite: (sprite["top"], sprite["left"], sprite["color"]))
    rect_order = sorted(rects, key=lambda rect: (rect["interior_area"], rect["top"], rect["left"]))

    for sprite in sprites:
        patch = sprite["patch"]
        h, w = patch.shape
        mask = patch != BG
        window = out[sprite["top"] : sprite["top"] + h, sprite["left"] : sprite["left"] + w]
        window[mask] = BG

    for sprite, rect in zip(sprite_order, rect_order):
        patch = sprite["patch"]
        h, w = patch.shape
        rr = rect["top"] + 1
        cc = rect["left"] + 1
        window = out[rr : rr + h, cc : cc + w]
        mask = patch != BG
        window[mask] = patch[mask]

    return out


def _order_ids(items: list[dict]) -> tuple[int, ...]:
    return tuple(int(item["id"]) for item in items)


def _semantic_scene_ok(grid: np.ndarray) -> bool:
    rects = _detect_hollow_rects(grid)
    sprites = _detect_sprites(grid)
    if len(rects) != 3 or len(sprites) != 3:
        return False
    if len({rect["interior_area"] for rect in rects}) != 3:
        return False
    if len({sprite["bbox"][0] for sprite in sprites}) != 3:
        return False

    rect_area = sorted(rects, key=lambda rect: (rect["interior_area"], rect["bbox"][0], rect["bbox"][1]))
    rect_row = sorted(rects, key=lambda rect: (rect["bbox"][0], rect["bbox"][1]))
    rect_col = sorted(rects, key=lambda rect: (rect["bbox"][1], rect["bbox"][0]))
    rect_desc = sorted(rects, key=lambda rect: (-rect["interior_area"], rect["bbox"][0], rect["bbox"][1]))

    sprite_top = sorted(sprites, key=lambda sprite: (sprite["bbox"][0], sprite["bbox"][1], sprite["color"]))
    sprite_left = sorted(sprites, key=lambda sprite: (sprite["bbox"][1], sprite["bbox"][0], sprite["color"]))
    sprite_color = sorted(sprites, key=lambda sprite: (sprite["color"], sprite["bbox"][0], sprite["bbox"][1]))
    sprite_bottom = sorted(sprites, key=lambda sprite: (sprite["bbox"][2], sprite["bbox"][1], sprite["color"]))

    if _order_ids(rect_area) in {_order_ids(rect_row), _order_ids(rect_col), _order_ids(rect_desc)}:
        return False
    if _order_ids(sprite_top) in {
        _order_ids(sprite_left),
        _order_ids(sprite_color),
        _order_ids(sprite_bottom),
    }:
        return False

    correct = _render_output_with_orders(grid, rect_area, sprite_top)
    for alt_rect in (rect_row, rect_col, rect_desc):
        if _order_ids(alt_rect) != _order_ids(rect_area):
            alt = _render_output_with_orders(grid, alt_rect, sprite_top, strict=False)
            if alt is not None and np.array_equal(correct, alt):
                return False

    for alt_sprite in (sprite_left, sprite_color, sprite_bottom):
        if _order_ids(alt_sprite) != _order_ids(sprite_top):
            alt = _render_output_with_orders(grid, rect_area, alt_sprite, strict=False)
            if alt is not None and np.array_equal(correct, alt):
                return False

    return True


def _generate_attempt() -> np.ndarray | None:
    height = int(RNG.integers(20, 29))
    width = int(RNG.integers(20, 29))
    rect_dims = _sample_rect_dims()
    colors = [int(c) for c in RNG.permutation(np.array(SPRITE_COLORS, dtype=int))]

    sprite_specs: list[dict] = []
    for rank, ((inner_h, inner_w), color) in enumerate(zip(rect_dims, colors)):
        mask = _sample_shape_for_dims(inner_h, inner_w)
        if mask is None:
            return None
        sprite_specs.append(
            {
                "rank": rank,
                "color": int(color),
                "patch": _shape_patch(mask, int(color)),
            }
        )

    occupied = np.zeros((height, width), dtype=bool)
    rects: list[dict] = []
    shuffled_rects = list(enumerate(rect_dims))
    RNG.shuffle(shuffled_rects)
    for rank, (inner_h, inner_w) in shuffled_rects:
        outer_h = inner_h + 2
        outer_w = inner_w + 2
        positions = _free_positions(
            occupied,
            outer_h,
            outer_w,
            pad=1,
            row_min=1,
            row_max=height - outer_h - 1,
        )
        if not positions:
            return None
        top, left = positions[int(RNG.integers(len(positions)))]
        _occupy(occupied, top, left, outer_h, outer_w, pad=1)
        rects.append(
            {
                "id": int(rank),
                "top": int(top),
                "left": int(left),
                "bbox": (int(top), int(left), int(top + outer_h - 1), int(left + outer_w - 1)),
                "inner_shape": (int(inner_h), int(inner_w)),
                "interior_area": int(inner_h * inner_w),
            }
        )

    sprites: list[dict] = []
    prev_top = 0
    for idx, spec in enumerate(sprite_specs):
        patch = spec["patch"]
        box_h, box_w = patch.shape
        remaining = len(sprite_specs) - idx - 1
        row_min = prev_top + (1 if idx else 0)
        row_max = height - box_h - 2 - remaining
        positions = _free_positions(
            occupied,
            box_h,
            box_w,
            pad=1,
            row_min=row_min,
            row_max=row_max,
        )
        if not positions:
            return None
        top, left = positions[int(RNG.integers(len(positions)))]
        _occupy(occupied, top, left, box_h, box_w, pad=1)
        prev_top = int(top)
        sprites.append(
            {
                "id": idx,
                "top": int(top),
                "left": int(left),
                "bbox": (int(top), int(left), int(top + box_h - 1), int(left + box_w - 1)),
                "color": int(spec["color"]),
                "patch": patch.copy(),
            }
        )

    inp = _build_input((height, width), rects, sprites)
    expected = _expected_from_layout((height, width), rects, sprites)
    if not np.array_equal(main(inp), expected):
        return None
    if not _semantic_scene_ok(inp):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(500):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a non-degenerate hollow-rectangle stamping scene")
