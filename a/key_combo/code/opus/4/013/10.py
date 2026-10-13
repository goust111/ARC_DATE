from __future__ import annotations

import numpy as np
from collections import deque


BG = 0
SPRITE_COLORS = (1, 2, 3, 4)
GREY = 5
CONTROL = 6

MAX_GEN_ATTEMPTS = 400
_GEN_COUNTER = 0
_GEN_RECIPES = [
    (3, 4),
    (4, 3),
    (3, 8),
    (4, 0),
    (3, 1),
    (4, 7),
    (3, 6),
    (4, 2),
    (3, 5),
    (4, 4),
]

_SHAPE_BANK = [
    np.array([[1, 0], [1, 1]], dtype=bool),
    np.array([[1, 1, 1], [0, 1, 0]], dtype=bool),
    np.array([[1, 1, 0], [0, 1, 1]], dtype=bool),
    np.array([[1, 1], [1, 1], [1, 0]], dtype=bool),
    np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool),
    np.array([[0, 1, 1], [1, 1, 0], [0, 1, 0]], dtype=bool),
    np.array([[1, 1, 1], [1, 0, 0], [1, 1, 0]], dtype=bool),
    np.array([[1, 0, 1], [1, 1, 1], [1, 0, 0]], dtype=bool),
    np.array([[1, 1], [1, 0], [1, 1], [0, 1]], dtype=bool),
    np.array([[1, 1, 1], [0, 1, 1], [0, 0, 1]], dtype=bool),
]


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_bbox(grid: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _paste(dst: np.ndarray, patch: np.ndarray, r0: int, c0: int, *, transparent: int | None = None) -> None:
    h, w = patch.shape
    view = dst[r0 : r0 + h, c0 : c0 + w]
    if transparent is None:
        view[:] = patch
    else:
        mask = patch != int(transparent)
        view[mask] = patch[mask]


def _mask_key(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    arr = np.asarray(mask, dtype=np.uint8)
    return tuple(map(int, arr.shape)), arr.tobytes()


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _connected_components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _find_color_components(grid: np.ndarray, colors: set[int]) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            color = int(g[r, c])
            if color not in colors or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(g[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            arr = np.array(coords, dtype=int)
            r0, c0 = arr.min(axis=0)
            r1, c1 = arr.max(axis=0)
            comps.append(
                {
                    "color": color,
                    "coords": arr,
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    "area": int(arr.shape[0]),
                }
            )
    return comps


def _hole_mask_from_rect_patch(rect_patch: np.ndarray) -> np.ndarray:
    black = np.asarray(rect_patch == BG, dtype=bool)
    h, w = black.shape
    exterior = np.zeros((h, w), dtype=bool)
    q: deque[tuple[int, int]] = deque()
    for r in range(h):
        for c in (0, w - 1):
            if black[r, c] and not exterior[r, c]:
                exterior[r, c] = True
                q.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if black[r, c] and not exterior[r, c]:
                exterior[r, c] = True
                q.append((r, c))
    while q:
        rr, cc = q.popleft()
        for nr, nc in _neighbors4(rr, cc):
            if 0 <= nr < h and 0 <= nc < w and black[nr, nc] and not exterior[nr, nc]:
                exterior[nr, nc] = True
                q.append((nr, nc))
    return black & ~exterior


def _extract_rectangles(grid: np.ndarray) -> list[dict]:
    rects: list[dict] = []
    for comp in _find_color_components(grid, {GREY}):
        box = comp["bbox"]
        patch = _crop_bbox(grid, box)
        hole_mask = _hole_mask_from_rect_patch(patch)
        hole_comps = _connected_components(hole_mask)
        assert len(hole_comps) == 1
        hb = _bbox(hole_mask)
        assert hb is not None
        hr0, hc0, hr1, hc1 = hb
        rects.append(
            {
                "bbox": box,
                "hole_top_left": (box[0] + hr0, box[1] + hc0),
                "hole_mask": hole_mask[hr0 : hr1 + 1, hc0 : hc1 + 1].copy(),
            }
        )
    rects.sort(key=lambda item: (item["bbox"][0], item["bbox"][1]))
    return rects


def _extract_sprites(grid: np.ndarray) -> list[dict]:
    sprites: list[dict] = []
    for comp in _find_color_components(grid, set(SPRITE_COLORS)):
        box = comp["bbox"]
        patch = _crop_bbox(grid, box)
        color = int(comp["color"])
        sprites.append(
            {
                "color": color,
                "coords": comp["coords"],
                "bbox": box,
                "patch": patch,
                "mask": patch == color,
            }
        )
    return sprites


def _control_slot(control: tuple[int, int], group_bbox: tuple[int, int, int, int]) -> int:
    r, c = control
    r0, c0, r1, c1 = group_bbox
    row_zone = 0 if r < r0 else 2 if r > r1 else 1
    col_zone = 0 if c < c0 else 2 if c > c1 else 1
    return row_zone * 3 + col_zone


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reassign shape-matched colored sprites to grey holes by a control-pixel-driven cyclic offset.

    Input structure:
    - Black background with 3 or 4 separated grey rectangles, each containing one black 4-connected hole.
    - Outside the grey group are equally many monochrome colored sprites, with one unique sprite shape per hole shape.
    - A single magenta control pixel lies outside every rectangle and sprite.

    Rule:
    - Sort grey rectangles by bbox top-left in row-major order to get holes H_0..H_{N-1}.
    - Match each colored sprite to the hole with the same cropped binary mask, then reorder sprites by the index of the hole they naturally fit.
    - Read the control pixel's 3x3 relative-position slot with respect to the union bbox of all grey rectangles, convert it to p = row_zone * 3 + col_zone, and set the cyclic shift d = p mod N.
    - Remove the control pixel and erase the sprites from their original places, then paste sprite i into hole (i + d) mod N.

    Key insight:
    - Shape equality first builds a natural sprite-to-hole ordering, and the control pixel does not pick a destination directly; it only rotates that ordering before embedding.
    """
    g = np.asarray(grid, dtype=int)
    rects = _extract_rectangles(g)
    sprites = _extract_sprites(g)
    control_coords = np.argwhere(g == CONTROL)

    n = len(rects)
    assert n in (3, 4)
    assert len(sprites) == n
    assert control_coords.shape == (1, 2)

    hole_key_to_index: dict[tuple[tuple[int, int], bytes], int] = {}
    for i, rect in enumerate(rects):
        key = _mask_key(rect["hole_mask"])
        assert key not in hole_key_to_index
        hole_key_to_index[key] = i

    natural_sprites: list[dict | None] = [None] * n
    for sprite in sprites:
        key = _mask_key(sprite["mask"])
        assert key in hole_key_to_index
        idx = hole_key_to_index[key]
        assert natural_sprites[idx] is None
        natural_sprites[idx] = sprite
    assert all(sprite is not None for sprite in natural_sprites)

    group_bbox = (
        min(rect["bbox"][0] for rect in rects),
        min(rect["bbox"][1] for rect in rects),
        max(rect["bbox"][2] for rect in rects),
        max(rect["bbox"][3] for rect in rects),
    )
    control = tuple(map(int, control_coords[0]))
    shift = _control_slot(control, group_bbox) % n

    out = g.copy()
    out[control] = BG
    for sprite in natural_sprites:
        assert sprite is not None
        rr, cc = sprite["coords"].T
        out[rr, cc] = BG

    for i, sprite in enumerate(natural_sprites):
        assert sprite is not None
        target = rects[(i + shift) % n]
        tr, tc = target["hole_top_left"]
        _paste(out, sprite["patch"], tr, tc, transparent=BG)
    return out


def _apply_random_transform(mask: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    out = np.array(mask, dtype=bool, copy=True)
    out = np.rot90(out, int(rng.integers(0, 4)))
    if bool(rng.integers(0, 2)):
        out = np.fliplr(out)
    return out.copy()


def _choose_shapes(n: int, rng: np.random.Generator) -> list[np.ndarray]:
    order = list(rng.permutation(len(_SHAPE_BANK)))
    chosen: list[np.ndarray] = []
    seen: set[tuple[tuple[int, int], bytes]] = set()
    for idx in order:
        for _ in range(8):
            mask = _apply_random_transform(_SHAPE_BANK[idx], rng)
            key = _mask_key(mask)
            if key not in seen:
                seen.add(key)
                chosen.append(mask)
                break
        if len(chosen) == n:
            break
    if len(chosen) != n:
        raise RuntimeError("failed to choose unique shapes")
    return chosen


def _build_rect_patch(shape: np.ndarray) -> np.ndarray:
    h, w = shape.shape
    rect = np.full((h + 2, w + 2), GREY, dtype=int)
    inner = rect[1 : 1 + h, 1 : 1 + w]
    inner[shape] = BG
    return rect


def _build_sprite_patch(shape: np.ndarray, color: int) -> np.ndarray:
    sprite = np.full(shape.shape, BG, dtype=int)
    sprite[shape] = int(color)
    return sprite


def _mark_blocked(blocked: np.ndarray, box: tuple[int, int, int, int], padding: int = 0) -> None:
    h, w = blocked.shape
    r0, c0, r1, c1 = box
    rr0 = max(0, int(r0) - padding)
    cc0 = max(0, int(c0) - padding)
    rr1 = min(h - 1, int(r1) + padding)
    cc1 = min(w - 1, int(c1) + padding)
    blocked[rr0 : rr1 + 1, cc0 : cc1 + 1] = True


def _random_box_location(
    blocked: np.ndarray,
    patch_shape: tuple[int, int],
    rng: np.random.Generator,
    *,
    border: int = 1,
) -> tuple[int, int] | None:
    h, w = map(int, patch_shape)
    H, W = blocked.shape
    candidates: list[tuple[int, int]] = []
    for r0 in range(border, H - h - border + 1):
        for c0 in range(border, W - w - border + 1):
            if not blocked[r0 : r0 + h, c0 : c0 + w].any():
                candidates.append((r0, c0))
    if not candidates:
        return None
    return candidates[int(rng.integers(len(candidates)))]


def _control_candidates(
    shape: tuple[int, int],
    group_bbox: tuple[int, int, int, int],
    rect_boxes: list[tuple[int, int, int, int]],
    slot: int,
) -> list[tuple[int, int]]:
    H, W = map(int, shape)
    r0, c0, r1, c1 = group_bbox
    row_zone = slot // 3
    col_zone = slot % 3
    if row_zone == 0:
        rows = range(1, r0)
    elif row_zone == 1:
        rows = range(r0, r1 + 1)
    else:
        rows = range(r1 + 1, H - 1)
    if col_zone == 0:
        cols = range(1, c0)
    elif col_zone == 1:
        cols = range(c0, c1 + 1)
    else:
        cols = range(c1 + 1, W - 1)

    candidates: list[tuple[int, int]] = []
    for r in rows:
        for c in cols:
            inside_rect = any(br0 <= r <= br1 and bc0 <= c <= bc1 for br0, bc0, br1, bc1 in rect_boxes)
            if not inside_rect:
                candidates.append((int(r), int(c)))
    return candidates


def _recipe_for_call() -> tuple[int, int]:
    global _GEN_COUNTER
    recipe = _GEN_RECIPES[_GEN_COUNTER % len(_GEN_RECIPES)]
    _GEN_COUNTER += 1
    return recipe


def _make_example(n: int, slot: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    shapes = _choose_shapes(n, rng)
    colors = [int(v) for v in rng.choice(np.array(SPRITE_COLORS), size=n, replace=False)]
    rect_patches = [_build_rect_patch(shape) for shape in shapes]
    sprite_patches = [_build_sprite_patch(shape, color) for shape, color in zip(shapes, colors)]

    if n == 4:
        active_slots = (0, 1, 2, 3)
    else:
        patterns = ((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3))
        active_slots = patterns[int(rng.integers(len(patterns)))]

    item_order = list(rng.permutation(n))
    slot_to_item = dict(zip(active_slots, item_order))

    row_heights = []
    for row in range(2):
        row_items = [slot_to_item[s] for s in active_slots if s // 2 == row]
        row_heights.append(max(rect_patches[idx].shape[0] for idx in row_items))
    col_widths = []
    for col in range(2):
        col_items = [slot_to_item[s] for s in active_slots if s % 2 == col]
        col_widths.append(max(rect_patches[idx].shape[1] for idx in col_items))

    gap_r = int(rng.integers(2, 5))
    gap_c = int(rng.integers(2, 5))
    cluster_h = row_heights[0] + gap_r + row_heights[1]
    cluster_w = col_widths[0] + gap_c + col_widths[1]

    top_margin = int(rng.integers(2, 6))
    bottom_margin = int(rng.integers(2, 6))
    left_margin = int(rng.integers(2, 6))
    right_margin = int(rng.integers(2, 6))
    H = cluster_h + top_margin + bottom_margin
    W = cluster_w + left_margin + right_margin
    if max(H, W) > 30:
        raise RuntimeError("grid too large")

    grid = np.full((H, W), BG, dtype=int)
    slot_origins = {
        0: (top_margin, left_margin),
        1: (top_margin, left_margin + col_widths[0] + gap_c),
        2: (top_margin + row_heights[0] + gap_r, left_margin),
        3: (top_margin + row_heights[0] + gap_r, left_margin + col_widths[0] + gap_c),
    }

    hole_infos: list[dict] = []
    rect_boxes: list[tuple[int, int, int, int]] = []
    for slot_idx in active_slots:
        item_idx = slot_to_item[slot_idx]
        rect_patch = rect_patches[item_idx]
        rh, rw = rect_patch.shape
        slot_r, slot_c = slot_origins[slot_idx]
        row = slot_idx // 2
        col = slot_idx % 2
        dr = row_heights[row] - rh
        dc = col_widths[col] - rw
        r0 = slot_r + int(rng.integers(dr + 1))
        c0 = slot_c + int(rng.integers(dc + 1))
        _paste(grid, rect_patch, r0, c0, transparent=BG)
        rect_box = (r0, c0, r0 + rh - 1, c0 + rw - 1)
        rect_boxes.append(rect_box)
        hole_infos.append(
            {
                "hole_box": (r0 + 1, c0 + 1, r0 + rh - 2, c0 + rw - 2),
                "hole_top_left": (r0 + 1, c0 + 1),
                "item_idx": item_idx,
                "rect_box": rect_box,
            }
        )

    hole_infos.sort(key=lambda info: (info["rect_box"][0], info["rect_box"][1]))
    group_bbox = (
        min(box[0] for box in rect_boxes),
        min(box[1] for box in rect_boxes),
        max(box[2] for box in rect_boxes),
        max(box[3] for box in rect_boxes),
    )

    control_candidates = _control_candidates(grid.shape, group_bbox, rect_boxes, slot)
    if not control_candidates:
        raise RuntimeError("no control placement")
    control = control_candidates[int(rng.integers(len(control_candidates)))]
    grid[control] = CONTROL

    blocked = np.zeros_like(grid, dtype=bool)
    _mark_blocked(blocked, group_bbox, padding=1)
    _mark_blocked(blocked, (control[0], control[1], control[0], control[1]), padding=1)

    placed_sprites: list[dict] = []
    natural_order_items = [info["item_idx"] for info in hole_infos]
    sprite_order = list(rng.permutation(n))
    for order_idx in sprite_order:
        item_idx = natural_order_items[order_idx]
        sprite_patch = sprite_patches[item_idx]
        loc = _random_box_location(blocked, sprite_patch.shape, rng, border=1)
        if loc is None:
            raise RuntimeError("no sprite placement")
        r0, c0 = loc
        _paste(grid, sprite_patch, r0, c0, transparent=BG)
        h, w = sprite_patch.shape
        sprite_box = (r0, c0, r0 + h - 1, c0 + w - 1)
        _mark_blocked(blocked, sprite_box, padding=1)
        placed_sprites.append({"item_idx": item_idx, "patch": sprite_patch, "bbox": sprite_box})

    shift = slot % n
    expected = grid.copy()
    expected[control] = BG
    for sprite in placed_sprites:
        r0, c0, r1, c1 = sprite["bbox"]
        patch = sprite["patch"]
        view = expected[r0 : r1 + 1, c0 : c1 + 1]
        view[patch != BG] = BG

    natural_patches = [sprite_patches[item_idx] for item_idx in natural_order_items]
    for i, sprite_patch in enumerate(natural_patches):
        tr, tc = hole_infos[(i + shift) % n]["hole_top_left"]
        _paste(expected, sprite_patch, tr, tc, transparent=BG)

    assert len({hole_infos[i]["item_idx"] for i in range(n)}) == n
    assert _control_slot(control, group_bbox) == slot
    return grid, expected


def generate_input() -> np.ndarray:
    n, slot = _recipe_for_call()
    for _ in range(MAX_GEN_ATTEMPTS):
        rng = np.random.default_rng()
        try:
            grid, expected = _make_example(n, slot, rng)
        except RuntimeError:
            continue
        if not (grid.ndim == 2 and expected.ndim == 2 and max(grid.shape) <= 30 and max(expected.shape) <= 30):
            continue
        try:
            actual = main(grid)
        except Exception:
            continue
        if np.array_equal(actual, expected):
            return grid
    raise RuntimeError("failed to generate valid input for puzzle 10")
