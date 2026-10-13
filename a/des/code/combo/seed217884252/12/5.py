from __future__ import annotations

import random

import numpy as np


BG = 0
DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _monochrome_components(grid: np.ndarray) -> list[dict[str, object]]:
    h, w = grid.shape
    visited = np.zeros((h, w), dtype=bool)
    components: list[dict[str, object]] = []

    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == BG or visited[r, c]:
                continue

            stack = [(r, c)]
            visited[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in DIR4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc] and int(grid[nr, nc]) == color:
                        visited[nr, nc] = True
                        stack.append((nr, nc))

            rows = [rr for rr, _ in coords]
            cols = [cc for _, cc in coords]
            r0, r1 = min(rows), max(rows)
            c0, c1 = min(cols), max(cols)
            mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
            for rr, cc in coords:
                mask[rr - r0, cc - c0] = True
            components.append(
                {
                    "color": color,
                    "bbox": (r0, c0, r1, c1),
                    "mask": mask,
                }
            )

    return components


def _rect_key(rect: tuple[int, int, int, int]) -> tuple[int, int, int, int, int]:
    top, left, height, width = rect
    return (height * width, height, width, -top, -left)


def _largest_inside_rectangle(
    mask: np.ndarray,
) -> tuple[int, tuple[int, int, int, int], set[tuple[int, int]]]:
    h, w = mask.shape
    heights = np.zeros(w, dtype=int)
    best_area = 0
    best_rect = (0, 0, 0, 0)
    best_dims: set[tuple[int, int]] = set()

    for r in range(h):
        for c in range(w):
            heights[c] = heights[c] + 1 if mask[r, c] else 0

        stack: list[tuple[int, int]] = []
        for c in range(w + 1):
            cur = int(heights[c]) if c < w else 0
            start = c
            while stack and stack[-1][1] > cur:
                left, height = stack.pop()
                start = left
                if height == 0:
                    continue
                width = c - left
                area = height * width
                rect = (r - height + 1, left, height, width)
                dims = (height, width)
                if area > best_area:
                    best_area = area
                    best_rect = rect
                    best_dims = {dims}
                elif area == best_area:
                    best_dims.add(dims)
                    if _rect_key(rect) > _rect_key(best_rect):
                        best_rect = rect
            if cur > 0 and (not stack or stack[-1][1] < cur):
                stack.append((start, cur))

    return best_area, best_rect, best_dims


def main(grid: np.ndarray) -> np.ndarray:
    """
    Sort foreground objects by the area of their largest inscribed rectangle,
    then redraw those rectangles from left to right with a one-column gap.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    objects = []
    for component in _monochrome_components(g):
        area, rect, _dims = _largest_inside_rectangle(component["mask"])
        _top, _left, height, width = rect
        assert area > 0
        objects.append(
            {
                "color": int(component["color"]),
                "area": int(area),
                "height": int(height),
                "width": int(width),
            }
        )

    assert objects
    objects.sort(key=lambda obj: obj["area"])

    out_h = max(int(obj["height"]) for obj in objects)
    out_w = sum(int(obj["width"]) for obj in objects) + len(objects) - 1
    out = np.full((out_h, out_w), BG, dtype=int)

    col = 0
    for obj in objects:
        rect_h = int(obj["height"])
        rect_w = int(obj["width"])
        out[out_h - rect_h : out_h, col : col + rect_w] = int(obj["color"])
        col += rect_w + 1

    return out


# --------------- generator ---------------

RECT_OPTIONS: tuple[tuple[int, int], ...] = (
    (2, 4),
    (3, 3),
    (2, 5),
    (3, 4),
    (3, 5),
    (4, 4),
    (4, 5),
    (5, 5),
)
COLOR_POOL = tuple(range(1, 10))

OBJECT_COUNT_RANGE = (3, 5)
BRANCH_MARGIN = 2
BRANCH_LENGTH_RANGE = (1, 2)
TIP_PROBABILITY = 0.65
OBJECT_SHAPE_ATTEMPTS = 200

PLACEMENT_BOARD_SIZE = 30
PLACEMENT_MARGIN = 3
PLACEMENT_ATTEMPTS = 250
CROP_BORDER_RANGE = (1, 3)
GENERATION_ATTEMPTS = 400


def _draw_branch(
    mask: np.ndarray,
    side: str,
    base_top: int,
    base_left: int,
    rect_h: int,
    rect_w: int,
) -> None:
    length = random.randint(*BRANCH_LENGTH_RANGE)

    if side == "top":
        anchor = random.randint(base_left, base_left + rect_w - 1)
        for step in range(1, length + 1):
            mask[base_top - step, anchor] = True
        tip_r, tip_c = base_top - length, anchor
        offsets = [dc for dc in (-1, 1) if 0 <= tip_c + dc < mask.shape[1]]
        if offsets and random.random() < TIP_PROBABILITY:
            mask[tip_r, tip_c + random.choice(offsets)] = True
        return

    if side == "bottom":
        anchor = random.randint(base_left, base_left + rect_w - 1)
        for step in range(1, length + 1):
            mask[base_top + rect_h - 1 + step, anchor] = True
        tip_r, tip_c = base_top + rect_h - 1 + length, anchor
        offsets = [dc for dc in (-1, 1) if 0 <= tip_c + dc < mask.shape[1]]
        if offsets and random.random() < TIP_PROBABILITY:
            mask[tip_r, tip_c + random.choice(offsets)] = True
        return

    if side == "left":
        anchor = random.randint(base_top, base_top + rect_h - 1)
        for step in range(1, length + 1):
            mask[anchor, base_left - step] = True
        tip_r, tip_c = anchor, base_left - length
        offsets = [dr for dr in (-1, 1) if 0 <= tip_r + dr < mask.shape[0]]
        if offsets and random.random() < TIP_PROBABILITY:
            mask[tip_r + random.choice(offsets), tip_c] = True
        return

    if side == "right":
        anchor = random.randint(base_top, base_top + rect_h - 1)
        for step in range(1, length + 1):
            mask[anchor, base_left + rect_w - 1 + step] = True
        tip_r, tip_c = anchor, base_left + rect_w - 1 + length
        offsets = [dr for dr in (-1, 1) if 0 <= tip_r + dr < mask.shape[0]]
        if offsets and random.random() < TIP_PROBABILITY:
            mask[tip_r + random.choice(offsets), tip_c] = True
        return

    raise ValueError(f"unknown side: {side}")


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return mask[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]


def _make_irregular_object(rect_h: int, rect_w: int, color: int) -> np.ndarray:
    target_area = rect_h * rect_w

    for _ in range(OBJECT_SHAPE_ATTEMPTS):
        mask = np.zeros((rect_h + 2 * BRANCH_MARGIN, rect_w + 2 * BRANCH_MARGIN), dtype=bool)
        base_top = BRANCH_MARGIN
        base_left = BRANCH_MARGIN
        mask[base_top : base_top + rect_h, base_left : base_left + rect_w] = True

        sides = ["top", "bottom", "left", "right"]
        random.shuffle(sides)
        for side in sides[: random.randint(1, len(sides))]:
            _draw_branch(mask, side, base_top, base_left, rect_h, rect_w)

        cropped = _crop_mask(mask)
        best_area, _best_rect, best_dims = _largest_inside_rectangle(cropped)
        if best_area != target_area:
            continue
        if best_dims != {(rect_h, rect_w)}:
            continue
        if int(cropped.sum()) == target_area:
            continue

        sprite = np.zeros(cropped.shape, dtype=int)
        sprite[cropped] = color
        return sprite

    raise RuntimeError("failed to build a valid irregular object")


def _dilate8(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    out = np.zeros_like(mask)
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            src_r0 = max(0, -dr)
            src_r1 = min(h, h - dr)
            src_c0 = max(0, -dc)
            src_c1 = min(w, w - dc)
            dst_r0 = max(0, dr)
            dst_r1 = min(h, h + dr)
            dst_c0 = max(0, dc)
            dst_c1 = min(w, w + dc)
            out[dst_r0:dst_r1, dst_c0:dst_c1] |= mask[src_r0:src_r1, src_c0:src_c1]
    return out


def _place_sprites(sprites: list[np.ndarray]) -> np.ndarray | None:
    board = np.zeros((PLACEMENT_BOARD_SIZE, PLACEMENT_BOARD_SIZE), dtype=int)
    occupied = np.zeros_like(board, dtype=bool)
    blocked = np.zeros_like(board, dtype=bool)

    order = list(range(len(sprites)))
    order.sort(key=lambda idx: sprites[idx].shape[0] * sprites[idx].shape[1], reverse=True)

    for idx in order:
        sprite = sprites[idx]
        sprite_mask = sprite != BG
        sh, sw = sprite.shape
        placed = False
        for _ in range(PLACEMENT_ATTEMPTS):
            top = random.randint(PLACEMENT_MARGIN, PLACEMENT_BOARD_SIZE - sh - PLACEMENT_MARGIN)
            left = random.randint(PLACEMENT_MARGIN, PLACEMENT_BOARD_SIZE - sw - PLACEMENT_MARGIN)
            if blocked[top : top + sh, left : left + sw][sprite_mask].any():
                continue

            window = board[top : top + sh, left : left + sw]
            window[sprite_mask] = sprite[sprite_mask]
            occupied[top : top + sh, left : left + sw] |= sprite_mask
            blocked = _dilate8(occupied)
            placed = True
            break

        if not placed:
            return None

    coords = np.argwhere(occupied)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    pad_top = random.randint(CROP_BORDER_RANGE[0], min(CROP_BORDER_RANGE[1], int(r0)))
    pad_left = random.randint(CROP_BORDER_RANGE[0], min(CROP_BORDER_RANGE[1], int(c0)))
    pad_bottom = random.randint(
        CROP_BORDER_RANGE[0],
        min(CROP_BORDER_RANGE[1], PLACEMENT_BOARD_SIZE - 1 - int(r1)),
    )
    pad_right = random.randint(
        CROP_BORDER_RANGE[0],
        min(CROP_BORDER_RANGE[1], PLACEMENT_BOARD_SIZE - 1 - int(c1)),
    )
    return board[
        int(r0) - pad_top : int(r1) + pad_bottom + 1,
        int(c0) - pad_left : int(c1) + pad_right + 1,
    ]


def _render_expected(rect_specs: list[tuple[int, int, int, int]]) -> np.ndarray:
    ordered = sorted(rect_specs, key=lambda spec: spec[0] * spec[1])
    out_h = max(h for h, _w, _area, _color in ordered)
    out_w = sum(w for _h, w, _area, _color in ordered) + len(ordered) - 1
    out = np.zeros((out_h, out_w), dtype=int)

    col = 0
    for rect_h, rect_w, _area, color in ordered:
        out[out_h - rect_h : out_h, col : col + rect_w] = color
        col += rect_w + 1
    return out


def _sample_rectangles() -> list[tuple[int, int, int, int]]:
    count = random.randint(*OBJECT_COUNT_RANGE)
    dims = random.sample(RECT_OPTIONS, count)
    colors = random.sample(COLOR_POOL, count)

    specs = []
    for (rect_h, rect_w), color in zip(dims, colors):
        if rect_h != rect_w and random.random() < 0.5:
            rect_h, rect_w = rect_w, rect_h
        specs.append((rect_h, rect_w, rect_h * rect_w, color))
    return specs


def _gen_attempt() -> np.ndarray | None:
    rect_specs = _sample_rectangles()
    sprites = []
    for rect_h, rect_w, _area, color in rect_specs:
        sprites.append(_make_irregular_object(rect_h, rect_w, color))

    inp = _place_sprites(sprites)
    if inp is None:
        return None

    expected = _render_expected(rect_specs)
    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
