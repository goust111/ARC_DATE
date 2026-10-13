from __future__ import annotations

import random
from collections import deque

import numpy as np


DIRECTIONS = ("up", "right", "down", "left")
FRAME_COLOR = 5
BACKGROUND = 0
COLOR_POOL = tuple(color for color in range(1, 10) if color != FRAME_COLOR)

BODY_SPECS = (
    (3, 3),
    (5, 3),
    (7, 3),
    (5, 5),
    (7, 5),
    (7, 7),
)

OBJECT_COUNT_CHOICES = (4, 5)
TAIL_LENGTH_RANGE = (1, 3)
CONTAINER_SLACK_RANGE = (0, 3)
GRID_H_RANGE = (24, 30)
GRID_W_RANGE = (24, 30)
PLACEMENT_PAD = 1
PLACEMENT_TRIES = 250
MAX_GENERATION_ATTEMPTS = 300


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stack each object's rectangular body into the unique hollow container.

    Input structure:
    - Background is black.
    - One monochrome hollow rectangular frame acts as the target container.
    - Each other monochrome component is a filled rectangle with a 1-cell-wide
      straight tail attached at the midpoint of one side.

    Rule:
    - Decode the tail direction for every colored object.
    - Keep only the rectangular body, flipping it left/right for a left tail
      and up/down for an up tail; right/down keep the body unchanged.
    - Sort bodies by area from large to small.
    - Stack them from the container bottom upward, centered horizontally with
      no vertical gaps.
    - Clear everything outside the container to background.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _background_color(g)
    container = _find_container(g, bg)
    objects = _find_objects(g, bg, int(container["color"]))
    assert objects, "expected at least one object"

    r0, c0, r1, c1 = map(int, container["bbox"])
    inner_r0, inner_c0 = r0 + 1, c0 + 1
    inner_r1, inner_c1 = r1 - 1, c1 - 1
    inner_h = inner_r1 - inner_r0 + 1
    inner_w = inner_c1 - inner_c0 + 1
    assert inner_h > 0 and inner_w > 0, "container must have a non-empty interior"

    max_body_w = max(int(obj["body"].shape[1]) for obj in objects)
    total_body_h = sum(int(obj["body"].shape[0]) for obj in objects)
    assert inner_w == max_body_w, "container width must match the widest body"
    assert inner_h >= total_body_h, "container must be tall enough for the full stack"

    ordered = sorted(objects, key=lambda obj: (-int(obj["area"]), int(obj["bbox"][0]), int(obj["bbox"][1])))

    out = np.full(g.shape, bg, dtype=int)
    sub = out[r0 : r1 + 1, c0 : c1 + 1]
    sub[container["mask"]] = int(container["color"])

    cursor = inner_r1
    for obj in ordered:
        body = np.asarray(obj["body"], dtype=int)
        body_h, body_w = map(int, body.shape)
        top = cursor - body_h + 1
        assert top >= inner_r0, "stack overflowed above the container interior"
        left = inner_c0 + (inner_w - body_w) // 2
        out[top : cursor + 1, left : left + body_w] = body
        cursor = top - 1

    return out


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")


def _background_color(grid: np.ndarray) -> int:
    flat = np.asarray(grid, dtype=int).ravel()
    if flat.size == 0:
        return 0
    return int(np.bincount(flat, minlength=max(10, int(flat.max()) + 1)).argmax())


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(grid: np.ndarray, bg: int, exclude_colors: set[int] | None = None) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = map(int, g.shape)
    excluded = set() if exclude_colors is None else set(exclude_colors)
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []

    for r0 in range(h):
        for c0 in range(w):
            color = int(g[r0, c0])
            if color == bg or color in excluded or seen[r0, c0]:
                continue
            seen[r0, c0] = True
            q = deque([(r0, c0)])
            coords: list[tuple[int, int]] = []
            while q:
                r, c = q.popleft()
                coords.append((r, c))
                for nr, nc in _neighbors4(r, c):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(g[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))

            rows = [r for r, _ in coords]
            cols = [c for _, c in coords]
            rr0, rr1 = min(rows), max(rows)
            cc0, cc1 = min(cols), max(cols)
            mask = np.zeros((rr1 - rr0 + 1, cc1 - cc0 + 1), dtype=bool)
            for r, c in coords:
                mask[r - rr0, c - cc0] = True
            comps.append(
                {
                    "color": color,
                    "bbox": (rr0, cc0, rr1, cc1),
                    "mask": mask,
                }
            )

    return comps


def _is_rect_frame(mask: np.ndarray) -> bool:
    h, w = map(int, mask.shape)
    if h < 3 or w < 3:
        return False
    frame = np.zeros((h, w), dtype=bool)
    frame[0, :] = True
    frame[-1, :] = True
    frame[:, 0] = True
    frame[:, -1] = True
    return bool(np.array_equal(mask, frame))


def _find_container(grid: np.ndarray, bg: int) -> dict:
    candidates = [comp for comp in _components(grid, bg) if _is_rect_frame(comp["mask"])]
    assert len(candidates) == 1, "expected exactly one hollow rectangular container"
    container = candidates[0]
    r0, c0, r1, c1 = map(int, container["bbox"])
    interior = grid[r0 + 1 : r1, c0 + 1 : c1]
    assert np.all(interior == bg), "container interior must start empty"
    return container


def _midpoints(length: int) -> set[int]:
    if length % 2 == 1:
        return {length // 2}
    return {length // 2 - 1, length // 2}


def _decode_object(mask: np.ndarray) -> tuple[str, tuple[int, int]]:
    matches: list[tuple[str, tuple[int, int]]] = []
    for direction in DIRECTIONS:
        body_shape = _match_direction(mask, direction)
        if body_shape is not None:
            matches.append((direction, body_shape))
    assert len(matches) == 1, "object must have a unique tail direction"
    return matches[0]


def _match_direction(mask: np.ndarray, direction: str) -> tuple[int, int] | None:
    h, w = map(int, mask.shape)

    if direction == "up":
        tail_len = 0
        tail_col = None
        while tail_len < h and int(mask[tail_len].sum()) == 1:
            col = int(np.flatnonzero(mask[tail_len])[0])
            if tail_col is None:
                tail_col = col
            elif col != tail_col:
                return None
            tail_len += 1
        if tail_len == 0 or tail_len >= h:
            return None
        body = mask[tail_len:, :]
        if not body.all() or tail_col not in _midpoints(w):
            return None
        recon = np.zeros_like(mask, dtype=bool)
        recon[tail_len:, :] = True
        recon[:tail_len, tail_col] = True
        return (body.shape[0], body.shape[1]) if np.array_equal(mask, recon) else None

    if direction == "down":
        tail_len = 0
        tail_col = None
        while tail_len < h and int(mask[h - 1 - tail_len].sum()) == 1:
            col = int(np.flatnonzero(mask[h - 1 - tail_len])[0])
            if tail_col is None:
                tail_col = col
            elif col != tail_col:
                return None
            tail_len += 1
        if tail_len == 0 or tail_len >= h:
            return None
        body = mask[: h - tail_len, :]
        if not body.all() or tail_col not in _midpoints(w):
            return None
        recon = np.zeros_like(mask, dtype=bool)
        recon[: h - tail_len, :] = True
        recon[h - tail_len :, tail_col] = True
        return (body.shape[0], body.shape[1]) if np.array_equal(mask, recon) else None

    if direction == "left":
        tail_len = 0
        tail_row = None
        while tail_len < w and int(mask[:, tail_len].sum()) == 1:
            row = int(np.flatnonzero(mask[:, tail_len])[0])
            if tail_row is None:
                tail_row = row
            elif row != tail_row:
                return None
            tail_len += 1
        if tail_len == 0 or tail_len >= w:
            return None
        body = mask[:, tail_len:]
        if not body.all() or tail_row not in _midpoints(h):
            return None
        recon = np.zeros_like(mask, dtype=bool)
        recon[:, tail_len:] = True
        recon[tail_row, :tail_len] = True
        return (body.shape[0], body.shape[1]) if np.array_equal(mask, recon) else None

    tail_len = 0
    tail_row = None
    while tail_len < w and int(mask[:, w - 1 - tail_len].sum()) == 1:
        row = int(np.flatnonzero(mask[:, w - 1 - tail_len])[0])
        if tail_row is None:
            tail_row = row
        elif row != tail_row:
            return None
        tail_len += 1
    if tail_len == 0 or tail_len >= w:
        return None
    body = mask[:, : w - tail_len]
    if not body.all() or tail_row not in _midpoints(h):
        return None
    recon = np.zeros_like(mask, dtype=bool)
    recon[:, : w - tail_len] = True
    recon[tail_row, w - tail_len :] = True
    return (body.shape[0], body.shape[1]) if np.array_equal(mask, recon) else None


def _body_for_object(color: int, body_shape: tuple[int, int], direction: str) -> np.ndarray:
    body_h, body_w = map(int, body_shape)
    body = np.full((body_h, body_w), int(color), dtype=int)
    if direction == "up":
        return np.flipud(body)
    if direction == "left":
        return np.fliplr(body)
    return body


def _find_objects(grid: np.ndarray, bg: int, container_color: int) -> list[dict]:
    objects: list[dict] = []
    for comp in _components(grid, bg, exclude_colors={container_color}):
        direction, body_shape = _decode_object(comp["mask"])
        body = _body_for_object(int(comp["color"]), body_shape, direction)
        objects.append(
            {
                "color": int(comp["color"]),
                "bbox": tuple(map(int, comp["bbox"])),
                "direction": direction,
                "body": body,
                "area": int(body.shape[0] * body.shape[1]),
            }
        )
    return objects


def _object_mask(body_h: int, body_w: int, direction: str, tail_len: int) -> np.ndarray:
    if direction == "up":
        mask = np.zeros((body_h + tail_len, body_w), dtype=bool)
        mask[tail_len:, :] = True
        mask[:tail_len, body_w // 2] = True
        return mask
    if direction == "down":
        mask = np.zeros((body_h + tail_len, body_w), dtype=bool)
        mask[:body_h, :] = True
        mask[body_h:, body_w // 2] = True
        return mask
    if direction == "left":
        mask = np.zeros((body_h, body_w + tail_len), dtype=bool)
        mask[:, tail_len:] = True
        mask[body_h // 2, :tail_len] = True
        return mask

    mask = np.zeros((body_h, body_w + tail_len), dtype=bool)
    mask[:, :body_w] = True
    mask[body_h // 2, body_w:] = True
    return mask


def _draw_frame(grid: np.ndarray, top: int, left: int, height: int, width: int, color: int) -> None:
    grid[top, left : left + width] = int(color)
    grid[top + height - 1, left : left + width] = int(color)
    grid[top : top + height, left] = int(color)
    grid[top : top + height, left + width - 1] = int(color)


def _paint_mask(grid: np.ndarray, top: int, left: int, mask: np.ndarray, color: int) -> None:
    h, w = map(int, mask.shape)
    sub = grid[top : top + h, left : left + w]
    sub[mask] = int(color)


def _mark_blocked(blocked: np.ndarray, top: int, left: int, height: int, width: int, pad: int) -> None:
    h, w = map(int, blocked.shape)
    r0 = max(0, top - pad)
    c0 = max(0, left - pad)
    r1 = min(h, top + height + pad)
    c1 = min(w, left + width + pad)
    blocked[r0:r1, c0:c1] = True


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    n_objects = rng.choice(OBJECT_COUNT_CHOICES)
    specs = rng.sample(BODY_SPECS, n_objects)
    colors = rng.sample(COLOR_POOL, n_objects)

    directions = list(DIRECTIONS)
    if n_objects > len(DIRECTIONS):
        directions.extend(rng.choice(DIRECTIONS) for _ in range(n_objects - len(DIRECTIONS)))
    rng.shuffle(directions)

    objects = []
    for (body_w, body_h), color, direction in zip(specs, colors, directions):
        tail_len = rng.randint(*TAIL_LENGTH_RANGE)
        mask = _object_mask(body_h, body_w, direction, tail_len)
        objects.append(
            {
                "color": int(color),
                "direction": direction,
                "body_w": int(body_w),
                "body_h": int(body_h),
                "area": int(body_w * body_h),
                "mask": mask,
            }
        )

    if len({obj["area"] for obj in objects}) != len(objects):
        return None

    inner_w = max(obj["body_w"] for obj in objects)
    inner_h = sum(obj["body_h"] for obj in objects) + rng.randint(*CONTAINER_SLACK_RANGE)
    outer_w = inner_w + 2
    outer_h = inner_h + 2
    obj_max_w = max(int(obj["mask"].shape[1]) for obj in objects)

    min_h = max(GRID_H_RANGE[0], outer_h + 2)
    min_w = max(GRID_W_RANGE[0], outer_w + obj_max_w + 3)
    if min_h > GRID_H_RANGE[1] or min_w > GRID_W_RANGE[1]:
        return None

    height = rng.randint(min_h, GRID_H_RANGE[1])
    width = rng.randint(min_w, GRID_W_RANGE[1])
    max_top = height - outer_h - 1
    max_left = width - outer_w - 1
    if max_top < 1 or max_left < 1:
        return None

    top = rng.randint(1, max_top)
    band = max(0, max_left // 4)
    center = max_left // 2
    left_lo = max(1, center - band)
    left_hi = min(max_left, center + band)
    if left_lo > left_hi:
        left_lo, left_hi = 1, max_left
    left = rng.randint(left_lo, left_hi)

    grid = np.full((height, width), BACKGROUND, dtype=int)
    _draw_frame(grid, top, left, outer_h, outer_w, FRAME_COLOR)

    blocked = np.zeros((height, width), dtype=bool)
    _mark_blocked(blocked, top, left, outer_h, outer_w, PLACEMENT_PAD)

    placement_order = sorted(
        objects,
        key=lambda obj: -(int(obj["mask"].shape[0]) * int(obj["mask"].shape[1])),
    )
    for obj in placement_order:
        mask = np.asarray(obj["mask"], dtype=bool)
        obj_h, obj_w = map(int, mask.shape)
        placed = False
        for _ in range(PLACEMENT_TRIES):
            r0 = rng.randint(0, height - obj_h)
            c0 = rng.randint(0, width - obj_w)
            if blocked[
                max(0, r0 - PLACEMENT_PAD) : min(height, r0 + obj_h + PLACEMENT_PAD),
                max(0, c0 - PLACEMENT_PAD) : min(width, c0 + obj_w + PLACEMENT_PAD),
            ].any():
                continue
            _paint_mask(grid, r0, c0, mask, int(obj["color"]))
            _mark_blocked(blocked, r0, c0, obj_h, obj_w, PLACEMENT_PAD)
            placed = True
            break
        if not placed:
            return None

    try:
        out = main(grid)
    except AssertionError:
        return None

    if out.shape != grid.shape:
        return None
    if int(np.count_nonzero(grid == FRAME_COLOR)) != int(np.count_nonzero(out == FRAME_COLOR)):
        return None
    return grid
