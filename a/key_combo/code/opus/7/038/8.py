from __future__ import annotations

import random
from collections import deque

import numpy as np


_BG = 0
_COLORS = tuple(range(1, 10))
_MAX_ATTEMPTS = 400
_GRID_SIDE_RANGE = (10, 16)
_ORIENTATIONS = ("vertical", "horizontal", "square")
_BBOX_OPTIONS = {
    "vertical": [(3, 1), (4, 1), (5, 1), (3, 2), (4, 2)],
    "horizontal": [(1, 3), (1, 4), (1, 5), (2, 3), (2, 4)],
    "square": [(2, 2), (3, 3)],
}


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _component_coords(mask: np.ndarray) -> list[tuple[int, int]]:
    coords = np.argwhere(mask)
    assert coords.size > 0, "expected a foreground sprite"
    start = tuple(map(int, coords[0]))
    q: deque[tuple[int, int]] = deque([start])
    seen = {start}

    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1] and bool(mask[nr, nc]) and (nr, nc) not in seen:
                seen.add((nr, nc))
                q.append((nr, nc))

    assert len(seen) == int(mask.sum()), "expected exactly one 4-connected component"
    return sorted(seen)


def _bbox_from_coords(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _extract_sprite(grid: np.ndarray) -> tuple[int, np.ndarray, tuple[int, int, int, int], int]:
    fg_mask = grid != _BG
    coords = _component_coords(fg_mask)
    colors = np.unique(grid[fg_mask])
    assert len(colors) == 1, "expected a monochrome sprite"
    color = int(colors[0])
    bbox = _bbox_from_coords(coords)
    r0, c0, r1, c1 = bbox
    sprite = np.array(grid[r0 : r1 + 1, c0 : c1 + 1], dtype=int)
    return color, sprite, bbox, len(coords)


def _copy_origins(bbox: tuple[int, int, int, int], sprite_shape: tuple[int, int], count: int) -> list[tuple[int, int]]:
    r0, c0, _r1, _c1 = bbox
    h, w = sprite_shape
    origins = [(r0, c0)]

    if h > w:
        above = (count - 1) // 2
        below = count - 1 - above
        origins.extend((r0 - k * h, c0) for k in range(1, above + 1))
        origins.extend((r0 + k * h, c0) for k in range(1, below + 1))
        return origins

    if w > h:
        left = (count - 1) // 2
        right = count - 1 - left
        origins.extend((r0, c0 - k * w) for k in range(1, left + 1))
        origins.extend((r0, c0 + k * w) for k in range(1, right + 1))
        return origins

    per_arm = (count - 1) // 4
    remainder = (count - 1) % 4
    arm_counts = {"up": per_arm, "right": per_arm, "down": per_arm, "left": per_arm}
    for direction in ("up", "right", "down", "left"):
        if remainder <= 0:
            break
        arm_counts[direction] += 1
        remainder -= 1

    origins.extend((r0 - k * h, c0) for k in range(1, arm_counts["up"] + 1))
    origins.extend((r0, c0 + k * w) for k in range(1, arm_counts["right"] + 1))
    origins.extend((r0 + k * h, c0) for k in range(1, arm_counts["down"] + 1))
    origins.extend((r0, c0 - k * w) for k in range(1, arm_counts["left"] + 1))
    return origins


def _paste_sprite(out: np.ndarray, sprite: np.ndarray, top: int, left: int) -> None:
    for dr, dc in np.argwhere(sprite != _BG):
        rr = int(top + dr)
        cc = int(left + dc)
        if 0 <= rr < out.shape[0] and 0 <= cc < out.shape[1]:
            out[rr, cc] = int(sprite[dr, dc])


def _sprite_intersects(shape: tuple[int, int], top: int, left: int, sprite_shape: tuple[int, int]) -> bool:
    h, w = sprite_shape
    return top < shape[0] and left < shape[1] and top + h > 0 and left + w > 0


def main(grid: np.ndarray) -> np.ndarray:
    """
    Tile the single sprite in a direction and count encoded by the sprite's own geometry.

    Input structure:
    - A black background contains exactly one small monochrome 4-connected sprite.
    - The sprite stays at least one cell away from every grid edge.
    - Its bounding box may be taller than wide, wider than tall, or square.

    Rule:
    - Read the sprite bbox height H and width W to choose the tiling direction:
      tall -> vertical, wide -> horizontal, square -> a cross using all four directions.
    - Count the sprite's foreground pixels N; the total number of placed copies, including the original, is N.
    - Place copies flush bbox-to-bbox with no gaps. Vertical copies are distributed above and below the
      original, horizontal copies to the left and right, and square sprites extend in up/right/down/left
      order after equal per-arm allocation. Pixels outside the grid are clipped away.

    Key insight:
    - The sprite is both the payload and the instruction: its aspect ratio encodes direction, and its
      pixel count encodes how many times that same sprite must be repeated.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    color, sprite, bbox, count = _extract_sprite(g)
    assert color != _BG, "foreground color must be non-black"

    out = np.full_like(g, _BG)
    for top, left in _copy_origins(bbox, sprite.shape, count):
        _paste_sprite(out, sprite, top, left)
    return out


def _min_area_for_bbox(h: int, w: int) -> int:
    if h == 1 or w == 1:
        return h * w
    return h + w - 1


def _make_sprite_mask(h: int, w: int, area: int) -> np.ndarray:
    assert _min_area_for_bbox(h, w) <= area <= h * w
    mask = np.zeros((h, w), dtype=bool)

    if h == 1 or w == 1:
        mask[:, :] = True
        return mask

    corner = random.choice(("TL", "TR", "BL", "BR"))
    if corner == "TL":
        mask[0, :] = True
        mask[:, 0] = True
    elif corner == "TR":
        mask[0, :] = True
        mask[:, -1] = True
    elif corner == "BL":
        mask[-1, :] = True
        mask[:, 0] = True
    else:
        mask[-1, :] = True
        mask[:, -1] = True

    while int(mask.sum()) < area:
        frontier: list[tuple[int, int]] = []
        seen = set()
        for r, c in np.argwhere(mask):
            for nr, nc in _neighbors4(int(r), int(c)):
                if 0 <= nr < h and 0 <= nc < w and not bool(mask[nr, nc]) and (nr, nc) not in seen:
                    seen.add((nr, nc))
                    frontier.append((nr, nc))
        pick_r, pick_c = random.choice(frontier)
        mask[pick_r, pick_c] = True
    return mask


def _sample_sprite() -> tuple[np.ndarray, str, int]:
    orientation = random.choice(_ORIENTATIONS)
    h, w = random.choice(_BBOX_OPTIONS[orientation])
    min_area = _min_area_for_bbox(h, w)
    area = random.randint(min_area, min(7, h * w))
    color = random.choice(_COLORS)
    mask = _make_sprite_mask(h, w, area)
    sprite = np.zeros((h, w), dtype=int)
    sprite[mask] = color
    return sprite, orientation, area


def _placement_candidates(
    grid_shape: tuple[int, int], sprite: np.ndarray
) -> list[tuple[int, int, bool]]:
    H, W = grid_shape
    h, w = sprite.shape
    color = int(sprite[sprite != 0][0])
    probe = np.zeros((H, W), dtype=int)
    probe[1 : 1 + h, 1 : 1 + w] = sprite
    _ = color

    candidates: list[tuple[int, int, bool]] = []
    count = int(np.count_nonzero(sprite))
    bbox = (0, 0, h - 1, w - 1)

    for r0 in range(1, H - h):
        for c0 in range(1, W - w):
            shifted_bbox = (r0, c0, r0 + h - 1, c0 + w - 1)
            origins = _copy_origins(shifted_bbox, sprite.shape, count)
            has_clip = any(
                top < 0 or left < 0 or top + h > H or left + w > W
                for top, left in origins
            )
            extra_visible = any(
                _sprite_intersects((H, W), top, left, sprite.shape)
                for top, left in origins
                if (top, left) != (r0, c0)
            )
            if extra_visible:
                candidates.append((r0, c0, has_clip))
    return candidates


def _build_input() -> np.ndarray | None:
    sprite, orientation, area = _sample_sprite()
    _ = (orientation, area)
    H = random.randint(*_GRID_SIDE_RANGE)
    W = random.randint(*_GRID_SIDE_RANGE)
    h, w = sprite.shape
    if H <= h + 2 or W <= w + 2:
        return None

    candidates = _placement_candidates((H, W), sprite)
    if not candidates:
        return None

    want_clip = bool(random.getrandbits(1))
    filtered = [(r0, c0) for r0, c0, has_clip in candidates if has_clip == want_clip]
    if not filtered:
        filtered = [(r0, c0) for r0, c0, _has_clip in candidates]

    r0, c0 = random.choice(filtered)
    grid = np.zeros((H, W), dtype=int)
    _paste_sprite(grid, sprite, r0, c0)

    out = main(grid)
    if np.array_equal(out, grid):
        return None
    if np.count_nonzero(grid) < 3 or np.count_nonzero(grid) > 7:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(_MAX_ATTEMPTS):
        grid = _build_input()
        if grid is None:
            continue
        try:
            out = main(grid)
        except Exception:
            continue
        if out.ndim != 2 or max(grid.shape) > 30 or max(out.shape) > 30:
            continue
        return grid
    raise RuntimeError("failed to generate a valid input for puzzle 8")
